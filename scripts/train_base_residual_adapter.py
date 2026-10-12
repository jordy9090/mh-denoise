"""Fit a fresh Base LoRA using verified TRAIN-only targets and holdout."""
import json
import math
import random
import shutil
import time

from residual_train_common import *


def train():
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from fullpaper_backbone_utils import load_fullpaper_tokenizer
    from train_gemma_risk_tune_from_sft import load_base_model
    from selective_risk_refinement_utils import build_risk_tune_prompt, configure_risk_contract
    out = ROOT / 'training'
    out.mkdir(exist_ok=True)
    alias_failures = [r for r in rows(ROOT / 'results/train_verify.jsonl')
                      if not r.get('ok') and r.get('error') == 'Missing verifier field: evidence']
    if alias_failures:
        summary = read(ROOT / 'data/curation_summary.json')
        if 'schema_recovered_verifier_results' not in summary:
            raise RuntimeError('Saved verifier evidence_phrase schema requires offline recovery before training; rerun pipeline without new API calls')
    if (out / 'manifest.json').exists():
        manifest = read(out / 'manifest.json')
        assert file_sha(Path(manifest['selected_adapter']) / 'adapter_model.safetensors') == manifest['selected_adapter_sha256']
        print('Reusing completed training', flush=True)
        return
    gpu_idle()
    configure_risk_contract('fullpaper_v1')
    args = read(ROOT / 'run_contract.json')['settings']['training']
    data_identity = {'train': file_sha(ROOT / 'data/train.jsonl'), 'holdout': file_sha(ROOT / 'data/holdout.jsonl'),
                     'settings_sha256': sha(args)}
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = load_fullpaper_tokenizer(str(TOKENIZER))
    tokenizer.padding_side = 'right'
    prepared, rejected, lengths = {}, [], []
    for split in ('train', 'holdout'):
        prepared[split] = []
        for r in rows(ROOT / f'data/{split}.jsonl'):
            prompt = build_risk_tune_prompt(tokenizer, r)
            # Match the generation helper's actual input IDs, including BOS policy.
            prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)['input_ids']
            target_ids = tokenizer(r['safe_response'], add_special_tokens=False, truncation=False)['input_ids'] + [tokenizer.eos_token_id]
            if len(prompt_ids) > args['max_source_len'] or len(target_ids) > args['max_target_len']:
                rejected.append({'canonical_id': r['canonical_id'], 'split': split, 'reason': 'would_truncate',
                                 'source_tokens': len(prompt_ids), 'target_tokens': len(target_ids)})
                continue
            prepared[split].append({'input_ids': prompt_ids + target_ids,
                                    'labels': [-100] * len(prompt_ids) + target_ids})
            lengths.append({'canonical_id': r['canonical_id'], 'split': split, 'decision': r['decision'],
                            'rendered_prompt': prompt, 'rendered_prompt_sha256': sha(prompt),
                            'input_token_ids_sha256': sha(prompt_ids), 'target_token_ids_sha256': sha(target_ids),
                            'prompt_tokens': len(prompt_ids), 'supervised_tokens': len(target_ids),
                            'prompt_loss_tokens': 0, 'response_sha256': sha(r['safe_response'])})
    write_rows(out / 'tokenization_audit.jsonl', lengths)
    write_rows(out / 'excluded_length.jsonl', rejected)
    fit_edit_count = sum(r['split'] == 'train' and r['decision'] == 'edit' for r in lengths)
    if fit_edit_count < 8 or len(prepared['holdout']) < 8:
        raise RuntimeError('Insufficient non-truncated verified data')

    def collate(batch):
        n = max(len(r['input_ids']) for r in batch)
        return {'input_ids': torch.tensor([r['input_ids'] + [tokenizer.pad_token_id] * (n - len(r['input_ids'])) for r in batch]),
                'attention_mask': torch.tensor([[1] * len(r['input_ids']) + [0] * (n - len(r['input_ids'])) for r in batch]),
                'labels': torch.tensor([r['labels'] + [-100] * (n - len(r['labels'])) for r in batch])}

    base = load_base_model(str(BASE), use_4bit=True)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    config = read(TOKENIZER / 'adapter_config.json')
    lora = LoraConfig(r=config['r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
                     target_modules=config['target_modules'], bias='none', task_type='CAUSAL_LM', init_lora_weights=True)
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    assert len(model.peft_config) == 1
    assert all(torch.count_nonzero(p).item() == 0 for n, p in model.named_parameters() if 'lora_B' in n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    device = next(model.parameters()).device
    steps_per_epoch = math.ceil(len(prepared['train']) / args['gradient_accumulation_steps'])
    total_steps = steps_per_epoch * args['max_epochs']
    opt = torch.optim.AdamW(trainable, lr=args['learning_rate'], weight_decay=args['weight_decay'])
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args['warmup_ratio']), total_steps)
    holdout = DataLoader(prepared['holdout'], batch_size=1, shuffle=False, collate_fn=collate)

    def loss_of(batch):
        labels = batch['labels'].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        logits = model(**inputs).logits[:, :-1].float().contiguous()
        targets = labels[:, 1:].contiguous()
        n = (targets != -100).sum()
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction='sum')
        return loss_sum / n, n

    def evaluate():
        model.eval()
        total_loss, count = 0., 0
        with torch.no_grad():
            for batch in holdout:
                loss, n = loss_of(batch)
                assert torch.isfinite(loss), 'Nonfinite holdout loss'
                total_loss += loss.item() * n.item()
                count += n.item()
        model.train()
        return total_loss / count

    initial_loss, step, best, patience, start_epoch, best_epoch = None, 0, float('inf'), 0, 0, None
    resume_path = out / 'resume.pt'
    if resume_path.exists():
        from peft import set_peft_model_state_dict
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        assert state['data_identity'] == data_identity, 'Resume training data/settings changed'
        set_peft_model_state_dict(model, state['adapter'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        step, best, patience, start_epoch, best_epoch, initial_loss = [state[k] for k in ('step', 'best', 'patience', 'next_epoch', 'best_epoch', 'initial_loss')]
    else:
        initial_loss = evaluate()
        write(out / 'initialization_audit.json', {'initial_holdout_nll': initial_loss,
            'init': 'fresh zero-output LoRA on Base', 'adapter_count': len(model.peft_config),
            'all_lora_B_initialized_to_zero': True, 'data_identity': data_identity,
            'trainable_parameters': sum(p.numel() for p in trainable), 'train_rows': len(prepared['train']),
            'holdout_rows': len(prepared['holdout']), 'actual_fit_edits': fit_edit_count})
    model.train()
    started = time.monotonic()
    stop_reason = 'max_epochs'
    for epoch in range(start_epoch, args['max_epochs']):
        loader = DataLoader(prepared['train'], batch_size=1, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(SEED + epoch))
        opt.zero_grad(set_to_none=True)
        for idx, batch in enumerate(loader):
            loss, _ = loss_of(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            # Normalize the final short accumulation group by its actual size.
            group_size = min(args['gradient_accumulation_steps'], len(loader) - (idx // args['gradient_accumulation_steps']) * args['gradient_accumulation_steps'])
            (loss / group_size).backward()
            if (idx + 1) % args['gradient_accumulation_steps'] == 0 or idx + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log = {'epoch': epoch + 1, 'step': step, 'loss': loss.item(), 'lr': sched.get_last_lr()[0]}
                append(out / 'training_log.jsonl', log)
                print(json.dumps(log), flush=True)
        valid_loss = evaluate()
        checkpoint = out / f'epoch_{epoch + 1}'
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        if valid_loss < best:
            best, best_epoch, patience = valid_loss, epoch + 1, 0
        else:
            patience += 1
        append(out / 'holdout_log.jsonl', {'epoch': epoch + 1, 'step': step, 'holdout_target_token_nll': valid_loss,
            'best_nll': best, 'best_epoch': best_epoch, 'patience_used': patience})
        from peft import get_peft_model_state_dict
        state = {'adapter': get_peft_model_state_dict(model), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'best': best,
                 'patience': patience, 'next_epoch': epoch + 1, 'best_epoch': best_epoch, 'initial_loss': initial_loss}
        state['data_identity'] = data_identity
        tmp = resume_path.with_suffix('.tmp')
        torch.save(state, tmp)
        tmp.replace(resume_path)
        if patience >= args['early_stopping_patience_epochs']:
            stop_reason = 'TRAIN_internal_holdout_early_stopping'
            break
    selected = out / f'epoch_{best_epoch}'
    assert selected.exists()
    write(out / 'manifest.json', {'status': 'complete', 'selected_adapter': str(selected),
        'selected_adapter_sha256': file_sha(selected / 'adapter_model.safetensors'), 'best_epoch': best_epoch,
        'best_holdout_nll': best, 'initial_holdout_nll': initial_loss, 'optimizer_steps': step,
        'train_rows': len(prepared['train']), 'holdout_rows': len(prepared['holdout']),
        'stop_reason': stop_reason, 'runtime_seconds': time.monotonic() - started,
        'settings': args, 'test_used': False, 'valid43_used_for_training_or_checkpoint_selection': False,
        'train_file_sha256': file_sha(ROOT / 'data/train.jsonl'), 'holdout_file_sha256': file_sha(ROOT / 'data/holdout.jsonl')})


if __name__ == '__main__':
    train()
