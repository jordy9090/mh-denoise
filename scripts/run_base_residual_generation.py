"""Generate new TRAIN drafts or VALID residuals, never TEST outputs."""
import argparse
import json
import time

from residual_train_common import *


def run(stage):
    import torch
    from fullpaper_backbone_utils import load_fullpaper_tokenizer
    from run_gemma_selective_risk_refinement import load_base_model
    from selective_risk_refinement_utils import (build_sft_prompt, build_risk_tune_prompt,
        configure_risk_contract, cleanup_response, resolve_generation_eos_token_ids)
    contract = read(ROOT / 'run_contract.json')
    source = rows(ROOT / ('inputs/train276.jsonl' if stage == 'train' else 'inputs/valid43.jsonl'))
    path = ROOT / f'generation/{stage}.jsonl'
    history = {r['canonical_id']: r for r in rows(path)}
    assert len(history) == len(rows(path)), 'Duplicate generations'
    assert set(history) <= {r['canonical_id'] for r in source}
    for r in source:
        if r['canonical_id'] in history:
            assert history[r['canonical_id']]['input_sha256'] == sha(r)
    pending = [r for r in source if r['canonical_id'] not in history]
    if not pending:
        return
    gpu_idle()
    configure_risk_contract('fullpaper_v1')
    torch.manual_seed(SEED)
    tok = load_fullpaper_tokenizer(str(TOKENIZER))
    tok.padding_side = 'left'
    model = load_base_model(str(BASE), use_4bit=True)
    adapter = None
    if stage == 'valid':
        from peft import PeftModel
        adapter = read(ROOT / 'training/manifest.json')['selected_adapter']
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    eos, eos_source = resolve_generation_eos_token_ids(model, tok)
    batch_size = contract['settings']['generation']['batch_size_' + stage]
    max_source = 1024 if stage == 'train' else 4096
    started = time.monotonic()
    done = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        if stage == 'train':
            prompts = [build_sft_prompt(tok, r) for r in batch]
        else:
            prompts = [build_risk_tune_prompt(tok, {**r, 'sft_response': r['base_response'],
                'z_t_from_sft': r['base_response'], 'aspect_conditioning_status': 'available',
                'risk_component_mode': 'full'}) for r in batch]
        full_ids = [tok(p, add_special_tokens=True, truncation=False)['input_ids'] for p in prompts]
        encoded = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_source).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=1536, do_sample=False,
                repetition_penalty=1.15, no_repeat_ngram_size=4, pad_token_id=tok.pad_token_id,
                eos_token_id=eos, use_cache=True)
        width = encoded['input_ids'].shape[-1]
        for idx, (r, prompt, sequence) in enumerate(zip(batch, prompts, output)):
            token_ids = sequence[width:].tolist()
            stop = next((i for i, token in enumerate(token_ids) if token in eos), None)
            if stop is not None:
                token_ids = token_ids[:stop + 1]
            raw = tok.decode(token_ids, skip_special_tokens=True)
            response = extract(raw)
            used = encoded['input_ids'][idx][encoded['attention_mask'][idx].bool()].tolist()
            result = {**r, 'input_sha256': sha(r), 'adapter': adapter, 'response_raw': raw,
                'response': response, 'legacy_response': cleanup_response(raw),
                'response_sha256': sha(response), 'surface_flags': flags(response),
                'rendered_prompt': prompt, 'rendered_prompt_sha256': sha(prompt),
                'input_token_ids_sha256': sha(used), 'input_tokens': len(used),
                'source_truncated': len(full_ids[idx]) > len(used), 'generated_token_ids': token_ids,
                'eos_ids': eos, 'eos_source': eos_source, 'eos_reached': stop is not None,
                'technical_valid': bool(response.strip()) and stop is not None and len(full_ids[idx]) == len(used)}
            if stage == 'valid':
                original = next(x for x in rows(PREVIOUS / 'generation/base_draft_second_pass_controls_valid43.jsonl')
                                if x['canonical_id'] == r['canonical_id'] and x['arm'] == 'base_self_refine')
                prov = original['prompt_provenance']
                assert result['rendered_prompt_sha256'] == prov['rendered_prompt_sha256'], 'VALID prompt changed'
                assert result['input_token_ids_sha256'] == prov['input_token_ids_used_sha256'], 'VALID tokens changed'
            append(path, result)
            history[r['canonical_id']] = result
            done += 1
        elapsed = time.monotonic() - started
        progress = {'stage': stage, 'completed': len(history), 'total': len(source),
                    'rows_per_minute': round(done / elapsed * 60, 3),
                    'eta_minutes': round((len(source) - len(history)) / (done / elapsed) / 60, 2)}
        write(ROOT / f'generation/{stage}_progress.json', progress)
        print(json.dumps(progress), flush=True)
    write(ROOT / f'generation/{stage}_manifest.json', {'count': len(history), 'technical_valid': sum(r['technical_valid'] for r in history.values()),
          'output_sha256': file_sha(path), 'runtime_seconds': time.monotonic() - started, 'adapter': adapter})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'valid'])
    run(parser.parse_args().stage)
