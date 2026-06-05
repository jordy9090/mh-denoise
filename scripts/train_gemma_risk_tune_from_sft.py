import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from selective_risk_refinement_utils import (
    build_risk_tune_prompt,
    canonical_example,
    clean_text,
    get_field,
    make_zt_from_response,
    read_jsonl,
    score_candidate,
    write_jsonl,
)


def load_base_model(model_name: str, use_4bit: bool):
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_classifier(path: str, device):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def enrich_rows(rows: List[Dict], router, router_tok, risk_model, risk_tok, device, args, split_name: str):
    enriched = []
    skipped = 0
    for raw in tqdm(rows, desc=f"enrich {split_name}"):
        row = canonical_example(raw)
        sft_response = clean_text(get_field(row, "sft_response", "professor_peft_response", "peft_response"))
        if not sft_response:
            skipped += 1
            continue
        row["sft_response"] = sft_response

        q = row["question"]
        score = score_candidate(
            q,
            sft_response,
            router,
            router_tok,
            risk_model,
            risk_tok,
            device,
            router_max_len=args.router_max_len,
            risk_max_len=args.risk_max_len,
        )
        z_t, infos = make_zt_from_response(
            sft_response,
            score["g"],
            score["risk_vecs"],
            strategy=args.zt_strategy,
            t=args.timestep,
            T=args.T,
            mask_token=args.mask_token,
            risk_threshold=args.risk_threshold,
            mask_threshold=args.mask_threshold,
            t2_frac=args.t2_frac,
            t3_frac=args.t3_frac,
            risk_tag_format=args.risk_tag_format,
        )

        row["g_sft"] = score["g"]
        row["sft_risk_score"] = score["risk_score"]
        row["sft_span_risks"] = score["span_risks"]
        row["z_t_from_sft"] = z_t
        row["z_t_from_sft_infos"] = infos
        row["zt_strategy"] = args.zt_strategy
        row["timestep"] = args.timestep
        enriched.append(row)
    print(f"{split_name} rows:", len(enriched), "skipped missing sft_response:", skipped)
    return enriched


def oversample_high_risk(rows: List[Dict], threshold: float, factor: int):
    if factor <= 1:
        return rows
    out = []
    high = 0
    for row in rows:
        out.append(row)
        if float(row.get("sft_risk_score", 0.0)) >= threshold:
            high += 1
            for _ in range(factor - 1):
                out.append(dict(row))
    print("high-risk rows:", high, "oversampled train rows:", len(out))
    return out


class RiskTuneDataset(Dataset):
    def __init__(self, rows, tokenizer, max_source_len, max_target_len, lambda_y=0.3, max_token_weight=1.5):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.lambda_y = float(lambda_y)
        self.max_token_weight = float(max_token_weight)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        prompt = build_risk_tune_prompt(self.tokenizer, ex)
        target = clean_text(get_field(ex, "safe_response", "target_response", "target", "response"))
        eos = self.tokenizer.eos_token or ""
        if eos and not target.endswith(eos):
            target = target + eos

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len,
        )["input_ids"]

        risk_weight = 1.0 + self.lambda_y * min(1.0, max(0.0, float(ex.get("sft_risk_score", 0.0))))
        risk_weight = min(self.max_token_weight, max(1.0, risk_weight))
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        token_weights = [0.0] * len(prompt_ids) + [risk_weight] * len(target_ids)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_weights": token_weights,
        }


@dataclass
class Collator:
    tokenizer: object
    max_len: int

    def __call__(self, batch):
        max_len = min(self.max_len, max(len(x["input_ids"]) for x in batch))
        pad = self.tokenizer.pad_token_id
        ids, masks, labels, weights = [], [], [], []
        for item in batch:
            input_ids = item["input_ids"][:max_len]
            attention_mask = item["attention_mask"][:max_len]
            labs = item["labels"][:max_len]
            token_weights = item["token_weights"][:max_len]
            n_pad = max_len - len(input_ids)
            ids.append(input_ids + [pad] * n_pad)
            masks.append(attention_mask + [0] * n_pad)
            labels.append(labs + [-100] * n_pad)
            weights.append(token_weights + [0.0] * n_pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": torch.tensor(weights, dtype=torch.float),
        }


def weighted_ce(logits, labels, token_weights):
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = token_weights[:, 1:].float().contiguous()
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    weights = torch.where(mask > 0, torch.clamp(shift_weights, min=1.0), torch.zeros_like(shift_weights))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        weights = batch.pop("token_weights").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = weighted_ce(model(**batch).logits, labels, weights)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_adapter(model, tokenizer, output_dir: Path, args, metrics=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    payload = vars(args).copy()
    if metrics:
        payload["metrics"] = metrics
    with open(output_dir / "risk_tune_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", "--model", dest="base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--init_adapter_dir", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--zt_strategy", choices=["threshold", "staged", "staged_risk", "risk_tag"], default="staged_risk")
    ap.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_train_epochs", "--epochs", dest="epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", "--batch_size", dest="batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", "--eval_batch_size", dest="eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", "--grad_accum", dest="grad_accum", type=int, default=8)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=1536)
    ap.add_argument("--max_source_len", type=int, default=None)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--router_max_len", type=int, default=512)
    ap.add_argument("--risk_max_len", type=int, default=384)
    ap.add_argument("--lambda_y", type=float, default=0.3)
    ap.add_argument("--max_token_weight", type=float, default=1.5)
    ap.add_argument("--risk_oversample_threshold", type=float, default=0.35)
    ap.add_argument("--risk_oversample_factor", type=int, default=2)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--timestep", type=int, default=3)
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--mask_threshold", type=float, default=0.35)
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--t2_frac", type=float, default=0.33)
    ap.add_argument("--t3_frac", type=float, default=0.66)
    ap.add_argument("--risk_tag_format", default="[Risk: {dim}] {span} [/Risk]")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--enable_gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    if args.max_source_len is None:
        args.max_source_len = max(128, int(args.max_length) - int(args.max_target_len))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_tok, router = load_classifier(args.router_dir, device)
    risk_tok, risk_model = load_classifier(args.risk_scorer_dir, device)

    train_rows = enrich_rows(read_jsonl(args.train_file), router, router_tok, risk_model, risk_tok, device, args, "train")
    valid_rows = enrich_rows(read_jsonl(args.valid_file), router, router_tok, risk_model, risk_tok, device, args, "valid")
    write_jsonl(train_rows, str(out / "risk_tune_train_enriched.jsonl"))
    write_jsonl(valid_rows, str(out / "risk_tune_valid_enriched.jsonl"))
    train_rows = oversample_high_risk(train_rows, args.risk_oversample_threshold, args.risk_oversample_factor)

    tokenizer = AutoTokenizer.from_pretrained(args.init_adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("base model:", args.base_model)
    print("init adapter:", args.init_adapter_dir)
    print("router:", args.router_dir)
    print("risk scorer:", args.risk_scorer_dir)
    print("load_in_4bit:", use_4bit)
    print("zt_strategy:", args.zt_strategy, "timestep:", args.timestep)
    print("lambda_y:", args.lambda_y, "lr:", args.learning_rate)
    base = load_base_model(args.base_model, use_4bit)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=args.enable_gradient_checkpointing)
    model = PeftModel.from_pretrained(base, args.init_adapter_dir, is_trainable=True)
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    model_device = next(model.parameters()).device

    train_ds = RiskTuneDataset(
        train_rows,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        lambda_y=args.lambda_y,
        max_token_weight=args.max_token_weight,
    )
    valid_ds = RiskTuneDataset(
        valid_rows,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        lambda_y=args.lambda_y,
        max_token_weight=args.max_token_weight,
    )
    collator = Collator(tokenizer, args.max_source_len + args.max_target_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_updates = args.max_train_steps or (updates_per_epoch * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_updates)
    opt = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_updates)

    best = float("inf")
    step = 0
    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for idx, batch in enumerate(pbar):
            labels = batch.pop("labels").to(model_device)
            weights = batch.pop("token_weights").to(model_device)
            batch = {k: v.to(model_device) for k, v in batch.items()}
            loss = weighted_ce(model(**batch).logits, labels, weights)
            (loss / args.grad_accum).backward()
            should_step = ((idx + 1) % args.grad_accum == 0) or (idx + 1 == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.set_postfix({"loss": round(float(loss.item()), 4), "step": step})

                if args.eval_every > 0 and step % args.eval_every == 0:
                    ev = evaluate(model, valid_loader, model_device)
                    print(f"[eval] step={step} loss={ev:.4f}")
                    if ev < best:
                        best = ev
                        save_adapter(model, tokenizer, out / "best", args, {"valid_loss": best, "step": step})
                        print("[save] best ->", out / "best")

                if args.save_every > 0 and step % args.save_every == 0:
                    save_adapter(model, tokenizer, out / f"step_{step}", args, {"step": step})
                    print("[save] checkpoint ->", out / f"step_{step}")

                if args.max_train_steps and step >= args.max_train_steps:
                    break
        if args.max_train_steps and step >= args.max_train_steps:
            break

    final_loss = evaluate(model, valid_loader, model_device) if len(valid_ds) else None
    if final_loss is not None and final_loss < best:
        best = final_loss
        save_adapter(model, tokenizer, out / "best", args, {"valid_loss": best, "step": step})
        print("[save] best ->", out / "best")
    save_adapter(model, tokenizer, out / "final", args, {"valid_loss": final_loss, "step": step})
    print("[done] best valid loss:", best)
    print("[done] final ->", out / "final")


if __name__ == "__main__":
    main()
