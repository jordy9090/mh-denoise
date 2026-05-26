import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_probs(probs):
    if not isinstance(probs, dict):
        return ""
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}={v:.2f}" for k, v in items])


def build_prompt(ex):
    q = ex["question"].strip()
    u = ex["unsafe_response"].strip()
    d = ex["condition_dim"]
    probs = format_probs(ex.get("condition_probs", {}))
    ctype = ex.get("corruption_type", "risk")
    t = float(ex.get("corruption_level", 1.0))
    draft = (ex.get("corrupted_draft") or "").strip()

    draft_block = draft if draft else "(no corrupted draft; rewrite directly from the unsafe response)"

    return (
        "You are a safety-aware mental-health response denoiser.\n"
        "Your task is to rewrite an unsafe or low-quality counseling response into a safe, empathetic, specific, and professionally bounded response.\n"
        "Return only the final response shown to the user.\n"
        "Do not mention labels, risk tags, corruption, drafts, analysis, or hidden reasoning.\n"
        "Do not diagnose. Do not prescribe medication. Do not shame the user. Do not make unsupported clinical claims.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Active violation dimension:\n{d}\n\n"
        f"Aspect probabilities:\n{probs}\n\n"
        f"Corruption source:\n{ctype}\n\n"
        f"Corruption level t:\n{t:.2f}\n\n"
        f"Corrupted draft z_t:\n{draft_block}\n\n"
        "Safe response:\n"
    )


class DenoiseDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_len=1024, max_target_len=256):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        prompt = build_prompt(ex)
        target = ex["safe_response"].strip()
        if self.tokenizer.eos_token and not target.endswith(self.tokenizer.eos_token):
            target = target + self.tokenizer.eos_token

        pids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]

        tids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len,
        )["input_ids"]

        input_ids = pids + tids
        labels = [-100] * len(pids) + tids
        attn = [1] * len(input_ids)
        weight = float(ex.get("loss_weight", 1.0))

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "loss_weight": weight,
        }


@dataclass
class Collator:
    tokenizer: object
    max_len: int

    def __call__(self, batch):
        max_len = min(self.max_len, max(len(x["input_ids"]) for x in batch))
        pad = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels, weights = [], [], [], []
        for x in batch:
            ids = x["input_ids"][:max_len]
            attn = x["attention_mask"][:max_len]
            lab = x["labels"][:max_len]
            n = max_len - len(ids)
            input_ids.append(ids + [pad] * n)
            attention_mask.append(attn + [0] * n)
            labels.append(lab + [-100] * n)
            weights.append(float(x["loss_weight"]))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weight": torch.tensor(weights, dtype=torch.float),
        }


def weighted_causal_lm_loss(logits, labels, example_weights):
    # logits: [B, L, V], labels: [B, L]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab = shift_logits.size(-1)
    tok_loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())

    mask = (shift_labels != -100).float()
    per_ex = (tok_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return (per_ex * example_weights.to(per_ex.device)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        weights = batch.pop("loss_weight").to(device)
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        loss = weighted_causal_lm_loss(logits, labels, weights)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max_source_len", type=int, default=1024)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="linear")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    device = next(model.parameters()).device

    train_ds = DenoiseDataset(args.train_file, tokenizer, args.max_source_len, args.max_target_len)
    valid_ds = DenoiseDataset(args.valid_file, tokenizer, args.max_source_len, args.max_target_len)
    collator = Collator(tokenizer, args.max_source_len + args.max_target_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2)

    update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup = int(update_steps * args.warmup_ratio)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, warmup, update_steps)

    print("device:", device)
    print("model:", args.model)
    print("train rows:", len(train_ds))
    print("valid rows:", len(valid_ds))
    print("update steps:", update_steps)

    best = 999.0
    global_step = 0
    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            weights = batch.pop("loss_weight").to(device)
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            logits = model(**inputs).logits
            loss = weighted_causal_lm_loss(logits, labels, weights) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

                global_step += 1
                pbar.set_postfix({"loss": round(float(loss.item() * args.grad_accum), 4), "gstep": global_step})

                if global_step % args.eval_every == 0:
                    ev = evaluate(model, valid_loader, device)
                    print(f"[eval] step={global_step} loss={ev:.4f}")
                    if ev < best:
                        best = ev
                        save_dir = out_dir / "best"
                        model.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        print(f"[save] best -> {save_dir}")

                if global_step % args.save_every == 0:
                    ckpt = out_dir / f"step_{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"[save] checkpoint -> {ckpt}")

    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)

    with open(out_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[done] saved final -> {final}")
    print(f"[done] best eval loss: {best:.4f}")


if __name__ == "__main__":
    main()
