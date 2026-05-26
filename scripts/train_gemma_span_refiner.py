import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


DIM_KEYWORDS = {
    "toxicity": [
        "sensitive", "overreact", "overreacting", "loosen", "banter", "handle",
        "honestly", "probably", "fault", "blame", "dramatic", "big deal",
        "mountain", "molehill", "paranoid", "conspiracies",
    ],
    "medical_advice": [
        "diagnosable", "diagnosis", "disorder", "condition", "symptom", "treatment",
        "medication", "medicine", "cognitive restructuring", "anxiety", "depression",
        "requires", "should", "need", "must", "criteria", "intervention",
    ],
    "factual_consistency": [
        "clearly", "always", "often", "direct indicator", "underlying", "suggests",
        "means", "drives", "manifest", "resilient", "stability", "known psychological",
    ],
    "specificity": [
        "general", "generally", "things", "stuff", "sometimes", "normal", "common",
        "focus", "helpful approach", "wellness", "support", "take care",
    ],
    "empathy": [
        "observe", "noticing", "common behavior", "pattern", "involved",
        "detached", "complicated",
    ],
    "overall_quality": [
        "just", "maybe", "things", "stuff", "handle", "figure out", "focus",
        "general", "complicated", "sometimes", "resolve", "next steps yourself",
    ],
}


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex: Dict, *names: str, default: str = "") -> str:
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def normalize_dim(dim: str) -> str:
    d = str(dim).strip().lower().replace("-", "_").replace(" ", "_")
    return d if d in DIM_KEYWORDS else "overall_quality"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def mask_bad_spans_by_text(unsafe: str, dim: str, seed: int) -> str:
    """
    Build a masked unsafe draft from the unsafe response.
    This aligns train-time input with inference-time input.
    """
    rng = random.Random(seed)
    dim = normalize_dim(dim)
    text = unsafe
    mask = "[MASK]"

    keys = DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]
    replaced_any = False

    # phrase-level masking
    for key in sorted(keys, key=len, reverse=True):
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        if pattern.search(text) and rng.random() < 0.9:
            text = pattern.sub(mask, text, count=1)
            replaced_any = True

    # sentence-level masking
    sents = split_sentences(text)
    if sents:
        scored = []
        for i, s in enumerate(sents):
            sl = s.lower()
            score = sum(1 for k in keys if k.lower() in sl)
            if len(s.split()) < 7:
                score += 0.3
            scored.append((score, i, s))

        scored.sort(reverse=True)
        n_mask = 1 if replaced_any else min(2, len(sents))

        for _, i, _ in scored[:n_mask]:
            if rng.random() < 0.85 or not replaced_any:
                sents[i] = mask
                replaced_any = True

        text = " ".join(sents)

    if mask not in text:
        text = mask + " " + text

    # Let the model add a safe closing if needed.
    if rng.random() < 0.7:
        if not text.rstrip().endswith((".", "!", "?")):
            text = text.rstrip() + "."
        text += " [MASK]"

    return text


def build_prompt(ex: Dict, seed: int) -> str:
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
    draft = mask_bad_spans_by_text(u, d, seed)

    return (
        "You are refining a mental-health counseling response.\n"
        "Rewrite the unsafe response into a safe, empathetic, specific, and professionally bounded response.\n"
        "Do not diagnose. Do not give direct medical instructions. Do not shame the user.\n\n"
        f"Question:\n{q}\n\n"
        f"Violated quality dimension:\n{d}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Masked unsafe draft:\n{draft}\n\n"
        "Safe response:\n"
    )


class GemmaSpanDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_source_len: int = 768,
        max_target_len: int = 256,
        examples_per_row: int = 3,
        seed: int = 42,
    ):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.examples_per_row = examples_per_row
        self.seed = seed

    def __len__(self):
        return len(self.rows) * self.examples_per_row

    def __getitem__(self, idx):
        row_idx = idx // self.examples_per_row
        variant_idx = idx % self.examples_per_row
        ex = self.rows[row_idx]

        prompt = build_prompt(ex, self.seed + row_idx * 37 + variant_idx)
        target = get_field(ex, "safe_response", "target_response", "response").strip()
        if not target.endswith(self.tokenizer.eos_token or ""):
            target = target + (self.tokenizer.eos_token or "")

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]

        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class CausalCollator:
    tokenizer: object
    max_len: int

    def __call__(self, batch):
        max_len = min(self.max_len, max(len(x["input_ids"]) for x in batch))
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []

        for x in batch:
            ids = x["input_ids"][:max_len]
            attn = x["attention_mask"][:max_len]
            lab = x["labels"][:max_len]

            pad = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append(attn + [0] * pad)
            labels.append(lab + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--examples_per_row", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["linear"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = next(model.parameters()).device

    train_ds = GemmaSpanDataset(
        args.train_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        args.examples_per_row,
        args.seed,
    )
    valid_ds = GemmaSpanDataset(
        args.valid_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        1,
        args.seed + 999,
    )

    collator = CausalCollator(tokenizer, args.max_source_len + args.max_target_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(update_steps * args.warmup_ratio)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, update_steps)

    print("device:", device)
    print("model:", args.model)
    print("train examples:", len(train_ds))
    print("valid examples:", len(valid_ds))
    print("update steps:", update_steps)

    best = float("inf")
    global_step = 0
    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
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
                        best_dir = out_dir / "best"
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        print(f"[save] best -> {best_dir}")

                if global_step % args.save_every == 0:
                    ckpt_dir = out_dir / f"step_{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[save] checkpoint -> {ckpt_dir}")

    final_dir = out_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(out_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[done] saved final -> {final_dir}")
    print(f"[done] best eval loss: {best:.4f}")


if __name__ == "__main__":
    main()
