import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
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


def mask_bad_spans_by_text(unsafe: str, dim: str, mask_token: str, seed: int, min_mask_sentences: int = 1) -> str:
    """
    Build a masked unsafe draft.
    We mask dimension-related bad spans and some low-quality sentences.
    This is heuristic, but it aligns train and inference: both start from unsafe draft.
    """
    rng = random.Random(seed)
    dim = normalize_dim(dim)
    text = unsafe

    keys = DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]
    replaced_any = False

    # phrase-level masking
    for key in sorted(keys, key=len, reverse=True):
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        if pattern.search(text) and rng.random() < 0.85:
            text = pattern.sub(mask_token, text, count=1)
            replaced_any = True

    # sentence-level masking if few phrase matches
    sents = split_sentences(text)
    if sents:
        scored = []
        for i, s in enumerate(sents):
            sl = s.lower()
            score = sum(1 for k in keys if k.lower() in sl)
            # generic / unsafe sentence gets higher chance
            if len(s.split()) < 6:
                score += 0.5
            scored.append((score, i, s))

        scored.sort(reverse=True)
        n_mask = min(len(sents), max(min_mask_sentences, 1 if replaced_any else 2))

        for _, i, _ in scored[:n_mask]:
            if rng.random() < 0.75 or not replaced_any:
                sents[i] = mask_token
                replaced_any = True

        text = " ".join(sents)

    # fallback: insert a mask at the beginning if no mask was made
    if mask_token not in text:
        text = mask_token + " " + text

    # add a small tail mask so decoder can add a safe closing sentence
    if rng.random() < 0.65:
        text = text.rstrip()
        if not text.endswith((".", "!", "?")):
            text += "."
        text += f" {mask_token}"

    return text


def build_input(ex: Dict, tokenizer, seed: int) -> str:
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))

    mask_token = tokenizer.mask_token if tokenizer.mask_token else "<mask>"
    draft = mask_bad_spans_by_text(u, d, mask_token, seed)

    # Instruction helps FLAN-T5 use condition. BART will just treat it as text.
    return (
        "Refine the unsafe counseling response into a safe, empathetic, specific, and professionally bounded response.\n"
        f"Question: {q}\n"
        f"Violated quality dimension: {d}\n"
        f"Unsafe response: {u}\n"
        f"Masked unsafe draft: {draft}\n"
        "Safe response:"
    )


class Seq2SeqRefineDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_source_len: int = 512,
        max_target_len: int = 224,
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

        inp = build_input(ex, self.tokenizer, self.seed + row_idx * 31 + variant_idx)
        target = get_field(ex, "safe_response", "target_response", "response")

        src = self.tokenizer(
            inp,
            max_length=self.max_source_len,
            truncation=True,
            padding=False,
        )
        tgt = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )

        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": tgt["input_ids"],
        }


@dataclass
class Seq2SeqCollator:
    tokenizer: object
    max_source_len: int
    max_target_len: int

    def __call__(self, batch):
        pad_id = self.tokenizer.pad_token_id
        max_src = min(self.max_source_len, max(len(x["input_ids"]) for x in batch))
        max_tgt = min(self.max_target_len, max(len(x["labels"]) for x in batch))

        input_ids, attention_mask, labels = [], [], []

        for x in batch:
            ids = x["input_ids"][:max_src]
            attn = x["attention_mask"][:max_src]
            lab = x["labels"][:max_tgt]

            src_pad = max_src - len(ids)
            tgt_pad = max_tgt - len(lab)

            input_ids.append(ids + [pad_id] * src_pad)
            attention_mask.append(attn + [0] * src_pad)
            labels.append(lab + [-100] * tgt_pad)

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
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.08)
    ap.add_argument("--examples_per_row", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=250)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # T5 has no mask_token. Add one for draft marking.
    added = False
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})
        tokenizer.mask_token = "<mask>"
        added = True

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if added:
        model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = Seq2SeqRefineDataset(
        args.train_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        args.examples_per_row,
        args.seed,
    )
    valid_ds = Seq2SeqRefineDataset(
        args.valid_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        1,
        args.seed + 999,
    )

    collator = Seq2SeqCollator(tokenizer, args.max_source_len, args.max_target_len)

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
