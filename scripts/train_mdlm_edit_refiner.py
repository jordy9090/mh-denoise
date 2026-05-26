import argparse
import json
import random
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[QUESTION]",
        "[UNSAFE]",
        "[DIMENSION]",
        "[DRAFT]",
    ]
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "that", "this", "it", "is", "are", "was", "were", "be", "been", "being",
    "you", "your", "i", "we", "they", "he", "she", "them", "as", "at", "by",
    ".", ",", "'", '"', "!", "?", ":", ";", "-", "##s", "##re", "##m"
}

DIM_KEYWORDS = {
    "toxicity": [
        "sensitive", "overreact", "overreacting", "loosen", "banter", "handle",
        "just", "probably", "honestly", "fault", "blame", "dramatic", "stupid",
    ],
    "medical_advice": [
        "diagnosable", "diagnosis", "disorder", "condition", "symptom", "treatment",
        "medication", "medicine", "cognitive", "restructuring", "therapy", "therapist",
        "anxiety", "depression", "should", "need", "requires",
    ],
    "factual_consistency": [
        "clearly", "always", "often", "direct", "indicator", "underlying", "suggests",
        "means", "because", "drives", "manifest", "resilient", "stability", "pattern",
    ],
    "specificity": [
        "general", "generally", "things", "stuff", "sometimes", "normal", "common",
        "focus", "helpful", "approach", "wellness", "support", "people",
    ],
    "empathy": [
        "observe", "noticing", "common", "behavior", "pattern", "situation",
        "involved", "feeling", "feelings", "awkwardness",
    ],
    "overall_quality": [
        "just", "maybe", "things", "stuff", "handle", "figure", "focus", "general",
        "complicated", "sometimes", "next", "yourself",
    ],
}

SAFE_ANCHOR_KEYWORDS = {
    "toxicity": ["understandable", "makes", "sense", "frustrated", "hurtful", "respectful", "okay"],
    "empathy": ["sounds", "sorry", "understandable", "frustrated", "distressing", "difficult"],
    "specificity": ["specific", "example", "try", "consider", "express", "step", "support"],
    "medical_advice": ["if", "suspect", "professional", "qualified", "authorities", "law", "emergency", "immediate"],
    "factual_consistency": ["may", "can", "might", "if", "could", "avoid", "assume"],
    "overall_quality": ["sounds", "important", "however", "consider", "support", "safety"],
}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex, *names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def normalize_dim(dim: str) -> str:
    d = dim.strip().lower().replace("-", "_").replace(" ", "_")
    if d in DIM_KEYWORDS:
        return d
    return "overall_quality"


def build_condition(ex: Dict) -> str:
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
    return (
        f"[QUESTION] {q}\n"
        f"[UNSAFE] {u}\n"
        f"[DIMENSION] {d}\n"
        f"[DRAFT]"
    )


def token_is_content(tok: str) -> bool:
    t = tok.lower()
    if t in STOPWORDS:
        return False
    if len(t) <= 1:
        return False
    if re.fullmatch(r"[^\w]+", t):
        return False
    return True


def expand_positions(pos: Set[int], n: int, radius: int = 2) -> Set[int]:
    out = set()
    for p in pos:
        for j in range(max(0, p - radius), min(n, p + radius + 1)):
            out.add(j)
    return out


def choose_corruption_positions(tokens: List[str], dim: str, min_rate: float, max_rate: float) -> Set[int]:
    n = len(tokens)
    dim = normalize_dim(dim)
    low_tokens = [t.lower().replace("##", "") for t in tokens]

    pos = set()

    # safe anchor spans: train the model to reconstruct useful safe phrases for that dimension
    anchors = SAFE_ANCHOR_KEYWORDS.get(dim, []) + SAFE_ANCHOR_KEYWORDS["overall_quality"]
    for i, t in enumerate(low_tokens):
        if any(k in t for k in anchors) and token_is_content(tokens[i]):
            pos.add(i)

    # random content spans, simulating diffusion timestep severity
    target_rate = random.uniform(min_rate, max_rate)
    content_positions = [i for i, t in enumerate(tokens) if token_is_content(t)]

    random.shuffle(content_positions)
    need = max(1, int(len(content_positions) * target_rate))

    for i in content_positions[:need]:
        pos.add(i)

    # span expansion makes corruption less token-confetti-like
    pos = expand_positions(pos, n, radius=random.choice([1, 2]))

    # never corrupt special/punctuation-like tokens too aggressively
    pos = {i for i in pos if 0 <= i < n and token_is_content(tokens[i])}

    return pos


def sample_replacement_from_unsafe(unsafe_tokens: List[int], tokenizer, avoid_ids: Set[int]):
    candidates = []
    for tid in unsafe_tokens:
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        if int(tid) not in avoid_ids and token_is_content(tok):
            candidates.append(int(tid))
    if not candidates:
        return tokenizer.mask_token_id
    return random.choice(candidates)


class EditDenoiseDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        max_source_len=256,
        max_target_len=160,
        min_corrupt_rate=0.15,
        max_corrupt_rate=0.55,
        replace_prob=0.35,
        examples_per_row=2,
    ):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.min_corrupt_rate = min_corrupt_rate
        self.max_corrupt_rate = max_corrupt_rate
        self.replace_prob = replace_prob
        self.examples_per_row = examples_per_row

    def __len__(self):
        return len(self.rows) * self.examples_per_row

    def __getitem__(self, idx):
        ex = self.rows[idx // self.examples_per_row]
        dim = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
        cond = build_condition(ex)
        safe = get_field(ex, "safe_response", "target_response", "response")
        unsafe = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")

        cond_ids = self.tokenizer(
            cond,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]

        safe_ids = self.tokenizer(
            safe,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len - 1,
        )["input_ids"] + [self.tokenizer.sep_token_id]

        unsafe_ids = self.tokenizer(
            unsafe,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len,
        )["input_ids"]

        safe_tokens = self.tokenizer.convert_ids_to_tokens(safe_ids)
        corrupt_pos = choose_corruption_positions(
            safe_tokens,
            dim,
            self.min_corrupt_rate,
            self.max_corrupt_rate,
        )

        avoid_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.mask_token_id,
            self.tokenizer.unk_token_id,
        }

        corrupted = list(safe_ids)
        labels = [-100] * len(safe_ids)

        for p in corrupt_pos:
            labels[p] = safe_ids[p]
            if random.random() < self.replace_prob:
                corrupted[p] = sample_replacement_from_unsafe(unsafe_ids, self.tokenizer, avoid_ids)
            else:
                corrupted[p] = self.tokenizer.mask_token_id

        input_ids = cond_ids + corrupted
        full_labels = [-100] * len(cond_ids) + labels
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": full_labels,
        }


@dataclass
class Collator:
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
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_corrupt_rate", type=float, default=0.12)
    ap.add_argument("--max_corrupt_rate", type=float, default=0.45)
    ap.add_argument("--replace_prob", type=float, default=0.35)
    ap.add_argument("--examples_per_row", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=250)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = EditDenoiseDataset(
        args.train_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        args.min_corrupt_rate,
        args.max_corrupt_rate,
        args.replace_prob,
        args.examples_per_row,
    )
    valid_ds = EditDenoiseDataset(
        args.valid_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        args.min_corrupt_rate,
        args.max_corrupt_rate,
        args.replace_prob,
        1,
    )

    collator = Collator(tokenizer, args.max_source_len + args.max_target_len)

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
