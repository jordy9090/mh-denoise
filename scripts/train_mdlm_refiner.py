import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
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
        "[SAFE]",
    ]
}


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex, *names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def build_condition(ex: Dict) -> str:
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = get_field(ex, "target_dimension", "dimension", "violated_dimension")
    return (
        f"[QUESTION] {q}\n"
        f"[UNSAFE] {u}\n"
        f"[DIMENSION] {d}\n"
        f"[SAFE]"
    )


class MaskedRefinementDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_source_len: int = 256,
        max_target_len: int = 160,
        min_mask_rate: float = 0.15,
        max_mask_rate: float = 0.95,
        mask_schedule: str = "uniform",
    ):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.min_mask_rate = min_mask_rate
        self.max_mask_rate = max_mask_rate
        self.mask_schedule = mask_schedule

    def __len__(self):
        return len(self.rows)

    def sample_mask_rate(self) -> float:
        # MDLM-style: sample a diffusion time t and map it to a mask ratio.
        # High t means more corrupted/masked target.
        if self.mask_schedule == "beta":
            t = random.betavariate(0.7, 0.7)
        else:
            t = random.random()
        return self.min_mask_rate + t * (self.max_mask_rate - self.min_mask_rate)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        cond = build_condition(ex)
        target = get_field(ex, "safe_response", "target_response", "response")

        cond_ids = self.tokenizer(
            cond,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]

        # target side: no [CLS], keep [SEP] as target terminator.
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len - 1,
        )["input_ids"]
        target_ids = target_ids + [self.tokenizer.sep_token_id]

        mask_rate = self.sample_mask_rate()

        corrupted_target = []
        labels = []
        maskable_positions = list(range(len(target_ids)))

        # ensure at least one masked position
        n_mask = max(1, int(round(len(maskable_positions) * mask_rate)))
        masked_set = set(random.sample(maskable_positions, min(n_mask, len(maskable_positions))))

        for j, tok in enumerate(target_ids):
            if j in masked_set:
                corrupted_target.append(self.tokenizer.mask_token_id)
                labels.append(tok)
            else:
                corrupted_target.append(tok)
                labels.append(-100)

        input_ids = cond_ids + corrupted_target
        attention_mask = [1] * len(input_ids)

        # labels only for target positions; source positions ignored
        full_labels = [-100] * len(cond_ids) + labels

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": full_labels,
            "mask_rate": mask_rate,
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
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_mask_rate", type=float, default=0.15)
    ap.add_argument("--max_mask_rate", type=float, default=0.95)
    ap.add_argument("--mask_schedule", choices=["uniform", "beta"], default="beta")
    ap.add_argument("--eval_every", type=int, default=100)
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

    train_ds = MaskedRefinementDataset(
        args.train_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        args.min_mask_rate,
        args.max_mask_rate,
        args.mask_schedule,
    )

    if args.valid_file:
        valid_ds = MaskedRefinementDataset(
            args.valid_file,
            tokenizer,
            args.max_source_len,
            args.max_target_len,
            args.min_mask_rate,
            args.max_mask_rate,
            args.mask_schedule,
        )
    else:
        valid_ds = None

    collator = Collator(tokenizer, args.max_source_len + args.max_target_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )

    valid_loader = None
    if valid_ds:
        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
        )

    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    print("device:", device)
    print("train rows:", len(train_ds))
    print("valid rows:", len(valid_ds) if valid_ds else 0)
    print("total update steps:", total_update_steps)

    global_step = 0
    best_eval = float("inf")
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix({"loss": round(loss.item() * args.grad_accum, 4), "gstep": global_step})

                if valid_loader and global_step % args.eval_every == 0:
                    eval_loss = evaluate(model, valid_loader, device)
                    print(f"[eval] step={global_step} loss={eval_loss:.4f}")

                    if eval_loss < best_eval:
                        best_eval = eval_loss
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
    if best_eval < float("inf"):
        print(f"[done] best eval loss: {best_eval:.4f}")


if __name__ == "__main__":
    main()
