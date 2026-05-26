import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


DIMS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]
DIM2ID = {d: i for i, d in enumerate(DIMS)}


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


def norm_dim(x):
    x = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    return x if x in DIM2ID else "overall_quality"


def build_text(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "Question:\n"
        + q.strip()
        + "\n\nUnsafe response:\n"
        + u.strip()
        + "\n\nTask: identify the primary violated response-quality dimension."
    )


class RouterDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        text = build_text(ex)
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )
        d = norm_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
        label = DIM2ID[d]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []

        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    conf_rows = []

    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch)
        logits = out.logits.float()
        loss = loss_fn(logits, labels)

        if torch.isfinite(loss):
            losses.append(float(loss.item()))

        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

        for g, p in zip(labels.detach().cpu().tolist(), pred.detach().cpu().tolist()):
            conf_rows.append((g, p))

    acc = correct / max(1, total)
    avg_loss = sum(losses) / max(1, len(losses))
    model.train()
    return avg_loss, acc, conf_rows


def save_model(model, tokenizer, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "dims.json", "w", encoding="utf-8") as f:
        json.dump(DIMS, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(DIMS),
        problem_type="single_label_classification",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_ds = RouterDataset(args.train_file, tokenizer, args.max_len)
    valid_ds = RouterDataset(args.valid_file, tokenizer, args.max_len)
    collator = Collator(tokenizer)

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

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    print("device:", device)
    print("model:", args.model)
    print("train rows:", len(train_ds))
    print("valid rows:", len(valid_ds))
    print("dims:", DIMS)
    print("total steps:", total_steps)

    best_loss = float("inf")
    best_acc = 0.0
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            logits = out.logits.float()
            loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {global_step}; skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix({"loss": round(float(loss.item()), 4), "gstep": global_step})

            if global_step % args.eval_every == 0:
                ev_loss, ev_acc, _ = evaluate(model, valid_loader, device)
                print(f"[eval] step={global_step} loss={ev_loss:.4f} top1={ev_acc:.4f}")

                if ev_loss < best_loss:
                    best_loss = ev_loss
                    best_acc = ev_acc
                    save_model(model, tokenizer, out_dir / "best")
                    print(f"[save] best -> {out_dir / 'best'}")

    save_model(model, tokenizer, out_dir / "final")
    print(f"[done] saved final -> {out_dir / 'final'}")
    print(f"[done] best valid loss: {best_loss:.4f}")
    print(f"[done] best valid top1: {best_acc:.4f}")


if __name__ == "__main__":
    main()
