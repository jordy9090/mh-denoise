import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex, names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def build_condition(ex):
    q = get_field(ex, ["question", "query", "user_question"])
    u = get_field(ex, ["unsafe_response", "corrupted_response", "unsafe"])
    d = get_field(ex, ["target_dimension", "dimension", "violated_dimension"])
    return (
        f"question: {q}\n"
        f"unsafe response: {u}\n"
        f"violated dimension: {d}\n"
        f"safe response:"
    )


class MaskedRefineDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_source_len=256,
        max_target_len=128,
        min_mask_prob=0.15,
        max_mask_prob=0.85,
        seed=42,
    ):
        self.rows = rows
        self.tok = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        condition = build_condition(ex)
        target = get_field(ex, ["safe_response", "target_response", "response", "trg"])

        src_ids = self.tok(
            condition,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]

        tgt_ids = self.tok(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_len,
        )["input_ids"]

        if len(tgt_ids) == 0:
            tgt_ids = [self.tok.unk_token_id]

        # MDLM-style: timestep t를 mask probability로 단순화
        mask_prob = self.rng.uniform(self.min_mask_prob, self.max_mask_prob)

        corrupted_tgt = []
        labels_tgt = []
        for tid in tgt_ids:
            if self.rng.random() < mask_prob:
                corrupted_tgt.append(self.tok.mask_token_id)
                labels_tgt.append(tid)
            else:
                corrupted_tgt.append(tid)
                labels_tgt.append(-100)

        # 최소 1개는 반드시 mask
        if all(x == -100 for x in labels_tgt):
            j = self.rng.randrange(len(tgt_ids))
            corrupted_tgt[j] = self.tok.mask_token_id
            labels_tgt[j] = tgt_ids[j]

        input_ids = (
            [self.tok.cls_token_id]
            + src_ids
            + [self.tok.sep_token_id]
            + corrupted_tgt
            + [self.tok.sep_token_id]
        )

        labels = (
            [-100] * (1 + len(src_ids) + 1)
            + labels_tgt
            + [-100]
        )

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/splits/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/models/bert_masked_discrete_refiner_v0")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.train_file)
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * 0.1))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    print("train rows:", len(train_rows))
    print("val rows:", len(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    train_ds = MaskedRefineDataset(
        train_rows,
        tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        seed=args.seed,
    )
    val_ds = MaskedRefineDataset(
        val_rows,
        tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        seed=args.seed + 1,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
