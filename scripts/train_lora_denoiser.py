import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_user_prompt(ex):
    return f"""
You are a mental-health response denoiser.

Task:
Rewrite the corrupted response into the safe target style.

Requirements:
- Be empathetic and specific.
- Stay clinically bounded.
- Do not diagnose.
- Do not give medication dosage or medical instructions.
- Remove judgmental, dismissive, unsupported, or unsafe content.
- Keep the response useful and natural.

User question:
{ex["question"]}

Corrupted response:
{ex["unsafe_response"]}

Violation dimension:
{ex["target_dimension"]}

Safe response:
""".strip()


class DenoisingSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=1024):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]

        messages = [{"role": "user", "content": build_user_prompt(ex)}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        target_text = ex["safe_response"].strip() + self.tokenizer.eos_token
        full_text = prompt_text + target_text

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))

        for i in range(prompt_len):
            labels[i] = -100

        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def split_rows(rows, val_ratio=0.1):
    if len(rows) < 20:
        return rows, rows

    n_val = max(1, int(len(rows) * val_ratio))
    return rows[:-n_val], rows[-n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default="outputs/models/gemma_lora_denoiser_v0")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    rows = load_jsonl(args.train_file)
    train_rows, val_rows = split_rows(rows)

    print(f"Loaded rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = DenoisingSFTDataset(train_rows, tokenizer, args.max_length)
    val_dataset = DenoisingSFTDataset(val_rows, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved LoRA denoiser to {args.output_dir}")


if __name__ == "__main__":
    main()