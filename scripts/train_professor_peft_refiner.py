import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


DIMS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]

MASK_MARKERS = ["<MASK>", "[MASK]", "[needs revision]", "[NEEDS REVISION]"]


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex: Dict, *names: str, default: str = "") -> str:
    for name in names:
        value = ex.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_draft(text: str) -> str:
    """Decoder-only LMs often copy literal mask tokens.

    The professor-style method still uses z_t as a corrupted draft, but we expose it
    to Gemma as a risk-deleted draft instead of a BERT-style mask-filling string.
    This keeps the denoising signal while reducing mask-token leakage.
    """
    text = str(text or "")
    for marker in MASK_MARKERS:
        text = text.replace(marker, " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if not text:
        return "No usable partial draft is available. Rewrite from the question and unsafe response."
    return text


def format_aspect_signal(ex: Dict) -> str:
    g = ex.get("g")
    if isinstance(g, list) and len(g) == len(DIMS):
        pairs = sorted(zip(DIMS, g), key=lambda x: float(x[1]), reverse=True)
        active = [f"{name}={float(score):.2f}" for name, score in pairs if float(score) >= 0.10]
        return ", ".join(active[:4]) if active else f"{pairs[0][0]}={float(pairs[0][1]):.2f}"

    for key in ["target_dimension", "condition_dim", "used_dimension", "predicted_dimension", "dimension"]:
        value = ex.get(key)
        if value:
            return str(value)
    return "unspecified"


def build_plain_sft_user_content(ex: Dict) -> str:
    q = clean_text(get_field(ex, "question", "query", "user_question"))
    u = clean_text(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"))
    return (
        "Rewrite the unsafe counseling response into a safe final response.\n"
        "Use only the user question and the unsafe response. "
        "Do not use hidden gold labels, target dimensions, router scores, or corrupted drafts.\n"
        "Return only the final response. Do not include section labels, analysis, or metadata.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        "Safety requirements:\n"
        "- Validate the user's feelings without blaming or shaming them.\n"
        "- Do not diagnose the user or state that they have a disorder.\n"
        "- Do not give direct medication or treatment instructions.\n"
        "- Give concrete, bounded next steps when appropriate.\n"
        "- Encourage professional or trusted support for serious or persistent distress."
    )


def build_user_content(ex: Dict, prompt_style: str = "professor") -> str:
    if prompt_style == "sft_plain":
        return build_plain_sft_user_content(ex)

    q = clean_text(get_field(ex, "question", "query", "user_question"))
    u = clean_text(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"))
    z = sanitize_draft(get_field(ex, "z_t", "draft", "masked_unsafe_draft", default=""))
    aspect = format_aspect_signal(ex)
    t = ex.get("t", ex.get("corruption_level", ""))
    T = ex.get("T", "")
    t_str = f"{t}/{T}" if str(t) and str(T) else str(t)

    return (
        "Rewrite the unsafe counseling response into a safe final response.\n"
        "Use the question, the unsafe response, the aspect signal, and the corrupted draft.\n"
        "Return only the final response. Do not include section labels, analysis, metadata, or the draft.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Aspect signal:\n{aspect}\n\n"
        f"Corruption strength:\n{t_str}\n\n"
        f"Risk-deleted draft z_t:\n{z}\n\n"
        "Safety requirements:\n"
        "- Validate the user's feelings without blaming or shaming them.\n"
        "- Do not diagnose the user or state that they have a disorder.\n"
        "- Do not give direct medication or treatment instructions.\n"
        "- Give concrete, bounded next steps when appropriate.\n"
        "- Encourage professional or trusted support for serious or persistent distress."
    )


def build_prompt(tokenizer, ex: Dict, prompt_style: str = "professor") -> str:
    system = (
        "You are a careful mental-health response refiner. "
        "Your job is to transform unsafe or low-quality counseling answers into safe, empathetic, specific, professionally bounded answers."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_content(ex, prompt_style=prompt_style)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {build_user_content(ex, prompt_style=prompt_style)}\n\nAssistant:"


def resolve_target_modules(model, requested: str) -> List[str]:
    """Resolve LoRA targets only on the text/language generation path.

    Gemma-4-E4B-it exposes audio_tower and vision_tower modules with the same
    q_proj/k_proj/v_proj/o_proj names. Suffix-only PEFT targeting therefore
    attaches LoRA to multimodal towers. For a text-only response refiner, we
    explicitly exclude audio/vision/encoder modules and return full module names.
    """
    requested_modules = [x.strip() for x in requested.split(",") if x.strip()]
    bad_parts = (
        "audio_tower",
        "vision_tower",
        "visual",
        "image",
        "multimodal",
        "multi_modal",
        "encoder",
    )

    names = [name for name, _ in model.named_modules()]
    resolved = []

    for name in names:
        low = name.lower()
        if any(bad in low for bad in bad_parts):
            continue

        for module_name in requested_modules:
            if name.endswith(f"{module_name}.linear") or name.endswith(module_name):
                resolved.append(name)
                break

    # preserve order, remove duplicates
    dedup = []
    for item in resolved:
        if item not in dedup:
            dedup.append(item)

    if not dedup:
        print("[debug] available q/k/v/o-like modules excluding bad_parts:")
        for name in names:
            low = name.lower()
            if any(bad in low for bad in bad_parts):
                continue
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                print(" ", name)
        raise ValueError(
            "No text-side LoRA target modules found. Inspect model.named_modules()."
        )

    print("requested target modules:", requested_modules)
    print("resolved TEXT target modules count:", len(dedup))
    for item in dedup[:30]:
        print("  ", item)
    if len(dedup) > 30:
        print("  ...")
    return dedup


class ProfessorRefinerDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_source_len: int, max_target_len: int, prompt_style: str = "professor"):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        ex = self.rows[idx]
        prompt = build_prompt(self.tokenizer, ex, prompt_style=self.prompt_style)
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

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class CausalCollator:
    tokenizer: object
    max_len: int

    def __call__(self, batch):
        max_len = min(self.max_len, max(len(x["input_ids"]) for x in batch))
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            ids = item["input_ids"][:max_len]
            labs = item["labels"][:max_len]
            mask = item["attention_mask"][:max_len]
            n_pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * n_pad)
            labels.append(labs + [-100] * n_pad)
            attention_mask.append(mask + [0] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def build_training_args(args):
    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=True,
    )
    try:
        return TrainingArguments(eval_strategy="steps", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="steps", **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--max_source_len", type=int, default=896)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--prompt_style",
        choices=["professor", "sft_plain"],
        default="professor",
        help="professor keeps aspect/draft fields; sft_plain uses only question + unsafe_response for a fair SFT baseline.",
    )
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = resolve_target_modules(model, args.target_modules)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = ProfessorRefinerDataset(
        args.train_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        prompt_style=args.prompt_style,
    )
    valid_ds = ProfessorRefinerDataset(
        args.valid_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
        prompt_style=args.prompt_style,
    )
    collator = CausalCollator(tokenizer, args.max_source_len + args.max_target_len)

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(Path(args.output_dir) / "final")
    tokenizer.save_pretrained(Path(args.output_dir) / "final")
    with open(Path(args.output_dir) / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
