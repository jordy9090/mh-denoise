import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import train_professor_peft_refiner as base


NON_TEXT_MARKERS = ["vision_tower", "visual", "audio_tower", "audio", "image"]


def resolve_text_only_target_modules(model, requested: str, output_dir: str):
    requested_modules = [x.strip() for x in requested.split(",") if x.strip()]
    text_hits = []
    non_text_hits = []
    all_projection_hits = []

    for name, _ in model.named_modules():
        if not any(name.endswith(f"self_attn.{proj}") for proj in requested_modules):
            continue

        all_projection_hits.append(name)
        low = name.lower()

        if any(marker in low for marker in NON_TEXT_MARKERS):
            non_text_hits.append(name)
            continue

        if name.startswith("model.language_model.layers."):
            text_hits.append(name)

    text_hits = list(dict.fromkeys(text_hits))

    print("requested target modules:", requested_modules)
    print("all projection hits:", len(all_projection_hits))
    print("text-only target modules:", len(text_hits))
    print("excluded non-text projection hits:", len(non_text_hits))

    print("first text-only targets:")
    for x in text_hits[:20]:
        print("  ", x)

    if non_text_hits:
        print("first excluded non-text targets:")
        for x in non_text_hits[:20]:
            print("  ", x)

    if not text_hits:
        raise RuntimeError("No text decoder LoRA targets found under model.language_model.layers.*")

    if any(any(marker in x.lower() for marker in NON_TEXT_MARKERS) for x in text_hits):
        raise RuntimeError("Non-text module leaked into LoRA target list")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "lora_target_modules.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "requested": requested_modules,
                "n_all_projection_hits": len(all_projection_hits),
                "n_text_targets": len(text_hits),
                "n_excluded_non_text_hits": len(non_text_hits),
                "text_targets": text_hits,
                "excluded_non_text_examples": non_text_hits[:80],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return text_hits


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

    target_modules = resolve_text_only_target_modules(model, args.target_modules, args.output_dir)

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

    train_ds = base.ProfessorRefinerDataset(
        args.train_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
    )
    valid_ds = base.ProfessorRefinerDataset(
        args.valid_file,
        tokenizer,
        args.max_source_len,
        args.max_target_len,
    )
    collator = base.CausalCollator(tokenizer, args.max_source_len + args.max_target_len)

    trainer = Trainer(
        model=model,
        args=base.build_training_args(args),
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
