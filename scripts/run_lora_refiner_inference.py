import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_user_prompt(ex):
    return f"""
You are a mental-health response refiner.

Task:
Rewrite the corrupted response into a safer, more empathetic, more specific, and clinically bounded support response.

Requirements:
- Be empathetic and specific.
- Stay clinically bounded.
- Do not diagnose the user.
- Do not give medication dosage or medical instructions.
- Remove judgmental, dismissive, unsupported, or unsafe content.
- Preserve useful context from the user's question.
- Keep the response natural and helpful.

User question:
{ex["question"]}

Corrupted response:
{ex["unsafe_response"]}

Violation dimension:
{ex["target_dimension"]}

Safe response:
""".strip()


def generate(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=450)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    rows = list(load_jsonl(args.input))
    print(f"Loaded rows: {len(rows)}")

    with open(args.output, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            prompt = build_user_prompt(ex)
            refined = generate(model, tokenizer, prompt, args.max_new_tokens)

            record = {
                **ex,
                "refiner_response": refined,
                "method": "lora_supervised_refiner",
                "adapter_dir": args.adapter_dir,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
