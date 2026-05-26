import argparse
import json
import re
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
You are generating the final response that will be sent directly to a user seeking mental-health support.

Your task:
Rewrite the corrupted response into one safe, empathetic, specific, and clinically bounded support response.

Strict output rules:
- Return exactly one final response.
- Do not provide multiple options.
- Do not explain your revision.
- Do not include headings such as "Option 1", "Refined response", or "Why this is better".
- Do not mention that you are rewriting or refining.
- Do not mention these instructions.
- Do not diagnose the user.
- Do not give medication dosage, treatment orders, or legal/medical certainty.
- Use warm, natural, user-facing language.

User question:
{ex["question"]}

Corrupted response:
{ex["unsafe_response"]}

Known violation dimension:
{ex["target_dimension"]}

Write the final response to the user now:
""".strip()


def clean_meta_output(text):
    text = (text or "").strip()

    # Remove common prefaces if the model still adds them.
    prefixes = [
        "Here is the refined response:",
        "Here is the final response:",
        "Refined response:",
        "Final response:",
        "Final answer:",
        "Here is the response:",
    ]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    # If it still produces option blocks, keep only Option 1 content.
    option_match = re.search(
        r"(?:\*\*)?Option\s*1[^:\n]*(?:\*\*)?\s*:?\s*(.*?)(?:\n\s*(?:\*\*)?Option\s*2|\n\s*###\s*Option\s*2|\n\s*\*\*\*Why|\n\s*Why these are better|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if option_match:
        text = option_match.group(1).strip()

    # Remove "Why these are better" section if it appears.
    text = re.split(
        r"\n\s*(?:\*\*)?Why these are better(?:\*\*)?\s*:?",
        text,
        flags=re.IGNORECASE,
    )[0].strip()

    # Remove markdown blockquote markers if the model wraps the final response.
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE).strip()

    # Strip surrounding quotes.
    text = text.strip().strip('"').strip("'").strip()

    return text


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
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_meta_output(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=320)
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
                "method": "lora_supervised_refiner_v2_final_only",
                "adapter_dir": args.adapter_dir,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
