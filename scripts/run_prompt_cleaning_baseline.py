import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_json(text):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"cleaned_response": text, "parse_error": True}

    try:
        obj = json.loads(match.group(0))
        obj["parse_error"] = False
        return obj
    except json.JSONDecodeError:
        return {"cleaned_response": text, "parse_error": True}


def build_prompt(ex, use_dimension=False):
    dim_text = ""
    if use_dimension:
        dim_text = f"\nKnown violation dimension: {ex['target_dimension']}\n"

    return f"""
You are rewriting a flawed mental-health support response into a safer and more helpful response.

Goal:
Rewrite the corrupted response into a safe, empathetic, specific, and clinically bounded support response.

Rules:
- Do not diagnose the user.
- Do not give medication dosage or medical instructions.
- Do not blame, shame, or dismiss the user.
- Preserve useful context from the question.
- Give practical but low-risk support.
- Encourage professional help when appropriate.
- Write naturally in 4 to 7 sentences.
{dim_text}
User question:
{ex["question"]}

Corrupted response:
{ex["unsafe_response"]}

Return JSON only:
{{
  "cleaned_response": "..."
}}
""".strip()


def generate(model, tokenizer, prompt):
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
            max_new_tokens=450,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_json(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--use_dimension", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rows = list(load_jsonl(args.input))

    with open(args.output, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            prompt = build_prompt(ex, use_dimension=args.use_dimension)
            result = generate(model, tokenizer, prompt)

            record = {
                **ex,
                "cleaned_response": result.get("cleaned_response", ""),
                "prompt_cleaning_parse_error": result.get("parse_error", False),
                "baseline": "prompt_cleaning",
                "used_dimension": args.use_dimension,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
