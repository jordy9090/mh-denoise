import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


RESPONSE_KEYS = [
    "cleaned_response",
    "cleaned_cleaned_response",
    "cleaned_Response",
    "response",
    "safe_response",
    "rewritten_response",
    "final_response",
]


def normalize_text(text):
    text = str(text or "")
    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_jsonish_response(text):
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    key_re = "|".join(re.escape(key) for key in RESPONSE_KEYS)
    match = re.search(rf'["\']?(?:{key_re})["\']?\s*:\s*["\']', cleaned, flags=re.I | re.S)
    if match:
        cleaned = cleaned[match.end():]
    cleaned = re.sub(r'["\']\s*,?\s*["\']?[^"\']*$', "", cleaned).strip()
    cleaned = re.sub(r'["\']\s*\}\s*$', "", cleaned).strip()
    cleaned = re.sub(r"^\s*\{\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*\}\s*$", "", cleaned).strip()
    cleaned = cleaned.replace("\\n", " ")
    cleaned = cleaned.replace('\\"', '"')
    cleaned = cleaned.replace("\\", " ")
    return normalize_text(cleaned)


def extract_json(text):
    raw_text = str(text or "").strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {
            "cleaned_response": normalize_text(raw_text),
            "parse_error": True,
            "raw_output": raw_text,
            "parsed_key": "",
        }

    try:
        obj = json.loads(match.group(0))
        for key in RESPONSE_KEYS:
            value = obj.get(key)
            if value is not None and str(value).strip():
                return {
                    "cleaned_response": normalize_text(value),
                    "parse_error": key != "cleaned_response",
                    "raw_output": raw_text,
                    "parsed_key": key,
                }
        return {
            "cleaned_response": strip_jsonish_response(match.group(0)),
            "parse_error": True,
            "raw_output": raw_text,
            "parsed_key": "",
        }
    except json.JSONDecodeError:
        return {
            "cleaned_response": strip_jsonish_response(raw_text),
            "parse_error": True,
            "raw_output": raw_text,
            "parsed_key": "",
        }


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


def generate(model, tokenizer, prompt, args):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_source_len,
    ).to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
    )
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_json(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--use_dimension", action="store_true")
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
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
            result = generate(model, tokenizer, prompt, args)

            record = {
                **ex,
                "cleaned_response": result.get("cleaned_response", ""),
                "prompt_cleaning_parse_error": result.get("parse_error", False),
                "prompt_cleaning_raw_output": result.get("raw_output", ""),
                "prompt_cleaning_parsed_key": result.get("parsed_key", ""),
                "baseline": "prompt_cleaning",
                "used_dimension": args.use_dimension,
                "decoding": {
                    "max_source_len": args.max_source_len,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
