import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_aspect_signal(ex):
    g = ex.get("g")
    if isinstance(g, list) and len(g) == len(DIMS):
        pairs = sorted(zip(DIMS, g), key=lambda x: float(x[1]), reverse=True)
        top = [f"{k}={float(v):.2f}" for k, v in pairs if float(v) >= 0.10]
        if not top:
            top = [f"{pairs[0][0]}={float(pairs[0][1]):.2f}"]
        return ", ".join(top)
    for key in ["condition_dim", "target_dimension", "used_dimension", "predicted_dimension"]:
        if ex.get(key):
            return str(ex[key])
    return "unspecified"


def build_prompt(ex):
    q = str(ex.get("question", ex.get("query", ""))).strip()
    u = str(ex.get("unsafe_response", ex.get("corrupted_response", ex.get("bad_response", "")))).strip()
    z = str(ex.get("z_t", ex.get("draft", ""))).strip() or "No draft."
    source = str(ex.get("source", ex.get("corruption_type", "unknown"))).strip()
    t = ex.get("t", ex.get("corruption_level", ""))
    aspect = format_aspect_signal(ex)
    return (
        "Rewrite the given response into a safe, helpful final response.\n"
        "Return only the final response. Do not include labels, notes, analysis, or metadata.\n\n"
        f"Question:\n{q}\n\n"
        f"Input response:\n{u}\n\n"
        f"Aspect signal:\n{aspect}\n\n"
        f"Draft:\n{z}\n\n"
        f"Setting:\nsource={source}, t={t}\n\n"
        "Final response:\n"
    )


def cleanup(text):
    text = text.strip()
    for marker in ["Final response:", "Safe response:", "Response:", "Analysis:", "Notes:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@torch.no_grad()
def generate_one(model, tok, prompt, max_source_len, max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_source_len).to(model.device)
    kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if temperature and temperature > 0:
        kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        kwargs.update(dict(do_sample=False))
    gen = model.generate(**kwargs)
    new_tokens = gen[0][enc["input_ids"].shape[-1]:]
    return cleanup(tok.decode(new_tokens, skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    outs = []
    for ex in tqdm(rows):
        prompt = build_prompt(ex)
        pred = generate_one(
            model,
            tok,
            prompt,
            args.max_source_len,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
        )
        out = dict(ex)
        out["peft_denoiser_response"] = pred
        out["method"] = "peft_lora_infermatch_denoiser"
        out["adapter_dir"] = args.adapter_dir
        outs.append(out)

    write_jsonl(outs, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
