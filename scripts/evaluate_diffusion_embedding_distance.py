import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode(tokenizer, text, max_len, device):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].float().to(device)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/refinement/semantic_diffusion_denoiser_v0_test.jsonl")
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--output", default="outputs/analysis/diffusion_embedding_distance.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base.eval()
    embed = base.get_input_embeddings()
    device = next(embed.parameters()).device

    rows = load_jsonl(args.input)

    unsafe_cos = []
    diffusion_cos = []

    with torch.no_grad():
        for ex in tqdm(rows):
            safe_ids, safe_mask = encode(tokenizer, ex["safe_response"], args.max_len, device)
            unsafe_ids, unsafe_mask = encode(tokenizer, ex["unsafe_response"], args.max_len, device)
            diff_ids, diff_mask = encode(tokenizer, ex.get("diffusion_response", ""), args.max_len, device)

            safe_vec = masked_mean(embed(safe_ids), safe_mask)
            unsafe_vec = masked_mean(embed(unsafe_ids), unsafe_mask)
            diff_vec = masked_mean(embed(diff_ids), diff_mask)

            unsafe_cos.append(F.cosine_similarity(unsafe_vec.float(), safe_vec.float()).item())
            diffusion_cos.append(F.cosine_similarity(diff_vec.float(), safe_vec.float()).item())

    result = {
        "n": len(rows),
        "unsafe_to_safe_cosine_mean": sum(unsafe_cos) / len(unsafe_cos),
        "diffusion_to_safe_cosine_mean": sum(diffusion_cos) / len(diffusion_cos),
        "improvement": (sum(diffusion_cos) / len(diffusion_cos)) - (sum(unsafe_cos) / len(unsafe_cos)),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
