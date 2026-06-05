import argparse

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from selective_risk_refinement_utils import (
    build_sft_prompt,
    canonical_example,
    generate_response,
    read_jsonl,
    write_jsonl,
)


def load_base_model(model_name: str, use_4bit: bool):
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--max_source_len", type=int, default=896)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument(
        "--sft_prompt_style",
        choices=["sft_plain", "professor"],
        default="sft_plain",
        help="Prompt style used by the first-stage SFT adapter.",
    )
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("adapter_dir:", args.adapter_dir)
    print("sft_prompt_style:", args.sft_prompt_style)
    print("load_in_4bit:", use_4bit)
    base = load_base_model(args.base_model, use_4bit)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    rows = read_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    outs = []
    for ex in tqdm(rows):
        row = canonical_example(ex)
        prompt = build_sft_prompt(tokenizer, row, prompt_style=args.sft_prompt_style)
        raw, cleaned = generate_response(
            model,
            tokenizer,
            prompt,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        out = dict(row)
        out["sft_response_raw"] = raw
        out["sft_response"] = cleaned
        out["sft_prompt_style"] = args.sft_prompt_style
        out["method"] = "sft_refiner_output_for_risk_tuning"
        outs.append(out)

    write_jsonl(outs, args.output)
    print("rows:", len(outs))
    print("saved to", args.output)


if __name__ == "__main__":
    main()
