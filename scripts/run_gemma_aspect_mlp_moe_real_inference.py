import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from aspect_residual_mlp_moe import (
    inject_aspect_residual_mlp_moe,
    load_mlp_moe_adapter,
    load_mlp_moe_config,
    set_mlp_moe_gates,
)
from run_gemma_aspect_moe_real_inference import (
    DIMS,
    build_prompt,
    cleanup,
    get_field,
    make_zt,
    make_zt_risk_tag,
    make_zt_staged,
    make_zt_staged_risk,
    parse_modes,
    predict_g,
    read_jsonl,
    require_checkpoint_dir,
    score_spans,
    split_sentences,
    write_jsonl,
)


def load_base_model(model_name, use_4bit):
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
            device_map={"": 0},
            trust_remote_code=True,
        )
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs.update({"torch_dtype": torch.bfloat16, "device_map": {"": 0}})
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--modes", default="empty,unsafe_t2,unsafe_t3,unsafe_t4")
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--zt_strategy", choices=["threshold", "staged", "staged_risk", "risk_tag"], default="threshold")
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--t2_frac", type=float, default=0.33)
    ap.add_argument("--t3_frac", type=float, default=0.66)
    ap.add_argument("--risk_tag_format", default="[Risk: {dim}] {span} [/Risk]")
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.adapter_dir = require_checkpoint_dir(args.adapter_dir, "--adapter_dir", ("mlp_moe_config.json", "mlp_moe_adapter.pt"))
    args.router_dir = require_checkpoint_dir(args.router_dir, "--router_dir", ("config.json",))
    args.risk_scorer_dir = require_checkpoint_dir(args.risk_scorer_dir, "--risk_scorer_dir", ("config.json",))
    print("adapter_dir:", args.adapter_dir)
    print("router_dir:", args.router_dir)
    print("risk_scorer_dir:", args.risk_scorer_dir)

    router_tok = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir).to(device).eval()

    risk_tok = AutoTokenizer.from_pretrained(args.risk_scorer_dir)
    risk_model = AutoModelForSequenceClassification.from_pretrained(args.risk_scorer_dir).to(device).eval()

    config = load_mlp_moe_config(args.adapter_dir)
    if config.get("moe_impl") != "residual_mlp_moe":
        raise ValueError(f"Expected residual_mlp_moe config, got {config.get('moe_impl')!r}")

    base_model = args.base_model or config.get("base_model")
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("load_in_4bit:", use_4bit)
    model = load_base_model(base_model, use_4bit)
    injected_layers = inject_aspect_residual_mlp_moe(
        model,
        num_experts=int(config.get("num_experts", len(DIMS))),
        bottleneck_size=int(config.get("bottleneck_size", 64)),
        dropout=float(config.get("dropout", 0.05)),
        residual_scale=float(config.get("residual_scale", 0.1)),
        layers=config.get("layers", "last_8"),
        use_shared=bool(config.get("use_shared", True)),
        activation=config.get("activation", "silu"),
        zero_init=bool(config.get("zero_init", True)),
    )
    print("moe_impl: residual_mlp_moe")
    print("injected layers:", injected_layers)
    load_mlp_moe_adapter(model, args.adapter_dir)
    model.eval()
    model_device = next(model.parameters()).device

    rows = read_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    modes = parse_modes(args.modes)
    outs = []

    for ex in tqdm(rows):
        q = get_field(ex, "question", "query", "user_question")
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        g = predict_g(router, router_tok, ex, device)

        spans = split_sentences(u) or [u]
        risk_vecs = score_spans(risk_model, risk_tok, q, spans, device)

        for source, t in modes:
            if source == "empty":
                z, infos = "", []
            elif args.zt_strategy == "staged":
                z, infos = make_zt_staged(u, g, risk_vecs, t, args.T, mask_token=args.mask_token)
            elif args.zt_strategy == "staged_risk":
                z, infos = make_zt_staged_risk(
                    u,
                    g,
                    risk_vecs,
                    t,
                    args.T,
                    mask_token=args.mask_token,
                    risk_threshold=args.risk_threshold,
                    t2_frac=args.t2_frac,
                    t3_frac=args.t3_frac,
                )
            elif args.zt_strategy == "risk_tag":
                z, infos = make_zt_risk_tag(
                    u,
                    g,
                    risk_vecs,
                    t,
                    args.T,
                    risk_threshold=args.risk_threshold,
                    t2_frac=args.t2_frac,
                    t3_frac=args.t3_frac,
                    risk_tag_format=args.risk_tag_format,
                )
            else:
                z, infos = make_zt(
                    u,
                    g,
                    risk_vecs,
                    t,
                    args.T,
                    mask_token=args.mask_token,
                    risk_threshold=args.risk_threshold,
                )

            prompt = build_prompt(q, u, z, g, source, t)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(model_device)
            gates = torch.tensor([g], dtype=torch.float32, device=model_device)
            set_mlp_moe_gates(model, gates)

            gen_kwargs = dict(
                **enc,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            if args.temperature > 0:
                gen_kwargs.update(dict(do_sample=True, temperature=args.temperature))
            else:
                gen_kwargs.update(dict(do_sample=False))

            with torch.no_grad():
                gen = model.generate(**gen_kwargs)

            new_tokens = gen[0][enc["input_ids"].shape[-1] :]
            raw = tok.decode(new_tokens, skip_special_tokens=True)

            out = dict(ex)
            out["mode"] = "empty" if source == "empty" else f"unsafe_t{t}"
            out["g"] = {DIMS[i]: float(g[i]) for i in range(len(DIMS))}
            out["z_t"] = z
            out["zt_strategy"] = args.zt_strategy
            out["span_risks"] = infos
            out["moe_impl"] = "residual_mlp_moe"
            out["moe_response_raw"] = raw
            out["moe_response"] = cleanup(raw)
            outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
