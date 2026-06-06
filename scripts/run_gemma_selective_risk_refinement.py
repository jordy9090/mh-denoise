import argparse
import statistics

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from selective_risk_refinement_utils import (
    DIMS,
    build_risk_tune_prompt,
    build_sft_prompt,
    canonical_example,
    clean_text,
    count_bad_safety_patterns,
    count_generic_phrases,
    generate_response,
    get_field,
    keyword_overlap_ratio,
    make_zt_from_response,
    read_jsonl,
    score_candidate,
    word_count,
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


def load_classifier(path: str, device):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def mean(values):
    values = [float(x) for x in values if x is not None]
    return statistics.mean(values) if values else 0.0


def focus_risk_from_score(score, focus_idx):
    g = score.get("g") or []
    risk_vecs = score.get("risk_vecs") or []
    if focus_idx < 0:
        return 0.0
    g_value = float(g[focus_idx]) if focus_idx < len(g) else 0.0
    best = 0.0
    for rv in risk_vecs:
        rv_value = float(rv[focus_idx]) if focus_idx < len(rv) else 0.0
        best = max(best, g_value * rv_value)
    return float(best)


def should_call_denoiser(sft_metrics, args):
    if not clean_text(sft_metrics["response"]):
        return True, "empty_sft_response"
    if sft_metrics["word_count"] < args.min_word_count:
        return True, "short_sft_response"
    if sft_metrics["bad_safety_count"] > 0:
        return True, "bad_safety_pattern"
    overall_trigger = sft_metrics["risk_score"] > args.gate_risk_threshold
    focus_trigger = sft_metrics["focus_risk_score"] > args.gate_focus_threshold
    if args.gate_strategy == "overall" and overall_trigger:
        return True, "risk_above_threshold"
    if args.gate_strategy == "aspect_only" and focus_trigger:
        return True, "focus_risk_above_threshold"
    if args.gate_strategy == "aspect_or_overall":
        if focus_trigger:
            return True, "focus_risk_above_threshold"
        if overall_trigger:
            return True, "risk_above_threshold"
    return False, "sft_safe"


def acceptance_reasons(sft_metrics, den_metrics, args):
    reasons = []
    if not clean_text(den_metrics["response"]):
        reasons.append("empty_denoiser_response")
    if den_metrics["word_count"] < args.min_word_count:
        reasons.append("short_denoiser_response")
    if den_metrics["risk_score"] > sft_metrics["risk_score"] - args.min_risk_delta:
        reasons.append("risk_not_improved")
    if args.gate_strategy != "overall":
        if den_metrics["focus_risk_score"] > sft_metrics["focus_risk_score"] - args.min_focus_risk_delta:
            reasons.append("focus_risk_not_improved")
    if den_metrics["bad_safety_count"] > sft_metrics["bad_safety_count"]:
        reasons.append("bad_safety_increased")
    if den_metrics["specificity_ratio"] < args.specificity_min_ratio:
        reasons.append("specificity_ratio_low")
    if den_metrics["generic_count"] > sft_metrics["generic_count"]:
        reasons.append("genericity_increased")
    min_overlap = max(0.0, sft_metrics["keyword_overlap"] - args.keyword_overlap_slack)
    if den_metrics["keyword_overlap"] < min_overlap:
        reasons.append("question_keyword_overlap_low")
    return reasons


def response_metrics(q, response, score, focus_idx, sft_word_count=None):
    words = word_count(response)
    return {
        "response": response,
        "risk_score": float(score["risk_score"]),
        "focus_risk_score": focus_risk_from_score(score, focus_idx),
        "g": score["g"],
        "span_risks": score["span_risks"],
        "word_count": words,
        "bad_safety_count": count_bad_safety_patterns(response),
        "generic_count": count_generic_phrases(response),
        "keyword_overlap": keyword_overlap_ratio(q, response),
        "specificity_ratio": words / max(1, sft_word_count if sft_word_count is not None else words),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--sft_adapter_dir", required=True)
    ap.add_argument("--risk_adapter_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--reuse_sft_response", action="store_true")
    ap.add_argument("--sft_response_field", default="sft_response")
    ap.add_argument(
        "--sft_prompt_style",
        choices=["sft_plain"],
        default="sft_plain",
        help="Prompt style used by the first-stage SFT adapter.",
    )
    ap.add_argument("--zt_strategy", choices=["threshold", "staged", "staged_risk", "risk_tag"], default="staged_risk")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--timestep", type=int, default=3)
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--gate_risk_threshold", type=float, default=0.35)
    ap.add_argument("--gate_strategy", choices=["overall", "aspect_only", "aspect_or_overall"], default="overall")
    ap.add_argument("--gate_focus_aspect", choices=DIMS, default="medical_advice")
    ap.add_argument("--gate_focus_threshold", type=float, default=0.15)
    ap.add_argument("--mask_threshold", type=float, default=0.35)
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--t2_frac", type=float, default=0.33)
    ap.add_argument("--t3_frac", type=float, default=0.66)
    ap.add_argument("--risk_tag_format", default="[Risk: {dim}] {span} [/Risk]")
    ap.add_argument("--min_risk_delta", type=float, default=0.01)
    ap.add_argument("--min_focus_risk_delta", type=float, default=0.0)
    ap.add_argument("--specificity_min_ratio", type=float, default=0.60)
    ap.add_argument("--keyword_overlap_slack", type=float, default=0.15)
    ap.add_argument("--min_word_count", type=int, default=20)
    ap.add_argument("--router_max_len", type=int, default=512)
    ap.add_argument("--risk_max_len", type=int, default=384)
    ap.add_argument("--max_source_len", type=int, default=896)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    focus_idx = DIMS.index(args.gate_focus_aspect)
    router_tok, router = load_classifier(args.router_dir, device)
    risk_tok, risk_model = load_classifier(args.risk_scorer_dir, device)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("sft_adapter_dir:", args.sft_adapter_dir)
    print("risk_adapter_dir:", args.risk_adapter_dir)
    print("router_dir:", args.router_dir)
    print("risk_scorer_dir:", args.risk_scorer_dir)
    print("sft_prompt_style:", args.sft_prompt_style)
    print("gate_strategy:", args.gate_strategy)
    print("gate_focus_aspect:", args.gate_focus_aspect)
    print("gate_focus_threshold:", args.gate_focus_threshold)
    print("min_focus_risk_delta:", args.min_focus_risk_delta)
    print("load_in_4bit:", use_4bit)
    base = load_base_model(args.base_model, use_4bit)
    model = PeftModel.from_pretrained(base, args.sft_adapter_dir, adapter_name="sft")
    model.load_adapter(args.risk_adapter_dir, adapter_name="risk")
    model.eval()

    rows = read_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    outs = []
    for raw in tqdm(rows):
        row = canonical_example(raw)
        q = row["question"]

        sft_raw = ""
        existing_sft = clean_text(get_field(row, args.sft_response_field, "professor_peft_response", "peft_response"))
        if args.reuse_sft_response and existing_sft:
            sft_response = existing_sft
        else:
            model.set_adapter("sft")
            sft_raw, sft_response = generate_response(
                model,
                tokenizer,
                build_sft_prompt(tokenizer, row),
                max_source_len=args.max_source_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

        sft_score = score_candidate(
            q,
            sft_response,
            router,
            router_tok,
            risk_model,
            risk_tok,
            device,
            router_max_len=args.router_max_len,
            risk_max_len=args.risk_max_len,
        )
        sft_metrics = response_metrics(q, sft_response, sft_score, focus_idx)
        call_denoiser, call_reason = should_call_denoiser(sft_metrics, args)

        z_t = ""
        z_t_infos = []
        denoiser_raw = ""
        denoiser_response = ""
        den_metrics = None
        accepted = False
        reject_reason = None
        final_response = sft_response
        final_risk_score = sft_metrics["risk_score"]

        if call_denoiser:
            z_t, z_t_infos = make_zt_from_response(
                sft_response,
                sft_metrics["g"],
                sft_score["risk_vecs"],
                strategy=args.zt_strategy,
                t=args.timestep,
                T=args.T,
                mask_token=args.mask_token,
                risk_threshold=args.risk_threshold,
                mask_threshold=args.mask_threshold,
                t2_frac=args.t2_frac,
                t3_frac=args.t3_frac,
                risk_tag_format=args.risk_tag_format,
            )
            risk_row = dict(row)
            risk_row["sft_response"] = sft_response
            risk_row["z_t_from_sft"] = z_t
            risk_row["g_sft"] = sft_metrics["g"]
            model.set_adapter("risk")
            denoiser_raw, denoiser_response = generate_response(
                model,
                tokenizer,
                build_risk_tune_prompt(tokenizer, risk_row),
                max_source_len=args.max_source_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            den_score = score_candidate(
                q,
                denoiser_response,
                router,
                router_tok,
                risk_model,
                risk_tok,
                device,
                router_max_len=args.router_max_len,
                risk_max_len=args.risk_max_len,
            )
            den_metrics = response_metrics(
                q,
                denoiser_response,
                den_score,
                focus_idx,
                sft_word_count=sft_metrics["word_count"],
            )
            reasons = acceptance_reasons(sft_metrics, den_metrics, args)
            if not reasons:
                accepted = True
                final_response = denoiser_response
                final_risk_score = den_metrics["risk_score"]
            else:
                reject_reason = ",".join(reasons)

        out = dict(row)
        out.update(
            {
                "sft_response_raw": sft_raw,
                "sft_response": sft_response,
                "z_t_from_sft": z_t,
                "z_t_from_sft_infos": z_t_infos,
                "denoiser_response_raw": denoiser_raw,
                "denoiser_response": denoiser_response,
                "final_response": final_response,
                "used_denoiser": bool(call_denoiser),
                "denoiser_call_reason": call_reason,
                "accepted_denoiser": bool(accepted),
                "reject_reason": reject_reason,
                "sft_risk_score": sft_metrics["risk_score"],
                "denoiser_risk_score": den_metrics["risk_score"] if den_metrics else None,
                "final_risk_score": final_risk_score,
                "sft_focus_risk_score": sft_metrics["focus_risk_score"],
                "denoiser_focus_risk_score": den_metrics["focus_risk_score"] if den_metrics else None,
                "final_focus_risk_score": (
                    den_metrics["focus_risk_score"] if accepted and den_metrics else sft_metrics["focus_risk_score"]
                ),
                "specificity_ratio": den_metrics["specificity_ratio"] if den_metrics else 1.0,
                "sft_bad_safety_count": sft_metrics["bad_safety_count"],
                "denoiser_bad_safety_count": den_metrics["bad_safety_count"] if den_metrics else None,
                "sft_generic_count": sft_metrics["generic_count"],
                "denoiser_generic_count": den_metrics["generic_count"] if den_metrics else None,
                "sft_keyword_overlap": sft_metrics["keyword_overlap"],
                "denoiser_keyword_overlap": den_metrics["keyword_overlap"] if den_metrics else None,
                "g_sft": sft_metrics["g"],
                "g_denoiser": den_metrics["g"] if den_metrics else [],
                "span_risks_sft": sft_metrics["span_risks"],
                "span_risks_denoiser": den_metrics["span_risks"] if den_metrics else [],
                "zt_strategy": args.zt_strategy,
                "sft_prompt_style": args.sft_prompt_style,
                "gate_strategy": args.gate_strategy,
                "gate_focus_aspect": args.gate_focus_aspect,
                "gate_focus_threshold": args.gate_focus_threshold,
                "min_focus_risk_delta": args.min_focus_risk_delta,
                "method": "selective_risk_aware_refinement",
            }
        )
        outs.append(out)

    write_jsonl(outs, args.output)
    used = [r for r in outs if r["used_denoiser"]]
    accepted = [r for r in outs if r["accepted_denoiser"]]
    rejected = [r for r in outs if r["used_denoiser"] and not r["accepted_denoiser"]]
    print("saved to", args.output)
    print("total examples:", len(outs))
    print("denoiser called:", len(used))
    print("denoiser accepted:", len(accepted))
    print("denoiser rejected:", len(rejected))
    print("average SFT risk:", round(mean(r["sft_risk_score"] for r in outs), 4))
    print("average final risk:", round(mean(r["final_risk_score"] for r in outs), 4))
    print("average SFT focus risk:", round(mean(r["sft_focus_risk_score"] for r in outs), 4))
    print("average final focus risk:", round(mean(r["final_focus_risk_score"] for r in outs), 4))
    print("average length ratio:", round(mean(r["specificity_ratio"] for r in used), 4) if used else 1.0)
    print("bad phrase count before:", sum(int(r["sft_bad_safety_count"]) for r in outs))
    print("bad phrase count after:", sum(count_bad_safety_patterns(r["final_response"]) for r in outs))
    print("generic phrase count before:", sum(int(r["sft_generic_count"]) for r in outs))
    print("generic phrase count after:", sum(count_generic_phrases(r["final_response"]) for r in outs))


if __name__ == "__main__":
    main()
