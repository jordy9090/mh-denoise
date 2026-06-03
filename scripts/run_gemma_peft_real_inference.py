import argparse, json, re, math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel


DIMS = ["overall_quality","empathy","specificity","medical_advice","factual_consistency","toxicity"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_field(ex, *names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def router_text(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "Question:\n" + q.strip()
        + "\n\nUnsafe response:\n" + u.strip()
        + "\n\nTask: identify all violated mental-health response quality dimensions."
    )


@torch.no_grad()
def predict_g(router, tok, ex, device):
    enc = tok(router_text(ex), return_tensors="pt", truncation=True, max_length=512).to(device)
    probs = torch.sigmoid(router(**enc).logits.float())[0].detach().cpu().tolist()
    if sum(probs) <= 1e-6:
        probs = [1 / len(DIMS)] * len(DIMS)
    return probs


def risk_text(q, span):
    return (
        "Question:\n" + q.strip()
        + "\n\nCandidate span:\n" + span.strip()
        + "\n\nTask: predict which counseling quality dimensions this span may violate."
    )


@torch.no_grad()
def score_spans(model, tok, q, spans, device):
    if not spans:
        return []
    enc = tok([risk_text(q, s) for s in spans], return_tensors="pt", padding=True, truncation=True, max_length=384).to(device)
    return torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()


def rl(g, risk_vec):
    return max(float(g[k]) * float(risk_vec[k]) for k in range(len(DIMS)))


def top_risk_dim(g, risk_vec):
    scores = []
    for k in range(len(DIMS)):
        rv = float(risk_vec[k]) if k < len(risk_vec) else 0.0
        scores.append(float(g[k]) * rv)
    if not scores:
        return 0.0, DIMS[0]
    best = max(range(len(scores)), key=lambda k: scores[k])
    return scores[best], DIMS[best]


def staged_mask_count(n_candidates, t, t2_frac, t3_frac):
    if t <= 1 or n_candidates == 0:
        return 0
    if t == 2:
        mask_count = math.ceil(float(t2_frac) * n_candidates)
    elif t == 3:
        mask_count = math.ceil(float(t3_frac) * n_candidates)
    else:
        mask_count = n_candidates
    return max(0, min(n_candidates, mask_count))


def make_zt(unsafe, g, risk_vecs, t, T, mask_token="<MASK>", rho=0.15, lambda_mask=0.75, risk_threshold=0.35, mask_threshold=0.35):
    spans = split_sentences(unsafe) or [unsafe.strip()]
    beta = t / float(T)
    parts, infos = [], []

    for span, rv in zip(spans, risk_vecs):
        r = rl(g, rv)
        pi = max(0.0, min(1.0, rho + lambda_mask * beta * r))
        p_mask = beta * pi
        should_mask = (p_mask >= mask_threshold) or (r >= risk_threshold and t >= 2)

        if should_mask:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(span)
            state = "UNSAFE"

        infos.append({"span": span, "r_l_g": r, "p_mask": p_mask, "state": state})

    return " ".join(parts), infos


def make_zt_staged(unsafe, g, risk_vecs, t, T, mask_token="<MASK>"):
    spans = split_sentences(unsafe) or [unsafe.strip()]
    n = len(spans)
    risks = []
    for i, span in enumerate(spans):
        risk_vec = risk_vecs[i] if i < len(risk_vecs) else [0.0] * len(DIMS)
        risks.append((i, span, rl(g, risk_vec)))

    ranked = sorted(risks, key=lambda x: x[2], reverse=True)
    ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(ranked)}

    if t <= 1 or n == 0:
        mask_count = 0
    elif t == 2:
        mask_count = math.ceil(0.33 * n)
    elif t == 3:
        mask_count = math.ceil(0.66 * n)
    else:
        mask_count = n
    if n > 0 and t >= 2:
        mask_count = max(1, mask_count)
    mask_count = min(n, mask_count)

    mask_indices = {idx for idx, _, _ in ranked[:mask_count]}
    risk_by_idx = {idx: risk for idx, _, risk in risks}

    parts, infos = [], []
    for i, span in enumerate(spans):
        if i in mask_indices:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(span)
            state = "UNSAFE"
        infos.append({
            "span": span,
            "r_l_g": risk_by_idx[i],
            "rank": ranks[i],
            "state": state,
            "strategy": "staged",
        })

    return " ".join(parts), infos


def make_zt_staged_risk(unsafe, g, risk_vecs, t, T, mask_token="<MASK>", risk_threshold=0.35, t2_frac=0.33, t3_frac=0.66):
    spans = split_sentences(unsafe) or [unsafe.strip()]
    risks = []
    candidates = []
    for i, span in enumerate(spans):
        risk_vec = risk_vecs[i] if i < len(risk_vecs) else [0.0] * len(DIMS)
        risk, top_dim = top_risk_dim(g, risk_vec)
        item = {"idx": i, "span": span, "risk": risk, "top_dim": top_dim}
        risks.append(item)
        if risk >= risk_threshold:
            candidates.append(item)

    ranked = sorted(candidates, key=lambda x: x["risk"], reverse=True)
    ranks = {item["idx"]: rank + 1 for rank, item in enumerate(ranked)}
    mask_count = staged_mask_count(len(ranked), t, t2_frac, t3_frac)
    mask_indices = {item["idx"] for item in ranked[:mask_count]}

    parts, infos = [], []
    for item in risks:
        if item["idx"] in mask_indices:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(item["span"])
            state = "UNSAFE"
        infos.append({
            "span": item["span"],
            "r_l_g": item["risk"],
            "top_dim": item["top_dim"],
            "rank": ranks.get(item["idx"]),
            "candidate": item["idx"] in ranks,
            "state": state,
            "strategy": "staged_risk",
        })

    return " ".join(parts), infos


def make_zt_risk_tag(unsafe, g, risk_vecs, t, T, risk_threshold=0.35, t2_frac=0.33, t3_frac=0.66, risk_tag_format="[Risk: {dim}] {span} [/Risk]"):
    spans = split_sentences(unsafe) or [unsafe.strip()]
    risks = []
    candidates = []
    for i, span in enumerate(spans):
        risk_vec = risk_vecs[i] if i < len(risk_vecs) else [0.0] * len(DIMS)
        risk, top_dim = top_risk_dim(g, risk_vec)
        item = {"idx": i, "span": span, "risk": risk, "top_dim": top_dim}
        risks.append(item)
        if risk >= risk_threshold:
            candidates.append(item)

    ranked = sorted(candidates, key=lambda x: x["risk"], reverse=True)
    ranks = {item["idx"]: rank + 1 for rank, item in enumerate(ranked)}
    mark_count = staged_mask_count(len(ranked), t, t2_frac, t3_frac)
    mark_indices = {item["idx"] for item in ranked[:mark_count]}

    parts, infos = [], []
    for item in risks:
        if item["idx"] in mark_indices:
            parts.append(risk_tag_format.format(dim=item["top_dim"], span=item["span"], risk=item["risk"]))
            state = "MARKED"
        else:
            parts.append(item["span"])
            state = "UNSAFE"
        infos.append({
            "span": item["span"],
            "r_l_g": item["risk"],
            "top_dim": item["top_dim"],
            "rank": ranks.get(item["idx"]),
            "candidate": item["idx"] in ranks,
            "state": state,
            "strategy": "risk_tag",
        })

    return " ".join(parts), infos


def build_prompt(q, u, z, g, source, t):
    if not z:
        z = "No draft. Rewrite directly from the unsafe response."

    g_text = ", ".join(f"{DIMS[i]}={float(g[i]):.2f}" for i in range(len(DIMS)))
    risk_tag_instruction = ""
    if "[Risk:" in z or "<RISK" in z:
        risk_tag_instruction = (
            "Spans marked with [Risk: ...] or <RISK ...> are likely unsafe or low-quality. "
            "Revise those spans while preserving useful context from the draft.\n\n"
        )

    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n"
        "Do not copy blaming, diagnostic, toxic, or unsupported wording from the unsafe response.\n\n"
        f"Aspect scores:\n{g_text}\n\n"
        f"Question:\n{q.strip()}\n\n"
        f"Unsafe response to fix:\n{u.strip()}\n\n"
        f"{risk_tag_instruction}"
        f"Draft to revise:\n{z.strip()}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Safe response:\n"
    )


def cleanup(text):
    text = text.strip()
    markers = [
        "Question:", "Unsafe response:", "Unsafe response to fix:",
        "Draft:", "Draft to revise:", "Corruption:",
        "Response:", "Safe response:", "Analysis:",
        "Thought:", "Reasoning:", "Final answer:", "Final response:"
    ]
    for m in markers:
        if m in text:
            before = text.split(m)[0].strip()
            after = text.split(m)[-1].strip()
            text = before if before else after
    text = text.replace("<MASK>", "").replace("[needs revision]", "").strip()
    text = re.sub(r"\[/?Risk[^\]]*\]", "", text)
    text = re.sub(r"</?RISK[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


def parse_modes(s):
    out = []
    for m in s.split(","):
        m = m.strip()
        if m == "empty":
            out.append(("empty", 0))
        elif m.startswith("unsafe_t"):
            out.append(("unsafe", int(m.replace("unsafe_t", ""))))
    return out


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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    router_tok = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir).to(device).eval()

    risk_tok = AutoTokenizer.from_pretrained(args.risk_scorer_dir)
    risk_model = AutoModelForSequenceClassification.from_pretrained(args.risk_scorer_dir).to(device).eval()

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
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    model_device = next(model.parameters()).device

    rows = read_jsonl(args.input)
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
                z, infos = make_zt(u, g, risk_vecs, t, args.T, mask_token=args.mask_token, risk_threshold=args.risk_threshold)

            prompt = build_prompt(q, u, z, g, source, t)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(model_device)

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

            new_tokens = gen[0][enc["input_ids"].shape[-1]:]
            raw = tok.decode(new_tokens, skip_special_tokens=True)

            out = dict(ex)
            out["mode"] = "empty" if source == "empty" else f"unsafe_t{t}"
            out["g"] = {DIMS[i]: float(g[i]) for i in range(len(DIMS))}
            out["z_t"] = z
            out["zt_strategy"] = args.zt_strategy
            out["span_risks"] = infos
            out["peft_response_raw"] = raw
            out["peft_response"] = cleanup(raw)
            outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
