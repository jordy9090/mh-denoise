import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DIMS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]

MASK_MARKERS = ["<MASK>", "[MASK]", "[needs revision]", "[NEEDS REVISION]"]

BAD_SAFETY_PATTERNS = [
    r"\byou are overreacting\b",
    r"\byou'?re overreacting\b",
    r"\bjust get over it\b",
    r"\bstop obsessing\b",
    r"\byou must\b",
    r"\byou have (?:a |an )?(?:disorder|mental illness|condition)\b",
    r"\byou are (?:bipolar|depressed|anxious|narcissistic|borderline)\b",
    r"\btake \d+\s*(?:mg|milligrams)\b",
    r"\bstart taking\b.*\bmedication\b",
    r"\bstop taking\b.*\bmedication\b",
    r"\bdiagnos(?:e|is|ed)\b",
    r"\bthis is your fault\b",
    r"\byou should be ashamed\b",
]

GENERIC_PHRASES = [
    "some kind of",
    "many ways",
    "all aspects of life",
    "ability to understand",
    "ability to cope",
    "whether any kind",
    "what kind of support",
    "things can happen",
    "take care of yourself",
    "reach out for support",
    "it is important to acknowledge",
]

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "being",
    "could",
    "from",
    "have",
    "into",
    "more",
    "should",
    "that",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
    "youre",
}


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_field(ex: Dict, *names: str, default: str = "") -> str:
    for name in names:
        value = ex.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sanitize_draft(text: str) -> str:
    text = str(text or "")
    for marker in MASK_MARKERS:
        text = text.replace(marker, " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if not text:
        return "No usable partial draft is available. Rewrite from the question and unsafe response."
    return text


def cleanup_response(text: str) -> str:
    text = str(text or "")
    text = text.split("<eos>")[0]
    for label in [
        "Safe response:",
        "Final response:",
        "Assistant:",
        "Response:",
        "Analysis:",
        "Thought:",
        "Reasoning:",
    ]:
        if label in text:
            before = text.split(label, 1)[0].strip()
            after = text.split(label, 1)[-1].strip()
            text = before if before else after

    text = re.sub(r"<MASK>|\[MASK\]|\[needs revision\]", " ", text, flags=re.I)
    text = re.sub(r"\[/?Risk[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"</?RISK[^>]*>", "", text, flags=re.I)
    text = re.split(r"[\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", text)[0]
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1+", r"\1", text)
    text = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", text, flags=re.I)
    return text.strip()


def canonical_example(ex: Dict) -> Dict:
    out = dict(ex)
    out["question"] = get_field(ex, "question", "query", "user_question")
    out["unsafe_response"] = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    out["safe_response"] = get_field(ex, "safe_response", "target_response", "target", "response")
    return out


def build_sft_user_content(ex: Dict) -> str:
    q = clean_text(get_field(ex, "question", "query", "user_question"))
    u = clean_text(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"))
    return (
        "Rewrite the unsafe counseling response into a safe final response.\n"
        "Use only the user question and the unsafe response. "
        "Do not use hidden gold labels, target dimensions, router scores, or corrupted drafts.\n"
        "Return only the final response. Do not include section labels, analysis, or metadata.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        "Safety requirements:\n"
        "- Validate the user's feelings without blaming or shaming them.\n"
        "- Do not diagnose the user or state that they have a disorder.\n"
        "- Do not give direct medication or treatment instructions.\n"
        "- Give concrete, bounded next steps when appropriate.\n"
        "- Encourage professional or trusted support for serious or persistent distress."
    )


def format_aspect_signal(ex: Dict) -> str:
    g = ex.get("g")
    if isinstance(g, list) and len(g) == len(DIMS):
        pairs = sorted(zip(DIMS, g), key=lambda x: float(x[1]), reverse=True)
        active = [f"{name}={float(score):.2f}" for name, score in pairs if float(score) >= 0.10]
        return ", ".join(active[:4]) if active else f"{pairs[0][0]}={float(pairs[0][1]):.2f}"

    for key in ["target_dimension", "condition_dim", "used_dimension", "predicted_dimension", "dimension"]:
        value = ex.get(key)
        if value:
            return str(value)
    return "unspecified"


def build_professor_sft_user_content(ex: Dict) -> str:
    q = clean_text(get_field(ex, "question", "query", "user_question"))
    u = clean_text(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"))
    z = sanitize_draft(get_field(ex, "z_t", "draft", "masked_unsafe_draft", default=""))
    aspect = format_aspect_signal(ex)
    t = ex.get("t", ex.get("corruption_level", ""))
    T = ex.get("T", "")
    t_str = f"{t}/{T}" if str(t) and str(T) else str(t)

    return (
        "Rewrite the unsafe counseling response into a safe final response.\n"
        "Use the question, the unsafe response, the aspect signal, and the corrupted draft.\n"
        "Return only the final response. Do not include section labels, analysis, metadata, or the draft.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Aspect signal:\n{aspect}\n\n"
        f"Corruption strength:\n{t_str}\n\n"
        f"Risk-deleted draft z_t:\n{z}\n\n"
        "Safety requirements:\n"
        "- Validate the user's feelings without blaming or shaming them.\n"
        "- Do not diagnose the user or state that they have a disorder.\n"
        "- Do not give direct medication or treatment instructions.\n"
        "- Give concrete, bounded next steps when appropriate.\n"
        "- Encourage professional or trusted support for serious or persistent distress."
    )


def apply_chat_prompt(tokenizer, user_content: str, system: Optional[str] = None) -> str:
    system = system or (
        "You are a careful mental-health response refiner. "
        "Your job is to transform unsafe or low-quality counseling answers into safe, empathetic, "
        "specific, professionally bounded answers."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {user_content}\n\nAssistant:"


def build_sft_prompt(tokenizer, ex: Dict, prompt_style: str = "sft_plain") -> str:
    if prompt_style == "professor":
        user_content = build_professor_sft_user_content(ex)
    elif prompt_style == "sft_plain":
        user_content = build_sft_user_content(ex)
    else:
        raise ValueError(f"Unsupported SFT prompt style: {prompt_style}")
    return apply_chat_prompt(tokenizer, user_content)


def format_g(g: Sequence[float]) -> str:
    return ", ".join(f"{DIMS[i]}={float(g[i]):.2f}" for i in range(min(len(g), len(DIMS))))


def build_risk_tune_user_content(ex: Dict) -> str:
    q = clean_text(get_field(ex, "question", "query", "user_question"))
    u = clean_text(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"))
    sft = clean_text(get_field(ex, "sft_response", "professor_peft_response", "peft_response"))
    z_t = clean_text(get_field(ex, "z_t_from_sft", "risk_corrupted_sft_response", "z_t"))
    g = ex.get("g_sft") or ex.get("g") or [0.0] * len(DIMS)
    if isinstance(g, dict):
        g = [float(g.get(dim, 0.0)) for dim in DIMS]
    g_text = format_g(g)
    risk_tag_instruction = ""
    if "[Risk:" in z_t or "<RISK" in z_t:
        risk_tag_instruction = (
            "Spans marked with risk tags are likely unsafe or low-quality. "
            "Revise those spans while preserving useful context from the SFT response.\n\n"
        )

    return (
        "Rewrite the SFT-refined counseling response into a safer final response while preserving useful specific content.\n"
        "Use the original unsafe response only as context for the user's situation; repair the SFT response.\n"
        "Return only the final response. Do not include section labels, analysis, metadata, or the draft.\n\n"
        f"Question:\n{q}\n\n"
        f"Original unsafe response:\n{u}\n\n"
        f"SFT refined response:\n{sft}\n\n"
        f"{risk_tag_instruction}"
        f"Risk-aware corrupted SFT response:\n{z_t}\n\n"
        f"Aspect scores of SFT response:\n{g_text}\n\n"
        "Safety requirements:\n"
        "- Validate the user's feelings without blaming or shaming them.\n"
        "- Do not diagnose the user or state that they have a disorder.\n"
        "- Do not give direct medication or treatment instructions.\n"
        "- Keep concrete details and bounded next steps when they are useful.\n"
        "- Encourage professional or trusted support for serious or persistent distress."
    )


def build_risk_tune_prompt(tokenizer, ex: Dict) -> str:
    return apply_chat_prompt(tokenizer, build_risk_tune_user_content(ex))


def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if p.strip()]


def router_text(q: str, response: str) -> str:
    return (
        "Question:\n"
        + q.strip()
        + "\n\nCandidate counseling response:\n"
        + response.strip()
        + "\n\nTask: identify all violated mental-health response quality dimensions."
    )


def risk_text(q: str, span: str) -> str:
    return (
        "Question:\n"
        + q.strip()
        + "\n\nCandidate span:\n"
        + span.strip()
        + "\n\nTask: predict which counseling quality dimensions this span may violate."
    )


@torch.no_grad()
def predict_g(router, tokenizer, q: str, response: str, device, max_len: int = 512) -> List[float]:
    enc = tokenizer(router_text(q, response), return_tensors="pt", truncation=True, max_length=max_len).to(device)
    probs = torch.sigmoid(router(**enc).logits.float())[0].detach().cpu().tolist()
    probs = [float(x) for x in probs[: len(DIMS)]]
    if len(probs) < len(DIMS):
        probs.extend([0.0] * (len(DIMS) - len(probs)))
    if sum(probs) <= 1e-6:
        probs = [1.0 / len(DIMS)] * len(DIMS)
    return probs


@torch.no_grad()
def score_spans(model, tokenizer, q: str, spans: Sequence[str], device, max_len: int = 384) -> List[List[float]]:
    if not spans:
        return []
    enc = tokenizer(
        [risk_text(q, span) for span in spans],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)
    rows = torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()
    out = []
    for row in rows:
        values = [float(x) for x in row[: len(DIMS)]]
        if len(values) < len(DIMS):
            values.extend([0.0] * (len(DIMS) - len(values)))
        out.append(values)
    return out


def weighted_risk(g: Sequence[float], risk_vec: Sequence[float]) -> float:
    if not risk_vec:
        return 0.0
    return max(float(g[k]) * float(risk_vec[k]) for k in range(min(len(g), len(risk_vec), len(DIMS))))


def top_risk_dim(g: Sequence[float], risk_vec: Sequence[float]) -> Tuple[float, str]:
    scores = []
    for k in range(len(DIMS)):
        rv = float(risk_vec[k]) if k < len(risk_vec) else 0.0
        gv = float(g[k]) if k < len(g) else 0.0
        scores.append(gv * rv)
    best = max(range(len(scores)), key=lambda k: scores[k])
    return float(scores[best]), DIMS[best]


def staged_mask_count(n_candidates: int, t: int, t2_frac: float, t3_frac: float) -> int:
    if t <= 1 or n_candidates <= 0:
        return 0
    if t == 2:
        count = math.ceil(float(t2_frac) * n_candidates)
    elif t == 3:
        count = math.ceil(float(t3_frac) * n_candidates)
    else:
        count = n_candidates
    return max(0, min(n_candidates, count))


def make_zt_from_response(
    response: str,
    g: Sequence[float],
    risk_vecs: Sequence[Sequence[float]],
    strategy: str = "staged_risk",
    t: int = 3,
    T: int = 4,
    mask_token: str = "<MASK>",
    risk_threshold: float = 0.35,
    mask_threshold: float = 0.35,
    rho: float = 0.15,
    lambda_mask: float = 0.75,
    t2_frac: float = 0.33,
    t3_frac: float = 0.66,
    risk_tag_format: str = "[Risk: {dim}] {span} [/Risk]",
) -> Tuple[str, List[Dict]]:
    spans = split_sentences(response) or [str(response or "").strip()]
    items = []
    for i, span in enumerate(spans):
        rv = risk_vecs[i] if i < len(risk_vecs) else [0.0] * len(DIMS)
        risk, top_dim = top_risk_dim(g, rv)
        items.append({"idx": i, "span": span, "risk": risk, "top_dim": top_dim})

    if strategy == "threshold":
        beta = t / float(T)
        parts, infos = [], []
        for item in items:
            pi = max(0.0, min(1.0, rho + lambda_mask * beta * item["risk"]))
            p_mask = beta * pi
            should_mask = (p_mask >= mask_threshold) or (item["risk"] >= risk_threshold and t >= 2)
            parts.append(mask_token if should_mask else item["span"])
            state = "MASK" if should_mask else "KEEP"
            infos.append({**item, "r_l_g": item["risk"], "p_mask": p_mask, "state": state, "strategy": strategy})
        return " ".join(parts), infos

    if strategy == "staged":
        ranked = sorted(items, key=lambda x: x["risk"], reverse=True)
    elif strategy in {"staged_risk", "risk_tag"}:
        ranked = sorted([x for x in items if x["risk"] >= risk_threshold], key=lambda x: x["risk"], reverse=True)
    else:
        raise ValueError(f"Unsupported z_t strategy: {strategy}")

    ranks = {item["idx"]: rank + 1 for rank, item in enumerate(ranked)}
    count = staged_mask_count(len(ranked), t, t2_frac, t3_frac)
    selected = {item["idx"] for item in ranked[:count]}

    parts, infos = [], []
    for item in items:
        if item["idx"] in selected and strategy == "risk_tag":
            parts.append(risk_tag_format.format(dim=item["top_dim"], span=item["span"], risk=item["risk"]))
            state = "MARKED"
        elif item["idx"] in selected:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(item["span"])
            state = "KEEP"
        infos.append(
            {
                **item,
                "r_l_g": item["risk"],
                "rank": ranks.get(item["idx"]),
                "candidate": item["idx"] in ranks,
                "state": state,
                "strategy": strategy,
            }
        )
    return " ".join(parts), infos


def score_candidate(
    q: str,
    response: str,
    router,
    router_tok,
    risk_model,
    risk_tok,
    device,
    router_max_len: int = 512,
    risk_max_len: int = 384,
) -> Dict:
    response = clean_text(response)
    g = predict_g(router, router_tok, q, response, device, max_len=router_max_len)
    spans = split_sentences(response) or ([response] if response else [])
    risk_vecs = score_spans(risk_model, risk_tok, q, spans, device, max_len=risk_max_len)
    infos = []
    risk_score = 0.0
    for span, rv in zip(spans, risk_vecs):
        risk, dim = top_risk_dim(g, rv)
        risk_score = max(risk_score, risk)
        infos.append({"span": span, "risk_vec": rv, "r_l_g": risk, "top_dim": dim})
    return {"g": g, "spans": spans, "risk_vecs": risk_vecs, "span_risks": infos, "risk_score": float(risk_score)}


def count_bad_safety_patterns(text: str) -> int:
    return sum(1 for pat in BAD_SAFETY_PATTERNS if re.search(pat, str(text or ""), flags=re.I))


def count_generic_phrases(text: str) -> int:
    low = str(text or "").lower()
    return sum(low.count(phrase) for phrase in GENERIC_PHRASES)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", str(text or "")))


def content_keywords(text: str) -> set:
    words = re.findall(r"[A-Za-z][A-Za-z']+", str(text or "").lower())
    return {w.strip("'") for w in words if len(w.strip("'")) >= 4 and w.strip("'") not in STOPWORDS}


def keyword_overlap_ratio(question: str, response: str) -> float:
    q_words = content_keywords(question)
    if not q_words:
        return 1.0
    r_words = content_keywords(response)
    return len(q_words & r_words) / max(1, len(q_words))


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_source_len: int = 896,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_source_len).to(device)
    kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature and temperature > 0:
        kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        kwargs.update({"do_sample": False})
    gen = model.generate(**kwargs)
    new_tokens = gen[0][enc["input_ids"].shape[-1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, cleanup_response(raw)
