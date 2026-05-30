import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_field(ex, *names, default=""):
    for name in names:
        if name in ex and ex[name] is not None:
            return str(ex[name])
    return default


def split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if part.strip()]


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
        probs = [1.0 / len(DIMS)] * len(DIMS)
    return [float(x) for x in probs]


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
    enc = tok(
        [risk_text(q, span) for span in spans],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384,
    ).to(device)
    return torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()


def rl(g, risk_vec):
    return max(float(g[k]) * float(risk_vec[k]) for k in range(len(DIMS)))


def make_zt(
    draft,
    g,
    risk_vecs,
    t,
    T,
    mask_token="<MASK>",
    rho=0.15,
    lambda_mask=0.75,
    risk_threshold=0.35,
    mask_threshold=0.35,
):
    spans = split_sentences(draft) or [str(draft or "").strip()]
    if t <= 0:
        z_t = " ".join(span for span in spans if span).strip()
        infos = [{"span": span, "r_l_g": 0.0, "p_mask": 0.0, "state": "DRAFT"} for span in spans if span]
        return z_t, infos

    beta = t / float(T)
    parts, infos = [], []

    for span, risk_vec in zip(spans, risk_vecs):
        risk = rl(g, risk_vec)
        pi = max(0.0, min(1.0, rho + lambda_mask * beta * risk))
        p_mask = beta * pi
        should_mask = (p_mask >= mask_threshold) or (risk >= risk_threshold and t >= 2)

        if should_mask:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(span)
            state = "DRAFT"

        infos.append({"span": span, "r_l_g": risk, "p_mask": p_mask, "state": state})

    return " ".join(parts).strip(), infos


def build_prompt(q, u, z, g, source, t):
    if not z:
        z = "No draft. Rewrite directly from the unsafe response."

    g_text = ", ".join(f"{DIMS[i]}={float(g[i]):.2f}" for i in range(len(DIMS)))
    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n"
        "Do not copy blaming, diagnostic, toxic, or unsupported wording from the unsafe response.\n\n"
        f"Aspect scores:\n{g_text}\n\n"
        f"Question:\n{q.strip()}\n\n"
        f"Unsafe response to fix:\n{u.strip()}\n\n"
        f"Draft to revise:\n{z.strip()}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Safe response:\n"
    )


def cleanup(text):
    text = str(text or "").strip()
    markers = [
        "Question:",
        "Unsafe response:",
        "Unsafe response to fix:",
        "Draft:",
        "Draft to revise:",
        "Corruption:",
        "Response:",
        "Safe response:",
        "Analysis:",
        "Thought:",
        "Reasoning:",
        "Final answer:",
        "Final response:",
    ]
    for marker in markers:
        if marker in text:
            before = text.split(marker)[0].strip()
            after = text.split(marker)[-1].strip()
            text = before if before else after
    text = text.replace("<MASK>", "").replace("[needs revision]", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


def parse_steps(value):
    steps = []
    for item in value.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    if not steps:
        raise ValueError("--iter_steps must contain at least one timestep")
    return steps


@torch.no_grad()
def generate_one(model, tok, prompt, args, model_device):
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
    gen = model.generate(**gen_kwargs)
    new_tokens = gen[0][enc["input_ids"].shape[-1] :]
    raw = tok.decode(new_tokens, skip_special_tokens=True)
    return raw, cleanup(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--iter_steps", default="4,3,2")
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = parse_steps(args.iter_steps)

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
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    outs = []
    for ex in tqdm(rows):
        q = get_field(ex, "question", "query", "user_question")
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        g = predict_g(router, router_tok, ex, device)

        current_draft = u
        step_records = []
        final_raw = ""
        final_response = ""

        for step_index, t in enumerate(steps):
            spans = split_sentences(current_draft) or [current_draft]
            risk_vecs = score_spans(risk_model, risk_tok, q, spans, device)
            z_t, infos = make_zt(current_draft, g, risk_vecs, t, args.T, mask_token=args.mask_token)
            prompt = build_prompt(q, u, z_t, g, "unsafe", t)
            raw, response = generate_one(model, tok, prompt, args, model_device)

            step_records.append({
                "step_index": step_index,
                "t": t,
                "input_draft": current_draft,
                "z_t": z_t,
                "span_risks": infos,
                "peft_response_raw": raw,
                "peft_response": response,
            })

            if response:
                current_draft = response
                final_response = response
            final_raw = raw

        out = dict(ex)
        out["mode"] = "iter_" + "_".join(str(t) for t in steps)
        out["g"] = {DIMS[i]: float(g[i]) for i in range(len(DIMS))}
        out["z_t"] = step_records[-1]["z_t"] if step_records else ""
        out["span_risks"] = step_records[-1]["span_risks"] if step_records else []
        out["iter_steps"] = steps
        out["iterative_steps"] = step_records
        out["peft_response_raw"] = final_raw
        out["peft_response"] = final_response
        outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
