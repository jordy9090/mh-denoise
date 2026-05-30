import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig

from aspect_moe_lora import DIMS, load_moe_adapter, load_moe_config, set_moe_gates, wrap_aspect_moe_layers


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
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


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
    unsafe,
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
    spans = split_sentences(unsafe) or [unsafe.strip()]
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
            state = "UNSAFE"

        infos.append({"span": span, "r_l_g": risk, "p_mask": p_mask, "state": state})

    return " ".join(parts), infos


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
    text = text.strip()
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


def parse_modes(value):
    out = []
    for mode in value.split(","):
        mode = mode.strip()
        if mode == "empty":
            out.append(("empty", 0))
        elif mode.startswith("unsafe_t"):
            out.append(("unsafe", int(mode.replace("unsafe_t", ""))))
    return out


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
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    router_tok = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir).to(device).eval()

    risk_tok = AutoTokenizer.from_pretrained(args.risk_scorer_dir)
    risk_model = AutoModelForSequenceClassification.from_pretrained(args.risk_scorer_dir).to(device).eval()

    config = load_moe_config(args.adapter_dir)
    base_model = args.base_model or config.get("base_model")
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("load_in_4bit:", use_4bit)
    model = load_base_model(base_model, use_4bit)
    wrapped_names = wrap_aspect_moe_layers(model, config)
    print("wrapped modules:", len(wrapped_names))
    print("first wrapped modules:", wrapped_names[:10])
    load_moe_adapter(model, args.adapter_dir)
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
            else:
                z, infos = make_zt(u, g, risk_vecs, t, args.T, mask_token=args.mask_token)

            prompt = build_prompt(q, u, z, g, source, t)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(model_device)
            gates = torch.tensor([g], dtype=torch.float32, device=model_device)
            set_moe_gates(model, gates)

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
            out["span_risks"] = infos
            out["moe_response_raw"] = raw
            out["moe_response"] = cleanup(raw)
            outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
