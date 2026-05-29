import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)


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


def get_field(ex, *names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class MultiAspectLoRALinear(nn.Module):
    def __init__(self, base, num_aspects=6, r=2, alpha=8, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = getattr(base, "in_features")
        self.out_features = getattr(base, "out_features")
        self.num_aspects = num_aspects
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        try:
            base_param = next(base.parameters())
            device = base_param.device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.A_sh = nn.Parameter(torch.empty(r, self.in_features, device=device, dtype=dtype))
        self.B_sh = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        self.A_exp = nn.Parameter(torch.empty(num_aspects, r, self.in_features, device=device, dtype=dtype))
        self.B_exp = nn.Parameter(torch.zeros(num_aspects, self.out_features, r, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.A_sh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_exp, a=math.sqrt(5))
        self._gates = None

    def set_gates(self, g):
        self._gates = g

    def forward(self, x):
        base_out = self.base(x)
        xd = self.dropout(x)

        device = xd.device
        dtype = xd.dtype

        A_sh = self.A_sh.to(device=device, dtype=dtype)
        B_sh = self.B_sh.to(device=device, dtype=dtype)
        A_exp = self.A_exp.to(device=device, dtype=dtype)
        B_exp = self.B_exp.to(device=device, dtype=dtype)

        sh = F.linear(F.linear(xd, A_sh), B_sh) * self.scaling

        if xd.dim() == 3 and self._gates is not None and self._gates.size(0) == xd.size(0):
            gates = self._gates.to(device=device, dtype=dtype)
            exp = 0
            for k in range(self.num_aspects):
                ok = F.linear(F.linear(xd, A_exp[k]), B_exp[k]) * self.scaling
                exp = exp + ok * gates[:, k].view(-1, 1, 1)
        else:
            if self._gates is None:
                gates = torch.ones(self.num_aspects, device=device, dtype=dtype) / self.num_aspects
            else:
                gates = self._gates.mean(dim=0).to(device=device, dtype=dtype)
            exp = 0
            for k in range(self.num_aspects):
                ok = F.linear(F.linear(xd, A_exp[k]), B_exp[k]) * self.scaling
                exp = exp + ok * gates[k]

        return base_out + sh + exp


def replace_with_moe_lora(model, target_names=("q_proj", "v_proj"), r=2, alpha=8):
    count = 0
    for name, module in list(model.named_modules()):
        if any(name.endswith(t) for t in target_names):
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent = model.get_submodule(parent_name)
            old = getattr(parent, child_name)
            if hasattr(old, "in_features") and hasattr(old, "out_features"):
                setattr(parent, child_name, MultiAspectLoRALinear(old, num_aspects=len(DIMS), r=r, alpha=alpha, dropout=0.0))
                count += 1
    print("replaced linear modules with MultiAspectLoRA:", count)
    return model


def set_all_gates(model, g):
    eps = 1e-4
    tau = (g + eps) / (g + eps).sum(dim=1, keepdim=True).clamp_min(eps)
    for m in model.modules():
        if isinstance(m, MultiAspectLoRALinear):
            m.set_gates(tau)


def router_text(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "Question:\n" + q.strip()
        + "\n\nUnsafe response:\n" + u.strip()
        + "\n\nTask: identify all violated mental-health response quality dimensions."
    )


@torch.no_grad()
def predict_g(router, router_tok, ex, device):
    text = router_text(ex)
    enc = router_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    probs = torch.sigmoid(router(**enc).logits.float())[0].detach().cpu().tolist()

    # avoid all-zero or tiny gate collapse
    s = sum(probs)
    if s <= 1e-6:
        probs = [1.0 / len(DIMS)] * len(DIMS)

    return probs


def risk_text(question, span):
    return (
        "Question:\n" + question.strip()
        + "\n\nCandidate span:\n" + span.strip()
        + "\n\nTask: predict which counseling quality dimensions this span may violate."
    )


@torch.no_grad()
def score_spans(risk_model, risk_tok, question, spans, device):
    if not spans:
        return []

    texts = [risk_text(question, s) for s in spans]
    enc = risk_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(device)
    probs = torch.sigmoid(risk_model(**enc).logits.float()).detach().cpu().tolist()
    return probs


def rl_from_g(g, risk_vec):
    return max(float(g[k]) * float(risk_vec[k]) for k in range(len(DIMS)))


def make_unsafe_zt(
    unsafe_response,
    g,
    risk_vecs,
    t,
    T,
    rho=0.15,
    lambda_mask=0.75,
    mask_token="[needs revision]",
    risk_threshold=0.35,
    mask_threshold=0.35,
):
    spans = split_sentences(unsafe_response) or [unsafe_response.strip()]
    beta = t / float(T)

    parts = []
    states = []
    span_infos = []

    for span, risk_vec in zip(spans, risk_vecs):
        r = rl_from_g(g, risk_vec)
        pi = max(0.0, min(1.0, rho + lambda_mask * beta * r))
        p_mask = beta * pi

        # Deterministic inference version of SAFE/UNSAFE/MASK.
        # Since y is unavailable at inference, SAFE cannot be realized.
        # We choose between UNSAFE and MASK.
        should_mask = (p_mask >= mask_threshold) or (r >= risk_threshold and t >= 2)

        if should_mask:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(span)
            state = "UNSAFE"

        states.append(state)
        span_infos.append({
            "span": span,
            "risk_vec": {DIMS[i]: float(risk_vec[i]) for i in range(len(DIMS))},
            "r_l_g": float(r),
            "beta": float(beta),
            "pi": float(pi),
            "p_mask": float(p_mask),
            "state": state,
        })

    return " ".join(parts), states, span_infos


def build_prompt(question, unsafe_response, z_t, source, t):
    if not z_t:
        z_t = "No draft. Rewrite directly from the unsafe response."

    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n"
        "Do not copy blaming, diagnostic, toxic, or unsupported wording from the unsafe response.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Unsafe response to fix:\n{unsafe_response.strip()}\n\n"
        f"Draft to revise:\n{z_t.strip()}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Safe response:\n"
    )


def cleanup(text):
    text = text.strip()
    markers = [
        "Question:", "Unsafe response:", "Unsafe response to fix:", "Draft:", "Draft to revise:",
        "Corruption:", "Response:", "Safe response:", "Analysis:", "Thought:", "Reasoning:",
        "Final answer:", "Final response:"
    ]
    for m in markers:
        if m in text:
            before = text.split(m)[0].strip()
            after = text.split(m)[-1].strip()
            text = before if before else after

    text = text.replace("[needs revision]", "").replace("<MASK>", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1{1,}", r"\1", text)
    return text.strip()


def parse_modes(modes):
    out = []
    for m in modes.split(","):
        m = m.strip()
        if not m:
            continue
        if m == "empty":
            out.append(("empty", 0))
        elif m.startswith("unsafe_t"):
            t = int(m.replace("unsafe_t", ""))
            out.append(("unsafe", t))
        else:
            raise ValueError(f"Unknown mode: {m}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_modules", default="q_proj,v_proj")
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--modes", default="empty,unsafe_t1,unsafe_t2,unsafe_t3,unsafe_t4")
    ap.add_argument("--mask_token", default="[needs revision]")
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--lambda_mask", type=float, default=0.75)
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--mask_threshold", type=float, default=0.35)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    args = ap.parse_args()

    # Router
    router_tok = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir)
    router_device = "cuda" if torch.cuda.is_available() else "cpu"
    router.to(router_device)
    router.eval()

    # Risk scorer
    risk_tok = AutoTokenizer.from_pretrained(args.risk_scorer_dir)
    risk_model = AutoModelForSequenceClassification.from_pretrained(args.risk_scorer_dir)
    risk_device = router_device
    risk_model.to(risk_device)
    risk_model.eval()

    # Gemma + MoE adapter
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

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = replace_with_moe_lora(
        model,
        target_names=tuple(x.strip() for x in args.target_modules.split(",") if x.strip()),
        r=args.r,
        alpha=args.alpha,
    )

    state_path = Path(args.adapter_dir) / "moe_lora.pt"
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded moe state:", state_path)
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))

    model.eval()
    model_device = next(model.parameters()).device

    modes = parse_modes(args.modes)
    rows = read_jsonl(args.input)
    outs = []

    for ex in tqdm(rows):
        question = get_field(ex, "question", "query", "user_question")
        unsafe = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")

        g = predict_g(router, router_tok, ex, router_device)
        spans = split_sentences(unsafe) or [unsafe.strip()]
        risk_vecs = score_spans(risk_model, risk_tok, question, spans, risk_device)

        for source, t in modes:
            if source == "empty":
                z_t = ""
                states = []
                span_infos = []
            else:
                z_t, states, span_infos = make_unsafe_zt(
                    unsafe_response=unsafe,
                    g=g,
                    risk_vecs=risk_vecs,
                    t=t,
                    T=args.T,
                    rho=args.rho,
                    lambda_mask=args.lambda_mask,
                    mask_token=args.mask_token,
                    risk_threshold=args.risk_threshold,
                    mask_threshold=args.mask_threshold,
                )

            prompt = build_prompt(question, unsafe, z_t, source, t)
            g_tensor = torch.tensor([g], dtype=torch.float, device=model_device)
            set_all_gates(model, g_tensor)

            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(model_device)

            kwargs = dict(
                **enc,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            if args.temperature and args.temperature > 0:
                kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
            else:
                kwargs.update(dict(do_sample=False))

            with torch.no_grad():
                gen = model.generate(**kwargs)

            new_tokens = gen[0][enc["input_ids"].shape[-1]:]
            raw = tok.decode(new_tokens, skip_special_tokens=True)

            out = dict(ex)
            out["mode"] = "empty" if source == "empty" else f"unsafe_t{t}"
            out["source"] = source
            out["t"] = t
            out["T"] = args.T
            out["g"] = {DIMS[i]: float(g[i]) for i in range(len(DIMS))}
            out["z_t"] = z_t
            out["states"] = states
            out["span_risks"] = span_infos
            out["real_moe_response_raw"] = raw
            out["real_moe_response"] = cleanup(raw)
            outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
