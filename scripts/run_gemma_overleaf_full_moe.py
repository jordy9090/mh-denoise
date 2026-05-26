import argparse, json, math, re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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


class MultiAspectLoRALinear(nn.Module):
    def __init__(self, base, num_aspects=6, r=2, alpha=8, dropout=0.05):
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


def replace_with_moe_lora(model, target_names=("q_proj", "v_proj"), r=2, alpha=8, dropout=0.05):
    count = 0
    for name, module in list(model.named_modules()):
        if any(name.endswith(t) for t in target_names):
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent = model.get_submodule(parent_name)
            old = getattr(parent, child_name)
            if hasattr(old, "in_features") and hasattr(old, "out_features"):
                setattr(parent, child_name, MultiAspectLoRALinear(old, num_aspects=len(DIMS), r=r, alpha=alpha, dropout=dropout))
                count += 1
    print("replaced linear modules with MultiAspectLoRA:", count)
    return model


def set_all_gates(model, g):
    eps = 1e-4
    tau = (g + eps) / (g + eps).sum(dim=1, keepdim=True).clamp_min(eps)
    for m in model.modules():
        if isinstance(m, MultiAspectLoRALinear):
            m.set_gates(tau)


def build_prompt(ex):
    q = ex["question"].strip()
    u = ex["unsafe_response"].strip()
    z = ex.get("z_t", "").strip()
    source = ex.get("source", "bridge")
    t = ex.get("t", 1)

    if not z:
        z = "No draft."

    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Draft:\n{z}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Response:\n"
    )


def cleanup(text):
    text = text.strip()
    markers = [
        "Question:", "Unsafe response:", "Draft:", "Corruption:", "Response:",
        "Analysis:", "Thought:", "Reasoning:", "Final answer:", "Final response:"
    ]
    for m in markers:
        if m in text:
            before = text.split(m)[0].strip()
            after = text.split(m)[-1].strip()
            text = before if before else after

    text = text.replace("<MASK>", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1{1,}", r"\1", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_modules", default="q_proj,v_proj")
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
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
        dropout=0.0,
    )

    state_path = Path(args.adapter_dir) / "moe_lora.pt"
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded moe state:", state_path)
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))

    model.eval()
    device = next(model.parameters()).device

    outs = []
    rows = read_jsonl(args.input)

    for ex in tqdm(rows):
        prompt = build_prompt(ex)
        g = torch.tensor([ex["g"]], dtype=torch.float, device=device)
        set_all_gates(model, g)

        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(device)

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
        out["moe_response_raw"] = raw
        out["moe_response"] = cleanup(raw)
        outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)


if __name__ == "__main__":
    main()
