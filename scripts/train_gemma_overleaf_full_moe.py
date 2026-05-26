import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_aspect_signal(ex):
    g = ex.get("g")
    if isinstance(g, list) and len(g) == len(DIMS):
        pairs = sorted(zip(DIMS, g), key=lambda x: float(x[1]), reverse=True)
        top = [f"{k}={float(v):.2f}" for k, v in pairs if float(v) >= 0.10]
        if not top:
            top = [f"{pairs[0][0]}={float(pairs[0][1]):.2f}"]
        return ", ".join(top)
    for key in ["condition_dim", "target_dimension", "used_dimension", "predicted_dimension"]:
        if ex.get(key):
            return str(ex[key])
    return "unspecified"


def build_prompt(ex):
    q = str(ex.get("question", ex.get("query", ""))).strip()
    u = str(ex.get("unsafe_response", ex.get("corrupted_response", ex.get("bad_response", "")))).strip()
    z = str(ex.get("z_t", ex.get("draft", ""))).strip() or "No draft."
    source = str(ex.get("source", ex.get("corruption_type", "unknown"))).strip()
    t = ex.get("t", ex.get("corruption_level", ""))
    aspect = format_aspect_signal(ex)
    return (
        "Rewrite the given response into a safe, helpful final response.\n"
        "Return only the final response. Do not include labels, notes, analysis, or metadata.\n\n"
        f"Question:\n{q}\n\n"
        f"Input response:\n{u}\n\n"
        f"Aspect signal:\n{aspect}\n\n"
        f"Draft:\n{z}\n\n"
        f"Setting:\nsource={source}, t={t}\n\n"
        "Final response:\n"
    )


def find_ranges(target, spans):
    ranges = []
    low = target.lower()
    for sp in spans:
        s = str(sp.get("safe_span", "")).strip()
        if not s:
            continue
        pos = low.find(s.lower())
        if pos >= 0:
            ranges.append((pos, pos + len(s), float(sp.get("risk", 0.0))))
    return ranges


class DenoiseDS(Dataset):
    def __init__(self, path, tok, max_source_len=768, max_target_len=192, lambda_y=1.5):
        self.rows = read_jsonl(path)
        self.tok = tok
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.lambda_y = lambda_y

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        prompt = build_prompt(ex)
        target = str(ex.get("safe_response", ex.get("target", ""))).strip()
        if self.tok.eos_token and not target.endswith(self.tok.eos_token):
            target += self.tok.eos_token
        pids = self.tok(prompt, add_special_tokens=True, truncation=True, max_length=self.max_source_len)["input_ids"]
        enc = self.tok(target, add_special_tokens=False, truncation=True, max_length=self.max_target_len, return_offsets_mapping=True)
        tids = enc["input_ids"]
        offs = enc.get("offset_mapping", [(0, 0)] * len(tids))
        ranges = find_ranges(target, ex.get("target_weight_spans", []))
        tw = []
        for a, b in offs:
            w = 1.0
            for s, e, r in ranges:
                if not (b <= s or a >= e):
                    w = max(w, 1.0 + self.lambda_y * r)
            tw.append(w)
        ids = pids + tids
        labels = [-100] * len(pids) + tids
        token_weights = [0.0] * len(pids) + tw
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels, "token_weights": token_weights}


@dataclass
class Collator:
    tok: object
    max_len: int
    def __call__(self, batch):
        m = min(self.max_len, max(len(x["input_ids"]) for x in batch))
        pad = self.tok.pad_token_id
        ids, masks, labels, tw = [], [], [], []
        for x in batch:
            ids0 = x["input_ids"][:m]
            masks0 = x["attention_mask"][:m]
            lab0 = x["labels"][:m]
            tw0 = x["token_weights"][:m]
            n = m - len(ids0)
            ids.append(ids0 + [pad] * n)
            masks.append(masks0 + [0] * n)
            labels.append(lab0 + [-100] * n)
            tw.append(tw0 + [0.0] * n)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels), "token_weights": torch.tensor(tw, dtype=torch.float)}


def weighted_loss(logits, labels, tw):
    sl = logits[:, :-1, :].float().contiguous()
    y = labels[:, 1:].contiguous()
    w = tw[:, 1:].contiguous().float()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), y.view(-1), ignore_index=-100, reduction="none").view(y.size())
    mask = (y != -100).float()
    weights = torch.where(mask > 0, torch.clamp(w, min=1.0, max=2.5), torch.zeros_like(w))
    return ((loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).mean()


@torch.no_grad()
def evaluate(model, loader, device, use_token_weights=False):
    model.eval()
    losses = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        tw = batch.pop("token_weights").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, labels=labels)
        loss = weighted_loss(out.logits, labels, tw) if use_token_weights else out.loss
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_adapter(model, tok, out):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    with open(out / "dims.json", "w") as f:
        json.dump(DIMS, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_target_len", type=int, default=192)
    ap.add_argument("--target_modules", default="q_proj,v_proj")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--use_token_weights", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()], bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    train = DenoiseDS(args.train_file, tok, args.max_source_len, args.max_target_len)
    valid = DenoiseDS(args.valid_file, tok, args.max_source_len, args.max_target_len)
    coll = Collator(tok, args.max_source_len + args.max_target_len)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=coll, num_workers=2)
    vl = DataLoader(valid, batch_size=args.batch_size, shuffle=False, collate_fn=coll, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates = math.ceil(len(tl) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * updates), updates)
    best = float("inf")
    step = 0
    model.train()
    opt.zero_grad(set_to_none=True)

    for ep in range(args.epochs):
        pbar = tqdm(tl, desc=f"epoch {ep + 1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            labels = batch.pop("labels").to(device)
            tw = batch.pop("token_weights").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outp = model(**batch, labels=labels)
            loss = weighted_loss(outp.logits, labels, tw) if args.use_token_weights else outp.loss
            loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                print("[warn] non-finite loss")
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.set_postfix({"loss": round(float(loss.item() * args.grad_accum), 4), "step": step})
                if step % args.eval_every == 0:
                    ev = evaluate(model, vl, device, args.use_token_weights)
                    print(f"[eval] step={step} loss={ev:.4f}")
                    if ev < best:
                        best = ev
                        save_adapter(model, tok, out / "best")
                        print("[save] best ->", out / "best")
                if step % args.save_every == 0:
                    save_adapter(model, tok, out / f"step_{step}")
    save_adapter(model, tok, out / "final")
    print("[done] best", best)


if __name__ == "__main__":
    main()
