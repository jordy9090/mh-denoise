import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import prepare_model_for_kbit_training

from aspect_moe_lora import (
    DIMS,
    count_trainable_parameters,
    freeze_model_parameters,
    initialize_shared_lora_from_peft,
    normalize_gates,
    save_moe_adapter,
    set_shared_lora_trainable,
    set_moe_gates,
    wrap_aspect_moe_layers,
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_g(ex):
    g = ex.get("g")
    if isinstance(g, list):
        vals = [float(x) for x in g[: len(DIMS)]]
    elif isinstance(g, dict):
        vals = [float(g.get(dim, 0.0)) for dim in DIMS]
    else:
        vals = [1.0 / len(DIMS)] * len(DIMS)

    if len(vals) < len(DIMS):
        vals.extend([0.0] * (len(DIMS) - len(vals)))
    vals = [max(0.0, float(x)) for x in vals[: len(DIMS)]]
    if sum(vals) <= 1e-8:
        vals = [1.0 / len(DIMS)] * len(DIMS)
    return vals


def build_prompt(ex):
    q = ex["question"].strip()
    u = ex["unsafe_response"].strip()
    z = (ex.get("z_t") or "").strip()
    source = ex.get("source", "unsafe")
    t = ex.get("t", 0)

    if not z:
        z = "No draft. Rewrite directly from the unsafe response."

    g = parse_g(ex)
    g_text = ", ".join(f"{DIMS[i]}={float(g[i]):.2f}" for i in range(len(DIMS)))

    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n"
        "Do not copy blaming, diagnostic, toxic, or unsupported wording from the unsafe response.\n\n"
        f"Aspect scores:\n{g_text}\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response to fix:\n{u}\n\n"
        f"Draft to revise:\n{z}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Safe response:\n"
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


class DenoiseMoEDS(Dataset):
    def __init__(self, path, tok, max_source_len=512, max_target_len=160, lambda_y=1.5):
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
        target = ex["safe_response"].strip()
        if self.tok.eos_token and not target.endswith(self.tok.eos_token):
            target += self.tok.eos_token

        pids = self.tok(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_len,
        )["input_ids"]

        try:
            enc = self.tok(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_len,
                return_offsets_mapping=True,
            )
            tids = enc["input_ids"]
            offs = enc["offset_mapping"]
        except Exception:
            tids = self.tok(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target_len,
            )["input_ids"]
            offs = [(0, 0)] * len(tids)

        ranges = find_ranges(target, ex.get("target_weight_spans", []))
        token_weights = []
        for a, b in offs:
            w = 1.0
            for s, e, r in ranges:
                if not (b <= s or a >= e):
                    w = max(w, 1.0 + self.lambda_y * r)
            token_weights.append(w)

        ids = pids + tids
        labels = [-100] * len(pids) + tids
        weights = [0.0] * len(pids) + token_weights

        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
            "token_weights": weights,
            "g": parse_g(ex),
        }


@dataclass
class Collator:
    tok: object
    max_len: int

    def __call__(self, batch):
        m = min(self.max_len, max(len(x["input_ids"]) for x in batch))
        pad = self.tok.pad_token_id

        ids, masks, labels, weights, gates = [], [], [], [], []
        for x in batch:
            ids0 = x["input_ids"][:m]
            masks0 = x["attention_mask"][:m]
            lab0 = x["labels"][:m]
            w0 = x["token_weights"][:m]
            n = m - len(ids0)

            ids.append(ids0 + [pad] * n)
            masks.append(masks0 + [0] * n)
            labels.append(lab0 + [-100] * n)
            weights.append(w0 + [0.0] * n)
            gates.append(x["g"])

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": torch.tensor(weights, dtype=torch.float),
            "g": torch.tensor(gates, dtype=torch.float),
        }


def weighted_loss(logits, labels, token_weights):
    sl = logits[:, :-1, :].float().contiguous()
    y = labels[:, 1:].contiguous()
    w = token_weights[:, 1:].float().contiguous()
    vocab = sl.size(-1)

    loss = F.cross_entropy(
        sl.view(-1, vocab),
        y.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(y.size())

    mask = (y != -100).float()
    weights = torch.where(mask > 0, torch.clamp(w, min=1.0, max=2.5), torch.zeros_like(w))
    per = (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return per.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        gates = batch.pop("g").to(device)
        labels = batch.pop("labels").to(device)
        token_weights = batch.pop("token_weights").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        set_moe_gates(model, gates)
        logits = model(**batch).logits
        loss = weighted_loss(logits, labels, token_weights)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def dataset_summary(name, dataset):
    rows = dataset.rows
    print(f"{name} rows:", len(rows))
    print(f"{name} source distribution:", Counter(str(r.get("source", "missing")) for r in rows))
    print(f"{name} g_source distribution:", Counter(str(r.get("g_source", "missing")) for r in rows))
    has_weight_spans = any(bool(r.get("target_weight_spans")) for r in rows)
    print(f"{name} target weight spans present:", has_weight_spans)
    print(f"{name} lambda_y:", dataset.lambda_y)
    print(f"{name} risk-weighted loss active:", has_weight_spans and dataset.lambda_y != 0.0)


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


def build_moe_config(args, wrapped_names=None, init_report=None):
    return {
        "dims": DIMS,
        "base_model": args.model,
        "target_regex": args.target_regex,
        "r_shared": args.r_shared,
        "r_expert": args.r_expert,
        "alpha_shared": args.alpha_shared,
        "alpha_expert": args.alpha_expert,
        "dropout": args.dropout,
        "moe_eps": args.moe_eps,
        "wrapped_module_names": wrapped_names or [],
        "init_shared_adapter_dir": args.init_shared_adapter_dir,
        "freeze_shared_lora": args.freeze_shared_lora,
        "init_shared_report": init_report,
        "lambda_y": args.lambda_y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--r_shared", type=int, default=8)
    ap.add_argument("--r_expert", type=int, default=8)
    ap.add_argument("--alpha_shared", type=int, default=16)
    ap.add_argument("--alpha_expert", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_regex",
        default=r".*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
    )
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--moe_eps", type=float, default=1e-4)
    ap.add_argument("--init_shared_adapter_dir", default="")
    ap.add_argument("--allow_partial_shared_init", action="store_true")
    ap.add_argument("--no_preserve_init_scale", action="store_true")
    ap.add_argument("--freeze_shared_lora", action="store_true")
    ap.add_argument("--no_initial_eval", action="store_true")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--lambda_y",
        type=float,
        default=1.5,
        help="Risk-weight strength for target tokens. Use 0.0 for MoE denoising SFT without risk-weighted CE.",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("load_in_4bit:", use_4bit)
    model = load_base_model(args.model, use_4bit)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    freeze_model_parameters(model)
    config = build_moe_config(args)
    wrapped_names = wrap_aspect_moe_layers(model, config)
    init_report = None
    if args.init_shared_adapter_dir:
        init_report = initialize_shared_lora_from_peft(
            model,
            args.init_shared_adapter_dir,
            require_all=not args.allow_partial_shared_init,
            preserve_scale=not args.no_preserve_init_scale,
        )
    if args.freeze_shared_lora:
        set_shared_lora_trainable(model, False)
        print("shared LoRA trainable: False")
    else:
        print("shared LoRA trainable: True")
    config = build_moe_config(args, wrapped_names, init_report)
    print("wrapped modules:", len(wrapped_names))
    print("first wrapped modules:", wrapped_names[:10])

    trainable, total, pct = count_trainable_parameters(model)
    print(f"trainable params: {trainable:,}")
    print(f"all params: {total:,}")
    print(f"trainable percent: {pct:.4f}")

    device = next(model.parameters()).device
    train = DenoiseMoEDS(args.train_file, tok, args.max_source_len, args.max_target_len, lambda_y=args.lambda_y)
    valid = DenoiseMoEDS(args.valid_file, tok, args.max_source_len, args.max_target_len, lambda_y=args.lambda_y)
    dataset_summary("train", train)
    dataset_summary("valid", valid)

    coll = Collator(tok, args.max_source_len + args.max_target_len)
    tl = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=coll,
        num_workers=args.num_workers,
    )
    vl = DataLoader(
        valid,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=coll,
        num_workers=args.num_workers,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    planned_updates = math.ceil(len(tl) / args.grad_accum) * args.epochs
    if args.max_train_steps is not None:
        planned_updates = min(planned_updates, args.max_train_steps)
    planned_updates = max(1, planned_updates)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * planned_updates), planned_updates)

    best = 999.0
    step = 0
    printed_tau = False
    stop = False
    model.train()
    opt.zero_grad(set_to_none=True)

    if args.init_shared_adapter_dir and not args.no_initial_eval:
        ev = evaluate(model, vl, device)
        best = ev
        print(f"[eval] step=0 initialized_shared_loss={ev:.4f}")
        save_moe_adapter(model, tok, out / "best", config)

    for ep in range(args.epochs):
        pbar = tqdm(tl, desc=f"epoch {ep+1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            gates = batch.pop("g").to(device)
            labels = batch.pop("labels").to(device)
            token_weights = batch.pop("token_weights").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            if not printed_tau:
                tau = normalize_gates(gates, args.moe_eps)
                mean_tau = tau.mean(dim=0).detach().cpu().tolist()
                print("mean tau first batch:", {DIMS[j]: round(float(mean_tau[j]), 4) for j in range(len(DIMS))})
                printed_tau = True

            set_moe_gates(model, gates)
            logits = model(**batch).logits
            loss = weighted_loss(logits, labels, token_weights) / args.grad_accum

            if not torch.isfinite(loss):
                print("[warn] non-finite loss")
                opt.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                shown_loss = float(loss.item() * args.grad_accum)
                pbar.set_postfix({"loss": round(shown_loss, 4), "step": step})

                if step % args.eval_every == 0:
                    ev = evaluate(model, vl, device)
                    print(f"[eval] step={step} loss={ev:.4f}")
                    if ev < best:
                        best = ev
                        save_moe_adapter(model, tok, out / "best", config)

                if step % args.save_every == 0:
                    save_moe_adapter(model, tok, out / f"step_{step}", config)

                if args.max_train_steps is not None and step >= args.max_train_steps:
                    stop = True
                    break

        if stop:
            break

    save_moe_adapter(model, tok, out / "final", config)
    print("[done] best", best)
    print("[done] final ->", out / "final")


if __name__ == "__main__":
    main()
