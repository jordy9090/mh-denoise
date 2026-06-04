import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import prepare_model_for_kbit_training

from aspect_moe_lora import DIMS, freeze_model_parameters
from aspect_residual_mlp_moe import (
    count_trainable_parameters,
    inject_aspect_residual_mlp_moe,
    residual_mlp_trainable_parameters,
    save_mlp_moe_adapter,
    set_mlp_moe_gates,
)
from train_gemma_full_joint_denoiser import (
    BasePairDS,
    build_lm_batch,
    build_online_zt,
    build_prompt,
    collate_raw,
    mean_list,
    parse_timesteps,
    row_question,
    row_safe,
    row_unsafe,
    router_text,
    tensor_to_g_values,
    weighted_ce,
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


def load_router(router_dir, device):
    tok = AutoTokenizer.from_pretrained(router_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(router_dir).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def load_risk_scorer(risk_scorer_dir, device):
    tok = AutoTokenizer.from_pretrained(risk_scorer_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(risk_scorer_dir).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def predict_g_batch(router, router_tok, examples, device, max_len=512):
    texts = [router_text(row_question(ex), row_unsafe(ex)) for ex in examples]
    enc = router_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    probs = torch.sigmoid(router(**enc).logits.float())
    fallback = torch.full_like(probs, 1.0 / len(DIMS))
    probs = torch.where(probs.sum(dim=-1, keepdim=True) <= 1e-6, fallback, probs)
    return probs


def prepare_batch_prompts(examples, gates, timesteps, risk_model, risk_tok, risk_device, cache, args):
    prompts = []
    targets = []
    weight_spans = []
    masked_counts = []
    g_lists = tensor_to_g_values(gates)
    for ex, g_values, t in zip(examples, g_lists, timesteps):
        q = row_question(ex)
        u = row_unsafe(ex)
        y = row_safe(ex)
        z_t, _infos, weights, masked = build_online_zt(ex, g_values, t, risk_model, risk_tok, risk_device, cache, args)
        prompts.append(build_prompt(q, u, z_t, g_values, "online_unsafe", t))
        targets.append(y)
        weight_spans.append(weights)
        masked_counts.append(masked)
    return prompts, targets, weight_spans, masked_counts


def sample_timesteps(n, choices, rng):
    return [int(rng.choice(choices)) for _ in range(n)]


def compute_step(model, router, router_tok, lm_tok, examples, rng, risk_model, risk_tok, device, cache, args, train=True):
    gates = predict_g_batch(router, router_tok, examples, device, max_len=args.router_max_len)
    timesteps = sample_timesteps(len(examples), args.train_timesteps if train else args.valid_timesteps, rng)
    prompts, targets, weight_spans, masked_counts = prepare_batch_prompts(
        examples, gates, timesteps, risk_model, risk_tok, device, cache, args
    )
    batch = build_lm_batch(
        lm_tok,
        prompts,
        targets,
        device,
        args.max_source_len,
        args.max_target_len,
        weight_spans=weight_spans,
        lambda_y=args.lambda_y,
    )
    set_mlp_moe_gates(model, gates)
    labels = batch.pop("labels")
    weights = batch.pop("token_weights")
    loss = weighted_ce(model(**batch).logits, labels, weights)
    return loss, gates, timesteps, masked_counts


@torch.no_grad()
def evaluate(model, router, router_tok, lm_tok, loader, risk_model, risk_tok, device, cache, args):
    model.eval()
    losses = []
    masked = []
    t_counter = Counter()
    rng = random.Random(args.seed + 1009)
    for examples in loader:
        loss, _gates, timesteps, masked_counts = compute_step(
            model, router, router_tok, lm_tok, examples, rng, risk_model, risk_tok, device, cache, args, train=False
        )
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
            masked.extend(masked_counts)
            t_counter.update(timesteps)
    model.train()
    return {
        "loss": mean_list(losses),
        "avg_masked": mean_list(masked),
        "t": dict(t_counter),
    }


def build_config(args, injected_layers):
    return {
        "moe_impl": "residual_mlp_moe",
        "base_model": args.base_model,
        "dims": DIMS,
        "num_experts": args.num_experts,
        "bottleneck_size": args.mlp_bottleneck_size,
        "dropout": args.mlp_dropout,
        "residual_scale": args.mlp_residual_scale,
        "layers": args.mlp_layers,
        "injected_layers": injected_layers,
        "use_shared": args.mlp_use_shared,
        "activation": args.mlp_activation,
        "zero_init": args.mlp_zero_init,
        "lambda_y": args.lambda_y,
        "router_dir": args.router_dir,
        "risk_scorer_dir": args.risk_scorer_dir,
        "zt_strategy": args.zt_strategy,
    }


def print_startup(args, train_ds, valid_ds, injected_layers, model, trainable_params):
    trainable, total, pct = count_trainable_parameters(model)
    print("script: train_gemma_aspect_mlp_moe_refiner.py")
    print("moe_impl: residual_mlp_moe")
    print("base model:", args.base_model)
    print("train rows:", len(train_ds), "valid rows:", len(valid_ds))
    print("router:", args.router_dir, "frozen: True")
    print("risk scorer:", args.risk_scorer_dir, "frozen: True")
    print("num experts:", args.num_experts)
    print("mlp bottleneck:", args.mlp_bottleneck_size)
    print("mlp layers:", args.mlp_layers)
    print("injected layers:", injected_layers)
    print("lambda_y:", args.lambda_y)
    print("zt_strategy:", args.zt_strategy)
    print("train timesteps:", args.train_timesteps)
    print("valid timesteps:", args.valid_timesteps)
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    print("residual_mlp_moe params:", sum(p.numel() for p in trainable_params))
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("[trainable]", name, tuple(p.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", "--model", dest="base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--moe_impl", default="residual_mlp_moe", choices=["residual_mlp_moe"])
    ap.add_argument("--num_experts", type=int, default=6)
    ap.add_argument("--mlp_bottleneck_size", type=int, default=64)
    ap.add_argument("--mlp_dropout", type=float, default=0.05)
    ap.add_argument("--mlp_residual_scale", type=float, default=0.1)
    ap.add_argument("--mlp_layers", default="last_8")
    ap.add_argument("--mlp_activation", choices=["silu", "gelu"], default="silu")
    ap.add_argument("--mlp_use_shared", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mlp_zero_init", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", "--epochs", dest="num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", "--batch_size", dest="batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", "--grad_accum", dest="grad_accum", type=int, default=8)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=1536)
    ap.add_argument("--max_source_len", type=int, default=None)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--router_max_len", type=int, default=512)
    ap.add_argument("--lambda_y", type=float, default=0.0)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--train_timesteps", default="2,3,4")
    ap.add_argument("--valid_timesteps", default="2")
    ap.add_argument("--zt_strategy", choices=["threshold", "staged"], default="threshold")
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--lambda_mask", type=float, default=0.75)
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--mask_threshold", type=float, default=0.35)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--enable_gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    args.train_timesteps = parse_timesteps(args.train_timesteps) or [2, 3, 4]
    args.valid_timesteps = parse_timesteps(args.valid_timesteps) or [2]
    if args.max_source_len is None:
        args.max_source_len = max(128, int(args.max_length) - int(args.max_target_len))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lm_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if lm_tok.pad_token is None:
        lm_tok.pad_token = lm_tok.eos_token
    lm_tok.padding_side = "right"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("load_in_4bit:", use_4bit)
    model = load_base_model(args.base_model, use_4bit)
    model.config.use_cache = False
    if use_4bit:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.enable_gradient_checkpointing,
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)
    if args.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False

    freeze_model_parameters(model)
    injected_layers = inject_aspect_residual_mlp_moe(
        model,
        num_experts=args.num_experts,
        bottleneck_size=args.mlp_bottleneck_size,
        dropout=args.mlp_dropout,
        residual_scale=args.mlp_residual_scale,
        layers=args.mlp_layers,
        use_shared=args.mlp_use_shared,
        activation=args.mlp_activation,
        zero_init=args.mlp_zero_init,
    )
    trainable_params = residual_mlp_trainable_parameters(model)
    if not trainable_params:
        raise RuntimeError("No residual_mlp_moe trainable parameters found after injection")

    device = next(model.parameters()).device
    router_tok, router = load_router(args.router_dir, device)
    risk_tok, risk_model = load_risk_scorer(args.risk_scorer_dir, device)

    train_ds = BasePairDS(args.train_file)
    valid_ds = BasePairDS(args.valid_file)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_raw,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_raw,
        num_workers=args.num_workers,
    )

    config = build_config(args, injected_layers)
    print_startup(args, train_ds, valid_ds, injected_layers, model, trainable_params)

    opt = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    planned_updates = math.ceil(len(train_loader) / args.grad_accum) * args.num_train_epochs
    if args.max_train_steps is not None:
        planned_updates = min(planned_updates, args.max_train_steps)
    planned_updates = max(1, planned_updates)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * planned_updates), planned_updates)

    rng = random.Random(args.seed)
    train_cache = {}
    valid_cache = {}
    best = float("inf")
    step = 0
    stop = False
    model.train()
    opt.zero_grad(set_to_none=True)

    for ep in range(args.num_train_epochs):
        accum = defaultdict(list)
        t_counter = Counter()
        pbar = tqdm(train_loader, desc=f"epoch {ep + 1}/{args.num_train_epochs}")
        for i, examples in enumerate(pbar):
            loss_raw, gates, timesteps, masked_counts = compute_step(
                model, router, router_tok, lm_tok, examples, rng, risk_model, risk_tok, device, train_cache, args, train=True
            )
            loss = loss_raw / args.grad_accum
            if not torch.isfinite(loss):
                print("[warn] non-finite loss")
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            accum["loss"].append(float(loss_raw.detach().item()))
            accum["masked"].extend(float(x) for x in masked_counts)
            t_counter.update(timesteps)

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.set_postfix({
                    "loss": round(mean_list(accum["loss"]), 4),
                    "masked": round(mean_list(accum["masked"]), 2),
                    "step": step,
                })

                if step % args.eval_every == 0:
                    metrics = evaluate(model, router, router_tok, lm_tok, valid_loader, risk_model, risk_tok, device, valid_cache, args)
                    print(
                        f"[eval] step={step} loss={metrics['loss']:.4f} "
                        f"avg_masked={metrics['avg_masked']:.2f} t={metrics['t']}"
                    )
                    if metrics["loss"] < best:
                        best = metrics["loss"]
                        save_mlp_moe_adapter(model, lm_tok, out / "best", config)

                if step % args.save_every == 0:
                    save_mlp_moe_adapter(model, lm_tok, out / f"step_{step}", config)

                accum = defaultdict(list)
                t_counter = Counter()
                if args.max_train_steps is not None and step >= args.max_train_steps:
                    stop = True
                    break
        if stop:
            break

    final_metrics = evaluate(model, router, router_tok, lm_tok, valid_loader, risk_model, risk_tok, device, valid_cache, args)
    save_mlp_moe_adapter(model, lm_tok, out / "final", config)
    with open(out / "final" / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print("[done] best", best)
    print("[done] final metrics", final_metrics)
    print("[done] final ->", out / "final")


if __name__ == "__main__":
    main()
