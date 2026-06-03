import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import prepare_model_for_kbit_training

from aspect_moe_lora import (
    DIMS,
    AspectMoELinear,
    count_trainable_parameters,
    freeze_model_parameters,
    initialize_shared_lora_from_peft,
    normalize_gates,
    save_moe_adapter,
    set_moe_gates,
    wrap_aspect_moe_layers,
)
from prepare_overleaf_infermatch_data import get_field, monotonic_align, split_sentences


DIM2ID = {d: i for i, d in enumerate(DIMS)}
DIM_ALIASES = {
    "medical": "medical_advice",
    "medical_boundary": "medical_advice",
    "medical-advice": "medical_advice",
    "factual": "factual_consistency",
    "fact": "factual_consistency",
    "factual-consistency": "factual_consistency",
    "overall": "overall_quality",
    "overall-quality": "overall_quality",
}


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def normalize_dim(value):
    value = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return DIM_ALIASES.get(value, value)


def row_question(ex):
    return get_field(ex, "question", "query", "input_question", "user_question")


def row_unsafe(ex):
    return get_field(ex, "unsafe_response", "corrupted_response", "response", "unsafe", "bad_response")


def row_safe(ex):
    return get_field(ex, "safe_response", "target", "target_response", "output", "safe")


def gold_dimensions(ex):
    if isinstance(ex.get("target_dimensions"), list):
        return ex["target_dimensions"]
    for key in ["target_dimension", "dimension", "violation_dimension", "violated_dimension", "dim", "condition_dim"]:
        value = ex.get(key)
        if value is not None and str(value).strip():
            return [value]
    return []


def gold_vector_from_row(ex):
    for key in ["d", "g_gold", "gold_aspect", "aspect_vector", "labels"]:
        value = ex.get(key)
        if isinstance(value, list) and len(value) >= len(DIMS):
            vals = [float(x) for x in value[: len(DIMS)]]
            if any(vals):
                return vals
        if isinstance(value, dict):
            vals = [float(value.get(dim, 0.0)) for dim in DIMS]
            if any(vals):
                return vals

    vec = [0.0] * len(DIMS)
    dims = gold_dimensions(ex)
    if not dims:
        vec[DIM2ID["overall_quality"]] = 1.0
        return vec
    for raw in dims:
        dim = normalize_dim(raw)
        vec[DIM2ID.get(dim, DIM2ID["overall_quality"])] = 1.0
    if not any(vec):
        vec[DIM2ID["overall_quality"]] = 1.0
    return vec


def router_text(q, u):
    return (
        "Question:\n"
        + q.strip()
        + "\n\nUnsafe response:\n"
        + u.strip()
        + "\n\nTask: identify all violated mental-health response quality dimensions."
    )


def risk_text(q, span):
    return (
        "Question:\n"
        + q.strip()
        + "\n\nCandidate span:\n"
        + span.strip()
        + "\n\nTask: predict which counseling quality dimensions this span may violate."
    )


def aspect_text(g_values):
    return ", ".join(f"{DIMS[i]}={float(g_values[i]):.2f}" for i in range(len(DIMS)))


def build_prompt(q, u, z_t, g_values, source, t):
    z = str(z_t or "").strip()
    if not z:
        z = "No draft. Rewrite directly from the unsafe response."
    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n"
        "Do not copy blaming, diagnostic, toxic, or unsupported wording from the unsafe response.\n\n"
        f"Aspect scores:\n{aspect_text(g_values)}\n\n"
        f"Question:\n{q.strip()}\n\n"
        f"Unsafe response to fix:\n{u.strip()}\n\n"
        f"Draft to revise:\n{z}\n\n"
        f"Corruption: {source}, t={t}\n\n"
        "Safe response:\n"
    )


def example_cache_key(q, u, y):
    payload = (q or "") + "\n<UNSAFE>\n" + (u or "") + "\n<SAFE>\n" + (y or "")
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def rl(g, risk_vec):
    return max(float(g[k]) * float(risk_vec[k]) for k in range(len(DIMS)))


def score_spans(risk_model, risk_tok, q, spans, device, max_len=384):
    if not spans:
        return []
    enc = risk_tok(
        [risk_text(q, span) for span in spans],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(risk_model(**enc).logits.float()).detach().cpu().tolist()
    return [[float(x) for x in row[: len(DIMS)]] for row in probs]


def alignment_with_risk(ex, risk_model, risk_tok, device, cache):
    q = row_question(ex)
    u = row_unsafe(ex)
    y = row_safe(ex)
    key = example_cache_key(q, u, y)
    if key in cache:
        return cache[key]

    pairs = monotonic_align(u, y)
    unsafe_spans = [p.get("a", "") for p in pairs if p.get("a", "").strip()]
    scored = score_spans(risk_model, risk_tok, q, unsafe_spans, device)
    score_iter = iter(scored)
    enriched = []
    for pair in pairs:
        item = dict(pair)
        if item.get("a", "").strip():
            item["risk_vec"] = next(score_iter, [0.0] * len(DIMS))
        else:
            item["risk_vec"] = [0.0] * len(DIMS)
        enriched.append(item)
    if not enriched:
        fallback = split_sentences(u) or [u]
        enriched = [
            {"a": span, "b": "", "op": "delete", "align_score": 0.0, "risk_vec": rv}
            for span, rv in zip(fallback, score_spans(risk_model, risk_tok, q, fallback, device))
        ]

    cache[key] = enriched
    return enriched


def weight_record(pair, risk, state, p_mask=0.0):
    return {
        "safe_span": pair.get("b", ""),
        "unsafe_span": pair.get("a", ""),
        "risk": float(risk),
        "state": state,
        "op": pair.get("op", ""),
        "p_mask": float(p_mask),
        "align_score": float(pair.get("align_score", 0.0)),
    }


def build_zt_threshold(pairs, g_values, t, T, args):
    beta = t / float(T)
    parts = []
    infos = []
    weights = []
    masked = 0

    for pair in pairs:
        a = pair.get("a", "")
        b = pair.get("b", "")
        if not a:
            continue

        risk = rl(g_values, pair.get("risk_vec", [0.0] * len(DIMS)))
        pi = clamp01(args.rho + args.lambda_mask * beta * risk)
        p_mask = beta * pi
        should_mask = (p_mask >= args.mask_threshold) or (risk >= args.risk_threshold and t >= 2)
        if should_mask:
            parts.append(args.mask_token)
            state = "MASK"
            masked += 1
        else:
            parts.append(a)
            state = "UNSAFE"
        infos.append({"span": a, "r_l_g": risk, "p_mask": p_mask, "state": state})
        if b:
            weights.append(weight_record(pair, risk, state, p_mask))

    z_t = " ".join(parts).strip()
    return z_t, infos, weights, masked


def staged_mask_count(n, t, t2_frac=0.33, t3_frac=0.66):
    if t <= 1 or n == 0:
        return 0
    if t == 2:
        return max(1, min(n, math.ceil(t2_frac * n)))
    if t == 3:
        return max(1, min(n, math.ceil(t3_frac * n)))
    return n


def build_zt_staged(pairs, g_values, t, args):
    candidates = []
    for idx, pair in enumerate(pairs):
        a = pair.get("a", "")
        if not a:
            continue
        risk = rl(g_values, pair.get("risk_vec", [0.0] * len(DIMS)))
        candidates.append((idx, pair, risk))

    ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
    mask_count = staged_mask_count(len(ranked), t)
    mask_indices = {idx for idx, _, _ in ranked[:mask_count]}
    ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(ranked)}

    parts = []
    infos = []
    weights = []
    for idx, pair, risk in candidates:
        if idx in mask_indices:
            parts.append(args.mask_token)
            state = "MASK"
        else:
            parts.append(pair.get("a", ""))
            state = "UNSAFE"
        infos.append({
            "span": pair.get("a", ""),
            "r_l_g": risk,
            "rank": ranks.get(idx),
            "state": state,
            "strategy": "staged",
        })
        if pair.get("b"):
            weights.append(weight_record(pair, risk, state, 1.0 if state == "MASK" else 0.0))

    z_t = " ".join(parts).strip()
    return z_t, infos, weights, mask_count


def build_online_zt(ex, g_values, t, risk_model, risk_tok, device, cache, args):
    pairs = alignment_with_risk(ex, risk_model, risk_tok, device, cache)
    if args.zt_strategy == "staged":
        z_t, infos, weights, masked = build_zt_staged(pairs, g_values, t, args)
    else:
        z_t, infos, weights, masked = build_zt_threshold(pairs, g_values, t, args.T, args)
    if not z_t:
        z_t = row_unsafe(ex)
    return z_t, infos, weights, masked


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


def target_token_weights(tok, target, spans, lambda_y):
    try:
        enc = tok(
            target,
            add_special_tokens=False,
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
    except Exception:
        ids = tok(target, add_special_tokens=False, truncation=True)["input_ids"]
        offsets = [(0, 0)] * len(ids)

    ranges = find_ranges(target, spans)
    weights = []
    for a, b in offsets:
        weight = 1.0
        for s, e, risk in ranges:
            if not (b <= s or a >= e):
                weight = max(weight, 1.0 + float(lambda_y) * risk)
        weights.append(weight)
    return weights


def build_lm_batch(tok, prompts, targets, device, max_source_len, max_target_len, weight_spans=None, lambda_y=0.0):
    batch_ids = []
    batch_masks = []
    batch_labels = []
    batch_weights = []
    pad = tok.pad_token_id

    for i, (prompt, raw_target) in enumerate(zip(prompts, targets)):
        target = raw_target.strip()
        if tok.eos_token and not target.endswith(tok.eos_token):
            target += tok.eos_token

        pids = tok(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_source_len,
        )["input_ids"]
        tids = tok(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_len,
        )["input_ids"]

        ids = pids + tids
        labels = [-100] * len(pids) + tids
        if lambda_y > 0 and weight_spans is not None:
            tw = target_token_weights(tok, target, weight_spans[i], lambda_y)[: len(tids)]
        else:
            tw = [1.0] * len(tids)
        weights = [0.0] * len(pids) + tw

        batch_ids.append(ids)
        batch_masks.append([1] * len(ids))
        batch_labels.append(labels)
        batch_weights.append(weights)

    max_len = max(len(x) for x in batch_ids)
    for ids, masks, labels, weights in zip(batch_ids, batch_masks, batch_labels, batch_weights):
        n = max_len - len(ids)
        ids.extend([pad] * n)
        masks.extend([0] * n)
        labels.extend([-100] * n)
        weights.extend([0.0] * n)

    return {
        "input_ids": torch.tensor(batch_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_masks, dtype=torch.long, device=device),
        "labels": torch.tensor(batch_labels, dtype=torch.long, device=device),
        "token_weights": torch.tensor(batch_weights, dtype=torch.float, device=device),
    }


def weighted_ce(logits, labels, token_weights):
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


class BasePairDS(Dataset):
    def __init__(self, path):
        self.rows = read_jsonl(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = dict(self.rows[idx])
        row["_idx"] = idx
        return row


def collate_raw(batch):
    return batch


def encode_router_batch(router_tok, examples, device, max_len=512):
    texts = [router_text(row_question(ex), row_unsafe(ex)) for ex in examples]
    return router_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)


def gold_tensor(examples, device):
    return torch.tensor([gold_vector_from_row(ex) for ex in examples], dtype=torch.float32, device=device)


def tensor_to_g_values(g_tensor):
    return g_tensor.detach().float().cpu().tolist()


@dataclass
class StepBatch:
    total: torch.Tensor
    loss_den: torch.Tensor
    loss_sft: torch.Tensor
    loss_router: torch.Tensor
    g_pred: torch.Tensor
    g_used: torch.Tensor
    tf_mask: torch.Tensor
    timesteps: list
    masked_counts: list


def sample_timesteps(n, choices, rng):
    return [int(rng.choice(choices)) for _ in range(n)]


def prepare_prompts(examples, g_used, timesteps, risk_model, risk_tok, risk_device, cache, args):
    den_prompts = []
    sft_prompts = []
    targets = []
    weight_spans = []
    masked_counts = []
    g_lists = tensor_to_g_values(g_used)

    for ex, g_values, t in zip(examples, g_lists, timesteps):
        q = row_question(ex)
        u = row_unsafe(ex)
        y = row_safe(ex)
        z_t, _infos, weights, masked = build_online_zt(ex, g_values, t, risk_model, risk_tok, risk_device, cache, args)
        den_prompts.append(build_prompt(q, u, z_t, g_values, "online_unsafe", t))
        sft_prompts.append(build_prompt(q, u, "", g_values, "direct", 0))
        targets.append(y)
        weight_spans.append(weights)
        masked_counts.append(masked)

    return den_prompts, sft_prompts, targets, weight_spans, masked_counts


def compute_joint_step(
    model,
    router,
    router_tok,
    lm_tok,
    examples,
    rng,
    risk_model,
    risk_tok,
    device,
    risk_device,
    cache,
    args,
    train=True,
    valid_g_source="pred",
    valid_timesteps=None,
):
    router_inputs = encode_router_batch(router_tok, examples, device, max_len=args.router_max_len)
    d_gold = gold_tensor(examples, device)
    router_logits = router(**router_inputs).logits.float()
    g_pred = torch.sigmoid(router_logits)
    loss_router = F.binary_cross_entropy_with_logits(router_logits, d_gold)

    if train:
        tf_mask = (torch.rand((len(examples), 1), device=device) < args.aspect_tf_prob).float()
        g_used = tf_mask * d_gold + (1.0 - tf_mask) * g_pred
        timesteps = sample_timesteps(len(examples), args.train_timesteps, rng)
    else:
        tf_mask = torch.zeros((len(examples), 1), device=device)
        if valid_g_source == "gold":
            tf_mask.fill_(1.0)
            g_used = d_gold
        else:
            g_used = g_pred
        timesteps = list(valid_timesteps or [args.valid_t])
        if len(timesteps) == 1:
            timesteps = timesteps * len(examples)
        elif len(timesteps) != len(examples):
            timesteps = [timesteps[i % len(timesteps)] for i in range(len(examples))]

    den_prompts, sft_prompts, targets, weight_spans, masked_counts = prepare_prompts(
        examples, g_used, timesteps, risk_model, risk_tok, risk_device, cache, args
    )
    den_batch = build_lm_batch(
        lm_tok,
        den_prompts,
        targets,
        device,
        args.max_source_len,
        args.max_target_len,
        weight_spans=weight_spans,
        lambda_y=args.lambda_y,
    )
    sft_batch = build_lm_batch(
        lm_tok,
        sft_prompts,
        targets,
        device,
        args.max_source_len,
        args.max_target_len,
        weight_spans=None,
        lambda_y=0.0,
    )

    set_moe_gates(model, g_used, detach=False if train else True)
    den_labels = den_batch.pop("labels")
    den_weights = den_batch.pop("token_weights")
    loss_den = weighted_ce(model(**den_batch).logits, den_labels, den_weights)

    set_moe_gates(model, g_used, detach=False if train else True)
    sft_labels = sft_batch.pop("labels")
    sft_weights = sft_batch.pop("token_weights")
    loss_sft = weighted_ce(model(**sft_batch).logits, sft_labels, sft_weights)

    total = loss_den + args.lambda_sft * loss_sft + args.lambda_router * loss_router
    return StepBatch(
        total=total,
        loss_den=loss_den,
        loss_sft=loss_sft,
        loss_router=loss_router,
        g_pred=g_pred,
        g_used=g_used,
        tf_mask=tf_mask,
        timesteps=timesteps,
        masked_counts=masked_counts,
    )


def mean_list(values):
    return sum(values) / max(1, len(values))


def summarize_g(tensor):
    vals = tensor.detach().float().mean(dim=0).cpu().tolist()
    return {DIMS[i]: round(float(vals[i]), 4) for i in range(len(DIMS))}


@torch.no_grad()
def evaluate(model, router, router_tok, lm_tok, loader, risk_model, risk_tok, device, risk_device, cache, args):
    model.eval()
    router.eval()
    losses = defaultdict(list)
    t_counter = Counter()
    masked = []
    rng = random.Random(args.seed + 1009)
    valid_ts = args.valid_timesteps or [args.valid_t]

    for examples in loader:
        for vt in valid_ts:
            step = compute_joint_step(
                model,
                router,
                router_tok,
                lm_tok,
                examples,
                rng,
                risk_model,
                risk_tok,
                device,
                risk_device,
                cache,
                args,
                train=False,
                valid_g_source=args.valid_g_source,
                valid_timesteps=[vt],
            )
            if torch.isfinite(step.total):
                losses["total"].append(float(step.total.item()))
                losses["den"].append(float(step.loss_den.item()))
                losses["sft"].append(float(step.loss_sft.item()))
                losses["router"].append(float(step.loss_router.item()))
                masked.extend(step.masked_counts)
                t_counter.update(step.timesteps)

    model.train()
    router.train()
    return {
        "total": mean_list(losses["total"]),
        "den": mean_list(losses["den"]),
        "sft": mean_list(losses["sft"]),
        "router": mean_list(losses["router"]),
        "avg_masked": mean_list(masked),
        "t": dict(t_counter),
    }


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


def load_router(args, device):
    source = args.router_init_dir or args.router_model
    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if args.router_init_dir:
        model = AutoModelForSequenceClassification.from_pretrained(args.router_init_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.router_model,
            num_labels=len(DIMS),
            problem_type="multi_label_classification",
        )
    return tok, model.to(device)


def load_risk_scorer(args, device):
    tok = AutoTokenizer.from_pretrained(args.risk_scorer_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(args.risk_scorer_dir).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def build_moe_config(args, wrapped_names, init_report=None):
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
        "wrapped_module_names": wrapped_names,
        "init_shared_adapter_dir": args.init_shared_adapter_dir,
        "freeze_shared_lora": False,
        "full_joint": True,
        "moe_gates_detach": False,
        "lambda_sft": args.lambda_sft,
        "lambda_router": args.lambda_router,
        "lambda_y": args.lambda_y,
        "aspect_tf_prob": args.aspect_tf_prob,
        "risk_scorer_dir": args.risk_scorer_dir,
        "risk_scorer_frozen": True,
        "init_shared_report": init_report,
    }


def save_joint(model, lm_tok, router, router_tok, output_dir, config, args, metrics=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_moe_adapter(model, lm_tok, output_dir, config)
    router_dir = output_dir / "router"
    router.save_pretrained(router_dir)
    router_tok.save_pretrained(router_dir)
    payload = {
        "args": vars(args),
        "config": config,
        "metrics": metrics or {},
    }
    with open(output_dir / "joint_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("[save] joint checkpoint ->", output_dir)


def split_moe_params(model):
    shared = []
    expert = []
    for module in model.modules():
        if isinstance(module, AspectMoELinear):
            shared.extend([module.A_shared, module.B_shared])
            expert.extend([module.A_expert, module.B_expert])
    return [p for p in shared if p.requires_grad], [p for p in expert if p.requires_grad]


def parse_timesteps(value):
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def print_startup(args, train_ds, valid_ds, wrapped_names, model, router, shared_params, expert_params):
    trainable, total, pct = count_trainable_parameters(model)
    router_trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print("script: train_gemma_full_joint_denoiser.py")
    print("base model:", args.model)
    print("train rows:", len(train_ds), "valid rows:", len(valid_ds))
    print("source: base pair JSON, not precomputed denoising JSON")
    print("router init:", args.router_init_dir or args.router_model)
    print("risk scorer:", args.risk_scorer_dir, "frozen:", True)
    print("init shared adapter:", args.init_shared_adapter_dir or "")
    print("wrapped modules:", len(wrapped_names))
    print("first wrapped modules:", wrapped_names[:10])
    print("shared LoRA trainable:", all(p.requires_grad for p in shared_params))
    print("expert LoRA trainable:", all(p.requires_grad for p in expert_params))
    print(f"moe trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    print(f"router trainable params: {router_trainable:,}")
    print("shared params:", sum(p.numel() for p in shared_params))
    print("expert params:", sum(p.numel() for p in expert_params))
    print("lambda_sft:", args.lambda_sft)
    print("lambda_router:", args.lambda_router)
    print("lambda_y:", args.lambda_y)
    print("aspect_tf_prob:", args.aspect_tf_prob)
    print("zt_strategy:", args.zt_strategy)
    print("train timesteps:", args.train_timesteps)
    print("valid g source:", args.valid_g_source, "valid timesteps:", args.valid_timesteps)
    print("MoE gates detach: False")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--router_model", default="bert-base-uncased")
    ap.add_argument("--router_init_dir", default="")
    ap.add_argument("--risk_scorer_dir", default="outputs/models/span_risk_multilabel_v1/best")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--lr_lora", type=float, default=5e-6)
    ap.add_argument("--lr_expert", type=float, default=1e-5)
    ap.add_argument("--lr_router", type=float, default=1e-5)
    ap.add_argument("--r_shared", type=int, default=8)
    ap.add_argument("--r_expert", type=int, default=8)
    ap.add_argument("--alpha_shared", type=int, default=16)
    ap.add_argument("--alpha_expert", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--moe_eps", type=float, default=1e-4)
    ap.add_argument("--target_regex", default=r".*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
    ap.add_argument("--init_shared_adapter_dir", default="")
    ap.add_argument("--allow_partial_shared_init", action="store_true")
    ap.add_argument("--no_preserve_init_scale", action="store_true")
    ap.add_argument("--lambda_sft", type=float, default=0.3)
    ap.add_argument("--lambda_router", type=float, default=0.1)
    ap.add_argument("--lambda_y", type=float, default=0.0)
    ap.add_argument("--aspect_tf_prob", type=float, default=0.5)
    ap.add_argument("--lambda_risk", type=float, default=0.0)
    ap.add_argument("--train_risk_scorer", action="store_true")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--train_timesteps", default="2,3,4")
    ap.add_argument("--valid_timesteps", default="2")
    ap.add_argument("--valid_t", type=int, default=2)
    ap.add_argument("--valid_g_source", choices=["pred", "gold"], default="pred")
    ap.add_argument("--zt_strategy", choices=["threshold", "staged"], default="threshold")
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--lambda_mask", type=float, default=0.75)
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--mask_threshold", type=float, default=0.35)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--router_max_len", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    if args.train_risk_scorer or args.lambda_risk != 0.0:
        raise ValueError("Full joint v1 keeps the span-risk scorer frozen; train_risk_scorer/lambda_risk are not supported yet.")

    args.aspect_tf_prob = clamp01(args.aspect_tf_prob)
    args.train_timesteps = parse_timesteps(args.train_timesteps)
    args.valid_timesteps = parse_timesteps(args.valid_timesteps)
    if not args.train_timesteps:
        args.train_timesteps = [2, 3, 4]
    if not args.valid_timesteps:
        args.valid_timesteps = [args.valid_t]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lm_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if lm_tok.pad_token is None:
        lm_tok.pad_token = lm_tok.eos_token
    lm_tok.padding_side = "right"

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    print("load_in_4bit:", use_4bit)
    model = load_base_model(args.model, use_4bit)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    freeze_model_parameters(model)

    base_config = {
        "dims": DIMS,
        "target_regex": args.target_regex,
        "r_shared": args.r_shared,
        "r_expert": args.r_expert,
        "alpha_shared": args.alpha_shared,
        "alpha_expert": args.alpha_expert,
        "dropout": args.dropout,
        "moe_eps": args.moe_eps,
    }
    wrapped_names = wrap_aspect_moe_layers(model, base_config)
    init_report = None
    if args.init_shared_adapter_dir:
        init_report = initialize_shared_lora_from_peft(
            model,
            args.init_shared_adapter_dir,
            require_all=not args.allow_partial_shared_init,
            preserve_scale=not args.no_preserve_init_scale,
        )
    config = build_moe_config(args, wrapped_names, init_report)

    device = next(model.parameters()).device
    router_tok, router = load_router(args, device)
    risk_tok, risk_model = load_risk_scorer(args, device)
    risk_device = device

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

    shared_params, expert_params = split_moe_params(model)
    router_params = [p for p in router.parameters() if p.requires_grad]
    opt_groups = [
        {"params": shared_params, "lr": args.lr_lora, "name": "shared_lora"},
        {"params": expert_params, "lr": args.lr_expert, "name": "expert_lora"},
        {"params": router_params, "lr": args.lr_router, "name": "router"},
    ]
    params = shared_params + expert_params + router_params
    opt = torch.optim.AdamW(opt_groups, weight_decay=0.01)
    planned_updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    if args.max_train_steps is not None:
        planned_updates = min(planned_updates, args.max_train_steps)
    planned_updates = max(1, planned_updates)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * planned_updates), planned_updates)

    print_startup(args, train_ds, valid_ds, wrapped_names, model, router, shared_params, expert_params)

    train_cache = {}
    valid_cache = {}
    rng = random.Random(args.seed)
    best = float("inf")
    step = 0
    stop = False
    model.train()
    router.train()
    opt.zero_grad(set_to_none=True)

    for ep in range(args.epochs):
        accum = defaultdict(list)
        t_counter = Counter()
        pbar = tqdm(train_loader, desc=f"epoch {ep + 1}/{args.epochs}")
        for i, examples in enumerate(pbar):
            result = compute_joint_step(
                model,
                router,
                router_tok,
                lm_tok,
                examples,
                rng,
                risk_model,
                risk_tok,
                device,
                risk_device,
                train_cache,
                args,
                train=True,
            )
            loss = result.total / args.grad_accum
            if not torch.isfinite(loss):
                print("[warn] non-finite loss")
                opt.zero_grad(set_to_none=True)
                continue

            loss.backward()
            accum["total"].append(float(result.total.detach().item()))
            accum["den"].append(float(result.loss_den.detach().item()))
            accum["sft"].append(float(result.loss_sft.detach().item()))
            accum["router"].append(float(result.loss_router.detach().item()))
            accum["gold_frac"].append(float(result.tf_mask.detach().mean().item()))
            accum["masked"].extend(float(x) for x in result.masked_counts)
            t_counter.update(result.timesteps)

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                postfix = {
                    "loss": round(mean_list(accum["total"]), 4),
                    "den": round(mean_list(accum["den"]), 4),
                    "sft": round(mean_list(accum["sft"]), 4),
                    "router": round(mean_list(accum["router"]), 4),
                    "gold_g": round(mean_list(accum["gold_frac"]), 3),
                    "masked": round(mean_list(accum["masked"]), 2),
                    "step": step,
                }
                pbar.set_postfix(postfix)

                if step % args.eval_every == 0:
                    metrics = evaluate(model, router, router_tok, lm_tok, valid_loader, risk_model, risk_tok, device, risk_device, valid_cache, args)
                    print(
                        f"[eval] step={step} total={metrics['total']:.4f} den={metrics['den']:.4f} "
                        f"sft={metrics['sft']:.4f} router={metrics['router']:.4f} "
                        f"avg_masked={metrics['avg_masked']:.2f} t={metrics['t']}"
                    )
                    print("[debug] mean_g_pred:", summarize_g(result.g_pred))
                    print("[debug] mean_g_used:", summarize_g(result.g_used))
                    if metrics["total"] < best:
                        best = metrics["total"]
                        save_joint(model, lm_tok, router, router_tok, out / "best", config, args, metrics)

                if step % args.save_every == 0:
                    save_joint(model, lm_tok, router, router_tok, out / f"step_{step}", config, args)

                accum = defaultdict(list)
                t_counter = Counter()
                if args.max_train_steps is not None and step >= args.max_train_steps:
                    stop = True
                    break

        if stop:
            break

    final_metrics = evaluate(model, router, router_tok, lm_tok, valid_loader, risk_model, risk_tok, device, risk_device, valid_cache, args)
    save_joint(model, lm_tok, router, router_tok, out / "final", config, args, final_metrics)
    print("[done] best", best)
    print("[done] final metrics", final_metrics)
    print("[done] final ->", out / "final")


if __name__ == "__main__":
    main()
