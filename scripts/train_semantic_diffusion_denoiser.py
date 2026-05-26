import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DIMENSIONS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows, val_ratio=0.1):
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[:-n_val], rows[-n_val:]


def make_condition_text(ex):
    return f"""User question:
{ex["question"]}

Corrupted response:
{ex["unsafe_response"]}

Known violation dimension:
{ex["target_dimension"]}""".strip()


class DiffusionDenoiseDataset(Dataset):
    def __init__(self, rows, tokenizer, max_cond_len=256, max_resp_len=160):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_cond_len = max_cond_len
        self.max_resp_len = max_resp_len

    def __len__(self):
        return len(self.rows)

    def encode_text(self, text, max_len):
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    def __getitem__(self, idx):
        ex = self.rows[idx]

        cond_ids, cond_mask = self.encode_text(make_condition_text(ex), self.max_cond_len)
        safe_ids, safe_mask = self.encode_text(ex["safe_response"], self.max_resp_len)
        unsafe_ids, unsafe_mask = self.encode_text(ex["unsafe_response"], self.max_resp_len)

        dim = ex.get("target_dimension", "overall_quality")
        dim_id = DIMENSIONS.index(dim) if dim in DIMENSIONS else 0

        return {
            "cond_ids": torch.tensor(cond_ids, dtype=torch.long),
            "cond_mask": torch.tensor(cond_mask, dtype=torch.float),
            "safe_ids": torch.tensor(safe_ids, dtype=torch.long),
            "safe_mask": torch.tensor(safe_mask, dtype=torch.float),
            "unsafe_ids": torch.tensor(unsafe_ids, dtype=torch.long),
            "unsafe_mask": torch.tensor(unsafe_mask, dtype=torch.float),
            "dim_id": torch.tensor(dim_id, dtype=torch.long),
        }


def sinusoidal_timestep_embedding(timesteps, dim):
    """
    timesteps: [B]
    return: [B, dim]
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SemanticDiffusionDenoiser(nn.Module):
    """
    Lightweight Transformer denoiser.

    Input:
      y_t: corrupted/noisy response embedding [B, L, D]
      cond: condition embedding pooled from question + unsafe_response + dimension
      t: timestep

    Output:
      predicted clean safe response embedding y_0_hat [B, L, D]

    This is not Causal LM SFT.
    Loss is MSE over response embeddings.
    """
    def __init__(
        self,
        embed_dim,
        num_dims=6,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        max_resp_len=160,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_resp_len = max_resp_len

        self.dim_embed = nn.Embedding(num_dims, embed_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, max_resp_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, y_t, cond_vec, dim_id, t, resp_padding_mask=None):
        B, L, D = y_t.shape

        t_emb = sinusoidal_timestep_embedding(t, D)
        t_emb = self.t_proj(t_emb).unsqueeze(1)

        dim_emb = self.dim_embed(dim_id).unsqueeze(1)
        cond_emb = self.cond_proj(cond_vec).unsqueeze(1)

        h = y_t + self.pos_embed[:, :L, :] + t_emb + dim_emb + cond_emb

        # Transformer expects True for padding positions.
        key_padding_mask = None
        if resp_padding_mask is not None:
            key_padding_mask = resp_padding_mask.bool()

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.out_norm(h)
        return self.out(h)


def masked_mean(x, mask):
    """
    x: [B, L, D]
    mask: [B, L], 1 for real tokens
    """
    mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def make_semantic_noisy_sample(x0, x_unsafe, t, T, noise_scale=0.05):
    """
    Semantic corruption path:
      y_t = (1 - lambda_t) * safe_embedding + lambda_t * unsafe_embedding + small Gaussian smoothing

    lambda_t increases with t.
    t=0   : mostly safe
    t=T-1 : mostly unsafe/corrupted

    This uses the LLM-generated unsafe response as semantic noise endpoint.
    """
    B = x0.shape[0]
    device = x0.device

    lam = (t.float() / max(T - 1, 1)).view(B, 1, 1)
    eps = torch.randn_like(x0)

    # small Gaussian smoothing, not the main corruption signal
    y_t = (1.0 - lam) * x0 + lam * x_unsafe + noise_scale * lam * eps
    return y_t


def train_one_epoch(
    model,
    embed_layer,
    loader,
    optimizer,
    device,
    T,
    noise_scale,
    grad_clip,
):
    model.train()
    total = 0.0
    steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            cond_emb = embed_layer(batch["cond_ids"])
            safe_emb = embed_layer(batch["safe_ids"])
            unsafe_emb = embed_layer(batch["unsafe_ids"])

            cond_vec = masked_mean(cond_emb, batch["cond_mask"])

        B = safe_emb.size(0)
        t = torch.randint(low=0, high=T, size=(B,), device=device)

        y_t = make_semantic_noisy_sample(
            safe_emb,
            unsafe_emb,
            t=t,
            T=T,
            noise_scale=noise_scale,
        )

        resp_padding_mask = batch["safe_mask"] == 0
        pred = model(
            y_t=y_t,
            cond_vec=cond_vec,
            dim_id=batch["dim_id"],
            t=t,
            resp_padding_mask=resp_padding_mask,
        )

        mask = batch["safe_mask"].unsqueeze(-1)
        mse = ((pred - safe_emb) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)

        optimizer.zero_grad(set_to_none=True)
        mse.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total += mse.item()
        steps += 1
        pbar.set_postfix(loss=f"{mse.item():.4f}")

    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, embed_layer, loader, device, T, noise_scale):
    model.eval()
    total = 0.0
    steps = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        cond_emb = embed_layer(batch["cond_ids"])
        safe_emb = embed_layer(batch["safe_ids"])
        unsafe_emb = embed_layer(batch["unsafe_ids"])
        cond_vec = masked_mean(cond_emb, batch["cond_mask"])

        B = safe_emb.size(0)
        # evaluate harder corruption: near terminal semantic-noise state
        t = torch.full((B,), T - 1, device=device, dtype=torch.long)

        y_t = make_semantic_noisy_sample(
            safe_emb,
            unsafe_emb,
            t=t,
            T=T,
            noise_scale=noise_scale,
        )

        pred = model(
            y_t=y_t,
            cond_vec=cond_vec,
            dim_id=batch["dim_id"],
            t=t,
            resp_padding_mask=batch["safe_mask"] == 0,
        )

        mask = batch["safe_mask"].unsqueeze(-1)
        mse = ((pred - safe_emb) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)

        total += mse.item()
        steps += 1

    return total / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default="outputs/models/semantic_diffusion_denoiser_v0")
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--max_cond_len", type=int, default=256)
    parser.add_argument("--max_resp_len", type=int, default=160)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading rows...")
    rows = load_jsonl(args.train_file)
    train_rows, val_rows = split_rows(rows)
    print("rows:", len(rows), "train:", len(train_rows), "val:", len(val_rows))

    print("Loading tokenizer/base embeddings:", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base.eval()

    embed_layer = base.get_input_embeddings()
    embed_dim = embed_layer.embedding_dim
    device = next(embed_layer.parameters()).device
    print("embed_dim:", embed_dim, "device:", device)

    # Freeze base model / embedding table
    for p in base.parameters():
        p.requires_grad = False

    train_ds = DiffusionDenoiseDataset(
        train_rows, tokenizer, args.max_cond_len, args.max_resp_len
    )
    val_ds = DiffusionDenoiseDataset(
        val_rows, tokenizer, args.max_cond_len, args.max_resp_len
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SemanticDiffusionDenoiser(
        embed_dim=embed_dim,
        num_dims=len(DIMENSIONS),
        n_layers=args.layers,
        n_heads=args.heads,
        max_resp_len=args.max_resp_len,
    ).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            embed_layer=embed_layer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            T=args.T,
            noise_scale=args.noise_scale,
            grad_clip=args.grad_clip,
        )
        val_loss = evaluate(
            model=model,
            embed_layer=embed_layer,
            loader=val_loader,
            device=device,
            T=args.T,
            noise_scale=args.noise_scale,
        )

        rec = {"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss}
        history.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "embed_dim": embed_dim,
                "dimensions": DIMENSIONS,
                "best_val_mse": best_val,
            }
            torch.save(ckpt, f"{args.output_dir}/best.pt")
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint to {args.output_dir}/best.pt")

    with open(f"{args.output_dir}/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("Done. Best val MSE:", best_val)


if __name__ == "__main__":
    main()
