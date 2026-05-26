import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_condition_text(ex):
    return f"""User question:
{ex["question"]}

Corrupted response:
{ex["unsafe_response"]}

Known violation dimension:
{ex["target_dimension"]}""".strip()


def sinusoidal_timestep_embedding(timesteps, dim):
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

        key_padding_mask = None
        if resp_padding_mask is not None:
            key_padding_mask = resp_padding_mask.bool()

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.out_norm(h)
        return self.out(h)


def encode(tokenizer, text, max_len, device):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].float().to(device)


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def round_embeddings_to_tokens(z, embed_weight, tokenizer, topk=1):
    """
    z: [L, D]
    nearest-neighbor rounding to token embeddings.
    """
    z_norm = F.normalize(z.float(), dim=-1)
    w_norm = F.normalize(embed_weight.float(), dim=-1)
    sims = z_norm @ w_norm.T
    ids = sims.argmax(dim=-1).tolist()

    # remove special/pad tokens
    bad_ids = set()
    for tok in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
        if tok is not None:
            bad_ids.add(tok)

    cleaned = []
    for tid in ids:
        if tid in bad_ids:
            continue
        cleaned.append(tid)

    text = tokenizer.decode(cleaned, skip_special_tokens=True)
    return " ".join(text.split()).strip()


@torch.no_grad()
def semantic_reverse_sample(model, unsafe_emb, cond_vec, dim_id, unsafe_mask, T):
    """
    Start from unsafe response embedding as terminal semantic-corrupted state.
    Iteratively predict clean y0 and move toward it.
    This is a simple DDIM-like deterministic reverse path for v0.
    """
    z = unsafe_emb.clone()
    B = z.shape[0]

    for step in reversed(range(T)):
        t = torch.full((B,), step, device=z.device, dtype=torch.long)
        pred_x0 = model(
            y_t=z,
            cond_vec=cond_vec,
            dim_id=dim_id,
            t=t,
            resp_padding_mask=unsafe_mask == 0,
        )

        if step > 0:
            # deterministic interpolation toward predicted clean embedding
            lam_prev = step / max(T - 1, 1)
            z = (1.0 - lam_prev) * pred_x0 + lam_prev * unsafe_emb
        else:
            z = pred_x0

    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", default="outputs/models/semantic_diffusion_denoiser_v0/best.pt")
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--max_cond_len", type=int, default=256)
    parser.add_argument("--max_resp_len", type=int, default=160)
    parser.add_argument("--T", type=int, default=20)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]
    embed_dim = ckpt["embed_dim"]

    print("Loading tokenizer/base model:", args.base_model)
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
    device = next(embed_layer.parameters()).device

    for p in base.parameters():
        p.requires_grad = False

    model = SemanticDiffusionDenoiser(
        embed_dim=embed_dim,
        num_dims=len(DIMENSIONS),
        n_layers=train_args.get("layers", 4),
        n_heads=train_args.get("heads", 8),
        max_resp_len=train_args.get("max_resp_len", args.max_resp_len),
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    embed_weight = embed_layer.weight.detach().to(device)

    rows = load_jsonl(args.input)
    print("rows:", len(rows))

    with open(args.output, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            cond_ids, cond_mask = encode(tokenizer, make_condition_text(ex), args.max_cond_len, device)
            unsafe_ids, unsafe_mask = encode(tokenizer, ex["unsafe_response"], args.max_resp_len, device)

            cond_emb = embed_layer(cond_ids)
            unsafe_emb = embed_layer(unsafe_ids)
            cond_vec = masked_mean(cond_emb, cond_mask)

            dim = ex.get("target_dimension", "overall_quality")
            dim_idx = DIMENSIONS.index(dim) if dim in DIMENSIONS else 0
            dim_id = torch.tensor([dim_idx], device=device, dtype=torch.long)

            pred_emb = semantic_reverse_sample(
                model=model,
                unsafe_emb=unsafe_emb.float(),
                cond_vec=cond_vec.float(),
                dim_id=dim_id,
                unsafe_mask=unsafe_mask,
                T=args.T,
            )

            text = round_embeddings_to_tokens(
                pred_emb[0],
                embed_weight=embed_weight,
                tokenizer=tokenizer,
            )

            rec = {
                **ex,
                "diffusion_response": text,
                "method": "semantic_diffusion_denoiser_v0",
                "ckpt": args.ckpt,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Saved to", args.output)


if __name__ == "__main__":
    main()
