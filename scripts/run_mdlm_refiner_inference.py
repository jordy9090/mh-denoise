import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def build_condition(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = get_field(ex, "target_dimension", "dimension", "violated_dimension")
    return (
        f"[QUESTION] {q}\n"
        f"[UNSAFE] {u}\n"
        f"[DIMENSION] {d}\n"
        f"[SAFE]"
    )


def estimate_target_len(ex, tokenizer, mode, max_target_len):
    if mode == "oracle":
        y = get_field(ex, "safe_response", "target_response", "response")
        ids = tokenizer(y, add_special_tokens=False)["input_ids"]
        return min(max(len(ids) + 1, 16), max_target_len)

    if mode == "unsafe":
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        ids = tokenizer(u, add_special_tokens=False)["input_ids"]
        # safe response often needs a little more room than unsafe draft
        return min(max(int(len(ids) * 1.15) + 4, 32), max_target_len)

    return max_target_len


@torch.no_grad()
def iterative_unmask(
    model,
    tokenizer,
    ex,
    device,
    max_source_len=256,
    max_target_len=160,
    length_mode="unsafe",
    steps=12,
    temperature=1.0,
    top_k=20,
):
    model.eval()

    cond = build_condition(ex)
    cond_ids = tokenizer(
        cond,
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_len,
    )["input_ids"]

    target_len = estimate_target_len(ex, tokenizer, length_mode, max_target_len)

    target_ids = [tokenizer.mask_token_id] * target_len
    input_ids = cond_ids + target_ids
    target_start = len(cond_ids)

    special_ban = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
    }
    if tokenizer.sep_token_id is not None:
        # allow SEP only in later steps via heuristic? For now ban early, append cleanup later.
        pass

    for s in range(steps, 0, -1):
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)

        logits = model(input_ids=ids, attention_mask=attn).logits[0]

        masked_positions = [
            pos for pos in range(target_start, len(input_ids))
            if input_ids[pos] == tokenizer.mask_token_id
        ]
        if not masked_positions:
            break

        pos_tensor = torch.tensor(masked_positions, dtype=torch.long, device=device)
        pos_logits = logits[pos_tensor] / max(temperature, 1e-6)

        # ban bad special tokens
        for bid in special_ban:
            if bid is not None:
                pos_logits[:, bid] = -float("inf")

        # Keep SEP possible only after at least 30% generation, but not too early.
        if tokenizer.sep_token_id is not None and s > int(steps * 0.65):
            pos_logits[:, tokenizer.sep_token_id] = -float("inf")

        probs = torch.softmax(pos_logits, dim=-1)

        if top_k and top_k > 0:
            vals, inds = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            vals = vals / vals.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(vals, num_samples=1).squeeze(-1)
            pred_ids = inds.gather(1, choice.unsqueeze(-1)).squeeze(-1)
            conf = vals.gather(1, choice.unsqueeze(-1)).squeeze(-1)
        else:
            conf, pred_ids = torch.max(probs, dim=-1)

        # confidence-based reverse unmasking:
        # each step fixes a fraction of remaining masks and carries them over.
        n_remaining = len(masked_positions)
        n_to_fill = max(1, math.ceil(n_remaining / s))

        fill_order = torch.argsort(conf, descending=True)[:n_to_fill].tolist()

        for idx in fill_order:
            pos = masked_positions[idx]
            tok = int(pred_ids[idx].item())
            input_ids[pos] = tok

    # Replace any remaining masks with period-ish safe fallback tokens.
    period_id = tokenizer.convert_tokens_to_ids(".")
    input_ids = [period_id if t == tokenizer.mask_token_id else t for t in input_ids]

    gen_ids = input_ids[target_start:]
    # truncate at first SEP if model produced it
    if tokenizer.sep_token_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(tokenizer.sep_token_id)]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = " ".join(text.split())

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--length_mode", choices=["unsafe", "oracle", "fixed"], default="unsafe")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    for ex in tqdm(list(read_jsonl(args.input))):
        pred = iterative_unmask(
            model,
            tokenizer,
            ex,
            device,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            length_mode=args.length_mode,
            steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        out = dict(ex)
        out["mdlm_response"] = pred
        out["method"] = "mdlm_masked_discrete_refiner_v0"
        out["length_mode"] = args.length_mode
        out["sampling_steps"] = args.steps
        rows.append(out)

    write_jsonl(rows, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
