import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Set, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "that", "this", "it", "is", "are", "was", "were", "be", "been", "being",
    "you", "your", "i", "we", "they", "he", "she", "them", "as", "at", "by",
    ".", ",", "'", '"', "!", "?", ":", ";", "-", "##s", "##re", "##m"
}

DIM_KEYWORDS = {
    "toxicity": [
        "sensitive", "overreact", "loosen", "banter", "handle", "honestly", "probably",
        "fault", "blame", "dramatic", "big", "deal", "mountain", "molehill",
    ],
    "medical_advice": [
        "diagnosable", "diagnosis", "disorder", "condition", "symptom", "treatment",
        "medication", "medicine", "cognitive", "restructuring", "anxiety", "depression",
        "requires", "should", "need", "must",
    ],
    "factual_consistency": [
        "clearly", "always", "often", "direct", "indicator", "underlying", "suggests",
        "means", "drives", "manifest", "resilient", "stability",
    ],
    "specificity": [
        "general", "generally", "things", "stuff", "sometimes", "normal", "common",
        "focus", "helpful", "approach", "wellness",
    ],
    "empathy": [
        "observe", "noticing", "common", "behavior", "pattern", "involved", "detached",
    ],
    "overall_quality": [
        "just", "maybe", "things", "stuff", "handle", "figure", "focus", "general",
        "complicated", "sometimes", "resolve",
    ],
}

BAD_OUTPUT_MARKERS = [
    "[MASK]", "[CLS]", "[SEP]", "[UNK]",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
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


def normalize_dim(dim):
    d = str(dim).strip().lower().replace("-", "_").replace(" ", "_")
    return d if d in DIM_KEYWORDS else "overall_quality"


def build_condition(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
    return (
        f"[QUESTION] {q}\n"
        f"[UNSAFE] {u}\n"
        f"[DIMENSION] {d}\n"
        f"[DRAFT]"
    )


def token_is_content(tok):
    t = tok.lower()
    if t in STOPWORDS:
        return False
    if len(t) <= 1:
        return False
    if re.fullmatch(r"[^\w]+", t):
        return False
    return True


def expand(pos: Set[int], n: int, radius: int) -> Set[int]:
    out = set()
    for p in pos:
        for j in range(max(0, p - radius), min(n, p + radius + 1)):
            out.add(j)
    return out


def choose_mask_positions(tokens: List[str], dim: str, min_rate: float, max_rate: float, seed: int):
    rng = random.Random(seed)
    dim = normalize_dim(dim)
    low = [t.lower().replace("##", "") for t in tokens]
    n = len(tokens)

    pos = set()
    keys = DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]

    for i, t in enumerate(low):
        if token_is_content(tokens[i]) and any(k in t for k in keys):
            pos.add(i)

    pos = expand(pos, n, radius=2)

    content = [i for i, t in enumerate(tokens) if token_is_content(t)]
    rng.shuffle(content)

    target_rate = rng.uniform(min_rate, max_rate)
    min_need = max(1, int(len(content) * target_rate))

    for i in content[:min_need]:
        pos.add(i)

    # avoid masking too much punctuation / fragments
    pos = {i for i in pos if token_is_content(tokens[i])}

    return pos


def repetition_penalty(text):
    toks = text.lower().split()
    if not toks:
        return 999
    repeat = 0
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            repeat += 1
    return repeat / max(1, len(toks))


def bad_keyword_penalty(text, dim):
    t = text.lower()
    keys = DIM_KEYWORDS.get(normalize_dim(dim), [])
    return sum(1 for k in keys if k in t)


def punctuation_penalty(text):
    if not text:
        return 999
    punct = sum(1 for ch in text if ch in ",;'\"")
    return punct / max(1, len(text))


def score_candidate(text, dim):
    if not text.strip():
        return -999
    score = 0.0
    score -= repetition_penalty(text) * 8.0
    score -= punctuation_penalty(text) * 6.0
    score -= bad_keyword_penalty(text, dim) * 0.8
    for m in BAD_OUTPUT_MARKERS:
        if m.lower() in text.lower():
            score -= 10
    words = text.split()
    if len(words) < 12:
        score -= 4
    if len(words) > 140:
        score -= 2
    if "i'm sorry" in text.lower() or "it sounds" in text.lower() or "understandable" in text.lower():
        score += 1.5
    return score


@torch.no_grad()
def refine_once(
    model,
    tokenizer,
    ex,
    device,
    seed,
    max_source_len=256,
    max_target_len=160,
    min_mask_rate=0.25,
    max_mask_rate=0.55,
    steps=8,
    top_k=8,
    temperature=0.8,
):
    rng = random.Random(seed)
    dim = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
    unsafe = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    cond = build_condition(ex)

    cond_ids = tokenizer(
        cond,
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_len,
    )["input_ids"]

    draft_ids = tokenizer(
        unsafe,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_len - 1,
    )["input_ids"]

    # add a few masks at the end so the model can add a safe closing phrase
    tail_masks = rng.randint(4, 10)
    draft_ids = draft_ids + [tokenizer.mask_token_id] * tail_masks + [tokenizer.sep_token_id]

    tokens = tokenizer.convert_ids_to_tokens(draft_ids)
    mask_pos_local = choose_mask_positions(tokens, dim, min_mask_rate, max_mask_rate, seed)

    # always mask tail masks
    for i, tid in enumerate(draft_ids):
        if tid == tokenizer.mask_token_id:
            mask_pos_local.add(i)

    corrupted = list(draft_ids)
    for p in mask_pos_local:
        if 0 <= p < len(corrupted) and corrupted[p] != tokenizer.sep_token_id:
            corrupted[p] = tokenizer.mask_token_id

    input_ids = cond_ids + corrupted
    target_start = len(cond_ids)

    ban_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
    }
    comma_id = tokenizer.convert_tokens_to_ids(",")
    quote_id = tokenizer.convert_tokens_to_ids("'")
    semicolon_id = tokenizer.convert_tokens_to_ids(";")
    colon_id = tokenizer.convert_tokens_to_ids(":")
    for x in [comma_id, quote_id, semicolon_id, colon_id]:
        if isinstance(x, int) and x >= 0:
            ban_ids.add(x)

    for s in range(steps, 0, -1):
        mask_positions = [
            i for i in range(target_start, len(input_ids))
            if input_ids[i] == tokenizer.mask_token_id
        ]
        if not mask_positions:
            break

        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=attn).logits[0]

        pos_tensor = torch.tensor(mask_positions, dtype=torch.long, device=device)
        pos_logits = logits[pos_tensor] / max(temperature, 1e-6)

        for bid in ban_ids:
            if bid is not None and 0 <= int(bid) < pos_logits.size(-1):
                pos_logits[:, int(bid)] = -float("inf")

        # keep SEP from appearing too early
        if tokenizer.sep_token_id is not None and s > 2:
            pos_logits[:, tokenizer.sep_token_id] = -float("inf")

        probs = torch.softmax(pos_logits, dim=-1)
        vals, inds = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)

        chosen_idx = torch.multinomial(vals, num_samples=1).squeeze(-1)
        chosen = inds.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)
        conf = vals.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)

        n_to_fill = max(1, len(mask_positions) // s)
        order = torch.argsort(conf, descending=True)[:n_to_fill].tolist()

        for oi in order:
            input_ids[mask_positions[oi]] = int(chosen[oi].item())

    # remaining masks -> remove by replacing with period
    period_id = tokenizer.convert_tokens_to_ids(".")
    input_ids = [period_id if tid == tokenizer.mask_token_id else tid for tid in input_ids]

    gen_ids = input_ids[target_start:]

    if tokenizer.sep_token_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(tokenizer.sep_token_id)]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = cleanup(text)
    return text


def cleanup(text):
    text = text.replace(" ##", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1{1,}", r"\1", text)
    text = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", text, flags=re.I)
    text = re.sub(r"'\s+'", "'", text)
    text = text.replace(" ' s", "'s").replace(" ' re", "'re").replace(" ' m", "'m")
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=160)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min_mask_rate", type=float, default=0.25)
    ap.add_argument("--max_mask_rate", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    for idx, ex in enumerate(tqdm(list(read_jsonl(args.input)))):
        dim = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
        candidates = []
        for c in range(args.candidates):
            text = refine_once(
                model,
                tokenizer,
                ex,
                device,
                seed=args.seed + idx * 100 + c,
                max_source_len=args.max_source_len,
                max_target_len=args.max_target_len,
                min_mask_rate=args.min_mask_rate,
                max_mask_rate=args.max_mask_rate,
                steps=args.steps,
                top_k=args.top_k,
                temperature=args.temperature,
            )
            candidates.append((score_candidate(text, dim), text))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text = candidates[0]

        out = dict(ex)
        out["mdlm_edit_response"] = best_text
        out["mdlm_edit_score"] = best_score
        out["mdlm_edit_candidates"] = [t for _, t in candidates[: min(5, len(candidates))]]
        out["method"] = "mdlm_edit_refiner_v1"
        rows.append(out)

    write_jsonl(rows, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
