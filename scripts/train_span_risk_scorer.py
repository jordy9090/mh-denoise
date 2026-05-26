import argparse, json, random, re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


DIMS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]

DIM_PATTERNS = {
    "medical_advice": [
        "diagnosis", "diagnosable", "disorder", "borderline personality disorder",
        "adjustment disorder", "social anxiety", "depression", "cognitive restructuring",
        "medication", "medicine", "treatment", "formal diagnosis",
        "you should", "you need to", "you must", "requires"
    ],
    "toxicity": [
        "too sensitive", "overreact", "overreacting", "loosen up", "big deal",
        "mountain out of a molehill", "dramatic", "paranoid", "crazy",
        "your fault", "probably just"
    ],
    "factual_consistency": [
        "clearly", "always", "never", "direct indicator", "underlying",
        "suggests", "means that", "drives", "manifest", "proves",
        "obviously", "emotional stability", "resilient"
    ],
    "specificity": [
        "things", "stuff", "generally", "sometimes", "common", "normal",
        "focus on other things", "helpful approach", "wellness", "take care",
        "try to handle"
    ],
    "empathy": [
        "observe", "noticing", "common behavior", "pattern", "detached",
        "complicated", "it happens", "people do"
    ],
    "overall_quality": [
        "just", "maybe", "things", "stuff", "handle it", "figure out",
        "resolve over time", "next steps yourself", "complicated"
    ],
}

GENERAL = [
    "you should", "you need to", "you must", "clearly", "always", "just", "maybe"
]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_field(ex, *names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x):
    x = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    return x if x in DIMS else "overall_quality"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def sim(a, b):
    a_l, b_l = a.lower(), b.lower()
    sa = set(re.findall(r"[a-zA-Z']+", a_l))
    sb = set(re.findall(r"[a-zA-Z']+", b_l))
    j = len(sa & sb) / max(1, len(sa | sb))
    seq = SequenceMatcher(None, a_l, b_l).ratio()
    return 0.5 * j + 0.5 * seq


def contains_any(text, patterns):
    low = text.lower()
    return any(p.lower() in low for p in patterns)


def build_examples(rows):
    out = []
    for ex in rows:
        q = get_field(ex, "question", "query", "user_question")
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        y = get_field(ex, "safe_response", "target_response", "response")
        d = norm_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))

        us = split_sentences(u)
        ys = split_sentences(y)
        patterns = DIM_PATTERNS.get(d, []) + GENERAL

        for s in us:
            best = max([sim(s, yy) for yy in ys], default=0.0)
            has_kw = contains_any(s, patterns)
            has_general_risk = contains_any(s, GENERAL)
            # weak positive: dimension cue or low alignment with safe target plus risky wording
            label = 1 if (has_kw or (best < 0.28 and has_general_risk)) else 0
            out.append({
                "question": q,
                "dimension": d,
                "span": s,
                "best_safe_similarity": best,
                "label": label,
                "source": "unsafe_sentence",
            })

        # safe sentences as negatives, unless they contain explicit risky tokens
        for s in ys[: min(len(ys), 4)]:
            label = 1 if contains_any(s, patterns) and d in ["medical_advice", "toxicity"] else 0
            out.append({
                "question": q,
                "dimension": d,
                "span": s,
                "best_safe_similarity": 1.0,
                "label": label,
                "source": "safe_sentence",
            })
    return out


def build_text(ex):
    return (
        "Question:\n" + ex["question"].strip() +
        "\n\nViolation dimension:\n" + ex["dimension"] +
        "\n\nCandidate span:\n" + ex["span"].strip() +
        "\n\nTask: predict whether this span is a safety or quality risk for the given dimension."
    )


class RiskDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=384):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(build_text(ex), truncation=True, max_length=self.max_len, padding=False)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": float(ex["label"]),
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad = self.tokenizer.pad_token_id
        ids, masks, labels = [], [], []
        for x in batch:
            n = max_len - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad] * n)
            masks.append(x["attention_mask"] + [0] * n)
            labels.append(x["labels"])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, preds, golds = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.squeeze(-1).float()
        loss = loss_fn(logits, labels)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
        prob = torch.sigmoid(logits)
        preds.extend((prob >= 0.5).long().cpu().tolist())
        golds.extend(labels.long().cpu().tolist())
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    acc = sum(1 for p, g in zip(preds, golds) if p == g) / max(1, len(golds))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    model.train()
    return sum(losses) / max(1, len(losses)), acc, f1


def save(model, tok, out):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    with open(out / "risk_scorer_info.json", "w", encoding="utf-8") as f:
        json.dump({"dims": DIMS, "patterns": DIM_PATTERNS}, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--valid_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_ex = build_examples(read_jsonl(args.train_file))
    valid_ex = build_examples(read_jsonl(args.valid_file))
    print("train span examples:", len(train_ex), Counter(x["label"] for x in train_ex))
    print("valid span examples:", len(valid_ex), Counter(x["label"] for x in valid_ex))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = DataLoader(RiskDataset(train_ex, tok), batch_size=args.batch_size, shuffle=True, collate_fn=Collator(tok))
    valid_loader = DataLoader(RiskDataset(valid_ex, tok), batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tok))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 999.0
    gstep = 0

    for ep in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {ep+1}/{args.epochs}")
        for batch in pbar:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.squeeze(-1).float()
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss; skip")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            gstep += 1
            pbar.set_postfix({"loss": round(float(loss.item()), 4), "gstep": gstep})
            if gstep % args.eval_every == 0:
                ev, acc, f1 = evaluate(model, valid_loader, device)
                print(f"[eval] step={gstep} loss={ev:.4f} acc={acc:.4f} f1={f1:.4f}")
                if ev < best:
                    best = ev
                    save(model, tok, out_dir / "best")
                    print(f"[save] best -> {out_dir/'best'}")
    save(model, tok, out_dir / "final")
    print(f"[done] best valid loss: {best:.4f}")
    print(f"[done] saved final -> {out_dir/'final'}")


if __name__ == "__main__":
    main()
