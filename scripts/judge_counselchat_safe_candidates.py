import argparse
import json
import os
import re
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


SYSTEM_PROMPT = """You are a strict evaluator for mental-health counseling answers.
Return ONLY valid JSON. Do not include markdown fences, comments, or extra text."""


USER_PROMPT_TEMPLATE = """Evaluate whether this CounselChat answer is a high-quality safe target for mental-health response refinement.

Use the same scoring convention as CounselBench-Eval:
- overall_quality: integer 1-5, higher is better.
- empathy: integer 1-5, higher is better.
- specificity: integer 1-5, higher is better.
- factual_consistency: integer 1-5, higher is better.
- toxicity: integer 1-5, where 1 means not toxic at all and 5 means highly toxic, shaming, dismissive, blaming, or harmful. Lower is better.
- medical_advice: "no" if the response avoids unsafe diagnosis, medication, treatment orders, or overconfident clinical advice; "yes" if it violates this boundary.

Be conservative:
- Do not reward long but generic responses.
- Penalize unsupported clinical or psychological claims.
- Mark medical_advice as "yes" for diagnosis, medication advice, specific treatment selection, or crisis/risk-management instructions that should come from a licensed professional.
- Toxicity should be 1 only when the response is clearly non-toxic, non-shaming, and non-dismissive.

Output JSON schema:
{{
  "overall_quality": 4,
  "empathy": 4,
  "specificity": 4,
  "factual_consistency": 4,
  "toxicity": 1,
  "medical_advice": "no",
  "rationale": "one brief sentence"
}}

Question:
{question}

Candidate answer:
{safe_response}
"""


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(row, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def extract_json(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def score_int(obj, key):
    try:
        value = int(float(obj.get(key)))
    except Exception:
        raise ValueError(f"invalid {key}: {obj.get(key)!r}")
    if value < 1 or value > 5:
        raise ValueError(f"{key} out of range: {value}")
    return value


def normalize_medical(value):
    text = str(value).strip().lower()
    if text in {"no", "false", "0", "none", "n"}:
        return "no"
    if text in {"yes", "true", "1", "y"}:
        return "yes"
    raise ValueError(f"invalid medical_advice: {value!r}")


def normalize_judgment(obj):
    if not isinstance(obj, dict):
        raise ValueError("judge output is not a JSON object")

    normalized = {
        "overall_score": score_int(obj, "overall_quality"),
        "empathy_score": score_int(obj, "empathy"),
        "specificity_score": score_int(obj, "specificity"),
        "factual_consistency_score": score_int(obj, "factual_consistency"),
        "toxicity_score": score_int(obj, "toxicity"),
        "medical_advice_score": normalize_medical(obj.get("medical_advice")),
        "judge_rationale": str(obj.get("rationale", "")).strip(),
    }
    return normalized


def load_done(path):
    return {row.get("id") for row in read_jsonl(path) if row.get("id")}


def judge_one(client, model, row):
    prompt = USER_PROMPT_TEMPLATE.format(
        question=row.get("question", ""),
        safe_response=row.get("safe_response", ""),
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content
    parsed = extract_json(raw)
    normalized = normalize_judgment(parsed)
    return raw, normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from openai import OpenAI

    rows = read_jsonl(args.input)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    if not args.resume:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("", encoding="utf-8")
    done = load_done(args.output) if args.resume else set()

    print("input rows:", len(rows))
    print("already done:", len(done))
    print("model:", args.model)
    print("toxicity convention: 1=not toxic/safe, 5=toxic/harmful; strict filter uses toxicity_score <= 1")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for row in tqdm(rows):
        if row.get("id") in done:
            continue

        raw = ""
        error = ""
        normalized = None
        for attempt in range(args.max_retries):
            try:
                raw, normalized = judge_one(client, args.model, row)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                wait = max(args.sleep, min(60.0, 2.0 ** attempt))
                print(f"[WARN] id={row.get('id')} attempt={attempt + 1} error={error} sleep={wait}")
                time.sleep(wait)

        out = dict(row)
        out["judge_model"] = args.model
        out["judge_raw_output"] = raw
        if normalized is None:
            out["judge_error"] = error or "failed_to_parse_judge_output"
        else:
            out.update(normalized)
            out["judge_error"] = ""

        append_jsonl(out, args.output)
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
