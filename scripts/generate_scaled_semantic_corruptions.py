import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import torch
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


DIMENSION_INSTRUCTIONS = {
    "overall_quality": """
Create a low-quality counseling response.
It should be vague, shallow, minimally helpful, and weakly supportive.
Avoid explicit toxicity or dangerous instructions.
""".strip(),

    "empathy": """
Create a response with poor empathy.
It should be emotionally flat, overly procedural, or fail to validate the user's feelings.
Avoid insults, explicit toxicity, or dangerous instructions.
""".strip(),

    "specificity": """
Create a generic response with low specificity.
It should sound broadly supportive but fail to address the user's concrete situation.
Avoid explicit toxicity or dangerous instructions.
""".strip(),

    "medical_advice": """
Create a response that violates clinical/professional boundaries.
It may sound overconfident, diagnostic, or directive about treatment.
Do NOT include medication dosages, self-harm instructions, or operational dangerous advice.
The violation should be overconfident clinical advice or diagnosis speculation.
""".strip(),

    "factual_consistency": """
Create a response with unsupported or overconfident factual claims.
It should introduce questionable psychological explanations or unwarranted certainty.
Avoid explicit toxicity or dangerous instructions.
""".strip(),

    "toxicity": """
Create a response with a judgmental, dismissive, or invalidating tone.
It should be clearly worse in tone than a safe counseling response.
Do NOT include slurs, threats, self-harm instructions, or explicit dangerous advice.
""".strip(),
}


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(record: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def pick_field(ex: Dict, candidates: List[str]) -> Optional[str]:
    for key in candidates:
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def normalize_source_rows(rows: List[Dict]) -> List[Dict]:
    """
    Accepts either:
    1. base QA rows with question + safe_response/response/answer
    2. existing corruption rows with question + safe_response + unsafe_response + target_dimension

    Returns unique question-level rows:
    {question_id, question, safe_response}
    """
    unique = {}
    for i, ex in enumerate(rows):
        question = pick_field(ex, [
            "question",
            "query",
            "prompt",
            "user_question",
            "input",
        ])
        safe_response = pick_field(ex, [
            "safe_response",
            "response",
            "answer",
            "reference",
            "target",
            "chosen",
            "gold_response",
        ])

        if not question or not safe_response:
            continue

        qid = ex.get("question_id") or ex.get("qid") or ex.get("id")
        if not qid:
            qid = stable_id(question)

        key = str(qid)
        if key not in unique:
            unique[key] = {
                "question_id": key,
                "question": question,
                "safe_response": safe_response,
            }

    return list(unique.values())


def load_done_ids(path: str):
    done = set()
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ex = json.loads(line)
                    done.add(ex.get("id"))
                except Exception:
                    pass
    return done


def build_prompt(question: str, safe_response: str, dimension: str) -> str:
    return f"""
You are creating synthetic corrupted training data for safety research on mental-health QA.

Given a user question and a safe counseling-style response, generate ONE intentionally flawed response that violates the target quality dimension.

Target violation dimension:
{dimension}

Violation instruction:
{DIMENSION_INSTRUCTIONS[dimension]}

Important constraints:
- Output only the corrupted response.
- Do not mention that this is synthetic.
- Do not explain the violation.
- Keep it plausible as a model response.
- Keep it concise, around 80-160 words.
- Do not include self-harm instructions, illegal instructions, medication dosages, or operational dangerous details.
- The response should be flawed specifically along the target dimension while remaining related to the question.

User question:
{question}

Safe response:
{safe_response}

Corrupted response:
""".strip()


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Remove common accidental prefixes.
    prefixes = [
        "Corrupted response:",
        "Unsafe response:",
        "Here is the corrupted response:",
        "Here is one possible corrupted response:",
    ]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    return text.strip().strip('"').strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("Loading source:", args.source_file)
    raw_rows = load_jsonl(args.source_file)
    base_rows = normalize_source_rows(raw_rows)

    print("raw rows:", len(raw_rows))
    print("unique usable question rows:", len(base_rows))

    if len(base_rows) == 0:
        raise RuntimeError(
            "No usable rows found. Need fields like question + safe_response/response/answer."
        )

    selected = base_rows[args.start_offset: args.start_offset + args.max_questions]
    print("selected questions:", len(selected))
    print("expected output rows:", len(selected) * len(DIMENSIONS))

    done_ids = load_done_ids(args.output)
    print("already done:", len(done_ids))

    print("Loading tokenizer/model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()

    for ex in tqdm(selected):
        qid = ex["question_id"]
        question = ex["question"]
        safe_response = ex["safe_response"]

        for dim in DIMENSIONS:
            rid = f"{qid}__{dim}"
            if rid in done_ids:
                continue

            prompt = build_prompt(question, safe_response, dim)
            unsafe = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            record = {
                "id": rid,
                "question_id": qid,
                "question": question,
                "safe_response": safe_response,
                "target_dimension": dim,
                "unsafe_response": unsafe,
                "generator_model": args.model,
                "prompt_version": "semantic_corruption_v2_scaled",
            }
            append_jsonl(record, args.output)
            done_ids.add(rid)

    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
