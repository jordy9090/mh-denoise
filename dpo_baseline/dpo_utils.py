from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import yaml


DIMS: Tuple[str, ...] = (
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
)

SYSTEM_PROMPT = (
    "You are a careful mental-health response refiner. "
    "Rewrite the initial draft into one safe, empathetic, specific, and professionally bounded response. "
    "Preserve useful content, correct unsafe or unsupported claims, avoid diagnosis and direct treatment instructions, "
    "and return only the final response."
)

FORBIDDEN_PROMPT_KEYS: Tuple[str, ...] = (
    "target_dimension",
    "violation_vector",
    "brief_reason",
    "counselor_annotation",
    "counsellor_annotation",
    "expert_annotation",
    "therapist_annotation",
    "human_label",
)


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | os.PathLike[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temp.replace(output)


def write_json(value: Mapping[str, Any] | Sequence[Any], path: str | os.PathLike[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(output)


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return value


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_first_text(row: Mapping[str, Any], names: Sequence[str], *, required: bool = True) -> str:
    for name in names:
        if name in row:
            text = clean_text(row.get(name))
            if text:
                return text
    if required:
        raise ValueError(f"Missing non-empty field; expected one of {list(names)} in row id={row.get('id')!r}")
    return ""


def extract_refinement_fields(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    question = get_first_text(row, ("question", "query", "user_question"))
    initial_draft = get_first_text(row, ("unsafe_response", "corrupted_response", "bad_response"))
    safe_response = get_first_text(row, ("safe_response", "target_response", "target", "response"))
    return question, initial_draft, safe_response


def question_key(row: Mapping[str, Any]) -> str:
    explicit = clean_text(row.get("question_id"))
    if explicit:
        return explicit

    row_id = clean_text(row.get("id"))
    dim = clean_text(row.get("target_dimension"))
    if row_id and dim:
        suffix = "_" + dim
        if row_id.endswith(suffix):
            return row_id[: -len(suffix)]
    if row_id:
        return row_id

    question = get_first_text(row, ("question", "query", "user_question"))
    return "q_" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def row_key(row: Mapping[str, Any], index: int | None = None) -> str:
    explicit = clean_text(row.get("id"))
    if explicit:
        return explicit
    qid = question_key(row)
    suffix = str(index) if index is not None else hashlib.sha256(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]
    return f"{qid}_{suffix}"


def build_prompt_messages(question: str, initial_draft: str) -> List[Dict[str, str]]:
    question = clean_text(question)
    initial_draft = clean_text(initial_draft)
    if not question or not initial_draft:
        raise ValueError("Both question and initial draft are required")
    user_content = f"User question:\n{question}\n\nInitial draft:\n{initial_draft}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def fit_refinement_prompt_to_budget(
    tokenizer: Any,
    question: str,
    initial_draft: str,
    max_source_len: int,
) -> Tuple[List[Dict[str, str]], str, str, bool]:
    if max_source_len <= 0:
        raise ValueError("max_source_len must be positive")
    question = clean_text(question)
    initial_draft = clean_text(initial_draft)
    prompt = build_prompt_messages(question, initial_draft)
    if len(chat_template_ids(tokenizer, prompt, add_generation_prompt=True)) <= max_source_len:
        return prompt, question, initial_draft, False

    draft_ids = tokenizer(initial_draft, add_special_tokens=False)["input_ids"]
    low, high = 1, len(draft_ids)
    best_draft = ""
    while low <= high:
        middle = (low + high) // 2
        candidate_draft = clean_text(tokenizer.decode(draft_ids[:middle], skip_special_tokens=True))
        candidate_prompt = build_prompt_messages(question, candidate_draft)
        if len(chat_template_ids(tokenizer, candidate_prompt, add_generation_prompt=True)) <= max_source_len:
            best_draft = candidate_draft
            low = middle + 1
        else:
            high = middle - 1
    if best_draft:
        return build_prompt_messages(question, best_draft), question, best_draft, True

    # Keep the starts of both fields, matching the legacy right-side truncation direction.
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    one_draft_token = clean_text(tokenizer.decode(draft_ids[:1], skip_special_tokens=True))
    for question_count in range(len(question_ids), 0, -1):
        candidate_question = clean_text(tokenizer.decode(question_ids[:question_count], skip_special_tokens=True))
        candidate_prompt = build_prompt_messages(candidate_question, one_draft_token)
        if len(chat_template_ids(tokenizer, candidate_prompt, add_generation_prompt=True)) <= max_source_len:
            return candidate_prompt, candidate_question, one_draft_token, True
    raise ValueError(f"The fixed prompt template cannot fit within max_source_len={max_source_len}")


def build_completion_message(response: str) -> List[Dict[str, str]]:
    response = clean_text(response)
    if not response:
        raise ValueError("Completion response must be non-empty")
    return [{"role": "assistant", "content": response}]


def completion_token_length(
    tokenizer: Any,
    prompt: Sequence[Mapping[str, Any]],
    response: str,
) -> int:
    prompt_ids = chat_template_ids(tokenizer, prompt, add_generation_prompt=True)
    full_ids = chat_template_ids(
        tokenizer, [*prompt, *build_completion_message(response)], add_generation_prompt=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat-template prefix mismatch while enforcing the completion budget")
    return len(full_ids) - len(prompt_ids)


def truncate_completion_to_budget(
    tokenizer: Any,
    prompt: Sequence[Mapping[str, Any]],
    response: str,
    max_tokens: int,
) -> Tuple[str, bool]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    response = clean_text(response)
    if completion_token_length(tokenizer, prompt, response) <= max_tokens:
        return response, False

    plain_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    low, high = 1, len(plain_ids)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = clean_text(tokenizer.decode(plain_ids[:middle], skip_special_tokens=True))
        if candidate and completion_token_length(tokenizer, prompt, candidate) <= max_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise ValueError(f"A non-empty completion cannot fit within max_tokens={max_tokens}")
    return best, True


def validate_prompt_contract(prompt: Sequence[Mapping[str, Any]], question: str, initial_draft: str) -> None:
    if len(prompt) != 2:
        raise ValueError(f"Expected exactly two prompt messages, received {len(prompt)}")
    if prompt[0].get("role") != "system" or prompt[1].get("role") != "user":
        raise ValueError("Prompt must contain one system message followed by one user message")
    serialized = json.dumps(list(prompt), ensure_ascii=False)
    if clean_text(question) not in serialized:
        raise ValueError("Question is absent from the prompt")
    if clean_text(initial_draft) not in serialized:
        raise ValueError("Initial draft is absent from the prompt")
    lowered = serialized.lower()
    leaked = [key for key in FORBIDDEN_PROMPT_KEYS if key.lower() in lowered]
    if leaked:
        raise ValueError(f"Forbidden training metadata leaked into prompt: {leaked}")


def validate_pair_record(record: Mapping[str, Any]) -> None:
    prompt = record.get("prompt")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if not isinstance(prompt, list) or not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError("DPO record must contain conversational prompt/chosen/rejected lists")
    if len(chosen) != 1 or chosen[0].get("role") != "assistant":
        raise ValueError("chosen must contain one assistant completion")
    if len(rejected) != 1 or rejected[0].get("role") != "assistant":
        raise ValueError("rejected must contain one assistant completion")
    chosen_text = clean_text(chosen[0].get("content"))
    rejected_text = clean_text(rejected[0].get("content"))
    if not chosen_text or not rejected_text:
        raise ValueError("chosen and rejected completions must be non-empty")
    if normalize_text(chosen_text) == normalize_text(rejected_text):
        raise ValueError(f"chosen and rejected are equivalent for pair id={record.get('pair_id')!r}")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    question = clean_text(metadata.get("question"))
    initial_draft = clean_text(metadata.get("initial_draft"))
    if question and initial_draft:
        validate_prompt_contract(prompt, question, initial_draft)


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: str | os.PathLike[str]) -> Dict[str, Any]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    selected = [
        p
        for p in files
        if p.name
        in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
    ]
    if not selected:
        selected = files[:50]
    entries = {str(p.relative_to(root)): file_sha256(p) for p in selected}
    combined = hashlib.sha256(
        json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {"path": str(root), "sha256": combined, "files": entries}


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "||".join(str(x) for x in (base_seed, *parts))
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return value % (2**31 - 1)


def set_python_seed(seed: int) -> None:
    random.seed(seed)


def group_by_question(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[question_key(row)].append(row)
    return dict(grouped)


def assert_question_disjoint(
    named_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[str]]:
    keys = {name: {question_key(row) for row in rows} for name, rows in named_rows.items()}
    overlaps: Dict[str, List[str]] = {}
    names = sorted(keys)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common = sorted(keys[left] & keys[right])
            if common:
                overlaps[f"{left}__{right}"] = common
    if overlaps:
        counts = {name: len(values) for name, values in overlaps.items()}
        raise ValueError(f"Question leakage across splits: {counts}")
    return {name: sorted(values) for name, values in keys.items()}


def lexical_similarity(left: str, right: str) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "run"


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def cleanup_generation(text: str) -> str:
    text = str(text or "").strip()
    prefixes = (
        "Here is the refined response:",
        "Here is the final response:",
        "Refined response:",
        "Final response:",
        "Final answer:",
    )
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            break
    text = text.replace("<MASK>", " ").replace("[MASK]", " ")
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip("'").strip()
    return text


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def content_words(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "with",
        "you",
        "your",
    }
    return {token for token in normalize_text(text).split() if len(token) > 2 and token not in stop}


def specificity_proxy(question: str, response: str) -> float:
    q_words = content_words(question)
    r_words = content_words(response)
    if not r_words:
        return 0.0
    overlap = len(q_words & r_words) / max(1, len(q_words))
    lexical_density = min(1.0, len(r_words) / 45.0)
    action_markers = (
        "try ",
        "consider ",
        "you could",
        "one step",
        "for example",
        "notice ",
        "write ",
        "talk ",
        "reach out",
        "contact ",
    )
    action = min(1.0, sum(marker in response.lower() for marker in action_markers) / 2.0)
    return 0.50 * overlap + 0.30 * lexical_density + 0.20 * action


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_mapping(value, full))
        else:
            result[full] = value
    return result


def set_nested(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, MutableMapping):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value
