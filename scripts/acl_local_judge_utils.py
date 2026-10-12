"""Local generation helpers preserved from the frozen ACL evaluation runner.

Only the five helpers used by the pointwise evaluators are included. Their
function bodies are unchanged; unrelated development experiments are omitted.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError as error:
        if "enable_thinking" not in str(error):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)

def generation_eos_ids(model, tokenizer) -> list[int]:
    for value in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        ids = list(dict.fromkeys(int(item) for item in values if item is not None))
        if ids:
            return ids
    return []

def strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.I | re.S).strip()

def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = strip_reasoning(text)
    attempts = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match and match.group(0) != cleaned:
        attempts.append(match.group(0))
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, ""
    return None, "no_valid_json_object"

def load_local_generator(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    return tokenizer, model
