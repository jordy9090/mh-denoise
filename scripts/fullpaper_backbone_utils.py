#!/usr/bin/env python3
"""Model-loading helpers shared by the four frozen full-paper backbones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def _config_source(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    adapter_config = path / "adapter_config.json"
    if path.is_dir() and not (path / "config.json").exists() and adapter_config.exists():
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
        base = payload.get("base_model_name_or_path")
        if base:
            return str(base)
    return model_name_or_path


def backbone_model_type(model_name_or_path: str) -> str:
    config = AutoConfig.from_pretrained(
        _config_source(model_name_or_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    return str(getattr(config, "model_type", ""))


def load_text_generation_model(model_name_or_path: str, **kwargs: Any):
    """Load the official local architecture without enabling cached remote code.

    Gemma 4 and Ministral 3 are conditional-generation wrappers even for the
    text-only path.  Transformers maps Gemma 4 through AutoModelForCausalLM,
    while Ministral 3 is registered with AutoModelForImageTextToText.  Phi's
    cached remote module targets an older Transformers API, so main experiments
    deliberately use the installed official Phi3 implementation.
    """

    model_type = backbone_model_type(model_name_or_path)
    loader = AutoModelForImageTextToText if model_type == "mistral3" else AutoModelForCausalLM
    return loader.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        **kwargs,
    )


def load_fullpaper_tokenizer(model_name_or_path: str, **kwargs: Any):
    model_type = backbone_model_type(model_name_or_path)
    if model_type == "mistral3":
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
        trust_remote_code=False,
        **kwargs,
    )
