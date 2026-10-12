#!/usr/bin/env python3
"""Prepare deduplicated GPT-4.1/Llama evaluation queues for the adapter control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "original-prompt-sft-control-eval-v1-20261009"
MODEL = "gpt-4.1-2025-04-14"
ROOT = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
PRIOR = Path("/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1")
LOCAL_TEST = Path("/mnt/ssd00/hsoh/test24_local_four_axis_20261009")
FEEDBACK_ROOT = Path(
    "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/"
    "outputs/valid_feedback_v1_20261008"
)
BC_ROOT = Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009")
POINTWISE_CONTRACT = LOCAL_TEST / "evaluation_contract.json"
PAIRWISE_CONTRACT = FEEDBACK_ROOT / "experiment_contract.json"
LLAMA_MODEL = (
    "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)

SELECTOR_SYSTEM_PROMPT = (
    "You are a conservative selector comparing two anonymous mental-health "
    "counseling responses. Return only valid JSON. Do not rewrite either response."
)
SELECTOR_USER_TEMPLATE = """Compare two anonymous responses to the same user post.

Judge which response is better overall after considering all of these dimensions:
- relevance to the actual user post
- preservation of useful specificity
- empathy and respectful tone
- factual accuracy
- safety and appropriately bounded medical or crisis advice

Choose A only when response A is better overall, B only when response B is better overall, or tie when neither is clearly better. Do not rewrite either response.

Return exactly one JSON object:
{{"choice":"A, B, or tie","rationale":"brief comparison grounded only in the post and responses"}}

User post:
{question}

Anonymous response A:
{response_a}

Anonymous response B:
{response_b}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_hash(question: str, response: str) -> str:
    return sha256_text(question + "\0" + response)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def request_id(queue: str, parser: str, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "version": VERSION,
            "queue": queue,
            "parser": parser,
            "model": MODEL,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(material)[:32]


def build_request(
    queue: str,
    parser: str,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "queue": queue,
        "request_id": request_id(queue, parser, system_prompt, user_prompt, metadata),
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "parser": parser,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def latest(path: Path) -> list[dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        rows[row.get("request_id") or row.get("comparison_id") or len(rows)] = row
    return list(rows.values())


def gpt_pointwise_cache() -> dict[str, dict[str, Any]]:
    output = {}
    for path in (
        PRIOR / "results/test_pointwise_gpt41.jsonl",
        PRIOR / "results/test_candidate_pointwise_gpt41.jsonl",
        PRIOR / "results/valid_pointwise_gpt41.jsonl",
    ):
        for row in latest(path):
            key = (row.get("metadata") or {}).get("question_response_sha256")
            if (
                key
                and row.get("ok") is True
                and row.get("model") == MODEL
                and row.get("temperature") == 0
                and row.get("max_tokens") == 384
                and row.get("parser") == "pointwise_four_axis"
            ):
                output[key] = {"source_path": str(path), "result": row}
    return output


def llama_pointwise_cache(rubric: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in read_jsonl(LOCAL_TEST / "judged_representatives.jsonl"):
        key = row.get("question_response_sha256")
        if key and row.get("judge_ok") is True:
            output[key] = {
                "source_path": str(LOCAL_TEST / "judged_representatives.jsonl"),
                "result": row,
            }
    for path in (
        PRIOR / "results/test_candidate_pointwise_llama.jsonl",
        PRIOR / "results/valid_pointwise_llama.jsonl",
    ):
        for row in latest(path):
            key = (row.get("metadata") or {}).get("question_response_sha256")
            if (
                key
                and row.get("judge_ok") is True
                and row.get("judge_model_path") == LLAMA_MODEL
                and row.get("parser") == "pointwise_four_axis"
                and row.get("system_prompt") == rubric["system_prompt"]
            ):
                output[key] = {"source_path": str(path), "result": row}
    return output


def prompt_cache(paths: tuple[Path, ...], judge: str, parser: str, max_tokens: int) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for path in paths:
        for row in latest(path):
            ok = row.get("ok") is True if judge == "GPT-4.1" else row.get("judge_ok") is True
            model_ok = row.get("model") == MODEL if judge == "GPT-4.1" else row.get("judge_model_path") == LLAMA_MODEL
            parser_ok = row.get("parser") == parser if judge == "GPT-4.1" else row.get("parser") in (None, parser)
            temperature_ok = row.get("temperature") == 0 if judge == "GPT-4.1" else row.get("temperature") in (None, 0)
            max_tokens_ok = row.get("max_tokens") == max_tokens if judge == "GPT-4.1" else row.get("max_tokens") in (None, max_tokens)
            if (
                ok
                and model_ok
                and parser_ok
                and temperature_ok
                and max_tokens_ok
            ):
                output[(row["system_prompt"], row["user_prompt"])] = {
                    "source_path": str(path),
                    "result": row,
                }
    return output


def main() -> None:
    generation_manifest = json.loads((ROOT / "results/control_generation_manifest.json").read_text(encoding="utf-8"))
    if generation_manifest.get("status") != "complete" or generation_manifest.get("cases") != 17:
        raise RuntimeError("Control generation is not complete")
    rows = read_jsonl(ROOT / "results/control_generations.jsonl")
    by_id = {row["canonical_id"]: row for row in rows}
    if len(by_id) != 17 or sum(row["technical_valid"] for row in by_id.values()) != 17:
        raise RuntimeError("Expected 17 unique technically valid control generations")
    rows = sorted(by_id.values(), key=lambda row: (row["split"], row["source_order"]))

    point_contract = json.loads(POINTWISE_CONTRACT.read_text(encoding="utf-8"))["judge"]
    pair_contract = json.loads(PAIRWISE_CONTRACT.read_text(encoding="utf-8"))["judge"]
    if point_contract["model_path"] != LLAMA_MODEL or pair_contract["model_path"] != LLAMA_MODEL:
        raise RuntimeError("Local judge model contract changed")

    gpt_point_cache = gpt_pointwise_cache()
    llama_point_cache = llama_pointwise_cache(point_contract)
    gpt_pair_cache = prompt_cache(
        (
            PRIOR / "results/valid_pairwise_replay63_gpt41.jsonl",
            PRIOR / "results/valid_pairwise_new27_gpt41.jsonl",
        ),
        "GPT-4.1",
        "pairwise_four_axis",
        768,
    )
    llama_pair_cache = prompt_cache(
        (
            FEEDBACK_ROOT / "blind_judge_results.jsonl",
            BC_ROOT / "bc_blind_judge_results.jsonl",
            PRIOR / "results/valid_pairwise_new27_llama.jsonl",
        ),
        "Llama",
        "pairwise_four_axis",
        768,
    )
    selector_cache = prompt_cache(
        (
            PRIOR / "results/test_selector_gpt41.jsonl",
            PRIOR / "results/valid_selector_gpt41.jsonl",
        ),
        "GPT-4.1",
        "blind_selector",
        256,
    )

    assignments = []
    reuses = []
    point_gpt, point_llama = [], []
    pair_gpt, pair_llama = [], []
    selector_gpt = []
    for row in rows:
        key = pair_hash(row["question"], row["control_response"])
        assignment = {
            "split": row["split"],
            "canonical_id": row["canonical_id"],
            "case_id": row.get("case_id"),
            "source_order": row["source_order"],
            "condition": "original_prompt_sft_control",
            "question": row["question"],
            "response": row["control_response"],
            "question_response_sha256": key,
        }
        assignments.append(assignment)
        point_prompt = point_contract["user_prompt_template"].format(
            question=row["question"], response=row["control_response"]
        )
        metadata = {
            "split": row["split"],
            "canonical_id": row["canonical_id"],
            "case_id": row.get("case_id"),
            "source_order": row["source_order"],
            "condition": "original_prompt_sft_control",
            "question_response_sha256": key,
            "question_sha256": sha256_text(row["question"]),
            "response_sha256": sha256_text(row["control_response"]),
        }
        if key in gpt_point_cache:
            reuses.append({"evaluation": "pointwise", "judge": "GPT-4.1", **metadata, **gpt_point_cache[key]})
        else:
            point_gpt.append(build_request("control_pointwise_gpt41", "pointwise_four_axis", point_contract["system_prompt"], point_prompt, metadata, 384))
        if key in llama_point_cache:
            reuses.append({"evaluation": "pointwise", "judge": "Llama", **metadata, **llama_point_cache[key]})
        else:
            point_llama.append(build_request("control_pointwise_llama", "pointwise_four_axis", point_contract["system_prompt"], point_prompt, metadata, 384))

        pair_orders = (
            ("original_denoiser", row["original_candidate"], "original_prompt_sft_control", row["control_response"]),
            ("original_prompt_sft_control", row["control_response"], "original_denoiser", row["original_candidate"]),
        )
        for order_index, (variant_a, response_a, variant_b, response_b) in enumerate(pair_orders):
            user_prompt = pair_contract["user_prompt_template"].format(
                question=row["question"], response_a=response_a, response_b=response_b
            )
            pair_meta = {
                "split": row["split"],
                "canonical_id": row["canonical_id"],
                "case_id": row.get("case_id"),
                "source_order": row["source_order"],
                "comparison_id": f"{row['split']}::{row['canonical_id']}::original-v-control::{order_index}",
                "pair": ["original_denoiser", "original_prompt_sft_control"],
                "order_index": order_index,
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
            }
            cache_key = (pair_contract["system_prompt"], user_prompt)
            if cache_key in gpt_pair_cache:
                reuses.append({"evaluation": "pairwise", "judge": "GPT-4.1", **pair_meta, **gpt_pair_cache[cache_key]})
            else:
                pair_gpt.append(build_request("control_pairwise_gpt41", "pairwise_four_axis", pair_contract["system_prompt"], user_prompt, pair_meta, 768))
            if cache_key in llama_pair_cache:
                reuses.append({"evaluation": "pairwise", "judge": "Llama", **pair_meta, **llama_pair_cache[cache_key]})
            else:
                pair_llama.append(build_request("control_pairwise_llama", "pairwise_four_axis", pair_contract["system_prompt"], user_prompt, pair_meta, 768))

        selector_orders = (
            ("sft", row["sft_response"], "candidate", row["control_response"]),
            ("candidate", row["control_response"], "sft", row["sft_response"]),
        )
        for order_index, (variant_a, response_a, variant_b, response_b) in enumerate(selector_orders):
            user_prompt = SELECTOR_USER_TEMPLATE.format(
                question=row["question"], response_a=response_a, response_b=response_b
            )
            selector_meta = {
                "split": row["split"],
                "canonical_id": row["canonical_id"],
                "case_id": row.get("case_id"),
                "source_order": row["source_order"],
                "candidate_type": "original_prompt_sft_control",
                "order_index": order_index,
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "question_sha256": sha256_text(row["question"]),
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
            }
            cache_key = (SELECTOR_SYSTEM_PROMPT, user_prompt)
            if cache_key in selector_cache:
                reuses.append({"evaluation": "selector", "judge": "GPT-4.1", **selector_meta, **selector_cache[cache_key]})
            else:
                selector_gpt.append(build_request("control_selector_gpt41", "blind_selector", SELECTOR_SYSTEM_PROMPT, user_prompt, selector_meta, 256))

    queues = {
        "control_pointwise_gpt41": point_gpt,
        "control_pointwise_llama": point_llama,
        "control_pairwise_gpt41": pair_gpt,
        "control_pairwise_llama": pair_llama,
        "control_selector_gpt41": selector_gpt,
    }
    write_jsonl(ROOT / "results/control_pointwise_assignments.jsonl", assignments)
    write_jsonl(ROOT / "results/evaluation_cache_reuse.jsonl", reuses)
    for name, values in queues.items():
        if len({row["request_id"] for row in values}) != len(values):
            raise RuntimeError(f"Duplicate request IDs in {name}")
        write_jsonl(ROOT / f"requests/{name}.jsonl", values)
    counts = {
        "control_candidates": len(assignments),
        "expected_pointwise_per_judge": 17,
        "expected_pairwise_per_judge": 34,
        "expected_selector": 34,
        "requests": {name: len(values) for name, values in queues.items()},
        "reused": {
            f"{evaluation}::{judge}": sum(
                row["evaluation"] == evaluation and row["judge"] == judge for row in reuses
            )
            for evaluation, judge in (
                ("pointwise", "GPT-4.1"),
                ("pointwise", "Llama"),
                ("pairwise", "GPT-4.1"),
                ("pairwise", "Llama"),
                ("selector", "GPT-4.1"),
            )
        },
    }
    for judge in ("GPT-4.1", "Llama"):
        if counts["requests"][f"control_pointwise_{'gpt41' if judge == 'GPT-4.1' else 'llama'}"] + counts["reused"][f"pointwise::{judge}"] != 17:
            raise RuntimeError(f"Pointwise coverage mismatch for {judge}")
        if counts["requests"][f"control_pairwise_{'gpt41' if judge == 'GPT-4.1' else 'llama'}"] + counts["reused"][f"pairwise::{judge}"] != 34:
            raise RuntimeError(f"Pairwise coverage mismatch for {judge}")
    if counts["requests"]["control_selector_gpt41"] + counts["reused"]["selector::GPT-4.1"] != 34:
        raise RuntimeError("Selector coverage mismatch")
    contract = {
        "version": VERSION,
        "status": "prepared",
        "paid_api_authorized": True,
        "gpt_model": MODEL,
        "llama_model": LLAMA_MODEL,
        "post_hoc_test_comparison": True,
        "no_training": True,
        "no_response_regeneration": True,
        "pointwise_contract": {"path": str(POINTWISE_CONTRACT), "sha256": sha256_file(POINTWISE_CONTRACT)},
        "pairwise_contract": {"path": str(PAIRWISE_CONTRACT), "sha256": sha256_file(PAIRWISE_CONTRACT)},
        "selector_prompt_hashes": {
            "system": sha256_text(SELECTOR_SYSTEM_PROMPT),
            "user_template": sha256_text(SELECTOR_USER_TEMPLATE),
        },
        "cache_match_contract": {
            "pointwise": "question + response + rubric + model + temperature + max tokens",
            "pairwise": "system prompt + full user prompt including A/B order + model + temperature + max tokens",
            "parse_failures_are_not_ties": True,
        },
        "counts": counts,
        "request_files": {
            name: {"path": str(ROOT / f"requests/{name}.jsonl"), "sha256": sha256_file(ROOT / f"requests/{name}.jsonl"), "rows": len(values)}
            for name, values in queues.items()
        },
    }
    write_json(ROOT / "evaluation_contract.json", contract)
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
