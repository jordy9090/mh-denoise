#!/usr/bin/env python3
"""Guarded 24-row paired Gemma/Qwen diagnostic with a common GPT-4.1 judge."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from corruption_contract_v2 import (
    AXES,
    GeneratedStage,
    JudgeResult,
    StageRequest,
    acceptance_decision,
    build_judge_prompt,
    build_stage_prompt,
    judge_result_dict,
    parse_json_object,
    parse_judge_result,
    validate_judge_evidence,
)
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, normalize_text, read_jsonl, sha256_file, stable_random_key, write_json, write_jsonl
from run_gemma_corruption_pilot_v2 import balanced_specs
from source_integrity_contract import VERSION as SOURCE_INTEGRITY_VERSION, contract_hash as source_integrity_contract_hash, surface_flags


PILOT_SIZE = 24
PILOT_SEED = 20260907
LARGEST_GENERATOR_ESTIMATE_MIB = 16 * 1024
VRAM_SAFETY_MARGIN_MIB = 8 * 1024
DEFAULT_REQUIRED_FREE_VRAM_MIB = LARGEST_GENERATOR_ESTIMATE_MIB + VRAM_SAFETY_MARGIN_MIB
JUDGE_MODEL = "gpt-4.1-2025-04-14"
JUDGE_TEMPERATURE = 0
JUDGE_SEED = 20260907
JUDGE_MAX_TOKENS = 1200
JUDGE_API_PATH = "/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are an independent, conservative auditor of counseling-response degradation. "
    "Return only valid JSON matching the requested schema. Never infer the generator or its intent."
)
ELIGIBILITY_PROMPT = """Assess whether this source QA is suitable as a clean training target for SFT, a DPO chosen response,
and a reconstruction target. The response need not be ideal or stylistically perfect. Mark an axis only for a CLEAR, MATERIAL
defect—not a minor imperfection or failure to achieve the maximum possible score. Evaluate all six axes independently.

Axis definitions:
- overall_quality: a clear material failure of usefulness, coherence, organization, relevance, or responsiveness; do not
  flag merely because an otherwise useful response could be more polished or comprehensive.
- empathy: clearly detached, minimizing, dismissive, blaming, or invalidating framing; absence of maximal warmth alone is not a defect.
- specificity: clearly generic, non-actionable, irrelevant, or materially insufficient for the user's situation; useful general
  guidance is not a defect merely because more tailoring is possible.
- factual_consistency: a clear falsehood, fabrication, contradiction, or overconfident unsupported factual/psychological claim;
  do not flag cautious possibilities or ordinary non-controversial general advice merely for lacking citations.
- medical_boundary: clear unsupported diagnostic certainty or directive/prescriptive clinical treatment selection beyond
  informational support. Cautious possibility language, naming common therapy approaches, suggesting professional assessment,
  or mentioning general treatment options is not a material boundary violation by itself.
- toxicity_or_harm: clear blame, shame, stigmatization, harmful minimization, or dangerous/counterproductive guidance.

Also treat these observed integrity failures as clear material defects under the closest applicable axis: non-response
meta commentary about generation, a dialogue, counseling technique, or future/serial content; a false claim of human
personal experience or professional identity; a specific user history, diagnosis, or treatment history asserted as known
when unsupported by the question; or a switch into the user's first-person voice. Do not flag ordinary first-person
phrasing used by the responder. General career, relationship, communication, or other nonclinical life directiveness alone
is not clinical diagnosis or medical/treatment advice and must not be marked as medical_boundary.

For each material defect, provide evidence_source `response`, `question`, or `whole_response`. Copy a short exact verbatim span
from that source when using `response` or `question`; `whole_response` may have an empty span but requires a concrete reason.
For no material defect, use evidence_source `none`, empty evidence_span, and a short reason.

Return JSON only with exactly this shape:
{{"axes": {{
  "overall_quality": {{"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}},
  "empathy": {{"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}},
  "specificity": {{"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}},
  "factual_consistency": {{"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}},
  "medical_boundary": {{"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}},
  "toxicity_or_harm": {{"material_degradation": false, "evidence_source": "none", "evidence_span": "", "reason": ""}}
}}, "relevance_ok": true, "fluent": true, "useful_content": true,
"actionable_physical_harm": false, "summary_reason": "short reason"}}.

User message:
{question}

Source response:
{response}"""

GENERATORS = {
    "gemma": {
        "repo": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "snapshot": "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2",
        "enable_thinking": False,
    },
    "qwen": {
        "repo": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "snapshot": "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "enable_thinking": False,
    },
}


def prompt_sha256() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + ELIGIBILITY_PROMPT + "\n" + build_judge_prompt("{question}", "{clean}", "{candidate}")).encode()).hexdigest()


def load_repo_env() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def deterministic_generation_seed(
    generation_seed_base: int, canonical_id: str, stage_index: int, attempt: int
) -> int:
    material = f"{generation_seed_base}:{canonical_id}:{stage_index}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def generation_prompt_sha256() -> str:
    prompts = []
    for axis in AXES:
        prompts.append(build_stage_prompt(StageRequest("{id}", "train", "{question}", "{clean}", "{current}", (axis,), (), axis, 1, 0, 1)))
    return hashlib.sha256("\n---AXIS---\n".join(prompts).encode()).hexdigest()


def paired_specs(seed: int) -> list[tuple[str, ...]]:
    specs, stats = balanced_specs(PILOT_SIZE, seed)
    assert stats["axis_count_counts"] == {"1": 12, "2": 8, "3": 4}
    ordered = []
    for index, spec in enumerate(specs):
        ordered.append(tuple(sorted(spec, key=lambda axis: stable_random_key(seed, f"{index}:{axis}"))))
    return ordered


def candidate_rows(path: Path, seed: int) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row["split"] == "train"]
    rows.sort(key=lambda row: stable_random_key(seed, row["canonical_id"]))
    return rows


def strip_reasoning(text: str) -> tuple[str, bool]:
    original = text.strip()
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", original, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*<\|channel\>thought.*?<channel\|>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    residual = bool(re.search(r"</?think>|<\|channel\>thought|<channel\|>", cleaned, flags=re.IGNORECASE))
    return cleaned.strip(), residual or cleaned.strip() != original


class ExternalJudge:
    def __init__(
        self,
        base_url: str,
        max_tokens: int = JUDGE_MAX_TOKENS,
        budget_guard: Any | None = None,
        budget_max_input_tokens: int | None = None,
        budget_worker_id: str = "single-worker",
    ) -> None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; external GPT-4.1 QC is mandatory")
        self.key = key
        self.url = base_url.rstrip("/") + JUDGE_API_PATH
        self.max_tokens = max_tokens
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.elapsed = 0.0
        self.last_truncated: list[bool] = []
        self.returned_models: set[str] = set()
        self.system_fingerprints: set[str] = set()
        self._lock = threading.Lock()
        self.budget_guard = budget_guard
        self.budget_max_input_tokens = budget_max_input_tokens
        self.budget_worker_id = budget_worker_id
        if self.budget_guard is not None and not self.budget_max_input_tokens:
            raise ValueError("budget_max_input_tokens is required with a shared API budget")

    def call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "seed": JUDGE_SEED,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        }
        reservation_id = None
        if self.budget_guard is not None:
            # UTF-8 bytes are a conservative tokenizer-independent upper bound
            # on content tokens; the fixed allowance covers chat framing.
            prompt_token_upper_bound = len((SYSTEM_PROMPT + prompt).encode("utf-8")) + 256
            if prompt_token_upper_bound > int(self.budget_max_input_tokens):
                raise ValueError(
                    "Prompt may exceed the configured pre-reservation input-token ceiling: "
                    f"{prompt_token_upper_bound} > {self.budget_max_input_tokens}"
                )
            reservation_id = self.budget_guard.reserve(
                max_input_tokens=int(self.budget_max_input_tokens),
                max_output_tokens=int(self.max_tokens),
                worker_id=self.budget_worker_id,
                purpose=f"external_judge:{JUDGE_MODEL}",
            )
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
        except Exception as exc:
            if reservation_id is not None:
                self.budget_guard.fail(reservation_id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with self._lock:
                self.elapsed += time.monotonic() - started
                self.calls += 1
        usage = body.get("usage") or {}
        if reservation_id is not None:
            self.budget_guard.settle(
                reservation_id,
                actual_input_tokens=int(usage.get("prompt_tokens") or 0),
                actual_output_tokens=int(usage.get("completion_tokens") or 0),
            )
        with self._lock:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            if body.get("model"):
                self.returned_models.add(body["model"])
            if body.get("system_fingerprint"):
                self.system_fingerprints.add(body["system_fingerprint"])
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"judge truncation: {choice.get('finish_reason')}")
        raw = choice["message"]["content"]
        return raw, body

    def eligibility(self, row: dict[str, Any]) -> dict[str, Any]:
        raw, _ = self.call(ELIGIBILITY_PROMPT.format(question=row["question"], response=row["clean_response"]))
        result = parse_json_object(raw)
        if set(result.get("axes", {})) != set(AXES):
            raise ValueError("eligibility judge did not return all six axes")
        if any(not isinstance(result.get(k), bool) for k in ("relevance_ok", "fluent", "useful_content", "actionable_physical_harm")):
            raise ValueError("invalid eligibility booleans")
        raw_judge_axes = copy.deepcopy(result["axes"])
        degraded = []
        question_norm, response_norm = normalize_text(row["question"]), normalize_text(row["clean_response"])
        for axis in AXES:
            decision = result["axes"][axis]
            material = decision.get("material_degradation")
            source = decision.get("evidence_source")
            evidence = str(decision.get("evidence_span") or "").strip()
            reason = str(decision.get("reason") or "").strip()
            if not isinstance(material, bool) or source not in {"response", "question", "whole_response", "none"}:
                raise ValueError(f"invalid eligibility decision for {axis}")
            if material:
                if source == "none" or not reason:
                    raise ValueError(f"material degradation lacks evidence/reason for {axis}")
                if source == "response" and normalize_text(evidence) not in response_norm:
                    raise ValueError(f"response evidence is not verbatim for {axis}")
                if source == "question" and normalize_text(evidence) not in question_norm:
                    raise ValueError(f"question evidence is not verbatim for {axis}")
                degraded.append(axis)
            else:
                decision["evidence_source"] = "none"; decision["evidence_span"] = ""
        integrity_flags = surface_flags(row["question"], row["clean_response"], "clean")
        integrity_axis = {
            "non_response_meta": "overall_quality",
            "speaker_switch_question_copy": "overall_quality",
            "false_identity": "factual_consistency",
            "unsupported_specific_history": "factual_consistency",
        }
        for flag in integrity_flags:
            axis = integrity_axis[flag["kind"]]
            if axis not in degraded:
                degraded.append(axis)
            if not result["axes"][axis]["material_degradation"]:
                result["axes"][axis] = {
                    "material_degradation": True,
                    "evidence_source": "response",
                    "evidence_span": flag["text"],
                    "reason": "Deterministic narrow source-integrity check: " + flag["kind"],
                }
        globals_ok = result["relevance_ok"] and result["fluent"] and result["useful_content"] and not result["actionable_physical_harm"]
        if not globals_ok and not degraded:
            raise ValueError("global ineligibility must be represented by at least one material baseline axis")
        eligible = not degraded
        return {**result, "raw_judge_axes": raw_judge_axes, "source_integrity_flags": integrity_flags,
                "source_integrity_contract_version": SOURCE_INTEGRITY_VERSION,
                "source_integrity_contract_sha256": source_integrity_contract_hash(),
                "baseline_degraded_axes": degraded, "eligible": eligible,
                "reason": str(result.get("summary_reason") or "")}

    def judge(self, request: StageRequest, candidate: GeneratedStage) -> JudgeResult:
        raw, _ = self.call(build_judge_prompt(request.question, request.clean_response, candidate.text))
        result = parse_judge_result(parse_json_object(raw), raw)
        validate_judge_evidence(result, candidate=candidate.text, clean_response=request.clean_response, normalize=normalize_text)
        return result

    def metadata(self) -> dict[str, Any]:
        return {"requested_model": JUDGE_MODEL, "returned_models": sorted(self.returned_models), "system_fingerprints": sorted(self.system_fingerprints), "endpoint": JUDGE_API_PATH, "temperature": JUDGE_TEMPERATURE, "seed": JUDGE_SEED, "max_tokens": self.max_tokens, "response_format": "json_object", "prompt_sha256": prompt_sha256(), "calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "elapsed_seconds": self.elapsed}


@dataclass(frozen=True)
class LocalGeneration:
    text: str
    raw_output: str
    current_response_tokens: int
    max_new_tokens: int
    generated_tokens: int
    eos_reached: bool
    truncation_reason: str | None


def adaptive_max_new_tokens(current_response_tokens: int, ceiling: int = 1536) -> int:
    return min(max(current_response_tokens + 256, 768), ceiling)


def generation_stop_metadata(
    token_ids: list[int], eos_token_ids: set[int], max_new_tokens: int
) -> tuple[bool, str | None]:
    eos_reached = any(int(token_id) in eos_token_ids for token_id in token_ids)
    truncation_reason = (
        "max_new_tokens_reached_without_eos"
        if len(token_ids) >= max_new_tokens and not eos_reached
        else None
    )
    return eos_reached, truncation_reason


class LocalGenerator:
    def __init__(self, name: str, batch_size: int, max_input_tokens: int) -> None:
        self.name, self.spec = name, GENERATORS[name]
        snapshot = Path(self.spec["snapshot"])
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa")
        self.model.eval()
        self.batch_size, self.max_input_tokens = batch_size, max_input_tokens
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.elapsed = 0.0
        configured_eos = self.model.generation_config.eos_token_id
        if configured_eos is None:
            configured_eos = self.tokenizer.eos_token_id
        values = configured_eos if isinstance(configured_eos, (list, tuple)) else [configured_eos]
        self.eos_token_ids = {int(token_id) for token_id in values if token_id is not None}
        configured_pad = self.model.generation_config.pad_token_id
        self.pad_token_id = configured_pad if configured_pad is not None else self.tokenizer.pad_token_id

    def generate(
        self, requests: list[StageRequest], max_new_tokens_override: int | None = None
    ) -> list[LocalGeneration]:
        generated = []
        for request in requests:
            rendered = self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": build_stage_prompt(request)}]],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )[0]
            inputs = self.tokenizer(rendered, truncation=True, max_length=self.max_input_tokens, return_tensors="pt").to(self.model.device)
            current_response_tokens = len(self.tokenizer(request.current_response, add_special_tokens=False)["input_ids"])
            max_new_tokens = (
                max_new_tokens_override
                if max_new_tokens_override is not None
                else adaptive_max_new_tokens(current_response_tokens)
            )
            random.seed(request.generation_seed); torch.manual_seed(request.generation_seed); torch.cuda.manual_seed_all(request.generation_seed)
            started = time.monotonic()
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.75, top_p=0.9, pad_token_id=self.pad_token_id,
                    eos_token_id=sorted(self.eos_token_ids), use_cache=True,
                )[0]
            self.elapsed += time.monotonic() - started; self.calls += 1
            width = inputs["input_ids"].shape[1]
            self.prompt_tokens += int(inputs["attention_mask"].sum())
            token_ids = output[width:]
            actual_tokens = len(token_ids)
            eos_reached, truncation_reason = generation_stop_metadata(
                token_ids.tolist(), self.eos_token_ids, max_new_tokens
            )
            self.completion_tokens += actual_tokens
            raw = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            clean, reasoning = strip_reasoning(raw)
            generated.append(LocalGeneration(clean, raw, current_response_tokens, max_new_tokens, actual_tokens, eos_reached, truncation_reason))
        return generated

    def close(self) -> None:
        del self.model
        gc.collect(); torch.cuda.empty_cache()


def gpu_status(selected_gpu: int, required_free_vram_mib: int) -> dict[str, Any]:
    """Read-only VRAM admission check; existing compute contexts are informational."""
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    GPUs = []
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        GPUs.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "total_vram_mib": int(fields[3]),
                "free_vram_mib": int(fields[4]),
                "used_vram_mib": int(fields[5]),
            }
        )
    matches = [gpu for gpu in GPUs if gpu["index"] == selected_gpu]
    if len(matches) != 1:
        raise RuntimeError(f"selected physical GPU {selected_gpu} was not reported by nvidia-smi")
    status = matches[0]
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    processes = []
    for line in process_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[3] == status["uuid"]:
            processes.append(
                {"pid": int(fields[0]), "process_name": fields[1], "used_vram_mib": int(fields[2])}
            )
    status.update(
        {
            "selected_gpu": selected_gpu,
            "required_free_vram_mib": required_free_vram_mib,
            "sufficient_free_vram": status["free_vram_mib"] >= required_free_vram_mib,
            "existing_compute_processes": processes,
        }
    )
    return status


def preflight(canonical: Path, selected_gpu: int, required_free_vram_mib: int) -> dict[str, Any]:
    generators = {}
    for name, spec in GENERATORS.items():
        snapshot = Path(spec["snapshot"])
        generators[name] = {**spec, "chat_template_sha256": sha256_file(snapshot / "chat_template.jinja")}
    return {"pilot_size": 24, "split": "train", "axis_count_counts": {"1": 12, "2": 8, "3": 4}, "seed": PILOT_SEED, "canonical_sha256": sha256_file(canonical), "judge": {"available": bool(os.getenv("OPENAI_API_KEY")), "requested_model": JUDGE_MODEL, "prompt_sha256": prompt_sha256(), "temperature": 0, "seed": JUDGE_SEED, "max_tokens": JUDGE_MAX_TOKENS, "response_format": "json_object"}, "generation": {"prompt_sha256": generation_prompt_sha256(), "temperature": 0.75, "top_p": 0.9, "max_new_tokens_policy": "min(max(current_response_tokens + 256, 768), 1536)", "max_attempts_per_stage": 2}, "generators": generators, "gpu": gpu_status(selected_gpu, required_free_vram_mib), "evidence_contract": {"types": ["span", "omission", "whole_response", "none"], "span_source": "candidate", "omission_source": "clean_response"}}


def load_frozen_selection(path: Path, canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["canonical_id"]: row for row in canonical_rows}
    selected = []
    for slot, item in enumerate(payload["rows"]):
        clean = by_id[item["canonical_id"]]
        if clean["split"] != "train":
            raise AssertionError("frozen paired selection contains a non-TRAIN row")
        selected.append({"slot": slot, "clean": clean, "intended_axes": tuple(item["intended_axes"]), "axis_count": item["axis_count"], "eligibility": item["eligibility"]})
    if len(selected) != PILOT_SIZE:
        raise AssertionError("frozen paired selection is not 24 rows")
    return selected


def selected_length_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for name, spec in GENERATORS.items():
        tokenizer = AutoTokenizer.from_pretrained(Path(spec["snapshot"]), local_files_only=True)
        lengths = sorted(
            (
                len(tokenizer(item["clean"]["clean_response"], add_special_tokens=False)["input_ids"]),
                item["clean"]["canonical_id"],
            )
            for item in selected
        )
        points = {"shortest": lengths[0], "median": lengths[len(lengths) // 2], "longest": lengths[-1]}
        summary[name] = {
            label: {"canonical_id": item[1], "clean_response_tokens": item[0], "max_new_tokens": adaptive_max_new_tokens(item[0])}
            for label, item in points.items()
        }
        summary[name]["median_token_length_all_24"] = statistics.median(length for length, _ in lengths)
    return summary


def choose_eligible(judge: ExternalJudge, candidates: list[dict[str, Any]], specs: list[tuple[str, ...]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected, rejected = [], []
    cursor = 0
    for slot, axes in enumerate(specs):
        while cursor < len(candidates):
            row = candidates[cursor]; cursor += 1
            try:
                eligibility = judge.eligibility(row)
            except Exception as exc:
                rejected.append({"canonical_id": row["canonical_id"], "status": "eligibility_qc_failure", "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if eligibility["eligible"] and eligibility["baseline_degraded_axes"] == []:
                selected.append({"slot": slot, "clean": row, "intended_axes": axes, "axis_count": len(axes), "eligibility": eligibility})
                break
            rejected.append({"canonical_id": row["canonical_id"], "status": "source_ineligible", "intended_axes": list(axes), "baseline_degraded_axes": eligibility["baseline_degraded_axes"], "axis_evidence": eligibility["axes"], "reason": eligibility["reason"]})
        else:
            raise RuntimeError("candidate pool exhausted before 24 eligible rows were found")
    return selected, rejected


def refusal(text: str) -> bool:
    prefix = normalize_text(text)[:240]
    return any(term in prefix for term in ("i cannot help", "i can't help", "i cannot comply", "i'm unable to"))


def run_generator(name: str, selected: list[dict[str, Any]], judge: ExternalJudge, args, *, generation_seed_base: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    backend = LocalGenerator(name, args.batch_size, args.max_input_tokens)
    judge_start = (judge.calls, judge.prompt_tokens, judge.completion_tokens, judge.elapsed)
    states = [{**item, "current": item["clean"]["clean_response"], "completed": [], "history": [], "infrastructure_failures": [], "failed": False, "failure": None, "failure_class": None} for item in selected]
    try:
        for stage in range(3):
            pending = [s for s in states if not s["failed"] and len(s["intended_axes"]) > stage]
            feedback: dict[str, str] = {}
            for attempt in (1, 2):
                if not pending: break
                retry = []
                for state in pending:
                    request = StageRequest(state["clean"]["canonical_id"], "train", state["clean"]["question"], state["clean"]["clean_response"], state["current"], tuple(state["intended_axes"]), tuple(state["completed"]), state["intended_axes"][stage], stage + 1, deterministic_generation_seed(generation_seed_base, state["clean"]["canonical_id"], stage + 1, attempt), attempt, feedback.get(state["clean"]["canonical_id"]))
                    candidate = None
                    allowance = None
                    for infra_attempt in range(1, 5):
                        try:
                            candidate = backend.generate([request], allowance)[0]
                        except Exception as exc:
                            state["infrastructure_failures"].append({"kind": "generation_api_failure", "stage_index": stage + 1, "semantic_attempt": attempt, "infrastructure_attempt": infra_attempt, "reason": f"{type(exc).__name__}: {exc}"})
                            if infra_attempt < 4:
                                time.sleep(min(2 ** (infra_attempt - 1), 8))
                            continue
                        if not candidate.truncation_reason:
                            break
                        state["infrastructure_failures"].append({"kind": "generation_truncation", "stage_index": stage + 1, "semantic_attempt": attempt, "infrastructure_attempt": infra_attempt, "current_response_tokens": candidate.current_response_tokens, "max_new_tokens": candidate.max_new_tokens, "generated_tokens": candidate.generated_tokens, "eos_reached": candidate.eos_reached, "truncation_reason": candidate.truncation_reason})
                        if candidate.max_new_tokens >= 2048:
                            candidate = None
                            break
                        allowance = min(max(candidate.max_new_tokens + 256, candidate.max_new_tokens * 2), 2048)
                        candidate = None
                    if candidate is None:
                        state["failed"] = True; state["failure_class"] = "infrastructure"; state["failure"] = "generation_infrastructure_retries_exhausted"
                        continue

                    reason = None; result = None
                    reasoning_removed = normalize_text(candidate.raw_output) != normalize_text(candidate.text)
                    if not candidate.text: reason = "empty_generation"
                    elif refusal(candidate.text): reason = "refusal"
                    elif reasoning_removed and re.search(r"</?think>|<\|channel\>thought|<channel\|>", candidate.text, re.I): reason = "reasoning_leak"
                    else:
                        for judge_infra_attempt in range(1, 5):
                            try:
                                result = judge.judge(request, candidate)
                                break
                            except Exception as exc:
                                state["infrastructure_failures"].append({"kind": "judge_parse_or_evidence_failure", "stage_index": stage + 1, "semantic_attempt": attempt, "infrastructure_attempt": judge_infra_attempt, "reason": f"{type(exc).__name__}: {exc}"})
                                if judge_infra_attempt < 4:
                                    time.sleep(min(2 ** (judge_infra_attempt - 1), 8))
                        if result is None:
                            state["failed"] = True; state["failure_class"] = "infrastructure"; state["failure"] = "judge_infrastructure_retries_exhausted"
                            continue
                        _, reason = acceptance_decision(result, tuple(state["completed"]) + (request.target_axis,))
                    accepted = reason is None
                    event = {"stage_index": stage + 1, "target_axis": request.target_axis, "attempt": attempt, "seed": request.generation_seed, "generation_seed_base": generation_seed_base, "candidate_response": candidate.text, "raw_generation": candidate.raw_output, "reasoning_removed": reasoning_removed, "current_response_tokens": candidate.current_response_tokens, "max_new_tokens": candidate.max_new_tokens, "generated_tokens": candidate.generated_tokens, "eos_reached": candidate.eos_reached, "truncation_reason": candidate.truncation_reason, "judge": judge_result_dict(result) if result else None, "accepted": accepted, "failure_reason": reason}
                    state["history"].append(event)
                    if accepted:
                        state["current"] = candidate.text; state["completed"].append(request.target_axis)
                    else:
                        feedback[state["clean"]["canonical_id"]] = reason or "rejected"; retry.append(state)
                pending = retry
            for state in pending:
                if not state["failed"]:
                    state["failed"] = True; state["failure_class"] = "semantic"; state["failure"] = feedback.get(state["clean"]["canonical_id"], "stage_failed")
        records, failures = [], []
        for state in states:
            base = {"canonical_id": state["clean"]["canonical_id"], "source": state["clean"]["source"], "split": "train", "question": state["clean"]["question"], "clean_response": state["clean"]["clean_response"], "eligibility": state["eligibility"], "intended_axes": list(state["intended_axes"]), "axis_count": state["axis_count"], "generator_repo": backend.spec["repo"], "generator_revision": backend.spec["revision"], "enable_thinking": False, "judge_requested_model": JUDGE_MODEL, "judge_prompt_sha256": prompt_sha256(), "stage_history": state["history"], "infrastructure_failures": state["infrastructure_failures"]}
            if state["failed"]:
                failures.append({**base, "completion_status": "dropped", "completed_axes": state["completed"], "failure_class": state["failure_class"], "failure_reason": state["failure"]})
            else:
                final_judge = state["history"][-1]["judge"]
                realized_axes = final_judge["realized_axes"]
                records.append({**base, "completion_status": "accepted", "corrupted_response": state["current"], "realized_axes": realized_axes, "unintended_axes": [axis for axis in realized_axes if axis not in state["intended_axes"]], "axis_judgments": final_judge["axes"], "qc_pass": True})
        metrics = {"generator_calls": backend.calls, "generator_prompt_tokens": backend.prompt_tokens, "generator_completion_tokens": backend.completion_tokens, "generator_elapsed_seconds": backend.elapsed, "judge_calls": judge.calls - judge_start[0], "judge_prompt_tokens": judge.prompt_tokens - judge_start[1], "judge_completion_tokens": judge.completion_tokens - judge_start[2], "judge_elapsed_seconds": judge.elapsed - judge_start[3]}
        return records, failures, metrics
    finally:
        backend.close()


def report(results: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]], judge_meta: dict[str, Any], path: Path) -> None:
    lines = ["# Paired Gemma/Qwen corruption diagnostic", "", "Both generators used the same 24 eligible TRAIN rows, ordered axes, semantic stage prompt, two-attempt budget, and blinded external GPT-4.1 judge.", ""]
    for name, (records, failures, metrics) in results.items():
        all_rows = records + failures
        events = [e for r in all_rows for e in r["stage_history"]]
        first = sum(r["completion_status"] == "accepted" and all(e["attempt"] == 1 for e in r["stage_history"] if e["accepted"]) for r in all_rows)
        lines += [f"## {name}", "", f"- First-attempt full yield: {first}/24", f"- Final yield: {len(records)}/24", f"- Refusals: {sum(e['failure_reason']=='refusal' for e in events)}", f"- Generation truncations: {sum(e['failure_reason']=='generation_truncation' for e in events)}", f"- Judge/parse/evidence failures: {sum(str(e['failure_reason']).startswith('judge_or_evidence_failure') for e in events)}", f"- Calls: {metrics['generator_calls']} generator + {metrics['judge_calls']} judge", f"- Tokens: {metrics['generator_prompt_tokens'] + metrics['judge_prompt_tokens']} prompt + {metrics['generator_completion_tokens'] + metrics['judge_completion_tokens']} completion", f"- Combined measured elapsed per accepted example: {(metrics['generator_elapsed_seconds'] + metrics['judge_elapsed_seconds'])/len(records):.2f}s" if records else "- Combined measured elapsed per accepted example: undefined (zero accepted)", "", "| Axis | Stage successes / attempts | Final retained / requested |", "|---|---:|---:|"]
        for axis in AXES:
            axis_events = [e for e in events if e["target_axis"] == axis]
            requested = sum(axis in r["intended_axes"] for r in all_rows)
            retained = sum(axis in r.get("realized_axes", []) for r in records if axis in r["intended_axes"])
            lines.append(f"| `{axis}` | {sum(e['accepted'] for e in axis_events)}/{len(axis_events)} | {retained}/{requested} |")
        extra = sum(len(set(r["realized_axes"]) - set(r["intended_axes"])) for r in records)
        slots = sum(len(AXES) - len(r["intended_axes"]) for r in records)
        lines += ["", f"Overlapping/unintended degradation: {extra}/{slots} available unselected slots.", ""]
        lines += ["### Concrete Overall and factual examples", "", "| Axis | Outcome | ID | Candidate excerpt | QC reason |", "|---|---|---|---|---|"]
        for axis in ("overall_quality", "factual_consistency"):
            successes = [(r, e) for r in all_rows for e in r["stage_history"] if e["target_axis"] == axis and e["accepted"]]
            rejected = [(r, e) for r in all_rows for e in r["stage_history"] if e["target_axis"] == axis and not e["accepted"]]
            for outcome, items in (("success", successes[:1]), ("failure", rejected[:1])):
                if not items:
                    lines.append(f"| `{axis}` | {outcome} | — | — | no example |")
                    continue
                row, event = items[0]
                excerpt = event["candidate_response"][:300].replace("\n", " ").replace("|", "\\|")
                if event["judge"]:
                    decision = event["judge"]["axes"][axis]
                    reason = decision["reason"]
                else:
                    reason = event["failure_reason"]
                escaped_reason = str(reason).replace("|", "\\|")
                lines.append(f"| `{axis}` | {outcome} | `{row['canonical_id']}` | {excerpt} | {escaped_reason} |")
        lines.append("")
    lines += ["## Common judge", "", "```json", json.dumps(judge_meta, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "paired_generator_diagnostic_24"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--selected-gpu", type=int, default=0)
    parser.add_argument("--required-free-vram-mib", type=int, default=DEFAULT_REQUIRED_FREE_VRAM_MIB)
    args = parser.parse_args()
    canonical, out = Path(args.canonical).resolve(), Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    info = preflight(canonical, args.selected_gpu, args.required_free_vram_mib); write_json(out / "preflight.json", info)
    print(json.dumps({"gpu": info["gpu"]}), flush=True)
    specs = paired_specs(PILOT_SEED)
    candidates = candidate_rows(canonical, PILOT_SEED)
    pending = [{"slot": i, "intended_axes": list(spec), "axis_count": len(spec)} for i, spec in enumerate(specs)]
    write_json(out / "pending_selection.json", {"status": "pending_external_eligibility", "candidate_ids": [r["canonical_id"] for r in candidates[:96]], "slots": pending})
    if args.preflight_only:
        print(json.dumps(info)); return
    frozen_selection_path = out / "selection.json"
    selected = None
    if frozen_selection_path.exists():
        selected = load_frozen_selection(frozen_selection_path, candidates)
        length_summary = selected_length_summary(selected)
        write_json(out / "adaptive_length_preflight.json", length_summary)
        print(json.dumps({"adaptive_length_preflight": length_summary}), flush=True)
    if not info["judge"]["available"]:
        raise RuntimeError("External GPT-4.1 judge unavailable; refusing to generate with self-QC")
    if not info["gpu"]["sufficient_free_vram"]:
        raise RuntimeError(
            f"insufficient free VRAM on GPU {args.selected_gpu}: "
            f"{info['gpu']['free_vram_mib']} MiB < {args.required_free_vram_mib} MiB required"
        )
    judge = ExternalJudge(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    if selected is None:
        selected, ineligible = choose_eligible(judge, candidates, specs)
        write_json(out / "selection.json", {"status": "eligible", "seed": PILOT_SEED, "rows": [{"canonical_id": s["clean"]["canonical_id"], "split": "train", "intended_axes": list(s["intended_axes"]), "axis_count": s["axis_count"], "eligibility": s["eligibility"]} for s in selected]})
        write_jsonl(out / "source_ineligible.jsonl", ineligible)
        length_summary = selected_length_summary(selected)
        write_json(out / "adaptive_length_preflight.json", length_summary)
        print(json.dumps({"adaptive_length_preflight": length_summary}), flush=True)
    results = {}
    for name in ("gemma", "qwen"):
        current_gpu = gpu_status(args.selected_gpu, args.required_free_vram_mib)
        print(json.dumps({"generator": name, "gpu": current_gpu}), flush=True)
        if not current_gpu["sufficient_free_vram"]:
            raise RuntimeError(
                f"insufficient free VRAM before {name}: {current_gpu['free_vram_mib']} MiB "
                f"< {args.required_free_vram_mib} MiB required"
            )
        result = run_generator(name, selected, judge, args, generation_seed_base=PILOT_SEED)
        results[name] = result
        write_jsonl(out / f"{name}_accepted.jsonl", result[0])
        write_jsonl(out / f"{name}_failures.jsonl", result[1])
    judge_meta = judge.metadata()
    write_json(out / "judge_manifest.json", judge_meta)
    report(results, judge_meta, out / "paired_report.md")
    print(json.dumps({name: {"accepted": len(value[0]), "failed": len(value[1])} for name, value in results.items()}))


if __name__ == "__main__":
    main()
