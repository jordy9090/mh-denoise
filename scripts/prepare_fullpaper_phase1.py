#!/usr/bin/env python3
"""Build Phase 1 provenance and contamination manifests for the full-paper data.

This script is intentionally limited to inventory, integrity fingerprints, and
duplicate-candidate discovery.  It does not create train/validation/test splits,
corruptions, prompts for training, or model outputs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.ipc as ipc
from sklearn.feature_extraction.text import TfidfVectorizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/fullpaper_phase1/manifests"
RAW_ROOT = REPO_ROOT / "data/fullpaper_phase1/raw_sources"
COUNSELBENCH_ARROW = Path(
    "/home/user/.cache/huggingface/datasets/izi-ano___counsel_bench-eval/"
    "default/0.0.0/8d56a96ea1de3f3f190f77f4ca9bc3503d731af7/"
    "counsel_bench-eval-test.arrow"
)

SOURCE_REVISIONS = {
    "CounselBench-100": "8d56a96ea1de3f3f190f77f4ca9bc3503d731af7",
    "CounselChat": "17501f72697cf8018aaf496162e9dc1408a64e67",
    "MentalChat16K": "5f60cd380cfc58f0f12d44892bed41ee3670a70a",
    "Psych8k": "091787feccbce3e0adfd03b1ea3063f3d938c32d",
    "PsyQA": "e224c7e518c98a0c3df11e2fc5e6698044d8e156",
}

QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def clean_display(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value).replace("\r\n", "\n").replace("\r", "\n")).strip()


def raw_text(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(QUOTE_TRANSLATION).lower()
    return SPACE_RE.sub(" ", text).strip()


def combined_question(title: Any, body: Any) -> str:
    title_text = raw_text(title)
    body_text = raw_text(body)
    if title_text and body_text and title_text != body_text:
        return f"{title_text}\n\n{body_text}"
    return title_text or body_text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def file_record(path: Path, include_rows: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return record
    record.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if include_rows:
        if path.suffix == ".jsonl":
            rows = []
            invalid_rows = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        invalid_rows += 1
            record["rows"] = len(rows)
            record["invalid_json_rows"] = invalid_rows
            distributions = {}
            for field in ["judge_model", "judge_ok", "system"]:
                values = Counter(str(row.get(field)) for row in rows if field in row)
                if values:
                    distributions[field] = dict(values)
            if distributions:
                record["field_distributions"] = distributions
            if rows and any("example_id" in row for row in rows):
                record["unique_example_ids"] = len(
                    {row.get("example_id") for row in rows if row.get("example_id") is not None}
                )
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                record["rows"] = sum(1 for _ in csv.DictReader(handle))
    return record


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def parse_python_constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str):
                        raise TypeError(f"{name} is not a string")
                    return value
    raise KeyError(f"Missing Python constant {name} in {path}")


def checkpoint_record(name: str, path: Path, role: str) -> dict[str, Any]:
    config_names = [
        "adapter_config.json",
        "risk_tune_config.json",
        "config.json",
        "dims.json",
        "patterns.json",
        "eval_metrics.json",
        "trainer_state.json",
    ]
    configs = []
    for filename in config_names:
        candidate = path / filename
        if candidate.is_file():
            item = file_record(candidate)
            try:
                item["content"] = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:  # preserve fingerprint even if config parsing fails
                item["parse_error"] = f"{type(exc).__name__}: {exc}"
            configs.append(item)
    weights = []
    for filename in ["adapter_model.safetensors", "model.safetensors"]:
        candidate = path / filename
        if candidate.is_file():
            weights.append(file_record(candidate))
    return {
        "name": name,
        "role": role,
        "path": str(path.resolve()),
        "exists": path.is_dir(),
        "configs": configs,
        "weights": weights,
    }


def build_exp295_freeze() -> dict[str, Any]:
    run_script = REPO_ROOT / "run_exp295_test_main_table_len256.sh"
    judge_script = REPO_ROOT / "scripts/run_refinement_llm_judge.py"
    run_text = run_script.read_text(encoding="utf-8")
    judge_model_match = re.search(r"^JUDGE_MODEL=(\S+)$", run_text, flags=re.MULTILINE)
    judge_model = judge_model_match.group(1) if judge_model_match else "unresolved"

    checkpoints = [
        checkpoint_record(
            "sft_plain",
            REPO_ROOT / "outputs/models/gemma4_peft_sft_plain_exp295/final",
            "SFT baseline and initialization",
        ),
        checkpoint_record(
            "dpo_checkpoint_200",
            REPO_ROOT
            / "outputs/models/gemma4_dpo_minimal_legacy_exp295_beta0p1_seed42_mc512/checkpoint-200",
            "DPO evaluated checkpoint",
        ),
        checkpoint_record(
            "dpo_checkpoint_234",
            REPO_ROOT
            / "outputs/models/gemma4_dpo_minimal_legacy_exp295_beta0p1_seed42_mc512/checkpoint-234",
            "DPO final optimizer checkpoint",
        ),
        checkpoint_record(
            "dpo_final",
            REPO_ROOT / "outputs/models/gemma4_dpo_minimal_legacy_exp295_beta0p1_seed42_mc512/final",
            "DPO exported final adapter",
        ),
        checkpoint_record(
            "denoising_sft_no_risk_weight",
            REPO_ROOT / "outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/best",
            "lambda=0 ablation",
        ),
        checkpoint_record(
            "risk_aware_denoising_refiner",
            REPO_ROOT / "outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/best",
            "Proposed risk-weighted refiner",
        ),
        checkpoint_record(
            "selective_risk_tuned",
            REPO_ROOT
            / "outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/best",
            "Selective proposed risk adapter",
        ),
        checkpoint_record(
            "aspect_router",
            REPO_ROOT / "outputs/models/aspect_router_exp295_multilabel/final",
            "Selective/proposed router",
        ),
        checkpoint_record(
            "span_risk_scorer",
            REPO_ROOT / "outputs/models/span_risk_multilabel_v1/best",
            "Selective/proposed span-risk scorer",
        ),
    ]

    result_paths = [
        "outputs/refinement/sft_plain_exp295_test_outputs_len256.jsonl",
        "outputs/refinement/denoising_sft_no_risk_weight_exp295_test_len256_t2.jsonl",
        "outputs/refinement/risk_aware_denoising_refiner_exp295_test_len256_t2.jsonl",
        "outputs/refinement/selective_sft_plain_risk_tuned_exp295_test_len256_medical_focus_th001_t2.jsonl",
        "outputs/refinement/selective_sft_plain_risk_tuned_exp295_test_len256_medical_focus_th001_t2_trunc_filtered.jsonl",
        "outputs/refinement/dpo_minimal_legacy_exp295_test_beta0p1_seed42_mc512_checkpoint200_len256.jsonl",
        "outputs/refinement/dpo_minimal_legacy_exp295_test_beta0p1_seed42_mc512_len256.jsonl",
        "outputs/eval_inputs/exp295_test_main_table_len256_counselbench_input.jsonl",
        "outputs/eval/exp295_test_main_table_len256_counselbench_judged.jsonl",
        "outputs/analysis/exp295_test_main_table_len256_counselbench_by_system.csv",
        "outputs/eval_inputs/dpo_minimal_legacy_exp295_checkpoint200_len256_counselbench_input.jsonl",
        "outputs/eval/dpo_minimal_legacy_exp295_checkpoint200_len256_counselbench_judged.jsonl",
        "outputs/eval_inputs/dpo_minimal_legacy_exp295_final234_len256_counselbench_input.jsonl",
        "outputs/eval/dpo_minimal_legacy_exp295_final234_len256_counselbench_judged.jsonl",
        "outputs/eval/exp295_test_main_table_plus_dpo_checkpoint200_len256_counselbench_judged.jsonl",
        "outputs/analysis/exp295_test_main_table_plus_dpo_checkpoint200_len256_counselbench_by_system.csv",
    ]

    provenance_files = [
        "data/raw/exp295_safe_targets.jsonl",
        "data/splits_exp295/train_mdlm.jsonl",
        "data/splits_exp295/valid_mdlm.jsonl",
        "data/splits_exp295/test.jsonl",
        "data/dpo_exp295_minimal/train.jsonl",
        "data/dpo_exp295_minimal/valid.jsonl",
        "data/dpo_exp295_minimal/manifest.json",
        "outputs/models/gemma4_peft_sft_plain_exp295/train_args.json",
        "outputs/models/gemma4_dpo_minimal_legacy_exp295_beta0p1_seed42_mc512/preflight_manifest.json",
        "outputs/models/gemma4_dpo_minimal_legacy_exp295_beta0p1_seed42_mc512/training_manifest.json",
        "run_exp295_test_main_table_len256.sh",
        "scripts/run_refinement_llm_judge.py",
    ]

    counselbench_prompt = parse_python_constant(judge_script, "COUNSELBENCH_TEMPLATE")
    counselbench_system = parse_python_constant(judge_script, "COUNSELBENCH_SYSTEM_PROMPT")
    return {
        "freeze_version": "exp295-integrity-freeze-v1",
        "scope": "CIKM/legacy baseline-check only; forbidden as full-paper main data",
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "status_short": git_value("status", "--short").splitlines(),
        },
        "checkpoints": checkpoints,
        "provenance_files": [file_record(REPO_ROOT / item, include_rows=True) for item in provenance_files],
        "results": [file_record(REPO_ROOT / item, include_rows=True) for item in result_paths],
        "judge": {
            "model": judge_model,
            "api": "OpenAI chat.completions",
            "rubric_style": "counselbench",
            "system_prompt": counselbench_system,
            "system_prompt_sha256": text_hash(counselbench_system),
            "prompt_template": counselbench_prompt,
            "prompt_template_sha256": text_hash(counselbench_prompt),
            "dimensions": [
                "Overall",
                "Empathy",
                "Specificity",
                "Medical Advice",
                "Factual Consistency",
                "Toxicity",
            ],
            "decoding": {
                "temperature": 0,
                "max_tokens": 500,
                "seed": None,
                "top_p": "API default (not set)",
                "max_retries": 3,
                "sleep_seconds": 0.5,
                "resume": True,
            },
            "source_script": file_record(judge_script),
            "invocation_script": file_record(run_script),
        },
    }


def load_counselbench() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with COUNSELBENCH_ARROW.open("rb") as handle:
        table = ipc.open_stream(handle).read_all()
    raw_rows = table.to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["questionID"])].append(row)

    questions = []
    for qid in sorted(grouped, key=lambda value: int(re.fullmatch(r"questionID_(\d+)", value).group(1))):
        rows = grouped[qid]
        first = rows[0]
        numeric_id = int(re.fullmatch(r"questionID_(\d+)", qid).group(1))
        title = raw_text(first.get("questionTitle"))
        body = raw_text(first.get("questionText"))
        question = combined_question(title, body)
        responses = sorted({raw_text(row.get("response")) for row in rows if raw_text(row.get("response"))})
        questions.append(
            {
                "source": "CounselBench-100",
                "record_id": qid,
                "question_id": qid,
                "numeric_question_id": numeric_id,
                "question_title": title,
                "question_body": body,
                "question": question,
                "question_normalized": normalize_text(question),
                "topic": clean_display(first.get("topic")),
                "survey_ids": sorted({row.get("survey_id") for row in rows}),
                "responders": sorted({clean_display(row.get("responder")) for row in rows}),
                "response_count": len(rows),
                "unique_responses": responses,
            }
        )
    metadata = {
        "path": str(COUNSELBENCH_ARROW),
        "bytes": COUNSELBENCH_ARROW.stat().st_size,
        "sha256": sha256_file(COUNSELBENCH_ARROW),
        "evaluation_rows": len(raw_rows),
        "unique_questions": len(questions),
        "schema": table.schema.names,
    }
    return questions, metadata


def load_counselchat() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = RAW_ROOT / "counsel_chat/20220401_counsel_chat.csv"
    rows = read_csv(path)
    records = []
    for index, row in enumerate(rows):
        title = raw_text(row.get("questionTitle"))
        body = raw_text(row.get("questionText"))
        records.append(
            {
                "source": "CounselChat",
                "source_component": "counselchat",
                "record_id": f"counselchat:row:{index}",
                "cluster_id": f"counselchat:questionID:{row.get('questionID')}",
                "question_id": str(row.get("questionID", "")),
                "question_title": title,
                "question_body": body,
                "question": combined_question(title, body),
                "response": raw_text(row.get("answerText")),
                "metadata": dict(row),
            }
        )
    return records, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "schema": list(rows[0]) if rows else [],
    }


def load_mentalchat() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    files = []
    for component, filename in [
        ("interview", "Interview_Data_6K.csv"),
        ("synthetic", "Synthetic_Data_10K.csv"),
    ]:
        path = RAW_ROOT / "mentalchat16k" / filename
        rows = read_csv(path)
        for index, row in enumerate(rows):
            records.append(
                {
                    "source": "MentalChat16K",
                    "source_component": component,
                    "record_id": f"mentalchat16k:{component}:row:{index}",
                    "cluster_id": f"mentalchat16k:{component}:row:{index}",
                    "question_id": "",
                    "question_title": "",
                    "question_body": raw_text(row.get("input")),
                    "question": raw_text(row.get("input")),
                    "response": raw_text(row.get("output")),
                    "metadata": dict(row),
                }
            )
        files.append(
            {
                "component": component,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(rows),
                "schema": list(rows[0]) if rows else [],
            }
        )
    return records, files


def load_psyqa_sample() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = RAW_ROOT / "psyqa_official_repo/PsyQA_example.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    records = []
    answer_rows = 0
    for index, row in enumerate(rows):
        answers = row.get("answers") or []
        answer_rows += len(answers)
        title = raw_text(row.get("question"))
        body = raw_text(row.get("description"))
        records.append(
            {
                "source": "PsyQA-sample",
                "source_component": "official_sample",
                "record_id": f"psyqa:questionID:{row.get('questionID', index)}",
                "cluster_id": f"psyqa:questionID:{row.get('questionID', index)}",
                "question_id": str(row.get("questionID", "")),
                "question_title": title,
                "question_body": body,
                "question": combined_question(title, body),
                "responses": [raw_text(answer.get("answer_text")) for answer in answers],
                "metadata": dict(row),
            }
        )
    return records, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "question_rows": len(rows),
        "answer_rows": answer_rows,
        "schema": list(rows[0]) if rows else [],
        "answer_schema": list(rows[0]["answers"][0]) if rows and rows[0].get("answers") else [],
    }


def missing_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    if not rows:
        return {}
    return {
        field: sum(not clean_display(row.get(field)) for row in rows)
        for field in rows[0]
    }


def build_candidate_inventory(
    counselchat_meta: dict[str, Any],
    mental_files: list[dict[str, Any]],
    psyqa_meta: dict[str, Any],
) -> dict[str, Any]:
    counsel_rows = read_csv(Path(counselchat_meta["path"]))
    mental_by_component = {
        item["component"]: read_csv(Path(item["path"])) for item in mental_files
    }
    counsel_questions = {
        combined_question(row.get("questionTitle"), row.get("questionText")) for row in counsel_rows
    }
    counsel_normalized = {normalize_text(value) for value in counsel_questions if normalize_text(value)}
    mental_unique = {
        component: {
            "raw_exact_questions": len({raw_text(row.get("input")) for row in rows if raw_text(row.get("input"))}),
            "display_canonical_questions": len({clean_display(row.get("input")) for row in rows if clean_display(row.get("input"))}),
            "normalized_questions": len({normalize_text(row.get("input")) for row in rows if normalize_text(row.get("input"))}),
        }
        for component, rows in mental_by_component.items()
    }
    return {
        "inventory_version": "fullpaper-candidate-intake-v1",
        "decision_status": "inventory_only; no final inclusion decision",
        "datasets": [
            {
                "dataset": "MentalChat16K",
                "official_source": "https://huggingface.co/datasets/ShenLab/MentalChat16K",
                "revision": SOURCE_REVISIONS["MentalChat16K"],
                "license": "MIT (dataset card)",
                "access_status": "recovered locally at pinned revision",
                "raw_files": mental_files,
                "observed_total_rows": sum(item["rows"] for item in mental_files),
                "observed_unique_questions_by_component": mental_unique,
                "documented_counts": {"interview": 6338, "synthetic": 9775, "sum": 16113},
                "schema": ["instruction", "input", "output"],
                "original_ids": [],
                "available_group_ids": ["source_component (derived from immutable source filename)"],
                "missing_fields": [
                    "original row ID",
                    "source label within a merged table",
                    "topic",
                    "interview transcript/document/session/recording ID",
                    "turn/page index",
                ],
                "missing_counts": {
                    component: missing_counts(rows) for component, rows in mental_by_component.items()
                },
                "single_turn_suitability": "Structurally suitable as instruction/input/output QA; interview rows are transcript-derived summaries, but session grouping cannot be verified from released fields.",
                "blockers": [
                    "README counts exceed observed CSV counts by 29 rows in total.",
                    "The 378 source transcripts described by the card cannot be reconstructed into groups from the released three-column CSV schema.",
                ],
            },
            {
                "dataset": "CounselChat",
                "official_source": "https://huggingface.co/datasets/nbertagnolli/counsel-chat",
                "revision": SOURCE_REVISIONS["CounselChat"],
                "license": "MIT (README; Hub card metadata does not expose a license field)",
                "access_status": "recovered locally at pinned revision",
                "raw_files": [counselchat_meta],
                "observed_total_rows": counselchat_meta["rows"],
                "observed_unique_question_ids": len({row.get("questionID") for row in counsel_rows}),
                "observed_unique_question_urls": len({row.get("questionLink") for row in counsel_rows}),
                "observed_unique_raw_exact_combined_questions": len(counsel_questions),
                "observed_unique_normalized_combined_questions": len(counsel_normalized),
                "schema": counselchat_meta["schema"],
                "original_ids": ["questionID", "questionLink"],
                "available_group_ids": ["questionID", "questionLink", "therapistURL"],
                "missing_fields": ["response ID", "source session ID", "recording ID", "original row ID"],
                "missing_counts": missing_counts(counsel_rows),
                "single_turn_suitability": "Suitable after grouping all therapist responses by questionID and excluding CounselBench-linked question clusters.",
                "blockers": [
                    "Raw source includes identifiable therapist information and is explicitly described as non-anonymized.",
                    "Multiple response rows per question require group-aware handling and a separately documented response-selection policy.",
                ],
            },
            {
                "dataset": "Psych8k",
                "official_source": "https://huggingface.co/datasets/EmoCareAI/Psych8k",
                "upstream_project": "https://github.com/EmoCareAI/ChatPsychiatrist",
                "revision": SOURCE_REVISIONS["Psych8k"],
                "upstream_project_revision_observed": "7d131fae7f797e965ac19413a29722f236a29d5b",
                "license": "CC-BY-NC-SA-4.0 (dataset card)",
                "access_status": "blocked: manual gated access and contact-information agreement required",
                "raw_files": [
                    {
                        "expected_filename": "Alexander_Street_shareGPT_2.0.json",
                        "remote_bytes": 6575030,
                        "raw_sha256": None,
                        "local_path": None,
                    }
                ],
                "documented_rows": 8187,
                "observed_total_rows": None,
                "schema": None,
                "expected_format_not_verified": "ShareGPT-style conversations inferred from filename; must be verified after authorized download",
                "original_ids": None,
                "available_group_ids": None,
                "missing_fields": "confirmation pending authorized raw access",
                "single_turn_suitability": "Documented as 8,187 single-turn QA pairs derived via GPT-4 from about 260 counseling recordings; recording-level IDs must be verified before use.",
                "blockers": [
                    "Manual Hugging Face gate has not been accepted in this Phase 1 run.",
                    "Raw hash, schema, IDs, recording groups, and overlaps cannot be reported without authorized access.",
                    "Non-commercial ShareAlike license compatibility requires project-level review.",
                ],
            },
            {
                "dataset": "PsyQA",
                "official_source": "https://github.com/thu-coai/PsyQA",
                "revision": SOURCE_REVISIONS["PsyQA"],
                "license": "No open-data license stated; full dataset is controlled by a signed user agreement",
                "access_status": "official 100-question sample recovered; full dataset unavailable without agreement and author approval",
                "raw_files": [psyqa_meta],
                "observed_sample_questions": psyqa_meta["question_rows"],
                "observed_sample_answers": psyqa_meta["answer_rows"],
                "full_row_count": None,
                "schema": psyqa_meta["schema"],
                "answer_schema": psyqa_meta["answer_schema"],
                "original_ids": ["questionID"],
                "available_group_ids": ["questionID"],
                "missing_fields": ["source document/session/user ID", "timestamp", "full dataset files"],
                "single_turn_suitability": "Question/description with one or more long Chinese answers is structurally single-turn, but it is language-mismatched to the current English setting and full-data access is controlled.",
                "blockers": [
                    "Only the official sample can be audited locally.",
                    "Full-data license/usage terms require the signed agreement; unofficial mirrors are not treated as authoritative.",
                    "English duplicate and semantic detectors are not valid for Chinese semantic review.",
                ],
            },
        ],
    }


def collapse_questions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["cluster_id"]].append(record)
    collapsed = []
    for cluster_id, rows in sorted(grouped.items()):
        first = rows[0]
        question = first["question"]
        responses = []
        for row in rows:
            if row.get("response"):
                responses.append(row["response"])
            responses.extend(value for value in row.get("responses", []) if value)
        collapsed.append(
            {
                "source": first["source"],
                "source_component": first.get("source_component"),
                "record_id": cluster_id,
                "question_id": first.get("question_id", ""),
                "question": question,
                "question_normalized": normalize_text(question),
                "member_record_ids": [row["record_id"] for row in rows],
                "response_rows": len(rows) if first["source"] == "CounselChat" else len(responses),
                "responses": responses,
                "metadata": first.get("metadata", {}),
            }
        )
    return collapsed


def overlap_summary(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    def maps(rows: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
        output: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            value = row[field]
            if value:
                output[value].append(row["record_id"])
        return output

    left_exact = maps(left, "question")
    right_exact = maps(right, "question")
    left_norm = maps(left, "question_normalized")
    right_norm = maps(right, "question_normalized")
    exact_keys = set(left_exact) & set(right_exact)
    norm_keys = set(left_norm) & set(right_norm)
    normalized_only = norm_keys - {normalize_text(value) for value in exact_keys}
    return {
        "exact_question_keys": len(exact_keys),
        "exact_left_records": len({item for key in exact_keys for item in left_exact[key]}),
        "exact_right_records": len({item for key in exact_keys for item in right_exact[key]}),
        "normalized_question_keys": len(norm_keys),
        "normalized_left_records": len({item for key in norm_keys for item in left_norm[key]}),
        "normalized_right_records": len({item for key in norm_keys for item in right_norm[key]}),
        "normalized_only_keys": len(normalized_only),
        "exact_examples": [
            {
                "text_sha256": text_hash(value),
                "left_ids": left_exact[value][:10],
                "right_ids": right_exact[value][:10],
            }
            for value in sorted(exact_keys)[:10]
        ],
        "normalized_only_examples": [
            {
                "normalized_sha256": text_hash(value),
                "left_ids": left_norm[value][:10],
                "right_ids": right_norm[value][:10],
            }
            for value in sorted(normalized_only)[:10]
        ],
    }


def response_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    def response_maps(rows: list[dict[str, Any]], normalized: bool) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            for response in row.get("responses", []):
                if not response:
                    continue
                key = normalize_text(response) if normalized else response
                result[key].add(row["record_id"])
        return result

    left_exact = response_maps(left, False)
    right_exact = response_maps(right, False)
    left_norm = response_maps(left, True)
    right_norm = response_maps(right, True)
    exact = set(left_exact) & set(right_exact)
    normalized = set(left_norm) & set(right_norm)
    return {
        "exact_response_keys": len(exact),
        "exact_left_question_clusters": len({x for key in exact for x in left_exact[key]}),
        "exact_right_question_clusters": len({x for key in exact for x in right_exact[key]}),
        "normalized_response_keys": len(normalized),
        "normalized_left_question_clusters": len({x for key in normalized for x in left_norm[key]}),
        "normalized_right_question_clusters": len({x for key in normalized for x in right_norm[key]}),
    }


def pairwise_overlap(source_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    names = list(source_rows)
    comparisons = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = source_rows[left_name]
            right = source_rows[right_name]
            comparisons.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "question_overlap": overlap_summary(left, right),
                    "response_overlap": response_overlap(left, right),
                }
            )
    return {
        "normalization": "HTML entity decode/tag removal, Unicode NFKC, lowercase, compatible quote/apostrophe/dash mapping, trim and whitespace collapse",
        "representation": {
            "CounselBench-100": "questionTitle + blank line + questionText, with non-empty fallback",
            "CounselChat": "questionTitle + blank line + questionText, with non-empty fallback",
            "MentalChat16K": "input",
            "PsyQA-sample": "question + blank line + description, with non-empty fallback",
        },
        "comparisons": comparisons,
        "unavailable": ["Psych8k (manual gated access)"],
    }


def build_exclusion_manifest(
    benchmark: list[dict[str, Any]], source_rows: dict[str, list[dict[str, Any]]], benchmark_meta: dict[str, Any]
) -> list[dict[str, Any]]:
    cc = source_rows["CounselChat"]
    cc_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cc:
        cc_by_qid[row.get("question_id", "")].append(row)

    manifest = []
    for question in benchmark:
        exact_matches: dict[str, list[str]] = {}
        normalized_matches: dict[str, list[str]] = {}
        for source_name, rows in source_rows.items():
            if source_name == "CounselBench-100":
                continue
            exact_matches[source_name] = [row["record_id"] for row in rows if row["question"] == question["question"]]
            normalized_matches[source_name] = [
                row["record_id"]
                for row in rows
                if row["question_normalized"] == question["question_normalized"]
                and row["question"] != question["question"]
            ]

        direct_cc = cc_by_qid.get(str(question["numeric_question_id"]), [])
        cc_urls = sorted(
            {
                clean_display(row.get("metadata", {}).get("questionLink"))
                for row in direct_cc
                if clean_display(row.get("metadata", {}).get("questionLink"))
            }
        )
        manifest.append(
            {
                "manifest_version": "counselbench100-exclusion-v1",
                "canonical_question_cluster_id": f"counselbench:{question['question_id']}",
                "benchmark_dataset": "izi-ano/CounselBench-Eval",
                "benchmark_revision": SOURCE_REVISIONS["CounselBench-100"],
                "benchmark_arrow_sha256": benchmark_meta["sha256"],
                "question_id": question["question_id"],
                "numeric_question_id": question["numeric_question_id"],
                "question_title": question["question_title"],
                "question_text": question["question_body"],
                "combined_question": question["question"],
                "normalized_question": question["question_normalized"],
                "combined_question_sha256": text_hash(question["question"]),
                "normalized_question_sha256": text_hash(question["question_normalized"]),
                "topic": question["topic"],
                "survey_ids": question["survey_ids"],
                "responders": question["responders"],
                "evaluation_rows": question["response_count"],
                "unique_evaluation_response_hashes": [text_hash(value) for value in question["unique_responses"]],
                "direct_counselchat_match": {
                    "question_id": str(question["numeric_question_id"]),
                    "matched_cluster_ids": sorted({row["record_id"] for row in direct_cc}),
                    "response_rows": sum(row["response_rows"] for row in direct_cc),
                    "question_urls": cc_urls,
                },
                "exact_question_matches": exact_matches,
                "normalized_only_question_matches": normalized_matches,
                "exclusion_policy": "Exclude every candidate row in any cluster connected by confirmed ID, URL, exact, or normalized question match; near/semantic candidates require manual approval.",
            }
        )
    return manifest


def unique_by_normalized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["question_normalized"]
        if not key:
            continue
        if key not in output:
            output[key] = dict(row)
            output[key]["equivalent_record_ids"] = [row["record_id"]]
        else:
            output[key]["equivalent_record_ids"].append(row["record_id"])
    return list(output.values())


def confirmed_residual_sources(
    benchmark: list[dict[str, Any]], sources: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    """Remove confirmed benchmark links before producing a review-only queue."""
    benchmark_normalized = {row["question_normalized"] for row in benchmark}
    benchmark_numeric_ids = {str(row["numeric_question_id"]) for row in benchmark}
    residual: dict[str, list[dict[str, Any]]] = {}
    for source_name, rows in sources.items():
        if source_name == "CounselBench-100":
            residual[source_name] = rows
            continue
        kept = []
        for row in rows:
            if row["question_normalized"] in benchmark_normalized:
                continue
            if source_name == "CounselChat" and row.get("question_id") in benchmark_numeric_ids:
                continue
            kept.append(row)
        residual[source_name] = kept
    return residual


def top_indices(matrix: Any, k: int) -> list[list[int]]:
    if matrix.shape[1] <= k:
        return [list(np.argsort(-matrix.getrow(index).toarray().ravel())) for index in range(matrix.shape[0])]
    output = []
    for index in range(matrix.shape[0]):
        values = matrix.getrow(index).toarray().ravel()
        picked = np.argpartition(-values, k - 1)[:k]
        output.append(list(picked[np.argsort(-values[picked])]))
    return output


def lexical_candidates(
    benchmark: list[dict[str, Any]], sources: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    benchmark_texts = [row["question_normalized"] for row in benchmark]
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_name, source_rows in sources.items():
        if source_name in {"CounselBench-100", "PsyQA-sample"}:
            continue
        rows = unique_by_normalized(source_rows)
        if not rows:
            continue
        source_texts = [row["question_normalized"] for row in rows]
        all_texts = benchmark_texts + source_texts
        char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, norm="l2")
        word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, norm="l2")
        char_matrix = char_vectorizer.fit_transform(all_texts)
        word_matrix = word_vectorizer.fit_transform(all_texts)
        split = len(benchmark_texts)
        char_scores = char_matrix[split:] @ char_matrix[:split].T
        word_scores = word_matrix[split:] @ word_matrix[:split].T
        char_top = top_indices(char_scores, 3)
        word_top = top_indices(word_scores, 3)
        for source_index, row in enumerate(rows):
            for bench_index in sorted(set(char_top[source_index]) | set(word_top[source_index])):
                char_score = float(char_scores[source_index, bench_index])
                word_score = float(word_scores[source_index, bench_index])
                bench = benchmark[bench_index]
                if row["question_normalized"] == bench["question_normalized"]:
                    continue
                broad = char_score >= 0.70 or word_score >= 0.70
                strict = char_score >= 0.85 and word_score >= 0.75
                if not broad:
                    continue
                key = (source_name, row["record_id"], bench["question_id"])
                candidates[key] = {
                    "queue_version": "counselbench-manual-review-v1",
                    "review_status": "pending",
                    "candidate_source": source_name,
                    "candidate_record_id": row["record_id"],
                    "equivalent_record_ids": row["equivalent_record_ids"],
                    "benchmark_question_id": bench["question_id"],
                    "candidate_question_excerpt": row["question"][:240],
                    "benchmark_question_excerpt": bench["question"][:240],
                    "near_duplicate": {
                        "char_tfidf_3_5_cosine": char_score,
                        "word_tfidf_1_2_cosine": word_score,
                        "band": "strict" if strict else "broad",
                        "automatic_exclusion": False,
                    },
                    "semantic_duplicate": None,
                    "manual_review_fields": {
                        "same_client_situation": None,
                        "shared_specific_events_relationships_duration_symptoms": None,
                        "topic_only_similarity": None,
                        "decision": None,
                        "reviewer": None,
                        "notes": "",
                    },
                }
    return candidates


def encode_bge(texts: list[str], model_path: Path, batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = model(**encoded).last_hidden_state[:, 0]
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            vectors.append(output.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def add_semantic_candidates(
    queue: dict[tuple[str, str, str], dict[str, Any]],
    benchmark: list[dict[str, Any]],
    sources: dict[str, list[dict[str, Any]]],
    model_path: Path,
) -> dict[str, Any]:
    supported_sources = ["CounselChat", "MentalChat16K-interview", "MentalChat16K-synthetic"]
    source_rows = {name: unique_by_normalized(sources[name]) for name in supported_sources}
    benchmark_vectors = encode_bge([row["question"] for row in benchmark], model_path)
    total_candidates = 0
    for source_name, rows in source_rows.items():
        vectors = encode_bge([row["question"] for row in rows], model_path)
        scores = vectors @ benchmark_vectors.T
        k = min(3, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        for row_index, indices in enumerate(top):
            indices = indices[np.argsort(-scores[row_index, indices])]
            row = rows[row_index]
            for bench_index in indices:
                score = float(scores[row_index, bench_index])
                if score < 0.80:
                    continue
                bench = benchmark[int(bench_index)]
                if row["question_normalized"] == bench["question_normalized"]:
                    continue
                key = (source_name, row["record_id"], bench["question_id"])
                if key not in queue:
                    queue[key] = {
                        "queue_version": "counselbench-manual-review-v1",
                        "review_status": "pending",
                        "candidate_source": source_name,
                        "candidate_record_id": row["record_id"],
                        "equivalent_record_ids": row["equivalent_record_ids"],
                        "benchmark_question_id": bench["question_id"],
                        "candidate_question_excerpt": row["question"][:240],
                        "benchmark_question_excerpt": bench["question"][:240],
                        "near_duplicate": None,
                        "semantic_duplicate": None,
                        "manual_review_fields": {
                            "same_client_situation": None,
                            "shared_specific_events_relationships_duration_symptoms": None,
                            "topic_only_similarity": None,
                            "decision": None,
                            "reviewer": None,
                            "notes": "",
                        },
                    }
                queue[key]["semantic_duplicate"] = {
                    "model": "BAAI/bge-small-en-v1.5",
                    "model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                    "pooling": "CLS",
                    "max_length": 512,
                    "similarity": "cosine after L2 normalization",
                    "cosine": score,
                    "band": "strict" if score >= 0.90 else "broad",
                    "automatic_exclusion": False,
                }
                total_candidates += 1
    return {
        "status": "completed",
        "model_path": str(model_path),
        "candidate_pairs_at_or_above_0.80": total_candidates,
        "unsupported": ["PsyQA-sample (Chinese; English embedding threshold is invalid)", "Psych8k (raw unavailable)"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip_semantic", action="store_true")
    parser.add_argument(
        "--embedding_model",
        type=Path,
        default=Path(
            "/home/user/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/"
            "snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        ),
    )
    args = parser.parse_args()

    benchmark, benchmark_meta = load_counselbench()
    counsel_rows, counsel_meta = load_counselchat()
    mental_rows, mental_files = load_mentalchat()
    psyqa_rows, psyqa_meta = load_psyqa_sample()

    collapsed_benchmark = [
        {
            **row,
            "responses": row.pop("unique_responses"),
        }
        for row in [dict(item) for item in benchmark]
    ]
    collapsed_counsel = collapse_questions(counsel_rows)
    collapsed_mental_interview = collapse_questions(
        [row for row in mental_rows if row["source_component"] == "interview"]
    )
    collapsed_mental_synthetic = collapse_questions(
        [row for row in mental_rows if row["source_component"] == "synthetic"]
    )
    collapsed_psyqa = collapse_questions(psyqa_rows)
    source_rows = {
        "CounselBench-100": collapsed_benchmark,
        "CounselChat": collapsed_counsel,
        "MentalChat16K-interview": collapsed_mental_interview,
        "MentalChat16K-synthetic": collapsed_mental_synthetic,
        "PsyQA-sample": collapsed_psyqa,
    }

    exp295_freeze = build_exp295_freeze()
    inventory = build_candidate_inventory(counsel_meta, mental_files, psyqa_meta)
    overlaps = pairwise_overlap(source_rows)
    exclusion = build_exclusion_manifest(benchmark, source_rows, benchmark_meta)
    residual_sources = confirmed_residual_sources(benchmark, source_rows)
    queue = lexical_candidates(benchmark, residual_sources)

    if args.skip_semantic:
        semantic_status = {"status": "skipped by --skip_semantic"}
    elif not args.embedding_model.is_dir():
        semantic_status = {"status": "blocked: local embedding model directory is absent"}
    else:
        semantic_status = add_semantic_candidates(queue, benchmark, residual_sources, args.embedding_model)

    review_rows = sorted(
        queue.values(),
        key=lambda row: (
            row["candidate_source"],
            row["candidate_record_id"],
            row["benchmark_question_id"],
        ),
    )
    overlap_counts = {
        "manual_review_pairs": len(review_rows),
        "near_pairs": sum(row.get("near_duplicate") is not None for row in review_rows),
        "semantic_pairs": sum(row.get("semantic_duplicate") is not None for row in review_rows),
        "semantic_status": semantic_status,
    }
    overlaps["manual_review_queue"] = overlap_counts
    overlaps["benchmark_source"] = benchmark_meta

    write_json(args.output_dir / "exp295_integrity_freeze.json", exp295_freeze)
    write_json(args.output_dir / "candidate_intake_inventory.json", inventory)
    write_jsonl(args.output_dir / "counselbench100_exclusion_manifest.jsonl", exclusion)
    write_json(args.output_dir / "exact_normalized_overlap_report.json", overlaps)
    write_jsonl(args.output_dir / "manual_review_queue.jsonl", review_rows)

    output_files = []
    for path in sorted(args.output_dir.glob("*")):
        if path.is_file() and path.name != "phase1_manifest.json":
            output_files.append(file_record(path, include_rows=True))
    write_json(
        args.output_dir / "phase1_manifest.json",
        {
            "phase": 1,
            "scope": "integrity freeze, candidate intake, confirmed exact/normalized overlap, review-only near/semantic candidates",
            "forbidden_outputs": ["corruptions", "final split", "training", "final dataset inclusion decision"],
            "repository": str(REPO_ROOT),
            "git_commit": git_value("rev-parse", "HEAD"),
            "outputs": output_files,
            "blockers": [
                "Psych8k raw is unavailable until its manual Hugging Face gate is accepted.",
                "Full PsyQA requires the official signed user agreement and author approval.",
                "MentalChat16K does not expose the transcript/session IDs needed to group its 378 source transcripts.",
                "Every near/semantic queue item requires manual review before exclusion or retention.",
                "The BGE semantic thresholds are provisional and must be calibrated on manually labeled same-situation versus topic-only pairs.",
                "The DPO final-234 judged artifact is incomplete (37 of 354 input rows); checkpoint-200 is the complete evaluated DPO artifact.",
            ],
        },
    )

    print(json.dumps({"output_dir": str(args.output_dir), **overlap_counts}, indent=2))


if __name__ == "__main__":
    main()
