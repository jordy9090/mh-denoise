import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.run_fullpaper_corruption_production import (
    LOCAL_JUDGE_VERSION,
    atomic_write_jsonl,
    export,
    initialize_run_contract,
    load_selection,
)
from scripts.build_fullpaper_training_data import verify_production_source
from scripts.train_dpo_minimal import validate_fullpaper_pair_rows


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canonical_row(canonical_id, split, question_group):
    return {
        "canonical_id": canonical_id,
        "split": split,
        "question": f"question {canonical_id}",
        "clean_response": f"response {canonical_id}",
        "question_normalized_sha256": question_group,
        "duplicate_cluster_id": f"duplicate-{canonical_id}",
        "source_group_id": f"source-group-{canonical_id}",
        "source": "MentalChat16K",
        "source_component": "synthetic",
    }


def clean_qc(canonical_id, split):
    return {
        "canonical_id": canonical_id,
        "split": split,
        "qc_ok": True,
        "eligible": True,
        "baseline_degraded_axes": [],
    }


def write_selection(path, rows):
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_selection_supports_canonical_train_and_valid_and_one_answer_per_question(tmp_path):
    canonical = {
        "train-id": canonical_row("train-id", "train", "question-a"),
        "valid-id": canonical_row("valid-id", "valid", "question-b"),
    }
    qc = {cid: clean_qc(cid, row["split"]) for cid, row in canonical.items()}
    selection = tmp_path / "selection.json"
    write_selection(selection, [
        {"canonical_id": "train-id", "split": "train", "intended_axes": ["empathy"]},
        {"canonical_id": "valid-id", "split": "valid", "intended_axes": ["specificity"]},
    ])
    selected = load_selection(selection, 2, canonical, qc, set(), {"train", "valid"})
    assert [row["split"] for row in selected] == ["train", "valid"]

    duplicate = copy.deepcopy(canonical)
    duplicate["valid-id"]["question_normalized_sha256"] = "question-a"
    with pytest.raises(ValueError, match="More than one response"):
        load_selection(selection, 2, duplicate, qc, set(), {"train", "valid"})


def test_selection_rejects_holds_and_nonclean_rows(tmp_path):
    canonical = {"held": canonical_row("held", "train", "question-held")}
    qc = {"held": clean_qc("held", "train")}
    selection = tmp_path / "selection.json"
    write_selection(selection, [{"canonical_id": "held", "intended_axes": ["empathy"]}])
    with pytest.raises(ValueError, match="known held"):
        load_selection(selection, 1, canonical, qc, {"held"}, {"train"})
    qc["held"]["baseline_degraded_axes"] = ["overall_quality"]
    with pytest.raises(ValueError, match="clean-QC"):
        load_selection(selection, 1, canonical, qc, set(), {"train"})


def test_run_contract_is_atomic_and_refuses_changed_restart(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    contract = {"selection_sha256": "a", "seed": 7, "axes": ["empathy"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fingerprints = list(pool.map(lambda _: initialize_run_contract(output, contract), range(8)))
    assert len(set(fingerprints)) == 1
    changed = {**contract, "seed": 8}
    with pytest.raises(RuntimeError, match="differs"):
        initialize_run_contract(output, changed)


def test_atomic_checkpoint_never_exposes_mixed_jsonl(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    versions = [[{"writer": writer, "row": row} for row in range(20)] for writer in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda rows: atomic_write_jsonl(checkpoint, rows), versions))
    observed = read_jsonl(checkpoint)
    assert len(observed) == 20
    assert len({row["writer"] for row in observed}) == 1


def test_existing_results_export_and_convert_as_distinct_train_valid(tmp_path):
    originals = read_jsonl(ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120/accepted.jsonl")[:2]
    results = copy.deepcopy(originals)
    canonical = []
    for index, (row, split) in enumerate(zip(results, ("train", "valid"), strict=True)):
        row.update(
            {
                "split": split,
                "question_normalized_sha256": f"question-{index}",
                "duplicate_cluster_id": f"duplicate-{index}",
                "source_group_id": f"source-{index}",
            }
        )
        side = "clean" if index == 0 else "candidate"
        source_text = row["clean_response" if side == "clean" else "corrupted_response"]
        text = source_text[: min(40, len(source_text))]
        row["final_grade"]["local_supervision"] = [{
            "axis": "empathy" if index == 0 else "factual_consistency",
            "side": side,
            "scope": "local_support" if index == 0 else "local_defect",
            "label": 0 if index == 0 else 1,
            "start": 0, "end": len(text), "text": text,
            "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "reason": "typed fixture evidence",
            "annotation_source": "typed-fixture",
            "model_input_contract": "complete_question_plus_exact_span",
        }]
        row["final_grade"]["response_level_evidence"] = []
        row["final_grade"]["clean_target_eligibility"] = {"disposition": "pass"}
        canonical.append(
            {
                "canonical_id": row["canonical_id"],
                "split": split,
                "question_normalized_sha256": f"question-{index}",
                "duplicate_cluster_id": f"duplicate-{index}",
                "source_group_id": f"source-{index}",
                "source_question_id": str(index),
            }
        )
    source = tmp_path / "source"
    source.mkdir()
    export(source, results, {"run_fingerprint": "local-contract-test", "version": LOCAL_JUDGE_VERSION})
    assert len(read_jsonl(source / "train_sft.jsonl")) == 1
    assert len(read_jsonl(source / "valid_sft.jsonl")) == 1
    canonical_path = tmp_path / "canonical.jsonl"
    canonical_path.write_text("".join(json.dumps(row) + "\n" for row in canonical), encoding="utf-8")
    converted = tmp_path / "converted"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_fullpaper_training_data.py"),
            "--source-kind", "production",
            "--source-dir", str(source),
            "--output-dir", str(converted),
            "--canonical-file", str(canonical_path),
            "--tokenizer-dir", str(ROOT / "outputs/models/fullpaper_dev57_gemma_sft_smoke2/final"),
            "--scorer-tokenizer-dir", str(ROOT / "outputs/models/fullpaper_dev57_span_scorer_smoke2/final"),
            "--development-eval-size", "0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    train = read_jsonl(converted / "dpo_train.jsonl")
    valid = read_jsonl(converted / "dpo_valid.jsonl")
    assert validate_fullpaper_pair_rows(train, "train")["rows"] == 1
    assert validate_fullpaper_pair_rows(valid, "valid")["rows"] == 1
    manifest = json.loads((converted / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_valid_overlap_checks"]["all_zero"] is True
    assert manifest["source_production_manifest"]["status"] == "complete"
    assert manifest["scorer_tokenizer_audit"]["max_length"] == 512


def test_failed_export_leaves_no_consumable_training_artifacts(tmp_path):
    row = copy.deepcopy(read_jsonl(ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120/accepted.jsonl")[0])
    row.update({
        "split": "train",
        "question_normalized_sha256": "question-failed",
        "duplicate_cluster_id": "duplicate-failed",
        "source_group_id": "source-failed",
    })
    clean_text = row["clean_response"]
    evidence = clean_text[: min(40, len(clean_text))]
    row["final_grade"]["local_supervision"] = [{
        "axis": "factual_consistency", "side": "clean", "scope": "local_defect",
        "label": 1, "start": 0, "end": len(evidence), "text": evidence,
        "source_sha256": hashlib.sha256(clean_text.encode()).hexdigest(),
        "reason": "fixture explicit clean defect", "annotation_source": "fixture",
    }]
    row["final_grade"]["response_level_evidence"] = []
    row["final_grade"]["clean_target_eligibility"] = {"disposition": "pass"}
    source = tmp_path / "failed-source"
    source.mkdir()
    with pytest.raises(AssertionError):
        export(source, [row], {"run_fingerprint": "failed", "version": LOCAL_JUDGE_VERSION})
    assert not (source / "train_sft.jsonl").exists()
    assert not (source / "production_manifest.json").exists()
    with pytest.raises(RuntimeError, match="no successful"):
        verify_production_source(source)
