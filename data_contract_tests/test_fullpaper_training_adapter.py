import hashlib
import json
from pathlib import Path

from scripts.train_dpo_minimal import validate_fullpaper_pair_rows
from scripts.run_fullpaper_corruption_production import source_verified_spans


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2"
ADAPTED = ROOT / "data/fullpaper_acl_pipeline/fullpaper_training_v1"
AXES = {
    "overall_quality", "empathy", "specificity", "factual_consistency",
    "medical_boundary", "toxicity_or_harm",
}


def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def test_flat_sft_and_dpo_preserve_source_text_exactly():
    source = {row["metadata"]["canonical_id"]: row for row in rows(SOURCE / "train_sft.jsonl")}
    flat = {row["canonical_id"]: row for row in rows(ADAPTED / "sft_all57.jsonl")}
    dpo = {row["id"]: row for row in rows(ADAPTED / "dpo_all57.jsonl")}
    assert set(source) == set(flat) == set(dpo) and len(source) == 57
    for canonical_id, original in source.items():
        assert flat[canonical_id]["question"] == original["input"]["question"]
        assert flat[canonical_id]["unsafe_response"] == original["input"]["corrupted_response"]
        assert flat[canonical_id]["safe_response"] == original["target"]
        assert dpo[canonical_id]["chosen"] == original["target"]
        assert dpo[canonical_id]["rejected"] == original["input"]["corrupted_response"]


def test_smoke_partition_is_disjoint_train_origin_and_dynamic_dpo_contract_passes():
    train = rows(ADAPTED / "dpo_train49.jsonl")
    development = rows(ADAPTED / "dpo_dev8_train_origin.jsonl")
    assert len(train) == 49 and len(development) == 8
    assert {row["id"] for row in train}.isdisjoint({row["id"] for row in development})
    assert validate_fullpaper_pair_rows(train, "train")["rows"] == 49
    assert validate_fullpaper_pair_rows(development, "development_eval_train_origin")["rows"] == 8


def test_scorer_uses_exact_side_offsets_and_keeps_unknown_masked():
    sft = {row["canonical_id"]: row for row in rows(ADAPTED / "sft_all57.jsonl")}
    scorer = rows(ADAPTED / "scorer_all_verified_spans.jsonl")
    assert len(scorer) == 306
    assert any(value is None for row in scorer for value in row["labels"].values())
    for row in scorer:
        source_text = (
            sft[row["canonical_id"]]["unsafe_response"]
            if row["span_side"] == "candidate"
            else sft[row["canonical_id"]]["safe_response"]
        )
        assert source_text[row["span_start"] : row["span_end"]] == row["span"]
        assert hashlib.sha256(source_text.encode()).hexdigest() == row["span_source_sha256"]
        assert set(row["labels"]) == AXES
        assert all(row["label_mask"][axis] == (row["labels"][axis] is not None) for axis in AXES)


def test_legacy_untyped_production_evidence_is_not_promoted_to_local_labels():
    accepted = rows(ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120/accepted.jsonl")[0]
    verified, excluded = source_verified_spans(accepted)
    assert verified == []
    assert any(item["reason"] == "legacy_untyped_response_evidence_not_local_supervision" for item in excluded)
    assert all(item["reason"] for item in excluded)


def test_versioned_dynamic_fullpaper_dpo_contract_accepts_nonfixed_counts_and_valid_role():
    row = rows(ADAPTED / "dpo_train49.jsonl")[0]
    dynamic = json.loads(json.dumps(row))
    dynamic["audit_metadata"].update(
        {
            "dataset": "fullpaper_production_v1",
            "provenance": "versioned full-paper accepted TRAIN corruption data",
            "contract_version": "fullpaper-dpo-v1",
            "question_normalized_sha256": "normalized-question-1",
        }
    )
    dynamic["question_group_id"] = "normalized-question-1"
    assert validate_fullpaper_pair_rows([dynamic], "train")["rows"] == 1
    dynamic_valid = json.loads(json.dumps(dynamic))
    dynamic_valid["audit_metadata"].update({"original_split": "valid", "development_role": "valid"})
    assert validate_fullpaper_pair_rows([dynamic_valid], "valid")["not_final_valid_or_test"] is False
