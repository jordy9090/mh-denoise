import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fullpaper_acl_pipeline import read_jsonl, sha256_file
from repair_dev120_audit import AUDIT_SHA256, ORIGINAL, OUTPUT, PRIMARY, SECONDARY
from source_integrity_contract import exact_offset, surface_flags


def test_original_dev120_artifacts_remain_immutable():
    assert {name: sha256_file(ORIGINAL / name) for name in AUDIT_SHA256} == AUDIT_SHA256


def test_integrity_contract_covers_observed_omissions():
    assert surface_flags("", "This revised dialogue offers a better answer.", "clean")
    assert surface_flags("", "I'm here as a fellow human.", "clean")
    # Unsupported-history detection is semantic and now comes from the typed
    # clean-target QC gate, not a phrase copied from a historical failure.
    assert not surface_flags("", "Continuing to attend grief counseling sessions will help.", "clean")
    question = "I feel helpless and frustrated, as I want to support him but don't know how to effectively encourage him to make these changes."
    assert any(flag["kind"] == "speaker_switch_question_copy" for flag in surface_flags(question, question, "candidate"))
    assert not surface_flags("I was diagnosed with cancer.", "Your cancer journey may be hard.", "clean")


def test_exact_offset_rejects_nonverbatim_evidence():
    source = "A precise source sentence."
    assert exact_offset(source, "precise source") == {"start": 2, "end": 16, "text": "precise source"}
    assert exact_offset(source, "precise shortened sentence") is None


def test_corrected_exports_contract():
    if not (OUTPUT / "repair_manifest.json").exists():
        return
    manifest = json.loads((OUTPUT / "repair_manifest.json").read_text())
    sft = list(read_jsonl(OUTPUT / "train_sft.jsonl"))
    dpo = list(read_jsonl(OUTPUT / "train_dpo.jsonl"))
    held = list(read_jsonl(OUTPUT / "held_out.jsonl"))
    secondary = list(read_jsonl(OUTPUT / "secondary_flags.jsonl"))
    held_ids = set(PRIMARY) | set(SECONDARY)
    assert manifest["original_input_accounting"] == {"accepted": 70, "rejected": 43, "qc_conflicts": 7, "total": 120}
    assert len(sft) == len(dpo) == 57
    assert len(held) == 13 and len(secondary) == 3
    assert {row["canonical_id"] for row in held} == held_ids
    assert {row["canonical_id"] for row in secondary} == set(SECONDARY)
    assert held_ids.isdisjoint({row["metadata"]["canonical_id"] for row in sft + dpo})
    accepted = {row["canonical_id"]: row for row in read_jsonl(ORIGINAL / "accepted.jsonl")}
    for row in sft + dpo:
        cid = row["metadata"]["canonical_id"]
        assert row["metadata"]["raw_paired_qc_preserved"] == accepted[cid]["final_grade"]
    for span in read_jsonl(OUTPUT / "source_verified_spans.jsonl"):
        field = "clean_response" if span["side"] == "clean" else "corrupted_response"
        source = accepted[span["canonical_id"]][field]
        assert source[span["start"]:span["end"]] == span["text"]
