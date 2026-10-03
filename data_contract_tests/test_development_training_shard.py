import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120"


def rows(name):
    with (OUT / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_frozen_selection_contract():
    selection = json.loads((OUT / "selection.json").read_text())["rows"]
    assert len(selection) == 120
    assert len({row["canonical_id"] for row in selection}) == 120
    assert {n: sum(row["axis_count"] == n for row in selection) for n in (1, 2, 3)} == {1: 60, 2: 42, 3: 18}
    assert {axis: sum(axis in row["intended_axes"] for row in selection) for axis in ("overall_quality", "empathy", "specificity", "factual_consistency", "medical_boundary", "toxicity_or_harm")} == {axis: 33 for axis in ("overall_quality", "empathy", "specificity", "factual_consistency", "medical_boundary", "toxicity_or_harm")}


def test_terminal_partition_and_acceptance_contract():
    accepted, rejected, conflicts = rows("accepted.jsonl"), rows("rejected.jsonl"), rows("qc_conflicts.jsonl")
    assert (len(accepted), len(rejected), len(conflicts)) == (70, 43, 7)
    assert len(accepted) + len(rejected) + len(conflicts) == 120
    assert all(row["split"] == "train" and row["baseline_degraded_axes"] == [] for row in accepted)
    assert all(set(row["intended_axes"]) <= set(row["realized_axes"]) for row in accepted)
    assert len({row["duplicate_cluster_id"] for row in accepted}) == len(accepted)


def test_sft_dpo_inputs_and_targets():
    accepted, sft, dpo = rows("accepted.jsonl"), rows("train_sft.jsonl"), rows("train_dpo.jsonl")
    assert len(sft) == len(dpo) == len(accepted) == 70
    for source, sft_row, dpo_row in zip(accepted, sft, dpo, strict=True):
        assert set(sft_row["input"]) == {"question", "corrupted_response"}
        assert sft_row["input"] == dpo_row["input"]
        assert sft_row["target"] == dpo_row["chosen"] == source["clean_response"]
        assert dpo_row["rejected"] == source["corrupted_response"]
        assert sft_row["input"]["corrupted_response"] == dpo_row["rejected"]
        assert "intended_axes" not in sft_row["input"] and "realized_axes" not in sft_row["input"]
        assert "paired_qc" in sft_row["metadata"]
