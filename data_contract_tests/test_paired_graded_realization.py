import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corruption_contract_v2 import paired_graded_realized_axes


def test_paired_graded_realization_directions_and_medical_boundary():
    clean = {"overall_quality": 4, "empathy": 4, "specificity": 3, "factual_consistency": 4, "toxicity_or_harm": 1, "medical_boundary": False}
    candidate = {"overall_quality": 3, "empathy": 4, "specificity": 2, "factual_consistency": 3, "toxicity_or_harm": 2, "medical_boundary": True}
    assert paired_graded_realized_axes(clean, candidate) == (
        "overall_quality", "specificity", "factual_consistency", "medical_boundary", "toxicity_or_harm"
    )


def test_paired_graded_realization_requires_meaningful_change():
    scores = {"overall_quality": 4, "empathy": 4, "specificity": 3, "factual_consistency": 4, "toxicity_or_harm": 1, "medical_boundary": False}
    assert paired_graded_realized_axes(scores, dict(scores)) == ()
