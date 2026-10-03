import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from corruption_contract_v2 import AXES, STAGE_DIRECTIVES  # noqa: E402
from run_qwen_single_axis_calibration import CALIBRATION_SEED, single_axis_specs  # noqa: E402


def test_single_axis_design_is_exact_and_deterministic():
    specs = single_axis_specs(CALIBRATION_SEED)
    assert specs == single_axis_specs(CALIBRATION_SEED)
    assert len(specs) == 30
    assert Counter(spec[0] for spec in specs) == {axis: 5 for axis in AXES}
    assert all(len(spec) == 1 for spec in specs)


def test_realistic_axis_operator_contract():
    assert "Do not inject unrelated topics" in STAGE_DIRECTIVES["overall_quality"]
    assert "Avoid overt hostility" in STAGE_DIRECTIVES["empathy"]
    assert "Remove or generalize multiple details" in STAGE_DIRECTIVES["specificity"]
    assert "unsupported or inaccurate claim" in STAGE_DIRECTIVES["factual_consistency"]
    assert "do not invent or assert a diagnosis" in STAGE_DIRECTIVES["medical_boundary"]
    assert "Prohibit slurs" in STAGE_DIRECTIVES["toxicity_or_harm"]
