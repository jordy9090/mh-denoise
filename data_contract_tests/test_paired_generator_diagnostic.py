import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from corruption_contract_v2 import AXES  # noqa: E402
from run_paired_generator_diagnostic import (  # noqa: E402
    GENERATORS,
    DEFAULT_REQUIRED_FREE_VRAM_MIB,
    PILOT_SEED,
    candidate_rows,
    adaptive_max_new_tokens,
    paired_specs,
    gpu_status,
    generation_stop_metadata,
    deterministic_generation_seed,
    strip_reasoning,
)


def test_paired_specs_have_required_shape_and_attention_axes():
    specs = paired_specs(PILOT_SEED)
    assert specs == paired_specs(PILOT_SEED)
    assert Counter(map(len, specs)) == {1: 12, 2: 8, 3: 4}
    marginal = Counter(axis for spec in specs for axis in spec)
    assert max(marginal.values()) - min(marginal.values()) <= 1
    assert marginal["overall_quality"] >= 7
    assert marginal["factual_consistency"] >= 7
    assert set(marginal) == set(AXES)


def test_candidates_are_train_only():
    rows = candidate_rows(ROOT / "data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl", PILOT_SEED)
    assert rows and all(row["split"] == "train" for row in rows)


def test_pinned_models_and_nonthinking_settings():
    assert set(GENERATORS) == {"gemma", "qwen"}
    assert GENERATORS["gemma"]["revision"] == "ee0ef6023621cff504d758262d4e04895a5af4a2"
    assert GENERATORS["qwen"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert all(spec["enable_thinking"] is False for spec in GENERATORS.values())


def test_reasoning_is_removed_and_residual_is_detected():
    clean, changed = strip_reasoning("<think>hidden</think>\n\nVisible response")
    assert clean == "Visible response" and changed
    clean, changed = strip_reasoning("Visible <think>leak</think>")
    assert changed and "<think>" in clean


def test_gpu_admission_uses_free_vram_not_process_presence(monkeypatch):
    outputs = iter([
        "0, GPU-test, NVIDIA H100 PCIe, 81559, 70000, 11559\n",
        "1234, python, 2048, GPU-test\n",
    ])

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(
        "run_paired_generator_diagnostic.subprocess.run",
        lambda *args, **kwargs: Result(next(outputs)),
    )
    status = gpu_status(0, DEFAULT_REQUIRED_FREE_VRAM_MIB)
    assert status["existing_compute_processes"]
    assert status["free_vram_mib"] >= DEFAULT_REQUIRED_FREE_VRAM_MIB
    assert status["sufficient_free_vram"] is True


def test_adaptive_generation_allowance_policy():
    assert adaptive_max_new_tokens(10) == 768
    assert adaptive_max_new_tokens(512) == 768
    assert adaptive_max_new_tokens(513) == 769
    assert adaptive_max_new_tokens(1280) == 1536
    assert adaptive_max_new_tokens(5000) == 1536


def test_truncation_requires_budget_exhaustion_without_eos():
    assert generation_stop_metadata([4, 5, 1], {1}, 3) == (True, None)
    assert generation_stop_metadata([4, 5], {1}, 3) == (False, None)
    assert generation_stop_metadata([4, 5, 6], {1}, 3) == (
        False,
        "max_new_tokens_reached_without_eos",
    )


def test_generation_seed_is_per_example_stage_and_attempt():
    seed = deterministic_generation_seed(20260908, "qa_123", 1, 1)
    assert seed == deterministic_generation_seed(20260908, "qa_123", 1, 1)
    assert len({
        seed,
        deterministic_generation_seed(20260908, "qa_124", 1, 1),
        deterministic_generation_seed(20260908, "qa_123", 2, 1),
        deterministic_generation_seed(20260908, "qa_123", 1, 2),
    }) == 4
