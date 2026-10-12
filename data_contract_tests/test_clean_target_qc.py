import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fullpaper_acl_pipeline import read_jsonl  # noqa: E402
from run_clean_target_qc import stratified_sample  # noqa: E402


def test_clean_target_sample_is_deterministic_and_stratified():
    rows = list(read_jsonl(ROOT / "data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"))
    first = stratified_sample(rows, 120, 20260908)
    second = stratified_sample(rows, 120, 20260908)
    assert [row["canonical_id"] for row in first] == [row["canonical_id"] for row in second]
    assert len({row["canonical_id"] for row in first}) == 120
    assert set(Counter((row["source"], row["split"]) for row in first)) == set(
        Counter((row["source"], row["split"]) for row in rows)
    )
