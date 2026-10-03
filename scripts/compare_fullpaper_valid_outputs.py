#!/usr/bin/env python3
"""Join SFT, DPO, and Proposed outputs on the exact same canonical VALID IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)).resolve() for name in ("sft", "dpo", "proposed")}
    rows = {name: read(path) for name, path in paths.items()}

    def index(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {str(row.get("canonical_id") or row.get("id")): row for row in data}
        if len(result) != len(data) or "None" in result:
            raise RuntimeError("Missing or duplicate canonical VALID ID")
        return result

    indexed = {name: index(data) for name, data in rows.items()}
    membership = {name: set(data) for name, data in indexed.items()}
    if len({frozenset(ids) for ids in membership.values()}) != 1:
        raise RuntimeError({name: len(ids) for name, ids in membership.items()})

    combined = []
    for canonical_id in sorted(membership["sft"]):
        sft, dpo, proposed = (indexed[name][canonical_id] for name in ("sft", "dpo", "proposed"))
        combined.append({
            "canonical_id": canonical_id,
            "split": sft.get("split"),
            "question": sft["question"],
            "clean_reference": sft.get("safe_response"),
            "corrupted_input": sft.get("unsafe_response"),
            "sft_response": sft.get("sft_response"),
            "dpo_response": dpo.get("sft_response"),
            "proposed_response": proposed.get("final_response"),
            "proposed_used_denoiser": proposed.get("used_denoiser"),
            "proposed_accepted_denoiser": proposed.get("accepted_denoiser"),
            "proposed_reject_reason": proposed.get("reject_reason"),
            "proposed_internal_risk": {
                "sft": proposed.get("sft_risk_score"),
                "denoiser": proposed.get("denoiser_risk_score"),
                "final": proposed.get("final_risk_score"),
            },
        })
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "complete_structural_comparison_no_external_quality_judge",
        "rows": len(combined),
        "same_canonical_valid_membership": True,
        "warning": "Internal losses and risk scores are not comparable counseling-quality rankings across objectives.",
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "output": {"path": str(output), "sha256": sha256(output)},
        "proposed_denoiser_called": sum(bool(row["proposed_used_denoiser"]) for row in combined),
        "proposed_denoiser_accepted": sum(bool(row["proposed_accepted_denoiser"]) for row in combined),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
