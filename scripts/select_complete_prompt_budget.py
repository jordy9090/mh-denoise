#!/usr/bin/env python3
"""Choose the smallest audited source limit that preserves every prompt section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", action="append", required=True)
    parser.add_argument("--model-context-limit", type=int, required=True)
    parser.add_argument("--generation-budget", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.audit_manifest]
    common_limits = set(manifests[0]["limits"])
    for manifest in manifests[1:]:
        common_limits &= set(manifest["limits"])
    complete = []
    for value in sorted(map(int, common_limits)):
        if all(
            manifest["limits"][str(value)]["truncated_rows"] == 0
            and manifest["limits"][str(value)]["rows_with_any_incomplete_section"] == 0
            for manifest in manifests
        ):
            complete.append(value)
    if not complete:
        raise RuntimeError("No audited source limit preserves every prompt section")
    selected = complete[0]
    if selected + args.generation_budget > args.model_context_limit:
        raise RuntimeError(
            f"Complete prompt plus generation budget exceeds model context: "
            f"{selected}+{args.generation_budget}>{args.model_context_limit}"
        )
    payload = {
        "status": "selected_from_complete_cpu_prompt_audits",
        "selected_max_source_len": selected,
        "generation_budget": args.generation_budget,
        "model_context_limit": args.model_context_limit,
        "audits": [str(Path(path).resolve()) for path in args.audit_manifest],
        "candidate_complete_limits": complete,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
