#!/usr/bin/env python3
"""Build a history-free, allowlisted ACL source export without running models."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


SCRIPT_NAMES = (
    "acl_local_judge_utils.py",
    "audit_frozen_test_inference_lengths.py",
    "audit_selective_refinement_prompts.py",
    "build_development_corruption_shard.py",
    "build_dpo_minimal_pairs.py",
    "build_frozen_revision2_test_partial_export.py",
    "build_production_reuse_export.py",
    "build_provisional_production_corrected_export.py",
    "build_sft_outputs_for_risk_tuning.py",
    "corruption_contract.py",
    "corruption_contract_v2.py",
    "evaluate_natural.py",
    "fullpaper_acl_pipeline.py",
    "fullpaper_backbone_utils.py",
    "fullpaper_risk_contract.py",
    "fullpaper_scorer_export_contract.py",
    "h100_test_surface_relaxed_utils.py",
    "local_qwen_production_qc_v4.py",
    "prepare_fullpaper_phase1.py",
    "prepare_rubric_aligned_gpt41_rejudge.py",
    "rejudge_production_reuse_subset.py",
    "render_fullpaper_dpo_for_backbone.py",
    "run_acl_component_llama_pointwise.py",
    "run_acl_component_pointwise.py",
    "run_arr_revision2_first_backbone.sh",
    "run_clean_target_qc.py",
    "run_corrected_test60_pointwise_publication.py",
    "run_fullpaper_corruption_production.py",
    "run_gemma_corruption_pilot.py",
    "run_gemma_corruption_pilot_v2.py",
    "run_gemma_selective_risk_refinement.py",
    "run_gpt41_checkpoint_queue.py",
    "run_h100_test_surface_relaxed_live_tmux.sh",
    "run_independent.py",
    "run_local_qwen_judge_comparison.py",
    "run_natural.py",
    "run_ordinary_gpt_selector.py",
    "run_paired_generator_diagnostic.py",
    "run_refinement_llm_judge.py",
    "run_test40_extraction_fix.py",
    "run_test60_rubric_aligned_selection.py",
    "select_arr_valid_risk_thresholds.py",
    "select_complete_prompt_budget.py",
    "selective_risk_refinement_utils.py",
    "shared_api_budget.py",
    "source_integrity_contract.py",
    "train_dpo_minimal.py",
    "train_fullpaper_risk_model.py",
    "train_gemma_risk_tune_from_sft.py",
    "train_professor_peft_refiner.py",
    "train_professor_peft_refiner_textonly.py",
)
TEST_PATHS = (
    "data_contract_tests/test_fullpaper_risk_contract.py",
    "data_contract_tests/test_paired_graded_realization.py",
    "data_contract_tests/test_shared_api_budget.py",
    "tests/test_acceptance_policy.py",
    "tests/test_dpo_minimal.py",
    "tests/test_production_reuse_adjudication.py",
    "tests/test_render_fullpaper_dpo_for_backbone.py",
    "tests/test_select_arr_valid_risk_thresholds.py",
    "tests/test_selective_risk_refinement_modes.py",
)
ALLOWLIST = (
    "README.md",
    "docs/ACL_REPRODUCIBILITY.md",
    *(f"scripts/{name}" for name in SCRIPT_NAMES),
    *TEST_PATHS,
)

# Filesystem locations and the reviewed Git identity change only in export copies.
PATH_RULES = (
    (r"/home/[^/\s\"']+/\.cache/", "/workspace/cache/"),
    (
        r"/home/[^/\s\"']+/anaconda3/envs/[^/\s\"']+/bin/python",
        "/workspace/env/bin/python",
    ),
    (
        r"/home/[^/\s\"']+/[^/\s\"']+/mh-denoise"
        r"(?:-(?:surface-relaxed-v1|feedback-v1-valid)[^/\s\"']*)?"
        r"(?=/|[\"']|$)",
        "/workspace/mh-denoise",
    ),
    (r"/home/[^/\s\"']+/[^/\s\"']+/", "/workspace/artifacts/"),
    (r"/mnt/[^/\s\"']+/[^/\s\"']*transformers-kernels", "/workspace/kernels"),
    (r"/mnt/[^/\s\"']+/[^/\s\"']*-hf/", "/workspace/models/"),
    (r"/mnt/[^/\s\"']+/[^/\s\"']+/", "/workspace/artifacts/"),
)
FORBIDDEN_CONTENT = (
    ("personal filesystem path", re.compile(r"/(?:home|Users|mnt|root)/")),
    ("email address", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("credential token", re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b")),
    (
        "literal API credential",
        re.compile(r"(?i)(?:api_key|access_token|secret_key)\s*=\s*['\"][^'\"\s]{12,}['\"]"),
    ),
)
PRIVATE_COMMIT_NAMES = frozenset({"EXPECTED_WORKTREE_COMMIT"})
PRIVATE_COMMIT_PLACEHOLDER = "PRIVATE_PROVENANCE_COMMIT"


def portable_paths(text: str) -> tuple[str, int]:
    substitutions = 0
    for pattern, replacement in PATH_RULES:
        text, count = re.subn(pattern, replacement, text)
        substitutions += count
    return text, substitutions


def git_commit_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or not (
                target.id == "COMMIT" or target.id.endswith("_COMMIT")
            ):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                if re.fullmatch(r"[0-9a-fA-F]{40}", node.value.value):
                    yield target.id, node.value


def neutralize_git_metadata(text: str, filename: str) -> tuple[str, int]:
    tree = ast.parse(text, filename=filename)
    raw = text.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = set()
    for name, value in git_commit_literals(tree):
        if name not in PRIVATE_COMMIT_NAMES:
            raise ValueError(f"{filename}: unreviewed Git commit assignment {name}")
        start = offsets[value.lineno - 1] + value.col_offset
        end = offsets[value.end_lineno - 1] + value.end_col_offset
        replacements.add((start, end))
    for start, end in sorted(replacements, reverse=True):
        raw = raw[:start] + json.dumps(PRIVATE_COMMIT_PLACEHOLDER).encode("ascii") + raw[end:]
    return raw.decode("utf-8"), len(replacements)


def normalized_ast(text: str, filename: str, *, normalize_paths: bool) -> str:
    tree = ast.parse(text, filename=filename)
    if normalize_paths:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value, _ = portable_paths(node.value)
        for name, value in git_commit_literals(tree):
            if name not in PRIVATE_COMMIT_NAMES:
                raise ValueError(f"{filename}: unreviewed Git commit assignment {name}")
            value.value = PRIVATE_COMMIT_PLACEHOLDER
    return ast.dump(tree, include_attributes=False)


def check_prompt_constants(original: str, exported: str, filename: str) -> None:
    def constants(text: str) -> dict[str, str]:
        values = {}
        for node in ast.walk(ast.parse(text, filename=filename)):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and any(
                    marker in target.id.upper() for marker in ("PROMPT", "TEMPLATE", "DIRECTIVE")
                ):
                    values[target.id] = ast.dump(node.value, include_attributes=False)
        return values

    if constants(original) != constants(exported):
        raise ValueError(f"{filename}: prompt/template/directive definition changed")


def check_content(relative: str, text: str, denied_tokens: list[str]) -> None:
    for label, pattern in FORBIDDEN_CONTENT:
        if pattern.search(text):
            raise ValueError(f"{relative}: forbidden {label}")
    for token in denied_tokens:
        if token.casefold() in text.casefold() or token.casefold() in relative.casefold():
            raise ValueError(f"{relative}: forbidden identity token")


def local_imports(tree: ast.AST) -> set[str]:
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "scripts":
                imports.update(f"scripts.{alias.name}" for alias in node.names)
            elif node.module:
                imports.add(node.module)
    return imports


def check_dependencies(relative: str, text: str, source: Path) -> None:
    if not relative.endswith((".py", ".sh")):
        return
    paths = {
        f"scripts/{name}"
        for name in re.findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z_0-9]*\.(?:py|sh))\b", text)
    }
    if relative.endswith(".py"):
        for module in local_imports(ast.parse(text, filename=relative)):
            components = module.split(".")
            name = components[1] if components[0] == "scripts" and len(components) > 1 else components[0]
            paths.add(f"scripts/{name}.py")
    for path in sorted(paths):
        if (source / path).is_file() and path not in ALLOWLIST:
            raise ValueError(f"{relative}: unexported local dependency {path}")


def check_markdown_links(relative: str, text: str) -> None:
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)", text):
        if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        target = target.split("#", 1)[0]
        parts: list[str] = list(Path(relative).parent.parts)
        for part in target.split("/"):
            if part == "..":
                if not parts:
                    raise ValueError(f"{relative}: link escapes export")
                parts.pop()
            elif part not in ("", "."):
                parts.append(part)
        if "/".join(parts) not in ALLOWLIST:
            raise ValueError(f"{relative}: link absent from export: {target}")


def build_export(source: Path, destination: Path, denied_tokens: list[str]) -> dict:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination must not exist; existing files are never overwritten")
    resolved_destination = destination.resolve()
    if resolved_destination == source or source in resolved_destination.parents:
        raise ValueError("Destination must be outside the development repository")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    rendered: dict[str, str] = {}
    records = []
    for relative in ALLOWLIST:
        path = source / relative
        if path.is_symlink() or not path.is_file() or source not in path.resolve().parents:
            raise ValueError(f"{relative}: expected a regular source file inside the repository")
        if any(parent.is_symlink() for parent in path.parents if parent != source and source in parent.parents):
            raise ValueError(f"{relative}: symlinked source directory is not permitted")
        original_bytes = path.read_bytes()
        original = original_bytes.decode("utf-8")
        text, count = portable_paths(original)
        metadata_count = 0
        if path.suffix == ".py":
            text, metadata_count = neutralize_git_metadata(text, relative)
            if any(git_commit_literals(ast.parse(text, filename=relative))):
                raise ValueError(f"{relative}: development Git commit remains")
        check_content(relative, text, denied_tokens)
        check_dependencies(relative, text, source)
        if path.suffix == ".py":
            check_prompt_constants(original, text, relative)
            before = normalized_ast(original, relative, normalize_paths=True)
            after = normalized_ast(text, relative, normalize_paths=False)
            if before != after:
                raise ValueError(f"{relative}: modification exceeds filesystem or reviewed Git metadata literals/comments")
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n"], input=text, text=True, check=True, capture_output=True)
        elif path.suffix == ".md":
            check_markdown_links(relative, text)
        else:
            raise ValueError(f"{relative}: unexpected source suffix")
        rendered[relative] = text
        records.append({
            "path": relative,
            "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "export_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "filesystem_substitutions": count,
            "private_provenance_substitutions": metadata_count,
        })
    manifest = {
        "schema": "acl-review-source-export-v1",
        "scope": "Allowlisted source and synthetic contract tests only; no history or research artifacts",
        "path_policy": "Historical filesystem defaults use /workspace in this copy; configure authorized dependencies before execution",
        "private_provenance_policy": "Reviewed development Git commit checks are neutralized; historical launchers require authorized private context",
        "verification": "Python AST equality after filesystem/reviewed Git metadata normalization; unchanged prompt definitions; shell syntax; links; local imports/script references; configured identity scan",
        "reproduction_limit": "No training, generation, evaluation, API calls, installation or artifact availability is established by export",
        "files": records,
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    check_content("REVIEW_EXPORT_MANIFEST.json", manifest_text, denied_tokens)
    # Validate all material before creating the new destination. Never remove it on failure.
    destination.mkdir()
    for relative, text in rendered.items():
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        output.chmod((source / relative).stat().st_mode & 0o777)
    (destination / "REVIEW_EXPORT_MANIFEST.json").write_text(manifest_text, encoding="utf-8")
    return {
        "file_count": len(records),
        "filesystem_substitutions": sum(r["filesystem_substitutions"] for r in records),
        "private_provenance_substitutions": sum(r["private_provenance_substitutions"] for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True, help="New directory outside the development repository")
    parser.add_argument("--deny-token", action="append", default=[], help="Additional private identity token; never recorded in the export")
    args = parser.parse_args()
    if any(not token.strip() for token in args.deny_token):
        parser.error("Identity tokens must not be empty")
    try:
        result = build_export(args.source, args.destination, args.deny_token)
    except (OSError, UnicodeError, ValueError, SyntaxError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Export refused: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
