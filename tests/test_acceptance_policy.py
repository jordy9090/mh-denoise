"""CPU-only policy tests; importing the GPU inference stack is unnecessary."""
import argparse
import ast
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_gemma_selective_risk_refinement.py"
TREE = ast.parse(RUNNER.read_text(encoding="utf-8"))


def load_functions(tree, names):
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"math": math, "clean_text": lambda value: (value or "").strip()}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


FUNCTIONS = load_functions(TREE, {"acceptance_reasons", "acceptance_policy_settings"})
acceptance_reasons = FUNCTIONS["acceptance_reasons"]
acceptance_policy_settings = FUNCTIONS["acceptance_policy_settings"]


def args(**updates):
    values = dict(
        acceptance_policy="surface_relaxed_v1", min_word_count=20,
        min_risk_delta=0.01, min_focus_risk_delta=0.0,
        gate_strategy="overall", specificity_min_ratio=0.60,
        keyword_overlap_slack=0.15,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def metrics(**updates):
    values = dict(
        response="A nonempty response for policy testing.", word_count=50,
        risk_score=0.50, focus_risk_score=0.20, bad_safety_count=0,
        specificity_ratio=1.0, generic_count=0, keyword_overlap=0.50,
    )
    values.update(updates)
    return values


class AcceptancePolicyTest(unittest.TestCase):
    def setUp(self):
        self.sft = metrics(risk_score=0.80)
        self.candidate = metrics()

    def test_only_two_surface_vetoes_are_demoted(self):
        candidate = metrics(specificity_ratio=0.23, generic_count=1)
        self.assertEqual(acceptance_reasons(self.sft, candidate, args(acceptance_policy="legacy")),
                         ["specificity_ratio_low", "genericity_increased"])
        self.assertEqual(acceptance_reasons(self.sft, candidate, args()), [])

    def test_default_for_programmatic_callers_remains_legacy(self):
        settings = args()
        del settings.acceptance_policy
        candidate = metrics(specificity_ratio=0.23, generic_count=1)
        self.assertEqual(acceptance_reasons(self.sft, candidate, settings),
                         ["specificity_ratio_low", "genericity_increased"])
        self.assertEqual(acceptance_policy_settings(settings)["acceptance_policy"], "legacy")

    def test_improvement_boundary_and_no_improvement(self):
        settings = args()
        boundary = self.sft["risk_score"] - settings.min_risk_delta
        self.assertEqual(acceptance_reasons(self.sft, metrics(risk_score=boundary), settings), [])
        self.assertIn("risk_not_improved", acceptance_reasons(
            self.sft, metrics(risk_score=math.nextafter(boundary, math.inf)), settings))
        self.assertIn("risk_not_improved", acceptance_reasons(
            self.sft, metrics(risk_score=self.sft["risk_score"]), settings))

    def test_empty_and_minimum_length_checks_remain(self):
        self.assertEqual(acceptance_reasons(self.sft, metrics(response=" ", word_count=0), args()),
                         ["empty_denoiser_response", "short_denoiser_response"])
        self.assertEqual(acceptance_reasons(self.sft, metrics(word_count=20), args()), [])
        self.assertEqual(acceptance_reasons(self.sft, metrics(word_count=19), args()),
                         ["short_denoiser_response"])

    def test_bad_safety_and_keyword_vetoes_remain(self):
        candidate = metrics(bad_safety_count=1, keyword_overlap=0.0,
                            specificity_ratio=0.2, generic_count=1)
        self.assertEqual(acceptance_reasons(self.sft, candidate, args()),
                         ["bad_safety_increased", "question_keyword_overlap_low"])
        boundary = self.sft["keyword_overlap"] - args().keyword_overlap_slack
        self.assertEqual(acceptance_reasons(self.sft, metrics(keyword_overlap=boundary), args()), [])
        self.assertIn("question_keyword_overlap_low", acceptance_reasons(
            self.sft, metrics(keyword_overlap=math.nextafter(boundary, -math.inf)), args()))

    def test_missing_or_nonfinite_overall_risk_fails_closed(self):
        for side, invalid in itertools.product(("sft", "denoiser"), (None, math.nan, math.inf, -math.inf, "bad")):
            with self.subTest(side=side, invalid=invalid):
                sft, candidate = dict(self.sft), dict(self.candidate)
                (sft if side == "sft" else candidate)["risk_score"] = invalid
                self.assertIn(f"invalid_{side}_risk_score", acceptance_reasons(sft, candidate, args()))
        candidate = dict(self.candidate)
        del candidate["risk_score"]
        self.assertIn("invalid_denoiser_risk_score", acceptance_reasons(self.sft, candidate, args()))

    def test_focus_risk_applies_only_when_enabled(self):
        candidate = metrics(focus_risk_score=0.4)
        self.assertEqual(acceptance_reasons(self.sft, candidate, args()), [])
        for strategy in ("aspect_only", "aspect_or_overall"):
            self.assertIn("focus_risk_not_improved", acceptance_reasons(
                self.sft, candidate, args(gate_strategy=strategy)))
            self.assertIn("invalid_denoiser_focus_risk_score", acceptance_reasons(
                self.sft, metrics(focus_risk_score=math.nan), args(gate_strategy=strategy)))
        candidate = dict(self.candidate)
        del candidate["focus_risk_score"]
        self.assertEqual(acceptance_reasons(self.sft, candidate, args()), [])
        self.assertIn("invalid_denoiser_focus_risk_score", acceptance_reasons(
            self.sft, candidate, args(gate_strategy="aspect_only")))

    def test_invalid_delta_fails_closed_for_new_policy(self):
        self.assertIn("invalid_risk_score_delta", acceptance_reasons(
            self.sft, self.candidate, args(min_risk_delta=math.nan)))

    def test_legacy_nan_semantics_are_unchanged(self):
        self.assertEqual(acceptance_reasons(
            self.sft, metrics(risk_score=math.nan), args(acceptance_policy="legacy")), [])

    def test_unknown_policy_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown acceptance policy"):
            acceptance_reasons(self.sft, self.candidate, args(acceptance_policy="typo"))
        with self.assertRaisesRegex(ValueError, "Unknown acceptance policy"):
            acceptance_policy_settings(args(acceptance_policy="typo"))

    def test_settings_capture_exact_policy(self):
        settings = acceptance_policy_settings(args())
        for key in ("min_word_count", "min_risk_delta", "min_focus_risk_delta", "gate_strategy",
                    "specificity_min_ratio", "keyword_overlap_slack"):
            self.assertEqual(settings[key], getattr(args(), key))
        self.assertFalse(settings["surface_heuristic_vetoes_enabled"])
        self.assertTrue(settings["require_finite_risk"])

    def test_cli_default_and_choices(self):
        main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        statements = []
        for statement in main.body:
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ap" for t in statement.targets):
                statements.append(statement)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                method = statement.value.func
                if isinstance(method, ast.Attribute) and isinstance(method.value, ast.Name) and method.value.id == "ap" and method.attr == "add_argument":
                    statements.append(statement)
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(RUNNER), "exec"), namespace)
        parser = namespace["ap"]
        policy = next(action for action in parser._actions if action.dest == "acceptance_policy")
        self.assertEqual(policy.default, "legacy")
        self.assertEqual(policy.choices, ["legacy", "surface_relaxed_v1"])

    def test_legacy_matches_original_function_when_base_available(self):
        original = ROOT.parent / "base" / "scripts" / RUNNER.name
        if not original.exists():
            self.skipTest("Original source is not bundled with this installation")
        old = load_functions(ast.parse(original.read_text(encoding="utf-8")), {"acceptance_reasons"})["acceptance_reasons"]
        for risk, words, ratio, generic, safety, overlap, strategy in itertools.product(
                (0.40, 0.79, 0.80, math.nan), (0, 19, 20), (0.20, 0.60),
                (0, 1), (0, 1), (0.10, 0.50), ("overall", "aspect_only")):
            candidate = metrics(risk_score=risk, word_count=words, specificity_ratio=ratio,
                                generic_count=generic, bad_safety_count=safety,
                                keyword_overlap=overlap, response="" if words == 0 else "Text")
            settings = args(acceptance_policy="legacy", gate_strategy=strategy)
            self.assertEqual(acceptance_reasons(self.sft, candidate, settings), old(self.sft, candidate, settings))


if __name__ == "__main__":
    unittest.main()
