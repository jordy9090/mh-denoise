from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate_comparison import paired_cluster_differences  # noqa: E402
from build_preference_pairs import build_hard_records, build_minimal_records  # noqa: E402
from dpo_utils import assert_question_disjoint, normalize_text, validate_pair_record  # noqa: E402
from generate_hard_candidates import select_hard_candidate  # noqa: E402


class PreferenceDataContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "id": "q1_empathy",
                "question": "I feel overwhelmed. What can I do tonight?",
                "safe_response": "That sounds exhausting. Choose one small calming step and contact support if you feel unsafe.",
                "unsafe_response": "You definitely have a serious disorder and should double your medication.",
                "target_dimension": "empathy",
                "violation_vector": {"empathy": 1},
            },
            {
                "id": "q2_specificity",
                "question": "I cannot sleep before exams.",
                "safe_response": "Try a brief wind-down routine and write tomorrow's first task before getting into bed.",
                "unsafe_response": "Just relax. It is all in your head.",
                "target_dimension": "specificity",
                "violation_vector": {"specificity": 1},
            },
        ]

    def test_minimal_pair_uses_only_question_and_initial_draft(self) -> None:
        records = build_minimal_records(self.rows)
        self.assertEqual(len(records), 2)
        for source, record in zip(self.rows, records):
            validate_pair_record(record)
            prompt_text = json.dumps(record["prompt"], ensure_ascii=False).lower()
            self.assertIn(source["question"].lower(), prompt_text)
            self.assertIn(source["unsafe_response"].lower(), prompt_text)
            self.assertNotIn("target_dimension", prompt_text)
            self.assertNotIn(source["target_dimension"], prompt_text)
            self.assertEqual(record["chosen"][0]["content"], source["safe_response"])
            self.assertEqual(record["rejected"][0]["content"], source["unsafe_response"])

    def test_hard_k4_builds_one_pair_per_source(self) -> None:
        candidates = []
        for row in self.rows:
            candidates.append(
                {
                    "source_row_id": row["id"],
                    "candidate_budget_k": 4,
                    "selection_version": "test",
                    "hard_rejected_matches_input": False,
                    "hard_rejected": {
                        "text": row["unsafe_response"] + " This is an automated hard negative.",
                        "hard_score": 0.9,
                        "policy_logp_per_token": -1.0,
                        "router_risk": 0.8,
                        "chosen_similarity": 0.4,
                        "origin": "same_question_automated_corruption",
                        "weak_dimension": row["target_dimension"],
                    },
                }
            )
        records = build_hard_records(self.rows, candidates, k=4)
        self.assertEqual(len(records), len(self.rows))
        self.assertTrue(all(record["metadata"]["pair_regime"] == "hard_k4" for record in records))
        self.assertTrue(all(record["metadata"]["hard_negative"]["candidate_budget_k"] == 4 for record in records))

    def test_hard_selector_prefers_plausible_risky_candidate(self) -> None:
        pool = [
            {"text": "weak one", "origin": "same_question_automated_corruption"},
            {"text": "hard one", "origin": "same_question_automated_corruption"},
        ]
        selected, scored = select_hard_candidate(
            pool,
            chosen="safe answer",
            policy_logps=[-4.0, -1.0],
            router_scores=[
                {"router_risk": 0.4, "router_probs": {key: 0.4 for key in ("overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity")}},
                {"router_risk": 0.8, "router_probs": {key: 0.8 for key in ("overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity")}},
            ],
            chosen_router_risk=0.1,
            min_policy_risk_margin=0.05,
            policy_weight=0.45,
            similarity_weight=0.35,
            risk_weight=0.20,
        )
        self.assertEqual(selected["text"], "hard one")
        self.assertEqual(len(scored), 2)

    def test_question_split_leakage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_question_disjoint({"train": [self.rows[0]], "valid": [self.rows[0]]})

    def test_paired_question_bootstrap_aligns_examples(self) -> None:
        baseline = []
        candidate = []
        for index, row in enumerate(self.rows):
            common = {
                **row,
                "source_row_id": row["id"],
                "judgment": {
                    "overall_quality": 3,
                    "empathy": 3,
                    "specificity": 3,
                    "factual_consistency": 3,
                    "toxicity": 2,
                    "medical_advice": "yes",
                },
            }
            baseline.append(common)
            candidate.append(
                {
                    **common,
                    "judgment": {
                        **common["judgment"],
                        "overall_quality": 4,
                        "toxicity": 1,
                        "medical_advice": "no",
                    },
                }
            )
        result = paired_cluster_differences(candidate, baseline, samples=100, seed=7)
        self.assertAlmostEqual(
            result["candidate_minus_baseline"]["overall_quality"]["mean_difference"], 1.0
        )
        self.assertAlmostEqual(
            result["candidate_minus_baseline"]["toxicity"]["mean_difference"], -1.0
        )

    def test_normalization_detects_equivalent_text(self) -> None:
        self.assertEqual(normalize_text("Safe answer!"), normalize_text("safe answer"))


if __name__ == "__main__":
    unittest.main()
