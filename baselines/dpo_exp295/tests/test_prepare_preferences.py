from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_preferences as pp  # noqa: E402


DIMS = ["overall_quality", "empathy", "specificity", "medical_advice"]


def rows_for(question_id: str, question: str, safe: str) -> list[dict]:
    rows = []
    for index, dim in enumerate(DIMS):
        rows.append(
            {
                "id": f"{question_id}_{dim}",
                "question_id": question_id,
                "question": question,
                "safe_response": safe,
                "unsafe_response": f"Flawed answer {index} for {question}",
                "target_dimension": dim,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class PreparePreferencesTest(unittest.TestCase):
    def test_builds_minimal_and_hard_k4_without_label_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = rows_for("train_q1", "I feel overwhelmed.", "It sounds like a lot to carry.")
            train += rows_for("train_q2", "I cannot sleep.", "A gentle sleep routine may help.")
            valid = rows_for("valid_q1", "I keep worrying.", "That worry sounds exhausting.")
            test = rows_for("test_q1", "I feel alone.", "Feeling alone can hurt.")
            paths = {}
            for name, rows in (("train", train), ("valid", valid), ("test", test)):
                paths[name] = root / f"{name}.jsonl"
                write_jsonl(paths[name], rows)

            manifest = pp.prepare_all(
                specs=(
                    pp.SplitSpec("train", paths["train"], len(train)),
                    pp.SplitSpec("valid", paths["valid"], len(valid)),
                    pp.SplitSpec("test", paths["test"], len(test)),
                ),
                output_dir=root / "out",
                modes=("minimal", "hard_k4"),
                k=4,
                seed=42,
            )

            self.assertEqual(manifest["output"]["minimal"]["train"]["rows"], len(train))
            self.assertEqual(manifest["output"]["hard_k4"]["train"]["rows"], len(train) * 4)
            prepared = pp.read_jsonl(root / "out" / "hard_k4" / "train.jsonl")
            self.assertEqual(len(prepared), len(train) * 4)
            for row in prepared:
                prompt = json.dumps(row["prompt"]).lower()
                self.assertNotIn("target_dimension", prompt)
                self.assertNotIn("violation dimension:", prompt)
                self.assertEqual(
                    row["prompt"],
                    pp.build_prompt_messages(row["question"], row["initial_draft"]),
                )
                self.assertEqual(row["pair_mode"], "hard_k4")
                self.assertEqual(len(row["chosen"]), 1)
                self.assertEqual(len(row["rejected"]), 1)

    def test_detects_question_leakage_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = rows_for("shared", "Same question", "Safe response")
            paths = {}
            for name, rows in (("train", shared), ("valid", shared), ("test", rows_for("test", "Other", "Safe"))):
                paths[name] = root / f"{name}.jsonl"
                write_jsonl(paths[name], rows)

            with self.assertRaises(pp.DataValidationError):
                pp.prepare_all(
                    specs=(
                        pp.SplitSpec("train", paths["train"], len(shared)),
                        pp.SplitSpec("valid", paths["valid"], len(shared)),
                        pp.SplitSpec("test", paths["test"], 4),
                    ),
                    output_dir=root / "out",
                    modes=("minimal",),
                    k=4,
                    seed=42,
                )

    def test_rejects_inconsistent_safe_targets_within_question(self) -> None:
        rows = rows_for("q", "Question", "Safe A")
        rows[-1]["safe_response"] = "Safe B"
        with self.assertRaises(pp.DataValidationError):
            pp.validate_split_rows(rows, "train", len(rows))


if __name__ == "__main__":
    unittest.main()
