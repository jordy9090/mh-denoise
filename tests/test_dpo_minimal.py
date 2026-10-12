import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_dpo_minimal_pairs as pairs
import train_dpo_minimal as trainer
from selective_risk_refinement_utils import build_sft_prompt


TRAIN_SOURCE = REPO_ROOT / "data/splits_exp295/train_mdlm.jsonl"
VALID_SOURCE = REPO_ROOT / "data/splits_exp295/valid_mdlm.jsonl"
TEST_SOURCE = REPO_ROOT / "data/splits_exp295/test.jsonl"
PROVENANCE_SOURCE = REPO_ROOT / "data/raw/exp295_safe_targets.jsonl"
PAIR_DIR = REPO_ROOT / "data/dpo_exp295_minimal"
SFT_ADAPTER = REPO_ROOT / "outputs/models/gemma4_peft_sft_plain_exp295/final"
HAS_EXP295 = all(path.is_file() for path in (TRAIN_SOURCE, VALID_SOURCE, TEST_SOURCE, PROVENANCE_SOURCE))
HAS_DPO_TEST_DEPS = all(importlib.util.find_spec(name) is not None for name in ("datasets", "peft", "trl"))


class FakeTokenizer:
    chat_template = "fake-chat-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return rendered + ("<GEN>" if add_generation_prompt else "")


@unittest.skipUnless(HAS_EXP295, "local ignored exp295 artifacts are unavailable")
class RealExp295ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_rows = {
            "train": pairs.read_jsonl(TRAIN_SOURCE),
            "valid": pairs.read_jsonl(VALID_SOURCE),
            "test": pairs.read_jsonl(TEST_SOURCE),
        }

    def test_real_split_counts_dimensions_and_leakage(self):
        audits = {
            split: pairs.validate_split(rows, split)
            for split, rows in self.split_rows.items()
        }
        pairs.validate_no_split_leakage(audits)
        self.assertEqual(audits["train"]["rows"], 1242)
        self.assertEqual(audits["valid"]["rows"], 174)
        self.assertEqual(audits["test"]["rows"], 354)
        self.assertEqual(audits["train"]["questions"], 207)
        self.assertEqual(audits["valid"]["questions"], 29)
        self.assertEqual(audits["test"]["questions"], 59)
        self.assertEqual(set(audits["train"]["dimension_counts"]), set(pairs.DIMENSIONS))

    def test_real_provenance(self):
        audit = pairs.validate_provenance(
            pairs.read_jsonl(PROVENANCE_SOURCE),
            self.split_rows,
        )
        self.assertEqual(
            audit["question_counts"],
            {"counselbench_eval_strict": 99, "counselchat_judged": 196},
        )
        self.assertEqual(
            audit["split_question_counts"]["train"],
            {"counselbench_eval_strict": 71, "counselchat_judged": 136},
        )
        self.assertEqual(
            audit["split_question_counts"]["valid"],
            {"counselbench_eval_strict": 8, "counselchat_judged": 21},
        )
        self.assertEqual(
            audit["split_question_counts"]["test"],
            {"counselbench_eval_strict": 20, "counselchat_judged": 39},
        )

    def test_test_pair_construction_is_forbidden(self):
        with self.assertRaisesRegex(pairs.DataContractError, "forbidden"):
            pairs.build_split_pairs(self.split_rows["test"], "test", lambda row: "prompt")

    def test_generated_pair_files_are_train_valid_only(self):
        train_pairs = pairs.read_jsonl(PAIR_DIR / "train.jsonl")
        valid_pairs = pairs.read_jsonl(PAIR_DIR / "valid.jsonl")
        self.assertEqual(trainer.validate_pair_rows(train_pairs, "train")["rows"], 1242)
        self.assertEqual(trainer.validate_pair_rows(valid_pairs, "valid")["rows"], 174)
        self.assertFalse((PAIR_DIR / "test.jsonl").exists())
        manifest = json.loads((PAIR_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tokenization_audit"]["train"]["prompt_over_limit"], 0)
        self.assertEqual(manifest["tokenization_audit"]["valid"]["prompt_over_limit"], 0)
        self.assertEqual(manifest["tokenization_audit"]["train"]["chosen_prefix_mismatches"], 0)
        self.assertEqual(manifest["tokenization_audit"]["valid"]["rejected_prefix_mismatches"], 0)


class PairMappingTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "id": "exp300_safe_0001_toxicity",
            "question": "  Why do I feel this way?  ",
            "unsafe_response": "  You are definitely broken.  ",
            "safe_response": "  That sounds painful, and support may help.  ",
            "target_dimension": "toxicity",
            "violation_vector": {
                "overall_quality": 0,
                "empathy": 0,
                "specificity": 0,
                "medical_advice": 0,
                "factual_consistency": 0,
                "toxicity": 1,
            },
            "brief_reason": "AUDIT_SECRET_REASON_DO_NOT_LEAK",
            "source": "legacy",
            "generator": "generator",
            "version": "exp295_v1",
        }

    def test_pair_reuses_exact_sft_plain_prompt_and_mapping(self):
        tokenizer = FakeTokenizer()
        pair = pairs.build_pair(self.row, lambda row: build_sft_prompt(tokenizer, dict(row)))
        expected = build_sft_prompt(
            tokenizer,
            {"question": self.row["question"], "unsafe_response": self.row["unsafe_response"]},
        )
        self.assertEqual(pair["prompt"], expected)
        self.assertEqual(pair["chosen"], "That sounds painful, and support may help.")
        self.assertEqual(pair["rejected"], "You are definitely broken.")
        self.assertNotIn("AUDIT_SECRET_REASON_DO_NOT_LEAK", pair["prompt"])
        self.assertNotIn("violation_vector", pair["prompt"])
        self.assertEqual(pair["audit_metadata"]["brief_reason"], "AUDIT_SECRET_REASON_DO_NOT_LEAK")

    def test_model_only_projection_removes_all_audit_metadata(self):
        pair = pairs.build_pair(self.row, lambda row: "q+d prompt")
        projected = trainer.model_only_rows([pair])
        self.assertEqual(set(projected[0]), {"prompt", "chosen", "rejected"})
        self.assertNotIn("audit_metadata", projected[0])
        self.assertNotIn("id", projected[0])

    def test_split_leakage_guard(self):
        audits = {
            "train": {"question_group_ids": ["q1"], "normalized_questions": ["same question"]},
            "valid": {"question_group_ids": ["q2"], "normalized_questions": ["same question"]},
            "test": {"question_group_ids": ["q3"], "normalized_questions": ["different"]},
        }
        with self.assertRaisesRegex(pairs.DataContractError, "Leakage"):
            pairs.validate_no_split_leakage(audits)


class CollatorTests(unittest.TestCase):
    def test_separate_budgets_preserve_completion_and_eos(self):
        collator = trainer.DPOPreferenceCollator(
            pad_token_id=0,
            eos_token_id=2,
            max_prompt_length=4,
            max_completion_length=3,
        )
        batch = collator(
            [
                {
                    "prompt_ids": [1, 3, 4, 5, 6, 7],
                    "chosen_ids": [10, 11, 12, 13, 2],
                    "rejected_ids": [20, 21, 2],
                }
            ]
        )
        self.assertEqual(batch["input_ids"].shape, (2, 7))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 3, 4, 5, 10, 11, 2])
        self.assertEqual(batch["input_ids"][1].tolist(), [1, 3, 4, 5, 20, 21, 2])
        self.assertEqual(batch["completion_mask"][0].tolist(), [0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), [7, 7])

    def test_reference_logps_are_forwarded(self):
        collator = trainer.DPOPreferenceCollator(0, 2, 4, 3)
        batch = collator(
            [
                {
                    "prompt_ids": [1],
                    "chosen_ids": [3, 2],
                    "rejected_ids": [4, 2],
                    "ref_chosen_logps": -1.0,
                    "ref_rejected_logps": -2.0,
                }
            ]
        )
        self.assertEqual(batch["ref_chosen_logps"].tolist(), [-1.0])
        self.assertEqual(batch["ref_rejected_logps"].tolist(), [-2.0])


class _AdapterPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.default = nn.Linear(2, 2)
        self.ref = nn.Linear(2, 2)
        self.ref.load_state_dict(self.default.state_dict())
        for parameter in self.ref.parameters():
            parameter.requires_grad = False


class _FakePeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = _AdapterPair()
        self.base_weight = nn.Parameter(torch.ones(1), requires_grad=False)


class ReferenceAuditTests(unittest.TestCase):
    def test_policy_cast_leaves_frozen_reference_unchanged(self):
        model = _FakePeftModel()
        reference_before = model.lora.ref.weight.detach().clone()
        audit = trainer.cast_policy_adapter_dtype(model, torch.bfloat16)
        self.assertEqual(audit["dtype"], "torch.bfloat16")
        self.assertEqual(model.lora.default.weight.dtype, torch.bfloat16)
        self.assertEqual(model.lora.ref.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(model.lora.ref.weight, reference_before))

    def test_equal_frozen_reference_passes(self):
        audit = trainer.audit_adapter_state(_FakePeftModel())
        self.assertEqual(audit["max_abs_diff"], 0.0)
        self.assertEqual(audit["reference_trainable_tensor_count"], 0)
        self.assertEqual(audit["unexpected_trainable_tensor_count"], 0)
        self.assertGreater(audit["policy_trainable_tensor_count"], 0)

    def test_changed_reference_fails(self):
        model = _FakePeftModel()
        with torch.no_grad():
            model.lora.ref.weight[0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "initialization differs"):
            trainer.audit_adapter_state(model)

    @unittest.skipUnless(
        SFT_ADAPTER.is_dir() and HAS_DPO_TEST_DEPS,
        "local ignored SFT tokenizer or optional DPO test dependencies are unavailable",
    )
    def test_trl_creates_exact_frozen_reference_adapter_on_cpu(self):
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
        from trl import DPOConfig, DPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(tokenizer),
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        dataset = Dataset.from_list(
            [
                {"prompt": "Question: hello\nAnswer:", "chosen": " safe", "rejected": " unsafe"},
                {"prompt": "Question: help\nAnswer:", "chosen": " bounded", "rejected": " diagnose"},
            ]
        )
        with tempfile.TemporaryDirectory() as output_dir:
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=DPOConfig(
                    output_dir=output_dir,
                    use_cpu=True,
                    per_device_train_batch_size=1,
                    max_length=64,
                    gradient_checkpointing=False,
                    precompute_ref_log_probs=True,
                    precompute_ref_batch_size=1,
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                data_collator=trainer.DPOPreferenceCollator(
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_prompt_length=48,
                    max_completion_length=16,
                ),
            )
            audit = trainer.audit_adapter_state(dpo_trainer.model)
            self.assertIn("ref_chosen_logps", dpo_trainer.train_dataset.column_names)
            self.assertIn("ref_rejected_logps", dpo_trainer.train_dataset.column_names)
        self.assertEqual(audit["max_abs_diff"], 0.0)
        self.assertEqual(audit["reference_trainable_tensor_count"], 0)


@unittest.skipUnless(SFT_ADAPTER.is_dir(), "local ignored SFT checkpoint is unavailable")
class CheckpointContractTests(unittest.TestCase):
    def test_actual_sft_checkpoint_contract(self):
        audit = trainer.validate_checkpoint_contract(SFT_ADAPTER, "google/gemma-4-E4B-it")
        self.assertEqual(audit["lora_r"], 8)
        self.assertEqual(audit["lora_alpha"], 16)
        self.assertEqual(audit["target_module_count"], 132)
        self.assertEqual(
            audit["adapter_sha256"],
            "f9b0c24bcd0ad05e133352e0c8fa8482b4a6b43766b84fa6c7686e2f2850bc50",
        )


if __name__ == "__main__":
    unittest.main()
