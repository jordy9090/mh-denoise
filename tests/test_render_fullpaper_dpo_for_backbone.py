import hashlib

import pytest

from scripts.render_fullpaper_dpo_for_backbone import render_pair


class FakeTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT:" + messages[-1]["content"]


def frozen_rows():
    sft = {
        "id": "qa_test",
        "question": "How are you?",
        "safe_response": "Safe answer.",
        "unsafe_response": "Risky answer.",
        "split": "valid",
        "question_normalized_sha256": "qhash",
    }
    dpo = {
        "chosen": "Safe answer.",
        "rejected": "Risky answer.",
        "input": {"question": "How are you?", "corrupted_response": "Risky answer."},
        "metadata": {"canonical_id": "qa_test", "source_marker": "preserved"},
    }
    return sft, dpo


def test_render_pair_normalizes_nested_frozen_dpo_contract():
    sft, dpo = frozen_rows()
    row = render_pair(sft, dpo, FakeTokenizer(), "repo", "revision")
    assert row["id"] == "qa_test"
    assert row["question_group_id"] == "qhash"
    assert row["chosen"] == "Safe answer."
    assert row["rejected"] == "Risky answer."
    assert row["audit_metadata"]["contract_version"] == "fullpaper-dpo-v1"
    assert row["audit_metadata"]["original_split"] == "valid"
    assert row["audit_metadata"]["development_role"] == "valid"
    assert row["audit_metadata"]["source_marker"] == "preserved"
    assert row["audit_metadata"]["chosen_sha256"] == hashlib.sha256(b"Safe answer.").hexdigest()


def test_render_pair_rejects_sft_dpo_text_mismatch():
    sft, dpo = frozen_rows()
    dpo["input"]["question"] = "Different question"
    with pytest.raises(RuntimeError, match="question or response text mismatch"):
        render_pair(sft, dpo, FakeTokenizer(), "repo", "revision")
