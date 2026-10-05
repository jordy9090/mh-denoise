import scripts.selective_risk_refinement_utils as utils

from scripts.selective_risk_refinement_utils import (
    build_risk_tune_user_content,
    make_zt_from_response,
)


def test_no_mask_preserves_sft_draft_byte_for_byte():
    response = "First sentence.\n\n  Second sentence with  spacing.  "
    z_t, info = make_zt_from_response(
        response,
        g=[1.0] * 6,
        risk_vecs=[[1.0] * 6, [1.0] * 6],
        strategy="no_mask",
    )
    assert z_t == response
    assert info and all(item["state"] == "KEEP" for item in info)
    assert all(item["p_mask"] == 0.0 for item in info)


def test_without_router_uses_scorer_max_and_never_calls_router(monkeypatch):
    monkeypatch.setattr(utils, "predict_g", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router called")))
    monkeypatch.setattr(utils, "score_spans", lambda *args, **kwargs: [[0.1, 0.7, 0.2, 0.0, 0.1, 0.3]])
    score = utils.score_candidate(
        "question", "One sentence.", None, None, object(), object(), "cpu",
        component_mode="without_router",
    )
    assert score["g"] == []
    assert score["risk_score"] == 0.7
    assert score["span_risks"][0]["top_dim"] == utils.DIMS[1]
    assert score["aspect_conditioning_status"] == "unavailable_without_router"


def test_without_scorer_uses_router_max_and_never_calls_scorer(monkeypatch):
    monkeypatch.setattr(utils, "predict_g", lambda *args, **kwargs: [0.1, 0.2, 0.8, 0.3, 0.0, 0.4])
    monkeypatch.setattr(utils, "score_spans", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scorer called")))
    score = utils.score_candidate(
        "question", "One sentence.", object(), object(), None, None, "cpu",
        component_mode="without_scorer",
    )
    assert score["risk_score"] == 0.8
    assert score["risk_vecs"] == [] and score["span_risks"] == []
    assert score["g"][2] == 0.8


def test_without_router_prompt_does_not_inject_zero_aspect_conditioning():
    prompt = build_risk_tune_user_content({
        "question": "Q", "unsafe_response": "U", "sft_response": "S", "z_t_from_sft": "Z",
        "g_sft": [], "aspect_conditioning_status": "unavailable_without_router",
    })
    assert "unavailable (Router removed)" in prompt
    assert "overall_quality=" not in prompt


def test_generation_stop_tokens_prefer_model_generation_config():
    class Object:
        pass

    model = Object()
    model.generation_config = Object()
    model.generation_config.eos_token_id = [1, 106, 50]
    model.config = Object()
    model.config.eos_token_id = [1, 106]
    tokenizer = Object()
    tokenizer.eos_token_id = 1

    token_ids, source = utils.resolve_generation_eos_token_ids(model, tokenizer)
    assert token_ids == [1, 106, 50]
    assert source == "model.generation_config"


def test_generation_stop_tokens_fall_back_to_tokenizer():
    class Object:
        pass

    model = Object()
    tokenizer = Object()
    tokenizer.eos_token_id = 7

    token_ids, source = utils.resolve_generation_eos_token_ids(model, tokenizer)
    assert token_ids == [7]
    assert source == "tokenizer"
