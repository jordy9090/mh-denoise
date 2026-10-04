from scripts.selective_risk_refinement_utils import make_zt_from_response


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
