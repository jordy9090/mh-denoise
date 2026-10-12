from scripts.select_arr_valid_risk_thresholds import metrics_at_threshold, select_threshold


def test_select_threshold_uses_both_valid_classes_and_separates_them():
    selected = select_threshold([0.8, 0.9, 0.1, 0.2], [1, 1, 0, 0])
    assert 0.2 < selected["threshold"] < 0.8
    assert selected["balanced_accuracy"] == 1.0
    assert selected["tp"] == 2 and selected["tn"] == 2


def test_threshold_comparator_matches_runtime_strict_greater_than():
    result = metrics_at_threshold([0.35, 0.36], [0, 1], 0.35)
    assert result["tn"] == 1 and result["tp"] == 1
