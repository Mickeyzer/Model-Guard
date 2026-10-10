from types import SimpleNamespace

from middleware.feature_extractor import extract_behavioral_features


def row(features, gap_ms, raw_pred, raw_probs, returned_pred=None, returned_probs=None):
    return SimpleNamespace(
        features=features, inter_query_time_ms=gap_ms,
        raw_prediction=raw_pred, raw_probabilities=raw_probs,
        returned_prediction=returned_pred, returned_probabilities=returned_probs,
    )


def test_empty_window_is_all_zero():
    assert set(extract_behavioral_features([]).values()) == {0.0}


def test_mean_gap_ignores_first_query():
    # The first gap points outside the window (here: an hour earlier)
    rows = [row([5.0, 3.0, 1.5, 0.2], gap, 0, [1.0, 0.0, 0.0]) for gap in (3_600_000, 1000, 3000)]
    assert extract_behavioral_features(rows)["mean_inter_query_time_ms"] == 2000


def test_prediction_features_use_raw_not_returned_outputs():
    # The real model is confident and consistent; the returned answers were
    # fakes spread across classes. Only the real outputs should count.
    rows = [
        row([5.0, 3.0, 1.5, 0.2], 1000, 0, [1.0, 0.0, 0.0], returned_pred=i % 3, returned_probs=[0.4, 0.3, 0.3])
        for i in range(6)
    ]
    feats = extract_behavioral_features(rows)
    assert feats["prediction_entropy"] == 0.0
    assert feats["boundary_proximity_score"] == 0.0


def test_blocked_rows_without_model_output_are_skipped():
    rows = [row([5.0, 3.0, 1.5, 0.2], 1000, None, None) for _ in range(4)]
    rows.append(row([6.0, 3.0, 4.5, 1.5], 1000, 1, [0.0, 0.6, 0.4]))
    feats = extract_behavioral_features(rows)
    assert feats["query_count_1min"] == 5.0
    assert feats["boundary_proximity_score"] == 1.0
