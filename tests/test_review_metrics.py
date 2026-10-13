import numpy as np
import pytest

from imdb_insights.reviews.metrics import evaluate, per_rating_mae, point_metrics


def test_perfect_predictions():
    y = np.array([1, 2, 4, 7, 9, 10])
    m = point_metrics(y, y.astype(float))
    assert m["mae"] == 0 and m["rmse"] == 0 and m["within_1"] == 1
    assert m["polarity_accuracy"] == 1 and m["spearman"] == pytest.approx(1)


def test_predictions_are_clipped_to_the_scale():
    m = point_metrics(np.array([10, 1]), np.array([14.0, -3.0]))
    assert m["mae"] == 0


def test_polarity_ignores_neutral_ratings_and_uses_5_5_threshold():
    y_true = np.array([5, 6, 7, 4])
    y_pred = np.array([1.0, 10.0, 5.6, 5.4])  # neutral rows would be "wrong" but are skipped
    assert point_metrics(y_true, y_pred)["polarity_accuracy"] == 1


def test_bias_sign():
    assert point_metrics(np.array([5, 5]), np.array([6.0, 7.0]))["bias"] == pytest.approx(1.5)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(1, 11, 500)
    pred = y + rng.normal(0, 1.5, 500)
    r = evaluate(y, pred, n_boot=200)
    assert r["mae_lo"] <= r["mae"] <= r["mae_hi"]
    assert r["polarity_accuracy_lo"] <= r["polarity_accuracy"] <= r["polarity_accuracy_hi"]
    assert r["n"] == 500


def test_constant_predictions_have_no_rank_correlation():
    assert np.isnan(point_metrics(np.array([1, 5, 10]), np.array([6.0, 6.0, 6.0]))["spearman"])


def test_per_rating_mae():
    assert per_rating_mae(np.array([1, 1, 10]), np.array([3.0, 1.0, 9.0])) == {1: 1.0, 10: 1.0}
