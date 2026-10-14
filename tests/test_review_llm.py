"""The LLM sampling and calibration helpers (the rater itself needs a GPU and model files)."""

import numpy as np
import pandas as pd
import pytest

from imdb_insights.reviews import llm


def _splits():
    rows = []
    for i in range(3000):
        rating = [1, 2, 3, 4, 7, 8, 9, 10][i % 8] if i % 3 else 1  # skewed towards 1
        rows.append({"review_id": f"t{i}", "split": "test", "leaked": i < 30,
                     "text": f"review {i}", "rating": rating})  # fmt: skip
    for i in range(800):
        rows.append({"review_id": f"o{i}", "split": "test_ood", "leaked": False,
                     "text": f"tmdb review {i}", "rating": 1 + i % 10})  # fmt: skip
    for i in range(1000):
        rows.append({"review_id": f"v{i}", "split": "validation", "leaked": False,
                     "text": f"val review {i}", "rating": [1, 4, 7, 10][i % 4]})  # fmt: skip
    return pd.DataFrame(rows)


def test_sample_sizes_rating_mix_and_no_leaked_rows():
    splits = _splits()
    sample = llm.build_sample(splits, n_test=600, n_ood=200, n_validation=100)
    sizes = sample["split"].value_counts()
    assert abs(sizes["test"] - 600) <= 8
    assert abs(sizes["test_ood"] - 200) <= 10
    assert abs(sizes["validation"] - 100) <= 4
    assert not sample["leaked"].any()
    clean = splits[(splits["split"] == "test") & ~splits["leaked"]]
    expected = clean["rating"].value_counts(normalize=True)
    observed = sample.query("split == 'test'")["rating"].value_counts(normalize=True)
    assert (observed - expected).abs().max() < 0.01


def test_sample_is_reproducible():
    a = llm.build_sample(_splits(), n_test=100, n_ood=50, n_validation=20)
    b = llm.build_sample(_splits(), n_test=100, n_ood=50, n_validation=20)
    assert a["review_id"].tolist() == b["review_id"].tolist()


def test_calibration_recovers_a_linear_distortion():
    truth = np.array([1, 2, 4, 7, 9, 10], dtype=float)
    raw = 0.5 * truth + 3  # an LLM that squashes the scale towards the middle
    slope, intercept = llm.calibrate(raw, truth)
    assert slope == pytest.approx(2) and intercept == pytest.approx(-6)
