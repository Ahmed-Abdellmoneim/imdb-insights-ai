import re

import pandas as pd
import pytest

pytest.importorskip("anthropic")  # optional ml dependency group

from imdb_insights.reviews import llm  # noqa: E402


def _splits():
    rows = []
    for i in range(3000):
        rating = [1, 2, 3, 4, 7, 8, 9, 10][i % 8] if i % 3 else 1  # skewed towards 1
        rows.append({"review_id": f"stanford-test-pos-{i}", "split": "test", "leaked": i < 30,
                     "text": f"review {i}", "rating": rating})  # fmt: skip
    for i in range(800):
        rows.append({"review_id": f"tmdb-5b0f{i:04x}", "split": "test_ood", "leaked": False,
                     "text": f"tmdb review {i}", "rating": 1 + i % 10})  # fmt: skip
    return pd.DataFrame(rows)


def test_sample_keeps_rating_mix_and_skips_leaked_rows():
    splits = _splits()
    sample = llm.build_sample(splits, n_test=600, n_ood=200)
    test = sample[sample["split"] == "test"]
    assert abs(len(test) - 600) <= 8 and abs((sample["split"] == "test_ood").sum() - 200) <= 10
    assert not sample["leaked"].any()
    clean = splits[(splits["split"] == "test") & ~splits["leaked"]]
    expected = clean["rating"].value_counts(normalize=True)
    observed = test["rating"].value_counts(normalize=True)
    assert (observed - expected).abs().max() < 0.01


def test_requests_have_valid_ids_and_structured_output():
    sample = llm.build_sample(_splits(), n_test=20, n_ood=10)
    requests = llm.build_requests(sample)
    assert len(requests) == len(sample)
    for request in requests:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request["custom_id"])
        params = request["params"]
        assert params["model"] == llm.MODEL
        assert params["output_config"]["format"]["schema"]["required"] == ["rating"]
        assert params["messages"][0]["content"].startswith("<review>")


def test_cost_estimate_is_positive_and_scales():
    sample = llm.build_sample(_splits(), n_test=100, n_ood=50)
    assert 0 < llm.estimate_cost(sample) < llm.estimate_cost(pd.concat([sample, sample]))
