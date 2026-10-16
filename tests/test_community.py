"""Community reviews: checks, storage, moderation, the rating head and the warehouse merge."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from imdb_insights.community import cli, rater
from imdb_insights.community.store import LocalStore, open_store
from imdb_insights.community.submissions import (
    Decision,
    Submission,
    approved_reviews,
    clean,
    flags,
    latest,
    problems,
)
from imdb_insights.config import get_settings

TEXT = (
    "A patient, beautifully shot drama that earns every minute of its running time. The lead "
    "performance carries the quieter middle section, and the score never tells you what to "
    "feel. The ending lands because the film has been honest with its characters throughout."
)


def make(text: str = TEXT, **kwargs) -> Submission:
    fields = {"tmdb_id": 1, "movie_title": "Movie 1 (2001)", "rating": 8, "session": "abc"}
    return Submission(text=text, **(fields | kwargs))


def test_problems_accepts_a_real_review_and_explains_refusals():
    assert problems(TEXT) == []
    assert "at least 150 characters" in problems("Great film!")[0]
    assert "30 words" in problems("Supercalifragilistic " * 10)[0]
    assert "at most 40" in problems(TEXT, author="x" * 41)[0]


def test_flags_point_at_contact_details_links_and_shouting():
    assert flags(TEXT) == []
    found = flags(TEXT + " Mail me at someone@example.com or call +1 555 123 4567, see www.x.io")
    assert {"email address", "phone number", "link"} <= set(found)
    assert "mostly capitals" in flags(TEXT.upper())
    assert "repetitive wording" in flags("so good " * 40)


def test_clean_trims_lines_and_collapses_blank_runs():
    assert clean("  first  \r\n\r\n\r\n\n  second ") == "first\n\nsecond"
    assert clean(None) == ""


def test_local_store_round_trip_and_latest_decision_wins(tmp_path):
    store = LocalStore(tmp_path)
    a, b = make(), make(rating=3)
    store.add_submission(a)
    store.add_submission(b)
    now = datetime.now(UTC)
    earlier = now - timedelta(hours=1)
    rejected = Decision(submission_id=a.submission_id, status="rejected", decided_at=earlier)
    store.add_decisions([rejected])
    store.add_decisions([Decision(submission_id=a.submission_id, status="approved")])
    submissions, decisions = store.load()
    assert {s.submission_id for s in submissions} == {a.submission_id, b.submission_id}
    decided = latest(decisions)
    assert decided[a.submission_id].status == "approved"
    assert b.submission_id not in decided


def test_only_approved_reviews_reach_the_warehouse_table():
    a, b = make(headline="Worth it"), make()
    decisions = {
        a.submission_id: Decision(submission_id=a.submission_id, status="approved"),
        b.submission_id: Decision(submission_id=b.submission_id, status="rejected"),
    }
    rows = approved_reviews([a, b], decisions)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["review_id"] == f"community-{a.submission_id}"
    assert row["source"] == "community" and row["author"] == "Anonymous"
    assert row["content"].startswith("Worth it\n\n")


def test_similar_reviews_catches_copied_text():
    original = make(created_at=datetime.now(UTC) - timedelta(days=1))
    copy = make(text=TEXT + " Also, popcorn.")
    existing = pd.DataFrame([{"tmdb_id": 1, "author": "critic", "content": "Intro. " + TEXT}])
    found = cli.similar_reviews(copy, [original, copy], existing)
    assert any("submission" in f for f in found)
    assert any("TMDB review by critic" in f for f in found)
    assert cli.similar_reviews(original, [original, copy], existing[:0]) == []


def test_rating_head_round_trip_and_clipping(tmp_path):
    path = tmp_path / "head.json"
    coef = [0.0] * 384
    coef[0] = 100.0
    rater.save_head({"coef": coef, "intercept": 5.5}, path)
    head = rater.RatingHead.load(path)
    embeddings = np.zeros((3, 384), dtype=np.float32)
    embeddings[1, 0], embeddings[2, 0] = 1.0, -1.0
    assert head.predict(embeddings).tolist() == [5.5, 10.0, 1.0]
    assert len(head.version) == 10
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))  # a Windows checkout
    assert rater.RatingHead.load(crlf).version == head.version
    with pytest.raises(ValueError):
        rater.save_head({"coef": [1.0], "intercept": 0}, path)


def test_shipped_rating_head_matches_the_encoder():
    import json

    from imdb_insights.assistant.embedder import REPO

    assert rater.RatingHead.load().coef.shape == (384,)
    meta = json.loads(rater.HEAD_PATH.read_text(encoding="utf-8"))
    assert meta["encoder"] == REPO  # the app's ONNX encoder is the model it was trained on
    assert meta["validation_mae"] < 1.5


def test_report_scores_the_model_on_approved_reviews():
    good = make(rating=9, model_rating=8.2, model_version="v1")
    bad = make(rating=2, model_rating=3.1, model_version="v1", flags=["link"])
    pending = make(rating=5, model_rating=5.0, model_version="v1")
    decisions = {
        good.submission_id: Decision(submission_id=good.submission_id, status="approved"),
        bad.submission_id: Decision(submission_id=bad.submission_id, status="approved"),
    }
    text = cli.report([good, bad, pending], decisions)
    assert "3 submissions: 2 approved, 0 rejected, 1 pending" in text
    assert "| `v1` | 2 | 0.95" in text
    assert "flags were raised on 1" in text


def test_store_choice_and_token_fallback(tmp_path, monkeypatch):
    assert isinstance(open_store(tmp_path, None, None), LocalStore)
    monkeypatch.setenv("HF_TOKEN", "hf_main")
    get_settings.cache_clear()
    assert get_settings().hf_community_token.get_secret_value() == "hf_main"
    monkeypatch.setenv("HF_COMMUNITY_TOKEN", "hf_scoped")
    get_settings.cache_clear()
    assert get_settings().hf_community_token.get_secret_value() == "hf_scoped"


def test_ingest_adds_approved_reviews_for_known_movies(tmp_path, monkeypatch):
    from imdb_insights.ingest.cli import _with_community

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    store = LocalStore(tmp_path / "community" / "local")
    known, unknown = make(tmdb_id=1), make(tmdb_id=999)
    for s in (known, unknown):
        store.add_submission(s)
    store.add_decisions(
        [Decision(submission_id=s.submission_id, status="approved") for s in (known, unknown)]
    )
    tmdb = pd.DataFrame([{"review_id": "t1", "tmdb_id": 1, "source": "tmdb", "content": "x"}])
    reviews, added = _with_community(tmdb, pd.DataFrame({"tmdb_id": [1, 2]}))
    assert added == 1
    assert reviews["source"].tolist() == ["tmdb", "community"]
