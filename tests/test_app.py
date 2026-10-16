"""Render every app page against a small warehouse built from fake API data."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_pipeline import run_collect

from imdb_insights.config import get_settings
from imdb_insights.ingest import pipeline


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("warehouse")
    col, _ = run_collect(tmp / "raw", omdb_quota=20)  # partial run: exercises the banner
    tables = pipeline.build_tables(col)
    report = pipeline.quality_report(col, tables)
    pipeline.write_outputs(tables, report, tmp / "processed", started=pipeline.datetime.now())
    return tmp


@pytest.fixture(autouse=True)
def use_warehouse(warehouse, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(warehouse))
    get_settings.cache_clear()
    st.cache_data.clear()
    yield
    get_settings.cache_clear()


def _page_script(page_name: str) -> None:
    from imdb_insights.app import pages

    getattr(pages, page_name)()


@pytest.mark.parametrize("page", ["top250_page", "movie_page", "insights_page", "about_page"])
def test_page_renders_without_errors(page):
    at = AppTest.from_function(_page_script, args=(page,), default_timeout=30).run()
    assert not at.exception, at.exception


def test_top250_shows_ranked_movies_and_partial_banner():
    at = AppTest.from_function(_page_script, args=("top250_page",), default_timeout=30).run()
    assert at.metric[0].value == "20"  # only the 20 movies with IMDb ratings are ranked
    assert any("provisional" in w.value for w in at.warning)


def test_movie_page_follows_query_param():
    at = AppTest.from_function(_page_script, args=("movie_page",), default_timeout=30)
    at.query_params["id"] = "7"
    at.run()
    assert not at.exception
    assert at.title[0].value == "Movie 7"


def test_missing_database_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def script():
        from imdb_insights.app.main import main

        main()

    at = AppTest.from_function(script, default_timeout=30).run()
    assert "No data yet" in at.error[0].value


def _ask_page_script() -> None:
    from imdb_insights.app.assistant_page import assistant_page

    assistant_page()


def test_ask_page_explains_how_to_add_a_key_when_none_is_set():
    at = AppTest.from_function(_ask_page_script, default_timeout=30).run()
    assert not at.exception
    assert "LLM_API_KEY" in at.info[0].value
    assert not at.chat_input  # nothing to type into without a key


def test_streamlit_secrets_reach_the_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))  # no warehouse: main() stops early
    monkeypatch.delenv("GROQ", raising=False)
    monkeypatch.delenv("groq", raising=False)
    get_settings.cache_clear()

    def script():
        from imdb_insights.app.main import main

        main()

    at = AppTest.from_function(script, default_timeout=30)
    at.secrets["groq"] = "gsk_from_secrets"
    at.run()
    assert get_settings().llm_api_key.get_secret_value() == "gsk_from_secrets"


def test_ask_page_lists_secret_names_but_never_values():
    at = AppTest.from_function(_ask_page_script, default_timeout=30)
    at.secrets["HF_TOKEN"] = "hf_secret_value"
    at.secrets["llm"] = {"key": "gsk_secret_value"}  # nested: not read
    at.run()
    hint = at.caption[-1].value
    assert "HF_TOKEN" in hint and "[llm] key" in hint
    assert "secret_value" not in hint


def _write_page_script() -> None:
    from imdb_insights.app.review_page import review_page

    review_page()


REVIEW = (
    "A patient, beautifully shot drama that earns every minute of its running time. The lead "
    "performance carries the quieter middle section, and the score never tells you what to "
    "feel. The ending lands because the film has been honest with its characters throughout."
)


def _fill(at, text=REVIEW, rating=8, consent=True):
    at.selectbox[0].set_value(7)
    at.button_group[0].set_value(rating)
    at.text_area[0].input(text)
    at.checkbox[0].set_value(consent)
    at.button[0].click()
    return at.run()


def test_write_page_saves_a_review_and_shows_the_model_guess(monkeypatch, warehouse):
    from imdb_insights.community.store import LocalStore

    monkeypatch.setattr("imdb_insights.app.review_page._guess", lambda text: (6.5, "test"))
    at = AppTest.from_function(_write_page_script, default_timeout=30).run()
    assert not at.exception
    at = _fill(at)
    assert not at.exception, at.exception
    assert "Thank you" in at.success[0].value
    assert [m.value for m in at.metric] == ["8/10", "6.5/10"]
    submissions, _ = LocalStore(warehouse / "community" / "local").load()
    saved = [s for s in submissions if s.text == REVIEW][-1]
    assert (saved.tmdb_id, saved.rating, saved.model_rating) == (7, 8, 6.5)
    assert at.text_area[0].value == ""  # a fresh form for the next review

    at = _fill(at)  # the same text again
    assert "already submitted" in at.error[0].value


def test_write_page_refuses_short_reviews_without_consent(monkeypatch):
    monkeypatch.setattr("imdb_insights.app.review_page._guess", lambda text: (None, None))
    at = _fill(AppTest.from_function(_write_page_script, default_timeout=30).run(),
               text="Great!", consent=False)  # fmt: skip
    errors = " ".join(e.value for e in at.error)
    assert "at least 150 characters" in errors and "consent" in errors
    assert not at.success


def test_write_page_is_off_when_deployed_without_a_community_dataset(monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/data")
    get_settings.cache_clear()
    at = AppTest.from_function(_write_page_script, default_timeout=30).run()
    assert "not switched on" in at.info[0].value
    assert not at.text_area
