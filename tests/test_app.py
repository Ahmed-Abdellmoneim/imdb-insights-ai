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
