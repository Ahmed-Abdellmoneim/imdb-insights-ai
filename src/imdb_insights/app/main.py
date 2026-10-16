"""App entry point: navigation and the empty-database guard."""

from __future__ import annotations

import os

import streamlit as st

from ..config import get_settings
from . import data, pages
from .assistant_page import assistant_page
from .review_page import review_page


def _export_secrets() -> None:
    """Make Streamlit secrets visible to the settings, which read environment variables.

    Only top-level string secrets are copied, and real environment variables win.
    """
    try:
        items = dict(st.secrets)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return  # no secrets file: local development with .env
    for key, value in items.items():
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value
    get_settings.cache_clear()


def main() -> None:
    st.set_page_config(page_title="IMDb Insights", page_icon=":material/movie:", layout="wide")
    _export_secrets()

    data.sync_from_hub()
    if not data.database_exists():
        st.error(
            "No data yet. Build the warehouse first:\n\n`uv run python -m imdb_insights.ingest`",
            icon=":material/database:",
        )
        st.stop()

    # Kept in pages.PAGES so other pages can link to it.
    pages.PAGES["write"] = st.Page(
        review_page, title="Write a review", icon=":material/rate_review:", url_path="write"
    )
    nav = st.navigation(
        [
            st.Page(
                pages.top250_page,
                title="Top 250",
                icon=":material/leaderboard:",
                url_path="top250",
                default=True,
            ),
            st.Page(pages.movie_page, title="Movie", icon=":material/movie:", url_path="movie"),
            st.Page(
                pages.insights_page,
                title="Insights",
                icon=":material/insights:",
                url_path="insights",
            ),
            st.Page(assistant_page, title="Ask", icon=":material/forum:", url_path="ask"),
            pages.PAGES["write"],
            st.Page(
                pages.about_page, title="About the data", icon=":material/info:", url_path="about"
            ),
        ]
    )
    nav.run()
