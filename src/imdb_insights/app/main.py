"""App entry point: navigation and the empty-database guard."""

from __future__ import annotations

import streamlit as st

from . import data, pages


def main() -> None:
    st.set_page_config(page_title="IMDb Insights", page_icon=":material/movie:", layout="wide")

    data.sync_from_hub()
    if not data.database_exists():
        st.error(
            "No data yet. Build the warehouse first:\n\n`uv run python -m imdb_insights.ingest`",
            icon=":material/database:",
        )
        st.stop()

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
            st.Page(
                pages.about_page, title="About the data", icon=":material/info:", url_path="about"
            ),
        ]
    )
    nav.run()
