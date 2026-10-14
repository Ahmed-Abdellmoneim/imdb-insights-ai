"""Read-only queries against the warehouse, cached per Streamlit server process."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from .. import publish
from ..config import get_settings
from ..ingest.pipeline import DB_FILENAME

log = logging.getLogger(__name__)

# Directors and top-billed cast, used by several pages.
GENRES_CTE = """
    genres as (
        select tmdb_id, list(genre order by genre) as genres from movie_genres group by tmdb_id
    )
"""
DIRECTORS_CTE = """
    directors as (
        select c.tmdb_id, list(p.name order by p.name) as directors
        from credits c join people p using (person_id)
        where c.job = 'Director'
        group by c.tmdb_id
    )
"""


def processed_dir() -> Path:
    return get_settings().data_dir / "processed"


def database_path() -> Path:
    return processed_dir() / DB_FILENAME


def database_exists() -> bool:
    return database_path().exists()


@st.cache_resource(ttl=timedelta(hours=1), show_spinner="Fetching the latest data…")
def sync_from_hub() -> str | None:
    """Download the published warehouse when configured (deployments); at most hourly.

    Returns the published run's finish time. Failures keep whatever data is already here.
    """
    settings = get_settings()
    if not settings.hf_dataset_repo or not settings.hf_token:
        return None  # local development: use the locally built warehouse
    before = manifest_finished_at()
    try:
        publish.pull_warehouse(
            settings.data_dir, settings.hf_dataset_repo, settings.hf_token.get_secret_value()
        )
    except Exception:  # noqa: BLE001 - a failed refresh must never take the app down
        log.exception("Could not fetch the published warehouse; keeping the current data")
    after = manifest_finished_at()
    if after != before:
        st.cache_data.clear()  # new data: drop cached query results
    return after


def manifest_finished_at() -> str | None:
    path = processed_dir() / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("finished_at")


def query(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    # A short-lived connection per query (results are cached by the callers), so the file
    # is not held open and the ingestion pipeline can swap in a fresh database.
    with duckdb.connect(str(processed_dir() / DB_FILENAME), read_only=True) as con:
        return con.execute(sql, params or []).df()


@st.cache_data
def manifest() -> dict[str, Any]:
    path = processed_dir() / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def top250() -> pd.DataFrame:
    return query(f"""
        with {GENRES_CTE}, {DIRECTORS_CTE}
        select m.rank, m.tmdb_id, m.imdb_id, m.title, m.year,
               -- Table thumbnails: TMDB's 154px size instead of 500px (about 10x lighter).
               replace(m.poster_url, '/w500/', '/w154/') as poster_thumb_url,
               m.imdb_rating, m.imdb_votes, m.weighted_rating, m.tmdb_rating, m.metascore,
               m.rotten_tomatoes, m.runtime_minutes, m.original_language, m.oscar_wins,
               coalesce(g.genres, []) as genres, coalesce(d.directors, []) as directors
        from movies m
        left join genres g using (tmdb_id)
        left join directors d using (tmdb_id)
        where m.rank is not null
        order by m.rank
    """)


@st.cache_data
def movie(tmdb_id: int) -> dict[str, Any]:
    df = query(
        f"""
        with {GENRES_CTE}, {DIRECTORS_CTE}
        select m.*, coalesce(g.genres, []) as genres, coalesce(d.directors, []) as directors
        from movies m left join genres g using (tmdb_id) left join directors d using (tmdb_id)
        where m.tmdb_id = ?
        """,
        [tmdb_id],
    )
    if df.empty:
        return {}
    # DuckDB returns LIST columns as NumPy arrays; plain lists are simpler for the UI.
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else (None if pd.isna(v) else v)
        for k, v in df.iloc[0].items()
    }


def _none_for_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/NA with None so ``if value:`` checks behave in the UI."""
    return df.astype(object).where(df.notna(), None)


@st.cache_data
def cast(tmdb_id: int, limit: int = 12) -> pd.DataFrame:
    return _none_for_missing(
        query(
            """
        select p.name, c.character, p.profile_url
        from credits c join people p using (person_id)
        where c.tmdb_id = ? and c.credit_type = 'cast'
        order by c.billing limit ?
        """,
            [tmdb_id, limit],
        )
    )


KEY_JOBS = [
    "Director", "Screenplay", "Writer", "Story", "Novel",
    "Original Music Composer", "Director of Photography", "Editor",
]  # fmt: skip


@st.cache_data
def key_crew(tmdb_id: int) -> pd.DataFrame:
    """Key crew grouped by job, in the conventional credits order."""
    return query(
        """
        select c.job, list(p.name order by p.name) as names
        from credits c join people p using (person_id)
        where c.tmdb_id = ? and list_contains(?, c.job)
        group by c.job
        order by list_position(?, c.job)
        """,
        [tmdb_id, KEY_JOBS, KEY_JOBS],
    )


@st.cache_data
def keywords(tmdb_id: int) -> list[str]:
    df = query("select keyword from movie_keywords where tmdb_id = ? order by keyword", [tmdb_id])
    return df["keyword"].tolist()


@st.cache_data
def reviews(tmdb_id: int) -> pd.DataFrame:
    return _none_for_missing(
        query(
            """
            select author, rating, content, created_at, url
            from reviews where tmdb_id = ? order by created_at desc
            """,
            [tmdb_id],
        )
    )


# --- Insights -------------------------------------------------------------------------


@st.cache_data
def by_decade() -> pd.DataFrame:
    return query("""
        select (year // 10) * 10 as decade, count(*) as films,
               round(avg(imdb_rating), 2) as avg_imdb_rating
        from movies where rank is not null and year is not null
        group by decade order by decade
    """)


@st.cache_data
def genre_stats() -> pd.DataFrame:
    return query("""
        select g.genre, count(*) as films, round(avg(m.imdb_rating), 2) as avg_imdb_rating
        from movies m join movie_genres g using (tmdb_id)
        where m.rank is not null
        group by g.genre order by films desc
    """)


@st.cache_data
def top_people(job: str, max_billing: int | None = None, limit: int = 15) -> pd.DataFrame:
    """People with the most Top 250 films in a given job (``Actor`` uses billing order)."""
    billing_filter = "and c.billing < ?" if max_billing is not None else ""
    params: list[Any] = [job] + ([max_billing] if max_billing is not None else []) + [limit]
    return query(
        f"""
        select p.name, count(distinct c.tmdb_id) as films,
               round(avg(m.imdb_rating), 2) as avg_imdb_rating,
               string_agg(distinct m.title, '; ' order by m.title) as titles
        from credits c join people p using (person_id) join movies m using (tmdb_id)
        where m.rank is not null and c.job = ? {billing_filter}
        group by p.name
        order by films desc, avg_imdb_rating desc
        limit ?
        """,
        params,
    )


@st.cache_data
def ratings_comparison() -> pd.DataFrame:
    return query("""
        select title, year, imdb_rating, tmdb_rating, metascore, rotten_tomatoes, imdb_votes
        from movies where rank is not null
    """)


@st.cache_data
def money() -> pd.DataFrame:
    return query("""
        select title, year, budget_usd, revenue_usd,
               round(revenue_usd / budget_usd, 1) as revenue_multiple
        from movies
        where rank is not null and budget_usd >= 100000 and revenue_usd > 0
    """)
