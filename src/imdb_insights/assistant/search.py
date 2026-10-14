"""Hybrid search over review passages: BM25 keywords + vector similarity, fused with RRF.

Keyword search finds exact names and phrases ("Hans Zimmer", "plot hole"); vector search
finds paraphrases ("the story drags" for "pacing is slow"). Reciprocal rank fusion merges
the two rankings without having to calibrate their very different score scales.
All SQL here is fixed and parameterized: no model-written text is ever executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .embedder import DIMENSIONS, Embedder

RRF_K = 60  # standard constant from Cormack et al. (2009)
CANDIDATES = 50  # taken from each ranking before fusion


@dataclass
class Movie:
    tmdb_id: int
    title: str
    year: int | None


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("LOAD fts")
    return con


def find_movie(con: duckdb.DuckDBPyConnection, title: str) -> Movie | None:
    """Best match for a title: exact (case-insensitive) first, then contains; most-voted wins."""
    row = con.execute(
        """
        select tmdb_id, title, year from movies
        where title ilike ? or original_title ilike ? or title ilike '%' || ? || '%'
        order by (lower(title) = lower(?)) desc, tmdb_votes desc nulls last
        limit 1
        """,
        [title, title, title, title],
    ).fetchone()
    return Movie(int(row[0]), row[1], row[2]) if row else None


def _ranked(con, sql: str, params: list) -> list[str]:
    return [r[0] for r in con.execute(sql, params).fetchall()]


def hybrid_search(
    con: duckdb.DuckDBPyConnection,
    embedder: Embedder,
    query: str,
    *,
    k: int = 8,
    tmdb_id: int | None = None,
) -> pd.DataFrame:
    """Top ``k`` passages with their review and movie metadata, best first."""
    where, params = ("and tmdb_id = ?", [tmdb_id]) if tmdb_id is not None else ("", [])
    keyword = _ranked(
        con,
        f"""
        select chunk_id from (
            select chunk_id, tmdb_id, fts_main_review_chunks.match_bm25(chunk_id, ?) as score
            from review_chunks
        ) where score is not null {where}
        order by score desc limit {CANDIDATES}
        """,
        [query, *params],
    )
    vector = _ranked(
        con,
        f"""
        select chunk_id from review_chunks
        where true {where}
        order by array_cosine_similarity(embedding, ?::FLOAT[{DIMENSIONS}]) desc
        limit {CANDIDATES}
        """,
        [*params, embedder.embed_query(query).tolist()],
    )
    scores: dict[str, float] = {}
    for ranking in (keyword, vector):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    if not top:
        return pd.DataFrame()
    meta = con.execute(
        """
        select c.chunk_id, c.text, r.review_id, r.author, r.rating, r.url,
               m.tmdb_id, m.title, m.year
        from review_chunks c
        join reviews r using (review_id)
        join movies m on m.tmdb_id = c.tmdb_id
        where c.chunk_id in (select unnest(?))
        """,
        [top],
    ).df()
    meta["score"] = meta["chunk_id"].map(scores)
    meta["in_keyword"] = meta["chunk_id"].isin(keyword)
    meta["in_vector"] = meta["chunk_id"].isin(vector)
    return meta.sort_values("score", ascending=False).reset_index(drop=True)
