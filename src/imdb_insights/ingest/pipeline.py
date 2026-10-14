"""Collect candidates from TMDB, enrich with IMDb ratings from OMDb, rank, and write tables.

Every API response is cached under ``<data_dir>/raw``, so an interrupted run (e.g. the OMDb
daily quota running out) resumes where it stopped and only fetches what is missing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .http import ApiError
from .omdb import OmdbClient, OmdbError, OmdbMovie
from .ranking import rank_top
from .tmdb import TmdbClient, TmdbMovie, image_url

log = logging.getLogger(__name__)

DB_FILENAME = "imdb_insights.duckdb"
# How far above the candidate floor (in TMDB rating points) the weakest ranked movie must be
# for us to trust that no Top 250 movie was left out of the pool.
RATING_MARGIN = 0.3
# Likewise for votes: the ranked movie with the fewest TMDB votes should have at least this
# multiple of the candidate vote floor.
VOTE_MARGIN = 2.0

OMDB_COLUMNS = [
    "imdb_rating", "imdb_votes", "metascore", "rotten_tomatoes", "box_office_usd", "rated",
    "awards", "oscar_wins", "oscar_nominations", "award_wins", "award_nominations",
]  # fmt: skip
# Nullable integer columns; without this pandas turns them into floats when values are missing.
INT_COLUMNS = [
    "year", "runtime_minutes", "budget_usd", "revenue_usd", "tmdb_votes", "imdb_votes",
    "metascore", "rotten_tomatoes", "box_office_usd", "oscar_wins", "oscar_nominations",
    "award_wins", "award_nominations",
]  # fmt: skip
# Text columns, typed explicitly: a column that happens to be all-NULL would otherwise be
# stored as INTEGER and break string functions downstream.
STRING_COLUMNS = [
    "imdb_id", "title", "original_title", "original_language", "overview", "tagline",
    "collection", "poster_url", "backdrop_url", "rated", "awards", "omdb_status",
]  # fmt: skip
OMDB_WORKERS = 4
# Tables kept out of the Parquet export (embeddings are large and only the app needs them).
NO_PARQUET_EXPORT = {"review_chunks"}

# Explicit schemas for the tables that can legitimately be empty (e.g. no reviews yet), so
# the warehouse always has every table. `movies` is never empty and keeps inferred types.
SCHEMAS = {
    "movie_genres": "tmdb_id BIGINT, genre VARCHAR, PRIMARY KEY (tmdb_id, genre)",
    "movie_keywords": "tmdb_id BIGINT, keyword VARCHAR",
    "people": "person_id BIGINT PRIMARY KEY, name VARCHAR, gender INTEGER, profile_url VARCHAR",
    "credits": (
        "tmdb_id BIGINT, person_id BIGINT, credit_type VARCHAR, department VARCHAR, "
        "job VARCHAR, character VARCHAR, billing INTEGER"
    ),
    "reviews": (
        "review_id VARCHAR PRIMARY KEY, tmdb_id BIGINT, source VARCHAR, author VARCHAR, "
        "rating DOUBLE, content VARCHAR, created_at TIMESTAMPTZ, url VARCHAR"
    ),
    # Review passages for the assistant's search (built by imdb_insights.assistant.index;
    # empty, but present, when the search index is skipped).
    "review_chunks": (
        "chunk_id VARCHAR PRIMARY KEY, review_id VARCHAR, tmdb_id BIGINT, chunk_no INTEGER, "
        "text VARCHAR, embedding FLOAT[384]"
    ),
}


@dataclass
class Collection:
    candidate_ids: list[int]
    min_rating: float
    min_votes: int
    # Human-readable description of each candidate query, for the manifest and the app.
    pools: list[str] = field(default_factory=list)
    movies: list[TmdbMovie] = field(default_factory=list)
    omdb: dict[str, OmdbMovie | None] = field(default_factory=dict)
    omdb_pending: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    omdb_stopped: str | None = None
    # Entries served from an expired cache because a refresh was not possible this run.
    stale: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LowVoteTier:
    """Extra candidates with few TMDB votes (``min_votes`` up to the main floor).

    Some films with hundreds of thousands of IMDb votes have only a few hundred on TMDB
    (e.g. Indian cinema), so a single TMDB vote floor would silently miss them.
    """

    min_rating: float
    min_votes: int


async def collect(
    tmdb: TmdbClient,
    omdb: OmdbClient,
    *,
    min_rating: float,
    min_votes: int,
    snapshot: str,
    low_votes: LowVoteTier | None = None,
) -> Collection:
    ids = await tmdb.discover_top_rated(
        min_rating=min_rating, min_votes=min_votes, snapshot=snapshot
    )
    floor_rating, floor_votes = min_rating, min_votes
    pools = [f"rated {min_rating}+ with {min_votes:,}+ TMDB votes"]
    if low_votes and low_votes.min_votes < min_votes:
        ids += await tmdb.discover_top_rated(
            min_rating=low_votes.min_rating,
            min_votes=low_votes.min_votes,
            max_votes=min_votes - 1,
            snapshot=snapshot,
        )
        ids = list(dict.fromkeys(ids))
        floor_rating = min(min_rating, low_votes.min_rating)
        floor_votes = low_votes.min_votes
        pools.append(
            f"rated {low_votes.min_rating}+ with {low_votes.min_votes:,}–{min_votes - 1:,} "
            "TMDB votes"
        )
    log.info("Discovered %d candidate movies", len(ids))
    result = Collection(
        candidate_ids=ids, min_rating=floor_rating, min_votes=floor_votes, pools=pools
    )

    async def one_movie(tmdb_id: int) -> TmdbMovie | None:
        try:
            return await tmdb.movie(tmdb_id)
        except (ApiError, ValueError) as exc:
            if stale := tmdb.cached_movie(tmdb_id):
                result.stale.append(f"tmdb:{tmdb_id}")
                return stale
            result.failures[f"tmdb:{tmdb_id}"] = str(exc)
            return None

    result.movies = [m for m in await asyncio.gather(*map(one_movie, ids)) if m]
    log.info("Fetched TMDB details for %d movies", len(result.movies))

    # Ratings that are still fresh in the cache need no request at all.
    imdb_ids = list(dict.fromkeys(m.imdb_id for m in result.movies if m.imdb_id))
    queue = []
    for imdb_id in imdb_ids:
        if omdb.http.has_fresh(f"omdb/{imdb_id}"):
            result.omdb[imdb_id] = await omdb.movie(imdb_id)
        else:
            queue.append(imdb_id)

    # A few workers share one queue, so once OMDb refuses (quota or key problems affect every
    # later request too) at most `OMDB_WORKERS` requests are already in flight. A short quota
    # goes where it matters most: movies with no rating at all first, best TMDB-rated first
    # (the likeliest to enter the ranking), then refreshes of expired ratings.
    tmdb_rating = {m.imdb_id: m.tmdb_rating or 0 for m in result.movies}
    queue.sort(
        key=lambda imdb_id: (
            omdb.http.read_cache(f"omdb/{imdb_id}") is not None,
            -tmdb_rating.get(imdb_id, 0),
        )
    )

    async def rating_worker() -> None:
        while queue and not result.omdb_stopped:
            imdb_id = queue.pop(0)
            try:
                result.omdb[imdb_id] = await omdb.movie(imdb_id)
            except OmdbError as exc:
                result.omdb_stopped = result.omdb_stopped or str(exc)
                result.omdb_pending.append(imdb_id)
            except (ApiError, ValueError) as exc:
                result.failures[f"omdb:{imdb_id}"] = str(exc)

    await asyncio.gather(*(rating_worker() for _ in range(OMDB_WORKERS)))
    result.omdb_pending += queue

    # Ratings we could not refresh this run fall back to the last known value.
    still_pending = []
    for imdb_id in result.omdb_pending:
        if stale := omdb.cached_movie(imdb_id):
            result.omdb[imdb_id] = stale
            result.stale.append(f"omdb:{imdb_id}")
        else:
            still_pending.append(imdb_id)
    result.omdb_pending = still_pending
    log.info("Fetched IMDb ratings for %d movies", len(result.omdb))
    if result.omdb_stopped:
        log.warning("OMDb stopped (%s); %d pending", result.omdb_stopped, len(result.omdb_pending))
    return result


def build_tables(col: Collection) -> dict[str, pd.DataFrame]:
    position = {tmdb_id: i for i, tmdb_id in enumerate(col.candidate_ids, start=1)}
    movies, genres, keywords, people, credits, reviews = [], [], [], {}, [], []

    for m in col.movies:
        row: dict[str, Any] = m.model_dump(
            exclude={
                "genres",
                "keywords",
                "cast",
                "crew",
                "reviews",
                "poster_path",
                "backdrop_path",
            }  # fmt: skip
        )
        row["year"] = m.release_date.year if m.release_date else None
        row["poster_url"] = image_url(m.poster_path)
        row["backdrop_url"] = image_url(m.backdrop_path, "w1280")
        row["candidate_position"] = position.get(m.tmdb_id)

        rating = col.omdb.get(m.imdb_id) if m.imdb_id else None
        row |= {c: getattr(rating, c) if rating else None for c in OMDB_COLUMNS}
        row["omdb_status"] = (
            "ok" if rating
            else "no_imdb_id" if not m.imdb_id
            else "pending" if m.imdb_id in col.omdb_pending
            else "not_found" if m.imdb_id in col.omdb
            else "failed"
        )  # fmt: skip
        movies.append(row)

        genres += [{"tmdb_id": m.tmdb_id, "genre": g} for g in m.genres]
        keywords += [{"tmdb_id": m.tmdb_id, "keyword": k} for k in m.keywords]
        for person in [*m.cast, *m.crew]:
            people[person.person_id] = {
                "person_id": person.person_id,
                "name": person.name,
                "gender": person.gender,
                "profile_url": image_url(person.profile_path, "w185"),
            }
        credits += [
            {"tmdb_id": m.tmdb_id, "person_id": c.person_id, "credit_type": "cast",
             "department": "Acting", "job": "Actor", "character": c.character,
             "billing": c.billing}
            for c in m.cast
        ]  # fmt: skip
        credits += [
            {"tmdb_id": m.tmdb_id, "person_id": c.person_id, "credit_type": "crew",
             "department": c.department, "job": c.job, "character": None, "billing": None}
            for c in m.crew
        ]  # fmt: skip
        reviews += [{"tmdb_id": m.tmdb_id, "source": "tmdb"} | r.model_dump() for r in m.reviews]

    movies_df = pd.DataFrame(movies)
    movies_df = movies_df.astype(
        dict.fromkeys(INT_COLUMNS, "Int64") | dict.fromkeys(STRING_COLUMNS, "string")
    )
    movies_df = rank_top(movies_df).sort_values("rank", na_position="last")

    return {
        "movies": movies_df.reset_index(drop=True),
        "movie_genres": pd.DataFrame(genres),
        "movie_keywords": pd.DataFrame(keywords),
        "people": pd.DataFrame(list(people.values())),
        "credits": pd.DataFrame(credits).astype({"billing": "Int64"}),
        "reviews": pd.DataFrame(reviews),
    }


def quality_report(col: Collection, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Checks that tell us whether the ranking can be trusted."""
    movies = tables["movies"]
    ranked = movies[movies["rank"].notna()]
    lowest = float(ranked["tmdb_rating"].min()) if len(ranked) else None
    fewest_votes = int(ranked["tmdb_votes"].min()) if len(ranked) else None
    warnings = []
    if len(ranked) < 250:
        warnings.append(f"only {len(ranked)} movies are eligible for the ranking")
    if col.omdb_pending:
        warnings.append(
            f"{len(col.omdb_pending)} movies still need IMDb ratings, so the ranking is "
            "incomplete; re-run after the OMDb quota resets (cached data is reused)"
        )
    # Candidates are the TMDB movies above both floors. If a ranked movie sits close to either
    # floor, movies just below it could also belong in the Top 250. (TMDB ratings run up to
    # ~0.6 below IMDb's, and films popular on IMDb can have few TMDB votes.)
    if lowest is not None and lowest - col.min_rating < RATING_MARGIN:
        warnings.append(
            f"a ranked movie has TMDB rating {lowest:.2f}, within {RATING_MARGIN} of the "
            f"candidate floor {col.min_rating}; lower --min-rating to be sure none are missed"
        )
    if fewest_votes is not None and fewest_votes < VOTE_MARGIN * col.min_votes:
        warnings.append(
            f"a ranked movie has only {fewest_votes} TMDB votes, under {VOTE_MARGIN}x the "
            f"candidate floor {col.min_votes}; lower --min-votes to be sure none are missed"
        )
    return {
        "candidates": len(col.candidate_ids),
        "ranked": len(ranked),
        "complete": not col.omdb_pending and not col.failures,
        "candidate_pools": col.pools,
        "candidate_min_tmdb_rating": col.min_rating,
        "candidate_min_tmdb_votes": col.min_votes,
        "lowest_tmdb_rating_in_ranking": lowest,
        "fewest_tmdb_votes_in_ranking": fewest_votes,
        "omdb_status": movies["omdb_status"].value_counts().to_dict(),
        "omdb_stopped": col.omdb_stopped,
        "stale_entries": len(col.stale),
        "failures": col.failures,
        "warnings": warnings,
    }


def write_outputs(
    tables: dict[str, pd.DataFrame], report: dict[str, Any], out_dir: Path,
    started: datetime,
) -> Path:  # fmt: skip
    """Write the tables to a DuckDB database, plus a Parquet copy of each, and a manifest.

    The database is built in a temporary file and then swapped in with one rename, so
    readers never see a half-written warehouse.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / DB_FILENAME
    tmp_path = db_path.with_suffix(".building")
    tmp_path.unlink(missing_ok=True)
    with duckdb.connect(str(tmp_path)) as con:
        for name, df in tables.items():
            if name in SCHEMAS:
                # Fixed schema: the table exists (empty if need be) with stable column types.
                con.execute(f"CREATE OR REPLACE TABLE {name} ({SCHEMAS[name]})")
                if not df.empty:
                    con.register("frame", df)
                    con.execute(f"INSERT INTO {name} BY NAME SELECT * FROM frame")
                    con.unregister("frame")
            else:
                con.register("frame", df)
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame")
                con.unregister("frame")
            if name in NO_PARQUET_EXPORT:
                continue
            parquet = (out_dir / f"{name}.parquet").as_posix().replace("'", "''")
            con.execute(f"COPY {name} TO '{parquet}' (FORMAT parquet)")
        if "review_chunks" in tables:
            # BM25 keyword index for the assistant's hybrid search, built before the swap so
            # readers always get the index together with the passages it covers.
            con.execute("INSTALL fts; LOAD fts")
            con.execute("PRAGMA create_fts_index('review_chunks', 'chunk_id', 'text', overwrite=1)")
    try:
        os.replace(tmp_path, db_path)
    except PermissionError as exc:
        raise RuntimeError(
            f"{db_path} is locked by another process (an open database client?). "
            f"The new database was kept at {tmp_path}; close the other process and re-run."
        ) from exc
    manifest = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "rows": {name: len(df) for name, df in tables.items()},
        "quality": report,
        "attribution": "This product uses the TMDB API but is not endorsed or certified by "
        "TMDB. IMDb ratings via the OMDb API.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return db_path
