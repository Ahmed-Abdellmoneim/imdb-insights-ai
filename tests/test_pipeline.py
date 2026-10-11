"""End-to-end pipeline tests against fake TMDB and OMDb APIs."""

import asyncio
import os
import time
from datetime import timedelta

import duckdb
import httpx

from imdb_insights.ingest import pipeline
from imdb_insights.ingest.omdb import OmdbClient
from imdb_insights.ingest.tmdb import TmdbClient

N_MOVIES = 30


def fake_tmdb(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/discover/movie"):
        ids = [{"id": i} for i in range(1, N_MOVIES + 1)]
        return httpx.Response(200, json={"total_pages": 1, "results": ids})
    tmdb_id = int(request.url.path.rsplit("/", 1)[-1])
    return httpx.Response(
        200,
        json={
            "id": tmdb_id,
            "title": f"Movie {tmdb_id}",
            "imdb_id": f"tt{tmdb_id:07d}",
            "release_date": "2000-01-01",
            "vote_average": 8.0,
            "vote_count": 1000,
            "genres": [{"id": 18, "name": "Drama"}],
            "credits": {
                "cast": [{"id": 1, "name": "Star", "character": "Hero", "order": 0}],
                "crew": [{"id": 2, "name": "Boss", "department": "Directing", "job": "Director"}],
            },
            "reviews": {
                "total_results": 1,
                "results": [
                    {"id": f"r{tmdb_id}", "content": "Good.", "author_details": {"rating": 8.0}}
                ],
            },
        },
    )


def omdb_handler(quota: int):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) > quota:
            return httpx.Response(
                401, json={"Response": "False", "Error": "Request limit reached!"}
            )
        imdb_id = request.url.params["i"]
        n = int(imdb_id[2:])
        return httpx.Response(
            200,
            json={
                "Response": "True",
                "imdbID": imdb_id,
                "Title": f"Movie {n}",
                "imdbRating": f"{7 + n / 10:.1f}" if n <= 25 else "8.0",
                "imdbVotes": "100,000",
            },
        )

    return handler, calls


def run_collect(tmp_path, omdb_quota: int):
    handler, calls = omdb_handler(omdb_quota)

    async def run():
        async with (
            TmdbClient("t", tmp_path, transport=httpx.MockTransport(fake_tmdb)) as tmdb,
            OmdbClient("k", tmp_path, transport=httpx.MockTransport(handler)) as omdb,
        ):
            tmdb.http.min_interval = omdb.http.min_interval = 0
            return await pipeline.collect(tmdb, omdb, min_rating=7.0, min_votes=1, snapshot="today")

    return asyncio.run(run()), calls


def test_full_run_builds_ranked_tables(tmp_path):
    col, _ = run_collect(tmp_path, omdb_quota=1000)
    tables = pipeline.build_tables(col)
    report = pipeline.quality_report(col, tables)

    movies = tables["movies"]
    assert len(movies) == N_MOVIES
    assert (movies["omdb_status"] == "ok").all()
    assert movies.iloc[0]["title"] == "Movie 25"  # highest IMDb rating ranks first
    assert movies["rank"].iloc[0] == 1
    assert report["complete"] is True
    assert report["ranked"] == N_MOVIES
    assert len(tables["credits"]) == 2 * N_MOVIES
    assert set(tables["credits"]["credit_type"]) == {"cast", "crew"}
    assert len(tables["people"]) == 2

    db = pipeline.write_outputs(tables, report, tmp_path / "out", started=pipeline.datetime.now())
    with duckdb.connect(str(db), read_only=True) as con:
        assert con.sql("select count(*) from movies").fetchone()[0] == N_MOVIES
        assert con.sql("select count(*) from reviews").fetchone()[0] == N_MOVIES
    assert (tmp_path / "out" / "movies.parquet").exists()


def test_expired_ratings_fall_back_to_cache_when_quota_runs_out(tmp_path):
    run_collect(tmp_path, omdb_quota=1000)  # day 1: everything cached
    old = time.time() - 60 * 86400
    for f in (tmp_path / "omdb").glob("*.json"):
        os.utime(f, (old, old))

    handler, calls = omdb_handler(quota=5)

    async def run():
        async with (
            TmdbClient("t", tmp_path, transport=httpx.MockTransport(fake_tmdb)) as tmdb,
            OmdbClient(
                "k", tmp_path, max_age=timedelta(days=14),
                transport=httpx.MockTransport(handler),
            ) as omdb,
        ):  # fmt: skip
            tmdb.http.min_interval = omdb.http.min_interval = 0
            return await pipeline.collect(tmdb, omdb, min_rating=7, min_votes=1, snapshot="d2")

    col = asyncio.run(run())
    assert len(col.omdb) == N_MOVIES  # every movie still has a rating
    assert not col.omdb_pending
    assert len(col.stale) == N_MOVIES - 5
    report = pipeline.quality_report(col, pipeline.build_tables(col))
    assert report["complete"] is True
    assert report["stale_entries"] == N_MOVIES - 5


def test_movies_without_any_rating_are_fetched_first(tmp_path):
    run_collect(tmp_path, omdb_quota=10)  # day 1: 10 cached, 20 never fetched
    old = time.time() - 60 * 86400
    for f in (tmp_path / "omdb").glob("*.json"):
        os.utime(f, (old, old))  # the cached 10 are also due for a refresh

    handler, calls = omdb_handler(quota=20)

    async def run():
        async with (
            TmdbClient("t", tmp_path, transport=httpx.MockTransport(fake_tmdb)) as tmdb,
            OmdbClient(
                "k", tmp_path, max_age=timedelta(days=14),
                transport=httpx.MockTransport(handler),
            ) as omdb,
        ):  # fmt: skip
            tmdb.http.min_interval = omdb.http.min_interval = 0
            return await pipeline.collect(tmdb, omdb, min_rating=7, min_votes=1, snapshot="d2")

    col = asyncio.run(run())
    # The 20-request quota went to the 20 movies that had nothing; the rest kept old data.
    assert len(col.omdb) == N_MOVIES
    assert not col.omdb_pending
    assert len(col.stale) == 10


def test_fresh_ratings_are_not_reported_stale_when_quota_is_exhausted(tmp_path):
    run_collect(tmp_path, omdb_quota=10)  # 10 ratings cached today, 20 never fetched
    handler, calls = omdb_handler(quota=0)  # quota already gone

    async def run():
        async with (
            TmdbClient("t", tmp_path, transport=httpx.MockTransport(fake_tmdb)) as tmdb,
            OmdbClient(
                "k", tmp_path, max_age=timedelta(days=14),
                transport=httpx.MockTransport(handler),
            ) as omdb,
        ):  # fmt: skip
            tmdb.http.min_interval = omdb.http.min_interval = 0
            return await pipeline.collect(tmdb, omdb, min_rating=7, min_votes=1, snapshot="d2")

    col = asyncio.run(run())
    assert len(col.omdb) == 10  # the fresh ones, served from cache
    assert col.stale == []  # ...and not mislabelled as stale
    assert len(col.omdb_pending) == N_MOVIES - 10


def test_rewrite_swaps_in_a_fresh_database(tmp_path):
    col, _ = run_collect(tmp_path, omdb_quota=1000)
    tables = pipeline.build_tables(col)
    report = pipeline.quality_report(col, tables)
    out = tmp_path / "out"
    pipeline.write_outputs(tables, report, out, started=pipeline.datetime.now())

    fewer = {name: df.head(5) if name == "movies" else df for name, df in tables.items()}
    db = pipeline.write_outputs(fewer, report, out, started=pipeline.datetime.now())
    with duckdb.connect(str(db), read_only=True) as con:
        assert con.sql("select count(*) from movies").fetchone()[0] == 5
    assert not list(out.glob("*.building"))


def test_empty_tables_still_exist_with_their_schema(tmp_path):
    col, _ = run_collect(tmp_path, omdb_quota=1000)
    tables = pipeline.build_tables(col)
    tables["reviews"] = pipeline.pd.DataFrame()
    db = pipeline.write_outputs(
        tables, pipeline.quality_report(col, tables), tmp_path / "out", pipeline.datetime.now()
    )
    with duckdb.connect(str(db), read_only=True) as con:
        assert con.sql("select count(*) from reviews").fetchone()[0] == 0
        columns = [c[0] for c in con.sql("describe reviews").fetchall()]
    assert {"review_id", "tmdb_id", "rating", "content"} <= set(columns)


def test_quota_stop_is_graceful_and_resumable(tmp_path):
    col, calls = run_collect(tmp_path, omdb_quota=10)
    tables = pipeline.build_tables(col)
    report = pipeline.quality_report(col, tables)

    assert col.omdb_stopped and "Request limit reached" in col.omdb_stopped
    assert len(col.omdb) == 10
    assert len(col.omdb_pending) == N_MOVIES - 10
    # Only the requests already in flight when the quota ran out may be wasted.
    assert len(calls) <= 10 + pipeline.OMDB_WORKERS
    assert report["complete"] is False
    assert (tables["movies"]["omdb_status"] == "pending").sum() == N_MOVIES - 10

    # Next day: the quota has reset, and only the pending movies are requested.
    col2, calls2 = run_collect(tmp_path, omdb_quota=1000)
    assert len(col2.omdb) == N_MOVIES
    assert len(calls2) == N_MOVIES - 10
