"""The MCP server, exercised through a real MCP client connected in-process."""

import asyncio

import numpy as np
import pytest
from mcp import Client
from test_assistant_search import FakeEmbedder
from test_pipeline import run_collect

from imdb_insights.assistant import index, sql
from imdb_insights.ingest import pipeline
from imdb_insights.mcp_server import create_server

TOOLS = {"top_movies", "movie_details", "query_database", "search_reviews"}


class FakePassageEmbedder(FakeEmbedder):
    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mcp")
    col, _ = run_collect(tmp / "raw", omdb_quota=1000)
    tables = pipeline.build_tables(col)
    tables["review_chunks"] = index.build_review_chunks(
        tables["reviews"], FakePassageEmbedder(), index.EmbeddingCache(tmp / "cache.parquet")
    )
    report = pipeline.quality_report(col, tables)
    db = pipeline.write_outputs(tables, report, tmp / "processed", pipeline.datetime.now())
    return create_server(db, embedder_factory=FakePassageEmbedder)


def run(server, fn):
    async def go():
        async with Client(server) as client:
            return await fn(client)

    return asyncio.run(go())


def call(server, tool, args):
    return run(server, lambda c: c.call_tool(tool, args))


def test_tools_are_discoverable_read_only_and_typed(server):
    tools = run(server, lambda c: c.list_tools()).tools
    assert {t.name for t in tools} == TOOLS
    for tool in tools:
        assert tool.annotations.read_only_hint and not tool.annotations.destructive_hint
        assert tool.description and tool.output_schema  # structured output declared


def test_top_movies_is_structured_and_filterable(server):
    result = call(server, "top_movies", {"limit": 3})
    movies = result.structured_content["movies"]
    assert [m["rank"] for m in movies] == [1, 2, 3]
    assert movies[0]["directors"] == ["Boss"] and movies[0]["genres"] == ["Drama"]
    assert call(server, "top_movies", {"genre": "comedy"}).structured_content == {"movies": []}
    assert call(server, "top_movies", {"decade": 1990}).structured_content == {"movies": []}
    assert len(call(server, "top_movies", {"decade": 2000}).structured_content["movies"]) == 25


def test_movie_details(server):
    details = call(server, "movie_details", {"title": "movie 7"}).structured_content
    assert details["title"] == "Movie 7" and details["crew"] == {"Director": ["Boss"]}
    assert details["cast"] == [{"name": "Star", "character": "Hero"}]
    missing = call(server, "movie_details", {"title": "Nope"}).structured_content
    assert "No movie matching" in missing["error"]


def test_query_database_runs_selects_and_refuses_writes(server):
    ok = call(server, "query_database", {"query": "select count(*) as n from movies"})
    assert ok.structured_content == {"columns": ["n"], "rows": [{"n": 30}], "truncated": False}
    refused = call(server, "query_database", {"query": "drop table movies"})
    assert refused.is_error and "Only SELECT" in refused.content[0].text


def test_search_reviews_returns_cited_passages(server):
    result = call(server, "search_reviews", {"query": "good", "movie": "Movie 3", "limit": 2})
    passages = result.structured_content["passages"]
    assert passages and {p["title"] for p in passages} == {"Movie 3"}
    assert {"author", "rating", "url", "text"} <= set(passages[0])


def test_schema_resource(server):
    result = run(server, lambda c: c.read_resource("imdb-insights://schema"))
    assert result.contents[0].text == sql.SCHEMA_GUIDE


def test_missing_warehouse_gives_instructions(tmp_path):
    lonely = create_server(tmp_path / "nothing.duckdb", embedder_factory=FakePassageEmbedder)
    result = call(lonely, "top_movies", {})
    assert result.is_error and "publish pull" in result.content[0].text
