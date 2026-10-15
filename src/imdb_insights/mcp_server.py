"""MCP server: the movie warehouse as tools for any MCP client (Claude Desktop, IDEs, agents).

The client's own model does the reasoning, so this server needs no LLM key. All tools are
read-only and declared as such; SQL goes through the same guardrails as the assistant.

Run over stdio:  ``uv run python -m imdb_insights.mcp_server``
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .assistant import search, sql
from .config import get_settings
from .ingest.pipeline import DB_FILENAME

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
INSTRUCTIONS = (
    "Movie data for an IMDb-style Top 250 and thousands of other highly rated films: IMDb, "
    "TMDB, Metacritic and Rotten Tomatoes ratings, credits, keywords, budgets, awards and "
    "user reviews. Use top_movies and movie_details for common questions, search_reviews for "
    "what reviewers say, and query_database (read the imdb-insights://schema resource first) "
    "for anything else."
)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-friendly rows: NaN/NA become None, NumPy scalars and arrays become Python."""
    clean = df.astype(object).where(df.notna(), None)
    return [
        {k: v.tolist() if hasattr(v, "tolist") else v for k, v in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def create_server(
    db_path: Path | None = None, embedder_factory: Callable | None = None
) -> MCPServer:
    db_path = db_path or get_settings().data_dir / "processed" / DB_FILENAME

    @cache
    def embedder():
        if embedder_factory:
            return embedder_factory()
        from .assistant.embedder import Embedder

        return Embedder(get_settings().data_dir / "models" / "bge-small-en-v1.5")

    def require_warehouse() -> None:
        if not db_path.exists():
            raise ToolError(
                f"No warehouse at {db_path}. Build it with `python -m imdb_insights.ingest` "
                "or download it with `python -m imdb_insights.publish pull`."
            )

    def fetch(statement: str, params: list | None = None) -> pd.DataFrame:
        require_warehouse()
        with duckdb.connect(str(db_path), read_only=True) as con:
            return con.execute(statement, params or []).df()

    server = MCPServer("imdb-insights", instructions=INSTRUCTIONS, version="2.0.0")

    @server.resource("imdb-insights://schema", name="schema", mime_type="text/plain")
    def schema() -> str:
        """Tables and columns of the movie database, with SQL conventions."""
        return sql.SCHEMA_GUIDE

    @server.tool(annotations=READ_ONLY)
    def top_movies(
        limit: int = 25, genre: str | None = None, decade: int | None = None
    ) -> dict[str, Any]:
        """The IMDb-style Top 250 in rank order, optionally filtered by genre (e.g. "Drama")
        and decade (e.g. 1990 for the 1990s). Returns rank, title, year, ratings and directors."""
        limit = max(1, min(int(limit), 250))
        movies = _records(
            fetch(
                """
            select m.rank, m.title, m.year, m.imdb_rating, m.imdb_votes, m.metascore,
                   (select list(distinct g.genre order by g.genre) from movie_genres g
                    where g.tmdb_id = m.tmdb_id) as genres,
                   (select list(p.name order by p.name) from credits c join people p
                    using (person_id) where c.tmdb_id = m.tmdb_id and c.job = 'Director')
                    as directors
            from movies m
            where m.rank is not null
              and (?::varchar is null or exists (select 1 from movie_genres g
                   where g.tmdb_id = m.tmdb_id and g.genre ilike ?))
              and (?::int is null or m.year // 10 * 10 = ?)
            order by m.rank limit ?
            """,
                [genre, genre, decade, decade, limit],
            )
        )
        return {"movies": movies}

    @server.tool(annotations=READ_ONLY)
    def movie_details(title: str) -> dict[str, Any]:
        """Everything about one movie, found by title: ratings from four sources, rank,
        runtime, budget and revenue, awards, directors and writers, top cast, keywords."""
        require_warehouse()
        with duckdb.connect(str(db_path), read_only=True) as con:
            movie = search.find_movie(con, title)
        if movie is None:
            return {"error": f"No movie matching {title!r}."}
        tmdb_id = movie.tmdb_id
        details = _records(
            fetch(
                """
            select title, year, rank, imdb_rating, imdb_votes, tmdb_rating, metascore,
                   rotten_tomatoes, runtime_minutes, rated, budget_usd, revenue_usd, awards,
                   overview, tagline, imdb_id
            from movies where tmdb_id = ?
            """,
                [tmdb_id],
            )
        )[0]
        crew = fetch(
            """
            select c.job, list(p.name order by p.name) as names from credits c
            join people p using (person_id)
            where c.tmdb_id = ? and c.job in ('Director', 'Screenplay', 'Writer', 'Story',
                                               'Novel', 'Original Music Composer')
            group by c.job
            """,
            [tmdb_id],
        )
        details["crew"] = {row.job: list(row.names) for row in crew.itertuples()}
        details["cast"] = _records(
            fetch(
                "select p.name, c.character from credits c join people p using (person_id) "
                "where c.tmdb_id = ? and c.credit_type = 'cast' order by c.billing limit 8",
                [tmdb_id],
            )
        )
        details["keywords"] = fetch(
            "select keyword from movie_keywords where tmdb_id = ? order by keyword limit 15",
            [tmdb_id],
        )["keyword"].tolist()
        return details

    @server.tool(annotations=READ_ONLY)
    def query_database(query: str) -> dict[str, Any]:
        """Run one read-only DuckDB SELECT (see the imdb-insights://schema resource).
        Results are capped at 50 rows; writes, file access and multiple statements are refused."""
        require_warehouse()
        try:
            result = sql.run_query(db_path, query)
        except (sql.UnsafeQueryError, duckdb.Error, TimeoutError) as exc:
            # Anticipated: the caller's model reads the message and fixes its query. (Only
            # ToolError messages reach the client; anything else is reported as a crash.)
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        return {
            "columns": list(result.frame.columns),
            "rows": _records(result.frame),
            "truncated": result.truncated,
        }

    @server.tool(annotations=READ_ONLY)
    def search_reviews(query: str, movie: str | None = None, limit: int = 6) -> dict[str, Any]:
        """User-review passages about a topic (e.g. "pacing", "the soundtrack"), optionally for
        one movie. Hybrid keyword + semantic search; each passage has its author, rating and URL."""
        require_warehouse()
        con = search.connect(db_path)
        try:
            found = search.find_movie(con, movie) if movie else None
            if movie and found is None:
                return {"error": f"No movie matching {movie!r}.", "passages": []}
            hits = search.hybrid_search(
                con,
                embedder(),
                query,
                k=max(1, min(int(limit), 20)),
                tmdb_id=found.tmdb_id if found else None,
            )
        finally:
            con.close()
        if hits.empty:
            return {"passages": []}
        return {"passages": _records(hits[["title", "year", "author", "rating", "url", "text"]])}

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
