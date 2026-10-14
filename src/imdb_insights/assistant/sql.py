"""Read-only SQL for model-written queries, with layered guardrails.

1. Exactly one statement, and it must parse as SELECT (DuckDB's own parser decides).
2. The connection is read-only, so nothing can be written even if a check were bypassed.
3. External access is disabled: a SELECT like ``read_csv('/etc/passwd')`` parses as a valid
   SELECT, but DuckDB refuses to touch files, URLs or extensions (and the setting cannot be
   turned back on from inside a query).
4. Results are capped, and a timer interrupts queries that run too long.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

MAX_ROWS = 50
TIMEOUT_SECONDS = 10

SCHEMA_GUIDE = """\
DuckDB tables (one row per ...):
- movies (movie): tmdb_id, imdb_id, title, original_title, original_language, year,
  release_date, runtime_minutes, overview, tagline, collection, budget_usd, revenue_usd,
  tmdb_rating, tmdb_votes, imdb_rating, imdb_votes, metascore (0-100),
  rotten_tomatoes (0-100), box_office_usd, rated, awards (text), oscar_wins,
  oscar_nominations, award_wins, award_nominations, countries (VARCHAR[]),
  languages (VARCHAR[]), companies (VARCHAR[]), weighted_rating,
  rank (1-250 for our IMDb-style Top 250, NULL for other candidates)
- movie_genres (movie x genre): tmdb_id, genre
- movie_keywords (movie x keyword): tmdb_id, keyword
- people (person): person_id, name, gender, profile_url
- credits (movie x person x role): tmdb_id, person_id, credit_type ('cast' or 'crew'),
  department, job (e.g. 'Director', 'Screenplay', 'Original Music Composer'),
  character, billing (cast order, 0 = top billed)
- reviews (user review): review_id, tmdb_id, author, rating (1-10, may be NULL), content,
  created_at, url
Notes: "the Top 250" means movies WHERE rank IS NOT NULL. Use ILIKE for name matching.
List columns: use list_contains(countries, 'Japan') or unnest(...).
"""


class UnsafeQueryError(ValueError):
    """The query is not a single read-only SELECT."""


@dataclass
class QueryResult:
    frame: pd.DataFrame
    truncated: bool

    def to_text(self) -> str:
        """Compact CSV for the model to read."""
        if self.frame.empty:
            return "(no rows)"
        text = self.frame.to_csv(index=False).strip()
        return text + (f"\n(first {len(self.frame)} rows shown)" if self.truncated else "")


def check_query(sql: str) -> str:
    """Return the cleaned query, or raise ``UnsafeQueryError``."""
    sql = sql.strip().rstrip(";").strip()
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        raise UnsafeQueryError(f"Could not parse the query: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeQueryError("Send exactly one SQL statement.")
    if statements[0].type.name != "SELECT":
        raise UnsafeQueryError(f"Only SELECT queries are allowed, not {statements[0].type.name}.")
    return sql


def run_query(db_path: Path, sql: str, max_rows: int = MAX_ROWS) -> QueryResult:
    sql = check_query(sql)
    con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
    timer = threading.Timer(TIMEOUT_SECONDS, con.interrupt)
    timer.start()
    try:
        frame = con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {max_rows + 1}").df()
    except duckdb.InterruptException as exc:
        raise TimeoutError(f"Query took longer than {TIMEOUT_SECONDS}s") from exc
    finally:
        timer.cancel()
        con.close()
    return QueryResult(frame.head(max_rows), truncated=len(frame) > max_rows)
