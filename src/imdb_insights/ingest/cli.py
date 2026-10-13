"""Command-line entry point: ``uv run python -m imdb_insights.ingest --help``."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ..config import Settings, get_settings
from . import pipeline
from .omdb import OmdbClient
from .tmdb import TmdbClient

app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class Options:
    min_rating: float
    min_votes: int
    low_votes: pipeline.LowVoteTier
    tmdb_max_age: timedelta
    omdb_max_age: timedelta


@app.command()
def ingest(
    min_rating: Annotated[float, typer.Option(help="Minimum TMDB rating for a candidate.")] = 6.9,
    min_votes: Annotated[int, typer.Option(help="Minimum TMDB votes for a candidate.")] = 250,
    low_vote_min_rating: Annotated[
        float, typer.Option(help="Minimum TMDB rating for low-vote candidates.")
    ] = 7.0,
    low_vote_min_votes: Annotated[
        int, typer.Option(help="Low-vote candidates: TMDB votes from this up to --min-votes.")
    ] = 50,
    tmdb_max_age_days: Annotated[
        float, typer.Option(help="Re-fetch TMDB details older than this (on average).")
    ] = 7,
    omdb_max_age_days: Annotated[
        float, typer.Option(help="Re-fetch IMDb ratings older than this (on average).")
    ] = 21,
    data_dir: Annotated[Path | None, typer.Option(help="Defaults to DATA_DIR or ./data.")] = None,
) -> None:
    """Build the movie tables and our own IMDb-style Top 250."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
    # httpx logs full request URLs at INFO, and OMDb's key travels in the query string.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    if not settings.tmdb_api_key or not settings.omdb_api_key:
        console.print("[red]Set TMDB_API_KEY and OMDB_API_KEY in .env (see .env.example).[/red]")
        raise typer.Exit(code=2)

    data_dir = data_dir or settings.data_dir
    started = datetime.now(UTC)
    options = Options(
        min_rating=min_rating,
        min_votes=min_votes,
        low_votes=pipeline.LowVoteTier(low_vote_min_rating, low_vote_min_votes),
        tmdb_max_age=timedelta(days=tmdb_max_age_days),
        omdb_max_age=timedelta(days=omdb_max_age_days),
    )
    tables, report = asyncio.run(_run(settings, data_dir, options, started))

    out_dir = data_dir / "processed"
    db_path = pipeline.write_outputs(tables, report, out_dir, started)

    summary = Table(title=f"Wrote {db_path}")
    summary.add_column("table")
    summary.add_column("rows", justify="right")
    for name, df in tables.items():
        summary.add_row(name, f"{len(df):,}")
    console.print(summary)
    if report["stale_entries"]:
        console.print(f"{report['stale_entries']} entries could not be refreshed; kept older data")
    for warning in report["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if not report["complete"]:
        # Distinct exit code so schedulers can tell "partial, resume later" from a crash.
        raise typer.Exit(code=3)


async def _run(settings: Settings, data_dir: Path, options: Options, started: datetime):
    raw = data_dir / "raw"
    async with (
        TmdbClient(
            settings.tmdb_api_key.get_secret_value(), raw, max_age=options.tmdb_max_age
        ) as tmdb,
        OmdbClient(
            settings.omdb_api_key.get_secret_value(), raw, max_age=options.omdb_max_age
        ) as omdb,
    ):
        col = await pipeline.collect(
            tmdb,
            omdb,
            min_rating=options.min_rating,
            min_votes=options.min_votes,
            low_votes=options.low_votes,
            snapshot=started.date().isoformat(),
        )
        console.print(
            f"Network requests: TMDB {tmdb.http.network_requests}, "
            f"OMDb {omdb.http.network_requests} (the rest came from cache)"
        )
    tables = pipeline.build_tables(col)
    return tables, pipeline.quality_report(col, tables)
