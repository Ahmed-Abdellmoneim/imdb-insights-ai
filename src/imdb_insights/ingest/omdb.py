"""OMDb API: IMDb ratings and vote counts, Metascore, Rotten Tomatoes, and awards.

OMDb returns almost everything as display strings ("3,235,958", "$28,767,189", "N/A"), so
``parse_omdb`` normalizes them into typed fields. The free tier allows 1,000 requests a day;
responses are cached on disk so each movie only costs one request.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .http import ApiError, JsonApiClient

BASE_URL = "https://www.omdbapi.com/"
# OMDb's replies for IDs it has no record of ("Error getting data." is what a
# well-formed but unknown IMDb ID returns).
NOT_FOUND_ERRORS = {"Movie not found!", "Incorrect IMDb ID.", "Error getting data."}


class OmdbError(ApiError):
    """OMDb rejected the request (bad key, quota exhausted, ...)."""


class OmdbMovie(BaseModel):
    model_config = ConfigDict(frozen=True)

    imdb_id: str = Field(pattern=r"^tt\d+$")
    title: str
    type: str | None = None
    year: int | None = None
    rated: str | None = None
    released: date | None = None
    runtime_minutes: int | None = None
    genres: list[str] = []
    directors: list[str] = []
    writers: list[str] = []
    actors: list[str] = []
    plot: str | None = None
    languages: list[str] = []
    countries: list[str] = []
    poster_url: str | None = None
    imdb_rating: float | None = Field(default=None, ge=0, le=10)
    imdb_votes: int | None = Field(default=None, ge=0)
    metascore: int | None = Field(default=None, ge=0, le=100)
    rotten_tomatoes: int | None = Field(default=None, ge=0, le=100)
    box_office_usd: int | None = None
    awards: str | None = None
    oscar_wins: int = 0
    oscar_nominations: int | None = 0
    award_wins: int | None = None
    award_nominations: int | None = None


def _na(value: Any) -> Any:
    return None if value in (None, "", "N/A") else value


def _int(value: str | None) -> int | None:
    digits = re.sub(r"[^\d]", "", _na(value) or "")
    return int(digits) if digits else None


def _list(value: str | None) -> list[str]:
    return [part.strip() for part in (_na(value) or "").split(",") if part.strip()]


def _released(value: str | None) -> date | None:
    try:
        return datetime.strptime(_na(value) or "", "%d %b %Y").date()
    except ValueError:
        return None


def parse_awards(text: str | None) -> dict[str, int | None]:
    """Extract counts from strings like 'Won 7 Oscars. 21 wins & 43 nominations total'.

    In OMDb's wording, "Nominated for N Oscars" means N nominations and no wins, while
    "Won N Oscars" gives wins only (the nomination count is not stated).
    """
    text = _na(text) or ""
    won = re.search(r"Won (\d+) Oscars?", text)
    nominated = re.search(r"Nominated for (\d+) Oscars?", text)
    wins = re.search(r"(\d+) wins?", text)
    noms = re.search(r"(\d+) nominations?", text)
    return {
        "oscar_wins": int(won.group(1)) if won else 0,
        # Unknown when the text only says "Won N Oscars"; zero when no Oscar is mentioned.
        "oscar_nominations": int(nominated.group(1)) if nominated else (None if won else 0),
        "award_wins": int(wins.group(1)) if wins else None,
        "award_nominations": int(noms.group(1)) if noms else None,
    }


def parse_omdb(data: dict[str, Any]) -> OmdbMovie | None:
    """Return the movie, or ``None`` when OMDb says it does not exist."""
    if data.get("Response") == "False":
        error = data.get("Error", "unknown error")
        if error in NOT_FOUND_ERRORS:
            return None
        raise OmdbError(f"OMDb error: {error}", body=data)

    ratings = {r.get("Source"): r.get("Value") for r in data.get("Ratings") or []}
    rating = _na(data.get("imdbRating"))
    return OmdbMovie(
        imdb_id=data["imdbID"],
        title=data["Title"],
        type=_na(data.get("Type")),
        year=_int((data.get("Year") or "")[:4]),
        rated=_na(data.get("Rated")),
        released=_released(data.get("Released")),
        runtime_minutes=_int(data.get("Runtime")),
        genres=_list(data.get("Genre")),
        directors=_list(data.get("Director")),
        writers=_list(data.get("Writer")),
        actors=_list(data.get("Actors")),
        plot=_na(data.get("Plot")),
        languages=_list(data.get("Language")),
        countries=_list(data.get("Country")),
        poster_url=_na(data.get("Poster")),
        imdb_rating=float(rating) if rating else None,
        imdb_votes=_int(data.get("imdbVotes")),
        metascore=_int(data.get("Metascore")),
        rotten_tomatoes=_int(ratings.get("Rotten Tomatoes")),
        box_office_usd=_int(data.get("BoxOffice")),
        awards=_na(data.get("Awards")),
        **parse_awards(data.get("Awards")),
    )


class OmdbClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path | None,
        *,
        max_age: timedelta | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.http = JsonApiClient(
            BASE_URL,
            cache_dir,
            params={"apikey": api_key},
            max_concurrency=4,
            min_interval=0.15,
            max_age=max_age,
            transport=transport,
        )

    async def __aenter__(self) -> OmdbClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.http.__aexit__(*exc)

    async def movie(self, imdb_id: str) -> OmdbMovie | None:
        try:
            data = await self.http.get_json(
                "",
                {"i": imdb_id, "plot": "full"},
                cache_key=f"omdb/{imdb_id}",
                cacheable=lambda d: d.get("Response") == "True",
            )
        except ApiError as exc:
            # Bad keys and exhausted quotas come back as HTTP 401 with a JSON reason.
            if isinstance(exc.body, dict) and exc.body.get("Error"):
                raise OmdbError(f"OMDb error: {exc.body['Error']}", exc.status, exc.body) from exc
            raise
        return parse_omdb(data)

    def cached_movie(self, imdb_id: str) -> OmdbMovie | None:
        """The last successful response for ``imdb_id``, however old, or ``None``."""
        data = self.http.read_cache(f"omdb/{imdb_id}")
        return parse_omdb(data) if data else None

    async def movies(self, imdb_ids: Iterable[str]) -> dict[str, OmdbMovie | None]:
        """Fetch several movies concurrently; raises the first error encountered."""
        ids = list(dict.fromkeys(imdb_ids))
        results = await asyncio.gather(*(self.movie(i) for i in ids))
        return dict(zip(ids, results, strict=True))
