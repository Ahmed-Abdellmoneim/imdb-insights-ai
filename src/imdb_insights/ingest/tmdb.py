"""TMDB API: candidate discovery, movie details, credits, keywords, and user reviews.

One ``/movie/{id}`` request with ``append_to_response`` returns everything for a movie, so
enriching N movies costs N requests. TMDB allows roughly 50 requests/second; we stay well
under that.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import ftfy
import httpx
from pydantic import BaseModel, ConfigDict, Field

from .http import JsonApiClient

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
DOCUMENTARY, TV_MOVIE = 99, 10770


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)


class CastMember(Record):
    person_id: int
    name: str
    character: str | None = None
    billing: int
    gender: int | None = None
    profile_path: str | None = None


class CrewMember(Record):
    person_id: int
    name: str
    department: str | None = None
    job: str | None = None
    gender: int | None = None
    profile_path: str | None = None


class TmdbReview(Record):
    review_id: str
    author: str | None = None
    rating: float | None = Field(default=None, ge=0, le=10)
    content: str
    created_at: datetime | None = None
    url: str | None = None


class TmdbMovie(Record):
    tmdb_id: int
    imdb_id: str | None = Field(default=None, pattern=r"^tt\d+$")
    title: str
    original_title: str | None = None
    original_language: str | None = None
    release_date: date | None = None
    runtime_minutes: int | None = None
    overview: str | None = None
    tagline: str | None = None
    genres: list[str] = []
    poster_path: str | None = None
    backdrop_path: str | None = None
    budget_usd: int | None = None
    revenue_usd: int | None = None
    tmdb_rating: float | None = None
    tmdb_votes: int | None = None
    popularity: float | None = None
    collection: str | None = None
    countries: list[str] = []
    languages: list[str] = []
    companies: list[str] = []
    keywords: list[str] = []
    cast: list[CastMember] = []
    crew: list[CrewMember] = []
    reviews: list[TmdbReview] = []
    review_total: int | None = None


def image_url(path: str | None, size: str = "w500") -> str | None:
    return f"{IMAGE_BASE_URL}{size}{path}" if path else None


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _positive(value: int | None) -> int | None:
    """TMDB uses 0 for unknown budgets, revenues and runtimes."""
    return value or None


_BLOCK_TAG = re.compile(r"<\s*(br|/p|/h\d|/li|/div)\s*/?\s*>", re.IGNORECASE)
# Only real HTML tag names, so text like "x<y and z>w" survives.
_HTML_TAG = re.compile(
    r"</?\s*(p|br|h[1-6]|em|strong|i|b|u|s|a|ul|ol|li|div|span|blockquote|hr|img|sup|sub|"
    r"small|code|pre|table|thead|tbody|tr|td|th|font|center)\b[^<>]*/?>",
    re.IGNORECASE,
)


def clean_review_text(text: str) -> str:
    """Plain text for reviews: HTML removed, encoding damage repaired, line ends unified.

    About 5% of TMDB reviews contain HTML (``<p>``, ``<h2>``...). A handful contain text
    decoded with the wrong code page (e.g. Windows "\\x85" for an ellipsis), which
    ``ftfy.fix_encoding`` repairs without touching anything else (curly quotes stay).
    Markdown is kept: the app renders it.
    """
    text = ftfy.fix_encoding(text).replace("\r\n", "\n").replace("\r", "\n")
    text = _BLOCK_TAG.sub("\n", text)
    text = html.unescape(_HTML_TAG.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _names(items: list[dict] | None, key: str = "name") -> list[str]:
    return [i[key] for i in items or [] if i.get(key)]


def parse_movie(data: dict[str, Any]) -> TmdbMovie:
    credits = data.get("credits") or {}
    reviews = data.get("reviews") or {}
    return TmdbMovie(
        tmdb_id=data["id"],
        imdb_id=data.get("imdb_id") or (data.get("external_ids") or {}).get("imdb_id") or None,
        title=data["title"],
        original_title=data.get("original_title"),
        original_language=data.get("original_language"),
        release_date=_date(data.get("release_date")),
        runtime_minutes=_positive(data.get("runtime")),
        overview=data.get("overview") or None,
        tagline=data.get("tagline") or None,
        genres=_names(data.get("genres")),
        poster_path=data.get("poster_path"),
        backdrop_path=data.get("backdrop_path"),
        budget_usd=_positive(data.get("budget")),
        revenue_usd=_positive(data.get("revenue")),
        tmdb_rating=data.get("vote_average"),
        tmdb_votes=data.get("vote_count"),
        popularity=data.get("popularity"),
        collection=(data.get("belongs_to_collection") or {}).get("name"),
        countries=_names(data.get("production_countries")),
        languages=_names(data.get("spoken_languages"), "english_name"),
        companies=_names(data.get("production_companies")),
        keywords=_names((data.get("keywords") or {}).get("keywords")),
        cast=[
            CastMember(
                person_id=c["id"],
                name=c["name"],
                character=c.get("character") or None,
                billing=c.get("order", i),
                gender=c.get("gender"),
                profile_path=c.get("profile_path"),
            )
            for i, c in enumerate(credits.get("cast") or [])
        ],
        crew=[
            CrewMember(
                person_id=c["id"],
                name=c["name"],
                department=c.get("department"),
                job=c.get("job"),
                gender=c.get("gender"),
                profile_path=c.get("profile_path"),
            )
            for c in credits.get("crew") or []
        ],
        reviews=[
            TmdbReview(
                review_id=r["id"],
                author=r.get("author") or (r.get("author_details") or {}).get("username"),
                rating=(r.get("author_details") or {}).get("rating"),
                content=clean_review_text(r["content"]),
                created_at=r.get("created_at"),
                url=r.get("url"),
            )
            for r in reviews.get("results") or []
            if (r.get("content") or "").strip()
        ],
        review_total=reviews.get("total_results"),
    )


class TmdbClient:
    def __init__(
        self,
        token: str,
        cache_dir: Path | None,
        *,
        max_age: timedelta | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.http = JsonApiClient(
            BASE_URL,
            cache_dir,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={"language": "en-US"},
            max_concurrency=8,
            min_interval=0.03,
            max_age=max_age,
            transport=transport,
        )

    async def __aenter__(self) -> TmdbClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.http.__aexit__(*exc)

    async def discover_top_rated(
        self,
        *,
        min_rating: float,
        min_votes: int,
        max_votes: int | None = None,
        max_pages: int = 500,
        exclude_genres: Iterable[int] = (DOCUMENTARY, TV_MOVIE),
        snapshot: str | None = None,
    ) -> list[int]:
        """Return TMDB ids of every movie within the thresholds, best rated first.

        ``snapshot`` (e.g. today's date) namespaces the cache, since rankings drift.
        TMDB serves at most 500 pages of 20 results.
        """
        params = {
            "sort_by": "vote_average.desc",
            "vote_average.gte": min_rating,
            "vote_count.gte": min_votes,
            "without_genres": "|".join(map(str, exclude_genres)),
            "include_adult": "false",
            "include_video": "false",
        }
        if max_votes is not None:
            params["vote_count.lte"] = max_votes
        excluded = "_".join(map(str, exclude_genres))  # no "|": it is invalid in Windows paths
        query = f"rating{min_rating}-votes{min_votes}-{max_votes or ''}-without{excluded}"
        ids: list[int] = []
        for page in range(1, max_pages + 1):
            data = await self.http.get_json(
                "/discover/movie",
                params | {"page": page},
                cache_key=f"tmdb/discover/{snapshot}/{query}/{page}" if snapshot else None,
            )
            ids += [m["id"] for m in data.get("results") or []]
            if page >= (data.get("total_pages") or 0):
                break
        return list(dict.fromkeys(ids))

    async def movie(self, tmdb_id: int) -> TmdbMovie:
        data = await self.http.get_json(
            f"/movie/{tmdb_id}",
            {"append_to_response": "credits,keywords,reviews,external_ids"},
            cache_key=f"tmdb/movie/{tmdb_id}",
        )
        return parse_movie(data)

    def cached_movie(self, tmdb_id: int) -> TmdbMovie | None:
        """The last successful response for ``tmdb_id``, however old, or ``None``."""
        data = self.http.read_cache(f"tmdb/movie/{tmdb_id}")
        return parse_movie(data) if data else None
