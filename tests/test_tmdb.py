import asyncio
from datetime import date

import httpx
import pytest

from imdb_insights.ingest.tmdb import TmdbClient, clean_review_text, image_url, parse_movie


def test_parse_real_response(load_fixture):
    movie = parse_movie(load_fixture("tmdb/movie_278.json"))
    assert movie.tmdb_id == 278
    assert movie.imdb_id == "tt0111161"
    assert movie.title == "The Shawshank Redemption"
    assert movie.release_date == date(1994, 9, 23)
    assert movie.runtime_minutes == 142
    assert movie.budget_usd == 25_000_000
    assert movie.genres == ["Drama", "Crime"]
    assert [c.name for c in movie.cast[:2]] == ["Tim Robbins", "Morgan Freeman"]
    assert movie.cast[0].character == "Andy Dufresne"
    assert movie.cast[0].billing == 0
    assert {"Director", "Screenplay", "Novel"} <= {c.job for c in movie.crew}
    assert [r.rating for r in movie.reviews] == [9.0, 10.0, 6.0]
    assert movie.review_total and movie.review_total >= len(movie.reviews)
    assert len(movie.keywords) == 6


def test_unknown_values_become_none():
    movie = parse_movie(
        {
            "id": 1,
            "title": "X",
            "budget": 0,
            "revenue": 0,
            "runtime": 0,
            "release_date": "",
            "imdb_id": "",
            "overview": "",
        }
    )
    assert movie.budget_usd is None
    assert movie.revenue_usd is None
    assert movie.runtime_minutes is None
    assert movie.release_date is None
    assert movie.imdb_id is None
    assert movie.overview is None


def test_imdb_id_falls_back_to_external_ids():
    movie = parse_movie({"id": 1, "title": "X", "external_ids": {"imdb_id": "tt42"}})
    assert movie.imdb_id == "tt42"


def test_blank_reviews_are_dropped():
    movie = parse_movie(
        {
            "id": 1,
            "title": "X",
            "reviews": {
                "total_results": 2,
                "results": [
                    {"id": "a", "content": "  ", "author_details": {}},
                    {"id": "b", "content": "Great.", "author_details": {"rating": None}},
                ],
            },
        }
    )
    assert [r.review_id for r in movie.reviews] == ["b"]


def test_image_url():
    assert image_url("/p.jpg") == "https://image.tmdb.org/t/p/w500/p.jpg"
    assert image_url(None) is None


def test_discover_follows_pages_until_the_last(tmp_path):
    pages_seen = []

    def handler(request):
        page = int(request.url.params["page"])
        pages_seen.append(page)
        assert request.headers["Authorization"] == "Bearer token"
        assert request.url.params["vote_average.gte"] == "7.3"
        return httpx.Response(
            200, json={"total_pages": 3, "results": [{"id": page * 10}, {"id": page * 10 + 1}]}
        )

    async def run():
        async with TmdbClient("token", tmp_path, transport=httpx.MockTransport(handler)) as c:
            c.http.min_interval = 0
            return await c.discover_top_rated(min_rating=7.3, min_votes=600, snapshot="d")

    assert asyncio.run(run()) == [10, 11, 20, 21, 30, 31]
    assert pages_seen == [1, 2, 3]
    # Cache paths must be valid on Windows (no "|" from the genre filter).
    assert all("|" not in str(p) for p in tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<h2>Legacy</h2><p>It <em>lingers</em>.</p>", "Legacy\nIt lingers."),
        ("Line one<br/>Line two<br>Three", "Line one\nLine two\nThree"),
        ("Fish &amp; chips", "Fish & chips"),
        ("Windows line\r\n\r\nends", "Windows line\n\nends"),
        ("just prior to D-Day\x85.", "just prior to D-Day\u2026."),
        ("It\u2019s fine, curly quotes stay", "It\u2019s fine, curly quotes stay"),
        ("**Bold** markdown stays", "**Bold** markdown stays"),
        ("if x<y and z>w then 3 < 5", "if x<y and z>w then 3 < 5"),
        ("a\n\n\n\n\nb", "a\n\nb"),
    ],
)
def test_clean_review_text(raw, clean):
    assert clean_review_text(raw) == clean
