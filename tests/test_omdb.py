import asyncio
from datetime import date

import httpx
import pytest

from imdb_insights.ingest.omdb import OmdbClient, OmdbError, parse_awards, parse_omdb


def test_parse_real_response(load_fixture):
    movie = parse_omdb(load_fixture("omdb/tt0111161.json"))
    assert movie.imdb_id == "tt0111161"
    assert movie.title == "The Shawshank Redemption"
    assert movie.year == 1994
    assert movie.released == date(1994, 10, 14)
    assert movie.runtime_minutes == 142
    assert movie.imdb_rating == 9.3
    assert movie.imdb_votes > 3_000_000
    assert movie.metascore == 82
    assert movie.rotten_tomatoes == 89
    assert movie.box_office_usd == 28_767_189
    assert movie.directors == ["Frank Darabont"]
    assert movie.genres == ["Drama"]
    assert (movie.oscar_wins, movie.oscar_nominations) == (0, 7)


def test_missing_values_become_none(load_fixture):
    movie = parse_omdb(load_fixture("omdb/tt0050083.json"))  # 12 Angry Men: no box office
    assert movie.box_office_usd is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Won 3 Oscars. 31 wins & 31 nominations total",
            {"oscar_wins": 3, "oscar_nominations": None, "award_wins": 31, "award_nominations": 31},
        ),
        (
            "Nominated for 7 Oscars. 21 wins & 43 nominations total",
            {"oscar_wins": 0, "oscar_nominations": 7, "award_wins": 21, "award_nominations": 43},
        ),
        (
            "Won 1 Oscar. 1 win & 1 nomination total",
            {"oscar_wins": 1, "oscar_nominations": None, "award_wins": 1, "award_nominations": 1},
        ),
        (
            "N/A",
            {
                "oscar_wins": 0,
                "oscar_nominations": 0,
                "award_wins": None,
                "award_nominations": None,
            },
        ),
    ],
)
def test_parse_awards(text, expected):
    assert parse_awards(text) == expected


@pytest.mark.parametrize("error", ["Movie not found!", "Error getting data.", "Incorrect IMDb ID."])
def test_unknown_ids_return_none(error):
    assert parse_omdb({"Response": "False", "Error": error}) is None


def test_other_errors_raise():
    with pytest.raises(OmdbError):
        parse_omdb({"Response": "False", "Error": "Something new"})


def _client(handler, cache_dir=None):
    client = OmdbClient("test-key", cache_dir, transport=httpx.MockTransport(handler))
    client.http.min_interval = 0
    return client


async def _movies(client, ids):
    async with client:
        return await client.movies(ids)


def test_client_fetches_and_caches(tmp_path, load_fixture):
    body = load_fixture("omdb/tt0111161.json")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    for _ in range(2):
        result = asyncio.run(_movies(_client(handler, tmp_path), ["tt0111161", "tt0111161"]))
    assert result["tt0111161"].title == "The Shawshank Redemption"
    assert len(calls) == 1


def test_quota_exhausted_raises_omdb_error():
    def handler(request):
        return httpx.Response(401, json={"Response": "False", "Error": "Request limit reached!"})

    with pytest.raises(OmdbError, match="Request limit reached"):
        asyncio.run(_movies(_client(handler), ["tt0111161"]))


def test_not_found_is_not_cached(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"Response": "False", "Error": "Error getting data."})

    result = asyncio.run(_movies(_client(handler, tmp_path), ["tt0000000"]))
    assert result == {"tt0000000": None}
    assert not list(tmp_path.rglob("*.json"))
