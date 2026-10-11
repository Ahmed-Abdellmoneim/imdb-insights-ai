import asyncio

import httpx
import pytest

from imdb_insights.ingest.http import ApiError, JsonApiClient


def make_client(handler, cache_dir=None, **kwargs):
    return JsonApiClient(
        "https://api.test",
        cache_dir,
        min_interval=0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def _get(client, *args, **kwargs):
    async with client:
        return await client.get_json(*args, **kwargs)


def test_retries_rate_limit_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    assert asyncio.run(_get(make_client(handler), "/x")) == {"ok": True}
    assert len(calls) == 3


def test_gives_up_after_retries_and_keeps_error_body():
    client = make_client(lambda r: httpx.Response(503, json={"msg": "down"}), retries=2)
    with pytest.raises(ApiError) as exc:
        asyncio.run(_get(client, "/x"))
    assert exc.value.status == 503
    assert exc.value.body == {"msg": "down"}


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"Error": "Invalid API key!"})

    with pytest.raises(ApiError):
        asyncio.run(_get(make_client(handler), "/x"))
    assert len(calls) == 1


def test_network_errors_are_retried():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=[1])

    assert asyncio.run(_get(make_client(handler), "/x")) == [1]


def test_cache_hit_skips_network(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"n": len(calls)})

    first = asyncio.run(_get(make_client(handler, tmp_path), "/x", cache_key="a/b"))
    second = asyncio.run(_get(make_client(handler, tmp_path), "/x", cache_key="a/b"))
    assert first == second == {"n": 1}
    assert (tmp_path / "a" / "b.json").exists()


def test_uncacheable_responses_are_not_stored(tmp_path):
    client = make_client(lambda r: httpx.Response(200, json={"Response": "False"}), tmp_path)
    asyncio.run(_get(client, "/x", cache_key="k", cacheable=lambda d: d["Response"] == "True"))
    assert not (tmp_path / "k.json").exists()


def test_default_params_are_sent_but_not_used_in_cache_names(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json={})

    client = make_client(handler, tmp_path, params={"apikey": "secret"})
    asyncio.run(_get(client, "/x", {"i": "tt1"}, cache_key="omdb/tt1"))
    assert seen[0].params["apikey"] == "secret"
    assert seen[0].params["i"] == "tt1"
    assert [p.name for p in tmp_path.rglob("*.json")] == ["tt1.json"]
