"""Async JSON client shared by the API integrations.

Handles the parts every API needs: bounded concurrency, a minimum gap between requests,
retries with exponential backoff (honouring ``Retry-After``) on rate limits, server errors
and network failures, and an on-disk cache of successful responses so re-runs are free.
Secrets passed in ``params`` never reach cache file names; callers choose the cache key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    """A request failed permanently."""

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class JsonApiClient:
    def __init__(
        self,
        base_url: str,
        cache_dir: Path | None,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        max_concurrency: int = 5,
        min_interval: float = 0.1,
        retries: int = 4,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.retries = retries
        self.min_interval = min_interval
        self.network_requests = 0
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._pace_lock = asyncio.Lock()
        self._last_request = 0.0
        self._client = httpx.AsyncClient(
            base_url=base_url,
            params=params,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    async def __aenter__(self) -> JsonApiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    def _cache_file(self, cache_key: str | None) -> Path | None:
        if self.cache_dir is None or cache_key is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    async def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        cache_key: str | None = None,
        cacheable: Callable[[Any], bool] | None = None,
    ) -> Any:
        """GET ``path`` and return its decoded JSON, using the cache when possible.

        ``cacheable`` decides whether a successful response is worth keeping; APIs that
        report errors inside a 200 response use it to keep those out of the cache.
        """
        cache_file = self._cache_file(cache_key)
        if cache_file and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        data = await self._fetch(path, params)
        if cache_file and (cacheable is None or cacheable(data)):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    async def _pace(self) -> None:
        async with self._pace_lock:
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    async def _fetch(self, path: str, params: dict[str, Any] | None) -> Any:
        async with self._semaphore:
            for attempt in range(1, self.retries + 1):
                await self._pace()
                self.network_requests += 1
                try:
                    resp = await self._client.get(path, params=params)
                except httpx.TransportError as exc:
                    if attempt == self.retries:
                        raise ApiError(f"GET {path} failed: {exc}") from exc
                    await asyncio.sleep(_backoff(attempt))
                    continue

                if resp.status_code in RETRY_STATUSES and attempt < self.retries:
                    delay = _retry_after(resp) or _backoff(attempt)
                    log.warning("GET %s -> %d, retrying in %.1fs", path, resp.status_code, delay)
                    await asyncio.sleep(delay)
                    continue
                if resp.is_error:
                    raise ApiError(
                        f"GET {path} -> HTTP {resp.status_code}",
                        status=resp.status_code,
                        body=_safe_json(resp),
                    )
                return resp.json()
        raise AssertionError("unreachable")


def _backoff(attempt: int) -> float:
    return min(60.0, 2**attempt) + random.uniform(0, 1)


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text[:500]
