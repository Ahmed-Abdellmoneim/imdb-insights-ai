import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def load(relative: str):
        return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))

    return load


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retries should not make the test suite sleep."""
    monkeypatch.setattr("imdb_insights.ingest.http._backoff", lambda attempt: 0)
