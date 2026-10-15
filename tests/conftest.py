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
def no_real_credentials(monkeypatch):
    """Tests must never use the developer's real keys from .env (environment variables win
    over the .env file, and blank values mean "not configured")."""
    from imdb_insights.config import get_settings

    for name in (
        "TMDB_API_KEY",
        "OMDB_API_KEY",
        "HF_TOKEN",
        "HF_DATASET_REPO",
        "LLM_API_KEY",
        "GROQ_API_KEY",
        "GROQ",
    ):
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retries should not make the test suite sleep."""
    monkeypatch.setattr("imdb_insights.ingest.http._backoff", lambda attempt: 0)
