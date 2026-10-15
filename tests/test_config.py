import pytest

from imdb_insights.config import Settings


def test_keys_are_cleaned_of_paste_artifacts(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "\ufeffeyJabc \n")
    monkeypatch.setenv("HF_DATASET_REPO", "  me/data ")
    monkeypatch.setenv("OMDB_API_KEY", "   ")
    settings = Settings(_env_file=None)
    assert settings.tmdb_api_key.get_secret_value() == "eyJabc"
    assert settings.hf_dataset_repo == "me/data"
    assert settings.omdb_api_key is None  # blank means "not configured"


@pytest.mark.parametrize("name", ["LLM_API_KEY", "GROQ_API_KEY", "groq", "GROQ"])
def test_llm_key_accepts_common_names(monkeypatch, name):
    for other in ("LLM_API_KEY", "GROQ_API_KEY", "GROQ"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "gsk_test")
    assert Settings(_env_file=None).llm_api_key.get_secret_value() == "gsk_test"


def test_blank_llm_api_key_falls_through_to_groq(monkeypatch):
    # .env.example ships a blank LLM_API_KEY= line; a key added as "groq" must still work.
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("GROQ", "gsk_real")
    assert Settings(_env_file=None).llm_api_key.get_secret_value() == "gsk_real"
