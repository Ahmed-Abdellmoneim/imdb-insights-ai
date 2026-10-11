from imdb_insights.config import Settings


def test_keys_are_cleaned_of_paste_artifacts(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "﻿eyJabc \n")
    monkeypatch.setenv("HF_DATASET_REPO", "  me/data ")
    monkeypatch.setenv("OMDB_API_KEY", "   ")
    settings = Settings(_env_file=None)
    assert settings.tmdb_api_key.get_secret_value() == "eyJabc"
    assert settings.hf_dataset_repo == "me/data"
    assert settings.omdb_api_key is None  # blank means "not configured"
