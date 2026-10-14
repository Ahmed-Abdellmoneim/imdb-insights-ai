"""Project settings, read from environment variables or a local ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    omdb_api_key: SecretStr | None = None
    tmdb_api_key: SecretStr | None = None
    data_dir: Path = Path("data")
    # Private Hugging Face dataset that stores the published warehouse and the API cache.
    hf_dataset_repo: str | None = None
    hf_token: SecretStr | None = None

    @field_validator("omdb_api_key", "tmdb_api_key", "hf_token", "hf_dataset_repo", mode="before")
    @classmethod
    def _clean(cls, value: object) -> object:
        """Drop the whitespace and byte-order marks that copy-pasting often adds.

        An empty value means "not configured".
        """
        if isinstance(value, str):
            value = value.strip().lstrip("﻿").strip()
            return value or None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
