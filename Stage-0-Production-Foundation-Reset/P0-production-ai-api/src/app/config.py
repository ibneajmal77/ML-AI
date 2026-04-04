from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "P0 Production AI API"
    app_env: str = "local"
    debug: bool = True
    api_key: str = "change-me"
    database_path: Path = Path("./data/app.db")
    default_model_name: str = "stage0-local-summarizer"
    request_timeout_seconds: int = Field(default=20, ge=5, le=120)
    job_work_simulation_seconds: int = Field(default=2, ge=0, le=30)
    rate_limit_per_minute: int = Field(default=30, ge=1, le=500)


@lru_cache
def get_settings() -> Settings:
    return Settings()

