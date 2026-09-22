"""Runtime configuration. Every value can be set through AGENT_* environment variables or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # "rules" runs a deterministic backend (tests, CI, local dev without keys);
    # "openai" runs the same graph against an OpenAI-compatible chat model.
    backend: Literal["rules", "openai"] = "rules"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None  # falls back to OPENAI_API_KEY
    openai_base_url: str | None = None  # any OpenAI-compatible endpoint
    openai_timeout_s: float = 30.0

    # Below this classification confidence the graph asks a clarifying question
    # instead of drafting an answer.
    clarify_threshold: float = 0.6
    max_snippets: int = 3
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
