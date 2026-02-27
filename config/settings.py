"""Configuration settings loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLM provider to use",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )

    # Tool API Keys
    nasa_api_key: str | None = Field(
        default=None,
        description="NASA API key",
    )
    logmeal_api_key: str | None = Field(
        default=None,
        description="LogMeal API key",
    )

    # Performance Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Max API retry attempts")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_tokens: int = Field(default=4096, description="Max tokens per LLM request")
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=0.3,
        description="LLM temperature for deterministic outputs",
    )

    # Tool latency constraint
    max_tool_latency: float = Field(
        default=2.0,
        description="Max seconds per tool call before timeout",
    )

    # Evaluation rate limiting
    eval_rate_limit_initial_rps: float = Field(
        default=0.1,
        gt=0.0,
        description="Requests per second for evaluation LLM calls (default 0.1 = 10s between requests)",
    )

    # Phoenix evaluation method
    phoenix_evaluation_method: Literal["categorical", "discrete", "continuous"] = Field(
        default="categorical",
        description="Phoenix scoring method: categorical (binary 0/1), discrete (5-point scale), continuous (0.00-1.00)",
    )

    model_config = {"extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance loaded from environment."""
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),  # type: ignore[arg-type]
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        nasa_api_key=os.getenv("NASA_API_KEY"),
        logmeal_api_key=os.getenv("LOGMEAL_API_KEY"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tool_latency=float(os.getenv("MAX_TOOL_LATENCY", "2.0")),
        eval_rate_limit_initial_rps=float(os.getenv("EVAL_RATE_LIMIT_INITIAL_RPS", "0.1")),
        phoenix_evaluation_method=os.getenv("PHOENIX_EVALUATION_METHOD", "categorical"),  # type: ignore[arg-type]
    )
