"""Base tool class with retry logic and common functionality."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx

from config.settings import get_settings


T = TypeVar("T")


class ToolError(Exception):
    """Base exception for tool errors."""

    def __init__(self, message: str, tool_name: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.recoverable = recoverable


class BaseTool(ABC):
    """Abstract base class for all API wrapper tools."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: httpx.AsyncClient | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for logging and identification."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        ...

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout)
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with retry logic and exponential backoff."""
        client = await self.get_client()
        last_exception: Exception | None = None

        for attempt in range(self.settings.max_retries):
            try:
                start_time = time.time()
                response = await asyncio.wait_for(
                    client.request(method, url, **kwargs),
                    timeout=self.settings.max_tool_latency,
                )
                elapsed = time.time() - start_time

                if elapsed > self.settings.max_tool_latency:
                    raise ToolError(
                        f"Request exceeded max latency: {elapsed:.2f}s > {self.settings.max_tool_latency}s",
                        self.name,
                    )

                response.raise_for_status()
                return response

            except asyncio.TimeoutError:
                last_exception = ToolError(
                    f"Request timed out after {self.settings.max_tool_latency}s",
                    self.name,
                )
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status >= 500:
                    last_exception = e
                elif status == 429:
                    last_exception = ToolError(
                        f"Rate limited (HTTP 429). Retry-After: {e.response.headers.get('Retry-After', 'unknown')}",
                        self.name,
                        recoverable=True,
                    )
                else:
                    raise ToolError(
                        f"HTTP error {status}: {e.response.text[:200]}",
                        self.name,
                        recoverable=False,
                    ) from e
            except httpx.RequestError as e:
                last_exception = e

            if attempt < self.settings.max_retries - 1:
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)

        raise ToolError(
            f"Failed after {self.settings.max_retries} attempts: {last_exception}",
            self.name,
        )

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """HTTP GET with retry."""
        return await self._request_with_retry("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """HTTP POST with retry."""
        return await self._request_with_retry("POST", url, **kwargs)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters."""
        ...

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    @abstractmethod
    def _get_parameters_schema(self) -> dict[str, Any]:
        """Get JSON schema for tool parameters."""
        ...
