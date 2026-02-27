"""Rate limiter with exponential backoff for API calls."""

from __future__ import annotations

import asyncio
import time
from typing import TypeVar, Callable, Any, cast

T = TypeVar("T")


class RateLimiter:
    """
    Rate limiter with exponential backoff.
    
    Starts at a conservative rate (default 0.1 RPS = 10 seconds between calls)
    and backs off exponentially when rate limit errors are encountered.
    """

    def __init__(
        self,
        initial_rps: float = 0.1,
        backoff_factor: float = 2.0,
        recovery_factor: float = 1.1,
        max_retries: int = 10,
    ) -> None:
        """
        Initialize rate limiter.
        
        Args:
            initial_rps: Starting requests per second (default 0.1 = 10s between requests)
            backoff_factor: Factor to divide RPS by on rate limit (default 2.0)
            recovery_factor: Factor to multiply RPS by on success (default 1.1)
            max_retries: Maximum retry attempts before giving up (default 10)
        """
        self.initial_rps = initial_rps
        self.current_rps = initial_rps
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.max_retries = max_retries
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def delay_seconds(self) -> float:
        """Get current delay between requests in seconds."""
        return 1.0 / self.current_rps

    async def wait(self) -> None:
        """Wait appropriate time before next request."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_time = self.delay_seconds - elapsed
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            self._last_request_time = time.monotonic()

    def backoff(self) -> None:
        """Reduce RPS after rate limit error."""
        self.current_rps = self.current_rps / self.backoff_factor

    def recover(self) -> None:
        """Slowly increase RPS after successful request."""
        self.current_rps = min(
            self.initial_rps,
            self.current_rps * self.recovery_factor
        )

    def reset(self) -> None:
        """Reset to initial RPS."""
        self.current_rps = self.initial_rps

    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        service_name: str = "API",
        metric_name: str = "",
        rate_limit_exceptions: tuple[type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> T:
        """
        Execute function with rate limiting and retry on rate limit errors.
        
        Args:
            func: Async or sync function to execute
            *args: Positional arguments to pass to func
            service_name: Name of the service for logging (e.g., "DeepEval")
            metric_name: Name of the metric being evaluated for logging
            rate_limit_exceptions: Exception types that indicate rate limiting
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result from func
            
        Raises:
            Last exception if all retries exhausted
        """
        last_exception: Exception | None = None
        log_prefix = f"[{service_name}/{metric_name}]" if metric_name else f"[{service_name}]"
        
        for attempt in range(1, self.max_retries + 1):
            await self.wait()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self.recover()
                return cast(T, result)
                
            except rate_limit_exceptions as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check for rate limiting
                is_rate_limit = "rate" in error_str or "limit" in error_str or "throttl" in error_str
                
                # Check for timeout errors (including wrapped RetryError with TimeoutError)
                is_timeout = (
                    "timeout" in error_str
                    or "timed out" in error_str
                    or isinstance(e, (TimeoutError, asyncio.TimeoutError))
                    or "retryerror" in error_str and "timeouterror" in error_str
                )
                
                if is_rate_limit:
                    self.backoff()
                    print(
                        f"{log_prefix} Rate limited on attempt {attempt}/{self.max_retries}, "
                        f"backing off to {self.delay_seconds:.1f}s between requests"
                    )
                    if attempt < self.max_retries:
                        continue
                elif is_timeout:
                    self.backoff()
                    print(
                        f"{log_prefix} Timeout on attempt {attempt}/{self.max_retries}, "
                        f"backing off to {self.delay_seconds:.1f}s between requests"
                    )
                    if attempt < self.max_retries:
                        continue
                
                raise
        
        if last_exception:
            raise last_exception
        raise RuntimeError(f"{log_prefix} Max retries ({self.max_retries}) exhausted")


# Global rate limiter instance for shared use across evaluators
_global_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance using settings."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        from config.settings import get_settings
        settings = get_settings()
        _global_rate_limiter = RateLimiter(
            initial_rps=settings.eval_rate_limit_initial_rps,
            backoff_factor=2.0,
            recovery_factor=1.1,
            max_retries=10,
        )
    return _global_rate_limiter
