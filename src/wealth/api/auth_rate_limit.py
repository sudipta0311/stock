"""Small shared limiter for unauthenticated sign-in and refresh requests."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request

_RATE_LIMIT_REQUESTS = 10
_RATE_LIMIT_WINDOW_SECONDS = 60


class AuthRateLimiter:
    """Per-process sliding-window limiter keyed by the address Render appends."""

    def __init__(
        self,
        limit: int = _RATE_LIMIT_REQUESTS,
        window_seconds: int = _RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._limit = limit
        self._window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def retry_after(self, key: str, *, now: float | None = None) -> int | None:
        current = time.monotonic() if now is None else now
        cutoff = current - self._window_seconds
        with self._lock:
            for address in tuple(self._attempts):
                attempts = self._attempts[address]
                while attempts and attempts[0] <= cutoff:
                    attempts.popleft()
                if not attempts:
                    del self._attempts[address]

            attempts = self._attempts[key]
            if len(attempts) >= self._limit:
                return max(1, math.ceil(attempts[0] + self._window_seconds - current))
            attempts.append(current)
            return None


def client_address(request: Request) -> str:
    """Use Render's right-most forwarded hop, falling back to the socket peer."""
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    if hops:
        return hops[-1]
    return request.client.host if request.client is not None else "unknown"
