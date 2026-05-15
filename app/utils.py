"""Logging and resilient HTTP client for LLM providers."""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import httpx

LOG = logging.getLogger("shl_agent")


def setup_logging(level: str | None = None) -> None:
    lvl = level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _sleep_backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
    time.sleep(min(cap, base * (2**attempt)) + random.random() * 0.15)


def post_json_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_s: float,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """
    POST JSON with retries on 429 / 5xx / transient network errors.
    Raises last error if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=timeout_s) as client:
                r = client.post(url, headers=headers, json=payload)
                if r.status_code == 429 or r.status_code >= 500:
                    LOG.warning("LLM HTTP %s, attempt %s", r.status_code, attempt + 1)
                    _sleep_backoff(attempt)
                    continue
                r.raise_for_status()
                return r.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_exc = e
            LOG.warning("LLM request error %s, attempt %s", e, attempt + 1)
            _sleep_backoff(attempt)
    assert last_exc is not None
    raise last_exc
