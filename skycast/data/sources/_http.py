"""Shared HTTP JSON retry helper with Retry-After and exponential backoff support."""

from __future__ import annotations

import email.utils
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import httpx

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def parse_retry_after(
    header_value: str | None,
    *,
    now: datetime | None = None,
    max_seconds: float = 60.0,
) -> float:
    """Parse an HTTP `Retry-After` header (seconds or HTTP-date) into bounded seconds."""
    if not header_value:
        return 0.0
    raw = header_value.strip()
    if not raw:
        return 0.0
    try:
        seconds = float(raw)
        if seconds != seconds:  # NaN
            return 0.0
        return max(0.0, min(seconds, max_seconds))
    except ValueError:
        pass

    try:
        target = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return 0.0
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    delta = (target - ref).total_seconds()
    return max(0.0, min(delta, max_seconds))


def request_json_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    error_cls: type[RuntimeError] = RuntimeError,
    source_name: str = "HTTP API",
    max_retries: int = 2,
    backoff_seconds: float = 0.4,
    max_retry_after_seconds: float = 60.0,
    validate_payload: Callable[[dict[str, Any], str], None] | None = None,
) -> dict[str, Any]:
    """Execute a GET request expecting a JSON object, honouring Retry-After and backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, params=params, headers=headers)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                retry_after = parse_retry_after(
                    response.headers.get("Retry-After"),
                    max_seconds=max_retry_after_seconds,
                )
                exp_backoff = backoff_seconds * (2**attempt) if backoff_seconds > 0 else 0.0
                delay = max(exp_backoff, retry_after)
                if delay > 0:
                    time.sleep(delay)
                continue

            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise error_cls(
                    f"Expected JSON object from {source_name} ({url}), got {type(payload).__name__}"
                )
            if validate_payload is not None:
                validate_payload(payload, url)
            return payload
        except error_cls:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_retries:
                if backoff_seconds > 0:
                    time.sleep(backoff_seconds * (2**attempt))
                continue
            raise error_cls(f"{source_name} network error for {url}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise error_cls(
                f"{source_name} HTTP {exc.response.status_code} for {url}: "
                f"{exc.response.text[:300]}"
            ) from exc
        except ValueError as exc:
            raise error_cls(f"Invalid JSON from {source_name} ({url}): {exc}") from exc

    raise error_cls(f"{source_name} request failed for {url}: {last_exc}")
