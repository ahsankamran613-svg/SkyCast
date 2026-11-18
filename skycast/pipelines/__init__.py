"""Executable backfill, hourly ingestion, and training pipeline entrypoints."""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_backfill",
    "run_hourly_ingest",
]


def __getattr__(name: str) -> Any:
    if name == "run_backfill":
        from skycast.pipelines.backfill import run_backfill

        return run_backfill
    if name == "run_hourly_ingest":
        from skycast.pipelines.hourly_ingest import run_hourly_ingest

        return run_hourly_ingest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
