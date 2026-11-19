"""Executable backfill, hourly ingestion, and training pipeline entrypoints."""

from __future__ import annotations

from typing import Any

__all__ = [
    "bootstrap_if_needed",
    "run_backfill",
    "run_hourly_ingest",
    "run_scheduler_loop",
    "run_training",
]


def __getattr__(name: str) -> Any:
    if name == "bootstrap_if_needed":
        from skycast.pipelines.scheduler import bootstrap_if_needed

        return bootstrap_if_needed
    if name == "run_backfill":
        from skycast.pipelines.backfill import run_backfill

        return run_backfill
    if name == "run_hourly_ingest":
        from skycast.pipelines.hourly_ingest import run_hourly_ingest

        return run_hourly_ingest
    if name == "run_scheduler_loop":
        from skycast.pipelines.scheduler import run_scheduler_loop

        return run_scheduler_loop
    if name == "run_training":
        from skycast.pipelines.train import run_training

        return run_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
