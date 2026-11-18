"""Shared timestamp and pipeline-run lifecycle helpers for ingestion pipelines."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from skycast.data.storage import SkyCastStorage


def to_utc_hour(value: datetime | pd.Timestamp | None) -> pd.Timestamp:
    """Normalize a timestamp (or current UTC time) down to the exact UTC hour."""
    if value is None:
        return pd.Timestamp.now(tz="UTC").floor("h")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("h")


def safe_record_pipeline_failure(
    storage: SkyCastStorage,
    *,
    run_id: str,
    pipeline_type: str,
    started_at_utc: datetime,
    error: Exception,
    logger: logging.Logger,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort failure audit logging that never masks the original pipeline exception."""
    try:
        storage.record_pipeline_run(
            run_id=run_id,
            pipeline_type=pipeline_type,
            status="failed",
            started_at_utc=started_at_utc,
            completed_at_utc=datetime.now(timezone.utc),
            metadata=metadata,
            error_message=str(error),
        )
    except Exception as log_exc:
        logger.exception(
            "Failed to record pipeline failure status for %s (%s): %s",
            run_id,
            pipeline_type,
            log_exc,
        )
