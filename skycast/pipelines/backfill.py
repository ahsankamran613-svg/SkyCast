"""Historical backfill pipeline for Lahore weather, air quality, and engineered features."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd

from skycast.config.settings import Settings, get_settings
from skycast.data.fusion import fuse_weather_sources
from skycast.data.imputation import impute_weather_dataframe
from skycast.data.sources.met_no import MetNoError, fetch_met_no_forecast
from skycast.data.sources.open_meteo import (
    OpenMeteoError,
    fetch_open_meteo_forecast,
    fetch_open_meteo_historical,
)
from skycast.data.storage import SkyCastStorage
from skycast.features.engineer import engineer_features

logger = logging.getLogger(__name__)


def _to_utc_hour(value: datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").floor("h")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("h")


def _coalesce_open_meteo(
    historical_df: pd.DataFrame | None,
    recent_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if historical_df is None or historical_df.empty:
        return recent_df
    if recent_df is None or recent_df.empty:
        return historical_df

    merged = pd.merge(
        historical_df,
        recent_df,
        on="timestamp_utc",
        how="outer",
        suffixes=("", "__recent"),
    )
    for col in recent_df.columns:
        if col == "timestamp_utc":
            continue
        rec_col = f"{col}__recent"
        if rec_col in merged.columns:
            merged[col] = merged[rec_col].combine_first(merged[col])
            merged = merged.drop(columns=[rec_col])
    return merged.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)


def run_backfill(
    days: int = 90,
    *,
    end_utc: datetime | pd.Timestamp | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Backfill `days` of Lahore hourly weather & AQI through the current UTC hour."""
    if days < 1:
        raise ValueError("days must be >= 1.")

    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)
    now_hour = _to_utc_hour(end_utc)
    start_ts = (now_hour - pd.Timedelta(days=days)).floor("h")
    start_date = start_ts.date()
    end_date = now_hour.date()

    run_id = f"backfill-{now_hour.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    store.record_pipeline_run(
        run_id=run_id,
        pipeline_type="backfill",
        status="running",
        started_at_utc=started_at,
        metadata={"days": days, "start_utc": start_ts.isoformat(), "end_utc": now_hour.isoformat()},
    )

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)

    try:
        hist_df: pd.DataFrame | None = None
        recent_df: pd.DataFrame | None = None
        met_df: pd.DataFrame | None = None

        try:
            hist_df = fetch_open_meteo_historical(
                start_date,
                end_date,
                settings=cfg,
                client=http,
            )
        except OpenMeteoError as exc:
            logger.warning("Historical Open-Meteo fetch failed during backfill: %s", exc)

        try:
            recent_df = fetch_open_meteo_forecast(
                settings=cfg,
                past_days=2,
                forecast_days=1,
                client=http,
            )
        except OpenMeteoError as exc:
            logger.warning("Recent Open-Meteo forecast supplement failed: %s", exc)

        open_meteo_df = _coalesce_open_meteo(hist_df, recent_df)

        try:
            met_df = fetch_met_no_forecast(settings=cfg, client=http)
        except MetNoError as exc:
            logger.warning("Met.no fetch failed during backfill (continuing with Open-Meteo): %s", exc)

        if (open_meteo_df is None or open_meteo_df.empty) and (met_df is None or met_df.empty):
            raise RuntimeError("Both Open-Meteo and Met.no failed during backfill.")

        fused_df = fuse_weather_sources(
            open_meteo_df,
            met_df,
            settings=cfg,
            start_utc=start_ts,
            end_utc=now_hour,
            align_grid=True,
        )
        imputed_res = impute_weather_dataframe(fused_df, settings=cfg)
        featured_df = engineer_features(imputed_res.df, settings=cfg, include_targets=False)
        persist_summary = store.upsert_weather_features(featured_df)

        completed_at = datetime.now(timezone.utc)
        audit_dict = imputed_res.audit.to_dict()
        metadata = {
            "days": days,
            "rows_processed": int(len(featured_df)),
            "start_utc": start_ts.isoformat(),
            "end_utc": now_hour.isoformat(),
            "imputation_audit": audit_dict,
            "storage": persist_summary,
        }
        store.record_pipeline_run(
            run_id=run_id,
            pipeline_type="backfill",
            status="completed",
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            metadata=metadata,
        )
        return {
            "run_id": run_id,
            "status": "completed",
            **metadata,
        }
    except Exception as exc:
        store.record_pipeline_run(
            run_id=run_id,
            pipeline_type="backfill",
            status="failed",
            started_at_utc=started_at,
            completed_at_utc=datetime.now(timezone.utc),
            error_message=str(exc),
        )
        raise
    finally:
        if own_client:
            http.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Backfill SkyCast Lahore historical features.")
    parser.add_argument("--days", type=int, default=90, help="Number of past days to backfill.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local SQLite offline mode regardless of cloud env credentials.",
    )
    args = parser.parse_args(argv)

    cfg = get_settings()
    if args.offline:
        cfg = cfg.model_copy(update={"skycast_offline_mode": "true"})

    result = run_backfill(days=args.days, settings=cfg)
    logger.info(
        "Backfill completed: run_id=%s rows=%s mode=%s missing_before=%s missing_after=%s",
        result["run_id"],
        result["rows_processed"],
        result["storage"]["storage_mode"],
        result["imputation_audit"]["missing_before"],
        result["imputation_audit"]["missing_after"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
