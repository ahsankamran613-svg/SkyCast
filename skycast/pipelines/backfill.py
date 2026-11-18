"""Historical backfill pipeline for Lahore weather, air quality, and engineered features."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd

from skycast.config.settings import Settings, TARGET_METRICS, get_settings
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
from skycast.pipelines._common import safe_record_pipeline_failure, to_utc_hour

logger = logging.getLogger(__name__)


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
    days: int | None = None,
    *,
    end_utc: datetime | pd.Timestamp | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    client: httpx.Client | None = None,
    allow_extended_days: bool = False,
    allow_high_imputation: bool = False,
) -> dict[str, Any]:
    """Backfill `days` of Lahore hourly weather & AQI through the current UTC hour."""
    cfg = settings or get_settings()
    resolved_days = cfg.backfill_default_days if days is None else days
    if resolved_days < 1:
        raise ValueError("days must be >= 1.")
    if not allow_extended_days and resolved_days > cfg.backfill_max_days:
        raise ValueError(
            f"days ({resolved_days}) exceeds backfill_max_days ({cfg.backfill_max_days}); "
            "Open-Meteo Air Quality archive has limited history. Pass allow_extended_days=True to override."
        )

    store = storage or SkyCastStorage(cfg)
    now_hour = to_utc_hour(end_utc)
    start_ts = (now_hour - pd.Timedelta(days=resolved_days)).floor("h")
    start_date = start_ts.date()
    end_date = now_hour.date()

    run_id = f"backfill-{now_hour.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    store.record_pipeline_run(
        run_id=run_id,
        pipeline_type="backfill",
        status="running",
        started_at_utc=started_at,
        metadata={
            "days": resolved_days,
            "start_utc": start_ts.isoformat(),
            "end_utc": now_hour.isoformat(),
        },
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
            logger.warning(
                "Met.no fetch failed during backfill (continuing with Open-Meteo): %s", exc
            )

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

        if not allow_high_imputation:
            constant_targets = [
                m for m in TARGET_METRICS if m in imputed_res.audit.constant_fallback_columns
            ]
            if constant_targets:
                raise RuntimeError(
                    f"Backfill aborted: target metric(s) {constant_targets} had no upstream "
                    "observations and fell back to physical constants."
                )
            if imputed_res.audit.imputation_rate_pct > cfg.max_imputation_rate_pct:
                raise RuntimeError(
                    f"Backfill aborted: overall imputation rate "
                    f"{imputed_res.audit.imputation_rate_pct:.2f}% exceeds "
                    f"max_imputation_rate_pct ({cfg.max_imputation_rate_pct:.2f}%)."
                )

        featured_df = engineer_features(imputed_res.df, settings=cfg, include_targets=False)
        persist_summary = store.upsert_weather_features(featured_df)

        completed_at = datetime.now(timezone.utc)
        audit_dict = imputed_res.audit.to_dict()
        metadata = {
            "days": resolved_days,
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
        safe_record_pipeline_failure(
            store,
            run_id=run_id,
            pipeline_type="backfill",
            started_at_utc=started_at,
            error=exc,
            logger=logger,
        )
        raise
    finally:
        if own_client:
            http.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Backfill SkyCast Lahore historical features.")
    parser.add_argument(
        "--days",
        type=int,
        default=cfg.backfill_default_days,
        help=f"Number of past days to backfill (1..{cfg.backfill_max_days}).",
    )
    parser.add_argument(
        "--allow-extended-days",
        action="store_true",
        help="Allow backfilling beyond backfill_max_days (AQ history may be missing).",
    )
    parser.add_argument(
        "--allow-high-imputation",
        action="store_true",
        help="Persist backfill even if imputation exceeds quality thresholds.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local SQLite offline mode regardless of cloud env credentials.",
    )
    args = parser.parse_args(argv)

    if args.offline:
        cfg = cfg.model_copy(update={"skycast_offline_mode": "true"})

    result = run_backfill(
        days=args.days,
        settings=cfg,
        allow_extended_days=args.allow_extended_days,
        allow_high_imputation=args.allow_high_imputation,
    )
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
