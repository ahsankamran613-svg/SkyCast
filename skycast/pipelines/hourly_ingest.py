"""Hourly ingestion pipeline: fetch, fuse, impute, engineer features, persist & invoke agents."""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import pandas as pd

from skycast.config.settings import POLLUTANT_COLUMNS, Settings, TARGET_METRICS, get_settings
from skycast.data.fusion import fuse_weather_sources
from skycast.data.imputation import impute_weather_dataframe
from skycast.data.sources.met_no import MetNoError, fetch_met_no_forecast
from skycast.data.sources.open_meteo import OpenMeteoError, fetch_open_meteo_forecast
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


def _stored_history_to_open_meteo_context(stored_df: pd.DataFrame) -> pd.DataFrame:
    """Convert stored fused history into Open-Meteo-shaped lookback context rows."""
    if stored_df.empty:
        return pd.DataFrame()
    mapping = {
        "timestamp_utc": "timestamp_utc",
        "temperature_2m": "temperature_2m",
        "apparent_temperature": "apparent_temperature",
        "weather_code": "weather_code",
        "relative_humidity_2m": "relative_humidity_2m",
        "precipitation_mm": "precipitation",
        "precipitation_prob": "precipitation_probability",
        "uv_index": "uv_index",
        "wind_speed_10m": "wind_speed_10m",
        "wind_direction_10m": "wind_direction_10m",
        "surface_pressure": "surface_pressure",
        "cloud_cover": "cloud_cover",
        "us_aqi": "us_aqi",
        **{p: p for p in POLLUTANT_COLUMNS},
    }
    cols = {target: stored_df[src] for src, target in mapping.items() if src in stored_df.columns}
    # Prefer raw open_meteo readings when stored so fusion does not double-blend Met.no history
    raw_overrides = {
        "temp_openmeteo": "temperature_2m",
        "humidity_openmeteo": "relative_humidity_2m",
        "precip_openmeteo": "precipitation",
        "wind_speed_openmeteo": "wind_speed_10m",
        "wind_dir_openmeteo": "wind_direction_10m",
        "uv_openmeteo": "uv_index",
    }
    out = pd.DataFrame(cols)
    for raw_col, target in raw_overrides.items():
        if raw_col in stored_df.columns and target in out.columns:
            raw_s = pd.to_numeric(stored_df[raw_col], errors="coerce")
            out[target] = raw_s.combine_first(pd.to_numeric(out[target], errors="coerce"))
    return out


def _coalesce_fresh_with_context(
    fresh_om: pd.DataFrame | None,
    stored_df: pd.DataFrame,
) -> pd.DataFrame | None:
    context_df = _stored_history_to_open_meteo_context(stored_df)
    if fresh_om is None or fresh_om.empty:
        return None if context_df.empty else context_df
    if context_df.empty:
        return fresh_om

    merged = pd.merge(
        fresh_om,
        context_df,
        on="timestamp_utc",
        how="outer",
        suffixes=("", "__ctx"),
    )
    for col in context_df.columns:
        if col == "timestamp_utc":
            continue
        ctx_col = f"{col}__ctx"
        if ctx_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[ctx_col])
            merged = merged.drop(columns=[ctx_col])
    return merged.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)


def _maybe_run_agent_pipeline(
    *,
    run_id: str,
    latest_row: dict[str, Any],
    settings: Settings,
    storage: SkyCastStorage,
) -> dict[str, Any]:
    """Invoke the Component 4 LangGraph agent workflow if available."""
    try:
        from skycast.agents.graph import run_agent_pipeline
    except ImportError:
        logger.info("LangGraph agent graph not yet implemented; skipping agent step.")
        return {"status": "not_available"}

    result = run_agent_pipeline(
        run_id=run_id,
        latest_features=latest_row,
        settings=settings,
        storage=storage,
    )
    return {"status": "completed", "result": result}


def run_hourly_ingest(
    *,
    lookback_hours: int = 168,
    reference_utc: datetime | pd.Timestamp | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    client: httpx.Client | None = None,
    run_agents: bool = True,
) -> dict[str, Any]:
    """Fetch newest Lahore observations, fuse, impute, engineer features, persist, and run agents."""
    if lookback_hours < 48:
        raise ValueError("lookback_hours must be >= 48 so lag_48h features are populated.")

    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)
    now_hour = _to_utc_hour(reference_utc)
    window_start = now_hour - pd.Timedelta(hours=lookback_hours)

    run_id = f"hourly-{now_hour.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    store.record_pipeline_run(
        run_id=run_id,
        pipeline_type="hourly_ingest",
        status="running",
        started_at_utc=started_at,
        metadata={"reference_utc": now_hour.isoformat(), "lookback_hours": lookback_hours},
    )

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)

    try:
        stored_df = store.read_weather_features(
            start_utc=window_start,
            end_utc=now_hour,
            expand_features_json=False,
        )

        past_days = max(2, min(92, math.ceil(lookback_hours / 24) + 1))
        fresh_om: pd.DataFrame | None = None
        fresh_met: pd.DataFrame | None = None

        try:
            fresh_om = fetch_open_meteo_forecast(
                settings=cfg,
                past_days=past_days,
                forecast_days=1,
                client=http,
            )
        except OpenMeteoError as exc:
            logger.warning("Open-Meteo hourly fetch failed: %s", exc)

        try:
            fresh_met = fetch_met_no_forecast(settings=cfg, client=http)
        except MetNoError as exc:
            logger.warning("Met.no hourly fetch failed: %s", exc)

        om_has_rows = fresh_om is not None and not fresh_om.loc[fresh_om["timestamp_utc"] <= now_hour].empty
        met_has_rows = fresh_met is not None and not fresh_met.loc[fresh_met["timestamp_utc"] <= now_hour].empty
        if not om_has_rows and not met_has_rows:
            raise RuntimeError(
                "Both Open-Meteo and Met.no failed to return observations for the current hourly window."
            )

        om_combined = _coalesce_fresh_with_context(fresh_om, stored_df)
        grid_start = window_start
        if stored_df.empty and om_combined is not None and not om_combined.empty:
            grid_start = max(window_start, pd.to_datetime(om_combined["timestamp_utc"], utc=True).min())

        fused_df = fuse_weather_sources(
            om_combined,
            fresh_met,
            settings=cfg,
            start_utc=grid_start,
            end_utc=now_hour,
            align_grid=True,
        )
        imputed_res = impute_weather_dataframe(fused_df, settings=cfg)
        featured_df = engineer_features(imputed_res.df, settings=cfg, include_targets=False)

        if stored_df.empty:
            to_persist = featured_df
        else:
            upsert_cutoff = now_hour - pd.Timedelta(hours=48)
            to_persist = featured_df.loc[featured_df["timestamp_utc"] >= upsert_cutoff]
            if to_persist.empty:
                to_persist = featured_df.tail(1)

        persist_summary = store.upsert_weather_features(to_persist)

        latest_row = featured_df.iloc[-1].to_dict()
        agent_summary: dict[str, Any] = {"status": "skipped"}
        if run_agents:
            agent_summary = _maybe_run_agent_pipeline(
                run_id=run_id,
                latest_row=latest_row,
                settings=cfg,
                storage=store,
            )

        latest_snapshot = {
            "timestamp_utc": pd.Timestamp(latest_row["timestamp_utc"]).isoformat(),
            **{m: float(latest_row[m]) for m in TARGET_METRICS if pd.notna(latest_row.get(m))},
            "sources_available": latest_row.get("sources_available", []),
            "is_imputed": bool(latest_row.get("is_imputed", False)),
        }
        metadata = {
            "reference_utc": now_hour.isoformat(),
            "rows_evaluated": int(len(featured_df)),
            "rows_persisted": int(len(to_persist)),
            "imputation_audit": imputed_res.audit.to_dict(),
            "storage": persist_summary,
            "latest_observation": latest_snapshot,
            "agents": agent_summary,
        }
        store.record_pipeline_run(
            run_id=run_id,
            pipeline_type="hourly_ingest",
            status="completed",
            started_at_utc=started_at,
            completed_at_utc=datetime.now(timezone.utc),
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
            pipeline_type="hourly_ingest",
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
    parser = argparse.ArgumentParser(description="Run hourly SkyCast Lahore ingestion pipeline.")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=168,
        help="Lookback context window in hours (minimum 48).",
    )
    parser.add_argument(
        "--skip-agents",
        action="store_true",
        help="Skip invoking the LangGraph agent workflow after ingestion.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local SQLite offline mode regardless of cloud env credentials.",
    )
    args = parser.parse_args(argv)

    cfg = get_settings()
    if args.offline:
        cfg = cfg.model_copy(update={"skycast_offline_mode": "true"})

    result = run_hourly_ingest(
        lookback_hours=args.lookback_hours,
        settings=cfg,
        run_agents=not args.skip_agents,
    )
    logger.info(
        "Hourly ingest completed: run_id=%s persisted=%s latest=%s",
        result["run_id"],
        result["rows_persisted"],
        result["latest_observation"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
