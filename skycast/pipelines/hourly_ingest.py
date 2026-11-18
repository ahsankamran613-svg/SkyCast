"""Hourly ingestion pipeline: fetch, fuse, impute, engineer features, persist & invoke agents."""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd

from skycast.config.settings import POLLUTANT_COLUMNS, Settings, TARGET_METRICS, get_settings
from skycast.data.fusion import (
    CANONICAL_FUSED_COLUMNS,
    fuse_weather_sources,
    weighted_circular_fuse,
    weighted_linear_fuse,
)
from skycast.data.imputation import impute_weather_dataframe
from skycast.data.sources.met_no import MetNoError, fetch_met_no_forecast, wmo_code_to_symbol
from skycast.data.sources.open_meteo import OpenMeteoError, fetch_open_meteo_forecast
from skycast.data.storage import SkyCastStorage, _parse_json_or_default
from skycast.features.engineer import engineer_features
from skycast.pipelines._common import safe_record_pipeline_failure, to_utc_hour

logger = logging.getLogger(__name__)

_OPEN_METEO_RAW_COLS: tuple[str, ...] = (
    "temp_openmeteo",
    "humidity_openmeteo",
    "precip_openmeteo",
    "wind_speed_openmeteo",
    "wind_dir_openmeteo",
    "uv_openmeteo",
)
_MET_NO_RAW_COLS: tuple[str, ...] = (
    "temp_metno",
    "humidity_metno",
    "precip_metno",
    "wind_speed_metno",
    "wind_dir_metno",
    "uv_metno",
)
_OPEN_METEO_ONLY_METRIC_COLS: tuple[str, ...] = (
    "us_aqi",
    "precipitation_prob",
    *POLLUTANT_COLUMNS,
)
_PAIRED_BLEND_COLS: tuple[tuple[str, str, str, bool], ...] = (
    ("temperature_2m", "temp_openmeteo", "temp_metno", False),
    ("relative_humidity_2m", "humidity_openmeteo", "humidity_metno", False),
    ("precipitation_mm", "precip_openmeteo", "precip_metno", False),
    ("wind_speed_10m", "wind_speed_openmeteo", "wind_speed_metno", False),
    ("wind_direction_10m", "wind_dir_openmeteo", "wind_dir_metno", True),
)
_AUX_BLEND_COLS: tuple[str, ...] = (
    "apparent_temperature",
    "surface_pressure",
    "cloud_cover",
)


def _minimum_lookback_hours(settings: Settings) -> int:
    windows = (*settings.lag_hours, *settings.rolling_windows, *settings.roc_windows)
    return max(windows) if windows else 1


def _prepare_masked_stored_frame(
    stored_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex stored rows onto `hourly_index` and restore NaNs on previously-imputed cells."""
    norm = stored_df.copy()
    norm["timestamp_utc"] = pd.to_datetime(norm["timestamp_utc"], utc=True).dt.floor("h")
    norm = norm.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")
    reindexed = norm.set_index("timestamp_utc").reindex(hourly_index)

    for col in CANONICAL_FUSED_COLUMNS:
        if col == "timestamp_utc":
            continue
        if col not in reindexed.columns:
            if col == "sources_available":
                reindexed[col] = [[] for _ in range(len(hourly_index))]
            elif col == "condition_symbol":
                reindexed[col] = None
            else:
                reindexed[col] = np.nan

    sources_clean: list[list[str]] = []
    for raw_src in reindexed["sources_available"].tolist():
        parsed = _parse_json_or_default(raw_src, [])
        sources_clean.append([str(s) for s in parsed] if isinstance(parsed, list) else [])
    reindexed["sources_available"] = sources_clean

    raw_imputed = (
        reindexed["imputed_columns"].tolist()
        if "imputed_columns" in reindexed.columns
        else [[] for _ in range(len(hourly_index))]
    )
    for ts, imp_val in zip(hourly_index, raw_imputed):
        imp_cols = _parse_json_or_default(imp_val, [])
        if not isinstance(imp_cols, list) or not imp_cols:
            continue
        for col_name in imp_cols:
            col = str(col_name)
            if col in reindexed.columns and col not in {"timestamp_utc", "sources_available"}:
                reindexed.at[ts, col] = None if col == "condition_symbol" else np.nan
            if col == "weather_code" and "condition_symbol" in reindexed.columns:
                reindexed.at[ts, "condition_symbol"] = None

    reindexed = reindexed.reset_index().rename(columns={"index": "timestamp_utc"})
    return reindexed


def _merge_fresh_with_stored(
    stored_df: pd.DataFrame,
    fresh_om: pd.DataFrame | None,
    fresh_met: pd.DataFrame | None,
    *,
    settings: Settings,
    grid_start: pd.Timestamp,
    now_hour: pd.Timestamp,
) -> pd.DataFrame:
    """Merge fresh fused observations into stored history without overwriting imputation flags."""
    fresh_fused = fuse_weather_sources(
        fresh_om,
        fresh_met,
        settings=settings,
        start_utc=grid_start,
        end_utc=now_hour,
        align_grid=True,
    )
    if stored_df.empty:
        return fresh_fused

    hourly_index = pd.DatetimeIndex(fresh_fused["timestamp_utc"])
    stored_base = _prepare_masked_stored_frame(stored_df, hourly_index)
    out = stored_base[[*CANONICAL_FUSED_COLUMNS]].copy()

    w_om = settings.open_meteo_fusion_weight
    w_met = settings.met_no_fusion_weight

    fresh_has_om = fresh_fused["sources_available"].map(lambda s: "open_meteo" in s)
    fresh_has_met = fresh_fused["sources_available"].map(lambda s: "met_no" in s)
    stored_has_om = stored_base["sources_available"].map(lambda s: "open_meteo" in s)
    stored_has_met = stored_base["sources_available"].map(lambda s: "met_no" in s)

    # Full fresh overwrite where fresh covers everything or stored lacks the complementary source
    overwrite_full = (fresh_has_om & fresh_has_met) | (
        fresh_has_om & ~stored_has_met
    ) | (fresh_has_met & ~stored_has_om)
    if overwrite_full.any():
        for col in CANONICAL_FUSED_COLUMNS:
            if col == "timestamp_utc":
                continue
            out.loc[overwrite_full, col] = fresh_fused.loc[overwrite_full, col]

    # Case E: Fresh Open-Meteo arrived for rows that also retain historical Met.no data
    om_plus_stored_met = fresh_has_om & (~fresh_has_met) & stored_has_met
    if om_plus_stored_met.any():
        idx = om_plus_stored_met
        for col in (*_OPEN_METEO_RAW_COLS, *_OPEN_METEO_ONLY_METRIC_COLS):
            out.loc[idx, col] = fresh_fused.loc[idx, col]
        # Prefer fresh Open-Meteo actual UV over clear-sky fallback when present
        fresh_uv = fresh_fused.loc[idx, "uv_index"]
        out.loc[idx, "uv_index"] = fresh_uv.combine_first(stored_base.loc[idx, "uv_index"])
        # Prefer Open-Meteo weather_code & aligned condition_symbol
        fresh_wmo = fresh_fused.loc[idx, "weather_code"]
        out.loc[idx, "weather_code"] = fresh_wmo.combine_first(stored_base.loc[idx, "weather_code"])
        has_fresh_wmo = fresh_wmo.notna()
        if has_fresh_wmo.any():
            wmo_idx = fresh_wmo.index[has_fresh_wmo]
            out.loc[wmo_idx, "condition_symbol"] = fresh_wmo.loc[wmo_idx].map(wmo_code_to_symbol)
        # Re-blend paired raw columns cleanly using fresh OM + stored Met.no
        for target_col, om_col, met_col, is_circ in _PAIRED_BLEND_COLS:
            om_s = pd.to_numeric(out.loc[idx, om_col], errors="coerce")
            met_s = pd.to_numeric(out.loc[idx, met_col], errors="coerce")
            out.loc[idx, target_col] = (
                weighted_circular_fuse(om_s, met_s, w_om, w_met)
                if is_circ
                else weighted_linear_fuse(om_s, met_s, w_om, w_met)
            )
        # Blend aux columns ONLY if stored row did not already blend Open-Meteo + Met.no
        only_met_before = idx & (~stored_has_om)
        if only_met_before.any():
            for aux_col in _AUX_BLEND_COLS:
                om_aux = pd.to_numeric(fresh_fused.loc[only_met_before, aux_col], errors="coerce")
                met_aux = pd.to_numeric(stored_base.loc[only_met_before, aux_col], errors="coerce")
                out.loc[only_met_before, aux_col] = weighted_linear_fuse(om_aux, met_aux, w_om, w_met)
        out.loc[idx, "source_disagreement_temp"] = (
            pd.to_numeric(out.loc[idx, "temp_openmeteo"], errors="coerce")
            - pd.to_numeric(out.loc[idx, "temp_metno"], errors="coerce")
        ).abs()
        for row_i in out.index[idx]:
            out.at[row_i, "sources_available"] = ["open_meteo", "met_no"]

    # Case F: Fresh Met.no arrived while Open-Meteo is down, and stored row retains Open-Meteo data
    met_plus_stored_om = fresh_has_met & (~fresh_has_om) & stored_has_om
    if met_plus_stored_om.any():
        idx = met_plus_stored_om
        for col in _MET_NO_RAW_COLS:
            out.loc[idx, col] = fresh_fused.loc[idx, col]
        out.loc[idx, "uv_index"] = stored_base.loc[idx, "uv_index"].combine_first(
            fresh_fused.loc[idx, "uv_index"]
        )
        out.loc[idx, "weather_code"] = stored_base.loc[idx, "weather_code"].combine_first(
            fresh_fused.loc[idx, "weather_code"]
        )
        missing_sym = out.loc[idx, "condition_symbol"].isna()
        if missing_sym.any():
            sym_idx = missing_sym.index[missing_sym]
            out.loc[sym_idx, "condition_symbol"] = fresh_fused.loc[sym_idx, "condition_symbol"]
        for target_col, om_col, met_col, is_circ in _PAIRED_BLEND_COLS:
            om_s = pd.to_numeric(out.loc[idx, om_col], errors="coerce")
            met_s = pd.to_numeric(out.loc[idx, met_col], errors="coerce")
            out.loc[idx, target_col] = (
                weighted_circular_fuse(om_s, met_s, w_om, w_met)
                if is_circ
                else weighted_linear_fuse(om_s, met_s, w_om, w_met)
            )
        only_om_before = idx & (~stored_has_met)
        if only_om_before.any():
            for aux_col in _AUX_BLEND_COLS:
                om_aux = pd.to_numeric(stored_base.loc[only_om_before, aux_col], errors="coerce")
                met_aux = pd.to_numeric(fresh_fused.loc[only_om_before, aux_col], errors="coerce")
                out.loc[only_om_before, aux_col] = weighted_linear_fuse(om_aux, met_aux, w_om, w_met)
        out.loc[idx, "source_disagreement_temp"] = (
            pd.to_numeric(out.loc[idx, "temp_openmeteo"], errors="coerce")
            - pd.to_numeric(out.loc[idx, "temp_metno"], errors="coerce")
        ).abs()
        for row_i in out.index[idx]:
            out.at[row_i, "sources_available"] = ["open_meteo", "met_no"]

    return out


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
    except ModuleNotFoundError as exc:
        if exc.name in {"skycast.agents", "skycast.agents.graph"}:
            logger.info("LangGraph agent graph not yet implemented; skipping agent step.")
            return {"status": "not_available"}
        raise

    try:
        result = run_agent_pipeline(
            run_id=run_id,
            latest_features=latest_row,
            settings=settings,
            storage=storage,
        )
        return {"status": "completed", "result": result}
    except Exception as exc:
        logger.exception("Agent pipeline execution failed for run_id=%s: %s", run_id, exc)
        return {"status": "failed", "error": str(exc)}


def run_hourly_ingest(
    *,
    lookback_hours: int | None = None,
    reference_utc: datetime | pd.Timestamp | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    client: httpx.Client | None = None,
    run_agents: bool = True,
) -> dict[str, Any]:
    """Fetch newest Lahore observations, fuse, impute, engineer features, persist, and run agents."""
    cfg = settings or get_settings()
    resolved_lookback = cfg.hourly_lookback_hours if lookback_hours is None else lookback_hours
    min_lookback = _minimum_lookback_hours(cfg)
    if resolved_lookback < min_lookback:
        raise ValueError(
            f"lookback_hours ({resolved_lookback}) must be >= {min_lookback} "
            "to populate configured lag/rolling/RoC features."
        )

    store = storage or SkyCastStorage(cfg)
    now_hour = to_utc_hour(reference_utc)
    window_start = now_hour - pd.Timedelta(hours=resolved_lookback)

    run_id = f"hourly-{now_hour.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    started_at = datetime.now(timezone.utc)
    store.record_pipeline_run(
        run_id=run_id,
        pipeline_type="hourly_ingest",
        status="running",
        started_at_utc=started_at,
        metadata={"reference_utc": now_hour.isoformat(), "lookback_hours": resolved_lookback},
    )

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)

    try:
        stored_df = store.read_weather_features(
            start_utc=window_start,
            end_utc=now_hour,
            expand_features_json=False,
        )

        past_days = max(2, min(92, math.ceil(resolved_lookback / 24) + 1))
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

        om_has_rows = (
            fresh_om is not None and not fresh_om.loc[fresh_om["timestamp_utc"] <= now_hour].empty
        )
        met_has_rows = (
            fresh_met is not None and not fresh_met.loc[fresh_met["timestamp_utc"] <= now_hour].empty
        )
        if not om_has_rows and not met_has_rows:
            raise RuntimeError(
                "Both Open-Meteo and Met.no failed to return observations for the current hourly window."
            )

        candidate_starts: list[pd.Timestamp] = []
        if not stored_df.empty:
            candidate_starts.append(pd.to_datetime(stored_df["timestamp_utc"], utc=True).min())
        if fresh_om is not None and not fresh_om.empty:
            candidate_starts.append(pd.to_datetime(fresh_om["timestamp_utc"], utc=True).min())
        if fresh_met is not None and not fresh_met.empty:
            candidate_starts.append(pd.to_datetime(fresh_met["timestamp_utc"], utc=True).min())
        grid_start = (
            max(window_start, min(candidate_starts)) if candidate_starts else window_start
        )

        fused_df = _merge_fresh_with_stored(
            stored_df,
            fresh_om,
            fresh_met,
            settings=cfg,
            grid_start=grid_start,
            now_hour=now_hour,
        )
        imputed_res = impute_weather_dataframe(fused_df, settings=cfg)
        featured_df = engineer_features(imputed_res.df, settings=cfg, include_targets=False)

        if stored_df.empty:
            to_persist = featured_df
        else:
            upsert_cutoff = now_hour - pd.Timedelta(hours=cfg.hourly_upsert_cutoff_hours)
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
        safe_record_pipeline_failure(
            store,
            run_id=run_id,
            pipeline_type="hourly_ingest",
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
    min_lookback = _minimum_lookback_hours(cfg)
    parser = argparse.ArgumentParser(description="Run hourly SkyCast Lahore ingestion pipeline.")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=cfg.hourly_lookback_hours,
        help=f"Lookback context window in hours (minimum {min_lookback}).",
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
