"""Causal feature engineering (35+ temporal, cyclical, rolling, lag, interaction) & 24h targets."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from skycast.config.settings import (
    POLLUTANT_COLUMNS,
    Settings,
    TARGET_METRICS,
    get_settings,
)

ENGINEERED_SERIES_FOR_TEMPORAL_DYNAMICS: tuple[str, ...] = (
    "us_aqi",
    "temperature_2m",
    "precipitation_mm",
    "precipitation_prob",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "surface_pressure",
    "pm2_5",
)

TOP_LEVEL_ENGINEERED_COLUMNS: tuple[str, ...] = (
    "wind_dir_sin",
    "wind_dir_cos",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_lahore_smog_season",
    "is_monsoon_season",
)

BASE_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "us_aqi",
    "temperature_2m",
    "precipitation_mm",
    "precipitation_prob",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "apparent_temperature",
    "surface_pressure",
    "cloud_cover",
    "weather_code",
    *POLLUTANT_COLUMNS,
    "source_disagreement_temp",
    "source_count",
    "is_dual_source",
    "wind_dir_sin",
    "wind_dir_cos",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_lahore_smog_season",
    "is_monsoon_season",
    "wind_u",
    "wind_v",
    "temp_apparent_delta",
    "heat_stress_index",
    "pm25_pm10_ratio",
    "aqi_wind_dispersion",
    "temp_diurnal_anomaly",
    "aqi_diurnal_anomaly",
)


def get_dynamic_feature_names(settings: Settings | None = None) -> list[str]:
    """Return ordered list of RoC, rolling, and lag feature column names."""
    cfg = settings or get_settings()
    names: list[str] = []
    for col in ENGINEERED_SERIES_FOR_TEMPORAL_DYNAMICS:
        for k in cfg.roc_windows:
            names.append(f"{col}_roc_{k}h")
        for w in cfg.rolling_windows:
            names.extend(
                [
                    f"{col}_roll_{w}h_mean",
                    f"{col}_roll_{w}h_std",
                    f"{col}_roll_{w}h_min",
                    f"{col}_roll_{w}h_max",
                ]
            )
        for h in cfg.lag_hours:
            names.append(f"{col}_lag_{h}h")
    return names


def get_model_feature_columns(
    df: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Return all deterministic causal numeric/boolean feature columns suitable for ML."""
    expected = [*BASE_MODEL_FEATURE_COLUMNS, *get_dynamic_feature_names(settings)]
    if df is None:
        return expected
    return [col for col in expected if col in df.columns]


def _safe_series(df: pd.DataFrame, column: str, default: float = float("nan")) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").astype("float64")
    return pd.Series(default, index=df.index, dtype="float64")


def _align_to_hourly_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex | None]:
    """Reindex a sorted UTC DataFrame onto a contiguous 1h grid if gaps exist."""
    if df.empty or "timestamp_utc" not in df.columns:
        return df, None
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True).dt.floor("h")
    out = out.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")
    orig_ts = pd.DatetimeIndex(out["timestamp_utc"])
    full_grid = pd.date_range(start=orig_ts.min(), end=orig_ts.max(), freq="1h", tz="UTC")
    if len(full_grid) == len(orig_ts):
        return out.reset_index(drop=True), orig_ts
    grid_df = (
        out.set_index("timestamp_utc")
        .reindex(full_grid)
        .rename_axis("timestamp_utc")
        .reset_index()
    )
    return grid_df, orig_ts


def _compute_source_indicators(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "sources_available" in df.columns:
        counts: list[float] = []
        for val in df["sources_available"].tolist():
            if isinstance(val, (list, tuple, set, np.ndarray)):
                counts.append(float(min(len(val), 2)))
            elif isinstance(val, str):
                stripped = val.strip()
                if stripped in {"", "[]", "{}"}:
                    counts.append(0.0)
                else:
                    counts.append(float(min(len([s for s in stripped.strip("[]{}").split(",") if s.strip()]), 2)))
            else:
                counts.append(0.0)
        count_s = pd.Series(counts, index=df.index, dtype="float64")
        dual_s = (count_s >= 2.0).astype("float64")
        return count_s, dual_s

    if "temp_openmeteo" in df.columns and "temp_metno" in df.columns:
        has_om = pd.to_numeric(df["temp_openmeteo"], errors="coerce").notna()
        has_met = pd.to_numeric(df["temp_metno"], errors="coerce").notna()
        count_s = (has_om.astype("float64") + has_met.astype("float64"))
        dual_s = (has_om & has_met).astype("float64")
        return count_s, dual_s

    count_s = pd.Series(1.0, index=df.index, dtype="float64")
    dual_s = pd.Series(0.0, index=df.index, dtype="float64")
    return count_s, dual_s


def add_forecast_targets(
    df: pd.DataFrame,
    settings: Settings | None = None,
    *,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Append causal forward-shifted targets (`target_{metric}_t{h}`) without modifying features.

    Rows in the newest `max(horizons)` hours lack complete future targets and get
    `has_full_24h_targets = False` so training splits exclude them cleanly.
    """
    if df.empty:
        out = df.copy()
        out["has_full_24h_targets"] = pd.Series(dtype="bool")
        return out

    cfg = settings or get_settings()
    horizon_list = (
        tuple(horizons)
        if horizons is not None
        else tuple(range(1, cfg.forecast_horizon_hours + 1))
    )

    grid_df, orig_ts = _align_to_hourly_grid(df)
    target_cols: dict[str, pd.Series] = {}

    for metric in TARGET_METRICS:
        base_series = _safe_series(grid_df, metric, default=float("nan"))
        for h in horizon_list:
            target_cols[f"target_{metric}_t{h}"] = base_series.shift(-h)

    if "wind_dir_sin" in grid_df.columns and "wind_dir_cos" in grid_df.columns:
        sin_s = _safe_series(grid_df, "wind_dir_sin", default=float("nan"))
        cos_s = _safe_series(grid_df, "wind_dir_cos", default=float("nan"))
    else:
        wd_rad = np.deg2rad(_safe_series(grid_df, "wind_direction_10m", default=float("nan")))
        sin_s = pd.Series(np.sin(wd_rad), index=grid_df.index, dtype="float64")
        cos_s = pd.Series(np.cos(wd_rad), index=grid_df.index, dtype="float64")

    for h in horizon_list:
        target_cols[f"target_wind_dir_sin_t{h}"] = sin_s.shift(-h)
        target_cols[f"target_wind_dir_cos_t{h}"] = cos_s.shift(-h)

    full_mask = pd.Series(True, index=grid_df.index, dtype="bool")
    for metric in TARGET_METRICS:
        for h in horizon_list:
            full_mask &= target_cols[f"target_{metric}_t{h}"].notna()
    target_cols["has_full_24h_targets"] = full_mask

    existing_clean = grid_df.drop(
        columns=[c for c in target_cols if c in grid_df.columns], errors="ignore"
    )
    out = pd.concat([existing_clean, pd.DataFrame(target_cols, index=grid_df.index)], axis=1)
    if orig_ts is not None and len(out) != len(orig_ts):
        out = out.loc[out["timestamp_utc"].isin(orig_ts)].reset_index(drop=True)
    return out


def engineer_features(
    df: pd.DataFrame,
    settings: Settings | None = None,
    *,
    include_targets: bool = False,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Compute cyclical time, circular wind, RoC, rolling, lag, and interaction features."""
    if df.empty:
        out = df.copy()
        if include_targets:
            return add_forecast_targets(out, settings=settings, horizons=horizons)
        return out

    if "timestamp_utc" not in df.columns:
        raise ValueError("DataFrame must include 'timestamp_utc' for feature engineering.")

    cfg = settings or get_settings()
    grid_df, orig_ts = _align_to_hourly_grid(df)

    # Local Lahore time (Asia/Karachi, UTC+5) for physically aligned diurnal & seasonal signals
    ts_utc = pd.DatetimeIndex(grid_df["timestamp_utc"])
    ts_local = ts_utc.tz_convert(cfg.lahore_timezone)

    hours = ts_local.hour.to_numpy(dtype="float64")
    dows = ts_local.dayofweek.to_numpy(dtype="float64")
    months = ts_local.month.to_numpy(dtype="float64")

    wind_dir_deg = _safe_series(grid_df, "wind_direction_10m")
    wind_rad = np.deg2rad(wind_dir_deg)
    wind_dir_sin = pd.Series(np.sin(wind_rad), index=grid_df.index, dtype="float64")
    wind_dir_cos = pd.Series(np.cos(wind_rad), index=grid_df.index, dtype="float64")

    source_count_s, is_dual_source_s = _compute_source_indicators(grid_df)
    disagreement_s = _safe_series(grid_df, "source_disagreement_temp").fillna(0.0)

    new_cols: dict[str, Any] = {
        "source_disagreement_temp": disagreement_s,
        "source_count": source_count_s,
        "is_dual_source": is_dual_source_s,
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
        "hour_sin": np.sin(2.0 * np.pi * hours / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hours / 24.0),
        "day_of_week_sin": np.sin(2.0 * np.pi * dows / 7.0),
        "day_of_week_cos": np.cos(2.0 * np.pi * dows / 7.0),
        "month_sin": np.sin(2.0 * np.pi * (months - 1.0) / 12.0),
        "month_cos": np.cos(2.0 * np.pi * (months - 1.0) / 12.0),
        "is_weekend": pd.Series(ts_local.dayofweek >= 5, index=grid_df.index, dtype="bool"),
        "is_lahore_smog_season": pd.Series(
            np.isin(ts_local.month, cfg.lahore_smog_months), index=grid_df.index, dtype="bool"
        ),
        "is_monsoon_season": pd.Series(
            np.isin(ts_local.month, cfg.lahore_monsoon_months), index=grid_df.index, dtype="bool"
        ),
    }

    temp_s = _safe_series(grid_df, "temperature_2m")
    app_s = _safe_series(grid_df, "apparent_temperature")
    rh_s = _safe_series(grid_df, "relative_humidity_2m")
    wind_spd_s = _safe_series(grid_df, "wind_speed_10m")
    us_aqi_s = _safe_series(grid_df, "us_aqi")
    pm25_s = _safe_series(grid_df, "pm2_5")
    pm10_s = _safe_series(grid_df, "pm10")

    # Physical & Lahore-specific interaction features
    new_cols["wind_u"] = -wind_spd_s * wind_dir_sin
    new_cols["wind_v"] = -wind_spd_s * wind_dir_cos
    new_cols["temp_apparent_delta"] = app_s - temp_s
    new_cols["heat_stress_index"] = temp_s * (rh_s / 100.0)
    new_cols["pm25_pm10_ratio"] = pm25_s / pm10_s.clip(lower=1.0)
    new_cols["aqi_wind_dispersion"] = us_aqi_s / (wind_spd_s + 1.0)

    # Strictly causal Rate-of-Change, Rolling, and Lag features on contiguous 1h grid
    series_map: dict[str, pd.Series] = {
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
    }
    for col in ENGINEERED_SERIES_FOR_TEMPORAL_DYNAMICS:
        if col not in series_map:
            series_map[col] = _safe_series(grid_df, col)

    for col, s in series_map.items():
        for k in cfg.roc_windows:
            new_cols[f"{col}_roc_{k}h"] = s - s.shift(k)

        for w in cfg.rolling_windows:
            roll = s.rolling(window=w, min_periods=1)
            new_cols[f"{col}_roll_{w}h_mean"] = roll.mean()
            new_cols[f"{col}_roll_{w}h_std"] = roll.std(ddof=0).fillna(0.0)
            new_cols[f"{col}_roll_{w}h_min"] = roll.min()
            new_cols[f"{col}_roll_{w}h_max"] = roll.max()

        for h in cfg.lag_hours:
            new_cols[f"{col}_lag_{h}h"] = s.shift(h)

    new_cols["temp_diurnal_anomaly"] = temp_s - new_cols["temperature_2m_roll_24h_mean"]
    new_cols["aqi_diurnal_anomaly"] = us_aqi_s - new_cols["us_aqi_roll_24h_mean"]

    base_clean = grid_df.drop(columns=[c for c in new_cols if c in grid_df.columns], errors="ignore")
    featured = pd.concat([base_clean, pd.DataFrame(new_cols, index=grid_df.index)], axis=1)

    if include_targets:
        featured = add_forecast_targets(featured, settings=cfg, horizons=horizons)

    if orig_ts is not None and len(featured) != len(orig_ts):
        featured = featured.loc[featured["timestamp_utc"].isin(orig_ts)].reset_index(drop=True)

    return featured


def split_training_and_inference_rows(
    df: pd.DataFrame,
    settings: Settings | None = None,
    *,
    exclude_tier3_imputed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an engineered+targeted DataFrame into trainable rows and latest forecast-only rows."""
    if df.empty:
        return df.copy(), df.copy()
    if "has_full_24h_targets" not in df.columns:
        raise ValueError("DataFrame must include targets (`include_targets=True`) before splitting.")

    cfg = settings or get_settings()
    feature_cols = get_model_feature_columns(df, settings=cfg)
    has_targets = df["has_full_24h_targets"].astype(bool)
    features_complete = ~df[feature_cols].isna().any(axis=1)
    trainable_mask = has_targets & features_complete

    if exclude_tier3_imputed and "imputation_tier" in df.columns:
        is_tier3 = df["imputation_tier"].astype(str) == "tier3_climatology_ffill"
        horizon = cfg.forecast_horizon_hours
        if "timestamp_utc" in df.columns and not df.empty:
            ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("h")
            full_grid = pd.date_range(start=ts_utc.min(), end=ts_utc.max(), freq="1h", tz="UTC")
            if len(full_grid) != len(ts_utc):
                grid_tier3 = (
                    pd.Series(is_tier3.to_numpy(dtype=bool), index=ts_utc)
                    .groupby(level=0)
                    .any()
                    .reindex(full_grid, fill_value=False)
                )
                grid_window = grid_tier3.copy()
                for h in range(1, horizon + 1):
                    grid_window |= grid_tier3.shift(-h, fill_value=False)
                tier3_window = pd.Series(
                    grid_window.reindex(ts_utc, fill_value=False).to_numpy(dtype=bool),
                    index=df.index,
                )
            else:
                tier3_window = is_tier3.copy()
                for h in range(1, horizon + 1):
                    tier3_window |= is_tier3.shift(-h, fill_value=False)
        else:
            tier3_window = is_tier3.copy()
            for h in range(1, horizon + 1):
                tier3_window |= is_tier3.shift(-h, fill_value=False)
        trainable_mask &= ~tier3_window

    train_df = df.loc[trainable_mask].reset_index(drop=True)
    inference_df = df.loc[~has_targets].reset_index(drop=True)
    return train_df, inference_df


def sanitize_json_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars and non-finite floats into strict JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [sanitize_json_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): sanitize_json_scalar(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_features_json_payload(
    row: Mapping[str, Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Extract secondary engineered features into the `features_json` JSONB dictionary."""
    keys = [
        "source_count",
        "is_dual_source",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "wind_u",
        "wind_v",
        "temp_apparent_delta",
        "heat_stress_index",
        "pm25_pm10_ratio",
        "aqi_wind_dispersion",
        "temp_diurnal_anomaly",
        "aqi_diurnal_anomaly",
        *get_dynamic_feature_names(settings),
    ]
    payload: dict[str, Any] = {}
    for key in keys:
        if key in row:
            payload[key] = sanitize_json_scalar(row[key])
    return payload
