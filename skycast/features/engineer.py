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


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").astype("float64")
    return pd.Series(default, index=df.index, dtype="float64")


def add_forecast_targets(
    df: pd.DataFrame,
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

    horizon_list = tuple(horizons) if horizons is not None else tuple(range(1, 25))
    target_cols: dict[str, pd.Series] = {}

    for metric in TARGET_METRICS:
        base_series = _safe_series(df, metric, default=float("nan"))
        for h in horizon_list:
            target_cols[f"target_{metric}_t{h}"] = base_series.shift(-h)

    if "wind_dir_sin" in df.columns and "wind_dir_cos" in df.columns:
        sin_s = _safe_series(df, "wind_dir_sin", default=float("nan"))
        cos_s = _safe_series(df, "wind_dir_cos", default=float("nan"))
    else:
        wd_rad = np.deg2rad(_safe_series(df, "wind_direction_10m", default=float("nan")))
        sin_s = pd.Series(np.sin(wd_rad), index=df.index, dtype="float64")
        cos_s = pd.Series(np.cos(wd_rad), index=df.index, dtype="float64")

    for h in horizon_list:
        target_cols[f"target_wind_dir_sin_t{h}"] = sin_s.shift(-h)
        target_cols[f"target_wind_dir_cos_t{h}"] = cos_s.shift(-h)

    max_h = max(horizon_list)
    full_mask = pd.Series(True, index=df.index, dtype="bool")
    for metric in TARGET_METRICS:
        full_mask &= target_cols[f"target_{metric}_t{max_h}"].notna()
    target_cols["has_full_24h_targets"] = full_mask

    existing_clean = df.drop(columns=[c for c in target_cols if c in df.columns], errors="ignore")
    return pd.concat([existing_clean, pd.DataFrame(target_cols, index=df.index)], axis=1)


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
            return add_forecast_targets(out, horizons=horizons)
        return out

    if "timestamp_utc" not in df.columns:
        raise ValueError("DataFrame must include 'timestamp_utc' for feature engineering.")

    cfg = settings or get_settings()
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    # Local Lahore time (Asia/Karachi, UTC+5) for physically aligned diurnal & seasonal signals
    ts_utc = pd.DatetimeIndex(out["timestamp_utc"])
    ts_local = ts_utc.tz_convert(cfg.lahore_timezone)

    hours = ts_local.hour.to_numpy(dtype="float64")
    dows = ts_local.dayofweek.to_numpy(dtype="float64")
    months = ts_local.month.to_numpy(dtype="float64")

    wind_dir_deg = _safe_series(out, "wind_direction_10m", default=0.0)
    wind_rad = np.deg2rad(wind_dir_deg)
    wind_dir_sin = pd.Series(np.sin(wind_rad), index=out.index, dtype="float64")
    wind_dir_cos = pd.Series(np.cos(wind_rad), index=out.index, dtype="float64")

    new_cols: dict[str, Any] = {
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
        "hour_sin": np.sin(2.0 * np.pi * hours / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hours / 24.0),
        "day_of_week_sin": np.sin(2.0 * np.pi * dows / 7.0),
        "day_of_week_cos": np.cos(2.0 * np.pi * dows / 7.0),
        "month_sin": np.sin(2.0 * np.pi * (months - 1.0) / 12.0),
        "month_cos": np.cos(2.0 * np.pi * (months - 1.0) / 12.0),
        "is_weekend": pd.Series(ts_local.dayofweek >= 5, index=out.index, dtype="bool"),
        "is_lahore_smog_season": pd.Series(
            np.isin(ts_local.month, cfg.lahore_smog_months), index=out.index, dtype="bool"
        ),
        "is_monsoon_season": pd.Series(
            np.isin(ts_local.month, cfg.lahore_monsoon_months), index=out.index, dtype="bool"
        ),
    }

    temp_s = _safe_series(out, "temperature_2m")
    app_s = _safe_series(out, "apparent_temperature")
    rh_s = _safe_series(out, "relative_humidity_2m")
    wind_spd_s = _safe_series(out, "wind_speed_10m")
    us_aqi_s = _safe_series(out, "us_aqi")
    pm25_s = _safe_series(out, "pm2_5")
    pm10_s = _safe_series(out, "pm10")

    # Physical & Lahore-specific interaction features
    new_cols["wind_u"] = -wind_spd_s * wind_dir_sin
    new_cols["wind_v"] = -wind_spd_s * wind_dir_cos
    new_cols["temp_apparent_delta"] = app_s - temp_s
    new_cols["heat_stress_index"] = temp_s * (rh_s / 100.0)
    new_cols["pm25_pm10_ratio"] = pm25_s / pm10_s.clip(lower=1.0)
    new_cols["aqi_wind_dispersion"] = us_aqi_s / (wind_spd_s + 1.0)

    # Strictly causal Rate-of-Change, Rolling, and Lag features
    series_map: dict[str, pd.Series] = {
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
    }
    for col in ENGINEERED_SERIES_FOR_TEMPORAL_DYNAMICS:
        if col not in series_map:
            series_map[col] = _safe_series(out, col)

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

    base_clean = out.drop(columns=[c for c in new_cols if c in out.columns], errors="ignore")
    featured = pd.concat([base_clean, pd.DataFrame(new_cols, index=out.index)], axis=1)

    if include_targets:
        featured = add_forecast_targets(featured, horizons=horizons)

    return featured


def split_training_and_inference_rows(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an engineered+targeted DataFrame into trainable rows and latest forecast-only rows."""
    if df.empty:
        return df.copy(), df.copy()
    if "has_full_24h_targets" not in df.columns:
        raise ValueError("DataFrame must include targets (`include_targets=True`) before splitting.")

    feature_cols = get_model_feature_columns(df, settings=settings)
    has_targets = df["has_full_24h_targets"].astype(bool)
    features_complete = ~df[feature_cols].isna().any(axis=1)

    train_df = df.loc[has_targets & features_complete].reset_index(drop=True)
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
