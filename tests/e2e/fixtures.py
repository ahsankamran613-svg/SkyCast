"""Deterministic Lahore data generators, Settings factories, and SQLite seeders for E2E tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skycast.config.settings import METRIC_SPECS, Settings
from skycast.data.storage import SkyCastStorage
from skycast.features.engineer import (
    engineer_features,
    get_model_feature_columns,
)


def make_e2e_settings(
    tmp_path: Path,
    *,
    horizon_hours: int = 24,
    offline: bool = True,
    **overrides: Any,
) -> Settings:
    """Build an isolated `Settings` object backed by `tmp_path/.skycast_local`."""
    defaults: dict[str, Any] = {
        "skycast_env": "test",
        "skycast_local_data_dir": tmp_path / ".skycast_local",
        "forecast_horizon_hours": int(horizon_hours),
        "skycast_offline_mode": "true" if offline else "false",
    }
    if offline:
        defaults.update(
            {
                "supabase_url": "",
                "supabase_service_role_key": "",
                "hopsworks_api_key": "",
                "hopsworks_project_name": "",
            }
        )
    else:
        defaults.update(
            {
                "supabase_url": "https://e2e-fake.supabase.co",
                "supabase_service_role_key": "e2e-fake-service-role-key",
                "hopsworks_api_key": "e2e-fake-hopsworks-key",
                "hopsworks_project_name": "skycast_e2e",
            }
        )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _resolve_end_timestamp(
    *,
    n_hours: int,
    include_smog: bool,
    include_monsoon: bool,
    end_utc: datetime | pd.Timestamp | str | None,
) -> pd.Timestamp:
    if end_utc is None:
        if include_monsoon and include_smog:
            return pd.Timestamp("2026-10-16T23:00:00Z")
        if include_monsoon and not include_smog:
            return pd.Timestamp("2026-08-30T23:00:00Z")
        if include_smog and not include_monsoon:
            return pd.Timestamp("2026-11-30T23:00:00Z")
        return pd.Timestamp("2026-04-30T23:00:00Z")
    if isinstance(end_utc, str) and end_utc.strip().lower() == "now":
        return pd.Timestamp.now(tz="UTC").floor("h")
    ts = pd.to_datetime(end_utc, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid end_utc timestamp: {end_utc!r}")
    return ts.floor("h")


def generate_lahore_feature_dataframe(
    n_hours: int = 720,
    *,
    horizon_hours: int = 24,
    seed: int = 42,
    include_smog: bool = True,
    include_monsoon: bool = True,
    dry_mode: bool = False,
    wind_wraparound: bool = True,
    zero_var_cols: Sequence[str] = (),
    end_utc: datetime | pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Generate deterministic, physically realistic Lahore hourly weather & target DataFrame.

    Returns an engineered + target-shifted DataFrame (via `engineer_features(..., include_targets=True)`)
    with all 270 model feature columns, `target_{metric}_t{1..H}`, `target_wind_dir_{sin,cos}_t{1..H}`,
    and `has_full_24h_targets`.
    """
    if n_hours <= 0:
        raise ValueError(f"n_hours must be positive, got {n_hours}")

    cfg = Settings(_env_file=None, forecast_horizon_hours=int(horizon_hours))
    end_ts = _resolve_end_timestamp(
        n_hours=n_hours,
        include_smog=include_smog,
        include_monsoon=include_monsoon,
        end_utc=end_utc,
    )
    timestamps = pd.date_range(end=end_ts, periods=n_hours, freq="1h", tz="UTC")
    ts_local = timestamps.tz_convert(cfg.lahore_timezone)

    rng = np.random.default_rng(seed)
    idx = np.arange(n_hours, dtype="float64")
    local_hours = ts_local.hour.to_numpy(dtype="float64")
    local_months = ts_local.month.to_numpy(dtype="int64")

    is_smog_month = np.isin(local_months, cfg.lahore_smog_months)
    is_monsoon_month = np.isin(local_months, cfg.lahore_monsoon_months)

    diurnal = np.sin(2.0 * np.pi * (local_hours - 8.0) / 24.0)
    synoptic = np.sin(2.0 * np.pi * idx / 120.0)

    # Temperature (°C): warmer in monsoon, cooler in smog months
    seasonal_temp_base = np.where(is_monsoon_month, 31.5, np.where(is_smog_month, 21.5, 27.0))
    temperature_2m = np.clip(
        seasonal_temp_base + 6.5 * diurnal + 1.8 * synoptic + rng.normal(0.0, 0.45, n_hours),
        METRIC_SPECS["temperature_2m"].min_value,
        METRIC_SPECS["temperature_2m"].max_value,
    )

    # Humidity (%)
    humidity_base = np.where(is_monsoon_month, 68.0, np.where(is_smog_month, 58.0, 44.0))
    relative_humidity_2m = np.clip(
        humidity_base - 16.0 * diurnal + 5.0 * synoptic + rng.normal(0.0, 1.5, n_hours),
        METRIC_SPECS["relative_humidity_2m"].min_value,
        METRIC_SPECS["relative_humidity_2m"].max_value,
    )

    # Precipitation (mm/h) & Probability (%)
    precipitation_mm = np.zeros(n_hours, dtype="float64")
    precipitation_prob = np.clip(
        8.0 + 4.0 * synoptic + rng.uniform(0.0, 6.0, n_hours),
        METRIC_SPECS["precipitation_prob"].min_value,
        METRIC_SPECS["precipitation_prob"].max_value,
    )
    if not dry_mode and include_monsoon:
        rain_candidates = np.where(is_monsoon_month)[0]
        if len(rain_candidates) == 0:
            rain_candidates = np.arange(n_hours)
        burst_centers = rain_candidates[:: max(36, len(rain_candidates) // 5)]
        for center in burst_centers:
            for offset in range(-2, 3):
                pos = int(center) + offset
                if 0 <= pos < n_hours:
                    intensity = max(0.8, 12.0 - 3.5 * abs(offset) + float(rng.uniform(0.0, 6.0)))
                    precipitation_mm[pos] = min(
                        METRIC_SPECS["precipitation_mm"].max_value, intensity
                    )
                    precipitation_prob[pos] = float(rng.uniform(70.0, 96.0))
    elif dry_mode:
        precipitation_prob = np.clip(
            rng.uniform(0.0, 8.0, n_hours),
            METRIC_SPECS["precipitation_prob"].min_value,
            METRIC_SPECS["precipitation_prob"].max_value,
        )

    # UV Index (daytime peak around 13:00 PKT, zero at night)
    day_solar = np.maximum(0.0, np.sin(np.pi * (local_hours - 6.0) / 12.0))
    uv_scale = np.where(precipitation_mm > 1.0, 3.0, 9.2)
    uv_index = np.clip(
        uv_scale * day_solar + np.where(day_solar > 0.05, rng.normal(0.0, 0.25, n_hours), 0.0),
        METRIC_SPECS["uv_index"].min_value,
        METRIC_SPECS["uv_index"].max_value,
    )

    # Wind Speed (km/h): calmer in smog stagnation, brisker in monsoon
    wind_base = np.where(is_smog_month & include_smog, 5.5, np.where(is_monsoon_month, 12.5, 9.5))
    wind_speed_10m = np.clip(
        wind_base + 3.2 * np.cos(2.0 * np.pi * idx / 24.0) + 1.8 * (precipitation_mm > 0.5)
        + rng.normal(0.0, 0.7, n_hours),
        METRIC_SPECS["wind_speed_10m"].min_value,
        METRIC_SPECS["wind_speed_10m"].max_value,
    )

    # Wind Direction (degrees, [0, 360))
    if wind_wraparound:
        # Oscillates smoothly across 350..360 and 0..10 degrees
        wind_direction_10m = (
            355.0 + 12.0 * np.sin(2.0 * np.pi * idx / 48.0) + rng.normal(0.0, 2.0, n_hours)
        ) % 360.0
    else:
        wind_direction_10m = (
            180.0 + 45.0 * np.sin(2.0 * np.pi * idx / 48.0) + rng.normal(0.0, 3.0, n_hours)
        ) % 360.0

    # US AQI & Air Quality Pollutants
    if include_smog:
        smog_active = is_smog_month if np.any(is_smog_month) else np.ones(n_hours, dtype=bool)
        aqi_base = np.where(smog_active, 245.0, 95.0)
        aqi_amp = np.where(smog_active, 55.0, 22.0)
    else:
        aqi_base = np.full(n_hours, 88.0)
        aqi_amp = np.full(n_hours, 18.0)

    us_aqi = np.clip(
        aqi_base - aqi_amp * diurnal - 4.0 * wind_speed_10m + rng.normal(0.0, 6.0, n_hours),
        METRIC_SPECS["us_aqi"].min_value,
        METRIC_SPECS["us_aqi"].max_value,
    )
    pm2_5 = np.clip(us_aqi * 0.42 + rng.normal(0.0, 2.5, n_hours), 2.0, 350.0)
    pm10 = np.clip(pm2_5 * 1.65 + 12.0 + rng.normal(0.0, 4.0, n_hours), 5.0, 500.0)
    ozone = np.clip(38.0 + 18.0 * np.maximum(0.0, diurnal) + rng.normal(0.0, 2.0, n_hours), 5.0, 180.0)
    nitrogen_dioxide = np.clip(28.0 + 10.0 * (1.0 - np.abs(diurnal)) + rng.normal(0.0, 2.0, n_hours), 2.0, 150.0)
    sulphur_dioxide = np.clip(12.0 + 4.0 * (us_aqi / 200.0) + rng.normal(0.0, 1.0, n_hours), 1.0, 80.0)
    carbon_monoxide = np.clip(380.0 + 1.8 * us_aqi + rng.normal(0.0, 15.0, n_hours), 100.0, 2500.0)

    # Auxiliary weather columns
    apparent_temperature = np.clip(
        temperature_2m + 0.04 * (relative_humidity_2m - 45.0) + rng.normal(0.0, 0.2, n_hours),
        -25.0,
        65.0,
    )
    surface_pressure = 1012.0 + 3.5 * synoptic + rng.normal(0.0, 0.4, n_hours)
    cloud_cover = np.clip(
        np.where(precipitation_mm > 0.0, 82.0, np.where(is_monsoon_month, 48.0, 22.0))
        + rng.normal(0.0, 5.0, n_hours),
        0.0,
        100.0,
    )
    weather_code = np.where(
        precipitation_mm > 3.0,
        63,
        np.where(precipitation_mm > 0.0, 51, np.where(cloud_cover > 55.0, 2, 0)),
    ).astype("int64")
    condition_symbol = np.where(
        precipitation_mm > 3.0,
        "rain",
        np.where(precipitation_mm > 0.0, "lightrain", np.where(cloud_cover > 55.0, "partlycloudy_day", "clearsky_day")),
    )

    # Multi-source raw transparency columns
    temp_delta = rng.normal(0.25, 0.15, n_hours)
    temp_openmeteo = temperature_2m + 0.4 * temp_delta
    temp_metno = temperature_2m - 0.6 * temp_delta
    source_disagreement_temp = np.abs(temp_openmeteo - temp_metno)

    raw_df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "us_aqi": np.round(us_aqi, 3),
            "temperature_2m": np.round(temperature_2m, 3),
            "precipitation_mm": np.round(precipitation_mm, 3),
            "precipitation_prob": np.round(precipitation_prob, 3),
            "relative_humidity_2m": np.round(relative_humidity_2m, 3),
            "uv_index": np.round(uv_index, 3),
            "wind_speed_10m": np.round(wind_speed_10m, 3),
            "wind_direction_10m": np.round(wind_direction_10m, 3),
            "apparent_temperature": np.round(apparent_temperature, 3),
            "weather_code": weather_code,
            "condition_symbol": condition_symbol,
            "surface_pressure": np.round(surface_pressure, 3),
            "cloud_cover": np.round(cloud_cover, 3),
            "pm2_5": np.round(pm2_5, 3),
            "pm10": np.round(pm10, 3),
            "ozone": np.round(ozone, 3),
            "nitrogen_dioxide": np.round(nitrogen_dioxide, 3),
            "sulphur_dioxide": np.round(sulphur_dioxide, 3),
            "carbon_monoxide": np.round(carbon_monoxide, 3),
            "temp_openmeteo": np.round(temp_openmeteo, 3),
            "temp_metno": np.round(temp_metno, 3),
            "humidity_openmeteo": np.round(relative_humidity_2m + 0.8, 3),
            "humidity_metno": np.round(relative_humidity_2m - 1.2, 3),
            "precip_openmeteo": np.round(precipitation_mm, 3),
            "precip_metno": np.round(precipitation_mm, 3),
            "wind_speed_openmeteo": np.round(wind_speed_10m + 0.3, 3),
            "wind_speed_metno": np.round(np.maximum(0.0, wind_speed_10m - 0.4), 3),
            "wind_dir_openmeteo": np.round(wind_direction_10m, 3),
            "wind_dir_metno": np.round((wind_direction_10m + 4.0) % 360.0, 3),
            "uv_openmeteo": np.round(uv_index, 3),
            "uv_metno": np.round(uv_index, 3),
            "source_disagreement_temp": np.round(source_disagreement_temp, 3),
            "sources_available": [["open_meteo", "met_no"] for _ in range(n_hours)],
            "is_imputed": [False] * n_hours,
            "imputation_tier": ["none"] * n_hours,
            "imputed_columns": [[] for _ in range(n_hours)],
        }
    )

    for col in zero_var_cols:
        if col in raw_df.columns:
            raw_val = raw_df[col].iloc[0]
            if isinstance(raw_val, (int, float, np.floating, np.integer)) and not isinstance(
                raw_val, (bool, np.bool_)
            ):
                raw_df[col] = float(round(float(raw_val)))
            else:
                raw_df[col] = raw_val

    featured = engineer_features(raw_df, settings=cfg, include_targets=True)

    for col in zero_var_cols:
        if col in featured.columns:
            non_na = featured[col].dropna()
            const_val = non_na.iloc[0] if not non_na.empty else 1.0
            if isinstance(const_val, (int, float, np.floating, np.integer)) and not isinstance(
                const_val, (bool, np.bool_)
            ):
                featured[col] = float(round(float(const_val)))
            else:
                featured[col] = const_val

    return featured


def seed_sqlite_history(
    storage: SkyCastStorage,
    df: pd.DataFrame | None = None,
    *,
    raw_obs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Initialize `storage` SQLite DB and persist hourly weather feature history locally."""
    storage.init_local_db()
    if raw_obs_df is not None:
        seed_df = engineer_features(raw_obs_df, settings=storage.settings, include_targets=True)
    elif df is not None:
        expected_cols = set(get_model_feature_columns(settings=storage.settings))
        if expected_cols.issubset(df.columns):
            seed_df = df.copy()
        else:
            seed_df = engineer_features(df, settings=storage.settings, include_targets=True)
    else:
        seed_df = generate_lahore_feature_dataframe(
            horizon_hours=storage.settings.forecast_horizon_hours,
        )

    return storage.upsert_weather_features(
        seed_df,
        write_local=True,
        write_supabase=False,
        write_hopsworks=False,
    )
