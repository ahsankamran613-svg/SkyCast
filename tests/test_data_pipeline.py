"""Verification tests for Component 2: ingestion, fusion, 3-tier imputation, features, and pipelines."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

from skycast.config.settings import Settings, TARGET_METRICS
from skycast.data.fusion import SourceFusionError, fuse_weather_sources
from skycast.data.imputation import (
    TIER_1_LINEAR,
    TIER_2_DIURNAL,
    TIER_3_CLIMATOLOGY,
    TIER_NONE,
    impute_weather_dataframe,
)
from skycast.data.sources.met_no import (
    compute_steadman_apparent_temperature,
    fetch_met_no_forecast,
    metno_symbol_to_wmo_code,
    parse_met_no_timeseries,
    sea_level_to_surface_pressure,
    wmo_code_to_symbol,
)
from skycast.data.sources.open_meteo import (
    OPEN_METEO_AIR_QUALITY_FIELDS,
    OPEN_METEO_ARCHIVE_WEATHER_FIELDS,
    OPEN_METEO_WEATHER_FIELDS,
    OpenMeteoError,
    fetch_open_meteo_forecast,
    fetch_open_meteo_historical,
)
from skycast.data.storage import (
    SkyCastStorage,
    StorageError,
    dataframe_to_hopsworks_frame,
    dataframe_to_supabase_weather_records,
)
from skycast.features.engineer import (
    engineer_features,
    get_model_feature_columns,
    split_training_and_inference_rows,
)
from skycast.pipelines.backfill import run_backfill
from skycast.pipelines.hourly_ingest import run_hourly_ingest


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "skycast_offline_mode": "true",
        "skycast_local_data_dir": tmp_path / ".skycast_local",
        "supabase_url": "",
        "supabase_service_role_key": "",
        "hopsworks_api_key": "",
        "hopsworks_project_name": "",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _build_synthetic_open_meteo_payload(
    timestamps: pd.DatetimeIndex,
    fields: tuple[str, ...],
    *,
    temp_offset: float = 0.0,
) -> dict[str, Any]:
    hours = np.arange(len(timestamps), dtype="float64")
    diurnal = np.sin(2.0 * np.pi * (timestamps.hour.to_numpy() - 6) / 24.0)

    values_map: dict[str, list[Any]] = {
        "temperature_2m": [round(float(24.0 + temp_offset + 6.0 * d), 2) for d in diurnal],
        "apparent_temperature": [round(float(25.0 + temp_offset + 6.0 * d), 2) for d in diurnal],
        "weather_code": [2 if int(h) % 6 == 0 else 0 for h in hours],
        "relative_humidity_2m": [round(float(55.0 - 15.0 * d), 2) for d in diurnal],
        "precipitation": [0.0 for _ in hours],
        "precipitation_probability": [10.0 for _ in hours],
        "uv_index": [round(float(max(0.0, 6.0 * d)), 2) for d in diurnal],
        "wind_speed_10m": [round(float(8.0 + 3.0 * np.cos(h / 5.0)), 2) for h in hours],
        "wind_direction_10m": [round(float((300.0 + 5.0 * h) % 360.0), 1) for h in hours],
        "surface_pressure": [round(float(1013.0 + 1.5 * np.sin(h / 12.0)), 2) for h in hours],
        "cloud_cover": [25.0 for _ in hours],
        "us_aqi": [round(float(135.0 - 20.0 * d), 1) for d in diurnal],
        "pm2_5": [round(float(48.0 - 8.0 * d), 2) for d in diurnal],
        "pm10": [round(float(88.0 - 10.0 * d), 2) for d in diurnal],
        "nitrogen_dioxide": [26.0 for _ in hours],
        "ozone": [44.0 for _ in hours],
        "sulphur_dioxide": [11.0 for _ in hours],
        "carbon_monoxide": [420.0 for _ in hours],
    }

    hourly: dict[str, Any] = {
        "time": [ts.strftime("%Y-%m-%dT%H:%M") for ts in timestamps],
    }
    for field in fields:
        hourly[field] = values_map[field]
    return {"hourly": hourly}


def _build_synthetic_metno_payload(
    timestamps: pd.DatetimeIndex,
    *,
    include_uv: bool = True,
    include_apparent: bool = True,
) -> dict[str, Any]:
    timeseries = []
    for idx, ts in enumerate(timestamps):
        diurnal = math.sin(2.0 * math.pi * (ts.hour - 6) / 24.0)
        details: dict[str, Any] = {
            "air_temperature": round(25.0 + 6.0 * diurnal, 2),
            "relative_humidity": round(52.0 - 14.0 * diurnal, 2),
            "wind_speed": 2.5,  # 2.5 m/s == 9.0 km/h
            "wind_from_direction": float((310.0 + 4.0 * idx) % 360.0),
            "air_pressure_at_sea_level": 1012.5,
            "cloud_area_fraction": 20.0,
        }
        if include_uv:
            details["ultraviolet_index_clear_sky"] = round(max(0.0, 6.5 * diurnal), 2)
        if include_apparent:
            details["apparent_air_temperature"] = round(25.5 + 6.0 * diurnal, 2)

        timeseries.append(
            {
                "time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data": {
                    "instant": {"details": details},
                    "next_1_hours": {
                        "summary": {"symbol_code": "partlycloudy_day"},
                        "details": {"precipitation_amount": 0.0},
                    },
                },
            }
        )
    return {
        "properties": {
            "meta": {"units": {"wind_speed": "m/s"}},
            "timeseries": timeseries,
        }
    }


def test_open_meteo_forecast_and_partial_fallback(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-11-17T00:00:00Z", periods=6, freq="1h", tz="UTC")
    weather_payload = _build_synthetic_open_meteo_payload(ts, OPEN_METEO_WEATHER_FIELDS)
    aq_payload = _build_synthetic_open_meteo_payload(ts, OPEN_METEO_AIR_QUALITY_FIELDS)

    httpx_mock.add_response(
        url=httpx.URL(cfg.open_meteo_forecast_url).copy_with(
            params={
                "latitude": cfg.lahore_lat,
                "longitude": cfg.lahore_lon,
                "timezone": "UTC",
                "past_days": 2,
                "forecast_days": 2,
                "hourly": ",".join(OPEN_METEO_WEATHER_FIELDS),
            }
        ),
        json=weather_payload,
    )
    httpx_mock.add_response(
        url=httpx.URL(cfg.open_meteo_air_quality_url).copy_with(
            params={
                "latitude": cfg.lahore_lat,
                "longitude": cfg.lahore_lon,
                "timezone": "UTC",
                "past_days": 2,
                "forecast_days": 2,
                "hourly": ",".join(OPEN_METEO_AIR_QUALITY_FIELDS),
            }
        ),
        json=aq_payload,
    )

    df = fetch_open_meteo_forecast(cfg, past_days=2, forecast_days=2, backoff_seconds=0.0)
    assert len(df) == 6
    assert str(df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert df["us_aqi"].notna().all()
    assert df["temperature_2m"].notna().all()

    # Partial failure: AQ returns 500 -> weather still returned when allow_partial=True
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(cfg.open_meteo_forecast_url)}.*"),
        json=weather_payload,
    )
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(cfg.open_meteo_air_quality_url)}.*"),
        status_code=500,
    )
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(cfg.open_meteo_air_quality_url)}.*"),
        status_code=500,
    )
    httpx_mock.add_response(
        url=re.compile(rf"^{re.escape(cfg.open_meteo_air_quality_url)}.*"),
        status_code=500,
    )

    with httpx.Client() as client:
        partial_df = fetch_open_meteo_forecast(
            cfg, past_days=2, forecast_days=2, client=client, allow_partial=True, backoff_seconds=0.0
        )
    assert len(partial_df) == 6
    assert partial_df["temperature_2m"].notna().all()
    assert partial_df["us_aqi"].isna().all()


def test_open_meteo_both_fail_raises(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    cfg = _make_settings(tmp_path)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)

    with pytest.raises(OpenMeteoError):
        fetch_open_meteo_forecast(cfg, backoff_seconds=0.0)


def test_open_meteo_historical_merges_archive_histfc_and_aq(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-11-10T00:00:00Z", "2026-11-11T23:00:00Z", freq="1h", tz="UTC")

    arch_payload = _build_synthetic_open_meteo_payload(ts, OPEN_METEO_ARCHIVE_WEATHER_FIELDS)
    hist_fc_payload = _build_synthetic_open_meteo_payload(
        ts,
        (
            "temperature_2m",
            "apparent_temperature",
            "weather_code",
            "relative_humidity_2m",
            "precipitation",
            "precipitation_probability",
            "uv_index",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cloud_cover",
        ),
    )
    aq_payload = _build_synthetic_open_meteo_payload(
        ts, (*OPEN_METEO_AIR_QUALITY_FIELDS, "uv_index")
    )

    def _dispatch(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_archive_url):
            return httpx.Response(200, json=arch_payload)
        if url.startswith(cfg.open_meteo_historical_forecast_url):
            return httpx.Response(200, json=hist_fc_payload)
        if url.startswith(cfg.open_meteo_air_quality_url):
            return httpx.Response(200, json=aq_payload)
        return httpx.Response(404)

    httpx_mock.add_callback(_dispatch)
    httpx_mock.add_callback(_dispatch)
    httpx_mock.add_callback(_dispatch)

    df = fetch_open_meteo_historical("2026-11-10", "2026-11-11", settings=cfg, backoff_seconds=0.0)
    assert len(df) == 48
    assert df["precipitation_probability"].notna().all()
    assert df["uv_index"].notna().all()
    assert df["us_aqi"].notna().all()


def test_metno_user_agent_units_and_steadman_fallback(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-11-18T10:00:00Z", periods=4, freq="1h", tz="UTC")
    payload = _build_synthetic_metno_payload(ts, include_uv=True, include_apparent=False)

    captured_headers: dict[str, str] = {}

    def _handle(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json=payload)

    httpx_mock.add_callback(_handle)
    df = fetch_met_no_forecast(cfg, backoff_seconds=0.0)

    assert captured_headers.get("user-agent") == cfg.met_no_user_agent
    assert len(df) == 4
    # 2.5 m/s converted to 9.0 km/h
    assert df["wind_speed"].iloc[0] == pytest.approx(9.0)
    assert df["weather_code"].iloc[0] == pytest.approx(2.0)
    assert df["ultraviolet_index_clear_sky"].notna().all()

    # Verify Steadman apparent temperature matches explicit formula when missing from payload
    expected_at = compute_steadman_apparent_temperature(
        float(df["air_temperature"].iloc[0]),
        float(df["relative_humidity"].iloc[0]),
        2.5,
    )
    assert expected_at is not None
    assert df["apparent_air_temperature"].iloc[0] == pytest.approx(expected_at, rel=1e-5)
    assert metno_symbol_to_wmo_code("rainandthunder_night") == 95


def test_fusion_weighted_consensus_circular_wind_and_grid_gaps(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    # Open-Meteo missing hour 02:00Z intentionally to test continuous grid creation
    om_ts = pd.DatetimeIndex(
        [
            "2026-11-18T00:00:00Z",
            "2026-11-18T01:00:00Z",
            "2026-11-18T03:00:00Z",
        ]
    )
    om_df = pd.DataFrame(
        {
            "timestamp_utc": om_ts,
            "temperature_2m": [20.0, 22.0, 26.0],
            "apparent_temperature": [20.0, 22.0, 26.0],
            "weather_code": [0.0, 0.0, 2.0],
            "relative_humidity_2m": [50.0, 50.0, 40.0],
            "precipitation": [0.0, 0.0, 1.0],
            "precipitation_probability": [0.0, 5.0, 60.0],
            "uv_index": [0.0, 1.0, 5.0],
            "wind_speed_10m": [10.0, 10.0, 15.0],
            "wind_direction_10m": [350.0, 180.0, 90.0],
            "surface_pressure": [1012.0, 1012.0, 1010.0],
            "cloud_cover": [10.0, 10.0, 60.0],
            "us_aqi": [110.0, 120.0, 140.0],
            "pm2_5": [35.0, 40.0, 45.0],
            "pm10": [70.0, 75.0, 80.0],
            "nitrogen_dioxide": [20.0, 22.0, 25.0],
            "ozone": [40.0, 42.0, 45.0],
            "sulphur_dioxide": [10.0, 10.0, 12.0],
            "carbon_monoxide": [400.0, 410.0, 430.0],
        }
    )

    met_ts = pd.DatetimeIndex(["2026-11-18T00:00:00Z", "2026-11-18T03:00:00Z"])
    met_df = pd.DataFrame(
        {
            "timestamp_utc": met_ts,
            "air_temperature": [25.0, 21.0],
            "apparent_air_temperature": [25.0, 21.0],
            "relative_humidity": [60.0, 50.0],
            "precipitation_amount": [np.nan, 2.0],
            "wind_speed": [20.0, 5.0],
            "wind_from_direction": [10.0, 100.0],
            "ultraviolet_index_clear_sky": [0.0, 7.0],
            "air_pressure_at_sea_level": [1014.0, 1011.0],
            "cloud_area_fraction": [20.0, 70.0],
            "symbol_code": ["clearsky_night", "lightrain_day"],
            "weather_code": [0.0, 61.0],
        }
    )

    fused = fuse_weather_sources(om_df, met_df, settings=cfg, align_grid=True)
    # Grid spans 00:00..03:00 (4 rows, with 02:00 exposed as empty NaN gap)
    assert len(fused) == 4
    assert fused.loc[0, "temperature_2m"] == pytest.approx(0.6 * 20.0 + 0.4 * 25.0)
    assert fused.loc[0, "source_disagreement_temp"] == pytest.approx(5.0)
    assert fused.loc[0, "sources_available"] == ["open_meteo", "met_no"]

    # Circular wind average of 350° (0.6) and 10° (0.4) must stay near North (~358°), never South (~214°)
    wd_0 = float(fused.loc[0, "wind_direction_10m"])
    assert wd_0 > 350.0 or wd_0 < 10.0

    # Hour 01:00 only had Open-Meteo; missing source_disagreement_temp must remain NaN
    assert fused.loc[1, "sources_available"] == ["open_meteo"]
    assert fused.loc[1, "temperature_2m"] == pytest.approx(22.0)
    assert pd.isna(fused.loc[1, "source_disagreement_temp"])

    # Hour 02:00 missing in both -> exposed as NaN for imputer
    assert fused.loc[2, "sources_available"] == []
    assert pd.isna(fused.loc[2, "temperature_2m"])
    assert pd.isna(fused.loc[2, "source_disagreement_temp"])

    # Single-source Met.no fallback test
    met_only = fuse_weather_sources(None, met_df, settings=cfg, align_grid=False)
    assert len(met_only) == 2
    assert met_only.loc[0, "sources_available"] == ["met_no"]
    assert met_only.loc[0, "temperature_2m"] == pytest.approx(25.0)

    with pytest.raises(SourceFusionError):
        fuse_weather_sources(None, None, settings=cfg)


def test_three_tier_imputation_and_circular_wind(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    # 16 days (384 hours) so 7-day diurnal lookback and monthly climatology have rich context
    ts = pd.date_range("2026-10-01T00:00:00Z", periods=384, freq="1h", tz="UTC")
    hours = ts.hour.to_numpy()

    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "us_aqi": 120.0 + 15.0 * np.cos(2 * np.pi * hours / 24.0),
            "temperature_2m": 25.0 + 7.0 * np.sin(2 * np.pi * (hours - 6) / 24.0),
            "precipitation_mm": np.zeros(len(ts)),
            "precipitation_prob": np.full(len(ts), 10.0),
            "relative_humidity_2m": 55.0 - 15.0 * np.sin(2 * np.pi * (hours - 6) / 24.0),
            "uv_index": np.maximum(0.0, 6.0 * np.sin(2 * np.pi * (hours - 6) / 24.0)),
            "wind_speed_10m": np.full(len(ts), 10.0),
            "wind_direction_10m": np.full(len(ts), 180.0),
            "apparent_temperature": 25.0 + 7.0 * np.sin(2 * np.pi * (hours - 6) / 24.0),
            "surface_pressure": np.full(len(ts), 1012.0),
            "cloud_cover": np.full(len(ts), 20.0),
            "weather_code": np.zeros(len(ts)),
            "condition_symbol": ["clearsky"] * len(ts),
            "pm2_5": np.full(len(ts), 42.0),
            "pm10": np.full(len(ts), 80.0),
            "ozone": np.full(len(ts), 40.0),
            "nitrogen_dioxide": np.full(len(ts), 25.0),
            "sulphur_dioxide": np.full(len(ts), 10.0),
            "carbon_monoxide": np.full(len(ts), 400.0),
            "source_disagreement_temp": np.zeros(len(ts)),
            "sources_available": [["open_meteo"]] * len(ts),
        }
    )

    # Inject Tier 1 gap (2h: indices 200..201) + circular wind crossing 350 -> 10
    df.loc[199, "wind_direction_10m"] = 350.0
    df.loc[200:201, ["temperature_2m", "wind_direction_10m"]] = np.nan
    df.loc[202, "wind_direction_10m"] = 10.0
    expected_t200 = df.loc[199, "temperature_2m"] + (
        df.loc[202, "temperature_2m"] - df.loc[199, "temperature_2m"]
    ) * (1.0 / 3.0)

    # Inject Tier 2 gap (10h: indices 220..229)
    expected_tier2_h224 = float(
        np.nanmedian(
            [
                df.loc[idx, "temperature_2m"]
                for idx in range(224 - 24 * 7, 224, 24)
            ]
        )
    )
    df.loc[220:229, "temperature_2m"] = np.nan

    # Inject Tier 3 gap (36h outage: indices 260..295)
    df.loc[260:295, ["temperature_2m", "us_aqi", "weather_code"]] = np.nan

    res = impute_weather_dataframe(df, settings=cfg)
    imp = res.df

    assert res.audit.missing_after == 0
    assert res.audit.tier1_count > 0
    assert res.audit.tier2_count > 0
    assert res.audit.tier3_count > 0

    # Tier 1 exact linear check & circular wind wraparound check
    assert imp.loc[200, "imputation_tier"] == TIER_1_LINEAR
    assert imp.loc[200, "is_imputed"] is np.bool_(True) or imp.loc[200, "is_imputed"] is True
    assert imp.loc[200, "temperature_2m"] == pytest.approx(expected_t200, rel=1e-5)
    # Interpolated wind between 350° and 10° must pass through North (350..360 or 0..10), not South (180)
    wd_200 = float(imp.loc[200, "wind_direction_10m"])
    assert wd_200 > 345.0 or wd_200 < 15.0

    # Tier 2 check: all 10 hours tagged tier2_diurnal_median and match 7-day diurnal median
    assert all(imp.loc[220:229, "imputation_tier"] == TIER_2_DIURNAL)
    assert imp.loc[224, "temperature_2m"] == pytest.approx(expected_tier2_h224, rel=1e-5)

    # Tier 3 check: all 36 hours tagged tier3_climatology_ffill with zero NaNs
    assert all(imp.loc[260:295, "imputation_tier"] == TIER_3_CLIMATOLOGY)
    assert imp.loc[260:295, "temperature_2m"].notna().all()
    assert imp.loc[260:295, "us_aqi"].notna().all()
    assert "temperature_2m" in imp.loc[260, "imputed_columns"]
    assert imp.loc[150, "imputation_tier"] == TIER_NONE
    assert not bool(imp.loc[150, "is_imputed"])


def test_feature_engineering_zero_leakage_and_24h_targets(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-11-01T00:00:00Z", periods=120, freq="1h", tz="UTC")
    base_df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "us_aqi": np.linspace(100.0, 220.0, len(ts)),
            "temperature_2m": 20.0 + 5.0 * np.sin(np.arange(len(ts)) / 4.0),
            "precipitation_mm": np.zeros(len(ts)),
            "precipitation_prob": np.full(len(ts), 15.0),
            "relative_humidity_2m": np.full(len(ts), 60.0),
            "uv_index": np.full(len(ts), 4.0),
            "wind_speed_10m": np.full(len(ts), 12.0),
            "wind_direction_10m": np.linspace(0.0, 355.0, len(ts)),
            "apparent_temperature": 21.0 + 5.0 * np.sin(np.arange(len(ts)) / 4.0),
            "surface_pressure": np.full(len(ts), 1013.0),
            "cloud_cover": np.full(len(ts), 30.0),
            "weather_code": np.zeros(len(ts), dtype="int64"),
            "condition_symbol": ["clearsky"] * len(ts),
            "pm2_5": np.linspace(30.0, 90.0, len(ts)),
            "pm10": np.linspace(60.0, 150.0, len(ts)),
            "ozone": np.full(len(ts), 45.0),
            "nitrogen_dioxide": np.full(len(ts), 25.0),
            "sulphur_dioxide": np.full(len(ts), 12.0),
            "carbon_monoxide": np.full(len(ts), 450.0),
            "source_disagreement_temp": np.zeros(len(ts)),
        }
    )

    featured = engineer_features(base_df, settings=cfg, include_targets=True)
    feature_cols = get_model_feature_columns(featured, settings=cfg)
    assert len(feature_cols) >= 35

    # Cyclical unit-circle checks
    assert np.allclose(featured["hour_sin"] ** 2 + featured["hour_cos"] ** 2, 1.0)
    assert np.allclose(featured["wind_dir_sin"] ** 2 + featured["wind_dir_cos"] ** 2, 1.0)
    assert featured["is_lahore_smog_season"].all()  # November is Lahore smog season
    assert not featured["is_monsoon_season"].any()

    # Target shift checks (h=1 and h=24)
    for metric in TARGET_METRICS:
        assert featured.loc[50, f"target_{metric}_t1"] == pytest.approx(base_df.loc[51, metric])
        assert featured.loc[50, f"target_{metric}_t24"] == pytest.approx(base_df.loc[74, metric])

    train_df, inf_df = split_training_and_inference_rows(featured, settings=cfg)
    assert len(inf_df) == 24  # newest 24h have no complete target yet
    assert len(train_df) == len(ts) - 48 - 24  # first 48h lack lag_48h, last 24h lack t+24 target
    assert not train_df[feature_cols].isna().any().any()

    # Zero future-leakage test: perturb future rows t >= 80 and verify features at t < 80 are unchanged
    perturbed_df = base_df.copy()
    perturbed_df.loc[80:, "temperature_2m"] += 99.0
    perturbed_df.loc[80:, "us_aqi"] += 200.0
    perturbed_featured = engineer_features(perturbed_df, settings=cfg, include_targets=False)

    pd.testing.assert_frame_equal(
        featured.loc[:79, feature_cols],
        perturbed_featured.loc[:79, feature_cols],
    )


def test_storage_offline_and_cloud_serialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-11-16T00:00:00Z", periods=60, freq="1h", tz="UTC")
    raw = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "us_aqi": np.full(len(ts), 130.0),
            "temperature_2m": np.full(len(ts), 22.5),
            "precipitation_mm": np.zeros(len(ts)),
            "precipitation_prob": np.full(len(ts), 5.0),
            "relative_humidity_2m": np.full(len(ts), 55.0),
            "uv_index": np.full(len(ts), 2.0),
            "wind_speed_10m": np.full(len(ts), 8.0),
            "wind_direction_10m": np.full(len(ts), 315.0),
            "apparent_temperature": np.full(len(ts), 22.0),
            "surface_pressure": np.full(len(ts), 1013.0),
            "cloud_cover": np.full(len(ts), 15.0),
            "weather_code": np.zeros(len(ts), dtype="int64"),
            "condition_symbol": ["clearsky"] * len(ts),
            "pm2_5": np.full(len(ts), 42.0),
            "pm10": np.full(len(ts), 80.0),
            "ozone": np.full(len(ts), 40.0),
            "nitrogen_dioxide": np.full(len(ts), 25.0),
            "sulphur_dioxide": np.full(len(ts), 10.0),
            "carbon_monoxide": np.full(len(ts), 400.0),
            "temp_openmeteo": np.full(len(ts), 22.5),
            "temp_metno": np.full(len(ts), np.nan),
            "source_disagreement_temp": np.zeros(len(ts)),
            "sources_available": [["open_meteo"]] * len(ts),
            "is_imputed": np.zeros(len(ts), dtype=bool),
            "imputation_tier": ["none"] * len(ts),
            "imputed_columns": [[]] * len(ts),
        }
    )
    featured = engineer_features(raw, settings=cfg, include_targets=False)

    # Verify JSON-safe serialization (NaN converted to None, never literal float('nan'))
    supa_records = dataframe_to_supabase_weather_records(featured.tail(5), settings=cfg)
    assert len(supa_records) == 5
    assert supa_records[0]["temp_metno"] is None
    assert supa_records[0]["sources_available"] == ["open_meteo"]
    assert isinstance(supa_records[0]["features_json"], dict)

    # Verify Hopsworks schema frame
    hw_df = dataframe_to_hopsworks_frame(featured, settings=cfg)
    assert hw_df["timestamp_epoch_ms"].dtype == "int64"
    assert str(hw_df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert isinstance(hw_df["sources_available"].iloc[0], str)

    # Offline SQLite write/read round-trip
    store = SkyCastStorage(cfg)
    write_summary = store.upsert_weather_features(featured)
    assert write_summary["storage_mode"] == "local_offline"
    assert write_summary["sqlite_rows"] == 60

    read_df = store.read_weather_features(limit_hours=12, expand_features_json=True)
    assert len(read_df) == 12
    assert "temperature_2m_lag_48h" in read_df.columns
    assert read_df["temperature_2m_lag_48h"].notna().all()

    # Cloud dual-write mock test (Supabase + Hopsworks)
    cloud_cfg = _make_settings(
        tmp_path,
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
        hopsworks_api_key="hw-key",
        hopsworks_project_name="skycast_test",
    )
    cloud_store = SkyCastStorage(cloud_cfg)

    supabase_upserts: list[dict[str, Any]] = []

    class _FakeTable:
        def __init__(self, table_name: str) -> None:
            self.table_name = table_name

        def upsert(self, payload: Any, on_conflict: str = "") -> _FakeTable:
            supabase_upserts.append({"table": self.table_name, "payload": payload, "on_conflict": on_conflict})
            return self

        def execute(self) -> Any:
            return type("Resp", (), {"data": []})()

    class _FakeSupabase:
        def table(self, name: str) -> _FakeTable:
            return _FakeTable(name)

    hopsworks_inserts: list[pd.DataFrame] = []

    class _FakeFG:
        def insert(self, df_in: pd.DataFrame, write_options: dict[str, Any] | None = None) -> None:
            hopsworks_inserts.append(df_in)

    class _FakeFS:
        def get_or_create_feature_group(self, **kwargs: Any) -> _FakeFG:
            assert kwargs["online_enabled"] is True
            assert kwargs["primary_key"] == ["timestamp_epoch_ms"]
            assert kwargs["event_time"] == "timestamp_utc"
            return _FakeFG()

    monkeypatch.setattr(cloud_store, "_get_supabase", lambda: _FakeSupabase())
    monkeypatch.setattr(cloud_store, "_get_hopsworks_feature_store", lambda: _FakeFS())

    cloud_summary = cloud_store.upsert_weather_features(featured)
    assert cloud_summary["storage_mode"] == "cloud_dual"
    assert cloud_summary["supabase_rows"] == 60
    assert cloud_summary["hopsworks_rows"] == 60
    assert len(supabase_upserts) == 1
    assert len(hopsworks_inserts) == 1


def test_backfill_and_hourly_ingest_pipelines_offline(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path)
    store = SkyCastStorage(cfg)
    ref_hour = pd.Timestamp("2026-11-18T13:00:00Z")

    # 3-day backfill + recent forecast + Met.no
    hist_ts = pd.date_range("2026-11-15T00:00:00Z", "2026-11-18T13:00:00Z", freq="1h", tz="UTC")
    recent_ts = pd.date_range("2026-11-16T00:00:00Z", "2026-11-19T12:00:00Z", freq="1h", tz="UTC")
    met_ts = pd.date_range("2026-11-18T13:00:00Z", "2026-11-19T12:00:00Z", freq="1h", tz="UTC")

    arch_payload = _build_synthetic_open_meteo_payload(hist_ts, OPEN_METEO_ARCHIVE_WEATHER_FIELDS)
    hist_fc_payload = _build_synthetic_open_meteo_payload(hist_ts, OPEN_METEO_WEATHER_FIELDS)
    aq_hist_payload = _build_synthetic_open_meteo_payload(
        hist_ts, (*OPEN_METEO_AIR_QUALITY_FIELDS, "uv_index")
    )
    recent_w_payload = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_WEATHER_FIELDS)
    recent_aq_payload = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_AIR_QUALITY_FIELDS)
    met_payload = _build_synthetic_metno_payload(met_ts)

    def _mock_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_archive_url):
            return httpx.Response(200, json=arch_payload)
        if url.startswith(cfg.open_meteo_historical_forecast_url):
            return httpx.Response(200, json=hist_fc_payload)
        if url.startswith(cfg.open_meteo_forecast_url):
            return httpx.Response(200, json=recent_w_payload)
        if url.startswith(cfg.open_meteo_air_quality_url):
            if "start_date=" in url:
                return httpx.Response(200, json=aq_hist_payload)
            return httpx.Response(200, json=recent_aq_payload)
        if url.startswith(cfg.met_no_forecast_url):
            return httpx.Response(200, json=met_payload)
        return httpx.Response(404)

    for _ in range(6):
        httpx_mock.add_callback(_mock_router)

    backfill_res = run_backfill(days=3, end_utc=ref_hour, settings=cfg, storage=store)
    assert backfill_res["status"] == "completed"
    expected_backfill_rows = 3 * 24 + 1
    assert backfill_res["rows_processed"] == expected_backfill_rows
    assert backfill_res["imputation_audit"]["missing_after"] == 0

    stored_after_backfill = store.read_weather_features(expand_features_json=True)
    assert len(stored_after_backfill) == expected_backfill_rows
    # Latest hour (13:00Z) fused both Open-Meteo and Met.no; no future hours beyond 13:00Z leaked in
    assert stored_after_backfill["timestamp_utc"].max() == ref_hour
    assert stored_after_backfill.iloc[-1]["sources_available"] == ["open_meteo", "met_no"]

    # Now advance 1 hour to 14:00Z and simulate Open-Meteo down, Met.no online!
    next_hour = ref_hour + pd.Timedelta(hours=1)
    met_next_ts = pd.date_range("2026-11-18T14:00:00Z", "2026-11-19T12:00:00Z", freq="1h", tz="UTC")
    met_next_payload = _build_synthetic_metno_payload(met_next_ts)

    def _hourly_outage_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.met_no_forecast_url):
            return httpx.Response(200, json=met_next_payload)
        return httpx.Response(503, text="Open-Meteo outage")

    # 3 attempts weather + 3 attempts AQ + 1 attempt Met.no
    for _ in range(7):
        httpx_mock.add_callback(_hourly_outage_router)

    ingest_res = run_hourly_ingest(
        lookback_hours=72,
        reference_utc=next_hour,
        settings=cfg,
        storage=store,
        run_agents=False,
    )
    assert ingest_res["status"] == "completed"
    assert ingest_res["imputation_audit"]["missing_after"] == 0
    assert ingest_res["latest_observation"]["timestamp_utc"] == next_hour.isoformat()
    assert ingest_res["latest_observation"]["sources_available"] == ["met_no"]

    latest_stored = store.read_weather_features(limit_hours=2, expand_features_json=True)
    assert latest_stored["timestamp_utc"].iloc[-1] == next_hour
    assert latest_stored["temperature_2m_lag_48h"].notna().all()
    assert store.get_latest_imputation_audit() is not None


def test_hourly_ingest_cold_start_and_total_outage_logging(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path)
    store = SkyCastStorage(cfg)
    ref_hour = pd.Timestamp("2026-11-18T13:00:00Z")

    # Cold start (no prior backfill) with both sources online
    recent_ts = pd.date_range("2026-11-15T13:00:00Z", "2026-11-19T12:00:00Z", freq="1h", tz="UTC")
    met_ts = pd.date_range("2026-11-18T13:00:00Z", "2026-11-19T12:00:00Z", freq="1h", tz="UTC")
    w_payload = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_WEATHER_FIELDS)
    aq_payload = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_AIR_QUALITY_FIELDS)
    met_payload = _build_synthetic_metno_payload(met_ts)

    def _cold_start_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_forecast_url):
            return httpx.Response(200, json=w_payload)
        if url.startswith(cfg.open_meteo_air_quality_url):
            return httpx.Response(200, json=aq_payload)
        if url.startswith(cfg.met_no_forecast_url):
            return httpx.Response(200, json=met_payload)
        return httpx.Response(404)

    for _ in range(3):
        httpx_mock.add_callback(_cold_start_router)

    cold_res = run_hourly_ingest(
        lookback_hours=72,
        reference_utc=ref_hour,
        settings=cfg,
        storage=store,
        run_agents=False,
    )
    assert cold_res["status"] == "completed"
    # Cold start persists the full 73-hour lookback window rather than only 49h
    assert cold_res["rows_persisted"] == 73
    assert len(store.read_weather_features()) == 73

    # Total outage (all external APIs return 503): must fail loudly and log status="failed"
    for _ in range(9):
        httpx_mock.add_response(status_code=503, text="All sources down")

    with pytest.raises(RuntimeError, match="Both Open-Meteo and Met.no failed"):
        run_hourly_ingest(
            lookback_hours=72,
            reference_utc=ref_hour + pd.Timedelta(hours=1),
            settings=cfg,
            storage=store,
            run_agents=False,
        )

    failed_run = store.get_latest_pipeline_run("hourly_ingest")
    assert failed_run is not None
    assert failed_run["status"] == "failed"
    assert "Both Open-Meteo and Met.no failed" in (failed_run["error_message"] or "")


def test_fusion_harmonizes_pressure_precip_shift_uv_and_symbols(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    om_ts = pd.DatetimeIndex(["2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z"])
    om_df = pd.DataFrame(
        {
            "timestamp_utc": om_ts,
            "temperature_2m": [24.0, 24.0],
            "apparent_temperature": [24.0, 24.0],
            "weather_code": [0.0, 2.0],
            "relative_humidity_2m": [50.0, 50.0],
            "precipitation": [0.5, 2.0],
            "precipitation_probability": [10.0, 40.0],
            "uv_index": [3.0, np.nan],
            "wind_speed_10m": [10.0, 10.0],
            "wind_direction_10m": [180.0, 180.0],
            "surface_pressure": [988.0, 988.0],
            "cloud_cover": [100.0, 100.0],
        }
    )
    # Met.no at 10:00Z carries next_1_hours precip=4.0 (covering 10:00..11:00Z -> aligns to 11:00Z),
    # MSL pressure ~1013.5 hPa, clear-sky UV 9.0, and conflicting symbol vs Open-Meteo weather_code=0.
    met_ts = pd.DatetimeIndex(["2026-11-18T10:00:00Z", "2026-11-18T11:00:00Z"])
    met_df = pd.DataFrame(
        {
            "timestamp_utc": met_ts,
            "air_temperature": [24.0, 24.0],
            "relative_humidity": [50.0, 50.0],
            "precipitation_amount": [4.0, 8.0],
            "wind_speed": [10.0, 10.0],
            "wind_from_direction": [180.0, 180.0],
            "ultraviolet_index_clear_sky": [9.0, 8.0],
            "air_pressure_at_sea_level": [1013.5, 1013.5],
            "cloud_area_fraction": [100.0, 100.0],
            "symbol_code": ["lightrain_day", "cloudy_day"],
            "weather_code": [61.0, 3.0],
        }
    )

    fused = fuse_weather_sources(om_df, met_df, settings=cfg, align_grid=True)

    # High #6: Met.no MSL pressure is reduced to Lahore elevation (~217m) before 60/40 blend
    expected_met_surf = sea_level_to_surface_pressure(1013.5, cfg.lahore_elevation_m, 24.0)
    assert expected_met_surf is not None
    assert 986.5 <= expected_met_surf <= 990.0
    assert fused.loc[0, "surface_pressure"] == pytest.approx(0.6 * 988.0 + 0.4 * expected_met_surf)

    # High #7: Met.no next_1_hours at 10:00Z (4.0) aligns with Open-Meteo preceding-hour at 11:00Z (2.0)
    assert fused.loc[0, "precipitation_mm"] == pytest.approx(0.5)
    assert fused.loc[1, "precipitation_mm"] == pytest.approx(0.6 * 2.0 + 0.4 * 4.0)

    # High #8: Open-Meteo actual UV (3.0) is not inflated by clear-sky UV (9.0); missing UV uses
    # cloud-attenuated Met.no fallback (8.0 * (1 - 0.75 * 1.0^3) = 2.0)
    assert fused.loc[0, "uv_index"] == pytest.approx(3.0)
    assert fused.loc[1, "uv_index"] == pytest.approx(2.0)

    # Medium: condition_symbol follows weather_code priority (Open-Meteo 0 -> clearsky, not lightrain)
    assert fused.loc[0, "weather_code"] == pytest.approx(0.0)
    assert fused.loc[0, "condition_symbol"] == "clearsky"
    assert wmo_code_to_symbol(None) == "unknown"


def test_imputation_causal_no_future_leakage_and_wind_fallback(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-10-01T00:00:00Z", periods=300, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "us_aqi": np.full(len(ts), np.nan),  # 100% missing -> triggers constant fallback & warning
            "temperature_2m": 22.0 + 4.0 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24.0),
            "precipitation_mm": np.zeros(len(ts)),
            "precipitation_prob": np.zeros(len(ts)),
            "relative_humidity_2m": np.full(len(ts), 55.0),
            "uv_index": np.full(len(ts), 3.0),
            "wind_speed_10m": np.full(len(ts), 8.0),
            "wind_direction_10m": np.full(len(ts), np.nan),  # 100% missing -> must fall back to 315°
            "apparent_temperature": np.full(len(ts), 22.0),
            "surface_pressure": np.full(len(ts), 988.0),
            "cloud_cover": np.full(len(ts), 20.0),
            "weather_code": np.zeros(len(ts)),
            "condition_symbol": ["clearsky"] * len(ts),
            "pm2_5": np.full(len(ts), 40.0),
            "pm10": np.full(len(ts), 80.0),
            "ozone": np.full(len(ts), 40.0),
            "nitrogen_dioxide": np.full(len(ts), 25.0),
            "sulphur_dioxide": np.full(len(ts), 10.0),
            "carbon_monoxide": np.full(len(ts), 400.0),
        }
    )
    # Inject a Tier-2 gap (10h) and Tier-3 gap (30h) well before t=200
    df.loc[30:39, "temperature_2m"] = np.nan
    df.loc[80:109, "temperature_2m"] = np.nan

    perturbed = df.copy()
    perturbed.loc[200:, "temperature_2m"] += 80.0

    res_a = impute_weather_dataframe(df, settings=cfg)
    res_b = impute_weather_dataframe(perturbed, settings=cfg)

    # High #9: Tier 2 and Tier 3 must be strictly causal (changing future rows leaves t < 200 identical)
    pd.testing.assert_series_equal(
        res_a.df.loc[:199, "temperature_2m"],
        res_b.df.loc[:199, "temperature_2m"],
    )

    # High #5: All-NaN wind_direction_10m defaults to 315° (not 0° North)
    assert np.allclose(res_a.df["wind_direction_10m"], 315.0)

    # High #4: Constant fallback and excessive imputation columns surfaced in audit
    assert "us_aqi" in res_a.audit.constant_fallback_columns
    assert "wind_direction_10m" in res_a.audit.constant_fallback_columns
    assert "us_aqi" in res_a.audit.high_imputation_columns
    assert np.allclose(res_a.df["us_aqi"], 110.0)


def test_feature_engineering_gaps_missing_cols_and_tier3_target_exclusion(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    ts = pd.date_range("2026-11-01T00:00:00Z", periods=120, freq="1h", tz="UTC")
    base_df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "us_aqi": np.linspace(100.0, 180.0, len(ts)),
            "temperature_2m": 20.0 + 5.0 * np.sin(np.arange(len(ts)) / 4.0),
            "precipitation_mm": np.zeros(len(ts)),
            "precipitation_prob": np.full(len(ts), 10.0),
            "relative_humidity_2m": np.full(len(ts), 60.0),
            "uv_index": np.full(len(ts), 4.0),
            "wind_speed_10m": np.full(len(ts), 10.0),
            "wind_direction_10m": np.linspace(10.0, 350.0, len(ts)),
            "apparent_temperature": np.full(len(ts), 21.0),
            "surface_pressure": np.full(len(ts), 988.0),
            "cloud_cover": np.full(len(ts), 20.0),
            "weather_code": np.zeros(len(ts), dtype="int64"),
            "pm2_5": np.full(len(ts), 40.0),
            "pm10": np.full(len(ts), 80.0),
            "ozone": np.full(len(ts), 45.0),
            "nitrogen_dioxide": np.full(len(ts), 25.0),
            "sulphur_dioxide": np.full(len(ts), 12.0),
            "carbon_monoxide": np.full(len(ts), 450.0),
            "sources_available": [
                ["open_meteo", "met_no"] if i % 2 == 0 else ["open_meteo"]
                for i in range(len(ts))
            ],
            "source_disagreement_temp": [0.5 if i % 2 == 0 else np.nan for i in range(len(ts))],
        }
    )

    # Drop row 60 (creating a missing hour) and verify lags/targets are NaN across the gap
    gapped = base_df.drop(index=[60]).reset_index(drop=True)
    featured_gapped = engineer_features(gapped, settings=cfg, include_targets=True)
    row_61 = featured_gapped.loc[featured_gapped["timestamp_utc"] == ts[61]].iloc[0]
    row_59 = featured_gapped.loc[featured_gapped["timestamp_utc"] == ts[59]].iloc[0]
    assert pd.isna(row_61["temperature_2m_lag_1h"])
    assert pd.isna(row_59["target_temperature_2m_t1"])
    assert bool(row_59["has_full_24h_targets"]) is False
    assert featured_gapped.loc[0, "is_dual_source"] == 1.0
    assert featured_gapped.loc[1, "is_dual_source"] == 0.0

    no_pm25 = base_df.drop(columns=["pm2_5"])
    featured_no_pm25 = engineer_features(no_pm25, settings=cfg, include_targets=True)
    assert featured_no_pm25["pm25_pm10_ratio"].isna().all()
    assert featured_no_pm25["pm2_5_lag_1h"].isna().all()
    train_incomplete, _ = split_training_and_inference_rows(featured_no_pm25, settings=cfg)
    assert len(train_incomplete) == 0

    # Tier-3 imputed spans (80..82) must exclude rows whose features or 24h targets touch 80..82
    with_tier3 = base_df.copy()
    with_tier3["imputation_tier"] = TIER_NONE
    with_tier3.loc[80:82, "imputation_tier"] = TIER_3_CLIMATOLOGY
    feat_t3 = engineer_features(with_tier3, settings=cfg, include_targets=True)
    train_t3, _ = split_training_and_inference_rows(feat_t3, settings=cfg, exclude_tier3_imputed=True)
    excluded_ts = set(ts[56:83])
    train_ts = set(train_t3["timestamp_utc"])
    assert excluded_ts.isdisjoint(train_ts)
    assert ts[55] in train_ts
    assert ts[83] in train_ts


def test_hourly_ingest_preserves_imputed_flags_and_avoids_double_blend(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path)
    store = SkyCastStorage(cfg)
    t_base = pd.Timestamp("2026-11-18T10:00:00Z")

    # Cold start over 72h where the final hour (10:00Z) has NO Open-Meteo AQI & temp (Met.no only)
    # and hour 08:00Z is completely missing in both sources (imputed!).
    recent_ts = pd.date_range("2026-11-15T10:00:00Z", "2026-11-18T10:00:00Z", freq="1h", tz="UTC")
    w_payload = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_WEATHER_FIELDS)
    aq_payload = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_AIR_QUALITY_FIELDS)

    gap_idx = len(recent_ts) - 3  # 08:00Z
    w_payload["hourly"]["temperature_2m"][gap_idx] = None
    aq_payload["hourly"]["us_aqi"][gap_idx] = None

    met_ts_10 = pd.date_range("2026-11-18T10:00:00Z", "2026-11-18T12:00:00Z", freq="1h", tz="UTC")
    met_payload_10 = _build_synthetic_metno_payload(met_ts_10)

    def _router_run1(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_forecast_url):
            return httpx.Response(200, json=w_payload)
        if url.startswith(cfg.open_meteo_air_quality_url):
            return httpx.Response(200, json=aq_payload)
        if url.startswith(cfg.met_no_forecast_url):
            return httpx.Response(200, json=met_payload_10)
        return httpx.Response(404)

    for _ in range(3):
        httpx_mock.add_callback(_router_run1)

    res1 = run_hourly_ingest(lookback_hours=72, reference_utc=t_base, settings=cfg, storage=store, run_agents=False)
    assert res1["status"] == "completed"

    gap_ts = recent_ts[gap_idx]
    stored1 = store.read_weather_features(start_utc=gap_ts, end_utc=t_base, expand_features_json=False)
    gap_row1 = stored1.loc[stored1["timestamp_utc"] == gap_ts].iloc[0]
    ten_row1 = stored1.loc[stored1["timestamp_utc"] == t_base].iloc[0]
    assert bool(gap_row1["is_imputed"]) is True
    assert "us_aqi" in gap_row1["imputed_columns"]
    assert "temperature_2m" in gap_row1["imputed_columns"]
    assert ten_row1["sources_available"] == ["open_meteo", "met_no"]
    original_temp_10 = float(ten_row1["temperature_2m"])
    original_pressure_10 = float(ten_row1["surface_pressure"])

    # Run 2 at 11:00Z: Open-Meteo is DOWN (503) and Met.no forecast overlaps 10:00Z..12:00Z!
    t_next = t_base + pd.Timedelta(hours=1)

    def _router_run2_outage(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.met_no_forecast_url):
            return httpx.Response(200, json=met_payload_10)
        return httpx.Response(503, text="Open-Meteo outage")

    for _ in range(7):
        httpx_mock.add_callback(_router_run2_outage)

    res2 = run_hourly_ingest(lookback_hours=72, reference_utc=t_next, settings=cfg, storage=store, run_agents=False)
    assert res2["imputation_audit"]["missing_before"] > 0

    stored2 = store.read_weather_features(start_utc=gap_ts, end_utc=t_next, expand_features_json=False)
    gap_row2 = stored2.loc[stored2["timestamp_utc"] == gap_ts].iloc[0]
    ten_row2 = stored2.loc[stored2["timestamp_utc"] == t_base].iloc[0]

    # High #3: previously-imputed gap row MUST keep is_imputed=True and imputed_columns
    assert bool(gap_row2["is_imputed"]) is True
    assert "us_aqi" in gap_row2["imputed_columns"]
    assert "temperature_2m" in gap_row2["imputed_columns"]
    # Dual-source row at 10:00Z MUST NOT double-blend surface_pressure or temperature_2m
    assert ten_row2["sources_available"] == ["open_meteo", "met_no"]
    assert float(ten_row2["temperature_2m"]) == pytest.approx(original_temp_10)
    assert float(ten_row2["surface_pressure"]) == pytest.approx(original_pressure_10)


def test_backfill_bounds_and_imputation_gate(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    cfg = _make_settings(tmp_path, http_backoff_seconds=0.0)
    store = SkyCastStorage(cfg)

    with pytest.raises(ValueError, match="exceeds backfill_max_days"):
        run_backfill(days=cfg.backfill_max_days + 1, settings=cfg, storage=store)

    # Full AQ failure across entire 2-day backfill must abort rather than fabricating us_aqi=110
    ref_hour = pd.Timestamp("2026-11-18T13:00:00Z")
    hist_ts = pd.date_range("2026-11-16T00:00:00Z", "2026-11-18T13:00:00Z", freq="1h", tz="UTC")
    w_payload = _build_synthetic_open_meteo_payload(hist_ts, OPEN_METEO_WEATHER_FIELDS)
    arch_payload = _build_synthetic_open_meteo_payload(hist_ts, OPEN_METEO_ARCHIVE_WEATHER_FIELDS)
    met_payload = _build_synthetic_metno_payload(pd.date_range("2026-11-18T13:00:00Z", periods=2, freq="1h", tz="UTC"))

    def _no_aq_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_archive_url):
            return httpx.Response(200, json=arch_payload)
        if url.startswith(cfg.open_meteo_historical_forecast_url):
            return httpx.Response(200, json=w_payload)
        if url.startswith(cfg.open_meteo_forecast_url):
            return httpx.Response(200, json=w_payload)
        if url.startswith(cfg.met_no_forecast_url):
            return httpx.Response(200, json=met_payload)
        return httpx.Response(503, text="AQ down")

    for _ in range(10):
        httpx_mock.add_callback(_no_aq_router)

    with pytest.raises(RuntimeError, match="fell back to physical constants"):
        run_backfill(days=2, end_utc=ref_hour, settings=cfg, storage=store)


def test_storage_supabase_pagination_unconfigured_guard_arrays_and_hopsworks_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    # High #2: offline=false with zero cloud credentials must report unconfigured and fail loudly
    unconf_cfg = _make_settings(tmp_path, skycast_offline_mode="false")
    unconf_store = SkyCastStorage(unconf_cfg)
    assert unconf_store.storage_mode == "unconfigured"
    minimal_df = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2026-11-18T00:00:00Z")], "temperature_2m": [20.0]})
    with pytest.raises(StorageError, match="unconfigured"):
        unconf_store.upsert_weather_features(minimal_df)
    with pytest.raises(StorageError, match="unconfigured"):
        unconf_store.read_weather_features()
    with pytest.raises(StorageError, match="unconfigured"):
        unconf_store.record_pipeline_run("r1", "hourly_ingest", "running")

    # Medium: SQLite handles must close cleanly so directory removal succeeds immediately on Windows
    off_cfg = _make_settings(tmp_path / "sqlite_lock_test")
    off_store = SkyCastStorage(off_cfg)
    off_store.init_local_db()
    off_store.record_pipeline_run("r1", "hourly_ingest", "completed")
    assert off_store.get_latest_pipeline_run("hourly_ingest") is not None
    shutil.rmtree(off_cfg.skycast_local_data_dir)

    # High #1: Supabase pagination past 1000 rows (2350 rows across 3 pages)
    supa_cfg = _make_settings(
        tmp_path,
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
    )
    supa_store = SkyCastStorage(supa_cfg)
    all_ts = pd.date_range("2026-08-01T00:00:00Z", periods=2350, freq="1h", tz="UTC")
    # Stored newest-first in PostgREST simulation
    db_rows = [
        {
            "timestamp_utc": ts.isoformat(),
            "temperature_2m": float(idx),
            "sources_available": ["open_meteo"],
            "imputed_columns": [],
            "features_json": {},
        }
        for idx, ts in reversed(list(enumerate(all_ts)))
    ]

    class _PagedSelectQuery:
        def __init__(self, rows_desc: list[dict[str, Any]]) -> None:
            self._rows = rows_desc
            self._start = 0
            self._end = 999  # Default PostgREST cap if range omitted

        def select(self, *_args: Any) -> _PagedSelectQuery:
            return self

        def order(self, *_args: Any, **_kwargs: Any) -> _PagedSelectQuery:
            return self

        def gte(self, *_args: Any) -> _PagedSelectQuery:
            return self

        def lte(self, *_args: Any) -> _PagedSelectQuery:
            return self

        def range(self, start: int, end: int) -> _PagedSelectQuery:
            self._start = start
            self._end = end
            return self

        def execute(self) -> Any:
            return type("Resp", (), {"data": self._rows[self._start : self._end + 1]})()

    class _PagedSupabase:
        def table(self, _name: str) -> _PagedSelectQuery:
            return _PagedSelectQuery(db_rows)

    monkeypatch.setattr(supa_store, "_get_supabase", lambda: _PagedSupabase())
    full_supa = supa_store.read_weather_features(expand_features_json=False)
    assert len(full_supa) == 2350
    assert full_supa["temperature_2m"].iloc[0] == 0.0
    assert full_supa["temperature_2m"].iloc[-1] == 2349.0

    limited_supa = supa_store.read_weather_features(limit_hours=1500, expand_features_json=False)
    assert len(limited_supa) == 1500
    assert limited_supa["temperature_2m"].iloc[0] == 850.0

    # Medium: Hopsworks time-pushdown and numpy array preservation in _parse_json_or_default
    hw_cfg = _make_settings(
        tmp_path,
        skycast_offline_mode="false",
        hopsworks_api_key="hw-key",
        hopsworks_project_name="proj",
    )
    hw_store = SkyCastStorage(hw_cfg)
    captured_hw_kwargs: dict[str, Any] = {}

    class _FakeReadFG:
        def read(self, **kwargs: Any) -> pd.DataFrame:
            captured_hw_kwargs.update(kwargs)
            return pd.DataFrame(
                {
                    "timestamp_utc": [pd.Timestamp("2026-11-18T10:00:00Z")],
                    "sources_available": [np.array(["open_meteo", "met_no"], dtype=object)],
                    "imputed_columns": [np.array(["us_aqi"], dtype=object)],
                }
            )

    class _FakeReadFS:
        def get_feature_group(self, **_kwargs: Any) -> _FakeReadFG:
            return _FakeReadFG()

    monkeypatch.setattr(hw_store, "_get_hopsworks_feature_store", lambda: _FakeReadFS())
    hw_df = hw_store.read_weather_features(
        start_utc=pd.Timestamp("2026-11-18T00:00:00Z"),
        end_utc=pd.Timestamp("2026-11-18T12:00:00Z"),
        prefer_hopsworks=True,
    )
    assert "start_time" in captured_hw_kwargs
    assert "end_time" in captured_hw_kwargs
    assert hw_df.loc[0, "sources_available"] == ["open_meteo", "met_no"]
    assert hw_df.loc[0, "imputed_columns"] == ["us_aqi"]


def test_http_retry_after_pacing_and_agent_error_isolation(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import time
    import types

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(float(s)))

    cfg = _make_settings(tmp_path, open_meteo_chunk_delay_seconds=0.3)
    assert "https://github.com/" in cfg.met_no_user_agent

    # Verify Retry-After header honoured on HTTP 429 + exponential backoff on 503
    ts = pd.date_range("2026-11-18T00:00:00Z", periods=2, freq="1h", tz="UTC")
    met_payload = _build_synthetic_metno_payload(ts)
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "3"})
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(status_code=200, json=met_payload)

    met_df = fetch_met_no_forecast(cfg, max_retries=2, backoff_seconds=0.4)
    assert len(met_df) == 2
    assert sleeps == [pytest.approx(3.0), pytest.approx(0.8)]

    # Verify chunk pacing across > chunk_days historical spans
    sleeps.clear()
    arch_payload = _build_synthetic_open_meteo_payload(ts, OPEN_METEO_ARCHIVE_WEATHER_FIELDS)
    hist_payload = _build_synthetic_open_meteo_payload(ts, OPEN_METEO_WEATHER_FIELDS)
    aq_payload = _build_synthetic_open_meteo_payload(ts, (*OPEN_METEO_AIR_QUALITY_FIELDS, "uv_index"))

    def _chunk_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_archive_url):
            return httpx.Response(200, json=arch_payload)
        if url.startswith(cfg.open_meteo_historical_forecast_url):
            return httpx.Response(200, json=hist_payload)
        return httpx.Response(200, json=aq_payload)

    for _ in range(6):
        httpx_mock.add_callback(_chunk_router)

    fetch_open_meteo_historical(
        "2026-11-15",
        "2026-11-18",
        settings=cfg,
        chunk_days=2,
        chunk_delay_seconds=0.3,
        backoff_seconds=0.0,
    )
    assert sleeps == [pytest.approx(0.3)]

    # Dynamic lookback minimum derives from cfg.lag_hours
    custom_lag_cfg = _make_settings(tmp_path, lag_hours=(1, 72))
    with pytest.raises(ValueError, match="must be >= 72"):
        run_hourly_ingest(lookback_hours=48, settings=custom_lag_cfg, run_agents=False)

    # Agent pipeline: transitive ModuleNotFoundError inside graph module must propagate;
    # runtime exception inside run_agent_pipeline must NOT mark persisted ingestion failed.
    ref_hour = pd.Timestamp("2026-11-18T13:00:00Z")
    recent_ts = pd.date_range("2026-11-15T13:00:00Z", "2026-11-18T13:00:00Z", freq="1h", tz="UTC")
    w_p = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_WEATHER_FIELDS)
    aq_p = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_AIR_QUALITY_FIELDS)
    met_p = _build_synthetic_metno_payload(pd.date_range("2026-11-18T13:00:00Z", periods=2, freq="1h", tz="UTC"))

    def _ok_router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_forecast_url):
            return httpx.Response(200, json=w_p)
        if url.startswith(cfg.open_meteo_air_quality_url):
            return httpx.Response(200, json=aq_p)
        return httpx.Response(200, json=met_p)

    for _ in range(6):
        httpx_mock.add_callback(_ok_router)

    store = SkyCastStorage(cfg)
    broken_mod = types.ModuleType("skycast.agents.graph")

    def _raise_runtime(**_kwargs: Any) -> Any:
        raise RuntimeError("agent LLM crash")

    broken_mod.run_agent_pipeline = _raise_runtime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skycast.agents.graph", broken_mod)

    res_agent_err = run_hourly_ingest(lookback_hours=72, reference_utc=ref_hour, settings=cfg, storage=store, run_agents=True)
    assert res_agent_err["status"] == "completed"
    assert res_agent_err["rows_persisted"] == 73
    assert res_agent_err["agents"]["status"] == "failed"
    assert "agent LLM crash" in res_agent_err["agents"]["error"]

    # Broken internal import inside skycast.agents.graph must propagate, not report not_available
    monkeypatch.delitem(sys.modules, "skycast.agents.graph", raising=False)
    import builtins

    orig_import = builtins.__import__

    def _broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "skycast.agents.graph":
            raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)
    with pytest.raises(ModuleNotFoundError, match="missing_dep"):
        run_hourly_ingest(
            lookback_hours=72,
            reference_utc=ref_hour + pd.Timedelta(hours=1),
            settings=cfg,
            storage=store,
            run_agents=True,
        )
    monkeypatch.setattr(builtins, "__import__", orig_import)

    # Secondary failure in record_pipeline_run must not mask the original ingestion exception
    for _ in range(9):
        httpx_mock.add_response(status_code=503)

    orig_record = store.record_pipeline_run

    def _flaky_record(run_id: str, pipeline_type: str, status: str, **kwargs: Any) -> None:
        if status == "failed":
            raise StorageError("DB unreachable during error log")
        orig_record(run_id, pipeline_type, status, **kwargs)

    monkeypatch.setattr(store, "record_pipeline_run", _flaky_record)
    with pytest.raises(RuntimeError, match="Both Open-Meteo and Met.no failed"):
        run_hourly_ingest(
            lookback_hours=72,
            reference_utc=ref_hour + pd.Timedelta(hours=2),
            settings=cfg,
            storage=store,
            run_agents=False,
        )


