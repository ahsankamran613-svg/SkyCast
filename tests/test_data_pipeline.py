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
            "precipitation_amount": [0.0, 2.0],
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

    # Hour 01:00 only had Open-Meteo
    assert fused.loc[1, "sources_available"] == ["open_meteo"]
    assert fused.loc[1, "temperature_2m"] == pytest.approx(22.0)

    # Hour 02:00 missing in both -> exposed as NaN for imputer
    assert fused.loc[2, "sources_available"] == []
    assert pd.isna(fused.loc[2, "temperature_2m"])

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

