"""Comprehensive tests for SkyCast Component 3 M2 (`skycast.ml.eda` and `skycast.ml.evaluate`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skycast.config.settings import Settings
from skycast.features.engineer import (
    engineer_features,
    sanitize_json_scalar,
    split_training_and_inference_rows,
)
from skycast.ml.eda import compute_eda_summary, save_eda_summary_json
from skycast.ml.evaluate import (
    ChronologicalSplit,
    apply_residual_quantiles,
    build_comparison_series_rows,
    clamp_predictions_and_bounds,
    compute_empirical_residual_quantiles,
    compute_improvement_over_naive_pct,
    extract_target_matrix,
    extract_wind_component_targets,
    get_target_column_names,
    get_wind_component_target_columns,
    reconstruct_wind_direction,
    score_multi_horizon,
    split_and_scale,
)


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


def _build_synthetic_weather_df(
    hours: int,
    *,
    start_utc: str = "2026-10-01T00:00:00Z",
    constant_precip: bool = False,
) -> pd.DataFrame:
    ts = pd.date_range(start=start_utc, periods=hours, freq="1h", tz="UTC")
    idx = np.arange(hours, dtype=np.float64)
    # Use Asia/Karachi diurnal signal (UTC+5)
    pkt_hours = (ts.hour.to_numpy(dtype=np.float64) + 5.0) % 24.0
    diurnal = np.sin(2.0 * np.pi * (pkt_hours - 6.0) / 24.0)

    wind_speed = 8.0 + 3.5 * np.sin(2.0 * np.pi * idx / 24.0) + 0.5 * np.cos(idx / 7.0)
    humidity = 60.0 - 15.0 * diurnal + 2.0 * np.sin(idx / 11.0)
    # Inverse correlation with wind_speed, positive with humidity
    pm2_5 = 90.0 - 5.0 * wind_speed + 0.8 * humidity + 0.2 * np.sin(idx / 5.0)
    us_aqi = 60.0 + 1.5 * pm2_5

    precip = np.zeros(hours, dtype=np.float64) if constant_precip else np.maximum(0.0, 2.0 * np.sin(idx / 13.0))

    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "us_aqi": us_aqi,
            "temperature_2m": 25.0 + 6.0 * diurnal + 0.05 * idx,
            "precipitation_mm": precip,
            "precipitation_prob": 10.0 + 15.0 * (precip > 0).astype(np.float64),
            "relative_humidity_2m": humidity,
            "uv_index": np.maximum(0.0, 7.0 * diurnal),
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": (345.0 + 4.0 * idx) % 360.0,
            "apparent_temperature": 26.0 + 6.0 * diurnal + 0.05 * idx,
            "weather_code": np.where(precip > 0, 61, 0),
            "surface_pressure": 1012.0 + 1.5 * np.sin(idx / 12.0),
            "cloud_cover": 20.0 + 10.0 * np.cos(idx / 8.0),
            "pm2_5": pm2_5,
            "pm10": 1.6 * pm2_5 + 10.0,
            "ozone": 42.0 + 8.0 * diurnal,
            "nitrogen_dioxide": 25.0 - 4.0 * diurnal,
            "sulphur_dioxide": 11.0 + 1.0 * np.sin(idx / 9.0),
            "carbon_monoxide": 420.0 + 30.0 * np.cos(idx / 6.0),
            "sources_available": [["open_meteo", "met_no"] for _ in range(hours)],
            "is_imputed": [i % 20 == 0 for i in range(hours)],
            "imputation_tier": ["tier1_linear" if i % 20 == 0 else "none" for i in range(hours)],
        }
    )


def _build_trainable_df(
    tmp_path: Path,
    *,
    raw_hours: int = 280,
    settings: Settings | None = None,
    constant_precip: bool = False,
) -> tuple[Settings, pd.DataFrame]:
    cfg = settings or _make_settings(tmp_path)
    raw = _build_synthetic_weather_df(raw_hours, constant_precip=constant_precip)
    featured = engineer_features(raw, settings=cfg, include_targets=True)
    trainable, _ = split_training_and_inference_rows(featured, settings=cfg)
    return cfg, trainable


# ---------------------------------------------------------------------------
# 1. Embargoed Chronological Split & +1000.0 Leakage Perturbation
# ---------------------------------------------------------------------------


def test_split_and_scale_embargo_and_leakage_perturbation(tmp_path: Path) -> None:
    cfg, trainable = _build_trainable_df(tmp_path, raw_hours=300)
    H = cfg.forecast_horizon_hours
    split = split_and_scale(trainable, settings=cfg)
    assert isinstance(split, ChronologicalSplit)
    assert split.embargo_hours == H

    n_eff = len(trainable) - 2 * H
    expected_train = int(n_eff * cfg.train_split_ratio)
    expected_val = int(n_eff * cfg.val_split_ratio)
    expected_test = n_eff - expected_train - expected_val

    assert len(split.train_df) == expected_train
    assert len(split.val_df) == expected_val
    assert len(split.test_df) == expected_test
    assert split.embargo_train_val_df is not None and len(split.embargo_train_val_df) == H
    assert split.embargo_val_test_df is not None and len(split.embargo_val_test_df) == H

    assert split.val_df["timestamp_utc"].iloc[0] > split.train_df["timestamp_utc"].iloc[-1] + pd.Timedelta(hours=H)
    assert split.test_df["timestamp_utc"].iloc[0] > split.val_df["timestamp_utc"].iloc[-1] + pd.Timedelta(hours=H)

    # Perturb all post-train rows (embargo, val, test) by +1000.0
    train_end_ts = split.train_df["timestamp_utc"].max()
    perturbed = trainable.copy()
    post_mask = perturbed["timestamp_utc"] > train_end_ts
    assert int(post_mask.sum()) > 0
    for col in [*split.feature_columns, *get_target_column_names("temperature_2m", settings=cfg)]:
        if pd.api.types.is_bool_dtype(perturbed[col].dtype):
            continue
        perturbed.loc[post_mask, col] = perturbed.loc[post_mask, col].astype(np.float64) + 1000.0

    split_pert = split_and_scale(perturbed, settings=cfg)

    pd.testing.assert_frame_equal(split.train_df, split_pert.train_df)
    assert np.array_equal(split.scaler.mean_, split_pert.scaler.mean_)
    assert np.array_equal(split.scaler.scale_, split_pert.scaler.scale_)
    assert np.array_equal(split.X_train_scaled, split_pert.X_train_scaled)
    assert not np.array_equal(split.X_val_scaled, split_pert.X_val_scaled)
    assert not np.array_equal(split.X_test_scaled, split_pert.X_test_scaled)

    # Train empirical residual quantiles remain bit-for-bit identical
    y_train = extract_target_matrix(split.train_df, "temperature_2m", settings=cfg)
    y_train_pert = extract_target_matrix(split_pert.train_df, "temperature_2m", settings=cfg)
    naive_pred = np.full_like(y_train, 25.0)
    ql_1, qh_1 = compute_empirical_residual_quantiles(y_train, naive_pred, settings=cfg, metric="temperature_2m")
    ql_2, qh_2 = compute_empirical_residual_quantiles(y_train_pert, naive_pred, settings=cfg, metric="temperature_2m")
    assert np.array_equal(ql_1, ql_2)
    assert np.array_equal(qh_1, qh_2)


# ---------------------------------------------------------------------------
# 2. Non-default Settings Proof (No Hardcoded Constants)
# ---------------------------------------------------------------------------


def test_custom_settings_no_hardcoded_constants(tmp_path: Path) -> None:
    cfg_custom = _make_settings(
        tmp_path,
        forecast_horizon_hours=12,
        train_split_ratio=0.6,
        val_split_ratio=0.2,
        test_split_ratio=0.2,
        lahore_timezone="UTC",
        sarimax_conf_alpha=0.1,
        supabase_history_days=2,
    )
    _, trainable = _build_trainable_df(tmp_path, raw_hours=180, settings=cfg_custom)
    split = split_and_scale(trainable, settings=cfg_custom)

    assert split.embargo_hours == 12
    n_eff = len(trainable) - 24
    assert len(split.train_df) == int(n_eff * 0.6)
    assert len(split.val_df) == int(n_eff * 0.2)
    assert len(split.test_df) == n_eff - len(split.train_df) - len(split.val_df)
    assert split.val_df["timestamp_utc"].iloc[0] > split.train_df["timestamp_utc"].iloc[-1] + pd.Timedelta(hours=12)

    y_train = extract_target_matrix(split.train_df, "temperature_2m", settings=cfg_custom)
    assert y_train.shape == (len(split.train_df), 12)

    # Alpha=0.1 yields strictly wider interval than alpha=0.2 on spread residuals
    rng = np.random.default_rng(42)
    yt = rng.normal(25.0, 4.0, size=(200, 12))
    yp = np.full((200, 12), 25.0)
    ql_01, qh_01 = compute_empirical_residual_quantiles(yt, yp, settings=cfg_custom)
    ql_02, qh_02 = compute_empirical_residual_quantiles(yt, yp, alpha=0.2, settings=cfg_custom)
    assert np.all((qh_01 - ql_01) >= (qh_02 - ql_02))

    # Diurnal heatmap timezone shifts by -5h between Asia/Karachi and UTC
    raw = _build_synthetic_weather_df(72)
    cfg_pkt = _make_settings(tmp_path, lahore_timezone="Asia/Karachi")
    eda_utc = compute_eda_summary(raw, settings=cfg_custom)
    eda_pkt = compute_eda_summary(raw, settings=cfg_pkt)
    assert eda_utc["summary_json"]["diurnal_heatmap"]["timezone"] == "UTC"
    assert eda_pkt["summary_json"]["diurnal_heatmap"]["timezone"] == "Asia/Karachi"

    utc_hours = eda_utc["summary_json"]["diurnal_heatmap"]["metrics"]["temperature_2m"]
    pkt_hours = eda_pkt["summary_json"]["diurnal_heatmap"]["metrics"]["temperature_2m"]
    peak_utc = max(utc_hours, key=lambda r: r["mean"])["hour"]
    peak_pkt = max(pkt_hours, key=lambda r: r["mean"])["hour"]
    assert (peak_pkt - peak_utc) % 24 == 5

    # Seasonal decomposition recent window capped at supabase_history_days * 24 (= 48)
    decomp = eda_utc["summary_json"]["seasonal_decomposition"]["temperature_2m"]
    assert decomp["status"] == "available"
    assert len(decomp["timestamps_utc"]) == 48


# ---------------------------------------------------------------------------
# 3. Circular Wind Math, (sin, cos) Targets & Post-Sanitization Bounds
# ---------------------------------------------------------------------------


def test_circular_wind_scoring_sin_cos_and_bounds_invariant(tmp_path: Path) -> None:
    # Shortest angular error across 0/360 wrap
    scores_wrap = score_multi_horizon(
        np.array([[359.0, 1.0]]),
        np.array([[1.0, 359.0]]),
        metric="wind_direction_10m",
    )
    assert scores_wrap["rmse"] == pytest.approx(2.0, abs=1e-9)
    assert scores_wrap["mae"] == pytest.approx(2.0, abs=1e-9)

    # Angular R² properties
    yt = np.array([[350.0, 10.0], [355.0, 5.0]])
    assert score_multi_horizon(yt, yt, metric="wind_direction_10m")["r2"] == pytest.approx(1.0)
    # Circular mean of {350, 10, 355, 5} is 0.0 deg
    assert score_multi_horizon(yt, np.zeros_like(yt), metric="wind_direction_10m")["r2"] == pytest.approx(0.0, abs=1e-9)
    # Opposite direction (180 deg) performs strictly worse than circular mean
    assert score_multi_horizon(yt, np.full_like(yt, 180.0), metric="wind_direction_10m")["r2"] < 0.0

    # (sin, cos) extraction and reconstruction
    cfg, trainable = _build_trainable_df(tmp_path, raw_hours=96)
    sin_cols, cos_cols = get_wind_component_target_columns(settings=cfg)
    assert len(sin_cols) == cfg.forecast_horizon_hours
    assert len(cos_cols) == cfg.forecast_horizon_hours
    y_sin, y_cos = extract_wind_component_targets(trainable, settings=cfg)
    y_deg_direct = extract_target_matrix(trainable, "wind_direction_10m", settings=cfg)
    y_deg_recon = reconstruct_wind_direction(y_sin, y_cos)
    ang_diff = ((y_deg_recon - y_deg_direct + 180.0) % 360.0) - 180.0
    assert np.all(np.abs(ang_diff) < 1e-6)
    assert reconstruct_wind_direction(0.0, 0.0) == 0.0
    assert reconstruct_wind_direction(-1e-16, 1.0) == 0.0

    # Circular bounds invariant across adversarial edge values + sanitize_json_scalar
    edge_preds = np.array([[-10.0, -1e-17, -1e-7, 0.0, 359.9999, 359.999999, 360.0, 370.0]])
    raw_low = edge_preds - 25.0
    raw_high = edge_preds + 25.0
    p_c, l_c, u_c = clamp_predictions_and_bounds("wind_direction_10m", edge_preds, raw_low, raw_high)
    assert np.all((0.0 <= l_c) & (l_c <= p_c) & (p_c <= u_c) & (u_c <= 359.9999))

    for l_val, p_val, u_val in zip(l_c.ravel(), p_c.ravel(), u_c.ravel()):
        sl = sanitize_json_scalar(l_val)
        sp = sanitize_json_scalar(p_val)
        su = sanitize_json_scalar(u_val)
        assert 0.0 <= sl <= sp <= su <= 359.9999

    # apply_residual_quantiles with biased quantiles (both positive & both negative)
    q_low = np.full(8, 5.0)
    q_high = np.full(8, 15.0)
    p_a, l_a, u_a = apply_residual_quantiles("wind_direction_10m", edge_preds, q_low, q_high)
    assert np.all((0.0 <= l_a) & (l_a <= p_a) & (p_a <= u_a) & (u_a <= 359.9999))

    # Non-circular clamping with biased negative quantiles
    t_pred = np.array([[-30.0, 25.0, 70.0]])
    tp, tl, tu = apply_residual_quantiles(
        "temperature_2m",
        t_pred,
        np.array([-10.0, -5.0, -2.0]),
        np.array([-1.0, 3.0, 5.0]),
    )
    assert np.all((-20.0 <= tl) & (tl <= tp) & (tp <= tu) & (tu <= 60.0))


# ---------------------------------------------------------------------------
# 4. EDA Summary & Explicit Save Helper
# ---------------------------------------------------------------------------


def test_eda_summary_available_unavailable_corr_audit_and_save(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)
    df_96 = _build_synthetic_weather_df(96, constant_precip=True)
    # Introduce genuine missing values in uv_index (ensure zero .fillna(0)!)
    df_96.loc[0, "uv_index"] = np.nan
    df_96.loc[1, "uv_index"] = np.nan

    audit_input = {"tier1_count": 5, "tier2_count": 0, "tier3_count": 0, "imputation_rate_pct": 1.2}
    summary = compute_eda_summary(df_96, settings=cfg, imputation_audit=audit_input, run_id="eda_test_96")

    assert set(summary.keys()) == {
        "run_id",
        "computed_at_utc",
        "sample_count",
        "date_range_start",
        "date_range_end",
        "summary_json",
    }
    assert summary["run_id"] == "eda_test_96"
    assert summary["sample_count"] == 96
    sj = summary["summary_json"]

    # Zero .fillna(0) check on missing values
    uv_dist = sj["distributions"]["uv_index"]
    assert uv_dist["valid_count"] == 94
    assert uv_dist["missing_count"] == 2
    assert "circular_mean_deg" in sj["distributions"]["wind_direction_10m"]

    # Correlations: pm2_5 vs wind_speed < 0, pm2_5 vs humidity > 0, constant precip -> None
    corr = sj["correlation_matrix"]
    assert corr["highlights"]["pm2_5_vs_wind_speed_10m"] < 0.0
    assert corr["highlights"]["pm2_5_vs_relative_humidity_2m"] > 0.0
    assert "precipitation_mm" in corr["constant_columns"]
    assert corr["matrix"]["precipitation_mm"]["temperature_2m"] is None

    # Seasonal decomposition available on >= 48 contiguous rows; unavailable on uv_index if <48? Wait: 94 contiguous -> available
    assert sj["seasonal_decomposition"]["temperature_2m"]["status"] == "available"
    assert sj["seasonal_decomposition"]["temperature_2m"]["period"] == 24
    assert sj["seasonal_decomposition"]["pm2_5"]["status"] == "available"

    # Provided imputation audit
    assert sj["imputation_audit"]["source"] == "provided"
    assert sj["imputation_audit"]["tier1_count"] == 5

    # Strict JSON safety + tmp_path save without touching docs/
    json.dumps(summary, allow_nan=False)
    out_file = save_eda_summary_json(summary, output_path=tmp_path / "eda_summary.json", settings=cfg)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "eda_test_96"

    # < 48 rows -> seasonal decomposition explicitly unavailable (never fake zeros)
    df_47 = _build_synthetic_weather_df(47)
    summary_47 = compute_eda_summary(df_47, settings=cfg)
    assert summary_47["summary_json"]["seasonal_decomposition"]["temperature_2m"]["status"] == "unavailable"
    assert summary_47["summary_json"]["imputation_audit"]["source"] == "dataframe_derived"
    json.dumps(summary_47, allow_nan=False)

    # Gapped series with 60 valid rows split into two 30h blocks -> unavailable due to contiguity < 48
    df_gapped = _build_synthetic_weather_df(70)
    df_gapped = pd.concat([df_gapped.iloc[:30], df_gapped.iloc[40:]]).reset_index(drop=True)
    # Also drop tier columns to verify explicit unavailable imputation_audit
    df_gapped = df_gapped.drop(columns=["is_imputed", "imputation_tier"])
    summary_gapped = compute_eda_summary(df_gapped, settings=cfg)
    assert summary_gapped["summary_json"]["seasonal_decomposition"]["temperature_2m"]["status"] == "unavailable"
    assert summary_gapped["summary_json"]["imputation_audit"]["status"] == "unavailable"
    # Partial hours (30 + 30 across 70h span leaves some hours present/absent in smaller slice)
    df_partial = _build_synthetic_weather_df(10)
    summary_partial = compute_eda_summary(df_partial, settings=cfg)
    empty_buckets = [
        h for h in summary_partial["summary_json"]["diurnal_heatmap"]["metrics"]["temperature_2m"]
        if h["count"] == 0
    ]
    assert len(empty_buckets) == 14
    assert all(b["mean"] is None and b["median"] is None and b["std"] is None for b in empty_buckets)


# ---------------------------------------------------------------------------
# 5. Failure, Cold-Start & Degenerate Input Guards
# ---------------------------------------------------------------------------


def test_eda_and_eval_failure_and_degenerate_inputs(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path)

    # Empty / missing timestamp_utc / NaT timestamps
    with pytest.raises(ValueError):
        compute_eda_summary(pd.DataFrame(), settings=cfg)
    with pytest.raises(ValueError):
        compute_eda_summary(pd.DataFrame({"temperature_2m": [20.0]}), settings=cfg)
    with pytest.raises(ValueError):
        compute_eda_summary(pd.DataFrame({"timestamp_utc": ["not-a-date"], "temperature_2m": [20.0]}), settings=cfg)

    with pytest.raises(ValueError):
        split_and_scale(pd.DataFrame(), settings=cfg)
    with pytest.raises(ValueError):
        split_and_scale(pd.DataFrame({"feat": [1.0, 2.0]}), settings=cfg)

    # Insufficient rows after 2*H embargo
    _, trainable_full = _build_trainable_df(tmp_path, raw_hours=150, settings=cfg)
    with pytest.raises(ValueError):
        split_and_scale(trainable_full.iloc[:40], settings=cfg)

    # Duplicate timestamps rejected
    dup_df = pd.concat([trainable_full.iloc[:5], trainable_full.iloc[:5]], ignore_index=True)
    with pytest.raises(ValueError):
        split_and_scale(dup_df, settings=cfg)

    # Sub-hourly timestamps where 24 skipped rows span <= 24h embargo gap
    sub_hourly = trainable_full.iloc[:60].copy()
    sub_hourly["timestamp_utc"] = pd.date_range("2026-10-01T00:00:00Z", periods=60, freq="30min", tz="UTC")
    with pytest.raises(ValueError, match="embargo gap"):
        split_and_scale(sub_hourly, settings=cfg)

    # NaN in feature column -> fail fast (no silent .fillna(0))
    nan_feat_df = trainable_full.copy()
    nan_feat_df.loc[0, "temperature_2m"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        split_and_scale(nan_feat_df, settings=cfg)

    # Inf in target column -> fail fast
    inf_tgt_df = trainable_full.copy()
    inf_tgt_df.loc[0, "target_temperature_2m_t1"] = np.inf
    with pytest.raises(ValueError, match="target"):
        split_and_scale(inf_tgt_df, settings=cfg)

    # Zero-variance target & naive_rmse == 0 deterministic finite guards
    zeros = np.zeros((10, 24), dtype=np.float64)
    perfect_zero = score_multi_horizon(zeros, zeros, metric="precipitation_mm")
    assert perfect_zero["rmse"] == 0.0 and perfect_zero["r2"] == 1.0
    imperfect_zero = score_multi_horizon(zeros, np.ones_like(zeros), metric="precipitation_mm")
    assert imperfect_zero["r2"] == 0.0
    assert compute_improvement_over_naive_pct(0.0, 0.0) == 0.0
    assert np.isfinite(compute_improvement_over_naive_pct(1.5, 0.0))
    with pytest.raises(ValueError):
        compute_improvement_over_naive_pct(-0.1, 1.0)

    # Non-finite prediction guards
    with pytest.raises(ValueError):
        score_multi_horizon(zeros, np.full_like(zeros, np.nan), metric="precipitation_mm")
    with pytest.raises(ValueError):
        clamp_predictions_and_bounds("temperature_2m", np.array([[np.inf]]), np.array([[0.0]]), np.array([[10.0]]))

    # Constant feature column scales cleanly with scale_ == 1.0 and zero NaNs
    const_cfg, const_trainable = _build_trainable_df(tmp_path, raw_hours=150, constant_precip=True)
    const_split = split_and_scale(const_trainable, settings=const_cfg)
    assert np.isfinite(const_split.X_train_scaled).all()
    precip_idx = const_split.feature_columns.index("precipitation_mm")
    assert const_split.scaler.scale_[precip_idx] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. model_comparison_series Builder
# ---------------------------------------------------------------------------


def test_build_comparison_series_rows_alignment_and_sanitization(tmp_path: Path) -> None:
    cfg = _make_settings(tmp_path, forecast_horizon_hours=12)
    origins = pd.date_range("2026-10-10T00:00:00Z", periods=3, freq="1h", tz="UTC")
    actual = np.full((3, 12), 20.0)
    actual[:, 11] = [21.5, np.nan, 23.5]

    sarimax_preds = np.full((3, 12), 19.0)
    sarimax_preds[:, 11] = [21.0, 22.0, 23.0]
    rf_preds = np.full((3, 12), 20.0)
    rf_preds[:, 11] = [21.25, 22.25, 23.25]

    rows = build_comparison_series_rows(
        train_run_id="train_run_001",
        metric="temperature_2m",
        origin_timestamps=origins,
        actual=actual,
        predictions_by_family={
            "SARIMAX": sarimax_preds,
            "random_forest": rf_preds,
            "LSTM": None,
        },
        settings=cfg,
    )
    assert len(rows) == 3
    assert rows[0]["timestamp_utc"] == "2026-10-10T12:00:00+00:00"
    assert rows[0]["actual"] == 21.5
    assert rows[1]["actual"] is None
    assert rows[0]["sarimax"] == 21.0
    assert rows[0]["random_forest"] == 21.25
    assert rows[0]["ridge"] is None
    assert rows[0]["lstm"] is None
    assert rows[0]["naive_24h"] is None
    json.dumps(rows, allow_nan=False)

    # Circular clamping in comparison series
    wind_rows = build_comparison_series_rows(
        train_run_id="train_run_002",
        metric="wind_direction_10m",
        origin_timestamps=origins[:1],
        actual=np.array([350.0]),
        predictions_by_family={"Naive24h": np.array([360.0])},
        settings=cfg,
    )
    assert wind_rows[0]["naive_24h"] == 0.0

    # Unknown model family rejected
    with pytest.raises(ValueError, match="Unknown model family"):
        build_comparison_series_rows(
            train_run_id="train_run_001",
            metric="temperature_2m",
            origin_timestamps=origins,
            actual=actual,
            predictions_by_family={"XGBoost": sarimax_preds},
            settings=cfg,
        )
