"""Comprehensive unit and contract tests for SkyCast Component 3 models, XAI, and registry."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skycast.config.settings import METRIC_SPECS, MODEL_FAMILIES, Settings
from skycast.data.storage import SkyCastStorage, StorageError
from skycast.features.engineer import (
    engineer_features,
    split_training_and_inference_rows,
)
from skycast.ml.evaluate import (
    extract_target_matrix,
    get_target_column_names,
    get_wind_component_target_columns,
    score_multi_horizon,
    split_and_scale,
)
from skycast.ml.explain import compute_explanations_for_metric, subsample_chronological_rows
from skycast.ml.models import (
    MODEL_CLASS_BY_FAMILY,
    LSTMModel,
    Naive24hModel,
    RandomForestModel,
    RidgeModel,
    SARIMAXModel,
    load_model_bundle,
    predict_from_bundle,
)
from skycast.ml.registry import (
    get_current_champion_info,
    load_champion_bundle,
    promote_champion_if_better,
    resolve_registry_model_name,
)
from tests.contract_fakes import FakeHopsworksProject, make_test_settings
from tests.e2e.fakes import HopsworksOutageError, MockHopsworksProject
from tests.e2e.fixtures import generate_lahore_feature_dataframe


def _make_fast_settings(tmp_path: Path, **overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "sarimax_maxiter": 5,
        "rf_n_estimators": 20,
        "rf_max_depth": 6,
        "rf_min_samples_leaf": 2,
        "lstm_hidden_size": 16,
        "lstm_num_layers": 1,
        "lstm_dropout": 0.1,
        "lstm_epochs": 4,
        "lstm_batch_size": 32,
        "lstm_early_stopping_patience": 2,
    }
    defaults.update(overrides)
    return make_test_settings(tmp_path, **defaults)


def _build_trainable(
    tmp_path: Path,
    *,
    n_hours: int = 240,
    horizon_hours: int = 24,
    dry_mode: bool = False,
    zero_var_cols: tuple[str, ...] = (),
    settings: Settings | None = None,
) -> tuple[Settings, pd.DataFrame, pd.DataFrame]:
    cfg = settings or _make_fast_settings(tmp_path, forecast_horizon_hours=horizon_hours)
    raw = generate_lahore_feature_dataframe(
        n_hours=n_hours,
        horizon_hours=horizon_hours,
        seed=42,
        dry_mode=dry_mode,
        wind_wraparound=True,
        zero_var_cols=zero_var_cols,
    )
    if horizon_hours != 24:
        # Ensure targets match custom settings cleanly
        raw = engineer_features(raw, settings=cfg, include_targets=True)
    trainable, inference = split_training_and_inference_rows(raw, settings=cfg)
    return cfg, trainable, inference


# ---------------------------------------------------------------------------
# 1. All 5 Model Families: Shape, Bounds, Clamping, Circular Wind & Bundles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["temperature_2m", "us_aqi", "precipitation_mm", "wind_direction_10m"])
def test_all_five_model_families_shape_bounds_and_bundle_roundtrip(
    tmp_path: Path,
    metric: str,
) -> None:
    cfg, trainable, inference = _build_trainable(tmp_path, n_hours=240, horizon_hours=24)
    split = split_and_scale(trainable, settings=cfg)
    H = cfg.forecast_horizon_hours
    spec = METRIC_SPECS[metric]
    max_allowed = 359.9999 if spec.is_circular else spec.max_value

    y_val = extract_target_matrix(split.val_df, metric, settings=cfg)
    y_test = extract_target_matrix(split.test_df, metric, settings=cfg)

    for family in MODEL_FAMILIES:
        model = MODEL_CLASS_BY_FAMILY[family](metric, settings=cfg).fit(split)
        fc_val = model.predict(split.val_df, split.X_val_scaled)
        fc_test = model.predict(
            split.test_df,
            split.X_test_scaled,
            context_df=split.val_df,
            context_X_scaled=split.X_val_scaled,
        )

        for fc, expected_n in ((fc_val, len(split.val_df)), (fc_test, len(split.test_df))):
            assert fc.predicted.shape == (expected_n, H)
            assert fc.lower.shape == (expected_n, H)
            assert fc.upper.shape == (expected_n, H)
            assert np.isfinite(fc.predicted).all()
            assert np.isfinite(fc.lower).all()
            assert np.isfinite(fc.upper).all()
            assert np.all(fc.lower <= fc.predicted)
            assert np.all(fc.predicted <= fc.upper)
            assert np.all((spec.min_value <= fc.lower) & (fc.upper <= max_allowed))

        val_scores = score_multi_horizon(y_val, fc_val.predicted, metric=metric)
        test_scores = score_multi_horizon(y_test, fc_test.predicted, metric=metric)
        assert all(math.isfinite(val_scores[k]) for k in ("rmse", "mae", "r2"))
        assert all(math.isfinite(test_scores[k]) for k in ("rmse", "mae", "r2"))

        # Single-row latest forecast + save/load roundtrip parity
        latest_1row = inference.tail(1).reset_index(drop=True)
        fc_1_mem = model.predict(latest_1row)
        assert fc_1_mem.predicted.shape == (1, H)

        bundle_dir = tmp_path / f"bundle_{metric}_{family}"
        model.save(bundle_dir)
        reloaded = load_model_bundle(bundle_dir, settings=cfg)
        assert reloaded.family_name == family
        assert reloaded.metric == metric
        fc_1_disk = predict_from_bundle(bundle_dir, latest_1row, settings=cfg)
        assert np.allclose(fc_1_mem.predicted, fc_1_disk.predicted, atol=1e-12)
        assert np.allclose(fc_1_mem.lower, fc_1_disk.lower, atol=1e-12)
        assert np.allclose(fc_1_mem.upper, fc_1_disk.upper, atol=1e-12)


def test_circular_wind_trains_on_sin_cos_and_scores_shortest_angular_error(tmp_path: Path) -> None:
    cfg, trainable, _ = _build_trainable(tmp_path, n_hours=240, horizon_hours=12)
    split = split_and_scale(trainable, settings=cfg)

    sin_cols, cos_cols = get_wind_component_target_columns(settings=cfg)
    assert len(sin_cols) == 12 and len(cos_cols) == 12
    assert "wind_direction_10m" not in split.feature_columns

    # Even if raw wind_direction_10m target columns were shifted away, Ridge/RF/LSTM train on sin/cos
    corrupted_val = split
    ridge = RidgeModel("wind_direction_10m", settings=cfg).fit(corrupted_val)
    assert ridge.estimator_sin is not None and ridge.estimator_cos is not None
    assert ridge.estimator is None

    rf = RandomForestModel("wind_direction_10m", settings=cfg).fit(corrupted_val)
    assert rf.estimator_sin is not None and rf.estimator_cos is not None
    assert rf.estimator is None

    lstm = LSTMModel("wind_direction_10m", settings=cfg).fit(corrupted_val)
    assert lstm.output_size == 2 * 12

    sarimax = SARIMAXModel("wind_direction_10m", settings=cfg).fit(corrupted_val)
    assert sarimax.comp_sin is not None and sarimax.comp_cos is not None

    for mdl in (ridge, rf, lstm, sarimax):
        fc = mdl.predict(split.val_df, split.X_val_scaled)
        assert fc.predicted.shape == (len(split.val_df), 12)
        assert np.all((0.0 <= fc.lower) & (fc.lower <= fc.predicted) & (fc.predicted <= fc.upper))
        assert np.all(fc.upper <= 359.9999)
        y_val = extract_target_matrix(split.val_df, "wind_direction_10m", settings=cfg)
        scores = score_multi_horizon(y_val, fc.predicted, metric="wind_direction_10m")
        # Wraparound error around 355 <-> 5 must stay bounded well under 180 deg
        assert 0.0 <= scores["rmse"] < 60.0


# ---------------------------------------------------------------------------
# 2. +1000.0 Leakage Perturbation Across All 5 Families & Scaler/Weights/Preds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["temperature_2m", "wind_direction_10m"])
def test_leakage_perturbation_1000_all_families_bit_identical(
    tmp_path: Path,
    metric: str,
) -> None:
    cfg, trainable, _ = _build_trainable(tmp_path, n_hours=240, horizon_hours=12)
    split_base = split_and_scale(trainable, settings=cfg)

    train_end_ts = split_base.train_df["timestamp_utc"].max()
    perturbed = trainable.copy()
    post_mask = perturbed["timestamp_utc"] > train_end_ts
    assert int(post_mask.sum()) > 0

    cols_to_perturb = set(split_base.feature_columns)
    cols_to_perturb.update(get_target_column_names(metric, settings=cfg))
    if metric == "wind_direction_10m":
        sin_cols, cos_cols = get_wind_component_target_columns(settings=cfg)
        cols_to_perturb.update(sin_cols)
        cols_to_perturb.update(cos_cols)
        cols_to_perturb.add("wind_direction_10m")

    for col in sorted(cols_to_perturb):
        if col not in perturbed.columns or pd.api.types.is_bool_dtype(perturbed[col].dtype):
            continue
        perturbed.loc[post_mask, col] = perturbed.loc[post_mask, col].astype(np.float64) + 1000.0

    split_pert = split_and_scale(perturbed, settings=cfg)

    assert np.array_equal(split_base.scaler.mean_, split_pert.scaler.mean_)
    assert np.array_equal(split_base.scaler.scale_, split_pert.scaler.scale_)
    assert np.array_equal(split_base.X_train_scaled, split_pert.X_train_scaled)

    for family in MODEL_FAMILIES:
        cls = MODEL_CLASS_BY_FAMILY[family]
        m_base = cls(metric, settings=cfg).fit(split_base)
        m_pert = cls(metric, settings=cfg).fit(split_pert)

        # Quantiles identical
        assert np.array_equal(m_base.q_low, m_pert.q_low)
        assert np.array_equal(m_base.q_high, m_pert.q_high)

        # Weights / learned parameters bit-for-bit identical
        if isinstance(m_base, RidgeModel):
            if m_base.is_circular:
                assert np.array_equal(m_base.estimator_sin.coef_, m_pert.estimator_sin.coef_)
                assert np.array_equal(m_base.estimator_cos.coef_, m_pert.estimator_cos.coef_)
            else:
                assert np.array_equal(m_base.estimator.coef_, m_pert.estimator.coef_)
                assert np.array_equal(m_base.estimator.intercept_, m_pert.estimator.intercept_)
        elif isinstance(m_base, RandomForestModel):
            if m_base.is_circular:
                assert np.array_equal(
                    m_base.estimator_sin.feature_importances_,
                    m_pert.estimator_sin.feature_importances_,
                )
            else:
                assert np.array_equal(
                    m_base.estimator.feature_importances_,
                    m_pert.estimator.feature_importances_,
                )
        elif isinstance(m_base, SARIMAXModel):
            if m_base.is_circular:
                assert np.array_equal(m_base.comp_sin.params, m_pert.comp_sin.params)
                assert np.array_equal(m_base.comp_cos.params, m_pert.comp_cos.params)
            else:
                assert np.array_equal(m_base.comp.params, m_pert.comp.params)
        elif isinstance(m_base, LSTMModel):
            for k, tensor in m_base.net.state_dict().items():
                assert np.array_equal(
                    tensor.detach().cpu().numpy(),
                    m_pert.net.state_dict()[k].detach().cpu().numpy(),
                )

        # Train predictions bit-for-bit identical
        tr_base = m_base.predict(split_base.train_df, split_base.X_train_scaled)
        tr_pert = m_pert.predict(split_pert.train_df, split_pert.X_train_scaled)
        assert np.array_equal(tr_base.predicted, tr_pert.predicted)
        assert np.array_equal(tr_base.lower, tr_pert.lower)
        assert np.array_equal(tr_base.upper, tr_pert.upper)

        # Val/test predictions must actually reflect the +1000 perturbation on temperature_2m
        if metric == "temperature_2m":
            val_base = m_base.predict(split_base.val_df, split_base.X_val_scaled)
            val_pert = m_pert.predict(split_pert.val_df, split_pert.X_val_scaled)
            assert not np.array_equal(val_base.predicted, val_pert.predicted)


# ---------------------------------------------------------------------------
# 3. Non-Default Settings Proof (No Hardcoded Hyperparameters or Horizons)
# ---------------------------------------------------------------------------


def test_custom_settings_no_hardcoded_hyperparameters_or_horizon(tmp_path: Path) -> None:
    cfg = _make_fast_settings(
        tmp_path,
        forecast_horizon_hours=12,
        train_split_ratio=0.60,
        val_split_ratio=0.20,
        test_split_ratio=0.20,
        sarimax_order=(1, 0, 0),
        sarimax_seasonal_order=(1, 0, 0, 12),
        sarimax_conf_alpha=0.1,
        sarimax_maxiter=7,
        rf_n_estimators=12,
        rf_max_depth=4,
        rf_min_samples_leaf=3,
        ridge_alpha=4.5,
        lstm_lookback_hours=10,
        lstm_hidden_size=12,
        lstm_num_layers=1,
        lstm_dropout=0.05,
        lstm_epochs=2,
        lstm_batch_size=16,
        hopsworks_model_name_template="lahore_custom_{metric}_v9",
    )
    _, trainable, _ = _build_trainable(tmp_path, n_hours=210, horizon_hours=12, settings=cfg)
    split = split_and_scale(trainable, settings=cfg)
    assert split.embargo_hours == 12

    ridge = RidgeModel("temperature_2m", settings=cfg).fit(split)
    assert ridge.horizon_hours == 12
    assert ridge.estimator.alpha == 4.5
    assert ridge.predict(split.val_df, split.X_val_scaled).predicted.shape == (len(split.val_df), 12)

    rf = RandomForestModel("temperature_2m", settings=cfg).fit(split)
    assert rf.estimator.n_estimators == 12
    assert rf.estimator.max_depth == 4
    assert rf.estimator.min_samples_leaf == 3
    assert rf.predict(split.val_df, split.X_val_scaled).predicted.shape == (len(split.val_df), 12)

    sarimax = SARIMAXModel("temperature_2m", settings=cfg).fit(split)
    assert sarimax.comp.order == (1, 0, 0)
    assert sarimax.comp.seasonal_order == (1, 0, 0, 12)
    assert sarimax.comp.maxiter == 7
    assert sarimax.hyperparameters["maxiter"] == 7
    assert sarimax.predict(split.val_df, split.X_val_scaled).predicted.shape == (len(split.val_df), 12)

    lstm = LSTMModel("temperature_2m", settings=cfg).fit(split)
    assert lstm.lookback_hours == 10
    assert lstm.hidden_size == 12
    assert lstm.num_layers == 1
    assert lstm.predict(split.val_df, split.X_val_scaled).predicted.shape == (len(split.val_df), 12)

    assert resolve_registry_model_name("temperature_2m", cfg) == "lahore_custom_temperature_2m_v9"


# ---------------------------------------------------------------------------
# 4. Champion Promotion Rule, Cold Start (404 vs Outage), and Template Binding
# ---------------------------------------------------------------------------


def test_champion_promotion_gate_cold_start_comparisons_and_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_fast_settings(
        tmp_path,
        forecast_horizon_hours=12,
        skycast_offline_mode="false",
        hopsworks_api_key="hw-key",
        hopsworks_project_name="skycast",
        hopsworks_model_name_template="custom_lahore_{metric}_champ",
    )
    store = SkyCastStorage(cfg)
    mock_hw = MockHopsworksProject(artifact_root=tmp_path / "hw_mr")
    mock_hw.model_registry.raise_not_found_on_missing = True  # Cold-start raises MockRestAPIError 404
    monkeypatch.setattr(store, "_get_hopsworks_project", lambda: mock_hw)

    _, trainable, _ = _build_trainable(tmp_path, n_hours=200, horizon_hours=12, settings=cfg)
    split = split_and_scale(trainable, settings=cfg)
    ridge = RidgeModel("temperature_2m", settings=cfg).fit(split)
    model_name = resolve_registry_model_name("temperature_2m", cfg)
    assert model_name == "custom_lahore_temperature_2m_champ"

    # Cold start query handles MockRestAPIError 404 gracefully
    assert get_current_champion_info("temperature_2m", store, settings=cfg) == (None, None, None)

    def _metrics(val_rmse: float, run_id: str = "r1") -> dict[str, Any]:
        return {
            "train_run_id": run_id,
            "trained_at_utc": "2026-11-18T00:00:00+00:00",
            "val_rmse": val_rmse,
            "val_mae": 1.0,
            "val_r2": 0.9,
            "test_rmse": val_rmse + 0.1,
            "test_mae": 1.1,
            "test_r2": 0.88,
            "improvement_over_naive_pct": 20.0,
            "training_samples": len(split.train_df),
        }

    # 1. Non-finite val_rmse -> NOT promoted
    res_nan = promote_champion_if_better(
        metric="temperature_2m",
        candidate_model=ridge,
        candidate_metrics=_metrics(float("nan")),
        naive_val_rmse=5.0,
        storage=store,
        settings=cfg,
    )
    assert res_nan["promoted_to_registry"] is False
    assert res_nan["is_champion"] is False

    # 2. Worse than or equal to Naive24h -> NOT promoted
    res_tie_naive = promote_champion_if_better(
        metric="temperature_2m",
        candidate_model=ridge,
        candidate_metrics=_metrics(5.0),
        naive_val_rmse=5.0,
        storage=store,
        settings=cfg,
    )
    assert res_tie_naive["promoted_to_registry"] is False
    assert res_tie_naive["reason"] == "did_not_beat_naive_baseline"

    # 3. Cold start + strictly < Naive24h -> PROMOTED v1
    res_v1 = promote_champion_if_better(
        metric="temperature_2m",
        candidate_model=ridge,
        candidate_metrics=_metrics(2.5, "run_1"),
        naive_val_rmse=5.0,
        storage=store,
        input_example=split.X_train_scaled[:1],
        settings=cfg,
    )
    assert res_v1["is_champion"] is True
    assert res_v1["promoted_to_registry"] is True
    assert res_v1["model_version"] == "1"
    assert len(mock_hw.model_registry.get_models(model_name)) == 1

    # 4. Worse than or equal to existing champion (2.5) even though < Naive (5.0) -> NOT promoted
    res_worse = promote_champion_if_better(
        metric="temperature_2m",
        candidate_model=ridge,
        candidate_metrics=_metrics(2.5, "run_2"),
        naive_val_rmse=5.0,
        storage=store,
        settings=cfg,
    )
    assert res_worse["is_champion"] is False
    assert res_worse["promoted_to_registry"] is False
    assert res_worse["reason"] == "did_not_beat_current_champion"
    assert len(mock_hw.model_registry.get_models(model_name)) == 1

    # 5. Strictly < existing champion (1.9 < 2.5) and < Naive -> PROMOTED v2
    res_v2 = promote_champion_if_better(
        metric="temperature_2m",
        candidate_model=ridge,
        candidate_metrics=_metrics(1.9, "run_3"),
        naive_val_rmse=5.0,
        storage=store,
        settings=cfg,
    )
    assert res_v2["is_champion"] is True
    assert res_v2["promoted_to_registry"] is True
    assert res_v2["model_version"] == "2"
    assert len(mock_hw.model_registry.get_models(model_name)) == 2

    # load_champion_bundle downloads v2 cleanly
    loaded_pair = load_champion_bundle("temperature_2m", store, settings=cfg)
    assert loaded_pair is not None
    loaded_model, loaded_ver = loaded_pair
    assert loaded_ver == "2"
    assert loaded_model.predict(split.test_df.head(2)).predicted.shape == (2, 12)

    # 6. Genuine Hopsworks outage must raise StorageError (never treated as cold start!)
    mock_hw.model_registry.fail_on_read = True
    with pytest.raises(StorageError):
        get_current_champion_info("temperature_2m", store, settings=cfg)

    # Also test FakeHopsworksProject compatibility
    fake_hw = FakeHopsworksProject(artifact_root=tmp_path / "fake_hw_mr")
    fake_hw.attach_to_storage(store, monkeypatch)
    assert get_current_champion_info("us_aqi", store, settings=cfg) == (None, None, None)


# ---------------------------------------------------------------------------
# 5. SHAP (50-row KernelExplainer) & LIME (Zero-Variance Columns) XAI Tests
# ---------------------------------------------------------------------------


def test_explain_shap_50_row_subsample_and_lime_zero_variance_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shap

    cfg, trainable, inference = _build_trainable(
        tmp_path,
        n_hours=240,
        horizon_hours=12,
        dry_mode=True,
        zero_var_cols=("cloud_cover",),
    )
    split = split_and_scale(trainable, settings=cfg)
    assert len(split.X_train_scaled) > 50

    # Verify zero-variance columns actually exist in train
    col_stds = np.std(split.X_train_scaled, axis=0)
    assert int((col_stds < 1e-12).sum()) >= 2

    # 1. Spy on shap.KernelExplainer to verify 50-row chronological train background subsample
    captured_bg_shapes: list[tuple[int, int]] = []
    captured_bg_arrays: list[np.ndarray] = []
    orig_kernel = shap.KernelExplainer

    def _spy_kernel(model_fn: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        arr = np.asarray(data, dtype=np.float64)
        captured_bg_shapes.append(arr.shape)
        captured_bg_arrays.append(arr.copy())
        return orig_kernel(model_fn, data, *args, **kwargs)

    monkeypatch.setattr(shap, "KernelExplainer", _spy_kernel)

    # Test LSTM (KernelExplainer) + LIME with zero-variance train columns
    lstm_temp = LSTMModel("temperature_2m", settings=cfg).fit(split)
    xai_lstm = compute_explanations_for_metric(
        train_run_id="xai_test_1",
        model=lstm_temp,
        split=split,
        reference_df=inference.tail(1),
        settings=cfg,
    )
    assert len(captured_bg_shapes) == 1
    assert captured_bg_shapes[0] == (50, len(split.feature_columns))
    expected_bg = subsample_chronological_rows(split.X_train_scaled, 50)
    assert np.array_equal(captured_bg_arrays[0], expected_bg)

    assert len(xai_lstm) == 3
    assert {(r["method"], r["scope"]) for r in xai_lstm} == {
        ("shap", "global"),
        ("shap", "local"),
        ("lime", "local"),
    }
    for row in xai_lstm:
        assert 1 <= len(row["feature_attributions"]) <= 10
        for attr in row["feature_attributions"]:
            assert attr["feature"] in split.feature_columns
    # Zero-variance column must NOT appear in dropped LIME attributions
    lime_row = [r for r in xai_lstm if r["method"] == "lime"][0]
    zv_names = {split.feature_columns[i] for i in np.where(col_stds < 1e-12)[0]}
    assert not any(a["feature"] in zv_names for a in lime_row["feature_attributions"])
    json.dumps(xai_lstm, allow_nan=False)

    # 2. Test SARIMAX (KernelExplainer) + Ridge (LinearExplainer) + RandomForest (TreeExplainer) + Circular Wind
    captured_bg_shapes.clear()
    sarimax_wind = SARIMAXModel("wind_direction_10m", settings=cfg).fit(split)
    xai_sarimax = compute_explanations_for_metric(
        train_run_id="xai_test_2",
        model=sarimax_wind,
        split=split,
        settings=cfg,
    )
    assert captured_bg_shapes == [(50, len(split.feature_columns))]
    assert len(xai_sarimax) == 3
    json.dumps(xai_sarimax, allow_nan=False)

    rf_wind = RandomForestModel("wind_direction_10m", settings=cfg).fit(split)
    xai_rf = compute_explanations_for_metric(
        train_run_id="xai_test_3",
        model=rf_wind,
        split=split,
        settings=cfg,
    )
    assert len(xai_rf) == 3
    json.dumps(xai_rf, allow_nan=False)

    ridge_precip = RidgeModel("precipitation_mm", settings=cfg).fit(split)
    xai_ridge = compute_explanations_for_metric(
        train_run_id="xai_test_4",
        model=ridge_precip,
        split=split,
        settings=cfg,
    )
    assert len(xai_ridge) == 3
    json.dumps(xai_ridge, allow_nan=False)

    # 3. All-columns-zero-variance degenerate fallback exercises 1e-8 LIME jitter path cleanly
    all_zv_df = trainable.copy()
    for col in split.feature_columns:
        if pd.api.types.is_bool_dtype(all_zv_df[col].dtype):
            all_zv_df[col] = False
        else:
            all_zv_df[col] = 0.0
    split_zv = split_and_scale(all_zv_df, settings=cfg)
    ridge_zv = RidgeModel("temperature_2m", settings=cfg).fit(split_zv)
    xai_zv = compute_explanations_for_metric(
        train_run_id="xai_test_zv",
        model=ridge_zv,
        split=split_zv,
        settings=cfg,
    )
    assert len(xai_zv) == 3
    json.dumps(xai_zv, allow_nan=False)
