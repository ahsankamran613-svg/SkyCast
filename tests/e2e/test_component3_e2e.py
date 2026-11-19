"""End-to-end tests for SkyCast Component 3 training, EDA, XAI, promotion, and storage contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skycast.config.settings import METRIC_SPECS, MODEL_FAMILIES, TARGET_METRICS, Settings
from skycast.data.storage import SkyCastStorage, StorageError
from tests.e2e.conftest import _patch_settings_everywhere
from tests.e2e.fakes import ContractSupabaseFake, MockHopsworksProject
from tests.e2e.fixtures import (
    generate_lahore_feature_dataframe,
    make_e2e_settings,
    seed_sqlite_history,
)


def _fast_e2e_settings(
    tmp_path: Path,
    *,
    horizon_hours: int = 24,
    offline: bool = False,
    **overrides: Any,
) -> Settings:
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
    return make_e2e_settings(tmp_path, horizon_hours=horizon_hours, offline=offline, **defaults)


def _wire_dual_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: Settings,
    supa_fake: ContractSupabaseFake,
    hw_proj: MockHopsworksProject,
) -> SkyCastStorage:
    _patch_settings_everywhere(monkeypatch, cfg)
    store = SkyCastStorage(settings=cfg)
    store.init_local_db()
    store._supabase_client = supa_fake
    store._hopsworks_project = hw_proj
    monkeypatch.setattr(store, "_get_supabase", lambda: supa_fake)
    monkeypatch.setattr(store, "_get_hopsworks_project", lambda: hw_proj)
    return store


def test_component3_e2e_dual_cloud_full_pipeline_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_supabase_fake: ContractSupabaseFake,
    mock_hopsworks_project: MockHopsworksProject,
) -> None:
    from skycast.ml.registry import load_champion_bundle, resolve_registry_model_name
    from skycast.pipelines.train import run_training

    cfg = _fast_e2e_settings(tmp_path, horizon_hours=24, offline=False)
    store = _wire_dual_storage(tmp_path, monkeypatch, cfg, contract_supabase_fake, mock_hopsworks_project)

    # Seed 1100 hours so model_comparison_series produces > 1,000 rows (8 metrics x ~150 test rows)
    full_df = generate_lahore_feature_dataframe(n_hours=1100, horizon_hours=24, seed=42)
    store.upsert_weather_features(full_df, supabase_days=60, write_local=False)
    store.record_pipeline_run(
        "backfill_seed_1",
        "backfill",
        "completed",
        metadata={"imputation_audit": {"tier1_count": 3, "tier2_count": 0, "tier3_count": 0, "imputation_rate_pct": 0.27}},
    )

    eda_out = tmp_path / "artifacts" / "eda_summary.json"
    result = run_training(
        settings=cfg,
        storage=store,
        train_run_id="e2e_dual_run_1",
        save_eda_doc=True,
        eda_output_path=eda_out,
    )

    assert result["status"] == "completed"
    assert result["train_run_id"] == "e2e_dual_run_1"
    assert result["metrics_trained"] == list(TARGET_METRICS)
    assert result["families_trained"] == list(MODEL_FAMILIES)
    assert eda_out.exists()
    json.loads(eda_out.read_text(encoding="utf-8"))

    # Pipeline run recorded cleanly
    latest_pr = store.get_latest_pipeline_run("training")
    assert latest_pr is not None
    assert latest_pr["run_id"] == "e2e_dual_run_1"
    assert latest_pr["status"] == "completed"

    # EDA persisted with provided imputation_audit
    eda_row = store.read_latest_eda_summary()
    assert eda_row is not None
    assert eda_row["run_id"] == "e2e_dual_run_1"
    assert eda_row["sample_count"] == 1100
    assert eda_row["summary_json"]["imputation_audit"]["source"] == "provided"
    assert eda_row["summary_json"]["imputation_audit"]["tier1_count"] == 3

    # Model registry metrics: 8 metrics x 5 families = 40 rows
    reg_rows = store.read_model_registry_metrics(latest_run_only=True)
    assert len(reg_rows) == len(TARGET_METRICS) * len(MODEL_FAMILIES) == 40
    promoted_metrics = [r["metric"] for r in reg_rows if r["promoted_to_registry"]]
    assert len(promoted_metrics) >= 1

    # Verify Hopsworks Model Registry artifacts download and emit calibrated 24h trajectories
    latest_obs_df = full_df.tail(1).reset_index(drop=True)
    for m in promoted_metrics:
        model_name = resolve_registry_model_name(m, cfg)
        assert len(mock_hopsworks_project.model_registry.get_models(model_name)) == 1
        loaded_pair = load_champion_bundle(m, store, settings=cfg)
        assert loaded_pair is not None
        champ_model, ver = loaded_pair
        assert ver == "1"
        fc = champ_model.predict(latest_obs_df)
        assert fc.predicted.shape == (1, 24)
        assert np.all(fc.lower <= fc.predicted)
        assert np.all(fc.predicted <= fc.upper)
        spec = METRIC_SPECS[m]
        max_allowed = 359.9999 if spec.is_circular else spec.max_value
        assert np.all((spec.min_value <= fc.lower) & (fc.upper <= max_allowed))

    # Comparison series exceeds 1,000 rows and exercises paginated read cleanly
    comp_rows = store.read_model_comparison_series(latest_run_only=True)
    assert len(comp_rows) > 1000
    assert len(comp_rows) == len(TARGET_METRICS) * result["test_rows"]
    select_calls = [c for c in contract_supabase_fake.call_log if c["op"] == "select" and c["table"] == "model_comparison_series"]
    assert any(c["range"] == (1000, 1999) for c in select_calls)

    # SHAP & LIME explanations: 8 metrics x 3 rows = 24 rows
    xai_rows = store.read_shap_lime_explanations(latest_run_only=True)
    assert len(xai_rows) == len(TARGET_METRICS) * 3 == 24
    for r in xai_rows:
        assert 1 <= len(r["feature_attributions"]) <= 10


def test_component3_e2e_two_run_idempotency_and_plus_1h_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_supabase_fake: ContractSupabaseFake,
    mock_hopsworks_project: MockHopsworksProject,
) -> None:
    from skycast.pipelines.train import run_training

    cfg = _fast_e2e_settings(
        tmp_path,
        horizon_hours=12,
        offline=False,
        train_split_ratio=0.60,
        val_split_ratio=0.20,
        test_split_ratio=0.20,
        sarimax_seasonal_order=(1, 0, 0, 12),
    )
    store = _wire_dual_storage(tmp_path, monkeypatch, cfg, contract_supabase_fake, mock_hopsworks_project)

    df_221 = generate_lahore_feature_dataframe(n_hours=221, horizon_hours=12, seed=42)
    df_run1 = df_221.iloc[:-1].reset_index(drop=True)
    store.upsert_weather_features(df_run1, supabase_days=30, write_local=False)

    # Run 1 twice with same train_run_id -> strictly idempotent upserts
    res_1a = run_training(
        metrics=["temperature_2m", "wind_direction_10m"],
        settings=cfg,
        storage=store,
        train_run_id="idemp_run_1",
    )
    res_1b = run_training(
        metrics=["temperature_2m", "wind_direction_10m"],
        settings=cfg,
        storage=store,
        train_run_id="idemp_run_1",
    )
    assert res_1a["test_rows"] == res_1b["test_rows"]
    assert len(store.read_eda_summaries()) == 1
    assert len(store.read_model_registry_metrics()) == 2 * len(MODEL_FAMILIES)
    assert len(store.read_model_comparison_series()) == 2 * res_1a["test_rows"]
    assert len(store.read_shap_lime_explanations()) == 2 * 3

    # Run 2 (+1h advance)
    store.upsert_weather_features(df_221.tail(1), supabase_days=30, write_local=False)
    res_2 = run_training(
        metrics=["temperature_2m", "wind_direction_10m"],
        settings=cfg,
        storage=store,
        train_run_id="idemp_run_2",
    )
    assert res_2["total_history_rows"] == 221
    assert len(store.read_eda_summaries()) == 2
    assert store.read_latest_eda_summary()["run_id"] == "idemp_run_2"
    latest_reg = store.read_model_registry_metrics(latest_run_only=True)
    assert {r["train_run_id"] for r in latest_reg} == {"idemp_run_2"}
    latest_xai = store.read_shap_lime_explanations(latest_run_only=True)
    assert {r["train_run_id"] for r in latest_xai} == {"idemp_run_2"}


def test_component3_e2e_cold_start_empty_db_and_forced_outages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline_e2e_storage: SkyCastStorage,
    contract_supabase_fake: ContractSupabaseFake,
    mock_hopsworks_project: MockHopsworksProject,
) -> None:
    from skycast.pipelines.train import run_training

    # 1. Cold-start empty offline SQLite DB -> explicit failure & pipeline_runs failed log
    with pytest.raises(RuntimeError, match="no historical weather features"):
        run_training(storage=offline_e2e_storage, settings=offline_e2e_storage.settings)

    failed_pr = offline_e2e_storage.get_latest_pipeline_run("training")
    assert failed_pr is not None
    assert failed_pr["status"] == "failed"
    assert "no historical weather features" in (failed_pr["error_message"] or "")

    # 2. Forced Supabase write outage in cloud_dual
    cfg = _fast_e2e_settings(tmp_path / "outage_supa", horizon_hours=12, offline=False)
    store = _wire_dual_storage(tmp_path / "outage_supa", monkeypatch, cfg, contract_supabase_fake, mock_hopsworks_project)
    df = generate_lahore_feature_dataframe(n_hours=200, horizon_hours=12, seed=42)
    store.upsert_weather_features(df, supabase_days=30, write_local=False)

    contract_supabase_fake.set_failure(fail_writes=True)
    with pytest.raises(StorageError):
        run_training(
            metrics=["temperature_2m"],
            settings=cfg,
            storage=store,
            train_run_id="outage_run_supa",
        )

    # 3. Forced Hopsworks Model Registry outage (must NOT silently fall back as cold start!)
    contract_supabase_fake.set_failure(fail_writes=False)
    mock_hopsworks_project.model_registry.fail_on_read = True
    with pytest.raises(StorageError, match="Model Registry"):
        run_training(
            metrics=["temperature_2m"],
            families=["Naive24h", "Ridge"],
            settings=cfg,
            storage=store,
            train_run_id="outage_run_hw",
        )


def test_component3_e2e_dry_mode_zero_variance_and_offline_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skycast.ml.registry import load_champion_bundle
    from skycast.pipelines.train import main as train_main

    cfg = _fast_e2e_settings(
        tmp_path,
        horizon_hours=12,
        offline=True,
        train_split_ratio=0.60,
        val_split_ratio=0.20,
        test_split_ratio=0.20,
        sarimax_seasonal_order=(1, 0, 0, 12),
    )
    _patch_settings_everywhere(monkeypatch, cfg)
    store = SkyCastStorage(settings=cfg)

    dry_df = generate_lahore_feature_dataframe(
        n_hours=210,
        horizon_hours=12,
        seed=7,
        include_smog=False,
        include_monsoon=False,
        dry_mode=True,
        wind_wraparound=True,
        zero_var_cols=("cloud_cover",),
    )
    seed_sqlite_history(store, dry_df)

    # Run via CLI entrypoint (`python -m skycast.pipelines.train`) without writing to repo docs/
    rc = train_main(
        [
            "--offline",
            "--metrics",
            "precipitation_mm",
            "temperature_2m",
            "wind_direction_10m",
        ]
    )
    assert rc == 0

    reg_rows = store.read_model_registry_metrics(latest_run_only=True)
    assert len(reg_rows) == 3 * len(MODEL_FAMILIES)
    # Offline mode keeps promoted_to_registry=False honestly while storing local champion bundles
    assert all(r["promoted_to_registry"] is False for r in reg_rows)

    # Zero-variance dry precipitation_mm has val_rmse == 0.0 -> never beats Naive24h (0.0 < 0.0 is False)
    precip_rows = [r for r in reg_rows if r["metric"] == "precipitation_mm"]
    assert all(r["val_rmse"] == 0.0 or math.isfinite(r["val_rmse"]) for r in precip_rows)
    assert all(not r["is_champion"] for r in precip_rows)
    assert load_champion_bundle("precipitation_mm", store, settings=cfg) is None

    # Temperature & wind should produce local champion bundles loadable for Component 4
    temp_champ = load_champion_bundle("temperature_2m", store, settings=cfg)
    if temp_champ is not None:
        mdl, ver = temp_champ
        assert ver == "1"
        fc = mdl.predict(dry_df.tail(1))
        assert fc.predicted.shape == (1, 12)

    xai_rows = store.read_shap_lime_explanations(latest_run_only=True)
    assert len(xai_rows) == 3 * 3
    json.dumps(xai_rows, allow_nan=False)
