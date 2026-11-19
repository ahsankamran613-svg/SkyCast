"""Component 3 M1 tests: SkyCastStorage schema parity, pagination, idempotency, and Hopsworks mirroring."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skycast.config.settings import MODEL_FAMILIES, TARGET_METRICS
from skycast.data.storage import SkyCastStorage, StorageError, StorageValidationError
from tests.contract_fakes import (
    ContractViolationError,
    FakeHopsworksProject,
    FakeSupabaseClient,
    make_test_settings,
)


def _sample_forecast_row(
    *,
    run_id: str = "fc_run_1",
    issued_at_utc: str = "2026-11-18T00:00:00+00:00",
    horizon_hour: int = 1,
    metric: str = "temperature_2m",
    model_family: str = "Ridge",
    predicted_value: float = 24.5,
    lower_bound: float = 22.0,
    upper_bound: float = 27.0,
    **extra: Any,
) -> dict[str, Any]:
    offset_hours = 1 if isinstance(horizon_hour, bool) else int(horizon_hour)
    target_ts = (
        pd.to_datetime(issued_at_utc, utc=True) + pd.Timedelta(hours=offset_hours)
    ).isoformat()
    row: dict[str, Any] = {
        "run_id": run_id,
        "issued_at_utc": issued_at_utc,
        "target_timestamp_utc": target_ts,
        "horizon_hour": horizon_hour,
        "metric": metric,
        "model_family": model_family,
        "model_version": "v1",
        "predicted_value": predicted_value,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "naive_baseline_value": 23.0,
        "actual_value": None,
    }
    row.update(extra)
    return row


def _sample_registry_row(
    *,
    train_run_id: str = "train_1",
    trained_at_utc: str = "2026-11-18T00:00:00+00:00",
    metric: str = "temperature_2m",
    model_family: str = "RandomForest",
    val_rmse: float = 1.25,
    is_champion: bool = True,
    promoted_to_registry: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "train_run_id": train_run_id,
        "trained_at_utc": trained_at_utc,
        "metric": metric,
        "model_family": model_family,
        "model_version": "3",
        "val_rmse": val_rmse,
        "val_mae": 0.95,
        "val_r2": 0.88,
        "test_rmse": 1.35,
        "test_mae": 1.05,
        "test_r2": 0.85,
        "improvement_over_naive_pct": 22.5,
        "is_champion": is_champion,
        "promoted_to_registry": promoted_to_registry,
        "hyperparameters": {"n_estimators": 200, "max_depth": 12},
        "training_samples": 1400,
    }
    row.update(extra)
    return row


def _sample_comparison_row(
    *,
    train_run_id: str = "train_1",
    metric: str = "temperature_2m",
    timestamp_utc: str = "2026-11-18T01:00:00+00:00",
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "train_run_id": train_run_id,
        "metric": metric,
        "timestamp_utc": timestamp_utc,
        "actual": 22.0,
        "sarimax": 21.8,
        "random_forest": 22.1,
        "ridge": 21.9,
        "lstm": 22.2,
        "naive_24h": 20.5,
    }
    row.update(extra)
    return row


def _sample_explanation_row(
    *,
    train_run_id: str = "train_1",
    computed_at_utc: str = "2026-11-18T00:00:00+00:00",
    metric: str = "us_aqi",
    model_family: str = "RandomForest",
    method: str = "shap",
    scope: str = "global",
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "train_run_id": train_run_id,
        "computed_at_utc": computed_at_utc,
        "metric": metric,
        "model_family": model_family,
        "method": method,
        "scope": scope,
        "base_value": 120.5,
        "predicted_value": 145.2,
        "reference_timestamp_utc": "2026-11-17T23:00:00+00:00",
        "feature_attributions": [
            {"feature": "us_aqi_lag_1h", "importance": 18.4, "value": 140.0},
            {"feature": "pm2_5", "importance": 9.2, "value": 52.0},
        ],
    }
    row.update(extra)
    return row


def _sample_eda_summary(
    *,
    run_id: str = "eda_run_1",
    computed_at_utc: str = "2026-11-18T00:00:00+00:00",
    sample_count: int = 2160,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "computed_at_utc": computed_at_utc,
        "sample_count": sample_count,
        "date_range_start": "2026-08-20T00:00:00+00:00",
        "date_range_end": "2026-11-18T00:00:00+00:00",
        "summary_json": {
            "distributions": {"temperature_2m": {"mean": 26.4, "std": 4.2}},
            "diurnal_heatmap": {"hours": list(range(24))},
            "correlation_matrix": {"temperature_2m": {"us_aqi": -0.31}},
        },
    }
    row.update(extra)
    return row


def test_component3_tables_sqlite_roundtrip_and_sorting(tmp_path: Path) -> None:
    cfg = make_test_settings(tmp_path)
    store = SkyCastStorage(cfg)

    # 1. eda_summaries (verifying np.ndarray, plain datetime, and NaN sanitization)
    eda_res = store.upsert_eda_summary(
        _sample_eda_summary(
            summary_json={
                "arr": np.array([1.23456789, float("nan")]),
                "plain_dt": datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc),
                "bad_inf": float("inf"),
            }
        )
    )
    assert eda_res == {"storage_mode": "local_offline", "sqlite_rows": 1, "supabase_rows": 0}
    latest_eda = store.read_latest_eda_summary()
    assert latest_eda is not None
    assert latest_eda["sample_count"] == 2160
    assert latest_eda["summary_json"]["arr"] == [1.234568, None]
    assert latest_eda["summary_json"]["bad_inf"] is None
    assert latest_eda["summary_json"]["plain_dt"].startswith("2026-11-18T06:00:00")
    json.dumps(latest_eda, allow_nan=False)
    assert len(store.read_eda_summaries()) == 1

    # 2. model_registry_metrics
    reg_res = store.upsert_model_registry_metrics(
        [
            _sample_registry_row(model_family="Naive24h", val_rmse=2.1, is_champion=False, promoted_to_registry=False),
            _sample_registry_row(model_family="RandomForest", val_rmse=1.2, is_champion=True, promoted_to_registry=True),
        ]
    )
    assert reg_res == {"storage_mode": "local_offline", "sqlite_rows": 2, "supabase_rows": 0}
    champs = store.read_model_registry_metrics(metric="temperature_2m", is_champion=True)
    assert len(champs) == 1
    assert champs[0]["model_family"] == "RandomForest"
    assert champs[0]["is_champion"] is True
    assert champs[0]["promoted_to_registry"] is True
    assert champs[0]["hyperparameters"] == {"n_estimators": 200, "max_depth": 12}

    # 3. model_comparison_series (inserted out-of-order -> must return timestamp_utc ASC; NaN -> None)
    comp_res = store.upsert_model_comparison_series(
        [
            _sample_comparison_row(timestamp_utc="2026-11-18T03:00:00+00:00", lstm=float("nan")),
            _sample_comparison_row(timestamp_utc="2026-11-18T01:00:00+00:00", lstm=21.5),
            _sample_comparison_row(timestamp_utc="2026-11-18T02:00:00+00:00", lstm=None),
        ]
    )
    assert comp_res == {"storage_mode": "local_offline", "sqlite_rows": 3, "supabase_rows": 0}
    comp_rows = store.read_model_comparison_series(metric="temperature_2m")
    assert [r["timestamp_utc"] for r in comp_rows] == [
        "2026-11-18T01:00:00+00:00",
        "2026-11-18T02:00:00+00:00",
        "2026-11-18T03:00:00+00:00",
    ]
    assert comp_rows[2]["lstm"] is None
    json.dumps(comp_rows, allow_nan=False)

    # 4. shap_lime_explanations
    xai_res = store.upsert_shap_lime_explanations(
        [
            _sample_explanation_row(method="shap", scope="global"),
            _sample_explanation_row(method="lime", scope="local", base_value=float("nan")),
        ]
    )
    assert xai_res == {"storage_mode": "local_offline", "sqlite_rows": 2, "supabase_rows": 0}
    xai_rows = store.read_shap_lime_explanations(metric="us_aqi", method="lime", scope="local")
    assert len(xai_rows) == 1
    assert xai_rows[0]["base_value"] is None
    assert len(xai_rows[0]["feature_attributions"]) == 2

    # 5. forecast_ledger (inserted h=24 then h=1 -> ordered horizon_hour ASC)
    fc_res = store.upsert_forecast_ledger(
        [
            _sample_forecast_row(horizon_hour=24, predicted_value=26.0, lower_bound=23.0, upper_bound=29.0),
            _sample_forecast_row(horizon_hour=1, predicted_value=24.0, lower_bound=22.0, upper_bound=26.0),
        ]
    )
    assert fc_res == {
        "storage_mode": "local_offline",
        "sqlite_rows": 2,
        "supabase_rows": 0,
        "hopsworks_rows": 0,
    }
    fc_rows = store.read_forecast_ledger(metric="temperature_2m", model_family="Ridge")
    assert [r["horizon_hour"] for r in fc_rows] == [1, 24]
    assert all(r["lower_bound"] <= r["predicted_value"] <= r["upper_bound"] for r in fc_rows)


@pytest.mark.parametrize("mode", ["local_offline", "cloud_supabase"])
def test_python_check_and_domain_parity_rejects_invalid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    if mode == "local_offline":
        cfg = make_test_settings(tmp_path / mode)
        store = SkyCastStorage(cfg)
        fake_supa = None
    else:
        cfg = make_test_settings(
            tmp_path / mode,
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )
        store = SkyCastStorage(cfg)
        fake_supa = FakeSupabaseClient()
        monkeypatch.setattr(store, "_get_supabase", lambda: fake_supa)

    # forecast_ledger CHECK & domain bounds
    invalid_forecast_cases = [
        _sample_forecast_row(horizon_hour=0),
        _sample_forecast_row(horizon_hour=25),
        _sample_forecast_row(horizon_hour=True),
        _sample_forecast_row(lower_bound=25.0, predicted_value=24.0, upper_bound=26.0),
        _sample_forecast_row(lower_bound=20.0, predicted_value=27.0, upper_bound=26.0),
        # Post-6-decimal-rounding inversion: 359.9999999 rounds to 360.0 > 359.9999
        _sample_forecast_row(
            metric="wind_direction_10m",
            lower_bound=350.0,
            predicted_value=359.9999999,
            upper_bound=359.9999,
        ),
        _sample_forecast_row(predicted_value=float("nan")),
        _sample_forecast_row(predicted_value=float("inf")),
        _sample_forecast_row(metric="bad_metric"),
        _sample_forecast_row(model_family="XGBoost"),
        _sample_forecast_row(run_id=" "),
    ]
    for bad in invalid_forecast_cases:
        with pytest.raises(StorageValidationError):
            store.upsert_forecast_ledger([bad])

    # shap_lime_explanations CHECKs
    for bad_xai in [
        _sample_explanation_row(method="gradcam"),
        _sample_explanation_row(scope="batch"),
        _sample_explanation_row(metric="bad_metric"),
        _sample_explanation_row(model_family="CatBoost"),
        _sample_explanation_row(feature_attributions={"not": "a list"}),
    ]:
        with pytest.raises(StorageValidationError):
            store.upsert_shap_lime_explanations([bad_xai])

    # model_registry_metrics CHECKs
    for bad_reg in [
        _sample_registry_row(metric="bad_metric"),
        _sample_registry_row(model_family="LightGBM"),
        _sample_registry_row(val_rmse=-0.1),
        _sample_registry_row(val_rmse=float("nan")),
        _sample_registry_row(test_r2=float("inf")),
        _sample_registry_row(hyperparameters=["not-a-dict"]),
    ]:
        with pytest.raises(StorageValidationError):
            store.upsert_model_registry_metrics([bad_reg])

    # model_comparison_series CHECKs
    for bad_comp in [
        _sample_comparison_row(metric="bad_metric"),
        _sample_comparison_row(train_run_id=""),
        _sample_comparison_row(actual="not_a_float"),
    ]:
        with pytest.raises(StorageValidationError):
            store.upsert_model_comparison_series([bad_comp])

    # eda_summaries CHECKs
    for bad_eda in [
        _sample_eda_summary(run_id=""),
        _sample_eda_summary(sample_count=-1),
        _sample_eda_summary(summary_json=["not", "dict"]),
    ]:
        with pytest.raises(StorageValidationError):
            store.upsert_eda_summary(bad_eda)

    # pipeline_runs CHECKs
    with pytest.raises(StorageValidationError):
        store.record_pipeline_run("r1", "unknown_type", "running")
    with pytest.raises(StorageValidationError):
        store.record_pipeline_run("r1", "training", "pending")
    with pytest.raises(StorageValidationError):
        store.record_pipeline_run("", "training", "running")

    # Intra-batch duplicate ON CONFLICT key rejection
    with pytest.raises(StorageValidationError):
        store.upsert_forecast_ledger(
            [_sample_forecast_row(horizon_hour=1), _sample_forecast_row(horizon_hour=1)]
        )
    with pytest.raises(StorageValidationError):
        store.upsert_model_registry_metrics(
            [_sample_registry_row(model_family="Ridge"), _sample_registry_row(model_family="Ridge")]
        )
    with pytest.raises(StorageValidationError):
        store.upsert_model_comparison_series(
            [
                _sample_comparison_row(timestamp_utc="2026-11-18T00:00:00Z"),
                _sample_comparison_row(timestamp_utc="2026-11-18T00:00:00+00:00"),
            ]
        )
    with pytest.raises(StorageValidationError):
        store.upsert_shap_lime_explanations(
            [_sample_explanation_row(), _sample_explanation_row()]
        )

    # Verify nothing invalid touched DB / FakeSupabase
    if fake_supa is not None:
        assert fake_supa.call_log == []
    else:
        assert not store.local_db_path.exists()
        # Also verify fresh SQLite schema itself enforces CHECK constraints
        db_path = store.init_local_db()
        conn = sqlite3.connect(db_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO forecast_ledger (run_id, issued_at_utc, target_timestamp_utc, "
                    "horizon_hour, metric, model_family, predicted_value, lower_bound, upper_bound, created_at) "
                    "VALUES ('r', '2026-11-18T00:00:00+00:00', '2026-11-18T01:00:00+00:00', "
                    "25, 'us_aqi', 'Ridge', 100.0, 90.0, 110.0, '2026-11-18T00:00:00+00:00')"
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO shap_lime_explanations (train_run_id, computed_at_utc, metric, "
                    "model_family, method, scope, feature_attributions, created_at) "
                    "VALUES ('r', '2026-11-18T00:00:00+00:00', 'us_aqi', 'Ridge', 'gradcam', 'local', '[]', '2026-11-18T00:00:00+00:00')"
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO pipeline_runs (run_id, pipeline_type, status, started_at_utc, created_at) "
                    "VALUES ('r', 'bad', 'running', '2026-11-18T00:00:00+00:00', '2026-11-18T00:00:00+00:00')"
                )
        finally:
            conn.close()


def test_storage_modes_offline_supabase_hopsworks_dual_and_unconfigured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. unconfigured
    unconf_store = SkyCastStorage(make_test_settings(tmp_path / "unconf", skycast_offline_mode="false"))
    assert unconf_store.storage_mode == "unconfigured"
    for fn in (
        lambda: unconf_store.upsert_eda_summary(_sample_eda_summary()),
        lambda: unconf_store.read_eda_summaries(),
        lambda: unconf_store.read_latest_eda_summary(),
        lambda: unconf_store.upsert_model_registry_metrics([]),
        lambda: unconf_store.read_model_registry_metrics(),
        lambda: unconf_store.upsert_model_comparison_series([]),
        lambda: unconf_store.read_model_comparison_series(),
        lambda: unconf_store.upsert_shap_lime_explanations([]),
        lambda: unconf_store.read_shap_lime_explanations(),
        lambda: unconf_store.upsert_forecast_ledger([]),
        lambda: unconf_store.read_forecast_ledger(),
        lambda: unconf_store.record_pipeline_run("r1", "training", "running"),
        lambda: unconf_store.get_hopsworks_model_registry(),
    ):
        with pytest.raises(StorageError, match="unconfigured"):
            fn()

    # 2. local_offline -> Hopsworks MR unavailable
    off_store = SkyCastStorage(make_test_settings(tmp_path / "off"))
    with pytest.raises(StorageError):
        off_store.get_hopsworks_model_registry()

    # 3. cloud_supabase -> writes/reads Supabase only, zero Hopsworks calls
    supa_cfg = make_test_settings(
        tmp_path / "supa",
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
    )
    supa_store = SkyCastStorage(supa_cfg)
    assert supa_store.storage_mode == "cloud_supabase"
    fake_supa = FakeSupabaseClient()
    monkeypatch.setattr(supa_store, "_get_supabase", lambda: fake_supa)
    monkeypatch.setattr(
        supa_store,
        "_get_hopsworks_project",
        lambda: pytest.fail("Hopsworks must not be called in cloud_supabase mode"),
    )
    res_supa = supa_store.upsert_forecast_ledger([_sample_forecast_row()])
    assert res_supa == {
        "storage_mode": "cloud_supabase",
        "sqlite_rows": 0,
        "supabase_rows": 1,
        "hopsworks_rows": 0,
    }
    assert len(supa_store.read_forecast_ledger()) == 1
    with pytest.raises(StorageError):
        supa_store.get_hopsworks_model_registry()

    # 4. cloud_hopsworks -> forecast_ledger goes to Hopsworks FG; 4 non-FG tables go to SQLite
    hw_cfg = make_test_settings(
        tmp_path / "hw",
        skycast_offline_mode="false",
        hopsworks_api_key="hw-key",
        hopsworks_project_name="skycast",
    )
    hw_store = SkyCastStorage(hw_cfg)
    assert hw_store.storage_mode == "cloud_hopsworks"
    fake_hw = FakeHopsworksProject(artifact_root=tmp_path / "hw_mr")
    fake_hw.attach_to_storage(hw_store, monkeypatch)

    hw_fc_res = hw_store.upsert_forecast_ledger([_sample_forecast_row(horizon_hour=3)])
    assert hw_fc_res == {
        "storage_mode": "cloud_hopsworks",
        "sqlite_rows": 0,
        "supabase_rows": 0,
        "hopsworks_rows": 1,
    }
    fg = fake_hw.feature_store.get_feature_group(
        hw_cfg.hopsworks_forecast_ledger_fg, hw_cfg.hopsworks_feature_group_version
    )
    assert fg.primary_key == ["run_id", "metric", "model_family", "horizon_hour"]
    assert fg.event_time == "issued_at_utc"
    assert fg.online_enabled is True
    assert fg.insert_calls[0]["write_options"] == {"wait_for_job": False}
    hw_read = hw_store.read_forecast_ledger(metric="temperature_2m")
    assert len(hw_read) == 1
    assert hw_read[0]["horizon_hour"] == 3

    assert hw_store.upsert_eda_summary(_sample_eda_summary())["sqlite_rows"] == 1
    assert hw_store.read_latest_eda_summary() is not None
    assert hw_store.upsert_model_registry_metrics([_sample_registry_row()])["sqlite_rows"] == 1
    assert len(hw_store.read_model_registry_metrics()) == 1
    assert hw_store.upsert_model_comparison_series([_sample_comparison_row()])["sqlite_rows"] == 1
    assert len(hw_store.read_model_comparison_series()) == 1
    assert hw_store.upsert_shap_lime_explanations([_sample_explanation_row()])["sqlite_rows"] == 1
    assert len(hw_store.read_shap_lime_explanations()) == 1
    assert hw_store.get_hopsworks_model_registry() is fake_hw.model_registry

    # 5. cloud_dual -> forecast_ledger writes both Supabase & Hopsworks FG
    dual_cfg = make_test_settings(
        tmp_path / "dual",
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
        hopsworks_api_key="hw-key",
        hopsworks_project_name="skycast",
    )
    dual_store = SkyCastStorage(dual_cfg)
    assert dual_store.storage_mode == "cloud_dual"
    dual_supa = FakeSupabaseClient()
    dual_hw = FakeHopsworksProject(artifact_root=tmp_path / "dual_hw_mr")
    dual_hw.attach_to_storage(dual_store, monkeypatch, fake_supabase=dual_supa)

    dual_res = dual_store.upsert_forecast_ledger(
        [
            _sample_forecast_row(horizon_hour=1, predicted_value=24.0, lower_bound=22.0, upper_bound=26.0),
            _sample_forecast_row(horizon_hour=2, predicted_value=25.0, lower_bound=23.0, upper_bound=27.0),
        ]
    )
    assert dual_res == {
        "storage_mode": "cloud_dual",
        "sqlite_rows": 0,
        "supabase_rows": 2,
        "hopsworks_rows": 2,
    }
    assert len(dual_supa.get_rows("forecast_ledger")) == 2
    assert len(dual_store.read_forecast_ledger()) == 2
    assert len(dual_store.read_forecast_ledger(prefer_hopsworks=True)) == 2
    assert dual_store.get_hopsworks_model_registry() is dual_hw.model_registry


def test_supabase_pagination_over_1000_rows_component3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_test_settings(
        tmp_path,
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
    )
    store = SkyCastStorage(cfg)
    fake_supa = FakeSupabaseClient()
    monkeypatch.setattr(store, "_get_supabase", lambda: fake_supa)

    base_ts = pd.Timestamp("2026-08-01T00:00:00Z")
    # 2,350 comparison series rows
    comp_rows = [
        _sample_comparison_row(
            train_run_id="run_big",
            metric="temperature_2m",
            timestamp_utc=(base_ts + pd.Timedelta(hours=i)).isoformat(),
            actual=float(i),
        )
        for i in range(2350)
    ]
    fake_supa.seed_rows("model_comparison_series", comp_rows)
    got_comp = store.read_model_comparison_series(train_run_id="run_big", metric="temperature_2m")
    assert len(got_comp) == 2350
    assert got_comp[0]["actual"] == 0.0
    assert got_comp[-1]["actual"] == 2349.0

    # 2,350 forecast_ledger rows (across 100 run_ids x 24h subset)
    fc_rows = [
        _sample_forecast_row(
            run_id=f"fc_{i // 24:03d}",
            issued_at_utc=(base_ts + pd.Timedelta(hours=i // 24)).isoformat(),
            horizon_hour=(i % 24) + 1,
            predicted_value=20.0 + (i % 10),
            lower_bound=18.0,
            upper_bound=35.0,
        )
        for i in range(2350)
    ]
    fake_supa.seed_rows("forecast_ledger", fc_rows)
    got_fc = store.read_forecast_ledger()
    assert len(got_fc) == 2350

    # 2,050 model_registry_metrics rows (across 52 runs x 8 metrics x 5 families = 2080)
    reg_rows = [
        _sample_registry_row(
            train_run_id=f"tr_{r:03d}",
            trained_at_utc=(base_ts + pd.Timedelta(hours=r)).isoformat(),
            metric=m,
            model_family=f,
        )
        for r in range(52)
        for m in TARGET_METRICS
        for f in MODEL_FAMILIES
    ]
    fake_supa.seed_rows("model_registry_metrics", reg_rows)
    got_reg = store.read_model_registry_metrics()
    assert len(got_reg) == len(reg_rows) == 2080

    # 1,250 shap_lime_explanations rows (across 125 runs x 8 metrics x 2 methods -> subset 1250)
    xai_rows = []
    for r in range(80):
        for m in TARGET_METRICS:
            for method, scope in (("shap", "global"), ("lime", "local")):
                xai_rows.append(
                    _sample_explanation_row(
                        train_run_id=f"tr_{r:03d}",
                        computed_at_utc=(base_ts + pd.Timedelta(hours=r)).isoformat(),
                        metric=m,
                        method=method,
                        scope=scope,
                    )
                )
    fake_supa.seed_rows("shap_lime_explanations", xai_rows)
    got_xai = store.read_shap_lime_explanations()
    assert len(got_xai) == len(xai_rows) == 1280

    select_calls = [c for c in fake_supa.call_log if c["op"] == "select"]
    assert any(c["range"] == (2000, 2999) for c in select_calls)


@pytest.mark.parametrize("mode", ["local_offline", "cloud_supabase"])
def test_two_run_idempotent_upserts_and_conflict_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    if mode == "local_offline":
        cfg = make_test_settings(tmp_path / mode)
        store = SkyCastStorage(cfg)
    else:
        cfg = make_test_settings(
            tmp_path / mode,
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )
        store = SkyCastStorage(cfg)
        fake_supa = FakeSupabaseClient()
        monkeypatch.setattr(store, "_get_supabase", lambda: fake_supa)

    t1 = "2026-11-18T00:00:00+00:00"
    t2 = "2026-11-18T01:00:00+00:00"

    # Run 1 initial + Run 1 overwrite (idempotent update in-place)
    store.upsert_eda_summary(_sample_eda_summary(run_id="r1", computed_at_utc=t1, sample_count=100))
    store.upsert_eda_summary(_sample_eda_summary(run_id="r1", computed_at_utc=t1, sample_count=150))
    assert len(store.read_eda_summaries()) == 1
    assert store.read_latest_eda_summary()["sample_count"] == 150

    store.upsert_model_registry_metrics([_sample_registry_row(train_run_id="r1", trained_at_utc=t1, val_rmse=1.9)])
    store.upsert_model_registry_metrics([_sample_registry_row(train_run_id="r1", trained_at_utc=t1, val_rmse=1.1)])
    reg_r1 = store.read_model_registry_metrics()
    assert len(reg_r1) == 1
    assert reg_r1[0]["val_rmse"] == 1.1

    store.upsert_model_comparison_series([_sample_comparison_row(train_run_id="r1", actual=20.0)])
    store.upsert_model_comparison_series([_sample_comparison_row(train_run_id="r1", actual=21.0)])
    comp_r1 = store.read_model_comparison_series()
    assert len(comp_r1) == 1
    assert comp_r1[0]["actual"] == 21.0

    store.upsert_shap_lime_explanations([_sample_explanation_row(train_run_id="r1", computed_at_utc=t1, predicted_value=100.0)])
    store.upsert_shap_lime_explanations([_sample_explanation_row(train_run_id="r1", computed_at_utc=t1, predicted_value=110.0)])
    xai_r1 = store.read_shap_lime_explanations()
    assert len(xai_r1) == 1
    assert xai_r1[0]["predicted_value"] == 110.0

    store.upsert_forecast_ledger([_sample_forecast_row(run_id="r1", issued_at_utc=t1, predicted_value=24.0)])
    store.upsert_forecast_ledger([_sample_forecast_row(run_id="r1", issued_at_utc=t1, predicted_value=25.0)])
    fc_r1 = store.read_forecast_ledger()
    assert len(fc_r1) == 1
    assert fc_r1[0]["predicted_value"] == 25.0

    # Run 2 (+1h advance)
    store.upsert_eda_summary(_sample_eda_summary(run_id="r2", computed_at_utc=t2, sample_count=200))
    assert store.read_latest_eda_summary()["run_id"] == "r2"

    store.upsert_model_registry_metrics([_sample_registry_row(train_run_id="r2", trained_at_utc=t2, val_rmse=0.9)])
    assert [r["train_run_id"] for r in store.read_model_registry_metrics(latest_run_only=True)] == ["r2"]

    store.upsert_model_comparison_series([_sample_comparison_row(train_run_id="r2", actual=22.0)])
    assert [r["train_run_id"] for r in store.read_model_comparison_series(latest_run_only=True)] == ["r2"]

    store.upsert_shap_lime_explanations([_sample_explanation_row(train_run_id="r2", computed_at_utc=t2, predicted_value=120.0)])
    assert [r["train_run_id"] for r in store.read_shap_lime_explanations(latest_run_only=True)] == ["r2"]

    store.upsert_forecast_ledger([_sample_forecast_row(run_id="r2", issued_at_utc=t2, predicted_value=26.0)])
    assert [r["run_id"] for r in store.read_forecast_ledger(latest_run_only=True)] == ["r2"]


def test_cold_start_empty_db_and_forced_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cold start SQLite (DB file does not exist yet)
    off_store = SkyCastStorage(make_test_settings(tmp_path / "cold_off"))
    assert off_store.read_latest_eda_summary() is None
    assert off_store.read_eda_summaries() == []
    assert off_store.read_model_registry_metrics(latest_run_only=True) == []
    assert off_store.read_model_comparison_series(latest_run_only=True) == []
    assert off_store.read_shap_lime_explanations(latest_run_only=True) == []
    assert off_store.read_forecast_ledger(latest_run_only=True) == []
    assert off_store.upsert_forecast_ledger([])["sqlite_rows"] == 0
    assert not off_store.local_db_path.exists()

    # Forced Supabase failures
    supa_cfg = make_test_settings(
        tmp_path / "fail_supa",
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
    )
    supa_store = SkyCastStorage(supa_cfg)
    failing_supa = FakeSupabaseClient(
        fail_rules={
            ("*", "select"): RuntimeError("PostgREST 503"),
            ("*", "upsert"): RuntimeError("PostgREST 500"),
        }
    )
    monkeypatch.setattr(supa_store, "_get_supabase", lambda: failing_supa)

    with pytest.raises(StorageError, match="PostgREST 500"):
        supa_store.upsert_eda_summary(_sample_eda_summary())
    with pytest.raises(StorageError, match="PostgREST 503"):
        supa_store.read_model_registry_metrics()
    with pytest.raises(StorageError, match="PostgREST 500"):
        supa_store.upsert_forecast_ledger([_sample_forecast_row()])

    # Forced Hopsworks failures
    hw_cfg = make_test_settings(
        tmp_path / "fail_hw",
        skycast_offline_mode="false",
        hopsworks_api_key="hw-key",
        hopsworks_project_name="skycast",
    )
    hw_store = SkyCastStorage(hw_cfg)
    failing_hw = FakeHopsworksProject(
        artifact_root=tmp_path / "fail_hw_mr",
        fail_rules={("fg_insert", "*"): RuntimeError("Hopsworks Kafka timeout")},
    )
    failing_hw.attach_to_storage(hw_store, monkeypatch)

    with pytest.raises(StorageError, match="Hopsworks Kafka timeout"):
        hw_store.upsert_forecast_ledger([_sample_forecast_row()])

    def _broken_mr() -> Any:
        raise RuntimeError("Hopsworks MR unreachable")

    monkeypatch.setattr(failing_hw, "get_model_registry", _broken_mr)
    with pytest.raises(StorageError, match="Hopsworks MR unreachable"):
        hw_store.get_hopsworks_model_registry()


def test_non_default_settings_hopsworks_fg_and_model_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_dir = tmp_path / "custom_local_dir"
    cfg = make_test_settings(
        tmp_path,
        skycast_offline_mode="false",
        skycast_local_data_dir=custom_dir,
        hopsworks_api_key="hw-key",
        hopsworks_project_name="custom_proj",
        hopsworks_forecast_ledger_fg="custom_ledger_fg",
        hopsworks_feature_group_version=3,
        hopsworks_model_name_template="custom_{metric}_v2",
    )
    store = SkyCastStorage(cfg)
    fake_hw = FakeHopsworksProject(artifact_root=tmp_path / "custom_mr")
    fake_hw.attach_to_storage(store, monkeypatch)

    store.upsert_forecast_ledger([_sample_forecast_row()])
    assert ("custom_ledger_fg", 3) in fake_hw.feature_store.feature_groups
    assert ("lahore_forecast_ledger", 1) not in fake_hw.feature_store.feature_groups

    # Non-FG table writes to custom_dir/skycast.db
    store.upsert_eda_summary(_sample_eda_summary())
    assert (custom_dir / "skycast.db").exists()

    # FakeHopsworksModelRegistry save & download round-trip with custom model name template
    mr = store.get_hopsworks_model_registry()
    model_name = cfg.hopsworks_model_name_template.format(metric="us_aqi")
    assert mr.get_best_model(model_name, metric="val_rmse", direction="min") is None
    bundle_dir = tmp_path / "bundle_src"
    bundle_dir.mkdir()
    (bundle_dir / "model.json").write_text('{"family": "Ridge"}', encoding="utf-8")
    created = mr.python.create_model(name=model_name, metrics={"val_rmse": 12.4})
    saved = created.save(str(bundle_dir))
    assert saved.version == 1
    best = mr.get_best_model(model_name, metric="val_rmse", direction="min")
    assert best is not None and best.training_metrics["val_rmse"] == 12.4
    dl_dir = Path(best.download(str(tmp_path / "bundle_dl")))
    assert (dl_dir / "model.json").exists()


def test_windows_sqlite_handle_closure_after_component3_writes(tmp_path: Path) -> None:
    cfg = make_test_settings(tmp_path / "win_lock_c3")
    store = SkyCastStorage(cfg)
    store.record_pipeline_run("r1", "training", "completed")
    store.upsert_eda_summary(_sample_eda_summary())
    store.upsert_model_registry_metrics([_sample_registry_row()])
    store.upsert_model_comparison_series([_sample_comparison_row()])
    store.upsert_shap_lime_explanations([_sample_explanation_row()])
    store.upsert_forecast_ledger([_sample_forecast_row()])

    assert store.read_latest_eda_summary() is not None
    assert len(store.read_model_registry_metrics()) == 1
    assert len(store.read_model_comparison_series()) == 1
    assert len(store.read_shap_lime_explanations()) == 1
    assert len(store.read_forecast_ledger()) == 1

    shutil.rmtree(cfg.skycast_local_data_dir)
    assert not cfg.skycast_local_data_dir.exists()


def test_storage_import_isolation_from_heavy_ml_and_hopsworks() -> None:
    probe = (
        "import json, sys, skycast.data.storage; "
        "heavy = ['torch','shap','lime','statsmodels','sklearn','hopsworks','hsml','hsfs','skycast.ml']; "
        "print(json.dumps([m for m in heavy if m in sys.modules]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(proc.stdout.strip())
    assert loaded == []


def test_zero_d_ndarray_and_nested_pd_nat_sanitization(tmp_path: Path) -> None:
    cfg = make_test_settings(tmp_path)
    store = SkyCastStorage(cfg)

    store.upsert_eda_summary(
        _sample_eda_summary(
            run_id="eda_0d_nat",
            summary_json={
                "scalar_arr": np.array(3.1415926),
                "nan_arr": np.array(float("nan")),
                "nat_arr": np.array(np.datetime64("NaT")),
                "nested_nat": pd.NaT,
                "list_with_nat": [pd.NaT, np.array(2.5)],
            },
        )
    )
    eda = store.read_latest_eda_summary(run_id="eda_0d_nat")
    assert eda is not None
    assert eda["summary_json"]["scalar_arr"] == 3.141593
    assert eda["summary_json"]["nan_arr"] is None
    assert eda["summary_json"]["nat_arr"] is None
    assert eda["summary_json"]["nested_nat"] is None
    assert eda["summary_json"]["list_with_nat"] == [None, 2.5]
    json.dumps(eda, allow_nan=False)

    store.upsert_model_registry_metrics(
        [
            _sample_registry_row(
                train_run_id="tr_0d_nat",
                hyperparameters={"alpha": np.array(0.25), "cutoff": pd.NaT},
            )
        ]
    )
    reg = store.read_model_registry_metrics(train_run_id="tr_0d_nat")
    assert reg[0]["hyperparameters"] == {"alpha": 0.25, "cutoff": None}

    store.upsert_shap_lime_explanations(
        [
            _sample_explanation_row(
                train_run_id="tr_0d_nat",
                feature_attributions=[
                    {"feature": "f1", "importance": np.array(1.25), "ts": pd.NaT}
                ],
            )
        ]
    )
    xai = store.read_shap_lime_explanations(train_run_id="tr_0d_nat")
    assert xai[0]["feature_attributions"] == [{"feature": "f1", "importance": 1.25, "ts": None}]

    store.record_pipeline_run(
        "pr_0d_nat",
        "training",
        "completed",
        metadata={"arr0": np.array(7.0), "missing_ts": pd.NaT},
    )
    latest_pr = store.get_latest_pipeline_run("training")
    assert latest_pr is not None
    assert latest_pr["metadata_json"] == {"arr0": 7.0, "missing_ts": None}


@pytest.mark.parametrize("mode", ["local_offline", "cloud_supabase"])
def test_non_negative_float_sub_rounding_and_negative_zero_and_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    if mode == "local_offline":
        store = SkyCastStorage(make_test_settings(tmp_path / mode))
    else:
        cfg = make_test_settings(
            tmp_path / mode,
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )
        store = SkyCastStorage(cfg)
        fake_supa = FakeSupabaseClient()
        monkeypatch.setattr(store, "_get_supabase", lambda: fake_supa)

    # Strictly negative sub-rounding floats must be rejected before 6-decimal rounding
    for bad_neg in (-1e-9, -4e-7):
        for field in ("val_rmse", "val_mae", "test_rmse", "test_mae"):
            with pytest.raises(StorageValidationError):
                store.upsert_model_registry_metrics([_sample_registry_row(**{field: bad_neg})])

    # -0.0 normalizes to +0.0 cleanly (including non-negative=False like val_r2=-1e-9)
    store.upsert_model_registry_metrics(
        [
            _sample_registry_row(
                train_run_id="tr_neg_zero",
                val_rmse=-0.0,
                val_mae=-0.0,
                val_r2=-1e-9,
            )
        ]
    )
    rows = store.read_model_registry_metrics(train_run_id="tr_neg_zero")
    assert len(rows) == 1
    assert rows[0]["val_rmse"] == 0.0 and math.copysign(1.0, rows[0]["val_rmse"]) == 1.0
    assert rows[0]["val_r2"] == 0.0 and math.copysign(1.0, rows[0]["val_r2"]) == 1.0

    # Pre-rounding bound inversion must be rejected even when 6-decimal rounding makes values equal
    with pytest.raises(StorageValidationError):
        store.upsert_forecast_ledger(
            [
                _sample_forecast_row(
                    lower_bound=24.0000004,
                    predicted_value=24.0000003,
                    upper_bound=25.0,
                )
            ]
        )
    with pytest.raises(StorageValidationError):
        store.upsert_forecast_ledger(
            [
                _sample_forecast_row(
                    lower_bound=23.0,
                    predicted_value=24.0000004,
                    upper_bound=24.0000003,
                )
            ]
        )


def test_forecast_ledger_latest_run_only_cross_backend_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t1 = "2026-11-18T00:00:00+00:00"
    t2 = "2026-11-18T02:00:00+00:00"
    seed_rows = [
        _sample_forecast_row(
            run_id="run_older",
            issued_at_utc=t1,
            horizon_hour=1,
            model_family="Ridge",
            predicted_value=21.0,
            lower_bound=19.0,
        ),
        _sample_forecast_row(
            run_id="run_older",
            issued_at_utc=t1,
            horizon_hour=2,
            model_family="LSTM",
            predicted_value=22.0,
            lower_bound=20.0,
        ),
        _sample_forecast_row(
            run_id="run_newer",
            issued_at_utc=t2,
            horizon_hour=1,
            model_family="SARIMAX",
            predicted_value=25.0,
        ),
    ]

    off_store = SkyCastStorage(make_test_settings(tmp_path / "off"))
    supa_store = SkyCastStorage(
        make_test_settings(
            tmp_path / "supa",
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )
    )
    fake_supa = FakeSupabaseClient()
    monkeypatch.setattr(supa_store, "_get_supabase", lambda: fake_supa)

    hw_store = SkyCastStorage(
        make_test_settings(
            tmp_path / "hw",
            skycast_offline_mode="false",
            hopsworks_api_key="hw-key",
            hopsworks_project_name="skycast",
        )
    )
    fake_hw = FakeHopsworksProject(artifact_root=tmp_path / "hw_mr")
    fake_hw.attach_to_storage(hw_store, monkeypatch)

    dual_store = SkyCastStorage(
        make_test_settings(
            tmp_path / "dual",
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
            hopsworks_api_key="hw-key",
            hopsworks_project_name="skycast",
        )
    )
    dual_supa = FakeSupabaseClient()
    dual_hw = FakeHopsworksProject(artifact_root=tmp_path / "dual_hw_mr")
    dual_hw.attach_to_storage(dual_store, monkeypatch, fake_supabase=dual_supa)

    for st in (off_store, supa_store, hw_store, dual_store):
        st.upsert_forecast_ledger(seed_rows)

    # Latest run globally is run_newer (which has only SARIMAX, not Ridge)
    assert off_store.read_forecast_ledger(model_family="Ridge", latest_run_only=True) == []
    assert supa_store.read_forecast_ledger(model_family="Ridge", latest_run_only=True) == []
    assert hw_store.read_forecast_ledger(model_family="Ridge", latest_run_only=True) == []
    assert dual_store.read_forecast_ledger(model_family="Ridge", latest_run_only=True) == []
    assert (
        dual_store.read_forecast_ledger(
            model_family="Ridge", latest_run_only=True, prefer_hopsworks=True
        )
        == []
    )

    # end_target_utc / end_issued_utc excluding run_newer must also return [] on all backends
    cutoff = "2026-11-18T01:30:00+00:00"
    for kwargs in (
        {"end_target_utc": cutoff, "latest_run_only": True},
        {"end_issued_utc": cutoff, "latest_run_only": True},
    ):
        assert off_store.read_forecast_ledger(**kwargs) == []
        assert supa_store.read_forecast_ledger(**kwargs) == []
        assert hw_store.read_forecast_ledger(**kwargs) == []
        assert dual_store.read_forecast_ledger(**kwargs) == []
        assert dual_store.read_forecast_ledger(**kwargs, prefer_hopsworks=True) == []

    # Matching model_family='SARIMAX' returns run_newer identically across all backends
    for res in (
        off_store.read_forecast_ledger(model_family="SARIMAX", latest_run_only=True),
        supa_store.read_forecast_ledger(model_family="SARIMAX", latest_run_only=True),
        hw_store.read_forecast_ledger(model_family="SARIMAX", latest_run_only=True),
        dual_store.read_forecast_ledger(model_family="SARIMAX", latest_run_only=True),
        dual_store.read_forecast_ledger(
            model_family="SARIMAX", latest_run_only=True, prefer_hopsworks=True
        ),
    ):
        assert [r["run_id"] for r in res] == ["run_newer"]

    # Identical-timestamp tiebreaking in cloud_hopsworks also selects newest inserted run
    hw_tie_store = SkyCastStorage(
        make_test_settings(
            tmp_path / "hw_tie",
            skycast_offline_mode="false",
            hopsworks_api_key="hw-key",
            hopsworks_project_name="skycast",
        )
    )
    hw_tie_fake = FakeHopsworksProject(artifact_root=tmp_path / "hw_tie_mr")
    hw_tie_fake.attach_to_storage(hw_tie_store, monkeypatch)
    same_ts = "2026-11-19T00:00:00+00:00"
    hw_tie_store.upsert_forecast_ledger([_sample_forecast_row(run_id="tie_A", issued_at_utc=same_ts)])
    hw_tie_store.upsert_forecast_ledger([_sample_forecast_row(run_id="tie_B", issued_at_utc=same_ts)])
    assert [r["run_id"] for r in hw_tie_store.read_forecast_ledger(latest_run_only=True)] == ["tie_B"]


@pytest.mark.parametrize("mode", ["local_offline", "cloud_supabase"])
def test_identical_timestamp_latest_run_only_tiebreaker_id_desc_and_edge_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    if mode == "local_offline":
        store = SkyCastStorage(make_test_settings(tmp_path / mode))
    else:
        cfg = make_test_settings(
            tmp_path / mode,
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )
        store = SkyCastStorage(cfg)
        fake_supa = FakeSupabaseClient()
        monkeypatch.setattr(store, "_get_supabase", lambda: fake_supa)

    # Negative limit & falsy non-dict metadata must raise StorageValidationError (even on cold start)
    for bad_limit in (-1, -5, True):
        with pytest.raises(StorageValidationError):
            store.read_eda_summaries(limit=bad_limit)
    assert store.read_eda_summaries(limit=0) == []

    for bad_meta in ([], "", 0, False):
        with pytest.raises(StorageValidationError):
            store.record_pipeline_run("r_bad_meta", "training", "running", metadata=bad_meta)  # type: ignore[arg-type]

    same_ts = "2026-11-19T00:00:00+00:00"
    for rid in ("tie_A", "tie_B"):
        store.upsert_eda_summary(_sample_eda_summary(run_id=rid, computed_at_utc=same_ts))
        store.upsert_model_registry_metrics([_sample_registry_row(train_run_id=rid, trained_at_utc=same_ts)])
        store.upsert_model_comparison_series([_sample_comparison_row(train_run_id=rid)])
        store.upsert_shap_lime_explanations([_sample_explanation_row(train_run_id=rid, computed_at_utc=same_ts)])
        store.upsert_forecast_ledger([_sample_forecast_row(run_id=rid, issued_at_utc=same_ts)])

    assert store.read_latest_eda_summary()["run_id"] == "tie_B"
    assert [r["train_run_id"] for r in store.read_model_registry_metrics(latest_run_only=True)] == ["tie_B"]
    assert [r["train_run_id"] for r in store.read_model_comparison_series(latest_run_only=True)] == ["tie_B"]
    assert [r["train_run_id"] for r in store.read_shap_lime_explanations(latest_run_only=True)] == ["tie_B"]
    assert [r["run_id"] for r in store.read_forecast_ledger(latest_run_only=True)] == ["tie_B"]


def test_contract_fakes_tightened_direct_call_rejections(tmp_path: Path) -> None:
    fake_supa = FakeSupabaseClient()

    # Invalid ISO timestamp strings rejected directly
    bad_fc_ts = _sample_forecast_row()
    bad_fc_ts["issued_at_utc"] = "not-a-timestamp"
    with pytest.raises(ContractViolationError):
        fake_supa.table("forecast_ledger").upsert(
            [bad_fc_ts],
            on_conflict="run_id,metric,model_family,horizon_hour",
        ).execute()
    with pytest.raises(ContractViolationError):
        fake_supa.table("model_comparison_series").upsert(
            [_sample_comparison_row(timestamp_utc="")],
            on_conflict="train_run_id,metric,timestamp_utc",
        ).execute()

    # Explicit None on defaulted NOT NULL timestamp columns rejected directly
    with pytest.raises(ContractViolationError):
        fake_supa.table("eda_summaries").upsert(
            {"run_id": "e1", "computed_at_utc": None},
            on_conflict="run_id",
        ).execute()
    with pytest.raises(ContractViolationError):
        fake_supa.table("pipeline_runs").upsert(
            {"run_id": "p1", "pipeline_type": "training", "status": "running", "started_at_utc": None},
            on_conflict="run_id",
        ).execute()
    with pytest.raises(ContractViolationError):
        fake_supa.table("model_registry_metrics").upsert(
            [_sample_registry_row(trained_at_utc=None)],
            on_conflict="train_run_id,metric,model_family",
        ).execute()
    with pytest.raises(ContractViolationError):
        fake_supa.table("shap_lime_explanations").upsert(
            [_sample_explanation_row(computed_at_utc=None)],
            on_conflict="train_run_id,metric,method,scope",
        ).execute()

    # FakeHopsworksFeatureGroup rejects bool/float horizon_hour and invalid metric/model_family
    fake_hw = FakeHopsworksProject(artifact_root=tmp_path / "hw_mr")
    fg = fake_hw.feature_store.get_or_create_feature_group(
        name="lahore_forecast_ledger",
        version=1,
        primary_key=["run_id", "metric", "model_family", "horizon_hour"],
        event_time="issued_at_utc",
        online_enabled=True,
    )
    for bad_df in (
        pd.DataFrame([_sample_forecast_row(horizon_hour=True)]),
        pd.DataFrame([_sample_forecast_row(horizon_hour=1.5)]),
        pd.DataFrame([_sample_forecast_row(horizon_hour=1), _sample_forecast_row(horizon_hour=2.0)]),
        pd.DataFrame([_sample_forecast_row(metric="bad_metric")]),
        pd.DataFrame([_sample_forecast_row(model_family="XGBoost")]),
    ):
        with pytest.raises(ContractViolationError):
            fg.insert(bad_df)
