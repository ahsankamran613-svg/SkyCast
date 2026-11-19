"""Self-check tests verifying EM1 E2E test infrastructure, contract fakes, fixtures, and isolation."""

from __future__ import annotations

import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import skycast.config as skycast_config
from skycast.config.settings import METRIC_SPECS, MODEL_FAMILIES, TARGET_METRICS, Settings
from skycast.data.storage import SkyCastStorage
from skycast.features.engineer import (
    get_model_feature_columns,
    split_training_and_inference_rows,
)
from tests.e2e.fakes import (
    ContractSupabaseFake,
    ContractViolationError,
    HopsworksOutageError,
    MockHopsworksProject,
    MockRestAPIError,
    SupabaseOutageError,
)
from tests.e2e.fixtures import (
    generate_lahore_feature_dataframe,
    make_e2e_settings,
    seed_sqlite_history,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_LOCAL_DB = _REPO_ROOT / ".skycast_local" / "skycast.db"


# -----------------------------------------------------------------------------
# 1. Workspace Isolation & Settings Guard
# -----------------------------------------------------------------------------


def test_e2e_settings_isolation_and_local_db_guard(
    e2e_tmp_settings: Settings,
    offline_e2e_storage: SkyCastStorage,
    tmp_path: Path,
) -> None:
    expected_dir = tmp_path / ".skycast_local"
    assert e2e_tmp_settings.skycast_local_data_dir == expected_dir
    assert e2e_tmp_settings.is_offline_mode is True
    settings_mod = sys.modules["skycast.config.settings"]
    assert settings_mod.get_settings() is e2e_tmp_settings
    assert skycast_config.get_settings() is e2e_tmp_settings
    assert settings_mod.settings is e2e_tmp_settings
    assert skycast_config.settings is e2e_tmp_settings

    assert offline_e2e_storage.storage_mode == "local_offline"
    assert offline_e2e_storage.local_db_path == expected_dir / "skycast.db"
    assert offline_e2e_storage.local_db_path.exists()
    assert offline_e2e_storage.local_db_path.resolve() != _REPO_LOCAL_DB.resolve()

    # Guard actively blocks any attempt to open repo .skycast_local/skycast.db
    unisolated_cfg = Settings(
        _env_file=None,
        skycast_offline_mode="true",
        skycast_local_data_dir=_REPO_ROOT / ".skycast_local",
    )
    unisolated_storage = SkyCastStorage(settings=unisolated_cfg)
    with pytest.raises(AssertionError, match="E2E isolation violation"):
        unisolated_storage.init_local_db()


def test_dual_e2e_storage_wires_contract_fakes(dual_e2e_storage: SkyCastStorage) -> None:
    assert dual_e2e_storage.storage_mode == "cloud_dual"
    assert hasattr(dual_e2e_storage, "supabase_fake")
    assert hasattr(dual_e2e_storage, "hopsworks_project")
    assert dual_e2e_storage._get_supabase() is dual_e2e_storage.supabase_fake  # type: ignore[attr-defined]

    import hopsworks

    project = hopsworks.login(host="ignored", project="ignored", api_key_value="ignored")
    assert project is dual_e2e_storage.hopsworks_project  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# 2. ContractSupabaseFake Schema, Checks, Uniques, Pagination & Outages
# -----------------------------------------------------------------------------


def test_contract_supabase_rejects_nan_and_inf(contract_supabase_fake: ContractSupabaseFake) -> None:
    base_row = {
        "run_id": "eda-nan-1",
        "sample_count": 100,
        "summary_json": {"val": float("nan")},
    }
    with pytest.raises(ContractViolationError, match="allow_nan=False"):
        contract_supabase_fake.table("eda_summaries").upsert(base_row, on_conflict="run_id").execute()

    inf_row = {
        "run_id": "fc-inf-1",
        "issued_at_utc": "2026-10-16T00:00:00Z",
        "target_timestamp_utc": "2026-10-16T01:00:00Z",
        "horizon_hour": 1,
        "metric": "temperature_2m",
        "model_family": "Ridge",
        "predicted_value": float("inf"),
        "lower_bound": 20.0,
        "upper_bound": 30.0,
    }
    with pytest.raises(ContractViolationError, match="allow_nan=False"):
        contract_supabase_fake.table("forecast_ledger").insert(inf_row).execute()


def test_contract_supabase_enforces_all_sql_and_domain_checks(
    contract_supabase_fake: ContractSupabaseFake,
) -> None:
    # forecast_ledger: horizon_hour 0, 25, bool rejected; bounds invariant; metric/family enums
    valid_fc = {
        "run_id": "fc-1",
        "issued_at_utc": "2026-10-16T00:00:00Z",
        "target_timestamp_utc": "2026-10-16T01:00:00Z",
        "horizon_hour": 1,
        "metric": "temperature_2m",
        "model_family": "Ridge",
        "predicted_value": 25.0,
        "lower_bound": 23.0,
        "upper_bound": 27.0,
    }
    for bad_h in (0, 25, True):
        bad = {**valid_fc, "horizon_hour": bad_h}
        with pytest.raises(ContractViolationError, match="horizon_hour"):
            contract_supabase_fake.table("forecast_ledger").insert(bad).execute()

    with pytest.raises(ContractViolationError, match="Bounds invariant"):
        contract_supabase_fake.table("forecast_ledger").insert(
            {**valid_fc, "lower_bound": 26.0, "predicted_value": 25.0}
        ).execute()

    with pytest.raises(ContractViolationError, match="metric"):
        contract_supabase_fake.table("forecast_ledger").insert(
            {**valid_fc, "metric": "invalid_metric"}
        ).execute()

    with pytest.raises(ContractViolationError, match="model_family"):
        contract_supabase_fake.table("forecast_ledger").insert(
            {**valid_fc, "model_family": "XGBoost"}
        ).execute()

    # shap_lime_explanations: method & scope CHECK
    valid_xai = {
        "train_run_id": "tr-1",
        "metric": "us_aqi",
        "model_family": "RandomForest",
        "method": "shap",
        "scope": "global",
        "feature_attributions": [{"feature": "pm2_5", "importance": 0.4}],
    }
    with pytest.raises(ContractViolationError, match="method"):
        contract_supabase_fake.table("shap_lime_explanations").insert(
            {**valid_xai, "method": "ig"}
        ).execute()
    with pytest.raises(ContractViolationError, match="scope"):
        contract_supabase_fake.table("shap_lime_explanations").insert(
            {**valid_xai, "scope": "regional"}
        ).execute()

    # model_registry_metrics: negative RMSE rejected
    with pytest.raises(ContractViolationError, match="val_rmse"):
        contract_supabase_fake.table("model_registry_metrics").insert(
            {
                "train_run_id": "tr-1",
                "metric": "us_aqi",
                "model_family": "Ridge",
                "val_rmse": -1.0,
                "val_mae": 1.0,
                "val_r2": 0.5,
                "test_rmse": 2.0,
                "test_mae": 1.5,
                "test_r2": 0.4,
            }
        ).execute()

    # eda_summaries: negative sample_count rejected
    with pytest.raises(ContractViolationError, match="sample_count"):
        contract_supabase_fake.table("eda_summaries").insert(
            {"run_id": "eda-neg", "sample_count": -5}
        ).execute()

    # pipeline_runs: pipeline_type & status CHECK
    with pytest.raises(ContractViolationError, match="pipeline_type"):
        contract_supabase_fake.table("pipeline_runs").insert(
            {"run_id": "pr-1", "pipeline_type": "infer", "status": "running"}
        ).execute()
    with pytest.raises(ContractViolationError, match="status"):
        contract_supabase_fake.table("pipeline_runs").insert(
            {"run_id": "pr-1", "pipeline_type": "training", "status": "queued"}
        ).execute()

    # weather_alerts, weather_reports, agent_call_log, agent_graph_trace, drift_log CHECKs
    with pytest.raises(ContractViolationError, match="severity"):
        contract_supabase_fake.table("weather_alerts").insert(
            {
                "run_id": "r1",
                "metric": "us_aqi",
                "severity": "extreme",
                "title": "t",
                "trigger_value": 200.0,
                "threshold": 150.0,
                "advisory_general": "a",
                "advisory_sensitive": "b",
                "advisory_outdoor_workers": "c",
            }
        ).execute()
    with pytest.raises(ContractViolationError, match="trigger_reason"):
        contract_supabase_fake.table("weather_reports").insert(
            {
                "run_id": "r1",
                "trigger_reason": "cron_unknown",
                "headline": "h",
                "summary": "s",
                "bulletin_markdown": "m",
            }
        ).execute()
    with pytest.raises(ContractViolationError, match="status"):
        contract_supabase_fake.table("agent_call_log").insert(
            {
                "run_id": "r1",
                "agent_name": "alert",
                "provider": "groq",
                "model": "m",
                "prompt": "p",
                "status": "timeout",
            }
        ).execute()
    with pytest.raises(ContractViolationError, match="status"):
        contract_supabase_fake.table("agent_graph_trace").insert(
            {
                "run_id": "r1",
                "node_name": "forecast",
                "step_order": 1,
                "status": "pending",
                "started_at": "2026-10-16T00:00:00Z",
            }
        ).execute()
    with pytest.raises(ContractViolationError, match="drift_type"):
        contract_supabase_fake.table("drift_log").insert(
            {
                "metric": "us_aqi",
                "drift_type": "concept_drift",
                "threshold": 0.25,
                "status": "no_drift",
                "action_taken": "none",
            }
        ).execute()


def test_contract_supabase_unique_and_upsert_semantics(
    contract_supabase_fake: ContractSupabaseFake,
) -> None:
    row_v1 = {
        "train_run_id": "tr-1",
        "metric": "temperature_2m",
        "model_family": "Ridge",
        "val_rmse": 2.5,
        "val_mae": 1.8,
        "val_r2": 0.85,
        "test_rmse": 2.7,
        "test_mae": 1.9,
        "test_r2": 0.82,
    }
    contract_supabase_fake.table("model_registry_metrics").upsert(
        row_v1, on_conflict="train_run_id,metric,model_family"
    ).execute()

    # Plain insert on same UNIQUE key must raise
    with pytest.raises(ContractViolationError, match="UNIQUE constraint"):
        contract_supabase_fake.table("model_registry_metrics").insert(row_v1).execute()

    # Invalid on_conflict must raise
    with pytest.raises(ContractViolationError, match="on_conflict"):
        contract_supabase_fake.table("model_registry_metrics").upsert(
            row_v1, on_conflict="train_run_id,metric"
        ).execute()

    # Valid upsert updates in-place without duplicating
    row_v2 = {**row_v1, "val_rmse": 2.1, "is_champion": True}
    contract_supabase_fake.table("model_registry_metrics").upsert(
        row_v2, on_conflict="train_run_id,metric,model_family"
    ).execute()
    stored = contract_supabase_fake.get_rows("model_registry_metrics")
    assert len(stored) == 1
    assert stored[0]["val_rmse"] == 2.1
    assert stored[0]["is_champion"] is True
    assert stored[0]["id"] == 1

    # Batch rollback: if row 2 violates UNIQUE, row 1 must not be committed
    bad_batch = [
        {**row_v1, "model_family": "LSTM"},
        row_v1,
    ]
    with pytest.raises(ContractViolationError, match="UNIQUE constraint"):
        contract_supabase_fake.table("model_registry_metrics").insert(bad_batch).execute()
    stored_after_fail = contract_supabase_fake.get_rows("model_registry_metrics")
    assert len(stored_after_fail) == 1
    assert stored_after_fail[0]["model_family"] == "Ridge"


def test_contract_supabase_1000_row_cap_pagination_and_filters(
    contract_supabase_fake: ContractSupabaseFake,
) -> None:
    base_ts = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    total_rows = 2350
    batch = [
        {
            "train_run_id": "tr-big",
            "metric": "temperature_2m",
            "timestamp_utc": (base_ts + timedelta(hours=i)).isoformat(),
            "actual": float(25.0 + (i % 10)),
            "ridge": float(25.2 + (i % 10)),
        }
        for i in range(total_rows)
    ]
    contract_supabase_fake.table("model_comparison_series").upsert(
        batch, on_conflict="train_run_id,metric,timestamp_utc"
    ).execute()

    # Unpaginated read and overly wide .range() both cap at 1000 rows
    unpaged = (
        contract_supabase_fake.table("model_comparison_series")
        .select("*")
        .order("timestamp_utc", desc=False)
        .execute()
    )
    assert len(unpaged.data) == 1000

    wide_range = (
        contract_supabase_fake.table("model_comparison_series")
        .select("*")
        .range(0, 2349)
        .execute()
    )
    assert len(wide_range.data) == 1000

    # Proper .range(offset, offset + 999) pagination retrieves all 2350 rows cleanly
    collected: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while True:
        resp = (
            contract_supabase_fake.table("model_comparison_series")
            .select("*", count="exact")
            .eq("train_run_id", "tr-big")
            .order("timestamp_utc", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        assert resp.count == total_rows
        collected.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size

    assert len(collected) == total_rows
    assert collected[0]["timestamp_utc"] < collected[-1]["timestamp_utc"]

    # Range filtering with UTC ISO Z vs +00:00
    filtered = (
        contract_supabase_fake.table("model_comparison_series")
        .select("timestamp_utc,actual")
        .gte("timestamp_utc", (base_ts + timedelta(hours=100)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        .lte("timestamp_utc", (base_ts + timedelta(hours=109)).isoformat())
        .order("timestamp_utc", desc=True)
        .limit(5)
        .execute()
    )
    assert len(filtered.data) == 5
    assert set(filtered.data[0].keys()) == {"timestamp_utc", "actual"}


def test_contract_supabase_injectable_outages(contract_supabase_fake: ContractSupabaseFake) -> None:
    contract_supabase_fake.set_failure(fail_writes=True)
    with pytest.raises(SupabaseOutageError):
        contract_supabase_fake.table("eda_summaries").upsert(
            {"run_id": "eda-outage", "sample_count": 10}, on_conflict="run_id"
        ).execute()

    contract_supabase_fake.set_failure(fail_writes=False, fail_reads=True)
    with pytest.raises(SupabaseOutageError):
        contract_supabase_fake.table("eda_summaries").select("*").execute()

    contract_supabase_fake.set_failure(fail_reads=False, fail_on_tables={"forecast_ledger"})
    with pytest.raises(SupabaseOutageError):
        contract_supabase_fake.table("forecast_ledger").select("*").execute()


# -----------------------------------------------------------------------------
# 3. MockHopsworks Project, Feature Store & Model Registry
# -----------------------------------------------------------------------------


def test_mock_hopsworks_feature_store_roundtrip_and_outage(
    mock_hopsworks_project: MockHopsworksProject,
) -> None:
    fs = mock_hopsworks_project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="lahore_forecast_ledger",
        version=1,
        primary_key=["run_id", "metric", "model_family", "horizon_hour"],
        event_time="target_timestamp_utc",
        online_enabled=True,
    )
    t0 = pd.Timestamp("2026-10-16T01:00:00Z")
    df1 = pd.DataFrame(
        {
            "run_id": ["r1", "r1"],
            "metric": ["us_aqi", "us_aqi"],
            "model_family": ["Ridge", "Ridge"],
            "horizon_hour": [1, 2],
            "target_timestamp_utc": [t0, t0 + pd.Timedelta(hours=1)],
            "predicted_value": [150.0, 155.0],
        }
    )
    fg.insert(df1, operation="upsert")
    # Upsert update on horizon_hour=2 should deduplicate
    df2 = pd.DataFrame(
        {
            "run_id": ["r1"],
            "metric": ["us_aqi"],
            "model_family": ["Ridge"],
            "horizon_hour": [2],
            "target_timestamp_utc": [t0 + pd.Timedelta(hours=1)],
            "predicted_value": [162.5],
        }
    )
    fg.insert(df2, operation="upsert")

    read_df = fs.get_feature_group("lahore_forecast_ledger", version=1).read(
        start_time=t0 + pd.Timedelta(hours=1)
    )
    assert len(read_df) == 1
    assert float(read_df["predicted_value"].iloc[0]) == pytest.approx(162.5)

    with pytest.raises(MockRestAPIError):
        fs.get_feature_group("non_existent_fg", version=1)

    fs.fail_on_insert = True
    with pytest.raises(HopsworksOutageError):
        fg.insert(df2)


def test_mock_hopsworks_model_registry_cold_start_versioning_and_download(
    tmp_path: Path,
) -> None:
    project = MockHopsworksProject(artifact_root=tmp_path / "hw_artifacts")
    mr = project.get_model_registry()

    # Cold start returns empty / None
    assert mr.get_models("lahore_us_aqi_champion") == []
    assert mr.get_model("lahore_us_aqi_champion") is None
    assert mr.get_best_model("lahore_us_aqi_champion", "val_rmse", "min") is None

    # Save v1 and v2 with real artifact files
    src_v1 = tmp_path / "bundle_v1"
    src_v1.mkdir()
    (src_v1 / "model.json").write_text('{"version": 1}', encoding="utf-8")
    m1 = mr.python.create_model(
        name="lahore_us_aqi_champion",
        metrics={"val_rmse": 18.4, "val_mae": 12.1},
        description="v1",
    )
    saved_v1 = m1.save(src_v1)
    assert saved_v1.version == 1

    src_v2 = tmp_path / "bundle_v2"
    src_v2.mkdir()
    (src_v2 / "model.json").write_text('{"version": 2}', encoding="utf-8")
    m2 = mr.python.create_model(
        name="lahore_us_aqi_champion",
        metrics={"val_rmse": 14.2, "val_mae": 9.8},
        description="v2",
    )
    saved_v2 = m2.save(src_v2)
    assert saved_v2.version == 2

    best = mr.get_best_model("lahore_us_aqi_champion", "val_rmse", "min")
    assert best is not None
    assert best.version == 2

    dl_dir = Path(best.download(tmp_path / "downloaded_best"))
    assert (dl_dir / "model.json").read_text(encoding="utf-8") == '{"version": 2}'

    # Optional raise_not_found_on_missing=True mode + outage mode
    strict_mr = MockHopsworksProject(
        artifact_root=tmp_path / "hw_strict"
    ).get_model_registry()
    strict_mr.raise_not_found_on_missing = True
    with pytest.raises(MockRestAPIError):
        strict_mr.get_models("missing_model")

    mr.fail_on_save = True
    with pytest.raises(HopsworksOutageError):
        mr.python.create_model(name="lahore_us_aqi_champion", metrics={"val_rmse": 10.0}).save(src_v2)


# -----------------------------------------------------------------------------
# 4. Synthetic Lahore Data Generator & SQLite Seeder
# -----------------------------------------------------------------------------


def test_generate_lahore_feature_dataframe_determinism_and_schema(
    lahore_30d_df: pd.DataFrame,
    lahore_custom_h12_df: pd.DataFrame,
    e2e_tmp_settings: Settings,
) -> None:
    df_dup = generate_lahore_feature_dataframe(n_hours=720, horizon_hours=24, seed=42)
    df_other = generate_lahore_feature_dataframe(n_hours=720, horizon_hours=24, seed=99)
    pd.testing.assert_frame_equal(lahore_30d_df, df_dup)
    assert not lahore_30d_df["temperature_2m"].equals(df_other["temperature_2m"])

    # Contiguous 1h UTC grid
    ts = pd.to_datetime(lahore_30d_df["timestamp_utc"], utc=True)
    assert len(ts) == 720
    assert (ts.diff().dropna() == pd.Timedelta(hours=1)).all()

    # Exact 270 model features and valid domain bounds
    feat_cols = get_model_feature_columns(lahore_30d_df, settings=e2e_tmp_settings)
    assert len(feat_cols) == 270
    for metric, spec in METRIC_SPECS.items():
        vals = lahore_30d_df[metric]
        assert (vals >= spec.min_value).all()
        if spec.is_circular:
            assert (vals < spec.max_value).all()
        else:
            assert (vals <= spec.max_value).all()

    # Train/inference split counts for H=24 and H=12
    train_24, inf_24 = split_training_and_inference_rows(lahore_30d_df, settings=e2e_tmp_settings)
    assert len(train_24) == 720 - 48 - 24
    assert len(inf_24) == 24
    assert not train_24[feat_cols].isna().any().any()

    cfg_h12 = Settings(_env_file=None, forecast_horizon_hours=12)
    assert "target_temperature_2m_t12" in lahore_custom_h12_df.columns
    assert "target_temperature_2m_t13" not in lahore_custom_h12_df.columns
    train_12, inf_12 = split_training_and_inference_rows(lahore_custom_h12_df, settings=cfg_h12)
    assert len(train_12) == 720 - 48 - 12
    assert len(inf_12) == 12


def test_generate_lahore_feature_dataframe_regime_switches() -> None:
    # Smog + monsoon + wind wraparound
    mixed = generate_lahore_feature_dataframe(
        n_hours=720, include_smog=True, include_monsoon=True, dry_mode=False, wind_wraparound=True
    )
    assert bool(mixed["is_lahore_smog_season"].any())
    assert bool(mixed["is_monsoon_season"].any())
    assert float(mixed["us_aqi"].max()) >= 150.0
    assert float(mixed["precipitation_mm"].max()) > 0.0
    assert ((mixed["wind_direction_10m"] >= 350.0) & (mixed["wind_direction_10m"] < 360.0)).any()
    assert ((mixed["wind_direction_10m"] >= 0.0) & (mixed["wind_direction_10m"] <= 10.0)).any()

    # Dry mode + zero-variance column
    dry_zv = generate_lahore_feature_dataframe(
        n_hours=240,
        include_smog=False,
        include_monsoon=False,
        dry_mode=True,
        wind_wraparound=False,
        zero_var_cols=("cloud_cover",),
    )
    assert (dry_zv["precipitation_mm"] == 0.0).all()
    assert dry_zv["cloud_cover"].nunique() == 1
    assert float(dry_zv["cloud_cover"].std(ddof=0)) == 0.0
    assert not bool(dry_zv["is_lahore_smog_season"].any())
    assert not bool(dry_zv["is_monsoon_season"].any())


def test_seed_sqlite_history_roundtrip_with_read_training_history(
    offline_e2e_storage: SkyCastStorage,
    tmp_path: Path,
) -> None:
    recent_df = generate_lahore_feature_dataframe(n_hours=120, horizon_hours=24, end_utc="now")
    summary = seed_sqlite_history(offline_e2e_storage, recent_df)
    assert summary["sqlite_rows"] == 120

    loaded = offline_e2e_storage.read_training_history(days=30)
    assert len(loaded) == 120
    feat_cols = get_model_feature_columns(loaded, settings=offline_e2e_storage.settings)
    assert len(feat_cols) == 270
    assert offline_e2e_storage.local_db_path == tmp_path / ".skycast_local" / "skycast.db"

    # Non-default Settings helper check
    custom_cfg = make_e2e_settings(tmp_path, horizon_hours=12, offline=True, ridge_alpha=3.5)
    assert custom_cfg.forecast_horizon_hours == 12
    assert custom_cfg.ridge_alpha == 3.5
    assert len(TARGET_METRICS) == 8
    assert len(MODEL_FAMILIES) == 5


# -----------------------------------------------------------------------------
# 5. AST Lazy-Import Collection Hygiene Check
# -----------------------------------------------------------------------------


def test_e2e_suite_has_no_top_level_ml_or_train_imports() -> None:
    e2e_dir = Path(__file__).resolve().parent
    forbidden_prefixes = ("skycast.ml", "skycast.pipelines.train")
    for py_file in sorted(e2e_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"Top-level import '{alias.name}' forbidden in {py_file.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), (
                    f"Top-level from-import '{node.module}' forbidden in {py_file.name}"
                )
