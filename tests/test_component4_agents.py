"""Component 4 tests: LLM failover cascade, 4-node LangGraph routing, drift, and idempotency."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

from skycast.agents.drift import backfill_forecast_ledger_actuals, evaluate_and_log_drift
from skycast.agents.graph import run_agent_pipeline
from skycast.agents.llm_client import ResilientLLMClient
from skycast.config.settings import METRIC_SPECS, TARGET_METRICS, Settings
from skycast.data.storage import SkyCastStorage
from skycast.ml.eda import compute_eda_summary
from skycast.pipelines.train import run_training
from tests.e2e.conftest import _patch_settings_everywhere
from tests.e2e.fakes import ContractSupabaseFake, MockHopsworksProject
from tests.e2e.fixtures import generate_lahore_feature_dataframe, make_e2e_settings


def _wire_dual_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: Settings,
) -> tuple[SkyCastStorage, ContractSupabaseFake, MockHopsworksProject]:
    _patch_settings_everywhere(monkeypatch, cfg)
    store = SkyCastStorage(cfg)
    store.init_local_db()
    supa_fake = ContractSupabaseFake(strict_domain_enums=True)
    hw_proj = MockHopsworksProject(artifact_root=tmp_path / "hw_mr")
    monkeypatch.setattr(store, "_get_supabase", lambda: supa_fake)
    monkeypatch.setattr(store, "_get_hopsworks_project", lambda: hw_proj)
    return store, supa_fake, hw_proj


def _install_fake_llm_sdks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    groq_handler: Any,
    gemini_handler: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groq_calls: list[dict[str, Any]] = []
    gemini_calls: list[dict[str, Any]] = []

    class _FakeGroqCompletions:
        def create(self, **kwargs: Any) -> Any:
            groq_calls.append(dict(kwargs))
            text = groq_handler(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    class _FakeGroq:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.chat = SimpleNamespace(completions=_FakeGroqCompletions())

    class _FakeGenAIModels:
        def generate_content(self, **kwargs: Any) -> Any:
            gemini_calls.append(dict(kwargs))
            text = gemini_handler(kwargs)
            return SimpleNamespace(text=text)

    class _FakeGenAIClient:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.models = _FakeGenAIModels()

    fake_groq_mod = ModuleType("groq")
    fake_groq_mod.Groq = _FakeGroq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq", fake_groq_mod)

    import google.genai as real_genai

    monkeypatch.setattr(real_genai, "Client", _FakeGenAIClient)
    return groq_calls, gemini_calls


def _seed_baselines_and_window(
    store: SkyCastStorage,
    cfg: Settings,
    *,
    n_live_hours: int,
    n_ledger_hours: int,
    end_ts: pd.Timestamp,
    live_temp_shift: float = 0.0,
    pred_error_offset: float = 0.0,
    dry_mode: bool = True,
    persist_baseline_history: bool = True,
) -> pd.DataFrame:
    baseline_end = end_ts - pd.Timedelta(hours=max(1, n_live_hours))
    base_df = generate_lahore_feature_dataframe(
        n_hours=240,
        horizon_hours=cfg.forecast_horizon_hours,
        seed=42,
        include_smog=False,
        include_monsoon=False,
        dry_mode=dry_mode,
        end_utc=baseline_end,
    )
    eda = compute_eda_summary(base_df, settings=cfg, run_id="eda_baseline_1")
    store.upsert_eda_summary(eda)

    reg_rows = [
        {
            "train_run_id": "train_baseline_1",
            "trained_at_utc": (end_ts - pd.Timedelta(hours=24)).isoformat(),
            "metric": m,
            "model_family": "Ridge",
            "model_version": "1",
            "val_rmse": 1.0,
            "val_mae": 0.8,
            "val_r2": 0.9,
            "test_rmse": 1.1,
            "test_mae": 0.9,
            "test_r2": 0.88,
            "improvement_over_naive_pct": 15.0,
            "is_champion": True,
            "promoted_to_registry": True,
            "hyperparameters": {"alpha": 1.0},
            "training_samples": 168,
        }
        for m in TARGET_METRICS
    ]
    store.upsert_model_registry_metrics(reg_rows)

    if persist_baseline_history:
        store.upsert_weather_features(base_df)

    if n_live_hours > 0:
        rng = np.random.default_rng(7)
        sample_idx = np.sort(rng.choice(len(base_df), size=n_live_hours, replace=False))
        live_full = base_df.iloc[sample_idx].copy().reset_index(drop=True)
        live_ts = pd.date_range(end=end_ts, periods=n_live_hours, freq="1h", tz="UTC")
        live_full["timestamp_utc"] = live_ts
        if live_temp_shift != 0.0:
            live_full["temperature_2m"] = live_full["temperature_2m"] + live_temp_shift
        store.upsert_weather_features(live_full)
    else:
        live_full = base_df.tail(0)

    if n_ledger_hours > 0:
        obs_by_ts = {
            pd.Timestamp(r["timestamp_utc"]): r for r in live_full.to_dict(orient="records")
        }
        target_range = pd.date_range(end=end_ts, periods=n_ledger_hours, freq="1h", tz="UTC")
        fc_rows: list[dict[str, Any]] = []
        for idx, t_ts in enumerate(target_range):
            obs_row = obs_by_ts.get(pd.Timestamp(t_ts))
            for m in TARGET_METRICS:
                obs_val = float(obs_row[m]) if obs_row is not None else 20.0
                spec = METRIC_SPECS[m]
                pred_val = spec.clamp(obs_val + pred_error_offset)
                if spec.is_circular and pred_val >= 360.0:
                    pred_val = 0.0
                low_val = spec.clamp(pred_val - 1.0) if not spec.is_circular else 0.0
                high_val = spec.clamp(pred_val + 1.0) if not spec.is_circular else 359.9999
                low_val = min(low_val, pred_val)
                high_val = max(high_val, pred_val)
                fc_rows.append(
                    {
                        "run_id": f"past_fc_{idx:03d}",
                        "issued_at_utc": (t_ts - pd.Timedelta(hours=cfg.forecast_horizon_hours)).isoformat(),
                        "target_timestamp_utc": t_ts.isoformat(),
                        "horizon_hour": cfg.forecast_horizon_hours,
                        "metric": m,
                        "model_family": "Ridge",
                        "model_version": "1",
                        "predicted_value": pred_val,
                        "lower_bound": low_val,
                        "upper_bound": high_val,
                        "naive_baseline_value": pred_val,
                        "actual_value": None,
                    }
                )
        store.upsert_forecast_ledger(fc_rows)

    return base_df


def test_forced_3tier_outage_cascade_immediate_logging_and_deterministic_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_e2e_settings(
        tmp_path,
        offline=False,
        groq_api_key="test-groq-key",
        gemini_api_key="test-gemini-key",
    )
    store, fake_supa, fake_hw = _wire_dual_store(tmp_path, monkeypatch, cfg)

    # Track exact chronological order of SDK calls vs storage.insert_agent_call_logs
    events: list[str] = []
    orig_insert_calls = store.insert_agent_call_logs

    def _spy_insert_calls(rows: Any, **kwargs: Any) -> dict[str, Any]:
        for r in rows:
            events.append(
                f"log:{r['agent_name']}:attempt={r['attempt_order']}:status={r['status']}:provider={r['provider']}"
            )
        return orig_insert_calls(rows, **kwargs)

    monkeypatch.setattr(store, "insert_agent_call_logs", _spy_insert_calls)

    def _groq_fail(kwargs: dict[str, Any]) -> str:
        model = kwargs["model"]
        events.append(f"call:groq:{model}:reasoning={kwargs.get('reasoning_effort')}")
        if model == cfg.PRIMARY_LLM_MODEL:
            raise RuntimeError("Groq 429 Too Many Requests")
        raise RuntimeError("Groq 503 Service Unavailable")

    def _gemini_timeout(kwargs: dict[str, Any]) -> str:
        model = kwargs["model"]
        events.append(f"call:gemini:{model}")
        raise TimeoutError("Gemini request timed out after 15.0s")

    groq_calls, gemini_calls = _install_fake_llm_sdks(
        monkeypatch, groq_handler=_groq_fail, gemini_handler=_gemini_timeout
    )

    # First verify isolated ResilientLLMClient call logs 3 failed attempts immediately before deterministic fallback
    llm_client = ResilientLLMClient(settings=cfg, storage=store)
    isolated = llm_client.generate(
        run_id="cascade_unit_1",
        agent_name="alert_agent",
        system_prompt="Return JSON",
        user_prompt="Lahore AQI 260",
        deterministic_fallback_fn=lambda: {
            "alerts": [
                {
                    "rule_id": "aqi_very_unhealthy",
                    "metric": "us_aqi",
                    "title": "Fallback alert",
                    "advisory_general": "Stay inside",
                    "advisory_sensitive": "Mask indoors",
                    "advisory_outdoor_workers": "Wear N95",
                }
            ]
        },
        storage=store,
    )
    assert isolated.used_deterministic_template is True
    assert isolated.provider == "deterministic_template"
    assert isolated.fallback_used is True

    # Verify Groq passed reasoning_effort="low", Gemini omitted reasoning_effort completely
    assert len(groq_calls) == 2
    assert all(c["reasoning_effort"] == cfg.GROQ_REASONING_EFFORT for c in groq_calls)
    assert len(gemini_calls) == 1
    assert "reasoning_effort" not in gemini_calls[0]
    assert not hasattr(gemini_calls[0]["config"], "reasoning_effort")

    # Verify immediate logging order before next tier
    assert events == [
        f"call:groq:{cfg.PRIMARY_LLM_MODEL}:reasoning=low",
        "log:alert_agent:attempt=1:status=failed:provider=groq",
        f"call:gemini:{cfg.FALLBACK_LLM_MODEL}",
        "log:alert_agent:attempt=2:status=failed:provider=gemini",
        f"call:groq:{cfg.TERTIARY_LLM_MODEL}:reasoning=low",
        "log:alert_agent:attempt=3:status=failed:provider=groq",
    ]

    unit_logs = store.read_agent_call_log(run_id="cascade_unit_1")
    assert [r["attempt_order"] for r in unit_logs] == [1, 2, 3]
    assert [r["status"] for r in unit_logs] == ["failed", "failed", "failed"]
    assert [r["fallback_used"] for r in unit_logs] == [False, True, True]
    assert [r["reasoning_effort"] for r in unit_logs] == ["low", None, "low"]
    assert "429" in unit_logs[0]["error_message"]
    assert "timed out" in unit_logs[1]["error_message"]
    assert "503" in unit_logs[2]["error_message"]
    assert all(r["latency_ms"] >= 0.0 for r in unit_logs)

    # Now execute full 4-node pipeline under total 3-tier outage with hazard AQI & daily hour (01:00Z = 06:00 PKT)
    df = generate_lahore_feature_dataframe(n_hours=60, horizon_hours=24, seed=42, end_utc="2026-11-19T01:00:00Z")
    latest_row = df.iloc[-1].to_dict()
    latest_row["us_aqi"] = 255.0
    latest_row["temperature_2m"] = 45.2  # triggers both us_aqi critical and temperature_2m critical
    store.upsert_weather_features(df)

    res = run_agent_pipeline(
        run_id="cascade_e2e_1",
        timestamp_utc="2026-11-19T01:00:00+00:00",
        latest_features=latest_row,
        settings=cfg,
        storage=store,
    )

    assert len(res["forecasts_24h"]) == len(TARGET_METRICS)
    assert len(res["alerts_payload"]) >= 2
    for alert in res["alerts_payload"]:
        assert alert["llm_provider"] == "deterministic_template"
        assert alert["advisory_general"].strip()
        assert alert["advisory_sensitive"].strip()
        assert alert["advisory_outdoor_workers"].strip()

    report = res["weather_report"]
    assert report is not None
    assert report["trigger_reason"] == "daily_schedule"
    assert report["fallback_used"] is True
    assert report["llm_provider"] == "deterministic_template"
    assert report["active_alert_count"] == len(res["alerts_payload"])

    e2e_calls = store.read_agent_call_log(run_id="cascade_e2e_1")
    alert_calls = [c for c in e2e_calls if c["agent_name"] == "alert_agent"]
    report_calls = [c for c in e2e_calls if c["agent_name"] == "weather_report_agent"]
    assert [c["attempt_order"] for c in alert_calls] == [1, 2, 3]
    assert [c["attempt_order"] for c in report_calls] == [1, 2, 3]
    assert all(c["status"] == "failed" for c in e2e_calls)

    traces = store.read_agent_graph_trace(run_id="cascade_e2e_1")
    assert [(t["step_order"], t["node_name"], t["status"]) for t in traces] == [
        (1, "forecast_agent", "completed"),
        (2, "alert_agent", "completed"),
        (3, "drift_agent", "completed"),
        (4, "weather_report_agent", "completed"),
    ]
    assert len(fake_supa.get_rows("weather_alerts")) == len(res["alerts_payload"])
    assert len(fake_hw.feature_store.get_feature_group(cfg.hopsworks_agent_log_fg, 1).df) == 9


def test_llm_failover_tiers_and_malformed_json_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_e2e_settings(
        tmp_path,
        offline=True,
        groq_api_key="test-groq",
        gemini_api_key="test-gemini",
        gemini_fallback_model="gemini-3.5-flash-lite-custom",
        groq_tertiary_model="qwen-3.8-27b-custom",
    )
    store = SkyCastStorage(cfg)

    # Case 1: Groq tier 1 returns malformed JSON -> logged as failed, Gemini tier 2 succeeds!
    def _groq_bad_json(kwargs: dict[str, Any]) -> str:
        assert kwargs["reasoning_effort"] == "low"
        return "not valid json"

    def _gemini_good_json(kwargs: dict[str, Any]) -> str:
        assert "reasoning_effort" not in kwargs
        assert kwargs["model"] == "gemini-3.5-flash-lite-custom"
        return json.dumps(
            {
                "headline": "Lahore Evening Bulletin",
                "summary": "Clear skies across Lahore.",
                "bulletin_markdown": "## Bulletin\nClear skies.",
                "highlights": ["Clear skies", "Moderate AQI"],
            }
        )

    _install_fake_llm_sdks(
        monkeypatch, groq_handler=_groq_bad_json, gemini_handler=_gemini_good_json
    )
    client = ResilientLLMClient(settings=cfg, storage=store)
    from skycast.agents.templates import validate_llm_report_payload

    res_gemini = client.generate(
        run_id="tier2_ok",
        agent_name="weather_report_agent",
        system_prompt="Return JSON",
        user_prompt="Summary",
        deterministic_fallback_fn=lambda: {"headline": "x", "summary": "x", "bulletin_markdown": "x", "highlights": ["x"]},
        validator_fn=validate_llm_report_payload,
        storage=store,
    )
    assert res_gemini.provider == "gemini"
    assert res_gemini.model == "gemini-3.5-flash-lite-custom"
    assert res_gemini.reasoning_effort is None
    assert res_gemini.fallback_used is True

    logs = store.read_agent_call_log(run_id="tier2_ok")
    assert [(r["attempt_order"], r["provider"], r["status"]) for r in logs] == [
        (1, "groq", "failed"),
        (2, "gemini", "success"),
    ]


def test_drift_missing_baseline_and_insufficient_window_write_check_failed(
    tmp_path: Path,
) -> None:
    cfg = make_e2e_settings(tmp_path, offline=True)
    store = SkyCastStorage(cfg)
    now_ts = pd.Timestamp("2026-11-19T10:00:00Z")

    # 1. Missing baselines altogether (even with 80h of observations present!)
    df = generate_lahore_feature_dataframe(n_hours=80, horizon_hours=24, seed=42, end_utc=now_ts)
    store.upsert_weather_features(df)
    drift_rows_1, retrain_1, _ = evaluate_and_log_drift(
        run_id="drift_missing_base",
        timestamp_utc=now_ts,
        storage=store,
        settings=cfg,
    )
    assert retrain_1 is False
    assert len(drift_rows_1) == 2 * len(TARGET_METRICS)
    for row in drift_rows_1:
        assert row["status"] == "check_failed"
        assert row["score"] is None
        assert row["secondary_score"] is None
        assert row["error_message"] and "Missing" in row["error_message"]

    # 2. Baselines exist, but live rows < 48 and matched forecast_ledger rows < 72
    store_2 = SkyCastStorage(make_e2e_settings(tmp_path / "insufficient", offline=True))
    _seed_baselines_and_window(
        store_2,
        cfg,
        n_live_hours=30,  # < 48 data_drift_window_hours
        n_ledger_hours=40,  # < 72 model_drift_window_hours
        end_ts=now_ts,
        persist_baseline_history=False,
    )
    drift_rows_2, retrain_2, _ = evaluate_and_log_drift(
        run_id="drift_short_win",
        timestamp_utc=now_ts,
        storage=store_2,
        settings=cfg,
    )
    assert retrain_2 is False
    for row in drift_rows_2:
        assert row["status"] == "check_failed"
        assert row["score"] is None
        assert row["secondary_score"] is None
        assert "Insufficient" in (row["error_message"] or "")


def test_drift_backfill_no_drift_and_detected_triggers_github_dispatch(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    now_ts = pd.Timestamp("2026-11-19T12:00:00Z")

    # Non-default drift windows + thresholds + GitHub workflow dispatch
    cfg = make_e2e_settings(
        tmp_path / "healthy",
        offline=True,
        data_drift_window_hours=48,
        model_drift_window_hours=48,
        data_drift_psi_threshold=0.25,
        model_drift_rmse_ratio_threshold=1.35,
        github_token="ghp-test-token",
        github_repo="owner/skycast",
        github_train_workflow="custom_train.yml",
        github_ref="release",
    )
    store = SkyCastStorage(cfg)
    _seed_baselines_and_window(
        store,
        cfg,
        n_live_hours=48,
        n_ledger_hours=48,
        end_ts=now_ts,
        live_temp_shift=0.0,
        pred_error_offset=0.2,
        dry_mode=True,
    )

    # Confirm actual_value starts as None, then evaluate_and_log_drift backfills before scoring
    before_ledger = store.read_forecast_ledger()
    assert all(r["actual_value"] is None for r in before_ledger)

    rows_ok, retrain_ok, summary_ok = evaluate_and_log_drift(
        run_id="drift_ok_run",
        timestamp_utc=now_ts,
        storage=store,
        settings=cfg,
    )
    assert retrain_ok is False
    assert summary_ok["backfilled_actual_rows"] == 48 * len(TARGET_METRICS)
    assert all(r["status"] == "no_drift" for r in rows_ok)
    after_ledger = store.read_forecast_ledger()
    assert all(r["actual_value"] is not None for r in after_ledger)

    # Now inject data drift (+15C shift -> PSI clipped 1e-6 & KS p < 0.01) AND model drift (+4.0 error)
    drift_cfg = make_e2e_settings(
        tmp_path / "drifted",
        offline=True,
        data_drift_window_hours=48,
        model_drift_window_hours=48,
        github_token="ghp-test-token",
        github_repo="owner/skycast",
        github_train_workflow="custom_train.yml",
        github_ref="release",
    )
    drift_store = SkyCastStorage(drift_cfg)
    _seed_baselines_and_window(
        drift_store,
        drift_cfg,
        n_live_hours=48,
        n_ledger_hours=48,
        end_ts=now_ts,
        live_temp_shift=15.0,
        pred_error_offset=4.0,
        dry_mode=True,
    )

    expected_dispatch_url = "https://api.github.com/repos/owner/skycast/actions/workflows/custom_train.yml/dispatches"
    httpx_mock.add_response(method="POST", url=expected_dispatch_url, status_code=204)

    rows_drift, retrain_drift, _ = evaluate_and_log_drift(
        run_id="drift_alert_run",
        timestamp_utc=now_ts,
        storage=drift_store,
        settings=drift_cfg,
    )
    assert retrain_drift is True
    temp_data = next(r for r in rows_drift if r["metric"] == "temperature_2m" and r["drift_type"] == "data_drift")
    temp_model = next(r for r in rows_drift if r["metric"] == "temperature_2m" and r["drift_type"] == "model_drift")
    assert temp_data["status"] == "drift_detected"
    assert temp_data["score"] > drift_cfg.data_drift_psi_threshold
    assert temp_data["secondary_score"] < drift_cfg.data_drift_ks_pvalue_threshold
    assert temp_data["retrain_triggered"] is True
    assert temp_data["action_taken"] == "github_workflow_dispatch_triggered"
    assert temp_model["status"] == "drift_detected"
    assert temp_model["score"] > drift_cfg.model_drift_rmse_ratio_threshold

    sent_req = httpx_mock.get_requests()[0]
    assert json.loads(sent_req.content.decode("utf-8"))["ref"] == "release"

    # Forced GitHub workflow_dispatch failure (HTTP 500) must never report retrain_triggered=True
    httpx_mock.add_response(method="POST", url=expected_dispatch_url, status_code=500, text="Server Error")
    rows_disp_fail, retrain_disp_fail, _ = evaluate_and_log_drift(
        run_id="drift_dispatch_fail_run",
        timestamp_utc=now_ts,
        storage=drift_store,
        settings=drift_cfg,
    )
    assert retrain_disp_fail is False
    drifted_failed = [r for r in rows_disp_fail if r["status"] == "drift_detected"]
    assert drifted_failed
    assert all(r["retrain_triggered"] is False for r in drifted_failed)
    assert all(r["action_taken"] == "retrain_dispatch_failed" for r in drifted_failed)
    assert all("500" in (r["error_message"] or "") for r in drifted_failed)


def test_timezone_routing_and_non_default_settings_horizon_and_tz(
    tmp_path: Path,
) -> None:
    # Default Lahore TZ (Asia/Karachi = UTC+5): 01:00 UTC == 06:00 PKT -> daily_schedule!
    # 09:00 UTC == 14:00 PKT with calm metrics -> weather_report_agent skipped!
    cfg_default = make_e2e_settings(tmp_path / "pkt", offline=True)
    store_pkt = SkyCastStorage(cfg_default)
    calm_df = generate_lahore_feature_dataframe(
        n_hours=60,
        horizon_hours=24,
        seed=7,
        include_smog=False,
        include_monsoon=False,
        dry_mode=True,
        end_utc="2026-04-15T09:00:00Z",
    )
    calm_row = calm_df.iloc[-1].to_dict()
    for m, val in {
        "us_aqi": 45.0,
        "temperature_2m": 25.0,
        "precipitation_mm": 0.0,
        "precipitation_prob": 5.0,
        "relative_humidity_2m": 45.0,
        "uv_index": 2.0,
        "wind_speed_10m": 8.0,
        "wind_direction_10m": 180.0,
    }.items():
        calm_row[m] = val
    calm_df.iloc[-1, calm_df.columns.get_loc("us_aqi")] = 45.0
    store_pkt.upsert_weather_features(calm_df)

    res_skip = run_agent_pipeline(
        run_id="pkt_14h_calm",
        timestamp_utc="2026-04-15T09:00:00+00:00",
        latest_features=calm_row,
        settings=cfg_default,
        storage=store_pkt,
    )
    assert res_skip["alerts_payload"] == []
    assert res_skip["weather_report"] is None
    assert store_pkt.read_latest_weather_report() is None
    skip_traces = store_pkt.read_agent_graph_trace(run_id="pkt_14h_calm")
    assert skip_traces[-1]["node_name"] == "weather_report_agent"
    assert skip_traces[-1]["status"] == "skipped"
    assert skip_traces[-1]["branch_decision"] == "normal_hourly_no_alert"

    # Non-default Settings: custom timezone UTC, custom daily_report_local_hour=15, horizon=12, data_drift_window_hours=18
    cfg_custom = make_e2e_settings(
        tmp_path / "custom",
        horizon_hours=12,
        offline=True,
        lahore_timezone="UTC",
        daily_report_local_hour=15,
        data_drift_window_hours=18,
    )
    store_custom = SkyCastStorage(cfg_custom)
    ts_custom = pd.Timestamp("2026-04-15T15:00:00Z")
    calm_custom_row = dict(calm_row)
    calm_custom_row["timestamp_utc"] = ts_custom.isoformat()
    calm_df_custom = calm_df.copy()
    calm_df_custom["timestamp_utc"] = pd.date_range(end=ts_custom, periods=len(calm_df_custom), freq="1h", tz="UTC")
    # Seed EDA baseline on the full 60h dataframe, then test custom 18h drift window
    store_custom.upsert_eda_summary(compute_eda_summary(calm_df_custom, settings=cfg_custom, run_id="eda_custom_1"))
    store_custom.upsert_weather_features(calm_df_custom.tail(18))

    res_daily = run_agent_pipeline(
        run_id="custom_15utc_daily",
        timestamp_utc=ts_custom,
        latest_features=calm_custom_row,
        settings=cfg_custom,
        storage=store_custom,
    )
    assert res_daily["weather_report"] is not None
    assert res_daily["weather_report"]["trigger_reason"] == "daily_schedule"
    assert len(res_daily["forecasts_24h"]["temperature_2m"]) == 12
    assert len(store_custom.read_forecast_ledger(run_id="custom_15utc_daily")) == 12 * len(TARGET_METRICS)
    # Only 18h stored: custom data_drift_window_hours=18 runs data_drift (not check_failed!)
    data_drifts = [d for d in res_daily["drift_results"] if d["drift_type"] == "data_drift"]
    assert all(d["status"] != "check_failed" for d in data_drifts)


def test_hourly_ingest_integrates_component4_agent_pipeline(
    tmp_path: Path,
    httpx_mock: HTTPXMock,
) -> None:
    from skycast.data.sources.open_meteo import OPEN_METEO_AIR_QUALITY_FIELDS, OPEN_METEO_WEATHER_FIELDS
    from skycast.pipelines.hourly_ingest import run_hourly_ingest
    from tests.test_data_pipeline import _build_synthetic_metno_payload, _build_synthetic_open_meteo_payload

    cfg = make_e2e_settings(tmp_path / "ingest_agents", offline=True)
    store = SkyCastStorage(cfg)
    ref_hour = pd.Timestamp("2026-11-19T01:00:00Z")  # 06:00 PKT -> triggers daily report

    recent_ts = pd.date_range(ref_hour - pd.Timedelta(hours=72), ref_hour, freq="1h", tz="UTC")
    met_ts = pd.date_range(ref_hour, ref_hour + pd.Timedelta(hours=2), freq="1h", tz="UTC")
    w_p = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_WEATHER_FIELDS)
    aq_p = _build_synthetic_open_meteo_payload(recent_ts, OPEN_METEO_AIR_QUALITY_FIELDS)
    met_p = _build_synthetic_metno_payload(met_ts)

    def _router(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(cfg.open_meteo_forecast_url):
            return httpx.Response(200, json=w_p)
        if url.startswith(cfg.open_meteo_air_quality_url):
            return httpx.Response(200, json=aq_p)
        return httpx.Response(200, json=met_p)

    for _ in range(3):
        httpx_mock.add_callback(_router)

    res = run_hourly_ingest(
        lookback_hours=72,
        reference_utc=ref_hour,
        settings=cfg,
        storage=store,
        run_agents=True,
    )
    assert res["status"] == "completed"
    assert res["agents"]["status"] == "completed"
    assert len(store.read_forecast_ledger(run_id=res["run_id"])) == 24 * len(TARGET_METRICS)
    assert store.read_latest_weather_report(run_id=res["run_id"]) is not None
    traces = store.read_agent_graph_trace(run_id=res["run_id"])
    assert len(traces) == 4


def test_real_groq_and_gemini_sdk_wire_serialization_cascade(tmp_path: Path) -> None:
    """Exercise actual `groq.Groq` and `google.genai.Client` wire serialization via MockTransport."""
    import groq
    from google import genai
    from google.genai import types

    cfg = make_e2e_settings(
        tmp_path,
        offline=True,
        groq_api_key="gsk-real-sdk-test",
        gemini_api_key="AIza-real-sdk-test",
        llm_timeout_seconds=2.0,
    )
    store = SkyCastStorage(cfg)
    captured_requests: list[tuple[str, dict[str, Any]]] = []

    def _groq_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append(("groq", body))
        if body.get("model") == cfg.PRIMARY_LLM_MODEL:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(503, json={"error": {"message": "Service unavailable"}})

    def _gemini_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured_requests.append(("gemini", body))
        raise httpx.ReadTimeout("Gemini HTTP read timeout", request=request)

    real_groq_client = groq.Groq(
        api_key=cfg.groq_api_key,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_groq_transport)),
    )
    real_gemini_client = genai.Client(
        api_key=cfg.gemini_api_key,
        http_options=types.HttpOptions(
            timeout=2000,
            retry_options=types.HttpRetryOptions(attempts=1),
            httpx_client=httpx.Client(transport=httpx.MockTransport(_gemini_transport)),
        ),
    )

    client = ResilientLLMClient(
        settings=cfg,
        storage=store,
        groq_client=real_groq_client,
        gemini_client=real_gemini_client,
    )
    res = client.generate(
        run_id="real_sdk_cascade",
        agent_name="alert_agent",
        system_prompt="Return JSON",
        user_prompt="Lahore AQI 230",
        deterministic_fallback_fn=lambda: {
            "alerts": [
                {
                    "rule_id": "aqi_very_unhealthy",
                    "metric": "us_aqi",
                    "title": "Very unhealthy air quality",
                    "advisory_general": "Stay indoors.",
                    "advisory_sensitive": "Avoid outdoor air.",
                    "advisory_outdoor_workers": "Wear N95 mask.",
                }
            ]
        },
        storage=store,
    )

    assert res.used_deterministic_template is True
    assert [src for src, _ in captured_requests] == ["groq", "gemini", "groq"]
    assert captured_requests[0][1]["model"] == cfg.PRIMARY_LLM_MODEL
    assert captured_requests[0][1]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in json.dumps(captured_requests[1][1])
    assert captured_requests[2][1]["model"] == cfg.TERTIARY_LLM_MODEL
    assert captured_requests[2][1]["reasoning_effort"] == "low"

    logs = store.read_agent_call_log(run_id="real_sdk_cascade")
    assert [(r["attempt_order"], r["provider"], r["status"], r["fallback_used"]) for r in logs] == [
        (1, "groq", "failed", False),
        (2, "gemini", "failed", True),
        (3, "groq", "failed", True),
    ]



def test_multi_run_plus_1h_idempotency_with_trained_champions_and_contract_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_e2e_settings(
        tmp_path,
        horizon_hours=12,
        offline=False,
        train_split_ratio=0.60,
        val_split_ratio=0.20,
        test_split_ratio=0.20,
        rf_n_estimators=10,
        rf_max_depth=4,
        rf_min_samples_leaf=2,
        data_drift_window_hours=24,
        model_drift_window_hours=24,
        groq_api_key="test-groq",
        gemini_api_key="test-gemini",
    )
    store, fake_supa, fake_hw = _wire_dual_store(tmp_path, monkeypatch, cfg)

    df_full = generate_lahore_feature_dataframe(
        n_hours=210,
        horizon_hours=12,
        seed=42,
        end_utc="2026-11-19T02:00:00Z",
    )
    df_t0 = df_full.iloc[:-1].reset_index(drop=True)  # ends at 01:00Z (06:00 PKT -> daily report!)
    t0 = pd.Timestamp("2026-11-19T01:00:00Z")
    t1 = pd.Timestamp("2026-11-19T02:00:00Z")

    store.upsert_weather_features(df_t0, supabase_days=30, write_local=False)

    # Train lightweight Ridge/Naive24h champions via Component 3 so forecast_node loads real Hopsworks champions
    train_res = run_training(
        families=["Naive24h", "Ridge"],
        settings=cfg,
        storage=store,
        train_run_id="c4_train_1",
    )
    assert train_res["status"] == "completed"

    def _groq_ok(kwargs: dict[str, Any]) -> str:
        assert kwargs["reasoning_effort"] == "low"
        prompt_text = kwargs["messages"][1]["content"]
        if "triggered_hazards" in prompt_text:
            payload = json.loads(prompt_text)
            return json.dumps(
                {
                    "alerts": [
                        {
                            "rule_id": h["rule_id"],
                            "metric": h["metric"],
                            "title": f"Lahore Alert: {h['metric']}",
                            "advisory_general": "General advisory.",
                            "advisory_sensitive": "Sensitive advisory.",
                            "advisory_outdoor_workers": "Worker advisory.",
                        }
                        for h in payload["triggered_hazards"]
                    ]
                }
            )
        return json.dumps(
            {
                "headline": "Lahore Daily Broadcast",
                "summary": "Forecast updated across all 8 series.",
                "bulletin_markdown": "## Lahore Forecast\nAll 8 series updated.",
                "highlights": ["Champion models active", "24h trajectory saved"],
            }
        )

    _install_fake_llm_sdks(
        monkeypatch,
        groq_handler=_groq_ok,
        gemini_handler=lambda _k: "",
    )

    # Run 1 at T0 twice with identical run_id -> strictly idempotent upserts on forecast_ledger, weather_reports, agent_graph_trace
    row_t0 = df_t0.iloc[-1].to_dict()
    res_1a = run_agent_pipeline(
        run_id="agent_idemp_t0",
        timestamp_utc=t0,
        latest_features=row_t0,
        settings=cfg,
        storage=store,
    )
    fc_count_1a = len(store.read_forecast_ledger(run_id="agent_idemp_t0"))
    trace_count_1a = len(store.read_agent_graph_trace(run_id="agent_idemp_t0"))
    report_count_1a = len(store.read_weather_reports())

    res_1b = run_agent_pipeline(
        run_id="agent_idemp_t0",
        timestamp_utc=t0,
        latest_features=row_t0,
        settings=cfg,
        storage=store,
    )
    assert res_1a["weather_report"]["trigger_reason"] == "daily_schedule"
    assert res_1b["weather_report"]["trigger_reason"] == "daily_schedule"
    assert len(store.read_forecast_ledger(run_id="agent_idemp_t0")) == fc_count_1a == 12 * len(TARGET_METRICS)
    assert len(store.read_agent_graph_trace(run_id="agent_idemp_t0")) == trace_count_1a == 4
    assert len(store.read_weather_reports()) == report_count_1a == 1

    # Verify real Ridge champion models from Hopsworks were used for forecast generation
    assert any(
        meta["model_family"] == "Ridge" for meta in res_1a["champion_models_used"].values()
    )

    # Run 2 (+1h advance to T1 = 02:00Z = 07:00 PKT): ingest T1 observation and run agent_run_t1
    df_t1_row = df_full.tail(1)
    store.upsert_weather_features(df_t1_row, supabase_days=30, write_local=False)

    res_2 = run_agent_pipeline(
        run_id="agent_idemp_t1",
        timestamp_utc=t1,
        latest_features=df_t1_row.iloc[-1].to_dict(),
        settings=cfg,
        storage=store,
    )
    assert len(res_2["forecasts_24h"]["temperature_2m"]) == 12

    # Run 1's horizon_hour=1 forecasts targeting T1 must now have actual_value backfilled in forecast_ledger!
    run1_h1_rows = [
        r
        for r in store.read_forecast_ledger(run_id="agent_idemp_t0")
        if int(r["horizon_hour"]) == 1
    ]
    assert len(run1_h1_rows) == len(TARGET_METRICS)
    for r in run1_h1_rows:
        expected_obs = float(df_t1_row.iloc[-1][r["metric"]])
        assert r["actual_value"] == pytest.approx(expected_obs, abs=1e-5)

    # All Supabase rows pass strict JSON (allow_nan=False)
    for table_name in (
        "forecast_ledger",
        "weather_alerts",
        "weather_reports",
        "agent_call_log",
        "agent_graph_trace",
        "drift_log",
    ):
        json.dumps(fake_supa.get_rows(table_name), allow_nan=False)

    # Also confirm Hopsworks FG mirrors received forecast_ledger, agent_call_log, and drift_log
    assert not fake_hw.feature_store.get_feature_group(cfg.hopsworks_forecast_ledger_fg, 1).df.empty
    assert not fake_hw.feature_store.get_feature_group(cfg.hopsworks_agent_log_fg, 1).df.empty
    assert not fake_hw.feature_store.get_feature_group(cfg.hopsworks_drift_log_fg, 1).df.empty
