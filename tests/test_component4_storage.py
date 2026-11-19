"""Component 4 tests: SkyCastStorage contracts, pagination, Hopsworks mirroring, and SQLite parity."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from skycast.config.settings import TARGET_METRICS
from skycast.data.storage import SkyCastStorage, StorageError, StorageValidationError
from tests.e2e.fakes import ContractSupabaseFake, ContractViolationError, MockHopsworksProject
from tests.e2e.fixtures import make_e2e_settings


def _sample_alert_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "agent_run_1",
        "timestamp": "2026-11-19T01:00:00+00:00",
        "target_timestamp_utc": "2026-11-19T05:00:00+00:00",
        "metric": "us_aqi",
        "severity": "critical",
        "title": "Very unhealthy smog in Lahore",
        "trigger_value": 245.5,
        "threshold": 200.0,
        "advisory_general": "Limit outdoor exertion and wear an N95 mask.",
        "advisory_sensitive": "Remain indoors with clean air.",
        "advisory_outdoor_workers": "Wear a fitted N95 respirator throughout outdoor shifts.",
        "is_active": True,
        "llm_provider": "groq",
        "llm_model": "openai/gpt-oss-120b",
    }
    row.update(overrides)
    return row


def _sample_report_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "agent_run_1",
        "issued_at_utc": "2026-11-19T01:00:00+00:00",
        "trigger_reason": "daily_schedule",
        "headline": "Lahore Morning Weather & Smog Bulletin",
        "summary": "Cool morning with elevated particulate levels.",
        "bulletin_markdown": "# Lahore Bulletin\n\nAQI remains elevated.",
        "highlights": ["Morning temperature 16C", "Peak AQI near 245"],
        "active_alert_count": 1,
        "llm_provider": "groq",
        "llm_model": "openai/gpt-oss-120b",
        "fallback_used": False,
    }
    row.update(overrides)
    return row


def _sample_call_log_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "agent_run_1",
        "timestamp_utc": "2026-11-19T01:00:01+00:00",
        "agent_name": "alert_agent",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "reasoning_effort": "low",
        "prompt": "Generate alerts JSON",
        "response": '{"alerts": []}',
        "latency_ms": 182.4,
        "status": "success",
        "fallback_used": False,
        "attempt_order": 1,
        "error_message": None,
    }
    row.update(overrides)
    return row


def _sample_graph_trace_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "agent_run_1",
        "node_name": "forecast_agent",
        "step_order": 1,
        "status": "completed",
        "started_at": "2026-11-19T01:00:00+00:00",
        "duration_ms": 42.1,
        "branch_decision": "forecasts_ready",
        "summary_json": {"metrics_forecasted": 8},
        "error_message": None,
    }
    row.update(overrides)
    return row


def _sample_drift_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": "agent_run_1",
        "timestamp": "2026-11-19T01:00:02+00:00",
        "metric": "temperature_2m",
        "drift_type": "data_drift",
        "test_name": "psi_ks_2samp",
        "score": 0.08,
        "secondary_score": 0.65,
        "threshold": 0.25,
        "status": "no_drift",
        "retrain_triggered": False,
        "action_taken": "none",
        "error_message": None,
        "details_json": {"psi": 0.08, "ks_pvalue": 0.65},
    }
    row.update(overrides)
    return row


def _wire_fake_supabase(
    store: SkyCastStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> ContractSupabaseFake:
    fake_supa = ContractSupabaseFake(strict_domain_enums=True)
    monkeypatch.setattr(store, "_get_supabase", lambda: fake_supa)
    return fake_supa


def test_component4_tables_sqlite_and_cloud_dual_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Offline SQLite roundtrip
    off_cfg = make_e2e_settings(tmp_path / "offline", offline=True)
    off_store = SkyCastStorage(off_cfg)
    assert off_store.insert_weather_alerts([_sample_alert_row()])["sqlite_rows"] == 1
    assert off_store.upsert_weather_report(_sample_report_row())["sqlite_rows"] == 1
    assert off_store.insert_agent_call_logs([_sample_call_log_row()])["sqlite_rows"] == 1
    assert off_store.upsert_agent_graph_trace([_sample_graph_trace_row()])["sqlite_rows"] == 1
    assert (
        off_store.insert_drift_logs(
            [_sample_drift_row(status="check_failed", score=None, secondary_score=None, error_message="no baseline")]
        )["sqlite_rows"]
        == 1
    )

    alerts = off_store.read_weather_alerts(latest_run_only=True)
    assert len(alerts) == 1 and alerts[0]["is_active"] is True
    report = off_store.read_latest_weather_report()
    assert report is not None and report["highlights"] == ["Morning temperature 16C", "Peak AQI near 245"]
    calls = off_store.read_agent_call_log(run_id="agent_run_1")
    assert len(calls) == 1 and calls[0]["fallback_used"] is False
    traces = off_store.read_agent_graph_trace(run_id="agent_run_1")
    assert len(traces) == 1 and traces[0]["summary_json"] == {"metrics_forecasted": 8}
    drifts = off_store.read_drift_log(status="check_failed")
    assert len(drifts) == 1 and drifts[0]["score"] is None

    # Cloud dual roundtrip (ContractSupabaseFake + MockHopsworksProject) with custom FG names
    dual_cfg = make_e2e_settings(
        tmp_path / "dual",
        offline=False,
        hopsworks_agent_log_fg="custom_agent_log_fg",
        hopsworks_drift_log_fg="custom_drift_log_fg",
        hopsworks_feature_group_version=2,
    )
    dual_store = SkyCastStorage(dual_cfg)
    fake_supa = _wire_fake_supabase(dual_store, monkeypatch)
    fake_hw = MockHopsworksProject(artifact_root=tmp_path / "hw_mr")
    monkeypatch.setattr(dual_store, "_get_hopsworks_project", lambda: fake_hw)

    assert dual_store.insert_weather_alerts([_sample_alert_row()])["supabase_rows"] == 1
    assert dual_store.upsert_weather_report(_sample_report_row())["supabase_rows"] == 1
    call_res = dual_store.insert_agent_call_logs([_sample_call_log_row()])
    assert call_res == {
        "storage_mode": "cloud_dual",
        "sqlite_rows": 0,
        "supabase_rows": 1,
        "hopsworks_rows": 1,
    }
    assert dual_store.upsert_agent_graph_trace([_sample_graph_trace_row()])["supabase_rows"] == 1
    drift_res = dual_store.insert_drift_logs([_sample_drift_row()])
    assert drift_res == {
        "storage_mode": "cloud_dual",
        "sqlite_rows": 0,
        "supabase_rows": 1,
        "hopsworks_rows": 1,
    }

    assert ("custom_agent_log_fg", 2) in fake_hw.feature_store.feature_groups
    assert ("custom_drift_log_fg", 2) in fake_hw.feature_store.feature_groups
    assert len(fake_supa.get_rows("weather_alerts")) == 1
    assert len(dual_store.read_agent_call_log(prefer_hopsworks=True)) == 1
    assert len(dual_store.read_drift_log(prefer_hopsworks=True)) == 1


@pytest.mark.parametrize("mode", ["local_offline", "cloud_supabase"])
def test_component4_check_unique_and_nan_validation_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    if mode == "local_offline":
        store = SkyCastStorage(make_e2e_settings(tmp_path / mode, offline=True))
        fake_supa = None
    else:
        store = SkyCastStorage(make_e2e_settings(tmp_path / mode, offline=False, hopsworks_api_key="", hopsworks_project_name=""))
        fake_supa = _wire_fake_supabase(store, monkeypatch)

    # weather_alerts invalid severity/metric/non-finite floats/empty strings
    for bad_alert in (
        _sample_alert_row(severity="urgent"),
        _sample_alert_row(metric="invalid_metric"),
        _sample_alert_row(trigger_value=float("nan")),
        _sample_alert_row(threshold=float("inf")),
        _sample_alert_row(advisory_outdoor_workers=" "),
    ):
        with pytest.raises(StorageValidationError):
            store.insert_weather_alerts([bad_alert])

    # weather_reports invalid trigger_reason/non-list highlights/negative active_alert_count
    for bad_report in (
        _sample_report_row(trigger_reason="cron"),
        _sample_report_row(highlights={"not": "list"}),
        _sample_report_row(active_alert_count=-1),
        _sample_report_row(headline=""),
    ):
        with pytest.raises(StorageValidationError):
            store.upsert_weather_report(bad_report)

    # agent_call_log invalid status/attempt_order/negative latency
    for bad_call in (
        _sample_call_log_row(status="timeout"),
        _sample_call_log_row(attempt_order=0),
        _sample_call_log_row(attempt_order=True),
        _sample_call_log_row(latency_ms=-5.0),
        _sample_call_log_row(latency_ms=float("nan")),
    ):
        with pytest.raises(StorageValidationError):
            store.insert_agent_call_logs([bad_call])

    # agent_graph_trace invalid status/step_order/non-dict summary/intra-batch UNIQUE duplicate
    for bad_trace in (
        _sample_graph_trace_row(status="running"),
        _sample_graph_trace_row(step_order=0),
        _sample_graph_trace_row(summary_json=["not-a-dict"]),
    ):
        with pytest.raises(StorageValidationError):
            store.upsert_agent_graph_trace([bad_trace])

    with pytest.raises(StorageValidationError):
        store.upsert_agent_graph_trace(
            [_sample_graph_trace_row(), _sample_graph_trace_row()]
        )

    # drift_log invalid drift_type/status/non-finite threshold
    for bad_drift in (
        _sample_drift_row(drift_type="concept_drift"),
        _sample_drift_row(status="unknown"),
        _sample_drift_row(threshold=float("nan")),
        _sample_drift_row(action_taken=""),
    ):
        with pytest.raises(StorageValidationError):
            store.insert_drift_logs([bad_drift])

    # Verify NaN inside JSON payloads gets sanitized cleanly and passes allow_nan=False
    if fake_supa is not None:
        assert fake_supa.call_log == []
    else:
        assert not store.local_db_path.exists()
        db_path = store.init_local_db()
        conn = sqlite3.connect(db_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO weather_alerts (run_id, timestamp, metric, severity, title, "
                    "trigger_value, threshold, advisory_general, advisory_sensitive, advisory_outdoor_workers, created_at) "
                    "VALUES ('r', '2026-11-19T00:00:00+00:00', 'us_aqi', 'urgent', 't', 1.0, 1.0, 'g', 's', 'w', '2026-11-19T00:00:00+00:00')"
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO weather_reports (run_id, issued_at_utc, trigger_reason, headline, summary, bulletin_markdown, created_at) "
                    "VALUES ('r', '2026-11-19T00:00:00+00:00', 'bad_reason', 'h', 's', 'b', '2026-11-19T00:00:00+00:00')"
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO agent_graph_trace (run_id, node_name, step_order, status, started_at, created_at) "
                    "VALUES ('r', 'n', 1, 'pending', '2026-11-19T00:00:00+00:00', '2026-11-19T00:00:00+00:00')"
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO drift_log (timestamp, metric, drift_type, threshold, status, action_taken, created_at) "
                    "VALUES ('2026-11-19T00:00:00+00:00', 'us_aqi', 'bad_type', 0.25, 'no_drift', 'none', '2026-11-19T00:00:00+00:00')"
                )
        finally:
            conn.close()


def test_component4_supabase_pagination_over_1000_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_e2e_settings(
        tmp_path,
        offline=False,
        hopsworks_api_key="",
        hopsworks_project_name="",
    )
    store = SkyCastStorage(cfg)
    fake_supa = _wire_fake_supabase(store, monkeypatch)
    base_ts = pd.Timestamp("2026-11-01T00:00:00Z")

    n_rows = 1250
    alert_rows = [
        _sample_alert_row(
            run_id=f"r_{i}",
            timestamp=(base_ts + pd.Timedelta(minutes=i)).isoformat(),
            metric=TARGET_METRICS[i % len(TARGET_METRICS)],
        )
        for i in range(n_rows)
    ]
    call_rows = [
        _sample_call_log_row(
            run_id=f"r_{i}",
            timestamp_utc=(base_ts + pd.Timedelta(minutes=i)).isoformat(),
        )
        for i in range(n_rows)
    ]
    trace_rows = [
        _sample_graph_trace_row(
            run_id=f"r_{i}",
            started_at=(base_ts + pd.Timedelta(minutes=i)).isoformat(),
        )
        for i in range(n_rows)
    ]
    drift_rows = [
        _sample_drift_row(
            run_id=f"r_{i}",
            timestamp=(base_ts + pd.Timedelta(minutes=i)).isoformat(),
        )
        for i in range(n_rows)
    ]
    report_rows = [
        _sample_report_row(
            run_id=f"r_{i}",
            issued_at_utc=(base_ts + pd.Timedelta(minutes=i)).isoformat(),
        )
        for i in range(n_rows)
    ]

    store.insert_weather_alerts(alert_rows)
    store.insert_agent_call_logs(call_rows)
    store.upsert_agent_graph_trace(trace_rows)
    store.insert_drift_logs(drift_rows)
    for r in report_rows:
        fake_supa.table("weather_reports").upsert(r, on_conflict="run_id").execute()

    assert len(store.read_weather_alerts()) == n_rows
    assert len(store.read_agent_call_log()) == n_rows
    assert len(store.read_agent_graph_trace()) == n_rows
    assert len(store.read_drift_log()) == n_rows
    assert len(store.read_weather_reports()) == n_rows

    select_calls = [c for c in fake_supa.call_log if c["op"] == "select"]
    assert any(c["range"] == (1000, 1999) for c in select_calls)


def test_component4_unconfigured_cold_start_and_forced_outages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unconf = SkyCastStorage(
        make_e2e_settings(
            tmp_path / "unconf",
            offline=False,
            supabase_url="",
            supabase_service_role_key="",
            hopsworks_api_key="",
            hopsworks_project_name="",
        )
    )
    assert unconf.storage_mode == "unconfigured"
    for fn in (
        lambda: unconf.insert_weather_alerts([_sample_alert_row()]),
        lambda: unconf.read_weather_alerts(),
        lambda: unconf.upsert_weather_report(_sample_report_row()),
        lambda: unconf.read_weather_reports(),
        lambda: unconf.read_latest_weather_report(),
        lambda: unconf.insert_agent_call_logs([_sample_call_log_row()]),
        lambda: unconf.read_agent_call_log(),
        lambda: unconf.upsert_agent_graph_trace([_sample_graph_trace_row()]),
        lambda: unconf.read_agent_graph_trace(),
        lambda: unconf.insert_drift_logs([_sample_drift_row()]),
        lambda: unconf.read_drift_log(),
    ):
        with pytest.raises(StorageError, match="unconfigured"):
            fn()

    # Cold-start empty offline SQLite DB returns clean empty results
    cold = SkyCastStorage(make_e2e_settings(tmp_path / "cold", offline=True))
    assert cold.read_weather_alerts() == []
    assert cold.read_weather_reports() == []
    assert cold.read_latest_weather_report() is None
    assert cold.read_agent_call_log() == []
    assert cold.read_agent_graph_trace() == []
    assert cold.read_drift_log() == []
    assert not cold.local_db_path.exists()

    # Forced Supabase failures raise StorageError (never silent fallbacks)
    supa_store = SkyCastStorage(
        make_e2e_settings(
            tmp_path / "supa_fail",
            offline=False,
            hopsworks_api_key="",
            hopsworks_project_name="",
        )
    )
    fake_supa = _wire_fake_supabase(supa_store, monkeypatch)
    fake_supa.set_failure(fail_reads=True, fail_writes=True)
    for fn in (
        lambda: supa_store.insert_weather_alerts([_sample_alert_row()]),
        lambda: supa_store.read_weather_alerts(),
        lambda: supa_store.upsert_weather_report(_sample_report_row()),
        lambda: supa_store.read_weather_reports(),
        lambda: supa_store.insert_agent_call_logs([_sample_call_log_row()]),
        lambda: supa_store.read_agent_call_log(),
        lambda: supa_store.upsert_agent_graph_trace([_sample_graph_trace_row()]),
        lambda: supa_store.read_agent_graph_trace(),
        lambda: supa_store.insert_drift_logs([_sample_drift_row()]),
        lambda: supa_store.read_drift_log(),
    ):
        with pytest.raises(StorageError):
            fn()

    # Forced JSON NaN in details_json gets sanitized to None (strict allow_nan=False)
    res_nan = cold.insert_drift_logs(
        [_sample_drift_row(score=float("nan"), details_json={"bad": float("inf")})]
    )
    assert res_nan["sqlite_rows"] == 1
    got = cold.read_drift_log()
    assert got[0]["score"] is None
    assert got[0]["details_json"]["bad"] is None
    json.dumps(got, allow_nan=False)
