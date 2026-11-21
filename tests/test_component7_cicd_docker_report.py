"""Component 7 tests: GitHub Actions workflows, Docker portability, local scheduler, and Final Report."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from api.index import create_app
from skycast.config.settings import TARGET_METRICS, Settings
from skycast.data.storage import SkyCastStorage
from skycast.pipelines.scheduler import bootstrap_if_needed, run_scheduler_loop

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


# -----------------------------------------------------------------------------
# 1. GitHub Actions CI/CD Workflow Contracts
# -----------------------------------------------------------------------------


def _load_workflow(filename: str) -> dict[str, Any]:
    path = WORKFLOWS_DIR / filename
    assert path.is_file(), f"Missing workflow file: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _workflow_triggers(wf: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 parses unquoted `on:` as boolean True
    triggers = wf.get("on") if "on" in wf else wf.get(True)
    assert isinstance(triggers, dict), "Workflow must define triggers under 'on:'"
    return triggers


def test_hourly_pipeline_workflow_contract() -> None:
    wf = _load_workflow("hourly_pipeline.yml")
    triggers = _workflow_triggers(wf)

    assert "workflow_dispatch" in triggers
    assert "schedule" not in triggers, (
        "hourly_pipeline.yml schedule is managed by Supabase pg_cron to avoid double-triggering"
    )

    perms = wf.get("permissions", {})
    assert perms.get("actions") == "write", (
        "hourly_pipeline.yml must grant actions:write so Drift Agent can dispatch training_pipeline.yml"
    )

    job = wf["jobs"]["hourly_ingest"]
    steps = job["steps"]
    py_step = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python@"))
    assert str(py_step["with"]["python-version"]) == "3.12"

    run_step = next(s for s in steps if "skycast.pipelines.hourly_ingest" in str(s.get("run", "")))
    env = run_step["env"]
    assert env["GROQ_REASONING_EFFORT"] == "low"
    assert env["GITHUB_TRAIN_WORKFLOW"] == "training_pipeline.yml"
    for required_env in (
        "HOPSWORKS_API_KEY",
        "HOPSWORKS_PROJECT_NAME",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPO",
    ):
        assert required_env in env


def test_training_pipeline_workflow_matches_drift_dispatch_inputs() -> None:
    """Verify training_pipeline.yml declares exact workflow_dispatch inputs sent by drift.py."""
    wf = _load_workflow("training_pipeline.yml")
    triggers = _workflow_triggers(wf)

    assert "workflow_dispatch" in triggers
    assert "schedule" not in triggers, (
        "training_pipeline.yml schedule is managed by Supabase pg_cron to avoid double-triggering"
    )

    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
    # Exact keys posted by skycast/agents/drift.py:_trigger_retraining()
    for required_input in ("trigger_reason", "run_id", "metrics"):
        assert required_input in dispatch_inputs, (
            f"training_pipeline.yml missing workflow_dispatch input '{required_input}' sent by drift.py"
        )

    job = wf["jobs"]["train"]
    steps = job["steps"]
    py_step = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python@"))
    assert str(py_step["with"]["python-version"]) == "3.12"

    train_step = next(s for s in steps if "skycast.pipelines.train" in str(s.get("run", "")))
    assert "--save-eda-json" in train_step["run"]
    env = train_step["env"]
    for required_env in (
        "HOPSWORKS_API_KEY",
        "HOPSWORKS_PROJECT_NAME",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        assert required_env in env


def test_supabase_pg_cron_dispatch_migration_contract() -> None:
    """Verify Supabase pg_cron + pg_net migration schedules both workflows at exact minutes."""
    sql_path = REPO_ROOT / "supabase" / "migrations" / "002_pg_cron_github_dispatch.sql"
    assert sql_path.exists(), "002_pg_cron_github_dispatch.sql migration missing"
    sql = sql_path.read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS pg_cron;" in sql
    assert "CREATE EXTENSION IF NOT EXISTS pg_net;" in sql
    assert "vault.decrypted_secrets" in sql
    assert "github_actions_pat" in sql
    assert "RAISE EXCEPTION" in sql
    assert "net.http_post(" in sql
    assert "'2 * * * *'" in sql
    assert "hourly_pipeline.yml" in sql
    assert "'2 1 * * *'" in sql
    assert "training_pipeline.yml" in sql
    assert '"trigger_reason": "scheduled"' in sql


def test_ci_workflow_covers_pytest_web_and_docker() -> None:
    wf = _load_workflow("ci.yml")
    triggers = _workflow_triggers(wf)
    assert "push" in triggers and "pull_request" in triggers

    jobs = wf["jobs"]
    assert "python-tests" in jobs
    assert "web-build" in jobs
    assert "docker-validate" in jobs

    web_steps = jobs["web-build"]["steps"]
    node_step = next(s for s in web_steps if str(s.get("uses", "")).startswith("actions/setup-node@"))
    assert str(node_step["with"]["node-version"]) == "24"
    build_run = "\n".join(str(s.get("run", "")) for s in web_steps)
    assert "npm run build" in build_run


# -----------------------------------------------------------------------------
# 2. Docker & Portability Contracts
# -----------------------------------------------------------------------------


def test_dockerfile_and_nginx_preserve_api_isolation_and_proxy() -> None:
    dockerfile_text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for stage in ("AS api", "AS worker", "AS web-build", "AS web"):
        assert stage in dockerfile_text

    # Isolate api stage text and confirm it does NOT install requirements-ml.txt
    api_stage = dockerfile_text.split("AS api", 1)[1].split("AS worker", 1)[0]
    assert "requirements.txt" in api_stage
    assert "requirements-ml.txt" not in api_stage
    assert "uvicorn" in api_stage
    assert "api.index:app" in api_stage

    worker_stage = dockerfile_text.split("AS worker", 1)[1].split("AS web-build", 1)[0]
    assert "requirements-ml.txt" in worker_stage
    assert "skycast.pipelines.scheduler" in worker_stage

    nginx_conf = (REPO_ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    assert "location /api/" in nginx_conf
    assert "proxy_pass http://api:8000;" in nginx_conf
    assert "try_files $uri $uri/ /index.html;" in nginx_conf

    assert (REPO_ROOT / ".dockerignore").is_file()


def test_docker_compose_structure_and_cli_validation() -> None:
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.is_file()
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = compose["services"]
    assert set(services.keys()) == {"api", "web", "scheduler"}
    assert "skycast_data" in compose["volumes"]

    # Both api and scheduler share the named local data volume and tolerate missing .env
    for svc_name in ("api", "scheduler"):
        svc = services[svc_name]
        assert "skycast_data:/app/.skycast_local" in svc["volumes"]
        env_files = svc["env_file"]
        assert any(
            isinstance(item, dict) and item.get("path") == ".env" and item.get("required") is False
            for item in env_files
        )

    assert services["web"]["build"]["target"] == "web"
    assert services["api"]["build"]["target"] == "api"
    assert services["scheduler"]["build"]["target"] == "worker"

    if shutil.which("docker"):
        proc = subprocess.run(
            ["docker", "compose", "config", "-q"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"docker compose config failed: {proc.stderr}"


# -----------------------------------------------------------------------------
# 3. Local Pipeline Scheduler (Cold Start, Idempotency, Non-Default Settings, Failures)
# -----------------------------------------------------------------------------


def _make_local_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        skycast_offline_mode="true",
        skycast_local_data_dir=tmp_path / ".skycast_local",
        **overrides,
    )


def test_scheduler_cold_start_bootstrap_and_non_default_settings(tmp_path: Path) -> None:
    """Cold start invokes backfill -> train -> ingest using cfg.backfill_default_days when unset."""
    cfg = _make_local_settings(tmp_path, backfill_default_days=17)
    store = SkyCastStorage(cfg)
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_backfill(**kwargs: Any) -> dict[str, Any]:
        calls.append(("backfill", kwargs))
        return {"run_id": "bf-1"}

    def fake_train(**kwargs: Any) -> dict[str, Any]:
        calls.append(("train", kwargs))
        return {"train_run_id": "tr-1"}

    def fake_ingest(**kwargs: Any) -> dict[str, Any]:
        calls.append(("ingest", kwargs))
        return {"run_id": "ing-1"}

    summary = bootstrap_if_needed(
        bootstrap_days=None,
        settings=cfg,
        storage=store,
        backfill_fn=fake_backfill,
        train_fn=fake_train,
        ingest_fn=fake_ingest,
    )

    assert summary["backfilled"] is True
    assert summary["trained"] is True
    assert summary["ingested"] is True
    assert [name for name, _ in calls] == ["backfill", "train", "ingest"]
    # Proves backfill_default_days comes from Settings, not a hardcoded constant
    assert calls[0][1]["days"] == 17


def test_scheduler_idempotent_bootstrap_skips_populated_stages(tmp_path: Path) -> None:
    """Populated local SQLite DB causes bootstrap_if_needed to execute zero redundant runs."""
    import pandas as pd

    cfg = _make_local_settings(tmp_path)
    store = SkyCastStorage(cfg)
    now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)

    # Seed minimal rows in weather_features_hourly, model_registry_metrics, and forecast_ledger
    store.upsert_weather_features(
        pd.DataFrame(
            [
                {
                    "timestamp_utc": now,
                    "us_aqi": 120.0,
                    "temperature_2m": 24.0,
                    "precipitation_mm": 0.0,
                    "precipitation_prob": 10.0,
                    "relative_humidity_2m": 55.0,
                    "uv_index": 3.0,
                    "wind_speed_10m": 8.0,
                    "wind_direction_10m": 180.0,
                    "is_imputed": False,
                    "imputation_tier": "none",
                    "imputed_columns": [],
                    "sources_available": ["open_meteo"],
                }
            ]
        )
    )
    store.upsert_model_registry_metrics(
        [
            {
                "train_run_id": "tr-existing",
                "trained_at_utc": now.isoformat(),
                "metric": "temperature_2m",
                "model_family": "Ridge",
                "model_version": "1",
                "val_rmse": 1.2,
                "val_mae": 0.9,
                "val_r2": 0.9,
                "test_rmse": 1.3,
                "test_mae": 1.0,
                "test_r2": 0.88,
                "improvement_over_naive_pct": 20.0,
                "is_champion": True,
                "promoted_to_registry": True,
                "hyperparameters": {},
                "training_samples": 100,
            }
        ]
    )
    store.upsert_forecast_ledger(
        [
            {
                "run_id": "fc-existing",
                "issued_at_utc": now.isoformat(),
                "target_timestamp_utc": (now + timedelta(hours=1)).isoformat(),
                "horizon_hour": 1,
                "metric": "temperature_2m",
                "model_family": "Ridge",
                "model_version": "1",
                "predicted_value": 24.5,
                "lower_bound": 23.0,
                "upper_bound": 26.0,
                "naive_baseline_value": 24.0,
                "actual_value": None,
            }
        ]
    )

    invocations: list[str] = []
    summary = bootstrap_if_needed(
        settings=cfg,
        storage=store,
        backfill_fn=lambda **_: invocations.append("backfill") or {},
        train_fn=lambda **_: invocations.append("train") or {},
        ingest_fn=lambda **_: invocations.append("ingest") or {},
    )
    assert invocations == []
    assert summary["backfilled"] is False
    assert summary["trained"] is False
    assert summary["ingested"] is False

    # Running loop with once=True executes exactly one hourly_ingest when already bootstrapped
    loop_res = run_scheduler_loop(
        once=True,
        settings=cfg,
        storage=store,
        ingest_fn=lambda **_: invocations.append("ingest_once") or {"run_id": "ing-2"},
    )
    assert invocations == ["ingest_once"]
    assert loop_res["ingest_runs"] == 1


def test_scheduler_failure_propagation_and_loop_recovery(tmp_path: Path) -> None:
    """once=True propagates errors; recurring loop logs explicit errors and recovers cleanly."""
    cfg = _make_local_settings(tmp_path)
    store = SkyCastStorage(cfg)

    with pytest.raises(ValueError, match="interval_seconds"):
        run_scheduler_loop(interval_seconds=0, settings=cfg, storage=store)

    with pytest.raises(RuntimeError, match="forced backfill failure"):
        run_scheduler_loop(
            once=True,
            settings=cfg,
            storage=store,
            backfill_fn=lambda **_: (_ for _ in ()).throw(RuntimeError("forced backfill failure")),
        )

    # Recurring mode: first tick fails, second tick succeeds and triggers stop_event
    stop_event = threading.Event()
    tick = {"count": 0}

    def flaky_ingest(**_: Any) -> dict[str, Any]:
        tick["count"] += 1
        if tick["count"] == 1:
            raise RuntimeError("transient upstream timeout")
        stop_event.set()
        return {"run_id": "ing-recovered"}

    res = run_scheduler_loop(
        skip_bootstrap=True,
        interval_seconds=1,
        train_interval_hours=24,
        once=False,
        settings=cfg,
        storage=store,
        stop_event=stop_event,
        ingest_fn=flaky_ingest,
    )
    assert tick["count"] == 2
    assert res["ingest_runs"] == 1
    assert len(res["errors"]) == 1
    assert res["errors"][0]["stage"] == "hourly_ingest"
    assert "transient upstream timeout" in res["errors"][0]["error"]


# -----------------------------------------------------------------------------
# 4. Final Technical Report & README Verification + Live /api/report Endpoint
# -----------------------------------------------------------------------------


def test_final_report_and_readme_exist_and_serve_via_api(tmp_path: Path) -> None:
    report_path = REPO_ROOT / "docs" / "FINAL_REPORT.md"
    readme_path = REPO_ROOT / "README.md"
    assert report_path.is_file(), "docs/FINAL_REPORT.md must exist"
    assert readme_path.is_file(), "README.md must exist"

    report_text = report_path.read_text(encoding="utf-8")
    report_lower = report_text.lower()
    assert len(report_text.split()) >= 800, "FINAL_REPORT.md should be comprehensive"

    # Verify all 6 required 10Pearls submission rubric sections & locked specs are covered
    for required_term in (
        "open-meteo",
        "met.no",
        "imputation",
        "eda",
        "sarimax",
        "randomforest",
        "ridge",
        "lstm",
        "naive24h",
        "shap",
        "lime",
        "langgraph",
        "openai/gpt-oss-120b",
        "gemini-3.5-flash-lite",
        "qwen-3.8-27b",
        "general public",
        "sensitive groups",
        "outdoor workers",
        "psi",
    ):
        assert required_term in report_lower, f"FINAL_REPORT.md missing required topic: {required_term}"

    # Verify all 8 target metrics are documented in the report
    for metric in TARGET_METRICS:
        assert metric in report_text, f"FINAL_REPORT.md missing metric: {metric}"

    readme_text = readme_path.read_text(encoding="utf-8")
    assert "docker compose up --build" in readme_text
    assert "docs/FINAL_REPORT.md" in readme_text

    # Verify default FastAPI /api/report serves docs/FINAL_REPORT.md cleanly
    cfg = _make_local_settings(tmp_path)
    client = TestClient(create_app(settings=cfg))
    resp = client.get("/api/report")
    assert resp.status_code == 200
    payload = resp.json()
    json.dumps(payload, allow_nan=False)
    assert payload["status"] == "ok"
    assert payload["available"] is True
    assert payload["filename"] == "FINAL_REPORT.md"
    assert payload["word_count"] >= 800
    assert payload["updated_at_utc"] is not None
    assert payload["markdown"] == report_text
