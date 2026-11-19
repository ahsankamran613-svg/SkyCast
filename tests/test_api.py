"""Component 5 tests: FastAPI import isolation, cold-start resilience, >1,000-row pagination, sanitization, and non-default Settings."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.index import create_app
from api.queries import ReadOnlyStore, sanitize_json_scalar
from skycast.config.settings import MODEL_FAMILIES, TARGET_METRICS, Settings, get_settings
from skycast.data.storage import SkyCastStorage
from tests.e2e.fakes import ContractSupabaseFake
from tests.e2e.fixtures import make_e2e_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _seed_weather_rows(
    base_ts: datetime,
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(count):
        ts = base_ts + timedelta(hours=i)
        rows.append(
            {
                "timestamp_utc": _iso(ts),
                "us_aqi": 110.0 + (i % 35),
                "temperature_2m": 22.0 + (i % 10) * 0.5,
                "precipitation_mm": 0.2 if i % 5 == 0 else 0.0,
                "precipitation_prob": 15.0 + (i % 20),
                "relative_humidity_2m": 58.0 + (i % 15),
                "uv_index": 3.5 + (i % 4),
                "wind_speed_10m": 12.0 + (i % 6),
                "wind_direction_10m": float((45 + i * 10) % 360),
                "apparent_temperature": 21.5 + (i % 10) * 0.5,
                "weather_code": 2,
                "condition_symbol": "partlycloudy",
                "surface_pressure": 1012.0,
                "cloud_cover": 25.0,
                "pm2_5": 42.0 + (i % 12),
                "pm10": 85.0 + (i % 20),
                "ozone": 34.0,
                "nitrogen_dioxide": 28.0,
                "sulphur_dioxide": 8.0,
                "carbon_monoxide": 450.0,
                "source_disagreement_temp": 0.4,
                "sources_available": ["open_meteo", "met_no"],
                "is_imputed": False,
                "imputation_tier": "none",
                "imputed_columns": [],
                "wind_dir_sin": 0.707,
                "wind_dir_cos": 0.707,
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "is_weekend": False,
                "is_lahore_smog_season": True,
                "is_monsoon_season": False,
                "features_json": {"temp_lag_1h": 21.5},
            }
        )
    return rows


def _insert_weather_sqlite(db_path: Path, rows: list[dict[str, Any]]) -> None:
    now_iso = _iso(datetime.now(timezone.utc))
    cols = list(rows[0].keys())
    all_cols = [*cols, "created_at", "updated_at"]
    placeholders = ", ".join(["?"] * len(all_cols))
    sql = f"INSERT OR REPLACE INTO weather_features_hourly ({', '.join(all_cols)}) VALUES ({placeholders})"
    with sqlite3.connect(db_path) as conn:
        for row in rows:
            vals: list[Any] = []
            for c in cols:
                v = row[c]
                if c in {"sources_available", "imputed_columns", "features_json"}:
                    vals.append(json.dumps(v))
                elif c in {
                    "is_imputed",
                    "is_weekend",
                    "is_lahore_smog_season",
                    "is_monsoon_season",
                }:
                    vals.append(None if v is None else int(bool(v)))
                else:
                    vals.append(v)
            vals.extend([now_iso, now_iso])
            conn.execute(sql, vals)
        conn.commit()


def _make_forecast_rows(
    run_id: str,
    issued_at: datetime,
    *,
    horizon: int = 24,
    metrics: tuple[str, ...] = TARGET_METRICS,
    model_family: str = "Ridge",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for h in range(1, horizon + 1):
            base_val = 25.0 + h * 0.1 if metric != "wind_direction_10m" else float((90 + h * 5) % 360)
            rows.append(
                {
                    "run_id": run_id,
                    "issued_at_utc": _iso(issued_at),
                    "target_timestamp_utc": _iso(issued_at + timedelta(hours=h)),
                    "horizon_hour": h,
                    "metric": metric,
                    "model_family": model_family,
                    "model_version": "1",
                    "predicted_value": base_val,
                    "lower_bound": max(0.0, base_val - 2.0),
                    "upper_bound": base_val + 2.0,
                    "naive_baseline_value": base_val - 0.5,
                    "actual_value": None,
                }
            )
    return rows


# -----------------------------------------------------------------------------
# 1. Strict Import Boundary & vercel.json Configuration
# -----------------------------------------------------------------------------


def test_import_isolation_blocks_heavy_ml_packages(tmp_path: Path) -> None:
    """Block heavy ML, agent, numpy, and pandas packages in a fresh interpreter and import api.index."""
    script = f"""
import json
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

BLOCKED = {{
    "skycast.ml",
    "skycast.agents",
    "skycast.data",
    "skycast.features",
    "skycast.pipelines",
    "torch",
    "shap",
    "lime",
    "statsmodels",
    "sklearn",
    "scikit-learn",
    "hopsworks",
    "langgraph",
    "groq",
    "google.genai",
    "numpy",
    "pandas",
    "scipy",
}}

class _Blocker(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        for prefix in BLOCKED:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError(f"Blocked import attempted: {{fullname}}")
        return None

sys.meta_path.insert(0, _Blocker())
for mod in list(BLOCKED):
    sys.modules[mod] = None

import api.index
from fastapi.testclient import TestClient
from skycast.config.settings import Settings

cfg = Settings(
    _env_file=None,
    skycast_offline_mode="true",
    skycast_local_data_dir=Path({str(tmp_path / '.skycast_local')!r}),
)
client = TestClient(api.index.create_app(settings=cfg, report_path=Path({str(tmp_path / 'MISSING.md')!r})))
routes = [
    "/api/health",
    "/api/dashboard/overview",
    "/api/forecasts/history",
    "/api/models/leaderboard",
    "/api/eda/summary",
    "/api/agents/logs",
    "/api/report",
]
statuses = {{}}
for r in routes:
    resp = client.get(r)
    assert resp.status_code == 200, (r, resp.text)
    statuses[r] = resp.json()["status"]
print(json.dumps({{"loaded_ok": True, "statuses": statuses}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Import isolation failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    assert parsed["loaded_ok"] is True
    assert all(v == "bootstrapping" for v in parsed["statuses"].values())


def test_vercel_json_routing_and_bundling() -> None:
    vercel_path = REPO_ROOT / "vercel.json"
    assert vercel_path.is_file()
    cfg = json.loads(vercel_path.read_text(encoding="utf-8"))

    assert cfg["outputDirectory"] == "web/dist"
    assert "npm run build" in cfg["buildCommand"]

    fn_cfg = cfg["functions"]["api/index.py"]
    assert "docs/FINAL_REPORT.md" in fn_cfg["includeFiles"]
    for heavy_dir in ("skycast/ml/**", "skycast/agents/**", "skycast/data/**", "tests/**"):
        assert heavy_dir in fn_cfg["excludeFiles"]

    rewrites = cfg["rewrites"]
    assert any(
        r["source"].startswith("/api/") and r["destination"] == "/api/index.py"
        for r in rewrites
    )
    assert any(r["destination"] == "/index.html" for r in rewrites)


# -----------------------------------------------------------------------------
# 2. Cold-Start Resilience (0 Rows + Missing FINAL_REPORT.md)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("backend_mode", ["missing_sqlite_file", "empty_sqlite_tables", "empty_supabase"])
def test_cold_start_all_routes_return_200_bootstrapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_mode: str,
) -> None:
    if backend_mode == "empty_supabase":
        cfg = make_e2e_settings(
            tmp_path,
            skycast_offline_mode="false",
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="service-role-key",
        )
        fake_supa = ContractSupabaseFake(strict_domain_enums=True)
        monkeypatch.setattr(ReadOnlyStore, "_get_supabase", lambda self: fake_supa)
    else:
        cfg = make_e2e_settings(tmp_path)
        if backend_mode == "empty_sqlite_tables":
            SkyCastStorage(cfg).init_local_db()

    missing_report = tmp_path / "docs" / "FINAL_REPORT.md"
    app = create_app(settings=cfg, report_path=missing_report)
    client = TestClient(app)

    routes = [
        "/api/health",
        "/api/dashboard/overview",
        "/api/forecasts/history",
        "/api/models/leaderboard",
        "/api/eda/summary",
        "/api/agents/logs",
        "/api/report",
    ]
    payloads: dict[str, dict[str, Any]] = {}
    for route in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"{route} returned {resp.status_code}: {resp.text}"
        data = resp.json()
        json.dumps(data, allow_nan=False)
        assert data["status"] == "bootstrapping", f"{route} status was {data['status']}"
        payloads[route] = data

    health = payloads["/api/health"]
    assert health["city"] == cfg.city_name
    assert health["last_ingestion_timestamp"] is None
    assert health["last_pipeline_run"] is None
    assert set(health["pipelines"].keys()) == {"hourly_ingest", "training", "backfill"}

    overview = payloads["/api/dashboard/overview"]
    assert overview["current_conditions"] is None
    assert overview["observed_24h"] == []
    assert set(overview["forecasts_24h"].keys()) == set(TARGET_METRICS)
    assert all(overview["forecasts_24h"][m] == [] for m in TARGET_METRICS)
    assert overview["latest_report"] is None
    assert overview["active_alerts"] == []

    fc_hist = payloads["/api/forecasts/history"]
    assert fc_hist["latest_run_id"] is None
    assert fc_hist["observed_history"] == []
    assert fc_hist["history"] == []
    assert fc_hist["total_count"] == 0

    lb = payloads["/api/models/leaderboard"]
    assert lb["leaderboard"] == []
    assert lb["comparison_series"] == []
    assert lb["explanations"] == {"shap": {"global": [], "local": []}, "lime": {"local": []}}
    assert set(lb["champions"].keys()) == set(TARGET_METRICS)

    eda = payloads["/api/eda/summary"]
    assert eda["sample_count"] == 0
    assert eda["distributions"] == {}
    assert eda["diurnal_heatmap"]["hours"] == list(range(24))
    assert eda["correlation_matrix"]["columns"] == []
    assert eda["imputation_audit"]["status"] == "unavailable"

    agents = payloads["/api/agents/logs"]
    assert agents["call_logs"] == []
    assert agents["failed_calls"] == []
    assert agents["drift_logs"] == []
    assert agents["graph_trace"] == []
    assert agents["summary"]["total_calls"] == 0

    report = payloads["/api/report"]
    assert report["available"] is False
    assert report["markdown"] == ""
    assert report["updated_at"] is None


# -----------------------------------------------------------------------------
# 3. Populated DB: >1,000-Row Pagination & Non-Finite Sanitization
# -----------------------------------------------------------------------------


def test_supabase_over_1000_row_pagination_and_nan_inf_sanitization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_e2e_settings(
        tmp_path,
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
    )
    fake_supa = ContractSupabaseFake(max_rows_per_request=1000, strict_domain_enums=True)
    monkeypatch.setattr(ReadOnlyStore, "_get_supabase", lambda self: fake_supa)

    base_ts = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)

    # 1,250 rows in forecast_ledger (> 1,000 PostgREST page limit)
    ledger_rows: list[dict[str, Any]] = []
    for run_idx in range(52):
        issued = base_ts + timedelta(hours=run_idx)
        for h in range(1, 25):
            ledger_rows.append(
                {
                    "run_id": f"fc_run_{run_idx:03d}",
                    "issued_at_utc": _iso(issued),
                    "target_timestamp_utc": _iso(issued + timedelta(hours=h)),
                    "horizon_hour": h,
                    "metric": "temperature_2m",
                    "model_family": "Ridge",
                    "model_version": "1",
                    "predicted_value": 24.0 + h * 0.1,
                    "lower_bound": 22.0,
                    "upper_bound": 28.0,
                    "naive_baseline_value": 23.0,
                    "actual_value": 24.1 if run_idx < 50 else None,
                }
            )
    ledger_rows.append(
        {
            "run_id": "fc_run_052",
            "issued_at_utc": _iso(base_ts + timedelta(hours=52)),
            "target_timestamp_utc": _iso(base_ts + timedelta(hours=53)),
            "horizon_hour": 1,
            "metric": "temperature_2m",
            "model_family": "Ridge",
            "model_version": "1",
            "predicted_value": 25.0,
            "lower_bound": 23.0,
            "upper_bound": 27.0,
            "naive_baseline_value": 24.0,
            "actual_value": None,
        }
    )
    # 52 * 24 + 1 = 1249; add 1 more horizon so > 1250
    ledger_rows.append(
        {
            "run_id": "fc_run_052",
            "issued_at_utc": _iso(base_ts + timedelta(hours=52)),
            "target_timestamp_utc": _iso(base_ts + timedelta(hours=54)),
            "horizon_hour": 2,
            "metric": "temperature_2m",
            "model_family": "Ridge",
            "model_version": "1",
            "predicted_value": 25.2,
            "lower_bound": 23.0,
            "upper_bound": 27.0,
            "naive_baseline_value": 24.0,
            "actual_value": None,
        }
    )
    assert len(ledger_rows) == 1250

    # Inject raw rows (including non-finite floats directly in fake tables to test read-layer sanitization!)
    fake_supa.tables["forecast_ledger"].extend(
        {**row, "id": idx + 1, "created_at": _iso(base_ts)}
        for idx, row in enumerate(ledger_rows)
    )
    # Corrupt one optional float with NaN and one with +Inf directly in the backing table
    fake_supa.tables["forecast_ledger"][0]["naive_baseline_value"] = float("nan")
    fake_supa.tables["forecast_ledger"][1]["actual_value"] = float("inf")

    # 1,200 rows in model_comparison_series
    comp_rows = [
        {
            "id": i + 1,
            "train_run_id": "train_big_1",
            "metric": "temperature_2m",
            "timestamp_utc": _iso(base_ts + timedelta(hours=i)),
            "actual": 22.0 + (i % 7),
            "sarimax": float("nan") if i == 5 else 21.8,
            "random_forest": float("-inf") if i == 6 else 22.1,
            "ridge": 22.0,
            "lstm": 21.9,
            "naive_24h": 20.5,
            "created_at": _iso(base_ts),
        }
        for i in range(1200)
    ]
    fake_supa.tables["model_comparison_series"].extend(comp_rows)

    # 1,150 rows in agent_call_log + 1 NaN latency_ms + fallback rows
    call_rows = [
        {
            "id": i + 1,
            "run_id": f"agent_run_{i // 2}",
            "timestamp_utc": _iso(base_ts + timedelta(minutes=i)),
            "agent_name": "alert_agent",
            "provider": "groq" if i % 3 != 0 else "gemini",
            "model": "openai/gpt-oss-120b" if i % 3 != 0 else "gemini-3.5-flash-lite",
            "reasoning_effort": "low" if i % 3 != 0 else None,
            "prompt": "prompt",
            "response": None if i % 4 == 0 else "{}",
            "latency_ms": float("nan") if i == 0 else 120.0,
            "status": "failed" if i % 4 == 0 else "success",
            "fallback_used": (i % 3 == 0),
            "attempt_order": 1,
            "error_message": "Rate limited" if i % 4 == 0 else None,
            "created_at": _iso(base_ts),
        }
        for i in range(1150)
    ]
    fake_supa.tables["agent_call_log"].extend(call_rows)

    # Seed NaN/Inf inside EDA summary JSONB & weather observation
    fake_supa.tables["eda_summaries"].append(
        {
            "id": 1,
            "run_id": "eda_nan_test",
            "computed_at_utc": _iso(base_ts),
            "sample_count": 100,
            "date_range_start": _iso(base_ts),
            "date_range_end": _iso(base_ts + timedelta(hours=99)),
            "summary_json": {
                "distributions": {"temperature_2m": {"mean": float("nan"), "max": float("inf")}},
                "correlation_matrix": {"matrix": {"temperature_2m": {"us_aqi": float("-inf")}}},
            },
            "created_at": _iso(base_ts),
        }
    )

    app = create_app(settings=cfg, report_path=tmp_path / "FINAL_REPORT.md")
    client = TestClient(app)

    # 1) Forecast history pagination (>1000 rows) + NaN/Inf sanitization
    fake_supa.call_log.clear()
    fc_resp = client.get("/api/forecasts/history?metric=temperature_2m")
    assert fc_resp.status_code == 200
    fc_data = fc_resp.json()
    json.dumps(fc_data, allow_nan=False)
    assert fc_data["total_count"] == 1250
    assert len(fc_data["history"]) == 1250
    fc_ranges = [
        entry["range"]
        for entry in fake_supa.call_log
        if entry["table"] == "forecast_ledger" and entry["op"] == "select"
    ]
    assert (0, 999) in fc_ranges
    assert any(start >= 1000 for start, _ in fc_ranges)
    assert any(item["naive_baseline_value"] is None for item in fc_data["history"])
    assert any(item["actual_value"] is None for item in fc_data["history"])

    # 2) Leaderboard comparison_series pagination (>1000 rows) + NaN/-Inf sanitization
    fake_supa.call_log.clear()
    lb_resp = client.get("/api/models/leaderboard?metric=temperature_2m")
    assert lb_resp.status_code == 200
    lb_data = lb_resp.json()
    json.dumps(lb_data, allow_nan=False)
    assert len(lb_data["comparison_series"]) == 1200
    assert lb_data["comparison_series"][5]["sarimax"] is None
    assert lb_data["comparison_series"][6]["random_forest"] is None
    comp_ranges = [
        entry["range"]
        for entry in fake_supa.call_log
        if entry["table"] == "model_comparison_series" and entry["op"] == "select"
    ]
    assert (0, 999) in comp_ranges
    assert (1000, 1999) in comp_ranges

    # 3) Agent logs pagination (>1000 rows) + NaN sanitization
    fake_supa.call_log.clear()
    ag_resp = client.get("/api/agents/logs")
    assert ag_resp.status_code == 200
    ag_data = ag_resp.json()
    json.dumps(ag_data, allow_nan=False)
    assert len(ag_data["call_logs"]) == 1150
    assert ag_data["summary"]["total_calls"] == 1150
    ag_ranges = [
        entry["range"]
        for entry in fake_supa.call_log
        if entry["table"] == "agent_call_log" and entry["op"] == "select"
    ]
    assert (0, 999) in ag_ranges
    assert (1000, 1999) in ag_ranges

    # 4) EDA summary NaN/Inf sanitization
    eda_resp = client.get("/api/eda/summary")
    assert eda_resp.status_code == 200
    eda_data = eda_resp.json()
    json.dumps(eda_data, allow_nan=False)
    assert eda_data["distributions"]["temperature_2m"]["mean"] is None
    assert eda_data["distributions"]["temperature_2m"]["max"] is None
    assert eda_data["correlation_matrix"]["matrix"]["temperature_2m"]["us_aqi"] is None


def test_sqlite_non_finite_json_and_inf_sanitization(tmp_path: Path) -> None:
    cfg = make_e2e_settings(tmp_path)
    store = SkyCastStorage(cfg)
    store.init_local_db()

    ts = datetime(2026, 11, 19, 1, 0, tzinfo=timezone.utc)
    weather_rows = _seed_weather_rows(ts, 1)
    _insert_weather_sqlite(store.local_db_path, weather_rows)

    # Inject Inf into SQLite REAL and NaN/Infinity into TEXT JSON directly
    with sqlite3.connect(store.local_db_path) as conn:
        conn.execute(
            "UPDATE weather_features_hourly SET source_disagreement_temp = ?",
            (float("inf"),),
        )
        conn.execute(
            """
            INSERT INTO eda_summaries (
                run_id, computed_at_utc, sample_count, date_range_start,
                date_range_end, summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "eda_inf",
                _iso(ts),
                10,
                _iso(ts),
                _iso(ts),
                '{"distributions": {"us_aqi": {"mean": NaN, "max": Infinity}}}',
                _iso(ts),
            ),
        )
        conn.commit()

    client = TestClient(create_app(settings=cfg))
    ov = client.get("/api/dashboard/overview")
    assert ov.status_code == 200
    ov_json = ov.json()
    json.dumps(ov_json, allow_nan=False)
    assert ov_json["current_conditions"]["source_disagreement_temp"] is None

    eda = client.get("/api/eda/summary")
    assert eda.status_code == 200
    eda_json = eda.json()
    json.dumps(eda_json, allow_nan=False)
    assert eda_json["distributions"]["us_aqi"]["mean"] is None
    assert eda_json["distributions"]["us_aqi"]["max"] is None


# -----------------------------------------------------------------------------
# 4. Populated SQLite End-to-End Payload Verification, Enrichment & Filters
# -----------------------------------------------------------------------------


def test_populated_sqlite_all_endpoints_and_actual_enrichment(tmp_path: Path) -> None:
    cfg = make_e2e_settings(tmp_path)
    store = SkyCastStorage(cfg)
    store.init_local_db()

    base_ts = datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc)
    weather_rows = _seed_weather_rows(base_ts, 30)
    _insert_weather_sqlite(store.local_db_path, weather_rows)

    store.record_pipeline_run(
        run_id="hourly_run_1",
        pipeline_type="hourly_ingest",
        status="completed",
        started_at_utc=base_ts + timedelta(hours=29),
        completed_at_utc=base_ts + timedelta(hours=29, minutes=1),
        metadata={"imputation_audit": {"status": "available", "tier_1_count": 2}},
    )

    # Ledger run issued at hour 10 so target horizons 1..20 overlap observed weather_rows!
    fc_rows = _make_forecast_rows("fc_run_1", base_ts + timedelta(hours=10), horizon=24)
    store.upsert_forecast_ledger(fc_rows)

    # Model registry, comparison series, XAI, EDA
    store.upsert_model_registry_metrics(
        [
            {
                "train_run_id": "train_1",
                "trained_at_utc": _iso(base_ts),
                "metric": "temperature_2m",
                "model_family": "Ridge",
                "model_version": "1",
                "val_rmse": 1.1,
                "val_mae": 0.9,
                "val_r2": 0.91,
                "test_rmse": 1.2,
                "test_mae": 1.0,
                "test_r2": 0.89,
                "improvement_over_naive_pct": 25.0,
                "is_champion": True,
                "promoted_to_registry": True,
                "hyperparameters": {"alpha": 1.0},
                "training_samples": 500,
            },
            {
                "train_run_id": "train_1",
                "trained_at_utc": _iso(base_ts),
                "metric": "temperature_2m",
                "model_family": "Naive24h",
                "model_version": None,
                "val_rmse": 1.5,
                "val_mae": 1.2,
                "val_r2": 0.75,
                "test_rmse": 1.6,
                "test_mae": 1.3,
                "test_r2": 0.70,
                "improvement_over_naive_pct": 0.0,
                "is_champion": False,
                "promoted_to_registry": False,
                "hyperparameters": {},
                "training_samples": 500,
            },
        ]
    )
    store.upsert_model_comparison_series(
        [
            {
                "train_run_id": "train_1",
                "metric": "temperature_2m",
                "timestamp_utc": _iso(base_ts + timedelta(hours=1)),
                "actual": 22.5,
                "sarimax": 22.1,
                "random_forest": 22.4,
                "ridge": 22.6,
                "lstm": 22.3,
                "naive_24h": 21.0,
            }
        ]
    )
    store.upsert_shap_lime_explanations(
        [
            {
                "train_run_id": "train_1",
                "computed_at_utc": _iso(base_ts),
                "metric": "temperature_2m",
                "model_family": "Ridge",
                "method": "shap",
                "scope": "global",
                "base_value": 22.0,
                "predicted_value": 24.5,
                "reference_timestamp_utc": _iso(base_ts),
                "feature_attributions": [{"feature": "temp_lag_1h", "importance": 1.5}],
            },
            {
                "train_run_id": "train_1",
                "computed_at_utc": _iso(base_ts),
                "metric": "temperature_2m",
                "model_family": "Ridge",
                "method": "lime",
                "scope": "local",
                "base_value": 22.0,
                "predicted_value": 24.5,
                "reference_timestamp_utc": _iso(base_ts),
                "feature_attributions": [{"feature": "temp_lag_1h", "weight": 1.4}],
            },
        ]
    )
    store.upsert_eda_summary(
        {
            "run_id": "eda_1",
            "computed_at_utc": _iso(base_ts),
            "sample_count": 30,
            "date_range_start": _iso(base_ts),
            "date_range_end": _iso(base_ts + timedelta(hours=29)),
            "summary_json": {
                "distributions": {"temperature_2m": {"status": "available", "mean": 24.0}},
                "diurnal_heatmap": {"timezone": "Asia/Karachi", "hours": list(range(24)), "metrics": {}},
                "correlation_matrix": {"columns": ["temperature_2m"], "matrix": {"temperature_2m": {"temperature_2m": 1.0}}},
                "seasonal_decomposition": {"temperature_2m": {"status": "unavailable"}},
                "imputation_audit": {"status": "unavailable"},
            },
        }
    )

    store.insert_weather_alerts(
        [
            {
                "run_id": "fc_run_1",
                "timestamp": _iso(base_ts + timedelta(hours=10)),
                "target_timestamp_utc": _iso(base_ts + timedelta(hours=12)),
                "metric": "us_aqi",
                "severity": "warning",
                "title": "Unhealthy air quality",
                "trigger_value": 165.0,
                "threshold": 150.0,
                "advisory_general": "Limit exertion.",
                "advisory_sensitive": "Stay indoors.",
                "advisory_outdoor_workers": "Wear N95.",
                "is_active": True,
                "llm_provider": "groq",
                "llm_model": "openai/gpt-oss-120b",
            }
        ]
    )
    store.upsert_weather_report(
        {
            "run_id": "fc_run_1",
            "issued_at_utc": _iso(base_ts + timedelta(hours=10)),
            "trigger_reason": "hazard_alert",
            "headline": "Smog Alert Across Lahore",
            "summary": "AQI elevated above 150.",
            "bulletin_markdown": "## Lahore Bulletin\nStay safe.",
            "highlights": ["AQI 165"],
            "active_alert_count": 1,
            "llm_provider": "groq",
            "llm_model": "openai/gpt-oss-120b",
            "fallback_used": False,
        }
    )
    store.insert_agent_call_logs(
        [
            {
                "run_id": "fc_run_1",
                "timestamp_utc": _iso(base_ts + timedelta(hours=10)),
                "agent_name": "alert_agent",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "reasoning_effort": "low",
                "prompt": "prompt",
                "response": None,
                "latency_ms": 50.0,
                "status": "failed",
                "fallback_used": False,
                "attempt_order": 1,
                "error_message": "429",
            },
            {
                "run_id": "fc_run_1",
                "timestamp_utc": _iso(base_ts + timedelta(hours=10, seconds=1)),
                "agent_name": "alert_agent",
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
                "reasoning_effort": None,
                "prompt": "prompt",
                "response": '{"alerts": []}',
                "latency_ms": 95.0,
                "status": "success",
                "fallback_used": True,
                "attempt_order": 2,
                "error_message": None,
            },
        ]
    )
    store.upsert_agent_graph_trace(
        [
            {
                "run_id": "fc_run_1",
                "node_name": "alert_agent",
                "step_order": 2,
                "status": "completed",
                "started_at": _iso(base_ts + timedelta(hours=10)),
                "duration_ms": 150.0,
                "branch_decision": "hazards_triggered",
                "summary_json": {"triggered_count": 1},
            }
        ]
    )
    store.insert_drift_logs(
        [
            {
                "run_id": "fc_run_1",
                "timestamp": _iso(base_ts + timedelta(hours=10)),
                "metric": "temperature_2m",
                "drift_type": "model_drift",
                "test_name": "rmse_ratio",
                "score": None,
                "threshold": 1.35,
                "status": "check_failed",
                "retrain_triggered": False,
                "action_taken": "check_failed",
                "error_message": "Insufficient matched actuals.",
            }
        ]
    )

    report_file = tmp_path / "FINAL_REPORT.md"
    report_file.write_text("# SkyCast Final Report\n\nAll tests verified.", encoding="utf-8")

    client = TestClient(create_app(settings=cfg, report_path=report_file))

    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["last_ingestion_timestamp"] == weather_rows[-1]["timestamp_utc"]
    assert health["pipelines"]["hourly_ingest"]["run_id"] == "hourly_run_1"

    overview = client.get("/api/dashboard/overview").json()
    assert overview["status"] == "ok"
    assert len(overview["observed_24h"]) == 24
    # Chronological oldest -> newest
    assert overview["observed_24h"][0]["timestamp_utc"] < overview["observed_24h"][-1]["timestamp_utc"]
    assert overview["current_conditions"]["timestamp_utc"] == weather_rows[-1]["timestamp_utc"]
    assert len(overview["forecasts_24h"]["temperature_2m"]) == 24
    assert len(overview["active_alerts"]) == 1
    assert overview["latest_report"]["headline"] == "Smog Alert Across Lahore"

    fc_hist = client.get("/api/forecasts/history?metric=temperature_2m").json()
    assert fc_hist["status"] == "ok"
    assert fc_hist["total_count"] == 24
    # Horizons 1..19 overlap weather_rows (hour 11..29), so actual_value is enriched in-memory!
    enriched_h1 = next(r for r in fc_hist["history"] if r["horizon_hour"] == 1)
    assert enriched_h1["actual_value"] is not None
    # Horizon 24 is hour 34 (> hour 29), so actual_value stays None
    future_h24 = next(r for r in fc_hist["history"] if r["horizon_hour"] == 24)
    assert future_h24["actual_value"] is None

    lb = client.get("/api/models/leaderboard").json()
    assert lb["status"] == "ok"
    assert lb["champions"]["temperature_2m"]["model_family"] == "Ridge"
    assert len(lb["explanations"]["shap"]["global"]) == 1
    assert len(lb["explanations"]["lime"]["local"]) == 1

    eda = client.get("/api/eda/summary").json()
    assert eda["status"] == "ok"
    # Fallback imputation_audit pulled from completed pipeline_run when eda summary was unavailable
    assert eda["imputation_audit"]["status"] == "available"

    agents = client.get("/api/agents/logs").json()
    assert agents["status"] == "ok"
    assert len(agents["call_logs"]) == 2
    assert len(agents["failed_calls"]) == 2  # 1 failed + 1 fallback_used=True
    assert agents["drift_logs"][0]["status"] == "check_failed"
    assert len(agents["graph_trace"]) == 1

    # Filtered queries
    assert len(client.get("/api/agents/logs?status=check_failed").json()["drift_logs"]) == 1
    assert client.get("/api/agents/logs?status=check_failed").json()["call_logs"] == []
    assert len(client.get("/api/agents/logs?status=failed").json()["call_logs"]) == 1
    assert len(client.get("/api/agents/logs?provider=gemini").json()["call_logs"]) == 1
    assert len(client.get("/api/agents/logs?agent_name=alert_agent").json()["graph_trace"]) == 1
    # PKT date for 2026-11-18T10:00:00Z is 2026-11-18T15:00:00+05:00 -> 2026-11-18
    assert len(client.get("/api/agents/logs?date=2026-11-18").json()["call_logs"]) == 2
    assert client.get("/api/agents/logs?date=2026-11-19").json()["call_logs"] == []
    assert client.get("/api/agents/logs?date=invalid-date").status_code == 422

    report = client.get("/api/report").json()
    assert report["status"] == "ok"
    assert report["available"] is True
    assert "SkyCast Final Report" in report["markdown"]
    assert report["updated_at"] is not None

    # Empty FINAL_REPORT.md should still return 200 bootstrapping cleanly
    empty_report = tmp_path / "EMPTY_REPORT.md"
    empty_report.write_text("   \n", encoding="utf-8")
    empty_client = TestClient(create_app(settings=cfg, report_path=empty_report))
    empty_rep_data = empty_client.get("/api/report").json()
    assert empty_rep_data["status"] == "bootstrapping"
    assert empty_rep_data["available"] is False


# -----------------------------------------------------------------------------
# 5. Non-Default Settings, Multi-Run +1h Idempotency & Failure Modes
# -----------------------------------------------------------------------------


def test_non_default_settings_and_multi_run_idempotency(tmp_path: Path) -> None:
    custom_cfg = Settings(
        _env_file=None,
        skycast_offline_mode="true",
        skycast_local_data_dir=tmp_path / ".skycast_local",
        city_name="LahoreCustom",
        lahore_lat=31.55,
        lahore_lon=74.31,
        lahore_timezone="UTC",
        forecast_horizon_hours=6,
        daily_report_local_hour=9,
        groq_primary_model="custom/groq-primary",
        gemini_fallback_model="custom-gemini-lite",
        groq_tertiary_model="custom-qwen-27b",
        cors_origins="https://skycast.example.com",
    )
    store = SkyCastStorage(custom_cfg)
    store.init_local_db()

    t1 = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    _insert_weather_sqlite(store.local_db_path, _seed_weather_rows(t1, 10))
    store.upsert_forecast_ledger(_make_forecast_rows("run_1", t1 + timedelta(hours=9), horizon=12))
    store.insert_weather_alerts(
        [
            {
                "run_id": "run_1",
                "timestamp": _iso(t1 + timedelta(hours=9)),
                "metric": "us_aqi",
                "severity": "info",
                "title": "Moderate haze",
                "trigger_value": 120.0,
                "threshold": 100.0,
                "advisory_general": "Routine",
                "advisory_sensitive": "Caution",
                "advisory_outdoor_workers": "Mask ready",
                "is_active": True,
            }
        ]
    )

    client = TestClient(create_app(settings=custom_cfg))
    ov1 = client.get("/api/dashboard/overview").json()
    assert ov1["city"] == "LahoreCustom"
    assert ov1["timezone"] == "UTC"
    assert ov1["forecast_horizon_hours"] == 6
    # Respects configured horizon=6 even though DB has 12 horizons
    assert len(ov1["forecasts_24h"]["temperature_2m"]) == 6
    assert len(ov1["active_alerts"]) == 1

    ag1 = client.get("/api/agents/logs").json()
    assert ag1["daily_report_local_hour"] == 9
    assert [s["model"] for s in ag1["llm_fallback_chain"]] == [
        "custom/groq-primary",
        "custom-gemini-lite",
        "custom-qwen-27b",
    ]

    # +1h consecutive run clears hazard via alert_agent no_hazards_triggered and updates forecasts
    t2 = t1 + timedelta(hours=10)
    _insert_weather_sqlite(store.local_db_path, _seed_weather_rows(t2, 1))
    store.upsert_forecast_ledger(_make_forecast_rows("run_2", t2, horizon=6))
    store.upsert_agent_graph_trace(
        [
            {
                "run_id": "run_2",
                "node_name": "alert_agent",
                "step_order": 2,
                "status": "completed",
                "started_at": _iso(t2),
                "duration_ms": 10.0,
                "branch_decision": "no_hazards_triggered",
                "summary_json": {"triggered_count": 0},
            }
        ]
    )

    ov2 = client.get("/api/dashboard/overview").json()
    assert ov2["current_conditions"]["timestamp_utc"] == _iso(t2)
    assert ov2["forecasts_24h"]["temperature_2m"][0]["run_id"] == "run_2"
    assert ov2["active_alerts"] == []


def test_error_handling_unconfigured_supabase_outage_and_invalid_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unconfigured_cfg = make_e2e_settings(
        tmp_path,
        skycast_offline_mode="false",
        supabase_url="",
        supabase_service_role_key="",
        hopsworks_api_key="",
        hopsworks_project_name="",
    )
    client_unconf = TestClient(create_app(settings=unconfigured_cfg))
    assert client_unconf.get("/api/health").status_code == 503
    assert client_unconf.get("/api/dashboard/overview").status_code == 503

    outage_cfg = make_e2e_settings(
        tmp_path,
        skycast_offline_mode="false",
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="service-role-key",
    )
    failing_supa = ContractSupabaseFake(fail_reads=True)
    monkeypatch.setattr(ReadOnlyStore, "_get_supabase", lambda self: failing_supa)
    client_outage = TestClient(create_app(settings=outage_cfg))
    assert client_outage.get("/api/dashboard/overview").status_code == 503

    # Invalid metric returns 422
    offline_cfg = make_e2e_settings(tmp_path)
    client_ok = TestClient(create_app(settings=offline_cfg))
    assert client_ok.get("/api/forecasts/history?metric=not_a_metric").status_code == 422
    assert client_ok.get("/api/models/leaderboard?metric=not_a_metric").status_code == 422
    assert client_ok.get("/api/agents/logs?metric=not_a_metric").status_code == 422


def test_sanitize_json_scalar_handles_special_values() -> None:
    payload = {
        "nan": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "neg_zero": -0.0,
        "nested": [1.23456789, {"bad": float("nan")}, True, 42],
    }
    cleaned = sanitize_json_scalar(payload)
    assert cleaned["nan"] is None
    assert cleaned["pos_inf"] is None
    assert cleaned["neg_inf"] is None
    assert cleaned["neg_zero"] == 0.0
    assert cleaned["nested"] == [1.234568, {"bad": None}, True, 42]
    json.dumps(cleaned, allow_nan=False)
