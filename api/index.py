"""Vercel Serverless FastAPI read layer for SkyCast (AgentWeather)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as date_type, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.queries import (
    ReadOnlyStore,
    ReadStoreError,
    parse_utc_datetime,
    sanitize_json_scalar,
    utc_hour_key,
)
from skycast.config.settings import (
    CORE_METRIC_GROUPS,
    METRIC_SPECS,
    MODEL_FAMILIES,
    POLLUTANT_COLUMNS,
    TARGET_METRICS,
    Settings,
    get_settings,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPORT_PATH = _REPO_ROOT / "docs" / "FINAL_REPORT.md"


def _metric_specs_payload() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "name": spec.name,
            "group": spec.group,
            "label": spec.label,
            "unit": spec.unit,
            "chart_token": spec.chart_token,
            "min_value": spec.min_value,
            "max_value": spec.max_value,
            "is_circular": spec.is_circular,
        }
        for name, spec in METRIC_SPECS.items()
    }


def _validate_metric_param(metric: str | None) -> str | None:
    if metric is not None and metric not in METRIC_SPECS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown metric '{metric}'. Expected one of {TARGET_METRICS}.",
        )
    return metric


def _validate_model_family_param(model_family: str | None) -> str | None:
    if model_family is not None and model_family not in MODEL_FAMILIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model_family '{model_family}'. Expected one of {MODEL_FAMILIES}.",
        )
    return model_family


def _build_latest_forecasts_map(
    rows: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
    max_horizon: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {m: [] for m in metrics}
    for row in rows:
        metric_name = str(row.get("metric") or "")
        if metric_name not in grouped:
            continue
        horizon = row.get("horizon_hour")
        if isinstance(horizon, int) and horizon > max_horizon:
            continue
        grouped[metric_name].append(dict(row))
    for metric_name in grouped:
        grouped[metric_name].sort(key=lambda r: int(r.get("horizon_hour") or 0))
    return grouped


def _enrich_ledger_actuals(
    store: ReadOnlyStore,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backfill missing `actual_value` entries in-memory from `weather_features_hourly`."""
    if not rows:
        return rows

    missing_targets: list[datetime] = []
    for row in rows:
        if row.get("actual_value") is not None:
            continue
        dt = parse_utc_datetime(row.get("target_timestamp_utc"))
        if dt is not None:
            missing_targets.append(dt.replace(minute=0, second=0, microsecond=0))

    if not missing_targets:
        return rows

    min_dt = min(missing_targets)
    max_dt = max(missing_targets).replace(minute=59, second=59, microsecond=999999)
    obs_cols = ",".join(["timestamp_utc", *TARGET_METRICS])
    obs_rows = store.read_weather_features_recent(
        limit_hours=None,
        start_utc=min_dt.isoformat(),
        end_utc=max_dt.isoformat(),
        columns=obs_cols,
    )
    if not obs_rows:
        return rows

    obs_lookup: dict[tuple[str, str], Any] = {}
    for obs in obs_rows:
        hour_key = utc_hour_key(obs.get("timestamp_utc"))
        if hour_key is None:
            continue
        for m in TARGET_METRICS:
            val = obs.get(m)
            if val is not None:
                obs_lookup[(hour_key, m)] = val

    if not obs_lookup:
        return rows

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("actual_value") is None:
            hour_key = utc_hour_key(item.get("target_timestamp_utc"))
            metric_name = str(item.get("metric") or "")
            if hour_key is not None and (hour_key, metric_name) in obs_lookup:
                item["actual_value"] = obs_lookup[(hour_key, metric_name)]
        enriched.append(item)
    return enriched


def _filter_rows_by_local_date(
    rows: Sequence[dict[str, Any]],
    *,
    ts_field: str,
    target_date: str | None,
    tz_name: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if target_date is None:
        out = list(rows)
    else:
        tz = ZoneInfo(tz_name)
        out = []
        for row in rows:
            dt = parse_utc_datetime(row.get(ts_field))
            if dt is not None and dt.astimezone(tz).date().isoformat() == target_date:
                out.append(row)
    return out[:limit] if limit is not None else out


def create_app(
    settings: Settings | None = None,
    *,
    report_path: Path | None = None,
) -> FastAPI:
    """Create and configure the SkyCast FastAPI application."""
    boot_cfg = settings or get_settings()
    app = FastAPI(
        title="SkyCast (AgentWeather) API",
        version="0.1.0",
        description="Lightweight read-only serverless API for Lahore weather & AQI forecasting.",
    )
    app.state.explicit_settings = settings
    app.state.report_path = report_path or _DEFAULT_REPORT_PATH

    app.add_middleware(
        CORSMiddleware,
        allow_origins=boot_cfg.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.exception_handler(ReadStoreError)
    async def _handle_read_store_error(_: Request, exc: ReadStoreError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=sanitize_json_scalar({"status": "error", "detail": str(exc)}),
        )

    def _get_cfg(request: Request) -> Settings:
        explicit = getattr(request.app.state, "explicit_settings", None)
        return explicit if explicit is not None else get_settings()

    def _get_store(cfg: Settings = Depends(_get_cfg)) -> ReadOnlyStore:
        return ReadOnlyStore(settings=cfg)

    @app.get("/api/health")
    def get_health(
        cfg: Settings = Depends(_get_cfg),
        store: ReadOnlyStore = Depends(_get_store),
    ) -> dict[str, Any]:
        runs = store.read_latest_pipeline_runs(limit=15)
        latest_obs = store.read_weather_features_recent(limit_hours=1)
        last_run = runs[0] if runs else None
        pipelines: dict[str, dict[str, Any] | None] = {
            "hourly_ingest": None,
            "training": None,
            "backfill": None,
        }
        for run in runs:
            ptype = str(run.get("pipeline_type") or "")
            if ptype in pipelines and pipelines[ptype] is None:
                pipelines[ptype] = run

        last_ingestion_ts = latest_obs[-1].get("timestamp_utc") if latest_obs else None
        status = "ok" if (latest_obs or last_run is not None) else "bootstrapping"
        return sanitize_json_scalar(
            {
                "status": status,
                "city": cfg.city_name,
                "timezone": cfg.lahore_timezone,
                "coordinates": {"lat": cfg.lahore_lat, "lon": cfg.lahore_lon},
                "storage_mode": store.storage_mode,
                "last_ingestion_timestamp": last_ingestion_ts,
                "last_pipeline_run": last_run,
                "pipelines": pipelines,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.get("/api/dashboard/overview")
    def get_dashboard_overview(
        cfg: Settings = Depends(_get_cfg),
        store: ReadOnlyStore = Depends(_get_store),
    ) -> dict[str, Any]:
        observed_24h = store.read_weather_features_recent(limit_hours=24)
        latest_ledger = store.read_forecast_ledger(latest_run_only=True)
        latest_report = store.read_latest_weather_report()
        active_alerts = store.read_active_alerts()

        current_conditions = dict(observed_24h[-1]) if observed_24h else None
        horizon = int(cfg.forecast_horizon_hours)
        forecasts_24h = _build_latest_forecasts_map(
            latest_ledger, metrics=TARGET_METRICS, max_horizon=horizon
        )
        has_forecasts = any(bool(pts) for pts in forecasts_24h.values())
        updated_at_utc = (
            current_conditions.get("timestamp_utc")
            if current_conditions is not None
            else (latest_ledger[0].get("issued_at_utc") if latest_ledger else None)
        )
        status = "ok" if (current_conditions is not None or has_forecasts) else "bootstrapping"

        return sanitize_json_scalar(
            {
                "status": status,
                "city": cfg.city_name,
                "timezone": cfg.lahore_timezone,
                "coordinates": {"lat": cfg.lahore_lat, "lon": cfg.lahore_lon},
                "storage_mode": store.storage_mode,
                "updated_at_utc": updated_at_utc,
                "forecast_horizon_hours": horizon,
                "current_conditions": current_conditions,
                "observed_24h": observed_24h,
                "forecasts_24h": forecasts_24h,
                "latest_report": latest_report,
                "active_alerts": active_alerts,
                "pollutant_columns": list(POLLUTANT_COLUMNS),
                "core_metric_groups": {k: list(v) for k, v in CORE_METRIC_GROUPS.items()},
                "metric_specs": _metric_specs_payload(),
            }
        )

    @app.get("/api/forecasts/history")
    def get_forecasts_history(
        metric: str | None = Query(default=None),
        model_family: str | None = Query(default=None),
        horizon_hour: int | None = Query(default=None, ge=1, le=24),
        run_id: str | None = Query(default=None),
        observed_hours: int = Query(default=48, ge=1, le=720),
        limit: int | None = Query(default=None, ge=1, le=10000),
        cfg: Settings = Depends(_get_cfg),
        store: ReadOnlyStore = Depends(_get_store),
    ) -> dict[str, Any]:
        _validate_metric_param(metric)
        _validate_model_family_param(model_family)

        history_rows = store.read_forecast_ledger(
            run_id=run_id,
            metric=metric,
            model_family=model_family,
            horizon_hour=horizon_hour,
            latest_run_only=False,
            limit=limit,
        )
        history_rows = _enrich_ledger_actuals(store, history_rows)

        latest_rows = store.read_forecast_ledger(
            run_id=run_id,
            metric=metric,
            model_family=model_family,
            latest_run_only=(run_id is None),
        )
        latest_rows = _enrich_ledger_actuals(store, latest_rows)

        observed_history = store.read_weather_features_recent(limit_hours=observed_hours)
        target_metrics = (metric,) if metric is not None else TARGET_METRICS
        horizon = int(cfg.forecast_horizon_hours)
        latest_forecast = _build_latest_forecasts_map(
            latest_rows, metrics=target_metrics, max_horizon=horizon
        )
        latest_run_id = latest_rows[0].get("run_id") if latest_rows else None
        status = "ok" if (history_rows or latest_rows or observed_history) else "bootstrapping"

        return sanitize_json_scalar(
            {
                "status": status,
                "city": cfg.city_name,
                "timezone": cfg.lahore_timezone,
                "metric": metric,
                "model_family": model_family,
                "horizon_hour": horizon_hour,
                "forecast_horizon_hours": horizon,
                "latest_run_id": latest_run_id,
                "latest_forecast": latest_forecast,
                "observed_history": observed_history,
                "history": history_rows,
                "total_count": len(history_rows),
                "metric_specs": _metric_specs_payload(),
            }
        )

    @app.get("/api/models/leaderboard")
    def get_models_leaderboard(
        metric: str | None = Query(default=None),
        train_run_id: str | None = Query(default=None),
        latest_run_only: bool = Query(default=True),
        cfg: Settings = Depends(_get_cfg),
        store: ReadOnlyStore = Depends(_get_store),
    ) -> dict[str, Any]:
        _validate_metric_param(metric)

        leaderboard = store.read_model_registry_metrics(
            train_run_id=train_run_id,
            metric=metric,
            latest_run_only=latest_run_only,
        )
        comparison_series = store.read_model_comparison_series(
            train_run_id=train_run_id,
            metric=metric,
            latest_run_only=latest_run_only,
        )
        explanation_rows = store.read_shap_lime_explanations(
            train_run_id=train_run_id,
            metric=metric,
            latest_run_only=latest_run_only,
        )

        target_metrics = (metric,) if metric is not None else TARGET_METRICS
        champions: dict[str, dict[str, Any] | None] = {m: None for m in target_metrics}
        for row in leaderboard:
            m_name = str(row.get("metric") or "")
            if m_name in champions and row.get("is_champion") and champions[m_name] is None:
                champions[m_name] = row

        explanations: dict[str, dict[str, list[dict[str, Any]]]] = {
            "shap": {"global": [], "local": []},
            "lime": {"local": []},
        }
        for exp in explanation_rows:
            method = str(exp.get("method") or "")
            scope = str(exp.get("scope") or "")
            if method in explanations and scope in explanations[method]:
                explanations[method][scope].append(exp)

        resolved_run_id = train_run_id
        if resolved_run_id is None:
            for source in (leaderboard, comparison_series, explanation_rows):
                if source and source[0].get("train_run_id"):
                    resolved_run_id = str(source[0]["train_run_id"])
                    break

        trained_at_utc = leaderboard[0].get("trained_at_utc") if leaderboard else None
        status = (
            "ok"
            if (leaderboard or comparison_series or explanation_rows)
            else "bootstrapping"
        )

        return sanitize_json_scalar(
            {
                "status": status,
                "city": cfg.city_name,
                "timezone": cfg.lahore_timezone,
                "train_run_id": resolved_run_id,
                "trained_at_utc": trained_at_utc,
                "metric": metric,
                "model_families": list(MODEL_FAMILIES),
                "leaderboard": leaderboard,
                "champions": champions,
                "comparison_series": comparison_series,
                "explanations": explanations,
                "metric_specs": _metric_specs_payload(),
            }
        )

    @app.get("/api/eda/summary")
    def get_eda_summary(
        run_id: str | None = Query(default=None),
        cfg: Settings = Depends(_get_cfg),
        store: ReadOnlyStore = Depends(_get_store),
    ) -> dict[str, Any]:
        eda_row = store.read_latest_eda_summary(run_id=run_id)
        audit_fallback = store.read_latest_imputation_audit()

        if eda_row is None:
            return sanitize_json_scalar(
                {
                    "status": "bootstrapping",
                    "run_id": None,
                    "computed_at_utc": None,
                    "sample_count": 0,
                    "date_range_start": None,
                    "date_range_end": None,
                    "metadata": {
                        "city": cfg.city_name,
                        "timezone": cfg.lahore_timezone,
                        "sample_count": 0,
                    },
                    "distributions": {},
                    "diurnal_heatmap": {
                        "timezone": cfg.lahore_timezone,
                        "hours": list(range(24)),
                        "metrics": {},
                    },
                    "correlation_matrix": {
                        "columns": [],
                        "method": "pearson",
                        "min_periods": 3,
                        "matrix": {},
                        "constant_columns": [],
                        "highlights": {},
                    },
                    "seasonal_decomposition": {},
                    "imputation_audit": audit_fallback
                    or {
                        "status": "unavailable",
                        "source": "bootstrapping",
                        "total_rows": 0,
                    },
                }
            )

        summary_json = eda_row.get("summary_json") or {}
        if not isinstance(summary_json, dict):
            summary_json = {}

        imputation_audit = summary_json.get("imputation_audit")
        if (
            not isinstance(imputation_audit, dict)
            or imputation_audit.get("status") == "unavailable"
        ) and audit_fallback is not None:
            imputation_audit = audit_fallback

        return sanitize_json_scalar(
            {
                "status": "ok",
                "run_id": eda_row.get("run_id"),
                "computed_at_utc": eda_row.get("computed_at_utc"),
                "sample_count": eda_row.get("sample_count", 0),
                "date_range_start": eda_row.get("date_range_start"),
                "date_range_end": eda_row.get("date_range_end"),
                "metadata": summary_json.get("metadata")
                or {
                    "city": cfg.city_name,
                    "timezone": cfg.lahore_timezone,
                    "sample_count": eda_row.get("sample_count", 0),
                },
                "distributions": summary_json.get("distributions") or {},
                "diurnal_heatmap": summary_json.get("diurnal_heatmap")
                or {
                    "timezone": cfg.lahore_timezone,
                    "hours": list(range(24)),
                    "metrics": {},
                },
                "correlation_matrix": summary_json.get("correlation_matrix")
                or {
                    "columns": [],
                    "method": "pearson",
                    "min_periods": 3,
                    "matrix": {},
                    "constant_columns": [],
                    "highlights": {},
                },
                "seasonal_decomposition": summary_json.get("seasonal_decomposition") or {},
                "imputation_audit": imputation_audit
                or {
                    "status": "unavailable",
                    "source": "unavailable",
                    "total_rows": eda_row.get("sample_count", 0),
                },
            }
        )

    @app.get("/api/agents/logs")
    def get_agents_logs(
        run_id: str | None = Query(default=None),
        agent_name: str | None = Query(default=None),
        provider: str | None = Query(default=None),
        status: str | None = Query(default=None),
        drift_status: str | None = Query(default=None),
        drift_type: str | None = Query(default=None),
        metric: str | None = Query(default=None),
        date: str | None = Query(default=None),
        limit: int | None = Query(default=None, ge=1, le=10000),
        cfg: Settings = Depends(_get_cfg),
        store: ReadOnlyStore = Depends(_get_store),
    ) -> dict[str, Any]:
        _validate_metric_param(metric)
        if date is not None:
            try:
                date_type.fromisoformat(date)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid date '{date}'. Expected ISO format YYYY-MM-DD.",
                ) from exc
        db_limit = None if date is not None else limit

        if status is None or status in {"success", "failed"}:
            call_logs = store.read_agent_call_log(
                run_id=run_id,
                agent_name=agent_name,
                status=status,
                provider=provider,
                limit=db_limit,
            )
        else:
            call_logs = []

        if status is None or status in {"completed", "skipped", "failed"}:
            graph_trace = store.read_agent_graph_trace(
                run_id=run_id,
                node_name=agent_name,
                status=status,
                limit=db_limit,
            )
        else:
            graph_trace = []

        resolved_drift_status = drift_status or (
            status if status in {"no_drift", "drift_detected", "check_failed"} else None
        )
        if status is None or drift_status is not None or resolved_drift_status is not None:
            drift_logs = store.read_drift_log(
                run_id=run_id,
                metric=metric,
                drift_type=drift_type,
                status=resolved_drift_status,
                limit=db_limit,
            )
        else:
            drift_logs = []

        if date is not None:
            call_logs = _filter_rows_by_local_date(
                call_logs,
                ts_field="timestamp_utc",
                target_date=date,
                tz_name=cfg.lahore_timezone,
                limit=limit,
            )
            graph_trace = _filter_rows_by_local_date(
                graph_trace,
                ts_field="started_at",
                target_date=date,
                tz_name=cfg.lahore_timezone,
                limit=limit,
            )
            drift_logs = _filter_rows_by_local_date(
                drift_logs,
                ts_field="timestamp",
                target_date=date,
                tz_name=cfg.lahore_timezone,
                limit=limit,
            )

        failed_calls = [
            row
            for row in call_logs
            if row.get("status") == "failed" or bool(row.get("fallback_used"))
        ]
        latest_run_id = None
        for source in (graph_trace, call_logs, drift_logs):
            if source and source[0].get("run_id"):
                latest_run_id = str(source[0]["run_id"])
                break

        resp_status = (
            "ok" if (call_logs or drift_logs or graph_trace) else "bootstrapping"
        )
        return sanitize_json_scalar(
            {
                "status": resp_status,
                "city": cfg.city_name,
                "timezone": cfg.lahore_timezone,
                "daily_report_local_hour": int(cfg.daily_report_local_hour),
                "llm_fallback_chain": [dict(step) for step in cfg.llm_fallback_chain],
                "call_logs": call_logs,
                "failed_calls": failed_calls,
                "drift_logs": drift_logs,
                "graph_trace": graph_trace,
                "summary": {
                    "total_calls": len(call_logs),
                    "failed_calls": sum(1 for r in call_logs if r.get("status") == "failed"),
                    "fallback_calls": sum(1 for r in call_logs if r.get("fallback_used")),
                    "drift_detected_count": sum(
                        1 for r in drift_logs if r.get("status") == "drift_detected"
                    ),
                    "drift_check_failed_count": sum(
                        1 for r in drift_logs if r.get("status") == "check_failed"
                    ),
                    "trace_steps": len(graph_trace),
                    "latest_run_id": latest_run_id,
                },
            }
        )

    @app.get("/api/report")
    def get_report(request: Request) -> dict[str, Any]:
        target_path: Path = getattr(
            request.app.state, "report_path", _DEFAULT_REPORT_PATH
        )
        if target_path.is_file():
            markdown = target_path.read_text(encoding="utf-8")
            updated_at = datetime.fromtimestamp(
                target_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            words = len(markdown.split())
            is_ready = bool(markdown.strip())
            return sanitize_json_scalar(
                {
                    "status": "ok" if is_ready else "bootstrapping",
                    "title": "SkyCast (AgentWeather) Technical Report",
                    "filename": target_path.name,
                    "path": "docs/FINAL_REPORT.md",
                    "available": is_ready,
                    "markdown": markdown,
                    "updated_at": updated_at,
                    "updated_at_utc": updated_at,
                    "word_count": words,
                    "message": None
                    if is_ready
                    else "Technical report docs/FINAL_REPORT.md is empty.",
                }
            )

        return sanitize_json_scalar(
            {
                "status": "bootstrapping",
                "title": "SkyCast (AgentWeather) Technical Report",
                "filename": target_path.name,
                "path": "docs/FINAL_REPORT.md",
                "available": False,
                "markdown": "",
                "updated_at": None,
                "updated_at_utc": None,
                "word_count": 0,
                "message": "Technical report docs/FINAL_REPORT.md has not been generated yet.",
            }
        )

    return app


app = create_app()
