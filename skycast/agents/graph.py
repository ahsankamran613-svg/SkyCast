"""Stateful 4-node LangGraph workflow: Forecast -> Alert -> Drift -> Weather Report."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from skycast.agents.drift import evaluate_and_log_drift
from skycast.agents.llm_client import ResilientLLMClient
from skycast.agents.state import AgentWeatherState
from skycast.agents.templates import (
    build_deterministic_alerts,
    build_deterministic_weather_report,
    validate_llm_alerts_payload,
    validate_llm_report_payload,
)
from skycast.config.settings import (
    HAZARD_RULES,
    METRIC_SPECS,
    TARGET_METRICS,
    HazardRule,
    Settings,
    get_settings,
)
from skycast.data.storage import (
    VALID_REPORT_TRIGGER_REASONS,
    SkyCastStorage,
    _parse_json_or_default,
)
from skycast.features.engineer import sanitize_json_scalar
from skycast.ml.registry import load_champion_bundle

logger = logging.getLogger(__name__)

_SEVERITY_RANK: dict[str, int] = {"info": 1, "warning": 2, "critical": 3}


def _to_utc_hour_ts(value: datetime | pd.Timestamp | str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").floor("h")
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return ts.floor("h")


def _flatten_feature_row(raw_row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw_row)
    fj = _parse_json_or_default(row.get("features_json"), {})
    if isinstance(fj, dict):
        for k, v in fj.items():
            if k not in row or row[k] is None:
                row[k] = v
    return row


def _sanitize_ledger_bound_triplet(
    metric: str,
    raw_pred: float,
    raw_low: float,
    raw_high: float,
) -> tuple[float, float, float]:
    spec = METRIC_SPECS[metric]
    pred_clamped = spec.clamp(float(raw_pred))
    if spec.is_circular:
        low_clamped = float(np.clip(float(raw_low), 0.0, 359.9999))
        high_clamped = float(np.clip(float(raw_high), 0.0, 359.9999))
        if pred_clamped >= 360.0:
            pred_clamped = 0.0
    else:
        low_clamped = spec.clamp(float(raw_low))
        high_clamped = spec.clamp(float(raw_high))

    pred_s = float(sanitize_json_scalar(pred_clamped))
    if spec.is_circular and pred_s >= 360.0:
        pred_s = 0.0
    low_s = float(sanitize_json_scalar(low_clamped))
    high_s = float(sanitize_json_scalar(high_clamped))
    if spec.is_circular:
        if low_s >= 360.0:
            low_s = 359.9999
        if high_s >= 360.0:
            high_s = 359.9999

    low_s = min(low_s, pred_s)
    high_s = max(high_s, pred_s)
    return pred_s, low_s, high_s


def _extract_finite_origin_value(origin_row: Mapping[str, Any], metric: str) -> float | None:
    raw = origin_row.get(metric)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    clamped = METRIC_SPECS[metric].clamp(val)
    cleaned = float(sanitize_json_scalar(clamped))
    if METRIC_SPECS[metric].is_circular and cleaned >= 360.0:
        return 0.0
    return cleaned


def _has_finite_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.Series:
    if df.empty or not feature_columns:
        return pd.Series(dtype="bool")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return pd.Series(False, index=df.index, dtype="bool")
    ok = pd.Series(True, index=df.index, dtype="bool")
    for col in feature_columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s.dtype):
            continue
        num = pd.to_numeric(s, errors="coerce")
        ok &= num.notna() & np.isfinite(num.to_numpy(dtype=np.float64, na_value=np.nan))
    return ok


def _select_contiguous_context(
    history_df: pd.DataFrame,
    *,
    origin_ts: pd.Timestamp,
    feature_columns: Sequence[str],
) -> pd.DataFrame | None:
    if history_df.empty or not feature_columns:
        return None
    prior = history_df.loc[history_df["timestamp_utc"] < origin_ts].sort_values("timestamp_utc")
    if prior.empty:
        return None
    valid_mask = _has_finite_features(prior, feature_columns)
    if valid_mask.empty or not bool(valid_mask.iloc[-1]):
        return None
    # Keep trailing contiguous finite rows ending at origin_ts - 1h
    last_ts = pd.to_datetime(prior["timestamp_utc"].iloc[-1], utc=True).floor("h")
    if abs((origin_ts - last_ts).total_seconds() - 3600.0) > 60.0:
        return None
    # Trim any leading non-finite rows after the most recent invalid row
    bad_positions = np.flatnonzero(~valid_mask.to_numpy(dtype=bool))
    if len(bad_positions) > 0:
        prior = prior.iloc[int(bad_positions[-1]) + 1 :]
    return prior.reset_index(drop=True) if not prior.empty else None


def _is_more_extreme_hazard(
    rule: HazardRule,
    candidate_val: float,
    existing_val: float,
) -> bool:
    if rule.comparator in {"lt", "le"}:
        return candidate_val < existing_val
    return candidate_val > existing_val


def _evaluate_hazards(
    *,
    timestamp_utc: str,
    latest_features: Mapping[str, Any],
    forecasts_24h: Mapping[str, Sequence[Mapping[str, Any]]],
    horizon_hours: int,
) -> list[dict[str, Any]]:
    slices: list[tuple[str, dict[str, float | None]]] = []

    current_obs: dict[str, float | None] = {}
    for m in TARGET_METRICS:
        v = latest_features.get(m)
        if v is not None:
            try:
                f = float(v)
                current_obs[m] = f if math.isfinite(f) else None
            except (TypeError, ValueError):
                current_obs[m] = None
        else:
            current_obs[m] = None
    slices.append((timestamp_utc, current_obs))

    for h in range(1, horizon_hours + 1):
        obs_h: dict[str, float | None] = {}
        ts_h: str | None = None
        for m in TARGET_METRICS:
            pts = forecasts_24h.get(m, [])
            matched = next((p for p in pts if int(p["horizon_hour"]) == h), None)
            if matched is not None:
                obs_h[m] = float(matched["predicted_value"])
                if ts_h is None:
                    ts_h = str(matched["target_timestamp_utc"])
            else:
                obs_h[m] = None
        if ts_h is not None:
            slices.append((ts_h, obs_h))

    best_by_metric: dict[str, dict[str, Any]] = {}
    for slice_ts, obs in slices:
        for rule in HAZARD_RULES:
            if not rule.matches(obs):
                continue
            val = float(obs[rule.metric])  # type: ignore[arg-type]
            cand_rank = _SEVERITY_RANK[rule.severity]
            existing = best_by_metric.get(rule.metric)
            should_replace = False
            if existing is None:
                should_replace = True
            else:
                ex_rank = _SEVERITY_RANK[str(existing["severity"])]
                if cand_rank > ex_rank:
                    should_replace = True
                elif cand_rank == ex_rank and _is_more_extreme_hazard(
                    rule, val, float(existing["trigger_value"])
                ):
                    should_replace = True

            if should_replace:
                best_by_metric[rule.metric] = {
                    "rule_id": rule.rule_id,
                    "metric": rule.metric,
                    "severity": rule.severity,
                    "default_title": rule.title,
                    "trigger_value": float(sanitize_json_scalar(val)),
                    "threshold": float(sanitize_json_scalar(rule.threshold)),
                    "target_timestamp_utc": slice_ts,
                }

    return sorted(
        best_by_metric.values(),
        key=lambda item: (-_SEVERITY_RANK[str(item["severity"])], str(item["metric"])),
    )


def _resolve_report_decision(
    state: AgentWeatherState,
    cfg: Settings,
) -> tuple[bool, str]:
    manual_reason = state.get("report_trigger_reason")
    if isinstance(manual_reason, str) and manual_reason in {"manual", "backfill_seed"}:
        return True, manual_reason

    utc_dt = pd.to_datetime(state["timestamp_utc"], utc=True).to_pydatetime()
    local_dt = utc_dt.astimezone(ZoneInfo(cfg.lahore_timezone))
    if local_dt.hour == int(cfg.daily_report_local_hour):
        return True, "daily_schedule"

    alerts = state.get("alerts_payload") or []
    if len(alerts) > 0:
        return True, "hazard_alert"

    return False, "normal_hourly_no_alert"


def _record_trace_entry(
    storage: SkyCastStorage,
    state: AgentWeatherState,
    *,
    node_name: str,
    step_order: int,
    status: str,
    started_at_iso: str,
    duration_ms: float,
    branch_decision: str | None = None,
    summary_json: Mapping[str, Any] | None = None,
    error_message: str | None = None,
) -> list[dict[str, Any]]:
    row = sanitize_json_scalar(
        {
            "run_id": str(state["run_id"]),
            "node_name": node_name,
            "step_order": int(step_order),
            "status": status,
            "started_at": started_at_iso,
            "duration_ms": max(0.0, float(duration_ms)),
            "branch_decision": branch_decision,
            "summary_json": dict(summary_json or {}),
            "error_message": error_message,
        }
    )
    storage.upsert_agent_graph_trace([row])
    existing = list(state.get("graph_trace") or [])
    existing.append(row)
    return existing


def _safe_record_failed_trace(
    storage: SkyCastStorage,
    state: AgentWeatherState,
    *,
    node_name: str,
    step_order: int,
    started_at_iso: str,
    t0: float,
    exc: Exception,
) -> None:
    try:
        _record_trace_entry(
            storage,
            state,
            node_name=node_name,
            step_order=step_order,
            status="failed",
            started_at_iso=started_at_iso,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            branch_decision="node_error",
            summary_json={"error_type": type(exc).__name__},
            error_message=str(exc) or type(exc).__name__,
        )
    except Exception:
        logger.exception("Failed recording error trace for node=%s", node_name)


def build_agent_graph(
    *,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    llm_client: ResilientLLMClient | None = None,
    retrain_dispatcher: Callable[[dict[str, Any]], bool] | None = None,
) -> Any:
    """Compile the 4-node LangGraph `StateGraph(AgentWeatherState)`."""
    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)
    client = llm_client or ResilientLLMClient(settings=cfg, storage=store)

    def forecast_node(state: AgentWeatherState) -> dict[str, Any]:
        started_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        run_id = str(state["run_id"])
        horizon_hours = int(cfg.forecast_horizon_hours)

        try:
            raw_latest = state.get("latest_features")
            if isinstance(raw_latest, Mapping) and raw_latest.get("timestamp_utc") is not None:
                origin_row = _flatten_feature_row(raw_latest)
                origin_ts = _to_utc_hour_ts(origin_row["timestamp_utc"])
                history_df = store.read_weather_features(
                    limit_hours=max(72, int(cfg.lstm_lookback_hours) + 48),
                    end_utc=origin_ts.to_pydatetime(),
                    expand_features_json=True,
                )
            else:
                history_df = store.read_weather_features(
                    limit_hours=max(72, int(cfg.lstm_lookback_hours) + 48),
                    expand_features_json=True,
                )
                if history_df.empty:
                    raise RuntimeError(
                        "Forecast Agent aborted: no weather_features_hourly rows available."
                    )
                origin_row = _flatten_feature_row(history_df.iloc[-1].to_dict())
                origin_ts = _to_utc_hour_ts(origin_row["timestamp_utc"])

            origin_row["timestamp_utc"] = origin_ts.isoformat()
            origin_df = pd.DataFrame([{**origin_row, "timestamp_utc": origin_ts}])
            if not history_df.empty:
                history_df = history_df.copy()
                history_df["timestamp_utc"] = pd.to_datetime(
                    history_df["timestamp_utc"], utc=True
                ).dt.floor("h")

            forecasts_24h: dict[str, list[dict[str, Any]]] = {}
            champion_models_used: dict[str, dict[str, Any]] = {}
            ledger_rows: list[dict[str, Any]] = []
            issued_iso = origin_ts.isoformat()

            for metric in TARGET_METRICS:
                naive_val = _extract_finite_origin_value(origin_row, metric)
                bundle = load_champion_bundle(metric, store, settings=cfg)
                used_fallback_reason: str | None = None

                if bundle is not None:
                    model, version = bundle
                    origin_ok = bool(_has_finite_features(origin_df, model.feature_columns).iloc[0])
                    if origin_ok:
                        ctx_df = _select_contiguous_context(
                            history_df,
                            origin_ts=origin_ts,
                            feature_columns=model.feature_columns,
                        )
                        fc = model.predict(origin_df, context_df=ctx_df)
                        steps = min(horizon_hours, int(fc.predicted.shape[1]))
                        model_family = model.family_name
                        model_version = version
                        pred_arr = fc.predicted[0, :steps]
                        low_arr = fc.lower[0, :steps]
                        high_arr = fc.upper[0, :steps]
                    else:
                        used_fallback_reason = "missing_origin_features_for_champion"
                else:
                    used_fallback_reason = "no_promoted_champion_in_registry"

                if used_fallback_reason is not None:
                    if naive_val is None:
                        raise RuntimeError(
                            f"Forecast Agent cannot predict '{metric}': no champion bundle and origin observation is missing."
                        )
                    model_family = "Naive24h"
                    model_version = None
                    steps = horizon_hours
                    pred_arr = np.full(steps, naive_val, dtype=np.float64)
                    low_arr = np.full(steps, naive_val, dtype=np.float64)
                    high_arr = np.full(steps, naive_val, dtype=np.float64)

                champion_models_used[metric] = {
                    "model_family": model_family,
                    "model_version": model_version,
                    "fallback_reason": used_fallback_reason,
                }

                metric_points: list[dict[str, Any]] = []
                for idx in range(steps):
                    h = idx + 1
                    target_ts = (origin_ts + timedelta(hours=h)).isoformat()
                    pred_v, low_v, high_v = _sanitize_ledger_bound_triplet(
                        metric,
                        float(pred_arr[idx]),
                        float(low_arr[idx]),
                        float(high_arr[idx]),
                    )
                    row = {
                        "run_id": run_id,
                        "issued_at_utc": issued_iso,
                        "target_timestamp_utc": target_ts,
                        "horizon_hour": h,
                        "metric": metric,
                        "model_family": model_family,
                        "model_version": model_version,
                        "predicted_value": pred_v,
                        "lower_bound": low_v,
                        "upper_bound": high_v,
                        "naive_baseline_value": naive_val,
                        "actual_value": None,
                    }
                    ledger_rows.append(row)
                    metric_points.append(row)

                forecasts_24h[metric] = metric_points

            store.upsert_forecast_ledger(ledger_rows)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            updated_trace = _record_trace_entry(
                store,
                state,
                node_name="forecast_agent",
                step_order=1,
                status="completed",
                started_at_iso=started_iso,
                duration_ms=duration_ms,
                branch_decision="forecasts_ready",
                summary_json={
                    "metrics_forecasted": len(forecasts_24h),
                    "ledger_rows_written": len(ledger_rows),
                    "horizon_hours": horizon_hours,
                    "champion_models_used": champion_models_used,
                },
            )
            return {
                "timestamp_utc": issued_iso,
                "latest_features": origin_row,
                "forecasts_24h": forecasts_24h,
                "champion_models_used": champion_models_used,
                "graph_trace": updated_trace,
            }
        except Exception as exc:
            _safe_record_failed_trace(
                store,
                state,
                node_name="forecast_agent",
                step_order=1,
                started_at_iso=started_iso,
                t0=t0,
                exc=exc,
            )
            raise

    def alert_node(state: AgentWeatherState) -> dict[str, Any]:
        started_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        run_id = str(state["run_id"])
        city = str(state.get("city") or cfg.city_name)

        try:
            hazards = _evaluate_hazards(
                timestamp_utc=str(state["timestamp_utc"]),
                latest_features=state.get("latest_features") or {},
                forecasts_24h=state.get("forecasts_24h") or {},
                horizon_hours=int(cfg.forecast_horizon_hours),
            )
            existing_calls = list(state.get("llm_call_logs") or [])

            if not hazards:
                updated_trace = _record_trace_entry(
                    store,
                    state,
                    node_name="alert_agent",
                    step_order=2,
                    status="completed",
                    started_at_iso=started_iso,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    branch_decision="no_hazards_triggered",
                    summary_json={"triggered_count": 0},
                )
                return {
                    "triggered_hazards": [],
                    "alerts_payload": [],
                    "llm_call_logs": existing_calls,
                    "graph_trace": updated_trace,
                }

            sys_prompt = (
                f"You are SkyCast Alert Agent for {city}, Pakistan. "
                "Respond ONLY with a valid JSON object containing key 'alerts': a list of objects, "
                "one per triggered hazard, each with keys 'rule_id', 'metric', 'title', "
                "'advisory_general', 'advisory_sensitive', and 'advisory_outdoor_workers'. "
                "Keep advisories concise, practical, and specific to each audience."
            )
            user_prompt = json.dumps(
                sanitize_json_scalar(
                    {
                        "city": city,
                        "timestamp_utc": state["timestamp_utc"],
                        "triggered_hazards": hazards,
                    }
                ),
                allow_nan=False,
            )
            llm_res = client.generate(
                run_id=run_id,
                agent_name="alert_agent",
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                deterministic_fallback_fn=lambda: build_deterministic_alerts(hazards, city=city),
                json_mode=True,
                validator_fn=lambda parsed: validate_llm_alerts_payload(parsed, hazards),
                storage=store,
            )

            gen_by_rule = {
                str(item["rule_id"]): item
                for item in (llm_res.parsed_json or {}).get("alerts", [])
            }
            alerts_payload: list[dict[str, Any]] = []
            for hazard in hazards:
                gen = gen_by_rule[str(hazard["rule_id"])]
                alerts_payload.append(
                    sanitize_json_scalar(
                        {
                            "run_id": run_id,
                            "timestamp": str(state["timestamp_utc"]),
                            "target_timestamp_utc": str(hazard["target_timestamp_utc"]),
                            "metric": str(hazard["metric"]),
                            "severity": str(hazard["severity"]),
                            "title": str(gen["title"]),
                            "trigger_value": float(hazard["trigger_value"]),
                            "threshold": float(hazard["threshold"]),
                            "advisory_general": str(gen["advisory_general"]),
                            "advisory_sensitive": str(gen["advisory_sensitive"]),
                            "advisory_outdoor_workers": str(gen["advisory_outdoor_workers"]),
                            "is_active": True,
                            "llm_provider": llm_res.provider,
                            "llm_model": llm_res.model,
                        }
                    )
                )

            store.insert_weather_alerts(alerts_payload)
            updated_calls = [*existing_calls, *llm_res.attempt_logs]
            updated_trace = _record_trace_entry(
                store,
                state,
                node_name="alert_agent",
                step_order=2,
                status="completed",
                started_at_iso=started_iso,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                branch_decision="hazards_triggered",
                summary_json={
                    "triggered_count": len(alerts_payload),
                    "severities": [a["severity"] for a in alerts_payload],
                    "llm_provider": llm_res.provider,
                    "llm_model": llm_res.model,
                    "fallback_used": llm_res.fallback_used,
                },
            )
            return {
                "triggered_hazards": hazards,
                "alerts_payload": alerts_payload,
                "llm_call_logs": updated_calls,
                "graph_trace": updated_trace,
            }
        except Exception as exc:
            _safe_record_failed_trace(
                store,
                state,
                node_name="alert_agent",
                step_order=2,
                started_at_iso=started_iso,
                t0=t0,
                exc=exc,
            )
            raise

    def drift_node(state: AgentWeatherState) -> dict[str, Any]:
        started_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        run_id = str(state["run_id"])

        try:
            drift_rows, retrain_triggered, drift_summary = evaluate_and_log_drift(
                run_id=run_id,
                timestamp_utc=str(state["timestamp_utc"]),
                storage=store,
                settings=cfg,
                retrain_dispatcher=retrain_dispatcher,
            )
            should_report, reason = _resolve_report_decision(state, cfg)
            branch = "generate_weather_report" if should_report else "skip_weather_report"
            updated_trace = _record_trace_entry(
                store,
                state,
                node_name="drift_agent",
                step_order=3,
                status="completed",
                started_at_iso=started_iso,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                branch_decision=branch,
                summary_json={**drift_summary, "report_candidate_reason": reason},
            )
            return {
                "drift_results": drift_rows,
                "retrain_triggered": retrain_triggered,
                "graph_trace": updated_trace,
            }
        except Exception as exc:
            _safe_record_failed_trace(
                store,
                state,
                node_name="drift_agent",
                step_order=3,
                started_at_iso=started_iso,
                t0=t0,
                exc=exc,
            )
            raise

    def weather_report_node(state: AgentWeatherState) -> dict[str, Any]:
        started_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        run_id = str(state["run_id"])

        try:
            _, trigger_reason = _resolve_report_decision(state, cfg)
            if trigger_reason not in VALID_REPORT_TRIGGER_REASONS:
                raise ValueError(f"Invalid weather_reports.trigger_reason: {trigger_reason!r}")

            shap_drivers = store.read_shap_lime_explanations(
                method="shap", scope="global", latest_run_only=True
            )
            latest_features = state.get("latest_features") or {}
            forecasts_24h = state.get("forecasts_24h") or {}
            alerts_payload = state.get("alerts_payload") or []
            drift_results = state.get("drift_results") or []

            compact_forecast = {
                metric: {
                    "min": min(float(p["predicted_value"]) for p in pts),
                    "max": max(float(p["predicted_value"]) for p in pts),
                    "t_end": float(pts[-1]["predicted_value"]),
                    "model_family": pts[-1]["model_family"],
                }
                for metric, pts in forecasts_24h.items()
                if pts
            }
            compact_shap = [
                {
                    "metric": r.get("metric"),
                    "top_features": (r.get("feature_attributions") or [])[:3],
                }
                for r in shap_drivers[:4]
            ]

            sys_prompt = (
                f"You are SkyCast Chief Meteorologist for {cfg.city_name}, Pakistan. "
                "Respond ONLY with a valid JSON object having keys 'headline' (string), "
                "'summary' (2-3 sentences string), 'bulletin_markdown' (Markdown string), "
                "and 'highlights' (list of 3-5 concise string bullets)."
            )
            user_prompt = json.dumps(
                sanitize_json_scalar(
                    {
                        "city": cfg.city_name,
                        "timestamp_utc": state["timestamp_utc"],
                        "trigger_reason": trigger_reason,
                        "current_conditions": {
                            m: latest_features.get(m) for m in TARGET_METRICS if m in latest_features
                        },
                        "forecast_summary_24h": compact_forecast,
                        "active_alerts": alerts_payload,
                        "shap_drivers": compact_shap,
                        "drift_summary": {
                            "drift_detected": [
                                d["metric"] for d in drift_results if d.get("status") == "drift_detected"
                            ],
                            "check_failed": sum(
                                1 for d in drift_results if d.get("status") == "check_failed"
                            ),
                        },
                    }
                ),
                allow_nan=False,
            )

            llm_res = client.generate(
                run_id=run_id,
                agent_name="weather_report_agent",
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                deterministic_fallback_fn=lambda: build_deterministic_weather_report(
                    timestamp_utc=str(state["timestamp_utc"]),
                    trigger_reason=trigger_reason,
                    latest_features=latest_features,
                    forecasts_24h=forecasts_24h,
                    alerts_payload=alerts_payload,
                    shap_drivers=shap_drivers,
                    drift_results=drift_results,
                    settings=cfg,
                ),
                json_mode=True,
                validator_fn=validate_llm_report_payload,
                storage=store,
            )

            parsed = llm_res.parsed_json or {}
            report_row = sanitize_json_scalar(
                {
                    "run_id": run_id,
                    "issued_at_utc": str(state["timestamp_utc"]),
                    "trigger_reason": trigger_reason,
                    "headline": str(parsed["headline"]),
                    "summary": str(parsed["summary"]),
                    "bulletin_markdown": str(parsed["bulletin_markdown"]),
                    "highlights": list(parsed["highlights"]),
                    "active_alert_count": len(alerts_payload),
                    "llm_provider": llm_res.provider,
                    "llm_model": llm_res.model,
                    "fallback_used": bool(llm_res.fallback_used),
                }
            )
            store.upsert_weather_report(report_row)

            existing_calls = list(state.get("llm_call_logs") or [])
            updated_calls = [*existing_calls, *llm_res.attempt_logs]
            updated_trace = _record_trace_entry(
                store,
                state,
                node_name="weather_report_agent",
                step_order=4,
                status="completed",
                started_at_iso=started_iso,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                branch_decision=trigger_reason,
                summary_json={
                    "trigger_reason": trigger_reason,
                    "active_alert_count": len(alerts_payload),
                    "llm_provider": llm_res.provider,
                    "llm_model": llm_res.model,
                    "fallback_used": llm_res.fallback_used,
                },
            )
            return {
                "weather_report": report_row,
                "llm_call_logs": updated_calls,
                "graph_trace": updated_trace,
            }
        except Exception as exc:
            _safe_record_failed_trace(
                store,
                state,
                node_name="weather_report_agent",
                step_order=4,
                started_at_iso=started_iso,
                t0=t0,
                exc=exc,
            )
            raise

    def skip_weather_report_node(state: AgentWeatherState) -> dict[str, Any]:
        started_iso = datetime.now(timezone.utc).isoformat()
        utc_dt = pd.to_datetime(state["timestamp_utc"], utc=True).to_pydatetime()
        local_dt = utc_dt.astimezone(ZoneInfo(cfg.lahore_timezone))
        updated_trace = _record_trace_entry(
            store,
            state,
            node_name="weather_report_agent",
            step_order=4,
            status="skipped",
            started_at_iso=started_iso,
            duration_ms=0.0,
            branch_decision="normal_hourly_no_alert",
            summary_json={
                "local_hour": int(local_dt.hour),
                "daily_report_local_hour": int(cfg.daily_report_local_hour),
                "active_alert_count": len(state.get("alerts_payload") or []),
            },
        )
        return {
            "weather_report": None,
            "graph_trace": updated_trace,
        }

    def _route_after_drift(state: AgentWeatherState) -> str:
        should_report, _ = _resolve_report_decision(state, cfg)
        return "generate" if should_report else "skip"

    workflow = StateGraph(AgentWeatherState)
    workflow.add_node("forecast_agent", forecast_node)
    workflow.add_node("alert_agent", alert_node)
    workflow.add_node("drift_agent", drift_node)
    workflow.add_node("weather_report_agent", weather_report_node)
    workflow.add_node("skip_weather_report_agent", skip_weather_report_node)

    workflow.add_edge(START, "forecast_agent")
    workflow.add_edge("forecast_agent", "alert_agent")
    workflow.add_edge("alert_agent", "drift_agent")
    workflow.add_conditional_edges(
        "drift_agent",
        _route_after_drift,
        {
            "generate": "weather_report_agent",
            "skip": "skip_weather_report_agent",
        },
    )
    workflow.add_edge("weather_report_agent", END)
    workflow.add_edge("skip_weather_report_agent", END)

    return workflow.compile()


def run_agent_pipeline(
    *,
    run_id: str | None = None,
    timestamp_utc: datetime | pd.Timestamp | str | None = None,
    latest_features: Mapping[str, Any] | None = None,
    report_trigger_reason: str | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    llm_client: ResilientLLMClient | None = None,
    retrain_dispatcher: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Execute the compiled 4-node SkyCast LangGraph agentic pipeline end-to-end."""
    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)

    raw_ts = timestamp_utc
    if raw_ts is None and isinstance(latest_features, Mapping):
        raw_ts = latest_features.get("timestamp_utc")
    ts_hour = _to_utc_hour_ts(raw_ts)

    resolved_run_id = (
        run_id.strip()
        if isinstance(run_id, str) and run_id.strip()
        else f"agent-{ts_hour.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"
    )

    initial_state: AgentWeatherState = {
        "run_id": resolved_run_id,
        "timestamp_utc": ts_hour.isoformat(),
        "city": cfg.city_name,
        "latest_features": dict(latest_features) if latest_features is not None else {},
        "forecasts_24h": {},
        "champion_models_used": {},
        "triggered_hazards": [],
        "alerts_payload": [],
        "drift_results": [],
        "retrain_triggered": False,
        "weather_report": None,
        "llm_call_logs": [],
        "graph_trace": [],
        "report_trigger_reason": report_trigger_reason,
    }

    compiled = build_agent_graph(
        settings=cfg,
        storage=store,
        llm_client=llm_client,
        retrain_dispatcher=retrain_dispatcher,
    )
    final_state = compiled.invoke(initial_state)
    return dict(final_state)
