"""Shared LangGraph state definition for the 4-node SkyCast agent workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentWeatherState(TypedDict, total=False):
    """Mutable state carried across Forecast -> Alert -> Drift -> Weather Report nodes."""

    run_id: str
    timestamp_utc: str
    city: str
    latest_features: dict[str, Any]
    forecasts_24h: dict[str, list[dict[str, Any]]]
    champion_models_used: dict[str, dict[str, Any]]
    triggered_hazards: list[dict[str, Any]]
    alerts_payload: list[dict[str, Any]]
    drift_results: list[dict[str, Any]]
    retrain_triggered: bool
    weather_report: dict[str, Any] | None
    llm_call_logs: list[dict[str, Any]]
    graph_trace: list[dict[str, Any]]
    report_trigger_reason: str | None
