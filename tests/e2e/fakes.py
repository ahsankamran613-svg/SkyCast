"""Contract-enforcing Supabase & Hopsworks fakes for SkyCast Component 3 E2E tests."""

from __future__ import annotations

import copy
import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from skycast.config.settings import MODEL_FAMILIES, TARGET_METRICS

_TARGET_METRICS_SET = frozenset(TARGET_METRICS)
_MODEL_FAMILIES_SET = frozenset(MODEL_FAMILIES)


class ContractViolationError(RuntimeError):
    """Raised when a Supabase payload or query violates PostgreSQL/Component-3 contracts."""


class SupabaseOutageError(RuntimeError):
    """Raised when an injected Supabase read/write outage triggers."""


class HopsworksOutageError(RuntimeError):
    """Raised when an injected Hopsworks login/FG/MR outage triggers."""


class MockRestAPIError(HopsworksOutageError):
    """Simulates Hopsworks RestAPIError (e.g., 404 model/FG not found)."""


@dataclass(frozen=True)
class ContractSupabaseResponse:
    """PostgREST-compatible response container."""

    data: list[dict[str, Any]]
    count: int | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _parse_utc_timestamp(value: Any, *, field: str = "timestamp") -> datetime:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ContractViolationError(f"Invalid TIMESTAMPTZ for '{field}': {value!r}")
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ContractViolationError(f"Empty TIMESTAMPTZ for '{field}'.")
        if stripped.endswith("Z"):
            stripped = stripped[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(stripped)
        except ValueError as exc:
            raise ContractViolationError(
                f"Invalid TIMESTAMPTZ string for '{field}': {value!r}"
            ) from exc
    else:
        raise ContractViolationError(
            f"Unsupported TIMESTAMPTZ value for '{field}': {value!r}"
        )

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _require_enum_val(row: Mapping[str, Any], field: str, allowed: frozenset[str], table: str) -> None:
    val = row.get(field)
    if not isinstance(val, str) or val not in allowed:
        raise ContractViolationError(
            f"[{table}] CHECK violation on '{field}': {val!r} not in {sorted(allowed)}"
        )


def _check_horizon_1_to_24(row: Mapping[str, Any]) -> None:
    h = row.get("horizon_hour")
    if not _is_int_not_bool(h) or not (1 <= int(h) <= 24):
        raise ContractViolationError(
            f"[forecast_ledger] CHECK violation on 'horizon_hour': {h!r} (expected integer 1..24)"
        )


def _check_forecast_ledger_domain(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "metric", _TARGET_METRICS_SET, "forecast_ledger")
    _require_enum_val(row, "model_family", _MODEL_FAMILIES_SET, "forecast_ledger")
    for col in ("predicted_value", "lower_bound", "upper_bound"):
        if not _is_finite_number(row.get(col)):
            raise ContractViolationError(
                f"[forecast_ledger] '{col}' must be finite numeric, got {row.get(col)!r}"
            )
    pred = float(row["predicted_value"])
    low = float(row["lower_bound"])
    high = float(row["upper_bound"])
    if not (low <= pred <= high):
        raise ContractViolationError(
            f"[forecast_ledger] Bounds invariant violated: lower_bound={low} <= "
            f"predicted_value={pred} <= upper_bound={high} failed"
        )
    for opt_col in ("naive_baseline_value", "actual_value"):
        if row.get(opt_col) is not None and not _is_finite_number(row.get(opt_col)):
            raise ContractViolationError(
                f"[forecast_ledger] '{opt_col}' must be finite or None, got {row.get(opt_col)!r}"
            )


def _check_model_registry_metrics_domain(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "metric", _TARGET_METRICS_SET, "model_registry_metrics")
    _require_enum_val(row, "model_family", _MODEL_FAMILIES_SET, "model_registry_metrics")
    for col in ("val_rmse", "val_mae", "test_rmse", "test_mae"):
        val = row.get(col)
        if not _is_finite_number(val) or float(val) < 0.0:
            raise ContractViolationError(
                f"[model_registry_metrics] '{col}' must be finite and >= 0, got {val!r}"
            )
    for col in ("val_r2", "test_r2", "improvement_over_naive_pct"):
        val = row.get(col)
        if not _is_finite_number(val):
            raise ContractViolationError(
                f"[model_registry_metrics] '{col}' must be finite, got {val!r}"
            )


def _check_model_comparison_series_domain(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "metric", _TARGET_METRICS_SET, "model_comparison_series")
    for col in ("actual", "sarimax", "random_forest", "ridge", "lstm", "naive_24h"):
        if row.get(col) is not None and not _is_finite_number(row.get(col)):
            raise ContractViolationError(
                f"[model_comparison_series] '{col}' must be finite or None, got {row.get(col)!r}"
            )


def _check_shap_lime_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "method", frozenset({"shap", "lime"}), "shap_lime_explanations")
    _require_enum_val(row, "scope", frozenset({"global", "local"}), "shap_lime_explanations")


def _check_shap_lime_domain(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "metric", _TARGET_METRICS_SET, "shap_lime_explanations")
    _require_enum_val(row, "model_family", _MODEL_FAMILIES_SET, "shap_lime_explanations")
    for col in ("base_value", "predicted_value"):
        if row.get(col) is not None and not _is_finite_number(row.get(col)):
            raise ContractViolationError(
                f"[shap_lime_explanations] '{col}' must be finite or None, got {row.get(col)!r}"
            )


def _check_weather_alerts_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "severity", frozenset({"info", "warning", "critical"}), "weather_alerts")


def _check_weather_alerts_domain(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "metric", _TARGET_METRICS_SET, "weather_alerts")
    for col in ("trigger_value", "threshold"):
        if not _is_finite_number(row.get(col)):
            raise ContractViolationError(
                f"[weather_alerts] '{col}' must be finite, got {row.get(col)!r}"
            )


def _check_weather_reports_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(
        row,
        "trigger_reason",
        frozenset({"daily_schedule", "hazard_alert", "manual", "backfill_seed"}),
        "weather_reports",
    )


def _check_agent_call_log_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(row, "status", frozenset({"success", "failed"}), "agent_call_log")


def _check_agent_graph_trace_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(
        row,
        "status",
        frozenset({"completed", "skipped", "failed"}),
        "agent_graph_trace",
    )


def _check_drift_log_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(
        row,
        "drift_type",
        frozenset({"data_drift", "model_drift"}),
        "drift_log",
    )
    _require_enum_val(
        row,
        "status",
        frozenset({"no_drift", "drift_detected", "check_failed"}),
        "drift_log",
    )


def _check_eda_summaries_domain(row: Mapping[str, Any]) -> None:
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ContractViolationError(f"[eda_summaries] 'run_id' must be non-empty string, got {run_id!r}")
    sample_count = row.get("sample_count")
    if not _is_int_not_bool(sample_count) or int(sample_count) < 0:
        raise ContractViolationError(
            f"[eda_summaries] 'sample_count' must be integer >= 0, got {sample_count!r}"
        )


def _check_pipeline_runs_sql(row: Mapping[str, Any]) -> None:
    _require_enum_val(
        row,
        "pipeline_type",
        frozenset({"backfill", "hourly_ingest", "training"}),
        "pipeline_runs",
    )
    _require_enum_val(
        row,
        "status",
        frozenset({"running", "completed", "failed"}),
        "pipeline_runs",
    )


def _check_pipeline_runs_domain(row: Mapping[str, Any]) -> None:
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ContractViolationError(f"[pipeline_runs] 'run_id' must be non-empty string, got {run_id!r}")


@dataclass(frozen=True)
class TableContract:
    """PostgreSQL table schema contract derived from `001_init_skycast.sql`."""

    name: str
    columns: frozenset[str]
    not_null: frozenset[str]
    defaults: Mapping[str, Any | Callable[[], Any]]
    unique_sets: tuple[tuple[str, ...], ...]
    sql_checks: tuple[Callable[[dict[str, Any]], None], ...] = ()
    domain_checks: tuple[Callable[[dict[str, Any]], None], ...] = ()
    timetz_columns: frozenset[str] = frozenset()
    identity_pk: str | None = "id"
    touch_updated_at: bool = False


TABLE_CONTRACTS: dict[str, TableContract] = {
    "weather_features_hourly": TableContract(
        name="weather_features_hourly",
        columns=frozenset(
            {
                "timestamp_utc",
                "us_aqi",
                "temperature_2m",
                "precipitation_mm",
                "precipitation_prob",
                "relative_humidity_2m",
                "uv_index",
                "wind_speed_10m",
                "wind_direction_10m",
                "apparent_temperature",
                "weather_code",
                "condition_symbol",
                "surface_pressure",
                "cloud_cover",
                "pm2_5",
                "pm10",
                "ozone",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "carbon_monoxide",
                "temp_openmeteo",
                "temp_metno",
                "humidity_openmeteo",
                "humidity_metno",
                "precip_openmeteo",
                "precip_metno",
                "wind_speed_openmeteo",
                "wind_speed_metno",
                "wind_dir_openmeteo",
                "wind_dir_metno",
                "uv_openmeteo",
                "uv_metno",
                "source_disagreement_temp",
                "sources_available",
                "is_imputed",
                "imputation_tier",
                "imputed_columns",
                "wind_dir_sin",
                "wind_dir_cos",
                "hour_sin",
                "hour_cos",
                "is_weekend",
                "is_lahore_smog_season",
                "is_monsoon_season",
                "features_json",
                "created_at",
                "updated_at",
            }
        ),
        not_null=frozenset(
            {
                "timestamp_utc",
                "sources_available",
                "is_imputed",
                "imputation_tier",
                "imputed_columns",
                "features_json",
                "created_at",
                "updated_at",
            }
        ),
        defaults={
            "sources_available": list,
            "is_imputed": False,
            "imputation_tier": "none",
            "imputed_columns": list,
            "features_json": dict,
            "created_at": _now_utc_iso,
            "updated_at": _now_utc_iso,
        },
        unique_sets=(("timestamp_utc",),),
        timetz_columns=frozenset({"timestamp_utc", "created_at", "updated_at"}),
        identity_pk=None,
        touch_updated_at=True,
    ),
    "forecast_ledger": TableContract(
        name="forecast_ledger",
        columns=frozenset(
            {
                "id",
                "run_id",
                "issued_at_utc",
                "target_timestamp_utc",
                "horizon_hour",
                "metric",
                "model_family",
                "model_version",
                "predicted_value",
                "lower_bound",
                "upper_bound",
                "naive_baseline_value",
                "actual_value",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "issued_at_utc",
                "target_timestamp_utc",
                "horizon_hour",
                "metric",
                "model_family",
                "predicted_value",
                "lower_bound",
                "upper_bound",
                "created_at",
            }
        ),
        defaults={"created_at": _now_utc_iso},
        unique_sets=(("run_id", "metric", "model_family", "horizon_hour"),),
        sql_checks=(_check_horizon_1_to_24,),
        domain_checks=(_check_forecast_ledger_domain,),
        timetz_columns=frozenset({"issued_at_utc", "target_timestamp_utc", "created_at"}),
    ),
    "model_registry_metrics": TableContract(
        name="model_registry_metrics",
        columns=frozenset(
            {
                "id",
                "train_run_id",
                "trained_at_utc",
                "metric",
                "model_family",
                "model_version",
                "val_rmse",
                "val_mae",
                "val_r2",
                "test_rmse",
                "test_mae",
                "test_r2",
                "improvement_over_naive_pct",
                "is_champion",
                "promoted_to_registry",
                "hyperparameters",
                "training_samples",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "train_run_id",
                "trained_at_utc",
                "metric",
                "model_family",
                "val_rmse",
                "val_mae",
                "val_r2",
                "test_rmse",
                "test_mae",
                "test_r2",
                "improvement_over_naive_pct",
                "is_champion",
                "promoted_to_registry",
                "hyperparameters",
                "created_at",
            }
        ),
        defaults={
            "trained_at_utc": _now_utc_iso,
            "improvement_over_naive_pct": 0.0,
            "is_champion": False,
            "promoted_to_registry": False,
            "hyperparameters": dict,
            "created_at": _now_utc_iso,
        },
        unique_sets=(("train_run_id", "metric", "model_family"),),
        domain_checks=(_check_model_registry_metrics_domain,),
        timetz_columns=frozenset({"trained_at_utc", "created_at"}),
    ),
    "model_comparison_series": TableContract(
        name="model_comparison_series",
        columns=frozenset(
            {
                "id",
                "train_run_id",
                "metric",
                "timestamp_utc",
                "actual",
                "sarimax",
                "random_forest",
                "ridge",
                "lstm",
                "naive_24h",
                "created_at",
            }
        ),
        not_null=frozenset({"id", "train_run_id", "metric", "timestamp_utc", "created_at"}),
        defaults={"created_at": _now_utc_iso},
        unique_sets=(("train_run_id", "metric", "timestamp_utc"),),
        domain_checks=(_check_model_comparison_series_domain,),
        timetz_columns=frozenset({"timestamp_utc", "created_at"}),
    ),
    "shap_lime_explanations": TableContract(
        name="shap_lime_explanations",
        columns=frozenset(
            {
                "id",
                "train_run_id",
                "computed_at_utc",
                "metric",
                "model_family",
                "method",
                "scope",
                "base_value",
                "predicted_value",
                "reference_timestamp_utc",
                "feature_attributions",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "train_run_id",
                "computed_at_utc",
                "metric",
                "model_family",
                "method",
                "scope",
                "feature_attributions",
                "created_at",
            }
        ),
        defaults={
            "computed_at_utc": _now_utc_iso,
            "feature_attributions": list,
            "created_at": _now_utc_iso,
        },
        unique_sets=(("train_run_id", "metric", "method", "scope"),),
        sql_checks=(_check_shap_lime_sql,),
        domain_checks=(_check_shap_lime_domain,),
        timetz_columns=frozenset({"computed_at_utc", "reference_timestamp_utc", "created_at"}),
    ),
    "weather_alerts": TableContract(
        name="weather_alerts",
        columns=frozenset(
            {
                "id",
                "run_id",
                "timestamp",
                "target_timestamp_utc",
                "metric",
                "severity",
                "title",
                "trigger_value",
                "threshold",
                "advisory_general",
                "advisory_sensitive",
                "advisory_outdoor_workers",
                "is_active",
                "llm_provider",
                "llm_model",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "timestamp",
                "metric",
                "severity",
                "title",
                "trigger_value",
                "threshold",
                "advisory_general",
                "advisory_sensitive",
                "advisory_outdoor_workers",
                "is_active",
                "created_at",
            }
        ),
        defaults={"timestamp": _now_utc_iso, "is_active": True, "created_at": _now_utc_iso},
        unique_sets=(),
        sql_checks=(_check_weather_alerts_sql,),
        domain_checks=(_check_weather_alerts_domain,),
        timetz_columns=frozenset({"timestamp", "target_timestamp_utc", "created_at"}),
    ),
    "weather_reports": TableContract(
        name="weather_reports",
        columns=frozenset(
            {
                "id",
                "run_id",
                "issued_at_utc",
                "trigger_reason",
                "headline",
                "summary",
                "bulletin_markdown",
                "highlights",
                "active_alert_count",
                "llm_provider",
                "llm_model",
                "fallback_used",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "issued_at_utc",
                "trigger_reason",
                "headline",
                "summary",
                "bulletin_markdown",
                "highlights",
                "active_alert_count",
                "fallback_used",
                "created_at",
            }
        ),
        defaults={
            "issued_at_utc": _now_utc_iso,
            "highlights": list,
            "active_alert_count": 0,
            "fallback_used": False,
            "created_at": _now_utc_iso,
        },
        unique_sets=(("run_id",),),
        sql_checks=(_check_weather_reports_sql,),
        timetz_columns=frozenset({"issued_at_utc", "created_at"}),
    ),
    "agent_call_log": TableContract(
        name="agent_call_log",
        columns=frozenset(
            {
                "id",
                "run_id",
                "timestamp_utc",
                "agent_name",
                "provider",
                "model",
                "reasoning_effort",
                "prompt",
                "response",
                "latency_ms",
                "status",
                "fallback_used",
                "attempt_order",
                "error_message",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "timestamp_utc",
                "agent_name",
                "provider",
                "model",
                "prompt",
                "latency_ms",
                "status",
                "fallback_used",
                "attempt_order",
                "created_at",
            }
        ),
        defaults={
            "timestamp_utc": _now_utc_iso,
            "latency_ms": 0.0,
            "fallback_used": False,
            "attempt_order": 1,
            "created_at": _now_utc_iso,
        },
        unique_sets=(),
        sql_checks=(_check_agent_call_log_sql,),
        timetz_columns=frozenset({"timestamp_utc", "created_at"}),
    ),
    "agent_graph_trace": TableContract(
        name="agent_graph_trace",
        columns=frozenset(
            {
                "id",
                "run_id",
                "node_name",
                "step_order",
                "status",
                "started_at",
                "duration_ms",
                "branch_decision",
                "summary_json",
                "error_message",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "node_name",
                "step_order",
                "status",
                "started_at",
                "duration_ms",
                "summary_json",
                "created_at",
            }
        ),
        defaults={"duration_ms": 0.0, "summary_json": dict, "created_at": _now_utc_iso},
        unique_sets=(("run_id", "node_name", "step_order"),),
        sql_checks=(_check_agent_graph_trace_sql,),
        timetz_columns=frozenset({"started_at", "created_at"}),
    ),
    "drift_log": TableContract(
        name="drift_log",
        columns=frozenset(
            {
                "id",
                "run_id",
                "timestamp",
                "metric",
                "drift_type",
                "test_name",
                "score",
                "secondary_score",
                "threshold",
                "status",
                "retrain_triggered",
                "action_taken",
                "error_message",
                "details_json",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "timestamp",
                "metric",
                "drift_type",
                "threshold",
                "status",
                "retrain_triggered",
                "action_taken",
                "details_json",
                "created_at",
            }
        ),
        defaults={
            "timestamp": _now_utc_iso,
            "retrain_triggered": False,
            "details_json": dict,
            "created_at": _now_utc_iso,
        },
        unique_sets=(),
        sql_checks=(_check_drift_log_sql,),
        timetz_columns=frozenset({"timestamp", "created_at"}),
    ),
    "eda_summaries": TableContract(
        name="eda_summaries",
        columns=frozenset(
            {
                "id",
                "run_id",
                "computed_at_utc",
                "sample_count",
                "date_range_start",
                "date_range_end",
                "summary_json",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "computed_at_utc",
                "sample_count",
                "summary_json",
                "created_at",
            }
        ),
        defaults={
            "computed_at_utc": _now_utc_iso,
            "sample_count": 0,
            "summary_json": dict,
            "created_at": _now_utc_iso,
        },
        unique_sets=(("run_id",),),
        domain_checks=(_check_eda_summaries_domain,),
        timetz_columns=frozenset(
            {"computed_at_utc", "date_range_start", "date_range_end", "created_at"}
        ),
    ),
    "pipeline_runs": TableContract(
        name="pipeline_runs",
        columns=frozenset(
            {
                "id",
                "run_id",
                "pipeline_type",
                "status",
                "started_at_utc",
                "completed_at_utc",
                "storage_mode",
                "metadata_json",
                "error_message",
                "created_at",
            }
        ),
        not_null=frozenset(
            {
                "id",
                "run_id",
                "pipeline_type",
                "status",
                "started_at_utc",
                "storage_mode",
                "metadata_json",
                "created_at",
            }
        ),
        defaults={
            "started_at_utc": _now_utc_iso,
            "storage_mode": "cloud",
            "metadata_json": dict,
            "created_at": _now_utc_iso,
        },
        unique_sets=(("run_id",),),
        sql_checks=(_check_pipeline_runs_sql,),
        domain_checks=(_check_pipeline_runs_domain,),
        timetz_columns=frozenset({"started_at_utc", "completed_at_utc", "created_at"}),
    ),
}


class ContractTableQuery:
    """Fluent PostgREST-compatible table query builder enforcing schema contracts."""

    def __init__(self, fake: ContractSupabaseFake, table_name: str) -> None:
        self._fake = fake
        self._table_name = table_name
        self._contract = TABLE_CONTRACTS[table_name]
        self._op: str | None = None
        self._select_cols: list[str] | None = None
        self._count_mode: str | None = None
        self._head_only: bool = False
        self._write_rows: list[dict[str, Any]] = []
        self._is_upsert: bool = False
        self._ignore_duplicates: bool = False
        self._on_conflict: str = ""
        self._filters: list[tuple[str, str, Any]] = []
        self._orders: list[tuple[str, bool, bool]] = []
        self._limit_val: int | None = None
        self._range_bounds: tuple[int, int] | None = None

    def select(
        self,
        *columns: str,
        count: str | None = None,
        head: bool | None = None,
    ) -> ContractTableQuery:
        self._op = "select"
        self._count_mode = count
        self._head_only = bool(head)
        parsed_cols: list[str] = []
        for col_arg in columns:
            for part in str(col_arg).split(","):
                stripped = part.strip()
                if stripped:
                    parsed_cols.append(stripped)
        if not parsed_cols or "*" in parsed_cols:
            self._select_cols = None
        else:
            unknown = [c for c in parsed_cols if c not in self._contract.columns]
            if unknown:
                raise ContractViolationError(
                    f"[{self._table_name}] Unknown column(s) in select: {unknown}"
                )
            self._select_cols = parsed_cols
        return self

    def insert(
        self,
        json_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        upsert: bool = False,
        ignore_duplicates: bool = False,
        on_conflict: str = "",
        **kwargs: Any,
    ) -> ContractTableQuery:
        del kwargs
        self._op = "write"
        self._is_upsert = bool(upsert)
        self._ignore_duplicates = bool(ignore_duplicates)
        self._on_conflict = str(on_conflict or "")
        self._write_rows = self._coerce_payload_list(json_payload)
        return self

    def upsert(
        self,
        json_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        on_conflict: str = "",
        ignore_duplicates: bool = False,
        **kwargs: Any,
    ) -> ContractTableQuery:
        del kwargs
        self._op = "write"
        self._is_upsert = True
        self._ignore_duplicates = bool(ignore_duplicates)
        self._on_conflict = str(on_conflict or "")
        self._write_rows = self._coerce_payload_list(json_payload)
        return self

    @staticmethod
    def _coerce_payload_list(
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(payload, Mapping):
            return [dict(payload)]
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            return [dict(item) for item in payload]
        raise ContractViolationError(f"Write payload must be dict or sequence of dicts, got {type(payload).__name__}")

    def _add_filter(self, op: str, column: str, value: Any) -> ContractTableQuery:
        if column not in self._contract.columns:
            raise ContractViolationError(
                f"[{self._table_name}] Unknown filter column '{column}'"
            )
        self._filters.append((op, column, value))
        return self

    def eq(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("eq", column, value)

    def neq(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("neq", column, value)

    def gt(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("gt", column, value)

    def gte(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("gte", column, value)

    def lt(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("lt", column, value)

    def lte(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("lte", column, value)

    def in_(self, column: str, values: Sequence[Any]) -> ContractTableQuery:
        return self._add_filter("in", column, list(values))

    def is_(self, column: str, value: Any) -> ContractTableQuery:
        return self._add_filter("is", column, value)

    def order(
        self,
        column: str,
        *,
        desc: bool = False,
        nullsfirst: bool = False,
    ) -> ContractTableQuery:
        if column not in self._contract.columns:
            raise ContractViolationError(
                f"[{self._table_name}] Unknown order column '{column}'"
            )
        self._orders.append((column, bool(desc), bool(nullsfirst)))
        return self

    def limit(self, count: int) -> ContractTableQuery:
        if not _is_int_not_bool(count) or count < 0:
            raise ContractViolationError(f"limit() requires non-negative integer, got {count!r}")
        self._limit_val = int(count)
        return self

    def range(self, start: int, end: int) -> ContractTableQuery:
        if not _is_int_not_bool(start) or not _is_int_not_bool(end) or start < 0:
            raise ContractViolationError(f"Invalid range({start!r}, {end!r})")
        self._range_bounds = (int(start), int(end))
        return self

    def execute(self) -> ContractSupabaseResponse:
        if self._op == "write":
            return self._fake._execute_write(
                table_name=self._table_name,
                rows=self._write_rows,
                is_upsert=self._is_upsert,
                ignore_duplicates=self._ignore_duplicates,
                on_conflict=self._on_conflict,
            )
        if self._op is None:
            self.select("*")
        return self._fake._execute_read(
            table_name=self._table_name,
            select_cols=self._select_cols,
            count_mode=self._count_mode,
            head_only=self._head_only,
            filters=self._filters,
            orders=self._orders,
            limit_val=self._limit_val,
            range_bounds=self._range_bounds,
        )


class ContractSupabaseFake:
    """In-memory PostgREST fake enforcing `001_init_skycast.sql` and 1,000-row pagination."""

    def __init__(
        self,
        *,
        max_rows_per_request: int = 1000,
        fail_reads: bool = False,
        fail_writes: bool = False,
        fail_on_tables: set[str] | None = None,
        strict_domain_enums: bool = True,
        failure_hook: Callable[[str, str], None] | None = None,
    ) -> None:
        self.max_rows_per_request = int(max_rows_per_request)
        self.fail_reads = bool(fail_reads)
        self.fail_writes = bool(fail_writes)
        self.fail_on_tables: set[str] = set(fail_on_tables or ())
        self.strict_domain_enums = bool(strict_domain_enums)
        self.failure_hook = failure_hook
        self.tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_CONTRACTS}
        self._next_ids: dict[str, int] = {name: 1 for name in TABLE_CONTRACTS}
        self.call_log: list[dict[str, Any]] = []

    def set_failure(
        self,
        *,
        fail_reads: bool | None = None,
        fail_writes: bool | None = None,
        fail_on_tables: set[str] | None = None,
    ) -> None:
        if fail_reads is not None:
            self.fail_reads = bool(fail_reads)
        if fail_writes is not None:
            self.fail_writes = bool(fail_writes)
        if fail_on_tables is not None:
            self.fail_on_tables = set(fail_on_tables)

    def table(self, name: str) -> ContractTableQuery:
        if name not in TABLE_CONTRACTS:
            raise ContractViolationError(f"Unknown Supabase table '{name}'.")
        return ContractTableQuery(self, name)

    def get_rows(self, table_name: str) -> list[dict[str, Any]]:
        if table_name not in TABLE_CONTRACTS:
            raise ContractViolationError(f"Unknown Supabase table '{table_name}'.")
        return copy.deepcopy(self.tables[table_name])

    def _check_outage(self, table_name: str, op: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(table_name, op)
        if table_name in self.fail_on_tables:
            raise SupabaseOutageError(f"Injected Supabase outage on table '{table_name}' ({op}).")
        if op == "read" and self.fail_reads:
            raise SupabaseOutageError(f"Injected Supabase read outage on '{table_name}'.")
        if op == "write" and self.fail_writes:
            raise SupabaseOutageError(f"Injected Supabase write outage on '{table_name}'.")

    def _canonical_key_value(self, contract: TableContract, col: str, val: Any) -> Any:
        if val is None:
            return None
        if col in contract.timetz_columns:
            return _parse_utc_timestamp(val, field=col).isoformat()
        if isinstance(val, (dict, list)):
            return json.dumps(val, sort_keys=True, allow_nan=False)
        return val

    def _unique_tuple(
        self,
        contract: TableContract,
        row: Mapping[str, Any],
        cols: tuple[str, ...],
    ) -> tuple[Any, ...]:
        return tuple(self._canonical_key_value(contract, c, row.get(c)) for c in cols)

    def _resolve_conflict_target(
        self,
        contract: TableContract,
        is_upsert: bool,
        on_conflict: str,
    ) -> tuple[str, ...] | None:
        if not is_upsert:
            return None
        if on_conflict.strip():
            cols = tuple(c.strip() for c in on_conflict.split(",") if c.strip())
            unknown = [c for c in cols if c not in contract.columns]
            if unknown:
                raise ContractViolationError(
                    f"[{contract.name}] Invalid on_conflict column(s): {unknown}"
                )
            matched = [u for u in contract.unique_sets if set(u) == set(cols) and len(u) == len(cols)]
            if not matched:
                raise ContractViolationError(
                    f"[{contract.name}] on_conflict={on_conflict!r} does not match any UNIQUE/PK "
                    f"constraint {contract.unique_sets}"
                )
            return matched[0]
        if contract.identity_pk is None and len(contract.unique_sets) == 1:
            return contract.unique_sets[0]
        if contract.identity_pk is not None:
            return (contract.identity_pk,)
        return None

    def _validate_and_prepare_row(
        self,
        contract: TableContract,
        raw_row: Mapping[str, Any],
        *,
        check_raw_json: bool = True,
    ) -> dict[str, Any]:
        if check_raw_json:
            # 1. Strict JSON check on raw input (rejects NaN, Inf, -Inf anywhere)
            try:
                json.dumps(raw_row, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ContractViolationError(
                    f"[{contract.name}] Payload failed json.dumps(..., allow_nan=False): {exc}"
                ) from exc

        # 2. Reject unknown columns
        unknown_cols = sorted(set(raw_row.keys()) - contract.columns)
        if unknown_cols:
            raise ContractViolationError(
                f"[{contract.name}] Unknown column(s) in write payload: {unknown_cols}"
            )

        # 3. Copy & apply Postgres DEFAULTs
        row = copy.deepcopy(dict(raw_row))
        for col, default_val in contract.defaults.items():
            if col not in row:
                row[col] = default_val() if callable(default_val) else copy.deepcopy(default_val)

        if contract.identity_pk is not None and row.get(contract.identity_pk) is None:
            row[contract.identity_pk] = self._next_ids[contract.name]
            self._next_ids[contract.name] += 1

        # 4. Validate TIMESTAMPTZ parseability
        for ts_col in contract.timetz_columns:
            if row.get(ts_col) is not None:
                _parse_utc_timestamp(row[ts_col], field=ts_col)

        # 5. Enforce NOT NULL constraints
        missing_not_null = [c for c in sorted(contract.not_null) if row.get(c) is None]
        if missing_not_null:
            raise ContractViolationError(
                f"[{contract.name}] NOT NULL violation on column(s): {missing_not_null}"
            )

        # 6. Enforce SQL CHECK constraints
        for check_fn in contract.sql_checks:
            check_fn(row)

        # 7. Enforce strict Component 3 Python domain contracts if enabled
        if self.strict_domain_enums:
            for domain_fn in contract.domain_checks:
                domain_fn(row)

        # 8. Final strict JSON serialization check after defaults
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractViolationError(
                f"[{contract.name}] Normalized row failed json.dumps(..., allow_nan=False): {exc}"
            ) from exc

        return row

    def _execute_write(
        self,
        *,
        table_name: str,
        rows: list[dict[str, Any]],
        is_upsert: bool,
        ignore_duplicates: bool,
        on_conflict: str,
    ) -> ContractSupabaseResponse:
        self._check_outage(table_name, "write")
        contract = TABLE_CONTRACTS[table_name]
        conflict_target = self._resolve_conflict_target(contract, is_upsert, on_conflict)

        # Validate raw batch JSON before any mutation
        try:
            json.dumps(rows, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractViolationError(
                f"[{table_name}] Batch payload failed json.dumps(..., allow_nan=False): {exc}"
            ) from exc

        saved_next_id = self._next_ids[table_name]
        try:
            prepared_rows = [
                self._validate_and_prepare_row(contract, r, check_raw_json=False)
                for r in rows
            ]
            base_rows = self.tables[table_name]
            base_len = len(base_rows)
            pending_updates: dict[int, dict[str, Any]] = {}
            new_rows: list[dict[str, Any]] = []

            def _get_row(idx: int) -> dict[str, Any]:
                if idx < base_len:
                    return pending_updates.get(idx, base_rows[idx])
                return new_rows[idx - base_len]

            conflict_index: dict[tuple[Any, ...], int] = {}
            if conflict_target is not None:
                for idx, ex in enumerate(base_rows):
                    c_key = self._unique_tuple(contract, ex, conflict_target)
                    conflict_index.setdefault(c_key, idx)

            unique_indexes: list[tuple[tuple[str, ...], dict[tuple[Any, ...], int]]] = []
            for u_cols in contract.unique_sets:
                u_map: dict[tuple[Any, ...], int] = {}
                for idx, ex in enumerate(base_rows):
                    u_key = self._unique_tuple(contract, ex, u_cols)
                    if any(v is None for v in u_key):
                        continue
                    u_map.setdefault(u_key, idx)
                unique_indexes.append((u_cols, u_map))

            seen_batch_conflict_keys: set[tuple[Any, ...]] = set()
            result_rows: list[dict[str, Any]] = []

            for incoming in prepared_rows:
                matched_idx: int | None = None
                inc_conflict_key: tuple[Any, ...] | None = None
                if conflict_target is not None:
                    inc_conflict_key = self._unique_tuple(contract, incoming, conflict_target)
                    if inc_conflict_key in seen_batch_conflict_keys:
                        raise ContractViolationError(
                            f"[{table_name}] Duplicate conflict key {inc_conflict_key} on {conflict_target} "
                            "within a single upsert batch."
                        )
                    seen_batch_conflict_keys.add(inc_conflict_key)
                    matched_idx = conflict_index.get(inc_conflict_key)

                # Verify all UNIQUE constraints against rows other than matched_idx
                inc_unique_keys: list[
                    tuple[tuple[str, ...], dict[tuple[Any, ...], int], tuple[Any, ...] | None]
                ] = []
                for u_cols, u_map in unique_indexes:
                    inc_u_key = self._unique_tuple(contract, incoming, u_cols)
                    if any(v is None for v in inc_u_key):
                        inc_unique_keys.append((u_cols, u_map, None))
                        continue
                    other_idx = u_map.get(inc_u_key)
                    if other_idx is not None and other_idx != matched_idx:
                        raise ContractViolationError(
                            f"[{table_name}] UNIQUE constraint violation on {u_cols}: {inc_u_key}"
                        )
                    inc_unique_keys.append((u_cols, u_map, inc_u_key))

                if matched_idx is not None:
                    old_row = _get_row(matched_idx)
                    if ignore_duplicates:
                        result_rows.append(copy.deepcopy(old_row))
                    else:
                        preserved_id = (
                            old_row.get(contract.identity_pk)
                            if contract.identity_pk
                            else None
                        )
                        preserved_created = old_row.get("created_at")
                        updated = copy.deepcopy(old_row)
                        updated.update(incoming)
                        if contract.identity_pk and preserved_id is not None:
                            updated[contract.identity_pk] = preserved_id
                        if preserved_created is not None:
                            updated["created_at"] = preserved_created
                        if contract.touch_updated_at:
                            updated["updated_at"] = _now_utc_iso()

                        for u_cols, u_map, _ in inc_unique_keys:
                            old_u_key = self._unique_tuple(contract, old_row, u_cols)
                            if not any(v is None for v in old_u_key) and u_map.get(old_u_key) == matched_idx:
                                del u_map[old_u_key]
                            new_u_key = self._unique_tuple(contract, updated, u_cols)
                            if not any(v is None for v in new_u_key):
                                other_u_idx = u_map.get(new_u_key)
                                if other_u_idx is not None and other_u_idx != matched_idx:
                                    raise ContractViolationError(
                                        f"[{table_name}] UNIQUE constraint violation on {u_cols}: {new_u_key}"
                                    )
                                u_map[new_u_key] = matched_idx

                        if matched_idx < base_len:
                            pending_updates[matched_idx] = updated
                        else:
                            new_rows[matched_idx - base_len] = updated
                        result_rows.append(copy.deepcopy(updated))
                else:
                    new_idx = base_len + len(new_rows)
                    new_rows.append(incoming)
                    if conflict_target is not None and inc_conflict_key is not None:
                        conflict_index[inc_conflict_key] = new_idx
                    for _, u_map, inc_u_key in inc_unique_keys:
                        if inc_u_key is not None:
                            u_map[inc_u_key] = new_idx
                    result_rows.append(copy.deepcopy(incoming))
        except Exception:
            self._next_ids[table_name] = saved_next_id
            raise

        for idx, updated_row in pending_updates.items():
            base_rows[idx] = updated_row
        if new_rows:
            base_rows.extend(new_rows)
        self.call_log.append(
            {
                "op": "upsert" if is_upsert else "insert",
                "table": table_name,
                "count": len(prepared_rows),
                "on_conflict": on_conflict,
            }
        )
        return ContractSupabaseResponse(data=result_rows)

    def _compare_values(
        self,
        contract: TableContract,
        column: str,
        row_val: Any,
        filter_val: Any,
    ) -> tuple[Any, Any]:
        if column in contract.timetz_columns:
            lhs = _parse_utc_timestamp(row_val, field=column)
            rhs = _parse_utc_timestamp(filter_val, field=column)
            return lhs, rhs
        return row_val, filter_val

    def _matches_filters(
        self,
        contract: TableContract,
        row: Mapping[str, Any],
        filters: Sequence[tuple[str, str, Any]],
    ) -> bool:
        for op, col, fval in filters:
            rval = row.get(col)
            if op == "is":
                if fval is None and rval is not None:
                    return False
                if fval is not None and rval is not fval and rval != fval:
                    return False
                continue
            if op == "in":
                if col in contract.timetz_columns:
                    if rval is None:
                        return False
                    lhs = _parse_utc_timestamp(rval, field=col)
                    targets = {_parse_utc_timestamp(v, field=col) for v in fval}
                    if lhs not in targets:
                        return False
                else:
                    if rval not in fval:
                        return False
                continue
            if op == "eq":
                if rval is None or fval is None:
                    if rval is not fval:
                        return False
                else:
                    lhs, rhs = self._compare_values(contract, col, rval, fval)
                    if lhs != rhs:
                        return False
                continue
            if op == "neq":
                if rval is None or fval is None:
                    if rval is fval:
                        return False
                else:
                    lhs, rhs = self._compare_values(contract, col, rval, fval)
                    if lhs == rhs:
                        return False
                continue
            if rval is None or fval is None:
                return False
            lhs, rhs = self._compare_values(contract, col, rval, fval)
            if op == "gt" and not (lhs > rhs):
                return False
            if op == "gte" and not (lhs >= rhs):
                return False
            if op == "lt" and not (lhs < rhs):
                return False
            if op == "lte" and not (lhs <= rhs):
                return False
        return True

    def _execute_read(
        self,
        *,
        table_name: str,
        select_cols: list[str] | None,
        count_mode: str | None,
        head_only: bool,
        filters: Sequence[tuple[str, str, Any]],
        orders: Sequence[tuple[str, bool, bool]],
        limit_val: int | None,
        range_bounds: tuple[int, int] | None,
    ) -> ContractSupabaseResponse:
        self._check_outage(table_name, "read")
        contract = TABLE_CONTRACTS[table_name]
        matched = [
            r
            for r in self.tables[table_name]
            if self._matches_filters(contract, r, filters)
        ]
        total_count = len(matched) if count_mode == "exact" else None

        # Multi-column stable sort applied from least-significant to most-significant key
        for col, desc, nullsfirst in reversed(orders):
            def _sort_key(item: Mapping[str, Any], *, _col: str = col) -> Any:
                v = item.get(_col)
                if v is None:
                    return None
                if _col in contract.timetz_columns:
                    return _parse_utc_timestamp(v, field=_col)
                return v

            non_null = [r for r in matched if r.get(col) is not None]
            nulls = [r for r in matched if r.get(col) is None]
            non_null.sort(key=_sort_key, reverse=desc)
            if nullsfirst:
                matched = nulls + non_null
            else:
                matched = non_null + nulls

        if range_bounds is not None:
            start, end = range_bounds
            if end < start:
                sliced = []
            else:
                capped_end = min(end + 1, start + self.max_rows_per_request)
                sliced = matched[start:capped_end]
        else:
            sliced = matched[: self.max_rows_per_request]

        if limit_val is not None:
            sliced = sliced[: min(limit_val, self.max_rows_per_request)]

        if head_only:
            out_data: list[dict[str, Any]] = []
        elif select_cols is not None:
            out_data = [copy.deepcopy({c: r.get(c) for c in select_cols}) for r in sliced]
        else:
            out_data = [copy.deepcopy(r) for r in sliced]

        self.call_log.append(
            {
                "op": "select",
                "table": table_name,
                "returned": len(out_data),
                "range": range_bounds,
                "limit": limit_val,
            }
        )
        return ContractSupabaseResponse(data=out_data, count=total_count)


# -----------------------------------------------------------------------------
# Hopsworks Feature Store & Model Registry Fakes (`hopsworks==5.2.3`)
# -----------------------------------------------------------------------------


class MockHopsworksFeatureGroup:
    """In-memory mock of `hsfs.feature_group.FeatureGroup`."""

    def __init__(
        self,
        *,
        name: str,
        version: int = 1,
        description: str = "",
        primary_key: Sequence[str] | None = None,
        event_time: str | None = None,
        online_enabled: bool = False,
        feature_store: MockHopsworksFeatureStore | None = None,
    ) -> None:
        self.name = name
        self.version = int(version)
        self.description = description
        self.primary_key = list(primary_key or [])
        self.event_time = event_time
        self.online_enabled = bool(online_enabled)
        self._fs = feature_store
        self.df: pd.DataFrame = pd.DataFrame()
        self.insert_calls: list[dict[str, Any]] = []

    def insert(
        self,
        features: pd.DataFrame,
        overwrite: bool = False,
        operation: str = "upsert",
        write_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[None, None]:
        del kwargs
        if self._fs is not None and self._fs.fail_on_insert:
            raise HopsworksOutageError(
                f"Injected Hopsworks Feature Group insert failure for '{self.name}'."
            )
        if not isinstance(features, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(features).__name__}")

        incoming = features.copy()
        if overwrite or self.df.empty:
            self.df = incoming.reset_index(drop=True)
        else:
            combined = pd.concat([self.df, incoming], ignore_index=True)
            if (
                operation == "upsert"
                and self.primary_key
                and all(pk in combined.columns for pk in self.primary_key)
            ):
                combined = combined.drop_duplicates(subset=self.primary_key, keep="last")
            self.df = combined.reset_index(drop=True)

        self.insert_calls.append(
            {
                "rows": len(incoming),
                "overwrite": bool(overwrite),
                "operation": operation,
                "write_options": dict(write_options or {}),
            }
        )
        return None, None

    def read(
        self,
        *,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        del kwargs
        if self._fs is not None and self._fs.fail_on_read:
            raise HopsworksOutageError(
                f"Injected Hopsworks Feature Group read failure for '{self.name}'."
            )
        if self.df.empty:
            return self.df.copy()

        out = self.df.copy()
        if self.event_time and self.event_time in out.columns:
            ts_series = pd.to_datetime(out[self.event_time], utc=True)
            out[self.event_time] = ts_series
            if start_time is not None:
                start_ts = pd.to_datetime(start_time, utc=True)
                out = out.loc[out[self.event_time] >= start_ts]
            if end_time is not None:
                end_ts = pd.to_datetime(end_time, utc=True)
                out = out.loc[out[self.event_time] <= end_ts]
            out = out.sort_values(self.event_time)

        if limit is not None:
            out = out.tail(int(limit))
        return out.reset_index(drop=True)


class MockHopsworksFeatureStore:
    """In-memory mock of `hsfs.feature_store.FeatureStore`."""

    def __init__(self, *, fail_on_insert: bool = False, fail_on_read: bool = False) -> None:
        self.fail_on_insert = bool(fail_on_insert)
        self.fail_on_read = bool(fail_on_read)
        self.feature_groups: dict[tuple[str, int], MockHopsworksFeatureGroup] = {}

    def get_or_create_feature_group(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        online_enabled: bool = False,
        primary_key: Sequence[str] | None = None,
        event_time: str | None = None,
        **kwargs: Any,
    ) -> MockHopsworksFeatureGroup:
        del kwargs
        key = (name, int(version))
        if key not in self.feature_groups:
            self.feature_groups[key] = MockHopsworksFeatureGroup(
                name=name,
                version=int(version),
                description=description,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=online_enabled,
                feature_store=self,
            )
        return self.feature_groups[key]

    def get_feature_group(
        self,
        name: str,
        version: int | None = None,
    ) -> MockHopsworksFeatureGroup:
        if self.fail_on_read:
            raise HopsworksOutageError(
                f"Injected Hopsworks Feature Store read failure for '{name}'."
            )
        if version is not None:
            key = (name, int(version))
            if key not in self.feature_groups:
                raise MockRestAPIError(f"Feature group '{name}' v{version} does not exist.")
            return self.feature_groups[key]

        matches = [
            fg for (fg_name, _), fg in sorted(self.feature_groups.items()) if fg_name == name
        ]
        if not matches:
            raise MockRestAPIError(f"Feature group '{name}' does not exist.")
        return matches[-1]


class MockHopsworksModel:
    """In-memory + temp-directory mock of `hsml.model.Model`."""

    def __init__(
        self,
        *,
        registry: MockHopsworksModelRegistry,
        name: str,
        version: int,
        training_metrics: Mapping[str, Any] | None = None,
        description: str | None = None,
        input_example: Any = None,
        model_schema: Any = None,
        tags: Any = None,
        project_name: str = "skycast_e2e",
    ) -> None:
        self._registry = registry
        self.name = name
        self.version = int(version)
        self.training_metrics: dict[str, Any] = dict(training_metrics or {})
        self.description = description
        self.input_example = input_example
        self.model_schema = model_schema
        if isinstance(tags, list):
            self.tags: dict[str, Any] = {str(t["name"]): t["value"] for t in tags if isinstance(t, Mapping) and "name" in t}
        else:
            self.tags = dict(tags or {})
        self.project_name = project_name
        self.id = f"{name}_v{self.version}"
        self.model_files_path: Path | None = None
        self.model_path: str | None = None

    def save(
        self,
        model_path: str | Path,
        await_registration: int = 480,
        keep_original_files: bool = False,
        upload_configuration: Any = None,
        **kwargs: Any,
    ) -> MockHopsworksModel:
        del await_registration, keep_original_files, upload_configuration, kwargs
        if self._registry.fail_on_save:
            raise HopsworksOutageError(
                f"Injected Hopsworks Model Registry save failure for '{self.name}' v{self.version}."
            )
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model artifact path does not exist: {src}")

        dest = self._registry.artifact_root / self.name / f"v{self.version}"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src.name)

        self.model_files_path = dest
        self.model_path = str(dest)
        self._registry._register_saved_model(self)
        return self

    def download(self, local_path: str | Path | None = None) -> str:
        if self._registry.fail_on_read:
            raise HopsworksOutageError(
                f"Injected Hopsworks Model Registry download failure for '{self.name}' v{self.version}."
            )
        if self.model_files_path is None or not self.model_files_path.exists():
            raise MockRestAPIError(
                f"No saved artifacts found for model '{self.name}' v{self.version}."
            )
        target = (
            Path(local_path)
            if local_path is not None
            else self._registry.artifact_root / "_downloads" / self.name / f"v{self.version}"
        )
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.model_files_path, target)
        return str(target)


class MockHopsworksPythonModelApi:
    """Mock of `mr.python` (`hsml.python.model_api.ModelApi`)."""

    def __init__(self, registry: MockHopsworksModelRegistry) -> None:
        self._registry = registry

    def create_model(
        self,
        name: str,
        version: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        description: str | None = None,
        input_example: Any = None,
        model_schema: Any = None,
        feature_view: Any = None,
        training_dataset_version: int | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> MockHopsworksModel:
        del feature_view, training_dataset_version
        resolved_version = (
            int(version)
            if version is not None
            else self._registry._allocate_next_version(name)
        )
        return MockHopsworksModel(
            registry=self._registry,
            name=name,
            version=resolved_version,
            training_metrics=metrics,
            description=description,
            input_example=input_example,
            model_schema=model_schema,
            tags=tags,
            project_name=self._registry.project_name,
        )


class MockHopsworksModelRegistry:
    """In-memory + temp-directory mock of `hsml.model_registry.ModelRegistry`."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        fail_on_read: bool = False,
        fail_on_save: bool = False,
        raise_not_found_on_missing: bool = False,
        project_name: str = "skycast_e2e",
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.fail_on_read = bool(fail_on_read)
        self.fail_on_save = bool(fail_on_save)
        self.raise_not_found_on_missing = bool(raise_not_found_on_missing)
        self.project_name = project_name
        self._models: dict[str, list[MockHopsworksModel]] = {}
        self._python_api = MockHopsworksPythonModelApi(self)

    @property
    def python(self) -> MockHopsworksPythonModelApi:
        return self._python_api

    def _allocate_next_version(self, name: str) -> int:
        existing = self._models.get(name, [])
        return max((m.version for m in existing), default=0) + 1

    def _register_saved_model(self, model: MockHopsworksModel) -> None:
        bucket = self._models.setdefault(model.name, [])
        bucket[:] = [m for m in bucket if m.version != model.version]
        bucket.append(model)
        bucket.sort(key=lambda m: m.version)

    def get_models(self, name: str) -> list[MockHopsworksModel]:
        if self.fail_on_read:
            raise HopsworksOutageError(
                f"Injected Hopsworks Model Registry read failure for '{name}'."
            )
        models = list(self._models.get(name, []))
        if not models and self.raise_not_found_on_missing:
            raise MockRestAPIError(f"Model '{name}' not found in registry.")
        return models

    def get_model(
        self,
        name: str,
        version: int | None = None,
    ) -> MockHopsworksModel | None:
        if self.fail_on_read:
            raise HopsworksOutageError(
                f"Injected Hopsworks Model Registry read failure for '{name}'."
            )
        models = self._models.get(name, [])
        if not models:
            if self.raise_not_found_on_missing:
                raise MockRestAPIError(f"Model '{name}' not found in registry.")
            return None
        if version is None:
            return models[-1]
        for model in models:
            if model.version == int(version):
                return model
        if self.raise_not_found_on_missing:
            raise MockRestAPIError(f"Model '{name}' v{version} not found in registry.")
        return None

    def get_best_model(
        self,
        name: str,
        metric: str,
        direction: str = "min",
    ) -> MockHopsworksModel | None:
        if self.fail_on_read:
            raise HopsworksOutageError(
                f"Injected Hopsworks Model Registry read failure for '{name}'."
            )
        models = [
            m
            for m in self._models.get(name, [])
            if metric in m.training_metrics and m.training_metrics[metric] is not None
        ]
        if not models:
            if self.raise_not_found_on_missing:
                raise MockRestAPIError(f"No models with metric '{metric}' for '{name}'.")
            return None
        norm = direction.strip().lower()
        if norm in {"min", "asc", "ascending"}:
            return min(models, key=lambda m: float(m.training_metrics[metric]))
        if norm in {"max", "desc", "descending"}:
            return max(models, key=lambda m: float(m.training_metrics[metric]))
        raise ValueError(f"Unsupported get_best_model direction: {direction!r}")


class MockHopsworksProject:
    """In-memory mock of `hopsworks.project.Project` and `hopsworks.login(...)`."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        name: str = "skycast_e2e",
        fail_on_login: bool = False,
        feature_store: MockHopsworksFeatureStore | None = None,
        model_registry: MockHopsworksModelRegistry | None = None,
    ) -> None:
        self.name = name
        self.fail_on_login = bool(fail_on_login)
        self.artifact_root = Path(artifact_root)
        self.feature_store = feature_store or MockHopsworksFeatureStore()
        self.model_registry = model_registry or MockHopsworksModelRegistry(
            artifact_root=self.artifact_root,
            project_name=name,
        )
        self.login_calls: list[dict[str, Any]] = []

    def get_feature_store(self, name: str | None = None) -> MockHopsworksFeatureStore:
        del name
        return self.feature_store

    def get_model_registry(self) -> MockHopsworksModelRegistry:
        return self.model_registry

    def login(self, **kwargs: Any) -> MockHopsworksProject:
        self.login_calls.append(dict(kwargs))
        if self.fail_on_login:
            raise HopsworksOutageError("Injected Hopsworks login outage.")
        return self
