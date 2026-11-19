"""Contract-enforcing fakes for Supabase/PostgREST and Hopsworks (Feature Store + Model Registry)."""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skycast.config.settings import MODEL_FAMILIES, Settings, TARGET_METRICS

_TARGET_METRICS_SET = frozenset(TARGET_METRICS)
_MODEL_FAMILIES_SET = frozenset(MODEL_FAMILIES)
_TIMESTAMP_HINT_COLS = frozenset(
    {
        "timestamp_utc",
        "issued_at_utc",
        "target_timestamp_utc",
        "trained_at_utc",
        "computed_at_utc",
        "started_at_utc",
        "completed_at_utc",
        "reference_timestamp_utc",
        "date_range_start",
        "date_range_end",
        "created_at",
        "updated_at",
        "timestamp",
        "started_at",
    }
)


class ContractViolationError(RuntimeError):
    """Raised when a write violates Postgres CHECK, NOT NULL, UNIQUE, or strict JSON contracts."""


def make_test_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Create isolated Settings that ignore `.env` and store SQLite under `tmp_path`."""
    defaults: dict[str, Any] = {
        "_env_file": None,
        "skycast_offline_mode": "true",
        "skycast_local_data_dir": tmp_path / ".skycast_local",
        "supabase_url": "",
        "supabase_service_role_key": "",
        "hopsworks_api_key": "",
        "hopsworks_project_name": "",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _parse_utc_timestamp(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        dt = val.to_pydatetime()
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    if isinstance(val, datetime):
        return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val.astimezone(timezone.utc)
    if isinstance(val, str):
        try:
            s = val[:-1] + "+00:00" if val.endswith("Z") else val
            dt = datetime.fromisoformat(s)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        ts = pd.to_datetime(val, utc=True)
        if not pd.isna(ts):
            return ts.to_pydatetime()
    except Exception:
        pass
    return None


def _normalize_conflict_value(col: str, val: Any) -> Any:
    if val is None:
        return None
    if col in _TIMESTAMP_HINT_COLS or col.endswith("_utc"):
        dt = _parse_utc_timestamp(val)
        if dt is not None:
            return dt.isoformat()
    return val


def _assert_finite_number(val: Any, field_name: str, *, non_negative: bool = False) -> float:
    if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
        raise ContractViolationError(f"Column '{field_name}' must be numeric, got {val!r}.")
    f = float(val)
    if not math.isfinite(f):
        raise ContractViolationError(f"Column '{field_name}' must be finite, got {val!r}.")
    if non_negative and f < 0.0:
        raise ContractViolationError(f"Column '{field_name}' must be >= 0, got {f}.")
    return f


def _assert_valid_timestamp(val: Any, field_name: str, *, required: bool = True) -> None:
    if val is None:
        if required:
            raise ContractViolationError(f"Column '{field_name}' is NOT NULL.")
        return
    if isinstance(val, (bool, np.bool_)) or not isinstance(val, (str, datetime, pd.Timestamp)):
        raise ContractViolationError(
            f"Column '{field_name}' must be an ISO timestamp string, got {val!r}."
        )
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ContractViolationError(f"Column '{field_name}' must be a non-empty ISO timestamp string.")
        try:
            iso_s = f"{s[:-1]}+00:00" if s.endswith(("Z", "z")) else s
            datetime.fromisoformat(iso_s)
        except ValueError as exc:
            raise ContractViolationError(
                f"Column '{field_name}' must be a valid ISO timestamp string, got {val!r}."
            ) from exc
        return
    if isinstance(val, pd.Timestamp) and pd.isna(val):
        raise ContractViolationError(f"Column '{field_name}' cannot be NaT.")
    if isinstance(val, datetime):
        try:
            if pd.isna(val):
                raise ContractViolationError(f"Column '{field_name}' cannot be NaT.")
        except (TypeError, ValueError):
            pass


def _validate_eda_summaries_row(row: dict[str, Any]) -> None:
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ContractViolationError("eda_summaries.run_id must be a non-empty string.")
    _assert_valid_timestamp(row.get("computed_at_utc"), "eda_summaries.computed_at_utc", required=True)
    _assert_valid_timestamp(row.get("date_range_start"), "eda_summaries.date_range_start", required=False)
    _assert_valid_timestamp(row.get("date_range_end"), "eda_summaries.date_range_end", required=False)
    _assert_valid_timestamp(row.get("created_at"), "eda_summaries.created_at", required=True)
    sample_count = row.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ContractViolationError(f"eda_summaries.sample_count must be int >= 0, got {sample_count!r}.")
    if not isinstance(row.get("summary_json"), dict):
        raise ContractViolationError("eda_summaries.summary_json must be a JSON object (dict).")


def _validate_model_registry_metrics_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("train_run_id"), str) or not row["train_run_id"].strip():
        raise ContractViolationError("model_registry_metrics.train_run_id must be non-empty.")
    _assert_valid_timestamp(row.get("trained_at_utc"), "model_registry_metrics.trained_at_utc", required=True)
    _assert_valid_timestamp(row.get("created_at"), "model_registry_metrics.created_at", required=True)
    if row.get("metric") not in _TARGET_METRICS_SET:
        raise ContractViolationError(f"Invalid model_registry_metrics.metric: {row.get('metric')!r}")
    if row.get("model_family") not in _MODEL_FAMILIES_SET:
        raise ContractViolationError(
            f"Invalid model_registry_metrics.model_family: {row.get('model_family')!r}"
        )
    for col in ("val_rmse", "val_mae", "test_rmse", "test_mae"):
        _assert_finite_number(row.get(col), col, non_negative=True)
    for col in ("val_r2", "test_r2", "improvement_over_naive_pct"):
        _assert_finite_number(row.get(col), col)
    for bcol in ("is_champion", "promoted_to_registry"):
        if not isinstance(row.get(bcol), bool):
            raise ContractViolationError(f"model_registry_metrics.{bcol} must be bool.")
    if not isinstance(row.get("hyperparameters"), dict):
        raise ContractViolationError("model_registry_metrics.hyperparameters must be dict.")


def _validate_model_comparison_series_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("train_run_id"), str) or not row["train_run_id"].strip():
        raise ContractViolationError("model_comparison_series.train_run_id must be non-empty.")
    _assert_valid_timestamp(row.get("timestamp_utc"), "model_comparison_series.timestamp_utc", required=True)
    _assert_valid_timestamp(row.get("created_at"), "model_comparison_series.created_at", required=True)
    if row.get("metric") not in _TARGET_METRICS_SET:
        raise ContractViolationError(f"Invalid model_comparison_series.metric: {row.get('metric')!r}")
    for col in ("actual", "sarimax", "random_forest", "ridge", "lstm", "naive_24h"):
        val = row.get(col)
        if val is not None:
            _assert_finite_number(val, col)


def _validate_shap_lime_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("train_run_id"), str) or not row["train_run_id"].strip():
        raise ContractViolationError("shap_lime_explanations.train_run_id must be non-empty.")
    _assert_valid_timestamp(row.get("computed_at_utc"), "shap_lime_explanations.computed_at_utc", required=True)
    _assert_valid_timestamp(
        row.get("reference_timestamp_utc"),
        "shap_lime_explanations.reference_timestamp_utc",
        required=False,
    )
    _assert_valid_timestamp(row.get("created_at"), "shap_lime_explanations.created_at", required=True)
    if row.get("metric") not in _TARGET_METRICS_SET:
        raise ContractViolationError(f"Invalid shap_lime_explanations.metric: {row.get('metric')!r}")
    if row.get("model_family") not in _MODEL_FAMILIES_SET:
        raise ContractViolationError(
            f"Invalid shap_lime_explanations.model_family: {row.get('model_family')!r}"
        )
    if row.get("method") not in {"shap", "lime"}:
        raise ContractViolationError(f"CHECK failed on shap_lime_explanations.method: {row.get('method')!r}")
    if row.get("scope") not in {"global", "local"}:
        raise ContractViolationError(f"CHECK failed on shap_lime_explanations.scope: {row.get('scope')!r}")
    for col in ("base_value", "predicted_value"):
        if row.get(col) is not None:
            _assert_finite_number(row[col], col)
    if not isinstance(row.get("feature_attributions"), list):
        raise ContractViolationError("shap_lime_explanations.feature_attributions must be list.")


def _validate_forecast_ledger_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("run_id"), str) or not row["run_id"].strip():
        raise ContractViolationError("forecast_ledger.run_id must be non-empty.")
    _assert_valid_timestamp(row.get("issued_at_utc"), "forecast_ledger.issued_at_utc", required=True)
    _assert_valid_timestamp(
        row.get("target_timestamp_utc"), "forecast_ledger.target_timestamp_utc", required=True
    )
    _assert_valid_timestamp(row.get("created_at"), "forecast_ledger.created_at", required=True)
    horizon = row.get("horizon_hour")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not (1 <= horizon <= 24):
        raise ContractViolationError(f"CHECK failed on forecast_ledger.horizon_hour: {horizon!r}")
    if row.get("metric") not in _TARGET_METRICS_SET:
        raise ContractViolationError(f"Invalid forecast_ledger.metric: {row.get('metric')!r}")
    if row.get("model_family") not in _MODEL_FAMILIES_SET:
        raise ContractViolationError(f"Invalid forecast_ledger.model_family: {row.get('model_family')!r}")
    pred = _assert_finite_number(row.get("predicted_value"), "predicted_value")
    low = _assert_finite_number(row.get("lower_bound"), "lower_bound")
    high = _assert_finite_number(row.get("upper_bound"), "upper_bound")
    if not (low <= pred <= high):
        raise ContractViolationError(
            f"forecast_ledger bounds violated: {low} <= {pred} <= {high} is False."
        )
    for col in ("naive_baseline_value", "actual_value"):
        if row.get(col) is not None:
            _assert_finite_number(row[col], col)


def _validate_pipeline_runs_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("run_id"), str) or not row["run_id"].strip():
        raise ContractViolationError("pipeline_runs.run_id must be non-empty.")
    _assert_valid_timestamp(row.get("started_at_utc"), "pipeline_runs.started_at_utc", required=True)
    _assert_valid_timestamp(row.get("completed_at_utc"), "pipeline_runs.completed_at_utc", required=False)
    _assert_valid_timestamp(row.get("created_at"), "pipeline_runs.created_at", required=True)
    if row.get("pipeline_type") not in {"backfill", "hourly_ingest", "training"}:
        raise ContractViolationError(
            f"CHECK failed on pipeline_runs.pipeline_type: {row.get('pipeline_type')!r}"
        )
    if row.get("status") not in {"running", "completed", "failed"}:
        raise ContractViolationError(f"CHECK failed on pipeline_runs.status: {row.get('status')!r}")
    if not isinstance(row.get("metadata_json"), dict):
        raise ContractViolationError("pipeline_runs.metadata_json must be dict.")


def _validate_weather_features_row(row: dict[str, Any]) -> None:
    _assert_valid_timestamp(row.get("timestamp_utc"), "weather_features_hourly.timestamp_utc", required=True)
    _assert_valid_timestamp(row.get("created_at"), "weather_features_hourly.created_at", required=True)
    _assert_valid_timestamp(row.get("updated_at"), "weather_features_hourly.updated_at", required=True)


@dataclass(frozen=True, slots=True)
class _TableContract:
    unique_sets: tuple[tuple[str, ...], ...]
    not_null: tuple[str, ...]
    defaults: dict[str, Any]
    row_validator: Callable[[dict[str, Any]], None]


_TABLE_CONTRACTS: dict[str, _TableContract] = {
    "eda_summaries": _TableContract(
        unique_sets=(("run_id",),),
        not_null=("run_id", "computed_at_utc", "created_at"),
        defaults={"sample_count": 0, "summary_json": {}},
        row_validator=_validate_eda_summaries_row,
    ),
    "model_registry_metrics": _TableContract(
        unique_sets=(("train_run_id", "metric", "model_family"),),
        not_null=(
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
            "created_at",
        ),
        defaults={
            "improvement_over_naive_pct": 0.0,
            "is_champion": False,
            "promoted_to_registry": False,
            "hyperparameters": {},
        },
        row_validator=_validate_model_registry_metrics_row,
    ),
    "model_comparison_series": _TableContract(
        unique_sets=(("train_run_id", "metric", "timestamp_utc"),),
        not_null=("train_run_id", "metric", "timestamp_utc", "created_at"),
        defaults={},
        row_validator=_validate_model_comparison_series_row,
    ),
    "shap_lime_explanations": _TableContract(
        unique_sets=(("train_run_id", "metric", "method", "scope"),),
        not_null=("train_run_id", "computed_at_utc", "metric", "model_family", "method", "scope", "created_at"),
        defaults={"feature_attributions": []},
        row_validator=_validate_shap_lime_row,
    ),
    "forecast_ledger": _TableContract(
        unique_sets=(("run_id", "metric", "model_family", "horizon_hour"),),
        not_null=(
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
        ),
        defaults={},
        row_validator=_validate_forecast_ledger_row,
    ),
    "pipeline_runs": _TableContract(
        unique_sets=(("run_id",),),
        not_null=("run_id", "pipeline_type", "status", "started_at_utc", "created_at"),
        defaults={"storage_mode": "cloud", "metadata_json": {}},
        row_validator=_validate_pipeline_runs_row,
    ),
    "weather_features_hourly": _TableContract(
        unique_sets=(("timestamp_utc",),),
        not_null=("timestamp_utc", "created_at", "updated_at"),
        defaults={
            "sources_available": [],
            "is_imputed": False,
            "imputation_tier": "none",
            "imputed_columns": [],
            "features_json": {},
        },
        row_validator=_validate_weather_features_row,
    ),
}


def _coerce_comparable(col: str, value: Any) -> Any:
    if value is None:
        return None
    if col in _TIMESTAMP_HINT_COLS or col.endswith("_utc"):
        dt = _parse_utc_timestamp(value)
        if dt is not None:
            return dt
    return value


def _sort_key_for_column(value: Any, col: str) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    coerced = _coerce_comparable(col, value)
    return (0, coerced)


class FakeSupabaseQuery:
    """Contract-enforcing PostgREST SELECT builder with 1,000-row `.range()` cap."""

    def __init__(self, client: FakeSupabaseClient, table_name: str, columns: str = "*") -> None:
        self._client = client
        self._table_name = table_name
        self._columns = columns
        self._filters: list[tuple[str, str, Any]] = []
        self._orders: list[tuple[str, bool]] = []
        self._limit: int | None = None
        self._range_start: int = 0
        self._range_end: int = 999
        self._range_explicit: bool = False

    def eq(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("eq", col, val))
        return self

    def neq(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("neq", col, val))
        return self

    def gt(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("gt", col, val))
        return self

    def gte(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("gte", col, val))
        return self

    def lt(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("lt", col, val))
        return self

    def lte(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("lte", col, val))
        return self

    def in_(self, col: str, values: Iterable[Any]) -> FakeSupabaseQuery:
        self._filters.append(("in", col, list(values)))
        return self

    def is_(self, col: str, val: Any) -> FakeSupabaseQuery:
        self._filters.append(("is", col, val))
        return self

    def order(self, col: str, desc: bool = False) -> FakeSupabaseQuery:
        self._orders.append((col, desc))
        return self

    def limit(self, count: int) -> FakeSupabaseQuery:
        self._limit = int(count)
        return self

    def range(self, start: int, end: int) -> FakeSupabaseQuery:
        self._range_start = int(start)
        self._range_end = int(end)
        self._range_explicit = True
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        for op, col, expected in self._filters:
            actual = row.get(col)
            if op == "is":
                if actual is not expected and actual != expected:
                    return False
                continue
            if op == "in":
                exp_set = {_coerce_comparable(col, v) for v in expected}
                if _coerce_comparable(col, actual) not in exp_set:
                    return False
                continue
            lhs = _coerce_comparable(col, actual)
            rhs = _coerce_comparable(col, expected)
            if op == "eq":
                if lhs != rhs:
                    return False
            elif op == "neq":
                if lhs == rhs:
                    return False
            else:
                if lhs is None or rhs is None:
                    return False
                if op == "gt" and not (lhs > rhs):
                    return False
                if op == "gte" and not (lhs >= rhs):
                    return False
                if op == "lt" and not (lhs < rhs):
                    return False
                if op == "lte" and not (lhs <= rhs):
                    return False
        return True

    def execute(self) -> SimpleNamespace:
        self._client._raise_if_failed(self._table_name, "select")
        rows = [copy.deepcopy(r) for r in self._client.tables[self._table_name] if self._matches(r)]
        for col, desc in reversed(self._orders):
            rows.sort(key=lambda r, c=col: _sort_key_for_column(r.get(c), c), reverse=desc)

        effective_end = min(self._range_end, self._range_start + 999)
        sliced = rows[self._range_start : effective_end + 1]
        if self._limit is not None:
            sliced = sliced[: self._limit]

        if self._columns.strip() != "*":
            wanted = [c.strip() for c in self._columns.split(",") if c.strip()]
            sliced = [{c: r.get(c) for c in wanted} for r in sliced]

        self._client.call_log.append(
            {
                "table": self._table_name,
                "op": "select",
                "range": (self._range_start, effective_end),
                "range_explicit": self._range_explicit,
                "returned": len(sliced),
            }
        )
        return SimpleNamespace(data=copy.deepcopy(sliced))


class FakeSupabaseWriteQuery:
    """Contract-enforcing PostgREST INSERT / UPSERT builder."""

    def __init__(
        self,
        client: FakeSupabaseClient,
        table_name: str,
        op: str,
        payload: list[dict[str, Any]],
        *,
        on_conflict: str = "",
        ignore_duplicates: bool = False,
    ) -> None:
        self._client = client
        self._table_name = table_name
        self._op = op
        self._payload = payload
        self._on_conflict = on_conflict
        self._ignore_duplicates = ignore_duplicates
        self._select_cols: str | None = None

    def select(self, columns: str = "*") -> FakeSupabaseWriteQuery:
        self._select_cols = columns
        return self

    def execute(self) -> SimpleNamespace:
        self._client._raise_if_failed(self._table_name, self._op)
        written = self._client._apply_write(
            self._table_name,
            self._op,
            self._payload,
            on_conflict=self._on_conflict,
            ignore_duplicates=self._ignore_duplicates,
        )
        self._client.call_log.append(
            {
                "table": self._table_name,
                "op": self._op,
                "on_conflict": self._on_conflict,
                "count": len(written),
            }
        )
        if self._select_cols and self._select_cols.strip() != "*":
            wanted = [c.strip() for c in self._select_cols.split(",") if c.strip()]
            written = [{c: r.get(c) for c in wanted} for r in written]
        return SimpleNamespace(data=copy.deepcopy(written))


class FakeSupabaseTable:
    """Table entry point matching `supabase.client.SyncRequestBuilder`."""

    def __init__(self, client: FakeSupabaseClient, table_name: str) -> None:
        self._client = client
        self._table_name = table_name

    def select(self, columns: str = "*") -> FakeSupabaseQuery:
        return FakeSupabaseQuery(self._client, self._table_name, columns=columns)

    def upsert(
        self,
        json_payload: dict[str, Any] | list[dict[str, Any]],
        on_conflict: str = "",
        count: str | None = None,
        ignore_duplicates: bool = False,
    ) -> FakeSupabaseWriteQuery:
        del count
        rows = [json_payload] if isinstance(json_payload, dict) else list(json_payload)
        self._assert_strict_json(rows)
        contract = _TABLE_CONTRACTS[self._table_name]
        conflict_cols = tuple(c.strip() for c in on_conflict.split(",") if c.strip())
        if conflict_cols not in contract.unique_sets:
            raise ContractViolationError(
                f"Invalid on_conflict={on_conflict!r} for table '{self._table_name}'; "
                f"expected one of {contract.unique_sets}."
            )
        return FakeSupabaseWriteQuery(
            self._client,
            self._table_name,
            "upsert",
            copy.deepcopy(rows),
            on_conflict=on_conflict,
            ignore_duplicates=ignore_duplicates,
        )

    def insert(
        self,
        json_payload: dict[str, Any] | list[dict[str, Any]],
    ) -> FakeSupabaseWriteQuery:
        rows = [json_payload] if isinstance(json_payload, dict) else list(json_payload)
        self._assert_strict_json(rows)
        return FakeSupabaseWriteQuery(
            self._client,
            self._table_name,
            "insert",
            copy.deepcopy(rows),
        )

    def _assert_strict_json(self, rows: list[dict[str, Any]]) -> None:
        try:
            json.dumps(rows, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractViolationError(
                f"Payload for '{self._table_name}' failed strict JSON serialization (allow_nan=False): {exc}"
            ) from exc


class FakeSupabaseClient:
    """In-memory PostgREST simulator enforcing Postgres schema contracts and 1,000-row cap."""

    def __init__(self, *, fail_rules: dict[tuple[str, str], Exception] | None = None) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {name: [] for name in _TABLE_CONTRACTS}
        self.call_log: list[dict[str, Any]] = []
        self.fail_rules: dict[tuple[str, str], Exception] = dict(fail_rules or {})
        self._next_id: dict[str, int] = {name: 1 for name in _TABLE_CONTRACTS}

    def table(self, name: str) -> FakeSupabaseTable:
        if name not in _TABLE_CONTRACTS:
            raise ContractViolationError(f"Unknown Supabase table '{name}'.")
        return FakeSupabaseTable(self, name)

    def seed_rows(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        if table_name not in _TABLE_CONTRACTS:
            raise ContractViolationError(f"Unknown Supabase table '{table_name}'.")
        contract = _TABLE_CONTRACTS[table_name]
        conflict_cols = contract.unique_sets[0]
        self._apply_write(
            table_name,
            "upsert",
            copy.deepcopy(rows),
            on_conflict=",".join(conflict_cols),
        )

    def get_rows(self, table_name: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self.tables[table_name])

    def _raise_if_failed(self, table_name: str, op: str) -> None:
        exc = self.fail_rules.get((table_name, op)) or self.fail_rules.get(("*", op))
        if exc is not None:
            raise exc

    def _apply_write(
        self,
        table_name: str,
        op: str,
        payload: list[dict[str, Any]],
        *,
        on_conflict: str = "",
        ignore_duplicates: bool = False,
    ) -> list[dict[str, Any]]:
        contract = _TABLE_CONTRACTS[table_name]
        conflict_cols = tuple(c.strip() for c in on_conflict.split(",") if c.strip())
        existing = self.tables[table_name]
        written: list[dict[str, Any]] = []
        batch_seen: set[tuple[Any, ...]] = set()
        conflict_index: dict[tuple[Any, ...], int] = {}
        unique_indexes: list[tuple[tuple[str, ...], set[tuple[Any, ...]]]] = []
        if op == "upsert":
            conflict_index = {
                tuple(_normalize_conflict_value(c, cur.get(c)) for c in conflict_cols): idx
                for idx, cur in enumerate(existing)
            }
        else:
            for u_set in contract.unique_sets:
                unique_indexes.append(
                    (
                        u_set,
                        {
                            tuple(_normalize_conflict_value(c, cur.get(c)) for c in u_set)
                            for cur in existing
                        },
                    )
                )

        for raw_row in payload:
            row = copy.deepcopy(raw_row)
            for k, default_val in contract.defaults.items():
                if k not in row:
                    row[k] = copy.deepcopy(default_val)
            now_iso = _now_utc_iso()
            if "created_at" not in row:
                row["created_at"] = now_iso
            if table_name == "weather_features_hourly" and "updated_at" not in row:
                row["updated_at"] = now_iso
            if table_name == "eda_summaries" and "computed_at_utc" not in row:
                row["computed_at_utc"] = now_iso
            if table_name == "model_registry_metrics" and "trained_at_utc" not in row:
                row["trained_at_utc"] = now_iso
            if table_name == "shap_lime_explanations" and "computed_at_utc" not in row:
                row["computed_at_utc"] = now_iso
            if table_name == "pipeline_runs" and "started_at_utc" not in row:
                row["started_at_utc"] = now_iso

            for col in contract.not_null:
                if row.get(col) is None:
                    raise ContractViolationError(f"NOT NULL violation on {table_name}.{col}.")

            contract.row_validator(row)

            if op == "upsert":
                key = tuple(_normalize_conflict_value(c, row.get(c)) for c in conflict_cols)
                if key in batch_seen:
                    raise ContractViolationError(
                        f"ON CONFLICT DO UPDATE cannot affect row a second time in {table_name}: {key}"
                    )
                batch_seen.add(key)

                match_idx = conflict_index.get(key)
                if match_idx is not None:
                    if ignore_duplicates:
                        written.append(existing[match_idx])
                        continue
                    preserved_id = existing[match_idx].get("id")
                    preserved_created = existing[match_idx].get("created_at")
                    if preserved_id is not None:
                        row["id"] = preserved_id
                    if preserved_created is not None:
                        row["created_at"] = preserved_created
                    existing[match_idx] = row
                    written.append(row)
                else:
                    if table_name != "weather_features_hourly" and "id" not in row:
                        row["id"] = self._next_id[table_name]
                        self._next_id[table_name] += 1
                    conflict_index[key] = len(existing)
                    existing.append(row)
                    written.append(row)
            else:
                for u_set, seen_set in unique_indexes:
                    key = tuple(_normalize_conflict_value(c, row.get(c)) for c in u_set)
                    if key in seen_set:
                        raise ContractViolationError(
                            f"UNIQUE violation on {table_name}{u_set}: {key}"
                        )
                    seen_set.add(key)
                if table_name != "weather_features_hourly" and "id" not in row:
                    row["id"] = self._next_id[table_name]
                    self._next_id[table_name] += 1
                existing.append(row)
                written.append(row)

        return written


class FakeHopsworksFeatureGroup:
    """Contract-enforcing Hopsworks Feature Group fake."""

    def __init__(
        self,
        name: str,
        version: int,
        *,
        description: str = "",
        primary_key: list[str] | None = None,
        event_time: str | None = None,
        online_enabled: bool = False,
        fail_rules: dict[tuple[str, str], Exception] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.primary_key = list(primary_key or [])
        self.event_time = event_time
        self.online_enabled = online_enabled
        self._fail_rules = fail_rules if fail_rules is not None else {}
        self._rows_df: pd.DataFrame | None = None
        self.insert_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []

    def _raise_if_failed(self, op: str) -> None:
        exc = self._fail_rules.get((op, self.name)) or self._fail_rules.get((op, "*"))
        if exc is not None:
            raise exc

    def insert(
        self,
        features: pd.DataFrame,
        overwrite: bool = False,
        operation: str = "upsert",
        write_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._raise_if_failed("fg_insert")
        if not isinstance(features, pd.DataFrame):
            raise ContractViolationError("Hopsworks FG.insert requires a pandas DataFrame.")
        for pk in self.primary_key:
            if pk not in features.columns:
                raise ContractViolationError(f"Hopsworks FG '{self.name}' missing primary_key '{pk}'.")
            if features[pk].isna().any():
                raise ContractViolationError(f"Hopsworks FG '{self.name}' primary_key '{pk}' has nulls.")

        is_forecast_ledger = (
            "horizon_hour" in features.columns
            or "forecast_ledger" in self.name
            or "horizon_hour" in self.primary_key
        )
        if is_forecast_ledger:
            if "horizon_hour" not in features.columns:
                raise ContractViolationError(f"Hopsworks FG '{self.name}' missing 'horizon_hour'.")
            if pd.api.types.is_bool_dtype(features["horizon_hour"]) or pd.api.types.is_float_dtype(
                features["horizon_hour"]
            ):
                raise ContractViolationError(
                    f"Hopsworks FG '{self.name}' horizon_hour must be integer dtype, "
                    f"got {features['horizon_hour'].dtype}."
                )
            for row in features.to_dict(orient="records"):
                raw_h = row.get("horizon_hour")
                if isinstance(raw_h, (bool, np.bool_)) or not isinstance(raw_h, (int, np.integer)):
                    raise ContractViolationError(
                        f"Hopsworks FG horizon_hour must be integer, got {raw_h!r}."
                    )
                h = int(raw_h)
                if not (1 <= h <= 24):
                    raise ContractViolationError(f"Hopsworks FG horizon_hour out of 1..24: {h}")
                if row.get("metric") not in _TARGET_METRICS_SET:
                    raise ContractViolationError(
                        f"Hopsworks FG invalid metric: {row.get('metric')!r}"
                    )
                if row.get("model_family") not in _MODEL_FAMILIES_SET:
                    raise ContractViolationError(
                        f"Hopsworks FG invalid model_family: {row.get('model_family')!r}"
                    )
                pred = _assert_finite_number(row.get("predicted_value"), "predicted_value")
                low = _assert_finite_number(row.get("lower_bound"), "lower_bound")
                high = _assert_finite_number(row.get("upper_bound"), "upper_bound")
                if not (low <= pred <= high):
                    raise ContractViolationError(
                        f"Hopsworks FG bounds violated: {low} <= {pred} <= {high}"
                    )

        incoming = features.copy()
        self.insert_calls.append(
            {
                "rows": len(incoming),
                "operation": operation,
                "write_options": dict(write_options or {}),
                "kwargs": dict(kwargs),
                "df": incoming.copy(),
            }
        )
        if overwrite or self._rows_df is None or self._rows_df.empty:
            if self.primary_key:
                self._rows_df = incoming.drop_duplicates(subset=self.primary_key, keep="last").reset_index(
                    drop=True
                )
            else:
                self._rows_df = incoming.reset_index(drop=True)
        else:
            combined = pd.concat([self._rows_df, incoming], ignore_index=True)
            if self.primary_key:
                combined = combined.drop_duplicates(subset=self.primary_key, keep="last")
            self._rows_df = combined.reset_index(drop=True)

    def read(
        self,
        start_time: Any = None,
        end_time: Any = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self._raise_if_failed("fg_read")
        self.read_calls.append(
            {"start_time": start_time, "end_time": end_time, "limit": limit, "kwargs": dict(kwargs)}
        )
        if self._rows_df is None or self._rows_df.empty:
            return pd.DataFrame()
        df = self._rows_df.copy()
        if self.event_time and self.event_time in df.columns:
            ts = pd.to_datetime(df[self.event_time], utc=True)
            if start_time is not None:
                df = df.loc[ts >= pd.to_datetime(start_time, utc=True)]
                ts = pd.to_datetime(df[self.event_time], utc=True)
            if end_time is not None:
                df = df.loc[ts <= pd.to_datetime(end_time, utc=True)]
        if limit is not None:
            df = df.head(int(limit))
        return df.reset_index(drop=True)


class FakeHopsworksFeatureStore:
    """Contract-enforcing Hopsworks Feature Store fake."""

    def __init__(self, *, fail_rules: dict[tuple[str, str], Exception] | None = None) -> None:
        self.feature_groups: dict[tuple[str, int], FakeHopsworksFeatureGroup] = {}
        self.fail_rules: dict[tuple[str, str], Exception] = dict(fail_rules or {})

    def get_or_create_feature_group(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        primary_key: list[str] | None = None,
        event_time: str | None = None,
        online_enabled: bool = False,
        **kwargs: Any,
    ) -> FakeHopsworksFeatureGroup:
        del kwargs
        key = (name, int(version))
        if key not in self.feature_groups:
            self.feature_groups[key] = FakeHopsworksFeatureGroup(
                name=name,
                version=int(version),
                description=description,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=online_enabled,
                fail_rules=self.fail_rules,
            )
        else:
            fg = self.feature_groups[key]
            if primary_key is not None:
                fg.primary_key = list(primary_key)
            if event_time is not None:
                fg.event_time = event_time
            fg.online_enabled = online_enabled
        return self.feature_groups[key]

    def get_feature_group(self, name: str, version: int = 1) -> FakeHopsworksFeatureGroup:
        exc = self.fail_rules.get(("fg_get", name)) or self.fail_rules.get(("fg_get", "*"))
        if exc is not None:
            raise exc
        key = (name, int(version))
        if key not in self.feature_groups:
            raise RuntimeError(f"Feature group {name!r} version {version} does not exist.")
        return self.feature_groups[key]


class FakeHopsworksModel:
    """Fake `hsml.model.Model` supporting artifact save/download and training_metrics."""

    def __init__(
        self,
        registry: FakeHopsworksModelRegistry,
        *,
        name: str,
        version: int,
        training_metrics: dict[str, Any] | None = None,
        description: str | None = None,
        tags: Any = None,
        model_schema: Any = None,
        input_example: Any = None,
    ) -> None:
        self._registry = registry
        self.name = name
        self.version = int(version)
        self.training_metrics: dict[str, Any] = dict(training_metrics or {})
        self.description = description
        self.tags = tags
        self.model_schema = model_schema
        self.input_example = input_example
        self._saved_dir: Path | None = None

    def save(
        self,
        model_path: str,
        await_registration: int = 480,
        keep_original_files: bool = False,
        upload_configuration: dict[str, Any] | None = None,
    ) -> FakeHopsworksModel:
        del await_registration, keep_original_files, upload_configuration
        self._registry._raise_if_failed("mr_save", self.name)
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model path does not exist: {src}")
        dest = self._registry._artifact_root / self.name / f"v{self.version}"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src.name)
        self._saved_dir = dest
        self._registry._register_model(self)
        return self

    def download(self, local_path: str | None = None) -> str:
        self._registry._raise_if_failed("mr_read", self.name)
        if self._saved_dir is None or not self._saved_dir.exists():
            raise RuntimeError(f"Model {self.name} v{self.version} has no saved artifacts.")
        dest = Path(local_path) if local_path else Path(tempfile.mkdtemp(prefix=f"hw_{self.name}_"))
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self._saved_dir, dest)
        return str(dest)


class FakeHopsworksPythonModelApi:
    """Fake `mr.python` factory matching `hsml.python.model_api`."""

    def __init__(self, registry: FakeHopsworksModelRegistry) -> None:
        self._registry = registry

    def create_model(
        self,
        name: str,
        version: int | None = None,
        metrics: dict[str, Any] | None = None,
        description: str | None = None,
        input_example: Any = None,
        model_schema: Any = None,
        feature_view: Any = None,
        training_dataset_version: int | None = None,
        tags: Any = None,
    ) -> FakeHopsworksModel:
        del feature_view, training_dataset_version
        metrics_clean = dict(metrics or {})
        try:
            json.dumps(metrics_clean, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractViolationError(f"Model metrics failed strict JSON check: {exc}") from exc
        existing = self._registry.get_models(name)
        resolved_version = int(version) if version is not None else (
            max((m.version for m in existing), default=0) + 1
        )
        return FakeHopsworksModel(
            self._registry,
            name=name,
            version=resolved_version,
            training_metrics=metrics_clean,
            description=description,
            tags=tags,
            model_schema=model_schema,
            input_example=input_example,
        )


class FakeHopsworksModelRegistry:
    """Contract-enforcing Hopsworks Model Registry fake."""

    def __init__(
        self,
        *,
        artifact_root: Path | None = None,
        fail_rules: dict[tuple[str, str], Exception] | None = None,
    ) -> None:
        self._models: dict[str, list[FakeHopsworksModel]] = {}
        self._artifact_root = artifact_root or Path(tempfile.mkdtemp(prefix="fake_hw_mr_"))
        self.fail_rules: dict[tuple[str, str], Exception] = dict(fail_rules or {})
        self._python_api = FakeHopsworksPythonModelApi(self)

    @property
    def python(self) -> FakeHopsworksPythonModelApi:
        return self._python_api

    def _raise_if_failed(self, op: str, name: str) -> None:
        exc = self.fail_rules.get((op, name)) or self.fail_rules.get((op, "*"))
        if exc is not None:
            raise exc

    def _register_model(self, model: FakeHopsworksModel) -> None:
        bucket = self._models.setdefault(model.name, [])
        bucket[:] = [m for m in bucket if m.version != model.version]
        bucket.append(model)
        bucket.sort(key=lambda m: m.version)

    def get_models(self, name: str) -> list[FakeHopsworksModel]:
        self._raise_if_failed("mr_read", name)
        return list(self._models.get(name, []))

    def get_model(self, name: str, version: int | None = None) -> FakeHopsworksModel | None:
        self._raise_if_failed("mr_read", name)
        models = self._models.get(name, [])
        if not models:
            return None
        if version is None:
            return models[-1]
        for m in models:
            if m.version == int(version):
                return m
        return None

    def get_best_model(
        self,
        name: str,
        metric: str,
        direction: str = "min",
    ) -> FakeHopsworksModel | None:
        self._raise_if_failed("mr_read", name)
        candidates = [m for m in self._models.get(name, []) if metric in m.training_metrics]
        if not candidates:
            return None
        if direction == "min":
            return min(candidates, key=lambda m: (float(m.training_metrics[metric]), -m.version))
        if direction == "max":
            return max(candidates, key=lambda m: (float(m.training_metrics[metric]), m.version))
        raise ValueError(f"Unsupported direction: {direction!r}")


@dataclass
class FakeHopsworksProject:
    """Fake Hopsworks Project exposing Feature Store and Model Registry."""

    artifact_root: Path | None = None
    fail_rules: dict[tuple[str, str], Exception] = field(default_factory=dict)
    feature_store: FakeHopsworksFeatureStore = field(init=False)
    model_registry: FakeHopsworksModelRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.feature_store = FakeHopsworksFeatureStore(fail_rules=self.fail_rules)
        self.model_registry = FakeHopsworksModelRegistry(
            artifact_root=self.artifact_root,
            fail_rules=self.fail_rules,
        )

    def get_feature_store(self, name: str | None = None) -> FakeHopsworksFeatureStore:
        del name
        return self.feature_store

    def get_model_registry(self) -> FakeHopsworksModelRegistry:
        return self.model_registry

    def attach_to_storage(
        self,
        storage: Any,
        monkeypatch: pytest.MonkeyPatch,
        fake_supabase: FakeSupabaseClient | None = None,
    ) -> None:
        monkeypatch.setattr(storage, "_get_hopsworks_project", lambda: self)
        if fake_supabase is not None:
            monkeypatch.setattr(storage, "_get_supabase", lambda: fake_supabase)
