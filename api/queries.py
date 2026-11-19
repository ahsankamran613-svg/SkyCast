"""Lightweight, ML-free read-only query helpers for Vercel FastAPI endpoints."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skycast.config.settings import Settings, get_settings

WEATHER_OVERVIEW_COLUMNS: tuple[str, ...] = (
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
    "created_at",
    "updated_at",
)

WEATHER_OVERVIEW_SELECT: str = ",".join(WEATHER_OVERVIEW_COLUMNS)


class ReadStoreError(RuntimeError):
    """Raised when the read layer is unconfigured or a storage query fails."""


def sanitize_json_scalar(value: Any) -> Any:
    """Recursively convert values into strict JSON-safe primitives (`NaN`/`Inf` -> `None`)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        rounded = round(value, 6)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, datetime):
        dt = (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
        return dt.isoformat()
    if isinstance(value, (list, tuple)):
        return [sanitize_json_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): sanitize_json_scalar(v) for k, v in value.items()}

    # Duck-type numpy/pandas scalars without importing numpy or pandas.
    if type(value).__name__ == "NaTType":
        return None
    if hasattr(value, "to_pydatetime") and hasattr(value, "isoformat"):
        try:
            iso = value.isoformat()
            return None if iso == "NaT" else iso
        except Exception:
            return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return sanitize_json_scalar(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return sanitize_json_scalar(value.tolist())
        except Exception:
            pass
    return value


def parse_json_or_default(raw: Any, default: Any) -> Any:
    """Parse JSON text or coerce sequence/mapping values into expected container types."""
    if raw is None:
        return default
    if isinstance(default, list):
        if isinstance(raw, list):
            return raw
        if isinstance(raw, (tuple, set)):
            return list(raw)
    elif isinstance(raw, type(default)):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, type(default)) else default
        except json.JSONDecodeError:
            return default
    return default


def parse_utc_datetime(value: Any) -> datetime | None:
    """Parse ISO string or datetime into a timezone-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        return (
            dt.replace(tzinfo=timezone.utc)
            if dt.tzinfo is None
            else dt.astimezone(timezone.utc)
        )
    return None


def utc_hour_key(value: Any) -> str | None:
    """Normalize a timestamp into a canonical UTC hour bucket string."""
    dt = parse_utc_datetime(value)
    if dt is None:
        return None
    floored = dt.replace(minute=0, second=0, microsecond=0)
    return floored.isoformat()


def _coerce_bool(value: Any, *, default: bool | None = False) -> bool | None:
    if value is None:
        return default
    return bool(value)


class ReadOnlyStore:
    """Lightweight read-only facade over Supabase Postgres and Local SQLite."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._supabase_client: Any = None

    @property
    def local_db_path(self) -> Path:
        return self.settings.skycast_local_data_dir / "skycast.db"

    @property
    def storage_mode(self) -> str:
        if self.settings.is_offline_mode:
            return "local_offline"
        if self.settings.has_hopsworks_credentials and self.settings.has_supabase_credentials:
            return "cloud_dual"
        if self.settings.has_supabase_credentials:
            return "cloud_supabase"
        if self.settings.has_hopsworks_credentials:
            return "cloud_hopsworks"
        return "unconfigured"

    def _require_configured(self) -> None:
        mode = self.storage_mode
        if mode == "unconfigured":
            raise ReadStoreError(
                "No storage backend is active for mode 'unconfigured': "
                "configure Supabase credentials or enable offline mode."
            )
        if mode == "cloud_hopsworks":
            raise ReadStoreError(
                "FastAPI read layer requires Supabase credentials or local offline mode; "
                "Hopsworks-only mode is unsupported in the lightweight API."
            )

    def _use_sqlite(self) -> bool:
        self._require_configured()
        return self.settings.is_offline_mode or not self.settings.has_supabase_credentials

    def _sqlite_ready(self) -> bool:
        return self.local_db_path.is_file() and self.local_db_path.stat().st_size > 0

    @contextmanager
    def _sqlite_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.local_db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _get_supabase(self) -> Any:
        if self._supabase_client is None:
            from supabase import create_client

            self._supabase_client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_role_key,
            )
        return self._supabase_client

    def _read_supabase_paginated(
        self,
        table_name: str,
        *,
        columns: str = "*",
        filters: Sequence[tuple[str, str, Any]] = (),
        orders: Sequence[tuple[str, bool]] = (),
        limit: int | None = None,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        try:
            client = self._get_supabase()
            order_specs = list(orders)
            if table_name != "weather_features_hourly" and not any(
                col == "id" for col, _ in order_specs
            ):
                order_specs.append(("id", False))

            rows: list[dict[str, Any]] = []
            offset = 0

            def _build_query() -> Any:
                q = client.table(table_name).select(columns)
                for op, col, val in filters:
                    if op == "eq":
                        q = q.eq(col, val)
                    elif op == "gte":
                        q = q.gte(col, val)
                    elif op == "lte":
                        q = q.lte(col, val)
                    else:
                        raise ValueError(f"Unsupported filter op: {op}")
                for col, desc in order_specs:
                    q = q.order(col, desc=desc)
                return q

            while True:
                remaining = (int(limit) - len(rows)) if limit is not None else page_size
                if remaining <= 0:
                    break
                step = min(page_size, remaining)
                response = _build_query().range(offset, offset + step - 1).execute()
                batch = list(response.data or [])
                rows.extend(batch)
                if len(batch) < step:
                    break
                offset += step
            return rows
        except ReadStoreError:
            raise
        except Exception as exc:
            raise ReadStoreError(f"Supabase {table_name} read failed: {exc}") from exc

    def _query_table(
        self,
        table_name: str,
        *,
        columns: str = "*",
        filters: Sequence[tuple[str, str, Any]] = (),
        sqlite_order_by: str,
        supabase_orders: Sequence[tuple[str, bool]],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._use_sqlite():
            if not self._sqlite_ready():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            op_map = {"eq": "=", "gte": ">=", "lte": "<="}
            for op, col, val in filters:
                sql_op = op_map.get(op)
                if sql_op is None:
                    raise ValueError(f"Unsupported filter op: {op}")
                clauses.append(f"{col} {sql_op} ?")
                params.append(int(val) if isinstance(val, bool) else val)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
            sql = f"SELECT {columns} FROM {table_name} {where_sql} ORDER BY {sqlite_order_by} {limit_sql}"
            try:
                with self._sqlite_connection() as conn:
                    raw_rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in raw_rows]
            except sqlite3.Error as exc:
                raise ReadStoreError(f"SQLite {table_name} read failed: {exc}") from exc

        return self._read_supabase_paginated(
            table_name,
            columns=columns,
            filters=filters,
            orders=supabase_orders,
            limit=limit,
        )

    def _resolve_latest_run_id(
        self,
        table_name: str,
        order_col: str,
        *,
        run_col: str = "run_id",
        eq_filters: Sequence[tuple[str, str, Any]] = (),
    ) -> str | None:
        rows = self._query_table(
            table_name,
            columns=f"{run_col},{order_col},id",
            filters=eq_filters,
            sqlite_order_by=f"{order_col} DESC, id DESC",
            supabase_orders=((order_col, True), ("id", True)),
            limit=1,
        )
        if not rows or rows[0].get(run_col) is None:
            return None
        return str(rows[0][run_col])

    # ------------------------------------------------------------------
    # Table-specific read helpers
    # ------------------------------------------------------------------

    def read_latest_pipeline_runs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._query_table(
            "pipeline_runs",
            sqlite_order_by="started_at_utc DESC, id DESC",
            supabase_orders=(("started_at_utc", True), ("id", True)),
            limit=limit,
        )
        for row in rows:
            row["metadata_json"] = parse_json_or_default(row.get("metadata_json"), {})
        return rows

    def read_latest_imputation_audit(self) -> dict[str, Any] | None:
        rows = self._query_table(
            "pipeline_runs",
            columns="metadata_json,started_at_utc,id",
            filters=(("eq", "status", "completed"),),
            sqlite_order_by="started_at_utc DESC, id DESC",
            supabase_orders=(("started_at_utc", True), ("id", True)),
            limit=10,
        )
        for row in rows:
            meta = parse_json_or_default(row.get("metadata_json"), {})
            if isinstance(meta, Mapping) and "imputation_audit" in meta:
                audit = meta["imputation_audit"]
                return dict(audit) if isinstance(audit, Mapping) else None
        return None

    def read_weather_features_recent(
        self,
        *,
        limit_hours: int | None = 24,
        start_utc: str | None = None,
        end_utc: str | None = None,
        columns: str = WEATHER_OVERVIEW_SELECT,
    ) -> list[dict[str, Any]]:
        filters: list[tuple[str, str, Any]] = []
        if start_utc is not None:
            filters.append(("gte", "timestamp_utc", start_utc))
        if end_utc is not None:
            filters.append(("lte", "timestamp_utc", end_utc))

        rows = self._query_table(
            "weather_features_hourly",
            columns=columns,
            filters=filters,
            sqlite_order_by="timestamp_utc DESC",
            supabase_orders=(("timestamp_utc", True),),
            limit=limit_hours,
        )
        for row in rows:
            if "sources_available" in row:
                row["sources_available"] = [
                    str(s) for s in parse_json_or_default(row.get("sources_available"), [])
                ]
            if "imputed_columns" in row:
                row["imputed_columns"] = [
                    str(c) for c in parse_json_or_default(row.get("imputed_columns"), [])
                ]
            if "features_json" in row:
                row["features_json"] = parse_json_or_default(row.get("features_json"), {})
            if "is_imputed" in row:
                row["is_imputed"] = bool(_coerce_bool(row.get("is_imputed"), default=False))
            for bcol in ("is_weekend", "is_lahore_smog_season", "is_monsoon_season"):
                if bcol in row:
                    row[bcol] = _coerce_bool(row.get(bcol), default=None)
        rows.reverse()
        return rows

    def read_forecast_ledger(
        self,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        model_family: str | None = None,
        horizon_hour: int | None = None,
        latest_run_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            eq_filters = [("eq", "metric", metric)] if metric is not None else ()
            resolved_run_id = self._resolve_latest_run_id(
                "forecast_ledger", "issued_at_utc", eq_filters=eq_filters
            )
            if resolved_run_id is None:
                return []

        filters: list[tuple[str, str, Any]] = []
        if resolved_run_id is not None:
            filters.append(("eq", "run_id", resolved_run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))
        if model_family is not None:
            filters.append(("eq", "model_family", model_family))
        if horizon_hour is not None:
            filters.append(("eq", "horizon_hour", int(horizon_hour)))

        return self._query_table(
            "forecast_ledger",
            filters=filters,
            sqlite_order_by="issued_at_utc DESC, horizon_hour ASC, metric ASC, id ASC",
            supabase_orders=(
                ("issued_at_utc", True),
                ("horizon_hour", False),
                ("metric", False),
                ("id", False),
            ),
            limit=limit,
        )

    def read_model_registry_metrics(
        self,
        *,
        train_run_id: str | None = None,
        metric: str | None = None,
        latest_run_only: bool = True,
    ) -> list[dict[str, Any]]:
        resolved_run_id = train_run_id
        if latest_run_only and resolved_run_id is None:
            eq_filters = [("eq", "metric", metric)] if metric is not None else ()
            resolved_run_id = self._resolve_latest_run_id(
                "model_registry_metrics",
                "trained_at_utc",
                run_col="train_run_id",
                eq_filters=eq_filters,
            )
            if resolved_run_id is None:
                return []

        filters: list[tuple[str, str, Any]] = []
        if resolved_run_id is not None:
            filters.append(("eq", "train_run_id", resolved_run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))

        rows = self._query_table(
            "model_registry_metrics",
            filters=filters,
            sqlite_order_by="trained_at_utc DESC, metric ASC, val_rmse ASC, id ASC",
            supabase_orders=(
                ("trained_at_utc", True),
                ("metric", False),
                ("val_rmse", False),
                ("id", False),
            ),
        )
        for row in rows:
            row["is_champion"] = bool(_coerce_bool(row.get("is_champion"), default=False))
            row["promoted_to_registry"] = bool(
                _coerce_bool(row.get("promoted_to_registry"), default=False)
            )
            row["hyperparameters"] = parse_json_or_default(row.get("hyperparameters"), {})
        return rows

    def read_model_comparison_series(
        self,
        *,
        train_run_id: str | None = None,
        metric: str | None = None,
        latest_run_only: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        resolved_run_id = train_run_id
        if latest_run_only and resolved_run_id is None:
            eq_filters = [("eq", "metric", metric)] if metric is not None else ()
            resolved_run_id = self._resolve_latest_run_id(
                "model_comparison_series",
                "created_at",
                run_col="train_run_id",
                eq_filters=eq_filters,
            )
            if resolved_run_id is None:
                return []

        filters: list[tuple[str, str, Any]] = []
        if resolved_run_id is not None:
            filters.append(("eq", "train_run_id", resolved_run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))

        return self._query_table(
            "model_comparison_series",
            filters=filters,
            sqlite_order_by="timestamp_utc ASC, id ASC",
            supabase_orders=(("timestamp_utc", False), ("id", False)),
            limit=limit,
        )

    def read_shap_lime_explanations(
        self,
        *,
        train_run_id: str | None = None,
        metric: str | None = None,
        method: str | None = None,
        scope: str | None = None,
        latest_run_only: bool = True,
    ) -> list[dict[str, Any]]:
        resolved_run_id = train_run_id
        if latest_run_only and resolved_run_id is None:
            eq_filters = [("eq", "metric", metric)] if metric is not None else ()
            resolved_run_id = self._resolve_latest_run_id(
                "shap_lime_explanations",
                "computed_at_utc",
                run_col="train_run_id",
                eq_filters=eq_filters,
            )
            if resolved_run_id is None:
                return []

        filters: list[tuple[str, str, Any]] = []
        if resolved_run_id is not None:
            filters.append(("eq", "train_run_id", resolved_run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))
        if method is not None:
            filters.append(("eq", "method", method))
        if scope is not None:
            filters.append(("eq", "scope", scope))

        rows = self._query_table(
            "shap_lime_explanations",
            filters=filters,
            sqlite_order_by="computed_at_utc DESC, metric ASC, method ASC, scope ASC, id ASC",
            supabase_orders=(
                ("computed_at_utc", True),
                ("metric", False),
                ("method", False),
                ("scope", False),
                ("id", False),
            ),
        )
        for row in rows:
            row["feature_attributions"] = parse_json_or_default(
                row.get("feature_attributions"), []
            )
        return rows

    def read_latest_eda_summary(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        filters: list[tuple[str, str, Any]] = [("eq", "run_id", run_id)] if run_id else []
        rows = self._query_table(
            "eda_summaries",
            filters=filters,
            sqlite_order_by="computed_at_utc DESC, id DESC",
            supabase_orders=(("computed_at_utc", True), ("id", True)),
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        row["summary_json"] = parse_json_or_default(row.get("summary_json"), {})
        return row

    def read_active_alerts(self) -> list[dict[str, Any]]:
        latest_alert_run = self._resolve_latest_run_id(
            "weather_alerts",
            "timestamp",
            eq_filters=(("eq", "is_active", True),),
        )
        if latest_alert_run is None:
            return []

        rows = self._query_table(
            "weather_alerts",
            filters=(("eq", "run_id", latest_alert_run), ("eq", "is_active", True)),
            sqlite_order_by="timestamp DESC, id DESC",
            supabase_orders=(("timestamp", True), ("id", True)),
        )
        if not rows:
            return []

        # Suppress stale alerts if a strictly newer alert_agent run explicitly cleared hazards.
        latest_trace = self._query_table(
            "agent_graph_trace",
            filters=(("eq", "node_name", "alert_agent"),),
            sqlite_order_by="started_at DESC, id DESC",
            supabase_orders=(("started_at", True), ("id", True)),
            limit=1,
        )
        if latest_trace:
            trace = latest_trace[0]
            trace_dt = parse_utc_datetime(trace.get("started_at"))
            alert_dt = parse_utc_datetime(rows[0].get("timestamp"))
            if (
                trace.get("run_id") != latest_alert_run
                and trace.get("branch_decision") == "no_hazards_triggered"
                and trace_dt is not None
                and alert_dt is not None
                and trace_dt > alert_dt
            ):
                return []

        for row in rows:
            row["is_active"] = bool(_coerce_bool(row.get("is_active"), default=True))
        return rows

    def read_latest_weather_report(self) -> dict[str, Any] | None:
        rows = self._query_table(
            "weather_reports",
            sqlite_order_by="issued_at_utc DESC, id DESC",
            supabase_orders=(("issued_at_utc", True), ("id", True)),
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        row["highlights"] = parse_json_or_default(row.get("highlights"), [])
        row["fallback_used"] = bool(_coerce_bool(row.get("fallback_used"), default=False))
        return row

    def read_agent_call_log(
        self,
        *,
        run_id: str | None = None,
        agent_name: str | None = None,
        status: str | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[tuple[str, str, Any]] = []
        if run_id is not None:
            filters.append(("eq", "run_id", run_id))
        if agent_name is not None:
            filters.append(("eq", "agent_name", agent_name))
        if status is not None:
            filters.append(("eq", "status", status))
        if provider is not None:
            filters.append(("eq", "provider", provider))

        sqlite_order = (
            "attempt_order ASC, id ASC"
            if run_id is not None
            else "timestamp_utc DESC, attempt_order ASC, id DESC"
        )
        supabase_orders: tuple[tuple[str, bool], ...] = (
            (("attempt_order", False), ("id", False))
            if run_id is not None
            else (("timestamp_utc", True), ("attempt_order", False), ("id", True))
        )
        rows = self._query_table(
            "agent_call_log",
            filters=filters,
            sqlite_order_by=sqlite_order,
            supabase_orders=supabase_orders,
            limit=limit,
        )
        for row in rows:
            row["fallback_used"] = bool(_coerce_bool(row.get("fallback_used"), default=False))
        return rows

    def read_drift_log(
        self,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        drift_type: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[tuple[str, str, Any]] = []
        if run_id is not None:
            filters.append(("eq", "run_id", run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))
        if drift_type is not None:
            filters.append(("eq", "drift_type", drift_type))
        if status is not None:
            filters.append(("eq", "status", status))

        rows = self._query_table(
            "drift_log",
            filters=filters,
            sqlite_order_by="timestamp DESC, id DESC",
            supabase_orders=(("timestamp", True), ("id", True)),
            limit=limit,
        )
        for row in rows:
            row["retrain_triggered"] = bool(
                _coerce_bool(row.get("retrain_triggered"), default=False)
            )
            row["details_json"] = parse_json_or_default(row.get("details_json"), {})
        return rows

    def read_agent_graph_trace(
        self,
        *,
        run_id: str | None = None,
        node_name: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[tuple[str, str, Any]] = []
        if run_id is not None:
            filters.append(("eq", "run_id", run_id))
        if node_name is not None:
            filters.append(("eq", "node_name", node_name))
        if status is not None:
            filters.append(("eq", "status", status))

        sqlite_order = (
            "step_order ASC, id ASC"
            if run_id is not None
            else "started_at DESC, step_order ASC, id ASC"
        )
        supabase_orders: tuple[tuple[str, bool], ...] = (
            (("step_order", False), ("id", False))
            if run_id is not None
            else (("started_at", True), ("step_order", False), ("id", True))
        )
        rows = self._query_table(
            "agent_graph_trace",
            filters=filters,
            sqlite_order_by=sqlite_order,
            supabase_orders=supabase_orders,
            limit=limit,
        )
        for row in rows:
            row["summary_json"] = parse_json_or_default(row.get("summary_json"), {})
        return rows
