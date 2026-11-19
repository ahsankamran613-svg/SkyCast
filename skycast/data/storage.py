"""Dual persistence layer: Hopsworks Feature Store, Supabase Postgres, and Local SQLite fallback."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from skycast.config.settings import (
    MODEL_FAMILIES,
    POLLUTANT_COLUMNS,
    Settings,
    TARGET_METRICS,
    get_settings,
)
from skycast.features.engineer import (
    build_features_json_payload,
    get_model_feature_columns,
    sanitize_json_scalar,
)

logger = logging.getLogger(__name__)

VALID_EXPLANATION_METHODS: frozenset[str] = frozenset({"shap", "lime"})
VALID_EXPLANATION_SCOPES: frozenset[str] = frozenset({"global", "local"})
VALID_PIPELINE_TYPES: frozenset[str] = frozenset({"backfill", "hourly_ingest", "training"})
VALID_PIPELINE_STATUSES: frozenset[str] = frozenset({"running", "completed", "failed"})
VALID_ALERT_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})
VALID_REPORT_TRIGGER_REASONS: frozenset[str] = frozenset(
    {"daily_schedule", "hazard_alert", "manual", "backfill_seed"}
)
VALID_AGENT_CALL_STATUSES: frozenset[str] = frozenset({"success", "failed"})
VALID_GRAPH_TRACE_STATUSES: frozenset[str] = frozenset({"completed", "skipped", "failed"})
VALID_DRIFT_TYPES: frozenset[str] = frozenset({"data_drift", "model_drift"})
VALID_DRIFT_STATUSES: frozenset[str] = frozenset({"no_drift", "drift_detected", "check_failed"})
_TARGET_METRIC_SET: frozenset[str] = frozenset(TARGET_METRICS)
_MODEL_FAMILY_SET: frozenset[str] = frozenset(MODEL_FAMILIES)


class StorageError(RuntimeError):
    """Raised when a configured storage backend fails."""


class StorageValidationError(StorageError, ValueError):
    """Raised when a record violates SQL schema CHECK, NOT NULL, finite, or UNIQUE constraints."""


_SQLITE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS weather_features_hourly (
        timestamp_utc TEXT PRIMARY KEY,
        us_aqi REAL,
        temperature_2m REAL,
        precipitation_mm REAL,
        precipitation_prob REAL,
        relative_humidity_2m REAL,
        uv_index REAL,
        wind_speed_10m REAL,
        wind_direction_10m REAL,
        apparent_temperature REAL,
        weather_code INTEGER,
        condition_symbol TEXT,
        surface_pressure REAL,
        cloud_cover REAL,
        pm2_5 REAL,
        pm10 REAL,
        ozone REAL,
        nitrogen_dioxide REAL,
        sulphur_dioxide REAL,
        carbon_monoxide REAL,
        temp_openmeteo REAL,
        temp_metno REAL,
        humidity_openmeteo REAL,
        humidity_metno REAL,
        precip_openmeteo REAL,
        precip_metno REAL,
        wind_speed_openmeteo REAL,
        wind_speed_metno REAL,
        wind_dir_openmeteo REAL,
        wind_dir_metno REAL,
        uv_openmeteo REAL,
        uv_metno REAL,
        source_disagreement_temp REAL,
        sources_available TEXT NOT NULL DEFAULT '[]',
        is_imputed INTEGER NOT NULL DEFAULT 0,
        imputation_tier TEXT NOT NULL DEFAULT 'none',
        imputed_columns TEXT NOT NULL DEFAULT '[]',
        wind_dir_sin REAL,
        wind_dir_cos REAL,
        hour_sin REAL,
        hour_cos REAL,
        is_weekend INTEGER,
        is_lahore_smog_season INTEGER,
        is_monsoon_season INTEGER,
        features_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_weather_features_ts_desc
        ON weather_features_hourly (timestamp_utc DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS forecast_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        issued_at_utc TEXT NOT NULL,
        target_timestamp_utc TEXT NOT NULL,
        horizon_hour INTEGER NOT NULL CHECK (horizon_hour BETWEEN 1 AND 24),
        metric TEXT NOT NULL,
        model_family TEXT NOT NULL,
        model_version TEXT,
        predicted_value REAL NOT NULL,
        lower_bound REAL NOT NULL,
        upper_bound REAL NOT NULL,
        naive_baseline_value REAL,
        actual_value REAL,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, metric, model_family, horizon_hour)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_forecast_ledger_metric_issued_desc
        ON forecast_ledger (metric, issued_at_utc DESC, horizon_hour ASC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_forecast_ledger_target_ts
        ON forecast_ledger (target_timestamp_utc DESC, metric)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_forecast_ledger_run_id
        ON forecast_ledger (run_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS model_registry_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        train_run_id TEXT NOT NULL,
        trained_at_utc TEXT NOT NULL,
        metric TEXT NOT NULL,
        model_family TEXT NOT NULL,
        model_version TEXT,
        val_rmse REAL NOT NULL,
        val_mae REAL NOT NULL,
        val_r2 REAL NOT NULL,
        test_rmse REAL NOT NULL,
        test_mae REAL NOT NULL,
        test_r2 REAL NOT NULL,
        improvement_over_naive_pct REAL NOT NULL DEFAULT 0.0,
        is_champion INTEGER NOT NULL DEFAULT 0,
        promoted_to_registry INTEGER NOT NULL DEFAULT 0,
        hyperparameters TEXT NOT NULL DEFAULT '{}',
        training_samples INTEGER,
        created_at TEXT NOT NULL,
        UNIQUE (train_run_id, metric, model_family)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_model_registry_metrics_lookup
        ON model_registry_metrics (metric, trained_at_utc DESC, is_champion)
    """,
    """
    CREATE TABLE IF NOT EXISTS model_comparison_series (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        train_run_id TEXT NOT NULL,
        metric TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,
        actual REAL,
        sarimax REAL,
        random_forest REAL,
        ridge REAL,
        lstm REAL,
        naive_24h REAL,
        created_at TEXT NOT NULL,
        UNIQUE (train_run_id, metric, timestamp_utc)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_model_comparison_series_lookup
        ON model_comparison_series (metric, train_run_id, timestamp_utc ASC)
    """,
    """
    CREATE TABLE IF NOT EXISTS shap_lime_explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        train_run_id TEXT NOT NULL,
        computed_at_utc TEXT NOT NULL,
        metric TEXT NOT NULL,
        model_family TEXT NOT NULL,
        method TEXT NOT NULL CHECK (method IN ('shap', 'lime')),
        scope TEXT NOT NULL CHECK (scope IN ('global', 'local')),
        base_value REAL,
        predicted_value REAL,
        reference_timestamp_utc TEXT,
        feature_attributions TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        UNIQUE (train_run_id, metric, method, scope)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_shap_lime_lookup
        ON shap_lime_explanations (metric, method, computed_at_utc DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        target_timestamp_utc TEXT,
        metric TEXT NOT NULL,
        severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
        title TEXT NOT NULL,
        trigger_value REAL NOT NULL,
        threshold REAL NOT NULL,
        advisory_general TEXT NOT NULL,
        advisory_sensitive TEXT NOT NULL,
        advisory_outdoor_workers TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        llm_provider TEXT,
        llm_model TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL UNIQUE,
        issued_at_utc TEXT NOT NULL,
        trigger_reason TEXT NOT NULL CHECK (
            trigger_reason IN ('daily_schedule', 'hazard_alert', 'manual', 'backfill_seed')
        ),
        headline TEXT NOT NULL,
        summary TEXT NOT NULL,
        bulletin_markdown TEXT NOT NULL,
        highlights TEXT NOT NULL DEFAULT '[]',
        active_alert_count INTEGER NOT NULL DEFAULT 0,
        llm_provider TEXT,
        llm_model TEXT,
        fallback_used INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_call_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        reasoning_effort TEXT,
        prompt TEXT NOT NULL,
        response TEXT,
        latency_ms REAL NOT NULL DEFAULT 0.0,
        status TEXT NOT NULL CHECK (status IN ('success', 'failed')),
        fallback_used INTEGER NOT NULL DEFAULT 0,
        attempt_order INTEGER NOT NULL DEFAULT 1,
        error_message TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_graph_trace (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        node_name TEXT NOT NULL,
        step_order INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('completed', 'skipped', 'failed')),
        started_at TEXT NOT NULL,
        duration_ms REAL NOT NULL DEFAULT 0.0,
        branch_decision TEXT,
        summary_json TEXT NOT NULL DEFAULT '{}',
        error_message TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (run_id, node_name, step_order)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS drift_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        timestamp TEXT NOT NULL,
        metric TEXT NOT NULL,
        drift_type TEXT NOT NULL CHECK (drift_type IN ('data_drift', 'model_drift')),
        test_name TEXT,
        score REAL,
        secondary_score REAL,
        threshold REAL NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('no_drift', 'drift_detected', 'check_failed')),
        retrain_triggered INTEGER NOT NULL DEFAULT 0,
        action_taken TEXT NOT NULL,
        error_message TEXT,
        details_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eda_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL UNIQUE,
        computed_at_utc TEXT NOT NULL,
        sample_count INTEGER NOT NULL DEFAULT 0,
        date_range_start TEXT,
        date_range_end TEXT,
        summary_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_eda_summaries_computed_desc
        ON eda_summaries (computed_at_utc DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL UNIQUE,
        pipeline_type TEXT NOT NULL CHECK (pipeline_type IN ('backfill', 'hourly_ingest', 'training')),
        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
        started_at_utc TEXT NOT NULL,
        completed_at_utc TEXT,
        storage_mode TEXT NOT NULL DEFAULT 'local_offline',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        error_message TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sqlite_pipeline_runs_started_desc
        ON pipeline_runs (started_at_utc DESC)
    """,
)

_WEATHER_FEATURE_SCALAR_COLS: tuple[str, ...] = (
    *TARGET_METRICS,
    "apparent_temperature",
    "surface_pressure",
    "cloud_cover",
    *POLLUTANT_COLUMNS,
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
    "wind_dir_sin",
    "wind_dir_cos",
    "hour_sin",
    "hour_cos",
)


def _utc_iso(value: datetime | pd.Timestamp | str) -> str:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    if ts.microsecond == 0:
        return ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _prepare_json_input(value: Any) -> Any:
    """Convert numpy arrays and plain datetimes before `sanitize_json_scalar`."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _prepare_json_input(value.item())
        return [_prepare_json_input(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _prepare_json_input(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prepare_json_input(v) for v in value]
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value
    if isinstance(value, datetime):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        dt = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        return dt.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _sanitize_json_container(
    value: Any,
    *,
    expected_type: type[dict] | type[list],
    field: str,
) -> tuple[Any, str]:
    if expected_type is list and isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, expected_type):
        raise StorageValidationError(
            f"Field '{field}' must be a {expected_type.__name__}, got {type(value).__name__}."
        )
    cleaned = sanitize_json_scalar(_prepare_json_input(value))
    try:
        serialized = json.dumps(cleaned, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StorageValidationError(f"Field '{field}' is not strict JSON-serializable: {exc}") from exc
    return cleaned, serialized


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StorageValidationError(f"Field '{field}' must be a non-empty string.")
    return value


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str):
        raise StorageValidationError(f"Field '{field}' must be a string or None.")
    return value


def _require_enum(value: Any, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise StorageValidationError(
            f"Invalid '{field}' value {value!r}; expected one of {sorted(allowed)}."
        )
    return value


def _require_int(
    value: Any,
    field: str,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise StorageValidationError(f"Field '{field}' must be an integer, got {value!r}.")
    int_val = int(value)
    if min_val is not None and int_val < min_val:
        raise StorageValidationError(f"Field '{field}' must be >= {min_val}, got {int_val}.")
    if max_val is not None and int_val > max_val:
        raise StorageValidationError(f"Field '{field}' must be <= {max_val}, got {int_val}.")
    return int_val


def _require_finite_float(
    value: Any,
    field: str,
    *,
    non_negative: bool = False,
) -> float:
    if value is None or isinstance(value, (bool, np.bool_, str)):
        raise StorageValidationError(f"Field '{field}' must be a finite number, got {value!r}.")
    try:
        raw_float = float(value)
    except (TypeError, ValueError) as exc:
        raise StorageValidationError(f"Field '{field}' must be a finite number, got {value!r}.") from exc
    if not math.isfinite(raw_float):
        raise StorageValidationError(f"Field '{field}' must be finite, got {value!r}.")
    if non_negative and raw_float < 0.0:
        raise StorageValidationError(f"Field '{field}' must be >= 0.0, got {raw_float}.")
    cleaned = sanitize_json_scalar(raw_float)
    if cleaned is None or not math.isfinite(float(cleaned)):
        raise StorageValidationError(f"Field '{field}' must be finite, got {value!r}.")
    res = float(cleaned)
    if res == 0.0:
        res = 0.0
    if non_negative and res < 0.0:
        raise StorageValidationError(f"Field '{field}' must be >= 0.0, got {res}.")
    return res


def _optional_finite_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, str)):
        raise StorageValidationError(f"Field '{field}' must be numeric or None, got {value!r}.")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        raw_float = float(value)
    except (TypeError, ValueError) as exc:
        raise StorageValidationError(f"Field '{field}' must be numeric or None, got {value!r}.") from exc
    if not math.isfinite(raw_float):
        return None
    cleaned = sanitize_json_scalar(raw_float)
    if cleaned is None:
        return None
    res = float(cleaned)
    return 0.0 if res == 0.0 else res


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise StorageValidationError(f"Field '{field}' must be a boolean, got {value!r}.")
    return bool(value)


def _require_utc_iso(value: Any, field: str) -> str:
    if not isinstance(value, (datetime, pd.Timestamp, str)):
        raise StorageValidationError(f"Field '{field}' must be a valid UTC timestamp, got {value!r}.")
    if isinstance(value, str) and not value.strip():
        raise StorageValidationError(f"Field '{field}' must be a non-empty UTC timestamp.")
    try:
        return _utc_iso(value)
    except Exception as exc:
        raise StorageValidationError(f"Invalid UTC timestamp for '{field}': {value!r}") from exc


def _assert_json_safe_record(record: Mapping[str, Any], table_name: str) -> None:
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StorageValidationError(
            f"Record for '{table_name}' failed strict JSON check: {exc}"
        ) from exc


def _records_to_hopsworks_forecast_ledger_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a schema-consistent DataFrame for Hopsworks `lahore_forecast_ledger`."""
    return pd.DataFrame(
        {
            "run_id": [str(r["run_id"]) for r in records],
            "issued_at_utc": pd.to_datetime([r["issued_at_utc"] for r in records], utc=True),
            "target_timestamp_utc": pd.to_datetime([r["target_timestamp_utc"] for r in records], utc=True),
            "horizon_hour": pd.Series([int(r["horizon_hour"]) for r in records], dtype="int64"),
            "metric": [str(r["metric"]) for r in records],
            "model_family": [str(r["model_family"]) for r in records],
            "model_version": [str(r["model_version"]) if r.get("model_version") is not None else "" for r in records],
            "predicted_value": pd.Series([float(r["predicted_value"]) for r in records], dtype="float64"),
            "lower_bound": pd.Series([float(r["lower_bound"]) for r in records], dtype="float64"),
            "upper_bound": pd.Series([float(r["upper_bound"]) for r in records], dtype="float64"),
            "naive_baseline_value": pd.Series(
                [float(r["naive_baseline_value"]) if r.get("naive_baseline_value") is not None else np.nan for r in records],
                dtype="float64",
            ),
            "actual_value": pd.Series(
                [float(r["actual_value"]) if r.get("actual_value") is not None else np.nan for r in records],
                dtype="float64",
            ),
        }
    )


def _parse_json_or_default(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(default, list):
        if isinstance(raw, list):
            return raw
        if isinstance(raw, np.ndarray):
            return raw.tolist()
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


def dataframe_to_supabase_weather_records(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Serialize an engineered weather DataFrame into `weather_features_hourly` rows."""
    if df.empty:
        return []
    cfg = settings or get_settings()
    records: list[dict[str, Any]] = []
    for row_dict in df.to_dict(orient="records"):
        ts_str = _utc_iso(row_dict["timestamp_utc"])
        payload: dict[str, Any] = {"timestamp_utc": ts_str}

        for col in _WEATHER_FEATURE_SCALAR_COLS:
            payload[col] = sanitize_json_scalar(row_dict.get(col))

        wmo = sanitize_json_scalar(row_dict.get("weather_code"))
        payload["weather_code"] = int(round(wmo)) if wmo is not None else None
        sym = row_dict.get("condition_symbol")
        payload["condition_symbol"] = str(sym) if sym is not None and not pd.isna(sym) else None

        sources_raw = _parse_json_or_default(row_dict.get("sources_available"), [])
        payload["sources_available"] = [str(s) for s in sources_raw]

        imputed_raw = _parse_json_or_default(row_dict.get("imputed_columns"), [])
        payload["imputed_columns"] = [str(c) for c in imputed_raw]

        payload["is_imputed"] = bool(row_dict.get("is_imputed", False))
        payload["imputation_tier"] = str(row_dict.get("imputation_tier") or "none")

        for bool_col in ("is_weekend", "is_lahore_smog_season", "is_monsoon_season"):
            val = row_dict.get(bool_col)
            payload[bool_col] = None if val is None or pd.isna(val) else bool(val)

        payload["features_json"] = build_features_json_payload(row_dict, settings=cfg)
        records.append(payload)

    return records


def dataframe_to_hopsworks_frame(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Build a schema-consistent DataFrame for Hopsworks `lahore_weather_features`."""
    cfg = settings or get_settings()
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    epoch_ms = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)

    cols: dict[str, Any] = {
        "timestamp_epoch_ms": epoch_ms.astype("int64"),
        "timestamp_utc": ts,
        "condition_symbol": (
            df["condition_symbol"].fillna("unknown").astype(str)
            if "condition_symbol" in df.columns
            else pd.Series("unknown", index=df.index, dtype=str)
        ),
        "sources_available": [
            json.dumps([str(s) for s in _parse_json_or_default(v, [])])
            for v in df.get("sources_available", [[] for _ in range(len(df))])
        ],
        "is_imputed": df.get("is_imputed", pd.Series(False, index=df.index)).astype(bool),
        "imputation_tier": df.get("imputation_tier", pd.Series("none", index=df.index)).astype(str),
        "imputed_columns": [
            json.dumps([str(c) for c in _parse_json_or_default(v, [])])
            for v in df.get("imputed_columns", [[] for _ in range(len(df))])
        ],
    }

    for raw_col in (
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
    ):
        if raw_col in df.columns:
            cols[raw_col] = pd.to_numeric(df[raw_col], errors="coerce").astype("float64")
        else:
            cols[raw_col] = pd.Series(float("nan"), index=df.index, dtype="float64")

    if "wind_direction_10m" in df.columns:
        cols["wind_direction_10m"] = pd.to_numeric(
            df["wind_direction_10m"], errors="coerce"
        ).astype("float64")

    for col in get_model_feature_columns(settings=cfg):
        if col in {"is_weekend", "is_lahore_smog_season", "is_monsoon_season"}:
            cols[col] = df[col].fillna(False).astype(bool) if col in df.columns else False
        elif col == "weather_code":
            if col in df.columns:
                wmo_num = pd.to_numeric(df[col], errors="coerce")
                missing_wmo = int(wmo_num.isna().sum())
                if missing_wmo > 0:
                    logger.warning(
                        "Hopsworks serialization found %d missing weather_code row(s); encoding as -1.",
                        missing_wmo,
                    )
                cols[col] = wmo_num.fillna(-1).astype("int64")
            else:
                cols[col] = pd.Series(-1, index=df.index, dtype="int64")
        else:
            cols[col] = (
                pd.to_numeric(df[col], errors="coerce").astype("float64")
                if col in df.columns
                else pd.Series(float("nan"), index=df.index, dtype="float64")
            )

    return pd.DataFrame(cols, index=df.index)


class SkyCastStorage:
    """Unified persistence facade across Local SQLite, Supabase Postgres, and Hopsworks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._supabase_client: Any = None
        self._hopsworks_project: Any = None
        self._initialized_sqlite_paths: set[Path] = set()

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

    @contextmanager
    def _sqlite_connection(self) -> Iterator[sqlite3.Connection]:
        self.local_db_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = self.local_db_path.resolve()
        conn = sqlite3.connect(self.local_db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            if resolved not in self._initialized_sqlite_paths:
                conn.execute("PRAGMA journal_mode=WAL")
                for stmt in _SQLITE_SCHEMA_STATEMENTS:
                    conn.execute(stmt)
                conn.commit()
                self._initialized_sqlite_paths.add(resolved)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_local_db(self) -> Path:
        """Ensure the local SQLite database and all 12 schema tables exist."""
        with self._sqlite_connection():
            pass
        return self.local_db_path

    def _require_configured(self) -> None:
        if self.storage_mode == "unconfigured":
            raise StorageError(
                "No storage backend is active for mode 'unconfigured': "
                "configure Supabase/Hopsworks credentials or enable offline mode."
            )

    def _get_supabase(self) -> Any:
        if self._supabase_client is None:
            from supabase import create_client

            self._supabase_client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_role_key,
            )
        return self._supabase_client

    def _get_hopsworks_project(self) -> Any:
        if self._hopsworks_project is None:
            import hopsworks

            self._hopsworks_project = hopsworks.login(
                host=self.settings.hopsworks_host,
                project=self.settings.hopsworks_project_name,
                api_key_value=self.settings.hopsworks_api_key,
            )
        return self._hopsworks_project

    def _get_hopsworks_feature_store(self) -> Any:
        return self._get_hopsworks_project().get_feature_store()

    def _get_hopsworks_model_registry(self) -> Any:
        return self._get_hopsworks_project().get_model_registry()

    def get_hopsworks_model_registry(self) -> Any:
        """Return the Hopsworks Model Registry handle when cloud Hopsworks is active."""
        self._require_configured()
        if self.settings.is_offline_mode or not self.settings.has_hopsworks_credentials:
            raise StorageError(
                "Hopsworks Model Registry is unavailable in mode "
                f"'{self.storage_mode}': configure Hopsworks credentials and disable offline mode."
            )
        try:
            return self._get_hopsworks_model_registry()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Hopsworks Model Registry access failed: {exc}") from exc

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
            if not any(col == "id" for col, _ in order_specs):
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
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Supabase {table_name} read failed: {exc}") from exc

    def _insert_supabase_records(
        self,
        table_name: str,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 250,
    ) -> int:
        if not records:
            return 0
        try:
            client = self._get_supabase()
            record_list = [dict(r) for r in records]
            for i in range(0, len(record_list), batch_size):
                batch = record_list[i : i + batch_size]
                client.table(table_name).insert(batch).execute()
            return len(record_list)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Supabase {table_name} insert failed: {exc}") from exc

    def _upsert_supabase_records(
        self,
        table_name: str,
        records: Sequence[Mapping[str, Any]],
        *,
        on_conflict: str,
        batch_size: int = 250,
    ) -> int:
        if not records:
            return 0
        try:
            client = self._get_supabase()
            record_list = [dict(r) for r in records]
            for i in range(0, len(record_list), batch_size):
                batch = record_list[i : i + batch_size]
                client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
            return len(record_list)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Supabase {table_name} upsert failed: {exc}") from exc

    def upsert_weather_features(
        self,
        df: pd.DataFrame,
        *,
        supabase_days: int | None = None,
        write_local: bool | None = None,
        write_hopsworks: bool = True,
        write_supabase: bool = True,
    ) -> dict[str, Any]:
        """Persist engineered hourly weather features to configured storage backends."""
        if df.empty:
            return {
                "storage_mode": self.storage_mode,
                "sqlite_rows": 0,
                "supabase_rows": 0,
                "hopsworks_rows": 0,
            }

        ordered = df.copy()
        ordered["timestamp_utc"] = pd.to_datetime(ordered["timestamp_utc"], utc=True)
        ordered = ordered.sort_values("timestamp_utc").drop_duplicates(
            subset=["timestamp_utc"], keep="last"
        )

        use_offline = self.settings.is_offline_mode
        should_write_local = use_offline if write_local is None else write_local
        will_write_supabase = (
            not use_offline and write_supabase and self.settings.has_supabase_credentials
        )
        will_write_hopsworks = (
            not use_offline and write_hopsworks and self.settings.has_hopsworks_credentials
        )
        if not (should_write_local or will_write_supabase or will_write_hopsworks):
            raise StorageError(
                f"No storage backend is active for mode '{self.storage_mode}': "
                "configure Supabase/Hopsworks credentials or enable offline mode."
            )

        summary: dict[str, Any] = {
            "storage_mode": self.storage_mode,
            "sqlite_rows": 0,
            "supabase_rows": 0,
            "hopsworks_rows": 0,
        }

        if should_write_local:
            records = dataframe_to_supabase_weather_records(ordered, settings=self.settings)
            summary["sqlite_rows"] = self._upsert_sqlite_weather_features(records)

        if will_write_supabase:
            days_window = supabase_days
            if days_window is None:
                days_window = (
                    self.settings.supabase_history_days
                    if self.settings.has_hopsworks_credentials
                    else max(self.settings.supabase_history_days, 180)
                )
            cutoff = ordered["timestamp_utc"].max() - pd.Timedelta(days=days_window)
            supa_df = ordered.loc[ordered["timestamp_utc"] >= cutoff]
            supa_records = dataframe_to_supabase_weather_records(
                supa_df, settings=self.settings
            )
            summary["supabase_rows"] = self._upsert_supabase_weather_features(supa_records)

        if will_write_hopsworks:
            hw_df = dataframe_to_hopsworks_frame(ordered, settings=self.settings)
            summary["hopsworks_rows"] = self._upsert_hopsworks_feature_group(hw_df)

        return summary

    def _upsert_sqlite_weather_features(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        now_iso = _now_utc_iso()
        cols = list(records[0].keys())
        all_cols = [*cols, "created_at", "updated_at"]
        placeholders = ", ".join(["?"] * len(all_cols))
        update_assignments = ", ".join(
            f"{c} = excluded.{c}" for c in [*cols, "updated_at"] if c != "timestamp_utc"
        )
        sql = (
            f"INSERT INTO weather_features_hourly ({', '.join(all_cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT(timestamp_utc) DO UPDATE SET {update_assignments}"
        )

        rows_to_insert: list[tuple[Any, ...]] = []
        for rec in records:
            row_vals: list[Any] = []
            for c in cols:
                val = rec[c]
                if c in {"sources_available", "imputed_columns", "features_json"}:
                    row_vals.append(json.dumps(val))
                elif c in {"is_imputed", "is_weekend", "is_lahore_smog_season", "is_monsoon_season"}:
                    row_vals.append(None if val is None else int(bool(val)))
                else:
                    row_vals.append(val)
            row_vals.extend([now_iso, now_iso])
            rows_to_insert.append(tuple(row_vals))

        with self._sqlite_connection() as conn:
            conn.executemany(sql, rows_to_insert)
        return len(rows_to_insert)

    def _upsert_supabase_weather_features(
        self,
        records: list[dict[str, Any]],
        *,
        batch_size: int = 250,
    ) -> int:
        if not records:
            return 0
        try:
            client = self._get_supabase()
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                client.table("weather_features_hourly").upsert(
                    batch, on_conflict="timestamp_utc"
                ).execute()
            return len(records)
        except Exception as exc:
            raise StorageError(f"Supabase weather_features_hourly upsert failed: {exc}") from exc

    def _upsert_hopsworks_feature_group(self, hw_df: pd.DataFrame) -> int:
        if hw_df.empty:
            return 0
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_or_create_feature_group(
                name=self.settings.hopsworks_feature_group_name,
                version=self.settings.hopsworks_feature_group_version,
                description="Hourly fused Lahore weather, air quality pollutants, and engineered temporal features.",
                primary_key=["timestamp_epoch_ms"],
                event_time="timestamp_utc",
                online_enabled=True,
            )
            fg.insert(hw_df, write_options={"wait_for_job": False})
            return len(hw_df)
        except Exception as exc:
            raise StorageError(f"Hopsworks Feature Group insert failed: {exc}") from exc

    def read_weather_features(
        self,
        *,
        limit_hours: int | None = None,
        start_utc: datetime | pd.Timestamp | str | None = None,
        end_utc: datetime | pd.Timestamp | str | None = None,
        expand_features_json: bool = True,
        prefer_hopsworks: bool = False,
    ) -> pd.DataFrame:
        """Read stored hourly weather features from Local SQLite, Supabase, or Hopsworks."""
        if self.storage_mode == "unconfigured":
            raise StorageError(
                "No storage backend is active for mode 'unconfigured': "
                "configure Supabase/Hopsworks credentials or enable offline mode."
            )
        rows: list[dict[str, Any]]
        if (
            prefer_hopsworks
            and not self.settings.is_offline_mode
            and self.settings.has_hopsworks_credentials
        ):
            return self._read_hopsworks_feature_group(
                start_utc=start_utc, end_utc=end_utc, limit_hours=limit_hours
            )

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.settings.is_offline_mode and self.settings.has_hopsworks_credentials:
                return self._read_hopsworks_feature_group(
                    start_utc=start_utc, end_utc=end_utc, limit_hours=limit_hours
                )
            rows = self._read_sqlite_weather_features(
                limit_hours=limit_hours, start_utc=start_utc, end_utc=end_utc
            )
        else:
            rows = self._read_supabase_weather_features(
                limit_hours=limit_hours, start_utc=start_utc, end_utc=end_utc
            )

        if not rows:
            return pd.DataFrame()

        df = self._records_to_dataframe(rows, expand_features_json=expand_features_json)
        return df.sort_values("timestamp_utc").reset_index(drop=True)

    def read_training_history(self, *, days: int | None = None) -> pd.DataFrame:
        """Read full historical engineered features for training/EDA (Hopsworks-first when online)."""
        start_utc = (
            datetime.now(timezone.utc) - timedelta(days=days)
            if days is not None
            else None
        )
        return self.read_weather_features(
            start_utc=start_utc,
            expand_features_json=True,
            prefer_hopsworks=True,
        )

    def _read_sqlite_weather_features(
        self,
        *,
        limit_hours: int | None,
        start_utc: datetime | pd.Timestamp | str | None,
        end_utc: datetime | pd.Timestamp | str | None,
    ) -> list[dict[str, Any]]:
        if not self.local_db_path.exists():
            return []

        clauses: list[str] = []
        params: list[Any] = []
        if start_utc is not None:
            clauses.append("timestamp_utc >= ?")
            params.append(_utc_iso(start_utc))
        if end_utc is not None:
            clauses.append("timestamp_utc <= ?")
            params.append(_utc_iso(end_utc))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = f"LIMIT {int(limit_hours)}" if limit_hours is not None else ""
        query = (
            f"SELECT * FROM weather_features_hourly {where_sql} "
            f"ORDER BY timestamp_utc DESC {limit_sql}"
        )

        with self._sqlite_connection() as conn:
            cursor = conn.execute(query, params)
            raw_rows = [dict(r) for r in cursor.fetchall()]

        for r in raw_rows:
            r["sources_available"] = _parse_json_or_default(r.get("sources_available"), [])
            r["imputed_columns"] = _parse_json_or_default(r.get("imputed_columns"), [])
            r["features_json"] = _parse_json_or_default(r.get("features_json"), {})
            for bcol in ("is_imputed", "is_weekend", "is_lahore_smog_season", "is_monsoon_season"):
                if r.get(bcol) is not None:
                    r[bcol] = bool(r[bcol])
        raw_rows.reverse()
        return raw_rows

    def _read_supabase_weather_features(
        self,
        *,
        limit_hours: int | None,
        start_utc: datetime | pd.Timestamp | str | None,
        end_utc: datetime | pd.Timestamp | str | None,
    ) -> list[dict[str, Any]]:
        try:
            client = self._get_supabase()
            page_size = 1000
            rows: list[dict[str, Any]] = []
            offset = 0

            def _build_query() -> Any:
                q = client.table("weather_features_hourly").select("*").order(
                    "timestamp_utc", desc=True
                )
                if start_utc is not None:
                    q = q.gte("timestamp_utc", _utc_iso(start_utc))
                if end_utc is not None:
                    q = q.lte("timestamp_utc", _utc_iso(end_utc))
                return q

            while True:
                remaining = (int(limit_hours) - len(rows)) if limit_hours is not None else page_size
                if remaining <= 0:
                    break
                step = min(page_size, remaining)
                response = _build_query().range(offset, offset + step - 1).execute()
                batch = list(response.data or [])
                rows.extend(batch)
                if len(batch) < step:
                    break
                offset += step

            rows.reverse()
            return rows
        except Exception as exc:
            raise StorageError(f"Supabase weather_features_hourly read failed: {exc}") from exc

    def _read_hopsworks_feature_group(
        self,
        *,
        start_utc: datetime | pd.Timestamp | str | None,
        end_utc: datetime | pd.Timestamp | str | None,
        limit_hours: int | None,
    ) -> pd.DataFrame:
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_feature_group(
                name=self.settings.hopsworks_feature_group_name,
                version=self.settings.hopsworks_feature_group_version,
            )
            read_kwargs: dict[str, Any] = {}
            if start_utc is not None:
                read_kwargs["start_time"] = pd.to_datetime(start_utc, utc=True).to_pydatetime()
            if end_utc is not None:
                read_kwargs["end_time"] = pd.to_datetime(end_utc, utc=True).to_pydatetime()
            if read_kwargs:
                try:
                    df = fg.read(**read_kwargs)
                except TypeError:
                    df = fg.read()
            else:
                df = fg.read()
        except Exception as exc:
            raise StorageError(f"Hopsworks Feature Group read failed: {exc}") from exc

        if df.empty:
            return df
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        if start_utc is not None:
            df = df.loc[df["timestamp_utc"] >= pd.to_datetime(start_utc, utc=True)]
        if end_utc is not None:
            df = df.loc[df["timestamp_utc"] <= pd.to_datetime(end_utc, utc=True)]
        df = df.sort_values("timestamp_utc")
        if limit_hours is not None:
            df = df.tail(int(limit_hours))
        for col in ("sources_available", "imputed_columns"):
            if col in df.columns:
                df[col] = [_parse_json_or_default(v, []) for v in df[col]]
        return df.reset_index(drop=True)

    @staticmethod
    def _records_to_dataframe(
        rows: list[dict[str, Any]],
        *,
        expand_features_json: bool = True,
    ) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        if expand_features_json and "features_json" in df.columns:
            expanded_dicts = [
                _parse_json_or_default(v, {}) for v in df["features_json"].tolist()
            ]
            if expanded_dicts:
                expanded_df = pd.DataFrame(expanded_dicts, index=df.index)
                overlap = [c for c in expanded_df.columns if c in df.columns]
                if overlap:
                    expanded_df = expanded_df.drop(columns=overlap)
                df = pd.concat([df, expanded_df], axis=1)
        return df

    def record_pipeline_run(
        self,
        run_id: str,
        pipeline_type: str,
        status: str,
        *,
        started_at_utc: datetime | str | None = None,
        completed_at_utc: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Insert or update a pipeline execution audit record."""
        self._require_configured()
        run_id_clean = _require_non_empty_str(run_id, "run_id")
        pipeline_type_clean = _require_enum(pipeline_type, VALID_PIPELINE_TYPES, "pipeline_type")
        status_clean = _require_enum(status, VALID_PIPELINE_STATUSES, "status")

        now_iso = _now_utc_iso()
        started_iso = (
            _require_utc_iso(started_at_utc, "started_at_utc")
            if started_at_utc is not None
            else now_iso
        )
        completed_iso = (
            _require_utc_iso(completed_at_utc, "completed_at_utc")
            if completed_at_utc is not None
            else None
        )
        if metadata is None:
            metadata = {}
        meta_clean, meta_json_str = _sanitize_json_container(
            metadata, expected_type=dict, field="metadata"
        )
        err_clean = _optional_str(error_message, "error_message")

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            with self._sqlite_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_runs (
                        run_id, pipeline_type, status, started_at_utc,
                        completed_at_utc, storage_mode, metadata_json,
                        error_message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        pipeline_type = excluded.pipeline_type,
                        status = excluded.status,
                        completed_at_utc = excluded.completed_at_utc,
                        storage_mode = excluded.storage_mode,
                        metadata_json = excluded.metadata_json,
                        error_message = excluded.error_message
                    """,
                    (
                        run_id_clean,
                        pipeline_type_clean,
                        status_clean,
                        started_iso,
                        completed_iso,
                        self.storage_mode,
                        meta_json_str,
                        err_clean,
                        now_iso,
                    ),
                )
            return

        payload = {
            "run_id": run_id_clean,
            "pipeline_type": pipeline_type_clean,
            "status": status_clean,
            "started_at_utc": started_iso,
            "completed_at_utc": completed_iso,
            "storage_mode": self.storage_mode,
            "metadata_json": meta_clean,
            "error_message": err_clean,
        }
        _assert_json_safe_record(payload, "pipeline_runs")
        self._upsert_supabase_records("pipeline_runs", [payload], on_conflict="run_id")

    def get_latest_pipeline_run(
        self,
        pipeline_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch the most recent pipeline run record (optionally filtered by type)."""
        if self.storage_mode == "unconfigured":
            raise StorageError(
                "No storage backend is active for mode 'unconfigured': "
                "configure Supabase/Hopsworks credentials or enable offline mode."
            )
        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return None
            with self._sqlite_connection() as conn:
                if pipeline_type:
                    row = conn.execute(
                        "SELECT * FROM pipeline_runs WHERE pipeline_type = ? "
                        "ORDER BY started_at_utc DESC, id DESC LIMIT 1",
                        (pipeline_type,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM pipeline_runs ORDER BY started_at_utc DESC, id DESC LIMIT 1"
                    ).fetchone()
            if row is None:
                return None
            data = dict(row)
            data["metadata_json"] = _parse_json_or_default(data.get("metadata_json"), {})
            return data

        try:
            client = self._get_supabase()
            query = client.table("pipeline_runs").select("*").order(
                "started_at_utc", desc=True
            ).limit(1)
            if pipeline_type:
                query = query.eq("pipeline_type", pipeline_type)
            resp = query.execute()
            rows = resp.data or []
            return rows[0] if rows else None
        except Exception as exc:
            raise StorageError(f"Supabase pipeline_runs read failed: {exc}") from exc

    def get_latest_imputation_audit(self) -> dict[str, Any] | None:
        """Return the latest imputation audit dictionary logged by backfill or hourly_ingest."""
        if self.storage_mode == "unconfigured":
            raise StorageError(
                "No storage backend is active for mode 'unconfigured': "
                "configure Supabase/Hopsworks credentials or enable offline mode."
            )
        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return None
            with self._sqlite_connection() as conn:
                rows = conn.execute(
                    "SELECT metadata_json FROM pipeline_runs "
                    "WHERE status = 'completed' ORDER BY started_at_utc DESC, id DESC LIMIT 10"
                ).fetchall()
            for r in rows:
                meta = _parse_json_or_default(r["metadata_json"], {})
                if isinstance(meta, dict) and "imputation_audit" in meta:
                    return meta["imputation_audit"]
            return None

        try:
            client = self._get_supabase()
            resp = (
                client.table("pipeline_runs")
                .select("metadata_json")
                .eq("status", "completed")
                .order("started_at_utc", desc=True)
                .limit(10)
                .execute()
            )
            for r in resp.data or []:
                meta = _parse_json_or_default(r.get("metadata_json"), {})
                if isinstance(meta, dict) and "imputation_audit" in meta:
                    return meta["imputation_audit"]
            return None
        except Exception as exc:
            raise StorageError(f"Failed reading latest imputation audit: {exc}") from exc

    def _resolve_non_fg_write_targets(self, write_local: bool | None) -> tuple[bool, bool]:
        use_local = self.settings.is_offline_mode or not self.settings.has_supabase_credentials
        should_write_local = use_local if write_local is None else write_local
        will_write_supabase = (
            not self.settings.is_offline_mode and self.settings.has_supabase_credentials
        )
        if not (should_write_local or will_write_supabase):
            raise StorageError(
                f"No storage backend is active for mode '{self.storage_mode}': "
                "configure Supabase credentials or enable local write."
            )
        return should_write_local, will_write_supabase

    def _resolve_fg_write_targets(
        self,
        write_local: bool | None,
        write_supabase: bool,
        write_hopsworks: bool,
    ) -> tuple[bool, bool, bool]:
        use_offline = self.settings.is_offline_mode
        should_write_local = use_offline if write_local is None else write_local
        will_write_supabase = (
            not use_offline and write_supabase and self.settings.has_supabase_credentials
        )
        will_write_hopsworks = (
            not use_offline and write_hopsworks and self.settings.has_hopsworks_credentials
        )
        if not (should_write_local or will_write_supabase or will_write_hopsworks):
            raise StorageError(
                f"No storage backend is active for mode '{self.storage_mode}': "
                "configure Supabase/Hopsworks credentials or enable offline/local mode."
            )
        return should_write_local, will_write_supabase, will_write_hopsworks

    # ------------------------------------------------------------------
    # Component 3: EDA Summaries (`eda_summaries`)
    # ------------------------------------------------------------------
    def upsert_eda_summary(
        self,
        summary: Mapping[str, Any],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and upsert a single EDA summary record into `eda_summaries`."""
        self._require_configured()
        run_id = _require_non_empty_str(summary.get("run_id"), "run_id")
        computed_at_utc = (
            _require_utc_iso(summary["computed_at_utc"], "computed_at_utc")
            if summary.get("computed_at_utc") is not None
            else _now_utc_iso()
        )
        sample_count = _require_int(
            summary["sample_count"] if "sample_count" in summary else 0,
            "sample_count",
            min_val=0,
        )
        date_range_start = (
            _require_utc_iso(summary["date_range_start"], "date_range_start")
            if summary.get("date_range_start") is not None
            else None
        )
        date_range_end = (
            _require_utc_iso(summary["date_range_end"], "date_range_end")
            if summary.get("date_range_end") is not None
            else None
        )
        if "summary_json" in summary:
            summary_clean, summary_json_str = _sanitize_json_container(
                summary["summary_json"], expected_type=dict, field="summary_json"
            )
        else:
            summary_clean, summary_json_str = {}, "{}"

        record = {
            "run_id": run_id,
            "computed_at_utc": computed_at_utc,
            "sample_count": sample_count,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "summary_json": summary_clean,
        }
        _assert_json_safe_record(record, "eda_summaries")

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            with self._sqlite_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO eda_summaries (
                        run_id, computed_at_utc, sample_count,
                        date_range_start, date_range_end, summary_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        computed_at_utc = excluded.computed_at_utc,
                        sample_count = excluded.sample_count,
                        date_range_start = excluded.date_range_start,
                        date_range_end = excluded.date_range_end,
                        summary_json = excluded.summary_json
                    """,
                    (
                        run_id,
                        computed_at_utc,
                        sample_count,
                        date_range_start,
                        date_range_end,
                        summary_json_str,
                        now_iso,
                    ),
                )
            result["sqlite_rows"] = 1

        if will_write_supabase:
            result["supabase_rows"] = self._upsert_supabase_records(
                "eda_summaries", [record], on_conflict="run_id"
            )

        return result

    def read_eda_summaries(
        self,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read EDA summary records ordered newest-first (`computed_at_utc DESC`)."""
        self._require_configured()
        if limit is not None:
            limit = _require_int(limit, "limit", min_val=0)
        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if run_id is not None:
                clauses.append("run_id = ?")
                params.append(run_id)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            limit_sql = f"LIMIT {limit}" if limit is not None else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM eda_summaries {where_sql} "
                    f"ORDER BY computed_at_utc DESC, id DESC {limit_sql}",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            filters: list[tuple[str, str, Any]] = []
            if run_id is not None:
                filters.append(("eq", "run_id", run_id))
            rows = self._read_supabase_paginated(
                "eda_summaries",
                filters=filters,
                orders=(("computed_at_utc", True), ("id", True)),
                limit=limit,
            )

        for r in rows:
            r["summary_json"] = _parse_json_or_default(r.get("summary_json"), {})
        return rows

    def read_latest_eda_summary(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        """Read the latest EDA summary record (optionally for a specific `run_id`)."""
        rows = self.read_eda_summaries(run_id=run_id, limit=1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Component 3: Model Registry Metrics (`model_registry_metrics`)
    # ------------------------------------------------------------------
    def upsert_model_registry_metrics(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and upsert evaluation leaderboard rows into `model_registry_metrics`."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        cleaned_records: list[dict[str, Any]] = []
        serialized_params: list[str] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for row in row_list:
            train_run_id = _require_non_empty_str(row.get("train_run_id"), "train_run_id")
            trained_at_utc = (
                _require_utc_iso(row["trained_at_utc"], "trained_at_utc")
                if row.get("trained_at_utc") is not None
                else _now_utc_iso()
            )
            metric = _require_enum(row.get("metric"), _TARGET_METRIC_SET, "metric")
            model_family = _require_enum(row.get("model_family"), _MODEL_FAMILY_SET, "model_family")
            conflict_key = (train_run_id, metric, model_family)
            if conflict_key in seen_keys:
                raise StorageValidationError(
                    f"Duplicate conflict key in model_registry_metrics batch: {conflict_key}"
                )
            seen_keys.add(conflict_key)

            model_version = _optional_str(row.get("model_version"), "model_version")
            val_rmse = _require_finite_float(row.get("val_rmse"), "val_rmse", non_negative=True)
            val_mae = _require_finite_float(row.get("val_mae"), "val_mae", non_negative=True)
            val_r2 = _require_finite_float(row.get("val_r2"), "val_r2")
            test_rmse = _require_finite_float(row.get("test_rmse"), "test_rmse", non_negative=True)
            test_mae = _require_finite_float(row.get("test_mae"), "test_mae", non_negative=True)
            test_r2 = _require_finite_float(row.get("test_r2"), "test_r2")
            improvement = _require_finite_float(
                row["improvement_over_naive_pct"] if "improvement_over_naive_pct" in row else 0.0,
                "improvement_over_naive_pct",
            )
            is_champion = (
                _require_bool(row["is_champion"], "is_champion") if "is_champion" in row else False
            )
            promoted = (
                _require_bool(row["promoted_to_registry"], "promoted_to_registry")
                if "promoted_to_registry" in row
                else False
            )
            if "hyperparameters" in row:
                hyper_clean, hyper_json_str = _sanitize_json_container(
                    row["hyperparameters"], expected_type=dict, field="hyperparameters"
                )
            else:
                hyper_clean, hyper_json_str = {}, "{}"
            training_samples = (
                _require_int(row["training_samples"], "training_samples", min_val=0)
                if row.get("training_samples") is not None
                else None
            )

            record = {
                "train_run_id": train_run_id,
                "trained_at_utc": trained_at_utc,
                "metric": metric,
                "model_family": model_family,
                "model_version": model_version,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "improvement_over_naive_pct": improvement,
                "is_champion": is_champion,
                "promoted_to_registry": promoted,
                "hyperparameters": hyper_clean,
                "training_samples": training_samples,
            }
            _assert_json_safe_record(record, "model_registry_metrics")
            cleaned_records.append(record)
            serialized_params.append(hyper_json_str)

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO model_registry_metrics (
                    train_run_id, trained_at_utc, metric, model_family, model_version,
                    val_rmse, val_mae, val_r2, test_rmse, test_mae, test_r2,
                    improvement_over_naive_pct, is_champion, promoted_to_registry,
                    hyperparameters, training_samples, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(train_run_id, metric, model_family) DO UPDATE SET
                    trained_at_utc = excluded.trained_at_utc,
                    model_version = excluded.model_version,
                    val_rmse = excluded.val_rmse,
                    val_mae = excluded.val_mae,
                    val_r2 = excluded.val_r2,
                    test_rmse = excluded.test_rmse,
                    test_mae = excluded.test_mae,
                    test_r2 = excluded.test_r2,
                    improvement_over_naive_pct = excluded.improvement_over_naive_pct,
                    is_champion = excluded.is_champion,
                    promoted_to_registry = excluded.promoted_to_registry,
                    hyperparameters = excluded.hyperparameters,
                    training_samples = excluded.training_samples
            """
            params_list = [
                (
                    rec["train_run_id"],
                    rec["trained_at_utc"],
                    rec["metric"],
                    rec["model_family"],
                    rec["model_version"],
                    rec["val_rmse"],
                    rec["val_mae"],
                    rec["val_r2"],
                    rec["test_rmse"],
                    rec["test_mae"],
                    rec["test_r2"],
                    rec["improvement_over_naive_pct"],
                    int(rec["is_champion"]),
                    int(rec["promoted_to_registry"]),
                    hyper_str,
                    rec["training_samples"],
                    now_iso,
                )
                for rec, hyper_str in zip(cleaned_records, serialized_params, strict=True)
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            result["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            result["supabase_rows"] = self._upsert_supabase_records(
                "model_registry_metrics",
                cleaned_records,
                on_conflict="train_run_id,metric,model_family",
            )

        return result

    def read_model_registry_metrics(
        self,
        *,
        train_run_id: str | None = None,
        metric: str | None = None,
        model_family: str | None = None,
        is_champion: bool | None = None,
        latest_run_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Read model leaderboard metrics ordered by `trained_at_utc DESC, metric ASC, val_rmse ASC`."""
        self._require_configured()
        use_sqlite = self.settings.is_offline_mode or not self.settings.has_supabase_credentials
        resolved_run_id = train_run_id

        if use_sqlite:
            if not self.local_db_path.exists():
                return []
            if latest_run_only and resolved_run_id is None:
                with self._sqlite_connection() as conn:
                    if metric is not None:
                        latest = conn.execute(
                            "SELECT train_run_id FROM model_registry_metrics WHERE metric = ? "
                            "ORDER BY trained_at_utc DESC, id DESC LIMIT 1",
                            (metric,),
                        ).fetchone()
                    else:
                        latest = conn.execute(
                            "SELECT train_run_id FROM model_registry_metrics "
                            "ORDER BY trained_at_utc DESC, id DESC LIMIT 1"
                        ).fetchone()
                if latest is None:
                    return []
                resolved_run_id = str(latest["train_run_id"])

            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("train_run_id = ?")
                params.append(resolved_run_id)
            if metric is not None:
                clauses.append("metric = ?")
                params.append(metric)
            if model_family is not None:
                clauses.append("model_family = ?")
                params.append(model_family)
            if is_champion is not None:
                clauses.append("is_champion = ?")
                params.append(int(bool(is_champion)))
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM model_registry_metrics {where_sql} "
                    "ORDER BY trained_at_utc DESC, metric ASC, val_rmse ASC, id ASC",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            if latest_run_only and resolved_run_id is None:
                latest_filters: list[tuple[str, str, Any]] = (
                    [("eq", "metric", metric)] if metric is not None else []
                )
                latest_rows = self._read_supabase_paginated(
                    "model_registry_metrics",
                    columns="train_run_id,trained_at_utc,id",
                    filters=latest_filters,
                    orders=(("trained_at_utc", True), ("id", True)),
                    limit=1,
                )
                if not latest_rows:
                    return []
                resolved_run_id = str(latest_rows[0]["train_run_id"])

            filters: list[tuple[str, str, Any]] = []
            if resolved_run_id is not None:
                filters.append(("eq", "train_run_id", resolved_run_id))
            if metric is not None:
                filters.append(("eq", "metric", metric))
            if model_family is not None:
                filters.append(("eq", "model_family", model_family))
            if is_champion is not None:
                filters.append(("eq", "is_champion", bool(is_champion)))
            rows = self._read_supabase_paginated(
                "model_registry_metrics",
                filters=filters,
                orders=(("trained_at_utc", True), ("metric", False), ("val_rmse", False), ("id", False)),
            )

        for r in rows:
            r["is_champion"] = bool(r.get("is_champion", False))
            r["promoted_to_registry"] = bool(r.get("promoted_to_registry", False))
            r["hyperparameters"] = _parse_json_or_default(r.get("hyperparameters"), {})
        return rows

    # ------------------------------------------------------------------
    # Component 3: Model Comparison Series (`model_comparison_series`)
    # ------------------------------------------------------------------
    def upsert_model_comparison_series(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and upsert aligned test/backtest curve rows into `model_comparison_series`."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        series_cols = ("actual", "sarimax", "random_forest", "ridge", "lstm", "naive_24h")
        cleaned_records: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for row in row_list:
            train_run_id = _require_non_empty_str(row.get("train_run_id"), "train_run_id")
            metric = _require_enum(row.get("metric"), _TARGET_METRIC_SET, "metric")
            timestamp_utc = _require_utc_iso(row.get("timestamp_utc"), "timestamp_utc")
            conflict_key = (train_run_id, metric, timestamp_utc)
            if conflict_key in seen_keys:
                raise StorageValidationError(
                    f"Duplicate conflict key in model_comparison_series batch: {conflict_key}"
                )
            seen_keys.add(conflict_key)

            record: dict[str, Any] = {
                "train_run_id": train_run_id,
                "metric": metric,
                "timestamp_utc": timestamp_utc,
            }
            for col in series_cols:
                record[col] = _optional_finite_float(row.get(col), col)
            _assert_json_safe_record(record, "model_comparison_series")
            cleaned_records.append(record)

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO model_comparison_series (
                    train_run_id, metric, timestamp_utc, actual, sarimax,
                    random_forest, ridge, lstm, naive_24h, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(train_run_id, metric, timestamp_utc) DO UPDATE SET
                    actual = excluded.actual,
                    sarimax = excluded.sarimax,
                    random_forest = excluded.random_forest,
                    ridge = excluded.ridge,
                    lstm = excluded.lstm,
                    naive_24h = excluded.naive_24h
            """
            params_list = [
                (
                    rec["train_run_id"],
                    rec["metric"],
                    rec["timestamp_utc"],
                    rec["actual"],
                    rec["sarimax"],
                    rec["random_forest"],
                    rec["ridge"],
                    rec["lstm"],
                    rec["naive_24h"],
                    now_iso,
                )
                for rec in cleaned_records
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            result["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            result["supabase_rows"] = self._upsert_supabase_records(
                "model_comparison_series",
                cleaned_records,
                on_conflict="train_run_id,metric,timestamp_utc",
            )

        return result

    def read_model_comparison_series(
        self,
        *,
        metric: str | None = None,
        train_run_id: str | None = None,
        start_utc: datetime | pd.Timestamp | str | None = None,
        end_utc: datetime | pd.Timestamp | str | None = None,
        latest_run_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Read model comparison series ordered chronologically (`timestamp_utc ASC`)."""
        self._require_configured()
        use_sqlite = self.settings.is_offline_mode or not self.settings.has_supabase_credentials
        resolved_run_id = train_run_id
        start_iso = _require_utc_iso(start_utc, "start_utc") if start_utc is not None else None
        end_iso = _require_utc_iso(end_utc, "end_utc") if end_utc is not None else None

        if use_sqlite:
            if not self.local_db_path.exists():
                return []
            if latest_run_only and resolved_run_id is None:
                with self._sqlite_connection() as conn:
                    if metric is not None:
                        latest = conn.execute(
                            "SELECT train_run_id FROM model_comparison_series WHERE metric = ? "
                            "ORDER BY created_at DESC, id DESC LIMIT 1",
                            (metric,),
                        ).fetchone()
                    else:
                        latest = conn.execute(
                            "SELECT train_run_id FROM model_comparison_series "
                            "ORDER BY created_at DESC, id DESC LIMIT 1"
                        ).fetchone()
                if latest is None:
                    return []
                resolved_run_id = str(latest["train_run_id"])

            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("train_run_id = ?")
                params.append(resolved_run_id)
            if metric is not None:
                clauses.append("metric = ?")
                params.append(metric)
            if start_iso is not None:
                clauses.append("timestamp_utc >= ?")
                params.append(start_iso)
            if end_iso is not None:
                clauses.append("timestamp_utc <= ?")
                params.append(end_iso)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM model_comparison_series {where_sql} "
                    "ORDER BY timestamp_utc ASC, id ASC",
                    params,
                ).fetchall()
            return [dict(r) for r in raw_rows]

        if latest_run_only and resolved_run_id is None:
            latest_filters: list[tuple[str, str, Any]] = (
                [("eq", "metric", metric)] if metric is not None else []
            )
            latest_rows = self._read_supabase_paginated(
                "model_comparison_series",
                columns="train_run_id,created_at,id",
                filters=latest_filters,
                orders=(("created_at", True), ("id", True)),
                limit=1,
            )
            if not latest_rows:
                return []
            resolved_run_id = str(latest_rows[0]["train_run_id"])

        filters: list[tuple[str, str, Any]] = []
        if resolved_run_id is not None:
            filters.append(("eq", "train_run_id", resolved_run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))
        if start_iso is not None:
            filters.append(("gte", "timestamp_utc", start_iso))
        if end_iso is not None:
            filters.append(("lte", "timestamp_utc", end_iso))
        return self._read_supabase_paginated(
            "model_comparison_series",
            filters=filters,
            orders=(("timestamp_utc", False), ("id", False)),
        )

    # ------------------------------------------------------------------
    # Component 3: SHAP & LIME Explanations (`shap_lime_explanations`)
    # ------------------------------------------------------------------
    def upsert_shap_lime_explanations(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and upsert SHAP/LIME feature attributions into `shap_lime_explanations`."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        cleaned_records: list[dict[str, Any]] = []
        serialized_attrs: list[str] = []
        seen_keys: set[tuple[str, str, str, str]] = set()

        for row in row_list:
            train_run_id = _require_non_empty_str(row.get("train_run_id"), "train_run_id")
            computed_at_utc = (
                _require_utc_iso(row["computed_at_utc"], "computed_at_utc")
                if row.get("computed_at_utc") is not None
                else _now_utc_iso()
            )
            metric = _require_enum(row.get("metric"), _TARGET_METRIC_SET, "metric")
            model_family = _require_enum(row.get("model_family"), _MODEL_FAMILY_SET, "model_family")
            method = _require_enum(row.get("method"), VALID_EXPLANATION_METHODS, "method")
            scope = _require_enum(row.get("scope"), VALID_EXPLANATION_SCOPES, "scope")
            conflict_key = (train_run_id, metric, method, scope)
            if conflict_key in seen_keys:
                raise StorageValidationError(
                    f"Duplicate conflict key in shap_lime_explanations batch: {conflict_key}"
                )
            seen_keys.add(conflict_key)

            base_value = _optional_finite_float(row.get("base_value"), "base_value")
            predicted_value = _optional_finite_float(row.get("predicted_value"), "predicted_value")
            reference_ts = (
                _require_utc_iso(row["reference_timestamp_utc"], "reference_timestamp_utc")
                if row.get("reference_timestamp_utc") is not None
                else None
            )
            if "feature_attributions" in row:
                attrs_clean, attrs_json_str = _sanitize_json_container(
                    row["feature_attributions"], expected_type=list, field="feature_attributions"
                )
            else:
                attrs_clean, attrs_json_str = [], "[]"

            record = {
                "train_run_id": train_run_id,
                "computed_at_utc": computed_at_utc,
                "metric": metric,
                "model_family": model_family,
                "method": method,
                "scope": scope,
                "base_value": base_value,
                "predicted_value": predicted_value,
                "reference_timestamp_utc": reference_ts,
                "feature_attributions": attrs_clean,
            }
            _assert_json_safe_record(record, "shap_lime_explanations")
            cleaned_records.append(record)
            serialized_attrs.append(attrs_json_str)

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO shap_lime_explanations (
                    train_run_id, computed_at_utc, metric, model_family,
                    method, scope, base_value, predicted_value,
                    reference_timestamp_utc, feature_attributions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(train_run_id, metric, method, scope) DO UPDATE SET
                    computed_at_utc = excluded.computed_at_utc,
                    model_family = excluded.model_family,
                    base_value = excluded.base_value,
                    predicted_value = excluded.predicted_value,
                    reference_timestamp_utc = excluded.reference_timestamp_utc,
                    feature_attributions = excluded.feature_attributions
            """
            params_list = [
                (
                    rec["train_run_id"],
                    rec["computed_at_utc"],
                    rec["metric"],
                    rec["model_family"],
                    rec["method"],
                    rec["scope"],
                    rec["base_value"],
                    rec["predicted_value"],
                    rec["reference_timestamp_utc"],
                    attrs_str,
                    now_iso,
                )
                for rec, attrs_str in zip(cleaned_records, serialized_attrs, strict=True)
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            result["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            result["supabase_rows"] = self._upsert_supabase_records(
                "shap_lime_explanations",
                cleaned_records,
                on_conflict="train_run_id,metric,method,scope",
            )

        return result

    def read_shap_lime_explanations(
        self,
        *,
        metric: str | None = None,
        train_run_id: str | None = None,
        method: str | None = None,
        scope: str | None = None,
        latest_run_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Read SHAP/LIME explanations ordered by `computed_at_utc DESC, metric ASC, method ASC, scope ASC`."""
        self._require_configured()
        use_sqlite = self.settings.is_offline_mode or not self.settings.has_supabase_credentials
        resolved_run_id = train_run_id

        if use_sqlite:
            if not self.local_db_path.exists():
                return []
            if latest_run_only and resolved_run_id is None:
                with self._sqlite_connection() as conn:
                    if metric is not None:
                        latest = conn.execute(
                            "SELECT train_run_id FROM shap_lime_explanations WHERE metric = ? "
                            "ORDER BY computed_at_utc DESC, id DESC LIMIT 1",
                            (metric,),
                        ).fetchone()
                    else:
                        latest = conn.execute(
                            "SELECT train_run_id FROM shap_lime_explanations "
                            "ORDER BY computed_at_utc DESC, id DESC LIMIT 1"
                        ).fetchone()
                if latest is None:
                    return []
                resolved_run_id = str(latest["train_run_id"])

            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("train_run_id = ?")
                params.append(resolved_run_id)
            if metric is not None:
                clauses.append("metric = ?")
                params.append(metric)
            if method is not None:
                clauses.append("method = ?")
                params.append(method)
            if scope is not None:
                clauses.append("scope = ?")
                params.append(scope)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM shap_lime_explanations {where_sql} "
                    "ORDER BY computed_at_utc DESC, metric ASC, method ASC, scope ASC, id ASC",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            if latest_run_only and resolved_run_id is None:
                latest_filters: list[tuple[str, str, Any]] = (
                    [("eq", "metric", metric)] if metric is not None else []
                )
                latest_rows = self._read_supabase_paginated(
                    "shap_lime_explanations",
                    columns="train_run_id,computed_at_utc,id",
                    filters=latest_filters,
                    orders=(("computed_at_utc", True), ("id", True)),
                    limit=1,
                )
                if not latest_rows:
                    return []
                resolved_run_id = str(latest_rows[0]["train_run_id"])

            filters: list[tuple[str, str, Any]] = []
            if resolved_run_id is not None:
                filters.append(("eq", "train_run_id", resolved_run_id))
            if metric is not None:
                filters.append(("eq", "metric", metric))
            if method is not None:
                filters.append(("eq", "method", method))
            if scope is not None:
                filters.append(("eq", "scope", scope))
            rows = self._read_supabase_paginated(
                "shap_lime_explanations",
                filters=filters,
                orders=(
                    ("computed_at_utc", True),
                    ("metric", False),
                    ("method", False),
                    ("scope", False),
                    ("id", False),
                ),
            )

        for r in rows:
            r["feature_attributions"] = _parse_json_or_default(r.get("feature_attributions"), [])
        return rows

    # ------------------------------------------------------------------
    # Component 3 & 4: Forecast Ledger (`forecast_ledger` + Hopsworks Mirror)
    # ------------------------------------------------------------------
    def upsert_forecast_ledger(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
        write_hopsworks: bool = True,
        write_supabase: bool = True,
    ) -> dict[str, Any]:
        """Validate and upsert 1..24h forecast trajectories across SQLite/Supabase/Hopsworks."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {
                "storage_mode": self.storage_mode,
                "sqlite_rows": 0,
                "supabase_rows": 0,
                "hopsworks_rows": 0,
            }

        cleaned_records: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, int]] = set()

        for row in row_list:
            run_id = _require_non_empty_str(row.get("run_id"), "run_id")
            issued_at_utc = _require_utc_iso(row.get("issued_at_utc"), "issued_at_utc")
            target_timestamp_utc = _require_utc_iso(
                row.get("target_timestamp_utc"), "target_timestamp_utc"
            )
            horizon_hour = _require_int(
                row.get("horizon_hour"), "horizon_hour", min_val=1, max_val=24
            )
            metric = _require_enum(row.get("metric"), _TARGET_METRIC_SET, "metric")
            model_family = _require_enum(row.get("model_family"), _MODEL_FAMILY_SET, "model_family")
            conflict_key = (run_id, metric, model_family, horizon_hour)
            if conflict_key in seen_keys:
                raise StorageValidationError(
                    f"Duplicate conflict key in forecast_ledger batch: {conflict_key}"
                )
            seen_keys.add(conflict_key)

            model_version = _optional_str(row.get("model_version"), "model_version")
            predicted_value = _require_finite_float(row.get("predicted_value"), "predicted_value")
            lower_bound = _require_finite_float(row.get("lower_bound"), "lower_bound")
            upper_bound = _require_finite_float(row.get("upper_bound"), "upper_bound")
            raw_pred = float(row["predicted_value"])
            raw_low = float(row["lower_bound"])
            raw_high = float(row["upper_bound"])
            if not (raw_low <= raw_pred <= raw_high) or not (
                lower_bound <= predicted_value <= upper_bound
            ):
                raise StorageValidationError(
                    "forecast_ledger requires lower_bound <= predicted_value <= upper_bound "
                    f"(got lower={lower_bound}, predicted={predicted_value}, upper={upper_bound})."
                )
            naive_baseline = _optional_finite_float(
                row.get("naive_baseline_value"), "naive_baseline_value"
            )
            actual_value = _optional_finite_float(row.get("actual_value"), "actual_value")

            record = {
                "run_id": run_id,
                "issued_at_utc": issued_at_utc,
                "target_timestamp_utc": target_timestamp_utc,
                "horizon_hour": horizon_hour,
                "metric": metric,
                "model_family": model_family,
                "model_version": model_version,
                "predicted_value": predicted_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "naive_baseline_value": naive_baseline,
                "actual_value": actual_value,
            }
            _assert_json_safe_record(record, "forecast_ledger")
            cleaned_records.append(record)

        should_write_local, will_write_supabase, will_write_hopsworks = (
            self._resolve_fg_write_targets(write_local, write_supabase, write_hopsworks)
        )

        summary: dict[str, Any] = {
            "storage_mode": self.storage_mode,
            "sqlite_rows": 0,
            "supabase_rows": 0,
            "hopsworks_rows": 0,
        }

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO forecast_ledger (
                    run_id, issued_at_utc, target_timestamp_utc, horizon_hour,
                    metric, model_family, model_version, predicted_value,
                    lower_bound, upper_bound, naive_baseline_value, actual_value,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, metric, model_family, horizon_hour) DO UPDATE SET
                    issued_at_utc = excluded.issued_at_utc,
                    target_timestamp_utc = excluded.target_timestamp_utc,
                    model_version = excluded.model_version,
                    predicted_value = excluded.predicted_value,
                    lower_bound = excluded.lower_bound,
                    upper_bound = excluded.upper_bound,
                    naive_baseline_value = excluded.naive_baseline_value,
                    actual_value = excluded.actual_value
            """
            params_list = [
                (
                    rec["run_id"],
                    rec["issued_at_utc"],
                    rec["target_timestamp_utc"],
                    rec["horizon_hour"],
                    rec["metric"],
                    rec["model_family"],
                    rec["model_version"],
                    rec["predicted_value"],
                    rec["lower_bound"],
                    rec["upper_bound"],
                    rec["naive_baseline_value"],
                    rec["actual_value"],
                    now_iso,
                )
                for rec in cleaned_records
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            summary["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            summary["supabase_rows"] = self._upsert_supabase_records(
                "forecast_ledger",
                cleaned_records,
                on_conflict="run_id,metric,model_family,horizon_hour",
            )

        if will_write_hopsworks:
            summary["hopsworks_rows"] = self._upsert_hopsworks_forecast_ledger(cleaned_records)

        return summary

    def _upsert_hopsworks_forecast_ledger(self, records: Sequence[Mapping[str, Any]]) -> int:
        if not records:
            return 0
        hw_df = _records_to_hopsworks_forecast_ledger_frame(records)
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_or_create_feature_group(
                name=self.settings.hopsworks_forecast_ledger_fg,
                version=self.settings.hopsworks_feature_group_version,
                description="Multi-horizon 24h forecast trajectories across Lahore weather and AQI metrics.",
                primary_key=["run_id", "metric", "model_family", "horizon_hour"],
                event_time="issued_at_utc",
                online_enabled=True,
            )
            fg.insert(hw_df, write_options={"wait_for_job": False})
            return len(hw_df)
        except Exception as exc:
            raise StorageError(f"Hopsworks forecast_ledger Feature Group insert failed: {exc}") from exc

    def read_forecast_ledger(
        self,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        model_family: str | None = None,
        start_target_utc: datetime | pd.Timestamp | str | None = None,
        end_target_utc: datetime | pd.Timestamp | str | None = None,
        start_issued_utc: datetime | pd.Timestamp | str | None = None,
        end_issued_utc: datetime | pd.Timestamp | str | None = None,
        latest_run_only: bool = False,
        prefer_hopsworks: bool = False,
    ) -> list[dict[str, Any]]:
        """Read forecast ledger rows ordered by `issued_at_utc DESC, horizon_hour ASC, metric ASC`."""
        self._require_configured()
        start_target_iso = (
            _require_utc_iso(start_target_utc, "start_target_utc")
            if start_target_utc is not None
            else None
        )
        end_target_iso = (
            _require_utc_iso(end_target_utc, "end_target_utc")
            if end_target_utc is not None
            else None
        )
        start_issued_iso = (
            _require_utc_iso(start_issued_utc, "start_issued_utc")
            if start_issued_utc is not None
            else None
        )
        end_issued_iso = (
            _require_utc_iso(end_issued_utc, "end_issued_utc")
            if end_issued_utc is not None
            else None
        )

        if (
            prefer_hopsworks
            and not self.settings.is_offline_mode
            and self.settings.has_hopsworks_credentials
        ):
            return self._read_hopsworks_forecast_ledger(
                run_id=run_id,
                metric=metric,
                model_family=model_family,
                start_target_iso=start_target_iso,
                end_target_iso=end_target_iso,
                start_issued_iso=start_issued_iso,
                end_issued_iso=end_issued_iso,
                latest_run_only=latest_run_only,
            )

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.settings.is_offline_mode and self.settings.has_hopsworks_credentials:
                return self._read_hopsworks_forecast_ledger(
                    run_id=run_id,
                    metric=metric,
                    model_family=model_family,
                    start_target_iso=start_target_iso,
                    end_target_iso=end_target_iso,
                    start_issued_iso=start_issued_iso,
                    end_issued_iso=end_issued_iso,
                    latest_run_only=latest_run_only,
                )
            if not self.local_db_path.exists():
                return []
            resolved_run_id = run_id
            if latest_run_only and resolved_run_id is None:
                with self._sqlite_connection() as conn:
                    if metric is not None:
                        latest = conn.execute(
                            "SELECT run_id FROM forecast_ledger WHERE metric = ? "
                            "ORDER BY issued_at_utc DESC, id DESC LIMIT 1",
                            (metric,),
                        ).fetchone()
                    else:
                        latest = conn.execute(
                            "SELECT run_id FROM forecast_ledger "
                            "ORDER BY issued_at_utc DESC, id DESC LIMIT 1"
                        ).fetchone()
                if latest is None:
                    return []
                resolved_run_id = str(latest["run_id"])

            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("run_id = ?")
                params.append(resolved_run_id)
            if metric is not None:
                clauses.append("metric = ?")
                params.append(metric)
            if model_family is not None:
                clauses.append("model_family = ?")
                params.append(model_family)
            if start_target_iso is not None:
                clauses.append("target_timestamp_utc >= ?")
                params.append(start_target_iso)
            if end_target_iso is not None:
                clauses.append("target_timestamp_utc <= ?")
                params.append(end_target_iso)
            if start_issued_iso is not None:
                clauses.append("issued_at_utc >= ?")
                params.append(start_issued_iso)
            if end_issued_iso is not None:
                clauses.append("issued_at_utc <= ?")
                params.append(end_issued_iso)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM forecast_ledger {where_sql} "
                    "ORDER BY issued_at_utc DESC, horizon_hour ASC, metric ASC, id ASC",
                    params,
                ).fetchall()
            return [dict(r) for r in raw_rows]

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            latest_filters: list[tuple[str, str, Any]] = (
                [("eq", "metric", metric)] if metric is not None else []
            )
            latest_rows = self._read_supabase_paginated(
                "forecast_ledger",
                columns="run_id,issued_at_utc,id",
                filters=latest_filters,
                orders=(("issued_at_utc", True), ("id", True)),
                limit=1,
            )
            if not latest_rows:
                return []
            resolved_run_id = str(latest_rows[0]["run_id"])

        filters: list[tuple[str, str, Any]] = []
        if resolved_run_id is not None:
            filters.append(("eq", "run_id", resolved_run_id))
        if metric is not None:
            filters.append(("eq", "metric", metric))
        if model_family is not None:
            filters.append(("eq", "model_family", model_family))
        if start_target_iso is not None:
            filters.append(("gte", "target_timestamp_utc", start_target_iso))
        if end_target_iso is not None:
            filters.append(("lte", "target_timestamp_utc", end_target_iso))
        if start_issued_iso is not None:
            filters.append(("gte", "issued_at_utc", start_issued_iso))
        if end_issued_iso is not None:
            filters.append(("lte", "issued_at_utc", end_issued_iso))
        return self._read_supabase_paginated(
            "forecast_ledger",
            filters=filters,
            orders=(("issued_at_utc", True), ("horizon_hour", False), ("metric", False), ("id", False)),
        )

    def _read_hopsworks_forecast_ledger(
        self,
        *,
        run_id: str | None,
        metric: str | None,
        model_family: str | None,
        start_target_iso: str | None,
        end_target_iso: str | None,
        start_issued_iso: str | None,
        end_issued_iso: str | None,
        latest_run_only: bool,
    ) -> list[dict[str, Any]]:
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_feature_group(
                name=self.settings.hopsworks_forecast_ledger_fg,
                version=self.settings.hopsworks_feature_group_version,
            )
            read_kwargs: dict[str, Any] = {}
            if not (latest_run_only and run_id is None):
                if start_issued_iso is not None:
                    read_kwargs["start_time"] = pd.to_datetime(start_issued_iso, utc=True).to_pydatetime()
                if end_issued_iso is not None:
                    read_kwargs["end_time"] = pd.to_datetime(end_issued_iso, utc=True).to_pydatetime()
            if read_kwargs:
                try:
                    df = fg.read(**read_kwargs)
                except TypeError:
                    df = fg.read()
            else:
                df = fg.read()
        except Exception as exc:
            raise StorageError(f"Hopsworks forecast_ledger Feature Group read failed: {exc}") from exc

        if df.empty:
            return []
        work = df.copy()
        work["issued_at_utc"] = pd.to_datetime(work["issued_at_utc"], utc=True)
        work["target_timestamp_utc"] = pd.to_datetime(work["target_timestamp_utc"], utc=True)
        if metric is not None:
            work = work.loc[work["metric"] == metric]
        if work.empty:
            return []

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            max_issued = work["issued_at_utc"].max()
            resolved_run_id = str(work.loc[work["issued_at_utc"] == max_issued, "run_id"].iloc[-1])

        if resolved_run_id is not None:
            work = work.loc[work["run_id"] == resolved_run_id]
        if model_family is not None:
            work = work.loc[work["model_family"] == model_family]
        if start_target_iso is not None:
            work = work.loc[work["target_timestamp_utc"] >= pd.to_datetime(start_target_iso, utc=True)]
        if end_target_iso is not None:
            work = work.loc[work["target_timestamp_utc"] <= pd.to_datetime(end_target_iso, utc=True)]
        if start_issued_iso is not None:
            work = work.loc[work["issued_at_utc"] >= pd.to_datetime(start_issued_iso, utc=True)]
        if end_issued_iso is not None:
            work = work.loc[work["issued_at_utc"] <= pd.to_datetime(end_issued_iso, utc=True)]
        if work.empty:
            return []

        work = work.sort_values(
            by=["issued_at_utc", "horizon_hour", "metric"],
            ascending=[False, True, True],
        )
        records: list[dict[str, Any]] = []
        for row in work.to_dict(orient="records"):
            mv = row.get("model_version")
            records.append(
                {
                    "run_id": str(row["run_id"]),
                    "issued_at_utc": _utc_iso(row["issued_at_utc"]),
                    "target_timestamp_utc": _utc_iso(row["target_timestamp_utc"]),
                    "horizon_hour": int(row["horizon_hour"]),
                    "metric": str(row["metric"]),
                    "model_family": str(row["model_family"]),
                    "model_version": None if mv is None or pd.isna(mv) or mv == "" else str(mv),
                    "predicted_value": float(row["predicted_value"]),
                    "lower_bound": float(row["lower_bound"]),
                    "upper_bound": float(row["upper_bound"]),
                    "naive_baseline_value": _optional_finite_float(
                        row.get("naive_baseline_value"), "naive_baseline_value"
                    ),
                    "actual_value": _optional_finite_float(row.get("actual_value"), "actual_value"),
                }
            )
        return records

    def _resolve_latest_run_id(
        self,
        table_name: str,
        order_col: str,
        *,
        run_col: str = "run_id",
        eq_filter: tuple[str, Any] | None = None,
    ) -> str | None:
        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return None
            with self._sqlite_connection() as conn:
                if eq_filter is not None and eq_filter[1] is not None:
                    row = conn.execute(
                        f"SELECT {run_col} FROM {table_name} WHERE {eq_filter[0]} = ? "
                        f"ORDER BY {order_col} DESC, id DESC LIMIT 1",
                        (eq_filter[1],),
                    ).fetchone()
                else:
                    row = conn.execute(
                        f"SELECT {run_col} FROM {table_name} "
                        f"ORDER BY {order_col} DESC, id DESC LIMIT 1"
                    ).fetchone()
            return None if row is None or row[run_col] is None else str(row[run_col])

        filters: list[tuple[str, str, Any]] = (
            [("eq", eq_filter[0], eq_filter[1])]
            if eq_filter is not None and eq_filter[1] is not None
            else []
        )
        rows = self._read_supabase_paginated(
            table_name,
            columns=f"{run_col},{order_col},id",
            filters=filters,
            orders=((order_col, True), ("id", True)),
            limit=1,
        )
        if not rows or rows[0].get(run_col) is None:
            return None
        return str(rows[0][run_col])

    # ------------------------------------------------------------------
    # Component 4: Weather Alerts (`weather_alerts`)
    # ------------------------------------------------------------------
    def insert_weather_alerts(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and insert 3-audience hazard alerts into `weather_alerts`."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        cleaned_records: list[dict[str, Any]] = []
        for row in row_list:
            run_id = _require_non_empty_str(row.get("run_id"), "run_id")
            timestamp = (
                _require_utc_iso(row["timestamp"], "timestamp")
                if row.get("timestamp") is not None
                else _now_utc_iso()
            )
            target_ts = (
                _require_utc_iso(row["target_timestamp_utc"], "target_timestamp_utc")
                if row.get("target_timestamp_utc") is not None
                else None
            )
            metric = _require_enum(row.get("metric"), _TARGET_METRIC_SET, "metric")
            severity = _require_enum(row.get("severity"), VALID_ALERT_SEVERITIES, "severity")
            title = _require_non_empty_str(row.get("title"), "title")
            trigger_value = _require_finite_float(row.get("trigger_value"), "trigger_value")
            threshold = _require_finite_float(row.get("threshold"), "threshold")
            advisory_general = _require_non_empty_str(
                row.get("advisory_general"), "advisory_general"
            )
            advisory_sensitive = _require_non_empty_str(
                row.get("advisory_sensitive"), "advisory_sensitive"
            )
            advisory_outdoor = _require_non_empty_str(
                row.get("advisory_outdoor_workers"), "advisory_outdoor_workers"
            )
            is_active = (
                _require_bool(row["is_active"], "is_active") if "is_active" in row else True
            )
            llm_provider = _optional_str(row.get("llm_provider"), "llm_provider")
            llm_model = _optional_str(row.get("llm_model"), "llm_model")

            record = {
                "run_id": run_id,
                "timestamp": timestamp,
                "target_timestamp_utc": target_ts,
                "metric": metric,
                "severity": severity,
                "title": title,
                "trigger_value": trigger_value,
                "threshold": threshold,
                "advisory_general": advisory_general,
                "advisory_sensitive": advisory_sensitive,
                "advisory_outdoor_workers": advisory_outdoor,
                "is_active": is_active,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
            _assert_json_safe_record(record, "weather_alerts")
            cleaned_records.append(record)

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO weather_alerts (
                    run_id, timestamp, target_timestamp_utc, metric, severity,
                    title, trigger_value, threshold, advisory_general,
                    advisory_sensitive, advisory_outdoor_workers, is_active,
                    llm_provider, llm_model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params_list = [
                (
                    rec["run_id"],
                    rec["timestamp"],
                    rec["target_timestamp_utc"],
                    rec["metric"],
                    rec["severity"],
                    rec["title"],
                    rec["trigger_value"],
                    rec["threshold"],
                    rec["advisory_general"],
                    rec["advisory_sensitive"],
                    rec["advisory_outdoor_workers"],
                    int(rec["is_active"]),
                    rec["llm_provider"],
                    rec["llm_model"],
                    now_iso,
                )
                for rec in cleaned_records
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            result["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            result["supabase_rows"] = self._insert_supabase_records(
                "weather_alerts", cleaned_records
            )

        return result

    def read_weather_alerts(
        self,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        severity: str | None = None,
        is_active: bool | None = None,
        latest_run_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read hazard alerts ordered newest-first (`timestamp DESC, id DESC`)."""
        self._require_configured()
        if limit is not None:
            limit = _require_int(limit, "limit", min_val=0)
            if limit == 0:
                return []

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            resolved_run_id = self._resolve_latest_run_id(
                "weather_alerts", "timestamp", eq_filter=("metric", metric)
            )
            if resolved_run_id is None:
                return []

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("run_id = ?")
                params.append(resolved_run_id)
            if metric is not None:
                clauses.append("metric = ?")
                params.append(metric)
            if severity is not None:
                clauses.append("severity = ?")
                params.append(severity)
            if is_active is not None:
                clauses.append("is_active = ?")
                params.append(int(bool(is_active)))
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            limit_sql = f"LIMIT {limit}" if limit is not None else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM weather_alerts {where_sql} "
                    f"ORDER BY timestamp DESC, id DESC {limit_sql}",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            filters: list[tuple[str, str, Any]] = []
            if resolved_run_id is not None:
                filters.append(("eq", "run_id", resolved_run_id))
            if metric is not None:
                filters.append(("eq", "metric", metric))
            if severity is not None:
                filters.append(("eq", "severity", severity))
            if is_active is not None:
                filters.append(("eq", "is_active", bool(is_active)))
            rows = self._read_supabase_paginated(
                "weather_alerts",
                filters=filters,
                orders=(("timestamp", True), ("id", True)),
                limit=limit,
            )

        for r in rows:
            r["is_active"] = bool(r.get("is_active", True))
        return rows

    # ------------------------------------------------------------------
    # Component 4: Weather Reports (`weather_reports`)
    # ------------------------------------------------------------------
    def upsert_weather_report(
        self,
        report: Mapping[str, Any],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and upsert a weather broadcast report into `weather_reports`."""
        self._require_configured()
        run_id = _require_non_empty_str(report.get("run_id"), "run_id")
        issued_at_utc = (
            _require_utc_iso(report["issued_at_utc"], "issued_at_utc")
            if report.get("issued_at_utc") is not None
            else _now_utc_iso()
        )
        trigger_reason = _require_enum(
            report.get("trigger_reason"), VALID_REPORT_TRIGGER_REASONS, "trigger_reason"
        )
        headline = _require_non_empty_str(report.get("headline"), "headline")
        summary = _require_non_empty_str(report.get("summary"), "summary")
        bulletin_markdown = _require_non_empty_str(
            report.get("bulletin_markdown"), "bulletin_markdown"
        )
        if "highlights" in report:
            highlights_clean, highlights_str = _sanitize_json_container(
                report["highlights"], expected_type=list, field="highlights"
            )
        else:
            highlights_clean, highlights_str = [], "[]"
        active_alert_count = (
            _require_int(report["active_alert_count"], "active_alert_count", min_val=0)
            if "active_alert_count" in report
            else 0
        )
        llm_provider = _optional_str(report.get("llm_provider"), "llm_provider")
        llm_model = _optional_str(report.get("llm_model"), "llm_model")
        fallback_used = (
            _require_bool(report["fallback_used"], "fallback_used")
            if "fallback_used" in report
            else False
        )

        record = {
            "run_id": run_id,
            "issued_at_utc": issued_at_utc,
            "trigger_reason": trigger_reason,
            "headline": headline,
            "summary": summary,
            "bulletin_markdown": bulletin_markdown,
            "highlights": highlights_clean,
            "active_alert_count": active_alert_count,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "fallback_used": fallback_used,
        }
        _assert_json_safe_record(record, "weather_reports")

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            with self._sqlite_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO weather_reports (
                        run_id, issued_at_utc, trigger_reason, headline, summary,
                        bulletin_markdown, highlights, active_alert_count,
                        llm_provider, llm_model, fallback_used, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        issued_at_utc = excluded.issued_at_utc,
                        trigger_reason = excluded.trigger_reason,
                        headline = excluded.headline,
                        summary = excluded.summary,
                        bulletin_markdown = excluded.bulletin_markdown,
                        highlights = excluded.highlights,
                        active_alert_count = excluded.active_alert_count,
                        llm_provider = excluded.llm_provider,
                        llm_model = excluded.llm_model,
                        fallback_used = excluded.fallback_used
                    """,
                    (
                        run_id,
                        issued_at_utc,
                        trigger_reason,
                        headline,
                        summary,
                        bulletin_markdown,
                        highlights_str,
                        active_alert_count,
                        llm_provider,
                        llm_model,
                        int(fallback_used),
                        now_iso,
                    ),
                )
            result["sqlite_rows"] = 1

        if will_write_supabase:
            result["supabase_rows"] = self._upsert_supabase_records(
                "weather_reports", [record], on_conflict="run_id"
            )

        return result

    def read_weather_reports(
        self,
        *,
        run_id: str | None = None,
        trigger_reason: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read weather broadcast reports ordered newest-first (`issued_at_utc DESC`)."""
        self._require_configured()
        if limit is not None:
            limit = _require_int(limit, "limit", min_val=0)
            if limit == 0:
                return []

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if run_id is not None:
                clauses.append("run_id = ?")
                params.append(run_id)
            if trigger_reason is not None:
                clauses.append("trigger_reason = ?")
                params.append(trigger_reason)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            limit_sql = f"LIMIT {limit}" if limit is not None else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM weather_reports {where_sql} "
                    f"ORDER BY issued_at_utc DESC, id DESC {limit_sql}",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            filters: list[tuple[str, str, Any]] = []
            if run_id is not None:
                filters.append(("eq", "run_id", run_id))
            if trigger_reason is not None:
                filters.append(("eq", "trigger_reason", trigger_reason))
            rows = self._read_supabase_paginated(
                "weather_reports",
                filters=filters,
                orders=(("issued_at_utc", True), ("id", True)),
                limit=limit,
            )

        for r in rows:
            r["highlights"] = _parse_json_or_default(r.get("highlights"), [])
            r["fallback_used"] = bool(r.get("fallback_used", False))
        return rows

    def read_latest_weather_report(self, *, run_id: str | None = None) -> dict[str, Any] | None:
        """Read the latest weather broadcast report."""
        rows = self.read_weather_reports(run_id=run_id, limit=1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Component 4: Agent LLM Call Log (`agent_call_log` + Hopsworks Mirror)
    # ------------------------------------------------------------------
    def insert_agent_call_logs(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
        write_hopsworks: bool = True,
        write_supabase: bool = True,
    ) -> dict[str, Any]:
        """Validate and log LLM call attempts across SQLite/Supabase/Hopsworks."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {
                "storage_mode": self.storage_mode,
                "sqlite_rows": 0,
                "supabase_rows": 0,
                "hopsworks_rows": 0,
            }

        cleaned_records: list[dict[str, Any]] = []
        for row in row_list:
            run_id = _require_non_empty_str(row.get("run_id"), "run_id")
            timestamp_utc = (
                _require_utc_iso(row["timestamp_utc"], "timestamp_utc")
                if row.get("timestamp_utc") is not None
                else _now_utc_iso()
            )
            agent_name = _require_non_empty_str(row.get("agent_name"), "agent_name")
            provider = _require_non_empty_str(row.get("provider"), "provider")
            model = _require_non_empty_str(row.get("model"), "model")
            reasoning_effort = _optional_str(row.get("reasoning_effort"), "reasoning_effort")
            prompt = _require_non_empty_str(row.get("prompt"), "prompt")
            response = _optional_str(row.get("response"), "response")
            latency_ms = (
                _require_finite_float(row["latency_ms"], "latency_ms", non_negative=True)
                if "latency_ms" in row
                else 0.0
            )
            status = _require_enum(row.get("status"), VALID_AGENT_CALL_STATUSES, "status")
            fallback_used = (
                _require_bool(row["fallback_used"], "fallback_used")
                if "fallback_used" in row
                else False
            )
            attempt_order = (
                _require_int(row["attempt_order"], "attempt_order", min_val=1, max_val=32767)
                if "attempt_order" in row
                else 1
            )
            error_message = _optional_str(row.get("error_message"), "error_message")

            record = {
                "run_id": run_id,
                "timestamp_utc": timestamp_utc,
                "agent_name": agent_name,
                "provider": provider,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "prompt": prompt,
                "response": response,
                "latency_ms": latency_ms,
                "status": status,
                "fallback_used": fallback_used,
                "attempt_order": attempt_order,
                "error_message": error_message,
            }
            _assert_json_safe_record(record, "agent_call_log")
            cleaned_records.append(record)

        should_write_local, will_write_supabase, will_write_hopsworks = (
            self._resolve_fg_write_targets(write_local, write_supabase, write_hopsworks)
        )
        summary = {
            "storage_mode": self.storage_mode,
            "sqlite_rows": 0,
            "supabase_rows": 0,
            "hopsworks_rows": 0,
        }

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO agent_call_log (
                    run_id, timestamp_utc, agent_name, provider, model,
                    reasoning_effort, prompt, response, latency_ms, status,
                    fallback_used, attempt_order, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params_list = [
                (
                    rec["run_id"],
                    rec["timestamp_utc"],
                    rec["agent_name"],
                    rec["provider"],
                    rec["model"],
                    rec["reasoning_effort"],
                    rec["prompt"],
                    rec["response"],
                    rec["latency_ms"],
                    rec["status"],
                    int(rec["fallback_used"]),
                    rec["attempt_order"],
                    rec["error_message"],
                    now_iso,
                )
                for rec in cleaned_records
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            summary["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            summary["supabase_rows"] = self._insert_supabase_records(
                "agent_call_log", cleaned_records
            )

        if will_write_hopsworks:
            summary["hopsworks_rows"] = self._insert_hopsworks_agent_call_logs(cleaned_records)

        return summary

    def _insert_hopsworks_agent_call_logs(self, records: Sequence[Mapping[str, Any]]) -> int:
        if not records:
            return 0
        hw_df = pd.DataFrame(
            {
                "log_uid": [
                    f"{r['run_id']}:{r['agent_name']}:{r['attempt_order']}:{uuid4().hex[:8]}"
                    for r in records
                ],
                "run_id": [str(r["run_id"]) for r in records],
                "timestamp_utc": pd.to_datetime([r["timestamp_utc"] for r in records], utc=True),
                "agent_name": [str(r["agent_name"]) for r in records],
                "provider": [str(r["provider"]) for r in records],
                "model": [str(r["model"]) for r in records],
                "reasoning_effort": [
                    "" if r.get("reasoning_effort") is None else str(r["reasoning_effort"])
                    for r in records
                ],
                "prompt": [str(r["prompt"]) for r in records],
                "response": ["" if r.get("response") is None else str(r["response"]) for r in records],
                "latency_ms": pd.Series([float(r["latency_ms"]) for r in records], dtype="float64"),
                "status": [str(r["status"]) for r in records],
                "fallback_used": pd.Series([bool(r["fallback_used"]) for r in records], dtype="bool"),
                "attempt_order": pd.Series([int(r["attempt_order"]) for r in records], dtype="int64"),
                "error_message": [
                    "" if r.get("error_message") is None else str(r["error_message"])
                    for r in records
                ],
            }
        )
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_or_create_feature_group(
                name=self.settings.hopsworks_agent_log_fg,
                version=self.settings.hopsworks_feature_group_version,
                description="Agentic LLM invocation and failover telemetry log.",
                primary_key=["log_uid"],
                event_time="timestamp_utc",
                online_enabled=True,
            )
            fg.insert(hw_df, write_options={"wait_for_job": False})
            return len(hw_df)
        except Exception as exc:
            raise StorageError(
                f"Hopsworks agent_call_log Feature Group insert failed: {exc}"
            ) from exc

    def read_agent_call_log(
        self,
        *,
        run_id: str | None = None,
        agent_name: str | None = None,
        status: str | None = None,
        provider: str | None = None,
        latest_run_only: bool = False,
        limit: int | None = None,
        prefer_hopsworks: bool = False,
    ) -> list[dict[str, Any]]:
        """Read agent LLM call logs ordered by `attempt_order ASC` (per run) or `timestamp_utc DESC`."""
        self._require_configured()
        if limit is not None:
            limit = _require_int(limit, "limit", min_val=0)
            if limit == 0:
                return []

        if (
            prefer_hopsworks
            and not self.settings.is_offline_mode
            and self.settings.has_hopsworks_credentials
        ) or (
            not self.settings.is_offline_mode
            and not self.settings.has_supabase_credentials
            and self.settings.has_hopsworks_credentials
        ):
            return self._read_hopsworks_agent_call_log(
                run_id=run_id,
                agent_name=agent_name,
                status=status,
                provider=provider,
                latest_run_only=latest_run_only,
                limit=limit,
            )

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            resolved_run_id = self._resolve_latest_run_id("agent_call_log", "timestamp_utc")
            if resolved_run_id is None:
                return []

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("run_id = ?")
                params.append(resolved_run_id)
            if agent_name is not None:
                clauses.append("agent_name = ?")
                params.append(agent_name)
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            if provider is not None:
                clauses.append("provider = ?")
                params.append(provider)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_sql = (
                "ORDER BY attempt_order ASC, id ASC"
                if resolved_run_id is not None
                else "ORDER BY timestamp_utc DESC, attempt_order ASC, id DESC"
            )
            limit_sql = f"LIMIT {limit}" if limit is not None else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM agent_call_log {where_sql} {order_sql} {limit_sql}",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            filters: list[tuple[str, str, Any]] = []
            if resolved_run_id is not None:
                filters.append(("eq", "run_id", resolved_run_id))
            if agent_name is not None:
                filters.append(("eq", "agent_name", agent_name))
            if status is not None:
                filters.append(("eq", "status", status))
            if provider is not None:
                filters.append(("eq", "provider", provider))
            orders: tuple[tuple[str, bool], ...] = (
                (("attempt_order", False), ("id", False))
                if resolved_run_id is not None
                else (("timestamp_utc", True), ("attempt_order", False), ("id", True))
            )
            rows = self._read_supabase_paginated(
                "agent_call_log", filters=filters, orders=orders, limit=limit
            )

        for r in rows:
            r["fallback_used"] = bool(r.get("fallback_used", False))
        return rows

    def _read_hopsworks_agent_call_log(
        self,
        *,
        run_id: str | None,
        agent_name: str | None,
        status: str | None,
        provider: str | None,
        latest_run_only: bool,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_feature_group(
                name=self.settings.hopsworks_agent_log_fg,
                version=self.settings.hopsworks_feature_group_version,
            )
            df = fg.read()
        except Exception as exc:
            raise StorageError(f"Hopsworks agent_call_log Feature Group read failed: {exc}") from exc

        if df.empty:
            return []
        work = df.copy()
        work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            max_ts = work["timestamp_utc"].max()
            resolved_run_id = str(work.loc[work["timestamp_utc"] == max_ts, "run_id"].iloc[-1])

        if resolved_run_id is not None:
            work = work.loc[work["run_id"] == resolved_run_id]
        if agent_name is not None:
            work = work.loc[work["agent_name"] == agent_name]
        if status is not None:
            work = work.loc[work["status"] == status]
        if provider is not None:
            work = work.loc[work["provider"] == provider]
        if work.empty:
            return []

        if resolved_run_id is not None:
            work = work.sort_values(by=["attempt_order", "timestamp_utc"], ascending=[True, True])
        else:
            work = work.sort_values(by=["timestamp_utc", "attempt_order"], ascending=[False, True])
        if limit is not None:
            work = work.head(limit)

        records: list[dict[str, Any]] = []
        for row in work.to_dict(orient="records"):
            records.append(
                {
                    "run_id": str(row["run_id"]),
                    "timestamp_utc": _utc_iso(row["timestamp_utc"]),
                    "agent_name": str(row["agent_name"]),
                    "provider": str(row["provider"]),
                    "model": str(row["model"]),
                    "reasoning_effort": _optional_str(
                        None if row.get("reasoning_effort") == "" else row.get("reasoning_effort"),
                        "reasoning_effort",
                    ),
                    "prompt": str(row["prompt"]),
                    "response": _optional_str(
                        None if row.get("response") == "" else row.get("response"),
                        "response",
                    ),
                    "latency_ms": float(row["latency_ms"]),
                    "status": str(row["status"]),
                    "fallback_used": bool(row["fallback_used"]),
                    "attempt_order": int(row["attempt_order"]),
                    "error_message": _optional_str(
                        None if row.get("error_message") == "" else row.get("error_message"),
                        "error_message",
                    ),
                }
            )
        return records

    # ------------------------------------------------------------------
    # Component 4: LangGraph Execution Trace (`agent_graph_trace`)
    # ------------------------------------------------------------------
    def upsert_agent_graph_trace(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
    ) -> dict[str, Any]:
        """Validate and upsert node execution traces into `agent_graph_trace`."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        cleaned_records: list[dict[str, Any]] = []
        serialized_summaries: list[str] = []
        seen_keys: set[tuple[str, str, int]] = set()

        for row in row_list:
            run_id = _require_non_empty_str(row.get("run_id"), "run_id")
            node_name = _require_non_empty_str(row.get("node_name"), "node_name")
            step_order = _require_int(row.get("step_order"), "step_order", min_val=1, max_val=32767)
            conflict_key = (run_id, node_name, step_order)
            if conflict_key in seen_keys:
                raise StorageValidationError(
                    f"Duplicate conflict key in agent_graph_trace batch: {conflict_key}"
                )
            seen_keys.add(conflict_key)

            status = _require_enum(row.get("status"), VALID_GRAPH_TRACE_STATUSES, "status")
            started_at = _require_utc_iso(row.get("started_at"), "started_at")
            duration_ms = (
                _require_finite_float(row["duration_ms"], "duration_ms", non_negative=True)
                if "duration_ms" in row
                else 0.0
            )
            branch_decision = _optional_str(row.get("branch_decision"), "branch_decision")
            if "summary_json" in row:
                summary_clean, summary_str = _sanitize_json_container(
                    row["summary_json"], expected_type=dict, field="summary_json"
                )
            else:
                summary_clean, summary_str = {}, "{}"
            error_message = _optional_str(row.get("error_message"), "error_message")

            record = {
                "run_id": run_id,
                "node_name": node_name,
                "step_order": step_order,
                "status": status,
                "started_at": started_at,
                "duration_ms": duration_ms,
                "branch_decision": branch_decision,
                "summary_json": summary_clean,
                "error_message": error_message,
            }
            _assert_json_safe_record(record, "agent_graph_trace")
            cleaned_records.append(record)
            serialized_summaries.append(summary_str)

        should_write_local, will_write_supabase = self._resolve_non_fg_write_targets(write_local)
        result = {"storage_mode": self.storage_mode, "sqlite_rows": 0, "supabase_rows": 0}

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO agent_graph_trace (
                    run_id, node_name, step_order, status, started_at,
                    duration_ms, branch_decision, summary_json, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_name, step_order) DO UPDATE SET
                    status = excluded.status,
                    started_at = excluded.started_at,
                    duration_ms = excluded.duration_ms,
                    branch_decision = excluded.branch_decision,
                    summary_json = excluded.summary_json,
                    error_message = excluded.error_message
            """
            params_list = [
                (
                    rec["run_id"],
                    rec["node_name"],
                    rec["step_order"],
                    rec["status"],
                    rec["started_at"],
                    rec["duration_ms"],
                    rec["branch_decision"],
                    summary_str,
                    rec["error_message"],
                    now_iso,
                )
                for rec, summary_str in zip(cleaned_records, serialized_summaries, strict=True)
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            result["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            result["supabase_rows"] = self._upsert_supabase_records(
                "agent_graph_trace",
                cleaned_records,
                on_conflict="run_id,node_name,step_order",
            )

        return result

    def read_agent_graph_trace(
        self,
        *,
        run_id: str | None = None,
        node_name: str | None = None,
        status: str | None = None,
        latest_run_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read LangGraph execution trace steps ordered by `step_order ASC` (or `started_at DESC`)."""
        self._require_configured()
        if limit is not None:
            limit = _require_int(limit, "limit", min_val=0)
            if limit == 0:
                return []

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            resolved_run_id = self._resolve_latest_run_id("agent_graph_trace", "started_at")
            if resolved_run_id is None:
                return []

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("run_id = ?")
                params.append(resolved_run_id)
            if node_name is not None:
                clauses.append("node_name = ?")
                params.append(node_name)
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_sql = (
                "ORDER BY step_order ASC, id ASC"
                if resolved_run_id is not None
                else "ORDER BY started_at DESC, step_order ASC, id ASC"
            )
            limit_sql = f"LIMIT {limit}" if limit is not None else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM agent_graph_trace {where_sql} {order_sql} {limit_sql}",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            filters: list[tuple[str, str, Any]] = []
            if resolved_run_id is not None:
                filters.append(("eq", "run_id", resolved_run_id))
            if node_name is not None:
                filters.append(("eq", "node_name", node_name))
            if status is not None:
                filters.append(("eq", "status", status))
            orders: tuple[tuple[str, bool], ...] = (
                (("step_order", False), ("id", False))
                if resolved_run_id is not None
                else (("started_at", True), ("step_order", False), ("id", False))
            )
            rows = self._read_supabase_paginated(
                "agent_graph_trace", filters=filters, orders=orders, limit=limit
            )

        for r in rows:
            r["summary_json"] = _parse_json_or_default(r.get("summary_json"), {})
        return rows

    # ------------------------------------------------------------------
    # Component 4: Data & Model Drift Log (`drift_log` + Hopsworks Mirror)
    # ------------------------------------------------------------------
    def insert_drift_logs(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        write_local: bool | None = None,
        write_hopsworks: bool = True,
        write_supabase: bool = True,
    ) -> dict[str, Any]:
        """Validate and insert data/model drift check records across SQLite/Supabase/Hopsworks."""
        self._require_configured()
        row_list = list(rows)
        if not row_list:
            return {
                "storage_mode": self.storage_mode,
                "sqlite_rows": 0,
                "supabase_rows": 0,
                "hopsworks_rows": 0,
            }

        cleaned_records: list[dict[str, Any]] = []
        serialized_details: list[str] = []

        for row in row_list:
            run_id = _optional_str(row.get("run_id"), "run_id")
            timestamp = (
                _require_utc_iso(row["timestamp"], "timestamp")
                if row.get("timestamp") is not None
                else _now_utc_iso()
            )
            metric = _require_non_empty_str(row.get("metric"), "metric")
            drift_type = _require_enum(row.get("drift_type"), VALID_DRIFT_TYPES, "drift_type")
            test_name = _optional_str(row.get("test_name"), "test_name")
            score = _optional_finite_float(row.get("score"), "score")
            secondary_score = _optional_finite_float(row.get("secondary_score"), "secondary_score")
            threshold = _require_finite_float(row.get("threshold"), "threshold")
            status = _require_enum(row.get("status"), VALID_DRIFT_STATUSES, "status")
            retrain_triggered = (
                _require_bool(row["retrain_triggered"], "retrain_triggered")
                if "retrain_triggered" in row
                else False
            )
            action_taken = _require_non_empty_str(row.get("action_taken"), "action_taken")
            error_message = _optional_str(row.get("error_message"), "error_message")
            if "details_json" in row:
                details_clean, details_str = _sanitize_json_container(
                    row["details_json"], expected_type=dict, field="details_json"
                )
            else:
                details_clean, details_str = {}, "{}"

            record = {
                "run_id": run_id,
                "timestamp": timestamp,
                "metric": metric,
                "drift_type": drift_type,
                "test_name": test_name,
                "score": score,
                "secondary_score": secondary_score,
                "threshold": threshold,
                "status": status,
                "retrain_triggered": retrain_triggered,
                "action_taken": action_taken,
                "error_message": error_message,
                "details_json": details_clean,
            }
            _assert_json_safe_record(record, "drift_log")
            cleaned_records.append(record)
            serialized_details.append(details_str)

        should_write_local, will_write_supabase, will_write_hopsworks = (
            self._resolve_fg_write_targets(write_local, write_supabase, write_hopsworks)
        )
        summary = {
            "storage_mode": self.storage_mode,
            "sqlite_rows": 0,
            "supabase_rows": 0,
            "hopsworks_rows": 0,
        }

        if should_write_local:
            now_iso = _now_utc_iso()
            sql = """
                INSERT INTO drift_log (
                    run_id, timestamp, metric, drift_type, test_name, score,
                    secondary_score, threshold, status, retrain_triggered,
                    action_taken, error_message, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params_list = [
                (
                    rec["run_id"],
                    rec["timestamp"],
                    rec["metric"],
                    rec["drift_type"],
                    rec["test_name"],
                    rec["score"],
                    rec["secondary_score"],
                    rec["threshold"],
                    rec["status"],
                    int(rec["retrain_triggered"]),
                    rec["action_taken"],
                    rec["error_message"],
                    details_str,
                    now_iso,
                )
                for rec, details_str in zip(cleaned_records, serialized_details, strict=True)
            ]
            with self._sqlite_connection() as conn:
                conn.executemany(sql, params_list)
            summary["sqlite_rows"] = len(cleaned_records)

        if will_write_supabase:
            summary["supabase_rows"] = self._insert_supabase_records("drift_log", cleaned_records)

        if will_write_hopsworks:
            summary["hopsworks_rows"] = self._insert_hopsworks_drift_logs(
                cleaned_records, serialized_details
            )

        return summary

    def _insert_hopsworks_drift_logs(
        self,
        records: Sequence[Mapping[str, Any]],
        serialized_details: Sequence[str],
    ) -> int:
        if not records:
            return 0
        hw_df = pd.DataFrame(
            {
                "log_uid": [
                    f"{r.get('run_id') or 'na'}:{r['metric']}:{r['drift_type']}:{uuid4().hex[:8]}"
                    for r in records
                ],
                "run_id": ["" if r.get("run_id") is None else str(r["run_id"]) for r in records],
                "timestamp": pd.to_datetime([r["timestamp"] for r in records], utc=True),
                "metric": [str(r["metric"]) for r in records],
                "drift_type": [str(r["drift_type"]) for r in records],
                "test_name": [
                    "" if r.get("test_name") is None else str(r["test_name"]) for r in records
                ],
                "score": pd.Series(
                    [float(r["score"]) if r.get("score") is not None else np.nan for r in records],
                    dtype="float64",
                ),
                "secondary_score": pd.Series(
                    [
                        float(r["secondary_score"])
                        if r.get("secondary_score") is not None
                        else np.nan
                        for r in records
                    ],
                    dtype="float64",
                ),
                "threshold": pd.Series([float(r["threshold"]) for r in records], dtype="float64"),
                "status": [str(r["status"]) for r in records],
                "retrain_triggered": pd.Series(
                    [bool(r["retrain_triggered"]) for r in records], dtype="bool"
                ),
                "action_taken": [str(r["action_taken"]) for r in records],
                "error_message": [
                    "" if r.get("error_message") is None else str(r["error_message"])
                    for r in records
                ],
                "details_json": list(serialized_details),
            }
        )
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_or_create_feature_group(
                name=self.settings.hopsworks_drift_log_fg,
                version=self.settings.hopsworks_feature_group_version,
                description="Data drift (PSI/KS) and model drift (RMSE ratio) detection logs.",
                primary_key=["log_uid"],
                event_time="timestamp",
                online_enabled=True,
            )
            fg.insert(hw_df, write_options={"wait_for_job": False})
            return len(hw_df)
        except Exception as exc:
            raise StorageError(f"Hopsworks drift_log Feature Group insert failed: {exc}") from exc

    def read_drift_log(
        self,
        *,
        run_id: str | None = None,
        metric: str | None = None,
        drift_type: str | None = None,
        status: str | None = None,
        latest_run_only: bool = False,
        limit: int | None = None,
        prefer_hopsworks: bool = False,
    ) -> list[dict[str, Any]]:
        """Read drift check records ordered newest-first (`timestamp DESC, id DESC`)."""
        self._require_configured()
        if limit is not None:
            limit = _require_int(limit, "limit", min_val=0)
            if limit == 0:
                return []

        if (
            prefer_hopsworks
            and not self.settings.is_offline_mode
            and self.settings.has_hopsworks_credentials
        ) or (
            not self.settings.is_offline_mode
            and not self.settings.has_supabase_credentials
            and self.settings.has_hopsworks_credentials
        ):
            return self._read_hopsworks_drift_log(
                run_id=run_id,
                metric=metric,
                drift_type=drift_type,
                status=status,
                latest_run_only=latest_run_only,
                limit=limit,
            )

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            resolved_run_id = self._resolve_latest_run_id(
                "drift_log", "timestamp", eq_filter=("metric", metric)
            )
            if resolved_run_id is None:
                return []

        if self.settings.is_offline_mode or not self.settings.has_supabase_credentials:
            if not self.local_db_path.exists():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if resolved_run_id is not None:
                clauses.append("run_id = ?")
                params.append(resolved_run_id)
            if metric is not None:
                clauses.append("metric = ?")
                params.append(metric)
            if drift_type is not None:
                clauses.append("drift_type = ?")
                params.append(drift_type)
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            limit_sql = f"LIMIT {limit}" if limit is not None else ""
            with self._sqlite_connection() as conn:
                raw_rows = conn.execute(
                    f"SELECT * FROM drift_log {where_sql} ORDER BY timestamp DESC, id DESC {limit_sql}",
                    params,
                ).fetchall()
            rows = [dict(r) for r in raw_rows]
        else:
            filters: list[tuple[str, str, Any]] = []
            if resolved_run_id is not None:
                filters.append(("eq", "run_id", resolved_run_id))
            if metric is not None:
                filters.append(("eq", "metric", metric))
            if drift_type is not None:
                filters.append(("eq", "drift_type", drift_type))
            if status is not None:
                filters.append(("eq", "status", status))
            rows = self._read_supabase_paginated(
                "drift_log",
                filters=filters,
                orders=(("timestamp", True), ("id", True)),
                limit=limit,
            )

        for r in rows:
            r["retrain_triggered"] = bool(r.get("retrain_triggered", False))
            r["details_json"] = _parse_json_or_default(r.get("details_json"), {})
        return rows

    def _read_hopsworks_drift_log(
        self,
        *,
        run_id: str | None,
        metric: str | None,
        drift_type: str | None,
        status: str | None,
        latest_run_only: bool,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        try:
            fs = self._get_hopsworks_feature_store()
            fg = fs.get_feature_group(
                name=self.settings.hopsworks_drift_log_fg,
                version=self.settings.hopsworks_feature_group_version,
            )
            df = fg.read()
        except Exception as exc:
            raise StorageError(f"Hopsworks drift_log Feature Group read failed: {exc}") from exc

        if df.empty:
            return []
        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        if metric is not None:
            work = work.loc[work["metric"] == metric]
        if work.empty:
            return []

        resolved_run_id = run_id
        if latest_run_only and resolved_run_id is None:
            max_ts = work["timestamp"].max()
            raw_latest = work.loc[work["timestamp"] == max_ts, "run_id"].iloc[-1]
            resolved_run_id = "" if raw_latest is None else str(raw_latest)
            if not resolved_run_id:
                return []

        if resolved_run_id is not None:
            work = work.loc[work["run_id"] == resolved_run_id]
        if drift_type is not None:
            work = work.loc[work["drift_type"] == drift_type]
        if status is not None:
            work = work.loc[work["status"] == status]
        if work.empty:
            return []

        work = work.sort_values(by=["timestamp", "metric", "drift_type"], ascending=[False, True, True])
        if limit is not None:
            work = work.head(limit)

        records: list[dict[str, Any]] = []
        for row in work.to_dict(orient="records"):
            raw_run_id = row.get("run_id")
            records.append(
                {
                    "run_id": None if raw_run_id is None or raw_run_id == "" else str(raw_run_id),
                    "timestamp": _utc_iso(row["timestamp"]),
                    "metric": str(row["metric"]),
                    "drift_type": str(row["drift_type"]),
                    "test_name": _optional_str(
                        None if row.get("test_name") == "" else row.get("test_name"),
                        "test_name",
                    ),
                    "score": _optional_finite_float(row.get("score"), "score"),
                    "secondary_score": _optional_finite_float(
                        row.get("secondary_score"), "secondary_score"
                    ),
                    "threshold": float(row["threshold"]),
                    "status": str(row["status"]),
                    "retrain_triggered": bool(row["retrain_triggered"]),
                    "action_taken": str(row["action_taken"]),
                    "error_message": _optional_str(
                        None if row.get("error_message") == "" else row.get("error_message"),
                        "error_message",
                    ),
                    "details_json": _parse_json_or_default(row.get("details_json"), {}),
                }
            )
        return records
