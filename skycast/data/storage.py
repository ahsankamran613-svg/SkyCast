"""Dual persistence layer: Hopsworks Feature Store, Supabase Postgres, and Local SQLite fallback."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skycast.config.settings import POLLUTANT_COLUMNS, Settings, TARGET_METRICS, get_settings
from skycast.features.engineer import (
    build_features_json_payload,
    get_model_feature_columns,
    sanitize_json_scalar,
)

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when a configured storage backend fails."""


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
        horizon_hour INTEGER NOT NULL,
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
    CREATE TABLE IF NOT EXISTS shap_lime_explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        train_run_id TEXT NOT NULL,
        computed_at_utc TEXT NOT NULL,
        metric TEXT NOT NULL,
        model_family TEXT NOT NULL,
        method TEXT NOT NULL,
        scope TEXT NOT NULL,
        base_value REAL,
        predicted_value REAL,
        reference_timestamp_utc TEXT,
        feature_attributions TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        UNIQUE (train_run_id, metric, method, scope)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        target_timestamp_utc TEXT,
        metric TEXT NOT NULL,
        severity TEXT NOT NULL,
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
        trigger_reason TEXT NOT NULL,
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
        status TEXT NOT NULL,
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
        status TEXT NOT NULL,
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
        drift_type TEXT NOT NULL,
        test_name TEXT,
        score REAL,
        secondary_score REAL,
        threshold REAL NOT NULL,
        status TEXT NOT NULL,
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
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL UNIQUE,
        pipeline_type TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at_utc TEXT NOT NULL,
        completed_at_utc TEXT,
        storage_mode TEXT NOT NULL DEFAULT 'local_offline',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        error_message TEXT,
        created_at TEXT NOT NULL
    )
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
    if ts.microsecond == 0:
        return ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


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

    def _get_supabase(self) -> Any:
        if self._supabase_client is None:
            from supabase import create_client

            self._supabase_client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_role_key,
            )
        return self._supabase_client

    def _get_hopsworks_feature_store(self) -> Any:
        if self._hopsworks_project is None:
            import hopsworks

            self._hopsworks_project = hopsworks.login(
                host=self.settings.hopsworks_host,
                project=self.settings.hopsworks_project_name,
                api_key_value=self.settings.hopsworks_api_key,
            )
        return self._hopsworks_project.get_feature_store()

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
        now_iso = _now_utc_iso()
        started_iso = _utc_iso(started_at_utc) if started_at_utc else now_iso
        completed_iso = _utc_iso(completed_at_utc) if completed_at_utc else None
        meta_clean = sanitize_json_scalar(metadata or {})

        if self.storage_mode == "unconfigured":
            raise StorageError(
                "No storage backend is active for mode 'unconfigured': "
                "configure Supabase/Hopsworks credentials or enable offline mode."
            )

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
                        status = excluded.status,
                        completed_at_utc = excluded.completed_at_utc,
                        storage_mode = excluded.storage_mode,
                        metadata_json = excluded.metadata_json,
                        error_message = excluded.error_message
                    """,
                    (
                        run_id,
                        pipeline_type,
                        status,
                        started_iso,
                        completed_iso,
                        self.storage_mode,
                        json.dumps(meta_clean),
                        error_message,
                        now_iso,
                    ),
                )
            return

        try:
            client = self._get_supabase()
            client.table("pipeline_runs").upsert(
                {
                    "run_id": run_id,
                    "pipeline_type": pipeline_type,
                    "status": status,
                    "started_at_utc": started_iso,
                    "completed_at_utc": completed_iso,
                    "storage_mode": self.storage_mode,
                    "metadata_json": meta_clean,
                    "error_message": error_message,
                },
                on_conflict="run_id",
            ).execute()
        except Exception as exc:
            raise StorageError(f"Supabase pipeline_runs write failed: {exc}") from exc

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
