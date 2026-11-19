"""Automated Exploratory Data Analysis (EDA) for SkyCast Lahore weather & AQ data."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from skycast.config.settings import (
    AUX_WEATHER_COLUMNS,
    CIRCULAR_METRICS,
    METRIC_SPECS,
    POLLUTANT_COLUMNS,
    TARGET_METRICS,
    Settings,
    get_settings,
)
from skycast.features.engineer import sanitize_json_scalar

_SEASONAL_PERIOD_HOURS = 24
_MIN_SEASONAL_DECOMPOSE_ROWS = 2 * _SEASONAL_PERIOD_HOURS  # 48

_EXTRA_COLUMN_META: dict[str, tuple[str, str]] = {
    "pm2_5": ("PM2.5", "µg/m³"),
    "pm10": ("PM10", "µg/m³"),
    "ozone": ("Ozone", "µg/m³"),
    "nitrogen_dioxide": ("Nitrogen Dioxide", "µg/m³"),
    "sulphur_dioxide": ("Sulphur Dioxide", "µg/m³"),
    "carbon_monoxide": ("Carbon Monoxide", "µg/m³"),
    "apparent_temperature": ("Feels Like", "°C"),
    "weather_code": ("Weather Code", "WMO"),
    "surface_pressure": ("Surface Pressure", "hPa"),
    "cloud_cover": ("Cloud Cover", "%"),
}

_CORRELATION_HIGHLIGHT_PAIRS: dict[str, tuple[str, str]] = {
    "pm2_5_vs_wind_speed_10m": ("pm2_5", "wind_speed_10m"),
    "pm2_5_vs_relative_humidity_2m": ("pm2_5", "relative_humidity_2m"),
    "us_aqi_vs_pm2_5": ("us_aqi", "pm2_5"),
    "temperature_2m_vs_relative_humidity_2m": ("temperature_2m", "relative_humidity_2m"),
}


def _column_label_unit(column: str) -> tuple[str, str]:
    if column in METRIC_SPECS:
        spec = METRIC_SPECS[column]
        return spec.label, spec.unit
    return _EXTRA_COLUMN_META.get(column, (column, ""))


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _compute_circular_mean_deg(values: np.ndarray) -> float:
    rad = np.deg2rad(values)
    sin_mean = float(np.mean(np.sin(rad)))
    cos_mean = float(np.mean(np.cos(rad)))
    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return 0.0
    deg = float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0)
    return 0.0 if deg >= 360.0 else deg


def _compute_distributions(df: pd.DataFrame, sample_count: int) -> dict[str, dict[str, Any]]:
    dist_cols: list[str] = [*TARGET_METRICS, *POLLUTANT_COLUMNS]
    for aux_col in AUX_WEATHER_COLUMNS:
        if aux_col in df.columns and aux_col not in dist_cols:
            if pd.api.types.is_numeric_dtype(df[aux_col]) or _to_numeric_clean(df[aux_col]).notna().any():
                dist_cols.append(aux_col)

    distributions: dict[str, dict[str, Any]] = {}
    for col in dist_cols:
        label, unit = _column_label_unit(col)
        if col not in df.columns:
            distributions[col] = {
                "present": False,
                "status": "missing_column",
                "label": label,
                "unit": unit,
                "valid_count": 0,
                "missing_count": sample_count,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "q05": None,
                "q25": None,
                "q50": None,
                "q75": None,
                "q95": None,
                "histogram": [],
            }
            continue

        cleaned = _to_numeric_clean(df[col])
        valid = cleaned.dropna()
        valid_count = int(len(valid))
        missing_count = int(sample_count - valid_count)

        if valid_count == 0:
            entry: dict[str, Any] = {
                "present": True,
                "status": "all_missing",
                "label": label,
                "unit": unit,
                "valid_count": 0,
                "missing_count": missing_count,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "q05": None,
                "q25": None,
                "q50": None,
                "q75": None,
                "q95": None,
                "histogram": [],
            }
            if col in CIRCULAR_METRICS:
                entry["circular_mean_deg"] = None
            distributions[col] = entry
            continue

        arr = valid.to_numpy(dtype="float64")
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        q05, q25, q50, q75, q95 = (
            float(q) for q in np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
        )
        bins = 10 if max_v > min_v else 1
        counts, edges = np.histogram(arr, bins=bins)
        histogram = [
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

        entry = {
            "present": True,
            "status": "available",
            "label": label,
            "unit": unit,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if valid_count >= 2 else None,
            "min": min_v,
            "max": max_v,
            "q05": q05,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q95": q95,
            "histogram": histogram,
        }
        if col in CIRCULAR_METRICS:
            entry["circular_mean_deg"] = _compute_circular_mean_deg(arr)
        distributions[col] = entry

    return distributions


def _compute_diurnal_heatmap(df: pd.DataFrame, tz_name: str) -> dict[str, Any]:
    local_hours = pd.DatetimeIndex(df["timestamp_utc"]).tz_convert(tz_name).hour.to_numpy()
    heatmap_cols: list[str] = [
        *TARGET_METRICS,
        *[c for c in POLLUTANT_COLUMNS if c in df.columns],
    ]
    metrics_map: dict[str, list[dict[str, Any]]] = {}

    for col in heatmap_cols:
        if col not in df.columns:
            metrics_map[col] = [
                {"hour": h, "mean": None, "median": None, "std": None, "count": 0}
                for h in range(24)
            ]
            continue

        cleaned = _to_numeric_clean(df[col]).to_numpy(dtype="float64")
        hourly_rows: list[dict[str, Any]] = []
        for h in range(24):
            bucket = cleaned[local_hours == h]
            valid = bucket[np.isfinite(bucket)]
            count = int(len(valid))
            if count == 0:
                hourly_rows.append(
                    {"hour": h, "mean": None, "median": None, "std": None, "count": 0}
                )
            else:
                hourly_rows.append(
                    {
                        "hour": h,
                        "mean": float(np.mean(valid)),
                        "median": float(np.median(valid)),
                        "std": float(np.std(valid, ddof=1)) if count >= 2 else None,
                        "count": count,
                    }
                )
        metrics_map[col] = hourly_rows

    return {
        "timezone": tz_name,
        "hours": list(range(24)),
        "metrics": metrics_map,
    }


def _compute_correlation_matrix(df: pd.DataFrame) -> dict[str, Any]:
    columns = [c for c in (*TARGET_METRICS, *POLLUTANT_COLUMNS) if c in df.columns]
    if not columns:
        return {
            "columns": [],
            "method": "pearson",
            "min_periods": 3,
            "matrix": {},
            "constant_columns": [],
            "highlights": {k: None for k in _CORRELATION_HIGHLIGHT_PAIRS},
        }

    numeric_df = pd.DataFrame({col: _to_numeric_clean(df[col]) for col in columns}, index=df.index)
    constant_columns = [
        col for col in columns if int(numeric_df[col].nunique(dropna=True)) <= 1
    ]
    corr_df = numeric_df.corr(method="pearson", min_periods=3)

    matrix: dict[str, dict[str, float | None]] = {}
    for row_col in columns:
        row_map: dict[str, float | None] = {}
        for col_b in columns:
            val = corr_df.loc[row_col, col_b]
            row_map[col_b] = None if pd.isna(val) or not math.isfinite(float(val)) else float(val)
        matrix[row_col] = row_map

    highlights: dict[str, float | None] = {}
    for key, (col_a, col_b) in _CORRELATION_HIGHLIGHT_PAIRS.items():
        if col_a in matrix and col_b in matrix[col_a]:
            highlights[key] = matrix[col_a][col_b]
        else:
            highlights[key] = None

    return {
        "columns": columns,
        "method": "pearson",
        "min_periods": 3,
        "matrix": matrix,
        "constant_columns": constant_columns,
        "highlights": highlights,
    }


def _longest_contiguous_non_nan(series: pd.Series) -> pd.Series:
    not_na = series.notna()
    if not not_na.any():
        return series.iloc[0:0]
    group_ids = (~not_na).cumsum()
    best_slice = series.iloc[0:0]
    for _, grp in series.loc[not_na].groupby(group_ids.loc[not_na], sort=True):
        if len(grp) >= len(best_slice):
            best_slice = grp
    return best_slice


def _compute_seasonal_decomposition(df: pd.DataFrame, cfg: Settings) -> dict[str, Any]:
    decomp_cols: list[str] = list(TARGET_METRICS)
    if "pm2_5" in df.columns and "pm2_5" not in decomp_cols:
        decomp_cols.append("pm2_5")

    ts_min = df["timestamp_utc"].min()
    ts_max = df["timestamp_utc"].max()
    full_grid = pd.date_range(start=ts_min, end=ts_max, freq="1h", tz="UTC")
    indexed_df = df.set_index("timestamp_utc")
    window_cap = int(cfg.supabase_history_days) * _SEASONAL_PERIOD_HOURS

    results: dict[str, dict[str, Any]] = {}
    for col in decomp_cols:
        if col not in indexed_df.columns:
            results[col] = {
                "status": "unavailable",
                "reason": "missing_column",
                "required_rows": _MIN_SEASONAL_DECOMPOSE_ROWS,
                "valid_count": 0,
                "period": _SEASONAL_PERIOD_HOURS,
            }
            continue

        grid_series = _to_numeric_clean(indexed_df[col]).reindex(full_grid)
        valid_count = int(grid_series.notna().sum())
        if valid_count < _MIN_SEASONAL_DECOMPOSE_ROWS:
            results[col] = {
                "status": "unavailable",
                "reason": "insufficient_rows",
                "required_rows": _MIN_SEASONAL_DECOMPOSE_ROWS,
                "valid_count": valid_count,
                "period": _SEASONAL_PERIOD_HOURS,
            }
            continue

        contiguous = _longest_contiguous_non_nan(grid_series)
        if len(contiguous) < _MIN_SEASONAL_DECOMPOSE_ROWS:
            results[col] = {
                "status": "unavailable",
                "reason": "insufficient_contiguous_non_nan_observations",
                "required_rows": _MIN_SEASONAL_DECOMPOSE_ROWS,
                "longest_contiguous_count": int(len(contiguous)),
                "valid_count": valid_count,
                "period": _SEASONAL_PERIOD_HOURS,
            }
            continue

        decomp = seasonal_decompose(
            contiguous,
            model="additive",
            period=_SEASONAL_PERIOD_HOURS,
            extrapolate_trend="period",
        )
        local_hours = contiguous.index.tz_convert(cfg.lahore_timezone).hour
        seasonal_by_hour = decomp.seasonal.groupby(local_hours).mean()
        seasonal_cycle_24h = [
            {
                "hour": h,
                "seasonal": float(seasonal_by_hour.loc[h]) if h in seasonal_by_hour.index else None,
            }
            for h in range(24)
        ]

        tail_slice = slice(-window_cap, None)
        results[col] = {
            "status": "available",
            "period": _SEASONAL_PERIOD_HOURS,
            "sample_count_used": int(len(contiguous)),
            "seasonal_cycle_24h": seasonal_cycle_24h,
            "summary": {
                "observed_mean": float(decomp.observed.mean()),
                "trend_start": float(decomp.trend.iloc[0]),
                "trend_end": float(decomp.trend.iloc[-1]),
                "seasonal_amplitude": float(decomp.seasonal.max() - decomp.seasonal.min()),
                "residual_std": float(decomp.resid.std(ddof=1)),
            },
            "timestamps_utc": [ts.isoformat() for ts in contiguous.index[tail_slice]],
            "observed": decomp.observed.iloc[tail_slice].tolist(),
            "trend": decomp.trend.iloc[tail_slice].tolist(),
            "seasonal": decomp.seasonal.iloc[tail_slice].tolist(),
            "resid": decomp.resid.iloc[tail_slice].tolist(),
        }

    return results


def _build_imputation_audit(
    df: pd.DataFrame,
    imputation_audit: dict[str, Any] | None,
    sample_count: int,
) -> dict[str, Any]:
    observed_missing: dict[str, int] = {}
    for col in (*TARGET_METRICS, *POLLUTANT_COLUMNS, *AUX_WEATHER_COLUMNS):
        if col in df.columns:
            cleaned = _to_numeric_clean(df[col])
            observed_missing[col] = int(cleaned.isna().sum())

    if imputation_audit is not None:
        if not isinstance(imputation_audit, Mapping):
            raise ValueError("imputation_audit must be a mapping when provided.")
        payload = dict(imputation_audit)
        payload["status"] = "available"
        payload["source"] = "provided"
        payload.setdefault("observed_missing_counts", observed_missing)
        return payload

    has_tier = "imputation_tier" in df.columns
    has_flag = "is_imputed" in df.columns
    if has_tier or has_flag:
        tier_counts: dict[str, int] = {}
        if has_tier:
            tier_series = df["imputation_tier"].astype(str)
            tier_counts = {str(k): int(v) for k, v in tier_series.value_counts().items()}
        if has_flag:
            imputed_rows_count = int(df["is_imputed"].fillna(False).astype(bool).sum())
        else:
            imputed_rows_count = int(
                sum(cnt for tier, cnt in tier_counts.items() if tier.lower() != "none")
            )
        return {
            "status": "available",
            "source": "dataframe_derived",
            "total_rows": sample_count,
            "imputation_tier_present": has_tier,
            "tier_counts": tier_counts,
            "imputed_rows_count": imputed_rows_count,
            "observed_missing_counts": observed_missing,
        }

    return {
        "status": "unavailable",
        "source": "unavailable",
        "reason": "no_imputation_audit_or_tier_columns",
        "total_rows": sample_count,
        "observed_missing_counts": observed_missing,
    }


def compute_eda_summary(
    df: pd.DataFrame,
    *,
    settings: Settings | None = None,
    imputation_audit: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compute an `eda_summaries`-compatible EDA payload strictly parameterized by Settings."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame must be non-empty for EDA computation.")
    if "timestamp_utc" not in df.columns:
        raise ValueError("Input DataFrame must include 'timestamp_utc'.")

    cfg = settings or get_settings()
    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if ts_utc.isna().any():
        raise ValueError("Input DataFrame contains invalid or NaT 'timestamp_utc' values.")

    if run_id is None:
        resolved_run_id = f"eda_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    else:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string.")
        resolved_run_id = run_id.strip()

    work_df = df.copy()
    work_df["timestamp_utc"] = ts_utc
    work_df = (
        work_df.sort_values("timestamp_utc", kind="mergesort")
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .reset_index(drop=True)
    )

    sample_count = int(len(work_df))
    date_range_start = work_df["timestamp_utc"].min().isoformat()
    date_range_end = work_df["timestamp_utc"].max().isoformat()
    computed_at_utc = datetime.now(timezone.utc).isoformat()

    summary_json: dict[str, Any] = {
        "metadata": {
            "city": cfg.city_name,
            "timezone": cfg.lahore_timezone,
            "sample_count": sample_count,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "seasonal_period": _SEASONAL_PERIOD_HOURS,
        },
        "distributions": _compute_distributions(work_df, sample_count),
        "diurnal_heatmap": _compute_diurnal_heatmap(work_df, cfg.lahore_timezone),
        "correlation_matrix": _compute_correlation_matrix(work_df),
        "seasonal_decomposition": _compute_seasonal_decomposition(work_df, cfg),
        "imputation_audit": _build_imputation_audit(work_df, imputation_audit, sample_count),
    }

    payload = {
        "run_id": resolved_run_id,
        "computed_at_utc": computed_at_utc,
        "sample_count": sample_count,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "summary_json": summary_json,
    }
    sanitized = sanitize_json_scalar(payload)
    json.dumps(sanitized, allow_nan=False)
    return sanitized


def save_eda_summary_json(
    summary: Mapping[str, Any],
    output_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Sanitize and persist an EDA summary JSON artifact (defaults to docs/eda_summary.json)."""
    _ = settings or get_settings()
    target_path = Path(output_path) if output_path is not None else Path("docs/eda_summary.json")
    sanitized = sanitize_json_scalar(dict(summary))
    serialized = json.dumps(sanitized, allow_nan=False, indent=2)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(serialized, encoding="utf-8")
    return target_path
