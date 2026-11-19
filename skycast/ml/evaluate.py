"""Chronological embargoed splitting, train-only scaling, bounds, circular wind, and multi-horizon scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from skycast.config.settings import (
    METRIC_SPECS,
    TARGET_METRICS,
    Settings,
    get_settings,
)
from skycast.features.engineer import get_model_feature_columns, sanitize_json_scalar

_CIRCULAR_MAX_SAFE = 359.9999
_EPS = 1e-15

_NON_FEATURE_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "id",
        "timestamp_utc",
        "timestamp_epoch_ms",
        "has_full_24h_targets",
        "created_at",
        "updated_at",
        "condition_symbol",
        "sources_available",
        "is_imputed",
        "imputation_tier",
        "imputed_columns",
        "features_json",
        "wind_direction_10m",
    }
)

_FAMILY_TO_SERIES_COL: dict[str, str] = {
    "SARIMAX": "sarimax",
    "sarimax": "sarimax",
    "RandomForest": "random_forest",
    "random_forest": "random_forest",
    "Ridge": "ridge",
    "ridge": "ridge",
    "LSTM": "lstm",
    "lstm": "lstm",
    "Naive24h": "naive_24h",
    "naive_24h": "naive_24h",
}
_SERIES_MODEL_COLS: tuple[str, ...] = ("sarimax", "random_forest", "ridge", "lstm", "naive_24h")


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Chronological train/val/test split with H-hour embargoes and train-fitted scaler."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    embargo_hours: int
    embargo_train_val_df: pd.DataFrame | None = None
    embargo_val_test_df: pd.DataFrame | None = None


def _is_excluded_column(col: str) -> bool:
    if col in _NON_FEATURE_EXCLUSIONS:
        return True
    if col.startswith("target_"):
        return True
    if col.endswith("_openmeteo") or col.endswith("_metno"):
        return True
    return False


def _resolve_feature_columns(
    df: pd.DataFrame,
    cfg: Settings,
    feature_columns: Sequence[str] | None,
) -> list[str]:
    if feature_columns is not None:
        resolved = list(feature_columns)
        if not resolved:
            raise ValueError("feature_columns cannot be empty.")
        if len(set(resolved)) != len(resolved):
            raise ValueError("feature_columns must not contain duplicate names.")
        missing = [c for c in resolved if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature_columns missing from DataFrame: {missing}")
        invalid = [c for c in resolved if _is_excluded_column(c)]
        if invalid:
            raise ValueError(f"Excluded non-feature columns in feature_columns: {invalid}")
        return resolved

    resolved = list(get_model_feature_columns(df, settings=cfg))
    seen = set(resolved)
    for col in df.columns:
        if col in seen or _is_excluded_column(col):
            continue
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            continue
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            resolved.append(col)
            seen.add(col)

    if not resolved:
        raise ValueError("No valid feature columns could be resolved from DataFrame.")
    return resolved


def _extract_finite_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    split_name: str,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    bad_cols: list[str] = []
    for col in feature_columns:
        series = df[col]
        try:
            if pd.api.types.is_bool_dtype(series.dtype):
                col_arr = series.to_numpy(dtype=np.float64)
            else:
                col_arr = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            bad_cols.append(col)
            continue
        if not np.isfinite(col_arr).all():
            bad_cols.append(col)
            continue
        arrays.append(col_arr)

    if bad_cols:
        raise ValueError(
            f"Split '{split_name}' contains NaN, Inf, or non-numeric values in feature column(s): {bad_cols}"
        )
    return np.column_stack(arrays)


def _validate_split_targets(df: pd.DataFrame, split_name: str) -> None:
    if "has_full_24h_targets" in df.columns:
        flags = df["has_full_24h_targets"]
        if flags.isna().any() or not flags.astype(bool).all():
            raise ValueError(
                f"Split '{split_name}' contains rows with has_full_24h_targets != True."
            )

    target_cols = [c for c in df.columns if c.startswith("target_")]
    bad_targets: list[str] = []
    for col in target_cols:
        try:
            arr = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            bad_targets.append(col)
            continue
        if not np.isfinite(arr).all():
            bad_targets.append(col)
    if bad_targets:
        raise ValueError(
            f"Split '{split_name}' contains NaN, Inf, or non-numeric values in target column(s): {bad_targets}"
        )


def split_and_scale(
    df: pd.DataFrame,
    *,
    settings: Settings | None = None,
    feature_columns: Sequence[str] | None = None,
) -> ChronologicalSplit:
    """Chronologically split with H-hour embargoes and fit StandardScaler on train only."""
    cfg = settings or get_settings()
    embargo = int(cfg.forecast_horizon_hours)
    if embargo < 1:
        raise ValueError(f"forecast_horizon_hours must be >= 1, got {embargo}.")
    ratio_sum = cfg.train_split_ratio + cfg.val_split_ratio + cfg.test_split_ratio
    if (
        cfg.train_split_ratio <= 0.0
        or cfg.val_split_ratio <= 0.0
        or cfg.test_split_ratio <= 0.0
        or abs(ratio_sum - 1.0) > 1e-6
    ):
        raise ValueError("Train/Val/Test split ratios must be positive and sum to 1.0.")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame must be non-empty for split_and_scale.")
    if "timestamp_utc" not in df.columns:
        raise ValueError("Input DataFrame must include 'timestamp_utc'.")

    ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if ts_utc.isna().any():
        raise ValueError("Input DataFrame contains invalid or NaT 'timestamp_utc' values.")

    ordered = df.copy()
    ordered["timestamp_utc"] = ts_utc
    ordered = ordered.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    if ordered["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate 'timestamp_utc' rows are not allowed in chronological split.")

    n_total = len(ordered)
    n_eff = n_total - 2 * embargo
    if n_eff < 3:
        raise ValueError(
            f"Insufficient rows ({n_total}) after applying 2 * {embargo}h embargo (N_eff={n_eff})."
        )

    n_train = int(n_eff * cfg.train_split_ratio)
    n_val = int(n_eff * cfg.val_split_ratio)
    n_test = n_eff - n_train - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Split yielded empty partition (train={n_train}, val={n_val}, test={n_test})."
        )

    val_start = n_train + embargo
    val_end = val_start + n_val
    test_start = val_end + embargo

    train_df = ordered.iloc[:n_train].reset_index(drop=True)
    embargo_train_val_df = ordered.iloc[n_train:val_start].reset_index(drop=True)
    val_df = ordered.iloc[val_start:val_end].reset_index(drop=True)
    embargo_val_test_df = ordered.iloc[val_end:test_start].reset_index(drop=True)
    test_df = ordered.iloc[test_start:].reset_index(drop=True)

    required_gap = pd.Timedelta(hours=embargo)
    gap_train_val = val_df["timestamp_utc"].iloc[0] - train_df["timestamp_utc"].iloc[-1]
    gap_val_test = test_df["timestamp_utc"].iloc[0] - val_df["timestamp_utc"].iloc[-1]
    if gap_train_val <= required_gap or gap_val_test <= required_gap:
        raise ValueError(
            f"Timestamp embargo gap must exceed {embargo}h "
            f"(got train->val={gap_train_val}, val->test={gap_val_test})."
        )

    resolved_features = _resolve_feature_columns(ordered, cfg, feature_columns)

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        _validate_split_targets(split_df, split_name)

    X_train = _extract_finite_feature_matrix(train_df, resolved_features, "train")
    X_val = _extract_finite_feature_matrix(val_df, resolved_features, "val")
    X_test = _extract_finite_feature_matrix(test_df, resolved_features, "test")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return ChronologicalSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=resolved_features,
        scaler=scaler,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        embargo_hours=embargo,
        embargo_train_val_df=embargo_train_val_df,
        embargo_val_test_df=embargo_val_test_df,
    )


def _resolve_horizons(
    settings: Settings | None = None,
    horizons: Sequence[int] | None = None,
) -> list[int]:
    if horizons is None:
        cfg = settings or get_settings()
        return list(range(1, int(cfg.forecast_horizon_hours) + 1))
    resolved = [int(h) for h in horizons]
    if not resolved or any(h < 1 for h in resolved):
        raise ValueError("horizons must be a non-empty sequence of integers >= 1.")
    return resolved


def _validate_metric(metric: str) -> None:
    if metric not in METRIC_SPECS:
        raise ValueError(f"Unknown metric '{metric}'. Expected one of {TARGET_METRICS}.")


def get_target_column_names(
    metric: str,
    *,
    settings: Settings | None = None,
    horizons: Sequence[int] | None = None,
) -> list[str]:
    """Return ordered multi-horizon target column names for a metric."""
    _validate_metric(metric)
    return [f"target_{metric}_t{h}" for h in _resolve_horizons(settings, horizons)]


def get_wind_component_target_columns(
    *,
    settings: Settings | None = None,
    horizons: Sequence[int] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (sin_columns, cos_columns) target column lists for wind_direction_10m."""
    hs = _resolve_horizons(settings, horizons)
    return (
        [f"target_wind_dir_sin_t{h}" for h in hs],
        [f"target_wind_dir_cos_t{h}" for h in hs],
    )


def _extract_numeric_target_block(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("DataFrame must be non-empty when extracting target matrices.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required target column(s): {missing}")
    mat = df.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(mat).all():
        raise ValueError("Extracted target matrix contains NaN or Inf values.")
    return mat


def extract_target_matrix(
    df: pd.DataFrame,
    metric: str,
    *,
    settings: Settings | None = None,
    horizons: Sequence[int] | None = None,
) -> np.ndarray:
    """Extract finite (n_samples, H) float64 target matrix for `metric`."""
    cols = get_target_column_names(metric, settings=settings, horizons=horizons)
    return _extract_numeric_target_block(df, cols)


def extract_wind_component_targets(
    df: pd.DataFrame,
    *,
    settings: Settings | None = None,
    horizons: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite (Y_sin, Y_cos) target matrices of shape (n_samples, H)."""
    sin_cols, cos_cols = get_wind_component_target_columns(settings=settings, horizons=horizons)
    return (
        _extract_numeric_target_block(df, sin_cols),
        _extract_numeric_target_block(df, cos_cols),
    )


def reconstruct_wind_direction(
    sin_arr: np.ndarray | float,
    cos_arr: np.ndarray | float,
) -> np.ndarray | float:
    """Reconstruct wind direction degrees in [0, 360) from (sin, cos) components."""
    is_scalar = np.ndim(sin_arr) == 0 and np.ndim(cos_arr) == 0
    sin_a = np.asarray(sin_arr, dtype=np.float64)
    cos_a = np.asarray(cos_arr, dtype=np.float64)
    if sin_a.shape != cos_a.shape:
        raise ValueError(
            f"sin_arr and cos_arr shapes must match, got {sin_a.shape} vs {cos_a.shape}."
        )
    if not (np.isfinite(sin_a).all() and np.isfinite(cos_a).all()):
        raise ValueError("sin_arr and cos_arr must contain only finite values.")

    deg = np.degrees(np.arctan2(sin_a, cos_a)) % 360.0
    degenerate = (np.abs(sin_a) < 1e-12) & (np.abs(cos_a) < 1e-12)
    deg = np.where(degenerate | (deg >= 360.0), 0.0, deg)
    if is_scalar:
        return float(deg.item())
    return deg


def compute_empirical_residual_quantiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    alpha: float | None = None,
    settings: Settings | None = None,
    metric: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-horizon empirical residual quantiles (q_low, q_high) on train residuals."""
    cfg = settings or get_settings()
    resolved_alpha = float(cfg.sarimax_conf_alpha if alpha is None else alpha)
    if not (0.0 < resolved_alpha < 1.0) or not math.isfinite(resolved_alpha):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {resolved_alpha}.")

    is_circular = False
    if metric is not None:
        _validate_metric(metric)
        is_circular = METRIC_SPECS[metric].is_circular

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim != 2 or yt.shape[0] < 1 or yt.shape[1] < 1:
        raise ValueError(
            f"y_true and y_pred must have matching 2D non-empty shapes, got {yt.shape} vs {yp.shape}."
        )
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("y_true and y_pred must contain only finite values.")

    if is_circular:
        residuals = ((yt - yp + 180.0) % 360.0) - 180.0
    else:
        residuals = yt - yp

    q_low = np.quantile(residuals, resolved_alpha / 2.0, axis=0)
    q_high = np.quantile(residuals, 1.0 - (resolved_alpha / 2.0), axis=0)
    return np.asarray(q_low, dtype=np.float64), np.asarray(q_high, dtype=np.float64)


def _clamp_with_spec(arr: np.ndarray, metric: str) -> np.ndarray:
    spec = METRIC_SPECS[metric]
    flat = [spec.clamp(float(v)) for v in arr.ravel()]
    clamped = np.asarray(flat, dtype=np.float64).reshape(arr.shape)
    if spec.is_circular:
        clamped = np.where(clamped >= 360.0, 0.0, clamped)
        clamped = np.clip(clamped, 0.0, _CIRCULAR_MAX_SAFE)
    return clamped


def clamp_predictions_and_bounds(
    metric: str,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clamp predictions and bounds via METRIC_SPECS and guarantee lower <= pred <= upper."""
    _validate_metric(metric)
    spec = METRIC_SPECS[metric]

    pred_a = np.asarray(pred, dtype=np.float64)
    lower_a = np.asarray(lower, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    if not (pred_a.shape == lower_a.shape == upper_a.shape):
        raise ValueError(
            f"pred, lower, and upper shapes must match, got {pred_a.shape}, {lower_a.shape}, {upper_a.shape}."
        )
    if not (
        np.isfinite(pred_a).all() and np.isfinite(lower_a).all() and np.isfinite(upper_a).all()
    ):
        raise ValueError("pred, lower, and upper must contain only finite values.")

    pred_c = _clamp_with_spec(pred_a, metric)
    if spec.is_circular:
        lower_c = np.clip(lower_a, 0.0, _CIRCULAR_MAX_SAFE)
        upper_c = np.clip(upper_a, 0.0, _CIRCULAR_MAX_SAFE)
    else:
        lower_c = _clamp_with_spec(lower_a, metric)
        upper_c = _clamp_with_spec(upper_a, metric)

    lower_c = np.minimum(lower_c, pred_c)
    upper_c = np.maximum(upper_c, pred_c)
    return pred_c, lower_c, upper_c


def apply_residual_quantiles(
    metric: str,
    pred: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-horizon residual quantiles to predictions and clamp safely."""
    _validate_metric(metric)
    pred_a = np.asarray(pred, dtype=np.float64)
    ql = np.asarray(q_low, dtype=np.float64)
    qh = np.asarray(q_high, dtype=np.float64)
    if not (np.isfinite(pred_a).all() and np.isfinite(ql).all() and np.isfinite(qh).all()):
        raise ValueError("pred, q_low, and q_high must contain only finite values.")

    pred_w = _clamp_with_spec(pred_a, metric)
    raw_lower = pred_w + ql
    raw_upper = pred_w + qh
    return clamp_predictions_and_bounds(metric, pred_w, raw_lower, raw_upper)


def score_multi_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str,
) -> dict[str, float]:
    """Compute flattened 1..H RMSE, MAE, and R² (circular-aware for wind_direction_10m)."""
    _validate_metric(metric)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape or yt.ndim not in (1, 2) or yt.size < 1:
        raise ValueError(
            f"y_true and y_pred must have matching 1D/2D non-empty shapes, got {yt.shape} vs {yp.shape}."
        )
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("y_true and y_pred must contain only finite values.")

    yt_f = yt.ravel()
    yp_f = yp.ravel()

    if METRIC_SPECS[metric].is_circular:
        err = ((yp_f - yt_f + 180.0) % 360.0) - 180.0
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

        rad = np.deg2rad(yt_f)
        mean_sin = float(np.mean(np.sin(rad)))
        mean_cos = float(np.mean(np.cos(rad)))
        if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
            circ_mean = 0.0
        else:
            circ_mean = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)
            if circ_mean >= 360.0:
                circ_mean = 0.0
        dev = ((yt_f - circ_mean + 180.0) % 360.0) - 180.0
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum(dev ** 2))
        r2 = (1.0 if ss_res <= _EPS else 0.0) if ss_tot <= _EPS else float(1.0 - (ss_res / ss_tot))
    else:
        err = yp_f - yt_f
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt_f - float(np.mean(yt_f))) ** 2))
        r2 = (1.0 if ss_res <= _EPS else 0.0) if ss_tot <= _EPS else float(1.0 - (ss_res / ss_tot))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_improvement_over_naive_pct(model_rmse: float, naive_rmse: float) -> float:
    """Compute percentage RMSE improvement over Naive24h with finite zero-baseline guards."""
    m = float(model_rmse)
    n = float(naive_rmse)
    if not (math.isfinite(m) and math.isfinite(n)) or m < 0.0 or n < 0.0:
        raise ValueError(
            f"model_rmse and naive_rmse must be finite and >= 0, got {model_rmse!r}, {naive_rmse!r}."
        )
    if n <= _EPS:
        return 0.0 if m <= _EPS else -100.0
    return float(((n - m) / n) * 100.0)


def _extract_horizon_series_vector(
    values: np.ndarray,
    n_rows: int,
    horizon: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"Series '{label}' length {arr.shape[0]} does not match origin_timestamps ({n_rows})."
            )
        return arr
    if arr.ndim == 2:
        if arr.shape[0] != n_rows or arr.shape[1] < horizon:
            raise ValueError(
                f"Series '{label}' shape {arr.shape} cannot provide horizon h={horizon} for {n_rows} rows."
            )
        return arr[:, horizon - 1]
    raise ValueError(f"Series '{label}' must be 1D or 2D, got ndim={arr.ndim}.")


def build_comparison_series_rows(
    train_run_id: str,
    metric: str,
    origin_timestamps: pd.Series | pd.DatetimeIndex | Sequence[Any],
    actual: np.ndarray,
    predictions_by_family: Mapping[str, np.ndarray | None],
    *,
    settings: Settings | None = None,
    horizon_hours: int | None = None,
) -> list[dict[str, Any]]:
    """Build sanitized t+H rows matching `model_comparison_series` schema."""
    if not isinstance(train_run_id, str) or not train_run_id.strip():
        raise ValueError("train_run_id must be a non-empty string.")
    _validate_metric(metric)

    cfg = settings or get_settings()
    H = int(cfg.forecast_horizon_hours if horizon_hours is None else horizon_hours)
    if H < 1:
        raise ValueError(f"horizon_hours must be >= 1, got {H}.")

    ts_idx = pd.to_datetime(origin_timestamps, utc=True, errors="coerce")
    if len(ts_idx) == 0 or ts_idx.isna().any():
        raise ValueError("origin_timestamps must be non-empty and contain valid timestamps.")
    n_rows = len(ts_idx)

    actual_vec = _extract_horizon_series_vector(actual, n_rows, H, "actual")

    series_vectors: dict[str, np.ndarray | None] = {col: None for col in _SERIES_MODEL_COLS}
    for raw_key, arr in predictions_by_family.items():
        if raw_key not in _FAMILY_TO_SERIES_COL:
            raise ValueError(f"Unknown model family key in predictions_by_family: '{raw_key}'.")
        canonical_col = _FAMILY_TO_SERIES_COL[raw_key]
        if series_vectors[canonical_col] is not None and arr is not None:
            raise ValueError(f"Duplicate entries mapping to '{canonical_col}' in predictions_by_family.")
        if arr is not None:
            vec = _extract_horizon_series_vector(arr, n_rows, H, raw_key)
            if METRIC_SPECS[metric].is_circular:
                finite_mask = np.isfinite(vec)
                if finite_mask.any():
                    vec = vec.copy()
                    vec[finite_mask] = _clamp_with_spec(vec[finite_mask], metric)
            series_vectors[canonical_col] = vec

    target_timestamps = pd.DatetimeIndex(ts_idx) + pd.Timedelta(hours=H)
    run_id_clean = train_run_id.strip()

    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        row: dict[str, Any] = {
            "train_run_id": run_id_clean,
            "metric": metric,
            "timestamp_utc": target_timestamps[i].isoformat(),
            "actual": sanitize_json_scalar(actual_vec[i]),
        }
        for col in _SERIES_MODEL_COLS:
            vec = series_vectors[col]
            row[col] = None if vec is None else sanitize_json_scalar(vec[i])
        rows.append(row)

    json.dumps(rows, allow_nan=False)
    return rows
