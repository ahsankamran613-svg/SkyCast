"""Documented 3-tier missing-data imputation with circular wind handling and audit logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from skycast.config.settings import (
    METRIC_SPECS,
    POLLUTANT_COLUMNS,
    Settings,
    TARGET_METRICS,
    get_settings,
)
from skycast.data.sources.met_no import (
    compute_steadman_apparent_temperature,
    wmo_code_to_symbol,
)

TIER_NONE = "none"
TIER_1_LINEAR = "tier1_linear"
TIER_2_DIURNAL = "tier2_diurnal_median"
TIER_3_CLIMATOLOGY = "tier3_climatology_ffill"

_TIER_SEVERITY: dict[str, int] = {
    TIER_NONE: 0,
    TIER_1_LINEAR: 1,
    TIER_2_DIURNAL: 2,
    TIER_3_CLIMATOLOGY: 3,
}

CONTINUOUS_IMPUTE_COLUMNS: tuple[str, ...] = (
    "us_aqi",
    "temperature_2m",
    "precipitation_mm",
    "precipitation_prob",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "apparent_temperature",
    "surface_pressure",
    "cloud_cover",
    *POLLUTANT_COLUMNS,
)

ALL_IMPUTED_TARGET_COLUMNS: tuple[str, ...] = (
    *CONTINUOUS_IMPUTE_COLUMNS,
    "wind_direction_10m",
    "weather_code",
)

# Physical Lahore climatology fallbacks used only when an entire series has no valid points.
_PHYSICAL_FALLBACK_DEFAULTS: dict[str, float] = {
    "us_aqi": 110.0,
    "temperature_2m": 24.0,
    "precipitation_mm": 0.0,
    "precipitation_prob": 0.0,
    "relative_humidity_2m": 55.0,
    "uv_index": 0.0,
    "wind_speed_10m": 8.0,
    "wind_direction_10m": 315.0,
    "apparent_temperature": 24.0,
    "surface_pressure": 1012.0,
    "cloud_cover": 20.0,
    "weather_code": 0.0,
    "pm2_5": 38.0,
    "pm10": 75.0,
    "ozone": 45.0,
    "nitrogen_dioxide": 28.0,
    "sulphur_dioxide": 12.0,
    "carbon_monoxide": 450.0,
}

_CLAMP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    **{
        name: (spec.min_value, spec.max_value)
        for name, spec in METRIC_SPECS.items()
        if not spec.is_circular
    },
    "apparent_temperature": (-30.0, 65.0),
    "surface_pressure": (800.0, 1100.0),
    "cloud_cover": (0.0, 100.0),
    **{col: (0.0, None) for col in POLLUTANT_COLUMNS},
}


@dataclass(slots=True)
class ImputationAuditReport:
    """Structured per-run summary of missingness and 3-tier gap imputation."""

    total_rows: int
    total_cells_checked: int
    missing_before: int
    missing_after: int
    tier1_count: int
    tier2_count: int
    tier3_count: int
    rows_imputed: int
    imputation_rate_pct: float
    per_column: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImputationResult:
    """Container holding the imputed DataFrame and its audit report."""

    df: pd.DataFrame
    audit: ImputationAuditReport


def _contiguous_missing_runs(is_missing: np.ndarray) -> list[tuple[int, int]]:
    """Return `[start, end)` index slices for every contiguous True run."""
    runs: list[tuple[int, int]] = []
    n = len(is_missing)
    i = 0
    while i < n:
        if is_missing[i]:
            j = i + 1
            while j < n and is_missing[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _clamp_scalar(column: str, value: float) -> float:
    bounds = _CLAMP_BOUNDS.get(column)
    if not bounds:
        return value
    low, high = bounds
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _diurnal_lookup_value(
    orig_values: np.ndarray,
    timestamps_ns: np.ndarray,
    hours: np.ndarray,
    idx: int,
    lookback_ns: int,
) -> float | None:
    """Compute rolling lookback median for the same UTC hour-of-day (with full-series fallback)."""
    target_hour = hours[idx]
    target_t = timestamps_ns[idx]
    window_start = target_t - lookback_ns

    # Primary: strict causal rolling N-day lookback for the exact hour-of-day
    mask = (
        (timestamps_ns >= window_start)
        & (timestamps_ns < target_t)
        & (hours == target_hour)
        & (~np.isnan(orig_values))
    )
    if np.any(mask):
        return float(np.nanmedian(orig_values[mask]))

    # Secondary: any valid observation at the same hour-of-day
    same_hour_mask = (hours == target_hour) & (~np.isnan(orig_values))
    if np.any(same_hour_mask):
        return float(np.nanmedian(orig_values[same_hour_mask]))

    return None


def _climatological_value(
    orig_values: np.ndarray,
    months: np.ndarray,
    hours: np.ndarray,
    idx: int,
    fallback: float,
) -> float:
    """Return `(month, hour)` median, falling back to month median, global median, or default."""
    m = months[idx]
    h = hours[idx]

    mh_mask = (months == m) & (hours == h) & (~np.isnan(orig_values))
    if np.any(mh_mask):
        return float(np.nanmedian(orig_values[mh_mask]))

    m_mask = (months == m) & (~np.isnan(orig_values))
    if np.any(m_mask):
        return float(np.nanmedian(orig_values[m_mask]))

    valid = orig_values[~np.isnan(orig_values)]
    if len(valid) > 0:
        return float(np.nanmedian(valid))

    return fallback


def _impute_continuous_series(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
    *,
    column_name: str,
    short_gap_hours: int,
    medium_gap_hours: int,
    diurnal_lookback_days: int,
    clamp_bounds: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Apply explicit 3-tier imputation to a continuous numeric series."""
    orig = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=True)
    filled = orig.copy()
    n = len(orig)
    tiers = [TIER_NONE] * n

    if n == 0:
        return filled, tiers

    missing_mask = np.isnan(orig)
    if not np.any(missing_mask):
        return filled, tiers

    ts_ns = timestamps.asi8
    hours = timestamps.hour.to_numpy()
    months = timestamps.month.to_numpy()
    lookback_ns = int(diurnal_lookback_days * 24 * 3600 * 1_000_000_000)
    fallback_default = _PHYSICAL_FALLBACK_DEFAULTS.get(column_name, 0.0)

    runs = _contiguous_missing_runs(missing_mask)
    for start, end in runs:
        run_len = end - start
        has_left = start > 0 and not np.isnan(orig[start - 1])
        has_right = end < n and not np.isnan(orig[end])

        # Tier 1: Short interior gaps (<= short_gap_hours) -> time-aware linear interpolation
        if run_len <= short_gap_hours and has_left and has_right:
            t_left = float(ts_ns[start - 1])
            t_right = float(ts_ns[end])
            v_left = float(orig[start - 1])
            v_right = float(orig[end])
            span = t_right - t_left
            for idx in range(start, end):
                frac = (float(ts_ns[idx]) - t_left) / span if span > 0 else 0.5
                val = v_left + frac * (v_right - v_left)
                filled[idx] = _clamp_scalar(column_name, val) if clamp_bounds else val
                tiers[idx] = TIER_1_LINEAR
            continue

        # Tier 2: Medium gaps (4..24h, or short boundary gaps lacking a right neighbor)
        if run_len <= medium_gap_hours:
            for idx in range(start, end):
                diurnal_val = _diurnal_lookup_value(orig, ts_ns, hours, idx, lookback_ns)
                if diurnal_val is not None:
                    filled[idx] = (
                        _clamp_scalar(column_name, diurnal_val)
                        if clamp_bounds
                        else diurnal_val
                    )
                    tiers[idx] = TIER_2_DIURNAL
                    continue

                # Fall through to Tier 3 if no diurnal match exists yet
                clim_val = _climatological_value(orig, months, hours, idx, fallback_default)
                if has_left:
                    last_val = float(orig[start - 1])
                    last_clim = _climatological_value(
                        orig, months, hours, start - 1, fallback_default
                    )
                    dt_hours = max(1.0, float(ts_ns[idx] - ts_ns[start - 1]) / 3_600_000_000_000.0)
                    blended = clim_val + (last_val - last_clim) * float(np.exp(-dt_hours / 12.0))
                else:
                    blended = clim_val
                filled[idx] = _clamp_scalar(column_name, blended) if clamp_bounds else blended
                tiers[idx] = TIER_3_CLIMATOLOGY
            continue

        # Tier 3: Extended outage (> medium_gap_hours) -> damped forward-fill into monthly-hour climatology
        for idx in range(start, end):
            clim_val = _climatological_value(orig, months, hours, idx, fallback_default)
            if has_left:
                last_val = float(orig[start - 1])
                last_clim = _climatological_value(orig, months, hours, start - 1, fallback_default)
                dt_hours = max(1.0, float(ts_ns[idx] - ts_ns[start - 1]) / 3_600_000_000_000.0)
                blended = clim_val + (last_val - last_clim) * float(np.exp(-dt_hours / 12.0))
            else:
                blended = clim_val
            filled[idx] = _clamp_scalar(column_name, blended) if clamp_bounds else blended
            tiers[idx] = TIER_3_CLIMATOLOGY

    return filled, tiers


def _impute_weather_code_series(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
    *,
    short_gap_hours: int,
    medium_gap_hours: int,
    diurnal_lookback_days: int,
) -> tuple[np.ndarray, list[str]]:
    """Impute discrete WMO weather codes via nearest/mode fallback without fractional codes."""
    orig = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=True)
    filled = orig.copy()
    n = len(orig)
    tiers = [TIER_NONE] * n

    missing_mask = np.isnan(orig)
    if not np.any(missing_mask):
        return np.rint(filled).astype("int64"), tiers

    ts_ns = timestamps.asi8
    hours = timestamps.hour.to_numpy()
    months = timestamps.month.to_numpy()
    lookback_ns = int(diurnal_lookback_days * 24 * 3600 * 1_000_000_000)

    def _mode(arr: np.ndarray, default: float = 0.0) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return default
        vals, counts = np.unique(np.rint(valid), return_counts=True)
        return float(vals[int(np.argmax(counts))])

    runs = _contiguous_missing_runs(missing_mask)
    for start, end in runs:
        run_len = end - start
        has_left = start > 0 and not np.isnan(orig[start - 1])
        has_right = end < n and not np.isnan(orig[end])

        if run_len <= short_gap_hours and has_left and has_right:
            for idx in range(start, end):
                dist_left = idx - (start - 1)
                dist_right = end - idx
                filled[idx] = orig[start - 1] if dist_left <= dist_right else orig[end]
                tiers[idx] = TIER_1_LINEAR
            continue

        if run_len <= medium_gap_hours:
            for idx in range(start, end):
                target_t = ts_ns[idx]
                mask = (
                    (ts_ns >= target_t - lookback_ns)
                    & (ts_ns < target_t)
                    & (hours == hours[idx])
                    & (~np.isnan(orig))
                )
                if np.any(mask):
                    filled[idx] = _mode(orig[mask])
                    tiers[idx] = TIER_2_DIURNAL
                    continue
                same_h = (hours == hours[idx]) & (~np.isnan(orig))
                if np.any(same_h):
                    filled[idx] = _mode(orig[same_h])
                    tiers[idx] = TIER_2_DIURNAL
                    continue
                filled[idx] = orig[start - 1] if has_left else _mode(orig, 0.0)
                tiers[idx] = TIER_3_CLIMATOLOGY
            continue

        for idx in range(start, end):
            mh = (months == months[idx]) & (hours == hours[idx]) & (~np.isnan(orig))
            if np.any(mh):
                filled[idx] = _mode(orig[mh])
            elif has_left:
                filled[idx] = orig[start - 1]
            else:
                filled[idx] = _mode(orig, 0.0)
            tiers[idx] = TIER_3_CLIMATOLOGY

    return np.rint(filled).astype("int64"), tiers


def impute_weather_dataframe(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> ImputationResult:
    """Apply the documented 3-tier imputation strategy and attach per-row audit metadata."""
    if df.empty:
        out = df.copy()
        out["is_imputed"] = pd.Series(dtype="bool")
        out["imputation_tier"] = pd.Series(dtype="object")
        out["imputed_columns"] = pd.Series(dtype="object")
        return ImputationResult(
            df=out,
            audit=ImputationAuditReport(
                total_rows=0,
                total_cells_checked=0,
                missing_before=0,
                missing_after=0,
                tier1_count=0,
                tier2_count=0,
                tier3_count=0,
                rows_imputed=0,
                imputation_rate_pct=0.0,
                per_column={},
            ),
        )

    if "timestamp_utc" not in df.columns:
        raise ValueError("DataFrame must include 'timestamp_utc' for time-aware imputation.")

    cfg = settings or get_settings()
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(out["timestamp_utc"])
    n_rows = len(out)

    short_h = cfg.imputation_short_gap_hours
    medium_h = cfg.imputation_medium_gap_hours
    lookback_d = cfg.imputation_diurnal_lookback_days

    row_highest_tier = [TIER_NONE] * n_rows
    row_imputed_cols: list[list[str]] = [[] for _ in range(n_rows)]
    per_column_audit: dict[str, dict[str, int]] = {}
    tier_totals = {TIER_1_LINEAR: 0, TIER_2_DIURNAL: 0, TIER_3_CLIMATOLOGY: 0}
    missing_before_total = 0
    missing_after_total = 0
    cells_checked_total = 0

    def _record_column_tiers(col_name: str, missing_before: int, missing_after: int, tiers: list[str]) -> None:
        nonlocal missing_before_total, missing_after_total, cells_checked_total
        t1 = sum(1 for t in tiers if t == TIER_1_LINEAR)
        t2 = sum(1 for t in tiers if t == TIER_2_DIURNAL)
        t3 = sum(1 for t in tiers if t == TIER_3_CLIMATOLOGY)
        per_column_audit[col_name] = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "tier1_linear": t1,
            "tier2_diurnal_median": t2,
            "tier3_climatology_ffill": t3,
        }
        missing_before_total += missing_before
        missing_after_total += missing_after
        cells_checked_total += n_rows
        tier_totals[TIER_1_LINEAR] += t1
        tier_totals[TIER_2_DIURNAL] += t2
        tier_totals[TIER_3_CLIMATOLOGY] += t3

        for idx, tier in enumerate(tiers):
            if tier != TIER_NONE:
                row_imputed_cols[idx].append(col_name)
                if _TIER_SEVERITY[tier] > _TIER_SEVERITY[row_highest_tier[idx]]:
                    row_highest_tier[idx] = tier

    # 1. Impute continuous meteorological & pollutant series
    for col in CONTINUOUS_IMPUTE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        raw_series = pd.to_numeric(out[col], errors="coerce")
        missing_before = int(raw_series.isna().sum())
        filled_arr, tiers = _impute_continuous_series(
            raw_series,
            timestamps,
            column_name=col,
            short_gap_hours=short_h,
            medium_gap_hours=medium_h,
            diurnal_lookback_days=lookback_d,
            clamp_bounds=True,
        )
        out[col] = filled_arr
        missing_after = int(pd.isna(out[col]).sum())
        _record_column_tiers(col, missing_before, missing_after, tiers)

    # 2. Impute circular wind_direction_10m via sin/cos components
    if "wind_direction_10m" not in out.columns:
        out["wind_direction_10m"] = np.nan
    orig_wd = pd.to_numeric(out["wind_direction_10m"], errors="coerce")
    wd_missing_mask = orig_wd.isna().to_numpy()
    wd_missing_before = int(wd_missing_mask.sum())

    if wd_missing_before > 0:
        rad = np.deg2rad(orig_wd.to_numpy(dtype="float64"))
        sin_series = pd.Series(np.sin(rad), index=out.index)
        cos_series = pd.Series(np.cos(rad), index=out.index)

        sin_filled, sin_tiers = _impute_continuous_series(
            sin_series,
            timestamps,
            column_name="wind_dir_sin",
            short_gap_hours=short_h,
            medium_gap_hours=medium_h,
            diurnal_lookback_days=lookback_d,
            clamp_bounds=False,
        )
        cos_filled, _ = _impute_continuous_series(
            cos_series,
            timestamps,
            column_name="wind_dir_cos",
            short_gap_hours=short_h,
            medium_gap_hours=medium_h,
            diurnal_lookback_days=lookback_d,
            clamp_bounds=False,
        )
        reconstructed = np.rad2deg(np.arctan2(sin_filled, cos_filled)) % 360.0
        wd_out = orig_wd.to_numpy(dtype="float64", copy=True) % 360.0
        wd_out[wd_missing_mask] = reconstructed[wd_missing_mask]
        out["wind_direction_10m"] = wd_out
        wd_tiers = sin_tiers
    else:
        out["wind_direction_10m"] = orig_wd.to_numpy(dtype="float64") % 360.0
        wd_tiers = [TIER_NONE] * n_rows

    wd_missing_after = int(pd.isna(out["wind_direction_10m"]).sum())
    _record_column_tiers("wind_direction_10m", wd_missing_before, wd_missing_after, wd_tiers)

    # 3. Impute discrete WMO weather_code & backfill condition_symbol
    if "weather_code" not in out.columns:
        out["weather_code"] = np.nan
    wmo_raw = pd.to_numeric(out["weather_code"], errors="coerce")
    wmo_missing_before = int(wmo_raw.isna().sum())
    wmo_filled, wmo_tiers = _impute_weather_code_series(
        wmo_raw,
        timestamps,
        short_gap_hours=short_h,
        medium_gap_hours=medium_h,
        diurnal_lookback_days=lookback_d,
    )
    out["weather_code"] = wmo_filled
    wmo_missing_after = int(pd.isna(out["weather_code"]).sum())
    _record_column_tiers("weather_code", wmo_missing_before, wmo_missing_after, wmo_tiers)

    if "condition_symbol" not in out.columns:
        out["condition_symbol"] = None
    missing_sym = out["condition_symbol"].isna() | (out["condition_symbol"].astype(str).str.strip() == "")
    if missing_sym.any():
        out.loc[missing_sym, "condition_symbol"] = out.loc[missing_sym, "weather_code"].map(
            wmo_code_to_symbol
        )

    # Ensure apparent_temperature reflects Steadman formula if all apparent_temperature inputs were originally absent
    if "apparent_temperature" in per_column_audit and per_column_audit["apparent_temperature"]["missing_before"] == n_rows:
        for idx in range(n_rows):
            steadman = compute_steadman_apparent_temperature(
                float(out.at[idx, "temperature_2m"]),
                float(out.at[idx, "relative_humidity_2m"]),
                float(out.at[idx, "wind_speed_10m"]) / 3.6,
            )
            if steadman is not None:
                out.at[idx, "apparent_temperature"] = steadman

    if "source_disagreement_temp" in out.columns:
        out["source_disagreement_temp"] = pd.to_numeric(
            out["source_disagreement_temp"], errors="coerce"
        ).fillna(0.0)
    else:
        out["source_disagreement_temp"] = 0.0

    if "sources_available" not in out.columns:
        out["sources_available"] = [[] for _ in range(n_rows)]

    # Attach row-level imputation audit columns matching Supabase schema
    out["is_imputed"] = [tier != TIER_NONE for tier in row_highest_tier]
    out["imputation_tier"] = row_highest_tier
    out["imputed_columns"] = [sorted(set(cols)) for cols in row_imputed_cols]

    rows_imputed = int(sum(1 for flag in out["is_imputed"] if flag))
    rate_pct = (
        round((missing_before_total / cells_checked_total) * 100.0, 4)
        if cells_checked_total > 0
        else 0.0
    )

    audit = ImputationAuditReport(
        total_rows=n_rows,
        total_cells_checked=cells_checked_total,
        missing_before=missing_before_total,
        missing_after=missing_after_total,
        tier1_count=tier_totals[TIER_1_LINEAR],
        tier2_count=tier_totals[TIER_2_DIURNAL],
        tier3_count=tier_totals[TIER_3_CLIMATOLOGY],
        rows_imputed=rows_imputed,
        imputation_rate_pct=rate_pct,
        per_column=per_column_audit,
    )
    return ImputationResult(df=out, audit=audit)
