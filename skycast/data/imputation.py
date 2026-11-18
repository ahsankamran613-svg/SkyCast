"""Documented 3-tier missing-data imputation with circular wind handling and audit logging."""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from skycast.config.settings import (
    POLLUTANT_COLUMNS,
    Settings,
    get_settings,
)
from skycast.data.fusion import DOMAIN_CLAMP_BOUNDS
from skycast.data.sources.met_no import (
    compute_steadman_apparent_temperature,
    wmo_code_to_symbol,
)

logger = logging.getLogger(__name__)

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

_DEFAULT_WIND_DIR_RAD = float(np.deg2rad(315.0))

# Physical Lahore climatology fallbacks used only when no prior valid observations exist.
_PHYSICAL_FALLBACK_DEFAULTS: dict[str, float] = {
    "us_aqi": 110.0,
    "temperature_2m": 24.0,
    "precipitation_mm": 0.0,
    "precipitation_prob": 0.0,
    "relative_humidity_2m": 55.0,
    "uv_index": 0.0,
    "wind_speed_10m": 8.0,
    "wind_direction_10m": 315.0,
    "wind_dir_sin": float(np.sin(_DEFAULT_WIND_DIR_RAD)),
    "wind_dir_cos": float(np.cos(_DEFAULT_WIND_DIR_RAD)),
    "apparent_temperature": 24.0,
    "surface_pressure": 987.5,  # Standard station pressure at Lahore elevation (~217 m)
    "cloud_cover": 20.0,
    "weather_code": 0.0,
    "pm2_5": 38.0,
    "pm10": 75.0,
    "ozone": 45.0,
    "nitrogen_dioxide": 28.0,
    "sulphur_dioxide": 12.0,
    "carbon_monoxide": 450.0,
}

_CLAMP_BOUNDS: dict[str, tuple[float | None, float | None]] = DOMAIN_CLAMP_BOUNDS


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
    per_column: dict[str, dict[str, int | float]] = field(default_factory=dict)
    constant_fallback_count: int = 0
    constant_fallback_columns: list[str] = field(default_factory=list)
    high_imputation_columns: list[str] = field(default_factory=list)
    max_column_imputation_pct: float = 0.0

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


class _CausalLookupIndex:
    """Pre-indexed O(log n) lookup tables for diurnal medians/modes and climatology."""

    def __init__(
        self,
        orig_values: np.ndarray,
        ts_ns: np.ndarray,
        hours: np.ndarray,
        months: np.ndarray,
        *,
        causal_only: bool = True,
    ) -> None:
        self.causal_only = causal_only
        valid_mask = ~np.isnan(orig_values)
        self.valid_ts = ts_ns[valid_mask]
        self.valid_vals = orig_values[valid_mask]

        self.ts_by_hour: dict[int, np.ndarray] = {}
        self.vals_by_hour: dict[int, np.ndarray] = {}
        self.ts_by_mh: dict[tuple[int, int], np.ndarray] = {}
        self.vals_by_mh: dict[tuple[int, int], np.ndarray] = {}
        self.ts_by_month: dict[int, np.ndarray] = {}
        self.vals_by_month: dict[int, np.ndarray] = {}

        if len(self.valid_vals) > 0:
            valid_hours = hours[valid_mask]
            valid_months = months[valid_mask]
            h_buckets: dict[int, list[int]] = defaultdict(list)
            m_buckets: dict[int, list[int]] = defaultdict(list)
            mh_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
            for pos, (m_val, h_val) in enumerate(zip(valid_months.tolist(), valid_hours.tolist())):
                h_int = int(h_val)
                m_int = int(m_val)
                h_buckets[h_int].append(pos)
                m_buckets[m_int].append(pos)
                mh_buckets[(m_int, h_int)].append(pos)

            for h_int, idxs in h_buckets.items():
                idx_arr = np.asarray(idxs, dtype=np.intp)
                self.ts_by_hour[h_int] = self.valid_ts[idx_arr]
                self.vals_by_hour[h_int] = self.valid_vals[idx_arr]
            for m_int, idxs in m_buckets.items():
                idx_arr = np.asarray(idxs, dtype=np.intp)
                self.ts_by_month[m_int] = self.valid_ts[idx_arr]
                self.vals_by_month[m_int] = self.valid_vals[idx_arr]
            for key, idxs in mh_buckets.items():
                idx_arr = np.asarray(idxs, dtype=np.intp)
                self.ts_by_mh[key] = self.valid_ts[idx_arr]
                self.vals_by_mh[key] = self.valid_vals[idx_arr]

        self._hour_prefix_cache: dict[tuple[int, int, str], float] = {}
        self._mh_cache: dict[tuple[int, int, int, str], float] = {}
        self._month_cache: dict[tuple[int, int, str], float] = {}
        self._global_cache: dict[tuple[int, str], float] = {}

    @staticmethod
    def _mode(arr: np.ndarray) -> float:
        vals, counts = np.unique(np.rint(arr), return_counts=True)
        return float(vals[int(np.argmax(counts))])

    def _aggregate(self, arr: np.ndarray, agg: str) -> float:
        if agg == "mode":
            return self._mode(arr)
        return float(np.median(arr))

    def _upper_bound(self, ts_arr: np.ndarray, target_t: int) -> int:
        if not self.causal_only:
            return len(ts_arr)
        return bisect_left(ts_arr, target_t)

    def diurnal_lookup(
        self,
        target_t: int,
        hour: int,
        lookback_ns: int,
        *,
        agg: str = "median",
    ) -> float | None:
        ts_h = self.ts_by_hour.get(hour)
        if ts_h is None or len(ts_h) == 0:
            return None

        hi = self._upper_bound(ts_h, target_t)
        if hi == 0:
            return None
        lo = bisect_left(ts_h, target_t - lookback_ns, 0, hi)
        vals_h = self.vals_by_hour[hour]
        if lo < hi:
            return self._aggregate(vals_h[lo:hi], agg)

        # Secondary strictly-causal fallback: any prior observation at the same hour-of-day.
        cache_key = (hour, hi, agg)
        cached = self._hour_prefix_cache.get(cache_key)
        if cached is None:
            cached = self._aggregate(vals_h[:hi], agg)
            self._hour_prefix_cache[cache_key] = cached
        return cached

    def climatological_lookup(
        self,
        target_t: int,
        month: int,
        hour: int,
        fallback: float,
        *,
        agg: str = "median",
    ) -> tuple[float, bool]:
        """Return `(value, used_constant_fallback)` strictly respecting causality."""
        mh_key = (month, hour)
        ts_mh = self.ts_by_mh.get(mh_key)
        if ts_mh is not None:
            hi_mh = self._upper_bound(ts_mh, target_t)
            if hi_mh > 0:
                c_key = (month, hour, hi_mh, agg)
                cached = self._mh_cache.get(c_key)
                if cached is None:
                    cached = self._aggregate(self.vals_by_mh[mh_key][:hi_mh], agg)
                    self._mh_cache[c_key] = cached
                return cached, False

        ts_m = self.ts_by_month.get(month)
        if ts_m is not None:
            hi_m = self._upper_bound(ts_m, target_t)
            if hi_m > 0:
                c_key_m = (month, hi_m, agg)
                cached_m = self._month_cache.get(c_key_m)
                if cached_m is None:
                    cached_m = self._aggregate(self.vals_by_month[month][:hi_m], agg)
                    self._month_cache[c_key_m] = cached_m
                return cached_m, False

        if len(self.valid_ts) > 0:
            hi_all = self._upper_bound(self.valid_ts, target_t)
            if hi_all > 0:
                c_key_g = (hi_all, agg)
                cached_g = self._global_cache.get(c_key_g)
                if cached_g is None:
                    cached_g = self._aggregate(self.valid_vals[:hi_all], agg)
                    self._global_cache[c_key_g] = cached_g
                return cached_g, False

        return fallback, True


def _diurnal_lookup_value(
    orig_values: np.ndarray,
    timestamps_ns: np.ndarray,
    hours: np.ndarray,
    idx: int,
    lookback_ns: int,
    *,
    causal_only: bool = True,
) -> float | None:
    """Compute causal rolling lookback median for the same UTC hour-of-day."""
    target_hour = hours[idx]
    target_t = timestamps_ns[idx]
    window_start = target_t - lookback_ns

    mask = (
        (timestamps_ns >= window_start)
        & (timestamps_ns < target_t)
        & (hours == target_hour)
        & (~np.isnan(orig_values))
    )
    if np.any(mask):
        return float(np.nanmedian(orig_values[mask]))

    same_hour_mask = (hours == target_hour) & (~np.isnan(orig_values))
    if causal_only:
        same_hour_mask &= timestamps_ns < target_t
    if np.any(same_hour_mask):
        return float(np.nanmedian(orig_values[same_hour_mask]))

    return None


def _climatological_value(
    orig_values: np.ndarray,
    months: np.ndarray,
    hours: np.ndarray,
    idx: int,
    fallback: float,
    *,
    timestamps_ns: np.ndarray | None = None,
    causal_only: bool = True,
) -> float:
    """Return `(month, hour)` median, falling back to month median, global median, or default."""
    m = months[idx]
    h = hours[idx]
    causal_mask = np.ones(len(orig_values), dtype=bool)
    if causal_only and timestamps_ns is not None:
        causal_mask = timestamps_ns < timestamps_ns[idx]

    mh_mask = (months == m) & (hours == h) & (~np.isnan(orig_values)) & causal_mask
    if np.any(mh_mask):
        return float(np.nanmedian(orig_values[mh_mask]))

    m_mask = (months == m) & (~np.isnan(orig_values)) & causal_mask
    if np.any(m_mask):
        return float(np.nanmedian(orig_values[m_mask]))

    valid = orig_values[(~np.isnan(orig_values)) & causal_mask]
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
    damping_tau_hours: float = 12.0,
    clamp_bounds: bool = True,
    causal_only: bool = True,
) -> tuple[np.ndarray, list[str], int]:
    """Apply explicit 3-tier imputation to a continuous numeric series."""
    orig = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=True)
    filled = orig.copy()
    n = len(orig)
    tiers = [TIER_NONE] * n
    constant_fallback_count = 0

    if n == 0:
        return filled, tiers, constant_fallback_count

    missing_mask = np.isnan(orig)
    if not np.any(missing_mask):
        return filled, tiers, constant_fallback_count

    ts_ns = timestamps.asi8
    hours = timestamps.hour.to_numpy()
    months = timestamps.month.to_numpy()
    lookback_ns = int(diurnal_lookback_days * 24 * 3600 * 1_000_000_000)
    fallback_default = _PHYSICAL_FALLBACK_DEFAULTS.get(column_name, 0.0)
    lookup = _CausalLookupIndex(orig, ts_ns, hours, months, causal_only=causal_only)

    runs = _contiguous_missing_runs(missing_mask)
    for start, end in runs:
        run_len = end - start
        has_left = start > 0 and not np.isnan(orig[start - 1])
        has_right = end < n and not np.isnan(orig[end])
        time_gap_hours = (
            float(ts_ns[end] - ts_ns[start - 1]) / 3_600_000_000_000.0 - 1.0
            if (has_left and has_right)
            else float(run_len)
        )

        # Tier 1: Short interior gaps (<= short_gap_hours in real elapsed time) -> linear interp
        if run_len <= short_gap_hours and time_gap_hours <= float(short_gap_hours) and has_left and has_right:
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

        last_val = 0.0
        last_clim = 0.0
        t_last = 0
        if has_left:
            last_val = float(orig[start - 1])
            t_last = int(ts_ns[start - 1])
            last_clim, _ = lookup.climatological_lookup(
                t_last,
                int(months[start - 1]),
                int(hours[start - 1]),
                fallback_default,
            )

        # Tier 2: Medium gaps (4..24h, or short boundary gaps lacking a right neighbor)
        if run_len <= medium_gap_hours and time_gap_hours <= float(medium_gap_hours):
            for idx in range(start, end):
                t_cur = int(ts_ns[idx])
                h_cur = int(hours[idx])
                m_cur = int(months[idx])
                diurnal_val = lookup.diurnal_lookup(t_cur, h_cur, lookback_ns)
                if diurnal_val is not None:
                    filled[idx] = (
                        _clamp_scalar(column_name, diurnal_val)
                        if clamp_bounds
                        else diurnal_val
                    )
                    tiers[idx] = TIER_2_DIURNAL
                    continue

                clim_val, used_fallback = lookup.climatological_lookup(
                    t_cur, m_cur, h_cur, fallback_default
                )
                if has_left:
                    dt_hours = max(1.0, float(t_cur - t_last) / 3_600_000_000_000.0)
                    blended = clim_val + (last_val - last_clim) * float(
                        np.exp(-dt_hours / damping_tau_hours)
                    )
                else:
                    blended = clim_val
                    if used_fallback:
                        constant_fallback_count += 1
                filled[idx] = _clamp_scalar(column_name, blended) if clamp_bounds else blended
                tiers[idx] = TIER_3_CLIMATOLOGY
            continue

        # Tier 3: Extended outage (> medium_gap_hours) -> damped ffill into monthly-hour climatology
        for idx in range(start, end):
            t_cur = int(ts_ns[idx])
            clim_val, used_fallback = lookup.climatological_lookup(
                t_cur, int(months[idx]), int(hours[idx]), fallback_default
            )
            if has_left:
                dt_hours = max(1.0, float(t_cur - t_last) / 3_600_000_000_000.0)
                blended = clim_val + (last_val - last_clim) * float(
                    np.exp(-dt_hours / damping_tau_hours)
                )
            else:
                blended = clim_val
                if used_fallback:
                    constant_fallback_count += 1
            filled[idx] = _clamp_scalar(column_name, blended) if clamp_bounds else blended
            tiers[idx] = TIER_3_CLIMATOLOGY

    return filled, tiers, constant_fallback_count


def _impute_weather_code_series(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
    *,
    short_gap_hours: int,
    medium_gap_hours: int,
    diurnal_lookback_days: int,
    causal_only: bool = True,
) -> tuple[np.ndarray, list[str], int]:
    """Impute discrete WMO weather codes via nearest/causal mode without fractional codes."""
    orig = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=True)
    filled = orig.copy()
    n = len(orig)
    tiers = [TIER_NONE] * n
    constant_fallback_count = 0

    missing_mask = np.isnan(orig)
    if not np.any(missing_mask):
        return filled, tiers, constant_fallback_count

    ts_ns = timestamps.asi8
    hours = timestamps.hour.to_numpy()
    months = timestamps.month.to_numpy()
    lookback_ns = int(diurnal_lookback_days * 24 * 3600 * 1_000_000_000)
    fallback_code = _PHYSICAL_FALLBACK_DEFAULTS["weather_code"]
    lookup = _CausalLookupIndex(orig, ts_ns, hours, months, causal_only=causal_only)

    runs = _contiguous_missing_runs(missing_mask)
    for start, end in runs:
        run_len = end - start
        has_left = start > 0 and not np.isnan(orig[start - 1])
        has_right = end < n and not np.isnan(orig[end])
        time_gap_hours = (
            float(ts_ns[end] - ts_ns[start - 1]) / 3_600_000_000_000.0 - 1.0
            if (has_left and has_right)
            else float(run_len)
        )

        if run_len <= short_gap_hours and time_gap_hours <= float(short_gap_hours) and has_left and has_right:
            t_left = ts_ns[start - 1]
            t_right = ts_ns[end]
            for idx in range(start, end):
                dist_left = ts_ns[idx] - t_left
                dist_right = t_right - ts_ns[idx]
                filled[idx] = orig[start - 1] if dist_left <= dist_right else orig[end]
                tiers[idx] = TIER_1_LINEAR
            continue

        if run_len <= medium_gap_hours and time_gap_hours <= float(medium_gap_hours):
            for idx in range(start, end):
                t_cur = int(ts_ns[idx])
                mode_val = lookup.diurnal_lookup(t_cur, int(hours[idx]), lookback_ns, agg="mode")
                if mode_val is not None:
                    filled[idx] = mode_val
                    tiers[idx] = TIER_2_DIURNAL
                    continue

                if has_left:
                    filled[idx] = orig[start - 1]
                else:
                    clim_mode, used_fallback = lookup.climatological_lookup(
                        t_cur, int(months[idx]), int(hours[idx]), fallback_code, agg="mode"
                    )
                    filled[idx] = clim_mode
                    if used_fallback:
                        constant_fallback_count += 1
                tiers[idx] = TIER_3_CLIMATOLOGY
            continue

        for idx in range(start, end):
            t_cur = int(ts_ns[idx])
            clim_mode, used_fallback = lookup.climatological_lookup(
                t_cur, int(months[idx]), int(hours[idx]), fallback_code, agg="mode"
            )
            if not used_fallback:
                filled[idx] = clim_mode
            elif has_left:
                filled[idx] = orig[start - 1]
            else:
                filled[idx] = fallback_code
                constant_fallback_count += 1
            tiers[idx] = TIER_3_CLIMATOLOGY

    return np.rint(filled), tiers, constant_fallback_count


def impute_weather_dataframe(
    df: pd.DataFrame,
    settings: Settings | None = None,
    *,
    causal_only: bool = True,
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
    tau_h = cfg.imputation_damping_tau_hours

    row_highest_tier = [TIER_NONE] * n_rows
    row_imputed_cols: list[list[str]] = [[] for _ in range(n_rows)]
    per_column_audit: dict[str, dict[str, int | float]] = {}
    tier_totals = {TIER_1_LINEAR: 0, TIER_2_DIURNAL: 0, TIER_3_CLIMATOLOGY: 0}
    missing_before_total = 0
    missing_after_total = 0
    cells_checked_total = 0
    constant_fallback_total = 0
    constant_fallback_columns: list[str] = []
    high_imputation_columns: list[str] = []
    max_col_pct = 0.0

    def _record_column_tiers(
        col_name: str,
        missing_before: int,
        missing_after: int,
        tiers: list[str],
        constant_fallback_count: int,
    ) -> None:
        nonlocal missing_before_total, missing_after_total, cells_checked_total
        nonlocal constant_fallback_total, max_col_pct
        t1 = sum(1 for t in tiers if t == TIER_1_LINEAR)
        t2 = sum(1 for t in tiers if t == TIER_2_DIURNAL)
        t3 = sum(1 for t in tiers if t == TIER_3_CLIMATOLOGY)
        col_pct = round((missing_before / n_rows) * 100.0, 4) if n_rows > 0 else 0.0
        max_col_pct = max(max_col_pct, col_pct)
        if col_pct > cfg.max_imputation_rate_pct:
            high_imputation_columns.append(col_name)
        if constant_fallback_count > 0:
            constant_fallback_columns.append(col_name)
            constant_fallback_total += constant_fallback_count

        per_column_audit[col_name] = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "tier1_linear": t1,
            "tier2_diurnal_median": t2,
            "tier3_climatology_ffill": t3,
            "constant_fallback": constant_fallback_count,
            "imputation_rate_pct": col_pct,
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
        filled_arr, tiers, fb_count = _impute_continuous_series(
            raw_series,
            timestamps,
            column_name=col,
            short_gap_hours=short_h,
            medium_gap_hours=medium_h,
            diurnal_lookback_days=lookback_d,
            damping_tau_hours=tau_h,
            clamp_bounds=True,
            causal_only=causal_only,
        )
        out[col] = filled_arr
        missing_after = int(pd.isna(out[col]).sum())
        _record_column_tiers(col, missing_before, missing_after, tiers, fb_count)

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

        sin_filled, sin_tiers, sin_fb = _impute_continuous_series(
            sin_series,
            timestamps,
            column_name="wind_dir_sin",
            short_gap_hours=short_h,
            medium_gap_hours=medium_h,
            diurnal_lookback_days=lookback_d,
            damping_tau_hours=tau_h,
            clamp_bounds=False,
            causal_only=causal_only,
        )
        cos_filled, _, cos_fb = _impute_continuous_series(
            cos_series,
            timestamps,
            column_name="wind_dir_cos",
            short_gap_hours=short_h,
            medium_gap_hours=medium_h,
            diurnal_lookback_days=lookback_d,
            damping_tau_hours=tau_h,
            clamp_bounds=False,
            causal_only=causal_only,
        )
        reconstructed = np.rad2deg(np.arctan2(sin_filled, cos_filled)) % 360.0
        degenerate = (np.abs(sin_filled) < 1e-12) & (np.abs(cos_filled) < 1e-12)
        if np.any(degenerate):
            reconstructed[degenerate] = _PHYSICAL_FALLBACK_DEFAULTS["wind_direction_10m"]
        wd_out = orig_wd.to_numpy(dtype="float64", copy=True) % 360.0
        wd_out[wd_missing_mask] = reconstructed[wd_missing_mask]
        out["wind_direction_10m"] = wd_out
        wd_tiers = sin_tiers
        wd_fb = max(sin_fb, cos_fb)
    else:
        out["wind_direction_10m"] = orig_wd.to_numpy(dtype="float64") % 360.0
        wd_tiers = [TIER_NONE] * n_rows
        wd_fb = 0

    wd_missing_after = int(pd.isna(out["wind_direction_10m"]).sum())
    _record_column_tiers("wind_direction_10m", wd_missing_before, wd_missing_after, wd_tiers, wd_fb)

    # 3. Impute discrete WMO weather_code & backfill condition_symbol
    if "weather_code" not in out.columns:
        out["weather_code"] = np.nan
    wmo_raw = pd.to_numeric(out["weather_code"], errors="coerce")
    wmo_missing_before = int(wmo_raw.isna().sum())
    wmo_filled_float, wmo_tiers, wmo_fb = _impute_weather_code_series(
        wmo_raw,
        timestamps,
        short_gap_hours=short_h,
        medium_gap_hours=medium_h,
        diurnal_lookback_days=lookback_d,
        causal_only=causal_only,
    )
    wmo_missing_after = int(np.isnan(wmo_filled_float).sum())
    if wmo_missing_after == 0:
        out["weather_code"] = np.rint(wmo_filled_float).astype("int64")
    else:
        out["weather_code"] = pd.Series(wmo_filled_float, index=out.index).round().astype("Int64")
    _record_column_tiers("weather_code", wmo_missing_before, wmo_missing_after, wmo_tiers, wmo_fb)

    if "condition_symbol" not in out.columns:
        out["condition_symbol"] = None
    missing_sym = out["condition_symbol"].isna() | (
        out["condition_symbol"].astype(str).str.strip() == ""
    )
    if missing_sym.any():
        out.loc[missing_sym, "condition_symbol"] = out.loc[missing_sym, "weather_code"].map(
            lambda code: wmo_code_to_symbol(code) if pd.notna(code) else None
        )

    # Recompute Steadman apparent_temperature if all apparent_temperature inputs were absent
    if (
        "apparent_temperature" in per_column_audit
        and per_column_audit["apparent_temperature"]["missing_before"] == n_rows
    ):
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
        )
    else:
        out["source_disagreement_temp"] = np.nan

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

    if constant_fallback_columns:
        logger.warning(
            "Imputation used physical constant fallbacks for columns %s (%d cells).",
            constant_fallback_columns,
            constant_fallback_total,
        )
    if high_imputation_columns:
        logger.warning(
            "Imputation rate exceeded %.1f%% for columns: %s (max=%.1f%%).",
            cfg.max_imputation_rate_pct,
            high_imputation_columns,
            max_col_pct,
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
        constant_fallback_count=constant_fallback_total,
        constant_fallback_columns=constant_fallback_columns,
        high_imputation_columns=high_imputation_columns,
        max_column_imputation_pct=max_col_pct,
    )
    return ImputationResult(df=out, audit=audit)
