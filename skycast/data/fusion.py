"""Multi-source UTC time-alignment and weighted sensor fusion for Open-Meteo and Met.no."""

from __future__ import annotations

from datetime import datetime
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
    metno_symbol_to_wmo_code,
    normalize_symbol_code,
    sea_level_to_surface_pressure,
    wmo_code_to_symbol,
)

DOMAIN_CLAMP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
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

RAW_TRANSPARENCY_COLUMNS: tuple[str, ...] = (
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
)

CANONICAL_FUSED_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    *TARGET_METRICS,
    "apparent_temperature",
    "weather_code",
    "condition_symbol",
    "surface_pressure",
    "cloud_cover",
    *POLLUTANT_COLUMNS,
    *RAW_TRANSPARENCY_COLUMNS,
)


class SourceFusionError(ValueError):
    """Raised when multi-source fusion cannot construct a valid timeline."""


def _prepare_source_frame(
    df: pd.DataFrame | None,
    *,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if "timestamp_utc" not in df.columns:
        raise SourceFusionError("Source DataFrame must contain a 'timestamp_utc' column.")

    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce").dt.floor("h")
    out = out.dropna(subset=["timestamp_utc"])
    if start_ts is not None:
        out = out.loc[out["timestamp_utc"] >= start_ts]
    if end_ts is not None:
        out = out.loc[out["timestamp_utc"] <= end_ts]
    if out.empty:
        return None
    out = out.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")
    return out.reset_index(drop=True)


def _to_utc_timestamp(value: datetime | pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.floor("h")


def _extract_numeric_series(
    frame: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    column: str,
    *,
    shift_hours: int = 0,
) -> pd.Series:
    if frame is None or column not in frame.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    if shift_hours == 0:
        indexed = frame.set_index("timestamp_utc")[column]
        return pd.to_numeric(indexed.reindex(index), errors="coerce").astype("float64")

    ts = frame["timestamp_utc"] + pd.Timedelta(hours=shift_hours)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
    indexed = pd.Series(values, index=ts)
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    return indexed.reindex(index).astype("float64")


def _extract_object_series(
    frame: pd.DataFrame | None,
    index: pd.DatetimeIndex,
    column: str,
) -> pd.Series:
    if frame is None or column not in frame.columns:
        return pd.Series(None, index=index, dtype="object")
    indexed = frame.set_index("timestamp_utc")[column]
    return indexed.reindex(index).astype("object")


def _weighted_linear_fuse(
    om: pd.Series,
    met: pd.Series,
    w_om: float,
    w_met: float,
) -> pd.Series:
    return weighted_linear_fuse(om, met, w_om, w_met)


def weighted_linear_fuse(
    om: pd.Series,
    met: pd.Series,
    w_om: float,
    w_met: float,
) -> pd.Series:
    """Fuse two aligned float series via weighted average with single-source fallback."""
    both = om.notna() & met.notna()
    fused = pd.Series(np.nan, index=om.index, dtype="float64")
    fused.loc[both] = w_om * om.loc[both] + w_met * met.loc[both]
    only_om = om.notna() & ~met.notna()
    fused.loc[only_om] = om.loc[only_om]
    only_met = ~om.notna() & met.notna()
    fused.loc[only_met] = met.loc[only_met]
    return fused


def _weighted_circular_fuse(
    om_deg: pd.Series,
    met_deg: pd.Series,
    w_om: float,
    w_met: float,
) -> pd.Series:
    return weighted_circular_fuse(om_deg, met_deg, w_om, w_met)


def weighted_circular_fuse(
    om_deg: pd.Series,
    met_deg: pd.Series,
    w_om: float,
    w_met: float,
) -> pd.Series:
    """Fuse directional angles (0-360°) via weighted vector averaging on the unit circle."""
    both = om_deg.notna() & met_deg.notna()
    fused = pd.Series(np.nan, index=om_deg.index, dtype="float64")

    if both.any():
        om_rad = np.deg2rad(om_deg.loc[both])
        met_rad = np.deg2rad(met_deg.loc[both])
        x = w_om * np.cos(om_rad) + w_met * np.cos(met_rad)
        y = w_om * np.sin(om_rad) + w_met * np.sin(met_rad)
        angles = np.rad2deg(np.arctan2(y, x)) % 360.0
        # Handle exact 180-deg equal-weight cancellation by prioritizing Open-Meteo
        degenerate = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12)
        if degenerate.any():
            angles = angles.where(~degenerate, om_deg.loc[both] % 360.0)
        fused.loc[both] = angles

    only_om = om_deg.notna() & ~met_deg.notna()
    if only_om.any():
        fused.loc[only_om] = om_deg.loc[only_om] % 360.0

    only_met = ~om_deg.notna() & met_deg.notna()
    if only_met.any():
        fused.loc[only_met] = met_deg.loc[only_met] % 360.0

    return fused


def _clamp_series(series: pd.Series, min_value: float | None, max_value: float | None, *, is_circular: bool = False) -> pd.Series:
    out = series.copy()
    valid = out.notna()
    if not valid.any():
        return out
    if is_circular:
        out.loc[valid] = out.loc[valid] % 360.0
        return out
    if min_value is not None or max_value is not None:
        out.loc[valid] = out.loc[valid].clip(lower=min_value, upper=max_value)
    return out


def fuse_weather_sources(
    open_meteo_df: pd.DataFrame | None,
    met_no_df: pd.DataFrame | None,
    settings: Settings | None = None,
    *,
    start_utc: datetime | pd.Timestamp | str | None = None,
    end_utc: datetime | pd.Timestamp | str | None = None,
    align_grid: bool = True,
) -> pd.DataFrame:
    """Time-align and fuse Open-Meteo and Met.no into a continuous UTC hourly DataFrame."""
    cfg = settings or get_settings()
    start_ts = _to_utc_timestamp(start_utc)
    end_ts = _to_utc_timestamp(end_utc)

    om = _prepare_source_frame(open_meteo_df, start_ts=start_ts, end_ts=end_ts)
    met_full = _prepare_source_frame(met_no_df, start_ts=None, end_ts=end_ts)
    met = _prepare_source_frame(met_no_df, start_ts=start_ts, end_ts=end_ts)

    if om is None and met is None:
        raise SourceFusionError(
            "Both Open-Meteo and Met.no datasets are empty within the requested window."
        )

    all_ts_parts: list[pd.Series] = []
    if om is not None:
        all_ts_parts.append(om["timestamp_utc"])
    if met is not None:
        all_ts_parts.append(met["timestamp_utc"])
    combined_ts = pd.concat(all_ts_parts, ignore_index=True)

    grid_start = start_ts if start_ts is not None else combined_ts.min()
    grid_end = end_ts if end_ts is not None else combined_ts.max()
    if grid_start > grid_end:
        raise SourceFusionError(
            f"Fusion window start ({grid_start}) is after end ({grid_end})."
        )

    if align_grid:
        hourly_index = pd.date_range(start=grid_start, end=grid_end, freq="1h", tz="UTC")
    else:
        hourly_index = pd.DatetimeIndex(sorted(combined_ts.unique()), tz="UTC")

    w_om = cfg.open_meteo_fusion_weight
    w_met = cfg.met_no_fusion_weight

    # Raw Open-Meteo aligned series
    temp_om = _extract_numeric_series(om, hourly_index, "temperature_2m")
    app_om = _extract_numeric_series(om, hourly_index, "apparent_temperature")
    wmo_om = _extract_numeric_series(om, hourly_index, "weather_code")
    rh_om = _extract_numeric_series(om, hourly_index, "relative_humidity_2m")
    precip_om = _extract_numeric_series(om, hourly_index, "precipitation")
    precip_prob_om = _extract_numeric_series(om, hourly_index, "precipitation_probability")
    uv_om = _extract_numeric_series(om, hourly_index, "uv_index")
    wind_spd_om = _extract_numeric_series(om, hourly_index, "wind_speed_10m")
    wind_dir_om = _extract_numeric_series(om, hourly_index, "wind_direction_10m")
    pressure_om = _extract_numeric_series(om, hourly_index, "surface_pressure")
    cloud_om = _extract_numeric_series(om, hourly_index, "cloud_cover")

    # Raw Met.no aligned series
    temp_met = _extract_numeric_series(met, hourly_index, "air_temperature")
    app_met = _extract_numeric_series(met, hourly_index, "apparent_air_temperature")
    wmo_met = _extract_numeric_series(met, hourly_index, "weather_code")
    rh_met = _extract_numeric_series(met, hourly_index, "relative_humidity")
    # Met.no `next_1_hours.precipitation_amount` at t covers [t, t+1h]; shift +1h to match
    # Open-Meteo's ending-hour [t-1h, t] convention.
    precip_met = _extract_numeric_series(
        met_full, hourly_index, "precipitation_amount", shift_hours=1
    )
    uv_met = _extract_numeric_series(met, hourly_index, "ultraviolet_index_clear_sky")
    wind_spd_met = _extract_numeric_series(met, hourly_index, "wind_speed")
    wind_dir_met = _extract_numeric_series(met, hourly_index, "wind_from_direction")
    pressure_met_msl = _extract_numeric_series(met, hourly_index, "air_pressure_at_sea_level")
    cloud_met = _extract_numeric_series(met, hourly_index, "cloud_area_fraction")
    symbol_met = _extract_object_series(met, hourly_index, "symbol_code")

    # Convert Met.no sea-level pressure to Lahore station surface pressure before fusing.
    ref_temp_for_pressure = temp_met.combine_first(temp_om)
    pressure_met_surface = pd.Series(np.nan, index=hourly_index, dtype="float64")
    has_p_met = pressure_met_msl.notna()
    if has_p_met.any():
        pressure_met_surface.loc[has_p_met] = [
            sea_level_to_surface_pressure(
                float(p_val),
                cfg.lahore_elevation_m,
                None if pd.isna(t_val) else float(t_val),
            )
            for p_val, t_val in zip(
                pressure_met_msl.loc[has_p_met].tolist(),
                ref_temp_for_pressure.loc[has_p_met].tolist(),
            )
        ]

    # Canonical weighted fusion across physically commensurate variables
    temp_fused = _weighted_linear_fuse(temp_om, temp_met, w_om, w_met)
    rh_fused = _weighted_linear_fuse(rh_om, rh_met, w_om, w_met)
    precip_fused = _weighted_linear_fuse(precip_om, precip_met, w_om, w_met)
    wind_spd_fused = _weighted_linear_fuse(wind_spd_om, wind_spd_met, w_om, w_met)
    wind_dir_fused = _weighted_circular_fuse(wind_dir_om, wind_dir_met, w_om, w_met)
    pressure_fused = _weighted_linear_fuse(pressure_om, pressure_met_surface, w_om, w_met)
    cloud_fused = _weighted_linear_fuse(cloud_om, cloud_met, w_om, w_met)
    apparent_fused = _weighted_linear_fuse(app_om, app_met, w_om, w_met)

    # Open-Meteo uv_index is cloud-attenuated; Met.no supplies clear-sky UV only. Use Open-Meteo
    # directly when available and cloud-attenuate Met.no clear-sky UV only during fallback.
    uv_met_fallback = pd.Series(np.nan, index=hourly_index, dtype="float64")
    has_uv_met = uv_met.notna()
    if has_uv_met.any():
        cloud_for_uv = cloud_fused.fillna(0.0).clip(lower=0.0, upper=100.0)
        cmf = 1.0 - 0.75 * ((cloud_for_uv / 100.0) ** 3)
        uv_met_fallback.loc[has_uv_met] = uv_met.loc[has_uv_met] * cmf.loc[has_uv_met]
    uv_fused = uv_om.combine_first(uv_met_fallback)

    # Steadman apparent temperature fallback where both apparent fields are absent
    missing_app = apparent_fused.isna() & temp_fused.notna() & rh_fused.notna()
    if missing_app.any():
        for ts in hourly_index[missing_app]:
            spd_kmh = wind_spd_fused.loc[ts]
            spd_ms = 0.0 if pd.isna(spd_kmh) else float(spd_kmh) / 3.6
            steadman = compute_steadman_apparent_temperature(
                float(temp_fused.loc[ts]),
                float(rh_fused.loc[ts]),
                spd_ms,
            )
            if steadman is not None:
                apparent_fused.loc[ts] = steadman

    wmo_from_symbol = symbol_met.map(
        lambda s: float(c) if (c := metno_symbol_to_wmo_code(s)) is not None else np.nan
    )
    wmo_met_resolved = wmo_met.combine_first(pd.to_numeric(wmo_from_symbol, errors="coerce"))
    weather_code_fused = wmo_om.combine_first(wmo_met_resolved)

    # Keep condition_symbol consistent with weather_code_fused source priority (Open-Meteo first).
    condition_symbol = pd.Series(None, index=hourly_index, dtype="object")
    has_om_wmo = wmo_om.notna()
    if has_om_wmo.any():
        condition_symbol.loc[has_om_wmo] = wmo_om.loc[has_om_wmo].map(wmo_code_to_symbol)
    fallback_mask = (~has_om_wmo) & weather_code_fused.notna()
    if fallback_mask.any():
        norm_met_sym = symbol_met.loc[fallback_mask].map(normalize_symbol_code)
        derived_sym = weather_code_fused.loc[fallback_mask].map(wmo_code_to_symbol)
        condition_symbol.loc[fallback_mask] = norm_met_sym.where(
            norm_met_sym.notna() & (norm_met_sym != ""),
            derived_sym,
        )

    # Open-Meteo only pollutants & AQI & rain probability
    us_aqi = _extract_numeric_series(om, hourly_index, "us_aqi")
    precip_prob = precip_prob_om

    app_low, app_high = DOMAIN_CLAMP_BOUNDS["apparent_temperature"]
    pres_low, pres_high = DOMAIN_CLAMP_BOUNDS["surface_pressure"]
    cloud_low, cloud_high = DOMAIN_CLAMP_BOUNDS["cloud_cover"]

    data: dict[str, Any] = {
        "timestamp_utc": hourly_index,
        "us_aqi": us_aqi,
        "temperature_2m": temp_fused,
        "precipitation_mm": precip_fused,
        "precipitation_prob": precip_prob,
        "relative_humidity_2m": rh_fused,
        "uv_index": uv_fused,
        "wind_speed_10m": wind_spd_fused,
        "wind_direction_10m": wind_dir_fused,
        "apparent_temperature": _clamp_series(apparent_fused, app_low, app_high),
        "weather_code": weather_code_fused,
        "condition_symbol": condition_symbol,
        "surface_pressure": _clamp_series(pressure_fused, pres_low, pres_high),
        "cloud_cover": _clamp_series(cloud_fused, cloud_low, cloud_high),
    }

    for pollutant in POLLUTANT_COLUMNS:
        series = _extract_numeric_series(om, hourly_index, pollutant)
        data[pollutant] = _clamp_series(series, 0.0, None)

    # Clamp all 8 target metrics to physical domain bounds
    for metric_name, spec in METRIC_SPECS.items():
        data[metric_name] = _clamp_series(
            data[metric_name],
            spec.min_value,
            spec.max_value,
            is_circular=spec.is_circular,
        )

    # Raw source transparency & multi-source disagreement features
    data["temp_openmeteo"] = temp_om
    data["temp_metno"] = temp_met
    data["humidity_openmeteo"] = rh_om
    data["humidity_metno"] = rh_met
    data["precip_openmeteo"] = precip_om
    data["precip_metno"] = precip_met
    data["wind_speed_openmeteo"] = wind_spd_om
    data["wind_speed_metno"] = wind_spd_met
    data["wind_dir_openmeteo"] = _clamp_series(wind_dir_om, 0.0, 360.0, is_circular=True)
    data["wind_dir_metno"] = _clamp_series(wind_dir_met, 0.0, 360.0, is_circular=True)
    data["uv_openmeteo"] = uv_om
    data["uv_metno"] = uv_met
    # Leave missing disagreement as NaN when either source lacks temperature.
    data["source_disagreement_temp"] = (temp_om - temp_met).abs()

    om_present = (
        temp_om.notna()
        | rh_om.notna()
        | precip_om.notna()
        | wind_spd_om.notna()
        | us_aqi.notna()
    )
    met_present = (
        temp_met.notna()
        | rh_met.notna()
        | precip_met.notna()
        | wind_spd_met.notna()
    )

    sources_list: list[list[str]] = []
    for has_om, has_met in zip(om_present.tolist(), met_present.tolist()):
        row_sources: list[str] = []
        if has_om:
            row_sources.append("open_meteo")
        if has_met:
            row_sources.append("met_no")
        sources_list.append(row_sources)
    data["sources_available"] = sources_list

    result = pd.DataFrame(data)
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result = result.sort_values("timestamp_utc").reset_index(drop=True)
    return result
