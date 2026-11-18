"""Met.no Locationforecast 2.0 API client with Steadman apparent temperature fallback."""

from __future__ import annotations

import math
from typing import Any

import httpx
import numpy as np
import pandas as pd

from skycast.config.settings import Settings, get_settings
from skycast.data.sources._http import request_json_with_retry

MET_NO_OUTPUT_COLUMNS: tuple[str, ...] = (
    "air_temperature",
    "apparent_air_temperature",
    "relative_humidity",
    "precipitation_amount",
    "wind_speed",
    "wind_from_direction",
    "ultraviolet_index_clear_sky",
    "air_pressure_at_sea_level",
    "cloud_area_fraction",
    "symbol_code",
    "weather_code",
)

_SYMBOL_TO_WMO: dict[str, int] = {
    "clearsky": 0,
    "fair": 1,
    "partlycloudy": 2,
    "cloudy": 3,
    "fog": 45,
    "lightrain": 61,
    "rain": 63,
    "heavyrain": 65,
    "lightrainshowers": 80,
    "rainshowers": 81,
    "heavyrainshowers": 82,
    "lightsleet": 68,
    "sleet": 68,
    "heavysleet": 69,
    "lightsleetshowers": 68,
    "sleetshowers": 68,
    "heavysleetshowers": 69,
    "lightsnow": 71,
    "snow": 73,
    "heavysnow": 75,
    "lightsnowshowers": 85,
    "snowshowers": 85,
    "heavysnowshowers": 86,
    "lightrainandthunder": 95,
    "rainandthunder": 95,
    "heavyrainandthunder": 99,
    "lightrainshowersandthunder": 95,
    "rainshowersandthunder": 95,
    "heavyrainshowersandthunder": 99,
    "lightsleetandthunder": 95,
    "sleetandthunder": 95,
    "heavysleetandthunder": 99,
    "lightsnowandthunder": 95,
    "snowandthunder": 95,
    "heavysnowandthunder": 99,
}

_WMO_TO_SYMBOL: dict[int, str] = {
    0: "clearsky",
    1: "fair",
    2: "partlycloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "lightrain",
    53: "lightrain",
    55: "rain",
    56: "sleet",
    57: "sleet",
    61: "lightrain",
    63: "rain",
    65: "heavyrain",
    66: "sleet",
    67: "heavysleet",
    68: "sleet",
    69: "heavysleet",
    71: "lightsnow",
    73: "snow",
    75: "heavysnow",
    77: "snow",
    80: "lightrainshowers",
    81: "rainshowers",
    82: "heavyrainshowers",
    85: "lightsnowshowers",
    86: "heavysnowshowers",
    95: "rainandthunder",
    96: "heavyrainandthunder",
    99: "heavyrainandthunder",
}


class MetNoError(RuntimeError):
    """Raised when a Met.no API request or payload parsing fails."""


def compute_steadman_apparent_temperature(
    temp_c: float | None,
    relative_humidity_pct: float | None,
    wind_speed_ms: float | None,
) -> float | None:
    """Compute Steadman (1994) shade apparent temperature in °C.

    Formula:
        e = (RH / 100) * 6.105 * exp((17.27 * T) / (237.7 + T))
        AT = T + 0.33 * e - 0.70 * wind_speed_ms - 4.00
    """
    if temp_c is None or relative_humidity_pct is None:
        return None
    try:
        t = float(temp_c)
        rh = float(relative_humidity_pct)
        v = 0.0 if wind_speed_ms is None else float(wind_speed_ms)
    except (TypeError, ValueError):
        return None

    if math.isnan(t) or math.isnan(rh):
        return None
    if math.isnan(v):
        v = 0.0

    rh_clamped = max(0.0, min(100.0, rh))
    v_clamped = max(0.0, v)
    vapor_pressure = (rh_clamped / 100.0) * 6.105 * math.exp((17.27 * t) / (237.7 + t))
    return t + 0.33 * vapor_pressure - 0.70 * v_clamped - 4.00


def sea_level_to_surface_pressure(
    pressure_msl_hpa: float | None,
    elevation_m: float,
    temp_c: float | None = None,
) -> float | None:
    """Convert sea-level pressure (hPa) to station surface pressure at `elevation_m`."""
    if pressure_msl_hpa is None:
        return None
    try:
        p_msl = float(pressure_msl_hpa)
        elev = float(elevation_m)
    except (TypeError, ValueError):
        return None
    if math.isnan(p_msl) or math.isnan(elev):
        return None
    if abs(elev) < 1e-6:
        return p_msl

    temp_k = 288.15
    if temp_c is not None:
        try:
            t_val = float(temp_c)
            if not math.isnan(t_val):
                temp_k = max(180.0, min(340.0, t_val + 273.15))
        except (TypeError, ValueError):
            pass

    lapse = 0.0065
    ratio = 1.0 - (lapse * elev) / (temp_k + lapse * elev)
    if ratio <= 0.0:
        return None
    return p_msl * (ratio**5.25588)


def _normalize_symbol_code(symbol_code: str | None) -> str | None:
    if not symbol_code:
        return None
    normalized = str(symbol_code).strip().lower()
    for suffix in ("_day", "_night", "_polartwilight"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized or None


def normalize_symbol_code(symbol_code: str | None) -> str | None:
    """Normalize a Met.no `symbol_code` by removing day/night/polar-twilight suffixes."""
    return _normalize_symbol_code(symbol_code)


def metno_symbol_to_wmo_code(symbol_code: str | None) -> int | None:
    """Map a Met.no `symbol_code` (with optional `_day`/`_night` suffix) to a WMO code."""
    base = _normalize_symbol_code(symbol_code)
    if base is None:
        return None
    if base in _SYMBOL_TO_WMO:
        return _SYMBOL_TO_WMO[base]
    if "thunder" in base:
        return 95
    if "snow" in base:
        return 73
    if "sleet" in base:
        return 68
    if "rain" in base:
        return 63
    if "fog" in base:
        return 45
    if "cloud" in base:
        return 3
    if "clear" in base:
        return 0
    return None


def wmo_code_to_symbol(weather_code: int | float | None) -> str:
    """Map a WMO weather code to a human-readable Met.no-compatible condition symbol."""
    if weather_code is None:
        return "unknown"
    try:
        if pd.isna(weather_code):
            return "unknown"
        code = int(round(float(weather_code)))
    except (TypeError, ValueError):
        return "unknown"

    if code < 0:
        return "unknown"
    if code in _WMO_TO_SYMBOL:
        return _WMO_TO_SYMBOL[code]
    if code >= 95:
        return "rainandthunder"
    if 80 <= code <= 82:
        return "rainshowers"
    if 71 <= code <= 77 or code in (85, 86):
        return "snow"
    if 61 <= code <= 69:
        return "rain"
    if 51 <= code <= 57:
        return "lightrain"
    if code in (45, 48):
        return "fog"
    if code <= 1:
        return "clearsky"
    if code == 2:
        return "partlycloudy"
    return "cloudy"


def _wind_speed_multiplier(wind_unit: str | None) -> float:
    unit = (wind_unit or "m/s").strip().lower()
    if unit in {"m/s", "mps"}:
        return 3.6
    if unit in {"km/h", "kmh"}:
        return 1.0
    if unit in {"knots", "kn", "kt"}:
        return 1.852
    return 3.6


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_met_no_timeseries(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse a Met.no Locationforecast 2.0 GeoJSON payload into a UTC DataFrame."""
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        raise MetNoError("Met.no payload missing 'properties' object.")

    timeseries = properties.get("timeseries")
    if not isinstance(timeseries, list):
        raise MetNoError("Met.no 'properties' object missing 'timeseries' array.")

    if not timeseries:
        empty: dict[str, Any] = {"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]")}
        for col in MET_NO_OUTPUT_COLUMNS:
            empty[col] = pd.Series(dtype="object" if col == "symbol_code" else "float64")
        return pd.DataFrame(empty)

    units = properties.get("meta", {}).get("units", {})
    wind_mult = _wind_speed_multiplier(units.get("wind_speed"))

    rows: list[dict[str, Any]] = []
    for entry in timeseries:
        if not isinstance(entry, dict):
            continue
        raw_time = entry.get("time")
        ts = pd.to_datetime(raw_time, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        ts = ts.floor("h")

        data = entry.get("data") or {}
        instant_details = (data.get("instant") or {}).get("details") or {}
        next_1h = data.get("next_1_hours") or {}
        next_6h = data.get("next_6_hours") or {}
        next_12h = data.get("next_12_hours") or {}

        temp_c = _to_float(instant_details.get("air_temperature"))
        rh_pct = _to_float(instant_details.get("relative_humidity"))
        wind_raw = _to_float(instant_details.get("wind_speed"))
        wind_kmh = wind_raw * wind_mult if not math.isnan(wind_raw) else float("nan")
        wind_ms = wind_kmh / 3.6 if not math.isnan(wind_kmh) else float("nan")

        apparent = _to_float(instant_details.get("apparent_air_temperature"))
        if math.isnan(apparent):
            steadman = compute_steadman_apparent_temperature(
                None if math.isnan(temp_c) else temp_c,
                None if math.isnan(rh_pct) else rh_pct,
                None if math.isnan(wind_ms) else wind_ms,
            )
            apparent = float("nan") if steadman is None else steadman

        wind_dir = _to_float(instant_details.get("wind_from_direction"))
        if not math.isnan(wind_dir):
            wind_dir = wind_dir % 360.0

        precip_1h = _to_float((next_1h.get("details") or {}).get("precipitation_amount"))
        symbol_code = (
            (next_1h.get("summary") or {}).get("symbol_code")
            or (next_6h.get("summary") or {}).get("symbol_code")
            or (next_12h.get("summary") or {}).get("symbol_code")
        )
        wmo = metno_symbol_to_wmo_code(symbol_code)

        rows.append(
            {
                "timestamp_utc": ts,
                "air_temperature": temp_c,
                "apparent_air_temperature": apparent,
                "relative_humidity": rh_pct,
                "precipitation_amount": precip_1h,
                "wind_speed": wind_kmh,
                "wind_from_direction": wind_dir,
                "ultraviolet_index_clear_sky": _to_float(
                    instant_details.get("ultraviolet_index_clear_sky")
                ),
                "air_pressure_at_sea_level": _to_float(
                    instant_details.get("air_pressure_at_sea_level")
                ),
                "cloud_area_fraction": _to_float(instant_details.get("cloud_area_fraction")),
                "symbol_code": str(symbol_code) if symbol_code else None,
                "weather_code": float(wmo) if wmo is not None else float("nan"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", *MET_NO_OUTPUT_COLUMNS])

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in MET_NO_OUTPUT_COLUMNS if c != "symbol_code"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["wind_speed"] = np.where(df["wind_speed"].notna(), df["wind_speed"].clip(lower=0.0), np.nan)
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="first")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def fetch_met_no_forecast(
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch Lahore hourly forecast from Met.no Locationforecast 2.0."""
    cfg = settings or get_settings()
    retries = cfg.http_max_retries if max_retries is None else max_retries
    backoff = cfg.http_backoff_seconds if backoff_seconds is None else backoff_seconds
    headers = {
        "User-Agent": cfg.met_no_user_agent,
        "Accept": "application/json",
    }
    params = {
        "lat": round(cfg.lahore_lat, 4),
        "lon": round(cfg.lahore_lon, 4),
    }

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)
    try:
        payload = request_json_with_retry(
            http,
            cfg.met_no_forecast_url,
            params=params,
            headers=headers,
            error_cls=MetNoError,
            source_name="Met.no",
            max_retries=retries,
            backoff_seconds=backoff,
        )
        return parse_met_no_timeseries(payload)
    finally:
        if own_client:
            http.close()
