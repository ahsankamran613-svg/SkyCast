"""Open-Meteo Weather, Historical Archive, and Air Quality API client."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import httpx
import pandas as pd

from skycast.config.settings import Settings, get_settings
from skycast.data.sources._http import request_json_with_retry

logger = logging.getLogger(__name__)

OPEN_METEO_WEATHER_FIELDS: tuple[str, ...] = (
    "temperature_2m",
    "apparent_temperature",
    "weather_code",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
)

# Open-Meteo Archive (ERA5) does not provide precipitation_probability or uv_index.
OPEN_METEO_ARCHIVE_WEATHER_FIELDS: tuple[str, ...] = (
    "temperature_2m",
    "apparent_temperature",
    "weather_code",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
)

OPEN_METEO_HISTORICAL_FORECAST_SUPPLEMENT_FIELDS: tuple[str, ...] = (
    "temperature_2m",
    "apparent_temperature",
    "weather_code",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
)

OPEN_METEO_AIR_QUALITY_FIELDS: tuple[str, ...] = (
    "us_aqi",
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
)

ALL_OPEN_METEO_COLUMNS: tuple[str, ...] = (
    *OPEN_METEO_WEATHER_FIELDS,
    *OPEN_METEO_AIR_QUALITY_FIELDS,
)


class OpenMeteoError(RuntimeError):
    """Raised when an Open-Meteo API request or payload parsing fails."""


def _validate_open_meteo_payload(payload: dict[str, Any], url: str) -> None:
    if payload.get("error"):
        reason = payload.get("reason", "unknown API error")
        raise OpenMeteoError(f"Open-Meteo returned error from {url}: {reason}")


def _request_json_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    *,
    max_retries: int = 2,
    backoff_seconds: float = 0.4,
) -> dict[str, Any]:
    """Execute an Open-Meteo GET request with bounded retries on transient errors."""
    return request_json_with_retry(
        client,
        url,
        params=params,
        error_cls=OpenMeteoError,
        source_name="Open-Meteo",
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        validate_payload=_validate_open_meteo_payload,
    )


def parse_open_meteo_hourly(
    payload: dict[str, Any],
    *,
    expected_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Parse an Open-Meteo `hourly` JSON payload into a UTC-aligned DataFrame."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise OpenMeteoError("Open-Meteo payload missing 'hourly' object.")

    raw_times = hourly.get("time")
    if not isinstance(raw_times, list):
        raise OpenMeteoError("Open-Meteo 'hourly' object missing 'time' array.")

    columns = list(expected_columns) if expected_columns is not None else [
        key for key in hourly.keys() if key != "time"
    ]
    if not raw_times:
        empty_data: dict[str, Any] = {"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]")}
        for col in columns:
            empty_data[col] = pd.Series(dtype="float64")
        return pd.DataFrame(empty_data)

    timestamps = pd.to_datetime(raw_times, utc=True, errors="coerce").floor("h")
    n_rows = len(raw_times)
    frame_data: dict[str, Any] = {"timestamp_utc": timestamps}

    for col in columns:
        values = hourly.get(col)
        if not isinstance(values, list):
            frame_data[col] = pd.Series([float("nan")] * n_rows, dtype="float64")
            continue
        if len(values) < n_rows:
            values = [*values, *([None] * (n_rows - len(values)))]
        elif len(values) > n_rows:
            values = values[:n_rows]
        frame_data[col] = pd.to_numeric(
            pd.Series(values, dtype="object"), errors="coerce"
        ).astype("float64")

    df = pd.DataFrame(frame_data)
    df = df.dropna(subset=["timestamp_utc"])
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def _merge_open_meteo_frames(
    weather_df: pd.DataFrame | None,
    air_quality_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge weather and air-quality hourly DataFrames on `timestamp_utc`."""
    if weather_df is not None and air_quality_df is not None:
        overlap_cols = [
            col for col in air_quality_df.columns
            if col != "timestamp_utc" and col in weather_df.columns
        ]
        merged = pd.merge(
            weather_df,
            air_quality_df,
            on="timestamp_utc",
            how="outer",
            suffixes=("", "__aq"),
        )
        for col in overlap_cols:
            aq_col = f"{col}__aq"
            merged[col] = merged[col].combine_first(merged[aq_col])
            merged = merged.drop(columns=[aq_col])
    elif weather_df is not None:
        merged = weather_df.copy()
    elif air_quality_df is not None:
        merged = air_quality_df.copy()
    else:
        return pd.DataFrame(columns=["timestamp_utc", *ALL_OPEN_METEO_COLUMNS])

    for col in ALL_OPEN_METEO_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")

    ordered_cols = ["timestamp_utc", *ALL_OPEN_METEO_COLUMNS]
    extra_cols = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[[*ordered_cols, *extra_cols]]
    merged = merged.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return merged.reset_index(drop=True)


def fetch_open_meteo_weather(
    settings: Settings | None = None,
    *,
    past_days: int = 2,
    forecast_days: int = 2,
    client: httpx.Client | None = None,
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch hourly Lahore weather observations + short-term forecast from Open-Meteo."""
    cfg = settings or get_settings()
    retries = cfg.http_max_retries if max_retries is None else max_retries
    backoff = cfg.http_backoff_seconds if backoff_seconds is None else backoff_seconds
    params = {
        "latitude": cfg.lahore_lat,
        "longitude": cfg.lahore_lon,
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "hourly": ",".join(OPEN_METEO_WEATHER_FIELDS),
    }
    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)
    try:
        payload = _request_json_with_retry(
            http,
            cfg.open_meteo_forecast_url,
            params,
            max_retries=retries,
            backoff_seconds=backoff,
        )
        return parse_open_meteo_hourly(payload, expected_columns=OPEN_METEO_WEATHER_FIELDS)
    finally:
        if own_client:
            http.close()


def fetch_open_meteo_air_quality(
    settings: Settings | None = None,
    *,
    past_days: int | None = 2,
    forecast_days: int | None = 2,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    include_uv_fallback: bool = False,
    client: httpx.Client | None = None,
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch hourly Lahore pollutants and US AQI from Open-Meteo Air Quality API."""
    cfg = settings or get_settings()
    retries = cfg.http_max_retries if max_retries is None else max_retries
    backoff = cfg.http_backoff_seconds if backoff_seconds is None else backoff_seconds
    fields = list(OPEN_METEO_AIR_QUALITY_FIELDS)
    if include_uv_fallback and "uv_index" not in fields:
        fields.append("uv_index")

    params: dict[str, Any] = {
        "latitude": cfg.lahore_lat,
        "longitude": cfg.lahore_lon,
        "timezone": "UTC",
        "hourly": ",".join(fields),
    }
    if start_date is not None and end_date is not None:
        params["start_date"] = _coerce_date(start_date).isoformat()
        params["end_date"] = _coerce_date(end_date).isoformat()
    else:
        params["past_days"] = past_days if past_days is not None else 2
        params["forecast_days"] = forecast_days if forecast_days is not None else 2

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)
    try:
        payload = _request_json_with_retry(
            http,
            cfg.open_meteo_air_quality_url,
            params,
            max_retries=retries,
            backoff_seconds=backoff,
        )
        return parse_open_meteo_hourly(payload, expected_columns=fields)
    finally:
        if own_client:
            http.close()


def fetch_open_meteo_forecast(
    settings: Settings | None = None,
    *,
    past_days: int = 2,
    forecast_days: int = 2,
    client: httpx.Client | None = None,
    allow_partial: bool = True,
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch and merge Open-Meteo hourly weather and air-quality series for Lahore."""
    cfg = settings or get_settings()
    retries = cfg.http_max_retries if max_retries is None else max_retries
    backoff = cfg.http_backoff_seconds if backoff_seconds is None else backoff_seconds
    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)
    weather_df: pd.DataFrame | None = None
    aq_df: pd.DataFrame | None = None
    errors: list[str] = []

    try:
        try:
            weather_df = fetch_open_meteo_weather(
                cfg,
                past_days=past_days,
                forecast_days=forecast_days,
                client=http,
                max_retries=retries,
                backoff_seconds=backoff,
            )
        except OpenMeteoError as exc:
            errors.append(str(exc))
            if not allow_partial:
                raise
            logger.warning("Open-Meteo weather fetch failed (continuing partial): %s", exc)

        try:
            aq_df = fetch_open_meteo_air_quality(
                cfg,
                past_days=past_days,
                forecast_days=forecast_days,
                client=http,
                max_retries=retries,
                backoff_seconds=backoff,
            )
        except OpenMeteoError as exc:
            errors.append(str(exc))
            if not allow_partial:
                raise
            logger.warning("Open-Meteo air quality fetch failed (continuing partial): %s", exc)

        if weather_df is None and aq_df is None:
            raise OpenMeteoError("; ".join(errors) or "Open-Meteo weather and air quality both failed.")

        return _merge_open_meteo_frames(weather_df, aq_df)
    finally:
        if own_client:
            http.close()


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def fetch_open_meteo_historical(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
    chunk_days: int | None = None,
    chunk_delay_seconds: float | None = None,
    allow_partial: bool = True,
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
) -> pd.DataFrame:
    """Fetch chunked historical weather (Archive + Historical Forecast) and AQI for Lahore."""
    cfg = settings or get_settings()
    start_d = _coerce_date(start_date)
    end_d = _coerce_date(end_date)
    if start_d > end_d:
        raise ValueError(f"start_date ({start_d}) must be <= end_date ({end_d}).")
    resolved_chunk_days = cfg.open_meteo_chunk_days if chunk_days is None else chunk_days
    if resolved_chunk_days < 1:
        raise ValueError("chunk_days must be >= 1.")
    retries = cfg.http_max_retries if max_retries is None else max_retries
    backoff = cfg.http_backoff_seconds if backoff_seconds is None else backoff_seconds
    pace_delay = (
        cfg.open_meteo_chunk_delay_seconds
        if chunk_delay_seconds is None
        else chunk_delay_seconds
    )

    own_client = client is None
    http = client or httpx.Client(timeout=cfg.http_timeout_seconds)
    chunks: list[pd.DataFrame] = []

    try:
        cursor = start_d
        while cursor <= end_d:
            chunk_end = min(cursor + timedelta(days=resolved_chunk_days - 1), end_d)
            weather_df = _fetch_historical_weather_chunk(
                cfg,
                http,
                cursor,
                chunk_end,
                allow_partial=allow_partial,
                max_retries=retries,
                backoff_seconds=backoff,
            )

            aq_df: pd.DataFrame | None = None
            try:
                aq_df = fetch_open_meteo_air_quality(
                    cfg,
                    start_date=cursor,
                    end_date=chunk_end,
                    include_uv_fallback=True,
                    client=http,
                    max_retries=retries,
                    backoff_seconds=backoff,
                )
            except OpenMeteoError as exc:
                if not allow_partial or weather_df is None:
                    raise
                logger.warning(
                    "Open-Meteo historical AQI failed for %s..%s: %s",
                    cursor,
                    chunk_end,
                    exc,
                )

            if weather_df is None and aq_df is None:
                raise OpenMeteoError(
                    f"Both historical weather and AQI failed for {cursor}..{chunk_end}."
                )

            merged = _merge_open_meteo_frames(weather_df, aq_df)
            chunks.append(merged)
            cursor = chunk_end + timedelta(days=1)
            if cursor <= end_d and pace_delay > 0:
                time.sleep(pace_delay)

        if not chunks:
            return pd.DataFrame(columns=["timestamp_utc", *ALL_OPEN_METEO_COLUMNS])

        combined = pd.concat(chunks, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp_utc"], keep="last")
        combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

        start_ts = pd.Timestamp(start_d, tz="UTC")
        end_ts = pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(hours=23)
        mask = (combined["timestamp_utc"] >= start_ts) & (combined["timestamp_utc"] <= end_ts)
        return combined.loc[mask].reset_index(drop=True)
    finally:
        if own_client:
            http.close()


def _fetch_historical_weather_chunk(
    cfg: Settings,
    http: httpx.Client,
    start_d: date,
    end_d: date,
    *,
    allow_partial: bool,
    max_retries: int,
    backoff_seconds: float,
) -> pd.DataFrame | None:
    archive_df: pd.DataFrame | None = None
    hist_fc_df: pd.DataFrame | None = None
    errors: list[str] = []

    try:
        archive_payload = _request_json_with_retry(
            http,
            cfg.open_meteo_archive_url,
            {
                "latitude": cfg.lahore_lat,
                "longitude": cfg.lahore_lon,
                "timezone": "UTC",
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "hourly": ",".join(OPEN_METEO_ARCHIVE_WEATHER_FIELDS),
            },
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        archive_df = parse_open_meteo_hourly(
            archive_payload, expected_columns=OPEN_METEO_ARCHIVE_WEATHER_FIELDS
        )
    except OpenMeteoError as exc:
        errors.append(str(exc))
        logger.warning("Open-Meteo Archive API failed for %s..%s: %s", start_d, end_d, exc)

    # Supplement precipitation_probability, uv_index, and any recent ERA5 archive lag.
    try:
        hist_fc_payload = _request_json_with_retry(
            http,
            cfg.open_meteo_historical_forecast_url,
            {
                "latitude": cfg.lahore_lat,
                "longitude": cfg.lahore_lon,
                "timezone": "UTC",
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "hourly": ",".join(OPEN_METEO_HISTORICAL_FORECAST_SUPPLEMENT_FIELDS),
            },
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        hist_fc_df = parse_open_meteo_hourly(
            hist_fc_payload, expected_columns=OPEN_METEO_HISTORICAL_FORECAST_SUPPLEMENT_FIELDS
        )
    except OpenMeteoError as exc:
        errors.append(str(exc))
        logger.warning(
            "Open-Meteo historical-forecast supplement unavailable for %s..%s: %s",
            start_d,
            end_d,
            exc,
        )

    if archive_df is None and hist_fc_df is None:
        if allow_partial:
            return None
        raise OpenMeteoError("; ".join(errors))

    if archive_df is not None and hist_fc_df is not None:
        merged = pd.merge(
            archive_df,
            hist_fc_df,
            on="timestamp_utc",
            how="outer",
            suffixes=("", "__histfc"),
        )
        for col in OPEN_METEO_HISTORICAL_FORECAST_SUPPLEMENT_FIELDS:
            fc_col = f"{col}__histfc"
            if col in merged.columns and fc_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[fc_col])
                merged = merged.drop(columns=[fc_col])
            elif fc_col in merged.columns:
                merged = merged.rename(columns={fc_col: col})
        return merged.sort_values("timestamp_utc").reset_index(drop=True)

    return archive_df if archive_df is not None else hist_fc_df
