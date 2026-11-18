"""Zero-key external weather and air-quality data source clients."""

from skycast.data.sources.met_no import (
    MetNoError,
    compute_steadman_apparent_temperature,
    fetch_met_no_forecast,
    metno_symbol_to_wmo_code,
    normalize_symbol_code,
    parse_met_no_timeseries,
    sea_level_to_surface_pressure,
    wmo_code_to_symbol,
)
from skycast.data.sources.open_meteo import (
    OPEN_METEO_AIR_QUALITY_FIELDS,
    OPEN_METEO_ARCHIVE_WEATHER_FIELDS,
    OPEN_METEO_WEATHER_FIELDS,
    OpenMeteoError,
    fetch_open_meteo_air_quality,
    fetch_open_meteo_forecast,
    fetch_open_meteo_historical,
    fetch_open_meteo_weather,
    parse_open_meteo_hourly,
)

__all__ = [
    "MetNoError",
    "OPEN_METEO_AIR_QUALITY_FIELDS",
    "OPEN_METEO_ARCHIVE_WEATHER_FIELDS",
    "OPEN_METEO_WEATHER_FIELDS",
    "OpenMeteoError",
    "compute_steadman_apparent_temperature",
    "fetch_met_no_forecast",
    "fetch_open_meteo_air_quality",
    "fetch_open_meteo_forecast",
    "fetch_open_meteo_historical",
    "fetch_open_meteo_weather",
    "metno_symbol_to_wmo_code",
    "normalize_symbol_code",
    "parse_met_no_timeseries",
    "parse_open_meteo_hourly",
    "sea_level_to_surface_pressure",
    "wmo_code_to_symbol",
]
