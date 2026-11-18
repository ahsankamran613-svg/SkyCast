"""Central Pydantic configuration, Lahore domain constants, metric specs, and hazard rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------------------------------------------------------------------
# Locked LLM Fallback Chain Constants (implementation_plan.md Component 1 & 4)
# -----------------------------------------------------------------------------
PRIMARY_LLM_PROVIDER = "groq"
PRIMARY_LLM_MODEL = "openai/gpt-oss-120b"
FALLBACK_LLM_PROVIDER = "gemini"
FALLBACK_LLM_MODEL = "gemini-3.5-flash-lite"
TERTIARY_LLM_PROVIDER = "groq"
TERTIARY_LLM_MODEL = "qwen-3.8-27b"
GROQ_REASONING_EFFORT = "low"  # Both Groq API models must use low reasoning.

SeverityLevel = Literal["info", "warning", "critical"]
ComparatorOp = Literal["gt", "ge", "lt", "le", "between_gt_le", "between_ge_lt"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Declarative specification for one of the 8 modeled target time series."""

    name: str
    group: str
    label: str
    unit: str
    chart_token: str
    min_value: float
    max_value: float
    is_circular: bool = False
    open_meteo_source: Literal["weather", "air_quality"] = "weather"
    open_meteo_field: str = ""
    met_no_field: str | None = None

    def clamp(self, value: float) -> float:
        """Clamp a numeric prediction into valid domain bounds (or wrap degrees)."""
        if self.is_circular:
            return value % 360.0
        return max(self.min_value, min(self.max_value, value))


METRIC_SPECS: dict[str, MetricSpec] = {
    "us_aqi": MetricSpec(
        name="us_aqi",
        group="aqi",
        label="Air quality",
        unit="AQI",
        chart_token="--chart-2",
        min_value=0.0,
        max_value=500.0,
        open_meteo_source="air_quality",
        open_meteo_field="us_aqi",
        met_no_field=None,
    ),
    "temperature_2m": MetricSpec(
        name="temperature_2m",
        group="temperature",
        label="Temperature",
        unit="°C",
        chart_token="--chart-1",
        min_value=-20.0,
        max_value=60.0,
        open_meteo_source="weather",
        open_meteo_field="temperature_2m",
        met_no_field="air_temperature",
    ),
    "precipitation_mm": MetricSpec(
        name="precipitation_mm",
        group="precipitation",
        label="Precipitation",
        unit="mm/h",
        chart_token="--chart-3",
        min_value=0.0,
        max_value=250.0,
        open_meteo_source="weather",
        open_meteo_field="precipitation",
        met_no_field="precipitation_amount",
    ),
    "precipitation_prob": MetricSpec(
        name="precipitation_prob",
        group="precipitation",
        label="Chance of rain",
        unit="%",
        chart_token="--chart-3",
        min_value=0.0,
        max_value=100.0,
        open_meteo_source="weather",
        open_meteo_field="precipitation_probability",
        met_no_field=None,
    ),
    "relative_humidity_2m": MetricSpec(
        name="relative_humidity_2m",
        group="humidity",
        label="Humidity",
        unit="%",
        chart_token="--chart-4",
        min_value=0.0,
        max_value=100.0,
        open_meteo_source="weather",
        open_meteo_field="relative_humidity_2m",
        met_no_field="relative_humidity",
    ),
    "uv_index": MetricSpec(
        name="uv_index",
        group="uv",
        label="UV index",
        unit="UV",
        chart_token="--chart-5",
        min_value=0.0,
        max_value=20.0,
        open_meteo_source="weather",
        open_meteo_field="uv_index",
        met_no_field="ultraviolet_index_clear_sky",
    ),
    "wind_speed_10m": MetricSpec(
        name="wind_speed_10m",
        group="wind",
        label="Wind speed",
        unit="km/h",
        chart_token="--chart-6",
        min_value=0.0,
        max_value=250.0,
        open_meteo_source="weather",
        open_meteo_field="wind_speed_10m",
        met_no_field="wind_speed",
    ),
    "wind_direction_10m": MetricSpec(
        name="wind_direction_10m",
        group="wind",
        label="Wind direction",
        unit="°",
        chart_token="--chart-6",
        min_value=0.0,
        max_value=360.0,
        is_circular=True,
        open_meteo_source="weather",
        open_meteo_field="wind_direction_10m",
        met_no_field="wind_from_direction",
    ),
}

TARGET_METRICS: tuple[str, ...] = tuple(METRIC_SPECS.keys())
CIRCULAR_METRICS: tuple[str, ...] = tuple(
    name for name, spec in METRIC_SPECS.items() if spec.is_circular
)
CORE_METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "aqi": ("us_aqi",),
    "temperature": ("temperature_2m",),
    "precipitation": ("precipitation_mm", "precipitation_prob"),
    "humidity": ("relative_humidity_2m",),
    "uv": ("uv_index",),
    "wind": ("wind_speed_10m", "wind_direction_10m"),
}
POLLUTANT_COLUMNS: tuple[str, ...] = (
    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
)
AUX_WEATHER_COLUMNS: tuple[str, ...] = (
    "apparent_temperature",
    "weather_code",
    "surface_pressure",
    "cloud_cover",
)
MODEL_FAMILIES: tuple[str, ...] = (
    "Naive24h",
    "SARIMAX",
    "RandomForest",
    "Ridge",
    "LSTM",
)


@dataclass(frozen=True, slots=True)
class HazardRule:
    """Lahore-calibrated deterministic hazard threshold rule."""

    rule_id: str
    metric: str
    severity: SeverityLevel
    comparator: ComparatorOp
    threshold: float
    title: str
    upper_threshold: float | None = None
    secondary_metric: str | None = None
    secondary_comparator: ComparatorOp | None = None
    secondary_threshold: float | None = None

    @staticmethod
    def _compare(
        value: float,
        comparator: ComparatorOp,
        threshold: float,
        upper: float | None = None,
    ) -> bool:
        if comparator == "gt":
            return value > threshold
        if comparator == "ge":
            return value >= threshold
        if comparator == "lt":
            return value < threshold
        if comparator == "le":
            return value <= threshold
        if comparator == "between_gt_le":
            return upper is not None and threshold < value <= upper
        if comparator == "between_ge_lt":
            return upper is not None and threshold <= value < upper
        raise ValueError(f"Unsupported comparator: {comparator}")

    def matches(self, observations: Mapping[str, float | None]) -> bool:
        """Return True if the metric values in `observations` trigger this hazard rule."""
        primary = observations.get(self.metric)
        if primary is None:
            return False
        if not self._compare(float(primary), self.comparator, self.threshold, self.upper_threshold):
            return False
        if self.secondary_metric is not None and self.secondary_comparator is not None:
            secondary = observations.get(self.secondary_metric)
            if secondary is None or self.secondary_threshold is None:
                return False
            return self._compare(
                float(secondary), self.secondary_comparator, self.secondary_threshold
            )
        return True


# Ordered from highest severity to lowest within each metric group so callers can pick
# the most severe active alert per hazard condition cleanly.
HAZARD_RULES: tuple[HazardRule, ...] = (
    # 1. Air Quality Index (US EPA thresholds)
    HazardRule(
        rule_id="aqi_hazardous",
        metric="us_aqi",
        severity="critical",
        comparator="gt",
        threshold=300.0,
        title="Hazardous air quality",
    ),
    HazardRule(
        rule_id="aqi_very_unhealthy",
        metric="us_aqi",
        severity="critical",
        comparator="between_gt_le",
        threshold=200.0,
        upper_threshold=300.0,
        title="Very unhealthy air quality",
    ),
    HazardRule(
        rule_id="aqi_unhealthy",
        metric="us_aqi",
        severity="warning",
        comparator="between_gt_le",
        threshold=150.0,
        upper_threshold=200.0,
        title="Unhealthy air quality",
    ),
    HazardRule(
        rule_id="aqi_sensitive",
        metric="us_aqi",
        severity="info",
        comparator="between_gt_le",
        threshold=100.0,
        upper_threshold=150.0,
        title="Unhealthy air for sensitive groups",
    ),
    # 2. Temperature (Lahore heatwaves & winter cold snaps)
    HazardRule(
        rule_id="temp_extreme_heat",
        metric="temperature_2m",
        severity="critical",
        comparator="gt",
        threshold=44.0,
        title="Extreme heat emergency",
    ),
    HazardRule(
        rule_id="temp_heatwave",
        metric="temperature_2m",
        severity="warning",
        comparator="between_gt_le",
        threshold=40.0,
        upper_threshold=44.0,
        title="Heatwave warning",
    ),
    HazardRule(
        rule_id="temp_cold_wave",
        metric="temperature_2m",
        severity="critical",
        comparator="lt",
        threshold=4.0,
        title="Cold wave warning",
    ),
    # 3. Precipitation (monsoon flood & heavy rain alerts)
    HazardRule(
        rule_id="precip_urban_flood",
        metric="precipitation_mm",
        severity="critical",
        comparator="gt",
        threshold=15.0,
        title="Urban flood risk",
    ),
    HazardRule(
        rule_id="precip_heavy_rain_mm",
        metric="precipitation_mm",
        severity="warning",
        comparator="between_gt_le",
        threshold=7.5,
        upper_threshold=15.0,
        title="Heavy rain expected",
    ),
    HazardRule(
        rule_id="precip_high_probability",
        metric="precipitation_prob",
        severity="warning",
        comparator="gt",
        threshold=80.0,
        title="Heavy rain likely",
    ),
    # 4. Humidity (heat stress & extreme dryness)
    HazardRule(
        rule_id="humidity_muggy_heat_stress",
        metric="relative_humidity_2m",
        severity="warning",
        comparator="gt",
        threshold=85.0,
        secondary_metric="temperature_2m",
        secondary_comparator="gt",
        secondary_threshold=32.0,
        title="Muggy heat stress warning",
    ),
    HazardRule(
        rule_id="humidity_extreme_dry",
        metric="relative_humidity_2m",
        severity="info",
        comparator="lt",
        threshold=15.0,
        title="Extreme dry air advisory",
    ),
    # 5. UV Index
    HazardRule(
        rule_id="uv_extreme",
        metric="uv_index",
        severity="critical",
        comparator="ge",
        threshold=11.0,
        title="Extreme UV index",
    ),
    HazardRule(
        rule_id="uv_very_high",
        metric="uv_index",
        severity="warning",
        comparator="between_ge_lt",
        threshold=8.0,
        upper_threshold=11.0,
        title="Very high UV index",
    ),
    HazardRule(
        rule_id="uv_high",
        metric="uv_index",
        severity="info",
        comparator="between_ge_lt",
        threshold=6.0,
        upper_threshold=8.0,
        title="High UV index",
    ),
    # 6. Wind Speed
    HazardRule(
        rule_id="wind_severe_squall",
        metric="wind_speed_10m",
        severity="critical",
        comparator="gt",
        threshold=60.0,
        title="Severe squall warning",
    ),
    HazardRule(
        rule_id="wind_strong_gale",
        metric="wind_speed_10m",
        severity="warning",
        comparator="between_gt_le",
        threshold=40.0,
        upper_threshold=60.0,
        title="Strong wind expected",
    ),
)


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Environment & Local Offline Mode ---
    skycast_env: Literal["development", "staging", "production", "test"] = "development"
    skycast_offline_mode: Literal["auto", "true", "false"] = "auto"
    skycast_local_data_dir: Path = Path(".skycast_local")

    # --- Target City (Lahore, Pakistan) ---
    city_name: str = "Lahore"
    lahore_lat: float = Field(default=31.5204, ge=-90.0, le=90.0)
    lahore_lon: float = Field(default=74.3587, ge=-180.0, le=180.0)
    lahore_timezone: str = "Asia/Karachi"

    # --- External Data Sources (No Key Required) ---
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    met_no_forecast_url: str = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    met_no_user_agent: str = "SkyCast-10Pearls-Portfolio/1.0"
    http_timeout_seconds: float = Field(default=20.0, gt=0.0)

    # --- Fusion, Imputation & Feature Engineering ---
    open_meteo_fusion_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    met_no_fusion_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    imputation_short_gap_hours: int = Field(default=3, ge=1)
    imputation_medium_gap_hours: int = Field(default=24, ge=4)
    imputation_diurnal_lookback_days: int = Field(default=7, ge=1)
    forecast_horizon_hours: int = Field(default=24, ge=1, le=72)
    supabase_history_days: int = Field(default=14, ge=1)
    lag_hours: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
    rolling_windows: tuple[int, ...] = (6, 12, 24)
    roc_windows: tuple[int, ...] = (1, 3, 6)
    lahore_smog_months: tuple[int, ...] = (10, 11, 12, 1, 2)
    lahore_monsoon_months: tuple[int, ...] = (7, 8, 9)

    # --- Hopsworks Serverless (Feature Store & Model Registry) ---
    hopsworks_api_key: str = ""
    hopsworks_project_name: str = ""
    hopsworks_host: str = "c.app.hopsworks.ai"
    hopsworks_feature_group_name: str = "lahore_weather_features"
    hopsworks_feature_group_version: int = 1
    hopsworks_forecast_ledger_fg: str = "lahore_forecast_ledger"
    hopsworks_agent_log_fg: str = "lahore_agent_call_log"
    hopsworks_drift_log_fg: str = "lahore_drift_log"
    hopsworks_model_name_template: str = "lahore_{metric}_champion"

    # --- Supabase PostgreSQL Cache ---
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # --- Agentic LLM Fallback Chain ---
    groq_api_key: str = ""
    groq_primary_model: str = Field(
        default=PRIMARY_LLM_MODEL,
        validation_alias=AliasChoices("GROQ_PRIMARY_MODEL", "PRIMARY_LLM_MODEL"),
    )
    gemini_api_key: str = ""
    gemini_fallback_model: str = Field(
        default=FALLBACK_LLM_MODEL,
        validation_alias=AliasChoices("GEMINI_FALLBACK_MODEL", "FALLBACK_LLM_MODEL"),
    )
    groq_tertiary_model: str = Field(
        default=TERTIARY_LLM_MODEL,
        validation_alias=AliasChoices("GROQ_TERTIARY_MODEL", "TERTIARY_LLM_MODEL"),
    )
    groq_reasoning_effort: Literal["low"] = GROQ_REASONING_EFFORT
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0)

    # --- GitHub Actions Drift Retrain Trigger ---
    github_token: str = ""
    github_repo: str = ""
    github_train_workflow: str = "training_pipeline.yml"
    github_ref: str = "main"

    # --- ML Training Split & Model Hyperparameters ---
    train_split_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_split_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_split_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    random_seed: int = 42

    sarimax_order: tuple[int, int, int] = (1, 0, 1)
    sarimax_seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 24)
    sarimax_conf_alpha: float = 0.2

    rf_n_estimators: int = 200
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 4

    ridge_alpha: float = 1.0

    lstm_lookback_hours: int = 24
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 30
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 1e-3
    lstm_early_stopping_patience: int = 5

    # --- Drift Detection Thresholds ---
    data_drift_psi_threshold: float = Field(default=0.25, gt=0.0)
    data_drift_ks_pvalue_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)
    data_drift_window_hours: int = Field(default=48, ge=12)
    model_drift_rmse_ratio_threshold: float = Field(default=1.35, gt=1.0)
    model_drift_window_hours: int = Field(default=72, ge=24)

    # --- Weather Report Schedule (PKT) ---
    daily_report_local_hour: int = Field(default=6, ge=0, le=23)

    # --- API / CORS ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("skycast_offline_mode", mode="before")
    @classmethod
    def _normalize_offline_mode(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return "true"
        if normalized in {"0", "false", "no", "off"}:
            return "false"
        if normalized == "auto":
            return "auto"
        raise ValueError("SKYCAST_OFFLINE_MODE must be one of: auto, true, false")

    @field_validator("groq_reasoning_effort", mode="before")
    @classmethod
    def _enforce_low_reasoning_effort(cls, value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized != "low":
            raise ValueError(
                "GROQ_REASONING_EFFORT must be 'low' for both Groq models "
                "(openai/gpt-oss-120b and qwen-3.8-27b)."
            )
        return normalized

    @model_validator(mode="after")
    def _validate_ratios(self) -> Settings:
        split_sum = self.train_split_ratio + self.val_split_ratio + self.test_split_ratio
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Train/Val/Test split ratios must sum to 1.0 (got {split_sum:.4f})."
            )
        weight_sum = self.open_meteo_fusion_weight + self.met_no_fusion_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Fusion weights must sum to 1.0 (got {weight_sum:.4f})."
            )
        return self

    # --- Uppercase Properties Aligned with Component 1 Spec ---
    @property
    def PRIMARY_LLM_PROVIDER(self) -> str:
        return PRIMARY_LLM_PROVIDER

    @property
    def PRIMARY_LLM_MODEL(self) -> str:
        return self.groq_primary_model

    @property
    def FALLBACK_LLM_PROVIDER(self) -> str:
        return FALLBACK_LLM_PROVIDER

    @property
    def FALLBACK_LLM_MODEL(self) -> str:
        return self.gemini_fallback_model

    @property
    def TERTIARY_LLM_PROVIDER(self) -> str:
        return TERTIARY_LLM_PROVIDER

    @property
    def TERTIARY_LLM_MODEL(self) -> str:
        return self.groq_tertiary_model

    @property
    def GROQ_REASONING_EFFORT(self) -> str:
        return self.groq_reasoning_effort

    # --- Credential & Mode Helpers ---
    @property
    def has_hopsworks_credentials(self) -> bool:
        return bool(self.hopsworks_api_key.strip() and self.hopsworks_project_name.strip())

    @property
    def has_supabase_credentials(self) -> bool:
        return bool(self.supabase_url.strip() and self.supabase_service_role_key.strip())

    @property
    def has_groq_credentials(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def has_gemini_credentials(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def has_github_credentials(self) -> bool:
        return bool(self.github_token.strip() and self.github_repo.strip())

    @property
    def is_offline_mode(self) -> bool:
        if self.skycast_offline_mode == "true":
            return True
        if self.skycast_offline_mode == "false":
            return False
        return not (self.has_supabase_credentials or self.has_hopsworks_credentials)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def llm_fallback_chain(self) -> tuple[dict[str, str | None], ...]:
        """Ordered cloud LLM chain: Groq primary -> Gemini fallback -> Groq tertiary."""
        return (
            {
                "provider": self.PRIMARY_LLM_PROVIDER,
                "model": self.PRIMARY_LLM_MODEL,
                "reasoning_effort": self.GROQ_REASONING_EFFORT,
            },
            {
                "provider": self.FALLBACK_LLM_PROVIDER,
                "model": self.FALLBACK_LLM_MODEL,
                "reasoning_effort": None,
            },
            {
                "provider": self.TERTIARY_LLM_PROVIDER,
                "model": self.TERTIARY_LLM_MODEL,
                "reasoning_effort": self.GROQ_REASONING_EFFORT,
            },
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings instance."""
    return Settings()


settings = get_settings()
