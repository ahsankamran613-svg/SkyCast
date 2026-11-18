"""Verification tests for Component 1: layout, configuration, dependencies, and SQL schema."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from skycast.config.settings import (
    CORE_METRIC_GROUPS,
    FALLBACK_LLM_MODEL,
    FALLBACK_LLM_PROVIDER,
    GROQ_REASONING_EFFORT,
    HAZARD_RULES,
    METRIC_SPECS,
    MODEL_FAMILIES,
    PRIMARY_LLM_MODEL,
    PRIMARY_LLM_PROVIDER,
    TARGET_METRICS,
    TERTIARY_LLM_MODEL,
    TERTIARY_LLM_PROVIDER,
    Settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_ML_PACKAGES = {"torch", "shap", "lime", "statsmodels", "scikit-learn", "hopsworks"}


def test_repository_structure_exists() -> None:
    expected_dirs = [
        REPO_ROOT / "skycast",
        REPO_ROOT / "skycast" / "config",
        REPO_ROOT / "skycast" / "data",
        REPO_ROOT / "skycast" / "data" / "sources",
        REPO_ROOT / "skycast" / "features",
        REPO_ROOT / "skycast" / "ml",
        REPO_ROOT / "skycast" / "ml" / "models",
        REPO_ROOT / "skycast" / "agents",
        REPO_ROOT / "skycast" / "pipelines",
        REPO_ROOT / "api",
        REPO_ROOT / "web",
        REPO_ROOT / "supabase" / "migrations",
        REPO_ROOT / "docs",
        REPO_ROOT / "tests",
    ]
    for path in expected_dirs:
        assert path.is_dir(), f"Expected directory missing: {path}"


def test_settings_defaults_and_lahore_targets() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.city_name == "Lahore"
    assert cfg.lahore_lat == pytest.approx(31.5204)
    assert cfg.lahore_lon == pytest.approx(74.3587)
    assert cfg.lahore_timezone == "Asia/Karachi"

    assert len(CORE_METRIC_GROUPS) == 6
    assert len(TARGET_METRICS) == 8
    grouped_targets = {m for metrics in CORE_METRIC_GROUPS.values() for m in metrics}
    assert grouped_targets == set(TARGET_METRICS)
    assert MODEL_FAMILIES == ("Naive24h", "SARIMAX", "RandomForest", "Ridge", "LSTM")


def test_metric_clamping_and_circular_wind() -> None:
    assert METRIC_SPECS["us_aqi"].clamp(-10.0) == 0.0
    assert METRIC_SPECS["us_aqi"].clamp(650.0) == 500.0
    assert METRIC_SPECS["precipitation_prob"].clamp(120.0) == 100.0
    assert METRIC_SPECS["wind_direction_10m"].clamp(365.0) == 5.0
    assert METRIC_SPECS["wind_direction_10m"].clamp(-15.0) == 345.0


def test_llm_fallback_chain_and_low_reasoning_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    assert PRIMARY_LLM_PROVIDER == "groq"
    assert PRIMARY_LLM_MODEL == "openai/gpt-oss-120b"
    assert FALLBACK_LLM_PROVIDER == "gemini"
    assert FALLBACK_LLM_MODEL == "gemini-3.5-flash-lite"
    assert TERTIARY_LLM_PROVIDER == "groq"
    assert TERTIARY_LLM_MODEL == "qwen-3.8-27b"
    assert GROQ_REASONING_EFFORT == "low"

    cfg = Settings(_env_file=None)
    chain = cfg.llm_fallback_chain
    assert [step["provider"] for step in chain] == ["groq", "gemini", "groq"]
    assert [step["model"] for step in chain] == [
        "openai/gpt-oss-120b",
        "gemini-3.5-flash-lite",
        "qwen-3.8-27b",
    ]
    assert chain[0]["reasoning_effort"] == "low"
    assert chain[1]["reasoning_effort"] is None
    assert chain[2]["reasoning_effort"] == "low"

    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite-custom")
    monkeypatch.setenv("GROQ_TERTIARY_MODEL", "qwen-3.8-27b-custom")
    overridden = Settings(_env_file=None)
    assert overridden.FALLBACK_LLM_MODEL == "gemini-3.5-flash-lite-custom"
    assert overridden.TERTIARY_LLM_MODEL == "qwen-3.8-27b-custom"

    monkeypatch.setenv("GROQ_REASONING_EFFORT", "high")
    with pytest.raises(ValidationError, match="GROQ_REASONING_EFFORT must be 'low'"):
        Settings(_env_file=None)


def test_offline_mode_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("HOPSWORKS_API_KEY", raising=False)
    monkeypatch.delenv("HOPSWORKS_PROJECT_NAME", raising=False)

    assert Settings(_env_file=None).is_offline_mode is True

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    assert Settings(_env_file=None).is_offline_mode is False

    monkeypatch.setenv("SKYCAST_OFFLINE_MODE", "true")
    assert Settings(_env_file=None).is_offline_mode is True


def test_hazard_rules_evaluation() -> None:
    rules_by_id = {rule.rule_id: rule for rule in HAZARD_RULES}

    assert rules_by_id["aqi_sensitive"].matches({"us_aqi": 125.0}) is True
    assert rules_by_id["aqi_unhealthy"].matches({"us_aqi": 175.0}) is True
    assert rules_by_id["aqi_very_unhealthy"].matches({"us_aqi": 250.0}) is True
    assert rules_by_id["aqi_hazardous"].matches({"us_aqi": 320.0}) is True

    # Compound humidity + heat rule requires both thresholds
    assert rules_by_id["humidity_muggy_heat_stress"].matches(
        {"relative_humidity_2m": 88.0, "temperature_2m": 30.0}
    ) is False
    assert rules_by_id["humidity_muggy_heat_stress"].matches(
        {"relative_humidity_2m": 88.0, "temperature_2m": 35.0}
    ) is True

    assert rules_by_id["temp_heatwave"].matches({"temperature_2m": 42.0}) is True
    assert rules_by_id["temp_extreme_heat"].matches({"temperature_2m": 45.0}) is True
    assert rules_by_id["temp_cold_wave"].matches({"temperature_2m": 2.5}) is True
    assert rules_by_id["wind_strong_gale"].matches({"wind_speed_10m": 48.0}) is True
    assert rules_by_id["wind_severe_squall"].matches({"wind_speed_10m": 65.0}) is True


def _extract_pkg_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        pkg = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip().lower()
        if pkg:
            names.add(pkg)
    return names


def test_dependency_split_for_vercel_bundle_limit() -> None:
    root_reqs = _extract_pkg_names((REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())
    api_reqs = _extract_pkg_names((REPO_ROOT / "api" / "requirements.txt").read_text(encoding="utf-8").splitlines())
    ml_reqs = _extract_pkg_names((REPO_ROOT / "requirements-ml.txt").read_text(encoding="utf-8").splitlines())

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core_deps = _extract_pkg_names(pyproject["project"]["dependencies"])
    ml_optional_deps = _extract_pkg_names(pyproject["project"]["optional-dependencies"]["ml"])

    for heavy_pkg in HEAVY_ML_PACKAGES:
        assert heavy_pkg not in root_reqs
        assert heavy_pkg not in api_reqs
        assert heavy_pkg not in core_deps
        assert heavy_pkg in ml_reqs
        assert heavy_pkg in ml_optional_deps


def test_env_example_contains_required_keys() -> None:
    env_text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    required_keys = [
        "LAHORE_LAT=31.5204",
        "LAHORE_LON=74.3587",
        "LAHORE_TIMEZONE=Asia/Karachi",
        "HOPSWORKS_API_KEY=",
        "HOPSWORKS_PROJECT_NAME=",
        "SUPABASE_URL=",
        "SUPABASE_SERVICE_ROLE_KEY=",
        "VITE_SUPABASE_URL=",
        "VITE_SUPABASE_ANON_KEY=",
        "GROQ_API_KEY=",
        "GROQ_PRIMARY_MODEL=openai/gpt-oss-120b",
        "GROQ_TERTIARY_MODEL=qwen-3.8-27b",
        "GROQ_REASONING_EFFORT=low",
        "GEMINI_API_KEY=",
        "GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite",
        "GITHUB_TOKEN=",
        "GITHUB_REPO=",
    ]
    for key in required_keys:
        assert key in env_text, f"Missing key in .env.example: {key}"


def test_supabase_migration_schema_completeness() -> None:
    sql_path = REPO_ROOT / "supabase" / "migrations" / "001_init_skycast.sql"
    sql = sql_path.read_text(encoding="utf-8")

    # Balanced parentheses check
    assert sql.count("(") == sql.count(")")

    expected_tables = [
        "public.weather_features_hourly",
        "public.forecast_ledger",
        "public.model_registry_metrics",
        "public.model_comparison_series",
        "public.shap_lime_explanations",
        "public.weather_alerts",
        "public.weather_reports",
        "public.agent_call_log",
        "public.agent_graph_trace",
        "public.drift_log",
        "public.eda_summaries",
        "public.pipeline_runs",
    ]
    for table in expected_tables:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql

    required_snippets = [
        "horizon_hour SMALLINT NOT NULL CHECK (horizon_hour BETWEEN 1 AND 24)",
        "severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical'))",
        "drift_type TEXT NOT NULL CHECK (drift_type IN ('data_drift', 'model_drift'))",
        "status TEXT NOT NULL CHECK (status IN ('no_drift', 'drift_detected', 'check_failed'))",
        "status TEXT NOT NULL CHECK (status IN ('completed', 'skipped', 'failed'))",
        "advisory_general TEXT NOT NULL",
        "advisory_sensitive TEXT NOT NULL",
        "advisory_outdoor_workers TEXT NOT NULL",
        "apparent_temperature DOUBLE PRECISION",
        "weather_code INTEGER",
        "pm2_5 DOUBLE PRECISION",
        "nitrogen_dioxide DOUBLE PRECISION",
        "sarimax DOUBLE PRECISION",
        "random_forest DOUBLE PRECISION",
        "ridge DOUBLE PRECISION",
        "lstm DOUBLE PRECISION",
        "naive_24h DOUBLE PRECISION",
    ]
    for snippet in required_snippets:
        assert snippet in sql, f"Missing expected SQL snippet: {snippet}"
