"""Deterministic Lahore alert & bulletin templates plus LLM JSON payload validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from skycast.config.settings import METRIC_SPECS, Settings, TARGET_METRICS, get_settings
from skycast.features.engineer import sanitize_json_scalar

_HAZARD_TEMPLATES: dict[str, dict[str, str]] = {
    "aqi_hazardous": {
        "title": "Hazardous air quality emergency in Lahore",
        "advisory_general": (
            "Remain indoors with windows sealed and avoid all non-essential outdoor activity."
        ),
        "advisory_sensitive": (
            "Children, older adults, and respiratory/cardiac patients should stay indoors with clean filtered air and keep rescue medication ready."
        ),
        "advisory_outdoor_workers": (
            "Suspend non-emergency outdoor work or enforce NIOSH N95/FFP2 respirators with frequent indoor rest breaks."
        ),
    },
    "aqi_very_unhealthy": {
        "title": "Very unhealthy smog and particulate levels",
        "advisory_general": (
            "Limit prolonged outdoor exertion, keep car and home ventilation on recirculation, and wear a well-fitted N95 mask outdoors."
        ),
        "advisory_sensitive": (
            "Avoid all outdoor exercise; children, elderly residents, and asthma patients should remain indoors."
        ),
        "advisory_outdoor_workers": (
            "Wear a fitted N95 respirator throughout outdoor shifts, rotate heavy duties, and monitor for coughing or shortness of breath."
        ),
    },
    "aqi_unhealthy": {
        "title": "Unhealthy air quality across Lahore",
        "advisory_general": (
            "Reduce prolonged or heavy outdoor exertion and wear an N95/KN95 mask during peak traffic hours."
        ),
        "advisory_sensitive": (
            "Sensitive groups should reschedule outdoor activities indoors and avoid congested roadways."
        ),
        "advisory_outdoor_workers": (
            "Use N95 respiratory protection during roadside or dusty tasks and take shaded breaks away from traffic."
        ),
    },
    "aqi_sensitive": {
        "title": "Air quality unhealthy for sensitive groups",
        "advisory_general": (
            "Active residents may continue routine outdoor plans while watching for haze buildup during evening inversion."
        ),
        "advisory_sensitive": (
            "Children, older adults, and people with asthma or heart disease should shorten intense outdoor exertion."
        ),
        "advisory_outdoor_workers": (
            "Keep dust masks accessible near busy corridors and limit high-exertion tasks during stagnant morning or night hours."
        ),
    },
    "temp_extreme_heat": {
        "title": "Extreme heat emergency (>44°C)",
        "advisory_general": (
            "Stay out of direct sun between 11:00 and 16:00 PKT, hydrate continuously with ORS/water, and check on vulnerable neighbors."
        ),
        "advisory_sensitive": (
            "Keep infants, elderly residents, and chronic-illness patients in cooled spaces and watch closely for heat exhaustion."
        ),
        "advisory_outdoor_workers": (
            "Shift heavy construction, delivery, and traffic duties to early morning/evening; mandate shaded cooling breaks every 20 minutes."
        ),
    },
    "temp_heatwave": {
        "title": "Heatwave warning (>40°C)",
        "advisory_general": (
            "Avoid peak afternoon sun, wear loose light-colored clothing, and drink water before feeling thirsty."
        ),
        "advisory_sensitive": (
            "Older adults, young children, and pregnant individuals should stay indoors during peak afternoon heat."
        ),
        "advisory_outdoor_workers": (
            "Schedule strenuous outdoor shifts before 11:00 PKT, enforce regular hydration stops, and use wide-brim sun protection."
        ),
    },
    "temp_cold_wave": {
        "title": "Cold wave warning (<4°C)",
        "advisory_general": (
            "Dress in insulated layers during night and early morning hours and ensure safe, ventilated indoor heating."
        ),
        "advisory_sensitive": (
            "Keep infants and older adults warm overnight and protect against cold-induced respiratory or cardiovascular strain."
        ),
        "advisory_outdoor_workers": (
            "Wear windproof insulated gloves, headgear, and layered outerwear during early-morning and night shifts."
        ),
    },
    "precip_urban_flood": {
        "title": "Urban flood risk from intense rainfall",
        "advisory_general": (
            "Avoid underpasses and low-lying roads prone to waterlogging, postpone non-essential travel, and stay clear of downed power lines."
        ),
        "advisory_sensitive": (
            "Keep essential medicines and backup lighting ready indoors and avoid wading through standing stormwater."
        ),
        "advisory_outdoor_workers": (
            "Halt elevated or electrical outdoor work during downpours and move equipment above street drainage level."
        ),
    },
    "precip_heavy_rain_mm": {
        "title": "Heavy rain expected across Lahore",
        "advisory_general": (
            "Allow extra commute time for slippery roads and localized ponding, and secure outdoor items."
        ),
        "advisory_sensitive": (
            "Avoid unnecessary travel in heavy rain and keep mobility aids and prescriptions dry and accessible."
        ),
        "advisory_outdoor_workers": (
            "Wear waterproof high-visibility gear, use non-slip footwear, and avoid exposed scaffolding during heavy bursts."
        ),
    },
    "precip_high_probability": {
        "title": "High probability of heavy rain",
        "advisory_general": (
            "Carry rain protection and monitor local drainage conditions before commuting."
        ),
        "advisory_sensitive": (
            "Plan indoor alternatives for errands and ensure damp-sensitive medical supplies are protected."
        ),
        "advisory_outdoor_workers": (
            "Prepare waterproof gear and stage tarpaulins over open-air materials before showers begin."
        ),
    },
    "humidity_muggy_heat_stress": {
        "title": "Muggy heat stress warning",
        "advisory_general": (
            "High humidity slows sweat cooling; limit midday exertion, use fans or AC, and replenish electrolytes."
        ),
        "advisory_sensitive": (
            "Older adults and cardiac or respiratory patients should stay in ventilated or air-conditioned rooms."
        ),
        "advisory_outdoor_workers": (
            "Increase hydration and shaded rest intervals since evaporative cooling is impaired in muggy air."
        ),
    },
    "humidity_extreme_dry": {
        "title": "Extreme dry air advisory (<15% RH)",
        "advisory_general": (
            "Increase fluid intake, moisturize exposed skin, and watch for static or dry-brush fire hazards."
        ),
        "advisory_sensitive": (
            "People with asthma, allergies, or dry-eye conditions should keep nasal saline and prescribed inhalers nearby."
        ),
        "advisory_outdoor_workers": (
            "Carry extra drinking water and wear protective eyewear against windborne dust in dry air."
        ),
    },
    "uv_extreme": {
        "title": "Extreme UV radiation warning (UV >= 11)",
        "advisory_general": (
            "Seek shade between 10:00 and 16:00 PKT and wear UV-blocking sunglasses, SPF 50+ sunscreen, and a hat."
        ),
        "advisory_sensitive": (
            "Keep children and sun-sensitive individuals indoors or under full shade during solar noon."
        ),
        "advisory_outdoor_workers": (
            "Use long-sleeve UV-protective clothing, neck flaps, and shaded rotation schedules during midday hours."
        ),
    },
    "uv_very_high": {
        "title": "Very high UV index advisory (UV 8-10)",
        "advisory_general": (
            "Minimize direct midday sun exposure and apply broad-spectrum SPF 30+ sunscreen."
        ),
        "advisory_sensitive": (
            "Protect children and photosensitive individuals with hats, long sleeves, and shaded play or rest areas."
        ),
        "advisory_outdoor_workers": (
            "Wear sun-shielding headwear, UV-rated eyewear, and light long-sleeve shirts throughout daylight shifts."
        ),
    },
    "uv_high": {
        "title": "High UV index advisory (UV 6-7)",
        "advisory_general": (
            "Wear sunglasses and sunscreen when outdoors around midday."
        ),
        "advisory_sensitive": (
            "Limit direct midday sun exposure for young children and photosensitive individuals."
        ),
        "advisory_outdoor_workers": (
            "Use wide-brim hats or caps and UV sunglasses during peak midday sun."
        ),
    },
    "wind_severe_squall": {
        "title": "Severe wind squall and dust storm warning (>60 km/h)",
        "advisory_general": (
            "Stay indoors away from windows, secure rooftop tanks/solar panels, and avoid parking under trees or billboards."
        ),
        "advisory_sensitive": (
            "Close doors and windows tightly to block airborne dust; keep asthma inhalers immediately accessible."
        ),
        "advisory_outdoor_workers": (
            "Suspend crane, scaffolding, and two-wheeler delivery operations immediately until gusts subside."
        ),
    },
    "wind_strong_gale": {
        "title": "Strong wind and dust advisory (>40 km/h)",
        "advisory_general": (
            "Use caution when driving high-profile vehicles or motorbikes and secure loose outdoor objects."
        ),
        "advisory_sensitive": (
            "Wear a dust mask outdoors if wind kicks up particulate matter and limit exposure for asthma patients."
        ),
        "advisory_outdoor_workers": (
            "Secure scaffolding, wear eye and dust protection, and pause elevated work during sudden gusts."
        ),
    },
}


def build_deterministic_alerts(
    hazards: Sequence[Mapping[str, Any]],
    *,
    city: str = "Lahore",
) -> dict[str, Any]:
    """Synthesize structured 3-audience hazard alerts without cloud LLM access."""
    alerts: list[dict[str, Any]] = []
    for hazard in hazards:
        rule_id = str(hazard["rule_id"])
        metric = str(hazard["metric"])
        severity = str(hazard["severity"])
        trigger_val = float(hazard["trigger_value"])
        threshold_val = float(hazard["threshold"])
        spec = METRIC_SPECS[metric]

        tpl = _HAZARD_TEMPLATES.get(rule_id)
        if tpl is None:
            tpl = {
                "title": f"{severity.capitalize()} {spec.label.lower()} advisory in {city}",
                "advisory_general": (
                    f"{spec.label} reached {trigger_val:.1f} {spec.unit} (threshold {threshold_val:.1f} {spec.unit}); adjust outdoor plans accordingly."
                ),
                "advisory_sensitive": (
                    f"Sensitive groups should take precautions as {spec.label.lower()} reaches {trigger_val:.1f} {spec.unit}."
                ),
                "advisory_outdoor_workers": (
                    f"Outdoor workers should apply protective gear and rest protocols for {spec.label.lower()} ({trigger_val:.1f} {spec.unit})."
                ),
            }

        alerts.append(
            {
                "rule_id": rule_id,
                "metric": metric,
                "title": tpl["title"],
                "advisory_general": tpl["advisory_general"],
                "advisory_sensitive": tpl["advisory_sensitive"],
                "advisory_outdoor_workers": tpl["advisory_outdoor_workers"],
            }
        )
    return {"alerts": alerts}


def validate_llm_alerts_payload(
    raw_json: Mapping[str, Any],
    hazards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate LLM-generated 3-audience alert JSON against triggered hazards."""
    items = raw_json.get("alerts")
    if not isinstance(items, list) or not items:
        raise ValueError("LLM alerts response must include a non-empty 'alerts' list.")

    by_rule: dict[str, Mapping[str, Any]] = {}
    by_metric: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Each alert entry must be a JSON object.")
        rid = str(item.get("rule_id") or "").strip()
        met = str(item.get("metric") or "").strip()
        if rid:
            by_rule[rid] = item
        if met:
            by_metric[met] = item

    validated: list[dict[str, Any]] = []
    for idx, hazard in enumerate(hazards):
        rule_id = str(hazard["rule_id"])
        metric = str(hazard["metric"])
        matched = by_rule.get(rule_id) or by_metric.get(metric)
        if matched is None and idx < len(items) and isinstance(items[idx], Mapping):
            matched = items[idx]
        if matched is None:
            raise ValueError(f"LLM alerts response omitted hazard '{rule_id}' ({metric}).")

        required_fields = (
            "title",
            "advisory_general",
            "advisory_sensitive",
            "advisory_outdoor_workers",
        )
        cleaned: dict[str, Any] = {"rule_id": rule_id, "metric": metric}
        for field in required_fields:
            val = matched.get(field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"LLM alert for '{rule_id}' missing non-empty '{field}'."
                )
            cleaned[field] = val.strip()
        validated.append(cleaned)

    return {"alerts": validated}


def _format_local_time(timestamp_utc: str, tz_name: str) -> str:
    dt = pd.to_datetime(timestamp_utc, utc=True).to_pydatetime().astimezone(ZoneInfo(tz_name))
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def build_deterministic_weather_report(
    *,
    timestamp_utc: str,
    trigger_reason: str,
    latest_features: Mapping[str, Any],
    forecasts_24h: Mapping[str, Sequence[Mapping[str, Any]]],
    alerts_payload: Sequence[Mapping[str, Any]],
    shap_drivers: Sequence[Mapping[str, Any]],
    drift_results: Sequence[Mapping[str, Any]],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Synthesize a deterministic broadcast-grade Lahore weather & AQI report."""
    cfg = settings or get_settings()
    local_str = _format_local_time(timestamp_utc, cfg.lahore_timezone)

    temp_now = latest_features.get("temperature_2m")
    aqi_now = latest_features.get("us_aqi")
    wind_now = latest_features.get("wind_speed_10m")
    hum_now = latest_features.get("relative_humidity_2m")

    temp_fc = forecasts_24h.get("temperature_2m", [])
    aqi_fc = forecasts_24h.get("us_aqi", [])
    rain_fc = forecasts_24h.get("precipitation_mm", [])

    temp_vals = [float(p["predicted_value"]) for p in temp_fc if p.get("predicted_value") is not None]
    aqi_vals = [float(p["predicted_value"]) for p in aqi_fc if p.get("predicted_value") is not None]
    rain_vals = [float(p["predicted_value"]) for p in rain_fc if p.get("predicted_value") is not None]

    temp_range = (
        f"{min(temp_vals):.1f}°C to {max(temp_vals):.1f}°C"
        if temp_vals
        else (f"{float(temp_now):.1f}°C" if temp_now is not None else "N/A")
    )
    aqi_peak = (
        f"{max(aqi_vals):.0f}"
        if aqi_vals
        else (f"{float(aqi_now):.0f}" if aqi_now is not None else "N/A")
    )
    rain_total = f"{sum(rain_vals):.1f} mm" if rain_vals else "0.0 mm"

    if alerts_payload:
        top_alert = max(
            alerts_payload,
            key=lambda a: {"critical": 3, "warning": 2, "info": 1}.get(str(a["severity"]), 0),
        )
        headline = f"{cfg.city_name} Weather & AQI Bulletin: {top_alert['title']}"
    else:
        headline = f"{cfg.city_name} 24-Hour Weather & Air Quality Outlook ({local_str})"

    temp_now_str = f"{float(temp_now):.1f}°C" if temp_now is not None else "N/A"
    aqi_now_str = f"{float(aqi_now):.0f}" if aqi_now is not None else "N/A"
    wind_now_str = f"{float(wind_now):.1f} km/h" if wind_now is not None else "N/A"
    hum_now_str = f"{float(hum_now):.0f}%" if hum_now is not None else "N/A"

    summary = (
        f"As of {local_str}, {cfg.city_name} stands at {temp_now_str} with US AQI {aqi_now_str}, "
        f"humidity {hum_now_str}, and winds at {wind_now_str}. Over the next {cfg.forecast_horizon_hours} hours, "
        f"temperatures will range {temp_range} with peak projected AQI near {aqi_peak} and {rain_total} cumulative rain; "
        f"{len(alerts_payload)} active hazard alert(s) are in effect."
    )

    highlights: list[str] = [
        f"24h temperature range: {temp_range} (current {temp_now_str}).",
        f"Peak projected US AQI: {aqi_peak} (current {aqi_now_str}).",
        f"Expected {cfg.forecast_horizon_hours}h rainfall accumulation: {rain_total}.",
    ]
    for alert in list(alerts_payload)[:2]:
        highlights.append(f"[{str(alert['severity']).upper()}] {alert['title']}.")

    lines: list[str] = [
        f"# {headline}",
        "",
        f"*Issued: {local_str} | Trigger: `{trigger_reason}`*",
        "",
        "## Current Conditions",
        f"- **Temperature:** {temp_now_str}",
        f"- **US AQI:** {aqi_now_str}",
        f"- **Humidity:** {hum_now_str}",
        f"- **Wind Speed:** {wind_now_str}",
        "",
        f"## {cfg.forecast_horizon_hours}-Hour Forecast Trajectory",
    ]
    for metric in TARGET_METRICS:
        pts = forecasts_24h.get(metric, [])
        if not pts:
            continue
        spec = METRIC_SPECS[metric]
        vals = [float(p["predicted_value"]) for p in pts]
        t_end = pts[-1]["predicted_value"]
        lines.append(
            f"- **{spec.label} ({spec.unit}):** min {min(vals):.1f}, max {max(vals):.1f}, T+{len(pts)}h {float(t_end):.1f} ({pts[-1]['model_family']})"
        )

    lines.extend(["", "## Active Hazard Advisories"])
    if alerts_payload:
        for alert in alerts_payload:
            lines.append(
                f"- **[{str(alert['severity']).upper()}] {alert['title']}** "
                f"(value {float(alert['trigger_value']):.1f} vs threshold {float(alert['threshold']):.1f}): "
                f"{alert['advisory_general']}"
            )
    else:
        lines.append("- No meteorological or air-quality thresholds breached in the forecast window.")

    if shap_drivers:
        lines.extend(["", "## Key Forecast Drivers (SHAP)"])
        for row in list(shap_drivers)[:4]:
            attrs = row.get("feature_attributions") or []
            top_names = [str(a.get("feature")) for a in attrs[:3] if isinstance(a, Mapping) and a.get("feature")]
            if top_names:
                lines.append(f"- **{row.get('metric')}**: {', '.join(top_names)}")

    drifted = [d["metric"] for d in drift_results if d.get("status") == "drift_detected"]
    failed_checks = [d["metric"] for d in drift_results if d.get("status") == "check_failed"]
    lines.extend(
        [
            "",
            "## Monitoring & Drift Status",
            (
                f"- Drift detected on: {', '.join(drifted)}."
                if drifted
                else "- No statistical data or model drift detected across monitored metrics."
            ),
        ]
    )
    if failed_checks:
        lines.append(f"- Incomplete baseline/window checks (`check_failed`): {len(failed_checks)} metric check(s).")

    return sanitize_json_scalar(
        {
            "headline": headline,
            "summary": summary,
            "bulletin_markdown": "\n".join(lines),
            "highlights": highlights,
        }
    )


def validate_llm_report_payload(raw_json: Mapping[str, Any]) -> dict[str, Any]:
    """Validate LLM-generated weather broadcast bulletin JSON."""
    for field in ("headline", "summary", "bulletin_markdown"):
        val = raw_json.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"LLM weather report missing non-empty '{field}'.")

    highlights_raw = raw_json.get("highlights")
    if not isinstance(highlights_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in highlights_raw
    ):
        raise ValueError("LLM weather report 'highlights' must be a list of non-empty strings.")

    return sanitize_json_scalar(
        {
            "headline": str(raw_json["headline"]).strip(),
            "summary": str(raw_json["summary"]).strip(),
            "bulletin_markdown": str(raw_json["bulletin_markdown"]).strip(),
            "highlights": [str(item).strip() for item in highlights_raw],
        }
    )
