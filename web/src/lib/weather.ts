import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudRain,
  CloudSnow,
  CloudSun,
  Sun,
  type LucideIcon,
} from "lucide-react"

import type {
  MetricSpec,
  MetricSpecsMap,
  WeatherObservation,
} from "@/lib/api"

export const POLLUTANT_LABELS: Record<string, string> = {
  pm2_5: "PM2.5",
  pm10: "PM10",
  ozone: "O3",
  nitrogen_dioxide: "NO2",
  sulphur_dioxide: "SO2",
  carbon_monoxide: "CO",
}

export const CANONICAL_MODEL_FAMILIES = [
  "SARIMAX",
  "RandomForest",
  "Ridge",
  "LSTM",
  "Naive24h",
] as const

const MODEL_FAMILY_LABELS: Record<string, string> = {
  sarimax: "SARIMAX",
  random_forest: "Random Forest",
  randomforest: "Random Forest",
  rf: "Random Forest",
  ridge: "Ridge",
  lstm: "LSTM",
  naive_24h: "24h naive baseline",
  naive24h: "24h naive baseline",
  naive: "24h naive baseline",
}

const SOURCE_LABELS: Record<string, string> = {
  open_meteo: "Open-Meteo",
  "open-meteo": "Open-Meteo",
  met_no: "Met.no",
  "met.no": "Met.no",
}

const KNOWN_FEATURE_LABELS: Record<string, string> = {
  us_aqi: "Air quality",
  temperature_2m: "Temperature",
  apparent_temperature: "Feels like temperature",
  precipitation_mm: "Precipitation",
  precipitation_prob: "Chance of rain",
  relative_humidity_2m: "Humidity",
  uv_index: "UV index",
  wind_speed_10m: "Wind speed",
  wind_direction_10m: "Wind direction",
  wind_dir_sin: "Wind direction (N/S)",
  wind_dir_cos: "Wind direction (E/W)",
  surface_pressure: "Surface pressure",
  cloud_cover: "Cloud cover",
  weather_code: "Weather condition",
  pm2_5: "PM2.5",
  pm10: "PM10",
  ozone: "Ozone (O3)",
  nitrogen_dioxide: "Nitrogen dioxide (NO2)",
  sulphur_dioxide: "Sulphur dioxide (SO2)",
  carbon_monoxide: "Carbon monoxide (CO)",
  hour_sin: "Time of day (cycle)",
  hour_cos: "Time of day (phase)",
  day_of_week_sin: "Day of week (cycle)",
  day_of_week_cos: "Day of week (phase)",
  month_sin: "Season of year (cycle)",
  month_cos: "Season of year (phase)",
  is_weekend: "Weekend indicator",
  is_lahore_smog_season: "Smog season",
  is_monsoon_season: "Monsoon season",
  source_disagreement_temp: "Source temperature spread",
}

const DRIFT_TEST_LABELS: Record<string, string> = {
  psi_ks: "Distribution shift (PSI & KS)",
  psi: "Population stability index",
  ks: "Kolmogorov-Smirnov test",
  rmse_ratio: "Rolling forecast error ratio",
}

const DRIFT_ACTION_LABELS: Record<string, string> = {
  none: "No action needed",
  workflow_dispatch_triggered: "Retraining triggered",
  local_retrain_triggered: "Retraining triggered",
  logged_failure: "Check incomplete",
  retrain_dispatch_failed: "Retraining request failed",
}

export interface ConditionDisplay {
  label: string
  icon: LucideIcon
}

const COMPASS_POINTS = [
  "N",
  "NNE",
  "NE",
  "ENE",
  "E",
  "ESE",
  "SE",
  "SSE",
  "S",
  "SSW",
  "SW",
  "WSW",
  "W",
  "WNW",
  "NW",
  "NNW",
] as const

/**
 * Maps WMO weather interpretation codes (0–99) and optional Met.no condition
 * symbols into plain sentence-case labels and Lucide weather icons.
 */
export function resolveWeatherCondition(
  weatherCode: number | null | undefined,
  conditionSymbol?: string | null
): ConditionDisplay {
  if (typeof weatherCode === "number" && Number.isFinite(weatherCode)) {
    const code = Math.round(weatherCode)
    if (code === 0) return { label: "Clear sky", icon: Sun }
    if (code === 1) return { label: "Mainly clear", icon: CloudSun }
    if (code === 2) return { label: "Partly cloudy", icon: CloudSun }
    if (code === 3) return { label: "Overcast", icon: Cloud }
    if (code === 45 || code === 48) return { label: "Fog", icon: CloudFog }
    if (code >= 51 && code <= 57) return { label: "Drizzle", icon: CloudDrizzle }
    if (code >= 61 && code <= 67) return { label: "Rain", icon: CloudRain }
    if (code >= 71 && code <= 77) return { label: "Snow", icon: CloudSnow }
    if (code >= 80 && code <= 82) return { label: "Rain showers", icon: CloudRain }
    if (code === 85 || code === 86) return { label: "Snow showers", icon: CloudSnow }
    if (code >= 95 && code <= 99) return { label: "Thunderstorm", icon: CloudLightning }
  }

  const symbol = conditionSymbol?.toLowerCase().trim() ?? ""
  if (symbol) {
    if (symbol.includes("thunder")) return { label: "Thunderstorm", icon: CloudLightning }
    if (symbol.includes("snow") || symbol.includes("sleet")) {
      return { label: "Snow", icon: CloudSnow }
    }
    if (symbol.includes("rain") || symbol.includes("shower")) {
      return { label: "Rain", icon: CloudRain }
    }
    if (symbol.includes("drizzle")) return { label: "Drizzle", icon: CloudDrizzle }
    if (symbol.includes("fog") || symbol.includes("haze") || symbol.includes("mist")) {
      return { label: "Fog or haze", icon: CloudFog }
    }
    if (symbol.includes("partly") || symbol.includes("fair")) {
      return { label: "Partly cloudy", icon: CloudSun }
    }
    if (symbol.includes("cloud")) return { label: "Cloudy", icon: Cloud }
    if (symbol.includes("clear") || symbol.includes("sun")) {
      return { label: "Clear sky", icon: Sun }
    }
  }

  return { label: "Conditions unavailable", icon: Cloud }
}

export function normalizeDegrees(degrees: number | null | undefined): number | null {
  if (typeof degrees !== "number" || !Number.isFinite(degrees)) {
    return null
  }
  const mod = ((degrees % 360) + 360) % 360
  return mod === 360 ? 0 : mod
}

export function windDirectionLabel(degrees: number | null | undefined): string {
  const norm = normalizeDegrees(degrees)
  if (norm === null) {
    return "—"
  }
  const index = Math.round(norm / 22.5) % 16
  return COMPASS_POINTS[index]
}

export function formatMetricValue(
  value: number | null | undefined,
  decimals = 1
): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—"
  }
  return value.toFixed(decimals)
}

export function chartTokenToCssVar(chartToken: string | null | undefined): string {
  const trimmed = chartToken?.trim()
  if (!trimmed) {
    return "var(--primary)"
  }
  if (trimmed.startsWith("var(")) {
    return trimmed
  }
  return trimmed.startsWith("--") ? `var(${trimmed})` : `var(--${trimmed})`
}

export function getMetricSpec(
  metric: string,
  metricSpecs?: MetricSpecsMap | null
): MetricSpec | undefined {
  return metricSpecs?.[metric]
}

export function getMetricColorVar(
  metric: string,
  metricSpecs?: MetricSpecsMap | null
): string {
  const spec = getMetricSpec(metric, metricSpecs)
  return chartTokenToCssVar(spec?.chart_token)
}

export function readObservationNumeric(
  obs: WeatherObservation | null | undefined,
  field: string
): number | null {
  if (!obs) {
    return null
  }
  const value = (obs as unknown as Record<string, unknown>)[field]
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

const MODEL_FAMILY_COLORS: Record<string, string> = {
  sarimax: "var(--model-sarimax)",
  random_forest: "var(--model-rf)",
  randomforest: "var(--model-rf)",
  rf: "var(--model-rf)",
  ridge: "var(--model-ridge)",
  lstm: "var(--model-lstm)",
  naive_24h: "var(--model-naive)",
  naive24h: "var(--model-naive)",
  naive: "var(--model-naive)",
}

export function formatSourceLabel(source: string): string {
  const normalized = source.trim().toLowerCase()
  return SOURCE_LABELS[normalized] ?? humanizeIdentifier(source)
}

export function formatFeatureName(feature: string | null | undefined): string {
  if (!feature) {
    return "—"
  }
  const trimmed = feature.trim()
  if (!trimmed) {
    return "—"
  }
  const lower = trimmed.toLowerCase()
  if (lower in KNOWN_FEATURE_LABELS) {
    return KNOWN_FEATURE_LABELS[lower]
  }

  // Handle patterns like `lag_1h`, `temperature_2m_lag_6h`, `rolling_mean_24h`, `roc_3h`
  const lagOnly = lower.match(/^lag_(\d+)h$/)
  if (lagOnly) {
    const hrs = Number(lagOnly[1])
    return hrs === 1 ? "Previous hour value" : `Value ${hrs}h earlier`
  }

  const metricLag = lower.match(/^(.+)_lag_(\d+)h$/)
  if (metricLag) {
    const base = KNOWN_FEATURE_LABELS[metricLag[1]] ?? humanizeIdentifier(metricLag[1])
    return `${base} (${metricLag[2]}h ago)`
  }

  const rollingOnly = lower.match(/^rolling_(mean|std|min|max)_(\d+)h$/)
  if (rollingOnly) {
    const statMap: Record<string, string> = {
      mean: "average",
      std: "variability",
      min: "minimum",
      max: "maximum",
    }
    return `${rollingOnly[2]}-hour ${statMap[rollingOnly[1]] ?? rollingOnly[1]}`
  }

  const metricRolling = lower.match(/^(.+)_rolling_(mean|std|min|max)_(\d+)h$/)
  if (metricRolling) {
    const base = KNOWN_FEATURE_LABELS[metricRolling[1]] ?? humanizeIdentifier(metricRolling[1])
    const statMap: Record<string, string> = {
      mean: "avg",
      std: "spread",
      min: "min",
      max: "max",
    }
    return `${base} (${metricRolling[3]}h ${statMap[metricRolling[2]] ?? metricRolling[2]})`
  }

  const rocOnly = lower.match(/^roc_(\d+)h$/)
  if (rocOnly) {
    return `${rocOnly[1]}-hour change`
  }

  const metricRoc = lower.match(/^(.+)_roc_(\d+)h$/)
  if (metricRoc) {
    const base = KNOWN_FEATURE_LABELS[metricRoc[1]] ?? humanizeIdentifier(metricRoc[1])
    return `${base} (${metricRoc[2]}h change)`
  }

  return humanizeIdentifier(trimmed)
}

export function formatLimeRule(rule: string | null | undefined): string {
  if (!rule) {
    return "—"
  }
  const trimmed = rule.trim()
  if (!trimmed) {
    return "—"
  }
  return trimmed.replace(/[a-zA-Z][a-zA-Z0-9_]{2,}/g, (token) => {
    if (token.includes("_") || token.toLowerCase() in KNOWN_FEATURE_LABELS) {
      return formatFeatureName(token)
    }
    return token
  })
}

export function formatDriftTestName(testName: string | null | undefined): string {
  if (!testName) {
    return "—"
  }
  const normalized = testName.trim().toLowerCase()
  return DRIFT_TEST_LABELS[normalized] ?? humanizeIdentifier(testName)
}

export function formatDriftAction(action: string | null | undefined): string {
  if (!action) {
    return "—"
  }
  const normalized = action.trim().toLowerCase()
  return DRIFT_ACTION_LABELS[normalized] ?? humanizeIdentifier(action)
}

export function formatModelLabel(
  modelFamily: string | null | undefined
): string {
  if (!modelFamily) {
    return "—"
  }
  const normalized = modelFamily.trim().toLowerCase()
  if (normalized in MODEL_FAMILY_LABELS) {
    return MODEL_FAMILY_LABELS[normalized]
  }
  const readable = modelFamily.trim().replace(/_/g, " ")
  return readable.charAt(0).toUpperCase() + readable.slice(1)
}

export function getModelColorVar(
  modelFamily: string | null | undefined
): string {
  if (!modelFamily) {
    return "var(--primary)"
  }
  const normalized = modelFamily.trim().toLowerCase()
  return MODEL_FAMILY_COLORS[normalized] ?? "var(--primary)"
}

export function humanizeIdentifier(value: string | null | undefined): string {
  if (!value) {
    return "—"
  }
  const trimmed = value.trim()
  if (!trimmed) {
    return "—"
  }
  const readable = trimmed.replace(/[_-]+/g, " ")
  return readable.charAt(0).toUpperCase() + readable.slice(1)
}

export function resolveMetricLabel(
  metric: string | null | undefined,
  metricSpecs?: MetricSpecsMap | null,
  overrides?: Record<string, { label?: string | null }> | null
): string {
  if (!metric) {
    return "—"
  }
  const overrideLabel = overrides?.[metric]?.label?.trim()
  if (overrideLabel) {
    return overrideLabel
  }
  const specLabel = metricSpecs?.[metric]?.label?.trim()
  if (specLabel) {
    return specLabel
  }
  if (metric in POLLUTANT_LABELS) {
    return POLLUTANT_LABELS[metric]
  }
  return humanizeIdentifier(metric)
}

export function orderMetricKeys(
  keys: string[],
  coreMetricGroups?: Record<string, string[]> | null
): string[] {
  const ordered: string[] = []
  const seen = new Set<string>()

  if (coreMetricGroups) {
    for (const members of Object.values(coreMetricGroups)) {
      const primary = members[0]
      if (primary && keys.includes(primary) && !seen.has(primary)) {
        seen.add(primary)
        ordered.push(primary)
      }
    }
  }

  for (const key of keys) {
    if (!seen.has(key)) {
      seen.add(key)
      ordered.push(key)
    }
  }

  return ordered
}

export function pickDefaultMetric(
  metricSpecs?: MetricSpecsMap | null,
  fallbackKeys: string[] = []
): string | null {
  const keys = Object.keys(metricSpecs ?? {})
  const candidates = keys.length > 0 ? keys : fallbackKeys
  if (candidates.includes("temperature_2m")) {
    return "temperature_2m"
  }
  return candidates[0] ?? null
}
