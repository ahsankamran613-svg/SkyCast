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

interface BaseFeatureDef {
  label: string
  description: string
  unit?: string
}

const KNOWN_FEATURE_DEFS: Record<string, BaseFeatureDef> = {
  us_aqi: {
    label: "Air quality (AQI)",
    description: "Current US Air Quality Index (higher means more polluted).",
    unit: "AQI",
  },
  temperature_2m: {
    label: "Temperature",
    description: "Air temperature measured 2 meters above ground.",
    unit: "°C",
  },
  apparent_temperature: {
    label: "Feels-like temperature",
    description: "Perceived temperature combining heat, humidity, and wind.",
    unit: "°C",
  },
  precipitation_mm: {
    label: "Rainfall",
    description: "Hourly precipitation amount.",
    unit: "mm/h",
  },
  precipitation_prob: {
    label: "Chance of rain",
    description: "Estimated probability of rain for the hour.",
    unit: "%",
  },
  relative_humidity_2m: {
    label: "Humidity",
    description: "Relative moisture in the air at 2 meters above ground.",
    unit: "%",
  },
  uv_index: {
    label: "UV index",
    description: "Ultraviolet sunlight intensity (0 at night, highest at midday).",
  },
  wind_speed_10m: {
    label: "Wind speed",
    description: "Wind speed measured 10 meters above ground.",
    unit: "km/h",
  },
  wind_direction_10m: {
    label: "Wind direction",
    description: "Compass direction the wind is blowing from (0°–360°).",
    unit: "°",
  },
  wind_dir_sin: {
    label: "Wind direction (N/S)",
    description: "North–south component of the wind direction.",
  },
  wind_dir_cos: {
    label: "Wind direction (E/W)",
    description: "East–west component of the wind direction.",
  },
  wind_u: {
    label: "East–west wind",
    description: "East-to-west wind velocity component.",
    unit: "km/h",
  },
  wind_v: {
    label: "North–south wind",
    description: "North-to-south wind velocity component.",
    unit: "km/h",
  },
  surface_pressure: {
    label: "Air pressure",
    description: "Atmospheric pressure at ground level (shifts signal incoming weather).",
    unit: "hPa",
  },
  cloud_cover: {
    label: "Cloud cover",
    description: "Percentage of sky covered by clouds.",
    unit: "%",
  },
  weather_code: {
    label: "Weather condition",
    description: "Standard WMO sky and precipitation condition code.",
  },
  pm2_5: {
    label: "PM2.5",
    description: "Fine airborne smoke/dust particles smaller than 2.5 microns.",
    unit: "µg/m³",
  },
  pm10: {
    label: "PM10",
    description: "Coarse airborne dust particles smaller than 10 microns.",
    unit: "µg/m³",
  },
  ozone: {
    label: "Ozone (O3)",
    description: "Ground-level ozone concentration.",
    unit: "µg/m³",
  },
  nitrogen_dioxide: {
    label: "Nitrogen dioxide (NO2)",
    description: "Traffic and combustion pollutant concentration.",
    unit: "µg/m³",
  },
  sulphur_dioxide: {
    label: "Sulphur dioxide (SO2)",
    description: "Industrial and fuel-burning pollutant concentration.",
    unit: "µg/m³",
  },
  carbon_monoxide: {
    label: "Carbon monoxide (CO)",
    description: "Combustion gas concentration in the air.",
    unit: "µg/m³",
  },
  hour_sin: {
    label: "Time of day (cycle)",
    description: "Smooth 24-hour clock signal capturing morning vs. evening.",
  },
  hour_cos: {
    label: "Time of day (phase)",
    description: "Smooth 24-hour clock signal capturing midday vs. midnight.",
  },
  day_of_week_sin: {
    label: "Day of week (cycle)",
    description: "Weekly cycle capturing weekday vs. weekend patterns.",
  },
  day_of_week_cos: {
    label: "Day of week (phase)",
    description: "Weekly cycle phase capturing mid-week vs. weekend shifts.",
  },
  month_sin: {
    label: "Season of year (cycle)",
    description: "Annual seasonal cycle capturing spring vs. autumn transitions.",
  },
  month_cos: {
    label: "Season of year (phase)",
    description: "Annual seasonal phase capturing summer vs. winter.",
  },
  is_weekend: {
    label: "Weekend",
    description: "Indicates Saturday or Sunday (typically lower commuter emissions).",
  },
  is_lahore_smog_season: {
    label: "Smog season",
    description: "Indicates Lahore's winter smog months (Oct–Feb).",
  },
  is_monsoon_season: {
    label: "Monsoon season",
    description: "Indicates Lahore's monsoon months (Jul–Sep).",
  },
  source_disagreement_temp: {
    label: "Source temperature spread",
    description: "Temperature difference between Open-Meteo and Met.no readings.",
    unit: "°C",
  },
  source_count: {
    label: "Active weather sources",
    description: "Number of live weather providers reporting for this hour.",
  },
  is_dual_source: {
    label: "Dual-source verified",
    description: "Whether both Open-Meteo and Met.no reported readings this hour.",
  },
  temp_apparent_delta: {
    label: "Feels-like gap",
    description: "Difference between feels-like and actual air temperature.",
    unit: "°C",
  },
  heat_stress_index: {
    label: "Heat-humidity index",
    description: "Combined heat and humidity discomfort indicator.",
  },
  pm25_pm10_ratio: {
    label: "Fine-to-coarse dust ratio",
    description: "Share of fine smoke particles (PM2.5) relative to coarse dust (PM10).",
  },
  aqi_wind_dispersion: {
    label: "Wind air-clearing factor",
    description: "Air pollution adjusted for wind speed (calm winds trap smog).",
  },
  temp_diurnal_anomaly: {
    label: "Temp vs. 24h average",
    description: "How much warmer or cooler it is now compared to the past 24h average.",
    unit: "°C",
  },
  aqi_diurnal_anomaly: {
    label: "Air quality vs. 24h average",
    description: "How much better or worse air quality is now compared to the past 24h average.",
    unit: "AQI",
  },
}

const KNOWN_FEATURE_LABELS: Record<string, string> = Object.fromEntries(
  Object.entries(KNOWN_FEATURE_DEFS).map(([k, v]) => [k, v.label])
)

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

export interface FeatureMeta {
  label: string
  description: string
  unit: string
  raw: string
}

function lowerPhrasePreservingAcronyms(label: string): string {
  // Keep acronyms like UV, PM2.5, PM10, AQI, O3, NO2, SO2, CO, N/S, E/W uppercase.
  return label.replace(/^[A-Z](?=[a-z])/, (ch) => ch.toLowerCase())
}

function resolveBaseDef(baseKey: string): BaseFeatureDef {
  const exact = KNOWN_FEATURE_DEFS[baseKey]
  if (exact) return exact
  const label = humanizeIdentifier(baseKey)
  return {
    label,
    description: `Observed ${lowerPhrasePreservingAcronyms(label)} signal.`,
  }
}

function formatRollStatMeta(
  baseKey: string,
  hours: string,
  stat: string,
  raw: string
): FeatureMeta {
  const base = resolveBaseDef(baseKey)
  const phrase = lowerPhrasePreservingAcronyms(base.label)
  const unit = base.unit ?? ""

  if (stat === "mean") {
    return {
      label: `${hours}h avg ${phrase}`,
      description: `Average ${phrase} over the past ${hours} hours.`,
      unit,
      raw,
    }
  }
  if (stat === "max") {
    return {
      label: `${hours}h peak ${phrase}`,
      description: `Highest ${phrase} recorded over the past ${hours} hours.`,
      unit,
      raw,
    }
  }
  if (stat === "min") {
    return {
      label: `${hours}h lowest ${phrase}`,
      description: `Lowest ${phrase} recorded over the past ${hours} hours.`,
      unit,
      raw,
    }
  }
  if (stat === "std") {
    return {
      label: `${hours}h ${phrase} variation`,
      description: `How much ${phrase} fluctuated over the past ${hours} hours (rolling std).`,
      unit,
      raw,
    }
  }
  return {
    label: `${hours}h ${phrase} (${stat})`,
    description: `${hours}-hour rolling ${stat} of ${phrase}.`,
    unit,
    raw,
  }
}

export function describeFeature(feature: string | null | undefined): FeatureMeta {
  if (!feature) {
    return { label: "—", description: "", unit: "", raw: "" }
  }
  const trimmed = feature.trim()
  if (!trimmed) {
    return { label: "—", description: "", unit: "", raw: "" }
  }
  const lower = trimmed.toLowerCase()

  // 1. Exact base / derived feature match
  if (lower in KNOWN_FEATURE_DEFS) {
    const def = KNOWN_FEATURE_DEFS[lower]
    return {
      label: def.label,
      description: def.description,
      unit: def.unit ?? "",
      raw: trimmed,
    }
  }

  // 2. Actual SkyCast rolling pattern: `{col}_roll_{w}h_{mean|std|min|max}`
  const metricRoll = lower.match(/^(.+)_roll_(\d+)h_(mean|std|min|max)$/)
  if (metricRoll) {
    return formatRollStatMeta(metricRoll[1], metricRoll[2], metricRoll[3], trimmed)
  }

  // 3. Legacy rolling pattern: `{col}_rolling_{mean|std|min|max}_{w}h`
  const metricRollingLegacy = lower.match(/^(.+)_rolling_(mean|std|min|max)_(\d+)h$/)
  if (metricRollingLegacy) {
    return formatRollStatMeta(
      metricRollingLegacy[1],
      metricRollingLegacy[3],
      metricRollingLegacy[2],
      trimmed
    )
  }

  // 4. Metric lag pattern: `{col}_lag_{h}h`
  const metricLag = lower.match(/^(.+)_lag_(\d+)h$/)
  if (metricLag) {
    const base = resolveBaseDef(metricLag[1])
    const hrs = Number(metricLag[2])
    const phrase = lowerPhrasePreservingAcronyms(base.label)
    return {
      label: `${base.label} (${hrs}h ago)`,
      description:
        hrs === 1
          ? `${base.label} recorded 1 hour earlier.`
          : `${base.label} recorded ${hrs} hours earlier (${phrase}).`,
      unit: base.unit ?? "",
      raw: trimmed,
    }
  }

  // 5. Metric rate-of-change pattern: `{col}_roc_{k}h`
  const metricRoc = lower.match(/^(.+)_roc_(\d+)h$/)
  if (metricRoc) {
    const base = resolveBaseDef(metricRoc[1])
    const hrs = Number(metricRoc[2])
    const phrase = lowerPhrasePreservingAcronyms(base.label)
    return {
      label: `${hrs}h ${phrase} change`,
      description: `How much ${phrase} rose or fell over the past ${hrs} ${hrs === 1 ? "hour" : "hours"}.`,
      unit: base.unit ?? "",
      raw: trimmed,
    }
  }

  // 6. Standalone shorthand patterns (`lag_1h`, `rolling_mean_24h`, `roc_3h`)
  const lagOnly = lower.match(/^lag_(\d+)h$/)
  if (lagOnly) {
    const hrs = Number(lagOnly[1])
    return {
      label: hrs === 1 ? "Previous hour value" : `Value ${hrs}h earlier`,
      description: `Reading recorded ${hrs} ${hrs === 1 ? "hour" : "hours"} earlier.`,
      unit: "",
      raw: trimmed,
    }
  }

  const rocOnly = lower.match(/^roc_(\d+)h$/)
  if (rocOnly) {
    const hrs = Number(rocOnly[1])
    return {
      label: `${hrs}h recent change`,
      description: `How much the reading changed over the past ${hrs} ${hrs === 1 ? "hour" : "hours"}.`,
      unit: "",
      raw: trimmed,
    }
  }

  const fallback = humanizeIdentifier(trimmed)
  return {
    label: fallback,
    description: `Engineered weather model signal (${trimmed}).`,
    unit: "",
    raw: trimmed,
  }
}

export function formatFeatureName(feature: string | null | undefined): string {
  return describeFeature(feature).label
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

function formatRuleNumber(raw: string): string {
  const num = Number(raw)
  if (!Number.isFinite(num)) {
    return raw
  }
  if (Math.abs(num) >= 100) {
    return num.toFixed(0)
  }
  const fixed = Math.abs(num) >= 10 ? num.toFixed(1) : num.toFixed(2)
  return fixed.replace(/\.0+$/, "").replace(/(\.\d*[1-9])0+$/, "$1")
}

/**
 * Converts a raw LIME threshold rule (e.g. `18.40 < temperature_2m_lag_1h <= 25.10`)
 * into a plain-English phrase readable by non-technical reviewers.
 */
export function humanizeLimeRule(rule: string | null | undefined): string {
  if (!rule) {
    return "—"
  }
  const trimmed = rule.trim()
  if (!trimmed) {
    return "—"
  }

  // Range: `low < feature <= high` (or <=, <)
  const rangeMatch = trimmed.match(
    /^(-?\d+(?:\.\d+)?)\s*(<=?)\s*([a-zA-Z][a-zA-Z0-9_]*)\s*(<=?)\s*(-?\d+(?:\.\d+)?)$/
  )
  if (rangeMatch) {
    const [, low, , feat, , high] = rangeMatch
    return `${formatFeatureName(feat)} is between ${formatRuleNumber(low)} and ${formatRuleNumber(high)}`
  }

  // Single comparison: `feature <= value` / `feature > value` / etc.
  const singleMatch = trimmed.match(
    /^([a-zA-Z][a-zA-Z0-9_]*)\s*(<=|>=|<|>|==|=)\s*(-?\d+(?:\.\d+)?)$/
  )
  if (singleMatch) {
    const [, feat, op, val] = singleMatch
    const name = formatFeatureName(feat)
    const num = formatRuleNumber(val)
    if (op === "<=") return `${name} is ${num} or lower`
    if (op === "<") return `${name} is below ${num}`
    if (op === ">=") return `${name} is ${num} or higher`
    if (op === ">") return `${name} is above ${num}`
    return `${name} is ${num}`
  }

  // Reversed single comparison: `value < feature`
  const reversedMatch = trimmed.match(
    /^(-?\d+(?:\.\d+)?)\s*(<=|>=|<|>)\s*([a-zA-Z][a-zA-Z0-9_]*)$/
  )
  if (reversedMatch) {
    const [, val, op, feat] = reversedMatch
    const name = formatFeatureName(feat)
    const num = formatRuleNumber(val)
    if (op === "<=") return `${name} is ${num} or higher`
    if (op === "<") return `${name} is above ${num}`
    if (op === ">=") return `${name} is ${num} or lower`
    return `${name} is below ${num}`
  }

  return formatLimeRule(trimmed)
}

export type ComparisonModelKey =
  | "sarimax"
  | "random_forest"
  | "ridge"
  | "lstm"
  | "naive_24h"

export function mapModelFamilyToSeriesKey(
  modelFamily: string | null | undefined
): ComparisonModelKey | null {
  if (!modelFamily) return null
  const normalized = modelFamily.trim().toLowerCase()
  if (normalized === "sarimax") return "sarimax"
  if (
    normalized === "random_forest" ||
    normalized === "randomforest" ||
    normalized === "rf"
  ) {
    return "random_forest"
  }
  if (normalized === "ridge") return "ridge"
  if (normalized === "lstm") return "lstm"
  if (
    normalized === "naive_24h" ||
    normalized === "naive24h" ||
    normalized === "naive"
  ) {
    return "naive_24h"
  }
  return null
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
