import {
  CircleAlert,
  CircleCheck,
  Flame,
  Info,
  OctagonAlert,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react"

export type AqiLevelKey =
  | "good"
  | "moderate"
  | "sensitive"
  | "unhealthy"
  | "very"
  | "hazardous"

export interface AqiLevelMeta {
  key: AqiLevelKey
  min: number
  max: number
  label: string
  shortLabel: string
  badge: string
  textClass: string
  bgClass: string
  borderClass: string
  fill: string
  icon: LucideIcon
}

export const AQI_LEVELS: readonly AqiLevelMeta[] = [
  {
    key: "good",
    min: 0,
    max: 50,
    label: "Good",
    shortLabel: "Good",
    badge: "bg-aqi-good-bg text-aqi-good-fg border-aqi-good",
    textClass: "text-aqi-good-fg",
    bgClass: "bg-aqi-good-bg",
    borderClass: "border-aqi-good",
    fill: "var(--aqi-good)",
    icon: CircleCheck,
  },
  {
    key: "moderate",
    min: 51,
    max: 100,
    label: "Moderate",
    shortLabel: "Moderate",
    badge: "bg-aqi-moderate-bg text-aqi-moderate-fg border-aqi-moderate",
    textClass: "text-aqi-moderate-fg",
    bgClass: "bg-aqi-moderate-bg",
    borderClass: "border-aqi-moderate",
    fill: "var(--aqi-moderate)",
    icon: Info,
  },
  {
    key: "sensitive",
    min: 101,
    max: 150,
    label: "Unhealthy for sensitive groups",
    shortLabel: "Sensitive",
    badge: "bg-aqi-sensitive-bg text-aqi-sensitive-fg border-aqi-sensitive",
    textClass: "text-aqi-sensitive-fg",
    bgClass: "bg-aqi-sensitive-bg",
    borderClass: "border-aqi-sensitive",
    fill: "var(--aqi-sensitive)",
    icon: CircleAlert,
  },
  {
    key: "unhealthy",
    min: 151,
    max: 200,
    label: "Unhealthy",
    shortLabel: "Unhealthy",
    badge: "bg-aqi-unhealthy-bg text-aqi-unhealthy-fg border-aqi-unhealthy",
    textClass: "text-aqi-unhealthy-fg",
    bgClass: "bg-aqi-unhealthy-bg",
    borderClass: "border-aqi-unhealthy",
    fill: "var(--aqi-unhealthy)",
    icon: TriangleAlert,
  },
  {
    key: "very",
    min: 201,
    max: 300,
    label: "Very unhealthy",
    shortLabel: "Very unhealthy",
    badge: "bg-aqi-very-bg text-aqi-very-fg border-aqi-very",
    textClass: "text-aqi-very-fg",
    bgClass: "bg-aqi-very-bg",
    borderClass: "border-aqi-very",
    fill: "var(--aqi-very)",
    icon: OctagonAlert,
  },
  {
    key: "hazardous",
    min: 301,
    max: Infinity,
    label: "Hazardous",
    shortLabel: "Hazardous",
    badge: "bg-aqi-hazardous-bg text-aqi-hazardous-fg border-aqi-hazardous",
    textClass: "text-aqi-hazardous-fg",
    bgClass: "bg-aqi-hazardous-bg",
    borderClass: "border-aqi-hazardous",
    fill: "var(--aqi-hazardous)",
    icon: Flame,
  },
] as const

export function aqiLevel(value: number): AqiLevelMeta {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0
  return (
    AQI_LEVELS.find((level) => safe <= level.max) ??
    AQI_LEVELS[AQI_LEVELS.length - 1]
  )
}

export function getAqiLevel(
  value: number | null | undefined
): AqiLevelMeta | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null
  }
  return aqiLevel(value)
}

const AQI_BAND_UPPER_BOUNDS = [50, 100, 150, 200, 300, 500] as const

/**
 * Maps an EPA US AQI reading into `[0, 1]` across 6 equal visual gauge segments
 * corresponding to the 6 AQI categories so Good (0-50) and Hazardous (301-500)
 * remain proportionally legible along the semicircle.
 */
export function aqiGaugeRatio(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0
  }
  const clamped = Math.max(0, Math.min(500, value))
  const segmentSize = 1 / AQI_BAND_UPPER_BOUNDS.length
  let lower = 0

  for (let idx = 0; idx < AQI_BAND_UPPER_BOUNDS.length; idx++) {
    const upper = AQI_BAND_UPPER_BOUNDS[idx]
    if (clamped <= upper) {
      const localFraction = (clamped - lower) / (upper - lower)
      return Math.min(1, Math.max(0, (idx + localFraction) * segmentSize))
    }
    lower = upper
  }

  return 1
}
