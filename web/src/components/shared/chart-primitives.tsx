import { useEffect, useRef, useState } from "react"
import { cn } from "cn"

import type { MetricSpec, MetricSpecsMap } from "@/lib/api"
import { AQI_LEVELS, getAqiLevel } from "@/lib/aqi"
import { formatLocalDateTime } from "@/lib/time"
import { formatMetricValue, getMetricColorVar, getMetricSpec } from "@/lib/weather"

export interface AqiReferenceBand {
  key: string
  label: string
  fill: string
  y1: number
  y2: number
}

const CONTINUOUS_AQI_BOUNDS: ReadonlyArray<{
  key: string
  low: number
  high: number
}> = [
  { key: "good", low: 0, high: 50 },
  { key: "moderate", low: 50, high: 100 },
  { key: "sensitive", low: 100, high: 150 },
  { key: "unhealthy", low: 150, high: 200 },
  { key: "very", low: 200, high: 300 },
  { key: "hazardous", low: 300, high: 500 },
]

export function useFirstLoadChartAnimation(hasData: boolean): boolean {
  const [shouldAnimate, setShouldAnimate] = useState<boolean>(() => {
    if (typeof window === "undefined") {
      return false
    }
    return !window.matchMedia("(prefers-reduced-motion: reduce)").matches
  })
  const completedRef = useRef(false)

  useEffect(() => {
    if (!hasData || !shouldAnimate || completedRef.current) {
      return
    }
    const timer = window.setTimeout(() => {
      completedRef.current = true
      setShouldAnimate(false)
    }, 650)
    return () => window.clearTimeout(timer)
  }, [hasData, shouldAnimate])

  return shouldAnimate
}

export function computeChartDomain(
  values: Array<number | null | undefined>,
  spec?: MetricSpec
): [number, number] {
  const finite = values.filter(
    (v): v is number => typeof v === "number" && Number.isFinite(v)
  )

  if (finite.length === 0) {
    return [0, 1]
  }

  const rawMin = Math.min(...finite)
  const rawMax = Math.max(...finite)
  const nonNegativeMetric = spec?.min_value === 0 && rawMin >= 0

  if (rawMin === rawMax) {
    if (nonNegativeMetric && rawMin === 0) {
      return [0, 1]
    }
    const pad = Math.max(1, Math.abs(rawMin) * 0.1)
    const minBound = nonNegativeMetric ? Math.max(0, rawMin - pad) : rawMin - pad
    return [Number(minBound.toFixed(2)), Number((rawMax + pad).toFixed(2))]
  }

  const span = rawMax - rawMin
  const pad = span * 0.1
  let domainMin = rawMin - pad
  let domainMax = rawMax + pad

  if (nonNegativeMetric) {
    domainMin = Math.max(0, domainMin)
  }
  if (spec?.max_value === 100 && rawMax <= 100) {
    domainMax = Math.min(100, domainMax)
  }

  return [Number(domainMin.toFixed(2)), Number(domainMax.toFixed(2))]
}

export function getAqiReferenceBands(
  domainMin: number,
  domainMax: number
): AqiReferenceBand[] {
  if (!Number.isFinite(domainMin) || !Number.isFinite(domainMax) || domainMax <= domainMin) {
    return []
  }

  const bands: AqiReferenceBand[] = []
  for (const bound of CONTINUOUS_AQI_BOUNDS) {
    const level = AQI_LEVELS.find((item) => item.key === bound.key)
    if (!level) {
      continue
    }
    const upperBound =
      bound.key === "hazardous" ? Math.max(bound.high, domainMax) : bound.high
    const y1 = Math.max(bound.low, domainMin)
    const y2 = Math.min(upperBound, domainMax)
    if (y2 > y1) {
      bands.push({
        key: level.key,
        label: level.label,
        fill: level.fill,
        y1,
        y2,
      })
    }
  }
  return bands
}

export interface MetricSelectorTabsProps {
  metrics: string[]
  selectedMetric: string
  onSelectMetric: (metric: string) => void
  metricSpecs?: MetricSpecsMap | null
  ariaLabel?: string
  className?: string
}

export function MetricSelectorTabs({
  metrics,
  selectedMetric,
  onSelectMetric,
  metricSpecs,
  ariaLabel = "Select weather metric",
  className,
}: MetricSelectorTabsProps) {
  if (metrics.length === 0) {
    return null
  }

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "flex items-center gap-1 overflow-x-auto rounded-lg bg-muted p-1",
        className
      )}
    >
      {metrics.map((metricKey) => {
        const spec = getMetricSpec(metricKey, metricSpecs)
        const colorVar = getMetricColorVar(metricKey, metricSpecs)
        const isActive = metricKey === selectedMetric

        return (
          <button
            key={metricKey}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onSelectMetric(metricKey)}
            className={cn(
              "inline-flex min-h-9 shrink-0 items-center gap-2 rounded-md px-3 py-1.5 text-sm whitespace-nowrap transition-colors duration-120",
              isActive
                ? "bg-card font-semibold text-foreground shadow-(--shadow-card)"
                : "font-medium text-foreground/75 hover:text-foreground"
            )}
          >
            <span
              className="size-2 shrink-0 rounded-full"
              style={{ backgroundColor: colorVar }}
              aria-hidden="true"
            />
            <span>{spec?.label ?? metricKey}</span>
          </button>
        )
      })}
    </div>
  )
}

export interface ChartLegendItem {
  key: string
  label: string
  color: string
  variant: "solid" | "dashed" | "dotted" | "band"
}

export function ChartLegendBar({
  items,
  className,
}: {
  items: ChartLegendItem[]
  className?: string
}) {
  if (items.length === 0) {
    return null
  }

  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-center gap-x-5 gap-y-1.5 pt-3 text-xs text-muted-foreground",
        className
      )}
    >
      {items.map((item) => (
        <div key={item.key} className="inline-flex items-center gap-2">
          {item.variant === "band" ? (
            <span
              className="h-2 w-4 shrink-0 rounded-full"
              style={{
                backgroundColor: item.color,
                opacity: 0.25,
                border: `1px solid ${item.color}`,
              }}
              aria-hidden="true"
            />
          ) : item.variant === "dashed" ? (
            <span
              className="h-0 w-4 shrink-0 border-t-2 border-dashed"
              style={{ borderColor: item.color }}
              aria-hidden="true"
            />
          ) : item.variant === "dotted" ? (
            <span
              className="h-0 w-4 shrink-0 border-t-2 border-dotted"
              style={{ borderColor: item.color }}
              aria-hidden="true"
            />
          ) : (
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: item.color }}
              aria-hidden="true"
            />
          )}
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  )
}

export interface TooltipEntryRow {
  key: string
  label: string
  valueText: string
  color: string
  dashed?: boolean
  band?: boolean
}

export function SkyCastChartTooltipCard({
  timestampUtc,
  timeZone,
  rows,
  aqiValue,
  footerNote,
}: {
  timestampUtc?: string | null
  timeZone?: string | null
  rows: TooltipEntryRow[]
  aqiValue?: number | null
  footerNote?: string | null
}) {
  if (rows.length === 0) {
    return null
  }

  const aqiMeta = getAqiLevel(aqiValue)
  const AqiIcon = aqiMeta?.icon

  return (
    <div className="max-w-[280px] rounded-sm border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-(--shadow-popover)">
      {timestampUtc ? (
        <div className="mb-1.5 font-medium text-popover-foreground tabular-nums">
          {formatLocalDateTime(timestampUtc, timeZone)}
        </div>
      ) : null}

      <div className="flex flex-col gap-1">
        {rows.map((row) => (
          <div
            key={row.key}
            className="flex items-center justify-between gap-4 tabular-nums"
          >
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              {row.band ? (
                <span
                  className="h-2 w-3 shrink-0 rounded-full"
                  style={{ backgroundColor: row.color, opacity: 0.35 }}
                  aria-hidden="true"
                />
              ) : row.dashed ? (
                <span
                  className="h-0 w-3 shrink-0 border-t-2 border-dashed"
                  style={{ borderColor: row.color }}
                  aria-hidden="true"
                />
              ) : (
                <span
                  className="size-2 shrink-0 rounded-full"
                  style={{ backgroundColor: row.color }}
                  aria-hidden="true"
                />
              )}
              <span>{row.label}</span>
            </span>
            <span className="font-semibold text-popover-foreground">
              {row.valueText}
            </span>
          </div>
        ))}
      </div>

      {aqiMeta && AqiIcon ? (
        <div className="mt-1.5 flex items-center gap-1.5 border-t border-border pt-1.5">
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
              aqiMeta.badge
            )}
          >
            <AqiIcon className="size-3 shrink-0" strokeWidth={1.75} />
            <span>{aqiMeta.label}</span>
          </span>
        </div>
      ) : null}

      {footerNote ? (
        <div className="mt-1.5 border-t border-border pt-1 text-[11px] text-muted-foreground">
          {footerNote}
        </div>
      ) : null}
    </div>
  )
}

export function formatValueWithUnit(
  value: number | null | undefined,
  unit?: string,
  decimals = 1
): string {
  const formatted = formatMetricValue(value, decimals)
  if (formatted === "—") {
    return formatted
  }
  return unit ? `${formatted} ${unit}` : formatted
}
