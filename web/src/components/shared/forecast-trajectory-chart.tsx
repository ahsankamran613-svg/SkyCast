import { useMemo, type ReactNode } from "react"
import { Inbox } from "lucide-react"
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"

import { ChartCard, type ChartTableColumn } from "@/components/shared/chart-card"
import {
  ChartLegendBar,
  SkyCastChartTooltipCard,
  computeChartDomain,
  formatValueWithUnit,
  getAqiReferenceBands,
  useFirstLoadChartAnimation,
  type ChartLegendItem,
  type TooltipEntryRow,
} from "@/components/shared/chart-primitives"
import { EmptyState } from "@/components/shared/empty-state"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import type {
  ForecastPoint,
  MetricSpecsMap,
  WeatherObservation,
} from "@/lib/api"
import { formatLocalTime, parseIsoDate } from "@/lib/time"
import {
  formatMetricValue,
  formatModelLabel,
  getMetricColorVar,
  getMetricSpec,
  readObservationNumeric,
} from "@/lib/weather"

export interface TrajectoryRow extends Record<string, unknown> {
  timestamp_utc: string
  observed: number | null
  forecast: number | null
  lower_bound: number | null
  upper_bound: number | null
  band: [number, number] | null
  model_family: string | null
  horizon_hour: number | null
}

export interface ForecastTrajectoryChartProps {
  metric: string
  metricSpecs?: MetricSpecsMap | null
  observed: WeatherObservation[]
  forecasts: ForecastPoint[]
  lookbackHours: number
  horizonHours?: number
  timeZone?: string | null
  title?: string
  subtitle?: string
  controls?: ReactNode
  subHeader?: ReactNode
  chartHeightClass?: string
  emptyMessage?: string
  showModelMeta?: boolean
  className?: string
}

function normalizeIso(ts: string | null | undefined): string | null {
  const parsed = parseIsoDate(ts)
  return parsed ? parsed.toISOString() : null
}

function formatAxisTick(value: unknown): string {
  const num = Number(value)
  if (!Number.isFinite(num)) {
    return ""
  }
  return Number.isInteger(num) ? String(num) : num.toFixed(1)
}

export function ForecastTrajectoryChart({
  metric,
  metricSpecs,
  observed,
  forecasts,
  lookbackHours,
  horizonHours,
  timeZone,
  title,
  subtitle,
  controls,
  subHeader,
  chartHeightClass = "h-[240px] w-full aspect-auto md:h-[320px]",
  emptyMessage = "No recent observations or forecast points are available for this metric yet.",
  showModelMeta = false,
  className,
}: ForecastTrajectoryChartProps) {
  const spec = getMetricSpec(metric, metricSpecs)
  const colorVar = getMetricColorVar(metric, metricSpecs)
  const unit = spec?.unit ?? ""
  const metricLabel = spec?.label ?? metric

  const { rows, nowTimestampUtc, latestIssuedAtUtc, activeModel } = useMemo(() => {
    const byTimestamp = new Map<string, TrajectoryRow>()

    const sortedObserved = [...observed]
      .filter((obs) => Boolean(parseIsoDate(obs.timestamp_utc)))
      .sort(
        (a, b) =>
          (parseIsoDate(a.timestamp_utc)?.getTime() ?? 0) -
          (parseIsoDate(b.timestamp_utc)?.getTime() ?? 0)
      )

    const slicedObserved = sortedObserved.slice(-Math.max(1, lookbackHours))
    let lastObsIso: string | null = null

    for (const obs of slicedObserved) {
      const iso = normalizeIso(obs.timestamp_utc)
      if (!iso) {
        continue
      }
      lastObsIso = iso
      byTimestamp.set(iso, {
        timestamp_utc: iso,
        observed: readObservationNumeric(obs, metric),
        forecast: null,
        lower_bound: null,
        upper_bound: null,
        band: null,
        model_family: null,
        horizon_hour: null,
      })
    }

    const sortedForecasts = [...forecasts]
      .filter((pt) => Boolean(parseIsoDate(pt.target_timestamp_utc)))
      .sort(
        (a, b) =>
          (parseIsoDate(a.target_timestamp_utc)?.getTime() ?? 0) -
          (parseIsoDate(b.target_timestamp_utc)?.getTime() ?? 0)
      )

    let latestIssued: string | null = null
    let modelName: string | null = null

    for (const pt of sortedForecasts) {
      const iso = normalizeIso(pt.target_timestamp_utc)
      if (!iso) {
        continue
      }
      if (pt.issued_at_utc && (!latestIssued || pt.issued_at_utc > latestIssued)) {
        latestIssued = pt.issued_at_utc
      }
      if (!modelName && pt.model_family) {
        modelName = pt.model_family
      }

      const forecastVal =
        typeof pt.predicted_value === "number" && Number.isFinite(pt.predicted_value)
          ? pt.predicted_value
          : null
      const lowVal =
        typeof pt.lower_bound === "number" && Number.isFinite(pt.lower_bound)
          ? pt.lower_bound
          : null
      const highVal =
        typeof pt.upper_bound === "number" && Number.isFinite(pt.upper_bound)
          ? pt.upper_bound
          : null
      const bandTuple: [number, number] | null =
        lowVal !== null && highVal !== null
          ? [Math.min(lowVal, highVal), Math.max(lowVal, highVal)]
          : null

      const existing = byTimestamp.get(iso)
      if (existing) {
        existing.forecast = forecastVal
        existing.lower_bound = bandTuple ? bandTuple[0] : null
        existing.upper_bound = bandTuple ? bandTuple[1] : null
        existing.band = bandTuple
        existing.model_family = pt.model_family ?? null
        existing.horizon_hour =
          typeof pt.horizon_hour === "number" ? pt.horizon_hour : null
      } else {
        byTimestamp.set(iso, {
          timestamp_utc: iso,
          observed: null,
          forecast: forecastVal,
          lower_bound: bandTuple ? bandTuple[0] : null,
          upper_bound: bandTuple ? bandTuple[1] : null,
          band: bandTuple,
          model_family: pt.model_family ?? null,
          horizon_hour:
            typeof pt.horizon_hour === "number" ? pt.horizon_hour : null,
        })
      }
    }

    const merged = Array.from(byTimestamp.values()).sort(
      (a, b) => Date.parse(a.timestamp_utc) - Date.parse(b.timestamp_utc)
    )

    return {
      rows: merged,
      nowTimestampUtc: lastObsIso,
      latestIssuedAtUtc: latestIssued,
      activeModel: modelName,
    }
  }, [observed, forecasts, lookbackHours, metric])

  const hasNumericValues = useMemo(
    () =>
      rows.some(
        (row) =>
          row.observed !== null ||
          row.forecast !== null ||
          row.lower_bound !== null ||
          row.upper_bound !== null
      ),
    [rows]
  )

  const hasBand = useMemo(() => rows.some((row) => row.band !== null), [rows])
  const shouldAnimate = useFirstLoadChartAnimation(hasNumericValues)

  const [domainMin, domainMax] = useMemo(() => {
    const allNumbers: Array<number | null> = []
    for (const row of rows) {
      allNumbers.push(row.observed, row.forecast, row.lower_bound, row.upper_bound)
    }
    return computeChartDomain(allNumbers, spec)
  }, [rows, spec])

  const aqiBands = useMemo(() => {
    if (metric !== "us_aqi") {
      return []
    }
    return getAqiReferenceBands(domainMin, domainMax)
  }, [metric, domainMin, domainMax])

  const chartConfig = useMemo(
    () => ({
      observed: {
        label: "Observed",
        color: colorVar,
      },
      forecast: {
        label: "Forecast",
        color: colorVar,
      },
      band: {
        label: "Confidence band",
        color: colorVar,
      },
    }),
    [colorVar]
  )

  const legendItems = useMemo<ChartLegendItem[]>(() => {
    const items: ChartLegendItem[] = [
      {
        key: "observed",
        label: `Observed (${lookbackHours}h)`,
        color: colorVar,
        variant: "solid",
      },
      {
        key: "forecast",
        label: horizonHours ? `Forecast (${horizonHours}h)` : "Forecast",
        color: colorVar,
        variant: "dashed",
      },
    ]
    if (hasBand) {
      items.push({
        key: "band",
        label: "Confidence interval",
        color: colorVar,
        variant: "band",
      })
    }
    return items
  }, [lookbackHours, horizonHours, colorVar, hasBand])

  const tableColumns = useMemo<ChartTableColumn<TrajectoryRow>[]>(() => {
    const cols: ChartTableColumn<TrajectoryRow>[] = [
      {
        key: "timestamp_utc",
        header: "Time",
        render: (row) => (
          <TimestampPkt
            timestamp={row.timestamp_utc}
            timeZone={timeZone}
            mode="datetime"
          />
        ),
      },
      {
        key: "observed",
        header: `Observed${unit ? ` (${unit})` : ""}`,
        align: "right",
        render: (row) => formatMetricValue(row.observed, 1),
      },
      {
        key: "forecast",
        header: `Forecast${unit ? ` (${unit})` : ""}`,
        align: "right",
        render: (row) => formatMetricValue(row.forecast, 1),
      },
      {
        key: "band",
        header: `Interval${unit ? ` (${unit})` : ""}`,
        align: "right",
        render: (row) =>
          row.lower_bound !== null && row.upper_bound !== null
            ? `${formatMetricValue(row.lower_bound, 1)} – ${formatMetricValue(
                row.upper_bound,
                1
              )}`
            : "—",
      },
    ]
    if (showModelMeta) {
      cols.push({
        key: "model_family",
        header: "Model",
        render: (row) => formatModelLabel(row.model_family),
      })
    }
    return cols
  }, [timeZone, unit, showModelMeta])

  const resolvedTitle = title ?? `${metricLabel} outlook`
  const resolvedSubtitle =
    subtitle ??
    `Last ${lookbackHours}h observed and ${horizonHours ?? 24}h forecast trajectory`

  const footerContent =
    latestIssuedAtUtc || (showModelMeta && activeModel) ? (
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span>
          {latestIssuedAtUtc ? (
            <>
              Issued{" "}
              <TimestampPkt
                timestamp={latestIssuedAtUtc}
                timeZone={timeZone}
                mode="datetime"
                className="font-medium text-foreground"
              />
            </>
          ) : null}
        </span>
        {showModelMeta && activeModel ? (
          <span>
            Active model:{" "}
            <span className="font-medium text-foreground">
              {formatModelLabel(activeModel)}
            </span>
          </span>
        ) : null}
      </div>
    ) : undefined

  return (
    <ChartCard
      title={resolvedTitle}
      subtitle={resolvedSubtitle}
      ariaLabel={`${metricLabel} observed and forecast trajectory chart`}
      controls={controls}
      subHeader={subHeader}
      tableColumns={tableColumns}
      tableRows={rows}
      emptyTableMessage={emptyMessage}
      footer={footerContent}
      className={className}
    >
      {!hasNumericValues ? (
        <EmptyState
          icon={Inbox}
          title={`No ${metricLabel.toLowerCase()} trajectory yet`}
          description={emptyMessage}
          className="border-0 bg-muted shadow-none"
        />
      ) : (
        <div>
          <ChartContainer config={chartConfig} className={chartHeightClass}>
            <ComposedChart
              data={rows}
              margin={{ top: 18, right: 12, bottom: 4, left: 0 }}
            >
              {aqiBands.map((band) => (
                <ReferenceArea
                  key={band.key}
                  y1={band.y1}
                  y2={band.y2}
                  fill={band.fill}
                  fillOpacity={0.08}
                  ifOverflow="hidden"
                />
              ))}

              <CartesianGrid vertical={false} stroke="var(--border)" />

              <XAxis
                dataKey="timestamp_utc"
                tickFormatter={(iso) => formatLocalTime(String(iso), timeZone)}
                tickLine={false}
                axisLine={false}
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                minTickGap={28}
              />

              <YAxis
                domain={[domainMin, domainMax]}
                tickFormatter={formatAxisTick}
                tickLine={false}
                axisLine={false}
                width={44}
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              />

              {nowTimestampUtc ? (
                <ReferenceLine
                  x={nowTimestampUtc}
                  stroke="var(--foreground)"
                  strokeOpacity={0.4}
                  strokeWidth={1}
                  label={{
                    value: "Now",
                    position: "top",
                    fill: "var(--muted-foreground)",
                    fontSize: 12,
                  }}
                />
              ) : null}

              <ChartTooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) {
                    return null
                  }
                  const row = payload[0]?.payload as TrajectoryRow | undefined
                  if (!row) {
                    return null
                  }

                  const tooltipRows: TooltipEntryRow[] = []
                  if (row.observed !== null) {
                    tooltipRows.push({
                      key: "observed",
                      label: "Observed",
                      valueText: formatValueWithUnit(row.observed, unit, 1),
                      color: colorVar,
                    })
                  }
                  if (row.forecast !== null) {
                    tooltipRows.push({
                      key: "forecast",
                      label:
                        row.horizon_hour !== null
                          ? `Forecast (+${row.horizon_hour}h)`
                          : "Forecast",
                      valueText: formatValueWithUnit(row.forecast, unit, 1),
                      color: colorVar,
                      dashed: true,
                    })
                  }
                  if (row.lower_bound !== null && row.upper_bound !== null) {
                    tooltipRows.push({
                      key: "band",
                      label: "Interval",
                      valueText: `${formatMetricValue(
                        row.lower_bound,
                        1
                      )} – ${formatMetricValue(row.upper_bound, 1)}${
                        unit ? ` ${unit}` : ""
                      }`,
                      color: colorVar,
                      band: true,
                    })
                  }

                  const primaryValue = row.observed ?? row.forecast
                  const footerNote =
                    showModelMeta && row.model_family
                      ? `Model: ${formatModelLabel(row.model_family)}`
                      : null

                  return (
                    <SkyCastChartTooltipCard
                      timestampUtc={row.timestamp_utc}
                      timeZone={timeZone}
                      rows={tooltipRows}
                      aqiValue={metric === "us_aqi" ? primaryValue : null}
                      footerNote={footerNote}
                    />
                  )
                }}
              />

              {hasBand ? (
                <Area
                  type="monotone"
                  dataKey="band"
                  stroke="none"
                  fill={colorVar}
                  fillOpacity={0.15}
                  isAnimationActive={false}
                  activeDot={false}
                  connectNulls={false}
                />
              ) : null}

              <Line
                type="monotone"
                dataKey="observed"
                stroke={colorVar}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, strokeWidth: 2 }}
                connectNulls={false}
                isAnimationActive={shouldAnimate}
                animationDuration={600}
                animationEasing="ease-out"
              />

              <Line
                type="monotone"
                dataKey="forecast"
                stroke={colorVar}
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={false}
                activeDot={{ r: 5, strokeWidth: 2 }}
                connectNulls={false}
                isAnimationActive={shouldAnimate}
                animationDuration={600}
                animationEasing="ease-out"
              />
            </ComposedChart>
          </ChartContainer>

          <ChartLegendBar items={legendItems} />
        </div>
      )}
    </ChartCard>
  )
}
