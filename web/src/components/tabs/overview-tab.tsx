import { Fragment, useMemo, useState, type ReactNode } from "react"
import { FileText } from "lucide-react"
import { cn } from "cn"

import { AdvisoryCards } from "@/components/shared/advisory-cards"
import { AlertBannerList } from "@/components/shared/alert-banner"
import { AqiGauge } from "@/components/shared/aqi-gauge"
import { MetricSelectorTabs } from "@/components/shared/chart-primitives"
import {
  BootstrappingState,
  EmptyState,
  ErrorState,
} from "@/components/shared/empty-state"
import { ForecastTrajectoryChart } from "@/components/shared/forecast-trajectory-chart"
import { Sparkline } from "@/components/shared/sparkline"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { WindCompass } from "@/components/shared/wind-compass"
import { Badge } from "@/components/ui/badge"
import type {
  DashboardOverviewResponse,
  ReportTriggerReason,
} from "@/lib/api"
import { getAqiLevel } from "@/lib/aqi"
import {
  POLLUTANT_LABELS,
  formatMetricValue,
  formatSourceLabel,
  getMetricColorVar,
  getMetricSpec,
  orderMetricKeys,
  pickDefaultMetric,
  readObservationNumeric,
  resolveWeatherCondition,
} from "@/lib/weather"

export interface OverviewTabProps {
  overview: DashboardOverviewResponse | null
  isLoading: boolean
  isRefreshing: boolean
  errorMessage: string | null
  city: string | null
  timeZone: string | null
  onRefresh: () => void
}

const REPORT_TRIGGER_LABELS: Record<ReportTriggerReason, string> = {
  daily_schedule: "Daily briefing",
  hazard_alert: "Hazard update",
  manual: "Weather update",
  backfill_seed: "Station briefing",
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g)
  return tokens.map((token, idx) => {
    if (token.startsWith("**") && token.endsWith("**") && token.length > 4) {
      return (
        <strong key={idx} className="font-semibold text-foreground">
          {token.slice(2, -2)}
        </strong>
      )
    }
    if (token.startsWith("`") && token.endsWith("`") && token.length > 2) {
      return (
        <code
          key={idx}
          className="rounded-sm bg-muted px-1 py-0.5 text-[0.9em] text-foreground"
        >
          {token.slice(1, -1)}
        </code>
      )
    }
    if (token.startsWith("*") && token.endsWith("*") && token.length > 2) {
      return <em key={idx}>{token.slice(1, -1)}</em>
    }
    return <Fragment key={idx}>{token}</Fragment>
  })
}

function BulletinMarkdown({ markdown }: { markdown: string }) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n")
  const blocks: ReactNode[] = []
  let paragraphLines: string[] = []
  let bulletItems: string[] = []
  let orderedItems: string[] = []
  let keyCounter = 0

  const flushParagraph = () => {
    if (paragraphLines.length === 0) return
    const text = paragraphLines.join(" ").trim()
    paragraphLines = []
    if (!text) return
    blocks.push(
      <p key={`p-${keyCounter++}`} className="text-foreground/95">
        {renderInlineMarkdown(text)}
      </p>
    )
  }

  const flushBullets = () => {
    if (bulletItems.length === 0) return
    const items = bulletItems
    bulletItems = []
    blocks.push(
      <ul
        key={`ul-${keyCounter++}`}
        className="list-disc space-y-1.5 pl-5 text-foreground/95"
      >
        {items.map((item, idx) => (
          <li key={idx}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>
    )
  }

  const flushOrdered = () => {
    if (orderedItems.length === 0) return
    const items = orderedItems
    orderedItems = []
    blocks.push(
      <ol
        key={`ol-${keyCounter++}`}
        className="list-decimal space-y-1.5 pl-5 text-foreground/95"
      >
        {items.map((item, idx) => (
          <li key={idx}>{renderInlineMarkdown(item)}</li>
        ))}
      </ol>
    )
  }

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (!line) {
      flushParagraph()
      flushBullets()
      flushOrdered()
      continue
    }

    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/)
    if (headingMatch) {
      flushParagraph()
      flushBullets()
      flushOrdered()
      blocks.push(
        <h4
          key={`h-${keyCounter++}`}
          className="pt-1 text-sm font-semibold text-foreground"
        >
          {renderInlineMarkdown(headingMatch[2])}
        </h4>
      )
      continue
    }

    const bulletMatch = line.match(/^[-*]\s+(.+)$/)
    if (bulletMatch) {
      flushParagraph()
      flushOrdered()
      bulletItems.push(bulletMatch[1])
      continue
    }

    const orderedMatch = line.match(/^\d+\.\s+(.+)$/)
    if (orderedMatch) {
      flushParagraph()
      flushBullets()
      orderedItems.push(orderedMatch[1])
      continue
    }

    flushBullets()
    flushOrdered()
    paragraphLines.push(line)
  }

  flushParagraph()
  flushBullets()
  flushOrdered()

  return (
    <div className="max-w-[68ch] space-y-3 text-base/6.5 md:text-bulletin">
      {blocks}
    </div>
  )
}

function extractHighlights(rawHighlights: unknown[] | undefined): string[] {
  if (!Array.isArray(rawHighlights)) {
    return []
  }
  const items: string[] = []
  for (const entry of rawHighlights) {
    if (typeof entry === "string" && entry.trim()) {
      items.push(entry.trim())
    } else if (entry && typeof entry === "object") {
      const obj = entry as Record<string, unknown>
      const text =
        (typeof obj.summary === "string" && obj.summary) ||
        (typeof obj.detail === "string" && obj.detail) ||
        (typeof obj.text === "string" && obj.text) ||
        (typeof obj.title === "string" && obj.title) ||
        null
      if (text && text.trim()) {
        items.push(text.trim())
      }
    }
  }
  return items
}

export function OverviewTab({
  overview,
  isLoading,
  isRefreshing,
  errorMessage,
  city,
  timeZone,
  onRefresh,
}: OverviewTabProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null)

  const statCardMetrics = useMemo(() => {
    if (!overview) return []
    const groups = Object.values(overview.core_metric_groups ?? {})
    const primary = groups
      .map((members) => members[0])
      .filter((metric): metric is string => Boolean(metric))
    if (primary.length > 0) {
      return primary
    }
    return Object.keys(overview.metric_specs ?? {}).slice(0, 6)
  }, [overview])

  const chartMetrics = useMemo(() => {
    if (!overview) return []
    const forecastKeys = Object.keys(overview.forecasts_24h ?? {})
    const rawPool = Array.from(
      new Set([
        ...statCardMetrics,
        ...forecastKeys,
        ...Object.keys(overview.metric_specs ?? {}),
      ])
    )
    const relevant =
      forecastKeys.length > 0
        ? rawPool.filter(
            (key) => forecastKeys.includes(key) || statCardMetrics.includes(key)
          )
        : rawPool
    return orderMetricKeys(relevant, overview.core_metric_groups)
  }, [overview, statCardMetrics])

  const activeChartMetric = useMemo(() => {
    if (selectedMetric && chartMetrics.includes(selectedMetric)) {
      return selectedMetric
    }
    return (
      pickDefaultMetric(overview?.metric_specs, chartMetrics) ??
      chartMetrics[0] ??
      null
    )
  }, [selectedMetric, chartMetrics, overview])

  if (isLoading && !overview) {
    return (
      <div className="flex flex-col gap-6 md:gap-8">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="h-[220px] animate-pulse rounded-2xl border border-border bg-muted lg:col-span-7" />
          <div className="h-[220px] animate-pulse rounded-xl border border-border bg-muted lg:col-span-5" />
        </div>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 md:gap-4 xl:grid-cols-6">
          {Array.from({ length: 6 }).map((_, idx) => (
            <div
              key={idx}
              className="h-[132px] animate-pulse rounded-lg border border-border bg-muted"
            />
          ))}
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="h-[380px] animate-pulse rounded-xl border border-border bg-muted lg:col-span-8" />
          <div className="h-[380px] animate-pulse rounded-xl border border-border bg-muted lg:col-span-4" />
        </div>
      </div>
    )
  }

  if (errorMessage && !overview) {
    return (
      <ErrorState
        message={errorMessage}
        onRetry={onRefresh}
        isRetrying={isRefreshing}
      />
    )
  }

  if (!overview || overview.status === "bootstrapping") {
    return (
      <BootstrappingState
        city={city ?? undefined}
        onRefresh={onRefresh}
        isRefreshing={isRefreshing}
      />
    )
  }

  const current = overview.current_conditions
  const condition = resolveWeatherCondition(
    current?.weather_code,
    current?.condition_symbol
  )
  const ConditionIcon = condition.icon
  const tempSpec = getMetricSpec("temperature_2m", overview.metric_specs)
  const windSpec = getMetricSpec("wind_speed_10m", overview.metric_specs)
  const report = overview.latest_report
  const reportHighlights = extractHighlights(report?.highlights)
  const bulletinBody =
    report?.bulletin_markdown?.trim() || report?.summary?.trim() || ""

  return (
    <div className="flex flex-col gap-6 md:gap-8">
      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh latest station data"
          message={errorMessage}
          onRetry={onRefresh}
          isRetrying={isRefreshing}
        />
      ) : null}

      <AlertBannerList
        alerts={overview.active_alerts}
        timeZone={timeZone}
        metricSpecs={overview.metric_specs}
      />

      {/* Row 1: Hero current conditions (7 cols) + AQI gauge & pollutants (5 cols) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <section className="flex min-h-[220px] flex-col justify-between gap-6 rounded-2xl border border-border bg-accent p-5 text-accent-foreground shadow-(--shadow-card) md:p-7 lg:col-span-7">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-accent-foreground">
                <ConditionIcon className="size-5 shrink-0" strokeWidth={1.75} />
                <span>{condition.label}</span>
              </div>
              <div className="mt-3 flex items-baseline gap-2">
                <span className="font-display text-[3.5rem]/none font-semibold tracking-tight text-foreground tabular-nums md:text-hero">
                  {formatMetricValue(current?.temperature_2m, 1)}
                </span>
                <span className="text-sm font-medium text-muted-foreground">
                  {tempSpec?.unit || "°C"}
                </span>
              </div>
              <p className="mt-2 text-sm text-foreground">
                Feels like{" "}
                <span className="font-semibold tabular-nums">
                  {formatMetricValue(current?.apparent_temperature, 1)}
                  {tempSpec?.unit || "°C"}
                </span>
              </p>
            </div>

            <WindCompass
              speed={current?.wind_speed_10m}
              directionDeg={current?.wind_direction_10m}
              unit={windSpec?.unit || "km/h"}
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/70 pt-3 text-xs text-foreground">
            <span>
              Observed{" "}
              <TimestampPkt
                timestamp={current?.timestamp_utc}
                timeZone={timeZone}
                mode="datetime"
              />
            </span>
            {current?.sources_available?.length ? (
              <span className="text-foreground/80">
                Data: {current.sources_available.map(formatSourceLabel).join(" & ")}
              </span>
            ) : null}
          </div>
        </section>

        <section className="flex flex-col justify-between gap-4 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-(--shadow-card) lg:col-span-5">
          <h2 className="font-display text-[1.25rem]/6.5 font-semibold text-foreground md:text-h2">
            Air quality
          </h2>
          <AqiGauge value={current?.us_aqi} />
          {overview.pollutant_columns.length > 0 ? (
            <dl className="grid grid-cols-3 gap-3 border-t border-border pt-3 sm:grid-cols-6 lg:grid-cols-3">
              {overview.pollutant_columns.map((col) => {
                const pollutantSpec = getMetricSpec(col, overview.metric_specs)
                return (
                  <div key={col} className="min-w-0">
                    <dt className="text-xs text-muted-foreground">
                      {POLLUTANT_LABELS[col] ?? pollutantSpec?.label ?? col}
                    </dt>
                    <dd className="flex items-baseline gap-1 text-sm font-semibold text-foreground tabular-nums">
                      <span>
                        {formatMetricValue(
                          readObservationNumeric(current, col),
                          1
                        )}
                      </span>
                      {pollutantSpec?.unit ? (
                        <span className="text-xs font-medium text-muted-foreground">
                          {pollutantSpec.unit}
                        </span>
                      ) : null}
                    </dd>
                  </div>
                )
              })}
            </dl>
          ) : null}
        </section>
      </div>

      {/* Row 2: 6 Metric Stat Cards with 24h sparklines */}
      {statCardMetrics.length > 0 ? (
        <section aria-label="Core weather metrics">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 md:gap-4 xl:grid-cols-6">
            {statCardMetrics.map((metricKey) => {
              const spec = getMetricSpec(metricKey, overview.metric_specs)
              const colorVar = getMetricColorVar(
                metricKey,
                overview.metric_specs
              )
              const currentVal = readObservationNumeric(current, metricKey)
              const series = overview.observed_24h.map((row) =>
                readObservationNumeric(row, metricKey)
              )
              const aqiMeta =
                metricKey === "us_aqi" ? getAqiLevel(currentVal) : null
              const AqiIcon = aqiMeta?.icon

              return (
                <article
                  key={metricKey}
                  className="flex min-h-[132px] flex-col justify-between gap-3 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5"
                >
                  <div>
                    <div className="flex items-center justify-between gap-1.5">
                      <div className="text-[13px]/4.5 font-medium text-muted-foreground">
                        {spec?.label ?? metricKey}
                      </div>
                      {aqiMeta && AqiIcon ? (
                        <span
                          className={cn(
                            "inline-flex h-5 items-center gap-1 rounded-full border px-1.5 text-[11px] font-medium",
                            aqiMeta.badge
                          )}
                        >
                          <AqiIcon className="size-3 shrink-0" strokeWidth={1.75} />
                          <span className="truncate">{aqiMeta.shortLabel}</span>
                        </span>
                      ) : null}
                    </div>
                    <div className="mt-1.5 flex items-baseline gap-1">
                      <span className="font-display text-[1.75rem]/8 font-semibold tracking-tight text-foreground tabular-nums md:text-stat">
                        {formatMetricValue(currentVal, 1)}
                      </span>
                      {spec?.unit ? (
                        <span className="text-sm font-medium text-muted-foreground">
                          {spec.unit}
                        </span>
                      ) : null}
                    </div>
                  </div>

                  <Sparkline
                    values={series}
                    color={aqiMeta?.fill ?? colorVar}
                    ariaLabel={`${spec?.label ?? metricKey} 24-hour trend`}
                  />
                </article>
              )
            })}
          </div>
        </section>
      ) : null}

      {/* Row 3: 24h Forecast Chart (8 cols) + Weather Bulletin (4 cols) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="lg:col-span-8">
          {activeChartMetric ? (
            <ForecastTrajectoryChart
              metric={activeChartMetric}
              metricSpecs={overview.metric_specs}
              observed={overview.observed_24h}
              forecasts={overview.forecasts_24h[activeChartMetric] ?? []}
              lookbackHours={12}
              horizonHours={overview.forecast_horizon_hours}
              timeZone={timeZone}
              title={`${getMetricSpec(activeChartMetric, overview.metric_specs)?.label ?? activeChartMetric} outlook`}
              subtitle={`Last 12h observed lead-in and ${overview.forecast_horizon_hours}h forecast with confidence interval`}
              subHeader={
                chartMetrics.length > 1 ? (
                  <MetricSelectorTabs
                    metrics={chartMetrics}
                    selectedMetric={activeChartMetric}
                    onSelectMetric={setSelectedMetric}
                    metricSpecs={overview.metric_specs}
                    ariaLabel="24-hour forecast metric"
                  />
                ) : undefined
              }
            />
          ) : (
            <EmptyState
              icon={FileText}
              title="No forecast metrics configured"
              description="Forecast series will appear here once metric definitions and forecast runs are available."
            />
          )}
        </div>

        <aside
          aria-label="Weather bulletin"
          className="flex flex-col justify-between gap-4 rounded-xl border border-border bg-card p-6 text-card-foreground shadow-(--shadow-card) lg:col-span-4"
        >
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-display text-[1.25rem]/6.5 font-semibold text-foreground md:text-h2">
                Weather bulletin
              </h2>
              {report ? (
                <Badge variant="secondary" className="h-6 px-2 text-xs font-medium">
                  {REPORT_TRIGGER_LABELS[report.trigger_reason] ??
                    "Station briefing"}
                </Badge>
              ) : null}
            </div>

            {report ? (
              <div className="flex flex-col gap-3">
                {report.headline ? (
                  <h3 className="text-base/6 font-semibold text-foreground">
                    {report.headline}
                  </h3>
                ) : null}

                {bulletinBody ? (
                  <BulletinMarkdown markdown={bulletinBody} />
                ) : null}

                {reportHighlights.length > 0 ? (
                  <div className="rounded-lg bg-muted p-3.5 text-sm text-foreground">
                    <div className="mb-1.5 text-[13px]/4.5 font-medium text-foreground">
                      Key points
                    </div>
                    <ul className="list-disc space-y-1 pl-4 text-sm/5 text-foreground/90">
                      {reportHighlights.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="rounded-lg bg-muted p-4 text-sm/5 text-foreground">
                <p className="font-medium">No weather bulletin yet</p>
                <p className="mt-1 text-foreground/80">
                  Daily forecasts generate a plain-language briefing here once
                  the report agent completes a run.
                </p>
              </div>
            )}
          </div>

          {report ? (
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
              <span>
                Issued{" "}
                <TimestampPkt
                  timestamp={report.issued_at_utc}
                  timeZone={timeZone}
                  mode="datetime"
                  className="font-medium text-foreground"
                />
              </span>
              {report.active_alert_count > 0 ? (
                <span className="tabular-nums">
                  {report.active_alert_count}{" "}
                  {report.active_alert_count === 1 ? "active alert" : "active alerts"}
                </span>
              ) : null}
            </div>
          ) : null}
        </aside>
      </div>

      {/* Row 4: 3-column Audience Advisories */}
      <AdvisoryCards alerts={overview.active_alerts} />
    </div>
  )
}
