import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ChevronDown, ChevronUp, HelpCircle, History, RotateCcw } from "lucide-react"
import { cn } from "cn"
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  XAxis,
  YAxis,
} from "recharts"

import { ChartCard, type ChartTableColumn } from "@/components/shared/chart-card"
import {
  ChartLegendBar,
  MetricSelectorTabs,
  SkyCastChartTooltipCard,
  computeChartDomain,
  formatValueWithUnit,
  getAqiReferenceBands,
  useFirstLoadChartAnimation,
  type ChartLegendItem,
  type TooltipEntryRow,
} from "@/components/shared/chart-primitives"
import {
  BootstrappingState,
  EmptyState,
  ErrorState,
} from "@/components/shared/empty-state"
import { ForecastTrajectoryChart } from "@/components/shared/forecast-trajectory-chart"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  fetchForecastsHistory,
  type ForecastPoint,
  type ForecastsHistoryResponse,
  type MetricSpecsMap,
} from "@/lib/api"
import { formatLocalDateTime, formatLocalTime, parseIsoDate } from "@/lib/time"
import {
  formatMetricValue,
  formatModelLabel,
  getMetricColorVar,
  getMetricSpec,
  orderMetricKeys,
  pickDefaultMetric,
} from "@/lib/weather"

export interface ForecastsTabProps {
  city?: string | null
  timeZone?: string | null
  metricSpecsHint?: MetricSpecsMap | null
  coreMetricGroupsHint?: Record<string, string[]> | null
  refreshNonce?: number
  onShellRefresh?: () => void
}

interface HistoryChartRow extends Record<string, unknown> {
  target_timestamp_utc: string
  actual: number | null
  predicted: number | null
  naive: number | null
  horizon_hour: number
  model_family: string
  issued_at_utc: string
}

const DEFAULT_LIMIT = 240
const EXPANDED_LIMIT = 500
const STANDARD_HORIZONS = [1, 3, 6, 12, 24] as const
const STANDARD_MODELS = ["SARIMAX", "RandomForest", "Ridge", "LSTM"] as const

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

function formatSignedDelta(
  actual: number | null,
  predicted: number | null
): string {
  if (
    typeof actual !== "number" ||
    !Number.isFinite(actual) ||
    typeof predicted !== "number" ||
    !Number.isFinite(predicted)
  ) {
    return "—"
  }
  const diff = actual - predicted
  const prefix = diff > 0 ? "+" : ""
  return `${prefix}${diff.toFixed(1)}`
}

export function ForecastsTab({
  city,
  timeZone,
  metricSpecsHint,
  coreMetricGroupsHint,
  refreshNonce = 0,
  onShellRefresh,
}: ForecastsTabProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(() =>
    pickDefaultMetric(metricSpecsHint)
  )
  const [lookbackHours, setLookbackHours] = useState<12 | 24>(24)
  const [selectedHorizon, setSelectedHorizon] = useState<string>("all")
  const [selectedModel, setSelectedModel] = useState<string>("all")
  const [showNaiveBaseline, setShowNaiveBaseline] = useState<boolean>(false)
  const [ledgerExpanded, setLedgerExpanded] = useState<boolean>(false)
  const [showAllLedgerCols, setShowAllLedgerCols] = useState<boolean>(false)
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT)
  const [localRetryNonce, setLocalRetryNonce] = useState<number>(0)

  const [data, setData] = useState<ForecastsHistoryResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isFetching, setIsFetching] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const [seenModels, setSeenModels] = useState<string[]>([])
  const [seenHorizons, setSeenHorizons] = useState<number[]>([])

  const satisfiedKeyRef = useRef<string | null>(null)

  const queryMetric = selectedMetric ?? pickDefaultMetric(metricSpecsHint) ?? null

  const buildQueryKey = useCallback(
    (metricKey: string | null) =>
      JSON.stringify({
        metric: metricKey ?? "",
        horizon: selectedHorizon,
        model: selectedModel,
        limit,
        refreshNonce,
        localRetryNonce,
      }),
    [selectedHorizon, selectedModel, limit, refreshNonce, localRetryNonce]
  )

  useEffect(() => {
    const queryKey = buildQueryKey(queryMetric)
    if (satisfiedKeyRef.current === queryKey) {
      return
    }

    const controller = new AbortController()
    setIsFetching(true)
    setErrorMessage(null)

    const parsedHorizon =
      selectedHorizon !== "all" ? Number(selectedHorizon) : undefined

    void fetchForecastsHistory(
      {
        metric: queryMetric ?? undefined,
        model_family: selectedModel !== "all" ? selectedModel : undefined,
        horizon_hour:
          typeof parsedHorizon === "number" && Number.isFinite(parsedHorizon)
            ? parsedHorizon
            : undefined,
        observed_hours: 24,
        limit,
      },
      { signal: controller.signal }
    )
      .then((response) => {
        if (controller.signal.aborted) {
          return
        }

        const resolvedMetric =
          response.metric ??
          pickDefaultMetric(
            response.metric_specs,
            Object.keys(response.latest_forecast ?? {})
          )

        const batchModels = new Set<string>()
        const batchHorizons = new Set<number>()
        for (const pt of response.history ?? []) {
          if (pt.model_family) batchModels.add(pt.model_family)
          if (
            typeof pt.horizon_hour === "number" &&
            Number.isFinite(pt.horizon_hour)
          ) {
            batchHorizons.add(pt.horizon_hour)
          }
        }
        for (const pts of Object.values(response.latest_forecast ?? {})) {
          for (const pt of pts) {
            if (pt.model_family) batchModels.add(pt.model_family)
            if (
              typeof pt.horizon_hour === "number" &&
              Number.isFinite(pt.horizon_hour)
            ) {
              batchHorizons.add(pt.horizon_hour)
            }
          }
        }
        if (batchModels.size > 0) {
          setSeenModels((prev) => Array.from(new Set([...prev, ...batchModels])))
        }
        if (batchHorizons.size > 0) {
          setSeenHorizons((prev) =>
            Array.from(new Set([...prev, ...batchHorizons])).sort((a, b) => a - b)
          )
        }

        satisfiedKeyRef.current = buildQueryKey(response.metric ?? queryMetric)
        if (!selectedMetric && resolvedMetric) {
          setSelectedMetric(resolvedMetric)
        }
        setData(response)
        setIsLoading(false)
        setIsFetching(false)
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) {
          return
        }
        if (err instanceof DOMException && err.name === "AbortError") {
          return
        }
        setErrorMessage(
          "Couldn't load forecasts. Check your connection and refresh."
        )
        setIsLoading(false)
        setIsFetching(false)
      })

    return () => controller.abort()
  }, [
    buildQueryKey,
    queryMetric,
    selectedHorizon,
    selectedModel,
    selectedMetric,
    limit,
  ])

  const handleRetry = useCallback(() => {
    satisfiedKeyRef.current = null
    if (onShellRefresh) {
      onShellRefresh()
    } else {
      setLocalRetryNonce((n) => n + 1)
    }
  }, [onShellRefresh])

  const handleResetFilters = useCallback(() => {
    setSelectedHorizon("all")
    setSelectedModel("all")
  }, [])

  const metricSpecs = data?.metric_specs ?? metricSpecsHint ?? null
  const resolvedTimeZone = data?.timezone ?? timeZone ?? null

  const availableMetrics = useMemo(() => {
    const keys = Array.from(
      new Set([
        ...Object.keys(metricSpecs ?? {}),
        ...Object.keys(data?.latest_forecast ?? {}),
      ])
    )
    return orderMetricKeys(keys, coreMetricGroupsHint)
  }, [metricSpecs, data?.latest_forecast, coreMetricGroupsHint])

  const activeMetric = useMemo(() => {
    if (queryMetric && availableMetrics.includes(queryMetric)) {
      return queryMetric
    }
    return (
      pickDefaultMetric(metricSpecs, availableMetrics) ??
      availableMetrics[0] ??
      "temperature_2m"
    )
  }, [queryMetric, availableMetrics, metricSpecs])

  const activeSpec = useMemo(
    () => getMetricSpec(activeMetric, metricSpecs),
    [activeMetric, metricSpecs]
  )
  const activeColorVar = getMetricColorVar(activeMetric, metricSpecs)
  const activeUnit = activeSpec?.unit ?? ""
  const activeLabel = activeSpec?.label ?? activeMetric

  const horizonOptions = useMemo(() => {
    const maxH = data?.forecast_horizon_hours ?? 24
    const defaults = STANDARD_HORIZONS.filter((h) => h <= maxH)
    return Array.from(new Set([...defaults, ...seenHorizons])).sort(
      (a, b) => a - b
    )
  }, [data?.forecast_horizon_hours, seenHorizons])

  const modelOptions = useMemo(() => {
    return Array.from(new Set([...STANDARD_MODELS, ...seenModels]))
  }, [seenModels])

  const activeLatestForecast = useMemo(
    () => data?.latest_forecast?.[activeMetric] ?? [],
    [data?.latest_forecast, activeMetric]
  )

  // Next-24h snapshot (predicted high/low with times)
  const forecastSnapshot = useMemo(() => {
    let highVal = -Infinity
    let highTs: string | null = null
    let lowVal = Infinity
    let lowTs: string | null = null

    for (const pt of activeLatestForecast) {
      if (
        typeof pt.predicted_value !== "number" ||
        !Number.isFinite(pt.predicted_value)
      ) {
        continue
      }
      if (pt.predicted_value > highVal) {
        highVal = pt.predicted_value
        highTs = pt.target_timestamp_utc
      }
      if (pt.predicted_value < lowVal) {
        lowVal = pt.predicted_value
        lowTs = pt.target_timestamp_utc
      }
    }

    return {
      hasPoints: Number.isFinite(highVal) && Number.isFinite(lowVal),
      highVal: Number.isFinite(highVal) ? highVal : null,
      highTs,
      lowVal: Number.isFinite(lowVal) ? lowVal : null,
      lowTs,
    }
  }, [activeLatestForecast])

  const metricScopedHistory = useMemo(() => {
    const raw = data?.history ?? []
    return raw.filter((pt) => !pt.metric || pt.metric === activeMetric)
  }, [data?.history, activeMetric])

  // Deduplicate by target_timestamp_utc for the comparison line chart
  const historyChartRows = useMemo<HistoryChartRow[]>(() => {
    const byTarget = new Map<string, ForecastPoint>()

    for (const pt of metricScopedHistory) {
      const iso = normalizeIso(pt.target_timestamp_utc)
      if (!iso) {
        continue
      }
      const existing = byTarget.get(iso)
      if (!existing) {
        byTarget.set(iso, pt)
        continue
      }
      const existingHasActual =
        typeof existing.actual_value === "number" &&
        Number.isFinite(existing.actual_value)
      const nextHasActual =
        typeof pt.actual_value === "number" && Number.isFinite(pt.actual_value)

      if (nextHasActual && !existingHasActual) {
        byTarget.set(iso, pt)
        continue
      }
      if (pt.horizon_hour < existing.horizon_hour) {
        byTarget.set(iso, pt)
        continue
      }
      if (
        pt.horizon_hour === existing.horizon_hour &&
        pt.issued_at_utc > existing.issued_at_utc
      ) {
        byTarget.set(iso, pt)
      }
    }

    return Array.from(byTarget.entries())
      .map(([iso, pt]) => ({
        target_timestamp_utc: iso,
        actual:
          typeof pt.actual_value === "number" && Number.isFinite(pt.actual_value)
            ? pt.actual_value
            : null,
        predicted:
          typeof pt.predicted_value === "number" &&
          Number.isFinite(pt.predicted_value)
            ? pt.predicted_value
            : null,
        naive:
          typeof pt.naive_baseline_value === "number" &&
          Number.isFinite(pt.naive_baseline_value)
            ? pt.naive_baseline_value
            : null,
        horizon_hour: pt.horizon_hour,
        model_family: pt.model_family,
        issued_at_utc: pt.issued_at_utc,
      }))
      .sort(
        (a, b) =>
          Date.parse(a.target_timestamp_utc) -
          Date.parse(b.target_timestamp_utc)
      )
  }, [metricScopedHistory])

  const hasNaiveSeries = useMemo(
    () => historyChartRows.some((row) => row.naive !== null),
    [historyChartRows]
  )

  const hasActualSeries = useMemo(
    () => historyChartRows.some((row) => row.actual !== null),
    [historyChartRows]
  )

  const verificationStats = useMemo(() => {
    let count = 0
    let absErrSum = 0
    let sqErrSum = 0
    let baselineCount = 0
    let modelBaselineErrSum = 0
    let naiveBaselineErrSum = 0

    for (const row of historyChartRows) {
      if (row.actual === null || row.predicted === null) {
        continue
      }
      const err = row.actual - row.predicted
      count += 1
      absErrSum += Math.abs(err)
      sqErrSum += err * err

      if (row.naive !== null) {
        baselineCount += 1
        modelBaselineErrSum += Math.abs(err)
        naiveBaselineErrSum += Math.abs(row.actual - row.naive)
      }
    }

    const mae = count > 0 ? absErrSum / count : null
    const rmse = count > 0 ? Math.sqrt(sqErrSum / count) : null
    const improvementPct =
      baselineCount > 0 && naiveBaselineErrSum > 0
        ? ((naiveBaselineErrSum - modelBaselineErrSum) / naiveBaselineErrSum) *
          100
        : null

    return { count, mae, rmse, improvementPct }
  }, [historyChartRows])

  const shouldAnimateHistory = useFirstLoadChartAnimation(
    historyChartRows.length > 0
  )

  const [historyDomainMin, historyDomainMax] = useMemo(() => {
    const nums: Array<number | null> = []
    for (const row of historyChartRows) {
      nums.push(row.actual, row.predicted)
      if (showNaiveBaseline) {
        nums.push(row.naive)
      }
    }
    return computeChartDomain(nums, activeSpec)
  }, [historyChartRows, showNaiveBaseline, activeSpec])

  const historyAqiBands = useMemo(() => {
    if (activeMetric !== "us_aqi") {
      return []
    }
    return getAqiReferenceBands(historyDomainMin, historyDomainMax)
  }, [activeMetric, historyDomainMin, historyDomainMax])

  // Use neutral dark stroke for Actual and metric accent color for Forecast so they separate cleanly.
  const historyChartConfig = useMemo(
    () => ({
      actual: { label: "Actual observed", color: "var(--foreground)" },
      predicted: { label: "Model forecast", color: activeColorVar },
      naive: { label: "24h baseline", color: "var(--model-naive)" },
    }),
    [activeColorVar]
  )

  const historyLegendItems = useMemo<ChartLegendItem[]>(() => {
    const items: ChartLegendItem[] = [
      {
        key: "actual",
        label: "Actual observed",
        color: "var(--foreground)",
        variant: "solid",
      },
      {
        key: "predicted",
        label: "Model forecast",
        color: activeColorVar,
        variant: "dashed",
      },
    ]
    if (showNaiveBaseline && hasNaiveSeries) {
      items.push({
        key: "naive",
        label: "24h baseline",
        color: "var(--model-naive)",
        variant: "dotted",
      })
    }
    return items
  }, [activeColorVar, showNaiveBaseline, hasNaiveSeries])

  const historyChartTableColumns = useMemo<
    ChartTableColumn<HistoryChartRow>[]
  >(
    () => [
      {
        key: "target_timestamp_utc",
        header: "Target time",
        render: (row) => (
          <TimestampPkt
            timestamp={row.target_timestamp_utc}
            timeZone={resolvedTimeZone}
            mode="datetime"
          />
        ),
      },
      {
        key: "actual",
        header: `Actual${activeUnit ? ` (${activeUnit})` : ""}`,
        align: "right",
        render: (row) => formatMetricValue(row.actual, 1),
      },
      {
        key: "predicted",
        header: `Forecast${activeUnit ? ` (${activeUnit})` : ""}`,
        align: "right",
        render: (row) => formatMetricValue(row.predicted, 1),
      },
      {
        key: "naive",
        header: `24h baseline${activeUnit ? ` (${activeUnit})` : ""}`,
        align: "right",
        render: (row) => formatMetricValue(row.naive, 1),
      },
      {
        key: "horizon_hour",
        header: "Horizon",
        align: "right",
        render: (row) => `+${row.horizon_hour}h`,
      },
      {
        key: "model_family",
        header: "Model",
        render: (row) => formatModelLabel(row.model_family),
      },
    ],
    [resolvedTimeZone, activeUnit]
  )

  const sortedLedgerRows = useMemo(() => {
    return [...metricScopedHistory].sort((a, b) => {
      const targetDiff =
        (parseIsoDate(b.target_timestamp_utc)?.getTime() ?? 0) -
        (parseIsoDate(a.target_timestamp_utc)?.getTime() ?? 0)
      if (targetDiff !== 0) {
        return targetDiff
      }
      if (a.horizon_hour !== b.horizon_hour) {
        return a.horizon_hour - b.horizon_hour
      }
      return (
        (parseIsoDate(b.issued_at_utc)?.getTime() ?? 0) -
        (parseIsoDate(a.issued_at_utc)?.getTime() ?? 0)
      )
    })
  }, [metricScopedHistory])

  if (isLoading && !data) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-11 w-full animate-pulse rounded-lg bg-muted" />
        <div className="h-[380px] animate-pulse rounded-xl border border-border bg-muted" />
        <div className="h-[340px] animate-pulse rounded-xl border border-border bg-muted" />
      </div>
    )
  }

  if (errorMessage && !data) {
    return (
      <ErrorState
        title="Couldn't load forecasts"
        message={errorMessage}
        onRetry={handleRetry}
        isRetrying={isFetching}
      />
    )
  }

  if (!data || data.status === "bootstrapping") {
    return (
      <BootstrappingState
        city={data?.city ?? city ?? undefined}
        onRefresh={handleRetry}
        isRefreshing={isFetching}
      />
    )
  }

  const hasActiveFilters =
    selectedHorizon !== "all" || selectedModel !== "all"

  return (
    <div className="flex flex-col gap-6 md:gap-8">
      {/* Header + Metric selector tabs */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
              Forecasts and accuracy check
            </h2>
            <p className="mt-1 max-w-[68ch] text-sm/5 text-muted-foreground">
              Where {activeLabel.toLowerCase()} is headed over the next{" "}
              {data.forecast_horizon_hours} hours, and how accurately recent
              forecasts matched real weather.
            </p>
          </div>
          {isFetching ? (
            <span className="text-xs font-medium text-muted-foreground">
              Updating view...
            </span>
          ) : null}
        </div>

        <MetricSelectorTabs
          metrics={availableMetrics}
          selectedMetric={activeMetric}
          onSelectMetric={setSelectedMetric}
          metricSpecs={metricSpecs}
          ariaLabel="Forecast metric selector"
        />
      </div>

      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh forecasts"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      ) : null}

      {/* Section 1: 24h Trajectory Chart with Next-24h Snapshot */}
      <ForecastTrajectoryChart
        metric={activeMetric}
        metricSpecs={metricSpecs}
        observed={data.observed_history}
        forecasts={activeLatestForecast}
        lookbackHours={lookbackHours}
        horizonHours={data.forecast_horizon_hours}
        timeZone={resolvedTimeZone}
        title={`Next ${data.forecast_horizon_hours}h ${activeLabel.toLowerCase()} outlook`}
        subtitle={`Last ${lookbackHours}h observed readings followed by ${data.forecast_horizon_hours}h forecast with confidence range`}
        subHeader={
          forecastSnapshot.hasPoints ? (
            <dl className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
              <div className="rounded-lg bg-muted px-3 py-2">
                <dt className="text-xs text-muted-foreground">
                  Expected high
                </dt>
                <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                  {formatValueWithUnit(forecastSnapshot.highVal, activeUnit, 1)}
                  {forecastSnapshot.highTs ? (
                    <span className="ml-1.5 text-xs font-normal text-muted-foreground">
                      at {formatLocalTime(forecastSnapshot.highTs, resolvedTimeZone)}
                    </span>
                  ) : null}
                </dd>
              </div>
              <div className="rounded-lg bg-muted px-3 py-2">
                <dt className="text-xs text-muted-foreground">Expected low</dt>
                <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                  {formatValueWithUnit(forecastSnapshot.lowVal, activeUnit, 1)}
                  {forecastSnapshot.lowTs ? (
                    <span className="ml-1.5 text-xs font-normal text-muted-foreground">
                      at {formatLocalTime(forecastSnapshot.lowTs, resolvedTimeZone)}
                    </span>
                  ) : null}
                </dd>
              </div>
              <div className="col-span-2 rounded-lg bg-muted px-3 py-2 sm:col-span-1">
                <dt className="text-xs text-muted-foreground">
                  Recent accuracy
                </dt>
                <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                  {verificationStats.mae !== null
                    ? `±${formatValueWithUnit(verificationStats.mae, activeUnit, 1)} avg miss`
                    : `${activeLatestForecast.length}h forecast active`}
                </dd>
              </div>
            </dl>
          ) : undefined
        }
        controls={
          <div
            role="group"
            aria-label="Observed lookback window"
            className="inline-flex h-8 items-center rounded-md bg-muted p-0.5"
          >
            {([12, 24] as const).map((hrs) => {
              const active = lookbackHours === hrs
              return (
                <button
                  key={hrs}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setLookbackHours(hrs)}
                  className={cn(
                    "inline-flex h-7 items-center rounded-sm px-2.5 text-xs font-medium transition-colors duration-120",
                    active
                      ? "bg-card text-foreground shadow-(--shadow-card)"
                      : "text-foreground/75 hover:text-foreground"
                  )}
                >
                  {hrs}h history
                </button>
              )
            })}
          </div>
        }
      />

      {/* Section 2: Forecast vs. Actual Accuracy Check */}
      <ChartCard
        title={`How accurate were recent ${activeLabel.toLowerCase()} forecasts?`}
        titleInfo="Compares past model predictions (colored dashed line) against what actually happened (solid dark line). Enable 24h baseline to compare against simply repeating yesterday's weather."
        subtitle={
          hasActualSeries
            ? "Solid dark line is actual weather; colored dashed line is what SkyCast predicted"
            : "Predicted history (actual observations fill in as target hours pass)"
        }
        ariaLabel={`${activeLabel} forecast versus actual history chart`}
        controls={
          <div className="flex flex-wrap items-center gap-2">
            <Select value={selectedHorizon} onValueChange={setSelectedHorizon}>
              <SelectTrigger
                aria-label="Filter by forecast horizon"
                className="h-8 min-w-[122px] bg-card text-xs"
              >
                <SelectValue placeholder="Horizon" />
              </SelectTrigger>
              <SelectContent position="popper">
                <SelectItem value="all">Best horizon</SelectItem>
                {horizonOptions.map((h) => (
                  <SelectItem key={h} value={String(h)}>
                    +{h}h ahead
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger
                aria-label="Filter by model family"
                className="h-8 min-w-[128px] bg-card text-xs"
              >
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent position="popper">
                <SelectItem value="all">All models</SelectItem>
                {modelOptions.map((model) => (
                  <SelectItem key={model} value={model}>
                    {formatModelLabel(model)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {hasNaiveSeries ? (
              <Button
                type="button"
                variant={showNaiveBaseline ? "secondary" : "outline"}
                size="sm"
                className="h-8 px-2.5 text-xs font-medium"
                aria-pressed={showNaiveBaseline}
                onClick={() => setShowNaiveBaseline((prev) => !prev)}
              >
                24h baseline
              </Button>
            ) : null}
          </div>
        }
        subHeader={
          verificationStats.count > 0 ? (
            <dl className="grid grid-cols-3 gap-2.5">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="flex items-center gap-1 text-xs text-muted-foreground">
                      <span>Average miss</span>
                      <HelpCircle className="size-3" strokeWidth={1.75} />
                    </dt>
                    <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                      ±{formatValueWithUnit(verificationStats.mae, activeUnit, 2)}
                    </dd>
                  </div>
                </TooltipTrigger>
                <TooltipContent className="text-xs tabular-nums">
                  MAE: {formatValueWithUnit(verificationStats.mae, activeUnit, 2)}{" "}
                  · RMSE:{" "}
                  {formatValueWithUnit(verificationStats.rmse, activeUnit, 2)}
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="flex items-center gap-1 text-xs text-muted-foreground">
                      <span>Vs. 24h baseline</span>
                      <HelpCircle className="size-3" strokeWidth={1.75} />
                    </dt>
                    <dd
                      className={cn(
                        "mt-0.5 text-sm font-semibold tabular-nums",
                        verificationStats.improvementPct !== null &&
                          verificationStats.improvementPct > 0
                          ? "text-aqi-good-fg"
                          : "text-foreground"
                      )}
                    >
                      {verificationStats.improvementPct !== null
                        ? `${verificationStats.improvementPct >= 0 ? "+" : ""}${verificationStats.improvementPct.toFixed(1)}%`
                        : "—"}
                    </dd>
                  </div>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs">
                  Error reduction compared to predicting yesterday&apos;s weather
                  at the same hour.
                </TooltipContent>
              </Tooltip>

              <div className="rounded-lg bg-muted px-3 py-2">
                <dt className="text-xs text-muted-foreground">Hours checked</dt>
                <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                  {verificationStats.count}
                </dd>
              </div>
            </dl>
          ) : undefined
        }
        tableColumns={historyChartTableColumns}
        tableRows={historyChartRows}
        emptyTableMessage="No historical forecast records match the active filters."
        footer={
          selectedHorizon === "all" && historyChartRows.length > 0 ? (
            <span>
              Chart shows the closest-horizon forecast per hour; expand the
              hourly ledger below to inspect every individual run.
            </span>
          ) : undefined
        }
      >
        {historyChartRows.length === 0 ? (
          <div className="flex flex-col items-start justify-between gap-4 rounded-lg bg-muted p-5 sm:flex-row sm:items-center">
            <div className="flex items-start gap-3">
              <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-card text-muted-foreground">
                <History className="size-5" strokeWidth={1.75} aria-hidden="true" />
              </div>
              <div>
                <h4 className="text-sm font-semibold text-foreground">
                  No matching forecast history
                </h4>
                <p className="mt-1 max-w-[68ch] text-sm/5 text-foreground/80">
                  {hasActiveFilters
                    ? "No stored forecast records match the selected horizon and model filters."
                    : "Historical forecast comparisons appear once prediction runs and matching observations are recorded."}
                </p>
              </div>
            </div>
            {hasActiveFilters ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-1.5 bg-card px-3"
                onClick={handleResetFilters}
              >
                <RotateCcw className="size-3.5" strokeWidth={1.75} />
                <span>Reset filters</span>
              </Button>
            ) : null}
          </div>
        ) : (
          <div>
            <ChartContainer
              config={historyChartConfig}
              className="h-[240px] w-full aspect-auto md:h-[300px]"
            >
              <ComposedChart
                data={historyChartRows}
                margin={{ top: 12, right: 12, bottom: 4, left: 0 }}
              >
                {historyAqiBands.map((band) => (
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
                  dataKey="target_timestamp_utc"
                  tickFormatter={(iso) =>
                    formatLocalDateTime(String(iso), resolvedTimeZone, {
                      includeZone: false,
                    })
                  }
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                  minTickGap={42}
                />

                <YAxis
                  domain={[historyDomainMin, historyDomainMax]}
                  tickFormatter={formatAxisTick}
                  tickLine={false}
                  axisLine={false}
                  width={44}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                />

                <ChartTooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) {
                      return null
                    }
                    const row = payload[0]?.payload as
                      | HistoryChartRow
                      | undefined
                    if (!row) {
                      return null
                    }

                    const tooltipRows: TooltipEntryRow[] = []
                    if (row.actual !== null) {
                      tooltipRows.push({
                        key: "actual",
                        label: "Actual observed",
                        valueText: formatValueWithUnit(
                          row.actual,
                          activeUnit,
                          1
                        ),
                        color: "var(--foreground)",
                      })
                    }
                    if (row.predicted !== null) {
                      tooltipRows.push({
                        key: "predicted",
                        label: `Forecast (+${row.horizon_hour}h)`,
                        valueText: formatValueWithUnit(
                          row.predicted,
                          activeUnit,
                          1
                        ),
                        color: activeColorVar,
                        dashed: true,
                      })
                    }
                    if (showNaiveBaseline && row.naive !== null) {
                      tooltipRows.push({
                        key: "naive",
                        label: "24h baseline",
                        valueText: formatValueWithUnit(
                          row.naive,
                          activeUnit,
                          1
                        ),
                        color: "var(--model-naive)",
                        dashed: true,
                      })
                    }

                    const aqiProbe = row.actual ?? row.predicted
                    return (
                      <SkyCastChartTooltipCard
                        timestampUtc={row.target_timestamp_utc}
                        timeZone={resolvedTimeZone}
                        rows={tooltipRows}
                        aqiValue={activeMetric === "us_aqi" ? aqiProbe : null}
                        footerNote={`Model: ${formatModelLabel(row.model_family)}`}
                      />
                    )
                  }}
                />

                {showNaiveBaseline && hasNaiveSeries ? (
                  <Line
                    type="monotone"
                    dataKey="naive"
                    stroke="var(--model-naive)"
                    strokeWidth={1.75}
                    strokeDasharray="2 4"
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 2 }}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                ) : null}

                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="var(--foreground)"
                  strokeWidth={2.25}
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 2 }}
                  connectNulls={false}
                  isAnimationActive={shouldAnimateHistory}
                  animationDuration={600}
                  animationEasing="ease-out"
                />

                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke={activeColorVar}
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={false}
                  activeDot={{ r: 5, strokeWidth: 2 }}
                  connectNulls={false}
                  isAnimationActive={shouldAnimateHistory}
                  animationDuration={600}
                  animationEasing="ease-out"
                />
              </ComposedChart>
            </ChartContainer>

            <ChartLegendBar items={historyLegendItems} />
          </div>
        )}
      </ChartCard>

      {/* Section 3: Collapsible Hourly Forecast Ledger */}
      <section className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base/6 font-semibold text-card-foreground">
              Hourly forecast ledger
            </h3>
            <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
              {data.total_count.toLocaleString()}{" "}
              {data.total_count === 1 ? "stored record" : "stored records"} for{" "}
              {activeLabel}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {hasActiveFilters ? (
              <Badge variant="secondary" className="h-6 px-2 text-xs">
                {selectedHorizon !== "all" ? `+${selectedHorizon}h` : "Best horizon"}
                {" / "}
                {selectedModel !== "all"
                  ? formatModelLabel(selectedModel)
                  : "All models"}
              </Badge>
            ) : null}

            {ledgerExpanded ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 px-2.5 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => setShowAllLedgerCols((prev) => !prev)}
              >
                {showAllLedgerCols ? "Compact columns" : "All columns"}
              </Button>
            ) : null}

            {ledgerExpanded &&
            sortedLedgerRows.length < data.total_count &&
            limit < EXPANDED_LIMIT ? (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 px-3 text-xs font-medium"
                disabled={isFetching}
                onClick={() => setLimit(EXPANDED_LIMIT)}
              >
                Load more
              </Button>
            ) : null}

            <Button
              type="button"
              variant="outline"
              size="sm"
              aria-expanded={ledgerExpanded}
              onClick={() => setLedgerExpanded((prev) => !prev)}
              className="h-8 gap-1.5 px-3 text-xs font-medium"
            >
              <span>{ledgerExpanded ? "Hide ledger" : "Inspect ledger"}</span>
              {ledgerExpanded ? (
                <ChevronUp className="size-3.5" strokeWidth={1.75} />
              ) : (
                <ChevronDown className="size-3.5" strokeWidth={1.75} />
              )}
            </Button>
          </div>
        </div>

        {ledgerExpanded ? (
          sortedLedgerRows.length === 0 ? (
            <EmptyState
              icon={History}
              title="No ledger entries yet"
              description={
                hasActiveFilters
                  ? "Clear your horizon or model filter to view stored hourly forecast records."
                  : "Forecast records appear here as prediction pipelines run."
              }
              actionLabel={hasActiveFilters ? "Reset filters" : undefined}
              onAction={hasActiveFilters ? handleResetFilters : undefined}
              className="border-0 bg-muted shadow-none"
            />
          ) : (
            <Table
              aria-label={`${activeLabel} hourly forecast ledger`}
              containerClassName="max-h-[420px] overflow-auto rounded-md border border-border"
            >
              <TableHeader className="sticky top-0 z-10 bg-muted">
                <TableRow className="hover:bg-muted">
                  <TableHead className="h-10 px-4 py-3 text-xs font-medium text-foreground">
                    Target time
                  </TableHead>
                  {showAllLedgerCols ? (
                    <TableHead className="h-10 px-4 py-3 text-xs font-medium text-foreground">
                      Issued
                    </TableHead>
                  ) : null}
                  <TableHead className="h-10 px-4 py-3 text-right text-xs font-medium text-foreground">
                    Horizon
                  </TableHead>
                  <TableHead className="h-10 px-4 py-3 text-xs font-medium text-foreground">
                    Model
                  </TableHead>
                  <TableHead className="h-10 px-4 py-3 text-right text-xs font-medium text-foreground">
                    Forecast{activeUnit ? ` (${activeUnit})` : ""}
                  </TableHead>
                  {showAllLedgerCols ? (
                    <>
                      <TableHead className="h-10 px-4 py-3 text-right text-xs font-medium text-foreground">
                        Interval
                      </TableHead>
                      <TableHead className="h-10 px-4 py-3 text-right text-xs font-medium text-foreground">
                        24h baseline
                      </TableHead>
                    </>
                  ) : null}
                  <TableHead className="h-10 px-4 py-3 text-right text-xs font-medium text-foreground">
                    Actual{activeUnit ? ` (${activeUnit})` : ""}
                  </TableHead>
                  <TableHead className="h-10 px-4 py-3 text-right text-xs font-medium text-foreground">
                    Diff
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedLedgerRows.map((row) => {
                  const hasBand =
                    typeof row.lower_bound === "number" &&
                    Number.isFinite(row.lower_bound) &&
                    typeof row.upper_bound === "number" &&
                    Number.isFinite(row.upper_bound)

                  return (
                    <TableRow
                      key={`${row.id}-${row.run_id}-${row.horizon_hour}-${row.model_family}`}
                      className="h-[52px] md:h-11"
                    >
                      <TableCell className="px-4 py-3 text-sm tabular-nums">
                        <TimestampPkt
                          timestamp={row.target_timestamp_utc}
                          timeZone={resolvedTimeZone}
                          mode="datetime"
                        />
                      </TableCell>
                      {showAllLedgerCols ? (
                        <TableCell className="px-4 py-3 text-sm text-foreground/80 tabular-nums">
                          <TimestampPkt
                            timestamp={row.issued_at_utc}
                            timeZone={resolvedTimeZone}
                            mode="datetime"
                          />
                        </TableCell>
                      ) : null}
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        +{row.horizon_hour}h
                      </TableCell>
                      <TableCell className="px-4 py-3 text-sm">
                        {formatModelLabel(row.model_family)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm font-medium tabular-nums">
                        {formatMetricValue(row.predicted_value, 1)}
                      </TableCell>
                      {showAllLedgerCols ? (
                        <>
                          <TableCell className="px-4 py-3 text-right text-sm text-foreground/80 tabular-nums">
                            {hasBand
                              ? `${formatMetricValue(
                                  Math.min(row.lower_bound, row.upper_bound),
                                  1
                                )} – ${formatMetricValue(
                                  Math.max(row.lower_bound, row.upper_bound),
                                  1
                                )}`
                              : "—"}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                            {formatMetricValue(row.naive_baseline_value, 1)}
                          </TableCell>
                        </>
                      ) : null}
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        {formatMetricValue(row.actual_value, 1)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm font-medium tabular-nums">
                        {formatSignedDelta(
                          row.actual_value,
                          row.predicted_value
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )
        ) : null}
      </section>
    </div>
  )
}
