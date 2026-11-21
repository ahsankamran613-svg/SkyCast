import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Award,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  HelpCircle,
  Layers,
} from "lucide-react"
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
import { ExplainabilityPanel } from "@/components/shared/explainability-panel"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
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
  fetchModelsLeaderboard,
  type MetricSpecsMap,
  type ModelRegistryMetricRow,
  type ModelsLeaderboardResponse,
} from "@/lib/api"
import { formatLocalTime, parseIsoDate } from "@/lib/time"
import {
  formatMetricValue,
  formatModelLabel,
  getMetricSpec,
  getModelColorVar,
  humanizeIdentifier,
  mapModelFamilyToSeriesKey,
  orderMetricKeys,
  pickDefaultMetric,
} from "@/lib/weather"

export interface ModelsTabProps {
  city?: string | null
  timeZone?: string | null
  metricSpecsHint?: MetricSpecsMap | null
  coreMetricGroupsHint?: Record<string, string[]> | null
  refreshNonce?: number
  onShellRefresh?: () => void
}

type LeaderboardSortKey =
  | "model_family"
  | "val_rmse"
  | "val_mae"
  | "val_r2"
  | "test_rmse"
  | "test_mae"
  | "test_r2"
  | "improvement_over_naive_pct"

type SortDirection = "asc" | "desc"
type LeaderboardViewMode = "overview" | "full"

const COMPARISON_SERIES_META = [
  { key: "actual", label: "Actual", color: "var(--foreground)", dashed: false },
  { key: "sarimax", label: "SARIMAX", color: "var(--model-sarimax)", dashed: false },
  {
    key: "random_forest",
    label: "Random Forest",
    color: "var(--model-rf)",
    dashed: false,
  },
  { key: "ridge", label: "Ridge", color: "var(--model-ridge)", dashed: false },
  { key: "lstm", label: "LSTM", color: "var(--model-lstm)", dashed: false },
  {
    key: "naive_24h",
    label: "24h baseline",
    color: "var(--model-naive)",
    dashed: true,
  },
] as const

type ComparisonSeriesKey = (typeof COMPARISON_SERIES_META)[number]["key"]

interface ComparisonChartRow extends Record<string, unknown> {
  timestamp_utc: string
  actual: number | null
  sarimax: number | null
  random_forest: number | null
  ridge: number | null
  lstm: number | null
  naive_24h: number | null
}

function isFiniteNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatAxisTick(value: unknown): string {
  const num = Number(value)
  if (!Number.isFinite(num)) {
    return ""
  }
  return Number.isInteger(num) ? String(num) : num.toFixed(1)
}

function formatSignedPercent(value: number | null | undefined): string {
  if (!isFiniteNum(value)) {
    return "—"
  }
  const prefix = value > 0 ? "+" : ""
  return `${prefix}${value.toFixed(1)}%`
}

function formatR2Friendly(r2: number | null | undefined): string {
  if (!isFiniteNum(r2)) return "—"
  if (r2 >= 0 && r2 <= 1) {
    return `${Math.round(r2 * 100)}%`
  }
  return r2.toFixed(2)
}

function getSortableNumber(
  row: ModelRegistryMetricRow,
  key: Exclude<LeaderboardSortKey, "model_family">
): number | null {
  const raw = row[key]
  return isFiniteNum(raw) ? raw : null
}

function normalizeIso(ts: string | null | undefined): string | null {
  const parsed = parseIsoDate(ts)
  return parsed ? parsed.toISOString() : null
}

function MetricInfoIcon({ text, label }: { text: string; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className="inline-flex size-4 shrink-0 items-center justify-center rounded-full text-foreground/60 transition-colors hover:bg-card/60 hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          <HelpCircle className="size-3.5" strokeWidth={1.75} />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-left text-xs leading-relaxed">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

export function ModelsTab({
  city,
  timeZone,
  metricSpecsHint,
  coreMetricGroupsHint,
  refreshNonce = 0,
  onShellRefresh,
}: ModelsTabProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(() =>
    pickDefaultMetric(metricSpecsHint)
  )
  const [latestRunOnly, setLatestRunOnly] = useState<boolean>(true)
  const [leaderboardView, setLeaderboardView] =
    useState<LeaderboardViewMode>("overview")
  const [sortKey, setSortKey] = useState<LeaderboardSortKey>("test_rmse")
  const [sortDir, setSortDir] = useState<SortDirection>("asc")
  const [expandedRowIds, setExpandedRowIds] = useState<Set<string>>(new Set())

  // Custom series visibility override per metric (null = default to Actual + Champion)
  const [customVisibleSeries, setCustomVisibleSeries] = useState<
    Set<ComparisonSeriesKey> | null
  >(null)
  const [localRetryNonce, setLocalRetryNonce] = useState<number>(0)

  const [data, setData] = useState<ModelsLeaderboardResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isFetching, setIsFetching] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const satisfiedKeyRef = useRef<string | null>(null)
  const queryMetric = selectedMetric ?? pickDefaultMetric(metricSpecsHint) ?? null

  const buildQueryKey = useCallback(
    (metricKey: string | null) =>
      JSON.stringify({
        metric: metricKey ?? "",
        latestRunOnly,
        refreshNonce,
        localRetryNonce,
      }),
    [latestRunOnly, refreshNonce, localRetryNonce]
  )

  useEffect(() => {
    const queryKey = buildQueryKey(queryMetric)
    if (satisfiedKeyRef.current === queryKey) {
      return
    }

    const controller = new AbortController()
    setIsFetching(true)
    setErrorMessage(null)

    void fetchModelsLeaderboard(
      {
        metric: queryMetric ?? undefined,
        ...(latestRunOnly ? {} : { latest_run_only: false }),
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
            Object.keys(response.champions ?? {})
          )

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
          "Couldn't load model accuracy data. Check your connection and refresh."
        )
        setIsLoading(false)
        setIsFetching(false)
      })

    return () => controller.abort()
  }, [buildQueryKey, queryMetric, latestRunOnly, selectedMetric])

  const handleRetry = useCallback(() => {
    satisfiedKeyRef.current = null
    if (onShellRefresh) {
      onShellRefresh()
    } else {
      setLocalRetryNonce((n) => n + 1)
    }
  }, [onShellRefresh])

  const metricSpecs = data?.metric_specs ?? metricSpecsHint ?? null
  const resolvedTimeZone = data?.timezone ?? timeZone ?? null

  const availableMetrics = useMemo(() => {
    const keys = new Set<string>([
      ...Object.keys(metricSpecs ?? {}),
      ...Object.keys(data?.champions ?? {}),
    ])
    for (const row of data?.leaderboard ?? []) {
      if (row.metric) keys.add(row.metric)
    }
    return orderMetricKeys(Array.from(keys), coreMetricGroupsHint)
  }, [metricSpecs, data?.champions, data?.leaderboard, coreMetricGroupsHint])

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
  const activeUnit = activeSpec?.unit ?? ""
  const activeLabel = activeSpec?.label ?? humanizeIdentifier(activeMetric)

  const metricLeaderboardRows = useMemo(() => {
    const rows = data?.leaderboard ?? []
    return rows.filter((row) => !row.metric || row.metric === activeMetric)
  }, [data?.leaderboard, activeMetric])

  const activeChampion = useMemo<ModelRegistryMetricRow | null>(() => {
    const fromChampions = data?.champions?.[activeMetric]
    if (fromChampions) {
      return fromChampions
    }
    const flagged = metricLeaderboardRows.find((row) => row.is_champion)
    if (flagged) {
      return flagged
    }
    const withRmse = metricLeaderboardRows.filter((row) => isFiniteNum(row.test_rmse))
    if (withRmse.length === 0) {
      return metricLeaderboardRows[0] ?? null
    }
    return [...withRmse].sort((a, b) => a.test_rmse - b.test_rmse)[0] ?? null
  }, [data?.champions, activeMetric, metricLeaderboardRows])

  const sortedLeaderboard = useMemo(() => {
    const copy = [...metricLeaderboardRows]
    copy.sort((a, b) => {
      if (sortKey === "model_family") {
        const cmp = formatModelLabel(a.model_family).localeCompare(
          formatModelLabel(b.model_family)
        )
        return sortDir === "asc" ? cmp : -cmp
      }

      const numA = getSortableNumber(a, sortKey)
      const numB = getSortableNumber(b, sortKey)
      if (numA === null && numB === null) return 0
      if (numA === null) return 1
      if (numB === null) return -1
      const diff = numA - numB
      return sortDir === "asc" ? diff : -diff
    })
    return copy
  }, [metricLeaderboardRows, sortKey, sortDir])

  // Latest-run per model family bars for visual RMSE scan
  const benchmarkBars = useMemo(() => {
    const byModel = new Map<string, ModelRegistryMetricRow>()
    for (const row of metricLeaderboardRows) {
      if (!isFiniteNum(row.test_rmse)) continue
      const existing = byModel.get(row.model_family)
      if (!existing) {
        byModel.set(row.model_family, row)
        continue
      }
      if (row.is_champion && !existing.is_champion) {
        byModel.set(row.model_family, row)
        continue
      }
      if (row.trained_at_utc > existing.trained_at_utc) {
        byModel.set(row.model_family, row)
      }
    }
    return Array.from(byModel.values()).sort((a, b) => a.test_rmse - b.test_rmse)
  }, [metricLeaderboardRows])

  const maxBenchmarkRmse = useMemo(() => {
    if (benchmarkBars.length === 0) return 0
    return Math.max(...benchmarkBars.map((row) => row.test_rmse))
  }, [benchmarkBars])

  const handleSort = useCallback(
    (nextKey: LeaderboardSortKey) => {
      if (sortKey === nextKey) {
        setSortDir((prev) => (prev === "asc" ? "desc" : "asc"))
      } else {
        setSortKey(nextKey)
        const defaultDesc =
          nextKey === "val_r2" ||
          nextKey === "test_r2" ||
          nextKey === "improvement_over_naive_pct"
        setSortDir(defaultDesc ? "desc" : "asc")
      }
    },
    [sortKey]
  )

  const toggleExpandedRow = useCallback((rowKey: string) => {
    setExpandedRowIds((prev) => {
      const next = new Set(prev)
      if (next.has(rowKey)) {
        next.delete(rowKey)
      } else {
        next.add(rowKey)
      }
      return next
    })
  }, [])

  // Comparison series preparation
  const comparisonRows = useMemo<ComparisonChartRow[]>(() => {
    const raw = (data?.comparison_series ?? []).filter(
      (pt) => !pt.metric || pt.metric === activeMetric
    )

    const mapped: ComparisonChartRow[] = []
    for (const pt of raw) {
      const iso = normalizeIso(pt.timestamp_utc)
      if (!iso) continue
      mapped.push({
        timestamp_utc: iso,
        actual: isFiniteNum(pt.actual) ? pt.actual : null,
        sarimax: isFiniteNum(pt.sarimax) ? pt.sarimax : null,
        random_forest: isFiniteNum(pt.random_forest) ? pt.random_forest : null,
        ridge: isFiniteNum(pt.ridge) ? pt.ridge : null,
        lstm: isFiniteNum(pt.lstm) ? pt.lstm : null,
        naive_24h: isFiniteNum(pt.naive_24h) ? pt.naive_24h : null,
      })
    }

    return mapped.sort(
      (a, b) => Date.parse(a.timestamp_utc) - Date.parse(b.timestamp_utc)
    )
  }, [data?.comparison_series, activeMetric])

  const presentSeriesMeta = useMemo(() => {
    return COMPARISON_SERIES_META.filter((series) =>
      comparisonRows.some((row) => row[series.key] !== null)
    )
  }, [comparisonRows])

  // Default to Actual + Champion model only so the chart is never a 6-line spaghetti plot.
  const defaultVisibleKeys = useMemo(() => {
    const keys = new Set<ComparisonSeriesKey>()
    const presentKeys = new Set(presentSeriesMeta.map((s) => s.key))
    if (presentKeys.has("actual")) {
      keys.add("actual")
    }
    const champKey = mapModelFamilyToSeriesKey(activeChampion?.model_family)
    if (champKey && presentKeys.has(champKey)) {
      keys.add(champKey)
    } else {
      const firstModel = presentSeriesMeta.find(
        (s) => s.key !== "actual" && s.key !== "naive_24h"
      )
      if (firstModel) {
        keys.add(firstModel.key)
      }
    }
    return keys
  }, [presentSeriesMeta, activeChampion?.model_family])

  const activeVisibleKeys = useMemo(
    () => customVisibleSeries ?? defaultVisibleKeys,
    [customVisibleSeries, defaultVisibleKeys]
  )

  const visibleSeriesMeta = useMemo(
    () => presentSeriesMeta.filter((series) => activeVisibleKeys.has(series.key)),
    [presentSeriesMeta, activeVisibleKeys]
  )

  const toggleSeriesVisibility = useCallback(
    (key: ComparisonSeriesKey) => {
      setCustomVisibleSeries((prev) => {
        const base = new Set(prev ?? defaultVisibleKeys)
        if (base.has(key)) {
          base.delete(key)
        } else {
          base.add(key)
        }
        return base
      })
    },
    [defaultVisibleKeys]
  )

  const allSeriesSelected =
    presentSeriesMeta.length > 0 &&
    visibleSeriesMeta.length === presentSeriesMeta.length

  const [comparisonMin, comparisonMax] = useMemo(() => {
    const nums: Array<number | null> = []
    for (const row of comparisonRows) {
      for (const series of visibleSeriesMeta) {
        nums.push(row[series.key])
      }
    }
    return computeChartDomain(nums, activeSpec)
  }, [comparisonRows, visibleSeriesMeta, activeSpec])

  const comparisonAqiBands = useMemo(() => {
    if (activeMetric !== "us_aqi") return []
    return getAqiReferenceBands(comparisonMin, comparisonMax)
  }, [activeMetric, comparisonMin, comparisonMax])

  const shouldAnimateComparison = useFirstLoadChartAnimation(
    comparisonRows.length > 0
  )

  const comparisonChartConfig = useMemo(() => {
    const cfg: Record<string, { label: string; color: string }> = {}
    for (const item of COMPARISON_SERIES_META) {
      cfg[item.key] = { label: item.label, color: item.color }
    }
    return cfg
  }, [])

  const comparisonLegendItems = useMemo<ChartLegendItem[]>(
    () =>
      visibleSeriesMeta.map((series) => ({
        key: series.key,
        label: series.label,
        color: series.color,
        variant: series.dashed ? "dashed" : "solid",
      })),
    [visibleSeriesMeta]
  )

  const comparisonTableColumns = useMemo<ChartTableColumn<ComparisonChartRow>[]>(
    () => [
      {
        key: "timestamp_utc",
        header: "Timestamp",
        render: (row) => (
          <TimestampPkt
            timestamp={row.timestamp_utc}
            timeZone={resolvedTimeZone}
            mode="datetime"
          />
        ),
      },
      ...presentSeriesMeta.map((series) => ({
        key: series.key,
        header: `${series.label}${activeUnit ? ` (${activeUnit})` : ""}`,
        align: "right" as const,
        render: (row: ComparisonChartRow) => formatMetricValue(row[series.key], 1),
      })),
    ],
    [resolvedTimeZone, presentSeriesMeta, activeUnit]
  )

  // Scoped explainability records for active metric
  const shapGlobalRecords = useMemo(() => {
    return (data?.explanations?.shap?.global ?? []).filter(
      (rec) => !rec.metric || rec.metric === activeMetric
    )
  }, [data?.explanations?.shap?.global, activeMetric])

  const shapLocalRecords = useMemo(() => {
    return (data?.explanations?.shap?.local ?? []).filter(
      (rec) => !rec.metric || rec.metric === activeMetric
    )
  }, [data?.explanations?.shap?.local, activeMetric])

  const limeLocalRecords = useMemo(() => {
    return (data?.explanations?.lime?.local ?? []).filter(
      (rec) => !rec.metric || rec.metric === activeMetric
    )
  }, [data?.explanations?.lime?.local, activeMetric])

  const hasExplanations =
    shapGlobalRecords.length > 0 ||
    shapLocalRecords.length > 0 ||
    limeLocalRecords.length > 0

  if (isLoading && !data) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-11 w-full animate-pulse rounded-lg bg-muted" />
        <div className="h-[124px] animate-pulse rounded-xl border border-border bg-muted" />
        <div className="h-[280px] animate-pulse rounded-xl border border-border bg-muted" />
        <div className="h-[340px] animate-pulse rounded-xl border border-border bg-muted" />
      </div>
    )
  }

  if (errorMessage && !data) {
    return (
      <ErrorState
        title="Couldn't load model leaderboard"
        message={errorMessage}
        onRetry={handleRetry}
        isRetrying={isFetching}
      />
    )
  }

  const hasAnyModelData =
    (data?.leaderboard.length ?? 0) > 0 ||
    (data?.comparison_series.length ?? 0) > 0 ||
    hasExplanations

  if (!data || (data.status === "bootstrapping" && !hasAnyModelData)) {
    return (
      <BootstrappingState
        city={data?.city ?? city ?? undefined}
        onRefresh={handleRetry}
        isRefreshing={isFetching}
      />
    )
  }

  const sortIconFor = (key: LeaderboardSortKey) => {
    if (sortKey !== key) {
      return (
        <ArrowUpDown
          className="size-3.5 text-muted-foreground"
          strokeWidth={1.75}
        />
      )
    }
    return sortDir === "asc" ? (
      <ArrowUp className="size-3.5 text-primary" strokeWidth={1.75} />
    ) : (
      <ArrowDown className="size-3.5 text-primary" strokeWidth={1.75} />
    )
  }

  const renderSortHead = (
    key: LeaderboardSortKey,
    label: string,
    align: "left" | "right" = "right"
  ) => (
    <TableHead
      aria-sort={
        sortKey === key
          ? sortDir === "asc"
            ? "ascending"
            : "descending"
          : "none"
      }
      className={cn(
        "h-10 px-4 py-2 text-xs font-medium text-foreground",
        align === "right" && "text-right"
      )}
    >
      <button
        type="button"
        onClick={() => handleSort(key)}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-sm py-1 hover:text-primary",
          align === "right" && "flex-row-reverse"
        )}
      >
        <span>{label}</span>
        {sortIconFor(key)}
      </button>
    </TableHead>
  )

  const championVerdict = (() => {
    if (!activeChampion) return null
    const maeStr = isFiniteNum(activeChampion.test_mae)
      ? `±${formatValueWithUnit(activeChampion.test_mae, activeUnit, 2)}`
      : null
    const imp = activeChampion.improvement_over_naive_pct
    if (maeStr && isFiniteNum(imp) && imp > 0) {
      return `Typically within ${maeStr} of actual readings — ${imp.toFixed(1)}% more accurate than repeating yesterday's weather.`
    }
    if (maeStr) {
      return `Typically within ${maeStr} of actual readings on unseen test hours.`
    }
    return `Selected as the top-performing model for ${activeLabel.toLowerCase()} on unseen test hours.`
  })()

  return (
    <div className="flex flex-col gap-6 md:gap-8">
      {/* Header + Metric selector */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
              Model accuracy and explainability
            </h2>
            <p className="mt-1 max-w-[68ch] text-sm/5 text-muted-foreground">
              Which forecasting model performs best, how closely it tracks real
              weather, and why it makes each prediction.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {isFetching ? (
              <span className="text-xs font-medium text-muted-foreground">
                Updating view...
              </span>
            ) : null}

            <div
              role="group"
              aria-label="Training run scope"
              className="inline-flex h-9 items-center rounded-full bg-muted p-1"
            >
              <button
                type="button"
                aria-pressed={latestRunOnly}
                onClick={() => setLatestRunOnly(true)}
                className={cn(
                  "inline-flex h-7 items-center rounded-full px-3 text-xs font-medium transition-colors duration-120",
                  latestRunOnly
                    ? "bg-card text-foreground shadow-(--shadow-card)"
                    : "text-foreground/75 hover:text-foreground"
                )}
              >
                Latest run
              </button>
              <button
                type="button"
                aria-pressed={!latestRunOnly}
                onClick={() => setLatestRunOnly(false)}
                className={cn(
                  "inline-flex h-7 items-center rounded-full px-3 text-xs font-medium transition-colors duration-120",
                  !latestRunOnly
                    ? "bg-card text-foreground shadow-(--shadow-card)"
                    : "text-foreground/75 hover:text-foreground"
                )}
              >
                All runs
              </button>
            </div>
          </div>
        </div>

        <MetricSelectorTabs
          metrics={availableMetrics}
          selectedMetric={activeMetric}
          onSelectMetric={(metric) => {
            setSelectedMetric(metric)
            setCustomVisibleSeries(null)
          }}
          metricSpecs={metricSpecs}
          ariaLabel="Model evaluation metric"
        />
      </div>

      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh model leaderboard"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      ) : null}

      {/* Champion Verdict Hero Strip */}
      {activeChampion ? (
        <section
          aria-label="Champion model summary"
          className="flex flex-col justify-between gap-5 rounded-xl border border-primary/35 bg-accent p-4 text-accent-foreground shadow-(--shadow-card) lg:flex-row lg:items-center md:p-5"
        >
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="default"
                className="h-6 gap-1 px-2.5 text-xs font-semibold"
              >
                <Award className="size-3.5" strokeWidth={1.75} />
                <span>Best model for {activeLabel}</span>
              </Badge>
              {activeChampion.promoted_to_registry ? (
                <Badge
                  variant="secondary"
                  className="h-6 gap-1 bg-card px-2.5 text-xs font-medium text-foreground"
                >
                  <CheckCircle2
                    className="size-3.5 text-aqi-good-fg"
                    strokeWidth={1.75}
                  />
                  <span>Active in production</span>
                </Badge>
              ) : null}
            </div>

            <h3 className="mt-2 font-display text-h2 font-semibold text-foreground">
              {formatModelLabel(activeChampion.model_family)}
            </h3>

            {championVerdict ? (
              <p className="mt-1 max-w-[60ch] text-sm/5 text-foreground/85">
                {championVerdict}
              </p>
            ) : null}

            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-foreground/75">
              <span>
                Trained{" "}
                <TimestampPkt
                  timestamp={activeChampion.trained_at_utc}
                  timeZone={resolvedTimeZone}
                  mode="datetime"
                  className="font-medium text-foreground"
                />
              </span>
              {isFiniteNum(activeChampion.training_samples) ? (
                <span className="tabular-nums">
                  {activeChampion.training_samples.toLocaleString()} training hours
                </span>
              ) : null}
              {isFiniteNum(activeChampion.test_rmse) ? (
                <span className="tabular-nums">
                  Test RMSE:{" "}
                  <span className="font-medium text-foreground">
                    {formatValueWithUnit(activeChampion.test_rmse, activeUnit, 2)}
                  </span>
                </span>
              ) : null}
            </div>
          </div>

          <dl className="grid shrink-0 grid-cols-3 gap-4 sm:gap-6">
            <div>
              <dt className="flex items-center gap-1 text-xs font-medium text-foreground/75">
                <span>Typical error</span>
                <MetricInfoIcon
                  label="About typical error"
                  text={`Mean Absolute Error (MAE) on unseen test hours. Lower is better. Test RMSE is ${formatValueWithUnit(activeChampion.test_rmse, activeUnit, 3)}.`}
                />
              </dt>
              <dd className="mt-0.5 flex items-baseline gap-1">
                <span className="font-display text-[1.625rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                  ±{formatMetricValue(activeChampion.test_mae, 2)}
                </span>
                {activeUnit ? (
                  <span className="text-xs font-medium text-foreground/75">
                    {activeUnit}
                  </span>
                ) : null}
              </dd>
            </div>

            <div>
              <dt className="flex items-center gap-1 text-xs font-medium text-foreground/75">
                <span>Pattern match</span>
                <MetricInfoIcon
                  label="About pattern match score"
                  text={`Test R² = ${formatMetricValue(activeChampion.test_r2, 3)}. Measures how closely predictions follow actual ups and downs (100% is a perfect match).`}
                />
              </dt>
              <dd className="mt-0.5 font-display text-[1.625rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                {formatR2Friendly(activeChampion.test_r2)}
              </dd>
            </div>

            <div>
              <dt className="flex items-center gap-1 text-xs font-medium text-foreground/75">
                <span>Vs. baseline</span>
                <MetricInfoIcon
                  label="About 24h baseline comparison"
                  text="Percentage error reduction compared to simply predicting yesterday's weather at the same hour."
                />
              </dt>
              <dd className="mt-0.5 font-display text-[1.625rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                {formatSignedPercent(activeChampion.improvement_over_naive_pct)}
              </dd>
            </div>
          </dl>
        </section>
      ) : null}

      {/* Section 1: Model Ranking (Overview bars by default, Full Val/Test table on click) */}
      <section className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-1.5">
              <h3 className="text-base/6 font-semibold text-card-foreground">
                {activeLabel} model ranking
              </h3>
              <MetricInfoIcon
                label="About model ranking"
                text="Four different model families are trained and tested on the same unseen hours. The model with the lowest Test RMSE is promoted as Champion."
              />
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
              {benchmarkBars.length}{" "}
              {benchmarkBars.length === 1 ? "model family" : "model families"}{" "}
              ranked by test error (shorter bar is more accurate)
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div
              role="tablist"
              aria-label="Leaderboard detail level"
              className="inline-flex h-8 items-center rounded-full bg-muted p-0.5"
            >
              {(
                [
                  { id: "overview", label: "Overview" },
                  { id: "full", label: "Full metrics" },
                ] as const
              ).map((tab) => {
                const active = leaderboardView === tab.id
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => setLeaderboardView(tab.id)}
                    className={cn(
                      "inline-flex h-7 items-center rounded-full px-3 text-xs font-medium transition-colors duration-120",
                      active
                        ? "bg-card text-foreground shadow-(--shadow-card)"
                        : "text-foreground/75 hover:text-foreground"
                    )}
                  >
                    {tab.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {sortedLeaderboard.length === 0 ? (
          <EmptyState
            icon={Layers}
            title={`No ${activeLabel.toLowerCase()} training metrics yet`}
            description="Training metrics appear here after the training pipeline evaluates candidate models on validation and test splits."
            className="border-0 bg-muted shadow-none"
          />
        ) : leaderboardView === "overview" ? (
          <div className="flex flex-col gap-3">
            {benchmarkBars.length > 0 && maxBenchmarkRmse > 0 ? (
              <div
                aria-label={`${activeLabel} model error ranking`}
                className="flex flex-col gap-2.5"
              >
                {benchmarkBars.map((row, index) => {
                  const widthPct = Math.max(
                    6,
                    Math.min(100, (row.test_rmse / maxBenchmarkRmse) * 100)
                  )
                  const colorVar = getModelColorVar(row.model_family)
                  const isBest =
                    row.is_champion ||
                    (activeChampion !== null &&
                      row.model_family === activeChampion.model_family)
                  const imp = row.improvement_over_naive_pct

                  return (
                    <div
                      key={row.model_family}
                      className={cn(
                        "grid grid-cols-[minmax(130px,195px)_minmax(0,1fr)_auto] items-center gap-3 rounded-lg px-3 py-2",
                        isBest ? "bg-accent/70" : "bg-muted/50"
                      )}
                    >
                      <div className="flex min-w-0 items-center gap-2 text-sm font-medium text-foreground">
                        <span className="w-5 shrink-0 text-xs font-semibold text-muted-foreground tabular-nums">
                          #{index + 1}
                        </span>
                        <span
                          className="size-2.5 shrink-0 rounded-full"
                          style={{ backgroundColor: colorVar }}
                          aria-hidden="true"
                        />
                        <span className="truncate">
                          {formatModelLabel(row.model_family)}
                        </span>
                        {isBest ? (
                          <Badge
                            variant="default"
                            className="hidden h-5 px-1.5 text-[10px] sm:inline-flex"
                          >
                            Best
                          </Badge>
                        ) : null}
                      </div>

                      <div className="h-6 w-full overflow-hidden rounded-sm bg-card/90">
                        <div
                          className="h-full rounded-sm transition-all duration-150"
                          style={{
                            width: `${widthPct}%`,
                            backgroundColor: colorVar,
                          }}
                        />
                      </div>

                      <div className="flex items-baseline justify-end gap-2.5 text-right text-xs tabular-nums">
                        <span className="font-semibold text-foreground">
                          {formatValueWithUnit(row.test_rmse, activeUnit, 2)}
                        </span>
                        <span
                          className={cn(
                            "hidden w-20 sm:inline",
                            isFiniteNum(imp) && imp > 0
                              ? "font-medium text-aqi-good-fg"
                              : "text-muted-foreground"
                          )}
                        >
                          {isFiniteNum(imp)
                            ? `${imp > 0 ? "+" : ""}${imp.toFixed(0)}% vs base`
                            : "—"}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
              <span>
                Bars show Test RMSE{activeUnit ? ` (${activeUnit})` : ""}. Switch
                to{" "}
                <button
                  type="button"
                  onClick={() => setLeaderboardView("full")}
                  className="font-medium text-primary underline-offset-2 hover:underline"
                >
                  Full metrics
                </button>{" "}
                for validation/test splits and hyperparameters.
              </span>
              {data.trained_at_utc ? (
                <span>
                  Updated{" "}
                  <TimestampPkt
                    timestamp={data.trained_at_utc}
                    timeZone={resolvedTimeZone}
                    mode="datetime"
                    className="font-medium text-foreground"
                  />
                </span>
              ) : null}
            </div>
          </div>
        ) : (
          /* Full sortable Val/Test metrics table */
          <Table
            aria-label={`${activeLabel} full model metrics`}
            containerClassName="rounded-md border border-border"
          >
            <TableHeader className="sticky top-0 z-10 bg-muted">
              <TableRow className="hover:bg-muted">
                {renderSortHead("model_family", "Model", "left")}
                {renderSortHead("val_rmse", "Val RMSE")}
                {renderSortHead("val_mae", "Val MAE")}
                {renderSortHead("val_r2", "Val R²")}
                {renderSortHead("test_rmse", "Test RMSE")}
                {renderSortHead("test_mae", "Test MAE")}
                {renderSortHead("test_r2", "Test R²")}
                {renderSortHead("improvement_over_naive_pct", "vs. Naive")}
                <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                  Config
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedLeaderboard.map((row) => {
                const rowKey = `${row.id}-${row.train_run_id}-${row.model_family}`
                const isExpanded = expandedRowIds.has(rowKey)
                const colorVar = getModelColorVar(row.model_family)
                const hyperEntries = Object.entries(row.hyperparameters ?? {})

                return (
                  <Fragment key={rowKey}>
                    <TableRow
                      className={cn(
                        "h-[52px] md:h-11",
                        row.is_champion &&
                          "bg-accent/70 text-foreground hover:bg-accent"
                      )}
                    >
                      <TableCell className="px-4 py-3 text-sm">
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            className="size-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: colorVar }}
                            aria-hidden="true"
                          />
                          <span className="font-semibold text-foreground">
                            {formatModelLabel(row.model_family)}
                          </span>
                          {row.is_champion ? (
                            <Badge
                              variant="default"
                              className="h-5 gap-1 px-2 text-[11px]"
                            >
                              <Award className="size-3" strokeWidth={1.75} />
                              <span>Champion</span>
                            </Badge>
                          ) : null}
                          {row.promoted_to_registry && !row.is_champion ? (
                            <Badge
                              variant="secondary"
                              className="h-5 px-2 text-[11px]"
                            >
                              Promoted
                            </Badge>
                          ) : null}
                        </div>
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        {formatMetricValue(row.val_rmse, 3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        {formatMetricValue(row.val_mae, 3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        {formatMetricValue(row.val_r2, 3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm font-semibold tabular-nums">
                        {formatMetricValue(row.test_rmse, 3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        {formatMetricValue(row.test_mae, 3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                        {formatMetricValue(row.test_r2, 3)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm font-medium tabular-nums">
                        {formatSignedPercent(row.improvement_over_naive_pct)}
                      </TableCell>
                      <TableCell className="px-4 py-3 text-right text-sm">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          aria-expanded={isExpanded}
                          onClick={() => toggleExpandedRow(rowKey)}
                          className="h-8 gap-1 px-2 text-xs text-foreground"
                        >
                          {isExpanded ? (
                            <ChevronDown className="size-3.5" strokeWidth={1.75} />
                          ) : (
                            <ChevronRight className="size-3.5" strokeWidth={1.75} />
                          )}
                          <span>Details</span>
                        </Button>
                      </TableCell>
                    </TableRow>

                    {isExpanded ? (
                      <TableRow className="bg-muted/70 hover:bg-muted/70">
                        <TableCell colSpan={9} className="p-4 whitespace-normal">
                          <div className="flex flex-col gap-3 rounded-lg bg-muted p-4 text-xs text-foreground">
                            <div className="flex flex-wrap items-center gap-x-6 gap-y-1">
                              <span>
                                Trained:{" "}
                                <TimestampPkt
                                  timestamp={row.trained_at_utc}
                                  timeZone={resolvedTimeZone}
                                  mode="datetime"
                                  className="font-medium text-foreground"
                                />
                              </span>
                              <span>
                                Samples:{" "}
                                <span className="font-semibold tabular-nums">
                                  {isFiniteNum(row.training_samples)
                                    ? row.training_samples.toLocaleString()
                                    : "—"}
                                </span>
                              </span>
                            </div>

                            {hyperEntries.length > 0 ? (
                              <div className="flex flex-wrap gap-2">
                                {hyperEntries.map(([paramKey, paramVal]) => (
                                  <span
                                    key={paramKey}
                                    className="rounded-sm bg-card px-2 py-1 text-xs text-foreground tabular-nums"
                                  >
                                    <span className="text-muted-foreground">
                                      {humanizeIdentifier(paramKey)}:
                                    </span>{" "}
                                    <span className="font-medium">
                                      {typeof paramVal === "object" &&
                                      paramVal !== null
                                        ? JSON.stringify(paramVal)
                                        : String(paramVal ?? "—")}
                                    </span>
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <p className="text-foreground/75">
                                Default configuration used for this model.
                              </p>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ) : null}
                  </Fragment>
                )
              })}
            </TableBody>
          </Table>
        )}
      </section>

      {/* Section 2: De-spaghettied Backtest Comparison Chart */}
      <ChartCard
        title={`How closely predictions track actual ${activeLabel.toLowerCase()}`}
        titleInfo="Shows model predictions on held-out test hours alongside real observations. Only Actual and the Champion model are shown by default to keep the chart easy to read."
        subtitle="Showing Actual vs. Champion by default — click any model pill to compare side-by-side"
        ariaLabel={`${activeLabel} model family comparison chart`}
        controls={
          presentSeriesMeta.length > 0 ? (
            <div
              role="group"
              aria-label="Toggle model lines"
              className="flex flex-wrap items-center gap-1.5"
            >
              {presentSeriesMeta.map((series) => {
                const isVisible = activeVisibleKeys.has(series.key)
                return (
                  <button
                    key={series.key}
                    type="button"
                    aria-pressed={isVisible}
                    onClick={() => toggleSeriesVisibility(series.key)}
                    className={cn(
                      "inline-flex h-8 items-center gap-1.5 rounded-full border px-2.5 text-xs font-medium transition-colors duration-120",
                      isVisible
                        ? "border-primary/50 bg-accent text-accent-foreground"
                        : "border-border bg-card text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <span
                      className="size-2 shrink-0 rounded-full"
                      style={{
                        backgroundColor: series.color,
                        opacity: isVisible ? 1 : 0.35,
                      }}
                      aria-hidden="true"
                    />
                    <span>{series.label}</span>
                  </button>
                )
              })}

              {presentSeriesMeta.length > 2 ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-8 px-2.5 text-xs text-muted-foreground hover:text-foreground"
                  onClick={() =>
                    setCustomVisibleSeries(
                      allSeriesSelected
                        ? null
                        : new Set(presentSeriesMeta.map((s) => s.key))
                    )
                  }
                >
                  {allSeriesSelected ? "Champion only" : "Compare all"}
                </Button>
              ) : null}
            </div>
          ) : undefined
        }
        tableColumns={comparisonTableColumns}
        tableRows={comparisonRows}
        emptyTableMessage={`No aligned comparison points recorded for ${activeLabel}.`}
      >
        {comparisonRows.length === 0 ? (
          <EmptyState
            icon={Layers}
            title={`No comparison curves for ${activeLabel}`}
            description="Aligned model-family test curves appear here after the training pipeline saves evaluation series."
            className="border-0 bg-muted shadow-none"
          />
        ) : visibleSeriesMeta.length === 0 ? (
          <EmptyState
            icon={Layers}
            title="All model lines hidden"
            description="Select one or more series pills above to display comparison curves."
            actionLabel="Show Champion vs. Actual"
            onAction={() => setCustomVisibleSeries(null)}
            className="border-0 bg-muted shadow-none"
          />
        ) : (
          <div>
            <ChartContainer
              config={comparisonChartConfig}
              className="h-[240px] w-full aspect-auto md:h-[320px]"
            >
              <ComposedChart
                data={comparisonRows}
                margin={{ top: 12, right: 12, bottom: 4, left: 0 }}
              >
                {comparisonAqiBands.map((band) => (
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
                  tickFormatter={(iso) =>
                    formatLocalTime(String(iso), resolvedTimeZone)
                  }
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                  minTickGap={34}
                />

                <YAxis
                  domain={[comparisonMin, comparisonMax]}
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
                      | ComparisonChartRow
                      | undefined
                    if (!row) {
                      return null
                    }

                    const tooltipRows: TooltipEntryRow[] = []
                    for (const series of visibleSeriesMeta) {
                      const val = row[series.key]
                      if (val !== null) {
                        tooltipRows.push({
                          key: series.key,
                          label: series.label,
                          valueText: formatValueWithUnit(val, activeUnit, 1),
                          color: series.color,
                          dashed: series.dashed,
                        })
                      }
                    }

                    const aqiProbe = row.actual ?? row.sarimax ?? row.random_forest
                    return (
                      <SkyCastChartTooltipCard
                        timestampUtc={row.timestamp_utc}
                        timeZone={resolvedTimeZone}
                        rows={tooltipRows}
                        aqiValue={activeMetric === "us_aqi" ? aqiProbe : null}
                      />
                    )
                  }}
                />

                {visibleSeriesMeta.map((series) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    stroke={series.color}
                    strokeWidth={series.key === "actual" ? 2.5 : 2}
                    strokeDasharray={series.dashed ? "6 4" : undefined}
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 2 }}
                    connectNulls={false}
                    isAnimationActive={shouldAnimateComparison}
                    animationDuration={600}
                    animationEasing="ease-out"
                  />
                ))}
              </ComposedChart>
            </ChartContainer>

            <ChartLegendBar items={comparisonLegendItems} />
          </div>
        )}
      </ChartCard>

      {/* Section 3: Focused Full-Width Explainability Panel (SHAP & LIME) */}
      <ExplainabilityPanel
        key={activeMetric}
        activeLabel={activeLabel}
        activeUnit={activeUnit}
        resolvedTimeZone={resolvedTimeZone}
        activeChampionFamily={activeChampion?.model_family ?? null}
        shapGlobalRecords={shapGlobalRecords}
        shapLocalRecords={shapLocalRecords}
        limeLocalRecords={limeLocalRecords}
      />
    </div>
  )
}
