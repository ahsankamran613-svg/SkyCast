import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Award,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Layers,
  Sparkles,
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
  fetchModelsLeaderboard,
  type ExplanationRecord,
  type MetricSpecsMap,
  type ModelRegistryMetricRow,
  type ModelsLeaderboardResponse,
  type ShapGlobalAttribution,
  type ShapLocalAttribution,
} from "@/lib/api"
import { formatLocalDateTime, formatLocalTime, parseIsoDate } from "@/lib/time"
import {
  formatFeatureName,
  formatLimeRule,
  formatMetricValue,
  formatModelLabel,
  getMetricSpec,
  getModelColorVar,
  humanizeIdentifier,
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
type ExplainMethod = "shap" | "lime"

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
    label: "24h naive baseline",
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

interface ShapGlobalRow extends Record<string, unknown> {
  feature: string
  label: string
  magnitude: number
  signedMean: number | null
  meanAbsShap: number | null
  importance: number | null
  referenceValue: number | null
}

interface ShapLocalWaterfallRow extends Record<string, unknown> {
  feature: string
  label: string
  contribution: number
  start: number
  end: number
  value: number | null
  scaledValue: number | null
}

interface LimeRuleRow extends Record<string, unknown> {
  feature: string
  label: string
  rule: string
  weight: number
  value: number | null
  scaledValue: number | null
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

function formatSignedValue(value: number | null | undefined, decimals = 3): string {
  if (!isFiniteNum(value)) {
    return "—"
  }
  const prefix = value > 0 ? "+" : ""
  return `${prefix}${value.toFixed(decimals)}`
}

function formatSignedPercent(value: number | null | undefined): string {
  if (!isFiniteNum(value)) {
    return "—"
  }
  const prefix = value > 0 ? "+" : ""
  return `${prefix}${value.toFixed(1)}%`
}

function formatFeatureLabel(feature: string): string {
  return formatFeatureName(feature)
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
  const [sortKey, setSortKey] = useState<LeaderboardSortKey>("test_rmse")
  const [sortDir, setSortDir] = useState<SortDirection>("asc")
  const [expandedRowIds, setExpandedRowIds] = useState<Set<string>>(new Set())

  const [hiddenSeries, setHiddenSeries] = useState<Set<ComparisonSeriesKey>>(
    new Set()
  )
  const [explainMethod, setExplainMethod] = useState<ExplainMethod>("shap")
  const [selectedExplainModel, setSelectedExplainModel] = useState<string>("")
  const [selectedShapLocalIndex, setSelectedShapLocalIndex] = useState<number>(0)
  const [selectedLimeLocalIndex, setSelectedLimeLocalIndex] = useState<number>(0)
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

  const activeSpec = getMetricSpec(activeMetric, metricSpecs)
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

  const visibleSeriesMeta = useMemo(
    () => presentSeriesMeta.filter((series) => !hiddenSeries.has(series.key)),
    [presentSeriesMeta, hiddenSeries]
  )

  const toggleSeriesVisibility = useCallback((key: ComparisonSeriesKey) => {
    setHiddenSeries((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }, [])

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

  // Explainability records
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

  const explainedModels = useMemo(() => {
    const models = new Set<string>()
    for (const rec of shapGlobalRecords) models.add(rec.model_family)
    for (const rec of shapLocalRecords) models.add(rec.model_family)
    for (const rec of limeLocalRecords) models.add(rec.model_family)
    return Array.from(models)
  }, [shapGlobalRecords, shapLocalRecords, limeLocalRecords])

  const activeChampionFamily = activeChampion?.model_family ?? null

  const activeExplainModel = useMemo(() => {
    if (selectedExplainModel && explainedModels.includes(selectedExplainModel)) {
      return selectedExplainModel
    }
    if (
      activeChampionFamily &&
      explainedModels.includes(activeChampionFamily)
    ) {
      return activeChampionFamily
    }
    return explainedModels[0] ?? ""
  }, [selectedExplainModel, explainedModels, activeChampionFamily])

  const modelShapGlobalRecord = useMemo<
    ExplanationRecord<ShapGlobalAttribution> | null
  >(() => {
    const matches = shapGlobalRecords.filter(
      (rec) => rec.model_family === activeExplainModel
    )
    return matches[0] ?? null
  }, [shapGlobalRecords, activeExplainModel])

  const modelShapLocalRecords = useMemo(() => {
    return shapLocalRecords.filter(
      (rec) => rec.model_family === activeExplainModel
    )
  }, [shapLocalRecords, activeExplainModel])

  const activeShapLocalRecord =
    modelShapLocalRecords[
      Math.min(selectedShapLocalIndex, Math.max(0, modelShapLocalRecords.length - 1))
    ] ?? null

  const modelLimeLocalRecords = useMemo(() => {
    return limeLocalRecords.filter(
      (rec) => rec.model_family === activeExplainModel
    )
  }, [limeLocalRecords, activeExplainModel])

  const activeLimeLocalRecord =
    modelLimeLocalRecords[
      Math.min(selectedLimeLocalIndex, Math.max(0, modelLimeLocalRecords.length - 1))
    ] ?? null

  const shapGlobalRows = useMemo<ShapGlobalRow[]>(() => {
    if (!modelShapGlobalRecord?.feature_attributions?.length) {
      return []
    }
    return modelShapGlobalRecord.feature_attributions
      .map((attr) => {
        const meanAbs = isFiniteNum(attr.mean_abs_shap)
          ? Math.abs(attr.mean_abs_shap)
          : null
        const imp = isFiniteNum(attr.importance) ? Math.abs(attr.importance) : null
        const signedMean = isFiniteNum(attr.mean_shap) ? attr.mean_shap : null
        const magnitude =
          meanAbs ?? imp ?? (signedMean !== null ? Math.abs(signedMean) : 0)
        return {
          feature: attr.feature,
          label: formatFeatureLabel(attr.feature),
          magnitude,
          signedMean,
          meanAbsShap: isFiniteNum(attr.mean_abs_shap) ? attr.mean_abs_shap : null,
          importance: isFiniteNum(attr.importance) ? attr.importance : null,
          referenceValue: isFiniteNum(attr.value) ? attr.value : null,
        }
      })
      .filter((row) => row.magnitude > 0 || row.signedMean !== null)
      .sort((a, b) => b.magnitude - a.magnitude)
      .slice(0, 10)
  }, [modelShapGlobalRecord])

  const maxShapGlobalMag = useMemo(() => {
    if (shapGlobalRows.length === 0) return 0
    return Math.max(...shapGlobalRows.map((row) => row.magnitude))
  }, [shapGlobalRows])

  const {
    waterfallRows,
    waterfallDomainMin,
    waterfallDomainMax,
    waterfallBaseValue,
    waterfallTopSum,
  } = useMemo(() => {
    const attrs = activeShapLocalRecord?.feature_attributions ?? []
    if (attrs.length === 0) {
      return {
        waterfallRows: [] as ShapLocalWaterfallRow[],
        waterfallDomainMin: 0,
        waterfallDomainMax: 1,
        waterfallBaseValue: 0,
        waterfallTopSum: 0,
      }
    }

    const sortedTop = [...attrs]
      .map((attr) => {
        const val = isFiniteNum(attr.shap_value)
          ? attr.shap_value
          : isFiniteNum(attr.contribution)
            ? attr.contribution
            : isFiniteNum(attr.importance)
              ? attr.importance
              : null
        return { attr, val }
      })
      .filter((item): item is { attr: ShapLocalAttribution; val: number } =>
        isFiniteNum(item.val)
      )
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
      .slice(0, 10)

    const base = isFiniteNum(activeShapLocalRecord?.base_value)
      ? activeShapLocalRecord.base_value
      : 0

    let running = base
    let minVal = base
    let maxVal = base
    let topSum = 0

    const rows: ShapLocalWaterfallRow[] = sortedTop.map(({ attr, val }) => {
      const start = running
      const end = running + val
      running = end
      topSum += val
      minVal = Math.min(minVal, start, end)
      maxVal = Math.max(maxVal, start, end)
      return {
        feature: attr.feature,
        label: formatFeatureLabel(attr.feature),
        contribution: val,
        start,
        end,
        value: isFiniteNum(attr.value) ? attr.value : null,
        scaledValue: isFiniteNum(attr.scaled_value) ? attr.scaled_value : null,
      }
    })

    if (isFiniteNum(activeShapLocalRecord?.predicted_value)) {
      minVal = Math.min(minVal, activeShapLocalRecord.predicted_value)
      maxVal = Math.max(maxVal, activeShapLocalRecord.predicted_value)
    }

    if (minVal === maxVal) {
      maxVal = minVal + 1
    }

    return {
      waterfallRows: rows,
      waterfallDomainMin: minVal,
      waterfallDomainMax: maxVal,
      waterfallBaseValue: base,
      waterfallTopSum: topSum,
    }
  }, [activeShapLocalRecord])

  const limeRows = useMemo<LimeRuleRow[]>(() => {
    const attrs = activeLimeLocalRecord?.feature_attributions ?? []
    return attrs
      .map((attr) => {
        const w = isFiniteNum(attr.weight)
          ? attr.weight
          : isFiniteNum(attr.contribution)
            ? attr.contribution
            : isFiniteNum(attr.importance)
              ? attr.importance
              : null
        return {
          feature: attr.feature,
          label: formatFeatureLabel(attr.feature),
          rule: attr.rule?.trim()
            ? formatLimeRule(attr.rule)
            : formatFeatureLabel(attr.feature),
          weight: w,
          value: isFiniteNum(attr.value) ? attr.value : null,
          scaledValue: isFiniteNum(attr.scaled_value) ? attr.scaled_value : null,
        }
      })
      .filter((item): item is LimeRuleRow => isFiniteNum(item.weight))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 10)
  }, [activeLimeLocalRecord])

  const maxLimeAbs = useMemo(() => {
    if (limeRows.length === 0) return 0
    return Math.max(...limeRows.map((row) => Math.abs(row.weight)))
  }, [limeRows])

  const shapGlobalTableColumns = useMemo<ChartTableColumn<ShapGlobalRow>[]>(
    () => [
      { key: "label", header: "Feature" },
      {
        key: "magnitude",
        header: "Mean |SHAP|",
        align: "right",
        render: (row) => formatMetricValue(row.magnitude, 4),
      },
      {
        key: "signedMean",
        header: "Mean signed SHAP",
        align: "right",
        render: (row) => formatSignedValue(row.signedMean, 4),
      },
      {
        key: "referenceValue",
        header: "Reference value",
        align: "right",
        render: (row) => formatMetricValue(row.referenceValue, 2),
      },
    ],
    []
  )

  const shapLocalTableColumns = useMemo<
    ChartTableColumn<ShapLocalWaterfallRow>[]
  >(
    () => [
      { key: "label", header: "Feature" },
      {
        key: "value",
        header: "Observed value",
        align: "right",
        render: (row) => formatMetricValue(row.value, 2),
      },
      {
        key: "contribution",
        header: "SHAP contribution",
        align: "right",
        render: (row) => formatSignedValue(row.contribution, 4),
      },
      {
        key: "end",
        header: "Cumulative output",
        align: "right",
        render: (row) => formatMetricValue(row.end, 3),
      },
    ],
    []
  )

  const limeTableColumns = useMemo<ChartTableColumn<LimeRuleRow>[]>(
    () => [
      { key: "rule", header: "Condition rule" },
      {
        key: "value",
        header: "Observed value",
        align: "right",
        render: (row) => formatMetricValue(row.value, 2),
      },
      {
        key: "weight",
        header: "LIME weight",
        align: "right",
        render: (row) => formatSignedValue(row.weight, 4),
      },
    ],
    []
  )

  if (isLoading && !data) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-11 w-full animate-pulse rounded-lg bg-muted" />
        <div className="h-[116px] animate-pulse rounded-xl border border-border bg-muted" />
        <div className="h-[340px] animate-pulse rounded-xl border border-border bg-muted" />
        <div className="h-[360px] animate-pulse rounded-xl border border-border bg-muted" />
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
    explainedModels.length > 0

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
      return <ArrowUpDown className="size-3.5 text-muted-foreground" strokeWidth={1.75} />
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

  const waterfallSpan = Math.max(1e-9, waterfallDomainMax - waterfallDomainMin)
  const baseLinePct =
    ((waterfallBaseValue - waterfallDomainMin) / waterfallSpan) * 100

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
              Validation and test error rankings against the 24-hour naive
              baseline, aligned comparison curves, and SHAP/LIME feature drivers.
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
            setSelectedShapLocalIndex(0)
            setSelectedLimeLocalIndex(0)
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

      {/* Champion summary telemetry strip */}
      {activeChampion ? (
        <section
          aria-label="Champion model summary"
          className="flex flex-col justify-between gap-4 rounded-xl border border-primary/40 bg-accent p-4 text-accent-foreground shadow-(--shadow-card) md:flex-row md:items-center md:p-5"
        >
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="default"
                className="h-6 gap-1 px-2.5 text-xs font-semibold"
              >
                <Award className="size-3.5" strokeWidth={1.75} />
                <span>Champion for {activeLabel}</span>
              </Badge>
              {activeChampion.promoted_to_registry ? (
                <Badge
                  variant="secondary"
                  className="h-6 gap-1 bg-card px-2.5 text-xs font-medium text-foreground"
                >
                  <CheckCircle2 className="size-3.5 text-aqi-good-fg" strokeWidth={1.75} />
                  <span>Promoted to registry</span>
                </Badge>
              ) : null}
            </div>

            <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h3 className="font-display text-h2 font-semibold text-foreground">
                {formatModelLabel(activeChampion.model_family)}
              </h3>
            </div>

            <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-foreground/80">
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
                  {activeChampion.training_samples.toLocaleString()} training rows
                </span>
              ) : null}
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:gap-5">
            <div>
              <dt className="text-xs font-medium text-foreground/75">Test RMSE</dt>
              <dd className="mt-0.5 flex items-baseline gap-1">
                <span className="font-display text-[1.75rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                  {formatMetricValue(activeChampion.test_rmse, 3)}
                </span>
                {activeUnit ? (
                  <span className="text-xs font-medium text-foreground/75">
                    {activeUnit}
                  </span>
                ) : null}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-foreground/75">Test MAE</dt>
              <dd className="mt-0.5 flex items-baseline gap-1">
                <span className="font-display text-[1.75rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                  {formatMetricValue(activeChampion.test_mae, 3)}
                </span>
                {activeUnit ? (
                  <span className="text-xs font-medium text-foreground/75">
                    {activeUnit}
                  </span>
                ) : null}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-foreground/75">Test R²</dt>
              <dd className="mt-0.5 font-display text-[1.75rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                {formatMetricValue(activeChampion.test_r2, 3)}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-foreground/75">
                vs. 24h naive
              </dt>
              <dd className="mt-0.5 font-display text-[1.75rem]/8 font-semibold text-foreground tabular-nums md:text-stat">
                {formatSignedPercent(activeChampion.improvement_over_naive_pct)}
              </dd>
            </div>
          </dl>
        </section>
      ) : null}

      {/* Section 1: Leaderboard */}
      <section className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-base/6 font-semibold text-card-foreground">
              {activeLabel} model leaderboard
            </h3>
            <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
              {sortedLeaderboard.length}{" "}
              {sortedLeaderboard.length === 1 ? "model" : "models"} evaluated
            </p>
          </div>
          {data.trained_at_utc ? (
            <span className="text-xs text-muted-foreground">
              Latest training{" "}
              <TimestampPkt
                timestamp={data.trained_at_utc}
                timeZone={resolvedTimeZone}
                mode="datetime"
                className="font-medium text-foreground"
              />
            </span>
          ) : null}
        </div>

        {sortedLeaderboard.length === 0 ? (
          <EmptyState
            icon={Layers}
            title={`No ${activeLabel.toLowerCase()} training metrics yet`}
            description="Training metrics appear here after the training pipeline evaluates candidate models on validation and test splits."
            className="border-0 bg-muted shadow-none"
          />
        ) : (
          <>
            {/* Visual Test RMSE comparison bar strip (28px bars, 12px gap) */}
            {benchmarkBars.length > 0 && maxBenchmarkRmse > 0 ? (
              <div
                aria-label={`${activeLabel} test RMSE bar comparison`}
                className="flex flex-col gap-3 rounded-lg bg-muted p-4"
              >
                <div className="flex items-center justify-between text-xs font-medium text-foreground">
                  <span>Test RMSE comparison (lower is better)</span>
                  <span className="text-foreground/75">
                    {activeUnit ? `Units: ${activeUnit}` : "Model error"}
                  </span>
                </div>
                <div className="flex flex-col gap-3">
                  {benchmarkBars.map((row) => {
                    const widthPct = Math.max(
                      4,
                      Math.min(100, (row.test_rmse / maxBenchmarkRmse) * 100)
                    )
                    const colorVar = getModelColorVar(row.model_family)
                    const isBest =
                      row.is_champion ||
                      (activeChampion !== null &&
                        row.model_family === activeChampion.model_family)

                    return (
                      <div
                        key={row.model_family}
                        className={cn(
                          "grid grid-cols-[minmax(110px,150px)_minmax(0,1fr)_auto] items-center gap-3 rounded-md px-2 py-1",
                          isBest && "bg-accent"
                        )}
                      >
                        <div className="flex min-w-0 items-center gap-2 text-xs font-medium text-foreground">
                          <span
                            className="size-2 shrink-0 rounded-full"
                            style={{ backgroundColor: colorVar }}
                            aria-hidden="true"
                          />
                          <span className="truncate">
                            {formatModelLabel(row.model_family)}
                          </span>
                        </div>

                        <div className="h-7 w-full overflow-hidden rounded-sm bg-card">
                          <div
                            className="h-full rounded-sm transition-all duration-120"
                            style={{
                              width: `${widthPct}%`,
                              backgroundColor: colorVar,
                            }}
                          />
                        </div>

                        <div className="flex items-baseline gap-2 text-right text-xs tabular-nums">
                          <span className="font-semibold text-foreground">
                            {formatMetricValue(row.test_rmse, 3)}
                          </span>
                          <span className="hidden w-16 text-foreground/75 sm:inline">
                            {formatSignedPercent(row.improvement_over_naive_pct)}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ) : null}

            {/* Full metrics table */}
            <Table
              aria-label={`${activeLabel} model leaderboard`}
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
          </>
        )}
      </section>

      {/* Section 2: Aligned Model-Family Comparison Chart */}
      <ChartCard
        title={`${activeLabel} model-family backtest comparison`}
        subtitle="Aligned evaluation trajectories across model families versus ground truth and 24h naive baseline"
        ariaLabel={`${activeLabel} model family comparison chart`}
        controls={
          presentSeriesMeta.length > 0 ? (
            <div
              role="group"
              aria-label="Toggle model lines"
              className="flex flex-wrap items-center gap-1.5"
            >
              {presentSeriesMeta.map((series) => {
                const isVisible = !hiddenSeries.has(series.key)
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
            description="Select one or more series toggles above to display comparison curves."
            actionLabel="Show all series"
            onAction={() => setHiddenSeries(new Set())}
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
                    strokeWidth={series.key === "actual" ? 2.25 : 2}
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

      {/* Section 3: SHAP / LIME Explanations (borderless section so inner ChartCards avoid double borders per Spec 5.2) */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-display text-[1.25rem]/6.5 font-semibold text-foreground md:text-h2">
              Prediction drivers (SHAP and LIME)
            </h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Orange bars push the {activeLabel.toLowerCase()} forecast higher;
              blue bars push it lower.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {explainedModels.length > 0 ? (
              <Select
                value={activeExplainModel}
                onValueChange={(val) => {
                  setSelectedExplainModel(val)
                  setSelectedShapLocalIndex(0)
                  setSelectedLimeLocalIndex(0)
                }}
              >
                <SelectTrigger
                  aria-label="Select explained model family"
                  className="h-9 min-w-[156px] bg-card text-xs"
                >
                  <SelectValue placeholder="Model family" />
                </SelectTrigger>
                <SelectContent position="popper">
                  {explainedModels.map((model) => (
                    <SelectItem key={model} value={model}>
                      {formatModelLabel(model)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : null}

            {/* 36px pill-shaped segmented toggle per Spec 6.1 */}
            <div
              role="tablist"
              aria-label="Explainability method"
              className="inline-flex h-9 items-center rounded-full bg-muted p-1"
            >
              {(["shap", "lime"] as const).map((method) => {
                const active = explainMethod === method
                return (
                  <button
                    key={method}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => setExplainMethod(method)}
                    className={cn(
                      "inline-flex h-7 items-center rounded-full px-3.5 text-xs font-medium uppercase transition-colors duration-120",
                      active
                        ? "bg-card text-foreground shadow-(--shadow-card)"
                        : "text-foreground/75 hover:text-foreground"
                    )}
                  >
                    {method}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {explainedModels.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title={`No SHAP or LIME explanations for ${activeLabel}`}
            description="Feature attribution records appear here after the training pipeline computes SHAP and LIME explanations."
            className="border-0 bg-muted shadow-none"
          />
        ) : explainMethod === "shap" ? (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            {/* Global SHAP Top-10 */}
            <ChartCard
              title="Global feature importance (SHAP)"
              subtitle={`Top ${shapGlobalRows.length || 10} overall drivers for ${formatModelLabel(activeExplainModel)}`}
              ariaLabel={`Global SHAP feature importance for ${formatModelLabel(activeExplainModel)}`}
              tableColumns={shapGlobalTableColumns}
              tableRows={shapGlobalRows}
              emptyTableMessage="No global SHAP attributions available for this model."
            >
              {shapGlobalRows.length === 0 ? (
                <EmptyState
                  icon={Sparkles}
                  title="No global SHAP summary"
                  description={`Global feature attributions were not recorded for ${formatModelLabel(activeExplainModel)}.`}
                  className="border-0 bg-muted shadow-none"
                />
              ) : (
                <div className="flex flex-col gap-2">
                  {shapGlobalRows.map((row) => {
                    const widthPct =
                      maxShapGlobalMag > 0
                        ? Math.max(4, (row.magnitude / maxShapGlobalMag) * 100)
                        : 0
                    const isNegative =
                      row.signedMean !== null && row.signedMean < 0
                    const barColor = isNegative
                      ? "var(--shap-neg)"
                      : "var(--shap-pos)"

                    return (
                      <Tooltip key={row.feature}>
                        <TooltipTrigger asChild>
                          <div
                            tabIndex={0}
                            className="grid grid-cols-[minmax(110px,155px)_minmax(0,1fr)_auto] items-center gap-3 rounded-sm px-1 py-0.5 hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                          >
                            <span
                              className="truncate text-xs font-medium text-foreground"
                              title={row.label}
                            >
                              {row.label}
                            </span>
                            <div className="h-7 w-full overflow-hidden rounded-sm bg-muted">
                              <div
                                className="h-full rounded-sm transition-all duration-120"
                                style={{
                                  width: `${widthPct}%`,
                                  backgroundColor: barColor,
                                }}
                              />
                            </div>
                            <span className="w-16 text-right text-xs font-semibold text-foreground tabular-nums">
                              {formatMetricValue(row.magnitude, 3)}
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent className="flex flex-col gap-0.5 text-left">
                          <span className="font-semibold text-popover-foreground">
                            {row.label}
                          </span>
                          <span className="text-muted-foreground tabular-nums">
                            Mean |SHAP|: {formatMetricValue(row.magnitude, 4)}
                          </span>
                          {row.signedMean !== null ? (
                            <span className="text-muted-foreground tabular-nums">
                              Mean signed SHAP: {formatSignedValue(row.signedMean, 4)}
                            </span>
                          ) : null}
                          {row.referenceValue !== null ? (
                            <span className="text-muted-foreground tabular-nums">
                              Mean value: {formatMetricValue(row.referenceValue, 2)}
                            </span>
                          ) : null}
                        </TooltipContent>
                      </Tooltip>
                    )
                  })}
                </div>
              )}
            </ChartCard>

            {/* Local SHAP Waterfall */}
            <ChartCard
              title="Local prediction waterfall (SHAP)"
              subtitle={
                activeShapLocalRecord
                  ? `Top ${waterfallRows.length} local contributions for ${formatModelLabel(activeExplainModel)}`
                  : `Local feature waterfall for ${formatModelLabel(activeExplainModel)}`
              }
              ariaLabel={`Local SHAP waterfall for ${formatModelLabel(activeExplainModel)}`}
              controls={
                modelShapLocalRecords.length > 1 ? (
                  <Select
                    value={String(selectedShapLocalIndex)}
                    onValueChange={(val) => setSelectedShapLocalIndex(Number(val))}
                  >
                    <SelectTrigger
                      aria-label="Select local SHAP reference timestamp"
                      className="h-8 min-w-[160px] bg-card text-xs"
                    >
                      <SelectValue placeholder="Reference time" />
                    </SelectTrigger>
                    <SelectContent position="popper">
                      {modelShapLocalRecords.map((rec, idx) => (
                        <SelectItem key={`${rec.id}-${idx}`} value={String(idx)}>
                          {rec.reference_timestamp_utc
                            ? formatLocalDateTime(
                                rec.reference_timestamp_utc,
                                resolvedTimeZone,
                                { includeZone: false }
                              )
                            : `Sample #${idx + 1}`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : undefined
              }
              subHeader={
                activeShapLocalRecord ? (
                  <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <dt className="text-xs text-foreground/75">Base value</dt>
                      <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                        {formatValueWithUnit(
                          activeShapLocalRecord.base_value,
                          activeUnit,
                          2
                        )}
                      </dd>
                    </div>
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <dt className="text-xs text-foreground/75">
                        Model prediction
                      </dt>
                      <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                        {formatValueWithUnit(
                          activeShapLocalRecord.predicted_value,
                          activeUnit,
                          2
                        )}
                      </dd>
                    </div>
                    <div className="col-span-2 rounded-lg bg-muted px-3 py-2 sm:col-span-1">
                      <dt className="text-xs text-foreground/75">
                        Reference hour
                      </dt>
                      <dd className="mt-0.5 text-xs font-medium text-foreground">
                        <TimestampPkt
                          timestamp={activeShapLocalRecord.reference_timestamp_utc}
                          timeZone={resolvedTimeZone}
                          mode="datetime"
                        />
                      </dd>
                    </div>
                  </dl>
                ) : undefined
              }
              tableColumns={shapLocalTableColumns}
              tableRows={waterfallRows}
              emptyTableMessage="No local SHAP attributions available for this model."
            >
              {waterfallRows.length === 0 ? (
                <EmptyState
                  icon={Sparkles}
                  title="No local SHAP waterfall"
                  description={`Local instance explanations were not recorded for ${formatModelLabel(activeExplainModel)}.`}
                  className="border-0 bg-muted shadow-none"
                />
              ) : (
                <div className="flex flex-col gap-2">
                  {waterfallRows.map((row) => {
                    const low = Math.min(row.start, row.end)
                    const high = Math.max(row.start, row.end)
                    const leftPct = Math.max(
                      0,
                      Math.min(100, ((low - waterfallDomainMin) / waterfallSpan) * 100)
                    )
                    const widthPct = Math.max(
                      2,
                      Math.min(100 - leftPct, ((high - low) / waterfallSpan) * 100)
                    )
                    const isPositive = row.contribution >= 0

                    return (
                      <Tooltip key={row.feature}>
                        <TooltipTrigger asChild>
                          <div
                            tabIndex={0}
                            className="grid grid-cols-[minmax(110px,155px)_minmax(0,1fr)_auto] items-center gap-3 rounded-sm px-1 py-0.5 hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                          >
                            <div className="min-w-0">
                              <div
                                className="truncate text-xs font-medium text-foreground"
                                title={row.label}
                              >
                                {row.label}
                              </div>
                              {row.value !== null ? (
                                <div className="text-[11px] text-muted-foreground tabular-nums">
                                  Val: {formatMetricValue(row.value, 2)}
                                </div>
                              ) : null}
                            </div>

                            <div className="relative h-7 w-full overflow-hidden rounded-sm bg-muted">
                              <div
                                aria-hidden="true"
                                className="absolute inset-y-0 w-px bg-foreground/30"
                                style={{
                                  left: `${Math.max(0, Math.min(100, baseLinePct))}%`,
                                }}
                              />
                              <div
                                className="absolute top-0 bottom-0 rounded-sm transition-all duration-120"
                                style={{
                                  left: `${leftPct}%`,
                                  width: `${widthPct}%`,
                                  backgroundColor: isPositive
                                    ? "var(--shap-pos)"
                                    : "var(--shap-neg)",
                                }}
                              />
                            </div>

                            <span
                              className={cn(
                                "w-16 text-right text-xs font-semibold tabular-nums",
                                isPositive ? "text-shap-pos" : "text-shap-neg"
                              )}
                            >
                              {formatSignedValue(row.contribution, 3)}
                            </span>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent className="flex flex-col gap-0.5 text-left">
                          <span className="font-semibold text-popover-foreground">
                            {row.label}
                          </span>
                          <span className="text-muted-foreground tabular-nums">
                            SHAP contribution:{" "}
                            {formatSignedValue(row.contribution, 4)}
                          </span>
                          <span className="text-muted-foreground tabular-nums">
                            Running step: {formatMetricValue(row.start, 3)} to{" "}
                            {formatMetricValue(row.end, 3)}
                          </span>
                          {row.value !== null ? (
                            <span className="text-muted-foreground tabular-nums">
                              Feature value: {formatMetricValue(row.value, 2)}
                            </span>
                          ) : null}
                        </TooltipContent>
                      </Tooltip>
                    )
                  })}

                  <div className="mt-1 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-2 text-xs text-muted-foreground tabular-nums">
                    <span>
                      Top {waterfallRows.length} features net:{" "}
                      <span className="font-medium text-foreground">
                        {formatSignedValue(waterfallTopSum, 3)}
                      </span>
                    </span>
                    <div className="flex items-center gap-4">
                      <span className="inline-flex items-center gap-1.5">
                        <span
                          className="size-2 rounded-full"
                          style={{ backgroundColor: "var(--shap-pos)" }}
                          aria-hidden="true"
                        />
                        Pushes up
                      </span>
                      <span className="inline-flex items-center gap-1.5">
                        <span
                          className="size-2 rounded-full"
                          style={{ backgroundColor: "var(--shap-neg)" }}
                          aria-hidden="true"
                        />
                        Pushes down
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </ChartCard>
          </div>
        ) : (
          /* LIME mode */
          <ChartCard
            title="Local interpretable explanation (LIME)"
            subtitle={`Top ${limeRows.length || 10} threshold rules for ${formatModelLabel(activeExplainModel)}`}
            ariaLabel={`LIME local explanation for ${formatModelLabel(activeExplainModel)}`}
            controls={
              modelLimeLocalRecords.length > 1 ? (
                <Select
                  value={String(selectedLimeLocalIndex)}
                  onValueChange={(val) => setSelectedLimeLocalIndex(Number(val))}
                >
                  <SelectTrigger
                    aria-label="Select local LIME reference timestamp"
                    className="h-8 min-w-[160px] bg-card text-xs"
                  >
                    <SelectValue placeholder="Reference time" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    {modelLimeLocalRecords.map((rec, idx) => (
                      <SelectItem key={`${rec.id}-${idx}`} value={String(idx)}>
                        {rec.reference_timestamp_utc
                          ? formatLocalDateTime(
                              rec.reference_timestamp_utc,
                              resolvedTimeZone,
                              { includeZone: false }
                            )
                          : `Sample #${idx + 1}`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : undefined
            }
            subHeader={
              activeLimeLocalRecord ? (
                <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="text-xs text-foreground/75">Base value</dt>
                    <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                      {formatValueWithUnit(
                        activeLimeLocalRecord.base_value,
                        activeUnit,
                        2
                      )}
                    </dd>
                  </div>
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="text-xs text-foreground/75">
                      Model prediction
                    </dt>
                    <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                      {formatValueWithUnit(
                        activeLimeLocalRecord.predicted_value,
                        activeUnit,
                        2
                      )}
                    </dd>
                  </div>
                  <div className="col-span-2 rounded-lg bg-muted px-3 py-2 sm:col-span-1">
                    <dt className="text-xs text-foreground/75">Reference hour</dt>
                    <dd className="mt-0.5 text-xs font-medium text-foreground">
                      <TimestampPkt
                        timestamp={activeLimeLocalRecord.reference_timestamp_utc}
                        timeZone={resolvedTimeZone}
                        mode="datetime"
                      />
                    </dd>
                  </div>
                </dl>
              ) : undefined
            }
            tableColumns={limeTableColumns}
            tableRows={limeRows}
            emptyTableMessage="No LIME explanations available for this model."
          >
            {limeRows.length === 0 ? (
              <EmptyState
                icon={Sparkles}
                title="No LIME rules recorded"
                description={`Local LIME surrogate weights were not recorded for ${formatModelLabel(activeExplainModel)}.`}
                className="border-0 bg-muted shadow-none"
              />
            ) : (
              <div className="flex flex-col gap-2">
                {limeRows.map((row, index) => {
                  const pct =
                    maxLimeAbs > 0
                      ? Math.max(3, (Math.abs(row.weight) / maxLimeAbs) * 50)
                      : 0
                  const isPositive = row.weight >= 0

                  return (
                    <Tooltip key={`${row.feature}-${index}`}>
                      <TooltipTrigger asChild>
                        <div
                          tabIndex={0}
                          className="grid grid-cols-[minmax(140px,240px)_minmax(0,1fr)_auto] items-center gap-3 rounded-sm px-1 py-0.5 hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                        >
                          <div className="min-w-0">
                            <div
                              className="truncate text-xs font-medium text-foreground"
                              title={row.rule}
                            >
                              {row.rule}
                            </div>
                            {row.value !== null ? (
                              <div className="text-[11px] text-muted-foreground tabular-nums">
                                {row.label}: {formatMetricValue(row.value, 2)}
                              </div>
                            ) : null}
                          </div>

                          <div className="relative h-7 w-full overflow-hidden rounded-sm bg-muted">
                            <div
                              aria-hidden="true"
                              className="absolute inset-y-0 left-1/2 w-px bg-foreground/30"
                            />
                            <div
                              className="absolute top-0 bottom-0 rounded-sm transition-all duration-120"
                              style={{
                                left: isPositive ? "50%" : `${50 - pct}%`,
                                width: `${pct}%`,
                                backgroundColor: isPositive
                                  ? "var(--shap-pos)"
                                  : "var(--shap-neg)",
                              }}
                            />
                          </div>

                          <span
                            className={cn(
                              "w-16 text-right text-xs font-semibold tabular-nums",
                              isPositive ? "text-shap-pos" : "text-shap-neg"
                            )}
                          >
                            {formatSignedValue(row.weight, 3)}
                          </span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="flex flex-col gap-0.5 text-left">
                        <span className="font-semibold text-popover-foreground">
                          {row.rule}
                        </span>
                        <span className="text-muted-foreground tabular-nums">
                          LIME weight: {formatSignedValue(row.weight, 4)}
                        </span>
                        {row.value !== null ? (
                          <span className="text-muted-foreground tabular-nums">
                            Observed value: {formatMetricValue(row.value, 2)}
                          </span>
                        ) : null}
                      </TooltipContent>
                    </Tooltip>
                  )
                })}
              </div>
            )}
          </ChartCard>
        )}
      </section>
    </div>
  )
}
