import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  AlertCircle,
  Grid2x2,
  Layers,
  ShieldAlert,
  TrendingUp,
} from "lucide-react"
import { cn } from "cn"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"

import { ChartCard, type ChartTableColumn } from "@/components/shared/chart-card"
import {
  ChartLegendBar,
  SkyCastChartTooltipCard,
  computeChartDomain,
  useFirstLoadChartAnimation,
  type TooltipEntryRow,
} from "@/components/shared/chart-primitives"
import {
  BootstrappingState,
  EmptyState,
  ErrorState,
} from "@/components/shared/empty-state"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Badge } from "@/components/ui/badge"
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
  fetchEdaSummary,
  type EdaDiurnalBucket,
  type EdaDistributionEntry,
  type EdaSeasonalDecompositionAvailable,
  type EdaSummaryResponse,
  type MetricSpecsMap,
} from "@/lib/api"
import { formatLocalDateTime, getTimeZoneLabel, parseIsoDate } from "@/lib/time"
import {
  formatMetricValue,
  getMetricColorVar,
  humanizeIdentifier,
  orderMetricKeys,
  resolveMetricLabel,
} from "@/lib/weather"

export interface EdaTabProps {
  city?: string | null
  timeZone?: string | null
  metricSpecsHint?: MetricSpecsMap | null
  coreMetricGroupsHint?: Record<string, string[]> | null
  refreshNonce?: number
  onShellRefresh?: () => void
}

type DiurnalAggregateMode = "mean" | "median"
type DecompositionViewMode = "cycle" | "trend" | "components"

const HEAT_TOKENS = [
  "var(--heat-1)",
  "var(--heat-2)",
  "var(--heat-3)",
  "var(--heat-4)",
  "var(--heat-5)",
] as const

interface HistogramChartRow extends Record<string, unknown> {
  label: string
  binStart: number
  binEnd: number
  count: number
}

interface DiurnalTableRow extends Record<string, unknown> {
  metricKey: string
  metricLabel: string
  unit: string
  minMean: number | null
  maxMean: number | null
  peakHour: number | null
  sampleTotal: number
}

interface CorrelationTableRow extends Record<string, unknown> {
  pair: string
  rowLabel: string
  colLabel: string
  correlation: number | null
}

interface SeasonalPointRow extends Record<string, unknown> {
  timestamp_utc: string
  observed: number | null
  trend: number | null
  seasonal: number | null
  resid: number | null
}

interface SeasonalCycleRow extends Record<string, unknown> {
  hour: number
  hourLabel: string
  seasonal: number | null
}

function isFiniteNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatAxisTick(value: unknown): string {
  const num = Number(value)
  if (!Number.isFinite(num)) return ""
  return Number.isInteger(num) ? String(num) : num.toFixed(1)
}

function formatBinRange(start: number, end: number): string {
  const s = Math.abs(start) >= 100 ? start.toFixed(0) : start.toFixed(1)
  const e = Math.abs(end) >= 100 ? end.toFixed(0) : end.toFixed(1)
  return `${s}–${e}`
}

function formatHourPadded(hour: number): string {
  return `${String(hour).padStart(2, "0")}:00`
}

function formatHourShort(hour: number): string {
  return String(hour).padStart(2, "0")
}

function pickHeatToken(value: number | null, min: number, max: number): string | null {
  if (!isFiniteNum(value)) return null
  if (max <= min) return HEAT_TOKENS[2]
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min)))
  const index = Math.min(4, Math.floor(ratio * 5))
  return HEAT_TOKENS[index]
}

function correlationCellStyle(r: number | null): {
  backgroundColor?: string
  label: string
} {
  if (!isFiniteNum(r)) {
    return { label: "—" }
  }
  const clamped = Math.max(-1, Math.min(1, r))
  const mixPct = Math.round(Math.abs(clamped) * 34)
  const targetVar = clamped >= 0 ? "var(--shap-pos)" : "var(--shap-neg)"
  return {
    backgroundColor: `color-mix(in srgb, ${targetVar} ${mixPct}%, var(--corr-neutral))`,
    label: clamped.toFixed(2),
  }
}

function formatHighlightPairLabel(
  key: string,
  metricSpecs?: MetricSpecsMap | null,
  distributions?: Record<string, EdaDistributionEntry>
): string {
  const separators = ["__vs__", "_vs_", "::", "|", ","]
  for (const sep of separators) {
    if (key.includes(sep)) {
      const parts = key.split(sep).map((p) => p.trim()).filter(Boolean)
      return parts
        .map((p) => resolveMetricLabel(p, metricSpecs, distributions))
        .join(" and ")
    }
  }
  return humanizeIdentifier(key)
}

function humanizeAuditSource(source: string | undefined): string {
  if (!source) return "Unavailable"
  if (source === "provided") return "Recorded by pipeline"
  if (source === "dataframe_derived") return "Derived from observations"
  if (source === "bootstrapping") return "Awaiting observations"
  if (source === "unavailable") return "Unavailable"
  return humanizeIdentifier(source)
}

function downsampleSeries<T>(rows: T[], maxPoints = 320): T[] {
  if (rows.length <= maxPoints) return rows
  const step = Math.ceil(rows.length / maxPoints)
  return rows.filter((_, idx) => idx % step === 0 || idx === rows.length - 1)
}

export function EdaTab({
  city,
  timeZone,
  metricSpecsHint,
  coreMetricGroupsHint,
  refreshNonce = 0,
  onShellRefresh,
}: EdaTabProps) {
  const [selectedDistKey, setSelectedDistKey] = useState<string>("")
  const [diurnalMode, setDiurnalMode] = useState<DiurnalAggregateMode>("mean")
  const [selectedDecompKey, setSelectedDecompKey] = useState<string>("")
  const [decompView, setDecompView] = useState<DecompositionViewMode>("cycle")
  const [localRetryNonce, setLocalRetryNonce] = useState<number>(0)

  const [data, setData] = useState<EdaSummaryResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isFetching, setIsFetching] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const satisfiedNonceRef = useRef<string | null>(null)

  useEffect(() => {
    const reqKey = `${refreshNonce}:${localRetryNonce}`
    if (satisfiedNonceRef.current === reqKey) {
      return
    }

    const controller = new AbortController()
    setIsFetching(true)
    setErrorMessage(null)

    void fetchEdaSummary(undefined, { signal: controller.signal })
      .then((response) => {
        if (controller.signal.aborted) return
        satisfiedNonceRef.current = reqKey
        setData(response)
        setIsLoading(false)
        setIsFetching(false)
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return
        if (err instanceof DOMException && err.name === "AbortError") return
        setErrorMessage(
          "Couldn't load exploratory analysis. Check your connection and refresh."
        )
        setIsLoading(false)
        setIsFetching(false)
      })

    return () => controller.abort()
  }, [refreshNonce, localRetryNonce])

  const handleRetry = useCallback(() => {
    satisfiedNonceRef.current = null
    if (onShellRefresh) {
      onShellRefresh()
    } else {
      setLocalRetryNonce((n) => n + 1)
    }
  }, [onShellRefresh])

  const resolvedTimeZone =
    data?.diurnal_heatmap?.timezone ?? data?.metadata?.timezone ?? timeZone ?? null
  const tzShortLabel = useMemo(() => {
    const refDate =
      parseIsoDate(data?.date_range_end) ??
      parseIsoDate(data?.computed_at_utc) ??
      new Date()
    return getTimeZoneLabel(refDate, resolvedTimeZone)
  }, [data?.date_range_end, data?.computed_at_utc, resolvedTimeZone])

  const distributions = useMemo(
    () => data?.distributions ?? {},
    [data?.distributions]
  )
  const distributionKeys = useMemo(() => {
    const keys = Object.keys(distributions)
    const ordered = orderMetricKeys(keys, coreMetricGroupsHint)
    // Put available distributions before missing/all_missing ones for smooth scanning.
    return [...ordered].sort((a, b) => {
      const aAvail = distributions[a]?.status === "available" ? 0 : 1
      const bAvail = distributions[b]?.status === "available" ? 0 : 1
      return aAvail - bAvail
    })
  }, [distributions, coreMetricGroupsHint])

  const activeDistKey = useMemo(() => {
    if (selectedDistKey && selectedDistKey in distributions) {
      return selectedDistKey
    }
    const available = distributionKeys.filter(
      (key) => distributions[key]?.status === "available"
    )
    if (available.includes("temperature_2m")) return "temperature_2m"
    if (available.includes("us_aqi")) return "us_aqi"
    return available[0] ?? distributionKeys[0] ?? ""
  }, [selectedDistKey, distributions, distributionKeys])

  const activeDist: EdaDistributionEntry | null = activeDistKey
    ? (distributions[activeDistKey] ?? null)
    : null
  const activeDistUnit = activeDist?.unit ?? ""

  const histogramRows = useMemo<HistogramChartRow[]>(() => {
    if (!activeDist || activeDist.status !== "available") return []
    return (activeDist.histogram ?? [])
      .filter((bin) => isFiniteNum(bin.bin_start) && isFiniteNum(bin.bin_end))
      .map((bin) => ({
        label: formatBinRange(bin.bin_start, bin.bin_end),
        binStart: bin.bin_start,
        binEnd: bin.bin_end,
        count: isFiniteNum(bin.count) ? bin.count : 0,
      }))
  }, [activeDist])

  const distChartColor = useMemo(() => {
    if (!activeDistKey) return "var(--primary)"
    return metricSpecsHint?.[activeDistKey]?.chart_token
      ? getMetricColorVar(activeDistKey, metricSpecsHint)
      : "var(--primary)"
  }, [activeDistKey, metricSpecsHint])

  const histogramTableColumns = useMemo<ChartTableColumn<HistogramChartRow>[]>(
    () => [
      {
        key: "label",
        header: `Range${activeDistUnit ? ` (${activeDistUnit})` : ""}`,
      },
      {
        key: "count",
        header: "Observations",
        align: "right",
        render: (row) => row.count.toLocaleString(),
      },
    ],
    [activeDistUnit]
  )

  // Diurnal heatmap data
  const diurnalHours = useMemo(() => {
    const raw = data?.diurnal_heatmap?.hours ?? []
    if (raw.length > 0) {
      return [...raw].sort((a, b) => a - b)
    }
    return Array.from({ length: 24 }, (_, idx) => idx)
  }, [data?.diurnal_heatmap?.hours])

  const diurnalMetricKeys = useMemo(() => {
    const keys = Object.keys(data?.diurnal_heatmap?.metrics ?? {})
    return orderMetricKeys(keys, coreMetricGroupsHint)
  }, [data?.diurnal_heatmap?.metrics, coreMetricGroupsHint])

  const diurnalRows = useMemo(() => {
    const metricsMap = data?.diurnal_heatmap?.metrics ?? {}
    return diurnalMetricKeys.map((metricKey) => {
      const buckets = metricsMap[metricKey] ?? []
      const byHour = new Map<number, EdaDiurnalBucket>()
      for (const bucket of buckets) {
        byHour.set(bucket.hour, bucket)
      }

      const values: number[] = []
      let peakHour: number | null = null
      let peakVal = -Infinity
      let sampleTotal = 0

      for (const hour of diurnalHours) {
        const b = byHour.get(hour)
        if (!b) continue
        sampleTotal += isFiniteNum(b.count) ? b.count : 0
        const v = diurnalMode === "median" ? b.median : b.mean
        if (isFiniteNum(v) && b.count > 0) {
          values.push(v)
          if (v > peakVal) {
            peakVal = v
            peakHour = hour
          }
        }
      }

      const minVal = values.length > 0 ? Math.min(...values) : null
      const maxVal = values.length > 0 ? Math.max(...values) : null

      return {
        metricKey,
        metricLabel: resolveMetricLabel(metricKey, metricSpecsHint, distributions),
        unit:
          distributions[metricKey]?.unit ??
          metricSpecsHint?.[metricKey]?.unit ??
          "",
        byHour,
        minVal,
        maxVal,
        peakHour,
        sampleTotal,
      }
    })
  }, [
    data?.diurnal_heatmap?.metrics,
    diurnalMetricKeys,
    diurnalHours,
    diurnalMode,
    metricSpecsHint,
    distributions,
  ])

  const diurnalTableRows = useMemo<DiurnalTableRow[]>(
    () =>
      diurnalRows.map((row) => ({
        metricKey: row.metricKey,
        metricLabel: row.metricLabel,
        unit: row.unit,
        minMean: row.minVal,
        maxMean: row.maxVal,
        peakHour: row.peakHour,
        sampleTotal: row.sampleTotal,
      })),
    [diurnalRows]
  )

  const diurnalTableColumns = useMemo<ChartTableColumn<DiurnalTableRow>[]>(
    () => [
      { key: "metricLabel", header: "Metric" },
      {
        key: "minMean",
        header: `Min ${diurnalMode}`,
        align: "right",
        render: (row) =>
          `${formatMetricValue(row.minMean, 1)}${row.unit ? ` ${row.unit}` : ""}`,
      },
      {
        key: "maxMean",
        header: `Max ${diurnalMode}`,
        align: "right",
        render: (row) =>
          `${formatMetricValue(row.maxMean, 1)}${row.unit ? ` ${row.unit}` : ""}`,
      },
      {
        key: "peakHour",
        header: `Peak hour (${tzShortLabel})`,
        align: "right",
        render: (row) =>
          row.peakHour !== null ? formatHourPadded(row.peakHour) : "—",
      },
      {
        key: "sampleTotal",
        header: "Samples",
        align: "right",
        render: (row) => row.sampleTotal.toLocaleString(),
      },
    ],
    [diurnalMode, tzShortLabel]
  )

  // Correlation matrix data
  const corrColumns = useMemo(
    () => data?.correlation_matrix?.columns ?? [],
    [data?.correlation_matrix?.columns]
  )
  const corrMatrix = useMemo(
    () => data?.correlation_matrix?.matrix ?? {},
    [data?.correlation_matrix?.matrix]
  )
  const corrHighlights = useMemo(() => {
    return Object.entries(data?.correlation_matrix?.highlights ?? {})
      .filter(([, val]) => isFiniteNum(val))
      .sort((a, b) => Math.abs(b[1] ?? 0) - Math.abs(a[1] ?? 0))
  }, [data?.correlation_matrix?.highlights])

  const corrTableRows = useMemo<CorrelationTableRow[]>(() => {
    const rows: CorrelationTableRow[] = []
    for (let i = 0; i < corrColumns.length; i++) {
      for (let j = i + 1; j < corrColumns.length; j++) {
        const rKey = corrColumns[i]
        const cKey = corrColumns[j]
        const val = corrMatrix?.[rKey]?.[cKey] ?? null
        const rowLabel = resolveMetricLabel(rKey, metricSpecsHint, distributions)
        const colLabel = resolveMetricLabel(cKey, metricSpecsHint, distributions)
        rows.push({
          pair: `${rowLabel} × ${colLabel}`,
          rowLabel,
          colLabel,
          correlation: isFiniteNum(val) ? val : null,
        })
      }
    }
    return rows.sort(
      (a, b) => Math.abs(b.correlation ?? 0) - Math.abs(a.correlation ?? 0)
    )
  }, [corrColumns, corrMatrix, metricSpecsHint, distributions])

  const corrTableColumns = useMemo<ChartTableColumn<CorrelationTableRow>[]>(
    () => [
      { key: "rowLabel", header: "Feature A" },
      { key: "colLabel", header: "Feature B" },
      {
        key: "correlation",
        header: `${humanizeIdentifier(data?.correlation_matrix?.method ?? "pearson")} r`,
        align: "right",
        render: (row) =>
          isFiniteNum(row.correlation) ? row.correlation.toFixed(3) : "—",
      },
    ],
    [data?.correlation_matrix?.method]
  )

  // Seasonal decomposition data
  const seasonalMap = useMemo(
    () => data?.seasonal_decomposition ?? {},
    [data?.seasonal_decomposition]
  )
  const seasonalKeys = useMemo(() => {
    const keys = Object.keys(seasonalMap)
    const ordered = orderMetricKeys(keys, coreMetricGroupsHint)
    return [...ordered].sort((a, b) => {
      const aAvail = seasonalMap[a]?.status === "available" ? 0 : 1
      const bAvail = seasonalMap[b]?.status === "available" ? 0 : 1
      return aAvail - bAvail
    })
  }, [seasonalMap, coreMetricGroupsHint])

  const activeDecompKey = useMemo(() => {
    if (selectedDecompKey && selectedDecompKey in seasonalMap) {
      return selectedDecompKey
    }
    const available = seasonalKeys.filter(
      (key) => seasonalMap[key]?.status === "available"
    )
    if (available.includes("temperature_2m")) return "temperature_2m"
    return available[0] ?? seasonalKeys[0] ?? ""
  }, [selectedDecompKey, seasonalMap, seasonalKeys])

  const activeDecomp = activeDecompKey ? (seasonalMap[activeDecompKey] ?? null) : null
  const activeDecompUnit =
    distributions[activeDecompKey]?.unit ??
    metricSpecsHint?.[activeDecompKey]?.unit ??
    ""
  const activeDecompLabel = resolveMetricLabel(
    activeDecompKey,
    metricSpecsHint,
    distributions
  )
  const activeDecompColor = activeDecompKey
    ? metricSpecsHint?.[activeDecompKey]?.chart_token
      ? getMetricColorVar(activeDecompKey, metricSpecsHint)
      : "var(--primary)"
    : "var(--primary)"

  const seasonalCycleRows = useMemo<SeasonalCycleRow[]>(() => {
    if (!activeDecomp || activeDecomp.status !== "available") return []
    return (activeDecomp.seasonal_cycle_24h ?? [])
      .slice()
      .sort((a, b) => a.hour - b.hour)
      .map((pt) => ({
        hour: pt.hour,
        hourLabel: formatHourShort(pt.hour),
        seasonal: isFiniteNum(pt.seasonal) ? pt.seasonal : null,
      }))
  }, [activeDecomp])

  const seasonalSeriesRows = useMemo<SeasonalPointRow[]>(() => {
    if (!activeDecomp || activeDecomp.status !== "available") return []
    const avail = activeDecomp as EdaSeasonalDecompositionAvailable
    const count = avail.timestamps_utc?.length ?? 0
    const rows: SeasonalPointRow[] = []
    for (let i = 0; i < count; i++) {
      const ts = avail.timestamps_utc[i]
      if (!ts) continue
      rows.push({
        timestamp_utc: ts,
        observed: isFiniteNum(avail.observed?.[i]) ? avail.observed[i] : null,
        trend: isFiniteNum(avail.trend?.[i]) ? avail.trend[i] : null,
        seasonal: isFiniteNum(avail.seasonal?.[i]) ? avail.seasonal[i] : null,
        resid: isFiniteNum(avail.resid?.[i]) ? avail.resid[i] : null,
      })
    }
    return rows
  }, [activeDecomp])

  const sampledSeasonalSeries = useMemo(
    () => downsampleSeries(seasonalSeriesRows, 300),
    [seasonalSeriesRows]
  )

  const [trendDomainMin, trendDomainMax] = useMemo(() => {
    const vals: Array<number | null> = []
    for (const row of sampledSeasonalSeries) {
      vals.push(row.observed, row.trend)
    }
    return computeChartDomain(vals)
  }, [sampledSeasonalSeries])

  const [compDomainMin, compDomainMax] = useMemo(() => {
    const vals: Array<number | null> = []
    for (const row of sampledSeasonalSeries) {
      vals.push(row.seasonal, row.resid)
    }
    return computeChartDomain(vals)
  }, [sampledSeasonalSeries])

  const [cycleDomainMin, cycleDomainMax] = useMemo(() => {
    return computeChartDomain(seasonalCycleRows.map((row) => row.seasonal))
  }, [seasonalCycleRows])

  const shouldAnimateSeasonal = useFirstLoadChartAnimation(
    seasonalCycleRows.length > 0 || sampledSeasonalSeries.length > 0
  )

  const seasonalTableColumns = useMemo<ChartTableColumn<SeasonalCycleRow>[]>(
    () => [
      {
        key: "hour",
        header: `Hour (${tzShortLabel})`,
        render: (row) => formatHourPadded(row.hour),
      },
      {
        key: "seasonal",
        header: `Seasonal effect${activeDecompUnit ? ` (${activeDecompUnit})` : ""}`,
        align: "right",
        render: (row) =>
          isFiniteNum(row.seasonal)
            ? `${row.seasonal > 0 ? "+" : ""}${row.seasonal.toFixed(2)}`
            : "—",
      },
    ],
    [tzShortLabel, activeDecompUnit]
  )

  const seasonalSeriesTableColumns = useMemo<
    ChartTableColumn<SeasonalPointRow>[]
  >(
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
      {
        key: "observed",
        header: "Observed",
        align: "right",
        render: (row) => formatMetricValue(row.observed, 2),
      },
      {
        key: "trend",
        header: "Trend",
        align: "right",
        render: (row) => formatMetricValue(row.trend, 2),
      },
      {
        key: "seasonal",
        header: "Seasonal",
        align: "right",
        render: (row) => formatMetricValue(row.seasonal, 2),
      },
      {
        key: "resid",
        header: "Residual",
        align: "right",
        render: (row) => formatMetricValue(row.resid, 2),
      },
    ],
    [resolvedTimeZone]
  )

  // Imputation audit data (supports both EDA summary shape and pipeline_runs ImputationAuditReport fallback shape)
  const audit = data?.imputation_audit
  const auditNormalized = useMemo(() => {
    if (!audit) {
      return {
        isAvailable: false,
        sourceLabel: "Unavailable",
        totalRows: null as number | null,
        imputedRows: null as number | null,
        tierTracked: false,
        tierEntries: [] as Array<{
          tier: string
          label: string
          count: number
          pct: number
        }>,
        missingEntries: [] as Array<{
          column: string
          label: string
          missingCount: number
          missingPct: number | null
        }>,
      }
    }

    const raw = audit as Record<string, unknown>
    const hasPipelineReportFields =
      isFiniteNum(raw.total_rows) &&
      (isFiniteNum(raw.rows_imputed) ||
        isFiniteNum(raw.tier1_count) ||
        (typeof raw.per_column === "object" && raw.per_column !== null))

    const isAvailable =
      audit.status === "available" ||
      (audit.status !== "unavailable" && hasPipelineReportFields)

    if (!isAvailable) {
      return {
        isAvailable: false,
        sourceLabel: humanizeAuditSource(audit.source),
        totalRows: null,
        imputedRows: null,
        tierTracked: false,
        tierEntries: [],
        missingEntries: [],
      }
    }

    const totalRows = isFiniteNum(audit.total_rows) ? audit.total_rows : null
    const imputedRows = isFiniteNum(audit.imputed_rows_count)
      ? audit.imputed_rows_count
      : isFiniteNum(raw.rows_imputed)
        ? raw.rows_imputed
        : null

    const tierLabelMap: Record<string, string> = {
      none: "Observed (no gap)",
      tier1_linear: "Tier 1: Linear interpolation (<=3h)",
      tier2_diurnal_median: "Tier 2: Diurnal median (4-24h)",
      tier3_climatology_ffill: "Tier 3: Climatology fallback (>24h)",
    }

    let rawTiers: Array<[string, number]> = []
    if (audit.tier_counts && typeof audit.tier_counts === "object") {
      rawTiers = Object.entries(audit.tier_counts).filter(([, c]) =>
        isFiniteNum(c)
      ) as Array<[string, number]>
    } else if (
      isFiniteNum(raw.tier1_count) ||
      isFiniteNum(raw.tier2_count) ||
      isFiniteNum(raw.tier3_count)
    ) {
      const candidates: Array<[string, number]> = [
        ["tier1_linear", isFiniteNum(raw.tier1_count) ? raw.tier1_count : 0],
        [
          "tier2_diurnal_median",
          isFiniteNum(raw.tier2_count) ? raw.tier2_count : 0,
        ],
        [
          "tier3_climatology_ffill",
          isFiniteNum(raw.tier3_count) ? raw.tier3_count : 0,
        ],
      ]
      rawTiers = candidates.filter(([, count]) => count > 0)
    }

    const tierTotal = rawTiers.reduce((acc, [, c]) => acc + c, 0)
    const tierEntries = rawTiers
      .map(([tier, count]) => ({
        tier,
        label: tierLabelMap[tier.toLowerCase()] ?? humanizeIdentifier(tier),
        count,
        pct: tierTotal > 0 ? (count / tierTotal) * 100 : 0,
      }))
      .sort((a, b) => b.count - a.count)

    const missingEntries: Array<{
      column: string
      label: string
      missingCount: number
      missingPct: number | null
    }> = []

    if (
      audit.observed_missing_counts &&
      typeof audit.observed_missing_counts === "object"
    ) {
      for (const [col, count] of Object.entries(audit.observed_missing_counts)) {
        if (!isFiniteNum(count)) continue
        missingEntries.push({
          column: col,
          label: resolveMetricLabel(col, metricSpecsHint, distributions),
          missingCount: count,
          missingPct:
            totalRows && totalRows > 0 ? (count / totalRows) * 100 : null,
        })
      }
    } else if (typeof raw.per_column === "object" && raw.per_column !== null) {
      for (const [col, stats] of Object.entries(
        raw.per_column as Record<string, Record<string, unknown>>
      )) {
        if (!stats || typeof stats !== "object") continue
        const before = isFiniteNum(stats.missing_before)
          ? stats.missing_before
          : isFiniteNum(stats.imputed_count)
            ? stats.imputed_count
            : null
        if (before === null) continue
        missingEntries.push({
          column: col,
          label: resolveMetricLabel(col, metricSpecsHint, distributions),
          missingCount: before,
          missingPct:
            totalRows && totalRows > 0 ? (before / totalRows) * 100 : null,
        })
      }
    }

    missingEntries.sort((a, b) => b.missingCount - a.missingCount)

    return {
      isAvailable: true,
      sourceLabel: audit.source
        ? humanizeAuditSource(audit.source)
        : "Recorded by pipeline",
      totalRows,
      imputedRows,
      tierTracked: Boolean(audit.imputation_tier_present ?? tierEntries.length > 0),
      tierEntries,
      missingEntries,
    }
  }, [audit, metricSpecsHint, distributions])

  if (isLoading && !data) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-20 animate-pulse rounded-xl border border-border bg-muted" />
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          <div className="h-[360px] animate-pulse rounded-xl border border-border bg-muted" />
          <div className="h-[360px] animate-pulse rounded-xl border border-border bg-muted" />
          <div className="h-[360px] animate-pulse rounded-xl border border-border bg-muted xl:col-span-2" />
        </div>
      </div>
    )
  }

  if (errorMessage && !data) {
    return (
      <ErrorState
        title="Couldn't load exploratory data analysis"
        message={errorMessage}
        onRetry={handleRetry}
        isRetrying={isFetching}
      />
    )
  }

  const hasAnySectionData =
    distributionKeys.length > 0 ||
    diurnalMetricKeys.length > 0 ||
    corrColumns.length > 0 ||
    seasonalKeys.length > 0

  if (!data || (data.status === "bootstrapping" && !hasAnySectionData)) {
    return (
      <BootstrappingState
        city={data?.metadata?.city ?? city ?? undefined}
        onRefresh={handleRetry}
        isRefreshing={isFetching}
      />
    )
  }

  const totalSamples = data.metadata?.sample_count ?? data.sample_count ?? 0

  return (
    <div className="flex flex-col gap-6 md:gap-8">
      {/* Header + Dataset metadata bar */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
              Exploratory data analysis
            </h2>
            <p className="mt-1 max-w-[68ch] text-sm/5 text-muted-foreground">
              Historical distributions, {tzShortLabel} diurnal cycles, feature
              correlations, seasonal decomposition, and imputation coverage.
            </p>
          </div>
          {isFetching ? (
            <span className="text-xs font-medium text-muted-foreground">
              Updating view...
            </span>
          ) : null}
        </div>

        <dl className="grid grid-cols-2 gap-3 rounded-xl border border-border bg-card p-4 shadow-(--shadow-card) md:grid-cols-4">
          <div>
            <dt className="text-xs text-muted-foreground">Observations</dt>
            <dd className="mt-0.5 font-display text-h2 font-semibold text-foreground tabular-nums">
              {totalSamples.toLocaleString()}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Date range</dt>
            <dd className="mt-1 text-xs font-medium text-foreground">
              {data.date_range_start || data.metadata?.date_range_start ? (
                <span className="inline-flex flex-wrap items-center gap-1">
                  <TimestampPkt
                    timestamp={
                      data.date_range_start ?? data.metadata?.date_range_start
                    }
                    timeZone={resolvedTimeZone}
                    mode="datetime"
                    showZone={false}
                  />
                  <span>to</span>
                  <TimestampPkt
                    timestamp={
                      data.date_range_end ?? data.metadata?.date_range_end
                    }
                    timeZone={resolvedTimeZone}
                    mode="datetime"
                  />
                </span>
              ) : (
                "—"
              )}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Seasonal period</dt>
            <dd className="mt-1 text-sm font-semibold text-foreground tabular-nums">
              {data.metadata?.seasonal_period ?? 24} hours
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Last computed</dt>
            <dd className="mt-1 text-xs font-medium text-foreground">
              <TimestampPkt
                timestamp={data.computed_at_utc}
                timeZone={resolvedTimeZone}
                mode="datetime"
              />
            </dd>
          </div>
        </dl>
      </div>

      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh exploratory analysis"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      ) : null}

      {/* 2-Column Grid of EDA Chart Cards */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Card 1: Distributions */}
        <ChartCard
          title="Variable distributions"
          subtitle={
            activeDist
              ? `${activeDist.label}${activeDist.unit ? ` (${activeDist.unit})` : ""} histogram and empirical quantiles`
              : "Histogram and summary quantiles across observed metrics"
          }
          ariaLabel="Feature distribution histogram"
          controls={
            distributionKeys.length > 0 ? (
              <Select value={activeDistKey} onValueChange={setSelectedDistKey}>
                <SelectTrigger
                  aria-label="Select distribution metric"
                  className="h-8 min-w-[168px] bg-card text-xs"
                >
                  <SelectValue placeholder="Select variable" />
                </SelectTrigger>
                <SelectContent position="popper">
                  {distributionKeys.map((key) => {
                    const entry = distributions[key]
                    const unavail = entry?.status !== "available"
                    return (
                      <SelectItem key={key} value={key}>
                        {entry?.label ?? resolveMetricLabel(key, metricSpecsHint)}
                        {unavail ? " (unavailable)" : ""}
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
            ) : undefined
          }
          subHeader={
            activeDist && activeDist.status === "available" ? (
              <dl className="grid grid-cols-3 gap-2 sm:grid-cols-4">
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Mean / Std</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(activeDist.mean, 1)} /{" "}
                    {formatMetricValue(activeDist.std, 1)}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Min / Max</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(activeDist.min, 1)} /{" "}
                    {formatMetricValue(activeDist.max, 1)}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">P05 / P50 / P95</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(activeDist.q05, 1)} /{" "}
                    {formatMetricValue(activeDist.q50, 1)} /{" "}
                    {formatMetricValue(activeDist.q95, 1)}
                  </dd>
                </div>
                <div className="col-span-3 rounded-lg bg-muted px-3 py-2 sm:col-span-1">
                  <dt className="text-xs text-foreground/75">Valid / Missing</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {activeDist.valid_count.toLocaleString()} /{" "}
                    {activeDist.missing_count.toLocaleString()}
                  </dd>
                </div>
              </dl>
            ) : undefined
          }
          tableColumns={histogramTableColumns}
          tableRows={histogramRows}
          emptyTableMessage="No histogram bins available for the selected metric."
          footer={
            activeDist && isFiniteNum(activeDist.circular_mean_deg) ? (
              <span className="tabular-nums">
                Circular mean direction:{" "}
                <span className="font-medium text-foreground">
                  {activeDist.circular_mean_deg.toFixed(1)}°
                </span>
              </span>
            ) : undefined
          }
        >
          {!activeDist ? (
            <EmptyState
              icon={Grid2x2}
              title="Distributions unavailable"
              description="Summary distributions will appear once the EDA pipeline computes feature statistics."
              className="border-0 bg-muted shadow-none"
            />
          ) : activeDist.status !== "available" ? (
            <div className="flex items-start gap-3 rounded-lg bg-muted p-4 text-sm text-foreground">
              <AlertCircle
                className="mt-0.5 size-5 shrink-0 text-alert-warning"
                strokeWidth={1.75}
              />
              <div>
                <p className="font-semibold">
                  {activeDist.label} distribution unavailable (
                  {humanizeIdentifier(activeDist.status)})
                </p>
                <p className="mt-1 text-xs text-foreground/80 tabular-nums">
                  Valid rows: {activeDist.valid_count.toLocaleString()}, missing
                  rows: {activeDist.missing_count.toLocaleString()}
                </p>
              </div>
            </div>
          ) : histogramRows.length === 0 ? (
            <EmptyState
              icon={Grid2x2}
              title="No histogram bins recorded"
              description={`Valid observations (${activeDist.valid_count}) did not produce histogram bins for ${activeDist.label}.`}
              className="border-0 bg-muted shadow-none"
            />
          ) : (
            <ChartContainer
              config={{
                count: { label: "Observations", color: distChartColor },
              }}
              className="h-[240px] w-full aspect-auto md:h-[260px]"
            >
              <BarChart
                data={histogramRows}
                margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
              >
                <CartesianGrid vertical={false} stroke="var(--border)" />
                <XAxis
                  dataKey="label"
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                  minTickGap={22}
                />
                <YAxis
                  tickFormatter={formatAxisTick}
                  tickLine={false}
                  axisLine={false}
                  width={44}
                  tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                />
                <ChartTooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null
                    const row = payload[0]?.payload as
                      | HistogramChartRow
                      | undefined
                    if (!row) return null
                    const share =
                      activeDist.valid_count > 0
                        ? ((row.count / activeDist.valid_count) * 100).toFixed(1)
                        : null
                    return (
                      <SkyCastChartTooltipCard
                        rows={[
                          {
                            key: "bin",
                            label: `${row.label}${activeDist.unit ? ` ${activeDist.unit}` : ""}`,
                            valueText: `${row.count.toLocaleString()} rows`,
                            color: distChartColor,
                          },
                        ]}
                        footerNote={
                          share !== null ? `${share}% of valid observations` : null
                        }
                      />
                    )
                  }}
                />
                <Bar
                  dataKey="count"
                  fill={distChartColor}
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={false}
                />
              </BarChart>
            </ChartContainer>
          )}
        </ChartCard>

        {/* Card 2: 24h PKT Diurnal Heatmap */}
        <ChartCard
          title={`24-hour diurnal cycle (${tzShortLabel})`}
          subtitle="Hourly variation normalized per metric row from low (pale) to peak (deep sky)"
          ariaLabel="24-hour diurnal cycle heatmap"
          controls={
            <div
              role="group"
              aria-label="Diurnal aggregation statistic"
              className="inline-flex h-8 items-center rounded-md bg-muted p-0.5"
            >
              {(["mean", "median"] as const).map((mode) => {
                const active = diurnalMode === mode
                return (
                  <button
                    key={mode}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setDiurnalMode(mode)}
                    className={cn(
                      "inline-flex h-7 items-center rounded-sm px-2.5 text-xs font-medium capitalize transition-colors duration-120",
                      active
                        ? "bg-card text-foreground shadow-(--shadow-card)"
                        : "text-foreground/75 hover:text-foreground"
                    )}
                  >
                    {mode}
                  </button>
                )
              })}
            </div>
          }
          tableColumns={diurnalTableColumns}
          tableRows={diurnalTableRows}
          emptyTableMessage="No diurnal buckets recorded."
          footer={
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span>Cells normalized within each variable row</span>
              <div className="flex items-center gap-1.5">
                <span>Low</span>
                {HEAT_TOKENS.map((token, idx) => (
                  <span
                    key={idx}
                    className="size-3 rounded-xs border border-border"
                    style={{ backgroundColor: token }}
                    aria-hidden="true"
                  />
                ))}
                <span>High</span>
              </div>
            </div>
          }
        >
          {diurnalRows.length === 0 ? (
            <EmptyState
              icon={Grid2x2}
              title="Diurnal heatmap unavailable"
              description="Hourly diurnal profiles appear once multi-day local timestamp buckets are computed."
              className="border-0 bg-muted shadow-none"
            />
          ) : (
            <div className="overflow-x-auto pb-1">
              <div className="inline-block min-w-full align-middle">
                {/* Hour header row */}
                <div className="flex items-center gap-1 pb-1.5">
                  <div className="sticky left-0 z-10 w-[124px] shrink-0 bg-card pr-2 text-xs font-medium text-muted-foreground sm:w-[144px]">
                    Metric ({tzShortLabel})
                  </div>
                  <div className="flex items-center gap-1">
                    {diurnalHours.map((hour) => (
                      <div
                        key={hour}
                        className="w-5 shrink-0 text-center text-[11px] text-muted-foreground tabular-nums sm:w-7"
                      >
                        {formatHourShort(hour)}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Metric heatmap rows */}
                <div className="flex flex-col gap-1">
                  {diurnalRows.map((row) => (
                    <div
                      key={row.metricKey}
                      className="flex items-center gap-1"
                    >
                      <div
                        className="sticky left-0 z-10 w-[124px] shrink-0 truncate bg-card pr-2 text-xs font-medium text-foreground sm:w-[144px]"
                        title={row.metricLabel}
                      >
                        {row.metricLabel}
                      </div>

                      <div className="flex items-center gap-1">
                        {diurnalHours.map((hour) => {
                          const bucket = row.byHour.get(hour)
                          const val = bucket
                            ? diurnalMode === "median"
                              ? bucket.median
                              : bucket.mean
                            : null
                          const validVal =
                            bucket && bucket.count > 0 && isFiniteNum(val)
                              ? val
                              : null
                          const heatColor =
                            validVal !== null &&
                            row.minVal !== null &&
                            row.maxVal !== null
                              ? pickHeatToken(validVal, row.minVal, row.maxVal)
                              : null

                          return (
                            <Tooltip key={hour}>
                              <TooltipTrigger asChild>
                                <button
                                  type="button"
                                  aria-label={`${row.metricLabel} at ${formatHourPadded(hour)} ${tzShortLabel}: ${
                                    validVal !== null
                                      ? `${formatMetricValue(validVal, 1)} ${row.unit}`
                                      : "no data"
                                  }`}
                                  className={cn(
                                    "flex size-5 shrink-0 items-center justify-center rounded-sm border border-border/50 text-[10px] transition-transform duration-120 hover:scale-105 focus-visible:outline-2 focus-visible:outline-ring sm:size-7",
                                    heatColor === null &&
                                      "bg-muted text-muted-foreground"
                                  )}
                                  style={
                                    heatColor
                                      ? { backgroundColor: heatColor }
                                      : undefined
                                  }
                                >
                                  {heatColor === null ? "—" : ""}
                                </button>
                              </TooltipTrigger>
                              <TooltipContent className="flex flex-col gap-0.5 text-left">
                                <span className="font-semibold text-popover-foreground">
                                  {row.metricLabel} at {formatHourPadded(hour)}{" "}
                                  {tzShortLabel}
                                </span>
                                {bucket && bucket.count > 0 ? (
                                  <>
                                    <span className="text-muted-foreground tabular-nums">
                                      Mean: {formatMetricValue(bucket.mean, 2)}
                                      {row.unit ? ` ${row.unit}` : ""}
                                    </span>
                                    <span className="text-muted-foreground tabular-nums">
                                      Median:{" "}
                                      {formatMetricValue(bucket.median, 2)}
                                      {row.unit ? ` ${row.unit}` : ""}
                                    </span>
                                    <span className="text-muted-foreground tabular-nums">
                                      Std dev: {formatMetricValue(bucket.std, 2)}{" "}
                                      ({bucket.count} samples)
                                    </span>
                                  </>
                                ) : (
                                  <span className="text-muted-foreground">
                                    No observations in this hour bucket
                                  </span>
                                )}
                              </TooltipContent>
                            </Tooltip>
                          )
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </ChartCard>

        {/* Card 3: Correlation Matrix (spans full 2 cols on xl) */}
        <ChartCard
          title="Feature correlation matrix"
          subtitle={
            corrColumns.length > 0
              ? `${humanizeIdentifier(data.correlation_matrix?.method ?? "pearson")} correlations (min ${data.correlation_matrix?.min_periods ?? 1} paired rows)`
              : "Pairwise statistical associations across weather and pollutant variables"
          }
          ariaLabel="Feature correlation matrix"
          className="xl:col-span-2"
          subHeader={
            corrHighlights.length > 0 ||
            (data.correlation_matrix?.constant_columns?.length ?? 0) > 0 ? (
              <div className="flex flex-col gap-2">
                {corrHighlights.length > 0 ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-medium text-muted-foreground">
                      Notable pairs:
                    </span>
                    {corrHighlights.slice(0, 6).map(([pairKey, val]) => (
                      <Badge
                        key={pairKey}
                        variant="secondary"
                        className="h-6 px-2.5 text-xs tabular-nums"
                      >
                        {formatHighlightPairLabel(
                          pairKey,
                          metricSpecsHint,
                          distributions
                        )}
                        :{" "}
                        {isFiniteNum(val)
                          ? `${val > 0 ? "+" : ""}${val.toFixed(2)}`
                          : "—"}
                      </Badge>
                    ))}
                  </div>
                ) : null}

                {(data.correlation_matrix?.constant_columns?.length ?? 0) > 0 ? (
                  <p className="text-xs text-muted-foreground">
                    Constant columns excluded from pairs:{" "}
                    {data.correlation_matrix.constant_columns
                      .map((col) =>
                        resolveMetricLabel(col, metricSpecsHint, distributions)
                      )
                      .join(", ")}
                  </p>
                ) : null}
              </div>
            ) : undefined
          }
          tableColumns={corrTableColumns}
          tableRows={corrTableRows}
          emptyTableMessage="No correlation pairs available."
          footer={
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span>
                Orange denotes positive correlation; blue denotes inverse
                relationship.
              </span>
              <span>Hover or focus any cell for exact r</span>
            </div>
          }
        >
          {corrColumns.length === 0 ? (
            <EmptyState
              icon={Grid2x2}
              title="Correlation matrix unavailable"
              description="Pairwise correlation coefficients appear once sufficient valid observations are available."
              className="border-0 bg-muted shadow-none"
            />
          ) : (
            <div className="overflow-x-auto pb-1">
              <div className="inline-block align-middle">
                {/* Column headers */}
                <div className="flex items-end gap-1 pb-1.5">
                  <div className="sticky left-0 z-10 w-[124px] shrink-0 bg-card pr-2 text-xs font-medium text-muted-foreground sm:w-[144px]" />
                  {corrColumns.map((colKey) => {
                    const label = resolveMetricLabel(
                      colKey,
                      metricSpecsHint,
                      distributions
                    )
                    return (
                      <div
                        key={colKey}
                        title={label}
                        className="w-9 shrink-0 truncate text-center text-[11px] font-medium text-muted-foreground sm:w-12"
                      >
                        {label}
                      </div>
                    )
                  })}
                </div>

                {/* Matrix rows */}
                <div className="flex flex-col gap-1">
                  {corrColumns.map((rowKey) => {
                    const rowLabel = resolveMetricLabel(
                      rowKey,
                      metricSpecsHint,
                      distributions
                    )
                    return (
                      <div key={rowKey} className="flex items-center gap-1">
                        <div
                          className="sticky left-0 z-10 w-[124px] shrink-0 truncate bg-card pr-2 text-xs font-medium text-foreground sm:w-[144px]"
                          title={rowLabel}
                        >
                          {rowLabel}
                        </div>

                        {corrColumns.map((colKey) => {
                          const rVal = corrMatrix?.[rowKey]?.[colKey] ?? null
                          const colLabel = resolveMetricLabel(
                            colKey,
                            metricSpecsHint,
                            distributions
                          )
                          const cell = correlationCellStyle(rVal)

                          return (
                            <Tooltip key={colKey}>
                              <TooltipTrigger asChild>
                                <button
                                  type="button"
                                  aria-label={`${rowLabel} and ${colLabel}: correlation ${cell.label}`}
                                  className={cn(
                                    "flex size-9 shrink-0 items-center justify-center rounded-sm border border-border/50 text-[11px] font-medium tabular-nums focus-visible:outline-2 focus-visible:outline-ring sm:size-12 sm:text-xs",
                                    isFiniteNum(rVal)
                                      ? "text-foreground"
                                      : "bg-muted text-muted-foreground"
                                  )}
                                  style={
                                    cell.backgroundColor
                                      ? { backgroundColor: cell.backgroundColor }
                                      : undefined
                                  }
                                >
                                  {cell.label}
                                </button>
                              </TooltipTrigger>
                              <TooltipContent className="flex flex-col gap-0.5 text-left">
                                <span className="font-semibold text-popover-foreground">
                                  {rowLabel} × {colLabel}
                                </span>
                                <span className="text-muted-foreground tabular-nums">
                                  Correlation:{" "}
                                  {isFiniteNum(rVal) ? rVal.toFixed(4) : "N/A"}
                                </span>
                              </TooltipContent>
                            </Tooltip>
                          )
                        })}
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}
        </ChartCard>

        {/* Card 4: Seasonal Decomposition */}
        <ChartCard
          title="Seasonal decomposition"
          subtitle={
            activeDecomp?.status === "available"
              ? `${activeDecompLabel} decomposed into trend, ${activeDecomp.period}h seasonal cycle, and residual`
              : `${activeDecompLabel || "Metric"} trend and diurnal seasonal separation`
          }
          ariaLabel="Seasonal decomposition chart"
          controls={
            <div className="flex flex-wrap items-center gap-2">
              {seasonalKeys.length > 0 ? (
                <Select
                  value={activeDecompKey}
                  onValueChange={setSelectedDecompKey}
                >
                  <SelectTrigger
                    aria-label="Select decomposition metric"
                    className="h-8 min-w-[152px] bg-card text-xs"
                  >
                    <SelectValue placeholder="Metric" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    {seasonalKeys.map((key) => {
                      const entry = seasonalMap[key]
                      const unavail = entry?.status !== "available"
                      return (
                        <SelectItem key={key} value={key}>
                          {resolveMetricLabel(key, metricSpecsHint, distributions)}
                          {unavail ? " (unavailable)" : ""}
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>
              ) : null}

              {activeDecomp?.status === "available" ? (
                <div
                  role="group"
                  aria-label="Decomposition chart view"
                  className="inline-flex h-8 items-center rounded-md bg-muted p-0.5"
                >
                  {(
                    [
                      { id: "cycle", label: "24h cycle" },
                      { id: "trend", label: "Trend" },
                      { id: "components", label: "Resid" },
                    ] as const
                  ).map((item) => {
                    const active = decompView === item.id
                    return (
                      <button
                        key={item.id}
                        type="button"
                        aria-pressed={active}
                        onClick={() => setDecompView(item.id)}
                        className={cn(
                          "inline-flex h-7 items-center rounded-sm px-2 text-xs font-medium transition-colors duration-120",
                          active
                            ? "bg-card text-foreground shadow-(--shadow-card)"
                            : "text-foreground/75 hover:text-foreground"
                        )}
                      >
                        {item.label}
                      </button>
                    )
                  })}
                </div>
              ) : null}
            </div>
          }
          subHeader={
            activeDecomp?.status === "available" ? (
              <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Observed mean</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(activeDecomp.summary?.observed_mean, 2)}
                    {activeDecompUnit ? ` ${activeDecompUnit}` : ""}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Trend shift</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(activeDecomp.summary?.trend_start, 1)} to{" "}
                    {formatMetricValue(activeDecomp.summary?.trend_end, 1)}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">
                    Seasonal amplitude
                  </dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(
                      activeDecomp.summary?.seasonal_amplitude,
                      2
                    )}
                    {activeDecompUnit ? ` ${activeDecompUnit}` : ""}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Residual std</dt>
                  <dd className="mt-0.5 text-xs font-semibold text-foreground tabular-nums">
                    {formatMetricValue(activeDecomp.summary?.residual_std, 2)}
                  </dd>
                </div>
              </dl>
            ) : undefined
          }
          tableColumns={
            decompView === "cycle"
              ? (seasonalTableColumns as ChartTableColumn<Record<string, unknown>>[])
              : (seasonalSeriesTableColumns as ChartTableColumn<Record<string, unknown>>[])
          }
          tableRows={
            decompView === "cycle"
              ? (seasonalCycleRows as Record<string, unknown>[])
              : (sampledSeasonalSeries as Record<string, unknown>[])
          }
          emptyTableMessage="No seasonal decomposition data available for this metric."
          footer={
            activeDecomp?.status === "available" ? (
              <span className="tabular-nums">
                Used {activeDecomp.sample_count_used.toLocaleString()} contiguous
                hourly samples (period {activeDecomp.period}h)
              </span>
            ) : undefined
          }
        >
          {!activeDecomp ? (
            <EmptyState
              icon={TrendingUp}
              title="Seasonal decomposition unavailable"
              description="Decomposition diagnostics appear once sufficient contiguous hourly observations are collected."
              className="border-0 bg-muted shadow-none"
            />
          ) : activeDecomp.status === "unavailable" ? (
            <div className="flex items-start gap-3 rounded-lg bg-muted p-4 text-sm text-foreground">
              <AlertCircle
                className="mt-0.5 size-5 shrink-0 text-alert-warning"
                strokeWidth={1.75}
              />
              <div>
                <p className="font-semibold">
                  {activeDecompLabel} seasonal decomposition unavailable
                </p>
                <p className="mt-1 text-xs text-foreground/80">
                  {activeDecomp.reason}
                </p>
                <p className="mt-2 text-xs text-foreground/80 tabular-nums">
                  Valid rows: {activeDecomp.valid_count.toLocaleString()} of{" "}
                  {activeDecomp.required_rows.toLocaleString()} required
                  {isFiniteNum(activeDecomp.longest_contiguous_count)
                    ? `, longest contiguous span: ${activeDecomp.longest_contiguous_count.toLocaleString()}h`
                    : ""}
                </p>
              </div>
            </div>
          ) : decompView === "cycle" ? (
            seasonalCycleRows.length === 0 ? (
              <EmptyState
                icon={TrendingUp}
                title="No 24h seasonal cycle points"
                description="Seasonal cycle weights were not returned for this variable."
                className="border-0 bg-muted shadow-none"
              />
            ) : (
              <ChartContainer
                config={{
                  seasonal: { label: "Seasonal delta", color: activeDecompColor },
                }}
                className="h-[240px] w-full aspect-auto md:h-[260px]"
              >
                <BarChart
                  data={seasonalCycleRows}
                  margin={{ top: 10, right: 12, bottom: 4, left: 0 }}
                >
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis
                    dataKey="hourLabel"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                    minTickGap={16}
                  />
                  <YAxis
                    domain={[cycleDomainMin, cycleDomainMax]}
                    tickFormatter={formatAxisTick}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                  />
                  <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1.5} />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const row = payload[0]?.payload as
                        | SeasonalCycleRow
                        | undefined
                      if (!row || row.seasonal === null) return null
                      return (
                        <SkyCastChartTooltipCard
                          rows={[
                            {
                              key: "seasonal",
                              label: `${formatHourPadded(row.hour)} ${tzShortLabel}`,
                              valueText: `${row.seasonal > 0 ? "+" : ""}${row.seasonal.toFixed(2)}${activeDecompUnit ? ` ${activeDecompUnit}` : ""}`,
                              color:
                                row.seasonal >= 0
                                  ? "var(--shap-pos)"
                                  : "var(--shap-neg)",
                            },
                          ]}
                        />
                      )
                    }}
                  />
                  <Bar dataKey="seasonal" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                    {seasonalCycleRows.map((entry) => (
                      <Cell
                        key={entry.hour}
                        fill={
                          (entry.seasonal ?? 0) >= 0
                            ? "var(--shap-pos)"
                            : "var(--shap-neg)"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ChartContainer>
            )
          ) : decompView === "trend" ? (
            <div>
              <ChartContainer
                config={{
                  observed: { label: "Observed", color: "var(--muted-foreground)" },
                  trend: { label: "Trend", color: activeDecompColor },
                }}
                className="h-[240px] w-full aspect-auto md:h-[260px]"
              >
                <ComposedChart
                  data={sampledSeasonalSeries}
                  margin={{ top: 10, right: 12, bottom: 4, left: 0 }}
                >
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis
                    dataKey="timestamp_utc"
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
                    domain={[trendDomainMin, trendDomainMax]}
                    tickFormatter={formatAxisTick}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                  />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const row = payload[0]?.payload as
                        | SeasonalPointRow
                        | undefined
                      if (!row) return null
                      const rows: TooltipEntryRow[] = []
                      if (row.observed !== null) {
                        rows.push({
                          key: "observed",
                          label: "Observed",
                          valueText: formatMetricValue(row.observed, 2),
                          color: "var(--muted-foreground)",
                        })
                      }
                      if (row.trend !== null) {
                        rows.push({
                          key: "trend",
                          label: "Trend",
                          valueText: formatMetricValue(row.trend, 2),
                          color: activeDecompColor,
                        })
                      }
                      return (
                        <SkyCastChartTooltipCard
                          timestampUtc={row.timestamp_utc}
                          timeZone={resolvedTimeZone}
                          rows={rows}
                        />
                      )
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="observed"
                    stroke="var(--muted-foreground)"
                    strokeOpacity={0.5}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="trend"
                    stroke={activeDecompColor}
                    strokeWidth={2.25}
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 2 }}
                    connectNulls={false}
                    isAnimationActive={shouldAnimateSeasonal}
                    animationDuration={600}
                    animationEasing="ease-out"
                  />
                </ComposedChart>
              </ChartContainer>
              <ChartLegendBar
                items={[
                  {
                    key: "observed",
                    label: "Observed",
                    color: "var(--muted-foreground)",
                    variant: "solid",
                  },
                  {
                    key: "trend",
                    label: "Smoothed trend",
                    color: activeDecompColor,
                    variant: "solid",
                  },
                ]}
              />
            </div>
          ) : (
            <div>
              <ChartContainer
                config={{
                  seasonal: { label: "Seasonal", color: "var(--chart-4)" },
                  resid: { label: "Residual", color: "var(--chart-6)" },
                }}
                className="h-[240px] w-full aspect-auto md:h-[260px]"
              >
                <ComposedChart
                  data={sampledSeasonalSeries}
                  margin={{ top: 10, right: 12, bottom: 4, left: 0 }}
                >
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis
                    dataKey="timestamp_utc"
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
                    domain={[compDomainMin, compDomainMax]}
                    tickFormatter={formatAxisTick}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                  />
                  <ReferenceLine y={0} stroke="var(--border)" strokeWidth={1.5} />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const row = payload[0]?.payload as
                        | SeasonalPointRow
                        | undefined
                      if (!row) return null
                      const rows: TooltipEntryRow[] = []
                      if (row.seasonal !== null) {
                        rows.push({
                          key: "seasonal",
                          label: "Seasonal",
                          valueText: formatMetricValue(row.seasonal, 2),
                          color: "var(--chart-4)",
                        })
                      }
                      if (row.resid !== null) {
                        rows.push({
                          key: "resid",
                          label: "Residual",
                          valueText: formatMetricValue(row.resid, 2),
                          color: "var(--chart-6)",
                        })
                      }
                      return (
                        <SkyCastChartTooltipCard
                          timestampUtc={row.timestamp_utc}
                          timeZone={resolvedTimeZone}
                          rows={rows}
                        />
                      )
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="seasonal"
                    stroke="var(--chart-4)"
                    strokeWidth={1.75}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="resid"
                    stroke="var(--chart-6)"
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ChartContainer>
              <ChartLegendBar
                items={[
                  {
                    key: "seasonal",
                    label: "Seasonal component",
                    color: "var(--chart-4)",
                    variant: "solid",
                  },
                  {
                    key: "resid",
                    label: "Residual",
                    color: "var(--chart-6)",
                    variant: "solid",
                  },
                ]}
              />
            </div>
          )}
        </ChartCard>

        {/* Card 5: Imputation Audit */}
        <section className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-base/6 font-semibold text-card-foreground">
                Data quality and imputation audit
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Missing observation counts and gap-filling tier coverage
              </p>
            </div>
            {audit ? (
              <Badge variant="secondary" className="h-6 px-2.5 text-xs font-medium">
                {auditNormalized.sourceLabel}
              </Badge>
            ) : null}
          </div>

          {!auditNormalized.isAvailable ? (
            <EmptyState
              icon={ShieldAlert}
              title="Imputation audit unavailable"
              description="Gap-filling audit details will appear once observation quality checks run."
              className="border-0 bg-muted shadow-none"
            />
          ) : (
            <div className="flex flex-col gap-4">
              <dl className="grid grid-cols-3 gap-2">
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Total rows</dt>
                  <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                    {auditNormalized.totalRows !== null
                      ? auditNormalized.totalRows.toLocaleString()
                      : "—"}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Imputed rows</dt>
                  <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                    {auditNormalized.imputedRows !== null
                      ? auditNormalized.imputedRows.toLocaleString()
                      : "—"}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3 py-2">
                  <dt className="text-xs text-foreground/75">Tier tracked</dt>
                  <dd className="mt-0.5 text-sm font-semibold text-foreground">
                    {auditNormalized.tierTracked ? "Yes" : "No"}
                  </dd>
                </div>
              </dl>

              {auditNormalized.tierEntries.length > 0 ? (
                <div className="flex flex-col gap-2 rounded-lg bg-muted p-3.5">
                  <div className="text-xs font-medium text-foreground">
                    Imputation tier breakdown
                  </div>
                  <div className="flex flex-col gap-2">
                    {auditNormalized.tierEntries.map((entry) => (
                      <div key={entry.tier} className="flex flex-col gap-1">
                        <div className="flex items-center justify-between text-xs text-foreground tabular-nums">
                          <span>{entry.label}</span>
                          <span className="font-semibold">
                            {entry.count.toLocaleString()} ({entry.pct.toFixed(1)}%)
                          </span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-card">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${Math.max(2, entry.pct)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {auditNormalized.missingEntries.length > 0 ? (
                <Table
                  aria-label="Observed missing counts by variable"
                  containerClassName="max-h-64 overflow-auto rounded-md border border-border"
                >
                  <TableHeader className="sticky top-0 z-10 bg-muted">
                    <TableRow className="hover:bg-muted">
                      <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                        Variable
                      </TableHead>
                      <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                        Missing rows
                      </TableHead>
                      <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                        Share
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {auditNormalized.missingEntries.map((entry) => (
                      <TableRow key={entry.column} className="h-[52px] md:h-11">
                        <TableCell className="px-4 py-2.5 text-sm font-medium text-foreground">
                          {entry.label}
                        </TableCell>
                        <TableCell className="px-4 py-2.5 text-right text-sm tabular-nums">
                          {entry.missingCount.toLocaleString()}
                        </TableCell>
                        <TableCell className="px-4 py-2.5 text-right text-sm text-muted-foreground tabular-nums">
                          {entry.missingPct !== null
                            ? `${entry.missingPct.toFixed(1)}%`
                            : "—"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <EmptyState
                  icon={Layers}
                  title="No missing column tallies"
                  description="Per-variable missing counts were not recorded in this audit summary."
                  className="border-0 bg-muted shadow-none"
                />
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
