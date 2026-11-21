import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Clock,
  Database,
  GitCompareArrows,
  Grid2x2,
  Layers,
  ShieldAlert,
  TrendingUp,
} from "lucide-react"
import { cn } from "cn"
import {
  Area,
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
import { EdaCorrelationsCard } from "@/components/shared/eda-correlations-card"
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

type EdaSubTab = "patterns" | "relationships" | "trends"
type DiurnalAggregateMode = "mean" | "median"
type DiurnalDisplayMode = "curve" | "heatmap"
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

interface DiurnalCurveRow extends Record<string, unknown> {
  hour: number
  hourLabel: string
  value: number | null
  mean: number | null
  median: number | null
  std: number | null
  count: number
}

interface DiurnalTableRow extends Record<string, unknown> {
  metricKey: string
  metricLabel: string
  unit: string
  minMean: number | null
  maxMean: number | null
  peakHour: number | null
  lowHour: number | null
  sampleTotal: number
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
  const [activeSubTab, setActiveSubTab] = useState<EdaSubTab>("patterns")
  const [selectedMetricKey, setSelectedMetricKey] = useState<string>("")
  const [diurnalMode, setDiurnalMode] = useState<DiurnalAggregateMode>("mean")
  const [diurnalDisplay, setDiurnalDisplay] =
    useState<DiurnalDisplayMode>("curve")
  const [selectedDecompKey, setSelectedDecompKey] = useState<string>("")
  const [decompView, setDecompView] = useState<DecompositionViewMode>("cycle")
  const [showMissingTable, setShowMissingTable] = useState<boolean>(false)
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
    return [...ordered].sort((a, b) => {
      const aAvail = distributions[a]?.status === "available" ? 0 : 1
      const bAvail = distributions[b]?.status === "available" ? 0 : 1
      return aAvail - bAvail
    })
  }, [distributions, coreMetricGroupsHint])

  const diurnalMetricKeys = useMemo(() => {
    const keys = Object.keys(data?.diurnal_heatmap?.metrics ?? {})
    return orderMetricKeys(keys, coreMetricGroupsHint)
  }, [data?.diurnal_heatmap?.metrics, coreMetricGroupsHint])

  const allPatternMetricKeys = useMemo(() => {
    const merged = Array.from(new Set([...distributionKeys, ...diurnalMetricKeys]))
    return orderMetricKeys(merged, coreMetricGroupsHint)
  }, [distributionKeys, diurnalMetricKeys, coreMetricGroupsHint])

  const activeMetricKey = useMemo(() => {
    if (selectedMetricKey && allPatternMetricKeys.includes(selectedMetricKey)) {
      return selectedMetricKey
    }
    const available = allPatternMetricKeys.filter(
      (key) =>
        distributions[key]?.status === "available" ||
        diurnalMetricKeys.includes(key)
    )
    if (available.includes("temperature_2m")) return "temperature_2m"
    if (available.includes("us_aqi")) return "us_aqi"
    return available[0] ?? allPatternMetricKeys[0] ?? ""
  }, [selectedMetricKey, allPatternMetricKeys, distributions, diurnalMetricKeys])

  const activeDist: EdaDistributionEntry | null = activeMetricKey
    ? (distributions[activeMetricKey] ?? null)
    : null
  const activeDistUnit =
    activeDist?.unit ?? metricSpecsHint?.[activeMetricKey]?.unit ?? ""
  const activeMetricLabel = resolveMetricLabel(
    activeMetricKey,
    metricSpecsHint,
    distributions
  )

  const activeMetricColor = useMemo(() => {
    if (!activeMetricKey) return "var(--primary)"
    return metricSpecsHint?.[activeMetricKey]?.chart_token
      ? getMetricColorVar(activeMetricKey, metricSpecsHint)
      : "var(--primary)"
  }, [activeMetricKey, metricSpecsHint])

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

  // Diurnal data
  const diurnalHours = useMemo(() => {
    const raw = data?.diurnal_heatmap?.hours ?? []
    if (raw.length > 0) {
      return [...raw].sort((a, b) => a - b)
    }
    return Array.from({ length: 24 }, (_, idx) => idx)
  }, [data?.diurnal_heatmap?.hours])

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
      let lowHour: number | null = null
      let peakVal = -Infinity
      let lowVal = Infinity
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
          if (v < lowVal) {
            lowVal = v
            lowHour = hour
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
        lowHour,
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

  const activeDiurnalRow = useMemo(() => {
    const match = diurnalRows.find((r) => r.metricKey === activeMetricKey)
    return match ?? diurnalRows[0] ?? null
  }, [diurnalRows, activeMetricKey])

  const diurnalCurveRows = useMemo<DiurnalCurveRow[]>(() => {
    if (!activeDiurnalRow) return []
    return diurnalHours.map((hour) => {
      const b = activeDiurnalRow.byHour.get(hour)
      const val = b
        ? diurnalMode === "median"
          ? b.median
          : b.mean
        : null
      return {
        hour,
        hourLabel: formatHourShort(hour),
        value: b && b.count > 0 && isFiniteNum(val) ? val : null,
        mean: b && isFiniteNum(b.mean) ? b.mean : null,
        median: b && isFiniteNum(b.median) ? b.median : null,
        std: b && isFiniteNum(b.std) ? b.std : null,
        count: b && isFiniteNum(b.count) ? b.count : 0,
      }
    })
  }, [activeDiurnalRow, diurnalHours, diurnalMode])

  const [diurnalCurveMin, diurnalCurveMax] = useMemo(
    () => computeChartDomain(diurnalCurveRows.map((r) => r.value)),
    [diurnalCurveRows]
  )

  const diurnalTableRows = useMemo<DiurnalTableRow[]>(
    () =>
      diurnalRows.map((row) => ({
        metricKey: row.metricKey,
        metricLabel: row.metricLabel,
        unit: row.unit,
        minMean: row.minVal,
        maxMean: row.maxVal,
        peakHour: row.peakHour,
        lowHour: row.lowHour,
        sampleTotal: row.sampleTotal,
      })),
    [diurnalRows]
  )

  const diurnalTableColumns = useMemo<ChartTableColumn<DiurnalTableRow>[]>(
    () => [
      { key: "metricLabel", header: "Metric" },
      {
        key: "lowHour",
        header: `Lowest (${tzShortLabel})`,
        align: "right",
        render: (row) =>
          row.lowHour !== null
            ? `${formatHourPadded(row.lowHour)} (${formatMetricValue(row.minMean, 1)})`
            : "—",
      },
      {
        key: "peakHour",
        header: `Peak (${tzShortLabel})`,
        align: "right",
        render: (row) =>
          row.peakHour !== null
            ? `${formatHourPadded(row.peakHour)} (${formatMetricValue(row.maxMean, 1)})`
            : "—",
      },
      {
        key: "sampleTotal",
        header: "Samples",
        align: "right",
        render: (row) => row.sampleTotal.toLocaleString(),
      },
    ],
    [tzShortLabel]
  )

  // Correlations data for Key Findings + Tab 2
  const corrColumns = useMemo(
    () => data?.correlation_matrix?.columns ?? [],
    [data?.correlation_matrix?.columns]
  )
  const corrMatrix = useMemo(
    () => data?.correlation_matrix?.matrix ?? {},
    [data?.correlation_matrix?.matrix]
  )

  const strongestCorrPair = useMemo(() => {
    let best: { a: string; b: string; r: number } | null = null
    for (let i = 0; i < corrColumns.length; i++) {
      for (let j = i + 1; j < corrColumns.length; j++) {
        const rKey = corrColumns[i]
        const cKey = corrColumns[j]
        const val = corrMatrix?.[rKey]?.[cKey]
        if (!isFiniteNum(val)) continue
        if (!best || Math.abs(val) > Math.abs(best.r)) {
          best = {
            a: resolveMetricLabel(rKey, metricSpecsHint, distributions),
            b: resolveMetricLabel(cKey, metricSpecsHint, distributions),
            r: val,
          }
        }
      }
    }
    return best
  }, [corrColumns, corrMatrix, metricSpecsHint, distributions])

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
        header: `Daily rhythm${activeDecompUnit ? ` (${activeDecompUnit})` : ""}`,
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
        header: "Daily rhythm",
        align: "right",
        render: (row) => formatMetricValue(row.seasonal, 2),
      },
      {
        key: "resid",
        header: "Unusual spike",
        align: "right",
        render: (row) => formatMetricValue(row.resid, 2),
      },
    ],
    [resolvedTimeZone]
  )

  // Imputation audit normalization
  const audit = data?.imputation_audit
  const auditNormalized = useMemo(() => {
    if (!audit) {
      return {
        isAvailable: false,
        sourceLabel: "Unavailable",
        totalRows: null as number | null,
        imputedRows: null as number | null,
        observedPct: null as number | null,
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
        observedPct: null,
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

    const observedPct =
      totalRows && totalRows > 0 && imputedRows !== null
        ? Math.max(0, Math.min(100, ((totalRows - imputedRows) / totalRows) * 100))
        : null

    const tierLabelMap: Record<string, string> = {
      none: "Direct sensor observation",
      tier1_linear: "Short gap fill (≤3h linear)",
      tier2_diurnal_median: "Medium gap fill (4–24h daily median)",
      tier3_climatology_ffill: "Long gap fill (>24h baseline)",
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
      observedPct,
      tierEntries,
      missingEntries,
    }
  }, [audit, metricSpecsHint, distributions])

  // Highlight diurnal row for Key Findings strip
  const headlineDiurnalRow = useMemo(() => {
    const temp = diurnalRows.find((r) => r.metricKey === "temperature_2m")
    return temp ?? diurnalRows[0] ?? null
  }, [diurnalRows])

  if (isLoading && !data) {
    return (
      <div className="flex flex-col gap-6">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div className="h-24 animate-pulse rounded-xl border border-border bg-muted" />
          <div className="h-24 animate-pulse rounded-xl border border-border bg-muted" />
          <div className="h-24 animate-pulse rounded-xl border border-border bg-muted" />
        </div>
        <div className="h-[340px] animate-pulse rounded-xl border border-border bg-muted" />
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
  const quickMetricPills = allPatternMetricKeys.slice(0, 6)
  const decompExplainer =
    decompView === "cycle"
      ? "Repeating 24-hour push above or below the daily average."
      : decompView === "trend"
        ? "Smoothed multi-day warming or cooling drift with daily cycles removed."
        : "Unexpected residual spikes not explained by the daily rhythm or multi-day trend."

  return (
    <div className="flex flex-col gap-6 md:gap-8">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
              Exploratory data analysis
            </h2>
            <p className="mt-1 max-w-[68ch] text-sm/5 text-muted-foreground">
              Key patterns, daily weather rhythms, variable relationships, and
              dataset health across historical observations.
            </p>
          </div>
          {isFetching ? (
            <span className="text-xs font-medium text-muted-foreground">
              Updating view...
            </span>
          ) : null}
        </div>

        {/* Auto-Insights Key Findings Strip */}
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {/* Card 1: Dataset coverage & health */}
          <div className="flex flex-col justify-between gap-2 rounded-xl border border-border bg-card p-4 shadow-(--shadow-card)">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <Database className="size-3.5 text-primary" strokeWidth={1.75} />
                Dataset health
              </span>
              {auditNormalized.observedPct !== null ? (
                <Badge variant="secondary" className="h-5 px-2 text-[11px] tabular-nums">
                  {auditNormalized.observedPct.toFixed(1)}% direct
                </Badge>
              ) : null}
            </div>
            <div>
              <p className="font-display text-h2 font-semibold text-foreground tabular-nums">
                {totalSamples.toLocaleString()} hours
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground">
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
                      showZone={false}
                    />
                  </span>
                ) : (
                  "Historical observation window"
                )}
              </p>
            </div>
          </div>

          {/* Card 2: Daily rhythm highlight */}
          <div className="flex flex-col justify-between gap-2 rounded-xl border border-border bg-card p-4 shadow-(--shadow-card)">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <Clock className="size-3.5 text-primary" strokeWidth={1.75} />
                Daily rhythm ({tzShortLabel})
              </span>
              {headlineDiurnalRow?.peakHour !== null &&
              headlineDiurnalRow?.peakHour !== undefined ? (
                <Badge variant="secondary" className="h-5 px-2 text-[11px] tabular-nums">
                  Peak {formatHourPadded(headlineDiurnalRow.peakHour)}
                </Badge>
              ) : null}
            </div>
            {headlineDiurnalRow &&
            headlineDiurnalRow.minVal !== null &&
            headlineDiurnalRow.maxVal !== null ? (
              <div>
                <p className="font-display text-h2 font-semibold text-foreground tabular-nums">
                  {(headlineDiurnalRow.maxVal - headlineDiurnalRow.minVal).toFixed(1)}
                  {headlineDiurnalRow.unit ? ` ${headlineDiurnalRow.unit}` : ""}{" "}
                  swing
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {headlineDiurnalRow.metricLabel} is typically lowest at{" "}
                  {headlineDiurnalRow.lowHour !== null
                    ? formatHourPadded(headlineDiurnalRow.lowHour)
                    : "—"}{" "}
                  and highest at{" "}
                  {headlineDiurnalRow.peakHour !== null
                    ? formatHourPadded(headlineDiurnalRow.peakHour)
                    : "—"}
                </p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                Hourly diurnal cycles will summarize once multi-day data is recorded.
              </p>
            )}
          </div>

          {/* Card 3: Strongest relationship */}
          <div className="flex flex-col justify-between gap-2 rounded-xl border border-border bg-card p-4 shadow-(--shadow-card)">
            <div className="flex items-center justify-between text-xs font-medium text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <GitCompareArrows
                  className="size-3.5 text-primary"
                  strokeWidth={1.75}
                />
                Strongest link
              </span>
              {strongestCorrPair ? (
                <Badge variant="secondary" className="h-5 px-2 text-[11px] tabular-nums">
                  r = {strongestCorrPair.r > 0 ? "+" : ""}
                  {strongestCorrPair.r.toFixed(2)}
                </Badge>
              ) : null}
            </div>
            {strongestCorrPair ? (
              <div>
                <p className="truncate font-display text-h2 font-semibold text-foreground">
                  {strongestCorrPair.a} &amp; {strongestCorrPair.b}
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {strongestCorrPair.r >= 0
                    ? "Tend to rise and fall together"
                    : "Tend to move in opposite directions"}
                </p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                Pairwise variable relationships appear once enough paired hours exist.
              </p>
            )}
          </div>
        </div>

        {/* Clean 3-Section Sub-Tab Switcher */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
          <div
            role="tablist"
            aria-label="EDA sections"
            className="inline-flex flex-wrap items-center gap-1 rounded-full bg-muted p-1"
          >
            {(
              [
                { id: "patterns", label: "Daily patterns & spread" },
                { id: "relationships", label: "Variable relationships" },
                { id: "trends", label: "Trends & data quality" },
              ] as const
            ).map((tab) => {
              const active = activeSubTab === tab.id
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setActiveSubTab(tab.id)}
                  className={cn(
                    "inline-flex h-8 items-center rounded-full px-3.5 text-xs font-medium transition-colors duration-120",
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

          {data.computed_at_utc ? (
            <span className="text-xs text-muted-foreground">
              Last computed{" "}
              <TimestampPkt
                timestamp={data.computed_at_utc}
                timeZone={resolvedTimeZone}
                mode="datetime"
                className="font-medium text-foreground"
              />
            </span>
          ) : null}
        </div>
      </div>

      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh exploratory analysis"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      ) : null}

      {/* ================================================================= */}
      {/* TAB 1: DAILY PATTERNS & SPREAD                                    */}
      {/* ================================================================= */}
      {activeSubTab === "patterns" ? (
        <div className="flex flex-col gap-6">
          {/* Shared Metric Selector driving both cards */}
          {allPatternMetricKeys.length > 0 ? (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div
                role="tablist"
                aria-label="Quick variable selector"
                className="flex flex-wrap items-center gap-1.5"
              >
                {quickMetricPills.map((key) => {
                  const active = key === activeMetricKey
                  const label = resolveMetricLabel(
                    key,
                    metricSpecsHint,
                    distributions
                  )
                  return (
                    <button
                      key={key}
                      type="button"
                      role="tab"
                      aria-selected={active}
                      onClick={() => setSelectedMetricKey(key)}
                      className={cn(
                        "inline-flex h-8 items-center rounded-full border px-3 text-xs font-medium transition-colors duration-120",
                        active
                          ? "border-primary/50 bg-accent text-accent-foreground"
                          : "border-border bg-card text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {label}
                    </button>
                  )
                })}
              </div>

              {allPatternMetricKeys.length > quickMetricPills.length ? (
                <Select
                  value={activeMetricKey}
                  onValueChange={setSelectedMetricKey}
                >
                  <SelectTrigger
                    aria-label="Select any weather or pollutant variable"
                    className="h-8 min-w-[180px] bg-card text-xs"
                  >
                    <SelectValue placeholder="All variables" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    {allPatternMetricKeys.map((key) => {
                      const entry = distributions[key]
                      const unavail = entry && entry.status !== "available"
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
            </div>
          ) : null}

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            {/* Card 1A: 24-Hour Daily Rhythm (Smooth Curve default | All-metrics Heatmap) */}
            <ChartCard
              title={
                diurnalDisplay === "curve"
                  ? `Typical 24-hour ${activeMetricLabel.toLowerCase()} rhythm (${tzShortLabel})`
                  : `24-hour rhythm across all metrics (${tzShortLabel})`
              }
              titleInfo="Shows how weather changes from midnight (00:00) through nightfall (23:00) in local time. Switch to All metrics to compare peak hours side-by-side."
              subtitle={
                diurnalDisplay === "curve" && activeDiurnalRow
                  ? `Average hourly ${activeDiurnalRow.metricLabel.toLowerCase()} across historical days`
                  : "Shaded per row from lowest hour (pale) to peak hour (deep blue)"
              }
              ariaLabel="24-hour daily rhythm"
              controls={
                <div className="flex flex-wrap items-center gap-1.5">
                  <div
                    role="group"
                    aria-label="Daily rhythm display"
                    className="inline-flex h-8 items-center rounded-md bg-muted p-0.5"
                  >
                    {(
                      [
                        { id: "curve", label: "Curve" },
                        { id: "heatmap", label: "All metrics" },
                      ] as const
                    ).map((opt) => {
                      const active = diurnalDisplay === opt.id
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          aria-pressed={active}
                          onClick={() => setDiurnalDisplay(opt.id)}
                          className={cn(
                            "inline-flex h-7 items-center rounded-sm px-2.5 text-xs font-medium transition-colors duration-120",
                            active
                              ? "bg-card text-foreground shadow-(--shadow-card)"
                              : "text-foreground/75 hover:text-foreground"
                          )}
                        >
                          {opt.label}
                        </button>
                      )
                    })}
                  </div>

                  <div
                    role="group"
                    aria-label="Diurnal statistic"
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
                            "inline-flex h-7 items-center rounded-sm px-2 text-xs font-medium capitalize transition-colors duration-120",
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
                </div>
              }
              subHeader={
                diurnalDisplay === "curve" &&
                activeDiurnalRow &&
                activeDiurnalRow.minVal !== null &&
                activeDiurnalRow.maxVal !== null ? (
                  <dl className="grid grid-cols-2 gap-2.5">
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <dt className="text-xs text-muted-foreground">
                        Lowest hour ({tzShortLabel})
                      </dt>
                      <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                        {activeDiurnalRow.lowHour !== null
                          ? formatHourPadded(activeDiurnalRow.lowHour)
                          : "—"}{" "}
                        <span className="font-normal text-muted-foreground">
                          ({formatMetricValue(activeDiurnalRow.minVal, 1)}
                          {activeDiurnalRow.unit ? ` ${activeDiurnalRow.unit}` : ""})
                        </span>
                      </dd>
                    </div>
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <dt className="text-xs text-muted-foreground">
                        Peak hour ({tzShortLabel})
                      </dt>
                      <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                        {activeDiurnalRow.peakHour !== null
                          ? formatHourPadded(activeDiurnalRow.peakHour)
                          : "—"}{" "}
                        <span className="font-normal text-muted-foreground">
                          ({formatMetricValue(activeDiurnalRow.maxVal, 1)}
                          {activeDiurnalRow.unit ? ` ${activeDiurnalRow.unit}` : ""})
                        </span>
                      </dd>
                    </div>
                  </dl>
                ) : undefined
              }
              tableColumns={diurnalTableColumns}
              tableRows={diurnalTableRows}
              emptyTableMessage="No diurnal buckets recorded."
              footer={
                diurnalDisplay === "heatmap" ? (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span>Each row scaled independently from low to peak</span>
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
                      <span>Peak</span>
                    </div>
                  </div>
                ) : (
                  <span>
                    Hours shown in local {tzShortLabel} time · Hover any hour for exact mean and median
                  </span>
                )
              }
            >
              {diurnalRows.length === 0 ? (
                <EmptyState
                  icon={Grid2x2}
                  title="Daily rhythm unavailable"
                  description="Hourly diurnal profiles appear once multi-day local timestamp buckets are computed."
                  className="border-0 bg-muted shadow-none"
                />
              ) : diurnalDisplay === "curve" ? (
                <ChartContainer
                  config={{
                    value: {
                      label: activeDiurnalRow?.metricLabel ?? activeMetricLabel,
                      color: activeMetricColor,
                    },
                  }}
                  className="h-[240px] w-full aspect-auto md:h-[260px]"
                >
                  <ComposedChart
                    data={diurnalCurveRows}
                    margin={{ top: 10, right: 12, bottom: 4, left: 0 }}
                  >
                    <CartesianGrid vertical={false} stroke="var(--border)" />
                    <XAxis
                      dataKey="hourLabel"
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                      minTickGap={18}
                    />
                    <YAxis
                      domain={[diurnalCurveMin, diurnalCurveMax]}
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
                          | DiurnalCurveRow
                          | undefined
                        if (!row) return null
                        const unit = activeDiurnalRow?.unit ?? ""
                        const rows: TooltipEntryRow[] = []
                        if (row.mean !== null) {
                          rows.push({
                            key: "mean",
                            label: "Mean",
                            valueText: `${formatMetricValue(row.mean, 1)}${unit ? ` ${unit}` : ""}`,
                            color: activeMetricColor,
                          })
                        }
                        if (row.median !== null) {
                          rows.push({
                            key: "median",
                            label: "Median",
                            valueText: `${formatMetricValue(row.median, 1)}${unit ? ` ${unit}` : ""}`,
                            color: "var(--muted-foreground)",
                          })
                        }
                        return (
                          <SkyCastChartTooltipCard
                            rows={rows}
                            footerNote={`${formatHourPadded(row.hour)} ${tzShortLabel} · ${row.count} days sampled`}
                          />
                        )
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="none"
                      fill={activeMetricColor}
                      fillOpacity={0.14}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={activeMetricColor}
                      strokeWidth={2.25}
                      dot={false}
                      activeDot={{ r: 5, strokeWidth: 2 }}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ChartContainer>
              ) : (
                <div className="overflow-x-auto pb-1">
                  <div className="inline-block min-w-full align-middle">
                    <div className="flex items-center gap-1 pb-1.5">
                      <div className="sticky left-0 z-10 w-[120px] shrink-0 bg-card pr-2 text-xs font-medium text-muted-foreground sm:w-[136px]">
                        Metric
                      </div>
                      <div className="flex items-center gap-1">
                        {diurnalHours.map((hour) => (
                          <div
                            key={hour}
                            className="w-5 shrink-0 text-center text-[11px] text-muted-foreground tabular-nums sm:w-6"
                          >
                            {formatHourShort(hour)}
                          </div>
                        ))}
                      </div>
                      <div className="w-12 shrink-0 pl-2 text-right text-[11px] font-medium text-muted-foreground">
                        Peak
                      </div>
                    </div>

                    <div className="flex flex-col gap-1">
                      {diurnalRows.map((row) => (
                        <div
                          key={row.metricKey}
                          className="flex items-center gap-1"
                        >
                          <div
                            className="sticky left-0 z-10 w-[120px] shrink-0 truncate bg-card pr-2 text-xs font-medium text-foreground sm:w-[136px]"
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
                                        "flex size-5 shrink-0 items-center justify-center rounded-xs border border-border/40 text-[10px] transition-transform duration-120 hover:scale-105 focus-visible:outline-2 focus-visible:outline-ring sm:size-6",
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
                                      {row.metricLabel} at{" "}
                                      {formatHourPadded(hour)} {tzShortLabel}
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
                                      </>
                                    ) : (
                                      <span className="text-muted-foreground">
                                        No observations in this hour
                                      </span>
                                    )}
                                  </TooltipContent>
                                </Tooltip>
                              )
                            })}
                          </div>

                          <div className="w-12 shrink-0 pl-2 text-right text-[11px] font-medium text-foreground tabular-nums">
                            {row.peakHour !== null
                              ? formatHourPadded(row.peakHour)
                              : "—"}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </ChartCard>

            {/* Card 1B: Typical Values & Spread (Histogram) */}
            <ChartCard
              title={`${activeMetricLabel} typical values & spread`}
              titleInfo="Shows how often different readings occurred in the historical dataset. Taller bars mark the most common values."
              subtitle={
                activeDist && activeDist.status === "available"
                  ? `Most readings fall between ${formatMetricValue(activeDist.q05, 1)} and ${formatMetricValue(activeDist.q95, 1)}${activeDistUnit ? ` ${activeDistUnit}` : ""}`
                  : "Frequency of historical readings across value ranges"
              }
              ariaLabel={`${activeMetricLabel} value distribution histogram`}
              subHeader={
                activeDist && activeDist.status === "available" ? (
                  <dl className="grid grid-cols-3 gap-2.5">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="rounded-lg bg-muted px-3 py-2">
                          <dt className="text-xs text-muted-foreground">
                            Typical
                          </dt>
                          <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                            {formatMetricValue(activeDist.q50, 1)}
                            {activeDistUnit ? ` ${activeDistUnit}` : ""}
                          </dd>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="text-xs tabular-nums">
                        Median (P50): {formatMetricValue(activeDist.q50, 2)} · Mean:{" "}
                        {formatMetricValue(activeDist.mean, 2)} · Std dev:{" "}
                        {formatMetricValue(activeDist.std, 2)}
                      </TooltipContent>
                    </Tooltip>

                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="rounded-lg bg-muted px-3 py-2">
                          <dt className="text-xs text-muted-foreground">
                            Usual range
                          </dt>
                          <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                            {formatMetricValue(activeDist.q05, 1)} –{" "}
                            {formatMetricValue(activeDist.q95, 1)}
                          </dd>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="text-xs">
                        Middle 90% of all recorded hours (5th to 95th percentile).
                      </TooltipContent>
                    </Tooltip>

                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="rounded-lg bg-muted px-3 py-2">
                          <dt className="text-xs text-muted-foreground">
                            Min – Max
                          </dt>
                          <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                            {formatMetricValue(activeDist.min, 1)} –{" "}
                            {formatMetricValue(activeDist.max, 1)}
                          </dd>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="text-xs tabular-nums">
                        Valid: {activeDist.valid_count.toLocaleString()} · Missing:{" "}
                        {activeDist.missing_count.toLocaleString()}
                      </TooltipContent>
                    </Tooltip>
                  </dl>
                ) : undefined
              }
              tableColumns={histogramTableColumns}
              tableRows={histogramRows}
              emptyTableMessage="No histogram bins available for the selected metric."
              footer={
                activeDist && isFiniteNum(activeDist.circular_mean_deg) ? (
                  <span className="tabular-nums">
                    Prevailing direction:{" "}
                    <span className="font-medium text-foreground">
                      {activeDist.circular_mean_deg.toFixed(0)}°
                    </span>
                  </span>
                ) : activeDist && activeDist.status === "available" ? (
                  <span className="tabular-nums">
                    Based on {activeDist.valid_count.toLocaleString()} valid
                    observations · Hover tiles above for exact percentiles
                  </span>
                ) : undefined
              }
            >
              {!activeDist ? (
                <EmptyState
                  icon={Grid2x2}
                  title="Distribution unavailable"
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
                      Valid rows: {activeDist.valid_count.toLocaleString()},
                      missing rows: {activeDist.missing_count.toLocaleString()}
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
                    count: { label: "Observations", color: activeMetricColor },
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
                                label: `${row.label}${activeDistUnit ? ` ${activeDistUnit}` : ""}`,
                                valueText: `${row.count.toLocaleString()} hours`,
                                color: activeMetricColor,
                              },
                            ]}
                            footerNote={
                              share !== null ? `${share}% of all hours` : null
                            }
                          />
                        )
                      }}
                    />
                    <Bar
                      dataKey="count"
                      fill={activeMetricColor}
                      radius={[4, 4, 0, 0]}
                      isAnimationActive={false}
                    />
                  </BarChart>
                </ChartContainer>
              )}
            </ChartCard>
          </div>
        </div>
      ) : activeSubTab === "relationships" ? (
        /* =============================================================== */
        /* TAB 2: VARIABLE RELATIONSHIPS (TOP LINKS + LOWER-TRIANGLE GRID) */
        /* =============================================================== */
        <EdaCorrelationsCard
          method={data.correlation_matrix?.method ?? "pearson"}
          minPeriods={data.correlation_matrix?.min_periods ?? 1}
          columns={corrColumns}
          matrix={corrMatrix}
          constantColumns={data.correlation_matrix?.constant_columns ?? []}
          metricSpecsHint={metricSpecsHint}
          distributions={distributions}
        />
      ) : (
        /* =============================================================== */
        /* TAB 3: TRENDS & DATA QUALITY                                    */
        /* =============================================================== */
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          {/* Card 3A: Time-Series Decomposition */}
          <ChartCard
            title="Trend & daily pattern breakdown"
            titleInfo="Seasonal decomposition separates a weather series into three layers: the repeating 24-hour daily rhythm, the slower multi-day trend, and unusual leftover spikes."
            subtitle={decompExplainer}
            ariaLabel="Time-series seasonal decomposition chart"
            controls={
              <div className="flex flex-wrap items-center gap-2">
                {seasonalKeys.length > 0 ? (
                  <Select
                    value={activeDecompKey}
                    onValueChange={setSelectedDecompKey}
                  >
                    <SelectTrigger
                      aria-label="Select decomposition metric"
                      className="h-8 min-w-[144px] bg-card text-xs"
                    >
                      <SelectValue placeholder="Metric" />
                    </SelectTrigger>
                    <SelectContent position="popper">
                      {seasonalKeys.map((key) => {
                        const entry = seasonalMap[key]
                        const unavail = entry?.status !== "available"
                        return (
                          <SelectItem key={key} value={key}>
                            {resolveMetricLabel(
                              key,
                              metricSpecsHint,
                              distributions
                            )}
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
                    aria-label="Decomposition layer"
                    className="inline-flex h-8 items-center rounded-md bg-muted p-0.5"
                  >
                    {(
                      [
                        { id: "cycle", label: "Daily rhythm" },
                        { id: "trend", label: "Multi-day trend" },
                        { id: "components", label: "Spikes" },
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
                <dl className="grid grid-cols-3 gap-2.5">
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="text-xs text-muted-foreground">Average</dt>
                    <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                      {formatMetricValue(activeDecomp.summary?.observed_mean, 1)}
                      {activeDecompUnit ? ` ${activeDecompUnit}` : ""}
                    </dd>
                  </div>
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="text-xs text-muted-foreground">
                      Trend shift
                    </dt>
                    <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                      {formatMetricValue(activeDecomp.summary?.trend_start, 1)} →{" "}
                      {formatMetricValue(activeDecomp.summary?.trend_end, 1)}
                    </dd>
                  </div>
                  <div className="rounded-lg bg-muted px-3 py-2">
                    <dt className="text-xs text-muted-foreground">
                      Daily swing
                    </dt>
                    <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                      ±
                      {formatMetricValue(
                        activeDecomp.summary?.seasonal_amplitude,
                        1
                      )}
                      {activeDecompUnit ? ` ${activeDecompUnit}` : ""}
                    </dd>
                  </div>
                </dl>
              ) : undefined
            }
            tableColumns={
              decompView === "cycle"
                ? (seasonalTableColumns as ChartTableColumn<Record<string, unknown>>[])
                : (seasonalSeriesTableColumns as ChartTableColumn<
                    Record<string, unknown>
                  >[])
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
                  {activeDecomp.sample_count_used.toLocaleString()} contiguous
                  hours ({activeDecomp.period}h period) · Residual std:{" "}
                  {formatMetricValue(activeDecomp.summary?.residual_std, 2)}
                </span>
              ) : undefined
            }
          >
            {!activeDecomp ? (
              <EmptyState
                icon={TrendingUp}
                title="Decomposition unavailable"
                description="Trend and daily pattern diagnostics appear once sufficient contiguous hourly observations are collected."
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
                    {activeDecompLabel} decomposition unavailable
                  </p>
                  <p className="mt-1 text-xs text-foreground/80">
                    {activeDecomp.reason}
                  </p>
                  <p className="mt-2 text-xs text-foreground/80 tabular-nums">
                    Valid rows: {activeDecomp.valid_count.toLocaleString()} of{" "}
                    {activeDecomp.required_rows.toLocaleString()} required
                  </p>
                </div>
              </div>
            ) : decompView === "cycle" ? (
              seasonalCycleRows.length === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No 24h cycle points"
                  description="Seasonal cycle weights were not returned for this variable."
                  className="border-0 bg-muted shadow-none"
                />
              ) : (
                <ChartContainer
                  config={{
                    seasonal: {
                      label: "Daily rhythm",
                      color: activeDecompColor,
                    },
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
                    <Bar
                      dataKey="seasonal"
                      radius={[3, 3, 0, 0]}
                      isAnimationActive={false}
                    >
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
                    observed: {
                      label: "Observed",
                      color: "var(--muted-foreground)",
                    },
                    trend: { label: "Multi-day trend", color: activeDecompColor },
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
                            label: "Multi-day trend",
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
                      strokeOpacity={0.45}
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
                      label: "Hourly observed",
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
                    resid: { label: "Unusual spike", color: "var(--chart-6)" },
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
                        if (row.resid !== null) {
                          rows.push({
                            key: "resid",
                            label: "Unusual residual",
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
                      dataKey="resid"
                      stroke="var(--chart-6)"
                      strokeWidth={1.75}
                      dot={false}
                      connectNulls={false}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ChartContainer>
                <ChartLegendBar
                  items={[
                    {
                      key: "resid",
                      label: "Residual (unexplained deviation)",
                      color: "var(--chart-6)",
                      variant: "solid",
                    },
                  ]}
                />
              </div>
            )}
          </ChartCard>

          {/* Card 3B: Data Quality & Gap-Filling Audit */}
          <section className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-base/6 font-semibold text-card-foreground">
                  Data quality &amp; gap filling
                </h3>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  How much of the dataset came directly from sensors versus
                  3-tier gap filling
                </p>
              </div>
              {audit ? (
                <Badge
                  variant="secondary"
                  className="h-6 px-2.5 text-xs font-medium"
                >
                  {auditNormalized.sourceLabel}
                </Badge>
              ) : null}
            </div>

            {!auditNormalized.isAvailable ? (
              <EmptyState
                icon={ShieldAlert}
                title="Data quality audit unavailable"
                description="Gap-filling audit details will appear once observation quality checks run."
                className="border-0 bg-muted shadow-none"
              />
            ) : (
              <div className="flex flex-col gap-4">
                {/* Visual Data Health Summary */}
                {auditNormalized.observedPct !== null ? (
                  <div className="rounded-lg bg-muted p-4">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-xs font-medium text-muted-foreground">
                        Direct sensor coverage
                      </span>
                      <span className="font-display text-h2 font-semibold text-foreground tabular-nums">
                        {auditNormalized.observedPct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-2 h-2.5 w-full overflow-hidden rounded-full bg-card">
                      <div
                        className="h-full rounded-full bg-aqi-good transition-all duration-150"
                        style={{
                          width: `${Math.max(2, auditNormalized.observedPct)}%`,
                        }}
                      />
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground tabular-nums">
                      {auditNormalized.totalRows?.toLocaleString()} total hours ·{" "}
                      {auditNormalized.imputedRows?.toLocaleString()} gap-filled
                    </p>
                  </div>
                ) : (
                  <dl className="grid grid-cols-2 gap-2.5">
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <dt className="text-xs text-muted-foreground">
                        Total hours
                      </dt>
                      <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                        {auditNormalized.totalRows !== null
                          ? auditNormalized.totalRows.toLocaleString()
                          : "—"}
                      </dd>
                    </div>
                    <div className="rounded-lg bg-muted px-3 py-2">
                      <dt className="text-xs text-muted-foreground">
                        Gap-filled hours
                      </dt>
                      <dd className="mt-0.5 text-sm font-semibold text-foreground tabular-nums">
                        {auditNormalized.imputedRows !== null
                          ? auditNormalized.imputedRows.toLocaleString()
                          : "—"}
                      </dd>
                    </div>
                  </dl>
                )}

                {auditNormalized.tierEntries.length > 0 ? (
                  <div className="flex flex-col gap-2.5">
                    <div className="text-xs font-medium text-foreground">
                      Gap-filling method breakdown
                    </div>
                    {auditNormalized.tierEntries.map((entry) => (
                      <div key={entry.tier} className="flex flex-col gap-1">
                        <div className="flex items-center justify-between text-xs text-foreground tabular-nums">
                          <span>{entry.label}</span>
                          <span className="font-semibold">
                            {entry.count.toLocaleString()} ({entry.pct.toFixed(1)}%)
                          </span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${Math.max(2, entry.pct)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null}

                {/* Collapsible Per-Variable Missing Table */}
                {auditNormalized.missingEntries.length > 0 ? (
                  <div className="flex flex-col gap-2 border-t border-border pt-3">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      aria-expanded={showMissingTable}
                      onClick={() => setShowMissingTable((prev) => !prev)}
                      className="h-8 justify-between px-2 text-xs font-medium text-foreground"
                    >
                      <span>
                        Inspect per-variable coverage (
                        {auditNormalized.missingEntries.length} variables)
                      </span>
                      {showMissingTable ? (
                        <ChevronUp className="size-3.5" strokeWidth={1.75} />
                      ) : (
                        <ChevronDown className="size-3.5" strokeWidth={1.75} />
                      )}
                    </Button>

                    {showMissingTable ? (
                      <Table
                        aria-label="Observed missing counts by variable"
                        containerClassName="max-h-60 overflow-auto rounded-md border border-border"
                      >
                        <TableHeader className="sticky top-0 z-10 bg-muted">
                          <TableRow className="hover:bg-muted">
                            <TableHead className="h-9 px-3 py-2 text-xs font-medium text-foreground">
                              Variable
                            </TableHead>
                            <TableHead className="h-9 px-3 py-2 text-right text-xs font-medium text-foreground">
                              Missing
                            </TableHead>
                            <TableHead className="h-9 px-3 py-2 text-right text-xs font-medium text-foreground">
                              Share
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {auditNormalized.missingEntries.map((entry) => (
                            <TableRow key={entry.column} className="h-10">
                              <TableCell className="px-3 py-2 text-xs font-medium text-foreground">
                                {entry.label}
                              </TableCell>
                              <TableCell className="px-3 py-2 text-right text-xs tabular-nums">
                                {entry.missingCount.toLocaleString()}
                              </TableCell>
                              <TableCell className="px-3 py-2 text-right text-xs text-muted-foreground tabular-nums">
                                {entry.missingPct !== null
                                  ? `${entry.missingPct.toFixed(1)}%`
                                  : "—"}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : null}
                  </div>
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
      )}
    </div>
  )
}
