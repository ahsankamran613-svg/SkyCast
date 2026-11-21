import { useMemo, useState } from "react"
import { ArrowDownRight, ArrowUpRight, ChevronDown, ChevronUp, Sparkles } from "lucide-react"
import { cn } from "cn"

import { ChartCard, type ChartTableColumn } from "@/components/shared/chart-card"
import { formatValueWithUnit } from "@/components/shared/chart-primitives"
import { EmptyState } from "@/components/shared/empty-state"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type {
  ExplanationRecord,
  LimeLocalAttribution,
  ShapGlobalAttribution,
  ShapLocalAttribution,
} from "@/lib/api"
import { formatLocalDateTime } from "@/lib/time"
import {
  describeFeature,
  formatLimeRule,
  formatMetricValue,
  formatModelLabel,
  humanizeLimeRule,
} from "@/lib/weather"

export interface ExplainabilityPanelProps {
  activeLabel: string
  activeUnit: string
  resolvedTimeZone: string | null
  activeChampionFamily: string | null
  shapGlobalRecords: ExplanationRecord<ShapGlobalAttribution>[]
  shapLocalRecords: ExplanationRecord<ShapLocalAttribution>[]
  limeLocalRecords: ExplanationRecord<LimeLocalAttribution>[]
}

type ExplainViewMode = "local_shap" | "global_shap" | "lime"

interface ShapGlobalRow extends Record<string, unknown> {
  feature: string
  label: string
  description: string
  unit: string
  magnitude: number
  sharePct: number
  signedMean: number | null
  referenceValue: number | null
}

interface ShapLocalRow extends Record<string, unknown> {
  feature: string
  label: string
  description: string
  unit: string
  contribution: number
  start: number
  end: number
  value: number | null
}

interface LimeRuleRow extends Record<string, unknown> {
  feature: string
  label: string
  description: string
  unit: string
  humanRule: string
  rawRule: string
  weight: number
  value: number | null
}

const DEFAULT_VISIBLE_DRIVERS = 5

function isFiniteNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatSignedValue(value: number | null | undefined, decimals = 2): string {
  if (!isFiniteNum(value)) return "—"
  const prefix = value > 0 ? "+" : ""
  return `${prefix}${value.toFixed(decimals)}`
}

function formatSignedWithUnit(
  value: number | null | undefined,
  unit: string,
  decimals = 2
): string {
  if (!isFiniteNum(value)) return "—"
  const prefix = value > 0 ? "+" : ""
  return `${prefix}${value.toFixed(decimals)}${unit ? ` ${unit}` : ""}`
}

export function ExplainabilityPanel({
  activeLabel,
  activeUnit,
  resolvedTimeZone,
  activeChampionFamily,
  shapGlobalRecords,
  shapLocalRecords,
  limeLocalRecords,
}: ExplainabilityPanelProps) {
  const [selectedView, setSelectedView] = useState<ExplainViewMode | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [selectedShapLocalIndex, setSelectedShapLocalIndex] = useState<number>(0)
  const [selectedLimeLocalIndex, setSelectedLimeLocalIndex] = useState<number>(0)
  const [showAllDrivers, setShowAllDrivers] = useState<boolean>(false)

  const explainedModels = useMemo(() => {
    const models = new Set<string>()
    for (const rec of shapLocalRecords) models.add(rec.model_family)
    for (const rec of shapGlobalRecords) models.add(rec.model_family)
    for (const rec of limeLocalRecords) models.add(rec.model_family)
    return Array.from(models)
  }, [shapGlobalRecords, shapLocalRecords, limeLocalRecords])

  const activeModel = useMemo(() => {
    if (selectedModel && explainedModels.includes(selectedModel)) {
      return selectedModel
    }
    if (activeChampionFamily && explainedModels.includes(activeChampionFamily)) {
      return activeChampionFamily
    }
    return explainedModels[0] ?? ""
  }, [selectedModel, explainedModels, activeChampionFamily])

  const modelShapGlobal = useMemo(
    () => shapGlobalRecords.find((r) => r.model_family === activeModel) ?? null,
    [shapGlobalRecords, activeModel]
  )

  const modelShapLocals = useMemo(
    () => shapLocalRecords.filter((r) => r.model_family === activeModel),
    [shapLocalRecords, activeModel]
  )

  const modelLimeLocals = useMemo(
    () => limeLocalRecords.filter((r) => r.model_family === activeModel),
    [limeLocalRecords, activeModel]
  )

  // Pick a smart default view so the user never lands on an empty tab if another has data.
  const activeView: ExplainViewMode = useMemo(() => {
    if (selectedView) return selectedView
    if (modelShapLocals.length > 0) return "local_shap"
    if (modelShapGlobal) return "global_shap"
    if (modelLimeLocals.length > 0) return "lime"
    return "local_shap"
  }, [selectedView, modelShapLocals.length, modelShapGlobal, modelLimeLocals.length])

  const activeShapLocal =
    modelShapLocals[
      Math.min(selectedShapLocalIndex, Math.max(0, modelShapLocals.length - 1))
    ] ?? null

  const activeLimeLocal =
    modelLimeLocals[
      Math.min(selectedLimeLocalIndex, Math.max(0, modelLimeLocals.length - 1))
    ] ?? null

  // Global SHAP rows with normalized percentage share
  const shapGlobalRows = useMemo<ShapGlobalRow[]>(() => {
    const attrs = modelShapGlobal?.feature_attributions ?? []
    if (attrs.length === 0) return []

    const parsed = attrs
      .map((attr) => {
        const meanAbs = isFiniteNum(attr.mean_abs_shap)
          ? Math.abs(attr.mean_abs_shap)
          : null
        const imp = isFiniteNum(attr.importance) ? Math.abs(attr.importance) : null
        const signedMean = isFiniteNum(attr.mean_shap) ? attr.mean_shap : null
        const magnitude =
          meanAbs ?? imp ?? (signedMean !== null ? Math.abs(signedMean) : 0)
        const meta = describeFeature(attr.feature)
        return {
          feature: attr.feature,
          label: meta.label,
          description: meta.description,
          unit: meta.unit,
          magnitude,
          signedMean,
          referenceValue: isFiniteNum(attr.value) ? attr.value : null,
        }
      })
      .filter((row) => row.magnitude > 0)
      .sort((a, b) => b.magnitude - a.magnitude)

    const totalMag = parsed.reduce((sum, r) => sum + r.magnitude, 0)
    return parsed.slice(0, 10).map((row) => ({
      ...row,
      sharePct: totalMag > 0 ? (row.magnitude / totalMag) * 100 : 0,
    }))
  }, [modelShapGlobal])

  const maxShapGlobalMag = useMemo(
    () => (shapGlobalRows.length > 0 ? Math.max(...shapGlobalRows.map((r) => r.magnitude)) : 0),
    [shapGlobalRows]
  )

  // Local SHAP rows
  const { shapLocalRows, maxShapLocalAbs, localPositiveSum, localNegativeSum } = useMemo(() => {
    const attrs = activeShapLocal?.feature_attributions ?? []
    if (attrs.length === 0) {
      return {
        shapLocalRows: [] as ShapLocalRow[],
        maxShapLocalAbs: 0,
        localPositiveSum: 0,
        localNegativeSum: 0,
      }
    }

    const sorted = [...attrs]
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

    const base = isFiniteNum(activeShapLocal?.base_value) ? activeShapLocal.base_value : 0
    let running = base
    let posSum = 0
    let negSum = 0

    const rows: ShapLocalRow[] = []
    for (const { attr, val } of sorted) {
      const start = running
      const end = running + val
      running = end
      if (val >= 0) posSum += val
      else negSum += val
      const meta = describeFeature(attr.feature)
      rows.push({
        feature: attr.feature,
        label: meta.label,
        description: meta.description,
        unit: meta.unit,
        contribution: val,
        start,
        end,
        value: isFiniteNum(attr.value) ? attr.value : null,
      })
    }

    const maxAbs = rows.length > 0 ? Math.max(...rows.map((r) => Math.abs(r.contribution))) : 0
    return {
      shapLocalRows: rows,
      maxShapLocalAbs: maxAbs,
      localPositiveSum: posSum,
      localNegativeSum: negSum,
    }
  }, [activeShapLocal])

  // LIME rows
  const limeRows = useMemo<LimeRuleRow[]>(() => {
    const attrs = activeLimeLocal?.feature_attributions ?? []
    return attrs
      .map((attr) => {
        const w = isFiniteNum(attr.weight)
          ? attr.weight
          : isFiniteNum(attr.contribution)
            ? attr.contribution
            : isFiniteNum(attr.importance)
              ? attr.importance
              : null
        const meta = describeFeature(attr.feature)
        const rawRule = attr.rule?.trim()
          ? formatLimeRule(attr.rule)
          : meta.label
        const humanRule = attr.rule?.trim()
          ? humanizeLimeRule(attr.rule)
          : meta.label
        return {
          feature: attr.feature,
          label: meta.label,
          description: meta.description,
          unit: meta.unit,
          humanRule,
          rawRule,
          weight: w,
          value: isFiniteNum(attr.value) ? attr.value : null,
        }
      })
      .filter((item): item is LimeRuleRow => isFiniteNum(item.weight))
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      .slice(0, 10)
  }, [activeLimeLocal])

  const maxLimeAbs = useMemo(
    () => (limeRows.length > 0 ? Math.max(...limeRows.map((r) => Math.abs(r.weight))) : 0),
    [limeRows]
  )

  // Deterministic plain-English takeaway sentences
  const localShapTakeaway = useMemo(() => {
    if (!activeShapLocal || shapLocalRows.length === 0) return null
    const base = activeShapLocal.base_value
    const pred = activeShapLocal.predicted_value
    const positives = shapLocalRows.filter((r) => r.contribution > 0)
    const negatives = shapLocalRows.filter((r) => r.contribution < 0)

    const netDelta =
      isFiniteNum(base) && isFiniteNum(pred)
        ? pred - base
        : localPositiveSum + localNegativeSum
    const pushingUp = netDelta >= 0

    const primaryList = pushingUp ? positives : negatives
    const counterList = pushingUp ? negatives : positives
    const top1 = primaryList[0] ?? shapLocalRows[0]
    const top2 = primaryList[1]
    const counter = counterList[0]

    const parts: string[] = []
    if (isFiniteNum(base)) {
      parts.push(`Starting from a typical ${formatValueWithUnit(base, activeUnit, 1)},`)
    } else {
      parts.push("This forecast was")
    }

    const directionWord = pushingUp ? "higher" : "lower"
    const top1Text = `${top1.label} (${formatSignedWithUnit(top1.contribution, activeUnit, 2)})`
    const top2Text = top2
      ? ` and ${top2.label} (${formatSignedWithUnit(top2.contribution, activeUnit, 2)})`
      : ""
    parts.push(`this forecast was pushed ${directionWord} mainly by ${top1Text}${top2Text}`)

    if (counter && Math.abs(counter.contribution) >= Math.abs(top1.contribution) * 0.15) {
      parts.push(
        `, partly offset by ${counter.label} (${formatSignedWithUnit(counter.contribution, activeUnit, 2)})`
      )
    }

    if (isFiniteNum(pred)) {
      parts.push(`, landing at ${formatValueWithUnit(pred, activeUnit, 1)}.`)
    } else {
      parts.push(".")
    }

    return parts.join(" ").replace(/\s+,/g, ",").replace(/\s+\./g, ".")
  }, [activeShapLocal, shapLocalRows, activeUnit, localPositiveSum, localNegativeSum])

  const globalShapTakeaway = useMemo(() => {
    if (shapGlobalRows.length === 0) return null
    const modelName = formatModelLabel(activeModel)
    const top1 = shapGlobalRows[0]
    const top2 = shapGlobalRows[1]
    if (top1 && top2) {
      const combined = Math.round(top1.sharePct + top2.sharePct)
      return `Across evaluated hours, ${top1.label} and ${top2.label} together account for about ${combined}% of what drives ${modelName}'s ${activeLabel.toLowerCase()} forecasts.`
    }
    return `${top1.label} is the primary signal driving ${modelName}'s ${activeLabel.toLowerCase()} forecasts (${Math.round(top1.sharePct)}% share).`
  }, [shapGlobalRows, activeModel, activeLabel])

  const limeTakeaway = useMemo(() => {
    if (!activeLimeLocal || limeRows.length === 0) return null
    const top = limeRows[0]
    const predText = isFiniteNum(activeLimeLocal.predicted_value)
      ? formatValueWithUnit(activeLimeLocal.predicted_value, activeUnit, 1)
      : `this ${activeLabel.toLowerCase()} forecast`
    const dir = top.weight >= 0 ? "upward" : "downward"
    return `To double-check the ${predText} forecast with simple threshold rules, the strongest ${dir} signal was: "${top.humanRule}".`
  }, [activeLimeLocal, limeRows, activeUnit, activeLabel])

  // Table columns for ChartCard data table toggle
  const shapLocalTableColumns = useMemo<ChartTableColumn<ShapLocalRow>[]>(
    () => [
      { key: "label", header: "Weather signal" },
      {
        key: "value",
        header: "Observed value",
        align: "right",
        render: (row) => formatMetricValue(row.value, 2),
      },
      {
        key: "contribution",
        header: `SHAP impact${activeUnit ? ` (${activeUnit})` : ""}`,
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
    [activeUnit]
  )

  const shapGlobalTableColumns = useMemo<ChartTableColumn<ShapGlobalRow>[]>(
    () => [
      { key: "label", header: "Weather signal" },
      {
        key: "sharePct",
        header: "Influence share",
        align: "right",
        render: (row) => `${row.sharePct.toFixed(1)}%`,
      },
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
    ],
    []
  )

  const limeTableColumns = useMemo<ChartTableColumn<LimeRuleRow>[]>(
    () => [
      { key: "humanRule", header: "Condition" },
      { key: "rawRule", header: "Exact rule" },
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

  if (explainedModels.length === 0) {
    return (
      <section className="rounded-xl border border-border bg-card p-5 shadow-(--shadow-card)">
        <EmptyState
          icon={Sparkles}
          title={`No forecast explanations for ${activeLabel} yet`}
          description="SHAP and LIME explanations appear here after the training pipeline computes feature drivers."
          className="border-0 bg-muted shadow-none"
        />
      </section>
    )
  }

  const activeTableColumns =
    activeView === "local_shap"
      ? (shapLocalTableColumns as ChartTableColumn<Record<string, unknown>>[])
      : activeView === "global_shap"
        ? (shapGlobalTableColumns as ChartTableColumn<Record<string, unknown>>[])
        : (limeTableColumns as ChartTableColumn<Record<string, unknown>>[])

  const activeTableRows =
    activeView === "local_shap"
      ? (shapLocalRows as Record<string, unknown>[])
      : activeView === "global_shap"
        ? (shapGlobalRows as Record<string, unknown>[])
        : (limeRows as Record<string, unknown>[])

  const activeRecordList =
    activeView === "local_shap"
      ? modelShapLocals
      : activeView === "lime"
        ? modelLimeLocals
        : []

  const activeLocalIndex =
    activeView === "local_shap" ? selectedShapLocalIndex : selectedLimeLocalIndex

  const visibleLocalShapRows = showAllDrivers
    ? shapLocalRows
    : shapLocalRows.slice(0, DEFAULT_VISIBLE_DRIVERS)
  const visibleGlobalShapRows = showAllDrivers
    ? shapGlobalRows
    : shapGlobalRows.slice(0, DEFAULT_VISIBLE_DRIVERS)
  const visibleLimeRows = showAllDrivers
    ? limeRows
    : limeRows.slice(0, DEFAULT_VISIBLE_DRIVERS)

  const activeTotalCount =
    activeView === "local_shap"
      ? shapLocalRows.length
      : activeView === "global_shap"
        ? shapGlobalRows.length
        : limeRows.length

  const localDelta =
    activeShapLocal &&
    isFiniteNum(activeShapLocal.predicted_value) &&
    isFiniteNum(activeShapLocal.base_value)
      ? activeShapLocal.predicted_value - activeShapLocal.base_value
      : localPositiveSum + localNegativeSum

  return (
    <ChartCard
      title={`Why did ${formatModelLabel(activeModel)} predict this?`}
      titleInfo="SHAP measures how much each weather signal pushed a forecast up or down from a typical baseline. LIME double-checks a single forecast using simple threshold rules."
      subtitle={`Plain-language breakdown of ${activeLabel.toLowerCase()} forecast drivers using SHAP and LIME`}
      ariaLabel={`SHAP and LIME explanation for ${formatModelLabel(activeModel)}`}
      controls={
        <div className="flex flex-wrap items-center gap-2">
          {explainedModels.length > 1 ? (
            <Select
              value={activeModel}
              onValueChange={(val) => {
                setSelectedModel(val)
                setSelectedShapLocalIndex(0)
                setSelectedLimeLocalIndex(0)
              }}
            >
              <SelectTrigger
                aria-label="Select explained model"
                className="h-8 min-w-[140px] bg-card text-xs"
              >
                <SelectValue placeholder="Model" />
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

          {activeRecordList.length > 1 ? (
            <Select
              value={String(Math.min(activeLocalIndex, activeRecordList.length - 1))}
              onValueChange={(val) => {
                const idx = Number(val)
                if (activeView === "local_shap") setSelectedShapLocalIndex(idx)
                if (activeView === "lime") setSelectedLimeLocalIndex(idx)
              }}
            >
              <SelectTrigger
                aria-label="Select forecast hour"
                className="h-8 min-w-[148px] bg-card text-xs"
              >
                <SelectValue placeholder="Forecast hour" />
              </SelectTrigger>
              <SelectContent position="popper">
                {activeRecordList.map((rec, idx) => (
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
          ) : null}

          <div
            role="tablist"
            aria-label="Explanation view"
            className="inline-flex h-8 items-center rounded-full bg-muted p-0.5"
          >
            {(
              [
                { id: "local_shap", label: "This forecast" },
                { id: "global_shap", label: "Overall drivers" },
                { id: "lime", label: "Rule check (LIME)" },
              ] as const
            ).map((tab) => {
              const active = activeView === tab.id
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => {
                    setSelectedView(tab.id)
                    setShowAllDrivers(false)
                  }}
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
      }
      tableColumns={activeTableColumns}
      tableRows={activeTableRows}
      emptyTableMessage="No explanation rows available for this view."
      footer={
        activeTotalCount > DEFAULT_VISIBLE_DRIVERS ? (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              Showing top {showAllDrivers ? activeTotalCount : DEFAULT_VISIBLE_DRIVERS} of{" "}
              {activeTotalCount} signals · Hover any bar for exact numbers
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 gap-1 px-2 text-xs font-medium text-foreground"
              onClick={() => setShowAllDrivers((prev) => !prev)}
            >
              {showAllDrivers ? (
                <>
                  <ChevronUp className="size-3.5" strokeWidth={1.75} />
                  <span>Show top {DEFAULT_VISIBLE_DRIVERS}</span>
                </>
              ) : (
                <>
                  <ChevronDown className="size-3.5" strokeWidth={1.75} />
                  <span>Show all {activeTotalCount} signals</span>
                </>
              )}
            </Button>
          </div>
        ) : (
          <span>Hover or focus any row for exact underlying values.</span>
        )
      }
    >
      {/* VIEW 1: LOCAL SHAP ("This forecast") */}
      {activeView === "local_shap" ? (
        shapLocalRows.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title="No single-hour SHAP breakdown yet"
            description={`Local feature drivers were not recorded for ${formatModelLabel(activeModel)}. Try switching to Overall drivers.`}
            className="border-0 bg-muted shadow-none"
          />
        ) : (
          <div className="flex flex-col gap-4">
            {/* Plain-English takeaway */}
            {localShapTakeaway ? (
              <div className="rounded-lg border border-primary/20 bg-accent/60 px-4 py-3 text-sm/6 text-foreground">
                {localShapTakeaway}
              </div>
            ) : null}

            {/* Compact visual equation strip */}
            {activeShapLocal ? (
              <dl className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                <div className="rounded-lg bg-muted px-3.5 py-2.5">
                  <dt className="text-xs text-muted-foreground">Typical average</dt>
                  <dd className="mt-0.5 text-base font-semibold text-foreground tabular-nums">
                    {formatValueWithUnit(activeShapLocal.base_value, activeUnit, 1)}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3.5 py-2.5">
                  <dt className="text-xs text-muted-foreground">Net push</dt>
                  <dd
                    className={cn(
                      "mt-0.5 flex items-center gap-1 text-base font-semibold tabular-nums",
                      localDelta >= 0 ? "text-shap-pos" : "text-shap-neg"
                    )}
                  >
                    {localDelta >= 0 ? (
                      <ArrowUpRight className="size-4 shrink-0" strokeWidth={2} />
                    ) : (
                      <ArrowDownRight className="size-4 shrink-0" strokeWidth={2} />
                    )}
                    <span>{formatSignedWithUnit(localDelta, activeUnit, 2)}</span>
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3.5 py-2.5">
                  <dt className="text-xs text-muted-foreground">Final forecast</dt>
                  <dd className="mt-0.5 text-base font-semibold text-foreground tabular-nums">
                    {formatValueWithUnit(activeShapLocal.predicted_value, activeUnit, 1)}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3.5 py-2.5">
                  <dt className="text-xs text-muted-foreground">Forecast hour</dt>
                  <dd className="mt-1 text-xs font-medium text-foreground">
                    <TimestampPkt
                      timestamp={activeShapLocal.reference_timestamp_utc}
                      timeZone={resolvedTimeZone}
                      mode="datetime"
                    />
                  </dd>
                </div>
              </dl>
            ) : null}

            {/* Diverging column header */}
            <div className="grid grid-cols-[minmax(140px,240px)_minmax(0,1fr)_88px] items-center gap-3 px-1 text-[11px] font-medium text-muted-foreground">
              <span>Weather signal</span>
              <div className="flex items-center justify-between px-1">
                <span className="text-shap-neg">← Pushes lower</span>
                <span>0</span>
                <span className="text-shap-pos">Pushes higher →</span>
              </div>
              <span className="text-right">Impact</span>
            </div>

            {/* Diverging bars */}
            <div className="flex flex-col gap-2">
              {visibleLocalShapRows.map((row) => {
                const pct =
                  maxShapLocalAbs > 0
                    ? Math.max(3, (Math.abs(row.contribution) / maxShapLocalAbs) * 50)
                    : 0
                const isPositive = row.contribution >= 0

                return (
                  <Tooltip key={row.feature}>
                    <TooltipTrigger asChild>
                      <div
                        tabIndex={0}
                        className="grid grid-cols-[minmax(140px,240px)_minmax(0,1fr)_88px] items-center gap-3 rounded-md px-1 py-1 transition-colors hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                      >
                        <div className="min-w-0">
                          <div
                            className="truncate text-sm font-medium text-foreground"
                            title={row.label}
                          >
                            {row.label}
                          </div>
                          {row.value !== null ? (
                            <div className="text-xs text-muted-foreground tabular-nums">
                              Currently {formatValueWithUnit(row.value, row.unit, 1)}
                            </div>
                          ) : null}
                        </div>

                        <div className="relative h-7 w-full overflow-hidden rounded-sm bg-muted">
                          <div
                            aria-hidden="true"
                            className="absolute inset-y-0 left-1/2 w-px bg-foreground/25"
                          />
                          <div
                            className="absolute top-1 bottom-1 rounded-xs transition-all duration-150"
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
                            "text-right text-xs font-semibold tabular-nums",
                            isPositive ? "text-shap-pos" : "text-shap-neg"
                          )}
                        >
                          {formatSignedWithUnit(row.contribution, activeUnit, 2)}
                        </span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent className="flex max-w-xs flex-col gap-1 text-left">
                      <span className="font-semibold text-popover-foreground">
                        {row.label}
                      </span>
                      {row.description ? (
                        <span className="text-xs leading-snug text-popover-foreground/85">
                          {row.description}
                        </span>
                      ) : null}
                      <span className="text-muted-foreground tabular-nums">
                        Impact on forecast: {formatSignedValue(row.contribution, 3)}
                        {activeUnit ? ` ${activeUnit}` : ""}
                      </span>
                      {row.value !== null ? (
                        <span className="text-muted-foreground tabular-nums">
                          Observed reading: {formatValueWithUnit(row.value, row.unit, 2)}
                        </span>
                      ) : null}
                      <span className="border-t border-border/60 pt-1 font-mono text-[10px] text-muted-foreground">
                        {row.feature}
                      </span>
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        )
      ) : activeView === "global_shap" ? (
        /* VIEW 2: GLOBAL SHAP ("Overall drivers") */
        shapGlobalRows.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title="No overall SHAP summary yet"
            description={`Global feature importance was not recorded for ${formatModelLabel(activeModel)}.`}
            className="border-0 bg-muted shadow-none"
          />
        ) : (
          <div className="flex flex-col gap-4">
            {globalShapTakeaway ? (
              <div className="rounded-lg border border-primary/20 bg-accent/60 px-4 py-3 text-sm/6 text-foreground">
                {globalShapTakeaway}
              </div>
            ) : null}

            <div className="grid grid-cols-[minmax(140px,240px)_minmax(0,1fr)_104px] items-center gap-3 px-1 text-[11px] font-medium text-muted-foreground">
              <span>Weather signal</span>
              <span>Relative influence across all evaluated hours</span>
              <span className="text-right">Share</span>
            </div>

            <div className="flex flex-col gap-2">
              {visibleGlobalShapRows.map((row, idx) => {
                const widthPct =
                  maxShapGlobalMag > 0
                    ? Math.max(4, (row.magnitude / maxShapGlobalMag) * 100)
                    : 0

                return (
                  <Tooltip key={row.feature}>
                    <TooltipTrigger asChild>
                      <div
                        tabIndex={0}
                        className="grid grid-cols-[minmax(140px,240px)_minmax(0,1fr)_104px] items-center gap-3 rounded-md px-1 py-1 transition-colors hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                      >
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="w-4 shrink-0 text-xs font-medium text-muted-foreground tabular-nums">
                            {idx + 1}.
                          </span>
                          <span
                            className="truncate text-sm font-medium text-foreground"
                            title={row.label}
                          >
                            {row.label}
                          </span>
                        </div>

                        <div className="h-7 w-full overflow-hidden rounded-sm bg-muted">
                          <div
                            className="h-full rounded-sm bg-primary/85 transition-all duration-150"
                            style={{ width: `${widthPct}%` }}
                          />
                        </div>

                        <div className="flex items-baseline justify-end gap-1.5 text-right tabular-nums">
                          <span className="text-sm font-semibold text-foreground">
                            {row.sharePct.toFixed(0)}%
                          </span>
                          <span className="text-[11px] text-muted-foreground">
                            ({formatMetricValue(row.magnitude, 2)})
                          </span>
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent className="flex max-w-xs flex-col gap-1 text-left">
                      <span className="font-semibold text-popover-foreground">
                        {row.label}
                      </span>
                      {row.description ? (
                        <span className="text-xs leading-snug text-popover-foreground/85">
                          {row.description}
                        </span>
                      ) : null}
                      <span className="text-muted-foreground tabular-nums">
                        Influence share: {row.sharePct.toFixed(1)}% (Mean |SHAP|{" "}
                        {formatMetricValue(row.magnitude, 3)})
                      </span>
                      {row.referenceValue !== null ? (
                        <span className="text-muted-foreground tabular-nums">
                          Average reading:{" "}
                          {formatValueWithUnit(row.referenceValue, row.unit, 2)}
                        </span>
                      ) : null}
                      <span className="border-t border-border/60 pt-1 font-mono text-[10px] text-muted-foreground">
                        {row.feature}
                      </span>
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        )
      ) : (
        /* VIEW 3: LIME ("Rule check") */
        limeRows.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title="No LIME rule check recorded"
            description={`Local LIME threshold rules were not recorded for ${formatModelLabel(activeModel)}.`}
            className="border-0 bg-muted shadow-none"
          />
        ) : (
          <div className="flex flex-col gap-4">
            {limeTakeaway ? (
              <div className="rounded-lg border border-primary/20 bg-accent/60 px-4 py-3 text-sm/6 text-foreground">
                {limeTakeaway}
              </div>
            ) : null}

            {activeLimeLocal ? (
              <dl className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
                <div className="rounded-lg bg-muted px-3.5 py-2.5">
                  <dt className="text-xs text-muted-foreground">Typical average</dt>
                  <dd className="mt-0.5 text-base font-semibold text-foreground tabular-nums">
                    {formatValueWithUnit(activeLimeLocal.base_value, activeUnit, 1)}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted px-3.5 py-2.5">
                  <dt className="text-xs text-muted-foreground">Model forecast</dt>
                  <dd className="mt-0.5 text-base font-semibold text-foreground tabular-nums">
                    {formatValueWithUnit(activeLimeLocal.predicted_value, activeUnit, 1)}
                  </dd>
                </div>
                <div className="col-span-2 rounded-lg bg-muted px-3.5 py-2.5 sm:col-span-1">
                  <dt className="text-xs text-muted-foreground">Forecast hour</dt>
                  <dd className="mt-1 text-xs font-medium text-foreground">
                    <TimestampPkt
                      timestamp={activeLimeLocal.reference_timestamp_utc}
                      timeZone={resolvedTimeZone}
                      mode="datetime"
                    />
                  </dd>
                </div>
              </dl>
            ) : null}

            <div className="grid grid-cols-[minmax(170px,300px)_minmax(0,1fr)_88px] items-center gap-3 px-1 text-[11px] font-medium text-muted-foreground">
              <span>Condition observed</span>
              <div className="flex items-center justify-between px-1">
                <span className="text-shap-neg">← Pushes lower</span>
                <span>0</span>
                <span className="text-shap-pos">Pushes higher →</span>
              </div>
              <span className="text-right">Effect</span>
            </div>

            <div className="flex flex-col gap-2">
              {visibleLimeRows.map((row, index) => {
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
                        className="grid grid-cols-[minmax(170px,300px)_minmax(0,1fr)_88px] items-center gap-3 rounded-md px-1 py-1 transition-colors hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                      >
                        <div className="min-w-0">
                          <div
                            className="truncate text-sm font-medium text-foreground"
                            title={row.humanRule}
                          >
                            {row.humanRule}
                          </div>
                          {row.value !== null ? (
                            <div className="text-xs text-muted-foreground tabular-nums">
                              Observed: {formatValueWithUnit(row.value, row.unit, 1)}
                            </div>
                          ) : null}
                        </div>

                        <div className="relative h-7 w-full overflow-hidden rounded-sm bg-muted">
                          <div
                            aria-hidden="true"
                            className="absolute inset-y-0 left-1/2 w-px bg-foreground/25"
                          />
                          <div
                            className="absolute top-1 bottom-1 rounded-xs transition-all duration-150"
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
                            "text-right text-xs font-semibold tabular-nums",
                            isPositive ? "text-shap-pos" : "text-shap-neg"
                          )}
                        >
                          {formatSignedValue(row.weight, 2)}
                        </span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent className="flex max-w-xs flex-col gap-1 text-left">
                      <span className="font-semibold text-popover-foreground">
                        {row.humanRule}
                      </span>
                      {row.description ? (
                        <span className="text-xs leading-snug text-popover-foreground/85">
                          {row.description}
                        </span>
                      ) : null}
                      <span className="text-muted-foreground tabular-nums">
                        LIME weight: {formatSignedValue(row.weight, 4)}
                      </span>
                      {row.value !== null ? (
                        <span className="text-muted-foreground tabular-nums">
                          Observed reading: {formatValueWithUnit(row.value, row.unit, 2)}
                        </span>
                      ) : null}
                      <span className="border-t border-border/60 pt-1 font-mono text-[10px] text-muted-foreground">
                        {row.rawRule}
                      </span>
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        )
      )}
    </ChartCard>
  )
}
