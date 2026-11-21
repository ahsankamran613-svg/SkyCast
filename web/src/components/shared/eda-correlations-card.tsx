import { useMemo, useState } from "react"
import { ChevronDown, ChevronUp, Grid2x2 } from "lucide-react"
import { cn } from "cn"

import { ChartCard, type ChartTableColumn } from "@/components/shared/chart-card"
import { EmptyState } from "@/components/shared/empty-state"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type { EdaDistributionEntry, MetricSpecsMap } from "@/lib/api"
import { humanizeIdentifier, resolveMetricLabel } from "@/lib/weather"

export interface CorrelationPairRow extends Record<string, unknown> {
  pair: string
  rowKey: string
  colKey: string
  rowLabel: string
  colLabel: string
  correlation: number | null
}

export interface EdaCorrelationsCardProps {
  method: string
  minPeriods: number
  columns: string[]
  matrix: Record<string, Record<string, number | null>>
  constantColumns: string[]
  metricSpecsHint?: MetricSpecsMap | null
  distributions: Record<string, EdaDistributionEntry>
}

type CorrViewMode = "top" | "matrix"

const DEFAULT_VISIBLE_PAIRS = 8

function isFiniteNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function describeCorrelation(r: number): {
  strength: string
  direction: string
  isPositive: boolean
} {
  const abs = Math.abs(r)
  const isPositive = r >= 0
  const strength =
    abs >= 0.7
      ? "Strong"
      : abs >= 0.4
        ? "Moderate"
        : abs >= 0.2
          ? "Mild"
          : "Weak"
  const direction = isPositive ? "move together" : "move opposite"
  return { strength, direction, isPositive }
}

function correlationCellStyle(r: number | null): {
  backgroundColor?: string
  label: string
} {
  if (!isFiniteNum(r)) {
    return { label: "—" }
  }
  const clamped = Math.max(-1, Math.min(1, r))
  const mixPct = Math.round(Math.abs(clamped) * 38)
  const targetVar = clamped >= 0 ? "var(--shap-pos)" : "var(--shap-neg)"
  return {
    backgroundColor: `color-mix(in srgb, ${targetVar} ${mixPct}%, var(--corr-neutral))`,
    label: clamped.toFixed(2),
  }
}

export function EdaCorrelationsCard({
  method,
  minPeriods,
  columns,
  matrix,
  constantColumns,
  metricSpecsHint,
  distributions,
}: EdaCorrelationsCardProps) {
  const [viewMode, setViewMode] = useState<CorrViewMode>("top")
  const [showAllPairs, setShowAllPairs] = useState<boolean>(false)

  const pairRows = useMemo<CorrelationPairRow[]>(() => {
    const rows: CorrelationPairRow[] = []
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        const rKey = columns[i]
        const cKey = columns[j]
        const val = matrix?.[rKey]?.[cKey] ?? null
        const rowLabel = resolveMetricLabel(rKey, metricSpecsHint, distributions)
        const colLabel = resolveMetricLabel(cKey, metricSpecsHint, distributions)
        rows.push({
          pair: `${rowLabel} & ${colLabel}`,
          rowKey: rKey,
          colKey: cKey,
          rowLabel,
          colLabel,
          correlation: isFiniteNum(val) ? val : null,
        })
      }
    }
    return rows.sort(
      (a, b) => Math.abs(b.correlation ?? 0) - Math.abs(a.correlation ?? 0)
    )
  }, [columns, matrix, metricSpecsHint, distributions])

  const validPairs = useMemo(
    () => pairRows.filter((row) => isFiniteNum(row.correlation)),
    [pairRows]
  )

  const visiblePairs = showAllPairs
    ? validPairs
    : validPairs.slice(0, DEFAULT_VISIBLE_PAIRS)

  const tableColumns = useMemo<ChartTableColumn<CorrelationPairRow>[]>(
    () => [
      { key: "rowLabel", header: "Variable A" },
      { key: "colLabel", header: "Variable B" },
      {
        key: "correlation",
        header: `${humanizeIdentifier(method)} r`,
        align: "right",
        render: (row) =>
          isFiniteNum(row.correlation)
            ? `${row.correlation > 0 ? "+" : ""}${row.correlation.toFixed(3)}`
            : "—",
      },
    ],
    [method]
  )

  const topSummary = useMemo(() => {
    const top = validPairs[0]
    if (!top || !isFiniteNum(top.correlation)) return null
    const { strength, direction } = describeCorrelation(top.correlation)
    return `Strongest link: ${top.rowLabel} and ${top.colLabel} ${direction} (${strength.toLowerCase()}, r = ${top.correlation > 0 ? "+" : ""}${top.correlation.toFixed(2)}).`
  }, [validPairs])

  return (
    <ChartCard
      title="How weather variables move together"
      titleInfo="Correlation (r) ranges from −1.0 to +1.0. Positive orange values mean two variables rise and fall together; negative blue values mean one tends to drop when the other rises."
      subtitle={
        columns.length > 0
          ? `${humanizeIdentifier(method)} correlation across ${columns.length} variables (min ${minPeriods} paired hours)`
          : "Pairwise statistical relationships across weather and air-quality metrics"
      }
      ariaLabel="Variable relationships and correlation matrix"
      controls={
        columns.length > 0 ? (
          <div
            role="tablist"
            aria-label="Correlation display mode"
            className="inline-flex h-8 items-center rounded-full bg-muted p-0.5"
          >
            {(
              [
                { id: "top", label: "Top links" },
                { id: "matrix", label: "Full matrix" },
              ] as const
            ).map((tab) => {
              const active = viewMode === tab.id
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setViewMode(tab.id)}
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
        ) : undefined
      }
      tableColumns={tableColumns}
      tableRows={pairRows}
      emptyTableMessage="No correlation pairs available."
      footer={
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span>
            {viewMode === "matrix"
              ? "Lower triangle shown to remove duplicate mirror cells · Hover any cell for exact r"
              : `Showing ${visiblePairs.length} of ${validPairs.length} variable pairs`}
          </span>
          {viewMode === "top" && validPairs.length > DEFAULT_VISIBLE_PAIRS ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 gap-1 px-2 text-xs font-medium text-foreground"
              onClick={() => setShowAllPairs((prev) => !prev)}
            >
              {showAllPairs ? (
                <>
                  <ChevronUp className="size-3.5" strokeWidth={1.75} />
                  <span>Show top {DEFAULT_VISIBLE_PAIRS}</span>
                </>
              ) : (
                <>
                  <ChevronDown className="size-3.5" strokeWidth={1.75} />
                  <span>Show all {validPairs.length} pairs</span>
                </>
              )}
            </Button>
          ) : null}
        </div>
      }
    >
      {columns.length === 0 ? (
        <EmptyState
          icon={Grid2x2}
          title="Relationships unavailable"
          description="Pairwise correlation coefficients appear once sufficient valid observations are available."
          className="border-0 bg-muted shadow-none"
        />
      ) : viewMode === "top" ? (
        <div className="flex flex-col gap-4">
          {topSummary ? (
            <div className="rounded-lg border border-primary/20 bg-accent/60 px-4 py-3 text-sm/6 text-foreground">
              {topSummary}
            </div>
          ) : null}

          <div className="grid grid-cols-[minmax(170px,290px)_minmax(0,1fr)_72px] items-center gap-3 px-1 text-[11px] font-medium text-muted-foreground">
            <span>Variable pair</span>
            <div className="flex items-center justify-between px-1">
              <span className="text-shap-neg">← Move opposite (−1)</span>
              <span>0</span>
              <span className="text-shap-pos">Move together (+1) →</span>
            </div>
            <span className="text-right">Score (r)</span>
          </div>

          <div className="flex flex-col gap-2">
            {visiblePairs.map((row) => {
              const r = row.correlation ?? 0
              const { strength, direction, isPositive } = describeCorrelation(r)
              const pct = Math.max(3, Math.min(50, Math.abs(r) * 50))

              return (
                <Tooltip key={`${row.rowKey}-${row.colKey}`}>
                  <TooltipTrigger asChild>
                    <div
                      tabIndex={0}
                      className="grid grid-cols-[minmax(170px,290px)_minmax(0,1fr)_72px] items-center gap-3 rounded-md px-1 py-1.5 transition-colors hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-1.5 text-sm font-medium text-foreground">
                          <span className="truncate">{row.rowLabel}</span>
                          <span className="text-muted-foreground">&amp;</span>
                          <span className="truncate">{row.colLabel}</span>
                        </div>
                        <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                          <Badge
                            variant="secondary"
                            className="h-4.5 px-1.5 text-[10px] font-medium"
                          >
                            {strength}
                          </Badge>
                          <span>{direction}</span>
                        </div>
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
                        {r > 0 ? "+" : ""}
                        {r.toFixed(2)}
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent className="flex flex-col gap-0.5 text-left">
                    <span className="font-semibold text-popover-foreground">
                      {row.rowLabel} × {row.colLabel}
                    </span>
                    <span className="text-muted-foreground tabular-nums">
                      {humanizeIdentifier(method)} r = {r > 0 ? "+" : ""}
                      {r.toFixed(4)}
                    </span>
                    <span className="text-muted-foreground">
                      {strength} relationship — variables {direction}
                    </span>
                  </TooltipContent>
                </Tooltip>
              )
            })}
          </div>

          {constantColumns.length > 0 ? (
            <p className="text-xs text-muted-foreground">
              Excluded constant variables (no variation in window):{" "}
              {constantColumns
                .map((col) =>
                  resolveMetricLabel(col, metricSpecsHint, distributions)
                )
                .join(", ")}
            </p>
          ) : null}
        </div>
      ) : (
        /* Lower-triangle correlation matrix */
        <div className="flex flex-col gap-3">
          <div className="overflow-x-auto pb-2">
            <div className="inline-block align-middle">
              {/* Matrix rows (lower triangle only: j < i) */}
              <div className="flex flex-col gap-1">
                {columns.map((rowKey, rowIdx) => {
                  const rowLabel = resolveMetricLabel(
                    rowKey,
                    metricSpecsHint,
                    distributions
                  )
                  return (
                    <div key={rowKey} className="flex items-center gap-1">
                      <div
                        className="sticky left-0 z-10 w-[132px] shrink-0 truncate bg-card pr-2 text-xs font-medium text-foreground sm:w-[152px]"
                        title={rowLabel}
                      >
                        {rowLabel}
                      </div>

                      {columns.map((colKey, colIdx) => {
                        if (colIdx > rowIdx) {
                          return (
                            <div
                              key={colKey}
                              aria-hidden="true"
                              className="size-10 shrink-0 sm:size-12"
                            />
                          )
                        }

                        if (colIdx === rowIdx) {
                          return (
                            <div
                              key={colKey}
                              aria-hidden="true"
                              className="flex size-10 shrink-0 items-center justify-center rounded-sm bg-muted/40 text-[10px] font-medium text-muted-foreground/70 sm:size-12"
                            >
                              1.0
                            </div>
                          )
                        }

                        const rVal = matrix?.[rowKey]?.[colKey] ?? null
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
                                  "flex size-10 shrink-0 items-center justify-center rounded-sm border border-border/50 text-[11px] font-medium tabular-nums focus-visible:outline-2 focus-visible:outline-ring sm:size-12 sm:text-xs",
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

              {/* Bottom column labels aligned cleanly under the lower triangle */}
              <div className="mt-2 flex items-start gap-1">
                <div className="sticky left-0 z-10 w-[132px] shrink-0 bg-card pr-2 sm:w-[152px]" />
                {columns.map((colKey) => {
                  const label = resolveMetricLabel(
                    colKey,
                    metricSpecsHint,
                    distributions
                  )
                  return (
                    <div
                      key={colKey}
                      title={label}
                      className="w-10 shrink-0 truncate text-center text-[11px] font-medium text-muted-foreground sm:w-12"
                    >
                      {label}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {constantColumns.length > 0 ? (
            <p className="text-xs text-muted-foreground">
              Constant variables excluded:{" "}
              {constantColumns
                .map((col) =>
                  resolveMetricLabel(col, metricSpecsHint, distributions)
                )
                .join(", ")}
            </p>
          ) : null}
        </div>
      )}
    </ChartCard>
  )
}
