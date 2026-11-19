import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  Clock,
  GitBranch,
  MinusCircle,
  OctagonAlert,
  RotateCcw,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react"

import {
  BootstrappingState,
  EmptyState,
  ErrorState,
} from "@/components/shared/empty-state"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  fetchAgentsLogs,
  type AgentCallLogRow,
  type AgentCallStatus,
  type AgentGraphTraceRow,
  type AgentsLogsResponse,
  type DriftLogRow,
  type DriftStatus,
  type DriftType,
  type GraphStepStatus,
  type MetricSpecsMap,
} from "@/lib/api"
import { getTimeZoneLabel, parseIsoDate } from "@/lib/time"
import {
  formatDriftAction,
  formatDriftTestName,
  formatMetricValue,
  humanizeIdentifier,
  resolveMetricLabel,
} from "@/lib/weather"

export interface AgentsDriftTabProps {
  city?: string | null
  timeZone?: string | null
  metricSpecsHint?: MetricSpecsMap | null
  refreshNonce?: number
  onShellRefresh?: () => void
}

type AgentsSubTab = "calls" | "failures" | "drift" | "trace"

const DEFAULT_LIMIT = 200
const EXPANDED_LIMIT = 500

const AGENT_ROLE_LABELS: Record<string, string> = {
  forecast_agent: "Forecast agent",
  forecast_node: "Forecast agent",
  alert_agent: "Alert agent",
  alert_node: "Alert agent",
  drift_agent: "Drift monitor",
  drift_node: "Drift monitor",
  weather_report_agent: "Weather report agent",
  weather_report_node: "Weather report agent",
}

const BRANCH_DECISION_LABELS: Record<string, string> = {
  proceed_alert: "Proceed to hazard check",
  hazards_active: "Hazard alerts generated",
  no_hazards: "No active hazards",
  run_weather_report: "Generate weather briefing",
  skip_weather_report: "Skip briefing (no new trigger)",
  complete: "Workflow complete",
}

function isFiniteNum(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatLatency(ms: number | null | undefined): string {
  if (!isFiniteNum(ms)) return "—"
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} s`
  }
  return `${Math.round(ms)} ms`
}

function formatHourSchedule(
  hour: number | null | undefined,
  tzLabel: string
): string {
  if (!isFiniteNum(hour)) return "—"
  const clamped = ((Math.round(hour) % 24) + 24) % 24
  return `${String(clamped).padStart(2, "0")}:00 ${tzLabel}`
}

function formatAgentRole(raw: string | null | undefined): string {
  if (!raw) return "—"
  const key = raw.trim().toLowerCase()
  return AGENT_ROLE_LABELS[key] ?? humanizeIdentifier(raw)
}

function formatBranchDecision(raw: string | null | undefined): string {
  if (!raw) return "—"
  const key = raw.trim().toLowerCase()
  return BRANCH_DECISION_LABELS[key] ?? humanizeIdentifier(raw)
}

function renderCallStatusBadge(status: AgentCallStatus) {
  if (status === "failed") {
    return (
      <span className="inline-flex h-6 items-center gap-1 rounded-full border border-alert-critical bg-alert-critical-bg px-2.5 text-xs font-medium text-alert-critical">
        <OctagonAlert className="size-3.5 shrink-0" strokeWidth={1.75} />
        <span>Failed</span>
      </span>
    )
  }
  return (
    <span className="inline-flex h-6 items-center gap-1 rounded-full border border-aqi-good bg-aqi-good-bg px-2.5 text-xs font-medium text-aqi-good-fg">
      <CircleCheck className="size-3.5 shrink-0" strokeWidth={1.75} />
      <span>Success</span>
    </span>
  )
}

function renderDriftStatusBadge(status: DriftStatus) {
  if (status === "drift_detected") {
    return (
      <span className="inline-flex h-6 items-center gap-1 rounded-full border border-alert-warning bg-alert-warning-bg px-2.5 text-xs font-medium text-alert-warning">
        <TriangleAlert className="size-3.5 shrink-0" strokeWidth={1.75} />
        <span>Drift detected</span>
      </span>
    )
  }
  if (status === "check_failed") {
    return (
      <span className="inline-flex h-6 items-center gap-1 rounded-full border border-alert-critical bg-alert-critical-bg px-2.5 text-xs font-medium text-alert-critical">
        <OctagonAlert className="size-3.5 shrink-0" strokeWidth={1.75} />
        <span>Check incomplete</span>
      </span>
    )
  }
  return (
    <span className="inline-flex h-6 items-center gap-1 rounded-full border border-aqi-good bg-aqi-good-bg px-2.5 text-xs font-medium text-aqi-good-fg">
      <CircleCheck className="size-3.5 shrink-0" strokeWidth={1.75} />
      <span>No drift</span>
    </span>
  )
}

function renderGraphStatusBadge(status: GraphStepStatus) {
  if (status === "failed") {
    return (
      <span className="inline-flex h-6 items-center gap-1 rounded-full border border-alert-critical bg-alert-critical-bg px-2.5 text-xs font-medium text-alert-critical">
        <OctagonAlert className="size-3.5 shrink-0" strokeWidth={1.75} />
        <span>Failed</span>
      </span>
    )
  }
  if (status === "skipped") {
    return (
      <span className="inline-flex h-6 items-center gap-1 rounded-full border border-border bg-muted px-2.5 text-xs font-medium text-foreground">
        <MinusCircle
          className="size-3.5 shrink-0 text-muted-foreground"
          strokeWidth={1.75}
        />
        <span>Skipped</span>
      </span>
    )
  }
  return (
    <span className="inline-flex h-6 items-center gap-1 rounded-full border border-aqi-good bg-aqi-good-bg px-2.5 text-xs font-medium text-aqi-good-fg">
      <CircleCheck className="size-3.5 shrink-0" strokeWidth={1.75} />
      <span>Completed</span>
    </span>
  )
}

export function AgentsDriftTab({
  city,
  timeZone,
  metricSpecsHint,
  refreshNonce = 0,
  onShellRefresh,
}: AgentsDriftTabProps) {
  const [subTab, setSubTab] = useState<AgentsSubTab>("calls")

  // Scoped per-subtab filters so switching tabs never leaks hidden filter parameters
  const [agentFilter, setAgentFilter] = useState<string>("all")
  const [callStatusFilter, setCallStatusFilter] = useState<string>("all")
  const [driftStatusFilter, setDriftStatusFilter] = useState<string>("all")
  const [driftTypeFilter, setDriftTypeFilter] = useState<string>("all")
  const [metricFilter, setMetricFilter] = useState<string>("all")
  const [dateFilter, setDateFilter] = useState<string>("")
  const [traceStatusFilter, setTraceStatusFilter] = useState<string>("all")
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT)
  const [localRetryNonce, setLocalRetryNonce] = useState<number>(0)

  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set())

  const [seenAgents, setSeenAgents] = useState<string[]>([])
  const [seenMetrics, setSeenMetrics] = useState<string[]>([])

  const [data, setData] = useState<AgentsLogsResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isFetching, setIsFetching] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const satisfiedKeyRef = useRef<string | null>(null)

  // Only include server query parameters that apply to the currently active sub-tab
  const activeQueryParams = useMemo(() => {
    const isCalls = subTab === "calls"
    const isCallsOrFailures = subTab === "calls" || subTab === "failures"
    const isDrift = subTab === "drift"

    return {
      agent_name:
        isCallsOrFailures && agentFilter !== "all" ? agentFilter : undefined,
      status: isCalls && callStatusFilter !== "all" ? callStatusFilter : undefined,
      drift_status:
        isDrift && driftStatusFilter !== "all"
          ? (driftStatusFilter as DriftStatus)
          : undefined,
      drift_type:
        isDrift && driftTypeFilter !== "all"
          ? (driftTypeFilter as DriftType)
          : undefined,
      metric: isDrift && metricFilter !== "all" ? metricFilter : undefined,
      date: dateFilter || undefined,
      limit,
    }
  }, [
    subTab,
    agentFilter,
    callStatusFilter,
    driftStatusFilter,
    driftTypeFilter,
    metricFilter,
    dateFilter,
    limit,
  ])

  const queryKey = useMemo(
    () =>
      JSON.stringify({
        params: activeQueryParams,
        refreshNonce,
        localRetryNonce,
      }),
    [activeQueryParams, refreshNonce, localRetryNonce]
  )

  useEffect(() => {
    if (satisfiedKeyRef.current === queryKey) {
      return
    }

    const controller = new AbortController()
    setIsFetching(true)
    setErrorMessage(null)

    void fetchAgentsLogs(activeQueryParams, { signal: controller.signal })
      .then((response) => {
        if (controller.signal.aborted) return

        const agents = new Set<string>()
        const metrics = new Set<string>()

        for (const row of response.call_logs ?? []) {
          if (row.agent_name) agents.add(row.agent_name)
        }
        for (const row of response.failed_calls ?? []) {
          if (row.agent_name) agents.add(row.agent_name)
        }
        for (const row of response.drift_logs ?? []) {
          if (row.metric) metrics.add(row.metric)
        }

        if (agents.size > 0) {
          setSeenAgents((prev) => Array.from(new Set([...prev, ...agents])).sort())
        }
        if (metrics.size > 0) {
          setSeenMetrics((prev) =>
            Array.from(new Set([...prev, ...metrics])).sort()
          )
        }

        satisfiedKeyRef.current = queryKey
        setData(response)
        setIsLoading(false)
        setIsFetching(false)
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return
        if (err instanceof DOMException && err.name === "AbortError") return
        setErrorMessage(
          "Couldn't load automation and drift activity. Check your connection and refresh."
        )
        setIsLoading(false)
        setIsFetching(false)
      })

    return () => controller.abort()
  }, [queryKey, activeQueryParams])

  const handleRetry = useCallback(() => {
    satisfiedKeyRef.current = null
    if (onShellRefresh) {
      onShellRefresh()
    } else {
      setLocalRetryNonce((n) => n + 1)
    }
  }, [onShellRefresh])

  const handleResetFilters = useCallback(() => {
    setAgentFilter("all")
    setCallStatusFilter("all")
    setDriftStatusFilter("all")
    setDriftTypeFilter("all")
    setMetricFilter("all")
    setDateFilter("")
    setTraceStatusFilter("all")
  }, [])

  const toggleExpanded = useCallback((key: string) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }, [])

  const resolvedTimeZone = data?.timezone ?? timeZone ?? null
  const tzShortLabel = useMemo(
    () => getTimeZoneLabel(new Date(), resolvedTimeZone),
    [resolvedTimeZone]
  )

  const sortedCallLogs = useMemo<AgentCallLogRow[]>(() => {
    return [...(data?.call_logs ?? [])].sort(
      (a, b) =>
        (parseIsoDate(b.timestamp_utc)?.getTime() ?? 0) -
        (parseIsoDate(a.timestamp_utc)?.getTime() ?? 0)
    )
  }, [data?.call_logs])

  const failedAndFallbackRows = useMemo<AgentCallLogRow[]>(() => {
    const byId = new Map<number, AgentCallLogRow>()
    for (const row of data?.failed_calls ?? []) {
      byId.set(row.id, row)
    }
    for (const row of data?.call_logs ?? []) {
      if (row.status === "failed" || row.fallback_used || row.attempt_order > 1) {
        byId.set(row.id, row)
      }
    }
    return Array.from(byId.values()).sort(
      (a, b) =>
        (parseIsoDate(b.timestamp_utc)?.getTime() ?? 0) -
        (parseIsoDate(a.timestamp_utc)?.getTime() ?? 0)
    )
  }, [data?.failed_calls, data?.call_logs])

  const sortedDriftLogs = useMemo<DriftLogRow[]>(() => {
    return [...(data?.drift_logs ?? [])].sort(
      (a, b) =>
        (parseIsoDate(b.timestamp)?.getTime() ?? 0) -
        (parseIsoDate(a.timestamp)?.getTime() ?? 0)
    )
  }, [data?.drift_logs])

  const sortedGraphTrace = useMemo<AgentGraphTraceRow[]>(() => {
    const raw = data?.graph_trace ?? []
    const filtered =
      traceStatusFilter === "all"
        ? raw
        : raw.filter((step) => step.status === traceStatusFilter)

    const runFirstTime = new Map<string, number>()
    for (const step of filtered) {
      const ts = parseIsoDate(step.started_at)?.getTime() ?? 0
      const prev = runFirstTime.get(step.run_id)
      if (prev === undefined || ts > prev) {
        runFirstTime.set(step.run_id, ts)
      }
    }

    return [...filtered].sort((a, b) => {
      if (a.run_id !== b.run_id) {
        const runDiff =
          (runFirstTime.get(b.run_id) ?? 0) - (runFirstTime.get(a.run_id) ?? 0)
        if (runDiff !== 0) return runDiff
        return b.run_id.localeCompare(a.run_id)
      }
      return a.step_order - b.step_order
    })
  }, [data?.graph_trace, traceStatusFilter])

  const hasActiveFilters = useMemo(() => {
    if (dateFilter !== "") return true
    if (subTab === "calls") {
      return agentFilter !== "all" || callStatusFilter !== "all"
    }
    if (subTab === "failures") {
      return agentFilter !== "all"
    }
    if (subTab === "drift") {
      return (
        metricFilter !== "all" ||
        driftTypeFilter !== "all" ||
        driftStatusFilter !== "all"
      )
    }
    return traceStatusFilter !== "all"
  }, [
    subTab,
    dateFilter,
    agentFilter,
    callStatusFilter,
    metricFilter,
    driftTypeFilter,
    driftStatusFilter,
    traceStatusFilter,
  ])

  if (isLoading && !data) {
    return (
      <div className="flex flex-col gap-6">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          {Array.from({ length: 6 }).map((_, idx) => (
            <div
              key={idx}
              className="h-24 animate-pulse rounded-lg border border-border bg-muted"
            />
          ))}
        </div>
        <div className="h-[360px] animate-pulse rounded-xl border border-border bg-muted" />
      </div>
    )
  }

  if (errorMessage && !data) {
    return (
      <ErrorState
        title="Couldn't load automation and drift activity"
        message={errorMessage}
        onRetry={handleRetry}
        isRetrying={isFetching}
      />
    )
  }

  const hasAnyTelemetry =
    (data?.call_logs.length ?? 0) > 0 ||
    (data?.failed_calls.length ?? 0) > 0 ||
    (data?.drift_logs.length ?? 0) > 0 ||
    (data?.graph_trace.length ?? 0) > 0

  if (
    !data ||
    (data.status === "bootstrapping" && !hasAnyTelemetry && !hasActiveFilters)
  ) {
    return (
      <BootstrappingState
        city={data?.city ?? city ?? undefined}
        onRefresh={handleRetry}
        isRefreshing={isFetching}
      />
    )
  }

  const summary = data.summary
  const canLoadMore =
    limit < EXPANDED_LIMIT &&
    ((data.call_logs?.length ?? 0) >= limit ||
      (data.drift_logs?.length ?? 0) >= limit ||
      (data.graph_trace?.length ?? 0) >= limit)

  const renderCallTable = (
    rows: AgentCallLogRow[],
    emptyTitle: string,
    emptyDesc: string
  ) => {
    if (rows.length === 0) {
      return (
        <EmptyState
          icon={Activity}
          title={emptyTitle}
          description={emptyDesc}
          actionLabel={hasActiveFilters ? "Reset filters" : undefined}
          onAction={hasActiveFilters ? handleResetFilters : undefined}
          className="border-0 bg-muted shadow-none"
        />
      )
    }

    return (
      <Table
        aria-label="Agent activity log"
        containerClassName="max-h-[480px] overflow-auto rounded-md border border-border"
      >
        <TableHeader className="sticky top-0 z-10 bg-muted">
          <TableRow className="hover:bg-muted">
            <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
              Status
            </TableHead>
            <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
              Timestamp
            </TableHead>
            <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
              Agent
            </TableHead>
            <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
              Routing
            </TableHead>
            <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
              Attempt
            </TableHead>
            <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
              Latency
            </TableHead>
            <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
              Summary
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const rowKey = `call-${row.id}-${row.attempt_order}`
            const isExpanded = expandedKeys.has(rowKey)
            const usedBackup = row.fallback_used || row.attempt_order > 1

            return (
              <Fragment key={rowKey}>
                <TableRow className="min-h-12">
                  <TableCell className="px-4 py-3 text-sm">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {renderCallStatusBadge(row.status)}
                      {usedBackup ? (
                        <span className="inline-flex h-6 items-center gap-1 rounded-full border border-alert-warning bg-alert-warning-bg px-2 text-xs font-medium text-alert-warning">
                          <TriangleAlert
                            className="size-3 shrink-0"
                            strokeWidth={1.75}
                          />
                          <span>Automatic backup</span>
                        </span>
                      ) : null}
                    </div>
                  </TableCell>
                  <TableCell className="px-4 py-3 text-sm tabular-nums">
                    <TimestampPkt
                      timestamp={row.timestamp_utc}
                      timeZone={resolvedTimeZone}
                      mode="datetime"
                    />
                  </TableCell>
                  <TableCell className="px-4 py-3 text-sm font-medium text-foreground">
                    {formatAgentRole(row.agent_name)}
                  </TableCell>
                  <TableCell className="px-4 py-3 text-sm text-foreground">
                    {usedBackup ? "Backup service" : "Primary service"}
                  </TableCell>
                  <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                    #{row.attempt_order}
                  </TableCell>
                  <TableCell className="px-4 py-3 text-right text-sm font-semibold tabular-nums">
                    {formatLatency(row.latency_ms)}
                  </TableCell>
                  <TableCell className="px-4 py-3 text-right text-sm">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      aria-expanded={isExpanded}
                      onClick={() => toggleExpanded(rowKey)}
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
                    <TableCell colSpan={7} className="p-4 whitespace-normal">
                      <div className="flex flex-col gap-2 rounded-lg bg-muted p-4 text-xs text-foreground">
                        <div className="flex flex-wrap items-center gap-x-6 gap-y-1">
                          <span>
                            Agent:{" "}
                            <span className="font-semibold">
                              {formatAgentRole(row.agent_name)}
                            </span>
                          </span>
                          <span>
                            Routing:{" "}
                            <span className="font-semibold">
                              {usedBackup
                                ? `Automatic backup (attempt #${row.attempt_order})`
                                : "Primary service"}
                            </span>
                          </span>
                          <span>
                            Response time:{" "}
                            <span className="font-semibold tabular-nums">
                              {formatLatency(row.latency_ms)}
                            </span>
                          </span>
                        </div>
                        <p className="text-foreground/85">
                          {row.status === "failed"
                            ? "This attempt did not complete within normal limits; automatic recovery was triggered so the forecast workflow continued uninterrupted."
                            : usedBackup
                              ? "Completed via automatic backup routing after an earlier attempt was unavailable."
                              : "Completed cleanly on the primary service."}
                        </p>
                      </div>
                    </TableCell>
                  </TableRow>
                ) : null}
              </Fragment>
            )
          })}
        </TableBody>
      </Table>
    )
  }

  return (
    <div className="flex flex-col gap-6 md:gap-8">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
            Agents and drift monitoring
          </h2>
          <p className="mt-1 max-w-[68ch] text-sm/5 text-muted-foreground">
            Automated advisory reliability, data and forecast drift checks, and
            workflow step timings.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <Clock className="size-3.5" strokeWidth={1.75} aria-hidden="true" />
            <span>
              Daily briefing:{" "}
              <span className="font-medium text-foreground tabular-nums">
                {formatHourSchedule(data.daily_report_local_hour, tzShortLabel)}
              </span>
            </span>
          </span>
          {isFetching ? (
            <span className="font-medium text-muted-foreground">
              Updating logs...
            </span>
          ) : null}
        </div>
      </div>

      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh automation and drift activity"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      ) : null}

      {/* Summary KPI strip */}
      <dl className="grid grid-cols-2 gap-3 md:grid-cols-3 md:gap-4 xl:grid-cols-6">
        <div className="rounded-lg border border-border bg-card p-4 shadow-(--shadow-card)">
          <dt className="text-xs font-medium text-muted-foreground">
            Total agent calls
          </dt>
          <dd className="mt-1 font-display text-stat font-semibold text-foreground tabular-nums">
            {summary.total_calls.toLocaleString()}
          </dd>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 shadow-(--shadow-card)">
          <dt className="text-xs font-medium text-muted-foreground">
            Failed attempts
          </dt>
          <dd className="mt-1 font-display text-stat font-semibold text-foreground tabular-nums">
            {summary.failed_calls.toLocaleString()}
          </dd>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 shadow-(--shadow-card)">
          <dt className="text-xs font-medium text-muted-foreground">
            Backup recoveries
          </dt>
          <dd className="mt-1 font-display text-stat font-semibold text-foreground tabular-nums">
            {summary.fallback_calls.toLocaleString()}
          </dd>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 shadow-(--shadow-card)">
          <dt className="text-xs font-medium text-muted-foreground">
            Drift detected
          </dt>
          <dd className="mt-1 font-display text-stat font-semibold text-foreground tabular-nums">
            {summary.drift_detected_count.toLocaleString()}
          </dd>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 shadow-(--shadow-card)">
          <dt className="text-xs font-medium text-muted-foreground">
            Incomplete checks
          </dt>
          <dd className="mt-1 font-display text-stat font-semibold text-foreground tabular-nums">
            {summary.drift_check_failed_count.toLocaleString()}
          </dd>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 shadow-(--shadow-card)">
          <dt className="text-xs font-medium text-muted-foreground">
            Workflow steps
          </dt>
          <dd className="mt-1 font-display text-stat font-semibold text-foreground tabular-nums">
            {summary.trace_steps.toLocaleString()}
          </dd>
        </div>
      </dl>

      {/* Sub-tabs + Scoped Filter Row + Active Table */}
      <Tabs
        value={subTab}
        onValueChange={(val) => setSubTab(val as AgentsSubTab)}
        className="gap-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <TabsList className="h-auto w-full flex-wrap justify-start gap-1 p-1 sm:h-10 sm:w-fit">
            <TabsTrigger value="calls" className="h-8 px-4 text-sm">
              Call log ({sortedCallLogs.length})
            </TabsTrigger>
            <TabsTrigger value="failures" className="h-8 px-4 text-sm">
              Failed calls and fallbacks ({failedAndFallbackRows.length})
            </TabsTrigger>
            <TabsTrigger value="drift" className="h-8 px-4 text-sm">
              Drift log ({sortedDriftLogs.length})
            </TabsTrigger>
            <TabsTrigger value="trace" className="h-8 px-4 text-sm">
              Graph trace ({sortedGraphTrace.length})
            </TabsTrigger>
          </TabsList>

          {canLoadMore ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 px-3 text-xs font-medium"
              disabled={isFetching}
              onClick={() => setLimit(EXPANDED_LIMIT)}
            >
              Load up to {EXPANDED_LIMIT} records
            </Button>
          ) : null}
        </div>

        {/* Filter Row */}
        <section
          aria-label="Log filters"
          className="flex flex-wrap items-end gap-3 rounded-xl border border-border bg-card p-4 shadow-(--shadow-card)"
        >
          {(subTab === "calls" || subTab === "failures") && (
            <>
              <div className="flex flex-col gap-1">
                <label
                  htmlFor="filter-agent"
                  className="text-xs font-medium text-muted-foreground"
                >
                  Agent
                </label>
                <Select value={agentFilter} onValueChange={setAgentFilter}>
                  <SelectTrigger
                    id="filter-agent"
                    className="h-10 min-w-[164px] bg-card text-sm"
                  >
                    <SelectValue placeholder="All agents" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value="all">All agents</SelectItem>
                    {seenAgents.map((agent) => (
                      <SelectItem key={agent} value={agent}>
                        {formatAgentRole(agent)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {subTab === "calls" ? (
                <div className="flex flex-col gap-1">
                  <label
                    htmlFor="filter-call-status"
                    className="text-xs font-medium text-muted-foreground"
                  >
                    Status
                  </label>
                  <Select
                    value={callStatusFilter}
                    onValueChange={setCallStatusFilter}
                  >
                    <SelectTrigger
                      id="filter-call-status"
                      className="h-10 min-w-[136px] bg-card text-sm"
                    >
                      <SelectValue placeholder="All statuses" />
                    </SelectTrigger>
                    <SelectContent position="popper">
                      <SelectItem value="all">All statuses</SelectItem>
                      <SelectItem value="success">Success</SelectItem>
                      <SelectItem value="failed">Failed</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              ) : null}
            </>
          )}

          {subTab === "drift" && (
            <>
              <div className="flex flex-col gap-1">
                <label
                  htmlFor="filter-drift-metric"
                  className="text-xs font-medium text-muted-foreground"
                >
                  Metric
                </label>
                <Select value={metricFilter} onValueChange={setMetricFilter}>
                  <SelectTrigger
                    id="filter-drift-metric"
                    className="h-10 min-w-[156px] bg-card text-sm"
                  >
                    <SelectValue placeholder="All metrics" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value="all">All metrics</SelectItem>
                    {seenMetrics.map((metric) => (
                      <SelectItem key={metric} value={metric}>
                        {resolveMetricLabel(metric, metricSpecsHint)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1">
                <label
                  htmlFor="filter-drift-type"
                  className="text-xs font-medium text-muted-foreground"
                >
                  Drift type
                </label>
                <Select
                  value={driftTypeFilter}
                  onValueChange={setDriftTypeFilter}
                >
                  <SelectTrigger
                    id="filter-drift-type"
                    className="h-10 min-w-[140px] bg-card text-sm"
                  >
                    <SelectValue placeholder="All types" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value="all">All types</SelectItem>
                    <SelectItem value="data_drift">Data drift</SelectItem>
                    <SelectItem value="model_drift">Model drift</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1">
                <label
                  htmlFor="filter-drift-status"
                  className="text-xs font-medium text-muted-foreground"
                >
                  Drift status
                </label>
                <Select
                  value={driftStatusFilter}
                  onValueChange={setDriftStatusFilter}
                >
                  <SelectTrigger
                    id="filter-drift-status"
                    className="h-10 min-w-[160px] bg-card text-sm"
                  >
                    <SelectValue placeholder="All statuses" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    <SelectItem value="all">All statuses</SelectItem>
                    <SelectItem value="no_drift">No drift</SelectItem>
                    <SelectItem value="drift_detected">Drift detected</SelectItem>
                    <SelectItem value="check_failed">
                      Check incomplete
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </>
          )}

          {subTab === "trace" && (
            <div className="flex flex-col gap-1">
              <label
                htmlFor="filter-trace-status"
                className="text-xs font-medium text-muted-foreground"
              >
                Step status
              </label>
              <Select
                value={traceStatusFilter}
                onValueChange={setTraceStatusFilter}
              >
                <SelectTrigger
                  id="filter-trace-status"
                  className="h-10 min-w-[144px] bg-card text-sm"
                >
                  <SelectValue placeholder="All steps" />
                </SelectTrigger>
                <SelectContent position="popper">
                  <SelectItem value="all">All steps</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="skipped">Skipped</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="flex flex-col gap-1">
            <label
              htmlFor="filter-date"
              className="text-xs font-medium text-muted-foreground"
            >
              Date ({tzShortLabel})
            </label>
            <input
              id="filter-date"
              type="date"
              value={dateFilter}
              onChange={(event) => setDateFilter(event.target.value)}
              className="h-10 rounded-lg border border-input bg-card px-3 text-sm text-foreground tabular-nums focus-visible:outline-2 focus-visible:outline-ring"
            />
          </div>

          {hasActiveFilters ? (
            <Button
              type="button"
              variant="outline"
              onClick={handleResetFilters}
              className="h-10 gap-1.5 px-3 text-sm"
            >
              <RotateCcw className="size-4" strokeWidth={1.75} />
              <span>Reset filters</span>
            </Button>
          ) : null}
        </section>

        {/* Sub-tab 1: Call Log */}
        <TabsContent value="calls">
          <div className="rounded-xl border border-border bg-card p-4 shadow-(--shadow-card) md:p-5">
            {renderCallTable(
              sortedCallLogs,
              "No agent calls match these filters",
              hasActiveFilters
                ? "Clear your agent, status, or date filters to view recorded activity."
                : "Agent call records appear here once advisory or report automation runs."
            )}
          </div>
        </TabsContent>

        {/* Sub-tab 2: Failed calls and fallbacks */}
        <TabsContent value="failures">
          <div className="rounded-xl border border-border bg-card p-4 shadow-(--shadow-card) md:p-5">
            {renderCallTable(
              failedAndFallbackRows,
              "No failed calls or fallbacks recorded",
              hasActiveFilters
                ? "No failed attempts or backup recoveries match the active filters."
                : "All recorded agent calls succeeded on the primary service without requiring backup recovery."
            )}
          </div>
        </TabsContent>

        {/* Sub-tab 3: Drift Log */}
        <TabsContent value="drift">
          <div className="rounded-xl border border-border bg-card p-4 shadow-(--shadow-card) md:p-5">
            {sortedDriftLogs.length === 0 ? (
              <EmptyState
                icon={ShieldCheck}
                title="No drift checks match these filters"
                description={
                  hasActiveFilters
                    ? "Clear your metric, drift type, status, or date filters to view drift history."
                    : "Data and forecast drift checks appear here once recent observations and errors are evaluated."
                }
                actionLabel={hasActiveFilters ? "Reset filters" : undefined}
                onAction={hasActiveFilters ? handleResetFilters : undefined}
                className="border-0 bg-muted shadow-none"
              />
            ) : (
              <Table
                aria-label="Drift detection log"
                containerClassName="max-h-[480px] overflow-auto rounded-md border border-border"
              >
                <TableHeader className="sticky top-0 z-10 bg-muted">
                  <TableRow className="hover:bg-muted">
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Status
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Checked
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Metric
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Check type
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                      Score / Threshold
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Outcome
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                      Summary
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedDriftLogs.map((row) => {
                    const rowKey = `drift-${row.id}`
                    const isExpanded = expandedKeys.has(rowKey)

                    return (
                      <Fragment key={rowKey}>
                        <TableRow className="min-h-12">
                          <TableCell className="px-4 py-3 text-sm">
                            {renderDriftStatusBadge(row.status)}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm tabular-nums">
                            <TimestampPkt
                              timestamp={row.timestamp}
                              timeZone={resolvedTimeZone}
                              mode="datetime"
                            />
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm font-medium text-foreground">
                            {resolveMetricLabel(row.metric, metricSpecsHint)}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm">
                            <div className="font-medium text-foreground">
                              {humanizeIdentifier(row.drift_type)}
                            </div>
                            {row.test_name ? (
                              <div className="text-xs text-muted-foreground">
                                {formatDriftTestName(row.test_name)}
                              </div>
                            ) : null}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-right text-sm tabular-nums">
                            <div className="font-semibold text-foreground">
                              {formatMetricValue(row.score, 3)} /{" "}
                              {formatMetricValue(row.threshold, 3)}
                            </div>
                            {isFiniteNum(row.secondary_score) ? (
                              <div className="text-xs text-muted-foreground">
                                Secondary: {row.secondary_score.toFixed(3)}
                              </div>
                            ) : null}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm">
                            <div className="flex flex-wrap items-center gap-1.5">
                              <span className="text-foreground">
                                {formatDriftAction(row.action_taken)}
                              </span>
                              {row.retrain_triggered ? (
                                <Badge
                                  variant="secondary"
                                  className="h-5 px-2 text-[11px]"
                                >
                                  Retrain triggered
                                </Badge>
                              ) : null}
                            </div>
                          </TableCell>
                          <TableCell className="px-4 py-3 text-right text-sm">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              aria-expanded={isExpanded}
                              onClick={() => toggleExpanded(rowKey)}
                              className="h-8 gap-1 px-2 text-xs text-foreground"
                            >
                              {isExpanded ? (
                                <ChevronDown
                                  className="size-3.5"
                                  strokeWidth={1.75}
                                />
                              ) : (
                                <ChevronRight
                                  className="size-3.5"
                                  strokeWidth={1.75}
                                />
                              )}
                              <span>Details</span>
                            </Button>
                          </TableCell>
                        </TableRow>

                        {isExpanded ? (
                          <TableRow className="bg-muted/70 hover:bg-muted/70">
                            <TableCell
                              colSpan={7}
                              className="p-4 whitespace-normal"
                            >
                              <div className="flex flex-col gap-2 rounded-lg bg-muted p-4 text-xs text-foreground">
                                <div className="flex flex-wrap items-center gap-x-6 gap-y-1">
                                  <span>
                                    Check:{" "}
                                    <span className="font-semibold">
                                      {formatDriftTestName(row.test_name)}
                                    </span>
                                  </span>
                                  <span>
                                    Retraining triggered:{" "}
                                    <span className="font-semibold">
                                      {row.retrain_triggered ? "Yes" : "No"}
                                    </span>
                                  </span>
                                </div>
                                <p className="text-foreground/85">
                                  {row.status === "drift_detected"
                                    ? `${resolveMetricLabel(row.metric, metricSpecsHint)} exceeded the configured drift threshold (${formatMetricValue(row.threshold, 3)}), triggering a model refresh.`
                                    : row.status === "check_failed"
                                      ? `Insufficient baseline or recent samples were available to evaluate ${resolveMetricLabel(row.metric, metricSpecsHint)} drift.`
                                      : `${resolveMetricLabel(row.metric, metricSpecsHint)} remains within normal baseline bounds.`}
                                </p>
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
          </div>
        </TabsContent>

        {/* Sub-tab 4: Graph Trace */}
        <TabsContent value="trace">
          <div className="rounded-xl border border-border bg-card p-4 shadow-(--shadow-card) md:p-5">
            {sortedGraphTrace.length === 0 ? (
              <EmptyState
                icon={GitBranch}
                title="No workflow steps recorded"
                description={
                  hasActiveFilters
                    ? "Clear your step status or date filters to view recorded workflow steps."
                    : "Workflow step timings and branch outcomes appear here once the multi-agent run completes."
                }
                actionLabel={hasActiveFilters ? "Reset filters" : undefined}
                onAction={hasActiveFilters ? handleResetFilters : undefined}
                className="border-0 bg-muted shadow-none"
              />
            ) : (
              <Table
                aria-label="Workflow execution trace"
                containerClassName="max-h-[480px] overflow-auto rounded-md border border-border"
              >
                <TableHeader className="sticky top-0 z-10 bg-muted">
                  <TableRow className="hover:bg-muted">
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Step
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Stage
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Status
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Decision
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-xs font-medium text-foreground">
                      Started
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                      Duration
                    </TableHead>
                    <TableHead className="h-10 px-4 py-2 text-right text-xs font-medium text-foreground">
                      Summary
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedGraphTrace.map((row) => {
                    const rowKey = `trace-${row.id}-${row.run_id}-${row.step_order}`
                    const isExpanded = expandedKeys.has(rowKey)
                    const summaryEntries = Object.entries(
                      row.summary_json ?? {}
                    ).filter(
                      ([, val]) =>
                        typeof val === "number" ||
                        typeof val === "boolean" ||
                        typeof val === "string"
                    )

                    return (
                      <Fragment key={rowKey}>
                        <TableRow className="min-h-12">
                          <TableCell className="px-4 py-3 text-sm font-semibold tabular-nums">
                            {String(row.step_order).padStart(2, "0")}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm font-medium text-foreground">
                            {formatAgentRole(row.node_name)}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm">
                            {renderGraphStatusBadge(row.status)}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm">
                            {row.branch_decision ? (
                              <Badge
                                variant="secondary"
                                className="h-6 px-2.5 text-xs font-medium"
                              >
                                {formatBranchDecision(row.branch_decision)}
                              </Badge>
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-sm tabular-nums">
                            <TimestampPkt
                              timestamp={row.started_at}
                              timeZone={resolvedTimeZone}
                              mode="datetime"
                            />
                          </TableCell>
                          <TableCell className="px-4 py-3 text-right text-sm font-semibold tabular-nums">
                            {formatLatency(row.duration_ms)}
                          </TableCell>
                          <TableCell className="px-4 py-3 text-right text-sm">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              aria-expanded={isExpanded}
                              onClick={() => toggleExpanded(rowKey)}
                              className="h-8 gap-1 px-2 text-xs text-foreground"
                            >
                              {isExpanded ? (
                                <ChevronDown
                                  className="size-3.5"
                                  strokeWidth={1.75}
                                />
                              ) : (
                                <ChevronRight
                                  className="size-3.5"
                                  strokeWidth={1.75}
                                />
                              )}
                              <span>Details</span>
                            </Button>
                          </TableCell>
                        </TableRow>

                        {isExpanded ? (
                          <TableRow className="bg-muted/70 hover:bg-muted/70">
                            <TableCell
                              colSpan={7}
                              className="p-4 whitespace-normal"
                            >
                              <div className="flex flex-col gap-3 rounded-lg bg-muted p-4 text-xs text-foreground">
                                {summaryEntries.length > 0 ? (
                                  <div className="flex flex-wrap gap-2">
                                    {summaryEntries.map(([key, val]) => (
                                      <span
                                        key={key}
                                        className="rounded-sm bg-card px-2 py-1 text-xs text-foreground tabular-nums"
                                      >
                                        <span className="text-muted-foreground">
                                          {humanizeIdentifier(key)}:
                                        </span>{" "}
                                        <span className="font-medium">
                                          {typeof val === "boolean"
                                            ? val
                                              ? "Yes"
                                              : "No"
                                            : typeof val === "string"
                                              ? humanizeIdentifier(val)
                                              : String(val)}
                                        </span>
                                      </span>
                                    ))}
                                  </div>
                                ) : (
                                  <p className="text-foreground/75">
                                    Step completed in{" "}
                                    {formatLatency(row.duration_ms)}.
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
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
