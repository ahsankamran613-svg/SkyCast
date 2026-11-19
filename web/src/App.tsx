import { useCallback, useEffect, useMemo, useState } from "react"

import { AppShell, NAV_TABS, type NavTabId } from "@/components/app-shell"
import { AgentsDriftTab } from "@/components/tabs/agents-drift-tab"
import { EdaTab } from "@/components/tabs/eda-tab"
import { ForecastsTab } from "@/components/tabs/forecasts-tab"
import { ModelsTab } from "@/components/tabs/models-tab"
import { OverviewTab } from "@/components/tabs/overview-tab"
import { TechnicalReportTab } from "@/components/tabs/technical-report-tab"
import {
  fetchDashboardOverview,
  fetchHealth,
  type DashboardOverviewResponse,
  type HealthResponse,
} from "@/lib/api"

function getInitialTab(): NavTabId {
  if (typeof window === "undefined") {
    return "overview"
  }
  const hash = window.location.hash.replace(/^#/, "") as NavTabId
  return NAV_TABS.some((tab) => tab.id === hash) ? hash : "overview"
}

interface StationSnapshot {
  health: HealthResponse | null
  overview: DashboardOverviewResponse | null
  errorMessage: string | null
  aborted: boolean
}

async function fetchStationSnapshot(
  signal?: AbortSignal
): Promise<StationSnapshot> {
  const [healthRes, overviewRes] = await Promise.allSettled([
    fetchHealth({ signal }),
    fetchDashboardOverview({ signal }),
  ])

  if (signal?.aborted) {
    return { health: null, overview: null, errorMessage: null, aborted: true }
  }

  const health = healthRes.status === "fulfilled" ? healthRes.value : null

  if (overviewRes.status === "fulfilled") {
    return {
      health,
      overview: overviewRes.value,
      errorMessage: null,
      aborted: false,
    }
  }

  const reason = overviewRes.reason
  if (reason instanceof DOMException && reason.name === "AbortError") {
    return { health: null, overview: null, errorMessage: null, aborted: true }
  }

  const errorMessage =
    "Couldn't load weather data. Check your connection and refresh."

  return { health, overview: null, errorMessage, aborted: false }
}

export function App() {
  const [activeTab, setActiveTab] = useState<NavTabId>(getInitialTab)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [overview, setOverview] = useState<DashboardOverviewResponse | null>(
    null
  )
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [refreshNonce, setRefreshNonce] = useState(0)

  const applySnapshot = useCallback((snapshot: StationSnapshot) => {
    if (snapshot.aborted) {
      return
    }
    if (snapshot.health) {
      setHealth(snapshot.health)
    }
    if (snapshot.overview) {
      setOverview(snapshot.overview)
    }
    setErrorMessage(snapshot.errorMessage)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void fetchStationSnapshot(controller.signal).then((snapshot) => {
      if (snapshot.aborted) {
        return
      }
      applySnapshot(snapshot)
      setIsLoading(false)
    })
    return () => controller.abort()
  }, [applySnapshot])

  const handleRefresh = useCallback(() => {
    setErrorMessage(null)
    setIsRefreshing(true)
    setRefreshNonce((n) => n + 1)
    void fetchStationSnapshot().then((snapshot) => {
      applySnapshot(snapshot)
      setIsRefreshing(false)
    })
  }, [applySnapshot])

  const handleTabChange = useCallback((tab: NavTabId) => {
    setActiveTab(tab)
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", `#${tab}`)
    }
  }, [])

  const city = overview?.city ?? health?.city ?? null
  const timeZone = overview?.timezone ?? health?.timezone ?? null
  const coordinates = overview?.coordinates ?? health?.coordinates ?? null
  const updatedAtUtc =
    overview?.updated_at_utc ?? health?.last_ingestion_timestamp ?? null

  const shellStatus = useMemo(() => {
    if (isLoading && !overview && !health) return "loading" as const
    if (errorMessage && !overview) return "error" as const
    return overview?.status ?? health?.status ?? "bootstrapping"
  }, [isLoading, overview, health, errorMessage])

  return (
    <AppShell
      activeTab={activeTab}
      onTabChange={handleTabChange}
      city={city}
      timeZone={timeZone}
      coordinates={coordinates}
      updatedAtUtc={updatedAtUtc}
      status={shellStatus}
      isRefreshing={isRefreshing}
      onRefresh={handleRefresh}
    >
      {activeTab === "overview" ? (
        <OverviewTab
          overview={overview}
          isLoading={isLoading}
          isRefreshing={isRefreshing}
          errorMessage={errorMessage}
          city={city}
          timeZone={timeZone}
          onRefresh={handleRefresh}
        />
      ) : activeTab === "forecasts" ? (
        <ForecastsTab
          city={city}
          timeZone={timeZone}
          metricSpecsHint={overview?.metric_specs ?? null}
          coreMetricGroupsHint={overview?.core_metric_groups ?? null}
          refreshNonce={refreshNonce}
          onShellRefresh={handleRefresh}
        />
      ) : activeTab === "models" ? (
        <ModelsTab
          city={city}
          timeZone={timeZone}
          metricSpecsHint={overview?.metric_specs ?? null}
          coreMetricGroupsHint={overview?.core_metric_groups ?? null}
          refreshNonce={refreshNonce}
          onShellRefresh={handleRefresh}
        />
      ) : activeTab === "eda" ? (
        <EdaTab
          city={city}
          timeZone={timeZone}
          metricSpecsHint={overview?.metric_specs ?? null}
          coreMetricGroupsHint={overview?.core_metric_groups ?? null}
          refreshNonce={refreshNonce}
          onShellRefresh={handleRefresh}
        />
      ) : activeTab === "agents" ? (
        <AgentsDriftTab
          city={city}
          timeZone={timeZone}
          metricSpecsHint={overview?.metric_specs ?? null}
          refreshNonce={refreshNonce}
          onShellRefresh={handleRefresh}
        />
      ) : (
        <TechnicalReportTab
          city={city}
          timeZone={timeZone}
          refreshNonce={refreshNonce}
          onShellRefresh={handleRefresh}
        />
      )}
    </AppShell>
  )
}

export default App
