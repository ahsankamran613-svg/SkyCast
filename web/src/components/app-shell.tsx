import { useState, type ReactNode } from "react"
import {
  Activity,
  ChartSpline,
  CloudSun,
  FileText,
  Grid2x2,
  Layers,
  MoreHorizontal,
  RefreshCw,
} from "lucide-react"
import { cn } from "cn"

import { ThemeToggle } from "@/components/theme-toggle"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type { ApiStatus, Coordinates } from "@/lib/api"

export type NavTabId =
  | "overview"
  | "forecasts"
  | "models"
  | "eda"
  | "agents"
  | "report"

export interface NavTabItem {
  id: NavTabId
  label: string
  mobileLabel: string
  description: string
  section: "weather" | "diagnostics"
  icon: typeof CloudSun
}

export const NAV_TABS: readonly NavTabItem[] = [
  {
    id: "overview",
    label: "Overview",
    mobileLabel: "Overview",
    description: "Current conditions, air quality, and 24-hour outlook",
    section: "weather",
    icon: CloudSun,
  },
  {
    id: "forecasts",
    label: "Forecasts",
    mobileLabel: "Forecasts",
    description: "24-hour trajectories, verification history, and hourly ledger",
    section: "weather",
    icon: ChartSpline,
  },
  {
    id: "models",
    label: "Models",
    mobileLabel: "Models",
    description: "Model accuracy rankings, backtest curves, and key drivers",
    section: "diagnostics",
    icon: Layers,
  },
  {
    id: "eda",
    label: "EDA",
    mobileLabel: "EDA",
    description: "Distributions, diurnal heatmaps, correlations, and data quality",
    section: "diagnostics",
    icon: Grid2x2,
  },
  {
    id: "agents",
    label: "Agents and drift",
    mobileLabel: "Agents",
    description: "Automation reliability, drift monitoring, and workflow trace",
    section: "diagnostics",
    icon: Activity,
  },
  {
    id: "report",
    label: "Technical report",
    mobileLabel: "Report",
    description: "End-to-end engineering documentation and evaluation summary",
    section: "diagnostics",
    icon: FileText,
  },
] as const

const WEATHER_NAV_TABS = NAV_TABS.filter((tab) => tab.section === "weather")
const DIAGNOSTICS_NAV_TABS = NAV_TABS.filter(
  (tab) => tab.section === "diagnostics"
)

const PRIMARY_MOBILE_TABS: readonly NavTabId[] = [
  "overview",
  "forecasts",
  "models",
  "agents",
]

const OVERFLOW_MOBILE_TABS: readonly NavTabId[] = ["eda", "report"]

function SkyCastMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 28 28"
      fill="none"
      className={cn("size-7 shrink-0", className)}
      aria-hidden="true"
    >
      <rect
        x="1.5"
        y="1.5"
        width="25"
        height="25"
        rx="7"
        className="fill-accent stroke-border"
        strokeWidth="1.25"
      />
      <path
        d="M6.5 17.5a7.5 7.5 0 0 1 15 0"
        className="stroke-primary"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M5.5 20.5h17"
        className="stroke-muted-foreground"
        strokeWidth="1.75"
        strokeLinecap="round"
      />
      <circle cx="19" cy="11.5" r="2.25" className="fill-chart-1" />
    </svg>
  )
}

function formatCoordinatePair(coords: Coordinates | null | undefined): string | null {
  if (!coords || !Number.isFinite(coords.lat) || !Number.isFinite(coords.lon)) {
    return null
  }
  const latHem = coords.lat >= 0 ? "N" : "S"
  const lonHem = coords.lon >= 0 ? "E" : "W"
  return `${Math.abs(coords.lat).toFixed(2)}°${latHem}, ${Math.abs(coords.lon).toFixed(2)}°${lonHem}`
}

export interface AppShellProps {
  activeTab: NavTabId
  onTabChange: (tab: NavTabId) => void
  city?: string | null
  timeZone?: string | null
  coordinates?: Coordinates | null
  updatedAtUtc?: string | null
  status?: ApiStatus | "error" | "loading"
  isRefreshing?: boolean
  onRefresh?: () => void
  children: ReactNode
}

export function AppShell({
  activeTab,
  onTabChange,
  city,
  timeZone,
  coordinates,
  updatedAtUtc,
  status = "loading",
  isRefreshing = false,
  onRefresh,
  children,
}: AppShellProps) {
  const [moreOpen, setMoreOpen] = useState(false)

  const coordText = formatCoordinatePair(coordinates)
  const isOverflowTabActive = OVERFLOW_MOBILE_TABS.includes(activeTab)
  const displayCity = city?.trim() || "Weather station"

  const statusIndicator = (() => {
    if (status === "ok") {
      return { dot: "bg-aqi-good", label: "Station online" }
    }
    if (status === "bootstrapping") {
      return { dot: "bg-alert-warning", label: "Updating data" }
    }
    if (status === "error") {
      return { dot: "bg-alert-critical", label: "Connection issue" }
    }
    return { dot: "bg-muted-foreground", label: "Connecting" }
  })()

  const renderDesktopNavItem = (tab: NavTabItem) => {
    const Icon = tab.icon
    const isActive = activeTab === tab.id
    return (
      <Tooltip key={tab.id}>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-current={isActive ? "page" : undefined}
            aria-label={tab.label}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              "flex h-10 w-full items-center justify-center gap-3 rounded-md px-3 text-sm transition-colors duration-120 xl:justify-start",
              isActive
                ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                : "text-sidebar-foreground/80 hover:bg-muted hover:text-sidebar-foreground"
            )}
          >
            <Icon className="size-5 shrink-0" strokeWidth={1.75} />
            <span className="hidden truncate xl:inline">{tab.label}</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="right" className="xl:hidden">
          {tab.label}
        </TooltipContent>
      </Tooltip>
    )
  }

  return (
    <div className="min-h-svh bg-background text-foreground">
      {/* Desktop / Tablet Sidebar (1024px+ rail, 1280px+ expanded) */}
      <aside className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-30 lg:flex lg:w-(--sidebar-w-rail) lg:flex-col lg:border-r lg:border-sidebar-border lg:bg-sidebar xl:w-(--sidebar-w)">
        <div className="flex h-(--header-h) items-center justify-center gap-3 border-b border-sidebar-border px-3 xl:justify-start xl:px-4">
          <SkyCastMark />
          <div className="hidden min-w-0 xl:block">
            <div className="font-display text-lg/5 font-semibold tracking-tight text-sidebar-foreground">
              SkyCast
            </div>
            <div className="text-xs text-muted-foreground">AgentWeather</div>
          </div>
        </div>

        <nav
          aria-label="Primary navigation"
          className="flex flex-1 flex-col gap-1 p-3"
        >
          <div className="hidden px-3 pt-1 pb-1.5 text-xs font-medium text-muted-foreground xl:block">
            Weather
          </div>
          {WEATHER_NAV_TABS.map(renderDesktopNavItem)}

          <div
            role="separator"
            className="my-2 border-t border-sidebar-border"
          />

          <div className="hidden px-3 pt-0.5 pb-1.5 text-xs font-medium text-muted-foreground xl:block">
            Diagnostics and research
          </div>
          {DIAGNOSTICS_NAV_TABS.map(renderDesktopNavItem)}
        </nav>

        <div className="hidden border-t border-sidebar-border px-4 py-3 text-xs text-muted-foreground xl:flex xl:flex-col xl:gap-1">
          <div className="flex items-center gap-2 text-sidebar-foreground">
            <span
              className={cn("size-2 shrink-0 rounded-full", statusIndicator.dot)}
              aria-hidden="true"
            />
            <span className="font-medium">{statusIndicator.label}</span>
          </div>
          {coordText ? (
            <div className="tabular-nums">{coordText}</div>
          ) : null}
        </div>
      </aside>

      {/* Main Column */}
      <div className="flex min-h-svh flex-col lg:pl-(--sidebar-w-rail) xl:pl-(--sidebar-w)">
        <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-3 border-b border-border bg-background/95 px-4 backdrop-blur-sm sm:px-6 lg:h-(--header-h) 2xl:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="lg:hidden">
              <SkyCastMark />
            </div>

            <div className="min-w-0">
              <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
                <h1 className="truncate font-display text-lg/6 font-semibold text-foreground sm:text-xl/6">
                  {displayCity}
                </h1>
                {updatedAtUtc ? (
                  <span className="text-xs text-muted-foreground sm:text-sm">
                    Updated{" "}
                    <TimestampPkt
                      timestamp={updatedAtUtc}
                      timeZone={timeZone}
                      mode="relative"
                      className="text-foreground"
                    />
                  </span>
                ) : (
                  <span className="text-xs text-muted-foreground sm:text-sm">
                    {status === "loading"
                      ? "Loading station data..."
                      : "Awaiting first observation"}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <ThemeToggle />
            {onRefresh ? (
              <Button
                type="button"
                variant="outline"
                onClick={onRefresh}
                disabled={isRefreshing}
                aria-label="Refresh data"
                className="h-11 min-w-11 gap-2 px-3 lg:h-10 lg:px-4"
              >
                <RefreshCw
                  className={cn("size-4 shrink-0", isRefreshing && "animate-spin")}
                  strokeWidth={1.75}
                />
                <span className="hidden sm:inline">Refresh data</span>
              </Button>
            ) : null}
          </div>
        </header>

        <main className="flex-1 px-4 pt-4 pb-24 sm:px-6 sm:pt-6 lg:pb-8 2xl:px-8">
          <div
            className={cn(
              "mx-auto w-full",
              activeTab === "report" ? "max-w-[800px]" : "max-w-(--page-max)"
            )}
          >
            {children}
          </div>
        </main>
      </div>

      {/* Mobile Bottom Navigation (<1024px) */}
      <nav
        aria-label="Mobile navigation"
        className="fixed inset-x-0 bottom-0 z-40 border-t border-border bg-sidebar/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-sm lg:hidden"
      >
        <div className="mx-auto flex h-(--bottom-nav-h) max-w-lg items-center justify-around px-1">
          {PRIMARY_MOBILE_TABS.map((tabId) => {
            const tab = NAV_TABS.find((item) => item.id === tabId)!
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                type="button"
                aria-current={isActive ? "page" : undefined}
                onClick={() => onTabChange(tab.id)}
                className={cn(
                  "flex min-h-14 min-w-[60px] flex-1 flex-col items-center justify-center gap-1 rounded-md px-1 text-xs transition-colors duration-120",
                  isActive
                    ? "font-medium text-primary"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Icon className="size-[22px]" strokeWidth={1.75} />
                <span className="truncate">{tab.mobileLabel}</span>
              </button>
            )
          })}

          <Sheet open={moreOpen} onOpenChange={setMoreOpen}>
            <SheetTrigger asChild>
              <button
                type="button"
                aria-expanded={moreOpen}
                aria-label="More views"
                className={cn(
                  "flex min-h-14 min-w-[60px] flex-1 flex-col items-center justify-center gap-1 rounded-md px-1 text-xs transition-colors duration-120",
                  isOverflowTabActive
                    ? "font-medium text-primary"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <MoreHorizontal className="size-[22px]" strokeWidth={1.75} />
                <span>More</span>
              </button>
            </SheetTrigger>
            <SheetContent
              side="bottom"
              className="rounded-t-2xl pb-[calc(1rem+env(safe-area-inset-bottom))]"
            >
              <SheetHeader className="pb-2">
                <SheetTitle>Additional views</SheetTitle>
                <SheetDescription>
                  Exploratory analysis and technical documentation for {displayCity}.
                </SheetDescription>
              </SheetHeader>

              <div className="flex flex-col gap-2 px-4">
                {OVERFLOW_MOBILE_TABS.map((tabId) => {
                  const tab = NAV_TABS.find((item) => item.id === tabId)!
                  const Icon = tab.icon
                  const isActive = activeTab === tab.id
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => {
                        onTabChange(tab.id)
                        setMoreOpen(false)
                      }}
                      className={cn(
                        "flex min-h-12 items-center gap-3 rounded-lg border px-4 py-3 text-left transition-colors",
                        isActive
                          ? "border-primary bg-accent text-accent-foreground"
                          : "border-border bg-card text-card-foreground hover:bg-muted"
                      )}
                    >
                      <Icon className="size-5 shrink-0" strokeWidth={1.75} />
                      <div className="min-w-0">
                        <div className="text-sm font-medium">{tab.label}</div>
                        <div className="text-xs text-muted-foreground">
                          {tab.description}
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>

              {coordText ? (
                <div className="mt-2 border-t border-border px-4 pt-3 text-xs text-muted-foreground">
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                    <span className="tabular-nums">{coordText}</span>
                    <span>Pakistan Time (PKT)</span>
                  </div>
                </div>
              ) : null}
            </SheetContent>
          </Sheet>
        </div>
      </nav>
    </div>
  )
}
