import { useState } from "react"
import { HardHat, HeartPulse, ShieldCheck, Users } from "lucide-react"
import { cn } from "cn"

import { EmptyState } from "@/components/shared/empty-state"
import type { AlertSeverity, WeatherAlert } from "@/lib/api"

export interface AdvisoryCardsProps {
  alerts: WeatherAlert[]
  className?: string
}

const SEVERITY_RANK: Record<AlertSeverity, number> = {
  critical: 3,
  warning: 2,
  info: 1,
}

export function AdvisoryCards({ alerts, className }: AdvisoryCardsProps) {
  const sortedAlerts = [...alerts].sort(
    (a, b) => (SEVERITY_RANK[b.severity] ?? 0) - (SEVERITY_RANK[a.severity] ?? 0)
  )
  const [selectedId, setSelectedId] = useState<number | null>(null)

  if (sortedAlerts.length === 0) {
    return (
      <EmptyState
        icon={ShieldCheck}
        title="No active hazard advisories"
        description="No alerts right now. Advisories for the general public, sensitive groups, and outdoor workers appear here when a forecast crosses a hazard threshold."
        className={className}
      />
    )
  }

  const activeAlert =
    sortedAlerts.find((item) => item.id === selectedId) ?? sortedAlerts[0]

  const audiences = [
    {
      key: "general",
      title: "General public",
      body: activeAlert.advisory_general,
      icon: Users,
    },
    {
      key: "sensitive",
      title: "Sensitive groups",
      body: activeAlert.advisory_sensitive,
      icon: HeartPulse,
    },
    {
      key: "workers",
      title: "Outdoor workers",
      body: activeAlert.advisory_outdoor_workers,
      icon: HardHat,
    },
  ] as const

  return (
    <section className={cn("flex flex-col gap-3", className)} aria-label="Health and safety advisories">
      {sortedAlerts.length > 1 ? (
        <div
          className="flex flex-wrap items-center gap-2"
          role="tablist"
          aria-label="Select active hazard advisory"
        >
          {sortedAlerts.map((item) => {
            const isSelected = item.id === activeAlert.id
            return (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={isSelected}
                onClick={() => setSelectedId(item.id)}
                className={cn(
                  "h-9 rounded-full border px-3 text-xs font-medium transition-colors",
                  isSelected
                    ? "border-primary bg-accent text-accent-foreground"
                    : "border-border bg-card text-foreground hover:bg-muted"
                )}
              >
                {item.title}
              </button>
            )
          })}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3 md:gap-4">
        {audiences.map((audience) => {
          const Icon = audience.icon
          return (
            <article
              key={audience.key}
              className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card)"
            >
              <header className="flex items-center gap-2">
                <Icon
                  className="size-4 shrink-0 text-primary"
                  strokeWidth={1.75}
                  aria-hidden="true"
                />
                <h3 className="text-sm font-semibold text-foreground">
                  {audience.title}
                </h3>
              </header>
              <p className="max-w-[68ch] text-sm/5 text-muted-foreground">
                {audience.body}
              </p>
            </article>
          )
        })}
      </div>
    </section>
  )
}
