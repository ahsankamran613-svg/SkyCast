import { Info, OctagonAlert, TriangleAlert, type LucideIcon } from "lucide-react"
import { cn } from "cn"

import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import type { AlertSeverity, MetricSpecsMap, WeatherAlert } from "@/lib/api"
import { formatMetricValue, getMetricSpec } from "@/lib/weather"

const SEVERITY_STYLES: Record<
  AlertSeverity,
  {
    label: string
    container: string
    badgeText: string
    icon: LucideIcon
  }
> = {
  info: {
    label: "Info",
    container: "border-border border-l-4 border-l-alert-info bg-alert-info-bg",
    badgeText: "text-alert-info",
    icon: Info,
  },
  warning: {
    label: "Warning",
    container:
      "border-border border-l-4 border-l-alert-warning bg-alert-warning-bg",
    badgeText: "text-alert-warning",
    icon: TriangleAlert,
  },
  critical: {
    label: "Critical",
    container:
      "border-border border-l-4 border-l-alert-critical bg-alert-critical-bg",
    badgeText: "text-alert-critical",
    icon: OctagonAlert,
  },
}

export interface AlertBannerProps {
  alert: WeatherAlert
  timeZone?: string | null
  metricSpecs?: MetricSpecsMap | null
  className?: string
}

export function AlertBanner({
  alert,
  timeZone,
  metricSpecs,
  className,
}: AlertBannerProps) {
  const style = SEVERITY_STYLES[alert.severity] ?? SEVERITY_STYLES.info
  const Icon = style.icon
  const spec = getMetricSpec(alert.metric, metricSpecs)
  const metricLabel = spec?.label ?? alert.metric
  const unit = spec?.unit ? ` ${spec.unit}` : ""

  return (
    <div
      role={alert.severity === "critical" ? "alert" : "status"}
      className={cn(
        "flex min-h-14 flex-col justify-between gap-2 rounded-lg border px-4 py-3 sm:flex-row sm:items-center sm:gap-4",
        style.container,
        className
      )}
    >
      <div className="flex items-start gap-3">
        <Icon
          className={cn("mt-0.5 size-5 shrink-0", style.badgeText)}
          strokeWidth={1.75}
          aria-hidden="true"
        />
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={cn(
                "text-xs font-semibold",
                style.badgeText
              )}
            >
              {style.label}
            </span>
            <h3 className="text-sm font-semibold text-foreground">
              {alert.title}
            </h3>
          </div>
          <p className="mt-0.5 text-sm text-foreground tabular-nums">
            {metricLabel} reached{" "}
            <span className="font-semibold">
              {formatMetricValue(alert.trigger_value, 1)}
              {unit}
            </span>{" "}
            (threshold {formatMetricValue(alert.threshold, 1)}
            {unit}).
          </p>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-1.5 pl-8 text-xs text-foreground sm:pl-0">
        <span>{alert.target_timestamp_utc ? "Target" : "Issued"}</span>
        <TimestampPkt
          timestamp={alert.target_timestamp_utc ?? alert.timestamp}
          timeZone={timeZone}
          mode="datetime"
          className="font-medium text-foreground"
        />
      </div>
    </div>
  )
}

export interface AlertBannerListProps {
  alerts: WeatherAlert[]
  timeZone?: string | null
  metricSpecs?: MetricSpecsMap | null
  className?: string
}

export function AlertBannerList({
  alerts,
  timeZone,
  metricSpecs,
  className,
}: AlertBannerListProps) {
  if (alerts.length === 0) {
    return null
  }

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {alerts.map((alert) => (
        <AlertBanner
          key={alert.id}
          alert={alert}
          timeZone={timeZone}
          metricSpecs={metricSpecs}
        />
      ))}
    </div>
  )
}
