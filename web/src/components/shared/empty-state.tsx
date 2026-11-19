import { AlertCircle, Clock, Inbox, RefreshCw, type LucideIcon } from "lucide-react"
import { cn } from "cn"

import { Button } from "@/components/ui/button"

export interface EmptyStateProps {
  title: string
  description: string
  icon?: LucideIcon
  actionLabel?: string
  onAction?: () => void
  isActionLoading?: boolean
  className?: string
}

export function EmptyState({
  title,
  description,
  icon: Icon = Inbox,
  actionLabel,
  onAction,
  isActionLoading = false,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-start justify-between gap-4 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-(--shadow-card) sm:flex-row sm:items-center",
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
          <Icon className="size-5" strokeWidth={1.75} aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <h3 className="text-base/6 font-semibold text-foreground">{title}</h3>
          <p className="mt-1 max-w-[68ch] text-sm/5 text-muted-foreground">
            {description}
          </p>
        </div>
      </div>

      {actionLabel && onAction ? (
        <Button
          type="button"
          variant="outline"
          className="h-10 shrink-0 gap-2 px-4"
          onClick={onAction}
          disabled={isActionLoading}
        >
          <RefreshCw
            className={cn("size-4", isActionLoading && "animate-spin")}
            strokeWidth={1.75}
          />
          <span>{actionLabel}</span>
        </Button>
      ) : null}
    </div>
  )
}

export interface BootstrappingStateProps {
  title?: string
  description?: string
  city?: string
  onRefresh?: () => void
  isRefreshing?: boolean
  className?: string
}

export function BootstrappingState({
  title = "Weather data is updating",
  description,
  city,
  onRefresh,
  isRefreshing = false,
  className,
}: BootstrappingStateProps) {
  const defaultDesc = city
    ? `Latest observations and forecasts for ${city} are being prepared. Check back shortly or refresh.`
    : "Latest observations and forecasts are being prepared. Check back shortly or refresh."

  return (
    <EmptyState
      icon={Clock}
      title={title}
      description={description ?? defaultDesc}
      actionLabel={onRefresh ? "Refresh data" : undefined}
      onAction={onRefresh}
      isActionLoading={isRefreshing}
      className={className}
    />
  )
}

export interface ErrorStateProps {
  title?: string
  message: string
  onRetry?: () => void
  isRetrying?: boolean
  className?: string
}

export function ErrorState({
  title = "Couldn't load station data",
  message,
  onRetry,
  isRetrying = false,
  className,
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-start justify-between gap-4 rounded-xl border border-border border-l-4 border-l-alert-critical bg-alert-critical-bg p-5 sm:flex-row sm:items-center",
        className
      )}
    >
      <div className="flex items-start gap-3">
        <AlertCircle
          className="mt-0.5 size-5 shrink-0 text-alert-critical"
          strokeWidth={1.75}
          aria-hidden="true"
        />
        <div className="min-w-0">
          <h3 className="text-base/6 font-semibold text-foreground">{title}</h3>
          <p className="mt-1 max-w-[68ch] text-sm/5 text-foreground">
            {message}
          </p>
        </div>
      </div>

      {onRetry ? (
        <Button
          type="button"
          variant="outline"
          className="h-10 shrink-0 gap-2 bg-card px-4"
          onClick={onRetry}
          disabled={isRetrying}
        >
          <RefreshCw
            className={cn("size-4", isRetrying && "animate-spin")}
            strokeWidth={1.75}
          />
          <span>Refresh data</span>
        </Button>
      ) : null}
    </div>
  )
}
