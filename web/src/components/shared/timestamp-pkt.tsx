import { cn } from "cn"

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  formatLocalDateTime,
  formatLocalTime,
  formatRelativeAge,
  formatUtcDateTime,
  getTimeZoneLabel,
  parseIsoDate,
  resolveTimeZone,
} from "@/lib/time"

export interface TimestampPktProps {
  timestamp: string | null | undefined
  timeZone?: string | null
  mode?: "datetime" | "time" | "relative" | "datetime-relative"
  showZone?: boolean
  fallback?: string
  className?: string
}

export function TimestampPkt({
  timestamp,
  timeZone,
  mode = "datetime",
  showZone = true,
  fallback = "—",
  className,
}: TimestampPktProps) {
  const parsed = parseIsoDate(timestamp)
  if (!parsed) {
    return (
      <span className={cn("text-muted-foreground tabular-nums", className)}>
        {fallback}
      </span>
    )
  }

  const resolvedTz = resolveTimeZone(timeZone)
  const zoneLabel = getTimeZoneLabel(parsed, resolvedTz)
  const localFull = formatLocalDateTime(timestamp, resolvedTz, {
    includeZone: true,
  })
  const utcFull = formatUtcDateTime(timestamp)
  const relativeText = formatRelativeAge(timestamp)

  let displayText: string
  if (mode === "time") {
    displayText = formatLocalTime(timestamp, resolvedTz, showZone)
  } else if (mode === "relative") {
    displayText = relativeText
  } else if (mode === "datetime-relative") {
    displayText = `${formatLocalDateTime(timestamp, resolvedTz, {
      includeZone: showZone,
    })} (${relativeText})`
  } else {
    displayText = formatLocalDateTime(timestamp, resolvedTz, {
      includeZone: showZone,
    })
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <time
          dateTime={parsed.toISOString()}
          tabIndex={0}
          className={cn(
            "cursor-help rounded-sm underline decoration-border decoration-dotted underline-offset-4 tabular-nums focus-visible:outline-2 focus-visible:outline-ring",
            className
          )}
        >
          {displayText}
        </time>
      </TooltipTrigger>
      <TooltipContent className="flex flex-col gap-0.5 text-left">
        <span className="font-medium text-popover-foreground tabular-nums">
          {localFull}
        </span>
        <span className="text-muted-foreground tabular-nums">
          {resolvedTz} ({zoneLabel})
        </span>
        <span className="text-muted-foreground tabular-nums">{utcFull}</span>
      </TooltipContent>
    </Tooltip>
  )
}
