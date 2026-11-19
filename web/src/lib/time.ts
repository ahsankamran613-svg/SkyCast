const DEFAULT_FALLBACK_TZ = "UTC"

export function resolveTimeZone(timeZone?: string | null): string {
  const candidate = timeZone?.trim()
  if (!candidate) {
    return DEFAULT_FALLBACK_TZ
  }
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: candidate })
    return candidate
  } catch {
    return DEFAULT_FALLBACK_TZ
  }
}

export function parseIsoDate(iso: string | null | undefined): Date | null {
  if (!iso || typeof iso !== "string") {
    return null
  }
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? null : date
}

export function getTimeZoneLabel(
  date: Date,
  timeZone?: string | null
): string {
  const tz = resolveTimeZone(timeZone)
  if (tz === "Asia/Karachi") {
    return "PKT"
  }
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      timeZoneName: "short",
    }).formatToParts(date)
    const tzPart = parts.find((part) => part.type === "timeZoneName")?.value
    return tzPart || tz
  } catch {
    return tz
  }
}

export function formatLocalDateTime(
  iso: string | null | undefined,
  timeZone?: string | null,
  options?: { includeSeconds?: boolean; includeZone?: boolean }
): string {
  const date = parseIsoDate(iso)
  if (!date) {
    return "—"
  }
  const tz = resolveTimeZone(timeZone)
  const includeSeconds = options?.includeSeconds ?? false
  const includeZone = options?.includeZone ?? true

  const formatted = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    ...(includeSeconds ? { second: "2-digit" } : {}),
    hour12: false,
  }).format(date)

  if (!includeZone) {
    return formatted
  }
  return `${formatted} ${getTimeZoneLabel(date, tz)}`
}

export function formatLocalTime(
  iso: string | null | undefined,
  timeZone?: string | null,
  includeZone = false
): string {
  const date = parseIsoDate(iso)
  if (!date) {
    return "—"
  }
  const tz = resolveTimeZone(timeZone)
  const formatted = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date)

  return includeZone ? `${formatted} ${getTimeZoneLabel(date, tz)}` : formatted
}

export function formatUtcDateTime(iso: string | null | undefined): string {
  const date = parseIsoDate(iso)
  if (!date) {
    return "—"
  }
  const formatted = new Intl.DateTimeFormat("en-GB", {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date)

  return `${formatted} UTC`
}

export function formatRelativeAge(
  iso: string | null | undefined,
  nowMs: number = Date.now()
): string {
  const date = parseIsoDate(iso)
  if (!date) {
    return "—"
  }
  const diffSeconds = Math.floor((nowMs - date.getTime()) / 1000)
  if (diffSeconds < 0) {
    const aheadMinutes = Math.round(Math.abs(diffSeconds) / 60)
    return aheadMinutes <= 1 ? "in a moment" : `in ${aheadMinutes} min`
  }
  if (diffSeconds < 60) {
    return "just now"
  }
  const diffMinutes = Math.floor(diffSeconds / 60)
  if (diffMinutes < 60) {
    return `${diffMinutes} min ago`
  }
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) {
    return `${diffHours} h ago`
  }
  const diffDays = Math.floor(diffHours / 24)
  return `${diffDays} d ago`
}
