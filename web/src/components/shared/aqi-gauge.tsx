import { Info } from "lucide-react"
import { cn } from "cn"

import { AQI_LEVELS, aqiGaugeRatio, getAqiLevel } from "@/lib/aqi"

export interface AqiGaugeProps {
  value: number | null | undefined
  className?: string
  showScaleTicks?: boolean
}

const CX = 120
const CY = 118
const MAIN_RADIUS = 88
const BAND_RADIUS = 104
const ARC_LENGTH = Math.PI * MAIN_RADIUS

function polarToCartesian(
  cx: number,
  cy: number,
  radius: number,
  angleDeg: number
): { x: number; y: number } {
  const rad = (angleDeg * Math.PI) / 180
  return {
    x: cx + radius * Math.cos(rad),
    y: cy - radius * Math.sin(rad),
  }
}

function describeArcSegment(
  cx: number,
  cy: number,
  radius: number,
  startDeg: number,
  endDeg: number
): string {
  const start = polarToCartesian(cx, cy, radius, startDeg)
  const end = polarToCartesian(cx, cy, radius, endDeg)
  return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${radius} ${radius} 0 0 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`
}

export function AqiGauge({
  value,
  className,
  showScaleTicks = true,
}: AqiGaugeProps) {
  const level = getAqiLevel(value)
  const ratio = aqiGaugeRatio(value)
  const dashOffset = ARC_LENGTH * (1 - ratio)
  const indicatorAngle = 180 - ratio * 180
  const indicatorPos = polarToCartesian(CX, CY, MAIN_RADIUS, indicatorAngle)

  const displayValue =
    typeof value === "number" && Number.isFinite(value)
      ? Math.round(value).toString()
      : "—"

  const ariaLabel = level
    ? `US AQI ${displayValue}, ${level.label}`
    : "US AQI data unavailable"

  const LevelIcon = level?.icon ?? Info

  return (
    <div
      className={cn("flex flex-col items-center gap-3", className)}
      role="img"
      aria-label={ariaLabel}
    >
      <div className="relative flex flex-col items-center">
        <svg
          viewBox="0 0 240 144"
          className="h-[120px] w-[200px] overflow-visible md:h-[140px] md:w-[240px]"
          aria-hidden="true"
        >
          {/* Subtle 6-band EPA category guide arc along outer rim */}
          {AQI_LEVELS.map((band, index) => {
            const bandSpan = 180 / AQI_LEVELS.length
            const gapDeg = 2
            const startDeg = 180 - index * bandSpan - (index === 0 ? 0 : gapDeg / 2)
            const endDeg =
              180 -
              (index + 1) * bandSpan +
              (index === AQI_LEVELS.length - 1 ? 0 : gapDeg / 2)

            return (
              <path
                key={band.key}
                d={describeArcSegment(CX, CY, BAND_RADIUS, startDeg, endDeg)}
                fill="none"
                stroke={band.fill}
                strokeWidth={3}
                strokeLinecap="round"
                strokeOpacity={level?.key === band.key ? 0.95 : 0.3}
              />
            )
          })}

          {/* Track arc */}
          <path
            d={`M ${CX - MAIN_RADIUS} ${CY} A ${MAIN_RADIUS} ${MAIN_RADIUS} 0 0 1 ${CX + MAIN_RADIUS} ${CY}`}
            fill="none"
            stroke="var(--muted)"
            strokeWidth={14}
            strokeLinecap="round"
          />

          {/* Active reading arc */}
          {level ? (
            <>
              <path
                d={`M ${CX - MAIN_RADIUS} ${CY} A ${MAIN_RADIUS} ${MAIN_RADIUS} 0 0 1 ${CX + MAIN_RADIUS} ${CY}`}
                fill="none"
                stroke={level.fill}
                strokeWidth={14}
                strokeLinecap="round"
                strokeDasharray={`${ARC_LENGTH} ${ARC_LENGTH}`}
                strokeDashoffset={dashOffset}
              />
              <circle
                cx={indicatorPos.x}
                cy={indicatorPos.y}
                r={4.5}
                fill="var(--card)"
                stroke={level.fill}
                strokeWidth={2.5}
              />
            </>
          ) : null}

          {showScaleTicks ? (
            <g className="fill-muted-foreground text-[11px] font-medium tabular-nums">
              <text x={CX - MAIN_RADIUS} y={CY + 18} textAnchor="middle">
                0
              </text>
              <text x={CX} y={CY - MAIN_RADIUS - 22} textAnchor="middle">
                150
              </text>
              <text x={CX + MAIN_RADIUS} y={CY + 18} textAnchor="middle">
                500
              </text>
            </g>
          ) : null}
        </svg>

        <div className="pointer-events-none absolute inset-x-0 bottom-5 flex flex-col items-center">
          <span className="font-display text-[2.5rem]/none font-semibold tracking-tight text-foreground tabular-nums md:text-display">
            {displayValue}
          </span>
          <span className="mt-1 text-xs font-medium text-muted-foreground">
            US AQI
          </span>
        </div>
      </div>

      {level ? (
        <span
          className={cn(
            "inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-sm font-semibold",
            level.badge
          )}
        >
          <LevelIcon className="size-4 shrink-0" strokeWidth={1.75} />
          <span>{level.label}</span>
        </span>
      ) : (
        <span className="inline-flex h-8 items-center gap-1.5 rounded-full border border-border bg-muted px-3 text-sm font-medium text-foreground">
          <Info className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
          <span>Air quality unavailable</span>
        </span>
      )}
    </div>
  )
}
