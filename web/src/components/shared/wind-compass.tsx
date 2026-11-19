import { cn } from "cn"

import {
  formatMetricValue,
  normalizeDegrees,
  windDirectionLabel,
} from "@/lib/weather"

export interface WindCompassProps {
  speed: number | null | undefined
  directionDeg: number | null | undefined
  unit?: string
  className?: string
}

const COMPASS_TICKS = [0, 45, 90, 135, 180, 225, 270, 315] as const

export function WindCompass({
  speed,
  directionDeg,
  unit = "km/h",
  className,
}: WindCompassProps) {
  const normDeg = normalizeDegrees(directionDeg)
  const cardinal = windDirectionLabel(directionDeg)
  const formattedSpeed = formatMetricValue(speed, 1)

  const ariaLabel =
    normDeg !== null
      ? `Wind ${formattedSpeed} ${unit} from ${cardinal} (${Math.round(normDeg)} degrees)`
      : `Wind ${formattedSpeed} ${unit}, direction unavailable`

  return (
    <div
      className={cn("flex items-center gap-4", className)}
      role="img"
      aria-label={ariaLabel}
    >
      <svg
        viewBox="0 0 120 120"
        className="size-24 shrink-0 md:size-[120px]"
        aria-hidden="true"
      >
        {/* Outer dial */}
        <circle
          cx={60}
          cy={60}
          r={52}
          fill="var(--card)"
          stroke="var(--border)"
          strokeWidth={1.5}
        />
        <circle
          cx={60}
          cy={60}
          r={38}
          fill="none"
          stroke="var(--border)"
          strokeWidth={1}
          strokeDasharray="2 4"
        />

        {/* 8-point dial ticks */}
        {COMPASS_TICKS.map((angle) => {
          const isCardinal = angle % 90 === 0
          return (
            <line
              key={angle}
              x1={60}
              y1={isCardinal ? 9 : 11}
              x2={60}
              y2={16}
              stroke={isCardinal ? "var(--muted-foreground)" : "var(--border)"}
              strokeWidth={isCardinal ? 1.5 : 1}
              transform={`rotate(${angle} 60 60)`}
            />
          )
        })}

        {/* Cardinal letters */}
        <g className="font-sans text-[10px] font-semibold">
          <text
            x={60}
            y={26}
            textAnchor="middle"
            fill="var(--primary)"
          >
            N
          </text>
          <text
            x={96}
            y={63.5}
            textAnchor="middle"
            fill="var(--muted-foreground)"
          >
            E
          </text>
          <text
            x={60}
            y={101}
            textAnchor="middle"
            fill="var(--muted-foreground)"
          >
            S
          </text>
          <text
            x={24}
            y={63.5}
            textAnchor="middle"
            fill="var(--muted-foreground)"
          >
            W
          </text>
        </g>

        {/* Wind-from directional pointer */}
        {normDeg !== null ? (
          <g transform={`rotate(${normDeg} 60 60)`}>
            <line
              x1={60}
              y1={84}
              x2={60}
              y2={30}
              stroke="var(--chart-6)"
              strokeWidth={2.5}
              strokeLinecap="round"
            />
            <polygon
              points="60,20 53,36 60,32 67,36"
              fill="var(--primary)"
            />
            <circle cx={60} cy={86} r={3} fill="var(--chart-6)" />
          </g>
        ) : null}

        <circle
          cx={60}
          cy={60}
          r={4}
          fill="var(--card)"
          stroke="var(--foreground)"
          strokeWidth={1.75}
        />
      </svg>

      <div className="min-w-0">
        <div className="flex items-baseline gap-1.5">
          <span className="font-display text-[1.75rem]/8 font-semibold tracking-tight text-foreground tabular-nums md:text-stat">
            {formattedSpeed}
          </span>
          <span className="text-sm font-medium text-muted-foreground">
            {unit}
          </span>
        </div>
        <p className="mt-1 text-sm text-muted-foreground tabular-nums">
          {normDeg !== null
            ? `From ${cardinal} (${Math.round(normDeg)}°)`
            : "Direction unavailable"}
        </p>
      </div>
    </div>
  )
}
