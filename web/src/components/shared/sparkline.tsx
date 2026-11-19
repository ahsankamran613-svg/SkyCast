import { useId } from "react"
import { cn } from "cn"

export interface SparklineProps {
  values: Array<number | null | undefined>
  color?: string
  ariaLabel?: string
  className?: string
  showEndDot?: boolean
}

interface Point {
  x: number
  y: number
}

const VIEW_WIDTH = 160
const VIEW_HEIGHT = 40
const PAD_X = 3
const PAD_Y = 5

export function Sparkline({
  values,
  color = "var(--primary)",
  ariaLabel,
  className,
  showEndDot = true,
}: SparklineProps) {
  const gradientId = useId().replace(/:/g, "")

  const numericValues = values.filter(
    (v): v is number => typeof v === "number" && Number.isFinite(v)
  )

  if (numericValues.length < 2 || values.length < 2) {
    return (
      <div
        className={cn("flex h-10 w-full items-center", className)}
        role={ariaLabel ? "img" : undefined}
        aria-label={ariaLabel}
        aria-hidden={ariaLabel ? undefined : true}
      >
        <svg viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`} className="h-10 w-full">
          <line
            x1={PAD_X}
            y1={VIEW_HEIGHT / 2}
            x2={VIEW_WIDTH - PAD_X}
            y2={VIEW_HEIGHT / 2}
            stroke="var(--border)"
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />
        </svg>
      </div>
    )
  }

  const minVal = Math.min(...numericValues)
  const maxVal = Math.max(...numericValues)
  const span = maxVal - minVal
  const usableW = VIEW_WIDTH - PAD_X * 2
  const usableH = VIEW_HEIGHT - PAD_Y * 2

  const segments: Point[][] = []
  let current: Point[] = []
  let lastValidPoint: Point | null = null

  for (let i = 0; i < values.length; i++) {
    const val = values[i]
    if (typeof val !== "number" || !Number.isFinite(val)) {
      if (current.length > 0) {
        segments.push(current)
        current = []
      }
      continue
    }

    const x =
      values.length === 1
        ? VIEW_WIDTH / 2
        : PAD_X + (i / (values.length - 1)) * usableW
    const y =
      span === 0
        ? VIEW_HEIGHT / 2
        : PAD_Y + (1 - (val - minVal) / span) * usableH

    const pt = { x, y }
    current.push(pt)
    lastValidPoint = pt
  }

  if (current.length > 0) {
    segments.push(current)
  }

  const linePath = segments
    .map((seg) =>
      seg
        .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
        .join(" ")
    )
    .join(" ")

  const areaPaths = segments
    .filter((seg) => seg.length >= 2)
    .map((seg) => {
      const top = seg
        .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
        .join(" ")
      const last = seg[seg.length - 1]
      const first = seg[0]
      return `${top} L ${last.x.toFixed(2)} ${VIEW_HEIGHT} L ${first.x.toFixed(2)} ${VIEW_HEIGHT} Z`
    })

  return (
    <div
      className={cn("h-10 w-full", className)}
      role={ariaLabel ? "img" : undefined}
      aria-label={ariaLabel}
      aria-hidden={ariaLabel ? undefined : true}
    >
      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        preserveAspectRatio="none"
        className="h-10 w-full overflow-visible"
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.18} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>

        {areaPaths.map((d, index) => (
          <path key={index} d={d} fill={`url(#${gradientId})`} />
        ))}

        <path
          d={linePath}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />

        {showEndDot && lastValidPoint ? (
          <circle
            cx={lastValidPoint.x}
            cy={lastValidPoint.y}
            r={2.75}
            fill={color}
          />
        ) : null}
      </svg>
    </div>
  )
}
