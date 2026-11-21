import { useState, type ReactNode } from "react"
import { HelpCircle, LineChart, Table2 } from "lucide-react"
import { cn } from "cn"

import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export interface ChartTableColumn<T extends Record<string, unknown>> {
  key: keyof T & string
  header: string
  align?: "left" | "right"
  render?: (row: T) => ReactNode
}

export interface ChartCardProps<T extends Record<string, unknown>> {
  title: string
  titleInfo?: string
  subtitle?: string
  ariaLabel: string
  controls?: ReactNode
  subHeader?: ReactNode
  tableColumns: ChartTableColumn<T>[]
  tableRows: T[]
  emptyTableMessage?: string
  footer?: ReactNode
  defaultView?: "chart" | "table"
  className?: string
  children: ReactNode
}

export function ChartCard<T extends Record<string, unknown>>({
  title,
  titleInfo,
  subtitle,
  ariaLabel,
  controls,
  subHeader,
  tableColumns,
  tableRows,
  emptyTableMessage = "No data points available to display.",
  footer,
  defaultView = "chart",
  className,
  children,
}: ChartCardProps<T>) {
  const [showTable, setShowTable] = useState(defaultView === "table")

  return (
    <section
      className={cn(
        "flex flex-col gap-4 rounded-xl border border-border bg-card p-4 text-card-foreground shadow-(--shadow-card) md:p-5",
        className
      )}
    >
      <div className="flex min-h-8 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <h3 className="text-base/6 font-semibold text-card-foreground">
              {title}
            </h3>
            {titleInfo ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label={`About ${title}`}
                    className="inline-flex size-5 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
                  >
                    <HelpCircle className="size-3.5" strokeWidth={1.75} />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-left text-xs leading-relaxed">
                  {titleInfo}
                </TooltipContent>
              </Tooltip>
            ) : null}
          </div>
          {subtitle ? (
            <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {controls}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant={showTable ? "secondary" : "outline"}
                size="icon"
                className="size-8 shrink-0"
                aria-label={showTable ? "Show as chart" : "Show as data table"}
                aria-pressed={showTable}
                onClick={() => setShowTable((prev) => !prev)}
              >
                {showTable ? (
                  <LineChart className="size-3.5" strokeWidth={1.75} />
                ) : (
                  <Table2 className="size-3.5" strokeWidth={1.75} />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {showTable ? "Switch to chart view" : "View underlying data table"}
            </TooltipContent>
          </Tooltip>
        </div>
      </div>

      {subHeader ? <div>{subHeader}</div> : null}

      {showTable ? (
        tableRows.length === 0 ? (
          <div className="rounded-md border border-border">
            <p className="p-6 text-center text-sm text-muted-foreground">
              {emptyTableMessage}
            </p>
          </div>
        ) : (
          <Table
            aria-label={`${title} data table`}
            containerClassName="max-h-80 overflow-auto rounded-md border border-border"
          >
            <TableHeader className="sticky top-0 z-10 bg-muted">
              <TableRow className="hover:bg-muted">
                {tableColumns.map((col) => (
                  <TableHead
                    key={col.key}
                    className={cn(
                      "h-10 px-4 py-3 text-xs font-medium text-foreground",
                      col.align === "right" && "text-right"
                    )}
                  >
                    {col.header}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {tableRows.map((row, rowIndex) => (
                <TableRow key={rowIndex} className="h-[52px] md:h-11">
                  {tableColumns.map((col) => {
                    const raw = row[col.key]
                    const rendered = col.render
                      ? col.render(row)
                      : raw === null || raw === undefined || raw === ""
                        ? "—"
                        : String(raw)
                    return (
                      <TableCell
                        key={col.key}
                        className={cn(
                          "px-4 py-3 text-sm tabular-nums",
                          col.align === "right" && "text-right"
                        )}
                      >
                        {rendered}
                      </TableCell>
                    )
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )
      ) : (
        <div role="region" aria-label={ariaLabel}>
          {children}
        </div>
      )}

      {footer ? (
        <div className="border-t border-border pt-3 text-xs text-muted-foreground">
          {footer}
        </div>
      ) : null}
    </section>
  )
}
