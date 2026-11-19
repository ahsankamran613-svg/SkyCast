import {
  isValidElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { BookOpen, Download, FileText, List } from "lucide-react"
import { cn } from "cn"
import Markdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"

import {
  BootstrappingState,
  EmptyState,
  ErrorState,
} from "@/components/shared/empty-state"
import { TimestampPkt } from "@/components/shared/timestamp-pkt"
import { Button } from "@/components/ui/button"
import {
  fetchTechnicalReport,
  type TechnicalReportResponse,
} from "@/lib/api"

export interface TechnicalReportTabProps {
  city?: string | null
  timeZone?: string | null
  refreshNonce?: number
  onShellRefresh?: () => void
}

interface TocHeading {
  id: string
  level: 2 | 3
  text: string
}

function stripInlineMarkdown(raw: string): string {
  return raw
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/~~([^~]+)~~/g, "$1")
    .replace(/\s+#+\s*$/, "")
    .trim()
}

function slugifyBase(text: string): string {
  const cleaned = text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
  return cleaned || "section"
}

function allocateHeadingSlug(base: string, seen: Map<string, number>): string {
  const count = seen.get(base) ?? 0
  seen.set(base, count + 1)
  return count === 0 ? base : `${base}-${count + 1}`
}

function extractTocHeadings(markdown: string): TocHeading[] {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n")
  const headings: TocHeading[] = []
  const seen = new Map<string, number>()
  let inFence = false

  for (const rawLine of lines) {
    const trimmed = rawLine.trim()
    if (/^(```|~~~)/.test(trimmed)) {
      inFence = !inFence
      continue
    }
    if (inFence) {
      continue
    }
    const match = trimmed.match(/^(#{2,3})\s+(.+)$/)
    if (!match) {
      continue
    }
    const level = match[1].length as 2 | 3
    const text = stripInlineMarkdown(match[2])
    if (!text) {
      continue
    }
    const id = allocateHeadingSlug(slugifyBase(text), seen)
    headings.push({ id, level, text })
  }

  return headings
}

function extractReactNodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node)
  }
  if (Array.isArray(node)) {
    return node.map(extractReactNodeText).join("")
  }
  if (isValidElement<{ children?: ReactNode }>(node)) {
    return extractReactNodeText(node.props.children)
  }
  return ""
}

function resolveDownloadFilename(rawName: string | null | undefined): string {
  const trimmed = rawName?.trim()
  if (!trimmed) {
    return "skycast-technical-report.md"
  }
  return trimmed.toLowerCase().endsWith(".md") ? trimmed : `${trimmed}.md`
}

export function TechnicalReportTab({
  city,
  timeZone,
  refreshNonce = 0,
  onShellRefresh,
}: TechnicalReportTabProps) {
  const [data, setData] = useState<TechnicalReportResponse | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [isFetching, setIsFetching] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [activeHeadingId, setActiveHeadingId] = useState<string>("")
  const [localRetryNonce, setLocalRetryNonce] = useState<number>(0)

  const satisfiedNonceRef = useRef<string | null>(null)

  useEffect(() => {
    const reqKey = `${refreshNonce}:${localRetryNonce}`
    if (satisfiedNonceRef.current === reqKey) {
      return
    }

    const controller = new AbortController()
    setIsFetching(true)
    setErrorMessage(null)

    void fetchTechnicalReport({ signal: controller.signal })
      .then((response) => {
        if (controller.signal.aborted) return
        satisfiedNonceRef.current = reqKey
        setData(response)
        setIsLoading(false)
        setIsFetching(false)
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return
        if (err instanceof DOMException && err.name === "AbortError") return
        setErrorMessage(
          "Couldn't load the technical report. Check your connection and refresh."
        )
        setIsLoading(false)
        setIsFetching(false)
      })

    return () => controller.abort()
  }, [refreshNonce, localRetryNonce])

  const handleRetry = useCallback(() => {
    satisfiedNonceRef.current = null
    if (onShellRefresh) {
      onShellRefresh()
    } else {
      setLocalRetryNonce((n) => n + 1)
    }
  }, [onShellRefresh])

  const markdown = data?.markdown ?? ""
  const tocHeadings = useMemo(
    () => (markdown.trim() ? extractTocHeadings(markdown) : []),
    [markdown]
  )

  useEffect(() => {
    if (tocHeadings.length === 0 || typeof IntersectionObserver === "undefined") {
      return
    }

    const elements = tocHeadings
      .map((heading) => document.getElementById(heading.id))
      .filter((el): el is HTMLElement => el !== null)

    if (elements.length === 0) {
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort(
            (a, b) => a.boundingClientRect.top - b.boundingClientRect.top
          )
        if (visible[0]?.target?.id) {
          setActiveHeadingId(visible[0].target.id)
        }
      },
      {
        rootMargin: "-80px 0px -65% 0px",
        threshold: [0, 1],
      }
    )

    for (const el of elements) {
      observer.observe(el)
    }

    return () => observer.disconnect()
  }, [tocHeadings])

  const handleJumpToHeading = useCallback((id: string) => {
    setActiveHeadingId(id)
    const target = document.getElementById(id)
    if (!target) return
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    target.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "start",
    })
  }, [])

  const handleDownloadReport = useCallback(() => {
    if (!data?.markdown) return
    const filename = resolveDownloadFilename(data.filename)
    const blob = new Blob([data.markdown], {
      type: "text/markdown;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }, [data])

  // Fresh slug tracker per render ensures deterministic StrictMode-safe IDs.
  const headingSlugs = new Map<string, number>()

  const markdownComponents: Components = {
    h1: ({ children }) => (
      <h1 className="mt-2 mb-4 scroll-mt-20 font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
        {children}
      </h1>
    ),
    h2: ({ children }) => {
      const text = extractReactNodeText(children).trim()
      const id = allocateHeadingSlug(slugifyBase(text), headingSlugs)
      return (
        <h2
          id={id}
          className="mt-8 mb-3 scroll-mt-20 border-b border-border pb-2 font-display text-[1.25rem]/6.5 font-semibold text-foreground first:mt-2 md:text-h2"
        >
          {children}
        </h2>
      )
    },
    h3: ({ children }) => {
      const text = extractReactNodeText(children).trim()
      const id = allocateHeadingSlug(slugifyBase(text), headingSlugs)
      return (
        <h3
          id={id}
          className="mt-6 mb-2 scroll-mt-20 font-sans text-base/6 font-semibold text-foreground"
        >
          {children}
        </h3>
      )
    },
    h4: ({ children }) => (
      <h4 className="mt-4 mb-2 scroll-mt-20 font-sans text-sm/5 font-semibold text-foreground">
        {children}
      </h4>
    ),
    p: ({ children }) => (
      <p className="my-3 text-base/7 text-foreground/95">{children}</p>
    ),
    ul: ({ children }) => (
      <ul className="my-3 list-disc space-y-1.5 pl-6 text-base/7 text-foreground/95">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="my-3 list-decimal space-y-1.5 pl-6 text-base/7 text-foreground/95">
        {children}
      </ol>
    ),
    li: ({ children }) => <li className="leading-7">{children}</li>,
    blockquote: ({ children }) => (
      <blockquote className="my-4 rounded-r-lg border-l-4 border-primary bg-muted px-4 py-3 text-sm/6 text-foreground">
        {children}
      </blockquote>
    ),
    hr: () => <hr className="my-6 border-border" />,
    a: ({ href, children }) => {
      const isExternal =
        typeof href === "string" &&
        (href.startsWith("http://") || href.startsWith("https://"))
      return (
        <a
          href={href}
          target={isExternal ? "_blank" : undefined}
          rel={isExternal ? "noopener noreferrer" : undefined}
          className="font-medium text-primary underline underline-offset-4 hover:opacity-80"
        >
          {children}
        </a>
      )
    },
    pre: ({ children }) => (
      <pre className="my-4 overflow-x-auto rounded-lg border border-border bg-muted p-4 font-mono text-xs/5 text-foreground [&_code]:bg-transparent [&_code]:p-0 [&_code]:text-inherit">
        {children}
      </pre>
    ),
    code: ({ className, children }) => {
      const isBlock = Boolean(className?.includes("language-"))
      if (isBlock) {
        return <code className={className}>{children}</code>
      }
      return (
        <code className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[0.875em] text-foreground">
          {children}
        </code>
      )
    },
    table: ({ children }) => (
      <div className="my-4 w-full overflow-x-auto rounded-lg border border-border">
        <table className="w-full border-collapse text-left text-sm">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
    tbody: ({ children }) => (
      <tbody className="[&_tr:last-child]:border-b-0">{children}</tbody>
    ),
    tr: ({ children }) => (
      <tr className="border-b border-border transition-colors hover:bg-muted/50">
        {children}
      </tr>
    ),
    th: ({ children }) => (
      <th className="px-3 py-2.5 text-xs font-semibold whitespace-nowrap text-foreground">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="px-3 py-2.5 text-sm text-foreground tabular-nums">
        {children}
      </td>
    ),
  }

  if (isLoading && !data) {
    return (
      <div className="mx-auto flex w-full max-w-[800px] flex-col gap-6">
        <div className="h-24 animate-pulse rounded-xl border border-border bg-muted" />
        <div className="h-[480px] animate-pulse rounded-xl border border-border bg-muted" />
      </div>
    )
  }

  if (errorMessage && !data) {
    return (
      <div className="mx-auto w-full max-w-[800px]">
        <ErrorState
          title="Couldn't load technical report"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      </div>
    )
  }

  if (
    !data ||
    (data.status === "bootstrapping" && !data.available && !markdown.trim())
  ) {
    return (
      <div className="mx-auto w-full max-w-[800px]">
        <BootstrappingState
          title="Technical report not ready yet"
          description="The technical report is not available yet. Check back shortly or refresh."
          city={city ?? undefined}
          onRefresh={handleRetry}
          isRefreshing={isFetching}
        />
      </div>
    )
  }

  const updatedIso = data.updated_at_utc ?? data.updated_at ?? null
  const readingMinutes =
    data.word_count > 0 ? Math.max(1, Math.round(data.word_count / 220)) : null

  return (
    <div className="mx-auto flex w-full max-w-[800px] flex-col gap-6">
      {errorMessage ? (
        <ErrorState
          title="Couldn't refresh technical report"
          message={errorMessage}
          onRetry={handleRetry}
          isRetrying={isFetching}
        />
      ) : null}

      {/* Report Header & Download Action */}
      <header className="flex flex-col justify-between gap-4 rounded-xl border border-border bg-card p-5 shadow-(--shadow-card) sm:flex-row sm:items-center md:p-6">
        <div className="min-w-0">
          <h2 className="font-display text-[1.625rem]/8 font-semibold text-foreground md:text-h1">
            {data.title?.trim() || "Technical report"}
          </h2>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
            {data.word_count > 0 ? (
              <span className="tabular-nums">
                {data.word_count.toLocaleString()} words
                {readingMinutes !== null ? ` (${readingMinutes} min read)` : ""}
              </span>
            ) : null}
            {updatedIso ? (
              <span>
                Updated{" "}
                <TimestampPkt
                  timestamp={updatedIso}
                  timeZone={timeZone}
                  mode="datetime"
                  className="font-medium text-foreground"
                />
              </span>
            ) : null}
          </div>
        </div>

        <Button
          type="button"
          variant="default"
          onClick={handleDownloadReport}
          disabled={!data.available || !markdown.trim()}
          className="h-11 shrink-0 gap-2 px-4 sm:h-10"
        >
          <Download className="size-4" strokeWidth={1.75} />
          <span>Download report</span>
        </Button>
      </header>

      {!data.available || !markdown.trim() ? (
        <EmptyState
          icon={FileText}
          title="Technical report unavailable"
          description={
            data.message?.trim() ||
            "The technical report markdown document has not been written yet. Generate the report and refresh."
          }
          actionLabel="Refresh data"
          onAction={handleRetry}
          isActionLoading={isFetching}
        />
      ) : (
        <div
          className={cn(
            "grid grid-cols-1 gap-6",
            tocHeadings.length > 0 && "lg:grid-cols-[minmax(0,1fr)_200px] lg:items-start"
          )}
        >
          {/* Mobile / Tablet Collapsible TOC */}
          {tocHeadings.length > 0 ? (
            <details className="rounded-xl border border-border bg-card p-4 shadow-(--shadow-card) lg:hidden">
              <summary className="flex cursor-pointer items-center justify-between text-sm font-semibold text-foreground select-none">
                <span className="inline-flex items-center gap-2">
                  <List className="size-4 text-primary" strokeWidth={1.75} />
                  <span>Table of contents ({tocHeadings.length})</span>
                </span>
              </summary>
              <nav aria-label="Report sections" className="mt-3 flex flex-col gap-1">
                {tocHeadings.map((heading) => (
                  <button
                    key={heading.id}
                    type="button"
                    onClick={() => handleJumpToHeading(heading.id)}
                    className={cn(
                      "min-h-9 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors duration-120 hover:bg-muted hover:text-foreground",
                      heading.level === 3 && "pl-5 text-xs",
                      activeHeadingId === heading.id
                        ? "bg-accent font-medium text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    {heading.text}
                  </button>
                ))}
              </nav>
            </details>
          ) : null}

          {/* Markdown Article Surface */}
          <article
            aria-label="Technical report content"
            className="min-w-0 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-(--shadow-card) md:p-7"
          >
            <Markdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {markdown}
            </Markdown>
          </article>

          {/* Sticky Desktop TOC */}
          {tocHeadings.length > 0 ? (
            <aside
              aria-label="Table of contents"
              className="hidden lg:sticky lg:top-20 lg:flex lg:max-h-[calc(100vh-6rem)] lg:flex-col lg:gap-3 lg:overflow-y-auto lg:rounded-xl lg:border lg:border-border lg:bg-card lg:p-4 lg:shadow-(--shadow-card)"
            >
              <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                <BookOpen className="size-4 text-primary" strokeWidth={1.75} />
                <span>Contents</span>
              </div>
              <nav className="flex flex-col gap-1">
                {tocHeadings.map((heading) => {
                  const isActive = activeHeadingId === heading.id
                  return (
                    <button
                      key={heading.id}
                      type="button"
                      onClick={() => handleJumpToHeading(heading.id)}
                      className={cn(
                        "rounded-md px-2 py-1.5 text-left text-xs/4.5 transition-colors duration-120 hover:bg-muted hover:text-foreground",
                        heading.level === 3 && "pl-4",
                        isActive
                          ? "bg-accent font-medium text-accent-foreground"
                          : "text-muted-foreground"
                      )}
                    >
                      {heading.text}
                    </button>
                  )
                })}
              </nav>
            </aside>
          ) : null}
        </div>
      )}
    </div>
  )
}
