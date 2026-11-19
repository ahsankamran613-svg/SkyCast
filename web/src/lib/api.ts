/**
 * Typed read-only REST client for the SkyCast FastAPI endpoints (`api/index.py`).
 */

export type ApiStatus = "ok" | "bootstrapping"

export type StorageMode =
  | "local_offline"
  | "cloud_dual"
  | "cloud_supabase"
  | "cloud_hopsworks"
  | "unconfigured"

export type PipelineType = "backfill" | "hourly_ingest" | "training"
export type PipelineStatus = "running" | "completed" | "failed"
export type AlertSeverity = "info" | "warning" | "critical"
export type ReportTriggerReason =
  | "daily_schedule"
  | "hazard_alert"
  | "manual"
  | "backfill_seed"
export type AgentCallStatus = "success" | "failed"
export type GraphStepStatus = "completed" | "skipped" | "failed"
export type DriftType = "data_drift" | "model_drift"
export type DriftStatus = "no_drift" | "drift_detected" | "check_failed"
export type ExplanationMethod = "shap" | "lime"
export type ExplanationScope = "global" | "local"

export interface Coordinates {
  lat: number
  lon: number
}

export interface MetricSpec {
  name: string
  group: string
  label: string
  unit: string
  chart_token: string
  min_value: number | null
  max_value: number | null
  is_circular: boolean
}

export type MetricSpecsMap = Record<string, MetricSpec>

export interface PipelineRun {
  id: number
  run_id: string
  pipeline_type: PipelineType
  status: PipelineStatus
  started_at_utc: string
  completed_at_utc: string | null
  storage_mode: string
  metadata_json: Record<string, unknown>
  error_message: string | null
  created_at: string
}

export interface HealthResponse {
  status: ApiStatus
  city: string
  timezone: string
  coordinates: Coordinates
  storage_mode: StorageMode | string
  last_ingestion_timestamp: string | null
  last_pipeline_run: PipelineRun | null
  pipelines: {
    hourly_ingest: PipelineRun | null
    training: PipelineRun | null
    backfill: PipelineRun | null
  }
  checked_at_utc: string
}

export interface WeatherObservation {
  timestamp_utc: string
  us_aqi: number | null
  temperature_2m: number | null
  precipitation_mm: number | null
  precipitation_prob: number | null
  relative_humidity_2m: number | null
  uv_index: number | null
  wind_speed_10m: number | null
  wind_direction_10m: number | null
  apparent_temperature: number | null
  weather_code: number | null
  condition_symbol: string | null
  surface_pressure: number | null
  cloud_cover: number | null
  pm2_5: number | null
  pm10: number | null
  ozone: number | null
  nitrogen_dioxide: number | null
  sulphur_dioxide: number | null
  carbon_monoxide: number | null
  source_disagreement_temp: number | null
  sources_available: string[]
  is_imputed: boolean
  imputation_tier: string
  imputed_columns: string[]
  wind_dir_sin: number | null
  wind_dir_cos: number | null
  hour_sin: number | null
  hour_cos: number | null
  is_weekend: boolean | null
  is_lahore_smog_season: boolean | null
  is_monsoon_season: boolean | null
  created_at: string
  updated_at: string
}

export interface ForecastPoint {
  id: number
  run_id: string
  issued_at_utc: string
  target_timestamp_utc: string
  horizon_hour: number
  metric: string
  model_family: string
  model_version: string | null
  predicted_value: number
  lower_bound: number
  upper_bound: number
  naive_baseline_value: number | null
  actual_value: number | null
  created_at: string
}

export interface WeatherReport {
  id: number
  run_id: string
  issued_at_utc: string
  trigger_reason: ReportTriggerReason
  headline: string
  summary: string
  bulletin_markdown: string
  highlights: unknown[]
  active_alert_count: number
  llm_provider: string | null
  llm_model: string | null
  fallback_used: boolean
  created_at: string
}

export interface WeatherAlert {
  id: number
  run_id: string
  timestamp: string
  target_timestamp_utc: string | null
  metric: string
  severity: AlertSeverity
  title: string
  trigger_value: number
  threshold: number
  advisory_general: string
  advisory_sensitive: string
  advisory_outdoor_workers: string
  is_active: boolean
  llm_provider: string | null
  llm_model: string | null
  created_at: string
}

export interface DashboardOverviewResponse {
  status: ApiStatus
  city: string
  timezone: string
  coordinates: Coordinates
  storage_mode: StorageMode | string
  updated_at_utc: string | null
  forecast_horizon_hours: number
  current_conditions: WeatherObservation | null
  observed_24h: WeatherObservation[]
  forecasts_24h: Record<string, ForecastPoint[]>
  latest_report: WeatherReport | null
  active_alerts: WeatherAlert[]
  pollutant_columns: string[]
  core_metric_groups: Record<string, string[]>
  metric_specs: MetricSpecsMap
}

export interface ForecastsHistoryParams {
  metric?: string
  model_family?: string
  horizon_hour?: number
  run_id?: string
  observed_hours?: number
  limit?: number
}

export interface ForecastsHistoryResponse {
  status: ApiStatus
  city: string
  timezone: string
  metric: string | null
  model_family: string | null
  horizon_hour: number | null
  forecast_horizon_hours: number
  latest_run_id: string | null
  latest_forecast: Record<string, ForecastPoint[]>
  observed_history: WeatherObservation[]
  history: ForecastPoint[]
  total_count: number
  metric_specs: MetricSpecsMap
}

export interface ModelsLeaderboardParams {
  metric?: string
  train_run_id?: string
  latest_run_only?: boolean
}

export interface ModelRegistryMetricRow {
  id: number
  train_run_id: string
  trained_at_utc: string
  metric: string
  model_family: string
  model_version: string | null
  val_rmse: number
  val_mae: number
  val_r2: number
  test_rmse: number
  test_mae: number
  test_r2: number
  improvement_over_naive_pct: number
  is_champion: boolean
  promoted_to_registry: boolean
  hyperparameters: Record<string, unknown>
  training_samples: number | null
  created_at: string
}

export interface ModelComparisonPoint {
  id: number
  train_run_id: string
  metric: string
  timestamp_utc: string
  actual: number | null
  sarimax: number | null
  random_forest: number | null
  ridge: number | null
  lstm: number | null
  naive_24h: number | null
  created_at: string
}

export interface ShapGlobalAttribution {
  feature: string
  importance: number | null
  mean_abs_shap: number | null
  mean_shap: number | null
  value: number | null
}

export interface ShapLocalAttribution {
  feature: string
  importance: number | null
  shap_value: number | null
  contribution: number | null
  value: number | null
  scaled_value: number | null
}

export interface LimeLocalAttribution {
  feature: string
  importance: number | null
  weight: number | null
  contribution: number | null
  rule: string
  value: number | null
  scaled_value: number | null
}

export type FeatureAttribution =
  | ShapGlobalAttribution
  | ShapLocalAttribution
  | LimeLocalAttribution

export interface ExplanationRecord<
  TAttr extends FeatureAttribution = FeatureAttribution,
> {
  id: number
  train_run_id: string
  computed_at_utc: string
  metric: string
  model_family: string
  method: ExplanationMethod
  scope: ExplanationScope
  base_value: number | null
  predicted_value: number | null
  reference_timestamp_utc: string | null
  feature_attributions: TAttr[]
  created_at: string
}

export interface ModelsLeaderboardResponse {
  status: ApiStatus
  city: string
  timezone: string
  train_run_id: string | null
  trained_at_utc: string | null
  metric: string | null
  model_families: string[]
  leaderboard: ModelRegistryMetricRow[]
  champions: Record<string, ModelRegistryMetricRow | null>
  comparison_series: ModelComparisonPoint[]
  explanations: {
    shap: {
      global: ExplanationRecord<ShapGlobalAttribution>[]
      local: ExplanationRecord<ShapLocalAttribution>[]
    }
    lime: {
      local: ExplanationRecord<LimeLocalAttribution>[]
    }
  }
  metric_specs: MetricSpecsMap
}

export interface EdaSummaryParams {
  run_id?: string
}

export interface EdaMetadata {
  city: string
  timezone: string
  sample_count: number
  date_range_start?: string | null
  date_range_end?: string | null
  seasonal_period?: number
}

export interface EdaHistogramBin {
  bin_start: number
  bin_end: number
  count: number
}

export interface EdaDistributionEntry {
  present: boolean
  status: "available" | "missing_column" | "all_missing"
  label: string
  unit: string
  valid_count: number
  missing_count: number
  mean: number | null
  std: number | null
  min: number | null
  max: number | null
  q05: number | null
  q25: number | null
  q50: number | null
  q75: number | null
  q95: number | null
  histogram: EdaHistogramBin[]
  circular_mean_deg?: number | null
}

export interface EdaDiurnalBucket {
  hour: number
  mean: number | null
  median: number | null
  std: number | null
  count: number
}

export interface EdaDiurnalHeatmap {
  timezone: string
  hours: number[]
  metrics: Record<string, EdaDiurnalBucket[]>
}

export interface EdaCorrelationMatrix {
  columns: string[]
  method: string
  min_periods: number
  matrix: Record<string, Record<string, number | null>>
  constant_columns: string[]
  highlights: Record<string, number | null>
}

export interface EdaSeasonalCyclePoint {
  hour: number
  seasonal: number | null
}

export interface EdaSeasonalDecompositionAvailable {
  status: "available"
  period: number
  sample_count_used: number
  seasonal_cycle_24h: EdaSeasonalCyclePoint[]
  summary: {
    observed_mean: number | null
    trend_start: number | null
    trend_end: number | null
    seasonal_amplitude: number | null
    residual_std: number | null
  }
  timestamps_utc: string[]
  observed: Array<number | null>
  trend: Array<number | null>
  seasonal: Array<number | null>
  resid: Array<number | null>
}

export interface EdaSeasonalDecompositionUnavailable {
  status: "unavailable"
  reason: string
  required_rows: number
  valid_count: number
  longest_contiguous_count?: number
  period: number
}

export type EdaSeasonalDecompositionEntry =
  | EdaSeasonalDecompositionAvailable
  | EdaSeasonalDecompositionUnavailable

export interface EdaImputationAudit {
  status: "available" | "unavailable"
  source: "provided" | "dataframe_derived" | "bootstrapping" | "unavailable" | string
  reason?: string
  total_rows?: number
  imputation_tier_present?: boolean
  tier_counts?: Record<string, number>
  imputed_rows_count?: number
  observed_missing_counts?: Record<string, number>
  [key: string]: unknown
}

export interface EdaSummaryResponse {
  status: ApiStatus
  run_id: string | null
  computed_at_utc: string | null
  sample_count: number
  date_range_start: string | null
  date_range_end: string | null
  metadata: EdaMetadata
  distributions: Record<string, EdaDistributionEntry>
  diurnal_heatmap: EdaDiurnalHeatmap
  correlation_matrix: EdaCorrelationMatrix
  seasonal_decomposition: Record<string, EdaSeasonalDecompositionEntry>
  imputation_audit: EdaImputationAudit
}

export interface AgentsLogsParams {
  run_id?: string
  agent_name?: string
  provider?: string
  status?: string
  drift_status?: DriftStatus
  drift_type?: DriftType
  metric?: string
  date?: string
  limit?: number
}

export interface LlmFallbackStep {
  provider: string
  model: string
  reasoning_effort?: string | null
  [key: string]: unknown
}

export interface AgentCallLogRow {
  id: number
  run_id: string
  timestamp_utc: string
  agent_name: string
  provider: string
  model: string
  reasoning_effort: string | null
  prompt: string
  response: string | null
  latency_ms: number
  status: AgentCallStatus
  fallback_used: boolean
  attempt_order: number
  error_message: string | null
  created_at: string
}

export interface DriftLogRow {
  id: number
  run_id: string | null
  timestamp: string
  metric: string
  drift_type: DriftType
  test_name: string | null
  score: number | null
  secondary_score: number | null
  threshold: number
  status: DriftStatus
  retrain_triggered: boolean
  action_taken: string
  error_message: string | null
  details_json: Record<string, unknown>
  created_at: string
}

export interface AgentGraphTraceRow {
  id: number
  run_id: string
  node_name: string
  step_order: number
  status: GraphStepStatus
  started_at: string
  duration_ms: number
  branch_decision: string | null
  summary_json: Record<string, unknown>
  error_message: string | null
  created_at: string
}

export interface AgentsLogsSummary {
  total_calls: number
  failed_calls: number
  fallback_calls: number
  drift_detected_count: number
  drift_check_failed_count: number
  trace_steps: number
  latest_run_id: string | null
}

export interface AgentsLogsResponse {
  status: ApiStatus
  city: string
  timezone: string
  daily_report_local_hour: number
  llm_fallback_chain: LlmFallbackStep[]
  call_logs: AgentCallLogRow[]
  failed_calls: AgentCallLogRow[]
  drift_logs: DriftLogRow[]
  graph_trace: AgentGraphTraceRow[]
  summary: AgentsLogsSummary
}

export interface TechnicalReportResponse {
  status: ApiStatus
  title: string
  filename: string
  path: string
  available: boolean
  markdown: string
  updated_at: string | null
  updated_at_utc: string | null
  word_count: number
  message: string | null
}

export interface RequestOptions {
  signal?: AbortSignal
}

export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  readonly url: string

  constructor(status: number, detail: string, url: string) {
    super(detail)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
    this.url = url
  }
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "")

function buildUrl(
  path: string,
  params?: Record<string, string | number | boolean | undefined | null>
): string {
  const search = new URLSearchParams()
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") {
        search.set(key, String(value))
      }
    }
  }
  const qs = search.toString()
  return `${API_BASE}${path}${qs ? `?${qs}` : ""}`
}

async function getJson<T>(url: string, options?: RequestOptions): Promise<T> {
  const response = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
    signal: options?.signal,
  })

  if (!response.ok) {
    let detail = response.statusText || `Request failed (${response.status})`
    try {
      const errBody = (await response.json()) as {
        detail?: unknown
        message?: unknown
      }
      if (typeof errBody?.detail === "string" && errBody.detail.trim()) {
        detail = errBody.detail
      } else if (
        typeof errBody?.message === "string" &&
        errBody.message.trim()
      ) {
        detail = errBody.message
      }
    } catch {
      // Response body was not JSON; preserve statusText fallback.
    }
    throw new ApiError(response.status, detail, url)
  }

  return (await response.json()) as T
}

export function fetchHealth(options?: RequestOptions): Promise<HealthResponse> {
  return getJson<HealthResponse>(buildUrl("/api/health"), options)
}

export function fetchDashboardOverview(
  options?: RequestOptions
): Promise<DashboardOverviewResponse> {
  return getJson<DashboardOverviewResponse>(
    buildUrl("/api/dashboard/overview"),
    options
  )
}

export function fetchForecastsHistory(
  params?: ForecastsHistoryParams,
  options?: RequestOptions
): Promise<ForecastsHistoryResponse> {
  return getJson<ForecastsHistoryResponse>(
    buildUrl(
      "/api/forecasts/history",
      params as Record<string, string | number | boolean | undefined>
    ),
    options
  )
}

export function fetchModelsLeaderboard(
  params?: ModelsLeaderboardParams,
  options?: RequestOptions
): Promise<ModelsLeaderboardResponse> {
  return getJson<ModelsLeaderboardResponse>(
    buildUrl(
      "/api/models/leaderboard",
      params as Record<string, string | number | boolean | undefined>
    ),
    options
  )
}

export function fetchEdaSummary(
  params?: EdaSummaryParams,
  options?: RequestOptions
): Promise<EdaSummaryResponse> {
  return getJson<EdaSummaryResponse>(
    buildUrl(
      "/api/eda/summary",
      params as Record<string, string | number | boolean | undefined>
    ),
    options
  )
}

export function fetchAgentsLogs(
  params?: AgentsLogsParams,
  options?: RequestOptions
): Promise<AgentsLogsResponse> {
  return getJson<AgentsLogsResponse>(
    buildUrl(
      "/api/agents/logs",
      params as Record<string, string | number | boolean | undefined>
    ),
    options
  )
}

export function fetchTechnicalReport(
  options?: RequestOptions
): Promise<TechnicalReportResponse> {
  return getJson<TechnicalReportResponse>(buildUrl("/api/report"), options)
}
