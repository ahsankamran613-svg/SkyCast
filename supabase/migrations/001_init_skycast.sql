-- =============================================================================
-- SkyCast (AgentWeather) — Initial Supabase PostgreSQL Schema
-- Fast-serving read cache and telemetry store for Lahore weather & AQI MLOps
-- =============================================================================

-- Trigger helper to automatically update `updated_at` on row updates
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- -----------------------------------------------------------------------------
-- 1. Hourly Fused Observations & Engineered Features (14-day serving window)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.weather_features_hourly (
    timestamp_utc TIMESTAMPTZ PRIMARY KEY,

    -- Canonical fused target metrics (6 core metric groups -> 8 target series)
    us_aqi DOUBLE PRECISION,
    temperature_2m DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    precipitation_prob DOUBLE PRECISION,
    relative_humidity_2m DOUBLE PRECISION,
    uv_index DOUBLE PRECISION,
    wind_speed_10m DOUBLE PRECISION,
    wind_direction_10m DOUBLE PRECISION,

    -- Auxiliary weather & condition fields for Hero panel & exogenous modeling
    apparent_temperature DOUBLE PRECISION,
    weather_code INTEGER,
    condition_symbol TEXT,
    surface_pressure DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,

    -- Air quality pollutants (Open-Meteo Air Quality API)
    pm2_5 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    ozone DOUBLE PRECISION,
    nitrogen_dioxide DOUBLE PRECISION,
    sulphur_dioxide DOUBLE PRECISION,
    carbon_monoxide DOUBLE PRECISION,

    -- Raw source transparency & multi-source disagreement features
    temp_openmeteo DOUBLE PRECISION,
    temp_metno DOUBLE PRECISION,
    humidity_openmeteo DOUBLE PRECISION,
    humidity_metno DOUBLE PRECISION,
    precip_openmeteo DOUBLE PRECISION,
    precip_metno DOUBLE PRECISION,
    wind_speed_openmeteo DOUBLE PRECISION,
    wind_speed_metno DOUBLE PRECISION,
    wind_dir_openmeteo DOUBLE PRECISION,
    wind_dir_metno DOUBLE PRECISION,
    uv_openmeteo DOUBLE PRECISION,
    uv_metno DOUBLE PRECISION,
    source_disagreement_temp DOUBLE PRECISION,
    sources_available TEXT[] NOT NULL DEFAULT '{}'::TEXT[],

    -- 3-tier imputation audit metadata
    is_imputed BOOLEAN NOT NULL DEFAULT FALSE,
    imputation_tier TEXT NOT NULL DEFAULT 'none',
    imputed_columns JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Key cyclical & seasonal indicators
    wind_dir_sin DOUBLE PRECISION,
    wind_dir_cos DOUBLE PRECISION,
    hour_sin DOUBLE PRECISION,
    hour_cos DOUBLE PRECISION,
    is_weekend BOOLEAN,
    is_lahore_smog_season BOOLEAN,
    is_monsoon_season BOOLEAN,

    -- Full 35+ lag / rolling / rate-of-change engineered feature dictionary
    features_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_features_hourly_ts_desc
    ON public.weather_features_hourly (timestamp_utc DESC);

DROP TRIGGER IF EXISTS trg_weather_features_hourly_updated_at
    ON public.weather_features_hourly;
CREATE TRIGGER trg_weather_features_hourly_updated_at
    BEFORE UPDATE ON public.weather_features_hourly
    FOR EACH ROW
    EXECUTE FUNCTION public.set_updated_at();

-- -----------------------------------------------------------------------------
-- 2. Forecast Ledger (24h trajectories across all 8 target series)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.forecast_ledger (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL,
    issued_at_utc TIMESTAMPTZ NOT NULL,
    target_timestamp_utc TIMESTAMPTZ NOT NULL,
    horizon_hour SMALLINT NOT NULL CHECK (horizon_hour BETWEEN 1 AND 24),
    metric TEXT NOT NULL,
    model_family TEXT NOT NULL,
    model_version TEXT,
    predicted_value DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION NOT NULL,
    upper_bound DOUBLE PRECISION NOT NULL,
    naive_baseline_value DOUBLE PRECISION,
    actual_value DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_forecast_ledger_run_metric_model_horizon
        UNIQUE (run_id, metric, model_family, horizon_hour)
);

CREATE INDEX IF NOT EXISTS idx_forecast_ledger_metric_issued_desc
    ON public.forecast_ledger (metric, issued_at_utc DESC, horizon_hour ASC);
CREATE INDEX IF NOT EXISTS idx_forecast_ledger_target_ts
    ON public.forecast_ledger (target_timestamp_utc DESC, metric);
CREATE INDEX IF NOT EXISTS idx_forecast_ledger_run_id
    ON public.forecast_ledger (run_id);

-- -----------------------------------------------------------------------------
-- 3. Model Registry Leaderboard Metrics (Validation & Test RMSE, MAE, R2)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.model_registry_metrics (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    train_run_id TEXT NOT NULL,
    trained_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric TEXT NOT NULL,
    model_family TEXT NOT NULL,
    model_version TEXT,
    val_rmse DOUBLE PRECISION NOT NULL,
    val_mae DOUBLE PRECISION NOT NULL,
    val_r2 DOUBLE PRECISION NOT NULL,
    test_rmse DOUBLE PRECISION NOT NULL,
    test_mae DOUBLE PRECISION NOT NULL,
    test_r2 DOUBLE PRECISION NOT NULL,
    improvement_over_naive_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    is_champion BOOLEAN NOT NULL DEFAULT FALSE,
    promoted_to_registry BOOLEAN NOT NULL DEFAULT FALSE,
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    training_samples INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_model_registry_metrics_run_metric_family
        UNIQUE (train_run_id, metric, model_family)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_metrics_lookup
    ON public.model_registry_metrics (metric, trained_at_utc DESC, is_champion);

-- -----------------------------------------------------------------------------
-- 4. Aligned Model Family Comparison Series (Test / Backtest Curves)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.model_comparison_series (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    train_run_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    actual DOUBLE PRECISION,
    sarimax DOUBLE PRECISION,
    random_forest DOUBLE PRECISION,
    ridge DOUBLE PRECISION,
    lstm DOUBLE PRECISION,
    naive_24h DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_model_comparison_series_run_metric_ts
        UNIQUE (train_run_id, metric, timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_model_comparison_series_lookup
    ON public.model_comparison_series (metric, train_run_id, timestamp_utc ASC);

-- -----------------------------------------------------------------------------
-- 5. SHAP & LIME Explainability Payloads
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.shap_lime_explanations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    train_run_id TEXT NOT NULL,
    computed_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric TEXT NOT NULL,
    model_family TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('shap', 'lime')),
    scope TEXT NOT NULL CHECK (scope IN ('global', 'local')),
    base_value DOUBLE PRECISION,
    predicted_value DOUBLE PRECISION,
    reference_timestamp_utc TIMESTAMPTZ,
    feature_attributions JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_shap_lime_run_metric_method_scope
        UNIQUE (train_run_id, metric, method, scope)
);

CREATE INDEX IF NOT EXISTS idx_shap_lime_lookup
    ON public.shap_lime_explanations (metric, method, computed_at_utc DESC);

-- -----------------------------------------------------------------------------
-- 6. Weather Hazard Alerts (3-Audience Actionable Advisories)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.weather_alerts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_timestamp_utc TIMESTAMPTZ,
    metric TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    title TEXT NOT NULL,
    trigger_value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    advisory_general TEXT NOT NULL,
    advisory_sensitive TEXT NOT NULL,
    advisory_outdoor_workers TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    llm_provider TEXT,
    llm_model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_alerts_active_ts_desc
    ON public.weather_alerts (is_active, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_weather_alerts_run_id
    ON public.weather_alerts (run_id, metric);

-- -----------------------------------------------------------------------------
-- 7. Weather Broadcast Bulletins (Daily 06:xx PKT & Alert-Triggered)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.weather_reports (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    issued_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason TEXT NOT NULL CHECK (
        trigger_reason IN ('daily_schedule', 'hazard_alert', 'manual', 'backfill_seed')
    ),
    headline TEXT NOT NULL,
    summary TEXT NOT NULL,
    bulletin_markdown TEXT NOT NULL,
    highlights JSONB NOT NULL DEFAULT '[]'::jsonb,
    active_alert_count INTEGER NOT NULL DEFAULT 0,
    llm_provider TEXT,
    llm_model TEXT,
    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_reports_issued_desc
    ON public.weather_reports (issued_at_utc DESC);

-- -----------------------------------------------------------------------------
-- 8. Agent LLM Call & Failover Telemetry Log
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.agent_call_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    reasoning_effort TEXT,
    prompt TEXT NOT NULL,
    response TEXT,
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL CHECK (status IN ('success', 'failed')),
    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
    attempt_order SMALLINT NOT NULL DEFAULT 1,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_call_log_ts_desc
    ON public.agent_call_log (timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_agent_call_log_status_ts
    ON public.agent_call_log (status, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_agent_call_log_run_id
    ON public.agent_call_log (run_id, attempt_order ASC);

-- -----------------------------------------------------------------------------
-- 9. LangGraph Execution Trace
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.agent_graph_trace (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    step_order SMALLINT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'skipped', 'failed')),
    started_at TIMESTAMPTZ NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    branch_decision TEXT,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_agent_graph_trace_run_node_step
        UNIQUE (run_id, node_name, step_order)
);

CREATE INDEX IF NOT EXISTS idx_agent_graph_trace_run_step
    ON public.agent_graph_trace (run_id, step_order ASC);
CREATE INDEX IF NOT EXISTS idx_agent_graph_trace_started_desc
    ON public.agent_graph_trace (started_at DESC);

-- -----------------------------------------------------------------------------
-- 10. Data & Model Drift Detection Log
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.drift_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric TEXT NOT NULL,
    drift_type TEXT NOT NULL CHECK (drift_type IN ('data_drift', 'model_drift')),
    test_name TEXT,
    score DOUBLE PRECISION,
    secondary_score DOUBLE PRECISION,
    threshold DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('no_drift', 'drift_detected', 'check_failed')),
    retrain_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    action_taken TEXT NOT NULL,
    error_message TEXT,
    details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_log_ts_desc
    ON public.drift_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_drift_log_metric_type_ts
    ON public.drift_log (metric, drift_type, timestamp DESC);

-- -----------------------------------------------------------------------------
-- 11. Pre-Computed EDA Summaries (Distributions, Diurnal, Correlations, Audit)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.eda_summaries (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    computed_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sample_count INTEGER NOT NULL DEFAULT 0,
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eda_summaries_computed_desc
    ON public.eda_summaries (computed_at_utc DESC);

-- -----------------------------------------------------------------------------
-- 12. Pipeline Execution Heartbeat & Status Log
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.pipeline_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    pipeline_type TEXT NOT NULL CHECK (pipeline_type IN ('backfill', 'hourly_ingest', 'training')),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    started_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at_utc TIMESTAMPTZ,
    storage_mode TEXT NOT NULL DEFAULT 'cloud',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_desc
    ON public.pipeline_runs (started_at_utc DESC);

-- -----------------------------------------------------------------------------
-- Row Level Security (RLS)
-- FastAPI uses SUPABASE_SERVICE_ROLE_KEY (bypasses RLS).
-- Public dashboard tables expose read-only SELECT to anon/authenticated;
-- writes are strictly restricted to service_role.
-- -----------------------------------------------------------------------------
ALTER TABLE public.weather_features_hourly ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.forecast_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_registry_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_comparison_series ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.shap_lime_explanations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weather_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.weather_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_call_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_graph_trace ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.drift_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.eda_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pipeline_runs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public read weather_features_hourly" ON public.weather_features_hourly;
CREATE POLICY "Allow public read weather_features_hourly"
    ON public.weather_features_hourly FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read forecast_ledger" ON public.forecast_ledger;
CREATE POLICY "Allow public read forecast_ledger"
    ON public.forecast_ledger FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read model_registry_metrics" ON public.model_registry_metrics;
CREATE POLICY "Allow public read model_registry_metrics"
    ON public.model_registry_metrics FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read model_comparison_series" ON public.model_comparison_series;
CREATE POLICY "Allow public read model_comparison_series"
    ON public.model_comparison_series FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read shap_lime_explanations" ON public.shap_lime_explanations;
CREATE POLICY "Allow public read shap_lime_explanations"
    ON public.shap_lime_explanations FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read weather_alerts" ON public.weather_alerts;
CREATE POLICY "Allow public read weather_alerts"
    ON public.weather_alerts FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read weather_reports" ON public.weather_reports;
CREATE POLICY "Allow public read weather_reports"
    ON public.weather_reports FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read agent_graph_trace" ON public.agent_graph_trace;
CREATE POLICY "Allow public read agent_graph_trace"
    ON public.agent_graph_trace FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read drift_log" ON public.drift_log;
CREATE POLICY "Allow public read drift_log"
    ON public.drift_log FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read eda_summaries" ON public.eda_summaries;
CREATE POLICY "Allow public read eda_summaries"
    ON public.eda_summaries FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Allow public read pipeline_runs" ON public.pipeline_runs;
CREATE POLICY "Allow public read pipeline_runs"
    ON public.pipeline_runs FOR SELECT TO anon, authenticated USING (true);
