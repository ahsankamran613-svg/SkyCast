-- ============================================================================
-- SkyCast: Supabase pg_cron + pg_net GitHub Actions Workflow Dispatcher
-- ============================================================================
-- Replaces GitHub Actions' low-priority built-in `on.schedule` cron with
-- exact-minute `workflow_dispatch` API triggers from Supabase.
--
-- Prerequisites (run ONCE in Supabase SQL Editor before or after this script):
--   1. Store your GitHub Fine-Grained PAT (scoped to ahsankamran613-svg/SkyCast
--      with "Actions: Read and write" permission) in Supabase Vault:
--
--        SELECT vault.create_secret(
--          '<YOUR_GITHUB_FINE_GRAINED_PAT>',
--          'github_actions_pat',
--          'GitHub Fine-Grained PAT (Actions: Read and write) for SkyCast workflow_dispatch'
--        );
--
--      (Or add it via Supabase Dashboard -> Project Settings -> Vault with
--       name = 'github_actions_pat'.)
--
--   2. Execute this script in the Supabase SQL Editor. Re-running it is safe
--      and idempotent (existing jobs with the same names are replaced).
--
-- Verification queries (run in Supabase SQL Editor):
--   SELECT jobid, jobname, schedule, active FROM cron.job;
--   SELECT * FROM cron.job_run_details ORDER BY start_time DESC LIMIT 10;
--   SELECT id, status_code, error_msg, created FROM net._http_response ORDER BY created DESC LIMIT 10;
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_cron;
CREATE EXTENSION IF NOT EXISTS pg_net;

-- Keep the dispatcher function in a private schema so PostgREST never exposes
-- it as a public RPC endpoint.
CREATE SCHEMA IF NOT EXISTS private;

REVOKE ALL ON SCHEMA private FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION private.dispatch_github_workflow(
  workflow_file TEXT,
  ref_name TEXT DEFAULT 'main',
  inputs JSONB DEFAULT '{}'::JSONB
)
RETURNS BIGINT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = private, pg_temp
AS $$
DECLARE
  v_token TEXT;
  v_url TEXT;
  v_body JSONB;
  v_request_id BIGINT;
BEGIN
  SELECT decrypted_secret
    INTO v_token
    FROM vault.decrypted_secrets
   WHERE name = 'github_actions_pat'
   ORDER BY created_at DESC
   LIMIT 1;

  IF v_token IS NULL OR btrim(v_token) = '' THEN
    RAISE EXCEPTION 'Missing required Supabase Vault secret: github_actions_pat';
  END IF;

  v_url := 'https://api.github.com/repos/ahsankamran613-svg/SkyCast/actions/workflows/'
           || workflow_file || '/dispatches';

  IF inputs IS NULL OR inputs = '{}'::JSONB THEN
    v_body := jsonb_build_object('ref', ref_name);
  ELSE
    v_body := jsonb_build_object('ref', ref_name, 'inputs', inputs);
  END IF;

  SELECT net.http_post(
    url := v_url,
    headers := jsonb_build_object(
      'Accept', 'application/vnd.github+json',
      'Authorization', 'Bearer ' || v_token,
      'X-GitHub-Api-Version', '2022-11-28',
      'Content-Type', 'application/json',
      'User-Agent', 'SkyCast-Supabase-pg_cron'
    ),
    body := v_body,
    timeout_milliseconds := 10000
  ) INTO v_request_id;

  RETURN v_request_id;
END;
$$;

REVOKE ALL ON FUNCTION private.dispatch_github_workflow(TEXT, TEXT, JSONB)
  FROM PUBLIC, anon, authenticated;

-- Idempotently remove any prior SkyCast cron jobs before scheduling
DO $$
BEGIN
  PERFORM cron.unschedule(jobid)
  FROM cron.job
  WHERE jobname IN (
    'skycast-hourly-pipeline',
    'skycast-daily-training-pipeline'
  );
END;
$$;

-- Hourly ingestion & LangGraph agent pipeline:
-- 2nd minute of every hour (UTC & PKT; 01:02 UTC = 06:02 PKT triggers the daily Weather Report Agent)
SELECT cron.schedule(
  'skycast-hourly-pipeline',
  '2 * * * *',
  $$SELECT private.dispatch_github_workflow('hourly_pipeline.yml', 'main');$$
);

-- Daily ML training, EDA, SHAP/LIME, and champion promotion:
-- 01:02 UTC = 06:02 PKT (2nd minute of 6 AM Pakistan Standard Time, UTC+5)
SELECT cron.schedule(
  'skycast-daily-training-pipeline',
  '2 1 * * *',
  $$SELECT private.dispatch_github_workflow(
      'training_pipeline.yml',
      'main',
      '{"trigger_reason": "scheduled"}'::JSONB
    );$$
);
