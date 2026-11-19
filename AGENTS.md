# SkyCast Project Rules (Components 3–7)

Always-on engineering and testing guardrails for `d:\SkyCast`. Keep edits tightly scoped to the component being built.

## 1. Zero Silent Fallbacks
- Never use `.fillna(0)` on features, metrics, or drift scores where `0` is a valid measurement. Keep `NaN` internally and either fail fast, log an explicit failure status, or pair imputation with an explicit indicator column.
- Never swallow exceptions or return fake empty/zero success states when credentials, baselines, or required inputs are missing.

## 2. Config From `Settings` Only
- Never hardcode values defined in `skycast/config/settings.py` (`Settings` / `get_settings()`): split ratios, `forecast_horizon_hours`, model hyperparameters, drift windows/thresholds, `llm_fallback_chain`, `GROQ_REASONING_EFFORT`, timeouts, `lahore_timezone`, or `daily_report_local_hour`.
- Accept `settings: Settings | None = None` and resolve `cfg = settings or get_settings()`.

## 3. Storage & SQL Schema Parity
- In `SkyCastStorage` (`skycast/data/storage.py`), every read/write method for the 10 remaining tables (`forecast_ledger`, `model_registry_metrics`, `model_comparison_series`, `shap_lime_explanations`, `weather_alerts`, `weather_reports`, `agent_call_log`, `agent_graph_trace`, `drift_log`, `eda_summaries`) must:
  - Check `if self.storage_mode == "unconfigured": raise StorageError(...)`.
  - Paginate Supabase reads with `.range()` (PostgREST 1,000-row cap).
  - Wrap SQLite operations in `with self._sqlite_connection() as conn:`.
- Enforce `supabase/migrations/001_init_skycast.sql` `CHECK` enums/bounds and `UNIQUE` constraints in Python before writing so SQLite never accepts rows Postgres rejects.

## 4. Sanitize All JSON & API Numbers
- Pass every float/dict destined for Postgres JSONB columns or FastAPI JSON responses through `sanitize_json_scalar` (`skycast/features/engineer.py`) so `NaN` and `inf` never cause serialization crashes.

## 5. Keep Things Lean (No Bloat)
- **Code & files:** Reuse existing helpers (`Settings`, `SkyCastStorage`, `sanitize_json_scalar`, `METRIC_SPECS`); avoid duplicate logic, speculative abstractions, dead code, or giant monolithic functions.
- **Dependencies:** Keep `api/` strictly within `api/requirements.txt` (never import `skycast.ml`, `torch`, `shap`, `lime`, `statsmodels`, `scikit-learn`, or `hopsworks` at top level in `api/`).
- **Context & rules:** Do not bloat `AGENTS.md` with component-specific implementation details or transient notes; keep component specifics in the task prompt.
- **Docs & summaries:** Report only verified file paths, symbols, behaviors, and test outputs actually observed.

## 6. Required Tests & Verification
- Don't rely on isolated happy-path unit tests alone. Include:
  1. **Contract-enforcing fakes** that check Postgres `CHECK` enums/bounds, `UNIQUE` keys, 1,000-row `.range()` pagination, and `json.dumps(..., allow_nan=False)`.
  2. **Non-default `Settings` tests** proving nothing is hardcoded.
  3. **Multi-run idempotency tests** (2 consecutive runs / +1h advance) verifying clean upserts.
  4. **Failure & cold-start tests** (empty DBs, missing baselines, forced API/LLM failures).
- Run Python tests inside the project venv before finishing:
  `.venv\Scripts\python.exe -m pytest -v`
