# SkyCast (`AgentWeather`)

**Serverless Agentic Multi-Metric Weather & Air Quality Forecasting System for Lahore, Pakistan (`31.5204° N, 74.3587° E`, `Asia/Karachi`)**

---

## System Overview

SkyCast ingests hourly weather and pollutant observations for Lahore from two zero-key APIs, fuses and imputes them causally, trains three distinct forecasting model families against a 24-hour naive persistence baseline, runs a 4-node stateful **LangGraph** agent workflow with a 3-tier cloud LLM fallback chain + deterministic safety net, and serves pre-computed insights through a lightweight **FastAPI** read layer and **React + Tailwind CSS v4 + shadcn/ui + Recharts** dashboard.

### Modeled Metrics (6 Groups / 8 Target Series)

| Group | Target Series | Unit | Lahore Hazard Thresholds |
| :--- | :--- | :--- | :--- |
| **Air Quality Index** | `us_aqi` | US AQI (0–500) | Info `101–150`, Warning `151–200`, Critical `>200` |
| **Temperature** | `temperature_2m` | °C | Warning `>40°C`, Critical `>44°C` or `<4°C` |
| **Precipitation** | `precipitation_mm`, `precipitation_prob` | mm/h, % | Warning `>7.5 mm/h` or `>80%`, Critical `>15 mm/h` |
| **Humidity** | `relative_humidity_2m` | % | Info `<15%`, Warning `>85%` when Temp `>32°C` |
| **UV Index** | `uv_index` | Index (0–20) | Info `6–7`, Warning `8–10`, Critical `>=11` |
| **Wind** | `wind_speed_10m`, `wind_direction_10m` | km/h, ° (sin/cos) | Warning `>40 km/h`, Critical `>60 km/h` |

### Architecture & Stack

- **Data Ingestion (No API Key):** Open-Meteo (Forecast, Historical Archive, Historical Forecast, Air Quality APIs) + Met.no Locationforecast 2.0 (`complete`), fused via `0.6 / 0.4` weighted consensus (with circular vector averaging for wind direction and elevation-adjusted pressure harmonization).
- **3-Tier Causal Imputation:** Tier 1 linear interpolation (`<=3h`), Tier 2 rolling 7-day causal diurnal median (`4–24h`), Tier 3 exponentially damped climatology (`>24h`), with row-level `is_imputed`, `imputation_tier`, and `imputed_columns` audit metadata.
- **ML & Explainability (`requirements-ml.txt`):** Strict chronological 70% / 15% / 15% split with 24-hour embargoes and train-only scaling; `SARIMAX` (`statsmodels`), `RandomForestRegressor` & `Ridge` (`scikit-learn`), and 2-layer CPU `PyTorch LSTM` benchmarked against `Naive24h` (`RMSE`, `MAE`, `R²`), plus `SHAP` (global & local) and `LIME` (local) attributions.
- **Stateful LangGraph Agentic Layer (`skycast/agents/`):** `Forecast Agent` → `Alert Agent` (3 audiences: General Public, Sensitive Groups, Outdoor Workers) → `Drift Agent` (PSI/KS data drift + rolling 72h RMSE model drift + GitHub Actions `workflow_dispatch`) → conditional `Weather Report Agent` (06:xx PKT daily schedule or active hazard alert).
- **LLM Fallback Chain:**
  1. Primary: **Groq API** (`openai/gpt-oss-120b`, `reasoning_effort="low"`)
  2. Secondary: **Gemini API** (`gemini-3.5-flash-lite`)
  3. Tertiary: **Groq API** (`qwen-3.8-27b`, `reasoning_effort="low"`)
  4. Final Safety Net: **Deterministic Rule-Based Template** (zero pipeline crashes during full LLM outages)
- **Dual Persistence:** **Hopsworks Serverless** (Feature Groups + Model Registry) + **Supabase PostgreSQL** (fast-serving dashboard cache), with automatic **Local SQLite Offline Mode** (`.skycast_local/skycast.db`) when cloud credentials are omitted.
- **Serving & UI:** Lightweight **FastAPI** (`api/index.py`, strictly isolated from heavy ML packages for Vercel's 500 MB limit) + **React 19 / Vite 8 / Tailwind CSS v4 / shadcn/ui / Recharts 3** 6-tab dashboard (`Overview`, `Forecasts`, `Models`, `EDA`, `Agents and drift`, `Technical report`).

---

## Quick Start (One-Command Docker Demo)

No cloud accounts or API keys are required to run the full stack locally. Open-Meteo and Met.no require zero keys, and SkyCast automatically activates Local SQLite Offline Mode when cloud credentials are left blank.

```bash
docker compose up --build
```

Services started:
- **Dashboard (`web`):** http://localhost:3000 (Nginx serving the production React SPA and proxying `/api/*` to FastAPI)
- **Backend API (`api`):** http://localhost:8000/api/health
- **Pipeline Scheduler (`scheduler`):** Automatically detects a cold start, runs a 30-day Lahore historical backfill, trains all model families + EDA + SHAP/LIME, runs the 4-node LangGraph agent workflow, and then loops hourly (`hourly_ingest`) and daily (`train`).

Optional Docker overrides (via `.env` or shell environment):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `SKYCAST_WEB_PORT` | `3000` | Host port for the React dashboard |
| `SKYCAST_API_PORT` | `8000` | Host port for FastAPI |
| `SKYCAST_DOCKER_BACKFILL_DAYS` | `30` | Days of Lahore history fetched on cold start |
| `SKYCAST_SCHEDULER_INTERVAL_SECONDS` | `3600` | Seconds between scheduled hourly runs |
| `SKYCAST_TRAIN_INTERVAL_HOURS` | `24` | Hours between scheduled retraining runs |

To stop and remove containers (keeping persisted `.skycast_local` volume):

```bash
docker compose down
```

---

## Local Development (Without Docker)

### 1. Python Environment (Python 3.12)

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-ml.txt
```

### 2. Run the End-to-End Local Pipelines

```bash
# 1. Backfill 30 days of Lahore weather & air quality observations
python -m skycast.pipelines.backfill --days 30 --offline

# 2. Compute EDA, train all 5 model families, compute SHAP/LIME, promote champions
python -m skycast.pipelines.train --offline --save-eda-json

# 3. Ingest latest hour and execute the 4-node LangGraph agent workflow
python -m skycast.pipelines.hourly_ingest --offline
```

### 3. Start Backend & Frontend Locally

```bash
# Terminal 1 — FastAPI API (port 8000)
uvicorn api.index:app --reload --port 8000

# Terminal 2 — React Vite Dev Server (port 5173, proxies /api -> 127.0.0.1:8000)
cd web
npm ci
npm run dev
```

---

## Cloud Deployment (Vercel + Supabase + Hopsworks + GitHub Actions)

1. **Supabase:** Run [supabase/migrations/001_init_skycast.sql](supabase/migrations/001_init_skycast.sql) in your Supabase SQL Editor.
2. **Environment Variables:** Copy [.env.example](.env.example) to `.env` (for local cloud runs) and add the following **GitHub Actions Repository Secrets**:
   - `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
   - `GROQ_API_KEY`, `GEMINI_API_KEY`
   - *(Optional)* `SKYCAST_GITHUB_TOKEN` (PAT with `actions:write` if you want to override the default `github.token` used by the Drift Agent to dispatch `training_pipeline.yml`)
3. **GitHub Actions Automation:**
   - [`.github/workflows/hourly_pipeline.yml`](.github/workflows/hourly_pipeline.yml) — Runs every hour at `:05 UTC` (`5 * * * *`) + `workflow_dispatch`.
   - [`.github/workflows/training_pipeline.yml`](.github/workflows/training_pipeline.yml) — Runs daily at `01:15 UTC` / `06:15 PKT` (`15 1 * * *`) + automatic `workflow_dispatch` triggered by the Drift Agent.
   - [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — Runs `pytest`, frontend lint/typecheck/build, and `docker compose config` on every push and pull request.
4. **Vercel Serverless:** Connect the repository to Vercel (`vercel.json` builds `web/dist`, deploys `api/index.py` with `docs/FINAL_REPORT.md` bundled, and excludes heavy ML packages). Set `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `SKYCAST_OFFLINE_MODE=false` in Vercel Project Environment Variables.

---

## Verification & Testing

```bash
# Full Python test suite (unit, storage contract fakes, ML, LangGraph agents, API, CI/Docker)
pytest -v

# Frontend lint, TypeScript typecheck, and production bundle
cd web && npm run lint && npm run typecheck && npm run build

# Validate Docker Compose syntax
docker compose config -q
```

---

## Documentation

- **[Technical Report (`docs/FINAL_REPORT.md`)](docs/FINAL_REPORT.md)** — Full technical report (also rendered in the dashboard's **Technical report** tab via `GET /api/report`).
- **[Implementation Plan (`implementation_plan.md`)](implementation_plan.md)** — Architecture and 7-component blueprint.
- **[Frontend Design Spec (`skycast-frontend-design.md`)](skycast-frontend-design.md)** — Sky Slate design system, typography, accessibility, and tab specifications.
