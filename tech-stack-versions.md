# SkyCast (`AgentWeather`) — Verified Tech Stack & Compatible Versions

*Verified on 2026-11-17 via Context7 MCP documentation, PyPI/npm metadata, and [implementation_plan.md](file:///d:/SkyCast/implementation_plan.md) / [skycast-frontend-design.md](file:///d:/SkyCast/skycast-frontend-design.md).*

---

## 1. Recommended Target Runtimes

| Runtime | Recommended Target | Local Machine Detected | Compatibility Basis |
| :--- | :--- | :--- | :--- |
| **Python** | **`3.12`** (`3.12.x`) | `3.12.3` | Default on Vercel (`3.12` default; `3.13` & `3.14` also supported), required (`>=3.12`) by latest `shap`, required (`>=3.11`) by `torch` and `google-genai`, and officially supported (`3.10–3.13`) by `langgraph` and `hopsworks`. |
| **Node.js** | **`22` LTS or `24` LTS** | `v24.11.1` (`npm 11.6.2`) | Vite 8 requires `^20.19.0 \|\| >=22.12.0`; GitHub Actions `v7` targets Node 24. |
| **Docker** | `python:3.12-slim` + `node:24-alpine` (or `node:22-alpine`) | `Docker 29.6.1` | Aligns container runtimes with Vercel (`3.12`) and Vite 8 (`Node 22.12+ / 24`). |

---

## 2. Python Stack

### 2.1 Data, Machine Learning & Explainability (`requirements-ml.txt`)

| Package | Latest Observed | Compatible Pin / Range | Notes & Constraints |
| :--- | :--- | :--- | :--- |
| **`numpy`** | `2.5.5` | `>=2.0.2,<2.6` | Latest `shap` requires `numpy>=2`, `statsmodels 0.15.0` targets `numpy>=2.0`, `pandas 3.1.x` requires `>=2.0.2`, and `statsmodels` caps `<3`. |
| **`pandas`** | `3.1.1` | `>=2.2.3,<3.0` *(conservative)* or `>=3.0,<3.2` | `statsmodels 0.15.0` supports `pandas>=3.0`, **but** `pandas 2.3.x` (`<3.0`) already supports `numpy 2.x` while avoiding `pandas 3.0+` default Copy-on-Write and `StringDtype` breaking changes in downstream libraries such as `hopsworks`/`pyarrow`/`lime`. |
| **`scipy`** | `1.18.1` | `>=1.13.0,<2.0` | Meets `statsmodels 0.15.0` (`>=1.13`), `scikit-learn 1.9.1` (`>=1.10.0`), and `numpy 2.x`. |
| **`statsmodels`** | `0.15.0` | `>=0.15.0,<0.16` | Supports `numpy>=2.0`, `pandas>=3.0`, and Polars; uses `meson-python` build backend (prebuilt PyPI wheels require no build tools). |
| **`scikit-learn`** | `1.9.1` | `>=1.8.0,<2.0` | Supports Python `3.10–3.14`, `numpy>=1.24.1`, and `scipy>=1.10.0`. |
| **`torch` (CPU)** | `2.15.0` | `>=2.14.0,<2.16` | Requires `Python >= 3.11`; install CPU wheel with `--index-url https://download.pytorch.org/whl/cpu`. |
| **`shap`** | `0.54.1` | `>=0.53.0,<0.55` | `0.53.0+` migrated core utilities away from `numba`/`llvmlite` to `nanobind` C++; **requires `Python >= 3.12` and `numpy >= 2`**. |
| **`lime`** | `0.2.0.1` | `0.2.0.1` *(with caveat)* | Last PyPI release was mid-2020 and pins `pyDOE2==1.3.0` alongside `scikit-learn>=0.18`, `scikit-image>=0.12`, `scipy`, `numpy`, `tqdm>=4.29.1`, `matplotlib`. Has NumPy 2.0 / Python 3.12+ compatibility risk; test `LimeTabularExplainer` early (may require `--no-deps` + explicit dependencies if `pip` metadata conflicts). |
| **`hopsworks`** | `5.2.3` | `hopsworks[python]~=<SERVER_X.Y>` | Hopsworks 4.0+ consolidated `hopsworks`, `hsfs`, and `hsml` into a single `hopsworks` package (`hopsworks.hsfs`, `hopsworks.hsml`). Supports Python `3.10–3.13`; **match the client major.minor version to your Hopsworks Serverless deployment**. |

### 2.2 Agentic Layer, LLM SDKs, Storage & Backend API (`api/requirements.txt` + `requirements-ml.txt`)

| Package | Latest Observed | Compatible Pin / Range | Notes & Constraints |
| :--- | :--- | :--- | :--- |
| **`langgraph`** | `1.2.22` | `>=1.2.0,<2.0` | Requires `Python >= 3.10` (`3.10–3.13` classifiers), `langchain-core>=1.4.7,<2`, `langgraph-checkpoint>=4.3.0,<5.0.0`, `langgraph-sdk>=0.4.6,<0.5.0`, `langgraph-prebuilt>=1.1.0,<1.2.0`, `pydantic>=2.7.4`. |
| **`groq`** | `1.7.0` | `>=1.7.0,<2.0` | Requires `Python >= 3.10`, `httpx>=0.23.0,<1`, `pydantic>=1.9.0,<3`. Target model `openai/gpt-oss-120b` is confirmed supported on Groq. |
| **`google-genai`** | `3.3.0` | `>=3.0.0,<4.0` | Requires `Python >= 3.11`. v3 defaults to Gemini Developer API `v1` and switched default HTTP transport to `httpx2>=2.0.0,<3.0.0` (separate namespace, compatible alongside `httpx 0.28.1`; optional `[httpx]` extra available). Target fallback model: **`gemini-3.5-flash-lite`**. |
| **`supabase`** | `2.34.1` | `>=2.31.0,<3.0` | Requires `Python >= 3.9`; uses `httpx<1.0`. |
| **`fastapi`** | `0.163.1` | `>=0.128.0,<1.0` | FastAPI `0.128.0` removed `pydantic.v1` support; pair with Pydantic v2 (`>=2.7.0,<3.0.0`). |
| **`pydantic`** | `2.14.3` | `>=2.10.0,<3.0` | Satisfies FastAPI (`>=2.7.0,<3.0.0`), LangGraph (`>=2.7.4`), and Groq (`<3`). |
| **`pydantic-settings`** | `2.16.3` | `>=2.15.0,<3.0` | Configuration management with Pydantic v2 (also included in `fastapi[all]`). |
| **`httpx`** | `0.28.1` | `>=0.28.1,<1.0` | Satisfies `groq`, `supabase`, `google-genai[httpx]`, and async weather API calls. |
| **`pytest`** | `9.2.0` | `>=9.0.0,<10.0` | Pair with `pytest-httpx` (`>=0.36.0`) for HTTP mock testing. |

---

## 3. Frontend Stack (`web/package.json`)

| Package | Latest Observed | Compatible Range | Notes & Constraints |
| :--- | :--- | :--- | :--- |
| **`react` & `react-dom`** | `19.3.1` | `^19.3.0` | Supported across Vite 8, Tailwind v4, shadcn/ui, Lucide, and Recharts v3. |
| **`react-is`** | `19.3.1` | `^19.3.0` | Required by Recharts; **must match installed `react` version**. |
| **`vite`** | `8.3.8` | `^8.3.0` | Requires Node `^20.19.0 \|\| >=22.12.0`. |
| **`@vitejs/plugin-react`** | `6.x` | `^6.0.0` | Built for Vite 8 using Oxc (no Babel dependency). |
| **`tailwindcss` & `@tailwindcss/vite`** | `4.3.5` | `^4.3.0` | First-party Vite plugin; supports `@import "tailwindcss";` and `@theme inline` used in [skycast-frontend-design.md](file:///d:/SkyCast/skycast-frontend-design.md). |
| **`shadcn` CLI** | `4.21.0` | `npx shadcn@latest init -t vite` | Supports Vite + Tailwind CSS v4 + React 19 + Recharts v3 (`npx shadcn@latest add chart`). |
| **`recharts`** | `3.10.1` | `^3.10.0` | Natively supports React 19 in `peerDependencies`. Reference CSS tokens directly (`var(--chart-1)`, not `hsl(var(--chart-1))`) and give `ChartContainer` an explicit height (`h-*`, `min-h-*`, or `aspect-*`). |
| **`lucide-react`** | `1.73.0` | `^1.67.0` | Compatible with React 19. |
| **`next-themes`** | `0.4.6` | `^0.4.6` *(optional / see note)* | Peer dependencies support React 19, **but** `ThemeScript` renders a `<script>` tag in JSX that triggers a React 19 console warning; see Section 4. |

---

## 4. Platform, CI/CD & Key Compatibility Notes

1. **Vite SPA vs. `next-themes` / `next/font/google`:**
   - [skycast-frontend-design.md](file:///d:/SkyCast/skycast-frontend-design.md#L392-L425) uses Next.js examples (`next/font/google` + `next-themes`), whereas [implementation_plan.md](file:///d:/SkyCast/implementation_plan.md#L286-L299) scaffolds a Vite React SPA (`web/src/App.tsx`).
   - In Vite + React 19, load `Barlow Semi Condensed` and `Public Sans` via Google Fonts `<link>` in `web/index.html` (or `@fontsource/*`), and prefer `shadcn/ui`'s official Vite `ThemeProvider` (`ui.shadcn.com/docs/dark-mode/vite`) with a small anti-FOUC inline `<script>` in `index.html` instead of `next-themes` to avoid the React 19 `<script>` render warning.
2. **Vercel Python Serverless Bundle Split (`500 MB` limit):**
   - Vercel Python Functions enforce a **500 MB** uncompressed standard bundle limit (5 GB on Large Functions beta) and automatically read root `pyproject.toml` / `requirements.txt`.
   - Keep `torch`, `shap`, `statsmodels`, `scikit-learn`, and `hopsworks` out of root `[project].dependencies` (place them in `requirements-ml.txt` or `[project.optional-dependencies]`), keep `api/requirements.txt` minimal (`fastapi`, `pydantic-settings`, `supabase`, `httpx`), and set `excludeFiles` in `vercel.json`.
3. **GitHub Actions `v7`:**
   - Use **`actions/checkout@v7`**, **`actions/setup-python@v7`** (`python-version: '3.12'`), and **`actions/setup-node@v7`** (`node-version: '24'`), which align with Node 24 GitHub runners.
4. **External Weather APIs:**
   - **Open-Meteo:** `/v1/forecast`, `/v1/archive`, and `/v1/air-quality` (`us_aqi` and pollutants).
   - **Met.no Locationforecast 2.0:** Confirm during `met_no.py` testing whether `/compact` returns `ultraviolet_index_clear_sky` for Lahore or if `/complete` is needed for UV data.
