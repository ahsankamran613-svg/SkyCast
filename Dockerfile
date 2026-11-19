# =============================================================================
# SkyCast (AgentWeather) — Multi-Stage Dockerfile
# Targets:
#   - api:    Lightweight FastAPI read-only serving layer (mirrors Vercel boundary)
#   - worker: Full ML + LangGraph agentic pipeline & local scheduler (CPU PyTorch)
#   - web:    Vite React production build served via Nginx with /api reverse proxy
# =============================================================================

# --- Stage 1: Lightweight FastAPI Read Layer ---
FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt "uvicorn>=0.34.0,<1.0"

# Preserve Vercel's strict import boundary: only api/, skycast/config/, and FINAL_REPORT.md
COPY api/ /app/api/
COPY skycast/__init__.py /app/skycast/__init__.py
COPY skycast/config/ /app/skycast/config/
COPY docs/ /app/docs/

RUN mkdir -p /app/.skycast_local

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]


# --- Stage 2: Full ML & Agentic Pipeline Worker / Local Scheduler ---
FROM python:3.12-slim AS worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-ml.txt /app/
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-ml.txt

COPY skycast/ /app/skycast/
COPY docs/ /app/docs/

RUN mkdir -p /app/.skycast_local

CMD ["python", "-m", "skycast.pipelines.scheduler"]


# --- Stage 3: React + Vite Production Build ---
FROM node:24-alpine AS web-build

WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./

# Empty base URL keeps API calls relative (/api/*) so Nginx proxies them to the api service
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build


# --- Stage 4: Nginx Static Frontend + /api Reverse Proxy ---
FROM nginx:1.27-alpine AS web

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-build /app/web/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:80/ || exit 1
