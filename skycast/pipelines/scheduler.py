"""Local pipeline scheduler for one-command Docker portability (`docker compose up --build`)."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

from skycast.config.settings import Settings, get_settings
from skycast.data.storage import SkyCastStorage
from skycast.pipelines.backfill import run_backfill
from skycast.pipelines.hourly_ingest import run_hourly_ingest
from skycast.pipelines.train import run_training

logger = logging.getLogger(__name__)


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw.strip())
    if value < 1:
        raise ValueError(f"Environment variable {name} must be >= 1 (got {value}).")
    return value


def bootstrap_if_needed(
    *,
    bootstrap_days: int | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    backfill_fn: Callable[..., dict[str, Any]] = run_backfill,
    train_fn: Callable[..., dict[str, Any]] = run_training,
    ingest_fn: Callable[..., dict[str, Any]] = run_hourly_ingest,
) -> dict[str, Any]:
    """Ensure cold-start storage has historical data, trained champions, and an initial forecast."""
    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)
    days = cfg.backfill_default_days if bootstrap_days is None else int(bootstrap_days)
    if days < 1:
        raise ValueError(f"bootstrap_days must be >= 1 (got {days}).")

    summary: dict[str, Any] = {
        "backfilled": False,
        "trained": False,
        "ingested": False,
        "storage_mode": store.storage_mode,
    }

    existing_features = store.read_weather_features(limit_hours=1, expand_features_json=False)
    if existing_features.empty:
        logger.info(
            "Cold start detected (no weather_features_hourly rows). Running %d-day backfill...",
            days,
        )
        backfill_res = backfill_fn(days=days, settings=cfg, storage=store)
        summary["backfilled"] = True
        summary["backfill_run_id"] = backfill_res.get("run_id")

    existing_models = store.read_model_registry_metrics(latest_run_only=True)
    if not existing_models:
        logger.info("No trained models found in storage. Running initial ML training + EDA...")
        train_res = train_fn(settings=cfg, storage=store, save_eda_doc=False)
        summary["trained"] = True
        summary["train_run_id"] = train_res.get("train_run_id")

    existing_forecasts = store.read_forecast_ledger(latest_run_only=True)
    if not existing_forecasts:
        logger.info("No forecast ledger entries found. Running initial hourly ingest + agents...")
        ingest_res = ingest_fn(settings=cfg, storage=store, run_agents=True)
        summary["ingested"] = True
        summary["ingest_run_id"] = ingest_res.get("run_id")

    return summary


def run_scheduler_loop(
    *,
    bootstrap_days: int | None = None,
    interval_seconds: int = 3600,
    train_interval_hours: int = 24,
    skip_bootstrap: bool = False,
    once: bool = False,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    stop_event: threading.Event | None = None,
    backfill_fn: Callable[..., dict[str, Any]] = run_backfill,
    train_fn: Callable[..., dict[str, Any]] = run_training,
    ingest_fn: Callable[..., dict[str, Any]] = run_hourly_ingest,
) -> dict[str, Any]:
    """Run cold-start bootstrap and then schedule hourly ingestion and daily retraining."""
    if interval_seconds < 1:
        raise ValueError(f"interval_seconds must be >= 1 (got {interval_seconds}).")
    if train_interval_hours < 1:
        raise ValueError(f"train_interval_hours must be >= 1 (got {train_interval_hours}).")

    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)
    stopper = stop_event or threading.Event()

    result: dict[str, Any] = {
        "bootstrap": None,
        "ingest_runs": 0,
        "train_runs": 0,
        "errors": [],
    }

    last_train_monotonic = time.monotonic()

    if not skip_bootstrap:
        try:
            boot_summary = bootstrap_if_needed(
                bootstrap_days=bootstrap_days,
                settings=cfg,
                storage=store,
                backfill_fn=backfill_fn,
                train_fn=train_fn,
                ingest_fn=ingest_fn,
            )
            result["bootstrap"] = boot_summary
            if boot_summary.get("trained"):
                result["train_runs"] += 1
                last_train_monotonic = time.monotonic()
            if boot_summary.get("ingested"):
                result["ingest_runs"] += 1
        except Exception as exc:
            logger.exception("Bootstrap failed: %s", exc)
            result["errors"].append({"stage": "bootstrap", "error": str(exc)})
            if once:
                raise

    if once:
        if result["ingest_runs"] == 0:
            ingest_fn(settings=cfg, storage=store, run_agents=True)
            result["ingest_runs"] += 1
        return result

    train_interval_seconds = float(train_interval_hours * 3600)
    logger.info(
        "Scheduler loop active: hourly_ingest every %ds, training every %dh (mode=%s).",
        interval_seconds,
        train_interval_hours,
        store.storage_mode,
    )

    while not stopper.wait(timeout=float(interval_seconds)):
        try:
            ingest_fn(settings=cfg, storage=store, run_agents=True)
            result["ingest_runs"] += 1
        except Exception as exc:
            logger.exception("Scheduled hourly_ingest failed: %s", exc)
            result["errors"].append({"stage": "hourly_ingest", "error": str(exc)})

        elapsed_since_train = time.monotonic() - last_train_monotonic
        if elapsed_since_train >= train_interval_seconds:
            try:
                train_fn(settings=cfg, storage=store, save_eda_doc=False)
                result["train_runs"] += 1
                last_train_monotonic = time.monotonic()
            except Exception as exc:
                logger.exception("Scheduled training run failed: %s", exc)
                result["errors"].append({"stage": "training", "error": str(exc)})

    logger.info("Scheduler shutdown signal received; exiting cleanly.")
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_settings()

    default_bootstrap_days = _parse_positive_int_env("SKYCAST_DOCKER_BACKFILL_DAYS", 30)
    default_interval = _parse_positive_int_env("SKYCAST_SCHEDULER_INTERVAL_SECONDS", 3600)
    default_train_hours = _parse_positive_int_env("SKYCAST_TRAIN_INTERVAL_HOURS", 24)

    parser = argparse.ArgumentParser(
        description="Run SkyCast local bootstrap and recurring pipeline scheduler."
    )
    parser.add_argument(
        "--bootstrap-days",
        type=int,
        default=default_bootstrap_days,
        help="Historical backfill days on cold start (default: 30).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=default_interval,
        help="Seconds between hourly ingestion runs (default: 3600).",
    )
    parser.add_argument(
        "--train-interval-hours",
        type=int,
        default=default_train_hours,
        help="Hours between scheduled ML retraining runs (default: 24).",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip initial cold-start backfill/training check.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run cold-start bootstrap (if needed) and a single hourly ingest, then exit.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local SQLite offline mode regardless of cloud env credentials.",
    )
    args = parser.parse_args(argv)

    if args.offline:
        cfg = cfg.model_copy(update={"skycast_offline_mode": "true"})

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s; stopping scheduler...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    run_scheduler_loop(
        bootstrap_days=args.bootstrap_days,
        interval_seconds=args.interval_seconds,
        train_interval_hours=args.train_interval_hours,
        skip_bootstrap=args.skip_bootstrap,
        once=args.once,
        settings=cfg,
        stop_event=stop_event,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
