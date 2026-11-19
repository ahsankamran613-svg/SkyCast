"""Data drift (PSI/KS-test), forecast actual backfill, model drift, and retrain dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from skycast.config.settings import METRIC_SPECS, Settings, TARGET_METRICS, get_settings
from skycast.data.storage import SkyCastStorage
from skycast.features.engineer import sanitize_json_scalar


def _to_utc_hour_ts(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return ts.floor("h")


def backfill_forecast_ledger_actuals(
    *,
    now_ts: pd.Timestamp,
    storage: SkyCastStorage,
) -> tuple[list[dict[str, Any]], int]:
    """Join past `forecast_ledger.target_timestamp_utc` against observed `weather_features_hourly`."""
    ledger_rows = storage.read_forecast_ledger(end_target_utc=now_ts.to_pydatetime())
    if not ledger_rows:
        return [], 0

    pending_indices = [
        idx
        for idx, row in enumerate(ledger_rows)
        if row.get("actual_value") is None
        and _to_utc_hour_ts(row["target_timestamp_utc"]) <= now_ts
    ]
    if not pending_indices:
        return ledger_rows, 0

    target_timestamps = [_to_utc_hour_ts(ledger_rows[i]["target_timestamp_utc"]) for i in pending_indices]
    min_target = min(target_timestamps)
    max_target = max(target_timestamps)

    obs_df = storage.read_weather_features(
        start_utc=min_target.to_pydatetime(),
        end_utc=max_target.to_pydatetime(),
        expand_features_json=False,
    )
    if obs_df.empty:
        return ledger_rows, 0

    obs_df = obs_df.copy()
    obs_df["timestamp_utc"] = pd.to_datetime(obs_df["timestamp_utc"], utc=True).dt.floor("h")
    obs_df = obs_df.drop_duplicates(subset=["timestamp_utc"], keep="last")

    obs_lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for row in obs_df.to_dict(orient="records"):
        ts_hour = _to_utc_hour_ts(row["timestamp_utc"])
        for metric in TARGET_METRICS:
            raw_val = row.get(metric)
            if raw_val is None:
                continue
            try:
                num_val = float(raw_val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(num_val):
                obs_lookup[(ts_hour, metric)] = float(sanitize_json_scalar(num_val))

    if not obs_lookup:
        return ledger_rows, 0

    rows_to_upsert: list[dict[str, Any]] = []
    for idx in pending_indices:
        rec = ledger_rows[idx]
        ts_hour = _to_utc_hour_ts(rec["target_timestamp_utc"])
        metric = str(rec["metric"])
        observed = obs_lookup.get((ts_hour, metric))
        if observed is None:
            continue
        updated = dict(rec)
        updated["actual_value"] = observed
        ledger_rows[idx] = updated
        rows_to_upsert.append(updated)

    if rows_to_upsert:
        storage.upsert_forecast_ledger(rows_to_upsert)

    return ledger_rows, len(rows_to_upsert)


def _compute_psi_and_ks_from_samples(
    baseline_values: np.ndarray,
    live_values: np.ndarray,
) -> tuple[float, float, float, int]:
    """Compute quantile-binned clipped PSI (`np.clip(p, 1e-6, None)`) and 2-sample KS test."""
    base_arr = np.asarray(baseline_values, dtype=np.float64)
    live_arr = np.asarray(live_values, dtype=np.float64)
    if len(base_arr) == 0 or len(live_arr) == 0:
        raise ValueError("Baseline and live arrays must be non-empty.")

    quantiles = np.unique(np.quantile(base_arr, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(quantiles) == 1:
        val = float(quantiles[0])
        edges = np.array([-np.inf, val - 1e-6, val + 1e-6, np.inf], dtype=np.float64)
    elif len(quantiles) == 2:
        edges = np.array([-np.inf, float(quantiles[0]) + 1e-6, np.inf], dtype=np.float64)
    else:
        edges = np.array([-np.inf, *quantiles[1:-1], np.inf], dtype=np.float64)

    base_counts, _ = np.histogram(base_arr, bins=edges)
    live_counts, _ = np.histogram(live_arr, bins=edges)
    expected_p = np.clip(base_counts.astype(np.float64) / float(len(base_arr)), 1e-6, None)
    actual_p = np.clip(live_counts.astype(np.float64) / float(len(live_arr)), 1e-6, None)
    psi = float(np.sum((actual_p - expected_p) * np.log(actual_p / expected_p)))
    ks_result = ks_2samp(base_arr, live_arr)
    return psi, float(ks_result.statistic), float(ks_result.pvalue), int(len(base_arr))


def compute_psi_and_ks_from_histogram(
    histogram: Sequence[Mapping[str, Any]],
    live_values: np.ndarray,
    *,
    dist_min: float | None = None,
    dist_max: float | None = None,
    q25: float | None = None,
    q50: float | None = None,
    q75: float | None = None,
) -> tuple[float, float, float, int]:
    """Compute clipped PSI (`np.clip(p, 1e-6, None)`) and 2-sample KS test vs EDA histogram."""
    if not histogram:
        raise ValueError("Baseline histogram is empty.")

    base_total = sum(int(b["count"]) for b in histogram)
    if base_total <= 0:
        raise ValueError("Baseline histogram has zero total count.")

    sample_chunks: list[np.ndarray] = []
    scale_factor = 1.0 if base_total <= 20000 else (20000.0 / float(base_total))
    is_constant_baseline = (
        len(histogram) == 1
        and dist_min is not None
        and dist_max is not None
        and float(dist_min) == float(dist_max)
    )

    if is_constant_baseline:
        cnt = max(1, int(round(base_total * scale_factor)))
        sample_chunks.append(np.full(cnt, float(dist_min), dtype=np.float64))
    else:
        spike_frac = 0.0
        if dist_min is not None:
            for frac, qv in ((0.75, q75), (0.50, q50), (0.25, q25)):
                if qv is not None and float(qv) == float(dist_min):
                    spike_frac = frac
                    break
        spike_rem = int(round(base_total * spike_frac * scale_factor)) if spike_frac > 0.0 else 0
        if spike_rem > 0 and dist_min is not None:
            sample_chunks.append(np.full(spike_rem, float(dist_min), dtype=np.float64))

        for bucket in histogram:
            raw_cnt = int(bucket["count"])
            if raw_cnt <= 0:
                continue
            cnt = max(1, int(round(raw_cnt * scale_factor)))
            take_spike = min(cnt, spike_rem)
            spike_rem -= take_spike
            rem_cnt = cnt - take_spike
            if rem_cnt <= 0:
                continue
            lo = float(bucket["bin_start"])
            hi = float(bucket["bin_end"])
            if hi <= lo:
                sample_chunks.append(np.full(rem_cnt, lo, dtype=np.float64))
            else:
                sample_chunks.append(np.linspace(lo, hi, rem_cnt + 2, dtype=np.float64)[1:-1])

    baseline_samples = np.concatenate(sample_chunks)
    psi, ks_stat, ks_pvalue, _ = _compute_psi_and_ks_from_samples(baseline_samples, live_values)
    return psi, ks_stat, ks_pvalue, int(base_total)


def _check_data_drift_for_metric(
    *,
    metric: str,
    run_id: str,
    timestamp_iso: str,
    live_df: pd.DataFrame,
    baseline_history_df: pd.DataFrame,
    latest_eda: Mapping[str, Any] | None,
    cfg: Settings,
) -> dict[str, Any]:
    window_hours = int(cfg.data_drift_window_hours)
    psi_threshold = float(cfg.data_drift_psi_threshold)
    ks_p_threshold = float(cfg.data_drift_ks_pvalue_threshold)

    def _failed(err: str, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return sanitize_json_scalar(
            {
                "run_id": run_id,
                "timestamp": timestamp_iso,
                "metric": metric,
                "drift_type": "data_drift",
                "test_name": "psi_ks_2samp",
                "score": None,
                "secondary_score": None,
                "threshold": psi_threshold,
                "status": "check_failed",
                "retrain_triggered": False,
                "action_taken": "check_failed",
                "error_message": err,
                "details_json": {"window_hours": window_hours, **(dict(extra) if extra else {})},
            }
        )

    if latest_eda is None:
        return _failed("Missing baseline stats: no eda_summaries record found.")
    summary_json = latest_eda.get("summary_json")
    distributions = summary_json.get("distributions") if isinstance(summary_json, Mapping) else None
    dist_entry = distributions.get(metric) if isinstance(distributions, Mapping) else None
    if not isinstance(dist_entry, Mapping) or dist_entry.get("status") != "available":
        return _failed(f"Missing baseline distribution stats in eda_summaries for '{metric}'.")

    histogram = dist_entry.get("histogram")
    if not isinstance(histogram, list) or not histogram:
        return _failed(f"Missing baseline histogram bins in eda_summaries for '{metric}'.")

    if live_df.empty or metric not in live_df.columns:
        return _failed(
            f"Insufficient live window rows for '{metric}': found 0, required {window_hours}.",
            extra={"matched_rows": 0},
        )

    live_series = pd.to_numeric(live_df[metric], errors="coerce").dropna()
    live_arr = live_series.to_numpy(dtype=np.float64)
    live_arr = live_arr[np.isfinite(live_arr)]
    if len(live_arr) < window_hours:
        return _failed(
            f"Insufficient live window rows for '{metric}': found {len(live_arr)}, required {window_hours}.",
            extra={"matched_rows": int(len(live_arr))},
        )

    base_arr: np.ndarray | None = None
    if not baseline_history_df.empty and metric in baseline_history_df.columns:
        raw_base = pd.to_numeric(baseline_history_df[metric], errors="coerce").dropna().to_numpy(
            dtype=np.float64
        )
        raw_base = raw_base[np.isfinite(raw_base)]
        if len(raw_base) >= window_hours:
            base_arr = raw_base

    try:
        if base_arr is not None:
            psi, ks_stat, ks_pvalue, baseline_count = _compute_psi_and_ks_from_samples(
                base_arr, live_arr
            )
            baseline_source = "historical_features"
        else:
            dist_min = float(dist_entry["min"]) if dist_entry.get("min") is not None else None
            dist_max = float(dist_entry["max"]) if dist_entry.get("max") is not None else None
            psi, ks_stat, ks_pvalue, baseline_count = compute_psi_and_ks_from_histogram(
                histogram,
                live_arr,
                dist_min=dist_min,
                dist_max=dist_max,
                q25=float(dist_entry["q25"]) if dist_entry.get("q25") is not None else None,
                q50=float(dist_entry["q50"]) if dist_entry.get("q50") is not None else None,
                q75=float(dist_entry["q75"]) if dist_entry.get("q75") is not None else None,
            )
            baseline_source = "eda_histogram"
    except Exception as exc:
        return _failed(f"Data drift calculation failed for '{metric}': {exc}")

    if not (np.isfinite(psi) and np.isfinite(ks_pvalue)):
        return _failed(f"Non-finite PSI or KS p-value computed for '{metric}'.")

    drifted = (psi > psi_threshold) or (ks_pvalue < ks_p_threshold)
    return sanitize_json_scalar(
        {
            "run_id": run_id,
            "timestamp": timestamp_iso,
            "metric": metric,
            "drift_type": "data_drift",
            "test_name": "psi_ks_2samp",
            "score": psi,
            "secondary_score": ks_pvalue,
            "threshold": psi_threshold,
            "status": "drift_detected" if drifted else "no_drift",
            "retrain_triggered": False,
            "action_taken": "none",
            "error_message": None,
            "details_json": {
                "psi": psi,
                "psi_threshold": psi_threshold,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "ks_pvalue_threshold": ks_p_threshold,
                "window_hours": window_hours,
                "matched_rows": int(len(live_arr)),
                "baseline_count": baseline_count,
                "baseline_source": baseline_source,
                "eda_run_id": latest_eda.get("run_id"),
            },
        }
    )


def _check_model_drift_for_metric(
    *,
    metric: str,
    run_id: str,
    timestamp_iso: str,
    now_ts: pd.Timestamp,
    ledger_rows: Sequence[Mapping[str, Any]],
    champion_by_metric: Mapping[str, Mapping[str, Any]],
    cfg: Settings,
) -> dict[str, Any]:
    window_hours = int(cfg.model_drift_window_hours)
    ratio_threshold = float(cfg.model_drift_rmse_ratio_threshold)
    window_start = now_ts - pd.Timedelta(hours=window_hours - 1)

    def _failed(err: str, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return sanitize_json_scalar(
            {
                "run_id": run_id,
                "timestamp": timestamp_iso,
                "metric": metric,
                "drift_type": "model_drift",
                "test_name": "rolling_rmse_ratio",
                "score": None,
                "secondary_score": None,
                "threshold": ratio_threshold,
                "status": "check_failed",
                "retrain_triggered": False,
                "action_taken": "check_failed",
                "error_message": err,
                "details_json": {"window_hours": window_hours, **(dict(extra) if extra else {})},
            }
        )

    champ = champion_by_metric.get(metric)
    if champ is None or champ.get("val_rmse") is None:
        return _failed(
            f"Missing champion validation RMSE baseline in model_registry_metrics for '{metric}'."
        )
    val_rmse = float(champ["val_rmse"])
    if not np.isfinite(val_rmse) or val_rmse < 0.0:
        return _failed(f"Invalid champion val_rmse baseline ({val_rmse!r}) for '{metric}'.")

    matched_rows: list[Mapping[str, Any]] = []
    distinct_hours: set[pd.Timestamp] = set()
    for row in ledger_rows:
        if row.get("metric") != metric:
            continue
        actual = row.get("actual_value")
        pred = row.get("predicted_value")
        if actual is None or pred is None:
            continue
        target_ts = _to_utc_hour_ts(row["target_timestamp_utc"])
        if window_start <= target_ts <= now_ts:
            matched_rows.append(row)
            distinct_hours.add(target_ts)

    if len(matched_rows) < window_hours or len(distinct_hours) < window_hours:
        return _failed(
            f"Insufficient realized forecast_ledger rows for '{metric}': "
            f"matched_rows={len(matched_rows)}, distinct_hours={len(distinct_hours)}, required={window_hours}.",
            extra={
                "matched_rows": len(matched_rows),
                "distinct_target_hours": len(distinct_hours),
            },
        )

    y_pred = np.asarray([float(r["predicted_value"]) for r in matched_rows], dtype=np.float64)
    y_true = np.asarray([float(r["actual_value"]) for r in matched_rows], dtype=np.float64)
    if METRIC_SPECS[metric].is_circular:
        err = ((y_pred - y_true + 180.0) % 360.0) - 180.0
    else:
        err = y_pred - y_true
    live_rmse = float(np.sqrt(np.mean(err ** 2)))

    if val_rmse <= 1e-12:
        rmse_ratio = (
            1.0 if live_rmse <= 1e-12 else float(max(ratio_threshold * 10.0, 100.0))
        )
    else:
        rmse_ratio = float(live_rmse / val_rmse)

    drifted = rmse_ratio > ratio_threshold
    return sanitize_json_scalar(
        {
            "run_id": run_id,
            "timestamp": timestamp_iso,
            "metric": metric,
            "drift_type": "model_drift",
            "test_name": "rolling_rmse_ratio",
            "score": rmse_ratio,
            "secondary_score": live_rmse,
            "threshold": ratio_threshold,
            "status": "drift_detected" if drifted else "no_drift",
            "retrain_triggered": False,
            "action_taken": "none",
            "error_message": None,
            "details_json": {
                "live_rmse": live_rmse,
                "val_rmse": val_rmse,
                "rmse_ratio": rmse_ratio,
                "threshold": ratio_threshold,
                "window_hours": window_hours,
                "matched_rows": len(matched_rows),
                "distinct_target_hours": len(distinct_hours),
                "champion_family": champ.get("model_family"),
                "champion_version": champ.get("model_version"),
            },
        }
    )


def _trigger_retraining(
    cfg: Settings,
    *,
    run_id: str,
    drifted_metrics: Sequence[str],
    retrain_dispatcher: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[bool, str, str | None]:
    payload = {
        "ref": cfg.github_ref,
        "inputs": {
            "trigger_reason": "drift_detected",
            "run_id": run_id,
            "metrics": ",".join(drifted_metrics),
        },
    }
    if retrain_dispatcher is not None:
        try:
            ok = bool(retrain_dispatcher(payload))
            if ok:
                return True, "local_retrain_triggered", None
            return False, "retrain_dispatch_failed", "Custom retrain dispatcher returned False."
        except Exception as exc:
            return False, "retrain_dispatch_failed", f"Retrain dispatch error: {exc}"

    if not cfg.has_github_credentials:
        return (
            False,
            "retrain_skipped_no_github_credentials",
            None,
        )

    url = (
        f"https://api.github.com/repos/{cfg.github_repo}/actions/workflows/"
        f"{cfg.github_train_workflow}/dispatches"
    )
    headers = {
        "Authorization": f"Bearer {cfg.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        with httpx.Client(timeout=float(cfg.http_timeout_seconds)) as client:
            resp = client.post(url, headers=headers, json=payload)
        if 200 <= resp.status_code < 300:
            return True, "github_workflow_dispatch_triggered", None
        return (
            False,
            "retrain_dispatch_failed",
            f"GitHub workflow_dispatch HTTP {resp.status_code}: {resp.text[:250]}",
        )
    except Exception as exc:
        return False, "retrain_dispatch_failed", f"GitHub workflow_dispatch request failed: {exc}"


def evaluate_and_log_drift(
    *,
    run_id: str,
    timestamp_utc: datetime | pd.Timestamp | str,
    storage: SkyCastStorage,
    settings: Settings | None = None,
    retrain_dispatcher: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Backfill actuals, evaluate data & model drift across all 8 metrics, and log to `drift_log`."""
    cfg = settings or storage.settings or get_settings()
    now_ts = _to_utc_hour_ts(timestamp_utc)
    timestamp_iso = now_ts.isoformat()

    # 1. Always backfill forecast_ledger.actual_value before checking model drift.
    ledger_rows, backfilled_count = backfill_forecast_ledger_actuals(
        now_ts=now_ts, storage=storage
    )

    # 2. Load live feature window and baseline stats.
    data_window_start = now_ts - pd.Timedelta(hours=int(cfg.data_drift_window_hours) - 1)
    live_df = storage.read_weather_features(
        start_utc=data_window_start.to_pydatetime(),
        end_utc=now_ts.to_pydatetime(),
        expand_features_json=False,
    )
    if not live_df.empty:
        live_df = live_df.copy()
        live_df["timestamp_utc"] = pd.to_datetime(live_df["timestamp_utc"], utc=True).dt.floor("h")
        live_df = live_df.drop_duplicates(subset=["timestamp_utc"], keep="last")
        live_df = live_df.loc[
            (live_df["timestamp_utc"] >= data_window_start) & (live_df["timestamp_utc"] <= now_ts)
        ]

    latest_eda = storage.read_latest_eda_summary()
    baseline_history_df = pd.DataFrame()
    if latest_eda is not None:
        eda_start_raw = latest_eda.get("date_range_start")
        eda_end_raw = latest_eda.get("date_range_end")
        hist_start = _to_utc_hour_ts(eda_start_raw).to_pydatetime() if eda_start_raw else None
        hist_end_bound = data_window_start - pd.Timedelta(hours=1)
        if eda_end_raw:
            eda_end_ts = _to_utc_hour_ts(eda_end_raw)
            if eda_end_ts < hist_end_bound:
                hist_end_bound = eda_end_ts
        if hist_start is None or pd.Timestamp(hist_start) <= hist_end_bound:
            baseline_history_df = storage.read_weather_features(
                start_utc=hist_start,
                end_utc=hist_end_bound.to_pydatetime(),
                expand_features_json=False,
            )

    reg_champions = storage.read_model_registry_metrics(is_champion=True)
    champion_by_metric: dict[str, Mapping[str, Any]] = {}
    for row in reg_champions:
        m = str(row.get("metric") or "")
        if m and m not in champion_by_metric:
            champion_by_metric[m] = row

    drift_rows: list[dict[str, Any]] = []
    for metric in TARGET_METRICS:
        drift_rows.append(
            _check_data_drift_for_metric(
                metric=metric,
                run_id=run_id,
                timestamp_iso=timestamp_iso,
                live_df=live_df,
                baseline_history_df=baseline_history_df,
                latest_eda=latest_eda,
                cfg=cfg,
            )
        )
        drift_rows.append(
            _check_model_drift_for_metric(
                metric=metric,
                run_id=run_id,
                timestamp_iso=timestamp_iso,
                now_ts=now_ts,
                ledger_rows=ledger_rows,
                champion_by_metric=champion_by_metric,
                cfg=cfg,
            )
        )

    drifted_rows = [r for r in drift_rows if r["status"] == "drift_detected"]
    retrain_triggered = False
    dispatch_action = "none"
    dispatch_error: str | None = None

    if drifted_rows:
        drifted_metrics = list(dict.fromkeys(str(r["metric"]) for r in drifted_rows))
        retrain_triggered, dispatch_action, dispatch_error = _trigger_retraining(
            cfg,
            run_id=run_id,
            drifted_metrics=drifted_metrics,
            retrain_dispatcher=retrain_dispatcher,
        )
        for row in drifted_rows:
            row["retrain_triggered"] = retrain_triggered
            row["action_taken"] = dispatch_action
            if dispatch_error is not None:
                row["error_message"] = dispatch_error
            if isinstance(row.get("details_json"), dict):
                row["details_json"]["retrain_action"] = dispatch_action
                if dispatch_error is not None:
                    row["details_json"]["dispatch_error"] = dispatch_error

    storage.insert_drift_logs(drift_rows)

    summary = {
        "backfilled_actual_rows": backfilled_count,
        "total_checks": len(drift_rows),
        "drift_detected_count": len(drifted_rows),
        "check_failed_count": sum(1 for r in drift_rows if r["status"] == "check_failed"),
        "no_drift_count": sum(1 for r in drift_rows if r["status"] == "no_drift"),
        "retrain_triggered": retrain_triggered,
        "dispatch_action": dispatch_action,
    }
    return drift_rows, retrain_triggered, summary
