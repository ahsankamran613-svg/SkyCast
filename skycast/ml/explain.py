"""SHAP (global & local) and LIME (local) explainability payloads for SkyCast models."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from skycast.config.settings import METRIC_SPECS, Settings, get_settings
from skycast.features.engineer import sanitize_json_scalar
from skycast.ml.evaluate import ChronologicalSplit, reconstruct_wind_direction
from skycast.ml.models.base import BaseForecastModel, _compute_circular_mean
from skycast.ml.models.classical import RandomForestModel, RidgeModel

_KERNEL_BG_MAX_ROWS = 50
_KERNEL_EVAL_MAX_ROWS = 10
_TREE_LINEAR_EVAL_MAX_ROWS = 100
_TOP_K_FEATURES = 10
_ZERO_VAR_STD_EPS = 1e-12
_LIME_JITTER_STD = 1e-8


def subsample_chronological_rows(X: np.ndarray, max_rows: int) -> np.ndarray:
    """Deterministically subsample up to `max_rows` rows preserving chronological order."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError(f"Expected non-empty 2D array for chronological subsample, got shape {arr.shape}.")
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}.")
    if len(arr) <= max_rows:
        return arr.copy()
    indices = np.linspace(0, len(arr) - 1, max_rows, dtype=int)
    return arr[indices].copy()


def _extract_horizon_shap(
    shap_values: Any,
    expected_value: Any,
    horizon_idx: int,
) -> tuple[np.ndarray, float]:
    sv = np.asarray(shap_values, dtype=np.float64)
    if sv.ndim == 2:
        sv_h = sv
    elif sv.ndim == 3:
        sv_h = sv[:, :, horizon_idx]
    else:
        raise ValueError(f"Unexpected SHAP values ndim={sv.ndim}, shape={sv.shape}.")

    ev_arr = np.asarray(expected_value, dtype=np.float64).ravel()
    if ev_arr.size == 0:
        ev_h = 0.0
    elif ev_arr.size == 1:
        ev_h = float(ev_arr[0])
    else:
        ev_h = float(ev_arr[horizon_idx])
    return sv_h, ev_h


def _format_top_global_attributions(
    feature_columns: list[str],
    importance_vec: np.ndarray,
    signed_mean_vec: np.ndarray,
    ref_raw_values: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(importance_vec), kind="mergesort")[:top_k]
    out: list[dict[str, Any]] = []
    for idx in order:
        out.append(
            {
                "feature": feature_columns[int(idx)],
                "importance": sanitize_json_scalar(float(importance_vec[idx])),
                "mean_abs_shap": sanitize_json_scalar(float(importance_vec[idx])),
                "mean_shap": sanitize_json_scalar(float(signed_mean_vec[idx])),
                "value": sanitize_json_scalar(float(ref_raw_values[idx])),
            }
        )
    return out


def _format_top_local_shap_attributions(
    feature_columns: list[str],
    local_shap_vec: np.ndarray,
    ref_raw_values: np.ndarray,
    ref_scaled_values: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(local_shap_vec), kind="mergesort")[:top_k]
    out: list[dict[str, Any]] = []
    for idx in order:
        val = float(local_shap_vec[idx])
        out.append(
            {
                "feature": feature_columns[int(idx)],
                "importance": sanitize_json_scalar(abs(val)),
                "shap_value": sanitize_json_scalar(val),
                "contribution": sanitize_json_scalar(val),
                "value": sanitize_json_scalar(float(ref_raw_values[idx])),
                "scaled_value": sanitize_json_scalar(float(ref_scaled_values[idx])),
            }
        )
    return out


def _compute_shap_for_estimator(
    *,
    model: BaseForecastModel,
    estimator: Any,
    X_eval: np.ndarray,
    X_ref: np.ndarray,
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute `(sv_eval, sv_local, base_value)` for a sklearn tree/linear estimator."""
    if isinstance(model, RandomForestModel):
        explainer = shap.TreeExplainer(estimator)
    elif isinstance(model, RidgeModel):
        explainer = shap.LinearExplainer(estimator, X_eval)
    else:
        raise TypeError(f"Unsupported estimator model family: {model.family_name}")

    sv_eval_raw = explainer.shap_values(X_eval)
    sv_ref_raw = explainer.shap_values(X_ref)
    sv_eval, base_val = _extract_horizon_shap(sv_eval_raw, explainer.expected_value, horizon_idx)
    sv_ref, _ = _extract_horizon_shap(sv_ref_raw, explainer.expected_value, horizon_idx)
    return sv_eval, sv_ref[0], base_val


def _compute_shap_arrays(
    *,
    model: BaseForecastModel,
    split: ChronologicalSplit,
    X_ref: np.ndarray,
    horizon_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute `(importance_global, mean_signed_global, local_shap, base_value, predicted_value)`."""
    X_train = np.asarray(split.X_train_scaled, dtype=np.float64)
    H = model.horizon_hours

    if isinstance(model, (RandomForestModel, RidgeModel)):
        X_eval = subsample_chronological_rows(X_train, _TREE_LINEAR_EVAL_MAX_ROWS)
        if not model.is_circular:
            sv_eval, sv_local, base_val = _compute_shap_for_estimator(
                model=model,
                estimator=model.estimator,
                X_eval=X_eval,
                X_ref=X_ref,
                horizon_idx=horizon_idx,
            )
            pred_val = float(model.predict_point_matrix_for_explain(X_ref)[0, horizon_idx])
            spec = METRIC_SPECS[model.metric]
            pred_val = spec.clamp(pred_val)
            return (
                np.mean(np.abs(sv_eval), axis=0),
                np.mean(sv_eval, axis=0),
                sv_local,
                base_val,
                pred_val,
            )

        # Circular wind_direction_10m for Tree / Linear explainers via (sin, cos) components
        sv_sin_eval, sv_sin_loc, base_sin = _compute_shap_for_estimator(
            model=model,
            estimator=model.estimator_sin,
            X_eval=X_eval,
            X_ref=X_ref,
            horizon_idx=horizon_idx,
        )
        sv_cos_eval, sv_cos_loc, base_cos = _compute_shap_for_estimator(
            model=model,
            estimator=model.estimator_cos,
            X_eval=X_eval,
            X_ref=X_ref,
            horizon_idx=horizon_idx,
        )

        sin_ref_arr = np.asarray(model.estimator_sin.predict(X_ref), dtype=np.float64).reshape(1, H)
        cos_ref_arr = np.asarray(model.estimator_cos.predict(X_ref), dtype=np.float64).reshape(1, H)
        sin_p = float(sin_ref_arr[0, horizon_idx])
        cos_p = float(cos_ref_arr[0, horizon_idx])

        r2 = max(sin_p * sin_p + cos_p * cos_p, 1e-6)
        w_sin = (180.0 / math.pi) * (cos_p / r2)
        w_cos = -(180.0 / math.pi) * (sin_p / r2)

        imp_global = np.mean(np.sqrt(sv_sin_eval ** 2 + sv_cos_eval ** 2), axis=0)
        mean_signed = w_sin * np.mean(sv_sin_eval, axis=0) + w_cos * np.mean(sv_cos_eval, axis=0)
        local_shap = w_sin * sv_sin_loc + w_cos * sv_cos_loc

        base_deg = float(reconstruct_wind_direction(base_sin, base_cos))
        pred_deg = float(reconstruct_wind_direction(sin_p, cos_p))
        return imp_global, mean_signed, local_shap, base_deg, pred_deg

    # Generic / LSTM / SARIMAX -> KernelExplainer on 50 chronological train background rows
    X_bg = subsample_chronological_rows(X_train, _KERNEL_BG_MAX_ROWS)
    X_eval = subsample_chronological_rows(X_train, _KERNEL_EVAL_MAX_ROWS)

    ref_angle: float | None = None
    if model.is_circular:
        bg_preds = model.predict_point_matrix_for_explain(X_bg)[:, horizon_idx]
        ref_angle = _compute_circular_mean(bg_preds)

    def _predict_kernel(X_batch: np.ndarray) -> np.ndarray:
        return model.predict_point_for_explain(X_batch, reference_angle_deg=ref_angle)

    explainer = shap.KernelExplainer(_predict_kernel, X_bg, seed=seed)
    n_features = X_train.shape[1]
    nsamples = min(200, max(64, n_features // 2))
    sv_eval = np.asarray(
        explainer.shap_values(X_eval, nsamples=nsamples, silent=True),
        dtype=np.float64,
    )
    sv_ref = np.asarray(
        explainer.shap_values(X_ref, nsamples=nsamples, silent=True),
        dtype=np.float64,
    )
    if sv_eval.ndim == 1:
        sv_eval = sv_eval[None, :]
    if sv_ref.ndim == 1:
        sv_ref = sv_ref[None, :]

    ev_raw = float(np.asarray(explainer.expected_value, dtype=np.float64).ravel()[0])
    pred_raw = float(_predict_kernel(X_ref)[0])

    spec = METRIC_SPECS[model.metric]
    base_val = spec.clamp(ev_raw)
    pred_val = spec.clamp(pred_raw)
    if model.is_circular:
        base_val = 0.0 if base_val >= 360.0 else base_val
        pred_val = 0.0 if pred_val >= 360.0 else pred_val

    return (
        np.mean(np.abs(sv_eval), axis=0),
        np.mean(sv_eval, axis=0),
        sv_ref[0],
        base_val,
        pred_val,
    )


def _compute_lime_local(
    *,
    model: BaseForecastModel,
    X_train_scaled: np.ndarray,
    x_ref_scaled: np.ndarray,
    ref_raw_values: np.ndarray,
    feature_columns: list[str],
    horizon_idx: int,
    seed: int,
    top_k: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Run `LimeTabularExplainer` handling zero-variance train columns cleanly."""
    ref_angle: float | None = None
    if model.is_circular:
        bg_sub = subsample_chronological_rows(X_train_scaled, min(50, len(X_train_scaled)))
        bg_preds = model.predict_point_matrix_for_explain(bg_sub)[:, horizon_idx]
        ref_angle = _compute_circular_mean(bg_preds)

    def _predict_scalar(X_batch: np.ndarray) -> np.ndarray:
        return model.predict_point_for_explain(X_batch, reference_angle_deg=ref_angle)

    col_stds = np.std(X_train_scaled, axis=0)
    active_mask = col_stds >= _ZERO_VAR_STD_EPS
    active_indices = np.where(active_mask)[0]

    if len(active_indices) > 0:
        X_lime_train = X_train_scaled[:, active_indices]
        x_lime_inst = x_ref_scaled[active_indices]
        lime_names = [feature_columns[int(i)] for i in active_indices]
        index_map = [int(i) for i in active_indices]

        def _predict_active(X_sub: np.ndarray) -> np.ndarray:
            full = np.repeat(x_ref_scaled[None, :], len(X_sub), axis=0)
            full[:, active_indices] = X_sub
            return _predict_scalar(full)

        lime_predict_fn = _predict_active
    else:
        rng = np.random.default_rng(seed)
        X_lime_train = X_train_scaled + rng.normal(0.0, _LIME_JITTER_STD, size=X_train_scaled.shape)
        x_lime_inst = x_ref_scaled.copy()
        lime_names = list(feature_columns)
        index_map = list(range(len(feature_columns)))
        lime_predict_fn = _predict_scalar

    n_lime_features = min(top_k, len(lime_names))
    explainer = LimeTabularExplainer(
        training_data=X_lime_train,
        feature_names=lime_names,
        mode="regression",
        discretize_continuous=True,
        random_state=seed,
    )
    num_samples = min(500, max(200, len(lime_names) * 2))
    exp = explainer.explain_instance(
        data_row=x_lime_inst,
        predict_fn=lime_predict_fn,
        num_features=n_lime_features,
        num_samples=num_samples,
    )

    exp_map = exp.as_map()
    label_key = 1 if 1 in exp_map else next(iter(exp_map.keys()))
    map_entries = exp_map[label_key]
    list_entries = exp.as_list(label=label_key)

    rule_by_rank = [str(rule) for rule, _ in list_entries]
    attributions: list[dict[str, Any]] = []
    for rank_i, (sub_idx_raw, weight_raw) in enumerate(map_entries[:top_k]):
        orig_idx = index_map[int(sub_idx_raw)]
        weight = float(weight_raw)
        rule = rule_by_rank[rank_i] if rank_i < len(rule_by_rank) else feature_columns[orig_idx]
        attributions.append(
            {
                "feature": feature_columns[orig_idx],
                "importance": sanitize_json_scalar(abs(weight)),
                "weight": sanitize_json_scalar(weight),
                "contribution": sanitize_json_scalar(weight),
                "rule": rule,
                "value": sanitize_json_scalar(float(ref_raw_values[orig_idx])),
                "scaled_value": sanitize_json_scalar(float(x_ref_scaled[orig_idx])),
            }
        )

    intercept_raw = float(exp.intercept[label_key]) if isinstance(exp.intercept, dict) else float(exp.intercept)
    pred_raw = float(_predict_scalar(x_ref_scaled[None, :])[0])
    spec = METRIC_SPECS[model.metric]
    base_val = spec.clamp(intercept_raw)
    pred_val = spec.clamp(pred_raw)
    if model.is_circular:
        base_val = 0.0 if base_val >= 360.0 else base_val
        pred_val = 0.0 if pred_val >= 360.0 else pred_val

    return base_val, pred_val, attributions


def compute_explanations_for_metric(
    *,
    train_run_id: str,
    model: BaseForecastModel,
    split: ChronologicalSplit,
    reference_df: pd.DataFrame | None = None,
    computed_at_utc: str | None = None,
    settings: Settings | None = None,
    top_k: int = _TOP_K_FEATURES,
) -> list[dict[str, Any]]:
    """Compute SHAP (`global`, `local`) and LIME (`local`) rows for `shap_lime_explanations`."""
    if not isinstance(train_run_id, str) or not train_run_id.strip():
        raise ValueError("train_run_id must be a non-empty string.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    cfg = settings or model.settings or get_settings()
    seed = int(cfg.random_seed)
    ts_computed = computed_at_utc or datetime.now(timezone.utc).isoformat()
    horizon_idx = model.horizon_hours - 1
    feature_columns = list(split.feature_columns)

    if reference_df is not None and not reference_df.empty:
        ref_row_df = reference_df.tail(1).reset_index(drop=True)
        X_ref = model._prepare_scaled_features(ref_row_df, None)
    else:
        ref_row_df = split.test_df.tail(1).reset_index(drop=True)
        X_ref = np.asarray(split.X_test_scaled[-1:], dtype=np.float64)

    ref_ts = pd.to_datetime(ref_row_df["timestamp_utc"].iloc[0], utc=True).isoformat()
    x_ref_scaled = X_ref[0]
    ref_raw_values = split.scaler.inverse_transform(X_ref)[0]

    imp_global, mean_signed, local_shap, shap_base, shap_pred = _compute_shap_arrays(
        model=model,
        split=split,
        X_ref=X_ref,
        horizon_idx=horizon_idx,
        seed=seed,
    )

    shap_global_attrs = _format_top_global_attributions(
        feature_columns,
        imp_global,
        mean_signed,
        ref_raw_values,
        top_k=top_k,
    )
    shap_local_attrs = _format_top_local_shap_attributions(
        feature_columns,
        local_shap,
        ref_raw_values,
        x_ref_scaled,
        top_k=top_k,
    )

    lime_base, lime_pred, lime_local_attrs = _compute_lime_local(
        model=model,
        X_train_scaled=np.asarray(split.X_train_scaled, dtype=np.float64),
        x_ref_scaled=x_ref_scaled,
        ref_raw_values=ref_raw_values,
        feature_columns=feature_columns,
        horizon_idx=horizon_idx,
        seed=seed,
        top_k=top_k,
    )

    records = [
        {
            "train_run_id": train_run_id.strip(),
            "computed_at_utc": ts_computed,
            "metric": model.metric,
            "model_family": model.family_name,
            "method": "shap",
            "scope": "global",
            "base_value": sanitize_json_scalar(shap_base),
            "predicted_value": sanitize_json_scalar(shap_pred),
            "reference_timestamp_utc": ref_ts,
            "feature_attributions": shap_global_attrs,
        },
        {
            "train_run_id": train_run_id.strip(),
            "computed_at_utc": ts_computed,
            "metric": model.metric,
            "model_family": model.family_name,
            "method": "shap",
            "scope": "local",
            "base_value": sanitize_json_scalar(shap_base),
            "predicted_value": sanitize_json_scalar(shap_pred),
            "reference_timestamp_utc": ref_ts,
            "feature_attributions": shap_local_attrs,
        },
        {
            "train_run_id": train_run_id.strip(),
            "computed_at_utc": ts_computed,
            "metric": model.metric,
            "model_family": model.family_name,
            "method": "lime",
            "scope": "local",
            "base_value": sanitize_json_scalar(lime_base),
            "predicted_value": sanitize_json_scalar(lime_pred),
            "reference_timestamp_utc": ref_ts,
            "feature_attributions": lime_local_attrs,
        },
    ]

    sanitized = sanitize_json_scalar(records)
    json.dumps(sanitized, allow_nan=False)
    return sanitized
