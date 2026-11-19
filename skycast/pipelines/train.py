"""End-to-end daily and drift-triggered ML training, EDA, evaluation, XAI, and registry pipeline."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from skycast.config.settings import (
    METRIC_SPECS,
    MODEL_FAMILIES,
    TARGET_METRICS,
    Settings,
    get_settings,
)
from skycast.data.storage import SkyCastStorage
from skycast.features.engineer import (
    add_forecast_targets,
    engineer_features,
    get_model_feature_columns,
    split_training_and_inference_rows,
)
from skycast.ml.eda import compute_eda_summary, save_eda_summary_json
from skycast.ml.evaluate import (
    build_comparison_series_rows,
    compute_improvement_over_naive_pct,
    extract_target_matrix,
    get_target_column_names,
    get_wind_component_target_columns,
    score_multi_horizon,
    split_and_scale,
)
from skycast.ml.explain import compute_explanations_for_metric
from skycast.ml.models import MODEL_CLASS_BY_FAMILY
from skycast.ml.registry import promote_champion_if_better
from skycast.pipelines._common import safe_record_pipeline_failure

logger = logging.getLogger(__name__)


def _ensure_engineered_with_targets(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    expected_features = set(get_model_feature_columns(settings=cfg))
    if not expected_features.issubset(df.columns):
        return engineer_features(df, settings=cfg, include_targets=True)

    required_targets: list[str] = ["has_full_24h_targets"]
    for metric in TARGET_METRICS:
        required_targets.extend(get_target_column_names(metric, settings=cfg))
    sin_cols, cos_cols = get_wind_component_target_columns(settings=cfg)
    required_targets.extend(sin_cols)
    required_targets.extend(cos_cols)

    if any(col not in df.columns for col in required_targets):
        return add_forecast_targets(df, settings=cfg)
    return df


def _resolve_reference_df(
    inference_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame | None:
    if inference_df.empty:
        return None
    complete_mask = ~inference_df[feature_columns].isna().any(axis=1)
    complete_rows = inference_df.loc[complete_mask]
    if complete_rows.empty:
        return None
    return complete_rows.tail(1).reset_index(drop=True)


def _validate_target_metrics(metrics: Sequence[str] | None) -> list[str]:
    if metrics is None:
        return list(TARGET_METRICS)
    resolved = [str(m) for m in metrics]
    if not resolved:
        raise ValueError("metrics cannot be empty.")
    invalid = [m for m in resolved if m not in METRIC_SPECS]
    if invalid:
        raise ValueError(f"Unknown metric(s): {invalid}. Expected subset of {TARGET_METRICS}.")
    return list(dict.fromkeys(resolved))


def _validate_families(families: Sequence[str] | None) -> list[str]:
    if families is None:
        return list(MODEL_FAMILIES)
    resolved = [str(f) for f in families]
    if not resolved:
        raise ValueError("families cannot be empty.")
    invalid = [f for f in resolved if f not in MODEL_CLASS_BY_FAMILY]
    if invalid:
        raise ValueError(f"Unknown model family(ies): {invalid}. Expected subset of {MODEL_FAMILIES}.")
    deduped = list(dict.fromkeys(resolved))
    if "Naive24h" not in deduped:
        deduped.insert(0, "Naive24h")
    # Order canonically by MODEL_FAMILIES for deterministic tie-breaking
    return [f for f in MODEL_FAMILIES if f in deduped]


def run_training(
    *,
    days: int | None = None,
    metrics: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    df: pd.DataFrame | None = None,
    train_run_id: str | None = None,
    settings: Settings | None = None,
    storage: SkyCastStorage | None = None,
    save_eda_doc: bool = False,
    eda_output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute Component 3 training, EDA, evaluation, XAI, and champion promotion."""
    cfg = settings or get_settings()
    store = storage or SkyCastStorage(cfg)
    target_metrics = _validate_target_metrics(metrics)
    target_families = _validate_families(families)

    started_at = datetime.now(timezone.utc)
    if train_run_id is not None:
        if not isinstance(train_run_id, str) or not train_run_id.strip():
            raise ValueError("train_run_id must be a non-empty string when provided.")
        resolved_run_id = train_run_id.strip()
    else:
        resolved_run_id = f"train-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"

    store.record_pipeline_run(
        run_id=resolved_run_id,
        pipeline_type="training",
        status="running",
        started_at_utc=started_at,
        metadata={
            "days": days,
            "metrics": target_metrics,
            "families": target_families,
            "forecast_horizon_hours": int(cfg.forecast_horizon_hours),
        },
    )

    try:
        history_df = df.copy() if df is not None else store.read_training_history(days=days)
        if history_df.empty:
            raise RuntimeError(
                "Training aborted: no historical weather features available in storage."
            )

        imputation_audit = store.get_latest_imputation_audit()
        eda_payload = compute_eda_summary(
            history_df,
            settings=cfg,
            imputation_audit=imputation_audit,
            run_id=resolved_run_id,
        )
        eda_storage_res = store.upsert_eda_summary(eda_payload)
        eda_doc_path: str | None = None
        if save_eda_doc:
            saved_path = save_eda_summary_json(
                eda_payload, output_path=eda_output_path, settings=cfg
            )
            eda_doc_path = str(saved_path)

        targeted_df = _ensure_engineered_with_targets(history_df, cfg)
        trainable_df, inference_df = split_training_and_inference_rows(targeted_df, settings=cfg)
        if trainable_df.empty:
            raise RuntimeError(
                "Training aborted: no complete trainable rows remain after target & feature filtering."
            )

        split = split_and_scale(trainable_df, settings=cfg)
        reference_df = _resolve_reference_df(inference_df, split.feature_columns)

        H = int(cfg.forecast_horizon_hours)
        horizons = range(1, H + 1)
        trained_at_iso = datetime.now(timezone.utc).isoformat()

        all_registry_rows: list[dict[str, Any]] = []
        all_comparison_rows: list[dict[str, Any]] = []
        all_explanation_rows: list[dict[str, Any]] = []
        metric_summaries: dict[str, dict[str, Any]] = {}

        for metric in target_metrics:
            y_val_true = extract_target_matrix(split.val_df, metric, horizons=horizons)
            y_test_true = extract_target_matrix(split.test_df, metric, horizons=horizons)

            fitted_models: dict[str, Any] = {}
            val_scores_by_family: dict[str, dict[str, float]] = {}
            test_scores_by_family: dict[str, dict[str, float]] = {}
            test_preds_by_family: dict[str, Any] = {}

            for family in target_families:
                model_cls = MODEL_CLASS_BY_FAMILY[family]
                model = model_cls(metric, settings=cfg).fit(split)
                val_fc = model.predict(split.val_df, split.X_val_scaled)
                test_fc = model.predict(
                    split.test_df,
                    split.X_test_scaled,
                    context_df=split.val_df,
                    context_X_scaled=split.X_val_scaled,
                )

                fitted_models[family] = model
                val_scores_by_family[family] = score_multi_horizon(
                    y_val_true, val_fc.predicted, metric=metric
                )
                test_scores_by_family[family] = score_multi_horizon(
                    y_test_true, test_fc.predicted, metric=metric
                )
                test_preds_by_family[family] = test_fc.predicted

            naive_val_rmse = float(val_scores_by_family["Naive24h"]["rmse"])

            # Select best finite non-naive candidate (tie-break deterministically by target_families order)
            non_naive_families = [f for f in target_families if f != "Naive24h"]
            finite_candidates = [
                f for f in non_naive_families if math.isfinite(val_scores_by_family[f]["rmse"])
            ]
            best_candidate_family = (
                min(finite_candidates, key=lambda f: val_scores_by_family[f]["rmse"])
                if finite_candidates
                else None
            )

            metric_reg_rows: dict[str, dict[str, Any]] = {}
            for family in target_families:
                val_s = val_scores_by_family[family]
                test_s = test_scores_by_family[family]
                imp_pct = compute_improvement_over_naive_pct(val_s["rmse"], naive_val_rmse)
                metric_reg_rows[family] = {
                    "train_run_id": resolved_run_id,
                    "trained_at_utc": trained_at_iso,
                    "metric": metric,
                    "model_family": family,
                    "model_version": None,
                    "val_rmse": float(val_s["rmse"]),
                    "val_mae": float(val_s["mae"]),
                    "val_r2": float(val_s["r2"]),
                    "test_rmse": float(test_s["rmse"]),
                    "test_mae": float(test_s["mae"]),
                    "test_r2": float(test_s["r2"]),
                    "improvement_over_naive_pct": float(imp_pct),
                    "is_champion": False,
                    "promoted_to_registry": False,
                    "hyperparameters": dict(fitted_models[family].hyperparameters),
                    "training_samples": int(len(split.train_df)),
                }

            promotion_res: dict[str, Any] = {
                "is_champion": False,
                "promoted_to_registry": False,
                "model_version": None,
                "reason": "no_non_naive_candidate",
            }
            if best_candidate_family is not None:
                promotion_res = promote_champion_if_better(
                    metric=metric,
                    candidate_model=fitted_models[best_candidate_family],
                    candidate_metrics=metric_reg_rows[best_candidate_family],
                    naive_val_rmse=naive_val_rmse,
                    storage=store,
                    input_example=split.X_train_scaled[:1],
                    settings=cfg,
                )
                metric_reg_rows[best_candidate_family]["is_champion"] = bool(
                    promotion_res["is_champion"]
                )
                metric_reg_rows[best_candidate_family]["promoted_to_registry"] = bool(
                    promotion_res["promoted_to_registry"]
                )
                metric_reg_rows[best_candidate_family]["model_version"] = promotion_res.get(
                    "model_version"
                )

            all_registry_rows.extend(metric_reg_rows[f] for f in target_families)

            comp_rows = build_comparison_series_rows(
                train_run_id=resolved_run_id,
                metric=metric,
                origin_timestamps=split.test_df["timestamp_utc"],
                actual=y_test_true,
                predictions_by_family=test_preds_by_family,
                settings=cfg,
                horizon_hours=H,
            )
            all_comparison_rows.extend(comp_rows)

            explain_family = best_candidate_family or "Naive24h"
            xai_rows = compute_explanations_for_metric(
                train_run_id=resolved_run_id,
                model=fitted_models[explain_family],
                split=split,
                reference_df=reference_df,
                computed_at_utc=trained_at_iso,
                settings=cfg,
            )
            all_explanation_rows.extend(xai_rows)

            metric_summaries[metric] = {
                "best_candidate_family": best_candidate_family,
                "naive_val_rmse": naive_val_rmse,
                "best_candidate_val_rmse": (
                    float(val_scores_by_family[best_candidate_family]["rmse"])
                    if best_candidate_family
                    else None
                ),
                "promotion": promotion_res,
                "explained_family": explain_family,
            }

        reg_storage_res = store.upsert_model_registry_metrics(all_registry_rows)
        comp_storage_res = store.upsert_model_comparison_series(all_comparison_rows)
        xai_storage_res = store.upsert_shap_lime_explanations(all_explanation_rows)

        completed_at = datetime.now(timezone.utc)
        run_metadata = {
            "days": days,
            "forecast_horizon_hours": H,
            "total_history_rows": int(len(history_df)),
            "trainable_rows": int(len(trainable_df)),
            "train_rows": int(len(split.train_df)),
            "val_rows": int(len(split.val_df)),
            "test_rows": int(len(split.test_df)),
            "feature_count": int(len(split.feature_columns)),
            "metrics_trained": target_metrics,
            "families_trained": target_families,
            "eda_doc_path": eda_doc_path,
            "storage": {
                "eda_summaries": eda_storage_res,
                "model_registry_metrics": reg_storage_res,
                "model_comparison_series": comp_storage_res,
                "shap_lime_explanations": xai_storage_res,
            },
            "metrics": metric_summaries,
        }
        store.record_pipeline_run(
            run_id=resolved_run_id,
            pipeline_type="training",
            status="completed",
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            metadata=run_metadata,
        )
        return {
            "train_run_id": resolved_run_id,
            "status": "completed",
            **run_metadata,
        }
    except Exception as exc:
        safe_record_pipeline_failure(
            store,
            run_id=resolved_run_id,
            pipeline_type="training",
            started_at_utc=started_at,
            error=exc,
            logger=logger,
        )
        raise


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Run SkyCast Component 3 ML training pipeline.")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional lookback window in days when reading training history.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=f"Optional subset of target metrics from {TARGET_METRICS}.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help=f"Optional subset of model families from {MODEL_FAMILIES}.",
    )
    parser.add_argument(
        "--save-eda-json",
        action="store_true",
        help="Persist docs/eda_summary.json in addition to storage upsert.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force local SQLite offline mode regardless of cloud env credentials.",
    )
    args = parser.parse_args(argv)

    if args.offline:
        cfg = cfg.model_copy(update={"skycast_offline_mode": "true"})

    result = run_training(
        days=args.days,
        metrics=args.metrics,
        families=args.families,
        settings=cfg,
        save_eda_doc=args.save_eda_json,
    )
    logger.info(
        "Training completed: train_run_id=%s train=%s val=%s test=%s metrics=%s",
        result["train_run_id"],
        result["train_rows"],
        result["val_rows"],
        result["test_rows"],
        list(result["metrics"].keys()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
