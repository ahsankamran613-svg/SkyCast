"""Champion model selection, Hopsworks Model Registry promotion, and local artifact registry."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from skycast.config.settings import METRIC_SPECS, TARGET_METRICS, Settings, get_settings
from skycast.data.storage import SkyCastStorage, StorageError
from skycast.features.engineer import sanitize_json_scalar
from skycast.ml.models.base import BaseForecastModel, load_model_bundle


def resolve_registry_model_name(metric: str, settings: Settings | None = None) -> str:
    """Resolve Hopsworks/local model name from `cfg.hopsworks_model_name_template`."""
    if metric not in METRIC_SPECS:
        raise ValueError(f"Unknown metric '{metric}'. Expected one of {TARGET_METRICS}.")
    cfg = settings or get_settings()
    return cfg.hopsworks_model_name_template.format(metric=metric)


def _is_hopsworks_registry_active(storage: SkyCastStorage, cfg: Settings) -> bool:
    return (not cfg.is_offline_mode) and cfg.has_hopsworks_credentials and (
        storage.storage_mode in {"cloud_hopsworks", "cloud_dual"}
    )


def _is_not_found_error(exc: Exception) -> bool:
    if type(exc).__name__ == "MockRestAPIError":
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    msg = str(exc).lower()
    if "http code: 404" in msg or ("404" in msg and "not found" in msg):
        return True
    if "does not exist" in msg or "not found in registry" in msg:
        return True
    return False


def _local_registry_dir(model_name: str, cfg: Settings) -> Path:
    return cfg.skycast_local_data_dir / "model_registry" / model_name


def get_current_champion_info(
    metric: str,
    storage: SkyCastStorage,
    *,
    settings: Settings | None = None,
) -> tuple[float | None, str | None, str | None]:
    """Return `(val_rmse, version, model_family)` for the current champion, or `(None, None, None)` on cold start."""
    cfg = settings or storage.settings or get_settings()
    model_name = resolve_registry_model_name(metric, cfg)

    if _is_hopsworks_registry_active(storage, cfg):
        mr = storage.get_hopsworks_model_registry()
        try:
            best = mr.get_best_model(model_name, metric="val_rmse", direction="min")
        except Exception as exc:
            if _is_not_found_error(exc):
                return None, None, None
            raise StorageError(f"Hopsworks Model Registry query failed for '{model_name}': {exc}") from exc

        if best is None:
            return None, None, None
        metrics_dict = getattr(best, "training_metrics", None) or {}
        raw_rmse = metrics_dict.get("val_rmse")
        if raw_rmse is None:
            return None, str(getattr(best, "version", "")) or None, None
        val_rmse = float(raw_rmse)
        if not math.isfinite(val_rmse):
            return None, str(getattr(best, "version", "")) or None, None
        tags = getattr(best, "tags", None)
        family = tags.get("model_family") if isinstance(tags, Mapping) else None
        ver = getattr(best, "version", None)
        return val_rmse, (None if ver is None else str(ver)), (None if family is None else str(family))

    champ_file = _local_registry_dir(model_name, cfg) / "champion.json"
    if not champ_file.exists():
        return None, None, None
    data = json.loads(champ_file.read_text(encoding="utf-8"))
    raw_rmse = data.get("val_rmse")
    val_rmse = float(raw_rmse) if raw_rmse is not None and math.isfinite(float(raw_rmse)) else None
    ver = data.get("version")
    fam = data.get("model_family")
    return val_rmse, (None if ver is None else str(ver)), (None if fam is None else str(fam))


def promote_champion_if_better(
    *,
    metric: str,
    candidate_model: BaseForecastModel,
    candidate_metrics: Mapping[str, Any],
    naive_val_rmse: float,
    storage: SkyCastStorage,
    input_example: np.ndarray | list[Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Promote candidate iff `val_rmse` is finite, `< Naive24h val_rmse`, and `< current champion val_rmse`."""
    cfg = settings or storage.settings or get_settings()
    model_name = resolve_registry_model_name(metric, cfg)
    cand_val_rmse = float(candidate_metrics["val_rmse"])
    naive_rmse = float(naive_val_rmse)

    if not math.isfinite(cand_val_rmse) or not math.isfinite(naive_rmse):
        return {
            "is_champion": False,
            "promoted_to_registry": False,
            "model_version": None,
            "model_name": model_name,
            "reason": "non_finite_val_rmse",
        }

    if not (cand_val_rmse < naive_rmse):
        return {
            "is_champion": False,
            "promoted_to_registry": False,
            "model_version": None,
            "model_name": model_name,
            "reason": "did_not_beat_naive_baseline",
        }

    curr_val_rmse, curr_version, _ = get_current_champion_info(metric, storage, settings=cfg)
    if curr_val_rmse is not None and not (cand_val_rmse < curr_val_rmse):
        return {
            "is_champion": False,
            "promoted_to_registry": False,
            "model_version": None,
            "model_name": model_name,
            "previous_champion_val_rmse": curr_val_rmse,
            "previous_champion_version": curr_version,
            "reason": "did_not_beat_current_champion",
        }

    metrics_payload = sanitize_json_scalar(
        {
            "val_rmse": cand_val_rmse,
            "val_mae": float(candidate_metrics["val_mae"]),
            "val_r2": float(candidate_metrics["val_r2"]),
            "test_rmse": float(candidate_metrics["test_rmse"]),
            "test_mae": float(candidate_metrics["test_mae"]),
            "test_r2": float(candidate_metrics["test_r2"]),
            "improvement_over_naive_pct": float(candidate_metrics["improvement_over_naive_pct"]),
        }
    )
    json.dumps(metrics_payload, allow_nan=False)

    bundle_extra_meta = {
        "train_run_id": candidate_metrics.get("train_run_id"),
        "trained_at_utc": candidate_metrics.get("trained_at_utc"),
        "training_samples": candidate_metrics.get("training_samples"),
        "metrics": metrics_payload,
    }

    if _is_hopsworks_registry_active(storage, cfg):
        mr = storage.get_hopsworks_model_registry()
        sanitized_example = None
        if input_example is not None:
            ex_list = np.asarray(input_example, dtype=np.float64).tolist()
            sanitized_example = sanitize_json_scalar(ex_list)

        with tempfile.TemporaryDirectory(prefix=f"skycast_{model_name}_") as tmp_dir:
            stage_dir = Path(tmp_dir) / "bundle"
            candidate_model.save(stage_dir, metadata=bundle_extra_meta)
            try:
                created = mr.python.create_model(
                    name=model_name,
                    metrics=metrics_payload,
                    description=(
                        f"SkyCast {candidate_model.family_name} champion for Lahore {metric} "
                        f"(H={candidate_model.horizon_hours}h)"
                    ),
                    input_example=sanitized_example,
                )
                saved = created.save(str(stage_dir))
            except Exception as exc:
                raise StorageError(
                    f"Hopsworks Model Registry save failed for '{model_name}': {exc}"
                ) from exc

        promoted_version = str(getattr(saved, "version", "1"))
        return {
            "is_champion": True,
            "promoted_to_registry": True,
            "model_version": promoted_version,
            "model_name": model_name,
            "previous_champion_val_rmse": curr_val_rmse,
            "previous_champion_version": curr_version,
            "reason": "promoted_to_hopsworks",
        }

    # Local offline / non-Hopsworks artifact registry mirror
    reg_dir = _local_registry_dir(model_name, cfg)
    reg_dir.mkdir(parents=True, exist_ok=True)
    next_ver = (int(curr_version) + 1) if (curr_version is not None and curr_version.isdigit()) else 1
    ver_dir = reg_dir / f"v{next_ver}"
    if ver_dir.exists():
        shutil.rmtree(ver_dir)
    candidate_model.save(ver_dir, metadata={**bundle_extra_meta, "version": next_ver})

    champ_record = sanitize_json_scalar(
        {
            "model_name": model_name,
            "metric": metric,
            "model_family": candidate_model.family_name,
            "version": str(next_ver),
            "bundle_dir": str(ver_dir),
            "val_rmse": cand_val_rmse,
            "metrics": metrics_payload,
            "train_run_id": candidate_metrics.get("train_run_id"),
            "trained_at_utc": candidate_metrics.get("trained_at_utc"),
        }
    )
    (reg_dir / "champion.json").write_text(
        json.dumps(champ_record, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    return {
        "is_champion": True,
        "promoted_to_registry": False,
        "model_version": str(next_ver),
        "model_name": model_name,
        "previous_champion_val_rmse": curr_val_rmse,
        "previous_champion_version": curr_version,
        "reason": "promoted_to_local_registry",
    }


def load_champion_bundle(
    metric: str,
    storage: SkyCastStorage,
    *,
    settings: Settings | None = None,
    download_dir: Path | None = None,
) -> tuple[BaseForecastModel, str | None] | None:
    """Load the active champion model bundle for `metric`, returning `None` on cold start."""
    cfg = settings or storage.settings or get_settings()
    model_name = resolve_registry_model_name(metric, cfg)

    if _is_hopsworks_registry_active(storage, cfg):
        mr = storage.get_hopsworks_model_registry()
        try:
            best = mr.get_best_model(model_name, metric="val_rmse", direction="min")
        except Exception as exc:
            if _is_not_found_error(exc):
                return None
            raise StorageError(f"Hopsworks Model Registry read failed for '{model_name}': {exc}") from exc

        if best is None:
            return None
        ver = getattr(best, "version", None)
        version_str = None if ver is None else str(ver)

        if download_dir is not None:
            target_dir = Path(download_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            dl_path = best.download(str(target_dir))
            return load_model_bundle(dl_path, settings=cfg), version_str

        with tempfile.TemporaryDirectory(prefix=f"skycast_dl_{model_name}_") as tmp_dir:
            dl_path = best.download(str(Path(tmp_dir) / "bundle"))
            return load_model_bundle(dl_path, settings=cfg), version_str

    champ_file = _local_registry_dir(model_name, cfg) / "champion.json"
    if not champ_file.exists():
        return None
    data = json.loads(champ_file.read_text(encoding="utf-8"))
    bundle_dir = Path(str(data["bundle_dir"]))
    if not bundle_dir.exists():
        return None
    version_str = str(data.get("version")) if data.get("version") is not None else None
    return load_model_bundle(bundle_dir, settings=cfg), version_str
