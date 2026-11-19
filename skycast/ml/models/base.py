"""Shared forecasting model interface, output contract, and artifact bundle serialization."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from skycast.config.settings import METRIC_SPECS, TARGET_METRICS, Settings, get_settings
from skycast.features.engineer import sanitize_json_scalar
from skycast.ml.evaluate import (
    ChronologicalSplit,
    apply_residual_quantiles,
    clamp_predictions_and_bounds,
    compute_empirical_residual_quantiles,
    extract_target_matrix,
    reconstruct_wind_direction,
)


class ForecastPrediction(NamedTuple):
    """Multi-horizon `(n_samples, H)` point forecasts and clamped prediction intervals."""

    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _unwrap_around_reference(angles_deg: np.ndarray, reference_deg: float) -> np.ndarray:
    """Unwrap angles smoothly around `reference_deg` to avoid 359 <-> 1 discontinuities."""
    return reference_deg + (((angles_deg - reference_deg + 180.0) % 360.0) - 180.0)


def _compute_circular_mean(angles_deg: np.ndarray) -> float:
    rad = np.deg2rad(angles_deg)
    sin_m = float(np.mean(np.sin(rad)))
    cos_m = float(np.mean(np.cos(rad)))
    if abs(sin_m) < 1e-12 and abs(cos_m) < 1e-12:
        return 0.0
    deg = float(np.degrees(np.arctan2(sin_m, cos_m)) % 360.0)
    return 0.0 if deg >= 360.0 else deg


def _extract_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required feature column(s): {missing}")
    arrays: list[np.ndarray] = []
    for col in feature_columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s.dtype):
            arr = s.to_numpy(dtype=np.float64)
        else:
            arr = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"Feature column '{col}' contains NaN or Inf values.")
        arrays.append(arr)
    return np.column_stack(arrays)


class BaseForecastModel(ABC):
    """Common interface for all 5 SkyCast forecasting model families."""

    family_name: str = ""

    def __init__(
        self,
        metric: str,
        *,
        settings: Settings | None = None,
    ) -> None:
        if metric not in METRIC_SPECS:
            raise ValueError(f"Unknown metric '{metric}'. Expected one of {TARGET_METRICS}.")
        self.settings = settings or get_settings()
        self.metric = metric
        self.horizon_hours = int(self.settings.forecast_horizon_hours)
        if self.horizon_hours < 1:
            raise ValueError(f"forecast_horizon_hours must be >= 1, got {self.horizon_hours}.")
        self.is_circular = METRIC_SPECS[metric].is_circular
        self.feature_columns: list[str] = []
        self.scaler: StandardScaler | None = None
        self.q_low: np.ndarray | None = None
        self.q_high: np.ndarray | None = None
        self.hyperparameters: dict[str, Any] = {}
        self.is_fitted: bool = False

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.family_name}({self.metric}) must be fitted before prediction.")

    def _prepare_scaled_features(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray | None,
    ) -> np.ndarray:
        if X_scaled is not None:
            arr = np.asarray(X_scaled, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] != len(df):
                raise ValueError(
                    f"X_scaled shape {arr.shape} incompatible with DataFrame length {len(df)}."
                )
            if self.feature_columns and arr.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"X_scaled feature count {arr.shape[1]} != expected {len(self.feature_columns)}."
                )
            if not np.isfinite(arr).all():
                raise ValueError("X_scaled contains NaN or Inf values.")
            return arr

        if self.scaler is None or not self.feature_columns:
            raise RuntimeError(
                f"{self.family_name}({self.metric}) has no stored scaler/feature_columns."
            )
        X_raw = _extract_feature_matrix(df, self.feature_columns)
        return self.scaler.transform(X_raw)

    def _fit_empirical_quantiles(
        self,
        train_df: pd.DataFrame,
        raw_train_pred: np.ndarray,
    ) -> None:
        y_train = extract_target_matrix(
            train_df,
            self.metric,
            horizons=range(1, self.horizon_hours + 1),
        )
        raw_clamped, _, _ = clamp_predictions_and_bounds(
            self.metric, raw_train_pred, raw_train_pred, raw_train_pred
        )
        self.q_low, self.q_high = compute_empirical_residual_quantiles(
            y_train,
            raw_clamped,
            alpha=float(self.settings.sarimax_conf_alpha),
            settings=self.settings,
            metric=self.metric,
        )

    def _apply_quantiles(self, raw_pred: np.ndarray) -> ForecastPrediction:
        if self.q_low is None or self.q_high is None:
            raise RuntimeError(f"{self.family_name}({self.metric}) missing residual quantiles.")
        pred_c, low_c, high_c = apply_residual_quantiles(
            self.metric, raw_pred, self.q_low, self.q_high
        )
        return ForecastPrediction(predicted=pred_c, lower=low_c, upper=high_c)

    @abstractmethod
    def fit(self, split: ChronologicalSplit) -> BaseForecastModel:
        """Fit model state exclusively on `split.train_df` / `split.X_train_scaled`."""

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray | None = None,
        *,
        context_df: pd.DataFrame | None = None,
        context_X_scaled: np.ndarray | None = None,
    ) -> ForecastPrediction:
        """Emit `(n_samples, H)` point forecasts and clamped intervals."""

    @abstractmethod
    def predict_point_matrix_for_explain(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict raw `(n_samples, H)` point matrix solely from scaled feature rows."""

    def predict_point_for_explain(
        self,
        X_scaled: np.ndarray,
        *,
        reference_angle_deg: float | None = None,
    ) -> np.ndarray:
        """Predict primary horizon `h=H` vector (unwrapped if circular) for SHAP/LIME."""
        mat = self.predict_point_matrix_for_explain(X_scaled)
        vec = np.asarray(mat[:, self.horizon_hours - 1], dtype=np.float64)
        if not self.is_circular:
            spec = METRIC_SPECS[self.metric]
            return np.clip(vec, spec.min_value, spec.max_value)
        ref = _compute_circular_mean(vec) if reference_angle_deg is None else float(reference_angle_deg)
        return _unwrap_around_reference(vec, ref)

    def _base_metadata(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "family_name": self.family_name,
            "model_family": self.family_name,
            "metric": self.metric,
            "horizon_hours": self.horizon_hours,
            "is_circular": self.is_circular,
            "feature_columns": list(self.feature_columns),
            "hyperparameters": sanitize_json_scalar(dict(self.hyperparameters)),
            "q_low": None if self.q_low is None else [float(v) for v in self.q_low],
            "q_high": None if self.q_high is None else [float(v) for v in self.q_high],
            "sarimax_conf_alpha": float(self.settings.sarimax_conf_alpha),
            "random_seed": int(self.settings.random_seed),
        }
        if extra:
            for k, v in extra.items():
                payload[str(k)] = sanitize_json_scalar(v)
        return payload

    def save(
        self,
        bundle_dir: Path | str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist model weights, train scaler, quantiles, and metadata into `bundle_dir`."""
        self._require_fitted()
        out_dir = Path(bundle_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.scaler is not None:
            joblib.dump(self.scaler, out_dir / "scaler.joblib")
        if self.q_low is not None and self.q_high is not None:
            np.savez(
                out_dir / "quantiles.npz",
                q_low=np.asarray(self.q_low, dtype=np.float64),
                q_high=np.asarray(self.q_high, dtype=np.float64),
            )
        extra_meta = self._save_model_artifacts(out_dir)
        merged_extra: dict[str, Any] = {}
        if extra_meta:
            merged_extra.update(extra_meta)
        if metadata:
            merged_extra.update(metadata)
        meta_payload = self._base_metadata(merged_extra)
        serialized = json.dumps(sanitize_json_scalar(meta_payload), allow_nan=False, indent=2)
        (out_dir / "metadata.json").write_text(serialized, encoding="utf-8")
        return out_dir

    @abstractmethod
    def _save_model_artifacts(self, out_dir: Path) -> dict[str, Any] | None:
        """Save family-specific artifacts under `out_dir` and return optional metadata additions."""

    @classmethod
    @abstractmethod
    def _load_from_dir(
        cls,
        bundle_dir: Path,
        meta: Mapping[str, Any],
        settings: Settings,
    ) -> BaseForecastModel:
        """Reconstruct a fitted model instance from `bundle_dir` and parsed `metadata.json`."""


def _init_common_from_metadata(
    model: BaseForecastModel,
    bundle_dir: Path,
    meta: Mapping[str, Any],
) -> None:
    model.feature_columns = [str(c) for c in meta.get("feature_columns", [])]
    scaler_path = bundle_dir / "scaler.joblib"
    if scaler_path.exists():
        model.scaler = joblib.load(scaler_path)
    quantiles_path = bundle_dir / "quantiles.npz"
    if quantiles_path.exists():
        with np.load(quantiles_path, allow_pickle=False) as q_npz:
            model.q_low = np.asarray(q_npz["q_low"], dtype=np.float64)
            model.q_high = np.asarray(q_npz["q_high"], dtype=np.float64)
    else:
        q_low = meta.get("q_low")
        q_high = meta.get("q_high")
        model.q_low = None if q_low is None else np.asarray(q_low, dtype=np.float64)
        model.q_high = None if q_high is None else np.asarray(q_high, dtype=np.float64)
    model.hyperparameters = dict(meta.get("hyperparameters") or {})
    model.is_fitted = True


def load_model_bundle(
    bundle_dir: Path | str,
    *,
    settings: Settings | None = None,
) -> BaseForecastModel:
    """Load any persisted SkyCast forecasting model bundle from `bundle_dir`."""
    path = Path(bundle_dir)
    meta_file = path / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Model bundle metadata not found: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    family = str(meta.get("family_name") or meta.get("model_family") or "")
    metric = str(meta["metric"])

    base_cfg = settings or get_settings()
    cfg = base_cfg.model_copy(
        update={
            "forecast_horizon_hours": int(meta.get("horizon_hours", base_cfg.forecast_horizon_hours)),
            "sarimax_conf_alpha": float(meta.get("sarimax_conf_alpha", base_cfg.sarimax_conf_alpha)),
            "random_seed": int(meta.get("random_seed", base_cfg.random_seed)),
        }
    )

    from skycast.ml.models import MODEL_CLASS_BY_FAMILY

    if family not in MODEL_CLASS_BY_FAMILY:
        raise ValueError(f"Unsupported model_family '{family}' in bundle {path}.")
    model_cls = MODEL_CLASS_BY_FAMILY[family]
    if metric not in METRIC_SPECS:
        raise ValueError(f"Unsupported metric '{metric}' in bundle {path}.")
    return model_cls._load_from_dir(path, meta, cfg)


def predict_from_bundle(
    bundle_dir: Path | str,
    df: pd.DataFrame,
    *,
    context_df: pd.DataFrame | None = None,
    settings: Settings | None = None,
) -> ForecastPrediction:
    """Load a persisted model bundle and run calibrated inference on `df`."""
    model = load_model_bundle(bundle_dir, settings=settings)
    return model.predict(df, context_df=context_df)
