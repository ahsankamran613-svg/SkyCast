"""Classical ML multi-horizon models: Ridge Regression and Random Forest Regressor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from skycast.config.settings import Settings
from skycast.ml.evaluate import (
    ChronologicalSplit,
    clamp_predictions_and_bounds,
    extract_target_matrix,
    extract_wind_component_targets,
    reconstruct_wind_direction,
)
from skycast.ml.models.base import (
    BaseForecastModel,
    ForecastPrediction,
    _init_common_from_metadata,
)


def _predict_2d(estimator: Any, X: np.ndarray, horizon_hours: int) -> np.ndarray:
    raw = np.asarray(estimator.predict(X), dtype=np.float64)
    return raw.reshape(len(X), horizon_hours)


class _ClassicalMultiHorizonBase(BaseForecastModel):
    """Shared multi-horizon training, circular wind, and bundle persistence for sklearn models."""

    def __init__(self, metric: str, *, settings: Settings | None = None) -> None:
        super().__init__(metric, settings=settings)
        self.estimator: Any = None
        self.estimator_sin: Any = None
        self.estimator_cos: Any = None

    def _build_estimator(self) -> Any:
        raise NotImplementedError

    def _fit_estimator(self, X_train: np.ndarray, Y_train: np.ndarray) -> Any:
        est = self._build_estimator()
        y_fit = Y_train[:, 0] if Y_train.shape[1] == 1 else Y_train
        est.fit(X_train, y_fit)
        return est

    def fit(self, split: ChronologicalSplit) -> _ClassicalMultiHorizonBase:
        self.feature_columns = list(split.feature_columns)
        self.scaler = split.scaler
        X_train = np.asarray(split.X_train_scaled, dtype=np.float64)
        horizons = range(1, self.horizon_hours + 1)

        if self.is_circular:
            y_sin, y_cos = extract_wind_component_targets(split.train_df, horizons=horizons)
            self.estimator_sin = self._fit_estimator(X_train, y_sin)
            self.estimator_cos = self._fit_estimator(X_train, y_cos)
            self.estimator = None
        else:
            y_train = extract_target_matrix(split.train_df, self.metric, horizons=horizons)
            self.estimator = self._fit_estimator(X_train, y_train)
            self.estimator_sin = None
            self.estimator_cos = None

        raw_train_pred = self.predict_point_matrix_for_explain(X_train)
        self._fit_empirical_quantiles(split.train_df, raw_train_pred)
        self.is_fitted = True
        return self

    def predict_point_matrix_for_explain(self, X_scaled: np.ndarray) -> np.ndarray:
        self._require_fitted_estimators()
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        if self.is_circular:
            sin_pred = _predict_2d(self.estimator_sin, X_arr, self.horizon_hours)
            cos_pred = _predict_2d(self.estimator_cos, X_arr, self.horizon_hours)
            return np.asarray(reconstruct_wind_direction(sin_pred, cos_pred), dtype=np.float64)
        return _predict_2d(self.estimator, X_arr, self.horizon_hours)

    def _require_fitted_estimators(self) -> None:
        if self.is_circular:
            if self.estimator_sin is None or self.estimator_cos is None:
                raise RuntimeError(f"{self.family_name}({self.metric}) estimators not fitted.")
        elif self.estimator is None:
            raise RuntimeError(f"{self.family_name}({self.metric}) estimator not fitted.")

    def predict(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray | None = None,
        *,
        context_df: pd.DataFrame | None = None,
        context_X_scaled: np.ndarray | None = None,
    ) -> ForecastPrediction:
        del context_df, context_X_scaled
        self._require_fitted()
        X = self._prepare_scaled_features(df, X_scaled)
        raw_pred = self.predict_point_matrix_for_explain(X)
        return self._apply_quantiles(raw_pred)

    def _save_model_artifacts(self, out_dir: Path) -> dict[str, Any] | None:
        if self.is_circular:
            joblib.dump(self.estimator_sin, out_dir / "estimator_sin.joblib")
            joblib.dump(self.estimator_cos, out_dir / "estimator_cos.joblib")
        else:
            joblib.dump(self.estimator, out_dir / "estimator.joblib")
        return None

    @classmethod
    def _load_from_dir(
        cls,
        bundle_dir: Path,
        meta: Mapping[str, Any],
        settings: Settings,
    ) -> _ClassicalMultiHorizonBase:
        model = cls(str(meta["metric"]), settings=settings)
        _init_common_from_metadata(model, bundle_dir, meta)
        if model.is_circular:
            model.estimator_sin = joblib.load(bundle_dir / "estimator_sin.joblib")
            model.estimator_cos = joblib.load(bundle_dir / "estimator_cos.joblib")
        else:
            model.estimator = joblib.load(bundle_dir / "estimator.joblib")
        return model


class RidgeModel(_ClassicalMultiHorizonBase):
    """Multi-output Ridge regression configured strictly from `Settings`."""

    family_name = "Ridge"

    def __init__(self, metric: str, *, settings: Settings | None = None) -> None:
        super().__init__(metric, settings=settings)
        self.hyperparameters = {
            "alpha": float(self.settings.ridge_alpha),
            "random_seed": int(self.settings.random_seed),
            "horizon_hours": self.horizon_hours,
            "conf_alpha": float(self.settings.sarimax_conf_alpha),
        }

    def _build_estimator(self) -> Ridge:
        return Ridge(
            alpha=float(self.settings.ridge_alpha),
            random_state=int(self.settings.random_seed),
        )


class RandomForestModel(_ClassicalMultiHorizonBase):
    """Multi-output Random Forest Regressor configured strictly from `Settings`."""

    family_name = "RandomForest"

    def __init__(self, metric: str, *, settings: Settings | None = None) -> None:
        super().__init__(metric, settings=settings)
        self.hyperparameters = {
            "n_estimators": int(self.settings.rf_n_estimators),
            "max_depth": int(self.settings.rf_max_depth),
            "min_samples_leaf": int(self.settings.rf_min_samples_leaf),
            "random_seed": int(self.settings.random_seed),
            "horizon_hours": self.horizon_hours,
            "conf_alpha": float(self.settings.sarimax_conf_alpha),
        }

    def _build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=int(self.settings.rf_n_estimators),
            max_depth=int(self.settings.rf_max_depth),
            min_samples_leaf=int(self.settings.rf_min_samples_leaf),
            random_state=int(self.settings.random_seed),
            n_jobs=1,
        )

    def predict(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray | None = None,
        *,
        context_df: pd.DataFrame | None = None,
        context_X_scaled: np.ndarray | None = None,
    ) -> ForecastPrediction:
        del context_df, context_X_scaled
        self._require_fitted()
        X = self._prepare_scaled_features(df, X_scaled)
        raw_pred = self.predict_point_matrix_for_explain(X)

        if self.is_circular:
            return self._apply_quantiles(raw_pred)

        if self.q_low is None or self.q_high is None:
            raise RuntimeError(f"RandomForestModel({self.metric}) missing residual quantiles.")

        tree_preds = np.stack(
            [_predict_2d(tree, X, self.horizon_hours) for tree in self.estimator.estimators_],
            axis=0,
        )
        if tree_preds.shape[0] >= 2:
            tree_std = np.std(tree_preds, axis=0, ddof=1)
        else:
            tree_std = np.zeros_like(raw_pred)

        z = float(norm.ppf(1.0 - float(self.settings.sarimax_conf_alpha) / 2.0))
        pred_c, _, _ = clamp_predictions_and_bounds(self.metric, raw_pred, raw_pred, raw_pred)
        raw_low = np.minimum(pred_c + self.q_low, pred_c - z * tree_std)
        raw_high = np.maximum(pred_c + self.q_high, pred_c + z * tree_std)
        pred_out, low_out, high_out = clamp_predictions_and_bounds(
            self.metric, pred_c, raw_low, raw_high
        )
        return ForecastPrediction(predicted=pred_out, lower=low_out, upper=high_out)
