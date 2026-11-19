"""24-hour naive persistence baseline model (`Naive24h`)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skycast.config.settings import Settings
from skycast.ml.evaluate import ChronologicalSplit, reconstruct_wind_direction
from skycast.ml.models.base import (
    BaseForecastModel,
    ForecastPrediction,
    _init_common_from_metadata,
)


class Naive24hModel(BaseForecastModel):
    """Multi-horizon persistence baseline predicting `y_{t+h} = y_t` for `h=1..H`."""

    family_name = "Naive24h"

    def __init__(self, metric: str, *, settings: Settings | None = None) -> None:
        super().__init__(metric, settings=settings)
        self.hyperparameters = {
            "strategy": "persistence_origin_t",
            "horizon_hours": self.horizon_hours,
            "conf_alpha": float(self.settings.sarimax_conf_alpha),
        }

    def _origin_series(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            raise ValueError("Input DataFrame must be non-empty for Naive24h prediction.")
        if self.is_circular:
            if "wind_direction_10m" in df.columns:
                arr = pd.to_numeric(df["wind_direction_10m"], errors="raise").to_numpy(dtype=np.float64)
            elif "wind_dir_sin" in df.columns and "wind_dir_cos" in df.columns:
                sin_v = pd.to_numeric(df["wind_dir_sin"], errors="raise").to_numpy(dtype=np.float64)
                cos_v = pd.to_numeric(df["wind_dir_cos"], errors="raise").to_numpy(dtype=np.float64)
                arr = np.asarray(reconstruct_wind_direction(sin_v, cos_v), dtype=np.float64)
            else:
                raise ValueError(
                    "DataFrame must contain 'wind_direction_10m' or ('wind_dir_sin', 'wind_dir_cos')."
                )
        else:
            if self.metric not in df.columns:
                raise ValueError(f"DataFrame missing target origin column '{self.metric}'.")
            arr = pd.to_numeric(df[self.metric], errors="raise").to_numpy(dtype=np.float64)

        if not np.isfinite(arr).all():
            raise ValueError(f"Origin series for '{self.metric}' contains NaN or Inf values.")
        return arr

    def _raw_predict_df(self, df: pd.DataFrame) -> np.ndarray:
        origin = self._origin_series(df)
        return np.repeat(origin[:, None], self.horizon_hours, axis=1)

    def fit(self, split: ChronologicalSplit) -> Naive24hModel:
        self.feature_columns = list(split.feature_columns)
        self.scaler = split.scaler
        raw_train_pred = self._raw_predict_df(split.train_df)
        self._fit_empirical_quantiles(split.train_df, raw_train_pred)
        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray | None = None,
        *,
        context_df: pd.DataFrame | None = None,
        context_X_scaled: np.ndarray | None = None,
    ) -> ForecastPrediction:
        del X_scaled, context_df, context_X_scaled
        self._require_fitted()
        raw_pred = self._raw_predict_df(df)
        return self._apply_quantiles(raw_pred)

    def predict_point_matrix_for_explain(self, X_scaled: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self.scaler is None or not self.feature_columns:
            raise RuntimeError("Naive24hModel requires scaler and feature_columns for explain.")
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        X_raw = self.scaler.inverse_transform(X_arr)
        if self.is_circular:
            sin_idx = self.feature_columns.index("wind_dir_sin")
            cos_idx = self.feature_columns.index("wind_dir_cos")
            origin = np.asarray(
                reconstruct_wind_direction(X_raw[:, sin_idx], X_raw[:, cos_idx]),
                dtype=np.float64,
            )
        else:
            col_idx = self.feature_columns.index(self.metric)
            origin = X_raw[:, col_idx]
        return np.repeat(origin[:, None], self.horizon_hours, axis=1)

    def _save_model_artifacts(self, out_dir: Path) -> dict[str, Any] | None:
        del out_dir
        return None

    @classmethod
    def _load_from_dir(
        cls,
        bundle_dir: Path,
        meta: Mapping[str, Any],
        settings: Settings,
    ) -> Naive24hModel:
        model = cls(str(meta["metric"]), settings=settings)
        _init_common_from_metadata(model, bundle_dir, meta)
        return model
