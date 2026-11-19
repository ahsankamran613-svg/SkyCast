"""Forecasting model families: Naive24h, SARIMAX, RandomForest, Ridge, and PyTorch LSTM."""

from __future__ import annotations

from skycast.ml.models.base import (
    BaseForecastModel,
    ForecastPrediction,
    load_model_bundle,
    predict_from_bundle,
)
from skycast.ml.models.baseline import Naive24hModel
from skycast.ml.models.classical import RandomForestModel, RidgeModel
from skycast.ml.models.deep_learning import LSTMModel, WeatherLSTMNet
from skycast.ml.models.statistical import SARIMAXModel

MODEL_CLASS_BY_FAMILY: dict[str, type[BaseForecastModel]] = {
    "Naive24h": Naive24hModel,
    "SARIMAX": SARIMAXModel,
    "RandomForest": RandomForestModel,
    "Ridge": RidgeModel,
    "LSTM": LSTMModel,
}

__all__ = [
    "BaseForecastModel",
    "ForecastPrediction",
    "LSTMModel",
    "MODEL_CLASS_BY_FAMILY",
    "Naive24hModel",
    "RandomForestModel",
    "RidgeModel",
    "SARIMAXModel",
    "WeatherLSTMNet",
    "load_model_bundle",
    "predict_from_bundle",
]
