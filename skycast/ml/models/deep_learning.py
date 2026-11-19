"""PyTorch CPU LSTM multi-horizon forecasting model."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from skycast.config.settings import Settings
from skycast.ml.evaluate import (
    ChronologicalSplit,
    extract_target_matrix,
    extract_wind_component_targets,
    reconstruct_wind_direction,
)
from skycast.ml.models.base import (
    BaseForecastModel,
    ForecastPrediction,
    _init_common_from_metadata,
)


class WeatherLSTMNet(nn.Module):
    """2-layer CPU LSTM network with dropout and direct multi-horizon projection."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = self.dropout(out[:, -1, :])
        return self.head(last_hidden)


def _build_causal_sequences(
    X_scaled: np.ndarray,
    lookback: int,
    context_X_scaled: np.ndarray | None = None,
) -> np.ndarray:
    """Build strictly causal `(N, lookback, F)` windows ending at each row `0..N-1`."""
    n_samples, n_features = X_scaled.shape
    if n_samples == 0:
        return np.zeros((0, lookback, n_features), dtype=np.float32)

    if lookback <= 1:
        return X_scaled[:, None, :].astype(np.float32)

    pad_needed = lookback - 1
    if context_X_scaled is not None and len(context_X_scaled) > 0:
        ctx_tail = np.asarray(context_X_scaled[-pad_needed:], dtype=np.float64)
        if len(ctx_tail) < pad_needed:
            left_pad = np.repeat(ctx_tail[:1], pad_needed - len(ctx_tail), axis=0)
            prefix = np.vstack([left_pad, ctx_tail])
        else:
            prefix = ctx_tail
    else:
        prefix = np.repeat(X_scaled[:1], pad_needed, axis=0)

    combined = np.vstack([prefix, X_scaled]).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(combined, window_shape=lookback, axis=0)
    # sliding_window_view returns (n_samples, n_features, lookback); copy into writable (n_samples, lookback, n_features)
    return np.array(np.swapaxes(windows, 1, 2), dtype=np.float32, copy=True)


class LSTMModel(BaseForecastModel):
    """Multi-horizon PyTorch CPU LSTM configured strictly from `Settings`."""

    family_name = "LSTM"

    def __init__(self, metric: str, *, settings: Settings | None = None) -> None:
        super().__init__(metric, settings=settings)
        self.lookback_hours = int(self.settings.lstm_lookback_hours)
        if self.lookback_hours < 1:
            raise ValueError(f"lstm_lookback_hours must be >= 1, got {self.lookback_hours}.")
        self.hidden_size = int(self.settings.lstm_hidden_size)
        self.num_layers = int(self.settings.lstm_num_layers)
        self.dropout_rate = float(self.settings.lstm_dropout)
        self.epochs = int(self.settings.lstm_epochs)
        self.batch_size = int(self.settings.lstm_batch_size)
        self.learning_rate = float(self.settings.lstm_learning_rate)
        self.patience = int(self.settings.lstm_early_stopping_patience)
        self.output_size = self.horizon_hours * (2 if self.is_circular else 1)

        self.net: WeatherLSTMNet | None = None
        self.input_size: int = 0
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

        self.hyperparameters = {
            "lookback_hours": self.lookback_hours,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "early_stopping_patience": self.patience,
            "random_seed": int(self.settings.random_seed),
            "horizon_hours": self.horizon_hours,
            "conf_alpha": float(self.settings.sarimax_conf_alpha),
        }

    def _decode_raw_output(self, raw_out: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError(f"LSTMModel({self.metric}) missing target normalization stats.")
        unscaled = raw_out * self.y_std[None, :] + self.y_mean[None, :]
        if self.is_circular:
            sin_p = unscaled[:, : self.horizon_hours]
            cos_p = unscaled[:, self.horizon_hours :]
            return np.asarray(reconstruct_wind_direction(sin_p, cos_p), dtype=np.float64)
        return unscaled

    def _run_forward_batches(self, windows: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError(f"LSTMModel({self.metric}) network not initialized.")
        self.net.eval()
        preds: list[np.ndarray] = []
        eval_batch = max(self.batch_size, 128)
        with torch.no_grad():
            for start in range(0, len(windows), eval_batch):
                xb = torch.from_numpy(windows[start : start + eval_batch])
                out = self.net(xb).cpu().numpy().astype(np.float64)
                preds.append(out)
        raw_out = np.vstack(preds) if preds else np.zeros((0, self.output_size), dtype=np.float64)
        return self._decode_raw_output(raw_out)

    def fit(self, split: ChronologicalSplit) -> LSTMModel:
        seed = int(self.settings.random_seed)
        np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.feature_columns = list(split.feature_columns)
        self.scaler = split.scaler
        X_train = np.asarray(split.X_train_scaled, dtype=np.float64)
        self.input_size = X_train.shape[1]

        horizons = range(1, self.horizon_hours + 1)
        if self.is_circular:
            y_sin, y_cos = extract_wind_component_targets(split.train_df, horizons=horizons)
            Y_train = np.hstack([y_sin, y_cos])
        else:
            Y_train = extract_target_matrix(split.train_df, self.metric, horizons=horizons)

        self.y_mean = np.mean(Y_train, axis=0)
        y_std_raw = np.std(Y_train, axis=0)
        self.y_std = np.where(y_std_raw < 1e-8, 1.0, y_std_raw)
        Y_norm = ((Y_train - self.y_mean[None, :]) / self.y_std[None, :]).astype(np.float32)

        X_seq = _build_causal_sequences(X_train, self.lookback_hours)

        # Internal train-only chronological split for EarlyStopping (zero val/test leakage!)
        n_seq = len(X_seq)
        if n_seq >= 20:
            es_n = max(4, int(round(0.15 * n_seq)))
            embargo = self.horizon_hours if n_seq >= (es_n + self.horizon_hours + 12) else 0
            fit_end = n_seq - es_n - embargo
            es_start = n_seq - es_n
            X_fit, Y_fit = X_seq[:fit_end], Y_norm[:fit_end]
            X_es, Y_es = X_seq[es_start:], Y_norm[es_start:]
        else:
            X_fit, Y_fit = X_seq, Y_norm
            X_es, Y_es = None, None

        self.net = WeatherLSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            output_size=self.output_size,
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        epochs_no_improve = 0
        eff_batch = max(1, min(self.batch_size, len(X_fit)))

        X_es_t = torch.from_numpy(X_es) if X_es is not None else None
        Y_es_t = torch.from_numpy(Y_es) if Y_es is not None else None

        for _epoch in range(self.epochs):
            self.net.train()
            perm = np_rng.permutation(len(X_fit))
            total_train_loss = 0.0
            total_train_count = 0

            for start in range(0, len(X_fit), eff_batch):
                idx = perm[start : start + eff_batch]
                xb = torch.from_numpy(X_fit[idx])
                yb = torch.from_numpy(Y_fit[idx])
                optimizer.zero_grad()
                pred_b = self.net(xb)
                loss = criterion(pred_b, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += float(loss.item()) * len(idx)
                total_train_count += len(idx)

            if X_es_t is not None and Y_es_t is not None:
                self.net.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.net(X_es_t), Y_es_t).item())
                monitor_loss = val_loss
            else:
                monitor_loss = total_train_loss / max(1, total_train_count)

            if monitor_loss < best_loss - 1e-6:
                best_loss = monitor_loss
                best_state = copy.deepcopy(self.net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()

        raw_train_pred = self._run_forward_batches(X_seq)
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
        self._require_fitted()
        X = self._prepare_scaled_features(df, X_scaled)
        ctx_X: np.ndarray | None = None
        if context_df is not None and not context_df.empty:
            use_ctx = True
            if "timestamp_utc" in df.columns and "timestamp_utc" in context_df.columns:
                df_first = pd.to_datetime(df["timestamp_utc"].iloc[0], utc=True)
                ctx_last = pd.to_datetime(context_df["timestamp_utc"].iloc[-1], utc=True)
                gap_sec = (df_first - ctx_last).total_seconds()
                use_ctx = abs(gap_sec - 3600.0) <= 60.0
            if use_ctx:
                ctx_X = self._prepare_scaled_features(context_df, context_X_scaled)
        windows = _build_causal_sequences(X, self.lookback_hours, context_X_scaled=ctx_X)
        raw_pred = self._run_forward_batches(windows)
        return self._apply_quantiles(raw_pred)

    def predict_point_matrix_for_explain(self, X_scaled: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X_arr = np.asarray(X_scaled, dtype=np.float32)
        windows = np.ascontiguousarray(np.repeat(X_arr[:, None, :], self.lookback_hours, axis=1))
        return self._run_forward_batches(windows)

    def _save_model_artifacts(self, out_dir: Path) -> dict[str, Any] | None:
        if self.net is None or self.y_mean is None or self.y_std is None:
            raise RuntimeError(f"Cannot save unfitted LSTMModel({self.metric}).")
        torch.save(self.net.state_dict(), out_dir / "lstm_weights.pt")
        np.savez(
            out_dir / "lstm_stats.npz",
            y_mean=np.asarray(self.y_mean, dtype=np.float64),
            y_std=np.asarray(self.y_std, dtype=np.float64),
        )
        return {
            "lstm_arch": {
                "input_size": int(self.input_size),
                "hidden_size": int(self.hidden_size),
                "num_layers": int(self.num_layers),
                "dropout_rate": float(self.dropout_rate),
                "output_size": int(self.output_size),
                "lookback_hours": int(self.lookback_hours),
            }
        }

    @classmethod
    def _load_from_dir(
        cls,
        bundle_dir: Path,
        meta: Mapping[str, Any],
        settings: Settings,
    ) -> LSTMModel:
        arch = meta["lstm_arch"]
        cfg = settings.model_copy(
            update={
                "lstm_lookback_hours": int(arch["lookback_hours"]),
                "lstm_hidden_size": int(arch["hidden_size"]),
                "lstm_num_layers": int(arch["num_layers"]),
                "lstm_dropout": float(arch["dropout_rate"]),
            }
        )
        model = cls(str(meta["metric"]), settings=cfg)
        _init_common_from_metadata(model, bundle_dir, meta)
        model.input_size = int(arch["input_size"])
        model.output_size = int(arch["output_size"])

        with np.load(bundle_dir / "lstm_stats.npz", allow_pickle=False) as stats:
            model.y_mean = np.asarray(stats["y_mean"], dtype=np.float64)
            model.y_std = np.asarray(stats["y_std"], dtype=np.float64)

        model.net = WeatherLSTMNet(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=model.dropout_rate,
            output_size=model.output_size,
        )
        state_dict = torch.load(bundle_dir / "lstm_weights.pt", map_location="cpu", weights_only=True)
        model.net.load_state_dict(state_dict)
        model.net.eval()
        return model
