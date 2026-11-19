"""Statistical multi-horizon SARIMAX forecasting model (`statsmodels`)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from skycast.config.settings import Settings
from skycast.ml.evaluate import (
    ChronologicalSplit,
    clamp_predictions_and_bounds,
    reconstruct_wind_direction,
)
from skycast.ml.models.base import (
    BaseForecastModel,
    ForecastPrediction,
    _init_common_from_metadata,
)

_MAX_EXOG_FEATURES = 6
_LAG_OFFSETS: tuple[int, ...] = (48, 24, 12, 6, 3, 1, 0)
_DEFAULT_EXOG_CANDIDATES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "pm2_5",
    "us_aqi",
    "apparent_temperature",
    "wind_dir_sin",
    "wind_dir_cos",
    "is_weekend",
    "is_lahore_smog_season",
    "is_monsoon_season",
)


def _compute_warmup_len(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> int:
    s = int(seasonal_order[3])
    burn = (int(seasonal_order[0]) + int(seasonal_order[1])) * max(s, 1) + sum(int(v) for v in order)
    if s > 0:
        return max(2 * s + 1, burn + 1)
    return max(12, burn + 2)


def _build_lag_interp_matrix(warmup_len: int, available_lags: list[int]) -> np.ndarray:
    """Build `(len(available_lags), warmup_len)` linear interpolation weights for causal prefix."""
    lags_sorted = sorted(set(available_lags), reverse=True)
    if 0 not in lags_sorted:
        lags_sorted.append(0)
    max_pos = warmup_len - 1
    anchor_x = np.asarray([max_pos - lag for lag in lags_sorted], dtype=np.float64)
    grid = np.arange(warmup_len, dtype=np.float64)
    weights = np.zeros((len(lags_sorted), warmup_len), dtype=np.float64)
    for j in range(len(lags_sorted)):
        unit = np.zeros(len(lags_sorted), dtype=np.float64)
        unit[j] = 1.0
        weights[j, :] = np.interp(grid, anchor_x, unit)
    return weights


def _select_exog_columns(
    target_col: str,
    metric: str,
    feature_columns: list[str],
    X_train_scaled: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> list[str]:
    """Select well-conditioned, non-collinear exogenous columns strictly on `train`."""
    excluded: set[str] = {target_col, metric}
    if metric == "temperature_2m":
        excluded.add("apparent_temperature")
    if metric == "wind_direction_10m" or target_col in {"wind_dir_sin", "wind_dir_cos"}:
        excluded.update({"wind_dir_sin", "wind_dir_cos", "wind_u", "wind_v"})

    candidates = [c for c in _DEFAULT_EXOG_CANDIDATES if c in feature_columns and c not in excluded]
    if not candidates:
        candidates = [
            c
            for c in feature_columns
            if c not in excluded and not any(tok in c for tok in ("_lag_", "_roll_", "_roc_", "hour_"))
        ]

    d = int(order[1])
    D = int(seasonal_order[1])
    s = int(seasonal_order[3])

    def _transformed_view(mat: np.ndarray) -> np.ndarray:
        out = mat
        if D > 0 and s > 0 and len(out) > s:
            out = out[s:] - out[:-s]
        if d > 0 and len(out) > d:
            out = np.diff(out, n=d, axis=0)
        return out

    selected: list[str] = []
    selected_indices: list[int] = []
    n_train = len(X_train_scaled)
    if n_train < 12:
        return []

    for col in candidates:
        if len(selected) >= _MAX_EXOG_FEATURES:
            break
        idx = feature_columns.index(col)
        col_vec = X_train_scaled[:, idx : idx + 1]
        if float(np.std(col_vec)) < 1e-10:
            continue
        diff_vec = _transformed_view(col_vec)
        if len(diff_vec) < 4 or float(np.std(diff_vec)) < 1e-6:
            continue

        trial_indices = [*selected_indices, idx]
        trial_mat = _transformed_view(X_train_scaled[:, trial_indices])
        if len(trial_mat) <= len(trial_indices) + 2:
            continue
        cond = float(np.linalg.cond(trial_mat))
        if np.isfinite(cond) and cond < 1e5:
            selected.append(col)
            selected_indices.append(idx)

    return selected


class _SARIMAXSeriesComponent:
    """Single-series SARIMAX state-space estimator with causal filtering & vectorized explain."""

    def __init__(
        self,
        *,
        target_col: str,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        conf_alpha: float,
        maxiter: int = 50,
    ) -> None:
        self.target_col = target_col
        self.order = tuple(int(v) for v in order)
        self.seasonal_order = tuple(int(v) for v in seasonal_order)
        self.conf_alpha = float(conf_alpha)
        self.maxiter = max(1, int(maxiter))
        self.warmup_len = _compute_warmup_len(self.order, self.seasonal_order)

        self.exog_columns: list[str] = []
        self.exog_indices: list[int] = []
        self.params: np.ndarray | None = None
        self.is_constant: bool = False
        self.constant_value: float = 0.0

        self.train_tail_y: np.ndarray | None = None
        self.train_tail_exog: np.ndarray | None = None
        self.train_last_ts: str | None = None

        # Cached state-space matrices for fast O(H) rollouts & explain filter
        self.ssm_T: np.ndarray | None = None
        self.ssm_Z: np.ndarray | None = None
        self.ssm_RQR: np.ndarray | None = None
        self.ssm_H: np.ndarray | None = None
        self.ssm_c_state: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.explain_initial_state: np.ndarray | None = None
        self.explain_kalman_gains: np.ndarray | None = None

        # Cached lag reconstruction metadata for X_scaled-only explain calls
        self.target_feature_idx: int | None = None
        self.target_mean: float = 0.0
        self.target_scale: float = 1.0
        self.lag_feature_indices: list[int] = []
        self.lag_means: list[float] = []
        self.lag_scales: list[float] = []
        self.lag_interp_weights: np.ndarray | None = None

    def _extract_target_series(self, df: pd.DataFrame) -> np.ndarray:
        if self.target_col not in df.columns:
            raise ValueError(f"DataFrame missing SARIMAX series column '{self.target_col}'.")
        arr = pd.to_numeric(df[self.target_col], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"SARIMAX series '{self.target_col}' contains NaN or Inf values.")
        return arr

    def _cache_lag_metadata(self, split: ChronologicalSplit) -> None:
        feat_cols = split.feature_columns
        means = np.asarray(split.scaler.mean_, dtype=np.float64)
        scales = np.asarray(split.scaler.scale_, dtype=np.float64)

        if self.target_col in feat_cols:
            idx = feat_cols.index(self.target_col)
            self.target_feature_idx = idx
            self.target_mean = float(means[idx])
            self.target_scale = float(scales[idx])

        avail_lags: list[int] = []
        self.lag_feature_indices = []
        self.lag_means = []
        self.lag_scales = []
        for lag in _LAG_OFFSETS:
            col_name = self.target_col if lag == 0 else f"{self.target_col}_lag_{lag}h"
            if col_name in feat_cols:
                idx = feat_cols.index(col_name)
                avail_lags.append(lag)
                self.lag_feature_indices.append(idx)
                self.lag_means.append(float(means[idx]))
                self.lag_scales.append(float(scales[idx]))
        if not avail_lags:
            avail_lags = [0]
        self.lag_interp_weights = _build_lag_interp_matrix(self.warmup_len, avail_lags)

    def _cache_ssm_matrices(self, fitted_res: Any) -> None:
        ssm = fitted_res.model.ssm
        self.ssm_T = np.asarray(ssm["transition", :, :, 0], dtype=np.float64).copy()
        self.ssm_Z = np.asarray(ssm["design", :, :, 0], dtype=np.float64).copy()
        R = np.asarray(ssm["selection", :, :, 0], dtype=np.float64)
        Q = np.asarray(ssm["state_cov", :, :, 0], dtype=np.float64)
        self.ssm_RQR = (R @ Q @ R.T).copy()
        self.ssm_H = np.asarray(ssm["obs_cov", :, :, 0], dtype=np.float64).copy()
        self.ssm_c_state = np.asarray(ssm["state_intercept", :, 0], dtype=np.float64).copy()
        k_exog = len(self.exog_columns)
        self.beta = (
            np.asarray(self.params[:k_exog], dtype=np.float64).copy()
            if k_exog > 0
            else np.zeros(0, dtype=np.float64)
        )

        # Precompute fixed Kalman gain sequence for `warmup_len` dense steps used in explain
        L = self.warmup_len
        ex_zero = np.zeros((L, k_exog), dtype=np.float64) if k_exog > 0 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_app = SARIMAX(
                np.zeros(L, dtype=np.float64),
                exog=ex_zero,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).filter(self.params)

        fr = ref_app.filter_results
        self.explain_initial_state = np.asarray(fr.initial_state, dtype=np.float64).copy()
        P_pred = np.asarray(fr.initial_state_cov, dtype=np.float64).copy()
        z_vec = self.ssm_Z[0]
        h_obs = float(self.ssm_H[0, 0])
        gains = np.zeros((L, len(self.explain_initial_state)), dtype=np.float64)
        for step in range(L):
            F = float(z_vec @ P_pred @ z_vec + h_obs)
            F = F if F > 1e-18 else 1e-18
            k_step = (P_pred @ z_vec) / F
            gains[step] = k_step
            P_filt = P_pred - np.outer(k_step, z_vec @ P_pred)
            P_pred = self.ssm_T @ P_filt @ self.ssm_T.T + self.ssm_RQR
        self.explain_kalman_gains = gains

    def fit(self, split: ChronologicalSplit, metric: str) -> _SARIMAXSeriesComponent:
        y_train = self._extract_target_series(split.train_df)
        X_train = np.asarray(split.X_train_scaled, dtype=np.float64)
        self._cache_lag_metadata(split)

        tail_len = max(self.warmup_len, 72)
        self.train_tail_y = y_train[-tail_len:].copy()
        self.train_last_ts = pd.to_datetime(split.train_df["timestamp_utc"].iloc[-1], utc=True).isoformat()

        if float(np.std(y_train)) < 1e-12:
            self.is_constant = True
            self.constant_value = float(y_train[0])
            self.exog_columns = []
            self.exog_indices = []
            self.train_tail_exog = np.zeros((len(self.train_tail_y), 0), dtype=np.float64)
            self.params = np.zeros(0, dtype=np.float64)
            return self

        self.is_constant = False
        cand_cols = _select_exog_columns(
            self.target_col,
            metric,
            split.feature_columns,
            X_train,
            self.order,
            self.seasonal_order,
        )
        cand_indices = [split.feature_columns.index(c) for c in cand_cols]

        fitted_res = None
        if cand_indices:
            exog_train = X_train[:, cand_indices]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = SARIMAX(
                        y_train,
                        exog=exog_train,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res_try = mod.fit(disp=False, maxiter=self.maxiter)
                params_try = np.asarray(res_try.params, dtype=np.float64)
                beta_try = params_try[: len(cand_indices)]
                if np.isfinite(params_try).all() and (
                    len(beta_try) == 0 or float(np.max(np.abs(beta_try))) < 1e4
                ):
                    fitted_res = res_try
                    self.exog_columns = cand_cols
                    self.exog_indices = cand_indices
            except (np.linalg.LinAlgError, ValueError):
                fitted_res = None

        if fitted_res is None:
            self.exog_columns = []
            self.exog_indices = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(
                    y_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted_res = mod.fit(disp=False, maxiter=self.maxiter)

        self.params = np.asarray(fitted_res.params, dtype=np.float64).copy()
        if not np.isfinite(self.params).all():
            raise RuntimeError(f"SARIMAX fit produced non-finite parameters for '{self.target_col}'.")
        self.train_tail_exog = (
            X_train[-tail_len:, self.exog_indices].copy()
            if self.exog_indices
            else np.zeros((len(self.train_tail_y), 0), dtype=np.float64)
        )
        self._cache_ssm_matrices(fitted_res)
        return self

    def _build_first_row_causal_prefix(self, row0_df: pd.DataFrame, y0: float) -> np.ndarray:
        """Build `(warmup_len - 1,)` strictly causal prefix prior to `row0` from `lag_*h` columns."""
        anchors: list[tuple[int, float]] = [(0, float(y0))]
        for lag in _LAG_OFFSETS:
            if lag == 0:
                continue
            col = f"{self.target_col}_lag_{lag}h"
            if col in row0_df.columns:
                val = pd.to_numeric(row0_df[col].iloc[0], errors="coerce")
                if pd.notna(val) and np.isfinite(float(val)):
                    anchors.append((lag, float(val)))
        anchors.sort(key=lambda pair: pair[0], reverse=True)
        max_pos = self.warmup_len - 1
        anchor_x = np.asarray([max_pos - lag for lag, _ in anchors], dtype=np.float64)
        anchor_y = np.asarray([val for _, val in anchors], dtype=np.float64)
        full_interp = np.interp(np.arange(self.warmup_len, dtype=np.float64), anchor_x, anchor_y)
        return full_interp[:-1]

    def _assemble_causal_series(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray,
        *,
        context_df: pd.DataFrame | None,
        context_X_scaled: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        y_curr = self._extract_target_series(df)
        ex_curr = (
            X_scaled[:, self.exog_indices].copy()
            if self.exog_indices
            else None
        )
        s = int(self.seasonal_order[3])
        max_bridge_gap = max(48, 2 * s)

        prefix_y: np.ndarray | None = None
        prefix_ex: np.ndarray | None = None

        has_ts = "timestamp_utc" in df.columns and not df["timestamp_utc"].isna().any()
        if has_ts:
            first_ts = pd.to_datetime(df["timestamp_utc"].iloc[0], utc=True)
            # Option 1: explicit context_df supplied (e.g. val_df before test_df or 72h recent before latest)
            if (
                context_df is not None
                and not context_df.empty
                and context_X_scaled is not None
                and "timestamp_utc" in context_df.columns
            ):
                ctx_ts_last = pd.to_datetime(context_df["timestamp_utc"].iloc[-1], utc=True)
                gap_hours = int(round((first_ts - ctx_ts_last).total_seconds() / 3600.0)) - 1
                if 0 <= gap_hours <= max_bridge_gap:
                    tail_n = max(self.warmup_len, 72)
                    ctx_y = self._extract_target_series(context_df)[-tail_n:]
                    if gap_hours > 0:
                        prefix_y = np.concatenate([ctx_y, np.full(gap_hours, np.nan, dtype=np.float64)])
                    else:
                        prefix_y = ctx_y
                    if self.exog_indices:
                        ctx_ex = np.asarray(context_X_scaled, dtype=np.float64)[-tail_n:, self.exog_indices]
                        if gap_hours > 0:
                            gap_ex = np.repeat(ctx_ex[-1:], gap_hours, axis=0)
                            prefix_ex = np.vstack([ctx_ex, gap_ex])
                        else:
                            prefix_ex = ctx_ex

            # Option 2: df immediately follows train_tail (e.g. val_df after train_df + embargo)
            if (
                prefix_y is None
                and self.train_last_ts is not None
                and self.train_tail_y is not None
            ):
                train_ts_last = pd.to_datetime(self.train_last_ts, utc=True)
                gap_hours = int(round((first_ts - train_ts_last).total_seconds() / 3600.0)) - 1
                if 0 <= gap_hours <= max_bridge_gap:
                    if gap_hours > 0:
                        prefix_y = np.concatenate(
                            [self.train_tail_y, np.full(gap_hours, np.nan, dtype=np.float64)]
                        )
                    else:
                        prefix_y = self.train_tail_y.copy()
                    if self.exog_indices and self.train_tail_exog is not None:
                        if gap_hours > 0:
                            gap_ex = np.repeat(self.train_tail_exog[-1:], gap_hours, axis=0)
                            prefix_ex = np.vstack([self.train_tail_exog, gap_ex])
                        else:
                            prefix_ex = self.train_tail_exog.copy()

        # Option 3: standalone start (train_df[0], standalone test_df, or 1-row latest_df)
        if prefix_y is None:
            prefix_y = self._build_first_row_causal_prefix(df.iloc[:1], float(y_curr[0]))
            if self.exog_indices and ex_curr is not None:
                prefix_ex = np.repeat(ex_curr[:1], len(prefix_y), axis=0)

        offset = len(prefix_y)
        full_y = np.concatenate([prefix_y, y_curr])
        full_ex = (
            np.vstack([prefix_ex, ex_curr])
            if (self.exog_indices and prefix_ex is not None and ex_curr is not None)
            else None
        )
        return full_y, full_ex, offset

    def predict_series(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray,
        horizon_hours: int,
        *,
        context_df: pd.DataFrame | None = None,
        context_X_scaled: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(mean, lower, upper)` of shape `(len(df), horizon_hours)`."""
        n = len(df)
        if self.is_constant:
            mean = np.full((n, horizon_hours), self.constant_value, dtype=np.float64)
            return mean, mean.copy(), mean.copy()

        full_y, full_ex, offset = self._assemble_causal_series(
            df,
            X_scaled,
            context_df=context_df,
            context_X_scaled=context_X_scaled,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = SARIMAX(
                full_y,
                exog=full_ex,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).filter(self.params)

        fr = filtered.filter_results
        T = self.ssm_T
        Z = self.ssm_Z[0]
        RQR = self.ssm_RQR
        h_obs = float(self.ssm_H[0, 0])
        c_state = self.ssm_c_state
        beta = self.beta
        z_crit = float(norm.ppf(1.0 - self.conf_alpha / 2.0))

        mean = np.zeros((n, horizon_hours), dtype=np.float64)
        lower = np.zeros((n, horizon_hours), dtype=np.float64)
        upper = np.zeros((n, horizon_hours), dtype=np.float64)

        ex_curr = X_scaled[:, self.exog_indices] if self.exog_indices else None
        ex_contrib = (
            ex_curr @ beta
            if (ex_curr is not None and len(beta) > 0)
            else np.zeros(n, dtype=np.float64)
        )

        # Vectorized time-invariant Kalman rollout across all n origin rows
        a_mat = fr.filtered_state[:, offset : offset + n].T.copy()
        P0 = np.ascontiguousarray(fr.filtered_state_cov[:, :, offset : offset + n])
        z_h = Z.copy()
        q_cum = 0.0

        for h in range(horizon_hours):
            a_mat = a_mat @ T.T + c_state[None, :]
            mu_h = (a_mat @ Z) + ex_contrib
            q_cum += float(z_h @ RQR @ z_h)
            z_h = T.T @ z_h
            # Compute z_h^T @ P0[:, :, i] @ z_h for all i without Python loop
            Pz = np.tensordot(z_h, P0, axes=(0, 0))  # (k, n)
            var_h = (z_h @ Pz) + q_cum + h_obs
            std_h = np.sqrt(np.maximum(var_h, 0.0))
            mean[:, h] = mu_h
            lower[:, h] = mu_h - z_crit * std_h
            upper[:, h] = mu_h + z_crit * std_h

        return mean, lower, upper

    def predict_point_from_scaled_batch(
        self,
        X_scaled: np.ndarray,
        horizon_hours: int,
    ) -> np.ndarray:
        """Vectorized Kalman causal-prefix rollout from scaled feature rows only (for SHAP/LIME)."""
        n = len(X_scaled)
        if self.is_constant:
            return np.full((n, horizon_hours), self.constant_value, dtype=np.float64)

        if self.lag_feature_indices and self.lag_interp_weights is not None:
            means = np.asarray(self.lag_means, dtype=np.float64)
            scales = np.asarray(self.lag_scales, dtype=np.float64)
            anchors_raw = X_scaled[:, self.lag_feature_indices] * scales[None, :] + means[None, :]
            y_seq = anchors_raw @ self.lag_interp_weights
        elif self.target_feature_idx is not None:
            y0 = X_scaled[:, self.target_feature_idx] * self.target_scale + self.target_mean
            y_seq = np.repeat(y0[:, None], self.warmup_len, axis=1)
        else:
            y_seq = np.zeros((n, self.warmup_len), dtype=np.float64)

        ex_curr = X_scaled[:, self.exog_indices] if self.exog_indices else None
        ex_contrib = (
            ex_curr @ self.beta if (ex_curr is not None and len(self.beta) > 0) else np.zeros(n, dtype=np.float64)
        )

        a0 = self.explain_initial_state
        T = self.ssm_T
        z_vec = self.ssm_Z[0]
        c_state = self.ssm_c_state
        gains = self.explain_kalman_gains

        a_pred = np.broadcast_to(a0, (n, len(a0))).copy()
        a_filt = a_pred
        for step in range(self.warmup_len):
            v = y_seq[:, step] - ex_contrib - (a_pred @ z_vec)
            a_filt = a_pred + v[:, None] * gains[step][None, :]
            a_pred = a_filt @ T.T + c_state[None, :]

        preds = np.zeros((n, horizon_hours), dtype=np.float64)
        a_roll = a_filt
        for h in range(horizon_hours):
            a_roll = a_roll @ T.T + c_state[None, :]
            preds[:, h] = (a_roll @ z_vec) + ex_contrib
        return preds

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "target_col": self.target_col,
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "conf_alpha": self.conf_alpha,
            "maxiter": self.maxiter,
            "warmup_len": self.warmup_len,
            "exog_columns": list(self.exog_columns),
            "exog_indices": list(self.exog_indices),
            "is_constant": self.is_constant,
            "constant_value": self.constant_value,
            "train_last_ts": self.train_last_ts,
            "target_feature_idx": self.target_feature_idx,
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "lag_feature_indices": list(self.lag_feature_indices),
            "lag_means": list(self.lag_means),
            "lag_scales": list(self.lag_scales),
        }

    def save_arrays(self, out_dir: Path, prefix: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        for attr in (
            "params",
            "train_tail_y",
            "train_tail_exog",
            "ssm_T",
            "ssm_Z",
            "ssm_RQR",
            "ssm_H",
            "ssm_c_state",
            "beta",
            "explain_initial_state",
            "explain_kalman_gains",
            "lag_interp_weights",
        ):
            val = getattr(self, attr)
            if val is not None:
                arrays[attr] = np.asarray(val, dtype=np.float64)
        np.savez(out_dir / f"{prefix}.npz", **arrays)

    @classmethod
    def from_state(cls, state: Mapping[str, Any], npz_path: Path) -> _SARIMAXSeriesComponent:
        comp = cls(
            target_col=str(state["target_col"]),
            order=tuple(int(v) for v in state["order"]),
            seasonal_order=tuple(int(v) for v in state["seasonal_order"]),
            conf_alpha=float(state["conf_alpha"]),
            maxiter=int(state.get("maxiter", 50)),
        )
        comp.warmup_len = int(state["warmup_len"])
        comp.exog_columns = [str(c) for c in state["exog_columns"]]
        comp.exog_indices = [int(i) for i in state["exog_indices"]]
        comp.is_constant = bool(state["is_constant"])
        comp.constant_value = float(state["constant_value"])
        comp.train_last_ts = state.get("train_last_ts")
        comp.target_feature_idx = state.get("target_feature_idx")
        comp.target_mean = float(state.get("target_mean", 0.0))
        comp.target_scale = float(state.get("target_scale", 1.0))
        comp.lag_feature_indices = [int(i) for i in state.get("lag_feature_indices", [])]
        comp.lag_means = [float(v) for v in state.get("lag_means", [])]
        comp.lag_scales = [float(v) for v in state.get("lag_scales", [])]

        with np.load(npz_path, allow_pickle=False) as loaded:
            for attr in loaded.files:
                setattr(comp, attr, np.asarray(loaded[attr], dtype=np.float64))
        return comp


class SARIMAXModel(BaseForecastModel):
    """Multi-horizon SARIMAX wrapper with Settings hyperparameters and circular wind support."""

    family_name = "SARIMAX"

    def __init__(self, metric: str, *, settings: Settings | None = None) -> None:
        super().__init__(metric, settings=settings)
        self.comp: _SARIMAXSeriesComponent | None = None
        self.comp_sin: _SARIMAXSeriesComponent | None = None
        self.comp_cos: _SARIMAXSeriesComponent | None = None
        self.hyperparameters = {
            "order": list(self.settings.sarimax_order),
            "seasonal_order": list(self.settings.sarimax_seasonal_order),
            "conf_alpha": float(self.settings.sarimax_conf_alpha),
            "maxiter": int(self.settings.sarimax_maxiter),
            "horizon_hours": self.horizon_hours,
        }

    def fit(self, split: ChronologicalSplit) -> SARIMAXModel:
        self.feature_columns = list(split.feature_columns)
        self.scaler = split.scaler
        order = self.settings.sarimax_order
        seasonal_order = self.settings.sarimax_seasonal_order
        alpha = float(self.settings.sarimax_conf_alpha)
        maxiter = int(self.settings.sarimax_maxiter)

        if self.is_circular:
            self.comp_sin = _SARIMAXSeriesComponent(
                target_col="wind_dir_sin",
                order=order,
                seasonal_order=seasonal_order,
                conf_alpha=alpha,
                maxiter=maxiter,
            ).fit(split, self.metric)
            self.comp_cos = _SARIMAXSeriesComponent(
                target_col="wind_dir_cos",
                order=order,
                seasonal_order=seasonal_order,
                conf_alpha=alpha,
                maxiter=maxiter,
            ).fit(split, self.metric)
            self.comp = None
            sin_tr, _, _ = self.comp_sin.predict_series(
                split.train_df, split.X_train_scaled, self.horizon_hours
            )
            cos_tr, _, _ = self.comp_cos.predict_series(
                split.train_df, split.X_train_scaled, self.horizon_hours
            )
            raw_train_pred = np.asarray(reconstruct_wind_direction(sin_tr, cos_tr), dtype=np.float64)
            self.hyperparameters["exog_columns"] = {
                "sin": list(self.comp_sin.exog_columns),
                "cos": list(self.comp_cos.exog_columns),
            }
        else:
            self.comp = _SARIMAXSeriesComponent(
                target_col=self.metric,
                order=order,
                seasonal_order=seasonal_order,
                conf_alpha=alpha,
                maxiter=maxiter,
            ).fit(split, self.metric)
            self.comp_sin = None
            self.comp_cos = None
            raw_train_pred, _, _ = self.comp.predict_series(
                split.train_df, split.X_train_scaled, self.horizon_hours
            )
            self.hyperparameters["exog_columns"] = list(self.comp.exog_columns)

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
        ctx_X = (
            self._prepare_scaled_features(context_df, context_X_scaled)
            if (context_df is not None and not context_df.empty)
            else None
        )

        if self.is_circular:
            if self.comp_sin is None or self.comp_cos is None:
                raise RuntimeError("SARIMAXModel circular components not fitted.")
            sin_p, _, _ = self.comp_sin.predict_series(
                df, X, self.horizon_hours, context_df=context_df, context_X_scaled=ctx_X
            )
            cos_p, _, _ = self.comp_cos.predict_series(
                df, X, self.horizon_hours, context_df=context_df, context_X_scaled=ctx_X
            )
            raw_deg = np.asarray(reconstruct_wind_direction(sin_p, cos_p), dtype=np.float64)
            return self._apply_quantiles(raw_deg)

        if self.comp is None:
            raise RuntimeError("SARIMAXModel component not fitted.")
        mean, raw_low, raw_high = self.comp.predict_series(
            df, X, self.horizon_hours, context_df=context_df, context_X_scaled=ctx_X
        )

        # Fallback per-row to empirical train quantiles if Kalman interval is non-finite or diffuse
        if self.q_low is not None and self.q_high is not None:
            bad_ci = (
                ~np.isfinite(raw_low)
                | ~np.isfinite(raw_high)
                | ((raw_high - raw_low) > 1e4)
                | (self.comp.is_constant)
            )
            if bad_ci.any():
                emp_pred, emp_low, emp_high = self._apply_quantiles(mean)
                mean = np.where(bad_ci, emp_pred, mean)
                raw_low = np.where(bad_ci, emp_low, raw_low)
                raw_high = np.where(bad_ci, emp_high, raw_high)

        pred_c, low_c, high_c = clamp_predictions_and_bounds(self.metric, mean, raw_low, raw_high)
        return ForecastPrediction(predicted=pred_c, lower=low_c, upper=high_c)

    def predict_point_matrix_for_explain(self, X_scaled: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        if self.is_circular:
            if self.comp_sin is None or self.comp_cos is None:
                raise RuntimeError("SARIMAXModel circular components not fitted.")
            sin_p = self.comp_sin.predict_point_from_scaled_batch(X_arr, self.horizon_hours)
            cos_p = self.comp_cos.predict_point_from_scaled_batch(X_arr, self.horizon_hours)
            return np.asarray(reconstruct_wind_direction(sin_p, cos_p), dtype=np.float64)
        if self.comp is None:
            raise RuntimeError("SARIMAXModel component not fitted.")
        return self.comp.predict_point_from_scaled_batch(X_arr, self.horizon_hours)

    def _save_model_artifacts(self, out_dir: Path) -> dict[str, Any] | None:
        if self.is_circular:
            if self.comp_sin is None or self.comp_cos is None:
                raise RuntimeError("Cannot save unfitted circular SARIMAXModel.")
            self.comp_sin.save_arrays(out_dir, "sarimax_sin")
            self.comp_cos.save_arrays(out_dir, "sarimax_cos")
            return {
                "sarimax_states": {
                    "sin": self.comp_sin.to_state_dict(),
                    "cos": self.comp_cos.to_state_dict(),
                }
            }
        if self.comp is None:
            raise RuntimeError("Cannot save unfitted SARIMAXModel.")
        self.comp.save_arrays(out_dir, "sarimax_main")
        return {"sarimax_states": {"main": self.comp.to_state_dict()}}

    @classmethod
    def _load_from_dir(
        cls,
        bundle_dir: Path,
        meta: Mapping[str, Any],
        settings: Settings,
    ) -> SARIMAXModel:
        model = cls(str(meta["metric"]), settings=settings)
        _init_common_from_metadata(model, bundle_dir, meta)
        states = meta["sarimax_states"]
        if model.is_circular:
            model.comp_sin = _SARIMAXSeriesComponent.from_state(
                states["sin"], bundle_dir / "sarimax_sin.npz"
            )
            model.comp_cos = _SARIMAXSeriesComponent.from_state(
                states["cos"], bundle_dir / "sarimax_cos.npz"
            )
        else:
            model.comp = _SARIMAXSeriesComponent.from_state(
                states["main"], bundle_dir / "sarimax_main.npz"
            )
        return model
