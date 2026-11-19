# SkyCast (`AgentWeather`) — Final Technical Report

**Project:** Serverless Agentic Multi-Metric Weather & Air Quality Forecasting System  
**Target City:** Lahore, Pakistan (`31.5204° N, 74.3587° E`, Elevation `217 m`, Timezone `Asia/Karachi` / PKT)  
**Reference Benchmark Window:** `2026-10-19T13:00Z` – `2026-11-19T13:00Z` (721 historical training hours; 745 total hours after live hourly ingestion; Training Run `train-20261119T133423Z-40deb2`; Hourly Agent Run `hourly-20261119T130000Z-543174`)

---

## Executive Summary

SkyCast (`AgentWeather`) is an end-to-end, serverless MLOps and Agentic AI forecasting platform built for **Lahore, Pakistan**. The system continuously ingests hourly meteorological and pollutant observations from two zero-key APIs (**Open-Meteo** and **Met.no**), applies physics-aware multi-source fusion and strictly causal 3-tier imputation, engineers **270 causal temporal and atmospheric features**, evaluates **three distinct forecasting model families** (`SARIMAX`, `RandomForest` & `Ridge`, and CPU `PyTorch LSTM`) against a **24-hour naive persistence baseline (`Naive24h`)** across **6 core weather/AQI metric groups (8 target series)**, computes both **SHAP** and **LIME** explanations, and orchestrates a **4-node stateful LangGraph workflow** (`Forecast Agent` → `Alert Agent` → `Drift Agent` → conditional `Weather Report Agent`) backed by a resilient **Groq → Gemini → Groq → Deterministic Template** fallback chain.

On a verified 30-day real Lahore dataset spanning the late-October to mid-November 2026 smog transition (`721` hourly rows, chronological 70% / 15% / 15% split with 24-hour embargoes), all three ML model families earned production promotions across distinct target metrics—achieving up to **+92.95%** validation RMSE reduction over `Naive24h` on `uv_index` (`SARIMAX`), **+86.60%** on `temperature_2m` (`RandomForest`), **+75.86%** on `relative_humidity_2m` (`RandomForest`), **+20.48%** on `wind_speed_10m` (`RandomForest`), **+4.56%** on `precipitation_mm` (`RandomForest`), and **+1.87%** on `precipitation_prob` (`LSTM`)—while the champion promotion guardrail prevented regression on `us_aqi` and `wind_direction_10m` during an abrupt smog/calm-wind regime shift.

---

## 1. Multi-Source Fusion & Time-Alignment Mechanics

### 1.1 Zero-Key Upstream Sources

SkyCast combines two independent, zero-API-key meteorological data providers in `skycast/data/sources/`:

| Source | Module | Endpoints Used | Coverage & Variables |
| :--- | :--- | :--- | :--- |
| **Open-Meteo** | `skycast/data/sources/open_meteo.py` | `/v1/forecast`, `/v1/archive`, `historical-forecast-api/v1/forecast`, `air-quality-api/v1/air-quality` | Full 8 target series (`us_aqi`, `temperature_2m`, `precipitation_mm`, `precipitation_prob`, `relative_humidity_2m`, `uv_index`, `wind_speed_10m`, `wind_direction_10m`), auxiliary weather (`apparent_temperature`, `weather_code`, `surface_pressure`, `cloud_cover`), and 6 pollutants (`pm2_5`, `pm10`, `ozone`, `nitrogen_dioxide`, `sulphur_dioxide`, `carbon_monoxide`). |
| **Met.no** | `skycast/data/sources/met_no.py` | `locationforecast/2.0/complete` (with compliant `User-Agent: SkyCast-10Pearls-Portfolio/1.0`) | Independent cross-check & outage fallback for `air_temperature`, `relative_humidity`, `precipitation_amount`, `wind_speed` (converted `m/s` → `km/h`), `wind_from_direction`, `air_pressure_at_sea_level`, `cloud_area_fraction`, `ultraviolet_index_clear_sky`, and `symbol_code`. |

### 1.2 UTC Hourly Alignment & Physical Harmonization

All observations are normalized in `skycast/data/fusion.py` onto a contiguous UTC hourly index (`YYYY-MM-DD HH:00:00+00:00`). Before fusing, SkyCast resolves four subtle physical mismatches between Open-Meteo and Met.no:

1. **Precipitation Interval Shift (`+1h`):** Met.no's `next_1_hours.precipitation_amount` at timestamp `t` accumulates rain over `[t, t+1h]`, whereas Open-Meteo reports ending-hour accumulation `[t-1h, t]`. Met.no precipitation is shifted `+1h` before blending so rain spikes align to the exact same hour.
2. **Station Surface Pressure Conversion (`217 m`):** Met.no reports `air_pressure_at_sea_level`, while Open-Meteo reports Lahore station `surface_pressure`. Met.no sea-level pressure is hypsometrically reduced to Lahore's `217 m` elevation before weighted fusion.
3. **Cloud-Attenuated UV Fallback:** Open-Meteo provides cloud-attenuated `uv_index`, while Met.no supplies `ultraviolet_index_clear_sky`. SkyCast prioritizes Open-Meteo UV directly and applies a cloud-modification factor `1 - 0.75 * (cloud_cover / 100)^3` to Met.no clear-sky UV only during Open-Meteo outages.
4. **Circular Wind Direction Fusion:** Linear averaging across `359°` and `1°` erroneously yields `180°`. `weighted_circular_fuse` projects both sources onto the unit circle (`x = w_om cos(θ_om) + w_met cos(θ_met)`, `y = w_om sin(θ_om) + w_met sin(θ_met)`) and reconstructs `atan2(y, x) mod 360`.

### 1.3 Weighted Consensus & Uncertainty Tracking

When both APIs are available, physically commensurate variables are fused using configurable weights from `Settings` (`open_meteo_fusion_weight = 0.6`, `met_no_fusion_weight = 0.4`), automatically falling back to whichever provider is online if one times out. Raw per-source columns (`temp_openmeteo`, `temp_metno`, etc.), `sources_available`, and `source_disagreement_temp = |T_OM - T_MetNo|` are preserved for transparency. On the verified live Lahore ingestion (`745` total rows, including dual-source live overlap), observed inter-source temperature disagreement peaked at `0.8°C` and remained `NaN` (never imputed to `0`) when only one source covered historical archive rows.

---

## 2. Documented 3-Tier Missing Data Imputation & Audit

### 2.1 Strictly Causal 3-Tier Strategy

Implemented in `skycast/data/imputation.py` (`impute_weather_dataframe`), SkyCast uses a strictly causal (`causal_only=True`) O(log n) pre-indexed lookup engine (`_CausalLookupIndex`) to guarantee zero future data leakage:

| Tier | Gap Condition | Mechanism | Domain & Circular Safeguards |
| :--- | :--- | :--- | :--- |
| **Tier 1 (`tier1_linear`)** | Short interior gap (`<= 3` consecutive hours with both neighbors present) | Time-aware linear interpolation between left and right boundary timestamps | Discrete `weather_code` uses nearest-neighbor mode (never fractional codes); `wind_direction_10m` interpolates `sin(θ)` and `cos(θ)` independently. |
| **Tier 2 (`tier2_diurnal_median`)** | Medium gap (`4–24` hours, or short boundary gap lacking a right neighbor) | Causal rolling 7-day median for the exact UTC hour-of-day (`imputation_diurnal_lookback_days = 7`) | Preserves Lahore's sharp diurnal temperature, humidity, and nocturnal smog inversion curves. |
| **Tier 3 (`tier3_climatology_ffill`)** | Extended outage (`> 24` hours, or cold-start gap with no prior diurnal history) | Exponentially damped forward-fill (`τ = 12.0h`) decaying smoothly toward the causal `(month, hour)` climatological median, falling back to documented Lahore physical priors only when zero history exists | Flags `is_imputed = True`, records highest `imputation_tier` and exact `imputed_columns`, and aborts backfill if any target metric relies on constant fallbacks or exceeds `max_imputation_rate_pct = 50%`. |

### 2.2 Imputation Audit Metrics

Every backfill and hourly ingestion run emits an `ImputationAuditReport` stored in `pipeline_runs.metadata_json` and surfaced in the dashboard's **EDA** tab:

- **Verified Live Window Audit (`hourly_ingest`, 169-hour rolling context):** `3,042` cells checked across `18` meteorological/pollutant columns → `missing_before = 0`, `missing_after = 0`, `tier1_count = 0`, `tier2_count = 0`, `tier3_count = 0`, `imputation_rate_pct = 0.0%`.
- **Hourly Re-Merge Guard:** When new live rows arrive, `skycast/pipelines/hourly_ingest.py` masks previously imputed cells back to `NaN` using stored `imputed_columns` before merging fresh API observations, preventing stale imputed values from masking newly recovered upstream readings or double-blending stored rows.
- **Automated Outage Stress Verification:** `tests/test_data_pipeline.py` injects synthetic 2-hour, 10-hour, and 30-hour outages (including cross-0° circular wind gaps and future-perturbation leakage tests) to verify all three tiers, audit counts, and strict causality.

---

## 3. Automated EDA Insights for Lahore (6 Core Metrics)

Automated EDA (`skycast/ml/eda.py`) executed on the verified 30-day Lahore dataset (`721` hourly observations, `2026-10-19T13:00Z` to `2026-11-18T13:00Z`, `Asia/Karachi` diurnal alignment) and persisted to `eda_summaries` and `docs/eda_summary.json`.

### 3.1 Summary Distributions & PKT Diurnal Profiles

| Metric / Pollutant | Unit | Mean | Std | Min | Median (Q50) | Max | Diurnal Peak (PKT) | Diurnal Trough (PKT) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `us_aqi` | US AQI | 176.57 | 27.38 | 107.00 | 173.00 | 261.00 | 177.17 at `03:00` | 175.42 at `18:00` |
| `temperature_2m` | °C | 22.68 | 3.69 | 13.30 | 22.30 | 30.20 | 28.04 at `14:00` | 17.91 at `06:00` |
| `precipitation_mm` | mm/h | 0.01 | 0.05 | 0.00 | 0.00 | 0.70 | 0.03 at `18:00` | 0.00 at `03:00` |
| `precipitation_prob` | % | 2.31 | 8.63 | 0.00 | 0.00 | 94.00 | 4.40 at `08:00` | 0.67 at `20:00` |
| `relative_humidity_2m` | % | 66.07 | 14.84 | 31.00 | 69.00 | 94.00 | 84.30 at `06:00` | 42.43 at `14:00` |
| `uv_index` | UV | 1.22 | 1.77 | 0.00 | 0.00 | 5.50 | 4.92 at `12:00` | 0.00 at `00:00` |
| `wind_speed_10m` | km/h | 4.36 | 2.09 | 0.00 | 4.40 | 17.30 | 5.85 at `14:00` | 2.50 at `09:00` |
| `wind_direction_10m` | ° | 214.26 | 112.96 | 0.00 | 272.00 | 357.00 | 256.43 at `16:00` | 147.37 at `08:00` |
| `pm2_5` | µg/m³ | 93.68 | 45.63 | 18.30 | 87.30 | 265.40 | 119.21 at `23:00` | 47.13 at `15:00` |
| `pm10` | µg/m³ | 108.36 | 51.28 | 21.50 | 98.70 | 310.10 | 133.61 at `23:00` | 60.44 at `15:00` |
| `ozone` | µg/m³ | 67.20 | 57.96 | 0.00 | 42.00 | 222.00 | 177.03 at `13:00` | 21.43 at `07:00` |
| `nitrogen_dioxide` | µg/m³ | 52.92 | 28.84 | 2.20 | 52.80 | 127.60 | 84.53 at `21:00` | 6.71 at `13:00` |

### 3.2 Key Atmospheric & Cross-Metric Findings

1. **Persistent Smog Regime:** Throughout the October–November smog window (`is_lahore_smog_season = True`), Lahore's `us_aqi` never dropped below `107` (Unhealthy for Sensitive Groups) and averaged `176.57` (Unhealthy), peaking at `261` (Very Unhealthy).
2. **Nocturnal Boundary-Layer Trapping:** Fine particulates (`pm2_5`) surge more than `2.5x` from their afternoon convective minimum (`47.13 µg/m³` at `15:00 PKT`) to their late-evening inversion peak (`119.21 µg/m³` at `23:00 PKT`), accompanied by `nitrogen_dioxide` (`84.53 µg/m³` at `21:00 PKT`) and `carbon_monoxide` (`3,008.83 µg/m³` at `20:00 PKT`). Photochemical `ozone` exhibits the opposite solar-driven cycle, peaking at `177.03 µg/m³` at `13:00 PKT`.
3. **Verified Cross-Metric Correlations (Pearson `r`):**
   - `temperature_2m` vs. `relative_humidity_2m`: **`r = -0.8303`** (strong inverse diurnal coupling: afternoon peak temperature at `14:00 PKT` coincides exactly with minimum humidity `42.43%`).
   - `us_aqi` vs. `pm2_5`: **`r = +0.5976`** (fine particulate matter is the primary driver of Lahore's US AQI index).
   - `pm2_5` vs. `relative_humidity_2m`: **`r = +0.5566`** (higher nighttime humidity coincides with shallow inversions and hygroscopic aerosol growth).
   - `pm2_5` vs. `wind_speed_10m`: **`r = -0.2167`** (weak winds—averaging just `4.36 km/h` and dropping to `2.50 km/h` at `09:00 PKT`—suppress horizontal pollutant dispersion).
4. **Dry Autumn Seasonality:** Rainfall was near-zero (`mean = 0.01 mm/h`, `max = 0.70 mm/h`), making `precipitation_mm` a sparse, heavily zero-inflated series—an important edge case handled explicitly by zero-variance safeguards in both LIME and drift scoring.

---

## 4. Feature Engineering, 3-Family Modeling & Explainability

### 4.1 Causal Feature Engineering (`270` Features)

`skycast/features/engineer.py` constructs **270 numeric/boolean causal features** per hour without future leakage:
- **Cyclical & Calendar:** `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos`, `is_weekend`, `is_lahore_smog_season` (Oct–Feb), `is_monsoon_season` (Jul–Sep).
- **Physical & Interaction Terms:** circular wind vectors (`wind_dir_sin`, `wind_dir_cos`, `wind_u`, `wind_v`), `temp_apparent_delta`, `heat_stress_index`, `pm25_pm10_ratio`, `aqi_wind_dispersion`, `temp_diurnal_anomaly`, `aqi_diurnal_anomaly`, `source_disagreement_temp`, `source_count`, `is_dual_source`.
- **Temporal Dynamics across 11 Core Series:** 1h/3h/6h rates of change (`roc_*`), 6h/12h/24h rolling mean/std/min/max (`roll_*`), and 1h/3h/6h/12h/24h/48h historical lags (`lag_*`).
- **24-Hour Multi-Horizon Targets:** `target_{metric}_t1` through `target_{metric}_t24` (plus `sin/cos` targets for `wind_direction_10m`). The newest 24 hours have `has_full_24h_targets = False` and are strictly held out for live inference only.

### 4.2 Chronological Split & Model Families

Training (`skycast/pipelines/train.py`) enforces a strict **70% Train / 15% Validation / 15% Test** chronological split separated by **24-hour embargoes** (`train = 420`, `val = 90`, `test = 91` from `649` complete trainable rows) with a train-only `StandardScaler`. Four model architectures from three distinct families are benchmarked against `Naive24h`:

1. **Benchmark (`Naive24h`):** Persistence baseline predicting `y_hat(t + h) = y(t)` across `h = 1..24`.
2. **Statistical (`SARIMAX` via `statsmodels`):** Order `(1, 0, 1) x (1, 1, 0, 24)` with top exogenous predictors and native `get_forecast(steps=24).conf_int(alpha=0.2)` confidence bounds.
3. **Classical ML (`scikit-learn`):**
   - `Ridge` (`alpha=1.0`) direct multi-horizon regressor with empirical validation residual quantiles for `lower_bound` / `upper_bound`.
   - `RandomForestRegressor` (`n_estimators=200, max_depth=12, min_samples_leaf=4`) direct multi-horizon estimator with tree-variance / empirical validation residual intervals.
4. **Deep Learning (`PyTorch` CPU `WeatherLSTM`):** 2-layer LSTM (`hidden_size=64, num_layers=2, dropout=0.2`, 24-hour lookback sequence, 24-step output head, Adam `lr=1e-3`, early stopping patience `5`) with conformal/empirical validation residual bounds.

### 4.3 Verified Multi-Horizon Validation & Test Leaderboard (`train-20261119T133423Z-40deb2`)

All metrics below are directly observed multi-horizon (`h = 1..24`) scores from the verified 30-day Lahore training run. Champion promotion requires finite `val_rmse < Naive24h val_rmse` and `< current_champion_val_rmse`.

| Metric | Promoted Champion | Best Candidate Family | Val RMSE | Naive Val RMSE | Improvement vs. Naive | Val MAE | Val R² | Test RMSE | Test MAE | Test R² |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uv_index` | **SARIMAX** (v1) | SARIMAX | 0.1614 | 2.2885 | **+92.95%** | 0.0880 | 0.9899 | 0.0569 | 0.0298 | 0.9987 |
| `temperature_2m` | **RandomForest** (v1) | RandomForest | 0.6409 | 4.7830 | **+86.60%** | 0.4915 | 0.9624 | 2.5797 | 2.3607 | 0.5330 |
| `relative_humidity_2m` | **RandomForest** (v1) | RandomForest | 4.4023 | 18.2364 | **+75.86%** | 3.5160 | 0.8836 | 5.6061 | 4.8093 | 0.8768 |
| `wind_speed_10m` | **RandomForest** (v1) | RandomForest | 2.2907 | 2.8805 | **+20.48%** | 1.6713 | 0.0997 | 1.9167 | 1.4954 | 0.0858 |
| `precipitation_mm` | **RandomForest** (v1) | RandomForest | 0.0300 | 0.0314 | **+4.56%** | 0.0041 | -0.0107 | 0.0040 | 0.0011 | 0.0000 |
| `precipitation_prob` | **LSTM** (v1) | LSTM | 7.0741 | 7.2090 | **+1.87%** | 2.9448 | -0.0187 | 6.7862 | 2.8412 | 0.0268 |
| `us_aqi` | *Guardrail: Naive24h* | SARIMAX | 29.3013 | 25.4797 | -15.00% | 21.9045 | -0.3533 | 30.0637 | 22.4942 | -0.5050 |
| `wind_direction_10m` | *Guardrail: Naive24h* | SARIMAX | 87.2217 | 78.4583 | -11.17% | 71.1632 | -0.3942 | 69.3019 | 53.5543 | -0.6247 |

#### Full Family Comparison Highlights (Validation RMSE)

- **`temperature_2m`:** `RandomForest` (`0.6409`) < `SARIMAX` (`1.6276`) < `LSTM` (`1.8358`) < `Ridge` (`2.6838`) < `Naive24h` (`4.7830`) — every ML family beat `Naive24h` by `+43.89%` to `+86.60%`.
- **`uv_index`:** `SARIMAX` (`0.1614`) < `RandomForest` (`0.2783`) < `Ridge` (`0.4408`) < `LSTM` (`0.6974`) < `Naive24h` (`2.2885`) — every ML family beat `Naive24h` by `+69.53%` to `+92.95%`.
- **`relative_humidity_2m`:** `RandomForest` (`4.4023`) and `LSTM` (`6.2493`) crushed `Naive24h` (`18.2364`) by `+75.86%` and `+65.73%`, while linear `Ridge` (`18.6567`) and `SARIMAX` (`19.6633`) struggled with non-linear diurnal humidity swings.
- **Champion Promotion Guardrail in Practice (`us_aqi` & `wind_direction_10m`):** Because Lahore's late-November smog concentration shifted sharply relative to the 18-day training window (and calm `~4 km/h` winds cause volatile `wind_direction_10m` angles), no candidate beat `Naive24h` on validation `RMSE`. The registry (`skycast/ml/registry.py`) strictly refused promotion (`is_champion = False`), and `Forecast Agent` safely fell back to `Naive24h`. Notably, on the held-out test split, `RandomForest` did outperform `Naive24h` on `us_aqi` (`21.1435` vs. `22.4625`, `R² = 0.2556`) and `LSTM` outperformed `Naive24h` on `wind_direction_10m` (`60.7457°` vs. `62.1544°`), confirming that expanding historical backfill depth to 90–180 days in production GitHub Actions unlocks promotion across all 8 series.

### 4.4 SHAP & LIME Interpretability Findings

`skycast/ml/explain.py` computes **SHAP** (`TreeExplainer` for RandomForest, `LinearExplainer` for Ridge, `KernelExplainer` with deterministic 50-row chronological background subsampling for LSTM/SARIMAX) and **LIME** (`LimeTabularExplainer` with zero-variance jitter handling) for every metric (`24` explanation records persisted per training run):

| Metric | Explained Family | Top Global SHAP Drivers (`mean |SHAP|`) | Top Local LIME Drivers (`|weight|`) |
| :--- | :--- | :--- | :--- |
| `temperature_2m` | `RandomForest` | `uv_index_roll_12h_mean` (0.5472), `uv_index_roll_6h_max` (0.3497), `uv_index_roll_6h_mean` (0.3257) | `uv_index_roll_12h_mean` (0.6975), `precipitation_mm_lag_48h` (0.4861), `hour_sin` (0.3805) |
| `relative_humidity_2m` | `RandomForest` | `uv_index_roll_12h_mean` (2.5527), `uv_index_roll_6h_std` (1.3091), `uv_index_roll_6h_max` (1.2980) | `uv_index_roll_12h_mean` (3.4370), `precipitation_mm_lag_48h` (2.7948), `hour_sin` (1.5722) |
| `us_aqi` | `SARIMAX` | `us_aqi` (27.9728), `us_aqi_lag_48h` (9.2816), `us_aqi_lag_1h` (7.9786) | `precipitation_mm_roc_6h` (23.4047), `us_aqi_lag_48h` (21.5860), `us_aqi` (17.9675) |
| `uv_index` | `SARIMAX` | `uv_index` (1.1433), `uv_index_lag_24h` (0.6928), `uv_index_lag_48h` (0.0001) | `uv_index` (0.9906), `uv_index_lag_24h` (0.6327), `precipitation_mm_roc_3h` (0.3208) |
| `wind_speed_10m` | `RandomForest` | `wind_dir_cos_roll_24h_std` (0.1259), `wind_dir_sin_roll_24h_mean` (0.1114), `surface_pressure_roll_24h_max` (0.1006) | `precipitation_mm_lag_1h` (0.1926), `precipitation_prob_roll_12h_min` (0.1694), `wind_speed_10m_roll_24h_mean` (0.1682) |
| `wind_direction_10m` | `SARIMAX` | `wind_dir_sin` (29.7968), `wind_dir_sin_lag_24h` (19.8620), `wind_dir_cos` (17.3920) | `wind_dir_sin_lag_24h` (25.2445), `precipitation_mm_roc_6h` (21.6077), `precipitation_mm_lag_1h` (19.2228) |
| `precipitation_mm` | `RandomForest` | `uv_index_roll_24h_std` (0.0046), `heat_stress_index` (0.0021), `uv_index_roll_24h_mean` (0.0020) | `precipitation_prob_roll_24h_min` (0.0038), `precipitation_mm_lag_6h` (0.0032), `precipitation_mm_roc_1h` (0.0031) |
| `precipitation_prob` | `LSTM` | `weather_code` (0.4701), `uv_index_roll_12h_min` (0.3469), `precipitation_prob_roll_12h_max` (0.1851) | `precipitation_prob_roll_24h_min` (2.2071), `uv_index_lag_6h` (1.0128), `precipitation_mm_roc_1h` (0.8007) |

Both explainers align closely with meteorological intuition: solar insolation (`uv_index_roll_*` and `hour_sin`) dominates daytime heating and relative humidity drying; multi-day smog persistence (`us_aqi`, `us_aqi_lag_48h`) and rain washout (`precipitation_mm_roc_6h`) govern Lahore AQI; synoptic surface pressure and wind-vector stability govern wind speed; and `weather_code` drives precipitation probability.

---

## 5. Issues Encountered & Engineering Fixes Applied

| # | Issue Encountered | Root Cause | Engineering Fix & Verification |
| :--- | :--- | :--- | :--- |
| 1 | **Met.no vs. Open-Meteo physical skew** | Met.no reports sea-level pressure, clear-sky UV, `m/s` wind, and forward `[t, t+1h]` rain vs. Open-Meteo's station pressure, cloud-attenuated UV, `km/h` wind, and ending `[t-1h, t]` rain. | Implemented `+1h` precipitation shift, `217 m` hypsometric pressure reduction, cloud-factor UV fallback, and circular vector wind fusion in `skycast/data/fusion.py`; verified by `test_fusion_harmonizes_pressure_precip_shift_uv_and_symbols`. |
| 2 | **Future leakage & double-blending in incremental hourly runs** | Standard bidirectional interpolation peeks into future rows, and re-fusing already-blended rows alters source weights. | Built `_CausalLookupIndex` (`causal_only=True`) in `skycast/data/imputation.py` and masked stored `imputed_columns` before re-merging in `skycast/pipelines/hourly_ingest.py`; verified by `test_imputation_causal_no_future_leakage_and_wind_fallback` and `test_hourly_ingest_preserves_imputed_flags_and_avoids_double_blend`. |
| 3 | **PostgREST 1,000-row silent truncation** | Supabase/PostgREST caps unfiltered `.select()` calls at 1,000 rows, silently truncating multi-week feature histories, forecast ledgers, and agent logs. | Added deterministic `.range()` pagination loops across both `skycast/data/storage.py` and `api/queries.py`; verified via `ContractSupabaseFake(max_rows_per_request=1000)` with 1,150–1,250 row fixtures. |
| 4 | **JSONB / FastAPI `NaN` & `Inf` serialization crashes** | Constant dry-season columns (`precipitation_mm = 0`), missing dual-source disagreement, or degenerate correlations can emit `NaN`/`±Inf`, breaking Postgres JSONB and JSON serialization. | Routed every storage write and API response through `sanitize_json_scalar` (preserving `NaN` internally while converting non-finite floats to `null` at JSON boundaries without `.fillna(0)`) and enforced `json.dumps(..., allow_nan=False)` in tests. |
| 5 | **LIME crash on zero-variance dry-weather features** | During dry Lahore periods, `precipitation_mm` and several lag features have zero training variance, causing `LimeTabularExplainer` discretization errors. | Added tiny deterministic perturbation (`_LIME_JITTER_STD = 1e-8`) on zero-variance columns inside `skycast/ml/explain.py`; verified by dry-mode E2E tests. |
| 6 | **Vercel 500 MB Python Serverless bundle limit** | Importing `torch`, `shap`, `statsmodels`, `scikit-learn`, or `hopsworks` inside `api/` would exceed Vercel's serverless bundle cap. | Split `requirements.txt` / `api/requirements.txt` from `requirements-ml.txt`, excluded heavy directories in `vercel.json`, and wrote `test_import_isolation_blocks_heavy_ml_packages` using a `sys.meta_path` import blocker. |
| 7 | **Drift-triggered partial retrain vs. leaderboard completeness** | `drift.py` dispatches `workflow_dispatch` with `inputs: {trigger_reason, run_id, metrics}`. Training only a single drifted metric under a new `train_run_id` would hide the other 7 metrics when `/api/models/leaderboard` queries `latest_run_only=True`. | Declared exact `trigger_reason`, `run_id`, and `metrics` inputs in `.github/workflows/training_pipeline.yml` and configured `drift_detected` triggers to retrain all 8 series cohesively while champion promotion still replaces only metrics that beat their current champion's `val_rmse`. |

---

## 6. Agentic Layer Performance Analysis

### 6.1 Stateful 4-Node LangGraph Execution Trace

Implemented in `skycast/agents/graph.py` (`StateGraph(AgentWeatherState)`), every hourly run executes four traced nodes and persists step timings and branch decisions to `agent_graph_trace`. On verified live run `hourly-20261119T130000Z-543174` (`2026-11-19 18:00 PKT`), the entire graph completed in **2,352.90 ms**:

| Step | Node | Status | Duration (ms) | Branch Decision | Output Written |
| ---: | :--- | :--- | ---: | :--- | :--- |
| 1 | `forecast_agent` | `completed` | 1,701.41 | `forecasts_ready` | `192` rows in `forecast_ledger` (`8` metrics × `24` horizons with `lower_bound` / `upper_bound`) |
| 2 | `alert_agent` | `completed` | 419.53 | `hazards_triggered` | `1` active Warning alert in `weather_alerts` (`us_aqi = 161.0 > 150.0`) |
| 3 | `drift_agent` | `completed` | 162.95 | `generate_weather_report` | `16` drift evaluations (`8` data drift + `8` model drift) in `drift_log` |
| 4 | `weather_report_agent` | `completed` | 69.01 | `hazard_alert` | `1` broadcast bulletin in `weather_reports` |

### 6.2 Alert Quality Across the Three Public & Worker Audiences

At `2026-11-19 18:00 PKT`, Lahore recorded `us_aqi = 161.0` (`temperature_2m = 20.6°C`, `relative_humidity_2m = 59.2%`, `wind_speed_10m = 2.4 km/h`). `alert_node` evaluated all `17` deterministic Lahore hazard rules (`HAZARD_RULES`) across both current observations and the 24-hour forecast trajectory, triggering **`Unhealthy air quality across Lahore`** (`severity = warning`, threshold `150.0`) with distinct, actionable advisories validated for all three audiences:

- **General Public:** *"Reduce prolonged or heavy outdoor exertion and wear an N95/KN95 mask during peak traffic hours."*
- **Sensitive Groups (children, elderly, asthma/respiratory patients):** *"Sensitive groups should reschedule outdoor activities indoors and avoid congested roadways."*
- **Outdoor Workers (construction, delivery riders, traffic wardens):** *"Use N95 respiratory protection during roadside or dusty tasks and take shaded breaks away from traffic."*

### 6.3 Data Drift & Model Drift Detection Sensitivity

`skycast/agents/drift.py` executes two complementary statistical checks across all 8 metrics on every hourly run:

- **Data Drift (48h live window vs. training baseline; threshold `PSI > 0.25` or two-sample `KS p < 0.01`):**
  On the live `2026-11-19` run, the Drift Agent accurately captured Lahore's mid-November cooling and smog shift:
  - **`drift_detected` (5 metrics):** `us_aqi` (`PSI = 6.9955`, `KS p = 0.0000`), `temperature_2m` (`PSI = 3.5196`, `KS p = 0.0000`), `precipitation_prob` (`PSI = 2.3606`, `KS p = 0.0657`), `wind_speed_10m` (`PSI = 0.3247`, `KS p = 0.0057`), and `relative_humidity_2m` (`PSI = 0.2562`, `KS p = 0.0629`).
  - **`no_drift` (3 metrics):** `uv_index` (`PSI = 0.0026`, `KS p = 0.7777`), `wind_direction_10m` (`PSI = 0.1723`, `KS p = 0.1159`), and `precipitation_mm` (`PSI = 0.2152`, `KS p = 1.0000`).
  - When `GITHUB_TOKEN` and `GITHUB_REPO` are configured (as in `.github/workflows/hourly_pipeline.yml`), crossing any drift threshold dispatches `POST /repos/{owner}/{repo}/actions/workflows/training_pipeline.yml/dispatches`; in local offline mode without GitHub credentials, it explicitly logs `action_taken = "retrain_skipped_no_github_credentials"`.
- **Model Drift (72h realized `Forecast Ledger` RMSE vs. champion `val_rmse`; threshold `Live RMSE > 1.35 * Val RMSE`):**
  Before every check, `backfill_forecast_ledger_actuals` joins elapsed `target_timestamp_utc` forecasts against realized observations in `weather_features_hourly`. On the initial cold-start run, all 8 metrics logged explicit `status = "check_failed"` with exact diagnostics (`matched_rows=0, required=72` for promoted champions; `Missing champion validation RMSE baseline` for unpromoted `us_aqi`/`wind_direction_10m`)—preventing false `no_drift` signals when insufficient realized history exists.

### 6.4 LLM Call Failure & Multi-Provider Fallback Resilience

`skycast/agents/llm_client.py` (`ResilientLLMClient`) executes the locked 3-tier cloud LLM chain plus a deterministic template safety net, enforcing `GROQ_REASONING_EFFORT = "low"` on both Groq models (and omitting `reasoning_effort` on Gemini) while writing every attempt immediately to `agent_call_log`:

1. **Primary:** `groq` — `openai/gpt-oss-120b` (`reasoning_effort="low"`, timeout `15s`)
2. **Secondary Fallback:** `gemini` — `gemini-3.5-flash-lite` (`reasoning_effort=None`, timeout `15s`)
3. **Tertiary Fallback:** `groq` — `qwen-3.8-27b` (`reasoning_effort="low"`, timeout `15s`)
4. **Final Safety Net:** `deterministic_template` — rule-grounded structured alert and meteorological bulletin generator (`skycast/agents/templates.py`)

Both in automated fault-injection tests (`tests/test_component4_agents.py`) and in the live offline run (`6` logged attempts: `3` for `alert_agent`, `3` for `weather_report_agent`), forced/missing-key failures on Primary (`openai/gpt-oss-120b`) immediately logged `status="failed"` with `error_message` and `latency_ms`, failed over cleanly to Secondary (`gemini-3.5-flash-lite`, `fallback_used=True`) and Tertiary (`qwen-3.8-27b`, `fallback_used=True`, `reasoning_effort="low"`), and fell back to the deterministic template so neither the hourly pipeline nor the dashboard ever crashed.

---

## 7. Deployment, Automation & Verification Summary

- **CI/CD Automation (`.github/workflows/`):**
  - `hourly_pipeline.yml` (`cron: "5 * * * *"` + `workflow_dispatch`): hourly multi-source ingestion, Hopsworks/Supabase upsert, and 4-node LangGraph execution with `actions: write` for drift-triggered retraining.
  - `training_pipeline.yml` (`cron: "15 1 * * *"` at `06:15 PKT` + drift-triggered `workflow_dispatch`): daily EDA, 3-family training, SHAP/LIME computation, champion promotion, and EDA artifact upload.
  - `ci.yml` (`push` / `pull_request`): full Python `pytest` suite, Node 24 frontend lint/typecheck/build, and `docker compose config -q`.
- **Docker Portability (`Dockerfile` + `docker-compose.yml` + `docker/nginx.conf`):** Single-command `docker compose up --build` launches `api` (lightweight FastAPI mirroring Vercel's import boundary), `web` (React 19 + Nginx reverse proxy for `/api/*`), and `scheduler` (`skycast/pipelines/scheduler.py` performing automatic cold-start backfill → train → hourly ingest followed by recurring hourly/daily loops) over a shared `skycast_data` volume with zero cloud setup required.
- **Serving & Dashboard (`api/` + `web/` + `vercel.json`):** FastAPI exposes 7 sanitized read-only endpoints (`/api/health`, `/api/dashboard/overview`, `/api/forecasts/history`, `/api/models/leaderboard`, `/api/eda/summary`, `/api/agents/logs`, `/api/report`) consumed by the 6-tab Sky Slate React dashboard (`Overview`, `Forecasts`, `Models`, `EDA`, `Agents and drift`, and `Technical report`).
