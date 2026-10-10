# Sentinel PdM: results

_Generated 2026-10-10 01:07 UTC by `python -m sentinel_pdm report` (sentinel-pdm v1.0.0) from the training run of 2026-10-09 20:33 UTC (v1.0.0). Interactive figures: [model_report.html](model_report.html). Method: [docs/METHODOLOGY.md](../docs/METHODOLOGY.md). Limitations: [docs/MODEL_CARD.md](../docs/MODEL_CARD.md)._

## Summary

- On 16 held-out pumps (16 pump-years, never seen in training) with 31 failures, of which 27 had a measurable precursor, the Sentinel hybrid caught 27/31 (27/27 predictable) with 0 false alarms and a median warning of 9.1 days before functional failure.
- Conventional alarm limits (ISO 10816 zone C, bearing / winding RTD alarms) caught 8/31 on the same pumps with a median warning of 32 h, usually too late to plan the repair.
- Repeated on 5 independent simulated fleets (150 held-out failures, 80 pump-years), the hybrid caught 124/124 predictable failures with 2 false alarms (gradient boosting alone 10; MSPC alone 15); alarm limits caught 43/124.
- At the stated cost assumptions this is a 68.8% maintenance-cost saving versus run-to-failure ($5.02M -> $1.57M for the test fleet) and 72.6% less downtime.
- In cross-validation on the training fleet, catching 56/65 failures cost the hybrid 0 false alarms and gradient boosting alone 60: requiring the supervised model and MSPC to agree is what removes nuisance alerts.

## Held-out test fleet

All five methods scored the same 16 pumps that no model or threshold was fitted on. ROC-AUC / PR-AUC rank pump-hours (label: failure within the horizon); the event columns count alert raises matched to failures by P-F interval. 'Predictable' excludes sudden failures, which have no precursor.

| Method | ROC-AUC | PR-AUC | Caught / failures | Predictable caught | False alarms | Median lead (days) | Saving vs run-to-failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sentinel hybrid (GBM + MSPC) | 0.939 | 0.682 | 27/31 | 27/27 | 0 | 9.1 | 68.8% |
| Gradient boosting only | 0.955 | 0.647 | 26/31 | 26/27 | 5 | 7.4 | 66.1% |
| MSPC (PCA T² / SPE) | 0.928 | 0.695 | 27/31 | 27/27 | 0 | 11.1 | 68.8% |
| Isolation Forest | 0.934 | 0.643 | 25/31 | 25/27 | 2 | 8.5 | 63.6% |
| Alarm limits (ISO 10816 + RTD) | 0.672 | 0.217 | 8/31 | 8/27 | 0 | 1.3 | 20.4% |

Ranking quality excluding the 7-day windows before sudden failures (which no sensor can rank):

| Method | ROC-AUC | PR-AUC | Brier |
| --- | ---: | ---: | ---: |
| Sentinel hybrid (GBM + MSPC) | 0.989 | 0.758 | 0.015 |
| Gradient boosting only | 0.979 | 0.707 | 0.017 |
| MSPC (PCA T² / SPE) | 0.990 | 0.777 | - |
| Isolation Forest | 0.984 | 0.716 | - |
| Alarm limits (ISO 10816 + RTD) | 0.680 | 0.229 | - |

### Cross-validation (training fleet, out-of-fold)

Each training pump scored by models that never saw it; thresholds were tuned on these scores.

| Method | ROC-AUC | PR-AUC | Caught / failures | False alarms | Median lead (days) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sentinel hybrid (GBM + MSPC) | 0.922 | 0.631 | 56/65 | 0 | 10.0 |
| Gradient boosting only | 0.922 | 0.672 | 55/65 | 5 | 8.9 |
| MSPC (PCA T² / SPE) | 0.922 | 0.586 | 57/65 | 6 | 10.5 |
| Isolation Forest | 0.919 | 0.552 | 49/65 | 13 | 9.0 |
| Alarm limits (ISO 10816 + RTD) | 0.706 | 0.215 | 16/65 | 0 | 2.4 |

## Economics

Caught failures become planned repairs; missed ones remain breakdowns with secondary damage and a longer outage; every false alarm costs an inspection. Test fleet, 16 pump-years.

| Policy | Breakdowns | Planned repairs | False alarms | Downtime (h) | Total cost | Per pump-year | Saving | Less downtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run to failure (reactive) | 31 | 0 | 0 | 1,488 | $5.02M | $314k | 0.0% | 0.0% |
| Sentinel hybrid (GBM + MSPC) | 4 | 27 | 0 | 408 | $1.57M | $98k | 68.8% | 72.6% |
| Gradient boosting only | 5 | 26 | 5 | 448 | $1.70M | $106k | 66.1% | 69.9% |
| MSPC (PCA T² / SPE) | 4 | 27 | 0 | 408 | $1.57M | $98k | 68.8% | 72.6% |
| Isolation Forest | 6 | 25 | 2 | 488 | $1.83M | $114k | 63.6% | 67.2% |
| Alarm limits (ISO 10816 + RTD) | 23 | 8 | 0 | 1,168 | $4.00M | $250k | 20.4% | 21.5% |

## Remaining useful life and diagnosis

RUL is conditional on a developing failure (P10 / P50 / P90 hours, Mondrian normalised conformal intervals). Evaluated on test-fleet hours before predictable failures.

| Window | Pump-hours | MAE of P50 (h) | P10-P90 coverage | Median interval width (h) |
| --- | ---: | ---: | ---: | ---: |
| Last 7 days before failure | 4,486 | 98 | 71% | 236 |
| Last 30 days before failure | 19,239 | 115 | 77% | 376 |

Failure-mode diagnosis: 89% of predictable test failures (24/27) correctly classified by averaging the mode probabilities over the 7 days before failure; 86% of individual pre-failure hours.

| Actual \ predicted | Bearing | Misalignment | Cavitation | Winding | Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Bearing** | 14 | 0 | 0 | 0 | 14/14 |
| **Misalignment** | 2 | 2 | 0 | 0 | 2/4 |
| **Cavitation** | 1 | 0 | 5 | 0 | 5/6 |
| **Winding** | 0 | 0 | 0 | 3 | 3/3 |

## Top-10 features

Permutation importance of the gradient-boosting risk ensemble on test-fleet hours (drop in PR-AUC when the feature is shuffled; mean ± std over repeats).

| # | Feature | Meaning | PR-AUC drop |
| ---: | --- | --- | ---: |
| 1 | `vib_hf_sl168` | High-frequency bearing band, 7-day trend | 0.0309 ± 0.0009 |
| 2 | `current_imbalance_m168` | Phase current imbalance, 7-day mean | 0.0198 ± 0.0017 |
| 3 | `vib_rms_sl168` | Overall vibration (RMS), 7-day trend | 0.0169 ± 0.0002 |
| 4 | `suction_pressure_sl168` | Suction pressure, 7-day trend | 0.0139 ± 0.0012 |
| 5 | `vib_kurtosis_dev` | Vibration kurtosis (impacting), deviation from baseline | 0.0092 ± 0.0009 |
| 6 | `current_imbalance_dev` | Phase current imbalance, deviation from baseline | 0.0088 ± 0.0004 |
| 7 | `vib_rms_dev` | Overall vibration (RMS), deviation from baseline | 0.0083 ± 0.0009 |
| 8 | `vib_kurtosis_m24` | Vibration kurtosis (impacting), 24 h mean | 0.0082 ± 0.0008 |
| 9 | `vib_hf_dev` | High-frequency bearing band, deviation from baseline | 0.0082 ± 0.0013 |
| 10 | `vib_hf_sl24` | High-frequency bearing band, 24 h trend | 0.0081 ± 0.0002 |

## Robustness across independent fleets

The entire pipeline (simulation, cleaning, features, grouped cross-validation, threshold choice, evaluation) re-run on 5 statistically independent fleets (seeds 101, 102, 103, 104, 105; 48 pumps x 8,760 h each, 16 held out per fleet). Pooled counts weight every failure equally; ± is the standard deviation across fleets.

| Method | Predictable caught (pooled) | Detection per fleet | False alarms (pooled) | False alarms / pump-year | Median lead (h) | Cost saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sentinel hybrid (GBM + MSPC) | 124/124 | 100 ± 0% | 2 | 0.03 ± 0.03 | 281 ± 47 | 65.6 ± 6.6% |
| Gradient boosting only | 120/124 | 97 ± 7% | 10 | 0.12 ± 0.25 | 239 ± 32 | 63.4 ± 8.0% |
| MSPC (PCA T² / SPE) | 124/124 | 100 ± 0% | 15 | 0.19 ± 0.12 | 236 ± 14 | 65.5 ± 6.7% |
| Isolation Forest | 117/124 | 94 ± 5% | 36 | 0.45 ± 0.74 | 195 ± 64 | 61.7 ± 7.8% |
| Alarm limits (ISO 10816 + RTD) | 43/124 | 35 ± 5% | 0 | 0.00 ± 0.00 | 64 ± 15 | 22.8 ± 4.2% |

- Pooled over all fleets the Sentinel hybrid caught 124 of 124 predictable failures (100.0%) with 2 false alarms in 80 pump-years; conventional alarm limits caught 43 of 124 (34.7%).
- Remaining useful life in the last 7 days before failure: median-estimate error 78 ± 15 h, P10-P90 coverage 84 ± 6% (target 80%). Failure mode correctly diagnosed for 90 ± 7% of predictable failures.

Per-fleet tables: [ROBUSTNESS.md](ROBUSTNESS.md).

_Source: `python -m sentinel_pdm robustness`._

## Real-data benchmark: UCI AI4I 2020

AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020; UCI id 601, CC BY 4.0): 10,000 machining runs with 339 failures (3.4%). Built independently of this project (its author describes it as synthetic data reflecting real industrial predictive-maintenance data), so it tests whether the approach only works on our own simulator. Protocol: 3 x repeated stratified 5-fold cross-validation on identical folds for every model; the failure-type columns never reach a model.

| Features | Model | ROC-AUC | PR-AUC | F1 (cross-fitted threshold) | Recall at 90% precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw sensors | Logistic regression | 0.895 ± 0.001 | 0.451 ± 0.001 | 0.478 ± 0.008 | 13% |
| Raw sensors | Random forest | 0.966 ± 0.002 | 0.801 ± 0.009 | 0.760 ± 0.008 | 60% |
| Raw sensors | Gradient boosting | 0.977 ± 0.000 | 0.820 ± 0.011 | 0.755 ± 0.014 | 60% |
| Physics-informed | Logistic regression | 0.917 ± 0.001 | 0.484 ± 0.000 | 0.488 ± 0.005 | 14% |
| Physics-informed | Random forest | 0.977 ± 0.001 | 0.900 ± 0.000 | 0.885 ± 0.005 | 84% |
| Physics-informed | Gradient boosting | 0.982 ± 0.001 | 0.890 ± 0.003 | 0.856 ± 0.001 | 82% |

- Physics-informed features (temperature rise, mechanical power, overstrain) lift the best PR-AUC from 0.820 (Gradient boosting, raw sensors) to 0.900 (Random forest): encoding the failure physics matters more than model capacity.
- At its best-F1 threshold the best physics-informed model catches 83% of failures at 96% precision (10.7 false alarms per 10,000 runs).
- Only 287 of 339 failures (85%) follow a mechanism visible in a single snapshot; tool-wear and random failures are not predictable from the inputs, the same lesson as the sudden pump failures.
- Blocked cross-validation (contiguous blocks of runs) gives PR-AUC 0.902 versus 0.900 with random folds: the result does not rely on leakage between neighbouring rows.

_Source: `python -m sentinel_pdm ai4i`._

## Data

| Item | Value |
| --- | --- |
| Fleet | 48 centrifugal pumps in 8 process units (32 training, 16 held-out test) |
| History | 8,760 h per pump (hourly historian averages from 2025-01-01); 420,480 pump-hours |
| Failures | 96 (bearing 38, cavitation 19, winding 14, misalignment 13, sudden 12); 65 in the training fleet, 31 in the test fleet |
| Test-fleet failures | bearing 14, cavitation 6, sudden 4, misalignment 4, winding 3 |
| Preventive maintenance | 93 PM work orders |
| Model features | 107 causal features per pump-hour |
| Data quality | pumps running 97.2% of hours; 0.52% of running sensor values missing; 4,792 spikes and 12,148 stuck-tag hours removed by causal cleaning |

Simulated with a physics-informed fleet model; see [METHODOLOGY](../docs/METHODOLOGY.md#2-the-simulated-fleet).

## Key configuration

| Setting | Value |
| --- | --- |
| Prediction horizon | failure within 168 h (7.0 days) |
| RUL target cap | 720 h; conformal P10-P90 target coverage 80% |
| Alarm budget | <= 0.5 false alarms per pump-year (thresholds minimise cost within this budget, tuned out-of-fold on the training fleet) |
| Alert logic | raise after 3 h above threshold, clear after 24 h below, re-alert every 168 h while active |
| Event matching | P-F interval; alerts later than F - 12 h count as late; 336 h fallback window when no P point is logged |
| Costs (USD) | lost production $2,500/h; unplanned repair $42,000 + 48 h down; planned repair $14,000 + 8 h down; false-alarm inspection $1,800 |
| Gradient boosting | 4-fold GroupKFold by pump; 250 iterations, learning rate 0.05, 15 leaves, min 200 samples/leaf, 0.5 feature fraction, L2 2.0; trained on one hour in 3 |
| Alert thresholds | Sentinel hybrid (GBM + MSPC) 0.188; Gradient boosting only 0.292; MSPC (PCA T² / SPE) 4.59 x 99 % control limit; Isolation Forest 1.10 x healthy P99; Alarm limits (ISO 10816 + RTD) 1.00 x alarm limit |

## Run details

- Environment: Python 3.12.10, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3
- Pipeline time: 133 s on CPU (simulate 1 s, clean 2 s, features 2 s, train 57 s, score 13 s, evaluate 1 s, importance 54 s)
- Model version 1.0.0, trained 2026-10-09 20:33 UTC; report generated 2026-10-10 01:07 UTC
