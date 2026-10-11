# Model card: Sentinel PdM v1.0.0

A summary of what the Sentinel PdM models are, what they were evaluated on, how well they did and
where they must not be trusted, in the spirit of *Model Cards for Model Reporting* (Mitchell et al.,
2019). The full method is in [METHODOLOGY.md](METHODOLOGY.md); every number below is copied from
`artifacts/metrics.json` (reference run, seed 42), `reports/robustness.json` (five independent
fleets) or `artifacts/ai4i_metrics.json` (real-data benchmark), and is reproduced in
[reports/RESULTS.md](../reports/RESULTS.md).

| | |
|---|---|
| **Developer** | Daiten Garza |
| **Version** | 1.0.0, trained 2026-10-09 |
| **Type** | Tabular time-series models: gradient boosting, multivariate statistical process control, quantile regression with conformal calibration |
| **Framework** | scikit-learn (trained with 1.9.1) on Python 3.10+; CPU only, about two minutes to train end to end |
| **Artifact** | `artifacts/models/bundle.joblib` (every model, thresholds, calibrators and reference statistics in one file) |
| **Licence** | MIT |

---

## 1. Model details

Sentinel PdM scores every running pump every hour and answers three questions: *will it fail within
the next 7 days*, *if a failure is developing, how long is left*, and *what is failing*.

| Output | Model | Notes |
|---|---|---|
| `risk`: P(failure within 7 days), drives alerts and status | **Sentinel hybrid**: the minimum of the percentile ranks of the gradient-boosting risk and the MSPC anomaly score (a soft AND), calibrated to a probability with isotonic regression | An alert needs the supervised model *and* the statistical monitor to agree |
| `risk_gbm` | `HistGradientBoostingClassifier`, cross-validation ensemble of 4 fold models, isotonic calibration | 250 iterations, learning rate 0.05, 15 leaves, at least 200 samples per leaf |
| `mspc_score` | PCA on the 17 deviation-from-baseline features (components for 90 % of variance); Hotelling T² and SPE (Q) against empirical 99 % control limits | 1.0 = on the 99 % control limit; fitted on healthy hours only |
| `rul_p10`, `rul_p50`, `rul_p90` | Three quantile-loss gradient-boosting regressors (α = 0.1, 0.5, 0.9), conditional on a developing failure, capped at 720 h | P10-P90 widened by Mondrian normalised conformalised quantile regression to an 80 % target |
| `prob_bearing`, `prob_misalignment`, `prob_cavitation`, `prob_winding` | Balanced-class gradient-boosting classifier trained on the pre-failure hours of predictable failures | The most likely mode selects the maintenance recommendation |
| `iforest_score`, `rule_score` | Isolation Forest; conventional alarm limits (2.8 mm/s overall vibration, 82 °C bearing RTD, 130 °C winding RTD) | Comparison baselines only, never used for decisions |

**Inputs.** Hourly historian averages of 12 sensor tags per pump (overall, 1x, 2x and
high-frequency vibration, kurtosis, bearing and winding RTDs, motor current, phase-current imbalance,
suction and discharge pressure, flow), the operating context (process load, speed, ambient
temperature) and the maintenance log (work orders). Causal cleaning and feature engineering turn
these into 107 features per pump-hour: six trailing statistics for each of 17 signals (the 12 tags,
four physics-derived signals such as differential head, and process load), including each signal's
deviation from the pump's own post-maintenance baseline, plus operating context and time since
maintenance.

**Alert logic.** An alert is raised when the hybrid risk stays at or above **0.188** for 3 h,
clears after 24 h below it, and re-notifies every 7 days while active. The threshold minimises
maintenance cost subject to an alarm budget of at most 0.5 false alarms per pump-year, tuned on
out-of-fold scores of the training pumps and then frozen. Each pump also gets a status:
*Critical* (alert and median remaining life <= 72 h), *Alert*, *Watch* (risk above half the
threshold, or MSPC beyond 1.5x its limit), *Normal* or *Offline*.

**Explanations.** Robust z-scores of each signal against healthy behaviour (the "drivers" shown with
every alert), permutation feature importance, and a maintenance recommendation per failure mode.

---

## 2. Intended use

**Primary use: decision support for reliability engineers**, maintenance planners and
rotating-equipment specialists responsible for a fleet of centrifugal process pumps.

* Prioritise which pumps to inspect, and when, across a fleet too large to trend by hand.
* Turn a developing failure into a planned repair: an alert is a **work request for an
  inspection** (vibration analysis, thermography, lubrication check), not an instruction to repair.
  The remaining-life estimate helps schedule the repair window; the diagnosed mode suggests parts,
  crafts and checks.
* Review the evidence: every alert comes with the signals that drive it, so an engineer can agree or
  disagree with it.

**Human in the loop.** A qualified person reviews every alert and confirms the defect by inspection
before any work is planned. The system recommends; people decide. Changes to thresholds or models
should go through the site's management-of-change process.

**Not a safety system.** Sentinel PdM must not be used:

* as, or as part of, a safety instrumented system, machine protection or any protective function;
* to bypass, inhibit, defeat or delay a protective trip, interlock or alarm (for example a vibration,
  high-temperature or low-flow trip) because the risk score is low;
* to start, stop or trip equipment automatically, or as a permissive in a control sequence;
* to extend inspection or preventive-maintenance intervals required by procedure, regulation or the
  manufacturer, or to skip an inspection because no alert was raised;
* on machine types, plants or data it has not been retrained and validated on;
* to evaluate the performance of individual people.

**A low risk score is not a clean bill of health.** About one failure in eight has no measurable
precursor (Section 5), and inputs unlike anything in training (a new pump type, a failed transmitter,
an unusual operating regime) can produce confident but meaningless scores.

---

## 3. Data

**Training and evaluation data are simulated.** Real run-to-failure histories with matching
maintenance logs are proprietary, so the models are trained on a physics-informed simulator of a
Gulf-Coast process plant (`src/sentinel_pdm/simulator.py`, described in
[METHODOLOGY.md, Section 2](METHODOLOGY.md#2-the-simulated-fleet)).

| Item | Reference run (seed 42) |
|---|---|
| Fleet | 48 centrifugal pumps in 8 process units; 32 training pumps, 16 held-out test pumps (split by pump) |
| History | 8,760 hourly records per pump (one year), 420,480 pump-hours |
| Failures | 96: bearing 38, cavitation 19, winding 14, misalignment 13, sudden (no precursor) 12 |
| Split | 65 failures in the training fleet; 31 in the test fleet, 27 of them predictable |
| Preventive maintenance | 93 PM work orders, 12 of which found a developing defect (right-censored degradations) |
| Realism built in | operating-point changes, VFD pumps, daily and seasonal ambient cycles, process outages, per-failure symptom variation and secondary damage, spikes, stuck tags, dropouts, historian outages, slow RTD drift |

Labels come only from the maintenance log, as they would in a plant; the simulator's ground-truth
columns are never model inputs. The data contains no personal information and no real company data.

**Real-data check.** To test the approach outside its own simulator, the same feature-design idea is
benchmarked on the public **UCI AI4I 2020 Predictive Maintenance Dataset** (Matzka, 2020; 10,000
machining runs, 339 failures; CC BY 4.0; DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)).
Its author describes it as synthetic data reflecting real industrial predictive maintenance; it was
built independently of this project. It checks the method, not the pump models: no pump model is
trained or evaluated on it.

---

## 4. Evaluation protocol

* **Held-out pumps.** 16 pumps are never seen by any model, calibrator or threshold. Inside the
  training fleet, every model is fitted with 4-fold cross-validation grouped by pump, and everything
  tuned (probability calibration, conformal offsets, hybrid fusion, alert thresholds) is tuned on
  out-of-fold scores.
* **Events, not just rows.** Alerts are matched to failures by the P-F interval: an alert counts as
  catching a failure only if it is raised after the defect became detectable (P) and at least 12 h
  before functional failure (F). Alerts before P, or matched to nothing, are false alarms; sudden
  failures stay in the denominator. Lead time is F minus the first useful alert.
* **Alarm budget and cost.** Thresholds minimise total maintenance cost within at most 0.5 false
  alarms per pump-year. Cost uses illustrative figures: unplanned repair $42,000 + 48 h down, planned
  repair $14,000 + 8 h down, lost production $2,500/h, false-alarm inspection $1,800.
* **Row-level metrics** for completeness: ROC-AUC, PR-AUC and Brier score on running hours with a
  known label.
* **Baselines on identical data**: gradient boosting alone, MSPC alone, Isolation Forest and
  conventional alarm limits.
* **Robustness.** The entire pipeline (simulation, cleaning, features, training, threshold choice,
  evaluation) is repeated on five independent fleets (seeds 101-105) to measure how much of each
  number is the luck of one draw.

---

## 5. Metrics

### 5.1 Held-out test fleet (reference run: 16 pumps, 16 pump-years, 31 failures)

| Method | ROC-AUC | PR-AUC | Failures caught | Predictable caught | False alarms | Median warning | Cost saving vs run-to-failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Sentinel hybrid (GBM + MSPC)** | 0.939 | 0.682 | **27/31** | **27/27** | **0** | 218 h (9.1 d) | **68.8 %** |
| Gradient boosting only | 0.955 | 0.647 | 26/31 | 26/27 | 5 | 178 h (7.4 d) | 66.1 % |
| MSPC (PCA T² / SPE) | 0.928 | 0.695 | 27/31 | 27/27 | 0 | 266 h (11.1 d) | 68.8 % |
| Isolation Forest | 0.934 | 0.643 | 25/31 | 25/27 | 2 | 203 h (8.5 d) | 63.6 % |
| Alarm limits (ISO 10816 + RTD) | 0.672 | 0.217 | 8/31 | 8/27 | 0 | 32 h (1.3 d) | 20.4 % |

Calibration (Brier score): hybrid 0.019, gradient boosting 0.021. Excluding the 7-day windows before
sudden failures, the hybrid's ROC-AUC is 0.989 and PR-AUC 0.758.

### 5.2 Five independent fleets (80 pump-years, 150 held-out failures, 124 predictable)

| Method | Predictable caught (pooled) | False alarms (pooled) | False alarms / pump-year | Median warning (h) | Cost saving |
|---|---:|---:|---:|---:|---:|
| **Sentinel hybrid (GBM + MSPC)** | **124/124** | **2** | 0.03 ± 0.03 | 281 ± 47 | 65.6 ± 6.6 % |
| Gradient boosting only | 120/124 | 10 | 0.12 ± 0.25 | 239 ± 32 | 63.4 ± 8.0 % |
| MSPC (PCA T² / SPE) | 124/124 | 15 | 0.19 ± 0.12 | 236 ± 14 | 65.5 ± 6.7 % |
| Isolation Forest | 117/124 | 36 | 0.45 ± 0.74 | 195 ± 64 | 61.7 ± 7.8 % |
| Alarm limits (ISO 10816 + RTD) | 43/124 | 0 | 0.00 ± 0.00 | 64 ± 15 | 22.8 ± 4.2 % |

± is the standard deviation across fleets. Two false alarms in 80 pump-years is consistent with a true
rate of at most about 0.08 per pump-year (95 % upper bound, Poisson).

### 5.3 Remaining useful life and diagnosis

| Metric | Reference test fleet | Five fleets (mean ± sd) |
|---|---:|---:|
| RUL error (MAE of P50), last 7 days before failure | 98 h | 78 ± 15 h |
| P10-P90 coverage, last 7 days (target 80 %) | 71 % | 83.6 ± 6.2 % |
| RUL error (MAE of P50), last 30 days | 115 h | - |
| P10-P90 coverage, last 30 days | 77 % | - |
| Failure mode correctly diagnosed (predictable failures) | 24/27 (89 %) | 90.3 ± 7.2 % |

Diagnosis on the reference fleet by mode: bearing 14/14, cavitation 5/6, winding 3/3,
misalignment 2/4 (both misses were read as bearing failures, the usual secondary damage of
misalignment).

### 5.4 Real-data benchmark: AI4I 2020 (3 x repeated stratified 5-fold cross-validation)

| Feature set (best model) | ROC-AUC | PR-AUC | F1 (cross-fitted threshold) | Recall at 90 % precision |
|---|---:|---:|---:|---:|
| Raw sensors (gradient boosting) | 0.977 | 0.820 | 0.755 | 60 % |
| Physics-informed (random forest) | 0.977 | **0.900** | **0.885** | **84 %** |

Three physics-informed features (temperature rise, mechanical power, tool wear x torque) add 0.08
PR-AUC; at the best-F1 threshold false alarms fall from 60 to 10.7 per 10,000 runs. Only 287 of 339
failures (85 %) follow a mechanism visible in a single snapshot, which caps any model.

---

## 6. Limitations

* **Simulated data.** All pump results come from a simulator that encodes one engineer's
  understanding of pump failure physics. Absolute numbers depend on its assumptions (P-F interval
  lengths, sensor noise, how strongly symptoms show); the comparison between methods on identical
  data is the meaningful part. The models have not been validated on real plant data and must be
  retrained and re-validated on a site's own historian and CMMS data before any use.
* **Sudden failures are undetectable.** About 12 % of simulated failures (4 of 31 in the reference
  test fleet, 26 of 150 across the robustness fleets) have no precursor. No sensor-based method
  catches them, and none of the reported numbers pretend otherwise.
* **Single plant archetype.** One kind of plant: motor-driven centrifugal process pumps, hourly
  historian averages, a fixed tag list, four failure modes. Other machines (compressors, turbines,
  reciprocating or vertical pumps), other failure modes (mechanical-seal leaks, for example) or
  sparse route-based vibration data are outside what was tested.
* **Costs are illustrative.** The dollar figures are plausible Gulf-Coast process-plant numbers, not
  any company's actuals; savings percentages move with them.
* **Small samples.** 31 test failures in the reference run; even the five-fleet study has 124
  predictable failures from one simulator. Differences between the machine-learning methods that sit
  inside the ± ranges should not be ranked.
* **RUL intervals are approximate.** Final-week P10-P90 coverage was 71 % on the reference fleet and
  83.6 ± 6.2 % across five fleets, against an 80 % target. Read P10 as "plan before this", not as a
  guarantee.
* **Diagnosis is weakest for misalignment**, whose secondary bearing damage can dominate the
  symptoms; a bearing work order on a pump with rising 2x vibration should include an alignment check.
* **Instrument problems.** Causal cleaning removes spikes, stuck tags and short gaps, and MSPC
  separates broken correlations from plausible ones, but a slowly drifting transmitter can still look
  like a developing fault. Every alert needs a physical inspection.
* **Out-of-distribution inputs.** Missing tags, new operating regimes or retrofits (a VFD added, a
  re-rated impeller) change the inputs; the models will still produce scores, with no guarantee they
  mean anything.
* **Alarm fatigue and automation bias.** A system that is usually right can teach people to stop
  looking for what it misses. Keep routine condition monitoring and operator rounds in place.

---

## 7. Maintenance and retraining

**Monitor continuously:**

* false alarms per pump-year, per unit, on a rolling 12-month window: the alarm budget (0.5) is the
  primary health metric;
* the inspection outcome of every alert (defect found / not found), recorded in the CMMS;
* every breakdown: was there an alert, how much warning did it give, was the mode right;
* data quality per pump: missing values, stuck tags, spikes, instrument drift;
* score drift: the share of running hours above the alert threshold and above the MSPC control limit
  on pumps believed healthy.

**Retrain when**

* the false-alarm rate exceeds the budget for a quarter, or a predictable failure is missed;
* equipment changes: pumps added or replaced, impellers re-rated, VFDs retrofitted, transmitters
  replaced or re-ranged;
* the process changes: new products or grades, new operating envelope;
* otherwise on a fixed cadence (every 6 to 12 months) as new work orders accumulate.

**How**

1. Make sure the maintenance log is complete. Feature windows and baselines restart at every work
   order, so a missing work order corrupts the baseline of the cycle that follows it.
2. Re-run the training. Note that `python -m sentinel_pdm train` as shipped always simulates a fresh
   fleet and overwrites `data/`; training on site data needs a small loader that builds the same three
   inputs from the historian and the CMMS (hourly readings without `gt_*` columns, the work-order log
   with failure modes and, ideally, P points from root-cause analysis, and the asset register) and
   passes them to the cleaning / feature / `train_models` steps in place of `simulate_fleet`. Those
   steps then refit every model with grouped cross-validation and re-tune calibration, conformal
   offsets and thresholds on the new out-of-fold scores.
3. Never mix a new model with old thresholds, or the other way round: thresholds, calibrators and
   conformal offsets are only valid for the exact models they were tuned with, which is why they are
   saved together in one bundle.
4. Compare the candidate with the current version on the same held-out pumps (detection, lead time,
   false alarms per pump-year, cost), and repeat the comparison across resampled splits before
   promoting it. Keep the previous `bundle.joblib` and `metrics.json` so a rollback is one file copy.
5. Record the change, the evidence and the approver under management of change.

---

## 8. Citation and contact

Sentinel PdM by **Daiten Garza**, MIT licence. AI4I 2020 data: Matzka, S. (2020). *AI4I 2020
Predictive Maintenance Dataset* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C (CC BY 4.0).

Reference: Mitchell, M. et al. "Model Cards for Model Reporting." *Proceedings of the Conference on
Fairness, Accountability, and Transparency (FAT\*)*, 2019.
