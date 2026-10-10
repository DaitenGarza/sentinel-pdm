# Robustness across independent fleets

_Generated 2026-10-10T02:03:43+00:00 by `python -m sentinel_pdm robustness` (Sentinel PdM v1.0.0)._

## Why this study

A single held-out test fleet produces only a few dozen failures, so one failure more or less moves a detection rate by several points and one nuisance alert moves the false-alarm rate noticeably. A method can look best on one fleet by luck. This study re-simulates the plant with different random seeds - new pumps, operating profiles, failure histories, instrument faults and train/test split - and re-runs the **entire** pipeline on each fleet: cleaning, features, grouped cross-validation training, cost-optimal threshold choice on the training pumps, and evaluation on pumps the models never saw. The spread across fleets is the honest error bar on every headline number.

## Setup

* **5 independent fleets** (seeds 101, 102, 103, 104, 105), each 48 pumps x 8,760 h of hourly data; 16 pumps per fleet are held out for testing.
* Test failures per fleet: 30.0 ± 1.6 (range 28 to 32); 150 test failures in total, 124 of them predictable (the rest are *sudden* failures with no measurable precursor, which no method can catch).
* Gradient boosting: 250 iterations, 4-fold cross-validation by pump. Alert thresholds are chosen per fleet on out-of-fold training scores under a budget of 0.5 false alarms per pump-year.
* Total compute: 6.3 min.

## Headline: test-fleet results, mean ± standard deviation across fleets

| Method | Predictable failures caught | False alarms / pump-year | Median lead time (h) | Cost saving vs run-to-failure |
|---|---:|---:|---:|---:|
| Sentinel hybrid (GBM + MSPC) | 100.0 ± 0.0 % | 0.03 ± 0.03 | 281 ± 47 | 65.6 ± 6.6 % |
| Gradient boosting only | 96.8 ± 7.2 % | 0.12 ± 0.25 | 239 ± 32 | 63.4 ± 8.0 % |
| MSPC (PCA T² / SPE) | 100.0 ± 0.0 % | 0.19 ± 0.12 | 236 ± 14 | 65.5 ± 6.7 % |
| Isolation Forest | 94.3 ± 4.6 % | 0.45 ± 0.74 | 195 ± 64 | 61.7 ± 7.8 % |
| Alarm limits (ISO 10816 + RTD) | 34.6 ± 4.5 % | 0.00 ± 0.00 | 64 ± 15 | 22.8 ± 4.2 % |

A failure counts as caught only if the method warned at least 12 h ahead, inside the P-F interval (after the defect became detectable). A false alarm is an alert whose inspection would have found nothing. Lead time runs from the first useful alert to the failure. The cost saving compares planned repairs plus inspections against running every pump to failure.

### Hour-by-hour ranking (mean ± sd)

| Method | ROC-AUC | PR-AUC | ROC-AUC, predictable | PR-AUC, predictable |
|---|---:|---:|---:|---:|
| Sentinel hybrid (GBM + MSPC) | 0.909 ± 0.055 | 0.636 ± 0.054 | 0.988 ± 0.002 | 0.747 ± 0.024 |
| Gradient boosting only | 0.911 ± 0.058 | 0.650 ± 0.063 | 0.986 ± 0.008 | 0.763 ± 0.026 |
| MSPC (PCA T² / SPE) | 0.901 ± 0.055 | 0.624 ± 0.065 | 0.988 ± 0.002 | 0.736 ± 0.039 |
| Isolation Forest | 0.897 ± 0.052 | 0.484 ± 0.053 | 0.981 ± 0.003 | 0.568 ± 0.054 |
| Alarm limits (ISO 10816 + RTD) | 0.701 ± 0.024 | 0.255 ± 0.054 | 0.735 ± 0.020 | 0.287 ± 0.060 |

"Predictable" leaves out the 7 days before *sudden* failures, which have no precursor for any method to rank; the gap between the two columns grows with the number of sudden failures in a fleet.

## Pooled over all fleets

Pooling weights every failure equally (a fleet with more failures counts more), which is the fairest single number when per-fleet counts are small.

| Method | Failures caught | Predictable caught | Detection (predictable) | False alarms | False alarms / pump-year | Worst fleet: detection (predictable) |
|---|---:|---:|---:|---:|---:|---:|
| Sentinel hybrid (GBM + MSPC) | 124/150 | 124/124 | 100.0 % | 2 | 0.03 | 100.0 % |
| Gradient boosting only | 120/150 | 120/124 | 96.8 % | 10 | 0.12 | 84.0 % |
| MSPC (PCA T² / SPE) | 124/150 | 124/124 | 100.0 % | 15 | 0.19 | 100.0 % |
| Isolation Forest | 117/150 | 117/124 | 94.4 % | 36 | 0.45 | 88.0 % |
| Alarm limits (ISO 10816 + RTD) | 43/150 | 43/124 | 34.7 % | 0 | 0.00 | 30.8 % |

## Remaining useful life and diagnosis (Sentinel models, test fleets)

| Metric | Mean ± sd | Range |
|---|---:|---:|
| RUL median-estimate MAE in the last 7 days before failure (h) | 78 ± 15 | 59 to 98 |
| RUL P10-P90 interval coverage, last 7 days (target 80 %) | 83.6 ± 6.2 % | 77.8 % to 93.3 % |
| Failure mode correctly diagnosed (predictable failures) | 90.3 ± 7.2 % | 80.0 % to 100.0 % |

## Per-fleet results

Each cell: failures caught / test failures, false alarms, median lead time.

| Seed | Test failures (predictable) | Sentinel hybrid (GBM + MSPC) | Gradient boosting only | MSPC (PCA T² / SPE) | Isolation Forest | Alarm limits (ISO 10816 + RTD) |
|---:|---:|---|---|---|---|---|
| 101 | 32 (24) | 24/32, 0 FA, 252 h | 24/32, 1 FA, 252 h | 24/32, 6 FA, 222 h | 22/32, 0 FA, 146 h | 8/32, 0 FA, 55 h |
| 102 | 31 (23) | 23/31, 1 FA, 356 h | 23/31, 0 FA, 277 h | 23/31, 2 FA, 253 h | 22/31, 6 FA, 206 h | 8/31, 0 FA, 63 h |
| 103 | 28 (26) | 26/28, 0 FA, 254 h | 26/28, 9 FA, 244 h | 26/28, 3 FA, 250 h | 25/28, 1 FA, 147 h | 8/28, 0 FA, 46 h |
| 104 | 29 (26) | 26/29, 1 FA, 300 h | 26/29, 0 FA, 232 h | 26/29, 3 FA, 224 h | 26/29, 28 FA, 302 h | 11/29, 0 FA, 84 h |
| 105 | 30 (25) | 25/30, 0 FA, 246 h | 21/30, 0 FA, 190 h | 25/30, 1 FA, 233 h | 22/30, 1 FA, 175 h | 8/30, 0 FA, 74 h |

| Seed | Hybrid ROC-AUC | Hybrid PR-AUC | Hybrid cost saving | RUL MAE, last 7 d (h) | RUL P10-P90 coverage | Diagnosis accuracy | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.879 | 0.589 | 59.3 % | 87 | 78.8 % | 91.7 % | 82 |
| 102 | 0.834 | 0.570 | 58.6 % | 59 | 85.2 % | 91.3 % | 81 |
| 103 | 0.961 | 0.667 | 73.4 % | 78 | 93.3 % | 100.0 % | 72 |
| 104 | 0.962 | 0.699 | 70.8 % | 69 | 83.1 % | 88.5 % | 72 |
| 105 | 0.909 | 0.655 | 65.8 % | 98 | 77.8 % | 80.0 % | 73 |

## Reading the result

* Across 5 fleets the Sentinel hybrid caught 124 of 124 predictable failures (100.0 %) with 2 false alarms in 80 pump-years (0.03 per pump-year). Conventional alarm limits caught 43 of 124 (34.7 %) with a median lead time of 64 ± 15 h, versus 281 ± 47 h for the hybrid.
* Methods whose ± ranges overlap should not be ranked against each other on these numbers alone; differences larger than the spread are robust to the luck of a single fleet.
* Everything is simulated: the absolute numbers depend on the simulator's assumptions (P-F interval lengths, sensor noise, share of sudden failures). The comparison between methods on identical data is the meaningful part.
