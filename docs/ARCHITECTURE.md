# Architecture

Sentinel PdM is a Python package (`src/sentinel_pdm`) plus a Streamlit dashboard (`app/`).
One command (`python -m sentinel_pdm train`) runs the whole pipeline and writes every artifact
the dashboard reads. Nothing needs a GPU, a cloud account or an API key.

```
                ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
 simulator.py ─►│ quality.py   │──►│ features.py  │──►│ models.py    │──►│ evaluation.py        │
 (48 pumps x    │ causal       │   │ 107 causal   │   │ GBM risk/RUL │   │ alert raises, P-F    │
  8760 h, gt_*) │ cleaning     │   │ features     │   │ /diagnosis + │   │ matching, economics, │
                └──────────────┘   └──────────────┘   │ MSPC, IForest│   │ cost-optimal         │
 labels.py ─── targets from the maintenance log ────► │ + hybrid     │   │ thresholds           │
                                                      └──────┬───────┘   └──────────┬───────────┘
                                                             ▼                      ▼
                                               scoring.py (bundle)        pipeline.py writes
                                               explain.py (drivers,       data/ + artifacts/
                                               recommendations)                 │
                                                                                ▼
                                                                   app/ (Streamlit dashboard)
```

## Package modules (`src/sentinel_pdm`)

| Module | Responsibility | Key API |
|---|---|---|
| `config.py` | Paths, sensor list, all tunable settings as dataclasses | `Config`, `SimConfig`, `LabelConfig`, `AlertConfig`, `CostConfig`, `TrainConfig`, `SENSORS`, `FAILURE_MODES`, `PROJECT_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`, `REPORTS_DIR` |
| `simulator.py` | Physics-informed fleet simulator | `simulate_fleet(SimConfig) -> FleetData(readings, events, assets)` |
| `quality.py` | Causal cleaning (stopped pump, impossible values, stuck tags, spikes, short gaps) | `clean_readings(df) -> (clean, quality_summary)` |
| `features.py` | Causal, maintenance-aware rolling features | `build_features(clean, events) -> DataFrame`, `FEATURE_COLUMNS`, `SIGNALS`, `signal_of(feature)` |
| `labels.py` | Targets: failure within 7 days, time-to-failure, RUL, next mode, censoring | `make_labels(keys, events, record_end, LabelConfig)` |
| `models.py` | GBM risk / RUL (conformal) / diagnosis, MSPC, Isolation Forest, hybrid, CV-ensemble training | `train_models(...) -> TrainedModels` |
| `evaluation.py` | Row metrics, alert raises, P-F event matching, policy cost, threshold choice | `alert_episodes`, `match_events`, `event_metrics`, `policy_cost`, `choose_threshold` |
| `explain.py` | Signal names/units, robust z-score drivers, status, maintenance recommendations | `SIGNAL_INFO`, `MODE_LABELS`, `top_drivers`, `recommendation`, `STATUS_COLORS` |
| `scoring.py` | Model bundle save/load; score any data | `load_bundle()`, `score_readings(raw, events, bundle) -> (clean, preds)` |
| `pipeline.py` | Orchestrates everything, writes artifacts | `run_pipeline(Config) -> metrics dict` |
| `signals.py` | Synthetic vibration waveform + FFT / envelope spectrum for the asset page | see module docstring |
| `ai4i.py` | Real-data benchmark on UCI AI4I 2020 | `run_ai4i_benchmark(download=True) -> dict` |
| `robustness.py` | Repeat the experiment on independent fleets (seeds) | `run_robustness(seeds, quick=False) -> dict` |
| `report.py` | Static results report | `build_report() -> Path` |
| `__main__.py` | CLI: `train`, `summary`, `ai4i`, `robustness`, `report`, `app` | `python -m sentinel_pdm --help` |

## Data and artifacts

All generated files are git-ignored (`data/`, `artifacts/`) except `reports/`.

### `data/readings.parquet` (raw, 420,480 rows = 48 pumps x 8,760 h)
`asset_id` (str, e.g. `P-101A`), `hour` (int, 0..8759), `timestamp` (datetime64, hourly from 2025-01-01),
sensors `vib_rms` mm/s, `vib_1x` mm/s, `vib_2x` mm/s, `vib_hf` g, `vib_kurtosis`, `bearing_temp` °C,
`motor_temp` °C, `motor_current` A, `current_imbalance` %, `suction_pressure` bar, `discharge_pressure` bar,
`flow` m³/h (float32, NaN = missing), `running` (int8 1/0), `load_pct` (0..105), `speed_rpm`, `ambient_temp` °C,
ground truth (simulation only, never used as model input): `gt_mode` (none / bearing / misalignment /
cavitation / winding / sudden), `gt_severity` (0..~1.3), `gt_state` (running / offline / repair / pm).

### `data/clean.parquet`
Same keys + `running`, context and sensor columns after causal cleaning (no `gt_*`).

### `data/events.csv` (maintenance log / CMMS work orders)
`event_id` (WO-10000...), `asset_id`, `event_type` (`failure` | `pm`), `hour` (start of downtime = failure hour),
`end_hour` (back in service), `mode` (failure mode; for PM the degradation it found or `none`),
`onset_hour` (degradation start), `p_hour` (P point: defect detectable by inspection; = hour for sudden failures),
`predictable` (bool, False for `sudden`), `downtime_h`, `timestamp`.

### `data/assets.csv`
`asset_id`, `unit` (e.g. `U-300 Separation`), `service` (e.g. `Column reflux pump`), `criticality` (A/B/C),
`rated_flow_m3h`, `rated_head_bar`, `rated_current_a`, `motor_kw`, `vfd` (bool), `install_year`,
`split` (`train` = 32 pumps used for training, `test` = 16 pumps held out).

### `data/quality.csv`
`asset_id`, `running_pct`, `missing_pct`, `spikes_removed`, `stuck_hours`.

### `artifacts/predictions.parquet` (420,480 rows, one per pump-hour)
Keys: `asset_id`, `hour`, `timestamp`, `split`, `running`.
Scores (float32, except `risk` and `risk_gbm` which are float64 so the calibration tie-break survives; NaN while stopped): `risk` (Sentinel hybrid P(failure within 7 d); drives alerts/status),
`risk_gbm` (gradient boosting only), `rul_p10`, `rul_p50`, `rul_p90` (hours, conditional on a developing
failure, capped at 720), `prob_bearing`, `prob_misalignment`, `prob_cavitation`, `prob_winding`,
`mspc_score` (1.0 = 99 % control limit), `iforest_score`, `rule_score` (>= 1 means a conventional alarm limit
is exceeded), `z_<signal>` for the 17 signals in `features.SIGNALS` (robust z-score vs. healthy baseline).
Derived: `pred_mode`, `status` (Critical / Alert / Watch / Normal / Offline).
Labels / truth (for evaluation pages): `label` (1 failure within 7 d, 0 no, -1 censored), `ttf` (hours to
next failure), `rul_target`, `next_mode`, `next_predictable`, `gt_mode`, `gt_severity`, `gt_state`.
Training-fleet rows hold **out-of-fold** scores (the pump was not in the model that scored it).

### `artifacts/metrics.json`
```
{ "methods": {"hybrid": "...", "ml": "...", "mspc": "...", "iforest": "...", "rule": "..."},
  "score_columns": {"hybrid": "risk", "ml": "risk_gbm", "mspc": "mspc_score", ...},
  "thresholds": {method: threshold},
  "test": { "row": {method: {roc_auc, pr_auc, positive_rate, n_rows, [brier]}},
            "row_predictable": {...same, excluding sudden-failure windows...},
            "event": {method: {failures, predictable_failures, caught, caught_predictable, detection_rate,
                               detection_rate_predictable, late_only, median_lead_h, mean_lead_h, alerts,
                               true_alerts, false_alarms, pm_confirmed_alerts, false_alarms_per_asset_year,
                               alert_precision}},
            "economics": {"reactive"|method: {total_cost, unplanned_failures, planned_repairs, false_alarms,
                               downtime_h, unplanned_cost_each, planned_cost_each, savings_vs_reactive,
                               savings_pct, downtime_reduction_pct, cost_per_pump_year}},
            "rul": {"le_168h"|"le_720h": {n_rows, mae_h, coverage_p10_p90, median_interval_h}},
            "diagnosis": {row_accuracy, event_accuracy, n_events, confusion: {labels, matrix}},
            "asset_years": 16.0 },
  "cv": { "row": {...}, "event": {...} },          # out-of-fold on the 32 training pumps
  "data": {n_assets, hours, n_train_assets, n_test_assets, rows, failures, failures_by_mode, failures_test,
           failures_train, pm_events, n_features},
  "config": Config.to_dict(), "timings_s": {...}, "environment": {...}, "generated_at": "...", "version": "1.0.0" }
```

### Other artifacts
* `artifacts/alerts.csv`: `asset_id, start_hour, end_hour, outcome (tp|late|pm_confirmed|unresolved|false_alarm), method, split`
* `artifacts/failure_outcomes.csv`: events columns + `caught, lead_h, late_only, method, split`
* `artifacts/threshold_curve.csv`: `threshold, caught, failures, false_alarms, false_alarms_per_asset_year, total_cost, within_budget, method` (training fleet, out-of-fold)
* `artifacts/feature_importance.csv`: `feature, importance, std, signal` (permutation importance, PR-AUC drop)
* `artifacts/models/bundle.joblib`: `scoring.ModelBundle`
* `artifacts/ai4i_metrics.json`, `artifacts/ai4i_importance.csv`: real-data benchmark (written by `ai4i.py`)

## Dashboard (`app/`)

`app/Home.py` is the entry point (fleet overview). Pages live in `app/pages/` and are numbered for ordering.
Every page starts with:

```python
import common as C
C.page_setup("Page title", "🔍")
C.require_artifacts()
```

`app/common.py` provides cached loaders (`load_predictions(columns)`, `load_clean`, `load_readings`,
`load_events`, `load_assets`, `load_alerts`, `load_failure_outcomes`, `load_threshold_curve`,
`load_feature_importance`, `load_metrics`, `load_quality`, `load_model_bundle`), `fleet_snapshot(hour)`,
formatting (`fmt_hours`, `fmt_money`, `fmt_pct`, `status_badge`), `style_fig(fig)` and colour maps
(`METHOD_COLORS`, `MODE_COLORS`, `STATUS_COLORS`). Charts use Plotly with the dark template.

## Conventions

* Python >= 3.10, type hints, module docstrings that explain the *why*.
* Only these runtime dependencies: numpy, pandas, scipy, scikit-learn, joblib, pyarrow, plotly, streamlit
  (+ pytest for tests). pandas 3 is supported: arrays from `.to_numpy()` may be read-only (copy before
  writing) and string columns use the `str` dtype.
* Nothing is fitted on test pumps; anything tuned (thresholds, calibration, conformal offsets) is tuned on
  out-of-fold scores of the training fleet.
* Runs on Windows first (`setup.bat`, `run.bat`), macOS/Linux via `setup.sh` / `run.sh`.
