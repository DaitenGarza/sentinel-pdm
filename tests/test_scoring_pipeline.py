"""End-to-end pipeline run and the scoring path that serves its models.

One small pipeline run (``pipeline_run``: 12 pumps x 6 months, into a temporary folder) is
shared by every test here. The tests check the contract the dashboard and the reports rely on:
every documented artifact exists with its documented columns, metrics.json has the documented
structure, and - most important - scoring the held-out pumps' *raw readings* through
``scoring.score_readings`` (the path used by the dashboard upload page and the live replay)
reproduces the predictions the pipeline wrote. If training and serving ever drift apart (a
feature computed differently, a model swapped, a threshold not saved) that test fails.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm import __version__, scoring
from sentinel_pdm.config import CONTEXT, FAILURE_MODES, SENSORS
from sentinel_pdm.features import FEATURE_COLUMNS, SIGNALS
from sentinel_pdm.pipeline import METHODS, SCORE_COLUMNS

STATUSES = {"Critical", "Alert", "Watch", "Normal", "Offline"}
PROB_COLUMNS = [f"prob_{m}" for m in FAILURE_MODES]
MODEL_COLUMNS = list(scoring.SCORE_COLUMNS)                      # NaN exactly while the pump is stopped
SERVED_FLOAT_COLUMNS = [*MODEL_COLUMNS, "rule_score", *[f"z_{s}" for s in SIGNALS]]
EVENT_COLUMNS = ["event_id", "asset_id", "event_type", "hour", "end_hour", "mode", "onset_hour", "p_hour",
                 "predictable", "downtime_h", "timestamp"]

# Every file the pipeline documents (docs/ARCHITECTURE.md) with the columns it must contain.
DOCUMENTED_FILES: dict[str, list[str]] = {
    "data/readings.parquet": ["asset_id", "hour", "timestamp", *SENSORS, "running", *CONTEXT,
                              "gt_mode", "gt_severity", "gt_state"],
    "data/clean.parquet": ["asset_id", "hour", "timestamp", "running", *CONTEXT, *SENSORS],
    "data/events.csv": EVENT_COLUMNS,
    "data/assets.csv": ["asset_id", "unit", "service", "criticality", "rated_flow_m3h", "rated_head_bar",
                        "rated_current_a", "motor_kw", "vfd", "install_year", "split"],
    "data/quality.csv": ["asset_id", "running_pct", "missing_pct", "spikes_removed", "stuck_hours"],
    "artifacts/predictions.parquet": ["asset_id", "hour", "timestamp", "split", "running", *SERVED_FLOAT_COLUMNS,
                                      "pred_mode", "status", "label", "ttf", "rul_target", "next_mode",
                                      "next_predictable", "gt_mode", "gt_severity", "gt_state"],
    "artifacts/alerts.csv": ["asset_id", "start_hour", "end_hour", "outcome", "method", "split"],
    "artifacts/failure_outcomes.csv": [*EVENT_COLUMNS, "caught", "lead_h", "late_only", "method", "split"],
    "artifacts/threshold_curve.csv": ["threshold", "caught", "failures", "false_alarms",
                                      "false_alarms_per_asset_year", "total_cost", "within_budget", "method"],
    "artifacts/feature_importance.csv": ["feature", "importance", "std", "signal"],
}


def _read(path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


@pytest.fixture(scope="module")
def metrics_json(pipeline_run) -> dict:
    def reject(token):
        raise ValueError(f"metrics.json is not strict JSON: contains {token}")

    return json.loads((pipeline_run.artifacts_dir / "metrics.json").read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture(scope="module")
def predictions(pipeline_run) -> pd.DataFrame:
    return pd.read_parquet(pipeline_run.artifacts_dir / "predictions.parquet")


@pytest.fixture(scope="module")
def bundle(pipeline_run) -> scoring.ModelBundle:
    return scoring.load_bundle(pipeline_run.artifacts_dir / "models")


@pytest.fixture(scope="module")
def holdout(pipeline_run, bundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw readings and maintenance log of the held-out test pumps, as a plant would export them."""
    test_ids = bundle.meta["test_assets"]
    raw = pd.read_parquet(pipeline_run.data_dir / "readings.parquet")
    events = pd.read_csv(pipeline_run.data_dir / "events.csv")
    return raw[raw["asset_id"].isin(test_ids)], events[events["asset_id"].isin(test_ids)]


@pytest.fixture(scope="module")
def served(holdout, bundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean, predictions) from ``score_readings`` on the held-out pumps."""
    raw, events = holdout
    return scoring.score_readings(raw, events, bundle)


# --------------------------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("relpath", DOCUMENTED_FILES)
def test_documented_artifact_exists_with_its_columns(pipeline_run, relpath):
    root = pipeline_run.data_dir if relpath.startswith("data/") else pipeline_run.artifacts_dir
    path = root / relpath.split("/", 1)[1]
    assert path.is_file(), f"{relpath} was not written"
    missing = set(DOCUMENTED_FILES[relpath]) - set(_read(path).columns)
    assert not missing, f"{relpath} lacks documented columns {sorted(missing)}"


def test_model_bundle_is_written_and_loads(pipeline_run, bundle, metrics_json):
    assert (pipeline_run.artifacts_dir / "models" / scoring.BUNDLE_FILE).is_file()
    assert isinstance(bundle, scoring.ModelBundle)
    assert bundle.thresholds == pytest.approx(metrics_json["thresholds"])
    assert bundle.config == pipeline_run.cfg.to_dict()
    assert bundle.meta["n_features"] == len(FEATURE_COLUMNS)
    assets = pd.read_csv(pipeline_run.data_dir / "assets.csv")
    for split in ("train", "test"):
        assert bundle.meta[f"{split}_assets"] == sorted(assets.loc[assets["split"] == split, "asset_id"])
    assert set(SIGNALS) <= set(bundle.ref_stats.index)


def test_load_bundle_explains_how_to_train_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m sentinel_pdm train"):
        scoring.load_bundle(tmp_path)


def test_predictions_cover_every_pump_hour(pipeline_run, predictions):
    n_assets, hours = pipeline_run.cfg.sim.n_assets, pipeline_run.cfg.sim.hours
    assert len(predictions) == n_assets * hours
    assert predictions.groupby("asset_id")["hour"].apply(lambda h: h.tolist() == list(range(hours))).all()
    assert all(predictions[c].dtype == np.float32 for c in SERVED_FLOAT_COLUMNS)

    running = predictions["running"].to_numpy() == 1
    test = predictions["split"].to_numpy() == "test"
    assert np.isnan(predictions[MODEL_COLUMNS].to_numpy()[~running]).all()
    assert np.isfinite(predictions[MODEL_COLUMNS].to_numpy()[running & test]).all()
    # On training pumps prob_* may be NaN: a CV fold whose training pumps saw fewer than two failure
    # modes fits no diagnosis model, so its validation pumps get no out-of-fold diagnosis.
    always = [c for c in MODEL_COLUMNS if c not in PROB_COLUMNS]
    assert np.isfinite(predictions[always].to_numpy()[running]).all()
    assert set(predictions["status"]) <= STATUSES
    assert ((predictions["status"] == "Offline") == ~running).all()
    assert set(predictions["pred_mode"]) <= {*FAILURE_MODES, "none"}


def test_status_uses_the_saved_alert_threshold(predictions, metrics_json):
    thr = metrics_json["thresholds"]["hybrid"]
    run_rows = predictions[predictions["running"] == 1]
    risk = run_rows["risk"].to_numpy(dtype=np.float64)
    clear = np.abs(risk - thr) > 1e-6               # float32 storage can blur rows sitting on the threshold
    alerting = run_rows["status"].isin(["Alert", "Critical"]).to_numpy()
    np.testing.assert_array_equal(alerting[clear], risk[clear] >= thr)
    critical = (run_rows["status"] == "Critical").to_numpy()
    assert (run_rows["rul_p50"].to_numpy()[critical] <= 72).all()


# --------------------------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------------------------
def test_metrics_json_has_the_documented_structure(pipeline_run, metrics_json):
    m = metrics_json
    assert {"methods", "score_columns", "thresholds", "test", "cv", "data", "config", "timings_s",
            "environment", "generated_at", "version"} <= set(m)
    assert m["methods"] == METHODS and set(m["methods"]) == {"hybrid", "ml", "mspc", "iforest", "rule"}
    assert m["score_columns"] == SCORE_COLUMNS and set(m["thresholds"]) == set(METHODS)
    assert m["version"] == __version__ and m["config"] == pipeline_run.cfg.to_dict()

    test = m["test"]
    assert {"row", "row_predictable", "event", "economics", "rul", "diagnosis", "asset_years"} <= set(test)
    row_keys = {"roc_auc", "pr_auc", "positive_rate", "n_rows"}
    event_keys = {"failures", "predictable_failures", "caught", "caught_predictable", "detection_rate",
                  "detection_rate_predictable", "late_only", "median_lead_h", "mean_lead_h", "alerts",
                  "true_alerts", "false_alarms", "pm_confirmed_alerts", "false_alarms_per_asset_year",
                  "alert_precision"}
    econ_keys = {"total_cost", "unplanned_failures", "planned_repairs", "false_alarms", "downtime_h",
                 "unplanned_cost_each", "planned_cost_each", "savings_vs_reactive", "savings_pct",
                 "downtime_reduction_pct", "cost_per_pump_year"}
    for section in (test["row"], test["row_predictable"], m["cv"]["row"]):
        assert set(section) == set(METHODS)
        for key, row in section.items():
            assert row_keys | ({"brier"} if key in ("hybrid", "ml") else set()) <= set(row)
    for section in (test["event"], m["cv"]["event"]):
        assert set(section) == set(METHODS) and all(event_keys <= set(v) for v in section.values())
    assert set(test["economics"]) == {"reactive", *METHODS}
    assert all(econ_keys <= set(v) for v in test["economics"].values())
    for band in test["rul"].values():
        assert {"n_rows", "mae_h", "coverage_p10_p90", "median_interval_h"} <= set(band)
    assert set(test["rul"]) <= {"le_168h", "le_720h"}
    assert "row_accuracy" in test["diagnosis"]
    assert test["asset_years"] == pytest.approx(m["data"]["n_test_assets"] * pipeline_run.cfg.sim.hours / 8760)


def test_metrics_data_section_matches_the_written_data(pipeline_run, metrics_json):
    d = metrics_json["data"]
    events = pd.read_csv(pipeline_run.data_dir / "events.csv")
    failures = events[events["event_type"] == "failure"]
    assert d["n_assets"] == pipeline_run.cfg.sim.n_assets and d["hours"] == pipeline_run.cfg.sim.hours
    assert d["rows"] == pipeline_run.cfg.sim.n_assets * pipeline_run.cfg.sim.hours
    assert d["n_train_assets"] + d["n_test_assets"] == d["n_assets"]
    assert d["failures"] == len(failures) == d["failures_train"] + d["failures_test"]
    assert sum(d["failures_by_mode"].values()) == d["failures"]
    assert d["pm_events"] == int((events["event_type"] == "pm").sum())
    assert d["n_features"] == len(FEATURE_COLUMNS)


def test_economics_are_relative_to_run_to_failure(metrics_json):
    econ, ev = metrics_json["test"]["economics"], metrics_json["test"]["event"]
    reactive = econ["reactive"]
    assert reactive["unplanned_failures"] == metrics_json["data"]["failures_test"]
    assert reactive["savings_vs_reactive"] == 0
    for key in METHODS:
        assert econ[key]["savings_vs_reactive"] == pytest.approx(reactive["total_cost"] - econ[key]["total_cost"])
        assert econ[key]["planned_repairs"] == ev[key]["caught"]
        assert econ[key]["false_alarms"] == ev[key]["false_alarms"]


def test_thresholds_are_sane(pipeline_run, metrics_json, predictions):
    thr = metrics_json["thresholds"]
    assert thr["rule"] == 1.0                                    # the plant's own alarm limits, not tuned
    assert 0.0 < thr["hybrid"] < 1.0 and 0.0 < thr["ml"] < 1.0   # probabilities
    assert thr["mspc"] > 0.0 and thr["iforest"] > 0.0            # anomaly scores (1.0 = 99 % of healthy)

    train = predictions[(predictions["split"] == "train") & (predictions["running"] == 1)]
    curve = pd.read_csv(pipeline_run.artifacts_dir / "threshold_curve.csv")
    for key in ("hybrid", "ml", "mspc", "iforest"):
        s = train[SCORE_COLUMNS[key]].to_numpy(dtype=np.float64)
        assert np.nanmin(s) <= thr[key] <= np.nanmax(s) * (1 + 1e-6), key   # tuned on training-fleet OOF scores
        c = curve[curve["method"] == key]
        chosen = c[np.isclose(c["threshold"], thr[key], rtol=1e-9)]
        assert len(chosen) == 1, key
        if c["within_budget"].any():
            assert bool(chosen["within_budget"].iloc[0]), f"{key} threshold breaks the false-alarm budget"


# --------------------------------------------------------------------------------------------
# train/serve consistency
# --------------------------------------------------------------------------------------------
def test_score_readings_reproduces_the_pipeline_predictions(served, predictions, bundle):
    """Scoring the held-out pumps' raw readings gives exactly what the pipeline wrote for them."""
    clean, preds = served
    expected = (predictions[predictions["asset_id"].isin(bundle.meta["test_assets"])]
                .sort_values(["asset_id", "hour"]).reset_index(drop=True))
    assert len(preds) == len(clean) == len(expected)
    np.testing.assert_array_equal(preds["asset_id"].to_numpy(), expected["asset_id"].to_numpy())
    np.testing.assert_array_equal(preds["hour"].to_numpy(), expected["hour"].to_numpy())

    for col in ("risk", "rul_p50", "mspc_score", *SERVED_FLOAT_COLUMNS):
        np.testing.assert_allclose(preds[col].to_numpy(dtype=np.float64), expected[col].to_numpy(dtype=np.float64),
                                   rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=col)   # float32 storage
    np.testing.assert_array_equal(preds["status"].to_numpy(), expected["status"].to_numpy())
    np.testing.assert_array_equal(preds["pred_mode"].to_numpy(), expected["pred_mode"].to_numpy())


def test_score_readings_cleans_like_the_pipeline(pipeline_run, served, bundle):
    clean, _ = served
    stored = pd.read_parquet(pipeline_run.data_dir / "clean.parquet")
    stored = (stored[stored["asset_id"].isin(bundle.meta["test_assets"])]
              .sort_values(["asset_id", "hour"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(clean[stored.columns], stored, check_dtype=False)


def test_score_readings_does_not_depend_on_row_order(holdout, served, bundle):
    raw, events = holdout
    shuffled = raw.sample(frac=1.0, random_state=0)
    _, preds = scoring.score_readings(shuffled, events.iloc[::-1], bundle)
    pd.testing.assert_frame_equal(preds, served[1])


@pytest.fixture(scope="module")
def one_pump(holdout) -> pd.DataFrame:
    raw, _ = holdout
    return raw[raw["asset_id"] == raw["asset_id"].iloc[0]]


@pytest.mark.parametrize("log", [
    None,
    pd.DataFrame(),
    pd.DataFrame(columns=["asset_id", "event_type", "hour", "end_hour", "mode"]),
], ids=["none", "empty-frame", "empty-with-columns"])
def test_score_readings_without_a_maintenance_log(one_pump, bundle, log):
    clean, preds = scoring.score_readings(one_pump, log, bundle)
    assert len(clean) == len(preds) == len(one_pump)
    np.testing.assert_array_equal(preds["hour"].to_numpy(), np.sort(one_pump["hour"].to_numpy()))
    running = preds["hour"].map(one_pump.set_index("hour")["running"]).to_numpy() == 1
    scores = preds[MODEL_COLUMNS].to_numpy(dtype=np.float64)
    assert np.isfinite(scores[running]).all() and np.isnan(scores[~running]).all()
    assert ((preds["risk"][running] >= 0) & (preds["risk"][running] <= 1)).all()
    assert set(preds["status"]) <= STATUSES
    assert ((preds["status"] == "Offline").to_numpy() == ~running).all()


def test_score_readings_without_timestamps(one_pump, bundle):
    _, preds = scoring.score_readings(one_pump.drop(columns="timestamp"), None, bundle)
    assert "timestamp" not in preds and list(preds.columns[:2]) == ["asset_id", "hour"]
    assert preds["risk"].notna().any()
