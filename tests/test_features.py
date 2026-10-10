"""Feature engineering: causality, maintenance awareness, units and scale invariance.

The causality tests are the most important ones in the suite. A feature that peeks even one
hour into the future makes a back-test look better than the model can ever be in the plant,
and nothing downstream would notice. Here the future is altered, cut off or given an extra
maintenance work order, and every feature up to hour t must stay identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm.config import SENSORS
from sentinel_pdm.features import (
    CONTEXT_FEATURES, DERIVED, FEATURE_COLUMNS, SCALE_FREE_SIGNALS, SIGNALS, SUFFIXES, build_features,
    maintenance_cycles, signal_of,
)

KEY_COLUMNS = ["asset_id", "hour", "running", "cycle"]
EMPTY_LOG = pd.DataFrame(columns=["asset_id", "event_type", "hour", "end_hour", "mode"])


def _work_order(asset_id: str, hour: int, end_hour: int, event_type: str = "pm") -> dict:
    return {"asset_id": asset_id, "event_type": event_type, "hour": hour, "end_hour": end_hour, "mode": "none"}


# --------------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------------
def test_feature_catalogue():
    assert len(SIGNALS) == len(SENSORS) + len(DERIVED) + 1 == 17
    assert len(FEATURE_COLUMNS) == len(SIGNALS) * len(SUFFIXES) + len(CONTEXT_FEATURES) == 107
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert set(SCALE_FREE_SIGNALS) <= set(SIGNALS)


@pytest.mark.parametrize("feature, signal", [("vib_rms_sl168", "vib_rms"), ("hyd_eff_dev", "hyd_eff"),
                                             ("bearing_rise_m24", "bearing_rise"), ("load_pct_sd24", "load_pct"),
                                             ("hours_since_maint", "hours_since_maint")])
def test_signal_of(feature, signal):
    assert signal_of(feature) == signal


def test_output_columns_dtypes_and_alignment(cleaned, features):
    clean = cleaned[0]
    assert list(features.columns) == KEY_COLUMNS + FEATURE_COLUMNS
    assert features.index.equals(clean.index)
    pd.testing.assert_series_equal(features["asset_id"], clean["asset_id"], check_names=False)
    assert all(features[c].dtype == np.float32 for c in FEATURE_COLUMNS)


def test_features_are_nan_while_not_running(features):
    stopped = features["running"] == 0
    assert stopped.any()
    assert features.loc[stopped, FEATURE_COLUMNS].isna().all().all()
    # ...and mostly defined while running (windows need a few hours of data after each restart)
    running = features.loc[~stopped]
    assert running["vib_rms_m24"].notna().mean() > 0.95


def test_works_without_a_maintenance_log(make_pump):
    feats = build_features(make_pump(hours=200), EMPTY_LOG)
    assert (feats["cycle"] == 0).all()
    assert feats["hours_since_maint"].isna().all() and feats["run_hours_since_maint"].isna().all()
    assert feats["vib_rms_m24"].iloc[24:].notna().all()


# --------------------------------------------------------------------------------------------
# causality
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t", [900, 2345, 3999])
def test_features_are_causal_when_future_data_changes(fleet, cleaned, features, t):
    clean = cleaned[0].copy()
    future = clean["hour"] > t
    rng = np.random.default_rng(t)
    for col in [*SENSORS, "load_pct", "ambient_temp"]:
        factor = rng.uniform(0.3, 3.0, future.sum()).astype(np.float32)
        clean.loc[future, col] = clean.loc[future, col] * factor
    clean.loc[future & (rng.random(len(clean)) < 0.1), "bearing_temp"] = np.nan
    clean.loc[future & (rng.random(len(clean)) < 0.02), "running"] = np.int8(0)
    altered = build_features(clean, fleet.events)
    past = features["hour"] <= t
    pd.testing.assert_frame_equal(altered.loc[past], features.loc[past], check_exact=True)
    assert not np.allclose(altered.loc[~past, "vib_rms_m24"], features.loc[~past, "vib_rms_m24"], equal_nan=True)


def test_features_are_causal_when_the_future_is_cut_off(fleet, cleaned, features):
    t = 2600
    clean = cleaned[0]
    keep = clean["hour"] <= t
    events = fleet.events[fleet.events["hour"] <= t]       # work orders raised by time t
    truncated = build_features(clean.loc[keep], events)
    pd.testing.assert_frame_equal(truncated, features.loc[keep], check_exact=True)


def test_future_maintenance_does_not_change_the_past(fleet, cleaned, features):
    t = 3000
    extra = pd.DataFrame([_work_order(a, t + 1, t + 12) for a in fleet.assets["asset_id"]])
    events = pd.concat([fleet.events, extra], ignore_index=True).sort_values(["asset_id", "hour"])
    with_extra = build_features(cleaned[0], events)
    past = features["hour"] <= t
    pd.testing.assert_frame_equal(with_extra.loc[past], features.loc[past], check_exact=True)
    assert (with_extra.loc[~past, "cycle"] > features.loc[~past, "cycle"]).all()


# --------------------------------------------------------------------------------------------
# maintenance awareness
# --------------------------------------------------------------------------------------------
@pytest.fixture
def repaired_pump(make_pump):
    """A pump vibrating at 5 mm/s that is repaired (hours 300-309) and comes back at 1 mm/s."""
    running = np.ones(600)
    running[300:310] = 0
    df = make_pump(hours=600, noise=0.0, running=running)
    df["vib_rms"] = np.where(df["hour"] < 300, 5.0, 1.0).astype(np.float32)
    log = pd.DataFrame([_work_order("P-101A", 300, 310, event_type="failure")])
    return df, build_features(df, log).set_index("hour")


def test_rolling_windows_restart_at_maintenance(repaired_pump):
    _, f = repaired_pump
    # 6 running hours after the repair: the 24 h window is defined and holds only new-cycle data,
    # the 7-day window (needs 24 h) is not defined yet instead of mixing in pre-repair vibration
    assert f.loc[315, "vib_rms_m24"] == pytest.approx(1.0)
    assert np.isnan(f.loc[315, "vib_rms_m168"])
    assert f.loc[340, "vib_rms_m168"] == pytest.approx(1.0)
    assert f.loc[340, "vib_rms_sd24"] == pytest.approx(0.0)
    # the healthy baseline is re-learned after the repair, so the deviation resets to zero
    assert f.loc[299, "vib_rms_dev"] == pytest.approx(0.0)
    assert f.loc[340, "vib_rms_dev"] == pytest.approx(0.0)


def test_cycle_and_time_since_maintenance(repaired_pump):
    _, f = repaired_pump
    assert (f.loc[:299, "cycle"] == 0).all() and (f.loc[300:, "cycle"] == 1).all()
    assert f.loc[:299, "hours_since_maint"].isna().all()       # install date unknown
    assert f.loc[330, "hours_since_maint"] == 20                 # back in service at hour 310
    assert f.loc[330, "run_hours_since_maint"] == 21             # running hours 310..330


def test_maintenance_cycles_per_asset():
    keys = pd.DataFrame({"asset_id": ["A"] * 6 + ["B"] * 3, "hour": [0, 10, 20, 30, 40, 50, 0, 10, 20]})
    log = pd.DataFrame([_work_order("A", 10, 15), _work_order("A", 40, 45, "failure")])
    out = maintenance_cycles(keys, log)
    assert out["cycle"].tolist() == [0, 1, 1, 1, 2, 2, 0, 0, 0]
    expected = [np.nan, np.nan, 5, 15, np.nan, 5, np.nan, np.nan, np.nan]   # NaN while the WO is open
    np.testing.assert_array_equal(out["hours_since_maint"].to_numpy(), expected)


# --------------------------------------------------------------------------------------------
# units and scale
# --------------------------------------------------------------------------------------------
def test_slope_sign_and_units_on_a_ramp(make_pump):
    df = make_pump(hours=400, noise=0.0)
    h = df["hour"].to_numpy(dtype=np.float64)
    df["bearing_temp"] = (55.0 + 0.5 * h).astype(np.float32)     # +0.5 degC/h = +12 degC/day
    df["flow"] = (300.0 - 0.3 * h).astype(np.float32)            # losing 7.2 m3/h per day
    f = build_features(df, EMPTY_LOG).set_index("hour")
    np.testing.assert_allclose(f.loc[10:, "bearing_temp_sl24"], 12.0, rtol=1e-4)
    np.testing.assert_allclose(f.loc[30:, "bearing_temp_sl168"], 12.0, rtol=1e-4)
    np.testing.assert_allclose(f.loc[10:, "bearing_rise_sl24"], 12.0, rtol=1e-4)
    # flow is scale-free: the slope is a fraction of the pump's own baseline flow per day
    baseline = np.median(300.0 - 0.3 * np.arange(168))
    assert (f.loc[10:, "flow_sl24"] < 0).all()
    np.testing.assert_allclose(f.loc[200:, "flow_sl24"], -7.2 / baseline, rtol=1e-4)
    # a flat signal has zero slope
    np.testing.assert_allclose(f.loc[10:, "vib_rms_sl24"], 0.0, atol=1e-6)


def test_scale_free_signals_ignore_pump_size(cleaned, fleet):
    """A pump twice the size (2x current, pressure and flow) has the same scale-free features."""
    clean = cleaned[0]
    pump_id = clean["asset_id"].iloc[0]
    one = clean[clean["asset_id"] == pump_id]
    big = one.copy()
    big["asset_id"] = "P-999X"
    for col in ("motor_current", "suction_pressure", "discharge_pressure", "flow"):
        big[col] = big[col] * np.float32(2.0)
    events = fleet.events[fleet.events["asset_id"] == pump_id]
    both = pd.concat([one, big], ignore_index=True)
    feats = build_features(both, pd.concat([events, events.assign(asset_id="P-999X")], ignore_index=True))
    a = feats[feats["asset_id"] == pump_id].reset_index(drop=True)
    b = feats[feats["asset_id"] == "P-999X"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS], rtol=1e-6)
    assert np.nanmean(b["motor_current_m24"]) == pytest.approx(np.nanmean(a["motor_current_m24"]))


def test_absolute_signals_keep_their_units(make_pump):
    """Vibration stays in mm/s: ISO 10816 severity limits are absolute, not relative to the pump."""
    small = make_pump(hours=100)
    big = small.assign(asset_id="P-999X", vib_rms=small["vib_rms"] * np.float32(2.0))
    f = build_features(pd.concat([small, big], ignore_index=True), EMPTY_LOG)
    a, b = (f.loc[f["asset_id"] == k, "vib_rms_m24"].to_numpy() for k in ("P-101A", "P-999X"))
    np.testing.assert_allclose(b[30:], 2.0 * a[30:], rtol=1e-5)
