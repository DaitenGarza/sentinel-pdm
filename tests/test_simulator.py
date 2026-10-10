"""Simulator: reproducibility, schema (docs/ARCHITECTURE.md) and physical plausibility.

The models are only as credible as the data they learn from, so these tests pin down the
properties the rest of the project relies on: the same seed gives the same plant, the
maintenance log agrees with the sensor history, healthy pumps vibrate like healthy pumps, and a
developing bearing defect announces itself in the high-frequency band before it fails.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm.config import ALL_MODES, FAILURE_MODES, ISO_ZONE_LIMITS, SENSORS, SimConfig
from sentinel_pdm.simulator import simulate_fleet

READING_COLUMNS = ["asset_id", "hour", "timestamp", *SENSORS, "running", "load_pct", "speed_rpm", "ambient_temp",
                   "gt_mode", "gt_severity", "gt_state"]
EVENT_COLUMNS = ["event_id", "asset_id", "event_type", "hour", "end_hour", "mode", "onset_hour", "p_hour",
                 "predictable", "downtime_h", "timestamp"]
ASSET_COLUMNS = ["asset_id", "unit", "service", "criticality", "rated_flow_m3h", "rated_head_bar", "rated_current_a",
                 "motor_kw", "vfd", "install_year", "split"]


@pytest.fixture(scope="module")
def year_fleet():
    """A larger fleet (24 pumps x 1 year) with plenty of bearing failures and PM finds."""
    return simulate_fleet(SimConfig(n_assets=24, hours=8760, seed=5))


# --------------------------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------------------------
def test_same_seed_gives_identical_fleet():
    cfg = SimConfig(n_assets=6, hours=1500, seed=3)
    a, b = simulate_fleet(cfg), simulate_fleet(SimConfig(n_assets=6, hours=1500, seed=3))
    pd.testing.assert_frame_equal(a.readings, b.readings)
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.assets, b.assets)


def test_different_seed_gives_different_fleet():
    a = simulate_fleet(SimConfig(n_assets=6, hours=1500, seed=3)).readings
    b = simulate_fleet(SimConfig(n_assets=6, hours=1500, seed=4)).readings
    assert not np.allclose(a["vib_rms"].to_numpy(), b["vib_rms"].to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------------
def test_readings_schema(fleet, sim_config):
    r = fleet.readings
    assert list(r.columns) == READING_COLUMNS
    assert len(r) == sim_config.n_assets * sim_config.hours
    assert pd.api.types.is_string_dtype(r["asset_id"])
    assert pd.api.types.is_integer_dtype(r["hour"])
    assert pd.api.types.is_datetime64_any_dtype(r["timestamp"])
    for col in [*SENSORS, "load_pct", "speed_rpm", "ambient_temp", "gt_severity"]:
        assert r[col].dtype == np.float32, col
    assert r["running"].dtype == np.int8
    assert set(r["running"].unique()) <= {0, 1}
    assert set(r["gt_mode"].unique()) <= {"none", *ALL_MODES}
    assert set(r["gt_state"].unique()) <= {"running", "offline", "repair", "pm"}
    assert r["load_pct"].between(0, 105).all()


def test_every_pump_has_a_complete_hourly_record(fleet, sim_config):
    r = fleet.readings
    assert r["asset_id"].nunique() == sim_config.n_assets
    for _, g in r.groupby("asset_id", sort=False):
        assert np.array_equal(g["hour"].to_numpy(), np.arange(sim_config.hours))
        assert (g["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
        assert g["timestamp"].iloc[0] == pd.Timestamp(sim_config.start)
    # sorted by asset, hour: the order every downstream stage assumes
    assert r.equals(r.sort_values(["asset_id", "hour"], kind="stable").reset_index(drop=True))


def test_events_and_assets_schema(fleet, sim_config):
    ev, assets = fleet.events, fleet.assets
    assert list(ev.columns) == EVENT_COLUMNS
    assert list(assets.columns) == ASSET_COLUMNS
    assert ev["event_id"].is_unique and ev["event_id"].str.match(r"^WO-\d{5}$").all()
    assert set(ev["event_type"]) <= {"failure", "pm"}
    assert ev["predictable"].dtype == bool
    assert assets["asset_id"].is_unique
    assert all(re.fullmatch(r"P-\d{3}[AB](-\d+)?", a) for a in assets["asset_id"])
    assert set(assets["criticality"]) <= {"A", "B", "C"}
    assert set(assets["split"]) == {"train", "test"}
    assert (assets["split"] == "test").sum() == round(sim_config.n_assets * sim_config.test_fraction)
    assert set(ev["asset_id"]) <= set(assets["asset_id"])


# --------------------------------------------------------------------------------------------
# maintenance log consistent with the sensor history
# --------------------------------------------------------------------------------------------
def test_event_hours_are_in_range_and_ordered(year_fleet):
    ev, hours = year_fleet.events, 8760
    assert ev["hour"].between(0, hours - 1).all()
    assert (ev["end_hour"] > ev["hour"]).all() and (ev["end_hour"] <= hours).all()
    assert (ev["downtime_h"] == ev["end_hour"] - ev["hour"]).all()
    for _, g in ev.groupby("asset_id"):
        # work orders on one pump never overlap: the next starts after the previous one closed
        assert (g["hour"].to_numpy()[1:] >= g["end_hour"].to_numpy()[:-1]).all()


def test_downtime_rows_are_not_running(year_fleet):
    r = year_fleet.readings.set_index(["asset_id", "hour"])
    for e in year_fleet.events.itertuples():
        window = r.loc[(e.asset_id, slice(e.hour, e.end_hour - 1)), :]
        assert (window["running"] == 0).all(), e.event_id
        expected_state = "repair" if e.event_type == "failure" else "pm"
        assert (window["gt_state"] == expected_state).all(), e.event_id


def test_failure_p_points_follow_the_pf_interval(year_fleet):
    ev = year_fleet.events
    fails = ev[ev["event_type"] == "failure"]
    assert len(fails) > 0 and set(fails["mode"]) <= set(ALL_MODES)
    assert (fails["onset_hour"] <= fails["p_hour"]).all()
    assert (fails["p_hour"] <= fails["hour"]).all()
    sudden = fails[fails["mode"] == "sudden"]
    assert len(sudden) > 0, "fixture fleet should contain sudden failures"
    assert (sudden["p_hour"] == sudden["hour"]).all()
    assert (~sudden["predictable"]).all()
    assert fails.loc[fails["mode"] != "sudden", "predictable"].all()


def test_pm_events_record_what_they_found(year_fleet):
    pm = year_fleet.events[year_fleet.events["event_type"] == "pm"]
    found, nothing = pm[pm["mode"] != "none"], pm[pm["mode"] == "none"]
    assert len(found) > 0 and len(nothing) > 0
    assert set(found["mode"]) <= set(FAILURE_MODES)
    assert found["p_hour"].notna().all()
    assert ((found["onset_hour"] <= found["p_hour"]) & (found["p_hour"] <= found["hour"])).all()
    assert nothing["p_hour"].isna().all()
    assert (pm["predictable"] == (pm["mode"] != "none")).all()


def test_ground_truth_matches_the_log(year_fleet):
    """The degrading mode recorded hour by hour is the mode of the failure it leads to."""
    r = year_fleet.readings.set_index(["asset_id", "hour"])
    fails = year_fleet.events[(year_fleet.events["event_type"] == "failure") & year_fleet.events["predictable"]]
    for e in fails.itertuples():
        before = r.loc[(e.asset_id, slice(e.onset_hour, e.hour - 1)), :]
        assert (before["gt_mode"] == e.mode).all(), e.event_id
        assert before["gt_severity"].iloc[-1] > before["gt_severity"].iloc[0]


# --------------------------------------------------------------------------------------------
# physical plausibility
# --------------------------------------------------------------------------------------------
def test_healthy_vibration_is_in_iso_zone_a_or_b(year_fleet):
    r = year_fleet.readings
    healthy = r[(r["gt_mode"] == "none") & (r["running"] == 1)]["vib_rms"].dropna()
    # ISO 10816-3 group 2: zone A/B (new or unrestricted operation) is below the B/C boundary.
    share_ab = (healthy <= ISO_ZONE_LIMITS["B/C"]).mean()
    assert share_ab >= 0.99, f"only {share_ab:.1%} of healthy hours in zone A/B"
    assert healthy.median() <= ISO_ZONE_LIMITS["A/B"]


def test_stopped_pump_reads_idle_values(fleet):
    off = fleet.readings[fleet.readings["running"] == 0]
    assert len(off) > 0
    assert (off[["load_pct", "speed_rpm"]].fillna(0) == 0).all().all()
    assert (off["vib_rms"].dropna() < 0.5).mean() > 0.95


def test_bearing_defect_raises_high_frequency_band_before_failure(year_fleet):
    r = year_fleet.readings
    ev = year_fleet.events
    bearing = ev[(ev["event_type"] == "failure") & (ev["mode"] == "bearing")]
    assert len(bearing) >= 3, "fixture fleet should contain several bearing failures"
    for e in bearing.itertuples():
        pump = r[(r["asset_id"] == e.asset_id) & (r["running"] == 1)].set_index("hour")
        healthy = pump.loc[max(e.onset_hour - 168, 0): e.onset_hour - 1]
        baseline = healthy.loc[healthy["gt_mode"] == "none", "vib_hf"].median()
        pump = pump["vib_hf"]
        mid = pump.loc[int(e.p_hour): int(e.p_hour) + 24].median()
        last_two_days = pump.loc[e.hour - 48: e.hour - 1].median()
        assert last_two_days > 2.0 * baseline, f"{e.event_id}: vib_hf {baseline:.2f} -> {last_two_days:.2f} g"
        assert last_two_days > mid, f"{e.event_id}: vib_hf should keep rising toward failure"
