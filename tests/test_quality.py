"""Causal data cleaning: each rule on hand-made data, plus the no-peeking guarantee.

Cleaning runs before every model, during training *and* online, so a rule that looked at the
future would leak information into the back-test and make the results look better than they
can be in the plant. The causality tests alter or drop everything after hour t and require the
cleaned history up to t to stay bit-for-bit identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm.config import SENSORS
from sentinel_pdm.quality import FFILL_LIMIT_H, STUCK_HOURS, WARMUP_H, clean_readings

QUALITY_COLUMNS = ["asset_id", "running_pct", "missing_pct", "spikes_removed", "stuck_hours"]


def _clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean, quality = clean_readings(raw)
    assert len(quality) == raw["asset_id"].nunique()
    return clean, quality.iloc[0]


# --------------------------------------------------------------------------------------------
# individual rules
# --------------------------------------------------------------------------------------------
def test_clean_data_passes_through_unchanged(make_pump):
    raw = make_pump(hours=300)
    clean, q = _clean(raw)
    for tag in SENSORS:
        np.testing.assert_array_equal(clean[tag].to_numpy(), raw[tag].to_numpy(), err_msg=tag)
    assert q["spikes_removed"] == 0 and q["stuck_hours"] == 0 and q["missing_pct"] == 0
    assert q["running_pct"] == 100.0


def test_output_keeps_schema_and_row_order(make_pump):
    raw = pd.concat([make_pump(hours=120, asset_id="P-101A"), make_pump(hours=120, asset_id="P-101B", seed=1)],
                    ignore_index=True)
    clean, quality = clean_readings(raw)
    assert list(clean.columns) == list(raw.columns)
    pd.testing.assert_frame_equal(clean[["asset_id", "hour", "running"]], raw[["asset_id", "hour", "running"]])
    assert all(clean[tag].dtype == np.float32 for tag in SENSORS)
    assert list(quality.columns) == QUALITY_COLUMNS
    assert list(quality["asset_id"]) == ["P-101A", "P-101B"]


@pytest.mark.parametrize("tag, spike", [("vib_rms", lambda x: 4.0 * x), ("motor_current", lambda x: 2.5 * x),
                                        ("flow", lambda x: 0.2 * x), ("bearing_temp", lambda x: x + 20.0)])
def test_injected_spike_is_removed_and_bridged(make_pump, tag, spike):
    raw = make_pump(hours=200)
    raw.loc[100, tag] = spike(raw.loc[100, tag])
    clean, q = _clean(raw)
    # the spike is replaced by the last good value (a one-hour gap is forward-filled)
    assert clean.loc[100, tag] == raw.loc[99, tag]
    assert q["spikes_removed"] == 1
    others = clean.index != 100
    np.testing.assert_array_equal(clean.loc[others, tag].to_numpy(), raw.loc[others, tag].to_numpy())


def test_genuine_step_change_survives_after_warmup_rows(make_pump):
    """A sustained level change is not a spike: once the trailing median catches up it is kept."""
    raw = make_pump(hours=300)
    raw.loc[150:, "vib_rms"] = raw.loc[150:, "vib_rms"] * 1.5   # within the 1.8x rate-of-change limit
    clean, q = _clean(raw)
    np.testing.assert_array_equal(clean["vib_rms"].to_numpy(), raw["vib_rms"].to_numpy())
    assert q["spikes_removed"] == 0


def test_stuck_tag_is_removed(make_pump):
    raw = make_pump(hours=200)
    raw.loc[50:89, "motor_current"] = raw.loc[50, "motor_current"]   # transmitter flat-lines for 40 h
    clean, q = _clean(raw)
    x = clean["motor_current"]
    first_flagged = 50 + STUCK_HOURS - 1          # only knowable once the value has repeated 6 times
    # the stuck stretch is bridged for at most FFILL_LIMIT_H hours, then stays missing
    assert x.loc[first_flagged + FFILL_LIMIT_H: 89].isna().all()
    assert x.loc[90:].notna().all(), "the tag recovers as soon as it moves again"
    assert q["stuck_hours"] == 90 - first_flagged


def test_short_gaps_filled_long_gaps_kept(make_pump):
    raw = make_pump(hours=300)
    raw.loc[100:103, "flow"] = np.nan               # 4 h dropout
    raw.loc[200:219, "flow"] = np.nan               # 20 h comms outage
    clean, q = _clean(raw)
    x = clean["flow"]
    assert (x.loc[100:103] == raw.loc[99, "flow"]).all()
    # causal: the first hours of any gap are bridged (nobody knows yet that it will be long)...
    assert (x.loc[200:200 + FFILL_LIMIT_H - 1] == raw.loc[199, "flow"]).all()
    # ...but beyond the limit the gap is kept as missing instead of inventing data
    assert x.loc[200 + FFILL_LIMIT_H: 219].isna().all()
    assert x.loc[220:].notna().all()
    assert q["missing_pct"] == pytest.approx(round(100 * 24 / (300 * len(SENSORS)), 2))


def test_not_running_rows_become_nan_and_are_not_filled(make_pump):
    running = np.ones(200)
    running[80:100] = 0
    raw = make_pump(hours=200, running=running)
    clean, q = _clean(raw)
    assert clean.loc[80:99, list(SENSORS)].isna().all().all()
    assert clean.loc[100:, list(SENSORS)].notna().all().all()
    assert q["running_pct"] == 90.0
    assert q["missing_pct"] == 0    # a stopped pump is not a data-quality problem


def test_restart_transient_is_not_mistaken_for_a_spike(make_pump):
    running = np.ones(200)
    running[80:100] = 0
    raw = make_pump(hours=200, running=running)
    # the winding heats up from cold after a restart (first-order thermal lag): +16 degC in the
    # first hour alone, far beyond the 9 degC/h spike limit, yet perfectly real
    raw.loc[100:100 + WARMUP_H - 1, "motor_temp"] -= (30 * 0.45 ** np.arange(WARMUP_H)).astype(np.float32)
    clean, q = _clean(raw)
    np.testing.assert_array_equal(clean.loc[100:, "motor_temp"].to_numpy(), raw.loc[100:, "motor_temp"].to_numpy())
    assert q["spikes_removed"] == 0


def test_physically_impossible_values_are_removed(make_pump):
    raw = make_pump(hours=200)
    raw.loc[60, "flow"] = -40.0
    raw.loc[61:62, "vib_rms"] = -1.0
    clean, _ = _clean(raw)
    assert (clean["flow"].dropna() >= 0).all() and (clean["vib_rms"].dropna() >= 0).all()
    assert clean.loc[60, "flow"] == raw.loc[59, "flow"]


def test_simulated_faults_are_cleaned(fleet, cleaned):
    """On the simulated historian data the cleaner finds the injected spikes and stuck tags."""
    clean, quality = cleaned
    assert list(quality.columns) == QUALITY_COLUMNS
    assert quality["spikes_removed"].sum() > 0 and quality["stuck_hours"].sum() > 0
    assert (quality["running_pct"].between(0, 100)).all()
    stopped = clean["running"] == 0
    assert clean.loc[stopped, list(SENSORS)].isna().all().all()
    for tag in ("vib_rms", "motor_current", "flow"):
        assert (clean[tag].dropna() >= 0).all()


# --------------------------------------------------------------------------------------------
# causality
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t", [700, 2222, 4000])
def test_cleaning_is_causal_when_the_future_changes(fleet, cleaned, t):
    raw = fleet.readings.copy()
    future = raw["hour"] > t
    rng = np.random.default_rng(t)
    for tag in SENSORS:
        raw.loc[future, tag] = (raw.loc[future, tag] * rng.uniform(0.2, 5.0, future.sum())).astype(np.float32)
    raw.loc[future & (rng.random(len(raw)) < 0.05), "vib_rms"] = np.nan
    raw.loc[future & (rng.random(len(raw)) < 0.02), "running"] = np.int8(0)
    altered, _ = clean_readings(raw)
    past = fleet.readings["hour"] <= t
    pd.testing.assert_frame_equal(altered.loc[past], cleaned[0].loc[past], check_exact=True)
    assert not np.allclose(altered.loc[~past, "vib_rms"], cleaned[0].loc[~past, "vib_rms"], equal_nan=True)


def test_cleaning_is_causal_when_the_future_is_cut_off(fleet, cleaned):
    t = 1500
    raw = fleet.readings
    truncated, _ = clean_readings(raw[raw["hour"] <= t].reset_index(drop=True))
    expected = cleaned[0][raw["hour"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, expected)
