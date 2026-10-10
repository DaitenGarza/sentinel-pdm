"""Targets from the maintenance log: horizon labels, censoring, capped RUL and the next failure mode.

All cases are hand-made so every expected value can be checked with pencil and paper. The
censoring rules matter for honest evaluation: an hour near the end of the record whose next
7 days are not observed must be "unknown" (-1), not "healthy" (0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm.config import LabelConfig
from sentinel_pdm.labels import make_labels

RECORD_END = 1000   # first hour after the record
CFG = LabelConfig(horizon_h=168, rul_cap_h=720)


def _event(asset_id, event_type, hour, mode, predictable):
    return {"asset_id": asset_id, "event_type": event_type, "hour": hour, "end_hour": hour + 24, "mode": mode,
            "predictable": predictable}


@pytest.fixture(scope="module")
def log() -> pd.DataFrame:
    """Pump A: bearing failure at 500, PM at 600 (ignored by labels), sudden failure at 900.
    Pump B: never fails."""
    return pd.DataFrame([
        _event("A", "failure", 500, "bearing", True),
        _event("A", "pm", 600, "none", False),
        _event("A", "failure", 900, "sudden", False),
    ])


@pytest.fixture(scope="module")
def labels(log) -> pd.DataFrame:
    keys = pd.DataFrame({"asset_id": ["A"] * RECORD_END + ["B"] * RECORD_END,
                         "hour": np.tile(np.arange(RECORD_END), 2)})
    out = make_labels(keys, log, RECORD_END, CFG)
    return pd.concat([keys, out], axis=1).set_index(["asset_id", "hour"])


def test_output_schema(labels):
    assert list(labels.columns) == ["ttf", "label", "rul_target", "next_mode", "next_predictable"]
    assert labels["label"].dtype == np.int8
    assert set(labels["label"].unique()) <= {-1, 0, 1}


@pytest.mark.parametrize("hour, ttf", [(0, 500), (400, 100), (499, 1), (500, 400), (899, 1), (900, np.nan),
                                       (999, np.nan)])
def test_time_to_failure_counts_to_the_next_failure_strictly_after(labels, hour, ttf):
    np.testing.assert_equal(labels.loc[("A", hour), "ttf"], ttf)


def test_horizon_labels(labels):
    a = labels.loc["A", "label"]
    assert a.loc[332:499].eq(1).all()       # failure at 500 is within 168 h
    assert a.loc[331] == 0                  # 169 h away: outside the horizon
    assert a.loc[500] == 0                  # the failure hour itself looks ahead to the next failure
    assert a.loc[732:899].eq(1).all()       # sudden failures are labelled too (they are real failures)
    assert a.loc[731] == 0


def test_pm_work_orders_do_not_create_labels(labels):
    a = labels.loc["A"]
    assert a.loc[590:610, "label"].eq(0).all()
    assert a.loc[600, "ttf"] == 300


def test_censoring_near_the_end_of_the_record(labels):
    b = labels.loc["B", "label"]
    # hour 831 looks ahead to hour 999, the last recorded hour: fully observed, no failure -> 0
    assert b.loc[:831].eq(0).all()
    # from hour 832 on, part of the next 7 days lies beyond the record: unknown, not healthy
    assert b.loc[832:].eq(-1).all()
    a = labels.loc["A", "label"]
    assert a.loc[900:].eq(-1).all()


def test_rul_is_capped_and_censored(labels):
    b = labels.loc["B", "rul_target"]
    assert b.loc[:280].eq(CFG.rul_cap_h).all()      # >= 720 h observed without failure -> capped
    assert b.loc[281:].isna().all()                  # less than 720 h of look-ahead: unknown
    a = labels.loc["A", "rul_target"]
    assert a.loc[0] == 500 and a.loc[499] == 1
    assert a.loc[950:].isna().all()


def test_rul_cap_applies_to_distant_failures():
    keys = pd.DataFrame({"asset_id": "C", "hour": [0, 100, 779, 780, 1400]})
    log = pd.DataFrame([_event("C", "failure", 1500, "winding", True)])
    out = make_labels(keys, log, 2000, CFG)
    assert out["rul_target"].tolist() == [720, 720, 720, 720, 100]
    assert out["ttf"].tolist() == [1500, 1400, 721, 720, 100]


def test_next_mode_and_predictability(labels):
    a = labels.loc["A"]
    assert a.loc[400, "next_mode"] == "bearing" and bool(a.loc[400, "next_predictable"])
    assert a.loc[600, "next_mode"] == "sudden" and not bool(a.loc[600, "next_predictable"])
    assert a.loc[950, "next_mode"] == "none" and not bool(a.loc[950, "next_predictable"])
    b = labels.loc["B"]
    assert b["next_mode"].eq("none").all() and not b["next_predictable"].any()


def test_keys_in_any_order_are_labelled_by_index(log, labels):
    rng = np.random.default_rng(0)
    keys = pd.DataFrame({"asset_id": ["A"] * RECORD_END + ["B"] * RECORD_END,
                         "hour": np.tile(np.arange(RECORD_END), 2)})
    shuffled = keys.iloc[rng.permutation(len(keys))]
    out = make_labels(shuffled, log, RECORD_END, CFG)
    assert out.index.equals(shuffled.index)
    ordered = pd.concat([shuffled, out], axis=1).set_index(["asset_id", "hour"]).sort_index()
    pd.testing.assert_frame_equal(ordered, labels.sort_index())


def test_custom_horizon():
    keys = pd.DataFrame({"asset_id": "A", "hour": np.arange(100)})
    log = pd.DataFrame([_event("A", "failure", 80, "cavitation", True)])
    out = make_labels(keys, log, 100, LabelConfig(horizon_h=24, rul_cap_h=48))
    assert out["label"].iloc[56:80].eq(1).all() and out["label"].iloc[:56].eq(0).all()
    assert out["rul_target"].iloc[0] == 48 and out["rul_target"].iloc[79] == 1


def test_pump_without_events_and_empty_log():
    keys = pd.DataFrame({"asset_id": "Z", "hour": np.arange(10)})
    empty = pd.DataFrame(columns=["asset_id", "event_type", "hour", "end_hour", "mode", "predictable"])
    out = make_labels(keys, empty, 10, CFG)
    assert out["ttf"].isna().all() and out["label"].eq(-1).all() and out["next_mode"].eq("none").all()
