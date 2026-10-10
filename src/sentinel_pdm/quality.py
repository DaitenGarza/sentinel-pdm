"""Causal data cleaning for historian sensor data.

Every rule only looks at the current and *past* values of a tag, so the exact same cleaning
can run online, hour by hour, without peeking at the future (which would silently leak
information into the model during back-testing).

Rules, applied per pump:
1. Pump not running  -> process readings are meaningless, set to NaN.
2. Physically impossible values (negative vibration, current, flow...) -> NaN.
3. Stuck / flatlined tag: the same value repeated for 6+ consecutive hours -> NaN.
4. Spikes: rate-of-change limits against a trailing median. Vibration / pressure / current do
   not double versus their last-24 h median between two hourly averages unless the transmitter
   glitched. Temperatures follow the daily cycle, so they are checked against the median of the
   last 3 hours instead (thermal mass makes a +10 degC jump in one hour implausible). The first
   hours after a restart are exempt: a motor legitimately heats up 20 degC in its first hour.
5. Short gaps (<= 6 h) are forward-filled (last good value), longer gaps stay missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SENSORS

STUCK_HOURS = 6
FFILL_LIMIT_H = 6
MEDIAN_WINDOW_H = 24
TEMP_MEDIAN_WINDOW_H = 3
WARMUP_H = 6
# Spike rule per tag: ("ratio", r) -> outside [median / r, median * r];  ("abs", d) -> |x - median| > d
SPIKE_RULES: dict[str, tuple[str, float]] = {
    "vib_rms": ("ratio", 1.8),
    "vib_1x": ("ratio", 1.8),
    "vib_2x": ("ratio", 2.2),
    "vib_hf": ("ratio", 2.2),
    "vib_kurtosis": ("ratio", 1.7),
    "bearing_temp": ("abs", 8.0),
    "motor_temp": ("abs", 9.0),
    "motor_current": ("ratio", 1.6),
    "current_imbalance": ("ratio", 2.5),
    "suction_pressure": ("ratio", 1.8),
    "discharge_pressure": ("ratio", 1.6),
    "flow": ("ratio", 1.6),
}
NON_NEGATIVE = {"vib_rms", "vib_1x", "vib_2x", "vib_hf", "vib_kurtosis", "motor_current",
                "current_imbalance", "flow"}


def _run_length_equal(x: np.ndarray) -> np.ndarray:
    """Length of the current run of identical consecutive values (NaN breaks a run)."""
    same = np.zeros(len(x), dtype=bool)
    same[1:] = x[1:] == x[:-1]  # NaN == NaN is False
    pos = np.arange(len(x))
    run_start = np.maximum.accumulate(np.where(same, 0, pos))
    return pos - run_start


def _hours_since_start(run: np.ndarray) -> np.ndarray:
    """Hours the pump has been running continuously (0 at the first running hour)."""
    pos = np.arange(len(run))
    last_stop = np.maximum.accumulate(np.where(run, -1, pos))
    return np.where(run, pos - last_stop - 1, -1)


def _clean_tag(x: np.ndarray, run: np.ndarray, tag: str) -> tuple[np.ndarray, dict]:
    raw_missing = np.isnan(x) & run
    x = x.copy()
    x[~run] = np.nan
    if tag in NON_NEGATIVE:
        x[x < 0] = np.nan
    stuck = _run_length_equal(x) >= STUCK_HOURS - 1
    x[stuck] = np.nan
    kind, lim = SPIKE_RULES[tag]
    window, min_p = (MEDIAN_WINDOW_H, 6) if kind == "ratio" else (TEMP_MEDIAN_WINDOW_H, 2)
    med = pd.Series(x).shift(1).rolling(window, min_periods=min_p).median().to_numpy(copy=True)
    with np.errstate(invalid="ignore"):
        if kind == "ratio":
            base = np.abs(med) + 1e-6
            spike = (np.abs(x) > base * lim) | (np.abs(x) < base / lim)
        else:
            spike = np.abs(x - med) > lim
    spike &= ~np.isnan(med) & ~np.isnan(x) & (_hours_since_start(run) >= WARMUP_H)
    x[spike] = np.nan
    filled = pd.Series(x).ffill(limit=FFILL_LIMIT_H).to_numpy(copy=True)
    filled[~run] = np.nan
    stats = {
        "missing": int(raw_missing.sum()),
        "spikes": int(spike.sum()),
        "stuck": int((stuck & run).sum()),
    }
    return filled, stats


def clean_readings(readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned readings, per-asset data-quality summary).

    Input must be sorted by asset_id, hour (the order produced by the simulator / loaders).
    """
    df = readings.copy()
    running = df["running"].to_numpy() == 1
    groups = df.groupby("asset_id", sort=False).indices
    stats: dict[str, dict[str, int]] = {a: {"missing": 0, "spikes": 0, "stuck": 0} for a in groups}
    for tag in SENSORS:
        values = df[tag].to_numpy(dtype=np.float64)
        out = np.empty_like(values)
        for asset_id, idx in groups.items():
            out[idx], st = _clean_tag(values[idx], running[idx], tag)
            for k, v in st.items():
                stats[asset_id][k] += v
        df[tag] = out.astype(np.float32)

    rows = []
    for asset_id, idx in groups.items():
        n_run = max(int(running[idx].sum()), 1)
        rows.append(
            {
                "asset_id": asset_id,
                "running_pct": round(100 * running[idx].mean(), 1),
                "missing_pct": round(100 * stats[asset_id]["missing"] / (n_run * len(SENSORS)), 2),
                "spikes_removed": stats[asset_id]["spikes"],
                "stuck_hours": stats[asset_id]["stuck"],
            }
        )
    return df, pd.DataFrame(rows)
