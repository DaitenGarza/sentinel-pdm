"""Feature engineering: turn hourly sensor readings into degradation indicators.

Design rules
------------
* **Strictly causal.** Every feature at hour t uses data from hours <= t only (trailing
  windows, no centring, no back-filling). A unit test perturbs the future and asserts the past
  features do not move.
* **Maintenance-aware.** Rolling windows restart at every maintenance work order taken from the
  maintenance log (CMMS), so a freshly repaired pump does not inherit the pre-failure trend.
* **Self-referenced.** Each pump is compared with *its own* healthy baseline: the median of the
  first 168 running hours after its last maintenance. A 2.0 mm/s vibration is normal for one
  pump and a warning sign for another; deviation from baseline captures that.
* **Physics first.** Pump head (discharge - suction), temperature rise over ambient, and a
  hydraulic-efficiency proxy (flow x head / current) are computed explicitly because they are
  what a rotating-equipment engineer would look at.
* **Operating-point aware.** Process load gets the same statistics as the sensors, so a pump
  pushed harder than during its baseline week is not mistaken for a degrading one.

For scale-dependent signals (current, pressures, flow, head, efficiency) every statistic is
divided by the pump's own baseline, so pumps of different sizes become comparable.

For each of the 17 signals (12 sensors, 4 physics-derived, process load) the following are produced:
  ``_m24``  24 h mean          ``_sd24``  24 h standard deviation
  ``_m168`` 7 day mean         ``_sl24``  24 h trend slope (units per day)
  ``_sl168`` 7 day slope       ``_dev``   24 h mean vs. post-maintenance baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SENSORS

DERIVED = ("head", "bearing_rise", "motor_rise", "hyd_eff")
# Process load is tracked like a signal so every model sees *operating-point* changes: when the
# plant pushes more throughput, head falls and current/flow rise along the pump curve - that is
# not degradation, and the load deviation lets the models tell the two apart.
OPERATING = ("load_pct",)
SIGNALS: tuple[str, ...] = SENSORS + DERIVED + OPERATING
# Deviation is relative (ratio - 1) for scale-type signals, absolute for temperatures/kurtosis.
ABSOLUTE_DEV = {"bearing_temp", "motor_temp", "bearing_rise", "motor_rise", "vib_kurtosis"}
# Signals whose absolute size depends on the pump (a 600 m3/h pump vs a 150 m3/h one): every
# statistic is expressed relative to the pump's own baseline so the model learns patterns
# ("flow down 8 %") rather than memorising individual pumps.
SCALE_FREE_SIGNALS = ("motor_current", "suction_pressure", "discharge_pressure", "flow", "head", "hyd_eff")
SHORT_W, LONG_W = 24, 168
BASELINE_H = 168
CONTEXT_FEATURES = ("load_pct", "ambient_temp", "speed_rpm", "hours_since_maint", "run_hours_since_maint")
SUFFIXES = ("_m24", "_sd24", "_m168", "_sl24", "_sl168", "_dev")
FEATURE_COLUMNS: list[str] = [f"{s}{suf}" for s in SIGNALS for suf in SUFFIXES] + list(CONTEXT_FEATURES)


def maintenance_cycles(keys: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Cycle index and time since last maintenance for each (asset_id, hour) row.

    A new cycle starts at every work order (failure repair or preventive maintenance).
    ``hours_since_maint`` is NaN before the first logged maintenance (install date unknown).
    """
    out = pd.DataFrame(index=keys.index)
    cycle = np.zeros(len(keys), dtype=np.int32)
    since = np.full(len(keys), np.nan)
    ev_by_asset = {a: g.sort_values("hour") for a, g in events.groupby("asset_id")} if len(events) else {}
    for asset_id, idx in keys.groupby("asset_id", sort=False).indices.items():
        hours = keys["hour"].to_numpy()[idx]
        ev = ev_by_asset.get(asset_id)
        if ev is None or ev.empty:
            continue
        starts = ev["hour"].to_numpy()
        ends = ev["end_hour"].to_numpy()
        c = np.searchsorted(starts, hours, side="right")
        cycle[idx] = c
        last_end = np.where(c > 0, ends[np.maximum(c - 1, 0)], np.nan)
        s = hours - last_end
        since[idx] = np.where(s >= 0, s, np.nan)
    out["cycle"] = cycle
    out["hours_since_maint"] = since
    return out


def _window_sums(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing window sums over axis 0 (window includes the current row)."""
    c = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
    n = a.shape[0]
    hi = np.arange(1, n + 1)
    lo = np.maximum(hi - w, 0)
    return c[hi] - c[lo]


def _rolling_block(x: np.ndarray, t: np.ndarray, w: int, min_count: int) -> tuple[np.ndarray, ...]:
    """NaN-aware trailing mean, std and OLS slope for every column of x."""
    valid = ~np.isnan(x)
    xz = np.where(valid, x, 0.0)
    tt = np.where(valid, (t - t[0])[:, None], 0.0)
    n = _window_sums(valid.astype(np.float64), w)
    sx = _window_sums(xz, w)
    sxx = _window_sums(xz * xz, w)
    st = _window_sums(tt, w)
    stt = _window_sums(tt * tt, w)
    sxt = _window_sums(xz * tt, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sx / n
        var = (sxx - sx * sx / n) / (n - 1)
        std = np.sqrt(np.clip(var, 0, None))
        den = n * stt - st * st
        slope = np.where(den > 1e-9, (n * sxt - sx * st) / den, np.nan)
    bad = n < min_count
    for arr in (mean, std, slope):
        arr[bad] = np.nan
    return mean, std, slope


def _baseline(x: np.ndarray) -> np.ndarray:
    """Expanding median over the first BASELINE_H valid samples, then frozen (per column)."""
    out = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        valid_idx = np.flatnonzero(~np.isnan(x[:, j]))[:BASELINE_H]
        if valid_idx.size == 0:
            continue
        out[valid_idx, j] = pd.Series(x[valid_idx, j]).expanding().median().to_numpy()
        out[:, j] = pd.Series(out[:, j]).ffill().to_numpy()
    return out


def add_derived_signals(clean: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=clean.index)
    df["head"] = clean["discharge_pressure"] - clean["suction_pressure"]
    df["bearing_rise"] = clean["bearing_temp"] - clean["ambient_temp"]
    df["motor_rise"] = clean["motor_temp"] - clean["ambient_temp"]
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = clean["flow"] * df["head"] / clean["motor_current"].where(clean["motor_current"] > 1)
    df["hyd_eff"] = eff
    return df


def build_features(
    clean: pd.DataFrame, events: pd.DataFrame, scale_free: tuple[str, ...] = SCALE_FREE_SIGNALS
) -> pd.DataFrame:
    """Compute the model feature matrix for cleaned readings (sorted by asset_id, hour).

    Returns a frame aligned with ``clean`` holding asset_id, hour, running, cycle and every
    column in FEATURE_COLUMNS (float32).
    """
    derived = add_derived_signals(clean)
    sig = np.column_stack(
        [clean[s].to_numpy(dtype=np.float64) if s in clean else derived[s].to_numpy(dtype=np.float64)
         for s in SIGNALS]
    )
    keys = clean[["asset_id", "hour"]]
    cyc = maintenance_cycles(keys, events)
    running = clean["running"].to_numpy() == 1
    hours = clean["hour"].to_numpy().astype(np.float64)

    n, k = sig.shape
    feats = {name: np.full(n, np.nan) for name in FEATURE_COLUMNS}
    run_hours = np.zeros(n)
    group_key = clean["asset_id"].astype(str) + "#" + cyc["cycle"].astype(str)
    for _, idx in clean.groupby(group_key, sort=False).indices.items():
        idx = np.sort(idx)
        x, t = sig[idx], hours[idx]
        m24, sd24, sl24 = _rolling_block(x, t, SHORT_W, min_count=6)
        m168, _, sl168 = _rolling_block(x, t, LONG_W, min_count=24)
        base = _baseline(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = m24 / np.where(np.abs(base) > 1e-6, base, np.nan) - 1.0
        dev = m24 - base
        with np.errstate(invalid="ignore", divide="ignore"):
            inv = 1.0 / np.where(np.abs(base) > 1e-6, np.abs(base), np.nan)
        for j, s in enumerate(SIGNALS):
            norm = inv[:, j] if s in scale_free else 1.0
            feats[f"{s}_m24"][idx] = m24[:, j] * norm
            feats[f"{s}_sd24"][idx] = sd24[:, j] * norm
            feats[f"{s}_m168"][idx] = m168[:, j] * norm
            feats[f"{s}_sl24"][idx] = sl24[:, j] * 24.0 * norm   # per day
            feats[f"{s}_sl168"][idx] = sl168[:, j] * 24.0 * norm
            feats[f"{s}_dev"][idx] = dev[:, j] if s in ABSOLUTE_DEV else rel[:, j]
        run_hours[idx] = np.cumsum(running[idx])

    feats["load_pct"] = clean["load_pct"].to_numpy(dtype=np.float64)
    feats["ambient_temp"] = clean["ambient_temp"].to_numpy(dtype=np.float64)
    feats["speed_rpm"] = clean["speed_rpm"].to_numpy(dtype=np.float64)
    feats["hours_since_maint"] = cyc["hours_since_maint"].to_numpy()
    feats["run_hours_since_maint"] = np.where(np.isnan(feats["hours_since_maint"]), np.nan, run_hours)

    out = pd.DataFrame({name: feats[name].astype(np.float32) for name in FEATURE_COLUMNS}, index=clean.index)
    out.insert(0, "asset_id", clean["asset_id"].to_numpy())
    out.insert(1, "hour", clean["hour"].to_numpy())
    out.insert(2, "running", clean["running"].to_numpy())
    out.insert(3, "cycle", cyc["cycle"].to_numpy())
    # Features are only defined while the pump runs.
    out.loc[~running, FEATURE_COLUMNS] = np.nan
    return out


def signal_of(feature: str) -> str:
    """Map a feature name back to the physical signal it summarises."""
    for suf in SUFFIXES:
        if feature.endswith(suf):
            return feature[: -len(suf)]
    return feature
