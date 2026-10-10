"""Supervised targets derived from the maintenance log.

For every hour of every pump:

* ``ttf``          hours until the next logged failure (NaN if none is observed)
* ``label``        1 = a failure happens within the horizon (default 7 days), 0 = it does not,
                   -1 = unknown (right-censored: the record ends before the horizon elapses)
* ``rul_target``   piecewise-linear remaining useful life: min(ttf, cap). Healthy hours far from
                   any failure get the cap; censored hours without enough look-ahead are NaN.
* ``next_mode``    failure mode of the next observed failure ("none" if no failure follows)
* ``next_predictable`` whether that next failure had a measurable precursor (not "sudden")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig


def make_labels(keys: pd.DataFrame, events: pd.DataFrame, record_end: int, cfg: LabelConfig) -> pd.DataFrame:
    """keys: frame with asset_id, hour (any order). record_end: first hour *after* the record."""
    n = len(keys)
    ttf = np.full(n, np.nan)
    next_mode = np.full(n, "none", dtype=object)
    next_predictable = np.zeros(n, dtype=bool)
    failures = events[events["event_type"] == "failure"]
    fail_by_asset = {a: g.sort_values("hour") for a, g in failures.groupby("asset_id")}
    for asset_id, idx in keys.groupby("asset_id", sort=False).indices.items():
        f = fail_by_asset.get(asset_id)
        if f is None or f.empty:
            continue
        hours = keys["hour"].to_numpy()[idx]
        fh = f["hour"].to_numpy()
        j = np.searchsorted(fh, hours, side="right")  # next failure strictly after this hour
        has = j < len(fh)
        jj = np.minimum(j, len(fh) - 1)
        ttf[idx] = np.where(has, fh[jj] - hours, np.nan)
        next_mode[idx] = np.where(has, f["mode"].to_numpy()[jj], "none")
        next_predictable[idx] = np.where(has, f["predictable"].to_numpy()[jj], False)

    hours_all = keys["hour"].to_numpy()
    lookahead = record_end - hours_all  # hours of future we can observe
    label = np.where(ttf <= cfg.horizon_h, 1, 0)
    censored = np.isnan(ttf) & (lookahead <= cfg.horizon_h)
    label = np.where(censored, -1, label)
    rul = np.where(np.isnan(ttf), np.where(lookahead >= cfg.rul_cap_h, cfg.rul_cap_h, np.nan),
                   np.minimum(ttf, cfg.rul_cap_h))
    return pd.DataFrame(
        {
            "ttf": ttf,
            "label": label.astype(np.int8),
            "rul_target": rul,
            "next_mode": next_mode,
            "next_predictable": next_predictable,
        },
        index=keys.index,
    )
