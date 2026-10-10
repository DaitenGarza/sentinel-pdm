"""Why the gradient-boosting models are kept as a cross-validation ensemble (docs/METHODOLOGY.md 6.7).

The alert threshold and the probability calibration are tuned on out-of-fold (OOF) scores. The
textbook alternative to an ensemble is to refit one model on all training pumps afterwards. This
script measures what that refit does on the held-out test fleet when it is put through the same
OOF-fitted calibration and the same alert threshold.

Run after `python -m sentinel_pdm train` (it reads data/ and artifacts/ and writes nothing):

    python scripts/refit_vs_ensemble.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sentinel_pdm.config import ARTIFACTS_DIR, DATA_DIR, Config
from sentinel_pdm.evaluation import alert_episodes, event_metrics, match_events
from sentinel_pdm.features import FEATURE_COLUMNS, build_features
from sentinel_pdm.labels import make_labels
from sentinel_pdm.models import _clf, calibrate, stride_mask
from sentinel_pdm.scoring import load_bundle


def main() -> None:
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(metrics["config"])
    bundle = load_bundle()
    clean = pd.read_parquet(DATA_DIR / "clean.parquet")
    events = pd.read_csv(DATA_DIR / "events.csv")
    assets = pd.read_csv(DATA_DIR / "assets.csv")

    feats = build_features(clean, events)
    labels = make_labels(feats[["asset_id", "hour"]], events, cfg.sim.hours, cfg.labels)
    split = feats["asset_id"].map(assets.set_index("asset_id")["split"]).to_numpy()
    running = feats["running"].to_numpy() == 1
    y = labels["label"].to_numpy()

    fit = running & (split == "train") & (y >= 0) & stride_mask(feats["hour"].to_numpy(), cfg.train.train_stride_h)
    print(f"Refitting one gradient-boosting model on {fit.sum():,} rows of {int((assets.split == 'train').sum())} "
          "training pumps...")
    refit = _clf(cfg.train).fit(feats.loc[fit, FEATURE_COLUMNS], y[fit])

    test = split == "test"
    frame = feats.loc[test, ["asset_id", "hour"]].copy()
    run_test = running[test]
    X = feats.loc[test].loc[run_test, FEATURE_COLUMNS]
    frame["refit"] = np.nan
    frame.loc[X.index, "refit"] = calibrate(bundle.risk.calibrator, refit.predict_proba(X)[:, 1])
    frame["ensemble"] = np.nan
    frame.loc[X.index, "ensemble"] = bundle.risk.predict_proba(X)

    thr = bundle.thresholds["ml"]
    ev_test = events[events["asset_id"].isin(assets.loc[assets.split == "test", "asset_id"])]
    years = (assets.split == "test").sum() * cfg.sim.hours / 8760.0
    print(f"\nHeld-out test fleet, gradient boosting only, same calibration and threshold ({thr:.3g}):\n")
    print(f"{'Scoring model':34s} {'Caught':>8s} {'False alarms':>13s} {'Median lead (h)':>16s}")
    for name, col in (("CV ensemble (shipped)", "ensemble"), ("Single model refitted on all data", "refit")):
        fo, ep = match_events(alert_episodes(frame, col, thr, cfg.alerts), ev_test, cfg.alerts, cfg.sim.hours)
        m = event_metrics(fo, ep, years)
        print(f"{name:34s} {m['caught']:>3d}/{m['failures']:<4d} {m['false_alarms']:>13d} {m['median_lead_h']:>16.0f}")
    print("\nThe refit sees a third more data than any fold model, so its score distribution differs from the "
          "out-of-fold one the threshold was tuned on: the same threshold no longer means the same alert rate.")


if __name__ == "__main__":
    main()
