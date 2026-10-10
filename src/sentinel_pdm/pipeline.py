"""End-to-end pipeline: simulate -> clean -> features -> labels -> train -> evaluate -> save.

Outputs (all under the project root):

data/
  readings.parquet      raw hourly historian data incl. ground-truth columns (gt_*)
  clean.parquet         cleaned sensor data used by the models and dashboard trends
  events.csv            maintenance log (failures + preventive maintenance work orders)
  assets.csv            asset register with the train/test split
  quality.csv           per-pump data-quality summary
artifacts/
  models/bundle.joblib  every trained model + thresholds + reference statistics
  predictions.parquet   hourly risk / RUL / diagnosis / anomaly scores for every pump
  metrics.json          all evaluation results (test fleet + cross-validation)
  alerts.csv            alert raises per method with their outcome
  failure_outcomes.csv  each failure: caught by which method, with how much lead time
  threshold_curve.csv   maintenance cost vs. alert threshold (training fleet, out-of-fold)
  feature_importance.csv permutation importance of the gradient-boosting risk model (test fleet)
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from . import __version__
from .config import ARTIFACTS_DIR, CONTEXT, DATA_DIR, FAILURE_MODES, MODELS_DIR, SENSORS, Config
from .evaluation import alert_episodes, choose_threshold, event_metrics, match_events, policy_cost, row_metrics
from .explain import reference_stats
from .features import FEATURE_COLUMNS, build_features, signal_of
from .labels import make_labels
from .models import train_models
from .quality import clean_readings
from .scoring import ModelBundle, assemble_predictions, finalize_predictions, model_scores, save_bundle
from .simulator import simulate_fleet

METHODS = {
    "hybrid": "Sentinel hybrid (GBM + MSPC)",
    "ml": "Gradient boosting only",
    "mspc": "MSPC (PCA T² / SPE)",
    "iforest": "Isolation Forest",
    "rule": "Alarm limits (ISO 10816 + RTD)",
}
SCORE_COLUMNS = {"hybrid": "risk", "ml": "risk_gbm", "mspc": "mspc_score", "iforest": "iforest_score",
                 "rule": "rule_score"}
FULL_PRECISION_COLUMNS = ["risk", "risk_gbm"]


def _clean_nans(obj):
    if isinstance(obj, dict):
        return {str(k): _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class _GBMEnsemble:
    """Minimal estimator wrapper so permutation importance scores the whole CV ensemble."""

    def __init__(self, risk_model):
        self.risk_model = risk_model

    def fit(self, X, y):  # pragma: no cover - never refitted
        return self


def _ap_scorer(est: _GBMEnsemble, X: pd.DataFrame, y: np.ndarray) -> float:
    return float(average_precision_score(y, est.risk_model.raw_score(X)))


def run_pipeline(
    cfg: Config | None = None,
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    log: Callable[[str], None] = print,
    importance: bool = True,
) -> dict:
    cfg = cfg or Config()
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    models_dir = artifacts_dir / MODELS_DIR.name
    for d in (data_dir, artifacts_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    def step(name: str, t0: float) -> float:
        timings[name] = round(time.perf_counter() - t0, 2)
        log(f"  done in {timings[name]:.1f}s")
        return time.perf_counter()

    # 1. data ---------------------------------------------------------------------------------
    t = time.perf_counter()
    log(f"[1/7] Simulating {cfg.sim.n_assets} pumps x {cfg.sim.hours} h of historian data...")
    fleet = simulate_fleet(cfg.sim)
    raw, events, assets = fleet.readings, fleet.events, fleet.assets
    fails = events[events["event_type"] == "failure"]
    log(f"  {len(raw):,} hourly rows | {len(fails)} failures ({int((~fails['predictable']).sum())} sudden) | "
        f"{int((events['event_type'] == 'pm').sum())} PM work orders")
    raw.to_parquet(data_dir / "readings.parquet", index=False)
    events.to_csv(data_dir / "events.csv", index=False)
    assets.to_csv(data_dir / "assets.csv", index=False)
    t = step("simulate", t)

    log("[2/7] Cleaning sensor data (stuck tags, spikes, dropouts)...")
    clean, quality = clean_readings(raw)
    keep = ["asset_id", "hour", "timestamp", "running", *CONTEXT, *SENSORS]
    clean[keep].to_parquet(data_dir / "clean.parquet", index=False)
    quality.to_csv(data_dir / "quality.csv", index=False)
    t = step("clean", t)

    log("[3/7] Engineering features and labels...")
    F = build_features(clean, events)
    L = make_labels(F[["asset_id", "hour"]], events, cfg.sim.hours, cfg.labels)
    split = F["asset_id"].map(assets.set_index("asset_id")["split"]).to_numpy()
    running = F["running"].to_numpy() == 1
    hours = F["hour"].to_numpy()
    y = L["label"].to_numpy()
    log(f"  {len(FEATURE_COLUMNS)} features | positive rate {np.mean(y[running & (y >= 0)] == 1):.2%}")
    t = step("features", t)

    # 2. models -------------------------------------------------------------------------------
    log(f"[4/7] Training models: {cfg.train.cv_folds}-fold cross-validation by pump on the training fleet...")
    tr = running & (split == "train")
    tm = train_models(F.loc[tr], L.loc[tr], F.loc[tr, "asset_id"].to_numpy(), hours[tr], cfg.train,
                      cfg.labels.rul_cap_h, log=log)
    healthy_tr = tr & (L["rul_target"].to_numpy() >= cfg.labels.rul_cap_h)
    bundle = ModelBundle(risk=tm.risk, rul=tm.rul, diagnosis=tm.diagnosis, mspc=tm.mspc, iforest=tm.iforest,
                         hybrid=tm.hybrid, thresholds={}, ref_stats=reference_stats(F.loc[healthy_tr]),
                         config=cfg.to_dict())
    t = step("train", t)

    # 3. scores for every hour of every pump ------------------------------------------------------
    log("[5/7] Scoring the test fleet and choosing alert thresholds by maintenance cost...")
    te = running & (split == "test")
    scores = pd.concat([tm.oof, model_scores(F.loc[te], bundle)]).sort_index()
    preds = assemble_predictions(F, clean, scores, bundle.ref_stats, threshold=0.5)
    preds.insert(0, "asset_id", F["asset_id"].to_numpy())
    preds.insert(1, "hour", hours)
    preds.insert(2, "timestamp", clean["timestamp"].to_numpy())
    preds.insert(3, "split", split)
    preds.insert(4, "running", F["running"].to_numpy())
    for c in ["label", "ttf", "rul_target", "next_mode", "next_predictable"]:
        preds[c] = L[c].to_numpy()
    for c in ["gt_mode", "gt_severity", "gt_state"]:
        preds[c] = raw[c].to_numpy()

    train_ids = sorted(assets.loc[assets["split"] == "train", "asset_id"])
    test_ids = sorted(assets.loc[assets["split"] == "test", "asset_id"])
    train_years = len(train_ids) * cfg.sim.hours / 8760.0
    test_years = len(test_ids) * cfg.sim.hours / 8760.0
    ev_tr = events[events["asset_id"].isin(train_ids)]
    ev_te = events[events["asset_id"].isin(test_ids)]
    frame_tr = preds[preds["split"] == "train"]
    frame_te = preds[preds["split"] == "test"]
    thresholds, curves = {"rule": 1.0}, []
    for key in ("hybrid", "ml", "mspc", "iforest"):
        thr, curve = choose_threshold(frame_tr, SCORE_COLUMNS[key], ev_tr, cfg.alerts, cfg.costs, cfg.sim.hours,
                                      train_years)
        thresholds[key] = thr
        curves.append(curve.assign(method=key))
    bundle.thresholds = thresholds
    preds = finalize_predictions(preds, preds["running"].to_numpy(), thresholds["hybrid"])
    log("  thresholds: " + ", ".join(f"{k}={v:.3g}" for k, v in thresholds.items()))
    t = step("score", t)

    # 4. evaluation ---------------------------------------------------------------------------
    log("[6/7] Evaluating on the held-out test fleet...")
    known = preds["running"].eq(1) & preds["label"].ge(0)
    yte = preds.loc[known & preds["split"].eq("test")]
    ytr = preds.loc[known & preds["split"].eq("train")]
    # "predictable only" excludes the 7-day windows before sudden failures, which no method can rank
    sudden_window = preds["label"].eq(1) & ~preds["next_predictable"].astype(bool)
    yte_p = preds.loc[known & preds["split"].eq("test") & ~sudden_window]
    metrics: dict = {"test": {"row": {}, "row_predictable": {}, "event": {}, "economics": {}},
                     "cv": {"row": {}, "event": {}}}
    all_alerts, outcomes = [], []
    for key, col in SCORE_COLUMNS.items():
        prob = key in ("hybrid", "ml")
        metrics["test"]["row"][key] = row_metrics(yte["label"].to_numpy(), yte[col].to_numpy(), prob)
        metrics["test"]["row_predictable"][key] = row_metrics(yte_p["label"].to_numpy(), yte_p[col].to_numpy(), prob)
        metrics["cv"]["row"][key] = row_metrics(ytr["label"].to_numpy(), ytr[col].to_numpy(), prob)
        for split_name, frame, ev, years in (("test", frame_te, ev_te, test_years),
                                             ("train", frame_tr, ev_tr, train_years)):
            ep = alert_episodes(frame, col, thresholds[key], cfg.alerts)
            fo, ep = match_events(ep, ev, cfg.alerts, cfg.sim.hours)
            metrics["test" if split_name == "test" else "cv"]["event"][key] = event_metrics(fo, ep, years)
            all_alerts.append(ep.assign(method=key, split=split_name))
            outcomes.append(fo.assign(method=key, split=split_name))

    n_fail_te = int((ev_te["event_type"] == "failure").sum())
    econ = {"reactive": policy_cost(n_fail_te, 0, 0, cfg.costs)}
    for key in SCORE_COLUMNS:
        em = metrics["test"]["event"][key]
        econ[key] = policy_cost(n_fail_te, em["caught"], em["false_alarms"], cfg.costs)
    for v in econ.values():
        v["savings_vs_reactive"] = econ["reactive"]["total_cost"] - v["total_cost"]
        v["savings_pct"] = 100 * v["savings_vs_reactive"] / max(econ["reactive"]["total_cost"], 1)
        v["downtime_reduction_pct"] = 100 * (1 - v["downtime_h"] / max(econ["reactive"]["downtime_h"], 1))
        v["cost_per_pump_year"] = v["total_cost"] / test_years
    metrics["test"]["economics"] = econ
    metrics["test"]["asset_years"] = test_years

    # RUL + diagnosis on the test fleet
    te_rows = preds["split"].eq("test") & preds["running"].eq(1)
    r = preds.loc[te_rows & preds["next_predictable"].astype(bool) & preds["ttf"].le(cfg.labels.rul_cap_h)]
    rul_metrics = {}
    for name, sub in (("le_168h", r[r["ttf"] <= 168]), ("le_720h", r)):
        if len(sub):
            rul_metrics[name] = {
                "n_rows": int(len(sub)),
                "mae_h": float(np.mean(np.abs(sub["rul_p50"] - sub["ttf"]))),
                "coverage_p10_p90": float(np.mean((sub["ttf"] >= sub["rul_p10"]) & (sub["ttf"] <= sub["rul_p90"]))),
                "median_interval_h": float(np.median(sub["rul_p90"] - sub["rul_p10"])),
            }
    metrics["test"]["rul"] = rul_metrics
    d = preds.loc[te_rows & preds["label"].eq(1) & preds["next_predictable"].astype(bool)]
    dx_metrics: dict = {"row_accuracy": float(np.mean(d["pred_mode"] == d["next_mode"])) if len(d) else None}
    prob_cols = [f"prob_{m}" for m in FAILURE_MODES]
    ev_pred = []
    fail_te = ev_te[(ev_te["event_type"] == "failure") & ev_te["predictable"]]
    for _, f in fail_te.iterrows():
        w = d[(d["asset_id"] == f["asset_id"]) & (d["hour"] < f["hour"])
              & (d["hour"] >= f["hour"] - cfg.labels.horizon_h)]
        if len(w):
            ev_pred.append((f["mode"], FAILURE_MODES[int(np.argmax(w[prob_cols].mean().to_numpy()))]))
    if ev_pred:
        labels = list(FAILURE_MODES)
        cm = pd.crosstab(pd.Categorical([a for a, _ in ev_pred], labels),
                         pd.Categorical([b for _, b in ev_pred], labels), dropna=False)
        dx_metrics.update({
            "event_accuracy": float(np.mean([a == b for a, b in ev_pred])),
            "n_events": len(ev_pred),
            "confusion": {"labels": labels, "matrix": cm.to_numpy().tolist()},
        })
    metrics["test"]["diagnosis"] = dx_metrics
    t = step("evaluate", t)

    # 5. explainability -------------------------------------------------------------------------
    imp = pd.DataFrame(columns=["feature", "importance", "std", "signal"])
    if importance:
        log("[7/7] Permutation importance on the test fleet...")
        sub = yte.sample(n=min(8000, len(yte)), random_state=cfg.train.random_state)
        Xs = F.loc[sub.index, FEATURE_COLUMNS]
        pi = permutation_importance(_GBMEnsemble(tm.risk), Xs, sub["label"].to_numpy(), scoring=_ap_scorer,
                                    n_repeats=2, random_state=cfg.train.random_state)
        imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": pi.importances_mean,
                            "std": pi.importances_std})
        imp["signal"] = imp["feature"].map(signal_of)
        imp = imp.sort_values("importance", ascending=False)
        t = step("importance", t)
    else:
        log("[7/7] Skipping permutation importance.")

    # 6. save -----------------------------------------------------------------------------------
    bundle.meta = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_features": len(FEATURE_COLUMNS),
        "train_assets": train_ids,
        "test_assets": test_ids,
    }
    save_bundle(bundle, models_dir)
    # float32 halves the file, but the two calibrated probabilities stay float64: their isotonic
    # tie-break is ~1e-9 apart, below float32 resolution, and dropping it would re-create ties that
    # change ROC / PR curves recomputed from this file (the dashboard does exactly that).
    float_cols = preds.select_dtypes(include=["float64"]).columns.difference(FULL_PRECISION_COLUMNS)
    preds[float_cols] = preds[float_cols].astype(np.float32)
    preds.to_parquet(artifacts_dir / "predictions.parquet", index=False)
    pd.concat(all_alerts, ignore_index=True).to_csv(artifacts_dir / "alerts.csv", index=False)
    pd.concat(outcomes, ignore_index=True).to_csv(artifacts_dir / "failure_outcomes.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(artifacts_dir / "threshold_curve.csv", index=False)
    imp.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    metrics.update({
        "generated_at": bundle.meta["trained_at"],
        "version": __version__,
        "methods": METHODS,
        "score_columns": SCORE_COLUMNS,
        "thresholds": thresholds,
        "data": {
            "n_assets": cfg.sim.n_assets, "hours": cfg.sim.hours,
            "n_train_assets": len(train_ids), "n_test_assets": len(test_ids),
            "rows": int(len(raw)), "failures": int(len(fails)),
            "failures_by_mode": fails.groupby("mode").size().to_dict(),
            "failures_test": n_fail_te, "failures_train": int((ev_tr["event_type"] == "failure").sum()),
            "pm_events": int((events["event_type"] == "pm").sum()),
            "n_features": len(FEATURE_COLUMNS),
        },
        "config": cfg.to_dict(),
        "timings_s": timings,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    })
    metrics["timings_s"]["total"] = round(time.perf_counter() - t_start, 1)
    (artifacts_dir / "metrics.json").write_text(json.dumps(_clean_nans(metrics), indent=2), encoding="utf-8")
    log(f"Pipeline finished in {metrics['timings_s']['total']:.0f}s. Artifacts in {artifacts_dir}")
    return metrics
