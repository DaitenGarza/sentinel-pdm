"""Load the trained model bundle and score sensor data with it.

The same code path scores the held-out test fleet during training, new data uploaded in the
dashboard, and the hour-by-hour live replay: raw readings -> cleaning -> features -> models.

Output columns (one row per pump-hour; NaN while the pump is stopped):
  risk            Sentinel hybrid P(failure within 7 days)        <- drives alerts and status
  risk_gbm        gradient-boosting-only P(failure within 7 days)
  rul_p10/50/90   remaining useful life quantiles, hours (meaningful when an alert is active)
  prob_<mode>     failure-mode probabilities; pred_mode = most likely mode
  mspc_score      MSPC anomaly score (1.0 = 99 % control limit)
  iforest_score   Isolation Forest anomaly score (1.0 = 99th percentile of healthy)
  rule_score      conventional alarm limits (>= 1.0 = alarm)
  z_<signal>      robust z-score of each signal's deviation from its healthy baseline
  status          Normal / Watch / Alert / Critical / Offline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FAILURE_MODES, MODELS_DIR
from .explain import driver_zscores
from .features import build_features
from .models import DiagnosisModel, HybridModel, RiskModel, RULModel, ScoreEnsemble, calibrate, rule_score
from .quality import clean_readings

BUNDLE_FILE = "bundle.joblib"
SCORE_COLUMNS = ["risk", "risk_gbm", "rul_p10", "rul_p50", "rul_p90",
                 *[f"prob_{m}" for m in FAILURE_MODES], "mspc_score", "iforest_score"]


@dataclass
class ModelBundle:
    risk: RiskModel
    rul: RULModel
    diagnosis: DiagnosisModel
    mspc: ScoreEnsemble
    iforest: ScoreEnsemble
    hybrid: HybridModel
    thresholds: dict[str, float]
    ref_stats: pd.DataFrame
    config: dict
    meta: dict = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / BUNDLE_FILE
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(models_dir: Path = MODELS_DIR) -> ModelBundle:
    path = models_dir / BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"No trained models at {path}. Run `python -m sentinel_pdm train` first.")
    return joblib.load(path)


def model_scores(X: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """All model outputs for feature rows of *running* pumps."""
    out = pd.DataFrame(index=X.index)
    if len(X) == 0:
        return out.reindex(columns=SCORE_COLUMNS)
    raw = bundle.risk.raw_score(X)
    mspc = bundle.mspc.score(X)
    out["risk"] = bundle.hybrid.predict_proba(raw, mspc)
    out["risk_gbm"] = calibrate(bundle.risk.calibrator, raw)
    out[["rul_p10", "rul_p50", "rul_p90"]] = bundle.rul.predict(X).to_numpy()
    out[[f"prob_{m}" for m in FAILURE_MODES]] = bundle.diagnosis.predict_proba(X).to_numpy()
    out["mspc_score"] = mspc
    out["iforest_score"] = bundle.iforest.score(X)
    return out


def assemble_predictions(
    features: pd.DataFrame, clean: pd.DataFrame, scores: pd.DataFrame, ref_stats: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Full-length prediction frame: model scores on running rows, NaN elsewhere, plus drivers and status."""
    preds = pd.DataFrame(index=features.index, columns=SCORE_COLUMNS, dtype=np.float64)
    preds.loc[scores.index, SCORE_COLUMNS] = scores[SCORE_COLUMNS].to_numpy(dtype=np.float64)
    running = features["running"].to_numpy()
    preds["rule_score"] = np.where(running == 1, rule_score(clean), np.nan)
    preds = pd.concat([preds, driver_zscores(features, ref_stats)], axis=1)
    return finalize_predictions(preds, running, threshold)


def score_features(features: pd.DataFrame, clean: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Score a feature frame (from ``build_features``)."""
    run = features["running"].to_numpy() == 1
    scores = model_scores(features.loc[run], bundle)
    return assemble_predictions(features, clean, scores, bundle.ref_stats, bundle.thresholds["hybrid"])


def finalize_predictions(preds: pd.DataFrame, running: np.ndarray, threshold: float) -> pd.DataFrame:
    """Add predicted failure mode and Normal/Watch/Alert/Critical/Offline status (vectorised).

    Critical = alert with median RUL <= 72 h; Alert = risk above the alert threshold;
    Watch = risk above half the threshold or MSPC beyond 1.5x its control limit.
    """
    prob_cols = [f"prob_{m}" for m in FAILURE_MODES]
    p = preds[prob_cols].to_numpy(dtype=np.float64)
    has = ~np.isnan(p).all(axis=1)
    mode = np.full(len(preds), "none", dtype=object)
    mode[has] = np.array(FAILURE_MODES, dtype=object)[np.nanargmax(np.where(has[:, None], p, 0), axis=1)[has]]
    preds["pred_mode"] = mode
    risk = preds["risk"].to_numpy(dtype=np.float64)
    rul50 = preds["rul_p50"].to_numpy(dtype=np.float64)
    mspc = preds["mspc_score"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        status = np.select(
            [running != 1, (risk >= threshold) & (rul50 <= 72), risk >= threshold,
             (risk >= threshold / 2) | (mspc >= 1.5)],
            ["Offline", "Critical", "Alert", "Watch"],
            default="Normal",
        )
    preds["status"] = status
    return preds


def score_readings(
    raw: pd.DataFrame, events: pd.DataFrame | None, bundle: ModelBundle
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full path for new data: clean -> features -> score.

    ``raw`` needs asset_id, hour, running, the sensor tags and the context columns
    (load_pct, speed_rpm, ambient_temp); ``timestamp`` is carried through when present.
    ``events`` is the maintenance log (may be empty / None).
    """
    raw = raw.sort_values(["asset_id", "hour"]).reset_index(drop=True)
    if events is None or len(events) == 0:
        events = pd.DataFrame(columns=["asset_id", "event_type", "hour", "end_hour", "mode"])
    clean, _ = clean_readings(raw)
    feats = build_features(clean, events)
    preds = score_features(feats, clean, bundle)
    preds.insert(0, "asset_id", clean["asset_id"].to_numpy())
    preds.insert(1, "hour", clean["hour"].to_numpy())
    if "timestamp" in clean:
        preds.insert(2, "timestamp", clean["timestamp"].to_numpy())
    return clean, preds
