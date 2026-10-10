"""Independent benchmark on a public dataset: UCI "AI4I 2020 Predictive Maintenance Dataset".

Why this module exists
----------------------
Everything else in Sentinel PdM is trained and evaluated on data from this project's own pump
simulator, so the obvious challenge from a reliability engineer is: *"does the approach only work
on data you generated yourself?"*. AI4I 2020 (Matzka, 2020) is an independent, widely cited public
benchmark from the UCI Machine Learning Repository (id 601, CC BY 4.0), built by someone else with
documented failure mechanisms. Note, to be precise: its author describes it as a *synthetic*
dataset "that reflects real predictive maintenance data encountered in industry" (real plant data
is rarely publishable). It is therefore not plant data, but it is outside data that this project
had no hand in, which is exactly what is needed to test the project's central claim:

    Encoding the physics of a failure mechanism as a feature beats throwing a bigger model at raw tags.

Dataset (10,000 rows = machining process runs)
----------------------------------------------
``UDI`` row id, ``Product ID`` serial, ``Type`` product quality L/M/H, ``Air temperature [K]``,
``Process temperature [K]``, ``Rotational speed [rpm]``, ``Torque [Nm]``, ``Tool wear [min]``,
target ``Machine failure`` (339 = 3.39 %), and the failure-type flags ``TWF``, ``HDF``, ``PWF``,
``OSF``, ``RNF``. Published failure mechanisms (the documented rules reproduce the HDF, PWF and
OSF flags of the file exactly):

* **HDF** heat-dissipation failure: process-air temperature difference < 8.6 K *and* rotational
  speed < 1380 rpm (heat cannot be carried away at low speed with a small temperature gradient).
* **PWF** power failure: mechanical power = torque x angular speed < 3500 W or > 9000 W.
* **OSF** overstrain failure: tool wear x torque > 11,000 / 12,000 / 13,000 min*Nm for product
  quality L / M / H.
* **TWF** tool-wear failure: the tool is replaced or fails at a random wear between 200 and 240 min.
  Most runs inside that window do *not* fail, so a single snapshot can at best say "the tool is in
  its end-of-life window"; TWF is largely unpredictable row by row.
* **RNF** random failure: 0.1 % chance per run regardless of the process; unpredictable by design.

Known quirks of the published file (reported in the metrics, not "fixed"): 9 failures carry no
type flag, and 18 of the 19 RNF flags sit on rows whose ``Machine failure`` is 0.

Protocol
--------
* **Target** ``Machine failure``. The inputs never include ``TWF/HDF/PWF/OSF/RNF`` (they are the
  failure *types*, i.e. the answer), ``UDI`` (row number) or ``Product ID`` (serial number):
  ``LEAKAGE_COLUMNS`` lists them and :func:`build_inputs` asserts they cannot reach a model.
* **Two feature sets** - ``raw``: the five measured variables + product quality (one-hot);
  ``physics``: the same + temperature rise, mechanical power and overstrain (:func:`physics_features`).
  Both include product quality, so the comparison isolates the three engineered features.
  The features encode *which physical quantity matters*, never the published *thresholds*;
  the models still have to learn where failure starts. (Hard-coding 8.6 K or 9000 W would simply
  re-implement the data generator.)
* **Three models** of increasing capacity: scaled logistic regression, random forest and
  histogram gradient boosting.
* **Repeated stratified 5-fold CV** (3 repeats, fixed seed). Every model and feature set is
  evaluated on the *same* folds, so differences are paired. Each repeat gives one out-of-fold (OOF)
  probability per row; metrics are computed per repeat on the full OOF vector and reported as
  mean and standard deviation over repeats.
* **Metrics** ROC-AUC; PR-AUC (average precision; the honest metric at 3.4 % prevalence);
  best-F1 (the maximum over thresholds, an optimistic upper bound) and cross-fitted F1 (each fold
  is scored with the best-F1 threshold of the *other* folds, i.e. what a deployed threshold would
  achieve); recall at precision >= 90 % (if a model never reaches 90 %, the recall at its best
  achievable precision is reported and flagged); confusion counts and per-failure-type recall
  at the best-F1 threshold for the best model of each feature set.
* **Permutation importance** for the best model of each feature set: PR-AUC drop on held-out folds
  when one input is shuffled. The three product-quality columns are shuffled *together* as one
  input (shuffling one-hot columns separately creates impossible rows such as "L and H").
* **Blocked-CV sanity check.** Rows are a time series (tool wear accumulates, temperatures follow
  a random walk), so random folds put near-identical neighbours in train and test. The best model
  of each feature set is re-scored with contiguous 200-row blocks kept together; a similar score
  shows the result does not rely on neighbour leakage.
* **Mechanism ceiling.** The share of failures explained by a deterministic snapshot mechanism
  (HDF, PWF or OSF). No snapshot model can be expected to catch much more than this.

Artifacts (``artifacts/``)
--------------------------
``ai4i_metrics.json``::

    { "dataset":   {name, uci_id, url, download_url, doi, citation, paper, paper_doi, license, synthetic_note,
                    n_rows, n_failures, failure_rate, failure_types: {TWF..RNF: n flags},
                    failure_types_on_failures: {TWF..RNF, none: n}, quirks: {...}, sha256},
      "protocol":  {target, cv, n_splits, n_repeats, random_state, precision_target, excluded_columns,
                    block_rows, n_rows_used},
      "feature_sets": {"raw"|"physics": {label, features: [...]}},
      "models":    {"logreg"|"rf"|"hgb": label},
      "results":   {"raw"|"physics": {model: {roc_auc, pr_auc, f1_best, f1_xfit, recall_at_p90,
                    precision_at_p90, threshold_best (each with a matching *_std), p90_reached (bool),
                    fit_time_s}}},
      "best":      {"raw": model, "physics": model, "overall": {feature_set, model}},
      "operating_point": {"raw"|"physics": {model, threshold, f1, precision, recall,
                    confusion: {tp, fp, fn, tn} (mean over repeats), false_alarms_per_10k}},
      "failure_type_recall": {"raw"|"physics": {model, types: {TWF|HDF|PWF|OSF|RNF|none:
                    {n, recall, recall_std, predictable}}}},
      "mechanism_ceiling": {failures, deterministic_failures, recall},
      "blocked_cv": {"raw"|"physics": {model, roc_auc, pr_auc, pr_auc_random_cv}},
      "runtime_s", "generated_at", "version", "environment" }

``ai4i_importance.csv``: ``feature_set, model, feature, label, kind (measured|quality|engineered),
importance (mean PR-AUC drop), std, rank``.

``ai4i_oof.csv`` (one row per run, for plotting): ``udi, type, air_temp_K, process_temp_K,
speed_rpm, torque_Nm, tool_wear_min, temp_diff_K, power_W, overstrain_minNm, machine_failure,
TWF, HDF, PWF, OSF, RNF, prob_raw, prob_physics`` where ``prob_*`` is the out-of-fold failure
probability of the best model of that feature set in the first CV repeat (each value comes from a
model that never saw the row). One repeat rather than the average over repeats keeps curves drawn
from this file consistent with the per-repeat metrics; averaging would act as a small ensemble and
score slightly higher than the table.

Run: ``python -m sentinel_pdm ai4i`` (downloads once to ``data/external/``) or ``--offline``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import ARTIFACTS_DIR, DATA_DIR

# --------------------------------------------------------------------------------------------
# dataset facts
# --------------------------------------------------------------------------------------------
DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"
UCI_ID = 601
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
DATASET_DOI = "10.24432/C5HS5C"
CITATION = ("Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C")
PAPER = ("S. Matzka, \"Explainable Artificial Intelligence for Predictive Maintenance Applications\", "
         "Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74.")
PAPER_DOI = "10.1109/AI4I49448.2020.00023"
LICENSE = "CC BY 4.0"
SYNTHETIC_NOTE = ("Synthetic dataset published by an independent author to reflect real predictive-maintenance "
                  "data encountered in industry; not generated by this project.")
DEFAULT_PATH = DATA_DIR / "external" / "ai4i2020.csv"
N_ROWS = 10_000

TARGET = "Machine failure"
FAILURE_TYPES: tuple[str, ...] = ("TWF", "HDF", "PWF", "OSF", "RNF")
FAILURE_TYPE_LABELS = {
    "TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power", "OSF": "Overstrain", "RNF": "Random",
    "none": "No type recorded",
}
# Which failure types a model can, in principle, recognise from one snapshot of the process.
PREDICTABLE_TYPES = {"TWF": False, "HDF": True, "PWF": True, "OSF": True, "RNF": False, "none": False}
# Identifiers and failure-type flags. Using any of them as an input would leak the answer.
LEAKAGE_COLUMNS: tuple[str, ...] = ("UDI", "Product ID", *FAILURE_TYPES, TARGET)

# Measured process variables: file column -> short model-input name.
SENSOR_COLUMNS = {
    "Air temperature [K]": "air_temp_K",
    "Process temperature [K]": "process_temp_K",
    "Rotational speed [rpm]": "speed_rpm",
    "Torque [Nm]": "torque_Nm",
    "Tool wear [min]": "tool_wear_min",
}
EXPECTED_COLUMNS: tuple[str, ...] = ("UDI", "Product ID", "Type", *SENSOR_COLUMNS, TARGET, *FAILURE_TYPES)
QUALITY_LEVELS: tuple[str, ...] = ("L", "M", "H")
QUALITY_COLUMNS = [f"type_{q}" for q in QUALITY_LEVELS]
ENGINEERED_COLUMNS = ["temp_diff_K", "power_W", "overstrain_minNm"]

FEATURE_SETS: dict[str, list[str]] = {
    "raw": [*SENSOR_COLUMNS.values(), *QUALITY_COLUMNS],
    "physics": [*SENSOR_COLUMNS.values(), *QUALITY_COLUMNS, *ENGINEERED_COLUMNS],
}
FEATURE_SET_LABELS = {"raw": "Raw sensors", "physics": "Physics-informed"}
MODEL_LABELS = {"logreg": "Logistic regression", "rf": "Random forest", "hgb": "Gradient boosting"}

# Display metadata per model input (one-hot quality columns are reported as one input, "type").
FEATURE_INFO: dict[str, dict[str, str]] = {
    "air_temp_K": {"label": "Air temperature", "unit": "K", "kind": "measured"},
    "process_temp_K": {"label": "Process temperature", "unit": "K", "kind": "measured"},
    "speed_rpm": {"label": "Rotational speed", "unit": "rpm", "kind": "measured"},
    "torque_Nm": {"label": "Torque", "unit": "N·m", "kind": "measured"},
    "tool_wear_min": {"label": "Tool wear", "unit": "min", "kind": "measured"},
    "type": {"label": "Product quality (L/M/H)", "unit": "", "kind": "quality"},
    "temp_diff_K": {"label": "Temperature rise ΔT", "unit": "K", "kind": "engineered"},
    "power_W": {"label": "Mechanical power", "unit": "W", "kind": "engineered"},
    "overstrain_minNm": {"label": "Overstrain (wear × torque)", "unit": "min·N·m", "kind": "engineered"},
}

PRECISION_TARGET = 0.9
RANDOM_STATE = 42
BLOCK_ROWS = 200

METRICS_FILE = "ai4i_metrics.json"
IMPORTANCE_FILE = "ai4i_importance.csv"
OOF_FILE = "ai4i_oof.csv"


def artifact_paths(artifacts_dir: Path | None = None) -> dict[str, Path]:
    """Paths of the three benchmark artifacts (``metrics``, ``importance``, ``oof``)."""
    d = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    return {"metrics": d / METRICS_FILE, "importance": d / IMPORTANCE_FILE, "oof": d / OOF_FILE}


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def validate_ai4i(df: pd.DataFrame, expected_rows: int | None = N_ROWS) -> None:
    """Raise ``ValueError`` unless ``df`` has the published AI4I schema.

    Checked: every expected column, row count (``None`` skips it), no missing values, product
    quality in {L, M, H} and 0/1 labels. A truncated download or a different file fails loudly
    here instead of producing a quietly wrong benchmark.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"AI4I data is missing columns {missing}; got {list(df.columns)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"AI4I data should have {expected_rows:,} rows, got {len(df):,}")
    if df[list(EXPECTED_COLUMNS)].isna().any().any():
        bad = df[list(EXPECTED_COLUMNS)].isna().sum()
        raise ValueError(f"AI4I data has missing values: {bad[bad > 0].to_dict()}")
    unknown = sorted(set(df["Type"].astype(str)) - set(QUALITY_LEVELS))
    if unknown:
        raise ValueError(f"Unknown product quality codes in 'Type': {unknown}")
    for c in (TARGET, *FAILURE_TYPES):
        if not set(pd.unique(df[c])) <= {0, 1}:
            raise ValueError(f"Column {c!r} must be 0/1")


def _download_csv(url: str = DOWNLOAD_URL, timeout: float = 60.0) -> bytes:
    """Fetch the UCI zip and return the bytes of ``ai4i2020.csv`` (extracted in memory)."""
    req = urllib.request.Request(url, headers={"User-Agent": f"sentinel-pdm/{__version__} (python urllib)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith("ai4i2020.csv")]
            if not names:
                raise ValueError(f"ai4i2020.csv not found in the downloaded archive (members: {zf.namelist()})")
            return zf.read(names[0])
    except zipfile.BadZipFile:
        # The repository has also served the bare CSV in the past; accept it if it parses.
        if payload[:3] == b"UDI":
            return payload
        raise ValueError("Downloaded file is neither a zip archive nor the AI4I CSV") from None


def _offline_message(path: Path, reason: str) -> str:
    return (f"{reason}\nDownload it manually from {DATASET_PAGE} (or {DOWNLOAD_URL}), extract ai4i2020.csv "
            f"to {path}, then run: python -m sentinel_pdm ai4i --offline")


def load_ai4i(download: bool = True, path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the AI4I 2020 CSV, downloading it once into ``path`` (``data/external/``) if needed.

    The CSV is validated before it is cached (written atomically), so a broken download never
    poisons the cache; a cached file that fails validation is re-downloaded when ``download`` is
    allowed. Without network access and without a cached copy, the error says how to fetch it
    manually.
    """
    path = Path(path)
    if path.exists():
        try:
            df = pd.read_csv(path)
            validate_ai4i(df)
            return df
        except ValueError as exc:  # pandas parser / decoding errors are ValueErrors too
            if not download:
                raise ValueError(_offline_message(path, f"Cached AI4I file {path} is invalid: {exc}")) from exc
    elif not download:
        raise FileNotFoundError(_offline_message(path, f"AI4I data not found at {path} (offline mode)."))

    try:
        raw = _download_csv()
    except OSError as exc:  # URLError, HTTPError and socket timeouts are all OSErrors
        raise ConnectionError(_offline_message(path, f"Could not download the AI4I dataset: {exc}")) from exc
    df = pd.read_csv(io.BytesIO(raw))
    validate_ai4i(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    return df


# --------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------
def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-informed inputs that mirror the published failure mechanisms.

    * ``temp_diff_K`` = process - air temperature (K). Heat leaves the process through this
      gradient; HDF occurs when it is below 8.6 K while speed is below 1380 rpm.
    * ``power_W`` = torque x rotational speed x 2*pi/60 (W), the mechanical power drawn by the
      process. PWF occurs below 3500 W (stall-like operation) or above 9000 W (overload). Neither raw
      torque nor raw speed alone separates these: they lie on hyperbolas in the torque-speed plane.
    * ``overstrain_minNm`` = tool wear x torque (min*N*m), a proxy for accumulated load on a worn
      tool. OSF occurs above 11,000 / 12,000 / 13,000 min*N*m for L / M / H quality products,
      so the product-quality one-hot (``type_L/M/H``) is returned alongside it.

    TWF (tool replaced or failed at a random wear of 200-240 min) and RNF (0.1 % random) have no
    snapshot signature beyond tool wear itself, so no feature is engineered for them.

    Only physical quantities are computed here, never the published thresholds; the models must
    learn the failure boundaries from data.
    """
    air = df["Air temperature [K]"].to_numpy(dtype=float)
    proc = df["Process temperature [K]"].to_numpy(dtype=float)
    rpm = df["Rotational speed [rpm]"].to_numpy(dtype=float)
    torque = df["Torque [Nm]"].to_numpy(dtype=float)
    wear = df["Tool wear [min]"].to_numpy(dtype=float)
    quality = df["Type"].astype(str).to_numpy()
    out = pd.DataFrame({
        "temp_diff_K": proc - air,
        "power_W": torque * rpm * 2.0 * np.pi / 60.0,
        "overstrain_minNm": wear * torque,
    }, index=df.index)
    for q, col in zip(QUALITY_LEVELS, QUALITY_COLUMNS):
        out[col] = (quality == q).astype(np.int8)
    return out


def build_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Every candidate model input: renamed measured variables + :func:`physics_features`.

    Built from an explicit allow-list of source columns, so identifiers and failure-type flags
    (``LEAKAGE_COLUMNS``) cannot reach a model even if the source frame is re-ordered or extended.
    """
    X = df[list(SENSOR_COLUMNS)].rename(columns=SENSOR_COLUMNS).astype(float)
    X = pd.concat([X, physics_features(df)], axis=1)
    leaked = set(X.columns) & set(LEAKAGE_COLUMNS)
    if leaked:  # pragma: no cover - guarded by construction; kept as a hard stop
        raise AssertionError(f"label leakage: {sorted(leaked)} must never be model inputs")
    return X[FEATURE_SETS["physics"]]


def _input_groups(features: list[str]) -> dict[str, list[int]]:
    """Map each reported input to its column indices; the quality one-hot is a single input."""
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(features):
        groups.setdefault("type" if f in QUALITY_COLUMNS else f, []).append(i)
    return groups


# --------------------------------------------------------------------------------------------
# models and metrics
# --------------------------------------------------------------------------------------------
def make_models(random_state: int = RANDOM_STATE, quick: bool = False) -> dict[str, ClassifierMixin]:
    """The three model families, from a linear baseline to boosted trees.

    Settings are standard and were not tuned on this dataset: the point is the feature-set
    comparison, and tuning on the evaluation folds would bias it.
    """
    n_trees = 60 if quick else 300
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "rf": RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=1, max_features="sqrt",
                                     n_jobs=-1, random_state=random_state),
        "hgb": HistGradientBoostingClassifier(max_iter=n_trees, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              early_stopping=False, random_state=random_state),
    }


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Maximum F1 over all thresholds and a threshold achieving it (predict failure if p >= t).

    Every threshold between the best cut-off score and the next lower score gives the same F1 on
    these rows; the midpoint is returned rather than the cut-off score itself so that, applied to
    new rows, it leaves a margin on both sides instead of sitting exactly on the weakest positive.
    """
    if y.sum() == 0:
        return 0.0, 0.5
    precision, recall, thr = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # the last point (recall 0) has no threshold
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    t = (thr[i - 1] + thr[i]) / 2 if i > 0 else thr[i]
    return float(f1[i]), float(t)


def f1_at(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum(pred & (y == 1)))
    denom = float(pred.sum() + (y == 1).sum())
    return 2 * tp / denom if denom else 0.0


def recall_at_precision(y: np.ndarray, p: np.ndarray, target: float = PRECISION_TARGET) -> tuple[float, float, bool]:
    """Highest recall with precision >= ``target``: (recall, precision, target_reached).

    A maintenance planner thinks in "of the alerts I act on, at least 9 in 10 must be real";
    this is the share of failures caught under that constraint. If the model never reaches the
    target, the recall at its best achievable precision is returned with ``target_reached=False``.
    """
    precision, recall, _ = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= target
    if ok.any():
        i = int(np.flatnonzero(ok)[np.argmax(recall[ok])])
        return float(recall[i]), float(precision[i]), True
    best = precision.max()
    i = int(np.flatnonzero(precision == best)[np.argmax(recall[precision == best])])
    return float(recall[i]), float(precision[i]), False


def _xfit_f1(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> float:
    """F1 when each fold is classified with the best-F1 threshold of the *other* folds."""
    pred = np.zeros(len(y), dtype=bool)
    for k in np.unique(fold):
        test = fold == k
        _, t = best_f1(y[~test], p[~test])
        pred[test] = p[test] >= t
    return f1_at(y, pred)


def _repeat_metrics(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> dict:
    f1, thr = best_f1(y, p)
    r90, p90, reached = recall_at_precision(y, p)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "f1_best": f1, "threshold_best": thr, "f1_xfit": _xfit_f1(y, p, fold),
            "recall_at_p90": r90, "precision_at_p90": p90, "p90_reached": reached}


def _summarise(per_repeat: list[dict]) -> dict:
    """Mean (and ``*_std``) over repeats; booleans are True only if true in every repeat."""
    out: dict = {}
    for k in per_repeat[0]:
        vals = [r[k] for r in per_repeat]
        if isinstance(vals[0], bool):
            out[k] = bool(all(vals))
        else:
            out[k] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


def _cv_oof(model: ClassifierMixin, X: np.ndarray, y: np.ndarray,
            splits: list[tuple[np.ndarray, np.ndarray]], n_splits: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Out-of-fold probabilities for each repeat: (oof[n_repeats, n], fold_id[n_repeats, n], fit seconds)."""
    n_repeats = len(splits) // n_splits
    oof = np.full((n_repeats, len(y)), np.nan)
    fold = np.full((n_repeats, len(y)), -1, dtype=int)
    t0 = time.perf_counter()
    for i, (tr, te) in enumerate(splits):
        r = i // n_splits
        m = clone(model).fit(X[tr], y[tr])
        oof[r, te] = m.predict_proba(X[te])[:, 1]
        fold[r, te] = i % n_splits
    return oof, fold, time.perf_counter() - t0


def grouped_permutation_importance(model: ClassifierMixin, X: np.ndarray, y: np.ndarray,
                                   groups: dict[str, list[int]], n_repeats: int,
                                   rng: np.random.Generator) -> dict[str, list[float]]:
    """PR-AUC drop when each input group is shuffled (columns of one group share a permutation).

    sklearn's ``permutation_importance`` shuffles columns one at a time, which for one-hot product
    quality creates impossible rows ("both L and H") and understates the input. Shuffling the
    group jointly keeps every row physically valid.
    """
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    drops: dict[str, list[float]] = {}
    for name, cols in groups.items():
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[np.ix_(rng.permutation(len(X)), cols)]
            drops.setdefault(name, []).append(float(base - average_precision_score(y, model.predict_proba(Xp)[:, 1])))
    return drops


def _type_masks(df: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    """Failures (target = 1) of each type; ``none`` = failures without any type flag."""
    flags = df[list(FAILURE_TYPES)].to_numpy()
    masks = {t: (flags[:, i] == 1) & (y == 1) for i, t in enumerate(FAILURE_TYPES)}
    masks["none"] = (flags.sum(axis=1) == 0) & (y == 1)
    return masks


def _jsonable(obj):
    """Make numpy scalars JSON-safe and turn NaN/inf into null (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


# --------------------------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------------------------
def run_ai4i_benchmark(
    download: bool = True,
    log: Callable[[str], None] = print,
    *,
    data: pd.DataFrame | None = None,
    artifacts_dir: Path | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    quick: bool = False,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run the raw-vs-physics benchmark and write the three ``ai4i_*`` artifacts.

    ``data`` replaces the downloaded file (any number of rows in the AI4I schema; used by the
    tests), ``quick`` uses smaller ensembles and fewer permutation repeats. Returns the metrics dict
    that is also written to ``ai4i_metrics.json`` (schema in the module docstring).
    """
    t_start = time.perf_counter()
    paths = artifact_paths(artifacts_dir)
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)

    if data is None:
        log(f"[1/5] Loading the UCI AI4I 2020 dataset ({'download if needed' if download else 'offline'})...")
        df = load_ai4i(download=download)
        sha = hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest()
    else:
        log("[1/5] Using the supplied AI4I-format data...")
        validate_ai4i(data, expected_rows=None)
        df, sha = data.reset_index(drop=True), None
    y = df[TARGET].to_numpy().astype(int)
    if y.sum() < n_splits or (y == 0).sum() < n_splits:
        raise ValueError(f"Need at least {n_splits} failures and non-failures for {n_splits}-fold CV")
    inputs = build_inputs(df)
    masks = _type_masks(df, y)
    log(f"  {len(df):,} runs | {int(y.sum())} failures ({y.mean():.2%}) | "
        + " ".join(f"{t} {int(df[t].sum())}" for t in FAILURE_TYPES))

    # 1. repeated CV: every model x feature set on the same folds ---------------------------------
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    splits = list(cv.split(np.zeros(len(y)), y))
    models = make_models(random_state, quick)
    log(f"[2/5] Cross-validating {len(models)} models x {len(FEATURE_SETS)} feature sets "
        f"({n_splits}-fold x {n_repeats} repeats = {len(splits) * len(models) * len(FEATURE_SETS)} fits)...")
    results: dict[str, dict[str, dict]] = {}
    oof_store: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for fs, cols in FEATURE_SETS.items():
        X = inputs[cols].to_numpy(dtype=float, copy=True)
        results[fs] = {}
        for key, model in models.items():
            oof, fold, secs = _cv_oof(model, X, y, splits, n_splits)
            res = _summarise([_repeat_metrics(y, oof[r], fold[r]) for r in range(n_repeats)])
            res["fit_time_s"] = round(secs, 2)
            results[fs][key] = res
            oof_store[(fs, key)] = (oof, fold)
            log(f"  {FEATURE_SET_LABELS[fs]:17s} {MODEL_LABELS[key]:20s} PR-AUC {res['pr_auc']:.3f} "
                f"ROC-AUC {res['roc_auc']:.3f}  ({secs:.1f}s)")

    best = {fs: max(results[fs], key=lambda k: results[fs][k]["pr_auc"]) for fs in FEATURE_SETS}
    overall = max(FEATURE_SETS, key=lambda fs: results[fs][best[fs]]["pr_auc"])
    best_info = {**best, "overall": {"feature_set": overall, "model": best[overall]}}

    # 2. operating point + failure-type recall of each feature set's best model ------------------
    log("[3/5] Operating point and failure-type recall of the best models...")
    operating, type_recall = {}, {}
    for fs, key in best.items():
        oof, _ = oof_store[(fs, key)]
        conf, type_r, ops = [], {t: [] for t in masks}, []
        for r in range(n_repeats):
            f1, thr = best_f1(y, oof[r])
            pred = oof[r] >= thr
            tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
            fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
            conf.append((tp, fp, fn, tn))
            ops.append((thr, f1, tp / max(tp + fp, 1), tp / max(tp + fn, 1)))
            for t, m in masks.items():
                if m.any():
                    type_r[t].append(float(pred[m].mean()))
        c = np.mean(conf, axis=0)
        o = np.mean(ops, axis=0)
        operating[fs] = {"model": key, "threshold": o[0], "f1": o[1], "precision": o[2], "recall": o[3],
                         "confusion": dict(zip(("tp", "fp", "fn", "tn"), c.tolist())),
                         "false_alarms_per_10k": float(c[1] / len(y) * 10_000)}
        type_recall[fs] = {"model": key, "types": {
            t: {"n": int(masks[t].sum()),
                "recall": float(np.mean(v)) if v else None,
                "recall_std": float(np.std(v)) if v else None,
                "predictable": PREDICTABLE_TYPES[t]}
            for t, v in type_r.items()}}

    # 3. blocked CV sanity check -----------------------------------------------------------------
    log(f"[4/5] Blocked-CV check (contiguous {BLOCK_ROWS}-row blocks) and permutation importance...")
    # small (test) datasets get smaller blocks so there are always >= 2 blocks per fold
    block_rows = max(min(BLOCK_ROWS, len(y) // (2 * n_splits)), 1)
    blocks = np.arange(len(y)) // block_rows
    bsplits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                   .split(np.zeros(len(y)), y, groups=blocks))
    blocked = {}
    for fs, key in best.items():
        X = inputs[FEATURE_SETS[fs]].to_numpy(dtype=float, copy=True)
        boof, _, _ = _cv_oof(models[key], X, y, bsplits, n_splits)
        blocked[fs] = {"model": key, "roc_auc": float(roc_auc_score(y, boof[0])),
                       "pr_auc": float(average_precision_score(y, boof[0])),
                       "pr_auc_random_cv": results[fs][key]["pr_auc"]}

    # 4. permutation importance on the held-out folds of the first repeat -------------------------
    rng = np.random.default_rng(random_state)
    pi_repeats = 2 if quick else 5
    imp_rows = []
    for fs, key in best.items():
        cols = FEATURE_SETS[fs]
        X = inputs[cols].to_numpy(dtype=float, copy=True)
        groups = _input_groups(cols)
        drops: dict[str, list[float]] = {g: [] for g in groups}
        for tr, te in splits[:n_splits]:
            m = clone(models[key]).fit(X[tr], y[tr])
            for g, d in grouped_permutation_importance(m, X[te], y[te], groups, pi_repeats, rng).items():
                drops[g].extend(d)
        for g, d in drops.items():
            info = FEATURE_INFO[g]
            imp_rows.append({"feature_set": fs, "model": key, "feature": g, "label": info["label"],
                             "kind": info["kind"], "importance": float(np.mean(d)), "std": float(np.std(d))})
    imp = pd.DataFrame(imp_rows).sort_values(["feature_set", "importance"], ascending=[True, False])
    imp["rank"] = imp.groupby("feature_set").cumcount() + 1

    # 5. artifacts ---------------------------------------------------------------------------------
    log("[5/5] Writing artifacts...")
    oof_df = pd.DataFrame({"udi": df["UDI"].to_numpy(), "type": df["Type"].astype(str).to_numpy()})
    for c in [*SENSOR_COLUMNS.values(), *ENGINEERED_COLUMNS]:
        oof_df[c] = inputs[c].to_numpy()
    oof_df["machine_failure"] = y
    for t in FAILURE_TYPES:
        oof_df[t] = df[t].to_numpy().astype(int)
    for fs, key in best.items():
        oof_df[f"prob_{fs}"] = oof_store[(fs, key)][0][0]
    oof_df.round(6).to_csv(paths["oof"], index=False)
    imp.to_csv(paths["importance"], index=False)

    n_fail = int(y.sum())
    deterministic = int(np.sum((y == 1) & (df[["HDF", "PWF", "OSF"]].to_numpy().sum(axis=1) > 0)))
    type_counts_on_failures = {t: int(m.sum()) for t, m in masks.items()}
    metrics = {
        "dataset": {
            "name": DATASET_NAME, "uci_id": UCI_ID, "url": DATASET_PAGE, "download_url": DOWNLOAD_URL,
            "doi": DATASET_DOI, "citation": CITATION, "paper": PAPER, "paper_doi": PAPER_DOI, "license": LICENSE,
            "synthetic_note": SYNTHETIC_NOTE, "n_rows": int(len(df)), "n_failures": n_fail,
            "failure_rate": float(y.mean()),
            "failure_types": {t: int(df[t].sum()) for t in FAILURE_TYPES},
            "failure_types_on_failures": type_counts_on_failures,
            "quirks": {
                "failures_without_type": type_counts_on_failures["none"],
                "type_flags_without_failure": int(np.sum((y == 0) & (df[list(FAILURE_TYPES)].to_numpy().sum(axis=1) > 0))),
                "multi_type_failures": int(np.sum(df[list(FAILURE_TYPES)].to_numpy().sum(axis=1) > 1)),
            },
            "sha256": sha,
        },
        "protocol": {
            "target": TARGET, "cv": "repeated stratified k-fold (same folds for every model and feature set)",
            "n_splits": n_splits, "n_repeats": n_repeats, "random_state": random_state,
            "precision_target": PRECISION_TARGET, "excluded_columns": list(LEAKAGE_COLUMNS),
            "block_rows": block_rows,
            "n_rows_used": int(len(df)), "quick": quick,
        },
        "feature_sets": {fs: {"label": FEATURE_SET_LABELS[fs], "features": cols} for fs, cols in FEATURE_SETS.items()},
        "models": MODEL_LABELS,
        "results": results,
        "best": best_info,
        "operating_point": operating,
        "failure_type_recall": type_recall,
        "mechanism_ceiling": {"failures": n_fail, "deterministic_failures": deterministic,
                              "recall": deterministic / n_fail if n_fail else None},
        "blocked_cv": blocked,
        "runtime_s": round(time.perf_counter() - t_start, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    metrics = _jsonable(metrics)
    paths["metrics"].write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    _print_summary(metrics, log, paths["metrics"])
    return metrics


def _print_summary(m: dict, log: Callable[[str], None], metrics_path: Path) -> None:
    """Compact console table (ASCII only, safe on Windows code pages)."""
    d, p = m["dataset"], m["protocol"]
    log(f"\nAI4I 2020 benchmark: {d['n_rows']:,} runs, {d['n_failures']} failures ({d['failure_rate']:.2%}), "
        f"{p['n_splits']}-fold CV x {p['n_repeats']} repeats (mean over repeats)\n")
    log(f"{'Features':17s} {'Model':20s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'F1best':>7s} {'F1xfit':>7s} {'Rec@P90':>8s}")
    for fs, per_model in m["results"].items():
        for key, r in per_model.items():
            star = "*" if key == m["best"][fs] else " "
            rec = f"{r['recall_at_p90']:.3f}" + ("" if r["p90_reached"] else "~")
            log(f"{FEATURE_SET_LABELS[fs]:17s} {MODEL_LABELS[key] + star:20s} {r['roc_auc']:8.3f} {r['pr_auc']:7.3f} "
                f"{r['f1_best']:7.3f} {r['f1_xfit']:7.3f} {rec:>8s}")
    log("  * best model of the feature set   ~ precision 90% never reached (recall at best precision shown)")
    for fs in ("raw", "physics"):
        op, tr = m["operating_point"][fs], m["failure_type_recall"][fs]["types"]
        types = " ".join(f"{t} {v['recall']:.2f}" for t, v in tr.items() if v["recall"] is not None)
        log(f"{FEATURE_SET_LABELS[fs]} best ({MODEL_LABELS[op['model']]}) at best-F1 threshold: recall "
            f"{op['recall']:.1%}, precision {op['precision']:.1%}, {op['false_alarms_per_10k']:.1f} false alarms "
            f"per 10k runs | by type: {types}")
    ceil = m["mechanism_ceiling"]
    if ceil["recall"] is not None:
        log(f"Mechanism ceiling: {ceil['deterministic_failures']}/{ceil['failures']} failures "
            f"({ceil['recall']:.1%}) have a deterministic snapshot cause (HDF/PWF/OSF).")
    for fs, b in m["blocked_cv"].items():
        log(f"Blocked CV ({FEATURE_SET_LABELS[fs]}, {MODEL_LABELS[b['model']]}): PR-AUC {b['pr_auc']:.3f} "
            f"vs {b['pr_auc_random_cv']:.3f} with random folds")
    log(f"Finished in {m['runtime_s']:.0f}s. Results: {metrics_path}")
