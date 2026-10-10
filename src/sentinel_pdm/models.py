"""Models: failure risk, remaining useful life, failure-mode diagnosis and anomaly detection.

Architecture
------------
* **Gradient boosting (supervised)** - scikit-learn HistGradientBoosting models learn from the
  maintenance history: probability of failure within 7 days, remaining useful life (quantile
  regression) and which failure mode is developing. They handle missing sensor values natively
  and train in seconds on a CPU.
* **MSPC (unsupervised)** - multivariate statistical process control with PCA Hotelling T^2 and
  SPE (Q) statistics, the classic chemical-plant monitoring technique. It only needs examples
  of *healthy* operation and models the normal correlations between signals, which makes it
  very robust to operating-point changes.
* **Isolation Forest (unsupervised)** - a generic anomaly detector, kept as a comparison.
* **Sentinel hybrid** - the alert score: an alert needs *both* the supervised failure risk and
  the MSPC anomaly score to be high (minimum of their percentile ranks, then calibrated to a
  probability). Each family catches almost every developing failure, but their false alarms
  come from different causes, so requiring agreement removes most nuisance alerts.

Leakage control
---------------
Models are always evaluated on pumps they never saw. Inside the training fleet, GroupKFold
splits *by pump*; every model (supervised and unsupervised) is fitted per fold, and the
out-of-fold (OOF) scores are what the probability calibration, the conformal RUL intervals
and the alert thresholds are tuned on. The K fold models are kept as a **cross-validation
ensemble** and averaged to score new pumps, so production uses exactly the models whose
behaviour was tuned (refitting one model on all data would silently move the false-alarm rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import FAILURE_MODES, TrainConfig
from .features import FEATURE_COLUMNS, SIGNALS

RUL_QUANTILES = (0.1, 0.5, 0.9)
# Conformal calibration bins on the predicted median RUL (hours): intervals are widened
# separately for "imminent", "this week", "next two weeks" and "later" predictions.
RUL_BINS = (0.0, 72.0, 168.0, 336.0, np.inf)
ANOMALY_FEATURES = [f"{s}_dev" for s in SIGNALS]

# Conventional DCS alarm limits used as the "what the plant does today" baseline.
RULE_LIMITS = {"vib_rms": 2.8, "bearing_temp": 82.0, "motor_temp": 130.0}  # ISO 10816 zone C / RTD alarms


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _hgb_params(tc: TrainConfig) -> dict:
    return dict(
        max_iter=tc.max_iter, learning_rate=tc.learning_rate, max_leaf_nodes=tc.max_leaf_nodes,
        min_samples_leaf=tc.min_samples_leaf, max_features=tc.max_features,
        l2_regularization=tc.l2_regularization, early_stopping=False, random_state=tc.random_state,
    )


def _clf(tc: TrainConfig, **kw) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**_hgb_params(tc), **kw)


def _reg(tc: TrainConfig, q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **_hgb_params(tc))


def stride_mask(hours: np.ndarray, stride: int) -> np.ndarray:
    return (hours % stride) == 0


def oof_folds(groups: np.ndarray, n_splits: int):
    n_groups = len(np.unique(groups))
    return GroupKFold(n_splits=min(n_splits, n_groups)).split(np.zeros(len(groups)), groups=groups)


def calibrate(calibrator: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    """Isotonic calibration plus a negligible tie-break.

    Isotonic regression maps whole ranges of raw scores to the same probability; the tiny
    raw-score term keeps the ranking (and therefore ROC/PR and alert ordering) intact without
    visibly changing any probability. The blend (rather than add-then-clip) keeps the tie-break
    alive in the top plateau too, where the isotonic fit reaches exactly 1.0. Both callers pass
    scores in [0, 1] (GBM probabilities, hybrid percentiles), where the result is strictly
    increasing in ``raw`` and moves no probability by more than 1e-6.
    """
    raw = np.nan_to_num(np.asarray(raw, dtype=np.float64))
    return (1.0 - 1e-6) * calibrator.predict(raw) + 1e-6 * np.clip(raw, 0.0, 1.0)


def _fit_calibrator(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    ok = ~np.isnan(raw)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[ok], y[ok])


# --------------------------------------------------------------------------------------------
# supervised models (cross-validation ensembles)
# --------------------------------------------------------------------------------------------
@dataclass
class RiskModel:
    """P(failure within the horizon) from gradient boosting alone."""

    members: list[HistGradientBoostingClassifier]
    calibrator: IsotonicRegression
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.features]
        return np.mean([m.predict_proba(Xf)[:, 1] for m in self.members], axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return calibrate(self.calibrator, self.raw_score(X))


@dataclass
class RULModel:
    """Conditional remaining useful life: *given* that a failure is developing, how long?

    Three quantile regressors give P10 / P50 / P90 hours. Raw quantile regressors are usually
    over-confident, so the interval is widened with normalised conformalised quantile regression
    (CQR) fitted on out-of-fold predictions, separately per predicted-RUL band (Mondrian CQR).
    """

    members: list[dict[float, HistGradientBoostingRegressor]]
    cap_h: float
    conformal_q: list[float] = field(default_factory=lambda: [0.0] * (len(RUL_BINS) - 1))
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.features]
        raw = np.mean([np.column_stack([m[q].predict(Xf) for q in RUL_QUANTILES]) for m in self.members], axis=0)
        return np.sort(np.clip(raw, 0, self.cap_h), axis=1)  # independent quantiles can cross

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.adjust(self.predict_raw(X), X.index)

    def adjust(self, q: np.ndarray, index=None) -> pd.DataFrame:
        width = np.maximum(q[:, 2] - q[:, 0], 1.0)
        b = np.clip(np.digitize(q[:, 1], RUL_BINS[1:-1]), 0, len(self.conformal_q) - 1)
        qhat = np.asarray(self.conformal_q)[b]
        lo = np.clip(q[:, 0] - qhat * width, 0, self.cap_h)
        hi = np.clip(q[:, 2] + qhat * width, 0, self.cap_h)
        return pd.DataFrame({"rul_p10": lo, "rul_p50": q[:, 1], "rul_p90": hi}, index=index)


def conformal_offsets(q: np.ndarray, y: np.ndarray, coverage: float) -> list[float]:
    """Normalised CQR offsets per predicted-RUL band (finite-sample corrected quantile)."""
    width = np.maximum(q[:, 2] - q[:, 0], 1.0)
    scores = np.maximum(q[:, 0] - y, y - q[:, 2]) / width
    b = np.clip(np.digitize(q[:, 1], RUL_BINS[1:-1]), 0, len(RUL_BINS) - 2)
    overall = _cqr_quantile(scores, coverage)
    out = []
    for k in range(len(RUL_BINS) - 1):
        sk = scores[b == k]
        out.append(_cqr_quantile(sk, coverage) if sk.size >= 30 else overall)
    return out


def _cqr_quantile(scores: np.ndarray, coverage: float) -> float:
    n = scores.size
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(max(np.quantile(scores, level), 0.0))


@dataclass
class DiagnosisModel:
    """Which failure mode is developing (trained on the pre-failure hours of known failures)."""

    members: list[HistGradientBoostingClassifier]
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        total = np.zeros((len(X), len(FAILURE_MODES)))
        for m in self.members:
            p = m.predict_proba(Xf)
            classes = [str(c) for c in m.classes_]
            for k, mode in enumerate(FAILURE_MODES):
                if mode in classes:
                    total[:, k] += p[:, classes.index(mode)]
        total /= max(len(self.members), 1)
        return pd.DataFrame(total, columns=[f"prob_{m}" for m in FAILURE_MODES], index=X.index)


# --------------------------------------------------------------------------------------------
# unsupervised anomaly detection (no failure labels needed)
# --------------------------------------------------------------------------------------------
@dataclass
class MSPCModel:
    """Multivariate statistical process control: PCA Hotelling T^2 and SPE (Q) statistics.

    Fitted on known-healthy operation; the score is the larger of T^2 / T^2_limit and
    SPE / SPE_limit, so 1.0 sits on the 99 % control limit.
    """

    scaler: StandardScaler
    pca: PCA
    t2_limit: float
    spe_limit: float
    features: list[str] = field(default_factory=lambda: list(ANOMALY_FEATURES))

    def stats(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (T^2, SPE, per-variable squared residuals = SPE contributions)."""
        Z = self.scaler.transform(X[self.features].fillna(0.0).to_numpy())
        scores = self.pca.transform(Z)
        t2 = np.sum(scores**2 / self.pca.explained_variance_, axis=1)
        resid = Z - self.pca.inverse_transform(scores)
        return t2, np.sum(resid**2, axis=1), resid**2

    def score(self, X: pd.DataFrame) -> np.ndarray:
        t2, spe, _ = self.stats(X)
        return np.maximum(t2 / self.t2_limit, spe / self.spe_limit)


def fit_mspc(X_healthy: pd.DataFrame, var_explained: float = 0.9) -> MSPCModel:
    feats = list(ANOMALY_FEATURES)
    Z = X_healthy[feats].fillna(0.0).to_numpy()
    scaler = StandardScaler().fit(Z)
    pca = PCA(n_components=var_explained, svd_solver="full").fit(scaler.transform(Z))
    m = MSPCModel(scaler, pca, 1.0, 1.0, feats)
    t2, spe, _ = m.stats(X_healthy)
    m.t2_limit, m.spe_limit = float(np.quantile(t2, 0.99)), float(np.quantile(spe, 0.99))
    return m


@dataclass
class IForestModel:
    model: IsolationForest
    scale: float
    features: list[str] = field(default_factory=lambda: list(ANOMALY_FEATURES))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X[self.features].fillna(0.0)) / self.scale


def fit_iforest(X_healthy: pd.DataFrame, random_state: int = 7) -> IForestModel:
    feats = list(ANOMALY_FEATURES)
    Z = X_healthy[feats].fillna(0.0)
    model = IsolationForest(n_estimators=150, max_samples=2048, random_state=random_state).fit(Z)
    s = -model.score_samples(Z)
    return IForestModel(model, float(np.quantile(s, 0.99)), feats)


@dataclass
class ScoreEnsemble:
    """Average of the per-fold unsupervised detectors (same idea as the supervised CV ensembles)."""

    members: list

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.score(X) for m in self.members], axis=0)


def rule_score(clean: pd.DataFrame) -> np.ndarray:
    """Conventional alarm limits. >= 1.0 means at least one tag is beyond its alarm limit."""
    parts = [clean[tag].to_numpy(dtype=np.float64) / lim for tag, lim in RULE_LIMITS.items()]
    return np.fmax.reduce(np.vstack(parts), axis=0)


# --------------------------------------------------------------------------------------------
# hybrid alert score
# --------------------------------------------------------------------------------------------
@dataclass
class ECDF:
    """Empirical CDF of a reference score distribution (maps a score to its percentile)."""

    x: np.ndarray
    p: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, n: int = 2001) -> "ECDF":
        v = np.asarray(values, dtype=np.float64)
        v = v[~np.isnan(v)]
        q = np.linspace(0.0, 1.0, n)
        xs = np.quantile(v, q)
        ux, idx = np.unique(xs[::-1], return_index=True)  # ties: keep the highest percentile
        return cls(ux, q[::-1][idx])

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        out = np.interp(np.nan_to_num(v, nan=-np.inf), self.x, self.p, left=0.0, right=1.0)
        return np.where(np.isnan(v), np.nan, out)


@dataclass
class HybridModel:
    """Alert score = min(percentile of GBM risk, percentile of MSPC score), calibrated to P(failure)."""

    ecdf_risk: ECDF
    ecdf_mspc: ECDF
    calibrator: IsotonicRegression

    def fuse(self, raw_risk: np.ndarray, mspc: np.ndarray) -> np.ndarray:
        # np.minimum (not fmin): if either detector has no score the fused score is unknown too.
        return np.minimum(self.ecdf_risk(raw_risk), self.ecdf_mspc(mspc))

    def predict_proba(self, raw_risk: np.ndarray, mspc: np.ndarray) -> np.ndarray:
        fused = self.fuse(raw_risk, mspc)
        return np.where(np.isnan(fused), np.nan, calibrate(self.calibrator, fused))


# --------------------------------------------------------------------------------------------
# training: one pass over the folds fits every model
# --------------------------------------------------------------------------------------------
@dataclass
class TrainedModels:
    risk: RiskModel
    rul: RULModel
    diagnosis: DiagnosisModel
    mspc: ScoreEnsemble
    iforest: ScoreEnsemble
    hybrid: HybridModel
    oof: pd.DataFrame  # out-of-fold scores for every training row


def train_models(
    X: pd.DataFrame, labels: pd.DataFrame, groups: np.ndarray, hours: np.ndarray, tc: TrainConfig,
    rul_cap_h: float, log=lambda msg: None,
) -> TrainedModels:
    """Fit every model with GroupKFold-by-pump on the training fleet.

    X: feature rows of running pumps. labels: aligned frame with label, ttf, rul_target,
    next_mode, next_predictable.
    """
    feats = list(FEATURE_COLUMNS)
    y = labels["label"].to_numpy()
    ttf = labels["ttf"].to_numpy()
    predictable = labels["next_predictable"].to_numpy().astype(bool)
    modes = labels["next_mode"].to_numpy()
    stride = stride_mask(hours, tc.train_stride_h)
    risk_fit = stride & (y >= 0)
    rul_fit = stride & predictable & (ttf <= rul_cap_h)
    dx_fit = (y == 1) & predictable
    healthy = labels["rul_target"].to_numpy() >= rul_cap_h   # >= 30 days from any failure

    n = len(X)
    oof_raw = np.full(n, np.nan)
    oof_mspc = np.full(n, np.nan)
    oof_if = np.full(n, np.nan)
    oof_q = np.full((n, len(RUL_QUANTILES)), np.nan)
    oof_dx = np.full((n, len(FAILURE_MODES)), np.nan)
    risk_m, rul_m, dx_m, mspc_m, if_m = [], [], [], [], []
    rng = np.random.default_rng(tc.random_state)

    for k, (tr, va) in enumerate(oof_folds(groups, tc.cv_folds), start=1):
        log(f"  fold {k}/{tc.cv_folds}: {len(np.unique(groups[tr]))} pumps train, "
            f"{len(np.unique(groups[va]))} validate")
        Xva = X.iloc[va]
        h_tr = tr[healthy[tr]]
        mspc = fit_mspc(X.iloc[h_tr])
        sub = rng.choice(h_tr, size=min(40000, h_tr.size), replace=False)
        iforest = fit_iforest(X.iloc[sub], tc.random_state + k)
        oof_mspc[va], oof_if[va] = mspc.score(Xva), iforest.score(Xva)
        mspc_m.append(mspc)
        if_m.append(iforest)

        t = tr[risk_fit[tr]]
        clf = _clf(tc).fit(X.iloc[t][feats], y[t])
        oof_raw[va] = clf.predict_proba(Xva[feats])[:, 1]
        risk_m.append(clf)

        t = tr[rul_fit[tr]]
        fold = {q: _reg(tc, q).fit(X.iloc[t][feats], ttf[t]) for q in RUL_QUANTILES}
        oof_q[va] = np.column_stack([fold[q].predict(Xva[feats]) for q in RUL_QUANTILES])
        rul_m.append(fold)

        t = tr[dx_fit[tr]]
        if len(np.unique(modes[t])) >= 2:
            dclf = _clf(tc, class_weight="balanced").fit(X.iloc[t][feats], modes[t])
            dx_m.append(dclf)
            oof_dx[va] = DiagnosisModel([dclf], feats).predict_proba(Xva).to_numpy()

    known = y >= 0
    risk = RiskModel(risk_m, _fit_calibrator(oof_raw[known], y[known]), feats)
    oof_q = np.sort(np.clip(oof_q, 0, rul_cap_h), axis=1)
    rul = RULModel(rul_m, rul_cap_h, features=feats)
    rul_eval = predictable & (ttf <= rul_cap_h)
    rul.conformal_q = conformal_offsets(oof_q[rul_eval], ttf[rul_eval], tc.rul_interval_coverage)

    hybrid = HybridModel(ECDF.fit(oof_raw), ECDF.fit(oof_mspc), IsotonicRegression())
    fused = hybrid.fuse(oof_raw, oof_mspc)
    hybrid.calibrator = _fit_calibrator(fused[known], y[known])

    oof = pd.DataFrame(index=X.index)
    oof["risk"] = hybrid.predict_proba(oof_raw, oof_mspc)
    oof["risk_gbm"] = calibrate(risk.calibrator, oof_raw)
    oof[["rul_p10", "rul_p50", "rul_p90"]] = rul.adjust(oof_q, X.index).to_numpy()
    oof[[f"prob_{m}" for m in FAILURE_MODES]] = oof_dx
    oof["mspc_score"] = oof_mspc
    oof["iforest_score"] = oof_if
    return TrainedModels(risk=risk, rul=rul, diagnosis=DiagnosisModel(dx_m, feats), mspc=ScoreEnsemble(mspc_m),
                         iforest=ScoreEnsemble(if_m), hybrid=hybrid, oof=oof)
