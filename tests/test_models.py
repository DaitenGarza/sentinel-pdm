"""Model building blocks and the cross-validated training run.

The small pieces (empirical CDF, hybrid fusion, isotonic calibration, conformal RUL offsets,
MSPC limits, alarm-limit baseline) are tested on hand-made or synthetic data where the right
answer is known. ``train_models`` is tested on the shared small fleet (``trained`` fixture) for
the properties every downstream number depends on: out-of-fold scores exist for every training
row and really come from a model that never saw the pump, and every output stays in range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from sentinel_pdm.config import FAILURE_MODES, LabelConfig
from sentinel_pdm.features import FEATURE_COLUMNS
from sentinel_pdm.models import (
    ANOMALY_FEATURES, ECDF, RUL_BINS, RULE_LIMITS, HybridModel, RULModel, calibrate, conformal_offsets, fit_mspc,
    rule_score,
)

OOF_COLUMNS = ["risk", "risk_gbm", "rul_p10", "rul_p50", "rul_p90", *[f"prob_{m}" for m in FAILURE_MODES],
               "mspc_score", "iforest_score"]
N_BANDS = len(RUL_BINS) - 1

# Reference score distributions shaped like the real ones: GBM risk piles up near 0, MSPC is skewed.
RISK_REF = np.random.default_rng(4).beta(0.5, 5.0, 5000)
MSPC_REF = np.random.default_rng(5).lognormal(size=5000)


def _isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Isotonic calibrator configured exactly like the ones ``train_models`` fits."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)


# --------------------------------------------------------------------------------------------
# empirical CDF (percentile ranks behind the hybrid score)
# --------------------------------------------------------------------------------------------
def test_ecdf_matches_the_empirical_distribution():
    ref = np.random.default_rng(0).normal(size=20_000)
    ecdf = ECDF.fit(ref)
    x = np.linspace(-3.0, 3.0, 61)
    np.testing.assert_allclose(ecdf(x), [(ref <= v).mean() for v in x], atol=2e-3)


def test_ecdf_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    ecdf = ECDF.fit(rng.lognormal(size=5000))
    x = np.sort(np.concatenate([rng.lognormal(sigma=2.0, size=2000), [-np.inf, -5.0, 0.0, 1e6, np.inf]]))
    p = ecdf(x)
    assert np.all(np.diff(p) >= 0)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert ecdf(np.array([-1.0, -np.inf])).tolist() == [0.0, 0.0]     # below everything seen
    assert ecdf(np.array([1e9, np.inf])).tolist() == [1.0, 1.0]       # above everything seen


def test_ecdf_passes_nan_through_and_ignores_it_when_fitting():
    ref = np.random.default_rng(2).normal(size=3000)
    ecdf = ECDF.fit(ref)
    p = ecdf(np.array([np.nan, 0.0, np.nan]))
    assert np.isnan(p[[0, 2]]).all() and np.isfinite(p[1])
    with_gaps = ECDF.fit(np.concatenate([ref, np.full(500, np.nan)]))   # stopped-pump rows in the reference
    np.testing.assert_array_equal(with_gaps.x, ecdf.x)
    np.testing.assert_array_equal(with_gaps.p, ecdf.p)


def test_ecdf_ties_take_the_highest_percentile():
    """A score shared by 60 % of the reference sits at the 60th percentile, not the 0th.

    Gradient-boosting probabilities and clipped anomaly scores have large tied blocks; mapping a
    tie to its lowest percentile would make the hybrid score treat them as "perfectly normal".
    """
    ref = np.concatenate([np.zeros(600), np.random.default_rng(3).uniform(1.0, 2.0, 400)])
    np.testing.assert_allclose(ECDF.fit(ref)(np.zeros(3)), 0.6, atol=2e-3)
    flat = ECDF.fit(np.full(100, 2.0))           # a fully tied reference still gives a valid map
    assert flat(np.array([1.9, 2.0, 2.1])).tolist() == [0.0, 1.0, 1.0]


# --------------------------------------------------------------------------------------------
# hybrid score
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hybrid() -> HybridModel:
    ramp = np.linspace(0.0, 1.0, 101)
    return HybridModel(ECDF.fit(RISK_REF), ECDF.fit(MSPC_REF), _isotonic(ramp, ramp ** 2))


def test_hybrid_fuse_is_the_minimum_of_the_two_percentiles(hybrid):
    rng = np.random.default_rng(6)
    risk, mspc = rng.beta(0.5, 5.0, 1000), rng.lognormal(size=1000)
    expected = np.minimum(hybrid.ecdf_risk(risk), hybrid.ecdf_mspc(mspc))
    np.testing.assert_array_equal(hybrid.fuse(risk, mspc), expected)


def test_hybrid_needs_both_detectors_to_agree(hybrid):
    risk_hi, risk_typ = np.quantile(RISK_REF, [0.999, 0.5])
    mspc_hi, mspc_typ = np.quantile(MSPC_REF, [0.999, 0.5])
    fused = hybrid.fuse(np.array([risk_hi, risk_hi, risk_typ]), np.array([mspc_hi, mspc_typ, mspc_hi]))
    assert fused[0] > 0.99                       # both extreme: extreme
    assert fused[1] < 0.6 and fused[2] < 0.6     # one extreme, the other typical: typical


def test_hybrid_probability_is_bounded_and_nan_while_stopped(hybrid):
    rng = np.random.default_rng(7)
    risk = np.append(rng.beta(0.5, 5.0, 500), np.nan)
    mspc = np.append(rng.lognormal(size=500), np.nan)
    p = hybrid.predict_proba(risk, mspc)
    assert np.isnan(p[-1])
    assert np.all((p[:-1] >= 0.0) & (p[:-1] <= 1.0))
    order = np.argsort(hybrid.fuse(risk, mspc)[:-1], kind="stable")
    assert np.all(np.diff(p[:-1][order]) >= 0)   # calibration never reorders the alert score


# --------------------------------------------------------------------------------------------
# probability calibration
# --------------------------------------------------------------------------------------------
def test_calibrate_keeps_the_ranking_and_stays_in_the_unit_interval():
    rng = np.random.default_rng(8)
    raw = rng.uniform(0.0, 1.0, 5000)
    y = (rng.random(5000) < 0.7 * raw ** 2).astype(int)
    iso = _isotonic(raw, y)
    plateaus = iso.predict(raw)
    assert len(np.unique(plateaus)) < 500        # isotonic regression alone ties whole blocks of rows

    cal = calibrate(iso, raw)
    assert np.all((cal >= 0.0) & (cal <= 1.0))
    np.testing.assert_allclose(cal, plateaus, atol=1e-6)      # no probability visibly moves...
    below_top = plateaus < 1.0                                # (the top plateau: see the next test)
    order = np.argsort(raw[below_top])
    assert np.all(np.diff(cal[below_top][order]) > 0)         # ...but every tie is broken in raw-score order

    odd = calibrate(iso, np.array([-0.5, 1.5, np.nan, -np.inf, np.inf]))
    assert np.all(np.isfinite(odd) & (odd >= 0.0) & (odd <= 1.0))


def test_calibrate_keeps_the_ranking_where_isotonic_regression_saturates():
    """The documented contract holds in the top plateau too (regression test).

    An earlier version added the tie-break and then clipped to [0, 1], so where the isotonic fit
    reaches y_max = 1.0 every raw score mapped to exactly 1.0. Isotonic regression reaches 1.0
    whenever the highest-scored calibration row is a failure, the normal case for a useful model.
    """
    raw = np.linspace(0.0, 1.0, 101)
    cal = calibrate(_isotonic(raw, (raw > 0.5).astype(int)), raw)
    assert np.all(np.diff(cal) > 0)


# --------------------------------------------------------------------------------------------
# conformal RUL intervals
# --------------------------------------------------------------------------------------------
NARROWING = (1.0, 0.75, 0.5, 0.35)   # raw interval width / true P10-P90 width, per predicted-RUL band


def _rul_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Heteroscedastic RUL problem with an over-confident quantile model.

    The true RUL is normal around the predicted median with a spread that grows with it. The raw
    P10/P90 are centred correctly but too narrow, by a different factor in each RUL band, which
    is what Mondrian (per-band) conformal calibration exists to fix.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(40.0, 600.0, n)
    sd = 4.0 + 0.2 * mu
    y = mu + sd * rng.standard_normal(n)
    band = np.digitize(mu, RUL_BINS[1:-1])
    half = 1.2816 * sd * np.asarray(NARROWING)[band]       # 1.2816 = z(0.9): exact P10-P90 when narrowing = 1
    return np.column_stack([mu - half, mu, mu + half]), y


def _covered(q: np.ndarray, y: np.ndarray, offsets: list[float]) -> np.ndarray:
    adj = RULModel(members=[], cap_h=1e9, conformal_q=offsets).adjust(q)
    return (y >= adj["rul_p10"].to_numpy()) & (y <= adj["rul_p90"].to_numpy())


def test_conformal_offsets_reach_the_target_coverage_in_every_band():
    target = 0.8
    q_cal, y_cal = _rul_problem(20_000, seed=9)
    q_new, y_new = _rul_problem(20_000, seed=10)         # coverage is checked on data not used to fit
    offsets = conformal_offsets(q_cal, y_cal, target)

    assert len(offsets) == N_BANDS and np.all(np.diff(offsets) > 0)    # narrower raw band -> larger offset
    raw_covered = (y_new >= q_new[:, 0]) & (y_new <= q_new[:, 2])
    assert raw_covered.mean() < 0.55                                     # the raw intervals are over-confident

    covered = _covered(q_new, y_new, offsets)
    assert covered.mean() == pytest.approx(target, abs=0.02)
    band = np.digitize(q_new[:, 1], RUL_BINS[1:-1])
    for k in range(N_BANDS):
        assert covered[band == k].mean() == pytest.approx(target, abs=0.035), f"band {k}"
    # normalised scores: quiet and noisy predictions are covered alike, not just on average
    quiet, noisy = q_new[:, 1] < 180, q_new[:, 1] > 460
    assert covered[quiet].mean() == pytest.approx(target, abs=0.035)
    assert covered[noisy].mean() == pytest.approx(target, abs=0.035)


def _finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    n = scores.size
    return float(np.quantile(scores, np.ceil((n + 1) * coverage) / n))


@pytest.mark.parametrize("n_sparse, falls_back", [(0, True), (29, True), (30, False)])
def test_conformal_offsets_use_the_overall_offset_for_sparse_bands(n_sparse, falls_back):
    """Bands with fewer than 30 calibration rows borrow the pooled offset instead of a noisy one."""
    coverage = 0.8
    # 200 rows in the 72-168 h band with normalised scores spread over [-0.5, 0.5] ...
    dense = np.linspace(-0.5, 0.5, 200)
    q_dense = np.tile([90.0, 100.0, 110.0], (dense.size, 1))
    # ... and n_sparse rows in the > 336 h band, all far outside their interval (score 3.0)
    q_sparse = np.tile([390.0, 400.0, 410.0], (n_sparse, 1))
    q = np.vstack([q_dense, q_sparse])
    scores = np.concatenate([dense, np.full(n_sparse, 3.0)])
    y = q[:, 2] + scores * (q[:, 2] - q[:, 0])    # score = max(p10 - y, y - p90) / width

    offsets = conformal_offsets(q, y, coverage)

    overall = _finite_sample_quantile(scores, coverage)
    assert offsets[0] == pytest.approx(overall) and offsets[2] == pytest.approx(overall)   # empty bands
    assert offsets[1] == pytest.approx(_finite_sample_quantile(dense, coverage))          # dense band: its own
    assert offsets[3] == pytest.approx(overall if falls_back else 3.0)


# --------------------------------------------------------------------------------------------
# MSPC and the alarm-limit baseline
# --------------------------------------------------------------------------------------------
def test_mspc_control_limits_and_a_broken_correlation():
    """~1 % of the healthy fitting data lies beyond each 99 % limit, and breaking the normal
    correlation between signals is flagged even when every single value stays in its range."""
    rng = np.random.default_rng(11)
    n, k = 5000, len(ANOMALY_FEATURES)
    latent = rng.normal(size=(n, 2)) * [1.0, 0.8]          # two operating "directions" (load, temperature)
    healthy = pd.DataFrame(latent @ rng.normal(size=(2, k)) + 0.05 * rng.normal(size=(n, k)),
                           columns=ANOMALY_FEATURES)
    mspc = fit_mspc(healthy)

    beyond = (mspc.score(healthy) > 1.0).mean()
    assert 0.009 <= beyond <= 0.025                 # T^2 and SPE each exceed their limit on ~1 %

    f0 = healthy.iloc[:, 0].to_numpy()
    broken = healthy.iloc[[0]].copy()
    broken.iloc[0, 0] = f0[np.argmax(np.abs(f0 - f0[0]))]   # a value seen in healthy data, wrong context
    assert mspc.score(broken)[0] > 5.0


@pytest.mark.parametrize("vib, bearing, motor, alarm", [
    (2.0, 60.0, 100.0, False),          # every tag inside its limit
    (2.79, 81.9, 129.9, False),         # every tag just inside
    (2.8, 60.0, 100.0, True),           # exactly on the ISO 10816 zone B/C boundary
    (4.6, 60.0, 100.0, True),           # zone D vibration
    (2.0, 82.5, 100.0, True),           # bearing RTD alarm
    (2.0, 60.0, 131.0, True),           # winding RTD alarm
    (np.nan, 85.0, np.nan, True),       # a missing tag does not mask another tag's alarm
    (np.nan, 60.0, np.nan, False),
])
def test_rule_score_alarms_exactly_when_a_limit_is_exceeded(vib, bearing, motor, alarm):
    clean = pd.DataFrame({"vib_rms": [vib], "bearing_temp": [bearing], "motor_temp": [motor]})
    assert bool(rule_score(clean)[0] >= 1.0) is alarm


def test_rule_score_is_the_worst_ratio_to_the_limit():
    rng = np.random.default_rng(12)
    values = {tag: np.round(rng.uniform(0.5, 1.5, 2000) * lim, 2) for tag, lim in RULE_LIMITS.items()}
    values["vib_rms"][:5] = RULE_LIMITS["vib_rms"]                      # include exact hits
    values["bearing_temp"][rng.random(2000) < 0.1] = np.nan
    clean = pd.DataFrame(values)
    score = rule_score(clean)
    ratios = np.column_stack([clean[t].to_numpy() / lim for t, lim in RULE_LIMITS.items()])
    beyond = np.column_stack([clean[t].to_numpy() >= lim for t, lim in RULE_LIMITS.items()]).any(axis=1)
    np.testing.assert_array_equal(score >= 1.0, beyond)
    np.testing.assert_allclose(score, np.nanmax(ratios, axis=1))
    all_missing = pd.DataFrame({t: [np.nan] for t in RULE_LIMITS})
    assert np.isnan(rule_score(all_missing)[0])     # no reading, no verdict


def test_rule_score_on_the_simulated_fleet(cleaned):
    clean = cleaned[0]
    beyond = np.column_stack([clean[t].to_numpy(dtype=np.float64) >= lim
                              for t, lim in RULE_LIMITS.items()]).any(axis=1)
    np.testing.assert_array_equal(rule_score(clean) >= 1.0, beyond)
    assert 0 < beyond.sum() < len(clean)            # the degrading pumps do trip the alarm limits


# --------------------------------------------------------------------------------------------
# train_models on the small fleet
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def unseen_rows(fleet, features) -> pd.DataFrame:
    """Feature rows of the running *test* pumps (never seen by any fold model)."""
    split = features["asset_id"].map(fleet.assets.set_index("asset_id")["split"])
    return features.loc[(features["running"] == 1) & (split == "test")]


def test_oof_scores_cover_every_training_row(trained, training_set):
    oof = trained.models.oof
    assert list(oof.columns) == OOF_COLUMNS
    assert oof.index.equals(training_set.X.index)
    assert (training_set.X["running"] == 1).all()
    assert np.isfinite(oof.to_numpy()).all()


def test_folds_split_by_pump(trained, training_set):
    groups, n = training_set.groups, len(training_set.groups)
    assert len(trained.folds) == trained.train_config.cv_folds
    validated = np.zeros(n, dtype=int)
    for tr, va in trained.folds:
        assert not set(groups[tr]) & set(groups[va])                  # no pump on both sides
        assert len(tr) + len(va) == n and not np.intersect1d(tr, va).size
        validated[va] += 1
    assert (validated == 1).all()                                     # every row scored out-of-fold once


def test_oof_scores_come_from_the_model_that_never_saw_the_pump(trained, training_set):
    tm, X = trained.models, training_set.X
    assert len(tm.risk.members) == len(tm.rul.members) == len(tm.mspc.members) == len(trained.folds)
    for k, (_, va) in enumerate(trained.folds):
        raw = tm.risk.members[k].predict_proba(X.iloc[va][FEATURE_COLUMNS])[:, 1]
        np.testing.assert_allclose(tm.oof["risk_gbm"].iloc[va], calibrate(tm.risk.calibrator, raw), rtol=1e-12)
        np.testing.assert_allclose(tm.oof["mspc_score"].iloc[va], tm.mspc.members[k].score(X.iloc[va]),
                                   rtol=1e-12)


def test_risk_is_a_probability(trained, unseen_rows):
    tm = trained.models
    for p in (tm.oof["risk"], tm.oof["risk_gbm"], tm.risk.predict_proba(unseen_rows),
              tm.hybrid.predict_proba(tm.risk.raw_score(unseen_rows), tm.mspc.score(unseen_rows))):
        p = np.asarray(p)
        assert np.isfinite(p).all() and (p >= 0.0).all() and (p <= 1.0).all()


def test_rul_quantiles_are_ordered_and_capped(trained, unseen_rows):
    cap = LabelConfig().rul_cap_h
    rul = trained.models.rul
    assert len(rul.conformal_q) == N_BANDS and min(rul.conformal_q) >= 0.0
    for frame in (trained.models.oof, rul.predict(unseen_rows)):
        q = frame[["rul_p10", "rul_p50", "rul_p90"]].to_numpy()
        assert np.isfinite(q).all()
        assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
        assert (q >= 0.0).all() and (q <= cap).all()


def test_diagnosis_probabilities_form_a_distribution(trained, unseen_rows):
    p = trained.models.diagnosis.predict_proba(unseen_rows)
    assert list(p.columns) == [f"prob_{m}" for m in FAILURE_MODES]
    assert ((p >= 0.0) & (p <= 1.0)).all().all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
