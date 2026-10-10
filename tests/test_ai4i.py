"""Offline tests for the UCI AI4I 2020 benchmark (sentinel_pdm.ai4i).

No test touches the network or the project's data/ and artifacts/ folders: downloads are faked
with an in-memory zip, and the end-to-end run uses a small synthetic dataset in the AI4I schema
whose failures follow the published mechanisms.
"""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm import ai4i as A


def synthetic_ai4i(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """AI4I-schema data generated with the published failure rules (HDF, PWF, OSF, TWF, RNF)."""
    rng = np.random.default_rng(seed)
    quality = rng.choice(np.array(["L", "M", "H"]), size=n, p=[0.5, 0.3, 0.2])
    air = np.round(rng.normal(300.0, 2.0, n), 1)
    proc = np.round(air + 10.0 + rng.normal(0.0, 1.0, n), 1)
    rpm = np.round(np.clip(rng.normal(1540.0, 180.0, n), 1170, 2890))
    torque = np.round(np.clip(rng.normal(40.0, 10.0, n), 3.0, 77.0), 1)
    wear = (np.arange(n) * 2) % 250
    power = torque * rpm * 2 * np.pi / 60
    limit = pd.Series(quality).map({"L": 11000, "M": 12000, "H": 13000}).to_numpy()
    flags = {
        "TWF": (wear >= 200) & (wear <= 240) & (rng.random(n) < 0.05),
        "HDF": (proc - air < 8.6) & (rpm < 1380),
        "PWF": (power < 3500) | (power > 9000),
        "OSF": wear * torque > limit,
        "RNF": rng.random(n) < 0.002,
    }
    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{q}{40000 + i}" for i, q in enumerate(quality)],
        "Type": quality,
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm.astype(int),
        "Torque [Nm]": torque,
        "Tool wear [min]": wear.astype(int),
    })
    df["Machine failure"] = np.any(np.column_stack(list(flags.values())), axis=1).astype(int)
    for k, v in flags.items():
        df[k] = v.astype(int)
    return df


# --------------------------------------------------------------------------------------------
# features and leakage
# --------------------------------------------------------------------------------------------
def test_physics_features_values():
    df = synthetic_ai4i(3).iloc[:3].copy()
    df["Type"] = ["L", "M", "H"]
    df["Air temperature [K]"] = [300.0, 301.0, 298.0]
    df["Process temperature [K]"] = [310.5, 309.0, 306.0]
    df["Rotational speed [rpm]"] = [1500, 1300, 2800]
    df["Torque [Nm]"] = [40.0, 60.0, 10.0]
    df["Tool wear [min]"] = [0, 200, 120]
    df.index = [10, 11, 12]

    f = A.physics_features(df)

    assert list(f.index) == [10, 11, 12]
    np.testing.assert_allclose(f["temp_diff_K"], [10.5, 8.0, 8.0])
    np.testing.assert_allclose(f["power_W"], [40 * 1500 * 2 * np.pi / 60, 60 * 1300 * 2 * np.pi / 60,
                                              10 * 2800 * 2 * np.pi / 60])
    assert f["power_W"].iloc[0] == pytest.approx(6283.185, abs=1e-3)
    np.testing.assert_allclose(f["overstrain_minNm"], [0.0, 12000.0, 1200.0])
    onehot = f[A.QUALITY_COLUMNS].to_numpy()
    np.testing.assert_array_equal(onehot, np.eye(3, dtype=int))


def test_inputs_never_contain_leakage_columns():
    df = synthetic_ai4i(300)
    X = A.build_inputs(df)

    assert list(X.columns) == A.FEATURE_SETS["physics"]
    assert not set(X.columns) & set(A.LEAKAGE_COLUMNS)
    for cols in A.FEATURE_SETS.values():
        assert not set(cols) & set(A.LEAKAGE_COLUMNS)
    assert set(A.FEATURE_SETS["raw"]) < set(A.FEATURE_SETS["physics"])
    assert not set(A.ENGINEERED_COLUMNS) & set(A.FEATURE_SETS["raw"])

    # Scrambling the answer, the failure-type flags and the identifiers must not move any input.
    scrambled = df.copy()
    scrambled[A.TARGET] = 1 - scrambled[A.TARGET]
    for t in A.FAILURE_TYPES:
        scrambled[t] = np.random.default_rng(1).permutation(scrambled[t].to_numpy())
    scrambled["UDI"] = scrambled["UDI"].to_numpy()[::-1]
    scrambled["Product ID"] = "X0"
    pd.testing.assert_frame_equal(A.build_inputs(scrambled), X)


def test_importance_groups_treat_quality_as_one_input():
    groups = A._input_groups(A.FEATURE_SETS["physics"])
    assert set(groups) == {*A.SENSOR_COLUMNS.values(), "type", *A.ENGINEERED_COLUMNS}
    assert len(groups["type"]) == 3
    assert set(groups) <= set(A.FEATURE_INFO)


# --------------------------------------------------------------------------------------------
# validation and loading
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("breakage", ["missing_column", "bad_type", "nan", "non_binary", "row_count"])
def test_validate_rejects_bad_data(breakage):
    df = synthetic_ai4i(50)
    expected_rows = None
    if breakage == "missing_column":
        df = df.drop(columns="Torque [Nm]")
    elif breakage == "bad_type":
        df.loc[0, "Type"] = "X"
    elif breakage == "nan":
        df["Torque [Nm]"] = df["Torque [Nm]"].astype(float)
        df.loc[3, "Torque [Nm]"] = np.nan
    elif breakage == "non_binary":
        df.loc[0, "Machine failure"] = 2
    elif breakage == "row_count":
        expected_rows = A.N_ROWS
    with pytest.raises(ValueError):
        A.validate_ai4i(df, expected_rows=expected_rows)


def test_validate_accepts_good_data():
    A.validate_ai4i(synthetic_ai4i(50), expected_rows=None)


def test_load_offline_without_file_explains_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        A.load_ai4i(download=False, path=tmp_path / "ai4i2020.csv")
    msg = str(exc.value)
    assert A.DOWNLOAD_URL in msg and "--offline" in msg


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.payload


def _zipped_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ai4i2020.csv", df.to_csv(index=False))
    return buf.getvalue()


def test_load_downloads_extracts_validates_and_caches(tmp_path, monkeypatch):
    full = synthetic_ai4i(A.N_ROWS, seed=3)
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.get_header("User-agent"), timeout))
        return _FakeResponse(_zipped_csv(full))

    monkeypatch.setattr(A.urllib.request, "urlopen", fake_urlopen)
    path = tmp_path / "external" / "ai4i2020.csv"
    df = A.load_ai4i(download=True, path=path)

    assert path.exists() and len(df) == A.N_ROWS
    assert list(df.columns) == list(A.EXPECTED_COLUMNS)
    url, agent, timeout = calls[0]
    assert url == A.DOWNLOAD_URL and agent.startswith("sentinel-pdm/") and timeout > 0

    def no_network(*_args, **_kwargs):
        raise AssertionError("the cached file should be used")

    monkeypatch.setattr(A.urllib.request, "urlopen", no_network)
    again = A.load_ai4i(download=True, path=path)
    pd.testing.assert_frame_equal(again, df)


def test_load_network_failure_gives_actionable_error(tmp_path, monkeypatch):
    def offline(*_args, **_kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(A.urllib.request, "urlopen", offline)
    path = tmp_path / "ai4i2020.csv"
    with pytest.raises(ConnectionError) as exc:
        A.load_ai4i(download=True, path=path)
    assert "manually" in str(exc.value) and not path.exists()


def test_corrupt_download_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(A.urllib.request, "urlopen", lambda req, timeout: _FakeResponse(_zipped_csv(synthetic_ai4i(40))))
    path = tmp_path / "ai4i2020.csv"
    with pytest.raises(ValueError, match="10,000 rows"):
        A.load_ai4i(download=True, path=path)
    assert not path.exists()


# --------------------------------------------------------------------------------------------
# metric helpers
# --------------------------------------------------------------------------------------------
def test_best_f1_and_recall_at_precision():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    f1, thr = A.best_f1(y, p)
    # midway between the highest negative (0.4) and the lowest positive (0.8)
    assert f1 == pytest.approx(1.0) and thr == pytest.approx(0.6)
    assert A.recall_at_precision(y, p, 0.9) == (1.0, 1.0, True)

    # The top-scored row is a false alarm: precision 0.9 is never reached, so report the best achievable.
    y2 = np.array([1, 0, 1, 0, 0])
    p2 = np.array([0.9, 0.95, 0.5, 0.4, 0.1])
    rec, prec, reached = A.recall_at_precision(y2, p2, 0.9)
    assert not reached and prec == pytest.approx(2 / 3) and rec == pytest.approx(1.0)


def test_cross_fitted_f1_uses_thresholds_from_other_folds():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.9, 0.3, 0.8, 0.1, 0.7])
    fold = np.array([0, 0, 1, 1, 2, 2])
    # Fold 2's positive (0.7) scores below every positive of folds 0-1, but above their negatives:
    # the midpoint threshold (0.55) learned on folds 0-1 still catches it.
    assert A._xfit_f1(y, p, fold) == pytest.approx(1.0)

    # A fold whose positive scores below the other folds' negatives is missed: no hindsight.
    p_hard = np.array([0.2, 0.9, 0.3, 0.8, 0.1, 0.25])
    assert A._xfit_f1(y, p_hard, fold) < 1.0


# --------------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def benchmark_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ai4i_artifacts")
    data = synthetic_ai4i(1200)
    lines: list[str] = []
    metrics = A.run_ai4i_benchmark(download=False, log=lines.append, data=data, artifacts_dir=out,
                                   n_splits=3, n_repeats=2, quick=True)
    return metrics, out, data, lines


def test_benchmark_writes_three_artifacts_with_strict_json(benchmark_run):
    metrics, out, _, lines = benchmark_run
    paths = A.artifact_paths(out)
    assert all(p.exists() for p in paths.values())

    def reject(name):
        raise AssertionError(f"non-standard JSON constant {name}")

    on_disk = json.loads(paths["metrics"].read_text(encoding="utf-8"), parse_constant=reject)
    assert on_disk == metrics
    assert any("AI4I 2020 benchmark" in line for line in lines)


def test_benchmark_metrics_schema(benchmark_run):
    metrics, _, data, _ = benchmark_run
    for key in ("dataset", "protocol", "feature_sets", "models", "results", "best", "operating_point",
                "failure_type_recall", "mechanism_ceiling", "blocked_cv", "runtime_s", "generated_at", "version"):
        assert key in metrics

    assert metrics["protocol"]["n_splits"] == 3 and metrics["protocol"]["n_repeats"] == 2
    assert set(metrics["protocol"]["excluded_columns"]) == set(A.LEAKAGE_COLUMNS)
    assert metrics["dataset"]["n_failures"] == int(data[A.TARGET].sum())
    assert metrics["dataset"]["license"] == "CC BY 4.0"

    for fs in A.FEATURE_SETS:
        assert metrics["feature_sets"][fs]["features"] == A.FEATURE_SETS[fs]
        assert set(metrics["results"][fs]) == set(A.MODEL_LABELS)
        for r in metrics["results"][fs].values():
            for k in ("roc_auc", "pr_auc", "f1_best", "f1_xfit", "recall_at_p90", "precision_at_p90"):
                assert 0.0 <= r[k] <= 1.0 and f"{k}_std" in r
            assert isinstance(r["p90_reached"], bool)
        assert metrics["best"][fs] in A.MODEL_LABELS
        op = metrics["operating_point"][fs]
        assert sum(op["confusion"].values()) == pytest.approx(len(data))
        types = metrics["failure_type_recall"][fs]["types"]
        assert set(types) == {*A.FAILURE_TYPES, "none"}
        for t, v in types.items():
            assert v["predictable"] == A.PREDICTABLE_TYPES[t]
            assert v["recall"] is None if v["n"] == 0 else 0.0 <= v["recall"] <= 1.0
        assert 0.0 <= metrics["blocked_cv"][fs]["pr_auc"] <= 1.0
    assert metrics["best"]["overall"]["feature_set"] in A.FEATURE_SETS


def test_benchmark_learns_the_mechanisms(benchmark_run):
    metrics, *_ = benchmark_run
    res, best = metrics["results"], metrics["best"]
    raw, phys = res["raw"][best["raw"]]["pr_auc"], res["physics"][best["physics"]]["pr_auc"]
    # Failures here follow the published rules exactly, so physics features must help.
    assert phys > 0.8
    assert phys > raw


def test_benchmark_oof_and_importance_files(benchmark_run):
    _, out, data, _ = benchmark_run
    paths = A.artifact_paths(out)
    oof = pd.read_csv(paths["oof"])
    assert list(oof.columns) == ["udi", "type", *A.SENSOR_COLUMNS.values(), *A.ENGINEERED_COLUMNS,
                                 "machine_failure", *A.FAILURE_TYPES, "prob_raw", "prob_physics"]
    assert len(oof) == len(data)
    np.testing.assert_array_equal(oof["machine_failure"], data[A.TARGET])
    for c in ("prob_raw", "prob_physics"):
        assert oof[c].between(0, 1).all() and oof[c].notna().all()

    imp = pd.read_csv(paths["importance"])
    assert list(imp.columns) == ["feature_set", "model", "feature", "label", "kind", "importance", "std", "rank"]
    assert set(imp.loc[imp["feature_set"] == "raw", "feature"]) == {*A.SENSOR_COLUMNS.values(), "type"}
    assert set(imp.loc[imp["feature_set"] == "physics", "feature"]) == {*A.SENSOR_COLUMNS.values(), "type",
                                                                        *A.ENGINEERED_COLUMNS}
    assert not set(imp["feature"]) & set(A.LEAKAGE_COLUMNS)
    assert imp.groupby("feature_set")["rank"].min().eq(1).all()


def test_benchmark_rejects_too_few_failures(tmp_path):
    data = synthetic_ai4i(200)
    data[A.TARGET] = 0
    data.loc[:1, A.TARGET] = 1
    with pytest.raises(ValueError, match="failures"):
        A.run_ai4i_benchmark(download=False, log=lambda _: None, data=data, artifacts_dir=tmp_path,
                             n_splits=3, n_repeats=1, quick=True)
