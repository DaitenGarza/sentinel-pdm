"""Shared pytest fixtures for the Sentinel PdM test suite.

Expensive objects (a simulated fleet, its cleaned data and features, a trained model set and
one complete pipeline run) are built once per session on deliberately small configurations, so
the whole suite runs in a few minutes on a laptop CPU. Everything a test writes goes under
pytest's ``tmp_path``: the real ``data/`` and ``artifacts/`` folders the dashboard reads are never
touched.

Session fixtures hand out *shared* objects. Tests that need to modify one must copy it first.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

# Make the package importable from a fresh clone even before ``pip install -e .``.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sentinel_pdm import models  # noqa: E402
from sentinel_pdm.config import (  # noqa: E402
    ARTIFACTS_DIR, CONTEXT, DATA_DIR, SENSORS, Config, LabelConfig, SimConfig, TrainConfig,
)
from sentinel_pdm.features import build_features  # noqa: E402
from sentinel_pdm.labels import make_labels  # noqa: E402
from sentinel_pdm.pipeline import run_pipeline  # noqa: E402
from sentinel_pdm.quality import clean_readings  # noqa: E402
from sentinel_pdm.simulator import FleetData, simulate_fleet  # noqa: E402

# Typical healthy levels of every tag on a mid-size process pump (used for hand-made data).
BASE_LEVELS: dict[str, float] = {
    "vib_rms": 1.6, "vib_1x": 0.6, "vib_2x": 0.3, "vib_hf": 0.25, "vib_kurtosis": 3.2,
    "bearing_temp": 55.0, "motor_temp": 85.0, "motor_current": 120.0, "current_imbalance": 1.2,
    "suction_pressure": 2.5, "discharge_pressure": 12.0, "flow": 300.0,
}
TEMPERATURE_TAGS = ("bearing_temp", "motor_temp")


# --------------------------------------------------------------------------------------------
# small configurations
# --------------------------------------------------------------------------------------------
def small_sim_config() -> SimConfig:
    """12 pumps x 6 months: about a dozen failures covering every mode, simulated in < 1 s.

    Smaller fleets (e.g. 8 pumps x 2,500 h) leave some cross-validation folds without a single
    predictable failure, which no supervised model can train on.
    """
    return SimConfig(n_assets=12, hours=4380, seed=1)


def small_config() -> Config:
    """Full pipeline configuration for tests: the small fleet and a short gradient-boosting run."""
    return Config(sim=small_sim_config(), train=TrainConfig(max_iter=40, cv_folds=2))


# --------------------------------------------------------------------------------------------
# hand-made data
# --------------------------------------------------------------------------------------------
def _pump_frame(hours: int = 240, asset_id: str = "P-101A", seed: int = 0, noise: float = 0.01,
                running: np.ndarray | None = None) -> pd.DataFrame:
    """One pump of hourly historian data at healthy levels, in the raw-readings schema.

    Small random noise keeps consecutive values distinct (the cleaner treats 6 identical hours
    as a stuck tag); ``noise=0`` gives perfectly flat signals for exact feature arithmetic.
    The same frame is valid input for ``clean_readings`` and ``build_features``.
    """
    rng = np.random.default_rng(seed)
    h = np.arange(hours)
    df = pd.DataFrame({"asset_id": asset_id, "hour": h.astype(np.int32),
                       "timestamp": pd.date_range("2025-01-01", periods=hours, freq="h")})
    for tag, level in BASE_LEVELS.items():
        if tag in TEMPERATURE_TAGS:
            x = level + 20 * noise * rng.normal(size=hours)       # ~0.2 degC noise at noise=0.01
        else:
            x = level * (1 + noise * rng.normal(size=hours))
        df[tag] = x.astype(np.float32)
    df["running"] = (np.ones(hours) if running is None else np.asarray(running)).astype(np.int8)
    df["load_pct"] = np.float32(80.0)
    df["speed_rpm"] = np.float32(3540.0)
    df["ambient_temp"] = np.float32(25.0)
    off = df["running"].to_numpy() == 0
    df.loc[off, ["load_pct", "speed_rpm", "motor_current", "flow"]] = 0.0
    return df[["asset_id", "hour", "timestamp", *SENSORS, "running", *CONTEXT]]


@pytest.fixture
def make_pump() -> Callable[..., pd.DataFrame]:
    """Factory for one pump of hand-made hourly data (see ``_pump_frame``)."""
    return _pump_frame


# --------------------------------------------------------------------------------------------
# simulated fleet and the stages built on it
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sim_config() -> SimConfig:
    return small_sim_config()


@pytest.fixture(scope="session")
def fleet(sim_config) -> FleetData:
    return simulate_fleet(sim_config)


@pytest.fixture(scope="session")
def cleaned(fleet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean readings, data-quality summary) of the small fleet."""
    return clean_readings(fleet.readings)


@pytest.fixture(scope="session")
def features(fleet, cleaned) -> pd.DataFrame:
    return build_features(cleaned[0], fleet.events)


@dataclass(frozen=True)
class TrainingSet:
    X: pd.DataFrame          # feature rows of running training pumps
    labels: pd.DataFrame     # aligned targets
    groups: np.ndarray       # asset_id per row (cross-validation groups)
    hours: np.ndarray


@pytest.fixture(scope="session")
def training_set(fleet, features, sim_config) -> TrainingSet:
    """Training-fleet rows exactly as ``run_pipeline`` hands them to ``train_models``."""
    labels = make_labels(features[["asset_id", "hour"]], fleet.events, sim_config.hours, LabelConfig())
    split = features["asset_id"].map(fleet.assets.set_index("asset_id")["split"]).to_numpy()
    rows = (features["running"].to_numpy() == 1) & (split == "train")
    X = features.loc[rows]
    return TrainingSet(X=X, labels=labels.loc[rows], groups=X["asset_id"].to_numpy(), hours=X["hour"].to_numpy())


@dataclass(frozen=True)
class TrainedRun:
    models: models.TrainedModels
    folds: list[tuple[np.ndarray, np.ndarray]]   # (train, validation) row positions actually used
    train_config: TrainConfig


@pytest.fixture(scope="session")
def trained(training_set) -> TrainedRun:
    """``train_models`` on the small training fleet, recording the folds it really used."""
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    real_folds = models.oof_folds

    def recording_folds(groups, n_splits):
        for tr, va in real_folds(groups, n_splits):
            folds.append((np.array(tr), np.array(va)))
            yield tr, va

    tc = TrainConfig(max_iter=40, cv_folds=3)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(models, "oof_folds", recording_folds)
        tm = models.train_models(training_set.X, training_set.labels, training_set.groups, training_set.hours,
                                 tc, rul_cap_h=LabelConfig().rul_cap_h)
    return TrainedRun(models=tm, folds=folds, train_config=tc)


# --------------------------------------------------------------------------------------------
# one complete pipeline run into a temporary directory
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineRun:
    cfg: Config
    metrics: dict
    data_dir: Path
    artifacts_dir: Path


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory) -> PipelineRun:
    root = tmp_path_factory.mktemp("pipeline")
    data_dir, artifacts_dir = root / "data", root / "artifacts"
    # Guard rail: the suite must never overwrite the artifacts the dashboard is showing.
    assert data_dir.resolve() != DATA_DIR.resolve() and artifacts_dir.resolve() != ARTIFACTS_DIR.resolve()
    cfg = small_config()
    metrics = run_pipeline(cfg, data_dir=data_dir, artifacts_dir=artifacts_dir, importance=False,
                           log=lambda *_: None)
    return PipelineRun(cfg=cfg, metrics=metrics, data_dir=data_dir, artifacts_dir=artifacts_dir)
