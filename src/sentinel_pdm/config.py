"""Central configuration: paths, simulation settings, labeling, alerting and cost assumptions.

Everything tunable lives here so the CLI, the pipeline and the dashboard agree on one set of
numbers. Dataclasses are plain and JSON-serialisable so a run's exact configuration is saved
next to its results.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the project root (folder holding pyproject.toml).

    Order: SENTINEL_PDM_HOME env var -> the source checkout this file lives in -> current dir.
    """
    env = os.environ.get("SENTINEL_PDM_HOME")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Raw sensor tags recorded for every pump (hourly averages from the plant historian).
SENSORS: tuple[str, ...] = (
    "vib_rms",             # overall velocity RMS, mm/s (ISO 10816-3 severity)
    "vib_1x",              # 1x running-speed amplitude, mm/s (unbalance)
    "vib_2x",              # 2x running-speed amplitude, mm/s (misalignment / looseness)
    "vib_hf",              # high-frequency bearing band (demodulated), g
    "vib_kurtosis",        # kurtosis of acceleration waveform (impulsiveness), healthy ~3
    "bearing_temp",        # drive-end bearing RTD, degC
    "motor_temp",          # stator winding RTD, degC
    "motor_current",       # average phase current, A
    "current_imbalance",   # phase current imbalance, %
    "suction_pressure",    # bar(g)
    "discharge_pressure",  # bar(g)
    "flow",                # m3/h
)

# Context signals (not failure indicators themselves, but needed to interpret the sensors).
CONTEXT: tuple[str, ...] = ("load_pct", "speed_rpm", "ambient_temp")

# Failure modes the simulator can produce. "sudden" failures have no measurable precursor.
FAILURE_MODES: tuple[str, ...] = ("bearing", "misalignment", "cavitation", "winding")
ALL_MODES: tuple[str, ...] = FAILURE_MODES + ("sudden",)

# ISO 10816-3, group 2 machines (15-300 kW) on rigid foundations: zone boundaries in mm/s RMS.
ISO_ZONE_LIMITS = {"A/B": 1.4, "B/C": 2.8, "C/D": 4.5}


@dataclass
class SimConfig:
    n_assets: int = 48
    hours: int = 8760                 # one year of hourly data per pump
    seed: int = 42
    start: str = "2025-01-01"
    test_fraction: float = 1 / 3      # share of pumps held out entirely for testing
    mean_healthy_hours: float = 2600  # Weibull scale of time-to-degradation-onset
    weibull_shape: float = 1.6
    sudden_fraction: float = 0.12     # failures with no precursor (process upset, foreign object...)
    pm_interval_hours: float = 4380   # routine preventive maintenance roughly every 6 months
    sensor_fault_rate: float = 1.0    # multiplier on dropout / spike / stuck-sensor frequency


@dataclass
class LabelConfig:
    horizon_h: int = 168      # predict "failure within the next 7 days"
    rul_cap_h: int = 720      # RUL regression target is capped at 30 days (piecewise-linear RUL)


@dataclass
class AlertConfig:
    debounce_h: int = 3          # score must stay above threshold this many hours to raise an alert
    clear_h: int = 24            # ...and below it this long before the alert is closed
    realert_h: int = 168         # a still-active alert re-notifies (new work request) after 7 days
    credit_window_h: int = 336   # an alert raised up to 14 days before a failure counts as catching it
    min_lead_h: int = 12         # an alert raised later than this is too late to plan the repair
    # Alarm budget: at most one nuisance alert per pump every two years. Operators stop trusting
    # a system that cries wolf (the ISA-18.2 / EEMUA 191 alarm-management lesson).
    max_false_alarms_per_asset_year: float = 0.5


@dataclass
class CostConfig:
    """Illustrative Gulf-Coast process-plant numbers (USD). All editable in the dashboard."""

    lost_production_per_h: float = 2500.0
    unplanned_repair_cost: float = 42000.0     # secondary damage: shaft, seal, impeller, expedite
    unplanned_downtime_h: float = 48.0
    planned_repair_cost: float = 14000.0
    planned_downtime_h: float = 8.0
    false_alarm_cost: float = 1800.0           # vibration analyst + mechanic inspection


@dataclass
class TrainConfig:
    """Gradient-boosting settings, chosen by grouped cross-validation on the training fleet only."""

    train_stride_h: int = 3        # hourly rows are highly autocorrelated; train on every 3rd hour
    cv_folds: int = 4
    max_iter: int = 250
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 200
    max_features: float = 0.5      # feature subsampling per split (regularisation)
    l2_regularization: float = 2.0
    rul_interval_coverage: float = 0.8   # conformal P10-P90 target coverage
    random_state: int = 7


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            sim=SimConfig(**d.get("sim", {})),
            labels=LabelConfig(**d.get("labels", {})),
            alerts=AlertConfig(**d.get("alerts", {})),
            costs=CostConfig(**d.get("costs", {})),
            train=TrainConfig(**d.get("train", {})),
        )
