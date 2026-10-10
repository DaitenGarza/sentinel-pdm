"""Physics-informed simulator for a fleet of centrifugal process pumps.

Why simulate? Real run-to-failure data from a plant historian is proprietary, and public
datasets rarely contain the full picture (multi-sensor trends + maintenance history + failure
modes for a whole fleet). The simulator reproduces the things that make real predictive
maintenance hard:

* four failure modes with different physical signatures and degradation speeds
  (rolling-element bearing wear, shaft misalignment, suction-side cavitation, stator winding
  insulation breakdown) plus *sudden* failures that have no measurable precursor;
* operating-point effects: process load swings, daily and seasonal ambient temperature,
  variable-speed (VFD) pumps, pump-curve head/flow coupling and bearing thermal lag;
* process outages where the pump is simply stopped (readings are meaningless);
* instrument problems: dropouts, comms outages, spikes, stuck ("flatlined") tags and slow RTD
  drift that looks like a hot bearing but is not;
* a maintenance history: preventive maintenance (PM) that renews the pump, sometimes catching
  an early degradation before it fails (right-censoring), and repair downtime after failures.

Severity of a degradation follows s(tau) = (exp(k*tau) - 1) / (exp(k) - 1) for tau in [0, 1]:
k > 0 gives the classic P-F curve shape (slow onset, accelerating to functional failure).
Each failure records its **P point** (``p_hour``): the first hour the defect is strong enough
(effective severity >= ``P_THRESHOLD``) that an inspection triggered by an alert would find it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi

from .config import FAILURE_MODES, SENSORS, SimConfig

# Degradation behaviour per failure mode: P-F curve curvature k, duration range (h) and the
# relative frequency among predictable failures (bearings dominate pump failure statistics).
MODE_PARAMS = {
    "bearing": {"k": 3.5, "dur": (300, 1100), "weight": 0.42},
    "misalignment": {"k": 0.9, "dur": (500, 1600), "weight": 0.22},
    "cavitation": {"k": 2.0, "dur": (150, 600), "weight": 0.20},
    "winding": {"k": 2.5, "dur": (250, 900), "weight": 0.16},
}

UNITS = {
    100: ("Feed Preparation", ["Feed charge pump", "Recycle pump", "Slop oil pump"]),
    200: ("Reaction", ["Reactor feed pump", "Quench pump", "Catalyst slurry pump"]),
    300: ("Separation", ["Column reflux pump", "Column bottoms pump", "Side-draw pump"]),
    400: ("Cooling Water", ["CW circulation pump", "CW booster pump", "Tower make-up pump"]),
    500: ("Product Transfer", ["Product rundown pump", "Loading pump", "Tank transfer pump"]),
    600: ("Wastewater", ["Effluent pump", "Sump pump", "Clarifier underflow pump"]),
    700: ("Steam & Condensate", ["Boiler feedwater pump", "Condensate return pump", "Deaerator pump"]),
    800: ("Tank Farm", ["Blending pump", "Truck rack pump", "Tank circulation pump"]),
}


P_THRESHOLD = 0.05  # effective severity at which an inspection can confirm the defect

# Symptom channels whose strength varies from one failure event to the next.
HF, KURT, BROAD, BTEMP, SHAFT, CURR, WIND, HYD = range(8)
N_CHANNELS = 8


@dataclass
class FleetData:
    readings: pd.DataFrame   # hourly sensor data + ground-truth columns (gt_*)
    events: pd.DataFrame     # maintenance log: failures and preventive maintenance
    assets: pd.DataFrame     # asset register with train/test split


def _severity(tau: np.ndarray, k: float) -> np.ndarray:
    return (np.exp(k * tau) - 1.0) / (np.exp(k) - 1.0)


def _lag(x: np.ndarray, tau_h: float) -> np.ndarray:
    """First-order thermal lag (exponential smoothing with time constant tau_h hours)."""
    a = np.exp(-1.0 / tau_h)
    b, den = [1.0 - a], [1.0, -a]
    y, _ = lfilter(b, den, x, zi=lfilter_zi(b, den) * x[0])
    return y


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    e = rng.normal(0.0, sigma, n)
    y, _ = lfilter([1.0], [1.0, -phi], e, zi=[0.0])
    return y


def build_asset_register(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng([cfg.seed, 9999])
    rows = []
    for i in range(cfg.n_assets):
        unit = 100 * (1 + (i // 6) % len(UNITS))
        pump_no = unit + 1 + (i % 6) // 2
        train_letter = "AB"[i % 2]
        suffix = "" if i < 6 * len(UNITS) else f"-{i // (6 * len(UNITS))}"
        unit_name, services = UNITS[unit]
        rows.append(
            {
                "asset_id": f"P-{pump_no}{train_letter}{suffix}",
                "unit": f"U-{unit} {unit_name}",
                "service": services[(i % 6) // 2],
                "criticality": rng.choice(["A", "B", "C"], p=[0.3, 0.5, 0.2]),
                "rated_flow_m3h": float(np.round(rng.uniform(120, 650), -1)),
                "rated_head_bar": float(np.round(rng.uniform(6, 18), 1)),
                "rated_current_a": float(np.round(rng.uniform(60, 220))),
                "motor_kw": 0.0,  # filled below
                "vfd": bool(rng.random() < 0.25),
                "install_year": int(rng.integers(2004, 2022)),
            }
        )
    df = pd.DataFrame(rows)
    df["motor_kw"] = np.round(df["rated_current_a"] * 0.48 * 1.732 * 0.88 * 0.92, 0)  # 480 V
    order = rng.permutation(cfg.n_assets)
    n_test = int(round(cfg.n_assets * cfg.test_fraction))
    df["split"] = "train"
    df.loc[order[:n_test], "split"] = "test"
    return df


def _ambient(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Gulf-Coast-like ambient temperature: seasonal + diurnal cycle + weather noise (degC)."""
    h = np.arange(cfg.hours)
    start = pd.Timestamp(cfg.start)
    doy_h = h + (start.dayofyear - 1) * 24
    seasonal = 21.0 - 8.5 * np.cos(2 * np.pi * (doy_h - 15 * 24) / 8760.0)
    diurnal = 4.5 * np.cos(2 * np.pi * ((h + start.hour) % 24 - 15) / 24.0)
    return seasonal + diurnal + _ar1(rng, cfg.hours, 0.985, 0.35)


def _simulate_asset(
    idx: int, asset: pd.Series, ambient: np.ndarray, cfg: SimConfig
) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng([cfg.seed, idx])
    T = cfg.hours
    h = np.arange(T)

    # ---------------- operating profile ----------------
    base_load = rng.uniform(0.62, 0.92)
    steps = np.zeros(T)
    t = 0
    while t < T:
        t += int(rng.exponential(350)) + 24
        if t < T:
            steps[t:] += rng.normal(0, 0.06)
    steps = np.clip(steps, -0.25, 0.15)
    load = base_load + steps + 0.04 * np.cos(2 * np.pi * (h % 24 - 14) / 24) + _ar1(rng, T, 0.995, 0.006)
    load = np.clip(load, 0.35, 1.05)

    running = np.ones(T, dtype=bool)
    state = np.full(T, "running", dtype=object)
    n_outages = rng.poisson(T / 1500)
    for _ in range(n_outages):
        s = int(rng.integers(0, T))
        running[s : s + int(rng.uniform(4, 36))] = False
    state[~running] = "offline"

    # ---------------- lifecycle: degradation, failures, PM ----------------
    mode_idx = np.zeros(T, dtype=np.int8)        # 0 = healthy, 1..4 FAILURE_MODES, 5 = sudden
    # secondary[i][j]: share of mode i's severity that shows up as mode-j symptoms on this pump
    secondary = np.where(rng.random((4, 4)) < 0.6, rng.uniform(0.15, 0.5, (4, 4)), 0.0)
    sev = np.zeros(T)
    effect_scale = np.ones(T)                    # per-event signature strength
    gains = np.ones((T, N_CHANNELS))             # per-event, per-symptom presentation
    cycle = np.zeros(T, dtype=np.int32)
    events: list[dict] = []
    life_scale = cfg.mean_healthy_hours * rng.uniform(0.7, 1.3)
    modes = list(MODE_PARAMS)
    weights = np.array([MODE_PARAMS[m]["weight"] for m in modes])
    weights /= weights.sum()

    t, c = 0, 0
    next_pm = rng.uniform(0.15, 1.0) * cfg.pm_interval_hours
    while t < T:
        onset = t + int(life_scale * rng.weibull(cfg.weibull_shape)) + 24
        sudden = rng.random() < cfg.sudden_fraction
        mode = "sudden" if sudden else str(rng.choice(modes, p=weights))
        dur = 0 if sudden else int(rng.uniform(*MODE_PARAMS[mode]["dur"]))
        fail_t = onset + dur
        cycle[t:] = c
        if next_pm < min(fail_t, T):
            # Preventive maintenance renews the pump before it fails.
            pm_t = int(next_pm)
            pm_end = pm_t + int(rng.uniform(8, 16))
            found, p_hour = "none", np.nan
            if pm_t > onset and not sudden:
                p_hour = _fill_degradation(mode_idx, sev, effect_scale, gains, onset, pm_t, dur, mode, rng)
                if np.isfinite(p_hour):
                    found = mode  # defect already detectable: the PM crew finds it
            events.append(
                {"event_type": "pm", "hour": pm_t, "end_hour": min(pm_end, T), "mode": found,
                 "onset_hour": onset if pm_t > onset and not sudden else pm_t, "p_hour": p_hour}
            )
            running[pm_t:pm_end] = False
            state[pm_t:pm_end] = "pm"
            t = pm_end
            c += 1
            next_pm = t + cfg.pm_interval_hours * rng.uniform(0.9, 1.1)
            continue
        if onset >= T:
            break
        p_hour = float(fail_t)  # sudden failures: no warning at all
        if not sudden:
            p_hour = _fill_degradation(mode_idx, sev, effect_scale, gains, onset, min(fail_t, T), dur, mode, rng)
            if not np.isfinite(p_hour):
                p_hour = float(fail_t)
        if fail_t >= T:
            break  # degradation still in progress at the end of the record ("now")
        repair = int(rng.uniform(24, 96))
        if sudden:
            mode_idx[max(fail_t - 1, 0)] = 5
        events.append(
            {"event_type": "failure", "hour": fail_t, "end_hour": min(fail_t + repair, T), "mode": mode,
             "onset_hour": onset, "p_hour": p_hour}
        )
        running[fail_t : fail_t + repair] = False
        state[fail_t : fail_t + repair] = "repair"
        t = fail_t + repair
        c += 1
        if next_pm < t:
            next_pm = t + cfg.pm_interval_hours * rng.uniform(0.9, 1.1)

    # ---------------- physics: healthy behaviour + degradation signatures ----------------
    s = {m: np.where(mode_idx == i + 1, sev * effect_scale, 0.0) for i, m in enumerate(FAILURE_MODES)}
    # Secondary damage: real failures rarely have one clean signature (misalignment overloads the
    # bearings, cavitation shakes the coupling, a hot bearing heats the motor end...).
    primary = {m: s[m].copy() for m in FAILURE_MODES}
    for i, m in enumerate(FAILURE_MODES):
        for other in FAILURE_MODES:
            if other != m:
                s[other] += primary[m] * secondary[i][FAILURE_MODES.index(other)]
    sb, sm, sc, sw = s["bearing"], s["misalignment"], s["cavitation"], s["winding"]

    # Per-cycle "new parts" variation of the healthy baseline.
    n_cycles = int(cycle.max()) + 1
    cyc_mult = rng.uniform(0.88, 1.12, size=(n_cycles, 4))[cycle]
    vb = rng.uniform(0.7, 1.15) * cyc_mult[:, 0]
    v1 = rng.uniform(0.35, 0.7) * cyc_mult[:, 1]
    v2 = rng.uniform(0.15, 0.35) * cyc_mult[:, 2]
    hf0 = rng.uniform(0.15, 0.3) * cyc_mult[:, 3]
    rise = rng.uniform(16, 26)
    imb0 = rng.uniform(0.8, 1.8)
    suc0 = rng.uniform(1.5, 4.0)

    flow_r = asset["rated_flow_m3h"]
    head_r = asset["rated_head_bar"]
    i_r = asset["rated_current_a"]
    if asset["vfd"]:
        speed = 3560 * np.clip(0.55 + 0.45 * load, 0.5, 1.0)
    else:
        speed = 3560 * (1 - 0.006 * load) + rng.normal(0, 1.5, T)
    sp = speed / 3560.0

    G = gains
    sch = sc * G[:, HYD]
    flow = flow_r * load * (1 - 0.10 * sch) * (1 + rng.normal(0, 0.015, T))
    suction = suc0 * (1 - 0.6 * np.clip(sch, 0, 1.2)) - 0.15 * load + rng.normal(0, 0.03, T)
    head = head_r * sp**2 * (1.12 - 0.12 * (load / np.maximum(sp, 0.3)) ** 2) * (1 - 0.08 * sch)
    discharge = suction + head + rng.normal(0, 0.05, T) * (1 + 5 * sch)
    current = i_r * (0.32 + 0.68 * load**1.1) * (1 + (0.03 * sm + 0.025 * sb + 0.03 * sw) * G[:, CURR])
    current *= 1 + rng.normal(0, 0.008, T)
    imbalance = imb0 + 7.0 * sw**1.3 * G[:, CURR] + rng.normal(0, 0.15, T)
    motor_temp_target = ambient + 35 + 45 * load**2 + (40 * sw**1.5 + 5 * sm) * G[:, WIND]
    bearing_temp_target = (ambient + rise * (0.55 + 0.45 * load)
                           + (28 * sb**3 + 14 * sm**1.2 + 3 * sc) * G[:, BTEMP])

    vib_1x = v1 * (0.85 + 0.3 * load) + 0.6 * sm * G[:, SHAFT]
    vib_2x = v2 * (0.9 + 0.2 * load) + (2.4 * sm**1.1 + 0.5 * sw) * G[:, SHAFT]
    vib_hf = hf0 + (2.8 * sb**0.7 + 1.6 * sc**1.2) * G[:, HF]
    kurt_bear = np.where(sb < 0.55, 7.0 * sb, 7.0 * 0.55 - 5.0 * (sb - 0.55))  # peaks then falls
    kurt = 3.0 + (np.clip(kurt_bear, 0, None) + 1.2 * sc) * G[:, KURT] + rng.gamma(4.0, 0.06, T)
    broadband = vb * (0.8 + 0.3 * load)
    vib_rms = np.sqrt(broadband**2 + vib_1x**2 + vib_2x**2
                      + ((4.5 * sb**2.2) ** 2 + (2.0 * sc**1.3) ** 2 + (0.4 * sw) ** 2) * G[:, BROAD] ** 2)
    noise = lambda sd: np.exp(rng.normal(0, sd, T))  # noqa: E731 - multiplicative sensor noise
    vib_rms *= noise(0.04)
    vib_1x *= noise(0.06)
    vib_2x *= noise(0.08)
    vib_hf *= noise(0.08)

    # ---------------- stopped pump: readings collapse to idle values ----------------
    off = ~running
    load_out = np.where(off, 0.0, load)
    speed = np.where(off, 0.0, speed)
    flow = np.where(off, 0.0, flow)
    current = np.where(off, 0.0, current)
    imbalance = np.where(off, 0.0, imbalance)
    suction = np.where(off, suc0 + rng.normal(0, 0.03, T), suction)
    discharge = np.where(off, suction + rng.normal(0, 0.03, T), discharge)
    for arr, idle in ((vib_rms, 0.05), (vib_1x, 0.02), (vib_2x, 0.02), (vib_hf, 0.02)):
        arr[off] = idle * np.exp(rng.normal(0, 0.3, off.sum()))
    kurt[off] = 3.0 + rng.normal(0, 0.3, off.sum())
    motor_temp_target = np.where(off, ambient, motor_temp_target)
    bearing_temp_target = np.where(off, ambient, bearing_temp_target)
    motor_temp = _lag(motor_temp_target, 2.0) + rng.normal(0, 0.4, T)
    bearing_temp = _lag(bearing_temp_target, 3.0) + rng.normal(0, 0.3, T)

    data = {
        "vib_rms": vib_rms, "vib_1x": vib_1x, "vib_2x": vib_2x, "vib_hf": vib_hf,
        "vib_kurtosis": kurt, "bearing_temp": bearing_temp, "motor_temp": motor_temp,
        "motor_current": current, "current_imbalance": imbalance,
        "suction_pressure": suction, "discharge_pressure": discharge, "flow": flow,
    }
    _inject_instrument_faults(data, rng, cfg.sensor_fault_rate)

    mode_names = np.array(["none", *FAILURE_MODES, "sudden"], dtype=object)
    data.update(
        {
            "running": running.astype(np.int8),
            "load_pct": np.round(100 * load_out, 2),
            "speed_rpm": np.round(speed, 1),
            "ambient_temp": ambient,
            "gt_mode": mode_names[mode_idx],
            "gt_severity": np.round(sev * (mode_idx > 0), 4),
            "gt_state": state,
        }
    )
    for e in events:
        e["asset_id"] = asset["asset_id"]
        e["predictable"] = e["mode"] not in ("sudden", "none")
        e["downtime_h"] = e["end_hour"] - e["hour"]
    return data, events


def _fill_degradation(mode_idx, sev, effect_scale, gains, onset, end, dur, mode, rng) -> float:
    """Write one degradation event into the arrays; return its P-point hour (NaN if never reached)."""
    if end <= onset:
        return float("nan")
    hours = np.arange(onset, end)
    tau = (hours - onset) / max(dur, 1)
    k = MODE_PARAMS[mode]["k"]
    jitter = 1 + 0.04 * np.cumsum(rng.normal(0, 1, hours.size)) / np.sqrt(np.arange(1, hours.size + 1))
    mode_idx[onset:end] = FAILURE_MODES.index(mode) + 1
    sev[onset:end] = np.clip(_severity(tau, k) * jitter, 0, 1.05)
    effect_scale[onset:end] = rng.uniform(0.4, 1.3)  # some failures announce themselves weakly
    # ...and each one presents differently (e.g. a bearing that never shows a temperature rise)
    gains[onset:end, :] = rng.uniform(0.35, 1.5, N_CHANNELS)
    detectable = np.flatnonzero(sev[onset:end] * effect_scale[onset:end] >= P_THRESHOLD)
    return float(onset + detectable[0]) if detectable.size else float("nan")


def _inject_instrument_faults(data: dict[str, np.ndarray], rng: np.random.Generator, rate: float) -> None:
    T = len(next(iter(data.values())))
    for name, x in data.items():
        # spikes
        n_spk = rng.poisson(T * 0.001 * rate)
        idx = rng.integers(0, T, n_spk)
        if name.endswith("temp"):
            x[idx] += rng.uniform(10, 30, n_spk)
        else:
            x[idx] *= rng.uniform(2.0, 4.0, n_spk)
        # stuck / flatlined tag
        for _ in range(rng.poisson(0.15 * rate)):
            s = int(rng.integers(0, T - 1))
            x[s : s + int(rng.uniform(24, 240))] = x[s]
        # dropouts
        for _ in range(rng.poisson(T * 0.0008 * rate)):
            s = int(rng.integers(0, T))
            x[s : s + 1 + int(rng.geometric(0.25))] = np.nan
    # slow RTD drift on the bearing temperature: looks like a hot bearing, is an instrument issue
    if rng.random() < 0.12 * rate:
        s = int(rng.integers(0, T - 600))
        length = int(rng.uniform(500, 2000))
        e = min(s + length, T)
        data["bearing_temp"][s:e] += np.linspace(0, rng.uniform(6, 15), e - s)
    # plant network / historian outage: every tag missing
    for _ in range(rng.poisson(T * 0.0003 * rate)):
        s = int(rng.integers(0, T))
        n = int(rng.uniform(1, 8))
        for x in data.values():
            x[s : s + n] = np.nan


def simulate_fleet(cfg: SimConfig | None = None) -> FleetData:
    cfg = cfg or SimConfig()
    assets = build_asset_register(cfg)
    ambient = _ambient(cfg, np.random.default_rng([cfg.seed, 7777]))
    timestamps = pd.date_range(cfg.start, periods=cfg.hours, freq="h")

    frames, all_events = [], []
    for i, asset in assets.iterrows():
        data, events = _simulate_asset(i, asset, ambient, cfg)
        df = pd.DataFrame(data)
        df.insert(0, "asset_id", asset["asset_id"])
        df.insert(1, "hour", np.arange(cfg.hours, dtype=np.int32))
        df.insert(2, "timestamp", timestamps)
        frames.append(df)
        all_events.extend(events)

    readings = pd.concat(frames, ignore_index=True)
    float_cols = [*SENSORS, "load_pct", "speed_rpm", "ambient_temp", "gt_severity"]
    readings[float_cols] = readings[float_cols].astype(np.float32)
    for col in SENSORS:
        readings[col] = readings[col].round(3)
    readings["ambient_temp"] = readings["ambient_temp"].round(2)

    events = pd.DataFrame(
        all_events,
        columns=["asset_id", "event_type", "hour", "end_hour", "mode", "onset_hour", "p_hour", "predictable",
                 "downtime_h"],
    )
    events = events.sort_values(["asset_id", "hour"]).reset_index(drop=True)
    events.insert(0, "event_id", [f"WO-{10000 + i}" for i in range(len(events))])
    events["timestamp"] = timestamps[events["hour"].clip(upper=cfg.hours - 1)]
    return FleetData(readings=readings, events=events, assets=assets)
