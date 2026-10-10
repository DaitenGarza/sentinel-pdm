"""Turn model outputs into something a reliability engineer can act on.

* ``SIGNAL_INFO``      human names and units for every signal
* ``reference_stats``  robust centre/scale of each signal's deviation-from-baseline on healthy data
* ``driver_zscores``   how unusual each signal is right now (robust z-score vs. healthy fleet)
* ``top_drivers``      the few signals that explain an alert, in plain English
* ``asset_status``     Normal / Watch / Alert / Critical / Offline
* ``RECOMMENDATIONS``  maintenance actions per failure mode (what to check, how to confirm)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SIGNALS

SIGNAL_INFO: dict[str, tuple[str, str]] = {
    "vib_rms": ("Overall vibration (RMS)", "mm/s"),
    "vib_1x": ("1x running-speed vibration", "mm/s"),
    "vib_2x": ("2x running-speed vibration", "mm/s"),
    "vib_hf": ("High-frequency bearing band", "g"),
    "vib_kurtosis": ("Vibration kurtosis (impacting)", ""),
    "bearing_temp": ("Bearing temperature", "°C"),
    "motor_temp": ("Motor winding temperature", "°C"),
    "motor_current": ("Motor current", "A"),
    "current_imbalance": ("Phase current imbalance", "%"),
    "suction_pressure": ("Suction pressure", "bar"),
    "discharge_pressure": ("Discharge pressure", "bar"),
    "flow": ("Flow", "m³/h"),
    "head": ("Pump differential head", "bar"),
    "bearing_rise": ("Bearing temp. rise over ambient", "°C"),
    "motor_rise": ("Winding temp. rise over ambient", "°C"),
    "hyd_eff": ("Hydraulic efficiency proxy", ""),
    "load_pct": ("Process load", "%"),
}

MODE_LABELS = {
    "bearing": "Rolling-element bearing wear",
    "misalignment": "Shaft / coupling misalignment",
    "cavitation": "Suction cavitation / impeller damage",
    "winding": "Motor stator winding deterioration",
    "sudden": "Sudden failure (no precursor)",
    "none": "Healthy",
}

RECOMMENDATIONS: dict[str, dict] = {
    "bearing": {
        "evidence": "Rising high-frequency bearing band and kurtosis (impacting), followed by overall "
                    "vibration and bearing temperature.",
        "actions": [
            "Take a vibration spectrum + envelope (demodulated) reading on the drive-end bearing; "
            "look for BPFO/BPFI defect frequencies.",
            "Check lubrication: grease condition, over/under-greasing, oil level and contamination.",
            "Stage a replacement bearing kit and schedule the change-out inside the RUL window.",
        ],
    },
    "misalignment": {
        "evidence": "Growing 2x running-speed vibration with bearing temperature and motor current creeping up.",
        "actions": [
            "Laser-align the pump/motor coupling; check for soft foot and pipe strain.",
            "Inspect the coupling elements for wear or cracking.",
            "Check baseplate grout and hold-down bolts if misalignment returns quickly.",
        ],
    },
    "cavitation": {
        "evidence": "Falling suction pressure, discharge-pressure pulsation, broadband vibration and lost "
                    "flow/head.",
        "actions": [
            "Check the suction strainer differential pressure and clean it.",
            "Verify NPSH available > NPSH required: suction vessel level, valve fully open, fluid temperature.",
            "Inspect the impeller for cavitation pitting at the next opportunity.",
        ],
    },
    "winding": {
        "evidence": "Winding temperature rising faster than load explains, with growing phase-current imbalance.",
        "actions": [
            "Infrared thermography of the motor and terminal box.",
            "Check supply voltage balance and phase currents; inspect the cooling fan and fins.",
            "Insulation-resistance (megger) / polarization-index test at the next stop.",
        ],
    },
}

STATUS_ORDER = ["Critical", "Alert", "Watch", "Normal", "Offline"]
STATUS_COLORS = {"Critical": "#d62728", "Alert": "#ff7f0e", "Watch": "#e7ba52", "Normal": "#2ca02c",
                 "Offline": "#7f7f7f"}


def reference_stats(dev_features: pd.DataFrame) -> pd.DataFrame:
    """Robust centre (median) and scale (IQR / 1.349) of each ``<signal>_dev`` column on healthy rows."""
    rows = []
    for s in SIGNALS:
        x = dev_features[f"{s}_dev"].to_numpy(dtype=np.float64)
        x = x[~np.isnan(x)]
        q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75]) if x.size else (0.0, 0.0, 1.0)
        rows.append({"signal": s, "center": q50, "scale": max((q75 - q25) / 1.349, 1e-6)})
    return pd.DataFrame(rows).set_index("signal")


def driver_zscores(features: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Robust z-score per signal: how far today's deviation-from-baseline is from healthy behaviour."""
    out = {}
    for s in SIGNALS:
        out[f"z_{s}"] = ((features[f"{s}_dev"] - ref.loc[s, "center"]) / ref.loc[s, "scale"]).astype(np.float32)
    return pd.DataFrame(out, index=features.index)


def top_drivers(zrow: pd.Series | dict, n: int = 3, min_abs_z: float = 2.0) -> list[dict]:
    """Largest |z| signals for one row. Returns dicts with signal, label, unit, z and a sentence."""
    items = []
    for s in SIGNALS:
        z = zrow.get(f"z_{s}") if isinstance(zrow, dict) else zrow.get(f"z_{s}", np.nan)
        if z is None or not np.isfinite(z) or abs(z) < min_abs_z:
            continue
        label, unit = SIGNAL_INFO[s]
        direction = "above" if z > 0 else "below"
        items.append({"signal": s, "label": label, "unit": unit, "z": float(z),
                      "text": f"{label} is {abs(z):.1f}σ {direction} its healthy baseline"})
    items.sort(key=lambda d: -abs(d["z"]))
    return items[:n]


def asset_status(running: bool, risk: float, threshold: float, rul_p50: float | None = None,
                 anomaly: float | None = None) -> str:
    if not running:
        return "Offline"
    if not np.isfinite(risk):
        return "Normal"
    if risk >= threshold and rul_p50 is not None and np.isfinite(rul_p50) and rul_p50 <= 72:
        return "Critical"
    if risk >= threshold:
        return "Alert"
    if risk >= threshold / 2 or (anomaly is not None and np.isfinite(anomaly) and anomaly >= 1.5):
        return "Watch"
    return "Normal"


def recommendation(mode: str, status: str, rul_p10: float | None = None) -> dict:
    """Maintenance recommendation for the most likely failure mode at a given status."""
    rec = RECOMMENDATIONS.get(mode, {"evidence": "", "actions": []})
    if status == "Critical":
        when = "Plan intervention within 24-48 h; prepare the standby pump / bypass now."
    elif status == "Alert":
        horizon = f" (P10 remaining life ≈ {rul_p10:.0f} h)" if rul_p10 is not None and np.isfinite(rul_p10) else ""
        when = f"Schedule repair inside the next planning window{horizon}; order parts now."
    elif status == "Watch":
        when = "Increase monitoring: route-based vibration reading this week, review trend daily."
    else:
        when = "No action. Continue normal monitoring."
    return {"mode": mode, "mode_label": MODE_LABELS.get(mode, mode), "when": when,
            "evidence": rec["evidence"], "actions": rec["actions"] if status in ("Critical", "Alert", "Watch") else []}
