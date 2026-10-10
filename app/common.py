"""Shared helpers for every dashboard page: page setup, cached data loaders, formatting, styling.

Pages import this module as ``import common as C`` (Streamlit puts the ``app`` folder on the
import path because Home.py lives there).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the package importable even when it has not been pip-installed (e.g. running from a fresh
# clone with only the requirements installed).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sentinel_pdm.config import ARTIFACTS_DIR, DATA_DIR, FAILURE_MODES, ISO_ZONE_LIMITS  # noqa: E402
from sentinel_pdm.explain import (  # noqa: E402
    MODE_LABELS, SIGNAL_INFO, STATUS_COLORS, STATUS_ORDER, asset_status, fmt_sigma, recommendation, top_drivers,
)

APP_NAME = "Sentinel PdM"
TAGLINE = "Fleet predictive maintenance for centrifugal pumps"

# Consistent colours across pages (dark industrial theme, see .streamlit/config.toml).
ACCENT = "#f59e0b"
METHOD_COLORS = {
    "hybrid": "#f59e0b",
    "ml": "#60a5fa",
    "mspc": "#34d399",
    "iforest": "#c084fc",
    "rule": "#9ca3af",
    "reactive": "#ef4444",
}
MODE_COLORS = {"bearing": "#f97316", "misalignment": "#3b82f6", "cavitation": "#14b8a6", "winding": "#a855f7",
               "sudden": "#ef4444", "none": "#6b7280"}
# Short failure-mode names for tight spots (table cells, metric values, chart legends);
# MODE_LABELS holds the full engineering names.
MODE_SHORT = {"bearing": "Bearing wear", "misalignment": "Misalignment", "cavitation": "Cavitation",
              "winding": "Winding", "sudden": "Sudden", "none": "None"}
PLOTLY_TEMPLATE = "plotly_dark"
Z_CLIP = 10.0        # driver bar charts cut at +/-10 sigma (see zscore_figure)

PREDICTIONS = ARTIFACTS_DIR / "predictions.parquet"
METRICS = ARTIFACTS_DIR / "metrics.json"


# --------------------------------------------------------------------------------------------
# page chrome
# --------------------------------------------------------------------------------------------
def page_setup(title: str, icon: str = "⚙️") -> None:
    """Call first on every page."""
    st.set_page_config(page_title=f"{title} · {APP_NAME}", page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
          div[data-testid="stMetric"] {background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                                       border-radius: 10px; padding: 12px 16px;}
          .sp-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:0.85rem;
                     color:#111; }
          .sp-card {border:1px solid rgba(255,255,255,0.10); border-radius:12px; padding:14px 18px;
                    background:rgba(255,255,255,0.025); margin-bottom:10px;}
          .sp-muted {color: rgba(250,250,250,0.6); font-size: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(f"### 🛡️ {APP_NAME}")
        st.caption(TAGLINE)
        m = load_metrics()
        if m:
            st.caption(f"Models trained {m.get('generated_at', '?')[:16].replace('T', ' ')} UTC · v{m.get('version', '?')}")


def artifacts_ready() -> bool:
    return PREDICTIONS.exists() and METRICS.exists()


def require_artifacts() -> None:
    """Stop the page with instructions when the pipeline has not been run yet."""
    if not artifacts_ready():
        st.error("No trained models found yet.")
        st.markdown(
            "Run **`setup.bat`** (Windows) or `python -m sentinel_pdm train` once. It simulates the plant "
            "data and trains every model (about 2-3 minutes). Then refresh this page."
        )
        st.stop()


def _mtime(path: Path) -> float:
    return os.path.getmtime(path) if path.exists() else 0.0


# --------------------------------------------------------------------------------------------
# cached loaders (the mtime argument invalidates the cache after retraining)
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading predictions...")
def _load_parquet(path: str, mtime: float, columns: tuple[str, ...] | None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def load_predictions(columns: list[str] | None = None) -> pd.DataFrame:
    """Hourly model outputs for every pump (see sentinel_pdm.scoring for the column meanings)."""
    return _load_parquet(str(PREDICTIONS), _mtime(PREDICTIONS), tuple(columns) if columns else None)


def load_clean(columns: list[str] | None = None) -> pd.DataFrame:
    """Cleaned sensor data: asset_id, hour, timestamp, running, load_pct, speed_rpm, ambient_temp, sensors."""
    p = DATA_DIR / "clean.parquet"
    return _load_parquet(str(p), _mtime(p), tuple(columns) if columns else None)


def load_readings(columns: list[str] | None = None) -> pd.DataFrame:
    """Raw (uncleaned) historian data incl. ground truth gt_mode / gt_severity / gt_state."""
    p = DATA_DIR / "readings.parquet"
    return _load_parquet(str(p), _mtime(p), tuple(columns) if columns else None)


@st.cache_data
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _csv(path: Path) -> pd.DataFrame:
    return _load_csv(str(path), _mtime(path)) if path.exists() else pd.DataFrame()


def load_events() -> pd.DataFrame:
    """Maintenance log: event_id, asset_id, event_type (failure|pm), hour, end_hour, mode, onset_hour,
    p_hour, predictable, downtime_h, timestamp."""
    return _csv(DATA_DIR / "events.csv")


def load_assets() -> pd.DataFrame:
    """Asset register: asset_id, unit, service, criticality, rated_flow_m3h, rated_head_bar,
    rated_current_a, motor_kw, vfd, install_year, split (train|test)."""
    return _csv(DATA_DIR / "assets.csv")


def load_quality() -> pd.DataFrame:
    return _csv(DATA_DIR / "quality.csv")


def load_alerts() -> pd.DataFrame:
    """Alert raises: asset_id, start_hour, end_hour, outcome, method, split."""
    return _csv(ARTIFACTS_DIR / "alerts.csv")


def load_failure_outcomes() -> pd.DataFrame:
    """One row per (failure, method, split) with caught / lead_h / late_only."""
    return _csv(ARTIFACTS_DIR / "failure_outcomes.csv")


def load_threshold_curve() -> pd.DataFrame:
    return _csv(ARTIFACTS_DIR / "threshold_curve.csv")


def load_feature_importance() -> pd.DataFrame:
    return _csv(ARTIFACTS_DIR / "feature_importance.csv")


@st.cache_data
def _load_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metrics() -> dict:
    return _load_json(str(METRICS), _mtime(METRICS)) if METRICS.exists() else {}


@st.cache_resource(show_spinner="Loading models...")
def _load_bundle(mtime: float):
    from sentinel_pdm.scoring import load_bundle

    return load_bundle()


def load_model_bundle():
    p = ARTIFACTS_DIR / "models" / "bundle.joblib"
    return _load_bundle(_mtime(p))


# --------------------------------------------------------------------------------------------
# fleet helpers
# --------------------------------------------------------------------------------------------
def hour_to_timestamp(hour: int) -> pd.Timestamp:
    m = load_metrics()
    start = m.get("config", {}).get("sim", {}).get("start", "2025-01-01")
    return pd.Timestamp(start) + pd.Timedelta(hours=int(hour))


def fleet_snapshot(hour: int | None = None) -> pd.DataFrame:
    """One row per pump at ``hour`` (default: the latest hour = "now"), joined with the asset register."""
    p = load_predictions()
    if hour is None:
        hour = int(p["hour"].max())
    snap = p[p["hour"] == hour].copy()
    assets = load_assets()
    if not assets.empty:
        snap = snap.merge(assets, on=["asset_id", "split"] if "split" in assets else ["asset_id"], how="left")
    order = {s: i for i, s in enumerate(STATUS_ORDER)}
    snap["_order"] = snap["status"].map(order).fillna(len(order))
    snap = snap.sort_values(["_order", "risk"], ascending=[True, False]).drop(columns="_order")
    return snap.reset_index(drop=True)


def drivers_text(row: pd.Series | dict, n: int = 3) -> str:
    items = top_drivers(row, n=n)
    return "; ".join(f"{d['label']} {fmt_sigma(d['z'])}" for d in items) if items else "-"


# --------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------
def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#888")
    return f'<span class="sp-badge" style="background:{color}">{status}</span>'


def fmt_hours(h: float | None) -> str:
    if h is None or not np.isfinite(h):
        return "-"
    if h >= 720:
        return "> 30 days"
    if h >= 48:
        return f"{h / 24:.1f} days"
    return f"{h:.0f} h"


def fmt_money(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:.0f}"


def fmt_pct(x: float | None, digits: int = 0) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def style_fig(fig, height: int | None = None, legend_top: bool = True):
    """Apply the shared Plotly look (dark template, tight margins).

    Plotly puts both the title and a top legend into the same top margin, where they overprint
    each other. A titled figure therefore gets its title pinned to the top-left corner and, when
    the legend sits on top, a taller margin so the legend fits between title and plot.
    """
    titled = bool(fig.layout.title.text)
    top = 78 if titled and legend_top else 40
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=top, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if titled:
        fig.update_layout(title=dict(x=0, xref="container", xanchor="left", y=1, yref="container",
                                     yanchor="top", pad=dict(l=12, t=8)))
    if height:
        fig.update_layout(height=height)
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def zscore_figure(z: pd.Series, title: str, clip: float = Z_CLIP, height: int | None = None) -> go.Figure:
    """Horizontal bars of driver z-scores (index: signal names), largest |z| at the top.

    A developed defect can sit hundreds of sigma away from a pump's tight healthy baseline, which
    would squash every other bar and the +/-2 sigma driver limits into a sliver. Bars are cut at
    +/-``clip`` instead, and each label carries the true value.
    """
    z = z.reindex(z.abs().sort_values(ascending=True).index)
    vals = z.to_numpy(dtype=float)
    colors = np.where(vals >= 2, "#ef4444", np.where(vals <= -2, "#60a5fa", "#6b7280"))
    labels = [SIGNAL_INFO[s][0] + (f" ({SIGNAL_INFO[s][1]})" if SIGNAL_INFO[s][1] else "") for s in z.index]
    # Long bars carry their value inside, short ones outside, so a label never spills over the
    # signal names (the chart is also used in a narrow column). constraintext="none" stops Plotly
    # from shrinking inside labels to the bar thickness.
    fig = go.Figure(go.Bar(
        x=np.clip(vals, -clip, clip), y=labels, orientation="h", marker_color=colors,
        text=[fmt_sigma(v) for v in vals],
        textposition=["inside" if abs(v) >= 0.6 * clip else "outside" for v in vals], insidetextanchor="end",
        constraintext="none", insidetextfont=dict(color="#ffffff"), cliponaxis=False, customdata=vals,
        hovertemplate="%{y}: %{customdata:+.1f}σ<extra></extra>",
    ))
    for v in (-2, 2):
        fig.add_vline(x=v, line=dict(color="#9ca3af", dash="dot", width=1))
    ticks = [-clip, -clip / 2, 0, clip / 2, clip]
    ticktext = [f"≤ {t:g}" if t == -clip else f"≥ {t:g}" if t == clip else f"{t:g}" for t in ticks]
    fig.update_xaxes(range=[-1.3 * clip, 1.3 * clip], tickvals=ticks, ticktext=ticktext, zeroline=True,
                     zerolinecolor="#4b5563", title_text="σ from healthy baseline")
    fig.update_layout(title=dict(text=title, font=dict(size=15)), bargap=0.25, showlegend=False)
    return style_fig(fig, height=height or 150 + 20 * len(z), legend_top=False)


__all__ = [
    "ACCENT", "APP_NAME", "FAILURE_MODES", "ISO_ZONE_LIMITS", "METHOD_COLORS", "MODE_COLORS", "MODE_LABELS",
    "MODE_SHORT", "SIGNAL_INFO", "STATUS_COLORS", "STATUS_ORDER", "Z_CLIP", "artifacts_ready", "asset_status",
    "drivers_text", "fleet_snapshot", "fmt_hours", "fmt_sigma", "fmt_money", "fmt_pct", "hour_to_timestamp", "load_alerts",
    "load_assets", "load_clean", "load_events", "load_failure_outcomes", "load_feature_importance", "load_metrics",
    "load_model_bundle", "load_predictions", "load_quality", "load_readings", "load_threshold_curve",
    "page_setup", "recommendation", "require_artifacts", "status_badge", "style_fig", "zscore_figure",
]
