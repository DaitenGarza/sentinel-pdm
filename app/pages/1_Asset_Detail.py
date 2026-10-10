"""Asset Detail: one pump on one screen, laid out the way a reliability engineer works an alert.

1. **Is it real?** Status, risk, remaining life and the recommended action at the as-of hour.
2. **Since when, and how fast?** The risk timeline next to the maintenance history, the remaining-life
   band and which failure mode the risk is coming from.
3. **Which signals say so?** Driver z-scores and the raw sensor trends (with the data cleaning
   visible), including ISO 10816-3 vibration zones.
4. **Confirm it like an analyst would.** A reconstructed vibration waveform with its spectrum and
   envelope spectrum, marked with the bearing defect frequencies.

Like the Fleet Overview, the page can be replayed at any past hour and never uses data after it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/ folder
import common as C

C.page_setup("Asset Detail", "🔍")
C.require_artifacts()

import zlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from sentinel_pdm import signals as S  # noqa: E402
from sentinel_pdm.config import SENSORS  # noqa: E402
from sentinel_pdm.features import SIGNALS, add_derived_signals  # noqa: E402

METRICS = C.load_metrics()
RUN_ID = str(METRICS.get("generated_at", ""))  # changes on every pipeline run, so caches refresh
THRESHOLDS = METRICS.get("thresholds", {})
THRESHOLD = float(THRESHOLDS.get("hybrid", 0.5))
START = C.hour_to_timestamp(0)

ELEVATED = ("Critical", "Alert", "Watch")
TAB_NAMES = ["📈 Risk & remaining life", "🌡️ Sensor trends", "〰️ Vibration analysis", "🧾 History & data quality"]
WINDOWS = {"30 d": 30, "90 d": 90, "365 d": 365, "All": None}
DEFAULT_SIGNALS = ["vib_rms", "vib_hf", "vib_kurtosis", "bearing_temp"]
TREND_SIGNALS = [*SENSORS, "head", "bearing_rise", "motor_rise", "hyd_eff", "load_pct"]
# The signals a rotating-equipment engineer pulls up first for each failure mode.
MODE_SIGNALS = {
    "bearing": ["vib_hf", "vib_kurtosis", "vib_rms", "bearing_temp"],
    "misalignment": ["vib_2x", "vib_1x", "bearing_temp", "motor_current"],
    "cavitation": ["suction_pressure", "discharge_pressure", "flow", "vib_hf"],
    "winding": ["motor_temp", "motor_rise", "current_imbalance", "motor_current"],
}
ISO_ZONE_FILL = {"A": "#2ca02c", "B": "#9acd32", "C": "#ff7f0e", "D": "#d62728"}
LINE_COLOR = "#93c5fd"
RAW_COLOR = "rgba(209, 213, 219, 0.35)"
FAILURE_COLOR = "#ef4444"
PM_COLOR = "#3b82f6"
ALERT_FILL = "rgba(255, 127, 14, 0.13)"
DOWN_FILL = "rgba(156, 163, 175, 0.10)"


# --------------------------------------------------------------------------------------------
# data (cached per pump: switching between pumps you looked at before is instant)
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def latest_hour(run_id: str) -> int:
    return int(C.load_predictions(["hour"])["hour"].max())


@st.cache_data(show_spinner=False, max_entries=64)
def fleet_at(hour: int, run_id: str) -> pd.DataFrame:
    return C.fleet_snapshot(hour)


@st.cache_data(show_spinner="Loading pump history...", max_entries=16)
def pump_predictions(asset_id: str, run_id: str) -> pd.DataFrame:
    p = C.load_predictions()
    return p[p["asset_id"] == asset_id].sort_values("hour").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=16)
def pump_sensors(asset_id: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cleaned, raw) hourly signals for one pump, both with the physics-derived signals added."""
    out = []
    for frame in (C.load_clean(), C.load_readings()):
        f = frame[frame["asset_id"] == asset_id].sort_values("hour").reset_index(drop=True)
        out.append(pd.concat([f, add_derived_signals(f)], axis=1))
    return out[0], out[1]


def alert_episodes(asset_id: str, as_of: int) -> pd.DataFrame:
    """Sentinel alert periods of one pump known at ``as_of``.

    A still-active alert is re-notified every 7 days, which shows up as back-to-back raises in
    alerts.csv. Those are merged into one continuous period here. Periods still open at ``as_of``
    are cut there, because the future is unknown at that point.
    """
    a = C.load_alerts()
    if a.empty:
        return pd.DataFrame(columns=["start_hour", "end_hour"])
    a = a[(a["asset_id"] == asset_id) & (a["method"] == "hybrid") & (a["start_hour"] <= as_of)]
    a = a.sort_values("start_hour")
    merged: list[list[int]] = []
    for s, e in zip(a["start_hour"], a["end_hour"]):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], int(e))
        else:
            merged.append([int(s), int(e)])
    out = pd.DataFrame(merged, columns=["start_hour", "end_hour"])
    out["end_hour"] = out["end_hour"].clip(upper=as_of)
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def vibration_analysis(speed: float, v1: float, v2: float, hf: float, kurt: float, defect: str, seed: int) -> dict:
    """Reconstruct the waveform and compute both spectra (cached: identical inputs give identical plots)."""
    fs = 25600
    t, x = S.synthesize_waveform(speed, v1, v2, hf, kurt, fs=fs, duration_s=1.0, seed=seed, defect=defect)
    f, amp = S.spectrum(x, fs)
    fe, env = S.envelope_spectrum(x, fs)
    return {"fs": fs, "t": t, "x": x, "f": f, "amp": amp, "fe": fe, "env": env}


# --------------------------------------------------------------------------------------------
# controls: pump + as-of (shared with Fleet Overview via st.session_state) + window
# --------------------------------------------------------------------------------------------
def _hour_of(date, hour_of_day: int) -> int:
    return int((pd.Timestamp(date) + pd.Timedelta(hours=int(hour_of_day)) - START) / pd.Timedelta(hours=1))


def _sync_as_of_widgets(hour: int) -> None:
    ts = C.hour_to_timestamp(hour)
    st.session_state["detail_asof_date"] = ts.date()
    st.session_state["detail_asof_hod"] = ts.hour


def _set_as_of(hour: int, last: int) -> None:
    hour = int(np.clip(hour, 0, last))
    st.session_state["as_of_hour"] = hour
    _sync_as_of_widgets(hour)


def _on_pick_as_of(last: int) -> None:
    _set_as_of(_hour_of(st.session_state["detail_asof_date"], st.session_state["detail_asof_hod"]), last)


def _on_pick_asset() -> None:
    st.session_state["asset_id"] = st.session_state["detail_asset"]


def _show_mode_signals(mode: str) -> None:
    """Button callback: plot the failure mode's signature signals and open the Sensor trends tab."""
    st.session_state["detail_signals"] = MODE_SIGNALS[mode]
    st.session_state["detail_tab"] = TAB_NAMES[1]


def current_as_of(last: int) -> int:
    hour = int(np.clip(st.session_state.get("as_of_hour", last), 0, last))
    st.session_state["as_of_hour"] = hour
    date = st.session_state.get("detail_asof_date")
    if date is None or _hour_of(date, st.session_state.get("detail_asof_hod", 0)) != hour:
        _sync_as_of_widgets(hour)  # first visit, or the hour was changed on another page
    return hour


def current_asset(asset_ids: list[str], as_of: int) -> str:
    """The pump to show: the one chosen on any page, else the riskiest pump at the as-of hour."""
    asset = st.session_state.get("asset_id")
    if asset not in asset_ids:
        snap = fleet_at(as_of, RUN_ID)
        asset = str(snap["asset_id"].iloc[0]) if not snap.empty else asset_ids[0]
        st.session_state["asset_id"] = asset
    if st.session_state.get("detail_asset") != asset:
        st.session_state["detail_asset"] = asset
    return asset


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def fmt_ts(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b %Y, %H:%M")


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and bool(np.isfinite(x))


def ts_of(hours) -> pd.DatetimeIndex:
    return START + pd.to_timedelta(np.asarray(hours, dtype=float), unit="h")


def band_outline(x: pd.DatetimeIndex, lo: np.ndarray, hi: np.ndarray) -> tuple[list, list]:
    """Closed outlines of a lo-hi band, one per run of finite values, separated by None.

    The band has gaps wherever the pump is stopped or not elevated. ``fill="tonexty"`` bridges
    those gaps with large spurious triangles; ``fill="toself"`` on separate outlines keeps them empty.
    """
    idx = np.flatnonzero(np.isfinite(lo) & np.isfinite(hi))
    xs: list = []
    ys: list = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []:
        xs += [*x[run], *x[run][::-1], None]
        ys += [*hi[run], *lo[run][::-1], None]
    return xs, ys


def last_valid(frame: pd.DataFrame, col: str, hour: int, lookback_h: int = 24) -> float:
    """Latest non-missing value of ``col`` at or before ``hour`` (cleaning blanks spikes and gaps)."""
    sel = frame[(frame["hour"] <= hour) & (frame["hour"] > hour - lookback_h)][col].dropna()
    return float(sel.iloc[-1]) if len(sel) else float("nan")


def add_event_markers(fig: go.Figure, events: pd.DataFrame, lo: int, hi: int, annotate: bool = True,
                      **subplot) -> None:
    """Failures (red) and PMs (blue) as vertical lines, with downtime lightly shaded."""
    ev = events[(events["hour"] >= lo) & (events["hour"] <= hi)]
    for _, e in ev.iterrows():
        x = C.hour_to_timestamp(int(e["hour"]))
        is_failure = e["event_type"] == "failure"
        fig.add_vrect(x0=x, x1=C.hour_to_timestamp(int(min(e["end_hour"], hi))), fillcolor=DOWN_FILL,
                      line_width=0, layer="below", **subplot)
        label = (f"✖ {C.MODE_SHORT.get(e['mode'], e['mode'])} failure" if is_failure
                 else "PM" + (f" (found {C.MODE_SHORT.get(e['mode'], e['mode']).lower()})" if e["mode"] != "none" else ""))
        fig.add_vline(x=x, line=dict(color=FAILURE_COLOR if is_failure else PM_COLOR, width=1.5 if is_failure else 1,
                                     dash="solid" if is_failure else "dot"), **subplot)
        if annotate:
            fig.add_annotation(x=x, y=1.0, yref="paper", text=label, showarrow=False, xanchor="left", yanchor="top",
                               xshift=3, font=dict(size=10, color=FAILURE_COLOR if is_failure else PM_COLOR))


def add_alert_shading(fig: go.Figure, episodes: pd.DataFrame, lo: int, hi: int, **subplot) -> None:
    for s, e in zip(episodes["start_hour"], episodes["end_hour"]):
        if e < lo or s > hi:
            continue
        fig.add_vrect(x0=C.hour_to_timestamp(max(s, lo)), x1=C.hour_to_timestamp(min(e, hi) + 1),
                      fillcolor=ALERT_FILL, line_width=0, layer="below", **subplot)


# --------------------------------------------------------------------------------------------
# page: controls
# --------------------------------------------------------------------------------------------
LAST = latest_hour(RUN_ID)
assets = C.load_assets()
asset_ids = sorted(assets["asset_id"].astype(str)) if not assets.empty else sorted(
    C.load_predictions(["asset_id"])["asset_id"].unique())
asset_info = assets.set_index("asset_id") if not assets.empty else pd.DataFrame(index=asset_ids)

as_of = current_as_of(LAST)
asset_id = current_asset(asset_ids, as_of)


def _asset_label(a: str) -> str:
    if a not in asset_info.index or "service" not in asset_info:
        return a
    r = asset_info.loc[a]
    held_out = " · held-out" if r.get("split") == "test" else ""
    return f"{a} · {r['service']} ({r['unit']}){held_out}"


# The title goes first: a widget row at the very top of the page slides its labels under the
# Streamlit toolbar. The pump and as-of hour are already settled (callbacks run before the script).
info = asset_info.loc[asset_id] if asset_id in asset_info.index else pd.Series(dtype=object)
c_title, c_back = st.columns([5, 1], vertical_alignment="center")
c_title.title(f"🔍 {asset_id}" + (f" · {info['service']}" if "service" in info else ""))
try:
    c_back.page_link("Home.py", label="Fleet overview", icon="⬅️")
except StreamlitPageNotFoundError:  # page opened on its own (e.g. under AppTest): no Home page registered
    pass

c_pump, c_date, c_hod, c_now, c_win = st.columns([3.2, 1.6, 1.0, 0.8, 1.9], vertical_alignment="bottom")
c_pump.selectbox("Pump", asset_ids, key="detail_asset", format_func=_asset_label, on_change=_on_pick_asset)
c_date.date_input("As of", key="detail_asof_date", min_value=START.date(),
                  max_value=C.hour_to_timestamp(LAST).date(), format="YYYY-MM-DD", on_change=_on_pick_as_of,
                  args=(LAST,), help="Replay the pump at any past date. Nothing after this hour is used.")
c_hod.selectbox("Hour", list(range(24)), key="detail_asof_hod", format_func=lambda h: f"{h:02d}:00",
                on_change=_on_pick_as_of, args=(LAST,))
c_now.button("Now", on_click=_set_as_of, args=(LAST, LAST), disabled=as_of >= LAST, width="stretch",
             help="Jump to the latest historian hour.")
window_label = c_win.segmented_control("Window", list(WINDOWS), default="90 d", key="detail_window", required=True,
                                       persist_state="session")

days = WINDOWS.get(window_label or "90 d")
lo = 0 if days is None else max(0, as_of - days * 24 + 1)
hi = as_of
as_of_ts = C.hour_to_timestamp(as_of)

preds = pump_predictions(asset_id, RUN_ID)
if preds.empty:
    st.error(f"No predictions found for {asset_id}.")
    st.stop()
clean, raw = pump_sensors(asset_id, RUN_ID)
events_all = C.load_events()
events = (events_all[events_all["asset_id"] == asset_id].sort_values("hour") if not events_all.empty
          else pd.DataFrame(columns=["event_id", "event_type", "hour", "end_hour", "mode", "predictable"]))
events_known = events[events["hour"] <= as_of]
episodes = alert_episodes(asset_id, as_of)
win = preds[(preds["hour"] >= lo) & (preds["hour"] <= hi)]
by_hour = preds.set_index("hour")

row_now = by_hour.loc[as_of] if as_of in by_hour.index else None
scored = preds[(preds["hour"] <= as_of) & preds["risk"].notna()]
ref = scored.iloc[-1] if len(scored) else None          # last scored (running) hour at or before as-of
ref_hour = int(ref["hour"]) if ref is not None else None
status_now = str(row_now["status"]) if row_now is not None else "Offline"
is_running = status_now != "Offline"

# --------------------------------------------------------------------------------------------
# page: header
# --------------------------------------------------------------------------------------------
split_note = ("Held-out test pump: the models never saw it" if info.get("split") == "test"
              else "Training pump: scores are out-of-fold (from models that never saw this pump)")
facts = [str(info[k]) for k in ("unit",) if k in info]
if "criticality" in info:
    facts.append(f"criticality {info['criticality']}")
if "motor_kw" in info:
    facts.append(f"{info['motor_kw']:.0f} kW motor" + (" on VFD" if bool(info.get("vfd")) else ""))
if "install_year" in info:
    facts.append(f"installed {info['install_year']}")
st.markdown(
    f"{C.status_badge(status_now)} &nbsp; <span class='sp-muted'>as of {fmt_ts(as_of_ts)}"
    f"{' (live)' if as_of >= LAST else ' (replay)'} · {' · '.join(facts)} · {split_note}</span>",
    unsafe_allow_html=True,
)

if ref is None:
    st.info("This pump has no scored running hours up to the as-of hour. Pick a later date.")
    st.stop()

if not is_running:
    down = events_known[(events_known["hour"] <= as_of) & (events_known["end_hour"] > as_of)]
    if len(down):
        e = down.iloc[-1]
        mode_down = C.MODE_SHORT.get(e["mode"], e["mode"]).lower()
        why = (f"in repair after a {mode_down} failure" if e["event_type"] == "failure"
               else "down for preventive maintenance")
        reason = f"The pump is {why} since {fmt_ts(C.hour_to_timestamp(int(e['hour'])))} (work order {e['event_id']})."
    else:
        reason = "The pump is stopped by the process (outage or standby), not for maintenance."
    st.info(f"⏸️ {reason} Values below are from the last running hour, {fmt_ts(C.hour_to_timestamp(ref_hour))}.")

status_ref = str(ref["status"])
elevated = status_ref in ELEVATED
mode = str(ref.get("pred_mode", "none"))
mode_prob = float(ref.get(f"prob_{mode}", np.nan)) if mode in C.FAILURE_MODES else float("nan")
prev = by_hour.loc[ref_hour - 24] if (ref_hour - 24) in by_hour.index else None

# Card widths follow their content ("Bearing wear" needs more room than "59%"); every card
# stretches to the row height so the row stays even whether or not a card has a second line.
m = st.columns([1, 1.1, 1.3, 1, 1])
risk_delta = (100 * (ref["risk"] - prev["risk"])) if prev is not None and finite(prev["risk"]) else None
moved = risk_delta is not None and abs(risk_delta) >= 0.5
m[0].metric("7-day failure risk", C.fmt_pct(ref["risk"]),
            delta=(f"{risk_delta:+.0f} pts" if moved else "unchanged") if risk_delta is not None else None,
            delta_color="inverse" if moved else "off", delta_arrow="auto" if moved else "off",
            delta_description="vs 24 h ago" if risk_delta is not None else None, height="stretch",
            help=f"Sentinel hybrid probability of failure within 7 days. Alert at {THRESHOLD:.0%}, Watch at "
                 f"{THRESHOLD / 2:.0%}. Gradient boosting alone: {C.fmt_pct(ref['risk_gbm'])}.")
if elevated and finite(ref["rul_p50"]) and ref["rul_p50"] < 720:
    m[1].metric("Remaining life (P50)", C.fmt_hours(ref["rul_p50"]),
                delta=f"P10–P90: {C.fmt_hours(ref['rul_p10'])} – {C.fmt_hours(ref['rul_p90'])}",
                delta_color="off", delta_arrow="off", height="stretch",
                help="Median remaining useful life with an 80% interval (conformal). Plan the repair before P10.")
else:
    m[1].metric("Remaining life (P50)", "> 30 days", height="stretch",
                help="The RUL model is conditional on a developing failure. It only means something once the "
                     "pump is on Watch or worse.")
if elevated and finite(mode_prob):
    m[2].metric("Likely failure mode", C.MODE_SHORT.get(mode, mode), delta=f"{mode_prob:.0%} probability",
                delta_color="off", delta_arrow="off", height="stretch", help=C.MODE_LABELS.get(mode, mode))
else:
    m[2].metric("Likely failure mode", "-", height="stretch",
                help="Shown when a failure appears to be developing (Watch or worse).")
m[3].metric("MSPC statistic", f"{ref['mspc_score']:.2f}x" if finite(ref["mspc_score"]) else "-",
            delta="of the 99% control limit", delta_color="off", delta_arrow="off", height="stretch",
            help="Multivariate SPC (PCA T² / SPE) against healthy operation. 1.0 = 99% control limit; Watch at 1.5.")
rule = ref.get("rule_score", np.nan)
m[4].metric("Conventional alarms", "Tripped" if finite(rule) and rule >= 1 else "Clear",
            delta=f"{rule:.2f} x limit" if finite(rule) else None, delta_color="off", delta_arrow="off",
            height="stretch",
            help="ISO 10816-3 zone C/D vibration and bearing / winding RTD alarm limits: what a DCS would show "
                 "without Sentinel. They typically trip days later than Sentinel's alert.")

rec = C.recommendation(mode if elevated else "none", status_ref, ref["rul_p10"] if elevated else None)
c_rec, c_drv = st.columns([3, 2], gap="medium")
with c_rec:
    actions = "".join(f"<li>{a}</li>" for a in rec["actions"])
    evidence = (f"<p style='margin:6px 0 0'><b>Typical evidence:</b> {rec['evidence']}</p>"
                if elevated and rec["evidence"] else "")
    title = (f"Recommendation · {rec['mode_label']}" if elevated else "Recommendation")
    st.markdown(
        f"<div class='sp-card'><div class='sp-muted'>{title}</div>"
        f"<p style='margin:6px 0 0'><b>When:</b> {rec['when']}</p>{evidence}"
        + (f"<p style='margin:8px 0 2px'><b>Actions</b></p><ol style='margin-bottom:0'>{actions}</ol>" if actions else "")
        + "</div>",
        unsafe_allow_html=True,
    )
with c_drv:
    items = C.top_drivers(ref, n=4)
    lines = "".join(f"<li>{d['text']}</li>" for d in items) or "<li>No signal is beyond ±2σ of its healthy baseline.</li>"
    st.markdown(f"<div class='sp-card'><div class='sp-muted'>Top drivers</div><ul style='margin:6px 0 0'>{lines}</ul>"
                "</div>", unsafe_allow_html=True)
    if elevated and mode in MODE_SIGNALS:
        st.button(f"Plot the {C.MODE_SHORT[mode].lower()} signature signals", on_click=_show_mode_signals,
                  args=(mode,), help="Sets the Sensor trends tab to the signals that reveal this failure mode.")

# --------------------------------------------------------------------------------------------
# tabs (only the open tab is computed)
# --------------------------------------------------------------------------------------------
t_risk, t_sens, t_vib, t_hist = st.tabs(TAB_NAMES, key="detail_tab", on_change="rerun")
x_range = [C.hour_to_timestamp(lo), C.hour_to_timestamp(hi)]

if t_risk.open is not False:
    with t_risk:
        show_gt = st.toggle("Show simulation ground truth", key="detail_gt", persist_state="page",
                            help="Overlay the simulator's true degradation severity (never seen by the models). "
                                 "This is only possible because the data is simulated.")
        x = ts_of(win["hour"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=x, y=win["risk"], name="Sentinel hybrid risk", line=dict(color=C.METHOD_COLORS["hybrid"],
                        width=2), customdata=win["status"], hovertemplate="%{y:.1%} · %{customdata}<extra>hybrid</extra>")
        fig.add_scatter(x=x, y=win["risk_gbm"], name="Gradient boosting only",
                        line=dict(color=C.METHOD_COLORS["ml"], width=1.2, dash="dot"),
                        hovertemplate="%{y:.1%}<extra>GBM only</extra>")
        if show_gt:
            fig.add_scatter(x=x, y=win["gt_severity"], name="True degradation severity (simulation)",
                            line=dict(color="#c084fc", width=1), fill="tozeroy", fillcolor="rgba(192,132,252,0.12)",
                            customdata=win["gt_mode"], hovertemplate="%{y:.2f} · %{customdata}<extra>ground truth</extra>",
                            secondary_y=True)
            fig.update_yaxes(title_text="Severity (0 = healthy, 1 = failure)", range=[0, 1.35], showgrid=False,
                             secondary_y=True)
        fig.add_hline(y=THRESHOLD, line=dict(color=C.STATUS_COLORS["Alert"], dash="dash", width=1),
                      annotation_text=f"Alert threshold {THRESHOLD:.0%}", annotation_position="top left",
                      annotation_font_size=10)
        fig.add_hline(y=THRESHOLD / 2, line=dict(color=C.STATUS_COLORS["Watch"], dash="dot", width=1),
                      annotation_text="Watch", annotation_position="bottom left", annotation_font_size=10)
        add_alert_shading(fig, episodes, lo, hi)
        add_event_markers(fig, events_known, lo, hi)
        fig.update_yaxes(title_text="P(failure within 7 days)", tickformat=".0%", range=[0, 1.02], secondary_y=False)
        fig.update_xaxes(range=x_range)
        fig.update_layout(hovermode="x unified", title=dict(text="Failure risk timeline", font=dict(size=15)))
        C.style_fig(fig, height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Orange bands: Sentinel alert periods. Red lines: failures (labelled with the logged failure "
                   "mode). Blue dotted lines: preventive maintenance. Grey: pump down for repair or PM. Gaps: pump "
                   "stopped, so not scored.")

        c_rul, c_mode = st.columns(2, gap="medium")
        with c_rul:
            band = win[win["status"].isin(ELEVATED)]
            rul = win[["hour", "rul_p10", "rul_p50", "rul_p90"]].copy()
            rul.loc[~win["status"].isin(ELEVATED).to_numpy(), ["rul_p10", "rul_p50", "rul_p90"]] = np.nan
            # Actual time to failure is shown only for failures that have already happened by the
            # as-of hour: that is what is *known* then.
            ttf = win["ttf"].where((win["hour"] + win["ttf"] <= as_of) & (win["ttf"] <= 720))
            fig = go.Figure()
            xr = ts_of(rul["hour"])
            bx, by = band_outline(xr, rul["rul_p10"].to_numpy(dtype=float), rul["rul_p90"].to_numpy(dtype=float))
            fig.add_scatter(x=bx, y=by, mode="lines", line=dict(width=0), fill="toself",
                            fillcolor="rgba(245,158,11,0.18)", name="P10–P90 interval", hoverinfo="skip")
            fig.add_scatter(x=xr, y=rul["rul_p50"], name="Predicted RUL (P50)",
                            line=dict(color=C.METHOD_COLORS["hybrid"], width=2),
                            customdata=np.stack([rul["rul_p10"], rul["rul_p90"]], axis=1),
                            hovertemplate="P50 %{y:.0f} h (P10 %{customdata[0]:.0f} – P90 %{customdata[1]:.0f})"
                                          "<extra></extra>")
            fig.add_scatter(x=xr, y=ttf, name="Actual time to failure", line=dict(color="#e5e7eb", width=1.5,
                            dash="dash"), hovertemplate="actual %{y:.0f} h<extra></extra>")
            add_event_markers(fig, events_known[events_known["event_type"] == "failure"], lo, hi, annotate=False)
            fig.update_yaxes(title_text="Remaining useful life (h)", range=[0, 740], rangemode="tozero")
            fig.update_xaxes(range=x_range)
            fig.update_layout(hovermode="x unified", title=dict(text="Remaining useful life", font=dict(size=15)))
            C.style_fig(fig, height=330)
            st.plotly_chart(fig, width="stretch")
            if band.empty and ttf.notna().sum() == 0:
                st.caption("No elevated-risk hours in this window, so there is no remaining-life estimate to show. "
                           "RUL is conditional on a developing failure.")
            else:
                st.caption("Shown only while the pump is on Watch or worse (RUL is conditional on a developing "
                           "failure, capped at 30 days). The dashed line is the true time to failure, for failures "
                           "that had already happened by the as-of hour.")

        with c_mode:
            fig = go.Figure()
            for mname in C.FAILURE_MODES:
                prob = win[f"prob_{mname}"]
                share = (win["risk"] * prob).fillna(0.0)
                fig.add_scatter(x=x, y=share, name=C.MODE_SHORT[mname], stackgroup="modes", mode="lines",
                                line=dict(width=0.5, color=C.MODE_COLORS[mname]),
                                customdata=prob, hovertemplate=f"{C.MODE_SHORT[mname]}: %{{y:.1%}} of risk "
                                                               "(P(mode) %{customdata:.0%})<extra></extra>")
            fig.add_hline(y=THRESHOLD, line=dict(color=C.STATUS_COLORS["Alert"], dash="dash", width=1))
            fig.update_yaxes(title_text="7-day risk by failure mode", tickformat=".0%", rangemode="tozero")
            fig.update_xaxes(range=x_range)
            fig.update_layout(hovermode="x unified", title=dict(text="Failure-mode diagnosis", font=dict(size=15)))
            C.style_fig(fig, height=330)
            st.plotly_chart(fig, width="stretch")
            st.caption("The 7-day risk split by the diagnosis model's mode probabilities. The stack height equals "
                       "the hybrid risk, and the colours show which failure mode is driving it.")

        c_z, c_p = st.columns([3, 2], gap="medium")
        with c_z:
            z = pd.Series({s: float(ref.get(f"z_{s}", np.nan)) for s in SIGNALS}).dropna()
            if z.empty:
                st.info("Driver z-scores are not available yet: the pump needs a week of running data after "
                        "maintenance to establish its healthy baseline.")
            else:
                fig = C.zscore_figure(z, f"What is driving it · {fmt_ts(C.hour_to_timestamp(ref_hour))}")
                st.plotly_chart(fig, width="stretch")
                st.caption("Red: above the healthy baseline, blue: below it. Beyond ±2σ (dotted) a signal counts as a "
                           f"driver. Bars stop at ±{C.Z_CLIP:g}σ so small deviations stay readable; the label gives "
                           "the full value. The baseline is the pump's own first week after its last maintenance.")
        with c_p:
            probs = pd.Series({mname: float(ref.get(f"prob_{mname}", np.nan)) for mname in C.FAILURE_MODES}).dropna()
            if probs.empty:
                st.info("No diagnosis available at this hour.")
            else:
                probs = probs.sort_values()
                fig = go.Figure(go.Bar(x=probs.to_numpy(), y=[C.MODE_LABELS[k] for k in probs.index], orientation="h",
                                       marker_color=[C.MODE_COLORS[k] for k in probs.index],
                                       opacity=1.0 if elevated else 0.45, text=[f"{v:.0%}" for v in probs],
                                       textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
                fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title_text="Probability (if a failure is developing)")
                fig.update_layout(title=dict(text="Failure-mode probabilities", font=dict(size=15)))
                C.style_fig(fig, height=260, legend_top=False)
                st.plotly_chart(fig, width="stretch")
                st.caption("Conditional on a failure developing." + ("" if elevated else
                           " The pump is currently healthy, so treat these as indicative only."))

if t_sens.open is not False:
    with t_sens:
        c_sel, c_raw = st.columns([4, 1.3], vertical_alignment="bottom")
        if "detail_signals" not in st.session_state:
            st.session_state["detail_signals"] = DEFAULT_SIGNALS
        chosen = c_sel.multiselect(
            "Signals", TREND_SIGNALS, key="detail_signals", max_selections=8, persist_state="page",
            format_func=lambda s: C.SIGNAL_INFO[s][0] + (f" ({C.SIGNAL_INFO[s][1]})" if C.SIGNAL_INFO[s][1] else ""),
        )
        show_raw = c_raw.toggle("Overlay raw readings", key="detail_raw", persist_state="page",
                                help="Faint markers: the uncleaned historian values, so you can see what the causal "
                                     "cleaning removed (spikes, stuck tags, readings while stopped).")
        if not chosen:
            st.info("Pick at least one signal to plot.")
        else:
            cw = clean[(clean["hour"] >= lo) & (clean["hour"] <= hi)]
            rw = raw[(raw["hour"] >= lo) & (raw["hour"] <= hi)]
            n = len(chosen)
            titles = [C.SIGNAL_INFO[s][0] + (f" · {C.SIGNAL_INFO[s][1]}" if C.SIGNAL_INFO[s][1] else "") for s in chosen]
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.25 / n),
                                subplot_titles=titles)
            fig.update_annotations(font_size=12, xanchor="left", x=0)  # subplot titles, top-left
            xc, xw = ts_of(cw["hour"]), ts_of(rw["hour"])
            removed = 0
            for i, s in enumerate(chosen, start=1):
                if show_raw:
                    fig.add_trace(go.Scattergl(x=xw, y=rw[s], mode="markers", marker=dict(size=3, color=RAW_COLOR),
                                               name="Raw historian value", legendgroup="raw", showlegend=i == 1,
                                               hovertemplate="raw %{y:.3g}<extra></extra>"), row=i, col=1)
                    both = rw[["hour", "running", s]].merge(cw[["hour", s]], on="hour", suffixes=("_raw", ""))
                    removed += int((both["running"].eq(1) & both[f"{s}_raw"].notna() & both[s].isna()).sum())
                fig.add_trace(go.Scatter(x=xc, y=cw[s], mode="lines", line=dict(color=LINE_COLOR, width=1.2),
                                         name="Cleaned", legendgroup="clean", showlegend=i == 1,
                                         hovertemplate="%{y:.3g}<extra>" + s + "</extra>"), row=i, col=1)
                if s == "vib_rms":
                    lim = C.ISO_ZONE_LIMITS
                    top = float(np.nanmax([cw[s].max() if cw[s].notna().any() else 0, lim["C/D"] * 1.25]))
                    bounds = [("A", 0, lim["A/B"]), ("B", lim["A/B"], lim["B/C"]), ("C", lim["B/C"], lim["C/D"]),
                              ("D", lim["C/D"], top * 1.05)]
                    for zone, y0, y1 in bounds:
                        fig.add_hrect(y0=y0, y1=y1, fillcolor=ISO_ZONE_FILL[zone], opacity=0.08, line_width=0,
                                      layer="below", row=i, col=1)
                        fig.add_annotation(x=1.0, xref="x domain" if i == 1 else f"x{i} domain", y=(y0 + y1) / 2,
                                           yref="y" if i == 1 else f"y{i}", text=f"Zone {zone}", showarrow=False,
                                           xanchor="right", font=dict(size=9, color=ISO_ZONE_FILL[zone]))
                    fig.update_yaxes(range=[0, top * 1.05], row=i, col=1)
                add_alert_shading(fig, episodes, lo, hi, row=i, col=1)
                add_event_markers(fig, events_known, lo, hi, annotate=False, row=i, col=1)
            fig.update_xaxes(range=x_range)
            fig.update_layout(hovermode="x unified")
            C.style_fig(fig, height=80 + 175 * n)
            fig.update_layout(legend=dict(x=1, xanchor="right"))   # the first subplot title sits top-left
            st.plotly_chart(fig, width="stretch")
            notes = ["Cleaned hourly values (what the models see)."]
            if "vib_rms" in chosen:
                notes.append("Overall vibration is banded by ISO 10816-3 zones for group 2 machines on rigid "
                             f"foundations: A/B {C.ISO_ZONE_LIMITS['A/B']}, B/C {C.ISO_ZONE_LIMITS['B/C']}, C/D "
                             f"{C.ISO_ZONE_LIMITS['C/D']} mm/s. Zone D is where conventional practice says the "
                             "machine is being damaged.")
            if show_raw:
                notes.append(f"In this window the causal cleaning blanked {removed:,} raw values of the selected "
                             "signals while the pump was running (spikes, stuck tags, impossible values).")
            notes.append("Orange bands: Sentinel alerts. Red / blue lines: failures / PMs.")
            st.caption(" ".join(notes))

if t_vib.open is not False:
    with t_vib:
        st.caption(
            "The historian stores hourly *band values* (1x, 2x, high-frequency band, kurtosis), not raw waveforms. "
            "Below is a physically motivated **reconstruction** of the housing acceleration at this pump's state, for "
            "illustration. It is what a vibration analyst would capture with a portable data collector to confirm a "
            "bearing defect before sending a mechanic: the defect shows up as impacts in the waveform, a resonance "
            "hump in the spectrum and discrete lines at a bearing defect frequency in the envelope spectrum."
        )
        vals = {c: last_valid(clean, c, ref_hour) for c in ("speed_rpm", "vib_1x", "vib_2x", "vib_hf", "vib_kurtosis")}
        if not finite(vals["speed_rpm"]) or vals["speed_rpm"] <= 0:
            st.info("No valid running speed around this hour, so there is nothing to reconstruct. Pick an hour "
                    "when the pump was running.")
        else:
            auto = "random" if (elevated and mode == "cavitation") else "outer"
            options = ["auto", "outer", "inner", "ball", "random"]
            c_src, c_note = st.columns([2, 3], vertical_alignment="bottom")
            choice = c_src.selectbox(
                "Impact source", options, key="detail_defect", persist_state="page",
                format_func=lambda o: f"Auto: {S.DEFECT_LABELS[auto]}" if o == "auto" else S.DEFECT_LABELS[o],
                help="Auto follows the diagnosis: random impacts for cavitation, otherwise an outer-race defect "
                     "(the most common bearing defect under a stationary radial load). Pick another to see how "
                     "a different defect would look.",
            )
            defect = auto if choice == "auto" else choice
            c_note.caption("Healthy kurtosis is about 3. The reconstruction turns only the kurtosis above "
                           f"{S.HEALTHY_KURTOSIS:g} into impacts, so a healthy pump shows a clean, random-looking "
                           "high-frequency band.")

            shaft_hz = vals["speed_rpm"] / 60.0
            freqs = S.DEFAULT_BEARING.defect_frequencies(shaft_hz)
            seed = zlib.crc32(f"{asset_id}:{ref_hour}".encode())
            va = vibration_analysis(vals["speed_rpm"], vals["vib_1x"], vals["vib_2x"], vals["vib_hf"],
                                    vals["vib_kurtosis"], defect, seed)

            # Each card's second line says what the band value reveals, which also keeps the row even.
            v = st.columns(5)
            info_delta = dict(delta_color="off", delta_arrow="off", height="stretch")
            v[0].metric("Shaft speed", f"{vals['speed_rpm']:,.0f} rpm", delta=f"1x = {shaft_hz:.1f} Hz", **info_delta)
            v[1].metric("1x velocity", f"{vals['vib_1x']:.2f} mm/s" if finite(vals["vib_1x"]) else "-",
                        delta="unbalance, bent shaft", **info_delta,
                        help="Vibration at running speed: unbalance, bent shaft, eccentric impeller.")
            v[2].metric("2x velocity", f"{vals['vib_2x']:.2f} mm/s" if finite(vals["vib_2x"]) else "-",
                        delta="misalignment, looseness", **info_delta,
                        help="Vibration at twice running speed: shaft / coupling misalignment, mechanical looseness.")
            v[3].metric("HF bearing band", f"{vals['vib_hf']:.2f} g" if finite(vals["vib_hf"]) else "-",
                        delta="bearing impacts, cavitation", **info_delta,
                        help="High-frequency acceleration band where bearing impacts ring the housing.")
            v[4].metric("Kurtosis", f"{vals['vib_kurtosis']:.1f}" if finite(vals["vib_kurtosis"]) else "-",
                        delta="healthy ≈ 3", **info_delta,
                        help="Impulsiveness of the acceleration waveform. Impacting raises it well above 3.")

            # Time waveform: the first 200 ms of the 1 s capture.
            n_show = int(0.2 * va["fs"])
            fig = go.Figure(go.Scattergl(x=1000 * va["t"][:n_show], y=va["x"][:n_show], mode="lines",
                                         line=dict(color=LINE_COLOR, width=1), hovertemplate="%{x:.1f} ms: %{y:.3f} g"
                                                                                         "<extra></extra>"))
            fig.update_xaxes(title_text="Time (ms)")
            fig.update_yaxes(title_text="Acceleration (g)")
            fig.update_layout(title=dict(text="Time waveform (first 200 ms of a 1 s capture at 25.6 kHz)",
                                         font=dict(size=15)))
            C.style_fig(fig, height=260, legend_top=False)
            st.plotly_chart(fig, width="stretch")

            c_fft, c_env = st.columns(2, gap="medium")
            with c_fft:
                sel = va["f"] <= 5000
                fig = go.Figure(go.Scatter(x=va["f"][sel], y=va["amp"][sel], mode="lines",
                                           line=dict(color=LINE_COLOR, width=1), name="Spectrum",
                                           hovertemplate="%{x:.0f} Hz: %{y:.4f} g<extra></extra>"))
                fig.add_vrect(x0=2000, x1=4500, fillcolor="rgba(245,158,11,0.07)", line_width=0, layer="below",
                              annotation_text="envelope band", annotation_position="top left",
                              annotation_font_size=10)
                # 1x and 2x are a few pixels apart on a 5 kHz axis: one label for the pair.
                for k_ord in (1, 2):
                    fig.add_vline(x=k_ord * shaft_hz, line=dict(color=C.ACCENT, dash="dot", width=1))
                fig.add_annotation(x=2 * shaft_hz, y=1, yref="paper", text="1x, 2x", showarrow=False, xanchor="left",
                                   yanchor="top", xshift=3, font=dict(size=10, color=C.ACCENT))
                fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 5000])
                fig.update_yaxes(title_text="Amplitude (g peak)", rangemode="tozero")
                fig.update_layout(title=dict(text="Acceleration spectrum (FFT, Hann window)", font=dict(size=15)))
                C.style_fig(fig, height=330, legend_top=False)
                st.plotly_chart(fig, width="stretch")
                st.caption("1x and 2x sit at the far left. Bearing impacts do not appear at their own frequency here; "
                           "they excite the housing resonance around 3 kHz. Zoom in with a box-drag.")
            with c_env:
                f_max = max(3.5 * freqs["BPFO"], 1.3 * freqs["BPFI"], 2.4 * freqs["BSF"])
                sel = (va["fe"] > 0) & (va["fe"] <= f_max)
                fig = go.Figure(go.Scatter(x=va["fe"][sel], y=va["env"][sel], mode="lines",
                                           line=dict(color=LINE_COLOR, width=1), name="Envelope spectrum",
                                           hovertemplate="%{x:.1f} Hz: %{y:.4f} g<extra></extra>"))
                markers = [("FTF", freqs["FTF"], "#9ca3af", 1), ("BSF", freqs["BSF"], C.MODE_COLORS["cavitation"], 1),
                           ("BPFO", freqs["BPFO"], C.MODE_COLORS["bearing"], 1),
                           ("2×BPFO", 2 * freqs["BPFO"], C.MODE_COLORS["bearing"], 0.5),
                           ("3×BPFO", 3 * freqs["BPFO"], C.MODE_COLORS["bearing"], 0.5),
                           ("BPFI", freqs["BPFI"], C.MODE_COLORS["winding"], 1)]
                for lab, fx, col, op in markers:
                    fig.add_vline(x=fx, line=dict(color=col, dash="dash", width=1), opacity=op,
                                  annotation_text=lab, annotation_position="top right", annotation_font_size=10,
                                  annotation_font_color=col)
                fig.update_xaxes(title_text="Frequency (Hz)", range=[0, f_max])
                fig.update_yaxes(title_text="Envelope amplitude (g)", rangemode="tozero")
                fig.update_layout(title=dict(text="Envelope spectrum (2–4.5 kHz band, Hilbert demodulation)",
                                             font=dict(size=15)))
                C.style_fig(fig, height=330, legend_top=False)
                st.plotly_chart(fig, width="stretch")

                lines = {"BPFO (outer race)": freqs["BPFO"], "BPFI (inner race)": freqs["BPFI"],
                         "2×BSF (rolling element)": 2 * freqs["BSF"]}
                prom = {k: S.line_prominence(va["fe"], va["env"], fx) for k, fx in lines.items()}
                best = max(prom, key=lambda k: prom[k] if np.isfinite(prom[k]) else -1)
                excess = (vals["vib_kurtosis"] - S.HEALTHY_KURTOSIS) if finite(vals["vib_kurtosis"]) else 0.0
                if np.isfinite(prom[best]) and prom[best] >= 5:
                    st.success(f"**{best}** line stands {prom[best]:.0f}x above the envelope noise floor: the impacts "
                               "repeat at that defect frequency, the classic confirmation of a damaged bearing.")
                elif excess > 0 and defect == "random":
                    st.warning("Impacts are present (raised kurtosis), but no defect frequency stands out in the "
                               "envelope spectrum. Random impacts point to **cavitation** rather than a bearing defect: "
                               "check suction pressure and NPSH margin.")
                else:
                    st.info("No discrete bearing defect line in the envelope spectrum: the high-frequency band is "
                            "random, as for a healthy bearing.")
            st.caption(
                f"Defect frequencies for a 6309-type bearing ({S.DEFAULT_BEARING.n_balls} balls) at "
                f"{vals['speed_rpm']:,.0f} rpm: "
                + " · ".join(f"{k} {fx:.1f} Hz ({fx / shaft_hz:.2f}x)" for k, fx in freqs.items())
                + ". Real lines sit 1-2 % lower because of rolling-element slip."
            )

if t_hist.open is not False:
    with t_hist:
        st.subheader("Maintenance history")
        if events_known.empty:
            st.info("No work orders for this pump up to the as-of hour.")
        else:
            outcomes = C.load_failure_outcomes()
            if not outcomes.empty:
                outcomes = outcomes[outcomes["method"] == "hybrid"].set_index("event_id")

            def warning_text(e: pd.Series) -> str:
                if e["event_type"] != "failure":
                    return (f"PM found developing {C.MODE_SHORT.get(e['mode'], e['mode']).lower()}"
                            if e["mode"] != "none" else "-")
                if not bool(e.get("predictable", True)):
                    return "No warning possible (sudden failure, no precursor)"
                if outcomes.empty or e["event_id"] not in outcomes.index:
                    return "-"
                o = outcomes.loc[e["event_id"]]
                if bool(o["caught"]) and finite(o["lead_h"]):
                    return f"✅ Warned {C.fmt_hours(o['lead_h'])} ahead"
                if bool(o.get("late_only", False)):
                    return "⚠️ Warned too late to plan"
                return "❌ Missed"

            hist = events_known.sort_values("hour", ascending=False)
            table = pd.DataFrame({
                "Work order": hist["event_id"].to_numpy(),
                "Type": np.where(hist["event_type"].eq("failure"), "Failure", "Preventive maintenance"),
                "Failure mode / finding": [C.MODE_LABELS.get(mo, mo) if mo != "none" else "Nothing found"
                                           for mo in hist["mode"]],
                "Down at": [C.hour_to_timestamp(int(h)) for h in hist["hour"]],
                "Back in service": [C.hour_to_timestamp(int(e)) if e <= as_of else pd.NaT for e in hist["end_hour"]],
                "Downtime (h)": [int(min(e, as_of) - h) for h, e in zip(hist["hour"], hist["end_hour"])],
                "Sentinel warning": [warning_text(e) for _, e in hist.iterrows()],
            })
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={
                             "Down at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                             "Back in service": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                             "Downtime (h)": st.column_config.NumberColumn(format="%d"),
                         })
            st.caption("Work orders from the maintenance log (CMMS) up to the as-of hour. A failure counts as "
                       "warned when a Sentinel alert was raised after the defect became detectable and at least "
                       "12 h before the breakdown, early enough to plan the repair.")

        c_q, c_a = st.columns(2, gap="large")
        with c_q:
            st.subheader("Data quality")
            q = C.load_quality()
            qr = q[q["asset_id"] == asset_id] if not q.empty else q
            if qr.empty:
                st.info("No data-quality summary available.")
            else:
                r = qr.iloc[0]
                med = q.median(numeric_only=True)
                qc = st.columns(2)
                qc[0].metric("Running time", f"{r['running_pct']:.1f}%", delta=f"fleet median {med['running_pct']:.1f}%",
                             delta_color="off", delta_arrow="off")
                qc[1].metric("Missing readings", f"{r['missing_pct']:.2f}%",
                             delta=f"fleet median {med['missing_pct']:.2f}%", delta_color="off", delta_arrow="off",
                             help="Share of running-hour sensor values that never arrived (dropouts, network outages).")
                qc = st.columns(2)
                qc[0].metric("Spikes removed", f"{int(r['spikes_removed']):,}",
                             delta=f"fleet median {med['spikes_removed']:.0f}", delta_color="off", delta_arrow="off",
                             help="Values rejected by the rate-of-change check against a trailing median.")
                qc[1].metric("Stuck-tag hours", f"{int(r['stuck_hours']):,}",
                             delta=f"fleet median {med['stuck_hours']:.0f}", delta_color="off", delta_arrow="off",
                             help="Hours a tag repeated exactly the same value (flatlined transmitter) and was blanked.")
                st.caption("Over the full one-year record, across all 12 sensor tags.")
        with c_a:
            st.subheader("Asset register")
            if info.empty:
                st.info("No asset-register entry for this pump.")
            else:
                reg = {
                    "Process unit": info.get("unit", "-"),
                    "Service": info.get("service", "-"),
                    "Criticality": info.get("criticality", "-"),
                    "Rated flow": f"{info['rated_flow_m3h']:.0f} m³/h" if "rated_flow_m3h" in info else "-",
                    "Rated head": f"{info['rated_head_bar']:.1f} bar" if "rated_head_bar" in info else "-",
                    "Motor": (f"{info['motor_kw']:.0f} kW, {info['rated_current_a']:.0f} A rated"
                              if "motor_kw" in info else "-"),
                    "Speed control": "Variable-frequency drive" if bool(info.get("vfd")) else "Fixed speed (DOL)",
                    "Installed": str(info.get("install_year", "-")),
                    "Data split": ("Held-out test pump" if info.get("split") == "test"
                                   else "Training pump (scores out-of-fold)"),
                }
                st.dataframe(pd.DataFrame({"Field": list(reg), "Value": [str(v) for v in reg.values()]}),
                             hide_index=True, width="stretch")
