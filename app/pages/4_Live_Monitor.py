"""Live Monitor: replay a held-out pump hour by hour, the way the control room would have seen it.

The point of a replay is to make the *timeline* of predictive maintenance tangible: the quiet weeks,
the first alert, the work order with the likely failure mode, the remaining-life estimate shrinking,
and finally the trip (with or without a warning). It is also an honesty check, so the page never
peeks at the future: every panel is built only from data with ``hour <= cursor``. Alert raises
come from ``alerts.csv`` but are shown only once their raise hour has passed, and an alert's
active span is clipped at the cursor.

Playback uses ``st.fragment(run_every=...)``: while playing, only the monitor panel re-runs once a
second and advances the cursor; when paused, ``run_every`` is ``None`` so nothing refreshes. The
replay pauses itself when the pump trips, so the outcome is not scrolled past.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # app/ folder
import common as C

C.page_setup("Live Monitor", "📡")
C.require_artifacts()

import html  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from sentinel_pdm.config import Config  # noqa: E402
from sentinel_pdm.features import SIGNALS  # noqa: E402

# --------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------
M = C.load_metrics()
CFG = Config.from_dict(M.get("config", {}))
THR = float(M.get("thresholds", {}).get("hybrid", 0.5))
VERSION = str(M.get("generated_at", ""))
LEAD_IN_H = 21 * 24                       # default replay start: 3 weeks before the chosen failure
TRACE_H = 21 * 24                         # risk trace window
SPARK_H = 7 * 24                          # sensor sparkline window
TICK_S = 1.0                              # seconds between auto-advances while playing
SPEEDS = [1, 2, 3, 6, 12, 24]             # hours advanced per tick
SPARKS = ["vib_rms", "vib_hf", "vib_2x", "bearing_temp", "motor_temp", "current_imbalance",
          "suction_pressure", "flow"]     # one or more leading indicators for every failure mode
PROB_COLS = [f"prob_{m}" for m in C.FAILURE_MODES]
PRED_COLS = ["asset_id", "hour", "timestamp", "running", "risk", "rul_p10", "rul_p50", "rul_p90", *PROB_COLS,
             "mspc_score", "pred_mode", "status", *[f"z_{s}" for s in SIGNALS]]


# --------------------------------------------------------------------------------------------
# data (cached per pump)
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading pump history...")
def pump_history(asset_id: str, version: str) -> pd.DataFrame:
    """One pump's hourly model outputs joined with its cleaned sensor values."""
    p = C.load_predictions(PRED_COLS)
    p = p[p["asset_id"] == asset_id]
    s = C.load_clean(["asset_id", "hour", *SPARKS])
    s = s[s["asset_id"] == asset_id].drop(columns="asset_id")
    out = p.merge(s, on="hour", how="left").sort_values("hour").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


@st.cache_data(show_spinner=False)
def test_pumps(version: str) -> pd.DataFrame:
    assets = C.load_assets()
    return assets[assets["split"] == "test"].sort_values("asset_id").reset_index(drop=True)


def pump_events(asset_id: str) -> pd.DataFrame:
    ev = C.load_events()
    return ev[ev["asset_id"] == asset_id].sort_values("hour").reset_index(drop=True)


def pump_raises(asset_id: str) -> pd.DataFrame:
    """Sentinel (hybrid) alert raises for the pump, flagged when they continue an earlier raise."""
    al = C.load_alerts()
    al = al[(al["asset_id"] == asset_id) & (al["method"] == "hybrid") & (al["split"] == "test")]
    al = al.sort_values("start_hour").reset_index(drop=True)
    prev_end = al["end_hour"].shift(1)
    al["continues"] = (al["start_hour"] == prev_end + 1).fillna(False).astype(bool)
    nxt = al["continues"].shift(-1).fillna(False).astype(bool)
    # an alert closes once the score has stayed below threshold for clear_h hours
    al["closed_at"] = np.where(nxt, np.nan, al["end_hour"] + CFG.alerts.clear_h)
    return al


def hybrid_outcomes() -> pd.DataFrame:
    fo = C.load_failure_outcomes()
    return fo[(fo["method"] == "hybrid") & (fo["split"] == "test")].reset_index(drop=True)


def showcase_failure() -> pd.Series | None:
    """A representative caught failure: lead time closest to the fleet median (not the best case)."""
    fo = hybrid_outcomes()
    caught = fo[fo["caught"].astype(bool) & (fo["hour"] >= LEAD_IN_H)]
    if caught.empty:
        return None
    med = caught["lead_h"].median()
    return caught.loc[(caught["lead_h"] - med).abs().idxmin()]


# --------------------------------------------------------------------------------------------
# replay state
# --------------------------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("lm_playing", False)


def _advance(hours: int) -> bool:
    """Move the cursor forward; stop on a trip or at the end of the record. Returns True if it paused."""
    cur = int(ss["lm_cursor"])
    target = min(cur + hours, int(ss["lm_end"]))
    trips = [f for f in ss.get("lm_trip_hours", []) if cur < f <= target]
    if trips:
        target = trips[0]
    ss["lm_cursor"] = target
    if trips or target >= int(ss["lm_end"]):
        ss["lm_playing"] = False
        return True
    return False


def _play() -> None:
    ss["lm_playing"] = True


def _pause() -> None:
    ss["lm_playing"] = False


def _step(hours: int) -> None:
    ss["lm_playing"] = False
    _advance(hours)


def _reset() -> None:
    ss["lm_playing"] = False
    ss["lm_cursor"] = int(ss["lm_start"])


# --------------------------------------------------------------------------------------------
# setup (sidebar)
# --------------------------------------------------------------------------------------------
pumps = test_pumps(VERSION)
if pumps.empty:
    st.warning("No held-out test pumps in this run.")
    st.stop()
outcomes = hybrid_outcomes()
fail_count = outcomes.groupby("asset_id").size()
show = showcase_failure()
pump_ids = pumps["asset_id"].tolist()
default_pump = pump_ids.index(show["asset_id"]) if show is not None and show["asset_id"] in pump_ids else 0
pump_info = pumps.set_index("asset_id")


def pump_label(a: str) -> str:
    r = pump_info.loc[a]
    n = int(fail_count.get(a, 0))
    return f"{a} · {r['service']} · {n} failure{'s' if n != 1 else ''}"


with st.sidebar:
    st.subheader("Replay setup")
    asset = st.selectbox("Held-out pump", pump_ids, index=default_pump, format_func=pump_label, key="lm_asset")
    hist = pump_history(asset, VERSION)
    events = pump_events(asset)
    record_end = int(hist["hour"].max())

    starts: list[tuple[str, int]] = []
    for e in events.itertuples():
        when = str(e.timestamp)[:10]
        if e.event_type == "failure":
            starts.append((f"3 weeks before the {e.mode} failure ({when})", max(int(e.hour) - LEAD_IN_H, 0)))
        elif e.mode != "none":
            starts.append((f"3 weeks before PM that found {e.mode} ({when})", max(int(e.hour) - LEAD_IN_H, 0)))
    starts.append(("Start of the record", 0))
    starts.append(("Custom day...", -1))
    default_start = 0
    pump_fo = outcomes[outcomes["asset_id"] == asset]
    caught = pump_fo[pump_fo["caught"].astype(bool)]
    if show is not None and show["asset_id"] == asset:
        pick = show
    elif not caught.empty:
        pick = caught.iloc[0]
    elif not pump_fo.empty:
        pick = pump_fo.iloc[0]
    else:
        pick = None
    if pick is not None:
        default_start = next((i for i, (_, h) in enumerate(starts) if h == max(int(pick["hour"]) - LEAD_IN_H, 0)), 0)
    start_i = st.selectbox("Start the replay", range(len(starts)), index=default_start,
                           format_func=lambda i: starts[i][0], key=f"lm_start_{asset}")
    start_hour = starts[start_i][1]
    if start_hour < 0:
        day = st.slider("Start day of the year", 0, max(record_end // 24, 0), 0, key=f"lm_day_{asset}")
        start_hour = int(day) * 24
    speed = st.select_slider("Playback speed", options=SPEEDS, value=6, key="lm_speed",
                             format_func=lambda h: f"{h} h per second")
    st.caption("The replay pauses automatically when the pump trips.")

anchor = (asset, start_hour)
if ss.get("lm_anchor") != anchor:          # new pump or new start point: rewind and pause
    ss["lm_anchor"] = anchor
    ss["lm_cursor"] = start_hour
    ss["lm_playing"] = False
ss["lm_start"] = start_hour
ss["lm_end"] = record_end
ss["lm_trip_hours"] = sorted(int(h) for h in events.loc[events["event_type"] == "failure", "hour"])
ss["lm_cursor"] = int(min(max(int(ss["lm_cursor"]), 0), record_end))

# --------------------------------------------------------------------------------------------
# header + transport controls
# --------------------------------------------------------------------------------------------
info = pump_info.loc[asset]
st.title("📡 Live Monitor")
st.caption(
    f"**{asset}** · {info['service']} · {info['unit']} · criticality {info['criticality']} · "
    f"{info['motor_kw']:.0f} kW{' · VFD' if bool(info['vfd']) else ''}. A held-out pump replayed hour by hour: "
    "only data up to the replay clock is shown, exactly what the control room would have had."
)
playing = bool(ss["lm_playing"])
at_end = int(ss["lm_cursor"]) >= record_end
b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 3])
b1.button("Play", icon=":material/play_arrow:", on_click=_play, disabled=playing or at_end, key="lm_play",
          width="stretch", type="primary")
b2.button("Pause", icon=":material/pause:", on_click=_pause, disabled=not playing, key="lm_pause", width="stretch")
b3.button(f"Step {speed} h", icon=":material/skip_next:", on_click=_step, args=(speed,), disabled=at_end,
          key="lm_step", width="stretch")
b4.button("Reset", icon=":material/replay:", on_click=_reset, key="lm_reset", width="stretch")
b5.caption("▶ playing" if playing else "⏸ paused" + (" (end of record)" if at_end else ""))


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------
def gauge(risk: float, status: str) -> go.Figure:
    color = C.STATUS_COLORS.get(status, "#9ca3af")
    running = np.isfinite(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if running else "gauge",
        value=100 * risk if running else 0,
        number=dict(suffix=" %", valueformat=".0f", font=dict(size=40, color=color)),
        title=dict(text="Failure risk, next 7 days", font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%", tickwidth=1),
            bar=dict(color=color if running else "rgba(0,0,0,0)", thickness=0.6),
            bgcolor="rgba(255,255,255,0.03)", borderwidth=0,
            steps=[dict(range=[0, 50 * THR], color="rgba(44,160,44,0.15)"),
                   dict(range=[50 * THR, 100 * THR], color="rgba(231,186,82,0.22)"),
                   dict(range=[100 * THR, 100], color="rgba(255,127,14,0.14)")],
            threshold=dict(line=dict(color="#e5e7eb", width=3), thickness=0.85, value=100 * THR),
        ),
    ))
    if not running:
        fig.add_annotation(text="Offline", x=0.5, y=0.2, showarrow=False, font=dict(size=26, color=color))
    # Side margins wide enough for the "0%" / "100%" end ticks of the dial.
    return C.style_fig(fig, height=230, legend_top=False).update_layout(margin=dict(l=40, r=48, t=50, b=5))


def risk_trace(past: pd.DataFrame, raises: pd.DataFrame, trips: pd.DataFrame, cursor: int) -> go.Figure:
    lo = max(cursor - TRACE_H, int(past["hour"].min()))
    w = past[past["hour"] >= lo]
    fig = go.Figure()
    for r in raises.itertuples():
        end = min(r.end_hour, cursor)
        if end >= lo:
            fig.add_vrect(x0=C.hour_to_timestamp(max(r.start_hour, lo)), x1=C.hour_to_timestamp(end),
                          fillcolor=C.STATUS_COLORS["Alert"], opacity=0.12, line_width=0, layer="below")
    fig.add_scatter(x=w["timestamp"], y=w["risk"], mode="lines", name="Sentinel risk", connectgaps=False,
                    line=dict(color=C.ACCENT, width=2), hovertemplate="%{x|%b %d %H:%M}<br>risk %{y:.1%}<extra></extra>")
    fig.add_hline(y=THR, line=dict(color="#e5e7eb", dash="dash", width=1),
                  annotation_text=f"alert {THR:.0%}", annotation_position="top left")
    fig.add_hline(y=THR / 2, line=dict(color=C.STATUS_COLORS["Watch"], dash="dot", width=1),
                  annotation_text="watch", annotation_position="bottom left")
    for t in trips.itertuples():
        if lo <= t.hour <= cursor:
            ts = C.hour_to_timestamp(t.hour)
            fig.add_vline(x=ts, line=dict(color="#ef4444", width=2))
            fig.add_annotation(x=ts, y=1, yref="paper", text="trip", showarrow=False, yanchor="bottom",
                               font=dict(color="#ef4444"))
    fig.update_layout(title="Risk, last 21 days (shaded: alert active)", showlegend=False,
                      yaxis=dict(range=[0, 1.02], tickformat=".0%", title=None),
                      xaxis=dict(range=[C.hour_to_timestamp(lo), C.hour_to_timestamp(cursor + 6)]))
    return C.style_fig(fig, height=330, legend_top=False)


def sparklines(past: pd.DataFrame, cursor: int) -> go.Figure:
    w = past[past["hour"] > cursor - SPARK_H]
    titles = []
    for s in SPARKS:
        label, unit = C.SIGNAL_INFO[s]
        last = w[s].dropna()
        val = f"{last.iat[-1]:.2f} {unit}".strip() if len(last) else "no data"
        titles.append(f"{label}<br><b>{val}</b>")
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles, vertical_spacing=0.22, horizontal_spacing=0.06)
    for i, s in enumerate(SPARKS):
        r, c = i // 4 + 1, i % 4 + 1
        fig.add_scatter(x=w["timestamp"], y=w[s], mode="lines", line=dict(color="#60a5fa", width=1.6),
                        connectgaps=False, showlegend=False,
                        hovertemplate="%{x|%b %d %H:%M}<br>%{y:.2f} " + C.SIGNAL_INFO[s][1] + "<extra></extra>",
                        row=r, col=c)
        if s == "vib_rms" and w[s].max() > 0.8 * C.ISO_ZONE_LIMITS["B/C"]:
            for zone in ("B/C", "C/D"):
                fig.add_hline(y=C.ISO_ZONE_LIMITS[zone], line=dict(color="#ef4444" if zone == "C/D" else "#e7ba52",
                                                                   dash="dot", width=1), row=r, col=c)
    fig.update_annotations(font_size=11)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(nticks=3, tickfont=dict(size=10))
    return C.style_fig(fig, height=330, legend_top=False).update_layout(margin=dict(l=10, r=10, t=50, b=5))


def event_log(past: pd.DataFrame, raises: pd.DataFrame, events: pd.DataFrame, start: int, cursor: int) -> list[tuple]:
    """(hour, icon, text) entries between the replay start and the cursor, newest first."""
    rows: list[tuple] = [(start, "⏺", "Replay started")]
    by_hour = past.set_index("hour")
    for r in raises.itertuples():
        if start <= r.start_hour <= cursor:
            at = by_hour.loc[r.start_hour] if r.start_hour in by_hour.index else None
            mode = C.MODE_LABELS.get(at["pred_mode"], at["pred_mode"]).lower() if at is not None else "unknown"
            risk = f", risk {at['risk']:.0%}" if at is not None and np.isfinite(at["risk"]) else ""
            if r.continues:
                rows.append((r.start_hour, "🔁", f"**Alert re-issued** (still active after "
                                                  f"{CFG.alerts.realert_h // 24} days): new work request"))
            else:
                rows.append((r.start_hour, "🔔", f"**Sentinel alert raised**: likely {mode}{risk}"))
        if np.isfinite(r.closed_at) and start <= r.closed_at <= cursor:
            rows.append((int(r.closed_at), "✅", f"Alert closed (risk below threshold for {CFG.alerts.clear_h} h)"))
    maint = []
    for e in events.itertuples():
        maint += [int(e.hour), int(e.end_hour)]
        found = "no defect" if e.mode == "none" else C.MODE_LABELS.get(e.mode, e.mode).lower()
        if start <= e.hour <= cursor:
            if e.event_type == "failure":
                rows.append((int(e.hour), "🛑", f"**Pump tripped**: {C.MODE_LABELS.get(e.mode, e.mode).lower()} "
                                                f"({e.event_id})"))
            else:
                rows.append((int(e.hour), "🧰", f"Preventive maintenance started ({e.event_id}), found {found}"))
        if start <= e.end_hour <= cursor:
            rows.append((int(e.end_hour), "🔧", f"Back in service after {int(e.downtime_h)} h"))
    # process stops / restarts that are not maintenance (operators see these too)
    run = past[["hour", "running"]]
    change = run[run["running"].diff().fillna(0) != 0]
    for h, r in zip(change["hour"], change["running"]):
        if start < h <= cursor and not any(abs(h - m) <= 1 for m in maint):
            rows.append((int(h), "⏯", "Pump restarted" if r == 1 else "Pump stopped by operations"))
    return sorted(rows, key=lambda x: -x[0])


def work_order(now: pd.Series, raised_hour: int) -> str:
    """The open work request. It stays open (with its actions) while the alert is active, even if the
    risk dips below the threshold for a few hours before the alert clears."""
    rec_status = now["status"] if now["status"] in ("Alert", "Critical") else "Alert"
    rec = C.recommendation(now["pred_mode"], rec_status, float(now["rul_p10"]))
    actions = "".join(f"<li>{html.escape(a)}</li>" for a in rec["actions"])
    drivers = C.drivers_text(now)
    p_mode = now.get(f"prob_{now['pred_mode']}", np.nan)
    conf = f" ({p_mode:.0%})" if np.isfinite(p_mode) else ""
    color = C.STATUS_COLORS.get(now["status"], "#ff7f0e")
    return f"""
<div class="sp-card" style="border-left: 4px solid {color};">
  <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
    <div><b>Work request WR-{html.escape(str(now['asset_id']))}-{raised_hour}</b> &nbsp;{C.status_badge(now['status'])}</div>
    <div class="sp-muted">raised {C.hour_to_timestamp(raised_hour):%Y-%m-%d %H:%M} · auto-generated by Sentinel</div>
  </div>
  <div style="margin-top:8px;"><b>Likely failure mode:</b> {html.escape(rec['mode_label'])}{conf}</div>
  <div><b>When:</b> {html.escape(rec['when'])}</div>
  <div class="sp-muted" style="margin-top:4px;"><b>Evidence:</b> {html.escape(rec['evidence'])}
       <br><b>Drivers now:</b> {html.escape(drivers)}</div>
  <ol style="margin:8px 0 0 0;">{actions}</ol>
</div>"""


@st.fragment(run_every=TICK_S if ss["lm_playing"] else None)
def monitor() -> None:
    """The live panel. On timed re-runs (playing) it advances the clock first, then redraws."""
    if ss["lm_playing"] and not ss.pop("lm_full_run", False):
        if _advance(int(ss["lm_speed"])):
            st.rerun(scope="app")            # paused itself: re-run the page so the timer stops
    cursor = int(ss["lm_cursor"])
    start = int(ss["lm_start"])
    past = hist[hist["hour"] <= cursor]
    now = past.iloc[-1]
    raises = pump_raises(asset)
    raises = raises[raises["start_hour"] <= cursor]
    trips = events[events["event_type"] == "failure"]

    status = str(now["status"])
    risk = float(now["risk"])
    elapsed = cursor - start
    st.progress(min(max(elapsed / max(record_end - start, 1), 0.0), 1.0),
                text=f"Replay clock **{now['timestamp']:%a %d %b %Y, %H:%M}** · "
                     f"{elapsed / 24:+.1f} days since replay start")

    # alert state, known at the cursor: last raise not yet closed
    alert_hour = None
    if not raises.empty:
        last = raises.iloc[-1]
        closed = np.isfinite(last["closed_at"]) and last["closed_at"] <= cursor
        if not closed:
            first = last
            for r in raises.iloc[::-1].itertuples():   # walk back over re-issues to the original raise
                first = r
                if not r.continues:
                    break
            alert_hour = int(first.start_hour)

    g1, g2, g3 = st.columns([1.15, 1, 1.35])
    g1.plotly_chart(gauge(risk, status), width="stretch", config={"displayModeBar": False}, key="lm_gauge")
    with g2:
        st.markdown(f"Status &nbsp;{C.status_badge(status)}", unsafe_allow_html=True)
        watch = np.isfinite(risk) and risk >= THR / 2
        if watch and np.isfinite(now["rul_p50"]):
            st.metric("Remaining useful life (P50)", C.fmt_hours(float(now["rul_p50"])),
                      delta=f"P10-P90: {C.fmt_hours(float(now['rul_p10']))} to {C.fmt_hours(float(now['rul_p90']))}",
                      delta_color="off", delta_arrow="off",
                      help="Conditional on a developing failure; capped at 30 days.")
        else:
            st.metric("Remaining useful life (P50)", "-", delta="shown once risk reaches Watch", delta_color="off",
                      delta_arrow="off")
    with g3:
        if watch:
            mode = str(now["pred_mode"])
            p = now.get(f"prob_{mode}", np.nan)
            # The short name fits the big metric font; the full one is in the help and the work request.
            st.metric("Likely failure mode", C.MODE_SHORT.get(mode, mode),
                      delta=f"{p:.0%} of mode probability" if np.isfinite(p) else None, delta_color="off",
                      delta_arrow="off", help=C.MODE_LABELS.get(mode, mode))
        else:
            st.metric("Likely failure mode", "No developing fault" if np.isfinite(risk) else "Pump offline")
        st.caption(f"**Top drivers:** {C.drivers_text(now)}" if np.isfinite(risk) else "No readings while stopped.")

    tripped = trips[(trips["hour"] <= cursor) & (trips["end_hour"] > cursor)]
    if not tripped.empty:
        t = tripped.iloc[-1]
        fo = outcomes[outcomes["event_id"] == t["event_id"]]
        o = fo.iloc[0] if not fo.empty else None
        head = (f"**{asset} tripped at {C.hour_to_timestamp(int(t['hour'])):%d %b %Y %H:%M}**: "
                f"{C.MODE_LABELS.get(t['mode'], t['mode']).lower()} ({t['event_id']}). Out of service "
                f"for {int(t['downtime_h'])} h.")
        if o is not None and bool(o["caught"]):
            st.success(f"{head}  \nSentinel **had warned {C.fmt_hours(float(o['lead_h']))} ahead** (first alert "
                       f"{C.hour_to_timestamp(int(t['hour'] - o['lead_h'])):%d %b %H:%M}): enough time to stage parts "
                       "and plan the repair instead of a breakdown.", icon="✅")
        elif o is not None and bool(o["late_only"]):
            st.warning(f"{head}  \nSentinel alerted, but less than {CFG.alerts.min_lead_h} h before the trip: "
                       "too late to plan.", icon="⚠️")
        elif not bool(t["predictable"]):
            st.error(f"{head}  \n**No warning.** A sudden failure (process upset, foreign object) with no measurable "
                     "precursor: no monitoring system could have seen it coming.", icon="🛑")
        else:
            st.error(f"{head}  \n**Missed:** Sentinel did not raise an actionable alert before this failure.", icon="🛑")
    elif alert_hour is not None and np.isfinite(risk):
        st.markdown(work_order(now, alert_hour), unsafe_allow_html=True)

    t1, t2 = st.columns([2, 1])
    t1.plotly_chart(risk_trace(past, raises, trips, cursor), width="stretch", key="lm_trace")
    with t2:
        st.markdown("**Event log**")
        with st.container(height=300, border=True):
            for h, icon, text in event_log(past, raises, events, start, cursor):
                st.markdown(f"<span class='sp-muted'>{C.hour_to_timestamp(h):%d %b %H:%M}</span> &nbsp;{icon} {text}",
                            unsafe_allow_html=True)
    st.plotly_chart(sparklines(past, cursor), width="stretch", key="lm_sparks")
    st.caption("Key sensors over the last 7 days (cleaned historian data). Vibration lines: ISO 10816-3 zone "
               "boundaries B/C and C/D, shown once vibration approaches them.")


ss["lm_full_run"] = True       # tells the fragment this run came from the page, not the timer
monitor()
