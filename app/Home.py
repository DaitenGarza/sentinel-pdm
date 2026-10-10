"""Fleet Overview: the control-room view of every pump, and the entry point of the dashboard.

A reliability engineer starts a shift here with three questions: which pumps need attention,
how bad is it (risk, remaining life, likely failure mode), and how did we get here (the last 30
days). Every panel can be *replayed* at any past hour, and replay uses only data up to that hour.
That is how you check that the system would have warned you in time, rather than taking the
held-out metrics on trust.

Run with ``streamlit run app/Home.py`` (or ``run.bat``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # app/ folder
import common as C

C.page_setup("Fleet Overview", "🛡️")
C.require_artifacts()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

ASSET_PAGE = "pages/1_Asset_Detail.py"
ELEVATED = ("Critical", "Alert", "Watch")
STATUS_ICON = {"Critical": "🔴", "Alert": "🟠", "Watch": "🟡", "Normal": "🟢", "Offline": "⚫"}
HEATMAP_DAYS = 30
ALERT_LOOKBACK_H = 30 * 24

METRICS = C.load_metrics()
RUN_ID = str(METRICS.get("generated_at", ""))  # changes on every pipeline run, so caches refresh
THRESHOLD = float(METRICS.get("thresholds", {}).get("hybrid", 0.5))
START = C.hour_to_timestamp(0)


# --------------------------------------------------------------------------------------------
# data (cached per as-of hour, so replaying back and forth is instant)
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def latest_hour(run_id: str) -> int:
    return int(C.load_predictions(["hour"])["hour"].max())


@st.cache_data(show_spinner=False, max_entries=64)
def snapshot(hour: int, run_id: str) -> pd.DataFrame:
    """One row per pump at ``hour``, riskiest first (see ``common.fleet_snapshot``)."""
    return C.fleet_snapshot(hour)


@st.cache_data(show_spinner=False, max_entries=32)
def risk_window(hour: int, days: int, run_id: str) -> pd.DataFrame:
    """Hourly risk and status of every pump in the ``days`` before ``hour`` (inclusive)."""
    p = C.load_predictions(["asset_id", "hour", "timestamp", "running", "risk", "status"])
    w = p[(p["hour"] > hour - days * 24) & (p["hour"] <= hour)]
    return w.reset_index(drop=True)


def fleet_trend(window: pd.DataFrame, hour: int, days: int) -> pd.DataFrame:
    """Daily samples (at the as-of hour of day) of fleet-level KPIs, for the metric sparklines."""
    w = window.assign(
        crit=window["status"].eq("Critical"),
        alert=window["status"].eq("Alert"),
        watch=window["status"].eq("Watch"),
        run=window["running"].eq(1),
    )
    hourly = w.groupby("hour").agg(running=("run", "sum"), critical=("crit", "sum"), alert=("alert", "sum"),
                                   watch=("watch", "sum"), expected=("risk", "sum"))
    sample = [h for h in range(hour - days * 24, hour + 1, 24) if h in hourly.index]
    return hourly.loc[sample]


# --------------------------------------------------------------------------------------------
# as-of control (shared with the Asset Detail page through st.session_state["as_of_hour"])
# --------------------------------------------------------------------------------------------
def _hour_of(date, hour_of_day: int) -> int:
    return int((pd.Timestamp(date) + pd.Timedelta(hours=int(hour_of_day)) - START) / pd.Timedelta(hours=1))


def _sync_widgets(hour: int) -> None:
    ts = C.hour_to_timestamp(hour)
    st.session_state["home_asof_date"] = ts.date()
    st.session_state["home_asof_hod"] = ts.hour


def _set_as_of(hour: int, last: int) -> None:
    hour = int(np.clip(hour, 0, last))
    st.session_state["as_of_hour"] = hour
    _sync_widgets(hour)


def as_of_picker(last: int) -> int:
    """Date + hour selector with day-step buttons. Returns the as-of hour index (0..last)."""
    hour = int(np.clip(st.session_state.get("as_of_hour", last), 0, last))
    st.session_state["as_of_hour"] = hour
    if (st.session_state.get("home_asof_date") is None
            or _hour_of(st.session_state["home_asof_date"], st.session_state.get("home_asof_hod", 0)) != hour):
        _sync_widgets(hour)  # first visit, or the hour was changed on another page

    def on_pick() -> None:
        _set_as_of(_hour_of(st.session_state["home_asof_date"], st.session_state["home_asof_hod"]), last)

    c_date, c_hod, c_back, c_fwd, c_now = st.columns([2.2, 1.2, 1, 1, 1], vertical_alignment="bottom")
    c_date.date_input("As of", key="home_asof_date", min_value=START.date(),
                      max_value=C.hour_to_timestamp(last).date(), format="YYYY-MM-DD", on_change=on_pick,
                      help="Replay the fleet at any past date. Every panel only uses data up to this hour.")
    c_hod.selectbox("Hour", list(range(24)), key="home_asof_hod", format_func=lambda h: f"{h:02d}:00",
                    on_change=on_pick)
    c_back.button("◀ 1 day", on_click=_set_as_of, args=(hour - 24, last), disabled=hour == 0,
                  width="stretch")
    c_fwd.button("1 day ▶", on_click=_set_as_of, args=(hour + 24, last), disabled=hour >= last,
                 width="stretch")
    c_now.button("Now", on_click=_set_as_of, args=(last, last), disabled=hour >= last, type="primary",
                 width="stretch")
    return int(st.session_state["as_of_hour"])


# --------------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------------
def fmt_ts(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b %Y, %H:%M")


def short_hours(h: float) -> str:
    """Compact duration for table cells: 27 h, 3.6 d, > 30 d."""
    if not np.isfinite(h):
        return "-"
    if h >= 720:
        return "> 30 d"
    return f"{h / 24:.1f} d" if h >= 48 else f"{h:.0f} h"


def rul_text(row: pd.Series) -> str:
    """RUL P50 (P10-P90). The RUL model is conditional on a developing failure, so it only means
    something once the pump is at least on Watch."""
    if row["status"] == "Offline":
        return "-"
    if row["status"] not in ELEVATED or not np.isfinite(row["rul_p50"]) or row["rul_p50"] >= 720:
        return "> 30 d"
    return f"{short_hours(row['rul_p50'])}  ({short_hours(row['rul_p10'])} – {short_hours(row['rul_p90'])})"


def mode_text(row: pd.Series) -> str:
    if row["status"] not in ELEVATED:
        return "-"
    mode = row.get("pred_mode", "none")
    prob = row.get(f"prob_{mode}", np.nan)
    if mode not in C.FAILURE_MODES or not np.isfinite(prob):
        return "-"
    return f"{C.MODE_SHORT[mode]} ({prob:.0%})"


def fleet_table(snap: pd.DataFrame) -> pd.DataFrame:
    """The control-room table. Kept to what decides the next action, so it fits a laptop screen
    without scrolling sideways: the process unit is in the asset tag (P-5xx = U-500) and the
    chart below the table, the MSPC statistic is on the asset page. The free-text drivers come
    last, so that is the only column that can be cut short (the full text shows on hover)."""
    running = snap["running"].eq(1)
    return pd.DataFrame({
        "Asset": snap["asset_id"],
        "Service": snap.get("service", pd.Series("", index=snap.index)),
        "Crit.": snap.get("criticality", pd.Series("", index=snap.index)),
        "Status": [f"{STATUS_ICON.get(s, '')} {s}" for s in snap["status"]],
        "Risk (7 d)": (100 * snap["risk"]).where(running).round(1),
        "RUL P50 (P10 – P90)": snap.apply(rul_text, axis=1),
        "Likely failure mode": snap.apply(mode_text, axis=1),
        "Split": np.where(snap["split"].eq("test"), "Held-out", "Training"),
        "Top drivers": [C.drivers_text(r, n=2) if r["running"] == 1 else "-" for _, r in snap.iterrows()],
    })


def risk_colorscale(threshold: float) -> list[list]:
    """Heatmap colours that line up with the status bands: calm greens below Watch, then the
    Watch / Alert status colours, deepening to red as the risk climbs."""
    watch = threshold / 2
    eps = 1e-4
    sc = C.STATUS_COLORS
    return [
        [0.0, "#12241a"], [watch - eps, "#2f7d46"],
        [watch, sc["Watch"]], [threshold - eps, sc["Watch"]],
        [threshold, sc["Alert"]], [min(threshold + 0.35, 0.95), sc["Critical"]], [1.0, "#8b0f12"],
    ]


# --------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------
LAST = latest_hour(RUN_ID)

st.title("🛡️ Fleet Overview")
st.markdown(
    "Sentinel watches **48 centrifugal process pumps** around the clock. It forecasts which ones are likely "
    "to fail in the next 7 days, how much life they have left and *why*.  \n"
    "That turns breakdowns into planned repairs. It combines gradient-boosted models of failure risk, remaining "
    "life and failure mode with multivariate statistical process control (MSPC)."
)

as_of = as_of_picker(LAST)
as_of_ts = C.hour_to_timestamp(as_of)
if as_of >= LAST:
    st.caption(f"🟢 **Live view.** Latest historian hour: {fmt_ts(as_of_ts)}.")
else:
    days_back = (LAST - as_of) / 24
    st.caption(f"⏪ **Replay.** The fleet as it looked on {fmt_ts(as_of_ts)}, {days_back:.1f} days before the "
               "latest data. Nothing after this hour is used.")

snap = snapshot(as_of, RUN_ID)
if snap.empty:
    st.warning("No predictions exist for this hour. Pick another date.")
    st.stop()
window = risk_window(as_of, HEATMAP_DAYS, RUN_ID)
trend = fleet_trend(window, as_of, HEATMAP_DAYS)
prev = snapshot(as_of - 24, RUN_ID) if as_of >= 24 else None

# ---------------- KPI row ----------------
running = snap["running"].eq(1)
counts = snap["status"].value_counts()
prev_counts = prev["status"].value_counts() if prev is not None else None
expected = float(np.nansum(snap.loc[running, "risk"]))
econ = METRICS.get("test", {}).get("economics", {})
cost_each = float(econ.get("hybrid", econ.get("reactive", {})).get("unplanned_cost_each", np.nan))
event = METRICS.get("test", {}).get("event", {}).get("hybrid", {})


def _delta(status: str) -> int | None:
    if prev_counts is None:
        return None
    return int(counts.get(status, 0) - prev_counts.get(status, 0))


def _spark(col: str) -> list[float] | None:
    return trend[col].astype(float).tolist() if len(trend) > 1 else None


# Two rows of four: the fleet right now, then what it means (exposure and the validated track
# record). Seven cards in one row truncate their labels and values on a laptop screen.
k = [*st.columns(4), *st.columns(4)]
k[0].metric("Pumps running", f"{int(running.sum())} / {len(snap)}", chart_data=_spark("running"),
            chart_type="area", height="stretch",
            help="Pumps with the motor running at the as-of hour. Stopped pumps (repair, PM, process outage) are "
                 "Offline and not scored.")
for col, status in zip(k[1:4], ("Critical", "Alert", "Watch")):
    col.metric(status, int(counts.get(status, 0)), delta=_delta(status), delta_color="inverse",
               delta_description="vs 24 h ago" if prev_counts is not None else None,
               chart_data=_spark(status.lower()), chart_type="area", height="stretch",
               help={"Critical": "Alert with median remaining life of 72 h or less: plan intervention within 24-48 h.",
                     "Alert": f"Hybrid 7-day failure risk at or above the cost-optimal threshold ({THRESHOLD:.0%}).",
                     "Watch": f"Risk above {THRESHOLD / 2:.0%} (half the threshold) or MSPC beyond 1.5x its "
                              "control limit: watch the trend more closely."}[status])
prev_expected = float(np.nansum(prev.loc[prev["running"].eq(1), "risk"])) if prev is not None else None
expected_spark = _spark("expected")
exp_delta = round(expected - prev_expected, 2) if prev_expected is not None else None
k[4].metric("Expected failures, 7 d", f"{expected:.2f}", delta=exp_delta, delta_color="inverse",
            delta_description="vs 24 h ago" if exp_delta is not None else None,
            chart_data=expected_spark, chart_type="area", height="stretch",
            help="Sum of every running pump's 7-day failure probability: how many breakdowns to expect "
                 "this week if nothing is done.")
if np.isfinite(cost_each):
    cost_delta = exp_delta * cost_each if exp_delta is not None else None
    k[5].metric("Cost exposure, 7 d", C.fmt_money(expected * cost_each),
                delta=(("+" if cost_delta >= 0 else "-") + C.fmt_money(abs(cost_delta))) if cost_delta else None,
                delta_color="inverse", chart_type="area", height="stretch",
                delta_description="vs 24 h ago" if cost_delta else None,
                chart_data=[v * cost_each for v in expected_spark] if expected_spark else None,
                help=f"Expected failures x {C.fmt_money(cost_each)} per unplanned failure (repair, secondary "
                     "damage and lost production, from the cost model).")
if event:
    n_test = METRICS.get("data", {}).get("n_test_assets", 16)
    missed = int(event.get("failures", 0) - event.get("caught", 0))
    missed_sudden = int(event.get("failures", 0) - event.get("predictable_failures", 0))
    k[6].metric(
        "Failures caught (held-out)", f"{event.get('caught', 0)} / {event.get('failures', 0)}",
        delta=f"{event.get('false_alarms', 0)} false alarms", delta_color="off", delta_arrow="off",
        height="stretch",
        help=f"On {n_test} pumps the models never saw, {event.get('caught', 0)} of {event.get('failures', 0)} "
             f"failures got a usable warning. {missed} were missed, {min(missed, missed_sudden)} of them sudden "
             "failures with no precursor.",
    )
    savings_pct = float(econ.get("hybrid", {}).get("savings_pct", np.nan))
    k[7].metric(
        "Median warning (held-out)", C.fmt_hours(event.get("median_lead_h") or np.nan),
        delta=f"{savings_pct:.0f}% lower maintenance cost" if np.isfinite(savings_pct) else None,
        delta_color="off", delta_arrow="off", height="stretch",
        help=f"Median time from the first alert to the breakdown on the {n_test} held-out pumps: the window to "
             "plan the repair. Cost saving is versus run-to-failure (see Maintenance Economics).",
    )

# ---------------- fleet table ----------------
st.subheader("Fleet status")
c_filter, c_hint = st.columns([2, 3], vertical_alignment="center")
view = c_filter.segmented_control("Show", ["All pumps", "Needs attention", "Held-out test pumps"],
                                  default="All pumps", key="home_view", label_visibility="collapsed")
c_hint.caption("Riskiest first. **Click a row** to open that pump's detail page. Held-out test pumps were never "
               "seen by the models; training pumps show out-of-fold scores.")
shown = snap
if view == "Needs attention":
    shown = snap[snap["status"].isin(ELEVATED)]
elif view == "Held-out test pumps":
    shown = snap[snap["split"].eq("test")]
shown = shown.reset_index(drop=True)
table = fleet_table(shown)

if table.empty:
    st.success("No pump needs attention at this hour. Every running pump is Normal.")
else:
    event_sel = st.dataframe(
        table, hide_index=True, key="fleet_table", on_select="rerun", selection_mode="single-row",
        height=min(38 + 35 * len(table), 460),
        column_config={
            "Asset": st.column_config.TextColumn(pinned=True, width=72),
            "Service": st.column_config.TextColumn(width=165),
            "Crit.": st.column_config.TextColumn(width=48, help="Criticality class: A = loss of production "
                                                 "or safety impact, B = spared / partial impact, C = low impact."),
            "Status": st.column_config.TextColumn(width=92),
            "Risk (7 d)": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f%%", width=92,
                help=f"Hybrid probability of failure within 7 days. Alert at {THRESHOLD:.0%}."),
            "RUL P50 (P10 – P90)": st.column_config.TextColumn(
                width=145, help="Remaining useful life: median and 80% interval. Shown once the pump is on Watch or "
                                "worse; a healthy pump has more than 30 days."),
            "Likely failure mode": st.column_config.TextColumn(
                width=140, help="Most likely developing failure mode and its probability (when elevated)."),
            "Top drivers": st.column_config.TextColumn(
                width=290, help="The two signals furthest from the pump's healthy baseline (robust z-score, "
                                "|z| >= 2). The asset page lists them all."),
            "Split": st.column_config.TextColumn(
                width=78, help=f"Held-out: one of the {METRICS.get('data', {}).get('n_test_assets', 16)} pumps never "
                               "used for training. Training: scored out-of-fold by models that never saw this pump."),
        },
    )
    rows = event_sel.selection.rows if event_sel is not None else []
    if rows:
        st.session_state["asset_id"] = str(table.iloc[rows[0]]["Asset"])
        st.switch_page(ASSET_PAGE)

    labels = {r["Asset"]: f"{r['Asset']} · {r['Status']} · {r['Service']}" for _, r in table.iterrows()}
    c_pick, c_go = st.columns([3, 1], vertical_alignment="bottom")
    pick = c_pick.selectbox("Open a pump", list(labels), key="home_pick", format_func=labels.get)
    if c_go.button("Open asset detail →", type="primary", width="stretch"):
        st.session_state["asset_id"] = pick
        st.switch_page(ASSET_PAGE)

# ---------------- heatmap + unit breakdown + recent alerts ----------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"Risk over the last {HEATMAP_DAYS} days")
    st.caption("Daily maximum of each pump's 7-day failure risk, in the same order as the table. Yellow cells "
               "crossed the Watch level, orange and red crossed the Alert threshold. Blank cells: stopped all day.")
    w = window.assign(day=window["timestamp"].dt.floor("D"))
    daily = w.groupby(["asset_id", "day"])["risk"].max().unstack("day")
    order = snap["asset_id"].tolist()
    days = pd.date_range(as_of_ts.floor("D") - pd.Timedelta(days=HEATMAP_DAYS - 1), as_of_ts.floor("D"), freq="D")
    daily = daily.reindex(index=order, columns=days)
    z = daily.to_numpy(dtype=float)
    hover = np.where(np.isfinite(z), np.char.mod("%.0f%%", np.nan_to_num(100 * z)), "stopped")
    fig = go.Figure(go.Heatmap(
        z=z, x=days, y=order, text=hover, zmin=0, zmax=1, xgap=1, ygap=1,
        colorscale=risk_colorscale(THRESHOLD),
        hovertemplate="<b>%{y}</b><br>%{x|%a %d %b %Y}<br>Max 7-day risk: %{text}<extra></extra>",
        colorbar=dict(title=dict(text="Risk", side="top"), thickness=12, len=0.6,
                      tickvals=[0, THRESHOLD / 2, THRESHOLD, 0.5, 1],
                      ticktext=["0%", f"Watch {THRESHOLD / 2:.0%}", f"Alert {THRESHOLD:.0%}", "50%", "100%"]),
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10), dtick=1)
    fig.update_xaxes(tickformat="%d %b", showgrid=False)
    C.style_fig(fig, height=140 + 15 * len(order), legend_top=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with right:
    st.subheader("Status by process unit")
    if "unit" in snap and snap["unit"].notna().any():
        units = sorted(snap["unit"].dropna().unique())
        fig = go.Figure()
        for status in C.STATUS_ORDER:
            sub = snap[snap["status"] == status]
            pumps = sub.groupby("unit")["asset_id"].apply(lambda s: ", ".join(s)).reindex(units).fillna("")
            n = sub.groupby("unit").size().reindex(units).fillna(0)
            if n.sum() == 0:
                continue
            fig.add_bar(y=units, x=n, name=status, orientation="h", marker_color=C.STATUS_COLORS[status],
                        customdata=pumps, hovertemplate=f"<b>%{{y}}</b><br>{status}: %{{x}}<br>%{{customdata}}"
                                                        "<extra></extra>")
        # Plotly lists stacked-bar legends in reverse; keep them in status order (worst first).
        fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Pumps", dtick=1)
        C.style_fig(fig, height=330)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Recent alerts")
    alerts = C.load_alerts()
    if not alerts.empty:
        alerts = alerts[(alerts["method"] == "hybrid") & (alerts["start_hour"] <= as_of)
                        & (alerts["start_hour"] > as_of - ALERT_LOOKBACK_H)].copy()
    if alerts.empty:
        st.success(f"No Sentinel alerts were raised in the {ALERT_LOOKBACK_H // 24} days before this hour.")
    else:
        all_alerts = C.load_alerts()
        hybrid = all_alerts[all_alerts["method"] == "hybrid"]
        renotify = set(zip(hybrid["asset_id"], hybrid["end_hour"] + 1))
        status_now = snap.set_index("asset_id")["status"]
        alerts = alerts.sort_values("start_hour", ascending=False)
        recent = pd.DataFrame({
            "Raised": [C.hour_to_timestamp(h) for h in alerts["start_hour"]],
            "Pump": alerts["asset_id"].to_numpy(),
            "Type": ["Re-notification" if (a, h) in renotify else "New alert"
                     for a, h in zip(alerts["asset_id"], alerts["start_hour"])],
            "State": np.where(alerts["end_hour"].to_numpy() >= as_of, "Active", "Cleared"),
            "Status now": [f"{STATUS_ICON.get(s, '')} {s}" for s in status_now.reindex(alerts["asset_id"]).fillna("-")],
        })
        st.caption("Sentinel alert raises in the last 30 days, newest first. A still-active alert is re-notified "
                   "every 7 days as a new work request.")
        st.dataframe(recent, hide_index=True, height=min(38 + 35 * len(recent), 320),
                     column_config={"Raised": st.column_config.DatetimeColumn(format="ddd DD MMM, HH:mm")})

# ---------------- explanation ----------------
with st.expander("How to read this"):
    st.markdown(
        f"""
**Status** (re-evaluated every hour for every running pump)

| Status | Rule | What to do |
|---|---|---|
| {STATUS_ICON['Critical']} **Critical** | Alert **and** median remaining life ≤ 72 h | Intervene within 24-48 h; prepare the standby pump |
| {STATUS_ICON['Alert']} **Alert** | Hybrid 7-day failure risk ≥ {THRESHOLD:.0%}, the cost-optimal threshold | Schedule the repair in the next planning window; order parts |
| {STATUS_ICON['Watch']} **Watch** | Risk ≥ {THRESHOLD / 2:.0%} (half the threshold) **or** MSPC ≥ 1.5x its 99% control limit | Take a route vibration reading; review the trend daily |
| {STATUS_ICON['Normal']} **Normal** | None of the above | Normal monitoring |
| {STATUS_ICON['Offline']} **Offline** | Pump stopped (repair, PM or process outage) | Not scored |

**Where the alert threshold comes from.** It was chosen on the training fleet to minimise total maintenance
cost (missed failures vs planned repairs vs wasted inspections), within a budget of at most one nuisance alert
per pump every two years.

**Training vs held-out pumps.** {METRICS.get('data', {}).get('n_train_assets', 32)} pumps were used for
training, and their scores here are *out-of-fold*: each was scored by a model that never saw it.
{METRICS.get('data', {}).get('n_test_assets', 16)} pumps were held out completely. Their numbers are the honest
test of how Sentinel would do on a new unit.

**Remaining life and failure mode** are conditional on a failure developing. They are shown only once a pump is
on Watch or worse; a healthy pump simply has more than 30 days.

**Top drivers** are the signals furthest from that pump's own healthy baseline after its last maintenance
(robust z-scores; ±2σ or more is listed).

**Replay.** Pick any past date and hour. Every panel is recomputed from data up to that hour, as the control
room would have seen it.

**Data.** The plant data is *simulated*: a physics-informed model of a 48-pump fleet over one year, with four
failure modes, sudden failures, process outages, sensor faults and a maintenance log. Real historian data
cannot be published. The pipeline is the same one you would run on historian exports.
"""
    )

st.caption(f"{C.APP_NAME} · simulated plant data · models trained {RUN_ID[:16].replace('T', ' ')} UTC")
