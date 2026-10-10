"""Model Performance: how good are the models, measured the way a maintenance team experiences them.

Nothing is fitted on this page; everything comes from the artifacts the training pipeline wrote.
Two evaluation sets are offered:

* the **held-out test fleet** - 16 pumps the models never saw. This is the honest number.
* **cross-validation** - out-of-fold scores on the 32 training pumps (every pump scored by models
  that never saw it). A second, larger sample, and the one the alert thresholds were tuned on.

Row-level metrics (ROC / PR curves, calibration) grade how well pump-hours are *ranked*.
Event-level metrics (failures caught, lead time, false alarms, cost) grade the *alerts* people
act on. The threshold explorer re-runs the production alert logic live, so the trade-off between
catching failures early and sending mechanics out for nothing is something you can feel.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # app/ folder
import common as C

C.page_setup("Model Performance", "📈")
C.require_artifacts()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

from sentinel_pdm.config import Config  # noqa: E402
from sentinel_pdm.evaluation import alert_episodes, event_metrics, match_events, policy_cost  # noqa: E402
from sentinel_pdm.features import SUFFIXES, signal_of  # noqa: E402

# --------------------------------------------------------------------------------------------
# settings derived from the trained run
# --------------------------------------------------------------------------------------------
M = C.load_metrics()
CFG = Config.from_dict(M.get("config", {}))
METHODS: dict[str, str] = M.get("methods", {})
KEYS: list[str] = list(METHODS)                     # hybrid, ml, mspc, iforest, rule
SCORE_COLS: dict[str, str] = M.get("score_columns", {})
THRESHOLDS: dict[str, float] = M.get("thresholds", {})
VERSION = str(M.get("generated_at", ""))            # cache key: cached results refresh after retraining
SHORT = {"hybrid": "Sentinel hybrid", "ml": "Gradient boosting", "mspc": "MSPC",
         "iforest": "Isolation Forest", "rule": "Alarm limits"}
SPLIT_LABELS = {"test": "Held-out test fleet", "cv": "Cross-validation (training fleet)"}
PRED_SPLIT = {"test": "test", "cv": "train"}        # metrics section -> `split` value in the artifacts
PRED_COLS = ["asset_id", "hour", "split", "running", "label", "ttf", "next_predictable",
             "rul_p10", "rul_p50", "rul_p90", *dict.fromkeys(SCORE_COLS.values())]
CALIBRATED = ("hybrid", "ml")                       # the two scores that are probabilities
# Metric second lines on this page are context, not a change over time: no arrow, no colour.
INFO_DELTA = {"delta_color": "off", "delta_arrow": "off"}
DAY = 24.0


def _reset_threshold(widget_key: str, value: float) -> None:
    """Button callback: runs before the script, so the slider can still be changed."""
    st.session_state[widget_key] = value


def short(key: str) -> str:
    return SHORT.get(key, METHODS.get(key, key))


def asset_years(split_key: str) -> float:
    """Pump-years of operation in an evaluation set (normalises false alarms and cost)."""
    data = M.get("data", {})
    hours = float(data.get("hours", CFG.sim.hours))
    if split_key == "test":
        return float(M.get("test", {}).get("asset_years") or data.get("n_test_assets", 0) * hours / 8760)
    return float(data.get("n_train_assets", 0) * hours / 8760)


def nan_float(x) -> float:
    """metrics.json stores NaN as null; turn it back into a float NaN for formatting."""
    return float("nan") if x is None else float(x)


# --------------------------------------------------------------------------------------------
# cached computations
# --------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing evaluation data...")
def split_frame(pred_split: str, version: str) -> pd.DataFrame:
    """Every pump-hour of one evaluation set, sorted by pump and hour (the order alert logic needs).

    Cached as a resource (shared, not copied) because the threshold explorer reads it on every
    slider move; callers must treat it as read-only.
    """
    p = C.load_predictions(PRED_COLS)
    f = p[p["split"] == pred_split]
    return f.sort_values(["asset_id", "hour"], kind="stable").reset_index(drop=True)


def labelled_rows(pred_split: str, exclude_sudden: bool = False) -> pd.DataFrame:
    """Rows usable for row-level metrics: pump running and the 7-day outcome is known."""
    f = split_frame(pred_split, VERSION)
    keep = f["running"].eq(1) & f["label"].ge(0)
    if exclude_sudden:
        # the 7 days before a failure without precursor: no method can rank these hours correctly
        keep &= ~(f["label"].eq(1) & ~f["next_predictable"].astype(bool))
    return f.loc[keep]


def split_events(pred_split: str) -> pd.DataFrame:
    assets = C.load_assets()
    ids = set(assets.loc[assets["split"] == pred_split, "asset_id"])
    ev = C.load_events()
    return ev[ev["asset_id"].isin(ids)].reset_index(drop=True)


def _sample_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sample a monotone step curve (x ascending) at grid points: y of the last point with x <= g."""
    i = np.searchsorted(x, grid, side="right") - 1
    return y[np.clip(i, 0, len(y) - 1)]


@st.cache_data(show_spinner="Computing ROC and precision-recall curves...")
def roc_pr_curves(split_key: str, exclude_sudden: bool, version: str) -> dict:
    """ROC and PR curves per method, resampled on fixed grids (a few hundred points instead of
    up to 130k) so the charts stay light without smoothing the curves."""
    rows = labelled_rows(PRED_SPLIT[split_key], exclude_sudden)
    y_all = rows["label"].to_numpy(dtype=np.int8)
    fpr_grid = np.unique(np.concatenate([np.logspace(-4, 0, 160), np.linspace(0, 1, 241)]))
    rec_grid = np.linspace(0, 1, 301)
    out = {"roc": [], "pr": [], "summary": [], "base_rate": float(y_all.mean()) if len(y_all) else np.nan}
    for key in KEYS:
        s = rows[SCORE_COLS[key]].to_numpy(dtype=np.float64)
        ok = ~np.isnan(s)
        y, s = y_all[ok], s[ok]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        # precision_recall_curve returns recall in decreasing order; flip to ascending, and take
        # the operating point with the highest threshold that still reaches each recall level.
        rec_a, prec_a = rec[::-1], prec[::-1]
        idx = np.clip(np.searchsorted(rec_a, rec_grid, side="left"), 0, len(rec_a) - 1)
        out["roc"].append(pd.DataFrame({"method": key, "fpr": fpr_grid,
                                        "tpr": _sample_curve(fpr, tpr, fpr_grid)}))
        out["pr"].append(pd.DataFrame({"method": key, "recall": rec_grid, "precision": prec_a[idx]}))
        out["summary"].append({"method": key, "roc_auc": float(roc_auc_score(y, s)),
                               "pr_auc": float(average_precision_score(y, s)), "n_rows": int(len(y))})
    return out


@st.cache_data(show_spinner=False)
def threshold_candidates(split_key: str, method: str, version: str) -> list[float]:
    """Slider stops: the same score quantiles the pipeline searched (80th-99.95th percentile),
    rounded to 4 significant digits, plus the exact production threshold."""
    s = split_frame(PRED_SPLIT[split_key], version)[SCORE_COLS[method]].to_numpy(dtype=np.float64)
    s = s[~np.isnan(s)]
    prod = float(THRESHOLDS.get(method, np.nanmedian(s) if len(s) else 0.5))
    if len(s) == 0:
        return [prod]
    qs = np.concatenate([np.linspace(0.80, 0.99, 39), np.linspace(0.991, 0.9995, 18)])
    cand = {float(f"{v:.4g}") for v in np.quantile(s, qs)}
    cand = {v for v in cand if abs(v - prod) > 1e-3 * max(abs(prod), 1e-9)}
    return sorted(cand | {prod})


@st.cache_data(show_spinner=False, max_entries=512)
def threshold_outcome(split_key: str, method: str, threshold: float, version: str) -> dict:
    """Re-run the production alert logic at one threshold: raises -> P-F matching -> metrics -> cost."""
    pred_split = PRED_SPLIT[split_key]
    frame = split_frame(pred_split, version)
    ep = alert_episodes(frame, SCORE_COLS[method], float(threshold), CFG.alerts)
    failures, ep = match_events(ep, split_events(pred_split), CFG.alerts, CFG.sim.hours)
    em = event_metrics(failures, ep, asset_years(split_key))
    cost = policy_cost(em["failures"], em["caught"], em["false_alarms"], CFG.costs)
    return {**em, "total_cost": cost["total_cost"], "downtime_h": cost["downtime_h"]}


def _risk_mass_bins(p: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Reliability bins that each hold the same *expected number of failure-hours*.

    Plain row quantiles are useless for a rare event: with a 4 % base rate, 11 of 12 bins sit
    below 3 % risk and the alerting range (10-100 %) collapses into one bin. Weighting the
    quantiles by the predicted risk resolves the range that drives decisions. Bins are built on
    unique score values, so a plateau of the isotonic calibrator is never split across bins.
    """
    order = np.argsort(p, kind="stable")
    p, y = p[order], y[order].astype(np.float64)
    uniq, start = np.unique(p, return_index=True)
    count = np.diff(np.append(start, len(p)))
    pos = np.add.reduceat(y, start)
    mass = uniq * count
    total = mass.sum()
    if total <= 0:
        b = np.zeros(len(uniq), dtype=int)
    else:
        mid = (np.cumsum(mass) - mass / 2) / total
        b = np.minimum((mid * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": b, "n": count, "sum_p": mass, "pos": pos, "lo": uniq, "hi": uniq})
    g = df.groupby("bin").agg(n=("n", "sum"), sum_p=("sum_p", "sum"), pos=("pos", "sum"),
                              lo=("lo", "min"), hi=("hi", "max"))
    g["predicted"] = g["sum_p"] / g["n"]
    g["observed"] = g["pos"] / g["n"]
    return g.reset_index(drop=True)


@st.cache_data(show_spinner="Building reliability diagram...")
def calibration_bins(version: str, n_bins: int = 12) -> pd.DataFrame:
    rows = labelled_rows("test")
    y = rows["label"].to_numpy(dtype=np.int8)
    parts = []
    for key in CALIBRATED:
        p = rows[SCORE_COLS[key]].to_numpy(dtype=np.float64)
        ok = ~np.isnan(p)
        if ok.sum():
            parts.append(_risk_mass_bins(np.clip(p[ok], 0, 1), y[ok], n_bins).assign(method=key))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@st.cache_data(show_spinner=False)
def rul_rows(version: str) -> pd.DataFrame:
    """Test hours inside the last 30 days before a failure that had a precursor."""
    f = split_frame("test", version)
    keep = (f["running"].eq(1) & f["next_predictable"].astype(bool) & f["ttf"].le(CFG.labels.rul_cap_h)
            & f["rul_p50"].notna())
    return f.loc[keep, ["asset_id", "hour", "ttf", "rul_p10", "rul_p50", "rul_p90"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# small presentation helpers
# --------------------------------------------------------------------------------------------
def fmt_days(h: float) -> str:
    return "-" if h is None or not np.isfinite(h) else f"{h / DAY:.1f} d"


def fmt_num(x: float, digits: int = 3) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def line_width(key: str) -> float:
    return 3.2 if key == "hybrid" else 1.8


def legend_inside(fig: go.Figure, corner: str) -> go.Figure:
    """Move the legend into an empty corner of the plot ("bottom right" / "top right").

    Six methods make a three-row legend in a half-width chart, too tall for the space between
    title and plot; the corners of ROC and PR plots stay empty, so it goes there instead.
    """
    v, h = corner.split()
    fig.update_layout(legend=dict(orientation="v", x=0.98 if h == "right" else 0.02, xanchor=h,
                                  y=0.03 if v == "bottom" else 0.97, yanchor=v,
                                  bgcolor="rgba(15, 17, 23, 0.8)", bordercolor="rgba(255,255,255,0.12)",
                                  borderwidth=1))
    return fig.update_layout(margin=dict(t=40))


SUFFIX_LABELS = {"_m24": "24 h mean", "_sd24": "24 h variability", "_m168": "7-day mean",
                 "_sl24": "24 h trend", "_sl168": "7-day trend", "_dev": "shift from baseline"}
CONTEXT_LABELS = {"load_pct": "Process load", "ambient_temp": "Ambient temperature", "speed_rpm": "Pump speed",
                  "hours_since_maint": "Hours since last maintenance",
                  "run_hours_since_maint": "Running hours since last maintenance"}


def signal_label(signal: str) -> str:
    if signal in C.SIGNAL_INFO:
        return C.SIGNAL_INFO[signal][0]
    return CONTEXT_LABELS.get(signal, signal.replace("_", " "))


def feature_label(feature: str) -> str:
    sig = signal_of(feature)
    suffix = feature[len(sig):]
    return f"{signal_label(sig)} · {SUFFIX_LABELS.get(suffix, suffix.strip('_'))}" if suffix in SUFFIXES \
        else signal_label(sig)


# --------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------
st.title("📈 Model Performance")
st.caption(
    "How good are the models, measured honestly: on pumps they never saw, counting alerts the way a "
    "maintenance planner does. Every number below is reproducible from `python -m sentinel_pdm train`."
)

split_key = st.segmented_control(
    "Evaluate on", options=list(SPLIT_LABELS), format_func=SPLIT_LABELS.get, default="test",
    required=True, key="mp_split",
    help="Held-out test fleet: 16 pumps never used for training or tuning. Cross-validation: "
         "out-of-fold scores on the 32 training pumps (each pump scored by models trained without it).",
) or "test"
pred_split = PRED_SPLIT[split_key]
section = M.get(split_key, {})
events_metrics: dict = section.get("event", {})
row_metrics: dict = section.get("row", {})
years = asset_years(split_key)
n_pumps = M.get("data", {}).get("n_test_assets" if split_key == "test" else "n_train_assets", 0)
st.caption(f"{SPLIT_LABELS[split_key]}: {n_pumps} pumps · {years:.0f} pump-years of hourly data.")

tab_score, tab_events, tab_thr, tab_prob, tab_dx = st.tabs(
    ["Scorecard", "Failures & lead time", "Threshold explorer", "Probability & RUL", "Diagnosis & drivers"]
)

# ---- Scorecard ---------------------------------------------------------------------------------
with tab_score:
    table_rows = []
    for key in KEYS:
        ev = events_metrics.get(key, {})
        rm = row_metrics.get(key, {})
        n_fail = int(ev.get("failures", 0))
        cost = policy_cost(n_fail, int(ev.get("caught", 0)), int(ev.get("false_alarms", 0)), CFG.costs)
        reactive = policy_cost(n_fail, 0, 0, CFG.costs)["total_cost"]   # run-to-failure baseline
        saving = (reactive - cost["total_cost"]) / reactive if reactive else np.nan
        table_rows.append({
            "key": key,
            "Method": METHODS.get(key, key),
            "ROC-AUC": nan_float(rm.get("roc_auc")),
            "PR-AUC": nan_float(rm.get("pr_auc")),
            "Failures caught": f"{ev.get('caught', 0)} / {n_fail}",
            "Predictable caught": f"{ev.get('caught_predictable', 0)} / {ev.get('predictable_failures', 0)}",
            "False alarms": int(ev.get("false_alarms", 0)),
            "Alert precision": nan_float(ev.get("alert_precision")),
            "Median lead (days)": nan_float(ev.get("median_lead_h")) / DAY,
            "Cost saving": saving,
        })
    table = pd.DataFrame(table_rows)

    hy = events_metrics.get("hybrid", {})
    hy_row = table.loc[table["key"] == "hybrid"].iloc[0] if (table["key"] == "hybrid").any() else None
    k1, k2, k3, k4 = st.columns(4)   # every card has a second line, so the row stays even
    n_hy, caught_hy = int(hy.get("failures", 0)), int(hy.get("caught", 0))
    k1.metric("Failures caught (Sentinel)", f"{caught_hy} / {n_hy}",
              delta=f"{C.fmt_pct(caught_hy / n_hy)} detection rate" if n_hy else None, **INFO_DELTA,
              help="Caught = at least one alert inside the P-F interval, 12 h or more before the trip.")
    n_sudden = n_hy - int(hy.get("predictable_failures", 0))
    k2.metric("Predictable failures caught", f"{hy.get('caught_predictable', 0)} / {hy.get('predictable_failures', 0)}",
              delta=f"{n_sudden} sudden failure{'s' if n_sudden != 1 else ''} excluded", **INFO_DELTA,
              help="Excludes 'sudden' failures (process upsets, foreign objects) that have no measurable precursor.")
    fa = int(hy.get("false_alarms", 0))
    k3.metric("False alarms", f"{fa}", help="Inspections that found nothing.",
              delta=f"{fa / years:.2f} per pump-year" if years else None, **INFO_DELTA)
    k4.metric("Median warning before trip", fmt_days(nan_float(hy.get("median_lead_h"))),
              delta=f"{C.fmt_pct(hy_row['Cost saving'], 1)} cost saving" if hy_row is not None else None,
              **INFO_DELTA)

    st.subheader("Head-to-head")

    def _highlight(row: pd.Series) -> list[str]:
        style = "background-color: rgba(245,158,11,0.16); font-weight: 600;" if row["key"] == "hybrid" else ""
        return [style] * len(row)

    styler = (
        table.style.apply(_highlight, axis=1)
        .format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Alert precision": "{:.0%}",
                 "Median lead (days)": "{:.1f}", "Cost saving": "{:.1%}"}, na_rep="-")
    )
    st.dataframe(styler, hide_index=True, width="stretch", column_order=[c for c in table.columns if c != "key"])

    ml_rm, hy_rm = row_metrics.get("ml", {}), row_metrics.get("hybrid", {})
    extra = ""
    if (nan_float(ml_rm.get("roc_auc")) > nan_float(hy_rm.get("roc_auc"))
            and events_metrics.get("ml", {}).get("false_alarms", 0) > hy.get("false_alarms", 0)):
        extra = (f" Here gradient boosting has the higher ROC-AUC, yet it sent people out "
                 f"{events_metrics['ml']['false_alarms']} times for nothing; the hybrid sent them "
                 f"{hy.get('false_alarms', 0)} times.")
    st.caption(
        "**Why event-level metrics matter more.** ROC-AUC and PR-AUC grade how well every pump-hour is "
        "ranked, but nobody acts on an hour: planners act on alerts. The event columns count whether each "
        "failure got an actionable warning (inside the P-F interval, at least 12 h before the trip), how "
        "early, and how many inspections found nothing. Cost saving is versus run-to-failure, using the "
        "cost assumptions on the Maintenance Economics page." + extra
    )

    st.subheader("ROC and precision-recall curves")
    exclude = st.checkbox(
        "Exclude the 7 days before sudden failures (unpredictable)", value=False, key="mp_exclude_sudden",
        help="Sudden failures have no precursor, so those hours are labelled 'failure within 7 days' "
             "but no sensor can tell them apart from healthy running. Excluding them shows how well "
             "the methods rank the failures that are physically predictable.",
    )
    curves = roc_pr_curves(split_key, exclude, VERSION)
    summ = {r["method"]: r for r in curves["summary"]}
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#6b7280", dash="dot", width=1),
                    hoverinfo="skip")
    for df in curves["roc"]:
        key = df["method"].iat[0]
        fig.add_scatter(x=df["fpr"], y=df["tpr"], mode="lines", line=dict(color=C.METHOD_COLORS.get(key), width=line_width(key)),
                        name=f"{short(key)} · {summ[key]['roc_auc']:.3f}",
                        hovertemplate=f"{short(key)}<br>false-positive rate %{{x:.3f}}<br>true-positive rate %{{y:.3f}}<extra></extra>")
    fig.update_layout(title="ROC (legend: AUC)", xaxis_title="False-positive rate (healthy hours flagged)",
                      yaxis_title="True-positive rate (pre-failure hours flagged)")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    c1.plotly_chart(legend_inside(C.style_fig(fig, height=430), "bottom right"), width="stretch")

    fig = go.Figure()
    base = curves["base_rate"]
    fig.add_scatter(x=[0, 1], y=[base, base], mode="lines", name=f"No skill · {base:.3f}",
                    line=dict(color="#6b7280", dash="dot", width=1), hoverinfo="skip")
    for df in curves["pr"]:
        key = df["method"].iat[0]
        fig.add_scatter(x=df["recall"], y=df["precision"], mode="lines",
                        line=dict(color=C.METHOD_COLORS.get(key), width=line_width(key), shape="hv"),
                        name=f"{short(key)} · {summ[key]['pr_auc']:.3f}",
                        hovertemplate=f"{short(key)}<br>recall %{{x:.3f}}<br>precision %{{y:.3f}}<extra></extra>")
    fig.update_layout(title="Precision-recall (legend: average precision)", xaxis_title="Recall (pre-failure hours flagged)",
                      yaxis_title="Precision (flagged hours that were pre-failure)")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    c2.plotly_chart(legend_inside(C.style_fig(fig, height=430), "top right"), width="stretch")
    n_rows = max((r["n_rows"] for r in curves["summary"]), default=0)
    st.caption(
        f"{n_rows:,} running pump-hours with a known 7-day outcome; failure-within-7-days base rate "
        f"{C.fmt_pct(base, 1)}. PR curves are the more telling view for a rare event: a model can look "
        "excellent on ROC while most of its flags are healthy hours."
    )

# ---- Failures & lead time ----------------------------------------------------------------------
with tab_events:
    fo = C.load_failure_outcomes()
    fo = fo[fo["split"] == pred_split].copy() if not fo.empty else fo
    if fo.empty:
        st.info("No failures in this evaluation set.")
    else:
        fo["timestamp"] = pd.to_datetime(fo["timestamp"])
        fo["predictable"] = fo["predictable"].astype(bool)
        fo["caught"] = fo["caught"].astype(bool)
        fo["late_only"] = fo["late_only"].astype(bool)

        st.subheader("Every failure, one row")
        ev_method = st.selectbox("Method", KEYS, format_func=lambda k: METHODS.get(k, k), key="mp_ev_method")
        sel = fo[fo["method"] == ev_method].sort_values("timestamp")
        outcome = np.where(sel["caught"], "Caught",
                           np.where(sel["late_only"], "Late (< 12 h)",
                                    np.where(sel["predictable"], "Missed", "Missed (no precursor)")))
        per_failure = pd.DataFrame({
            "Pump": sel["asset_id"].to_numpy(),
            "Failure time": sel["timestamp"].to_numpy(),
            "Mode": sel["mode"].map(lambda m: C.MODE_LABELS.get(m, m)).to_numpy(),
            "Outcome": outcome,
            "Lead time (days)": (sel["lead_h"] / DAY).to_numpy(),
            "P-F interval (days)": ((sel["hour"] - pd.to_numeric(sel["p_hour"], errors="coerce")) / DAY).to_numpy(),
            "Work order": sel["event_id"].to_numpy(),
        })
        n_caught = int(sel["caught"].sum())
        st.caption(
            f"{METHODS.get(ev_method, ev_method)} caught {n_caught} of {len(sel)} failures "
            f"({int(sel.loc[sel['predictable'], 'caught'].sum())} of {int(sel['predictable'].sum())} with a precursor). "
            "P-F interval = time from the defect becoming detectable (P) to the functional failure (F); "
            "lead time = F minus the first qualifying alert."
        )
        st.dataframe(
            per_failure, hide_index=True, width="stretch", height=min(38 + 35 * len(per_failure), 420),
            column_config={
                "Pump": st.column_config.TextColumn(width=80),
                "Failure time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm", width=150),
                "Mode": st.column_config.TextColumn(width=280),
                "Outcome": st.column_config.TextColumn(width=190),
                "Lead time (days)": st.column_config.NumberColumn(
                    format="%.1f", help="Empty when no qualifying alert came before the failure."),
                "P-F interval (days)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        c1, c2 = st.columns(2)
        fig = go.Figure()
        for key in KEYS:
            lead = fo.loc[(fo["method"] == key) & fo["caught"], "lead_h"] / DAY
            if lead.empty:
                continue
            fig.add_box(y=lead, name=short(key), marker_color=C.METHOD_COLORS.get(key), boxpoints="all",
                        jitter=0.45, pointpos=0, marker=dict(size=6, opacity=0.75), line=dict(width=1.5),
                        hovertemplate=f"{short(key)}<br>%{{y:.1f}} days<extra></extra>")
        fig.add_hline(y=CFG.alerts.min_lead_h / DAY, line=dict(color="#6b7280", dash="dot", width=1),
                      annotation_text=f"{CFG.alerts.min_lead_h} h minimum to plan", annotation_position="bottom right")
        fig.update_layout(title="Warning lead time of caught failures", yaxis_title="Days before failure",
                          showlegend=False)
        c1.plotly_chart(C.style_fig(fig, height=420), width="stretch")

        modes = [m for m in (*C.FAILURE_MODES, "sudden") if (fo["mode"] == m).any()]
        n_by_mode = fo[fo["method"] == KEYS[0]].groupby("mode").size()
        fig = go.Figure()
        for key in KEYS:
            g = fo[fo["method"] == key].groupby("mode")["caught"].agg(["sum", "size"]).reindex(modes)
            fig.add_bar(
                x=[f"{m.capitalize()} (n={int(n_by_mode.get(m, 0))})" for m in modes], y=g["sum"] / g["size"],
                name=short(key), marker_color=C.METHOD_COLORS.get(key),
                customdata=np.column_stack([g["sum"].fillna(0), g["size"].fillna(0)]),
                hovertemplate=f"{short(key)}<br>%{{x}}<br>caught %{{customdata[0]:.0f}} of %{{customdata[1]:.0f}}<extra></extra>",
            )
        fig.update_layout(title="Detection rate by failure mode", barmode="group", bargap=0.25, bargroupgap=0.08,
                          yaxis=dict(tickformat=".0%", range=[0, 1.05], title="Failures caught"))
        c2.plotly_chart(C.style_fig(fig, height=420), width="stretch")
        st.caption(
            "Sudden failures (process upsets, foreign objects) have no measurable precursor; every method "
            "misses them, which is why 'predictable caught' is reported separately. Conventional alarm "
            "limits only fire once the defect is already severe, so they catch few failures and late."
        )

# ---- Threshold explorer ------------------------------------------------------------------------
with tab_thr:
    st.subheader("What if the alert threshold were different?")
    st.caption(
        "Move the threshold and the production alert logic re-runs on the selected evaluation set: "
        f"{CFG.alerts.debounce_h} h debounce, {CFG.alerts.clear_h} h to clear, re-alert every "
        f"{CFG.alerts.realert_h / DAY:.0f} days while still active, P-F interval matching. Lower thresholds "
        "warn earlier and catch more, at the price of inspections that find nothing."
    )
    c_ctrl, c_res = st.columns([1, 2.2])
    with c_ctrl:
        thr_method = st.selectbox("Method", KEYS, format_func=lambda k: METHODS.get(k, k), key="mp_thr_method")
        prod_thr = float(THRESHOLDS.get(thr_method, np.nan))
        options = threshold_candidates(split_key, thr_method, VERSION)
        default = prod_thr if prod_thr in options else options[len(options) // 2]
        thr_key = f"mp_thr_{split_key}_{thr_method}"
        if st.session_state.get(thr_key) not in options:
            st.session_state[thr_key] = default      # first visit (or stale value): start at production
        thr = st.select_slider(
            "Alert threshold", options=options, format_func=lambda v: f"{v:.4g}", key=thr_key,
            help="Stops are the 80th to 99.95th percentiles of this method's score, plus the production "
                 "threshold (chosen on the training fleet: lowest cost within the alarm budget).",
        )
        st.caption(f"Production threshold: **{prod_thr:.4g}** · score column `{SCORE_COLS.get(thr_method, '?')}`")
        st.button("Reset to production threshold", key="mp_thr_reset", disabled=bool(np.isclose(thr, default)),
                  on_click=_reset_threshold, args=(thr_key, default))

    res = threshold_outcome(split_key, thr_method, float(thr), VERSION)
    ref = threshold_outcome(split_key, thr_method, float(default), VERSION)
    with c_res:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Failures caught", f"{res['caught']} / {res['failures']}",
                  delta=res["caught"] - ref["caught"] or None, help="Versus the production threshold.")
        r2.metric("False alarms", f"{res['false_alarms']}",
                  delta=(res["false_alarms"] - ref["false_alarms"]) or None, delta_color="inverse",
                  help=f"{res['false_alarms_per_asset_year']:.2f} per pump-year; budget "
                       f"{CFG.alerts.max_false_alarms_per_asset_year} per pump-year.")
        lead_delta = res["median_lead_h"] - ref["median_lead_h"]
        r3.metric("Median lead time", fmt_days(res["median_lead_h"]),
                  delta=f"{lead_delta / DAY:+.1f} d" if np.isfinite(lead_delta) and abs(lead_delta) >= 1 else None)
        cost_delta = res["total_cost"] - ref["total_cost"]
        r4.metric("Maintenance cost", C.fmt_money(res["total_cost"]),
                  delta=C.fmt_money(cost_delta) if abs(cost_delta) >= 1 else None, delta_color="inverse",
                  help=f"Over {years:.0f} pump-years: missed failures as breakdowns, caught ones as planned "
                       "repairs, plus an inspection per false alarm.")
        over_budget = res["false_alarms_per_asset_year"] > CFG.alerts.max_false_alarms_per_asset_year
        msg = (f"{res['alerts']} work requests raised · {res['true_alerts']} confirmed a developing failure · "
               f"{res['pm_confirmed_alerts']} matched a defect PM fixed · alert precision "
               f"{C.fmt_pct(res['alert_precision'])}.")
        if over_budget:
            st.warning(f"Over the alarm budget ({res['false_alarms_per_asset_year']:.2f} nuisance alerts per "
                       f"pump-year vs {CFG.alerts.max_false_alarms_per_asset_year}). {msg}")
        else:
            st.success(f"Within the alarm budget. {msg}")

    curve = C.load_threshold_curve()
    cm = curve[curve["method"] == thr_method].sort_values("threshold") if not curve.empty else curve
    if cm.empty:
        st.info("Alarm limits are fixed engineering limits (score 1.0 = a tag beyond its ISO 10816 / RTD alarm "
                "limit), so no threshold was tuned for them and there is no cost curve. The slider still shows "
                "what moving every limit up or down would do.")
    else:
        budget = CFG.alerts.max_false_alarms_per_asset_year
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.55, 0.45],
                            subplot_titles=("Total maintenance cost, training fleet ($M)",
                                            "False alarms per pump-year"))
        within = cm["within_budget"].astype(bool)
        color = C.METHOD_COLORS.get(thr_method, C.ACCENT)
        fig.add_scatter(x=cm["threshold"], y=cm["total_cost"] / 1e6, mode="lines", line=dict(color=color, width=2),
                        showlegend=False, hoverinfo="skip", row=1, col=1)
        for flag, name, symbol in ((True, "Within alarm budget", "circle"), (False, "Over alarm budget", "circle-open")):
            sub = cm[within == flag]
            fig.add_scatter(x=sub["threshold"], y=sub["total_cost"] / 1e6, mode="markers", name=name,
                            marker=dict(color=color, size=8, symbol=symbol, line=dict(width=1.5, color=color)),
                            customdata=np.column_stack([sub["caught"], sub["failures"], sub["false_alarms"]]),
                            hovertemplate="threshold %{x:.4g}<br>cost $%{y:.2f}M<br>caught %{customdata[0]} of "
                                          "%{customdata[1]}<br>false alarms %{customdata[2]}<extra></extra>",
                            row=1, col=1)
        fa_cap = max(3.0, budget * 6)
        fig.add_scatter(x=cm["threshold"], y=cm["false_alarms_per_asset_year"], mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=6), showlegend=False,
                        hovertemplate="threshold %{x:.4g}<br>%{y:.2f} false alarms per pump-year<extra></extra>",
                        row=2, col=1)
        fig.add_hline(y=budget, line=dict(color="#ef4444", dash="dot", width=1.5), row=2, col=1,
                      annotation_text=f"Alarm budget {budget} / pump-year", annotation_position="top right")
        for x, color, dash in ((prod_thr, C.ACCENT, "solid"), (float(thr), "#e5e7eb", "dash")):
            if np.isfinite(x):
                fig.add_vline(x=x, line=dict(color=color, dash=dash, width=1.5))
        if np.isfinite(prod_thr):
            fig.add_annotation(x=prod_thr, y=1, yref="paper", text="production", showarrow=False,
                               yanchor="bottom", font=dict(color=C.ACCENT, size=11))
        lo_t, hi_t = float(cm["threshold"].min()), float(cm["threshold"].max())
        if lo_t > 0 and hi_t / lo_t > 30:
            # Plotly's default log ticks label every minor step with a bare digit ("8 9 0.01 2 3 ...");
            # 1-2-5 steps with full numbers read at a glance.
            decades = range(int(np.floor(np.log10(lo_t))), int(np.ceil(np.log10(hi_t))) + 1)
            ticks = [m * 10.0**d for d in decades for m in (1, 2, 5) if lo_t / 1.5 <= m * 10.0**d <= hi_t * 1.5]
            fig.update_xaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
        fig.update_xaxes(title_text="Alert threshold", row=2, col=1)
        fig.update_yaxes(range=[0, fa_cap], row=2, col=1)
        st.plotly_chart(C.style_fig(fig, height=520), width="stretch")
        st.caption(
            "This curve is where the production threshold came from: the training fleet's out-of-fold scores, "
            f"lowest total cost subject to at most {budget} nuisance alerts per pump-year (the ISA-18.2 / "
            "EEMUA 191 lesson: operators ignore a system that cries wolf). Choosing it on the training fleet "
            f"keeps the test fleet untouched. False-alarm axis clipped at {fa_cap:g} per pump-year; hover for values."
        )

# ---- Probability & RUL ------------------------------------------------------------------------
with tab_prob:
    st.info("Calibration, remaining-useful-life and diagnosis results are always on the **held-out test fleet**: "
            "the calibrators and conformal intervals were fitted on the training fleet's out-of-fold scores, "
            "so only the test fleet can check them.", icon="ℹ️")
    st.subheader("Is a 30 % risk really 30 %?")
    cal = calibration_bins(VERSION)
    c1, c2 = st.columns([1.5, 1])
    if cal.empty:
        c1.info("No calibrated scores available.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                        line=dict(color="#6b7280", dash="dot", width=1), hoverinfo="skip")
        for key in CALIBRATED:
            g = cal[cal["method"] == key]
            if g.empty:
                continue
            fig.add_scatter(
                x=g["predicted"], y=g["observed"], mode="lines+markers", name=short(key),
                line=dict(color=C.METHOD_COLORS.get(key), width=line_width(key)),
                marker=dict(size=9, line=dict(width=1.5, color="#0f1117")),
                customdata=np.column_stack([g["n"], g["lo"], g["hi"]]),
                hovertemplate=f"{short(key)}<br>predicted %{{x:.1%}}<br>observed %{{y:.1%}}<br>"
                              "%{customdata[0]:,} hours (risk %{customdata[1]:.3f}-%{customdata[2]:.3f})<extra></extra>",
            )
        fig.update_layout(title="Reliability diagram (held-out test fleet)",
                          xaxis=dict(title="Predicted probability of failure within 7 days", tickformat=".0%", range=[0, 1]),
                          yaxis=dict(title="Observed frequency", tickformat=".0%", range=[0, 1.02]))
        c1.plotly_chart(C.style_fig(fig, height=430), width="stretch")
    with c2:
        test_row = M.get("test", {}).get("row", {})
        base = nan_float(test_row.get("hybrid", {}).get("positive_rate"))
        clim = base * (1 - base) if np.isfinite(base) else np.nan
        for key in CALIBRATED:
            b = nan_float(test_row.get(key, {}).get("brier"))
            skill = 1 - b / clim if np.isfinite(b) and clim else np.nan
            st.metric(f"Brier score · {short(key)}", fmt_num(b, 4),
                      delta=f"skill {C.fmt_pct(skill)} vs. base rate" if np.isfinite(skill) else None,
                      **INFO_DELTA, help="Mean squared error of the probability (lower is better).")
        st.caption(
            f"Always predicting the base rate ({C.fmt_pct(base, 1)}) would score a Brier of {fmt_num(clim, 4)}. "
            "Bins are quantiles of the risk-weighted distribution, so each holds the same expected number of "
            "failure-hours: plain row quantiles would put 11 of 12 bins below 3 % risk, where nothing is decided. "
            "Hourly rows are autocorrelated, so treat a single bin's wobble with caution."
        )

    st.subheader("Remaining useful life")
    rr = rul_rows(VERSION)
    rul_m = M.get("test", {}).get("rul", {})
    last7, last30 = rul_m.get("le_168h", {}), rul_m.get("le_720h", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean absolute error, last 7 days", f"{nan_float(last7.get('mae_h')):.0f} h" if last7 else "-",
              delta=f"P50 estimate, {last7.get('n_rows', 0):,} hours" if last7 else None, **INFO_DELTA)
    m2.metric("P10-P90 interval coverage", C.fmt_pct(nan_float(last7.get("coverage_p10_p90"))),
              delta="target 80 %", **INFO_DELTA,
              help="Share of hours where the true time-to-failure fell inside the predicted P10-P90 band.")
    m3.metric("Median interval width", fmt_days(nan_float(last7.get("median_interval_h"))),
              delta=f"30-day window: MAE {fmt_days(nan_float(last30.get('mae_h')))}, coverage "
                    f"{C.fmt_pct(nan_float(last30.get('coverage_p10_p90')))}" if last30 else None,
              **INFO_DELTA)
    if rr.empty:
        st.info("No pre-failure hours on the test fleet to evaluate.")
    else:
        c1, c2 = st.columns(2)
        cap_d = CFG.labels.rul_cap_h / DAY
        edges = np.arange(0, cap_d + 1, 1.0)
        h2, _, _ = np.histogram2d(rr["ttf"] / DAY, rr["rul_p50"] / DAY, bins=[edges, edges])
        col_sum = h2.sum(axis=1, keepdims=True)
        dens = np.divide(h2, col_sum, out=np.zeros_like(h2), where=col_sum > 0)
        centres = edges[:-1] + 0.5
        fig = go.Figure()
        fig.add_heatmap(x=centres, y=centres, z=dens.T, colorscale=[[0, "rgba(245,158,11,0)"], [0.05, "rgba(245,158,11,0.18)"],
                                                                    [1, C.ACCENT]],
                        colorbar=dict(title="share", tickformat=".0%", thickness=12),
                        hovertemplate="actual %{x:.0f} d<br>predicted %{y:.0f} d<br>%{z:.1%} of that day's hours<extra></extra>")
        med = rr.assign(d=np.floor(rr["ttf"] / DAY)).groupby("d")["rul_p50"].median() / DAY
        fig.add_scatter(x=med.index + 0.5, y=med.to_numpy(), mode="lines", name="Median prediction",
                        line=dict(color="#e5e7eb", width=2))
        fig.add_scatter(x=[0, cap_d], y=[0, cap_d], mode="lines", name="Perfect", line=dict(color="#6b7280", dash="dot", width=1))
        fig.update_layout(title="Predicted P50 vs. actual time to failure",
                          xaxis=dict(title="Actual time to failure (days)", range=[0, cap_d]),
                          yaxis=dict(title="Predicted remaining life, P50 (days)", range=[0, cap_d]))
        c1.plotly_chart(C.style_fig(fig, height=440), width="stretch")
        c1.caption(f"{len(rr):,} running hours in the last {cap_d:.0f} days before a failure with a precursor. "
                   "Each column is normalised: it shows where the predictions for that true horizon fall. "
                   f"Predictions are capped at {cap_d:.0f} days.")

        fo_all = C.load_failure_outcomes()
        ex = fo_all[(fo_all["method"] == "hybrid") & (fo_all["split"] == "test") & fo_all["predictable"].astype(bool)]
        ex = ex.sort_values("hour").reset_index(drop=True)
        with c2:
            if ex.empty:
                st.info("No predictable test failures to show.")
            else:
                med_lead = ex.loc[ex["caught"].astype(bool), "lead_h"].median()
                default_i = int((ex["lead_h"] - med_lead).abs().fillna(np.inf).to_numpy().argmin()) if np.isfinite(med_lead) else 0
                labels = [f"{r.asset_id} · {r.mode} · {str(r.timestamp)[:10]}" for r in ex.itertuples()]
                i = st.selectbox("Example failure", range(len(ex)), index=default_i, format_func=labels.__getitem__,
                                 key="mp_rul_example")
                f = ex.iloc[i]
                tr = split_frame("test", VERSION)
                tr = tr[(tr["asset_id"] == f["asset_id"]) & (tr["hour"] < f["hour"])
                        & (tr["hour"] >= f["hour"] - CFG.labels.rul_cap_h) & tr["running"].eq(1)]
                x = (tr["hour"] - f["hour"]) / DAY
                fig = go.Figure()
                fig.add_scatter(x=x, y=tr["rul_p90"] / DAY, mode="lines", line=dict(width=0), showlegend=False,
                                hoverinfo="skip")
                fig.add_scatter(x=x, y=tr["rul_p10"] / DAY, mode="lines", line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(245,158,11,0.22)", name="P10-P90 interval", hoverinfo="skip")
                fig.add_scatter(x=x, y=tr["rul_p50"] / DAY, mode="lines", name="Predicted P50",
                                line=dict(color=C.ACCENT, width=2.5),
                                hovertemplate="%{x:.1f} d<br>P50 %{y:.1f} d<extra></extra>")
                fig.add_scatter(x=x, y=-x, mode="lines", name="Actual remaining life",
                                line=dict(color="#e5e7eb", dash="dash", width=1.5), hoverinfo="skip")
                if bool(f["caught"]) and np.isfinite(f["lead_h"]):
                    fig.add_vline(x=-f["lead_h"] / DAY, line=dict(color=C.STATUS_COLORS["Alert"], dash="dot", width=1.5),
                                  annotation_text="first alert", annotation_position="top left")
                p_hour = pd.to_numeric(pd.Series([f["p_hour"]]), errors="coerce").iat[0]
                if np.isfinite(p_hour) and p_hour > f["hour"] - CFG.labels.rul_cap_h:
                    fig.add_vline(x=(p_hour - f["hour"]) / DAY, line=dict(color="#9ca3af", dash="dot", width=1),
                                  annotation_text="P (detectable)", annotation_position="bottom right")
                fig.update_layout(title=f"{f['asset_id']}: {C.MODE_LABELS.get(f['mode'], f['mode'])}",
                                  xaxis=dict(title="Days relative to failure", range=[-cap_d, 0]),
                                  yaxis=dict(title="Remaining life (days)", range=[0, cap_d * 1.05]))
                st.plotly_chart(C.style_fig(fig, height=390), width="stretch")
                st.caption("The RUL model answers 'if this pump is failing, how long has it got?'. Read it once an "
                           "alert is active; before that the band is wide by design.")

# ---- Diagnosis & drivers -----------------------------------------------------------------------
with tab_dx:
    st.subheader("Which failure mode is it?")
    dx = M.get("test", {}).get("diagnosis", {})
    conf = dx.get("confusion")
    c1, c2 = st.columns([1.4, 1])
    if not conf:
        c1.info("No diagnosis results in this run.")
    else:
        labels = conf["labels"]
        z = np.asarray(conf["matrix"], dtype=float)
        names = [lab.capitalize() for lab in labels]
        row_tot = z.sum(axis=1, keepdims=True)
        share = np.divide(z, row_tot, out=np.zeros_like(z), where=row_tot > 0)
        fig = go.Figure(go.Heatmap(
            z=share, x=names, y=names, colorscale=[[0, "rgba(245,158,11,0.04)"], [1, C.ACCENT]], zmin=0, zmax=1,
            text=z.astype(int), texttemplate="%{text}", textfont=dict(size=15), showscale=False,
            hovertemplate="actual %{y}<br>diagnosed %{x}<br>%{text} failures (%{z:.0%} of row)<extra></extra>",
        ))
        fig.update_layout(title="Confusion matrix, failures on the test fleet",
                          xaxis=dict(title="Diagnosed mode (week before the failure)", side="bottom"),
                          yaxis=dict(title="Actual mode (from the work order)", autorange="reversed"))
        c1.plotly_chart(C.style_fig(fig, height=400, legend_top=False), width="stretch")
    with c2:
        st.metric("Failures correctly diagnosed", C.fmt_pct(nan_float(dx.get("event_accuracy"))),
                  delta=f"{dx.get('n_events', 0)} failures with a precursor", **INFO_DELTA)
        st.metric("Hour-level accuracy", C.fmt_pct(nan_float(dx.get("row_accuracy"))),
                  delta="every running hour in the final week", **INFO_DELTA)
        st.caption("A failure counts as correctly diagnosed when the mode probabilities averaged over its final "
                   "week point to the mode recorded on the work order. The diagnosis tells the planner which "
                   "craft and parts to send (bearing kit vs. laser alignment vs. strainer cleaning vs. motor shop).")

    st.subheader("What the risk model looks at")
    imp = C.load_feature_importance()
    if imp.empty:
        st.info("Permutation importance was skipped in this run (`--no-importance`).")
    else:
        imp = imp.copy()
        imp["label"] = imp["feature"].map(feature_label)
        top = imp.sort_values("importance", ascending=False).head(20).iloc[::-1]
        c1, c2 = st.columns([1.35, 1])
        fig = go.Figure(go.Bar(
            x=top["importance"], y=top["label"], orientation="h", marker_color=C.METHOD_COLORS["ml"],
            error_x=dict(type="data", array=top["std"].fillna(0), color="#9ca3af", thickness=1, width=3),
            customdata=top["feature"], hovertemplate="%{y}<br><code>%{customdata}</code><br>PR-AUC drop %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(title="Top 20 features", xaxis_title="Drop in PR-AUC when shuffled", yaxis=dict(tickfont=dict(size=11)))
        c1.plotly_chart(C.style_fig(fig, height=620, legend_top=False), width="stretch")
        agg = (imp.assign(pos=imp["importance"].clip(lower=0)).groupby("signal")["pos"].sum()
               .sort_values(ascending=True))
        agg = agg[agg > 0]
        total = agg.sum()
        fig = go.Figure(go.Bar(
            x=agg / total if total else agg, y=[signal_label(s) for s in agg.index], orientation="h",
            marker_color=C.METHOD_COLORS["ml"],
            hovertemplate="%{y}<br>%{x:.1%} of total importance<extra></extra>",
        ))
        fig.update_layout(title="Importance by signal", xaxis=dict(title="Share of total importance", tickformat=".0%"),
                          yaxis=dict(tickfont=dict(size=11)))
        c2.plotly_chart(C.style_fig(fig, height=620, legend_top=False), width="stretch")
        st.caption(
            "Permutation importance of the gradient-boosting risk ensemble on held-out test hours: how much "
            "PR-AUC drops when one feature is shuffled (error bars: spread over repeats). Signal totals sum the "
            "positive importances of the six statistics of each signal. Correlated features share credit, so read "
            "this as 'what the model relies on', not as root cause. Slow trends (7-day slopes) dominate, which is "
            "what a vibration analyst would expect: developing faults are drifts, not spikes."
        )

# ---- Methodology -------------------------------------------------------------------------------
with st.expander("Methodology: how these numbers were produced"):
    budget = CFG.alerts.max_false_alarms_per_asset_year
    st.markdown(
        f"""
**Leakage control.** Pumps, not hours, are the unit of evaluation. Hourly rows of one pump are highly
autocorrelated, so a random row split would let the model memorise each pump's quirks and look far better
than it is. Training uses `GroupKFold` by pump ({CFG.train.cv_folds} folds); every training-fleet score shown
here is **out-of-fold** (the pump was not in the model that scored it). The deployed model is the ensemble of
the {CFG.train.cv_folds} fold models, and it scores the **{M.get('data', {}).get('n_test_assets', 16)} held-out
pumps** that were never used for training, threshold tuning, calibration or conformal intervals. All features are
strictly causal (trailing windows only, reset at every maintenance work order).

**P-F interval event matching.** Reliability-centred maintenance describes a defect by its P point (first
detectable by inspection) and F point (functional failure). An alert raised inside [P, F - {CFG.alerts.min_lead_h} h]
is a true positive: the inspection it triggers finds the defect and there is time to plan the repair. Raised in the
last {CFG.alerts.min_lead_h} h it is *late*; inside the P window of a defect that preventive maintenance fixed it is
*PM-confirmed*; in the last {CFG.alerts.credit_window_h / DAY:.0f} days of the record without a match it is
*unresolved*; anything else is a **false alarm**: someone was sent to inspect a healthy pump.

**Re-alerting.** An alert is raised once the score stays above threshold for {CFG.alerts.debounce_h} h and
closes after {CFG.alerts.clear_h} h below it. If it is still active after {CFG.alerts.realert_h / DAY:.0f} days
it is raised again as a new work request. Without this, a model that is "always on" would get credit for every
failure at the price of a single false alarm.

**Alarm budget.** At most **{budget} nuisance alerts per pump-year** (one every {1 / budget:.0f} years). The
ISA-18.2 / EEMUA 191 alarm-management lesson: operators stop trusting, and then stop answering, a system that
cries wolf.

**Cost-optimal thresholds.** Each method's threshold minimises total maintenance cost (breakdowns + planned
repairs + inspections) on the training fleet's out-of-fold scores, subject to the alarm budget. The test fleet
never influences the choice. Conventional alarm limits keep their engineering value (score 1.0).

**Why the hybrid.** Gradient boosting learns the *signatures* of known failure modes, so its false alarms come
from healthy hours that happen to resemble one (a hot day, a load change). MSPC learns what *healthy* looks like,
so its false alarms come from benign process states it has not seen. The causes are different, so the errors are
largely independent. The hybrid score is the **lower** of the two percentiles, calibrated to a probability: an
alert needs both a failure-like pattern and a genuine departure from healthy operation. Requiring agreement removes
most of either method's false alarms while keeping the failures both of them see.
"""
    )
