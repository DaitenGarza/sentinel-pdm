"""Independent benchmark on public data (UCI AI4I 2020): does physics-first feature design hold up?

Reads the artifacts written by ``python -m sentinel_pdm ai4i`` (``sentinel_pdm.ai4i``):
``ai4i_metrics.json``, ``ai4i_importance.csv`` and ``ai4i_oof.csv``. If they are missing, the page
explains how to create them and offers to run the benchmark in place (about a minute).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # app/ folder
import common as C  # noqa: E402

C.page_setup("Real Data Benchmark", "🧪")
C.require_artifacts()

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from sentinel_pdm import ai4i as A  # noqa: E402

# Plain ML on raw tags shares the "gradient boosting only" colour of the main pipeline; the
# physics-informed approach wears Sentinel's accent colour.
FS_COLORS = {"raw": C.METHOD_COLORS["ml"], "physics": C.ACCENT}
KIND_COLORS = {"measured": C.METHOD_COLORS["ml"], "quality": C.METHOD_COLORS["ml"], "engineered": C.ACCENT}
KIND_LABELS = {"measured": "Recorded in the file", "quality": "Recorded in the file", "engineered": "Engineered (physics)"}
FAIL_COLOR = C.METHOD_COLORS["reactive"]   # the project's colour for an unplanned failure
OK_COLOR = C.MODE_COLORS["none"]
TYPE_ORDER = ["HDF", "PWF", "OSF", "TWF", "RNF", "none"]
REQUIRED_KEYS = ("dataset", "protocol", "results", "best", "operating_point", "failure_type_recall",
                 "mechanism_ceiling", "blocked_cv", "feature_sets", "models")


# --------------------------------------------------------------------------------------------
# loaders (the mtime argument invalidates the cache after a re-run)
# --------------------------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def _load(paths: dict[str, Path]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    m = _read_json(str(paths["metrics"]), _mtime(paths["metrics"])) if paths["metrics"].exists() else {}
    imp = _read_csv(str(paths["importance"]), _mtime(paths["importance"])) if paths["importance"].exists() else pd.DataFrame()
    oof = _read_csv(str(paths["oof"]), _mtime(paths["oof"])) if paths["oof"].exists() else pd.DataFrame()
    return m, imp, oof


def _offer_run(reason: str) -> None:
    """Explain how to create the artifacts, offer to run the benchmark here, then stop the page."""
    st.info(reason)
    st.markdown(
        "The benchmark downloads the public **UCI AI4I 2020** dataset once (0.5 MB, CC BY 4.0) into "
        "`data/external/` and cross-validates 3 models on 2 feature sets, which takes about a minute. "
        "From a terminal:\n\n"
        "```\npython -m sentinel_pdm ai4i            # downloads the data if needed\n"
        "python -m sentinel_pdm ai4i --offline  # uses data/external/ai4i2020.csv\n```"
    )
    if st.button("Run benchmark now", type="primary", icon=":material/play_arrow:"):
        with st.status("Running the AI4I 2020 benchmark...", expanded=True) as status:
            def progress(line: str) -> None:
                if line.startswith("["):   # step headers only; the console table is not for the UI
                    status.write(line)
            try:
                A.run_ai4i_benchmark(log=progress)
            except (OSError, ValueError) as exc:   # offline / download / data validation problems
                status.update(label="Benchmark failed", state="error")
                st.error(str(exc))
                st.stop()
            status.update(label="Benchmark finished", state="complete")
        st.cache_data.clear()
        st.rerun()
    st.stop()


def _fs_label(m: dict, fs: str) -> str:
    return m["feature_sets"][fs]["label"]


def _best_label(m: dict, fs: str) -> str:
    return f"{_fs_label(m, fs)} · {m['models'][m['best'][fs]]}"


def _add_random_guess(fig: go.Figure, rate: float) -> None:
    """Dotted line at the failure rate (the precision of a random guess), named in the legend
    rather than annotated so the label never collides with the bars."""
    fig.add_hline(y=rate, line_dash="dot", line_color=OK_COLOR)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=OK_COLOR, dash="dot"),
                    name=f"Random guess ({rate:.3f})", hoverinfo="skip")


def _num(x, fmt: str = "{:.3f}") -> str:
    return "-" if x is None or not np.isfinite(x) else fmt.format(x)


# --------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------
st.title("🧪 Independent benchmark on public data")
st.caption("Does the physics-first approach behind Sentinel hold up on a dataset this project did not generate? "
           "Same question, outside data: the UCI AI4I 2020 predictive-maintenance benchmark.")

paths = A.artifact_paths()
m, imp, oof = _load(paths)
if not m:
    _offer_run("The public-data benchmark has not been run yet.")
if any(k not in m for k in REQUIRED_KEYS):
    _offer_run("The benchmark results on disk come from an older version of this project. Re-run to refresh them.")

ds, proto, res, best = m["dataset"], m["protocol"], m["results"], m["best"]
op, ceiling = m["operating_point"], m["mechanism_ceiling"]
raw_best, phys_best = res["raw"][best["raw"]], res["physics"][best["physics"]]

# --- headline ------------------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("PR-AUC, best model", f"{phys_best['pr_auc']:.3f}",
          delta=f"{phys_best['pr_auc'] - raw_best['pr_auc']:+.3f} vs raw sensors",
          help="Average precision of the best physics-informed model vs the best raw-sensor model. "
               f"A random guess scores the failure rate ({ds['failure_rate']:.3f}).")
k2.metric("Failures caught", C.fmt_pct(op["physics"]["recall"], 1),
          delta=f"{100 * (op['physics']['recall'] - op['raw']['recall']):+.1f} pts vs raw",
          help="Recall at the best-F1 alert threshold, out-of-fold.")
k3.metric("False alarms per 10,000 runs", f"{op['physics']['false_alarms_per_10k']:.0f}",
          delta=f"{op['physics']['false_alarms_per_10k'] - op['raw']['false_alarms_per_10k']:+.0f} vs raw",
          delta_color="inverse", help="At the same best-F1 threshold. Each false alarm is an unnecessary stop and inspection.")
k4.metric("Recall at 90 % precision", C.fmt_pct(phys_best["recall_at_p90"], 0),
          delta=f"{100 * (phys_best['recall_at_p90'] - raw_best['recall_at_p90']):+.0f} pts vs raw",
          help="Share of failures caught if at least 9 in 10 alerts must be real failures.")

lift = {k: res["physics"][k]["pr_auc"] - res["raw"][k]["pr_auc"] for k in res["raw"]}
raw_max = max(r["pr_auc"] for r in res["raw"].values())
beats_all_raw = phys_best["pr_auc"] > raw_max
lin = res["physics"].get("logreg")
# How much does picking a different non-linear model change things, compared with adding features?
raw_trees = [r["pr_auc"] for k, r in res["raw"].items() if k != "logreg"]
tree_spread = max(raw_trees) - min(raw_trees) if len(raw_trees) > 1 else None
fa_raw, fa_phys = op["raw"]["false_alarms_per_10k"], op["physics"]["false_alarms_per_10k"]
with st.container(border=True):
    st.markdown("**What this shows**")
    lines = [
        "**Physics features help every model.** PR-AUC lift from adding three engineered inputs: "
        + ", ".join(f"{m['models'][k]} {v:+.3f}" for k, v in lift.items()) + ".",
    ]
    if beats_all_raw:
        lines.append(
            f"**Domain knowledge beats a stronger learner.** The best raw-sensor model "
            f"({m['models'][best['raw']].lower()}, PR-AUC {raw_best['pr_auc']:.3f}) loses to a "
            f"{m['models'][best['physics']].lower()} given physics features ({phys_best['pr_auc']:.3f})."
            + (f" Swapping one tree model for another on raw sensors moves PR-AUC by only {tree_spread:.3f}."
               if tree_spread is not None else "")
            + (f" A linear model still cannot use the features well ({lin['pr_auc']:.3f}): failures begin at "
               "thresholds and bands, which need a non-linear learner." if lin and lin["pr_auc"] < phys_best["pr_auc"] - 0.1 else ""))
    if ceiling.get("recall"):
        lines.append(
            f"**Close to the ceiling.** {C.fmt_pct(op['physics']['recall'], 1)} of failures are caught; "
            f"{C.fmt_pct(ceiling['recall'], 1)} have a deterministic cause visible in a single snapshot. The misses are "
            "tool-wear and random failures, which the dataset's author generated at random. They are the same kind of "
            "failure as the precursor-free 'sudden' failures in the pump fleet.")
    if fa_raw > 0:
        lines.append(f"**Fewer wasted call-outs.** False alarms drop from {fa_raw:.0f} to {fa_phys:.0f} per 10,000 runs "
                     f"at the best-F1 threshold ({fa_raw / max(fa_phys, 1e-9):.1f}× fewer)." if fa_phys > 0 else
                     f"**No false alarms** at the best-F1 threshold (raw sensors: {fa_raw:.0f} per 10,000 runs).")
    lines.append("This is the same design rule as the pump models, which compute pump head, temperature rise over "
                 "ambient and a hydraulic-efficiency proxy explicitly rather than leaving the model to discover them.")
    st.markdown("\n".join(f"- {s}" for s in lines))

# --- dataset -----------------------------------------------------------------------------------------
st.subheader("The dataset")
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.markdown(
        f"**{ds['name']}** (UCI Machine Learning Repository, id {ds['uci_id']}): {ds['n_rows']:,} runs of a "
        f"machining process, {ds['n_failures']} ({C.fmt_pct(ds['failure_rate'], 1)}) ending in a machine failure. "
        "Each run records air and process temperature, rotational speed, torque, tool wear and the product "
        "quality grade (L/M/H). Five failure mechanisms are documented:"
    )
    on_fail = ds.get("failure_types_on_failures", {})
    mech = pd.DataFrame([
        ("HDF", "Heat dissipation", "ΔT (process - air) < 8.6 K and speed < 1380 rpm", "Yes"),
        ("PWF", "Power", "Mechanical power < 3,500 W or > 9,000 W", "Yes"),
        ("OSF", "Overstrain", "Wear × torque > 11,000 / 12,000 / 13,000 min·N·m (L/M/H)", "Yes"),
        ("TWF", "Tool wear", "Tool replaced or fails at a random wear of 200-240 min", "Wear window only"),
        ("RNF", "Random", "0.1 % chance per run, independent of the process", "No"),
    ], columns=["Code", "Failure type", "Published mechanism", "In one snapshot?"])
    mech.insert(2, "Failures", [on_fail.get(c) for c in mech["Code"]])
    st.dataframe(mech, hide_index=True, width="stretch",
                 column_config={
                     "Code": st.column_config.TextColumn(width=50),
                     "Failure type": st.column_config.TextColumn(width=112),
                     "Failures": st.column_config.NumberColumn(width=70, help="Failed runs carrying this flag"),
                     "Published mechanism": st.column_config.TextColumn(width=350),
                     "In one snapshot?": st.column_config.TextColumn(
                         width=135, help="Can a single run's readings reveal the cause, or only a history / chance?"),
                 })
    q = ds.get("quirks", {})
    st.caption(f"Data as published: {q.get('failures_without_type', 0)} failures carry no type flag, "
               f"{q.get('type_flags_without_failure', 0)} random-failure flags sit on runs not marked as failures, and "
               f"{q.get('multi_type_failures', 0)} runs have more than one flag. Nothing is relabelled.")
with c2:
    st.info(
        "**About \"real\" data.** UCI publishes AI4I 2020 as a synthetic dataset that “reflects real predictive "
        "maintenance data encountered in industry”, because real failure data is rarely publishable. What matters "
        "here is that it was built by an independent researcher, with documented failure physics. It tests the "
        "method on data this project had no hand in.",
        icon=":material/info:",
    )
    st.markdown(
        f"**Citation.** {ds['citation']}  \n"
        f"**Paper.** {ds['paper']} doi:{ds['paper_doi']}  \n"
        f"**Licence.** {ds['license']} · [dataset page]({ds['url']})"
    )

# --- results -------------------------------------------------------------------------------------------
st.subheader("Raw sensors vs physics-informed features")
st.caption(
    f"Target: *Machine failure*. Stratified {proto['n_splits']}-fold cross-validation × {proto['n_repeats']} repeats; "
    "every model and feature set sees the same folds. Raw = the five measured variables + product quality; "
    "physics-informed adds temperature rise, mechanical power and overstrain. Failure-type flags, row id and "
    "serial number are never inputs (they would leak the answer)."
)
models = list(m["models"])
fig = go.Figure()
for fs in ("raw", "physics"):
    vals = [res[fs][k]["pr_auc"] for k in models]
    fig.add_bar(name=_fs_label(m, fs), x=[m["models"][k] for k in models], y=vals,
                error_y=dict(type="data", array=[res[fs][k]["pr_auc_std"] for k in models], thickness=1.2, width=4),
                marker_color=FS_COLORS[fs], text=[f"{v:.3f}" for v in vals], textposition="outside",
                hovertemplate="%{x}<br>" + _fs_label(m, fs) + ": PR-AUC %{y:.3f}<extra></extra>")
_add_random_guess(fig, ds["failure_rate"])
fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06, yaxis_title="PR-AUC (average precision)",
                  yaxis_range=[0, 1.08], legend_title_text="")
st.plotly_chart(C.style_fig(fig, height=360), width="stretch")

rows = []
for fs in ("raw", "physics"):
    for k in models:
        r = res[fs][k]
        rows.append({
            "Features": _fs_label(m, fs), "Model": m["models"][k] + (" ★" if best[fs] == k else ""),
            "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "PR-AUC sd": r["pr_auc_std"],
            "F1 (best threshold)": r["f1_best"], "F1 (threshold from other folds)": r["f1_xfit"],
            "Caught at ≥ 90 % precision": 100 * r["recall_at_p90"] if r["p90_reached"] else None,
            "Fit time (s)": r["fit_time_s"],
        })
table = pd.DataFrame(rows)
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        **{c: st.column_config.NumberColumn(format="%.3f") for c in
           ("ROC-AUC", "PR-AUC", "PR-AUC sd", "F1 (best threshold)", "F1 (threshold from other folds)")},
        "Caught at ≥ 90 % precision": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=100,
            help="Recall with alert precision of at least 90 %. Empty: the model never reaches 90 % precision."),
        "Fit time (s)": st.column_config.NumberColumn(format="%.1f", help=(
            f"Total for all {proto['n_splits'] * proto['n_repeats']} cross-validation fits")),
    },
)
st.caption(
    f"Mean over {proto['n_repeats']} repeats; ★ = best model of each feature set by PR-AUC. PR-AUC is the headline "
    f"metric because only {C.fmt_pct(ds['failure_rate'], 1)} of runs fail: ROC-AUC looks excellent for every model "
    "and hides the false alarms. *F1 (best threshold)* picks the threshold with hindsight; *F1 (threshold from other "
    "folds)* sets it on the other folds first, which is what a deployed system would achieve."
)

# --- PR curves + failure-type recall ---------------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Precision-recall of the best models**")
    if oof.empty or not {"machine_failure", "prob_raw", "prob_physics"} <= set(oof.columns):
        st.warning("Out-of-fold predictions (ai4i_oof.csv) are missing. Re-run the benchmark to draw the curves.")
    else:
        y = oof["machine_failure"].to_numpy()
        fig = go.Figure()
        for fs in ("raw", "physics"):
            p = oof[f"prob_{fs}"].to_numpy()
            prec, rec, thr = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            fig.add_scatter(x=rec, y=prec, mode="lines", line=dict(color=FS_COLORS[fs], width=2),
                            name=f"{_fs_label(m, fs)} (AP {ap:.3f})",
                            hovertemplate="recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>")
            f1 = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1],
                           out=np.zeros(len(thr)), where=(prec[:-1] + rec[:-1]) > 0)
            i = int(np.argmax(f1))
            _, t_best = A.best_f1(y, p)   # same threshold rule as the benchmark itself
            fig.add_scatter(x=[rec[i]], y=[prec[i]], mode="markers", showlegend=False,
                            marker=dict(color=FS_COLORS[fs], size=11, line=dict(color="#0f1117", width=2)),
                            hovertemplate=(f"best-F1 point ({_fs_label(m, fs)})<br>threshold {t_best:.2f}<br>"
                                           "recall %{x:.1%}<br>precision %{y:.1%}<extra></extra>"))
        if ceiling.get("recall"):
            fig.add_vline(x=ceiling["recall"], line_dash="dash", line_color=OK_COLOR,
                          annotation_text="mechanism ceiling", annotation_position="bottom left")
        _add_random_guess(fig, ds["failure_rate"])
        fig.update_layout(xaxis_title="Recall (share of failures caught)", yaxis_title="Precision (share of alerts that are real)",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", yaxis_range=[0, 1.04], xaxis_range=[0, 1.01])
        st.plotly_chart(C.style_fig(fig, height=420), width="stretch")
        st.caption(f"Best models: {m['models'][best['raw']].lower()} on raw sensors, "
                   f"{m['models'][best['physics']].lower()} on physics-informed features. Out-of-fold probabilities "
                   "of the first CV repeat; dots mark the best-F1 operating point. Right of the dashed line, the "
                   "remaining failures have no deterministic snapshot cause (tool wear and random failures), so "
                   "precision collapses for every model.")

with right:
    st.markdown("**Which failures are caught? Recall by type at the best-F1 threshold**")
    ftr = m["failure_type_recall"]
    types = [t for t in TYPE_ORDER if t in ftr["physics"]["types"]]
    cats = [f"{A.FAILURE_TYPE_LABELS[t]}<br>{t if t != 'none' else 'no flag'} · n={ftr['physics']['types'][t]['n']}"
            for t in types]
    fig = go.Figure()
    for fs in ("raw", "physics"):
        tv = ftr[fs]["types"]
        vals = [tv[t]["recall"] for t in types]
        fig.add_bar(name=_fs_label(m, fs), y=cats, x=vals, orientation="h", marker_color=FS_COLORS[fs],
                    text=[_num(v, "{:.0%}") for v in vals], textposition="outside",
                    hovertemplate="%{y}<br>recall %{x:.1%}<extra>" + _fs_label(m, fs) + "</extra>")
    n_pred = sum(A.PREDICTABLE_TYPES[t] for t in types)
    if n_pred < len(types):
        fig.add_hrect(y0=n_pred - 0.5, y1=len(types) - 0.5, fillcolor=OK_COLOR, opacity=0.12, line_width=0)
    fig.update_layout(barmode="group", bargap=0.2, xaxis_tickformat=".0%", xaxis_range=[0, 1.14],
                      xaxis_title="Share of failures caught", legend_title_text="",
                      yaxis=dict(autorange="reversed", automargin=True))
    st.plotly_chart(C.style_fig(fig, height=420), width="stretch")
    st.caption("Physics features mainly close the gap on power and overstrain failures, whose boundaries are curves "
               "in the raw torque-speed and wear-torque planes. Shaded: failures with no snapshot signature. Tool-wear "
               "and random failures stay near zero for both: as in a plant, they call for a policy (replace tools "
               "before 200 min), not a predictor.")

# --- permutation importance ---------------------------------------------------------------------------------
st.subheader("What the models rely on")
if imp.empty:
    st.warning("Permutation importance (ai4i_importance.csv) is missing. Re-run the benchmark to show it.")
else:
    cols = st.columns(2, gap="large")
    for col, fs in zip(cols, ("raw", "physics")):
        sub = imp[imp["feature_set"] == fs].sort_values("importance")
        with col:
            st.markdown(f"**{_best_label(m, fs)}**")
            if sub.empty:
                st.caption("No importance rows for this feature set.")
                continue
            fig = go.Figure()
            for kind in dict.fromkeys(KIND_LABELS[k] for k in sub["kind"]):
                part = sub[sub["kind"].map(KIND_LABELS) == kind]
                fig.add_bar(x=part["importance"], y=part["label"], orientation="h", name=kind,
                            marker_color=KIND_COLORS[part["kind"].iloc[0]],
                            error_x=dict(type="data", array=part["std"], thickness=1.2, width=3),
                            hovertemplate="%{y}<br>PR-AUC drop %{x:.3f}<extra></extra>")
            fig.update_layout(xaxis_title="PR-AUC drop when shuffled", barmode="overlay", legend_title_text="",
                              yaxis=dict(categoryorder="array", categoryarray=sub["label"].tolist(), automargin=True))
            st.plotly_chart(C.style_fig(fig, height=330), width="stretch")
    st.caption(
        "Permutation importance on held-out folds: how much PR-AUC falls when one input is shuffled (product quality "
        "is shuffled as one input). The raw-sensor model needs every column at once to reconstruct the mechanisms; the "
        "physics model leans on the engineered inputs. Speed stays important because the heat-dissipation rule also "
        "needs speed < 1380 rpm. Raw torque now looks unimportant only because its information lives inside power "
        "and overstrain."
    )

# --- engineered features explained -------------------------------------------------------------------------------
st.subheader("The engineered features")
FORMULAS = [
    (r"\Delta T = T_\mathrm{process} - T_\mathrm{air}",
     "**Temperature rise** (K). Heat leaves the cutting zone through this gradient. Below 8.6 K at low speed "
     "(< 1380 rpm) it cannot be carried away: heat-dissipation failure."),
    (r"P = \tau \cdot n \cdot 2\pi / 60",
     "**Mechanical power** (W) from torque τ (N·m) and speed n (rpm). Outside 3,500-9,000 W the process stalls or "
     "overloads: power failure."),
    (r"S = t_\mathrm{wear} \cdot \tau",
     "**Overstrain** (min·N·m), a proxy for load on a worn tool. Above 11,000 / 12,000 / 13,000 for L / M / H quality "
     "products: overstrain failure."),
]
for formula, text in FORMULAS:
    fcol, tcol = st.columns([2, 3], vertical_alignment="center")
    fcol.latex(formula)
    tcol.markdown(text)
st.caption("The features encode *which physical quantity matters*, never the published thresholds; the models "
           "still learn where failure starts from the data.")

if oof.empty:
    st.warning("Out-of-fold data (ai4i_oof.csv) is missing. Re-run the benchmark to show the mechanism plots.")
else:
    def _mechanism_scatter(x: str, y: str, flag: str, xlab: str, ylab: str) -> go.Figure:
        """All runs in grey with runs carrying ``flag`` drawn on top in the failure colour."""
        hit = oof[flag] == 1
        fig = go.Figure()
        fig.add_scattergl(x=oof.loc[~hit, x], y=oof.loc[~hit, y], mode="markers", name="No such failure",
                          marker=dict(color=OK_COLOR, size=4, opacity=0.35),
                          hovertemplate=f"{xlab} %{{x}}<br>{ylab} %{{y}}<extra></extra>")
        fig.add_scattergl(x=oof.loc[hit, x], y=oof.loc[hit, y], mode="markers",
                          name=f"{A.FAILURE_TYPE_LABELS[flag]} failure ({flag}, n={int(hit.sum())})",
                          marker=dict(color=FAIL_COLOR, size=8, line=dict(color="#0f1117", width=1)),
                          hovertemplate=f"{flag}<br>{xlab} %{{x}}<br>{ylab} %{{y}}<extra></extra>")
        fig.update_layout(xaxis_title=xlab, yaxis_title=ylab)
        return fig

    t_hdf, t_pwf, t_osf, t_twf = st.tabs(["Heat dissipation (HDF)", "Power (PWF)", "Overstrain (OSF)", "Tool wear (TWF)"])
    with t_hdf:
        fig = _mechanism_scatter("speed_rpm", "temp_diff_K", "HDF", "Rotational speed (rpm)", "Temperature rise ΔT (K)")
        # Shape opacity would fade the outline along with the fill; a translucent fill keeps the
        # dashed boundary visible.
        fig.add_shape(type="rect", x0=oof["speed_rpm"].min(), x1=1380, y0=oof["temp_diff_K"].min() - 0.1, y1=8.6,
                      line=dict(color=FAIL_COLOR, dash="dash", width=1.5), fillcolor="rgba(239, 68, 68, 0.08)",
                      layer="below")
        fig.add_annotation(x=1380, y=8.6, text="ΔT < 8.6 K and n < 1380 rpm", showarrow=False, xanchor="left",
                           yanchor="bottom", xshift=4, font=dict(size=12), bgcolor="rgba(15, 17, 23, 0.85)")
        st.plotly_chart(C.style_fig(fig, height=420), width="stretch")
        st.caption("In raw tags this boundary spans three columns (two temperatures and speed). With ΔT it becomes a "
                   "rectangle in two, which a tree learns in two splits.")
    with t_pwf:
        fig = _mechanism_scatter("speed_rpm", "torque_Nm", "PWF", "Rotational speed (rpm)", "Torque (N·m)")
        n_grid = np.linspace(max(oof["speed_rpm"].min(), 1), oof["speed_rpm"].max(), 200)
        for watts in (3500, 9000):
            fig.add_scatter(x=n_grid, y=watts * 60 / (2 * np.pi * n_grid), mode="lines", showlegend=False,
                            line=dict(color=FAIL_COLOR, dash="dash", width=1.5), hoverinfo="skip")
            fig.add_annotation(x=n_grid[-1], y=watts * 60 / (2 * np.pi * n_grid[-1]), text=f"{watts:,} W",
                               showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12))
        fig.update_yaxes(range=[0, oof["torque_Nm"].max() * 1.05])
        st.plotly_chart(C.style_fig(fig, height=420), width="stretch")
        st.caption("Power failures lie outside the band between two constant-power hyperbolas. No single threshold on "
                   "torque or on speed separates them, but one engineered feature (power) does.")
    with t_osf:
        fig = _mechanism_scatter("tool_wear_min", "torque_Nm", "OSF", "Tool wear (min)", "Torque (N·m)")
        w_grid = np.linspace(max(oof["tool_wear_min"].quantile(0.25), 1), oof["tool_wear_min"].max(), 200)
        for grade, limit in (("L", 11000), ("M", 12000), ("H", 13000)):
            fig.add_scatter(x=w_grid, y=limit / w_grid, mode="lines", showlegend=False, hoverinfo="skip",
                            line=dict(color=FAIL_COLOR, dash="dash" if grade == "L" else "dot", width=1.3))
            fig.add_annotation(x=w_grid[-1], y=limit / w_grid[-1], text=f"{grade}: {limit:,}", showarrow=False,
                               xanchor="left", font=dict(size=11))
        fig.update_yaxes(range=[0, oof["torque_Nm"].max() * 1.05])
        st.plotly_chart(C.style_fig(fig, height=420), width="stretch")
        st.caption("Overstrain failures sit above constant wear × torque curves whose level depends on product quality "
                   "(L / M / H), which is why the quality grade is an input in both feature sets.")
    with t_twf:
        in_window = oof["tool_wear_min"].between(200, 240)
        twf = oof["TWF"] == 1
        fig = go.Figure()
        bins = dict(start=0, end=float(oof["tool_wear_min"].max()) + 5, size=5)
        fig.add_histogram(x=oof["tool_wear_min"], xbins=bins, name="All runs", marker_color=OK_COLOR,
                          hovertemplate="wear %{x} min<br>%{y} runs<extra></extra>")
        fig.add_histogram(x=oof.loc[twf, "tool_wear_min"], xbins=bins, name=f"Tool-wear failures (n={int(twf.sum())})",
                          marker_color=FAIL_COLOR, hovertemplate="wear %{x} min<br>%{y} failures<extra></extra>")
        fig.add_vrect(x0=200, x1=240, fillcolor=FAIL_COLOR, opacity=0.08, line_width=0,
                      annotation_text="200-240 min window", annotation_position="top left")
        fig.update_layout(barmode="overlay", xaxis_title="Tool wear (min)", yaxis_title="Runs", yaxis_type="log")
        st.plotly_chart(C.style_fig(fig, height=420), width="stretch")
        st.caption(f"{int(in_window.sum()):,} runs reached the 200-240 min window, but only {int((twf & in_window).sum())} "
                   "ended in a tool-wear failure (log scale). A snapshot can flag the window, not the failure. That is "
                   "why no model catches TWF, and why plants manage tool life with a replacement interval.")

# --- method notes ---------------------------------------------------------------------------------------------
with st.expander("Method notes and reproducibility"):
    blk = m["blocked_cv"]
    blk_txt = "; ".join(f"{_best_label(m, fs)}: PR-AUC {b['pr_auc']:.3f} vs {b['pr_auc_random_cv']:.3f} with random folds"
                        for fs, b in blk.items())
    st.markdown(
        f"""
- **Target and leakage control.** Target *Machine failure*. Never used as inputs: {", ".join(f"`{c}`" for c in proto["excluded_columns"])}.
  The failure-type flags are the answer, and the row id and serial number identify rows, not physics.
- **Cross-validation.** Repeated stratified {proto['n_splits']}-fold × {proto['n_repeats']} (seed {proto['random_state']}),
  with the same folds for every model and feature set, so differences are paired. Models use standard settings and
  were not tuned on this data, which would bias the comparison.
- **Time-blocked check.** Rows are a time series (tool wear accumulates; temperatures drift), so random folds put
  near-identical neighbours in training and test. Re-scoring the best models with contiguous {proto['block_rows']}-row
  blocks kept together gives {blk_txt}. The result does not depend on neighbour leakage.
- **Mechanism ceiling.** {ceiling['deterministic_failures']} of {ceiling['failures']} failures carry a heat-dissipation, power or
  overstrain flag, the only causes a single snapshot can reveal.
- **Reproduce.** `python -m sentinel_pdm ai4i` (about {m.get('runtime_s', 60):.0f} s; generated {m.get('generated_at', '?')[:16].replace('T', ' ')} UTC
  with scikit-learn {m.get('environment', {}).get('sklearn', '?')}). Data SHA-256: `{ds.get('sha256') or 'n/a'}`.
"""
    )
