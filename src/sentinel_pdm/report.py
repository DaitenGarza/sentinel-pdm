"""Static results report: ``reports/RESULTS.md`` and ``reports/model_report.html``.

Why a static report next to the dashboard? The dashboard is for exploring; a reviewer (a
reliability lead, a hiring manager, a CI job) needs one document that states what was measured,
on which pumps and under which assumptions, readable on GitHub without installing anything.

Everything is derived from the artifacts of the last ``python -m sentinel_pdm train`` run.
Headline numbers are copied from ``metrics.json`` (never recomputed, so the report cannot drift
from what the pipeline evaluated); only the curves that need hourly scores (ROC, PR,
calibration, RUL, the example timeline) are drawn from ``predictions.parquet``, test fleet only.

Optional inputs written by other commands are included when present:
``reports/robustness.json`` (``python -m sentinel_pdm robustness``) and
``artifacts/ai4i_metrics.json`` (``python -m sentinel_pdm ai4i``).
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

from . import __version__
from .config import ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR
from .explain import MODE_LABELS, SIGNAL_INFO
from .features import signal_of

RESULTS_MD = "RESULTS.md"
REPORT_HTML = "model_report.html"

# Same palette as the dashboard (app/common.py) so screenshots and the report look like one product.
METHOD_COLORS = {"hybrid": "#f59e0b", "ml": "#60a5fa", "mspc": "#34d399", "iforest": "#c084fc",
                 "rule": "#9ca3af", "reactive": "#ef4444"}
MODE_COLORS = {"bearing": "#f97316", "misalignment": "#3b82f6", "cavitation": "#14b8a6", "winding": "#a855f7",
               "sudden": "#ef4444", "none": "#6b7280"}
BG, PANEL, BORDER, TEXT, MUTED, ACCENT = "#0f1117", "#1a1d27", "#2a2f3d", "#e5e7eb", "#9ca3af", "#f59e0b"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

# Hourly columns needed from predictions.parquet (scores are added from metrics["score_columns"]).
_PRED_BASE = ["asset_id", "hour", "timestamp", "running", "label", "ttf", "next_predictable",
              "rul_p10", "rul_p50", "rul_p90"]
_STAT_TEXT = {"_m24": "24 h mean", "_sd24": "24 h std. dev.", "_m168": "7-day mean", "_sl24": "24 h trend",
              "_sl168": "7-day trend", "_dev": "deviation from baseline"}
_CONTEXT_TEXT = {"load_pct": "Process load (current hour)", "ambient_temp": "Ambient temperature",
                 "speed_rpm": "Shaft speed", "hours_since_maint": "Hours since last maintenance",
                 "run_hours_since_maint": "Running hours since last maintenance"}
_SHORT_MODE = {"bearing": "Bearing", "misalignment": "Misalignment", "cavitation": "Cavitation",
               "winding": "Winding"}
# Calibration bins: most hours sit near zero risk, so the low end is binned finely.
_CAL_EDGES = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])
_CAL_MIN_ROWS = 50
_THRESHOLD_UNITS = {"mspc": "x 99 % control limit", "iforest": "x healthy P99", "rule": "x alarm limit"}
_CURVE_POINTS = 600   # ROC / PR curves are thinned to this many points to keep the HTML small


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
@dataclass
class ReportData:
    """Everything the report needs, loaded once."""

    metrics: dict
    preds: pd.DataFrame                  # test-fleet hourly scores (needed columns only)
    alerts: pd.DataFrame
    outcomes: pd.DataFrame
    importance: pd.DataFrame
    curve: pd.DataFrame
    events: pd.DataFrame
    quality: pd.DataFrame
    data_dir: Path
    assets: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: dict | None = None
    ai4i: dict | None = None
    warnings: list[str] = field(default_factory=list)
    optional: list = field(default_factory=list)   # rendered optional sections, see optional_sections()

    @property
    def methods(self) -> dict[str, str]:
        return self.metrics.get("methods", {})

    @property
    def score_columns(self) -> dict[str, str]:
        return self.metrics.get("score_columns", {})


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_optional_json(path: Path, warnings: list[str]) -> dict | None:
    """Optional inputs are produced by other commands (maybe concurrently): absence is normal."""
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Skipped {path.name}: could not be read ({exc}).")
        return None
    return obj if isinstance(obj, dict) else None


def load_report_data(artifacts_dir: Path = ARTIFACTS_DIR, data_dir: Path = DATA_DIR,
                     reports_dir: Path = REPORTS_DIR) -> ReportData:
    """Read the artifacts of the last training run (test-fleet rows of the hourly predictions only)."""
    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No results at {metrics_path}. Run `python -m sentinel_pdm train` first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    warnings: list[str] = []

    preds = pd.DataFrame()
    pred_path = artifacts_dir / "predictions.parquet"
    if pred_path.exists():
        available = set(pq.read_schema(pred_path).names)
        wanted = list(dict.fromkeys(_PRED_BASE + list(metrics.get("score_columns", {}).values())))
        cols = [c for c in wanted if c in available]
        filters = [("split", "==", "test")] if "split" in available else None
        preds = pd.read_parquet(pred_path, columns=cols, filters=filters)
    else:
        warnings.append("predictions.parquet not found: curve-based figures are omitted.")

    d = ReportData(
        metrics=metrics,
        preds=preds,
        alerts=_read_csv(artifacts_dir / "alerts.csv"),
        outcomes=_read_csv(artifacts_dir / "failure_outcomes.csv"),
        importance=_read_csv(artifacts_dir / "feature_importance.csv"),
        curve=_read_csv(artifacts_dir / "threshold_curve.csv"),
        events=_read_csv(data_dir / "events.csv"),
        quality=_read_csv(data_dir / "quality.csv"),
        data_dir=data_dir,
        assets=_read_csv(data_dir / "assets.csv"),
        robustness=_read_optional_json(reports_dir / "robustness.json", warnings),
        ai4i=_read_optional_json(artifacts_dir / "ai4i_metrics.json", warnings),
        warnings=warnings,
    )
    d.optional = optional_sections(d, reports_dir)
    return d


# --------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(x)


def _num(x: Any, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if _finite(x) else "-"


def _pct(x: Any, digits: int = 1, fraction: bool = False) -> str:
    """Percentage; ``fraction=True`` when x is a 0..1 share rather than already in percent."""
    if not _finite(x):
        return "-"
    return f"{(100 * x if fraction else x):.{digits}f}%"


def _days(hours: Any) -> str:
    return f"{hours / 24:.1f}" if _finite(hours) else "-"


def _money(x: Any) -> str:
    if not _finite(x):
        return "-"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def _usd(x: Any) -> str:
    """Exact dollar amount (cost assumptions), as opposed to the rounded totals of ``_money``."""
    return f"${x:,.0f}" if _finite(x) else "-"


def _int(x: Any) -> str:
    return f"{int(x):,}" if _finite(x) else "-"


def _auto(v: Any) -> str:
    """Format a value of unknown meaning (used for the optional JSON sections)."""
    if isinstance(v, bool) or v is None:
        return "-" if v is None else str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if _finite(v):
        a = abs(v)
        return f"{v:,.0f}" if a >= 1000 else f"{v:.1f}" if a >= 10 else f"{v:.3f}"
    if isinstance(v, float):
        return "-"
    return str(v)


def _md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---:" if a == "c" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _html_table(header: list[str], rows: list[list[str]], align: str | None = None,
                highlight: int | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    cls = {"l": "", "r": ' class="num"', "c": ' class="ctr"'}
    head = "".join(f"<th{cls[a]}>{html.escape(h)}</th>" for h, a in zip(header, align))
    body = []
    for i, r in enumerate(rows):
        tr = ' class="hl"' if i == highlight else ""
        body.append(f"<tr{tr}>" + "".join(f"<td{cls[a]}>{html.escape(str(c))}</td>" for c, a in zip(r, align))
                    + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def describe_feature(feature: str) -> str:
    """Plain-English name of a model feature, e.g. ``vib_hf_sl168`` -> 'High-frequency bearing band, 7-day trend'."""
    if feature in _CONTEXT_TEXT:
        return _CONTEXT_TEXT[feature]
    sig = signal_of(feature)
    stat = _STAT_TEXT.get(feature[len(sig):], feature[len(sig):].lstrip("_"))
    name = SIGNAL_INFO.get(sig, (sig, ""))[0]
    return f"{name}, {stat}"


def _humanise_key(key: str) -> str:
    return key.replace("_", " ").replace(".", " · ").strip().capitalize() if key else "Summary"


# --------------------------------------------------------------------------------------------
# derived numbers shared by the Markdown and HTML outputs
# --------------------------------------------------------------------------------------------
def _test(m: dict, *keys: str, default: Any = None) -> Any:
    node: Any = m.get("test", {})
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _horizon_text(m: dict) -> str:
    h = m.get("config", {}).get("labels", {}).get("horizon_h", 168)
    return f"{h / 24:g} days" if h % 24 == 0 else f"{h} h"


def _confusion_correct(dx: dict) -> tuple[int, int] | None:
    cm = dx.get("confusion", {}).get("matrix")
    if not cm:
        return None
    arr = np.asarray(cm, dtype=int)
    return int(np.trace(arr)), int(arr.sum())


def _fa_needed(curve: pd.DataFrame, method: str, caught_target: int) -> int | None:
    """Fewest out-of-fold false alarms with which ``method`` catches at least ``caught_target`` failures."""
    if curve.empty or "method" not in curve:
        return None
    c = curve[(curve["method"] == method) & (curve["caught"] >= caught_target)]
    return int(c["false_alarms"].min()) if len(c) else None


def headline_sentences(d: ReportData) -> list[str]:
    """Plain-language summary built from metrics.json (so it stays true after retraining)."""
    m = d.metrics
    data, ev, econ = m.get("data", {}), _test(m, "event", default={}), _test(m, "economics", default={})
    hy, rule = ev.get("hybrid", {}), ev.get("rule", {})
    years = _test(m, "asset_years")
    out = []
    if hy:
        lead = (f" and a median warning of {_days(hy['median_lead_h'])} days before functional failure"
                if _finite(hy.get("median_lead_h")) else "")
        out.append(
            f"On {data.get('n_test_assets', '?')} held-out pumps ({_num(years, 0)} pump-years, never seen in "
            f"training) with {hy.get('failures', '?')} failures, of which {hy.get('predictable_failures', '?')} "
            f"had a measurable precursor, the Sentinel hybrid caught {hy.get('caught')}/{hy.get('failures')} "
            f"({hy.get('caught_predictable')}/{hy.get('predictable_failures')} predictable) with "
            f"{hy.get('false_alarms')} false alarms{lead}."
        )
    if rule:
        lead = (f" with a median warning of {_num(rule['median_lead_h'], 0)} h, usually too late to plan the repair"
                if _finite(rule.get("median_lead_h")) else "")
        out.append(
            f"Conventional alarm limits (ISO 10816 zone C, bearing / winding RTD alarms) caught "
            f"{rule.get('caught')}/{rule.get('failures')} on the same pumps{lead}."
        )
    multi_fleet = robustness_sentence(d.robustness)
    if multi_fleet:
        out.append(multi_fleet)
    if "hybrid" in econ and "reactive" in econ:
        e, r = econ["hybrid"], econ["reactive"]
        out.append(
            f"At the stated cost assumptions this is a {_pct(e.get('savings_pct'))} maintenance-cost saving "
            f"versus run-to-failure ({_money(r.get('total_cost'))} -> {_money(e.get('total_cost'))} for the test "
            f"fleet) and {_pct(e.get('downtime_reduction_pct'))} less downtime."
        )
    cv_hy = m.get("cv", {}).get("event", {}).get("hybrid", {})
    if cv_hy and not d.curve.empty:
        fa_ml = _fa_needed(d.curve, "ml", int(cv_hy.get("caught", 0)))
        fa_hy = _fa_needed(d.curve, "hybrid", int(cv_hy.get("caught", 0)))
        if fa_ml is not None and fa_hy is not None:
            out.append(
                f"In cross-validation on the training fleet, catching {cv_hy.get('caught')}/{cv_hy.get('failures')} "
                f"failures cost the hybrid {fa_hy} false alarms and gradient boosting alone {fa_ml}: requiring the "
                f"supervised model and MSPC to agree is what removes nuisance alerts."
            )
    return out


def _economic_components(m: dict) -> pd.DataFrame:
    """Cost of every policy split into unplanned breakdowns, planned repairs and false-alarm inspections."""
    econ = _test(m, "economics", default={})
    fa_cost = m.get("config", {}).get("costs", {}).get("false_alarm_cost", 0.0)
    rows = []
    for key in ["reactive", *m.get("methods", {})]:
        e = econ.get(key)
        if not e:
            continue
        rows.append({
            "key": key,
            "policy": "Run to failure" if key == "reactive" else m["methods"][key],
            "unplanned": e["unplanned_failures"] * e["unplanned_cost_each"],
            "planned": e["planned_repairs"] * e["planned_cost_each"],
            "false_alarms": e["false_alarms"] * fa_cost,
            "total": e["total_cost"],
            "savings_pct": e.get("savings_pct"),
        })
    return pd.DataFrame(rows)


def pick_example_failure(d: ReportData) -> pd.Series | None:
    """A representative caught failure for the timeline figure.

    Prefers a bearing failure (the commonest mode, and vib_hf is its classic indicator) whose
    warning time is closest to the hybrid's median lead, so the example is typical, not cherry-picked.
    """
    fo = d.outcomes
    if fo.empty or not {"method", "split", "caught", "lead_h"}.issubset(fo.columns):
        return None
    caught = fo[(fo["method"] == "hybrid") & (fo["split"] == "test") & fo["caught"].astype(bool)
                & fo["lead_h"].notna()]
    if caught.empty:
        return None
    if not d.preds.empty:
        caught = caught[caught["asset_id"].isin(set(d.preds["asset_id"].unique()))]
        if caught.empty:
            return None
    pool = caught[caught["mode"] == "bearing"] if (caught["mode"] == "bearing").any() else caught
    median_lead = float(pool["lead_h"].median())
    return pool.loc[(pool["lead_h"] - median_lead).abs().idxmin()]


# --------------------------------------------------------------------------------------------
# optional sections (robustness study, AI4I benchmark). Their JSON belongs to other modules, so
# each has a dedicated renderer for the documented layout and falls back to a generic table dump
# if that layout ever changes; a schema change must never break the main report.
# --------------------------------------------------------------------------------------------
@dataclass
class Block:
    """One piece of a report section, rendered to both Markdown and HTML."""

    kind: str                                   # "p" | "ul" | "table" | "link"
    text: str = ""
    items: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    align: str | None = None
    href: str = ""


def _mean_sd(stat: Any, fmt: str = "{:.2f}", fraction_pct: bool = False, unit: str = "") -> str:
    """'mean ± sd' from a {mean, std} dict (sd omitted when undefined, e.g. a single fleet).

    ``fraction_pct`` scales a 0..1 share to percent; the unit goes after the spread
    ("65.6 ± 6.6%") so the reader never wonders whether the ± is in points or relative.
    """
    if not isinstance(stat, dict) or not _finite(stat.get("mean")):
        return "-"
    scale, unit = (100.0, "%") if fraction_pct else (1.0, unit)
    out = fmt.format(scale * stat["mean"])
    if _finite(stat.get("std")):
        out += " ± " + fmt.format(scale * stat["std"])
    return out + unit


def robustness_sentence(r: dict | None) -> str | None:
    """One-line multi-fleet headline for the summary, or None when the study is absent or unreadable.

    The single held-out fleet has only ~30 failures, so the pooled multi-fleet counts are the
    stronger evidence; they belong next to the single-fleet headline, not only in a later section.
    """
    if not r:
        return None
    try:
        pooled, runs = r["aggregate"]["pooled"], r["runs"]
        hy, rule = pooled["hybrid"], pooled["rule"]
        others = [f"{label} {pooled[key]['false_alarms']}" for key, label in
                  (("ml", "gradient boosting alone"), ("mspc", "MSPC alone")) if key in pooled]
        return (f"Repeated on {len(runs)} independent simulated fleets ({hy['failures']} held-out failures, "
                f"{_num(hy['asset_years'], 0)} pump-years), the hybrid caught {hy['caught_predictable']}/"
                f"{hy['predictable_failures']} predictable failures with {hy['false_alarms']} false alarms"
                + (f" ({'; '.join(others)})" if others else "")
                + f"; alarm limits caught {rule['caught_predictable']}/{rule['predictable_failures']}.")
    except (KeyError, TypeError, ValueError):
        return None


def robustness_blocks(r: dict, reports_dir: Path) -> list[Block]:
    """Summary of ``reports/robustness.json`` (layout documented in ``robustness.py``)."""
    setup, agg, runs = r["setup"], r["aggregate"], r["runs"]
    methods = r.get("methods") or {}
    pooled = agg["pooled"]
    seeds = ", ".join(str(s) for s in setup.get("seeds", [run["seed"] for run in runs]))
    blocks = [Block("p", (
        f"The entire pipeline (simulation, cleaning, features, grouped cross-validation, threshold choice, "
        f"evaluation) re-run on {len(runs)} statistically independent fleets (seeds {seeds}; "
        f"{setup['n_assets']} pumps x {setup['hours']:,} h each, {setup['n_test_assets']} held out per fleet). "
        f"Pooled counts weight every failure equally; ± is the standard deviation across fleets."))]
    rows = []
    for key, label in methods.items():
        st, p = agg["methods"][key], pooled[key]
        rows.append([label, f"{p['caught_predictable']}/{p['predictable_failures']}",
                     _mean_sd(st["detection_rate_predictable"], "{:.0f}", fraction_pct=True), str(p["false_alarms"]),
                     _mean_sd(st["false_alarms_per_asset_year"], "{:.2f}"), _mean_sd(st["median_lead_h"], "{:.0f}"),
                     _mean_sd(st["savings_pct"], "{:.1f}", unit="%")])
    blocks.append(Block("table", header=["Method", "Predictable caught (pooled)", "Detection per fleet",
                                         "False alarms (pooled)", "False alarms / pump-year", "Median lead (h)",
                                         "Cost saving"], rows=rows))
    items = []
    h, rule = pooled.get("hybrid"), pooled.get("rule")
    if h:
        text = (f"Pooled over all fleets the Sentinel hybrid caught {h['caught_predictable']} of "
                f"{h['predictable_failures']} predictable failures "
                f"({_pct(h.get('detection_rate_predictable'), 1, True)}) with {h['false_alarms']} false alarms in "
                f"{_num(h.get('asset_years'), 0)} pump-years")
        if rule:
            text += (f"; conventional alarm limits caught {rule['caught_predictable']} of "
                     f"{rule['predictable_failures']} ({_pct(rule.get('detection_rate_predictable'), 1, True)})")
        items.append(text + ".")
    rul, dx = agg.get("rul", {}), agg.get("diagnosis", {})
    if rul or dx:
        target = setup.get("config", {}).get("train", {}).get("rul_interval_coverage")
        target_text = f" (target {_pct(target, 0, True)})" if _finite(target) else ""
        items.append(f"Remaining useful life in the last 7 days before failure: median-estimate error "
                     f"{_mean_sd(rul.get('mae_h'), '{:.0f}', unit=' h')}, P10-P90 coverage "
                     f"{_mean_sd(rul.get('coverage_p10_p90'), '{:.0f}', fraction_pct=True)}{target_text}. "
                     f"Failure mode correctly diagnosed for "
                     f"{_mean_sd(dx.get('event_accuracy'), '{:.0f}', fraction_pct=True)} of predictable failures.")
    if items:
        blocks.append(Block("ul", items=items))
    if (reports_dir / "ROBUSTNESS.md").exists():
        blocks.append(Block("link", "Per-fleet tables", href="ROBUSTNESS.md"))
    return blocks


def ai4i_blocks(a: dict) -> list[Block]:
    """Summary of ``artifacts/ai4i_metrics.json`` (layout documented in ``ai4i.py``)."""
    ds, res, proto = a["dataset"], a["results"], a.get("protocol", {})
    sets, models = a.get("feature_sets", {}), a.get("models", {})
    blocks = [Block("p", (
        f"{ds['name']} (Matzka, 2020; UCI id {ds.get('uci_id', 601)}, {ds.get('license', 'CC BY 4.0')}): "
        f"{ds['n_rows']:,} machining runs with {ds['n_failures']} failures ({_pct(ds.get('failure_rate'), 1, True)}). "
        f"Built independently of this project (its author describes it as synthetic data reflecting real industrial "
        f"predictive-maintenance data), so it tests whether the approach only works on our own simulator. "
        f"Protocol: {proto.get('n_repeats', '?')} x repeated stratified {proto.get('n_splits', '?')}-fold "
        f"cross-validation on identical folds for every model; the failure-type columns never reach a model."))]
    rows = []
    for set_key, per_model in res.items():
        for model_key, r in per_model.items():
            recall = _pct(r.get("recall_at_p90"), 0, True) + ("" if r.get("p90_reached", True) else " *")
            rows.append([sets.get(set_key, {}).get("label", set_key), models.get(model_key, model_key),
                         _mean_sd({"mean": r.get("roc_auc"), "std": r.get("roc_auc_std")}, "{:.3f}"),
                         _mean_sd({"mean": r.get("pr_auc"), "std": r.get("pr_auc_std")}, "{:.3f}"),
                         _mean_sd({"mean": r.get("f1_xfit"), "std": r.get("f1_xfit_std")}, "{:.3f}"), recall])
    blocks.append(Block("table", header=["Features", "Model", "ROC-AUC", "PR-AUC", "F1 (cross-fitted threshold)",
                                         "Recall at 90% precision"], rows=rows, align="llrrrr"))
    items = []
    best = a.get("best", {})
    if best.get("raw") in res.get("raw", {}) and best.get("physics") in res.get("physics", {}):
        raw_pr = res["raw"][best["raw"]]["pr_auc"]
        phys_pr = res["physics"][best["physics"]]["pr_auc"]
        items.append(f"Physics-informed features (temperature rise, mechanical power, overstrain) lift the best PR-AUC "
                     f"from {raw_pr:.3f} ({models.get(best['raw'], best['raw'])}, raw sensors) to {phys_pr:.3f} "
                     f"({models.get(best['physics'], best['physics'])}): encoding the failure physics matters more "
                     f"than model capacity.")
    op = a.get("operating_point", {}).get("physics")
    if op:
        items.append(f"At its best-F1 threshold the best physics-informed model catches {_pct(op['recall'], 0, True)} "
                     f"of failures at {_pct(op['precision'], 0, True)} precision "
                     f"({op['false_alarms_per_10k']:.1f} false alarms per 10,000 runs).")
    ceil = a.get("mechanism_ceiling")
    if ceil:
        items.append(f"Only {ceil['deterministic_failures']} of {ceil['failures']} failures "
                     f"({_pct(ceil['recall'], 0, True)}) follow a mechanism visible in a single snapshot; tool-wear "
                     f"and random failures are not predictable from the inputs, the same lesson as the sudden pump "
                     f"failures.")
    bcv = a.get("blocked_cv", {}).get("physics")
    if bcv:
        items.append(f"Blocked cross-validation (contiguous blocks of runs) gives PR-AUC {bcv['pr_auc']:.3f} versus "
                     f"{bcv['pr_auc_random_cv']:.3f} with random folds: the result does not rely on leakage between "
                     f"neighbouring rows.")
    if items:
        blocks.append(Block("ul", items=items))
    return blocks


_PREFERRED_COLS = ("roc_auc", "pr_auc", "caught", "failures", "detection_rate", "detection_rate_predictable",
                   "false_alarms", "false_alarms_per_asset_year", "median_lead_h", "savings_pct", "accuracy",
                   "precision", "recall", "f1", "mean", "std", "min", "max")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def summarise_json(obj: dict, max_tables: int = 4, max_rows: int = 15, max_cols: int = 8
                   ) -> tuple[list[tuple[str, Any]], list[tuple[str, pd.DataFrame]]]:
    """Return (top-level scalar facts, tables) found in a metrics JSON of unknown layout.

    A table is a dict whose values are dicts sharing numeric keys (e.g. method -> metrics), or a
    list of flat dicts (e.g. one entry per seed).
    """
    facts = [(k, v) for k, v in obj.items()
             if (_is_number(v) or isinstance(v, (str, bool))) and not (isinstance(v, str) and len(v) > 120)]
    tables: list[tuple[str, pd.DataFrame]] = []

    def order_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in _PREFERRED_COLS if c in df.columns] + [c for c in df.columns if c not in _PREFERRED_COLS]
        return df[cols[:max_cols]]

    def visit(node: Any, path: str, depth: int) -> None:
        if len(tables) >= max_tables or depth > 4:
            return
        if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
            df = pd.DataFrame([{k: v for k, v in x.items() if _is_number(v) or isinstance(v, str)} for x in node])
            if df.shape[1] >= 2:
                tables.append((path, order_cols(df).head(max_rows)))
            return
        if not isinstance(node, dict):
            return
        rows = {k: {kk: vv for kk, vv in v.items() if _is_number(vv)} for k, v in node.items() if isinstance(v, dict)}
        rows = {k: r for k, r in rows.items() if len(r) >= 2}
        if len(rows) >= 2:
            df = pd.DataFrame.from_dict(rows, orient="index")
            df = df.loc[:, df.notna().mean() >= 0.5]
            if df.shape[1] >= 2:
                tables.append((path, order_cols(df).head(max_rows)))
                for k, v in node.items():
                    if k not in rows:
                        visit(v, f"{path}.{k}" if path else k, depth + 1)
                return
        for k, v in node.items():
            visit(v, f"{path}.{k}" if path else k, depth + 1)

    visit(obj, "", 0)
    return facts, tables


def generic_blocks(obj: dict) -> list[Block]:
    """Fallback rendering of a JSON whose layout is not the documented one."""
    facts, tables = summarise_json(obj)
    shown = [(k, v) for k, v in facts if k not in ("generated_at", "version")][:10]
    blocks = [Block("ul", items=[f"{_humanise_key(k)}: {_auto(v)}" for k, v in shown])] if shown else []
    for path, df in tables:
        header = [_humanise_key(path)] + [str(c) for c in df.columns]
        rows = [[str(idx)] + [_auto(v.item() if hasattr(v, "item") else v) for v in r]
                for idx, r in zip(df.index, df.itertuples(index=False))]
        blocks.append(Block("table", header=header, rows=rows))
    return blocks or [Block("p", "No tabular results found.")]


def optional_sections(d: ReportData, reports_dir: Path) -> list[tuple[str, str, list[Block], str]]:
    """(id, title, blocks, source command) for each optional input that exists."""
    specs = (("robustness", "Robustness across independent fleets", d.robustness,
              lambda obj: robustness_blocks(obj, reports_dir), "python -m sentinel_pdm robustness"),
             ("ai4i", "Real-data benchmark: UCI AI4I 2020", d.ai4i, ai4i_blocks, "python -m sentinel_pdm ai4i"))
    out = []
    for sid, title, obj, render, command in specs:
        if obj is None:
            continue
        try:
            blocks = render(obj)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            d.warnings.append(f"{sid}: unexpected JSON layout ({exc!r}); showing a generic summary.")
            blocks = generic_blocks(obj)
        out.append((sid, title, blocks, command))
    return out


def _blocks_md(blocks: list[Block]) -> list[str]:
    out: list[str] = []
    for b in blocks:
        if b.kind == "p":
            out += [b.text, ""]
        elif b.kind == "ul":
            out += [f"- {item}" for item in b.items] + [""]
        elif b.kind == "table":
            out += [_md_table(b.header, b.rows, b.align), ""]
        elif b.kind == "link":
            out += [f"{b.text}: [{b.href}]({b.href}).", ""]
    return out


def _blocks_html(blocks: list[Block]) -> str:
    parts = []
    for b in blocks:
        if b.kind == "p":
            parts.append(f"<p>{html.escape(b.text)}</p>")
        elif b.kind == "ul":
            parts.append('<ul class="summary">' + "".join(f"<li>{html.escape(i)}</li>" for i in b.items) + "</ul>")
        elif b.kind == "table":
            parts.append(_html_table(b.header, b.rows, b.align))
        elif b.kind == "link":
            parts.append(f'<p class="note">{html.escape(b.text)}: <a href="{html.escape(b.href)}">'
                         f"{html.escape(b.href)}</a></p>")
    return "".join(parts)


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------
def headline_rows(m: dict) -> list[list[str]]:
    row, ev, econ = _test(m, "row", default={}), _test(m, "event", default={}), _test(m, "economics", default={})
    out = []
    for key, label in m.get("methods", {}).items():
        r, e, c = row.get(key, {}), ev.get(key, {}), econ.get(key, {})
        out.append([
            label, _num(r.get("roc_auc")), _num(r.get("pr_auc")),
            f"{e.get('caught', '-')}/{e.get('failures', '-')}",
            f"{e.get('caught_predictable', '-')}/{e.get('predictable_failures', '-')}",
            str(e.get("false_alarms", "-")), _days(e.get("median_lead_h")), _pct(c.get("savings_pct")),
        ])
    return out


HEADLINE_HEADER = ["Method", "ROC-AUC", "PR-AUC", "Caught / failures", "Predictable caught", "False alarms",
                   "Median lead (days)", "Saving vs run-to-failure"]


def economics_rows(m: dict) -> list[list[str]]:
    econ = _test(m, "economics", default={})
    out = []
    for key in ["reactive", *m.get("methods", {})]:
        e = econ.get(key)
        if not e:
            continue
        label = "Run to failure (reactive)" if key == "reactive" else m["methods"][key]
        out.append([label, str(e["unplanned_failures"]), str(e["planned_repairs"]), str(e["false_alarms"]),
                    _int(e["downtime_h"]), _money(e["total_cost"]), _money(e["cost_per_pump_year"]),
                    _pct(e["savings_pct"]), _pct(e["downtime_reduction_pct"])])
    return out


ECON_HEADER = ["Policy", "Breakdowns", "Planned repairs", "False alarms", "Downtime (h)", "Total cost",
               "Per pump-year", "Saving", "Less downtime"]


def _cv_rows(m: dict) -> list[list[str]]:
    row, ev = m.get("cv", {}).get("row", {}), m.get("cv", {}).get("event", {})
    out = []
    for key, label in m.get("methods", {}).items():
        r, e = row.get(key, {}), ev.get(key, {})
        if not e:
            continue
        out.append([label, _num(r.get("roc_auc")), _num(r.get("pr_auc")), f"{e.get('caught')}/{e.get('failures')}",
                    str(e.get("false_alarms")), _days(e.get("median_lead_h"))])
    return out


def _rul_rows(m: dict) -> list[list[str]]:
    rul = _test(m, "rul", default={})
    names = {"le_168h": "Last 7 days before failure", "le_720h": "Last 30 days before failure"}
    return [[names.get(k, k), _int(v.get("n_rows")), _num(v.get("mae_h"), 0),
             _pct(v.get("coverage_p10_p90"), 0, fraction=True), _num(v.get("median_interval_h"), 0)]
            for k, v in rul.items()]


def _confusion_md(dx: dict) -> str | None:
    conf = dx.get("confusion") or {}
    labels, matrix = conf.get("labels"), conf.get("matrix")
    if not labels or not matrix:
        return None
    header = ["Actual \\ predicted"] + [_SHORT_MODE.get(lab, lab) for lab in labels] + ["Correct"]
    rows = []
    for lab, r in zip(labels, matrix):
        n = sum(r)
        correct = r[labels.index(lab)]
        rows.append([f"**{_SHORT_MODE.get(lab, lab)}**", *[str(v) for v in r], f"{correct}/{n}" if n else "-"])
    return _md_table(header, rows, "l" + "r" * (len(labels) + 1))


def _importance_rows(imp: pd.DataFrame, n: int = 10) -> list[list[str]]:
    if imp.empty or "importance" not in imp:
        return []
    top = imp.sort_values("importance", ascending=False).head(n)
    return [[str(i), f"`{r.feature}`", describe_feature(r.feature), f"{r.importance:.4f} ± {r.std:.4f}"]
            for i, r in enumerate(top.itertuples(index=False), start=1)]


def _data_rows(d: ReportData) -> list[list[str]]:
    m = d.metrics
    data, cfg = m.get("data", {}), m.get("config", {}).get("sim", {})
    by_mode = data.get("failures_by_mode", {})
    modes = ", ".join(f"{k} {v}" for k, v in sorted(by_mode.items(), key=lambda kv: -kv[1]))
    # Smaller fleets (``train --quick``) fill fewer process units, so count them rather than assume 8.
    units = (f" in {d.assets['unit'].nunique()} process units"
             if not d.assets.empty and "unit" in d.assets else "")
    rows = [
        ["Fleet", f"{data.get('n_assets')} centrifugal pumps{units} "
                  f"({data.get('n_train_assets')} training, {data.get('n_test_assets')} held-out test)"],
        ["History", f"{_int(data.get('hours'))} h per pump (hourly historian averages from {cfg.get('start', '?')}); "
                    f"{_int(data.get('rows'))} pump-hours"],
        ["Failures", f"{data.get('failures')} ({modes}); {data.get('failures_train')} in the training fleet, "
                     f"{data.get('failures_test')} in the test fleet"],
        ["Preventive maintenance", f"{data.get('pm_events')} PM work orders"],
        ["Model features", f"{data.get('n_features')} causal features per pump-hour"],
    ]
    fo = d.outcomes
    if not fo.empty and {"method", "split", "mode"}.issubset(fo.columns):
        t = fo[(fo["method"] == "hybrid") & (fo["split"] == "test")]["mode"].value_counts()
        rows.insert(3, ["Test-fleet failures", ", ".join(f"{k} {v}" for k, v in t.items())])
    q = d.quality
    if not q.empty and {"running_pct", "missing_pct", "spikes_removed", "stuck_hours"}.issubset(q.columns):
        rows.append(["Data quality", f"pumps running {q['running_pct'].mean():.1f}% of hours; "
                                     f"{q['missing_pct'].mean():.2f}% of running sensor values missing; "
                                     f"{_int(q['spikes_removed'].sum())} spikes and {_int(q['stuck_hours'].sum())} "
                                     f"stuck-tag hours removed by causal cleaning"])
    return rows


def _threshold_text(method: str, value: Any) -> str:
    """Thresholds live on different scales: probability, multiple of a control limit, or alarm limit."""
    if not _finite(value):
        return "-"
    unit = _THRESHOLD_UNITS.get(method, "")
    return f"{value:.3f}" if not unit else f"{value:.2f} {unit}"


def _config_rows(m: dict) -> list[list[str]]:
    cfg = m.get("config", {})
    lab, al, co, tr = cfg.get("labels", {}), cfg.get("alerts", {}), cfg.get("costs", {}), cfg.get("train", {})
    thr = m.get("thresholds", {})
    methods = m.get("methods", {})
    rows = [
        ["Prediction horizon", f"failure within {lab.get('horizon_h')} h ({_days(lab.get('horizon_h'))} days)"],
        ["RUL target cap", f"{lab.get('rul_cap_h')} h; conformal P10-P90 target coverage "
                           f"{_pct(tr.get('rul_interval_coverage'), 0, fraction=True)}"],
        ["Alarm budget", f"<= {al.get('max_false_alarms_per_asset_year')} false alarms per pump-year "
                         f"(thresholds minimise cost within this budget, tuned out-of-fold on the training fleet)"],
        ["Alert logic", f"raise after {al.get('debounce_h')} h above threshold, clear after {al.get('clear_h')} h "
                        f"below, re-alert every {al.get('realert_h')} h while active"],
        ["Event matching", f"P-F interval; alerts later than F - {al.get('min_lead_h')} h count as late; "
                           f"{al.get('credit_window_h')} h fallback window when no P point is logged"],
        ["Costs (USD)", f"lost production {_usd(co.get('lost_production_per_h'))}/h; unplanned repair "
                        f"{_usd(co.get('unplanned_repair_cost'))} + {_int(co.get('unplanned_downtime_h'))} h down; "
                        f"planned repair {_usd(co.get('planned_repair_cost'))} + {_int(co.get('planned_downtime_h'))} h "
                        f"down; false-alarm inspection {_usd(co.get('false_alarm_cost'))}"],
        ["Gradient boosting", f"{tr.get('cv_folds')}-fold GroupKFold by pump; {tr.get('max_iter')} iterations, "
                              f"learning rate {tr.get('learning_rate')}, {tr.get('max_leaf_nodes')} leaves, "
                              f"min {tr.get('min_samples_leaf')} samples/leaf, {tr.get('max_features')} feature "
                              f"fraction, L2 {tr.get('l2_regularization')}; trained on one hour in {tr.get('train_stride_h')}"],
        ["Alert thresholds", "; ".join(f"{label} {_threshold_text(k, thr[k])}" for k, label in methods.items()
                                       if k in thr)],
    ]
    return rows


def build_results_markdown(d: ReportData) -> str:
    m = d.metrics
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    trained = str(m.get("generated_at", "?")).replace("T", " ")[:16]
    out = [
        "# Sentinel PdM: results",
        "",
        f"_Generated {now} by `python -m sentinel_pdm report` (sentinel-pdm v{__version__}) from the training run "
        f"of {trained} UTC (v{m.get('version', '?')}). Interactive figures: [model_report.html](model_report.html). "
        f"Method: [docs/METHODOLOGY.md](../docs/METHODOLOGY.md). Limitations: [docs/MODEL_CARD.md](../docs/MODEL_CARD.md)._",
        "",
        "## Summary",
        "",
        *[f"- {s}" for s in headline_sentences(d)],
        "",
        "## Held-out test fleet",
        "",
        f"All five methods scored the same {m.get('data', {}).get('n_test_assets', '?')} pumps that no model or "
        f"threshold was fitted on. ROC-AUC / PR-AUC rank pump-hours (label: failure within the horizon); the event "
        f"columns count alert raises matched to failures by P-F interval. 'Predictable' excludes sudden failures, "
        f"which have no precursor.",
        "",
        _md_table(HEADLINE_HEADER, headline_rows(m)),
        "",
    ]
    rp = _test(m, "row_predictable", default={})
    if rp:
        out += [
            "Ranking quality excluding the 7-day windows before sudden failures (which no sensor can rank):",
            "",
            _md_table(["Method", "ROC-AUC", "PR-AUC", "Brier"],
                      [[m["methods"].get(k, k), _num(v.get("roc_auc")), _num(v.get("pr_auc")), _num(v.get("brier"))]
                       for k, v in rp.items()]),
            "",
        ]
    cv = _cv_rows(m)
    if cv:
        out += [
            "### Cross-validation (training fleet, out-of-fold)",
            "",
            "Each training pump scored by models that never saw it; thresholds were tuned on these scores.",
            "",
            _md_table(["Method", "ROC-AUC", "PR-AUC", "Caught / failures", "False alarms", "Median lead (days)"], cv),
            "",
        ]
    out += [
        "## Economics",
        "",
        "Caught failures become planned repairs; missed ones remain breakdowns with secondary damage and a longer "
        "outage; every false alarm costs an inspection. Test fleet, "
        f"{_num(_test(m, 'asset_years'), 0)} pump-years.",
        "",
        _md_table(ECON_HEADER, economics_rows(m)),
        "",
    ]
    rul = _rul_rows(m)
    dx = _test(m, "diagnosis", default={}) or {}
    if rul or dx:
        out += ["## Remaining useful life and diagnosis", ""]
    if rul:
        out += [
            "RUL is conditional on a developing failure (P10 / P50 / P90 hours, Mondrian normalised conformal "
            "intervals). Evaluated on test-fleet hours before predictable failures.",
            "",
            _md_table(["Window", "Pump-hours", "MAE of P50 (h)", "P10-P90 coverage", "Median interval width (h)"], rul),
            "",
        ]
    if dx:
        cc = _confusion_correct(dx)
        line = f"Failure-mode diagnosis: {_pct(dx.get('event_accuracy'), 0, fraction=True)} of predictable test failures"
        line += f" ({cc[0]}/{cc[1]})" if cc else ""
        line += (f" correctly classified by averaging the mode probabilities over the 7 days before failure; "
                 f"{_pct(dx.get('row_accuracy'), 0, fraction=True)} of individual pre-failure hours.")
        out += [line, ""]
        cm = _confusion_md(dx)
        if cm:
            out += [cm, ""]
    imp = _importance_rows(d.importance)
    out += ["## Top-10 features", ""]
    if imp:
        out += [
            "Permutation importance of the gradient-boosting risk ensemble on test-fleet hours (drop in PR-AUC "
            "when the feature is shuffled; mean ± std over repeats).",
            "",
            _md_table(["#", "Feature", "Meaning", "PR-AUC drop"], imp, "rllr"),
            "",
        ]
    else:
        out += ["Not computed for this run (trained with `--no-importance`).", ""]
    for _, title, blocks, command in d.optional:
        out += [f"## {title}", "", *_blocks_md(blocks), f"_Source: `{command}`._", ""]
    out += [
        "## Data",
        "",
        _md_table(["Item", "Value"], _data_rows(d), "ll"),
        "",
        "Simulated with a physics-informed fleet model; see [METHODOLOGY](../docs/METHODOLOGY.md#2-the-simulated-fleet).",
        "",
        "## Key configuration",
        "",
        _md_table(["Setting", "Value"], _config_rows(m), "ll"),
        "",
    ]
    env, timings = m.get("environment", {}), m.get("timings_s", {})
    out += [
        "## Run details",
        "",
        f"- Environment: Python {env.get('python', '?')}, scikit-learn {env.get('sklearn', '?')}, "
        f"pandas {env.get('pandas', '?')}, numpy {env.get('numpy', '?')}",
        f"- Pipeline time: {_num(timings.get('total'), 0)} s on CPU "
        f"({', '.join(f'{k} {v:.0f} s' for k, v in timings.items() if k != 'total' and _finite(v))})",
        f"- Model version {m.get('version', '?')}, trained {trained} UTC; report generated {now}",
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------
def _style(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=TEXT, size=13),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)")
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def _thin(idx_len: int, n: int = _CURVE_POINTS) -> np.ndarray:
    return np.unique(np.linspace(0, idx_len - 1, min(n, idx_len)).round().astype(int))


def _known_rows(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty or "label" not in preds:
        return preds.iloc[0:0]
    return preds[(preds["running"] == 1) & (preds["label"] >= 0)]


def fig_roc_pr(d: ReportData) -> go.Figure | None:
    rows = _known_rows(d.preds)
    if rows.empty:
        return None
    y_all = rows["label"].to_numpy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve", "Precision-recall curve"),
                        horizontal_spacing=0.08)
    row_metrics = _test(d.metrics, "row", default={})
    for key, label in d.methods.items():
        col = d.score_columns.get(key)
        if col not in rows:
            continue
        s = rows[col].to_numpy(dtype=np.float64)
        ok = ~np.isnan(s)
        if ok.sum() == 0 or len(np.unique(y_all[ok])) < 2:
            continue
        y, s = y_all[ok], s[ok]
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        i, j = _thin(len(fpr)), _thin(len(rec))
        rm = row_metrics.get(key, {})
        color = METHOD_COLORS.get(key, TEXT)
        width = 3 if key == "hybrid" else 1.8
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode="lines", line=dict(color=color, width=width),
                                 name=f"{label}", legendgroup=key,
                                 hovertemplate=f"{label}<br>false-positive rate %{{x:.3f}}<br>"
                                               f"true-positive rate %{{y:.3f}}<extra>ROC-AUC {_num(rm.get('roc_auc'))}"
                                               f"</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rec[j], y=prec[j], mode="lines", line=dict(color=color, width=width),
                                 name=label, legendgroup=key, showlegend=False,
                                 hovertemplate=f"{label}<br>recall %{{x:.3f}}<br>precision %{{y:.3f}}"
                                               f"<extra>PR-AUC {_num(rm.get('pr_auc'))}</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    base = float(y_all.mean())
    fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             showlegend=False, hovertemplate=f"no-skill precision {base:.3f}<extra></extra>"),
                  row=1, col=2)
    fig.update_xaxes(title_text="False-positive rate (healthy hours flagged)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True-positive rate (pre-failure hours flagged)", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Precision", range=[0, 1.02], row=1, col=2)
    fig = _style(fig, height=490)
    fig.update_layout(margin=dict(t=90), legend=dict(y=1.14))
    return fig


def fig_threshold_tradeoff(d: ReportData) -> go.Figure | None:
    c = d.curve
    if c.empty or not {"method", "caught", "failures", "false_alarms_per_asset_year"}.issubset(c.columns):
        return None
    budget = d.metrics.get("config", {}).get("alerts", {}).get("max_false_alarms_per_asset_year")
    fig = go.Figure()
    for key, label in d.methods.items():
        g = c[c["method"] == key]
        if g.empty:
            continue
        g = g.sort_values(["false_alarms_per_asset_year", "caught"])
        recall = 100 * g["caught"] / g["failures"].clip(lower=1)
        color = METHOD_COLORS.get(key, TEXT)
        fig.add_trace(go.Scatter(
            x=g["false_alarms_per_asset_year"], y=recall, mode="lines+markers", name=label,
            line=dict(color=color, width=3 if key == "hybrid" else 1.8, shape="hv"), marker=dict(size=4),
            customdata=np.column_stack([g["caught"], g["failures"], g["false_alarms"], g["threshold"]]),
            hovertemplate=(f"{label}<br>threshold %{{customdata[3]:.3g}}<br>caught %{{customdata[0]:.0f}}/"
                           f"%{{customdata[1]:.0f}}<br>false alarms %{{customdata[2]:.0f}} "
                           f"(%{{x:.2f}} per pump-year)<extra></extra>")))
        thr = d.metrics.get("thresholds", {}).get(key)
        chosen = g[np.isclose(g["threshold"], thr)] if _finite(thr) else g.iloc[0:0]
        if len(chosen):
            ch = chosen.iloc[0]
            fig.add_trace(go.Scatter(
                x=[ch["false_alarms_per_asset_year"]], y=[100 * ch["caught"] / max(ch["failures"], 1)],
                mode="markers", marker=dict(size=14, color=color, symbol="star", line=dict(color=BG, width=1)),
                name=f"{label}: chosen threshold", showlegend=False,
                hovertemplate=f"{label}: chosen threshold<extra></extra>"))
    if _finite(budget):
        fig.add_vline(x=budget, line=dict(color=MUTED, dash="dash", width=1))
        fig.add_annotation(x=budget, y=1, yref="paper", text=f"alarm budget {budget}/pump-year", showarrow=False,
                           xanchor="left", yanchor="top", xshift=6, font=dict(color=MUTED, size=12))
    fig.update_xaxes(title_text="False alarms per pump-year (training fleet, out-of-fold)", range=[-0.05, 3])
    fig.update_yaxes(title_text="Failures caught (% of all, incl. sudden)", range=[0, 100])
    return _style(fig, height=420)


def fig_lead_times(d: ReportData) -> go.Figure | None:
    fo = d.outcomes
    if fo.empty or "lead_h" not in fo:
        return None
    fo = fo[(fo["split"] == "test") & fo["caught"].astype(bool) & fo["lead_h"].notna()]
    if fo.empty:
        return None
    fig = go.Figure()
    order = list(d.methods)
    for key in reversed(order):
        g = fo[fo["method"] == key]
        if g.empty:
            continue
        label = d.methods[key]
        fig.add_trace(go.Box(
            x=g["lead_h"] / 24, name=label, orientation="h", boxpoints="all", jitter=0.45, pointpos=0,
            marker=dict(color=METHOD_COLORS.get(key, TEXT), size=6, opacity=0.8),
            line=dict(color=METHOD_COLORS.get(key, TEXT)), fillcolor="rgba(255,255,255,0.03)",
            customdata=np.column_stack([g["asset_id"], g["mode"]]),
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>warning %{x:.1f} days<extra></extra>"))
    fig.update_xaxes(title_text="Warning before functional failure (days)", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _style(fig, height=380, legend=False)


def fig_policy_costs(d: ReportData) -> go.Figure | None:
    comp = _economic_components(d.metrics)
    if comp.empty:
        return None
    comp = comp.iloc[::-1]
    fig = go.Figure()
    parts = (("unplanned", "Unplanned breakdowns", "#ef4444"), ("planned", "Planned repairs", "#60a5fa"),
             ("false_alarms", "False-alarm inspections", "#e7ba52"))
    for col, name, color in parts:
        fig.add_trace(go.Bar(y=comp["policy"], x=comp[col] / 1e6, name=name, orientation="h", marker_color=color,
                             hovertemplate=f"{name}: $%{{x:.2f}}M<extra>%{{y}}</extra>"))
    labels = [f"{_money(t)}" + ("" if k == "reactive" else f"  ({_pct(s)} saved)")
              for k, t, s in zip(comp["key"], comp["total"], comp["savings_pct"])]
    fig.add_trace(go.Scatter(y=comp["policy"], x=comp["total"] / 1e6, mode="text", text=labels,
                             textposition="middle right", textfont=dict(color=TEXT, size=12), showlegend=False,
                             hoverinfo="skip"))
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(title_text="Maintenance cost over the test period (USD millions)",
                     range=[0, float(comp["total"].max()) / 1e6 * 1.35])
    return _style(fig, height=380)


def fig_calibration(d: ReportData) -> go.Figure | None:
    rows = _known_rows(d.preds)
    if rows.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             name="Perfect calibration", hoverinfo="skip"))
    row_metrics = _test(d.metrics, "row", default={})
    drawn = False
    for key in ("hybrid", "ml"):
        col = d.score_columns.get(key)
        if col not in rows:
            continue
        p = rows[col].to_numpy(dtype=np.float64)
        y = rows["label"].to_numpy(dtype=np.float64)
        ok = ~np.isnan(p)
        if not ok.any():
            continue
        b = np.digitize(p[ok], _CAL_EDGES[1:-1])
        g = pd.DataFrame({"b": b, "p": p[ok], "y": y[ok]}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"),
                                                                             n=("y", "size"))
        g = g[g["n"] >= _CAL_MIN_ROWS]
        if g.empty:
            continue
        drawn = True
        label = d.methods.get(key, key)
        brier = row_metrics.get(key, {}).get("brier")
        fig.add_trace(go.Scatter(
            x=g["p"], y=g["y"], mode="lines+markers", name=f"{label} (Brier {_num(brier, 4)})",
            line=dict(color=METHOD_COLORS[key], width=2.5 if key == "hybrid" else 1.8),
            marker=dict(size=np.clip(4 + 2.2 * np.log10(g["n"].to_numpy()), 6, 18), color=METHOD_COLORS[key]),
            customdata=g["n"],
            hovertemplate=(f"{label}<br>predicted %{{x:.3f}}<br>observed %{{y:.3f}}<br>"
                           f"%{{customdata:,}} pump-hours<extra></extra>")))
    if not drawn:
        return None
    fig.update_xaxes(title_text="Predicted P(failure within 7 days)", range=[0, 1])
    fig.update_yaxes(title_text="Observed failure frequency", range=[0, 1])
    return _style(fig, height=420)


def fig_rul(d: ReportData) -> go.Figure | None:
    p = d.preds
    need = {"ttf", "next_predictable", "rul_p10", "rul_p50", "rul_p90", "running"}
    if p.empty or not need.issubset(p.columns):
        return None
    cap = float(d.metrics.get("config", {}).get("labels", {}).get("rul_cap_h", 720))
    r = p[(p["running"] == 1) & p["next_predictable"].astype(bool) & (p["ttf"] <= cap) & p["rul_p50"].notna()]
    if r.empty:
        return None
    width = 24.0
    centre = (np.floor(r["ttf"].to_numpy() / width) * width + width / 2).clip(max=cap)
    g = (r.assign(bin=centre).groupby("bin")
         .agg(p10=("rul_p10", "median"), p50=("rul_p50", "median"), p90=("rul_p90", "median"),
              n=("rul_p50", "size")).reset_index())
    g = g[g["n"] >= 20]
    if g.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["bin"], y=g["p90"], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["bin"], y=g["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(245,158,11,0.18)", name="P10-P90 interval (median per day)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["bin"], y=g["p50"], mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                             marker=dict(size=5), name="Predicted P50 (median per day)",
                             customdata=np.column_stack([g["p10"], g["p90"], g["n"]]),
                             hovertemplate=("actual %{x:.0f} h to failure<br>P50 %{y:.0f} h<br>"
                                            "P10-P90 %{customdata[0]:.0f}-%{customdata[1]:.0f} h<br>"
                                            "%{customdata[2]:,} pump-hours<extra></extra>")))
    fig.add_trace(go.Scatter(x=[0, cap], y=[0, cap], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             name="Perfect prediction", hoverinfo="skip"))
    horizon = d.metrics.get("config", {}).get("labels", {}).get("horizon_h", 168)
    fig.add_vrect(x0=0, x1=horizon, fillcolor="rgba(255,255,255,0.04)", line_width=0)
    fig.add_annotation(x=horizon / 2, y=1, yref="paper", text="last 7 days", showarrow=False, yanchor="top",
                       font=dict(color=MUTED, size=12))
    fig.update_xaxes(title_text="Actual time to failure (h); time runs left to right", autorange="reversed")
    fig.update_yaxes(title_text="Predicted remaining useful life (h)", range=[0, cap * 1.02])
    return _style(fig, height=420)


def fig_confusion(d: ReportData) -> go.Figure | None:
    conf = (_test(d.metrics, "diagnosis", default={}) or {}).get("confusion") or {}
    labels, matrix = conf.get("labels"), conf.get("matrix")
    if not labels or not matrix:
        return None
    z = np.asarray(matrix, dtype=float)
    share = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    names = [_SHORT_MODE.get(lab, lab) for lab in labels]
    text = [[f"{int(v)}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=share, x=names, y=names, text=text, colorscale=[[0, "#232838"], [1, ACCENT]], zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3,
        hovertemplate="actual %{y}, diagnosed %{x}: %{text} failures<extra></extra>"))
    # Explicit annotations: dark digits on bright cells, light digits on dark ones.
    for i, actual in enumerate(names):
        for j, diagnosed in enumerate(names):
            fig.add_annotation(x=diagnosed, y=actual, text=text[i][j], showarrow=False,
                               font=dict(size=16, color=BG if share[i, j] > 0.5 else TEXT))
    fig.update_xaxes(title_text="Diagnosed mode (7-day average of mode probabilities)", side="bottom")
    fig.update_yaxes(title_text="Actual failure mode", autorange="reversed")
    return _style(fig, height=400, legend=False)


def fig_importance(d: ReportData, n: int = 15) -> go.Figure | None:
    imp = d.importance
    if imp.empty or "importance" not in imp:
        return None
    top = imp.sort_values("importance", ascending=False).head(n).iloc[::-1]
    labels = [describe_feature(f) for f in top["feature"]]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=labels, orientation="h", marker_color=ACCENT,
        error_x=dict(type="data", array=top["std"], color=MUTED, thickness=1),
        customdata=top["feature"], hovertemplate="%{customdata}<br>PR-AUC drop %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Drop in PR-AUC when the feature is shuffled")
    fig.update_yaxes(automargin=True)
    return _style(fig, height=max(360, 26 * len(top) + 80), legend=False)


def fig_example_timeline(d: ReportData) -> tuple[go.Figure, str] | None:
    """Hybrid risk, threshold, vib_hf, P-F interval, alert raises and the failure for one caught failure."""
    f = pick_example_failure(d)
    if f is None or d.preds.empty:
        return None
    asset, f_hour = str(f["asset_id"]), int(f["hour"])
    p_hour = float(f["p_hour"]) if _finite(f.get("p_hour")) else float(f_hour - f["lead_h"])
    lead = float(f["lead_h"])
    ev = d.events
    prev_end = 0
    if not ev.empty:
        before = ev[(ev["asset_id"] == asset) & (ev["hour"] < f_hour)]
        if len(before):
            prev_end = int(before["end_hour"].max())
    start = max(prev_end, f_hour - int(max(2.5 * lead, 720)))
    end = f_hour + 72
    risk_col = d.score_columns.get("hybrid", "risk")
    p = d.preds[(d.preds["asset_id"] == asset) & d.preds["hour"].between(start, end)].sort_values("hour")
    if p.empty or risk_col not in p:
        return None
    t0 = pd.Timestamp(d.metrics.get("config", {}).get("sim", {}).get("start", "2025-01-01"))

    def ts(h: float) -> pd.Timestamp:
        return t0 + pd.Timedelta(hours=float(h))

    clean_path = d.data_dir / "clean.parquet"
    vib = pd.DataFrame()
    if clean_path.exists():
        vib = pd.read_parquet(clean_path, columns=["asset_id", "hour", "vib_hf"],
                              filters=[("asset_id", "==", asset)])
        vib = vib[vib["hour"].between(start, end)].sort_values("hour")
    has_vib = not vib.empty and vib["vib_hf"].notna().any()

    fig = make_subplots(rows=2 if has_vib else 1, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.6, 0.4] if has_vib else None)
    x = [ts(h) for h in p["hour"]]
    thr = d.metrics.get("thresholds", {}).get("hybrid")
    fig.add_trace(go.Scatter(x=x, y=p[risk_col], mode="lines", line=dict(color=ACCENT, width=2),
                             name="Sentinel risk P(failure within 7 d)",
                             hovertemplate="%{x|%d %b %H:%M}<br>risk %{y:.3f}<extra></extra>"), row=1, col=1)
    if _finite(thr):
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[thr, thr], mode="lines",
                                 line=dict(color=MUTED, dash="dash", width=1.2), name=f"Alert threshold ({thr:.3f})",
                                 hoverinfo="skip"), row=1, col=1)
    al = d.alerts
    if not al.empty:
        raises = al[(al["method"] == "hybrid") & (al["asset_id"] == asset) & al["start_hour"].between(start, end)]
        if len(raises):
            rh = raises["start_hour"].to_numpy()
            ry = p.set_index("hour")[risk_col].reindex(rh).to_numpy(dtype=np.float64)
            ry = np.where(np.isnan(ry), thr if _finite(thr) else 0.5, ry)
            fig.add_trace(go.Scatter(
                x=[ts(h) for h in rh], y=ry, mode="markers", name="Alert raised (work request)",
                marker=dict(symbol="triangle-up", size=14, color="#fbbf24", line=dict(color=BG, width=1)),
                customdata=raises["outcome"],
                hovertemplate="alert raised %{x|%d %b %H:%M}<br>outcome: %{customdata}<extra></extra>"),
                row=1, col=1)
    if has_vib:
        fig.add_trace(go.Scatter(x=[ts(h) for h in vib["hour"]], y=vib["vib_hf"], mode="lines",
                                 line=dict(color="#cbd5e1", width=1.2),
                                 name="High-frequency bearing band vib_hf (g)",
                                 hovertemplate="%{x|%d %b %H:%M}<br>vib_hf %{y:.2f} g<extra></extra>"), row=2, col=1)
    fig.add_vrect(x0=ts(p_hour), x1=ts(f_hour), fillcolor="rgba(239,68,68,0.10)", line_width=0, row="all", col=1)
    fig.add_vline(x=ts(f_hour), line=dict(color="#ef4444", width=2), row="all", col=1)
    fig.add_annotation(x=ts(f_hour), y=1, yref="paper", text="F: functional failure", showarrow=False,
                       xanchor="right", yanchor="bottom", xshift=-4, font=dict(color="#ef4444", size=12))
    fig.add_annotation(x=ts(p_hour), y=1, yref="paper", text="P: defect detectable", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=4, font=dict(color="#fca5a5", size=12))
    fig.update_yaxes(title_text="Failure risk", range=[0, 1.02], row=1, col=1)
    if has_vib:
        fig.update_yaxes(title_text="vib_hf (g)", row=2, col=1)
    fig = _style(fig, height=520 if has_vib else 380)
    fig.update_layout(legend=dict(y=-0.08, yanchor="top"), margin=dict(t=40))
    mode = MODE_LABELS.get(str(f["mode"]), str(f["mode"])).lower()
    kind = "bearing failure" if str(f["mode"]) == "bearing" else "failure"
    caption = (f"{asset}, {mode} failure on {ts(f_hour):%d %b %Y}: the first actionable alert came "
               f"{lead:.0f} h ({lead / 24:.1f} days) before functional failure. The shaded band is the P-F interval "
               f"(from the first hour an inspection could confirm the defect to the failure); alerts inside it are "
               f"true positives. Chosen as the caught {kind} whose warning time is closest to the median of its "
               f"kind, so it is typical rather than the best case.")
    return fig, caption


# --------------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------------
_CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {BG}; color: {TEXT}; font: 15px/1.55 {FONT}; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; }}
header h1 {{ margin: 0; font-size: 2rem; letter-spacing: -0.01em; }}
header h1 span {{ color: {ACCENT}; }}
header p {{ margin: 6px 0 0; color: {MUTED}; }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 22px 0 8px; }}
nav a {{ color: {TEXT}; text-decoration: none; font-size: 0.85rem; padding: 4px 12px; border: 1px solid {BORDER};
        border-radius: 999px; background: {PANEL}; }}
nav a:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
.kpi {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; }}
.kpi .v {{ font-size: 1.7rem; font-weight: 700; color: {ACCENT}; font-variant-numeric: tabular-nums; }}
.kpi .l {{ font-size: 0.85rem; color: {MUTED}; }}
section {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 20px 22px;
          margin: 18px 0; }}
section h2 {{ margin: 0 0 6px; font-size: 1.25rem; }}
section h3 {{ margin: 18px 0 6px; font-size: 1.02rem; color: {TEXT}; }}
section p {{ margin: 6px 0 12px; color: #cbd5e1; max-width: 900px; }}
ul.summary {{ margin: 6px 0 0; padding-left: 20px; }}
ul.summary li {{ margin: 6px 0; }}
.note {{ font-size: 0.85rem; color: {MUTED}; }}
.tbl {{ overflow-x: auto; margin: 10px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid {BORDER}; text-align: left; white-space: nowrap; }}
th {{ white-space: normal; vertical-align: bottom; }}
td:not(.num) {{ white-space: normal; }}
td:first-child {{ min-width: 170px; }}
th {{ color: {MUTED}; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.num {{ text-align: right; }} .ctr {{ text-align: center; }}
tr.hl td {{ background: rgba(245,158,11,0.08); font-weight: 600; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
@media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
footer {{ color: {MUTED}; font-size: 0.82rem; margin-top: 28px; }}
code {{ background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
"""


class _FigureWriter:
    """Embeds Plotly figures; only the first one loads plotly.js (from the CDN)."""

    def __init__(self) -> None:
        self._js_done = False

    def __call__(self, fig: go.Figure | None, missing: str = "Not available for this run.") -> str:
        if fig is None:
            return f'<p class="note">{html.escape(missing)}</p>'
        out = fig.to_html(full_html=False, include_plotlyjs=False if self._js_done else "cdn",
                          config={"displaylogo": False, "responsive": True})
        self._js_done = True
        return out


def _section(sid: str, title: str, body: str, intro: str = "") -> str:
    p = f"<p>{intro}</p>" if intro else ""
    return f'<section id="{sid}"><h2>{html.escape(title)}</h2>{p}{body}</section>'


def _kpis(d: ReportData) -> str:
    m = d.metrics
    hy = _test(m, "event", "hybrid", default={}) or {}
    econ = _test(m, "economics", "hybrid", default={}) or {}
    rul = _test(m, "rul", "le_168h", default={}) or {}
    dx = _test(m, "diagnosis", default={}) or {}
    rule = _test(m, "event", "rule", default={}) or {}
    cards = [
        (f"{hy.get('caught', '-')}/{hy.get('failures', '-')}",
         f"failures caught ({hy.get('caught_predictable', '-')}/{hy.get('predictable_failures', '-')} predictable)"),
        (str(hy.get("false_alarms", "-")), f"false alarms in {_num(_test(m, 'asset_years'), 0)} pump-years"),
        (f"{_days(hy.get('median_lead_h'))} d", f"median warning (alarm limits: {_num(rule.get('median_lead_h'), 0)} h)"),
        (_pct(econ.get("savings_pct")), "maintenance-cost saving vs run-to-failure"),
        (f"{_num(rul.get('mae_h'), 0)} h", "RUL error (MAE) in the last 7 days"),
        (_pct(dx.get("event_accuracy"), 0, fraction=True), "failures correctly diagnosed"),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(lab)}</div></div>'
        for v, lab in cards) + "</div>"


def build_html(d: ReportData) -> str:
    m = d.metrics
    fw = _FigureWriter()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    trained = str(m.get("generated_at", "?")).replace("T", " ")[:16]
    n_test = m.get("data", {}).get("n_test_assets", "?")
    horizon = _horizon_text(m)
    pos_rate = _test(m, "row", "hybrid", "positive_rate")
    sections: list[tuple[str, str, str]] = []   # (id, nav label, html)

    summary = ('<ul class="summary">' + "".join(f"<li>{html.escape(s)}</li>" for s in headline_sentences(d))
               + "</ul>" + _html_table(HEADLINE_HEADER, headline_rows(m), highlight=0)
               + '<p class="note">Held-out test fleet. ROC-AUC / PR-AUC rank pump-hours; the event columns count '
                 'alert raises matched to failures by P-F interval. "Predictable" excludes sudden failures, which '
                 'have no precursor any sensor could see.</p>')
    sections.append(("summary", "Summary", _section("summary", "Summary", summary)))

    sections.append(("curves", "ROC + PR", _section(
        "curves", "Ranking quality: ROC and precision-recall", fw(fig_roc_pr(d)),
        f"Every running hour of the {n_test} test pumps, labelled 1 when a failure follows within {horizon}. "
        f"Failures are rare ({_pct(pos_rate, 1, fraction=True)} of hours), so the precision-recall curve is the "
        "more honest of the two: it shows how many flagged hours are real.")))

    sections.append(("tradeoff", "Alarm budget", _section(
        "tradeoff", "Detection versus nuisance alarms", fw(fig_threshold_tradeoff(d)),
        "Out-of-fold scores of the training fleet, swept over alert thresholds. Thresholds (stars) were chosen to "
        "minimise maintenance cost inside the alarm budget. The further left a curve reaches its plateau, the "
        "fewer nuisance alarms the method needs for the same detection rate.")))

    sections.append(("lead", "Lead time", _section(
        "lead", "Warning time before failure", fw(fig_lead_times(d)),
        "Time from the first actionable alert to functional failure, for every caught test-fleet failure. A week "
        "or more is enough to order parts and schedule the repair into a planned outage; alarm limits fire when "
        "the damage is already severe.")))

    econ_tbl = _html_table(ECON_HEADER, economics_rows(m), highlight=1)
    sections.append(("economics", "Economics", _section(
        "economics", "Maintenance policy cost", fw(fig_policy_costs(d)) + econ_tbl,
        "Caught failures become planned repairs; missed ones stay unplanned breakdowns with secondary damage and a "
        "longer outage; every false alarm costs an inspection. Cost assumptions are listed under Configuration.")))

    sections.append(("calibration", "Calibration", _section(
        "calibration", "Probability calibration", fw(fig_calibration(d)),
        "Predicted probability versus observed failure frequency on test-fleet hours. Isotonic calibration was "
        "fitted on out-of-fold training scores only, so this is a fair check: points on the diagonal mean a "
        f"'30% risk' really fails within {horizon} about 30% of the time.")))

    rul_rows = _rul_rows(m)
    rul_tbl = _html_table(["Window", "Pump-hours", "MAE of P50 (h)", "P10-P90 coverage", "Median interval (h)"],
                          rul_rows) if rul_rows else ""
    sections.append(("rul", "RUL", _section(
        "rul", "Remaining useful life: predicted versus actual", fw(fig_rul(d)) + rul_tbl,
        "Conditional RUL on test-fleet hours before predictable failures, aggregated per day of actual time to "
        "failure. Estimates tighten as failure approaches, which is when the planner needs them.")))

    dx = _test(m, "diagnosis", default={}) or {}
    cc = _confusion_correct(dx)
    dx_intro = (f"Most likely failure mode, from the mode probabilities averaged over the {horizon} before each "
                "predictable test-fleet failure.")
    if cc:
        dx_intro += f" {cc[0]} of {cc[1]} correct ({_pct(dx.get('event_accuracy'), 0, fraction=True)})."
    sections.append(("diagnosis", "Diagnosis", _section("diagnosis", "Failure-mode diagnosis",
                                                         fw(fig_confusion(d)), dx_intro)))

    top = (d.importance.sort_values("importance", ascending=False)["feature"].head(3).tolist()
           if not d.importance.empty and "importance" in d.importance else [])
    imp_intro = ("Permutation importance of the gradient-boosting ensemble on test-fleet hours: how much PR-AUC "
                 "drops when one feature is shuffled.")
    if top:
        imp_intro += " Leading features: " + "; ".join(describe_feature(f).lower() for f in top) + "."
    sections.append(("features", "Features", _section(
        "features", "What the risk model relies on",
        fw(fig_importance(d), "Permutation importance was not computed for this run."), html.escape(imp_intro))))

    ex = fig_example_timeline(d)
    sections.append(("example", "Example pump", _section(
        "example", "Example: one caught failure, hour by hour", fw(ex[0] if ex else None,
                                                                   "No caught failure in the test fleet."),
        html.escape(ex[1]) if ex else "")))

    nav_labels = {"robustness": "Robustness", "ai4i": "AI4I 2020"}
    for sid, title, blocks, command in d.optional:
        body = _blocks_html(blocks) + f'<p class="note">Source: <code>{html.escape(command)}</code></p>'
        sections.append((sid, nav_labels.get(sid, title), _section(sid, title, body)))

    cfg_body = (_html_table(["Item", "Value"], _data_rows(d), "ll") + "<h3>Configuration</h3>"
                + _html_table(["Setting", "Value"], _config_rows(m), "ll"))
    sections.append(("config", "Data + config", _section("config", "Data and configuration", cfg_body)))

    nav = "".join(f'<a href="#{sid}">{html.escape(label)}</a>' for sid, label, _ in sections)
    env = m.get("environment", {})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel PdM model report</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
<h1><span>Sentinel</span> PdM: model report</h1>
<p>Fleet predictive maintenance for centrifugal pumps. Held-out evaluation on {n_test} pumps the models never saw.</p>
<p class="note">Trained {html.escape(trained)} UTC (v{html.escape(str(m.get('version', '?')))}); report generated
{html.escape(now)} by <code>python -m sentinel_pdm report</code>.</p>
</header>
{_kpis(d)}
<nav>{nav}</nav>
{"".join(s for _, _, s in sections)}
<footer>Simulated data from a physics-informed fleet model; see docs/METHODOLOGY.md and docs/MODEL_CARD.md for the
method and its limitations. Decision support only: never a substitute for protective trips or safety systems.
Python {html.escape(str(env.get('python', '?')))}, scikit-learn {html.escape(str(env.get('sklearn', '?')))}.</footer>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------
def build_report(artifacts_dir: Path = ARTIFACTS_DIR, data_dir: Path = DATA_DIR,
                 reports_dir: Path = REPORTS_DIR) -> Path:
    """Write ``RESULTS.md`` and ``model_report.html`` into ``reports_dir``; return the HTML report path."""
    d = load_report_data(Path(artifacts_dir), Path(data_dir), Path(reports_dir))
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / RESULTS_MD
    html_path = reports_dir / REPORT_HTML
    md_path.write_text(build_results_markdown(d), encoding="utf-8", newline="\n")
    html_path.write_text(build_html(d), encoding="utf-8", newline="\n")
    for w in d.warnings:
        print(f"  note: {w}")
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    return html_path


__all__ = ["Block", "ReportData", "ai4i_blocks", "build_html", "build_report", "build_results_markdown",
           "describe_feature", "headline_sentences", "load_report_data", "pick_example_failure",
           "robustness_blocks", "robustness_sentence", "summarise_json"]
