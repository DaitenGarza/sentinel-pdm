"""Robustness study: repeat the whole experiment on independent simulated fleets.

Why this exists
---------------
One held-out test fleet of 16 pumps produces only about 30 failures in a year. With event counts
that small, a single failure more or less moves a detection rate by 3-4 points and a single
nuisance alert moves the false-alarm rate by 0.06 per pump-year, so a method can "win" on one
fleet by luck. A reliability engineer would not trust an MTBF estimated from three failures for
the same reason.

Re-simulating the plant with a different random seed gives a statistically independent fleet:
new pumps, operating profiles, failure histories, instrument faults and a new train/test split.
The *entire* pipeline is re-run on each fleet (simulate, clean, engineer features, train with
grouped cross-validation, choose alert thresholds on the training fleet, evaluate on the held-out
pumps), so the spread across seeds shows how much of each headline number is signal and how
much is the luck of one draw. Pooled totals (all caught failures / all failures across fleets)
are reported next to the per-fleet mean +- standard deviation because they weight every failure
equally instead of every fleet.

Every run writes into a throw-away temporary directory, so the study never touches the main
``data/`` and ``artifacts/`` folders the dashboard reads. Results go to
``reports/robustness.json`` (every number) and ``reports/ROBUSTNESS.md`` (tables for the README).

    python -m sentinel_pdm robustness --seeds 5            # full-size fleets, ~2-3 min per seed
    python -m sentinel_pdm robustness --seeds 3 --quick    # smaller fleets for a fast check
"""

from __future__ import annotations

import json
import math
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .config import REPORTS_DIR, Config
from .pipeline import METHODS, run_pipeline

JSON_FILE = "robustness.json"
MARKDOWN_FILE = "ROBUSTNESS.md"

# Smaller fleets for ``--quick`` (same settings as ``train --quick`` apart from the fleet size).
QUICK_N_ASSETS = 24
QUICK_HOURS = 4380
QUICK_MAX_ITER = 120

# Per-method test-fleet metrics collected from every run, with their report label and format.
# Event level: what a maintenance organisation experiences (warnings, nuisance alerts, money).
EVENT_METRICS: dict[str, tuple[str, str]] = {
    "detection_rate_predictable": ("Predictable failures caught", "pct"),
    "false_alarms_per_asset_year": ("False alarms / pump-year", "{:.2f}"),
    "median_lead_h": ("Median lead time (h)", "{:.0f}"),
    "savings_pct": ("Cost saving vs run-to-failure", "pct_points"),
}
# Row level: how well the hourly score ranks "fails within 7 days" above "does not". The
# "predictable" variants leave out the 7 days before sudden failures, which no method can rank.
ROW_METRICS: dict[str, tuple[str, str]] = {
    "roc_auc": ("ROC-AUC", "{:.3f}"),
    "pr_auc": ("PR-AUC", "{:.3f}"),
    "roc_auc_predictable": ("ROC-AUC, predictable", "{:.3f}"),
    "pr_auc_predictable": ("PR-AUC, predictable", "{:.3f}"),
}
# Event counts kept per run for the pooled totals.
COUNT_METRICS = ("caught", "failures", "caught_predictable", "predictable_failures", "false_alarms")


# --------------------------------------------------------------------------------------------
# one run -> one flat record
# --------------------------------------------------------------------------------------------
def _num(x) -> float:
    """Float or NaN (run_pipeline returns NaN or None for undefined values such as a median lead
    time when nothing was caught)."""
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def seed_record(seed: int, metrics: dict, runtime_s: float = math.nan) -> dict:
    """Flatten the test-fleet results of one ``run_pipeline`` call into the numbers the study compares."""
    test = metrics["test"]
    methods = {}
    for key in METHODS:
        row, ev, econ = test["row"][key], test["event"][key], test["economics"][key]
        row_p = test.get("row_predictable", {}).get(key, {})
        methods[key] = {
            "roc_auc": _num(row.get("roc_auc")),
            "pr_auc": _num(row.get("pr_auc")),
            "roc_auc_predictable": _num(row_p.get("roc_auc")),
            "pr_auc_predictable": _num(row_p.get("pr_auc")),
            "detection_rate_predictable": _num(ev.get("detection_rate_predictable")),
            "caught": int(ev["caught"]),
            "failures": int(ev["failures"]),
            "caught_predictable": int(ev["caught_predictable"]),
            "predictable_failures": int(ev["predictable_failures"]),
            "false_alarms": int(ev["false_alarms"]),
            "false_alarms_per_asset_year": _num(ev.get("false_alarms_per_asset_year")),
            "median_lead_h": _num(ev.get("median_lead_h")),
            "savings_pct": _num(econ.get("savings_pct")),
            "threshold": _num(metrics.get("thresholds", {}).get(key)),
        }
    rul = test.get("rul", {}).get("le_168h", {})
    dx = test.get("diagnosis", {})
    data = metrics.get("data", {})
    return {
        "seed": int(seed),
        "runtime_s": _num(runtime_s),
        "data": {
            "n_assets": data.get("n_assets"),
            "hours": data.get("hours"),
            "n_test_assets": data.get("n_test_assets"),
            "failures": data.get("failures"),
            "failures_test": data.get("failures_test"),
            "asset_years_test": _num(test.get("asset_years")),
        },
        "methods": methods,
        "rul": {"mae_h": _num(rul.get("mae_h")), "coverage_p10_p90": _num(rul.get("coverage_p10_p90")),
                "n_rows": int(rul.get("n_rows", 0))},
        "diagnosis": {"event_accuracy": _num(dx.get("event_accuracy")), "n_events": int(dx.get("n_events", 0))},
    }


# --------------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------------
def summarize(values: Sequence[float]) -> dict:
    """Mean, sample standard deviation, min and max over runs, ignoring undefined (NaN) values.

    ``n`` says how many runs had the value defined (e.g. a median lead time needs at least one
    caught failure); the standard deviation needs two.
    """
    v = np.array([_num(x) for x in values], dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else math.nan,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def aggregate(records: Sequence[dict]) -> dict:
    """Per-fleet mean / std / min / max of every metric plus pooled event totals across fleets."""
    methods, pooled = {}, {}
    asset_years = float(np.nansum([r["data"]["asset_years_test"] for r in records]))
    for key in METHODS:
        per = [r["methods"][key] for r in records]
        methods[key] = {m: summarize([p[m] for p in per])
                        for m in (*EVENT_METRICS, *ROW_METRICS, "caught", "failures", "false_alarms")}
        tot = {c: int(sum(p[c] for p in per)) for c in COUNT_METRICS}
        tot["asset_years"] = asset_years
        tot["detection_rate"] = tot["caught"] / tot["failures"] if tot["failures"] else math.nan
        tot["detection_rate_predictable"] = (tot["caught_predictable"] / tot["predictable_failures"]
                                             if tot["predictable_failures"] else math.nan)
        tot["false_alarms_per_asset_year"] = tot["false_alarms"] / asset_years if asset_years else math.nan
        pooled[key] = tot
    return {
        "methods": methods,
        "pooled": pooled,
        "rul": {m: summarize([r["rul"][m] for r in records]) for m in ("mae_h", "coverage_p10_p90")},
        "diagnosis": {"event_accuracy": summarize([r["diagnosis"]["event_accuracy"] for r in records])},
        "failures_test": summarize([r["data"]["failures_test"] for r in records]),
    }


# --------------------------------------------------------------------------------------------
# report writing
# --------------------------------------------------------------------------------------------
def _json_safe(obj):
    """Recursively convert numpy scalars and NaN/inf to JSON-friendly values (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _fmt(value: float, style: str) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if style == "pct":
        return f"{100 * value:.1f} %"
    if style == "pct_points":
        return f"{value:.1f} %"
    return style.format(value)


def _mean_sd(stat: dict, style: str) -> str:
    """'mean ± sd' (or just the mean when only one run defines the value)."""
    mean, sd = stat["mean"], stat["std"]
    if not math.isfinite(mean):
        return "n/a"
    if not math.isfinite(sd):
        return _fmt(mean, style)
    if style == "pct":
        return f"{100 * mean:.1f} ± {100 * sd:.1f} %"
    if style == "pct_points":
        return f"{mean:.1f} ± {sd:.1f} %"
    return f"{style.format(mean)} ± {style.format(sd)}"


def _range(stat: dict, style: str) -> str:
    if stat["n"] == 0:
        return "n/a"
    return f"{_fmt(stat['min'], style)} to {_fmt(stat['max'], style)}"


def render_markdown(result: dict) -> str:
    """Human-readable report: why, setup, headline table, pooled totals, per-fleet tables."""
    records, agg, setup = result["runs"], result["aggregate"], result["setup"]
    n = len(records)
    seeds = ", ".join(str(r["seed"]) for r in records)
    pooled = agg["pooled"]
    h, rule = pooled["hybrid"], pooled["rule"]
    lines = [
        "# Robustness across independent fleets",
        "",
        f"_Generated {result['generated_at']} by `python -m sentinel_pdm robustness` "
        f"(Sentinel PdM v{result['version']})._",
        "",
        "## Why this study",
        "",
        "A single held-out test fleet produces only a few dozen failures, so one failure more or less moves a "
        "detection rate by several points and one nuisance alert moves the false-alarm rate noticeably. A "
        "method can look best on one fleet by luck. This study re-simulates the plant with different random "
        "seeds - new pumps, operating profiles, failure histories, instrument faults and train/test split - and "
        "re-runs the **entire** pipeline on each fleet: cleaning, features, grouped cross-validation training, "
        "cost-optimal threshold choice on the training pumps, and evaluation on pumps the models never saw. "
        "The spread across fleets is the honest error bar on every headline number.",
        "",
        "## Setup",
        "",
        f"* **{n} independent fleets** (seeds {seeds}), each {setup['n_assets']} pumps x {setup['hours']:,} h "
        f"of hourly data; {setup['n_test_assets']} pumps per fleet are held out for testing.",
        f"* Test failures per fleet: {_mean_sd(agg['failures_test'], '{:.1f}')} "
        f"(range {_range(agg['failures_test'], '{:.0f}')}); "
        f"{h['failures']} test failures in total, {h['predictable_failures']} of them predictable "
        "(the rest are *sudden* failures with no measurable precursor, which no method can catch).",
        f"* Gradient boosting: {setup['max_iter']} iterations, {setup['cv_folds']}-fold cross-validation by pump. "
        f"Alert thresholds are chosen per fleet on out-of-fold training scores under a budget of "
        f"{setup['max_false_alarms_per_asset_year']} false alarms per pump-year.",
        f"* Total compute: {result['runtime_s'] / 60:.1f} min.",
    ]
    if setup.get("quick"):
        lines.append("* **Quick mode** (smaller fleets, shorter training): a smoke check, not the reported study.")
    lines += [
        "",
        "## Headline: test-fleet results, mean ± standard deviation across fleets",
        "",
        "| Method | " + " | ".join(label for label, _ in EVENT_METRICS.values()) + " |",
        "|---|" + "---:|" * len(EVENT_METRICS),
    ]
    for key, label in METHODS.items():
        st = agg["methods"][key]
        lines.append(f"| {label} | " + " | ".join(_mean_sd(st[m], style) for m, (_, style) in EVENT_METRICS.items())
                     + " |")
    lines += [
        "",
        "A failure counts as caught only if the method warned at least 12 h ahead, inside the P-F interval "
        "(after the defect became detectable). A false alarm is an alert whose inspection would have found "
        "nothing. Lead time runs from the first useful alert to the failure. The cost saving compares planned "
        "repairs plus inspections against running every pump to failure.",
        "",
        "### Hour-by-hour ranking (mean ± sd)",
        "",
        "| Method | " + " | ".join(label for label, _ in ROW_METRICS.values()) + " |",
        "|---|" + "---:|" * len(ROW_METRICS),
    ]
    for key, label in METHODS.items():
        st = agg["methods"][key]
        lines.append(f"| {label} | " + " | ".join(_mean_sd(st[m], style) for m, (_, style) in ROW_METRICS.items())
                     + " |")
    lines += [
        "",
        "\"Predictable\" leaves out the 7 days before *sudden* failures, which have no precursor for any method to "
        "rank; the gap between the two columns grows with the number of sudden failures in a fleet.",
        "",
        "## Pooled over all fleets",
        "",
        "Pooling weights every failure equally (a fleet with more failures counts more), which is the fairest "
        "single number when per-fleet counts are small.",
        "",
        "| Method | Failures caught | Predictable caught | Detection (predictable) | False alarms | "
        "False alarms / pump-year | Worst fleet: detection (predictable) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in METHODS.items():
        p, st = pooled[key], agg["methods"][key]
        lines.append(
            f"| {label} | {p['caught']}/{p['failures']} | {p['caught_predictable']}/{p['predictable_failures']} | "
            f"{_fmt(p['detection_rate_predictable'], 'pct')} | {p['false_alarms']} | "
            f"{_fmt(p['false_alarms_per_asset_year'], '{:.2f}')} | "
            f"{_fmt(st['detection_rate_predictable']['min'], 'pct')} |"
        )
    rul, dx = agg["rul"], agg["diagnosis"]
    lines += [
        "",
        "## Remaining useful life and diagnosis (Sentinel models, test fleets)",
        "",
        "| Metric | Mean ± sd | Range |",
        "|---|---:|---:|",
        f"| RUL median-estimate MAE in the last 7 days before failure (h) | {_mean_sd(rul['mae_h'], '{:.0f}')} | "
        f"{_range(rul['mae_h'], '{:.0f}')} |",
        f"| RUL P10-P90 interval coverage, last 7 days (target 80 %) | {_mean_sd(rul['coverage_p10_p90'], 'pct')} | "
        f"{_range(rul['coverage_p10_p90'], 'pct')} |",
        f"| Failure mode correctly diagnosed (predictable failures) | {_mean_sd(dx['event_accuracy'], 'pct')} | "
        f"{_range(dx['event_accuracy'], 'pct')} |",
        "",
        "## Per-fleet results",
        "",
        "Each cell: failures caught / test failures, false alarms, median lead time.",
        "",
        "| Seed | Test failures (predictable) | " + " | ".join(METHODS.values()) + " |",
        "|---:|---:|" + "---|" * len(METHODS),
    ]
    for r in records:
        m = r["methods"]
        cells = []
        for key in METHODS:
            e = m[key]
            lead = f"{e['median_lead_h']:.0f} h" if math.isfinite(e["median_lead_h"]) else "no lead"
            cells.append(f"{e['caught']}/{e['failures']}, {e['false_alarms']} FA, {lead}")
        lines.append(f"| {r['seed']} | {m['hybrid']['failures']} ({m['hybrid']['predictable_failures']}) | "
                     + " | ".join(cells) + " |")
    lines += [
        "",
        "| Seed | Hybrid ROC-AUC | Hybrid PR-AUC | Hybrid cost saving | RUL MAE, last 7 d (h) | RUL P10-P90 coverage | "
        "Diagnosis accuracy | Runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        hm = r["methods"]["hybrid"]
        lines.append(
            f"| {r['seed']} | {_fmt(hm['roc_auc'], '{:.3f}')} | {_fmt(hm['pr_auc'], '{:.3f}')} | "
            f"{_fmt(hm['savings_pct'], 'pct_points')} | {_fmt(r['rul']['mae_h'], '{:.0f}')} | "
            f"{_fmt(r['rul']['coverage_p10_p90'], 'pct')} | {_fmt(r['diagnosis']['event_accuracy'], 'pct')} | "
            f"{_fmt(r['runtime_s'], '{:.0f}')} |"
        )
    lines += [
        "",
        "## Reading the result",
        "",
        f"* Across {n} fleets the Sentinel hybrid caught {h['caught_predictable']} of {h['predictable_failures']} "
        f"predictable failures ({_fmt(h['detection_rate_predictable'], 'pct')}) with {h['false_alarms']} false "
        f"alarms in {h['asset_years']:.0f} pump-years ({_fmt(h['false_alarms_per_asset_year'], '{:.2f}')} per "
        f"pump-year). Conventional alarm limits caught {rule['caught_predictable']} of "
        f"{rule['predictable_failures']} ({_fmt(rule['detection_rate_predictable'], 'pct')}) with a median lead "
        f"time of {_mean_sd(agg['methods']['rule']['median_lead_h'], '{:.0f}')} h, versus "
        f"{_mean_sd(agg['methods']['hybrid']['median_lead_h'], '{:.0f}')} h for the hybrid.",
        "* Methods whose ± ranges overlap should not be ranked against each other on these numbers alone; "
        "differences larger than the spread are robust to the luck of a single fleet.",
        "* Everything is simulated: the absolute numbers depend on the simulator's assumptions (P-F interval "
        "lengths, sensor noise, share of sudden failures). The comparison between methods on identical data is "
        "the meaningful part.",
        "",
    ]
    return "\n".join(lines)


def write_reports(result: dict, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Write ``robustness.json`` and ``ROBUSTNESS.md`` into ``reports_dir``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = reports_dir / JSON_FILE, reports_dir / MARKDOWN_FILE
    json_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------------------------
def study_config(seed: int, quick: bool = False, base: Config | None = None) -> Config:
    """Configuration of one run: ``base`` (default settings if None) with this seed; smaller fleet if quick."""
    cfg = Config.from_dict(base.to_dict()) if base is not None else Config()  # copy: never mutate base
    cfg.sim.seed = int(seed)
    if quick:
        cfg.sim.n_assets, cfg.sim.hours, cfg.train.max_iter = QUICK_N_ASSETS, QUICK_HOURS, QUICK_MAX_ITER
    return cfg


def run_robustness(
    seeds: Sequence[int],
    quick: bool = False,
    log: Callable[[str], None] = print,
    base_config: Config | None = None,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Run the full pipeline once per seed (each in a temporary directory) and write the reports.

    ``base_config`` overrides the default settings (the seed is always replaced); ``reports_dir``
    is where robustness.json / ROBUSTNESS.md are written. Returns the full result dictionary
    (``setup``, ``runs`` = one record per seed, ``aggregate``).
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("run_robustness needs at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds}")
    t_start = time.perf_counter()
    records = []
    for i, seed in enumerate(seeds, start=1):
        cfg = study_config(seed, quick, base_config)
        log(f"=== Fleet {i}/{len(seeds)}: seed {seed} ({cfg.sim.n_assets} pumps x {cfg.sim.hours} h) ===")
        t0 = time.perf_counter()
        # ignore_cleanup_errors: on Windows a virus scanner can briefly hold a just-written file.
        with tempfile.TemporaryDirectory(prefix=f"sentinel_seed{seed}_", ignore_cleanup_errors=True) as tmp:
            metrics = run_pipeline(cfg, data_dir=Path(tmp) / "data", artifacts_dir=Path(tmp) / "artifacts",
                                   log=lambda msg: log(f"    {msg}"), importance=False)
        rec = seed_record(seed, metrics, time.perf_counter() - t0)
        records.append(rec)
        hm, rm = rec["methods"]["hybrid"], rec["methods"]["rule"]
        log(f"  seed {seed}: hybrid caught {hm['caught']}/{hm['failures']} with {hm['false_alarms']} false alarms; "
            f"alarm limits caught {rm['caught']}/{rm['failures']} ({rec['runtime_s']:.0f} s)")

    first = study_config(seeds[0], quick, base_config)
    shared_config = first.to_dict()   # identical for every run apart from the seed
    shared_config["sim"].pop("seed")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": __version__,
        "setup": {
            "seeds": seeds,
            "quick": bool(quick),
            "n_assets": first.sim.n_assets,
            "hours": first.sim.hours,
            "n_test_assets": records[0]["data"]["n_test_assets"],
            "max_iter": first.train.max_iter,
            "cv_folds": first.train.cv_folds,
            "max_false_alarms_per_asset_year": first.alerts.max_false_alarms_per_asset_year,
            "config": shared_config,
        },
        "methods": METHODS,
        "runs": records,
        "aggregate": aggregate(records),
        "runtime_s": time.perf_counter() - t_start,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    json_path, md_path = write_reports(result, reports_dir)
    agg = result["aggregate"]
    log(f"\nRobustness over {len(seeds)} fleets (mean +/- sd of the test-fleet results):")
    for key in METHODS:
        st, p = agg["methods"][key], agg["pooled"][key]
        log(f"  {key:8s} detection (predictable) {_mean_sd(st['detection_rate_predictable'], 'pct'):>18s} | "
            f"false alarms/pump-yr {_mean_sd(st['false_alarms_per_asset_year'], '{:.2f}'):>12s} | "
            f"median lead {_mean_sd(st['median_lead_h'], '{:.0f}'):>10s} h | "
            f"pooled {p['caught']}/{p['failures']} caught, {p['false_alarms']} FA".replace("±", "+/-"))
    log(f"Reports: {json_path} and {md_path}")
    return result
