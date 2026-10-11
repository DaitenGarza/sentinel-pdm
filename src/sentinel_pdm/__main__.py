"""Command-line interface.

    python -m sentinel_pdm train            simulate data, train + evaluate every model (default)
    python -m sentinel_pdm train --quick    smaller fleet, for a fast smoke run
    python -m sentinel_pdm summary          print the headline results of the last run
    python -m sentinel_pdm ai4i             real-data benchmark on the UCI AI4I 2020 dataset
    python -m sentinel_pdm robustness       repeat the experiment on several independent fleets
    python -m sentinel_pdm report           write reports/RESULTS.md and reports/model_report.html
    python -m sentinel_pdm app              launch the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import ARTIFACTS_DIR, PROJECT_ROOT, Config


def _cmd_train(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    cfg = Config()
    if args.quick:
        cfg.sim.n_assets, cfg.sim.hours, cfg.train.max_iter = 18, 4380, 120
    if args.assets:
        cfg.sim.n_assets = args.assets
    if args.hours:
        cfg.sim.hours = args.hours
    if args.seed is not None:
        cfg.sim.seed = args.seed
    run_pipeline(cfg, importance=not args.no_importance)
    return _cmd_summary(args)


def _cmd_summary(_: argparse.Namespace) -> int:
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        print("No results yet. Run: python -m sentinel_pdm train")
        return 1
    m = json.loads(path.read_text(encoding="utf-8"))
    ev, row, econ = m["test"]["event"], m["test"]["row"], m["test"]["economics"]
    d = m["data"]
    print(f"\nHeld-out test fleet: {d['n_test_assets']} pumps, {d['failures_test']} failures "
          f"(models never saw these pumps)\n")
    print(f"{'Method':34s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'Caught':>8s} {'Lead(h)':>8s} {'FalseAl':>8s} {'Savings':>9s}")
    for key, label in m["methods"].items():
        label = label.replace("²", "2")  # keep the console output ASCII-safe on Windows code pages
        e = ev[key]
        lead = e["median_lead_h"]
        print(f"{label:34s} {row[key]['roc_auc']:8.3f} {row[key]['pr_auc']:7.3f} "
              f"{e['caught']:>3d}/{e['failures']:<4d} {lead if lead is not None else float('nan'):8.0f} "
              f"{e['false_alarms']:8d} {econ[key]['savings_pct']:8.1f}%")
    rul = m["test"].get("rul", {}).get("le_168h")
    dx = m["test"].get("diagnosis", {})
    if rul:
        print(f"\nRUL (last 7 days before failure): MAE {rul['mae_h']:.0f} h, "
              f"P10-P90 coverage {rul['coverage_p10_p90']:.0%}")
    if dx.get("event_accuracy") is not None:
        print(f"Failure-mode diagnosis: {dx['event_accuracy']:.0%} of predictable failures correctly diagnosed")
    print(f"\nResults: {path}")
    return 0


def _cmd_ai4i(args: argparse.Namespace) -> int:
    from .ai4i import run_ai4i_benchmark

    run_ai4i_benchmark(download=not args.offline)
    return 0


def _cmd_robustness(args: argparse.Namespace) -> int:
    from .robustness import run_robustness

    run_robustness(seeds=list(range(args.first_seed, args.first_seed + args.seeds)), quick=args.quick)
    return 0


def _cmd_report(_: argparse.Namespace) -> int:
    from .report import build_report

    build_report()
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    app = PROJECT_ROOT / "app" / "Home.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.address", "localhost", "--browser.gatherUsageStats", "false"]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentinel-pdm", description="Fleet predictive maintenance for centrifugal pumps")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="simulate data, train and evaluate all models")
    t.add_argument("--quick", action="store_true", help="small fleet / short history for a fast run")
    t.add_argument("--assets", type=int, help="number of pumps to simulate")
    t.add_argument("--hours", type=int, help="hours of history per pump")
    t.add_argument("--seed", type=int, help="random seed")
    t.add_argument("--no-importance", action="store_true", help="skip permutation importance")
    sub.add_parser("summary", help="print headline results of the last training run")
    a4 = sub.add_parser("ai4i", help="benchmark on the real UCI AI4I 2020 predictive-maintenance dataset")
    a4.add_argument("--offline", action="store_true", help="do not download; use data/external/ai4i2020.csv")
    rb = sub.add_parser("robustness", help="re-run the whole experiment on several independent simulated fleets")
    rb.add_argument("--seeds", type=int, default=5, help="number of independent fleets")
    rb.add_argument("--first-seed", type=int, default=101)
    rb.add_argument("--quick", action="store_true", help="smaller fleets for a fast check")
    sub.add_parser("report", help="write reports/RESULTS.md and reports/model_report.html from the last run")
    a = sub.add_parser("app", help="launch the dashboard")
    a.add_argument("--port", type=int, default=8501)

    args = p.parse_args(argv)
    if args.cmd is None:
        args = p.parse_args(["train"])
    handlers = {"train": _cmd_train, "summary": _cmd_summary, "ai4i": _cmd_ai4i, "robustness": _cmd_robustness,
                "report": _cmd_report, "app": _cmd_app}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
