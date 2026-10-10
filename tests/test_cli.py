"""Command-line interface: help, the results summary, and how ``train`` options reach the pipeline.

Nothing here trains into the project's folders: ``summary`` reads a metrics.json copied from the
shared temporary pipeline run, and ``train`` is exercised with the pipeline replaced by a stub
that records the configuration it was given.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

import sentinel_pdm.__main__ as cli
import sentinel_pdm.pipeline
from sentinel_pdm.config import PROJECT_ROOT, Config

SUBCOMMANDS = ("train", "summary", "ai4i", "robustness", "report", "app")
DEFAULT = Config()


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """An empty artifacts folder that the CLI uses instead of the project's."""
    monkeypatch.setattr(cli, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def with_metrics(artifacts, pipeline_run):
    """The artifacts folder holding the metrics.json of the shared pipeline run."""
    shutil.copy(pipeline_run.artifacts_dir / "metrics.json", artifacts / "metrics.json")
    return json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# help
# --------------------------------------------------------------------------------------------
def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert all(cmd in out for cmd in SUBCOMMANDS)


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help_exits_cleanly(cmd, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([cmd, "--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_module_entry_point():
    """``python -m sentinel_pdm --help`` works in a fresh interpreter (the ``__main__`` guard)."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PROJECT_ROOT / "src"),
                                                                      os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-m", "sentinel_pdm", "--help"], capture_output=True, text=True,
                            env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "summary" in result.stdout


# --------------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------------
def test_summary_without_results_explains_what_to_run(artifacts, capsys):
    assert cli.main(["summary"]) == 1
    assert "python -m sentinel_pdm train" in capsys.readouterr().out


def test_summary_prints_the_headline_results(with_metrics, artifacts, capsys):
    m = with_metrics
    assert cli.main(["summary"]) == 0
    out = capsys.readouterr().out
    path = str(artifacts / "metrics.json")
    assert path in out
    out.replace(path, "").encode("ascii")    # the table stays printable on any Windows console code page
    assert f"Held-out test fleet: {m['data']['n_test_assets']} pumps, {m['data']['failures_test']} failures" in out
    lines = out.splitlines()
    for key, label in m["methods"].items():
        row = next((ln for ln in lines if ln.startswith(label.replace("²", "2"))), None)
        assert row is not None, f"no summary row for {key}"
        ev = m["test"]["event"][key]
        assert f"{m['test']['row'][key]['roc_auc']:.3f}" in row
        assert f"{ev['caught']}/{ev['failures']}" in row
    if m["test"]["rul"].get("le_168h"):
        assert "RUL (last 7 days before failure)" in out


def test_summary_tolerates_missing_optional_results(with_metrics, artifacts, capsys):
    """A run where nothing was caught has no lead times, no RUL rows and no diagnosed events."""
    m = with_metrics
    for ev in m["test"]["event"].values():
        ev["median_lead_h"] = None
    m["test"]["rul"], m["test"]["diagnosis"] = {}, {"row_accuracy": None}
    (artifacts / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    assert cli.main(["summary"]) == 0
    out = capsys.readouterr().out
    assert "nan" in out and "RUL" not in out and "diagnosis" not in out


# --------------------------------------------------------------------------------------------
# train: options reach the pipeline
# --------------------------------------------------------------------------------------------
@pytest.fixture
def fake_pipeline(monkeypatch, with_metrics):
    """Replace the pipeline with a stub that records its arguments (and never writes anything)."""
    calls: list[tuple[Config, bool]] = []

    def run_pipeline(cfg, importance=True, **_):
        calls.append((cfg, importance))
        return {}

    monkeypatch.setattr(sentinel_pdm.pipeline, "run_pipeline", run_pipeline)
    # Also patch the CLI module itself, so moving its import to module level can never make this
    # test train a full fleet into the project's data/ and artifacts/ folders.
    monkeypatch.setattr(cli, "run_pipeline", run_pipeline, raising=False)
    return calls


@pytest.mark.parametrize("argv, changed", [
    ([], {}),                                                       # no subcommand means train
    (["train"], {}),
    (["train", "--quick", "--seed", "5", "--no-importance"],
     {"n_assets": 18, "hours": 4380, "max_iter": 120, "seed": 5, "importance": False}),
    (["train", "--assets", "20", "--hours", "1000"], {"n_assets": 20, "hours": 1000}),
], ids=["no-args", "defaults", "quick", "size"])
def test_train_options_reach_the_pipeline(fake_pipeline, capsys, argv, changed):
    expected = {"n_assets": DEFAULT.sim.n_assets, "hours": DEFAULT.sim.hours, "max_iter": DEFAULT.train.max_iter,
                "seed": DEFAULT.sim.seed, "importance": True, **changed}
    assert cli.main(argv) == 0
    assert len(fake_pipeline) == 1
    cfg, importance = fake_pipeline[0]
    assert {"n_assets": cfg.sim.n_assets, "hours": cfg.sim.hours, "max_iter": cfg.train.max_iter,
            "seed": cfg.sim.seed, "importance": importance} == expected
    assert "Held-out test fleet" in capsys.readouterr().out          # train ends with the summary
