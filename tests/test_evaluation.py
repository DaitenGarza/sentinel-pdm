"""Evaluation as a maintenance organisation experiences it: alert raises, P-F matching, cost, thresholds.

Every case is hand-made with hours chosen so the expected outcome follows directly from the
rules in ``evaluation.py`` (debounce, clear time, re-alert period, P-F credit window, minimum
lead time, false-alarm budget). These numbers end up in the headline results, so the rules
are pinned down exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel_pdm.config import AlertConfig, CostConfig
from sentinel_pdm.evaluation import (
    alert_episodes, choose_threshold, event_metrics, match_events, policy_cost, row_metrics,
)

ALERTS = AlertConfig(debounce_h=3, clear_h=24, realert_h=168, credit_window_h=336, min_lead_h=12)
COSTS = CostConfig()   # unplanned 42,000 + 48 h x 2,500 = 162,000; planned 14,000 + 8 h x 2,500 = 34,000
UNPLANNED, PLANNED, INSPECTION = 162_000.0, 34_000.0, 1_800.0


def _scores(above_hours, n_hours: int = 600, asset_id: str = "P-1", high: float = 0.9) -> pd.DataFrame:
    """Score series that is ``high`` at the given hours and 0.1 elsewhere."""
    s = np.full(n_hours, 0.1)
    s[list(above_hours)] = high
    return pd.DataFrame({"asset_id": asset_id, "hour": np.arange(n_hours), "score": s})


def _raises(frame: pd.DataFrame, threshold: float = 0.5, cfg: AlertConfig = ALERTS) -> list[tuple[int, int]]:
    ep = alert_episodes(frame, "score", threshold, cfg)
    return list(zip(ep["start_hour"].tolist(), ep["end_hour"].tolist()))


# --------------------------------------------------------------------------------------------
# alert raising
# --------------------------------------------------------------------------------------------
def test_no_alert_when_score_stays_low():
    ep = alert_episodes(_scores([]), "score", 0.5, ALERTS)
    assert ep.empty and list(ep.columns) == ["asset_id", "start_hour", "end_hour"]


def test_short_blip_is_debounced():
    assert _raises(_scores(range(100, 102))) == []          # 2 h above threshold < 3 h debounce


def test_alert_raised_after_debounce():
    assert _raises(_scores(range(100, 120))) == [(102, 119)]   # third consecutive hour above


def test_threshold_is_inclusive():
    assert _raises(_scores(range(100, 110), high=0.5)) == [(102, 109)]


def test_short_dip_does_not_clear_the_alert():
    # 23 h below threshold (< clear_h = 24): still the same alert, no new work request
    assert _raises(_scores([*range(100, 120), *range(143, 160)])) == [(102, 159)]


def test_alert_clears_after_clear_time_and_raises_again():
    # 24 h below threshold: the first alert is closed, the second excursion is a new raise
    assert _raises(_scores([*range(100, 120), *range(144, 160)])) == [(102, 119), (146, 159)]


def test_still_active_alert_re_notifies_every_realert_period():
    assert _raises(_scores(range(0, 400))) == [(2, 169), (170, 337), (338, 399)]


def test_nan_scores_count_as_below_threshold():
    frame = _scores(range(100, 130))
    frame.loc[110:139, "score"] = np.nan                     # pump stopped for 30 h
    assert _raises(frame) == [(102, 109)]


def test_episodes_are_computed_per_pump():
    frame = pd.concat([_scores(range(100, 110), asset_id="P-1"), _scores(range(300, 310), asset_id="P-2")],
                      ignore_index=True)
    ep = alert_episodes(frame, "score", 0.5, ALERTS)
    assert ep.values.tolist() == [["P-1", 102, 109], ["P-2", 302, 309]]


# --------------------------------------------------------------------------------------------
# P-F matching
# --------------------------------------------------------------------------------------------
RECORD_END = 5000


@pytest.fixture
def log() -> pd.DataFrame:
    rows = [
        # pump A: bearing failure at 1000 detectable from 800; PM at 3000 found a defect detectable from 2900
        ("A", "failure", 1000, "bearing", 800.0, True),
        ("A", "pm", 3000, "bearing", 2900.0, True),
        # pump B: failure without a recorded P point (real CMMS data) -> P = F - credit window = 1664
        ("B", "failure", 2000, "cavitation", np.nan, True),
        # pump C: sudden failure, P = F: nothing can be caught
        ("C", "failure", 1500, "sudden", 1500.0, False),
    ]
    return pd.DataFrame(rows, columns=["asset_id", "event_type", "hour", "mode", "p_hour", "predictable"])


@pytest.fixture
def raises() -> pd.DataFrame:
    rows = [("A", 700), ("A", 850), ("A", 995), ("A", 2950), ("A", 4800), ("B", 1600), ("B", 1700), ("C", 1490)]
    return pd.DataFrame([(a, s, s + 10) for a, s in rows], columns=["asset_id", "start_hour", "end_hour"])


def _outcomes(ep: pd.DataFrame) -> dict[tuple[str, int], str]:
    return {(a, s): o for a, s, o in zip(ep["asset_id"], ep["start_hour"], ep["outcome"])}


def test_alert_outcomes(log, raises):
    _, ep = match_events(raises, log, ALERTS, RECORD_END)
    assert _outcomes(ep) == {
        ("A", 700): "false_alarm",      # before the defect was detectable: inspection finds nothing
        ("A", 850): "tp",               # inside [P, F - 12 h]
        ("A", 995): "late",             # 5 h before failure: too late to plan
        ("A", 2950): "pm_confirmed",    # defect confirmed (and fixed) by the PM work order
        ("A", 4800): "unresolved",      # last 14 days of the record: outcome not known yet
        ("B", 1600): "false_alarm",     # before the fallback P point (2000 - 336 = 1664)
        ("B", 1700): "tp",              # after the fallback P point
        ("C", 1490): "late",            # sudden failure: P = F, so never a usable warning
    }


def test_failure_outcomes(log, raises):
    failures, _ = match_events(raises, log, ALERTS, RECORD_END)
    f = failures.set_index("asset_id")
    assert f.loc["A", "caught"] and f.loc["A", "lead_h"] == 150 and not f.loc["A", "late_only"]
    assert f.loc["B", "caught"] and f.loc["B", "lead_h"] == 300
    assert not f.loc["C", "caught"] and np.isnan(f.loc["C", "lead_h"]) and f.loc["C", "late_only"]
    assert len(failures) == 3        # PM work orders are not failures


def test_lead_time_uses_the_first_useful_alert(log):
    eps = pd.DataFrame({"asset_id": ["A", "A"], "start_hour": [900, 820], "end_hour": [910, 830]})
    failures, ep = match_events(eps, log, ALERTS, RECORD_END)
    assert failures.set_index("asset_id").loc["A", "lead_h"] == 180
    assert (ep["outcome"] == "tp").all()


def test_fallback_p_point_when_the_log_has_no_p_hour(log, raises):
    _, ep = match_events(raises, log.drop(columns="p_hour"), ALERTS, RECORD_END)
    out = _outcomes(ep)
    assert out[("A", 700)] == "tp"            # 700 >= 1000 - 336
    assert out[("A", 2950)] == "pm_confirmed"
    assert out[("B", 1600)] == "false_alarm" and out[("B", 1700)] == "tp"


def test_event_metrics(log, raises):
    failures, ep = match_events(raises, log, ALERTS, RECORD_END)
    m = event_metrics(failures, ep, asset_years=2.0)
    assert m["failures"] == 3 and m["predictable_failures"] == 2
    assert m["caught"] == 2 and m["caught_predictable"] == 2
    assert m["detection_rate"] == pytest.approx(2 / 3) and m["detection_rate_predictable"] == 1.0
    assert m["late_only"] == 1
    assert m["median_lead_h"] == 225 and m["mean_lead_h"] == 225
    assert m["alerts"] == 8 and m["true_alerts"] == 2 and m["false_alarms"] == 2 and m["pm_confirmed_alerts"] == 1
    assert m["false_alarms_per_asset_year"] == 1.0
    assert m["alert_precision"] == 0.5


def test_event_metrics_without_failures_or_alerts():
    failures = pd.DataFrame({"predictable": pd.Series([], dtype=bool), "caught": pd.Series([], dtype=bool),
                             "lead_h": pd.Series([], dtype=float), "late_only": pd.Series([], dtype=bool)})
    episodes = pd.DataFrame({"outcome": pd.Series([], dtype=object)})
    m = event_metrics(failures, episodes, asset_years=1.0)
    assert m["failures"] == 0 and m["caught"] == 0 and m["false_alarms"] == 0
    assert np.isnan(m["detection_rate"]) and np.isnan(m["median_lead_h"]) and np.isnan(m["alert_precision"])


# --------------------------------------------------------------------------------------------
# economics
# --------------------------------------------------------------------------------------------
def test_policy_cost_arithmetic():
    c = policy_cost(n_failures=10, n_caught=7, n_false_alarms=3, costs=COSTS)
    assert c["unplanned_cost_each"] == UNPLANNED and c["planned_cost_each"] == PLANNED
    assert c["total_cost"] == 3 * UNPLANNED + 7 * PLANNED + 3 * INSPECTION == 729_400
    assert c["unplanned_failures"] == 3 and c["planned_repairs"] == 7 and c["false_alarms"] == 3
    assert c["downtime_h"] == 3 * 48 + 7 * 8


def test_run_to_failure_and_custom_costs():
    assert policy_cost(5, 0, 0, COSTS)["total_cost"] == 5 * UNPLANNED
    custom = CostConfig(lost_production_per_h=100, unplanned_repair_cost=1000, unplanned_downtime_h=10,
                        planned_repair_cost=200, planned_downtime_h=2, false_alarm_cost=50)
    assert policy_cost(4, 3, 2, custom)["total_cost"] == 1 * 2000 + 3 * 400 + 2 * 50


# --------------------------------------------------------------------------------------------
# threshold choice
# --------------------------------------------------------------------------------------------
@pytest.fixture
def threshold_case():
    """One pump, 5000 h. Failure 1 at 2000 (score 0.6 before it), failure 2 at 4000 (score 0.9 before it),
    nuisance excursions to 0.95 and 0.55 that are not failures.

    Threshold 0.5 catches both failures with 2 false alarms (cheapest); 0.8 catches one with 1 false alarm.
    """
    s = np.full(5000, 0.1)
    s[1850:2000] = 0.6
    s[3850:4000] = 0.9
    s[500:520] = 0.95
    s[1000:1020] = 0.55
    frame = pd.DataFrame({"asset_id": "P-1", "hour": np.arange(5000), "score": s})
    events = pd.DataFrame({"asset_id": "P-1", "event_type": "failure", "hour": [2000, 4000],
                           "mode": "bearing", "p_hour": [1800.0, 3800.0], "predictable": True})
    asset_years = 5000 / 8760
    return frame, events, asset_years


def _choose(case, budget, candidates=(0.5, 0.8)):
    frame, events, years = case
    cfg = AlertConfig(max_false_alarms_per_asset_year=budget)
    return choose_threshold(frame, "score", events, cfg, COSTS, 5000, years, candidates=np.array(candidates))


def test_cost_optimal_threshold_when_budget_allows(threshold_case):
    thr, curve = _choose(threshold_case, budget=5.0)
    assert thr == 0.5
    c = curve.set_index("threshold")
    assert c.loc[0.5, "caught"] == 2 and c.loc[0.5, "false_alarms"] == 2
    assert c.loc[0.5, "total_cost"] == 2 * PLANNED + 2 * INSPECTION
    assert c.loc[0.8, "caught"] == 1 and c.loc[0.8, "false_alarms"] == 1
    assert c.loc[0.8, "total_cost"] == UNPLANNED + PLANNED + INSPECTION
    assert curve["within_budget"].all()
    assert list(curve.columns) == ["threshold", "caught", "failures", "false_alarms", "false_alarms_per_asset_year",
                                   "total_cost", "within_budget"]


def test_false_alarm_budget_overrides_cost(threshold_case):
    # 0.5 gives 3.5 false alarms per pump-year, 0.8 gives 1.75: a budget of 2 forbids the cheaper one
    thr, curve = _choose(threshold_case, budget=2.0)
    assert thr == 0.8
    assert curve.set_index("threshold")["within_budget"].to_dict() == {0.5: False, 0.8: True}


def test_fallback_to_fewest_false_alarms_when_nothing_meets_the_budget(threshold_case):
    thr, curve = _choose(threshold_case, budget=0.5)
    assert not curve["within_budget"].any()
    assert thr == 0.8      # fewest false alarms wins, even though 0.5 would be cheaper


def test_default_candidates_come_from_score_quantiles(threshold_case):
    frame, events, years = threshold_case
    thr, curve = choose_threshold(frame, "score", events, ALERTS, COSTS, 5000, years)
    assert thr in set(curve["threshold"])
    assert curve["threshold"].is_monotonic_increasing and curve["threshold"].is_unique
    # candidates are upper quantiles (80th-99.95th percentile) of the scores themselves
    assert len(curve) > 1
    assert curve["threshold"].between(np.quantile(frame["score"], 0.8), frame["score"].max()).all()


# --------------------------------------------------------------------------------------------
# row level
# --------------------------------------------------------------------------------------------
def test_row_metrics_ignore_nan_scores():
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9, np.nan, np.nan])
    m = row_metrics(y, score, probabilistic=True)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["n_rows"] == 4 and m["positive_rate"] == 0.5
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]))
    assert "brier" not in row_metrics(y, score)
