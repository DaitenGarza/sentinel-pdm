"""Evaluation that matches how a maintenance organisation experiences a model.

Row-level metrics (ROC-AUC, PR-AUC, Brier) say how well hours are ranked, but planners care
about *alerts*: did we get a usable warning before each failure, how much lead time did it
give, and how many times were people sent out for nothing?

Alert logic (per pump), mimicking a DCS / CMMS notification:
  * an alert is **raised** once the score stays >= threshold for ``debounce_h`` hours;
  * it is closed after the score has been below threshold for ``clear_h`` hours;
  * if it is still active after ``realert_h`` hours it is raised again (a new work request).
    Without this, a model that is "always on" would get credit for every failure at the price
    of a single false alarm.

Every raise is one work request: someone inspects the pump. Matching uses the P-F interval
from reliability-centred maintenance: P is the point where the defect becomes detectable by an
inspection, F the functional failure. Each raise is classified as:
  * **tp**           raised inside [P, F - min_lead]: the inspection finds the defect and there is
                     time to plan the repair, so the breakdown is avoided;
  * **late**         raised in the last ``min_lead`` hours before F: too late to plan;
  * **pm_confirmed** raised inside [P, PM] of a degradation that preventive maintenance fixed;
  * **unresolved**   raised in the last ``credit_window_h`` of the record without a match: the
                     outcome is unknown (the pump may be failing right now);
  * **false_alarm**  everything else: an inspection that finds nothing.
A failure is **caught** if at least one tp raise precedes it; lead time = F - first tp raise.
Sudden failures have P = F, so no method can catch them; they are still counted as failures.
When the maintenance log has no P point (real data), P = F - ``credit_window_h`` is assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import AlertConfig, CostConfig


# --------------------------------------------------------------------------------------------
# row level
# --------------------------------------------------------------------------------------------
def row_metrics(y: np.ndarray, score: np.ndarray, probabilistic: bool = False) -> dict:
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    out = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "positive_rate": float(y.mean()),
        "n_rows": int(len(y)),
    }
    if probabilistic:
        out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    return out


# --------------------------------------------------------------------------------------------
# alert episodes
# --------------------------------------------------------------------------------------------
def _episodes_one(
    hours: np.ndarray, above: np.ndarray, debounce: int, clear: int, realert: int
) -> list[tuple[int, int]]:
    """Return [(raise_hour, end_hour)] alert raises for one pump (hours sorted, contiguous)."""
    if not above.any():
        return []
    a = above.astype(np.int8)
    d = np.diff(np.concatenate([[0], a, [0]]))
    run_starts = np.flatnonzero(d == 1)
    run_ends = np.flatnonzero(d == -1) - 1  # inclusive
    keep = (run_ends - run_starts + 1) >= debounce
    run_starts, run_ends = run_starts[keep], run_ends[keep]
    if run_starts.size == 0:
        return []
    raise_idx = run_starts + debounce - 1
    merged = []
    cur_s, cur_e = raise_idx[0], run_ends[0]
    for s_raise, s_run, e in zip(raise_idx[1:], run_starts[1:], run_ends[1:]):
        if s_run - cur_e - 1 < clear:  # gap shorter than clear time -> same alert
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s_raise, e
    merged.append((cur_s, cur_e))
    raises = []
    for s_i, e_i in merged:
        for r in range(s_i, e_i + 1, realert):
            raises.append((int(hours[r]), int(hours[min(r + realert - 1, e_i)])))
    return raises


def alert_episodes(frame: pd.DataFrame, score_col: str, threshold: float, cfg: AlertConfig) -> pd.DataFrame:
    """Alert raises for every pump. frame: asset_id, hour, <score_col>, sorted by asset_id, hour.

    NaN scores (pump stopped) count as 'below threshold'.
    """
    rows = []
    score = frame[score_col].to_numpy(dtype=np.float64)
    hours = frame["hour"].to_numpy()
    for asset_id, idx in frame.groupby("asset_id", sort=False).indices.items():
        above = np.nan_to_num(score[idx], nan=-np.inf) >= threshold
        for s, e in _episodes_one(hours[idx], above, cfg.debounce_h, cfg.clear_h, cfg.realert_h):
            rows.append((asset_id, s, e))
    return pd.DataFrame(rows, columns=["asset_id", "start_hour", "end_hour"])


def match_events(
    episodes: pd.DataFrame, events: pd.DataFrame, cfg: AlertConfig, record_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify failures (caught / missed) and alert raises (tp / late / pm_confirmed / unresolved / false_alarm)."""
    failures = events[events["event_type"] == "failure"].copy()
    pms = events[(events["event_type"] == "pm") & (events["mode"] != "none")]
    ep = episodes.copy().reset_index(drop=True)
    outcome = np.full(len(ep), "false_alarm", dtype=object)
    ep_asset = ep["asset_id"].to_numpy()
    ep_start = ep["start_hour"].to_numpy()

    def p_points(ev: pd.DataFrame) -> np.ndarray:
        fallback = ev["hour"].to_numpy(dtype=np.float64) - cfg.credit_window_h
        if "p_hour" not in ev:
            return fallback
        p = pd.to_numeric(ev["p_hour"], errors="coerce").to_numpy(dtype=np.float64)
        return np.where(np.isnan(p), fallback, p)

    caught, lead, late = [], [], []
    for asset_id, f_hour, p_hour in zip(failures["asset_id"].to_numpy(), failures["hour"].to_numpy(),
                                         p_points(failures)):
        same = ep_asset == asset_id
        tp = same & (ep_start >= p_hour) & (ep_start <= f_hour - cfg.min_lead_h)
        lt = same & (ep_start > f_hour - cfg.min_lead_h) & (ep_start < f_hour)
        outcome[tp] = "tp"
        outcome[lt & (outcome == "false_alarm")] = "late"
        caught.append(bool(tp.any()))
        lead.append(float(f_hour - ep_start[tp].min()) if tp.any() else np.nan)
        late.append(bool(lt.any()) and not tp.any())
    failures["caught"] = caught
    failures["lead_h"] = lead
    failures["late_only"] = late

    for asset_id, pm_hour, p_hour in zip(pms["asset_id"].to_numpy(), pms["hour"].to_numpy(), p_points(pms)):
        m = (outcome == "false_alarm") & (ep_asset == asset_id) & (ep_start <= pm_hour) & (ep_start >= p_hour)
        outcome[m] = "pm_confirmed"
    outcome[(outcome == "false_alarm") & (ep_start >= record_end - cfg.credit_window_h)] = "unresolved"
    ep["outcome"] = outcome
    return failures.reset_index(drop=True), ep


def event_metrics(failures: pd.DataFrame, episodes: pd.DataFrame, asset_years: float) -> dict:
    pred = failures[failures["predictable"]]
    n_fa = int((episodes["outcome"] == "false_alarm").sum())
    n_tp = int((episodes["outcome"] == "tp").sum())
    leads = failures.loc[failures["caught"], "lead_h"]
    return {
        "failures": int(len(failures)),
        "predictable_failures": int(len(pred)),
        "caught": int(failures["caught"].sum()),
        "caught_predictable": int(pred["caught"].sum()),
        "detection_rate": float(failures["caught"].mean()) if len(failures) else float("nan"),
        "detection_rate_predictable": float(pred["caught"].mean()) if len(pred) else float("nan"),
        "late_only": int(failures["late_only"].sum()),
        "median_lead_h": float(leads.median()) if len(leads) else float("nan"),
        "mean_lead_h": float(leads.mean()) if len(leads) else float("nan"),
        "alerts": int(len(episodes)),
        "true_alerts": n_tp,
        "false_alarms": n_fa,
        "pm_confirmed_alerts": int((episodes["outcome"] == "pm_confirmed").sum()),
        "false_alarms_per_asset_year": n_fa / asset_years if asset_years else float("nan"),
        "alert_precision": n_tp / (n_tp + n_fa) if (n_tp + n_fa) else float("nan"),
    }


# --------------------------------------------------------------------------------------------
# economics + threshold choice
# --------------------------------------------------------------------------------------------
def policy_cost(n_failures: int, n_caught: int, n_false_alarms: int, costs: CostConfig) -> dict:
    """Cost of a maintenance policy over the evaluated period.

    Caught failures become planned repairs; missed ones stay unplanned breakdowns with secondary
    damage and longer outages; every false alarm costs an inspection.
    """
    missed = n_failures - n_caught
    unplanned_each = costs.unplanned_repair_cost + costs.unplanned_downtime_h * costs.lost_production_per_h
    planned_each = costs.planned_repair_cost + costs.planned_downtime_h * costs.lost_production_per_h
    total = missed * unplanned_each + n_caught * planned_each + n_false_alarms * costs.false_alarm_cost
    return {
        "total_cost": float(total),
        "unplanned_failures": int(missed),
        "planned_repairs": int(n_caught),
        "false_alarms": int(n_false_alarms),
        "downtime_h": float(missed * costs.unplanned_downtime_h + n_caught * costs.planned_downtime_h),
        "unplanned_cost_each": float(unplanned_each),
        "planned_cost_each": float(planned_each),
    }


def choose_threshold(
    frame: pd.DataFrame, score_col: str, events: pd.DataFrame, cfg: AlertConfig, costs: CostConfig,
    record_end: int, asset_years: float, candidates: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Pick the alert threshold that minimises total maintenance cost on (out-of-fold) data,
    subject to the false-alarm budget (``max_false_alarms_per_asset_year``).

    If no candidate meets the budget, the threshold with the fewest false alarms is returned.
    """
    s = frame[score_col].to_numpy(dtype=np.float64)
    if candidates is None:
        qs = np.concatenate([np.linspace(0.80, 0.99, 39), np.linspace(0.991, 0.9995, 18)])
        candidates = np.unique(np.nanquantile(s, qs))
    n_fail = int((events["event_type"] == "failure").sum())
    rows = []
    for thr in candidates:
        ep = alert_episodes(frame, score_col, float(thr), cfg)
        failures, ep = match_events(ep, events, cfg, record_end)
        n_fa = int((ep["outcome"] == "false_alarm").sum())
        n_caught = int(failures["caught"].sum())
        c = policy_cost(n_fail, n_caught, n_fa, costs)
        rows.append({"threshold": float(thr), "caught": n_caught, "failures": n_fail, "false_alarms": n_fa,
                     "false_alarms_per_asset_year": n_fa / asset_years, "total_cost": c["total_cost"]})
    curve = pd.DataFrame(rows)
    ok = curve["false_alarms_per_asset_year"] <= cfg.max_false_alarms_per_asset_year
    curve["within_budget"] = ok
    pool = curve[ok] if ok.any() else curve[curve["false_alarms"] == curve["false_alarms"].min()]
    best = pool.loc[pool["total_cost"].idxmin(), "threshold"]
    return float(best), curve
