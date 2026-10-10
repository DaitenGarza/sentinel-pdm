"""Maintenance Economics: the business case for predictive maintenance, in your plant's numbers.

The model results are fixed (event counts measured on the held-out test fleet); the money is
not. Every cost assumption is editable, and the page recomputes each maintenance policy with
``sentinel_pdm.evaluation.policy_cost`` - the same function the pipeline used to choose the
alert thresholds - then scales from the test fleet's pump-years to the user's site.

Why the scaling is split in two: breakdowns and planned repairs scale with how often pumps fail
(editable, because the simulated fleet degrades faster than a typical refinery so the test has
enough failures to measure), while false-alarm inspections scale with pump-years of operation.
"""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # app/ folder
import common as C

C.page_setup("Maintenance Economics", "💰")
C.require_artifacts()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sentinel_pdm.config import Config, CostConfig  # noqa: E402
from sentinel_pdm.evaluation import policy_cost  # noqa: E402

# --------------------------------------------------------------------------------------------
# held-out results
# --------------------------------------------------------------------------------------------
M = C.load_metrics()
CFG = Config.from_dict(M.get("config", {}))
TEST = M.get("test", {})
EVENTS: dict = TEST.get("event", {})
DATA = M.get("data", {})
YEARS = float(TEST.get("asset_years") or DATA.get("n_test_assets", 16) * DATA.get("hours", 8760) / 8760)
N_FAIL = int(DATA.get("failures_test") or EVENTS.get("hybrid", {}).get("failures", 0))
N_PREDICTABLE = int(EVENTS.get("hybrid", {}).get("predictable_failures", N_FAIL))
SIM_RATE = N_FAIL / YEARS if YEARS else 0.0          # failures per pump-year on the simulated fleet

POLICIES = [p for p in ("reactive", "rule", "iforest", "mspc", "ml", "hybrid") if p == "reactive" or p in EVENTS]
NAMES = {"reactive": "Run to failure", "rule": "Alarm limits", "iforest": "Isolation Forest", "mspc": "MSPC",
         "ml": "Gradient boosting", "hybrid": "Sentinel hybrid"}
# held-out event counts per policy: (failures, caught, false alarms); run-to-failure catches nothing
COUNTS = {"reactive": (N_FAIL, 0, 0)} | {
    k: (int(EVENTS[k].get("failures", N_FAIL)), int(EVENTS[k].get("caught", 0)), int(EVENTS[k].get("false_alarms", 0)))
    for k in POLICIES if k != "reactive"
}

COST_INPUTS = [  # (CostConfig field, label, step, number format, help)
    ("lost_production_per_h", "Lost production ($ per hour down)", 250.0, "%.0f",
     "Margin lost while the pump is out of service and its duty is not covered (no spare, or a bottleneck unit)."),
    ("unplanned_repair_cost", "Unplanned repair cost ($)", 1000.0, "%.0f",
     "Breakdown repair including secondary damage (shaft, seal, impeller), overtime and expedited parts."),
    ("unplanned_downtime_h", "Unplanned downtime (h)", 4.0, "%.1f",
     "Hours from trip to back in service: diagnosis, waiting for parts and crew, repair, restart."),
    ("planned_repair_cost", "Planned repair cost ($)", 1000.0, "%.0f",
     "The same defect fixed early: parts staged, normal hours, no collateral damage."),
    ("planned_downtime_h", "Planned downtime (h)", 1.0, "%.1f",
     "Hours out of service for a planned repair, usually aligned with a process window or swing to the spare."),
    ("false_alarm_cost", "False-alarm inspection ($)", 100.0, "%.0f",
     "Vibration analyst plus mechanic sent to a pump that turns out to be healthy."),
]
# Metric second lines that give context rather than a change over time: no arrow, no colour.
INFO_DELTA = {"delta_color": "off", "delta_arrow": "off"}
DEFAULTS = {**asdict(CFG.costs), "pumps": 100, "system_cost": 50_000.0, "failure_rate": round(SIM_RATE, 2)}


def _reset_inputs() -> None:
    """Button callback (runs before the script, so widget values can still be set)."""
    for k, v in DEFAULTS.items():
        st.session_state[f"me_{k}"] = v


for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(f"me_{_k}", _v)


# --------------------------------------------------------------------------------------------
# economics
# --------------------------------------------------------------------------------------------
def annual_policy(key: str, costs: CostConfig, pumps: float, failure_rate: float) -> dict:
    """One policy's yearly cost at the user's site.

    ``policy_cost`` prices the held-out counts; failure-driven costs are then scaled by the site's
    failure rate and inspections by its pump-years.
    """
    n_fail, caught, fa = COUNTS[key]
    per_year = pumps / YEARS if YEARS else 0.0
    rate_factor = failure_rate / SIM_RATE if SIM_RATE else 0.0
    repairs = policy_cost(n_fail, caught, 0, costs)           # breakdowns + planned repairs, test fleet
    inspections = policy_cost(0, 0, fa, costs)                # false-alarm inspections, test fleet
    k_fail = per_year * rate_factor
    return {
        "policy": key,
        "breakdowns": (n_fail - caught) * k_fail,
        "planned_repairs": caught * k_fail,
        "false_alarms": fa * per_year,
        "breakdown_cost": (n_fail - caught) * repairs["unplanned_cost_each"] * k_fail,
        "planned_cost": caught * repairs["planned_cost_each"] * k_fail,
        "inspection_cost": inspections["total_cost"] * per_year,
        "total_cost": repairs["total_cost"] * k_fail + inspections["total_cost"] * per_year,
        "downtime_h": repairs["downtime_h"] * k_fail,
        "unplanned_each": repairs["unplanned_cost_each"],
        "planned_each": repairs["planned_cost_each"],
    }


def savings(key: str, costs: CostConfig, pumps: float, failure_rate: float) -> float:
    return (annual_policy("reactive", costs, pumps, failure_rate)["total_cost"]
            - annual_policy(key, costs, pumps, failure_rate)["total_cost"])


def money(x: float) -> str:
    """C.fmt_money with the sign in front of the currency symbol (-$1.2M, not $-1.2M)."""
    return f"-{C.fmt_money(-x)}" if np.isfinite(x) and x < 0 else C.fmt_money(x)


def md_money(x: float) -> str:
    """``money`` for Markdown text, where two dollar signs in a paragraph would open a LaTeX formula."""
    return money(x).replace("$", r"\$")


def fmt_period(years: float) -> str:
    """How long a cost takes to repay, in the unit a manager would quote."""
    if not np.isfinite(years) or years < 0:
        return "never at these assumptions"
    days = 365 * years
    if days < 14:
        return f"{max(days, 1):.0f} day{'s' if days >= 1.5 else ''}"
    if days < 120:
        return f"{days / 7:.0f} weeks"
    return f"{years * 12:.0f} months" if years < 3 else f"{years:.1f} years"


def money_axis(fig: go.Figure, axis: str = "x") -> None:
    fig.update_layout({f"{axis}axis": dict(tickprefix="$", tickformat="~s")})


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Your site")
    pumps = st.number_input("Pumps at your site", min_value=1, max_value=20_000, step=10, key="me_pumps",
                            help="Centrifugal pumps that would be monitored.")
    failure_rate = st.number_input(
        "Failures per pump-year", min_value=0.0, max_value=10.0, step=0.05, format="%.2f", key="me_failure_rate",
        help=f"Default: the simulated test fleet ({N_FAIL} failures in {YEARS:.0f} pump-years). Refinery "
             "centrifugal pumps commonly run 2-6 years between repairs (about 0.15-0.5 failures per pump-year); "
             "the simulator degrades pumps faster so a one-year test has enough failures to measure.",
    )
    system_cost = st.number_input(
        "Annual cost of running the system ($)", min_value=0.0, step=5_000.0, format="%.0f", key="me_system_cost",
        help="Software, historian integration and analyst time per year. Used only for the net-benefit and "
             "break-even figures; set 0 to see gross savings.")
    st.subheader("Cost assumptions")
    values = {}
    for field, label, step, fmt, help_text in COST_INPUTS:
        values[field] = st.number_input(label, min_value=0.0, step=step, format=fmt, key=f"me_{field}",
                                        help=help_text)
    st.button("Reset to defaults", on_click=_reset_inputs, key="me_reset", width="stretch")

costs = CostConfig(**values)
results = {k: annual_policy(k, costs, pumps, failure_rate) for k in POLICIES}
table = pd.DataFrame(results.values())
base = results["reactive"]
hy = results.get("hybrid", base)
gross = base["total_cost"] - hy["total_cost"]
net = gross - system_cost

# --------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------
st.title("💰 Maintenance Economics")
st.caption(
    f"What predictive maintenance is worth for **{pumps:,} pumps** failing **{failure_rate:.2f}** times per "
    "pump-year on average. Detection results are fixed (held-out test fleet); every cost is editable in the sidebar."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Annual savings vs. run-to-failure", money(gross),
          delta=f"{money(net)} net of running cost" if system_cost else
          (f"{gross / base['total_cost']:.0%} lower maintenance cost" if base["total_cost"] else None))
down_avoided = base["downtime_h"] - hy["downtime_h"]
k2.metric("Downtime avoided", f"{down_avoided:,.0f} h / yr",
          delta=f"{down_avoided / base['downtime_h']:.0%} less downtime" if base["downtime_h"] else None,
          **INFO_DELTA)
k3.metric("Breakdowns avoided", f"{hy['planned_repairs']:,.1f} / yr",
          delta=f"{hy['breakdowns']:,.1f} / yr still unplanned", **INFO_DELTA,
          help="Caught failures become planned repairs. The remainder are mostly 'sudden' failures with no "
               "measurable precursor, which no monitoring system can predict.")
margin = hy["unplanned_each"] - hy["planned_each"]
if system_cost and margin > 0:
    breakeven = system_cost / margin
    k4.metric("Break-even (breakdowns avoided)", f"{breakeven:,.1f} / yr",
              delta=f"running cost repaid in {fmt_period(system_cost / gross if gross > 0 else np.inf)}",
              **INFO_DELTA,
              help="Each breakdown turned into a planned repair saves "
                   f"{C.fmt_money(margin)}; the system pays its yearly running cost once it does that this many times.")
else:
    k4.metric("Cost per pump-year (Sentinel)", money(hy["total_cost"] / pumps),
              delta=f"vs. {money(base['total_cost'] / pumps)} run-to-failure", **INFO_DELTA)

if SIM_RATE and abs(failure_rate - SIM_RATE) < 0.01 and SIM_RATE > 0.6:
    st.info(
        f"The simulated fleet fails about **{SIM_RATE:.1f} times per pump-year**, several times more often than a "
        "typical refinery pump (roughly 0.15-0.5), so that a one-year test holds enough failures to measure. The "
        "absolute dollars above are therefore high: enter your own failure rate in the sidebar. The *percentage* "
        "saving barely depends on it.", icon="ℹ️")

with st.container(border=True):
    st.markdown("**Assumptions**")
    st.markdown(
        f"- Cost figures are **illustrative Gulf-Coast process-plant numbers** (USD); replace them with your "
        f"plant's own. A breakdown costs **{md_money(hy['unplanned_each'])}** "
        f"({md_money(costs.unplanned_repair_cost)} repair + {costs.unplanned_downtime_h:g} h × "
        f"{md_money(costs.lost_production_per_h)}/h); the same defect caught early costs "
        f"**{md_money(hy['planned_each'])}**.\n"
        f"- Detection and false-alarm rates come from the **simulated held-out test fleet**: {N_FAIL} failures "
        f"({N_FAIL - N_PREDICTABLE} sudden, with no precursor) in {YEARS:.0f} pump-years, scored by models that "
        "never saw those pumps.\n"
        "- Scaling is linear: breakdowns and repairs with your failure rate, inspections with pump-years. Spare-pump "
        "coverage, inventory savings and safety/environmental consequences are not modelled, so the case is "
        "conservative where a failure would also release product."
    )

# ---- cost and downtime by policy ---------------------------------------------------------------
st.subheader("What each maintenance policy costs per year")
c1, c2 = st.columns(2)
names = [NAMES[k] for k in table["policy"]]
colors = [C.METHOD_COLORS.get(k, "#9ca3af") for k in table["policy"]]
fig = go.Figure(go.Bar(
    x=table["total_cost"], y=names, orientation="h", marker_color=colors,
    text=[money(v) for v in table["total_cost"]], textposition="outside", cliponaxis=False,
    customdata=np.column_stack([table["breakdown_cost"], table["planned_cost"], table["inspection_cost"],
                                table["breakdowns"], table["planned_repairs"], table["false_alarms"]]),
    hovertemplate="<b>%{y}</b><br>total $%{x:,.0f}<br>breakdowns: %{customdata[3]:.1f} → $%{customdata[0]:,.0f}"
                  "<br>planned repairs: %{customdata[4]:.1f} → $%{customdata[1]:,.0f}"
                  "<br>false-alarm inspections: %{customdata[5]:.1f} → $%{customdata[2]:,.0f}<extra></extra>",
))
fig.update_layout(title="Annual maintenance cost", xaxis_title="USD per year", showlegend=False)
fig.update_xaxes(range=[0, max(table["total_cost"].max(), 1) * 1.18])
money_axis(fig)
c1.plotly_chart(C.style_fig(fig, height=360, legend_top=False), width="stretch")

fig = go.Figure(go.Bar(
    x=table["downtime_h"], y=names, orientation="h", marker_color=colors,
    text=[f"{v:,.0f} h" for v in table["downtime_h"]], textposition="outside", cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>%{x:,.0f} hours out of service per year<extra></extra>",
))
fig.update_layout(title="Pump downtime", xaxis_title="Hours out of service per year", showlegend=False)
fig.update_xaxes(range=[0, max(table["downtime_h"].max(), 1) * 1.18])
c2.plotly_chart(C.style_fig(fig, height=360, legend_top=False), width="stretch")

# ---- waterfall ---------------------------------------------------------------------------------
st.subheader("From run-to-failure to Sentinel")
steps = [
    ("Run-to-failure cost", base["total_cost"], "absolute"),
    ("Breakdowns avoided", -(base["breakdown_cost"] - hy["breakdown_cost"]), "relative"),
    ("Planned repairs instead", hy["planned_cost"] - base["planned_cost"], "relative"),
    ("False-alarm inspections", hy["inspection_cost"] - base["inspection_cost"], "relative"),
]
if system_cost:
    steps.append(("System running cost", system_cost, "relative"))
steps.append(("Sentinel cost" + (" incl. system" if system_cost else ""), None, "total"))
fig = go.Figure(go.Waterfall(
    orientation="v", x=[s[0] for s in steps], y=[s[1] if s[1] is not None else 0 for s in steps],
    measure=[s[2] for s in steps], connector=dict(line=dict(color="#4b5563", width=1, dash="dot")),
    increasing=dict(marker=dict(color="#ef4444")), decreasing=dict(marker=dict(color="#34d399")),
    totals=dict(marker=dict(color=C.ACCENT)),
    text=[money(base["total_cost"]), *[("+" if s[1] >= 0 else "") + money(s[1]) for s in steps[1:-1]],
          money(hy["total_cost"] + system_cost)],
    textposition="outside", cliponaxis=False,
    hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
))
fig.update_layout(title="Annual cost bridge", yaxis_title="USD per year", showlegend=False)
fig.update_yaxes(range=[0, base["total_cost"] * 1.15 if base["total_cost"] else 1])
money_axis(fig, "y")
st.plotly_chart(C.style_fig(fig, height=400, legend_top=False), width="stretch")
st.caption(
    "Green bars lower the bill, red bars raise it. Catching a failure does not make the repair free: the bearing "
    "still has to be changed, but on a planned day, with parts staged and before it wrecks the shaft and seal."
)

# ---- sensitivity -------------------------------------------------------------------------------
st.subheader("How robust is the case?")
c1, c2 = st.columns(2)
ml_policies = [k for k in POLICIES if k != "reactive"]


def sweep(field: str, grid: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for key in ml_policies:
        ys = [savings(key, CostConfig(**{**values, field: float(v)}), pumps, failure_rate) for v in grid]
        fig.add_scatter(x=grid, y=ys, mode="lines", name=NAMES[key],
                        line=dict(color=C.METHOD_COLORS.get(key), width=3.2 if key == "hybrid" else 1.8),
                        hovertemplate=f"{NAMES[key]}<br>%{{x:$,.0f}} → saves %{{y:$,.3s}} / yr<extra></extra>")
    fig.add_vline(x=values[field], line=dict(color="#e5e7eb", dash="dot", width=1),
                  annotation_text="current", annotation_position="bottom right")
    fig.add_hline(y=0, line=dict(color="#6b7280", width=1))
    money_axis(fig, "y")
    money_axis(fig, "x")
    return fig


lp = values["lost_production_per_h"]
fig = sweep("lost_production_per_h", np.linspace(0, max(3 * lp, 5_000), 41))
fig.update_layout(title="Savings vs. lost-production rate", xaxis_title="Lost production ($ per hour down)",
                  yaxis_title="Annual savings vs. run-to-failure")
c1.plotly_chart(C.style_fig(fig, height=400), width="stretch")
fa_cost = values["false_alarm_cost"]
fig = sweep("false_alarm_cost", np.linspace(0, max(10 * fa_cost, 20_000), 41))
fig.update_layout(title="Savings vs. cost of a false alarm", xaxis_title="Cost per false-alarm inspection ($)",
                  yaxis_title="Annual savings vs. run-to-failure")
c2.plotly_chart(C.style_fig(fig, height=400), width="stretch")
fa_hy = COUNTS.get("hybrid", (0, 0, 0))[2]
st.caption(
    "Every predictive policy gains as downtime gets more expensive; what separates them is how many failures they "
    "catch. A false alarm costs an inspection, so the cost of nuisance alerts only matters for methods that raise "
    f"them: the hybrid raised {fa_hy} on the test fleet"
    + (", so its line is flat." if fa_hy == 0 else ".")
    + " (The hidden cost of nuisance alerts, operators learning to ignore the system, is why the alert thresholds "
      "are also held to an alarm budget.)"
)

base_saving = savings("hybrid", costs, pumps, failure_rate) if "hybrid" in POLICIES else 0.0
tornado = []
for field, label, *_ in COST_INPUTS:
    lo = savings("hybrid", CostConfig(**{**values, field: values[field] * 0.5}), pumps, failure_rate)
    hi = savings("hybrid", CostConfig(**{**values, field: values[field] * 1.5}), pumps, failure_rate)
    tornado.append({"label": label, "lo": lo, "hi": hi, "spread": abs(hi - lo)})
tornado.append({"label": "Failures per pump-year", "lo": savings("hybrid", costs, pumps, failure_rate * 0.5),
                "hi": savings("hybrid", costs, pumps, failure_rate * 1.5), "spread": 0.0})
tornado[-1]["spread"] = abs(tornado[-1]["hi"] - tornado[-1]["lo"])
td = pd.DataFrame(tornado).sort_values("spread")
fig = go.Figure()
for col, name, color in (("lo", "Input −50 %", "rgba(245,158,11,0.45)"), ("hi", "Input +50 %", C.ACCENT)):
    fig.add_bar(y=td["label"], x=td[col] - base_saving, base=base_saving, orientation="h", name=name,
                marker_color=color, customdata=td[col],
                hovertemplate="%{y}<br>" + name + ": saves $%{customdata:,.0f} / yr<extra></extra>")
fig.add_vline(x=base_saving, line=dict(color="#e5e7eb", width=1),
              annotation_text=f"base {money(base_saving)}", annotation_position="top")
fig.update_layout(title="Sentinel savings when each input moves ±50 %", barmode="overlay",
                  xaxis_title="Annual savings vs. run-to-failure", bargap=0.35)
money_axis(fig)
st.plotly_chart(C.style_fig(fig, height=380), width="stretch")
st.caption("Longest bars first: the assumptions worth pinning down with your own CMMS data before taking the case "
           "to management. Savings scale one-for-one with the failure rate, so start there.")

# ---- table -------------------------------------------------------------------------------------
st.subheader("Policy comparison")
out = pd.DataFrame({
    "Policy": names,
    "Breakdowns / yr": table["breakdowns"],
    "Planned repairs / yr": table["planned_repairs"],
    "False-alarm inspections / yr": table["false_alarms"],
    "Downtime (h / yr)": table["downtime_h"],
    "Annual cost": table["total_cost"],
    "Savings vs. run-to-failure": base["total_cost"] - table["total_cost"],
    "Saving %": np.where(base["total_cost"] > 0, (base["total_cost"] - table["total_cost"]) / max(base["total_cost"], 1e-9),
                         np.nan) * 100,
    "Cost per pump-year": table["total_cost"] / pumps,
})
st.dataframe(
    out, hide_index=True, width="stretch",
    column_config={
        "Breakdowns / yr": st.column_config.NumberColumn(format="%.1f"),
        "Planned repairs / yr": st.column_config.NumberColumn(format="%.1f"),
        "False-alarm inspections / yr": st.column_config.NumberColumn(format="%.1f"),
        "Downtime (h / yr)": st.column_config.NumberColumn(format="%,.0f"),
        "Annual cost": st.column_config.NumberColumn(format="$%,.0f"),
        "Savings vs. run-to-failure": st.column_config.NumberColumn(format="$%,.0f"),
        "Saving %": st.column_config.NumberColumn(format="%.1f%%"),
        "Cost per pump-year": st.column_config.NumberColumn(format="$%,.0f"),
    },
)
st.download_button("Download this comparison (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name="sentinel_policy_economics.csv", mime="text/csv", key="me_download")
