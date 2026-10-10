"""Score Your Data: run the trained models on your own historian export.

The upload goes through exactly the path the held-out test fleet went through during training
(``sentinel_pdm.scoring.score_readings``: causal cleaning -> features -> every model), so what you
see here is what the system would have said, hour by hour, about your pumps.

Most of this page is input validation. Historian exports are messy (sub-hourly samples, gaps,
text in numeric columns, timestamps instead of hour counters), and a scoring tool that fails
with a stack trace, or silently scores garbage, is worse than none. Every problem gets a
specific message naming the column, the bad values and the CSV line numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # app/ folder
import common as C

C.page_setup("Score Your Data", "🧪")
C.require_artifacts()

import html  # noqa: E402
import io  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sentinel_pdm.config import SENSORS, Config  # noqa: E402
from sentinel_pdm.evaluation import alert_episodes  # noqa: E402
from sentinel_pdm.features import BASELINE_H, SIGNALS  # noqa: E402
from sentinel_pdm.scoring import score_readings  # noqa: E402

M = C.load_metrics()
CFG = Config.from_dict(M.get("config", {}))
VERSION = str(M.get("generated_at", ""))
THR = float(M.get("thresholds", {}).get("hybrid", 0.5))
MAX_ROWS = 500_000                 # about 57 pump-years of hourly data; keeps scoring under a minute
SAMPLE_DAYS = 60
SAMPLE_END_BEFORE_FAILURE_H = 48   # the sample stops 2 days before the trip: a failure still developing
DEFAULT_AMBIENT_C = 25.0
REQUIRED_NUMERIC = ("load_pct", "speed_rpm", *SENSORS)
OUTPUT_COLUMNS = ["asset_id", "hour", "timestamp", "running", "load_pct", "speed_rpm", "ambient_temp", *SENSORS]
EVENT_TYPES = ("failure", "pm")
EVENT_MODES = (*C.FAILURE_MODES, "sudden", "none")


class InputError(ValueError):
    """A problem with the user's file, phrased for the user (one message per problem)."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


# --------------------------------------------------------------------------------------------
# sample data
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing the sample...")
def sample_readings(version: str) -> tuple[pd.DataFrame, dict]:
    """60 days of one held-out pump, ending two days before a failure Sentinel caught.

    The window must start on a healthy, maintenance-free stretch: with no maintenance log the
    models take each pump's first week of running as its healthy baseline. Bearing failures are
    preferred (the textbook signature), then the cleanest baseline (longest healthy lead-in).
    """
    fo = C.load_failure_outcomes()
    fo = fo[(fo["method"] == "hybrid") & (fo["split"] == "test") & fo["caught"].astype(bool)
            & fo["predictable"].astype(bool)].copy()
    ev = C.load_events()
    window = SAMPLE_DAYS * 24
    fo["end"] = fo["hour"] - SAMPLE_END_BEFORE_FAILURE_H
    fo["start"] = fo["end"] - window
    prev_end = [ev.loc[(ev["asset_id"] == a) & (ev["hour"] < h), "end_hour"].max() for a, h in zip(fo["asset_id"], fo["hour"])]
    fo["prev_end"] = pd.Series(prev_end, index=fo.index).fillna(-1)
    fo["healthy_lead_in"] = fo["onset_hour"] - fo["start"]
    ok = (fo["start"] >= 0) & (fo["prev_end"] < fo["start"]) & (fo["healthy_lead_in"] >= 2 * BASELINE_H)
    pool = fo[ok]
    if pool.empty:                                   # different seed / shorter record: relax the rules
        pool = fo.assign(start=fo["start"].clip(lower=0))
    if pool.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {}
    pool = pool.assign(pref=(pool["mode"] == "bearing") & (pool["lead_h"] >= 168))
    pick = pool.sort_values(["pref", "healthy_lead_in"], ascending=False).iloc[0]
    r = C.load_readings()
    r = r[(r["asset_id"] == pick["asset_id"]) & (r["hour"] >= pick["start"]) & (r["hour"] < pick["end"])]
    sample = r[[c for c in OUTPUT_COLUMNS if c in r.columns]].reset_index(drop=True)   # no gt_* columns
    meta = {"asset_id": pick["asset_id"], "mode": pick["mode"], "event_id": pick["event_id"],
            "failure_time": str(pick["timestamp"])[:16], "lead_h": float(pick["lead_h"])}
    return sample, meta


# --------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------
def _lines(mask: pd.Series, n: int = 3) -> str:
    """CSV line numbers (header = line 1) of the first few rows where mask is True."""
    idx = np.flatnonzero(mask.to_numpy())[:n] + 2
    more = " ..." if mask.sum() > n else ""
    return ", ".join(str(i) for i in idx) + more


def _examples(values: pd.Series, n: int = 3) -> str:
    return ", ".join(f"'{v}'" for v in values.astype(str).unique()[:n])


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept column names in any case / with stray spaces (Historian exports vary)."""
    return df.rename(columns={c: str(c).strip().lower() for c in df.columns})


def _to_numeric(df: pd.DataFrame, col: str, problems: list[str]) -> pd.Series:
    raw = df[col]
    num = pd.to_numeric(raw, errors="coerce")
    text = raw.astype(str).str.strip()
    bad = num.isna() & raw.notna() & (text != "") & ~text.str.lower().isin(["nan", "na", "n/a", "null", "none"])
    if bad.any():
        problems.append(f"Column `{col}` must be numeric, but {int(bad.sum())} value(s) are not "
                        f"(e.g. {_examples(raw[bad])}; CSV line {_lines(bad)}).")
    return num.astype("float64")


@st.cache_data(show_spinner=False, max_entries=16)
def read_csv(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        raise InputError(["The file is empty."]) from None
    except UnicodeDecodeError:
        raise InputError(["The file is not UTF-8 text. Export it from Excel as 'CSV UTF-8'."]) from None
    except pd.errors.ParserError as e:
        raise InputError([f"The file could not be parsed as CSV: {str(e).strip()}"]) from None


def prepare_readings(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalise a readings table into the schema ``score_readings`` expects.

    Returns (readings sorted by asset_id/hour on a complete hourly grid, notes for the user).
    Raises InputError listing every problem found, so the user can fix the file in one go.
    """
    df = _canonical_columns(df.copy())
    problems: list[str] = []
    notes: list[str] = []
    missing = [c for c in ("asset_id", *REQUIRED_NUMERIC) if c not in df.columns]
    if "hour" not in df.columns and "timestamp" not in df.columns:
        missing.insert(1, "hour or timestamp")
    if missing:
        raise InputError([f"Missing required column(s): {', '.join(f'`{c}`' for c in missing)}. "
                          "See the expected schema above."])
    if len(df) == 0:
        raise InputError(["The file has a header but no data rows."])

    blank_id = df["asset_id"].isna() | (df["asset_id"].astype(str).str.strip() == "")
    if blank_id.any():
        problems.append(f"`asset_id` is empty on {int(blank_id.sum())} row(s) (CSV line {_lines(blank_id)}).")
    df["asset_id"] = df["asset_id"].astype(str).str.strip()

    for col in (*REQUIRED_NUMERIC, *[c for c in ("ambient_temp",) if c in df.columns]):
        df[col] = _to_numeric(df, col, problems)

    # running flag: optional; accept 0/1 and true/false
    if "running" in df.columns:
        txt = df["running"].astype(str).str.strip().str.lower()
        mapped = txt.map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "on": 1.0, "off": 0.0})
        run = pd.to_numeric(df["running"], errors="coerce").where(mapped.isna(), mapped)
        bad = run.isna() | ~run.isin([0, 1])
        if bad.any():
            problems.append(f"`running` must be 0 or 1 on every row; {int(bad.sum())} row(s) are not "
                            f"(e.g. {_examples(df.loc[bad, 'running'])}; CSV line {_lines(bad)}).")
        df["running"] = run.fillna(1).astype(np.int8)
    else:
        df["running"] = np.int8(1)
        notes.append("No `running` column: every row is assumed to be a running pump. Stopped periods will "
                     "look like abnormal readings, so add the run flag if you have it.")
    if "ambient_temp" not in df.columns or df["ambient_temp"].isna().all():
        df["ambient_temp"] = DEFAULT_AMBIENT_C
        notes.append(f"No `ambient_temp`: assumed {DEFAULT_AMBIENT_C:.0f} °C. Temperature-rise features will be "
                     "less accurate in hot or cold weather.")

    # time axis: integer hour, or a timestamp converted to hours since each pump's first reading
    use_ts = "timestamp" in df.columns and ("hour" not in df.columns or df["hour"].isna().all())
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
        bad = ts.isna() & df["timestamp"].notna()
        if bad.any():
            problems.append(f"`timestamp` could not be read as a date/time on {int(bad.sum())} row(s) "
                            f"(e.g. {_examples(df.loc[bad, 'timestamp'])}; CSV line {_lines(bad)}). "
                            "Use ISO format such as 2025-03-01 14:00.")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
        df["timestamp"] = ts
    if use_ts:
        if df["timestamp"].isna().any():
            problems.append(f"`timestamp` is empty on {int(df['timestamp'].isna().sum())} row(s) "
                            f"(CSV line {_lines(df['timestamp'].isna())}).")
    else:
        hour = _to_numeric(df, "hour", problems)
        bad = hour.isna() | (hour < 0) | (hour % 1 != 0)
        if bad.any():
            problems.append(f"`hour` must be a whole number >= 0 on every row; {int(bad.sum())} row(s) are not "
                            f"(e.g. {_examples(df.loc[bad, 'hour'])}; CSV line {_lines(bad)}).")
        df["hour"] = hour
    if problems:
        raise InputError(problems)

    if use_ts:
        df["timestamp"] = df["timestamp"].dt.floor("h")
        first = df.groupby("asset_id")["timestamp"].transform("min")
        df["hour"] = ((df["timestamp"] - first) / pd.Timedelta(hours=1)).round().astype(np.int64)
        notes.append("`hour` derived from `timestamp` (hours since each pump's first reading).")
    df["hour"] = df["hour"].astype(np.int64)

    # several readings in one hour (sub-hourly export): hourly averages, which is what the models learned on
    numeric = ["load_pct", "speed_rpm", "ambient_temp", *SENSORS]
    n_before = len(df)
    if df.duplicated(["asset_id", "hour"]).any():
        agg = {c: "mean" for c in numeric} | {"running": "max"}
        if "timestamp" in df.columns:
            agg["timestamp"] = "min"
        df = df.groupby(["asset_id", "hour"], as_index=False).agg(agg)
        notes.append(f"Averaged {n_before:,} readings into {len(df):,} hourly values (several readings per hour).")

    # complete hourly grid per pump: historian gaps become missing values the cleaner understands
    full = []
    for asset_id, g in df.groupby("asset_id", sort=True):
        hours = np.arange(int(g["hour"].min()), int(g["hour"].max()) + 1)
        full.append(g.set_index("hour").reindex(hours).rename_axis("hour").reset_index().assign(asset_id=asset_id))
    out = pd.concat(full, ignore_index=True)
    inserted = len(out) - len(df)
    if inserted:
        notes.append(f"{inserted:,} missing hour(s) inserted as data gaps (sensor values left empty).")
    out["running"] = out.groupby("asset_id")["running"].ffill().fillna(1).astype(np.int8)
    out["ambient_temp"] = out.groupby("asset_id")["ambient_temp"].ffill()
    if "timestamp" in out.columns:
        first = out.groupby("asset_id")["timestamp"].transform("min") - pd.to_timedelta(
            out.groupby("asset_id")["hour"].transform("min"), unit="h")
        out["timestamp"] = first + pd.to_timedelta(out["hour"], unit="h")
    if len(out) > MAX_ROWS:
        raise InputError([f"{len(out):,} pump-hours is more than this page scores at once ({MAX_ROWS:,}). "
                          "Split the file by pump or by period."])

    run_hours = out.groupby("asset_id")["running"].sum()
    short = run_hours[run_hours < 2 * BASELINE_H]
    if len(short):
        notes.append(f"{', '.join(short.index[:5])}{' ...' if len(short) > 5 else ''}: less than "
                     f"{2 * BASELINE_H // 24} days of running data. Each pump's first week of running is its "
                     "healthy baseline, so scores need a few weeks of history to be meaningful.")
    cols = [c for c in OUTPUT_COLUMNS if c in out.columns]
    return out[cols].sort_values(["asset_id", "hour"]).reset_index(drop=True), notes


def prepare_events(df: pd.DataFrame, readings: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate a maintenance log (asset_id, event_type, hour/end_hour or timestamp/end_timestamp, mode)."""
    df = _canonical_columns(df.copy())
    problems, notes = [], []
    missing = [c for c in ("asset_id", "event_type") if c not in df.columns]
    by_hour = {"hour", "end_hour"} <= set(df.columns)
    by_time = {"timestamp", "end_timestamp"} <= set(df.columns)
    if not by_hour and not by_time:
        missing.append("hour + end_hour (or timestamp + end_timestamp)")
    if missing:
        raise InputError([f"Maintenance log is missing column(s): {', '.join(missing)}."])
    if len(df) == 0:
        return pd.DataFrame(columns=["asset_id", "event_type", "hour", "end_hour", "mode"]), notes
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    bad = ~df["event_type"].isin(EVENT_TYPES)
    if bad.any():
        problems.append(f"Maintenance log `event_type` must be 'failure' or 'pm' (e.g. {_examples(df.loc[bad, 'event_type'])}; "
                        f"CSV line {_lines(bad)}).")
    df["mode"] = df["mode"].fillna("none").astype(str).str.strip().str.lower() if "mode" in df.columns else "none"
    df.loc[~df["mode"].isin(EVENT_MODES), "mode"] = "none"
    if by_hour:
        for col in ("hour", "end_hour"):
            df[col] = _to_numeric(df, col, problems)
    else:
        if "timestamp" not in readings.columns:
            raise InputError(["The maintenance log uses timestamps, so the readings must have a `timestamp` column too."])
        origin = readings.groupby("asset_id")["timestamp"].min() - pd.to_timedelta(
            readings.groupby("asset_id")["hour"].min(), unit="h")
        for src, col in (("timestamp", "hour"), ("end_timestamp", "end_hour")):
            ts = pd.to_datetime(df[src], errors="coerce", format="mixed")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
            bad = ts.isna()
            if bad.any():
                problems.append(f"Maintenance log `{src}` is not a date/time on CSV line {_lines(bad)}.")
            df[col] = ((ts - df["asset_id"].map(origin)) / pd.Timedelta(hours=1)).round()
    if problems:
        raise InputError(problems)
    bad = df["hour"].isna() | df["end_hour"].isna() | (df["end_hour"] < df["hour"])
    unknown = ~df["asset_id"].isin(readings["asset_id"].unique())
    if unknown.any():
        notes.append(f"Maintenance log: {int(unknown.sum())} event(s) for pumps not in the readings were ignored.")
    if (bad & ~unknown).any():
        raise InputError([f"Maintenance log: `end_hour` must be at or after `hour` (CSV line {_lines(bad & ~unknown)})."])
    df = df[~unknown & ~bad]
    df[["hour", "end_hour"]] = df[["hour", "end_hour"]].astype(np.int64)
    notes.append(f"Maintenance log: {len(df)} work order(s) applied (feature windows restart after each).")
    return df[["asset_id", "event_type", "hour", "end_hour", "mode"]].sort_values(["asset_id", "hour"]), notes


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def score(raw: pd.DataFrame, events: pd.DataFrame | None, version: str) -> pd.DataFrame:
    """Score with the deployed bundle (cached so widget changes do not re-score)."""
    _, preds = score_readings(raw, events, C.load_model_bundle())
    if "timestamp" not in preds.columns and "timestamp" in raw.columns:
        preds.insert(2, "timestamp", raw.sort_values(["asset_id", "hour"])["timestamp"].to_numpy())
    return preds


def pump_summary(preds: pd.DataFrame, raises: pd.DataFrame) -> pd.DataFrame:
    """Latest state of each pump: status now, plus risk / RUL / mode from its last running hour."""
    rows = []
    order = {s: i for i, s in enumerate(C.STATUS_ORDER)}
    for asset_id, g in preds.groupby("asset_id", sort=False):
        last = g.iloc[-1]
        run = g[g["risk"].notna()]
        r = run.iloc[-1] if len(run) else last
        risk = float(r["risk"]) if len(run) else np.nan
        watch = np.isfinite(risk) and risk >= THR / 2
        a = raises[raises["asset_id"] == asset_id]
        rows.append({
            "Pump": asset_id,
            "Status": str(last["status"]),
            "Risk (7 days)": 100 * risk,
            "RUL P50": C.fmt_hours(float(r["rul_p50"])) if watch else "-",
            "RUL P10-P90": (f"{C.fmt_hours(float(r['rul_p10']))} to {C.fmt_hours(float(r['rul_p90']))}"
                            if watch else "-"),
            "Likely mode": C.MODE_SHORT.get(r["pred_mode"], r["pred_mode"]) if watch else "-",
            "Alerts raised": len(a),
            "Last reading": last["timestamp"] if "timestamp" in g.columns else last["hour"],
            "Top drivers": C.drivers_text(r, n=2) if np.isfinite(risk) else "-",   # free text last: may be cut
            "_order": order.get(str(last["status"]), len(order)),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["_order", "Risk (7 days)"], ascending=[True, False]).drop(columns="_order")


# --------------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------------
st.title("🧪 Score Your Data")
st.caption(
    "Run the trained models on your own pumps. Upload an hourly historian export (one row per pump per hour) "
    "and get the same risk, remaining-life and failure-mode outputs the fleet pages show. Nothing leaves this "
    "machine: scoring runs locally."
)

with st.expander("Expected file format", expanded=False):
    schema = [("asset_id", "required", "text", "Pump tag, e.g. P-101A"),
              ("timestamp or hour", "one required", "date/time or integer",
               "ISO timestamps (sub-hourly is averaged to hourly), or an integer hour counter"),
              ("running", "optional, default 1", "0 / 1", "1 while the pump runs; readings while stopped are ignored"),
              ("load_pct", "required", "%", "Process load, 0-105"),
              ("speed_rpm", "required", "rpm", "Shaft speed (constant for fixed-speed pumps)"),
              ("ambient_temp", "optional, default 25", "°C", "Ambient temperature")]
    schema += [(s, "required", C.SIGNAL_INFO[s][1] or "-", C.SIGNAL_INFO[s][0]) for s in SENSORS]
    st.dataframe(pd.DataFrame(schema, columns=["Column", "Required", "Unit", "Meaning"]), hide_index=True,
                 width="stretch", height=38 + 35 * len(schema))
    st.markdown(
        "Empty cells are fine (treated as missing data). Column names are case-insensitive; extra columns are "
        "ignored. **Maintenance log (optional):** `asset_id`, `event_type` (`failure` or `pm`), `hour` and "
        "`end_hour` (or `timestamp` and `end_timestamp`), and `mode` (`bearing`, `misalignment`, `cavitation`, "
        "`winding`, `sudden`, `none`). Each work order restarts the pump's rolling windows and healthy baseline, "
        "exactly as the models were trained."
    )

sample, sample_meta = sample_readings(VERSION)
ss = st.session_state
ss.setdefault("sd_source", None)


def _use_sample() -> None:
    ss["sd_source"] = "sample"


def _use_upload() -> None:
    ss["sd_source"] = "upload"


c1, c2 = st.columns([1, 1.3])
with c1:
    st.markdown("**Try it with sample data**")
    if sample.empty:
        st.info("No sample available in this run.")
    else:
        st.caption(f"{SAMPLE_DAYS} days of held-out pump **{sample_meta['asset_id']}** (the models never saw it), "
                   f"ending two days before a {sample_meta['mode']} failure ({sample_meta['event_id']}, "
                   f"{sample_meta['failure_time']}). Ground-truth columns removed.")
        b1, b2 = st.columns(2)
        b1.download_button("Download sample CSV", sample.to_csv(index=False).encode("utf-8"),
                           file_name=f"sentinel_sample_{sample_meta['asset_id']}.csv", mime="text/csv",
                           icon=":material/download:", key="sd_download_sample", width="stretch")
        b2.button("Use sample data", icon=":material/play_arrow:", on_click=_use_sample, key="sd_use_sample",
                  type="primary", width="stretch")
with c2:
    up = st.file_uploader("Sensor readings (CSV)", type=["csv"], key="sd_readings", on_change=_use_upload)
    up_ev = st.file_uploader("Maintenance log (CSV, optional)", type=["csv"], key="sd_events", on_change=_use_upload)

source = ss["sd_source"]
if source == "upload" and up is None:   # the uploaded file was removed again
    source = None
if source is None:
    st.info("Upload a readings file or click **Use sample data** to see the scoring in action.", icon="⬆️")
    st.stop()

# ---- load + validate -----------------------------------------------------------------------
try:
    if source == "sample":
        raw_in, ev_in, label = sample, None, f"sample pump {sample_meta.get('asset_id', '')}"
    else:
        raw_in = read_csv(up.getvalue())
        ev_in = read_csv(up_ev.getvalue()) if up_ev is not None else None
        label = up.name
    readings, notes = prepare_readings(raw_in)
    events = None
    if ev_in is not None:
        events, ev_notes = prepare_events(ev_in, readings)
        notes += ev_notes
except InputError as e:
    st.error("**The file could not be scored.**\n\n" + "\n".join(f"- {p}" for p in e.problems), icon="🚫")
    st.stop()

for n in notes:
    st.warning(n, icon="⚠️")

with st.spinner(f"Scoring {len(readings):,} pump-hours from {label}: cleaning, features, models..."):
    preds = score(readings, events, VERSION)

frame = preds[["asset_id", "hour", "risk"]].sort_values(["asset_id", "hour"], kind="stable")
raises = alert_episodes(frame, "risk", THR, CFG.alerts)
summary = pump_summary(preds, raises)
has_ts = "timestamp" in preds.columns

st.subheader(f"Results: {label}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pumps scored", f"{preds['asset_id'].nunique():,}")
k2.metric("Pump-hours", f"{len(preds):,}",
          help=f"{int(preds['risk'].notna().sum()):,} of them with the pump running. Stopped hours are not scored.")
n_alert = int(summary["Status"].isin(["Alert", "Critical"]).sum())
k3.metric("Pumps in Alert or Critical now", f"{n_alert}")
k4.metric("Alerts raised over the period", f"{len(raises)}",
          help=f"Same alert logic as production: risk >= {THR:.0%} for {CFG.alerts.debounce_h} h raises an alert; "
               f"re-issued every {CFG.alerts.realert_h // 24} days while active.")


def _status_style(col: pd.Series) -> list[str]:
    return [f"background-color: {C.STATUS_COLORS.get(v, '#888')}; color: #111; font-weight: 600" for v in col]


st.dataframe(
    summary.style.apply(_status_style, subset=["Status"]),
    hide_index=True, width="stretch",
    column_config={
        "Pump": st.column_config.TextColumn(width=80),
        "Status": st.column_config.TextColumn(width=80),
        "Risk (7 days)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%", width=105,
                                                         help="Sentinel hybrid probability of failure within 7 days"),
        "RUL P50": st.column_config.TextColumn(width=85),
        "RUL P10-P90": st.column_config.TextColumn(width=145),
        "Likely mode": st.column_config.TextColumn(width=110),
        "Top drivers": st.column_config.TextColumn(
            width=300, help="The two signals furthest from the pump's healthy baseline; the chart below shows more."),
        "Alerts raised": st.column_config.NumberColumn(width=95),
        "Last reading": (st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm", width=140) if has_ts
                         else st.column_config.NumberColumn("Last hour", format="%d", width=140)),
    },
)
st.caption("Status is at the last hour in the file; risk, remaining life and mode are from each pump's last running "
           "hour. Remaining life and mode are shown once risk reaches Watch (half the alert threshold).")

# ---- per-pump detail ------------------------------------------------------------------------
pumps = summary["Pump"].tolist()
pump = st.selectbox("Pump detail", pumps, index=0, key="sd_pump")
g = preds[preds["asset_id"] == pump]
x = g["timestamp"] if has_ts else g["hour"]
d1, d2 = st.columns([1.7, 1])
fig = go.Figure()
for r in raises[raises["asset_id"] == pump].itertuples():
    x0 = g.loc[g["hour"] == r.start_hour, "timestamp"].iat[0] if has_ts else r.start_hour
    x1 = g.loc[g["hour"] == r.end_hour, "timestamp"].iat[0] if has_ts else r.end_hour
    fig.add_vrect(x0=x0, x1=x1, fillcolor=C.STATUS_COLORS["Alert"], opacity=0.12, line_width=0, layer="below")
fig.add_scatter(x=x, y=g["risk_gbm"], mode="lines", name="Gradient boosting only",
                line=dict(color=C.METHOD_COLORS["ml"], width=1), opacity=0.6,
                hovertemplate="%{x}<br>GBM risk %{y:.1%}<extra></extra>")
fig.add_scatter(x=x, y=g["risk"], mode="lines", name="Sentinel risk", line=dict(color=C.ACCENT, width=2.2),
                hovertemplate="%{x}<br>risk %{y:.1%}<extra></extra>")
fig.add_hline(y=THR, line=dict(color="#e5e7eb", dash="dash", width=1), annotation_text=f"alert {THR:.0%}",
              annotation_position="top left")
fig.add_hline(y=THR / 2, line=dict(color=C.STATUS_COLORS["Watch"], dash="dot", width=1), annotation_text="watch",
              annotation_position="bottom left")
fig.update_layout(title=f"{pump}: failure risk over time (shaded: alert active)",
                  yaxis=dict(range=[0, 1.02], tickformat=".0%", title="P(failure within 7 days)"),
                  xaxis_title=None if has_ts else "Hour")
d1.plotly_chart(C.style_fig(fig, height=380), width="stretch")

run = g[g["risk"].notna()]
with d2:
    if run.empty:
        st.info("This pump never ran in the file, so there is nothing to score.")
    else:
        last = run.iloc[-1]
        z = pd.Series({s: float(last.get(f"z_{s}", np.nan)) for s in SIGNALS}).dropna()
        z = z.reindex(z.abs().sort_values(ascending=False).index[:8])
        st.plotly_chart(C.zscore_figure(z, "What is unusual now", height=380), width="stretch")

if not run.empty and str(g.iloc[-1]["status"]) in ("Alert", "Critical", "Watch"):
    last = run.iloc[-1]
    rec = C.recommendation(last["pred_mode"], str(g.iloc[-1]["status"]), float(last["rul_p10"]))
    actions = "".join(f"<li>{html.escape(a)}</li>" for a in rec["actions"])
    st.markdown(
        f"""<div class="sp-card" style="border-left: 4px solid {C.STATUS_COLORS.get(g.iloc[-1]['status'], '#888')};">
        <b>Recommendation for {html.escape(pump)}</b> &nbsp;{C.status_badge(str(g.iloc[-1]['status']))}<br>
        <b>Likely failure mode:</b> {html.escape(rec['mode_label'])} · <b>When:</b> {html.escape(rec['when'])}<br>
        <span class="sp-muted"><b>Evidence:</b> {html.escape(rec['evidence'])}</span>
        <ol style="margin:8px 0 0 0;">{actions}</ol></div>""",
        unsafe_allow_html=True,
    )

st.download_button(
    "Download scored data (CSV)", preds.to_csv(index=False).encode("utf-8"),
    file_name=f"sentinel_scored_{Path(label).stem if source == 'upload' else 'sample'}.csv", mime="text/csv",
    icon=":material/download:", key="sd_download_scored",
    help="Every input hour with risk, risk_gbm, RUL P10/P50/P90, mode probabilities, anomaly scores, "
         "driver z-scores and status.",
)
