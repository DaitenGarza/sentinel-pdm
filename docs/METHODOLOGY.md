# Methodology

This document explains how Sentinel PdM works and why each design decision was made: the
physics-informed fleet simulator, causal data cleaning, feature engineering, the models, and an
evaluation protocol that measures what a maintenance organisation actually experiences (caught
failures, warning time, nuisance alarms and cost). Every number in the results section is copied
from a file the code writes: the single-fleet results from `artifacts/metrics.json` of the reference
run (48 pumps x 8,760 h, seed 42; `python -m sentinel_pdm train`), the five-fleet robustness study
from `reports/robustness.json` (`python -m sentinel_pdm robustness`) and the real-data benchmark
from `artifacts/ai4i_metrics.json` (`python -m sentinel_pdm ai4i`). `python -m sentinel_pdm report`
turns all three into [reports/RESULTS.md](../reports/RESULTS.md).

Module-level contracts (APIs, file schemas) live in [ARCHITECTURE.md](ARCHITECTURE.md); intended
use and limitations are summarised in the [model card](MODEL_CARD.md); the per-fleet robustness
tables are in [reports/ROBUSTNESS.md](../reports/ROBUSTNESS.md).

**Contents**

1. [Problem framing](#1-problem-framing)
2. [The simulated fleet](#2-the-simulated-fleet)
3. [Causal data cleaning](#3-causal-data-cleaning)
4. [Feature engineering](#4-feature-engineering)
5. [Labels and right-censoring](#5-labels-and-right-censoring)
6. [Models](#6-models)
7. [Evaluation design](#7-evaluation-design)
8. [Results](#8-results)
9. [Limitations and real plant data](#9-limitations-and-real-plant-data)
10. [Next steps](#10-next-steps)

---

## 1. Problem framing

### Why unplanned downtime matters

Centrifugal pumps are the most numerous rotating machines in a refinery or chemical plant, and most
of them are critical to throughput: a reflux pump trip can take a distillation column off-spec, a
boiler-feedwater pump trip can take down a steam header. When a pump runs to failure the plant pays
three times:

* **lost production** while the unit is down or rate-cut (the cost model uses $2,500 per hour);
* **secondary damage**: a bearing that seizes takes the shaft, mechanical seal and sometimes the
  impeller with it, and parts are expedited (unplanned repair $42,000 with 48 h of downtime);
* **safety and environmental exposure**: seal failures release process fluid.

A repair planned a week ahead is a different job: parts are staged, the spare pump is lined up, the
work is scheduled into a planned window (planned repair $14,000 with 8 h of downtime). With these
illustrative Gulf-Coast numbers an unplanned failure costs $162,000 against $34,000 planned, so every
failure converted from unplanned to planned is worth about $128,000. A false alarm is not free
either: a vibration analyst and a mechanic inspect a healthy pump ($1,800), and, worse, every
nuisance alert teaches operators to ignore the next one.

### The P-F curve

Reliability-centred maintenance describes a developing failure with the **P-F curve**. Condition
declines slowly at first and then accelerates. **P** (potential failure) is the first point at which
the defect can be detected by an inspection; **F** (functional failure) is when the pump can no
longer do its job. The **P-F interval** between them is the window in which condition monitoring has
to act, and the useful part of it ends some hours before F, when it is too late to plan.

```
 condition
    ^
    |---------------.___
    |                    `--.__   P: defect detectable (vibration analyst would confirm it)
    |                          `-.
    |                             `.
    |                               \   <- alert needs to land here: after P, well before F
    |                                \
    |                                 |  F: functional failure (trip, seizure, seal leak)
    +-----------------------------------------------> time
            healthy          |<---- P-F interval ---->|
```

Sentinel PdM therefore answers three questions for every running pump, every hour:

| Question | Output | Used for |
|---|---|---|
| Will this pump fail within the next 7 days? | calibrated probability `risk` | alerts, fleet status |
| If a failure is developing, how long is left? | remaining useful life P10 / P50 / P90 (h) | scheduling the repair window |
| What is failing? | probabilities of bearing / misalignment / cavitation / winding | the work order: parts, crafts, checks |

Success is measured the way a maintenance organisation experiences it: failures caught with usable
warning, warning time, nuisance alarms per pump-year, and maintenance cost (Section 7).

---

## 2. The simulated fleet

Source: `src/sentinel_pdm/simulator.py`.

### Why simulate

Real run-to-failure histories from a plant historian are proprietary, and public datasets rarely
contain the whole picture a reliability engineer works with: multi-sensor trends *and* a
maintenance log *and* failure modes for a whole fleet. A simulator also provides what real data
never does: ground truth for every hour (the true failure mode and severity, and the P point of each
failure), which makes it possible to score an alert against the P-F interval instead of guessing.

The simulator is deliberately not easy. It reproduces the things that make real predictive
maintenance hard: operating-point changes that look like faults, failures that present differently
each time, failures with no precursor, instrument problems that look like faults, process outages,
and preventive maintenance that hides failures that would have happened. To check that the results
are not the luck of one simulated fleet, the whole experiment is repeated on five independent fleets
(Section 8.5); to check that the approach is not tuned to its own simulator, the same ideas are also
benchmarked on the independent UCI AI4I 2020 dataset (Section 8.6).

### 2.1 Fleet and process units

48 centrifugal pumps in eight process units of a fictitious Gulf-Coast plant (U-100 Feed
Preparation, U-200 Reaction, U-300 Separation, U-400 Cooling Water, U-500 Product Transfer, U-600
Wastewater, U-700 Steam & Condensate, U-800 Tank Farm). Each unit has three services with an A/B
pump pair (for example `P-302A` / `P-302B`, the U-300 column bottoms pumps). Each pump gets a rated
flow (120-650 m³/h), rated head (6-18 bar), rated motor current (60-220 A; motor kW derived for a
480 V motor), criticality A/B/C (30 / 50 / 20 %), an install year, and a variable-speed drive (VFD)
with 25 % probability. One third of the pumps (16) are held out entirely for testing; the other 32
are the training fleet. One year of hourly data is simulated per pump (420,480 pump-hours).

### 2.2 Operating profile

* **Process load**: a per-pump base load (62-92 %), step changes about every two weeks (24 h plus
  an exponential gap with a 350 h mean; step size N(0, 0.06), cumulative shift clipped to -25 % /
  +15 %), a ±4 % daily cycle and slow AR(1) noise; clipped to 35-105 %.
* **Speed**: VFD pumps run at 3,560 x clip(0.55 + 0.45 x load, 0.5, 1.0) rpm; fixed-speed pumps at
  3,560 rpm minus a small load-dependent slip.
* **Pump curve**: differential head follows the affinity laws and a parabolic pump curve,
  `head = H_rated x n² x (1.12 - 0.12 x (load / n)²)` with n the speed ratio. On a fixed-speed pump
  more throughput moves the operating point right along the curve, so head falls while flow and
  current rise; on a VFD pump speed rises with load, so head rises too. Either way, a change in
  operating point changes almost every process signal at once without anything being wrong.
* **Ambient temperature**: Gulf-Coast seasonal cycle (about 12.5-29.5 °C monthly means), ±4.5 °C
  diurnal cycle and weather noise; bearing and winding temperatures follow it.
* **Thermal lag**: bearing and winding RTDs respond through first-order lags (3 h and 2 h).
* **Process outages**: on average about six unplanned stops a year per pump (4-36 h each) when the
  pump is simply not running; readings collapse to idle values and must not be scored.

### 2.3 Degradation and the severity curve

The life of each pump is a sequence of renewal cycles. In each cycle the time to degradation onset is
Weibull distributed (shape 1.6, i.e. wear-out behaviour; scale 2,600 h x a pump-specific factor
0.7-1.3). At onset a failure mode is drawn (bearing 42 %, misalignment 22 %, cavitation 20 %,
winding 16 % of predictable failures) and the defect grows over a mode-specific duration *D* with
severity

$$
s(\tau) = \frac{e^{k\tau} - 1}{e^{k} - 1}, \qquad \tau = \frac{t - t_\text{onset}}{D} \in [0, 1]
$$

which is the P-F curve: slow onset, accelerating towards functional failure. Larger *k* means a
longer quiet period and a steeper end. A slow multiplicative random walk (±4 % scale) makes every
trajectory slightly different, and the effective severity is scaled by a per-event factor 0.4-1.3,
because some failures announce themselves weakly. The **P point** of each failure (`p_hour` in the
maintenance log) is the first hour the effective severity reaches 0.05, the level at which an
inspection triggered by an alert would confirm the defect.

| Mode | k | Duration D (h) | Median P-F interval in this fleet | Physical signature |
|---|---:|---|---:|---|
| Rolling-element bearing wear | 3.5 | 300-1,100 | 450 h | high-frequency (demodulated) band and kurtosis first; kurtosis peaks then falls as damage spreads; then overall vibration and, late and steeply, bearing temperature (∝ s³); slight current rise |
| Shaft / coupling misalignment | 0.9 | 500-1,600 | 677 h | 2x running-speed vibration (and some 1x), bearing temperature, motor current; slow, almost linear growth |
| Suction cavitation / impeller damage | 2.0 | 150-600 | 282 h | suction pressure falls, discharge pressure pulsates (noise grows 5x), flow and head drop, broadband vibration and high-frequency band rise |
| Stator winding deterioration | 2.5 | 250-900 | 543 h | winding temperature rises beyond what load explains (∝ s^1.5), growing phase-current imbalance, slight 2x vibration |
| Sudden (no precursor) | - | 0 | 0 h | nothing: process upset, foreign object, operator error |

### 2.4 Per-event symptom variability and secondary damage

Two mechanisms stop the models from learning a lookup table of clean signatures:

* **Per-event presentation.** Every degradation event draws its own gain (0.35-1.5) for each of
  eight symptom channels: high-frequency band, kurtosis, broadband vibration, bearing temperature,
  shaft (1x/2x) vibration, motor current, winding temperature and hydraulics. A bearing failure may
  never show a temperature rise; a cavitation event may barely move the flow.
* **Secondary damage (cross-talk).** Real failures rarely have one clean signature: misalignment
  overloads the bearings, cavitation shakes the coupling, a hot bearing heats the motor end. Each
  pump has its own 4 x 4 cross-talk matrix (each entry non-zero with probability 0.6, value
  0.15-0.5) that adds a share of the primary mode's severity to the other modes' symptoms.

The healthy baseline also changes from one maintenance cycle to the next (±12 % "new parts"
variation in vibration levels), so a model cannot memorise a pump's absolute levels.

### 2.5 Sudden failures

12 % of failures are drawn as *sudden*: no degradation at all, P = F. They model process upsets,
foreign-object damage or operating errors. No sensor-based method can catch them, and the evaluation
says so explicitly: they count as failures (so detection rates are not inflated) and are reported
separately as "not predictable".

### 2.6 Maintenance: preventive renewal and repairs

* **Preventive maintenance (PM)** runs roughly every six months (4,380 h ± 10 %; the first PM is
  staggered) and takes the pump out for 8-16 h. PM renews the pump. If a degradation had already
  passed its P point, the PM crew finds the defect and the work order records the mode
  (12 of the 93 PM work orders in the reference run did). These degradations never become failures:
  they are **right-censored**, which matters for labels (Section 5) and for alert scoring (an alert
  before such a PM is confirmed, not false).
* **Failures** are followed by 24-96 h of repair downtime, logged as a corrective work order.
* The resulting maintenance log (`data/events.csv`) looks like a CMMS export: work order id, pump,
  type, start and end hour, mode found, onset and P hour (simulation only), predictable flag.

### 2.7 Instrument faults

Every tag can fail the way field instruments do:

* **spikes**: about 9 per tag per year (x2-4 for most tags, +10-30 °C for temperatures);
* **stuck / flatlined tags**: on average one episode per tag every ~7 years, 24-240 h long (the
  value freezes at the last reading);
* **dropouts**: about 7 per tag per year, geometric lengths of a few hours (5 h on average);
* **historian / network outages**: about 2-3 a year, 1-7 h with every tag missing;
* **slow RTD drift** on 12 % of pumps: the bearing temperature reading ramps up by 6-15 °C over
  500-2,000 h. It looks like a hot bearing and is an instrument problem; a single-tag alarm limit
  would eventually fire on it.

### 2.8 What the reference fleet contains

| Item | Value |
|---|---|
| Pumps | 48 (32 training, 16 held-out test) |
| History | 8,760 h per pump, 420,480 pump-hours |
| Failures | 96: bearing 38, cavitation 19, winding 14, misalignment 13, sudden 12 |
| Test-fleet failures | 31 (27 predictable, 4 sudden) |
| Preventive maintenance work orders | 93 (12 found a developing defect) |
| Pumps running | 97.2 % of hours on average |
| Running sensor values missing | 0.52 % |

---

## 3. Causal data cleaning

Source: `src/sentinel_pdm/quality.py`.

Every rule looks only at the current and **past** values of a tag, so exactly the same cleaning can
run online, hour by hour. A centred median filter or a back-fill would look better in a notebook and
silently leak the future into a back-test. Rules, per pump and per tag:

1. **Pump not running**: readings are meaningless (idle vibration, ambient temperatures) and are set
   to missing. Features and scores are only defined while the pump runs.
2. **Physically impossible values** (negative vibration, current, flow) are removed.
3. **Stuck tags**: once the same value has been reported for 6 consecutive hours it is removed from
   the sixth hour on, until the value changes (an hourly average of a live transmitter never repeats
   exactly).
4. **Spikes**: rate-of-change limits against a trailing median. Vibration, pressure, current and
   flow are compared with the median of the previous 24 h (ratio limits of 1.6-2.5 depending on the
   tag). Temperatures follow the daily cycle, so they are compared with the median of the previous
   3 h with an absolute limit (8 °C bearing, 9 °C winding): thermal mass makes a 10 °C jump in one
   hour implausible. The first 6 hours after a restart are exempt, because a motor legitimately heats
   up by 20 °C in its first hour.
5. **Short gaps** of up to 6 h are forward-filled with the last good value; longer gaps stay
   missing (the gradient-boosting models handle missing values natively).

In the reference run cleaning removed 4,792 spikes and 12,148 stuck-tag hours. A per-pump
data-quality summary (`data/quality.csv`) is shown in the dashboard because an engineer should know
when a prediction rests on poor data.

---

## 4. Feature engineering

Source: `src/sentinel_pdm/features.py`.

### 4.1 Signals

17 signals per pump-hour:

* **12 sensor tags**: overall vibration RMS (mm/s), 1x and 2x running-speed vibration (mm/s),
  high-frequency demodulated bearing band (g), vibration kurtosis, bearing RTD (°C), winding RTD
  (°C), motor current (A), phase-current imbalance (%), suction and discharge pressure (bar), flow
  (m³/h);
* **4 physics-derived signals**, because they are what a rotating-equipment engineer looks at:
  differential head (discharge - suction), bearing temperature rise over ambient, winding
  temperature rise over ambient, and a hydraulic-efficiency proxy flow x head / current (hydraulic
  power per unit of electrical input);
* **process load** (%), tracked like a sensor so every model sees the operating point (4.5).

### 4.2 Statistics

For each signal six statistics are computed over trailing windows:

| Suffix | Meaning | Why |
|---|---|---|
| `_m24` | 24 h mean | current level, daily cycle averaged out |
| `_sd24` | 24 h standard deviation | instability: pulsation, intermittent impacting |
| `_m168` | 7-day mean | slow level |
| `_sl24` | 24 h OLS trend slope (per day) | fast change |
| `_sl168` | 7-day OLS trend slope (per day) | the degradation trend itself |
| `_dev` | 24 h mean vs. post-maintenance baseline | how far from *this* pump's normal |

plus five context features (current load, ambient temperature, speed, hours since last maintenance,
running hours since last maintenance): 17 x 6 + 5 = **107 features**. Windows are NaN-aware (a
24 h window needs at least 6 valid hours, a 7-day window at least 24) and computed with cumulative
sums, so the whole fleet is featurised in about 2.5 s.

**Strictly causal.** Every feature at hour *t* uses data from hours <= *t* only: trailing windows,
no centring, no back-filling. The project's test suite perturbs future readings and asserts that
past features do not move.

### 4.3 Maintenance-aware windows

Rolling windows restart at every work order from the maintenance log. Without this, a freshly
repaired pump would inherit the pre-failure trend for a week and the models would see a "failing"
pump that has just been fixed. `hours_since_maint` and `run_hours_since_maint` also let the models
learn the wear-out behaviour of the Weibull life (they are undefined before the first logged work
order because the install date of the bearing is unknown, as in a real CMMS).

### 4.4 Self-referenced baselines and scale-free normalisation

A 2.0 mm/s vibration is normal for one pump and a warning sign for another. Each pump is therefore
compared with **its own** healthy baseline: the expanding median of the first 168 valid running
hours after its last maintenance, then frozen for the rest of the cycle. The `_dev` feature is the
absolute deviation for temperatures and kurtosis (°C and kurtosis units carry meaning) and the
relative deviation (ratio - 1) for everything else.

For scale-dependent signals (motor current, suction and discharge pressure, flow, head and the
efficiency proxy) *every* statistic is divided by the pump's own baseline. A 600 m³/h pump and a
150 m³/h pump then speak the same language ("flow down 8 %, falling 1 % per day"), and the models
learn patterns instead of memorising individual pumps, which is what makes them transfer to pumps
they have never seen.

### 4.5 Operating-point awareness: the P-302A finding

An early version of the feature set compared every signal with its baseline but did not track the
process load, and it raised alerts on a healthy pump. On the held-out pump **P-302A** (U-300 column
bottoms pump, VFD-driven) process load sat around 77 % during the January baseline week and between
85 % and 99 % (monthly medians) for the rest of the year; every work order restarts the baseline, but
load kept stepping after each restart. Along the pump curve flow, motor current and differential head
all moved away from the baseline together, and because a VFD pump speeds up with load, head moved
most: its deviation from baseline reached 6-10 robust standard deviations of the healthy-fleet
spread. A baseline-only model reads that as changed hydraulics, the signature of a hydraulic fault
such as impeller damage.

The fix is the physics: process load gets the same six statistics as every sensor, including
`load_pct_dev`, so a model can see that head, flow and current moved *because* load moved. The
episode is also the clearest argument for the hybrid (Section 6.6). Even with load features,
gradient boosting alone still raised 4 of its 5 test-fleet false alarms on P-302A in the reference
run. Three came while the pump ran at 90-97 % load; the fourth came 131 h before the P point of the
pump's later bearing failure, too early for an inspection to confirm anything. At all four, the MSPC
score sat below its 99 % control limit: MSPC models the normal correlation between load, head, flow
and current, so a coherent move along the pump curve stays inside its limits, and the hybrid, which
needs both to agree, raised none.

---

## 5. Labels and right-censoring

Source: `src/sentinel_pdm/labels.py`. Targets come only from the maintenance log, as they would in a
plant (the simulator's ground-truth columns are never model inputs).

For every pump-hour with time to the next logged failure `ttf`:

* **Risk label** (horizon *H* = 168 h): `1` if `ttf <= H`; `0` if not; `-1` (unknown) if no failure
  is observed and the record ends within *H* hours, because we cannot know whether the pump failed
  just after the record ends. Unknown rows are excluded from training and evaluation. 4.0 % of the
  training fleet's running hours are positive (3.8 % of the test fleet's).
* **RUL target**: `min(ttf, 720 h)`, the piecewise-linear RUL common in prognostics; hours far from
  any failure get the 30-day cap, hours without enough look-ahead are undefined.
* **Next mode** and **next predictable** describe the next failure (for diagnosis and for excluding
  sudden failures from RUL training).

**Right-censoring by preventive maintenance.** A degradation that PM fixed never produces a
failure, so its hours are labelled 0. Statistically this is right-censoring (we never learn when
the pump would have failed); operationally it is exactly what the plant observes. Training treats
these hours as "no failure within 7 days", which is conservative, and the evaluation credits an
alert raised during such a degradation as `pm_confirmed` rather than calling it false (Section 7.3).

---

## 6. Models

Source: `src/sentinel_pdm/models.py`. Everything is scikit-learn on a CPU; the full pipeline
(simulation, cleaning, features, training, scoring, evaluation and permutation importance) runs in
about two minutes on a laptop.

### 6.1 Validation design: grouped cross-validation by pump

Hourly rows of the same pump are highly correlated, so random K-fold would put near-identical hours
of the same degradation in train and validation and wildly overstate performance. Every model is
trained with **4-fold `GroupKFold` by pump** on the 32 training pumps. Each training pump is scored
by models that never saw it; these **out-of-fold (OOF)** scores are what the probability
calibration, the conformal RUL intervals, the hybrid fusion and the alert thresholds are tuned on.
Nothing at all is fitted on the 16 test pumps.

### 6.2 Gradient-boosting failure risk (supervised)

`HistGradientBoostingClassifier` predicts P(failure within 7 days) from the 107 features. It
handles missing values natively (instrument dropouts are routine) and trains in seconds. Settings,
chosen by grouped cross-validation on the training fleet only and deliberately regularised because
32 pumps is a small number of independent units:

| Setting | Value |
|---|---|
| Iterations / learning rate | 250 / 0.05 (no early stopping) |
| Max leaf nodes / min samples per leaf | 15 / 200 |
| Feature subsampling per split | 0.5 |
| L2 regularisation | 2.0 |
| Training rows | running hours with a known label, every 3rd hour (hourly rows are highly autocorrelated) |

The raw ensemble score is calibrated to a probability by **isotonic regression** fitted on the OOF
scores (a tiny 1e-6 x raw term breaks isotonic ties so the ranking, and therefore ROC/PR and alert
order, is preserved).

### 6.3 MSPC: multivariate statistical process control (unsupervised)

MSPC is the classic chemical-plant monitoring technique and needs only examples of healthy
operation. Inputs are the 17 `_dev` features (each signal's deviation from its own baseline),
missing values set to 0 (= at baseline). Fitted per fold on healthy rows (at least 30 days before any
logged failure):

1. **Standardise**: `z = (x - μ) / σ` per feature.
2. **PCA** keeping the smallest number of components *K* that explains **90 %** of the variance,
   with loadings `P` (17 x K), scores `t = Pᵀz` and component variances λ₁..λ_K.
3. **Hotelling's T²**, distance *inside* the model plane (normal variation pushed to an extreme):

   $$T^2 = \sum_{k=1}^{K} \frac{t_k^2}{\lambda_k}$$

4. **SPE (Q statistic)**, distance *from* the model plane (a broken correlation, such as head
   rising without load rising):

   $$\mathrm{SPE} = \lVert z - \hat z \rVert^2, \qquad \hat z = P\,P^\mathsf{T} z$$

5. **Control limits**: the empirical 99th percentiles of T² and SPE on the healthy training rows.
   Empirical limits are used instead of the F and χ² approximations because the features are
   heavy-tailed, not Gaussian.

The anomaly score is `max(T² / T²_lim, SPE / SPE_lim)`, so 1.0 sits on the 99 % control limit.
Because MSPC models the *correlation structure* of healthy operation, it is very robust to
operating-point changes. A 99 % limit is still exceeded by 1 % of healthy hours (about 88 h per
pump-year), which is why the alert threshold chosen under the alarm budget is several times the
limit (Section 7.4). The per-variable squared residuals (SPE contributions) are available for
explanation.

### 6.4 Isolation Forest (unsupervised comparison)

A generic anomaly detector on the same 17 deviation features: 150 trees, 2,048 samples per tree,
fitted per fold on up to 40,000 healthy rows; the score is normalised so 1.0 is the 99th percentile
of healthy data. It is kept as a comparison: a popular choice that does not exploit the structure of
the problem.

### 6.5 Conventional alarm limits (what the plant does today)

The baseline every reliability engineer recognises: an alarm when overall vibration exceeds
2.8 mm/s (the ISO 10816-3 zone B/C boundary for group 2 machines on rigid foundations), the bearing
RTD exceeds 82 °C, or the winding RTD exceeds 130 °C. The score is the largest of the three ratios
reading / limit, so 1.0 is the alarm.

### 6.6 Sentinel hybrid: fusing supervised risk and MSPC

The two model families fail differently. Gradient boosting knows what pre-failure patterns look
like but can over-react to unusual-but-healthy operation it has not seen (P-302A). MSPC is robust
to operating-point changes but fires on any broken correlation, including instrument problems. Each
family catches almost every developing failure; their false alarms come from different causes. The
hybrid therefore requires **both** to agree:

$$
u = \min\big(F_\text{GBM}(r),\; F_\text{MSPC}(q)\big), \qquad \text{risk} = g(u)
$$

where *r* is the raw gradient-boosting score, *q* the MSPC score, `F` the empirical CDF (percentile
rank) of each score over the OOF training rows, and *g* an isotonic calibration of *u* against the
OOF labels. Percentile ranks put two incomparable scales on a common footing; the minimum is a
soft AND; isotonic calibration turns the result back into a probability that "fails within 7 days"
can be read from.

Evidence that the fusion is worth it:

* **Held-out test fleet**: gradient boosting alone caught 26/31 failures with 5 false alarms; the
  hybrid caught 27/31 with **0**.
* **Cross-validation on the training fleet**: to catch 56 of 65 failures the hybrid needed 0 false
  alarms; gradient boosting alone needed **60** (1.9 per pump-year, almost four times the alarm
  budget), and its threshold had to be raised until it caught only 55 to fit inside the budget.
  MSPC alone had 6 false alarms in cross-validation.
* On this particular test fleet MSPC alone matched the hybrid (27/31, 0 false alarms) with a longer
  median lead (266 h versus 218 h). Requiring agreement can cost warning time: the hybrid can only
  fire once the slower of the two scores has risen. The supervised half is what provides a
  calibrated probability.
* **Five independent fleets** (Section 8.5) settle the comparison. Pooled over 150 held-out failures
  and 80 pump-years, the hybrid and MSPC alone each caught all 124 predictable failures and gradient
  boosting alone 120, but the hybrid did it with **2** false alarms against 15 for MSPC alone and 10
  for gradient boosting alone. Its median lead there (281 ± 47 h) was longer than either
  component's (MSPC 236 ± 14 h, gradient boosting 239 ± 32 h), so the warning-time cost seen on the
  reference fleet is not systematic.

### 6.7 Cross-validation ensembles

The usual recipe is to tune on cross-validation and then refit one final model on all the training
data. Here that silently breaks the alarm budget. Alert thresholds, the isotonic calibration and the
conformal offsets are all tuned on the OOF score distribution, but a model refitted on all 32 pumps
(a third more data than any fold model) produces a different score distribution, so a threshold
tuned on the OOF scores no longer means the same thing. `python scripts/refit_vs_ensemble.py`
measures the size of the effect on the reference run: a single gradient-boosting model refitted on
all 32 training pumps, put through the same OOF-fitted calibration and the same threshold, caught the
same 26 test-fleet failures as the ensemble but raised **11 false alarms instead of 5**.

The fix is to keep the K fold models as a **cross-validation ensemble** and average them to score new
pumps (risk, RUL, diagnosis, MSPC and Isolation Forest alike). Production then uses exactly the
models whose score distribution the thresholds, calibration and conformal offsets were tuned on, and
averaging four models also reduces variance.

### 6.8 Remaining useful life: conditional quantile regression with Mondrian CQR

RUL is modelled **conditionally**: *given* that a failure is developing, how long is left? Three
quantile-loss gradient-boosting regressors (α = 0.1, 0.5, 0.9) are trained on hours whose next
failure is predictable and at most 720 h away, with target `ttf`. (An unconditional RUL model spends
its capacity predicting "more than 30 days" for healthy pumps.) The quantiles are sorted so they
cannot cross, and clipped to [0, 720] h.

Raw quantile regressors are usually over-confident, so the P10-P90 interval is widened with
**normalised conformalised quantile regression** (CQR) on the OOF predictions. For each calibration
row the nonconformity score is

$$
E_i = \frac{\max\big(\hat q_{10}(x_i) - y_i,\; y_i - \hat q_{90}(x_i)\big)}{\max\big(\hat q_{90}(x_i) - \hat q_{10}(x_i),\; 1\,\text{h}\big)}
$$

and the offset is the finite-sample-corrected quantile
`Q = quantile(E, ceil((n + 1) x 0.8) / n)` for the 80 % target coverage. The interval becomes
`[q10 - Q·w, q90 + Q·w]` with `w = q90 - q10`. Normalising by the width makes the widening
proportional to the model's own uncertainty, and a **Mondrian** variant computes a separate `Q` for
each band of predicted median RUL (0-72 h, 72-168 h, 168-336 h, over 336 h; bands with fewer than 30
calibration rows use the pooled value), because errors near failure behave differently from errors a
month out.

### 6.9 Failure-mode diagnosis

A balanced-class `HistGradientBoostingClassifier` (same settings) is trained on the pre-failure
hours (label 1) of predictable failures, with the logged failure mode as the target, and averaged
across the fold models. For a failure-level diagnosis the hourly mode probabilities are averaged over
the 7 days before the failure and the largest wins. The dashboard turns the mode into a maintenance
recommendation (for example: envelope spectrum and lubrication check for bearings, laser alignment
and soft-foot check for misalignment, strainer and NPSH check for cavitation, thermography and
insulation-resistance test for windings).

### 6.10 Explanations and status

* **Drivers**: for each signal, a robust z-score of its `_dev` feature against healthy training data
  (median and IQR / 1.349). The largest |z| >= 2 become plain-English drivers ("High-frequency
  bearing band is 4.2σ above its healthy baseline").
* **Permutation importance** of the risk ensemble on test-fleet hours (drop in PR-AUC).
* **Status** per pump-hour: *Critical* (risk above the alert threshold and median RUL <= 72 h),
  *Alert* (risk above threshold), *Watch* (risk above half the threshold, or MSPC beyond 1.5x its
  limit), *Normal*, *Offline*.

---

## 7. Evaluation design

Source: `src/sentinel_pdm/evaluation.py` and `pipeline.py`.

### 7.1 Row-level metrics

ROC-AUC, PR-AUC (average precision) and, for the probabilistic scores, the Brier score, on every
running test-fleet hour with a known label. With 3.8 % positive hours PR-AUC is the more honest of
the two ranking metrics. A second set excludes the 7-day windows before sudden failures, which no
sensor can rank. Row metrics say how well hours are ranked; they do not say whether a planner got a
usable warning, so they are not the headline.

### 7.2 From scores to alerts

Per pump, mimicking a DCS / CMMS notification:

* an alert is **raised** once the score stays at or above the threshold for **3 h** (debounce);
* it **clears** after the score has been below the threshold for **24 h**;
* while still active it **re-alerts every 7 days** (168 h), each time as a new work request.

The re-alert rule matters: without it, a score that is permanently high would get credit for every
failure on that pump at the cost of a single false alarm. Each raise is one work request, so each
raise is scored.

### 7.3 Matching alerts to failures: the P-F interval

Each raise is classified using the P point recorded for each failure:

| Outcome | Rule | Meaning |
|---|---|---|
| `tp` | raised in [P, F - 12 h] | inspection finds the defect with time to plan |
| `late` | raised in (F - 12 h, F) | too late to plan; does not count as a catch |
| `pm_confirmed` | raised in [P, PM] of a degradation that PM fixed | the defect was real; PM got there first |
| `unresolved` | unmatched, raised in the last 14 days of the record | outcome unknown (the pump may be failing now) |
| `false_alarm` | everything else | an inspection that finds nothing |

A failure is **caught** if at least one `tp` raise precedes it; **lead time** = F minus the first
`tp` raise. An alert raised before P is a false alarm even if a failure follows weeks later, because
the inspection it triggers would find nothing. Sudden failures have P = F and can never be caught;
they stay in the denominator. When a maintenance log has no P point (real data), P = F - 14 days is
assumed.

### 7.4 Alarm budget and cost-optimal thresholds

Alarm-management practice (ISA-18.2, EEMUA 191) exists because operators stop trusting a system that
cries wolf. Sentinel PdM imposes an explicit **alarm budget of at most 0.5 false alarms per
pump-year** (one nuisance inspection per pump every two years). Within that budget the threshold of
each method is chosen to **minimise total maintenance cost** on the OOF scores of the training
fleet, sweeping candidate thresholds at 57 quantiles of the score (80th to 99.95th percentile). If no
candidate meets the budget, the one with the fewest false alarms is used. The thresholds are then
frozen and applied unchanged to the test fleet.

Chosen thresholds in the reference run: hybrid 0.188, gradient boosting 0.292 (probabilities), MSPC
4.59 x its 99 % control limit, Isolation Forest 1.10 x its healthy 99th percentile, alarm limits
1.0 (by definition).

### 7.5 Policy economics

Each method is evaluated as a maintenance policy over the test period:

$$
\text{cost} = n_\text{missed}\,(C_u + h_u\,c_\text{prod}) + n_\text{caught}\,(C_p + h_p\,c_\text{prod}) + n_\text{FA}\,C_\text{FA}
$$

with unplanned repair `C_u` = $42,000 and `h_u` = 48 h down, planned repair `C_p` = $14,000 and `h_p` =
8 h down, lost production `c_prod` = $2,500/h and false-alarm inspection `C_FA` = $1,800. The
reference policy is run-to-failure (every failure unplanned). All cost inputs are editable on the
dashboard's Maintenance Economics page, which recomputes every policy with the same function.

---

## 8. Results

Held-out test fleet: 16 pumps (16 pump-years) that no model, calibration or threshold was fitted
on; 31 failures, 27 of them predictable.

### 8.1 Detection, lead time and false alarms

| Method | ROC-AUC | PR-AUC | Caught | Predictable caught | False alarms | Median lead | Cost saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Sentinel hybrid (GBM + MSPC)** | 0.939 | 0.682 | **27/31** | **27/27** | **0** | 218 h (9.1 d) | **68.8 %** |
| Gradient boosting only | 0.955 | 0.647 | 26/31 | 26/27 | 5 | 178 h (7.4 d) | 66.1 % |
| MSPC (PCA T² / SPE) | 0.928 | 0.695 | 27/31 | 27/27 | 0 | 266 h (11.1 d) | 68.8 % |
| Isolation Forest | 0.934 | 0.643 | 25/31 | 25/27 | 2 | 203 h (8.5 d) | 63.6 % |
| Alarm limits (ISO 10816 + RTD) | 0.672 | 0.217 | 8/31 | 8/27 | 0 | 32 h (1.3 d) | 20.4 % |

* The hybrid caught **every predictable failure** with **no false alarms**; the four misses are
  the four sudden failures, which have no precursor. It raised 70 alerts in total: 65 true
  (including weekly re-alerts on long degradations), 4 confirmed by preventive maintenance and 1
  late re-alert.
* **Conventional alarm limits** caught 8 of 31, with a median warning of 32 h, usually too late to
  plan; by the time ISO zone C or an RTD alarm is reached the damage is advanced.
* Excluding the windows before sudden failures, ROC-AUC is 0.989 for the hybrid (PR-AUC 0.758).
  Gradient boosting has the best all-hours ROC-AUC (0.955), yet more false alarms: ranking hours
  well and raising few nuisance alerts are different things, which is why alerts are evaluated as
  events.

### 8.2 Cross-validation (training fleet, out-of-fold)

| Method | Caught | False alarms | Median lead |
|---|---:|---:|---:|
| Sentinel hybrid | 56/65 | 0 | 240 h |
| Gradient boosting only | 55/65 | 5 | 213 h |
| MSPC | 57/65 | 6 | 252 h |
| Isolation Forest | 49/65 | 13 | 215 h |
| Alarm limits | 16/65 | 0 | 58 h |

### 8.3 Economics (test fleet)

| Policy | Unplanned | Planned | False alarms | Downtime | Total cost | Per pump-year |
|---|---:|---:|---:|---:|---:|---:|
| Run to failure | 31 | 0 | 0 | 1,488 h | $5.02M | $314k |
| **Sentinel hybrid** | 4 | 27 | 0 | 408 h | **$1.57M** | $98k |
| Gradient boosting only | 5 | 26 | 5 | 448 h | $1.70M | $106k |
| MSPC | 4 | 27 | 0 | 408 h | $1.57M | $98k |
| Isolation Forest | 6 | 25 | 2 | 488 h | $1.83M | $114k |
| Alarm limits | 23 | 8 | 0 | 1,168 h | $4.00M | $250k |

The hybrid saves **$3.46M (68.8 %)** against run-to-failure on 16 pump-years and cuts failure
downtime by 72.6 %; conventional alarm limits save 20.4 %. The four remaining breakdowns are the
four sudden failures. The economics are dominated by caught failures (each is worth $128,000), so
methods that catch the same failures cost almost the same; false alarms matter more for trust than
for money at $1,800 each.

### 8.4 Remaining useful life and diagnosis

| RUL window (test fleet, predictable failures) | Pump-hours | MAE of P50 | P10-P90 coverage | Median interval width |
|---|---:|---:|---:|---:|
| Last 7 days before failure | 4,486 | 98 h | 71 % | 236 h |
| Last 30 days before failure | 19,239 | 115 h | 77 % | 376 h |

The P10-P90 interval targets 80 % coverage and reaches 77 % over the last 30 days but only 71 % in
the final week on this fleet. Conformal guarantees assume the calibration data and the new data are
exchangeable; the offsets were calibrated on the training pumps, and 27 held-out failures are a
small sample. Across the five robustness fleets the final-week coverage was 83.6 ± 6.2 % (range
77.8-93.3 %); the reference fleet's 71 % is lower than any of them, not the typical case. A planner
should read P10 as "plan before this", not as a guarantee.

Failure-mode diagnosis was correct for **24 of 27** predictable failures (89 %; 86 % of individual
pre-failure hours):

| Actual \ diagnosed | Bearing | Misalignment | Cavitation | Winding |
|---|---:|---:|---:|---:|
| Bearing | 14 | 0 | 0 | 0 |
| Misalignment | 2 | 2 | 0 | 0 |
| Cavitation | 1 | 0 | 5 | 0 |
| Winding | 0 | 0 | 0 | 3 |

Misalignment is the hardest mode: it overloads the bearings, and secondary bearing symptoms can
dominate the picture, so two misalignment failures were diagnosed as bearing failures. In practice a
bearing work order on a misaligned pump would include an alignment check.

The most important features (permutation importance, PR-AUC drop) are week-long trends and
baseline deviations: the 7-day trend of the high-frequency bearing band, the 7-day mean of
phase-current imbalance, the 7-day trend of overall vibration and of suction pressure, and the
baseline deviations of kurtosis, current imbalance and vibration. These are the same indicators a
vibration analyst and an electrical engineer would trend.

### 8.5 Robustness across independent fleets

One test fleet of 16 pumps produces about 30 failures, so one failure more or less moves a detection
rate by 3-4 points and one nuisance alert moves the false-alarm rate by 0.06 per pump-year; a method
can win on one fleet by luck. `python -m sentinel_pdm robustness` therefore re-simulates the plant
with five other seeds (101-105). Each seed is a statistically independent fleet: new pumps, operating
profiles, failure histories, instrument faults and train/test split. The **entire** pipeline is
re-run on each one (cleaning, features, grouped cross-validation, cost-optimal thresholds tuned on
that fleet's training pumps, evaluation on its 16 held-out pumps), in temporary folders that never
touch the reference run. Together the five test fleets hold 80 pump-years and 150 failures (28-32
per fleet), 124 of them predictable. Per-fleet tables are in
[reports/ROBUSTNESS.md](../reports/ROBUSTNESS.md).

| Method | Predictable caught (pooled) | Worst fleet | False alarms (pooled) | False alarms / pump-year | Median lead (h) | Cost saving |
|---|---:|---:|---:|---:|---:|---:|
| **Sentinel hybrid (GBM + MSPC)** | **124/124** | 100 % | **2** | 0.03 ± 0.03 | 281 ± 47 | 65.6 ± 6.6 % |
| Gradient boosting only | 120/124 | 84 % | 10 | 0.12 ± 0.25 | 239 ± 32 | 63.4 ± 8.0 % |
| MSPC (PCA T² / SPE) | 124/124 | 100 % | 15 | 0.19 ± 0.12 | 236 ± 14 | 65.5 ± 6.7 % |
| Isolation Forest | 117/124 | 88 % | 36 | 0.45 ± 0.74 | 195 ± 64 | 61.7 ± 7.8 % |
| Alarm limits (ISO 10816 + RTD) | 43/124 | 31 % | 0 | 0.00 ± 0.00 | 64 ± 15 | 22.8 ± 4.2 % |

Pooled counts weight every failure equally; ± is the standard deviation across the five fleets;
"worst fleet" is the lowest per-fleet detection rate of predictable failures.

* **The hybrid result holds.** It caught every predictable failure on every fleet with 2 false
  alarms in 80 pump-years (never more than one per fleet). Two in 80 pump-years is consistent with a
  true rate of at most 0.08 per pump-year (95 % upper bound, Poisson), well inside the budget of 0.5.
* **The budget is enforced on the training pumps, so the test pumps can overrun it**, which is what
  an overrun in production would look like. Gradient boosting alone did so once (9 false alarms,
  0.56 per pump-year, on seed 103) and Isolation Forest once (28 false alarms, 1.75 per pump-year,
  on seed 104). The hybrid, MSPC and the alarm limits stayed inside it on every fleet.
* **The hybrid's advantage is in alerts, not in ranking hours.** Its row-level ROC-AUC
  (0.909 ± 0.055) and PR-AUC (0.636 ± 0.054) are the same as gradient boosting's
  (0.911 ± 0.058 and 0.650 ± 0.063). This is the argument of Section 7 in numbers: ranking hours
  well and raising few nuisance alerts are different things.
* **Conventional alarm limits** caught 43 of 124 predictable failures (34.7 %) with a median warning
  of 64 ± 15 h, against 281 ± 47 h for the hybrid.
* **Remaining useful life and diagnosis**: in the last 7 days before failure the median-estimate
  error was 78 ± 15 h and the P10-P90 interval covered 83.6 ± 6.2 % of outcomes (target 80 %);
  the failure mode was diagnosed correctly for 90.3 ± 7.2 % of predictable failures.
* **What not to over-read.** The spread across fleets (the hybrid's cost saving ranged from 58.6 %
  to 73.4 %) is as large as several of the differences between the machine-learning methods.
  Differences inside the ± ranges should not be ranked; the gap to conventional alarm limits, and
  the hybrid's false-alarm count against its components', are far outside them.

### 8.6 Real-data benchmark: UCI AI4I 2020

Everything above uses this project's own simulator, so the obvious challenge is whether the approach
only works on data generated to suit it. `python -m sentinel_pdm ai4i` downloads the public AI4I 2020
Predictive Maintenance Dataset (Matzka, 2020; UCI Machine Learning Repository id 601,
DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), CC BY 4.0) and tests the project's central
idea on it: encoding the physics of a failure mechanism as a feature beats a bigger model on raw
tags. AI4I has 10,000 machining runs and 339 failures (3.4 %). It is not plant data either (its
author describes it as synthetic data reflecting real industrial predictive maintenance), but it was
built independently, with documented failure mechanisms, by someone with no stake in this project.

**Protocol.** Target `Machine failure`; the failure-type flags (TWF, HDF, PWF, OSF, RNF), the row id
and the product serial number never reach a model (the code asserts it). Two feature sets: *raw*
(air and process temperature, speed, torque, tool wear, product quality) and *physics-informed*
(the same plus three engineered quantities: temperature rise = process - air temperature,
mechanical power = torque x angular speed, and overstrain = tool wear x torque). The engineered
features say *which* physical quantity matters, never *where* failure starts: the published
thresholds (8.6 K, 3,500 / 9,000 W, 11,000-13,000 min·Nm) are not used, so the models still have to
learn them. Three models of increasing capacity, 3 x repeated stratified 5-fold cross-validation on
identical folds for every model and feature set; mean ± standard deviation over the repeats.
F1 is *cross-fitted*: each fold is scored at the best-F1 threshold of the other folds, which is what
a deployed threshold would achieve.

| Features | Model | ROC-AUC | PR-AUC | F1 (cross-fitted) | Recall at 90 % precision |
|---|---|---:|---:|---:|---:|
| Raw sensors | Logistic regression | 0.895 | 0.451 | 0.478 | 13 % |
| Raw sensors | Random forest | 0.966 | 0.801 | 0.760 | 60 % |
| Raw sensors | Gradient boosting | 0.977 | 0.820 | 0.755 | 60 % |
| Physics-informed | Logistic regression | 0.917 | 0.484 | 0.488 | 14 % |
| Physics-informed | Random forest | 0.977 | **0.900** | **0.885** | **84 %** |
| Physics-informed | Gradient boosting | 0.982 | 0.890 | 0.856 | 82 % |

* Three physics-informed features lift the best PR-AUC from **0.820** (gradient boosting, raw
  sensors) to **0.900** (random forest), and the best cross-fitted F1 from 0.760 to 0.885. Between
  the two tree models the choice of model moves PR-AUC by 0.01-0.02; adding the physics moves it by
  0.07-0.10. Logistic regression gains little (0.451 to 0.484) because the mechanisms are thresholds
  and windows (power fails both below 3,500 W and above 9,000 W), which a linear model cannot
  represent even when given the right quantity.
* At its best-F1 threshold the best physics-informed model catches 83 % of failures at 96 %
  precision, 10.7 false alarms per 10,000 runs; the best raw model catches 73 % at 81 % precision,
  60 false alarms per 10,000 runs. Better features buy both more catches and a sixth of the
  nuisance alarms.
* By failure type (best physics-informed model): heat-dissipation 96 %, power 99 % and overstrain
  99 % of failures caught, tool-wear 7 %. Only 287 of the 339 failures (85 %) follow a mechanism
  visible in a single snapshot; tool-wear and random failures are not predictable from the inputs.
  This is the same lesson as the sudden pump failures, and an honest report states the ceiling.
* Rows are a time series (tool wear accumulates), so random folds put near-identical neighbours in
  train and test. Blocked cross-validation (contiguous 200-row blocks) gives PR-AUC 0.902 against
  0.900 with random folds: the result does not rely on leakage between neighbouring rows.

---

## 9. Limitations and real plant data

**The data is simulated.** The simulator encodes the author's understanding of pump failure physics;
real failures are messier. Absolute numbers (27/31, 0 false alarms, 68.8 % saving) depend on its
assumptions: P-F interval lengths, sensor noise, the share of sudden failures, how strongly
symptoms show. The comparison between methods on identical data is the meaningful part, and the
AI4I benchmark and the multi-seed robustness study exist to keep the claims honest.

**Small numbers.** 31 test failures and 16 pump-years mean wide confidence intervals: 0 false
alarms in 16 pump-years is consistent with a true rate of up to about 0.19 per pump-year (95 %
upper bound, Poisson). The five-fleet study narrows this (2 false alarms in 80 pump-years: at most
about 0.08 per pump-year), but all five fleets come from the same simulator, so it measures the
luck of the draw, not how far the simulator is from a real plant.

**Sudden failures are out of reach.** About one failure in eight is simulated with no precursor
(4 of 31 in the reference test fleet, 26 of 150 across the robustness fleets). No sensor-based method
can warn of them, the system does not claim to, and they stay in every detection denominator.

**One plant archetype.** Every fleet is the same kind of plant: motor-driven centrifugal process
pumps, hourly historian data, the same tag list and the same failure physics.
Nothing here shows how the models behave on other machine types or on a plant whose data looks
different; that needs retraining and re-validation on the new population.

**Costs are illustrative.** The dollar figures are plausible Gulf-Coast process-plant numbers, not
any company's actuals. Savings percentages move with them; the dashboard lets a reader enter their
own.

**What changes with real historian data:**

* **Labels.** CMMS work orders are noisy: failure codes are free text or wrong, dates are when the
  work order was closed, not when the pump failed, and the P point is never recorded. The evaluation
  already falls back to a 14-day credit window when P is unknown; mode labels would need cleaning
  with the reliability engineers, and an alert-feedback loop ("inspection found / did not find a
  defect") becomes the most valuable label source.
* **Censoring.** Pumps are replaced, swapped between services, and PM is more frequent on critical
  pumps. Survival-style handling of censored degradations and pump-level random effects would
  matter more than here.
* **Sampling.** Historians store compressed, exception-reported values, not clean hourly averages;
  vibration is often a route-based monthly reading, not a continuous tag. Features would need
  resampling and the models would need to work with sparse vibration and dense process data.
* **Operating context.** Real plants have campaigns, grade changes, startups and shutdowns with
  transients. Load is a proxy for the operating point; real deployments would add process context
  (product grade, fluid temperature and density, valve positions) as features and in the MSPC model.
* **Fleet heterogeneity.** Different pump types (OH2, BB2, vertical turbine), seals and drivers.
  Self-referenced, scale-free features transfer well, but a large plant would likely group pumps by
  type or add type as a feature.
* **Data drift.** Retraining, threshold re-tuning on recent OOF scores and monitoring of the alarm
  rate per pump-year are required; the alarm budget is the natural monitoring metric.
* **Healthy-data contamination.** MSPC and Isolation Forest train on hours at least 30 days before a
  logged failure, which can include early degradations that PM later fixed. Real data has the same
  issue, only worse.

**What it is not.** A decision-support tool for reliability engineers, not a protection system. It
never replaces machine protection (vibration trips, high-temperature trips), and its outputs must
not be wired into a shutdown or permissive (see the [model card](MODEL_CARD.md)).

---

## 10. Next steps

* **Pilot on real data**: a historian extract for one unit plus two to three years of CMMS work
  orders; start by reproducing the conventional-alarm baseline and the false-alarm budget.
* **Alert feedback**: capture the inspection outcome of every alert and use it to retrain and
  re-tune thresholds; this also yields real P points.
* **Spectral features**: when waveform data is available, bearing defect frequencies (BPFO, BPFI,
  BSF, FTF) from envelope spectra instead of a single high-frequency band.
* **Survival models** (for example gradient-boosted Cox or accelerated-failure-time models) for RUL
  that use censored degradations directly.
* **Per-criticality thresholds**: a different cost and alarm budget for A-critical pumps than for
  C-critical ones.
* **Integration**: write status and recommendations back to the historian and raise notifications
  in the CMMS (SAP PM, Maximo) instead of a separate dashboard.

---

## References

* Moubray, J. *Reliability-Centered Maintenance*, 2nd ed. Industrial Press, 1997 (P-F interval).
* ISO 10816-3:2009 / ISO 20816-3:2022, *Mechanical vibration: evaluation of machine vibration by
  measurements on non-rotating parts*, Part 3 (industrial machines, 15 kW to 50 MW).
* ANSI/ISA-18.2-2016, *Management of Alarm Systems for the Process Industries*; EEMUA Publication
  191, *Alarm Systems: A Guide to Design, Management and Procurement*.
* MacGregor, J. F. and Kourti, T. "Statistical process control of multivariate processes."
  *Control Engineering Practice* 3(3), 1995 (MSPC, Hotelling T² and SPE).
* Liu, F. T., Ting, K. M. and Zhou, Z.-H. "Isolation Forest." *ICDM*, 2008.
* Romano, Y., Patterson, E. and Candès, E. "Conformalized Quantile Regression." *NeurIPS*, 2019.
* Matzka, S. "Explainable Artificial Intelligence for Predictive Maintenance Applications." *Third
  International Conference on Artificial Intelligence for Industries (AI4I)*, 2020; dataset: AI4I 2020
  Predictive Maintenance Dataset, UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C
  (CC BY 4.0).
