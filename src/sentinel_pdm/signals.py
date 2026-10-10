"""Vibration waveforms and spectra: how an analyst confirms a rolling-element bearing defect.

Why this module exists
----------------------
A plant historian stores *hourly band values* for each pump: overall velocity RMS, the 1x and
2x running-speed amplitudes, a high-frequency "bearing band" in g and the kurtosis of the
acceleration waveform. Those numbers trend well and feed the models, but when an alert comes in
a vibration analyst does not stop there. They take a raw acceleration waveform with a portable
data collector (typically 25.6 kHz for about one second) and read its spectrum and its
*envelope spectrum* to confirm what is wrong before a mechanic is sent out.

This module rebuilds an illustrative waveform from the hourly features so the dashboard can
show that confirmation step. The waveform is a physically motivated **reconstruction**, not
measured data: it has the same running speed, the same 1x / 2x amplitudes, the same
high-frequency band level and roughly the same impulsiveness as the pump's historian readings.

Envelope (demodulation) analysis
--------------------------------
A localised defect on a bearing race (a spall) produces a short impact every time a rolling
element passes over it. Each impact is far too brief to appear at its own repetition rate in
an ordinary spectrum. Instead it makes the bearing housing ring at its structural resonances
(typically 1-10 kHz), so the raw spectrum only shows a broad hump at high frequency, buried
under running-speed vibration and process noise. The repetition rate survives as an *amplitude
modulation* of that ringing, and envelope analysis recovers it in three steps:

1. **Band-pass** the waveform around the excited resonance (here 2-4.5 kHz). This removes the
   large low-frequency 1x / 2x components.
2. **Demodulate** by taking the magnitude of the analytic signal (Hilbert transform). That is
   the envelope of the ringing, and it turns each impact into a single pulse.
3. **FFT the envelope.** Impacts that repeat at a bearing defect frequency now show up as a
   discrete line at that frequency, plus its harmonics.

Which line appears identifies the damaged part. With shaft frequency f_r, n rolling elements,
ball diameter d, pitch diameter D and contact angle phi::

    FTF  = f_r / 2 * (1 - d/D cos phi)                  cage (fundamental train)
    BPFO = n f_r / 2 * (1 - d/D cos phi)                outer-race defect
    BPFI = n f_r / 2 * (1 + d/D cos phi)                inner-race defect
    BSF  = D / (2 d) * f_r * (1 - (d/D cos phi)^2)      ball spin (a ball defect hits both races: 2 x BSF)

An outer-race defect stays still in the load zone, so its impacts are all about as strong and
you get clean BPFO harmonics. An inner-race defect rotates in and out of the load zone, so its
impacts are amplitude-modulated at 1x and show as BPFI with +/-1x sidebands. Cavitation also
produces impacts, but at random times, so its envelope spectrum stays broadband with no
discrete lines. That difference is how an analyst tells cavitation from a bearing defect.

How the reconstruction is calibrated
------------------------------------
* **1x / 2x**: sinusoids. The historian reports velocity RMS in mm/s, and it is converted to
  acceleration at that frequency with a = 2 pi f v.
* **High-frequency band**: the structural resonance is driven partly by random forces
  (Gaussian, kurtosis 3) and partly by defect impacts. If the impacts carry an energy share s of
  the band, then for independent, zero-mean, unit-variance parts the band kurtosis is
  3 + s^2 (k_imp - 3). The share is set so the band kurtosis rises with the measured kurtosis
  above ``HEALTHY_KURTOSIS``. A healthy pump (kurtosis about 3.2 in hourly data) therefore gets
  no impacts at all, and the band's RMS equals the measured ``vib_hf``.
* A small broadband sensor noise floor.

Everything is deterministic for a given ``seed`` and takes a few milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, get_window, hilbert, lfilter, sosfiltfilt

G = 9.80665                 # standard gravity, m/s^2 (accelerometers report g)
HEALTHY_KURTOSIS = 3.5      # hourly kurtosis of a healthy pump scatters around 3.2; only the excess is impacts
NOISE_FLOOR_G = 0.01        # accelerometer + data-collector noise floor, g RMS
RESONANCE_DAMPING = 0.08    # damping ratio of the housing resonance excited by the impacts
TIMING_JITTER = 0.004       # ball slip: relative scatter of the interval between impacts
PRE_ROLL_S = 0.02           # simulated before t = 0 so filters start in steady state

Defect = Literal["outer", "inner", "ball", "random"]
DEFECT_LABELS: dict[str, str] = {
    "outer": "Outer-race defect (impacts at BPFO)",
    "inner": "Inner-race defect (impacts at BPFI, modulated at 1x)",
    "ball": "Rolling-element defect (impacts at 2 x BSF, modulated at FTF)",
    "random": "Random impacts (cavitation: no periodic defect)",
}


@dataclass(frozen=True)
class BearingGeometry:
    """Rolling-element bearing geometry used for the defect frequencies.

    The default is close to a 6309 deep-groove ball bearing (45 mm bore, 100 mm OD, 8 balls of
    11/16 in), a common drive-end bearing on 50-150 kW process pumps.
    """

    n_balls: int = 8
    ball_diameter_mm: float = 17.462
    pitch_diameter_mm: float = 72.5
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_balls < 3:
            raise ValueError("a rolling-element bearing needs at least 3 rolling elements")
        if not 0 < self.ball_diameter_mm < self.pitch_diameter_mm:
            raise ValueError("ball diameter must be positive and smaller than the pitch diameter")
        if not 0 <= self.contact_angle_deg < 90:
            raise ValueError("contact angle must be in [0, 90) degrees")

    def defect_frequencies(self, shaft_hz: float) -> dict[str, float]:
        """Return the bearing defect frequencies in Hz for a shaft speed in Hz.

        The keys are FTF (cage), BSF (ball spin), BPFO (outer race) and BPFI (inner race).
        """
        ratio = self.ball_diameter_mm / self.pitch_diameter_mm * np.cos(np.radians(self.contact_angle_deg))
        n = self.n_balls
        return {
            "FTF": float(shaft_hz / 2 * (1 - ratio)),
            "BSF": float(self.pitch_diameter_mm / (2 * self.ball_diameter_mm) * shaft_hz * (1 - ratio**2)),
            "BPFO": float(n * shaft_hz / 2 * (1 - ratio)),
            "BPFI": float(n * shaft_hz / 2 * (1 + ratio)),
        }


DEFAULT_BEARING = BearingGeometry()


def velocity_to_acceleration_g(velocity_mm_s: float, freq_hz: float) -> float:
    """Convert a vibration velocity (mm/s) at one frequency to acceleration (g).

    The result keeps the same measure as the input (RMS in gives RMS out). For a sinusoid,
    a = 2 pi f v.
    """
    return float(2 * np.pi * freq_hz * velocity_mm_s / 1000.0 / G)


def _clean_amplitude(x: float | None) -> float:
    """Treat a missing (NaN / None) or negative historian value as 'no such component'."""
    if x is None or not np.isfinite(x):
        return 0.0
    return max(float(x), 0.0)


def _kurtosis(x: np.ndarray) -> float:
    """Pearson kurtosis (3 for a Gaussian), the definition used by vibration data collectors."""
    x = x - x.mean()
    var = np.mean(x * x)
    return float(np.mean(x**4) / var**2) if var > 0 else 3.0


def _resonator(x: np.ndarray, fs: float, freq_hz: float, damping: float) -> np.ndarray:
    """Pass ``x`` through a single-degree-of-freedom structural resonance (impulse-invariant IIR).

    Its impulse response is r^k sin(theta k): a decaying ring at ``freq_hz``. That is how one
    bearing impact looks on an accelerometer mounted on the housing.
    """
    wn = 2 * np.pi * freq_hz
    wd = wn * np.sqrt(1 - damping**2)
    r = np.exp(-damping * wn / fs)
    theta = wd / fs
    return lfilter([0.0, r * np.sin(theta)], [1.0, -2 * r * np.cos(theta), r * r], x)


def _impact_train(
    defect: str, n: int, fs: float, shaft_hz: float, freqs: dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    """Unit impulses at the defect's impact times, with physically motivated amplitudes."""
    duration = n / fs
    if defect == "random":
        # Cavitation: bubble collapses at random (Poisson) times with very uneven strength.
        n_imp = rng.poisson(freqs["BPFO"] * duration)
        times = rng.uniform(0.0, duration, n_imp)
        amps = rng.exponential(1.0, n_imp)
    else:
        rate = {"outer": freqs["BPFO"], "inner": freqs["BPFI"], "ball": 2 * freqs["BSF"]}[defect]
        period = 1.0 / rate
        n_imp = int(np.ceil(duration * rate)) + 2
        # Rolling elements slip a little, so the interval between impacts scatters around 1 / rate.
        intervals = period * (1 + TIMING_JITTER * rng.standard_normal(n_imp))
        times = rng.uniform(0.0, period) + np.concatenate([[0.0], np.cumsum(intervals[:-1])])
        amps = 1 + 0.1 * rng.standard_normal(n_imp)
        if defect == "inner":
            # The defect rotates with the shaft through the load zone: strong once per revolution.
            amps *= 0.55 + 0.45 * np.cos(2 * np.pi * shaft_hz * times + rng.uniform(0, 2 * np.pi))
        elif defect == "ball":
            # The ball rides with the cage, so it is loaded once per cage revolution.
            amps *= 0.55 + 0.45 * np.cos(2 * np.pi * freqs["FTF"] * times + rng.uniform(0, 2 * np.pi))
    idx = np.round(times * fs).astype(np.int64)
    keep = (idx >= 0) & (idx < n)
    train = np.zeros(n)
    np.add.at(train, idx[keep], amps[keep])
    return train


def synthesize_waveform(
    speed_rpm: float,
    vib_1x: float,
    vib_2x: float,
    vib_hf: float,
    kurtosis: float,
    fs: float = 25600,
    duration_s: float = 1.0,
    seed: int = 0,
    geometry: BearingGeometry | None = None,
    *,
    defect: Defect = "outer",
    resonance_hz: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct an illustrative housing acceleration waveform from hourly vibration features.

    Parameters
    ----------
    speed_rpm : shaft speed. Must be positive, because a stopped pump has no waveform to analyse.
    vib_1x, vib_2x : 1x / 2x running-speed velocity, mm/s RMS.
    vib_hf : high-frequency bearing-band acceleration, g RMS.
    kurtosis : kurtosis of the acceleration waveform (healthy about 3). It sets how much of the
        high-frequency energy comes from defect impacts rather than random forces.
    fs, duration_s : sample rate (Hz) and record length (s), as on a portable data collector.
    seed : random seed. The same inputs and seed always give the same waveform.
    geometry : bearing geometry (default: ``DEFAULT_BEARING``, close to a 6309).
    defect : impact source. "outer" (BPFO, default), "inner" (BPFI with 1x modulation),
        "ball" (2 x BSF with cage modulation) or "random" (cavitation-like, no periodicity).
    resonance_hz : housing resonance that the impacts excite. It must sit inside the envelope
        band you analyse.

    Missing (NaN) amplitudes are treated as zero and a missing kurtosis as healthy.

    Returns
    -------
    (t, accel_g) : time in seconds starting at 0, and acceleration in g.
    """
    if speed_rpm is None or not np.isfinite(speed_rpm) or speed_rpm <= 0:
        raise ValueError("speed_rpm must be positive: a stopped pump has no running-speed vibration")
    if fs <= 0 or duration_s <= 0:
        raise ValueError("fs and duration_s must be positive")
    if not 0 < resonance_hz < fs / 2:
        raise ValueError(f"resonance_hz must lie between 0 and the Nyquist frequency ({fs / 2:g} Hz)")
    if defect not in DEFECT_LABELS:
        raise ValueError(f"defect must be one of {sorted(DEFECT_LABELS)}")

    geometry = geometry or DEFAULT_BEARING
    rng = np.random.default_rng(seed)
    v1, v2, hf_rms = _clean_amplitude(vib_1x), _clean_amplitude(vib_2x), _clean_amplitude(vib_hf)
    kurt = float(kurtosis) if kurtosis is not None and np.isfinite(kurtosis) else 3.0

    n = int(round(fs * duration_s))
    pre = int(round(fs * PRE_ROLL_S))
    t = np.arange(n) / fs
    shaft_hz = speed_rpm / 60.0
    freqs = geometry.defect_frequencies(shaft_hz)

    # Running-speed components: RMS velocity -> peak acceleration of a sinusoid.
    x = np.zeros(n)
    for order, v in ((1, v1), (2, v2)):
        a_peak = np.sqrt(2) * velocity_to_acceleration_g(v, order * shaft_hz)
        x += a_peak * np.sin(2 * np.pi * order * shaft_hz * t + rng.uniform(0, 2 * np.pi))

    # High-frequency band: one housing resonance driven by random forces and/or defect impacts.
    # Both are simulated with a pre-roll and trimmed so the record starts in steady state.
    noise = _resonator(rng.standard_normal(n + pre), fs, resonance_hz, RESONANCE_DAMPING)[pre:]
    noise /= noise.std()
    excess = max(kurt - HEALTHY_KURTOSIS, 0.0)
    share = 0.0
    if excess > 0 and hf_rms > 0:
        train = _impact_train(defect, n + pre, fs, shaft_hz, freqs, rng)
        impacts = _resonator(train, fs, resonance_hz, RESONANCE_DAMPING)[pre:]
        impacts -= impacts.mean()
        if impacts.std() > 0:
            impacts /= impacts.std()
            k_imp = _kurtosis(impacts)
            # kurt(band) = 3 + share^2 (k_imp - 3) for independent unit-variance parts.
            share = float(np.clip(np.sqrt(excess / max(k_imp - 3.0, 1e-6)), 0.0, 1.0))
            x += hf_rms * np.sqrt(share) * impacts
    x += hf_rms * np.sqrt(1.0 - share) * noise
    x += NOISE_FLOOR_G * rng.standard_normal(n)
    return t, x


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum with a Hann window.

    The mean is removed first (accelerometers are AC-coupled). Amplitudes are *peak* values,
    corrected for the Hann window's coherent gain, so a sinusoid of amplitude A shows up as a
    line of height ~A in the input's units.

    Returns
    -------
    (freqs_hz, amplitude)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1-D signal with at least 2 samples")
    if fs <= 0:
        raise ValueError("fs must be positive")
    x = np.nan_to_num(x - np.nanmean(x))
    window = get_window("hann", x.size)
    amp = 2.0 * np.abs(np.fft.rfft(x * window)) / window.sum()
    amp[0] /= 2.0
    if x.size % 2 == 0:
        amp[-1] /= 2.0  # the Nyquist bin has no mirror image either
    return np.fft.rfftfreq(x.size, d=1.0 / fs), amp


def envelope_spectrum(
    x: np.ndarray, fs: float, band: tuple[float, float] = (2000.0, 4500.0), order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (demodulated) spectrum: band-pass, Hilbert envelope, then FFT.

    The Butterworth band-pass runs forwards and backwards (zero phase), so the impact timing is
    preserved. Lines in the result are impact repetition rates (BPFO, BPFI, 2 x BSF, ...), in g.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = band
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f"band must satisfy 0 < low < high < fs/2 = {fs / 2:g} Hz, got {band}")
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if x.size <= padlen:
        raise ValueError(f"signal too short for the band-pass filter (need more than {padlen} samples)")
    filtered = sosfiltfilt(sos, np.nan_to_num(x - np.nanmean(x)))
    envelope = np.abs(hilbert(filtered))
    return spectrum(envelope, fs)


def peak_near(freqs: np.ndarray, amp: np.ndarray, target_hz: float, rel_tol: float = 0.02) -> tuple[float, float]:
    """Largest line within +/- ``rel_tol`` of ``target_hz``. Returns (freq_hz, amplitude), NaN if none.

    The tolerance absorbs bearing slip: real defect lines sit 1-2 % off the calculated value.
    """
    tol = max(abs(target_hz) * rel_tol, freqs[1] - freqs[0] if len(freqs) > 1 else 0.0)
    sel = np.flatnonzero(np.abs(freqs - target_hz) <= tol)
    if sel.size == 0:
        return float("nan"), float("nan")
    i = sel[np.argmax(amp[sel])]
    return float(freqs[i]), float(amp[i])


def line_prominence(freqs: np.ndarray, amp: np.ndarray, target_hz: float, rel_tol: float = 0.02) -> float:
    """How far a spectral line stands above its local floor (peak / median of the +/-25 % neighbourhood).

    Values near 1-3 mean there is no line, just noise. A clear defect line is typically above 5.
    This is the number an analyst checks by eye before writing "BPFO present" in a report.
    """
    _, peak = peak_near(freqs, amp, target_hz, rel_tol)
    near = np.abs(freqs - target_hz)
    floor_sel = (near <= 0.25 * target_hz) & (near > max(rel_tol * target_hz, 2 * (freqs[1] - freqs[0])))
    if not np.isfinite(peak) or not floor_sel.any():
        return float("nan")
    floor = float(np.median(amp[floor_sel]))
    return peak / floor if floor > 0 else float("inf")


__all__ = [
    "DEFAULT_BEARING", "DEFECT_LABELS", "HEALTHY_KURTOSIS", "BearingGeometry", "envelope_spectrum",
    "line_prominence", "peak_near", "spectrum", "synthesize_waveform", "velocity_to_acceleration_g",
]
