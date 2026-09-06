"""Private Module 12D1 experimental helpers.

Not a production AMC API. Not a classifier. Not Module 11 estimation.
Do not import this from src/iqwav.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.modulation import fm_modulate, pm_modulate

EPS = 1e-12
MAG_GUARD = 1e-12
PERIODICITY_LAGS = range(2, 33)
FFT_PAD = 8
FEATURE_NAMES = (
    "envelope_dispersion",
    "envelope_coherence",
    "phase_sparsity",
    "transition_periodicity",
    "c2",
    "c4",
    "c2_over_c4",
)

PRIMARY_LABELS = ("am", "angle", "bpsk", "qpsk")


def primary_label(label: str) -> str:
    if label in ("fm", "pm"):
        return "angle"
    return label


# ---------------------------------------------------------------------------
# A. FM/PM exact-identifiability construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FMPMIdentityResult:
    n: int
    fs: float
    f_norm: float
    beta: float
    delta_f_norm: float
    difference_amplitude: float
    constant_phase_rad: float
    max_abs_before: float
    rms_before: float
    max_abs_after: float
    rms_after: float
    max_abs_impaired: float
    rms_impaired: float
    increment_max_abs_err: float


def construct_fm_message_from_pm(
    m_p: np.ndarray,
) -> tuple[npt.NDArray[np.float64], float]:
    """Build a valid |m|<=1 FM message from a PM message difference.

    m_f[0] = 0
    m_f[n] = (m_p[n] - m_p[n-1]) / A, n >= 1

    where A is the peak absolute discrete difference.
    """
    m_p = np.asarray(m_p, dtype=np.float64)
    diff = np.empty_like(m_p)
    diff[0] = 0.0
    diff[1:] = m_p[1:] - m_p[:-1]
    amplitude = float(np.max(np.abs(diff[1:])))
    if amplitude <= EPS:
        raise ValueError("PM message difference amplitude is numerically zero.")
    m_f = diff / amplitude
    m_f[0] = 0.0
    m_f = np.clip(m_f, -1.0, 1.0)
    return m_f, amplitude


def fm_pm_identity_experiment(
    *,
    n: int = 4096,
    fs: float = 48_000.0,
    f_norm: float = 0.04,
    beta: float = 0.8,
) -> FMPMIdentityResult:
    """Construct matched FM/PM sinusoids under the IQWAV FM cumsum contract.

    Analytic relation, with A = max |m_p[n]-m_p[n-1]| and
    delta_f_norm = beta * A / (2 pi):

        phi_PM[n] = beta * m_p[n]
        phi_FM[n] = 2 pi * delta_f_norm * cumsum(m_f)[n]
                  = beta * (m_p[n] - m_p[0])

    so s_FM[n] = s_PM[n] * exp(-j * beta * m_p[0]).
    """
    n_axis = np.arange(n, dtype=np.float64)
    m_p = np.cos(2.0 * np.pi * f_norm * n_axis)
    m_f, amplitude = construct_fm_message_from_pm(m_p)
    delta_f_norm = beta * amplitude / (2.0 * np.pi)
    if not (0.02 <= delta_f_norm < 0.08):
        raise ValueError(
            f"delta_f_norm={delta_f_norm} is outside the accepted FM "
            "deviation-norm range [0.02, 0.08)."
        )
    if not (0.01 <= f_norm <= 0.08):
        raise ValueError(f"f_norm={f_norm} is outside analog_freq_norm_range.")
    if not (0.4 <= beta <= 1.2):
        raise ValueError(f"beta={beta} is outside pm_phase_deviation_range.")

    k = 2.0 * np.pi * delta_f_norm
    increment_left = k * m_f[1:]
    increment_right = beta * (m_p[1:] - m_p[:-1])
    increment_err = float(np.max(np.abs(increment_left - increment_right)))

    s_pm = pm_modulate(m_p, beta)
    s_fm = fm_modulate(m_f, fs, delta_f_norm * fs)
    constant_phase = -beta * m_p[0]
    max_abs_before = float(np.max(np.abs(s_fm - s_pm)))
    rms_before = float(np.sqrt(np.mean(np.abs(s_fm - s_pm) ** 2)))
    s_fm_aligned = s_fm * np.exp(-1j * constant_phase)
    max_abs_after = float(np.max(np.abs(s_fm_aligned - s_pm)))
    rms_after = float(np.sqrt(np.mean(np.abs(s_fm_aligned - s_pm) ** 2)))

    amplitude_scale = 1.7
    phase_rad = 0.6
    cfo_norm = 0.01
    impaired_fm = apply_frequency_offset(
        apply_phase_offset(amplitude_scale * s_fm_aligned, phase_rad),
        fs,
        cfo_norm * fs,
    )
    impaired_pm = apply_frequency_offset(
        apply_phase_offset(amplitude_scale * s_pm, phase_rad),
        fs,
        cfo_norm * fs,
    )
    noisy_fm = add_awgn(impaired_fm, 20.0, rng=np.random.default_rng(9))
    noise = noisy_fm - impaired_fm
    noisy_pm = impaired_pm + noise
    max_abs_impaired = float(np.max(np.abs(noisy_fm - noisy_pm)))
    rms_impaired = float(np.sqrt(np.mean(np.abs(noisy_fm - noisy_pm) ** 2)))
    return FMPMIdentityResult(
        n=n,
        fs=fs,
        f_norm=f_norm,
        beta=beta,
        delta_f_norm=float(delta_f_norm),
        difference_amplitude=amplitude,
        constant_phase_rad=float(constant_phase),
        max_abs_before=max_abs_before,
        rms_before=rms_before,
        max_abs_after=max_abs_after,
        rms_after=rms_after,
        max_abs_impaired=max_abs_impaired,
        rms_impaired=rms_impaired,
        increment_max_abs_err=increment_err,
    )


# ---------------------------------------------------------------------------
# B. Experiment-only features
# ---------------------------------------------------------------------------


class ZeroPowerRecordError(ValueError):
    """Raised when RMS normalization is undefined."""


def rms_normalize(samples: np.ndarray) -> npt.NDArray[np.complex128]:
    """r = x / sqrt(mean(|x|^2)). Does not subtract the complex mean."""
    x = np.asarray(samples, dtype=np.complex128)
    power = float(np.mean(np.abs(x) ** 2))
    if not math.isfinite(power) or power <= EPS:
        raise ZeroPowerRecordError("record power is numerically zero.")
    return x / math.sqrt(power)


def _envelope(r: np.ndarray) -> npt.NDArray[np.float64]:
    return np.abs(r).astype(np.float64, copy=False)


def envelope_dispersion(r: np.ndarray) -> float:
    """Ev = var(|r|) / (mean(|r|)^2 + eps), population variance (ddof=0)."""
    mag = _envelope(r)
    mean_mag = float(np.mean(mag))
    var_mag = float(np.var(mag, ddof=0))
    return var_mag / (mean_mag * mean_mag + EPS)


def envelope_coherence(r: np.ndarray) -> float:
    """Normalized lag-1 correlation of the centered envelope.

    e = |r| - mean(|r|). If either lagged vector has RMS below EPS,
    the correlation is undefined and this returns 0.0 (no envelope
    variation to correlate). Constant-envelope records therefore score 0.
    """
    mag = _envelope(r)
    e = mag - float(np.mean(mag))
    a = e[1:]
    b = e[:-1]
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def _phase_unit(r: np.ndarray) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.bool_]]:
    mag = _envelope(r)
    good = mag > MAG_GUARD
    u = np.zeros_like(r, dtype=np.complex128)
    u[good] = r[good] / mag[good]
    return u, good


def _residual_phase_steps(r: np.ndarray) -> npt.NDArray[np.float64]:
    """Adjacent phase steps after removing the common circular direction.

    Common direction is the argument of the sum of valid adjacent
    products. No modulation family is assumed. Low-magnitude samples
    are excluded from the circular mean and from the residual.
    """
    u, good = _phase_unit(r)
    pair = good[1:] & good[:-1]
    products = u[1:] * np.conj(u[:-1])
    valid = products[pair]
    if valid.size == 0:
        return np.zeros(0, dtype=np.float64)
    direction = np.sum(valid)
    if abs(direction) <= EPS:
        aligned = valid
    else:
        aligned = valid * np.conj(direction / abs(direction))
    return np.abs(np.angle(aligned)).astype(np.float64)


def phase_sparsity(r: np.ndarray) -> float:
    """1 - median(|step|) / (rms(|step|) + eps) on residual phase steps.

    Bounded in (-inf, 1]. Dense continuous phase walks score near 0.
    Sparse symbol-boundary steps score nearer 1. If residual RMS is
    below EPS (including a pure residual-free rotation), return 0.0.
    """
    steps = _residual_phase_steps(r)
    if steps.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(steps * steps)))
    if rms <= EPS:
        return 0.0
    return float(1.0 - np.median(steps) / (rms + EPS))


def _residual_step_energy_series(r: np.ndarray) -> npt.NDArray[np.float64]:
    """Length N-1 residual-step energy with zeros on unguarded pairs."""
    u, good = _phase_unit(r)
    pair = good[1:] & good[:-1]
    products = u[1:] * np.conj(u[:-1])
    valid = products[pair]
    energy = np.zeros(products.shape[0], dtype=np.float64)
    if valid.size == 0:
        return energy
    direction = np.sum(valid)
    if abs(direction) <= EPS:
        aligned = valid
    else:
        aligned = valid * np.conj(direction / abs(direction))
    energy[pair] = np.abs(np.angle(aligned)) ** 2
    return energy


def transition_periodicity(r: np.ndarray) -> float:
    """Maximum positive overlap-normalized residual-step-energy correlation.

    Lags 2..32 inclusive cover the current SPS domain {4, 8, 16} without
    calling symbol-rate or symbol-grid estimators. Zero-variance energy
    returns 0.0.
    """
    energy = _residual_step_energy_series(r)
    centered = energy - float(np.mean(energy))
    max_lag = min(32, centered.size - 1)
    if max_lag < 2:
        return 0.0
    peak = 0.0
    for lag in range(2, max_lag + 1):
        a = centered[:-lag]
        b = centered[lag:]
        denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
        if denom <= EPS:
            value = 0.0
        else:
            value = float(np.dot(a, b) / denom)
        if value > peak:
            peak = value
    return float(max(0.0, peak))


def m_power_concentration(r: np.ndarray, m: int) -> float:
    """max |FFT(u**M, 8N)|^2 / (N * sum(|u**M|^2)).

    u is phase-normalized. Low-magnitude samples are set to 0 so they
    do not contribute a phase. Score is ~1 for a bin-centered M-th-power
    tone and smaller when the M-th-power spectrum is spread.
    """
    u, _good = _phase_unit(r)
    v = u ** m
    power = float(np.sum(np.abs(v) ** 2))
    if power <= EPS:
        return 0.0
    n = v.size
    n_fft = FFT_PAD * n
    spectrum = np.fft.fft(v, n=n_fft)
    peak = float(np.max(np.abs(spectrum) ** 2))
    return peak / (n * power)


def extract_features(samples: np.ndarray) -> dict[str, float]:
    """Compute the 12D1 experimental feature vector from one IQ record."""
    r = rms_normalize(samples)
    c2 = m_power_concentration(r, 2)
    c4 = m_power_concentration(r, 4)
    return {
        "envelope_dispersion": envelope_dispersion(r),
        "envelope_coherence": envelope_coherence(r),
        "phase_sparsity": phase_sparsity(r),
        "transition_periodicity": transition_periodicity(r),
        "c2": c2,
        "c4": c4,
        "c2_over_c4": c2 / (c4 + EPS),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0.0,
            "mean": math.nan,
            "std": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "median": math.nan,
            "p75": math.nan,
            "p95": math.nan,
        }
    return {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }
