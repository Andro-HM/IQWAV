"""Frozen four-class, rule-based AMC baseline.

This classifier operates directly on unsynchronized complex IQ samples.
It implements the accepted Module 12D experimental feature equations and
fixed hierarchy for the primary labels ``am``, ``angle``, ``bpsk``, and
``qpsk``. ``angle`` intentionally combines the FM and PM truth families.

It is a bounded synthetic-domain baseline, not a universal or calibrated
AMC system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "AMCFeatures",
    "AMCResult",
    "PRIMARY_AMC_LABELS",
    "classify_amc_features",
    "classify_modulation",
    "extract_amc_features",
]

EPS = 1e-12
MAG_GUARD = 1e-12
FFT_PAD = 8
PRIMARY_AMC_LABELS = ("am", "angle", "bpsk", "qpsk")

# Frozen 12D2 thresholds. Comparisons are deliberately strict (>).
_AM_COHERENCE_THRESHOLD = 0.1488818140943029
_AM_DISPERSION_THRESHOLD = 0.24798287316712025
_PSK_SPARSITY_THRESHOLD = 0.4361237911585857
_PSK_PERIODICITY_THRESHOLD = 0.599624718621257
_BPSK_C2_THRESHOLD = 0.4141447011963815
_BPSK_C2_C4_RATIO_THRESHOLD = 2.847400439321907


@dataclass(frozen=True)
class AMCFeatures:
    """Frozen Module 12D feature vector, with no confidence interpretation."""

    envelope_dispersion: float
    envelope_coherence: float
    phase_step_sparsity: float
    transition_periodicity: float
    c2: float
    c4: float
    c2_c4_ratio: float


@dataclass(frozen=True)
class AMCResult:
    """Primary AMC label and transparent frozen-rule diagnostics."""

    label: str
    features: AMCFeatures
    am_gate_passed: bool
    psk_gate_passed: bool
    bpsk_gate_passed: bool


def _as_complex_iq(samples: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Validate and copy a one-dimensional finite complex-IQ record."""
    raw = np.asarray(samples)
    if raw.ndim != 1:
        raise ValueError("samples must be a one-dimensional complex IQ array.")
    if raw.size == 0:
        raise ValueError("samples must be non-empty.")
    if not np.iscomplexobj(raw):
        raise ValueError("samples must contain complex IQ values.")
    iq = np.asarray(raw, dtype=np.complex128)
    if not np.all(np.isfinite(iq.real)) or not np.all(np.isfinite(iq.imag)):
        raise ValueError("samples must contain only finite values.")
    return iq


def _rms_normalize(samples: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Frozen RMS normalization; intentionally does not remove complex mean."""
    power = float(np.mean(np.abs(samples) ** 2))
    if not math.isfinite(power) or power <= EPS:
        raise ValueError("sample power is numerically zero.")
    return samples / math.sqrt(power)


def _phase_unit(
    samples: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.bool_]]:
    magnitude = np.abs(samples)
    good = magnitude > MAG_GUARD
    unit = np.zeros_like(samples, dtype=np.complex128)
    unit[good] = samples[good] / magnitude[good]
    return unit, good


def _residual_phase_steps(samples: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    unit, good = _phase_unit(samples)
    pairs = good[1:] & good[:-1]
    products = unit[1:] * np.conj(unit[:-1])
    valid = products[pairs]
    if valid.size == 0:
        return np.zeros(0, dtype=np.float64)
    direction = np.sum(valid)
    aligned = valid if abs(direction) <= EPS else valid * np.conj(direction / abs(direction))
    return np.abs(np.angle(aligned)).astype(np.float64)


def _residual_step_energy_series(samples: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """Frozen N-1 energy series, retaining zeroes at unguarded pairs."""
    unit, good = _phase_unit(samples)
    pairs = good[1:] & good[:-1]
    products = unit[1:] * np.conj(unit[:-1])
    valid = products[pairs]
    energy = np.zeros(products.shape[0], dtype=np.float64)
    if valid.size == 0:
        return energy
    direction = np.sum(valid)
    aligned = valid if abs(direction) <= EPS else valid * np.conj(direction / abs(direction))
    energy[pairs] = np.abs(np.angle(aligned)) ** 2
    return energy


def _transition_periodicity(samples: npt.NDArray[np.complex128]) -> float:
    """Corrected frozen maximum-positive, overlap-normalized periodicity."""
    energy = _residual_step_energy_series(samples)
    centered = energy - float(np.mean(energy))
    max_lag = min(32, centered.size - 1)
    if max_lag < 2:
        return 0.0
    peak = 0.0
    for lag in range(2, max_lag + 1):
        a = centered[:-lag]
        b = centered[lag:]
        denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
        value = 0.0 if denominator <= EPS else float(np.dot(a, b) / denominator)
        if value > peak:
            peak = value
    return float(max(0.0, peak))


def _m_power_concentration(samples: npt.NDArray[np.complex128], m: int) -> float:
    """Frozen M-power coherent-line concentration using an 8N FFT."""
    unit, _good = _phase_unit(samples)
    powered = unit**m
    power = float(np.sum(np.abs(powered) ** 2))
    if power <= EPS:
        return 0.0
    n_samples = powered.size
    spectrum = np.fft.fft(powered, n=FFT_PAD * n_samples)
    peak = float(np.max(np.abs(spectrum) ** 2))
    return peak / (n_samples * power)


def extract_amc_features(samples: npt.ArrayLike) -> AMCFeatures:
    """Extract the frozen sample-only Module 12 AMC feature vector.

    A non-empty complex record is accepted. Very short valid records retain
    the experiment's guarded behavior: unavailable phase-step periodicity is
    reported as zero, not as a recommended reliable AMC length.
    """
    normalized = _rms_normalize(_as_complex_iq(samples))
    envelope = np.abs(normalized).astype(np.float64, copy=False)
    mean_envelope = float(np.mean(envelope))
    dispersion = float(np.var(envelope, ddof=0)) / (mean_envelope * mean_envelope + EPS)

    centered_envelope = envelope - mean_envelope
    a = centered_envelope[1:]
    b = centered_envelope[:-1]
    coherence_denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    coherence = 0.0 if coherence_denominator <= EPS else float(np.dot(a, b) / coherence_denominator)

    steps = _residual_phase_steps(normalized)
    if steps.size == 0:
        sparsity = 0.0
    else:
        rms = float(np.sqrt(np.mean(steps * steps)))
        sparsity = 0.0 if rms <= EPS else float(1.0 - np.median(steps) / (rms + EPS))

    c2 = _m_power_concentration(normalized, 2)
    c4 = _m_power_concentration(normalized, 4)
    return AMCFeatures(
        envelope_dispersion=dispersion,
        envelope_coherence=coherence,
        phase_step_sparsity=sparsity,
        transition_periodicity=_transition_periodicity(normalized),
        c2=c2,
        c4=c4,
        c2_c4_ratio=c2 / (c4 + EPS),
    )


def classify_amc_features(features: AMCFeatures) -> AMCResult:
    """Apply the frozen strict-threshold hierarchy to a feature vector."""
    am_gate = features.envelope_coherence > _AM_COHERENCE_THRESHOLD or features.envelope_dispersion > _AM_DISPERSION_THRESHOLD
    psk_gate = features.phase_step_sparsity > _PSK_SPARSITY_THRESHOLD or features.transition_periodicity > _PSK_PERIODICITY_THRESHOLD
    bpsk_gate = features.c2 > _BPSK_C2_THRESHOLD or features.c2_c4_ratio > _BPSK_C2_C4_RATIO_THRESHOLD
    if am_gate:
        label = "am"
    elif not psk_gate:
        label = "angle"
    elif bpsk_gate:
        label = "bpsk"
    else:
        label = "qpsk"
    return AMCResult(label, features, am_gate, psk_gate, bpsk_gate)


def classify_modulation(samples: npt.ArrayLike) -> AMCResult:
    """Classify unsynchronized complex IQ as AM, ANGLE, BPSK, or QPSK."""
    return classify_amc_features(extract_amc_features(samples))
