"""Explicit channel tuning: translate a selected capture band to baseband."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .filters import apply_fir_filter, design_lowpass_fir
from .impairments import apply_frequency_offset

__all__ = ["BandExtraction", "extract_band"]


@dataclass(frozen=True)
class BandExtraction:
    samples: npt.NDArray[np.complex128]
    lower_hz: float
    upper_hz: float
    center_hz: float
    translation_hz: float
    output_center_hz: float
    fs: float
    numtaps: int
    filter_delay_samples: float
    valid_start_sample: int
    valid_end_sample: int


def extract_band(samples: np.ndarray, fs: float, lower_hz: float, upper_hz: float, *, transition_hz: float, numtaps: int = 101) -> BandExtraction:
    """Mix ``[lower_hz, upper_hz]`` toward DC then causally low-pass filter.

    Returned samples stay tuned near DC; this is channel tuning, not CFO
    estimation or correction. The causal FIR start before ``valid_start`` is
    transient-contaminated. No decimation is performed.
    """
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite.")
    if not all(math.isfinite(value) for value in (lower_hz, upper_hz, transition_hz)):
        raise ValueError("band bounds and transition_hz must be finite.")
    nyquist = fs / 2.0
    if not -nyquist <= lower_hz < upper_hz <= nyquist:
        raise ValueError("band must lie within the signed Nyquist interval.")
    if transition_hz <= 0:
        raise ValueError("transition_hz must be positive.")
    half_width = (upper_hz - lower_hz) / 2.0
    cutoff = half_width + transition_hz
    if cutoff >= nyquist:
        raise ValueError("bandwidth plus transition must lie below Nyquist.")
    center = (lower_hz + upper_hz) / 2.0
    shifted = apply_frequency_offset(samples, fs, -center)
    taps = design_lowpass_fir(fs, cutoff, numtaps=numtaps)
    filtered = np.asarray(apply_fir_filter(shifted, taps), dtype=np.complex128)
    valid_start = min(filtered.size, numtaps - 1)
    return BandExtraction(filtered, float(lower_hz), float(upper_hz), float(center), float(-center), 0.0, float(fs), int(numtaps), (numtaps - 1) / 2.0, valid_start, int(filtered.size))
