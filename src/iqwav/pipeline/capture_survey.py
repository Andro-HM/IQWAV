"""Thin capture survey composition without receiver synchronization logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iqwav.detection import ActivityDetection, detect_activity
from iqwav.dsp import BandExtraction, extract_band
from iqwav.estimation import OccupiedBand, detect_occupied_bands

__all__ = ["CaptureRegion", "CaptureSurvey", "survey_capture"]


@dataclass(frozen=True)
class CaptureRegion:
    band: OccupiedBand
    start_sample: int
    end_sample: int
    status: str
    extraction: BandExtraction | None
    reason: str | None


@dataclass(frozen=True)
class CaptureSurvey:
    activity: ActivityDetection | None
    bands: tuple[OccupiedBand, ...]
    regions: tuple[CaptureRegion, ...]
    noise_floor_db: float


def survey_capture(samples: np.ndarray, fs: float, *, nperseg: int | None = None, threshold_db: float = 6.0, min_bins: int = 3, include_activity: bool = True, extract: bool = False, transition_hz: float | None = None, numtaps: int = 101) -> CaptureSurvey:
    """Survey a capture; activity is optional and never gates spectral bands."""
    activity = detect_activity(samples, fs=fs) if include_activity else None
    bands, noise_floor_db = detect_occupied_bands(samples, fs, nperseg=nperseg, threshold_db=threshold_db, min_bins=min_bins)
    regions = []
    for band in bands:
        extraction = None
        status, reason = "candidate", None
        if extract:
            try:
                width = band.bandwidth_hz
                extraction = extract_band(samples, fs, band.lower_hz, band.upper_hz, transition_hz=transition_hz if transition_hz is not None else max(width * 0.1, fs / max(4 * len(samples), 1)), numtaps=numtaps)
                status = "extracted"
            except ValueError as error:
                status, reason = "failed", str(error)
        regions.append(CaptureRegion(band, 0, int(np.asarray(samples).size), status, extraction, reason))
    return CaptureSurvey(activity, tuple(bands), tuple(regions), noise_floor_db)
