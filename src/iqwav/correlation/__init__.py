"""Correlation and statistical analysis utilities for IQWAV."""

from .autocorrelation import autocorrelation, normalized_autocorrelation
from .cross_correlation import cross_correlation, normalized_cross_correlation
from .peaks import find_correlation_peaks

__all__ = [
    "autocorrelation",
    "cross_correlation",
    "find_correlation_peaks",
    "normalized_autocorrelation",
    "normalized_cross_correlation",
]
