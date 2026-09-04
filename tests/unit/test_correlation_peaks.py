"""Unit tests for correlation peak detection in iqwav.correlation.peaks."""

import numpy as np
import pytest

from iqwav.correlation import cross_correlation, find_correlation_peaks


def test_peak_detection_at_known_lag():
    reference = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    delayed = np.pad(reference, (3, 0))
    lags, values = cross_correlation(delayed, reference)
    indices, peak_lags, peak_values = find_correlation_peaks(
        values, lags, min_height=0.99 * float(np.abs(values).max())
    )
    np.testing.assert_array_equal(peak_lags, [3])
    np.testing.assert_allclose(peak_values, np.sum(np.abs(reference) ** 2))
    assert indices.dtype == np.int64
    assert peak_lags.dtype == np.int64


def test_peak_detection_returns_expected_lags_and_values():
    lags = np.arange(-4, 5)
    values = np.array([0.0, 0.1, 0.2, 0.5, 0.1, 0.3, 1.0 + 1.0j, 0.2, 0.1])
    indices, peak_lags, peak_values = find_correlation_peaks(
        values, lags, min_height=0.4
    )
    np.testing.assert_array_equal(indices, [3, 6])
    np.testing.assert_array_equal(peak_lags, [-1, 2])
    np.testing.assert_array_equal(peak_values, [0.5, 1.0 + 1.0j])


def test_magnitude_detection_finds_negative_troughs():
    lags = np.arange(-3, 4)
    values = np.array([0.1, 0.2, -0.9, 0.2, 0.1, 0.15, 0.05])
    _, magnitude_lags, _ = find_correlation_peaks(values, lags)
    assert -1 in magnitude_lags
    _, signed_lags, _ = find_correlation_peaks(values, lags, use_magnitude=False)
    assert -1 not in signed_lags
    np.testing.assert_array_equal(signed_lags, [-2, 0, 2])


def test_min_distance_suppresses_close_peaks():
    lags = np.arange(-5, 6)
    values = np.array([0.0, 0.8, 0.9, 0.2, 0.85, 0.1, 0.3, 0.05, 0.4, 0.1, 0.0])
    _, plain_lags, _ = find_correlation_peaks(values, lags)
    np.testing.assert_array_equal(plain_lags, [-3, -1, 1, 3])
    _, spread_lags, _ = find_correlation_peaks(values, lags, min_distance=3)
    np.testing.assert_array_equal(spread_lags, [-3, 3])


def test_signed_detection_rejects_complex_correlation():
    lags = np.arange(-1, 2)
    values = np.array([0.0 + 1.0j, 1.0 + 1.0j, 0.0 + 1.0j])
    with pytest.raises(ValueError):
        find_correlation_peaks(values, lags, use_magnitude=False)


def test_peak_detection_rejects_invalid_arrays():
    values = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        find_correlation_peaks(values, np.array([0, 1]))
    with pytest.raises(ValueError):
        find_correlation_peaks(values, np.ones((3, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        find_correlation_peaks(values, np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        find_correlation_peaks(np.ones((3, 3)), np.arange(3))
    with pytest.raises(ValueError):
        find_correlation_peaks(np.array([]), np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        find_correlation_peaks(np.array([1.0, np.nan, 0.0]), np.arange(3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_distance": 0},
        {"min_distance": 1.5},
        {"min_distance": True},
        {"min_distance": "1"},
        {"min_height": -1.0},
        {"min_height": True},
        {"min_height": float("inf")},
        {"min_height": "high"},
        {"prominence": -0.5},
        {"prominence": float("nan")},
        {"prominence": True},
        {"use_magnitude": "yes"},
    ],
)
def test_peak_detection_rejects_invalid_options(kwargs):
    values = np.array([0.0, 1.0, 0.0])
    lags = np.array([-1, 0, 1])
    with pytest.raises(ValueError):
        find_correlation_peaks(values, lags, **kwargs)
