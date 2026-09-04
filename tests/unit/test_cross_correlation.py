"""Unit tests for cross-correlation utilities in iqwav.correlation."""

import numpy as np
import pytest

from iqwav.correlation import cross_correlation, normalized_cross_correlation

FUNCTIONS = [cross_correlation, normalized_cross_correlation]


def test_known_real_cross_correlation():
    lags, values = cross_correlation(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0])
    )
    np.testing.assert_array_equal(lags, [-1, 0, 1, 2])
    np.testing.assert_allclose(values, [1.0, 3.0, 5.0, 3.0])
    assert values.dtype == np.float64
    assert np.isrealobj(values)


def test_complex_conjugation_convention():
    first = np.array([1.0 + 1.0j, 2.0 - 1.0j])
    second = np.array([2.0j, 1.0 - 1.0j])
    lags, values = cross_correlation(first, second)
    np.testing.assert_array_equal(lags, [-1, 0, 1])
    expected = np.array(
        [
            first[0] * np.conj(second[1]),
            first[0] * np.conj(second[0]) + first[1] * np.conj(second[1]),
            first[1] * np.conj(second[0]),
        ]
    )
    np.testing.assert_allclose(values, expected)

    literal_first = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    literal_second = np.array([2.0, 1.0 + 1.0j])
    _, literal_values = cross_correlation(literal_first, literal_second)
    np.testing.assert_allclose(
        literal_values, [3.0 + 1.0j, 4.0 + 0.0j, 6.0 - 2.0j]
    )
    assert literal_values.dtype == np.complex128
    assert np.iscomplexobj(literal_values)


def test_delayed_copy_peaks_at_positive_delay():
    reference = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
    delayed = np.pad(reference, (3, 0))
    lags, values = cross_correlation(delayed, reference)
    assert lags[int(np.argmax(np.abs(values)))] == 3
    np.testing.assert_allclose(values[lags == 3], np.sum(np.abs(reference) ** 2))
    assert lags[0] == -(len(reference) - 1)
    assert lags[-1] == len(delayed) - 1
    assert lags.dtype == np.int64


def test_integer_inputs_promote_to_float64():
    lags, values = cross_correlation(np.array([1, 2]), np.array([3, 4]))
    assert values.dtype == np.float64
    assert np.isrealobj(values)


def test_normalized_identical_overlap_is_one_at_delay():
    reference = np.array([1.0, -2.0, 3.0, -4.0])
    delayed = np.pad(reference, (2, 0))
    lags, values = normalized_cross_correlation(delayed, reference)
    np.testing.assert_allclose(values[lags == 2], 1.0)
    assert np.all(np.abs(values) <= 1.0 + 1e-12)


def test_normalized_bounded_for_random_real_and_complex():
    rng = np.random.default_rng(7)
    _, real_rho = normalized_cross_correlation(
        rng.standard_normal(64), rng.standard_normal(64)
    )
    assert np.all(np.abs(real_rho) <= 1.0 + 1e-12)
    assert real_rho.dtype == np.float64
    _, complex_rho = normalized_cross_correlation(
        rng.standard_normal(64) + 1j * rng.standard_normal(64),
        rng.standard_normal(64) + 1j * rng.standard_normal(64),
    )
    assert np.all(np.abs(complex_rho) <= 1.0 + 1e-12)
    assert complex_rho.dtype == np.complex128


def test_normalized_rejects_zero_energy_inputs():
    with pytest.raises(ValueError, match="nonzero energy"):
        normalized_cross_correlation(np.zeros(4), np.ones(3))
    with pytest.raises(ValueError, match="nonzero energy"):
        normalized_cross_correlation(np.ones(3), np.zeros(4))


def test_normalized_zero_energy_overlaps_are_zero_and_finite():
    lags, values = normalized_cross_correlation(
        np.array([1.0, 0.0, 2.0]), np.array([0.0, 1.0])
    )
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values, [1.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("function", FUNCTIONS)
def test_cross_correlation_rejects_invalid_inputs(function):
    for bad in (
        np.array([]),
        np.ones((4, 4)),
        np.array([1.0, np.nan]),
        np.array([1.0 + np.inf * 1j, 2.0 + 1j]),
        np.array(["a", "b"]),
    ):
        with pytest.raises(ValueError):
            function(bad, np.ones(4))
        with pytest.raises(ValueError):
            function(np.ones(4), bad)
