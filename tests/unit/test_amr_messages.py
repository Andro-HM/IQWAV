"""Tests for harness-only analog message generators."""

import numpy as np
import pytest

from iqwav.amr import MESSAGE_FAMILIES, generate_analog_message


def test_families_are_shared_catalog():
    assert MESSAGE_FAMILIES == ("tone", "multi_tone", "bandlimited")


@pytest.mark.parametrize("family", MESSAGE_FAMILIES)
def test_message_is_real_normalized_and_finite(family):
    message, freqs = generate_analog_message(
        128, 8_000.0, family, np.random.default_rng(3), (0.02, 0.08)
    )
    assert message.shape == (128,)
    assert message.dtype == np.float64
    assert np.isrealobj(message)
    assert np.all(np.isfinite(message))
    assert np.max(np.abs(message)) <= 1.0 + 1e-15
    assert freqs
    assert all(0.02 <= value <= 0.08 for value in freqs)


def test_tone_and_multi_tone_and_bandlimited_are_distinct():
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)
    rng_c = np.random.default_rng(9)
    tone, _ = generate_analog_message(256, 8_000.0, "tone", rng_a, (0.02, 0.06))
    multi, _ = generate_analog_message(
        256, 8_000.0, "multi_tone", rng_b, (0.02, 0.06)
    )
    bandlimited, _ = generate_analog_message(
        256, 8_000.0, "bandlimited", rng_c, (0.02, 0.06)
    )
    assert not np.allclose(tone, multi)
    assert not np.allclose(tone, bandlimited)
    assert not np.allclose(multi, bandlimited)


def test_same_rng_seed_is_deterministic():
    first, freqs_a = generate_analog_message(
        64, 8_000.0, "bandlimited", np.random.default_rng(21), (0.03, 0.07)
    )
    second, freqs_b = generate_analog_message(
        64, 8_000.0, "bandlimited", np.random.default_rng(21), (0.03, 0.07)
    )
    np.testing.assert_array_equal(first, second)
    assert freqs_a == freqs_b


def test_invalid_family_rejected():
    with pytest.raises(ValueError):
        generate_analog_message(
            16, 8_000.0, "ssb", np.random.default_rng(0), (0.02, 0.08)
        )


@pytest.mark.parametrize("freq_norm_range", [(0.0, 0.1), (0.2, 0.1), (0.1, 0.6)])
def test_invalid_freq_range_rejected(freq_norm_range):
    with pytest.raises(ValueError):
        generate_analog_message(
            16, 8_000.0, "tone", np.random.default_rng(0), freq_norm_range
        )
