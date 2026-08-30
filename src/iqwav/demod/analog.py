"""Analog demodulation utilities."""

import numpy as np
import numpy.typing as npt

__all__ = ["fm_demodulate"]


def fm_demodulate(samples: np.ndarray) -> npt.NDArray[np.float64]:
    """FM phase discriminator for complex IQ samples.

    Computes the adjacent-sample complex phase difference

    ``angle(samples[1:] * conj(samples[:-1]))``

    which yields the FM-demodulated phase increment in radians per sample.
    No sample-rate scaling, DC removal, filtering, resampling, or
    normalization is applied.

    Args:
        samples: 1-D complex IQ sample array. Must contain only finite
            values and at least 2 samples.

    Returns:
        A 1-D float64 array of length ``len(samples) - 1`` with the
        per-sample phase increments in radians, each in ``(-pi, pi]``.

    Raises:
        ValueError: If ``samples`` is not one-dimensional, not
            complex-valued, contains non-finite values, or has fewer than
            2 samples.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if not np.iscomplexobj(samples):
        raise ValueError("samples must be complex-valued IQ data.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    if samples.shape[0] < 2:
        raise ValueError(
            f"samples must contain at least 2 samples, got {samples.shape[0]}."
        )
    return np.angle(samples[1:] * np.conj(samples[:-1]))
