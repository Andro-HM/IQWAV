"""Signal power and AWGN channel utilities."""

import math

import numpy as np
import numpy.typing as npt

__all__ = ["add_awgn", "signal_power"]


def signal_power(samples: np.ndarray) -> float:
    """Compute the mean power of a 1-D signal.

    Args:
        samples: 1-D real or complex sample array. Must be non-empty with
            only finite values.

    Returns:
        The mean power ``mean(|x|^2)`` as a Python float.

    Raises:
        ValueError: If ``samples`` is not a non-empty 1-D array of finite
            values.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.size == 0:
        raise ValueError("samples must contain at least one value.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    return float(np.mean(np.abs(samples) ** 2))


def add_awgn(
    samples: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Add additive white Gaussian noise at a requested SNR.

    The target noise power is ``signal_power / 10**(snr_db / 10)`` so the
    requested SNR is achieved in expectation. Real input gets real
    Gaussian noise; complex input gets circular complex Gaussian noise
    with the total noise power split equally between I and Q. A
    zero-power signal therefore yields zero noise.

    Args:
        samples: 1-D real or complex sample array. Must be non-empty with
            only finite values.
        snr_db: Desired signal-to-noise ratio in dB. Must be finite.
        rng: Optional ``numpy.random.Generator`` for reproducible noise.
            If None, a new ``numpy.random.default_rng()`` is created.

    Returns:
        The noisy samples with the same shape; real input stays real
        (float64), complex input stays complex (complex128).

    Raises:
        ValueError: If ``snr_db`` or ``rng`` is invalid, or ``samples`` is
            not a non-empty 1-D array of finite values.
    """
    if not math.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}.")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}."
        )
    if rng is None:
        rng = np.random.default_rng()
    samples = np.asarray(samples)
    power = signal_power(samples)
    noise_power = power / 10.0 ** (snr_db / 10.0)
    if np.iscomplexobj(samples):
        sigma = math.sqrt(noise_power / 2.0)
        noise = rng.normal(0.0, sigma, samples.shape) + 1j * rng.normal(
            0.0, sigma, samples.shape
        )
    else:
        sigma = math.sqrt(noise_power)
        noise = rng.normal(0.0, sigma, samples.shape)
    return samples + noise
