"""Blind constant carrier-phase-offset estimation for IQWAV BPSK/QPSK."""

from dataclasses import dataclass

import numpy as np

from .occupied_band import _validate_real_scalar

__all__ = ["PhaseOffsetEstimate", "estimate_phase_offset"]


_M_TH_POWER_REFERENCE = {
    # modulation: (power M, canonical constellation reference c_M = s**M)
    "bpsk": (2, 1.0 + 0.0j),
    "qpsk": (4, -1.0 + 0.0j),
}


@dataclass(frozen=True)
class PhaseOffsetEstimate:
    """Blind static phase-offset estimate for a BPSK or QPSK signal.

    ``phase_offset_rad`` is the estimated constant carrier phase offset
    wrapped into the canonical range of the modulation, and ``symmetry``
    is the M-th-power symmetry reliability measure in ``[0, 1]``, where
    near 1 means highly concentrated M-th-power samples.
    """

    phase_offset_rad: float
    symmetry: float


def estimate_phase_offset(
    samples: np.ndarray,
    modulation: str,
    *,
    min_symmetry: float = 0.05,
) -> PhaseOffsetEstimate:
    """Estimate a blind constant carrier phase offset of BPSK/QPSK IQ data.

    For a static phase rotation ``x[n] = s[n] * exp(j*phi)`` with unknown
    ``phi``, raising to the M-th power removes the unknown data because
    the canonical IQWAV constellations satisfy a fixed ``s**M`` relation:

    - BPSK (M = 2): symbols are ``+1`` and ``-1``, so ``s**2 = +1`` and
      the canonical reference is ``c_M = +1``.
    - QPSK (M = 4): the Gray mapper places every symbol at phase
      ``pi/4 + k*pi/2``, so every ideal symbol satisfies ``s**4 = -1``
      and the canonical reference is ``c_M = -1``.

    The M-th-power moment of the received signal is
    ``mean(x**M) = c_M * exp(j*M*phi) * magnitude``, so the known
    constellation reference is compensated before taking the angle::

        compensated_moment = mean(x**M) * conj(c_M)
        phi_hat            = angle(compensated_moment) / M

    For QPSK this is exactly equivalent to estimating the phase from
    ``-mean(x**4)``. Skipping the ``c_M`` compensation would leave a fixed
    ``pi/4`` constellation-orientation bias in every QPSK estimate.

    The raw estimate is then wrapped deterministically into the canonical
    range ``[-pi/2, +pi/2)`` for BPSK and ``[-pi/4, +pi/4)`` for QPSK, so
    the boundary behaviour is explicit and does not depend accidentally
    on ``np.angle``'s branch choice at ``+/-pi``.

    Rotational ambiguity:
        Blind PSK phase is intrinsically observable only modulo ``pi``
        (BPSK) or modulo ``pi/2`` (QPSK), because rotating the whole
        constellation by ``pi`` (BPSK) or by ``k*pi/2`` (QPSK) maps it
        onto itself. The returned phase alone therefore cannot recover
        absolute bit labeling; pilots, differential coding, framing or
        known headers are needed to resolve that ambiguity.

    Reliability measure::

        symmetry = abs(mean(x**M)) / mean(abs(x)**M)

    computed on magnitude-normalized samples; the positive real
    normalization leaves the phase and this ratio unchanged while
    avoiding overflow/underflow. It equals 1 when all ``x**M`` share a
    single phase and tends to 0 for structureless input. It is a
    bounded symmetry/concentration reliability measure only: it is NOT
    a calibrated confidence or probability and NOT an SNR estimate. If
    ``symmetry < min_symmetry`` the estimate is rejected with
    ``ValueError``; ``min_symmetry`` is an algorithmic reliability
    threshold only, not a calibrated confidence bound.

    Scope and assumptions:
        - The modulation family ("bpsk" or "qpsk") must already be known;
          the M-th-power method assumes it.
        - The phase offset must be constant over the whole observation
          block. A residual carrier-frequency offset that rotates the
          constellation significantly across the observation degrades
          the symmetry and biases the estimate; this is NOT carrier
          tracking and no tracking is attempted.
        - This is NOT a Costas loop, NOT a PLL, NOT timing recovery, NOT
          CFO estimation, and NOT AMR.
        - The estimator operates directly on the supplied 1-D complex
          samples: it does not require ``samples_per_symbol``, known
          symbol timing, or internal symbol-boundary recovery. It works
          on the current rectangular oversampled BPSK/QPSK waveforms and
          equally on already-extracted symbol-rate samples.
        - Moderate SNR is expected; severe noise gives low symmetry and
          possibly rejection.

    Args:
        samples: 1-D complex finite samples with at least 1 value and
            non-zero energy. Real-only input is rejected because the
            phase information is not preserved.
        modulation: Supported modulation family, exactly ``"bpsk"`` or
            ``"qpsk"``. Arbitrary M-PSK support is deliberately not
            advertised.
        min_symmetry: Minimum accepted M-th-power symmetry. Must be a
            finite real scalar in ``[0, 1]``; 0 is allowed for
            diagnostic use.

    Returns:
        The :class:`PhaseOffsetEstimate`.

    Raises:
        ValueError: If any argument is invalid, the input has zero
            energy, or ``symmetry < min_symmetry``.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be one-dimensional, got shape {samples.shape}."
        )
    if samples.dtype.kind != "c":
        raise ValueError(
            f"samples must be complex-valued IQ data, got dtype {samples.dtype}."
        )
    if samples.shape[0] < 1:
        raise ValueError(
            f"samples must contain at least 1 sample, got {samples.shape[0]}."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values.")
    if modulation not in _M_TH_POWER_REFERENCE:
        raise ValueError(
            f"modulation must be one of {tuple(_M_TH_POWER_REFERENCE)}, "
            f"got {modulation!r}."
        )
    symmetry_min = _validate_real_scalar(min_symmetry, "min_symmetry")
    if not 0.0 <= symmetry_min <= 1.0:
        raise ValueError(
            f"min_symmetry must satisfy 0 <= min_symmetry <= 1, "
            f"got {min_symmetry!r}."
        )

    order, reference = _M_TH_POWER_REFERENCE[modulation]
    scale = float(np.mean(np.abs(samples)))
    if scale == 0.0:
        raise ValueError(
            "samples have zero energy; cannot estimate a phase offset."
        )
    normalized = samples / scale
    moment = np.mean(normalized**order)
    compensated_moment = moment * np.conj(reference)
    phi_raw = float(np.angle(compensated_moment)) / order
    half_period = np.pi / order
    phase_offset_rad = (
        (phi_raw + half_period) % (2.0 * half_period) - half_period
    )
    symmetry = float(
        np.abs(moment) / np.mean(np.abs(normalized) ** order)
    )
    if symmetry < symmetry_min:
        raise ValueError(
            f"M-th-power symmetry {symmetry:.4f} is below min_symmetry "
            f"{min_symmetry!r}; no reliable static phase-offset estimate "
            f"is available."
        )
    return PhaseOffsetEstimate(
        phase_offset_rad=float(phase_offset_rad),
        symmetry=symmetry,
    )
