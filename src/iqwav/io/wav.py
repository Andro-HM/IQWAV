"""WAV file ingestion utilities."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

__all__ = ["load_wav", "load_wav_iq"]


def _validate_file_path(path) -> Path:
    """Validate that path points to an existing file and return it."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"Path does not point to an existing file: {str(path)!r}."
        )
    return path


def load_wav(path) -> tuple[int, np.ndarray]:
    """Load samples from a WAV file.

    Args:
        path: Filesystem path of the WAV file.

    Returns:
        A tuple ``(fs, samples)``: the integer sampling rate in Hz and the
        sample array exactly as SciPy reads it (shape ``(N,)`` for mono,
        ``(N, channels)`` for multi-channel). Values and dtype are
        preserved without amplitude normalization.

    Raises:
        ValueError: If ``path`` is missing, is not a file, or is not a
            readable valid WAV file.
    """
    path = _validate_file_path(path)
    try:
        fs, samples = wavfile.read(path)
    except Exception as exc:
        raise ValueError(
            f"Could not read WAV file {str(path)!r}: {exc}"
        ) from exc
    return int(fs), samples


def load_wav_iq(
    path,
    i_channel: int = 0,
    q_channel: int = 1,
) -> tuple[int, npt.NDArray[np.complex128]]:
    """Load two WAV channels as complex IQ data.

    Args:
        path: Filesystem path of the WAV file.
        i_channel: Index of the in-phase channel. Must be an integer in
            ``[0, channels)``.
        q_channel: Index of the quadrature channel. Must be an integer in
            ``[0, channels)`` and differ from ``i_channel``.

    Returns:
        A tuple ``(fs, iq)``: the integer sampling rate in Hz and the 1-D
        complex128 array ``samples[:, i_channel] +
        1j * samples[:, q_channel]``. Relative amplitudes are preserved;
        no normalization is applied.

    Raises:
        ValueError: If the WAV has fewer than 2 channels, a channel index
            is invalid, or both indices select the same channel.
    """
    fs, samples = load_wav(path)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError(
            f"WAV must contain at least 2 channels to build IQ data, "
            f"got shape {samples.shape}."
        )
    n_channels = samples.shape[1]
    for name, channel in (("i_channel", i_channel), ("q_channel", q_channel)):
        if (
            not isinstance(channel, (int, np.integer))
            or not 0 <= channel < n_channels
        ):
            raise ValueError(
                f"{name} must be an integer in [0, {n_channels - 1}], "
                f"got {channel!r}."
            )
    if i_channel == q_channel:
        raise ValueError(
            f"i_channel and q_channel must differ, both are {i_channel!r}."
        )
    iq = samples[:, i_channel] + 1j * samples[:, q_channel]
    return fs, iq.astype(np.complex128, copy=False)
