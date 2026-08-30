"""Raw headerless interleaved IQ file ingestion utilities."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["load_raw_iq"]


def load_raw_iq(
    path,
    dtype: np.dtype | str | type = np.float32,
    iq_order: str = "IQ",
) -> npt.NDArray[np.complex128]:
    """Load a headerless raw file of interleaved scalar I and Q values.

    The file is read as a flat array of ``dtype`` scalars, de-interleaved
    according to ``iq_order`` (``I0 Q0 I1 Q1 ...`` or
    ``Q0 I0 Q1 I1 ...``), and combined into complex IQ samples. The caller
    must explicitly know the file's dtype and IQ order; nothing is inferred
    or normalized.

    Args:
        path: Filesystem path of the raw file.
        dtype: Real numeric scalar dtype of the stored values. Complex
            dtypes are rejected.
        iq_order: Either ``"IQ"`` or ``"QI"``.

    Returns:
        A 1-D complex128 array of length ``scalar_count // 2``.

    Raises:
        ValueError: If ``path`` is missing or not a file, ``dtype`` is not
            a real numeric scalar dtype, ``iq_order`` is invalid, the file
            is empty, or the scalar count is odd.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"Path does not point to an existing file: {str(path)!r}."
        )
    try:
        dtype = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"Unsupported dtype: {dtype!r}.") from exc
    if dtype.kind not in "fiu":
        raise ValueError(
            f"dtype must be a real numeric scalar dtype, got {dtype!r}."
        )
    if iq_order not in ("IQ", "QI"):
        raise ValueError(
            f'iq_order must be exactly "IQ" or "QI", got {iq_order!r}.'
        )
    raw = np.fromfile(path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"Raw IQ file is empty: {str(path)!r}.")
    if raw.size % 2 != 0:
        raise ValueError(
            f"Raw scalar count {raw.size} is odd; interleaved IQ requires "
            f"an even count."
        )
    if iq_order == "IQ":
        i_values = raw[0::2]
        q_values = raw[1::2]
    else:
        q_values = raw[0::2]
        i_values = raw[1::2]
    iq = i_values + 1j * q_values
    return iq.astype(np.complex128, copy=False)
