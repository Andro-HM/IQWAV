"""Unit tests for WAV and raw IQ file ingestion in iqwav.io."""

import numpy as np
import pytest
from scipy.io import wavfile

from iqwav.io import load_raw_iq, load_wav, load_wav_iq


def test_load_wav_mono_round_trip(tmp_path):
    fs = 8000
    samples = np.array([100, -50, 25, 0, 999, -999], dtype=np.int16)
    path = tmp_path / "mono.wav"
    wavfile.write(path, fs, samples)
    loaded_fs, loaded = load_wav(path)
    assert loaded_fs == 8000
    assert isinstance(loaded_fs, int)
    assert loaded.dtype == np.int16
    assert loaded.shape == (6,)
    np.testing.assert_array_equal(loaded, samples)


def test_load_wav_stereo_round_trip(tmp_path):
    fs = 44100
    left = np.arange(-5, 5, dtype=np.int16)
    right = -left
    stereo = np.column_stack((left, right))
    path = tmp_path / "stereo.wav"
    wavfile.write(path, fs, stereo)
    loaded_fs, loaded = load_wav(path)
    assert loaded_fs == 44100
    assert loaded.dtype == np.int16
    assert loaded.shape == (10, 2)
    np.testing.assert_array_equal(loaded, stereo)


def test_load_wav_missing_or_directory_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_wav(tmp_path / "missing.wav")
    with pytest.raises(ValueError):
        load_wav(tmp_path)


def test_load_wav_invalid_wav_rejected(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(ValueError):
        load_wav(path)


def test_load_wav_iq_known_channels(tmp_path):
    i_data = np.array([10, -20, 30], dtype=np.int16)
    q_data = np.array([-5, 15, -25], dtype=np.int16)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 8000, np.column_stack((i_data, q_data)))
    loaded_fs, iq = load_wav_iq(path)
    assert loaded_fs == 8000
    assert iq.dtype == np.complex128
    assert iq.shape == (3,)
    np.testing.assert_array_equal(iq, i_data + 1j * q_data)


def test_load_wav_iq_channel_selection(tmp_path):
    tri = np.column_stack(
        (
            np.array([1, 2, 3], dtype=np.int16),
            np.array([4, 5, 6], dtype=np.int16),
            np.array([7, 8, 9], dtype=np.int16),
        )
    )
    path = tmp_path / "tri.wav"
    wavfile.write(path, 8000, tri)
    _, iq = load_wav_iq(path, i_channel=2, q_channel=0)
    np.testing.assert_array_equal(iq, tri[:, 2] + 1j * tri[:, 0])
    _, iq_swapped = load_wav_iq(path, i_channel=1, q_channel=2)
    np.testing.assert_array_equal(iq_swapped, tri[:, 1] + 1j * tri[:, 2])


def test_load_wav_iq_mono_rejected(tmp_path):
    path = tmp_path / "mono.wav"
    wavfile.write(path, 8000, np.zeros(6, dtype=np.int16))
    with pytest.raises(ValueError):
        load_wav_iq(path)


def test_load_wav_iq_invalid_channels_rejected(tmp_path):
    path = tmp_path / "stereo.wav"
    wavfile.write(
        path, 8000, np.zeros((6, 2), dtype=np.int16)
    )
    with pytest.raises(ValueError):
        load_wav_iq(path, i_channel=2, q_channel=0)
    with pytest.raises(ValueError):
        load_wav_iq(path, i_channel=-1, q_channel=0)
    with pytest.raises(ValueError):
        load_wav_iq(path, i_channel=0.5, q_channel=1)


def test_load_wav_iq_same_channel_rejected(tmp_path):
    path = tmp_path / "stereo.wav"
    wavfile.write(
        path, 8000, np.zeros((6, 2), dtype=np.int16)
    )
    with pytest.raises(ValueError):
        load_wav_iq(path, i_channel=1, q_channel=1)


def _write_interleaved(path, i_values, q_values, dtype, iq_order="IQ"):
    interleaved = np.empty(2 * len(i_values), dtype=dtype)
    if iq_order == "IQ":
        interleaved[0::2] = i_values
        interleaved[1::2] = q_values
    else:
        interleaved[0::2] = q_values
        interleaved[1::2] = i_values
    interleaved.tofile(path)


def test_load_raw_iq_float32_iq(tmp_path):
    i_values = np.array([1.5, -2.5, 3.25, 0.0], dtype=np.float32)
    q_values = np.array([-0.5, 4.0, -1.75, 2.5], dtype=np.float32)
    path = tmp_path / "data.iq"
    _write_interleaved(path, i_values, q_values, np.float32)
    iq = load_raw_iq(path)
    assert iq.dtype == np.complex128
    assert iq.shape == (4,)
    np.testing.assert_array_equal(iq, i_values + 1j * q_values)


def test_load_raw_iq_int16_iq(tmp_path):
    i_values = np.array([100, -200, 300, -400], dtype=np.int16)
    q_values = np.array([5, 10, -15, 20], dtype=np.int16)
    path = tmp_path / "data.iq"
    _write_interleaved(path, i_values, q_values, np.int16)
    iq = load_raw_iq(path, dtype=np.int16)
    assert iq.dtype == np.complex128
    np.testing.assert_array_equal(iq, i_values + 1j * q_values)


def test_load_raw_iq_qi_order(tmp_path):
    i_values = np.array([1.5, -2.5, 3.25], dtype=np.float32)
    q_values = np.array([-0.5, 4.0, -1.75], dtype=np.float32)
    path = tmp_path / "data_qi.iq"
    _write_interleaved(path, i_values, q_values, np.float32, iq_order="QI")
    iq = load_raw_iq(path, iq_order="QI")
    assert iq.dtype == np.complex128
    np.testing.assert_array_equal(iq, i_values + 1j * q_values)


def test_load_raw_iq_odd_count_rejected(tmp_path):
    path = tmp_path / "odd.iq"
    np.array([1.0, 2.0, 3.0], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError):
        load_raw_iq(path)


def test_load_raw_iq_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.iq"
    path.touch()
    with pytest.raises(ValueError):
        load_raw_iq(path)


def test_load_raw_iq_invalid_order_rejected(tmp_path):
    path = tmp_path / "data.iq"
    np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError):
        load_raw_iq(path, iq_order="iq")
    with pytest.raises(ValueError):
        load_raw_iq(path, iq_order="XY")


def test_load_raw_iq_complex_dtype_rejected(tmp_path):
    path = tmp_path / "data.iq"
    np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError):
        load_raw_iq(path, dtype=np.complex64)
    with pytest.raises(ValueError):
        load_raw_iq(path, dtype=np.complex128)


def test_load_raw_iq_missing_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_raw_iq(tmp_path / "missing.iq")
