import numpy as np
import pytest

from iqwav.detection import detect_activity


def test_activity_bursts_merge_actual_sample_gap_and_weighted_final_window():
    x = np.zeros(23, dtype=np.complex128)
    x[4:8] = 2.0
    x[12:23] = 3.0
    found = detect_activity(x, fs=10.0, window_size=4, relative_threshold_db=1.0, merge_gap_samples=4)
    assert [(r.start_sample, r.end_sample) for r in found.regions] == [(4, 23)]
    final = np.zeros(23, dtype=np.complex128)
    final[12:23] = 3.0
    last = detect_activity(final, fs=10.0, window_size=4, relative_threshold_db=1.0)
    assert last.regions[-1].average_power == pytest.approx(9.0)
    assert last.regions[-1].duration_s == pytest.approx(1.1)


def test_activity_constant_and_zero_input_produce_no_guaranteed_activity():
    assert not detect_activity(np.ones(128, dtype=np.complex128), window_size=16).regions
    assert not detect_activity(np.zeros(128, dtype=np.complex128), window_size=16).regions


def test_activity_is_deterministic_non_mutating_and_validates():
    x = np.zeros(64, dtype=np.complex128); x[16:32] = 2
    before = x.copy()
    assert detect_activity(x) == detect_activity(x)
    assert np.array_equal(x, before)
    with pytest.raises(ValueError): detect_activity(np.ones((2, 2)))
    with pytest.raises(ValueError): detect_activity(np.ones(4), window_size=0)
