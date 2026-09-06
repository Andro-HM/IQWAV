import numpy as np

from iqwav.pipeline import survey_capture


def test_survey_is_spectral_even_when_activity_is_empty_and_preserves_coordinates():
    fs = 1000.0; n = 4096
    x = np.exp(2j * np.pi * 150 * np.arange(n) / fs)
    result = survey_capture(x, fs, nperseg=512, threshold_db=6, min_bins=2, include_activity=True, extract=True, numtaps=51)
    assert not result.activity.regions
    assert result.bands
    assert all(region.start_sample == 0 and region.end_sample == n for region in result.regions)
    assert all(region.status in ("extracted", "failed") for region in result.regions)
