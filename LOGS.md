# IQWAV Development Log

This file records meaningful development progress in chronological order.

It answers:

- What was done?
- What currently works?
- What was tested?
- What decisions were made?
- What remains incomplete?
- What should happen next?

Newest entries should be added at the top below this introduction.


---


## 2026-08-30 — Signal Power and AWGN Utilities Implemented

### Added

Created:

`src/iqwav/dsp/noise.py`

with:

- `signal_power(samples)`
- `add_awgn(samples, snr_db, rng=None)`

### Capability

IQWAV can now:

- compute average signal power using `mean(|x|^2)`,
- add controlled additive white Gaussian noise,
- generate real Gaussian noise for real signals,
- generate circular complex Gaussian noise for IQ signals,
- target a requested SNR in dB,
- reproduce noise deterministically using a seeded NumPy RNG.

### Tests

Added:

`tests/unit/test_noise.py`

Current total:

- 154 tests passing.

Tests verify:

- known signal powers,
- real and complex noise behavior,
- shape and dtype preservation,
- seeded reproducibility,
- measured SNR near requested values,
- invalid-input handling.

### Manual Verification

Compared clean and noisy IQ signals in the learning notebook.

Verified that:

- the waveform becomes visibly noisy,
- the desired tone remains present,
- the spectrum develops a noise floor around the tone.

### Current Capability

IQWAV now supports:

known signal generation
→ controlled noise injection
→ FFT / PSD / spectrogram analysis
→ FIR filtering.

### Next

Continue controlled channel-impairment utilities.


## 2026-08-30 — FIR Filtering Utilities Implemented

### Added

Created:

`src/iqwav/dsp/filters.py`

with:

- `design_lowpass_fir`
- `design_highpass_fir`
- `design_bandpass_fir`
- `apply_fir_filter`

### Capability

IQWAV can now design and apply basic FIR filters for real and complex signals.

Supported filter types:

- low-pass,
- high-pass,
- band-pass.

Filters are designed using `scipy.signal.firwin` and applied using `scipy.signal.lfilter`.

### Validation

Added validation for:

- sampling frequency,
- cutoff frequencies,
- band-pass edge ordering,
- FIR tap count,
- signal shape and finiteness,
- filter-tap shape and finiteness.

### Tests

Added:

`tests/unit/test_filters.py`

Current total:

- 141 tests passing.

Tests verify:

- valid FIR coefficient generation,
- low-pass behavior,
- high-pass behavior,
- band-pass behavior,
- real and complex signal support,
- output-length preservation,
- invalid-input handling.

### Manual Verification

Created a mixed signal containing low- and high-frequency tones and applied a low-pass FIR filter.

Verified in the spectrum that the low-frequency component remained while the high-frequency component was strongly attenuated.

### Current Capability

IQWAV now supports:

known signal generation
→ FFT/PSD/spectrogram analysis
→ basic FIR filtering.

### Next

Continue foundational DSP utilities.


## 2026-08-30 — Spectrogram / Waterfall Data Utility Implemented

### Added

Created:

`src/iqwav/dsp/spectrogram.py`

with:

`spectrogram_data(samples, fs, nperseg=256, noverlap=None)`

### Capability

IQWAV can now compute time-frequency power data for a signal.

The function returns:

- time axis in seconds,
- frequency axis in Hz,
- spectrogram power matrix in linear units.

The frequency axis is arranged as:

negative frequencies → 0 → positive frequencies.

The returned power matrix has shape:

`(number of frequency bins, number of time segments)`

This data will later support the GUI waterfall / spectrogram view required by the SIH problem statement.

### Validation

Added checks for:

- invalid/non-finite sampling frequency,
- non-1-D signals,
- empty signals,
- NaN/Inf samples,
- invalid `nperseg`,
- invalid `noverlap`.

### Tests

Added:

`tests/unit/test_spectrogram.py`

Current total:

- 108 tests passing.

Tests verify:

- output dimensions,
- increasing time axis,
- centered frequency ordering,
- correct signed frequency detection across time,
- invalid-input handling.

### Manual Verification

Used a stationary `-100 Hz` IQ tone and plotted the returned spectrogram data.

Observed a horizontal power ridge around `-100 Hz` across time, as expected for a constant-frequency signal.

### Current Capability

IQWAV now supports:

known tone generation
→ FFT magnitude
→ periodogram PSD
→ Welch PSD
→ spectrogram / waterfall data.

### Next

Continue foundational DSP utilities.


## 2026-08-30 — PSD Utilities Implemented

### Added

Created:

`src/iqwav/dsp/psd.py`

with:

- `periodogram_psd(samples, fs)`
- `welch_psd(samples, fs, nperseg=None)`

### Capability

IQWAV can now estimate power spectral density using:

- a standard periodogram,
- Welch averaged PSD.

Both functions return:

- frequency axis in Hz,
- PSD values in linear units.

Outputs are arranged as:

negative frequencies → 0 → positive frequencies.

### Validation

Added checks for:

- invalid/non-finite sampling frequency,
- non-1-D signals,
- empty signals,
- NaN/Inf samples,
- invalid `nperseg`.

### Tests

Added:

`tests/unit/test_psd.py`

Current total:

- 89 tests passing.

Tests verify:

- output size,
- centered frequency ordering,
- real-tone symmetric PSD peaks,
- signed complex-IQ peak location,
- default and explicit Welch segment lengths,
- invalid-input handling.

### Manual Verification

Used the learning notebook to compare periodogram and Welch PSD.

Observed:

- periodogram gives a sharper/taller peak for the clean synthetic tone,
- Welch gives a broader/smoother peak due to segment averaging.

### Current Capability

IQWAV now supports:

known tone generation
→ FFT magnitude analysis
→ periodogram PSD
→ Welch PSD.

### Next

Continue foundational DSP processing and visualization utilities.


## 2026-08-30 — FFT Magnitude Spectrum Utility Implemented

### Added

Created:

`src/iqwav/dsp/spectrum.py`

with:

`magnitude_spectrum(samples, fs, fftshift=True)`

### Capability

The function:

- accepts real or complex 1-D NumPy signal arrays,
- computes the FFT,
- computes raw FFT magnitude,
- generates the corresponding frequency axis in Hz,
- optionally applies FFT shift so frequency ordering becomes:

negative frequencies → 0 → positive frequencies.

### Validation

Added checks for:

- invalid or non-finite sampling frequency,
- non-1-D input,
- empty arrays,
- NaN or infinite samples.

### Tests

Added:

`tests/unit/test_spectrum.py`

Current total:

- 66 tests passing.

Spectrum tests verify:

- output length,
- correct frequency-axis construction,
- real-tone peaks at ±f,
- complex IQ tone peak at the correct signed frequency,
- shifted and unshifted FFT ordering,
- invalid input handling.

### Manual Verification

Used:

`notebooks/learning/01_tone_generation.ipynb`

to visually inspect the generated IQ spectrum and confirmed the expected spectral peak location.

### Current Capability

IQWAV now supports:

known synthetic tone generation
→ FFT spectrum analysis
→ frequency-domain verification.

### Next

Continue building foundational DSP analysis utilities.

## 2026-08-29 — Synthetic Tone Generator Implemented

### Added

Created:

`src/iqwav/modulation/tones.py`

with reusable generators for:

- real cosine tones,
- complex IQ tones.

Both support:

- sampling frequency,
- tone frequency,
- duration,
- amplitude,
- phase.

### Validation

Added checks for:

- invalid sampling frequency,
- invalid duration,
- negative amplitude,
- non-finite values,
- Nyquist violations,
- impossible sample counts,
- extreme `fs * duration` overflow.

For complex IQ tones, the exact Nyquist boundary is rejected because positive and negative frequency become indistinguishable there.

### Tests

Added:

`tests/unit/test_tones.py`

Current result:

- 52 tests passing.

Tests cover:

- sample count,
- time spacing,
- data types,
- amplitude,
- phase,
- known sample sequences,
- FFT frequency location,
- positive/negative IQ behavior,
- Nyquist policy,
- invalid inputs.

### Manual Verification

Created:

`notebooks/learning/01_tone_generation.ipynb`

and visually verified:

- real cosine waveform,
- I and Q components,
- circular IQ trajectory,
- opposite rotation for positive and negative IQ frequency.

### Current Capability

IQWAV can now generate deterministic real and complex synthetic tones with known ground-truth parameters for downstream DSP testing.

### Next

Implement reusable FFT/spectrum analysis utilities for known synthetic signals.



## 2026-08-29 — Modules 7–8 Completed and Python Environment Established

### Learning Progress

Completed:

- Module 7 — Digital Communication Fundamentals
- Module 8 — Digital Modulation

Current learning boundary is now Modules 0–8.

Next:

- Module 9 — Correlation & Statistical Signal Analysis

### Development Environment

Created project virtual environment using Python 3.11.

Installed initial dependencies:

- NumPy
- SciPy
- Matplotlib
- pytest

Configured the project using `pyproject.toml`.

Installed IQWAV in editable development mode using:

`pip install -e .`

Verified that:

`import iqwav`

resolves directly to:

`src/iqwav/`

### New Implementation Boundary

The project may now implement:

- synthetic digital-modulation generators,
- digital signal visualizations,
- constellation handling,
- pulse-shaping experiments,
- controlled demodulation where parameters are known,
- DSP infrastructure supported by Modules 0–8.

Blind-analysis functionality remains deferred until the relevant later modules are completed.

### Next

Create the first production milestone: controlled synthetic-signal and DSP foundation.



## 2026-08-29 — Repository Foundation

### Status

IQWAV project repository initialized.

### Completed

Created the initial directory architecture containing:

- configs
- data
- docs
- gnuradio
- models
- native
- notebooks
- outputs
- scripts
- src
- tests

Created the Python package structure under:

`src/iqwav/`

with initial subsystem directories for:

- io
- dsp
- modulation
- estimation
- synchronization
- amr
- demod
- interleaving
- fec
- correlation
- framing
- pipeline
- ui
- utils

Created:

- README.md
- LOGS.md
- .gitignore
- requirements.txt
- pyproject.toml

Added `.gitkeep` placeholders where required so important empty directories can exist in Git.

Configured `.gitignore` to prevent large/generated/local files from accidentally entering the repository.

Git repository initialized with:

- default branch: `main`

### Current Learning Position

Completed:

Modules 0 through 6.

Latest completed module:

Module 6 — Analog Modulation.

Next learning module:

Module 7 — Digital Communication Fundamentals.

### Current Product Capability

No production DSP processing components have yet been implemented.

The repository currently provides the software/project foundation only.

### Important Decision

IQWAV will be developed progressively alongside the learning curriculum rather than attempting to implement the complete SIH system immediately.

Development loop:

Theory → Experiment → Implementation → Test → Integrate.

### Next

Establish the Python development environment and choose the first production milestone supported by Modules 0–6.