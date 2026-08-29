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