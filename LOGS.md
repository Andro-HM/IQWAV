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