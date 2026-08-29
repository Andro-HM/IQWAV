# IQWAV — SIH26147 Signal Analysis System

## 1. Project Identity

IQWAV is the implementation project for Smart India Hackathon 2026 Problem Statement SIH26147.

The objective is to build a GUI-based signal-analysis system capable of ingesting raw `.IQ` and `.wav` recordings, extracting signal parameters, demodulating supported signals, performing de-interleaving and FEC decoding, and analysing the resulting bitstream.

This repository is intended to remain understandable to:
- project team members,
- future contributors,
- ChatGPT,
- Claude,
- Gemini,
- Codex,
- coding agents,
- and other AI systems.

Any AI or contributor working on this project should read this file and `LOGS.md` before modifying the repository.

---

# 2. Official Problem

The problem concerns terrestrial RF signals captured from different sensors and locations in HF, VHF and UHF bands.

Recordings may be provided as `.IQ` or `.wav` files.

Because recordings may come from different sensors with different acquisition parameters, important characteristics such as:

- sampling frequency,
- modulation,
- FEC,
- interleaving,
- spectral structure,

may not be immediately known.

The required system should automate and improve this analysis.

---

# 3. Required Capabilities

The official problem statement requires the system to work toward the following capabilities.

## 3.1 Signal Parameter Identification

Identify or estimate properties such as:

- Sampling frequency
- Modulation type
- FEC
- Interleaving
- Additional useful signal parameters where feasible

Additional parameters developed by the project may include:

- occupied bandwidth,
- centre frequency,
- SNR,
- noise floor,
- carrier-frequency offset,
- symbol/baud rate,
- signal activity regions,
- confidence scores.

---

## 3.2 Signal Visualization

The GUI should expose useful signal representations including:

- Time-domain waveform
- Spectrum
- PSD
- Waterfall / spectrogram
- Constellation plot

---

## 3.3 Demodulation

Required modulation families include:

- FSK
- PSK
- QAM

The project may support additional modulation types where useful.

---

## 3.4 De-interleaving

Required target families include:

- Block
- Convolutional
- Diagonal
- Pseudo-random

---

## 3.5 Forward Error Correction

Required target families include:

- Short-constraint convolutional codes with Viterbi decoding
- Reed-Solomon block codes
- Concatenated codes
- LDPC

---

## 3.6 Bitstream Analysis

The recovered bitstream should eventually support:

- correlation,
- repeated-pattern detection,
- synchronization/header discovery,
- frame-boundary analysis,
- header/payload separation,
- payload presentation.

---

# 4. Intended End-to-End Pipeline

The long-term system is approximately:

IQ/WAV Input

→ File parsing and metadata handling

→ Signal visualization

→ Signal/activity detection

→ Spectrum and bandwidth analysis

→ Parameter estimation

→ Modulation recognition

→ Synchronization

→ Demodulation

→ Symbol-to-bit conversion

→ De-interleaving

→ FEC identification/decoding

→ Bitstream correlation

→ Framing/header detection

→ Payload recovery

→ GUI presentation

Not all stages currently exist.

The project is being implemented progressively as the required DSP and communication theory is learned.

---

# 5. Development Strategy

This project is NOT being built all at once.

The development loop is:

Theory
→ small experiment
→ implementation
→ testing
→ deliberate impairment/failure testing
→ integration
→ next topic

New production functionality should normally only be added once the underlying concept is sufficiently understood and testable.

Synthetic signals with known ground truth should be used before claiming that an algorithm works on unknown real-world recordings.

---

# 6. Current Learning State

Current completed curriculum:

- Module 0 — Mathematical revision
- Module 1 — Signals and systems fundamentals
- Module 2 — Complex signals and IQ
- Module 3 — Fourier analysis and spectrum
- Module 4 — Filtering
- Module 5 — Noise and channel effects
- Module 6 — Analog modulation
- Module 7 — Digital communication fundamentals
- Module 8 — Digital modulation

Current next module:

- Module 9 — Correlation & statistical signal analysis

Therefore the implementation may now include controlled digital-communication functionality supported by Modules 0–8.

Do NOT yet assume blind parameter estimation, synchronization, AMR, blind FEC or framing capability exists.

---

# 7. Repository Structure


## Current Directory Layout


## Current Directory Layout

```text
IQWAV/
│
├── README.md
├── LOGS.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── configs/
│
├── docs/
│   ├── architecture/
│   ├── decisions/
│   └── research/
│
├── data/
│   ├── raw/
│   ├── external/
│   ├── synthetic/
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   ├── learning/
│   └── experiments/
│
├── scripts/
│
├── src/
│   └── iqwav/
│       ├── io/
│       ├── dsp/
│       ├── modulation/
│       ├── estimation/
│       ├── synchronization/
│       ├── amr/
│       ├── demod/
│       ├── interleaving/
│       ├── fec/
│       ├── correlation/
│       ├── framing/
│       ├── pipeline/
│       ├── ui/
│       └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── models/
│   ├── checkpoints/
│   └── metadata/
│
├── outputs/
│   ├── plots/
│   ├── reports/
│   └── runs/
│
├── native/
│   └── cpp/
│
└── gnuradio/
```

## `src/iqwav/`

Contains production IQWAV code.

### `io/`
Input/output functionality.

Examples:
- IQ readers
- WAV readers
- datatype handling
- metadata handling

### `dsp/`
General DSP operations.

Examples:
- FFT
- PSD
- spectrograms
- filters
- resampling
- signal power
- noise estimation

### `modulation/`
Modulation-related functionality and signal generation.

### `estimation/`
Blind or semi-blind parameter estimation.

Future examples:
- bandwidth
- CFO
- SNR
- baud rate

### `synchronization/`
Carrier and symbol synchronization.

Future examples:
- Costas loop
- timing recovery

### `amr/`
Automatic Modulation Recognition.

### `demod/`
Demodulators.

Future targets include:
- FSK
- BPSK
- QPSK
- QAM

### `interleaving/`
Interleaver/de-interleaver functionality.

### `fec/`
Forward Error Correction functionality.

### `correlation/`
Bitstream and signal correlation functionality.

### `framing/`
Frame, header and payload recovery.

### `pipeline/`
Coordinates the complete end-to-end signal-processing pipeline.

### `ui/`
GUI-related code.

### `utils/`
Shared utilities that do not naturally belong to another subsystem.

---

# 8. Other Directories

## `data/raw/`
Local original recordings.

Large files must NOT normally be committed to Git.

## `data/external/`
Externally obtained datasets.

## `data/synthetic/`
Generated signals used for experiments and validation.

## `data/processed/`
Intermediate/generated datasets.

## `data/samples/`
Very small Git-safe sample signals useful for demonstrations and tests.

## `notebooks/learning/`
Learning-oriented experiments.

## `notebooks/experiments/`
Formal project experiments and algorithm investigations.

## `tests/unit/`
Tests for individual functions/modules.

## `tests/integration/`
Tests involving multiple components.

## `tests/fixtures/`
Small known-ground-truth inputs used by tests.

## `configs/`
Configuration files.

## `docs/architecture/`
System and software architecture documentation.

## `docs/research/`
Technical investigations and research notes.

## `docs/decisions/`
Important engineering decisions and their reasoning.

## `models/`
Machine-learning model information/checkpoints.

## `outputs/`
Generated plots, reports and experiment outputs.

## `native/cpp/`
Future C++ components if performance or integration requires them.

## `gnuradio/`
GNU Radio flowgraphs and experiments.

---

# 9. Source Code vs Experiments

Production functionality belongs in:

`src/iqwav/`

Notebooks are used for:

- learning,
- visualization,
- experimentation,
- prototyping,
- validating algorithms.

Once an algorithm is accepted, reusable implementation should move into `src/iqwav/`.

The notebook must not become the final application architecture.

---

# 10. Testing Principle

Whenever practical, algorithms should be tested against signals whose true parameters are already known.

Examples:

Known:
- sampling rate
- carrier frequency
- modulation
- SNR
- symbol rate
- CFO
- transmitted bits

Algorithm estimates the values.

Estimated values are then compared against ground truth.

Eventually the system should be evaluated using both synthetic and real recordings.

---

# 11. Data Policy

Do NOT commit huge IQ/WAV recordings directly to normal Git.

Large:
- raw RF captures,
- datasets,
- generated experiment outputs,
- ML checkpoints,

should normally remain outside Git or use an appropriate external storage mechanism.

Small deterministic test signals may be committed when useful.

---

# 12. AI / Contributor Handoff Protocol

Before making changes:

1. Read `README.md`.
2. Read the latest entries in `LOGS.md`.
3. Inspect relevant existing source files.
4. Do not assume unfinished functionality exists.
5. Do not rewrite working architecture without a clear reason.
6. Preserve compatibility with existing tests where possible.
7. Record meaningful completed work in `LOGS.md`.

When proposing major architectural changes, explain why the change is necessary before implementing it.

---

# 13. Current Development Status

Repository infrastructure has been created.

Git has been initialized locally.

No actual IQWAV processing functionality has yet been integrated into the production codebase.

The first implementation milestone will be chosen based on functionality supported by completed curriculum Modules 0–6.

For exact development history and current next action, see:

`LOGS.md`