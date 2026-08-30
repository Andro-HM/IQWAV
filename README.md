# IQWAV — SIH26147 Signal Analysis System

## 1. Project Identity

IQWAV is the implementation project for Smart India Hackathon 2026 Problem Statement **SIH26147**.

The long-term objective is to build a GUI-based signal-analysis system capable of ingesting raw `.IQ` and `.wav` recordings, extracting useful signal parameters, identifying and demodulating supported modulation families, performing de-interleaving and FEC decoding, analysing recovered bitstreams, and presenting the results clearly to an operator.

This repository is intended to remain understandable to project team members, future contributors, and AI/coding agents.

Any AI or contributor working on this project should read this file and the newest entries in `LOGS.md` before modifying the repository.

---

## 2. Official Problem Context

The problem concerns terrestrial RF signals captured from different sensors and locations in HF, VHF and UHF bands.

Recordings may be supplied as `.IQ` or `.wav` files.

Because captures may originate from different sensors, locations and acquisition configurations, important characteristics may not be immediately known or may vary between files. These include:

- sampling frequency,
- modulation type,
- FEC,
- interleaving,
- spectral structure,
- and other parameters required for interpretation.

The required system should automate and improve this analysis.

---

## 3. Target Capabilities

### 3.1 File Ingestion

Target support includes:

- standard WAV,
- multi-channel WAV used explicitly as I/Q,
- raw interleaved IQ,
- and later, where useful, metadata-aware RF recording formats.

Current raw-IQ ingestion is explicit rather than blind: datatype, I/Q order and acquisition metadata must presently be supplied by the operator or external metadata.

### 3.2 Signal Parameter Identification

Target parameters include:

- sampling frequency,
- occupied bandwidth,
- centre/carrier frequency,
- carrier-frequency offset,
- SNR,
- noise floor,
- symbol/baud rate,
- modulation type,
- FEC,
- interleaving,
- signal activity regions,
- confidence scores.

Not all of these are implemented yet.

### 3.3 Signal Visualization

The eventual GUI should expose:

- time-domain waveform,
- FFT spectrum,
- PSD,
- waterfall / spectrogram,
- constellation,
- recovered bitstream,
- framing/header information,
- recovered payload where possible.

### 3.4 Demodulation

Required or planned digital families include:

- FSK,
- PSK,
- QAM.

The project also includes analog-FM demodulation support because it is useful for validating the RF/IQ processing foundation on genuine OTA data.

### 3.5 De-interleaving

Target families include:

- block,
- convolutional,
- diagonal,
- pseudo-random.

### 3.6 Forward Error Correction

Target families include:

- short-constraint convolutional codes with Viterbi decoding,
- Reed-Solomon,
- concatenated codes,
- LDPC.

### 3.7 Bitstream Analysis

Recovered bitstreams should eventually support:

- correlation,
- repeated-pattern detection,
- synchronization/preamble discovery,
- frame-boundary analysis,
- header/payload separation,
- payload presentation.

---

## 4. Intended End-to-End Pipeline

```text
IQ / WAV input
    ↓
file parsing + metadata handling
    ↓
signal visualization
    ↓
signal/activity detection
    ↓
spectrum + occupied-bandwidth analysis
    ↓
parameter estimation
    ↓
automatic modulation recognition
    ↓
carrier / timing synchronization
    ↓
demodulation
    ↓
symbol-to-bit conversion
    ↓
de-interleaving
    ↓
FEC identification / decoding
    ↓
bitstream correlation
    ↓
framing / header detection
    ↓
payload recovery
    ↓
GUI presentation
```

Not all stages currently exist.

The project is being implemented progressively as the required DSP and communications theory is learned and verified.

---

## 5. Development Strategy

IQWAV is deliberately not being built all at once.

Preferred development loop:

```text
theory
→ small experiment
→ bounded implementation
→ focused tests
→ full regression test
→ deliberate impairment/failure testing
→ manual verification
→ real-data validation where appropriate
→ integration
→ documentation
→ next topic
```

Engineering rules:

1. Reusable production functionality belongs in `src/iqwav/`.
2. Learning and exploratory work belongs in notebooks.
3. Synthetic signals with known ground truth should be used before claiming algorithmic correctness.
4. Real RF captures should be used to expose assumptions and validate integration.
5. Passing synthetic tests alone must not be treated as proof of real-world performance.
6. New blind-estimation or receiver functionality should not be assumed until explicitly implemented and tested.
7. Large recordings and generated outputs should remain outside normal Git history.

---

## 6. Current Learning Boundary

Completed curriculum currently covers:

- Module 0 — Mathematical revision
- Module 1 — Signals and systems fundamentals
- Module 2 — Complex signals and IQ
- Module 3 — Fourier analysis and spectrum
- Module 4 — Filtering
- Module 5 — Noise and channel effects
- Module 6 — Analog modulation
- Module 7 — Digital communication fundamentals
- Module 8 — Digital modulation

The current implementation boundary should remain grounded in concepts supported by **Modules 0–8** unless the learning state is explicitly updated later.

Do not assume the following have already been learned or implemented merely because directories exist for them:

- correlation/statistical signal analysis,
- blind parameter estimation,
- synchronization,
- AMR,
- blind FEC,
- framing/payload recovery.

---

## 7. Repository Structure

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

### `src/iqwav/io/`

Current responsibilities:

- WAV loading,
- explicit WAV I/Q conversion,
- raw interleaved IQ loading.

### `src/iqwav/dsp/`

Current functionality:

- FFT magnitude spectrum,
- periodogram PSD,
- Welch PSD,
- spectrogram/waterfall data,
- FIR filtering,
- signal power,
- AWGN injection,
- controlled phase/frequency impairment injection.

### `src/iqwav/modulation/`

Current functionality:

- real tones,
- complex IQ tones,
- BPSK mapping,
- QPSK Gray mapping,
- rectangular sampled baseband waveforms.

### `src/iqwav/demod/`

Current functionality:

- known-timing hard-decision BPSK demodulation,
- known-timing hard-decision QPSK demodulation,
- FM phase-discriminator demodulation.

### `src/iqwav/estimation/`

Reserved for future blind or semi-blind estimation such as:

- occupied bandwidth,
- carrier/CFO,
- SNR/noise floor,
- baud/symbol rate.

### `src/iqwav/synchronization/`

Reserved for future carrier and symbol synchronization.

### `src/iqwav/amr/`

Reserved for Automatic Modulation Recognition.

### `src/iqwav/interleaving/`

Reserved for interleaver/de-interleaver functionality.

### `src/iqwav/fec/`

Reserved for FEC identification/decoding.

### `src/iqwav/correlation/`

Reserved for signal/bitstream correlation.

### `src/iqwav/framing/`

Reserved for frame, synchronization-word, header and payload recovery.

### `src/iqwav/pipeline/`

Reserved for the eventual end-to-end processing pipeline.

### `src/iqwav/ui/`

Reserved for GUI functionality.

---

## 8. Data and Supporting Directories

### `data/raw/`

Local original recordings. Large captures should not normally be committed.

### `data/external/`

Externally obtained recordings and datasets. Current real OTA validation data is stored here locally and excluded from Git.

### `data/synthetic/`

Generated signals used for experiments and validation.

### `data/processed/`

Intermediate/generated datasets.

### `data/samples/`

Very small Git-safe samples useful for tests or demonstrations.

### `notebooks/learning/`

Current learning notebooks:

- `01_tone_generation.ipynb`
- `02_digital_communication.ipynb`
- `03_file_io_and_signal_analysis.ipynb`

### `notebooks/experiments/`

Current formal experiments:

- `01_real_iq_smoke_test.ipynb`
- `02_real_fm_demodulation.ipynb`

### `tests/unit/`

Focused tests for individual production functions/modules.

### `tests/integration/`

Tests involving multiple components.

### `tests/fixtures/`

Small deterministic known-ground-truth test inputs.

### `docs/architecture/`, `docs/research/`, `docs/decisions/`

Architecture, research notes and engineering decisions.

### `outputs/`

Generated plots, reports, run artifacts and temporary outputs.

### `models/`

Future model metadata/checkpoints if learning-based components are introduced.

### `native/cpp/`

Future C++ components if performance/integration requires them.

### `gnuradio/`

GNU Radio flowgraphs and experiments.

---

## 9. Current Production Functionality

### 9.1 Synthetic Tone Generation

Implemented:

- `generate_real_tone(...)`
- `generate_iq_tone(...)`

Supports deterministic real and complex-IQ tones with amplitude/phase control and validation.

### 9.2 Spectrum and PSD

Implemented:

- `magnitude_spectrum(...)`
- `periodogram_psd(...)`
- `welch_psd(...)`

Supports real/complex 1-D input, centered two-sided frequency axes and signed complex-IQ frequency analysis.

### 9.3 Spectrogram / Waterfall

Implemented:

- `spectrogram_data(...)`

Returns time, centered frequency and linear time-frequency power suitable for plotting and future GUI use.

### 9.4 FIR Filtering

Implemented:

- `design_lowpass_fir(...)`
- `design_highpass_fir(...)`
- `design_bandpass_fir(...)`
- `apply_fir_filter(...)`

Supports basic FIR filtering for real and complex signals.

### 9.5 Signal Power and AWGN

Implemented:

- `signal_power(...)`
- `add_awgn(...)`

Supports average power measurement, real Gaussian noise, circular complex Gaussian noise and seeded reproducibility.

### 9.6 Controlled IQ Impairments

Implemented:

- frequency-offset injection,
- phase-offset injection.

These utilities inject known impairments for controlled experiments. They do not estimate or correct unknown impairments.

### 9.7 BPSK / QPSK Modulation

Implemented:

- BPSK mapping,
- Gray-coded QPSK mapping,
- rectangular sampled waveform generation.

BPSK:

```text
0 → +1
1 → -1
```

QPSK Gray mapping:

```text
00 → (+1,+1)/√2
01 → (-1,+1)/√2
11 → (-1,-1)/√2
10 → (+1,-1)/√2
```

### 9.8 Known-Timing BPSK / QPSK Demodulation

Implemented:

- `bpsk_demodulate(...)`
- `qpsk_demodulate(...)`

Current assumptions:

- symbol boundaries are known,
- samples-per-symbol is supplied,
- no timing recovery,
- no carrier recovery,
- no CFO correction,
- no phase correction.

These are controlled hard-decision receivers, not blind receivers.

### 9.9 WAV and Raw IQ Ingestion

Implemented:

- `load_wav(path)`
- `load_wav_iq(path, i_channel=0, q_channel=1)`
- `load_raw_iq(path, dtype=np.float32, iq_order="IQ")`

Raw IQ ingestion deliberately does not infer:

- datatype,
- endianness,
- I/Q ordering,
- sample rate,
- center frequency.

### 9.10 FM Phase-Discriminator Demodulation

Implemented:

- `fm_demodulate(samples)`

The production discriminator computes:

```text
angle(samples[1:] * conj(samples[:-1]))
```

and returns adjacent phase increments in **radians/sample**.

It intentionally does not:

- multiply by `Fs/(2π)`,
- remove DC,
- low-pass filter,
- resample,
- normalize,
- perform de-emphasis,
- decode stereo,
- estimate or correct carrier/CFO.

Those higher-level steps remain outside the reusable phase discriminator.

---

## 10. Real-World Validation

IQWAV has now been tested on genuine over-the-air SDR data, not only synthetic signals.

### 10.1 Validation Capture

```text
fm_rds_250k_1Msamples.iq
```

Known metadata:

- genuine OTA broadcast-FM recording,
- center frequency: 99.5 MHz,
- sample rate: 250 kHz,
- format: complex64 / interleaved float32 I,Q,
- 1,000,000 complex samples,
- duration: approximately 4 seconds.

The recording is stored locally under `data/external/` and excluded from Git.

### 10.2 Real-IQ Smoke Test

Production path exercised:

```text
real OTA IQ file
    ↓
load_raw_iq()
    ↓
complex IQ samples
    ↓
magnitude_spectrum()
    ↓
welch_psd()
    ↓
spectrogram_data()
```

Observed:

- loader output matched direct NumPy complex64 loading,
- time-domain I/Q was physically plausible,
- FFT showed broad broadcast-FM occupancy,
- Welch PSD showed consistent real-world spectral structure,
- waterfall showed sensible time-frequency behavior,
- no obvious axis corruption, clipping or file-format mismatch was observed.

### 10.3 Real FM Demodulation

The same capture was then processed as:

```text
real OTA IQ
    ↓
fm_demodulate()
    ↓
FM multiplex/baseband
    ↓
experiment-level low-pass filtering
    ↓
DC removal
    ↓
resampling
    ↓
WAV output
```

The resulting WAV contained approximately **4 seconds of clean, clearly intelligible English broadcast audio**.

This is an important integration milestone: IQWAV did not merely visualize a real RF recording; it recovered human-audible information from genuine OTA complex IQ data.

---

## 11. Testing Status

Current full test suite:

```text
279 tests passing
```

Coverage includes:

- tone generation,
- FFT spectrum,
- periodogram/Welch PSD,
- spectrogram,
- FIR filtering,
- signal power,
- AWGN,
- IQ impairments,
- BPSK/QPSK modulation,
- sampled waveforms,
- known-timing BPSK/QPSK demodulation,
- WAV/raw-IQ ingestion,
- FM phase-discriminator demodulation.

FM-specific tests cover:

- output shape,
- float64 output,
- positive and negative phase increments,
- wrapped phase differences,
- amplitude invariance,
- invalid real input,
- multidimensional input,
- insufficient samples,
- NaN/Inf rejection.

A passing suite does not imply that unimplemented blind receiver stages exist.

---

## 12. What IQWAV Can Do Today

### 12.1 File-to-DSP Analysis

```text
known-format WAV / raw IQ
    ↓
file ingestion
    ↓
real or complex NumPy samples
    ↓
FFT / PSD / spectrogram
    ↓
basic filtering / power analysis
```

### 12.2 Controlled Digital Communication

```text
bits
    ↓
BPSK / QPSK mapping
    ↓
rectangular sampled waveform
    ↓
controlled AWGN / CFO / phase impairment
    ↓
known-timing hard-decision demodulation
    ↓
recovered bits
```

### 12.3 Real OTA Analog-FM Validation

```text
real OTA complex IQ
    ↓
raw IQ ingestion
    ↓
spectral analysis
    ↓
FM phase discrimination
    ↓
experiment-level audio filtering/resampling
    ↓
clean recovered audio
```

---

## 13. What IQWAV Does NOT Yet Do

Do not claim these as current capabilities:

- blind raw-IQ datatype inference,
- blind endianness inference,
- blind IQ/QI ordering inference,
- automatic sampling-rate inference from arbitrary headerless data,
- automatic occupied-bandwidth estimation,
- carrier/CFO estimation,
- CFO correction,
- SNR estimation,
- noise-floor estimation,
- blind baud/symbol-rate estimation,
- symbol timing recovery,
- carrier recovery,
- automatic modulation recognition,
- FSK demodulation,
- general QAM demodulation,
- blind BPSK/QPSK demodulation,
- de-interleaving,
- FEC-family identification,
- Viterbi decoding,
- Reed-Solomon decoding,
- LDPC decoding,
- blind framing,
- header/payload recovery,
- complete end-to-end analysis pipeline,
- production GUI.

Directories for future subsystems are placeholders until actual functionality is implemented.

---

## 14. Source Code vs Notebooks

Production functionality belongs in:

```text
src/iqwav/
```

Notebooks are for:

- learning,
- visualization,
- experimentation,
- algorithm prototyping,
- integration checks,
- real-data validation.

Once an algorithm is accepted and reusable, its core logic should move into `src/iqwav/`.

A notebook must not become the final application architecture.

---

## 15. Testing and Validation Principles

Whenever practical, algorithms should first be tested against known ground truth.

Possible known properties:

- sample rate,
- carrier frequency,
- modulation,
- SNR,
- symbol rate,
- CFO,
- phase offset,
- transmitted symbols,
- transmitted bits.

Validation ladder:

```text
known synthetic ground truth
    ↓
unit test
    ↓
controlled impairment test
    ↓
integration test
    ↓
real recording
    ↓
failure analysis
```

For future blind estimators, numerical estimates should be compared against ground truth rather than judged only by visual plausibility.

Real-world validation should be repeated across multiple recordings before broad performance claims are made.

---

## 16. Data Policy

Do not commit large IQ/WAV recordings directly to normal Git.

Large items such as:

- raw RF captures,
- external datasets,
- generated experiment outputs,
- ML checkpoints,

should normally remain outside Git or use an appropriate external storage mechanism.

Current `.gitignore` policy excludes normal contents of:

- `data/raw/`,
- `data/external/`,
- `data/processed/`,
- `data/synthetic/`,
- generated output directories,
- model checkpoints.

Small deterministic test fixtures may be committed when useful.

---

## 17. AI / Contributor Handoff Protocol

Before modifying the repository:

1. Read `README.md`.
2. Read the newest entries in `LOGS.md`.
3. Inspect relevant source files and tests.
4. Determine the current learning/implementation boundary.
5. Do not assume unfinished functionality exists.
6. Do not rewrite working architecture without a clear reason.
7. Keep implementation milestones bounded.
8. Add focused tests for new production behavior.
9. Run relevant focused tests first.
10. Run the full regression suite once after focused tests pass.
11. Perform manual/notebook verification where it adds value.
12. Do not update `LOGS.md` until a meaningful milestone is verified.
13. Preserve compatibility with existing code/tests unless a deliberate change is justified.
14. Keep large datasets and generated artifacts out of Git.

When proposing major architectural changes, explain why the change is necessary before implementing it.

---

## 18. Current Development Status

### Completed foundation

IQWAV currently has:

- synthetic signal generation,
- complex-IQ handling,
- FFT/PSD/spectrogram analysis,
- FIR filtering,
- signal-power measurement,
- AWGN generation,
- controlled CFO/phase impairment injection,
- BPSK/QPSK modulation,
- sampled rectangular baseband waveforms,
- known-timing BPSK/QPSK hard demodulation,
- WAV ingestion,
- stereo-WAV I/Q conversion,
- known-format raw-IQ ingestion,
- FM phase-discriminator demodulation.

### Real-data status

IQWAV has successfully:

1. loaded a genuine OTA complex-IQ broadcast-FM capture;
2. displayed physically sensible FFT, PSD and waterfall representations;
3. processed that capture through the production FM discriminator;
4. recovered approximately four seconds of clean intelligible English broadcast audio.

### Automated status

```text
279 passed
```

### Current boundary

The current system is a **working DSP and controlled-demodulation foundation**, not yet a blind RF-analysis system.

Major SIH-specific intelligence layers still to be built later include:

```text
blind parameter estimation
→ synchronization
→ AMR
→ broader demodulation
→ de-interleaving
→ FEC identification/decoding
→ correlation/framing
→ payload recovery
→ integrated GUI
```

For exact chronological development history and the latest engineering notes, see:

```text
LOGS.md
```
