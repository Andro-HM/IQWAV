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

The target system eventually needs to move from:

```text
known signal + known parameters
        ↓
process correctly
```

toward:

```text
unknown or partially known IQ / WAV
        ↓
discover useful signal parameters
        ↓
synchronize
        ↓
identify modulation
        ↓
demodulate
        ↓
recover bits
        ↓
de-interleave / FEC / frame / payload analysis
```

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
- confidence / reliability information.

Some of these now have baseline production implementations; others remain future work.

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
spectrum + occupied-band analysis
    ↓
parameter estimation
    ↓
channel selection / channelization
    ↓
carrier / timing synchronization
    ↓
automatic modulation recognition
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
→ smallest controlled experiment
→ bounded production implementation
→ focused tests
→ full regression test
→ deliberate impairment/failure testing
→ manual verification
→ real-data validation where appropriate
→ documentation
→ commit
→ next topic
```

Engineering rules:

1. Reusable production functionality belongs in `src/iqwav/`.
2. Learning notebooks are for small, controlled concept validation.
3. Experiment notebooks are for realistic multi-stage or real-data work.
4. Synthetic signals with known ground truth should be used before claiming algorithmic correctness.
5. Real RF captures should be used to expose assumptions and validate integration.
6. Passing synthetic tests alone must not be treated as proof of broad real-world performance.
7. Blind-estimation or receiver functionality must not be claimed beyond its explicitly tested scope.
8. Large recordings and generated outputs should remain outside normal Git history.
9. `LOGS.md` is the chronological engineering record; update it only after a milestone is meaningfully verified.

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
- Module 9 — Correlation and statistical signal analysis
- Module 10 — Blind parameter estimation

The current implementation boundary is therefore **Modules 0–10**.

The next major learning/implementation phase is:

- Module 11 — Synchronization

Do not assume the following have already been implemented merely because directories exist for them:

- carrier-frequency correction/tracking,
- phase recovery,
- timing recovery,
- AMR,
- blind FEC,
- interleaver identification,
- framing/payload recovery,
- complete end-to-end automation.

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
├── docs/
├── data/
│   ├── raw/
│   ├── external/
│   ├── synthetic/
│   ├── processed/
│   └── samples/
├── notebooks/
│   ├── learning/
│   └── experiments/
├── scripts/
├── src/iqwav/
│   ├── io/
│   ├── dsp/
│   ├── modulation/
│   ├── estimation/
│   ├── synchronization/
│   ├── amr/
│   ├── demod/
│   ├── interleaving/
│   ├── fec/
│   ├── correlation/
│   ├── framing/
│   ├── pipeline/
│   ├── ui/
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── models/
├── outputs/
├── native/cpp/
└── gnuradio/
```

### Current production packages

`src/iqwav/io/`

- WAV loading,
- explicit WAV I/Q conversion,
- raw interleaved IQ loading.

`src/iqwav/dsp/`

- FFT magnitude spectrum,
- periodogram PSD,
- Welch PSD,
- spectrogram/waterfall data,
- FIR filtering,
- signal power,
- AWGN injection,
- controlled phase/frequency impairment injection.

`src/iqwav/modulation/`

- real tones,
- complex IQ tones,
- BPSK mapping,
- QPSK Gray mapping,
- rectangular sampled baseband waveforms.

`src/iqwav/demod/`

- known-timing hard-decision BPSK demodulation,
- known-timing hard-decision QPSK demodulation,
- FM phase-discriminator demodulation.

`src/iqwav/correlation/`

- `autocorrelation(...)`
- `normalized_autocorrelation(...)`
- `cross_correlation(...)`
- `normalized_cross_correlation(...)`
- `find_correlation_peaks(...)`

`src/iqwav/estimation/`

- `OccupiedBand`
- `detect_occupied_bands(...)`
- `SNREstimate`
- `estimate_band_snr(...)`
- `SymbolRateEstimate`
- `estimate_symbol_rate(...)`
- `FrequencyOffsetEstimate`
- `estimate_frequency_offset(...)`
- `PeakFrequencyEstimate`
- `estimate_peak_frequency(...)`
- `OccupiedBandwidthEstimate`
- `estimate_occupied_bandwidth(...)`
- `RectangularSymbolGridEstimate`
- `estimate_rectangular_symbol_grid(...)`

These are baseline blind/semi-blind estimators with explicitly limited scopes. They are not yet a universal unknown-signal-analysis engine.

`src/iqwav/synchronization/` is reserved for the next phase: CFO correction, carrier/phase recovery and symbol timing recovery.

`src/iqwav/amr/`, `interleaving/`, `fec/`, `framing/`, `pipeline/` and `ui/` remain future subsystems.

---

## 8. Data and Notebooks

Large RF recordings remain local under `data/` and are excluded from normal Git history.

### Learning notebooks

Current:

- `01_tone_generation.ipynb`
- `02_digital_communication.ipynb`
- `03_file_io_and_signal_analysis.ipynb`
- `04_correlation_and_blind_estimation.ipynb`

Learning notebooks are for the smallest controlled validation of newly added functions, usually against synthetic known ground truth.

### Experiment notebooks

Current:

- `01_real_iq_smoke_test.ipynb`
- `02_real_fm_demodulation.ipynb`
- `03_blind_occupied_band_real_iq.ipynb`

Experiment notebooks are for real-data or multi-stage system validation.

---

## 9. Current Production Functionality

### 9.1 Synthetic Signals

Implemented:

- `generate_real_tone(...)`
- `generate_iq_tone(...)`
- BPSK/QPSK mapping,
- rectangular waveform generation.

### 9.2 Spectrum / PSD / Waterfall

Implemented:

- `magnitude_spectrum(...)`
- `periodogram_psd(...)`
- `welch_psd(...)`
- `spectrogram_data(...)`

### 9.3 Filtering / Power / Noise

Implemented:

- low/high/band-pass FIR design,
- FIR application,
- `signal_power(...)`,
- `add_awgn(...)`.

### 9.4 Controlled IQ Impairments

Implemented:

- `apply_frequency_offset(...)`
- `apply_phase_offset(...)`

These inject known impairments; they do not correct unknown impairments.

### 9.5 Known-Timing Digital Demodulation

Implemented:

- `bpsk_demodulate(...)`
- `qpsk_demodulate(...)`

Current assumptions:

- symbol boundaries known,
- samples-per-symbol supplied,
- no timing recovery,
- no carrier recovery,
- no CFO/phase correction.

### 9.6 WAV and Raw IQ Ingestion

Implemented:

- `load_wav(path)`
- `load_wav_iq(path, i_channel=0, q_channel=1)`
- `load_raw_iq(path, dtype=np.float32, iq_order="IQ")`

Raw IQ ingestion deliberately does not infer datatype, endianness, I/Q ordering, sample rate or center frequency.

### 9.7 FM Phase-Discriminator Demodulation

Implemented:

- `fm_demodulate(samples)`

Core operation:

```text
angle(samples[1:] * conj(samples[:-1]))
```

It returns adjacent phase increments in radians/sample. Filtering, DC removal, resampling, de-emphasis and stereo/RDS decoding are intentionally outside this primitive.

### 9.8 Correlation

Implemented:

- `autocorrelation(samples, max_lag=None)`
- `normalized_autocorrelation(samples, max_lag=None)`
- `cross_correlation(first, second)`
- `normalized_cross_correlation(first, second)`
- `find_correlation_peaks(correlation, lags, ...)`

Autocorrelation convention:

```text
R[k] = (1 / (N-k)) Σ x[n+k] conj(x[n])
```

Validated on periodic real sequences, complex IQ phase progression and white noise.

Cross-correlation convention:

```text
r_xy[lag] = Σ x[n+lag] conj(y[n])
```

Cross-correlation returns an explicit lag array; a delayed first input peaks at lag `+d`. The normalized variant divides by exact per-lap overlap energies, is bounded by 1 in magnitude, and rejects zero-energy input. `find_correlation_peaks` finds local extrema of the correlation magnitude by default, handles complex correlations through magnitude, and returns peak indices, lags and original values in the caller's explicit lag convention.

### 9.9 Blind Spectral Occupied-Band Detection

Implemented:

- `detect_occupied_bands(samples, fs, *, nperseg=None, threshold_db=6.0, min_bins=3)`

Baseline:

```text
samples
    ↓
Welch PSD
    ↓
dB conversion
    ↓
median background estimate
    ↓
threshold
    ↓
contiguous occupied bins
    ↓
band edges / center / width / peak
```

The result reports frequencies relative to the capture/baseband center. Its width is threshold-defined spectral width; this is a different quantity from the cumulative-power occupied-bandwidth measurement in 9.14.

### 9.10 Blind In-Band SNR Estimation

Implemented:

- `estimate_band_snr(samples, fs, band, *, nperseg=None)`

Returns signal power, noise power, total in-band power, noise PSD and SNR.

Definition:

```text
SNR = estimated in-band signal power / estimated in-band noise power
```

This is not Eb/N0, Es/N0, BER, modulation quality or receiver noise figure.

### 9.11 Blind Rectangular-PSK Symbol-Rate Estimation

Implemented:

- `estimate_symbol_rate(samples, fs, *, min_sps=2, max_sps=64, min_score=0.10)`

Baseline:

```text
waveform
    ↓
|x[n+1] - x[n]|²
    ↓
mean removal
    ↓
normalized autocorrelation
    ↓
periodic symbol-boundary peak
    ↓
SPS
    ↓
symbol rate = Fs / SPS
```

Validated for rectangular/sample-and-hold BPSK/QPSK-like waveforms with integer SPS, enough transitions, known Fs and moderate SNR.

It is not yet general blind baud estimation for RRC/pulse-shaped/fractional-SPS signals.

### 9.12 Coarse PSK Frequency-Offset Estimation

Implemented:

- `estimate_frequency_offset(samples, fs, *, min_coherence=0.05)`

Baseline:

```text
R[1] = mean(x[n+1] conj(x[n]))
angle(R[1]) ≈ 2π Δf / Fs
Δf ≈ Fs * angle(R[1]) / (2π)
```

Returns CFO, per-sample phase increment and lag-1 coherence.

Validated for complex, oversampled rectangular BPSK/QPSK-like signals with constant CFO, known Fs and moderate SNR.

This is coarse estimation only; no carrier correction or tracking is performed. Finite QPSK records can show small residual bias because random symbol-boundary terms do not cancel exactly in a finite observation.

### 9.13 Dominant Spectral Peak Estimation

Implemented:

- `estimate_peak_frequency(samples, fs, *, refine=True)`

Locates the largest-magnitude FFT bin, optionally refined to sub-bin precision by bounded log-magnitude parabolic interpolation, and reports the dominant spectral component: signed frequency for complex input, non-negative frequency for real input, constant/DC-only input rejected.

This reports the strongest spectral component. It is not automatically the carrier or center frequency of a wideband modulated signal.

### 9.14 Cumulative-Power Occupied-Bandwidth Measurement

Implemented:

- `estimate_occupied_bandwidth(samples, fs, *, power_fraction=0.99)`

Returns the narrowest contiguous FFT-bin interval containing at least `power_fraction` of the total measured spectral power. For complex input the search is cyclic across the Nyquist boundary (a wrapping interval is reported with `wraps_nyquist=True` and `lower_hz > upper_hz`); for real input the conjugate-symmetric spectrum is folded onto the non-negative axis with edges clamped to `[0, fs/2]`.

This is distinct from `detect_occupied_bands` (9.9): the detector is noise-floor/threshold based and may return multiple bands; this measurement always returns exactly one cumulative-power containment interval and does not subtract noise. A 99% result means 99% of total measured FFT power, including noise, DC and interference — not 99% of signal-only power.

### 9.15 Rectangular Symbol-Grid Estimation

Implemented:

- `estimate_rectangular_symbol_grid(samples, fs, *, min_sps=2, max_sps=64, quality_ratio=0.75, min_quality=0.02)`

Groups first-difference magnitudes by `sample_index % P` for each candidate integer period, chance-corrects the strongest residue bin into a quality score, resolves divisor ambiguity by selecting the largest near-best candidate with a divisor-dominance guard, and reports the winning residue class as `boundary_offset`: symbol-start sample indices are congruent to it modulo `samples_per_symbol`.

`boundary_offset` is a block-level symbol-boundary phase estimate, NOT timing recovery.

This estimator and `estimate_symbol_rate` (9.11) are separate production paths: the former is HM's transition-autocorrelation conservative baseline, the latter is a bounded rectangular integer-SPS grid estimator that additionally returns boundary phase. They agree on clean rectangular PSK waveforms.

---

## 10. Known Metadata vs Estimated Parameters

### WAV

A standard WAV header normally supplies sample rate, channel count and sample encoding/bit depth. It does not reveal modulation, baud, FEC, framing, etc.

### Headerless raw IQ

A raw IQ sequence may not uniquely contain:

- datatype,
- byte order,
- IQ/QI ordering,
- sample rate,
- absolute RF center frequency.

These currently come from the operator or external metadata.

### Parameters currently estimated under defined conditions

IQWAV production baselines can estimate:

- occupied spectral regions,
- threshold-defined bandwidth,
- cumulative-power occupied bandwidth,
- dominant spectral component,
- relative spectral center,
- spectral noise floor,
- in-band SNR,
- rectangular-PSK symbol rate / integer SPS,
- rectangular symbol-grid boundary phase,
- coarse PSK CFO.

Known sample rate is assumed for these estimators. Raw-IQ sample rate and absolute RF center frequency cannot generally be inferred from samples alone; they still come from recording metadata or operator context.

Absolute RF frequency requires recording-center metadata:

```text
absolute RF frequency
    =
capture center metadata
    +
estimated relative frequency
```

---

## 11. Real-World Validation

### 11.1 PySDR 4-Second Broadcast-FM Capture

Capture:

```text
fm_rds_250k_1Msamples.iq
```

Known metadata:

- center frequency: 99.5 MHz,
- sample rate: 250 kHz,
- complex64 / interleaved float32 I,Q,
- 1,000,000 complex samples,
- approximately 4 seconds.

Production path exercised:

```text
real OTA IQ
    ↓
load_raw_iq()
    ↓
FFT / PSD / spectrogram
    ↓
fm_demodulate()
```

Experiment-level post-processing extracted 0–15 kHz mono-compatible audio, removed DC, resampled 250 kHz → 50 kHz and produced approximately 4 seconds of clean intelligible audio.

The demodulated FM multiplex showed expected structure including a 19 kHz stereo pilot and a feature near the 57 kHz RDS region. No stereo or RDS decoding is claimed.

### 11.2 Mumbai Wideband Broadcast-FM Capture

Capture:

```text
mumbai-10s-10M-92.3-8-10-25.iq
```

Known metadata:

- 10 MS/s,
- recording center 92.3 MHz,
- complex64,
- approximately 880 MB,
- approximately 11 seconds.

Manual FM experiment:

```text
10 MS/s wideband IQ
    ↓
IQ DC removal
    ↓
frequency translation of target near 92.7 MHz
    ↓
anti-alias filtering + 40× decimation
    ↓
250 kS/s channel
    ↓
fm_demodulate()
    ↓
15 kHz mono extraction
    ↓
50 µs de-emphasis
    ↓
50 kHz audio
```

Result: clear, intelligible approximately 11-second broadcast audio.

Frequency translation/channelization and audio post-processing remain experiment-level rather than a reusable production receiver pipeline.

### 11.3 Real Blind Occupied-Band Detection

Experiment:

```text
notebooks/experiments/03_blind_occupied_band_real_iq.ipynb
```

On a 0.2-second Mumbai chunk, the detector was given real IQ samples and known sample rate, but not station locations, number of stations or station bandwidths.

Major automatically detected RF centers, after applying known 92.3 MHz capture-center metadata, included approximately:

- 91.114 MHz,
- 91.898 MHz,
- 92.701 MHz,
- 93.482 MHz.

A narrow ~3.7 kHz region near 92.796 MHz was also detected and remains a fragment/candidate rather than a confirmed physical channel.

### 11.4 Real Blind In-Band SNR Estimation

| RF center | Detected BW | Estimated in-band SNR |
|---:|---:|---:|
| 91.114 MHz | 69.6 kHz | 7.86 dB |
| 91.898 MHz | 161.1 kHz | 19.12 dB |
| 92.701 MHz | 173.3 kHz | 18.17 dB |
| 92.796 MHz | 3.7 kHz | 6.36 dB |
| 93.482 MHz | 97.0 kHz | 15.59 dB |

These are baseline spectral estimates, not calibrated receiver measurements.

### 11.5 Current Real-Data Boundary

Real-data validation supports claims that IQWAV can:

- ingest genuine complex IQ,
- produce sensible spectrum/PSD/waterfall representations,
- recover FM information and clear audio,
- work with a wideband multi-station recording,
- automatically discover occupied spectral regions in real OTA IQ,
- estimate baseline in-band SNR for those regions.

It does not yet prove real-world blind PSK baud/CFO performance, synchronization, AMR or digital payload recovery.

---

## 12. Testing Status

Current full regression suite:

```text
567 tests passing
0 failures
0 skipped
```

Coverage includes:

- tone generation,
- FFT / PSD / spectrogram,
- FIR filtering,
- signal power / AWGN,
- IQ impairments,
- BPSK/QPSK modulation and waveforms,
- known-timing BPSK/QPSK demodulation,
- WAV/raw-IQ ingestion,
- FM demodulation,
- autocorrelation, cross-correlation and correlation peaks,
- occupied-band detection,
- in-band SNR estimation,
- rectangular-PSK symbol-rate estimation,
- coarse PSK CFO estimation,
- dominant spectral peak estimation,
- cumulative-power occupied-bandwidth measurement,
- rectangular symbol-grid estimation.

Recent focused milestones:

- autocorrelation: 30 tests,
- occupied-band detection: 34 tests,
- band SNR estimation: 22 tests,
- symbol-rate estimation: 37 tests,
- frequency-offset estimation: 30 tests,
- cross-correlation: 10 tests,
- correlation peaks: 18 tests,
- dominant spectral peak: 24 tests,
- occupied bandwidth: 29 tests,
- rectangular symbol grid: 54 tests.

Passing tests demonstrate correctness only within the tested assumptions.

---

## 13. What IQWAV Can Do Today

```text
known-format WAV / raw IQ
    ↓
file ingestion
    ↓
FFT / PSD / spectrogram
    ↓
filtering / power analysis
    ↓
correlation / peak analysis
```

Controlled digital path:

```text
bits
    ↓
BPSK / QPSK
    ↓
rectangular waveform
    ↓
AWGN / CFO / phase impairments
    ↓
known-timing demodulation
    ↓
bits
```

First blind/semi-blind parameter path:

```text
samples + known Fs
    ↓
occupied-band detection
    ↓
relative center / width / noise floor
    ↓
in-band SNR
```

Supported rectangular oversampled PSK path:

```text
samples + known Fs
    ↓
symbol-rate / SPS estimate   (or rectangular symbol grid + boundary phase)
    ↓
coarse CFO estimate
```

Real OTA path demonstrated:

```text
real wideband IQ
    ↓
blind occupied-band detection
    ↓
baseline per-band SNR
```

and separately:

```text
real FM IQ
    ↓
experiment-level channelization
    ↓
fm_demodulate()
    ↓
clear recovered audio
```

---

## 14. What IQWAV Does NOT Yet Do

Do not claim:

- blind raw-IQ datatype inference,
- blind endianness inference,
- blind IQ/QI ordering inference,
- automatic sampling-rate inference from arbitrary headerless raw IQ,
- automatic absolute RF-center inference from headerless raw IQ,
- regulatory/standards-compliant occupied-bandwidth measurement (the cumulative-power estimator is definition-specific),
- robust physical-channel merging / gap bridging,
- universal blind SNR estimation in arbitrary crowded/nonstationary spectra,
- general blind baud estimation for arbitrary pulse shaping/modulation,
- universal CFO estimation for arbitrary modulations,
- CFO correction,
- carrier tracking,
- phase recovery,
- symbol timing recovery,
- matched filtering / RRC receiver chain,
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
- complete end-to-end unknown-signal pipeline,
- production GUI.

---

## 15. Current Estimator Scope

### Occupied-band detector

Current limits:

- median floor assumes strong occupancy does not dominate most of the spectrum,
- no gap bridging,
- no physical-channel classification,
- threshold-defined width.

### Band SNR estimator

Current limits:

- requires a target band,
- assumes approximately broadband/stationary noise,
- crowded spectra can bias the out-of-band noise reference.

### Symbol-rate estimator

Current limits:

- rectangular BPSK/QPSK-like signals,
- integer SPS,
- no RRC/general pulse shaping,
- no fractional SPS,
- a harmonic can be selected if the true SPS is excluded from the search range.

### Frequency-offset estimator

Current limits:

- complex oversampled rectangular PSK-like input,
- constant CFO,
- coarse estimate only,
- no correction/tracking,
- principal-angle ambiguity,
- finite-record QPSK bias possible.

### Dominant spectral-peak estimator

Current limits:

- reports the strongest spectral component only,
- not automatically the carrier/center frequency of a wideband modulated signal,
- sub-bin refinement assumes an isolated peak under a rectangular window.

### Cumulative-power occupied-bandwidth estimator

Current limits:

- the power fraction counts all measured power, including noise, DC and interference,
- FFT-bin edge resolution of the analyzed block only,
- single-block measurement, no cross-block averaging.

### Rectangular symbol-grid estimator

Current limits:

- rectangular piecewise-constant symbols with integer SPS only; not RRC/pulse-shaped, not fractional-SPS,
- `boundary_offset` is a block-level symbol-boundary phase estimate, not timing recovery,
- symbol changes every m-th true boundary make the observable period m*SPS,
- an observable period outside the search range may yield an in-range divisor.

---

## 16. Selective Reconciliation with the Sayan Snapshot

Useful capabilities from the frozen Sayan snapshot `sayan-snapshot-9e927de` were reviewed and manually adapted into IQWAV's architecture. Nothing was merged or cherry-picked wholesale.

Integrated:

- cross-correlation and correlation peak detection,
- dominant spectral peak estimation,
- cumulative-power occupied-bandwidth measurement,
- rectangular symbol-grid estimator with boundary offset.

A 692-trial controlled benchmark compared HM's existing symbol-rate estimator with the residue-grid method. The residue-grid approach was substantially stronger under low-SNR AWGN, while both were perfect on clean/phase/amplitude/crop/CFO/short cases. HM's original estimator was intentionally retained as an independent conservative baseline rather than replaced; the residue-grid estimator lives separately as `estimate_rectangular_symbol_grid`. Detailed benchmark tables are in `LOGS.md`.

Parked (reviewed, not yet integrated):

- explicit-region noise/SNR estimation,
- known-reference frequency-offset estimation,
- controlled BPSK/QPSK classifier (candidate for Module 12).

Sayan's alternate autocorrelation, SNR, CFO and symbol-rate implementations have not replaced any HM production API.

---

## 17. Source Code vs Notebooks

Production functionality belongs in `src/iqwav/`.

Learning notebooks are for:

- smallest controlled validation,
- synthetic ground truth,
- understanding one new function/concept,
- immediate manual verification.

Experiment notebooks are for:

- real data,
- multi-stage processing,
- receiver chains,
- integration and realistic failure analysis.

A notebook must not become the final application architecture.

---

## 18. Testing and Validation Principles

Validation ladder:

```text
known synthetic ground truth
    ↓
unit test
    ↓
controlled impairment test
    ↓
manual learning-notebook verification
    ↓
integration test
    ↓
real recording where applicable
    ↓
failure analysis
```

Numerical estimators should be compared against ground truth rather than judged only visually.

Real-world validation should be repeated across multiple recordings before broad performance claims are made.

---

## 19. Data Policy

Do not commit large IQ/WAV recordings directly to normal Git.

Large raw RF captures, external datasets, generated outputs and model checkpoints should normally remain outside Git.

Current `.gitignore` policy excludes normal contents of:

- `data/raw/`,
- `data/external/`,
- `data/processed/`,
- `data/synthetic/`,
- generated outputs,
- model checkpoints.

Small deterministic fixtures may be committed when useful.

Ignored files can still be downloaded or created locally by collaborators; `.gitignore` only controls Git tracking.

---

## 20. AI / Contributor Handoff Protocol

Before modifying the repository:

1. Read `README.md`.
2. Read the newest entries in `LOGS.md`.
3. Inspect relevant source and tests.
4. Determine the current learning/implementation boundary.
5. Do not assume unfinished functionality exists.
6. Keep milestones bounded.
7. Add focused tests.
8. Run focused tests first.
9. Run the full regression suite once after focused tests pass.
10. Perform manual/notebook verification where useful.
11. Update `LOGS.md` only after meaningful verification.
12. Preserve working architecture unless change is justified.
13. Keep large data/generated artifacts out of Git.
14. Distinguish metadata from signal-derived estimates.
15. State assumptions and failure modes explicitly.
16. Do not turn a baseline result into a universal capability claim.

---

## 21. Current Development Status

### Completed foundation

IQWAV currently has:

- synthetic signal generation,
- complex-IQ handling,
- FFT/PSD/spectrogram analysis,
- FIR filtering,
- power and AWGN tools,
- controlled CFO/phase impairment injection,
- BPSK/QPSK modulation,
- rectangular sampled digital waveforms,
- known-timing BPSK/QPSK demodulation,
- WAV ingestion,
- stereo-WAV I/Q conversion,
- known-format raw-IQ ingestion,
- FM phase-discriminator demodulation,
- autocorrelation primitives,
- blind occupied-band detection baseline,
- spectral noise-floor estimation baseline,
- blind in-band SNR estimation baseline,
- rectangular-PSK symbol-rate estimation baseline,
- coarse PSK CFO estimation baseline,
- cross-correlation and correlation peak utilities,
- dominant spectral peak estimation baseline,
- cumulative-power occupied-bandwidth measurement,
- rectangular symbol-grid estimation baseline with block-level boundary phase.

### Real-data status

IQWAV has successfully:

1. loaded genuine OTA complex-IQ FM data;
2. produced sensible FFT, PSD and waterfall output;
3. recovered clean audio from a prepared 250 kS/s FM capture;
4. processed an approximately 880 MB, 10 MS/s multi-station Mumbai capture;
5. manually channelized a selected station and recovered approximately 11 seconds of clear audio;
6. observed expected FM multiplex structure including the 19 kHz stereo pilot;
7. automatically discovered major occupied regions in the real Mumbai capture;
8. estimated baseline in-band SNR values for those regions.

### Automated status

```text
567 passed
0 failures
0 skipped
```

### Current boundary

The current system is:

**working DSP + controlled demodulation + first blind/semi-blind parameter-estimation foundation**

The next major phase is:

```text
Module 11 — synchronization
    ↓
CFO correction
    ↓
carrier / phase recovery
    ↓
symbol timing recovery
    ↓
synchronized symbols
```

Then:

```text
Module 12 — AMR
→ broader automatic demodulation
→ de-interleaving
→ FEC identification / decoding
→ correlation / framing
→ payload recovery
→ integrated pipeline
→ GUI
```

For exact chronological engineering history, see:

```text
LOGS.md
```
