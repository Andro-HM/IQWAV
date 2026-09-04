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

## 2026-09-04 — Added rectangular symbol-grid estimator after HM-vs-Sayan benchmark

Completed a controlled head-to-head benchmark of HM's existing symbol-rate
estimator against Sayan's transition-residue estimator from frozen snapshot
`9e927de`.

Benchmark:
- 692 deterministic trials
- BPSK and QPSK
- SPS 2, 3, 4, 5, 8, 12, 16, 24, 32
- clean, phase, amplitude, crop, AWGN, CFO, short-block and pathological cases

Results:
- HM overall correct: 92.8%
- Sayan method overall correct: 95.4%
- HM false-estimate rate: 1.6%
- Sayan false-estimate rate: 1.7%
- HM rejection rate: 5.6%
- Sayan rejection rate: 2.9%

Low-SNR AWGN:
- 0 dB: HM 22.2% correct, Sayan 77.8%
- -5 dB: HM 0% correct, Sayan 44.4%

Boundary-phase benchmark:
- 194 cropped rectangular-waveform trials
- Sayan boundary offset exact in 194/194 cases

Architecture decision:
- HM's existing `estimate_symbol_rate()` remains unchanged as an independent,
  conservative transition-autocorrelation estimator.
- Sayan's method was not used as a replacement.
- Its transition-residue method was adapted into a separate bounded symbol-grid
  estimator.

Added:
- `RectangularSymbolGridEstimate`
- `estimate_rectangular_symbol_grid`

File:
- `src/iqwav/estimation/symbol_grid.py`

The new estimator returns:
- symbol rate
- integer samples per symbol
- block-level symbol-boundary offset
- quality
- concentration
- symbol count
- effective transition count
- searched SPS range

Important semantics:
- `boundary_offset` is a block-level symbol-grid phase estimate, not timing recovery.
- The estimator assumes rectangular, piecewise-constant symbols with integer SPS.
- It is not a general pulse-shaped or fractional-SPS blind baud estimator.
- Sparse/repeating symbol transitions can make only a multiple of the transmitter
  symbol period observable; if that period lies outside the search range, an
  in-range divisor may be returned. This is an identifiability limitation.

Validation:
- New symbol-grid tests: 54 passed
- Existing HM symbol-rate tests: 37 passed
- Full suite after integration: 567 passed


## 2026-09-04 — Integrated cumulative-power occupied-bandwidth estimator

Integrated and adapted the cumulative-power occupied-bandwidth capability
from frozen Sayan snapshot `9e927de` without merging or cherry-picking
Sayan's branch.

Added:
- `OccupiedBandwidthEstimate`
- `estimate_occupied_bandwidth`

Architecture:
- Added `src/iqwav/estimation/occupied_bandwidth.py`.
- HM's existing `OccupiedBand` and `detect_occupied_bands()` remain unchanged.
- The new estimator answers a different question: it finds the narrowest
  frequency interval containing a requested fraction of total measured FFT
  power.

Definition:
- FFT-bin power is `abs(FFT[k]) ** 2`.
- Default-style usage can request, for example, 0.99 of total measured power.
- Noise, interference, DC, and any other spectral energy all contribute.
- No noise-floor subtraction or signal-presence decision is performed.

HM-specific improvement:
- Complex-IQ frequency topology is treated as circular.
- A minimum-power interval may cross the Nyquist boundary.
- Wrapped results use `wraps_nyquist=True` and are interpreted as:
  `[lower_hz, +fs/2) U [-fs/2, upper_hz]`.
- This avoids falsely reporting nearly full-band widths when signal energy
  straddles +fs/2 and -fs/2.

Real-valued input:
- Conjugate-symmetric FFT power is folded onto the non-negative physical
  frequency axis.
- DC and Nyquist are counted once.
- Returned real-signal intervals remain inside `[0, fs/2]`.

Result includes:
- lower/upper frequency edges
- circular center frequency
- bandwidth
- requested power fraction
- achieved power fraction
- Nyquist-wrap flag

Validation:
- Focused occupied-bandwidth suite: 29 passed
- Full suite: 513 passed
- Previous full suite: 484 passed
- 29 new tests added

Nyquist-wrap validation:
- fs = 1000 Hz, N = 1000
- tones at +496 Hz and -494 Hz
- 99% interval selected as an 11-bin wrapped region
- bandwidth = 11 Hz rather than an approximately full-band linear interval

Status:
Validated on the `integrate-sayan` branch.


## 2026-09-04 — Integrated dominant spectral peak estimator from Sayan snapshot

Integrated the dominant spectral-frequency estimation capability from frozen
Sayan snapshot `9e927de` without merging or cherry-picking Sayan's branch.

Added:
- `PeakFrequencyEstimate`
- `estimate_peak_frequency`

Architecture:
- Added `src/iqwav/estimation/spectral_peak.py`.
- Reuses HM's existing `magnitude_spectrum` FFT primitive.
- Complex IQ searches the full signed two-sided spectrum.
- Real-valued signals search the non-negative spectral half.
- Optional three-point log-magnitude parabolic interpolation provides a
  sub-bin peak estimate.
- Raw FFT resolution remains `fs / N`.

Semantics:
- This reports the single strongest spectral component in the analyzed block.
- It is not automatically a carrier-frequency estimate, occupied-band center,
  CFO estimate, activity detector, bandwidth measurement, or SNR estimate.
- A wideband modulated signal may have its strongest spectral component away
  from its actual center frequency.

HM-specific integration improvements:
- Boolean sample rates are rejected.
- `refine` must explicitly be boolean.
- Non-numeric sample arrays are rejected cleanly.
- Constant/all-zero inputs are rejected.

Files:
- Added `src/iqwav/estimation/spectral_peak.py`
- Updated `src/iqwav/estimation/__init__.py`
- Added `tests/unit/test_spectral_peak.py`

Validation:
- Focused spectral-peak suite: 24 passed
- Full suite: 484 passed
- Previous full suite: 460 passed
- 24 new tests added

Status:
Validated on the `integrate-sayan` branch.


## 2026-09-04 — Integrated cross-correlation and correlation peak utilities from Sayan snapshot

Integrated selected correlation capabilities from frozen Sayan snapshot
`9e927de` without merging or cherry-picking Sayan's branch.

Added:
- `cross_correlation`
- `normalized_cross_correlation`
- `find_correlation_peaks`

Architecture:
- HM's existing `autocorrelation` and `normalized_autocorrelation`
  remain unchanged.
- Sayan's alternative autocorrelation implementation was intentionally
  not adopted because its lag/normalization/API semantics differ from
  HM's production convention used by symbol-rate and CFO estimation.
- Cross-correlation uses
  `r_xy[k] = Σ x[n+k] * conj(y[n])`.
- A delayed first input therefore produces a positive correlation lag.
- Normalization uses exact overlap energies for each lag.
- Correlation peak detection operates on magnitude by default and
  supports complex correlations.

Files:
- Added `src/iqwav/correlation/cross_correlation.py`
- Added `src/iqwav/correlation/peaks.py`
- Updated `src/iqwav/correlation/__init__.py`
- Added focused unit tests for cross-correlation and peak detection.

Validation:
- Focused correlation suite: 58 passed
- Full suite: 460 passed
- Previous full suite: 432 passed
- 28 new tests added
- Existing 30 HM autocorrelation tests remain passing

Status:
Validated on the `integrate-sayan` branch.


## 2026-09-01 — Coarse PSK frequency-offset estimation implemented and verified

### Implementation

Added production coarse frequency-offset estimation:

- `FrequencyOffsetEstimate`
- `estimate_frequency_offset(samples, fs, *, min_coherence=0.05)`

Location:

`src/iqwav/estimation/frequency_offset.py`

The estimator uses lag-1 complex autocorrelation:

`R[1] = mean(x[n+1] * conj(x[n]))`

For an oversampled PSK-like signal with constant frequency offset:

`angle(R[1]) ≈ 2π Δf / Fs`

therefore:

`Δf ≈ Fs * angle(R[1]) / (2π)`

A coherence measure:

`|R[1]| / R[0]`

is used to reject unreliable low-correlation inputs.

### Automated validation

Focused frequency-offset tests:

- 30 passed

Full project suite:

- 432 passed

Clean BPSK offsets from approximately -5 kHz to +5 kHz were recovered with very small error.

QPSK positive and negative offsets were also recovered successfully.

The estimator remained stable under:

- constant phase rotation
- amplitude scaling
- waveform cropping
- 20 dB AWGN
- 10 dB AWGN

### Manual notebook verification

Continued:

`notebooks/learning/04_correlation_and_blind_estimation.ipynb`

A QPSK waveform was given a known:

- Fs = 80 kS/s
- SPS = 8
- true CFO = +1000 Hz

A short record produced:

- estimated CFO ≈ 1051.86 Hz
- phase increment ≈ 0.08261 rad/sample
- coherence ≈ 0.875

A fixed phase rotation of 1.7 rad changed the estimate only by floating-point noise, confirming that constant phase cancels in adjacent-sample correlation.

At 10 dB AWGN:

- estimated CFO ≈ 1051.96 Hz
- coherence decreased to approximately 0.797

The similar clean/noisy estimates showed that the dominant short-record error was not AWGN but finite QPSK symbol-boundary averaging.

Using a much longer QPSK record reduced the error substantially:

- true CFO = 1000 Hz
- estimated CFO ≈ 1003.55 Hz
- error ≈ +3.55 Hz
- coherence ≈ 0.874

This confirmed that random QPSK boundary-phase contributions average out as more observations are available.

### Limitations

This remains a coarse estimator rather than carrier synchronization.

Current assumptions:

- complex IQ input
- known sample rate
- oversampled PSK-like waveform
- rectangular/sample-and-hold pulse structure
- constant frequency offset
- moderate SNR
- sufficient observation length

Not yet handled:

- CFO correction
- carrier tracking
- timing recovery
- pulse-shaped arbitrary signals
- one-sample-per-symbol random PSK
- ambiguity resolution outside the principal lag-1 phase range

### Result

PASS — IQWAV can now estimate coarse carrier-frequency offset from supported oversampled complex PSK signals using complex autocorrelation, with reliability indicated by lag-1 coherence.


## 2026-09-01 — Blind rectangular-PSK symbol-rate estimation implemented and verified

### Implementation

Added production symbol-rate estimation:

- `SymbolRateEstimate`
- `estimate_symbol_rate(samples, fs, *, min_sps=2, max_sps=64, min_score=0.10)`

Location:

`src/iqwav/estimation/symbol_rate.py`

The baseline estimator targets rectangular-pulse BPSK/QPSK-like waveforms with integer samples per symbol.

Processing:

`waveform`
→ adjacent transition energy `|x[n+1] - x[n]|²`
→ transition-energy mean removal
→ normalized autocorrelation
→ search for recurring symbol-boundary periodicity
→ select smallest reliable local autocorrelation peak
→ estimate samples per symbol
→ compute symbol rate as `Fs / SPS`

The smallest qualifying peak is selected because harmonics at `2*SPS`, `3*SPS`, etc. can also have strong autocorrelation.

### Automated validation

Focused symbol-rate tests:

- 37 passed

Full project test suite:

- 402 passed

Verified BPSK and QPSK recovery at:

- SPS = 4
- SPS = 8
- SPS = 16

For `Fs = 80 kS/s` and `SPS = 8`:

- estimated SPS = 8
- estimated symbol rate = 10,000 baud

The estimator also remained correct under:

- constant phase rotation
- amplitude scaling
- cropped/start-offset waveforms
- 20 dB AWGN
- 10 dB AWGN

### Manual notebook verification

Continued:

`notebooks/learning/04_correlation_and_blind_estimation.ipynb`

A synthetic BPSK waveform with hidden `SPS = 8` showed autocorrelation peaks at approximately:

- lag 8
- lag 16
- lag 24
- lag 32

The estimator correctly selected the fundamental lag:

- true SPS: 8
- estimated SPS: 8
- true rate: 10,000 baud
- estimated rate: 10,000 baud
- score: approximately 0.460

A QPSK waveform with 10 dB AWGN also returned:

- estimated SPS: 8
- estimated rate: 10,000 baud
- score: approximately 0.538

### Limitations

This is not yet general blind baud estimation.

Current assumptions include:

- rectangular/sample-and-hold pulse structure
- integer SPS
- BPSK/QPSK-like symbols
- sufficiently many symbol transitions
- moderate SNR
- known sampling rate

Not yet handled:

- RRC or other pulse shaping
- fractional SPS
- timing drift
- severe CFO
- matched filtering
- carrier recovery
- timing synchronization
- arbitrary modulation

If the true SPS is excluded from the search range but a harmonic remains inside it, a harmonic may be returned.

### Result

PASS — IQWAV can now infer symbol spacing and baud rate for supported rectangular-pulse PSK waveforms without being given samples-per-symbol explicitly.


## 2026-09-01 — Blind in-band SNR estimation implemented and verified

### Implementation

Added production spectral SNR estimation:

- `SNREstimate`
- `estimate_band_snr(samples, fs, band, *, nperseg=None)`

Location:

`src/iqwav/estimation/band_snr.py`

The estimator operates on an `OccupiedBand` and computes:

`IQ samples`
→ Welch PSD
→ target-band bins
→ median linear PSD outside target band
→ estimated in-band noise power
→ total in-band power
→ noise subtraction
→ signal power
→ SNR in dB

Definition:

`SNR = estimated in-band signal power / estimated in-band noise power`

This is not Eb/N0, Es/N0, CNR, BER, or receiver noise figure.

### Automated validation

Focused SNR tests:

- 22 passed

Full project test suite:

- 365 passed

Synthetic tests produced:

- designed 0 dB → approximately 0.30 dB
- designed 5 dB → approximately 5.13 dB
- designed 10 dB → approximately 10.07 dB
- designed 20 dB → approximately 20.06 dB

Additional tests verified expected changes with signal amplitude, noise amplitude, frequency translation, real input, marginal/no-signal cases, and composition with `detect_occupied_bands`.

### Manual synthetic verification

A synthetic approximately 60 kHz occupied band with a designed in-band SNR of 10 dB was processed using:

`detect_occupied_bands()`
→ `estimate_band_snr()`

Result:

- target SNR: 10 dB
- estimated SNR: approximately 10.05 dB
- estimated signal power: approximately 1.003
- estimated noise power: approximately 0.099
- total in-band power: approximately 1.102

Result matched the designed SNR closely.

### Real OTA validation

The estimator was applied to the occupied regions previously discovered automatically in the Mumbai 10 MS/s OTA capture.

Estimated values:

- 91.114 MHz, BW ≈ 69.6 kHz → SNR ≈ 7.86 dB
- 91.898 MHz, BW ≈ 161.1 kHz → SNR ≈ 19.12 dB
- 92.701 MHz, BW ≈ 173.3 kHz → SNR ≈ 18.17 dB
- 92.796 MHz, BW ≈ 3.7 kHz → SNR ≈ 6.36 dB
- 93.482 MHz, BW ≈ 97.0 kHz → SNR ≈ 15.59 dB

The narrow approximately 92.796 MHz region remains a detector fragment/candidate rather than a confirmed physical communication channel.

### Limitations

- assumes approximately broadband and locally stationary background noise
- estimates noise from out-of-band PSD bins
- sufficiently crowded spectra may bias the noise estimate
- depends on the correctness of the supplied `OccupiedBand`
- does not estimate Eb/N0, Es/N0, BER, or modulation quality
- does not determine whether an occupied region represents a genuine communication channel

### Result

PASS — IQWAV can now automatically discover occupied regions in real IQ data and estimate their in-band signal-to-noise ratio without access to a clean reference signal.


## 2026-09-01 — Blind occupied-band detection implemented and verified on real OTA IQ

### Implementation

Added production blind spectral occupancy detection:

- `OccupiedBand`
- `detect_occupied_bands(samples, fs, *, nperseg=None, threshold_db=6.0, min_bins=3)`

Location:

`src/iqwav/estimation/occupied_band.py`

The baseline detector:

`IQ samples`
→ Welch PSD
→ PSD in dB
→ median spectral noise-floor estimate
→ threshold above noise floor
→ contiguous occupied-bin grouping
→ minimum-width filtering
→ occupied-band parameter extraction

For each detected region it reports:

- lower frequency
- upper frequency
- center frequency
- bandwidth
- spectral peak frequency
- peak PSD
- peak margin above estimated noise floor

Frequencies are relative to the capture center. Absolute RF frequency is only obtained when external recording-center metadata is available.

### Automated validation

Focused occupied-band tests:

- 34 passed

Full project test suite:

- 343 passed

Synthetic validation recovered a deliberately hidden approximately 60 kHz-wide signal centered at +100 kHz:

- estimated lower edge ≈ 69.70 kHz
- estimated upper edge ≈ 130.25 kHz
- estimated center ≈ 99.98 kHz
- estimated bandwidth ≈ 60.55 kHz

A pure complex white-noise test returned no occupied regions.

### Real OTA validation

Created:

`notebooks/experiments/03_blind_occupied_band_real_iq.ipynb`

The detector was applied to a 0.2-second chunk of the previously validated Mumbai wideband FM capture:

- sample rate: 10 MS/s
- recording center: 92.3 MHz
- detector was NOT supplied station locations, bandwidths, or number of stations

Estimated noise floor:

- approximately -118.66 dB

Major automatically detected RF centers included:

- approximately 91.114 MHz
- approximately 91.898 MHz
- approximately 92.701 MHz
- approximately 93.482 MHz

These correspond closely to major spectral regions previously observed manually in the real capture.

A narrow additional approximately 3.7 kHz region near 92.796 MHz was also returned. Inspection showed a below-threshold gap separating it from the nearby large 92.7 MHz occupied region. The current baseline intentionally performs no gap bridging or physical-channel merging, so the regions remain separate.

### Limitations observed

- median noise-floor estimation assumes less than roughly half of the analyzed spectrum is strongly occupied
- no gap bridging or occupied-region merging
- no RF-center inference from raw IQ
- no SNR estimate yet
- a threshold-fragmented physical channel may appear as multiple occupied regions
- a detected narrow region is not automatically classified as a real communication channel

### Result

PASS — IQWAV can now automatically discover and measure occupied spectral regions in both controlled synthetic data and genuine wideband OTA IQ without being told where the signals are located.


## 2026-08-31 — Autocorrelation primitives implemented and verified

### Implementation

Added production correlation utilities:

- `autocorrelation(samples, max_lag=None)`
- `normalized_autocorrelation(samples, max_lag=None)`

Location:

`src/iqwav/correlation/autocorrelation.py`

The implementation supports real and complex 1-D signals and computes non-negative-lag autocorrelation using:

`R[k] = (1/(N-k)) Σ x[n+k] conj(x[n])`

Overlap normalization prevents artificial decay at larger lags.

Normalized autocorrelation divides by `R[0]`, giving unity at lag 0.

### Validation

Automated tests:

- 30 focused autocorrelation tests passed
- 309 total project tests passed

Manual notebook verification was added in:

`notebooks/learning/04_correlation_and_blind_estimation.ipynb`

Verified:

- period-3 sequence produced peaks at lags 0, 3, 6, 9, ...
- complex IQ tone preserved unit correlation magnitude and showed the expected phase progression
- lag-1 correlation phase matched `2πf/Fs`
- white noise showed approximately zero non-zero-lag correlation

### Result

PASS — IQWAV now has verified real/complex autocorrelation primitives suitable for later periodicity analysis, blind parameter estimation, synchronization, framing, and bitstream analysis.


## 2026-08-30 — Wideband OTA FM channelization and demodulation verified

### Experiment

Extended real-world FM validation using a wideband Mumbai broadcast-FM IQ capture:

- center frequency: 92.3 MHz
- sample rate: 10 MS/s
- file size: approximately 880 MB
- duration: approximately 11 seconds
- format: complex64
- capture contained multiple broadcast-FM stations

The external IQ recording remains under `data/external/` and is not committed to Git.

### Wideband analysis

A wideband PSD covering approximately 87.3–97.3 MHz showed multiple distinct FM broadcast stations.

A strong station around 92.7 MHz was selected for further processing.

### Channelization

The selected station was approximately +400 kHz relative to the 92.3 MHz recording center.

Processing performed:

`wideband IQ`
→ complex-IQ DC removal
→ frequency translation by approximately -400 kHz
→ target station centered near 0 Hz
→ anti-alias filtering and 40× decimation
→ 10 MS/s reduced to 250 kS/s

The resulting PSD confirmed that the selected FM channel remained while neighboring wideband stations were removed.

### FM demodulation

The isolated channel was processed using the production:

`fm_demodulate()`

The demodulated multiplex spectrum showed structure consistent with broadcast FM:

- strong 0–15 kHz program audio,
- clear ~19 kHz stereo pilot,
- energy in the 23–53 kHz stereo-difference region,
- a feature near the ~57 kHz RDS region.

No stereo or RDS decoding was performed.

### Audio recovery

Mono-compatible audio was recovered using:

`FM multiplex`
→ 15 kHz low-pass filtering
→ demodulated DC removal
→ 50 µs FM de-emphasis
→ resampling from 250 kS/s to 50 kS/s
→ normalization
→ 16-bit WAV

The complete capture produced approximately 11 seconds of clear, intelligible broadcast audio.

Processing of the large recording was performed in chunks rather than loading the entire capture into expanded complex arrays.

### Result

PASS — IQWAV successfully processed a genuine wideband multi-station OTA capture, selected and channelized one FM station, demodulated it with the production FM discriminator, identified expected multiplex structure, and recovered clear audio.


## 2026-08-30 — FM demodulation productionized and real-data verified

### Implementation
Added reusable FM phase-discriminator support:

- `src/iqwav/demod/analog.py`
  - `fm_demodulate(samples)`
- exported through `src/iqwav/demod/__init__.py`
- added focused unit tests in:
  - `tests/unit/test_analog_demodulation.py`

The discriminator computes:

`angle(samples[1:] * conj(samples[:-1]))`

and returns phase increment in radians/sample.

It intentionally does not perform:
- `Fs/(2π)` scaling,
- DC removal,
- filtering,
- resampling,
- normalization,
- de-emphasis,
- stereo decoding,
- carrier/CFO estimation.

### Automated verification
- Focused FM-demodulation tests: 11 passed.
- Full project suite: 279 passed.

Tests cover:
- output shape and dtype,
- positive and negative phase increments,
- wrapped phase differences,
- amplitude invariance,
- invalid real/multidimensional input,
- insufficient samples,
- NaN/Inf rejection.

### Real OTA integration verification
The production `fm_demodulate()` function replaced the manual discriminator in:

`notebooks/experiments/02_real_fm_demodulation.ipynb`

Using the genuine 99.5 MHz broadcast-FM IQ recording, the production function successfully produced the same demodulated multiplex spectrum and recovered approximately 4 seconds of clean, clearly intelligible English audio after low-pass filtering and resampling.

### Result
PASS — the reusable FM discriminator is unit-tested, integration-tested and manually verified on genuine OTA IQ data.


## 2026-08-30 — Real OTA IQ smoke test passed

### What was tested
- Downloaded a genuine over-the-air FM IQ recording:
  `fm_rds_250k_1Msamples.iq`
- Known metadata:
  - sample rate: 250 kHz
  - center frequency: 99.5 MHz
  - format: complex64 / interleaved float32 I,Q
  - 1,000,000 complex samples
  - duration: 4 seconds
- Stored locally under:
  `data/external/fm_rds_250k_1Msamples.iq`
  and kept out of Git by `.gitignore`.

### IQWAV path exercised
- `load_raw_iq()`
- `magnitude_spectrum()`
- `welch_psd()`
- `spectrogram_data()`

### Verification
- IQWAV loader matched direct NumPy complex64 loading.
- Time-domain I/Q samples looked physically plausible.
- FFT showed a broad real FM spectrum across the expected ±125 kHz Nyquist span.
- Welch PSD showed consistent occupied spectral structure.
- Waterfall showed time-varying broadband FM energy with sensible frequency/time orientation.
- No obvious corruption, axis error, clipping, or file-format mismatch was observed.

### Result
PASS — current IQWAV raw-IQ ingestion and basic spectral-analysis foundation successfully processed a genuine OTA SDR capture.

### Notes
- Real data is visibly less ideal than synthetic data: asymmetry, spectral bumps, offsets, and time-varying structure are present.
- These effects should not be artificially cleaned up at this stage; future estimators must handle them.
- No FM demodulation or blind parameter estimation was performed in this milestone.



## 2026-08-30 — WAV and Raw IQ File Ingestion Implemented

### Added

Created:

- `src/iqwav/io/wav.py`
- `src/iqwav/io/raw_iq.py`

Implemented:

- `load_wav(path)`
- `load_wav_iq(path, i_channel=0, q_channel=1)`
- `load_raw_iq(path, dtype=np.float32, iq_order="IQ")`

### Capability

IQWAV can now load signal recordings from disk instead of operating only on arrays generated inside Python.

Supported input paths:

- standard WAV files,
- multi-channel WAV interpreted explicitly as I/Q,
- headerless interleaved raw IQ files.

### WAV Behavior

`load_wav`:

- returns WAV sampling rate and samples,
- preserves SciPy-loaded dtype and values,
- supports mono and multi-channel WAV,
- performs no amplitude normalization,
- does not automatically guess I/Q channel meaning.

`load_wav_iq`:

- requires at least two WAV channels,
- explicitly selects I and Q channels,
- combines them as `I + jQ`,
- returns a one-dimensional `complex128` IQ array.

### Raw IQ Behavior

`load_raw_iq`:

- reads headerless raw files with `np.fromfile`,
- supports explicit `"IQ"` or `"QI"` interleaving,
- supports real scalar dtypes such as float32 and int16,
- returns `complex128` IQ samples.

It deliberately does not infer:

- dtype,
- endianness,
- IQ ordering,
- sampling rate,
- center frequency.

These must currently be provided from metadata or operator knowledge.

### Tests

Added:

`tests/unit/test_io.py`

Current total:

- 268 tests passing.

Tests verify:

- mono WAV round-trip,
- stereo WAV round-trip,
- sampling-rate preservation,
- dtype/value preservation,
- exact WAV I/Q reconstruction,
- alternate channel selection,
- float32 raw IQ reconstruction,
- int16 raw IQ reconstruction,
- QI ordering,
- invalid path/input/channel/order/dtype handling.

### Manual Verification

Created:

`notebooks/learning/03_file_io_and_signal_analysis.ipynb`

Verified the complete path:

`known IQ signal`
→ save as WAV/raw IQ
→ reload from disk
→ reconstruct complex IQ
→ FFT / PSD / spectrogram.

A known 125 Hz complex IQ tone was recovered from both WAV and raw IQ files, and FFT analysis detected the expected 125 Hz spectral peak.

The WAV-loaded and raw-loaded IQ arrays matched each other and matched the original signal within expected floating-point precision.

### Current Capability

IQWAV now supports:

`file on disk`
→ WAV/raw IQ ingestion
→ complex NumPy IQ samples
→ FFT
→ PSD
→ spectrogram
→ existing DSP and demodulation utilities.

### Limitation

Raw IQ is headerless, so its representation cannot currently be determined automatically.

### Next

Begin analysis of externally sourced/real IQ recordings rather than only self-generated files.



## 2026-08-30 — Known-Timing BPSK/QPSK Demodulation Implemented

### Added

Created:

`src/iqwav/demod/digital.py`

with:

- `bpsk_demodulate(samples, samples_per_symbol)`
- `qpsk_demodulate(samples, samples_per_symbol)`

### Capability

IQWAV can now perform known-timing hard-decision demodulation for BPSK and QPSK.

Receiver assumptions:

- symbol boundaries are already known,
- no timing recovery,
- no carrier recovery,
- no CFO correction,
- no phase correction.

For each symbol interval, samples are block-averaged and then mapped back to bits using the corresponding decision regions.

### BPSK Decision Rule

- `real(symbol_average) >= 0` → bit `0`
- `real(symbol_average) < 0` → bit `1`

### QPSK Decision Rule

Using the existing Gray mapping:

- `I >= 0, Q >= 0` → `00`
- `I < 0, Q >= 0` → `01`
- `I < 0, Q < 0` → `11`
- `I >= 0, Q < 0` → `10`

### Tests

Added:

`tests/unit/test_digital_demodulation.py`

Current total:

- 251 tests passing.

Tests verify:

- clean BPSK round-trip,
- clean QPSK round-trip,
- all four QPSK Gray-mapped quadrants,
- `samples_per_symbol = 1`,
- real BPSK input,
- correct recovered bit counts,
- successful seeded recovery after moderate AWGN,
- invalid input handling.

### Manual Verification

Verified end-to-end synthetic communication chains:

`bits → BPSK waveform → AWGN → BPSK demodulation → recovered bits`

and:

`bits → QPSK waveform → AWGN → QPSK demodulation → recovered bits`

Recovered bits matched the transmitted bits in the controlled notebook test.

### Current Capability

IQWAV now supports:

`bits`
→ BPSK/QPSK symbol mapping
→ sampled rectangular waveform
→ AWGN / CFO / phase impairment injection
→ known-timing hard-decision demodulation
→ recovered bits.

### Limitation

The receiver currently assumes perfect symbol timing and does not estimate or correct timing, carrier frequency offset, or phase.

### Next

Add real `.wav` and raw IQ file ingestion before expanding receiver complexity.


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