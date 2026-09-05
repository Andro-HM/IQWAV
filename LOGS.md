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

## 2026-09-06 — Module 11D: known-SPS integer symbol-timing recovery

### Decision

Candidate A — known-SPS transition-residue timing — is the production
Module 11D estimator.

Candidate B — whole-block least-squares timing — is **PARKED** because
it showed no benefit on the tested rectangular BPSK/QPSK campaign: A
and B never disagreed, and B saved no record. PARKED means keep it as
a future candidate. It is not scientifically disproven, not a rejected
algorithm, and not a demonstrated failure of the least-squares
objective. The head-to-head script is preserved:

`scripts/exp_11d_timing_headtohead.py`

Costas/PLL remain parked. No interpolator, resampler, circular shift,
Gardner, Mueller-Muller, or timing loop was added.

### Architecture

```text
estimate_symbol_timing(samples, samples_per_symbol)
    = measure integer boundary offset given known SPS

synchronization.correct_symbol_timing(
    samples, samples_per_symbol, boundary_offset
)
    = drop the leading partial symbol and trim the trailing incomplete
      samples
```

There is no `dsp.apply_timing_offset`. The synthetic impairment is a
crop of an aligned rectangular generator waveform, matching the existing
symbol-grid tests.

`estimate_rectangular_symbol_grid` remains a baud/grid estimator. Its
`boundary_offset` is still not timing recovery. Module 11D is the
known-SPS specialization that actually aligns the block. The grid
estimator's single-transition reject is unchanged: unknown-period search
cannot identify SPS from one impulse. Known-SPS timing can.

### Production algorithm (Candidate A)

```text
d[n] = |x[n] - x[n-1]|
accumulate d into bin n % SPS
boundary_offset = argmax bin
quality = (concentration - 1/SPS) / (1 - 1/SPS)
```

Reuses `_transition_profile` and `_score_period` from
`src/iqwav/estimation/symbol_grid.py` at the single known period.
No `min_quality` argument. Reject only when total transition magnitude
is zero. One clean transition remains estimable.

Canonical truth for `observed = clean[C:]`:

```text
boundary_offset = (-C) % SPS ∈ [0, SPS)
```

Correction:

```text
aligned = observed[offset : offset + n_complete * SPS]
n_complete = (N - offset) // SPS
```

### Head-to-head evidence (experiment, not unit tests)

4800 main records, rectangular BPSK/QPSK, `N_SYMBOLS=512`,
`SPS ∈ {2,4,8,16}`, every crop residue, 5 bit seeds × 5 noise seeds.

```text
clean     A 300/300 exact    B 300/300 exact    disagree 0
20 dB     A 1500/1500        B 1500/1500        disagree 0
10 dB     A 1500/1500        B 1500/1500        disagree 0
0 dB      A 1500/1500        B 1500/1500        disagree 0  (characterization)
```

Clean aligned output equalled the known clean slice for both (300/300).
After either correction, BER/SER matched the oracle-aligned floor
(0 at clean/20/10 dB). Unaligned `delay=0` BER stayed ~12–23%.
One clean transition: A 60/60, B 60/60. Zero transitions: both
unidentifiable, all residues tied, both report 0. Phase 0.8 rad and
amplitude ×3.7: both invariant. B's common-support score never disagreed
with `SSE/n_used`. B was 6–31× slower and saved no record.

B is therefore PARKED for lack of benefit in this campaign, not
because it was scientifically disproven.

### Public API

```python
@dataclass(frozen=True)
class SymbolTimingEstimate:
    boundary_offset: int
    quality: float

estimate_symbol_timing(samples, samples_per_symbol) -> SymbolTimingEstimate
correct_symbol_timing(samples, samples_per_symbol, boundary_offset)
```

`quality` is a chance-corrected concentration diagnostic, not calibrated
confidence, probability, or SNR.

### Files

- `src/iqwav/estimation/symbol_timing.py`
- `src/iqwav/synchronization/timing.py`
- `tests/unit/test_symbol_timing.py`
- `tests/unit/test_timing_correction.py`

### Automated validation

- focused 11D: 90 passed
- related regressions (symbol grid/rate, demod, waveform, 11A/11B/11C):
  374 passed
- full suite: 861 passed, 0 failed, 0 skipped
  (previous accepted 11C baseline 771, plus 90 Module 11D tests)

### Scope limitation

Integer-sample, known-SPS, block-level alignment for rectangular IQWAV
BPSK/QPSK. Not fractional timing, not RRC/pulse-shaping, not timing
drift, not baud re-estimation, not 1-sps (SPS must be >= 2), not real
OTA PSK validation. Zero-transition blocks are unidentifiable.
Synthetic 20 dB success is not real-world RF robustness.

### Next

Rectangular BPSK/QPSK synchronization for this signal model now has
coarse CFO (11A), static phase (11B), residual CFO (11C), and integer
timing (11D). Costas/PLL stay parked. Do not start AMC/Module 12, FEC,
or post-Snapshot-B Sayan timing work unless HM explicitly asks.


---

## 2026-09-06 — Module 11C production: whole-block M-th-power residual CFO

### Decision

The lag-1 M-th-power residual-frequency estimator recorded in the
2026-09-05 11C LOGS entry is **REJECTED** as the production algorithm.

The production Module 11C estimator is the frozen whole-block M-th-power
tone-frequency method (Astra-inspired): global magnitude scaling, raise
to M, maximise whole-block coherent tone power with 8x zero-padded FFT
initialisation and local Brent refinement, canonical M-th-power search
range, no deadband.

Costas/PLL remain parked. Timing recovery has not started.

### Why lag-1 11C was rejected

On the motivating clean QPSK block (seed 102, +1000 Hz, `Fs=80 kHz`,
SPS=8) lag-1 11C did remove the documented ~4.433 Hz leftover and 11B
then accepted. That clean success does **not** restore 11B under 20 dB
AWGN.

Same QPSK chain, 20 dB, noise seeds 1–3 (physical leftover after 11A,
then lag-1 11C):

```text
seed 1: 11A leftover +0.772 Hz → lag-1 leftover +1.636 Hz
        11B accept, symmetry 0.095, wrap_err +0.537 rad
seed 2: 11A leftover -0.012 Hz → lag-1 leftover -0.399 Hz
        11B accept, symmetry 0.413, wrap_err -0.514 rad
seed 3: 11A leftover -3.612 Hz → lag-1 leftover +1.038 Hz
        11B accept, symmetry 0.146, wrap_err +0.554 rad
```

Lag-1 **increased** an already-small 11A leftover on seeds 1 and 2.
Default 11B acceptance was not restoration: wrap error stayed ~0.5 rad
and symmetry collapsed. The lag-1 `coherence` / `min_coherence` gate
was not the failure mode and was not tuned.

That candidate remains in this file as historical evidence (entry below).
It is not the active estimator.

### Production algorithm

```text
z = (x / mean(|x|)) ** M          # BPSK M=2, QPSK M=4
P(df) = |sum z[n] exp(-j 2π M df n / Fs)|^2
```

- 8x zero-padded FFT: global peak inside the canonical residual range
- Brent maximises the same `P(df)` in a local window of 8 padded-FFT
  residual bins around that peak (`xatol=1e-6` Hz)
- no deadband, no per-record tuning
- correction reuses `correct_frequency_offset`

Canonical unambiguous ranges:

```text
BPSK: [-Fs/4, +Fs/4)
QPSK: [-Fs/8, +Fs/8)
```

Alias period `Fs/M`. This does not replace coarse 11A acquisition.

### Public API

```python
estimate_residual_frequency_offset(samples, fs, modulation)
    -> ResidualFrequencyOffsetEstimate(residual_frequency_hz, phase_increment_rad)
```

`phase_increment_rad = 2π * residual_frequency_hz / fs`.

Lag-1 `coherence` and `min_coherence` are **not** preserved. Those names
were lag-1 statistics of the rejected estimator. The validated
whole-block method has no lag-1 reject gate; keeping the field would
imply a reliability contract that was never validated for this
algorithm. 11C was never a committed production API, so this is not a
break of an accepted public surface. 11A `FrequencyOffsetEstimate.coherence`
is unchanged.

### Validation evidence (experiments, not unit-test campaigns)

Stage A (script, original 20 dB QPSK chain, seeds 1–3):
whole-block physical leftover 0.00017 / 0.00053 / 0.00048 Hz; 11B
symmetry ~0.962 matching oracle; 0 BER after 11B phase correction.

Stage B (100 QPSK + 100 BPSK held-out, still ±50 Hz development cap):
BPSK near oracle (max leftover 1.22e-3 Hz, 0 11B rejects, 0 BER).
QPSK median leftover 3.96e-4 Hz, but 8/100 11B rejects and BER 3.9%
from records whose 11A leftover was ~55–57 Hz, **outside ±50 Hz**.
Whole-block never worsened |11A leftover|. Worst five all saturated
the artificial cap at ~+49.4 Hz.

Stage B2 (same 200 records, now development data; only change is
canonical search range): all eight QPSK failures disappeared. QPSK
max leftover 1.62e-3 Hz; BPSK unchanged; 0 11B rejects; 0 BER; 0 alias;
0 wrong-peak; 0 worsenings. Runtime unchanged (~21 ms median).

Stage C (fresh held-out 20 dB, bits 30000–30009 × noise 40000–40009,
canonical range):

```text
200 records, whole-block vs 11A-only vs oracle
bias +1.48e-5 Hz, RMSE 5.15e-4 Hz
|phys| median/p95/max = 3.38e-4 / 1.03e-3 / 1.55e-3 Hz
alias/wrong-peak 0/0; worsens |11A| 0/200
11B rejects 0/200 (11A-only: 122/200)
BER 0 / 1228800; SER 0 / 819200
runtime median/p95 20.2 / 22.8 ms
```

Stage D (no 11A; exact injected residuals 0, ±0.001, ±0.01, ±0.1, ±1,
±4.433, ±5 Hz; 3×20 fresh seeds; 1560 records):

```text
QPSK: bias -4.01e-5 Hz, RMSE 5.33e-4 Hz, max |phys| 1.86e-3 Hz
BPSK: bias +1.41e-5 Hz, RMSE 4.58e-4 Hz, max |phys| 1.79e-3 Hz
true 0 Hz invented |CFO| max 1.12e-3 (QPSK) / 1.21e-3 (BPSK)
all 60/60 zero-Hz records increase |residual| as expected
no |phys| > 0.005 Hz; 0 alias; 0 wrong-peak
11B rejects 0; BER 0
no deadband added: invented millihertz error is downstream-negligible
```

The 0.005 Hz leftover bound is a research target used in these
campaigns, not a permanent product specification.

### Automated validation

- focused residual-frequency tests: 70 passed
- related regressions (coarse CFO, static phase, impairments,
  modulation, demodulation, waveform): 261 passed
- full suite: 771 passed, 0 failed, 0 skipped
  (committed 11B baseline 701, plus 70 whole-block 11C tests;
   the uncommitted lag-1 11C suite of 90 tests was replaced)

### Scope limitation

Constant residual CFO, known rectangular IQWAV BPSK/QPSK (integer-SPS
or ideal 1-sps samples), 20 dB AWGN rectangular campaign. Not carrier
tracking, not a Costas loop, not a PLL, not timing recovery, not AMR,
not pulse-shaped PSK, not a replacement for coarse 11A. Canonical range
`[-Fs/(2M), Fs/(2M))`. Synthetic held-out success is not real-world RF
validation.

### Next

Costas/PLL stay parked. Symbol timing recovery remains the next
synchronization stage after this residual-frequency refinement.

---

## 2026-09-05 — REJECTED / historical: lag-1 M-th-power residual-frequency candidate (not production 11C)

This entry is **not** the production Module 11C algorithm. It is the
experimental record of a lag-1 M-th-power residual estimator that was
**rejected** after the 20 dB QPSK downstream 11A→11C→11B chain failed
(see 2026-09-06). Evidence below is preserved; do not treat the APIs or
“next milestone” sentences in this entry as current.

### Scope

This historical experiment implemented only:

- one-shot residual constant-CFO estimation for rectangular IQWAV BPSK/QPSK
- reuse of existing `correct_frequency_offset` (no new correction primitive)

This is bounded leftover-frequency refinement after coarse 11A correction,
not general carrier tracking.

Not implemented, deliberately: Costas loop, PLL, NCO, loop filter,
time-varying frequency tracking, timing recovery, AMC/AMR, pulse-shaped /
RRC / fractional-SPS support, pipeline integration, and any Sayan
post-Snapshot-B work. Costas/PLL remain parked until evidence shows that
constant residual-frequency correction is insufficient.

Validated only on:

- rectangular integer-SPS IQWAV BPSK/QPSK waveforms
- ideal 1-sample-per-symbol samples of those same constellations

This does not imply support for arbitrary pulse-shaped oversampled PSK.

### Architecture

The existing inject / measure / remove separation is preserved. 11C adds
only the residual-frequency measurement:

```text
dsp.apply_frequency_offset()                      = inject known CFO (existing)
estimation.estimate_frequency_offset()            = coarse lag-1 CFO (existing)
synchronization.correct_frequency_offset()        = remove supplied CFO (existing)
estimation.estimate_residual_frequency_offset()   = residual M-th-power CFO (new)
synchronization.correct_frequency_offset()        = remove leftover CFO (reuse)
estimation.estimate_phase_offset()                = static phase (existing 11B)
synchronization.correct_phase_offset()            = remove static phase (existing)
```

Public API:

- `estimate_residual_frequency_offset(samples, fs, modulation, *, min_coherence=0.05)`
  returning a frozen
  `ResidualFrequencyOffsetEstimate(residual_frequency_hz, phase_increment_rad, coherence)`

`phase_increment_rad` is the per-sample residual-carrier increment of the
original samples, so
`residual_frequency_hz = fs * phase_increment_rad / (2*pi)`.
Callers pass `residual_frequency_hz` into existing
`correct_frequency_offset`.

The coarse lag-1 estimator was not changed.

### Estimator mathematics

After global magnitude scale `u = x / mean(|x|)`, raise to the constellation
power M (BPSK M=2, QPSK M=4) and take the overlap-normalized lag-1 product
of `z = u**M`:

```text
r1     = mean(z[1:] * conj(z[:-1]))
theta  = wrap(angle(r1)) into [-pi, +pi)
df_hat = fs * theta / (2*pi*M)
```

`c_M` constellation-reference compensation is not used: it is a constant
and cancels in the lag-1 product. Dividing the powered increment by M
recovers the original-sample residual CFO, not M times that CFO.

Canonical unambiguous ranges from wrapping the powered increment into
`[-pi, +pi)`:

```text
BPSK (M=2): [-fs/4, +fs/4)
QPSK (M=4): [-fs/8, +fs/8)
```

Offsets outside that interval alias with period `fs/M`. That range is
narrower than coarse lag-1 (`(-fs/2, fs/2]`), which is why 11C does not
replace 11A. Near-boundary recovery was tested inside the half-open
interval; exact open-endpoint uniqueness (`+fs/(2M)` vs `-fs/(2M)`) is
not claimed.

### Reliability measure

Overlap-normalized lag-1 coherence using the same overlap on both sides:

```text
coherence = |mean(z[1:] * conj(z[:-1]))|
            / sqrt(mean(|z[1:]|**2) * mean(|z[:-1]|**2))
```

Cauchy-Schwarz bounds this in `[0, 1]`. It is a reliability diagnostic
under the known rectangular BPSK/QPSK model only — NOT calibrated
confidence, NOT SNR, and NOT a general detector that rejects every
unstructured or mismatched signal. `min_coherence=0.05` follows the
existing HM estimator threshold style; 0 is allowed for diagnostic use.
No threshold was tuned to make tests pass.

### Scientific validation

- Clean rectangular BPSK/QPSK leftovers `{0, ±1, ±4.433, ±5, ±50, ±200}` Hz
  recovered to ~1e-12–1e-6 Hz; coherence = 1.0.
- M-scale regression: a 5 Hz leftover returns 5 Hz, not `M*5` Hz.
- Zero-CFO QPSK reports 0 Hz; constellation orientation does not invent
  a frequency.
- Static phase and amplitude scaling leave the frequency estimate unchanged.
- Ideal symbol-rate (1 sps) samples of the same constellations recover
  injected leftovers exactly. Not a pulse-shaped claim.
- Uncorrected QPSK `+1000 Hz` (inside `±fs/8` at 80 kS/s): 11C reports
  `1000.000 Hz`; lag-1 on the same block remains `+1004.433 Hz`. This
  shows 11C is unbiased on the documented finite-record QPSK case; it is
  not permission to delete coarse acquisition.
- After 11A correct-by-own-estimate on that QPSK block:

```text
true CFO                     = +1000.000 Hz
HM lag-1 estimate            = +1004.433 Hz
independent leftover after 11A = -4.433 Hz
11C leftover estimate          = -4.433 Hz
independent leftover after 11C = ~0 Hz (~6e-13 Hz)
```

  PRIMARY truth is the independent clean-reference phase-slope residual.
  Re-running 11C after correcting by its own estimate reports 0 Hz and is
  labeled self-consistency only.
- Motivating 11B chain (QPSK, `phi=0.8`, CFO `+1000 Hz`): after 11A alone,
  phase symmetry = 0.0323 and 11B rejects. After 11A then 11C, independent
  leftover ~0 Hz, 11B accepts with symmetry = 1.0. The returned phase is
  `0.8 - pi/2` (rotational ambiguity); wrapped residual vs 0.8 is ~0.
  Absolute bit labeling is still not recovered.
- 5 Hz stale-CFO block (QPSK seed 202, 0.2048 s): independent leftover
  before 11C = +5.000 Hz, 11B symmetry = 0.023. After 11C, independent
  leftover ~0 Hz and 11B accepts (symmetry = 1.0 at injected phase 0.3).
- BPSK 11A→11C→11B still works (coarse was already exact on that block).
- Alias limitation: QPSK leftover `+12000 Hz` (`> fs/8 = 10000 Hz`)
  returns the wrapped principal value `-8000 Hz` with high coherence.
  That is not successful recovery of the true offset.
- AWGN (deterministic noise seeds 1-3, true leftover 5 Hz), worst
  leftover errors and coherence:

```text
bpsk  20 dB: 0.84 Hz, coherence ~0.961
bpsk  10 dB: 9.33 Hz, coherence ~0.70
bpsk   0 dB: 166 Hz,  coherence ~0.14  (passes default threshold)
qpsk  20 dB: 1.93 Hz, coherence ~0.856
qpsk  10 dB: 36.5 Hz, coherence ~0.29
qpsk   0 dB: thousands of Hz, coherence ~0.004–0.007 (rejected)
```

  Asserted only: BPSK 20/10 dB and QPSK 20 dB, with gates above the
  measured worst errors. 10 dB QPSK and 0 dB were characterized and
  deliberately not used as pass gates. Fourth-power lag-1 degrades
  quickly; default `min_coherence` does not reject every poor estimate
  (BPSK 0 dB is the recorded example). No universal low-SNR robustness
  is claimed. This AWGN limitation is not evidence for a PLL: the
  motivating leftover is a constant frequency on a clean rectangular
  block, which 11C removes.
- Pure noise: this N=65536 record is rejected by the default threshold
  (coherence 0.0020 / 0.0025). Diagnostic mode (`min_coherence=0`) on
  N=4096 exposes low coherence (0.0061 / 0.0069). These are recorded
  diagnostic cases, not a claim that `min_coherence` rejects every
  unstructured or mismatched signal.

### Automated validation

- focused residual-frequency tests: 90 passed
- related regressions (coarse CFO estimation/correction, static phase
  estimation/correction, impairments, modulation, demodulation,
  waveform): 261 passed
- full suite: 791 passed, 0 failed, 0 skipped (baseline 701 + 90 new)

### Scope limitation

Constant residual CFO, known rectangular IQWAV BPSK/QPSK (oversampled
rectangular integer-SPS or ideal symbol-rate samples), one-shot block
estimate only. Not carrier tracking, not a Costas loop, not a PLL, not
timing recovery, not AMR, not pulse-shaped PSK. The unambiguous range is
`[-fs/(2M), fs/(2M))`. Coherence is not calibrated confidence.

### Next Module 11 milestone

Symbol timing recovery remains the remaining synchronization stage
(coarse CFO, residual-frequency refinement, and static carrier phase
are now done). Costas/PLL stay parked unless later evidence shows
constant residual-frequency correction is insufficient.

---

## 2026-09-05 — Module 11B: blind static carrier phase estimation and correction

### Scope

Implemented only:

- blind constant phase-offset estimation for IQWAV BPSK/QPSK
- caller-supplied constant phase correction

Not implemented, deliberately: Costas loop, PLL, dynamic phase tracking,
timing recovery, Gardner/Mueller-Muller, interpolation, AMC/AMR,
pipeline integration, and any Sayan post-Snapshot-B work.

### Architecture

The existing inject/measure/remove separation is preserved:

```text
dsp.apply_phase_offset()             = inject known phase impairment (existing, untouched)
estimation.estimate_phase_offset()   = estimate unknown static phase (new)
synchronization.correct_phase_offset() = remove supplied static phase (new)
```

Public APIs:

- `estimate_phase_offset(samples, modulation, *, min_symmetry=0.05)`
  returning a frozen `PhaseOffsetEstimate(phase_offset_rad, symmetry)`
- `correct_phase_offset(samples, phase_offset_rad)` performing only
  `y[n] = x[n] * exp(-j * phase_offset_rad)`, complex128, new array,
  exact inverse of `apply_phase_offset`

The estimator needs no `samples_per_symbol`, no known timing and no
symbol-boundary recovery; it works on rectangular oversampled waveforms
and equally on already-extracted symbol-rate samples.

### Estimator mathematics

M-th-power moment with canonical constellation-reference compensation:

```text
compensated_moment = mean(x**M) * conj(c_M)
phi_hat            = angle(compensated_moment) / M
```

- BPSK: M = 2, `c_M = +1` (symbols are +1/-1, so `s**2 = +1`), i.e.
  `phi_hat = angle(mean(x**2)) / 2`
- QPSK: M = 4, `c_M = -1` (the Gray mapper places every symbol at
  `pi/4 + k*pi/2`, so `s**4 = -1`), i.e. estimation from `-mean(x**4)`

The `-1` QPSK fourth-power reference was verified symbol by symbol:
every canonical ideal symbol satisfies `s**4 == -1` within 4.4e-16. A
naive uncompensated `angle(mean(x**4))/4` reports `pi/4`
(0.785398163 rad measured) on zero-phase IQWAV QPSK — the fixed
constellation-orientation bias the compensation removes. A dedicated
zero-phase regression test guards against this bug.

Canonical returned ranges use explicit deterministic wrapping (not
`np.angle`'s branch choice at +/-pi):

```text
BPSK: [-pi/2, +pi/2)   observable modulo pi
QPSK: [-pi/4, +pi/4)   observable modulo pi/2
```

### Reliability measure

```text
symmetry = abs(mean(x**M)) / mean(abs(x)**M)
```

computed on magnitude-normalized samples (the positive real
normalization leaves phase and ratio unchanged while preventing
overflow/underflow). Bounded in [0, 1]: 1 when all `x**M` share one
phase, order `1/sqrt(N)` for pure noise. It is a symmetry/concentration
reliability measure only — NOT calibrated confidence and NOT SNR.
`min_symmetry=0.05` follows the existing HM estimator threshold style
(matching `min_coherence`); 0 is allowed for diagnostic use. No
threshold was tuned to make tests pass.

### Scientific validation

- Clean BPSK/QPSK: exact recovery (~1e-16) for zero, positive,
  negative, outside-canonical and near-boundary phases; symmetry
  exactly 1.0; boundary cases (+pi/2, +pi/4) wrap deterministically.
- Estimate -> correct proof: the PRIMARY truth is the independent
  reference-ratio measurement `angle(mean(corrected/clean))` wrapped
  modulo pi (BPSK) / pi/2 (QPSK). Re-running the same estimator after
  correction is kept only as an explicitly labeled self-consistency
  check (the Module 11A lesson is deliberately not repeated).
- Rotational ambiguity demonstrated concretely: for true phase 0.8 the
  QPSK estimate returned `0.8 - pi/2` and correction produced
  `j * clean` — a valid constellation with different bits. Absolute
  bit labeling cannot be recovered from PSK rotational symmetry alone;
  pilots, differential coding, framing or known headers are needed.
- CFO isolation (correct TRUE CFO first, then estimate phase): exact
  for both modulations.
- BPSK coarse-CFO integration: exact (the lag-1 CFO estimator is
  essentially exact on this deterministic BPSK block).
- QPSK coarse-CFO integration: REJECTED by the reliability threshold
  (symmetry 0.0323). The known +4.433 Hz finite-record lag-1 bias
  (Module 11A) rotates the constellation through many cycles over the
  0.4096 s block. Documented as instability: Module 11B correctness
  does NOT depend on the CFO estimator being exact.
- Residual-CFO degradation characterized: CFO knowledge stale by 5 Hz
  over a 0.2048 s block (about one full constellation rotation)
  collapses symmetry from 1.0 to 0.023 and the default threshold
  rejects the input. Characterized, not solved — no PLL in this
  milestone.
- AWGN (deterministic noise seeds 1-3, true phase 0.8), worst wrapped
  errors and symmetry:

```text
bpsk  20 dB: 2.7e-4 rad, symmetry ~0.990
bpsk  10 dB: 1.1e-3 rad, symmetry ~0.909
bpsk   0 dB: 5.6e-3 rad, symmetry ~0.50
qpsk  20 dB: 6.0e-4 rad, symmetry ~0.961
qpsk  10 dB: 1.1e-3 rad, symmetry ~0.70
qpsk   0 dB: 1.5e-2 rad, symmetry ~0.14
```

  Also measured but deliberately NOT asserted: at -3 dB, BPSK stays
  accurate (worst 1.1e-2 rad, symmetry ~0.33) while QPSK symmetry
  drops to 0.054-0.068, near the default threshold. No universal
  low-SNR robustness is claimed.
- Pure noise: 65536-sample records are rejected by the default
  threshold (symmetry 0.0024 / 0.0075); diagnostic mode (`min_symmetry=0`)
  exposes explicitly low symmetry (0.018 / 0.044 at 4096 samples)
  instead of presenting an arbitrary phase as truth.

### Automated validation

- focused phase-estimation tests: 73 passed
- focused phase-correction tests: 27 passed
- related regressions (impairments, CFO estimation/correction,
  modulation, demodulation, waveform): 161 passed
- full suite: 701 passed, 0 failed, 0 skipped (baseline 601 + 100 new)

### Scope limitation

Static, known-modulation, constant block phase only. Not carrier
tracking, not a Costas loop, not a PLL, not timing recovery, not AMR.
The M-th-power method assumes the modulation family is already known
("bpsk"/"qpsk" only; arbitrary M-PSK is deliberately not advertised).
The symmetry measure is not calibrated confidence.

### Next Module 11 milestone

Symbol timing recovery is the remaining synchronization stage in the
Module 11 roadmap (CFO correction and static carrier phase are now
done).

---

## 2026-09-04 — Module 11A: controlled CFO correction integrated from Sayan snapshot B

### Source

Sayan frozen snapshot B:

`6bb80f6d6e522578a347e623d308263f055a32d9` (`Add controlled CFO correction`)

We deliberately used and adapted Sayan's correction primitive rather than
independently reimplementing the same idea.

### Mathematical convention (preserved from Sayan)

```text
r[n] = s[n] * exp(+j*2*pi*df*n/fs)
y[n] = r[n] * exp(-j*2*pi*df*n/fs)
```

with `n = 0 .. N-1`. A positive supplied CFO applies a negative correcting
rotation; supplying the true offset exactly inverts
`apply_frequency_offset()`.

### Architecture

Sayan placed the primitive under `dsp/frequency_correction.py`; IQWAV
places receiver-side correction under
`src/iqwav/synchronization/frequency.py`.

```text
dsp.apply_frequency_offset()          = inject/simulate CFO
estimation.estimate_frequency_offset() = measure coarse CFO
synchronization.correct_frequency_offset() = remove supplied CFO
```

Public API:

- `correct_frequency_offset(samples, fs, frequency_offset_hz)`

### Validation improvements over Sayan

- bool `fs` rejected
- bool frequency offset rejected
- HM validation conventions throughout
- explicit static-phase-vs-frequency-slope semantics
- estimate → correct integration validation

Static phase is NOT removed by CFO correction: correction removes the phase
slope with time, so a constant rotation `exp(j*phi0)` remains and the
constellation may become stationary yet stay rotated.

### Scientific validation finding

Using the same lag-1 CFO estimator both before and after correction can
self-cancel its finite-record QPSK bias and falsely report approximately
zero residual CFO. The same-estimator chain test is therefore documented as
an estimator self-consistency check, not proof of zero physical residual.

An independent truth-based synthetic test was added: after correction,
divide by the known clean QPSK reference, unwrap the ratio phase, fit the
phase slope versus sample index, and convert the slope to Hz.

Deterministic QPSK example (fs = 80 kS/s, SPS = 8):

```text
true CFO                     = +1000.000 Hz
HM estimated CFO             = +1004.433 Hz
expected physical residual   = -4.433 Hz
independently measured residual = -4.433 Hz
same-estimator residual      = ~0 Hz
```

The BPSK controlled case estimated +1000 Hz essentially exactly and produced
approximately zero independently measured residual.

### Automated validation

- focused frequency-correction tests: 34 passed
- existing HM CFO-estimator regression: 30 passed before the final
  validation refinement; the production estimator remained untouched
- full suite: 601 passed, 0 failed, 0 skipped

### Scope limitation

This is constant known/supplied CFO correction only. It is not phase
recovery, Costas/PLL carrier tracking, timing recovery, AMR, filtering, or
resampling.

### Next Module 11 milestone

Carrier phase recovery/correction.

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