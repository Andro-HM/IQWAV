"""Stage-C fresh held-out 20 dB evaluation of the frozen canonical-range candidate.

Temporary experiment only. Does not modify production files, LOGS, or tests.
Estimator is frozen exactly as Stage B2:
- z = globally_scaled_samples ** M
- 8x zero-padded FFT global peak initialization
- Brent local refinement of the same coherent objective
- canonical residual search BPSK [-Fs/4, Fs/4), QPSK [-Fs/8, Fs/8)
- REFINE_BINS = 8, xatol = 1e-6
- no deadband, no per-record / per-seed tuning

Seeds are disjoint from Stage A (102; 1/2/3) and Stage B/B2 (10000-10009; 20000-20009).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import estimate_frequency_offset, estimate_phase_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset, correct_phase_offset

FS = 80_000.0
SPS = 8
PHI = 0.8
CFO_HZ = 1000.0
SNR_DB = 20.0
QPSK_NBITS = 8192
BPSK_NBITS = 4096
FFT_PAD = 8
REFINE_BINS = 8
BIT_SEEDS = tuple(range(30_000, 30_010))
NOISE_SEEDS = tuple(range(40_000, 40_010))
M_OF = {"bpsk": 2, "qpsk": 4}
PHASE_PERIOD = {"bpsk": np.pi, "qpsk": np.pi / 2.0}


def canonical_limits(fs: float, m: int) -> tuple[float, float]:
    half = fs / (2.0 * m)
    return -half, half


def wrap_canonical(value: float, fs: float, m: int) -> float:
    half = fs / (2.0 * m)
    period = 2.0 * half
    return (value + half) % period - half


def _wrap_phase(error: float, period: float) -> float:
    return (error + period / 2.0) % period - period / 2.0


def physical_residual_hz(true_cfo: float, *corrections: float) -> float:
    leftover = true_cfo
    for value in corrections:
        leftover -= value
    return float(leftover)


def estimate_wholeblock_residual(samples: np.ndarray, fs: float, modulation: str) -> tuple[float, float]:
    """Frozen Stage-B2 canonical-range whole-block estimator."""
    m = M_OF[modulation]
    search_lo, search_hi = canonical_limits(fs, m)
    t0 = time.perf_counter()
    samples = np.asarray(samples)
    scale = float(np.mean(np.abs(samples)))
    if scale == 0.0:
        raise ValueError("zero-energy samples")
    z = (samples / scale) ** m
    n = np.arange(z.size, dtype=np.float64)
    nfft = FFT_PAD * z.size
    spectrum = np.fft.fft(z, n=nfft)
    freqs_z = np.fft.fftfreq(nfft, d=1.0 / fs)
    residual_axis = freqs_z / m
    mask = (residual_axis >= search_lo) & (residual_axis < search_hi)
    if not np.any(mask):
        raise ValueError("no FFT bins inside residual search range")
    peak_local = int(np.argmax(np.abs(spectrum[mask]) ** 2))
    df0 = float(residual_axis[mask][peak_local])

    z_bin_hz = fs / nfft
    df_bin_hz = z_bin_hz / m
    half_width = REFINE_BINS * df_bin_hz
    lo = max(search_lo, df0 - half_width)
    hi = min(search_hi, df0 + half_width)
    if hi <= lo:
        hi = min(search_hi, lo + df_bin_hz)

    def coherent_power(df: float) -> float:
        phasor = np.exp(-1j * 2.0 * np.pi * (m * df) * n / fs)
        return float(np.abs(np.dot(z, phasor)) ** 2)

    result = minimize_scalar(
        lambda df: -coherent_power(df),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-6},
    )
    df_hat = wrap_canonical(float(result.x), fs, m)
    elapsed = time.perf_counter() - t0
    return df_hat, elapsed


def bit_symbol_errors(samples, true_bits, modulation, sps):
    n_bits = int(true_bits.size)
    if modulation == "qpsk":
        n_sym = n_bits // 2
        true_pairs = true_bits.reshape(n_sym, 2)
        best_bit = n_bits
        best_sym = n_sym
        for k in range(4):
            bits = qpsk_demodulate(samples * (1j**k), sps)
            bit_err = int(np.count_nonzero(bits != true_bits))
            sym_err = int(np.count_nonzero(np.any(bits.reshape(n_sym, 2) != true_pairs, axis=1)))
            if bit_err < best_bit:
                best_bit = bit_err
                best_sym = sym_err
        return best_bit, n_bits, best_sym, n_sym
    best_bit = n_bits
    for sign in (1.0, -1.0):
        bits = bpsk_demodulate(samples * sign, sps)
        bit_err = int(np.count_nonzero(bits != true_bits))
        if bit_err < best_bit:
            best_bit = bit_err
    return best_bit, n_bits, best_bit, n_bits


def measure_phase(samples, modulation):
    try:
        est = estimate_phase_offset(samples, modulation)
        accepted = True
    except ValueError:
        est = estimate_phase_offset(samples, modulation, min_symmetry=0.0)
        accepted = False
    return {
        "accepted": accepted,
        "symmetry": float(est.symmetry),
        "phase_rad": float(est.phase_offset_rad),
        "wrap_err_rad": float(
            _wrap_phase(est.phase_offset_rad - PHI, PHASE_PERIOD[modulation])
        ),
    }


def classify_peak_error(est_hz: float, true_after_11a: float, fs: float, m: int) -> str | None:
    err = est_hz - true_after_11a
    if abs(err) <= 0.05:
        return None
    wrapped = wrap_canonical(err, fs, m)
    if abs(err) > 1.0 and abs(wrapped) < 0.05:
        return "alias"
    return "wrong-peak"


def evaluate(after_11a, extra_hz, true_bits, modulation, f_11a):
    corrected = correct_frequency_offset(after_11a, FS, extra_hz)
    phys = physical_residual_hz(CFO_HZ, f_11a, extra_hz)
    phase = measure_phase(corrected, modulation)
    phased = correct_phase_offset(corrected, phase["phase_rad"])
    bit_e, n_bits, sym_e, n_sym = bit_symbol_errors(phased, true_bits, modulation, SPS)
    return {
        "extra_hz": float(extra_hz),
        "physical_residual_hz": phys,
        **phase,
        "bit_errors": bit_e,
        "n_bits": n_bits,
        "symbol_errors": sym_e,
        "n_symbols": n_sym,
    }


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize(name: str, rows: list[dict], runtimes: np.ndarray | None = None) -> None:
    phys = np.array([r["physical_residual_hz"] for r in rows], dtype=np.float64)
    abs_phys = np.abs(phys)
    wrap = np.abs(np.array([r["wrap_err_rad"] for r in rows], dtype=np.float64))
    n_reject = sum(0 if r["accepted"] else 1 for r in rows)
    bit_e = sum(r["bit_errors"] for r in rows)
    n_bits = sum(r["n_bits"] for r in rows)
    sym_e = sum(r["symbol_errors"] for r in rows)
    n_sym = sum(r["n_symbols"] for r in rows)
    print(f"--- {name}  n={len(rows)} ---")
    print(f"  signed bias (phys Hz)     {np.mean(phys):+.6e}")
    print(f"  RMSE (phys Hz)            {np.sqrt(np.mean(phys**2)):.6e}")
    print(
        f"  |phys| median/p95/max Hz  {np.median(abs_phys):.6e} / "
        f"{pct(abs_phys, 95):.6e} / {np.max(abs_phys):.6e}"
    )
    print(f"  11B reject count          {n_reject} / {len(rows)}")
    print(
        f"  |wrap phase err| med/p95/max rad  "
        f"{np.median(wrap):.6e} / {pct(wrap, 95):.6e} / {np.max(wrap):.6e}"
    )
    print(f"  BER total                 {bit_e} / {n_bits}  = {bit_e / n_bits:.6e}")
    print(f"  SER total                 {sym_e} / {n_sym}  = {sym_e / n_sym:.6e}")
    if runtimes is not None and runtimes.size:
        print(
            f"  runtime median/p95 s      {np.median(runtimes):.4e} / "
            f"{pct(runtimes, 95):.4e}"
        )
    print()


def worsening_stats(after_11a: np.ndarray, after_ref: np.ndarray, label: str) -> None:
    a = np.abs(after_11a)
    r = np.abs(after_ref)
    worse = r > a
    n = int(np.count_nonzero(worse))
    print(f"--- {label} worsens |11A residual| ---")
    print(f"  count                     {n} / {after_11a.size}")
    if n:
        delta = r[worse] - a[worse]
        print(
            f"  |after| - |11A| median/p95/max Hz  "
            f"{np.median(delta):.6e} / {pct(delta, 95):.6e} / {np.max(delta):.6e}"
        )
    print()


def run_modulation(modulation: str):
    nbits = QPSK_NBITS if modulation == "qpsk" else BPSK_NBITS
    waveform_fn = qpsk_waveform if modulation == "qpsk" else bpsk_waveform
    m = M_OF[modulation]
    rows = {"A": [], "C": [], "D": []}
    times = []
    details_c = []
    n_alias = 0
    n_wrong = 0
    for bit_seed in BIT_SEEDS:
        bits = np.random.default_rng(bit_seed).integers(0, 2, nbits)
        clean = waveform_fn(bits, SPS)
        impaired = apply_frequency_offset(apply_phase_offset(clean, PHI), FS, CFO_HZ)
        for noise_seed in NOISE_SEEDS:
            noisy = add_awgn(impaired, SNR_DB, np.random.default_rng(noise_seed))
            coarse = estimate_frequency_offset(noisy, FS)
            f_11a = float(coarse.frequency_offset_hz)
            after_11a = correct_frequency_offset(noisy, FS, f_11a)
            true_after_11a = physical_residual_hz(CFO_HZ, f_11a)
            f_block, block_s = estimate_wholeblock_residual(after_11a, FS, modulation)
            event = classify_peak_error(f_block, true_after_11a, FS, m)
            if event == "alias":
                n_alias += 1
            elif event == "wrong-peak":
                n_wrong += 1
            row_a = evaluate(after_11a, 0.0, bits, modulation, f_11a)
            row_c = evaluate(after_11a, f_block, bits, modulation, f_11a)
            row_d = evaluate(after_11a, true_after_11a, bits, modulation, f_11a)
            rows["A"].append(row_a)
            rows["C"].append(row_c)
            rows["D"].append(row_d)
            times.append(block_s)
            details_c.append(
                {
                    "modulation": modulation,
                    "bit_seed": bit_seed,
                    "noise_seed": noise_seed,
                    "f_11a": f_11a,
                    "true_after_11a": true_after_11a,
                    "f_block": f_block,
                    "runtime_s": block_s,
                    "peak_event": event,
                    **row_c,
                }
            )
    return rows, np.array(times), details_c, n_alias, n_wrong


def main() -> None:
    print("Stage-C fresh held-out 20 dB evaluation")
    print(
        f"bit seeds {BIT_SEEDS[0]}-{BIT_SEEDS[-1]}  "
        f"noise seeds {NOISE_SEEDS[0]}-{NOISE_SEEDS[-1]}"
    )
    print(
        f"Fs={FS:.0f} SPS={SPS} phi={PHI} CFO={CFO_HZ:.0f} Hz SNR={SNR_DB:.0f} dB"
    )
    print("Frozen whole-block: FFT pad 8x, Brent, canonical M-th-power range")
    print("BER/SER: known-timing after 11B phase correction, best rotation")
    print("Primary frequency metric: injected physical residual")
    print()

    all_details = []
    combined = {"A": [], "C": [], "D": []}
    combined_t = []
    alias_total = 0
    wrong_total = 0

    for modulation in ("qpsk", "bpsk"):
        print(f"========== {modulation.upper()} (100 records) ==========")
        rows, times, details, n_alias, n_wrong = run_modulation(modulation)
        all_details.extend(details)
        for key in combined:
            combined[key].extend(rows[key])
        combined_t.extend(times.tolist())
        alias_total += n_alias
        wrong_total += n_wrong
        summarize("A 11A only", rows["A"])
        summarize("C whole-block", rows["C"], times)
        print(f"  alias events              {n_alias}")
        print(f"  wrong-peak events         {n_wrong}")
        print()
        summarize("D oracle", rows["D"])
        a = np.array([r["physical_residual_hz"] for r in rows["A"]])
        c = np.array([r["physical_residual_hz"] for r in rows["C"]])
        worsening_stats(a, c, f"{modulation} C whole-block")

    print("========== COMBINED 200 records ==========")
    summarize("A 11A only", combined["A"])
    summarize("C whole-block", combined["C"], np.array(combined_t))
    print(f"  alias events              {alias_total}")
    print(f"  wrong-peak events         {wrong_total}")
    print()
    summarize("D oracle", combined["D"])
    a = np.array([r["physical_residual_hz"] for r in combined["A"]])
    c = np.array([r["physical_residual_hz"] for r in combined["C"]])
    worsening_stats(a, c, "combined C whole-block")

    worst = sorted(all_details, key=lambda r: abs(r["physical_residual_hz"]), reverse=True)[:5]
    print("========== WORST 5 whole-block records by |physical residual| ==========")
    for i, r in enumerate(worst, 1):
        status = "accept" if r["accepted"] else "REJECT"
        print(
            f"{i}. {r['modulation']} bit_seed={r['bit_seed']} noise_seed={r['noise_seed']}\n"
            f"   11A={r['f_11a']:+.6f} Hz  true leftover after 11A={r['true_after_11a']:+.6f} Hz\n"
            f"   whole-block est={r['f_block']:+.6f} Hz  "
            f"phys leftover={r['physical_residual_hz']:+.6e} Hz  "
            f"event={r['peak_event']}\n"
            f"   11B {status}  wrap_err={r['wrap_err_rad']:+.6e} rad  "
            f"bit_err={r['bit_errors']}/{r['n_bits']}  "
            f"sym_err={r['symbol_errors']}/{r['n_symbols']}  "
            f"runtime={r['runtime_s']:.4e} s"
        )


if __name__ == "__main__":
    main()
