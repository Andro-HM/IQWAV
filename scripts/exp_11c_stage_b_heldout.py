"""Stage-B held-out 20 dB evaluation of the frozen Stage-A whole-block estimator.

Temporary experiment only. Does not modify production files, LOGS, tests,
or the Stage-A algorithm. Failed lag-1 11C remains untouched.

Frozen candidate (identical to Stage A):
- z = globally_scaled_samples ** M
- 8x zero-padded FFT initialization
- Brent refinement of the same coherent objective
- residual search [-50, +50] Hz
- REFINE_BINS = 8, xatol = 1e-6
- no deadband, no per-record / per-seed tuning

M is 4 for QPSK and 2 for BPSK (constellation power only).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import (
    estimate_frequency_offset,
    estimate_phase_offset,
    estimate_residual_frequency_offset,
)
from iqwav.modulation import bpsk_waveform, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset, correct_phase_offset

FS = 80_000.0
SPS = 8
PHI = 0.8
CFO_HZ = 1000.0
SNR_DB = 20.0
QPSK_NBITS = 8192  # 32768 samples at SPS 8
BPSK_NBITS = 4096  # same 32768-sample convention as Stage A QPSK
SEARCH_HZ = 50.0
FFT_PAD = 8
REFINE_BINS = 8

# Held-out seeds: disjoint from Stage A (bit seed 102, noise seeds 1/2/3)
# and from the original 11C characterization seeds 101/201/202.
BIT_SEEDS = tuple(range(10_000, 10_010))
NOISE_SEEDS = tuple(range(20_000, 20_010))

M_OF = {"bpsk": 2, "qpsk": 4}
PHASE_PERIOD = {"bpsk": np.pi, "qpsk": np.pi / 2.0}


def _wrap_phase(error: float, period: float) -> float:
    return (error + period / 2.0) % period - period / 2.0


def physical_residual_hz(true_cfo: float, *corrections: float) -> float:
    leftover = true_cfo
    for value in corrections:
        leftover -= value
    return float(leftover)


def independent_residual_hz(corrected: np.ndarray, clean: np.ndarray, fs: float) -> float:
    nonzero = np.abs(clean) > 0.0
    n = np.nonzero(nonzero)[0].astype(np.float64)
    phase = np.unwrap(np.angle(corrected[nonzero] / clean[nonzero]))
    slope = np.polyfit(n, phase, 1)[0]
    return float(slope * fs / (2.0 * np.pi))


def estimate_wholeblock_residual(samples: np.ndarray, fs: float, modulation: str) -> tuple[float, float]:
    """Frozen Stage-A estimator. Only M depends on modulation."""
    m = M_OF[modulation]
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
    mask = (residual_axis >= -SEARCH_HZ) & (residual_axis <= SEARCH_HZ)
    if not np.any(mask):
        raise ValueError("no FFT bins inside residual search range")
    peak_local = int(np.argmax(np.abs(spectrum[mask]) ** 2))
    df0 = float(residual_axis[mask][peak_local])

    z_bin_hz = fs / nfft
    df_bin_hz = z_bin_hz / m
    half_width = REFINE_BINS * df_bin_hz
    lo = max(-SEARCH_HZ, df0 - half_width)
    hi = min(SEARCH_HZ, df0 + half_width)
    if hi <= lo:
        hi = min(SEARCH_HZ, lo + df_bin_hz)

    def coherent_power(df: float) -> float:
        phasor = np.exp(-1j * 2.0 * np.pi * (m * df) * n / fs)
        return float(np.abs(np.dot(z, phasor)) ** 2)

    result = minimize_scalar(
        lambda df: -coherent_power(df),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-6},
    )
    df_hat = float(np.clip(result.x, -SEARCH_HZ, SEARCH_HZ))
    elapsed = time.perf_counter() - t0
    return df_hat, elapsed


def bit_symbol_errors(
    samples: np.ndarray, true_bits: np.ndarray, modulation: str, sps: int
) -> tuple[int, int, int, int]:
    """Return (bit_errors, n_bits, symbol_errors, n_symbols) after rotational search."""
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


def measure_phase(samples: np.ndarray, modulation: str) -> dict:
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


def evaluate(
    after_11a: np.ndarray,
    extra_hz: float,
    clean: np.ndarray,
    true_bits: np.ndarray,
    modulation: str,
    f_11a: float,
) -> dict:
    corrected = correct_frequency_offset(after_11a, FS, extra_hz)
    phys = physical_residual_hz(CFO_HZ, f_11a, extra_hz)
    phase = measure_phase(corrected, modulation)
    phased = correct_phase_offset(corrected, phase["phase_rad"])
    bit_e, n_bits, sym_e, n_sym = bit_symbol_errors(phased, true_bits, modulation, SPS)
    return {
        "extra_hz": float(extra_hz),
        "physical_residual_hz": phys,
        "independent_residual_hz": independent_residual_hz(corrected, clean, FS),
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
    sym = np.array([r["symmetry"] for r in rows], dtype=np.float64)
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
        f"  symmetry min/p5/median/p95/max  "
        f"{np.min(sym):.4f} / {pct(sym, 5):.4f} / {np.median(sym):.4f} / "
        f"{pct(sym, 95):.4f} / {np.max(sym):.4f}"
    )
    print(
        f"  |wrap phase err| med/p95/max rad  "
        f"{np.median(wrap):.6e} / {pct(wrap, 95):.6e} / {np.max(wrap):.6e}"
    )
    print(
        f"  BER total                 {bit_e} / {n_bits}  = {bit_e / n_bits:.6e}"
    )
    print(
        f"  SER total                 {sym_e} / {n_sym}  = {sym_e / n_sym:.6e}"
    )
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
        print(
            f"  those |after| median/max Hz        "
            f"{np.median(r[worse]):.6e} / {np.max(r[worse]):.6e}"
        )
    else:
        print("  no worsenings")
    print()


def run_modulation(modulation: str) -> tuple[dict[str, list[dict]], dict[str, list[float]], list[dict]]:
    nbits = QPSK_NBITS if modulation == "qpsk" else BPSK_NBITS
    waveform_fn = qpsk_waveform if modulation == "qpsk" else bpsk_waveform
    rows = {"A": [], "B": [], "C": [], "D": []}
    times = {"B": [], "C": []}
    details_c = []

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

            t1 = time.perf_counter()
            try:
                lag1 = estimate_residual_frequency_offset(after_11a, FS, modulation)
                f_lag1 = float(lag1.residual_frequency_hz)
                lag1_rejected = False
            except ValueError:
                f_lag1 = 0.0
                lag1_rejected = True
            lag1_s = time.perf_counter() - t1

            f_block, block_s = estimate_wholeblock_residual(after_11a, FS, modulation)

            row_a = evaluate(after_11a, 0.0, clean, bits, modulation, f_11a)
            row_b = evaluate(after_11a, f_lag1, clean, bits, modulation, f_11a)
            row_c = evaluate(after_11a, f_block, clean, bits, modulation, f_11a)
            row_d = evaluate(after_11a, true_after_11a, clean, bits, modulation, f_11a)
            row_b["lag1_rejected"] = lag1_rejected

            rows["A"].append(row_a)
            rows["B"].append(row_b)
            rows["C"].append(row_c)
            rows["D"].append(row_d)
            times["B"].append(lag1_s)
            times["C"].append(block_s)
            details_c.append(
                {
                    "modulation": modulation,
                    "bit_seed": bit_seed,
                    "noise_seed": noise_seed,
                    "f_11a": f_11a,
                    "true_after_11a": true_after_11a,
                    "f_block": f_block,
                    "runtime_s": block_s,
                    **row_c,
                }
            )
    return rows, times, details_c


def main() -> None:
    print("Stage-B held-out 20 dB evaluation")
    print(
        f"bit seeds {BIT_SEEDS[0]}-{BIT_SEEDS[-1]}  "
        f"noise seeds {NOISE_SEEDS[0]}-{NOISE_SEEDS[-1]}"
    )
    print(
        f"Fs={FS:.0f} SPS={SPS} phi={PHI} CFO={CFO_HZ:.0f} Hz SNR={SNR_DB:.0f} dB"
    )
    print("Frozen whole-block: FFT pad 8x, Brent, search [-50,+50] Hz, no deadband")
    print("BER/SER are known-timing after 11B phase correction, best rotation.")
    print("Primary frequency metric: injected physical residual.")
    print()

    all_details = []
    combined = {"A": [], "B": [], "C": [], "D": []}
    combined_t = {"B": [], "C": []}

    for modulation in ("qpsk", "bpsk"):
        print(f"========== {modulation.upper()} (100 records) ==========")
        rows, times, details = run_modulation(modulation)
        all_details.extend(details)
        for key in combined:
            combined[key].extend(rows[key])
        combined_t["B"].extend(times["B"])
        combined_t["C"].extend(times["C"])

        summarize("A 11A only", rows["A"])
        summarize("B lag-1 11C", rows["B"], np.array(times["B"]))
        n_lag1_rej = sum(1 for r in rows["B"] if r.get("lag1_rejected"))
        print(f"  (lag-1 11C estimator rejects: {n_lag1_rej} / {len(rows['B'])})")
        print()
        summarize("C whole-block", rows["C"], np.array(times["C"]))
        summarize("D oracle", rows["D"])

        a_abs = np.array([r["physical_residual_hz"] for r in rows["A"]])
        b_abs = np.array([r["physical_residual_hz"] for r in rows["B"]])
        c_abs = np.array([r["physical_residual_hz"] for r in rows["C"]])
        worsening_stats(a_abs, b_abs, f"{modulation} B lag-1")
        worsening_stats(a_abs, c_abs, f"{modulation} C whole-block")

    print("========== COMBINED 200 records ==========")
    summarize("A 11A only", combined["A"])
    summarize("B lag-1 11C", combined["B"], np.array(combined_t["B"]))
    n_lag1_rej = sum(1 for r in combined["B"] if r.get("lag1_rejected"))
    print(f"  (lag-1 11C estimator rejects: {n_lag1_rej} / {len(combined['B'])})")
    print()
    summarize("C whole-block", combined["C"], np.array(combined_t["C"]))
    summarize("D oracle", combined["D"])
    a_abs = np.array([r["physical_residual_hz"] for r in combined["A"]])
    b_abs = np.array([r["physical_residual_hz"] for r in combined["B"]])
    c_abs = np.array([r["physical_residual_hz"] for r in combined["C"]])
    worsening_stats(a_abs, b_abs, "combined B lag-1")
    worsening_stats(a_abs, c_abs, "combined C whole-block")

    worst = sorted(all_details, key=lambda r: abs(r["physical_residual_hz"]), reverse=True)[:5]
    print("========== WORST 5 whole-block records by |physical residual| ==========")
    for i, r in enumerate(worst, 1):
        status = "accept" if r["accepted"] else "REJECT"
        print(
            f"{i}. {r['modulation']} bit_seed={r['bit_seed']} noise_seed={r['noise_seed']}\n"
            f"   11A={r['f_11a']:+.6f} Hz  true leftover after 11A={r['true_after_11a']:+.6f} Hz\n"
            f"   whole-block est={r['f_block']:+.6f} Hz  "
            f"phys leftover={r['physical_residual_hz']:+.6e} Hz  "
            f"indep slope={r['independent_residual_hz']:+.6e} Hz\n"
            f"   11B {status}  sym={r['symmetry']:.4f}  "
            f"wrap_err={r['wrap_err_rad']:+.6e} rad  "
            f"bit_err={r['bit_errors']}/{r['n_bits']}  "
            f"sym_err={r['symbol_errors']}/{r['n_symbols']}  "
            f"runtime={r['runtime_s']:.4e} s"
        )


if __name__ == "__main__":
    main()
