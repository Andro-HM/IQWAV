"""Stage-D near-zero safety experiment for the frozen whole-block estimator.

Does not use 11A. Exact known residual CFOs are injected directly.
Estimator is frozen exactly as Stage C / B2 (canonical M-th-power range,
8x FFT global peak, local Brent, no deadband, no tuning).

Fresh seeds, disjoint from Stages A/B/C.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import estimate_phase_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform
from iqwav.synchronization import correct_frequency_offset, correct_phase_offset

FS = 80_000.0
SPS = 8
PHI = 0.8
SNR_DB = 20.0
QPSK_NBITS = 8192
BPSK_NBITS = 4096
FFT_PAD = 8
REFINE_BINS = 8
TARGET_HZ = 0.005
RESIDUALS_HZ = (
    0.0,
    0.001,
    -0.001,
    0.01,
    -0.01,
    0.1,
    -0.1,
    1.0,
    -1.0,
    4.433,
    -4.433,
    5.0,
    -5.0,
)
BIT_SEEDS = (50_000, 50_001, 50_002)
NOISE_SEEDS = tuple(range(60_000, 60_020))
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


def estimate_wholeblock_residual(samples: np.ndarray, fs: float, modulation: str) -> float:
    m = M_OF[modulation]
    search_lo, search_hi = canonical_limits(fs, m)
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
    df_bin_hz = (fs / nfft) / m
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
    return wrap_canonical(float(result.x), fs, m)


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
    wrap_err = _wrap_phase(est.phase_offset_rad - PHI, PHASE_PERIOD[modulation])
    return accepted, float(est.phase_offset_rad), float(wrap_err)


def classify_peak_error(est_hz: float, true_hz: float, fs: float, m: int) -> str | None:
    err = est_hz - true_hz
    if abs(err) <= 0.05:
        return None
    wrapped = wrap_canonical(err, fs, m)
    if abs(err) > 1.0 and abs(wrapped) < 0.05:
        return "alias"
    return "wrong-peak"


def evaluate_after(samples, extra_hz, true_residual, true_bits, modulation):
    corrected = correct_frequency_offset(samples, FS, extra_hz)
    phys = true_residual - extra_hz
    accepted, phase_rad, wrap_err = measure_phase(corrected, modulation)
    phased = correct_phase_offset(corrected, phase_rad)
    bit_e, n_bits, sym_e, n_sym = bit_symbol_errors(phased, true_bits, modulation, SPS)
    return {
        "phys": phys,
        "accepted": accepted,
        "wrap_err": wrap_err,
        "bit_errors": bit_e,
        "n_bits": n_bits,
        "symbol_errors": sym_e,
        "n_symbols": n_sym,
    }


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize_group(label: str, method: str, recs: list[dict], true_hz: float) -> None:
    phys = np.array([r[method]["phys"] for r in recs], dtype=np.float64)
    abs_phys = np.abs(phys)
    wrap = np.abs(np.array([r[method]["wrap_err"] for r in recs], dtype=np.float64))
    n_rej = sum(0 if r[method]["accepted"] else 1 for r in recs)
    bit_e = sum(r[method]["bit_errors"] for r in recs)
    n_bits = sum(r[method]["n_bits"] for r in recs)
    sym_e = sum(r[method]["symbol_errors"] for r in recs)
    n_sym = sum(r[method]["n_symbols"] for r in recs)
    print(f"  {method:8s}  n={len(recs)}")
    if method == "B":
        est = np.array([r["est"] for r in recs], dtype=np.float64)
        bias = est - true_hz
        print(f"    signed estimation bias Hz   {np.mean(bias):+.6e}")
        print(f"    estimation RMSE Hz          {np.sqrt(np.mean(bias**2)):.6e}")
        invented = np.abs(est)
        if true_hz == 0.0:
            print(
                f"    |invented CFO| med/p95/max  {np.median(invented):.6e} / "
                f"{pct(invented, 95):.6e} / {np.max(invented):.6e}"
            )
        n_alias = sum(1 for r in recs if r["peak_event"] == "alias")
        n_wrong = sum(1 for r in recs if r["peak_event"] == "wrong-peak")
        a_abs = np.abs(np.array([r["A"]["phys"] for r in recs]))
        b_abs = abs_phys
        n_reduce = int(np.count_nonzero(b_abs < a_abs))
        n_increase = int(np.count_nonzero(b_abs > a_abs))
        n_equal = len(recs) - n_reduce - n_increase
        print(f"    fraction reduces |res|      {n_reduce}/{len(recs)}")
        print(f"    fraction increases |res|    {n_increase}/{len(recs)}")
        if n_equal:
            print(f"    fraction unchanged |res|    {n_equal}/{len(recs)}")
        print(f"    alias / wrong-peak          {n_alias} / {n_wrong}")
        n_over = int(np.count_nonzero(abs_phys > TARGET_HZ))
        print(f"    |phys| > 0.005 Hz           {n_over}/{len(recs)}")
    print(
        f"    |final phys| med/p95/max Hz {np.median(abs_phys):.6e} / "
        f"{pct(abs_phys, 95):.6e} / {np.max(abs_phys):.6e}"
    )
    print(f"    11B reject                  {n_rej}/{len(recs)}")
    print(
        f"    |wrap err| med/p95/max rad  {np.median(wrap):.6e} / "
        f"{pct(wrap, 95):.6e} / {np.max(wrap):.6e}"
    )
    print(f"    BER                         {bit_e}/{n_bits} = {bit_e / n_bits:.6e}")
    print(f"    SER                         {sym_e}/{n_sym} = {sym_e / n_sym:.6e}")


def run_modulation(modulation: str) -> dict[float, list[dict]]:
    nbits = QPSK_NBITS if modulation == "qpsk" else BPSK_NBITS
    waveform_fn = qpsk_waveform if modulation == "qpsk" else bpsk_waveform
    m = M_OF[modulation]
    by_res: dict[float, list[dict]] = {hz: [] for hz in RESIDUALS_HZ}
    for bit_seed in BIT_SEEDS:
        bits = np.random.default_rng(bit_seed).integers(0, 2, nbits)
        clean = waveform_fn(bits, SPS)
        phased = apply_phase_offset(clean, PHI)
        for true_hz in RESIDUALS_HZ:
            impaired = apply_frequency_offset(phased, FS, true_hz)
            for noise_seed in NOISE_SEEDS:
                noisy = add_awgn(impaired, SNR_DB, np.random.default_rng(noise_seed))
                est = estimate_wholeblock_residual(noisy, FS, modulation)
                row_a = evaluate_after(noisy, 0.0, true_hz, bits, modulation)
                row_b = evaluate_after(noisy, est, true_hz, bits, modulation)
                row_c = evaluate_after(noisy, true_hz, true_hz, bits, modulation)
                by_res[true_hz].append(
                    {
                        "est": est,
                        "A": row_a,
                        "B": row_b,
                        "C": row_c,
                        "peak_event": classify_peak_error(est, true_hz, FS, m),
                    }
                )
    return by_res


def main() -> None:
    print("Stage-D near-zero safety (no 11A; exact injected residuals)")
    print(
        f"bit seeds {BIT_SEEDS}  noise seeds {NOISE_SEEDS[0]}-{NOISE_SEEDS[-1]}  "
        f"n={len(BIT_SEEDS)*len(NOISE_SEEDS)} per residual/modulation"
    )
    print(f"Fs={FS:.0f} SPS={SPS} N=32768 phi={PHI} SNR={SNR_DB:.0f} dB")
    print("Frozen whole-block: 8x FFT + Brent, canonical M-th-power range, no deadband")
    print(f"Research target |final phys| <= {TARGET_HZ} Hz (not a product spec)")
    print("At true 0 Hz, any noisy nonzero estimate increases |residual|; that alone is not failure.")
    print()

    t0 = time.perf_counter()
    for modulation in ("qpsk", "bpsk"):
        print(f"========== {modulation.upper()} ==========")
        by_res = run_modulation(modulation)
        all_recs = [r for recs in by_res.values() for r in recs]
        print(f"--- all residuals n={len(all_recs)} ---")
        summarize_group(modulation, "A", all_recs, true_hz=float("nan"))
        est = np.array([r["est"] for r in all_recs])
        truths = np.array(
            [hz for hz, recs in by_res.items() for _ in recs], dtype=np.float64
        )
        bias = est - truths
        phys_b = np.abs(np.array([r["B"]["phys"] for r in all_recs]))
        a_abs = np.abs(np.array([r["A"]["phys"] for r in all_recs]))
        n_reduce = int(np.count_nonzero(phys_b < a_abs))
        n_increase = int(np.count_nonzero(phys_b > a_abs))
        n_alias = sum(1 for r in all_recs if r["peak_event"] == "alias")
        n_wrong = sum(1 for r in all_recs if r["peak_event"] == "wrong-peak")
        n_over = int(np.count_nonzero(phys_b > TARGET_HZ))
        n_rej = sum(0 if r["B"]["accepted"] else 1 for r in all_recs)
        bit_e = sum(r["B"]["bit_errors"] for r in all_recs)
        n_bits = sum(r["B"]["n_bits"] for r in all_recs)
        print("  B        pooled vs true residual (mixed magnitudes)")
        print(f"    signed estimation bias Hz   {np.mean(bias):+.6e}")
        print(f"    estimation RMSE Hz          {np.sqrt(np.mean(bias**2)):.6e}")
        print(
            f"    |final phys| med/p95/max Hz {np.median(phys_b):.6e} / "
            f"{pct(phys_b, 95):.6e} / {np.max(phys_b):.6e}"
        )
        print(f"    fraction reduces |res|      {n_reduce}/{len(all_recs)}")
        print(f"    fraction increases |res|    {n_increase}/{len(all_recs)}")
        print(f"    alias / wrong-peak          {n_alias} / {n_wrong}")
        print(f"    |phys| > 0.005 Hz           {n_over}/{len(all_recs)}")
        print(f"    11B reject                  {n_rej}/{len(all_recs)}")
        print(f"    BER                         {bit_e}/{n_bits} = {bit_e / n_bits:.6e}")
        summarize_group(modulation, "C", all_recs, true_hz=float("nan"))
        print()
        for true_hz in RESIDUALS_HZ:
            recs = by_res[true_hz]
            print(f"--- true residual {true_hz:+.3f} Hz  n={len(recs)} ---")
            summarize_group(modulation, "A", recs, true_hz)
            summarize_group(modulation, "B", recs, true_hz)
            summarize_group(modulation, "C", recs, true_hz)
            print()
    print(f"elapsed {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
