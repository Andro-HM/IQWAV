"""Stage-B2 diagnostic: canonical M-th-power search range on the SAME 200 records.

Development-data diagnostic only. These records must not be reused later as
fresh held-out evidence.

Unchanged from Stage A/B except the search interval:
- z = globally_scaled_samples ** M
- 8x zero-padded FFT global peak initialization
- Brent local refinement of the same coherent objective around that peak
- no deadband, no per-record tuning
- search is now the canonical residual range, not the arbitrary +/-50 Hz cap:
    BPSK [-Fs/4, Fs/4)
    QPSK [-Fs/8, Fs/8)
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
BIT_SEEDS = tuple(range(10_000, 10_010))
NOISE_SEEDS = tuple(range(20_000, 20_010))
M_OF = {"bpsk": 2, "qpsk": 4}
PHASE_PERIOD = {"bpsk": np.pi, "qpsk": np.pi / 2.0}

# The eight Stage-B QPSK 11B rejects were all bit_seed=10000 with 11A
# leftover ~55-57 Hz. Worst-five noise seeds were recorded; all eight
# share that bit seed. We re-check every QPSK 11B reject vs Stage B.
STAGE_B_QPSK_REJECTS = 8
STAGE_B_RUNTIME = {
    "qpsk": (2.1365e-02, 3.4162e-02),
    "bpsk": (2.0536e-02, 3.0405e-02),
    "combined": (2.0953e-02, 3.3418e-02),
}


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
    return accepted, float(est.symmetry), float(est.phase_offset_rad), float(
        _wrap_phase(est.phase_offset_rad - PHI, PHASE_PERIOD[modulation])
    )


def classify_peak_error(est_hz: float, true_after_11a: float, fs: float, m: int) -> str | None:
    err = est_hz - true_after_11a
    if abs(err) <= 0.05:
        return None
    period = fs / m
    wrapped = wrap_canonical(err, fs, m)
    if abs(err) > 1.0 and abs(wrapped) < 0.05:
        return "alias"
    return "wrong-peak"


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def run_modulation(modulation: str) -> list[dict]:
    nbits = QPSK_NBITS if modulation == "qpsk" else BPSK_NBITS
    waveform_fn = qpsk_waveform if modulation == "qpsk" else bpsk_waveform
    m = M_OF[modulation]
    rows = []
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
            corrected = correct_frequency_offset(after_11a, FS, f_block)
            phys = physical_residual_hz(CFO_HZ, f_11a, f_block)
            accepted, symmetry, phase_rad, wrap_err = measure_phase(corrected, modulation)
            phased = correct_phase_offset(corrected, phase_rad)
            bit_e, n_bits, sym_e, n_sym = bit_symbol_errors(phased, bits, modulation, SPS)
            rows.append(
                {
                    "modulation": modulation,
                    "bit_seed": bit_seed,
                    "noise_seed": noise_seed,
                    "true_after_11a": true_after_11a,
                    "f_block": f_block,
                    "phys": phys,
                    "accepted": accepted,
                    "symmetry": symmetry,
                    "wrap_err": wrap_err,
                    "bit_errors": bit_e,
                    "n_bits": n_bits,
                    "symbol_errors": sym_e,
                    "n_symbols": n_sym,
                    "runtime_s": block_s,
                    "peak_event": classify_peak_error(f_block, true_after_11a, FS, m),
                    "worsened": abs(phys) > abs(true_after_11a),
                }
            )
    return rows


def report(label: str, rows: list[dict]) -> None:
    phys = np.array([r["phys"] for r in rows], dtype=np.float64)
    abs_phys = np.abs(phys)
    n_rej = sum(0 if r["accepted"] else 1 for r in rows)
    bit_e = sum(r["bit_errors"] for r in rows)
    n_bits = sum(r["n_bits"] for r in rows)
    sym_e = sum(r["symbol_errors"] for r in rows)
    n_sym = sum(r["n_symbols"] for r in rows)
    n_worse = sum(1 for r in rows if r["worsened"])
    n_alias = sum(1 for r in rows if r["peak_event"] == "alias")
    n_wrong = sum(1 for r in rows if r["peak_event"] == "wrong-peak")
    rt = np.array([r["runtime_s"] for r in rows], dtype=np.float64)
    print(f"--- {label} n={len(rows)} ---")
    print(
        f"  |phys| median/p95/max Hz  {np.median(abs_phys):.6e} / "
        f"{pct(abs_phys, 95):.6e} / {np.max(abs_phys):.6e}"
    )
    print(f"  11B reject count          {n_rej} / {len(rows)}")
    print(f"  BER                       {bit_e} / {n_bits} = {bit_e / n_bits:.6e}")
    print(f"  SER                       {sym_e} / {n_sym} = {sym_e / n_sym:.6e}")
    print(f"  worsens |11A residual|    {n_worse} / {len(rows)}")
    print(f"  alias events              {n_alias}")
    print(f"  wrong-peak events         {n_wrong}")
    print(f"  runtime median/p95 s      {np.median(rt):.4e} / {pct(rt, 95):.4e}")
    print()


def main() -> None:
    print("Stage-B2 diagnostic (SAME 200 records as Stage B; now development data)")
    print("Search: BPSK [-Fs/4, Fs/4), QPSK [-Fs/8, Fs/8)")
    print("FFT pad 8x + local Brent; no deadband; no per-record tuning")
    print()

    all_rows = []
    by_mod = {}
    for modulation in ("qpsk", "bpsk"):
        rows = run_modulation(modulation)
        by_mod[modulation] = rows
        all_rows.extend(rows)
        report(modulation.upper(), rows)
        prev = STAGE_B_RUNTIME[modulation]
        now = np.array([r["runtime_s"] for r in rows])
        print(
            f"  runtime vs Stage B +/-50 Hz: "
            f"median {np.median(now):.4e} vs {prev[0]:.4e} s; "
            f"p95 {pct(now, 95):.4e} vs {prev[1]:.4e} s"
        )
        print()

    report("COMBINED", all_rows)
    prev = STAGE_B_RUNTIME["combined"]
    now = np.array([r["runtime_s"] for r in all_rows])
    print(
        f"  runtime vs Stage B +/-50 Hz: "
        f"median {np.median(now):.4e} vs {prev[0]:.4e} s; "
        f"p95 {pct(now, 95):.4e} vs {prev[1]:.4e} s"
    )
    print()

    qpsk = by_mod["qpsk"]
    qpsk_rej = [r for r in qpsk if not r["accepted"]]
    print(f"Stage-B QPSK 11B rejects were {STAGE_B_QPSK_REJECTS}/100.")
    print(f"Stage-B2 QPSK 11B rejects: {len(qpsk_rej)}/100.")
    if not qpsk_rej:
        print("All eight previous whole-block QPSK 11B failures disappeared.")
    else:
        print("Remaining QPSK 11B rejects:")
        for r in qpsk_rej:
            print(
                f"  bit={r['bit_seed']} noise={r['noise_seed']} "
                f"11A_left={r['true_after_11a']:+.4f} Hz "
                f"est={r['f_block']:+.4f} Hz phys={r['phys']:+.4e} Hz "
                f"sym={r['symmetry']:.4f} event={r['peak_event']}"
            )

    events = [r for r in all_rows if r["peak_event"] is not None or not r["accepted"] or r["worsened"]]
    if events:
        print("Diagnostic events (reject, worsen, alias, or wrong-peak):")
        for r in events:
            print(
                f"  {r['modulation']} bit={r['bit_seed']} noise={r['noise_seed']} "
                f"11A_left={r['true_after_11a']:+.6f} est={r['f_block']:+.6f} "
                f"phys={r['phys']:+.6e} 11B={'accept' if r['accepted'] else 'REJECT'} "
                f"event={r['peak_event']} worsen={r['worsened']}"
            )
    else:
        print("No reject / worsen / alias / wrong-peak events on the 200 records.")


if __name__ == "__main__":
    main()
