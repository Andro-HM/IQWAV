"""Stage-A falsification: whole-block M-th-power tone-frequency residual CFO.

Temporary experiment only. Does not modify production code, exports, LOGS,
or tests. Does not replace the failed lag-1 11C baseline.

Candidate (QPSK, M=4): z = globally_scaled_samples ** 4; maximize whole-block
coherent tone power. 8x zero-padded FFT initializes the peak; Brent then
refines the actual coherent objective around that peak. Residual search is
restricted to [-50, +50] Hz. No deadband, no per-seed tuning.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar

from iqwav.demod import qpsk_demodulate
from iqwav.dsp import add_awgn, apply_frequency_offset, apply_phase_offset
from iqwav.estimation import (
    estimate_frequency_offset,
    estimate_phase_offset,
    estimate_residual_frequency_offset,
)
from iqwav.modulation import qpsk_waveform
from iqwav.synchronization import correct_frequency_offset, correct_phase_offset

FS = 80_000.0
SPS = 8
NBITS = 8192
BIT_SEED = 102
PHI = 0.8
CFO_HZ = 1000.0
SNR_DB = 20.0
NOISE_SEEDS = (1, 2, 3)
M = 4
SEARCH_HZ = 50.0
FFT_PAD = 8
# Local continuous refinement window around the FFT peak, in padded-FFT
# bins of the powered-tone frequency, mapped to residual Hz. Fixed; not
# tuned per seed.
REFINE_BINS = 8
PHASE_PERIOD = np.pi / 2.0


def _wrap_phase(error: float, period: float) -> float:
    return (error + period / 2.0) % period - period / 2.0


def independent_residual_hz(corrected: np.ndarray, clean: np.ndarray, fs: float) -> float:
    nonzero = np.abs(clean) > 0.0
    n = np.nonzero(nonzero)[0].astype(np.float64)
    phase = np.unwrap(np.angle(corrected[nonzero] / clean[nonzero]))
    slope = np.polyfit(n, phase, 1)[0]
    return float(slope * fs / (2.0 * np.pi))


def physical_residual_hz(true_cfo: float, *corrections: float) -> float:
    leftover = true_cfo
    for value in corrections:
        leftover -= value
    return float(leftover)


def qpsk_errors(samples: np.ndarray, true_bits: np.ndarray, sps: int) -> tuple[float, float]:
    """Known-timing QPSK BER/SER after best of 4 rotational ambiguities."""
    n_sym = true_bits.size // 2
    true_pairs = true_bits.reshape(n_sym, 2)
    best_ber = 1.0
    best_ser = 1.0
    for k in range(4):
        bits = qpsk_demodulate(samples * (1j**k), sps)
        ber = float(np.mean(bits != true_bits))
        ser = float(np.mean(np.any(bits.reshape(n_sym, 2) != true_pairs, axis=1)))
        if ber < best_ber:
            best_ber = ber
            best_ser = ser
    return best_ber, best_ser


def measure_phase(samples: np.ndarray) -> dict:
    try:
        est = estimate_phase_offset(samples, "qpsk")
        accepted = True
    except ValueError:
        est = estimate_phase_offset(samples, "qpsk", min_symmetry=0.0)
        accepted = False
    return {
        "accepted": accepted,
        "symmetry": float(est.symmetry),
        "phase_rad": float(est.phase_offset_rad),
        "wrap_err_rad": float(_wrap_phase(est.phase_offset_rad - PHI, PHASE_PERIOD)),
    }


def estimate_wholeblock_residual(samples: np.ndarray, fs: float) -> tuple[float, float]:
    """Return (residual_hz, elapsed_s). Search restricted to [-50, +50] Hz."""
    t0 = time.perf_counter()
    samples = np.asarray(samples)
    scale = float(np.mean(np.abs(samples)))
    if scale == 0.0:
        raise ValueError("zero-energy samples")
    z = (samples / scale) ** M
    n = np.arange(z.size, dtype=np.float64)
    nfft = FFT_PAD * z.size
    spectrum = np.fft.fft(z, n=nfft)
    freqs_z = np.fft.fftfreq(nfft, d=1.0 / fs)
    residual_axis = freqs_z / M
    mask = (residual_axis >= -SEARCH_HZ) & (residual_axis <= SEARCH_HZ)
    if not np.any(mask):
        raise ValueError("no FFT bins inside residual search range")
    peak_local = int(np.argmax(np.abs(spectrum[mask]) ** 2))
    df0 = float(residual_axis[mask][peak_local])

    z_bin_hz = fs / nfft
    df_bin_hz = z_bin_hz / M
    half_width = REFINE_BINS * df_bin_hz
    lo = max(-SEARCH_HZ, df0 - half_width)
    hi = min(SEARCH_HZ, df0 + half_width)
    if hi <= lo:
        hi = min(SEARCH_HZ, lo + df_bin_hz)

    def coherent_power(df: float) -> float:
        phasor = np.exp(-1j * 2.0 * np.pi * (M * df) * n / fs)
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


def evaluate_chain(
    after_11a: np.ndarray,
    extra_correction_hz: float,
    clean: np.ndarray,
    true_bits: np.ndarray,
    f_11a: float,
) -> dict:
    corrected = correct_frequency_offset(after_11a, FS, extra_correction_hz)
    phys = physical_residual_hz(CFO_HZ, f_11a, extra_correction_hz)
    phase = measure_phase(corrected)
    ber, ser = qpsk_errors(corrected, true_bits, SPS)
    phased = correct_phase_offset(corrected, phase["phase_rad"])
    ber_p, ser_p = qpsk_errors(phased, true_bits, SPS)
    return {
        "extra_hz": float(extra_correction_hz),
        "physical_residual_hz": phys,
        "independent_residual_hz": independent_residual_hz(corrected, clean, FS),
        **phase,
        "ber": ber,
        "ser": ser,
        "ber_after_11b": ber_p,
        "ser_after_11b": ser_p,
    }


def print_row(label: str, row: dict, est_hz=None, runtime_s=None) -> None:
    est = f"{est_hz:+.6f}" if est_hz is not None else "oracle"
    rt = f"{runtime_s*1e3:.2f} ms" if runtime_s is not None else "n/a"
    status = "accept" if row["accepted"] else "REJECT"
    print(
        f"  {label:18s} est={est:>12s} Hz  "
        f"phys={row['physical_residual_hz']:+.6f} Hz  "
        f"indep={row['independent_residual_hz']:+.6f} Hz  "
        f"11B {status:6s}  sym={row['symmetry']:.4f}  "
        f"wrap_err={row['wrap_err_rad']:+.6f} rad  "
        f"BER={row['ber']:.4e}  SER={row['ser']:.4e}  "
        f"BER@11B={row['ber_after_11b']:.4e}  SER@11B={row['ser_after_11b']:.4e}  "
        f"runtime={rt}"
    )


def main() -> None:
    bits = np.random.default_rng(BIT_SEED).integers(0, 2, NBITS)
    clean = qpsk_waveform(bits, SPS)
    impaired = apply_frequency_offset(apply_phase_offset(clean, PHI), FS, CFO_HZ)

    print("Stage-A whole-block M-th-power residual CFO experiment")
    print(
        f"QPSK seed={BIT_SEED} nbits={NBITS} sps={SPS} fs={FS:.0f} "
        f"phi={PHI} cfo={CFO_HZ:.0f} Hz  SNR={SNR_DB:.0f} dB"
    )
    print("Search [-50, +50] Hz; FFT pad 8x; Brent around FFT peak.")
    print("Primary frequency truth: injected physical residual.")
    print("Research target (not a product spec): |phys residual| = 0.005 Hz")
    print()

    any_bad = False
    for seed in NOISE_SEEDS:
        noisy = add_awgn(impaired, SNR_DB, np.random.default_rng(seed))
        coarse = estimate_frequency_offset(noisy, FS)
        f_11a = float(coarse.frequency_offset_hz)
        after_11a = correct_frequency_offset(noisy, FS, f_11a)
        true_after_11a = physical_residual_hz(CFO_HZ, f_11a)

        t1 = time.perf_counter()
        lag1 = estimate_residual_frequency_offset(after_11a, FS, "qpsk")
        lag1_s = time.perf_counter() - t1
        f_lag1 = float(lag1.residual_frequency_hz)

        f_block, block_s = estimate_wholeblock_residual(after_11a, FS)
        f_oracle = true_after_11a

        row_a = evaluate_chain(after_11a, 0.0, clean, bits, f_11a)
        row_b = evaluate_chain(after_11a, f_lag1, clean, bits, f_11a)
        row_c = evaluate_chain(after_11a, f_block, clean, bits, f_11a)
        row_d = evaluate_chain(after_11a, f_oracle, clean, bits, f_11a)

        print(f"=== noise seed {seed} ===")
        print(
            f"  11A={f_11a:.6f} Hz  true physical residual after 11A="
            f"{true_after_11a:+.6f} Hz  (CFO_true - 11A)"
        )
        print_row("A 11A only", row_a, est_hz=0.0, runtime_s=None)
        print_row("B lag-1 11C", row_b, est_hz=f_lag1, runtime_s=lag1_s)
        print_row("C whole-block", row_c, est_hz=f_block, runtime_s=block_s)
        print_row("D oracle", row_d, est_hz=f_oracle, runtime_s=None)

        phys_c = abs(row_c["physical_residual_hz"])
        if phys_c > 0.5:
            any_bad = True
            print(
                f"  FAIL-BAD: whole-block |physical residual|={phys_c:.4f} Hz "
                f"on seed {seed} (still Hz-scale)."
            )
        elif phys_c > 0.005:
            print(
                f"  NOTE: seed {seed} missed the 0.005 Hz research target "
                f"(|phys|={phys_c:.6f} Hz); not automatically 'failed badly'."
            )
        else:
            print(
                f"  TARGET MET: seed {seed} |phys residual|={phys_c:.6f} Hz "
                f"<= 0.005 Hz"
            )
        print()

    if any_bad:
        print("Stage-A STOP: whole-block candidate failed badly on at least one seed.")
    else:
        print(
            "Stage-A STOP: experiment finished. No further algorithm, "
            "no held-out set, no production change."
        )


if __name__ == "__main__":
    main()
