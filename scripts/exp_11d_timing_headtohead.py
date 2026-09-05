"""Head-to-head of two known-SPS integer timing estimators.

Accepted production verdict (2026-09-06):
    Candidate A (transition-residue) is the production Module 11D
    estimator. Candidate B (whole-block least-squares) is PARKED: it
    did not outperform A on any tested record. PARKED means keep as a
    future candidate, not that B failed or was scientifically rejected.

This script is preserved as the experiment record. Re-running it is
optional; it is not a production test.

Candidate A — transition-residue (production symbol-grid convention):
    d[n] = |x[n] - x[n-1]|
    accumulate into bin n % SPS
    delay_hat = argmax bin
    diagnostic = chance-corrected concentration of the winning bin

Candidate B — whole-block least-squares timing:
    for each residue r, drop r leading samples, partition into complete
    SPS-sample blocks, fit one complex mean per block, score
    MSE = SSE / n_samples_used
    delay_hat = argmin MSE
    a common-support winner (same n_blocks for every r) is computed only
    as a fairness diagnostic, not as a third candidate

Ground truth:
    received = clean[C:]
    true_delay = (-C) % SPS

Independent checks do not rerun the estimator after correction:
    aligned samples vs the known clean slice
    BER/SER vs the known remaining bits after existing known-timing demod
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from iqwav.demod import bpsk_demodulate, qpsk_demodulate
from iqwav.dsp import add_awgn, apply_phase_offset
from iqwav.modulation import bpsk_waveform, qpsk_waveform

FS = 80_000.0
N_SYMBOLS = 512
SPS_VALUES = (2, 4, 8, 16)
BIT_SEEDS = (70_000, 70_001, 70_002, 70_003, 70_004)
NOISE_SEEDS = (80_000, 80_001, 80_002, 80_003, 80_004)
MAIN_SNRS = (None, 20.0, 10.0)  # None = clean
CHAR_SNR = 0.0
PHASE_RAD = 0.8
AMP_SCALE = 3.7
TIE_ATOL = 1e-18
TIE_RTOL = 1e-12
ONE_TRANSITION_SYMBOLS = 64


@dataclass(frozen=True)
class TimingEstimate:
    delay: int
    diagnostic: float
    n_tied: int
    elapsed_s: float
    scores: tuple[float, ...]
    extra: dict


def true_delay(crop: int, sps: int) -> int:
    return int((-crop) % sps)


def make_bits(modulation: str, bit_seed: int, n_symbols: int) -> np.ndarray:
    rng = np.random.default_rng(bit_seed)
    n_bits = n_symbols if modulation == "bpsk" else 2 * n_symbols
    return rng.integers(0, 2, n_bits, dtype=np.int64)


def modulate(bits: np.ndarray, modulation: str, sps: int) -> np.ndarray:
    if modulation == "bpsk":
        return bpsk_waveform(bits, sps)
    return qpsk_waveform(bits, sps)


def one_transition_bits(modulation: str, n_symbols: int) -> np.ndarray:
    if modulation == "bpsk":
        bits = np.zeros(n_symbols, dtype=np.int64)
        bits[n_symbols // 2 :] = 1
        return bits
    bits = np.zeros(2 * n_symbols, dtype=np.int64)
    bits[2 * (n_symbols // 2) :] = 1
    return bits


def constant_bits(modulation: str, n_symbols: int) -> np.ndarray:
    n_bits = n_symbols if modulation == "bpsk" else 2 * n_symbols
    return np.zeros(n_bits, dtype=np.int64)


def align(samples: np.ndarray, delay: int, sps: int) -> np.ndarray:
    n_complete = (samples.size - delay) // sps
    if n_complete < 1:
        return samples[delay:delay]
    return samples[delay : delay + n_complete * sps]


def oracle_start_and_count(clean_len: int, crop: int, sps: int) -> tuple[int, int, int]:
    delay = true_delay(crop, sps)
    start = crop + delay
    n_complete = (clean_len - start) // sps
    return delay, start, n_complete


def remaining_bits(
    bits: np.ndarray, modulation: str, start_sample: int, n_complete: int, sps: int
) -> np.ndarray:
    start_sym = start_sample // sps
    if modulation == "bpsk":
        return bits[start_sym : start_sym + n_complete]
    return bits[2 * start_sym : 2 * (start_sym + n_complete)]


def estimate_a(samples: np.ndarray, sps: int) -> TimingEstimate:
    x = np.asarray(samples)
    t0 = time.perf_counter()
    if x.size < 2:
        elapsed = time.perf_counter() - t0
        return TimingEstimate(0, 0.0, sps, elapsed, tuple(0.0 for _ in range(sps)), {})
    magnitudes = np.abs(np.diff(x)).astype(np.float64)
    indices = np.arange(1, x.size, dtype=np.int64)
    binned = np.bincount(indices % sps, weights=magnitudes, minlength=sps).astype(
        np.float64
    )
    elapsed = time.perf_counter() - t0
    total = float(np.sum(binned))
    delay = int(np.argmax(binned))
    if total <= 0.0:
        n_tied = int(sps)
        concentration = 0.0
        quality = 0.0
    else:
        winning = float(binned[delay])
        n_tied = int(np.count_nonzero(np.isclose(binned, winning, rtol=TIE_RTOL, atol=TIE_ATOL)))
        concentration = min(winning / total, 1.0)
        chance = 1.0 / sps
        quality = (concentration - chance) / (1.0 - chance)
    return TimingEstimate(
        delay=delay,
        diagnostic=float(quality),
        n_tied=n_tied,
        elapsed_s=elapsed,
        scores=tuple(float(v) for v in binned),
        extra={"concentration": float(concentration if total > 0.0 else 0.0), "total": total},
    )


def estimate_b(samples: np.ndarray, sps: int) -> TimingEstimate:
    x = np.asarray(samples, dtype=np.complex128)
    n = int(x.size)
    mse = np.full(sps, np.inf, dtype=np.float64)
    mse_common = np.full(sps, np.inf, dtype=np.float64)
    n_blocks = np.array([(n - r) // sps for r in range(sps)], dtype=np.int64)
    positive = n_blocks[n_blocks >= 1]
    min_blocks = int(positive.min()) if positive.size else 0
    t0 = time.perf_counter()
    for r in range(sps):
        nb = int(n_blocks[r])
        if nb < 1:
            continue
        block = x[r : r + nb * sps].reshape(nb, sps)
        mu = block.mean(axis=1, keepdims=True)
        sse = float(np.sum(np.abs(block - mu) ** 2))
        mse[r] = sse / float(nb * sps)
        if min_blocks >= 1:
            block_c = x[r : r + min_blocks * sps].reshape(min_blocks, sps)
            mu_c = block_c.mean(axis=1, keepdims=True)
            sse_c = float(np.sum(np.abs(block_c - mu_c) ** 2))
            mse_common[r] = sse_c / float(min_blocks * sps)
    elapsed = time.perf_counter() - t0
    finite = np.isfinite(mse)
    if not np.any(finite):
        delay = 0
        n_tied = sps
        win = float("inf")
        gap = 0.0
    else:
        win = float(np.min(mse[finite]))
        delay = int(np.argmin(mse))
        n_tied = int(
            np.count_nonzero(
                finite & np.isclose(mse, win, rtol=TIE_RTOL, atol=TIE_ATOL)
            )
        )
        others = mse[finite & (np.arange(sps) != delay)]
        gap = float(np.min(others) - win) if others.size else 0.0
    common_delay = int(np.argmin(mse_common)) if np.any(np.isfinite(mse_common)) else 0
    return TimingEstimate(
        delay=delay,
        diagnostic=float(win) if np.isfinite(win) else float("inf"),
        n_tied=n_tied,
        elapsed_s=elapsed,
        scores=tuple(float(v) for v in mse),
        extra={
            "mse_gap": gap,
            "common_delay": common_delay,
            "common_disagrees": int(common_delay != delay),
            "min_blocks": min_blocks,
        },
    )


def bit_symbol_errors(
    samples: np.ndarray, true_bits: np.ndarray, modulation: str, sps: int
) -> tuple[int, int, int, int]:
    n_bits = int(true_bits.size)
    if n_bits == 0 or samples.size < sps:
        return n_bits, n_bits, n_bits if modulation == "bpsk" else n_bits // 2, max(
            n_bits // (1 if modulation == "bpsk" else 2), 0
        )
    if modulation == "qpsk":
        n_sym = n_bits // 2
        true_pairs = true_bits.reshape(n_sym, 2)
        best_bit = n_bits
        best_sym = n_sym
        for k in range(4):
            bits = qpsk_demodulate(samples * (1j**k), sps)
            n_use = min(bits.size, n_bits)
            use = bits[:n_use]
            ref = true_bits[:n_use]
            n_sym_use = n_use // 2
            bit_err = int(np.count_nonzero(use != ref)) + (n_bits - n_use)
            if n_sym_use > 0:
                rec_pairs = use.reshape(-1, 2)[:n_sym_use]
                ref_pairs = true_pairs[:n_sym_use]
                sym_err = int(np.count_nonzero(np.any(rec_pairs != ref_pairs, axis=1)))
            else:
                sym_err = n_sym
            sym_err += n_sym - n_sym_use
            if bit_err < best_bit:
                best_bit = bit_err
                best_sym = sym_err
        return best_bit, n_bits, best_sym, n_sym
    best_bit = n_bits
    for sign in (1.0, -1.0):
        bits = bpsk_demodulate(samples * sign, sps)
        n_use = min(bits.size, n_bits)
        bit_err = int(np.count_nonzero(bits[:n_use] != true_bits[:n_use])) + (
            n_bits - n_use
        )
        if bit_err < best_bit:
            best_bit = bit_err
    return best_bit, n_bits, best_bit, n_bits


def evaluate_alignment(
    received: np.ndarray,
    clean: np.ndarray,
    bits: np.ndarray,
    modulation: str,
    sps: int,
    crop: int,
    delay_hat: int,
    *,
    check_equality: bool,
) -> dict:
    oracle_delay, start, n_oracle = oracle_start_and_count(clean.size, crop, sps)
    aligned = align(received, delay_hat, sps)
    oracle_aligned_clean = clean[start : start + n_oracle * sps]
    oracle_bits = remaining_bits(bits, modulation, start, n_oracle, sps)
    exact = int(delay_hat == oracle_delay)
    equal = False
    if check_equality:
        equal = aligned.size == oracle_aligned_clean.size and np.array_equal(
            aligned, oracle_aligned_clean
        )
    bit_e, n_bits, sym_e, n_sym = bit_symbol_errors(
        aligned, oracle_bits, modulation, sps
    )
    oracle_aligned_rx = align(received, oracle_delay, sps)
    o_bit_e, o_n_bits, o_sym_e, o_n_sym = bit_symbol_errors(
        oracle_aligned_rx, oracle_bits, modulation, sps
    )
    naive = align(received, 0, sps)
    naive_bits = remaining_bits(bits, modulation, crop, naive.size // sps, sps)
    # Naive BER is vs the bits that would be correct IF delay 0 were right.
    # Independent floor uses oracle remaining bits against delay-0 demod when
    # crop==0; otherwise delay 0 is the wrong grid. Compare naive demod to
    # oracle remaining bits so a wrong grid is visible as BER.
    n_bit_e, n_n_bits, n_sym_e, n_n_sym = bit_symbol_errors(
        naive, oracle_bits, modulation, sps
    )
    return {
        "exact": exact,
        "equal_clean": int(equal),
        "oracle_delay": oracle_delay,
        "n_oracle": n_oracle,
        "bit_errors": bit_e,
        "n_bits": n_bits,
        "sym_errors": sym_e,
        "n_symbols": n_sym,
        "oracle_bit_errors": o_bit_e,
        "oracle_n_bits": o_n_bits,
        "oracle_sym_errors": o_sym_e,
        "oracle_n_symbols": o_n_sym,
        "naive_bit_errors": n_bit_e,
        "naive_n_bits": n_n_bits,
        "naive_sym_errors": n_sym_e,
        "naive_n_symbols": n_n_sym,
        "naive_bits_if_zero_true": naive_bits,
    }


def snr_label(snr: float | None) -> str:
    return "clean" if snr is None else f"{snr:.0f} dB"


def prepare_received(
    clean: np.ndarray, crop: int, snr: float | None, noise_seed: int | None
) -> np.ndarray:
    if snr is None:
        return clean[crop:]
    noisy = add_awgn(clean, snr, rng=np.random.default_rng(noise_seed))
    return noisy[crop:]


def run_main_grid() -> list[dict]:
    rows: list[dict] = []
    snrs: list[float | None] = list(MAIN_SNRS) + [CHAR_SNR]
    for modulation in ("bpsk", "qpsk"):
        for sps in SPS_VALUES:
            for bit_seed in BIT_SEEDS:
                bits = make_bits(modulation, bit_seed, N_SYMBOLS)
                clean = modulate(bits, modulation, sps)
                for snr in snrs:
                    noise_iter: tuple[int | None, ...]
                    if snr is None:
                        noise_iter = (None,)
                    else:
                        noise_iter = NOISE_SEEDS
                    for noise_seed in noise_iter:
                        # Add noise once per (clean, snr, noise_seed), then crop.
                        if snr is None:
                            observed = clean
                        else:
                            observed = add_awgn(
                                clean, snr, rng=np.random.default_rng(noise_seed)
                            )
                        for crop in range(sps):
                            received = observed[crop:]
                            est_a = estimate_a(received, sps)
                            est_b = estimate_b(received, sps)
                            check_eq = snr is None
                            ev_a = evaluate_alignment(
                                received,
                                clean,
                                bits,
                                modulation,
                                sps,
                                crop,
                                est_a.delay,
                                check_equality=check_eq,
                            )
                            ev_b = evaluate_alignment(
                                received,
                                clean,
                                bits,
                                modulation,
                                sps,
                                crop,
                                est_b.delay,
                                check_equality=check_eq,
                            )
                            rows.append(
                                {
                                    "modulation": modulation,
                                    "sps": sps,
                                    "crop": crop,
                                    "snr": snr,
                                    "bit_seed": bit_seed,
                                    "noise_seed": noise_seed,
                                    "true_delay": true_delay(crop, sps),
                                    "a_delay": est_a.delay,
                                    "b_delay": est_b.delay,
                                    "a_exact": ev_a["exact"],
                                    "b_exact": ev_b["exact"],
                                    "a_equal": ev_a["equal_clean"],
                                    "b_equal": ev_b["equal_clean"],
                                    "a_tied": est_a.n_tied,
                                    "b_tied": est_b.n_tied,
                                    "a_diag": est_a.diagnostic,
                                    "b_diag": est_b.diagnostic,
                                    "b_gap": est_b.extra["mse_gap"],
                                    "b_common_delay": est_b.extra["common_delay"],
                                    "b_common_disagrees": est_b.extra["common_disagrees"],
                                    "a_time": est_a.elapsed_s,
                                    "b_time": est_b.elapsed_s,
                                    "disagree": int(est_a.delay != est_b.delay),
                                    "a_bit_errors": ev_a["bit_errors"],
                                    "b_bit_errors": ev_b["bit_errors"],
                                    "o_bit_errors": ev_a["oracle_bit_errors"],
                                    "n_bit_errors": ev_a["naive_bit_errors"],
                                    "a_n_bits": ev_a["n_bits"],
                                    "b_n_bits": ev_b["n_bits"],
                                    "o_n_bits": ev_a["oracle_n_bits"],
                                    "n_n_bits": ev_a["naive_n_bits"],
                                    "a_sym_errors": ev_a["sym_errors"],
                                    "b_sym_errors": ev_b["sym_errors"],
                                    "o_sym_errors": ev_a["oracle_sym_errors"],
                                    "n_sym_errors": ev_a["naive_sym_errors"],
                                    "a_n_sym": ev_a["n_symbols"],
                                    "b_n_sym": ev_b["n_symbols"],
                                    "o_n_sym": ev_a["oracle_n_symbols"],
                                    "n_n_sym": ev_a["naive_n_symbols"],
                                }
                            )
    return rows


def run_invariance() -> list[dict]:
    rows: list[dict] = []
    bit_seed = BIT_SEEDS[0]
    for modulation in ("bpsk", "qpsk"):
        for sps in SPS_VALUES:
            bits = make_bits(modulation, bit_seed, N_SYMBOLS)
            clean = modulate(bits, modulation, sps)
            variants = {
                "baseline": clean,
                "phase": apply_phase_offset(clean, PHASE_RAD),
                "scale": clean * AMP_SCALE,
                "phase+scale": apply_phase_offset(clean, PHASE_RAD) * AMP_SCALE,
            }
            for crop in range(sps):
                baseline_rx = variants["baseline"][crop:]
                a0 = estimate_a(baseline_rx, sps).delay
                b0 = estimate_b(baseline_rx, sps).delay
                truth = true_delay(crop, sps)
                for name, waveform in variants.items():
                    if name == "baseline":
                        continue
                    received = waveform[crop:]
                    a = estimate_a(received, sps)
                    b = estimate_b(received, sps)
                    rows.append(
                        {
                            "modulation": modulation,
                            "sps": sps,
                            "crop": crop,
                            "variant": name,
                            "true_delay": truth,
                            "a_delay": a.delay,
                            "b_delay": b.delay,
                            "a_match_truth": int(a.delay == truth),
                            "b_match_truth": int(b.delay == truth),
                            "a_match_baseline": int(a.delay == a0),
                            "b_match_baseline": int(b.delay == b0),
                        }
                    )
    return rows


def run_pathological() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"constant": [], "one_transition": []}
    for kind in ("constant", "one_transition"):
        n_symbols = ONE_TRANSITION_SYMBOLS
        for modulation in ("bpsk", "qpsk"):
            if kind == "constant":
                bits = constant_bits(modulation, n_symbols)
            else:
                bits = one_transition_bits(modulation, n_symbols)
            for sps in SPS_VALUES:
                clean = modulate(bits, modulation, sps)
                n_trans = int(np.count_nonzero(np.abs(np.diff(clean)) > 0.0))
                for crop in range(sps):
                    received = clean[crop:]
                    n_trans_rx = int(
                        np.count_nonzero(np.abs(np.diff(received)) > 0.0)
                    )
                    a = estimate_a(received, sps)
                    b = estimate_b(received, sps)
                    truth = true_delay(crop, sps)
                    out[kind].append(
                        {
                            "modulation": modulation,
                            "sps": sps,
                            "crop": crop,
                            "true_delay": truth,
                            "n_trans_clean": n_trans,
                            "n_trans_rx": n_trans_rx,
                            "a_delay": a.delay,
                            "b_delay": b.delay,
                            "a_exact": int(a.delay == truth),
                            "b_exact": int(b.delay == truth),
                            "a_tied": a.n_tied,
                            "b_tied": b.n_tied,
                            "a_diag": a.diagnostic,
                            "b_diag": b.diagnostic,
                            "a_scores": a.scores,
                            "b_scores": b.scores,
                            "b_common_delay": b.extra["common_delay"],
                        }
                    )
    return out


def group_key(row: dict) -> tuple:
    return (snr_label(row["snr"]), row["modulation"], row["sps"])


def summarize_main(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    print()
    print("=== Exact boundary-offset accuracy ===")
    print(
        f"{'SNR':<8} {'mod':<5} {'SPS':>3} {'n':>5} "
        f"{'A_exact':>8} {'B_exact':>8} {'A_wrong':>8} {'B_wrong':>8} "
        f"{'disagree':>9} {'A_ties>1':>9} {'B_ties>1':>9} {'B_vs_common':>12}"
    )
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        n = len(g)
        a_exact = sum(r["a_exact"] for r in g)
        b_exact = sum(r["b_exact"] for r in g)
        a_tie = sum(r["a_tied"] > 1 for r in g)
        b_tie = sum(r["b_tied"] > 1 for r in g)
        dis = sum(r["disagree"] for r in g)
        common = sum(r["b_common_disagrees"] for r in g)
        print(
            f"{key[0]:<8} {key[1]:<5} {key[2]:>3} {n:>5} "
            f"{a_exact:>8} {b_exact:>8} {n - a_exact:>8} {n - b_exact:>8} "
            f"{dis:>9} {a_tie:>9} {b_tie:>9} {common:>12}"
        )

    print()
    print("=== Clean aligned-output equality to known clean slice ===")
    print(
        f"{'mod':<5} {'SPS':>3} {'n':>5} {'A_equal':>8} {'B_equal':>8} "
        f"{'A_eq_given_exact':>18} {'B_eq_given_exact':>18}"
    )
    clean_rows = [r for r in rows if r["snr"] is None]
    clean_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in clean_rows:
        clean_groups[(row["modulation"], row["sps"])].append(row)
    for key in sorted(clean_groups):
        g = clean_groups[key]
        n = len(g)
        a_eq = sum(r["a_equal"] for r in g)
        b_eq = sum(r["b_equal"] for r in g)
        a_ex = [r for r in g if r["a_exact"]]
        b_ex = [r for r in g if r["b_exact"]]
        a_cond = sum(r["a_equal"] for r in a_ex) if a_ex else 0
        b_cond = sum(r["b_equal"] for r in b_ex) if b_ex else 0
        print(
            f"{key[0]:<5} {key[1]:>3} {n:>5} {a_eq:>8} {b_eq:>8} "
            f"{a_cond:>8}/{len(a_ex):<8} {b_cond:>8}/{len(b_ex):<8}"
        )

    print()
    print("=== BER/SER after correction + known-timing demod (vs oracle remaining bits) ===")
    print(
        f"{'SNR':<8} {'mod':<5} {'SPS':>3} "
        f"{'A_BER':>10} {'B_BER':>10} {'oracle':>10} {'naive0':>10} "
        f"{'A_SER':>10} {'B_SER':>10}"
    )
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]

        def rate(err_key: str, n_key: str) -> str:
            err = sum(r[err_key] for r in g)
            tot = sum(r[n_key] for r in g)
            if tot == 0:
                return "n/a"
            return f"{err / tot:.3e}"

        print(
            f"{key[0]:<8} {key[1]:<5} {key[2]:>3} "
            f"{rate('a_bit_errors', 'a_n_bits'):>10} "
            f"{rate('b_bit_errors', 'b_n_bits'):>10} "
            f"{rate('o_bit_errors', 'o_n_bits'):>10} "
            f"{rate('n_bit_errors', 'n_n_bits'):>10} "
            f"{rate('a_sym_errors', 'a_n_sym'):>10} "
            f"{rate('b_sym_errors', 'b_n_sym'):>10}"
        )

    print()
    print("=== Conditional BER given exact offset (estimator already correct) ===")
    print(f"{'SNR':<8} {'mod':<5} {'SPS':>3} {'A_n_exact':>10} {'A_BER':>10} {'B_n_exact':>10} {'B_BER':>10}")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        a_ex = [r for r in g if r["a_exact"]]
        b_ex = [r for r in g if r["b_exact"]]

        def rate_rows(subset: list[dict], err_key: str, n_key: str) -> str:
            if not subset:
                return "n/a"
            err = sum(r[err_key] for r in subset)
            tot = sum(r[n_key] for r in subset)
            return f"{err / tot:.3e}" if tot else "n/a"

        print(
            f"{key[0]:<8} {key[1]:<5} {key[2]:>3} "
            f"{len(a_ex):>10} {rate_rows(a_ex, 'a_bit_errors', 'a_n_bits'):>10} "
            f"{len(b_ex):>10} {rate_rows(b_ex, 'b_bit_errors', 'b_n_bits'):>10}"
        )

    print()
    print("=== Runtime (estimator only, seconds) ===")
    print(
        f"{'SNR':<8} {'mod':<5} {'SPS':>3} "
        f"{'A_med':>10} {'A_p95':>10} {'B_med':>10} {'B_p95':>10} {'B/A med':>10}"
    )
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        a_t = np.array([r["a_time"] for r in g], dtype=np.float64)
        b_t = np.array([r["b_time"] for r in g], dtype=np.float64)
        ratio = float(np.median(b_t) / np.median(a_t)) if np.median(a_t) > 0 else float("inf")
        print(
            f"{key[0]:<8} {key[1]:<5} {key[2]:>3} "
            f"{np.median(a_t):>10.4e} {np.percentile(a_t, 95):>10.4e} "
            f"{np.median(b_t):>10.4e} {np.percentile(b_t, 95):>10.4e} "
            f"{ratio:>10.2f}"
        )

    print()
    print("=== Diagnostics (not used as a reject threshold) ===")
    print(
        f"{'SNR':<8} {'mod':<5} {'SPS':>3} "
        f"{'A_qual_med':>11} {'A_qual_min':>11} "
        f"{'B_mse_med':>11} {'B_gap_med':>11} {'B_gap_min':>11}"
    )
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        aq = np.array([r["a_diag"] for r in g], dtype=np.float64)
        bm = np.array([r["b_diag"] for r in g], dtype=np.float64)
        bg = np.array([r["b_gap"] for r in g], dtype=np.float64)
        print(
            f"{key[0]:<8} {key[1]:<5} {key[2]:>3} "
            f"{np.median(aq):>11.4f} {np.min(aq):>11.4f} "
            f"{np.median(bm):>11.3e} {np.median(bg):>11.3e} {np.min(bg):>11.3e}"
        )


def print_wrong_examples(rows: list[dict], limit: int = 20) -> None:
    wrong_a = [r for r in rows if not r["a_exact"]]
    wrong_b = [r for r in rows if not r["b_exact"]]
    print()
    print(f"=== Wrong-offset records: A={len(wrong_a)}  B={len(wrong_b)} ===")
    if wrong_a:
        print("First A failures:")
        for r in wrong_a[:limit]:
            print(
                f"  A  {snr_label(r['snr']):<8} {r['modulation']:<5} SPS={r['sps']:<2} "
                f"crop={r['crop']:<2} true={r['true_delay']:<2} hat={r['a_delay']:<2} "
                f"tied={r['a_tied']} qual={r['a_diag']:.4f} "
                f"bit_seed={r['bit_seed']} noise_seed={r['noise_seed']}"
            )
        if len(wrong_a) > limit:
            print(f"  ... {len(wrong_a) - limit} more A failures")
    if wrong_b:
        print("First B failures:")
        for r in wrong_b[:limit]:
            print(
                f"  B  {snr_label(r['snr']):<8} {r['modulation']:<5} SPS={r['sps']:<2} "
                f"crop={r['crop']:<2} true={r['true_delay']:<2} hat={r['b_delay']:<2} "
                f"tied={r['b_tied']} mse={r['b_diag']:.3e} gap={r['b_gap']:.3e} "
                f"bit_seed={r['bit_seed']} noise_seed={r['noise_seed']}"
            )
        if len(wrong_b) > limit:
            print(f"  ... {len(wrong_b) - limit} more B failures")
    if not wrong_a and not wrong_b:
        print("None.")


def print_pathological(data: dict[str, list[dict]]) -> None:
    print()
    print("=== Pathological: constant / zero-transition blocks ===")
    const = data["constant"]
    n = len(const)
    a_exact = sum(r["a_exact"] for r in const)
    b_exact = sum(r["b_exact"] for r in const)
    a_all_tie = sum(r["a_tied"] == r["sps"] for r in const)
    b_all_tie = sum(r["b_tied"] == r["sps"] for r in const)
    a_zero = sum(r["a_delay"] == 0 for r in const)
    b_zero = sum(r["b_delay"] == 0 for r in const)
    trans = {r["n_trans_rx"] for r in const}
    print(f"records={n}  observed transitions in cropped rx={sorted(trans)}")
    print(
        f"A exact vs arbitrary truth {a_exact}/{n}  (meaningless; unidentifiable)  "
        f"always-tied={a_all_tie}/{n}  reports 0={a_zero}/{n}"
    )
    print(
        f"B exact vs arbitrary truth {b_exact}/{n}  (meaningless; unidentifiable)  "
        f"always-tied={b_all_tie}/{n}  reports 0={b_zero}/{n}"
    )
    print("A/B both have identical scores across residues; argmax/argmin return residue 0.")

    print()
    print("=== Pathological: exactly one clean transition ===")
    one = data["one_transition"]
    n = len(one)
    a_exact = sum(r["a_exact"] for r in one)
    b_exact = sum(r["b_exact"] for r in one)
    a_tie = sum(r["a_tied"] > 1 for r in one)
    b_tie = sum(r["b_tied"] > 1 for r in one)
    trans = {r["n_trans_rx"] for r in one}
    print(f"records={n}  observed transitions in cropped rx={sorted(trans)}")
    print(f"A exact={a_exact}/{n}  ties>1={a_tie}/{n}")
    print(f"B exact={b_exact}/{n}  ties>1={b_tie}/{n}")
    wrong = [r for r in one if not r["a_exact"] or not r["b_exact"]]
    if wrong:
        print("Failures:")
        for r in wrong:
            print(
                f"  {r['modulation']:<5} SPS={r['sps']} crop={r['crop']} true={r['true_delay']} "
                f"A={r['a_delay']} (tied={r['a_tied']}, qual={r['a_diag']:.4f}) "
                f"B={r['b_delay']} (tied={r['b_tied']}, mse={r['b_diag']:.3e})"
            )
    else:
        print("Both estimators recovered every one-transition residue. Known-SPS makes a single boundary identifiable.")


def print_invariance(rows: list[dict]) -> None:
    print()
    print("=== Invariance: constant phase 0.8 rad and/or amplitude x3.7 ===")
    print(
        f"{'variant':<12} {'n':>5} {'A=truth':>8} {'B=truth':>8} "
        f"{'A=baseline':>11} {'B=baseline':>11}"
    )
    variants = sorted({r["variant"] for r in rows})
    for name in variants:
        g = [r for r in rows if r["variant"] == name]
        n = len(g)
        print(
            f"{name:<12} {n:>5} "
            f"{sum(r['a_match_truth'] for r in g):>8} "
            f"{sum(r['b_match_truth'] for r in g):>8} "
            f"{sum(r['a_match_baseline'] for r in g):>11} "
            f"{sum(r['b_match_baseline'] for r in g):>11}"
        )
    fail = [
        r
        for r in rows
        if not r["a_match_truth"]
        or not r["b_match_truth"]
        or not r["a_match_baseline"]
        or not r["b_match_baseline"]
    ]
    if fail:
        print("Invariance failures:")
        for r in fail[:20]:
            print(
                f"  {r['variant']:<12} {r['modulation']:<5} SPS={r['sps']} crop={r['crop']} "
                f"true={r['true_delay']} A={r['a_delay']} B={r['b_delay']}"
            )


def overall_counts(rows: list[dict], snrs: tuple[float | None, ...]) -> None:
    print()
    print("=== Overall counts by SNR class ===")
    for snr in snrs:
        g = [r for r in rows if r["snr"] == snr]
        n = len(g)
        print(
            f"{snr_label(snr):<8} n={n:<6} "
            f"A_exact={sum(r['a_exact'] for r in g)}/{n}  "
            f"B_exact={sum(r['b_exact'] for r in g)}/{n}  "
            f"disagree={sum(r['disagree'] for r in g)}  "
            f"B_common_disagrees={sum(r['b_common_disagrees'] for r in g)}"
        )


def main() -> None:
    print("IQWAV Module 11D experiment: known-SPS integer timing head-to-head")
    print("Candidate A = transition-residue  (ACCEPTED production 11D)")
    print("Candidate B = whole-block LS, MSE = SSE / n_samples_used  (PARKED)")
    print(f"N_SYMBOLS={N_SYMBOLS}  SPS={SPS_VALUES}  FS={FS}")
    print(f"BIT_SEEDS={BIT_SEEDS}")
    print(f"NOISE_SEEDS={NOISE_SEEDS}")
    print("Truth: received = clean[C:], true_delay = (-C) % SPS")
    print("No production files modified. No reject threshold tuned.")
    print()

    t0 = time.perf_counter()
    print("Running pathological cases...")
    path = run_pathological()
    print("Running invariance...")
    inv = run_invariance()
    print("Running main grid (clean / 20 dB / 10 dB / 0 dB)...")
    rows = run_main_grid()
    elapsed = time.perf_counter() - t0
    print(f"Finished {len(rows)} main records in {elapsed:.2f} s wall time.")

    print_pathological(path)
    print_invariance(inv)
    overall_counts(rows, (None, 20.0, 10.0, 0.0))
    summarize_main(rows)
    print()
    print("--- Main-grid failures excluding 0 dB characterization ---")
    print_wrong_examples([r for r in rows if r["snr"] in (None, 20.0, 10.0)])
    print()
    print("--- 0 dB characterization failures ---")
    print_wrong_examples([r for r in rows if r["snr"] == 0.0])

    main_decision = [r for r in rows if r["snr"] in (None, 20.0, 10.0)]
    a_wrong_main = sum(not r["a_exact"] for r in main_decision)
    b_wrong_main = sum(not r["b_exact"] for r in main_decision)
    zero = [r for r in rows if r["snr"] == 0.0]
    a_wrong_0 = sum(not r["a_exact"] for r in zero)
    b_wrong_0 = sum(not r["b_exact"] for r in zero)
    one = path["one_transition"]
    const = path["constant"]
    inv_ok = all(
        r["a_match_truth"] and r["b_match_truth"] and r["a_match_baseline"] and r["b_match_baseline"]
        for r in inv
    )
    clean = [r for r in rows if r["snr"] is None]
    a_eq = sum(r["a_equal"] for r in clean)
    b_eq = sum(r["b_equal"] for r in clean)

    print()
    print("=== Recommendation (from this campaign only) ===")
    print(
        f"Main (clean/20/10 dB): A_wrong={a_wrong_main}/{len(main_decision)}  "
        f"B_wrong={b_wrong_main}/{len(main_decision)}"
    )
    print(
        f"0 dB characterization: A_wrong={a_wrong_0}/{len(zero)}  "
        f"B_wrong={b_wrong_0}/{len(zero)}"
    )
    print(
        f"Clean equality: A={a_eq}/{len(clean)}  B={b_eq}/{len(clean)}"
    )
    print(
        f"One-transition exact: A={sum(r['a_exact'] for r in one)}/{len(one)}  "
        f"B={sum(r['b_exact'] for r in one)}/{len(one)}"
    )
    print(
        f"Zero-transition: both unidentifiable; A_tied_all={sum(r['a_tied']==r['sps'] for r in const)}/{len(const)}  "
        f"B_tied_all={sum(r['b_tied']==r['sps'] for r in const)}/{len(const)}"
    )
    print(f"Phase/scale invariance both exact: {inv_ok}")
    print(
        "Fairness: B primary score is SSE/n_used; common-support disagreements are listed above."
    )


if __name__ == "__main__":
    main()
