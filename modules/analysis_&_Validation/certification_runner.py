"""
MODULE: certification_runner.py
CLASSIFICATION: V11.0 Orchestrator
GOAL: Mock certification runner for the IRER validation suite.
CONTRACT ID: IO-CERT-V11
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from fmia_dynamics_solver import generate_mock_attractor_output
from spectral_analysis import compute_psd_heatmap
from spectral_validation import calculate_real_sse, PeakMatchResult

def _extract_peak_ks(psd_row: np.ndarray, k_vals: np.ndarray, n_peaks: int = 8) -> np.ndarray:
    """
    Extract top-n_peaks peak locations from a 1D PSD row.
    """
    idx_sorted = np.argsort(psd_row)[::-1]
    idx_top = np.unique(idx_sorted[:n_peaks])
    return k_vals[idx_top]

def mock_certification_run(
    n_times: int = 256,
    n_points: int = 256,
    L_domain: float = 2.0 * np.pi,
    dt: float = 0.01,
    max_prime: int = 31,
    tol: float = 0.1,
) -> Tuple[PeakMatchResult, float]:
    """
    Run a full mock certification pipeline on synthetic rho_history.
    """
    # 1) Mock attractor
    time, rho_hist = generate_mock_attractor_output(n_times, n_points)

    # 2) PSD heatmap
    times, k_vals, psd = compute_psd_heatmap(rho_hist, L_domain=L_domain, dt=dt)

    # Use final-time PSD as representative spectrum
    final_psd = psd[-1]
    peak_ks = _extract_peak_ks(final_psd, k_vals, n_peaks=8)

    # 3) Spectral validation against ln(primes)
    result = calculate_real_sse(peak_ks, max_prime=max_prime, tol=tol)

    # Quick sanity metric
    psd_peak_norm = float(np.linalg.norm(final_psd, ord=2))

    # Human-readable output
    print("=== IRER Mock Certification Run ===")
    print(f"Time steps       : {n_times}")
    print(f"Grid points      : {n_points}")
    print(f"Num peaks        : {len(peak_ks)}")
    print(f"Real SSE         : {result.sse:.6f}")
    print(f"Matched peaks    : {result.matched_peaks}")
    print(f"Target ln(primes): {result.target_ln_primes}")
    print(f"Unmatched targets: {result.unmatched_targets}")
    print(f"PSD L2 norm      : {psd_peak_norm:.6f}")
    print("====================================")
    if result.sse <= 0.05:
        print("STATUS: PASS (spectral resonance detected within tolerance).")
    elif result.sse < 1.0:
        print("STATUS: BORDERLINE (some resonance, but above strict threshold).")
    else:
        print("STATUS: FAIL (no clear log-prime resonance).")

    return result, psd_peak_norm

if __name__ == "__main__":
    mock_certification_run()