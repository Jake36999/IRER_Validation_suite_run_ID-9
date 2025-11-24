"""
MODULE: spectral_validation.py
CLASSIFICATION: V11.0 Validation Logic
GOAL: Spectral validation against ln(primes) targets.
CONTRACT ID: IO-VAL-V11
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

def _sieve_primes_upto(n: int) -> List[int]:
    """Simple Sieve of Eratosthenes up to n (inclusive)."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return [int(i) for i, is_p in enumerate(sieve) if is_p]

def generate_ln_prime_targets(max_prime: int) -> np.ndarray:
    """
    Generate ln(p) targets for primes p <= max_prime.
    """
    primes = _sieve_primes_upto(max_prime)
    return np.log(np.array(primes, dtype=float))

@dataclass
class PeakMatchResult:
    sse: float
    matched_peaks: List[float]
    target_ln_primes: List[float]
    unmatched_targets: List[float]

def match_peaks_to_ln_primes(
    peak_ks: np.ndarray,
    max_prime: int,
    tol: float,
) -> PeakMatchResult:
    """
    Match observed spectral peaks to ln(primes) targets within tolerance.
    """
    peak_ks = np.asarray(peak_ks, dtype=float)
    ln_primes = generate_ln_prime_targets(max_prime)

    matched_peaks: List[float] = []
    target_list: List[float] = []
    unmatched: List[float] = []

    sq_errors: List[float] = []

    for ln_p in ln_primes:
        idx = int(np.argmin(np.abs(peak_ks - ln_p)))
        diff = float(peak_ks[idx] - ln_p)
        if abs(diff) <= tol:
            matched_peaks.append(float(peak_ks[idx]))
            target_list.append(float(ln_p))
            sq_errors.append(diff * diff)
        else:
            unmatched.append(float(ln_p))

    if not sq_errors:
        sse_val = 1.0  # Sentinel for "no spectral resonance detected"
    else:
        sse_val = float(np.sum(sq_errors))

    return PeakMatchResult(
        sse=sse_val,
        matched_peaks=matched_peaks,
        target_ln_primes=target_list,
        unmatched_targets=unmatched,
    )

def calculate_real_sse(
    peak_ks: np.ndarray,
    max_prime: int,
    tol: float,
) -> PeakMatchResult:
    """
    Top-level entrypoint: compute PeakMatchResult and Real SSE sentinel.
    """
    return match_peaks_to_ln_primes(peak_ks, max_prime, tol)