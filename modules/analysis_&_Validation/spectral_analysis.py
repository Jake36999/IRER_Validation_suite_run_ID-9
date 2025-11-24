"""
MODULE: spectral_analysis.py
CLASSIFICATION: V11.0 Analysis Tool
GOAL: Spectral analysis utilities for rho_history.
CONTRACT ID: IO-SPEC-V11
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

def compute_psd_heatmap(
    rho_history: np.ndarray,
    L_domain: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a simple spatial-FFT PSD heatmap over time.

    Returns
    -------
    times : np.ndarray    shape (T,)
    ks    : np.ndarray    shape (N,)  (wave numbers)
    psd   : np.ndarray    shape (T, N) power spectral density
    """
    rho_history = np.asarray(rho_history, dtype=float)
    T, N = rho_history.shape

    dx = L_domain / float(N)
    times = np.arange(T) * dt
    k_vals = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

    psd = np.zeros((T, N), dtype=float)
    for t_idx in range(T):
        fft_vals = np.fft.fft(rho_history[t_idx])
        psd[t_idx] = np.abs(fft_vals) ** 2

    return times, k_vals, psd