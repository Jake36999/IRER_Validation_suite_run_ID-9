"""
MODULE: fmia_dynamics_solver.py
CLASSIFICATION: V11.0 Dynamics Integrator
GOAL: FMIA standalone dynamics integrator + mock attractor generator.
CONTRACT ID: IO-SOLV-V11
"""
from __future__ import annotations
from typing import Dict, Tuple, Any
import numpy as np
from numerics import rk4_step, JAX_AVAILABLE
from fmia_rhs import fmia_rhs
from io_hdf5 import save_fmia_results_hdf5

if JAX_AVAILABLE:
    import jax.numpy as jnp  # type: ignore
else:
    import numpy as jnp  # type: ignore

def integrate_fmia(
    rho0: np.ndarray,
    pi0: np.ndarray,
    t_end: float,
    dt: float,
    params: Dict[str, Any],
    metric: Any = None,
    save_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate FMIA dynamics for (rho, pi) using RK4.
    """
    rho0 = np.asarray(rho0, dtype=float)
    pi0 = np.asarray(pi0, dtype=float)

    n_steps = int(np.floor(t_end / dt))
    n_grid = rho0.shape[-1]

    # Pre-allocate with conservative size, then trim
    max_saved = n_steps // save_every + 2
    rho_hist = np.zeros((max_saved, n_grid), dtype=float)
    pi_hist = np.zeros((max_saved, n_grid), dtype=float)
    time_hist = np.zeros((max_saved,), dtype=float)

    state = (jnp.array(rho0), jnp.array(pi0))
    t = 0.0

    save_idx = 0
    rho_hist[save_idx] = np.array(state[0])
    pi_hist[save_idx] = np.array(state[1])
    time_hist[save_idx] = t
    save_idx += 1

    for step in range(1, n_steps + 1):
        state = rk4_step(fmia_rhs, state, t, dt, params, metric)
        t = step * dt

        if step % save_every == 0 or step == n_steps:
            rho_hist[save_idx] = np.array(state[0])
            pi_hist[save_idx] = np.array(state[1])
            time_hist[save_idx] = t
            save_idx += 1

    # Trim to actual saved length
    rho_hist = rho_hist[:save_idx]
    pi_hist = pi_hist[:save_idx]
    time_hist = time_hist[:save_idx]

    return time_hist, rho_hist, pi_hist

def generate_mock_attractor_output(
    n_times: int,
    n_points: int,
    baseline: float = 1.0,
    decay: float = 0.01,
    spatial_freq: float = 2.0,
    temporal_freq: float = 1.0,
    noise_level: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a mock 'attractor-like' rho_history for diagnostics / UI demos.
    """
    x = np.linspace(0, 2.0 * np.pi, n_points, endpoint=False)
    time = np.linspace(0.0, 1.0, n_times)

    rho = np.zeros((n_times, n_points), dtype=float)
    for i, t in enumerate(time):
        amp = baseline + np.exp(-decay * t) * np.cos(temporal_freq * 2.0 * np.pi * t)
        pattern = amp * np.cos(spatial_freq * x)
        noise = noise_level * np.random.randn(n_points)
        rho[i] = pattern + noise

    return time, rho

# Convenience re-export – matches spec name
def save_fmia_results(path: str, time: np.ndarray, rho: np.ndarray) -> None:
    """Thin wrapper to central IO function for FMIA-only runs."""
    save_fmia_results_hdf5(path, time, rho)