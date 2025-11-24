"""
MODULE: bssn_solver.py
CLASSIFICATION: V11.0 Legacy Solver
GOAL: Minimal BSSN-style geometry evolution driver (legacy / open-loop).
CONTRACT ID: IO-LEG-V11
"""
from __future__ import annotations
from typing import Callable, Dict, Any, Tuple
import numpy as np

def bssn_initial_state(N: int, dr: float) -> Dict[str, np.ndarray]:
    """
    Initialize a flat conformal metric state.
    """
    psi = np.ones(N, dtype=float)
    K = np.zeros(N, dtype=float)
    return {"psi": psi, "K": K}

def integrate_bssn(
    T_end: float,
    dt: float,
    N: int,
    source_func: Callable[[float], Dict[str, np.ndarray]],
    params: Dict[str, Any],
    initial_state: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Simple BSSN-like evolution for a conformal factor psi(x,t).
    """
    dr = float(params.get("dr", 1.0))
    kappa_g = float(params.get("kappa_g", 0.1))

    if initial_state is None:
        state = bssn_initial_state(N, dr)
    else:
        state = {k: np.array(v, copy=True) for k, v in initial_state.items()}

    n_steps = int(np.floor(T_end / dt))
    time = np.linspace(0.0, n_steps * dt, n_steps + 1)

    psi = state["psi"]
    K = state["K"]

    def laplacian_periodic_1d(field: np.ndarray) -> np.ndarray:
        return (np.roll(field, -1) - 2.0 * field + np.roll(field, 1)) / (dr * dr)

    for i in range(1, n_steps + 1):
        t = i * dt

        src = source_func(t)
        rho_energy = src["rho_energy"]

        # Very basic toy evolution:
        #  d psi / dt ~ kappa_g * rho_energy
        #  d K   / dt ~ laplacian(psi) - K damping
        dpsi_dt = kappa_g * rho_energy
        dK_dt = laplacian_periodic_1d(psi) - 0.5 * K

        psi = psi + dt * dpsi_dt
        K = K + dt * dK_dt

    final_state = {"psi": psi, "K": K}
    return time, final_state