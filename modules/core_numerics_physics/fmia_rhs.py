"""
MODULE: fmia_rhs.py
CLASSIFICATION: V11.0 Physics Kernel (RHS)
GOAL: FMIA RHS (equations of motion) for the informational field.
      Models a second-order system: state = (rho, pi).
CONTRACT ID: IO-PHYS-V11
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
from numerics import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax.numpy as jnp  # type: ignore
else:
    import numpy as jnp  # type: ignore

def _laplacian_periodic_1d(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Simple 1D periodic Laplacian."""
    return (jnp.roll(field, -1) - 2.0 * field + jnp.roll(field, 1)) / (dx * dx)

def fmia_rhs(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    t: float,
    params: Dict[str, Any],
    metric: Optional[Any] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute FMIA derivatives for (rho, pi) at time t.

    Parameters
    ----------
    state : (rho, pi)
        rho, pi arrays with same shape (1D spatial grid assumed here).
    t : float
        Current time (unused but kept for interface compatibility).
    params : dict
        Contains physics and discretization parameters.
    metric : Any, optional
        Placeholder for metric-aware Laplacian.

    Returns
    -------
    (d rho / dt, d pi / dt)
    """
    rho, pi = state

    D = float(params.get("D", 1.0))
    eps = float(params.get("epsilon", 0.0))
    lam = float(params.get("lam", 1.0))
    eta = float(params.get("eta", 0.0))
    dx = float(params.get("dx", 1.0))

    lap_rho = _laplacian_periodic_1d(rho, dx)

    # TODO: plug in proper non-local term via FFT if provided in params
    nonlocal_term = 0.0

    drho_dt = pi
    dpi_dt = D * lap_rho + eps * rho - lam * (rho ** 3) - eta * pi + nonlocal_term

    return drho_dt, dpi_dt