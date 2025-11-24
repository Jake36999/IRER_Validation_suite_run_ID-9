"""
MODULE: geometry_metric.py
CLASSIFICATION: V11.0 Geometry Constructor
GOAL: Conformal metric construction utilities.
      Implements g_mu_nu(x) = (rho_vac / rho(x))**alpha * eta_mu_nu
CONTRACT ID: IO-GEO-V11
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

def construct_conformal_metric(
    rho: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Construct a conformally flat 4D metric field from a scalar density rho(x).

    Parameters
    ----------
    rho : np.ndarray
        Scalar field (any shape, e.g. (N,) or (Nx, Ny)).
    params : dict
        Must contain:
            'rho_vac' : reference density
            'alpha'   : exponent for conformal factor

    Returns
    -------
    g : np.ndarray
        Metric tensor field with shape (4, 4, *rho.shape)
        where g[0,0] = -Omega, g[i,i] = +Omega for i=1..3, off-diagonals zero.
    """
    rho_vac = float(params.get("rho_vac", 1.0))
    alpha = float(params.get("alpha", 1.0))

    rho_reg = np.maximum(rho, 1e-8)
    Omega = (rho_vac / rho_reg) ** alpha  # conformal factor

    # Broadcast Omega to metric shape
    metric_shape = (4, 4) + rho.shape
    g = np.zeros(metric_shape, dtype=float)

    g[0, 0, ...] = -Omega
    for i in range(1, 4):
        g[i, i, ...] = Omega

    return g