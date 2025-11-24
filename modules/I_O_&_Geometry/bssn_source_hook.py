"""
MODULE: bssn_source_hook.py
CLASSIFICATION: V11.0 BSSN Interface
GOAL: Hook for mapping FMIA state -> BSSN-style source terms.
CONTRACT ID: IO-BSSN-V11
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
from numerics import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax.numpy as jnp  # type: ignore
else:
    import numpy as jnp  # type: ignore

def get_bssn_source_terms_for_evolution(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    params: Dict[str, Any],
) -> Dict[str, jnp.ndarray]:
    """
    Compute minimal source terms for a 1D BSSN evolution from FMIA state.

    Returns
    -------
    dict with keys:
      - 'rho_energy' (T_00 analog)
      - 'momentum_density' (T_0x analog)
    """
    rho, pi = state

    # Simple toy model: T_00 ~ 0.5 * pi^2 + V(rho), with V = 0.5 rho^2
    V = 0.5 * rho ** 2
    rho_energy = 0.5 * pi ** 2 + V

    # Momentum density ~ rho * pi in 1D
    momentum_density = rho * pi

    # Allow overall rescaling if desired
    energy_scale = float(params.get("energy_scale", 1.0))
    mom_scale = float(params.get("momentum_scale", 1.0))

    return {
        "rho_energy": energy_scale * rho_energy,
        "momentum_density": mom_scale * momentum_density,
    }