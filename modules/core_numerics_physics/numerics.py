"""
MODULE: numerics.py
CLASSIFICATION: V11.0 Numerical Utilities
GOAL: Generic numerical utilities (RK4, etc.) with optional JAX acceleration.
CONTRACT ID: IO-NUM-V11
"""
from __future__ import annotations
from typing import Any, Callable, Tuple

# Optional JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from jax.tree_util import tree_map
    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - JAX fallback
    import numpy as jnp  # type: ignore
    JAX_AVAILABLE = False

    def jit(fn: Callable) -> Callable:
        return fn

    def tree_map(fn: Callable, tree: Any) -> Any:
        # Very small stand-in for jax.tree_util.tree_map
        if isinstance(tree, (list, tuple)):
            return type(tree)(tree_map(fn, x) for x in tree)
        if isinstance(tree, dict):
            return {k: tree_map(fn, v) for k, v in tree.items()}
        return fn(tree)

def _add_state(a: Any, b: Any):
    """Elementwise addition for states (arrays, tuples, dicts)."""
    return tree_map(lambda x_y: x_y[0] + x_y[1], (a, b))

def _mul_state_scalar(a: Any, scalar: float):
    """Elementwise multiply by scalar for states."""
    return tree_map(lambda x: x * scalar, a)

def rk4_step(
    func: Callable[[Any, float, Any], Any],
    state: Any,
    t: float,
    dt: float,
    *params: Any,
) -> Any:
    """
    Generic 4th-order Runge–Kutta step.

    Parameters
    ----------
    func : callable
        f(state, t, *params) -> dstate_dt with same PyTree structure as `state`.
    state : PyTree
        Current state (array, tuple of arrays, dict of arrays, etc.).
    t : float
        Current time.
    dt : float
        Time step.
    *params :
        Extra parameters passed to `func`.

    Returns
    -------
    new_state : PyTree
        Updated state at t + dt.
    """
    k1 = func(state, t, *params)
    k2 = func(_add_state(state, _mul_state_scalar(k1, 0.5 * dt)), t + 0.5 * dt, *params)
    k3 = func(_add_state(state, _mul_state_scalar(k2, 0.5 * dt)), t + 0.5 * dt, *params)
    k4 = func(_add_state(state, _mul_state_scalar(k3, dt)), t + dt, *params)

    incr = _mul_state_scalar(
        _add_state(
            _add_state(k1, _mul_state_scalar(k2, 2.0)),
            _add_state(_mul_state_scalar(k3, 2.0), k4),
        ),
        dt / 6.0,
    )
    return _add_state(state, incr)