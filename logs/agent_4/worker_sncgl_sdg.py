"""
MODULE: worker_sncgl_sdg.py
CLASSIFICATION: Core Physics Worker (IRER V11.0)
GOAL: Executes the coupled S-NCGL/SDG simulation using JAX with RK4 integration.
CONTRACT ID: IO-WRK-V11
HASHING MANDATE: Variant A (Deterministic SHA1)
"""
import jax
import jax.numpy as jnp
import numpy as np
import json
import argparse
import os
import h5py
import time
import sys
from functools import partial
from typing import NamedTuple
import jsonschema

# --- JAX CONFIGURATION FOR NUMERICAL STABILITY (HARDENING) ---
# Enforce 64-bit precision for complex geometric/field calculations
jax.config.update("jax_enable_x64", True) 
# Disable NaN checking in production to avoid performance hit/false positives
jax.config.update("jax_debug_nans", False)

try:
    import settings
except ImportError:
    print("FATAL: 'settings.py' not found.", file=sys.stderr)
    sys.exit(1)

from solver_sdg import (
    calculate_informational_stress_energy,
    solve_sdg_geometry,
    apply_complex_diffusion,
)

# --- 1. JAX HPC Mandate: Explicit State Management ---
class SimState(NamedTuple):
    Psi: jnp.ndarray
    rho_s: jnp.ndarray
    g_mu_nu: jnp.ndarray
    k_sq: jnp.ndarray
    kernel_k: jnp.ndarray

# --- 2. Input Validation Logic (V11 Mandate) ---
def validate_config(params: dict):
    """Validates input parameters against the V11 Schema."""
    # In a full deployment, load this from 30_schemas/config_schema_v11.json
    # Here we define the critical constraints inline for portability if file is missing
    schema = {
        "type": "object",
        "required": ["grid_size", "t_steps", "dt"], 
        "properties": {
            "grid_size": {"type": "integer", "minimum": 64},
            "t_steps": {"type": "integer", "minimum": 1},
            "dt": {"type": "number", "exclusiveMinimum": 0}
        }
    }
    # Mapping backend keys to schema keys
    validation_obj = {
        "grid_size": params.get("N_grid", 64),
        "t_steps": params.get("T_steps", 200),
        "dt": params.get("dt", 0.01)
    }
    try:
        jsonschema.validate(instance=validation_obj, schema=schema)
    except jsonschema.ValidationError as e:
        print(f"FATAL: Input Configuration Validation Failed: {e}", file=sys.stderr)
        sys.exit(int(settings.SENTINEL_ARTIFACT_MISSING)) # Exit with failure code

# --- Core Physics ---
@jax.jit
def apply_non_local_term_optimized(psi_field, sncgl_g_nonlocal, kernel_k):
    density = jnp.abs(psi_field) ** 2
    density_k = jnp.fft.fft2(density)
    convolved_density = jnp.real(jnp.fft.ifft2(density_k * kernel_k))
    return sncgl_g_nonlocal * psi_field * convolved_density

@partial(jax.jit, static_argnames=('sncgl_epsilon', 'sncgl_lambda', 'sncgl_g_nonlocal', 'sdg_kappa', 'sdg_eta', 'spatial_resolution', 'sdg_alpha', 'sdg_rho_vac', 'dt'))
def _simulation_step_rk4(carry: SimState, _, sncgl_epsilon, sncgl_lambda, sncgl_g_nonlocal, sdg_kappa, sdg_eta, spatial_resolution, sdg_alpha, sdg_rho_vac, dt):
    state = carry
    Psi, rho_s, g_mu_nu = state.Psi, state.rho_s, state.g_mu_nu

    def compute_dPsi(current_Psi, current_g):
        linear = sncgl_epsilon * current_Psi
        nonlinear = (1.0 + 0.5j) * jnp.abs(current_Psi)**2 * current_Psi * sncgl_lambda
        diffusion = apply_complex_diffusion(current_Psi, sncgl_epsilon, current_g, spatial_resolution)
        nl_term = apply_non_local_term_optimized(current_Psi, sncgl_g_nonlocal, state.kernel_k)
        return linear + diffusion - nonlinear - nl_term

    # RK4 Integration
    k1 = compute_dPsi(Psi, g_mu_nu)
    k2 = compute_dPsi(Psi + 0.5 * dt * k1, g_mu_nu)
    k3 = compute_dPsi(Psi + 0.5 * dt * k2, g_mu_nu)
    k4 = compute_dPsi(Psi + dt * k3, g_mu_nu)
    new_Psi = Psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Geometric Feedback
    T_info = calculate_informational_stress_energy(new_Psi, sdg_kappa, sdg_eta)
    new_rho_s, new_g_mu_nu = solve_sdg_geometry(T_info, rho_s, spatial_resolution, sdg_alpha, sdg_rho_vac)

    return SimState(new_Psi, new_rho_s, new_g_mu_nu, state.k_sq, state.kernel_k), None

def run_simulation(params_path: str):
    with open(params_path, "r") as f:
        params = json.load(f)

    # V11 VALIDATION STEP
    validate_config(params)

    grid_size = int(params.get("N_grid", 64))
    steps = int(params.get("T_steps", 200))
    dt = params.get("dt", 0.01)
    
    # Initialization
    key = jax.random.PRNGKey(params.get("seed", 42))
    k1, k2 = jax.random.split(key)
    Psi = (jax.random.normal(k1, (grid_size, grid_size)) + 1j * jax.random.normal(k2, (grid_size, grid_size))) * 0.1
    
    rho_vac_val = params.get("sdg_rho_vac", 1.0)
    rho_s = jnp.ones((grid_size, grid_size)) * rho_vac_val
    
    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    g_init = jnp.tile(eta[:, :, None, None], (1, 1, grid_size, grid_size))

    # Pre-compute Kernels
    kx = jnp.fft.fftfreq(grid_size)
    ky = jnp.fft.fftfreq(grid_size)
    kx_g, ky_g = jnp.meshgrid(kx, ky, indexing="ij")
    k_sq = kx_g**2 + ky_g**2
    kernel_k = jnp.exp(-k_sq / (2.0 * (1.5**2)))

    initial_state = SimState(Psi, rho_s, g_init, k_sq, kernel_k)

    # Compile & Run
    rk4_step = partial(_simulation_step_rk4,
        sncgl_epsilon=params.get("sncgl_epsilon", 0.1), 
        sncgl_lambda=params.get("sncgl_lambda", 0.1),
        sncgl_g_nonlocal=params.get("sncgl_g_nonlocal", 0.0), 
        sdg_kappa=params.get("sdg_kappa", 0.1),
        sdg_eta=params.get("sdg_eta", 0.1), 
        spatial_resolution=grid_size,
        sdg_alpha=params.get("sdg_alpha", 1.0), 
        sdg_rho_vac=rho_vac_val, 
        dt=dt
    )
    
    start_time = time.time()
    try:
        final_state, _ = jax.lax.scan(rk4_step, initial_state, None, length=steps)
        final_state.Psi.block_until_ready()
    except Exception as e:
        print(f"FATAL: Physics Divergence or Runtime Error: {e}", file=sys.stderr)
        sys.exit(int(settings.SENTINEL_SCIENTIFIC_FAILURE))

    duration = time.time() - start_time
    
    # Simple Metrics for Log (Not used for final validation, just local check)
    sse = float(1.0 / (1.0 + 100 * jnp.var(jnp.abs(final_state.Psi)**2)))
    h_norm = float(jnp.sqrt(jnp.mean((final_state.g_mu_nu[0,0,:,:] + 1.0)**2)))

    return duration, sse, h_norm, final_state

def write_results(job_uuid, state, sse, h_norm):
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    path = settings.DATA_DIR / f"rho_history_{job_uuid}.h5"
    
    with h5py.File(path, "w") as f:
        f.create_dataset("final_psi", data=np.array(state.Psi))
        f.create_dataset("final_rho_s", data=np.array(state.rho_s))
        f.create_dataset("final_g_mu_nu", data=np.array(state.g_mu_nu))
        f.attrs[settings.SSE_METRIC_KEY] = sse
        f.attrs[settings.STABILITY_METRIC_KEY] = h_norm
    print(f"[Worker] Artifact saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--job_uuid", required=True)
    args = parser.parse_args()
    
    dur, sse, h_norm, final_state = run_simulation(args.config_path)
    print(f"[Worker] Done ({dur:.2f}s)")
    write_results(args.job_uuid, final_state, sse, h_norm)

# --- V11.0 END OF FILE ---
# INTEGRITY CHECK: SHA1_PLACEHOLDER
# SENTINEL REFERENCE: 999.0, 1002.0