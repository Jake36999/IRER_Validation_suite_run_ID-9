"""
worker_sncgl_sdg.py
CLASSIFICATION: Core Physics Worker (IRER V12.0 Gold Master)
GOAL: Executes the coupled S-NCGL/SDG simulation using JAX.
HARDENING: Full S-NCGL Master Equation Support (b3/c3 parameters).
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

# --- JAX CONFIGURATION ---
jax.config.update("jax_enable_x64", True) 
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

# --- 1. JAX State Management ---
class SimState(NamedTuple):
    Psi: jnp.ndarray
    rho_s: jnp.ndarray
    g_mu_nu: jnp.ndarray
    k_sq: jnp.ndarray
    kernel_k: jnp.ndarray

# --- Core Physics ---
@jax.jit
def apply_non_local_term_optimized(psi_field, sncgl_g_nonlocal, kernel_k):
    density = jnp.abs(psi_field) ** 2
    density_k = jnp.fft.fft2(density)
    convolved_density = jnp.real(jnp.fft.ifft2(density_k * kernel_k))
    return sncgl_g_nonlocal * psi_field * convolved_density

# V12 UPDATE: Added sncgl_b3 and sncgl_c3 to signature
@partial(jax.jit, static_argnames=('sncgl_epsilon', 'sncgl_lambda', 'sncgl_b3', 'sncgl_c3', 'sncgl_g_nonlocal', 'sdg_kappa', 'sdg_eta', 'spatial_resolution', 'sdg_alpha', 'sdg_rho_vac', 'dt'))
def _simulation_step_rk4(carry: SimState, _, sncgl_epsilon, sncgl_lambda, sncgl_b3, sncgl_c3, sncgl_g_nonlocal, sdg_kappa, sdg_eta, spatial_resolution, sdg_alpha, sdg_rho_vac, dt):
    state = carry
    Psi, rho_s, g_mu_nu = state.Psi, state.rho_s, state.g_mu_nu

    def compute_dPsi(current_Psi, current_g):
        linear = sncgl_epsilon * current_Psi
        
        # V12 HARDENING: Explicit b3/c3 Non-Linearity
        # Master Eq: -(b3 + i*c3)|A|^2 A
        nonlinear_coeff = sncgl_b3 + 1j * sncgl_c3
        nonlinear = nonlinear_coeff * jnp.abs(current_Psi)**2 * current_Psi * sncgl_lambda
        
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

    # V12 UPDATE: Parameter Extraction with Defaults
    b3_val = float(params.get("sncgl_b3", 1.0))
    c3_val = float(params.get("sncgl_c3", 0.5))

    # Compile & Run
    rk4_step = partial(_simulation_step_rk4,
        sncgl_epsilon=params["sncgl_epsilon"], 
        sncgl_lambda=params["sncgl_lambda"],
        sncgl_b3=b3_val,              # V12
        sncgl_c3=c3_val,              # V12
        sncgl_g_nonlocal=params["sncgl_g_nonlocal"], 
        sdg_kappa=params["sdg_kappa"],
        sdg_eta=params["sdg_eta"], 
        spatial_resolution=grid_size,
        sdg_alpha=params["sdg_alpha"], 
        sdg_rho_vac=rho_vac_val, 
        dt=dt
    )
    
    start_time = time.time()
    final_state, _ = jax.lax.scan(rk4_step, initial_state, None, length=steps)
    final_state.Psi.block_until_ready()
    duration = time.time() - start_time
    
    sse = float(1.0 / (1.0 + 100 * jnp.var(jnp.abs(final_state.Psi)**2)))
    h_norm = float(jnp.sqrt(jnp.mean((final_state.g_mu_nu[0,0,:,:] + 1.0)**2)))

    return duration, sse, h_norm, final_state

def write_results(job_uuid, state, sse, h_norm):
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    path = settings.DATA_DIR / f"rho_history_{job_uuid}.h5"
    
    with h5py.File(path, "w") as f:
        def chunk_shape(shape, dtype, target_bytes=4 * 1024 * 1024):
            if not shape:
                return None
            itemsize = np.dtype(dtype).itemsize
            chunk = list(shape)
            if np.prod(chunk) * itemsize <= target_bytes:
                return tuple(chunk)
            for dim in range(len(chunk) - 1, -1, -1):
                while chunk[dim] > 1 and np.prod(chunk) * itemsize > target_bytes:
                    chunk[dim] = max(1, chunk[dim] // 2)
            return tuple(chunk)

        def write_chunked_dataset(h5_file, name, array):
            shape = tuple(array.shape)
            dtype = array.dtype
            chunks = chunk_shape(shape, dtype)
            dataset = h5_file.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            )
            if not shape:
                dataset[()] = np.array(array)
                return
            chunk_dim0 = chunks[0] if chunks else shape[0]
            for start in range(0, shape[0], chunk_dim0):
                stop = min(start + chunk_dim0, shape[0])
                dataset[start:stop, ...] = np.array(array[start:stop, ...])

        write_chunked_dataset(f, "final_psi", state.Psi)
        write_chunked_dataset(f, "final_rho_s", state.rho_s)
        write_chunked_dataset(f, "final_g_mu_nu", state.g_mu_nu)
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
