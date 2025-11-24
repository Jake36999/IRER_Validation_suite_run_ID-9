"""
solver_sdg.py
CLASSIFICATION: V11.0 Geometric Solver (Run ID 14 Gold Master - Hotfixed)
GOAL: JAX-native SDG solver.
FIXES:
  - Removed 'dx'/'omega' from static_argnames (Fixed TypeError)
  - Updated T_info dtype to match Psi (Fixed Precision Warning)
"""
import jax
import jax.numpy as jnp
from functools import partial

# FIX 1: Removed 'omega' and 'dx' from static_argnames. 
# They are floats and should be traced, not hashed.
@partial(jax.jit, static_argnames=('iterations',))
def _jacobi_poisson_solver(source, x, dx, iterations, omega):
    """A JAX-jitted Jacobi-Poisson solver for the SDG geometry."""
    d_sq = dx * dx
    for _ in range(iterations):
        x_new = (
            jnp.roll(x, 1, axis=0) + jnp.roll(x, -1, axis=0) +
            jnp.roll(x, 1, axis=1) + jnp.roll(x, -1, axis=1) +
            source * d_sq
        ) / 4.0
        x = (1.0 - omega) * x + omega * x_new
    return x

@jax.jit
def _compute_spatial_christoffel(g_ij, dx):
    """
    Computes spatial Christoffel symbols Gamma^k_{ij} for a 2D metric g_ij.
    """
    N = g_ij.shape[0]
    
    inv_2x2 = jax.vmap(jax.vmap(jnp.linalg.inv))
    g_inv_ij = inv_2x2(g_ij)

    dg_dx = (jnp.roll(g_ij, -1, axis=0) - jnp.roll(g_ij, 1, axis=0)) / (2 * dx)
    dg_dy = (jnp.roll(g_ij, -1, axis=1) - jnp.roll(g_ij, 1, axis=1)) / (2 * dx)

    def get_dg(k, m, n):
        if k == 0: return dg_dx[:, :, m, n]
        return dg_dy[:, :, m, n]

    Gamma = jnp.zeros((N, N, 2, 2, 2)) 

    for k in range(2):
        for i in range(2):
            for j in range(2):
                term = jnp.zeros((N, N))
                for l in range(2):
                    val = get_dg(i, j, l) + get_dg(j, i, l) - get_dg(l, i, j)
                    term += g_inv_ij[:, :, k, l] * val
                Gamma = Gamma.at[:, :, k, i, j].set(0.5 * term)
    
    return Gamma

@jax.jit
def calculate_informational_stress_energy(Psi, sdg_kappa, sdg_eta):
    rho = jnp.abs(Psi)**2
    phi = jnp.angle(Psi)
    
    grad_phi_y, grad_phi_x = jnp.gradient(phi)
    grad_rho_y, grad_rho_x = jnp.gradient(jnp.sqrt(jnp.maximum(rho, 1e-9)))

    T_00 = (sdg_kappa * rho * (grad_phi_x**2 + grad_phi_y**2) +
            sdg_eta * (grad_rho_x**2 + grad_rho_y**2))
            
    # FIX 2: Dynamic dtype matching (supports float64/complex128)
    T_info = jnp.zeros(Psi.shape + (4, 4), dtype=Psi.dtype)
    # Cast T_00 to complex if needed to avoid warning
    T_info = T_info.at[:, :, 0, 0].set(T_00.astype(Psi.dtype))
    
    return jnp.moveaxis(T_info, (2,3), (0,1))

# FIX 3: Explicitly mark spatial_res as static to ensure dx is constant
@partial(jax.jit, static_argnames=('spatial_res',))
def solve_sdg_geometry(T_info, rho_s, spatial_res, alpha, rho_vac):
    dx = 1.0 / spatial_res
    T_00 = jnp.real(T_info[0, 0])
    
    rho_s_new = _jacobi_poisson_solver(T_00, rho_s, dx, 50, 1.8)
    rho_s_new = jnp.clip(rho_s_new, 1e-6, None)
    
    eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
    scale = (rho_vac / rho_s_new) ** alpha
    g_mu_nu = jnp.einsum('ab,xy->abxy', eta, scale)
    
    return rho_s_new, g_mu_nu

@partial(jax.jit, static_argnames=('spatial_resolution',))
def apply_complex_diffusion(Psi, epsilon, g_mu_nu, spatial_resolution):
    dx = 1.0 / spatial_resolution
    g_ij = jnp.moveaxis(g_mu_nu[1:3, 1:3], (0, 1), (2, 3))
    
    inv_2x2 = jax.vmap(jax.vmap(jnp.linalg.inv))
    g_inv = inv_2x2(g_ij)

    lap_flat = (jnp.roll(Psi, -1, 0) + jnp.roll(Psi, 1, 0) + 
                jnp.roll(Psi, -1, 1) + jnp.roll(Psi, 1, 1) - 4*Psi) / dx**2

    trace_g_inv = g_inv[..., 0, 0] + g_inv[..., 1, 1]
    
    return (epsilon * 0.5 + 1j * epsilon * 0.8) * lap_flat * trace_g_inv