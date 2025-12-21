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
from jax.scipy import ndimage
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

def _sample_metric_component(field, coord):
    return ndimage.map_coordinates(field, coord[:, None], order=1, mode="wrap")[0]


def _metric_at_coord(coord, g_ij):
    return jnp.stack(
        [
            jnp.stack(
                [_sample_metric_component(g_ij[..., i, j], coord) for j in range(2)],
                axis=-1,
            )
            for i in range(2)
        ],
        axis=-2,
    )

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

    metric = jnp.moveaxis(g_mu_nu, (0, 1), (2, 3))
    det_g = jnp.linalg.det(metric)
    sqrt_neg_g = jnp.sqrt(jnp.maximum(-det_g, 0.0))

    grid = jnp.stack(
        jnp.meshgrid(jnp.arange(g_ij.shape[0]), jnp.arange(g_ij.shape[1]), indexing="ij"),
        axis=-1,
    )
    g_ij_at = partial(_metric_at_coord, g_ij=g_ij)
    dg = jax.vmap(jax.vmap(jax.jacfwd(g_ij_at)))(grid)
    dg_k = jnp.moveaxis(dg, -1, -3)

    term = jnp.stack(
        [
            jnp.stack(
                [
                    dg_k[..., i, :, j] + dg_k[..., j, :, i] - dg_k[..., :, i, j]
                    for j in range(2)
                ],
                axis=-1,
            )
            for i in range(2)
        ],
        axis=-2,
    )
    Gamma = 0.5 * jnp.einsum("...kl,...lij->...kij", g_inv, term)

    dPsi_dy = jnp.gradient(Psi, dx, axis=0)
    dPsi_dx = jnp.gradient(Psi, dx, axis=1)
    dPsi = jnp.stack([dPsi_dy, dPsi_dx], axis=-1)

    d2Psi_dy2 = jnp.gradient(dPsi_dy, dx, axis=0)
    d2Psi_dx2 = jnp.gradient(dPsi_dx, dx, axis=1)
    d2Psi_dydx = jnp.gradient(dPsi_dy, dx, axis=1)
    d2Psi_dxdy = jnp.gradient(dPsi_dx, dx, axis=0)

    hessian = jnp.stack(
        [
            jnp.stack([d2Psi_dy2, d2Psi_dydx], axis=-1),
            jnp.stack([d2Psi_dxdy, d2Psi_dx2], axis=-1),
        ],
        axis=-2,
    )

    laplace_beltrami = jnp.einsum("...ij,...ij->...", g_inv, hessian) - jnp.einsum(
        "...ij,...kij,...k->...", g_inv, Gamma, dPsi
    )

    return (epsilon * 0.5 + 1j * epsilon * 0.8) * laplace_beltrami
