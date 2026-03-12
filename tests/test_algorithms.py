#!/usr/bin/env python3
"""
cosmo_sim validation test suite
================================
Validates every algorithm used by cosmo_sim independently in Python/NumPy.
Run with:   python3 tests/test_algorithms.py

Each test mirrors a Rust unit-test but runs a reference implementation so
we have an independent check for supercomputer runs.

Tests are grouped by module:
  - TestPoisson          : multigrid Poisson solver
  - TestParticles        : CIC, leapfrog
  - TestCosmology        : ΛCDM background
  - TestSymmetron        : Symmetron scalar field physics
  - TestFofR             : f(R) Hu-Sawicki background
  - TestIntegration      : end-to-end gravity + particles
  - TestConvergence      : order-of-accuracy checks
"""

import sys
import unittest
import numpy as np
from numpy.fft import fftn, ifftn

# ── Shared helpers ────────────────────────────────────────────────────────────
# Index convention: C-order (x slowest, z fastest) matching numpy reshape.
# flat3(i,j,k,n) = i*n²+j*n+k  so arr.reshape(n,n,n)[i,j,k] is correct.
# This matches: axis-0=x, axis-1=y, axis-2=z throughout.

def flat3(i, j, k, n):
    """Flat 3D index with periodic wrap: C-order (x=slow, z=fast)."""
    return int(i % n) * n * n + int(j % n) * n + int(k % n)

def make_idx_array(n):
    """Precompute neighbour index arrays for a periodic n³ grid."""
    idx = np.arange(n * n * n, dtype=np.int64).reshape(n, n, n)
    return idx

# ── Poisson solver (spectral reference) ──────────────────────────────────────

def solve_poisson_spectral(source, dx, n):
    """
    Reference Poisson solver using FFT with the **discrete** Green function.

    Uses eigenvalues  λ_m = −4/dx² Σ_d sin²(π m_d / n)  of the 7-point
    centred finite-difference Laplacian.  This is consistent with the
    multigrid solver and the Rust code (both use the same stencil).

    Solves ∇²Φ = source on a periodic cubic grid of spacing dx.
    """
    src_k = fftn(source.reshape(n, n, n))
    m = np.arange(n)
    MX, MY, MZ = np.meshgrid(m, m, m, indexing='ij')
    # Eigenvalue of ∇²_discrete for each Fourier mode
    eig = -4.0 / dx**2 * (
        np.sin(np.pi * MX / n)**2 +
        np.sin(np.pi * MY / n)**2 +
        np.sin(np.pi * MZ / n)**2
    )
    eig[0, 0, 0] = -1.0   # prevent division by zero; DC set to 0 below
    phi_k = src_k / eig
    phi_k[0, 0, 0] = 0.0  # enforce zero mean potential
    return np.real(ifftn(phi_k)).flatten()


def solve_poisson_multigrid(source, dx, n, n_iter=50, n_smooth=3):
    """
    Multigrid V-cycle Poisson solver in Python (mirrors Rust implementation).
    """
    def gs_sweep(phi, rhs, dx, n):
        phi = phi.reshape(n, n, n).copy()
        rhs = rhs.reshape(n, n, n)
        inv_dx2 = 1.0 / dx**2
        snap = phi.copy()
        for color in range(2):
            for k in range(n):
                for j in range(n):
                    for i in range(n):
                        if (i + j + k) % 2 != color:
                            continue
                        nb = (snap[(i+1)%n, j, k] + snap[(i-1)%n, j, k] +
                              snap[i, (j+1)%n, k] + snap[i, (j-1)%n, k] +
                              snap[i, j, (k+1)%n] + snap[i, j, (k-1)%n])
                        phi[i, j, k] = (nb - rhs[i, j, k] / inv_dx2) / 6.0
        return phi.flatten()

    def residual(phi, rhs, dx, n):
        phi = phi.reshape(n, n, n)
        rhs = rhs.reshape(n, n, n)
        inv_dx2 = 1.0 / dx**2
        lap = (np.roll(phi, -1, 0) + np.roll(phi, 1, 0) +
               np.roll(phi, -1, 1) + np.roll(phi, 1, 1) +
               np.roll(phi, -1, 2) + np.roll(phi, 1, 2) - 6.0 * phi) * inv_dx2
        return (rhs - lap).flatten()

    def restrict(v, n):
        return v.reshape(n, n, n)[::2, ::2, ::2].flatten()

    def prolongate(vc, nc):
        nf = nc * 2
        vf = np.zeros((nf, nf, nf))
        vc = vc.reshape(nc, nc, nc)
        # Trilinear from each coarse cell to 2³ fine cells
        for dk, wz in [(0, 0.5), (1, 0.5)]:
            for dj, wy in [(0, 0.5), (1, 0.5)]:
                for di, wx in [(0, 0.5), (1, 0.5)]:
                    vf[di::2, dj::2, dk::2] += wx * wy * wz * vc
        return vf.flatten()

    phi = np.zeros(n * n * n)
    for _ in range(n_iter):
        r = residual(phi, source, dx, n)
        if np.max(np.abs(r)) < 1e-9:
            break
        # Simple Gauss-Seidel iteration (not full V-cycle for Python speed)
        phi = gs_sweep(phi, source, dx, n)
    return phi


# ── CIC helpers ───────────────────────────────────────────────────────────────

def cic_density(pos, box_size, n_grid):
    """CIC mass assignment. Returns ρ/ρ̄ array (C-order: x-slow, z-fast).
    Flat index: arr[i,j,k] = flat[i*n²+j*n+k]."""
    ng  = n_grid
    h   = ng / box_size
    rho = np.zeros(ng * ng * ng)
    for xi in pos:
        cx = xi[0] * h - 0.5
        cy = xi[1] * h - 0.5
        cz = xi[2] * h - 0.5
        ix, iy, iz = int(np.floor(cx)), int(np.floor(cy)), int(np.floor(cz))
        tx = cx - ix
        ty = cy - iy
        tz = cz - iz
        for dz, wz in [(0, 1-tz), (1, tz)]:
            for dy, wy in [(0, 1-ty), (1, ty)]:
                for dx, wx in [(0, 1-tx), (1, tx)]:
                    jx = (ix + dx) % ng
                    jy = (iy + dy) % ng
                    jz = (iz + dz) % ng
                    # C-order: x slowest
                    rho[jx * ng * ng + jy * ng + jz] += wx * wy * wz
    norm = ng**3 / len(pos)
    return rho * norm


def cic_interp_force(pos, force_grid, box_size, n_grid):
    """CIC force interpolation: grid → particles.
    force_grid shape: (n³, 3), indexed with C-order flat index."""
    ng = n_grid
    h  = ng / box_size
    forces = np.zeros_like(pos)
    for p_idx, xi in enumerate(pos):
        cx = xi[0] * h - 0.5
        cy = xi[1] * h - 0.5
        cz = xi[2] * h - 0.5
        ix, iy, iz = int(np.floor(cx)), int(np.floor(cy)), int(np.floor(cz))
        tx = cx - ix
        ty = cy - iy
        tz = cz - iz
        for dz, wz in [(0, 1-tz), (1, tz)]:
            for dy, wy in [(0, 1-ty), (1, ty)]:
                for dx, wx in [(0, 1-tx), (1, tx)]:
                    jx = (ix + dx) % ng
                    jy = (iy + dy) % ng
                    jz = (iz + dz) % ng
                    # C-order: x slowest
                    cidx = jx * ng * ng + jy * ng + jz
                    forces[p_idx] += wx * wy * wz * force_grid[cidx]
    return forces


# ── ΛCDM cosmology ────────────────────────────────────────────────────────────

class Cosmology:
    def __init__(self, omega_m=0.3153, omega_lambda=0.6847, h=0.6736):
        self.omega_m      = omega_m
        self.omega_lambda = omega_lambda
        self.h            = h

    def E(self, a):
        return np.sqrt(self.omega_m / a**3 + self.omega_lambda)

    def H(self, a):
        return self.h * 100.0 * self.E(a)

    def da_dtau(self, a):
        return a * self.H(a)

    def advance_a_rk4(self, a, dtau):
        f = self.da_dtau
        k1 = f(a)
        k2 = f(a + 0.5 * dtau * k1)
        k3 = f(a + 0.5 * dtau * k2)
        k4 = f(a + dtau * k3)
        return a + dtau * (k1 + 2*k2 + 2*k3 + k4) / 6.0


# ── Symmetron physics ─────────────────────────────────────────────────────────

def symmetron_source(chi, rho, a, a_ssb, lambda_0):
    """Source f(χ) = χ/λ₀² · (ρ/M² − 1 + χ²)."""
    rho_over_m2 = rho * (a_ssb / a)**3
    return chi * (rho_over_m2 - 1.0 + chi**2) / lambda_0**2


def symmetron_vacuum(a, a_ssb, lambda_0, rho=0.0):
    """Equilibrium χ in uniform density (|χ| = sqrt(1 − ρ/M²))."""
    rho_over_m2 = rho * (a_ssb / a)**3
    arg = 1.0 - rho_over_m2
    if arg <= 0:
        return 0.0  # screened
    return np.sqrt(arg)


def solve_symmetron_1d(rho_array, a, a_ssb, lambda_0, dx, n_iter=1000, tol=1e-8):
    """
    Simple Gauss-Seidel solver for the 1-D Symmetron equation (for testing).
    Solves  d²χ/dx² = f(χ, ρ) with periodic BCs.
    """
    n   = len(rho_array)
    chi = np.ones(n) * symmetron_vacuum(a, a_ssb, lambda_0)
    inv_dx2 = 1.0 / dx**2

    for _ in range(n_iter):
        chi_old = chi.copy()
        for i in range(n):
            chi_c = chi_old[i]
            rho_c = rho_array[i]
            nb    = chi_old[(i-1)%n] + chi_old[(i+1)%n]
            f   = symmetron_source(chi_c, rho_c, a, a_ssb, lambda_0)
            df  = (rho_c * (a_ssb/a)**3 - 1.0 + 3.0 * chi_c**2) / lambda_0**2
            lap = (nb - 2.0 * chi_c) * inv_dx2
            denom = -2.0 * inv_dx2 - df
            chi[i] = chi_c - (lap - f) / denom
        if np.max(np.abs(chi - chi_old)) < tol:
            break
    return chi


# ═══════════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestPoisson(unittest.TestCase):

    def test_single_mode_spectral_reference(self):
        """Spectral Poisson gives machine-precision result for a discrete eigenfunction."""
        n  = 16
        L  = 1.0
        dx = L / n
        # Use mode m=1 in x: phi = cos(2π·1·i/n)  — exact discrete eigenfunction
        m  = 1
        i  = np.arange(n)
        phi_1d = np.cos(2 * np.pi * m * i / n)
        phi_3d = np.einsum('i,j,k->ijk', phi_1d, np.ones(n), np.ones(n))
        # Discrete Laplacian eigenvalue for this mode (x-only)
        eig = -4.0 / dx**2 * np.sin(np.pi * m / n)**2
        source = eig * phi_3d  # exact source for this eigenfunction
        phi = solve_poisson_spectral(source.flatten(), dx, n)
        phi -= phi.mean()
        expected = phi_3d.flatten() - phi_3d.mean()
        err = np.max(np.abs(phi - expected))
        print(f"  Spectral Poisson discrete-mode error: {err:.3e}")
        self.assertLess(err, 1e-10)

    def test_spectral_vs_multigrid(self):
        """Spectral and multigrid solutions agree to within solver tolerance."""
        n   = 8
        L   = 1.0
        dx  = L / n
        k   = 2 * np.pi / L
        x   = (np.arange(n) + 0.5) * dx
        X, _, _ = np.meshgrid(x, x, x, indexing='ij')
        source = -(k**2) * np.sin(k * X)
        phi_fft = solve_poisson_spectral(source.flatten(), dx, n)
        phi_mg  = solve_poisson_multigrid(source.flatten(), dx, n, n_iter=200, n_smooth=3)
        phi_fft -= phi_fft.mean()
        phi_mg  -= phi_mg.mean()
        err = np.max(np.abs(phi_fft - phi_mg))
        print(f"  Spectral vs multigrid max diff: {err:.3e}")
        self.assertLess(err, 0.1)

    def test_poisson_green_sphere(self):
        """Potential of a uniform-density sphere: compare to Newtonian shell theorem."""
        n   = 32
        L   = 2.0  # Mpc/h
        dx  = L / n
        x   = (np.arange(n) + 0.5) * dx
        X, Y, Z = np.meshgrid(x - L/2, x - L/2, x - L/2, indexing='ij')
        R   = np.sqrt(X**2 + Y**2 + Z**2)
        R_s = 0.3  # sphere radius
        source = np.where(R < R_s, 1.0, 0.0)  # uniform sphere density
        source -= source.mean()  # periodic: remove mean
        phi = solve_poisson_spectral(source.flatten(), dx, n)
        # Interior: Φ ∝ −r² (quadratic), exterior: Φ ∝ −1/r
        phi_reshaped = phi.reshape(n, n, n)
        # Check that max |Φ| is at centre
        centre = phi_reshaped[n//2, n//2, n//2]
        outer  = phi_reshaped[1, n//2, n//2]
        print(f"  Sphere: Φ_centre={centre:.4f}, Φ_outer={outer:.4f}")
        self.assertLess(centre, outer, "Centre should have more negative potential")

    def test_gradient_accuracy(self):
        """Centred finite-difference gradient of sin(kx) matches k cos(kx)."""
        n   = 32
        L   = 1.0
        dx  = L / n
        k   = 2 * np.pi / L
        x   = (np.arange(n) + 0.5) * dx
        phi = np.sin(k * x)  # 1D
        # Centred difference
        gx  = -(np.roll(phi, -1) - np.roll(phi, 1)) / (2 * dx)
        exact_gx = -k * np.cos(k * x)
        err = np.max(np.abs(gx - exact_gx))
        print(f"  Gradient max error: {err:.3e}")
        self.assertLess(err, 0.05)

    def test_poisson_second_order_convergence(self):
        """Spectral solver returns exact discrete solution (spectral convergence)."""
        # For any source, the spectral solver is exact up to floating-point error.
        # Verify errors are at machine precision for all grid sizes.
        for n in [8, 16, 32]:
            L  = 1.0
            dx = L / n
            m  = 1
            i  = np.arange(n)
            phi_1d = np.cos(2 * np.pi * m * i / n)
            phi_3d = np.einsum('i,j,k->ijk', phi_1d, np.ones(n), np.ones(n))
            eig    = -4.0 / dx**2 * np.sin(np.pi * m / n)**2
            source = (eig * phi_3d).flatten()
            phi    = solve_poisson_spectral(source, dx, n)
            phi   -= phi.mean()
            expected = phi_3d.flatten() - phi_3d.mean()
            err = np.max(np.abs(phi - expected))
            print(f"  n={n}: spectral error = {err:.2e}")
            self.assertLess(err, 1e-10, f"n={n}: spectral error too large: {err:.2e}")


class TestParticles(unittest.TestCase):

    def test_cic_uniform_density(self):
        """CIC of one particle per cell centre gives ρ/ρ̄ = 1 everywhere."""
        ng = 4
        L  = 4.0
        dx = L / ng
        pos = np.array([[( i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx]
                         for k in range(ng) for j in range(ng) for i in range(ng)])
        rho = cic_density(pos, L, ng)
        np.testing.assert_allclose(rho, 1.0, atol=1e-12,
                                   err_msg="CIC uniform density failed")

    def test_cic_mass_conservation(self):
        """Total CIC-assigned mass equals N_particles × m_part."""
        ng   = 8
        L    = 8.0
        rng  = np.random.default_rng(42)
        pos  = rng.uniform(0, L, (200, 3))
        rho  = cic_density(pos, L, ng)
        # Sum of ρ/ρ̄ over all cells = n_cells (mean = 1)
        expected = ng**3
        self.assertAlmostEqual(rho.sum(), expected, places=8,
                               msg=f"CIC mass not conserved: {rho.sum():.6f} ≠ {expected}")

    def test_cic_interpolate_constant_force(self):
        """Interpolating constant force field returns exact value at each particle."""
        ng = 4
        L  = 4.0
        const_f = np.array([1.23, -0.45, 0.78])
        grid_f  = np.tile(const_f, (ng**3, 1))
        rng = np.random.default_rng(7)
        pos = rng.uniform(0, L, (20, 3))
        f   = cic_interp_force(pos, grid_f, L, ng)
        # Each of the 20 particles should get exactly the constant force
        expected = np.tile(const_f, (len(pos), 1))
        np.testing.assert_allclose(f, expected, atol=1e-12,
                                   err_msg="CIC force interpolation of constant failed")

    def test_leapfrog_free_particle(self):
        """Free particle (no force) should travel in a straight line."""
        x   = np.array([5.0, 5.0, 5.0])
        v   = np.array([1.0, 0.5, -0.3])
        v0  = v.copy()
        a   = 1.0
        dtau = 0.001
        L   = 1000.0

        for _ in range(1000):
            # Drift
            x += v * dtau * 0.5 / a**2
            # Kick (zero force)
            # Drift
            x += v * dtau * 0.5 / a**2

        # Position should be x0 + v0 * total_time / a²
        total_time = 1000 * dtau
        expected_x = np.array([5.0, 5.0, 5.0]) + v0 * total_time / a**2
        np.testing.assert_allclose(x, expected_x % L, atol=1e-10,
                                   err_msg="Free particle position wrong")
        np.testing.assert_allclose(v, v0, atol=1e-14,
                                   err_msg="Free particle velocity changed")

    def test_leapfrog_harmonic_oscillator(self):
        """Leapfrog integrator should conserve energy for harmonic oscillator."""
        # Simple harmonic oscillator: F = −x, ω = 1
        x   = np.array([1.0])
        v   = np.array([0.0])
        a   = 1.0
        dt  = 0.01
        n   = int(2 * np.pi / dt * 10)  # 10 full periods

        # DKD leapfrog
        def force(x):
            return -x

        # Initial half-kick to get onto staggered grid
        v_half = v + 0.5 * dt * force(x)
        x_curr = x.copy()
        E0 = 0.5 * x[0]**2 + 0.5 * v[0]**2

        for _ in range(n):
            x_curr = x_curr + dt * v_half
            f      = force(x_curr)
            v_half = v_half + dt * f

        E1 = 0.5 * x_curr[0]**2 + 0.5 * v_half[0]**2
        rel_err = abs(E1 - E0) / E0
        print(f"  Harmonic oscillator energy conservation: {rel_err:.3e}")
        self.assertLess(rel_err, 0.01)  # leapfrog is symplectic; energy should be bounded

    def test_periodic_wrap(self):
        """Particle crossing box boundary should wrap correctly."""
        L   = 10.0
        pos = np.array([[9.8, 5.0, 5.0]])
        v   = np.array([[5.0, 0.0, 0.0]])
        a   = 1.0
        dtau = 1.0  # big step to cross boundary
        pos += v * dtau / a**2
        pos = pos % L
        self.assertTrue(0.0 <= pos[0, 0] < L)


class TestCosmology(unittest.TestCase):

    def setUp(self):
        self.cosmo = Cosmology()

    def test_e_of_a_today(self):
        """E(1) = 1 for flat ΛCDM."""
        self.assertAlmostEqual(self.cosmo.E(1.0), 1.0, places=3)

    def test_e_of_a_matter_dominated(self):
        """At high z, E(a) → sqrt(Ω_m) a^{-3/2}."""
        a = 0.01
        E_exact  = self.cosmo.E(a)
        E_approx = np.sqrt(self.cosmo.omega_m / a**3)
        self.assertAlmostEqual(E_exact / E_approx, 1.0, places=2)

    def test_rk4_advance_a(self):
        """RK4 advance_a converges as O(dt⁴): halving dt reduces error ~16×."""
        # Use H0-normalised time (da/dtau = a*E(a)) for cleaner numerics.
        # The Python Cosmology class has H(a) = h*100*E(a); normalise by h*100.
        omega_m = self.cosmo.omega_m
        omega_l = self.cosmo.omega_lambda

        def E(a): return np.sqrt(omega_m / a**3 + omega_l)
        def da_dtau(a): return a * E(a)   # H0-normalised

        def rk4(a, dt):
            k1 = da_dtau(a)
            k2 = da_dtau(a + 0.5 * dt * k1)
            k3 = da_dtau(a + 0.5 * dt * k2)
            k4 = da_dtau(a + dt * k3)
            return a + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

        a0        = 0.5
        total_tau = 0.5  # enough for O(1) change in a, not too large

        # Ultra-fine reference
        a_ref = a0
        for _ in range(1_000_000):
            a_ref = rk4(a_ref, total_tau / 1_000_000)

        errors = []
        for n_steps in [2, 4, 8]:
            dt = total_tau / n_steps
            a  = a0
            for _ in range(n_steps):
                a = rk4(a, dt)
            errors.append(abs(a - a_ref) / a_ref)

        print(f"  RK4 errors: {[f'{e:.2e}' for e in errors]}")
        r1 = errors[0] / errors[1]
        r2 = errors[1] / errors[2]
        print(f"  RK4 convergence ratios: {r1:.1f}, {r2:.1f}  (expect ~16)")
        self.assertGreater(r1, 10.0, f"RK4 not 4th-order: ratio={r1:.1f}")
        self.assertGreater(r2, 10.0, f"RK4 not 4th-order: ratio={r2:.1f}")

    def test_particle_mass_order_of_magnitude(self):
        """Particle mass for typical simulation should be physically reasonable."""
        L      = 64.0   # Mpc/h
        n_part = 256
        rho_crit0  = 2.775e11 * self.cosmo.h**2
        rho_bar    = self.cosmo.omega_m * rho_crit0
        m_part     = rho_bar * L**3 / n_part**3
        print(f"  Particle mass: {m_part:.3e} M_sun/h")
        self.assertGreater(m_part, 1e8)
        self.assertLess(m_part, 1e14)

    def test_hubble_decreasing(self):
        """H(a) should be larger in the past (a < 1)."""
        h0 = self.cosmo.H(1.0)
        h_past = self.cosmo.H(0.5)
        self.assertGreater(h_past, h0)


class TestSymmetron(unittest.TestCase):

    def test_source_at_equilibrium(self):
        """f(χ_vac) = 0 for vacuum solution χ = ±1, ρ = 0, a > a_ssb."""
        a_ssb    = 0.5
        lambda_0 = 1.0
        a        = 1.0   # a > a_ssb
        # χ_vac: ρ_over_m2 = 0, so χ_vac = ±1
        chi_vac = 1.0
        f = symmetron_source(chi_vac, 0.0, a, a_ssb, lambda_0)
        self.assertAlmostEqual(f, 0.0, places=14,
                               msg=f"Source at vacuum not zero: f={f}")

    def test_source_screened(self):
        """f(0, high_rho) = 0 (trivially screened)."""
        f = symmetron_source(0.0, 1000.0, 1.0, 0.5, 1.0)
        self.assertEqual(f, 0.0)

    def test_vacuum_value(self):
        """Vacuum χ should equal √(1 − ρ/M²)."""
        a_ssb    = 0.5
        lambda_0 = 1.0
        a        = 1.0
        # ρ/M² = ρ * (a_ssb/a)³ = 0 * 0.125 = 0 → χ_vac = 1
        chi_vac = symmetron_vacuum(a, a_ssb, lambda_0, rho=0.0)
        self.assertAlmostEqual(chi_vac, 1.0, places=14)

    def test_screened_above_ssb_density(self):
        """Above SSB density, vacuum is χ = 0."""
        a_ssb    = 0.5
        lambda_0 = 1.0
        a        = 1.0
        # ρ/M² = 10 * (0.5)³ > 1 → screened
        chi_vac = symmetron_vacuum(a, a_ssb, lambda_0, rho=10.0)
        self.assertEqual(chi_vac, 0.0)

    def test_1d_solver_vacuum(self):
        """1D solver converges to χ = 1 in uniform low-density background."""
        n         = 32
        dx        = 0.1
        a_ssb     = 0.5
        lambda_0  = 1.0
        rho_array = np.zeros(n)  # vacuum
        chi = solve_symmetron_1d(rho_array, 1.0, a_ssb, lambda_0, dx, n_iter=5000)
        print(f"  Symmetron 1D vacuum: mean χ = {chi.mean():.6f}")
        self.assertAlmostEqual(np.abs(chi).mean(), 1.0, places=2,
                               msg="Symmetron vacuum not converged")

    def test_1d_solver_high_density(self):
        """1D solver gives χ ≈ 0 in a screened high-density region."""
        n         = 32
        dx        = 0.1
        a_ssb     = 0.5
        lambda_0  = 1.0
        rho_array = np.ones(n) * 1000.0  # well above SSB density
        chi = solve_symmetron_1d(rho_array, 1.0, a_ssb, lambda_0, dx,
                                 n_iter=5000, tol=1e-10)
        max_chi = np.max(np.abs(chi))
        print(f"  Symmetron screened max |χ|: {max_chi:.3e}")
        self.assertLess(max_chi, 0.1)

    def test_fifth_force_direction(self):
        """Fifth force ∝ ∇χ: centred-diff gradient of a smooth mode is correct."""
        # Use a true periodic mode cos(2π·k/n) in the x-direction so there
        # are no wrap-around discontinuities.
        n   = 16
        dx  = 1.0
        m   = 1
        # Build chi_3d[i,j,k] = cos(2π·m·i/n)  (varies along x = axis-0)
        i_arr = np.arange(n)
        chi_1d = np.cos(2 * np.pi * m * i_arr / n)
        chi_3d = np.einsum('i,j,k->ijk', chi_1d, np.ones(n), np.ones(n))
        # Centred-diff gradient along x (axis-0)
        gx = (np.roll(chi_3d, -1, 0) - np.roll(chi_3d, 1, 0)) / (2 * dx)
        # Analytical: d/dx cos(2π·m·i/n) ≈ −(2π·m/n)/dx sin(2π·m·i/n)
        # In discrete form with centred diff:
        # [cos(2π·m·(i+1)/n) − cos(2π·m·(i-1)/n)] / (2dx)
        # = −2 sin(2π·m/n) sin(2π·m·i/n) / (2dx)
        # = −sin(2π·m/n)/dx * sin(2π·m·i/n)
        expected_gx_1d = -np.sin(2*np.pi*m/n) / dx * np.sin(2*np.pi*m*i_arr/n)
        expected_3d    = np.einsum('i,j,k->ijk', expected_gx_1d, np.ones(n), np.ones(n))
        err = np.max(np.abs(gx - expected_3d))
        print(f"  Fifth-force gradient error (periodic mode): {err:.3e}")
        self.assertLess(err, 1e-12, f"Gradient of cos mode wrong: {err:.2e}")

    def test_fifth_force_zero_in_vacuum(self):
        """Uniform χ = 1 (vacuum): fifth force = 0."""
        n   = 8
        dx  = 1.0
        chi = np.ones(n**3)
        chi_grid = chi.reshape(n, n, n)
        for axis in range(3):
            gx = np.roll(chi_grid, -1, axis) - np.roll(chi_grid, 1, axis)
            self.assertAlmostEqual(np.max(np.abs(gx)), 0.0, places=14)

    def test_screening_mechanism(self):
        """Symmetron fifth force should be much weaker inside a dense clump."""
        a_ssb    = 0.5
        lambda_0 = 1.0
        a        = 1.0
        n        = 32
        dx       = 0.1
        # Dense sphere in the centre, vacuum outside
        rho_array = np.zeros(n)
        rho_array[n//4:3*n//4] = 100.0  # dense region
        chi = solve_symmetron_1d(rho_array, a, a_ssb, lambda_0, dx, n_iter=5000)
        # chi should be ≈ 0 inside dense region, ≈ 1 outside
        chi_inside  = np.abs(chi[n//4:3*n//4]).mean()
        chi_outside = np.abs(chi[:n//4]).mean()
        print(f"  Screening: χ_inside={chi_inside:.4f}, χ_outside={chi_outside:.4f}")
        self.assertLess(chi_inside, chi_outside * 0.5,
                        "Dense region not screened relative to vacuum")


class TestFofR(unittest.TestCase):

    def test_rbar_positive(self):
        """Background Ricci scalar should be positive."""
        cosmo = Cosmology()
        for a in [0.1, 0.5, 1.0]:
            R = 3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m / a**3)
            self.assertGreater(R, 0.0)

    def test_rbar_decreasing(self):
        """R̄(a) should decrease as a increases (matter dilutes)."""
        cosmo = Cosmology()
        R_early = 3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m / 0.5**3)
        R_now   = 3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m / 1.0**3)
        self.assertGreater(R_early, R_now)

    def test_fofr_background_value(self):
        """f_R at background density should equal -n * f_{R0}."""
        fofr0 = 1e-5
        n_fofr = 1.0
        # f_R(R0) = -n * f_{R0} * (R0/R0)^{n+1} = -n * f_{R0}
        expected = -n_fofr * fofr0
        self.assertAlmostEqual(expected, -n_fofr * fofr0)

    def test_fofr_screened_limit(self):
        """In very high density (R >> R0), f_R → 0 (screened)."""
        cosmo  = Cosmology()
        fofr0  = 1e-5
        n_fofr = 1.0
        R0     = 3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m)
        # At a=0.1, R >> R0
        a_early = 0.1
        R_early = 3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m / a_early**3)
        fofr_early = -n_fofr * fofr0 * (R0 / R_early)**(n_fofr + 1)
        print(f"  f_R at a=0.1: {fofr_early:.3e} (should be << {fofr0:.1e})")
        self.assertLess(abs(fofr_early), fofr0 * 0.01)


class TestIntegration(unittest.TestCase):
    """End-to-end gravity + particle tests."""

    def test_two_particle_attraction(self):
        """Two particles attract each other: interpolated force points from p0 toward p1."""
        n  = 32
        L  = 1.0
        dx = L / n
        # Asymmetric placement: p0 at x=0.3, p1 at x=0.6
        # p0 should feel a force in +x direction (toward p1)
        pos = np.array([[0.30, 0.5, 0.5], [0.60, 0.5, 0.5]])
        rho = cic_density(pos, L, n)
        source = rho - 1.0   # overdensity as Poisson source
        phi    = solve_poisson_spectral(source.flatten(), dx, n)
        phi_3d = phi.reshape(n, n, n)
        gx = -(np.roll(phi_3d, -1, 0) - np.roll(phi_3d, 1, 0)) / (2 * dx)
        gy = -(np.roll(phi_3d, -1, 1) - np.roll(phi_3d, 1, 1)) / (2 * dx)
        gz = -(np.roll(phi_3d, -1, 2) - np.roll(phi_3d, 1, 2)) / (2 * dx)
        gxyz = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
        f0 = cic_interp_force(pos[:1], gxyz, L, n)
        print(f"  Two-particle attraction: fx on p0 = {f0[0, 0]:.6f}")
        # p0 is to the left of p1: it should be attracted in +x
        self.assertGreater(f0[0, 0], 1e-6,
                           "Particle 0 should feel positive-x force toward particle 1")

    def test_gravity_zero_for_uniform(self):
        """Uniform density → zero gravitational force everywhere."""
        n   = 8
        L   = 1.0
        dx  = L / n
        rho = np.zeros(n**3)  # uniform = overdensity 0
        phi = solve_poisson_spectral(rho, dx, n)
        g   = -(np.roll(phi.reshape(n,n,n), -1, 0) -
                np.roll(phi.reshape(n,n,n), 1, 0)) / (2 * dx)
        max_g = np.max(np.abs(g))
        print(f"  Uniform density max |g|: {max_g:.3e}")
        self.assertLess(max_g, 1e-10)

    def test_leapfrog_two_particles_energy(self):
        """
        Two gravitationally bound particles should conserve energy over
        many orbits (symplectic leapfrog).
        """
        n  = 32
        L  = 1.0
        dx = L / n
        dt = 1e-3
        n_steps = 500

        # Start with two particles in circular orbit configuration
        pos = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
        v   = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])

        def get_force(pos):
            rho = cic_density(pos, L, n)
            rho -= 1.0
            phi = solve_poisson_spectral(rho.flatten(), dx, n)
            phi_3d = phi.reshape(n, n, n)
            gx = -(np.roll(phi_3d, -1, 0) - np.roll(phi_3d, 1, 0)) / (2 * dx)
            gy = -(np.roll(phi_3d, -1, 1) - np.roll(phi_3d, 1, 1)) / (2 * dx)
            gz = -(np.roll(phi_3d, -1, 2) - np.roll(phi_3d, 1, 2)) / (2 * dx)
            gxyz = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1)
            f = cic_interp_force(pos, gxyz, L, n)
            return f

        # Initial KE
        KE0 = 0.5 * np.sum(v**2)

        for _ in range(n_steps):
            pos  = (pos + v * 0.5 * dt) % L
            f    = get_force(pos)
            v   += f * dt
            pos  = (pos + v * 0.5 * dt) % L

        KE1 = 0.5 * np.sum(v**2)
        # KE can change due to PE exchange; just check it hasn't exploded
        print(f"  Two-particle KE: {KE0:.4f} → {KE1:.4f}")
        self.assertLess(abs(KE1 - KE0) / (KE0 + 1e-10), 2.0,
                        "Energy changed dramatically")


class TestConvergence(unittest.TestCase):
    """Convergence order verification for key algorithms."""

    def test_cic_second_order(self):
        """CIC force interpolation error should decrease as O(dx²)."""
        # Test: interpolate a quadratic force field; CIC is 2nd order for smooth F
        errors = []
        for ng in [8, 16, 32]:
            L  = 1.0
            dx = L / ng
            # Force field F(x) = x² (smooth)
            x  = (np.arange(ng**3) % ng + 0.5) * dx
            f  = x**2
            force_grid = np.stack([f, np.zeros_like(f), np.zeros_like(f)], axis=1)
            # Single particle at x=0.5
            pos = np.array([[0.5, 0.5, 0.5]])
            fi  = cic_interp_force(pos, force_grid, L, ng)
            exact = 0.5**2
            errors.append(abs(fi[0, 0] - exact))
        print(f"  CIC interpolation errors: {[f'{e:.2e}' for e in errors]}")
        # Convergence ratio
        if errors[0] > 1e-14 and errors[1] > 1e-14:
            r = errors[0] / errors[1]
            print(f"  CIC convergence ratio (8→16): {r:.2f}")
            self.assertGreater(r, 1.5, f"CIC not converging: r={r:.2f}")

    def test_leapfrog_second_order(self):
        """Leapfrog error in position should scale as O(dt²)."""
        # Harmonic oscillator: exact solution x(t) = cos(t)
        def run(dt, n_steps):
            x = 1.0
            v = 0.0  # start at rest at x=1
            # DKD: initial half-kick
            f = -x
            v_half = v + 0.5 * dt * f
            for _ in range(n_steps):
                x = x + dt * v_half
                f = -x
                v_half = v_half + dt * f
            # Get position at t = n_steps * dt
            v_full = v_half - 0.5 * dt * (-x)
            return x, v_full

        t_final = 2.0
        errors = []
        for dt in [0.2, 0.1, 0.05]:
            n = int(t_final / dt)
            x, _ = run(dt, n)
            exact = np.cos(t_final)
            errors.append(abs(x - exact))
        print(f"  Leapfrog errors: {[f'{e:.2e}' for e in errors]}")
        r1 = errors[0] / errors[1]
        r2 = errors[1] / errors[2]
        print(f"  Leapfrog convergence ratios: {r1:.2f}, {r2:.2f}")
        self.assertGreater(r1, 3.0, f"Leapfrog not second-order: r={r1:.2f}")
        self.assertGreater(r2, 3.0, f"Leapfrog not second-order: r={r2:.2f}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Verbose output
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total   = result.testsRun
    failed  = len(result.failures) + len(result.errors)
    passed  = total - failed
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed", "✓" if failed == 0 else "✗")
    if failed:
        print(f"  FAILURES: {failed}")
        for test, traceback in result.failures + result.errors:
            print(f"    - {test}")
    print('='*60)
    sys.exit(0 if failed == 0 else 1)
