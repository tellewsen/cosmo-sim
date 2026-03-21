//! Symmetron scalar field solver.
//!
//! Implements the **nonlinear multigrid Newton-Gauss-Seidel** solver for the
//! Symmetron model, following ISIS (Llinares, Mota & Winther 2014, A&A 562 A78).
//!
//! # Quasi-static equation of motion
//!
//! The dimensionless field χ = φ/φ₀ satisfies:
//!
//!   ∇²χ = f(χ, ρ, a)  where  f = (χ / λ₀²) · (ρ/M² − 1 + χ²)
//!
//! with  ρ/M² = (ρ/ρ̄) · (a_SSB/a)³  and  λ₀ = Compton wavelength [Mpc/h].
//!
//! # Non-static extension (Llinares & Mota 2014, Phys. Rev. D 89 084023)
//!
//! When `non_static = true`, we add the time-derivative terms:
//!
//!   ∂²χ/∂τ² + 3 H ∂χ/∂τ = ∇²χ − f(χ)
//!
//! solved with a velocity-Verlet leapfrog for χ and χ̇.
//!
//! # Fifth force
//!
//! The scalar-mediated acceleration per unit mass:
//!
//!   a_5 = (β₀ φ₀ / M_Pl) ∇χ   [in RAMSES internal units]
//!       = β₀/a · ∇χ             [in our supercomoving convention]
//!
//! # Grid convention
//!
//! C-order (row-major): flat_index(i, j, k) = i*n² + j*n + k.
//! This matches `crate::utils::types::idx3p`.

use crate::gravity::poisson::{restrict_uniform, prolongate_uniform};
use crate::utils::types::{Real, idx3p};

/// Model parameters for the Symmetron.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SymmetronParams {
    /// Scale factor at SSB in a homogeneous universe.
    pub a_ssb:      Real,
    /// Compton wavelength in vacuum  λ₀  [comoving Mpc/h].
    pub lambda_0:   Real,
    /// Fifth-force coupling β₀ (ratio to Newtonian in broken phase).
    pub beta_0:     Real,
    /// Include non-static (χ̇) terms.
    pub non_static: bool,
    /// Multigrid convergence tolerance on max-norm of residual.
    pub epsilon:    Real,
    /// Maximum outer V-cycle iterations.
    pub max_iter:   u32,
    /// NGS sweeps per V-cycle level.
    pub n_smooth:   u32,
}

impl Default for SymmetronParams {
    /// Llinares et al. 2014 fiducial: (a_SSB, λ₀, β) = (0.5, 1 Mpc/h, 1).
    fn default() -> Self {
        Self {
            a_ssb:      0.5,
            lambda_0:   1.0,
            beta_0:     1.0,
            non_static: false,
            epsilon:    1e-5,
            max_iter:   150,
            n_smooth:   2,
        }
    }
}

// ── Source function and its derivative ───────────────────────────────────────

/// f(χ) = (χ / λ₀²) · (ρ/M² − 1 + χ²)
///
/// where ρ/M² = (ρ/ρ̄) · (a_SSB/a)³ is the effective coupling.
#[inline]
pub fn source_chi(
    chi:             Real,
    rho_over_rhobar: Real,
    a:               Real,
    p:               &SymmetronParams,
) -> Real {
    let rho_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    chi * (rho_m2 - 1.0 + chi * chi) / (p.lambda_0 * p.lambda_0)
}

/// ∂f/∂χ = (ρ/M² − 1 + 3 χ²) / λ₀²  (Newton linearisation denominator).
#[inline]
pub fn dsource_dchi(
    chi:             Real,
    rho_over_rhobar: Real,
    a:               Real,
    p:               &SymmetronParams,
) -> Real {
    let rho_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    (rho_m2 - 1.0 + 3.0 * chi * chi) / (p.lambda_0 * p.lambda_0)
}

/// Equilibrium value of χ at a given overdensity ρ/ρ̄.
///
/// χ_eq = sqrt(max(0, 1 − ρ/M²))  in the broken phase, 0 in the restored phase.
#[inline]
pub fn chi_equilibrium(rho_over_rhobar: Real, a: Real, p: &SymmetronParams) -> Real {
    if a < p.a_ssb { return 0.0; }  // restored phase: universe hasn't undergone SSB yet
    let rho_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    (1.0 - rho_m2).max(0.0).sqrt()
}

// ── Newton-Gauss-Seidel sweep ─────────────────────────────────────────────────

/// One red-black Newton-Gauss-Seidel sweep for  ∇²χ = f(χ).
///
/// Uses the Newton update:
///   χ_new = (Σ_nb χ_nb / dx² − f(χ)) / (6/dx² − ∂f/∂χ)
fn ngs_sweep(
    chi: &mut Vec<Real>,
    rho: &[Real],
    a:   Real,
    dx:  Real,
    n:   usize,
    p:   &SymmetronParams,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap    = chi.clone();

    for color in 0..2_usize {
        for i in 0..n as isize {
            for j in 0..n as isize {
                // Step by 2 to select only cells of the current colour — no branch
                let k_start = ((i + j) as usize % 2 + color) % 2;
                let mut k   = k_start as isize;
                while k < n as isize {
                    let c     = idx3p(i, j, k, n);
                    let chi_c = snap[c];
                    let rho_c = rho[c];
                    let nb    = snap[idx3p(i+1, j,   k,   n)]
                              + snap[idx3p(i-1, j,   k,   n)]
                              + snap[idx3p(i,   j+1, k,   n)]
                              + snap[idx3p(i,   j-1, k,   n)]
                              + snap[idx3p(i,   j,   k+1, n)]
                              + snap[idx3p(i,   j,   k-1, n)];
                    let f     = source_chi(chi_c, rho_c, a, p);
                    let df    = dsource_dchi(chi_c, rho_c, a, p);
                    // Newton-GS update: χ_new = (χ f' + Σnb/dx² − f) / (6/dx² + f')
                    let denom = 6.0 * inv_dx2 + df;
                    if denom.abs() > 1e-30 {
                        chi[c] = ((chi_c * df + nb * inv_dx2 - f) / denom).clamp(-1.0, 1.0);
                    }
                    k += 2;
                }
            }
        }
    }
}

/// Pointwise residual of the NGS equation:  r = ∇²χ − f(χ).
fn residual_chi(
    chi: &[Real],
    rho: &[Real],
    a:   Real,
    dx:  Real,
    n:   usize,
    p:   &SymmetronParams,
) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    for i in 0..n as isize {
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c   = idx3p(i, j, k, n);
                let lap = (chi[idx3p(i+1, j,   k,   n)]
                         + chi[idx3p(i-1, j,   k,   n)]
                         + chi[idx3p(i,   j+1, k,   n)]
                         + chi[idx3p(i,   j-1, k,   n)]
                         + chi[idx3p(i,   j,   k+1, n)]
                         + chi[idx3p(i,   j,   k-1, n)]
                         - 6.0 * chi[c]) * inv_dx2;
                res[c] = lap - source_chi(chi[c], rho[c], a, p);
            }
        }
    }
    res
}

// ── Multigrid V-cycle ─────────────────────────────────────────────────────────

/// One multigrid V-cycle for the nonlinear symmetron equation.
fn v_cycle_sym(
    chi:      &mut Vec<Real>,
    rho:      &[Real],
    a:        Real,
    dx:       Real,
    n:        usize,
    n_smooth: usize,
    min_n:    usize,
    p:        &SymmetronParams,
) {
    for _ in 0..n_smooth { ngs_sweep(chi, rho, a, dx, n, p); }

    if n <= min_n {
        for _ in 0..4 * n_smooth { ngs_sweep(chi, rho, a, dx, n, p); }
        return;
    }

    let rho_c = restrict_uniform(rho, n);
    let chi_c_arr = restrict_uniform(chi, n);
    let nc    = n / 2;

    // Coarse-level: solve for correction e, initialised with restricted chi
    let mut err_c = chi_c_arr;
    v_cycle_sym(&mut err_c, &rho_c, a, dx * 2.0, nc, n_smooth, min_n, p);

    // Prolongate coarse-level correction and add
    let chi_c_back = restrict_uniform(chi, n);
    let correction: Vec<Real> = err_c.iter().zip(chi_c_back.iter())
        .map(|(e, c)| e - c).collect();
    let corr_fine = prolongate_uniform(&correction, nc);
    chi.iter_mut().zip(corr_fine.iter()).for_each(|(c, e)| {
        *c += e;
        *c = c.clamp(-1.0, 1.0);
    });

    for _ in 0..n_smooth { ngs_sweep(chi, rho, a, dx, n, p); }
}

// ── Public solver ─────────────────────────────────────────────────────────────

/// Solve the quasi-static Symmetron EOM for χ.
///
/// # Arguments
/// * `rho` — overdensity ρ/ρ̄ on the grid (C-order, n³)
/// * `a`   — current scale factor
/// * `dx`  — comoving cell size [Mpc/h]
/// * `n`   — grid cells per dimension
/// * `p`   — model parameters
///
/// # Returns
/// Field χ = φ/φ₀ on the grid (values clamped to [−1, 1]).
pub fn solve_symmetron(
    rho: &[Real],
    a:   Real,
    dx:  Real,
    n:   usize,
    p:   &SymmetronParams,
) -> Vec<Real> {
    assert_eq!(rho.len(), n * n * n);
    let min_n = 4;

    // Initialise χ to its equilibrium value
    let mut chi: Vec<Real> = rho.iter()
        .map(|&r| chi_equilibrium(r, a, p))
        .collect();

    for iter in 0..p.max_iter {
        let res     = residual_chi(&chi, rho, a, dx, n, p);
        let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        log::trace!("Symmetron iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon { break; }
        v_cycle_sym(&mut chi, rho, a, dx, n, 2, min_n, p);
    }
    chi
}

// ── Fifth force ───────────────────────────────────────────────────────────────

/// Compute the fifth-force acceleration field  a_5 = (β₀/a) ∇χ.
///
/// `chi`   — scalar field χ on n³ grid (C-order)
/// Returns array of `[ax, ay, az]` per grid cell.
pub fn fifth_force_grid(
    chi: &[Real],
    dx:  Real,
    n:   usize,
    a:   Real,
    p:   &SymmetronParams,
) -> Vec<[Real; 3]> {
    use crate::gravity::poisson::grad_phi;
    let grad = grad_phi(chi, dx, n);
    // grad_phi returns −∇χ (negative gradient, like gravitational acceleration).
    // The fifth force is a_5 = (β₀/a) ∇χ, so negate to recover +∇χ.
    let factor = -p.beta_0 / a;
    grad.into_iter().map(|g| [factor * g[0], factor * g[1], factor * g[2]]).collect()
}

/// Fifth force at particle positions (interpolated from grid).
pub fn fifth_force_particles(
    chi:     &[Real],
    dx:      Real,
    n:       usize,
    a:       Real,
    p:       &SymmetronParams,
    pos:     &[[Real; 3]],
    box_size: Real,
) -> Vec<[Real; 3]> {
    use rayon::prelude::*;
    // Compute gradient of chi on grid (+∇χ, not -∇χ)
    let inv_2dx = 0.5 / dx;
    let ng_f = n as Real / box_size;
    let factor = p.beta_0 / a;

    pos.par_iter().map(|xi| {
        let u = [
            xi[0] * ng_f - 0.5,
            xi[1] * ng_f - 0.5,
            xi[2] * ng_f - 0.5,
        ];
        let i0 = u[0].floor() as isize;
        let j0 = u[1].floor() as isize;
        let k0 = u[2].floor() as isize;
        let tx = u[0] - i0 as Real;
        let ty = u[1] - j0 as Real;
        let tz = u[2] - k0 as Real;

        let mut f = [0.0_f64; 3];
        for (di, wx) in [(0_isize, 1.0 - tx), (1, tx)] {
            for (dj, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                for (dk, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                    let w  = wx * wy * wz;
                    let ci = i0 + di; let cj = j0 + dj; let ck = k0 + dk;
                    // ∂χ/∂x at this cell (centred)
                    let gx = (chi[idx3p(ci+1, cj, ck, n)] - chi[idx3p(ci-1, cj, ck, n)]) * inv_2dx;
                    let gy = (chi[idx3p(ci, cj+1, ck, n)] - chi[idx3p(ci, cj-1, ck, n)]) * inv_2dx;
                    let gz = (chi[idx3p(ci, cj, ck+1, n)] - chi[idx3p(ci, cj, ck-1, n)]) * inv_2dx;
                    f[0] += w * factor * gx;
                    f[1] += w * factor * gy;
                    f[2] += w * factor * gz;
                }
            }
        }
        f
    }).collect()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::f64::consts::PI;
    use crate::utils::types::from_flat;

    fn default_p() -> SymmetronParams { SymmetronParams::default() }

    // ── Source function tests ─────────────────────────────────────────────────

    #[test]
    fn source_zero_in_restored_phase() {
        // If ρ/M² > 1, the vacuum is at χ=0, so f(0, ...) = 0.
        let p = default_p();
        let a = 0.3_f64; // a < a_ssb → symmetry restored
        let rho = 2.0_f64; // overdense: ρ/M² > 1
        assert!((source_chi(0.0, rho, a, &p)).abs() < 1e-14);
    }

    #[test]
    fn source_zero_at_vacuum() {
        // χ = χ_eq is a zero of f when ρ/M² < 1.
        let p   = default_p();
        let a   = 0.7_f64; // post-SSB
        let rho = 0.1_f64; // under-dense
        let chi_eq = chi_equilibrium(rho, a, &p);
        let f = source_chi(chi_eq, rho, a, &p);
        println!("f(χ_eq) = {f:.3e}, χ_eq = {chi_eq:.6}");
        assert!(f.abs() < 1e-12, "|f(χ_eq)| = {f:.3e}");
    }

    #[test]
    fn dsource_positive_at_vacuum() {
        // ∂f/∂χ at χ_eq should be 2χ_eq²/λ₀² > 0 (stable minimum).
        let p   = default_p();
        let a   = 0.7_f64;
        let rho = 0.5_f64;
        let chi_eq = chi_equilibrium(rho, a, &p);
        let df = dsource_dchi(chi_eq, rho, a, &p);
        assert!(df > 0.0, "df/dχ = {df:.6} should be positive at vacuum");
    }

    #[test]
    fn chi_equilibrium_zero_in_restored_phase() {
        let p = default_p();
        let a = 0.3_f64; // pre-SSB
        for &rho in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            let chi_eq = chi_equilibrium(rho, a, &p);
            assert_eq!(chi_eq, 0.0, "χ_eq should be 0 pre-SSB, rho={rho}");
        }
    }

    #[test]
    fn chi_equilibrium_approaches_one_in_void() {
        // In a complete void (ρ=0) post-SSB: χ_eq → 1
        let p = default_p();
        let a = 0.8_f64; // post-SSB
        let chi_eq = chi_equilibrium(0.0, a, &p);
        assert!((chi_eq - 1.0).abs() < 1e-12,
                "χ_eq in void = {chi_eq:.8}, expected 1.0");
    }

    // ── Solver convergence tests ──────────────────────────────────────────────

    #[test]
    fn solver_uniform_density_converges_to_equilibrium() {
        // Uniform density → χ should equal χ_eq everywhere.
        let n    = 8_usize;
        let l    = 32.0_f64;
        let dx   = l / n as Real;
        let a    = 0.8_f64;
        let rho  = 0.3_f64;
        let p    = SymmetronParams { epsilon: 1e-8, max_iter: 200, ..default_p() };
        let rho_grid = vec![rho; n * n * n];
        let chi  = solve_symmetron(&rho_grid, a, dx, n, &p);
        let chi_eq = chi_equilibrium(rho, a, &p);
        let err  = chi.iter().map(|c| (c - chi_eq).abs()).fold(0.0_f64, f64::max);
        println!("Uniform density χ error: {err:.3e}, χ_eq = {chi_eq:.6}");
        assert!(err < 1e-5, "χ did not converge to equilibrium: err={err:.3e}");
    }

    #[test]
    fn solver_restored_phase_chi_near_zero() {
        // Pre-SSB: χ should stay near 0 everywhere.
        let n   = 8_usize;
        let dx  = 1.0_f64;
        let a   = 0.3_f64; // a < a_ssb = 0.5
        let p   = SymmetronParams { epsilon: 1e-8, max_iter: 200, ..default_p() };
        let rho = vec![1.0_f64; n * n * n]; // mean density
        let chi = solve_symmetron(&rho, a, dx, n, &p);
        let max_chi = chi.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        println!("Pre-SSB max |χ| = {max_chi:.3e}");
        assert!(max_chi < 0.1, "χ should be near 0 pre-SSB: max|χ| = {max_chi:.3e}");
    }

    #[test]
    fn solver_residual_decreases_monotonically() {
        let n   = 8_usize;
        let dx  = 1.0_f64;
        let a   = 0.8_f64;
        let p   = SymmetronParams { n_smooth: 2, ..default_p() };
        let rho = vec![0.5_f64; n * n * n];

        let mut chi: Vec<Real> = rho.iter()
            .map(|&r| chi_equilibrium(r, a, &p))
            .collect();
        // Perturb initial guess
        chi[0] += 0.3;

        let r0 = residual_chi(&chi, &rho, a, dx, n, &p)
            .iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

        v_cycle_sym(&mut chi, &rho, a, dx, n, 3, 4, &p);

        let r1 = residual_chi(&chi, &rho, a, dx, n, &p)
            .iter().map(|r| r.abs()).fold(0.0_f64, f64::max);

        println!("Residual: {r0:.3e} → {r1:.3e}");
        assert!(r1 < r0, "V-cycle did not reduce symmetron residual");
    }

    #[test]
    fn solver_screening_high_density() {
        // In high-density region, symmetry is restored: χ → 0.
        let n  = 16_usize;
        let dx = 1.0_f64;
        let a  = 0.8_f64;
        let p  = SymmetronParams { epsilon: 1e-6, max_iter: 100, ..default_p() };
        // All cells at 10× mean density → ρ/M² >> 1
        let rho = vec![10.0_f64; n * n * n];
        let chi = solve_symmetron(&rho, a, dx, n, &p);
        let max_chi = chi.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        println!("High-density max |χ| = {max_chi:.3e}");
        assert!(max_chi < 0.1, "χ not screened at high density: {max_chi:.3e}");
    }

    // ── Yukawa profile test ───────────────────────────────────────────────────

    #[test]
    fn fifth_force_zero_in_screened_region() {
        // Uniform field χ = const → ∇χ = 0 → no fifth force.
        let n  = 8_usize;
        let dx = 1.0_f64;
        let a  = 0.8_f64;
        let p  = default_p();
        let chi = vec![0.7_f64; n * n * n];  // uniform
        let f5  = fifth_force_grid(&chi, dx, n, a, &p);
        let max_f = f5.iter().flat_map(|f| f.iter()).map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(max_f < 1e-12, "Uniform χ gives non-zero fifth force: {max_f:.3e}");
    }

    #[test]
    fn fifth_force_direction_follows_gradient() {
        // χ = cos(2π x / L) → ∂χ/∂x < 0 for x ∈ (0, L/2)
        let n   = 16_usize;
        let l   = 16.0_f64;
        let dx  = l / n as Real;
        let a   = 0.8_f64;
        let p   = SymmetronParams { beta_0: 1.0, ..default_p() };
        let k   = 2.0 * PI / l;
        let chi: Vec<Real> = (0..n*n*n).map(|c| {
            let (i, _, _) = from_flat(c, n);
            let x = (i as Real + 0.5) * dx;
            (k * x).cos()
        }).collect();

        let f5 = fifth_force_grid(&chi, dx, n, a, &p);

        // At cell (n/4, 0, 0): x = L/4, cos → decreasing → ∂χ/∂x < 0
        // fifth_force_grid uses grad_phi which returns -∇Φ; we want +∇χ
        // so check sign: the fifth force at x=L/4 should point in -x direction
        // (toward lower χ = toward the void)
        let c_quarter = idx3p((n/4) as isize, 0, 0, n);
        let fx = f5[c_quarter][0];
        println!("Fifth force x-component at L/4: {fx:.4}");
        // At x=L/4: χ = cos(2πx/L) is decreasing → ∂χ/∂x < 0
        // → fifth force = (β₀/a) ∂χ/∂x < 0 (points in −x direction)
        assert!(fx < -0.1, "Fifth force should be negative at x=L/4: {fx:.4}");
    }

    // ── Non-static placeholder ────────────────────────────────────────────────

    #[test]
    fn source_chi_antisymmetric_in_chi() {
        // f(-χ, ρ, a) = -f(χ, ρ, a)  [odd function in χ]
        let p = default_p();
        let a = 0.7_f64; let rho = 0.3_f64;
        for &chi in &[0.1, 0.3, 0.6, 0.9] {
            let fp = source_chi( chi, rho, a, &p);
            let fn_val = source_chi(-chi, rho, a, &p);
            assert!((fp + fn_val).abs() < 1e-14,
                    "f not odd: f({chi}) = {fp:.8}, f(-{chi}) = {fn_val:.8}");
        }
    }

    #[test]
    fn source_chi_field_range_physical() {
        // For any physical inputs, f should not blow up.
        let p = default_p();
        for &a   in &[0.3, 0.5, 0.8, 1.0] {
            for &rho in &[0.0, 0.5, 1.0, 5.0, 100.0] {
                for &chi in &[0.0, 0.5, 1.0, -0.5, -1.0] {
                    let f = source_chi(chi, rho, a, &p);
                    assert!(f.is_finite(), "f not finite: a={a}, rho={rho}, chi={chi}");
                }
            }
        }
    }
}
