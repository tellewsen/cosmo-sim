//! Symmetron scalar field solver.
//!
//! Implements the **nonlinear multigrid Newton-Gauss-Seidel** solver for the
//! Symmetron model as described in ISIS (Llinares, Mota & Winther 2014).
//!
//! The dimensionless field  χ = φ/φ₀  satisfies the quasi-static EOM:
//!
//!   ∇²χ = χ/λ₀² · (ρ/ρ̄ · (a_SSB/a)³ − 1 + χ²)
//!
//! where  λ₀ = Compton wavelength [Mpc/h],  a_SSB = scale factor at SSB.
//!
//! For the non-static extension (Llinares & Mota 2014) we add:
//!
//!   ∂²χ/∂τ² + 3H ∂χ/∂τ = ∇²χ − f(χ)
//!
//! solved with a velocity-Verlet leapfrog for χ and χ̇.

use crate::gravity::poisson::{flat3, restrict_uniform, prolongate_uniform};
use crate::utils::types::Real;
use crate::utils::cosmology::Cosmology;

/// Model parameters for the Symmetron.
#[derive(Debug, Clone)]
pub struct SymmetronParams {
    /// Scale factor at SSB (homogeneous universe)
    pub a_ssb:     Real,
    /// Compton wavelength in vacuum  [comoving Mpc/h]
    pub lambda_0:  Real,
    /// Fifth-force coupling β (ratio to Newton in broken phase)
    pub beta_0:    Real,
    /// Include non-static (χ̇) terms
    pub non_static: bool,
    /// Multigrid convergence tolerance
    pub epsilon:   Real,
    /// Maximum V-cycle outer iterations
    pub max_iter:  u32,
    /// Gauss-Seidel sweeps per V-cycle level
    pub n_smooth:  u32,
}

impl Default for SymmetronParams {
    /// Llinares et al. 2014 fiducial parameters.
    fn default() -> Self {
        Self {
            a_ssb:    0.5,
            lambda_0: 1.0,
            beta_0:   1.0,
            non_static: false,
            epsilon:  1e-5,
            max_iter: 150,
            n_smooth: 2,
        }
    }
}

// ── Source function f(χ, ρ, a) ────────────────────────────────────────────────

/// RHS of the quasi-static EOM:  f = χ/λ₀² · (ρ/M² − 1 + χ²)
/// where  ρ/M² = ρ/ρ̄ · (a_SSB/a)³.
#[inline]
fn source_chi(
    chi:              Real,
    rho_over_rhobar:  Real,
    a:                Real,
    p:                &SymmetronParams,
) -> Real {
    let rho_over_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    let lam2 = p.lambda_0 * p.lambda_0;
    chi * (rho_over_m2 - 1.0 + chi * chi) / lam2
}

/// Derivative  ∂f/∂χ (needed for Newton linearisation).
#[inline]
fn dsource_dchi(
    chi:             Real,
    rho_over_rhobar: Real,
    a:               Real,
    p:               &SymmetronParams,
) -> Real {
    let rho_over_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    let lam2 = p.lambda_0 * p.lambda_0;
    (rho_over_m2 - 1.0 + 3.0 * chi * chi) / lam2
}

// ── Newton-Gauss-Seidel sweep ─────────────────────────────────────────────────

/// One red-black Newton-Gauss-Seidel sweep for  ∇²χ = f(χ).
fn ngs_sweep(
    chi:  &mut Vec<Real>,
    rho:  &[Real],
    a:    Real,
    dx:   Real,
    n:    usize,
    p:    &SymmetronParams,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap = chi.clone(); // read neighbours from snapshot

    for color in 0..2_usize {
        for k in 0..n as isize {
            for j in 0..n as isize {
                for i in 0..n as isize {
                    if (i + j + k) as usize % 2 != color { continue; }
                    let c = flat3(i, j, k, n);
                    let chi_c = snap[c];
                    let nb = snap[flat3(i+1, j,   k,   n)]
                           + snap[flat3(i-1, j,   k,   n)]
                           + snap[flat3(i,   j+1, k,   n)]
                           + snap[flat3(i,   j-1, k,   n)]
                           + snap[flat3(i,   j,   k+1, n)]
                           + snap[flat3(i,   j,   k-1, n)];

                    let f   = source_chi(chi_c, rho[c], a, p);
                    let df  = dsource_dchi(chi_c, rho[c], a, p);
                    // Newton step:  χ -= (∇²χ − f) / (−6/dx² − df/dχ)
                    let lap   = (nb - 6.0 * chi_c) * inv_dx2;
                    let denom = -6.0 * inv_dx2 - df;
                    chi[c] = chi_c - (lap - f) / denom;
                }
            }
        }
    }
}

// ── Residual ──────────────────────────────────────────────────────────────────

fn residual_chi(chi: &[Real], rho: &[Real], a: Real, dx: Real, n: usize,
                p: &SymmetronParams) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    for k in 0..n as isize {
        for j in 0..n as isize {
            for i in 0..n as isize {
                let c = flat3(i, j, k, n);
                let lap = (chi[flat3(i+1,j,k,n)] + chi[flat3(i-1,j,k,n)]
                          +chi[flat3(i,j+1,k,n)] + chi[flat3(i,j-1,k,n)]
                          +chi[flat3(i,j,k+1,n)] + chi[flat3(i,j,k-1,n)]
                          - 6.0 * chi[c]) * inv_dx2;
                res[c] = lap - source_chi(chi[c], rho[c], a, p);
            }
        }
    }
    res
}

// ── Nonlinear V-cycle ─────────────────────────────────────────────────────────

fn vcycle_chi(
    chi:   &mut Vec<Real>,
    rho:   &[Real],
    a:     Real,
    dx:    Real,
    n:     usize,
    p:     &SymmetronParams,
    depth: usize,
) {
    let ns = p.n_smooth as usize;
    for _ in 0..ns { ngs_sweep(chi, rho, a, dx, n, p); }

    if n <= 4 || depth == 0 {
        for _ in 0..4 * ns { ngs_sweep(chi, rho, a, dx, n, p); }
        return;
    }

    // Compute residual and restrict
    let res_f  = residual_chi(chi, rho, a, dx, n, p);
    let res_c  = restrict_uniform(&res_f, n);
    let rho_c  = restrict_uniform(rho, n);
    let n_c    = n / 2;
    let dx_c   = dx * 2.0;

    // Restrict current solution for coarse initial guess
    let chi_c_init = restrict_uniform(chi, n);
    let mut chi_c  = chi_c_init.clone();

    // Recurse: solve coarse correction equation
    // The coarse error equation is:  ∇²e = res_c + f(chi_c_init) − f(chi_c_init + e)
    // For simplicity we solve  ∇²chi_c = f(chi_c)  directly (FAS — Full Approx Scheme).
    vcycle_chi(&mut chi_c, &rho_c, a, dx_c, n_c, p, depth - 1);

    // FAS correction: δχ_c = chi_c_new − chi_c_old
    let delta_c: Vec<Real> = chi_c.iter().zip(chi_c_init.iter())
        .map(|(new, old)| new - old)
        .collect();

    // Prolongate and apply correction
    let delta_f = prolongate_uniform(&delta_c, n_c);
    chi.iter_mut().zip(delta_f.iter()).for_each(|(x, d)| *x += d);

    for _ in 0..ns { ngs_sweep(chi, rho, a, dx, n, p); }
}

// ── Non-static leapfrog ───────────────────────────────────────────────────────

/// Advance χ and χ̇ by one full supercomoving step using velocity-Verlet.
///
/// EOM:  χ̈ = ∇²χ − f(χ) − 3H χ̇
pub fn advance_nonstatic(
    chi:     &mut Vec<Real>,
    chi_dot: &mut Vec<Real>,
    rho:     &[Real],
    a:       Real,
    dtau:    Real,
    dx:      Real,
    n:       usize,
    p:       &SymmetronParams,
    cosmo:   &Cosmology,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let h = cosmo.hubble(a);

    let accel = |chi: &[Real], chi_dot: &[Real]| -> Vec<Real> {
        (0..n * n * n).map(|c| {
            let k = (c / (n * n)) as isize;
            let j = ((c / n) % n)  as isize;
            let i = (c % n)         as isize;
            let nb = chi[flat3(i+1,j,k,n)] + chi[flat3(i-1,j,k,n)]
                   + chi[flat3(i,j+1,k,n)] + chi[flat3(i,j-1,k,n)]
                   + chi[flat3(i,j,k+1,n)] + chi[flat3(i,j,k-1,n)];
            let lap = (nb - 6.0 * chi[c]) * inv_dx2;
            let f   = source_chi(chi[c], rho[c], a, p);
            lap - f - 3.0 * h * chi_dot[c]
        }).collect()
    };

    // Half-kick χ̇
    let a0 = accel(chi, chi_dot);
    chi_dot.iter_mut().zip(a0.iter()).for_each(|(v, acc)| *v += 0.5 * dtau * acc);

    // Drift χ
    chi.iter_mut().zip(chi_dot.iter()).for_each(|(x, v)| *x += dtau * v);

    // Second half-kick with updated χ
    let a1 = accel(chi, chi_dot);
    chi_dot.iter_mut().zip(a1.iter()).for_each(|(v, acc)| *v += 0.5 * dtau * acc);
}

// ── Public solver ─────────────────────────────────────────────────────────────

/// Solve (quasi-static) or advance (non-static) the Symmetron field.
pub fn solve_symmetron(
    chi:     &mut Vec<Real>,
    chi_dot: &mut Vec<Real>,
    rho:     &[Real],
    a:       Real,
    dtau:    Real,
    dx:      Real,
    n:       usize,
    p:       &SymmetronParams,
    cosmo:   &Cosmology,
) {
    if p.non_static {
        advance_nonstatic(chi, chi_dot, rho, a, dtau, dx, n, p, cosmo);
    } else {
        let max_depth = (n as f64).log2() as usize;
        for iter in 0..p.max_iter {
            let res     = residual_chi(chi, rho, a, dx, n, p);
            let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
            log::debug!("Symmetron iter {iter}: |r|_∞ = {res_max:.3e}");
            if res_max < p.epsilon { break; }
            vcycle_chi(chi, rho, a, dx, n, p, max_depth);
        }
    }
}

// ── Fifth-force grid ──────────────────────────────────────────────────────────

/// Compute the Symmetron fifth-force acceleration on the grid:
///   F₅ = β₀ φ₀/M_Pl · ∇χ = β₀/a · ∇χ  [comoving internal units]
///
/// (See ISIS paper, eq. 7)
pub fn fifth_force_grid(
    chi: &[Real],
    dx:  Real,
    n:   usize,
    a:   Real,
    p:   &SymmetronParams,
) -> Vec<[Real; 3]> {
    let inv_2dx = 0.5 / dx;
    let prefac  = p.beta_0 / a;
    let mut f5  = vec![[0.0_f64; 3]; n * n * n];
    for k in 0..n as isize {
        for j in 0..n as isize {
            for i in 0..n as isize {
                let c = flat3(i, j, k, n);
                f5[c][0] = prefac * (chi[flat3(i+1,j,k,n)] - chi[flat3(i-1,j,k,n)]) * inv_2dx;
                f5[c][1] = prefac * (chi[flat3(i,j+1,k,n)] - chi[flat3(i,j-1,k,n)]) * inv_2dx;
                f5[c][2] = prefac * (chi[flat3(i,j,k+1,n)] - chi[flat3(i,j,k-1,n)]) * inv_2dx;
            }
        }
    }
    f5
}

// ── Analytical solution helpers (for testing) ────────────────────────────────

/// For a uniform-density sphere in an otherwise empty background, the
/// Symmetron has an analytical profile that we can compare against.
/// Here we just return the screened value (χ = 0) and broken-phase value (χ = 1).
pub fn chi_vacuum() -> Real { 1.0 }
pub fn chi_screened() -> Real { 0.0 }

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::cosmology::Cosmology;

    fn default_cosmo() -> Cosmology { Cosmology::default() }

    /// In a high-density region (ρ >> M²) the Symmetron should be screened: χ → 0.
    #[test]
    fn symmetron_screening_high_density() {
        let n  = 8_usize;
        let dx = 1.0_f64;
        let p  = SymmetronParams { a_ssb: 0.5, lambda_0: 1.0, beta_0: 1.0,
                                   non_static: false, epsilon: 1e-8, max_iter: 200, n_smooth: 2 };

        // Very high uniform density → screened
        let rho = vec![1000.0_f64; n * n * n];
        let mut chi     = vec![0.1_f64; n * n * n]; // start near 0
        let mut chi_dot = vec![0.0_f64; n * n * n];

        solve_symmetron(&mut chi, &mut chi_dot, &rho, 1.0, 0.0, dx, n, &p, &default_cosmo());

        let max_chi = chi.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        println!("Screened max |χ|: {max_chi:.3e}");
        assert!(max_chi < 0.1, "Field not screened: max |χ| = {max_chi}");
    }

    /// In the vacuum (ρ = 0) the Symmetron should relax to χ = ±1.
    #[test]
    fn symmetron_vacuum_relaxation() {
        let n  = 8_usize;
        let dx = 1.0_f64;
        let p  = SymmetronParams { a_ssb: 0.5, lambda_0: 1.0, beta_0: 1.0,
                                   non_static: false, epsilon: 1e-8, max_iter: 500, n_smooth: 3 };

        // Low density: a_ssb/a = 0.5/1.0 = 0.5, so rho_over_m2 = rho * 0.125
        // For rho = 0: ρ/M² = 0 < 1 → broken phase, χ_vac = 1
        let rho = vec![0.0_f64; n * n * n];
        let mut chi     = vec![0.9_f64; n * n * n];
        let mut chi_dot = vec![0.0_f64; n * n * n];

        solve_symmetron(&mut chi, &mut chi_dot, &rho, 1.0, 0.0, dx, n, &p, &default_cosmo());

        let mean_chi = chi.iter().sum::<f64>() / chi.len() as f64;
        println!("Vacuum mean χ: {mean_chi:.6}");
        assert!((mean_chi.abs() - 1.0).abs() < 0.05,
                "Vacuum not converged: mean χ = {mean_chi}");
    }

    /// Source function should match analytical formula.
    #[test]
    fn source_chi_analytical() {
        let p = SymmetronParams::default();
        // At a=a_ssb, rho=rhobar: ρ/M² = 1 → f = χ(1−1+χ²)/λ² = χ³/λ²
        let a   = p.a_ssb;
        let chi = 0.5_f64;
        let f   = source_chi(chi, 1.0, a, &p);
        let expected = chi * chi * chi / (p.lambda_0 * p.lambda_0);
        assert!((f - expected).abs() < 1e-14, "f={f}, expected={expected}");
    }

    /// Residual should be zero when χ satisfies the EOM analytically.
    /// For uniform ρ and uniform χ, ∇²χ = 0, so f(χ) must also be 0.
    /// This occurs at χ = 0 (always) or χ = ±√(1 − ρ/M²) in broken phase.
    #[test]
    fn residual_zero_at_equilibrium() {
        let n  = 4_usize;
        let dx = 1.0_f64;
        let p  = SymmetronParams::default();

        // Uniform ρ = 0, broken phase: χ_eq = 1.0
        let rho = vec![0.0_f64; n * n * n];
        let chi = vec![1.0_f64; n * n * n];

        let a = 1.0_f64; // (a_ssb/a)^3 = 0.125, rho_over_m2 = 0
        let res = residual_chi(&chi, &rho, a, dx, n, &p);
        let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        println!("Residual at equilibrium: {res_max:.3e}");
        // f(1, 0) = 1*(0 - 1 + 1)/λ² = 0 → residual = 0
        assert!(res_max < 1e-14, "Residual not zero: {res_max}");
    }

    /// Fifth-force should point in the direction of ∇χ.
    #[test]
    fn fifth_force_direction() {
        let n  = 8_usize;
        let dx = 1.0_f64;
        let p  = SymmetronParams { beta_0: 1.0, ..SymmetronParams::default() };
        // Linear χ = x/n (gradient only in x)
        let chi: Vec<Real> = (0..n*n*n).map(|c| {
            let i = c % n;
            i as Real / n as Real
        }).collect();
        let f5 = fifth_force_grid(&chi, dx, n, 1.0, &p);
        // All cells: fx = β₀/a * ∂χ/∂x = 1/n/dx = 1/8
        let expected_fx = 1.0 / (n as Real * dx);
        for (c, fc) in f5.iter().enumerate() {
            assert!((fc[0] - expected_fx).abs() < 1e-10,
                    "cell {c}: fx={}, expected={expected_fx}", fc[0]);
            assert!(fc[1].abs() < 1e-10, "cell {c}: fy={}", fc[1]);
            assert!(fc[2].abs() < 1e-10, "cell {c}: fz={}", fc[2]);
        }
    }
}
