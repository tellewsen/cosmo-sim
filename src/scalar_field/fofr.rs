//! Hu-Sawicki f(R) modified gravity solver.
//!
//! The scalaron  f_R = df/dR  satisfies the quasi-static EOM
//!
//!   ∇²f_R = (1/3)[δR(f_R) − δρ/M_Pl²]
//!
//! where  δR = R − R̄  and the background  R̄(a)  is the ΛCDM Ricci scalar.
//!
//! Reference: Llinares et al. 2014 (ISIS paper §3.2), Hu & Sawicki 2007.

use crate::gravity::poisson::flat3;
use crate::utils::types::Real;
use crate::utils::cosmology::Cosmology;

/// Hu-Sawicki f(R) model parameters.
#[derive(Debug, Clone)]
pub struct FofrParams {
    /// |f_{R0}| — background field value today
    pub fofr0:    Real,
    /// Power-law index n
    pub n:        Real,
    /// Multigrid convergence tolerance
    pub epsilon:  Real,
    pub max_iter: u32,
    pub n_smooth: u32,
}

impl Default for FofrParams {
    fn default() -> Self {
        Self { fofr0: 1e-5, n: 1.0, epsilon: 1e-5, max_iter: 150, n_smooth: 2 }
    }
}

// ── Background quantities ─────────────────────────────────────────────────────

/// ΛCDM background Ricci scalar  R̄(a) in units of H₀².
/// Approximation:  R̄ ≈ 3H₀² [4Ω_Λ + Ω_m a^{-3}]  (matter + Λ, no radiation).
#[inline]
pub fn r_bar(a: Real, cosmo: &Cosmology) -> Real {
    3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m / (a * a * a))
}

/// Hu-Sawicki f_R at background Ricci scalar R:
///   f_R(R) = −n |f_{R0}| (R̄₀/R)^{n+1}
#[inline]
fn fofr_of_r(r: Real, r0: Real, p: &FofrParams) -> Real {
    -p.n * p.fofr0 * (r0 / r).powf(p.n + 1.0)
}

/// Invert Hu-Sawicki to get R from f_R.
/// f_R = −n f_{R0} (R0/R)^{n+1}  →  R = R0 * (n f_{R0} / |u|)^{1/(n+1)}
/// Valid only for u < 0; returns R̄ as fallback for unphysical values.
#[inline]
fn r_of_fofr(u: Real, r0: Real, p: &FofrParams, r_bar_a: Real) -> Real {
    if u >= 0.0 { return r_bar_a; } // screened / unphysical
    let exponent = 1.0 / (p.n + 1.0);
    r0 * (p.n * p.fofr0 / (-u)).powf(exponent)
}

// ── Source function ───────────────────────────────────────────────────────────

/// RHS of the f_R equation at a cell:
///   S(u) = (1/3)[(R(u) − R̄) − Ω_m (ρ/ρ̄ − 1) a^{-3}]
/// in units of H₀².
#[inline]
fn fofr_source(
    u:               Real,   // f_R field value
    rho_over_rhobar: Real,
    a:               Real,
    cosmo:           &Cosmology,
    p:               &FofrParams,
) -> Real {
    let r0      = r_bar(1.0, cosmo);
    let rbar_a  = r_bar(a, cosmo);
    let r_u     = r_of_fofr(u, r0, p, rbar_a);
    let delta_r = r_u - rbar_a;
    let delta_rho = (rho_over_rhobar - 1.0) * cosmo.omega_m / (a * a * a);
    (delta_r - delta_rho) / 3.0
}

/// Numerical derivative  dS/du via central finite difference.
#[inline]
fn dfofr_source_du(
    u:               Real,
    rho_over_rhobar: Real,
    a:               Real,
    cosmo:           &Cosmology,
    p:               &FofrParams,
) -> Real {
    let eps = u.abs().max(1e-30) * 1e-5 + 1e-40;
    let sp  = fofr_source(u + eps, rho_over_rhobar, a, cosmo, p);
    let sm  = fofr_source(u - eps, rho_over_rhobar, a, cosmo, p);
    (sp - sm) / (2.0 * eps)
}

// ── Newton-Gauss-Seidel sweep ─────────────────────────────────────────────────

fn fofr_ngs_sweep(
    u:    &mut Vec<Real>,
    rho:  &[Real],
    a:    Real,
    dx:   Real,
    n:    usize,
    p:    &FofrParams,
    cosmo: &Cosmology,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap = u.clone();

    for color in 0..2_usize {
        for k in 0..n as isize {
            for j in 0..n as isize {
                for i in 0..n as isize {
                    if (i + j + k) as usize % 2 != color { continue; }
                    let c = flat3(i, j, k, n);
                    let u_c = snap[c];
                    let nb  = snap[flat3(i+1,j,k,n)] + snap[flat3(i-1,j,k,n)]
                            + snap[flat3(i,j+1,k,n)] + snap[flat3(i,j-1,k,n)]
                            + snap[flat3(i,j,k+1,n)] + snap[flat3(i,j,k-1,n)];
                    let lap   = (nb - 6.0 * u_c) * inv_dx2;
                    let src   = fofr_source(u_c, rho[c], a, cosmo, p);
                    let dsrc  = dfofr_source_du(u_c, rho[c], a, cosmo, p);
                    let denom = -6.0 * inv_dx2 - dsrc;
                    let mut u_new = u_c - (lap - src) / denom;
                    // Enforce physical bound: f_R ≤ 0 for Hu-Sawicki
                    if u_new > 0.0 { u_new = -1e-30; }
                    u[c] = u_new;
                }
            }
        }
    }
}

// ── Residual ──────────────────────────────────────────────────────────────────

fn fofr_residual(
    u:    &[Real],
    rho:  &[Real],
    a:    Real,
    dx:   Real,
    n:    usize,
    p:    &FofrParams,
    cosmo: &Cosmology,
) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    for k in 0..n as isize {
        for j in 0..n as isize {
            for i in 0..n as isize {
                let c = flat3(i, j, k, n);
                let lap = (u[flat3(i+1,j,k,n)] + u[flat3(i-1,j,k,n)]
                          +u[flat3(i,j+1,k,n)] + u[flat3(i,j-1,k,n)]
                          +u[flat3(i,j,k+1,n)] + u[flat3(i,j,k-1,n)]
                          - 6.0 * u[c]) * inv_dx2;
                res[c] = lap - fofr_source(u[c], rho[c], a, cosmo, p);
            }
        }
    }
    res
}

// ── Public solver ─────────────────────────────────────────────────────────────

/// Solve the Hu-Sawicki f_R equation to convergence.
/// Initialises from the background value and returns the converged field.
pub fn solve_fofr(
    rho:   &[Real],
    a:     Real,
    dx:    Real,
    n:     usize,
    p:     &FofrParams,
    cosmo: &Cosmology,
) -> Vec<Real> {
    let r0     = r_bar(1.0, cosmo);
    let bg_val = fofr_of_r(r0, r0, p);
    let mut u  = vec![bg_val; n * n * n];
    for iter in 0..p.max_iter {
        let res     = fofr_residual(&u, rho, a, dx, n, p, cosmo);
        let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        log::debug!("f(R) iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon { break; }
        for _ in 0..p.n_smooth { fofr_ngs_sweep(&mut u, rho, a, dx, n, p, cosmo); }
    }
    u
}

/// Compute fifth-force acceleration:  F₅ = −(1/2) ∇f_R
pub fn fofr_fifth_force(u: &[Real], dx: Real, n: usize) -> Vec<[Real; 3]> {
    let inv_4dx = 0.25 / dx;
    let mut f5 = vec![[0.0_f64; 3]; n * n * n];
    for k in 0..n as isize {
        for j in 0..n as isize {
            for i in 0..n as isize {
                let c = flat3(i, j, k, n);
                f5[c][0] = -(u[flat3(i+1,j,k,n)] - u[flat3(i-1,j,k,n)]) * inv_4dx;
                f5[c][1] = -(u[flat3(i,j+1,k,n)] - u[flat3(i,j-1,k,n)]) * inv_4dx;
                f5[c][2] = -(u[flat3(i,j,k+1,n)] - u[flat3(i,j,k-1,n)]) * inv_4dx;
            }
        }
    }
    f5
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::cosmology::Cosmology;

    fn cosmo() -> Cosmology { Cosmology::default() }

    /// In the screened limit (high density), f_R should approach the
    /// background value.
    #[test]
    fn fofr_background_value_today() {
        let cosmo = cosmo();
        let r0    = r_bar(1.0, &cosmo);
        let p     = FofrParams::default();
        let fofr_bg = fofr_of_r(r0, r0, &p);
        // f_{R0} = −n * f_{R0} * (R0/R0)^{n+1} = −n * f_{R0}
        let expected = -p.n * p.fofr0;
        assert!((fofr_bg - expected).abs() / p.fofr0 < 1e-10,
                "f_R(R0)={fofr_bg}, expected={expected}");
    }

    /// r_bar should be positive and decrease with time.
    #[test]
    fn rbar_evolution() {
        let cosmo = cosmo();
        let r1 = r_bar(0.5, &cosmo);
        let r2 = r_bar(1.0, &cosmo);
        assert!(r1 > r2, "R̄ should decrease as a increases");
        assert!(r2 > 0.0, "R̄ should be positive");
    }

    /// Solve in uniform low-density: f_R should converge to a uniform value.
    #[test]
    fn fofr_uniform_density() {
        let n  = 4_usize;
        let dx = 1.0_f64;
        let p  = FofrParams { fofr0: 1e-5, n: 1.0, epsilon: 1e-7,
                              max_iter: 200, n_smooth: 3 };
        let cosmo = cosmo();
        let rho = vec![1.0_f64; n * n * n]; // mean density
        let r0  = r_bar(1.0, &cosmo);
        let bg_val = fofr_of_r(r0, r0, &p);
        let u = solve_fofr(&rho, 1.0, dx, n, &p, &cosmo);
        // All cells should remain at background value (uniform problem)
        let spread = u.iter().map(|v| (v - bg_val).abs())
                      .fold(0.0_f64, f64::max);
        println!("f(R) spread around bg: {spread:.3e}");
        assert!(spread < p.fofr0, "Solution drifted too far from background");
    }
}
