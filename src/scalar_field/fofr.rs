//! Hu-Sawicki f(R) modified gravity solver.
//!
//! The scalaron  f_R = df/dR  satisfies the quasi-static EOM
//!
//!   ∇²f_R = (1/3)[δR(f_R) − δρ/M_Pl²]
//!
//! where  δR = R − R̄  and the background  R̄(a)  is the ΛCDM Ricci scalar.
//!
//! # Multigrid solver
//!
//! The nonlinear EOM is solved with the **Full Approximation Scheme (FAS)
//! V-cycle multigrid** using Newton-Gauss-Seidel (NGS) smoothing.
//! This mirrors the Symmetron solver in `scalar_field/symmetron.rs` and the
//! linear Poisson solver in `gravity/poisson.rs`.
//!
//! # Grid convention
//!
//! C-order (row-major): flat(i, j, k) = i·n² + j·n + k.
//! All inner loops iterate i→j→k (slowest to fastest) to maximise
//! cache-line reuse on the k-axis — see `crate::utils::types`.
//!
//! References: Hu & Sawicki (2007, Phys. Rev. D 76, 064004);
//! Llinares, Mota & Winther (2014, A&A 562 A78, §3.2).

use rayon::prelude::*;
use crate::gravity::poisson::{restrict_uniform, prolongate_uniform};
use crate::utils::types::{Real, idx3p, from_flat};
use crate::utils::cosmology::Cosmology;

/// Hu-Sawicki f(R) model parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FofrParams {
    /// |f_{R0}| — background field value today
    pub fofr0:    Real,
    /// Power-law index n (Hu-Sawicki)
    pub n:        Real,
    /// Multigrid convergence tolerance on ‖residual‖_∞
    pub epsilon:  Real,
    /// Maximum outer V-cycle iterations
    pub max_iter: u32,
    /// NGS sweeps per V-cycle level
    pub n_smooth: u32,
}

impl Default for FofrParams {
    fn default() -> Self {
        Self { fofr0: 1e-5, n: 1.0, epsilon: 1e-5, max_iter: 150, n_smooth: 2 }
    }
}

// ── Background quantities ──────────────────────────────────────────────────────

/// ΛCDM background Ricci scalar  R̄(a) in units of H₀².
///
/// R̄ ≈ 3H₀² [4Ω_Λ + Ω_m a⁻³]  (matter + Λ, no radiation).
#[inline]
pub fn r_bar(a: Real, cosmo: &Cosmology) -> Real {
    3.0 * (4.0 * cosmo.omega_lambda + cosmo.omega_m / (a * a * a))
}

/// Hu-Sawicki f_R evaluated at Ricci scalar R:
///   f_R(R) = −n |f_{R0}| (R̄₀/R)^{n+1}
#[inline]
fn fofr_of_r(r: Real, r0: Real, p: &FofrParams) -> Real {
    -p.n * p.fofr0 * (r0 / r).powf(p.n + 1.0)
}

/// Invert Hu-Sawicki: R from f_R.
///
/// f_R = −n f_{R0} (R₀/R)^{n+1}  →  R = R₀ (n f_{R0} / |u|)^{1/(n+1)}
///
/// Returns R̄(a) as a fallback for unphysical u ≥ 0 (screened regime).
#[inline]
fn r_of_fofr(u: Real, r0: Real, p: &FofrParams, rbar_a: Real) -> Real {
    if u >= 0.0 { return rbar_a; }
    r0 * (p.n * p.fofr0 / (-u)).powf(1.0 / (p.n + 1.0))
}

// ── Source function and analytical derivative ──────────────────────────────────

/// RHS of the f_R equation at a single cell.
///
///   S(u) = (1/3) [ R(u) − R̄(a)  −  Ω_m (ρ/ρ̄ − 1) a⁻³ ]
///
/// Returns `(S, R(u))` so the caller can reuse R(u) for the derivative
/// without recomputing it.
#[inline]
fn fofr_source(
    u:               Real,
    rho_over_rhobar: Real,
    a:               Real,
    r0:              Real,   // r_bar(1.0, cosmo) — precomputed by caller
    rbar_a:          Real,   // r_bar(a,   cosmo) — precomputed by caller
    cosmo:           &Cosmology,
    p:               &FofrParams,
) -> (Real, Real) {
    let r_u       = r_of_fofr(u, r0, p, rbar_a);
    let delta_r   = r_u - rbar_a;
    let delta_rho = (rho_over_rhobar - 1.0) * cosmo.omega_m / (a * a * a);
    ((delta_r - delta_rho) / 3.0, r_u)
}

/// Analytical derivative  dS/du.
///
/// For u < 0:
///   R(u) = r0 (n f_{R0} / |u|)^α  with  α = 1/(n+1)
///   dR/du = α R(u) / (−u)
///   dS/du = α R(u) / (3(−u))
///
/// For u ≥ 0 (screened):  R is clamped to R̄, so  dS/du = 0.
#[inline]
fn dfofr_source_du(u: Real, r_u: Real, p: &FofrParams) -> Real {
    if u >= 0.0 { return 0.0; }
    r_u / (3.0 * (p.n + 1.0) * (-u))
}

// ── Newton-Gauss-Seidel sweep ──────────────────────────────────────────────────

/// One red-black Newton-Gauss-Seidel sweep for  ∇²u = S(u).
///
/// Uses the Newton update at each cell:
///   u_new = u − (∇²u − S) / (−6/dx² − dS/du)
///
/// Loop order i→j→k with step-by-2 k-striding (no branch inside the k-loop)
/// to match the cache pattern documented in `crate::utils::types`.
fn fofr_ngs_sweep(
    u:      &mut Vec<Real>,
    rho:    &[Real],
    a:      Real,
    dx:     Real,
    n:      usize,
    r0:     Real,
    rbar_a: Real,
    p:      &FofrParams,
    cosmo:  &Cosmology,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap = u.clone();

    for color in 0..2_usize {
        for i in 0..n as isize {
            for j in 0..n as isize {
                // Step k by 2, starting at the correct parity — no branch inside.
                let k_start = ((i + j) as usize % 2 + color) % 2;
                let mut k = k_start as isize;
                while k < n as isize {
                    let c   = idx3p(i, j, k, n);
                    let u_c = snap[c];
                    let nb  = snap[idx3p(i+1, j,   k,   n)]
                            + snap[idx3p(i-1, j,   k,   n)]
                            + snap[idx3p(i,   j+1, k,   n)]
                            + snap[idx3p(i,   j-1, k,   n)]
                            + snap[idx3p(i,   j,   k+1, n)]
                            + snap[idx3p(i,   j,   k-1, n)];
                    let (src, r_u) = fofr_source(u_c, rho[c], a, r0, rbar_a, cosmo, p);
                    let dsrc  = dfofr_source_du(u_c, r_u, p);
                    let lap   = (nb - 6.0 * u_c) * inv_dx2;
                    let denom = -6.0 * inv_dx2 - dsrc;
                    // Enforce physical bound: f_R ≤ 0 for Hu-Sawicki
                    u[c] = (u_c - (lap - src) / denom).min(0.0);
                    k += 2;
                }
            }
        }
    }
}

// ── Residual ──────────────────────────────────────────────────────────────────

/// Pointwise residual  r_i = ∇²u_i − S(u_i).
///
/// Parallelised over i-rows so the k-axis inner loop is contiguous in cache.
fn fofr_residual(
    u:      &[Real],
    rho:    &[Real],
    a:      Real,
    dx:     Real,
    n:      usize,
    r0:     Real,
    rbar_a: Real,
    p:      &FofrParams,
    cosmo:  &Cosmology,
) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    res.par_chunks_mut(n * n).enumerate().for_each(|(i, row)| {
        let i = i as isize;
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c   = idx3p(i, j, k, n);
                let lap = (u[idx3p(i+1, j,   k,   n)]
                         + u[idx3p(i-1, j,   k,   n)]
                         + u[idx3p(i,   j+1, k,   n)]
                         + u[idx3p(i,   j-1, k,   n)]
                         + u[idx3p(i,   j,   k+1, n)]
                         + u[idx3p(i,   j,   k-1, n)]
                         - 6.0 * u[c]) * inv_dx2;
                let (src, _) = fofr_source(u[c], rho[c], a, r0, rbar_a, cosmo, p);
                row[j as usize * n + k as usize] = lap - src;
            }
        }
    });
    res
}

// ── FAS multigrid V-cycle ──────────────────────────────────────────────────────

/// One Full Approximation Scheme (FAS) V-cycle for the f(R) equation.
///
/// Architecture mirrors `v_cycle_sym()` in `symmetron.rs`:
///   pre-smooth → restrict → coarse FAS solve → prolongate → apply → post-smooth.
fn fofr_v_cycle(
    u:        &mut Vec<Real>,
    rho:      &[Real],
    a:        Real,
    dx:       Real,
    n:        usize,
    n_smooth: usize,
    min_n:    usize,
    r0:       Real,
    rbar_a:   Real,
    p:        &FofrParams,
    cosmo:    &Cosmology,
) {
    for _ in 0..n_smooth { fofr_ngs_sweep(u, rho, a, dx, n, r0, rbar_a, p, cosmo); }

    if n <= min_n {
        for _ in 0..4 * n_smooth { fofr_ngs_sweep(u, rho, a, dx, n, r0, rbar_a, p, cosmo); }
        return;
    }

    let rho_c   = restrict_uniform(rho, n);
    let u_c_old = restrict_uniform(u, n);
    let nc      = n / 2;
    let mut u_c = u_c_old.clone();
    // Clamp coarse initial guess to the physical bound f_R ≤ 0
    u_c.iter_mut().for_each(|v| { if *v > 0.0 { *v = -1e-30; } });

    fofr_v_cycle(&mut u_c, &rho_c, a, dx * 2.0, nc, n_smooth, min_n, r0, rbar_a, p, cosmo);

    // FAS correction: δu_c = u_c_new − u_c_old → prolongate to fine → add
    let delta_c: Vec<Real> = u_c.iter().zip(u_c_old.iter())
        .map(|(new, old)| new - old).collect();
    let delta_f = prolongate_uniform(&delta_c, nc);
    u.iter_mut().zip(delta_f.iter()).for_each(|(v, d)| {
        *v = (*v + d).min(0.0);  // maintain f_R ≤ 0
    });

    for _ in 0..n_smooth { fofr_ngs_sweep(u, rho, a, dx, n, r0, rbar_a, p, cosmo); }
}

// ── Public solver ──────────────────────────────────────────────────────────────

/// Solve the Hu-Sawicki f_R equation to convergence.
///
/// Initialises from the background value  f_R(R̄₀)  and returns the field.
/// Uses FAS V-cycle multigrid for O(N log N) convergence per outer iteration.
pub fn solve_fofr(
    rho:   &[Real],
    a:     Real,
    dx:    Real,
    n:     usize,
    p:     &FofrParams,
    cosmo: &Cosmology,
) -> Vec<Real> {
    let r0     = r_bar(1.0, cosmo);
    let rbar_a = r_bar(a, cosmo);
    let bg_val = fofr_of_r(r0, r0, p);
    let mut u  = vec![bg_val; n * n * n];
    let _depth = (n as f64).log2() as usize;

    for iter in 0..p.max_iter {
        let res     = fofr_residual(&u, rho, a, dx, n, r0, rbar_a, p, cosmo);
        let res_max = res.par_iter().map(|r| r.abs()).reduce(|| 0.0_f64, f64::max);
        log::debug!("f(R) iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon { break; }
        fofr_v_cycle(&mut u, rho, a, dx, n, p.n_smooth as usize, 4, r0, rbar_a, p, cosmo);
    }
    u
}

/// Compute fifth-force acceleration:  F₅ = −(1/2) ∇f_R.
///
/// Parallelised over cells; loop order follows the C-order convention.
pub fn fofr_fifth_force(u: &[Real], dx: Real, n: usize) -> Vec<[Real; 3]> {
    let inv_4dx = 0.25 / dx;
    let mut f5 = vec![[0.0_f64; 3]; n * n * n];
    f5.par_iter_mut().enumerate().for_each(|(c, fc)| {
        let (i, j, k) = from_flat(c, n);
        let (i, j, k) = (i as isize, j as isize, k as isize);
        *fc = [
            -(u[idx3p(i+1, j,   k,   n)] - u[idx3p(i-1, j,   k,   n)]) * inv_4dx,
            -(u[idx3p(i,   j+1, k,   n)] - u[idx3p(i,   j-1, k,   n)]) * inv_4dx,
            -(u[idx3p(i,   j,   k+1, n)] - u[idx3p(i,   j,   k-1, n)]) * inv_4dx,
        ];
    });
    f5
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::cosmology::Cosmology;

    fn cosmo() -> Cosmology { Cosmology::default() }

    /// f_R(R̄₀) = −n |f_{R0}|  (by definition of the HS model).
    #[test]
    fn fofr_background_value_today() {
        let cosmo   = cosmo();
        let r0      = r_bar(1.0, &cosmo);
        let p       = FofrParams::default();
        let fofr_bg = fofr_of_r(r0, r0, &p);
        let expected = -p.n * p.fofr0;
        assert!((fofr_bg - expected).abs() / p.fofr0 < 1e-10,
                "f_R(R0)={fofr_bg}, expected={expected}");
    }

    /// R̄ should be positive and decrease as a increases (universe expands).
    #[test]
    fn rbar_evolution() {
        let cosmo = cosmo();
        let r1 = r_bar(0.5, &cosmo);
        let r2 = r_bar(1.0, &cosmo);
        assert!(r1 > r2, "R̄ should decrease as a increases");
        assert!(r2 > 0.0, "R̄ should be positive");
    }

    /// Uniform mean density → field stays at background value.
    #[test]
    fn fofr_uniform_density() {
        let n     = 4_usize;
        let dx    = 1.0_f64;
        let p     = FofrParams { fofr0: 1e-5, n: 1.0, epsilon: 1e-7,
                                 max_iter: 200, n_smooth: 3 };
        let cosmo = cosmo();
        let rho   = vec![1.0_f64; n * n * n];
        let r0    = r_bar(1.0, &cosmo);
        let bg    = fofr_of_r(r0, r0, &p);
        let u     = solve_fofr(&rho, 1.0, dx, n, &p, &cosmo);
        let spread = u.iter().map(|v| (v - bg).abs()).fold(0.0_f64, f64::max);
        println!("f(R) spread around bg: {spread:.3e}");
        assert!(spread < p.fofr0, "Solution drifted too far from background");
    }
}
