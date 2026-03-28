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
/// The quasi-static comoving equation with positions in Mpc/h is
///
///   ∇²f_R = (H₀ a / c)² / 3 × [ R(f_R) − R̄(a)  −  3 Ω_m (ρ/ρ̄ − 1) a⁻³ ]
///
/// where H₀ = 100 km/s/(Mpc/h) and c = 299 792 km/s.  The prefactor
/// (H₀ a/c)² converts the dimensionless Ricci-scalar terms (in H₀ = 1
/// units) so that ∇²f_R is in (Mpc/h)⁻², consistent with grid spacing dx
/// in Mpc/h.  Without this factor the effective Compton wavelength is
/// inflated by c/H₀ ≈ 3000, which screens the field on every grid cell.
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
    // H₀ = 100 km/s/(Mpc/h), c in km/s — conversion factor for code units.
    const H0_CODE: Real = 100.0;
    const C_CODE:  Real = 299_792.458;
    let prefac    = (H0_CODE * a / C_CODE).powi(2);
    let r_u       = r_of_fofr(u, r0, p, rbar_a);
    let delta_r   = r_u - rbar_a;
    // Factor 3 from 8πG ρ̄_m = 3 H₀² Ω_m / a³ (compare Poisson source).
    let delta_rho = 3.0 * (rho_over_rhobar - 1.0) * cosmo.omega_m / (a * a * a);
    (prefac * (delta_r - delta_rho) / 3.0, r_u)
}

/// Analytical derivative  dS/du.
///
/// For u < 0:
///   R(u) = r0 (n f_{R0} / |u|)^α  with  α = 1/(n+1)
///   dR/du = R(u) / ((n+1)(−u))
///   dS/du = prefac × R(u) / (3(n+1)(−u))   where prefac = (H₀ a/c)²
///
/// For u ≥ 0 (screened):  R is clamped to R̄, so  dS/du = 0.
///
/// `a` is the scale factor; `prefac` = (H0_CODE × a / C_CODE)² and must
/// match the prefactor computed in `fofr_source`.
#[inline]
fn dfofr_source_du(u: Real, r_u: Real, a: Real, p: &FofrParams) -> Real {
    if u >= 0.0 { return 0.0; }
    const H0_CODE: Real = 100.0;
    const C_CODE:  Real = 299_792.458;
    let prefac = (H0_CODE * a / C_CODE).powi(2);
    prefac * r_u / (3.0 * (p.n + 1.0) * (-u))
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
                    let dsrc  = dfofr_source_du(u_c, r_u, a, p);
                    let lap   = (nb - 6.0 * u_c) * inv_dx2;
                    let denom = -6.0 * inv_dx2 - dsrc;
                    // Enforce physical bound: f_R ≤ 0 for Hu-Sawicki.
                    // Guard against degenerate Jacobian (near u → 0⁻).
                    if denom.abs() > 1e-30 {
                        u[c] = (u_c - (lap - src) / denom).min(0.0);
                    }
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

    for iter in 0..p.max_iter {
        let res     = fofr_residual(&u, rho, a, dx, n, r0, rbar_a, p, cosmo);
        let res_max = res.par_iter().map(|r| r.abs()).reduce(|| 0.0_f64, f64::max);
        log::debug!("f(R) iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon { break; }
        fofr_v_cycle(&mut u, rho, a, dx, n, p.n_smooth as usize, 4, r0, rbar_a, p, cosmo);
    }
    u
}

/// Compute fifth-force acceleration:  a₅ = +(1/2) ∇f_R.
///
/// In Hu-Sawicki f(R) gravity the extra acceleration on a test particle is
///   a₅ = (1/2) ∇f_R
/// (Llinares, Mota & Winther 2014, A&A 562 A78, eq. 3.3).
/// Because f_R ≤ 0 is LESS negative at overdensities, ∇f_R points
/// toward dense regions, making a₅ attractive — consistent with gravity.
///
/// Parallelised over cells; loop order follows the C-order convention.
pub fn fofr_fifth_force(u: &[Real], dx: Real, n: usize) -> Vec<[Real; 3]> {
    let inv_4dx = 0.25 / dx;
    let mut f5 = vec![[0.0_f64; 3]; n * n * n];
    f5.par_iter_mut().enumerate().for_each(|(c, fc)| {
        let (i, j, k) = from_flat(c, n);
        let (i, j, k) = (i as isize, j as isize, k as isize);
        *fc = [
            (u[idx3p(i+1, j,   k,   n)] - u[idx3p(i-1, j,   k,   n)]) * inv_4dx,
            (u[idx3p(i,   j+1, k,   n)] - u[idx3p(i,   j-1, k,   n)]) * inv_4dx,
            (u[idx3p(i,   j,   k+1, n)] - u[idx3p(i,   j,   k-1, n)]) * inv_4dx,
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

    #[test]
    fn fofr_screening_high_density() {
        // In a strongly overdense region the chameleon mechanism forces u → 0.
        // Physical bound: f_R ≤ 0 must be maintained everywhere.
        let ng    = 8_usize;
        let cosmo = Cosmology::default();
        let p     = FofrParams { fofr0: 1e-5, n: 1.0, epsilon: 1e-8,
                                 max_iter: 200, n_smooth: 2 };
        let rho   = vec![1000.0_f64; ng * ng * ng];   // 1000× overdense
        let u     = solve_fofr(&rho, 1.0, 100.0 / ng as Real, ng, &p, &cosmo);
        assert!(
            u.iter().all(|&v| v <= 0.0),
            "f_R must stay ≤ 0 everywhere; max = {:.3e}",
            u.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
        // At 1000× overdensity the field should be driven very close to zero
        // (well-screened); a mean far below zero would indicate the solver
        // is not honouring the chameleon bound.
        let mean_u = u.iter().sum::<Real>() / u.len() as Real;
        assert!(mean_u > -1e-7,
            "screened field should be near zero; mean = {mean_u:.3e}");
    }

    #[test]
    fn fofr_fifth_force_direction() {
        // An overdense plane at i=0 has f_R LESS negative (screened) than the
        // background.  The fifth force a₅ = +(1/2)∇f_R therefore points
        // TOWARD the overdense plane (attractive).
        //
        // At i=1 (right of the spike), ∇f_R points in -x toward the less-negative
        // peak, so a₅[x] < 0 (toward i=0).  By symmetry at i=ng-1 (left), a₅[x] > 0.
        //
        // We use a moderate overdensity so the spike is not fully screened.
        let ng    = 16_usize;
        let cosmo = Cosmology::default();
        // fofr0 = 1e-4 (not 1e-5): keeps the spike partially unscreened
        // so the gradient has a well-defined sign.  At 1e-5 a 5× spike would
        // be nearly fully screened and ∇f_R ≈ 0.
        let p     = FofrParams { fofr0: 1e-4, n: 1.0, epsilon: 1e-8,
                                 max_iter: 200, n_smooth: 2 };
        let dx    = 100.0_f64 / ng as Real;
        let mut rho = vec![1.0_f64; ng * ng * ng];
        // Moderate density enhancement along the i=0 plane.
        for j in 0..ng { for k in 0..ng { rho[j * ng + k] = 5.0; } }
        let u  = solve_fofr(&rho, 1.0, dx, ng, &p, &cosmo);
        let f5 = fofr_fifth_force(&u, dx, ng);
        let j0 = ng / 2;
        let k0 = ng / 2;
        // At i=1: force must point toward i=0 (negative x direction).
        let idx1 = 1 * ng * ng + j0 * ng + k0;
        assert!(
            f5[idx1][0] < 0.0,
            "fifth force at i=1 should point toward density peak at i=0 (-x); got {:.3e}",
            f5[idx1][0]
        );
        // At i=ng-1 (left neighbour under periodic BC): force points in +x.
        let idx_left = (ng - 1) * ng * ng + j0 * ng + k0;
        assert!(
            f5[idx_left][0] > 0.0,
            "fifth force at i=ng-1 should point toward density peak at i=0 (+x); got {:.3e}",
            f5[idx_left][0]
        );
    }

    #[test]
    fn fofr_solver_converges_to_background() {
        // With a mild density perturbation the field at UNPERTURBED cells should
        // stay close to the background value u₀ = -n·fofr0 = -1e-5.
        // Compton wavelength ≈ 4 Mpc/h; dx = 100/8 = 12.5 Mpc/h >> λ_C, so the
        // perturbation from the overdense plane decays to < 10% of background
        // before reaching the nearest unperturbed cells.
        let ng       = 8_usize;
        let cosmo    = Cosmology::default();
        let dx       = 100.0_f64 / ng as Real;
        let expected = -1.0_f64 * 1e-5_f64;   // -n * fofr0

        // Small perturbation: one plane at i=0 is 2× denser, rest uniform.
        let mut rho = vec![1.0_f64; ng * ng * ng];
        for j in 0..ng { for k in 0..ng { rho[j * ng + k] = 2.0; } }

        let p = FofrParams { fofr0: 1e-5, n: 1.0, epsilon: 1e-10,
                             max_iter: 300, n_smooth: 4 };
        let u = solve_fofr(&rho, 1.0, dx, ng, &p, &cosmo);

        // Measure error only at the unperturbed cells (i != 0).
        let mut err_sum = 0.0_f64;
        let mut count   = 0_usize;
        for i in 1..ng {
            for j in 0..ng {
                for k in 0..ng {
                    err_sum += (u[i * ng * ng + j * ng + k] - expected).abs();
                    count   += 1;
                }
            }
        }
        let err = err_sum / count as Real;

        // Allow up to 2% deviation from background at unperturbed cells.
        // The physical perturbation ∝ exp(-dx/λ_C) ≈ exp(-2.9) ≈ 5% at i=1.
        assert!(err < 2e-2 * p.fofr0.abs(),
            "unperturbed cells should stay near background: err={err:.3e}, bg={expected:.3e}");
    }
}
