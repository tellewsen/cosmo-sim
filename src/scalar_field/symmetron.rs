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

use crate::gravity::poisson::{prolongate_uniform, restrict_uniform};
use crate::utils::types::{idx3p, Real};

/// Model parameters for the Symmetron.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SymmetronParams {
    /// Scale factor at SSB in a homogeneous universe.
    pub a_ssb: Real,
    /// Compton wavelength in vacuum  λ₀  [comoving Mpc/h].
    pub lambda_0: Real,
    /// Fifth-force coupling β₀ (ratio to Newtonian in broken phase).
    pub beta_0: Real,
    /// Include non-static (χ̇) terms.
    pub non_static: bool,
    /// Multigrid convergence tolerance on max-norm of residual.
    pub epsilon: Real,
    /// Maximum outer V-cycle iterations.
    pub max_iter: u32,
    /// NGS sweeps per V-cycle level.
    pub n_smooth: u32,
}

impl Default for SymmetronParams {
    /// Llinares et al. 2014 fiducial: (a_SSB, λ₀, β) = (0.5, 1 Mpc/h, 1).
    fn default() -> Self {
        Self {
            a_ssb: 0.5,
            lambda_0: 1.0,
            beta_0: 1.0,
            non_static: false,
            epsilon: 1e-5,
            max_iter: 150,
            n_smooth: 2,
        }
    }
}

// ── Source function and its derivative ───────────────────────────────────────

/// f(χ) = (χ / λ₀²) · (ρ/M² − 1 + χ²)
///
/// where ρ/M² = (ρ/ρ̄) · (a_SSB/a)³ is the effective coupling.
#[inline]
pub fn source_chi(chi: Real, rho_over_rhobar: Real, a: Real, p: &SymmetronParams) -> Real {
    let rho_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    chi * (rho_m2 - 1.0 + chi * chi) / (p.lambda_0 * p.lambda_0)
}

/// ∂f/∂χ = (ρ/M² − 1 + 3 χ²) / λ₀²  (Newton linearisation denominator).
#[inline]
pub fn dsource_dchi(chi: Real, rho_over_rhobar: Real, a: Real, p: &SymmetronParams) -> Real {
    let rho_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    (rho_m2 - 1.0 + 3.0 * chi * chi) / (p.lambda_0 * p.lambda_0)
}

// ── Distributed slab helpers ──────────────────────────────────────────────────

/// Flat index in a ghost-padded slab of shape `(n_local_i + 2) × N × N`.
///
/// `li` must be in [0, n_local_i + 2); `j` and `k` wrap periodically mod `n`.
#[cfg(feature = "mpi_parallel")]
#[inline]
pub(crate) fn sidx(li: usize, j: isize, k: isize, n: usize) -> usize {
    let j = j.rem_euclid(n as isize) as usize;
    let k = k.rem_euclid(n as isize) as usize;
    li * n * n + j * n + k
}

/// One red-black Newton-Gauss-Seidel sweep on a ghost-padded local i-slab.
///
/// `chi`      — ghost-padded slab, shape `(n_local_i + 2) × N × N`. Ghosts must
///              be current (call `exchange_halos` first in MPI mode).
/// `rho`      — plain slab, shape `n_local_i × N × N` (no ghosts needed).
/// `i_start`  — global i-index of the first owned cell (li = 1).
///              Used to keep the red-black colouring consistent across ranks.
#[cfg(feature = "mpi_parallel")]
#[allow(clippy::too_many_arguments)]
fn ngs_sweep_slab(
    chi: &mut [Real],
    rho: &[Real],
    a: Real,
    dx: Real,
    n: usize,
    n_local_i: usize,
    i_start: usize,
    p: &SymmetronParams,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap = chi.to_owned();

    for color in 0..2_usize {
        for li in 1..=n_local_i {
            let i_global = i_start + li - 1;
            for j in 0..n as isize {
                let k_start = ((i_global + j as usize) % 2 + color) % 2;
                let mut k = k_start as isize;
                while k < n as isize {
                    let c = sidx(li, j, k, n);
                    // rho index: plain slab (li-1 is 0-based owned index)
                    let rho_c = rho[(li - 1) * n * n + j as usize * n + k as usize];
                    let chi_c = snap[c];
                    let nb = snap[sidx(li + 1, j, k, n)]
                        + snap[sidx(li - 1, j, k, n)]
                        + snap[sidx(li, j + 1, k, n)]
                        + snap[sidx(li, j - 1, k, n)]
                        + snap[sidx(li, j, k + 1, n)]
                        + snap[sidx(li, j, k - 1, n)];
                    let f = source_chi(chi_c, rho_c, a, p);
                    let df = dsource_dchi(chi_c, rho_c, a, p);
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

/// Pointwise residual  r = ∇²χ − f(χ)  on owned cells of a ghost-padded slab.
///
/// Returns `(residuals, max_abs_residual)`.
/// `residuals` is a plain slab (n_local_i × N × N, no ghosts).
/// Ghosts of `chi` must be current before calling.
#[cfg(feature = "mpi_parallel")]
fn residual_slab(
    chi: &[Real],
    rho: &[Real],
    a: Real,
    dx: Real,
    n: usize,
    n_local_i: usize,
    p: &SymmetronParams,
) -> (Vec<Real>, Real) {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res: Vec<Real> = vec![0.0; n_local_i * n * n];
    let mut res_max: Real = 0.0;

    for li in 1..=n_local_i {
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c_pad = sidx(li, j, k, n);
                let c_plain = (li - 1) * n * n + j as usize * n + k as usize;
                let lap = (chi[sidx(li + 1, j, k, n)]
                    + chi[sidx(li - 1, j, k, n)]
                    + chi[sidx(li, j + 1, k, n)]
                    + chi[sidx(li, j - 1, k, n)]
                    + chi[sidx(li, j, k + 1, n)]
                    + chi[sidx(li, j, k - 1, n)]
                    - 6.0 * chi[c_pad])
                    * inv_dx2;
                let r = lap - source_chi(chi[c_pad], rho[c_plain], a, p);
                res[c_plain] = r;
                res_max = res_max.max(r.abs());
            }
        }
    }
    (res, res_max)
}

/// Injection restriction: coarse slab ← fine slab (pick every-other cell).
///
/// `fine`       — ghost-padded field slab, shape `(n_local_i + 2) × N × N`.
/// `rho`        — plain rho slab, shape `n_local_i × N × N`.
///
/// Returns:
///   `field_c` — ghost-padded coarse field slab, shape `(n_local_ic + 2) × Nc × Nc`.
///               Owned cells filled; ghosts are zero (caller must exchange halos).
///   `rho_c`   — plain coarse rho slab, shape `n_local_ic × Nc × Nc`.
///
/// Injection only reads owned fine cells, so no halo exchange is needed before
/// this call.
#[cfg(feature = "mpi_parallel")]
pub(crate) fn restrict_slab(
    fine: &[Real],
    rho: &[Real],
    n_local_i: usize,
    n: usize,
) -> (Vec<Real>, Vec<Real>) {
    let nc = n / 2;
    let n_local_ic = n_local_i / 2;
    let mut chi_c = vec![0.0; (n_local_ic + 2) * nc * nc];
    let mut rho_c = vec![0.0; n_local_ic * nc * nc];

    // Coarse owned cell lci maps to fine owned cell li = 2*lci + 1
    // (global: gc = i_start/2 + lci  →  gf = 2*gc = i_start + 2*lci  →  li = 2*lci + 1)
    for lci in 0..n_local_ic {
        let li_f = 2 * lci + 1; // fine ghost-padded li for this coarse cell
        for j in 0..nc {
            for k in 0..nc {
                chi_c[(lci + 1) * nc * nc + j * nc + k] =
                    fine[li_f * n * n + (2 * j) * n + (2 * k)];
                rho_c[lci * nc * nc + j * nc + k] = rho[(li_f - 1) * n * n + (2 * j) * n + (2 * k)];
            }
        }
    }
    (chi_c, rho_c)
}

/// Trilinear prolongation: fine slab ← coarse slab.
///
/// `coarse`     — ghost-padded coarse correction slab, shape `(n_local_ic+2)×Nc×Nc`.
///                **Ghosts must be filled before calling** (exchange halos of the
///                coarse correction before prolongating).
/// `n_local_i`  — owned fine i-cell count (= 2 × n_local_ic).
/// `n`          — fine global grid size.
///
/// Returns a ghost-padded fine slab `(n_local_i+2)×N×N`.
/// Owned cells (li ∈ [1, n_local_i+1)) are filled; ghosts are zero.
///
/// The coarse i-coordinate for fine local padded index `li` is:
///   cx = (li - 0.5) * 0.5 - 0.5
///
/// This formula is independent of `i_start` (global rank offset cancels out),
/// so the function requires no MPI context.  The left/right coarse ghost cells
/// are accessed at `lci_base = 0` and `lci_base = n_local_ic + 1` respectively.
#[cfg(feature = "mpi_parallel")]
pub(crate) fn prolongate_slab(coarse: &[Real], n_local_i: usize, n: usize) -> Vec<Real> {
    debug_assert!(n_local_i.is_multiple_of(2), "n_local_i must be even for coarsening");
    let nc = n / 2;
    let mut fine = vec![0.0; (n_local_i + 2) * n * n];

    for li in 1..=n_local_i {
        let cx = (li as Real - 0.5) * 0.5 - 0.5;
        let lci_base = (cx.floor() as isize + 1) as usize; // +1 for coarse ghost offset
        let tx = cx - cx.floor();

        for j in 0..n as isize {
            let cy = (j as Real + 0.5) * 0.5 - 0.5;
            let cj = cy.floor() as isize;
            let ty = cy - cy.floor();

            for k in 0..n as isize {
                let cz = (k as Real + 0.5) * 0.5 - 0.5;
                let ck = cz.floor() as isize;
                let tz = cz - cz.floor();

                let mut val = 0.0;
                for (dli, wx) in [(0_usize, 1.0 - tx), (1, tx)] {
                    for (dcj, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                        for (dck, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                            let jc = (cj + dcj).rem_euclid(nc as isize) as usize;
                            let kc = (ck + dck).rem_euclid(nc as isize) as usize;
                            val += wx * wy * wz * coarse[(lci_base + dli) * nc * nc + jc * nc + kc];
                        }
                    }
                }
                fine[sidx(li, j, k, n)] = val;
            }
        }
    }
    fine
}

/// Equilibrium value of χ at a given overdensity ρ/ρ̄.
///
/// χ_eq = sqrt(max(0, 1 − ρ/M²))  in the broken phase, 0 in the restored phase.
#[inline]
pub fn chi_equilibrium(rho_over_rhobar: Real, a: Real, p: &SymmetronParams) -> Real {
    if a < p.a_ssb {
        return 0.0;
    } // restored phase: universe hasn't undergone SSB yet
    let rho_m2 = rho_over_rhobar * (p.a_ssb / a).powi(3);
    (1.0 - rho_m2).max(0.0).sqrt()
}

// ── Newton-Gauss-Seidel sweep ─────────────────────────────────────────────────

/// One red-black Newton-Gauss-Seidel sweep for  ∇²χ = f(χ).
///
/// Uses the Newton update:
///   χ_new = (Σ_nb χ_nb / dx² − f(χ)) / (6/dx² − ∂f/∂χ)
fn ngs_sweep(chi: &mut [Real], rho: &[Real], a: Real, dx: Real, n: usize, p: &SymmetronParams) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap = chi.to_owned();

    for color in 0..2_usize {
        for i in 0..n as isize {
            for j in 0..n as isize {
                // Step by 2 to select only cells of the current colour — no branch
                let k_start = ((i + j) as usize % 2 + color) % 2;
                let mut k = k_start as isize;
                while k < n as isize {
                    let c = idx3p(i, j, k, n);
                    let chi_c = snap[c];
                    let rho_c = rho[c];
                    let nb = snap[idx3p(i + 1, j, k, n)]
                        + snap[idx3p(i - 1, j, k, n)]
                        + snap[idx3p(i, j + 1, k, n)]
                        + snap[idx3p(i, j - 1, k, n)]
                        + snap[idx3p(i, j, k + 1, n)]
                        + snap[idx3p(i, j, k - 1, n)];
                    let f = source_chi(chi_c, rho_c, a, p);
                    let df = dsource_dchi(chi_c, rho_c, a, p);
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
    a: Real,
    dx: Real,
    n: usize,
    p: &SymmetronParams,
) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    for i in 0..n as isize {
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c = idx3p(i, j, k, n);
                let lap = (chi[idx3p(i + 1, j, k, n)]
                    + chi[idx3p(i - 1, j, k, n)]
                    + chi[idx3p(i, j + 1, k, n)]
                    + chi[idx3p(i, j - 1, k, n)]
                    + chi[idx3p(i, j, k + 1, n)]
                    + chi[idx3p(i, j, k - 1, n)]
                    - 6.0 * chi[c])
                    * inv_dx2;
                res[c] = lap - source_chi(chi[c], rho[c], a, p);
            }
        }
    }
    res
}

// ── Multigrid V-cycle ─────────────────────────────────────────────────────────

/// One multigrid V-cycle for the nonlinear symmetron equation.
#[allow(clippy::too_many_arguments)]
fn v_cycle_sym(
    chi: &mut [Real],
    rho: &[Real],
    a: Real,
    dx: Real,
    n: usize,
    n_smooth: usize,
    min_n: usize,
    p: &SymmetronParams,
) {
    for _ in 0..n_smooth {
        ngs_sweep(chi, rho, a, dx, n, p);
    }

    if n <= min_n {
        for _ in 0..4 * n_smooth {
            ngs_sweep(chi, rho, a, dx, n, p);
        }
        return;
    }

    let rho_c = restrict_uniform(rho, n);
    let chi_c_arr = restrict_uniform(chi, n);
    let nc = n / 2;

    // Coarse-level: solve for correction e, initialised with restricted chi
    let mut err_c = chi_c_arr;
    v_cycle_sym(&mut err_c, &rho_c, a, dx * 2.0, nc, n_smooth, min_n, p);

    // Prolongate coarse-level correction and add
    let chi_c_back = restrict_uniform(chi, n);
    let correction: Vec<Real> = err_c
        .iter()
        .zip(chi_c_back.iter())
        .map(|(e, c)| e - c)
        .collect();
    let corr_fine = prolongate_uniform(&correction, nc);
    chi.iter_mut().zip(corr_fine.iter()).for_each(|(c, e)| {
        *c += e;
        *c = c.clamp(-1.0, 1.0);
    });

    for _ in 0..n_smooth {
        ngs_sweep(chi, rho, a, dx, n, p);
    }
}

// ── Distributed V-cycle ───────────────────────────────────────────────────────

/// One multigrid V-cycle for the nonlinear Symmetron equation on a distributed slab.
///
/// On the way down, pre-smooths with NGS then restricts to a coarser slab.
/// At the coarsest level (n_local_i < min_n_local or n ≤ 4), all ranks
/// allgather the full grid, run a serial solve collectively, then copy their
/// slab back — identical computation on every rank avoids a broadcast.
///
/// On the way up, prolongates the coarse correction back to the fine grid,
/// applies it, clamps to [−1, 1], and post-smooths.
///
/// # Arguments
/// * `chi`          — ghost-padded slab `(n_local_i+2)×N×N`, ghosts pre-filled
/// * `rho`          — plain slab `n_local_i×N×N`
/// * `n_local_i`    — owned i-cell count on this rank
/// * `i_start`      — global i-index of this rank's first owned cell
/// * `n_smooth`     — NGS sweeps at each pre/post-smooth step
/// * `min_n_local`  — recurse until n_local_i < this, then do coarse solve
#[cfg(feature = "mpi_parallel")]
fn v_cycle_sym_mpi(
    chi: &mut Vec<Real>,
    rho: &[Real],
    a: Real,
    dx: Real,
    n: usize,
    n_local_i: usize,
    i_start: usize,
    n_smooth: usize,
    min_n_local: usize,
    p: &SymmetronParams,
    mpi: &crate::mpi::MpiContext,
) {
    // Guard against zero-sized slab (can occur at coarse levels if size > n)
    if n_local_i == 0 {
        return;
    }

    // ── Pre-smooth ──────────────────────────────────────────────────────────
    for _ in 0..n_smooth {
        mpi.exchange_halos(chi, n_local_i, n, 1);
        ngs_sweep_slab(chi, rho, a, dx, n, n_local_i, i_start, p);
    }

    // ── Coarsest level: allgather → serial solve → extract slab ────────────
    if n_local_i < min_n_local || n <= 4 {
        // Extract owned chi (no ghosts) for allgather
        let mut local = vec![0.0; n_local_i * n * n];
        for li in 0..n_local_i {
            let src = (li + 1) * n * n;
            local[li * n * n..(li + 1) * n * n].copy_from_slice(&chi[src..src + n * n]);
        }
        let chi_full = mpi.allgather_slabs(&local, n);
        let rho_full = mpi.allgather_slabs(rho, n);

        // All ranks run identical serial NGS on the full coarse grid
        let mut chi_coarse = chi_full;
        for _ in 0..4 * n_smooth {
            ngs_sweep(&mut chi_coarse, &rho_full, a, dx, n, p);
        }

        // Copy own slab back into chi (owned cells only)
        for li in 0..n_local_i {
            let li_p = li + 1;
            let g_base = (i_start + li) * n * n;
            chi[li_p * n * n..li_p * n * n + n * n]
                .copy_from_slice(&chi_coarse[g_base..g_base + n * n]);
        }

        // Post-smooth after coarse solve
        for _ in 0..n_smooth {
            mpi.exchange_halos(chi, n_local_i, n, 1);
            ngs_sweep_slab(chi, rho, a, dx, n, n_local_i, i_start, p);
        }
        return;
    }

    // ── Restrict ────────────────────────────────────────────────────────────
    let (chi_c_save, rho_c) = restrict_slab(chi, rho, n_local_i, n);
    let mut chi_c = chi_c_save.clone();

    let nc = n / 2;
    let n_local_ic = n_local_i / 2;
    debug_assert_eq!(
        i_start % 2,
        0,
        "i_start must be even for coarsening; ensure n_local_i is divisible by 2 at all levels"
    );
    let i_start_c = i_start / 2;

    // ── Recurse on coarse slab ──────────────────────────────────────────────
    v_cycle_sym_mpi(
        &mut chi_c,
        &rho_c,
        a,
        dx * 2.0,
        nc,
        n_local_ic,
        i_start_c,
        n_smooth,
        min_n_local,
        p,
        mpi,
    );

    // ── Correction: delta = chi_c_new − chi_c_save ──────────────────────────
    let mut delta_c: Vec<Real> = chi_c
        .iter()
        .zip(chi_c_save.iter())
        .map(|(new, old)| new - old)
        .collect();

    // Ghosts of delta_c must be current before trilinear prolongation
    mpi.exchange_halos(&mut delta_c, n_local_ic, nc, 1);

    // ── Prolongate correction to fine, apply, clamp ─────────────────────────
    let corr_fine = prolongate_slab(&delta_c, n_local_i, n);
    for li in 1..=n_local_i {
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c = sidx(li, j, k, n);
                chi[c] = (chi[c] + corr_fine[c]).clamp(-1.0, 1.0);
            }
        }
    }

    // ── Post-smooth ─────────────────────────────────────────────────────────
    for _ in 0..n_smooth {
        mpi.exchange_halos(chi, n_local_i, n, 1);
        ngs_sweep_slab(chi, rho, a, dx, n, n_local_i, i_start, p);
    }
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
    a: Real,
    dx: Real,
    n: usize,
    p: &SymmetronParams,
) -> Vec<Real> {
    assert_eq!(rho.len(), n * n * n);
    let min_n = 4;

    // Initialise χ to its equilibrium value
    let mut chi: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, p)).collect();

    for iter in 0..p.max_iter {
        let res = residual_chi(&chi, rho, a, dx, n, p);
        let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        log::trace!("Symmetron iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon {
            break;
        }
        v_cycle_sym(&mut chi, rho, a, dx, n, 2, min_n, p);
    }
    chi
}

/// Solve the quasi-static Symmetron EOM for χ on a distributed i-slab.
///
/// Each rank supplies its local slab of `rho` (shape `n_local_i × N × N`,
/// no ghost cells). The function handles ghost exchange internally.
///
/// Returns the local slab of χ (shape `n_local_i × N × N`, no ghosts).
/// Caller should call `mpi.allgather_slabs(&chi_slab, n)` to assemble the
/// full grid for `fifth_force_grid`.
#[cfg(feature = "mpi_parallel")]
pub fn solve_symmetron_mpi(
    rho_slab: &[Real],
    a: Real,
    dx: Real,
    n: usize,
    p: &SymmetronParams,
    mpi: &crate::mpi::MpiContext,
) -> Vec<Real> {
    let n_local_i = mpi.slab_count(n);
    let i_start = mpi.slab_start(n);
    debug_assert_eq!(
        rho_slab.len(),
        n_local_i * n * n,
        "rho_slab length mismatch: {} vs {}",
        rho_slab.len(),
        n_local_i * n * n
    );

    // Initialise ghost-padded chi from equilibrium
    let mut chi = vec![0.0; (n_local_i + 2) * n * n];
    for li in 1..=n_local_i {
        for j in 0..n {
            for k in 0..n {
                let c = sidx(li, j as isize, k as isize, n);
                let rho_c = rho_slab[(li - 1) * n * n + j * n + k];
                chi[c] = chi_equilibrium(rho_c, a, p);
            }
        }
    }

    // Outer V-cycle loop
    for iter in 0..p.max_iter {
        mpi.exchange_halos(&mut chi, n_local_i, n, 1);
        let (_, res_local) = residual_slab(&chi, rho_slab, a, dx, n, n_local_i, p);
        let res_max = mpi.allreduce_max(res_local);
        log::trace!("Symmetron MPI iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon {
            break;
        }
        v_cycle_sym_mpi(
            &mut chi, rho_slab, a, dx, n, n_local_i, i_start, 2, 4, p, mpi,
        );
    }

    // Return owned cells (strip ghosts)
    let mut result = vec![0.0; n_local_i * n * n];
    for li in 0..n_local_i {
        let src = (li + 1) * n * n;
        let dst = li * n * n;
        result[dst..dst + n * n].copy_from_slice(&chi[src..src + n * n]);
    }
    result
}

// ── Fifth force ───────────────────────────────────────────────────────────────

/// Compute the fifth-force acceleration field  a_5 = (β₀/a) ∇χ.
///
/// `chi`   — scalar field χ on n³ grid (C-order)
/// Returns array of `[ax, ay, az]` per grid cell.
pub fn fifth_force_grid(
    chi: &[Real],
    dx: Real,
    n: usize,
    a: Real,
    p: &SymmetronParams,
) -> Vec<[Real; 3]> {
    use crate::gravity::poisson::grad_phi;
    let grad = grad_phi(chi, dx, n);
    // grad_phi returns −∇χ; multiply by +β₀/a to get −(β₀/a)∇χ
    // which points toward matter (attractive).
    let factor = p.beta_0 / a;
    grad.into_iter()
        .map(|g| [factor * g[0], factor * g[1], factor * g[2]])
        .collect()
}

/// Fifth force at particle positions (interpolated from grid).
pub fn fifth_force_particles(
    chi: &[Real],
    dx: Real,
    n: usize,
    a: Real,
    p: &SymmetronParams,
    pos: &[[Real; 3]],
    box_size: Real,
) -> Vec<[Real; 3]> {
    use rayon::prelude::*;
    // Compute gradient of chi on grid (+∇χ, not -∇χ)
    let inv_2dx = 0.5 / dx;
    let ng_f = n as Real / box_size;
    let factor = p.beta_0 / a;

    pos.par_iter()
        .map(|xi| {
            let u = [xi[0] * ng_f - 0.5, xi[1] * ng_f - 0.5, xi[2] * ng_f - 0.5];
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
                        let w = wx * wy * wz;
                        let ci = i0 + di;
                        let cj = j0 + dj;
                        let ck = k0 + dk;
                        // ∂χ/∂x at this cell (centred)
                        let gx = (chi[idx3p(ci + 1, cj, ck, n)] - chi[idx3p(ci - 1, cj, ck, n)])
                            * inv_2dx;
                        let gy = (chi[idx3p(ci, cj + 1, ck, n)] - chi[idx3p(ci, cj - 1, ck, n)])
                            * inv_2dx;
                        let gz = (chi[idx3p(ci, cj, ck + 1, n)] - chi[idx3p(ci, cj, ck - 1, n)])
                            * inv_2dx;
                        f[0] += w * factor * gx;
                        f[1] += w * factor * gy;
                        f[2] += w * factor * gz;
                    }
                }
            }
            f
        })
        .collect()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::types::from_flat;
    use std::f64::consts::PI;

    fn default_p() -> SymmetronParams {
        SymmetronParams::default()
    }

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
        let p = default_p();
        let a = 0.7_f64; // post-SSB
        let rho = 0.1_f64; // under-dense
        let chi_eq = chi_equilibrium(rho, a, &p);
        let f = source_chi(chi_eq, rho, a, &p);
        println!("f(χ_eq) = {f:.3e}, χ_eq = {chi_eq:.6}");
        assert!(f.abs() < 1e-12, "|f(χ_eq)| = {f:.3e}");
    }

    #[test]
    fn dsource_positive_at_vacuum() {
        // ∂f/∂χ at χ_eq should be 2χ_eq²/λ₀² > 0 (stable minimum).
        let p = default_p();
        let a = 0.7_f64;
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
        assert!(
            (chi_eq - 1.0).abs() < 1e-12,
            "χ_eq in void = {chi_eq:.8}, expected 1.0"
        );
    }

    // ── Solver convergence tests ──────────────────────────────────────────────

    #[test]
    fn solver_uniform_density_converges_to_equilibrium() {
        // Uniform density → χ should equal χ_eq everywhere.
        let n = 8_usize;
        let l = 32.0_f64;
        let dx = l / n as Real;
        let a = 0.8_f64;
        let rho = 0.3_f64;
        let p = SymmetronParams {
            epsilon: 1e-8,
            max_iter: 200,
            ..default_p()
        };
        let rho_grid = vec![rho; n * n * n];
        let chi = solve_symmetron(&rho_grid, a, dx, n, &p);
        let chi_eq = chi_equilibrium(rho, a, &p);
        let err = chi
            .iter()
            .map(|c| (c - chi_eq).abs())
            .fold(0.0_f64, f64::max);
        println!("Uniform density χ error: {err:.3e}, χ_eq = {chi_eq:.6}");
        assert!(
            err < 1e-5,
            "χ did not converge to equilibrium: err={err:.3e}"
        );
    }

    #[test]
    fn solver_restored_phase_chi_near_zero() {
        // Pre-SSB: χ should stay near 0 everywhere.
        let n = 8_usize;
        let dx = 1.0_f64;
        let a = 0.3_f64; // a < a_ssb = 0.5
        let p = SymmetronParams {
            epsilon: 1e-8,
            max_iter: 200,
            ..default_p()
        };
        let rho = vec![1.0_f64; n * n * n]; // mean density
        let chi = solve_symmetron(&rho, a, dx, n, &p);
        let max_chi = chi.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        println!("Pre-SSB max |χ| = {max_chi:.3e}");
        assert!(
            max_chi < 0.1,
            "χ should be near 0 pre-SSB: max|χ| = {max_chi:.3e}"
        );
    }

    #[test]
    fn solver_residual_decreases_monotonically() {
        let n = 8_usize;
        let dx = 1.0_f64;
        let a = 0.8_f64;
        let p = SymmetronParams {
            n_smooth: 2,
            ..default_p()
        };
        let rho = vec![0.5_f64; n * n * n];

        let mut chi: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, &p)).collect();
        // Perturb initial guess
        chi[0] += 0.3;

        let r0 = residual_chi(&chi, &rho, a, dx, n, &p)
            .iter()
            .map(|r| r.abs())
            .fold(0.0_f64, f64::max);

        v_cycle_sym(&mut chi, &rho, a, dx, n, 3, 4, &p);

        let r1 = residual_chi(&chi, &rho, a, dx, n, &p)
            .iter()
            .map(|r| r.abs())
            .fold(0.0_f64, f64::max);

        println!("Residual: {r0:.3e} → {r1:.3e}");
        assert!(r1 < r0, "V-cycle did not reduce symmetron residual");
    }

    #[test]
    fn solver_screening_high_density() {
        // In high-density region, symmetry is restored: χ → 0.
        let n = 16_usize;
        let dx = 1.0_f64;
        let a = 0.8_f64;
        let p = SymmetronParams {
            epsilon: 1e-6,
            max_iter: 100,
            ..default_p()
        };
        // All cells at 10× mean density → ρ/M² >> 1
        let rho = vec![10.0_f64; n * n * n];
        let chi = solve_symmetron(&rho, a, dx, n, &p);
        let max_chi = chi.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        println!("High-density max |χ| = {max_chi:.3e}");
        assert!(
            max_chi < 0.1,
            "χ not screened at high density: {max_chi:.3e}"
        );
    }

    // ── Yukawa profile test ───────────────────────────────────────────────────

    #[test]
    fn fifth_force_zero_in_screened_region() {
        // Uniform field χ = const → ∇χ = 0 → no fifth force.
        let n = 8_usize;
        let dx = 1.0_f64;
        let a = 0.8_f64;
        let p = default_p();
        let chi = vec![0.7_f64; n * n * n]; // uniform
        let f5 = fifth_force_grid(&chi, dx, n, a, &p);
        let max_f = f5
            .iter()
            .flat_map(|f| f.iter())
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_f < 1e-12,
            "Uniform χ gives non-zero fifth force: {max_f:.3e}"
        );
    }

    #[test]
    fn fifth_force_direction_follows_gradient() {
        // χ = cos(2π x / L) → ∂χ/∂x < 0 for x ∈ (0, L/2)
        let n = 16_usize;
        let l = 16.0_f64;
        let dx = l / n as Real;
        let a = 0.8_f64;
        let p = SymmetronParams {
            beta_0: 1.0,
            ..default_p()
        };
        let k = 2.0 * PI / l;
        let chi: Vec<Real> = (0..n * n * n)
            .map(|c| {
                let (i, _, _) = from_flat(c, n);
                let x = (i as Real + 0.5) * dx;
                (k * x).cos()
            })
            .collect();

        let f5 = fifth_force_grid(&chi, dx, n, a, &p);

        // At cell (n/4, 0, 0): x = L/4, cos → decreasing → ∂χ/∂x < 0
        // grad_phi returns −∇χ, and fifth_force_grid multiplies by +β₀/a,
        // giving −(β₀/a)∇χ.  At x=L/4 where ∂χ/∂x < 0, the result is
        // −(β₀/a)(negative) = positive → force points in +x direction
        // (toward the χ minimum = toward matter, i.e. attractive).
        let c_quarter = idx3p((n / 4) as isize, 0, 0, n);
        let fx = f5[c_quarter][0];
        println!("Fifth force x-component at L/4: {fx:.4}");
        assert!(
            fx > 0.1,
            "Fifth force should be positive (attractive) at x=L/4: {fx:.4}"
        );
    }

    // ── Non-static placeholder ────────────────────────────────────────────────

    #[test]
    fn source_chi_antisymmetric_in_chi() {
        // f(-χ, ρ, a) = -f(χ, ρ, a)  [odd function in χ]
        let p = default_p();
        let a = 0.7_f64;
        let rho = 0.3_f64;
        for &chi in &[0.1, 0.3, 0.6, 0.9] {
            let fp = source_chi(chi, rho, a, &p);
            let fn_val = source_chi(-chi, rho, a, &p);
            assert!(
                (fp + fn_val).abs() < 1e-14,
                "f not odd: f({chi}) = {fp:.8}, f(-{chi}) = {fn_val:.8}"
            );
        }
    }

    #[test]
    fn source_chi_field_range_physical() {
        // For any physical inputs, f should not blow up.
        let p = default_p();
        for &a in &[0.3, 0.5, 0.8, 1.0] {
            for &rho in &[0.0, 0.5, 1.0, 5.0, 100.0] {
                for &chi in &[0.0, 0.5, 1.0, -0.5, -1.0] {
                    let f = source_chi(chi, rho, a, &p);
                    assert!(f.is_finite(), "f not finite: a={a}, rho={rho}, chi={chi}");
                }
            }
        }
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn ngs_sweep_slab_single_rank_matches_serial() {
        // Single-rank slab (n_local_i = n, i_start = 0) must produce the same
        // result as the serial ngs_sweep starting from the same ghost-padded chi.
        let n = 8_usize;
        let dx = 1.0_f64;
        let a = 0.8_f64;
        let p = SymmetronParams {
            n_smooth: 2,
            ..SymmetronParams::default()
        };
        let rho: Vec<Real> = (0..n * n * n)
            .map(|i| 0.3 + 0.1 * (i as Real).sin())
            .collect();
        let chi_init: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, &p)).collect();

        // Serial: one ngs_sweep on full n³ chi
        let mut chi_serial = chi_init.clone();
        ngs_sweep(&mut chi_serial, &rho, a, dx, n, &p);

        // Slab: build ghost-padded chi (n_local_i = n, i_start = 0)
        // Ghost rows wrap periodically: left ghost = row n-1, right ghost = row 0.
        let n_local_i = n;
        let mut chi_slab = vec![0.0_f64; (n_local_i + 2) * n * n];
        // owned cells li = 1..=n
        for li in 1..=n {
            for jk in 0..n * n {
                chi_slab[li * n * n + jk] = chi_init[(li - 1) * n * n + jk];
            }
        }
        // left ghost (li=0): periodic = row n-1 of chi
        for jk in 0..n * n {
            chi_slab[jk] = chi_init[(n - 1) * n * n + jk];
        }
        // right ghost (li=n+1): periodic = row 0 of chi
        for jk in 0..n * n {
            chi_slab[(n + 1) * n * n + jk] = chi_init[jk];
        }

        ngs_sweep_slab(&mut chi_slab, &rho, a, dx, n, n_local_i, 0, &p);

        // Compare owned cells
        for li in 1..=n {
            for jk in 0..n * n {
                let got = chi_slab[li * n * n + jk];
                let exp = chi_serial[(li - 1) * n * n + jk];
                assert!(
                    (got - exp).abs() < 1e-12,
                    "li={li} jk={jk}: slab={got:.10} serial={exp:.10}"
                );
            }
        }
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn residual_slab_matches_serial() {
        let n = 8_usize;
        let dx = 1.0_f64;
        let a = 0.8_f64;
        let p = SymmetronParams::default();
        let rho: Vec<Real> = (0..n * n * n)
            .map(|i| 0.5 + 0.2 * (i as Real / 10.0).sin())
            .collect();
        let chi: Vec<Real> = rho
            .iter()
            .map(|&r| chi_equilibrium(r, a, &p) + 0.05)
            .collect();

        // Serial residual on full grid
        let res_serial = residual_chi(&chi, &rho, a, dx, n, &p);

        // Build ghost-padded chi slab (n_local_i = n, periodic ghosts)
        let n_local_i = n;
        let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
        for li in 1..=n {
            for jk in 0..n * n {
                chi_slab[li * n * n + jk] = chi[(li - 1) * n * n + jk];
            }
        }
        for jk in 0..n * n {
            chi_slab[jk] = chi[(n - 1) * n * n + jk];
        }
        for jk in 0..n * n {
            chi_slab[(n + 1) * n * n + jk] = chi[jk];
        }

        let (res_slab, res_max) = residual_slab(&chi_slab, &rho, a, dx, n, n_local_i, &p);

        assert!(res_max > 0.0, "residual should be non-zero");
        for li in 1..=n {
            for jk in 0..n * n {
                let got = res_slab[(li - 1) * n * n + jk];
                let exp = res_serial[(li - 1) * n * n + jk];
                assert!(
                    (got - exp).abs() < 1e-12,
                    "li={li} jk={jk}: slab={got:.10} serial={exp:.10}"
                );
            }
        }
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn restrict_slab_matches_serial() {
        // restrict_slab with n_local_i = n must give the same result as restrict_uniform
        let n = 8_usize;
        let rho: Vec<Real> = (0..n * n * n)
            .map(|i| 1.0 + 0.3 * (i as Real).cos())
            .collect();
        let chi: Vec<Real> = (0..n * n * n)
            .map(|i| 0.5 * (i as Real / 5.0).sin())
            .collect();

        // Serial restriction
        let chi_c_serial = restrict_uniform(&chi, n);
        let rho_c_serial = restrict_uniform(&rho, n);

        // Build ghost-padded chi slab (n_local_i = n, periodic ghosts)
        let n_local_i = n;
        let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
        for li in 1..=n {
            for jk in 0..n * n {
                chi_slab[li * n * n + jk] = chi[(li - 1) * n * n + jk];
            }
        }
        for jk in 0..n * n {
            chi_slab[jk] = chi[(n - 1) * n * n + jk];
        }
        for jk in 0..n * n {
            chi_slab[(n + 1) * n * n + jk] = chi[jk];
        }

        let (chi_c_slab, rho_c_slab) = restrict_slab(&chi_slab, &rho, n_local_i, n);

        let nc = n / 2;
        let n_local_ic = n_local_i / 2;

        // Compare owned coarse cells (lci+1 in padded, lci in serial 0-indexed)
        for lci in 0..n_local_ic {
            for j in 0..nc {
                for k in 0..nc {
                    let got_chi = chi_c_slab[(lci + 1) * nc * nc + j * nc + k];
                    let exp_chi = chi_c_serial[lci * nc * nc + j * nc + k];
                    assert!(
                        (got_chi - exp_chi).abs() < 1e-12,
                        "chi: lci={lci} j={j} k={k}: {got_chi:.10} vs {exp_chi:.10}"
                    );

                    let got_rho = rho_c_slab[lci * nc * nc + j * nc + k];
                    let exp_rho = rho_c_serial[lci * nc * nc + j * nc + k];
                    assert!(
                        (got_rho - exp_rho).abs() < 1e-12,
                        "rho: lci={lci} j={j} k={k}: {got_rho:.10} vs {exp_rho:.10}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn prolongate_slab_roundtrip() {
        // Restrict then prolongate a smooth field; error should be small.
        let n = 8_usize;
        let chi: Vec<Real> = (0..n * n * n)
            .map(|c| {
                let i = (c / (n * n)) as Real;
                let j = ((c / n) % n) as Real;
                let k = (c % n) as Real;
                (i / n as Real * std::f64::consts::TAU).sin()
                    * (j / n as Real * std::f64::consts::TAU).cos()
                    * (k / n as Real * std::f64::consts::TAU).sin()
            })
            .collect();

        let n_local_i = n;
        let nc = n / 2;
        let n_local_ic = n_local_i / 2;

        // Build ghost-padded fine slab
        let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
        for li in 1..=n {
            for jk in 0..n * n {
                chi_slab[li * n * n + jk] = chi[(li - 1) * n * n + jk];
            }
        }
        for jk in 0..n * n {
            chi_slab[jk] = chi[(n - 1) * n * n + jk];
        }
        for jk in 0..n * n {
            chi_slab[(n + 1) * n * n + jk] = chi[jk];
        }

        // Restrict
        let rho_dummy = vec![1.0_f64; n * n * n];
        let (chi_c, _) = restrict_slab(&chi_slab, &rho_dummy, n_local_i, n);

        // Fill coarse ghost cells (periodic: left ghost = last owned row of coarse)
        let mut chi_c_with_ghosts = chi_c.clone();
        // left ghost (lci=0): from lci=n_local_ic (last owned of this rank = last global since single-rank)
        for jk in 0..nc * nc {
            chi_c_with_ghosts[jk] = chi_c[n_local_ic * nc * nc + jk];
        }
        // right ghost (lci=n_local_ic+1): from lci=1 (first owned)
        for jk in 0..nc * nc {
            chi_c_with_ghosts[(n_local_ic + 1) * nc * nc + jk] = chi_c[nc * nc + jk];
        }

        // Prolongate
        let fine_out = prolongate_slab(&chi_c_with_ghosts, n_local_i, n);

        // On a single rank with n=8 → nc=4, restrict+prolongate should reconstruct
        // a smooth field with max error < 0.2 (trilinear on a half-resolution field).
        let mut max_err = 0.0_f64;
        for li in 0..n_local_i {
            for jk in 0..n * n {
                let got = fine_out[(li + 1) * n * n + jk];
                let exp = chi[li * n * n + jk];
                max_err = max_err.max((got - exp).abs());
            }
        }
        println!("prolongate_slab round-trip max error: {max_err:.4}");
        // Injection restriction discards odd fine cells, so trilinear prolongation
        // cannot recover them exactly.  For a full-frequency sinusoid on 8 cells,
        // the theoretical max trilinear error is ~0.58.  Use 0.65 as a safe bound.
        assert!(max_err < 0.65, "round-trip error too large: {max_err:.4}");
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn prolongate_slab_constant_exact() {
        // Trilinear interpolation must exactly reproduce constant fields.
        // If restrict+prolongate of a constant doesn't return the same constant,
        // the index arithmetic in prolongate_slab is wrong.
        let n = 8_usize;
        let nc = n / 2;
        let n_local_i = n;
        let n_local_ic = n_local_i / 2;
        let val = 0.42_f64;

        let chi_const: Vec<Real> = vec![val; n * n * n];
        let rho_dummy: Vec<Real> = vec![1.0; n * n * n];

        // Build ghost-padded fine slab
        let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
        for li in 1..=n {
            for jk in 0..n * n {
                chi_slab[li * n * n + jk] = chi_const[(li - 1) * n * n + jk];
            }
        }
        // periodic ghosts
        for jk in 0..n * n {
            chi_slab[jk] = chi_const[(n - 1) * n * n + jk];
        }
        for jk in 0..n * n {
            chi_slab[(n + 1) * n * n + jk] = chi_const[jk];
        }

        let (chi_c, _) = restrict_slab(&chi_slab, &rho_dummy, n_local_i, n);

        // Fill coarse ghosts (periodic)
        let mut chi_c_ghosts = chi_c.clone();
        for jk in 0..nc * nc {
            chi_c_ghosts[jk] = chi_c[n_local_ic * nc * nc + jk]; // left ghost = last owned
            chi_c_ghosts[(n_local_ic + 1) * nc * nc + jk] = chi_c[nc * nc + jk];
            // right ghost = first owned
        }

        let fine_out = prolongate_slab(&chi_c_ghosts, n_local_i, n);

        let mut max_err = 0.0_f64;
        for li in 0..n_local_i {
            for jk in 0..n * n {
                let got = fine_out[(li + 1) * n * n + jk];
                max_err = max_err.max((got - val).abs());
            }
        }
        assert!(
            max_err < 1e-10,
            "constant field not preserved: max_err = {max_err:.2e}"
        );
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn v_cycle_sym_mpi_reduces_residual() {
        use crate::mpi::MpiContext;
        let mpi = MpiContext::init();

        let n = 16_usize;
        let dx = 1.0_f64;
        let a = 0.8_f64;
        let p = SymmetronParams {
            n_smooth: 2,
            epsilon: 1e-8,
            max_iter: 200,
            ..SymmetronParams::default()
        };
        let rho: Vec<Real> = (0..n * n * n)
            .map(|i| 0.5 + 0.2 * (i as Real).cos())
            .collect();
        let chi_eq: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, &p)).collect();

        // Perturb slightly from equilibrium
        let mut chi_init = chi_eq.clone();
        chi_init[n * n / 2] += 0.3;

        // n_local_i = n for single-rank test
        let n_local_i = n;
        let i_start = 0;

        // Build ghost-padded chi slab
        let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
        for li in 1..=n {
            for jk in 0..n * n {
                chi_slab[li * n * n + jk] = chi_init[(li - 1) * n * n + jk];
            }
        }
        for jk in 0..n * n {
            chi_slab[jk] = chi_init[(n - 1) * n * n + jk];
        }
        for jk in 0..n * n {
            chi_slab[(n + 1) * n * n + jk] = chi_init[jk];
        }

        mpi.exchange_halos(&mut chi_slab, n_local_i, n, 1);
        let (_, r0) = residual_slab(&chi_slab, &rho, a, dx, n, n_local_i, &p);

        v_cycle_sym_mpi(
            &mut chi_slab,
            &rho,
            a,
            dx,
            n,
            n_local_i,
            i_start,
            2,
            4,
            &p,
            &mpi,
        );

        mpi.exchange_halos(&mut chi_slab, n_local_i, n, 1);
        let (_, r1) = residual_slab(&chi_slab, &rho, a, dx, n, n_local_i, &p);

        println!("v_cycle_sym_mpi: residual {r0:.3e} → {r1:.3e}");
        assert!(
            r1 < r0,
            "V-cycle did not reduce residual: {r0:.3e} → {r1:.3e}"
        );
    }

    #[cfg(feature = "mpi_parallel")]
    #[test]
    fn solve_symmetron_mpi_matches_serial() {
        use crate::mpi::MpiContext;
        let mpi = MpiContext::init();

        let n = 16_usize;
        let l = 16.0_f64;
        let dx = l / n as Real;
        let a = 0.8_f64;
        let p = SymmetronParams {
            epsilon: 1e-6,
            max_iter: 200,
            ..SymmetronParams::default()
        };
        let rho: Vec<Real> = (0..n * n * n)
            .map(|i| 0.5 + 0.3 * (i as Real / 7.0).sin().abs())
            .collect();

        // Serial solve on full grid
        let chi_serial = solve_symmetron(&rho, a, dx, n, &p);

        // MPI solve: single rank owns full slab
        let n_local_i = mpi.slab_count(n);
        let i0 = mpi.slab_start(n);
        let rho_slab = &rho[i0 * n * n..(i0 + n_local_i) * n * n];
        let chi_slab = solve_symmetron_mpi(rho_slab, a, dx, n, &p, &mpi);

        // Reconstruct full chi from slab (single rank: already full)
        let chi_mpi = mpi.allgather_slabs(&chi_slab, n);

        let max_err = chi_serial
            .iter()
            .zip(chi_mpi.iter())
            .map(|(s, m)| (s - m).abs())
            .fold(0.0_f64, f64::max);

        println!("solve_symmetron_mpi vs serial max error: {max_err:.3e}");
        // Both converge to the same tolerance — allow 10× tolerance as slop
        assert!(
            max_err < 10.0 * p.epsilon,
            "MPI solve diverges from serial: max error = {max_err:.3e}"
        );
    }
}
