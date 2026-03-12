//! Multigrid solver for Poisson's equation on a uniform periodic grid.
//!
//! Implements a **V-cycle** multigrid with red-black Gauss-Seidel smoothing,
//! matching the RAMSES approach (Guillet & Teyssier 2011).
//!
//! Solves:  ∇²Φ = S  in a periodic cubic box.
//!
//! The same restriction and prolongation operators are shared with the
//! nonlinear scalar-field multigrid in `scalar_field/`.

use rayon::prelude::*;
use crate::utils::types::Real;

// ── Index helper (public: used by scalar_field module) ────────────────────────

/// Flat 3-D index with periodic wrapping.
#[inline]
pub fn flat3(i: isize, j: isize, k: isize, n: usize) -> usize {
    let n = n as isize;
    (k.rem_euclid(n) * n * n + j.rem_euclid(n) * n + i.rem_euclid(n)) as usize
}

// ── Gauss-Seidel red-black smoother ──────────────────────────────────────────

/// One red-black Gauss-Seidel sweep for  ∇²Φ = S.
/// Updates `phi` in-place; reads neighbours from a snapshot for thread safety.
pub fn gs_sweep(phi: &mut [Real], rhs: &[Real], dx: Real, n: usize) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap = phi.to_vec(); // snapshot for safe neighbour reads
    for color in 0..2_usize {
        for k in 0..n as isize {
            for j in 0..n as isize {
                for i in 0..n as isize {
                    if (i + j + k) as usize % 2 != color { continue; }
                    let c = flat3(i, j, k, n);
                    let nb = snap[flat3(i+1, j,   k,   n)]
                           + snap[flat3(i-1, j,   k,   n)]
                           + snap[flat3(i,   j+1, k,   n)]
                           + snap[flat3(i,   j-1, k,   n)]
                           + snap[flat3(i,   j,   k+1, n)]
                           + snap[flat3(i,   j,   k-1, n)];
                    // Φ_new = (∑nb - dx² S) / 6
                    phi[c] = (nb - rhs[c] / inv_dx2) / 6.0;
                }
            }
        }
    }
}

// ── Residual ─────────────────────────────────────────────────────────────────

/// Pointwise residual  r_i = S_i − (∇²Φ)_i.
pub fn residual(phi: &[Real], rhs: &[Real], dx: Real, n: usize) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    for k in 0..n as isize {
        for j in 0..n as isize {
            for i in 0..n as isize {
                let c = flat3(i, j, k, n);
                let lap = (phi[flat3(i+1, j,   k,   n)]
                         + phi[flat3(i-1, j,   k,   n)]
                         + phi[flat3(i,   j+1, k,   n)]
                         + phi[flat3(i,   j-1, k,   n)]
                         + phi[flat3(i,   j,   k+1, n)]
                         + phi[flat3(i,   j,   k-1, n)]
                         - 6.0 * phi[c]) * inv_dx2;
                res[c] = rhs[c] - lap;
            }
        }
    }
    res
}

// ── Restriction ───────────────────────────────────────────────────────────────

/// Restrict a fine grid (n_fine³) to coarse ((n_fine/2)³) by injection.
/// Works for both residual fields and solution fields.
pub fn restrict_uniform(fine: &[Real], n_fine: usize) -> Vec<Real> {
    let n = n_fine / 2;
    let mut coarse = vec![0.0_f64; n * n * n];
    for k in 0..n as isize {
        for j in 0..n as isize {
            for i in 0..n as isize {
                coarse[flat3(i, j, k, n)] = fine[flat3(2*i, 2*j, 2*k, n_fine)];
            }
        }
    }
    coarse
}

// ── Prolongation (trilinear interpolation) ────────────────────────────────────

/// Prolongate a coarse grid ((n_coarse)³) to fine ((n_coarse*2)³) by
/// trilinear interpolation.
pub fn prolongate_uniform(coarse: &[Real], n_coarse: usize) -> Vec<Real> {
    let nf = n_coarse * 2;
    let nc = n_coarse;
    let mut fine = vec![0.0_f64; nf * nf * nf];
    // Fine cell (fi, fj, fk) centre in coarse-cell coordinates
    for fk in 0..nf as isize {
        for fj in 0..nf as isize {
            for fi in 0..nf as isize {
                // Fine cell centre ≡ (fi+0.5)/2 − 0.5 in coarse coords
                let cx = (fi as Real + 0.5) * 0.5 - 0.5;
                let cy = (fj as Real + 0.5) * 0.5 - 0.5;
                let cz = (fk as Real + 0.5) * 0.5 - 0.5;
                let ci = cx.floor() as isize;
                let cj = cy.floor() as isize;
                let ck = cz.floor() as isize;
                let tx = cx - ci as Real;
                let ty = cy - cj as Real;
                let tz = cz - ck as Real;
                let mut val = 0.0_f64;
                for (di, wx) in [(0_isize, 1.0 - tx), (1, tx)] {
                    for (dj, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                        for (dk, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                            val += wx * wy * wz * coarse[flat3(ci+di, cj+dj, ck+dk, nc)];
                        }
                    }
                }
                fine[flat3(fi, fj, fk, nf)] = val;
            }
        }
    }
    fine
}

// ── V-cycle ───────────────────────────────────────────────────────────────────

/// One multigrid V-cycle for  ∇²Φ = rhs.  Updates `phi` in-place.
pub fn v_cycle(
    phi:      &mut Vec<Real>,
    rhs:      &[Real],
    dx:       Real,
    n:        usize,
    n_smooth: usize,
    min_n:    usize,
) {
    for _ in 0..n_smooth { gs_sweep(phi, rhs, dx, n); }

    if n <= min_n {
        for _ in 0..4 * n_smooth { gs_sweep(phi, rhs, dx, n); }
        return;
    }

    let res   = residual(phi, rhs, dx, n);
    let res_c = restrict_uniform(&res, n);
    let n_c   = n / 2;
    let dx_c  = dx * 2.0;
    let mut err_c = vec![0.0_f64; n_c * n_c * n_c];

    v_cycle(&mut err_c, &res_c, dx_c, n_c, n_smooth, min_n);

    let err_f = prolongate_uniform(&err_c, n_c);
    phi.iter_mut().zip(err_f.iter()).for_each(|(p, e)| *p += e);

    for _ in 0..n_smooth { gs_sweep(phi, rhs, dx, n); }
}

// ── Public solver ─────────────────────────────────────────────────────────────

/// Solve  ∇²Φ = source  on an n³ periodic uniform grid.
///
/// # Arguments
/// * `source`   – RHS array length n³
/// * `dx`       – cell spacing [Mpc/h]
/// * `n`        – cells per dimension
/// * `epsilon`  – max-norm convergence criterion on residual
/// * `max_iter` – V-cycle limit
/// * `n_smooth` – Gauss-Seidel sweeps per level
pub fn solve_poisson(
    source:   &[Real],
    dx:       Real,
    n:        usize,
    epsilon:  Real,
    max_iter: u32,
    n_smooth: usize,
) -> Vec<Real> {
    assert_eq!(source.len(), n * n * n, "source length mismatch");
    let min_n = 4;
    let mut phi = vec![0.0_f64; n * n * n];

    for iter in 0..max_iter {
        let res     = residual(&phi, source, dx, n);
        let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        log::debug!("Poisson V-cycle {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < epsilon { break; }
        v_cycle(&mut phi, source, dx, n, n_smooth, min_n);
    }
    phi
}

// ── Gradient of potential ─────────────────────────────────────────────────────

/// Compute  g = −∇Φ  using centred finite differences.
/// Returns array of [gx, gy, gz] per cell.
pub fn grad_phi(phi: &[Real], dx: Real, n: usize) -> Vec<[Real; 3]> {
    let inv_2dx = 0.5 / dx;
    let mut g = vec![[0.0_f64; 3]; n * n * n];
    g.par_iter_mut().enumerate().for_each(|(c, gc)| {
        let k = (c / (n * n)) as isize;
        let j = ((c / n) % n)  as isize;
        let i = (c % n)         as isize;
        gc[0] = -(phi[flat3(i+1, j,   k,   n)] - phi[flat3(i-1, j,   k,   n)]) * inv_2dx;
        gc[1] = -(phi[flat3(i,   j+1, k,   n)] - phi[flat3(i,   j-1, k,   n)]) * inv_2dx;
        gc[2] = -(phi[flat3(i,   j,   k+1, n)] - phi[flat3(i,   j,   k-1, n)]) * inv_2dx;
    });
    g
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn max_err(a: &[Real], b: &[Real]) -> Real {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Solve ∇²Φ = S where S = −k² sin(k x) and check Φ ≈ sin(k x).
    #[test]
    fn poisson_single_mode_convergence() {
        let n  = 32_usize;
        let l  = 1.0_f64;
        let dx = l / n as Real;
        let k  = 2.0 * PI / l;

        let mut source = vec![0.0_f64; n * n * n];
        let mut exact  = vec![0.0_f64; n * n * n];
        for kk in 0..n as isize {
            for j in 0..n as isize {
                for i in 0..n as isize {
                    let c = flat3(i, j, kk, n);
                    let x = (i as Real + 0.5) * dx;
                    let phi_x = (k * x).sin();
                    source[c] = -k * k * phi_x;
                    exact[c]  = phi_x;
                }
            }
        }
        let phi = solve_poisson(&source, dx, n, 1e-9, 300, 2);
        // Remove DC offset
        let dc_phi   = phi.iter().sum::<f64>() / phi.len() as f64;
        let dc_exact = exact.iter().sum::<f64>() / exact.len() as f64;
        let phi_dc:  Vec<_> = phi.iter().map(|p| p - dc_phi).collect();
        let ex_dc:   Vec<_> = exact.iter().map(|e| e - dc_exact).collect();
        let err = max_err(&phi_dc, &ex_dc);
        println!("Poisson single-mode max error: {err:.3e}");
        assert!(err < 1e-3, "Poisson error too large: {err}");
    }

    /// Verify second-order convergence: halving dx should quarter the error.
    #[test]
    fn poisson_convergence_order() {
        let l = 1.0_f64;
        let k = 2.0 * PI / l;
        let errs: Vec<Real> = [8_usize, 16, 32].iter().map(|&n| {
            let dx = l / n as Real;
            let mut source = vec![0.0_f64; n*n*n];
            let mut exact  = vec![0.0_f64; n*n*n];
            for kk in 0..n as isize {
                for j in 0..n as isize {
                    for i in 0..n as isize {
                        let c = flat3(i, j, kk, n);
                        let x = (i as Real + 0.5) * dx;
                        let phi_x = (k * x).sin();
                        source[c] = -k * k * phi_x;
                        exact[c]  = phi_x;
                    }
                }
            }
            let phi  = solve_poisson(&source, dx, n, 1e-10, 500, 3);
            let dc_p = phi.iter().sum::<f64>() / phi.len() as f64;
            let dc_e = exact.iter().sum::<f64>() / exact.len() as f64;
            phi.iter().zip(exact.iter())
               .map(|(p, e)| ((p - dc_p) - (e - dc_e)).abs())
               .fold(0.0_f64, f64::max)
        }).collect();
        println!("Convergence errors: {errs:.3e?}");
        // Ratio should be ~4 (second-order)
        let r1 = errs[0] / errs[1];
        let r2 = errs[1] / errs[2];
        println!("Convergence ratios: {r1:.2}, {r2:.2}");
        assert!(r1 > 2.0, "Not converging (r={r1:.2})");
        assert!(r2 > 2.0, "Not converging (r={r2:.2})");
    }

    /// Gradient of a known potential: Φ = sin(kx), ∂Φ/∂x = k cos(kx).
    #[test]
    fn gradient_accuracy() {
        let n  = 32_usize;
        let l  = 1.0_f64;
        let dx = l / n as Real;
        let k  = 2.0 * PI / l;
        let phi: Vec<Real> = (0..n*n*n).map(|c| {
            let i = c % n;
            let x = (i as Real + 0.5) * dx;
            (k * x).sin()
        }).collect();
        let g = grad_phi(&phi, dx, n);
        // gx = -∂Φ/∂x = -k cos(kx)
        let max_err_gx = (0..n*n*n).map(|c| {
            let i = c % n;
            let x = (i as Real + 0.5) * dx;
            let exact_gx = -k * (k * x).cos();
            (g[c][0] - exact_gx).abs()
        }).fold(0.0_f64, f64::max);
        println!("Gradient max error: {max_err_gx:.3e}");
        assert!(max_err_gx < 0.05, "Gradient error too large: {max_err_gx}");
    }
}
