//! Poisson solver for  ∇²Φ = S  on a periodic uniform grid.
//!
//! # Backends
//!
//! 1. **FFT-based** (`solve_poisson_fft`, feature `fft_pm`) — exact for
//!    periodic uniform grids; O(N log N).  Primary solver for the PM step.
//!
//! 2. **Multigrid V-cycle** (`solve_poisson_mg`) — always available.
//!    Red-black GS smoothing + injection/trilinear prolongation
//!    (Guillet & Teyssier 2011). Used for AMR patches and scalar-field
//!    sub-problems where FFT doesn't apply.
//!
//! # Grid convention
//!
//! C-order (row-major): flat(i,j,k) = i·n² + j·n + k.
//! All inner loops iterate i→j→k (slowest to fastest) to maximise
//! cache-line reuse on the k-axis.

use crate::utils::types::{from_flat, idx3p, Real};
use rayon::prelude::*;

// ─── Laplacian stencil (6-point) ──────────────────────────────────────────────

/// Apply the 7-point (6-neighbour) discrete Laplacian to `phi` and subtract
/// from `rhs`, producing the residual  r = rhs − ∇²phi.
///
/// Parallelised over rows (i) so the k-axis inner loop is contiguous in cache.
pub fn residual(phi: &[Real], rhs: &[Real], dx: Real, n: usize) -> Vec<Real> {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res = vec![0.0_f64; n * n * n];
    // Parallel over the i-axis; j and k are sequential (cache-friendly)
    res.par_chunks_mut(n * n).enumerate().for_each(|(i, row)| {
        let i = i as isize;
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c = idx3p(i, j, k, n);
                let lap = (phi[idx3p(i + 1, j, k, n)]
                    + phi[idx3p(i - 1, j, k, n)]
                    + phi[idx3p(i, j + 1, k, n)]
                    + phi[idx3p(i, j - 1, k, n)]
                    + phi[idx3p(i, j, k + 1, n)]
                    + phi[idx3p(i, j, k - 1, n)]
                    - 6.0 * phi[c])
                    * inv_dx2;
                row[j as usize * n + k as usize] = rhs[c] - lap;
            }
        }
    });
    res
}

// ─── Red-black GS sweep ───────────────────────────────────────────────────────

/// One red-black Gauss-Seidel sweep for  ∇²Φ = S.
///
/// Reads neighbours from a snapshot taken at entry (thread-safe read).
/// Updates cells in two passes (even/odd i+j+k parity).
pub fn gs_sweep(phi: &mut [Real], rhs: &[Real], dx: Real, n: usize) {
    let inv_dx2 = 1.0 / (dx * dx);
    // Snapshot taken once so all neighbour reads use pre-sweep values.
    // This gives a parallel-safe Jacobi-style update across both color passes,
    // which has well-characterised high-frequency damping for the V-cycle.
    let snap = phi.to_vec();
    for color in 0..2_usize {
        // Iterate i→j→k for cache locality
        for i in 0..n as isize {
            for j in 0..n as isize {
                // Determine starting k parity for this (i,j) row
                let k_start = ((i + j) as usize % 2 + color) % 2;
                let mut k = k_start as isize;
                while k < n as isize {
                    let c = idx3p(i, j, k, n);
                    let nb = snap[idx3p(i + 1, j, k, n)]
                        + snap[idx3p(i - 1, j, k, n)]
                        + snap[idx3p(i, j + 1, k, n)]
                        + snap[idx3p(i, j - 1, k, n)]
                        + snap[idx3p(i, j, k + 1, n)]
                        + snap[idx3p(i, j, k - 1, n)];
                    phi[c] = (nb - rhs[c] / inv_dx2) / 6.0;
                    k += 2;
                }
            }
        }
    }
}

// ─── Restriction (injection) ──────────────────────────────────────────────────

/// Restrict fine (n³) → coarse ((n/2)³) by injection.
pub fn restrict_uniform(fine: &[Real], n_fine: usize) -> Vec<Real> {
    let nc = n_fine / 2;
    let mut coarse = vec![0.0_f64; nc * nc * nc];
    coarse
        .par_chunks_mut(nc * nc)
        .enumerate()
        .for_each(|(i, row)| {
            let i = i as isize;
            for j in 0..nc as isize {
                for k in 0..nc as isize {
                    row[j as usize * nc + k as usize] = fine[idx3p(2 * i, 2 * j, 2 * k, n_fine)];
                }
            }
        });
    coarse
}

// ─── Prolongation (trilinear) ─────────────────────────────────────────────────

/// Prolongate coarse (nc³) → fine (2nc³) by trilinear interpolation.
pub fn prolongate_uniform(coarse: &[Real], nc: usize) -> Vec<Real> {
    let nf = nc * 2;
    let mut fine = vec![0.0_f64; nf * nf * nf];
    fine.par_chunks_mut(nf * nf)
        .enumerate()
        .for_each(|(fi, row)| {
            let fi = fi as isize;
            for fj in 0..nf as isize {
                for fk in 0..nf as isize {
                    // Fine-cell centre in coarse coordinates
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
                                val += wx * wy * wz * coarse[idx3p(ci + di, cj + dj, ck + dk, nc)];
                            }
                        }
                    }
                    row[fj as usize * nf + fk as usize] = val;
                }
            }
        });
    fine
}

// ─── V-cycle ──────────────────────────────────────────────────────────────────

/// One multigrid V-cycle for  ∇²Φ = rhs.  Modifies `phi` in-place.
pub fn v_cycle(phi: &mut [Real], rhs: &[Real], dx: Real, n: usize, n_smooth: usize, min_n: usize) {
    for _ in 0..n_smooth {
        gs_sweep(phi, rhs, dx, n);
    }
    if n <= min_n {
        for _ in 0..4 * n_smooth {
            gs_sweep(phi, rhs, dx, n);
        }
        return;
    }
    let res_c = restrict_uniform(&residual(phi, rhs, dx, n), n);
    let nc = n / 2;
    let mut err_c = vec![0.0_f64; nc * nc * nc];
    v_cycle(&mut err_c, &res_c, dx * 2.0, nc, n_smooth, min_n);
    let err_f = prolongate_uniform(&err_c, nc);
    phi.iter_mut().zip(err_f.iter()).for_each(|(p, e)| *p += e);
    for _ in 0..n_smooth {
        gs_sweep(phi, rhs, dx, n);
    }
}

// ─── Multigrid solver ─────────────────────────────────────────────────────────

/// Solve  ∇²Φ = source  by iterated V-cycles.
///
/// Enforces the periodic compatibility condition (zero-mean source and solution).
pub fn solve_poisson_mg(
    source: &[Real],
    dx: Real,
    n: usize,
    epsilon: Real,
    max_iter: u32,
    n_smooth: usize,
) -> Vec<Real> {
    assert_eq!(source.len(), n * n * n);
    let src_mean = source.iter().sum::<f64>() / (n * n * n) as f64;
    let source_zm: Vec<Real> = source.iter().map(|s| s - src_mean).collect();
    let mut phi = vec![0.0_f64; n * n * n];
    for iter in 0..max_iter {
        let res = residual(&phi, &source_zm, dx, n);
        let res_max = res.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
        log::trace!("MG iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < epsilon {
            break;
        }
        v_cycle(&mut phi, &source_zm, dx, n, n_smooth, 4);
    }
    let mean = phi.iter().sum::<f64>() / phi.len() as f64;
    phi.iter_mut().for_each(|p| *p -= mean);
    phi
}

// ─── FFT solver ───────────────────────────────────────────────────────────────

/// Solve  ∇²Φ = S  using the DFT — exact (machine precision) for uniform
/// periodic grids; O(N log N).  Requires feature `fft_pm`.
///
/// The algorithm divides  Ŝ(k) / λ(k)  where
///   λ(ki,kj,kk) = −(4/dx²) [sin²(πki/n) + sin²(πkj/n) + sin²(πkk/n)]
/// is the eigenvalue of the 6-point discrete Laplacian.
#[cfg(feature = "fft_pm")]
pub fn solve_poisson_fft(source: &[Real], dx: Real, n: usize) -> Vec<Real> {
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::f64::consts::PI;

    let n3 = n * n * n;
    assert_eq!(source.len(), n3);

    let mut data: Vec<Complex<f64>> = source.iter().map(|&s| Complex::new(s, 0.0)).collect();
    let mut planner = FftPlanner::<f64>::new();

    // 3-D FFT as successive 1-D FFTs along k, j, i axes
    {
        let fft = planner.plan_fft_forward(n);
        for i in 0..n {
            for j in 0..n {
                fft.process(&mut data[i * n * n + j * n..i * n * n + j * n + n]);
            }
        }
    }
    {
        let fft = planner.plan_fft_forward(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    col[j] = data[i * n * n + j * n + k];
                }
                fft.process(&mut col);
                for j in 0..n {
                    data[i * n * n + j * n + k] = col[j];
                }
            }
        }
    }
    {
        let fft = planner.plan_fft_forward(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    col[i] = data[i * n * n + j * n + k];
                }
                fft.process(&mut col);
                for i in 0..n {
                    data[i * n * n + j * n + k] = col[i];
                }
            }
        }
    }

    // Divide by discrete Laplacian eigenvalue
    let inv_dx2 = 1.0 / (dx * dx);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let c = i * n * n + j * n + k;
                if i == 0 && j == 0 && k == 0 {
                    data[c] = Complex::new(0.0, 0.0);
                    continue;
                }
                let si = (PI * i as f64 / n as f64).sin();
                let sj = (PI * j as f64 / n as f64).sin();
                let sk = (PI * k as f64 / n as f64).sin();
                let lambda = -4.0 * inv_dx2 * (si * si + sj * sj + sk * sk);
                data[c] /= lambda;
            }
        }
    }

    // Inverse FFT along each axis
    {
        let ifft = planner.plan_fft_inverse(n);
        for i in 0..n {
            for j in 0..n {
                ifft.process(&mut data[i * n * n + j * n..i * n * n + j * n + n]);
            }
        }
    }
    {
        let ifft = planner.plan_fft_inverse(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    col[j] = data[i * n * n + j * n + k];
                }
                ifft.process(&mut col);
                for j in 0..n {
                    data[i * n * n + j * n + k] = col[j];
                }
            }
        }
    }
    {
        let ifft = planner.plan_fft_inverse(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    col[i] = data[i * n * n + j * n + k];
                }
                ifft.process(&mut col);
                for i in 0..n {
                    data[i * n * n + j * n + k] = col[i];
                }
            }
        }
    }

    let norm = n3 as f64;
    let mut phi: Vec<Real> = data.iter().map(|c| c.re / norm).collect();
    let mean = phi.iter().sum::<f64>() / norm;
    phi.iter_mut().for_each(|p| *p -= mean);
    phi
}

/// FFT Poisson solver that writes into pre-allocated buffers.
///
/// `phi` receives the result (length n³).
/// `fft_buf` is the complex work buffer (length n³).
/// `fft_col` is the column buffer for non-contiguous axis transforms (length n).
///
/// Use this variant in the step loop to avoid per-step heap allocations.
#[cfg(feature = "fft_pm")]
pub fn solve_poisson_fft_into(
    source: &[Real],
    dx: Real,
    n: usize,
    phi: &mut [Real],
    fft_buf: &mut [rustfft::num_complex::Complex<f64>],
    fft_col: &mut [rustfft::num_complex::Complex<f64>],
) {
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::f64::consts::PI;

    let n3 = n * n * n;
    assert_eq!(source.len(), n3);
    assert_eq!(phi.len(), n3);
    assert!(fft_buf.len() >= n3);
    assert!(fft_col.len() >= n);

    // Promote source to complex in the pre-allocated buffer
    for (c, &s) in source.iter().enumerate() {
        fft_buf[c] = Complex::new(s, 0.0);
    }

    let mut planner = FftPlanner::<f64>::new();

    // 3-D FFT as successive 1-D FFTs along k, j, i axes
    {
        let fft = planner.plan_fft_forward(n);
        for i in 0..n {
            for j in 0..n {
                fft.process(&mut fft_buf[i * n * n + j * n..i * n * n + j * n + n]);
            }
        }
    }
    {
        let fft = planner.plan_fft_forward(n);
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    fft_col[j] = fft_buf[i * n * n + j * n + k];
                }
                fft.process(&mut fft_col[..n]);
                for j in 0..n {
                    fft_buf[i * n * n + j * n + k] = fft_col[j];
                }
            }
        }
    }
    {
        let fft = planner.plan_fft_forward(n);
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    fft_col[i] = fft_buf[i * n * n + j * n + k];
                }
                fft.process(&mut fft_col[..n]);
                for i in 0..n {
                    fft_buf[i * n * n + j * n + k] = fft_col[i];
                }
            }
        }
    }

    // Divide by discrete Laplacian eigenvalue
    let inv_dx2 = 1.0 / (dx * dx);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let c = i * n * n + j * n + k;
                if i == 0 && j == 0 && k == 0 {
                    fft_buf[c] = Complex::new(0.0, 0.0);
                    continue;
                }
                let si = (PI * i as f64 / n as f64).sin();
                let sj = (PI * j as f64 / n as f64).sin();
                let sk = (PI * k as f64 / n as f64).sin();
                let lambda = -4.0 * inv_dx2 * (si * si + sj * sj + sk * sk);
                fft_buf[c] /= lambda;
            }
        }
    }

    // Inverse FFT along each axis
    {
        let ifft = planner.plan_fft_inverse(n);
        for i in 0..n {
            for j in 0..n {
                ifft.process(&mut fft_buf[i * n * n + j * n..i * n * n + j * n + n]);
            }
        }
    }
    {
        let ifft = planner.plan_fft_inverse(n);
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    fft_col[j] = fft_buf[i * n * n + j * n + k];
                }
                ifft.process(&mut fft_col[..n]);
                for j in 0..n {
                    fft_buf[i * n * n + j * n + k] = fft_col[j];
                }
            }
        }
    }
    {
        let ifft = planner.plan_fft_inverse(n);
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    fft_col[i] = fft_buf[i * n * n + j * n + k];
                }
                ifft.process(&mut fft_col[..n]);
                for i in 0..n {
                    fft_buf[i * n * n + j * n + k] = fft_col[i];
                }
            }
        }
    }

    let norm = n3 as f64;
    for c in 0..n3 {
        phi[c] = fft_buf[c].re / norm;
    }
    let mean = phi.iter().sum::<f64>() / norm;
    phi.iter_mut().for_each(|p| *p -= mean);
}

/// Stub when `fft_pm` is disabled: fall back to multigrid.
#[cfg(not(feature = "fft_pm"))]
pub fn solve_poisson_fft(source: &[Real], dx: Real, n: usize) -> Vec<Real> {
    solve_poisson_mg(source, dx, n, 1e-10, 500, 3)
}

// ─── Distributed FFT solver ──────────────────────────────────────────────────

/// Distributed FFT Poisson solver via slab-pencil transpose.
///
/// Uses the same discrete-Laplacian eigenvalue division as `solve_poisson_fft`,
/// but distributes the i-axis FFT pass across MPI ranks via `transpose_i_to_j`
/// / `transpose_j_to_i`.  On a single rank the result is bitwise identical to
/// the serial solver.
///
/// `local_source`: source term for this rank's i-slab, length = `n_local_i * n * n`.
/// `dx`: cell spacing.  `n`: global grid size.
/// `mpi`: MPI context (carries rank, size, transpose operations).
///
/// Returns the gravitational potential for this rank's i-slab.
#[cfg(all(feature = "fft_pm", feature = "mpi_parallel"))]
pub fn solve_poisson_fft_mpi(
    local_source: &[Real],
    dx: Real,
    n: usize,
    mpi: &crate::mpi::MpiContext,
) -> Vec<Real> {
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::f64::consts::PI;

    let n_local_i = mpi.slab_count(n);
    let n_local = n_local_i * n * n;
    assert_eq!(local_source.len(), n_local);

    // Promote to complex
    let mut data: Vec<Complex<f64>> = local_source.iter().map(|&s| Complex::new(s, 0.0)).collect();
    let mut planner = FftPlanner::<f64>::new();

    // ── Forward pass 1: 1D FFTs along k-axis (contiguous, no communication) ──
    {
        let fft = planner.plan_fft_forward(n);
        for i in 0..n_local_i {
            for j in 0..n {
                let start = i * n * n + j * n;
                fft.process(&mut data[start..start + n]);
            }
        }
    }

    // ── Forward pass 2: 1D FFTs along j-axis (gather within slab) ────────────
    {
        let fft = planner.plan_fft_forward(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n_local_i {
            for k in 0..n {
                for j in 0..n {
                    col[j] = data[i * n * n + j * n + k];
                }
                fft.process(&mut col);
                for j in 0..n {
                    data[i * n * n + j * n + k] = col[j];
                }
            }
        }
    }

    // ── Transpose i-slab → j-pencil (communication) ─────────────────────────
    // Split real and imaginary parts for the transpose (MpiContext works on Real).
    let re: Vec<Real> = data.iter().map(|c| c.re).collect();
    let im: Vec<Real> = data.iter().map(|c| c.im).collect();
    let re_j = mpi.transpose_i_to_j(&re, n);
    let im_j = mpi.transpose_i_to_j(&im, n);

    let n_local_j = mpi.slab_count(n); // same partitioning for j after transpose
    let mut data_j: Vec<Complex<f64>> = re_j
        .iter()
        .zip(im_j.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    // ── Forward pass 3: 1D FFTs along i-axis (now contiguous after transpose) ─
    // After transpose_i_to_j, layout is (n_local_j, n, n) with full i-range in
    // the middle index.  For each local j-slab and each k, extract the i-column,
    // transform, and write back.
    {
        let fft = planner.plan_fft_forward(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for lj in 0..n_local_j {
            for k in 0..n {
                // Gather i-column: data_j[lj * n * n + i * n + k] for i in 0..n
                for i in 0..n {
                    col[i] = data_j[lj * n * n + i * n + k];
                }
                fft.process(&mut col);
                for i in 0..n {
                    data_j[lj * n * n + i * n + k] = col[i];
                }
            }
        }
    }

    // ── Apply Green's function ───────────────────────────────────────────────
    // After transpose, rank owns j-indices [j_lo..j_hi).
    // Local index [lj][i][k] → global (i, j_lo + lj, k).
    let j_lo = mpi.slab_start(n);
    let inv_dx2 = 1.0 / (dx * dx);
    for lj in 0..n_local_j {
        let gj = j_lo + lj;
        for i in 0..n {
            for k in 0..n {
                let idx = lj * n * n + i * n + k;
                if i == 0 && gj == 0 && k == 0 {
                    data_j[idx] = Complex::new(0.0, 0.0);
                    continue;
                }
                let si = (PI * i as f64 / n as f64).sin();
                let sj = (PI * gj as f64 / n as f64).sin();
                let sk = (PI * k as f64 / n as f64).sin();
                let lambda = -4.0 * inv_dx2 * (si * si + sj * sj + sk * sk);
                data_j[idx] /= lambda;
            }
        }
    }

    // ── Inverse pass 1: inverse 1D FFTs along i-axis ─────────────────────────
    {
        let ifft = planner.plan_fft_inverse(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for lj in 0..n_local_j {
            for k in 0..n {
                for i in 0..n {
                    col[i] = data_j[lj * n * n + i * n + k];
                }
                ifft.process(&mut col);
                for i in 0..n {
                    data_j[lj * n * n + i * n + k] = col[i];
                }
            }
        }
    }

    // ── Transpose j-pencil → i-slab (communication) ─────────────────────────
    let re_j2: Vec<Real> = data_j.iter().map(|c| c.re).collect();
    let im_j2: Vec<Real> = data_j.iter().map(|c| c.im).collect();
    let re_i = mpi.transpose_j_to_i(&re_j2, n);
    let im_i = mpi.transpose_j_to_i(&im_j2, n);

    let mut data: Vec<Complex<f64>> = re_i
        .iter()
        .zip(im_i.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    // ── Inverse pass 2: inverse 1D FFTs along j-axis ─────────────────────────
    {
        let ifft = planner.plan_fft_inverse(n);
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n_local_i {
            for k in 0..n {
                for j in 0..n {
                    col[j] = data[i * n * n + j * n + k];
                }
                ifft.process(&mut col);
                for j in 0..n {
                    data[i * n * n + j * n + k] = col[j];
                }
            }
        }
    }

    // ── Inverse pass 3: inverse 1D FFTs along k-axis (contiguous) ────────────
    {
        let ifft = planner.plan_fft_inverse(n);
        for i in 0..n_local_i {
            for j in 0..n {
                let start = i * n * n + j * n;
                ifft.process(&mut data[start..start + n]);
            }
        }
    }

    // ── Normalise and subtract mean ──────────────────────────────────────────
    let n3 = (n * n * n) as f64;
    let mut phi: Vec<Real> = data.iter().map(|c| c.re / n3).collect();
    let local_sum: f64 = phi.iter().sum();
    let global_sum = mpi.allreduce_sum(local_sum);
    let mean = global_sum / n3;
    phi.iter_mut().for_each(|p| *p -= mean);
    phi
}

/// Primary solver: FFT when available (faster and exact), multigrid otherwise.
///
/// In distributed MPI builds with `fft_pm`, delegates to `solve_poisson_fft_mpi`.
/// The `mpi` parameter is only used when both `fft_pm` and `mpi_parallel` features
/// are active; otherwise it is ignored.
pub fn solve_poisson(
    source: &[Real],
    dx: Real,
    n: usize,
    epsilon: Real,
    max_iter: u32,
    n_smooth: usize,
    mpi: &crate::mpi::MpiContext,
) -> Vec<Real> {
    #[cfg(all(feature = "fft_pm", feature = "mpi_parallel"))]
    {
        let _ = (epsilon, max_iter, n_smooth);
        // source is the full N^3 grid (after allreduce_sum in simulation.rs).
        // Extract this rank's i-slab before calling the distributed FFT solver,
        // then allgather the local phi back into a full grid for grad_phi.
        let i_lo = mpi.slab_start(n);
        let i_hi = mpi.slab_end(n);
        let local_src = &source[i_lo * n * n..i_hi * n * n];
        let local_phi = solve_poisson_fft_mpi(local_src, dx, n, mpi);
        mpi.allgather_slabs(&local_phi, n)
    }
    #[cfg(all(feature = "fft_pm", not(feature = "mpi_parallel")))]
    {
        let _ = (epsilon, max_iter, n_smooth, mpi);
        solve_poisson_fft(source, dx, n)
    }
    #[cfg(not(feature = "fft_pm"))]
    {
        let _ = mpi;
        solve_poisson_mg(source, dx, n, epsilon, max_iter, n_smooth)
    }
}

/// Primary solver variant that writes into pre-allocated scratch buffers.
///
/// For the FFT-PM path (non-MPI), uses `solve_poisson_fft_into` to avoid
/// per-step heap allocations.  For MPI and multigrid paths, falls back to
/// allocating internally and copying into `phi`.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn solve_poisson_into(
    source: &[Real],
    dx: Real,
    n: usize,
    epsilon: Real,
    max_iter: u32,
    n_smooth: usize,
    mpi: &crate::mpi::MpiContext,
    phi: &mut [Real],
    #[cfg(feature = "fft_pm")] fft_buf: &mut [rustfft::num_complex::Complex<f64>],
    #[cfg(feature = "fft_pm")] fft_col: &mut [rustfft::num_complex::Complex<f64>],
) {
    #[cfg(all(feature = "fft_pm", not(feature = "mpi_parallel")))]
    {
        let _ = (epsilon, max_iter, n_smooth, mpi);
        solve_poisson_fft_into(source, dx, n, phi, fft_buf, fft_col);
    }
    // MPI and multigrid paths: allocate internally, copy result into phi
    #[cfg(all(feature = "fft_pm", feature = "mpi_parallel"))]
    {
        let _ = (epsilon, max_iter, n_smooth, fft_buf, fft_col);
        let i_lo = mpi.slab_start(n);
        let i_hi = mpi.slab_end(n);
        let local_src = &source[i_lo * n * n..i_hi * n * n];
        let local_phi = solve_poisson_fft_mpi(local_src, dx, n, mpi);
        let result = mpi.allgather_slabs(&local_phi, n);
        phi.copy_from_slice(&result);
    }
    #[cfg(not(feature = "fft_pm"))]
    {
        let _ = mpi;
        let result = solve_poisson_mg(source, dx, n, epsilon, max_iter, n_smooth);
        phi.copy_from_slice(&result);
    }
}

// ─── Gradient ─────────────────────────────────────────────────────────────────

/// Compute  g = −∇Φ  into a pre-allocated buffer.
///
/// `g` must have length `n³`.  Use this variant in the step loop to
/// avoid per-step heap allocations.
pub fn grad_phi_into(g: &mut [[Real; 3]], phi: &[Real], dx: Real, n: usize) {
    let inv_2dx = 0.5 / dx;
    assert_eq!(g.len(), n * n * n);
    g.par_iter_mut().enumerate().for_each(|(c, gc)| {
        let (i, j, k) = from_flat(c, n);
        let (i, j, k) = (i as isize, j as isize, k as isize);
        gc[0] = -(phi[idx3p(i + 1, j, k, n)] - phi[idx3p(i - 1, j, k, n)]) * inv_2dx;
        gc[1] = -(phi[idx3p(i, j + 1, k, n)] - phi[idx3p(i, j - 1, k, n)]) * inv_2dx;
        gc[2] = -(phi[idx3p(i, j, k + 1, n)] - phi[idx3p(i, j, k - 1, n)]) * inv_2dx;
    });
}

/// Compute  g = −∇Φ  via 2nd-order centred differences.
///
/// Parallelised over the flat cell index.  Returns `[gx, gy, gz]` per cell.
pub fn grad_phi(phi: &[Real], dx: Real, n: usize) -> Vec<[Real; 3]> {
    let inv_2dx = 0.5 / dx;
    let mut g = vec![[0.0_f64; 3]; n * n * n];
    g.par_iter_mut().enumerate().for_each(|(c, gc)| {
        let (i, j, k) = from_flat(c, n);
        let (i, j, k) = (i as isize, j as isize, k as isize);
        gc[0] = -(phi[idx3p(i + 1, j, k, n)] - phi[idx3p(i - 1, j, k, n)]) * inv_2dx;
        gc[1] = -(phi[idx3p(i, j + 1, k, n)] - phi[idx3p(i, j - 1, k, n)]) * inv_2dx;
        gc[2] = -(phi[idx3p(i, j, k + 1, n)] - phi[idx3p(i, j, k - 1, n)]) * inv_2dx;
    });
    g
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::types::idx3p;
    use std::f64::consts::PI;

    fn max_err(a: &[Real], b: &[Real]) -> Real {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn make_mode(n: usize, mode: usize) -> (Vec<Real>, Vec<Real>) {
        let dx = 1.0 / n as Real;
        let k = 2.0 * PI * mode as Real;
        let mut src = vec![0.0_f64; n * n * n];
        let mut phi = vec![0.0_f64; n * n * n];
        for i in 0..n as isize {
            for j in 0..n as isize {
                for k2 in 0..n as isize {
                    let c = idx3p(i, j, k2, n);
                    let x = (i as Real + 0.5) * dx;
                    let v = (k * x).sin();
                    src[c] = -k * k * v;
                    phi[c] = v;
                }
            }
        }
        (src, phi)
    }

    fn sub_mean(v: &[Real]) -> Vec<Real> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect()
    }

    // ── Multigrid ─────────────────────────────────────────────────────────────

    #[test]
    fn mg_single_mode_accuracy() {
        let n = 32_usize;
        let dx = 1.0 / n as Real;
        let (src, exact) = make_mode(n, 1);
        let phi = solve_poisson_mg(&src, dx, n, 1e-9, 300, 2);
        let err = max_err(&sub_mean(&phi), &sub_mean(&exact));
        println!("MG error: {err:.3e}");
        assert!(err < 5e-3, "MG error {err:.3e}");
    }

    #[test]
    fn mg_convergence_order() {
        let errs: Vec<Real> = [8_usize, 16, 32]
            .iter()
            .map(|&n| {
                let dx = 1.0 / n as Real;
                let (src, exact) = make_mode(n, 1);
                let phi = solve_poisson_mg(&src, dx, n, 1e-10, 500, 3);
                max_err(&sub_mean(&phi), &sub_mean(&exact))
            })
            .collect();
        let r1 = errs[0] / errs[1];
        let r2 = errs[1] / errs[2];
        println!("MG conv ratios: {r1:.2}, {r2:.2}");
        assert!(r1 > 2.0 && r2 > 2.0);
    }

    #[test]
    fn mg_residual_decreases() {
        let n = 16_usize;
        let dx = 1.0 / n as Real;
        let (src, _) = make_mode(n, 1);
        let src_zm = sub_mean(&src);
        let mut phi = vec![0.0_f64; n * n * n];
        let r0 = residual(&phi, &src_zm, dx, n)
            .iter()
            .map(|r| r.abs())
            .fold(0.0, f64::max);
        v_cycle(&mut phi, &src_zm, dx, n, 2, 4);
        let r1 = residual(&phi, &src_zm, dx, n)
            .iter()
            .map(|r| r.abs())
            .fold(0.0, f64::max);
        println!("Residual: {r0:.3e} → {r1:.3e}");
        assert!(r1 < r0);
    }

    // ── FFT ───────────────────────────────────────────────────────────────────

    #[cfg(feature = "fft_pm")]
    #[test]
    fn fft_single_mode_accuracy() {
        let n = 32_usize;
        let dx = 1.0 / n as Real;
        let (src, exact) = make_mode(n, 1);
        let phi = solve_poisson_fft(&src, dx, n);
        let err = max_err(&sub_mean(&phi), &sub_mean(&exact));
        println!("FFT error: {err:.3e}");
        assert!(err < 5e-3, "FFT error {err:.3e}"); // discretisation error O((k*dx)²)
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn fft_exact_discrete_residual() {
        // The FFT solves the DISCRETE Laplacian exactly (residual → machine eps).
        let n = 16_usize;
        let dx = 1.0 / n as Real;
        let (src, _) = make_mode(n, 1);
        let src_zm = sub_mean(&src);
        let phi = solve_poisson_fft(&src_zm, dx, n);
        // Check ∇²_disc phi = src_zm
        let lap: Vec<Real> = (0..n * n * n)
            .map(|c| {
                let (i, j, k) = from_flat(c, n);
                let (i, j, k) = (i as isize, j as isize, k as isize);
                let inv_dx2 = 1.0 / (dx * dx);
                (phi[idx3p(i + 1, j, k, n)]
                    + phi[idx3p(i - 1, j, k, n)]
                    + phi[idx3p(i, j + 1, k, n)]
                    + phi[idx3p(i, j - 1, k, n)]
                    + phi[idx3p(i, j, k + 1, n)]
                    + phi[idx3p(i, j, k - 1, n)]
                    - 6.0 * phi[c])
                    * inv_dx2
            })
            .collect();
        let res_err = max_err(&lap, &src_zm);
        println!("FFT discrete residual: {res_err:.3e}");
        assert!(res_err < 1e-9, "residual {res_err:.3e}");
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn fft_agrees_with_mg() {
        let n = 16_usize;
        let dx = 1.0 / n as Real;
        let (src, _) = make_mode(n, 1);
        let phi_fft = solve_poisson_fft(&src, dx, n);
        let phi_mg = solve_poisson_mg(&src, dx, n, 1e-9, 300, 3);
        let err = max_err(&sub_mean(&phi_fft), &sub_mean(&phi_mg));
        println!("FFT vs MG: {err:.3e}");
        assert!(err < 1e-3);
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn fft_linearity() {
        let n = 8_usize;
        let dx = 1.0 / n as Real;
        let (s1, _) = make_mode(n, 1);
        let (s2, _) = make_mode(n, 2);
        let phi12 = solve_poisson_fft(
            &s1.iter()
                .zip(s2.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
            dx,
            n,
        );
        let phi1 = solve_poisson_fft(&s1, dx, n);
        let phi2 = solve_poisson_fft(&s2, dx, n);
        let err = max_err(
            &phi12,
            &phi1
                .iter()
                .zip(phi2.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
        );
        assert!(err < 1e-10, "FFT not linear: {err:.3e}");
    }

    // ── Distributed FFT (single-rank, no MPI runtime needed) ───────────────

    #[cfg(all(feature = "fft_pm", feature = "mpi_parallel"))]
    #[test]
    fn mpi_fft_single_rank_matches_serial() {
        // On a single MPI rank the distributed FFT must give the same result
        // as the serial FFT (bitwise identical modulo FP ordering).
        let n = 16_usize;
        let dx = 1.0 / n as Real;
        let (src, _) = make_mode(n, 1);
        let mpi = crate::mpi::MpiContext::init();
        let phi_mpi = solve_poisson_fft_mpi(&src, dx, n, &mpi);
        let phi_ser = solve_poisson_fft(&src, dx, n);
        let err = max_err(&sub_mean(&phi_mpi), &sub_mean(&phi_ser));
        println!("MPI FFT vs serial FFT: {err:.3e}");
        assert!(err < 1e-9, "MPI FFT vs serial FFT: {err:.3e}");
    }

    // ── Gradient ──────────────────────────────────────────────────────────────

    #[test]
    fn gradient_accuracy_x() {
        let n = 64_usize;
        let dx = 1.0 / n as Real;
        let k = 2.0 * PI;
        let phi: Vec<Real> = (0..n * n * n)
            .map(|c| {
                let (i, _, _) = from_flat(c, n);
                (k * (i as Real + 0.5) * dx).sin()
            })
            .collect();
        let g = grad_phi(&phi, dx, n);
        let err = (0..n * n * n)
            .map(|c| {
                let (i, _, _) = from_flat(c, n);
                (g[c][0] - (-k * (k * (i as Real + 0.5) * dx).cos())).abs()
            })
            .fold(0.0_f64, f64::max);
        println!("Gradient x-error (n=64): {err:.3e}");
        assert!(err < 0.015);
    }

    #[test]
    fn gradient_convergence_order() {
        let k = 2.0 * PI;
        let errs: Vec<Real> = [16_usize, 32, 64]
            .iter()
            .map(|&n| {
                let dx = 1.0 / n as Real;
                let phi: Vec<Real> = (0..n * n * n)
                    .map(|c| {
                        let (i, _, _) = from_flat(c, n);
                        (k * (i as Real + 0.5) * dx).sin()
                    })
                    .collect();
                let g = grad_phi(&phi, dx, n);
                (0..n * n * n)
                    .map(|c| {
                        let (i, _, _) = from_flat(c, n);
                        (g[c][0] - (-k * (k * (i as Real + 0.5) * dx).cos())).abs()
                    })
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let r1 = errs[0] / errs[1];
        let r2 = errs[1] / errs[2];
        println!("Gradient conv: {r1:.2}, {r2:.2}");
        assert!(r1 > 2.5 && r2 > 2.5);
    }

    #[test]
    fn gradient_yz_zero_for_x_field() {
        let n = 16_usize;
        let dx = 1.0 / n as Real;
        let k = 2.0 * PI;
        let phi: Vec<Real> = (0..n * n * n)
            .map(|c| {
                let (i, _, _) = from_flat(c, n);
                (k * (i as Real + 0.5) * dx).sin()
            })
            .collect();
        let g = grad_phi(&phi, dx, n);
        let max_gy = g.iter().map(|gc| gc[1].abs()).fold(0.0, f64::max);
        let max_gz = g.iter().map(|gc| gc[2].abs()).fold(0.0, f64::max);
        assert!(max_gy < 1e-14, "gy = {max_gy:.3e}");
        assert!(max_gz < 1e-14, "gz = {max_gz:.3e}");
    }

    // ── Restriction / Prolongation ────────────────────────────────────────────

    #[test]
    fn restrict_constant_field() {
        let n = 8_usize;
        let fine = vec![3.14_f64; n * n * n];
        let coarse = restrict_uniform(&fine, n);
        assert!(coarse.iter().all(|&c| (c - 3.14).abs() < 1e-12));
    }

    #[test]
    fn restrict_injection_values() {
        let n = 8_usize;
        let fine: Vec<Real> = (0..n * n * n).map(|c| c as Real).collect();
        let coarse = restrict_uniform(&fine, n);
        let nc = n / 2;
        for i in 0..nc as isize {
            for j in 0..nc as isize {
                for k in 0..nc as isize {
                    let expected = fine[idx3p(2 * i, 2 * j, 2 * k, n)];
                    let got = coarse[idx3p(i, j, k, nc)];
                    assert!(
                        (got - expected).abs() < 1e-14,
                        "({i},{j},{k}): {got} ≠ {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn prolongate_constant_roundtrip() {
        let n = 8_usize;
        let fine_in = vec![2.71_f64; n * n * n];
        let coarse = restrict_uniform(&fine_in, n);
        let fine_out = prolongate_uniform(&coarse, n / 2);
        let err = max_err(&fine_in, &fine_out);
        assert!(err < 1e-12, "Constant roundtrip error: {err:.3e}");
    }
}
