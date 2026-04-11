//! MUSCL-Hancock Godunov hydro solver.
//!
//! Operates on the flat n³ gas arrays from `SimScratch`.
//! Each axis is swept independently (Lie-Trotter splitting).

use crate::hydro::riemann::{hll_flux, minmod, pressure};

const GAMMA: f64 = 5.0 / 3.0;

// ── Primitive ↔ Conserved conversions ────────────────────────────────────────

/// Primitive W = [ρ, vx, vy, vz, P]  →  Conserved U = [ρ, ρvx, ρvy, ρvz, E].
#[inline]
pub fn prim_to_cons(w: &[f64; 5], gamma: f64) -> [f64; 5] {
    let e = w[4] / (gamma - 1.0)
          + 0.5 * w[0] * (w[1]*w[1] + w[2]*w[2] + w[3]*w[3]);
    [w[0], w[0]*w[1], w[0]*w[2], w[0]*w[3], e]
}

/// Conserved U  →  Primitive W.
#[inline]
pub fn cons_to_prim(u: &[f64; 5], gamma: f64) -> [f64; 5] {
    let rho = u[0].max(1e-30);
    let vx  = u[1] / rho;
    let vy  = u[2] / rho;
    let vz  = u[3] / rho;
    let p   = pressure(u, gamma).max(1e-30 * rho);
    [rho, vx, vy, vz, p]
}

// ── Hancock half-step predictor ───────────────────────────────────────────────

/// Given primitive W and reconstructed slopes dW/dx, predict the half-step
/// evolution in the x-direction over Δt/2.
///
/// Euler equations in primitive form (x-sweep only):
///   ∂ρ/∂t  = −vx ∂ρ/∂x  − ρ ∂vx/∂x
///   ∂vx/∂t = −vx ∂vx/∂x − (1/ρ) ∂P/∂x
///   ∂vy/∂t = −vx ∂vy/∂x
///   ∂vz/∂t = −vx ∂vz/∂x
///   ∂P/∂t  = −γP ∂vx/∂x − vx ∂P/∂x
pub fn hancock_predict_x(w: &[f64; 5], dw: &[f64; 5], dt_half_over_dx: f64, gamma: f64) -> [f64; 5] {
    let [rho, vx, vy, vz, p] = *w;
    let [drho, dvx, dvy, dvz, dp] = *dw;
    let c = dt_half_over_dx; // Δt/(2Δx)
    [
        rho - c * (vx * drho + rho * dvx),
        vx  - c * (vx * dvx + dp / rho),
        vy  - c * (vx * dvy),
        vz  - c * (vx * dvz),
        p   - c * (gamma * p * dvx + vx * dp),
    ]
}

// ── 1D x-sweep ────────────────────────────────────────────────────────────────

/// Apply one Godunov sweep in the x-direction to the full n³ grid.
///
/// `gas_rho`, `gas_vel`, `gas_u` are flat C-order arrays of length n³.
/// `dx` is cell width [Mpc/h], `dt` is timestep [supercomoving τ].
/// `n` is the grid size per side.
///
/// Periodic boundary conditions.
pub fn sweep_x(
    gas_rho: &mut [f64],
    gas_vel: &mut [[f64; 3]],
    gas_u:   &mut [f64],
    dx: f64,
    dt: f64,
    n:  usize,
    gamma: f64,
) {
    let n2 = n * n;
    // Scratch: one conserved-variable array per cell
    let u_old: Vec<[f64; 5]> = (0..n*n*n).map(|idx| {
        let w = [gas_rho[idx], gas_vel[idx][0], gas_vel[idx][1],
                 gas_vel[idx][2], (gamma - 1.0) * gas_rho[idx] * gas_u[idx]];
        prim_to_cons(&w, gamma)
    }).collect();

    let mut u_new = u_old.clone();
    let dtdx  = dt / dx;
    let dt2dx = 0.5 * dtdx;

    // For each (j,k) column, sweep along i (x-axis)
    for j in 0..n {
        for k in 0..n {
            let col: Vec<[f64; 5]> = (0..n)
                .map(|i| u_old[i * n2 + j * n + k])
                .collect();

            // Convert column to primitive
            let wc: Vec<[f64; 5]> = col.iter()
                .map(|u| cons_to_prim(u, gamma))
                .collect();

            // Compute MinMod slopes
            let slopes: Vec<[f64; 5]> = (0..n).map(|i| {
                let il = if i == 0 { n - 1 } else { i - 1 };
                let ir = if i == n - 1 { 0 } else { i + 1 };
                let mut s = [0.0f64; 5];
                for q in 0..5 {
                    s[q] = minmod(wc[i][q] - wc[il][q], wc[ir][q] - wc[i][q]);
                }
                s
            }).collect();

            // Compute interface fluxes
            for i in 0..n {
                let ir = if i == n - 1 { 0 } else { i + 1 };

                // Hancock half-step on left and right cells
                let w_l = hancock_predict_x(&wc[i],  &slopes[i],  dt2dx, gamma);
                let w_r = hancock_predict_x(&wc[ir], &slopes[ir], dt2dx, gamma);

                // Right interface state of cell i: W_i + ½slope
                // Left  interface state of cell ir: W_ir − ½slope
                let mut if_l = [0.0f64; 5];
                let mut if_r = [0.0f64; 5];
                for q in 0..5 {
                    if_l[q] = w_l[q] + 0.5 * slopes[i][q];
                    if_r[q] = w_r[q] - 0.5 * slopes[ir][q];
                }

                // Floor densities and pressures
                if_l[0] = if_l[0].max(1e-30);
                if_r[0] = if_r[0].max(1e-30);
                if_l[4] = if_l[4].max(1e-30);
                if_r[4] = if_r[4].max(1e-30);

                let u_if_l = prim_to_cons(&if_l, gamma);
                let u_if_r = prim_to_cons(&if_r, gamma);
                let flux = hll_flux(&u_if_l, &u_if_r, gamma);

                // Update: cell i loses flux, cell ir gains flux
                for q in 0..5 {
                    u_new[i  * n2 + j * n + k][q] -= dtdx * flux[q];
                    u_new[ir * n2 + j * n + k][q] += dtdx * flux[q];
                }
            }
        }
    }

    // Write back
    for idx in 0..n*n*n {
        let w = cons_to_prim(&u_new[idx], gamma);
        gas_rho[idx]    = w[0].max(1e-30);
        gas_vel[idx][0] = w[1];
        gas_vel[idx][1] = w[2];
        gas_vel[idx][2] = w[3];
        gas_u[idx]      = (w[4] / ((gamma - 1.0) * w[0])).max(1e-30);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform state should be preserved (no net flux).
    #[test]
    fn sweep_x_preserves_uniform_state() {
        let n = 8;
        let n3 = n * n * n;
        let mut rho = vec![1.0f64; n3];
        let mut vel = vec![[0.0f64; 3]; n3];
        let mut u   = vec![1.5f64; n3]; // P = (γ-1)*ρ*u = 1.0 for γ=5/3

        sweep_x(&mut rho, &mut vel, &mut u, 1.0, 0.1, n, GAMMA);

        for i in 0..n3 {
            assert!((rho[i] - 1.0).abs() < 1e-12, "rho[{i}]={}", rho[i]);
            assert!((u[i]   - 1.5).abs() < 1e-12, "u[{i}]={}",   u[i]);
        }
    }

    /// Mass is conserved across a sweep (periodic BC).
    #[test]
    fn sweep_x_conserves_mass() {
        let n = 16;
        let n3 = n * n * n;
        // Random-ish density field
        let mut rho: Vec<f64> = (0..n3).map(|i| 0.5 + 0.5 * ((i as f64 * 0.3).sin())).collect();
        let mut vel = vec![[0.1f64, 0.0, 0.0]; n3];
        let mut u   = vec![1.5f64; n3];

        let mass_before: f64 = rho.iter().sum();
        sweep_x(&mut rho, &mut vel, &mut u, 1.0, 0.05, n, GAMMA);
        let mass_after: f64 = rho.iter().sum();

        assert!((mass_after - mass_before).abs() / mass_before < 1e-10,
            "mass not conserved: {mass_before} → {mass_after}");
    }
}
