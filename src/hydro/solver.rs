//! MUSCL-Hancock Godunov hydro solver.
//!
//! Operates on the flat n³ gas arrays from `SimScratch`.
//! Each axis is swept independently (Lie-Trotter splitting).

use crate::hydro::riemann::{hll_flux, minmod, pressure};

#[allow(dead_code)] // used in tests
const GAMMA: f64 = 5.0 / 3.0;

// ── Primitive ↔ Conserved conversions ────────────────────────────────────────

/// Primitive W = [ρ, vx, vy, vz, P]  →  Conserved U = [ρ, ρvx, ρvy, ρvz, E].
#[inline]
pub fn prim_to_cons(w: &[f64; 5], gamma: f64) -> [f64; 5] {
    let e = w[4] / (gamma - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
    [w[0], w[0] * w[1], w[0] * w[2], w[0] * w[3], e]
}

/// Conserved U  →  Primitive W.
#[inline]
pub fn cons_to_prim(u: &[f64; 5], gamma: f64) -> [f64; 5] {
    let rho = u[0].max(1e-30);
    let vx = u[1] / rho;
    let vy = u[2] / rho;
    let vz = u[3] / rho;
    let p = pressure(u, gamma).max(1e-30 * rho);
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
pub fn hancock_predict_x(
    w: &[f64; 5],
    dw: &[f64; 5],
    dt_half_over_dx: f64,
    gamma: f64,
) -> [f64; 5] {
    let [rho, vx, vy, vz, p] = *w;
    let [drho, dvx, dvy, dvz, dp] = *dw;
    let c = dt_half_over_dx; // Δt/(2Δx)
    [
        rho - c * (vx * drho + rho * dvx),
        vx - c * (vx * dvx + dp / rho),
        vy - c * (vx * dvy),
        vz - c * (vx * dvz),
        p - c * (gamma * p * dvx + vx * dp),
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
    gas_u: &mut [f64],
    dx: f64,
    dt: f64,
    n: usize,
    gamma: f64,
) {
    let n2 = n * n;
    // Scratch: one conserved-variable array per cell
    let u_old: Vec<[f64; 5]> = (0..n * n * n)
        .map(|idx| {
            let w = [
                gas_rho[idx],
                gas_vel[idx][0],
                gas_vel[idx][1],
                gas_vel[idx][2],
                (gamma - 1.0) * gas_rho[idx] * gas_u[idx],
            ];
            prim_to_cons(&w, gamma)
        })
        .collect();

    let mut u_new = u_old.clone();
    let dtdx = dt / dx;
    let dt2dx = 0.5 * dtdx;

    // For each (j,k) column, sweep along i (x-axis)
    for j in 0..n {
        for k in 0..n {
            let col: Vec<[f64; 5]> = (0..n).map(|i| u_old[i * n2 + j * n + k]).collect();

            // Convert column to primitive
            let wc: Vec<[f64; 5]> = col.iter().map(|u| cons_to_prim(u, gamma)).collect();

            // Compute MinMod slopes
            let slopes: Vec<[f64; 5]> = (0..n)
                .map(|i| {
                    let il = if i == 0 { n - 1 } else { i - 1 };
                    let ir = if i == n - 1 { 0 } else { i + 1 };
                    let mut s = [0.0f64; 5];
                    for q in 0..5 {
                        s[q] = minmod(wc[i][q] - wc[il][q], wc[ir][q] - wc[i][q]);
                    }
                    s
                })
                .collect();

            // Compute interface fluxes
            for i in 0..n {
                let ir = if i == n - 1 { 0 } else { i + 1 };

                // Hancock half-step on left and right cells
                let w_l = hancock_predict_x(&wc[i], &slopes[i], dt2dx, gamma);
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
                    u_new[i * n2 + j * n + k][q] -= dtdx * flux[q];
                    u_new[ir * n2 + j * n + k][q] += dtdx * flux[q];
                }
            }
        }
    }

    // Write back
    for idx in 0..n * n * n {
        let w = cons_to_prim(&u_new[idx], gamma);
        gas_rho[idx] = w[0].max(1e-30);
        gas_vel[idx][0] = w[1];
        gas_vel[idx][1] = w[2];
        gas_vel[idx][2] = w[3];
        gas_u[idx] = (w[4] / ((gamma - 1.0) * w[0])).max(1e-30);
    }
}

// ── Axis-permuted sweeps ──────────────────────────────────────────────────────

/// Transpose gas arrays to bring axis `ax` into the x-position,
/// call sweep_x, then transpose back.
#[allow(clippy::too_many_arguments)]
pub fn sweep_axis(
    gas_rho: &mut [f64],
    gas_vel: &mut [[f64; 3]],
    gas_u: &mut [f64],
    dx: f64,
    dt: f64,
    n: usize,
    gamma: f64,
    ax: usize, // 0=x, 1=y, 2=z
    p_rho: &mut [f64],
    p_vel: &mut [[f64; 3]],
    p_u: &mut [f64],
) {
    if ax == 0 {
        sweep_x(gas_rho, gas_vel, gas_u, dx, dt, n, gamma);
        return;
    }

    // ax=1: permuted (a,b,c) → original (b,a,c)
    // ax=2: permuted (a,b,c) → original (c,b,a)
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let p_idx = a * n * n + b * n + c;
                let o_idx = match ax {
                    1 => b * n * n + a * n + c,
                    2 => c * n * n + b * n + a,
                    _ => unreachable!(),
                };
                p_rho[p_idx] = gas_rho[o_idx];
                p_u[p_idx] = gas_u[o_idx];
                p_vel[p_idx] = match ax {
                    1 => [gas_vel[o_idx][1], gas_vel[o_idx][0], gas_vel[o_idx][2]],
                    2 => [gas_vel[o_idx][2], gas_vel[o_idx][1], gas_vel[o_idx][0]],
                    _ => unreachable!(),
                };
            }
        }
    }

    sweep_x(p_rho, p_vel, p_u, dx, dt, n, gamma);

    // Permute back
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let p_idx = a * n * n + b * n + c;
                let o_idx = match ax {
                    1 => b * n * n + a * n + c,
                    2 => c * n * n + b * n + a,
                    _ => unreachable!(),
                };
                gas_rho[o_idx] = p_rho[p_idx];
                gas_u[o_idx] = p_u[p_idx];
                gas_vel[o_idx] = match ax {
                    1 => [p_vel[p_idx][1], p_vel[p_idx][0], p_vel[p_idx][2]],
                    2 => [p_vel[p_idx][2], p_vel[p_idx][1], p_vel[p_idx][0]],
                    _ => unreachable!(),
                };
            }
        }
    }
}

// ── CFL + full 3D step ────────────────────────────────────────────────────────

/// Maximum stable hydro timestep: Δt = C_cfl × dx / max(|v| + c_s).
pub fn hydro_cfl(
    gas_rho: &[f64],
    gas_vel: &[[f64; 3]],
    gas_u: &[f64],
    dx: f64,
    courant: f64,
    gamma: f64,
) -> f64 {
    let v_max = gas_rho
        .iter()
        .enumerate()
        .map(|(i, &rho)| {
            let p = (gamma - 1.0) * rho * gas_u[i];
            let cs = (gamma * p / rho.max(1e-30)).sqrt();
            let v = (gas_vel[i][0].powi(2) + gas_vel[i][1].powi(2) + gas_vel[i][2].powi(2)).sqrt();
            v + cs
        })
        .fold(0.0_f64, f64::max);
    courant * dx / v_max.max(1e-10)
}

/// Full 3D hydro step: sweep x, then y, then z (Lie-Trotter splitting).
///
/// Also applies gravitational acceleration `grav` as a body-force source term
/// (operator-split after the flux update).
/// If `fifth_force` is `Some(...)`, it is added to the body force as well.
#[allow(clippy::too_many_arguments)]
pub fn step(
    gas_rho: &mut [f64],
    gas_vel: &mut [[f64; 3]],
    gas_u: &mut [f64],
    grav: &[[f64; 3]],
    fifth_force: Option<&[[f64; 3]]>,
    dx: f64,
    dt: f64,
    n: usize,
    gamma: f64,
    p_rho: &mut [f64],
    p_vel: &mut [[f64; 3]],
    p_u: &mut [f64],
) {
    sweep_axis(
        gas_rho, gas_vel, gas_u, dx, dt, n, gamma, 0, p_rho, p_vel, p_u,
    );
    sweep_axis(
        gas_rho, gas_vel, gas_u, dx, dt, n, gamma, 1, p_rho, p_vel, p_u,
    );
    sweep_axis(
        gas_rho, gas_vel, gas_u, dx, dt, n, gamma, 2, p_rho, p_vel, p_u,
    );

    // Gravity + fifth-force body force: operator-split kick on gas velocity and energy.
    for idx in 0..gas_rho.len() {
        let mut acc = grav[idx];
        if let Some(ff) = fifth_force {
            acc[0] += ff[idx][0];
            acc[1] += ff[idx][1];
            acc[2] += ff[idx][2];
        }
        // Gravity acts on kinetic energy only: Δv = a·dt.
        // Internal energy u is unchanged — the work done by gravity (v·a·dt) goes
        // entirely into ½|v|², so Δu = 0 for a gravity-only operator-split kick.
        gas_vel[idx][0] += acc[0] * dt;
        gas_vel[idx][1] += acc[1] * dt;
        gas_vel[idx][2] += acc[2] * dt;
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
        let mut u = vec![1.5f64; n3]; // P = (γ-1)*ρ*u = 1.0 for γ=5/3

        sweep_x(&mut rho, &mut vel, &mut u, 1.0, 0.1, n, GAMMA);

        for i in 0..n3 {
            assert!((rho[i] - 1.0).abs() < 1e-12, "rho[{i}]={}", rho[i]);
            assert!((u[i] - 1.5).abs() < 1e-12, "u[{i}]={}", u[i]);
        }
    }

    /// Mass is conserved across a sweep (periodic BC).
    #[test]
    fn sweep_x_conserves_mass() {
        let n = 16;
        let n3 = n * n * n;
        // Random-ish density field
        let mut rho: Vec<f64> = (0..n3)
            .map(|i| 0.5 + 0.5 * ((i as f64 * 0.3).sin()))
            .collect();
        let mut vel = vec![[0.1f64, 0.0, 0.0]; n3];
        let mut u = vec![1.5f64; n3];

        let mass_before: f64 = rho.iter().sum();
        sweep_x(&mut rho, &mut vel, &mut u, 1.0, 0.05, n, GAMMA);
        let mass_after: f64 = rho.iter().sum();

        assert!(
            (mass_after - mass_before).abs() / mass_before < 1e-10,
            "mass not conserved: {mass_before} → {mass_after}"
        );
    }

    #[test]
    fn sweep_y_conserves_mass() {
        let n = 8;
        let n3 = n * n * n;
        let mut rho: Vec<f64> = (0..n3).map(|i| 1.0 + 0.2 * ((i as f64).sin())).collect();
        let mut vel = vec![[0.0f64, 0.1, 0.0]; n3];
        let mut u = vec![1.5f64; n3];
        let mass_before: f64 = rho.iter().sum();
        let mut p_rho = vec![0.0f64; n3];
        let mut p_vel = vec![[0.0f64; 3]; n3];
        let mut p_u = vec![0.0f64; n3];
        sweep_axis(
            &mut rho, &mut vel, &mut u, 1.0, 0.05, n, GAMMA, 1, &mut p_rho, &mut p_vel, &mut p_u,
        );
        let mass_after: f64 = rho.iter().sum();
        assert!(
            (mass_after - mass_before).abs() / mass_before < 1e-10,
            "mass not conserved in y: {mass_before} → {mass_after}"
        );
    }

    #[test]
    fn hydro_cfl_is_positive_and_finite() {
        let n = 4;
        let n3 = n * n * n;
        let rho = vec![1.0f64; n3];
        let vel = vec![[100.0f64, 0.0, 0.0]; n3];
        let u = vec![1.5f64; n3];
        let dt = hydro_cfl(&rho, &vel, &u, 1.0, 0.8, GAMMA);
        assert!(dt > 0.0 && dt.is_finite(), "dt={dt}");
    }

    /// Sod shock tube: left (ρ=1, P=1) | right (ρ=0.125, P=0.1), γ=5/3.
    ///
    /// Uses an n×n×n grid with x-only initial conditions (y/z sweeps are
    /// no-ops since there are no transverse gradients).  With periodic BCs
    /// two counter-propagating shocks form: one at x=0.5 moving right, one
    /// at the periodic boundary moving left.  By t=0.2 they have created a
    /// compressed collision zone.
    ///
    /// Checks:
    /// - mass conservation
    /// - no negative or NaN densities
    /// - compression above the right-state value (shock formed)
    /// - peak density below the left-state (no unphysical gain)
    /// - density field has structure (solver is not just diffusing)
    #[test]
    fn sod_shock_tube_density_jump() {
        let n = 32;
        let n3 = n * n * n;
        let dx = 1.0 / n as f64;
        let gamma = GAMMA;

        let mut rho = vec![0.0f64; n3];
        let mut vel = vec![[0.0f64; 3]; n3];
        let mut u = vec![0.0f64; n3];
        let mut mass_init = 0.0f64;

        for i in 0..n {
            let (r, p) = if i < n / 2 { (1.0, 1.0) } else { (0.125, 0.1) };
            let ui = p / ((gamma - 1.0) * r);
            for j in 0..n {
                for k in 0..n {
                    let idx = i * n * n + j * n + k;
                    rho[idx] = r;
                    u[idx] = ui;
                    mass_init += r;
                }
            }
        }

        let grav = vec![[0.0f64; 3]; n3];
        let mut p_rho = vec![0.0f64; n3];
        let mut p_vel = vec![[0.0f64; 3]; n3];
        let mut p_u = vec![0.0f64; n3];
        for _ in 0..200 {
            step(
                &mut rho, &mut vel, &mut u, &grav, None, dx, 0.001, n, gamma, &mut p_rho,
                &mut p_vel, &mut p_u,
            );
        }

        let rho_max = rho.iter().cloned().fold(0.0_f64, f64::max);
        let rho_min = rho.iter().cloned().fold(f64::INFINITY, f64::min);
        let mass_final: f64 = rho.iter().sum();

        // Mass conservation (periodic BC)
        assert!(
            (mass_final - mass_init).abs() / mass_init < 1e-8,
            "mass not conserved: {mass_init:.6} → {mass_final:.6}"
        );
        // No NaN or negative density (stability)
        assert!(
            rho_min > 0.0,
            "negative or zero density: min ρ = {rho_min:.6}"
        );
        // Shock must have compressed gas above the initial right state (0.125)
        assert!(
            rho_max > 0.3,
            "no significant compression: max ρ = {rho_max:.3}"
        );
        // No supraphysical density gain (both shocks compressing against each other
        // can exceed single-shock post-density ≈0.43 but not the left-state 1.0)
        assert!(
            rho_max < 1.05,
            "supraphysical density: max ρ = {rho_max:.3}"
        );
        // Density field must retain structure (not fully diffused to mean ≈0.56)
        assert!(
            rho_max - rho_min > 0.3,
            "density too uniform: max={rho_max:.3}, min={rho_min:.3}"
        );
    }
}
