//! Dark-matter particle storage and leapfrog time integration.
//!
//! Velocities are stored as the **supercomoving peculiar velocity**
//!   ṽ = a² (dx/dτ)
//! which is conserved in the absence of forces, following the RAMSES
//! convention (Teyssier 2002, Martel & Shapiro 1998).
//!
//! The DKD leapfrog (Drift-Kick-Drift) in supercomoving time τ is:
//!
//!   Drift:  x += ṽ Δτ / (2 a²)
//!   Kick:   ṽ += a (g_gravity + g_fifth) Δτ
//!   Drift:  x += ṽ Δτ / (2 a²)

use rayon::prelude::*;
use crate::utils::types::Real;

// ── Particle storage (Structure-of-Arrays) ────────────────────────────────────

/// Dark-matter particle ensemble in Structure-of-Arrays layout.
pub struct Particles {
    /// Comoving positions  x ∈ [0, L)  [Mpc/h]
    pub x: Vec<[Real; 3]>,
    /// Supercomoving peculiar velocity  ṽ = a² dx/dτ
    pub v: Vec<[Real; 3]>,
    /// Particle mass  [M_sun/h]
    pub mass: Real,
    /// Number of particles
    pub n: usize,
    /// Comoving box side length  [Mpc/h]
    pub box_size: Real,
}

impl Particles {
    /// Allocate storage for `n` particles.
    pub fn new(n: usize, mass: Real, box_size: Real) -> Self {
        Self {
            x: vec![[0.0; 3]; n],
            v: vec![[0.0; 3]; n],
            mass,
            n,
            box_size,
        }
    }

    // ── DKD leapfrog ──────────────────────────────────────────────────────────

    /// **Drift** step:  x += ṽ * dtau / a²
    ///
    /// Call with `dtau/2` for the half-steps bracketing the kick.
    pub fn drift(&mut self, dtau: Real, a: Real) {
        let inv_a2 = 1.0 / (a * a);
        let l = self.box_size;
        self.x.par_iter_mut()
            .zip(self.v.par_iter())
            .for_each(|(xi, vi)| {
                for d in 0..3 {
                    xi[d] += vi[d] * dtau * inv_a2;
                    // Periodic wrap inline
                    if xi[d] < 0.0 { xi[d] += l; }
                    if xi[d] >= l  { xi[d] -= l; }
                }
            });
    }

    /// **Kick** step:  ṽ += a * (g_grav + g_fifth) * dtau
    ///
    /// `grav_accel` and `fifth_accel` are grid-interpolated accelerations
    /// per particle.  `fifth_accel` is None for standard gravity runs.
    pub fn kick(
        &mut self,
        grav_accel:  &[[Real; 3]],
        fifth_accel: Option<&[[Real; 3]]>,
        dtau:        Real,
        a:           Real,
    ) {
        // In supercomoving coordinates:  dṽ/dτ = a * F_comoving
        let a_dtau = a * dtau;
        self.v.par_iter_mut()
            .enumerate()
            .for_each(|(i, vi)| {
                for d in 0..3 {
                    vi[d] += grav_accel[i][d] * a_dtau;
                    if let Some(fa) = fifth_accel {
                        vi[d] += fa[i][d] * a_dtau;
                    }
                }
            });
    }

    /// Enforce periodic boundary conditions  x ∈ [0, L).
    pub fn wrap_positions(&mut self) {
        let l = self.box_size;
        self.x.par_iter_mut().for_each(|xi| {
            for d in 0..3 {
                if xi[d] < 0.0 { xi[d] += l; }
                if xi[d] >= l  { xi[d] -= l; }
            }
        });
    }

    // ── Cloud-In-Cell density assignment ─────────────────────────────────────

    /// Assign particle masses to an n_grid³ uniform grid using CIC.
    ///
    /// Returns a flat array of ρ/ρ̄ values (mean = 1.0) in Fortran-order
    /// (x varies fastest: index = k*n² + j*n + i).
    pub fn cic_density(&self, n_grid: usize) -> Vec<Real> {
        let ng = n_grid;
        // Serial CIC — collisions handled by single-threaded accumulation.
        // For very large N consider atomic adds or thread-local buffers.
        let mut rho = vec![0.0_f64; ng * ng * ng];
        let h_inv = ng as Real / self.box_size;

        for xi in &self.x {
            // Particle centre in fractional grid coordinates (cell-centred)
            let cx = xi[0] * h_inv - 0.5;
            let cy = xi[1] * h_inv - 0.5;
            let cz = xi[2] * h_inv - 0.5;

            let ix = cx.floor() as isize;
            let iy = cy.floor() as isize;
            let iz = cz.floor() as isize;

            let tx = cx - ix as Real; // 0 ≤ t < 1
            let ty = cy - iy as Real;
            let tz = cz - iz as Real;

            let ng_i = ng as isize;
            for (dz, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                for (dy, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                    for (dx, wx) in [(0_isize, 1.0 - tx), (1, tx)] {
                        let jx = (ix + dx).rem_euclid(ng_i) as usize;
                        let jy = (iy + dy).rem_euclid(ng_i) as usize;
                        let jz = (iz + dz).rem_euclid(ng_i) as usize;
                        rho[jz * ng * ng + jy * ng + jx] += wx * wy * wz;
                    }
                }
            }
        }

        // Normalise to ρ/ρ̄: each particle contributes 1, mean over n³ cells
        let norm = ng as Real * ng as Real * ng as Real / self.n as Real;
        rho.iter_mut().for_each(|r| *r *= norm);
        rho
    }

    // ── CIC force interpolation ───────────────────────────────────────────────

    /// Interpolate grid accelerations to particle positions using CIC.
    pub fn cic_interpolate_force(
        &self,
        grid_force: &[[Real; 3]],
        n_grid:     usize,
    ) -> Vec<[Real; 3]> {
        let ng = n_grid as isize;
        let h_inv = n_grid as Real / self.box_size;
        let mut forces = vec![[0.0_f64; 3]; self.n];

        forces.par_iter_mut().enumerate().for_each(|(p, fp)| {
            let xi = &self.x[p];
            let cx = xi[0] * h_inv - 0.5;
            let cy = xi[1] * h_inv - 0.5;
            let cz = xi[2] * h_inv - 0.5;

            let ix = cx.floor() as isize;
            let iy = cy.floor() as isize;
            let iz = cz.floor() as isize;

            let tx = cx - ix as Real;
            let ty = cy - iy as Real;
            let tz = cz - iz as Real;

            for (dz, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                for (dy, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                    for (dx, wx) in [(0_isize, 1.0 - tx), (1, tx)] {
                        let jx = (ix + dx).rem_euclid(ng) as usize;
                        let jy = (iy + dy).rem_euclid(ng) as usize;
                        let jz = (iz + dz).rem_euclid(ng) as usize;
                        let ci = jz * n_grid * n_grid + jy * n_grid + jx;
                        let w = wx * wy * wz;
                        for d in 0..3 {
                            fp[d] += w * grid_force[ci][d];
                        }
                    }
                }
            }
        });
        forces
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Drift with zero velocity should not move particles.
    #[test]
    fn drift_zero_velocity() {
        let mut p = Particles::new(4, 1.0, 10.0);
        p.x[0] = [1.0, 2.0, 3.0];
        p.x[1] = [9.9, 0.1, 5.0];
        let x_before = p.x.clone();
        p.drift(0.1, 1.0);
        for i in 0..4 {
            for d in 0..3 {
                assert!((p.x[i][d] - x_before[i][d]).abs() < 1e-15);
            }
        }
    }

    /// A particle drifting past the box boundary should wrap.
    #[test]
    fn drift_periodic_wrap() {
        let mut p = Particles::new(1, 1.0, 10.0);
        p.x[0] = [9.8, 5.0, 5.0];
        p.v[0] = [100.0, 0.0, 0.0]; // fast rightward
        p.drift(0.1, 1.0); // dx = 100 * 0.1 / 1 = 10 → wraps to ~9.8
        assert!(p.x[0][0] >= 0.0 && p.x[0][0] < 10.0,
                "x = {} not in [0,10)", p.x[0][0]);
    }

    /// CIC on a uniform particle grid should give density = 1 everywhere.
    #[test]
    fn cic_uniform_density() {
        let ng = 4_usize;
        let l  = 4.0_f64;
        let n  = ng * ng * ng;
        let mut p = Particles::new(n, 1.0, l);
        let dx = l / ng as Real;
        // Place one particle at the centre of each cell
        let mut idx = 0;
        for k in 0..ng {
            for j in 0..ng {
                for i in 0..ng {
                    p.x[idx] = [
                        (i as Real + 0.5) * dx,
                        (j as Real + 0.5) * dx,
                        (k as Real + 0.5) * dx,
                    ];
                    idx += 1;
                }
            }
        }
        let rho = p.cic_density(ng);
        for (c, &r) in rho.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-12, "cell {c}: rho = {r}");
        }
    }

    /// CIC density should sum to N (before normalisation): total mass conserved.
    #[test]
    fn cic_mass_conservation() {
        let ng = 8_usize;
        let n  = 100_usize;
        let mut p = Particles::new(n, 1.0, 8.0);
        // Random-ish positions
        for i in 0..n {
            p.x[i] = [
                (i as Real * 0.37).rem_euclid(8.0),
                (i as Real * 0.71).rem_euclid(8.0),
                (i as Real * 0.13).rem_euclid(8.0),
            ];
        }
        let rho = p.cic_density(ng);
        // Sum of ρ/ρ̄ * n_cells = N_cells  (mean is 1)
        let sum: Real = rho.iter().sum();
        let expected = (ng * ng * ng) as Real;
        assert!((sum - expected).abs() < 1e-8,
                "CIC mass not conserved: sum={sum}, expected={expected}");
    }

    /// Force interpolation from a constant-force grid gives constant force.
    #[test]
    fn cic_interpolate_constant_force() {
        let ng = 4_usize;
        let n  = 8_usize;
        let constant_g = [1.23_f64, -0.45, 0.78];
        let grid_force: Vec<[Real; 3]> = vec![constant_g; ng * ng * ng];
        let mut p = Particles::new(n, 1.0, 4.0);
        for i in 0..n {
            p.x[i] = [(i as Real * 0.5 + 0.1).rem_euclid(4.0), 2.0, 2.0];
        }
        let f = p.cic_interpolate_force(&grid_force, ng);
        for (i, fi) in f.iter().enumerate() {
            for d in 0..3 {
                assert!((fi[d] - constant_g[d]).abs() < 1e-12,
                        "particle {i} dim {d}: got {}, expected {}", fi[d], constant_g[d]);
            }
        }
    }

    /// Kick should change velocity proportional to force × dt.
    #[test]
    fn kick_linear() {
        let n = 4_usize;
        let mut p = Particles::new(n, 1.0, 10.0);
        let a = 0.5_f64;
        let dtau = 0.01_f64;
        let force = [1.0_f64, 0.0, 0.0];
        let forces: Vec<[Real; 3]> = vec![force; n];
        p.kick(&forces, None, dtau, a);
        for vi in &p.v {
            // Δv = a * F * dtau = 0.5 * 1.0 * 0.01 = 0.005
            assert!((vi[0] - a * dtau).abs() < 1e-15, "vx = {}", vi[0]);
            assert!(vi[1].abs() < 1e-15);
            assert!(vi[2].abs() < 1e-15);
        }
    }

    /// Energy (KE) should be conserved by drift+kick+drift with zero force.
    #[test]
    fn leapfrog_energy_conservation_free_particle() {
        let mut p = Particles::new(1, 1.0, 1000.0);
        p.x[0] = [500.0, 500.0, 500.0];
        p.v[0] = [1.0, 2.0, -1.5];
        let a    = 1.0_f64;
        let dtau = 0.001_f64;
        let zero_f: Vec<[Real; 3]> = vec![[0.0; 3]; 1];

        let ke = |v: &[Real; 3]| v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        let ke0 = ke(&p.v[0]);

        for _ in 0..1000 {
            p.drift(dtau * 0.5, a);
            p.kick(&zero_f, None, dtau, a);
            p.drift(dtau * 0.5, a);
        }
        let ke1 = ke(&p.v[0]);
        assert!((ke1 - ke0).abs() / ke0 < 1e-14, "KE changed: {ke0} → {ke1}");
    }
}
