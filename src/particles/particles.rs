//! Dark-matter particle storage and DKD leapfrog integration.
//!
//! # Conventions
//!
//! Positions are **comoving** coordinates  x ∈ [0, L)  in Mpc/h.
//!
//! Velocities are stored as the **supercomoving peculiar velocity**
//!   ṽ = a² (dx/dτ)
//! which satisfies  dṽ/dτ = a · F_comoving  and is conserved in the
//! absence of forces (Martel & Shapiro 1998; RAMSES §3).
//!
//! # DKD Leapfrog
//!
//! Given forces evaluated at x(τ_n):
//!
//!   Drift₁:  x(τ_n + Δτ/2) = x(τ_n) + ṽ(τ_n) Δτ / (2 a²)
//!   Kick:    ṽ(τ_n + Δτ)   = ṽ(τ_n) + a F Δτ
//!   Drift₂:  x(τ_n + Δτ)   = x(τ_n + Δτ/2) + ṽ(τ_n + Δτ) Δτ / (2 a²)
//!
//! # Grid convention
//!
//! CIC uses C-order (row-major) indexing throughout — see `crate::utils::types`.

use rayon::prelude::*;
use crate::utils::types::{Real, idx3p};

// ── Particle storage (Structure-of-Arrays) ────────────────────────────────────

/// Dark-matter particle ensemble in **Structure-of-Arrays** layout.
///
/// SoA gives better cache behaviour for SIMD-friendly inner loops:
/// drift and CIC kernels only access one field at a time.
pub struct Particles {
    /// Comoving positions  x ∈ [0, L)  [Mpc/h].  SoA: 3 × N f64.
    pub pos: Vec<[Real; 3]>,
    /// Supercomoving peculiar velocity  ṽ = a² dx/dτ  [km/s (Mpc/h)].
    pub vel: Vec<[Real; 3]>,
    /// Particle mass  [M_sun/h].
    pub mass: Real,
    /// Number of particles.
    pub n: usize,
    /// Comoving box side length  [Mpc/h].
    pub box_size: Real,
}

impl Particles {
    /// Allocate storage for `n` dark-matter particles.
    pub fn new(n: usize, mass: Real, box_size: Real) -> Self {
        Self {
            pos: vec![[0.0; 3]; n],
            vel: vec![[0.0; 3]; n],
            mass,
            n,
            box_size,
        }
    }

    // ── DKD leapfrog ──────────────────────────────────────────────────────────

    /// **Drift** half-step:  x += ṽ Δτ / (2 a²).
    ///
    /// Pass `dtau/2` for the two half-drifts bracketing the kick.
    #[inline]
    pub fn drift(&mut self, dtau_half: Real, a: Real) {
        let inv_a2 = 1.0 / (a * a);
        let l      = self.box_size;
        self.pos.par_iter_mut()
            .zip(self.vel.par_iter())
            .for_each(|(xi, vi)| {
                for d in 0..3 {
                    xi[d] += vi[d] * dtau_half * inv_a2;
                    xi[d] = xi[d].rem_euclid(l);
                }
            });
    }

    /// **Kick** step:  ṽ += (g_grav + g_fifth) Δτ.
    ///
    /// With τ = t (cosmic time) and ṽ = a² dx/dt, the equation of motion is
    /// dṽ/dt = -∇Φ, so the kick is simply  ṽ += g × Δτ  (no extra factor of a).
    ///
    /// `fifth_accel` may be `None` for pure Newtonian runs.
    pub fn kick(
        &mut self,
        grav_accel:  &[[Real; 3]],
        fifth_accel: Option<&[[Real; 3]]>,
        dtau:        Real,
        a:           Real,
    ) {
        let _ = a;  // τ = cosmic time: kick has no extra a factor
        self.vel.par_iter_mut()
            .enumerate()
            .for_each(|(i, vi)| {
                for d in 0..3 {
                    vi[d] += grav_accel[i][d] * dtau;
                }
                if let Some(fa) = fifth_accel {
                    for d in 0..3 {
                        vi[d] += fa[i][d] * dtau;
                    }
                }
            });
    }

    // ── Cloud-In-Cell (CIC) ───────────────────────────────────────────────────

    /// Deposit particle masses onto an n_grid³ grid using **Cloud-In-Cell**.
    ///
    /// Returns the overdensity field  δρ/ρ̄ + 1  (i.e. ρ normalised so the
    /// mean equals 1).  Grid layout is C-order (row-major).
    ///
    /// # Performance
    ///
    /// Inner loop is sequential to avoid atomic contention.  For large N,
    /// chunk the particles and accumulate into thread-local grids, then sum.
    pub fn cic_density(&self, ng: usize) -> Vec<Real> {
        let mut rho = vec![0.0_f64; ng * ng * ng];
        self.cic_density_into(&mut rho, ng);
        rho
    }

    /// CIC density assignment into a pre-allocated buffer.
    ///
    /// `rho` must have length `ng³`.  It is zeroed and then filled with the
    /// normalised overdensity field (mean = 1).  Use this variant in the step
    /// loop to avoid per-step heap allocations.
    pub fn cic_density_into(&self, rho: &mut [Real], ng: usize) {
        let dx_inv = ng as Real / self.box_size;
        let n3     = ng * ng * ng;
        assert_eq!(rho.len(), n3);

        // Zero the output buffer
        rho.iter_mut().for_each(|r| *r = 0.0);

        // Thread-local accumulation using rayon fold + reduce
        let local_sum: Vec<Real> = self.pos.par_iter()
            .fold(
                || vec![0.0_f64; n3],
                |mut local, xi| {
                    // Cell-corner index (integer part of scaled position)
                    let u = [
                        xi[0] * dx_inv - 0.5,
                        xi[1] * dx_inv - 0.5,
                        xi[2] * dx_inv - 0.5,
                    ];
                    let i0 = u[0].floor() as isize;
                    let j0 = u[1].floor() as isize;
                    let k0 = u[2].floor() as isize;
                    let tx = u[0] - i0 as Real;
                    let ty = u[1] - j0 as Real;
                    let tz = u[2] - k0 as Real;

                    // Deposit onto 2×2×2 stencil
                    for (di, wx) in [(0_isize, 1.0 - tx), (1, tx)] {
                        for (dj, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                            for (dk, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                                let c = idx3p(i0+di, j0+dj, k0+dk, ng);
                                local[c] += wx * wy * wz;
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0_f64; n3],
                |mut a, b| { a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += y); a },
            );

        // Copy thread-local result into the output buffer
        rho.copy_from_slice(&local_sum);

        // Normalise: divide by mean so that <ρ/ρ̄> = 1
        let scale = n3 as Real / self.n as Real;
        rho.iter_mut().for_each(|r| *r *= scale);
    }

    /// Interpolate a grid-defined vector field **back to particle positions**
    /// using CIC (same stencil as density assignment).
    ///
    /// `grid_vec` is a C-order array of `[fx, fy, fz]` per cell.
    pub fn cic_interpolate_force(
        &self,
        grid_vec: &[[Real; 3]],
        ng:       usize,
    ) -> Vec<[Real; 3]> {
        let dx_inv = ng as Real / self.box_size;
        self.pos.par_iter()
            .map(|xi| {
                let u = [
                    xi[0] * dx_inv - 0.5,
                    xi[1] * dx_inv - 0.5,
                    xi[2] * dx_inv - 0.5,
                ];
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
                            let c  = idx3p(i0+di, j0+dj, k0+dk, ng);
                            let w  = wx * wy * wz;
                            let gv = &grid_vec[c];
                            f[0] += w * gv[0];
                            f[1] += w * gv[1];
                            f[2] += w * gv[2];
                        }
                    }
                }
                f
            })
            .collect()
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Total kinetic energy  ½ Σ m |ṽ|² / a²  [M_sun/h (km/s)²].
    ///
    /// Uses supercomoving velocity: KE = ½ m (ṽ/a)² summed over particles.
    pub fn kinetic_energy(&self, a: Real) -> Real {
        let inv_a2 = 1.0 / (a * a);
        let half_m = 0.5 * self.mass;
        self.vel.par_iter()
            .map(|vi| {
                let v2 = vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2];
                half_m * v2 * inv_a2
            })
            .sum()
    }

    /// RMS velocity  sqrt(<|ṽ|²>) / a  [km/s].
    pub fn rms_velocity(&self, a: Real) -> Real {
        let inv_a2 = 1.0 / (a * a);
        let sum_v2: Real = self.vel.par_iter()
            .map(|vi| vi[0]*vi[0] + vi[1]*vi[1] + vi[2]*vi[2])
            .sum();
        (sum_v2 * inv_a2 / self.n as Real).sqrt()
    }

    /// Centre-of-mass position  [Mpc/h].
    pub fn centre_of_mass(&self) -> [Real; 3] {
        let n_f = self.n as Real;
        let mut com = [0.0_f64; 3];
        for xi in &self.pos {
            for d in 0..3 { com[d] += xi[d]; }
        }
        for d in 0..3 { com[d] /= n_f; }
        com
    }

    /// Enforce periodic boundary conditions  x ∈ [0, L)  (utility).
    pub fn wrap_positions(&mut self) {
        let l = self.box_size;
        self.pos.par_iter_mut().for_each(|xi| {
            for d in 0..3 {
                if xi[d] < 0.0 { xi[d] += l; }
                if xi[d] >= l  { xi[d] -= l; }
            }
        });
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::utils::types::idx3p;

    fn make_grid_particles(ng: usize, l: Real) -> Particles {
        let n = ng * ng * ng;
        let dx = l / ng as Real;
        let mut p = Particles::new(n, 1.0, l);
        let mut idx = 0;
        for i in 0..ng {
            for j in 0..ng {
                for k in 0..ng {
                    p.pos[idx] = [
                        (i as Real + 0.5) * dx,
                        (j as Real + 0.5) * dx,
                        (k as Real + 0.5) * dx,
                    ];
                    idx += 1;
                }
            }
        }
        p
    }

    // ── Drift tests ───────────────────────────────────────────────────────────

    #[test]
    fn drift_zero_velocity_noop() {
        let mut p = Particles::new(4, 1.0, 10.0);
        p.pos[0] = [1.0, 2.0, 3.0];
        p.pos[1] = [9.0, 0.5, 7.0];
        let before = p.pos.clone();
        p.drift(0.1, 1.0);
        for i in 0..4 {
            for d in 0..3 {
                assert!((p.pos[i][d] - before[i][d]).abs() < 1e-15,
                        "particle {i} dim {d} moved with zero velocity");
            }
        }
    }

    #[test]
    fn drift_correct_displacement() {
        // x += v * dt / a²  with a=2, dt=0.1, v=1.0 → Δx = 0.1/4 = 0.025
        let mut p = Particles::new(1, 1.0, 100.0);
        p.pos[0] = [50.0, 50.0, 50.0];
        p.vel[0] = [1.0, 0.0, 0.0];
        let a = 2.0_f64;
        let dt = 0.1_f64;
        p.drift(dt, a);
        let expected = 50.0 + 0.1 / 4.0;
        assert!((p.pos[0][0] - expected).abs() < 1e-14,
                "x = {}, expected {expected}", p.pos[0][0]);
    }

    #[test]
    fn drift_periodic_wrap() {
        let l = 10.0_f64;
        let mut p = Particles::new(1, 1.0, l);
        p.pos[0] = [9.8, 5.0, 5.0];
        p.vel[0] = [100.0, 0.0, 0.0];
        p.drift(0.1, 1.0); // Δx = 100*0.1 = 10 → wrap
        assert!(p.pos[0][0] >= 0.0 && p.pos[0][0] < l,
                "x = {} not in [0, {})", p.pos[0][0], l);
    }

    #[test]
    fn drift_negative_wrap() {
        let l = 10.0_f64;
        let mut p = Particles::new(1, 1.0, l);
        p.pos[0] = [0.1, 5.0, 5.0];
        p.vel[0] = [-100.0, 0.0, 0.0];
        p.drift(0.1, 1.0); // Δx = -10 → wrap to ~0.1
        assert!(p.pos[0][0] >= 0.0 && p.pos[0][0] < l,
                "x = {} not in [0, {})", p.pos[0][0], l);
    }

    // ── Kick tests ────────────────────────────────────────────────────────────

    #[test]
    fn kick_correct_dv() {
        // Δṽ = F * dtau  (no factor of a: dṽ/dt = g)
        let n = 3_usize;
        let mut p = Particles::new(n, 1.0, 10.0);
        let forces: Vec<[Real; 3]> = vec![[2.0, -1.0, 0.5]; n];
        let a = 0.5_f64; let dtau = 0.1_f64;
        p.kick(&forces, None, dtau, a);
        for vi in &p.vel {
            assert!((vi[0] - 2.0*dtau).abs() < 1e-15, "vx mismatch");
            assert!((vi[1] - (-1.0)*dtau).abs() < 1e-15, "vy mismatch");
            assert!((vi[2] - 0.5*dtau).abs() < 1e-15, "vz mismatch");
        }
    }

    #[test]
    fn kick_with_fifth_force_adds() {
        let mut p = Particles::new(1, 1.0, 10.0);
        let grav  = vec![[1.0_f64, 0.0, 0.0]; 1];
        let fifth = vec![[0.5_f64, 0.0, 0.0]; 1];
        let a = 1.0_f64; let dtau = 1.0_f64;
        p.kick(&grav, Some(&fifth), dtau, a);
        assert!((p.vel[0][0] - 1.5).abs() < 1e-14, "vx = {}", p.vel[0][0]);
    }

    // ── CIC density tests ─────────────────────────────────────────────────────

    #[test]
    fn cic_uniform_grid_density_one() {
        // One particle per cell, centred → all cells get rho = 1
        let ng = 4_usize;
        let p  = make_grid_particles(ng, ng as Real);
        let rho = p.cic_density(ng);
        for (c, &r) in rho.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-10,
                    "cell {c}: rho = {r:.6} (expected 1.0)");
        }
    }

    #[test]
    fn cic_mass_conservation() {
        // Sum of rho/rhobar * Ncells should equal Ncells (normalised mean = 1)
        let ng = 8_usize;
        let n  = 200_usize;
        let mut p = Particles::new(n, 1.0, ng as Real);
        for i in 0..n {
            p.pos[i] = [
                (i as Real * 0.37).rem_euclid(ng as Real),
                (i as Real * 0.71).rem_euclid(ng as Real),
                (i as Real * 0.13).rem_euclid(ng as Real),
            ];
        }
        let rho = p.cic_density(ng);
        let sum: Real = rho.iter().sum();
        let expected = (ng * ng * ng) as Real;
        assert!((sum - expected).abs() < 1e-8,
                "CIC mass not conserved: sum={sum:.6}, expected={expected}");
    }

    #[test]
    fn cic_particle_at_cell_centre() {
        // Single particle exactly at cell (1,0,0) centre → only that cell gets ρ>0
        let ng = 4_usize;
        let l  = 4.0_f64;
        let dx = l / ng as Real;
        let mut p = Particles::new(1, 1.0, l);
        p.pos[0] = [1.5 * dx, 0.5 * dx, 0.5 * dx];  // centre of cell (1,0,0)
        let rho = p.cic_density(ng);
        let n3  = ng * ng * ng;
        // cell (1,0,0) → idx3p(1,0,0,ng)
        let target_cell = idx3p(1, 0, 0, ng);
        let target_rho  = rho[target_cell];
        let other_sum: Real = rho.iter().enumerate()
            .filter(|&(c, _)| c != target_cell)
            .map(|(_, &r)| r)
            .sum();
        assert!((target_rho - n3 as Real).abs() < 1e-10,
                "target cell rho = {target_rho:.6}, expected {}", n3 as Real);
        assert!(other_sum < 1e-10, "other cells got rho: {other_sum:.3e}");
    }

    #[test]
    fn cic_interpolate_constant_force() {
        // A constant force field → every particle gets that force
        let ng = 4_usize;
        let n  = 8_usize;
        let cf = [1.23_f64, -0.45, 0.78];
        let grid: Vec<[Real; 3]> = vec![cf; ng * ng * ng];
        let mut p = Particles::new(n, 1.0, 4.0);
        for i in 0..n {
            p.pos[i] = [(i as Real * 0.5 + 0.1).rem_euclid(4.0), 2.0, 2.0];
        }
        let f = p.cic_interpolate_force(&grid, ng);
        for (i, fi) in f.iter().enumerate() {
            for d in 0..3 {
                assert!((fi[d] - cf[d]).abs() < 1e-12,
                        "p{i} d{d}: got {:.6}, expected {:.6}", fi[d], cf[d]);
            }
        }
    }

    #[test]
    fn cic_density_interpolate_adjointness() {
        // CIC density W and interpolation W^T should be adjoint:
        //   Σ_p f(x_p) = Σ_c ρ_c g_c  when f = interp(g)
        // This is a fundamental consistency check.
        let ng = 8_usize;
        let n  = 100_usize;
        let l  = 8.0_f64;
        let mut p = Particles::new(n, 1.0, l);
        for i in 0..n {
            p.pos[i] = [
                (i as Real * 0.37).rem_euclid(l),
                (i as Real * 0.71).rem_euclid(l),
                (i as Real * 0.13).rem_euclid(l),
            ];
        }
        // g = some grid field
        let g: Vec<[Real; 3]> = (0..ng*ng*ng)
            .map(|c| [(c as Real * 0.01).sin(); 3])
            .collect();

        let rho = p.cic_density(ng);
        let f   = p.cic_interpolate_force(&g, ng);

        // Σ_p f(x_p)  (not weighted by mass since rho is normalised)
        let sum_f: Real = f.iter().map(|fi| fi[0] + fi[1] + fi[2]).sum();

        // Σ_c rho_c g_c * (n_part / n_cells)  — unwind the normalisation
        let n3 = (ng * ng * ng) as Real;
        let sum_rg: Real = rho.iter().zip(g.iter())
            .map(|(r, gc)| r * (gc[0] + gc[1] + gc[2]))
            .sum::<Real>()
            * n as Real / n3;

        let rel = (sum_f - sum_rg).abs() / (sum_f.abs() + 1e-30);
        println!("CIC adjointness rel error: {rel:.3e}");
        assert!(rel < 1e-10, "CIC not adjoint: rel error = {rel:.3e}");
    }

    // ── Leapfrog tests ────────────────────────────────────────────────────────

    #[test]
    fn leapfrog_zero_force_conserves_energy() {
        let mut p = Particles::new(3, 1.0, 1000.0);
        p.pos[0] = [500.0, 500.0, 500.0];
        p.vel[0] = [1.0, 2.0, -1.5];
        p.pos[1] = [100.0, 200.0, 300.0];
        p.vel[1] = [-0.5, 0.7, 1.2];
        let zero_f: Vec<[Real; 3]> = vec![[0.0; 3]; 3];
        let a    = 1.0_f64;
        let dtau = 0.001_f64;
        let ke0  = p.kinetic_energy(a);
        for _ in 0..2000 {
            p.drift(dtau * 0.5, a);
            p.kick(&zero_f, None, dtau, a);
            p.drift(dtau * 0.5, a);
        }
        let ke1 = p.kinetic_energy(a);
        let rel = (ke1 - ke0).abs() / ke0;
        assert!(rel < 1e-13, "KE not conserved: Δ={rel:.3e}");
    }

    #[test]
    fn leapfrog_time_reversibility() {
        // Forward N steps, then reverse velocity and run N more steps with
        // the SAME force (force is position-based, doesn't flip sign under
        // time reversal). Should recover initial state exactly.
        let mut p = Particles::new(1, 1.0, 1e8);
        p.pos[0] = [500.1, 499.9, 500.5];
        p.vel[0] = [0.7, -0.3, 0.4];
        let fwd_f: Vec<[Real; 3]> = vec![[0.01, -0.005, 0.0]; 1];
        let a    = 1.0_f64;
        let dtau = 0.01_f64;
        let n    = 100_usize;

        let x0 = p.pos[0];
        let v0 = p.vel[0];

        // Forward N steps
        for _ in 0..n {
            p.drift(dtau * 0.5, a);
            p.kick(&fwd_f, None, dtau, a);
            p.drift(dtau * 0.5, a);
        }
        // Reverse velocity, run N steps with the SAME force
        p.vel[0] = [-p.vel[0][0], -p.vel[0][1], -p.vel[0][2]];
        for _ in 0..n {
            p.drift(dtau * 0.5, a);
            p.kick(&fwd_f, None, dtau, a);  // same force, not negated
            p.drift(dtau * 0.5, a);
        }
        // Flip velocity back to compare
        p.vel[0] = [-p.vel[0][0], -p.vel[0][1], -p.vel[0][2]];

        for d in 0..3 {
            assert!((p.pos[0][d] - x0[d]).abs() < 1e-10,
                    "pos[{d}]: {:.10} ≠ {:.10}", p.pos[0][d], x0[d]);
            assert!((p.vel[0][d] - v0[d]).abs() < 1e-10,
                    "vel[{d}]: {:.10} ≠ {:.10}", p.vel[0][d], v0[d]);
        }
    }

    #[test]
    fn leapfrog_constant_force_trajectory() {
        // Constant force F in x → parabolic trajectory.
        // x(t) = x0 + v0*t + ½ F t²  (in supercomoving time with a=1)
        let mut p   = Particles::new(1, 1.0, 1e6);
        let f_val   = 2.0_f64;
        let v0x     = 1.0_f64;
        let a       = 1.0_f64;
        let dtau    = 1e-4_f64;
        let n_steps = 1000_usize;
        let t_end   = dtau * n_steps as Real;
        p.vel[0][0] = v0x;
        let force: Vec<[Real; 3]> = vec![[f_val, 0.0, 0.0]; 1];

        for _ in 0..n_steps {
            p.drift(dtau * 0.5, a);
            p.kick(&force, None, dtau, a);
            p.drift(dtau * 0.5, a);
        }
        // Exact: x = v0*t + ½ F t²
        let x_exact = v0x * t_end + 0.5 * f_val * t_end * t_end;
        let err = (p.pos[0][0] - x_exact).abs();
        println!("Constant force trajectory error: {err:.3e}");
        assert!(err < 1e-6, "trajectory error {err:.3e}");
    }

    // ── Diagnostics tests ─────────────────────────────────────────────────────

    #[test]
    fn kinetic_energy_analytical() {
        let mut p = Particles::new(2, 2.0, 100.0);
        p.vel[0] = [3.0, 4.0, 0.0];  // |v|² = 25
        p.vel[1] = [0.0, 0.0, 5.0];  // |v|² = 25
        // KE = ½ m Σ |ṽ|²/a² = ½ * 2 * (25 + 25) / 1 = 50
        let ke = p.kinetic_energy(1.0);
        assert!((ke - 50.0).abs() < 1e-12, "KE = {ke:.6}");
    }

    #[test]
    fn centre_of_mass_symmetric() {
        let l = 10.0_f64;
        let mut p = Particles::new(2, 1.0, l);
        p.pos[0] = [3.0, 5.0, 5.0];
        p.pos[1] = [7.0, 5.0, 5.0];
        let com = p.centre_of_mass();
        assert!((com[0] - 5.0).abs() < 1e-14, "com_x = {}", com[0]);
        assert!((com[1] - 5.0).abs() < 1e-14, "com_y = {}", com[1]);
    }
}
