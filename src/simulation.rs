//! Top-level simulation driver.
//!
//! Implements the ISIS/RAMSES run loop:
//!
//!  1. Drift ½
//!  2. Assign density (CIC)
//!  3. Solve Poisson  →  gravitational force
//!  4. Solve scalar field  →  fifth force
//!  5. Kick (full step)
//!  6. Drift ½
//!  7. Advance scale factor (RK4)
//!  8. Refine AMR
//!  9. Write output if due

use crate::{
    amr::grid::AmrGrid,
    gravity::poisson::{grad_phi_into, solve_poisson_into},
    mpi::MpiContext,
    particles::particles::Particles,
    utils::{config::Config, cosmology::Cosmology, types::Real},
};
use rayon::prelude::*;

/// Pre-allocated scratch buffers reused across simulation steps.
///
/// At 256^3 the step loop allocates ~1 GB per step (rho, source, phi,
/// grav_grid, FFT complex buffer).  Allocating once here and reusing
/// eliminates heap churn that dominated wallclock time.
pub struct SimScratch {
    /// Density field (length n^3)
    pub rho:     Vec<Real>,
    /// Poisson source term (length n^3)
    pub source:  Vec<Real>,
    /// Gravitational potential (length n^3)
    pub phi:     Vec<Real>,
    /// Gravitational acceleration grid (length n^3)
    pub grav:    Vec<[Real; 3]>,
    /// FFT complex work buffer (length n^3, fft_pm only)
    #[cfg(feature = "fft_pm")]
    pub fft_buf: Vec<rustfft::num_complex::Complex<f64>>,
    /// FFT column buffer for non-contiguous axis transforms (length n, fft_pm only)
    #[cfg(feature = "fft_pm")]
    pub fft_col: Vec<rustfft::num_complex::Complex<f64>>,
}

impl SimScratch {
    /// Allocate all scratch buffers for an n^3 grid.
    pub fn new(n: usize) -> Self {
        let n3 = n * n * n;
        Self {
            rho:    vec![0.0_f64; n3],
            source: vec![0.0_f64; n3],
            phi:    vec![0.0_f64; n3],
            grav:   vec![[0.0_f64; 3]; n3],
            #[cfg(feature = "fft_pm")]
            fft_buf: vec![rustfft::num_complex::Complex::new(0.0, 0.0); n3],
            #[cfg(feature = "fft_pm")]
            fft_col: vec![rustfft::num_complex::Complex::new(0.0, 0.0); n],
        }
    }
}

#[cfg(feature = "symmetron")]
use crate::scalar_field::symmetron::{fifth_force_grid, solve_symmetron, SymmetronParams};

#[cfg(feature = "fofr")]
use crate::scalar_field::fofr::{fofr_fifth_force, solve_fofr, FofrParams};

/// The complete simulation state.
pub struct Simulation {
    pub config: Config,
    pub cosmology: Cosmology,
    pub grid: AmrGrid,
    pub particles: Particles,
    /// Domain-grid scalar field χ (symmetron or f_R)
    pub chi: Vec<Real>,
    /// Domain-grid χ̇ (non-static mode only)
    pub chi_dot: Vec<Real>,
    /// Current scale factor
    pub a: Real,
    /// Current supercomoving time step Δτ
    pub dtau: Real,
    /// Simulation step counter
    pub step: u64,
    /// MPI context for distributed runs (single-rank stub when compiled without `mpi_parallel`).
    pub mpi_ctx: MpiContext,
    /// Pre-allocated scratch buffers reused across steps to avoid heap churn.
    pub scratch: SimScratch,
}

impl Simulation {
    /// Construct and initialise from a `Config`.
    ///
    /// `mpi_ctx` is the MPI context, created once in `main()` and passed in.
    /// In non-MPI builds, use `MpiContext::init()` which returns a single-rank stub.
    pub fn new(config: Config, mpi_ctx: MpiContext) -> Self {
        let c = &config.cosmology;
        let cosmology = Cosmology {
            omega_m: c.omega_m,
            omega_lambda: c.omega_lambda,
            omega_b: c.omega_b,
            h: c.h,
            n_s: c.n_s,
            sigma8: c.sigma8,
        };

        let n = config.run.n_grid as usize;
        let box_size = config.run.box_size;
        let grid = AmrGrid::new_uniform(config.run.n_grid, config.run.levelmax, box_size);

        // Particle mass: M_part = ρ̄ * L³ / N_part
        let n_part = config.particles.n_part as usize;
        let n3_part = n_part * n_part * n_part;
        let m_part = if config.particles.m_part > 0.0 {
            config.particles.m_part
        } else {
            cosmology.rho_bar_matter() * box_size.powi(3) / n3_part as Real
        };

        let particles = Particles::new(n3_part, m_part, box_size);

        let n3_grid = n * n * n;
        let scratch = SimScratch::new(n);
        Self {
            a: config.run.a_init,
            dtau: config.run.dtau_max * 0.01,
            step: 0,
            chi: vec![0.0_f64; n3_grid],
            chi_dot: vec![0.0_f64; n3_grid],
            config,
            cosmology,
            grid,
            particles,
            mpi_ctx,
            scratch,
        }
    }

    /// Generate Zel'dovich approximation initial conditions in-place.
    ///
    /// Calls [`crate::ic::zeldovich::generate_zeldovich`] and writes the result
    /// directly into `self.particles.pos` and `self.particles.vel`.
    ///
    /// `seed` controls the random realisation; the same seed always produces
    /// the same field.  Use `seed = 42` for reproducible runs.
    pub fn generate_ic(&mut self, seed: u64) {
        use crate::ic::zeldovich::generate_zeldovich;
        let n_p = self.config.particles.n_part as usize;
        let box_size = self.config.run.box_size;
        let (pos, vel) = generate_zeldovich(n_p, box_size, self.a, &self.cosmology, seed);
        assert_eq!(
            pos.len(),
            self.particles.n,
            "IC generator produced {} positions but particles.n = {}",
            pos.len(),
            self.particles.n
        );
        self.particles.pos = pos;
        self.particles.vel = vel;
        log::info!(
            "Generated Zel'dovich IC: {} particles, seed={}",
            self.particles.n,
            seed
        );
    }

    /// Load particle initial conditions from a raw binary dump.
    ///
    /// Format: N_part × 6 × f64 LE, each particle as [x,y,z,vx,vy,vz].
    pub fn load_ic(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let np = self.particles.n;
        let nbytes = np * 6 * 8;
        let mut buf = vec![0u8; nbytes];
        f.read_exact(&mut buf)?;
        for i in 0..np {
            let b = i * 6 * 8;
            let r = |off: usize| f64::from_le_bytes(buf[b + off..b + off + 8].try_into().unwrap());
            self.particles.pos[i] = [r(0), r(8), r(16)];
            self.particles.vel[i] = [r(24), r(32), r(40)];
        }
        log::info!("Loaded {} particles from '{path}'", np);
        Ok(())
    }

    // ── Main run loop ─────────────────────────────────────────────────────────

    /// Run from `a_init` to `a_final`.
    pub fn run(&mut self) {
        let a_final = self.config.run.a_final;
        let n = self.config.run.n_grid as usize;
        let box_size = self.config.run.box_size;
        let dx = box_size / n as Real;

        let out_times = self.output_schedule();
        // When restarting from a snapshot, skip outputs already written.
        let mut next_out = out_times.partition_point(|&t| t <= self.a);

        log::info!("Starting simulation: a = {:.4} → {:.4}", self.a, a_final);
        if next_out > 0 {
            log::info!(
                "Restart: skipping {} output(s) already past a={:.4}",
                next_out,
                self.a
            );
        }

        while self.a < a_final {
            self.dtau = self.compute_dtau();

            // ── ½ Drift ────────────────────────────────────────────────────
            self.particles.drift(self.dtau * 0.5, self.a);

            // ── Particle migration (MPI) ──────────────────────────────────
            // After drift, particles may have crossed slab boundaries.
            #[cfg(feature = "mpi_parallel")]
            self.mpi_ctx.migrate_particles(
                &mut self.particles.pos,
                &mut self.particles.vel,
                n,
                box_size,
            );

            // Scale factor at the half-step midpoint, used for force evaluation
            // and the second drift.  Using a_start here instead of a_half causes
            // ~7× excess power growth when Δa/a is large (e.g. the first step).
            let a_half = self.cosmology.advance_a(self.a, self.dtau * 0.5);

            // ── Density assignment (into scratch.rho) ──────────────────────
            self.particles.cic_density_into(&mut self.scratch.rho, n);

            // In distributed mode, particles near slab edges deposit into
            // ghost cells that must be summed across ranks.
            #[cfg(feature = "mpi_parallel")]
            {
                let reduced = self.mpi_ctx.allreduce_sum_vec(&self.scratch.rho);
                self.scratch.rho.copy_from_slice(&reduced);
            }

            self.grid.set_domain_rho(&self.scratch.rho);

            // ── Poisson solver (into scratch.phi via scratch.fft_buf/col) ─
            let g_cfg = &self.config.gravity;
            // Source:  (3/2) H₀² Ω_m / a_half × δ  (evaluated at half-step)
            // advance_a uses cosmic time τ = t with da/dτ = a·H_phys(a), where
            // H_phys = h·100·E(a) km/s/Mpc.  The consistent H₀ for ∇²Φ is
            // H₀_phys = h·100 km/s/Mpc, NOT the h-unit constant 100 km/s/(Mpc/h).
            // Using 100 instead of h·100 inflates the force by 1/h² ≈ 2 and
            // causes ~3.7× excess in δ by z=2.
            let h0: Real = self.cosmology.h * 100.0; // H₀ physical [km/s/Mpc]
            let prefac: Real = 1.5 * h0 * h0 * self.cosmology.omega_m / a_half;
            for (s, r) in self.scratch.source.iter_mut().zip(self.scratch.rho.iter()) {
                *s = prefac * (r - 1.0);
            }
            solve_poisson_into(
                &self.scratch.source,
                dx,
                n,
                g_cfg.epsilon,
                g_cfg.n_mg_iter,
                g_cfg.n_smooth as usize,
                &self.mpi_ctx,
                &mut self.scratch.phi,
                #[cfg(feature = "fft_pm")]
                &mut self.scratch.fft_buf,
                #[cfg(feature = "fft_pm")]
                &mut self.scratch.fft_col,
            );

            // In distributed mode, the FFT-PM solver receives a local slab,
            // performs the distributed FFT, then allgathers phi back into a
            // full N^3 grid.  The potential is therefore globally consistent
            // and no halo exchange is needed for the gradient stencil
            // (periodic idx3p handles wrapping).
            // TODO: for a truly distributed multigrid solver, exchange_halos
            // on phi would be needed here after each V-cycle.

            grad_phi_into(&mut self.scratch.grav, &self.scratch.phi, dx, n);
            self.grid.set_domain_phi(&self.scratch.phi);
            self.grid.set_domain_grav(&self.scratch.grav);

            // ── Scalar field ───────────────────────────────────────────────
            let fifth_grid: Option<Vec<[Real; 3]>> = self.solve_scalar_field(dx, n);

            // ── Interpolate forces to particles ────────────────────────────
            let grav_part = self.particles.cic_interpolate_force(&self.scratch.grav, n);
            let fifth_part: Option<Vec<[Real; 3]>> = fifth_grid
                .as_ref()
                .map(|fg| self.particles.cic_interpolate_force(fg, n));

            // ── Kick ──────────────────────────────────────────────────────
            self.particles
                .kick(&grav_part, fifth_part.as_deref(), self.dtau, self.a);

            // ── ½ Drift ────────────────────────────────────────────────────
            // Use a_half: x advances from τ+Δτ/2 to τ+Δτ, so the factor 1/a²
            // should be evaluated at τ+Δτ/2, not the step start.
            self.particles.drift(self.dtau * 0.5, a_half);

            // ── Particle migration after second drift (MPI) ───────────────
            #[cfg(feature = "mpi_parallel")]
            self.mpi_ctx.migrate_particles(
                &mut self.particles.pos,
                &mut self.particles.vel,
                n,
                box_size,
            );

            // ── Advance a ─────────────────────────────────────────────────
            self.a = self.cosmology.advance_a(self.a, self.dtau);
            self.step += 1;

            if self.mpi_ctx.is_root() {
                log::info!(
                    "Step {:6}  a = {:.5}  Δτ = {:.3e}",
                    self.step,
                    self.a,
                    self.dtau
                );
            }

            // ── Output ────────────────────────────────────────────────────
            // In MPI mode, only rank 0 writes snapshots.  A full gather of
            // particles would be needed for the power spectrum; for now,
            // every rank writes independently (to be improved with gather).
            if next_out < out_times.len() && self.a >= out_times[next_out] {
                if self.mpi_ctx.is_root() {
                    self.write_snapshot(next_out);
                }
                next_out += 1;
            }
        }

        log::info!("Simulation done. Final a = {:.5}", self.a);
    }

    // ── Scalar field dispatch ─────────────────────────────────────────────────

    #[allow(unused_variables)]
    fn solve_scalar_field(&mut self, dx: Real, n: usize) -> Option<Vec<[Real; 3]>> {
        let sf = &self.config.scalar_field;

        #[cfg(feature = "symmetron")]
        if sf.a_ssb.is_some() {
            let p = SymmetronParams {
                a_ssb: sf.a_ssb.unwrap(),
                lambda_0: sf.lambda_0.unwrap_or(1.0),
                beta_0: sf.beta_0.unwrap_or(1.0),
                non_static: sf.non_static.unwrap_or(false),
                epsilon: sf.epsilon_mg.unwrap_or(1e-5),
                max_iter: 150,
                n_smooth: 2,
            };
            let chi_new = solve_symmetron(&self.scratch.rho, self.a, dx, n, &p);
            self.chi.copy_from_slice(&chi_new);
            let f5_raw = fifth_force_grid(&self.chi, dx, n, self.a, &p);
            self.grid.set_domain_scalar(&self.chi);
            // fifth_force_grid returns (β₀/a)∇χ with χ dimensionless (O(1)) and
            // ∇ in (Mpc/h)⁻¹.  Unlike f(R), the symmetron source has no (H₀/c)²
            // prefactor, so χ ~ O(1) — multiplying by c² would give F₅ ~ 10⁷ F_N.
            //
            // Unit derivation: the natural velocity scale for the symmetron is
            //   v₀ = H₀_h × λ₀   [km/s],   H₀_h = 100 km/s/(Mpc/h)
            // so the factor v₀² converts (Mpc/h)⁻¹ → (km/s)²/(Mpc/h), matching
            // the Poisson solver.  This gives F₅/F_N ~ 2β₀² at full coupling.
            let v0 = 100.0 * p.lambda_0;   // H₀_h × λ₀  [km/s]
            let factor = v0 * v0;
            let f5: Vec<[Real; 3]> = f5_raw.iter()
                .map(|v| [v[0] * factor, v[1] * factor, v[2] * factor])
                .collect();
            return Some(f5);
        }

        #[cfg(feature = "fofr")]
        if sf.fofr0.is_some() {
            let p = FofrParams {
                fofr0: sf.fofr0.unwrap(),
                n: sf.n_fofr.unwrap_or(1.0),
                epsilon: sf.epsilon_mg.unwrap_or(1e-10),
                max_iter: 150,
                n_smooth: 2,
            };
            let chi_new = solve_fofr(&self.scratch.rho, self.a, dx, n, &p, &self.cosmology);
            self.chi.copy_from_slice(&chi_new);
            let f5_raw = fofr_fifth_force(&self.chi, dx, n);
            self.grid.set_domain_scalar(&self.chi);
            // fofr_fifth_force returns (1/2)∇f_R with f_R dimensionless and ∇ in
            // (Mpc/h)⁻¹.  Physical fifth force: a₅ = (c²/2)∇f_R, so multiply by
            // c² = (299 792 km/s)² to match Poisson solver units (km/s)²/(Mpc/h).
            const C_CODE: Real = 299_792.458;
            let c2 = C_CODE * C_CODE;
            let f5: Vec<[Real; 3]> = f5_raw.iter()
                .map(|v| [v[0] * c2, v[1] * c2, v[2] * c2])
                .collect();
            return Some(f5);
        }

        None
    }

    // ── Time stepping ─────────────────────────────────────────────────────────

    fn compute_dtau(&self) -> Real {
        // CFL: Δτ < courant * dx / v_max  (parallel reduction over particles)
        let v_max_local: Real = self
            .particles
            .vel
            .par_iter()
            .flat_map(|v| v.par_iter().copied())
            .map(f64::abs)
            .reduce(|| 0.0_f64, f64::max)
            .max(1.0e-10);

        // In MPI mode, take the global maximum across all ranks so every
        // rank uses the same timestep (required for synchronous leapfrog).
        let v_max = self.mpi_ctx.allreduce_max(v_max_local);

        let n = self.config.run.n_grid as Real;
        let dx = self.config.run.box_size / n;
        let cfl = self.config.run.courant_factor * dx / v_max;

        // Scale-factor CFL: limit Δa/a ≤ 0.02 to keep a_half accurate.
        // da/dτ = a H(a), so Δτ_max = 0.02 / H(a).
        let cfl_a = 0.02 / self.cosmology.hubble(self.a);

        cfl.min(self.config.run.dtau_max).min(cfl_a)
    }

    // ── Output schedule ───────────────────────────────────────────────────────

    fn output_schedule(&self) -> Vec<Real> {
        if let Some(ref a_out) = self.config.output.a_out {
            return a_out.clone();
        }
        let n = self.config.output.n_output as usize;
        let a0 = self.config.run.a_init;
        let a1 = self.config.run.a_final;
        (1..=n)
            .map(|i| a0 + (a1 - a0) * i as Real / n as Real)
            .collect()
    }

    // ── Snapshot I/O ──────────────────────────────────────────────────────────

    /// Write a binary snapshot **and** a companion power spectrum text file.
    ///
    /// Snapshot format: [step: u64][a: f64][n_part: u64] then for each particle [x,y,z,vx,vy,vz].
    /// Power spectrum: `power_spectrum_{idx:04}.txt`, space-separated k / P(k) / N_modes.
    pub fn write_snapshot(&self, idx: usize) {
        use std::io::Write;
        let dir = &self.config.output.outdir;
        std::fs::create_dir_all(dir).expect("Cannot create output directory");

        // ── Binary snapshot ──────────────────────────────────────────────
        let snap_path = format!("{dir}/snapshot_{idx:04}.bin");
        let mut f = std::fs::File::create(&snap_path).expect("Cannot create snapshot file");
        f.write_all(&self.step.to_le_bytes()).unwrap();
        f.write_all(&self.a.to_le_bytes()).unwrap();
        let np = self.particles.n as u64;
        f.write_all(&np.to_le_bytes()).unwrap();
        for i in 0..self.particles.n {
            for d in 0..3 {
                f.write_all(&self.particles.pos[i][d].to_le_bytes())
                    .unwrap();
            }
            for d in 0..3 {
                f.write_all(&self.particles.vel[i][d].to_le_bytes())
                    .unwrap();
            }
        }
        log::info!("Snapshot {idx:04} written → {snap_path}  (a={:.5})", self.a);

        // ── Power spectrum ────────────────────────────────────────────────
        use crate::analysis::power_spectrum::compute_power_spectrum;
        let n_grid = self.config.run.n_grid as usize;
        let ps = compute_power_spectrum(
            &self.particles,
            n_grid,
            self.a,
            self.step,
            40,   // 40 log-spaced k bins
            true, // subtract shot-noise
        );
        let pk_path = format!("{dir}/power_spectrum_{idx:04}.txt");
        ps.write(&pk_path);
    }

    /// Read back a snapshot (for restart).
    pub fn read_snapshot(path: &str, config: Config, mpi_ctx: MpiContext) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut buf = [0u8; 8];

        macro_rules! read_u64 {
            () => {{
                f.read_exact(&mut buf)?;
                u64::from_le_bytes(buf)
            }};
        }
        macro_rules! read_f64 {
            () => {{
                f.read_exact(&mut buf)?;
                f64::from_le_bytes(buf)
            }};
        }

        let step = read_u64!();
        let a = read_f64!();
        let np = read_u64!() as usize;

        let mut sim = Simulation::new(config, mpi_ctx);
        sim.step = step;
        sim.a = a;

        for i in 0..np {
            for d in 0..3 {
                sim.particles.pos[i][d] = read_f64!();
            }
            for d in 0..3 {
                sim.particles.vel[i][d] = read_f64!();
            }
        }
        log::info!("Restarted from '{path}': step={step}, a={a:.5}");
        Ok(sim)
    }
}
