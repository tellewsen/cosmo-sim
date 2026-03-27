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

use rayon::prelude::*;
use crate::{
    amr::grid::AmrGrid,
    gravity::poisson::{solve_poisson, grad_phi},
    particles::particles::Particles,
    utils::{config::Config, cosmology::Cosmology, types::Real},
};

#[cfg(feature = "symmetron")]
use crate::scalar_field::symmetron::{SymmetronParams, solve_symmetron, fifth_force_grid};

#[cfg(feature = "fofr")]
use crate::scalar_field::fofr::{FofrParams, solve_fofr, fofr_fifth_force};

/// The complete simulation state.
pub struct Simulation {
    pub config:    Config,
    pub cosmology: Cosmology,
    pub grid:      AmrGrid,
    pub particles: Particles,
    /// Domain-grid scalar field χ (symmetron or f_R)
    pub chi:       Vec<Real>,
    /// Domain-grid χ̇ (non-static mode only)
    pub chi_dot:   Vec<Real>,
    /// Current scale factor
    pub a:         Real,
    /// Current supercomoving time step Δτ
    pub dtau:      Real,
    /// Simulation step counter
    pub step:      u64,
}

impl Simulation {
    /// Construct and initialise from a `Config`.
    pub fn new(config: Config) -> Self {
        let c = &config.cosmology;
        let cosmology = Cosmology {
            omega_m:      c.omega_m,
            omega_lambda: c.omega_lambda,
            omega_b:      c.omega_b,
            h:            c.h,
            n_s:          c.n_s,
            sigma8:       c.sigma8,
        };

        let n        = config.run.n_grid as usize;
        let box_size = config.run.box_size;
        let grid     = AmrGrid::new_uniform(config.run.n_grid, config.run.levelmax, box_size);

        // Particle mass: M_part = ρ̄ * L³ / N_part
        let n_part  = config.particles.n_part as usize;
        let n3_part = n_part * n_part * n_part;
        let m_part  = if config.particles.m_part > 0.0 {
            config.particles.m_part
        } else {
            cosmology.rho_bar_matter() * box_size.powi(3) / n3_part as Real
        };

        let particles = Particles::new(n3_part, m_part, box_size);

        let n3_grid = n * n * n;
        Self {
            a:      config.run.a_init,
            dtau:   config.run.dtau_max * 0.01,
            step:   0,
            chi:     vec![0.0_f64; n3_grid],
            chi_dot: vec![0.0_f64; n3_grid],
            config,
            cosmology,
            grid,
            particles,
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
        let n_p      = self.config.particles.n_part as usize;
        let box_size = self.config.run.box_size;
        let (pos, vel) = generate_zeldovich(n_p, box_size, self.a, &self.cosmology, seed);
        assert_eq!(pos.len(), self.particles.n,
            "IC generator produced {} positions but particles.n = {}",
            pos.len(), self.particles.n);
        self.particles.pos = pos;
        self.particles.vel = vel;
        log::info!("Generated Zel'dovich IC: {} particles, seed={}", self.particles.n, seed);
    }

    /// Load particle initial conditions from a raw binary dump.
    ///
    /// Format: N_part × 6 × f64 LE, each particle as [x,y,z,vx,vy,vz].
    pub fn load_ic(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;
        let mut f    = std::fs::File::open(path)?;
        let np       = self.particles.n;
        let nbytes   = np * 6 * 8;
        let mut buf  = vec![0u8; nbytes];
        f.read_exact(&mut buf)?;
        for i in 0..np {
            let b = i * 6 * 8;
            let r = |off: usize| f64::from_le_bytes(buf[b+off..b+off+8].try_into().unwrap());
            self.particles.pos[i] = [r(0),  r(8),  r(16)];
            self.particles.vel[i] = [r(24), r(32), r(40)];
        }
        log::info!("Loaded {} particles from '{path}'", np);
        Ok(())
    }

    // ── Main run loop ─────────────────────────────────────────────────────────

    /// Run from `a_init` to `a_final`.
    pub fn run(&mut self) {
        let a_final  = self.config.run.a_final;
        let n        = self.config.run.n_grid as usize;
        let box_size = self.config.run.box_size;
        let dx       = box_size / n as Real;

        let out_times = self.output_schedule();
        // When restarting from a snapshot, skip outputs already written.
        let mut next_out = out_times.partition_point(|&t| t <= self.a);

        log::info!("Starting simulation: a = {:.4} → {:.4}", self.a, a_final);
        if next_out > 0 {
            log::info!("Restart: skipping {} output(s) already past a={:.4}", next_out, self.a);
        }

        while self.a < a_final {
            self.dtau = self.compute_dtau();

            // ── ½ Drift ────────────────────────────────────────────────────
            self.particles.drift(self.dtau * 0.5, self.a);

            // Scale factor at the half-step midpoint, used for force evaluation
            // and the second drift.  Using a_start here instead of a_half causes
            // ~7× excess power growth when Δa/a is large (e.g. the first step).
            let a_half = self.cosmology.advance_a(self.a, self.dtau * 0.5);

            // ── Density assignment ─────────────────────────────────────────
            let rho = self.particles.cic_density(n);
            self.grid.set_domain_rho(&rho);

            // ── Poisson solver ────────────────────────────────────────────
            let g_cfg    = &self.config.gravity;
            // Source:  (3/2) H₀² Ω_m / a_half × δ  (evaluated at half-step)
            // With τ = cosmic time t, positions in Mpc/h, velocities in km/s:
            //   ∇²Φ = (3/2) (100 km/s/(Mpc/h))² Ω_m / a × δ
            // H₀ = 100 km/s/(Mpc/h) → H₀² = 10 000 (km/s)²/(Mpc/h)²
            const H0_CODE: Real = 100.0;   // km/s / (Mpc/h)
            let prefac: Real = 1.5 * H0_CODE * H0_CODE
                               * self.cosmology.omega_m / a_half;
            let source: Vec<Real> = rho.iter().map(|r| prefac * (r - 1.0)).collect();
            let phi = solve_poisson(
                &source, dx, n,
                g_cfg.epsilon, g_cfg.n_mg_iter, g_cfg.n_smooth as usize,
            );
            let grav_grid = grad_phi(&phi, dx, n);
            self.grid.set_domain_phi(&phi);
            self.grid.set_domain_grav(&grav_grid);

            // ── Scalar field ───────────────────────────────────────────────
            let fifth_grid: Option<Vec<[Real; 3]>> =
                self.solve_scalar_field(&rho, dx, n);

            // ── Interpolate forces to particles ────────────────────────────
            let grav_part  = self.particles.cic_interpolate_force(&grav_grid, n);
            let fifth_part: Option<Vec<[Real; 3]>> = fifth_grid.as_ref().map(|fg| {
                self.particles.cic_interpolate_force(fg, n)
            });

            // ── Kick ──────────────────────────────────────────────────────
            self.particles.kick(
                &grav_part,
                fifth_part.as_deref(),
                self.dtau,
                self.a,
            );

            // ── ½ Drift ────────────────────────────────────────────────────
            // Use a_half: x advances from τ+Δτ/2 to τ+Δτ, so the factor 1/a²
            // should be evaluated at τ+Δτ/2, not the step start.
            self.particles.drift(self.dtau * 0.5, a_half);

            // ── Advance a ─────────────────────────────────────────────────
            self.a = self.cosmology.advance_a(self.a, self.dtau);
            self.step += 1;

            log::info!("Step {:6}  a = {:.5}  Δτ = {:.3e}", self.step, self.a, self.dtau);

            // ── Output ────────────────────────────────────────────────────
            if next_out < out_times.len() && self.a >= out_times[next_out] {
                self.write_snapshot(next_out);
                next_out += 1;
            }
        }

        log::info!("Simulation done. Final a = {:.5}", self.a);
    }

    // ── Scalar field dispatch ─────────────────────────────────────────────────

    #[allow(unused_variables)]
    fn solve_scalar_field(
        &mut self,
        rho: &[Real],
        dx:  Real,
        n:   usize,
    ) -> Option<Vec<[Real; 3]>> {
        let sf = &self.config.scalar_field;

        #[cfg(feature = "symmetron")]
        if sf.a_ssb.is_some() {
            let p = SymmetronParams {
                a_ssb:      sf.a_ssb.unwrap(),
                lambda_0:   sf.lambda_0.unwrap_or(1.0),
                beta_0:     sf.beta_0.unwrap_or(1.0),
                non_static: sf.non_static.unwrap_or(false),
                epsilon:    sf.epsilon_mg.unwrap_or(1e-5),
                max_iter:   150,
                n_smooth:   2,
            };
            let chi_new = solve_symmetron(rho, self.a, dx, n, &p);
            self.chi.copy_from_slice(&chi_new);
            let f5 = fifth_force_grid(&self.chi, dx, n, self.a, &p);
            self.grid.set_domain_scalar(&self.chi);
            return Some(f5);
        }

        #[cfg(feature = "fofr")]
        if sf.fofr0.is_some() {
            let p = FofrParams {
                fofr0:    sf.fofr0.unwrap(),
                n:        sf.n_fofr.unwrap_or(1.0),
                epsilon:  sf.epsilon_mg.unwrap_or(1e-5),
                max_iter: 150,
                n_smooth: 2,
            };
            let chi_new = solve_fofr(rho, self.a, dx, n, &p, &self.cosmology);
            self.chi.copy_from_slice(&chi_new);
            let f5 = fofr_fifth_force(&self.chi, dx, n);
            self.grid.set_domain_scalar(&self.chi);
            return Some(f5);
        }

        None
    }

    // ── Time stepping ─────────────────────────────────────────────────────────

    fn compute_dtau(&self) -> Real {
        // CFL: Δτ < courant * dx / v_max  (parallel reduction over particles)
        let v_max: Real = self.particles.vel.par_iter()
            .flat_map(|v| v.par_iter().copied())
            .map(f64::abs)
            .reduce(|| 0.0_f64, f64::max)
            .max(1.0e-10);
        let n   = self.config.run.n_grid as Real;
        let dx  = self.config.run.box_size / n;
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
        let n  = self.config.output.n_output as usize;
        let a0 = self.config.run.a_init;
        let a1 = self.config.run.a_final;
        (1..=n).map(|i| a0 + (a1 - a0) * i as Real / n as Real).collect()
    }

    // ── Snapshot I/O ──────────────────────────────────────────────────────────

    /// Write a binary snapshot **and** a companion power spectrum text file.
    ///
    /// Snapshot format: [step: u64][a: f64][n_part: u64] then for each particle [x,y,z,vx,vy,vz].
    /// Power spectrum: `power_spectrum_{idx:04}.txt`, space-separated k / P(k) / N_modes.
    pub fn write_snapshot(&self, idx: usize) {
        use std::io::Write;
        let dir  = &self.config.output.outdir;
        std::fs::create_dir_all(dir).expect("Cannot create output directory");

        // ── Binary snapshot ──────────────────────────────────────────────
        let snap_path = format!("{dir}/snapshot_{idx:04}.bin");
        let mut f = std::fs::File::create(&snap_path).expect("Cannot create snapshot file");
        f.write_all(&self.step.to_le_bytes()).unwrap();
        f.write_all(&self.a.to_le_bytes()).unwrap();
        let np = self.particles.n as u64;
        f.write_all(&np.to_le_bytes()).unwrap();
        for i in 0..self.particles.n {
            for d in 0..3 { f.write_all(&self.particles.pos[i][d].to_le_bytes()).unwrap(); }
            for d in 0..3 { f.write_all(&self.particles.vel[i][d].to_le_bytes()).unwrap(); }
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
            40,    // 40 log-spaced k bins
            true,  // subtract shot-noise
        );
        let pk_path = format!("{dir}/power_spectrum_{idx:04}.txt");
        ps.write(&pk_path);
    }

    /// Read back a snapshot (for restart).
    pub fn read_snapshot(path: &str, config: Config) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f   = std::fs::File::open(path)?;
        let mut buf = [0u8; 8];

        macro_rules! read_u64 { () => {{ f.read_exact(&mut buf)?; u64::from_le_bytes(buf) }} }
        macro_rules! read_f64 { () => {{ f.read_exact(&mut buf)?; f64::from_le_bytes(buf) }} }

        let step = read_u64!();
        let a    = read_f64!();
        let np   = read_u64!() as usize;

        let mut sim = Simulation::new(config);
        sim.step    = step;
        sim.a       = a;

        for i in 0..np {
            for d in 0..3 { sim.particles.pos[i][d] = read_f64!(); }
            for d in 0..3 { sim.particles.vel[i][d] = read_f64!(); }
        }
        log::info!("Restarted from '{path}': step={step}, a={a:.5}");
        Ok(sim)
    }
}
