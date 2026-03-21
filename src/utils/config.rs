//! Simulation parameter file  (TOML, analogous to RAMSES `.nml` namelist).
//!
//! Load with:
//! ```no_run
//! use cosmo_sim::utils::config::Config;
//! let cfg = Config::from_file("params.toml").unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::utils::types::Real;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub run:          RunParams,
    pub cosmology:    CosmoParams,
    #[serde(default)]
    pub gravity:      GravityParams,
    pub particles:    ParticleParams,
    #[serde(default)]
    pub scalar_field: ScalarFieldParams,
    pub output:       OutputParams,
}

// ── Sub-structs ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunParams {
    /// Comoving box side length  [Mpc/h]
    pub box_size:       Real,
    /// Base (domain) grid cells per dimension
    pub n_grid:         u32,
    /// Maximum number of AMR refinement levels above the domain grid
    pub levelmax:       u8,
    /// AMR refinement threshold: refine cell if N_part > refine_thr
    pub refine_thr:     u32,
    /// Initial scale factor
    pub a_init:         Real,
    /// Final scale factor
    pub a_final:        Real,
    /// Maximum allowed time step in supercomoving time
    pub dtau_max:       Real,
    /// CFL-like safety factor for time stepping
    pub courant_factor: Real,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmoParams {
    pub omega_m:      Real,
    pub omega_lambda: Real,
    pub omega_b:      Real,
    pub h:            Real,
    pub n_s:          Real,
    pub sigma8:       Real,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityParams {
    /// Poisson solver convergence tolerance
    pub epsilon:      Real,
    /// Maximum multigrid V-cycle iterations
    pub n_mg_iter:    u32,
    /// Number of pre/post-smoothing sweeps (Gauss-Seidel)
    pub n_smooth:     u32,
}

impl Default for GravityParams {
    fn default() -> Self {
        Self { epsilon: 1e-4, n_mg_iter: 50, n_smooth: 2 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleParams {
    /// Number of dark matter particles per side  (total = n_part^3)
    pub n_part:       u32,
    /// Particle mass in M_sun/h (computed automatically if 0)
    pub m_part:       Real,
}

/// Scalar field / modified gravity parameters (ISIS-style).
/// Activate at compile time via cargo features `symmetron` or `fofr`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScalarFieldParams {
    // ── Symmetron ──────────────────────────────────────────────────────
    /// Scale factor at spontaneous symmetry breaking (vacuum, homogeneous)
    pub a_ssb:        Option<Real>,
    /// Compton wavelength in vacuum  [Mpc/h]
    pub lambda_0:     Option<Real>,
    /// Coupling strength β (fifth-force / Newton ratio in broken phase)
    pub beta_0:       Option<Real>,

    // ── f(R) Hu-Sawicki ────────────────────────────────────────────────
    /// f_{R0} present-day background value
    pub fofr0:        Option<Real>,
    /// Power-law index n
    pub n_fofr:       Option<Real>,

    // ── Solver ─────────────────────────────────────────────────────────
    /// Multigrid convergence tolerance for scalar field solver
    pub epsilon_mg:   Option<Real>,
    /// Include non-static (time-derivative) terms  (Llinares & Mota 2014)
    pub non_static:   Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParams {
    /// Directory for output files
    pub outdir:       String,
    /// Number of output snapshots
    pub n_output:     u32,
    /// Output snapshot scale factors (overrides n_output if set)
    pub a_out:        Option<Vec<Real>>,
    /// Write HDF5 (requires hdf5_output feature)
    #[serde(default)]
    pub hdf5:         bool,
}

// ── I/O ─────────────────────────────────────────────────────────────────────

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&text)?;
        Ok(cfg)
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}

/// Generate a ready-to-run example configuration (Planck 2018 cosmology,
/// 64^3 grid, Symmetron parameters matching Llinares et al. 2014).
pub fn example_config() -> Config {
    Config {
        run: RunParams {
            box_size:       64.0,
            n_grid:         64,
            levelmax:       3,
            refine_thr:     8,
            a_init:         0.02,
            a_final:        1.0,
            dtau_max:       0.01,
            courant_factor: 0.8,
        },
        cosmology: CosmoParams {
            omega_m:      0.3175,
            omega_lambda: 0.6825,
            omega_b:      0.049,
            h:            0.6711,
            n_s:          0.9624,
            sigma8:       0.8344,
        },
        gravity: GravityParams {
            epsilon:   1e-4,
            n_mg_iter: 50,
            n_smooth:  2,
        },
        particles: ParticleParams {
            n_part: 256,
            m_part: 0.0,   // auto-computed
        },
        scalar_field: ScalarFieldParams {
            // Symmetron parameters from Llinares et al. 2014
            a_ssb:      Some(0.5),
            lambda_0:   Some(1.0),   // Mpc/h
            beta_0:     Some(1.0),
            fofr0:      None,
            n_fofr:     None,
            epsilon_mg: Some(1e-5),
            non_static: Some(false),
        },
        output: OutputParams {
            outdir:   "output".into(),
            n_output: 10,
            a_out:    None,
            hdf5:     false,
        },
    }
}
