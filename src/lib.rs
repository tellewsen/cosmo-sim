//! cosmo_sim — cosmological N-body simulation library.
//!
//! Based on ISIS/RAMSES (Llinares, Mota & Winther 2014, A&A 562 A78;
//! Teyssier 2002, A&A 385 337).

pub mod utils {
    pub mod config;
    pub mod cosmology;
    pub mod types;
}

pub mod gravity {
    pub mod poisson;
}

#[allow(clippy::module_inception)]
pub mod particles {
    pub mod particles;
}

pub mod scalar_field {
    pub mod fofr;
    pub mod symmetron;
}

pub mod amr {
    pub mod grid;
}

pub mod ic {
    pub mod zeldovich;
}

pub mod analysis {
    pub mod power_spectrum;
}

pub mod mpi;

pub mod simulation;

#[cfg(feature = "hydro")]
pub mod hydro;

// Top-level re-exports used by main.rs and integration tests
pub use simulation::Simulation;
pub use utils::config::example_config;

// Index helpers re-exported for integration tests
pub use utils::types::{from_flat, idx3p};
