//! cosmo_sim — cosmological N-body simulation library.
//!
//! Based on ISIS/RAMSES (Llinares, Mota & Winther 2014, A&A 562 A78;
//! Teyssier 2002, A&A 385 337).

pub mod utils {
    pub mod types;
    pub mod cosmology;
    pub mod config;
}

pub mod gravity {
    pub mod poisson;
}

pub mod particles {
    pub mod particles;
}

pub mod scalar_field {
    pub mod symmetron;
    pub mod fofr;
}

pub mod amr {
    pub mod grid;
}

pub mod simulation;

// Top-level re-exports used by main.rs and integration tests
pub use simulation::Simulation;
pub use utils::config::example_config;

// Index helpers re-exported for integration tests
pub use utils::types::{idx3p, from_flat};
pub use gravity::poisson::{flat3p, flat3};
