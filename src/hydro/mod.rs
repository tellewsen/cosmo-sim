//! Eulerian gas hydrodynamics on the domain grid.
//!
//! Compiled only with `--features hydro`.
//! See `docs/specs/2026-04-11-hydro-design.md` for architecture.

pub mod riemann;
pub mod solver;
pub mod ic;
pub mod cooling;
pub mod star_formation;
pub mod feedback;
pub mod profiles;
