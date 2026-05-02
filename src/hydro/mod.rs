//! Eulerian gas hydrodynamics on the domain grid.
//!
//! Compiled only with `--features hydro`.
//! See `docs/specs/2026-04-11-hydro-design.md` for architecture.

pub mod cooling;
pub mod feedback;
pub mod ic;
pub mod profiles;
pub mod riemann;
pub mod solver;
pub mod star_formation;
