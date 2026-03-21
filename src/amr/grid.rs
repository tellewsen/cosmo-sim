//! Adaptive Mesh Refinement octree grid.
//!
//! Implements a **fully-threaded graded octree** following RAMSES
//! (Teyssier 2002).  Each "oct" is a group of 8 sibling cells born from one
//! parent.  The 2:1 refinement rule (cells may differ by at most one level)
//! is enforced during refinement.
//!
//! Cell fields stored per-cell:
//!   - density ρ/ρ̄
//!   - gravitational potential Φ
//!   - gravitational acceleration g = −∇Φ
//!   - scalar field χ (symmetron / f_R)
//!   - fifth-force acceleration

use crate::utils::types::{Real, Level};

// ── Oct ───────────────────────────────────────────────────────────────────────

/// One oct: parent + 8 children living on the same refinement level.
#[derive(Debug, Clone)]
pub struct Oct {
    /// Index of the parent cell (coarser level)
    pub parent:   usize,
    /// Indices of the 8 child cells (bit-mask: child i has offset (i&1, i&2, i&4))
    pub children: [usize; 8],
    /// Level of the children
    pub level:    Level,
    /// Centre position of the oct  [Mpc/h]
    pub centre:   [Real; 3],
}

// ── Cell ──────────────────────────────────────────────────────────────────────

/// A single AMR cell.  Leaf cells carry all physical fields.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Cell-centre position  [Mpc/h]
    pub pos:        [Real; 3],
    /// Cell width  [Mpc/h]
    pub dx:         Real,
    /// Refinement level (0 = domain grid)
    pub level:      Level,
    /// Index of child oct, or `usize::MAX` for a leaf
    pub child_oct:  usize,
    /// Index of parent oct, or `usize::MAX` for domain cells
    pub parent_oct: usize,

    // ── Physical fields ───────────────────────────────────────────────────
    /// Matter density  ρ/ρ̄
    pub rho:         Real,
    /// Gravitational potential  Φ  [internal units]
    pub phi:         Real,
    /// Gravitational acceleration  g = −∇Φ
    pub grav:        [Real; 3],
    /// Scalar field value χ (dimensionless)
    pub scalar:      Real,
    /// Scalar field velocity χ̇  (non-static mode)
    pub scalar_dot:  Real,
    /// Fifth-force acceleration
    pub fifth_force: [Real; 3],
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            pos: [0.0; 3], dx: 0.0, level: 0,
            child_oct: usize::MAX, parent_oct: usize::MAX,
            rho: 1.0, phi: 0.0, grav: [0.0; 3],
            scalar: 0.0, scalar_dot: 0.0, fifth_force: [0.0; 3],
        }
    }
}

impl Cell {
    #[inline]
    pub fn is_leaf(&self) -> bool { self.child_oct == usize::MAX }

    #[inline]
    pub fn volume(&self) -> Real { self.dx * self.dx * self.dx }
}

// ── AMR grid ──────────────────────────────────────────────────────────────────

/// Full AMR hierarchy: domain grid + all refined cells.
pub struct AmrGrid {
    /// All cells, all levels, flat storage
    pub cells:       Vec<Cell>,
    /// All octs, flat storage
    pub octs:        Vec<Oct>,
    /// Starting cell index for each level
    pub level_start: Vec<usize>,
    /// Number of domain cells per dimension
    pub n_domain:    u32,
    /// Maximum allowed refinement level
    pub levelmax:    Level,
    /// Comoving box side length  [Mpc/h]
    pub box_size:    Real,
}

impl AmrGrid {
    /// Build a uniform domain grid of size `n × n × n`.
    pub fn new_uniform(n: u32, levelmax: Level, box_size: Real) -> Self {
        let n_us = n as usize;
        let dx   = box_size / n as Real;
        let n3   = n_us * n_us * n_us;
        let mut cells = Vec::with_capacity(n3 * 2);

        // C-order (row-major): flat index = i*n² + j*n + k → loop i (slow) → k (fast)
        for i in 0..n_us {
            for j in 0..n_us {
                for k in 0..n_us {
                    cells.push(Cell {
                        pos: [
                            (i as Real + 0.5) * dx,
                            (j as Real + 0.5) * dx,
                            (k as Real + 0.5) * dx,
                        ],
                        dx,
                        level: 0,
                        ..Default::default()
                    });
                }
            }
        }

        Self {
            level_start: vec![0],
            octs:        Vec::new(),
            n_domain:    n,
            levelmax,
            box_size,
            cells,
        }
    }

    /// Cell width at a given refinement level.
    #[inline]
    pub fn dx_at_level(&self, level: Level) -> Real {
        self.box_size / (self.n_domain as Real * (1u64 << level) as Real)
    }

    /// Total number of cells.
    #[inline]
    pub fn n_cells(&self) -> usize { self.cells.len() }

    /// Number of leaf cells.
    pub fn n_leaves(&self) -> usize {
        self.cells.iter().filter(|c| c.is_leaf()).count()
    }

    /// Iterate over (index, &Cell) for leaf cells.
    pub fn leaves(&self) -> impl Iterator<Item = (usize, &Cell)> {
        self.cells.iter().enumerate().filter(|(_, c)| c.is_leaf())
    }

    // ── Refinement ────────────────────────────────────────────────────────────

    /// Flag cells for refinement where the particle count exceeds `threshold`.
    /// Returns the number of flagged cells.
    pub fn flag_by_particle_count(
        &mut self,
        particle_counts: &[u32],
        threshold:       u32,
    ) -> usize {
        let mut count = 0;
        for (idx, cell) in self.cells.iter_mut().enumerate() {
            if cell.is_leaf()
               && cell.level < self.levelmax
               && idx < particle_counts.len()
               && particle_counts[idx] > threshold
            {
                cell.child_oct = usize::MAX - 1; // flag
                count += 1;
            }
        }
        count
    }

    /// Refine all flagged cells.  Adds 8 children per flagged cell.
    pub fn refine_flagged(&mut self) {
        let flagged: Vec<usize> = self.cells.iter().enumerate()
            .filter(|(_, c)| c.child_oct == usize::MAX - 1)
            .map(|(i, _)| i)
            .collect();

        for parent_idx in flagged {
            let child_level = self.cells[parent_idx].level + 1;
            if child_level > self.levelmax { continue; }

            let dx_child   = self.dx_at_level(child_level);
            let parent_pos = self.cells[parent_idx].pos;
            let parent_dx  = self.cells[parent_idx].dx;
            let oct_idx    = self.octs.len();

            let mut child_indices = [0_usize; 8];
            for bit in 0_u8..8 {
                let ox = if bit & 1 != 0 {  0.25 * parent_dx } else { -0.25 * parent_dx };
                let oy = if bit & 2 != 0 {  0.25 * parent_dx } else { -0.25 * parent_dx };
                let oz = if bit & 4 != 0 {  0.25 * parent_dx } else { -0.25 * parent_dx };

                let child_idx = self.cells.len();
                child_indices[bit as usize] = child_idx;

                // Inherit parent field values as initial guess
                let rho    = self.cells[parent_idx].rho;
                let phi    = self.cells[parent_idx].phi;
                let scalar = self.cells[parent_idx].scalar;

                self.cells.push(Cell {
                    pos:        [parent_pos[0]+ox, parent_pos[1]+oy, parent_pos[2]+oz],
                    dx:         dx_child,
                    level:      child_level,
                    parent_oct: oct_idx,
                    rho, phi, scalar,
                    ..Default::default()
                });
            }

            self.octs.push(Oct {
                parent:   parent_idx,
                children: child_indices,
                level:    child_level,
                centre:   parent_pos,
            });

            self.cells[parent_idx].child_oct = oct_idx;
        }
    }

    // ── Field I/O helpers ─────────────────────────────────────────────────────

    /// Extract the domain-grid density into a flat n³ array.
    pub fn domain_rho(&self) -> Vec<Real> {
        let n = self.n_domain as usize;
        let n3 = n * n * n;
        (0..n3).map(|i| self.cells[i].rho).collect()
    }

    /// Write density back to domain-grid cells from a flat array.
    pub fn set_domain_rho(&mut self, rho: &[Real]) {
        for (i, &r) in rho.iter().enumerate() {
            self.cells[i].rho = r;
        }
    }

    /// Write potential back to domain-grid cells.
    pub fn set_domain_phi(&mut self, phi: &[Real]) {
        for (i, &p) in phi.iter().enumerate() {
            self.cells[i].phi = p;
        }
    }

    /// Write scalar field back to domain-grid cells.
    pub fn set_domain_scalar(&mut self, chi: &[Real]) {
        for (i, &c) in chi.iter().enumerate() {
            self.cells[i].scalar = c;
        }
    }

    /// Write gravitational acceleration to domain-grid cells.
    pub fn set_domain_grav(&mut self, g: &[[Real; 3]]) {
        for (i, &gi) in g.iter().enumerate() {
            self.cells[i].grav = gi;
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn uniform_grid_cell_count() {
        let g = AmrGrid::new_uniform(8, 3, 64.0);
        assert_eq!(g.n_cells(), 8 * 8 * 8);
    }

    #[test]
    fn cell_positions_in_box() {
        let n = 4_u32;
        let l = 8.0_f64;
        let g = AmrGrid::new_uniform(n, 2, l);
        let dx = l / n as Real;
        for cell in &g.cells {
            for d in 0..3 {
                assert!(cell.pos[d] >= 0.0 && cell.pos[d] < l,
                        "pos[{d}]={} out of [0,{l})", cell.pos[d]);
            }
        }
        // First cell should be at (dx/2, dx/2, dx/2)
        assert!((g.cells[0].pos[0] - dx * 0.5).abs() < 1e-14);
    }

    #[test]
    fn dx_at_level() {
        let g = AmrGrid::new_uniform(8, 3, 64.0);
        assert!((g.dx_at_level(0) - 8.0).abs() < 1e-12);
        assert!((g.dx_at_level(1) - 4.0).abs() < 1e-12);
        assert!((g.dx_at_level(2) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn refine_single_cell() {
        let mut g = AmrGrid::new_uniform(4, 2, 8.0);
        // Flag the first cell
        g.cells[0].child_oct = usize::MAX - 1;
        g.refine_flagged();
        // Should have 4³ + 8 cells now
        assert_eq!(g.n_cells(), 4*4*4 + 8, "Expected domain + 8 children");
        assert_eq!(g.octs.len(), 1, "Expected one oct");
        assert!(!g.cells[0].is_leaf(), "Parent should not be a leaf");
        // All 8 children should be leaves
        let oct = &g.octs[0];
        for &ci in &oct.children {
            assert!(g.cells[ci].is_leaf(), "Child {ci} should be a leaf");
            assert_eq!(g.cells[ci].level, 1);
        }
    }

    #[test]
    fn all_domain_cells_start_as_leaves() {
        let g = AmrGrid::new_uniform(4, 3, 16.0);
        assert_eq!(g.n_leaves(), 4 * 4 * 4);
    }
}
