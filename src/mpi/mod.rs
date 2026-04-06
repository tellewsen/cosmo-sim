//! MPI parallelism layer for distributed N-body simulation.
//!
//! # Domain decomposition
//!
//! Slab decomposition along the **i-axis** (x, slowest-varying in C-order).
//! Rank `r` owns i-slabs `[r*N/size .. (r+1)*N/size)` of the global N^3 grid.
//! Each rank allocates its local slab plus `n_ghost` layers on each side for
//! halo exchange (needed by stencil operations: Laplacian, gradient, CIC).
//!
//! This mirrors the RAMSES domain decomposition strategy and keeps the
//! fastest-varying index (k) contiguous in memory for cache efficiency.
//!
//! # Particle ownership
//!
//! A particle belongs to the rank whose slab contains its x-position.
//! After each drift step, particles that crossed slab boundaries are migrated
//! to the correct rank via point-to-point send/recv.
//!
//! # Reference
//!
//! Teyssier (2002), A&A 385, 337 -- RAMSES domain decomposition.

use crate::utils::types::Real;

/// MPI context holding rank, communicator size, and the MPI universe handle.
///
/// The universe must live for the entire program duration (MPI spec requirement).
/// Dropping it calls `MPI_Finalize`.
pub struct MpiContext {
    pub rank: usize,
    pub size: usize,
    /// Keep the universe alive; the communicator borrows from it.
    #[cfg(feature = "mpi")]
    _universe: ::mpi::environment::Universe,
}

impl MpiContext {
    /// Initialise MPI and return the context.
    ///
    /// Must be called exactly once, before any other MPI operation.
    /// In non-MPI builds this returns a single-rank stub.
    #[cfg(feature = "mpi")]
    pub fn init() -> Self {
        use ::mpi::traits::Communicator;
        let universe = ::mpi::initialize().expect("MPI_Init failed");
        let world = universe.world();
        let rank = world.rank() as usize;
        let size = world.size() as usize;
        log::info!("MPI initialised: rank {rank}/{size}");
        Self {
            rank,
            size,
            _universe: universe,
        }
    }

    /// Single-rank stub for non-MPI builds.
    #[cfg(not(feature = "mpi"))]
    pub fn init() -> Self {
        Self { rank: 0, size: 1 }
    }

    /// True on the root rank (rank 0).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    // ── Slab geometry ────────────────────────────────────────────────────────

    /// First global i-index owned by this rank (inclusive).
    #[inline]
    pub fn slab_start(&self, n_global: usize) -> usize {
        self.rank * n_global / self.size
    }

    /// One past the last global i-index owned by this rank (exclusive).
    #[inline]
    pub fn slab_end(&self, n_global: usize) -> usize {
        (self.rank + 1) * n_global / self.size
    }

    /// Number of i-slabs owned by this rank.
    #[inline]
    pub fn slab_count(&self, n_global: usize) -> usize {
        self.slab_end(n_global) - self.slab_start(n_global)
    }

    /// Lower x-boundary of this rank's slab in comoving coordinates.
    #[inline]
    pub fn x_lo(&self, n_global: usize, box_size: Real) -> Real {
        self.slab_start(n_global) as Real * box_size / n_global as Real
    }

    /// Upper x-boundary of this rank's slab in comoving coordinates.
    #[inline]
    pub fn x_hi(&self, n_global: usize, box_size: Real) -> Real {
        self.slab_end(n_global) as Real * box_size / n_global as Real
    }

    // ── Halo exchange ────────────────────────────────────────────────────────

    /// Exchange ghost slabs with neighbouring ranks along the i-axis.
    ///
    /// `grid` is the **local** slab of shape `(n_local_i + 2*n_ghost) * n * n`
    /// stored in C-order.  The first `n_ghost` and last `n_ghost` i-layers are
    /// ghost zones that get filled from the left and right neighbours.
    ///
    /// `n_local_i` is the number of owned i-slabs (excluding ghosts).
    /// `n` is the global grid size along j and k (same for all ranks).
    /// `n_ghost` is the number of ghost layers on each side.
    ///
    /// For periodic boundaries, rank 0's left neighbour is rank (size-1)
    /// and rank (size-1)'s right neighbour is rank 0.
    #[cfg(feature = "mpi")]
    pub fn exchange_halos(
        &self,
        grid: &mut [Real],
        n_local_i: usize,
        n: usize,
        n_ghost: usize,
    ) {
        use ::mpi::traits::{Communicator, Destination, Source};
        use ::mpi::point_to_point as p2p;

        let universe = unsafe {
            // SAFETY: we only access the world communicator which is valid
            // for the lifetime of _universe stored in self.
            &*(&self._universe as *const ::mpi::environment::Universe)
        };
        let world = universe.world();

        let slab_size = n * n; // one i-layer = n*n cells
        let ghost_count = n_ghost * slab_size;

        let left_rank = if self.rank == 0 { self.size - 1 } else { self.rank - 1 };
        let right_rank = if self.rank == self.size - 1 { 0 } else { self.rank + 1 };

        let left_proc = world.process_at_rank(left_rank as i32);
        let right_proc = world.process_at_rank(right_rank as i32);

        // Send right boundary -> right neighbour's left ghost;
        // receive from left neighbour into left ghost.
        {
            // Our rightmost n_ghost owned layers start at offset n_ghost*slab_size + (n_local_i - n_ghost)*slab_size
            let send_start = n_ghost * slab_size + (n_local_i - n_ghost) * slab_size;
            let send_end = send_start + ghost_count;
            let recv_start = 0;
            let recv_end = ghost_count;

            let send_buf = grid[send_start..send_end].to_vec();
            let mut recv_buf = vec![0.0_f64; ghost_count];

            // Tag 0: rightward sends, leftward receives
            p2p::send_receive_into(&right_proc, &send_buf[..], &left_proc, &mut recv_buf[..]);
            grid[recv_start..recv_end].copy_from_slice(&recv_buf);
        }

        // Send left boundary -> left neighbour's right ghost;
        // receive from right neighbour into right ghost.
        {
            let send_start = n_ghost * slab_size;
            let send_end = send_start + ghost_count;
            let recv_start = (n_ghost + n_local_i) * slab_size;
            let recv_end = recv_start + ghost_count;

            let send_buf = grid[send_start..send_end].to_vec();
            let mut recv_buf = vec![0.0_f64; ghost_count];

            // Tag 1: leftward sends, rightward receives
            p2p::send_receive_into(&left_proc, &send_buf[..], &right_proc, &mut recv_buf[..]);
            grid[recv_start..recv_end].copy_from_slice(&recv_buf);
        }
    }

    /// No-op halo exchange for non-MPI builds.
    #[cfg(not(feature = "mpi"))]
    pub fn exchange_halos(
        &self,
        _grid: &mut [Real],
        _n_local_i: usize,
        _n: usize,
        _n_ghost: usize,
    ) {
        // Single-rank: no ghosts to exchange.
    }

    // ── Particle migration ───────────────────────────────────────────────────

    /// Migrate particles that crossed slab boundaries after a drift step.
    ///
    /// Particles whose x-position falls outside `[x_lo, x_hi)` are sent to
    /// the appropriate neighbour.  Periodic wrapping is applied.
    ///
    /// `pos` and `vel` are modified in-place: departing particles are removed,
    /// arriving particles are appended.
    #[cfg(feature = "mpi")]
    pub fn migrate_particles(
        &self,
        pos: &mut Vec<[Real; 3]>,
        vel: &mut Vec<[Real; 3]>,
        n_global: usize,
        box_size: Real,
    ) {
        use ::mpi::traits::{Communicator, Destination, Source};

        let universe = unsafe {
            &*(&self._universe as *const ::mpi::environment::Universe)
        };
        let world = universe.world();

        let x_lo = self.x_lo(n_global, box_size);
        let x_hi = self.x_hi(n_global, box_size);

        let left_rank = if self.rank == 0 { self.size - 1 } else { self.rank - 1 };
        let right_rank = if self.rank == self.size - 1 { 0 } else { self.rank + 1 };

        // Partition particles into: keep, send-left, send-right.
        let mut keep_pos = Vec::with_capacity(pos.len());
        let mut keep_vel = Vec::with_capacity(vel.len());
        let mut left_pos: Vec<[Real; 3]> = Vec::new();
        let mut left_vel: Vec<[Real; 3]> = Vec::new();
        let mut right_pos: Vec<[Real; 3]> = Vec::new();
        let mut right_vel: Vec<[Real; 3]> = Vec::new();

        for i in 0..pos.len() {
            let x = pos[i][0].rem_euclid(box_size);
            if x < x_lo {
                let mut p = pos[i];
                p[0] = x; // wrapped
                left_pos.push(p);
                left_vel.push(vel[i]);
            } else if x >= x_hi {
                let mut p = pos[i];
                p[0] = x; // wrapped
                right_pos.push(p);
                right_vel.push(vel[i]);
            } else {
                keep_pos.push(pos[i]);
                keep_vel.push(vel[i]);
            }
        }

        // Flatten to f64 arrays for MPI transfer (6 f64 per particle).
        let flatten = |p: &[[Real; 3]], v: &[[Real; 3]]| -> Vec<Real> {
            let mut buf = Vec::with_capacity(p.len() * 6);
            for i in 0..p.len() {
                buf.extend_from_slice(&p[i]);
                buf.extend_from_slice(&v[i]);
            }
            buf
        };

        let left_proc = world.process_at_rank(left_rank as i32);
        let right_proc = world.process_at_rank(right_rank as i32);

        // Exchange with right neighbour: send right_buf, receive from right
        let send_right = flatten(&right_pos, &right_vel);
        let (recv_from_right, _status) = right_proc.send_receive(&send_right[..], &left_proc);

        // Exchange with left neighbour: send left_buf, receive from left
        let send_left = flatten(&left_pos, &left_vel);
        let (recv_from_left, _status) = left_proc.send_receive(&send_left[..], &right_proc);

        // Unpack received particles.
        let unflatten = |buf: &[Real], p: &mut Vec<[Real; 3]>, v: &mut Vec<[Real; 3]>| {
            let n = buf.len() / 6;
            for i in 0..n {
                p.push([buf[i*6], buf[i*6+1], buf[i*6+2]]);
                v.push([buf[i*6+3], buf[i*6+4], buf[i*6+5]]);
            }
        };

        unflatten(&recv_from_right, &mut keep_pos, &mut keep_vel);
        unflatten(&recv_from_left, &mut keep_pos, &mut keep_vel);

        *pos = keep_pos;
        *vel = keep_vel;
    }

    /// No-op particle migration for non-MPI builds.
    #[cfg(not(feature = "mpi"))]
    pub fn migrate_particles(
        &self,
        _pos: &mut Vec<[Real; 3]>,
        _vel: &mut Vec<[Real; 3]>,
        _n_global: usize,
        _box_size: Real,
    ) {
        // Single-rank: all particles stay local.
    }

    // ── Global reductions ────────────────────────────────────────────────────

    /// Global maximum across all ranks.
    #[cfg(feature = "mpi")]
    pub fn allreduce_max(&self, val: Real) -> Real {
        use ::mpi::traits::Communicator;
        use ::mpi::collective::SystemOperation;
        let world = self._universe.world();
        let mut result = 0.0_f64;
        world.all_reduce_into(&val, &mut result, SystemOperation::max());
        result
    }

    #[cfg(not(feature = "mpi"))]
    pub fn allreduce_max(&self, val: Real) -> Real {
        val
    }

    /// Global sum across all ranks.
    #[cfg(feature = "mpi")]
    pub fn allreduce_sum(&self, val: Real) -> Real {
        use ::mpi::traits::Communicator;
        use ::mpi::collective::SystemOperation;
        let world = self._universe.world();
        let mut result = 0.0_f64;
        world.all_reduce_into(&val, &mut result, SystemOperation::sum());
        result
    }

    #[cfg(not(feature = "mpi"))]
    pub fn allreduce_sum(&self, val: Real) -> Real {
        val
    }

    /// Gather a local `Vec<Real>` from all ranks onto rank 0.
    ///
    /// Returns `Some(gathered)` on rank 0, `None` on other ranks.
    /// Used for power spectrum computation where all density data must be
    /// assembled on one rank.
    #[cfg(feature = "mpi")]
    pub fn gather_to_root(&self, local: &[Real]) -> Option<Vec<Real>> {
        use ::mpi::traits::Communicator;
        use ::mpi::collective::Root;
        let world = self._universe.world();
        let root_process = world.process_at_rank(0);

        if self.is_root() {
            // Gather counts first so we know total size
            let local_len = local.len() as i32;
            let counts = root_process.gather(&local_len);
            // Slightly simplified: assume equal-sized slabs for now
            let total: i32 = counts.iter().sum();
            let mut gathered = vec![0.0_f64; total as usize];
            root_process.gather_into_root(local, &mut gathered);
            Some(gathered)
        } else {
            let local_len = local.len() as i32;
            root_process.gather(&local_len);
            root_process.gather_into(local);
            None
        }
    }

    #[cfg(not(feature = "mpi"))]
    pub fn gather_to_root(&self, local: &[Real]) -> Option<Vec<Real>> {
        Some(local.to_vec())
    }

    /// Allreduce a vector element-wise (sum).
    ///
    /// Used for accumulating density grids across ranks when particles
    /// deposit across slab boundaries.
    #[cfg(feature = "mpi")]
    pub fn allreduce_sum_vec(&self, local: &[Real]) -> Vec<Real> {
        use ::mpi::traits::Communicator;
        use ::mpi::collective::SystemOperation;
        let world = self._universe.world();
        let mut result = vec![0.0_f64; local.len()];
        world.all_reduce_into(local, &mut result[..], SystemOperation::sum());
        result
    }

    #[cfg(not(feature = "mpi"))]
    pub fn allreduce_sum_vec(&self, local: &[Real]) -> Vec<Real> {
        local.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_geometry_divides_evenly() {
        let ctx = MpiContext::init();
        let n = 64;
        assert_eq!(ctx.slab_start(n), 0);
        assert_eq!(ctx.slab_end(n), 64);
        assert_eq!(ctx.slab_count(n), 64);
    }

    #[test]
    fn slab_geometry_coordinates() {
        let ctx = MpiContext::init();
        let n = 64;
        let box_size = 250.0;
        assert!((ctx.x_lo(n, box_size) - 0.0).abs() < 1e-14);
        assert!((ctx.x_hi(n, box_size) - 250.0).abs() < 1e-14);
    }

    #[test]
    fn non_mpi_reductions_are_identity() {
        let ctx = MpiContext::init();
        assert!((ctx.allreduce_max(3.14) - 3.14).abs() < 1e-14);
        assert!((ctx.allreduce_sum(2.71) - 2.71).abs() < 1e-14);
    }

    #[test]
    fn non_mpi_gather_returns_copy() {
        let ctx = MpiContext::init();
        let data = vec![1.0, 2.0, 3.0];
        let result = ctx.gather_to_root(&data);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn non_mpi_allreduce_vec_is_identity() {
        let ctx = MpiContext::init();
        let data = vec![1.0, 2.0, 3.0];
        let result = ctx.allreduce_sum_vec(&data);
        assert_eq!(result, data);
    }
}
