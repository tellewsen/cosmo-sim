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
        use ::mpi::traits::Communicator;
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
            let send_start = n_ghost * slab_size + (n_local_i - n_ghost) * slab_size;
            let send_end = send_start + ghost_count;
            let recv_start = 0;
            let recv_end = ghost_count;

            let send_buf = grid[send_start..send_end].to_vec();
            let mut recv_buf = vec![0.0_f64; ghost_count];

            // send_receive_into(msg, destination, buf, source)
            p2p::send_receive_into(&send_buf[..], &right_proc, &mut recv_buf[..], &left_proc);
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

            p2p::send_receive_into(&send_buf[..], &left_proc, &mut recv_buf[..], &right_proc);
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
        use ::mpi::traits::Communicator;
        use ::mpi::point_to_point as p2p;

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

        let send_right = flatten(&right_pos, &right_vel);
        let send_left  = flatten(&left_pos,  &left_vel);

        // Step 1: exchange counts so each side knows how big a buffer to allocate.
        let right_count = send_right.len() as i64;
        let left_count  = send_left.len()  as i64;
        let mut recv_from_left_count  = 0i64;
        let mut recv_from_right_count = 0i64;
        p2p::send_receive_into(&right_count, &right_proc, &mut recv_from_left_count,  &left_proc);
        p2p::send_receive_into(&left_count,  &left_proc,  &mut recv_from_right_count, &right_proc);

        // Step 2: exchange the actual particle data.
        let mut recv_from_right = vec![0.0_f64; recv_from_right_count as usize];
        let mut recv_from_left  = vec![0.0_f64; recv_from_left_count  as usize];
        p2p::send_receive_into(&send_right[..], &right_proc, &mut recv_from_left[..],  &left_proc);
        p2p::send_receive_into(&send_left[..],  &left_proc,  &mut recv_from_right[..], &right_proc);

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
        use ::mpi::collective::{CommunicatorCollectives, SystemOperation};
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
        use ::mpi::collective::{CommunicatorCollectives, SystemOperation};
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
        use ::mpi::collective::{CommunicatorCollectives, Root};
        use ::mpi::datatype::PartitionMut;

        let world = self._universe.world();
        let root_proc = world.process_at_rank(0);

        // All ranks allgather each other's counts so everyone knows all sizes.
        let local_count = local.len() as i32;
        let mut all_counts = vec![0i32; self.size];
        world.all_gather_into(&local_count, &mut all_counts[..]);

        if self.is_root() {
            let total: usize = all_counts.iter().map(|&c| c as usize).sum();
            let displs: Vec<i32> = all_counts.iter()
                .scan(0i32, |acc, &c| { let d = *acc; *acc += c; Some(d) })
                .collect();
            let mut gathered = vec![0.0_f64; total];
            let mut partition = PartitionMut::new(
                &mut gathered[..],
                &all_counts[..],
                &displs[..],
            );
            root_proc.gather_varcount_into_root(local, &mut partition);
            Some(gathered)
        } else {
            root_proc.gather_varcount_into(local);
            None
        }
    }

    #[cfg(not(feature = "mpi"))]
    pub fn gather_to_root(&self, local: &[Real]) -> Option<Vec<Real>> {
        Some(local.to_vec())
    }

    /// Gather all particles from every rank onto rank 0.
    ///
    /// All ranks must call this (it is a collective operation).
    /// Returns `Some((pos, vel))` on rank 0 with the full particle set,
    /// `None` on all other ranks.
    ///
    /// Reuses `gather_to_root` by flattening pos+vel to a `Vec<Real>`
    /// (6 f64 per particle: x,y,z,vx,vy,vz).
    #[cfg(feature = "mpi")]
    pub fn gather_particles_to_root(
        &self,
        pos: &[[Real; 3]],
        vel: &[[Real; 3]],
    ) -> Option<(Vec<[Real; 3]>, Vec<[Real; 3]>)> {
        let mut flat = Vec::with_capacity(pos.len() * 6);
        for i in 0..pos.len() {
            flat.extend_from_slice(&pos[i]);
            flat.extend_from_slice(&vel[i]);
        }
        self.gather_to_root(&flat).map(|gathered| {
            let n = gathered.len() / 6;
            let mut all_pos = Vec::with_capacity(n);
            let mut all_vel = Vec::with_capacity(n);
            for i in 0..n {
                all_pos.push([gathered[i*6], gathered[i*6+1], gathered[i*6+2]]);
                all_vel.push([gathered[i*6+3], gathered[i*6+4], gathered[i*6+5]]);
            }
            (all_pos, all_vel)
        })
    }

    /// Non-MPI stub: single rank already has all particles.
    #[cfg(not(feature = "mpi"))]
    pub fn gather_particles_to_root(
        &self,
        pos: &[[Real; 3]],
        vel: &[[Real; 3]],
    ) -> Option<(Vec<[Real; 3]>, Vec<[Real; 3]>)> {
        Some((pos.to_vec(), vel.to_vec()))
    }

    /// Exchange density ghost contributions with neighbouring ranks.
    ///
    /// After per-rank CIC, each rank's `rho` (full N³ C-order array) contains
    /// only contributions from its own particles.  Due to the 2-cell CIC stencil,
    /// particles near a slab boundary deposit 1 cell into the adjacent slab.
    /// This method performs two O(N²) nearest-neighbour exchanges so that each
    /// rank's local slab `rho[i_lo*n²..i_hi*n²]` becomes fully correct.
    ///
    /// Replaces `allreduce_sum_vec` (O(N³) all-to-all) with O(N²) per rank.
    /// After the call only the local slab is correct; cells outside it retain
    /// the partial (single-rank) deposits and should not be read.
    ///
    /// `n` is the global grid side length (cubic grid assumed).
    #[cfg(feature = "mpi")]
    pub fn reduce_ghost_density(&self, rho: &mut [Real], n: usize) {
        if self.size == 1 {
            return; // Single MPI rank: CIC already wrote the complete density.
        }

        use ::mpi::traits::Communicator;
        use ::mpi::point_to_point as p2p;

        let universe = unsafe {
            &*(&self._universe as *const ::mpi::environment::Universe)
        };
        let world = universe.world();

        let i_lo = self.slab_start(n);
        let i_hi = self.slab_end(n);
        let n2 = n * n;

        let left_rank  = if self.rank == 0 { self.size - 1 } else { self.rank - 1 };
        let right_rank = if self.rank == self.size - 1 { 0 } else { self.rank + 1 };
        let left_proc  = world.process_at_rank(left_rank  as i32);
        let right_proc = world.process_at_rank(right_rank as i32);

        // Left ghost: rank r's CIC deposit into cell (i_lo − 1) mod n.
        let left_ghost = (i_lo + n - 1) % n;
        let send_left  = rho[left_ghost * n2..(left_ghost + 1) * n2].to_vec();

        // Right ghost: rank r's CIC deposit into cell i_hi mod n.
        let right_ghost = i_hi % n;
        let send_right  = rho[right_ghost * n2..(right_ghost + 1) * n2].to_vec();

        // Exchange 1: send left spillover to left neighbour;
        //             receive right neighbour's left spillover → add to our right boundary cell.
        // Right neighbour's left spillover lands in cell (i_lo_{r+1} − 1) = i_hi − 1.
        let mut recv_from_right = vec![0.0_f64; n2];
        p2p::send_receive_into(&send_left[..], &left_proc, &mut recv_from_right[..], &right_proc);
        let right_boundary = i_hi - 1; // always valid: i_hi ≥ 1
        for j in 0..n2 {
            rho[right_boundary * n2 + j] += recv_from_right[j];
        }

        // Exchange 2: send right spillover to right neighbour;
        //             receive left neighbour's right spillover → add to our left boundary cell.
        // Left neighbour's right spillover lands in cell i_hi_{r-1} = i_lo.
        let mut recv_from_left = vec![0.0_f64; n2];
        p2p::send_receive_into(&send_right[..], &right_proc, &mut recv_from_left[..], &left_proc);
        for j in 0..n2 {
            rho[i_lo * n2 + j] += recv_from_left[j];
        }
    }

    /// No-op stub: single rank already holds complete density after CIC.
    #[cfg(not(feature = "mpi"))]
    pub fn reduce_ghost_density(&self, _rho: &mut [Real], _n: usize) {}

    /// Allreduce a vector element-wise (sum).
    ///
    /// Retained for use cases other than density (e.g. diagnostic sums).
    #[cfg(feature = "mpi")]
    pub fn allreduce_sum_vec(&self, local: &[Real]) -> Vec<Real> {
        use ::mpi::collective::{CommunicatorCollectives, SystemOperation};
        let world = self._universe.world();
        let mut result = vec![0.0_f64; local.len()];
        world.all_reduce_into(local, &mut result[..], SystemOperation::sum());
        result
    }

    #[cfg(not(feature = "mpi"))]
    pub fn allreduce_sum_vec(&self, local: &[Real]) -> Vec<Real> {
        local.to_vec()
    }

    // ── Allgather slabs ──────────────────────────────────────────────────────

    /// Gather local i-slab data from all ranks into a full N^3 array.
    ///
    /// Each rank contributes its `slab_count(n) * n * n` slice.  The result
    /// is a full `n * n * n` vector where rank r's data sits at offset
    /// `slab_start(n) * n * n`.  Uses `MPI_Allgatherv` because slab sizes
    /// may differ when `n` is not evenly divisible by the number of ranks.
    #[cfg(feature = "mpi")]
    pub fn allgather_slabs(&self, local: &[Real], n: usize) -> Vec<Real> {
        use ::mpi::collective::CommunicatorCollectives;
        use ::mpi::datatype::PartitionMut;

        let universe = unsafe {
            &*(&self._universe as *const ::mpi::environment::Universe)
        };
        let world = universe.world();

        let n_local = self.slab_count(n) * n * n;
        debug_assert_eq!(local.len(), n_local);

        // Build counts and displacements for every rank
        let mut recv_counts = vec![0i32; self.size];
        let mut recv_displs = vec![0i32; self.size];
        for r in 0..self.size {
            let r_start = r * n / self.size;
            let r_end = (r + 1) * n / self.size;
            let r_count = (r_end - r_start) * n * n;
            recv_counts[r] = r_count as i32;
            recv_displs[r] = (r_start * n * n) as i32;
        }

        let mut global = vec![0.0_f64; n * n * n];
        let mut partition = PartitionMut::new(
            &mut global[..],
            &recv_counts[..],
            &recv_displs[..],
        );
        world.all_gather_varcount_into(local, &mut partition);
        global
    }

    /// Non-MPI stub: local data is already the full grid.
    #[cfg(not(feature = "mpi"))]
    pub fn allgather_slabs(&self, local: &[Real], _n: usize) -> Vec<Real> {
        local.to_vec()
    }

    // ── Distributed transpose (for FFT Poisson solver) ──────────────────────

    /// Transpose from i-slab layout to j-pencil layout.
    ///
    /// Input: each rank r owns i-slabs `[r*N/size .. (r+1)*N/size)`, full j
    /// and k extents.  `local_data` has shape `(n_local_i, n, n)` in C-order
    /// (i slowest, k fastest).
    ///
    /// Output: each rank r owns j-slabs `[r*N/size .. (r+1)*N/size)`, full i
    /// and k extents.  Returned vec has shape `(n_local_j, n, n)`.
    ///
    /// This is the communication step needed before the i-axis FFT pass so
    /// that each rank holds contiguous i-columns.
    ///
    /// `n` is the global grid size (cubic grid assumed: same in all three dims).
    #[cfg(feature = "mpi")]
    pub fn transpose_i_to_j(&self, local_data: &[Real], n: usize) -> Vec<Real> {
        let universe = unsafe {
            &*(&self._universe as *const ::mpi::environment::Universe)
        };
        let world = universe.world();

        let n_local_i = self.slab_count(n);
        debug_assert_eq!(local_data.len(), n_local_i * n * n);

        // Pack send buffers: for each target rank s, extract the sub-block
        // A[i_lo..i_hi][j_s_lo..j_s_hi][0..n] from our local data.
        let mut send_buf = vec![0.0; n_local_i * n * n];
        let mut send_counts = vec![0i32; self.size];
        let mut send_displs = vec![0i32; self.size];

        let mut offset = 0usize;
        for s in 0..self.size {
            let j_lo = s * n / self.size;
            let j_hi = (s + 1) * n / self.size;
            let nj_s = j_hi - j_lo;
            let block_size = n_local_i * nj_s * n;

            send_counts[s] = block_size as i32;
            send_displs[s] = offset as i32;

            // Pack: iterate over local i, then j in [j_lo..j_hi], then all k
            for li in 0..n_local_i {
                for j in j_lo..j_hi {
                    let src_start = li * n * n + j * n;
                    let dst_start = offset + li * nj_s * n + (j - j_lo) * n;
                    send_buf[dst_start..dst_start + n]
                        .copy_from_slice(&local_data[src_start..src_start + n]);
                }
            }
            offset += block_size;
        }

        // Compute recv counts: from each sender s, we receive
        // n_local_i_of_s * n_local_j * n elements.
        let n_local_j = self.slab_count(n); // same partitioning scheme
        let mut recv_counts = vec![0i32; self.size];
        let mut recv_displs = vec![0i32; self.size];
        let mut roffset = 0usize;
        for s in 0..self.size {
            let ni_s = (s + 1) * n / self.size - s * n / self.size;
            let block_size = ni_s * n_local_j * n;
            recv_counts[s] = block_size as i32;
            recv_displs[s] = roffset as i32;
            roffset += block_size;
        }

        let mut recv_buf = vec![0.0; roffset];

        // All-to-all-v exchange
        {
            use ::mpi::datatype::{Partition, PartitionMut};
            use ::mpi::collective::CommunicatorCollectives;

            let send_partition = Partition::new(&send_buf[..], &send_counts[..], &send_displs[..]);
            let mut recv_partition = PartitionMut::new(
                &mut recv_buf[..],
                &recv_counts[..],
                &recv_displs[..],
            );
            world.all_to_all_varcount_into(&send_partition, &mut recv_partition);
        }

        // Unpack recv buffers into j-pencil layout: output[lj][i][k]
        // where lj is local j index, i is global i, k is global k.
        let mut output = vec![0.0; n_local_j * n * n];

        for s in 0..self.size {
            let i_lo_s = s * n / self.size;
            let ni_s = (s + 1) * n / self.size - i_lo_s;
            let base = recv_displs[s] as usize;

            // Received block from rank s has shape (ni_s, n_local_j, n) in C-order
            for si in 0..ni_s {
                let gi = i_lo_s + si; // global i index
                for lj in 0..n_local_j {
                    let src_start = base + si * n_local_j * n + lj * n;
                    let dst_start = lj * n * n + gi * n;
                    output[dst_start..dst_start + n]
                        .copy_from_slice(&recv_buf[src_start..src_start + n]);
                }
            }
        }

        output
    }

    /// No-op transpose for non-MPI builds (single rank owns entire grid).
    #[cfg(not(feature = "mpi"))]
    pub fn transpose_i_to_j(&self, local_data: &[Real], _n: usize) -> Vec<Real> {
        local_data.to_vec()
    }

    /// Transpose from j-pencil layout back to i-slab layout.
    ///
    /// Inverse of `transpose_i_to_j`.  Input has shape `(n_local_j, n, n)`,
    /// output has shape `(n_local_i, n, n)`.
    #[cfg(feature = "mpi")]
    pub fn transpose_j_to_i(&self, local_data: &[Real], n: usize) -> Vec<Real> {
        let universe = unsafe {
            &*(&self._universe as *const ::mpi::environment::Universe)
        };
        let world = universe.world();

        let n_local_j = self.slab_count(n);
        debug_assert_eq!(local_data.len(), n_local_j * n * n);

        // Pack send buffers: for each target rank s, extract sub-block
        // B[lj][i_s_lo..i_s_hi][0..n] from local j-pencil data.
        let mut send_buf = vec![0.0; n_local_j * n * n];
        let mut send_counts = vec![0i32; self.size];
        let mut send_displs = vec![0i32; self.size];

        let mut offset = 0usize;
        for s in 0..self.size {
            let i_lo = s * n / self.size;
            let i_hi = (s + 1) * n / self.size;
            let ni_s = i_hi - i_lo;
            let block_size = n_local_j * ni_s * n;

            send_counts[s] = block_size as i32;
            send_displs[s] = offset as i32;

            // Pack: iterate over local j, then i in [i_lo..i_hi], then all k
            for lj in 0..n_local_j {
                for i in i_lo..i_hi {
                    let src_start = lj * n * n + i * n;
                    let dst_start = offset + lj * ni_s * n + (i - i_lo) * n;
                    send_buf[dst_start..dst_start + n]
                        .copy_from_slice(&local_data[src_start..src_start + n]);
                }
            }
            offset += block_size;
        }

        // Recv counts: from each sender s, we get n_local_j_of_s * n_local_i * n
        let n_local_i = self.slab_count(n);
        let mut recv_counts = vec![0i32; self.size];
        let mut recv_displs = vec![0i32; self.size];
        let mut roffset = 0usize;
        for s in 0..self.size {
            let nj_s = (s + 1) * n / self.size - s * n / self.size;
            let block_size = nj_s * n_local_i * n;
            recv_counts[s] = block_size as i32;
            recv_displs[s] = roffset as i32;
            roffset += block_size;
        }

        let mut recv_buf = vec![0.0; roffset];

        {
            use ::mpi::datatype::{Partition, PartitionMut};
            use ::mpi::collective::CommunicatorCollectives;

            let send_partition = Partition::new(&send_buf[..], &send_counts[..], &send_displs[..]);
            let mut recv_partition = PartitionMut::new(
                &mut recv_buf[..],
                &recv_counts[..],
                &recv_displs[..],
            );
            world.all_to_all_varcount_into(&send_partition, &mut recv_partition);
        }

        // Unpack: output has shape (n_local_i, n, n) in C-order (i slowest).
        // From rank s we received a block of shape (nj_s, n_local_i, n).
        let mut output = vec![0.0; n_local_i * n * n];

        for s in 0..self.size {
            let j_lo_s = s * n / self.size;
            let nj_s = (s + 1) * n / self.size - j_lo_s;
            let base = recv_displs[s] as usize;

            for sj in 0..nj_s {
                let gj = j_lo_s + sj; // global j index
                for li in 0..n_local_i {
                    let src_start = base + sj * n_local_i * n + li * n;
                    let dst_start = li * n * n + gj * n;
                    output[dst_start..dst_start + n]
                        .copy_from_slice(&recv_buf[src_start..src_start + n]);
                }
            }
        }

        output
    }

    /// No-op inverse transpose for non-MPI builds.
    #[cfg(not(feature = "mpi"))]
    pub fn transpose_j_to_i(&self, local_data: &[Real], _n: usize) -> Vec<Real> {
        local_data.to_vec()
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
    fn non_mpi_gather_particles_returns_copy() {
        let ctx = MpiContext::init();
        let pos = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let vel = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let result = ctx.gather_particles_to_root(&pos, &vel);
        assert!(result.is_some());
        let (rpos, rvel) = result.unwrap();
        assert_eq!(rpos, pos);
        assert_eq!(rvel, vel);
    }

    #[test]
    fn non_mpi_allreduce_vec_is_identity() {
        let ctx = MpiContext::init();
        let data = vec![1.0, 2.0, 3.0];
        let result = ctx.allreduce_sum_vec(&data);
        assert_eq!(result, data);
    }

    // ── Transpose tests ─────────────────────────────────────────────────────

    #[test]
    fn transpose_i_to_j_single_rank_is_identity() {
        let ctx = MpiContext::init();
        let n = 8;
        let data: Vec<Real> = (0..n * n * n).map(|i| i as Real).collect();
        let result = ctx.transpose_i_to_j(&data, n);
        assert_eq!(result.len(), data.len());
        assert_eq!(result, data);
    }

    #[test]
    fn transpose_j_to_i_single_rank_is_identity() {
        let ctx = MpiContext::init();
        let n = 8;
        let data: Vec<Real> = (0..n * n * n).map(|i| i as Real).collect();
        let result = ctx.transpose_j_to_i(&data, n);
        assert_eq!(result.len(), data.len());
        assert_eq!(result, data);
    }

    #[test]
    fn transpose_round_trip_single_rank() {
        let ctx = MpiContext::init();
        let n = 8;
        let data: Vec<Real> = (0..n * n * n)
            .map(|i| (i as Real).sin())
            .collect();
        let transposed = ctx.transpose_i_to_j(&data, n);
        let recovered = ctx.transpose_j_to_i(&transposed, n);
        assert_eq!(recovered.len(), data.len());
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "round-trip mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn transpose_preserves_length() {
        let ctx = MpiContext::init();
        for n in [4, 8, 16] {
            let data: Vec<Real> = vec![1.0; n * n * n];
            assert_eq!(ctx.transpose_i_to_j(&data, n).len(), n * n * n);
            assert_eq!(ctx.transpose_j_to_i(&data, n).len(), n * n * n);
        }
    }

    // ── Ghost density exchange tests ─────────────────────────────────────────

    /// Single-rank: reduce_ghost_density must not change any values (no-op).
    #[test]
    fn reduce_ghost_density_single_rank_noop() {
        let ctx = MpiContext::init();
        let n = 4;
        let data: Vec<Real> = (0..n * n * n).map(|i| (i + 1) as Real).collect();
        let mut rho = data.clone();
        ctx.reduce_ghost_density(&mut rho, n);
        assert_eq!(rho, data, "single-rank ghost exchange must not change rho");
    }

    /// Single-rank: density sum is conserved across the exchange.
    #[test]
    fn reduce_ghost_density_preserves_sum() {
        let ctx = MpiContext::init();
        let n = 8;
        let mut rho: Vec<Real> = (0..n * n * n).map(|i| (i as Real).sin().abs()).collect();
        let sum_before: Real = rho.iter().sum();
        ctx.reduce_ghost_density(&mut rho, n);
        let sum_after: Real = rho.iter().sum();
        assert!(
            (sum_after - sum_before).abs() < 1e-10 * sum_before.abs(),
            "single-rank exchange must not change total density: {sum_before} → {sum_after}"
        );
    }
}
