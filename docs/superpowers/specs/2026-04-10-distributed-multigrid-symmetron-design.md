# Distributed Multigrid — Symmetron Scalar Field Solver

**Date:** 2026-04-10
**Status:** Approved, pending implementation

---

## Goal

Distribute the Symmetron nonlinear multigrid solver across MPI ranks so that each
rank works only on its i-slab of the scalar field grid, eliminating O(N³) per-rank
memory and compute during the iterative solve. f(R) distribution follows as a
separate task.

---

## Context

Current state:
- `solve_symmetron(rho: &[Real], ...)` takes a full N³ grid, runs serially on every rank.
- Every rank holds and computes the complete chi grid — 2 GB at 512³ per rank.
- The density `rho` is already allreduced across ranks before the scalar field call.
- The Poisson solver already uses a distributed FFT-PM approach for comparison.

---

## Approach: Full distributed V-cycle with coarsest-level gather

Each rank owns an i-slab of chi and rho throughout all V-cycle levels. Ghost exchange
supplies the one boundary layer needed for the 7-point Laplacian stencil. Coarsening
continues until the local slab count drops below `min_n_local = 4`, at which point all
ranks gather to a full coarse grid, solve serially (cheaply), and extract their own
slab back.

---

## API

New function added to `src/scalar_field/symmetron.rs`, gated by `#[cfg(feature = "mpi_parallel")]`:

```rust
pub fn solve_symmetron_mpi(
    rho_slab: &[Real],   // local slab: n_local_i × N × N, no ghosts
    a: Real,
    dx: Real,
    n: usize,            // global grid size (N, same in all dims)
    p: &SymmetronParams,
    mpi: &MpiContext,
) -> Vec<Real>           // returns local slab: n_local_i × N × N, no ghosts
```

The existing `solve_symmetron` is unchanged. All existing tests continue to pass.

### Integration in `simulation.rs`

```rust
#[cfg(feature = "mpi_parallel")]
{
    let n_local_i = self.mpi_ctx.slab_count(n);
    let i0 = self.mpi_ctx.slab_start(n);
    let rho_slab = &self.scratch.rho[i0 * n * n..(i0 + n_local_i) * n * n];
    let chi_slab = solve_symmetron_mpi(rho_slab, self.a, dx, n, &p, &self.mpi_ctx);
    let chi_full = self.mpi_ctx.allgather_slabs(&chi_slab, n);
    self.chi.copy_from_slice(&chi_full);
}
#[cfg(not(feature = "mpi_parallel"))]
{
    let chi_new = solve_symmetron(&self.scratch.rho, self.a, dx, n, &p);
    self.chi.copy_from_slice(&chi_new);
}
```

After the solve, `fifth_force_grid` and CIC interpolation operate on the full chi grid
as before — those stay serial for now.

---

## Local slab layout

Each rank allocates chi and rho as padded slabs of shape `(n_local_i + 2) × N × N`
(1 ghost layer each side of the i-axis). j and k remain full-width and use periodic
modulo wrapping unchanged.

Flat index: `li * N² + j * N + k`

- `li = 0` — left ghost (filled from left neighbour)
- `li = 1 .. n_local_i` — owned cells
- `li = n_local_i + 1` — right ghost (filled from right neighbour)

---

## Halo exchange protocol

`mpi.exchange_halos(&mut grid, n_local_i, n, 1)` (already implemented) is called:

- Before each NGS sweep
- Before computing residuals for the convergence check
- Before restriction (so boundary fine cells can read i-neighbours)

The NGS sweep reads from a clone (`snap`) taken after the halo exchange, so there are
no cross-rank data races within a sweep (Jacobi-style red-black, same as the serial
implementation).

---

## Convergence check

```
for iter in 0..p.max_iter:
    exchange_halos(chi)
    res_max_local  = max |r[li,j,k]| over owned cells li ∈ [1, n_local_i+1)
    res_max_global = mpi.allreduce_max(res_max_local)   // one allreduce per outer iter
    if res_max_global < p.epsilon: break
    v_cycle_sym_mpi(...)
```

---

## V-cycle levels and coarsening

At each level, `n_local_i` halves. We coarsen recursively until `n_local_i < 4`
(matching the serial `min_n = 4`).

**Fine and intermediate levels (n_local_i >= 4):**
1. Pre-smooth: `n_smooth` calls to `ngs_sweep_slab` (each preceded by halo exchange)
2. Exchange halos of chi; compute residual on owned cells
3. Restrict chi and rho to coarse slabs via injection (no ghost exchange needed)
4. Recurse on coarse slab
5. Exchange halos of coarse correction; prolongate to fine slab via trilinear interp
6. Add correction to chi; clamp to [−1, 1]
7. Post-smooth: `n_smooth` calls to `ngs_sweep_slab`

**Coarsest level (n_local_i < 4):**
1. `allgather_slabs(chi_coarse_slab, n_coarse)` → full coarse grid on every rank
2. Every rank runs the existing serial `ngs_sweep` (4 × n_smooth sweeps) on the full
   coarse grid — O(n_coarse³) ≤ O(8³) = 512 cells, negligible
3. Each rank extracts its own slab: `full[i_lo * nc² .. i_hi * nc²]`

No scatter needed: `allgather_slabs` already places the full grid on every rank.
All ranks produce identical coarse solutions.

---

## Restrict and prolongate (slab variants)

**`restrict_slab`**: Injection — takes cell `(2*lci, j, k)` in the fine slab for each
coarse owned cell `lci ∈ [0, n_local_ic)`. Injection reads only cells at even fine
i-indices; all of these fall within the owned fine slab (never in ghosts). No halo
exchange needed before restriction. Works the same way for both chi and rho.

**`prolongate_slab`**: Trilinear interpolation. Fine cell `li` reads coarse cells at
`floor` and `ceil` in coarse i-coordinates. The first and last owned fine cells read
coarse ghost cells (i = −0.5 wraps to the left ghost; i = n_local_ic + 0.5 wraps to
the right ghost). Therefore: exchange halos on the coarse correction slab **before**
prolongating.

Both functions are private to `symmetron.rs`, gated by `#[cfg(feature = "mpi_parallel")]`.

---

## No new files, no new MPI primitives

All new code lives in `src/scalar_field/symmetron.rs` under `#[cfg(feature = "mpi_parallel")]`.
`MpiContext` needs no new methods — `exchange_halos` and `allgather_slabs` are
sufficient.

When f(R) is distributed later, the slab-aware primitives (`ngs_sweep_slab`,
`restrict_slab`, `prolongate_slab`, `v_cycle_mpi`) will be factored into a shared
`src/scalar_field/multigrid_mpi.rs` module.

---

## New private functions (all in `symmetron.rs`, `mpi_parallel` gated)

| Function | Purpose |
|----------|---------|
| `ngs_sweep_slab` | One red-black NGS sweep on a ghost-padded local slab |
| `residual_slab` | Pointwise residual on owned cells of a local slab |
| `restrict_slab` | Full-weighting restriction from fine to coarse local slab |
| `prolongate_slab` | Linear prolongation from coarse to fine local slab |
| `v_cycle_sym_mpi` | One distributed V-cycle; recurses on coarser slabs |

---

## What is NOT in scope

- Distributed `fifth_force_grid` / CIC interpolation (chi is allgathered before those)
- f(R) distributed multigrid (follow-on)
- Load balancing for non-power-of-two rank counts (n must be divisible by size at all levels)
- AMR integration
