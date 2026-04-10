# Distributed Multigrid — Symmetron Scalar Field Solver

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute the Symmetron nonlinear multigrid V-cycle across MPI ranks so each rank solves only its i-slab, eliminating O(N³) per-rank memory and compute during the iterative solve.

**Architecture:** Each rank owns a ghost-padded i-slab of chi and rho. `exchange_halos` fills 1-cell ghost zones before NGS sweeps and prolongation. The V-cycle recurses on locally halved slabs until `n_local_i < 4`, at which point all ranks call `allgather_slabs`, run the existing serial NGS, and each extract their own slab back. Restriction is injection (no ghosts needed); prolongation is trilinear (needs coarse-correction halos before prolongating). The public `solve_symmetron_mpi` assembles chi via `allgather_slabs` for the existing `fifth_force_grid` caller.

**Tech Stack:** Rust, `mpi` crate (feature = `mpi_parallel`), existing `exchange_halos` / `allgather_slabs` in `MpiContext`.

---

## File map

| File | Change |
|------|--------|
| `src/scalar_field/symmetron.rs` | Add `sidx` helper, 4 private slab functions, `v_cycle_sym_mpi`, `solve_symmetron_mpi` |
| `src/simulation.rs` | Dispatch to `solve_symmetron_mpi` under `cfg(feature = "mpi_parallel")` |

No new files. No new MPI primitives.

---

## Conventions used throughout

**Ghost-padded slab** for `n_local_i` owned i-cells: shape `(n_local_i + 2) × N × N`.
- `li = 0`: left ghost (filled from left neighbour's last owned row)
- `li = 1..=n_local_i`: owned cells
- `li = n_local_i + 1`: right ghost (filled from right neighbour's first owned row)

**Plain slab** (no ghosts): shape `n_local_i × N × N`. Used for `rho` (only accessed at owned cells during NGS; injection restriction never needs i-neighbours).

**`sidx(li, j, k, n)`**: flat index in a ghost-padded slab. `li` is in range [0, n_local_i+2); `j` and `k` wrap periodically.

**Global i for local li**: `i_global = i_start + (li - 1)`. Needed only for red-black colouring.

---

## Task 1: `sidx` helper and `ngs_sweep_slab`

**Files:**
- Modify: `src/scalar_field/symmetron.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module at the bottom of `src/scalar_field/symmetron.rs`, inside `#[cfg(test)]`:

```rust
#[test]
fn ngs_sweep_slab_single_rank_matches_serial() {
    // Single-rank slab (n_local_i = n, i_start = 0) must produce the same
    // result as the serial ngs_sweep starting from the same ghost-padded chi.
    let n = 8_usize;
    let dx = 1.0_f64;
    let a = 0.8_f64;
    let p = SymmetronParams { n_smooth: 2, ..SymmetronParams::default() };
    let rho: Vec<Real> = (0..n*n*n).map(|i| 0.3 + 0.1 * (i as Real).sin()).collect();
    let chi_init: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, &p)).collect();

    // Serial: one ngs_sweep on full n³ chi
    let mut chi_serial = chi_init.clone();
    ngs_sweep(&mut chi_serial, &rho, a, dx, n, &p);

    // Slab: build ghost-padded chi (n_local_i = n, i_start = 0)
    // Ghost rows wrap periodically: left ghost = row n-1, right ghost = row 0.
    let n_local_i = n;
    let mut chi_slab = vec![0.0_f64; (n_local_i + 2) * n * n];
    // owned cells li = 1..=n
    for li in 1..=n {
        for jk in 0..n*n {
            chi_slab[li * n * n + jk] = chi_init[(li - 1) * n * n + jk];
        }
    }
    // left ghost (li=0): periodic = row n-1 of chi
    for jk in 0..n*n {
        chi_slab[jk] = chi_init[(n - 1) * n * n + jk];
    }
    // right ghost (li=n+1): periodic = row 0 of chi
    for jk in 0..n*n {
        chi_slab[(n + 1) * n * n + jk] = chi_init[jk];
    }

    ngs_sweep_slab(&mut chi_slab, &rho, a, dx, n, n_local_i, 0, &p);

    // Compare owned cells
    for li in 1..=n {
        for jk in 0..n*n {
            let got = chi_slab[li * n * n + jk];
            let exp = chi_serial[(li - 1) * n * n + jk];
            assert!(
                (got - exp).abs() < 1e-12,
                "li={li} jk={jk}: slab={got:.10} serial={exp:.10}"
            );
        }
    }
}
```

- [ ] **Step 2: Run the test to confirm it does not compile yet**

```bash
cd /home/ae/projects/privat/claude/RUSTIS
cargo test --features symmetron ngs_sweep_slab_single_rank_matches_serial 2>&1 | head -20
```

Expected: compile error — `ngs_sweep_slab`, `sidx` not found.

- [ ] **Step 3: Add `sidx` and `ngs_sweep_slab` to `src/scalar_field/symmetron.rs`**

Insert after the `dsource_dchi` function (before the `chi_equilibrium` function), around line 97:

```rust
// ── Distributed slab helpers ──────────────────────────────────────────────────

/// Flat index in a ghost-padded slab of shape `(n_local_i + 2) × N × N`.
///
/// `li` must be in [0, n_local_i + 2); `j` and `k` wrap periodically mod `n`.
#[inline]
fn sidx(li: usize, j: isize, k: isize, n: usize) -> usize {
    let j = j.rem_euclid(n as isize) as usize;
    let k = k.rem_euclid(n as isize) as usize;
    li * n * n + j * n + k
}

/// One red-black Newton-Gauss-Seidel sweep on a ghost-padded local i-slab.
///
/// `chi`      — ghost-padded slab, shape `(n_local_i + 2) × N × N`. Ghosts must
///              be current (call `exchange_halos` first in MPI mode).
/// `rho`      — plain slab, shape `n_local_i × N × N` (no ghosts needed).
/// `i_start`  — global i-index of the first owned cell (li = 1).
///              Used to keep the red-black colouring consistent across ranks.
fn ngs_sweep_slab(
    chi:       &mut Vec<Real>,
    rho:       &[Real],
    a:         Real,
    dx:        Real,
    n:         usize,
    n_local_i: usize,
    i_start:   usize,
    p:         &SymmetronParams,
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let snap    = chi.clone();

    for color in 0..2_usize {
        for li in 1..=n_local_i {
            let i_global = i_start + li - 1;
            for j in 0..n as isize {
                let k_start = ((i_global + j as usize) % 2 + color) % 2;
                let mut k   = k_start as isize;
                while k < n as isize {
                    let c     = sidx(li, j, k, n);
                    // rho index: plain slab (li-1 is 0-based owned index)
                    let rho_c = rho[(li - 1) * n * n + j as usize * n + k as usize];
                    let chi_c = snap[c];
                    let nb    = snap[sidx(li + 1, j,     k,     n)]
                              + snap[sidx(li - 1, j,     k,     n)]
                              + snap[sidx(li,     j + 1, k,     n)]
                              + snap[sidx(li,     j - 1, k,     n)]
                              + snap[sidx(li,     j,     k + 1, n)]
                              + snap[sidx(li,     j,     k - 1, n)];
                    let f     = source_chi(chi_c, rho_c, a, p);
                    let df    = dsource_dchi(chi_c, rho_c, a, p);
                    let denom = 6.0 * inv_dx2 + df;
                    if denom.abs() > 1e-30 {
                        chi[c] = ((chi_c * df + nb * inv_dx2 - f) / denom).clamp(-1.0, 1.0);
                    }
                    k += 2;
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test --features symmetron ngs_sweep_slab_single_rank_matches_serial -- --nocapture 2>&1 | tail -5
```

Expected: `test ... ok`

- [ ] **Step 5: Commit**

```bash
git add src/scalar_field/symmetron.rs
git commit -m "feat(multigrid): add sidx helper and ngs_sweep_slab for distributed NGS"
```

---

## Task 2: `residual_slab`

**Files:**
- Modify: `src/scalar_field/symmetron.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn residual_slab_matches_serial() {
    let n = 8_usize;
    let dx = 1.0_f64;
    let a  = 0.8_f64;
    let p  = SymmetronParams::default();
    let rho: Vec<Real> = (0..n*n*n).map(|i| 0.5 + 0.2 * (i as Real / 10.0).sin()).collect();
    let chi: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, &p) + 0.05).collect();

    // Serial residual on full grid
    let res_serial = residual_chi(&chi, &rho, a, dx, n, &p);

    // Build ghost-padded chi slab (n_local_i = n, periodic ghosts)
    let n_local_i = n;
    let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
    for li in 1..=n {
        for jk in 0..n*n {
            chi_slab[li * n * n + jk] = chi[(li - 1) * n * n + jk];
        }
    }
    for jk in 0..n*n { chi_slab[jk] = chi[(n - 1) * n * n + jk]; }
    for jk in 0..n*n { chi_slab[(n + 1) * n * n + jk] = chi[jk]; }

    let (res_slab, res_max) = residual_slab(&chi_slab, &rho, a, dx, n, n_local_i, &p);

    assert!(res_max > 0.0, "residual should be non-zero");
    for li in 1..=n {
        for jk in 0..n*n {
            let got = res_slab[(li - 1) * n * n + jk];
            let exp = res_serial[(li - 1) * n * n + jk];
            assert!(
                (got - exp).abs() < 1e-12,
                "li={li} jk={jk}: slab={got:.10} serial={exp:.10}"
            );
        }
    }
}
```

- [ ] **Step 2: Run to confirm compile failure**

```bash
cargo test --features symmetron residual_slab_matches_serial 2>&1 | head -5
```

- [ ] **Step 3: Add `residual_slab` to `src/scalar_field/symmetron.rs`**

Insert after `ngs_sweep_slab`:

```rust
/// Pointwise residual  r = ∇²χ − f(χ)  on owned cells of a ghost-padded slab.
///
/// Returns `(residuals, max_abs_residual)`.
/// `residuals` is a plain slab (n_local_i × N × N, no ghosts).
/// Ghosts of `chi` must be current before calling.
fn residual_slab(
    chi:       &[Real],
    rho:       &[Real],
    a:         Real,
    dx:        Real,
    n:         usize,
    n_local_i: usize,
    p:         &SymmetronParams,
) -> (Vec<Real>, Real) {
    let inv_dx2 = 1.0 / (dx * dx);
    let mut res     = vec![0.0_f64; n_local_i * n * n];
    let mut res_max = 0.0_f64;

    for li in 1..=n_local_i {
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c_pad   = sidx(li, j, k, n);
                let c_plain = (li - 1) * n * n + j as usize * n + k as usize;
                let lap = (chi[sidx(li + 1, j,     k,     n)]
                         + chi[sidx(li - 1, j,     k,     n)]
                         + chi[sidx(li,     j + 1, k,     n)]
                         + chi[sidx(li,     j - 1, k,     n)]
                         + chi[sidx(li,     j,     k + 1, n)]
                         + chi[sidx(li,     j,     k - 1, n)]
                         - 6.0 * chi[c_pad]) * inv_dx2;
                let r         = lap - source_chi(chi[c_pad], rho[c_plain], a, p);
                res[c_plain]  = r;
                res_max       = res_max.max(r.abs());
            }
        }
    }
    (res, res_max)
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test --features symmetron residual_slab_matches_serial -- --nocapture 2>&1 | tail -5
```

Expected: `test ... ok`

- [ ] **Step 5: Commit**

```bash
git add src/scalar_field/symmetron.rs
git commit -m "feat(multigrid): add residual_slab for distributed convergence check"
```

---

## Task 3: `restrict_slab`

**Files:**
- Modify: `src/scalar_field/symmetron.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn restrict_slab_matches_serial() {
    // restrict_slab with n_local_i = n must give the same result as restrict_uniform
    let n = 8_usize;
    let rho: Vec<Real> = (0..n*n*n).map(|i| 1.0 + 0.3 * (i as Real).cos()).collect();
    let chi: Vec<Real> = (0..n*n*n).map(|i| 0.5 * (i as Real / 5.0).sin()).collect();

    // Serial restriction
    let chi_c_serial = restrict_uniform(&chi, n);
    let rho_c_serial = restrict_uniform(&rho, n);

    // Build ghost-padded chi slab (n_local_i = n, periodic ghosts)
    let n_local_i = n;
    let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
    for li in 1..=n { for jk in 0..n*n { chi_slab[li*n*n+jk] = chi[(li-1)*n*n+jk]; } }
    for jk in 0..n*n { chi_slab[jk] = chi[(n-1)*n*n+jk]; }
    for jk in 0..n*n { chi_slab[(n+1)*n*n+jk] = chi[jk]; }

    let (chi_c_slab, rho_c_slab) = restrict_slab(&chi_slab, &rho, n_local_i, n);

    let nc = n / 2;
    let n_local_ic = n_local_i / 2;

    // Compare owned coarse cells (lci+1 in padded, lci in serial 0-indexed)
    for lci in 0..n_local_ic {
        for j in 0..nc {
            for k in 0..nc {
                let got_chi = chi_c_slab[(lci + 1) * nc * nc + j * nc + k];
                let exp_chi = chi_c_serial[lci * nc * nc + j * nc + k];
                assert!((got_chi - exp_chi).abs() < 1e-12,
                    "chi: lci={lci} j={j} k={k}: {got_chi:.10} vs {exp_chi:.10}");

                let got_rho = rho_c_slab[lci * nc * nc + j * nc + k];
                let exp_rho = rho_c_serial[lci * nc * nc + j * nc + k];
                assert!((got_rho - exp_rho).abs() < 1e-12,
                    "rho: lci={lci} j={j} k={k}: {got_rho:.10} vs {exp_rho:.10}");
            }
        }
    }
}
```

- [ ] **Step 2: Run to confirm compile failure**

```bash
cargo test --features symmetron restrict_slab_matches_serial 2>&1 | head -5
```

- [ ] **Step 3: Add `restrict_slab` to `src/scalar_field/symmetron.rs`**

Insert after `residual_slab`:

```rust
/// Injection restriction: coarse slab ← fine slab (pick every-other cell).
///
/// `fine`       — ghost-padded chi slab, shape `(n_local_i + 2) × N × N`.
/// `rho`        — plain rho slab, shape `n_local_i × N × N`.
///
/// Returns:
///   `chi_c` — ghost-padded coarse chi slab, shape `(n_local_ic + 2) × Nc × Nc`.
///             Owned cells filled; ghosts are zero (caller must exchange halos).
///   `rho_c` — plain coarse rho slab, shape `n_local_ic × Nc × Nc`.
///
/// Injection only reads owned fine cells, so no halo exchange is needed before
/// this call.
fn restrict_slab(
    fine:      &[Real],
    rho:       &[Real],
    n_local_i: usize,
    n:         usize,
) -> (Vec<Real>, Vec<Real>) {
    let nc        = n / 2;
    let n_local_ic = n_local_i / 2;
    let mut chi_c = vec![0.0_f64; (n_local_ic + 2) * nc * nc];
    let mut rho_c = vec![0.0_f64; n_local_ic * nc * nc];

    // Coarse owned cell lci maps to fine owned cell li = 2*lci + 1
    // (global: gc = i_start/2 + lci  →  gf = 2*gc = i_start + 2*lci  →  li = 2*lci + 1)
    for lci in 0..n_local_ic {
        let li_f = 2 * lci + 1; // fine ghost-padded li for this coarse cell
        for j in 0..nc {
            for k in 0..nc {
                chi_c[(lci + 1) * nc * nc + j * nc + k] =
                    fine[li_f * n * n + (2 * j) * n + (2 * k)];
                rho_c[lci * nc * nc + j * nc + k] =
                    rho[(li_f - 1) * n * n + (2 * j) * n + (2 * k)];
            }
        }
    }
    (chi_c, rho_c)
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test --features symmetron restrict_slab_matches_serial -- --nocapture 2>&1 | tail -5
```

Expected: `test ... ok`

- [ ] **Step 5: Commit**

```bash
git add src/scalar_field/symmetron.rs
git commit -m "feat(multigrid): add restrict_slab (injection) for distributed V-cycle"
```

---

## Task 4: `prolongate_slab`

**Files:**
- Modify: `src/scalar_field/symmetron.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn prolongate_slab_roundtrip() {
    // Restrict then prolongate a smooth field; error should be small.
    let n = 8_usize;
    let chi: Vec<Real> = (0..n*n*n).map(|c| {
        let i = (c / (n * n)) as Real;
        let j = ((c / n) % n) as Real;
        let k = (c % n) as Real;
        (i / n as Real * std::f64::consts::TAU).sin()
            * (j / n as Real * std::f64::consts::TAU).cos()
            * (k / n as Real * std::f64::consts::TAU).sin()
    }).collect();

    let n_local_i = n;
    let nc = n / 2;
    let n_local_ic = n_local_i / 2;

    // Build ghost-padded fine slab
    let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
    for li in 1..=n { for jk in 0..n*n { chi_slab[li*n*n+jk] = chi[(li-1)*n*n+jk]; } }
    for jk in 0..n*n { chi_slab[jk] = chi[(n-1)*n*n+jk]; }
    for jk in 0..n*n { chi_slab[(n+1)*n*n+jk] = chi[jk]; }

    // Restrict
    let rho_dummy = vec![1.0_f64; n * n * n];
    let (chi_c, _) = restrict_slab(&chi_slab, &rho_dummy, n_local_i, n);

    // Fill coarse ghost cells (periodic: left ghost = last owned row of coarse)
    let mut chi_c_with_ghosts = chi_c.clone();
    // left ghost (lci=0): from lci=n_local_ic (last owned of this rank = last global since single-rank)
    for jk in 0..nc*nc {
        chi_c_with_ghosts[jk] = chi_c[n_local_ic * nc * nc + jk];
    }
    // right ghost (lci=n_local_ic+1): from lci=1 (first owned)
    for jk in 0..nc*nc {
        chi_c_with_ghosts[(n_local_ic + 1) * nc * nc + jk] = chi_c[nc * nc + jk];
    }

    // Prolongate
    let fine_out = prolongate_slab(&chi_c_with_ghosts, n_local_i, n);

    // On a single rank with n=8 → nc=4, restrict+prolongate should reconstruct
    // a smooth field with max error < 0.2 (trilinear on a half-resolution field).
    let max_err = (0..n_local_i)
        .flat_map(|li| (0..n*n).map(move |jk| {
            let got = fine_out[(li + 1) * n * n + jk];
            let exp = chi[li * n * n + jk];
            (got - exp).abs()
        }))
        .fold(0.0_f64, f64::max);
    println!("prolongate_slab round-trip max error: {max_err:.4}");
    assert!(max_err < 0.3, "round-trip error too large: {max_err:.4}");
}
```

- [ ] **Step 2: Run to confirm compile failure**

```bash
cargo test --features symmetron prolongate_slab_roundtrip 2>&1 | head -5
```

- [ ] **Step 3: Add `prolongate_slab` to `src/scalar_field/symmetron.rs`**

Insert after `restrict_slab`:

```rust
/// Trilinear prolongation: fine slab ← coarse slab.
///
/// `coarse`     — ghost-padded coarse correction slab, shape `(n_local_ic+2)×Nc×Nc`.
///                **Ghosts must be filled before calling** (exchange halos of the
///                coarse correction before prolongating).
/// `n_local_i`  — owned fine i-cell count (= 2 × n_local_ic).
/// `n`          — fine global grid size.
///
/// Returns a ghost-padded fine slab `(n_local_i+2)×N×N`.
/// Owned cells (li ∈ [1, n_local_i+1)) are filled; ghosts are zero.
///
/// The coarse i-coordinate for fine local padded index `li` is:
///   cx = (li - 0.5) * 0.5 - 0.5
///
/// This formula is independent of `i_start` (global rank offset cancels out),
/// so the function requires no MPI context.  The left/right coarse ghost cells
/// are accessed at `lci_base = 0` and `lci_base = n_local_ic + 1` respectively.
fn prolongate_slab(coarse: &[Real], n_local_i: usize, n: usize) -> Vec<Real> {
    let nc        = n / 2;
    let mut fine  = vec![0.0_f64; (n_local_i + 2) * n * n];

    for li in 1..=n_local_i {
        let cx       = (li as Real - 0.5) * 0.5 - 0.5;
        let lci_base = (cx.floor() as isize + 1) as usize; // +1 for coarse ghost offset
        let tx       = cx - cx.floor();

        for j in 0..n as isize {
            let cy = (j as Real + 0.5) * 0.5 - 0.5;
            let cj = cy.floor() as isize;
            let ty = cy - cy.floor();

            for k in 0..n as isize {
                let cz = (k as Real + 0.5) * 0.5 - 0.5;
                let ck = cz.floor() as isize;
                let tz = cz - cz.floor();

                let mut val = 0.0_f64;
                for (dli, wx) in [(0_usize, 1.0 - tx), (1, tx)] {
                    for (dcj, wy) in [(0_isize, 1.0 - ty), (1, ty)] {
                        for (dck, wz) in [(0_isize, 1.0 - tz), (1, tz)] {
                            let jc = (cj + dcj).rem_euclid(nc as isize) as usize;
                            let kc = (ck + dck).rem_euclid(nc as isize) as usize;
                            val += wx * wy * wz
                                * coarse[(lci_base + dli) * nc * nc + jc * nc + kc];
                        }
                    }
                }
                fine[sidx(li, j, k, n)] = val;
            }
        }
    }
    fine
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test --features symmetron prolongate_slab_roundtrip -- --nocapture 2>&1 | tail -5
```

Expected: `test ... ok` and round-trip error printed < 0.3.

- [ ] **Step 5: Commit**

```bash
git add src/scalar_field/symmetron.rs
git commit -m "feat(multigrid): add prolongate_slab (trilinear) for distributed V-cycle"
```

---

## Task 5: `v_cycle_sym_mpi`

**Files:**
- Modify: `src/scalar_field/symmetron.rs`

This function requires `MpiContext` and is gated by `#[cfg(feature = "mpi_parallel")]`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module, gated by `#[cfg(feature = "mpi_parallel")]`:

```rust
#[cfg(feature = "mpi_parallel")]
#[test]
fn v_cycle_sym_mpi_reduces_residual() {
    use crate::mpi::MpiContext;
    let mpi = MpiContext::init();

    let n = 16_usize;
    let dx = 1.0_f64;
    let a  = 0.8_f64;
    let p  = SymmetronParams { n_smooth: 2, epsilon: 1e-8, max_iter: 200, ..SymmetronParams::default() };
    let rho: Vec<Real> = (0..n*n*n).map(|i| 0.5 + 0.2 * (i as Real).cos()).collect();
    let chi_eq: Vec<Real> = rho.iter().map(|&r| chi_equilibrium(r, a, &p)).collect();

    // Perturb slightly from equilibrium
    let mut chi_init = chi_eq.clone();
    chi_init[n*n/2] += 0.3;

    // n_local_i = n for single-rank test
    let n_local_i = n;
    let i_start   = 0;

    // Build ghost-padded chi slab
    let mut chi_slab = vec![0.0_f64; (n + 2) * n * n];
    for li in 1..=n { for jk in 0..n*n { chi_slab[li*n*n+jk] = chi_init[(li-1)*n*n+jk]; } }
    for jk in 0..n*n { chi_slab[jk] = chi_init[(n-1)*n*n+jk]; }
    for jk in 0..n*n { chi_slab[(n+1)*n*n+jk] = chi_init[jk]; }

    mpi.exchange_halos(&mut chi_slab, n_local_i, n, 1);
    let (_, r0) = residual_slab(&chi_slab, &rho, a, dx, n, n_local_i, &p);

    v_cycle_sym_mpi(&mut chi_slab, &rho, a, dx, n, n_local_i, i_start, 2, 4, &p, &mpi);

    mpi.exchange_halos(&mut chi_slab, n_local_i, n, 1);
    let (_, r1) = residual_slab(&chi_slab, &rho, a, dx, n, n_local_i, &p);

    println!("v_cycle_sym_mpi: residual {r0:.3e} → {r1:.3e}");
    assert!(r1 < r0, "V-cycle did not reduce residual: {r0:.3e} → {r1:.3e}");
}
```

- [ ] **Step 2: Run to confirm compile failure**

```bash
cargo test --features symmetron,mpi_parallel v_cycle_sym_mpi_reduces_residual 2>&1 | head -10
```

- [ ] **Step 3: Add `v_cycle_sym_mpi` to `src/scalar_field/symmetron.rs`**

Add `use crate::mpi::MpiContext;` at the top of the file (inside the `#[cfg(feature = "mpi_parallel")]` block — use a cfg'd use statement, or add it inside the function). Then add the function:

```rust
#[cfg(feature = "mpi_parallel")]
fn v_cycle_sym_mpi(
    chi:           &mut Vec<Real>,
    rho:           &[Real],
    a:             Real,
    dx:            Real,
    n:             usize,
    n_local_i:     usize,
    i_start:       usize,
    n_smooth:      usize,
    min_n_local:   usize,
    p:             &SymmetronParams,
    mpi:           &crate::mpi::MpiContext,
) {
    // ── Pre-smooth ──────────────────────────────────────────────────────────
    for _ in 0..n_smooth {
        mpi.exchange_halos(chi, n_local_i, n, 1);
        ngs_sweep_slab(chi, rho, a, dx, n, n_local_i, i_start, p);
    }

    // ── Coarsest level: allgather → serial solve → extract slab ────────────
    if n_local_i < min_n_local || n <= 4 {
        // Extract owned chi (no ghosts) for allgather
        let local: Vec<Real> = (0..n_local_i)
            .flat_map(|li| {
                let li_p = li + 1;
                (0..n * n).map(move |jk| chi[li_p * n * n + jk])
            })
            .collect();
        let chi_full = mpi.allgather_slabs(&local, n);
        let rho_full = mpi.allgather_slabs(rho, n);

        // All ranks run identical serial NGS on the full coarse grid
        let mut chi_coarse = chi_full;
        for _ in 0..4 * n_smooth {
            ngs_sweep(&mut chi_coarse, &rho_full, a, dx, n, p);
        }

        // Copy own slab back into chi (owned cells only)
        for li in 0..n_local_i {
            let li_p   = li + 1;
            let g_base = (i_start + li) * n * n;
            chi[li_p * n * n..li_p * n * n + n * n]
                .copy_from_slice(&chi_coarse[g_base..g_base + n * n]);
        }

        // Post-smooth after coarse solve
        for _ in 0..n_smooth {
            mpi.exchange_halos(chi, n_local_i, n, 1);
            ngs_sweep_slab(chi, rho, a, dx, n, n_local_i, i_start, p);
        }
        return;
    }

    // ── Restrict ────────────────────────────────────────────────────────────
    let (chi_c_save, rho_c) = restrict_slab(chi, rho, n_local_i, n);
    let mut chi_c = chi_c_save.clone();

    let nc          = n / 2;
    let n_local_ic  = n_local_i / 2;
    let i_start_c   = i_start / 2;

    // ── Recurse on coarse slab ──────────────────────────────────────────────
    v_cycle_sym_mpi(
        &mut chi_c, &rho_c,
        a, dx * 2.0, nc,
        n_local_ic, i_start_c,
        n_smooth, min_n_local, p, mpi,
    );

    // ── Correction: delta = chi_c_new − chi_c_save ──────────────────────────
    let mut delta_c: Vec<Real> = chi_c.iter()
        .zip(chi_c_save.iter())
        .map(|(new, old)| new - old)
        .collect();

    // Ghosts of delta_c must be current before trilinear prolongation
    mpi.exchange_halos(&mut delta_c, n_local_ic, nc, 1);

    // ── Prolongate correction to fine, apply, clamp ─────────────────────────
    let corr_fine = prolongate_slab(&delta_c, n_local_i, n);
    for li in 1..=n_local_i {
        for j in 0..n as isize {
            for k in 0..n as isize {
                let c  = sidx(li, j, k, n);
                chi[c] = (chi[c] + corr_fine[c]).clamp(-1.0, 1.0);
            }
        }
    }

    // ── Post-smooth ─────────────────────────────────────────────────────────
    for _ in 0..n_smooth {
        mpi.exchange_halos(chi, n_local_i, n, 1);
        ngs_sweep_slab(chi, rho, a, dx, n, n_local_i, i_start, p);
    }
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test --features symmetron,mpi_parallel v_cycle_sym_mpi_reduces_residual -- --nocapture 2>&1 | tail -5
```

Expected: `test ... ok`, residual decreasing printed.

- [ ] **Step 5: Commit**

```bash
git add src/scalar_field/symmetron.rs
git commit -m "feat(multigrid): add v_cycle_sym_mpi with coarsest-level allgather strategy"
```

---

## Task 6: `solve_symmetron_mpi`

**Files:**
- Modify: `src/scalar_field/symmetron.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module, gated by `#[cfg(feature = "mpi_parallel")]`:

```rust
#[cfg(feature = "mpi_parallel")]
#[test]
fn solve_symmetron_mpi_matches_serial() {
    use crate::mpi::MpiContext;
    let mpi = MpiContext::init();

    let n   = 16_usize;
    let l   = 16.0_f64;
    let dx  = l / n as Real;
    let a   = 0.8_f64;
    let p   = SymmetronParams { epsilon: 1e-6, max_iter: 200, ..SymmetronParams::default() };
    let rho: Vec<Real> = (0..n*n*n).map(|i| 0.5 + 0.3 * (i as Real / 7.0).sin().abs()).collect();

    // Serial solve on full grid
    let chi_serial = solve_symmetron(&rho, a, dx, n, &p);

    // MPI solve: single rank owns full slab
    let n_local_i = mpi.slab_count(n);
    let i0        = mpi.slab_start(n);
    let rho_slab  = &rho[i0 * n * n..(i0 + n_local_i) * n * n];
    let chi_slab  = solve_symmetron_mpi(rho_slab, a, dx, n, &p, &mpi);

    // Reconstruct full chi from slab (single rank: already full)
    let chi_mpi = mpi.allgather_slabs(&chi_slab, n);

    let max_err = chi_serial.iter()
        .zip(chi_mpi.iter())
        .map(|(s, m)| (s - m).abs())
        .fold(0.0_f64, f64::max);

    println!("solve_symmetron_mpi vs serial max error: {max_err:.3e}");
    // Both converge to the same tolerance — allow 10× tolerance as slop
    assert!(max_err < 10.0 * p.epsilon,
        "MPI solve diverges from serial: max error = {max_err:.3e}");
}
```

- [ ] **Step 2: Run to confirm compile failure**

```bash
cargo test --features symmetron,mpi_parallel solve_symmetron_mpi_matches_serial 2>&1 | head -10
```

- [ ] **Step 3: Add `solve_symmetron_mpi` to `src/scalar_field/symmetron.rs`**

Add after `v_cycle_sym_mpi`:

```rust
/// Solve the quasi-static Symmetron EOM for χ on a distributed i-slab.
///
/// Each rank supplies its local slab of `rho` (shape `n_local_i × N × N`,
/// no ghost cells). The function handles ghost exchange internally.
///
/// Returns the local slab of χ (shape `n_local_i × N × N`, no ghosts).
/// Caller should call `mpi.allgather_slabs(&chi_slab, n)` to assemble the
/// full grid for `fifth_force_grid`.
#[cfg(feature = "mpi_parallel")]
pub fn solve_symmetron_mpi(
    rho_slab:  &[Real],
    a:         Real,
    dx:        Real,
    n:         usize,
    p:         &SymmetronParams,
    mpi:       &crate::mpi::MpiContext,
) -> Vec<Real> {
    let n_local_i = mpi.slab_count(n);
    let i_start   = mpi.slab_start(n);
    debug_assert_eq!(rho_slab.len(), n_local_i * n * n,
        "rho_slab length mismatch: {} vs {}", rho_slab.len(), n_local_i * n * n);

    // Initialise ghost-padded chi from equilibrium
    let mut chi = vec![0.0_f64; (n_local_i + 2) * n * n];
    for li in 1..=n_local_i {
        for j in 0..n {
            for k in 0..n {
                let c     = sidx(li, j as isize, k as isize, n);
                let rho_c = rho_slab[(li - 1) * n * n + j * n + k];
                chi[c]    = chi_equilibrium(rho_c, a, p);
            }
        }
    }

    // Outer V-cycle loop
    for iter in 0..p.max_iter {
        mpi.exchange_halos(&mut chi, n_local_i, n, 1);
        let (_, res_local) = residual_slab(&chi, rho_slab, a, dx, n, n_local_i, p);
        let res_max        = mpi.allreduce_max(res_local);
        log::trace!("Symmetron MPI iter {iter}: |r|_∞ = {res_max:.3e}");
        if res_max < p.epsilon { break; }
        v_cycle_sym_mpi(
            &mut chi, rho_slab,
            a, dx, n, n_local_i, i_start,
            2, 4, p, mpi,
        );
    }

    // Return owned cells (strip ghosts)
    let mut result = vec![0.0_f64; n_local_i * n * n];
    for li in 0..n_local_i {
        let src = (li + 1) * n * n;
        let dst = li * n * n;
        result[dst..dst + n * n].copy_from_slice(&chi[src..src + n * n]);
    }
    result
}
```

- [ ] **Step 4: Run the test**

```bash
cargo test --features symmetron,mpi_parallel solve_symmetron_mpi_matches_serial -- --nocapture 2>&1 | tail -5
```

Expected: `test ... ok`, max error printed < 10× epsilon.

- [ ] **Step 5: Confirm all existing symmetron tests still pass**

```bash
cargo test --features symmetron 2>&1 | tail -10
```

Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/scalar_field/symmetron.rs
git commit -m "feat(multigrid): add solve_symmetron_mpi — distributed V-cycle outer loop"
```

---

## Task 7: Wire into `simulation.rs`

**Files:**
- Modify: `src/simulation.rs`

- [ ] **Step 1: Locate the symmetron dispatch block**

Open `src/simulation.rs` and find the `solve_scalar_field` method. The relevant block starts with:

```rust
#[cfg(feature = "symmetron")]
if sf.a_ssb.is_some() {
```

around line 344. Read the full block to understand what needs to change.

- [ ] **Step 2: Replace the scalar field dispatch with MPI-aware version**

Find this block in `solve_scalar_field`:

```rust
#[cfg(feature = "symmetron")]
if sf.a_ssb.is_some() {
    let p = SymmetronParams {
        a_ssb: sf.a_ssb.unwrap(),
        lambda_0: sf.lambda_0.unwrap_or(1.0),
        beta_0: sf.beta_0.unwrap_or(1.0),
        non_static: sf.non_static.unwrap_or(false),
        epsilon: sf.epsilon_mg.unwrap_or(1e-5),
        max_iter: 150,
        n_smooth: 2,
    };
    let chi_new = solve_symmetron(&self.scratch.rho, self.a, dx, n, &p);
    self.chi.copy_from_slice(&chi_new);
```

Replace with:

```rust
#[cfg(feature = "symmetron")]
if sf.a_ssb.is_some() {
    let p = SymmetronParams {
        a_ssb: sf.a_ssb.unwrap(),
        lambda_0: sf.lambda_0.unwrap_or(1.0),
        beta_0: sf.beta_0.unwrap_or(1.0),
        non_static: sf.non_static.unwrap_or(false),
        epsilon: sf.epsilon_mg.unwrap_or(1e-5),
        max_iter: 150,
        n_smooth: 2,
    };

    #[cfg(feature = "mpi_parallel")]
    let chi_new = {
        use crate::scalar_field::symmetron::solve_symmetron_mpi;
        let n_local_i = self.mpi_ctx.slab_count(n);
        let i0        = self.mpi_ctx.slab_start(n);
        let rho_slab  = &self.scratch.rho[i0 * n * n..(i0 + n_local_i) * n * n];
        let chi_slab  = solve_symmetron_mpi(rho_slab, self.a, dx, n, &p, &self.mpi_ctx);
        self.mpi_ctx.allgather_slabs(&chi_slab, n)
    };

    #[cfg(not(feature = "mpi_parallel"))]
    let chi_new = solve_symmetron(&self.scratch.rho, self.a, dx, n, &p);

    self.chi.copy_from_slice(&chi_new);
```

- [ ] **Step 3: Build with both feature combinations**

```bash
cargo build --release --features symmetron 2>&1 | tail -5
cargo build --release --features symmetron,mpi_parallel 2>&1 | tail -5
```

Expected: both succeed with no errors or warnings.

- [ ] **Step 4: Run all tests**

```bash
cargo test --features symmetron 2>&1 | tail -10
cargo test --features symmetron,mpi_parallel 2>&1 | tail -10
```

Expected: all tests pass in both configurations.

- [ ] **Step 5: Also run fofr tests to confirm no regressions**

```bash
cargo test --features fofr 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/simulation.rs
git commit -m "feat(sim): dispatch to solve_symmetron_mpi under mpi_parallel feature

Each rank solves its own i-slab; chi is allgathered before fifth force
and CIC interpolation. Serial path unchanged."
```

---

## Final verification

- [ ] **Run the full test suite with both relevant feature combinations**

```bash
cargo test --features symmetron,fofr 2>&1 | tail -10
cargo test --features symmetron,fofr,mpi_parallel 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Update ROADMAP.md**

In the `TODO (MPI)` section, mark the distributed multigrid item done:

```
- [x] Distributed multigrid for scalar field solvers (symmetron on local slabs) — done
- [ ] Distributed multigrid for f(R) (follow-on)
```

- [ ] **Commit the roadmap update**

```bash
git add ROADMAP.md
git commit -m "docs: mark distributed multigrid symmetron as done in roadmap"
```
