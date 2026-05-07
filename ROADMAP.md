# ROADMAP & PROGRESS LOG

## Goal

Reproduce the ISIS (Llinares et al. 2014, A&A 562 A78) verification figures for ΛCDM and Symmetron runs:
- 64³ particles, 64 Mpc/h box, WMAP7 cosmology, seed=42
- Power spectrum ratio P_symmetron(k) / P_ΛCDM(k) vs k at z=0

---

## Status: IN PROGRESS

### Done

#### Bug 1 — Poisson source missing H₀² (FIXED, committed 73f71f7)

The Poisson source was `(3/2) Ω_m / a × δ` with no H₀² factor. It now correctly reads:

```rust
const H0_CODE: Real = 100.0;   // km/s / (Mpc/h)
let prefac = 1.5 * H0_CODE * H0_CODE * omega_m / a;
```

#### Bug 2 — Kick had spurious factor of a (FIXED, committed 73f71f7)

The equation of motion with ṽ = a² dx/dt and τ = cosmic time gives dṽ/dt = −∇Φ (no extra a).
The kick is now simply `ṽ += g × Δτ` without a factor of a.

#### Bug 3 — IC normalization: sqrt(P/V) → sqrt(N³ P/V) (FIXED, committed d02e207)

The forward FFT of N³ iid N(0,1) real samples gives E[|buf[k]|²] = N³. The power estimator
`P̂(k) = V/N⁶ × |δ̂|²` requires E[|δ̂|²] = N⁶ P/V, so the amplitude must be:

```
δ̂[k] = buf[k] × sqrt(N³ × P(k) / V)
```

Before fix, P̂ was suppressed by a factor of N³ ≈ 262,144 for 64³ grids.
Monte Carlo with N=8 confirmed: E[P̂]/P_theory = 1.003 after fix.

---

### ~~Known remaining issue — DKD timestep error at early times~~ RESOLVED

**Update 2026-03-28:** The apparent P_sim/P_lin ≈ 3–9 excess was **not** a DKD error.

**Root cause (found):** The σ₈ normalisation integral in diagnostic Python scripts used
`∫ P k² W² d(lnk)` instead of the correct `∫ P k³ W² d(lnk)` (= `∫ P k² W² dk`).
This caused A_s to be 8.3× too small in those scripts, making P_lin appear 8.3× too low.

**Verification:** `analysis/check_snapshot_pk.py` with the correct A_s = 3.66×10⁶:
- k=0.037 h/Mpc (N=12): P_sim/P_lin = 0.96–0.97 at a=0.33 and a=1.0
- k=0.105 h/Mpc (N=114): P_sim/P_lin = 0.98–1.008 at both epochs

ΛCDM linear growth test **PASSES**. DKD is correct. The scale-factor CFL implemented
in `simulation.rs` (Δa/a ≤ 0.02) is sufficient.

---

#### Bug 4 — f(R) source missing (H₀a/c)² prefactor (FIXED, committed ca11caf)

The quasi-static comoving f(R) equation is:
```
∇²f_R = (H₀a/c)²/3 × [δR − 3Ωm δ/a³]
```
where ∇ is w.r.t. comoving x in Mpc/h, H₀ = 100 km/s/(Mpc/h), c = 299,792 km/s.

The missing factor (H₀/c)² ≈ 1.11×10⁻⁷ inflated the effective Compton wavelength by
c/H₀ ≈ 3000. F5 (λ_C ≈ 4 Mpc/h physical) was treated as λ_C ≈ 0.0025 Mpc/h — completely
sub-grid → total chameleon screening → ΔP/P ≈ 0.

Also fixed: factor of 3 in the matter coupling (from 8πGρ̄_m = 3H₀²Ωm/a³).

#### Bug 5 — f(R) fifth force missing c² multiplier (FIXED, committed ca11caf)

`fofr_fifth_force` returns `(1/2)∇f_R` with f_R dimensionless and ∇ in (Mpc/h)⁻¹.
Physical fifth force is `a₅ = (c²/2)∇f_R` in (km/s)²/(Mpc/h) — matching Poisson units.
Applied multiplier c² = (299,792)² in `solve_scalar_field` in `simulation.rs`.

---

### In Progress

#### Winther 2015 validation suite (64³, 250 Mpc/h box)

ΛCDM and f(R) F5 confirmed. F6 rerun in progress.

- ΛCDM reference: `output_w15_lcdm_test/` — 367 steps, growth test ALL PASS
- Symmetron: `output_w15_symmetron_test/` — ΔP/P = 0 (dx=3.9 Mpc/h >> λ₀=1 Mpc/h, unresolved)
- f(R) F5: `output_w15_f5_test/` — ΔP/P up to 8% at k=0.08 h/Mpc, z=0 (CONFIRMED SIGNAL)
- f(R) F6: `output_w15_f6_test/` — ΔP/P = 0 (λ_C≈3.4 Mpc/h < dx=3.9 Mpc/h, sub-grid, expected)

**f(R) solver bugs fixed (commit ade6903, 2026-03-29):**

Two bugs caused the solver to exit at iteration 0 doing no work:

1. **epsilon_mg = 1e-5 too loose.** Actual f(R) residuals are O(10⁻⁸). The initial residual
   ~9×10⁻⁷ satisfied epsilon before any V-cycles ran. Fixed: default 1e-5 → 1e-10.

2. **Wrong epoch initialisation.** Used z=0 background `fofr_of_r(r0, r0)` at all epochs.
   At a=0.33 the correct background is 10× smaller. Fixed: `fofr_of_r(rbar_a, r0)`.

**Compton wavelength (correct formula):**

m² = (h0·a/c)² × R̄(a) / (9·(n+1)·|f̄_R(a)|),  λ_C = 1/√m² [Mpc/h]

At a=1: F5 → λ_C ≈ 10.7 Mpc/h >> dx=3.9 Mpc/h (well resolved)
        F6 → λ_C ≈ 3.4 Mpc/h < dx=3.9 Mpc/h (sub-grid → ΔP/P ≈ 0 expected)

**F5 ΔP/P results (fixed solver, seed=42, 64³ 250 Mpc/h):**

| a | z | ΔP/P (k≈0.037) | ΔP/P (k≈0.081) |
|---|---|---|---|
| 0.332 | 2.01 | +0.07% | +0.39% |
| 0.503 | 0.99 | +0.49% | +2.23% |
| 0.672 | 0.49 | +1.16% | +4.65% |
| 0.821 | 0.22 | +1.69% | +6.44% |
| 1.001 | 0.00 | +2.18% | +8.00% |

Monotonically growing signal confirms the fifth force is active and correctly resolved.

#### Symmetron 64 Mpc/h rerun

Completed: `output_symmetron_test/` with fixed binary (seed=42). Baseline: `output_lcdm_test/`.
dx = 1 Mpc/h = λ₀ → marginal resolution → ΔP/P = 0 (expected).

---

### Symmetron unit bug investigation (2026-03-29)

#### Bug 6 — wrong c² factor applied to symmetron fifth force (FIXED, commit 2ee9898)

**Status: fixed, but sign bug remains (see Bug 7 below)**

The previous agent (commit 068f770) copied the f(R) c² multiplier verbatim to the
symmetron branch. This was wrong: the symmetron source equation has no (H₀/c)² prefactor,
so χ ~ O(1). Multiplying by c² ≈ 9×10¹⁰ (km/s)² gave F₅ ~ 10⁷ × F_N, collapsing
the timestep to ~10⁻⁷ and v_max ≈ 9c.

**Fix:** replaced c² with `(H₀_h × λ₀)² = (100 × λ₀)² (km/s)²` — the natural velocity
scale for the symmetron field. This converts (Mpc/h)⁻¹ → (km/s)²/(Mpc/h).

Unit derivation:
- `fifth_force_grid` returns `(β₀/a)∇χ` in (Mpc/h)⁻¹ (χ dimensionless, ∇ in (Mpc/h)⁻¹)
- Multiply by `v₀² = (H₀_h λ₀ km/s)²` → (km/s)²/(Mpc/h) matching Poisson output
- For λ₀=1: factor = 10⁴ (km/s)² → F₅/F_N ~ 2β₀² at full coupling ✓

**Post-fix run result (seed=42, 64³ 64 Mpc/h):**

| a | z | ΔP/P (k≈0.10) | max |ΔP/P| |
|---|---|---|---|
| 0.330 | 2.03 | 0.000 | 0.000 |
| 0.502 | 0.99 | 0.000 | 0.000 |
| 0.672 | 0.49 | −0.05% | 4.49% |
| 0.823 | 0.22 | −0.10% | 7.85% |
| 1.000 | 0.00 | −0.13% | 8.05% |

Force IS active (non-zero ΔP/P starting at a > a_SSB=0.5) ✓
But sign is NEGATIVE — symmetron shows LESS power than ΛCDM — see Bug 7.

---

#### Bug 7 — symmetron fifth force has wrong sign (REPULSIVE instead of ATTRACTIVE)

**Status: FIXED, commit below**

**Observation:** ΔP/P < 0 after the unit fix. The symmetron fifth force should ENHANCE
clustering (more attractive force → ΔP/P > 0). The negative sign means particles are
being pushed APART rather than together.

**Root cause:** In `src/scalar_field/symmetron.rs`, `fifth_force_grid()`:

```rust
let factor = -p.beta_0 / a;  // intended to recover +∇χ from grad_phi's -∇χ
grad.into_iter().map(|g| [factor * g[0], ...]).collect()
// Returns: (β₀/a) ∇χ  — points AWAY from masses (REPULSIVE!)
```

`grad_phi` returns `−∇χ`. Multiplying by `factor = −β₀/a` gives `+(β₀/a)∇χ`.
Since χ is suppressed near overdensities, `∇χ` points AWAY from matter → repulsive.

**Correct behaviour:** For an attractive fifth force (toward masses), the output must be
`−(β₀/a)∇χ`. Since `grad_phi` already computes `−∇χ`, the fix is:

```rust
// In src/scalar_field/symmetron.rs, fifth_force_grid():
let factor = p.beta_0 / a;   // no minus sign — grad_phi(-∇χ) × (+β₀/a) = -(β₀/a)∇χ
```

**Physical check:**
- Gravity: `grad_phi(&phi)` returns `−∇Φ`, points toward potential minimum = toward matter ✓
- Symmetron SHOULD work the same way: `grad_phi(&chi)` × positive factor = `−(β₀/a)∇χ`,
  points toward χ minimum = toward matter ✓

**Fix location:** `src/scalar_field/symmetron.rs`, line ~281:
  change `let factor = -p.beta_0 / a;` → `let factor = p.beta_0 / a;`

Also update the comment: "grad_phi returns −∇χ; multiply by +β₀/a to get −(β₀/a)∇χ
which points toward matter (attractive)."

**Fix applied:** `src/scalar_field/symmetron.rs` line 281 — removed leading minus from factor.
Comment updated: "grad_phi returns −∇χ; multiply by +β₀/a to get −(β₀/a)∇χ which points toward matter (attractive)."

**Post-fix run result (seed=42, 64³ 64 Mpc/h, sign bug fixed):**

| a | z | ΔP/P (k≈0.10) | max |ΔP/P| |
|---|---|---|---|
| 0.330 | 2.03 | 0.0000 | 0.000 |
| 0.502 | 0.99 | 0.0000 | 0.000 |
| 0.672 | 0.49 | +0.05% | 4.60% |
| 0.823 | 0.22 | +0.10% | 7.95% |
| 1.000 | 0.00 | +0.12% | 7.70% |

**PASS:** ΔP/P > 0 after a_SSB=0.5, monotonically growing. Force is attractive and active. ✓
(max |ΔP/P| at smaller scales limited by dx=λ₀=1 Mpc/h marginal resolution, as expected)

---

### TODO

- [x] Verify symmetron 64 Mpc/h — ΔP/P = 0 (dx=λ₀ marginal, unresolved)
- [x] ΛCDM linear growth — ALL PASS (growth ratio test, N_modes≥50, k<0.13 h/Mpc)
- [x] Fix pk_theory.py growth test: ratio method replaces absolute P_sim/P_lin
- [x] Fix f(R) solver epsilon and epoch init (ade6903) — F5 confirmed with non-zero ΔP/P
- [x] F6 rerun with fixed solver — DONE; ΔP/P = 0 (sub-grid, expected)
- [x] Fix symmetron unit factor: `(H₀_h λ₀)²` instead of c² (commit 2ee9898) — force now active
- [x] Fix symmetron sign bug: `fifth_force_grid` must return `−(β₀/a)∇χ` (attractive)
- [x] Re-run symmetron 64 Mpc/h after sign fix; verify ΔP/P > 0 — PASS
- [x] MPI support for cluster-scale runs — IMPLEMENTED (see below)
- [x] Non-static Symmetron leapfrog — velocity-Verlet for χ and χ̇ implemented in `solve_scalar_field`; uses `residual_chi` with correct Hubble damping at `a_half`; previously `non_static=true` silently fell through to quasi-static solver
- [x] 128³ validation (64 Mpc/h, dx=0.5 Mpc/h=λ₀/2) — PASS, clear symmetron signal confirmed
- [x] Hydro (Option A): domain-grid Godunov hydro (`--features hydro`) — IMPLEMENTED
  - MUSCL-Hancock Godunov solver (3D, gravity body force, CFL), HLL Riemann, MinMod limiter
  - Radiative cooling (primordial H/He), stochastic SF (Schmidt-Katz), SNe thermal feedback
  - FoF halo finder + stacked profile output; HDF5 gas snapshot writer
  - `ellewsen_lcdm.toml` / `sym_a/b/c/d.toml` param files; `run_ellewsen.sh` driver
  - Reference profile infrastructure: `analysis/reference/` CSVs + `--reference-dir` overlay in plotter
- [x] GitHub Actions CI/CD — `.github/workflows/ci.yml` (fmt + clippy + test ×3) and
  `release.yml` (builds `cosmo_sim-hydro-linux-x86_64` + `cosmo_sim-hydro-sym-linux-x86_64`
  on `v*` tags, uploads to GitHub Release)
- [ ] 256³ run (128 Mpc/h, ISIS paper resolution) — pending desktop machine
- [ ] Production 512³ runs (250 Mpc/h) for full Winther 2015 comparison (needs HPC)
- [ ] Run all 5 Ellewsen configs (ΛCDM + Sym A/B/C/D), 256³ 64 Mpc/h — Sym A done; ΛCDM/B/C/D pending
- [ ] Compare stacked halo profiles to Ellewsen et al. Figs 3–5 — digitize arXiv:1803.04197 Figs 3–5 into `analysis/reference/` CSVs

#### Hydro Option B (preferred long-term)
Gas fields embedded in AMR Cell struct with octree Godunov solver and AMR sub-cycling.
This fully matches the RAMSES architecture. Implement after Option A validates the physics.
See `docs/specs/2026-04-11-hydro-design.md`.

#### Symmetron 128³ results (64 Mpc/h, dx=0.5 Mpc/h, seed=42)

| a | z | ΔP/P (k=1 h/Mpc) | ΔP/P (k=6 h/Mpc) |
|---|---|---|---|
| 0.330 | 2.03 | 0.000 | 0.000 |
| 0.502 | 0.99 | 0.000 | 0.000 |
| 0.672 | 0.49 | +0.9% | +5.9% |
| 0.823 | 0.22 | +1.1% | +9.8% |
| 1.000 | 0.00 | +1.1% | +7.9% |

Signal activates between z=1 and z=0.5 (a_SSB=0.5 ✓), grows monotonically with k.
Reaches ~8% enhancement at k≈6 h/Mpc at z=0. Positive and attractive ✓.

---

### MPI support (2026-04-06)

**Status: IMPLEMENTED — basic framework + distributed FFT Poisson solver**

Added `src/mpi/mod.rs` with the `mpi_parallel` cargo feature (depends on `mpi` crate 0.8).

**Design decisions:**

1. **Slab decomposition along i-axis (x):** Rank r owns i-slabs [r*N/size, (r+1)*N/size).
   This matches RAMSES and keeps the fastest-varying index (k) contiguous for cache efficiency.
   C-order layout means each i-slab is a contiguous block of N*N cells.

2. **Hybrid MPI+rayon:** Each rank uses rayon for intra-node parallelism. MPI handles
   inter-node communication. Set RAYON_NUM_THREADS to control threads per rank.

3. **Current density strategy:** allreduce_sum_vec on the full density grid after CIC.
   Correct but O(N³) communication per step — the remaining bottleneck for very large runs.

4. **Poisson solver:** Distributed FFT-PM via i-slab ↔ j-pencil transposes (commits
   316e565, 8d835e7, a9c390d). Each rank computes local k and j FFTs, then a global
   all-to-all transpose enables the i-axis FFT to run locally. solve_poisson extracts
   the rank's local slab, calls solve_poisson_fft_mpi, then allgather_slabs reassembles
   full phi for grad_phi. On a single rank this is bitwise identical to the serial solver.

5. **Particle migration:** After each drift, particles crossing slab boundaries are
   sent to left/right neighbour ranks. Periodic wrapping is handled.

**What works:**
- Non-MPI path unchanged and all 83 tests pass
- MpiContext with slab geometry, halo exchange, particle migration, reductions
- Simulation loop wired with cfg(feature = "mpi_parallel") at all key points
- Timestep synchronisation via allreduce_max(v_max)
- Output gated to rank 0

**TODO (MPI):**
- [x] Distributed FFT for Poisson solve (pencil decomposition) — done
- [x] Replace density allreduce with per-rank CIC + ghost exchange (last O(N³) comm)
  - `MpiContext::reduce_ghost_density`: two O(N²) neighbour exchanges replace the O(N³) `allreduce_sum_vec`
  - Only local slab [i_lo, i_hi) is correct after exchange; sufficient for Poisson and scalar-field solvers
- [x] Distributed multigrid for Symmetron scalar field solver (on local slabs)
- [x] Distributed multigrid for f(R) scalar field solver — `solve_fofr_mpi` mirrors symmetron MPI solver; `fofr_ngs_sweep_slab`, `fofr_residual_slab`, `v_cycle_fofr_mpi` added; dispatched in `simulation.rs` under `mpi_parallel` feature
- [x] Multi-rank integration test — `tests/mpi_ghost_exchange.rs` (harness=false); run via `make test-mpi` (mpirun -np 2); verifies ghost density exchange is correct across rank boundaries
- [x] Particle gather to rank 0 for snapshot/power spectrum output — `gather_particles_to_root` in `mpi/mod.rs`, wired in simulation loop output block
- [ ] Load balancing for AMR — **someday**: requires AMR V-cycle solver (prolongation/restriction) to be wired into the physics loop first; SFC (Hilbert-curve) oct partitioning only becomes useful once refined cells are actually solved on

**TODO (performance):**
- [x] Pre-allocate simulation scratch buffers (density, source, phi, grav_grid, FFT arrays)
  — done via `SimScratch` struct passed into step loop. Eliminates ~1 GB/step heap churn
  that caused the 36-hour Windows runtime.

---

## Analysis scripts

| Script | Purpose |
|--------|---------|
| `analysis/pk_theory.py` | BBKS linear P(k) + growth factors, compare to snapshots |
| `analysis/debug_pk.py` | Raw P(k) from snapshot binary |
| `analysis/ic_norm_check.py` | Python mirror of IC generator, normalization check |
| `analysis/ic_debug_norm.py` | Step-by-step IC normalization derivation |
| `analysis/ic_amplitude_check.py` | Expected IC amplitudes from theory |
| `analysis/plot_isis.py` | ISIS verification figure generator |
| `analysis/render_animation.py` | Cosmic web animation from snapshot series (MP4/GIF) |
| `analysis/params/lcdm_test.toml` | ΛCDM test run config (64³, 64 Mpc/h, WMAP7) |
| `analysis/params/symmetron_test.toml` | Symmetron test run config |
| `analysis/params/lcdm_ic_check.toml` | Short run (a=0.02→0.025) to check IC P(k) |
| `analysis/params/lcdm_anim.toml` | 128³ 64 Mpc/h run with 50 snapshots for animation |

---

## Animation

`analysis/render_animation.py` produces a dark matter cosmic web animation from any
snapshot series. Features:

- **Multi-scale Gaussian splatting**: 4 blur passes (σ=1, 5, 20, 60 px) composited with
  log1p weighting — filament-scale layer dominant, approximates adaptive SPH kernel rendering
- **Standard cosmic web colormap**: black → indigo → royal blue → cyan → white
  (matches Millennium/IllustrisTNG dark matter renders)
- **Cosmic time label**: z, scale factor a, and age in Myr/Gyr (ΛCDM cosmology)
- **Held last frame**: final snapshot stays on screen for 3 seconds to allow pausing

Usage:
```bash
uv run python analysis/render_animation.py \
    --snapdir output_lcdm_anim \
    --output animation.mp4 \
    --box-size 64
```

Recommended: run the simulation with `lcdm_anim.toml` (50 snapshots a=0.1→1.0) then
render at default 5 fps (= 10 s evolution + 3 s hold).
