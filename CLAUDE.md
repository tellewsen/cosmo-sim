# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RUSTIS (`cosmo_sim`) is a cosmological N-body simulation in Rust, porting ISIS (Llinares, Mota & Winther 2014) and RAMSES (Teyssier 2002) to Rust. It simulates dark matter particle dynamics under gravity and optional modified gravity scalar fields (Symmetron, f(R) Hu-Sawicki). The goal is Fortran-competitive performance with Rust's safety guarantees.

## Build

```bash
cargo build --release                                    # standard gravity
cargo build --release --features symmetron               # + Symmetron scalar field
cargo build --release --features fofr                    # + f(R) Hu-Sawicki
cargo build --release --features symmetron,fofr          # both
```

Features: `symmetron`, `fofr`, `hdf5_output`, `fft_pm` (default).

## Run

```bash
./target/release/cosmo_sim init --params params.toml     # generate example params
./target/release/cosmo_sim run --params params.toml      # run with Zel'dovich ICs
./target/release/cosmo_sim run --params params.toml --ic ics.bin  # run with external ICs
RUST_LOG=debug ./target/release/cosmo_sim run --params params.toml
```

## Tests and Analysis

```bash
cargo test --features fofr              # all Rust unit tests
python tests/validate_physics.py        # ~60 physics validation tests (numpy/scipy as reference)
python tests/test_algorithms.py         # unittest suite grouped by module
cargo bench                             # benches/bench_gravity.rs, benches/bench_cic.rs

# ΛCDM growth test (after running winther2015_lcdm_test.toml):
uv run python analysis/pk_theory.py --params analysis/params/winther2015_lcdm_test.toml \
    --growth-test output_w15_lcdm_test

# ΔP/P comparison (after running ΛCDM + MG sim):
uv run python analysis/check_delta_pk.py --lcdm output_w15_lcdm_test --mg output_w15_f5_test
```

Python tools require Python ≥3.11 with numpy, matplotlib, scipy. Use `uv run python` (see `pyproject.toml`).

## Architecture

**Physics loop** (per timestep, in `simulation.rs`):
1. Drift ½ — move particles (supercomoving DKD leapfrog)
2. CIC density assignment to grid
3. Poisson solve → gravitational acceleration
4. [If scalar field enabled] NGS multigrid solve → fifth force
5. Kick — apply accelerations to velocities
6. Drift ½
7. Advance scale factor (RK4)
8. AMR refinement
9. Snapshot output if due

**Key modules:**

| Module | Role |
|--------|------|
| `utils/types.rs` | `Real=f64`, `Vec3`, 3D C-order index helpers (`idx3`, `idx3p`, `from_flat`) |
| `utils/cosmology.rs` | ΛCDM background: H(a), D(a), ρ̄_m, supercomoving time conversion |
| `utils/config.rs` | TOML parsing: `RunParams`, `CosmoParams`, `GravityParams`, `ScalarFieldParams`, `OutputParams` |
| `gravity/poisson.rs` | V-cycle multigrid Poisson solver; optional FFT-PM (feature=`fft_pm`) |
| `particles/particles.rs` | SoA particle arrays (pos, vel); DKD leapfrog; CIC density assignment |
| `amr/grid.rs` | Graded octree AMR |
| `scalar_field/symmetron.rs` | Symmetron: nonlinear NGS multigrid + non-static mode |
| `scalar_field/fofr.rs` | f(R) Hu-Sawicki: nonlinear NGS multigrid |
| `ic/zeldovich.rs` | Zel'dovich approximation IC generator with 2LPT; IC normalization is `√(N³ P/V)` |
| `analysis/power_spectrum.rs` | P(k) estimator from snapshots |

**Performance design:** SoA particle layout for cache efficiency; `rayon` data-parallel iterators mirror OpenMP in RAMSES; release profile uses `lto="fat"`, `opt-level=3`, single codegen unit.

## Parameter Files

Located in `analysis/params/`. Structure:
```toml
[run]         # box_size, n_grid, levelmax, a_init, a_final, dtau_max, courant_factor
[cosmology]   # omega_m, omega_lambda, h, n_s, sigma8
[gravity]     # epsilon (tolerance), n_mg_iter, n_smooth
[particles]   # n_part, m_part
[scalar_field] # a_ssb, lambda_0, beta_0, non_static, epsilon_mg
[output]      # outdir, n_output, a_out, hdf5
```

Quick test configs: `lcdm_test.toml` (64³), `lcdm_full.toml` (512³).

## Analysis Scripts

Never run Python analysis inline (`python3 -c "..."`). Always write the script to
`analysis/<name>.py` first, then run with `uv run python analysis/<name>.py`.
This keeps every diagnostic reusable.

## Validation Status

ΛCDM linear growth: **PASS** (growth ratio test, N_modes≥50, k<0.13 h/Mpc, all epochs).
f(R) F5 fifth force: **CONFIRMED** (ΔP/P up to 8% at k=0.08 h/Mpc, z=0).
Symmetron: unresolved at 64³/250 Mpc/h (dx=3.9 Mpc/h >> λ₀=1 Mpc/h).
See `docs/validation/README.md` and `ROADMAP.md` for full results and bug history.

## f(R) solver notes

- `epsilon_mg` must be ≤ 1e-10 (source terms are O(10⁻⁸)); default 1e-5 is too loose.
- Initialise from epoch-specific background `fofr_of_r(rbar_a, r0, p)`, not z=0 value.
- F5 Compton wavelength λ_C ≈ 10.7 Mpc/h at z=0; F6 λ_C ≈ 3.4 Mpc/h.

**Reference papers:** Llinares, Mota & Winther 2014 (ISIS), Teyssier 2002 (RAMSES), Hu & Sawicki 2007 (f(R)), Llinares & Mota 2014 (non-static Symmetron).
