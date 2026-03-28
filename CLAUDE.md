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
python tests/validate_physics.py        # ~60 physics validation tests (numpy/scipy as reference)
python tests/test_algorithms.py         # unittest suite grouped by module

bash analysis/run_comparison.sh test symmetron    # 64³ quick run + ISIS comparison figures
bash analysis/run_comparison.sh full symmetron    # 512³ production run
bash analysis/run_comparison.sh test fofr

cargo bench                             # benches/bench_gravity.rs, benches/bench_cic.rs
```

Python tools require Python ≥3.11 with numpy, matplotlib, scipy. Use `uv` or `poetry` to manage the environment (see `pyproject.toml`).

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

## Known Issues (see ROADMAP.md)

**DKD timestep accuracy at early times:** The leapfrog uses the scale factor at step start instead of the half-step. For large Δa/a (≈65% at early times with a: 0.02→0.033), this causes 3–7× excess P(k) power growth. A scale-factor CFL condition and use of `a_half` in the Poisson source have been partially implemented but not fully validated. This is the primary open issue.

**Reference papers:** Llinares, Mota & Winther 2014 (ISIS), Teyssier 2002 (RAMSES), Hu & Sawicki 2007 (f(R)), Llinares & Mota 2014 (non-static Symmetron).
