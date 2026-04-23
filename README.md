# cosmo_sim

A cosmological N-body simulation in **Rust**, architecturally modelled on:

- **RAMSES** (Teyssier 2002) — Adaptive Mesh Refinement (AMR), Poisson solver, leapfrog integrator
- **ISIS** (Llinares, Mota & Winther 2014, *A&A* 562 A78, [arXiv:1307.6748](https://arxiv.org/abs/1307.6748)) — University of Oslo code adding nonlinear multigrid scalar-field solvers for modified gravity models (Symmetron, f(R)) on top of RAMSES
- Non-static extension: Llinares & Mota (2014, *Phys. Rev. D* 89, 084023)

The goal is **Fortran-competitive performance** via:
- Zero-cost abstractions (Rust ownership, no GC pauses)
- `rayon` data-parallel iterators (mirrors OpenMP in RAMSES/ISIS)
- Structure-of-Arrays layout for particle data (cache-friendly)
- Link-time optimisation (`lto = "fat"`, `opt-level = 3`)

---

## Physics implemented

| Component | Method | Reference |
|-----------|--------|-----------|
| Particle drift/kick | DKD leapfrog, supercomoving time | Martel & Shapiro 1998 |
| Density assignment | Cloud-In-Cell (CIC) | Hockney & Eastwood 1981 |
| Gravity | V-cycle multigrid Poisson; distributed FFT-PM (`mpi_parallel`) | Guillet & Teyssier 2011; RAMSES |
| **Symmetron** scalar field | Nonlinear Newton-GS multigrid; distributed FAS (`mpi_parallel`) | Llinares et al. 2014 (ISIS) |
| Symmetron non-static | Leapfrog χ + χ̇ | Llinares & Mota 2014 |
| **f(R) gravity** (Hu-Sawicki) | Nonlinear Newton-GS multigrid; distributed FAS (`mpi_parallel`) | Llinares et al. 2014 (ISIS) |
| AMR refinement | Graded octree | Teyssier 2002 (RAMSES) |
| Background cosmology | flat ΛCDM | Planck 2018 defaults |
| **Gas hydrodynamics** (`hydro`) | MUSCL-Hancock Godunov + HLL Riemann; operator-split cooling, SF, SN feedback | Teyssier 2002; Ellewsen et al. 2018 |

---

## Quick start

### Prerequisites

Rust stable toolchain + standard build tools.

For the `mpi_parallel` feature, you also need:
- An MPI implementation with dev headers (`libopenmpi-dev` or `mpich`)
- `clang` (used by bindgen to generate MPI bindings)

```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev clang

# Fedora/RHEL
sudo dnf install openmpi-devel clang

# macOS
brew install open-mpi llvm
```

### Build

```bash
# Standard gravity only
cargo build --release

# With Symmetron scalar field
cargo build --release --features symmetron

# With f(R) gravity
cargo build --release --features fofr

# Both scalar fields
cargo build --release --features symmetron,fofr

# Eulerian gas hydrodynamics (Ellewsen et al. baryonic physics)
cargo build --release --features hydro

# Hydro + Symmetron (for Ellewsen et al. comparison runs)
cargo build --release --features hydro,symmetron

# MPI multi-node (requires libopenmpi-dev + clang)
cargo build --release --features mpi_parallel

# MPI + hydro + Symmetron
cargo build --release --features mpi_parallel,hydro,symmetron
```

The `dist` profile (thin LTO, avoids OOM on memory-constrained systems) is available via the `Makefile`:

```bash
make hydro        # → build/hydro/dist/cosmo_sim
make hydro-sym    # → build/hydro-sym/dist/cosmo_sim
make hydro-fofr
make hydro-sym-fofr
make hydro-mpi
make hydro-sym-mpi
```

### Run

```bash
# Write example parameter file (Planck 2018, 256³ particles, Symmetron)
./target/release/cosmo_sim init --params params.toml

# Run (with initial conditions from e.g. MUSIC or GRAFIC)
./target/release/cosmo_sim run --params params.toml --ic ics.bin

# Run without IC (zero-velocity test, for debugging)
./target/release/cosmo_sim run --params params.toml
```

Set log verbosity:
```bash
RUST_LOG=debug ./target/release/cosmo_sim run --params params.toml
```

### MPI run

```bash
mpirun -np 4 ./target/release/cosmo_sim run --params params.toml
```

Each rank uses `rayon` for intra-node threading (hybrid MPI+threads).
Set `RAYON_NUM_THREADS` to control threads per rank.

---

## Parameter file (TOML)

```toml
[run]
box_size       = 64.0      # Comoving box side [Mpc/h]
n_grid         = 64        # Domain grid cells per side
levelmax       = 3         # Max AMR refinement levels
refine_thr     = 8         # Refine cell if N_particles > this
a_init         = 0.02      # Start scale factor
a_final        = 1.0       # End scale factor
dtau_max       = 0.01      # Max supercomoving time step
courant_factor = 0.8

[cosmology]                # Planck 2014 (matches ISIS papers)
omega_m      = 0.3175
omega_lambda = 0.6825
omega_b      = 0.049
h            = 0.6711
n_s          = 0.9624
sigma8       = 0.8344

[gravity]
epsilon   = 1e-4   # Poisson solver tolerance
n_mg_iter = 50
n_smooth  = 2

[particles]
n_part = 256       # 256³ particles total

[scalar_field]             # Symmetron (Llinares et al. 2014 fiducial)
a_ssb      = 0.5   # Scale factor at SSB
lambda_0   = 1.0   # Compton wavelength [Mpc/h]
beta_0     = 1.0   # Coupling strength
non_static = false # Set true for Llinares & Mota 2014 non-static run
epsilon_mg = 1e-5

[hydro]                    # Requires --features hydro
cooling            = true  # Radiative cooling (primordial H/He)
star_formation     = true  # Stochastic Schmidt SF
supernova_feedback = true  # Thermal SN feedback
# gamma          = 1.6667  # Adiabatic index (default 5/3)
# rho_sf_thresh  = 0.1     # SF density threshold [H/cm³]
# t_sf_thresh    = 2.0e4   # SF temperature threshold [K]

[output]
outdir   = "output"
n_output = 10
hdf5     = false
```

---

## Module structure

```
src/
├── main.rs                 CLI entry point (clap)
├── lib.rs                  Module declarations & re-exports
├── simulation.rs           Top-level run loop (DKD + Poisson + scalar field + hydro)
├── utils/
│   ├── types.rs            Real, Vec3, shared constants
│   ├── cosmology.rs        ΛCDM background, supercomoving time
│   └── config.rs           TOML parameter file (HydroParams added)
├── amr/
│   └── grid.rs             Octree AMR: cells, octs, refinement
├── gravity/
│   └── poisson.rs          V-cycle multigrid Poisson solver (serial + distributed FFT-PM)
├── hydro/                  Gas hydrodynamics (--features hydro)
│   ├── riemann.rs          HLL Riemann solver, MinMod slope limiter
│   ├── solver.rs           MUSCL-Hancock Godunov sweep (3D, operator-split gravity)
│   ├── ic.rs               Gas IC from DM density field; T↔u conversions
│   ├── cooling.rs          Radiative cooling (primordial H/He two-power-law)
│   ├── star_formation.rs   Stochastic Schmidt SF; StarParticle
│   ├── feedback.rs         Supernova thermal feedback (27-cell kernel)
│   └── profiles.rs         FoF halo finder + stacked gas/DM radial profiles
├── mpi/
│   └── mod.rs              Slab decomposition, halo exchange, particle migration, reductions
├── particles/
│   └── particles.rs        SoA particles, DKD leapfrog, CIC
└── scalar_field/
    ├── symmetron.rs        Symmetron: nonlinear NGS multigrid + non-static + MPI
    └── fofr.rs             Hu-Sawicki f(R): nonlinear NGS multigrid + MPI
```

---

## Initial conditions

Generate with [MUSIC](https://github.com/cosmo-sims/music) (C++, 2LPT) or
[GRAFIC](https://arxiv.org/abs/astro-ph/0012234) (Bertschinger 2001, used in ISIS papers).

The binary IC format expected by `load_ic()` is simply N×6 little-endian f64:
`[x, y, z, vx, vy, vz]` for each particle in Mpc/h and km/s.

---

## Performance notes

| N_part | Grid | ISIS (Fortran+MPI) | cosmo_sim (Rust+rayon) |
|--------|------|--------------------|------------------------|
| 256³   | 256³ | ~minutes/step      | comparable (single node) |
| 512³   | 512³ | requires MPI cluster | `--features mpi_parallel` |

MPI support uses i-slab domain decomposition with hybrid MPI+rayon threading.
Gravity uses a distributed FFT-PM Poisson solver (pencil transposes).
Scalar field solvers (Symmetron, f(R)) use distributed FAS multigrid on local slabs.

---

## References

- Teyssier, R. (2002). *A&A* 385, 337. [RAMSES]
- Llinares, C., Mota, D. F., & Winther, H. A. (2014). *A&A* 562, A78. [ISIS]
- Llinares, C. & Mota, D. F. (2014). *Phys. Rev. D* 89, 084023. [Non-static ISIS]
- Hinterbichler, K. & Khoury, J. (2010). *Phys. Rev. Lett.* 104, 231301. [Symmetron]
- Hu, W. & Sawicki, I. (2007). *Phys. Rev. D* 76, 064004. [f(R)]
- Martel, H. & Shapiro, P. R. (1998). *ApJ* 495, 521. [Supercomoving time]
- Guillet, T. & Teyssier, R. (2011). *J. Comput. Phys.* 230, 4756. [Multigrid Poisson]
- Ellewsen, J., Falck, B., & Mota, D. F. (2018). *A&A* 615, A134. [arXiv:1803.04197] [Hydro+Symmetron halos]
