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
| Gravity | V-cycle multigrid Poisson | Guillet & Teyssier 2011; RAMSES |
| **Symmetron** scalar field | Nonlinear Newton-GS multigrid | Llinares et al. 2014 (ISIS) |
| Symmetron non-static | Leapfrog χ + χ̇ | Llinares & Mota 2014 |
| **f(R) gravity** (Hu-Sawicki) | Nonlinear Newton-GS multigrid | Llinares et al. 2014 (ISIS) |
| AMR refinement | Graded octree | Teyssier 2002 (RAMSES) |
| Background cosmology | flat ΛCDM | Planck 2018 defaults |

---

## Quick start

### Build

```bash
# Standard gravity only
cargo build --release

# With Symmetron scalar field
cargo build --release --features symmetron

# With f(R) gravity
cargo build --release --features fofr

# Both
cargo build --release --features symmetron,fofr
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
├── simulation.rs           Top-level run loop (DKD + Poisson + scalar field)
├── utils/
│   ├── types.rs            Real, Vec3, shared constants
│   ├── cosmology.rs        ΛCDM background, supercomoving time
│   └── config.rs           TOML parameter file
├── amr/
│   └── grid.rs             Octree AMR: cells, octs, refinement
├── gravity/
│   └── poisson.rs          V-cycle multigrid Poisson solver
├── particles/
│   └── particles.rs        SoA particles, DKD leapfrog, CIC
└── scalar_field/
    ├── symmetron.rs        Symmetron: nonlinear NGS multigrid + non-static
    └── fofr.rs             Hu-Sawicki f(R): nonlinear NGS multigrid
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
| 512³   | 512³ | requires MPI cluster | MPI support: roadmap |

MPI support (via `rsmpi`) is the main missing piece for cluster-scale runs.
The data structures are designed for domain decomposition.

---

## References

- Teyssier, R. (2002). *A&A* 385, 337. [RAMSES]
- Llinares, C., Mota, D. F., & Winther, H. A. (2014). *A&A* 562, A78. [ISIS]
- Llinares, C. & Mota, D. F. (2014). *Phys. Rev. D* 89, 084023. [Non-static ISIS]
- Hinterbichler, K. & Khoury, J. (2010). *Phys. Rev. Lett.* 104, 231301. [Symmetron]
- Hu, W. & Sawicki, I. (2007). *Phys. Rev. D* 76, 064004. [f(R)]
- Martel, H. & Shapiro, P. R. (1998). *ApJ* 495, 521. [Supercomoving time]
- Guillet, T. & Teyssier, R. (2011). *J. Comput. Phys.* 230, 4756. [Multigrid Poisson]
