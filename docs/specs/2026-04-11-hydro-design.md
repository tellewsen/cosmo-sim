# Hydrodynamics Design Spec

**Date:** 2026-04-11  
**Goal:** Add Eulerian AMR hydrodynamics to RUSTIS to reproduce Ellewsen, Falck & Mota (2018) — stacked halo gas density, temperature, and DM density profiles for ΛCDM and four Symmetron variants with three baryonic physics tiers.

---

## Reference

**Ellewsen, Falck & Mota 2018** ([arXiv:1803.04197](https://arxiv.org/abs/1803.04197)):
- Box: 64 h⁻¹ Mpc, 256³ DM particles
- Gas solver: RAMSES-style AMR (l_min=8, l_max=14)
- Cosmology: Ω_m=0.3, Ω_b=0.05, Ω_Λ=0.7, h=0.67
- Symmetron variants: A (β=1, a_SSB=0.5, λ₀=1), B (β=1, a_SSB=0.33, λ₀=1), C (β=2, a_SSB=0.5, λ₀=1), D (β=1, a_SSB=0.25, λ₀=1)
- Observables: stacked halo profiles of ρ_gas(r), T(r), ρ_DM(r) at z=0; halo mass function

---

## Approach

**Option A (implement now):** Domain-grid Godunov hydro on the existing n³ uniform grid. Gas state lives in `SimScratch` alongside the gravity arrays. AMR refinement of gas is deferred.

**Option B (future, preferred long-term):** Gas fields embedded in the AMR `Cell` struct with full octree Godunov solver and AMR sub-cycling — matching RAMSES exactly. Record in ROADMAP as the target architecture after Option A validates the physics.

Rationale for starting with A: the domain grid at 256³/64 Mpc/h (dx ≈ 0.25 Mpc/h) is sufficient for cluster-scale halo profiles. AMR sub-cycling is the hardest part of RAMSES hydro and can be added incrementally without compromising scientific validity.

---

## Feature Flag

Compiled in with `--features hydro`. Non-hydro builds are completely unchanged — no runtime cost, no dead branches.

```bash
cargo build --release --features hydro
cargo build --release --features hydro,symmetron
cargo build --release --features hydro,fofr
```

---

## Architecture

### New module: `src/hydro/`

| File | Role |
|------|------|
| `src/hydro/mod.rs` | Public API, re-exports |
| `src/hydro/solver.rs` | MUSCL-Hancock Godunov step |
| `src/hydro/riemann.rs` | HLL Riemann solver |
| `src/hydro/cooling.rs` | Radiative cooling (Tier 2) |
| `src/hydro/star_formation.rs` | Stochastic SF + star particles (Tier 2) |
| `src/hydro/feedback.rs` | Supernova thermal feedback (Tier 3) |
| `src/hydro/profiles.rs` | Halo finding (FoF) + stacked profile output |

### Gas state in `SimScratch` (added under `#[cfg(feature = "hydro")]`)

| Field | Type | Description |
|-------|------|-------------|
| `gas_rho` | `Vec<Real>` (n³) | Gas density [M☉/h / (Mpc/h)³] |
| `gas_vel` | `Vec<[Real;3]>` (n³) | Gas velocity [km/s] |
| `gas_u` | `Vec<Real>` (n³) | Specific internal energy [km²/s²] |

### `Simulation` struct addition

```rust
#[cfg(feature = "hydro")]
pub star_particles: Vec<StarParticle>,  // spawned by SF model
```

```rust
pub struct StarParticle {
    pub pos:  [Real; 3],
    pub vel:  [Real; 3],
    pub mass: Real,
    pub age:  Real,  // scale factor at formation
}
```

### Step loop addition (between kick and second drift)

```rust
#[cfg(feature = "hydro")]
hydro::solver::step(&mut scratch.gas_rho, &mut scratch.gas_vel, &mut scratch.gas_u,
                    dx, dtau, n, &config.hydro, &mut star_particles);
```

---

## Godunov Solver

**Scheme:** MUSCL-Hancock with HLL Riemann solver — the same combination used by RAMSES.

**Per timestep (operator-split over x, y, z):**
1. **Reconstruct** — linear interpolation of primitive variables (ρ, v, P) to cell interfaces using MinMod slope limiter. Ensures TVD property.
2. **Predict** — half-step Hancock update of interface states for second-order time accuracy.
3. **Riemann solve** — HLL solver at each interface computes intercell fluxes from left/right states.
4. **Update** — conservative update of conserved variables (ρ, ρv, E) from net fluxes.

**EOS:** Ideal gas, γ = 5/3. P = (γ−1) ρ u.

**Gravity + fifth force coupling:** After the Godunov flux update, the gas momentum and energy are updated with the gravitational acceleration `g = scratch.grav` (and fifth-force acceleration when a scalar field feature is active) as operator-split body-force terms:

```
ρv += ρ g Δτ
E  += ρ v·g Δτ
```

Gas feels the same gravitational and fifth-force accelerations as DM particles. This is the mechanism by which modified gravity affects the gas profiles — the central comparison with Ellewsen et al.

**Timestep:** Δτ_hydro = C_cfl × dx / max(|v| + c_s). Global minimum via `allreduce_min`. Overall step = min(Δτ_gravity, Δτ_hydro).

**Gas ICs:** ρ_gas = (Ω_b / Ω_m) × ρ_DM from CIC grid. v_gas = local DM velocity field. Standard approach for adiabatic ICs without a separate baryon transfer function.

---

## Baryonic Physics Tiers

Controlled via a `[hydro]` section in the params TOML:

```toml
[hydro]
cooling = true
star_formation = true
supernova_feedback = true
```

### Tier 1 — Non-radiative (adiabatic)

Just the Godunov solver. No energy sources or sinks. Gas heats at shocks and cools adiabatically during expansion. Baseline for solver validation.

### Tier 2 — Cooling + Star Formation

**Cooling:** Operator-split from hydro. Each step, subtract Λ(ρ, T) Δt from u using a tabulated metal-free cooling function (Sutherland & Dopita 1993, primordial H/He). Temperature floor at 10⁴ K (photoionisation equilibrium assumption).

**Star formation:** Stochastic Schmidt law. Cells with ρ_gas > ρ_SF_thresh and T < T_SF_thresh spawn a star particle with probability:

p = 1 − exp(−ε_SF × Δt / t_ff)

where t_ff = √(3π / 32Gρ) is the local free-fall time. The spawned star particle inherits cell position and velocity; the cell loses that mass.

Default thresholds (matching RAMSES defaults):
- ρ_SF_thresh = 0.1 H/cm³ (converted to code units)
- T_SF_thresh = 2×10⁴ K
- ε_SF = 0.01 (1% efficiency per free-fall time)

### Tier 3 — + Supernova Feedback

Each star particle deposits thermal energy E_SN = ε_SN × M_star × c² into surrounding cells after a fixed delay t_delay ≈ 10 Myr (in code units). Energy distributed over the nearest 27 cells with a Gaussian kernel. Thermal feedback only (no kinetic), matching the simplest RAMSES feedback scheme and Ellewsen et al.'s setup.

Default: ε_SN = 10⁻⁵ (fraction of rest-mass energy, equivalent to ~10⁵¹ erg per 10 M☉ of stars formed).

---

## Output

### Existing output (unchanged)
- `snapshot_{idx:04}.bin` — DM particle positions + velocities
- `power_spectrum_{idx:04}.txt` — DM P(k)

### New output (hydro feature)
- `gas_snapshot_{idx:04}.bin` — gas grid (ρ, v, u) at each output epoch
- `halo_profiles_{idx:04}.txt` — stacked halo profiles (see below)

### Halo finding
Friends-of-friends on DM particles: linking length b = 0.2 × mean inter-particle spacing. Minimum halo mass: 32 particles. Two mass bins matching Ellewsen et al.:
- High mass: M > 10^13.5 M☉/h (screened — small modified gravity signal)
- Low mass: M < 10^12 M☉/h (unscreened — large modified gravity signal)

### Profile format
For each mass bin, radially binned in 20 log-spaced bins from 0.1 to 3 R_200:
```
# r/R200  rho_gas/rho_mean  T_K  rho_dm/rho_mean  N_halos
```

---

## Validation Plan

1. **Adiabatic shock tube** — 1D Sod shock tube: confirm density jump ratio = 4, pressure and velocity profiles match analytic solution.
2. **Sedov blast** — point explosion: confirm self-similar blast wave radius R ∝ t^(2/5).
3. **Cooling test** — isolated uniform gas: confirm cooling time matches analytic Λ(T)/ρ.
4. **Full run** — 64 h⁻¹ Mpc, 256³, Symmetron A, all three baryonic tiers at z=0. Compare stacked halo profiles to Ellewsen et al. Fig. 3–5.

---

## ROADMAP additions

- Option A (domain-grid hydro): this implementation
- Option B (preferred long-term): embed gas in AMR Cell struct with octree Godunov and AMR sub-cycling — full RAMSES parity
- 256³ production runs for all four Symmetron variants × three baryonic tiers
