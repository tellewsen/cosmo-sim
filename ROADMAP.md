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

Running as of 2026-03-28:
- ΛCDM reference: `output_w15_lcdm_test/` — complete (367 steps)
- Symmetron: `output_w15_symmetron_test/` — ΔP/P ≈ 0 (resolution-limited: dx=3.9 Mpc/h >> λ₀=1 Mpc/h)
- f(R) F5: `output_w15_f5_test/` — running with fix (ca11caf); first snapshot at a=0.33 shows ΔP/P ≈ 0.01% (near zero at z=2, signal expected at z=0)
- f(R) F6: `output_w15_f6_test/` — running with fix; F6 Compton wavelength λ_C≈1.4 Mpc/h, marginally unresolved at dx=3.9 Mpc/h

Expected outcomes after fix (64³, 250 Mpc/h):
- F5 (λ_C ≈ 4 Mpc/h): small but non-zero ΔP/P at z=0 (marginally resolved: λ_C/dx ≈ 1)
- F6 (λ_C ≈ 1.4 Mpc/h): ΔP/P ≈ 0 (unresolved at this resolution)

#### Symmetron 64 Mpc/h rerun

Running: `output_symmetron_test/` with fixed binary (seed=42). Baseline: `output_lcdm_test/`.
dx = 1 Mpc/h = λ₀ → marginal resolution. LCDM growth test: PASS (see DKD section).

---

### TODO

- [x] Wait for F5/F6 z=0 snapshots; generate ΔP/P figures — done (`figs_winther2015_test/`)
- [x] Verify symmetron 64 Mpc/h — ΔP/P = 0 (dx=λ₀ marginal, unresolved)
- [x] ΛCDM linear growth — PASS (confirmed 2026-03-28, see DKD section above)
- [ ] Check symmetron fifth force units: possible missing velocity-scale factor (needs resolved run)
- [ ] Production 512³ runs (250 Mpc/h) for full Winther 2015 comparison
- [ ] MPI support for cluster-scale runs

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
| `analysis/params/lcdm_test.toml` | ΛCDM test run config (64³, 64 Mpc/h, WMAP7) |
| `analysis/params/symmetron_test.toml` | Symmetron test run config |
| `analysis/params/lcdm_ic_check.toml` | Short run (a=0.02→0.025) to check IC P(k) |
