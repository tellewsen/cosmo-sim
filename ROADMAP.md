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

### Known remaining issue — DKD timestep error at early times

**Symptom:** P(k_f) at a=0.33 is ~7× larger than linear theory predicts, even with the IC
bug fixed.

**Diagnosis:**
- IC check (snapshot at a=0.032, 1 step): P_sim(k_f) = 5.5 vs theory 10.5, ratio ≈ 0.52
  (statistically reasonable for 3 independent modes: σ[P̂/P] ≈ 82%).
- Main run snapshot at a=0.33: P_sim(k_f) = 6070 vs theory 1100, ratio = 5.5.
- Sim grew P by 6070/5.5 ≈ 1104× from IC; linear theory predicts [D(0.33)/D(0.032)]² ≈ 104×.
  Excess growth factor ≈ 7×.

**Root cause:** The DKD leapfrog uses `self.a` (scale factor at step start) for:
1. Drift 2 (second half-step): should use a at t+Δτ/2 or t+Δτ
2. Poisson source: should use a at t+Δτ/2 (half-step)

For large Δa/a (e.g. first step: a goes 0.02 → 0.033, Δa/a ≈ 65%), using a_start
instead of a_half causes:
- Drift 2 too large by (a_half/a_start)² ≈ 1.8×
- Poisson source too strong by a_half/a_start ≈ 1.33×
Combined, P(k) grows ~3–7× too fast over the first few large steps.

**Planned fix (options, in order of preference):**

Option A: Add a scale-factor CFL: `Δτ < Δa_max / (a H(a))` with Δa_max/a ~ 0.02.
At a=0.02 this limits Δτ < 1.5e-6 (vs current 5e-5), requiring ~13,000 steps total.

Option B: Use a_half in the Poisson source and second drift:
```rust
let a_half = cosmology.advance_a(self.a, self.dtau * 0.5);
// use a_half for prefac and second drift
```

Option C: Switch to supercomoving time (dτ = dt/a²), matching RAMSES convention,
where the drift equation has no a-factor and the issue disappears.

---

### TODO

- [ ] Fix DKD timestep issue (implement Option A or B above)
- [ ] Re-run ΛCDM test after fix, verify P(k)/P_linear ≈ 1 on large scales at a=0.33
- [ ] Run Symmetron test simulation
- [ ] Produce ISIS comparison figure: P_sym/P_ΛCDM vs k
- [ ] Add scale-factor CFL to compute_dtau()
- [ ] Consider switching to supercomoving time integration (RAMSES convention)
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
