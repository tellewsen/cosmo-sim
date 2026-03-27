# RUSTIS Validation Suite

Reproducing Winther et al. 2015 (arXiv:1506.06384) and ISIS paper (Llinares, Mota & Winther 2014, arXiv:1307.6748) tests.

## Validation Scripts

| Script | Purpose |
|---|---|
| `analysis/validate_winther.sh` | Top-level orchestration: build → run 4 sims → ΛCDM growth test → ΔP/P figures |
| `analysis/pk_theory.py` | BBKS linear P(k), σ8 normalisation, growth factor D(a); `--growth-test` PASS/FAIL table |
| `analysis/plot_isis.py` | ΔP/P ratio plots and density slices vs ΛCDM baseline |
| `analysis/debug_pk.py` | Manual P(k) diagnostic from snapshot binary (development use) |
| `analysis/ic_norm_check.py` | Python mirror of Rust Zel'dovich IC generator — validates IC normalisation |
| `analysis/ic_amplitude_check.py` | Theoretical vs actual δ_rms at a_init |

## Parameter Files

All in `analysis/params/`. Winther 2015 cosmology: Ωm=0.269, ΩΛ=0.731, h=0.704, ns=0.966, σ8=0.8, box=250 Mpc/h.

| File | Model | Output dir |
|---|---|---|
| `winther2015_lcdm_test.toml` | ΛCDM, 64³ | `output_w15_lcdm_test/` |
| `winther2015_symmetron_test.toml` | Symmetron (a_ssb=0.5, β=1, λ=1 Mpc/h) | `output_w15_symmetron_test/` |
| `winther2015_f5_test.toml` | f(R) F5 (|f_R0|=1e-5, n=1) | `output_w15_f5_test/` |
| `winther2015_f6_test.toml` | f(R) F6 (|f_R0|=1e-6, n=1) | `output_w15_f6_test/` |

## Running the Full Suite

```bash
# Quick validation (64³, ~15 min total):
bash analysis/validate_winther.sh test

# Or step by step:
cargo build --release
./target/release/cosmo_sim run --params analysis/params/winther2015_lcdm_test.toml --seed 42

cargo build --release --features symmetron
./target/release/cosmo_sim run --params analysis/params/winther2015_symmetron_test.toml --seed 42

cargo build --release --features fofr
./target/release/cosmo_sim run --params analysis/params/winther2015_f5_test.toml --seed 42
./target/release/cosmo_sim run --params analysis/params/winther2015_f6_test.toml --seed 42

# ΛCDM linear growth test:
uv run python analysis/pk_theory.py \
    --params analysis/params/winther2015_lcdm_test.toml \
    --growth-test output_w15_lcdm_test

# ΔP/P figures:
uv run python analysis/plot_isis.py \
    --lcdm output_w15_lcdm_test \
    --mg   output_w15_symmetron_test \
    --label "Symmetron (a_ssb=0.5, β=1, λ=1 Mpc/h)" \
    --outdir figs_winther2015_test/symmetron
```

## Expected Results (Paper Reference)

Visual comparison against Winther et al. 2015 Figs 2–3 (arXiv:1506.06384):

| Test | Expected | Tolerance |
|---|---|---|
| ΛCDM growth: P_sim/P_lin at k < 0.1 h/Mpc | ≈ 1.0 | ±10% |
| Symmetron ΔP/P at k = 0.1–1 h/Mpc, z=0 | +5–15% | visual |
| f(R) F5 ΔP/P at k = 0.3–3 h/Mpc, z=0 | +20–50% | visual |
| f(R) F6 ΔP/P at k = 0.1–1 h/Mpc, z=0 | +5–15% | visual |

---

## Results Log

### 2026-03-27 — ΛCDM linear growth test (64 Mpc/h box, WMAP7 cosmology)

**Run:** `output_lcdm_test/` (lcdm_test.toml, seed=42, 362 steps)

**Result: FAIL**

```
File                              a      z    k [h/Mpc]   P_sim/P_lin  status
power_spectrum_0000.txt       0.330   2.03     0.1025         4.002    FAIL
power_spectrum_0000.txt       0.330   2.03     0.1450         9.441    FAIL
power_spectrum_0001.txt       0.502   0.99     0.1025         4.997    FAIL
power_spectrum_0002.txt       0.671   0.49     0.1025         5.615    FAIL
power_spectrum_0003.txt       0.823   0.22     0.1025         5.960    FAIL
power_spectrum_0004.txt       1.000  -0.00     0.1025         6.239    FAIL
power_spectrum_0004.txt       1.000  -0.00     0.1450        11.405    FAIL
```

**Root cause:** DKD timestep inaccuracy at early times. The leapfrog uses the scale factor at step start rather than half-step for the gravitational source term. At a_init=0.02 the first step has Δa/a ≈ 65%, leading to 4–11× excess power growth. This error accumulates — by z=0 the measured δ_rms is 15.75 vs theoretical ~0.8. Documented as primary open issue in CLAUDE.md/ROADMAP.md.

**Impact on ΔP/P comparisons:** The absolute normalisation is wrong, but ΔP/P = (P_MG − P_ΛCDM)/P_ΛCDM should still show the correct *shape* of modified-gravity enhancement since the error is common to both runs (same seed, same timestep sequence).

---

### 2026-03-27 — Winther 2015 ΛCDM reference run (250 Mpc/h box)

**Run:** `output_w15_lcdm_test/` (winther2015_lcdm_test.toml, seed=42, 367 steps, completed ~19:00 UTC)

Sample P(k) (shot-noise subtracted):

| a | k_fund [h/Mpc] | P(k_fund) [(Mpc/h)³] |
|---|---|---|
| 0.332 (z=2.0) | 0.02625 | 1.88 × 10⁴ |
| 1.001 (z=0.0) | 0.02625 | 2.24 × 10⁵ |

Growth test not yet run on this output (pending — needs `uv run python analysis/pk_theory.py --params analysis/params/winther2015_lcdm_test.toml --growth-test output_w15_lcdm_test`).

---

### Stale output note

`output_symmetron_test/` was generated with a pre-fix version of the IC generator. All P(k) bins are zero (δ_rms ≈ 0.0004 at z=0). Discard and regenerate with current binary.

---

## Next Steps

- [ ] Run Winther symmetron sim: `cargo build --release --features symmetron && ./target/release/cosmo_sim run --params analysis/params/winther2015_symmetron_test.toml --seed 42`
- [ ] Run Winther f(R) F5 sim: `cargo build --release --features fofr && ./target/release/cosmo_sim run --params analysis/params/winther2015_f5_test.toml --seed 42`
- [ ] Run Winther f(R) F6 sim: `./target/release/cosmo_sim run --params analysis/params/winther2015_f6_test.toml --seed 42`
- [ ] Run growth test on output_w15_lcdm_test
- [ ] Generate ΔP/P figures for each MG model
- [ ] Fix primary DKD issue (a_half fix validation)
- [ ] Record all results in ROADMAP.md
