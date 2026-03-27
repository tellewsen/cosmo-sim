#!/usr/bin/env bash
# validate_winther.sh — Reproduce the validation suite from Winther et al. (2015)
# arXiv:1506.06384, "Modified Gravity N-body Code Comparison Project".
#
# Runs ΛCDM + Symmetron + f(R) F5 + f(R) F6 simulations with the exact
# cosmological and model parameters from that paper, then:
#   1. Checks ΛCDM linear-growth accuracy (P_sim/P_lin at k < 0.1 h/Mpc)
#   2. Produces ΔP/P comparison plots for each modified-gravity model
#
# Usage
# -----
#   Quick test (64³, ~minutes):
#     bash analysis/validate_winther.sh test
#
#   Production (512³, ~hours):
#     bash analysis/validate_winther.sh full
#
# Arguments
# ---------
#   $1 = resolution: "test" (64³) or "full" (512³, not yet parameterised)
#   $2 = seed (optional, default 42)
#
# Expected results (visual comparison against paper figures 2–3)
# ─────────────────────────────────────────────────────────────
#   ΛCDM growth test:
#     P_sim / P_lin  within ±10% at k < 0.1 h/Mpc at all output redshifts
#
#   Symmetron (a_ssb=0.5, β=1, λ=1 Mpc/h):
#     ΔP/P ≈ +5–15% at k = 0.1–1 h/Mpc at z=0
#     Signal below screening scale suppressed (quasi-static approx.)
#
#   f(R) F5 (|f_R0|=1e-5, n=1):
#     ΔP/P ≈ +20–50% bump centred on k ~ 0.3–3 h/Mpc at z=0
#     Characteristic chameleon-screened plateau at large k
#
#   f(R) F6 (|f_R0|=1e-6, n=1):
#     ΔP/P ≈ +5–15% bump, narrower and shifted to smaller k than F5
#     More strongly screened — weaker signal at high k
#
# Reference: Winther et al. 2015 Fig 2 (P(k) comparison across codes) and
#            Fig 3 (ΔP/P ratios for F5, F6, Symmetron).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RESOLUTION=${1:-test}
SEED=${2:-42}
PARAMS_DIR="analysis/params"
FIGS_DIR="figs_winther2015_${RESOLUTION}"

echo "=== Winther et al. (2015) validation suite ==="
echo "  Resolution : $RESOLUTION"
echo "  Seed       : $SEED"
echo "  Output figs: $FIGS_DIR/"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "[1/6] Building binaries..."

cargo build --release 2>&1 | tail -3
echo "  → ΛCDM binary ready"

cargo build --release --features symmetron 2>&1 | tail -3
echo "  → symmetron binary ready"

cargo build --release --features fofr 2>&1 | tail -3
echo "  → fofr binary ready"

BIN="./target/release/cosmo_sim"

# ── ΛCDM reference run ────────────────────────────────────────────────────────
echo ""
echo "[2/6] Running ΛCDM reference (seed=$SEED)..."
LCDM_PARAMS="${PARAMS_DIR}/winther2015_lcdm_${RESOLUTION}.toml"
$BIN run --params "$LCDM_PARAMS" --seed "$SEED"
LCDM_DIR=$(grep 'outdir' "$LCDM_PARAMS" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
echo "  → Output in: $LCDM_DIR"

# ── Modified-gravity runs (all with same seed for like-for-like ΔP/P) ─────────
echo ""
echo "[3/6] Running Symmetron (seed=$SEED)..."
SYMM_PARAMS="${PARAMS_DIR}/winther2015_symmetron_${RESOLUTION}.toml"
$BIN run --params "$SYMM_PARAMS" --seed "$SEED"
SYMM_DIR=$(grep 'outdir' "$SYMM_PARAMS" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
echo "  → Output in: $SYMM_DIR"

echo ""
echo "[4/6] Running f(R) F5 (seed=$SEED)..."
F5_PARAMS="${PARAMS_DIR}/winther2015_f5_${RESOLUTION}.toml"
$BIN run --params "$F5_PARAMS" --seed "$SEED"
F5_DIR=$(grep 'outdir' "$F5_PARAMS" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
echo "  → Output in: $F5_DIR"

echo ""
echo "[5/6] Running f(R) F6 (seed=$SEED)..."
F6_PARAMS="${PARAMS_DIR}/winther2015_f6_${RESOLUTION}.toml"
$BIN run --params "$F6_PARAMS" --seed "$SEED"
F6_DIR=$(grep 'outdir' "$F6_PARAMS" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
echo "  → Output in: $F6_DIR"

# ── Analysis ──────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Analysis..."

# 6a: ΛCDM linear-growth PASS/FAIL test
echo ""
echo "── ΛCDM linear-growth test ──────────────────────────────────────────────"
python analysis/pk_theory.py \
    --params "$LCDM_PARAMS" \
    --growth-test \
    "$LCDM_DIR"

# 6b: ΔP/P figures for each modified-gravity model
mkdir -p "$FIGS_DIR"

echo ""
echo "── Symmetron ΔP/P figures ───────────────────────────────────────────────"
python analysis/plot_isis.py \
    --lcdm   "$LCDM_DIR" \
    --mg     "$SYMM_DIR" \
    --label  "Symmetron (a_ssb=0.5, λ=1 Mpc/h, β=1)" \
    --outdir "${FIGS_DIR}/symmetron"

echo ""
echo "── f(R) F5 ΔP/P figures ─────────────────────────────────────────────────"
python analysis/plot_isis.py \
    --lcdm   "$LCDM_DIR" \
    --mg     "$F5_DIR" \
    --label  "f(R) F5 (|f_R0|=1e-5, n=1)" \
    --outdir "${FIGS_DIR}/fofr_f5"

echo ""
echo "── f(R) F6 ΔP/P figures ─────────────────────────────────────────────────"
python analysis/plot_isis.py \
    --lcdm   "$LCDM_DIR" \
    --mg     "$F6_DIR" \
    --label  "f(R) F6 (|f_R0|=1e-6, n=1)" \
    --outdir "${FIGS_DIR}/fofr_f6"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Done! ==="
echo ""
echo "Figures in $FIGS_DIR/:"
echo "  symmetron/fig2_delta_pk.pdf  — ΔP/P for Symmetron  [cf. Winther 2015 Fig 3]"
echo "  fofr_f5/fig2_delta_pk.pdf    — ΔP/P for f(R) F5    [cf. Winther 2015 Fig 2–3]"
echo "  fofr_f6/fig2_delta_pk.pdf    — ΔP/P for f(R) F6    [cf. Winther 2015 Fig 2–3]"
echo ""
echo "Compare against arXiv:1506.06384 Fig 2 (P(k)) and Fig 3 (ΔP/P)."
