#!/usr/bin/env bash
# run_comparison.sh — Run ΛCDM + modified-gravity simulations and produce
# all ISIS verification plots.
#
# Usage
# -----
#   Quick test (64³, ~minutes):
#     bash analysis/run_comparison.sh test symmetron
#
#   Full resolution (256³, ~hours):
#     bash analysis/run_comparison.sh full symmetron
#
#   f(R) comparison:
#     bash analysis/run_comparison.sh test fofr
#
# Arguments
# ---------
#   $1 = resolution: "test" (64³) or "full" (256³)
#   $2 = model:      "symmetron" or "fofr"
#   $3 = seed        (optional, default 42)

set -euo pipefail

RESOLUTION=${1:-test}
MODEL=${2:-symmetron}
SEED=${3:-42}
PARAMS_DIR="analysis/params"
FIGS_DIR="figs_${MODEL}_${RESOLUTION}"

echo "=== cosmo_sim ISIS verification ==="
echo "  Resolution : $RESOLUTION"
echo "  Model      : $MODEL"
echo "  Seed       : $SEED"
echo ""

# ── Build ────────────────────────────────────────────────────────────────────
echo "[1/4] Building binaries..."

cargo build --release 2>&1 | tail -3
echo "  → ΛCDM binary ready"

cargo build --release --features "$MODEL" 2>&1 | tail -3
echo "  → $MODEL binary ready (same binary, activated by params)"
# Note: since the scalar-field solver is gated on [scalar_field] config section,
# the same binary handles both ΛCDM and MG runs.

BIN="./target/release/cosmo_sim"

# ── ΛCDM reference run ───────────────────────────────────────────────────────
echo ""
echo "[2/4] Running ΛCDM reference (seed=$SEED)..."
LCDM_PARAMS="${PARAMS_DIR}/lcdm_${RESOLUTION}.toml"
$BIN run --params "$LCDM_PARAMS" --seed "$SEED"
LCDM_DIR=$(grep 'outdir' "$LCDM_PARAMS" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
echo "  → Output in: $LCDM_DIR"

# ── Modified-gravity run ─────────────────────────────────────────────────────
echo ""
echo "[3/4] Running $MODEL (seed=$SEED, SAME seed as ΛCDM)..."
MG_PARAMS="${PARAMS_DIR}/${MODEL}_${RESOLUTION}.toml"
$BIN run --params "$MG_PARAMS" --seed "$SEED"
MG_DIR=$(grep 'outdir' "$MG_PARAMS" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
echo "  → Output in: $MG_DIR"

# ── Plot ─────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Producing ISIS verification plots..."

LABEL=""
if [ "$MODEL" = "symmetron" ]; then
    LABEL="Symmetron (a_ssb=0.5, λ=1 Mpc/h, β=1)"
elif [ "$MODEL" = "fofr" ]; then
    LABEL="f(R) Hu-Sawicki (|f_R0|=1e-5, n=1)"
fi

python analysis/plot_isis.py \
    --lcdm  "$LCDM_DIR"  \
    --mg    "$MG_DIR"    \
    --label "$LABEL"     \
    --outdir "$FIGS_DIR"

echo ""
echo "=== Done! ==="
echo "  Figures in: $FIGS_DIR/"
echo ""
echo "  fig1_pk_z0.pdf         — P(k) at z=0  (both runs)"
echo "  fig2_delta_pk.pdf      — ΔP/P = (P_MG − P_ΛCDM)/P_ΛCDM  [ISIS Fig 3]"
echo "  fig3_density_slice.pdf — Projected density field"
echo "  fig4_pk_evolution.pdf  — P(k) at all output times"
