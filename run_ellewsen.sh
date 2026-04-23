#!/usr/bin/env bash
# run_ellewsen.sh — Ellewsen et al. (2018) reproduction runs
#
# Downloads binaries from the v0.2.0 release, then runs all five configurations
# in sequence (ΛCDM + Symmetron A/B/C/D), followed by profile extraction and plotting.
#
# Usage:
#   bash run_ellewsen.sh          # full 256³ runs (~hours each)
#   bash run_ellewsen.sh --quick  # 64³ pipeline sanity check (~minutes)
#
# Prerequisites (WSL2):
#   uv  (pip install uv, or: curl -LsSf https://astral.sh/uv/install.sh | sh)

set -e

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
fi

RELEASE_BASE="https://github.com/tellewsen/rustis/releases/download/v0.2.0"
BIN_HYDRO="cosmo_sim-hydro-linux-x86_64"
BIN_HYDRO_SYM="cosmo_sim-hydro-sym-linux-x86_64"

# ── 1. Download binaries ──────────────────────────────────────────────────────

if [[ ! -x "./$BIN_HYDRO" ]]; then
  echo "Downloading $BIN_HYDRO..."
  curl -L "$RELEASE_BASE/$BIN_HYDRO" -o "$BIN_HYDRO"
  chmod +x "$BIN_HYDRO"
fi

if [[ ! -x "./$BIN_HYDRO_SYM" ]]; then
  echo "Downloading $BIN_HYDRO_SYM..."
  curl -L "$RELEASE_BASE/$BIN_HYDRO_SYM" -o "$BIN_HYDRO_SYM"
  chmod +x "$BIN_HYDRO_SYM"
fi

# ── 2. Download param files (if not already present) ─────────────────────────

if [[ ! -d analysis/params ]]; then
  echo "Downloading ellewsen-params.tar.gz..."
  curl -L "$RELEASE_BASE/ellewsen-params.tar.gz" -o ellewsen-params.tar.gz
  tar xzf ellewsen-params.tar.gz
fi

# ── 3. Runs ───────────────────────────────────────────────────────────────────

run() {
  local bin="$1" params="$2" label="$3"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $label"
  echo "  params: $params"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  RUST_LOG=info "./$bin" run --params "$params"
}

if [[ $QUICK -eq 1 ]]; then
  # ── Quick pipeline check (64³, ~minutes) ──────────────────────────────────
  # [ ] ΛCDM 64³ — pipeline sanity check
  run "$BIN_HYDRO" "analysis/params/ellewsen_lcdm_quick.toml" "ΛCDM quick (64³)"
else
  # ── Production runs (256³, ~hours each) ───────────────────────────────────
  # Run ΛCDM first; its output is the baseline for the ΔP/P and profile comparisons.

  # [ ] ΛCDM 256³
  run "$BIN_HYDRO" "analysis/params/ellewsen_lcdm.toml" "ΛCDM (256³)"

  # [ ] Symmetron A — a_SSB=0.5, β=1, λ₀=1
  run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_a.toml" "Symmetron A (256³)"

  # [ ] Symmetron B — a_SSB=0.33, β=1, λ₀=1
  run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_b.toml" "Symmetron B (256³)"

  # [ ] Symmetron C — a_SSB=0.5, β=2, λ₀=1
  run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_c.toml" "Symmetron C (256³)"

  # [ ] Symmetron D — a_SSB=0.25, β=1, λ₀=1
  run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_d.toml" "Symmetron D (256³)"
fi

# ── 4. Post-processing ────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Post-processing: extract profiles"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $QUICK -eq 1 ]]; then
  uv run python analysis/extract_profiles.py --snapdir output_ellewsen_lcdm_quick
else
  for outdir in output_ellewsen_lcdm output_ellewsen_sym_a output_ellewsen_sym_b \
                output_ellewsen_sym_c output_ellewsen_sym_d; do
    [[ -d "$outdir" ]] && uv run python analysis/extract_profiles.py --snapdir "$outdir"
  done

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Post-processing: plot halo profiles"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  uv run python analysis/plot_halo_profiles.py \
    --dirs output_ellewsen_lcdm output_ellewsen_sym_a output_ellewsen_sym_b \
           output_ellewsen_sym_c output_ellewsen_sym_d \
    --labels "ΛCDM" "Sym A" "Sym B" "Sym C" "Sym D" \
    --out figures/ellewsen_profiles.pdf
fi

echo ""
echo "All done."
