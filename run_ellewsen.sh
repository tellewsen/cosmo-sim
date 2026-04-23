#!/usr/bin/env bash
# run_ellewsen.sh — Ellewsen et al. (2018) reproduction runs
#
# Usage:
#   bash run_ellewsen.sh sym_b      # run one sim
#   bash run_ellewsen.sh sym_c
#   bash run_ellewsen.sh sym_d
#   bash run_ellewsen.sh plot       # extract profiles + plot (run after all sims done)
#   bash run_ellewsen.sh quick      # 64³ pipeline sanity check (~minutes)
#
# Status:
#   [ ] ΛCDM 256³     — bash run_ellewsen.sh lcdm  (re-run: now saves 5 epochs)
#   [x] Sym A 256³    — done on dev machine (output_ellewsen_sym_a/)
#   [ ] Sym B 256³    — bash run_ellewsen.sh sym_b
#   [ ] Sym C 256³    — bash run_ellewsen.sh sym_c
#   [ ] Sym D 256³    — bash run_ellewsen.sh sym_d
#   [ ] Plot           — bash run_ellewsen.sh plot
#
# Prerequisites:
#   uv  (curl -LsSf https://astral.sh/uv/install.sh | sh)
#        or on Windows: winget install astral-sh.uv
#
# On Windows run this script from Git Bash.

set -e

RELEASE_BASE="https://github.com/tellewsen/rustis/releases/download/v0.2.1"

# Detect platform: native Windows (Git Bash/MSYS2) vs Linux/WSL2
case "$(uname -s)" in
  MINGW*|CYGWIN*|MSYS*)
    BIN_HYDRO="cosmo_sim-hydro-windows-x86_64.exe"
    BIN_HYDRO_SYM="cosmo_sim-hydro-sym-windows-x86_64.exe"
    ;;
  *)
    BIN_HYDRO="cosmo_sim-hydro-linux-x86_64"
    BIN_HYDRO_SYM="cosmo_sim-hydro-sym-linux-x86_64"
    ;;
esac

setup() {
  if [[ ! -f "./$BIN_HYDRO" ]]; then
    echo "Downloading $BIN_HYDRO..."
    curl -L "$RELEASE_BASE/$BIN_HYDRO" -o "$BIN_HYDRO"
    chmod +x "$BIN_HYDRO" 2>/dev/null || true
  fi
  if [[ ! -f "./$BIN_HYDRO_SYM" ]]; then
    echo "Downloading $BIN_HYDRO_SYM..."
    curl -L "$RELEASE_BASE/$BIN_HYDRO_SYM" -o "$BIN_HYDRO_SYM"
    chmod +x "$BIN_HYDRO_SYM" 2>/dev/null || true
  fi
  if [[ ! -d analysis/params ]]; then
    echo "Downloading param files..."
    curl -L "$RELEASE_BASE/ellewsen-params.tar.gz" -o ellewsen-params.tar.gz
    tar xzf ellewsen-params.tar.gz
  fi
}

run() {
  local bin="$1" params="$2" label="$3"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $label"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  RUST_LOG=info "./$bin" run --params "$params"
  echo "Done: $label"
}

CMD="${1:-}"

case "$CMD" in
  lcdm)
    setup
    run "$BIN_HYDRO" "analysis/params/ellewsen_lcdm.toml" "ΛCDM (256³, 5 epochs)"
    ;;
  sym_b)
    setup
    run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_b.toml" "Symmetron B (a_SSB=0.33, β=1, λ₀=1) — 256³"
    ;;
  sym_c)
    setup
    run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_c.toml" "Symmetron C (a_SSB=0.5, β=2, λ₀=1) — 256³"
    ;;
  sym_d)
    setup
    run "$BIN_HYDRO_SYM" "analysis/params/ellewsen_sym_d.toml" "Symmetron D (a_SSB=0.25, β=1, λ₀=1) — 256³"
    ;;
  plot)
    DIRS=(); LABELS=()
    for pair in \
      "output_ellewsen_lcdm:ΛCDM" \
      "output_ellewsen_sym_a:Sym A" \
      "output_ellewsen_sym_b:Sym B" \
      "output_ellewsen_sym_c:Sym C" \
      "output_ellewsen_sym_d:Sym D"; do
      d="${pair%%:*}"; l="${pair##*:}"
      if [[ -d "$d" ]]; then
        uv run python analysis/extract_profiles.py --snapdir "$d"
        DIRS+=("$d"); LABELS+=("$l")
      fi
    done
    # ΔP/P comparison
    SYM_DIRS=(); SYM_LABELS=()
    for pair in \
      "output_ellewsen_sym_a:Sym A" \
      "output_ellewsen_sym_b:Sym B" \
      "output_ellewsen_sym_c:Sym C" \
      "output_ellewsen_sym_d:Sym D"; do
      d="${pair%%:*}"; l="${pair##*:}"
      [[ -d "$d" ]] && SYM_DIRS+=("$d") && SYM_LABELS+=("$l")
    done
    if [[ -d output_ellewsen_lcdm && ${#SYM_DIRS[@]} -gt 0 ]]; then
      uv run python analysis/check_delta_pk_hydro.py \
        --lcdm output_ellewsen_lcdm \
        --sym "${SYM_DIRS[@]}" --labels "${SYM_LABELS[@]}"
    fi

    mkdir -p figures
    uv run python analysis/plot_halo_profiles.py \
      --dirs "${DIRS[@]}" --labels "${LABELS[@]}" \
      --out figures/ellewsen_profiles.pdf
    echo "Plot written to figures/ellewsen_profiles.pdf"
    ;;
  quick)
    setup
    run "$BIN_HYDRO" "analysis/params/ellewsen_lcdm_quick.toml" "ΛCDM quick (64³)"
    uv run python analysis/extract_profiles.py --snapdir output_ellewsen_lcdm_quick
    ;;
  *)
    echo "Usage: bash run_ellewsen.sh <lcdm|sym_b|sym_c|sym_d|plot|quick>"
    echo ""
    grep "^#" "$0" | head -20
    exit 1
    ;;
esac
