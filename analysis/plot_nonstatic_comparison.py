#!/usr/bin/env python3
"""Two-panel comparison of non-static vs quasi-static Symmetron power spectra.

Panel 1 (top): ΔP/P vs k at z=0 for quasi-static and non-static Symmetron,
both measured against the ΛCDM baseline.
Panel 2 (bottom): ΔP/P vs k for the non-static run at all snapshot epochs,
coloured early→late (blue→red).

Usage
-----
  uv run python analysis/plot_nonstatic_comparison.py \
      --lcdm output_lcdm_test \
      --quasi output_symmetron_test \
      --nonstatic output_symmetron_nonstatic_test \
      --out figures/nonstatic_comparison.pdf
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers copied from analysis/check_delta_pk.py — keep behaviour identical.
# ---------------------------------------------------------------------------
def parse_a(path):
    """Read scale factor a from the first header line of a P(k) file."""
    with open(path) as f:
        line = f.readline()
    return float(line.split("a=")[1].split()[0])


def load_pk(path):
    """Load a power_spectrum_*.txt file as a (Nbin, 3) array: k, P(k), N_modes."""
    return np.loadtxt(path)


# ---------------------------------------------------------------------------
def list_snapshots(directory):
    files = sorted(glob.glob(os.path.join(directory, "power_spectrum_*.txt")))
    if not files:
        raise SystemExit(f"No power_spectrum files in {directory}")
    return files


def delta_pk(mg_path, lcdm_path, n_modes_min=10):
    """Return (k, ΔP/P) at bins where N_modes ≥ n_modes_min and P_ΛCDM > 0."""
    d_lcdm = load_pk(lcdm_path)
    d_mg = load_pk(mg_path)
    if d_lcdm.shape != d_mg.shape:
        raise SystemExit(
            f"Shape mismatch: {lcdm_path} {d_lcdm.shape} vs {mg_path} {d_mg.shape}"
        )
    k = d_lcdm[:, 0]
    p_lcdm = d_lcdm[:, 1]
    p_mg = d_mg[:, 1]
    n_modes = d_lcdm[:, 2]
    mask = (p_lcdm > 0) & (n_modes >= n_modes_min)
    ratio = np.zeros_like(p_lcdm)
    ratio[mask] = (p_mg[mask] - p_lcdm[mask]) / p_lcdm[mask]
    return k[mask], ratio[mask]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lcdm", required=True, help="ΛCDM output directory")
    p.add_argument("--quasi", required=True, help="Quasi-static Symmetron output directory")
    p.add_argument("--nonstatic", required=True, help="Non-static Symmetron output directory")
    p.add_argument(
        "--out",
        default="figures/nonstatic_comparison.pdf",
        help="Output PDF path (a .png twin is written alongside). Default: figures/nonstatic_comparison.pdf",
    )
    args = p.parse_args()

    lcdm_files = list_snapshots(args.lcdm)
    quasi_files = list_snapshots(args.quasi)
    nonst_files = list_snapshots(args.nonstatic)

    n_snap = len(lcdm_files)
    if not (len(quasi_files) == n_snap and len(nonst_files) == n_snap):
        raise SystemExit(
            f"Snapshot count mismatch: lcdm={n_snap}, quasi={len(quasi_files)}, "
            f"nonstatic={len(nonst_files)}"
        )

    # k-range filter for plotting
    K_MIN, K_MAX = 0.08, 0.4

    # ---- Panel 1: ΔP/P at z=0 ---------------------------------------------
    idx_z0 = n_snap - 1
    a_z0 = parse_a(lcdm_files[idx_z0])
    k_q, r_q = delta_pk(quasi_files[idx_z0], lcdm_files[idx_z0])
    k_n, r_n = delta_pk(nonst_files[idx_z0], lcdm_files[idx_z0])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 8.0), sharex=True, constrained_layout=True
    )

    ax1.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.plot(k_q, 100.0 * r_q, "-", color="C0", label="Quasi-static vs ΛCDM", linewidth=1.6)
    ax1.plot(k_n, 100.0 * r_n, "--", color="C3", label="Non-static vs ΛCDM", linewidth=1.6)
    ax1.set_xscale("log")
    ax1.set_xlim(K_MIN, K_MAX)
    ax1.set_ylabel(r"$\Delta P/P\ [\%]$")
    ax1.set_title(f"z = {1.0 / a_z0 - 1.0:.2f}")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, which="both", alpha=0.25)

    # ---- Panel 2: ΔP/P vs epoch (non-static only) -------------------------
    ax2.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)

    # Colour map: early (blue) → late (red)
    cmap = plt.get_cmap("coolwarm")
    for i in range(n_snap):
        a_i = parse_a(lcdm_files[i])
        z_i = 1.0 / a_i - 1.0
        k_i, r_i = delta_pk(nonst_files[i], lcdm_files[i])
        colour = cmap(i / max(n_snap - 1, 1))
        ax2.plot(
            k_i,
            100.0 * r_i,
            "-",
            color=colour,
            linewidth=1.5,
            label=f"z = {z_i:.2f}",
        )

    ax2.set_xscale("log")
    ax2.set_xlim(K_MIN, K_MAX)
    ax2.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$")
    ax2.set_ylabel(r"$\Delta P/P\ [\%]$")
    ax2.legend(loc="best", frameon=True, ncol=2)
    ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle(r"Symmetron $64^3$ $64\ \mathrm{Mpc}/h$ — non-static vs quasi-static")

    # ---- Save --------------------------------------------------------------
    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_pdf}  ({out_pdf.stat().st_size} bytes)")
    print(f"Wrote {out_png}  ({out_png.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
