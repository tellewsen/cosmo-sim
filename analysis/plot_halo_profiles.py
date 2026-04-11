"""
Stacked halo profile plotter for RUSTIS hydro runs.

Reads halo_profiles_{mass_bin}_{snapshot}.txt files written by the simulation
and produces a 3-panel figure (rho_gas, T, rho_dm vs r/R200) in the style of
Ellewsen, Falck & Mota (2018).

Usage:
    uv run python analysis/plot_halo_profiles.py \
        --dirs output_lcdm output_sym_a \
        --labels "ΛCDM" "Sym A" \
        --mass-bin highmass \
        --out figures/halo_profiles_highmass.pdf
"""

import argparse
import sys
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_profile_file(path: Path):
    """
    Parse a profile text file written by write_profiles() in profiles.rs.

    Returns:
        n_halos: int
        r: array (r/R200)
        rho_gas: array (rho_gas/rho_mean), NaN where N_cells == 0
        t_k: array (T in K), NaN where N_cells == 0
        rho_dm: array (rho_dm/rho_dm_mean), NaN where N_cells == 0
    """
    if not path.exists():
        return None, None, None, None, None

    n_halos = 0
    rows = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # First header: "# mass_bin=...  n_halos=N"
                m = re.search(r"n_halos=(\d+)", line)
                if m:
                    n_halos = int(m.group(1))
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            r, rg, t, rd, nc = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
            rows.append((r, rg, t, rd, nc))

    if not rows:
        return n_halos, None, None, None, None

    arr = np.array(rows)
    r    = arr[:, 0]
    rg   = arr[:, 1]
    t    = arr[:, 2]
    rd   = arr[:, 3]
    nc   = arr[:, 4].astype(int)

    # Mask bins with no cells
    mask = nc == 0
    rg[mask] = np.nan
    t[mask]  = np.nan
    rd[mask] = np.nan

    return n_halos, r, rg, t, rd


def main():
    parser = argparse.ArgumentParser(description="Plot stacked halo profiles from RUSTIS hydro runs")
    parser.add_argument("--dirs",     nargs="+", required=True,  help="Output directories to plot")
    parser.add_argument("--labels",   nargs="+", default=None,   help="Legend labels (one per directory)")
    parser.add_argument("--snapshot", default="0000",            help="4-digit snapshot index (default: 0000)")
    parser.add_argument("--mass-bin", default="highmass",        choices=["highmass", "lowmass"],
                        help="Which mass bin to plot (default: highmass)")
    parser.add_argument("--out",      default=None,              help="Output figure path (PDF/PNG)")
    args = parser.parse_args()

    dirs   = args.dirs
    labels = args.labels if args.labels else args.dirs
    if len(labels) != len(dirs):
        print("ERROR: --labels must have the same number of entries as --dirs", file=sys.stderr)
        sys.exit(1)

    snap     = args.snapshot
    mass_bin = args.mass_bin
    out_path = args.out or f"figures/halo_profiles_{mass_bin}.pdf"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    filename = f"halo_profiles_{mass_bin}_{snap}.txt"

    # ── Load data ─────────────────────────────────────────────────────────────
    datasets = []
    for d, lbl in zip(dirs, labels):
        path = Path(d) / filename
        n_halos, r, rg, t_k, rd = parse_profile_file(path)
        if r is None:
            print(f"WARNING: no data in {path} (n_halos={n_halos})", file=sys.stderr)
        datasets.append((lbl, n_halos, r, rg, t_k, rd))

    any_data = any(r is not None for _, _, r, *_ in datasets)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (lbl, n_halos, r, rg, t_k, rd), color in zip(datasets, colors):
        if r is None:
            continue
        kw = dict(color=color, lw=1.8)
        axes[0].plot(r, rg,  label=f"{lbl}  (n={n_halos})", **kw)
        axes[1].plot(r, t_k, label=f"{lbl}  (n={n_halos})", **kw)
        axes[2].plot(r, rd,  label=f"{lbl}  (n={n_halos})", **kw)

    # Panel styling
    ylabels = [
        r"$\rho_\mathrm{gas}\,/\,\bar{\rho}$",
        r"$T\;[\mathrm{K}]$",
        r"$\rho_\mathrm{DM}\,/\,\bar{\rho}_\mathrm{DM}$",
    ]
    for ax, ylabel in zip(axes, ylabels):
        ax.set_xscale("log")
        # Only use log y-scale if there are positive values to plot
        lines = ax.get_lines()
        has_positive = any_data and any(
            np.any(np.isfinite(l.get_ydata()) & (np.array(l.get_ydata()) > 0))
            for l in lines
        )
        if has_positive:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if any_data:
            ax.legend(fontsize=9)

    axes[2].set_xlabel(r"$r\,/\,R_{200}$", fontsize=13)
    axes[0].set_xlim(0.1, 3.0)

    mass_label = "M > 10^{13.5}" if mass_bin == "highmass" else "M < 10^{12}"
    fig.suptitle(
        rf"Stacked profiles — ${mass_label}\;M_\odot/h$",
        fontsize=13, y=0.995,
    )

    if not any_data:
        for ax in axes:
            ax.text(0.5, 0.5, "No halos in this mass bin",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")

    # Save PDF and PNG
    pdf_path = out_path if out_path.endswith(".pdf") else out_path.rsplit(".", 1)[0] + ".pdf"
    png_path = pdf_path.replace(".pdf", ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
