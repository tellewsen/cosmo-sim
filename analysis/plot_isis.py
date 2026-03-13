#!/usr/bin/env python3
"""
plot_isis.py — Reproduce the verification plots from Llinares, Mota & Winther (2014).

Reads the binary snapshots and companion power-spectrum text files written by
cosmo_sim, then produces:

  Figure 1  P(k) at z = 0                             (single run)
  Figure 2  ΔP/P = (P_MG − P_ΛCDM) / P_ΛCDM at z=0  (two runs)
  Figure 3  2-D projected density field at z = 0      (single run)
  Figure 4  Scalar-field distribution histogram        (if scalar_field/ dir)

Usage
-----
  # Show all figures for one run (P(k) + density slice)
  python analysis/plot_isis.py output_lcdm_test/

  # Compare modified gravity vs ΛCDM  (reproduces ISIS Fig 3-4)
  python analysis/plot_isis.py --lcdm output_lcdm_test/ --mg output_symmetron_test/ \\
                               --label "Symmetron (a_ssb=0.5, λ=1 Mpc/h, β=1)"

  # Save to files instead of showing
  python analysis/plot_isis.py --lcdm ... --mg ... --outdir figs/

Dependencies
------------
  numpy  matplotlib  (scipy optional, for smoothing)

Snapshot binary format
----------------------
  [step: uint64 LE][a: float64 LE][n_part: uint64 LE]
  then N × 6 × float64 LE :  x  y  z  vx  vy  vz  (Mpc/h and km/s units)
"""

import os
import glob
import struct
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; remove for live display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── I/O helpers ─────────────────────────────────────────────────────────────

def read_snapshot(path):
    """Read a cosmo_sim binary snapshot.

    Returns
    -------
    step : int
    a    : float  scale factor
    pos  : (N, 3) float64  comoving positions [Mpc/h]
    vel  : (N, 3) float64  supercomoving velocities
    """
    with open(path, "rb") as f:
        step   = struct.unpack("<Q", f.read(8))[0]
        a      = struct.unpack("<d", f.read(8))[0]
        n_part = struct.unpack("<Q", f.read(8))[0]
        raw    = np.frombuffer(f.read(n_part * 6 * 8), dtype="<f8").reshape(n_part, 6)
    pos = raw[:, :3].copy()
    vel = raw[:, 3:].copy()
    return step, a, pos, vel


def read_power_spectrum(path):
    """Read a cosmo_sim power-spectrum text file.

    Returns
    -------
    a      : float  scale factor
    k      : (M,) float64  [h/Mpc]
    pk     : (M,) float64  [(Mpc/h)^3]
    nmodes : (M,) int
    """
    a = None
    with open(path) as f:
        for line in f:
            if line.startswith("# a="):
                try:
                    a = float(line.split("a=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if not line.startswith("#"):
                break
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    k      = data[:, 0]
    pk     = data[:, 1]
    nmodes = data[:, 2].astype(int)
    return a, k, pk, nmodes


def find_pk_at(outdir, a_target=1.0):
    """Find the power-spectrum file whose scale factor is closest to a_target."""
    files = sorted(glob.glob(os.path.join(outdir, "power_spectrum_*.txt")))
    if not files:
        raise FileNotFoundError(f"No power_spectrum_*.txt found in {outdir}")
    best_path, best_a, best_k, best_pk, best_nm = None, None, None, None, None
    best_da = np.inf
    for fp in files:
        try:
            a, k, pk, nm = read_power_spectrum(fp)
        except Exception:
            continue
        if a is not None and abs(a - a_target) < best_da:
            best_da = abs(a - a_target)
            best_path, best_a, best_k, best_pk, best_nm = fp, a, k, pk, nm
    if best_path is None:
        raise RuntimeError(f"Could not parse any P(k) file in {outdir}")
    return best_a, best_k, best_pk, best_nm


def find_snapshot_at(outdir, a_target=1.0):
    """Find the snapshot file whose scale factor is closest to a_target."""
    files = sorted(glob.glob(os.path.join(outdir, "snapshot_*.bin")))
    if not files:
        raise FileNotFoundError(f"No snapshot_*.bin found in {outdir}")
    best_path, best_a = None, None
    best_da = np.inf
    for fp in files:
        try:
            _, a, _, _ = read_snapshot(fp)
        except Exception:
            continue
        if abs(a - a_target) < best_da:
            best_da = abs(a - a_target)
            best_path, best_a = fp, a
    return best_path, best_a


# ─── Figure 1 — P(k) at z = 0 ────────────────────────────────────────────────

def plot_pk(outdir, ax=None, label=None, color=None, a_target=1.0):
    """Plot P(k) at a_target from outdir."""
    a, k, pk, nmodes = find_pk_at(outdir, a_target)
    mask = nmodes > 0
    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False
    lbl = label or os.path.basename(outdir.rstrip("/"))
    ax.loglog(k[mask], pk[mask], color=color, label=f"{lbl}  (a={a:.3f})", lw=1.8)
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$", fontsize=13)
    ax.set_ylabel(r"$P(k)\;[(\mathrm{Mpc}/h)^3]$", fontsize=13)
    ax.set_title("Matter power spectrum", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if standalone:
        plt.tight_layout()
        return fig
    return ax


# ─── Figure 2 — ΔP/P (reproduces ISIS Fig 3 & 4) ────────────────────────────

def plot_delta_pk(lcdm_dir, mg_dir, label="Modified gravity",
                  ax=None, color="tab:red", a_target=1.0):
    """Plot ΔP/P = (P_MG − P_ΛCDM) / P_ΛCDM at a_target.

    Reproduces the format of Llinares, Mota & Winther (2014) Figs 3-4:
    x-axis k in h/Mpc (log), y-axis ΔP/P (linear).
    """
    a_lcdm, k_lcdm, pk_lcdm, nm_lcdm = find_pk_at(lcdm_dir, a_target)
    a_mg,   k_mg,   pk_mg,   nm_mg   = find_pk_at(mg_dir,   a_target)

    # Interpolate onto common k grid (in case bins differ)
    mask_lcdm = (nm_lcdm > 0) & (pk_lcdm > 0)
    mask_mg   = (nm_mg   > 0) & (pk_mg   > 0)
    k_common  = np.intersect1d(
        np.round(k_lcdm[mask_lcdm], 6),
        np.round(k_mg  [mask_mg  ], 6),
    )
    # If grids are identical (same n_grid and box_size) just use them directly
    if len(k_common) < 3:
        # Fall back: interpolate MG onto ΛCDM grid
        pk_mg_interp = np.interp(
            np.log(k_lcdm[mask_lcdm]),
            np.log(k_mg  [mask_mg  ]),
            pk_mg[mask_mg],
        )
        k_plot   = k_lcdm[mask_lcdm]
        dpk      = (pk_mg_interp - pk_lcdm[mask_lcdm]) / pk_lcdm[mask_lcdm]
    else:
        # Same k grid — direct subtraction
        idx_l = np.isin(np.round(k_lcdm, 6), k_common) & mask_lcdm
        idx_m = np.isin(np.round(k_mg,   6), k_common) & mask_mg
        k_plot = k_lcdm[idx_l]
        dpk    = (pk_mg[idx_m] - pk_lcdm[idx_l]) / pk_lcdm[idx_l]

    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False

    ax.semilogx(k_plot, dpk, color=color, lw=1.8,
                label=f"{label}\n(a={a_mg:.3f})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$", fontsize=13)
    ax.set_ylabel(r"$\Delta P/P = (P_\mathrm{MG} - P_\Lambda)/P_\Lambda$", fontsize=13)
    ax.set_title("Power spectrum ratio vs ΛCDM", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    if standalone:
        plt.tight_layout()
        return fig
    return ax


# ─── Figure 3 — Projected density field ─────────────────────────────────────

def plot_density_slice(outdir, ax=None, n_grid=256, a_target=1.0):
    """Plot 2-D projected (column-density) overdensity from the last snapshot.

    Projects all particles onto the x-y plane using a CIC-smoothed grid,
    mirroring the density colour maps in Llinares & Mota (2014) PRD.
    """
    snap_path, a = find_snapshot_at(outdir, a_target)
    _, a, pos, _ = read_snapshot(snap_path)
    box = pos.max()   # approximate box size from positions

    # 2-D CIC projection onto n_grid × n_grid
    dx = box / n_grid
    grid = np.zeros((n_grid, n_grid), dtype=np.float64)
    px = pos[:, 0]
    py = pos[:, 1]
    # CIC
    u = px / dx - 0.5
    v = py / dx - 0.5
    i0 = np.floor(u).astype(int) % n_grid
    j0 = np.floor(v).astype(int) % n_grid
    i1 = (i0 + 1) % n_grid
    j1 = (j0 + 1) % n_grid
    tx = u - np.floor(u)
    ty = v - np.floor(v)
    np.add.at(grid, (i0, j0), (1 - tx) * (1 - ty))
    np.add.at(grid, (i1, j0),      tx  * (1 - ty))
    np.add.at(grid, (i0, j1), (1 - tx) *      ty )
    np.add.at(grid, (i1, j1),      tx  *      ty )

    # Normalise to overdensity (1 + δ)
    mean = grid.mean()
    if mean > 0:
        grid /= mean

    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False

    vmin, vmax = 0.1, grid.max() * 0.7
    im = ax.imshow(
        np.log10(grid.T + 0.1),
        origin="lower",
        extent=[0, box, 0, box],
        cmap="inferno",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label=r"$\log_{10}(1+\delta)$", fraction=0.046, pad=0.04)
    ax.set_xlabel("x  [Mpc/h]", fontsize=12)
    ax.set_ylabel("y  [Mpc/h]", fontsize=12)
    ax.set_title(f"Projected density  (a = {a:.3f})", fontsize=12)

    if standalone:
        plt.tight_layout()
        return fig
    return ax


# ─── Figure 4 — P(k) evolution across multiple snapshots ─────────────────────

def plot_pk_evolution(outdir, ax=None, a_values=None):
    """Plot P(k) at several scale factors to show structure growth."""
    files = sorted(glob.glob(os.path.join(outdir, "power_spectrum_*.txt")))
    if not files:
        raise FileNotFoundError(f"No power_spectrum_*.txt found in {outdir}")

    parsed = []
    for fp in files:
        try:
            a, k, pk, nm = read_power_spectrum(fp)
            if a is not None:
                parsed.append((a, k, pk, nm))
        except Exception:
            pass
    parsed.sort(key=lambda x: x[0])

    if a_values is not None:
        # Pick snapshots closest to requested a values
        selected = []
        for av in a_values:
            best = min(parsed, key=lambda x: abs(x[0] - av))
            selected.append(best)
    else:
        selected = parsed  # all snapshots

    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False

    cmap = plt.cm.viridis
    n = len(selected)
    for idx, (a, k, pk, nm) in enumerate(selected):
        mask = nm > 0
        color = cmap(idx / max(n - 1, 1))
        ax.loglog(k[mask], pk[mask], color=color, lw=1.5, label=f"a = {a:.3f}")

    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$", fontsize=13)
    ax.set_ylabel(r"$P(k)\;[(\mathrm{Mpc}/h)^3]$", fontsize=13)
    ax.set_title(f"P(k) evolution — {os.path.basename(outdir.rstrip('/'))}", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    if standalone:
        plt.tight_layout()
        return fig
    return ax


# ─── Combined ISIS verification dashboard ─────────────────────────────────────

def make_verification_dashboard(lcdm_dir, mg_dir, mg_label, outdir_figs):
    """Produce all ISIS paper verification figures and save to outdir_figs."""
    os.makedirs(outdir_figs, exist_ok=True)

    # ── Fig 1: P(k) at z=0 ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    plot_pk(lcdm_dir,  ax=ax1, label="ΛCDM",    color="tab:blue")
    plot_pk(mg_dir,    ax=ax1, label=mg_label,  color="tab:red")
    ax1.set_title("Matter power spectrum at z = 0", fontsize=13)
    path1 = os.path.join(outdir_figs, "fig1_pk_z0.pdf")
    fig1.savefig(path1, bbox_inches="tight")
    print(f"Saved: {path1}")
    plt.close(fig1)

    # ── Fig 2: ΔP/P (ISIS Fig 3 format) ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    plot_delta_pk(lcdm_dir, mg_dir, label=mg_label, ax=ax2, color="tab:red")
    path2 = os.path.join(outdir_figs, "fig2_delta_pk.pdf")
    fig2.savefig(path2, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close(fig2)

    # ── Fig 3: density slice — ΛCDM and MG side by side ──────────────────
    fig3, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_density_slice(lcdm_dir, ax=axL)
    axL.set_title("ΛCDM", fontsize=12)
    plot_density_slice(mg_dir, ax=axR)
    axR.set_title(mg_label, fontsize=12)
    fig3.suptitle("Projected density field at z = 0", fontsize=13)
    path3 = os.path.join(outdir_figs, "fig3_density_slice.pdf")
    fig3.savefig(path3, bbox_inches="tight")
    print(f"Saved: {path3}")
    plt.close(fig3)

    # ── Fig 4: P(k) growth history for MG run ───────────────────────────
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    plot_pk_evolution(mg_dir, ax=ax4)
    path4 = os.path.join(outdir_figs, "fig4_pk_evolution.pdf")
    fig4.savefig(path4, bbox_inches="tight")
    print(f"Saved: {path4}")
    plt.close(fig4)


def make_single_dashboard(outdir, outdir_figs):
    """Produce figures for a single run (no comparison)."""
    os.makedirs(outdir_figs, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    plot_pk(outdir, ax=ax1, color="tab:blue")
    fig1.savefig(os.path.join(outdir_figs, "fig1_pk_z0.pdf"), bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {os.path.join(outdir_figs, 'fig1_pk_z0.pdf')}")

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    plot_pk_evolution(outdir, ax=ax2)
    fig2.savefig(os.path.join(outdir_figs, "fig4_pk_evolution.pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {os.path.join(outdir_figs, 'fig4_pk_evolution.pdf')}")

    fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
    plot_density_slice(outdir, ax=ax3)
    fig3.savefig(os.path.join(outdir_figs, "fig3_density_slice.pdf"), bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {os.path.join(outdir_figs, 'fig3_density_slice.pdf')}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot ISIS verification figures from cosmo_sim output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("single", nargs="?",
                   help="Output directory for a single-run plot")
    p.add_argument("--lcdm",    metavar="DIR",
                   help="Output directory of the ΛCDM reference run")
    p.add_argument("--mg",      metavar="DIR",
                   help="Output directory of the modified gravity run")
    p.add_argument("--label",   default="Modified gravity",
                   help="Label for the MG curve (default: 'Modified gravity')")
    p.add_argument("--outdir",  default="figs",
                   help="Directory to save figures (default: figs/)")
    p.add_argument("--a",       type=float, default=1.0,
                   help="Target scale factor for z=0 plots (default: 1.0)")
    args = p.parse_args()

    if args.lcdm and args.mg:
        print(f"Comparison mode: ΛCDM={args.lcdm}  MG={args.mg}")
        make_verification_dashboard(args.lcdm, args.mg, args.label, args.outdir)
    elif args.single:
        print(f"Single-run mode: {args.single}")
        make_single_dashboard(args.single, args.outdir)
    else:
        p.print_help()
        sys.exit(1)

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()
