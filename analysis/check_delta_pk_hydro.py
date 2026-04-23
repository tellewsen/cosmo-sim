#!/usr/bin/env python3
"""ΔP/P comparison for Ellewsen et al. hydro runs.

Compares power spectra from Symmetron variants (Sym A/B/C/D) against the
ΛCDM hydro baseline. Both runs output at a=0.298, 0.479, 0.646, 0.817, 1.0.
Each Sym snapshot is matched to the ΛCDM snapshot at the nearest scale factor
(tolerance 0.05); unmatched snapshots are skipped with a warning.

Usage
-----
  uv run python analysis/check_delta_pk_hydro.py --lcdm output_ellewsen_lcdm \\
      --sym output_ellewsen_sym_a output_ellewsen_sym_b \\
             output_ellewsen_sym_c output_ellewsen_sym_d \\
      --labels "Sym A" "Sym B" "Sym C" "Sym D"

  # Print only z=0 snapshot for each variant:
  uv run python analysis/check_delta_pk_hydro.py --lcdm output_ellewsen_lcdm \\
      --sym output_ellewsen_sym_a --labels "Sym A" --z0-only
"""

import argparse
import glob
import os
import sys

import numpy as np


def parse_a(path):
    """Read scale factor from the first header line of a power_spectrum file."""
    with open(path) as f:
        line = f.readline()
    return float(line.split("a=")[1].split()[0])


def load_pk(path):
    """Return (k, P, N_modes) arrays from a power_spectrum_*.txt file."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 2].astype(int)


def find_pk_files(directory):
    """Return sorted list of power_spectrum_*.txt paths in directory."""
    files = sorted(glob.glob(os.path.join(directory, "power_spectrum_*.txt")))
    if not files:
        sys.exit(f"No power_spectrum files found in {directory}")
    return files


def load_lcdm_catalog(lcdm_dir):
    """Load all ΛCDM power spectrum snapshots, keyed by scale factor."""
    files = find_pk_files(lcdm_dir)
    catalog = {}  # a -> (k, p, n_modes)
    for f in files:
        a = parse_a(f)
        catalog[a] = load_pk(f)
    return catalog


def match_lcdm(catalog, a_sym, tol=0.05):
    """Find the ΛCDM snapshot closest in scale factor to a_sym."""
    best_a = min(catalog, key=lambda a: abs(a - a_sym))
    if abs(best_a - a_sym) > tol:
        return None, None, None, best_a
    k, p, n = catalog[best_a]
    return k, p, n, best_a


def print_table(label, a_sym, z_sym, k, p_lcdm, p_sym, nk):
    """Print a ΔP/P table for one Sym snapshot vs ΛCDM z=0."""
    safe = p_lcdm > 0
    ratio = np.where(safe, (p_sym - p_lcdm) / p_lcdm, 0.0)
    z_sym_str = f"{z_sym:.2f}"
    print(f"  a={a_sym:.4f}  z={z_sym_str}")
    print(f"  {'k[h/Mpc]':>12}  {'P_ΛCDM':>12}  {'P_sym':>12}  {'ΔP/P':>10}")
    stride = max(1, len(k) // nk)
    for i in range(0, len(k), stride):
        print(f"  {k[i]:12.4f}  {p_lcdm[i]:12.3e}  {p_sym[i]:12.3e}  {ratio[i]:+10.4f}")
    max_abs = np.max(np.abs(ratio[safe])) if safe.any() else 0.0
    print(f"  max |ΔP/P| = {max_abs:.4f}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--lcdm", required=True, metavar="DIR",
                    help="ΛCDM hydro output directory")
    ap.add_argument("--sym", required=True, nargs="+", metavar="DIR",
                    help="Symmetron hydro output directory (one or more)")
    ap.add_argument("--labels", nargs="+", metavar="LABEL",
                    help="Labels for each --sym dir (default: directory names)")
    ap.add_argument("--nk", type=int, default=8,
                    help="k bins to print per snapshot (default 8)")
    ap.add_argument("--z0-only", action="store_true",
                    help="Print only the z=0 snapshot for each variant")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.sym):
        ap.error("--labels must have the same number of entries as --sym")

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.sym]

    # Load ΛCDM catalog (all epochs)
    lcdm_catalog = load_lcdm_catalog(args.lcdm)
    a_lcdm_vals = sorted(lcdm_catalog)
    print(f"ΛCDM: {args.lcdm}  ({len(lcdm_catalog)} snapshots: "
          f"a={[f'{a:.3f}' for a in a_lcdm_vals]})")
    k_sample, p_sample, _ = lcdm_catalog[a_lcdm_vals[0]]
    print(f"{len(k_sample)} k bins, k = [{k_sample[0]:.4f}, {k_sample[-1]:.4f}] h/Mpc")
    print()

    for sym_dir, label in zip(args.sym, labels):
        sym_files = find_pk_files(sym_dir)
        if args.z0_only:
            sym_files = [max(sym_files, key=parse_a)]

        print("━" * 60)
        print(f"  {label}  ({sym_dir})")
        print("━" * 60)

        for path in sym_files:
            a_sym = parse_a(path)
            z_sym = 1.0 / a_sym - 1.0
            k_sym, p_sym, _ = load_pk(path)

            k_lcdm, p_lcdm, _, a_matched = match_lcdm(lcdm_catalog, a_sym)
            if k_lcdm is None:
                print(f"  WARNING: no ΛCDM snapshot near a={a_sym:.4f} "
                      f"(closest a={a_matched:.4f}), skipping")
                continue

            if not np.allclose(k_sym, k_lcdm, rtol=1e-3):
                print(f"  WARNING: k-bin mismatch at a={a_sym:.4f}, skipping")
                continue

            if abs(a_matched - a_sym) > 1e-3:
                print(f"  (using ΛCDM a={a_matched:.4f} for Sym a={a_sym:.4f})")

            print_table(label, a_sym, z_sym, k_lcdm, p_lcdm, p_sym, args.nk)

    print("Done.")


if __name__ == "__main__":
    main()
