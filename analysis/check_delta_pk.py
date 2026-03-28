#!/usr/bin/env python3
"""Quick ΔP/P diagnostic: compare modified-gravity vs ΛCDM power spectra.

Usage
-----
  uv run python analysis/check_delta_pk.py --lcdm <dir> --mg <dir>
  uv run python analysis/check_delta_pk.py --lcdm output_w15_lcdm_test --mg output_w15_f5_test
  uv run python analysis/check_delta_pk.py --lcdm output_lcdm_test --mg output_symmetron_test

Reads all power_spectrum_NNNN.txt files present in both directories and prints
ΔP/P = (P_MG − P_ΛCDM) / P_ΛCDM at a few representative k values per snapshot.
Useful for quick sanity checks while simulations are still running.
"""

import argparse
import os
import glob
import numpy as np


def parse_header(path):
    with open(path) as f:
        line = f.readline()
    a = float(line.split("a=")[1].split()[0])
    return a


def load_pk(path):
    return np.loadtxt(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lcdm", required=True, metavar="DIR",
                   help="ΛCDM output directory")
    p.add_argument("--mg", required=True, metavar="DIR",
                   help="Modified gravity output directory")
    p.add_argument("--nk", type=int, default=8,
                   help="Number of k bins to print per snapshot (default 8)")
    args = p.parse_args()

    lcdm_files = sorted(glob.glob(os.path.join(args.lcdm, "power_spectrum_*.txt")))
    if not lcdm_files:
        p.error(f"No power_spectrum files found in {args.lcdm}")

    print(f"ΛCDM dir : {args.lcdm}")
    print(f"MG dir   : {args.mg}")
    print()

    for lcdm_path in lcdm_files:
        fname = os.path.basename(lcdm_path)
        mg_path = os.path.join(args.mg, fname)
        if not os.path.exists(mg_path):
            continue

        a_lcdm = parse_header(lcdm_path)
        a_mg   = parse_header(mg_path)
        if abs(a_lcdm - a_mg) > 1e-3:
            print(f"WARNING: scale factor mismatch {fname}: "
                  f"ΛCDM a={a_lcdm:.4f}, MG a={a_mg:.4f}")

        d_lcdm = load_pk(lcdm_path)
        d_mg   = load_pk(mg_path)

        if d_lcdm.shape != d_mg.shape:
            print(f"WARNING: shape mismatch in {fname}, skipping")
            continue

        k      = d_lcdm[:, 0]
        p_lcdm = d_lcdm[:, 1]
        p_mg   = d_mg[:, 1]

        # Avoid division by zero from empty bins
        safe = p_lcdm > 0
        ratio = np.where(safe, (p_mg - p_lcdm) / p_lcdm, 0.0)

        z = 1.0 / a_lcdm - 1.0
        print(f"a={a_lcdm:.4f}  z={z:.2f}  ({fname})")
        print(f"  {'k[h/Mpc]':>12}  {'P_ΛCDM':>12}  {'P_MG':>12}  {'ΔP/P':>10}")
        for i in range(min(args.nk, len(k))):
            print(f"  {k[i]:12.4f}  {p_lcdm[i]:12.3e}  {p_mg[i]:12.3e}  {ratio[i]:+10.4f}")
        max_abs = np.max(np.abs(ratio[safe])) if safe.any() else 0.0
        print(f"  max |ΔP/P| = {max_abs:.4f}")
        print()


if __name__ == "__main__":
    main()
