#!/usr/bin/env python3
"""Compare IC power spectrum to linear theory.

Reads the earliest power_spectrum file in a directory and computes
P_sim(k) / P_lin(k, a) to verify IC normalisation.

Usage
-----
  uv run python analysis/check_ic_amplitude.py --dir output_w15_ic_check \
      --params analysis/params/winther2015_lcdm_test.toml
"""

import argparse
import glob
import os
import tomllib
import numpy as np


# ── BBKS transfer function ────────────────────────────────────────────────────

def bbks(k, gamma):
    """Eisenstein-Hu-independent BBKS: T(k), k in h/Mpc. Vectorised."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    q = k / gamma
    small = q < 1e-10
    T = np.where(small, 1.0,
                 np.log(1 + 2.34*q) / (2.34*q)
                 / (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**0.25)
    return T.item() if T.size == 1 else T


def growth_factor(a, omega_m, omega_l):
    """Approximate D(a)/D(1) via Carroll, Press & Turner 1992."""
    omega_k = 1.0 - omega_m - omega_l
    # Integrate da/(aH)^3 from 0 to a, normalised to 1 at a=1
    from scipy import integrate
    def integrand(a_):
        E2 = omega_m/a_**3 + omega_k/a_**2 + omega_l
        return 1.0 / (a_ * np.sqrt(E2))**3
    # D ∝ H(a) ∫₀ᵃ da'/(a'H(a'))³
    H_a  = np.sqrt(omega_m/a**3  + omega_k/a**2  + omega_l)
    H_1  = np.sqrt(omega_m       + omega_k        + omega_l)
    I_a, _ = integrate.quad(integrand, 1e-6, a)
    I_1, _ = integrate.quad(integrand, 1e-6, 1.0)
    return (H_a * I_a) / (H_1 * I_1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, metavar="DIR",
                   help="Simulation output directory")
    p.add_argument("--params", required=True, metavar="TOML",
                   help="Simulation params .toml file (for cosmology)")
    args = p.parse_args()

    # Read cosmology
    with open(args.params, "rb") as f:
        cfg = tomllib.load(f)
    cosmo = cfg.get("cosmology", {})
    run   = cfg.get("run", {})
    h        = cosmo.get("h",            0.704)
    omega_m  = cosmo.get("omega_m",      0.269)
    omega_l  = cosmo.get("omega_lambda", 0.731)
    omega_b  = cosmo.get("omega_b",      0.0)
    n_s      = cosmo.get("n_s",          0.966)
    sigma8   = cosmo.get("sigma8",       0.8)
    box_size = run.get("box_size",       250.0)
    n_grid   = run.get("n_grid",         64)

    gamma = omega_m * h * np.exp(-omega_b * (1 + np.sqrt(2*h) / omega_m))

    def P_unnorm(k):
        return k**n_s * bbks(k, gamma)**2

    # σ8 normalisation
    k_norm = np.logspace(-3, 2, 5000)
    R8 = 8.0  # h/Mpc
    W = np.array([3*(np.sin(k*R8) - k*R8*np.cos(k*R8))/(k*R8)**3 for k in k_norm])
    sigma2_unnorm = np.trapezoid(P_unnorm(k_norm) * W**2 * k_norm**2, np.log(k_norm)) / (2*np.pi**2)
    A_s = sigma8**2 / sigma2_unnorm

    def P_lin_z0(k):
        return A_s * P_unnorm(k)

    # Load earliest power spectrum file
    files = sorted(glob.glob(os.path.join(args.dir, "power_spectrum_*.txt")))
    if not files:
        print(f"No power_spectrum files found in {args.dir}")
        return
    pk_file = files[0]

    with open(pk_file) as f:
        hdr = f.readline()
    a = float(hdr.split("a=")[1].split()[0])
    data = np.loadtxt(pk_file)

    D_a = growth_factor(a, omega_m, omega_l)
    D_1 = growth_factor(1.0, omega_m, omega_l)
    D_ratio = D_a / D_1

    print(f"File: {pk_file}")
    print(f"a = {a:.5f}  (z = {1/a-1:.2f})")
    print(f"D(a)/D(1) = {D_ratio:.6f}")
    print()
    print(f"{'k[h/Mpc]':>10}  {'P_sim':>12}  {'P_lin(a)':>12}  {'ratio':>8}  {'N_modes':>8}")

    k   = data[:, 0]
    pk  = data[:, 1]
    nm  = data[:, 2].astype(int)

    for i in range(len(k)):
        if pk[i] <= 0:
            continue
        p_lin_a = P_lin_z0(k[i]) * D_ratio**2
        ratio = pk[i] / p_lin_a
        print(f"{k[i]:10.4f}  {pk[i]:12.3e}  {p_lin_a:12.3e}  {ratio:8.3f}  {nm[i]:8d}")


if __name__ == "__main__":
    main()
