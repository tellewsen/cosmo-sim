"""
Plot halo mass function dN/dlog10M from RUSTIS hydro snapshot outputs.

Reads halo_mass_function_{idx:04}.txt files written by the simulation and
overlays a Press-Schechter (1974) or Sheth-Tormen (1999) analytic fit for
comparison.

Usage:
    uv run python analysis/plot_mass_function.py \
        --dirs output_ellewsen_lcdm output_ellewsen_sym_a \
        --labels "ΛCDM" "Sym A" \
        --snap 0000 \
        --out figures/mass_function.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── I/O ───────────────────────────────────────────────────────────────────────

def parse_mf_file(path: Path):
    """
    Parse a halo_mass_function_NNNN.txt file written by write_mass_function().

    Returns:
        n_halos: int
        log_m:   array of log10(M [M_sun/h]) bin centres
        dn:      array of dN/dlog10M [(Mpc/h)^-3]
    """
    if not path.exists():
        return None, None, None

    n_halos = 0
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.search(r"n_halos_total=(\d+)", line)
                if m:
                    n_halos = int(m.group(1))
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))

    if not rows:
        return n_halos, None, None

    arr = np.array(rows)
    return n_halos, arr[:, 0], arr[:, 1]


# ── Press-Schechter analytic fit ──────────────────────────────────────────────

def sigma_bbks(m_msun_h: np.ndarray, omega_m: float, h: float,
               n_s: float, sigma8: float) -> np.ndarray:
    """
    RMS density fluctuation σ(M) using BBKS transfer function (Bardeen et al. 1986).

    Normalised so that σ(R=8 Mpc/h) = sigma8.
    """
    # Characteristic scale R [Mpc/h] corresponding to mass M
    rho_mean = 2.775e11 * omega_m * h**2  # M_sun/(Mpc/h)^3
    r = (3 * m_msun_h / (4 * np.pi * rho_mean)) ** (1.0 / 3.0)

    k_arr = np.logspace(-3, 3, 2000)  # h/Mpc

    def window(k, r_val):
        x = k * r_val
        return 3 * (np.sin(x) - x * np.cos(x)) / x**3

    def window8(k):
        x = k * 8.0
        return 3 * (np.sin(x) - x * np.cos(x)) / x**3

    # BBKS transfer function
    def T_bbks(k):
        q = k / (omega_m * h**2)
        return (np.log(1 + 2.34 * q) / (2.34 * q) *
                (1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4) ** -0.25)

    pk_shape = k_arr**n_s * T_bbks(k_arr)**2

    # Normalise to sigma8
    integrand8 = pk_shape * window8(k_arr)**2 * k_arr**2
    norm = sigma8**2 / np.trapezoid(integrand8, k_arr)

    sigma = np.empty(len(r))
    for i, ri in enumerate(r):
        integrand = pk_shape * window(k_arr, ri)**2 * k_arr**2
        sigma[i] = np.sqrt(norm * np.trapezoid(integrand, k_arr))
    return sigma


def press_schechter(log10_m: np.ndarray, omega_m: float, h: float,
                    n_s: float, sigma8: float) -> np.ndarray:
    """
    Press-Schechter (1974) mass function dN/dlog10M [(Mpc/h)^-3].

    Uses delta_c = 1.686 (spherical collapse).
    """
    m = 10.0**log10_m
    rho_mean = 2.775e11 * omega_m * h**2  # M_sun/(Mpc/h)^3
    delta_c = 1.686

    sigma = sigma_bbks(m, omega_m, h, n_s, sigma8)

    # Numerical derivative d ln σ / d ln M
    dm = m * 0.01
    sigma_plus  = sigma_bbks(m * (1 + 0.01), omega_m, h, n_s, sigma8)
    sigma_minus = sigma_bbks(m * (1 - 0.01), omega_m, h, n_s, sigma8)
    dlnsigma_dlnm = (sigma_plus - sigma_minus) / (2 * dm / m * sigma)

    nu = delta_c / sigma
    f_nu = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)

    dn_dlnm = (rho_mean / m) * f_nu * np.abs(dlnsigma_dlnm)
    return dn_dlnm / np.log10(np.e)  # convert to dN/dlog10M


# ── Plot ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot halo mass function")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Output directories to plot")
    parser.add_argument("--labels", nargs="+",
                        help="Legend labels (defaults to dir names)")
    parser.add_argument("--snap", default="0000",
                        help="Snapshot index string, e.g. 0000")
    parser.add_argument("--omega-m",  type=float, default=0.3)
    parser.add_argument("--h",        type=float, default=0.67)
    parser.add_argument("--n-s",      type=float, default=0.96)
    parser.add_argument("--sigma8",   type=float, default=0.83)
    parser.add_argument("--ps",       action="store_true",
                        help="Overlay Press-Schechter fit")
    parser.add_argument("--out", default="figures/mass_function.pdf")
    args = parser.parse_args()

    dirs   = args.dirs
    labels = args.labels or dirs

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    any_data = False
    for i, (d, lab) in enumerate(zip(dirs, labels)):
        path = Path(d) / f"halo_mass_function_{args.snap}.txt"
        n_halos, log_m, dn = parse_mf_file(path)
        if log_m is None:
            print(f"WARNING: {path} not found or empty — skipping")
            continue
        mask = dn > 0
        ax.semilogy(log_m[mask], dn[mask], "o-", color=colors[i % len(colors)],
                    label=f"{lab} (N={n_halos})", markersize=4)
        any_data = True

    if args.ps:
        log_m_ref = np.linspace(10.5, 15.0, 200)
        dn_ps = press_schechter(log_m_ref, args.omega_m, args.h, args.n_s, args.sigma8)
        mask_ps = dn_ps > 0
        ax.semilogy(log_m_ref[mask_ps], dn_ps[mask_ps], "k--", lw=1.5,
                    label="Press-Schechter (1974)")

    if not any_data and not args.ps:
        print("No data to plot — check --dirs and --snap")
        return

    ax.set_xlabel(r"$\log_{10}(M\ [M_\odot/h])$")
    ax.set_ylabel(r"$dN/d\log_{10}M\ [(\mathrm{Mpc}/h)^{-3}]$")
    ax.set_title("Halo mass function")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(10, 15)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
