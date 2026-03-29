#!/usr/bin/env python3
"""Single-mode DKD leapfrog growth test.

Tracks the amplitude of a single Fourier mode through the same DKD integrator
as the Rust simulation and compares s(a) to D(a) from linear theory.

If the ratio s/D stays near 1, the time integration is correct.
If it grows, there is a dynamical error.

Usage
-----
  uv run python analysis/test_dkd_growth.py
"""

import numpy as np
from scipy import integrate as sci_integrate

# ── Cosmology ────────────────────────────────────────────────────────────────

omega_m = 0.269
omega_l = 0.731
h       = 0.704
H0      = h * 100.0   # H₀ physical = h×100 km/s/Mpc (matches Rust Poisson source)

def E(a):    return np.sqrt(omega_m/a**3 + omega_l)
def H_phys(a): return h * 100.0 * E(a)   # km/s/Mpc
def da_dtau(a): return a * H_phys(a)      # cosmic-time RHS

def advance_a(a, dtau):
    f  = da_dtau
    k1 = f(a);              k2 = f(a + 0.5*dtau*k1)
    k3 = f(a + 0.5*dtau*k2); k4 = f(a +     dtau*k3)
    return a + dtau*(k1 + 2*k2 + 2*k3 + k4)/6

def growth_factor(a):
    def intgd(ap): return 1.0 / (ap * H_phys(ap))**3
    Ia, _ = sci_integrate.quad(intgd, 1e-6, a)
    I1, _ = sci_integrate.quad(intgd, 1e-6, 1.0)
    return E(a)*Ia / (E(1.0)*I1)

def growth_rate(a):
    return (omega_m / (a**3 * E(a)**2))**0.55

# ── DKD for a single mode ────────────────────────────────────────────────────
#
# State: s = δ amplitude,  v = ṽ amplitude = a² dδ/dt.
#
# Poisson source: force = (3/2) H0² Ωm s / a_half    [same as simulation code]
# Drift:  Δs = v / a² × Δτ   (using a at step start or a_half)
# Kick:   Δv = force × Δτ

def run_dkd(a_init, a_final, dtau_max=5e-5, courant=0.8,
            box_size=250.0, n_grid=64):
    D0 = growth_factor(a_init)
    f0 = growth_rate(a_init)

    # IC: s = D(a_init),  ṽ = a² × f × H × D  (for unit displacement Ψ=1 with k=1)
    s = D0
    v = a_init**2 * f0 * H_phys(a_init) * D0

    a = a_init
    records = []

    for step in range(500_000):
        if a >= a_final:
            break

        # Timestep: scale-factor CFL limits Δa/a ≤ 0.02
        cfl_a  = 0.02 / H_phys(a)                  # scale-factor CFL
        # Courant: dx/v_max where v_max = ṽ/a² ~ peculiar velocity (km/s)
        v_pec  = abs(v) / a**2 if abs(v) > 0 else 1e-10
        dx     = box_size / n_grid
        cfl_v  = courant * dx / v_pec
        dtau   = min(dtau_max, cfl_a, cfl_v)

        # Don't overshoot a_final
        da_max = (a_final - a) * 0.5
        if da_dtau(a) * dtau > da_max:
            dtau = da_max / da_dtau(a)
        if dtau <= 0:
            break

        # ── ½ Drift ──────────────────────────────────────────────────────
        s += v / a**2 * dtau * 0.5

        # ── a_half ───────────────────────────────────────────────────────
        a_half = advance_a(a, dtau * 0.5)

        # ── Kick ─────────────────────────────────────────────────────────
        force = 1.5 * H0**2 * omega_m * s / a_half
        v    += force * dtau

        # ── ½ Drift (second) ─────────────────────────────────────────────
        s += v / a_half**2 * dtau * 0.5

        # ── Advance a ────────────────────────────────────────────────────
        a = advance_a(a, dtau)

        records.append((a, s))

    return records


def main():
    a_init   = 0.02
    a_final  = 0.35
    dtau_max = 5e-5

    print("Single-mode DKD growth test")
    print(f"a_init={a_init}, a_final={a_final}, dtau_max={dtau_max}")
    print(f"Cosmology: Ωm={omega_m}, ΩΛ={omega_l}, h={h}")
    print()

    records = run_dkd(a_init, a_final, dtau_max=dtau_max)
    print(f"Total steps: {len(records)}")
    print()
    print(f"{'a':>8}  {'s_DKD':>14}  {'D_lin':>14}  {'ratio':>8}")

    # Print at a ≈ 0.05, 0.10, 0.20, 0.33 and start/end
    checkpoints = [0.05, 0.10, 0.20, 0.33, 0.35]
    last_printed = set()
    for (a, s) in records:
        for ac in checkpoints:
            if abs(a - ac) < 0.003 and ac not in last_printed:
                D = growth_factor(a)
                print(f"{a:8.5f}  {s:14.8f}  {D:14.8f}  {s/D:8.4f}")
                last_printed.add(ac)


if __name__ == "__main__":
    main()
