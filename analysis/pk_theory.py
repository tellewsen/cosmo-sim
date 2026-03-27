"""Compute theoretical linear P(k) (BBKS) and compare to simulation snapshots.

Usage:
  uv run python analysis/pk_theory.py [snapshot_dir] [n_grid] [box_size_Mpc/h]
  uv run python analysis/pk_theory.py --params analysis/params/winther2015_lcdm_test.toml
  uv run python analysis/pk_theory.py --params ... --growth-test   # PASS/FAIL table

--params   : read cosmology and box_size from a simulation params .toml file
--growth-test : print structured PASS/FAIL table (P_sim/P_lin at k < 0.1 h/Mpc)
"""
import os
import sys
import glob
import struct
import tomllib
import argparse
import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--params",      default=None,  help="params .toml file")
parser.add_argument("--growth-test", action="store_true", help="PASS/FAIL output")
parser.add_argument("positional",    nargs="*")
args, _ = parser.parse_known_args()

# ── Cosmology defaults (WMAP7 / ISIS paper) ────────────────────────────────────
h       = 0.719
omega_m = 0.267
omega_l = 0.733
omega_b = 0.0449
n_s     = 0.963
sigma8  = 0.801
L_default = 64.0

if args.params:
    with open(args.params, "rb") as _f:
        _cfg = tomllib.load(_f)
    _cosmo  = _cfg.get("cosmology", {})
    _run    = _cfg.get("run", {})
    h       = _cosmo.get("h",            h)
    omega_m = _cosmo.get("omega_m",      omega_m)
    omega_l = _cosmo.get("omega_lambda", omega_l)
    omega_b = _cosmo.get("omega_b",      omega_b)
    n_s     = _cosmo.get("n_s",          n_s)
    sigma8  = _cosmo.get("sigma8",       sigma8)
    L_default = _run.get("box_size",     L_default)
    # derive snapshot dir from output section if not overridden positionally
    _outdir = _cfg.get("output", {}).get("outdir", None)
else:
    _outdir = None

pos = args.positional
snap_dir = pos[0] if pos else (_outdir or "output_lcdm_test")
n_grid   = int(pos[1])   if len(pos) > 1 else 64
L        = float(pos[2]) if len(pos) > 2 else L_default

# ── BBKS transfer function ─────────────────────────────────────────────────────
def bbks(k, gamma):
    """k in h/Mpc, T(k)."""
    q = k / gamma
    if q < 1e-6:
        return 1.0
    denom = (1.0 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)
    return np.log(1.0 + 2.34*q) / (2.34*q * denom**0.25)

bbks_v = np.vectorize(bbks)

gamma = omega_m * h * np.exp(-omega_b * (1.0 + np.sqrt(2*h) / omega_m))
print(f"gamma = {gamma:.4f} h/Mpc")

def power_unnorm(k):
    return k**n_s * bbks_v(k, gamma)**2

def tophat(x):
    x = np.atleast_1d(np.float64(x))
    out = np.where(x < 1e-3, 1.0 - x**2/10 + x**4/280,
                   3*(np.sin(x) - x*np.cos(x)) / x**3)
    return out

# ── σ_8 normalisation ─────────────────────────────────────────────────────────
k_arr = np.logspace(-4, 2, 10000)
dlnk  = np.diff(np.log(k_arr)).mean()
r     = 8.0   # Mpc/h
w     = tophat(k_arr * r)
sig8sq_unnorm = np.sum(power_unnorm(k_arr) * k_arr**3 * w**2 * dlnk) / (2*np.pi**2)
A_s   = sigma8**2 / sig8sq_unnorm
print(f"A_s = {A_s:.6e}  (sigma8 check: {(A_s*sig8sq_unnorm)**0.5:.4f})")

# ── Linear P(k) at z=0 ────────────────────────────────────────────────────────
def P_lin(k, D=1.0):
    """P(k) in (Mpc/h)^3 at growth factor D."""
    return A_s * power_unnorm(k) * D**2

print(f"\nLinear P(k) at z=0:")
for k_test in [0.05, 0.1, 0.2, 0.5, 1.0]:
    print(f"  P(k={k_test:.2f}) = {P_lin(k_test):.2e} (Mpc/h)^3")

# ── Growth factor (matter domination approximation) ───────────────────────────
def E(a):
    return np.sqrt(omega_m / a**3 + omega_l)

def growth_factor(a):
    """Numerical growth factor D(a), normalised to D(1)=1."""
    def integrand(a_):
        h_ = a_ * E(a_) * h * 100   # a*H in (km/s/Mpc) * (h for h factor)
        # Actually: integral of da / (a*H)^3
        aH = a_ * E(a_) * h * 100   # km/s/Mpc
        return 1.0 / aH**3

    # integrate 0→a
    a_arr = np.linspace(1e-4, a, 5000)
    da = np.diff(a_arr).mean()
    integ_a = np.sum(integrand(a_arr)) * da

    a_arr1 = np.linspace(1e-4, 1.0, 5000)
    da1 = np.diff(a_arr1).mean()
    integ_1 = np.sum(integrand(a_arr1)) * da1

    return (E(a) * integ_a) / (E(1.0) * integ_1)

print(f"\nGrowth factors:")
for a_test in [0.02, 0.33, 0.5, 0.67, 0.82, 1.0]:
    D = growth_factor(a_test)
    print(f"  D({a_test:.2f}) = {D:.4f}  ->  P(k=0.1) = {P_lin(0.1, D):.2e} (Mpc/h)^3  "
          f"delta_rms ~ sigma8*D = {sigma8*D:.4f}")

# ── Read snapshots and compare ─────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Simulation snapshots vs theory (k_fund = {2*np.pi/L:.4f} h/Mpc):")
print(f"{'─'*60}")

snap_files = sorted(glob.glob(f"{snap_dir}/snapshot_*.bin"))
if not snap_files:
    print(f"No snapshots found in '{snap_dir}' — run the simulation first.")
for snap in snap_files:
    with open(snap, "rb") as f:
        step   = struct.unpack("<Q", f.read(8))[0]
        a_snap = struct.unpack("<d", f.read(8))[0]
        n_part = struct.unpack("<Q", f.read(8))[0]
        data   = np.frombuffer(f.read(n_part*6*8), dtype="<f8").reshape(n_part, 6)
    pos = data[:, :3]

    # CIC density
    dx = L / n_grid
    rho = np.zeros((n_grid, n_grid, n_grid))
    p_sc = pos / dx - 0.5
    for xi in p_sc:
        i0, j0, k0 = int(np.floor(xi[0])), int(np.floor(xi[1])), int(np.floor(xi[2]))
        tx, ty, tz = xi[0]-i0, xi[1]-j0, xi[2]-k0
        for di, wx in [(0, 1-tx), (1, tx)]:
            for dj, wy in [(0, 1-ty), (1, ty)]:
                for dk, wz in [(0, 1-tz), (1, tz)]:
                    rho[(i0+di)%n_grid, (j0+dj)%n_grid, (k0+dk)%n_grid] += wx*wy*wz
    rho = rho * n_grid**3 / n_part
    delta = rho - 1.0

    dk_fft = np.fft.fftn(delta)
    N3     = n_grid**3
    norm   = L**3 / N3**2
    shot   = L**3 / n_part

    # P at lowest mode (k_fund)
    P_k001 = norm * abs(dk_fft[0, 0, 1])**2  # raw (no shot subtraction)

    # Theoretical
    D = growth_factor(a_snap)
    P_theory_001 = P_lin(2*np.pi/L, D)   # k = k_fundamental

    snap_idx = os.path.basename(snap).split("_")[1].split(".")[0]
    print(f"snap {snap_idx}  a={a_snap:.4f}  D={D:.4f}  "
          f"delta_rms={delta.std():.4f}  "
          f"P_sim[k_f]={P_k001:.2e}  P_theory[k_f]={P_theory_001:.2e}  "
          f"ratio={P_k001/P_theory_001:.3f}  "
          f"sigma8*D={sigma8*D:.4f}")

# ── Growth test: compare saved P(k) files vs linear theory at multiple z ──────
# Uses power_spectrum_*.txt files (averaged over many modes, more reliable than
# a single Fourier mode from the raw snapshot).
#
# PASS criteria (matches Winther et al. 2015 standard):
#   P_sim(k, a) / P_lin(k, a)  within 10% for the lowest N_LOW_K_BINS modes
#   below K_MAX_LINEAR h/Mpc (quasi-linear regime).
#
if args.growth_test:
    pk_files = sorted(glob.glob(f"{snap_dir}/power_spectrum_*.txt"))
    if not pk_files:
        print(f"\n[growth-test] No power_spectrum_*.txt files found in '{snap_dir}'.")
        print("  Run the simulation first, then re-run with --growth-test.")
        sys.exit(1)

    TOLERANCE = 0.10        # 10% allowed deviation from linear theory
    # Test the lowest N_LOW_K_BINS modes in each snapshot, up to a hard cap of
    # K_MAX_LINEAR h/Mpc.  For a 64 Mpc/h box k_f ≈ 0.098 h/Mpc and the first
    # bin lands at 0.1025 h/Mpc; using N=3 bins captures k ≈ 0.10–0.17 which
    # is safely in the quasi-linear regime at all redshifts tested here.
    N_LOW_K_BINS = 3
    K_MAX_LINEAR = 0.20     # h/Mpc — hard cap, never test modes above this

    print(f"\n{'═'*70}")
    print(f"LINEAR GROWTH TEST  (P_sim/P_lin, tolerance ±{TOLERANCE*100:.0f}%, k < {K_MAX_LINEAR} h/Mpc, lowest {N_LOW_K_BINS} bins)")
    print(f"{'═'*70}")
    print(f"{'File':<28} {'a':>6} {'z':>5}  {'k [h/Mpc]':>10}  {'P_sim/P_lin':>12}  {'status':>6}")
    print(f"{'─'*70}")

    any_fail = False
    for pk_file in pk_files:
        # Read header for scale factor
        a_pk = None
        with open(pk_file) as f:
            for line in f:
                if line.startswith("# a="):
                    try:
                        a_pk = float(line.split("a=")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                    break
        if a_pk is None:
            continue

        data = np.loadtxt(pk_file, comments="#")
        if data.ndim < 2 or data.shape[0] < 2:
            continue
        k_arr, pk_arr = data[:, 0], data[:, 1]

        D = growth_factor(a_pk)
        mask    = k_arr < K_MAX_LINEAR
        if not np.any(mask):
            continue
        k_test  = k_arr[mask][:N_LOW_K_BINS]
        pk_test = pk_arr[mask][:N_LOW_K_BINS]
        fname   = os.path.basename(pk_file)

        for k_val, p_sim in zip(k_test, pk_test):
            p_theory = P_lin(k_val, D)
            ratio = p_sim / p_theory
            status = "PASS" if abs(ratio - 1.0) <= TOLERANCE else "FAIL"
            if status == "FAIL":
                any_fail = True
            z_val = 1.0 / a_pk - 1.0
            print(f"{fname:<28} {a_pk:>6.3f} {z_val:>5.2f}  {k_val:>10.4f}  {ratio:>12.3f}  {status:>6}")

    print(f"{'─'*70}")
    overall = "ALL PASS" if not any_fail else "SOME FAIL — see rows above"
    print(f"Result: {overall}")
    print(f"{'═'*70}\n")
