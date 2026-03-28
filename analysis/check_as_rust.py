"""
Reproduce the Rust amplitude() function exactly and compare to Python scipy.

The goal: does Rust get a different A_s from the midpoint-rule sigma8 integral?
Any discrepancy > 1% would explain the P_sim / P_lin excess.
"""
import numpy as np
from scipy import integrate

omega_m = 0.269
omega_l = 0.731
h       = 0.704
n_s     = 0.966
sigma8  = 0.8
omega_b = 0.0455

gamma = omega_m * h * np.exp(-omega_b * (1 + np.sqrt(2*h) / omega_m))
print(f"gamma = {gamma:.8f}")

def bbks(k):
    q = k / gamma
    if q < 1e-6: return 1.0
    denom = 1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    return np.log(1+2.34*q) / (2.34*q * denom**0.25)

def power_unnorm(k):
    if k < 1e-10: return 0.0
    t = bbks(k)
    return k**n_s * t * t

def tophat(x):
    if x < 1e-3: return 1.0 - x*x/10 + x**4/280
    return 3.0 * (np.sin(x) - x*np.cos(x)) / x**3

# ── Rust midpoint rule: n=10000, lnk0=ln(1e-4), lnk1=ln(1e2) ────────────────
r = 8.0
n = 10000
lnk0 = np.log(1e-4)
lnk1 = np.log(1e2)
dlnk = (lnk1 - lnk0) / n

sig8sq_rust = 0.0
for i in range(n):
    k   = np.exp(lnk0 + (i + 0.5) * dlnk)
    w   = tophat(k * r)
    sig8sq_rust += power_unnorm(k) * k**3 * w**2 * dlnk

sig8sq_rust /= (2 * np.pi**2)
A_s_rust = sigma8**2 / sig8sq_rust
print(f"\nRust midpoint (n=10000, k in [1e-4, 1e2]):")
print(f"  sig8sq = {sig8sq_rust:.8e}")
print(f"  A_s    = {A_s_rust:.8e}")

# ── Python scipy quad ─────────────────────────────────────────────────────────
def integrand_scipy(lnk):
    k = np.exp(lnk)
    w = tophat(k * r)
    return power_unnorm(k) * k**3 * w**2 / (2 * np.pi**2)

sig8sq_scipy, err = integrate.quad(integrand_scipy, np.log(1e-4), np.log(1e3))
A_s_scipy = sigma8**2 / sig8sq_scipy
print(f"\nPython scipy.quad (k in [1e-4, 1e3]):")
print(f"  sig8sq = {sig8sq_scipy:.8e}")
print(f"  A_s    = {A_s_scipy:.8e}")

# ── Python trapezoid (20000 pts, matches check_sigma8.py) ────────────────────
k_norm = np.logspace(-4, 3, 20000)
R8 = 8.0
W  = np.array([tophat(k*R8) for k in k_norm])
def bbks_arr(k_arr):
    out = []
    for k in k_arr:
        out.append(bbks(k))
    return np.array(out)
Pu = k_norm**n_s * bbks_arr(k_norm)**2
s2u_trapz = np.trapezoid(Pu * W**2 * k_norm**2, np.log(k_norm)) / (2*np.pi**2)
A_s_trapz = sigma8**2 / s2u_trapz
print(f"\nPython trapezoid (20000 pts, k in [1e-4, 1e3]):")
print(f"  sig8sq = {s2u_trapz:.8e}")
print(f"  A_s    = {A_s_trapz:.8e}")

# ── Ratios ────────────────────────────────────────────────────────────────────
print(f"\nRatios:")
print(f"  A_s_rust / A_s_scipy   = {A_s_rust/A_s_scipy:.6f}")
print(f"  A_s_trapz / A_s_scipy  = {A_s_trapz/A_s_scipy:.6f}")
print(f"  A_s_rust / A_s_trapz   = {A_s_rust/A_s_trapz:.6f}")

if abs(A_s_rust/A_s_scipy - 1) > 0.01:
    print(f"\n*** WARNING: Rust A_s differs from scipy by {100*(A_s_rust/A_s_scipy - 1):.2f}% ***")
    print(f"    This would cause P_sim/P_lin = {A_s_rust/A_s_scipy:.2f}")
else:
    print(f"\nA_s values agree to <1% — sigma8 integral is NOT the source of excess")

# ── Check: what A_s would explain ratio=3 at k=0.026, a=0.33? ────────────────
from scipy import integrate as sci_integrate
def E(a):    return np.sqrt(omega_m/a**3 + omega_l)
def H_phys(a): return h * 100.0 * E(a)
def growth_factor(a):
    def intgd(ap): return 1.0/(ap*H_phys(ap))**3
    Ia, _ = sci_integrate.quad(intgd, 1e-6, a)
    I1, _ = sci_integrate.quad(intgd, 1e-6, 1.0)
    return E(a)*Ia/(E(1.0)*I1)

print(f"\nD(0.02) = {growth_factor(0.02):.6f}")
print(f"D(0.33) = {growth_factor(0.33):.6f}")
print(f"Ratio   = {growth_factor(0.33)/growth_factor(0.02):.4f}")
