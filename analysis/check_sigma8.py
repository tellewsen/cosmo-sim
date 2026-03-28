"""
Check sigma8 normalization in P_lin_z0 calculation.

Two hypotheses:
1. R8 = 8 Mpc/h (correct) vs 8 Mpc (wrong units) — would change A_s
2. k in Mpc/h (correct) vs k in Mpc (wrong) — would change where the integral is evaluated

Also print P_lin_z0 at several k values for cross-check.
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
print(f"Sugiyama gamma = {gamma:.6f}")

def bbks(k):
    k = np.asarray(k, dtype=float)
    q = k / gamma
    T = np.where(q < 1e-6, 1.0,
        np.log(1+2.34*q)/(2.34*q) / (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**0.25)
    return T

def P_unnorm(k): return k**n_s * bbks(k)**2

# ── Variant A: R8 = 8.0 Mpc/h  (standard, k in h/Mpc) ──────────────────────
k_norm = np.logspace(-4, 3, 20000)
R8_a = 8.0
W_a  = 3*(np.sin(k_norm*R8_a) - k_norm*R8_a*np.cos(k_norm*R8_a)) / (k_norm*R8_a)**3
s2u_a = np.trapezoid(P_unnorm(k_norm)*W_a**2*k_norm**2, np.log(k_norm)) / (2*np.pi**2)
A_s_a = sigma8**2 / s2u_a
print(f"\nVariant A  (R8=8 Mpc/h, k in h/Mpc): s2u={s2u_a:.4e}, A_s={A_s_a:.4e}")

# Check: recompute sigma8 from this A_s
def P_lin(k): return A_s_a * P_unnorm(k)
W2 = 3*(np.sin(k_norm*R8_a) - k_norm*R8_a*np.cos(k_norm*R8_a)) / (k_norm*R8_a)**3
s2_check = np.trapezoid(P_lin(k_norm)*W2**2*k_norm**2, np.log(k_norm)) / (2*np.pi**2)
print(f"  Recovered sigma8 = {np.sqrt(s2_check):.6f}  (should be {sigma8})")

# ── Variant B: R8 = 8/h = 11.36 Mpc  (wrong: treating k as Mpc not h/Mpc) ──
R8_b = 8.0 / h
W_b  = 3*(np.sin(k_norm*R8_b) - k_norm*R8_b*np.cos(k_norm*R8_b)) / (k_norm*R8_b)**3
s2u_b = np.trapezoid(P_unnorm(k_norm)*W_b**2*k_norm**2, np.log(k_norm)) / (2*np.pi**2)
A_s_b = sigma8**2 / s2u_b
print(f"\nVariant B  (R8=8/h Mpc/h = {R8_b:.2f}, confusing units): s2u={s2u_b:.4e}, A_s={A_s_b:.4e}")
print(f"  Ratio A_s_b / A_s_a = {A_s_b/A_s_a:.4f}")

# ── What Rust does: check the Rust IC normalization ───────────────────────────
# Rust sigma8 integral: integrate k^2 P(k) W^2 dk / 2pi^2
# with k in h/Mpc (k_f = 2pi/L, L in Mpc/h), R8=8 Mpc/h
# This should match Variant A.
print(f"\nRust uses: R8=8 Mpc/h, k in h/Mpc → should match Variant A")
print(f"  A_s (Variant A) = {A_s_a:.6e}")

# ── P_lin at reference k values (Variant A) ──────────────────────────────────
print(f"\nP_lin_z0(k) [Variant A, z=0, (Mpc/h)^3]:")
for k_ref in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    print(f"  k={k_ref:.3f} h/Mpc:  P={A_s_a * P_unnorm(k_ref):.4e} (Mpc/h)^3")

# ── Ratio: how much does Rust P_sim exceed P_lin? ────────────────────────────
# From simulation output at a=0.33:
# k=0.026, P_sim=8.3e4, P_lin(a=0.33)=?
def E(a):    return np.sqrt(omega_m/a**3 + omega_l)
def H_phys(a): return h * 100.0 * E(a)
def growth_factor(a):
    def intgd(ap): return 1.0/(ap*H_phys(ap))**3
    Ia, _ = integrate.quad(intgd, 1e-6, a)
    I1, _ = integrate.quad(intgd, 1e-6, 1.0)
    return E(a)*Ia/(E(1.0)*I1)

a_snap = 0.33
D_snap = growth_factor(a_snap)
print(f"\nD(a=0.33) = {D_snap:.6f}")
print(f"P_lin(k=0.026, a=0.33) = {A_s_a * P_unnorm(0.026) * D_snap**2:.4e} (Mpc/h)^3")
print(f"P_lin(k=0.037, a=0.33) = {A_s_a * P_unnorm(0.037) * D_snap**2:.4e} (Mpc/h)^3")
print(f"P_lin(k=0.074, a=0.33) = {A_s_a * P_unnorm(0.074) * D_snap**2:.4e} (Mpc/h)^3")

# ── The Rust sigma8 calculation ───────────────────────────────────────────────
# From zeldovich.rs: it computes sigma8 using same BBKS, same A_s normalization
# Let's see if maybe the Rust code has a factor of h missing somewhere in the
# sigma8 integral that would cause A_s to be inflated.
#
# If Rust computes: s2 = integral k^2 P(k) W^2 dk / 2pi^2
# but uses k_f = 2pi/L where L is box_size in Mpc/h (not Mpc)
# and R8 = 8.0 [Mpc/h], this is CORRECT.
#
# If instead Rust has R8 = 8.0 but interprets k as [1/Mpc] not [h/Mpc],
# that could cause an error.
print(f"\n── Checking if sigma8 computed with wrong k scale ──────────────────")
# If k is actually in h/Mpc but code treated sigma8 integral using k*h instead:
k_wrong = k_norm * h  # if code used k_norm * h in the sigma8 integral
W_wrong = 3*(np.sin(k_wrong*R8_a) - k_wrong*R8_a*np.cos(k_wrong*R8_a)) / (k_wrong*R8_a)**3
# P_unnorm is evaluated at k_wrong (so different shape)
s2u_w = np.trapezoid(P_unnorm(k_wrong)*W_wrong**2*k_wrong**2, np.log(k_wrong)) / (2*np.pi**2)
A_s_w = sigma8**2 / s2u_w
print(f"If sigma8 integral uses k*h: A_s = {A_s_w:.4e}, ratio to correct = {A_s_w/A_s_a:.4f}")

# ── How much would A_s need to change to explain 6.7x excess? ────────────────
print(f"\nTo explain 6.7x P excess: A_s would need to be {6.7:.1f}x larger = {A_s_a*6.7:.4e}")
print(f"  Or sigma8 would need to be sqrt(6.7) = {np.sqrt(6.7):.3f}x larger = {sigma8*np.sqrt(6.7):.4f}")
print(f"\nTo explain 3.1x P excess: A_s would need to be {3.1:.1f}x larger = {A_s_a*3.1:.4e}")
print(f"  Or sigma8 would need to be sqrt(3.1) = {np.sqrt(3.1):.3f}x larger = {sigma8*np.sqrt(3.1):.4f}")

# ── What if the simulation box_size was in Mpc, not Mpc/h? ───────────────────
# k_f = 2pi/250 h/Mpc — but if 250 was treated as Mpc, k_f_wrong = 2pi/(250/h)
# k modes would be h times smaller → P(k) shape would shift
# But this wouldn't change A_s, it would change which k we're reading P at
print(f"\n── Box size unit check ─────────────────────────────────────────────")
print(f"If box=250 Mpc/h: k_fund = {2*np.pi/250:.5f} h/Mpc (correct)")
print(f"If box=250 Mpc:   k_fund = {2*np.pi/250*h:.5f} h/Mpc (wrong, k too small)")
print(f"  At k_sim=0.026: this would correspond to k_true={0.026*h:.4f} h/Mpc")
print(f"  P_lin(k_true=0.018) = {A_s_a*P_unnorm(0.026*h)*D_snap**2:.4e}")
print(f"  P_sim=8.3e4, ratio = {8.3e4 / (A_s_a*P_unnorm(0.026*h)*D_snap**2):.2f}")
