"""
Replicate Rust Zel'dovich IC generator and measure P(k) of the generated
delta field to verify IC amplitude normalization.

Reproduces the Rust xorshift64 RNG + Box-Muller + 3D FFT + BBKS amplitude
step-by-step and computes P_IC(k) = <|delta_k|^2> * V / N^6.

Expected result: P_IC(k) == P_lin_z0(k) (before growth factor application).
"""
import numpy as np
from scipy import integrate

# ── Cosmology (winther2015_lcdm_test.toml) ────────────────────────────────
omega_m  = 0.269
omega_l  = 0.731
h        = 0.704
n_s      = 0.966
sigma8   = 0.8
omega_b  = 0.0455
n_p      = 64
box_size = 250.0
seed     = 42
a_init   = 0.02

def E(a):       return np.sqrt(omega_m/a**3 + omega_l)
def H_phys(a):  return h * 100.0 * E(a)

def growth_factor(a):
    def intgd(ap): return 1.0/(ap*H_phys(ap))**3
    Ia, _ = integrate.quad(intgd, 1e-6, a)
    I1, _ = integrate.quad(intgd, 1e-6, 1.0)
    return E(a)*Ia/(E(1.0)*I1)

# ── BBKS + normalization (matches Rust zeldovich.rs) ──────────────────────
gamma = omega_m * h * np.exp(-omega_b * (1 + np.sqrt(2*h) / omega_m))
print(f"Sugiyama gamma = {gamma:.6f}")

def bbks_scalar(k):
    """BBKS transfer function for scalar k."""
    q = k / gamma
    if q < 1e-6: return 1.0
    d = 1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    return np.log(1+2.34*q)/(2.34*q*d**0.25)

def bbks(k):
    """BBKS for scalar or array k."""
    k = np.asarray(k, dtype=float)
    scalar = k.ndim == 0
    k = np.atleast_1d(k)
    q = k / gamma
    T = np.where(q < 1e-6, 1.0,
        np.log(1+2.34*q)/(2.34*q) / (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**0.25)
    return float(T[0]) if scalar else T

def P_unnorm(k): return k**n_s * bbks(k)**2

k_norm = np.logspace(-4, 3, 20000)
R8  = 8.0
W   = np.array([3*(np.sin(k*R8)-k*R8*np.cos(k*R8))/(k*R8)**3 for k in k_norm])
s2u = np.trapezoid(P_unnorm(k_norm)*W**2*k_norm**2, np.log(k_norm))/(2*np.pi**2)
A_s = sigma8**2 / s2u
print(f"A_s = {A_s:.4e}")

def P_lin_z0(k): return A_s * P_unnorm(k)

# ── Rust xorshift64 RNG ────────────────────────────────────────────────────
def xorshift64(state):
    state = int(state)
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 7)
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

n3   = n_p**3
vol  = box_size**3
k_f  = 2*np.pi/box_size

d_a  = growth_factor(a_init)
print(f"D(a_init)/D(1) = {d_a:.6f}")

# ── Generate Gaussian noise (same as Rust: use FIRST of Box-Muller pair) ──
state = int(seed) | 1
noise = np.empty(n3)
for i in range(n3):
    state = xorshift64(state)
    u1 = (state >> 11) * (1.0 / (1<<53))
    state = xorshift64(state)
    u2 = (state >> 11) * (1.0 / (1<<53))
    u1 = max(u1, 1e-300)
    r  = np.sqrt(-2*np.log(u1))
    noise[i] = r * np.cos(2*np.pi*u2)   # first variate

# ── Forward 3D FFT of noise (matches Rust fft3d) ──────────────────────────
buf = noise.reshape(n_p, n_p, n_p)
buf_fft = np.fft.fftn(buf)   # unnormalized forward FFT, E[|buf_fft[m]|^2] = n3

# ── Scale each mode by sqrt(P_lin_z0(k) * n3 / vol) ──────────────────────
# This gives delta[k] with E[|delta[k]|^2] = P_lin_z0(k) * n3^2/vol
# Power spectrum: P_hat = vol/n3^2 * |delta|^2 --> should equal P_lin_z0

def mode_freq(i):
    return i if i <= n_p//2 else i - n_p

pk_sum = {}
nm     = {}
for mi in range(n_p):
    mx = mode_freq(mi)
    for mj in range(n_p):
        my = mode_freq(mj)
        for mk in range(n_p):
            mz = mode_freq(mk)
            if mi == 0 and mj == 0 and mk == 0: continue
            m2  = mx*mx + my*my + mz*mz
            k_m = k_f * np.sqrt(m2)
            pk_m  = A_s * P_unnorm(k_m)
            delta = buf_fft[mi, mj, mk] * np.sqrt(pk_m * n3 / vol)
            # Power per mode: V/N^6 * |delta|^2
            p_est = abs(delta)**2 * vol / n3**2
            key   = round(k_m, 5)
            pk_sum[key] = pk_sum.get(key, 0.0) + p_est
            nm[key]     = nm.get(key, 0) + 1

print()
print("IC delta-field P(k) vs P_lin_z0 (ratio should be 1.0):")
print(f"{'k[h/Mpc]':>10}  {'P_IC_delta':>14}  {'P_lin_z0':>14}  {'ratio':>8}  N_modes")
for k_key in sorted(pk_sum.keys()):
    n_modes = nm[k_key]
    if n_modes < 4: continue
    p_avg = pk_sum[k_key] / n_modes
    p_lin = P_lin_z0(k_key)
    print(f"{k_key:10.4f}  {p_avg:14.3e}  {p_lin:14.3e}  {p_avg/p_lin:8.4f}  {n_modes}")
    if k_key > 0.15: break
