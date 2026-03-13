"""Check IC amplitude: what delta_rms should the IC generator produce at a=0.02?

Computes the theoretical real-space delta_rms for the WMAP7 params on a 64^3 grid
and compares to what the simulation shows after propagating to z=0.
"""
import numpy as np

# WMAP7 cosmology
h       = 0.719
omega_m = 0.267
omega_l = 0.733
omega_b = 0.0449
n_s     = 0.963
sigma8  = 0.801
L       = 64.0   # Mpc/h
N       = 64

# BBKS
gamma = omega_m * h * np.exp(-omega_b * (1.0 + np.sqrt(2*h) / omega_m))
def bbks(k):
    q = k / gamma
    if q < 1e-6: return 1.0
    d = 1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    return np.log(1 + 2.34*q) / (2.34*q * d**0.25)
bbks_v = np.vectorize(bbks)

def power_unnorm(k):
    return k**n_s * bbks_v(k, )**2

# Normalise
k_arr = np.logspace(-4, 2, 10000)
dlnk  = np.diff(np.log(k_arr)).mean()
r = 8.0
w = np.where(k_arr*r < 1e-3, 1.0, 3*(np.sin(k_arr*r) - k_arr*r*np.cos(k_arr*r))/(k_arr*r)**3)
sig8sq_unnorm = np.sum(power_unnorm(k_arr) * k_arr**3 * w**2 * dlnk) / (2*np.pi**2)
A_s = sigma8**2 / sig8sq_unnorm
print(f"A_s = {A_s:.4e}")

# Growth factor at a=0.02
def E(a): return np.sqrt(omega_m/a**3 + omega_l)
def growth_factor(a):
    a_arr = np.linspace(1e-4, a, 5000)
    da = np.diff(a_arr).mean()
    aH = a_arr * E(a_arr) * h * 100
    integ_a = np.sum(da / aH**3)
    a1_arr = np.linspace(1e-4, 1.0, 5000)
    da1 = np.diff(a1_arr).mean()
    aH1 = a1_arr * E(a1_arr) * h * 100
    integ_1 = np.sum(da1 / aH1**3)
    return (E(a) * integ_a) / (E(1.0) * integ_1)

D_init = growth_factor(0.02)
D_z0   = growth_factor(1.0)
print(f"D(0.02) = {D_init:.6f},  D(1.0) = {D_z0:.6f}")

# Expected delta_rms at a=0.02 on the 64^3 grid (real space CIC, no smoothing)
# sigma^2 = (1/(2pi^2)) int P(k,a) k^2 dk from k_fund to k_nyq
k_fund = 2*np.pi / L
k_nyq  = np.pi * N / L
print(f"k_fund = {k_fund:.4f}, k_nyq = {k_nyq:.4f} h/Mpc")

# Fine k grid
k_int = np.logspace(np.log10(k_fund), np.log10(k_nyq), 5000)
P_z0 = A_s * power_unnorm(k_int)
P_init = D_init**2 * P_z0

sigma2_init = np.trapezoid(P_init * k_int**2, np.log(k_int)) / (2*np.pi**2)
sigma2_z0   = np.trapezoid(P_z0 * k_int**2, np.log(k_int)) / (2*np.pi**2)

print(f"\nExpected real-space delta_rms (no smoothing, k_fund to k_nyq):")
print(f"  at a=0.02: {np.sqrt(sigma2_init):.4f}")
print(f"  at z=0:    {np.sqrt(sigma2_z0):.4f}")

print(f"\nSimulation shows delta_rms at z=0: 0.0304")
print(f"Expected at z=0: {np.sqrt(sigma2_z0):.4f}")
print(f"Ratio (expected/sim): {np.sqrt(sigma2_z0)/0.0304:.2f}x")

# Also check: what is the IC delta_rms in the Fourier space of the IC generator?
# The IC uses: delta_code[k] = buf[k] * sqrt(A_s * P(k) / vol)
# where E[|buf[k]|^2] = N^3 (unnormalized FFT of N^3 iid N(0,1))
# After IFT with 1/N^3 factor: psi_real[c] = (1/N^3) IFFT(psi_k)[c]
# Real-space delta = D * ik.psi_real, and:
# FFT(delta)[k] = D * ik.psi_k = D * delta_code[k]
# P_measured = vol/N^6 * |D * delta_code[k]|^2

# Expected E[|delta_code[k]|^2] = N^3 * A_s * P(k) / vol
# E[P_measured(k)] = vol/N^6 * D^2 * N^3 * A_s * P(k) / vol = D^2 * A_s * P(k) / N^3

N3 = N**3
vol = L**3
print(f"\nNormalization check:")
print(f"  N^3 = {N3}")
print(f"  vol = {vol:.0f} (Mpc/h)^3")
print(f"  Expected P_measured(k=0.1, z=0) from IC formula = A_s*P(0.1)/N^3 = {A_s*power_unnorm(np.array([0.1]))[0]/N3:.4e}")
print(f"  Expected P_theory(k=0.1, z=0)                   = A_s*P(0.1)     = {A_s*power_unnorm(np.array([0.1]))[0]:.4e}")
print(f"  Simulation P_raw(k_fund, z=0) = 0.324 (Mpc/h)^3")
print(f"  Theory P(k_fund, z=0) = {A_s*power_unnorm(np.array([k_fund]))[0]:.4e} (Mpc/h)^3")

print(f"\n{'─'*60}")
print(f"HYPOTHESIS: IC normalization missing factor N^3 = {N3}")
print(f"  If P_measured = P_expected / N^3, then delta_rms_measured = delta_rms_expected / sqrt(N^3) = delta_rms_expected / {np.sqrt(N3):.0f}")
print(f"  Expected delta_rms(z=0) / 512 = {np.sqrt(sigma2_z0)/np.sqrt(N3):.6f}")
print(f"  vs simulation 0.030 — ratio = {np.sqrt(sigma2_z0)/np.sqrt(N3)/0.030:.2f}")

print(f"\n  If P_measured = P_expected * N^3, delta_rms_measured = delta_rms_expected * {np.sqrt(N3):.0f}")
