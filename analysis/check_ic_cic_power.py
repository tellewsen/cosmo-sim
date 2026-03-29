"""
Generate Zel'dovich ICs in Python (matching Rust IC generator exactly),
apply displacements, assign CIC density, and measure P(k).

Expected: P_CIC(k, a_init) == D(a_init)^2 * P_lin_z0(k)
i.e., ratio P_CIC / P_lin(a_init) == 1.

If the ratio is not 1, the Zel'dovich displacement step is incorrectly normalised.
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
n_p      = 32          # 32^3 for speed; same logic as 64^3
box_size = 250.0
seed     = 42
a_init   = 0.02

def E(a):      return np.sqrt(omega_m/a**3 + omega_l)
def H_phys(a): return h * 100.0 * E(a)

def growth_factor(a):
    def intgd(ap): return 1.0/(ap*H_phys(ap))**3
    Ia, _ = integrate.quad(intgd, 1e-6, a)
    I1, _ = integrate.quad(intgd, 1e-6, 1.0)
    return E(a)*Ia/(E(1.0)*I1)

# ── BBKS (matches Rust) ───────────────────────────────────────────────────
gamma = omega_m * h * np.exp(-omega_b * (1 + np.sqrt(2*h) / omega_m))

def bbks(k):
    k = np.asarray(k, dtype=float)
    scalar = k.ndim == 0
    k = np.atleast_1d(k)
    q = k / gamma
    T = np.where(q < 1e-6, 1.0,
        np.log(1+2.34*q)/(2.34*q) / (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**0.25)
    return float(T[0]) if scalar else T

def P_unnorm(k): return k**n_s * bbks(k)**2

k_norm = np.logspace(-4, 3, 20000)
R8 = 8.0
W  = np.array([3*(np.sin(k*R8)-k*R8*np.cos(k*R8))/(k*R8)**3 for k in k_norm])
s2u = np.trapezoid(P_unnorm(k_norm)*W**2*k_norm**2, np.log(k_norm))/(2*np.pi**2)
A_s = sigma8**2 / s2u

def P_lin_z0(k): return A_s * P_unnorm(k)

# ── Rust xorshift64 + Box-Muller ──────────────────────────────────────────
def xorshift64(state):
    state = int(state) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 7)
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state

n3   = n_p**3
vol  = box_size**3
k_f  = 2*np.pi/box_size

d_a = growth_factor(a_init)
f_a = (omega_m / (a_init**3 * E(a_init)**2))**0.55
h_a = H_phys(a_init)
print(f"D(a_init)/D(1) = {d_a:.6f},  f = {f_a:.4f},  H = {h_a:.2f} km/s/Mpc")
# Use D=1 to apply full z=0 displacement — allows P_CIC to be measured above shot noise
# Expected result: P_CIC / P_lin_z0 ~ 1 at linear scales if normalization is correct
d_a = 1.0
print(f"[Test mode: using D=1 to check normalization against P_lin_z0]")
print(f"gamma = {gamma:.6f},  A_s = {A_s:.4e}")
print(f"n_p = {n_p},  shot noise = {vol/n3:.2f} (Mpc/h)^3")

state = int(seed) | 1
noise = np.empty(n3)
for i in range(n3):
    state = xorshift64(state)
    u1 = (state >> 11) * (1.0 / (1<<53))
    state = xorshift64(state)
    u2 = (state >> 11) * (1.0 / (1<<53))
    u1 = max(u1, 1e-300)
    noise[i] = np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)

buf = noise.reshape(n_p, n_p, n_p)
buf_fft = np.fft.fftn(buf)   # unnormalized, E[|buf_fft[m]|^2] = n3

def mode_freq(i):
    return i if i <= n_p//2 else i - n_p

# ── Build displacement field Ψ in Fourier space ──────────────────────────
psi_fft = np.zeros((3, n_p, n_p, n_p), dtype=complex)

for mi in range(n_p):
    mx = mode_freq(mi)
    for mj in range(n_p):
        my = mode_freq(mj)
        for mk in range(n_p):
            mz = mode_freq(mk)
            if mi == 0 and mj == 0 and mk == 0:
                continue
            m2  = mx*mx + my*my + mz*mz
            k_m = k_f * np.sqrt(m2)
            pk_m  = A_s * P_unnorm(k_m)
            delta = buf_fft[mi, mj, mk] * np.sqrt(pk_m * n3 / vol)
            c1 = 1.0 / (k_f * m2)
            # psi_d = -i * m_d * c1 * delta  -->  real(IFFT * 1/n3)
            for dim, md in enumerate([mx, my, mz]):
                s = md * c1
                # -i*(re+i*im)*s = im*s + i*(-re*s)
                psi_fft[dim, mi, mj, mk] = complex(delta.imag * s, -delta.real * s)

# ── IFFT and normalise by 1/n3 ───────────────────────────────────────────
psi = np.zeros((3, n_p, n_p, n_p))
for dim in range(3):
    psi[dim] = np.fft.ifftn(psi_fft[dim]).real / n3  # matches Rust: *inv_n3 after IFFT

# Note: np.fft.ifftn already divides by n3, so we divide by n3 again to match Rust
# (Rust uses forward FFT then multiplies by 1/n3, equivalent to ifftn/1 × 1/n3 = ifftn)
# Actually Rust does: fft3d(forward) then x *= inv_n3 -- let's verify
# np.fft.ifftn = (1/n3) * FFT_forward_of_conjugate... no.
# Rust IFFT = forward FFT of input, then /n3.  This equals NUMPY ifftn * n3 / n3 = numpy ifftn.
# Actually numpy ifftn = (1/n3) * Σ x[k] e^{+2πi jk/n}, while
# numpy fftn   = Σ x[j] e^{-2πi jk/n} (unnormalized).
# Rust IFFT = plan_fft_inverse (which computes Σ x[k] e^{+2πi jk/n}, unnormalized) then / n3.
# So Rust IFFT(x) = numpy ifftn(x).  The extra /n3 in check above was wrong.
# psi[dim] = ifftn(psi_fft[dim]).real  (no extra /n3 needed)
for dim in range(3):
    psi[dim] = np.fft.ifftn(psi_fft[dim]).real   # correct: matches Rust plan_fft_inverse + /n3

# ── Place particles and apply displacements ───────────────────────────────
dx_q = box_size / n_p
pos = np.zeros((3, n3))
idx = 0
for i in range(n_p):
    for j in range(n_p):
        for k in range(n_p):
            for dim, coord in enumerate([i, j, k]):
                psi_val = psi[dim, i, j, k]
                pos[dim, idx] = ((coord + 0.5)*dx_q + d_a * psi_val) % box_size
            idx += 1

print(f"\nDisplacement RMS: {np.std(pos[0] - (np.arange(n3)//n_p//n_p % n_p + 0.5)*dx_q):.4f} Mpc/h")

# ── CIC density on n_g = n_p grid ────────────────────────────────────────
n_g = n_p
rho = np.zeros((n_g, n_g, n_g))
dx_g = box_size / n_g
for p in range(n3):
    for dim in range(3):
        xp = pos[dim, p]
        cell = xp / dx_g
        c0 = int(cell) % n_g
        c1 = (c0 + 1) % n_g
        w1 = cell - int(cell)
        w0 = 1.0 - w1
        # handled below for 3D
    # 3D CIC
    x, y, z = pos[0,p]/dx_g, pos[1,p]/dx_g, pos[2,p]/dx_g
    i0, j0, k0 = int(x)%n_g, int(y)%n_g, int(z)%n_g
    i1, j1, k1 = (i0+1)%n_g, (j0+1)%n_g, (k0+1)%n_g
    wx, wy, wz = x-int(x), y-int(y), z-int(z)
    rho[i0,j0,k0] += (1-wx)*(1-wy)*(1-wz)
    rho[i1,j0,k0] += wx*(1-wy)*(1-wz)
    rho[i0,j1,k0] += (1-wx)*wy*(1-wz)
    rho[i0,j0,k1] += (1-wx)*(1-wy)*wz
    rho[i1,j1,k0] += wx*wy*(1-wz)
    rho[i1,j0,k1] += wx*(1-wy)*wz
    rho[i0,j1,k1] += (1-wx)*wy*wz
    rho[i1,j1,k1] += wx*wy*wz

# Normalise: mean = 1 (= n3/n3 scale)
scale = n_g**3 / n3
rho *= scale
delta_cic = rho - 1.0

# ── Power spectrum from CIC density ──────────────────────────────────────
delta_fft = np.fft.fftn(delta_cic)
norm = vol / n_g**6

pk_sum = {}
nm_bins = {}
for mi in range(n_g):
    mx = mode_freq(mi)
    for mj in range(n_g):
        my = mode_freq(mj)
        for mk in range(n_g):
            mz = mode_freq(mk)
            if mi == 0 and mj == 0 and mk == 0: continue
            m2  = mx*mx + my*my + mz*mz
            k_m = k_f * np.sqrt(m2)
            p_est = norm * abs(delta_fft[mi,mj,mk])**2
            key = round(k_m, 5)
            pk_sum[key] = pk_sum.get(key, 0.0) + p_est
            nm_bins[key] = nm_bins.get(key, 0) + 1

shot = vol / n3
print(f"\nP_CIC(k, a_init) / P_lin(k, a_init)  [should be ~1]:")
print(f"Shot noise = {shot:.2f} (Mpc/h)^3")
print(f"{'k[h/Mpc]':>10}  {'P_CIC':>12}  {'P_lin(a)':>12}  {'ratio':>8}  N")
for k_key in sorted(pk_sum.keys()):
    n_modes = nm_bins[k_key]
    if n_modes < 4: continue
    p_avg = (pk_sum[k_key] / n_modes) - shot   # shot-noise subtraction
    p_lin_a = P_lin_z0(k_key) * d_a**2
    # Print all bins with signal (positive after shot subtraction or raw P if signal expected large)
    print(f"{k_key:10.4f}  {p_avg:12.3e}  {p_lin_a:12.3e}  {p_avg/p_lin_a:8.3f}  {n_modes}")
    if k_key > 0.3: break
