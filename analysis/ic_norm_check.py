"""Mirror the Rust Zel'dovich IC generator exactly and measure P(k).

Uses the same: buf = FFT(Gaussian), delta_k = buf * sqrt(A_s*P/vol),
psi_k = -ik/k^2 * delta_k, psi_real = IFFT(psi_k)/N^3, pos = q + D*psi.
Then CIC-deposit and measure P(k) to find the normalisation.
"""
import numpy as np

# WMAP7 params (lcdm_test.toml)
h        = 0.719
omega_m  = 0.267
omega_l  = 0.733
omega_b  = 0.0449
n_s      = 0.963
sigma8   = 0.801
L        = 64.0  # Mpc/h
N        = 16    # n_p: use 16 for speed (IC generates 16^3 particles)
a_init   = 0.02

gamma = omega_m * h * np.exp(-omega_b * (1.0 + np.sqrt(2*h) / omega_m))

def bbks(k):
    q = np.asarray(k) / gamma
    denom = 1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    # avoid log(1) at q=0
    out = np.where(q < 1e-6, 1.0, np.log(1 + 2.34*q) / (2.34*q * denom**0.25))
    return out

def power_unnorm(k):
    return np.asarray(k)**n_s * bbks(k)**2

k_arr = np.logspace(-4, 2, 10000)
dlnk  = np.diff(np.log(k_arr)).mean()
r = 8.0
w = np.where(k_arr*r < 1e-3, 1.0 - (k_arr*r)**2/10,
             3*(np.sin(k_arr*r) - k_arr*r*np.cos(k_arr*r))/(k_arr*r)**3)
sig8sq_unnorm = np.sum(power_unnorm(k_arr) * k_arr**3 * w**2 * dlnk) / (2*np.pi**2)
A_s = sigma8**2 / sig8sq_unnorm

def growth_factor(a):
    a_arr = np.linspace(1e-4, a, 5000)
    da = np.diff(a_arr).mean()
    E_arr = np.sqrt(omega_m/a_arr**3 + omega_l)
    aH = a_arr * E_arr * h * 100
    integ_a = np.sum(da / aH**3)
    a_arr1 = np.linspace(1e-4, 1.0, 5000)
    da1 = np.diff(a_arr1).mean()
    aH1 = a_arr1 * np.sqrt(omega_m/a_arr1**3 + omega_l) * h * 100
    integ_1 = np.sum(da1 / aH1**3)
    E_a = np.sqrt(omega_m/a**3 + omega_l)
    E_1 = np.sqrt(omega_m + omega_l)
    return (E_a * integ_a) / (E_1 * integ_1)

D_a = growth_factor(a_init)
E_a = np.sqrt(omega_m/a_init**3 + omega_l)
H_a = h * 100 * E_a   # km/s/Mpc
f_a = (omega_m/(a_init**3 * E_a**2))**0.55
print(f"A_s={A_s:.4e}, D({a_init})={D_a:.6f}, H({a_init})={H_a:.1f} km/s/Mpc, f={f_a:.4f}")

# ── Mirror Rust IC generator ──────────────────────────────────────────────────

N3   = N**3
vol  = L**3
k_f  = 2*np.pi / L     # fundamental wavenumber h/Mpc

rng  = np.random.default_rng(42)
buf_real = rng.standard_normal(N3)  # real-valued Gaussian noise

# Forward 3D FFT  (unnormalized, as Rust does)
buf = np.fft.fftn(buf_real.reshape(N, N, N))  # numpy normalises automatically... wait
# numpy.fft.fftn returns FFT with NO normalization: FFT[k] = sum_n x[n] exp(-2pi i k.n/N)
# This matches rustfft's plan_fft_forward which also doesn't normalise.

# Mode frequencies (same as Rust mode_freq)
def mode_freq_arr(n):
    m = np.arange(n, dtype=float)
    m[m > n//2] -= n
    return m

mx = mode_freq_arr(N)
my = mode_freq_arr(N)
mz = mode_freq_arr(N)
MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')
M2 = MX**2 + MY**2 + MZ**2
K_m = k_f * np.sqrt(M2)   # physical wavenumber [h/Mpc]

# Mask zero mode
zero_mask = (M2 == 0)
M2_safe = np.where(zero_mask, 1.0, M2)   # avoid div by zero

# P(k) at each mode
PK = A_s * power_unnorm(np.where(zero_mask, 1e-10, K_m))
PK[zero_mask] = 0.0

# delta_k = buf * sqrt(P/vol)  [RUST NORMALIZATION]
delta_k = buf * np.sqrt(PK / vol)
delta_k[zero_mask] = 0.0

# psi_k[d] = -i * m_d / (k_f * M2) * delta_k
c1 = np.where(zero_mask, 0.0, 1.0 / (k_f * M2_safe))
psi_k = np.stack([
    1j * MX * c1 * delta_k.imag - 1j * MX * c1 * delta_k.real * 0,  # wrong
    # Actually: psi_d = Complex{ delta.im * s, -delta.re * s } where s = m_d*c1
    # = s * (im delta + i * (-re delta)) = s * im(delta) - i * s * re(delta)
    # = Real part: s * im(delta) , Imaginary part: -s * re(delta)
    # = Complex(s * delta.im, -s * delta.re) = -i * s * (delta.re + i*delta.im) = -i*s*delta
    # So psi_d = -i * s * delta = -i * m_d*c1 * delta
    # In numpy: -1j * MX * c1 * delta_k (since c1 is real)
], axis=0)
# Redo properly:
psi_k_x = -1j * MX * c1 * delta_k
psi_k_y = -1j * MY * c1 * delta_k
psi_k_z = -1j * MZ * c1 * delta_k
# Zero mode:
psi_k_x[zero_mask] = 0.0
psi_k_y[zero_mask] = 0.0
psi_k_z[zero_mask] = 0.0

# IFFT (unnormalized) then divide by N^3
inv_n3 = 1.0 / N3
psi_x = np.real(np.fft.ifftn(psi_k_x) * N3 * inv_n3)   # ifftn normalizes by 1/N^3 internally in numpy!
psi_y = np.real(np.fft.ifftn(psi_k_y) * N3 * inv_n3)
psi_z = np.real(np.fft.ifftn(psi_k_z) * N3 * inv_n3)
# WAIT: numpy.fft.ifftn DOES normalize by 1/N^3. But Rust's IFFT does NOT normalize.
# So: rust psi_real = (1/N^3) * ifftn_unnormalized(psi_k)
#                   = (1/N^3) * N^3 * ifftn_normalized(psi_k)   [since ifftn_unnorm = N^3 * ifftn_norm]
#                   = ifftn_normalized(psi_k)
# So python: psi_real = np.real(np.fft.ifftn(psi_k))  (numpy already divides by N^3)
psi_x = np.real(np.fft.ifftn(psi_k_x))
psi_y = np.real(np.fft.ifftn(psi_k_y))
psi_z = np.real(np.fft.ifftn(psi_k_z))

# Particle positions
dx_q = L / N
qx = (np.arange(N)[:, None, None] + 0.5) * dx_q  * np.ones((N, N, N))
qy = (np.arange(N)[None, :, None] + 0.5) * dx_q  * np.ones((N, N, N))
qz = (np.arange(N)[None, None, :] + 0.5) * dx_q  * np.ones((N, N, N))

pos_x = (qx + D_a * psi_x) % L
pos_y = (qy + D_a * psi_y) % L
pos_z = (qz + D_a * psi_z) % L

pos = np.stack([pos_x.ravel(), pos_y.ravel(), pos_z.ravel()], axis=1)
print(f"pos std: {pos.std(axis=0)}  (expected ~{L/np.sqrt(12):.1f} for uniform)")
print(f"psi rms: {psi_x.std():.6f}, {psi_y.std():.6f}, {psi_z.std():.6f} Mpc/h")
print(f"D*psi rms: {D_a*psi_x.std():.6f} Mpc/h (displacement at a={a_init})")

# ── Measure P(k) from the IC positions ───────────────────────────────────────
# CIC density on N-grid (same N as IC)
dx = L / N
rho_cic = np.zeros((N, N, N))
p_sc = pos / dx - 0.5
for xi in p_sc:
    i0, j0, k0 = int(np.floor(xi[0])), int(np.floor(xi[1])), int(np.floor(xi[2]))
    tx, ty, tz = xi[0]-i0, xi[1]-j0, xi[2]-k0
    for di, wx in [(0, 1-tx), (1, tx)]:
        for dj, wy in [(0, 1-ty), (1, ty)]:
            for dk, wz in [(0, 1-tz), (1, tz)]:
                rho_cic[(i0+di)%N, (j0+dj)%N, (k0+dk)%N] += wx*wy*wz
rho_cic *= N3 / N3   # normalize (N particles, N cells → rho_bar=1)
delta_cic = rho_cic - 1.0

# FFT and power
dk_fft = np.fft.fftn(delta_cic)
norm_pk = vol / N3**2   # V/N^6

# Mode [0,0,1] = k_fund
P_sim_001 = norm_pk * abs(dk_fft[0, 0, 1])**2
P_theory_001 = A_s * power_unnorm(np.array([k_f]))[0] * D_a**2
print(f"\nIC P(k_fund={k_f:.4f}) at a={a_init}:")
print(f"  P_sim   = {P_sim_001:.4e} (Mpc/h)^3")
print(f"  P_theory = {P_theory_001:.4e} (Mpc/h)^3")
print(f"  ratio P_theory/P_sim = {P_theory_001/P_sim_001:.1f}")
print(f"  N^3 = {N3},  sqrt(N^3) = {np.sqrt(N3):.1f}")
print(f"  ratio / N^3 = {P_theory_001/P_sim_001/N3:.4f}")

# Also compute at z=0 (no D factor)
P_theory_001_z0 = A_s * power_unnorm(np.array([k_f]))[0]
print(f"\n  P_theory(k_fund, z=0) = {P_theory_001_z0:.4e} (Mpc/h)^3")

# Check delta_rms
print(f"\ndelta rms from CIC: {delta_cic.std():.6f}")
print(f"Expected (sigma8 * D_a): {sigma8 * D_a:.6f}")
print(f"Ratio: {delta_cic.std() / (sigma8 * D_a):.4f}")

# Also compute delta directly from div(psi)
div_psi = np.real(np.fft.ifftn(
    1j * MX * k_f * np.fft.fftn(psi_x) +
    1j * MY * k_f * np.fft.fftn(psi_y) +
    1j * MZ * k_f * np.fft.fftn(psi_z)
))
delta_linear = -D_a * div_psi
print(f"\ndelta rms from -D*div(psi): {delta_linear.std():.6f}")
