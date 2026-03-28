"""
Read a Rust binary snapshot, compute P(k) in Python, and compare to:
1. The Rust-written power_spectrum_NNNN.txt
2. The linear theory P_lin(k, a)

Snapshot format: [step:u64][a:f64][n_part:u64] then per particle [x,y,z,vx,vy,vz] as f64.

Usage:
  uv run python analysis/check_snapshot_pk.py output_w15_lcdm_fixed/snapshot_0000.bin
"""
import struct
import sys
import numpy as np
from scipy import integrate

# ── Cosmology (winther2015_lcdm_test.toml) ────────────────────────────────────
omega_m = 0.269
omega_l = 0.731
h       = 0.704
n_s     = 0.966
sigma8  = 0.8
omega_b = 0.0455
box_size = 250.0   # Mpc/h

def E(a):    return np.sqrt(omega_m/a**3 + omega_l)
def H_phys(a): return h * 100.0 * E(a)
def growth_factor(a):
    def intgd(ap): return 1.0/(ap*H_phys(ap))**3
    Ia, _ = integrate.quad(intgd, 1e-6, a)
    I1, _ = integrate.quad(intgd, 1e-6, 1.0)
    return E(a)*Ia/(E(1.0)*I1)

gamma = omega_m * h * np.exp(-omega_b * (1 + np.sqrt(2*h) / omega_m))
def bbks(k):
    k = np.asarray(k, dtype=float)
    q = k / gamma
    return np.where(q < 1e-6, 1.0,
        np.log(1+2.34*q)/(2.34*q) / (1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**0.25)

def P_unnorm(k): return k**n_s * bbks(k)**2

k_norm = np.logspace(-4, 3, 20000)
R8 = 8.0
W  = 3*(np.sin(k_norm*R8) - k_norm*R8*np.cos(k_norm*R8)) / (k_norm*R8)**3
s2u = np.trapezoid(P_unnorm(k_norm)*W**2*k_norm**3, np.log(k_norm)) / (2*np.pi**2)
A_s = sigma8**2 / s2u
def P_lin_z0(k): return A_s * P_unnorm(k)

# ── Read snapshot ─────────────────────────────────────────────────────────────
snap_path = sys.argv[1] if len(sys.argv) > 1 else "output_w15_lcdm_fixed/snapshot_0000.bin"
print(f"Reading snapshot: {snap_path}")

with open(snap_path, "rb") as f:
    step  = struct.unpack("<Q", f.read(8))[0]
    a_snap = struct.unpack("<d", f.read(8))[0]
    n_part = struct.unpack("<Q", f.read(8))[0]
    # Each particle: 6 doubles = 48 bytes
    raw = f.read()

print(f"step={step}, a={a_snap:.5f}, n_part={n_part}")

# Parse particle positions (first 3 floats per particle)
n_float = 6
data = np.frombuffer(raw, dtype=np.float64).reshape(n_part, n_float)
pos = data[:, :3]   # x, y, z in Mpc/h
print(f"pos range: x=[{pos[:,0].min():.2f}, {pos[:,0].max():.2f}] Mpc/h")

# ── CIC density ───────────────────────────────────────────────────────────────
n_g = int(round(n_part**(1/3)))
print(f"Using n_g={n_g}^3={n_g**3} grid")
assert n_g**3 == n_part, f"n_part={n_part} not a perfect cube"

vol = box_size**3
rho = np.zeros((n_g, n_g, n_g))
dx_g = box_size / n_g

for p in range(n_part):
    xp, yp, zp = pos[p]
    x, y, z = xp/dx_g, yp/dx_g, zp/dx_g
    i0, j0, k0 = int(x)%n_g, int(y)%n_g, int(z)%n_g
    i1, j1, k1 = (i0+1)%n_g, (j0+1)%n_g, (k0+1)%n_g
    # Shift: CIC corner is at cell center (like Rust: u = xi*dx_inv - 0.5)
    # Actually Rust uses: u = x * dx_inv - 0.5, then floor(u)
    # So the first cell boundary is at x=0.5*dx_g (not x=0)
    wx = x - int(x)
    wy = y - int(y)
    wz = z - int(z)
    rho[i0,j0,k0] += (1-wx)*(1-wy)*(1-wz)
    rho[i1,j0,k0] += wx*(1-wy)*(1-wz)
    rho[i0,j1,k0] += (1-wx)*wy*(1-wz)
    rho[i0,j0,k1] += (1-wx)*(1-wy)*wz
    rho[i1,j1,k0] += wx*wy*(1-wz)
    rho[i1,j0,k1] += wx*(1-wy)*wz
    rho[i0,j1,k1] += (1-wx)*wy*wz
    rho[i1,j1,k1] += wx*wy*wz

# Normalize: mean = 1
scale = n_g**3 / n_part
rho *= scale
delta = rho - 1.0
print(f"delta: mean={delta.mean():.2e}, std={delta.std():.4f}")

# ── P(k) estimator: norm = V/N^6, same as Rust ───────────────────────────────
delta_fft = np.fft.fftn(delta)
norm = vol / (n_g**3)**2

n3  = n_g**3
k_f = 2*np.pi / box_size
shot = vol / n_part

def mode_freq(i):
    return i if i <= n_g//2 else i - n_g

# Log-spaced bins from k_fund to k_nyq (40 bins, same as Rust)
k_nyq = np.pi * n_g / box_size
n_bins = 40
lnk0 = np.log(k_f)
lnk1 = np.log(k_nyq)
dlnk = (lnk1 - lnk0) / n_bins

pk_sum  = np.zeros(n_bins)
nmodes  = np.zeros(n_bins, dtype=int)

for mi in range(n_g):
    mx = mode_freq(mi)
    for mj in range(n_g):
        my = mode_freq(mj)
        for mk in range(n_g):
            mz = mode_freq(mk)
            if mi==0 and mj==0 and mk==0: continue
            m2 = mx*mx + my*my + mz*mz
            k_m = k_f * np.sqrt(m2)
            lnk = np.log(k_m)
            b = int((lnk - lnk0) / dlnk)
            if b < 0 or b >= n_bins: continue
            pk_sum[b] += norm * abs(delta_fft[mi,mj,mk])**2
            nmodes[b] += 1

k_bins = np.exp(lnk0 + (np.arange(n_bins) + 0.5) * dlnk)
pk_py = np.where(nmodes > 0, pk_sum / nmodes - shot, 0.0)

# ── Load Rust power spectrum for same snapshot ────────────────────────────────
import os
pk_dir = os.path.dirname(snap_path)
snap_idx = int(os.path.basename(snap_path).split("_")[1].split(".")[0])
rust_pk_path = f"{pk_dir}/power_spectrum_{snap_idx:04d}.txt"
rust_k, rust_pk = [], []
if os.path.exists(rust_pk_path):
    with open(rust_pk_path) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 2:
                rust_k.append(float(parts[0]))
                rust_pk.append(float(parts[1]))
    print(f"Loaded Rust P(k): {rust_pk_path}")
else:
    print(f"No Rust P(k) file found at {rust_pk_path}")

D_snap = growth_factor(a_snap)
print(f"\nD(a={a_snap:.4f}) = {D_snap:.6f}")
print(f"A_s = {A_s:.4e}")
print(f"Shot noise = {shot:.2f} (Mpc/h)^3\n")

print(f"{'k[h/Mpc]':>10} {'P_py':>12} {'P_rust':>12} {'P_lin(a)':>12} {'py/lin':>8} {'rust/lin':>8} {'N'}")
for b in range(n_bins):
    if nmodes[b] < 4: continue
    k_b = k_bins[b]
    p_py = pk_py[b]
    p_lin_a = P_lin_z0(k_b) * D_snap**2

    # Nearest Rust P(k) bin
    p_rust = float('nan')
    if rust_k:
        diffs = [abs(rk - k_b) for rk in rust_k]
        idx = diffs.index(min(diffs))
        if min(diffs) < 0.01 * k_b:
            p_rust = rust_pk[idx]

    print(f"{k_b:10.4f} {p_py:12.3e} {p_rust:12.3e} {p_lin_a:12.3e} "
          f"{p_py/p_lin_a:8.3f} {p_rust/p_lin_a:8.3f} {nmodes[b]}")
    if k_b > 0.25: break
