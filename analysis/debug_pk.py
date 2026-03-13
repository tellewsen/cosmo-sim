"""Quick diagnostic: read a snapshot and manually compute P(k) to verify the Rust estimator."""

import struct
import sys
import numpy as np

snapshot = sys.argv[1] if len(sys.argv) > 1 else "output_lcdm_test/snapshot_0004.bin"
n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 64
L = float(sys.argv[3]) if len(sys.argv) > 3 else 64.0

with open(snapshot, "rb") as f:
    step   = struct.unpack("<Q", f.read(8))[0]
    a      = struct.unpack("<d", f.read(8))[0]
    n_part = struct.unpack("<Q", f.read(8))[0]
    data   = np.frombuffer(f.read(n_part * 6 * 8), dtype="<f8").reshape(n_part, 6)

pos = data[:, :3]
vel = data[:, 3:]
print(f"step={step}  a={a:.4f}  n_part={n_part}")
print(f"pos x∈[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
print(f"pos std: {pos.std(axis=0)}")
print(f"vel rms: {np.sqrt((vel**2).mean(axis=0))}")

# CIC density field
dx = L / n_grid
rho = np.zeros((n_grid, n_grid, n_grid))
p_sc = pos / dx - 0.5  # shift to cell-corner convention
for xi in p_sc:
    i0 = int(np.floor(xi[0]))
    j0 = int(np.floor(xi[1]))
    k0 = int(np.floor(xi[2]))
    tx = xi[0] - i0
    ty = xi[1] - j0
    tz = xi[2] - k0
    for di, wx in [(0, 1 - tx), (1, tx)]:
        for dj, wy in [(0, 1 - ty), (1, ty)]:
            for dk, wz in [(0, 1 - tz), (1, tz)]:
                rho[(i0 + di) % n_grid, (j0 + dj) % n_grid, (k0 + dk) % n_grid] += wx * wy * wz

rho = rho * (n_grid ** 3) / n_part
delta = rho - 1.0
print(f"\ndelta: min={delta.min():.3f}  max={delta.max():.3f}  std={delta.std():.4f}  mean={delta.mean():.2e}")

# 3D FFT
dk     = np.fft.fftn(delta)
N3     = n_grid ** 3
norm   = L ** 3 / N3 ** 2
shot   = L ** 3 / n_part

print(f"\nnorm = V/N^6 = {norm:.3e}")
print(f"shot noise = V/N_part = {shot:.3f}  (Mpc/h)^3")

# A few individual modes
for mx, my, mz in [(0,0,1),(1,0,0),(2,0,0),(4,0,0)]:
    val = norm * abs(dk[mx, my, mz])**2
    print(f"  P[{mx},{my},{mz}] = {val:.3e}  (shot={shot:.2f})")

# Logarithmic binning
import numpy as np
k_fund = 2 * np.pi / L
k_nyq  = np.pi * n_grid / L
n_bins = 20

kx = np.fft.fftfreq(n_grid, d=1.0 / n_grid)  # integer wavenumbers
ky = kx.copy()
kz = kx.copy()
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
K2  = KX**2 + KY**2 + KZ**2
K   = np.sqrt(K2) * k_fund

mask   = K > 0
pk_raw = norm * np.abs(dk)**2

bins    = np.logspace(np.log10(k_fund), np.log10(k_nyq), n_bins + 1)
k_cens  = 0.5 * (bins[:-1] + bins[1:])
pk_bins = np.zeros(n_bins)
nm_bins = np.zeros(n_bins, dtype=int)
for b in range(n_bins):
    m = mask & (K >= bins[b]) & (K < bins[b + 1])
    nm_bins[b] = m.sum()
    if nm_bins[b] > 0:
        pk_bins[b] = pk_raw[m].mean() - shot

print(f"\nk [h/Mpc]   P(k) [(Mpc/h)^3]   N_modes")
for b in range(n_bins):
    if nm_bins[b] > 0:
        print(f"  {k_cens[b]:.4f}     {pk_bins[b]:.4e}     {nm_bins[b]}")
