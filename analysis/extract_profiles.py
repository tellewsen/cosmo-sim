"""
Post-process RUSTIS snapshots to extract stacked halo profiles.

Replicates the logic in src/hydro/profiles.rs (FoF + stacked_profiles +
write_profiles) in Python, so profiles can be computed from existing
snapshot binaries without rerunning the simulation.

Snapshot binary format (little-endian):
  snapshot_NNNN.bin:
    step:   u64
    a:      f64
    n_part: u64
    then n_part × (pos[3], vel[3]) all f64

  gas_snapshot_NNNN.bin:
    step: u64
    a:    f64
    n3:   u64
    then n3 × (rho, vx, vy, vz, u) all f64

Usage:
    uv run python analysis/extract_profiles.py \
        --dirs output_ellewsen_lcdm output_ellewsen_sym_a \
        --snap 0000
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_snapshot(path: Path):
    """Return (step, a, pos[N,3], vel[N,3]) from a DM snapshot binary."""
    with open(path, "rb") as f:
        step, = struct.unpack("<Q", f.read(8))
        a,    = struct.unpack("<d", f.read(8))
        n,    = struct.unpack("<Q", f.read(8))
        data = np.frombuffer(f.read(n * 6 * 8), dtype="<f8").reshape(n, 6)
    pos = data[:, :3].copy()
    vel = data[:, 3:].copy()
    return step, a, pos, vel


def read_gas_snapshot(path: Path):
    """Return (step, a, gas_rho[n3], gas_u[n3]) from a gas snapshot binary."""
    with open(path, "rb") as f:
        step, = struct.unpack("<Q", f.read(8))
        a,    = struct.unpack("<d", f.read(8))
        n3,   = struct.unpack("<Q", f.read(8))
        data = np.frombuffer(f.read(n3 * 5 * 8), dtype="<f8").reshape(n3, 5)
    gas_rho = data[:, 0].copy()
    gas_u   = data[:, 4].copy()
    return step, a, gas_rho, gas_u


# ── FoF ───────────────────────────────────────────────────────────────────────

def friends_of_friends(pos: np.ndarray, b: float, box_size: float) -> list[list[int]]:
    """
    Cell-list FoF. Mirrors profiles.rs::friends_of_friends exactly:
      - n_cells capped at ceil(N^(1/3)) + 1 to avoid huge cell grids for small b
      - Periodic minimum-image distance
      - Union-find with path compression
    """
    n = len(pos)
    if n == 0:
        return []

    n_cells_ideal = int(box_size / b)
    n_cells_cap   = int(np.ceil(n ** (1 / 3))) + 1
    n_cells = max(1, min(n_cells_ideal, n_cells_cap))
    cell_size = box_size / n_cells

    # Assign particles to cells
    ci = (pos[:, 0] / cell_size).astype(int) % n_cells
    cj = (pos[:, 1] / cell_size).astype(int) % n_cells
    ck = (pos[:, 2] / cell_size).astype(int) % n_cells
    flat = ci * n_cells * n_cells + cj * n_cells + ck

    cells = [[] for _ in range(n_cells ** 3)]
    for i in range(n):
        cells[flat[i]].append(i)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    b2 = b * b

    for ci_flat in range(n_cells ** 3):
        if not cells[ci_flat]:
            continue
        ci_i = ci_flat // (n_cells * n_cells)
        ci_j = (ci_flat // n_cells) % n_cells
        ci_k = ci_flat % n_cells

        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    ni = (ci_i + di + n_cells - 1) % n_cells
                    nj = (ci_j + dj + n_cells - 1) % n_cells
                    nk = (ci_k + dk + n_cells - 1) % n_cells
                    ni_flat = ni * n_cells * n_cells + nj * n_cells + nk
                    if ni_flat < ci_flat:
                        continue

                    for pos_in_cell, i in enumerate(cells[ci_flat]):
                        start = pos_in_cell + 1 if ni_flat == ci_flat else 0
                        for j in cells[ni_flat][start:]:
                            dx = pos[i, 0] - pos[j, 0]
                            if abs(dx) > box_size * 0.5:
                                dx -= np.copysign(box_size, dx)
                            dy = pos[i, 1] - pos[j, 1]
                            if abs(dy) > box_size * 0.5:
                                dy -= np.copysign(box_size, dy)
                            dz = pos[i, 2] - pos[j, 2]
                            if abs(dz) > box_size * 0.5:
                                dz -= np.copysign(box_size, dz)
                            if dx*dx + dy*dy + dz*dz < b2:
                                union(i, j)

    from collections import defaultdict
    group_map = defaultdict(list)
    for i in range(n):
        group_map[find(i)].append(i)
    return list(group_map.values())


# ── Stacked profiles ──────────────────────────────────────────────────────────

N_BINS     = 20
R_MIN_FRAC = 0.1
R_MAX_FRAC = 3.0
U_TO_T     = 0.59 / 8.314e-3   # K / (km/s)²


def r200(n_members: int, m_part: float, h: float) -> float:
    rho_crit = 2.775e11 * h * h
    m_halo   = n_members * m_part
    return (3.0 * m_halo / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)


def group_centre(pos: np.ndarray, members: list[int]) -> np.ndarray:
    return pos[members].mean(axis=0)


def stacked_profiles(
    dm_pos:  np.ndarray,
    gas_rho: np.ndarray,
    gas_u:   np.ndarray,
    n_grid:  int,
    box_size: float,
    m_part:  float,
    h:       float,
    m_min:   float,
    m_max:   float,
    groups:  list[list[int]],
):
    dx       = box_size / n_grid
    log_rmin = np.log(R_MIN_FRAC)
    log_rmax = np.log(R_MAX_FRAC)
    dlog     = (log_rmax - log_rmin) / N_BINS

    rho_gas_sum = np.zeros(N_BINS)
    t_sum       = np.zeros(N_BINS)
    rho_dm_sum  = np.zeros(N_BINS)
    counts      = np.zeros(N_BINS, dtype=int)
    n_halos     = 0

    for group in groups:
        m_halo = len(group) * m_part
        if m_halo < m_min or m_halo > m_max:
            continue
        n_halos += 1

        centre  = group_centre(dm_pos, group)
        r200_h  = r200(len(group), m_part, h)

        # DM particles in radial bins
        gpos = dm_pos[group]
        dr   = gpos - centre
        for d in range(3):
            mask = np.abs(dr[:, d]) > box_size * 0.5
            dr[mask, d] -= np.copysign(box_size, dr[mask, d])
        r = np.sqrt((dr ** 2).sum(axis=1)) / r200_h
        mask = (r >= R_MIN_FRAC) & (r < R_MAX_FRAC)
        bins = ((np.log(r[mask]) - log_rmin) / dlog).astype(int)
        bins = bins[bins < N_BINS]
        for b in bins:
            rho_dm_sum[b] += 1

        # Gas cells around halo
        r_max_phys   = R_MAX_FRAC * r200_h
        cells_per_side = int(np.ceil(2.0 * r_max_phys / dx)) + 2
        ci0 = int(centre[0] / dx) - cells_per_side // 2
        cj0 = int(centre[1] / dx) - cells_per_side // 2
        ck0 = int(centre[2] / dx) - cells_per_side // 2

        di_arr = np.arange(cells_per_side)
        dj_arr = np.arange(cells_per_side)
        dk_arr = np.arange(cells_per_side)
        di_g, dj_g, dk_g = np.meshgrid(di_arr, dj_arr, dk_arr, indexing="ij")
        ci_g = (ci0 + di_g) % n_grid
        cj_g = (cj0 + dj_g) % n_grid
        ck_g = (ck0 + dk_g) % n_grid
        cell_x = (ci_g + 0.5) * dx
        cell_y = (cj_g + 0.5) * dx
        cell_z = (ck_g + 0.5) * dx

        dcx = cell_x - centre[0]
        dcy = cell_y - centre[1]
        dcz = cell_z - centre[2]
        for dc in (dcx, dcy, dcz):
            mask = np.abs(dc) > box_size * 0.5
            dc[mask] -= np.copysign(box_size, dc[mask])

        r_cells = np.sqrt(dcx**2 + dcy**2 + dcz**2) / r200_h
        in_range = (r_cells >= R_MIN_FRAC) & (r_cells < R_MAX_FRAC)
        b_cells  = ((np.log(np.where(in_range, r_cells, 1.0)) - log_rmin) / dlog).astype(int)

        flat_idx = ci_g * n_grid * n_grid + cj_g * n_grid + ck_g
        valid    = in_range & (b_cells < N_BINS)

        for b_val in range(N_BINS):
            sel = valid & (b_cells == b_val)
            if not sel.any():
                continue
            idxs = flat_idx[sel]
            rho_gas_sum[b_val] += gas_rho[idxs].sum()
            t_sum[b_val]       += (gas_u[idxs] * U_TO_T).sum()
            counts[b_val]      += sel.sum()

    # Normalise
    r_mid   = np.exp(log_rmin + (np.arange(N_BINS) + 0.5) * dlog)
    rho_gas = np.where(counts > 0, rho_gas_sum / counts, 0.0)
    t_mean  = np.where(counts > 0, t_sum  / counts, 0.0)
    rho_dm  = rho_dm_sum / max(n_halos, 1)

    return r_mid, rho_gas, t_mean, rho_dm, counts, n_halos


def write_profiles(path: Path, r_mid, rho_gas, t_k, rho_dm, counts, n_halos, mass_label):
    with open(path, "w") as f:
        f.write(f"# mass_bin={mass_label}  n_halos={n_halos}\n")
        f.write("# r/R200  rho_gas/rho_mean  T_K  rho_dm/rho_mean  N_cells\n")
        for i in range(N_BINS):
            f.write(f"{r_mid[i]:.4e} {rho_gas[i]:.4e} {t_k[i]:.4e} {rho_dm[i]:.4e} {counts[i]}\n")
    print(f"  wrote {path}  (n_halos={n_halos})")


# ── Main ──────────────────────────────────────────────────────────────────────

def detect_last_snap(outdir: Path) -> str:
    """Find the highest-numbered snapshot_NNNN.bin in outdir.

    Returns the zero-padded index string (e.g. "0004"). Falls back to "0000"
    if none are present, so the caller's "snapshot not found" error path is
    preserved for empty directories (e.g. quick-run case).
    """
    snaps = sorted(outdir.glob("snapshot_[0-9][0-9][0-9][0-9].bin"))
    if not snaps:
        return "0000"
    # Strip "snapshot_" prefix and ".bin" suffix → 4-digit index
    return snaps[-1].stem.split("_")[-1]


def process_dir(outdir: Path, snap: str | None, params: dict):
    if snap is None:
        snap = detect_last_snap(outdir)
        print(f"[auto-snap] {outdir}: using snapshot {snap}")
    snap_path     = outdir / f"snapshot_{snap}.bin"
    gas_snap_path = outdir / f"gas_snapshot_{snap}.bin"

    if not snap_path.exists():
        print(f"SKIP {outdir}: {snap_path} not found")
        return
    if not gas_snap_path.exists():
        print(f"SKIP {outdir}: {gas_snap_path} not found")
        return

    print(f"\n=== {outdir} / snap {snap} ===")
    step, a, pos, vel = read_snapshot(snap_path)
    print(f"  DM: {len(pos)} particles, a={a:.4f} (z={1/a-1:.2f})")

    _, _, gas_rho, gas_u = read_gas_snapshot(gas_snap_path)
    n_grid = round(len(gas_rho) ** (1 / 3))
    print(f"  Gas grid: {n_grid}³ = {len(gas_rho)} cells")

    box_size = params["box_size"]
    h        = params["h"]
    omega_m  = params["omega_m"]
    n_part   = round(len(pos) ** (1 / 3))
    rho_crit = 2.775e11 * h * h
    m_part   = rho_crit * omega_m * box_size**3 / n_part**3
    print(f"  m_part = {m_part:.3e} M_sun/h")

    b = 0.2 * box_size / n_part
    print(f"  FoF linking length b = {b:.4f} Mpc/h")
    print(f"  Running FoF on {len(pos)} particles ...", flush=True)
    groups = friends_of_friends(pos, b, box_size)
    print(f"  Found {len(groups)} FoF groups")

    idx = int(snap)

    for mass_label, m_min, m_max in [
        ("M>10^13.5", 10**13.5, np.inf),
        ("M<10^12",   0.0,      10**12.0),
    ]:
        tag = "highmass" if ">" in mass_label else "lowmass"
        r_mid, rho_gas, t_k, rho_dm, counts, n_halos = stacked_profiles(
            pos, gas_rho, gas_u,
            n_grid, box_size, m_part, h, m_min, m_max, groups,
        )
        out_path = outdir / f"halo_profiles_{tag}_{snap}.txt"
        write_profiles(out_path, r_mid, rho_gas, t_k, rho_dm, counts, n_halos, mass_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs",     nargs="+", required=True,
                        help="Output directories to process")
    parser.add_argument("--snap",     default=None,
                        help="Zero-padded snapshot index (default: None — "
                             "auto-detect last snapshot per directory)")
    parser.add_argument("--box-size", type=float, default=64.0,
                        help="Box side length in Mpc/h (default: 64)")
    parser.add_argument("--h",        type=float, default=0.67,
                        help="Hubble parameter h (default: 0.67)")
    parser.add_argument("--omega-m",  type=float, default=0.3,
                        help="Matter density parameter (default: 0.3)")
    args = parser.parse_args()

    params = {
        "box_size": args.box_size,
        "h":        args.h,
        "omega_m":  args.omega_m,
    }

    for d in args.dirs:
        process_dir(Path(d), args.snap, params)


if __name__ == "__main__":
    main()
