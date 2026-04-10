"""Render dark matter density field animation from RUSTIS snapshot series.

Reads binary snapshots, projects all particles onto the XY plane, and renders
a multi-scale Gaussian density map with the standard cosmic web colormap used
by Millennium, Illustris, and similar simulations.

Usage:
  uv run python analysis/render_animation.py --snapdir output_lcdm_anim --output animation.mp4
  uv run python analysis/render_animation.py --snapdir output_lcdm_128 --output test.mp4 --box-size 64

Binary snapshot format (from RUSTIS src/simulation.rs):
  [step: u64 LE][a: f64 LE][n_part: u64 LE]
  then per particle: [x, y, z, vx, vy, vz: f64 LE]
  Positions in comoving Mpc/h.
"""

import os
import sys
import glob
import struct
import shutil
import argparse
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render dark matter density animation from RUSTIS snapshots"
    )
    p.add_argument("--snapdir",   required=True,       help="Directory with snapshot_XXXX.bin")
    p.add_argument("--output",    required=True,       help="Output MP4 or GIF path")
    p.add_argument("--gif",       action="store_true", help="Also produce a GIF alongside the MP4")
    p.add_argument("--params",    default=None,        help="Params .toml to read box_size")
    p.add_argument("--box-size",  type=float, default=None, help="Box size in Mpc/h (default: 64)")
    p.add_argument("--bins",      type=int,   default=2048,  help="Pixels per axis (default: 2048)")
    p.add_argument("--fps",       type=int,   default=5,     help="Frames per second (default: 5)")
    p.add_argument("--dpi",       type=int,   default=100,   help="Output DPI (default: 100)")
    p.add_argument("--omega-m",   type=float, default=0.272, help="Omega_m (default: 0.272 WMAP7)")
    p.add_argument("--omega-lambda", type=float, default=0.728, help="Omega_Lambda (default: 0.728)")
    p.add_argument("--hubble",    type=float, default=0.704, help="h (default: 0.704 WMAP7)")
    return p.parse_args()


# -- Snapshot I/O --------------------------------------------------------------

def read_snapshot(path):
    """Return (step, a, positions) where positions is (n_part, 3) float64."""
    with open(path, "rb") as f:
        step, a, n_part = struct.unpack("<QdQ", f.read(24))
        data = np.frombuffer(f.read(n_part * 6 * 8), dtype="<f8").reshape(n_part, 6)
    return step, a, data[:, :3]


def find_snapshots(snapdir):
    files = sorted(glob.glob(os.path.join(snapdir, "snapshot_*.bin")))
    if not files:
        print(f"ERROR: no snapshot_*.bin files in {snapdir}", file=sys.stderr)
        sys.exit(1)
    return files


def get_box_size(args):
    if args.box_size is not None:
        return args.box_size
    if args.params is not None:
        import tomllib
        with open(args.params, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("run", {}).get("box_size", 64.0)
    return 64.0


# -- Colormap ------------------------------------------------------------------
# Standard dark matter colormap used by Millennium, Bolshoi, and similar:
# deep black void -> dark indigo -> royal blue -> cyan -> white halo cores.
# Matches what viewers expect from popular cosmic web visualisations.

def make_cosmo_cmap():
    colors = [
        (0.000, ( 0.000,  0.000,  0.000)),  # black        — void
        (0.100, ( 0.020,  0.005,  0.090)),  # dark indigo  — faintest sheets
        (0.220, ( 0.040,  0.030,  0.280)),  # indigo-blue  — sheet walls
        (0.380, ( 0.000,  0.120,  0.520)),  # royal blue   — filament edges
        (0.540, ( 0.000,  0.380,  0.750)),  # bright blue  — filaments
        (0.680, ( 0.000,  0.620,  0.880)),  # sky blue     — dense filaments
        (0.800, ( 0.200,  0.820,  0.950)),  # cyan         — halo outskirts
        (0.900, ( 0.650,  0.930,  0.980)),  # pale cyan    — halo
        (0.960, ( 0.900,  0.970,  1.000)),  # near-white
        (1.000, ( 1.000,  1.000,  1.000)),  # white        — densest cores
    ]
    return LinearSegmentedColormap.from_list(
        "cosmo_millennium", [(pos, rgb) for pos, rgb in colors], N=512
    )


# -- Multi-scale Gaussian splatting --------------------------------------------
# Approximates the adaptive SPH kernel rendering used by professional tools
# (Splotch, yt-project).  Each scale contributes log1p(blur(counts)) * weight.
# Filament-scale layer (sigma≈20px ≈ 0.6 Mpc/h) carries the most weight so
# the web is visible rather than just halo cores.

_SPLAT_SCALES = [
    ( 1.0, 0.50),   # halo cores     ~0.03 Mpc/h
    ( 5.0, 0.80),   # halo outskirts ~0.16 Mpc/h
    (20.0, 1.00),   # filaments      ~0.63 Mpc/h  ← dominant
    (60.0, 0.65),   # LSS sheets     ~1.9  Mpc/h
]


def project_density(positions, box_size, bins, sigma_mult=1.0):
    """Project all particles onto a 2D multi-scale density grid.

    Returns a (bins, bins) float64 array ready for imshow vmax normalisation.
    sigma_mult scales all kernel widths (1.0 = reference schedule above).
    """
    from scipy.ndimage import gaussian_filter

    x, y = positions[:, 0], positions[:, 1]
    counts, _, _ = np.histogram2d(
        x, y, bins=bins, range=[[0, box_size], [0, box_size]],
    )
    counts = counts.T.astype(np.float64)

    result = np.zeros_like(counts)
    for sigma_px, weight in _SPLAT_SCALES:
        s = max(sigma_px * sigma_mult, 0.3)
        blurred = gaussian_filter(counts, sigma=s)
        result += weight * np.log1p(blurred)

    return result ** 0.5  # power-law stretch


# -- Cosmology -----------------------------------------------------------------

def cosmic_age_gyr(a, omega_m, omega_lambda, h):
    """Cosmic age at scale factor a in Gyr (flat ΛCDM)."""
    from scipy.integrate import quad
    H0_inv_gyr = 978.0 / (h * 100.0)

    def integrand(ap):
        return 1.0 / (ap * np.sqrt(omega_m / ap**3 + omega_lambda))

    result, _ = quad(integrand, 1e-6, min(a, 1.0))
    return H0_inv_gyr * result


# -- Frame rendering -----------------------------------------------------------

def render_frame(density, a, box_size, cmap, vmax, fig, ax,
                 omega_m, omega_lambda, h):
    ax.clear()
    ax.imshow(
        density,
        origin="lower",
        extent=[0, box_size, 0, box_size],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.axis("off")

    z = max(0.0, 1.0 / a - 1.0)
    age = cosmic_age_gyr(a, omega_m, omega_lambda, h)

    if z < 0.01:
        z_str = "z = 0"
    elif z >= 10:
        z_str = f"z = {z:.0f}"
    else:
        z_str = f"z = {z:.2f}"

    age_str = f"{age * 1000:.0f} Myr" if age < 1.0 else f"{age:.2f} Gyr"
    label = f"{z_str}  |  a = {min(a, 1.0):.3f}  |  {age_str}"

    ax.text(
        0.03, 0.03, label,
        transform=ax.transAxes,
        color="white",
        fontsize=20,
        fontfamily="monospace",
        fontweight="bold",
        alpha=0.90,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none",
                  boxstyle="round,pad=0.3"),
    )


# -- Video stitching -----------------------------------------------------------

def stitch_mp4(frame_dir, output_path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("WARNING: ffmpeg not found — PNG frames preserved.", file=sys.stderr)
        return False
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        output_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def stitch_gif(frame_dir, output_path, fps):
    try:
        from PIL import Image
    except ImportError:
        print("WARNING: Pillow not installed. Install with: uv add pillow", file=sys.stderr)
        return False
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not frame_paths:
        return False
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:],
        duration=1000 // fps, loop=0,
    )
    return True


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()
    box_size = get_box_size(args)
    snapshots = find_snapshots(args.snapdir)
    n_snaps = len(snapshots)

    print(f"Found {n_snaps} snapshots in {args.snapdir}")
    print(f"Box size: {box_size} Mpc/h  |  Resolution: {args.bins}x{args.bins} px")
    print(f"Cosmology: Omega_m={args.omega_m}, Omega_L={args.omega_lambda}, h={args.hubble}")

    cmap = make_cosmo_cmap()
    figsize = args.bins / args.dpi
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize), dpi=args.dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("black")

    tmpdir = tempfile.mkdtemp(prefix="rustis_anim_")
    print(f"Rendering {n_snaps} frames to {tmpdir}")

    # Calibrate vmax from the final snapshot (most evolved, most structure)
    _, _, pos_last = read_snapshot(snapshots[-1])
    d_last = project_density(pos_last, box_size, args.bins)
    vmax = np.percentile(d_last, 95)
    print(f"vmax = {vmax:.3f}  (95th percentile of final frame)")

    frame_idx = 0
    hold_frames = args.fps * 3  # 3-second hold on the last frame

    for i, snap_path in enumerate(snapshots):
        _, a, pos = read_snapshot(snap_path)
        density = project_density(pos, box_size, args.bins)
        render_frame(density, a, box_size, cmap, vmax, fig, ax,
                     args.omega_m, args.omega_lambda, args.hubble)

        # Save the frame; hold the last snapshot for 3 extra seconds
        repeat = hold_frames if i == n_snaps - 1 else 1
        for _ in range(repeat):
            out_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.png")
            fig.savefig(out_path, dpi=args.dpi, facecolor="black", pad_inches=0)
            frame_idx += 1

        z = max(0.0, 1.0 / a - 1.0)
        age = cosmic_age_gyr(a, args.omega_m, args.omega_lambda, args.hubble)
        age_str = f"{age*1000:.0f} Myr" if age < 1.0 else f"{age:.2f} Gyr"
        suffix = f" (held x{repeat})" if repeat > 1 else ""
        print(f"  [{i+1:3d}/{n_snaps}]  a={a:.4f}  z={z:6.2f}  {age_str:<10}  -> frame_{frame_idx-repeat:04d}.png{suffix}")

    plt.close(fig)

    output = args.output
    if output.endswith(".mp4"):
        success = stitch_mp4(tmpdir, output, args.fps)
    elif output.endswith(".gif"):
        success = stitch_gif(tmpdir, output, args.fps)
    else:
        print("ERROR: output must end in .mp4 or .gif", file=sys.stderr)
        sys.exit(1)

    if args.gif and output.endswith(".mp4"):
        stitch_gif(tmpdir, output.replace(".mp4", ".gif"), args.fps)

    if success:
        size_mb = os.path.getsize(output) / (1024 * 1024)
        total_frames = n_snaps + hold_frames - 1
        print(f"Done. {output}  ({size_mb:.1f} MB, {total_frames} frames at {args.fps} fps)")
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"Frames preserved at {tmpdir}")


if __name__ == "__main__":
    main()
