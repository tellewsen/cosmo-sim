"""Render dark matter density field animation from RUSTIS snapshot series.

Reads binary snapshots, projects particles through a thin Z-slab onto the XY
plane, and renders a logarithmic density map with a dark cosmological color
scheme.  Primary output is MP4 via ffmpeg subprocess; optionally also GIF.

Usage:
  uv run python analysis/render_animation.py --snapdir output_lcdm_anim --output animation.mp4
  uv run python analysis/render_animation.py --snapdir output_lcdm_anim --output animation.mp4 --gif
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Render dark matter density animation from RUSTIS snapshots"
    )
    p.add_argument("--snapdir", required=True, help="Directory with snapshot_XXXX.bin")
    p.add_argument("--output", required=True, help="Output MP4 or GIF path")
    p.add_argument("--gif", action="store_true", help="Also produce a GIF alongside the primary output")
    p.add_argument("--params", default=None, help="Params .toml to read box_size")
    p.add_argument("--box-size", type=float, default=None, help="Box size [Mpc/h] (default: 64)")
    p.add_argument("--slab-frac", type=float, default=0.125, help="Z-slab thickness as fraction of box (default: 1/8)")
    p.add_argument("--bins", type=int, default=800, help="Histogram bins per axis (default: 800)")
    p.add_argument("--fps", type=int, default=12, help="Frames per second (default: 12)")
    p.add_argument("--dpi", type=int, default=100, help="Output DPI (default: 100 => 800x800 px)")
    p.add_argument("--gaussian-sigma", type=float, default=1.0,
                   help="Gaussian smoothing sigma in pixels (default: 1.0, 0=none)")
    return p.parse_args()


# ── Snapshot I/O ───────────────────────────────────────────────────────────────

def read_snapshot(path):
    """Read a RUSTIS binary snapshot.

    Returns (step, a, positions) where positions is (n_part, 3) float64 array.
    """
    with open(path, "rb") as f:
        step, a, n_part = struct.unpack("<QdQ", f.read(24))
        data = np.frombuffer(f.read(n_part * 6 * 8), dtype="<f8").reshape(n_part, 6)
    return step, a, data[:, :3]


def find_snapshots(snapdir):
    """Find and sort snapshot files by index."""
    pattern = os.path.join(snapdir, "snapshot_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no snapshot_*.bin files in {snapdir}", file=sys.stderr)
        sys.exit(1)
    return files


def get_box_size(args):
    """Determine box size from args or params file."""
    if args.box_size is not None:
        return args.box_size
    if args.params is not None:
        import tomllib
        with open(args.params, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("run", {}).get("box_size", 64.0)
    return 64.0


# ── Colormap ───────────────────────────────────────────────────────────────────

def make_cosmo_cmap():
    """Black -> deep blue -> cyan -> white colormap for dark matter density."""
    colors = [
        (0.00, (0.0,  0.0,  0.0 )),   # black  (void)
        (0.15, (0.02, 0.01, 0.12)),    # near-black purple
        (0.30, (0.0,  0.03, 0.30)),    # deep blue
        (0.50, (0.0,  0.18, 0.50)),    # medium blue
        (0.65, (0.0,  0.45, 0.65)),    # teal
        (0.78, (0.0,  0.65, 0.80)),    # cyan
        (0.88, (0.35, 0.82, 0.92)),    # light cyan
        (0.95, (0.75, 0.93, 0.98)),    # pale cyan-white
        (1.00, (1.0,  1.0,  1.0 )),    # white  (densest halos)
    ]
    return LinearSegmentedColormap.from_list(
        "cosmo_dark", [(pos, rgb) for pos, rgb in colors], N=512,
    )


# ── Rendering ──────────────────────────────────────────────────────────────────

def project_density(positions, box_size, slab_frac, bins, sigma=1.0):
    """Project particles in a thin Z-slab onto a 2D log-density grid.

    Returns a (bins, bins) float64 array of log1p(counts), optionally
    Gaussian-smoothed.
    """
    z_center = box_size / 2.0
    z_half = box_size * slab_frac / 2.0

    z = positions[:, 2]
    mask = (z >= z_center - z_half) & (z < z_center + z_half)
    x = positions[mask, 0]
    y = positions[mask, 1]

    counts, _, _ = np.histogram2d(
        x, y, bins=bins, range=[[0, box_size], [0, box_size]],
    )

    density = np.log1p(counts).T  # transpose: x horizontal, y vertical

    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=sigma)

    return density


def render_frame(density, a, box_size, cmap, vmax, fig, ax):
    """Render a single density frame onto the given axes."""
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

    # Redshift label — bottom left
    z = max(0.0, 1.0 / a - 1.0)
    if z < 0.01:
        z_str = "z = 0"
    elif z >= 10:
        z_str = f"z = {z:.0f}"
    else:
        z_str = f"z = {z:.2f}"
    label = f"{z_str}  |  a = {min(a, 1.0):.2f}"
    ax.text(
        0.03, 0.03, label,
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        fontfamily="monospace",
        fontweight="normal",
        alpha=0.80,
        va="bottom",
        ha="left",
    )


def stitch_mp4(frame_dir, output_path, fps):
    """Use ffmpeg subprocess to stitch PNGs into MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("WARNING: ffmpeg not found. Cannot produce MP4.", file=sys.stderr)
        print("  Install with: sudo apt install ffmpeg", file=sys.stderr)
        print(f"  PNG frames are in {frame_dir}", file=sys.stderr)
        return False

    cmd = [
        ffmpeg,
        "-y",                           # overwrite
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",                   # high quality
        "-preset", "slow",
        output_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def stitch_gif(frame_dir, output_path, fps, n_frames):
    """Use Pillow to stitch PNGs into animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("WARNING: Pillow not installed. Cannot produce GIF.", file=sys.stderr)
        print("  Install with: uv add pillow", file=sys.stderr)
        return False

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not frame_paths:
        print("ERROR: no frames to stitch", file=sys.stderr)
        return False

    frames = [Image.open(p) for p in frame_paths]
    duration_ms = 1000 // fps

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    box_size = get_box_size(args)
    snapshots = find_snapshots(args.snapdir)
    n_frames = len(snapshots)

    print(f"Found {n_frames} snapshots in {args.snapdir}")
    print(f"Box size: {box_size} Mpc/h, slab fraction: {args.slab_frac}")
    print(f"Output resolution: {args.bins}x{args.bins} px")

    cmap = make_cosmo_cmap()

    # Figure setup -- borderless, black background
    figsize = args.bins / args.dpi
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize), dpi=args.dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("black")

    # Use a temp dir for intermediate PNGs
    tmpdir = tempfile.mkdtemp(prefix="rustis_anim_")
    print(f"Rendering frames to {tmpdir}")

    # First pass: compute global vmax for consistent color scale across frames
    # (read first and last snapshots as quick estimate)
    _, a_first, pos_first = read_snapshot(snapshots[0])
    d_first = project_density(pos_first, box_size, args.slab_frac, args.bins, args.gaussian_sigma)
    _, a_last, pos_last = read_snapshot(snapshots[-1])
    d_last = project_density(pos_last, box_size, args.slab_frac, args.bins, args.gaussian_sigma)
    # Use 99.8th percentile of last frame for vmax (most structure at late times)
    vmax = np.percentile(d_last, 99.8)
    print(f"Color scale: vmax = {vmax:.2f} (99.8th pctile of final frame)")

    # Render all frames as PNGs
    for i, snap_path in enumerate(snapshots):
        step, a, pos = read_snapshot(snap_path)
        density = project_density(pos, box_size, args.slab_frac, args.bins, args.gaussian_sigma)

        render_frame(density, a, box_size, cmap, vmax, fig, ax)

        out_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        fig.savefig(out_path, dpi=args.dpi, facecolor="black", pad_inches=0)

        z = max(0.0, 1.0 / a - 1.0)
        print(f"  [{i+1:3d}/{n_frames}] a={a:.4f}  z={z:7.2f}  -> {out_path}")

    plt.close(fig)

    # Stitch into video
    output = args.output
    success = False

    actual_output = output
    if output.endswith(".mp4"):
        success = stitch_mp4(tmpdir, output, args.fps)
        if not success:
            # Fallback: try GIF
            actual_output = output.replace(".mp4", ".gif")
            print(f"Falling back to GIF output: {actual_output}")
            success = stitch_gif(tmpdir, actual_output, args.fps, n_frames)
    elif output.endswith(".gif"):
        actual_output = output
        success = stitch_gif(tmpdir, output, args.fps, n_frames)
    else:
        print(f"ERROR: output must end in .mp4 or .gif", file=sys.stderr)
        sys.exit(1)

    # Optional extra GIF alongside MP4
    if args.gif and output.endswith(".mp4"):
        gif_path = output.replace(".mp4", ".gif")
        print(f"Also producing GIF: {gif_path}")
        stitch_gif(tmpdir, gif_path, args.fps, n_frames)

    if success:
        print(f"Done. Saved to {actual_output}")
        # Clean up temp frames
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"Frames preserved at {tmpdir}")


if __name__ == "__main__":
    main()
