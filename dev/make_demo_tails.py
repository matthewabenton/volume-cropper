"""Demo data for the magic wand: bunched, touching "tails" (experimental).

Makes a synthetic lightsheet-like volume of four curved, tapering tubes,
330-380 µm thick and 650-800 µm long, pressed into each other, and saves it as an OME-Zarr 0.5
image with a pyramid. The true object of every voxel is known, so the
segmentation can be scored.

Channels:
  0 "tissue (demo)": all tissue, with a nuclei-like texture, dimmer with depth.
  1 "tip marker (demo)": only the posterior 40 % of each tail.

Usage::

    python make_demo_tails.py OUTPUT_FOLDER

writes OUTPUT_FOLDER/touching_tails_demo.zarr and the true labels as
OUTPUT_FOLDER/touching_tails_demo_truth.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# Tails as quadratic Bézier centrelines (start, control, end), in µm as
# (z, y, x), with the radius at the start (the tail tapers to 75 % of it at
# the end). End to end, with their rounded ends, they are 650-800 µm long
# and 330-380 µm thick. Neighbours overlap by
# 20-80 µm, so they touch at a neck.
TAILS = [
    {"points": [(600, 420, 330), (620, 600, 390), (590, 790, 350)], "radius": 190},
    {"points": [(590, 450, 680), (610, 630, 720), (640, 830, 690)], "radius": 180},
    {"points": [(930, 640, 300), (900, 690, 510), (920, 650, 720)], "radius": 170},
    {"points": [(640, 1020, 850), (620, 1120, 990), (600, 1140, 1180)], "radius": 165},
]
EXTENT_UM = (1280.0, 1600.0, 1600.0)          # z, y, x
TAPER = 0.75                                   # radius at the end / at the start


def bezier(points, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """n points along a quadratic Bézier curve, and their position s (0-1)."""
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in points)
    s = np.linspace(0.0, 1.0, n)[:, None]
    curve = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2
    return curve, s[:, 0]


def phantom(voxel_um=(16.0, 16.0, 16.0), tails=TAILS, seed: int = 0):
    """The demo volume on a grid with this voxel size (z, y, x in µm).

    Returns (image, truth): image is (2, z, y, x) uint16 (tissue, tip
    marker); truth is (z, y, x) uint8 with 0 = background, 1.. = tail.
    Voxel i is centred at i * voxel_um (the OME-Zarr convention).
    """
    voxel = np.asarray(voxel_um, dtype=float)
    shape = tuple(int(round(e / v)) for e, v in zip(EXTENT_UM, voxel))
    # For every voxel: the smallest "distance to a tail's midline divided by
    # that tail's local radius" (inside the tail when <= 1), which tail it
    # is, and how far along that tail (for the tip marker).
    best = np.full(shape, np.inf, dtype=np.float32)
    truth = np.zeros(shape, dtype=np.uint8)
    along = np.zeros(shape, dtype=np.float32)
    for label, tail in enumerate(tails, start=1):
        curve, s = bezier(tail["points"])
        radius = tail["radius"] * (1 - (1 - TAPER) * s)
        reach = radius.max() + 2 * voxel.max()
        low = np.maximum(np.floor((curve.min(0) - reach) / voxel), 0).astype(int)
        high = np.minimum(np.ceil((curve.max(0) + reach) / voxel) + 1,
                          shape).astype(int)
        sub = tuple(slice(a, b) for a, b in zip(low, high))
        sub_shape = tuple(b - a for a, b in zip(low, high))
        # Mark the midline on the grid; the distance transform then gives,
        # for every voxel, the distance to the nearest midline point and
        # which point that is (for its radius and position along the tail).
        index = np.full(sub_shape, -1, dtype=np.int32)
        cells = np.round(curve / voxel).astype(int) - low
        ok = np.all((cells >= 0) & (cells < sub_shape), axis=1)
        index[tuple(cells[ok].T)] = np.nonzero(ok)[0]
        dist, nearest = ndi.distance_transform_edt(
            index < 0, sampling=voxel, return_indices=True)
        point = index[tuple(nearest)]
        normalised = (dist / radius[point]).astype(np.float32)
        closer = normalised < best[sub]
        best[sub] = np.where(closer, normalised, best[sub])
        truth[sub] = np.where(closer & (normalised <= 1), label, truth[sub])
        along[sub] = np.where(closer, s[point], along[sub])
    truth[best > 1] = 0

    rng = np.random.default_rng(seed)
    tissue = truth > 0
    # Nuclei-like texture: smoothed noise, about 10 µm grains.
    grain = ndi.gaussian_filter(rng.random(shape, dtype=np.float32),
                                sigma=np.maximum(5.0 / voxel, 0.5))
    grain = (grain - grain.mean()) / (grain.std() + 1e-9)
    depth = 1 - 0.35 * np.linspace(0, 1, shape[0])[:, None, None]   # dimmer deeper
    blur = np.maximum(4.0 / voxel, 0.3)                               # optics
    channel0 = ndi.gaussian_filter(
        tissue * (900 + 250 * grain) * depth, sigma=blur)
    tip = tissue & (along > 0.6)
    channel1 = ndi.gaussian_filter(tip * (700 + 150 * grain), sigma=blur)
    image = np.stack([channel0 + 110, channel1 + 90])
    image += rng.normal(0, 25, image.shape)                           # camera noise
    return np.clip(image, 0, 65535).astype(np.uint16), truth


DEMO_CHANNELS = [{"label": "tissue (demo)", "color": "00FFFF"},
                 {"label": "tip marker (demo)", "color": "FF00FF"}]


def write_ome_zarr(path, image, voxel_um, factors=((1, 1, 1), (1, 2, 2),
                                                   (2, 4, 4), (4, 8, 8)),
                   name="touching tails (demo)", channels=DEMO_CHANNELS,
                   extra_attrs=None):
    """Save (c, z, y, x) as a sharded OME-Zarr 0.5 image with a pyramid made
    by block averaging (like eubi-bridge's output, on a small scale).

    name and channels (label and colour per channel) go into the OME
    metadata; extra_attrs (a dict) is stored next to it, e.g. how a
    synthetic image was made."""
    import zarr
    from zarr.codecs import BloscCodec

    root = zarr.open_group(path, mode="w", zarr_format=3)
    datasets = []
    for level, factor in enumerate(factors):
        data = image
        if factor != (1, 1, 1):
            data = np.stack([_block_mean(c, factor) for c in image])
        chunks = (1, 32, 64, 64)
        shards = (1, 64, 128, 128)
        arr = root.create_array(
            name=str(level), shape=data.shape, dtype=data.dtype,
            chunks=chunks, shards=shards,
            compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
            fill_value=0, dimension_names=["c", "z", "y", "x"])
        arr[...] = data
        scale = [1.0] + [v * f for v, f in zip(voxel_um, factor)]
        datasets.append({"path": str(level), "coordinateTransformations": [
            {"type": "scale", "scale": scale}]})
    space = {"type": "space", "unit": "micrometer"}
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": name,
            "axes": [{"name": "c", "type": "channel"},
                     {"name": "z", **space}, {"name": "y", **space},
                     {"name": "x", **space}],
            "datasets": datasets}],
        "omero": {"channels": list(channels)},
    }
    if extra_attrs:
        root.attrs.update(extra_attrs)


def _block_mean(data, factor):
    """Average over blocks of factor voxels (the last block may be smaller)."""
    out = data.astype(np.float32)
    for axis, f in enumerate(factor):
        if f == 1:
            continue
        n = out.shape[axis]
        pad = (-n) % f
        if pad:
            widths = [(0, 0)] * out.ndim
            widths[axis] = (0, pad)
            out = np.pad(out, widths, mode="edge")
        shape = list(out.shape)
        shape[axis:axis + 1] = [shape[axis] // f, f]
        out = out.reshape(shape).mean(axis=axis + 1)
    return np.round(out).astype(data.dtype)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--voxel", type=float, nargs=3, default=[8.0, 4.0, 4.0],
                        metavar=("Z", "Y", "X"),
                        help="finest voxel size in µm (default 8 4 4)")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image, truth = phantom(args.voxel)
    write_ome_zarr(out / "touching_tails_demo.zarr", image, args.voxel)
    np.savez_compressed(out / "touching_tails_demo_truth.npz", labels=truth,
                        voxel_um=np.asarray(args.voxel),
                        tails=json.dumps(TAILS))
    print(f"wrote {out / 'touching_tails_demo.zarr'}: {image.shape} at "
          f"{args.voxel} µm, {len(TAILS)} tails")


if __name__ == "__main__":
    main()
