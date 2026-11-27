"""A synthetic time-lapse for checks, guide pictures and demo clips (0.17.0).

make_image(path) writes an OME-Zarr 0.5 (t c z y x = 8 x 2 x 48 x 160 x
320, voxels 4 x 2 x 2 µm, 15 min per frame, 3 levels): an "embryo" (a
textured ellipsoid, channel 0, with a brighter rim in channel 1) that
drifts about 300 µm along x and 40 µm along y over 8 timepoints and grows
by a third, in Poisson noise. centre_at(t) is its true centre, radii_at(t)
its true radii (µm). Synthetic data only; seed 0, so it is always the same.
"""
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

T, SHAPE = 8, (48, 160, 320)            # timepoints; z, y, x voxels
SCALE = (4.0, 2.0, 2.0)                 # µm
INTERVAL = 15.0                         # minutes


def centre_at(t):
    """The embryo's true centre (z, y, x in µm) at timepoint t."""
    return (96.0, 140.0 + 40.0 * t / (T - 1), 140.0 + 300.0 * t / (T - 1))


def radii_at(t):
    """The embryo's true radii (z, y, x in µm) at timepoint t."""
    grow = 1.0 + t / (3 * (T - 1))
    return (36.0 * grow, 45.0 * grow, 70.0 * grow)


def make_image(path):
    """The drifting, growing embryo as an OME-Zarr 0.5 (3 levels)."""
    rng = np.random.default_rng(0)
    grids = np.meshgrid(*[(np.arange(n) * s) for n, s in zip(SHAPE, SCALE)],
                        indexing="ij")
    texture = rng.normal(1.0, 0.25, size=SHAPE)
    data = np.empty((T, 2, *SHAPE), dtype=np.uint16)
    for t in range(T):
        inside = sum(((g - c) / r) ** 2 for g, c, r in
                     zip(grids, centre_at(t), radii_at(t)))
        embryo = np.where(inside <= 1.0, 1500.0 * texture, 0.0)
        rim = np.where(np.abs(inside - 1.0) < 0.12, 600.0, 0.0)
        data[t, 0] = rng.poisson(100 + embryo.clip(0))
        data[t, 1] = rng.poisson(80 + rim)
    root = zarr.open_group(path, mode="w", zarr_format=3)
    datasets = []
    level, prev = data, (1, 1, 1)
    for k, f in enumerate([(1, 1, 1), (1, 2, 2), (2, 4, 4)]):
        if k:
            step = [a // b for a, b in zip(f, prev)]
            n = [s // d for s, d in zip(level.shape[2:], step)]
            level = level[:, :, :n[0] * step[0], :n[1] * step[1],
                          :n[2] * step[2]].reshape(
                T, 2, n[0], step[0], n[1], step[1], n[2], step[2]).mean(
                axis=(3, 5, 7)).astype(np.uint16)
        prev = f
        arr = root.create_array(
            name=str(k), shape=level.shape, dtype=level.dtype,
            chunks=(1, 1, 16, 32, 32), shards=(1, 1, 16, 64, 64),
            compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
            fill_value=0, dimension_names=list("tczyx"))
        arr[...] = level
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": [INTERVAL, 1.0] + [
                s * n for s, n in zip(SCALE, f)]}]})
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "name": "drifting embryo",
        "axes": [{"name": "t", "type": "time", "unit": "minute"},
                 {"name": "c", "type": "channel"}] + [
            {"name": a, "type": "space", "unit": "micrometer"}
            for a in "zyx"],
        "datasets": datasets}],
        "omero": {"channels": [
            {"label": "embryo", "color": "00FF00",
             "window": {"start": 0, "end": 2500, "min": 0, "max": 65535}},
            {"label": "rim", "color": "FF00FF",
             "window": {"start": 0, "end": 800, "min": 0, "max": 65535}}]}}
    return Path(path)
