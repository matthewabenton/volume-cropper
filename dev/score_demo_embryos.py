"""Score the 3D shape through outlines on the embryo demo's objects.

For every object, outlines are traced from its true shape the way a person
would draw them (the outer edge of the largest piece, on 3 XY, 2 XZ and
2 YZ slices across it); the tool's shape through them (outline_mask) is
then compared with the object: how much of it is kept, how much else,
and whether the shape has holes (closed hollows, or holes on single
slices) or loose pieces that a person did not draw.

Usage (napari-env)::

    python score_demo_embryos.py DEMO_FOLDER [--margin VOXELS] [--views xy]

Tests the tool in VC_TOOL (default: the tool folder this script is in).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import approximate_polygon, find_contours

sys.path.insert(0, os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
from volume_cropper import core as vc  # noqa: E402

NAME = "embryo_tissue_demo"
ZYX = ["z", "y", "x"]


def edge(section):
    """The outer edge of the largest piece of a 2D mask, as (row, col)
    points; None if the section is empty."""
    pieces, count = ndi.label(section)
    if not count:
        return None
    sizes = ndi.sum(section, pieces, range(1, count + 1))
    largest = ndi.binary_fill_holes(pieces == 1 + int(np.argmax(sizes)))
    contour = max(find_contours(np.pad(largest, 1).astype(float), 0.5),
                  key=len) - 1
    return approximate_polygon(contour, tolerance=0.7)


def drawn_outlines(mask, voxel, views="xy xz yz"):
    """Outlines on 3 XY, 2 XZ and 2 YZ slices across the object."""
    idx = np.argwhere(mask)
    low, high = idx.min(0), idx.max(0)
    outlines = {}
    plans = {"xy": (0, ("y", "x"), (0.2, 0.5, 0.8)),
             "xz": (1, ("z", "x"), (0.35, 0.65)),
             "yz": (2, ("y", "z"), (0.35, 0.65))}
    for view in views.split():
        axis, plane, fractions = plans[view]
        keys = {}
        for f in fractions:
            i = int(round(low[axis] + f * (high[axis] - low[axis])))
            section = np.take(mask, i, axis=axis)       # rows, cols in zyx order
            rows_axis, cols_axis = [a for a in ZYX if a != ZYX[axis]]
            if (rows_axis, cols_axis) != plane:
                section = section.T
            points = edge(section)
            if points is None or len(points) < 3:
                continue
            scale = np.array([voxel[ZYX.index(plane[0])],
                              voxel[ZYX.index(plane[1])]])
            keys[float(i * voxel[axis])] = [tuple(p) for p in points * scale]
        if keys:
            outlines[plane] = keys
    return outlines


def topology(kept):
    """Loose pieces (beyond the largest), the volume of closed hollows (in
    voxels), and how many XY slices have a hole in the shape."""
    pieces, count = ndi.label(kept)
    hollow = int((ndi.binary_fill_holes(kept) & ~kept).sum())
    holed = sum(int((ndi.binary_fill_holes(s) & ~s).any()) for s in kept)
    return max(count - 1, 0), hollow, holed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("demo")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--views", default="xy xz yz")
    parser.add_argument("--steps", action="store_true",
                        help="score the shape after each polygon, in the "
                             "order a person might draw them")
    args = parser.parse_args(argv)
    if args.steps:
        return steps(args)
    truth = np.load(Path(args.demo) / f"{NAME}_truth.npz")
    labels, voxel = truth["labels"], truth["voxel_um"].astype(float)
    names = json.loads(str(truth["names"]))
    level = vc.Level("", None, list(voxel), [0.0, 0.0, 0.0])
    print(f"{'object':32s} kept  extra  other  pieces hollow holed-slices  s")
    for k, name in enumerate(names, start=1):
        mask = labels == k
        outlines = drawn_outlines(mask, voxel, args.views)
        idx = np.argwhere(mask)
        pad = np.ceil(np.array([30.0, 30.0, 30.0]) / voxel).astype(int)
        first = np.maximum(idx.min(0) - pad, 0)
        stop = np.minimum(idx.max(0) + pad + 1, labels.shape)
        shape = tuple(stop - first)
        t0 = time.perf_counter()
        kept = np.broadcast_to(vc.outline_mask(
            level, ZYX, list(first), shape, outlines, spatial=ZYX,
            margin=args.margin * min(voxel)), shape).copy()
        seconds = time.perf_counter() - t0
        block = tuple(slice(a, b) for a, b in zip(first, stop))
        obj, lab = mask[block], labels[block]
        other = kept & (lab > 0) & (lab != k)
        pieces, hollow, holed = topology(kept)
        print(f"{k} {name:30s} {(kept & obj).sum() / obj.sum():5.1%} "
              f"{(kept & ~obj).sum() / obj.sum():6.1%} "
              f"{other.sum() / obj.sum():6.1%} {pieces:6d} {hollow:6d} "
              f"{holed:6d} {seconds:5.1f}")


ORDER = [("y", "x", 1), ("y", "x", 0), ("y", "x", 2), ("z", "x", 0),
         ("z", "x", 1), ("y", "z", 0), ("y", "z", 1)]


def steps(args):
    """The shape after each polygon (XY middle, XY first, XY last, XZ,
    XZ, YZ, YZ): kept, extra, loose pieces, hollow voxels, holed slices."""
    truth = np.load(Path(args.demo) / f"{NAME}_truth.npz")
    labels, voxel = truth["labels"], truth["voxel_um"].astype(float)
    names = json.loads(str(truth["names"]))
    level = vc.Level("", None, list(voxel), [0.0, 0.0, 0.0])
    for k, name in enumerate(names, start=1):
        mask = labels == k
        full = drawn_outlines(mask, voxel)
        idx = np.argwhere(mask)
        pad = np.ceil(np.array([30.0, 30.0, 30.0]) / voxel).astype(int)
        first = np.maximum(idx.min(0) - pad, 0)
        stop = np.minimum(idx.max(0) + pad + 1, labels.shape)
        shape = tuple(stop - first)
        block = tuple(slice(a, b) for a, b in zip(first, stop))
        obj = mask[block]
        print(f"{k} {name}")
        outlines = {}
        for n, (a, b, j) in enumerate(ORDER, start=1):
            keys = full.get((a, b), {})
            if j >= len(keys):
                continue
            depth = sorted(keys)[j]
            outlines.setdefault((a, b), {})[depth] = keys[depth]
            kept = np.broadcast_to(vc.outline_mask(
                level, ZYX, list(first), shape, outlines, spatial=ZYX),
                shape).copy()
            pieces, hollow, holed = topology(kept)
            drawn = ", ".join(f"{p[0]}{p[1]}:{len(v)}"
                              for p, v in outlines.items())
            print(f"   {n} polygons ({drawn:18s}) kept "
                  f"{(kept & obj).sum() / obj.sum():5.1%} extra "
                  f"{(kept & ~obj).sum() / obj.sum():6.1%}  pieces {pieces}"
                  f"  hollow {hollow}  holed slices {holed}")


if __name__ == "__main__":
    main()
