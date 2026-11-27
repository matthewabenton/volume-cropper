"""Demo data: a multicolour embryo time-lapse (synthetic light sheet).

The user's request (2026-11-09): "make something more interesting that
looks like an actual multicolour embryo, like one in the synthetic embryos
image". An embryo's trunk and tail, drawn like object 1 of
make_demo_embryos.py (neural tube with a lumen, notochord, paired somites,
presomitic mesoderm (PSM), tail bud, surface ectoderm; channels nuclei,
membranes and a marker, imaged with the same light-sheet model), lies in
agarose with a small gastruloid-like aggregate beside it, over T
timepoints INTERVAL minutes apart:

- it drifts about 250 µm and turns slowly (about 28°), and grows a little;
- a new pair of somites buds off the PSM about every 3 timepoints;
- the marker shows waves running forward through the PSM, one per new
  somite (like the segmentation clock), and stays bright in the tail bud;
- its cells jitter a little, and some divide (condensed, bright nuclei that
  come and go);
- the gastruloid stays where it is.

The nuclei are placed once, in the embryo's own frame, and move with it,
so each nucleus keeps its shape and brightness from timepoint to timepoint.

Writes OUTPUT/embryo_timelapse_demo.zarr (OME-Zarr 0.5, t c z y x, 3
levels) and OUTPUT/embryo_timelapse_demo_truth.npz (labels per timepoint:
0 = none, 1 = embryo, 2 = gastruloid). Synthetic data only; seed 0.

Usage: python make_demo_embryo_timelapse.py OUTPUT_FOLDER [--step N]
                                             [--layout]
--step N: round every value to a multiple of N (mildly lossy, for a small
          demo: the user's wish of 2026-11-09). Measured on timepoint 7:
          N = 16 makes it 7x smaller on disk (20.1 -> 2.9 MB); the mean
          error inside the embryo is 4 counts, against about 22 counts of
          photon noise there; the background (camera offset and read
          noise) becomes flat. Default 1: exact.
--layout: only where the objects lie at the first, middle and last
          timepoint, and the gap between them.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from make_demo_embryos import (CAVITY, CHANNELS, DENSE, EPITHELIUM,
                               MESENCHYME, NUCLEI, SQUAMOUS, Gastruloid,
                               Result, Trunk, debris, grid_points, image_it,
                               membranes, smoothstep, unit)
from make_demo_tails import _block_mean

NAME = "embryo_timelapse_demo"
T, INTERVAL = 12, 15.0                      # timepoints, minutes apart
EXTENT_UM = (400.0, 900.0, 900.0)           # z, y, x
VOXEL = np.array([4.0, 2.5, 2.5])
FACTORS = [(1, 1, 1), (1, 2, 2), (2, 4, 4)]
# The embryo in its own frame (µm, z y x) and how it moves per timepoint:
# turned about CENTRE (in the y-x plane), grown, shifted.
TRUNK = [(200, 120, 250), (230, 300, 140), (170, 480, 270), (205, 600, 370)]
CENTRE = np.array([200.0, 360.0, 260.0])
DRIFT = np.array([0.0, 11.0, 21.0])
TURN = np.deg2rad(2.5)
GROW = 0.012
JITTER = 1.0                                # µm per timepoint, per cell
GASTRULOID = [(215, 765, 60), (225, 790, 105), (205, 820, 155),
              (215, 840, 200)]
GASTRULOID_RADIUS = 50.0
SOMITE_SPACING = 2 * 38 + 8.0               # as Trunk draws them
SOMITES_FIRST = 0.34                        # fraction with somites at t = 0
EVERY = 3                                   # timepoints per new somite pair
WAVELENGTH = 150.0                          # µm, of the marker waves


class ClockTrunk(Trunk):
    """Trunk at a timepoint: somites up to `somite_end` of its length, the
    PSM behind them, and marker waves in the PSM at `phase` (Trunk's own
    drawing otherwise)."""
    name = "embryo trunk and tail (time-lapse)"

    def __init__(self, controls):
        super().__init__(controls)
        self.at(0)

    def at(self, t):
        self.somite_end = SOMITES_FIRST + t * SOMITE_SPACING / (
            EVERY * self.curve.length)
        self.phase = t / EVERY

    def classify(self, p):
        res = Result(len(p))
        i, s, arc, t, n, b = self.curve.local(p)
        rn, rb = self.radii(s)
        f = (n / rn) ** 2 + (b / rb) ** 2 + (t / rn) ** 2 - 1
        grad = self.curve.frame_vector(i, t / rn ** 2, n / rn ** 2,
                                       b / rb ** 2)
        res.depth = np.minimum(-f / (2 * np.linalg.norm(grad, axis=1)
                                     + 1e-9), 1e4).astype(np.float32)
        res.inside = f <= 0
        res.set(np.ones(len(p), bool), MESENCHYME)
        side = np.sign(b) + (b == 0)
        lateral, dorsal = 0.55 * rb, 0.02 * rn
        radius = 38 - 8 * s
        k = np.round((arc - 30) / SOMITE_SPACING)
        centre_arc = 30 + k * SOMITE_SPACING
        in_range = (k >= 0) & (centre_arc < self.somite_end
                               * self.curve.length)
        dz = np.stack([arc - centre_arc, (n - dorsal) / 1.15,
                       b - side * lateral], axis=1)
        dist = np.linalg.norm(dz, axis=1)
        somite = in_range & (dist <= radius)
        res.set(somite, MESENCHYME)
        rim = somite & (dist > radius - 11)
        rim_axis = unit(self.curve.frame_vector(i, dz[:, 0], dz[:, 1],
                                                dz[:, 2]))
        res.set(rim, EPITHELIUM, rim_axis)
        psm_r = 0.36 * rb
        psm = ((s >= self.somite_end - 0.04)
               & (np.hypot(n - dorsal, b - side * lateral) <= psm_r)
               & ~somite)
        gradient = smoothstep((s - self.somite_end + 0.08) / 0.5) ** 1.3
        # waves running forward from the tail (one per new somite pair)
        from_tail = self.curve.length - arc
        wave = 0.5 + 0.5 * np.cos(2 * np.pi * (from_tail / WAVELENGTH
                                               - self.phase))
        res.set(psm, DENSE, marker=gradient * (0.15 + 0.85 * wave))
        bud = s > 0.88
        res.set(bud, DENSE, marker=np.ones(len(p), np.float32))
        noto = (np.hypot(n + 0.12 * rn, b) <= 11) & (s < 0.92)
        res.set(noto, DENSE)
        n0, an, ab = 0.42 * rn, 0.42 * rn, 0.26 * rb
        rho = np.hypot((n - n0) / an, b / ab)
        tube = (rho <= 1) & (s < 0.86)
        nt_axis = unit(self.curve.frame_vector(i, np.zeros_like(n), n - n0, b))
        res.set(tube, EPITHELIUM, nt_axis)
        lumen = tube & (np.hypot((n - n0) / (0.7 * an), b / (0.13 * ab)) <= 1)
        res.set(lumen, CAVITY)
        ecto = res.inside & (res.depth < 9)
        res.set(ecto, SQUAMOUS, unit(grad))
        return res


def pose(t):
    """The embryo's turn (3 x 3, about z), growth and shift at t."""
    c, s = np.cos(TURN * t), np.sin(TURN * t)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return rot, 1 + GROW * t, DRIFT * t


def to_lab(body, t):
    rot, scale, shift = pose(t)
    return CENTRE + shift + scale * (body - CENTRE) @ rot.T


def to_body(lab, t):
    rot, scale, shift = pose(t)
    return CENTRE + ((lab - CENTRE - shift) @ rot) / scale


def truth(t, trunk, gastruloid, shape):
    """Labels (1 embryo, 2 gastruloid) and tissue of every voxel at t."""
    labels = np.zeros(shape, np.uint8)
    tissue = np.zeros(shape, np.uint8)               # agarose
    depth = np.full(shape, -1e4, np.float32)
    for label, obj, moving in ((1, trunk, True), (2, gastruloid, False)):
        low, high = obj.bbox()
        if moving:
            corners = np.array([[a, b, c] for a in (low[0], high[0])
                                for b in (low[1], high[1])
                                for c in (low[2], high[2])])
            lab = to_lab(corners, t)
            low, high = lab.min(0), lab.max(0)
        lo = np.clip(np.floor(low / VOXEL).astype(int), 0, shape)
        hi = np.clip(np.ceil(high / VOXEL).astype(int) + 1, 0, shape)
        if np.any(hi <= lo):
            continue
        block = tuple(slice(a, b) for a, b in zip(lo, hi))
        p = grid_points(lo, hi, VOXEL)
        res = obj.classify(to_body(p, t) if moving else p)
        size = tuple(hi - lo)
        d = res.depth.reshape(size) * (pose(t)[1] if moving else 1.0)
        win = res.inside.reshape(size) & (d > depth[block])
        depth[block] = np.where(win, d, depth[block])
        labels[block] = np.where(win, label, labels[block])
        tissue[block] = np.where(win, res.tissue.reshape(size), tissue[block])
    return labels, tissue


def spacing_of(codes):
    table = np.full(8, NUCLEI[DENSE]["spacing"])
    for code, params in NUCLEI.items():
        table[code] = params["spacing"]
    return table[codes]


def place_nuclei(obj, rng):
    """Nuclei of an object (in its own frame), packed as the demo packs
    them: at its tissues' spacing, no two closer than 0.8 x spacing."""
    low, high = obj.bbox()
    count = int(np.prod(high - low) * 2.5 / 6.5 ** 3)
    p = rng.uniform(low, high, (count, 3))
    res = obj.classify(p)
    space = spacing_of(res.tissue)
    keep = res.inside & (res.tissue != CAVITY) \
        & (rng.random(count) < (6.5 / np.maximum(space, 6.5)) ** 3)
    p, space = p[keep], space[keep]
    order = rng.permutation(len(p))
    p, space = p[order], space[order]
    near = cKDTree(p).query_ball_point(p, 0.8 * space, workers=-1)
    accepted = np.zeros(len(p), bool)
    for k, others in enumerate(near):
        if not accepted[others].any():
            accepted[k] = True
    return p[accepted]


def splat(points, codes, axes, marker, look, dividing, shape):
    """The demo's nuclei drawing (make_demo_embryos.splat_nuclei), with each
    nucleus' own size, brightness and orientation (`look`) kept from
    timepoint to timepoint."""
    params = [NUCLEI.get(c, NUCLEI[DENSE]) for c in codes]
    semi = np.array([q["axes"] for q in params], np.float32) * look["size"]
    bright = np.array([q["bright"] for q in params], np.float32) \
        * look["bright"]
    semi[dividing] *= 0.6
    bright[dividing] *= 2.2
    first = unit(np.where(np.isnan(axes[:, :1]), look["axis"], axes))
    second = unit(np.cross(first, look["second"]))
    rot = np.stack([first, second, np.cross(first, second)], axis=2)
    nuclei = np.zeros(int(np.prod(shape)), np.float32)
    marks = np.zeros(int(np.prod(shape)), np.float32)
    reach = np.ceil((semi.max(1) * 1.35 + 1.0)[:, None] / VOXEL).astype(int)
    keys = reach[:, 0] * 10000 + reach[:, 1] * 100 + reach[:, 2]
    for key in np.unique(keys):
        group = np.nonzero(keys == key)[0]
        wz, wy, wx = reach[group[0]]
        offsets = np.stack(np.meshgrid(np.arange(-wz, wz + 1),
                                       np.arange(-wy, wy + 1),
                                       np.arange(-wx, wx + 1), indexing="ij"),
                           -1).reshape(-1, 3)
        for start in range(0, len(group), 3000):
            g = group[start:start + 3000]
            cells = np.round(points[g] / VOXEL).astype(int)[:, None, :] \
                + offsets[None]
            rel = (cells * VOXEL - points[g][:, None, :]).astype(np.float32)
            local = np.einsum("kmj,kja->kma", rel, rot[g])
            q = np.sqrt(np.sum((local / semi[g][:, None, :]) ** 2, axis=2))
            value = 1 / (1 + np.exp((q - 1) / 0.15))
            ok = np.all((cells >= 0) & (cells < shape), axis=2)
            flat = np.ravel_multi_index(tuple(cells[ok].T), shape)
            np.add.at(nuclei, flat, (value * bright[g][:, None])[ok])
            np.add.at(marks, flat, (value * marker[g][:, None])[ok])
    return nuclei.reshape(shape), marks.reshape(shape)


def make(out: Path, layout_only=False, seed=0, step=1, log=print):
    shape = tuple(int(round(e / v)) for e, v in zip(EXTENT_UM, VOXEL))
    rng = np.random.default_rng(seed)
    trunk = ClockTrunk(TRUNK)
    gastruloid = Gastruloid(GASTRULOID, radius=GASTRULOID_RADIUS)
    if layout_only:
        from scipy import ndimage as ndi
        for t in (0, T // 2, T - 1):
            trunk.at(t)
            labels, _ = truth(t, trunk, gastruloid, shape)
            for k in (1, 2):
                idx = np.argwhere(labels == k)
                lo, hi = idx.min(0) * VOXEL, (idx.max(0) + 1) * VOXEL
                log(f"t {t}: object {k} z {lo[0]:.0f}-{hi[0]:.0f} y "
                    f"{lo[1]:.0f}-{hi[1]:.0f} x {lo[2]:.0f}-{hi[2]:.0f} µm "
                    f"(field {EXTENT_UM})")
            gap = ndi.distance_transform_edt(labels != 1, sampling=VOXEL)[
                labels == 2].min()
            log(f"t {t}: gap between them {gap:.0f} µm")
        return
    # 1. nuclei, once: the embryo's in its own frame, the gastruloid's
    t0 = time.perf_counter()
    trunk.at(0)
    body = place_nuclei(trunk, rng)
    still = place_nuclei(gastruloid, rng)
    n_body, n = len(body), len(body) + len(still)
    look = {"size": rng.uniform(0.85, 1.15, (n, 3)).astype(np.float32),
            "bright": rng.lognormal(0.0, 0.25, n).astype(np.float32),
            "axis": unit(rng.standard_normal((n, 3))),
            "second": unit(rng.standard_normal((n, 3)))}
    walk = np.cumsum(rng.normal(0, JITTER, (T, n, 3)), axis=0)
    clock = rng.integers(0, 66, n)            # ~1.5 % divide at a time
    specks = debris(shape, VOXEL, rng, count=70)
    log(f"{n_body} + {len(still)} nuclei: {time.perf_counter() - t0:.0f} s")

    # 2. the image, timepoint by timepoint
    path = out / f"{NAME}.zarr"
    if path.exists():
        shutil.rmtree(path)
    import zarr
    from zarr.codecs import BloscCodec
    root = zarr.open_group(path, mode="w", zarr_format=3)
    arrays = []
    for k, f in enumerate(FACTORS):
        level_shape = (T, 3) + tuple(-(-n_ // ff) for n_, ff in zip(shape, f))
        arrays.append(root.create_array(
            name=str(k), shape=level_shape, dtype=np.uint16,
            chunks=(1, 1, 32, 64, 64), shards=(1, 1, 64, 128, 128),
            compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
            fill_value=0, dimension_names=list("tczyx")))
    all_labels = np.zeros((T,) + shape, np.uint8)
    for t in range(T):
        t0 = time.perf_counter()
        trunk.at(t)
        labels, tissue = truth(t, trunk, gastruloid, shape)
        all_labels[t] = labels
        moved = body + walk[t, :n_body]
        res = trunk.classify(moved)
        rot = pose(t)[0]
        points = np.concatenate([to_lab(moved, t), still + walk[t, n_body:]])
        sres = gastruloid.classify(still)
        codes = np.concatenate([res.tissue, sres.tissue])
        axes = np.concatenate([np.where(np.isnan(res.axis), np.nan,
                                        res.axis @ rot.T), sres.axis])
        marker = np.concatenate([res.marker, sres.marker])
        dividing = (clock + t) % 66 == 0
        nuclei, marks = splat(points, codes, axes.astype(np.float32),
                              marker, look, dividing, shape)
        membrane = membranes(points, codes, labels, tissue, VOXEL)
        image = image_it(nuclei, marks, membrane, np.zeros(shape, np.float32),
                         specks, labels, tissue, VOXEL, rng)
        if step > 1:                      # mildly lossy: see --step
            image = np.minimum(np.round(image / step) * step,
                               65535).astype(np.uint16)
        for k, f in enumerate(FACTORS):
            arrays[k][t] = image if f == (1, 1, 1) else np.stack(
                [_block_mean(c, f) for c in image])
        log(f"timepoint {t + 1}/{T}: {time.perf_counter() - t0:.0f} s, "
            f"{int(dividing.sum())} dividing")
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": "embryo time-lapse (demo)",
            "axes": [{"name": "t", "type": "time", "unit": "minute"},
                     {"name": "c", "type": "channel"}] + [
                {"name": a, "type": "space", "unit": "micrometer"}
                for a in "zyx"],
            "datasets": [{"path": str(k), "coordinateTransformations": [
                {"type": "scale", "scale": [INTERVAL, 1.0] + [
                    v * ff for v, ff in zip(VOXEL, f)]}]}
                for k, f in enumerate(FACTORS)]}],
        "omero": {"channels": list(CHANNELS)},
    }
    root.attrs["demo_generator"] = {
        "script": "make_demo_embryo_timelapse.py", "seed": seed,
        "timepoints": T, "interval_min": INTERVAL,
        "drift_um_per_timepoint": DRIFT.tolist(),
        "turn_deg_per_timepoint": float(np.rad2deg(TURN)),
        "growth_per_timepoint": GROW, "somite_every": EVERY,
        "values_rounded_to_multiples_of": step}
    np.savez_compressed(out / f"{NAME}_truth.npz", labels=all_labels,
                        voxel_um=VOXEL)
    log(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make(out, layout_only=args.layout, step=args.step)


if __name__ == "__main__":
    main()
