"""Demo data: embryo-like objects packed in tissue (synthetic lightsheet).

Makes a synthetic lightsheet-like volume of six objects inspired by mouse
development, lying close together and partly pressed against each other,
in a loose tissue (sparse cells, fibres, small vessels, debris), and saves
it as an OME-Zarr 0.5 image with a pyramid. The true object of every voxel
is known, so crops can be scored (how much of an object they keep, and
how much else).

Objects (labels 1-6):
  1 embryo trunk and tail: curved neural tube with a lumen, notochord,
    paired somites (epithelial rim, loose core), presomitic mesoderm (PSM)
    tapering into a tail bud, all under a thin surface ectoderm;
  2 blastocyst-like cyst: one-cell-thin shell around a fluid cavity, with
    an inner cell mass on one side;
  3 branching epithelial bud (narrow lumens) in a cap of mesenchyme, like
    an early gland or lung bud;
  4 looped heart-like tube: muscle wall, cardiac jelly, thin inner lining;
  5 limb bud: flat paddle of mesenchyme under an ectoderm with a ridge
    (AER) along its distal rim;
  6 gastruloid-like ovoid: dense aggregate with a marker at one pole.

Channels:
  0 "nuclei (demo)": every nucleus; shape, size, spacing and brightness
    depend on the tissue (round and dense, columnar epithelium, flat
    squamous, elongated muscle, sparse mesenchyme); some are dividing.
  1 "membranes (demo)": cell outlines between neighbouring nuclei, bright
    lumen surfaces, faint object surfaces; fibres between the objects.
  2 "marker (demo)": nuclear marker in the PSM and tail bud (graded),
    the gastruloid's pole and the bud epithelium.

Imaging: light sheet from the x = 0 side, dimmed by the tissue it has
crossed (with shadows behind debris), uneven sheet (stripes along x),
detection dimmer with depth (z), blur (more along z), scattered haze,
photon noise, camera offset and read noise; 16-bit.

Synthetic data only. Every random choice comes from one seed, and the
parameters are stored in the image's metadata ("demo_generator").

Usage::

    python make_demo_embryos.py OUTPUT_FOLDER [--voxel Z Y X] [--seed N]
    python make_demo_embryos.py OUTPUT_FOLDER --layout     (quick check)

writes OUTPUT_FOLDER/embryo_tissue_demo.zarr and the true labels as
OUTPUT_FOLDER/embryo_tissue_demo_truth.npz (labels: 0 = no object, 1-6;
tissue: the tissue type of every voxel, see TISSUES).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from make_demo_tails import write_ome_zarr

EXTENT_UM = (800.0, 1000.0, 1000.0)          # z, y, x
NAME = "embryo_tissue_demo"

# Tissue types (codes in the truth file).
AGAROSE, LOOSE, DENSE, EPITHELIUM, SQUAMOUS, MESENCHYME, MUSCLE, CAVITY = \
    range(8)
TISSUES = ["agarose (no tissue)", "loose tissue between objects",
           "dense (PSM, tail bud, inner cell mass, gastruloid)",
           "columnar epithelium", "squamous (flat) epithelium", "mesenchyme",
           "muscle", "cavity, lumen or cardiac jelly (no cells)"]
# Nuclei per tissue: semi-axes (µm; the first along the tissue's axis, e.g.
# apico-basal in an epithelium, or the normal of a flat one), typical
# spacing (µm), brightness, cell radius (µm, for the membranes) and
# membrane brightness.
NUCLEI = {
    DENSE: dict(axes=(4.6, 4.0, 3.8), spacing=9.0, bright=1.0, cell=6.5,
                membrane=0.8),
    EPITHELIUM: dict(axes=(6.5, 3.0, 2.8), spacing=6.5, bright=1.0, cell=5.5,
                     membrane=1.0),
    SQUAMOUS: dict(axes=(1.8, 6.0, 5.0), spacing=13.0, bright=0.9, cell=9.0,
                   membrane=0.9),
    MESENCHYME: dict(axes=(5.0, 3.5, 3.2), spacing=11.5, bright=0.75,
                     cell=8.0, membrane=0.6),
    MUSCLE: dict(axes=(7.5, 3.0, 2.8), spacing=9.0, bright=0.85, cell=7.0,
                 membrane=0.8),
    LOOSE: dict(axes=(5.5, 3.3, 3.0), spacing=19.0, bright=0.45, cell=8.0,
                membrane=0.4),
}
# Light: photons per unit of signal, per channel; camera offset and noise.
PHOTONS = (1400.0, 900.0, 1100.0)
OFFSET, READ_NOISE = 100.0, 2.0
PSF_SIGMA_UM = (2.5, 0.8, 0.8)               # z, y, x
HAZE_SIGMA_UM, HAZE = 12.0, 0.15
CHANNELS = [{"label": "nuclei (demo)", "color": "00FFFF"},
            {"label": "membranes (demo)", "color": "FF00FF"},
            {"label": "marker (demo)", "color": "FFFF00"}]


def unit(v):
    """Vectors scaled to length 1 (along the last axis)."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Curve:
    """A smooth centre line (cubic Bézier, 4 control points in µm as
    (z, y, x)), sampled densely, with a frame per sample: tangent T,
    normal N (as close to `up` as possible) and binormal B."""

    def __init__(self, controls, n: int = 2000, up=(1.0, 0.0, 0.0)):
        c = np.asarray(controls, dtype=float)
        s = np.linspace(0.0, 1.0, n)[:, None]
        self.points = ((1 - s) ** 3 * c[0] + 3 * (1 - s) ** 2 * s * c[1]
                       + 3 * (1 - s) * s ** 2 * c[2] + s ** 3 * c[3])
        self.s = s[:, 0]
        self.tangent = unit(np.gradient(self.points, axis=0))
        up = np.asarray(up, dtype=float)
        self.normal = unit(up - (self.tangent @ up)[:, None] * self.tangent)
        self.binormal = np.cross(self.tangent, self.normal)
        steps = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self.arc[-1])
        self.tree = cKDTree(self.points)

    def local(self, p):
        """For points p (n, 3): the nearest sample i, its position s (0-1)
        and arc length, and the offsets (t, n, b) along its frame; t is 0
        except beyond the ends (no sample of the curve lies further)."""
        _, i = self.tree.query(p, workers=-1)
        rel = p - self.points[i]
        t = dot(rel, self.tangent[i])
        last = len(self.points) - 1
        beyond = ((i == 0) & (t < 0)) | ((i == last) & (t > 0))
        return (i, self.s[i], self.arc[i], np.where(beyond, t, 0.0),
                dot(rel, self.normal[i]), dot(rel, self.binormal[i]))

    def frame_vector(self, i, t, n, b):
        """t T + n N + b B at samples i."""
        return (t[:, None] * self.tangent[i] + n[:, None] * self.normal[i]
                + b[:, None] * self.binormal[i])


def ellipsoid(p, centre, axes, radii):
    """Local coordinates q, approximate depth inside (µm, positive inside;
    first-order distance to the surface) and outward normal of an
    ellipsoid with orthonormal axes (rows) and radii."""
    q = (p - np.asarray(centre)) @ np.asarray(axes).T
    radii = np.asarray(radii, dtype=float)
    f = np.sum((q / radii) ** 2, axis=1) - 1
    grad = 2 * q / radii ** 2
    depth = -f / (np.linalg.norm(grad, axis=1) + 1e-9)
    return q, np.minimum(depth, 1e4), unit(grad @ np.asarray(axes))


def segment_distance(p, a, b):
    """Distance from points p to the segment a-b, the nearest point's
    position along it (0-1) and the direction from it to p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    u = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
    rel = p - (a + u[:, None] * ab)
    return np.linalg.norm(rel, axis=1), u, unit(rel)


def rotation(seed_vector):
    """Orthonormal axes (rows) with the first along seed_vector."""
    a = unit(np.asarray(seed_vector, dtype=float)[None])[0]
    helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else \
        np.array([0.0, 1.0, 0.0])
    b = unit(np.cross(a, helper)[None])[0]
    return np.stack([a, b, np.cross(a, b)])


class Result:
    """What an object says about points: inside it, how deep (µm), the
    tissue type, the tissue's axis (NaN = random) and the marker (0-1)."""

    def __init__(self, n):
        self.inside = np.zeros(n, dtype=bool)
        self.depth = np.full(n, -1e4, dtype=np.float32)
        self.tissue = np.full(n, MESENCHYME, dtype=np.uint8)
        self.axis = np.full((n, 3), np.nan, dtype=np.float32)
        self.marker = np.zeros(n, dtype=np.float32)

    def set(self, where, tissue, axis=None, marker=None):
        self.tissue[where] = tissue
        self.axis[where] = np.nan if axis is None else axis[where]
        if marker is not None:
            self.marker[where] = marker[where] if np.ndim(marker) else marker


class Trunk:
    """1: an embryo's trunk and tail (cross-section: dorsal = N)."""
    name = "embryo trunk and tail"

    def __init__(self, controls):
        self.curve = Curve(controls)

    def radii(self, s):
        return 92 - 32 * s, 122 - 52 * s        # dorso-ventral, lateral

    def bbox(self):
        r = 125
        return self.curve.points.min(0) - r, self.curve.points.max(0) + r

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
        # paraxial mesoderm: somites (s < 0.6) or PSM, on both sides
        side = np.sign(b) + (b == 0)
        lateral, dorsal = 0.55 * rb, 0.02 * rn
        radius = 38 - 8 * s
        spacing = 2 * 38 + 8.0
        k = np.round((arc - 30) / spacing)
        centre_arc = 30 + k * spacing
        in_somite_range = (k >= 0) & (centre_arc < 0.6 * self.curve.length)
        dz = np.stack([arc - centre_arc, (n - dorsal) / 1.15, b - side * lateral],
                      axis=1)
        dist = np.linalg.norm(dz, axis=1)
        somite = in_somite_range & (dist <= radius)
        res.set(somite, MESENCHYME)
        rim = somite & (dist > radius - 11)
        rim_axis = unit(self.curve.frame_vector(i, dz[:, 0], dz[:, 1],
                                                dz[:, 2]))
        res.set(rim, EPITHELIUM, rim_axis)
        psm_r = 0.36 * rb
        psm = ((s >= 0.55) & (np.hypot(n - dorsal, b - side * lateral) <= psm_r)
               & ~somite)
        gradient = smoothstep((s - 0.5) / 0.45) ** 1.3
        res.set(psm, DENSE, marker=gradient)
        bud = s > 0.88
        res.set(bud, DENSE, marker=np.ones(len(p), np.float32))
        # notochord, ventral to the neural tube
        noto = (np.hypot(n + 0.12 * rn, b) <= 11) & (s < 0.92)
        res.set(noto, DENSE)
        # neural tube (dorsal), with a slit-like lumen
        n0, an, ab = 0.42 * rn, 0.42 * rn, 0.26 * rb
        rho = np.hypot((n - n0) / an, b / ab)
        tube = (rho <= 1) & (s < 0.86)
        nt_axis = unit(self.curve.frame_vector(i, np.zeros_like(n), n - n0, b))
        res.set(tube, EPITHELIUM, nt_axis)
        lumen = tube & (np.hypot((n - n0) / (0.7 * an), b / (0.13 * ab)) <= 1)
        res.set(lumen, CAVITY)
        # surface ectoderm: the outer 9 µm
        ecto = res.inside & (res.depth < 9)
        res.set(ecto, SQUAMOUS, unit(grad))
        return res


class Cyst:
    """2: a blastocyst-like cyst."""
    name = "blastocyst-like cyst"

    def __init__(self, centre, radii, icm_direction):
        self.centre, self.radii = np.asarray(centre, float), np.asarray(radii)
        self.axes = np.eye(3)
        d = unit(np.asarray(icm_direction, float)[None])[0]
        along = 1 / np.sqrt(np.sum((d / self.radii) ** 2))   # radius along d
        self.icm_centre = self.centre + 0.62 * along * d
        self.icm_radius = 52.0

    def bbox(self):
        return self.centre - self.radii - 5, self.centre + self.radii + 5

    def classify(self, p):
        res = Result(len(p))
        _, depth, normal = ellipsoid(p, self.centre, self.axes, self.radii)
        res.depth, res.inside = depth.astype(np.float32), depth >= 0
        res.set(np.ones(len(p), bool), CAVITY)
        icm = np.linalg.norm(p - self.icm_centre, axis=1) <= self.icm_radius
        res.set(icm, DENSE)
        res.set(depth < 10, SQUAMOUS, normal)
        return res


class Bud:
    """3: a branching epithelial bud in a cap of mesenchyme."""
    name = "branching bud in mesenchyme"

    def __init__(self, segments, epithelium=19.0, tip=25.0, cap=48.0):
        self.segments = [(np.asarray(a, float), np.asarray(b, float), is_tip)
                         for a, b, is_tip in segments]
        self.epithelium, self.tip, self.cap = epithelium, tip, cap

    def bbox(self):
        ends = np.concatenate([[a, b] for a, b, _ in self.segments])
        r = self.tip + self.cap + 20
        return ends.min(0) - r, ends.max(0) + r

    def classify(self, p):
        res = Result(len(p))
        best = np.full(len(p), np.inf)
        direction = np.zeros((len(p), 3))
        radius = np.full(len(p), self.epithelium)
        for a, b, is_tip in self.segments:
            d, u, rel = segment_distance(p, a, b)
            # tips swell into a bulb
            r = self.epithelium + ((self.tip - self.epithelium)
                                   * smoothstep((u - 0.6) / 0.4) if is_tip
                                   else 0.0)
            closer = (d - r) < (best - radius)
            best = np.where(closer, d, best)
            radius = np.where(closer, r, radius)
            direction[closer] = rel[closer]
        gap = best - radius                       # outside the epithelium
        # the cap's thickness varies smoothly around the bud
        wobble = 1 + 0.25 * (np.sin(p[:, 0] / 37) * np.cos(p[:, 1] / 53)
                             + np.sin(p[:, 2] / 41))
        cap = self.cap * wobble
        res.depth = np.minimum(cap - gap, 1e4).astype(np.float32)
        res.inside = gap <= cap
        res.set(np.ones(len(p), bool), MESENCHYME)
        epi = gap <= 0
        res.set(epi, EPITHELIUM, direction, marker=np.ones(len(p), np.float32))
        res.set(best <= 4.5, CAVITY)
        return res


class HeartTube:
    """4: a looped, heart-like tube."""
    name = "looped heart-like tube"

    def __init__(self, controls, outer=56.0):
        self.curve = Curve(controls)
        self.outer = outer

    def bbox(self):
        r = self.outer + 5
        return self.curve.points.min(0) - r, self.curve.points.max(0) + r

    def classify(self, p):
        res = Result(len(p))
        i, s, arc, t, n, b = self.curve.local(p)
        rho = np.sqrt(n ** 2 + b ** 2 + t ** 2)
        res.depth = (self.outer - rho).astype(np.float32)
        res.inside = rho <= self.outer
        radial = unit(self.curve.frame_vector(i, t, n, b))
        around = unit(self.curve.frame_vector(i, np.zeros_like(n), -b, n))
        res.set(np.ones(len(p), bool), MUSCLE, around)
        res.set(rho < 30, CAVITY)                   # cardiac jelly
        res.set(rho < 22, SQUAMOUS, radial)         # inner lining
        res.set(rho < 17, CAVITY)                   # lumen
        return res


class LimbBud:
    """5: a limb bud: a flat paddle, cut flat at its base."""
    name = "limb bud"

    def __init__(self, centre, thickness_axis, length_axis, radii):
        self.centre = np.asarray(centre, float)
        u1 = unit(np.asarray(thickness_axis, float)[None])[0]
        u3 = unit(np.asarray(length_axis, float)[None])[0]
        u3 = unit((u3 - (u3 @ u1) * u1)[None])[0]
        self.axes = np.stack([u1, np.cross(u3, u1), u3])
        self.radii = np.asarray(radii, float)     # thickness, width, length

    def bbox(self):
        r = self.radii.max() + 5
        return self.centre - r, self.centre + r

    def classify(self, p):
        res = Result(len(p))
        q, depth, normal = ellipsoid(p, self.centre, self.axes, self.radii)
        base = q[:, 2] + 0.55 * self.radii[2]       # >= 0 above the cut
        depth = np.minimum(depth, base)
        res.depth, res.inside = depth.astype(np.float32), depth >= 0
        res.set(np.ones(len(p), bool), MESENCHYME)
        ridge = (q[:, 2] > 0.35 * self.radii[2]) & (np.abs(q[:, 0]) < 22)
        ecto = (depth < 7) & (base > 7)
        res.set(ecto, SQUAMOUS, normal)
        res.set((depth < 18) & ridge & (base > 7), EPITHELIUM, normal)
        return res


class Gastruloid:
    """6: a gently bent, gastruloid-like ovoid with a marker at one pole."""
    name = "gastruloid-like ovoid"

    def __init__(self, controls, radius=97.0):
        self.curve = Curve(controls)
        self.radius = radius

    def bbox(self):
        r = self.radius + 5
        return self.curve.points.min(0) - r, self.curve.points.max(0) + r

    def classify(self, p):
        res = Result(len(p))
        i, s, arc, t, n, b = self.curve.local(p)
        r = self.radius * (1 - np.abs(2 * s - 1) ** 2.4) ** 0.45
        rho = np.hypot(n, b)
        res.depth = np.where(t != 0, -np.abs(t), r - rho).astype(
            np.float32)
        res.inside = (rho <= r) & (t == 0)
        res.set(np.ones(len(p), bool), DENSE,
                marker=smoothstep((s - 0.68) / 0.27))
        return res


class Vessels:
    """Small vessels in the loose tissue (not objects): a flat lining
    around a lumen."""

    def __init__(self, curves, radius=12.0):
        self.curves = [Curve(c) for c in curves]
        self.radius = radius

    def classify(self, p):
        """(tissue code or 0, axis) for points p."""
        tissue = np.zeros(len(p), dtype=np.uint8)
        axis = np.full((len(p), 3), np.nan, dtype=np.float32)
        for curve in self.curves:
            d, _ = curve.tree.query(p, distance_upper_bound=self.radius + 2,
                                    workers=-1)
            near = np.nonzero(np.isfinite(d))[0]
            if not len(near):
                continue
            i, s, arc, t, n, b = curve.local(p[near])
            rho = np.hypot(n, b)
            wall = (rho <= self.radius) & (t == 0)
            tissue[near[wall]] = SQUAMOUS
            axis[near[wall]] = unit(curve.frame_vector(
                i, np.zeros_like(n), n, b))[wall]
            tissue[near[wall & (rho < self.radius - 4)]] = CAVITY
        return tissue, axis

    def sample(self, spacing, rng):
        """Candidate nuclei in the vessels' linings, with their axes."""
        points, axes = [], []
        for curve in self.curves:
            count = int(curve.length * 2 * np.pi * 10 / spacing ** 2 * 2.5)
            k = rng.integers(0, len(curve.points), count)
            angle = rng.uniform(0, 2 * np.pi, count)
            r = rng.uniform(self.radius - 3.5, self.radius - 0.5, count)
            radial = (np.cos(angle)[:, None] * curve.normal[k]
                      + np.sin(angle)[:, None] * curve.binormal[k])
            points.append(curve.points[k] + r[:, None] * radial)
            axes.append(radial)
        return np.concatenate(points), np.concatenate(axes).astype(np.float32)


def build_objects():
    """The six objects and the vessels, placed in the 0.8 x 1.0 x 1.0 mm
    volume (µm, z y x)."""
    objects = [
        Trunk([(400, 140, 200), (430, 340, 80), (370, 560, 250),
               (410, 700, 380)]),
        Cyst(centre=(350, 830, 300), radii=(95, 112, 104),
             icm_direction=(1.0, 0.2, 0.9)),
        Bud([((470, 150, 620), (450, 270, 670), False),
             ((450, 270, 670), (420, 340, 790), False),
             ((450, 270, 670), (500, 370, 600), False),
             ((420, 340, 790), (370, 320, 880), True),
             ((420, 340, 790), (470, 440, 840), True),
             ((500, 370, 600), (560, 450, 560), True),
             ((500, 370, 600), (420, 450, 610), True)]),
        HeartTube([(280, 470, 560), (160, 600, 770), (390, 640, 500),
                   (290, 780, 690)]),
        LimbBud(centre=(535, 610, 705), thickness_axis=(1.0, 0.1, 0.35),
                length_axis=(0.2, 0.4, 1.0), radii=(68, 165, 150)),
        Gastruloid([(470, 230, 290), (510, 330, 390), (450, 430, 410),
                    (480, 520, 500)]),
    ]
    vessels = Vessels([
        [(150, 0, 420), (200, 300, 380), (120, 600, 450), (160, 1000, 400)],
        [(700, 0, 900), (650, 300, 960), (720, 600, 880), (660, 1000, 940)],
        [(480, 600, 0), (520, 620, 300), (450, 560, 500), (500, 600, 1000)],
    ])
    return objects, vessels


def grid_points(low_idx, high_idx, voxel):
    """Voxel centres (µm) of the block [low_idx, high_idx)."""
    axes = [np.arange(a, b) * v for a, b, v in zip(low_idx, high_idx, voxel)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def truth_maps(objects, vessels, shape, voxel, rng):
    """Labels (0 = no object, 1-6) and tissue type of every voxel. Where
    objects overlap, a voxel belongs to the one it lies deeper inside, so
    objects pressed together meet at a flat contact face."""
    labels = np.zeros(shape, dtype=np.uint8)
    tissue = np.zeros(shape, dtype=np.uint8)
    depth = np.full(shape, -1e4, dtype=np.float32)
    for label, obj in enumerate(objects, start=1):
        low, high = obj.bbox()
        lo = np.clip(np.floor(low / voxel).astype(int), 0, shape)
        hi = np.clip(np.ceil(high / voxel).astype(int) + 1, 0, shape)
        if np.any(hi <= lo):
            continue
        block = tuple(slice(a, b) for a, b in zip(lo, hi))
        res = obj.classify(grid_points(lo, hi, voxel))
        size = tuple(hi - lo)
        win = res.inside.reshape(size) & (res.depth.reshape(size)
                                          > depth[block])
        depth[block] = np.where(win, res.depth.reshape(size), depth[block])
        labels[block] = np.where(win, label, labels[block])
        tissue[block] = np.where(win, res.tissue.reshape(size), tissue[block])
    # Loose tissue around and between the objects, up to 45-95 µm from
    # them, with vessels running through it; agarose beyond.
    outside = ndi.distance_transform_edt(labels == 0, sampling=voxel)
    patch = smooth_noise(shape, voxel, 60.0, rng)
    loose = (labels == 0) & (outside <= 70 + 25 * patch)
    tissue[loose] = LOOSE
    where = np.nonzero(loose)
    points = np.column_stack(where) * voxel
    vessel_tissue, _ = vessels.classify(points)
    hit = vessel_tissue > 0
    tissue[tuple(w[hit] for w in where)] = vessel_tissue[hit]
    return labels, tissue


def smooth_noise(shape, voxel, scale_um, rng, coarse=4):
    """Smooth random field in about [-1, 1] with features of scale_um."""
    sigma = np.maximum(scale_um / (np.asarray(voxel) * coarse), 0.5)
    small = tuple(max(int(np.ceil(n / coarse)), 1) for n in shape)
    field = ndi.gaussian_filter(rng.standard_normal(small).astype(np.float32),
                                sigma)
    field /= 2 * field.std() + 1e-9
    zoomed = ndi.zoom(field, [n / m for n, m in zip(shape, small)], order=1)
    return np.clip(zoomed[tuple(slice(0, n) for n in shape)], -1, 1)


def sample_nuclei(objects, vessels, labels, tissue, voxel, rng):
    """Nucleus centres with their tissue, axis and marker: candidates at a
    density set by each tissue's spacing, thinned so that no two are closer
    than 0.8 x their spacing (random sequential packing)."""
    shape = np.array(labels.shape)
    spacing = np.zeros(8)
    for code, params in NUCLEI.items():
        spacing[code] = params["spacing"]
    found = []

    def voxel_of(p):
        return tuple(np.clip(np.round(p / voxel).astype(int), 0,
                             shape - 1).T)

    for label, obj in enumerate(objects, start=1):
        low, high = obj.bbox()
        low, high = np.maximum(low, 0), np.minimum(high, shape * voxel)
        count = int(np.prod(high - low) * 2.5 / 6.5 ** 3)
        p = rng.uniform(low, high, (count, 3))
        res = obj.classify(p)
        keep = res.inside & (labels[voxel_of(p)] == label) \
            & (res.tissue != CAVITY)
        keep &= rng.random(count) < (6.5 / np.maximum(spacing[res.tissue],
                                                     6.5)) ** 3
        found.append((p[keep], res.tissue[keep], res.axis[keep],
                      res.marker[keep]))
    # loose tissue and vessel linings, with patchy density
    count = int(np.prod(shape * voxel) * 2.5 / NUCLEI[LOOSE]["spacing"] ** 3)
    p = rng.uniform(0, shape * voxel, (count, 3))
    code = tissue[voxel_of(p)]
    patch = smooth_noise(tuple(shape), voxel, 80.0, rng)[voxel_of(p)]
    keep = (code == LOOSE) & (rng.random(count) < 0.55 + 0.45 * patch)
    p = p[keep]
    found.append((p, np.full(len(p), LOOSE, np.uint8),
                  np.full((len(p), 3), np.nan, np.float32),
                  np.zeros(len(p), np.float32)))
    q, qaxis = vessels.sample(NUCLEI[SQUAMOUS]["spacing"], rng)
    inside = np.all((q >= 0) & (q < shape * voxel), axis=1)
    q, qaxis = q[inside], qaxis[inside]
    ok = (tissue[voxel_of(q)] == SQUAMOUS) & (labels[voxel_of(q)] == 0)
    found.append((q[ok], np.full(ok.sum(), SQUAMOUS, np.uint8), qaxis[ok],
                  np.zeros(ok.sum(), np.float32)))
    points = np.concatenate([f[0] for f in found])
    codes = np.concatenate([f[1] for f in found])
    axes = np.concatenate([f[2] for f in found])
    marker = np.concatenate([f[3] for f in found])
    # random sequential packing
    order = rng.permutation(len(points))
    points, codes, axes, marker = (a[order] for a in (points, codes, axes,
                                                      marker))
    radius = 0.8 * spacing[codes]
    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, radius, workers=-1)
    accepted = np.zeros(len(points), dtype=bool)
    for k, near in enumerate(neighbours):
        if not accepted[near].any():
            accepted[k] = True
    return (points[accepted], codes[accepted], axes[accepted],
            marker[accepted])


def splat_nuclei(points, codes, axes, marker, shape, voxel, rng):
    """Draw every nucleus as a soft-edged ellipsoid with its tissue's
    shape, oriented along its axis (random where there is none), into the
    nuclei and marker channels."""
    n = len(points)
    semi = np.array([NUCLEI[c]["axes"] for c in codes], dtype=np.float32)
    semi *= rng.uniform(0.85, 1.15, (n, 3)).astype(np.float32)
    bright = np.array([NUCLEI[c]["bright"] for c in codes], np.float32)
    bright *= rng.lognormal(0.0, 0.25, n).astype(np.float32)
    dividing = rng.random(n) < 0.015               # condensed, bright
    semi[dividing] *= 0.6
    bright[dividing] *= 2.2
    first = np.where(np.isnan(axes[:, :1]), unit(rng.standard_normal((n, 3))),
                     axes)
    first = unit(first)
    second = unit(np.cross(first, unit(rng.standard_normal((n, 3)))))
    third = np.cross(first, second)
    rot = np.stack([first, second, third], axis=2)  # columns = local axes
    nuclei = np.zeros(int(np.prod(shape)), dtype=np.float32)
    marks = np.zeros(int(np.prod(shape)), dtype=np.float32)
    reach = np.ceil((semi.max(1) * 1.35 + 1.0)[:, None] / voxel).astype(int)
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
            centre = np.round(points[g] / voxel).astype(int)
            cells = centre[:, None, :] + offsets[None]
            rel = (cells * voxel - points[g][:, None, :]).astype(np.float32)
            local = np.einsum("kmj,kja->kma", rel, rot[g])
            q = np.sqrt(np.sum((local / semi[g][:, None, :]) ** 2, axis=2))
            value = 1 / (1 + np.exp((q - 1) / 0.15))
            ok = np.all((cells >= 0) & (cells < shape), axis=2)
            flat = np.ravel_multi_index(tuple(cells[ok].T), shape)
            np.add.at(nuclei, flat, (value * bright[g][:, None])[ok])
            np.add.at(marks, flat, (value * marker[g][:, None])[ok])
    return nuclei.reshape(shape), marks.reshape(shape)


def membranes(points, codes, labels, tissue, voxel):
    """Cell outlines: voxels near the plane halfway between their two
    nearest nuclei, within a cell's reach of them; plus bright lumen
    surfaces and faint object surfaces."""
    out = np.zeros(labels.shape, dtype=np.float32)
    cell = np.zeros(8, np.float32)
    bright = np.zeros(8, np.float32)
    for code, params in NUCLEI.items():
        cell[code], bright[code] = params["cell"], params["membrane"]
    tree = cKDTree(points)
    where = np.nonzero((tissue != AGAROSE) & (tissue != CAVITY))
    flat = np.ravel_multi_index(where, labels.shape)
    for start in range(0, len(flat), 4_000_000):
        part = tuple(w[start:start + 4_000_000] for w in where)
        p = np.column_stack(part) * voxel
        (d1, d2), (i1, i2) = (a.T for a in tree.query(p, k=2, workers=-1))
        gap = np.linalg.norm(points[i1] - points[i2], axis=1) + 1e-6
        to_plane = (d2 ** 2 - d1 ** 2) / (2 * gap)    # distance to bisector
        code = codes[i1]
        value = np.exp(-(to_plane / 0.9) ** 2) * (d1 <= cell[code]) \
            * bright[code]
        out[part] = value
    cavity = (tissue == CAVITY) & (labels > 0)
    lining = ndi.binary_dilation(cavity) & ~cavity & (labels > 0)
    out[lining] += 1.2
    surface = (labels > 0) & ~ndi.binary_erosion(labels > 0)
    out[surface] += 0.15
    return out


def fibres(tissue, voxel, rng, count=600):
    """Wavy fibres in the loose tissue (random walks that keep their
    direction for a while), drawn as thin lines."""
    shape = np.array(tissue.shape)
    out = np.zeros(tissue.shape, dtype=np.float32)
    tries = rng.integers(0, shape, (count * 30, 3))
    tries = tries[tissue[tuple(tries.T)] == LOOSE][:count]
    count = len(tries)
    start = tries * voxel
    direction = unit(rng.standard_normal((count, 3)))
    alive = np.ones(count, dtype=bool)
    length = rng.uniform(60, 280, count)
    p = start.astype(float)
    for step in range(int(length.max() / 1.5)):
        direction = unit(direction + 0.18 * rng.standard_normal((count, 3)))
        p = p + 1.5 * direction
        cell = np.round(p / voxel).astype(int)
        inside = np.all((cell >= 0) & (cell < shape), axis=1)
        alive &= inside & (step * 1.5 < length)
        ok = np.nonzero(alive)[0]
        ok = ok[tissue[tuple(cell[ok].T)] == LOOSE]
        np.add.at(out, tuple(cell[ok].T), 1.0)
    return out


def debris(shape, voxel, rng, count=260):
    """Bright specks anywhere (dust, dead cells), and how strongly they
    absorb the light sheet (for the shadows behind them)."""
    points = rng.uniform(0, np.array(shape) * voxel, (count, 3))
    radius = rng.uniform(1.2, 3.5, count)
    out = np.zeros(shape, dtype=np.float32)
    for p, r in zip(points, radius):
        lo = np.maximum(np.floor((p - r - 2) / voxel).astype(int), 0)
        hi = np.minimum(np.ceil((p + r + 2) / voxel).astype(int) + 1, shape)
        if np.any(hi <= lo):
            continue
        block = tuple(slice(a, b) for a, b in zip(lo, hi))
        centres = grid_points(lo, hi, voxel).reshape(tuple(hi - lo) + (3,))
        d = np.linalg.norm(centres - p, axis=-1)
        out[block] += 1 / (1 + np.exp((d - r) / 0.6))
    return out


def image_it(nuclei, marks, membrane, fibre, specks, labels, tissue, voxel,
             rng):
    """The three channels as a light-sheet microscope would record them
    (uint16)."""
    background = np.select(
        [tissue == AGAROSE, tissue == LOOSE, tissue == CAVITY],
        [0.0, 0.6, 0.3], 1.0).astype(np.float32)
    # tissue fades out into the agarose over some 10 µm, not at a sharp edge
    background = ndi.gaussian_filter(background, 10.0 / voxel)
    chromatin = ndi.gaussian_filter(rng.standard_normal(nuclei.shape).astype(
        np.float32), 0.7)
    chromatin /= chromatin.std() + 1e-9
    fibre = ndi.gaussian_filter(fibre, np.maximum(0.6 / voxel, 0.3))
    fibre /= max(float(np.percentile(fibre[fibre > 0], 99)), 1e-6) \
        if np.any(fibre > 0) else 1.0
    signals = [
        nuclei * (1 + 0.3 * chromatin) + 0.06 * background + 0.25 * fibre
        + 3.0 * specks,
        membrane + 0.03 * background + 0.7 * fibre + 2.0 * specks,
        marks + 0.012 * background + 1.0 * specks,
    ]
    del chromatin
    # Light: the sheet comes from x = 0 and is absorbed by what it crosses
    # (debris strongly, so it casts shadows); the sheet is uneven across
    # (z, y); detection is dimmer deeper in z.
    absorb = np.select([tissue == AGAROSE, tissue == LOOSE, tissue == CAVITY],
                       [0.0, 0.00035, 0.0001], 0.0007).astype(np.float32)
    absorb += 0.25 * np.clip(specks, 0, 1)
    sheet = np.exp(-np.cumsum(absorb, axis=2) * voxel[2])
    streaks = ndi.gaussian_filter(rng.standard_normal(nuclei.shape[:2]),
                                  (0.6, 1.0))
    stripes = (1 + 0.07 * streaks / (streaks.std() + 1e-9))[:, :, None]
    detect = np.exp(-np.cumsum(absorb * 1.2, axis=0) * voxel[0])
    light = sheet * stripes * detect
    del absorb, sheet, detect
    psf = np.asarray(PSF_SIGMA_UM) / voxel
    out = np.zeros((3,) + nuclei.shape, dtype=np.uint16)
    for c, signal in enumerate(signals):
        signal = ndi.gaussian_filter(signal * light, psf)
        signal += HAZE * ndi.gaussian_filter(signal, HAZE_SIGMA_UM / voxel)
        counts = rng.poisson(np.maximum(signal, 0) * PHOTONS[c]).astype(
            np.float32)
        counts += OFFSET + rng.normal(0, READ_NOISE, counts.shape).astype(
            np.float32)
        out[c] = np.clip(np.round(counts), 0, 65535).astype(np.uint16)
    return out


def describe_layout(labels, voxel, objects):
    """Size of each object, and how each pair meets (contact face or gap)."""
    lines = []
    for k, obj in enumerate(objects, start=1):
        mask = labels == k
        if not mask.any():
            lines.append(f"{k} {obj.name}: MISSING")
            continue
        idx = np.argwhere(mask)
        ext = (idx.max(0) - idx.min(0) + 1) * voxel
        lines.append(f"{k} {obj.name}: {mask.sum() * np.prod(voxel) / 1e6:.2f}"
                     f" x 10^6 µm³, extent z {ext[0]:.0f} y {ext[1]:.0f} "
                     f"x {ext[2]:.0f} µm")
    for a in range(1, len(objects) + 1):
        dist = ndi.distance_transform_edt(labels != a, sampling=voxel)
        for b in range(a + 1, len(objects) + 1):
            mask_b = labels == b
            if not mask_b.any():
                continue
            gap = dist[mask_b].min()
            touching = mask_b & (dist <= voxel.max() * 1.01)
            if gap <= voxel.max() * 1.01:
                lines.append(f"  {a}-{b}: touching ({touching.sum()} voxels "
                             "on the contact face)")
            elif gap < 120:
                lines.append(f"  {a}-{b}: gap {gap:.0f} µm")
    return "\n".join(lines)


def make(voxel, seed=0, layout_only=False, log=print):
    voxel = np.asarray(voxel, dtype=float)
    shape = tuple(int(round(e / v)) for e, v in zip(EXTENT_UM, voxel))
    rng = np.random.default_rng(seed)
    objects, vessels = build_objects()
    t0 = time.perf_counter()
    labels, tissue = truth_maps(objects, vessels, shape, voxel, rng)
    log(f"truth maps {shape}: {time.perf_counter() - t0:.0f} s")
    if layout_only:
        log(describe_layout(labels, voxel, objects))
        return None, labels, tissue, objects
    for k, obj in enumerate(objects, start=1):
        size = np.sum(labels == k) * np.prod(voxel) / 1e6
        log(f"  {k} {obj.name}: {size:.2f} x 10^6 µm³")
    t0 = time.perf_counter()
    points, codes, axes, marker = sample_nuclei(objects, vessels, labels,
                                                tissue, voxel, rng)
    log(f"{len(points)} nuclei: {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    nuclei, marks = splat_nuclei(points, codes, axes, marker, shape, voxel,
                                 rng)
    log(f"nuclei drawn: {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    membrane = membranes(points, codes, labels, tissue, voxel)
    log(f"membranes: {time.perf_counter() - t0:.0f} s")
    fibre = fibres(tissue, voxel, rng)
    specks = debris(shape, voxel, rng)
    t0 = time.perf_counter()
    image = image_it(nuclei, marks, membrane, fibre, specks, labels, tissue,
                     voxel, rng)
    log(f"imaged: {time.perf_counter() - t0:.0f} s")
    return image, labels, tissue, objects


def remove_old(path: Path):
    """Delete an earlier output. Something on Windows marks some of its
    folders read-only (seen 2026-10-30), which stops a plain delete, so
    they are made writable first."""
    def writable_and_retry(function, name, _error):
        os.chmod(name, stat.S_IWRITE)
        function(name)
    if path.exists():
        shutil.rmtree(path, onexc=writable_and_retry)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--voxel", type=float, nargs=3, default=[4.0, 2.0, 2.0],
                        metavar=("Z", "Y", "X"),
                        help="finest voxel size in µm (default 4 2 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", action="store_true",
                        help="only build the true labels and describe how "
                             "the objects meet (quick, for placing them)")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    image, labels, tissue, objects = make(args.voxel, args.seed, args.layout)
    names = [obj.name for obj in objects]
    np.savez_compressed(out / f"{NAME}_truth.npz", labels=labels,
                        tissue=tissue, voxel_um=np.asarray(args.voxel),
                        names=json.dumps(names), tissues=json.dumps(TISSUES),
                        seed=args.seed)
    if args.layout:
        print(f"wrote {out / (NAME + '_truth.npz')} (layout only)")
        return
    import scipy
    import zarr
    record = {"demo_generator": {
        "script": "make_demo_embryos.py (dev)",
        "description": "synthetic demo: six embryo-like objects packed in "
                       "loose tissue; true labels in "
                       f"{NAME}_truth.npz",
        "objects": names, "seed": args.seed, "voxel_um": args.voxel,
        "nuclei": {TISSUES[k]: v for k, v in NUCLEI.items()},
        "photons_per_unit": PHOTONS, "camera_offset": OFFSET,
        "read_noise": READ_NOISE, "psf_sigma_um": PSF_SIGMA_UM,
        "haze": {"sigma_um": HAZE_SIGMA_UM, "fraction": HAZE},
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "zarr": zarr.__version__}}}
    remove_old(out / f"{NAME}.zarr")
    write_ome_zarr(out / f"{NAME}.zarr", image, args.voxel,
                   name="embryo-like objects in tissue (demo)",
                   channels=CHANNELS, extra_attrs=record)
    print(f"wrote {out / (NAME + '.zarr')}: {image.shape} at {args.voxel} µm "
          f"in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
