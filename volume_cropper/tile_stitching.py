"""Stitching the tiles of a tile scan (a mosaic) into one image.

Used for Leica LIF tile scans (file_images.LifReader, 0.17.2; the user's
choice, 2026-11-10: open a tile scan as one stitched image). This module
knows nothing about files: it is given a function that reads part of a
tile, and the tiles' stage positions.

Why not just the stage positions: on the real tile scans tried they were
several pixels off where the tiles' overlaps match (the camera slightly
turned against the stage, the scale a little off, and the stage itself
erring by several pixels). So the tiles are placed where their
overlaps match, as Fiji's Grid/Collection stitching does (Preibisch,
Saalfeld & Tomancak 2009, Bioinformatics 25:1463-1465), but by whole voxels
only, so that every voxel keeps its measured value:

1. Each tile's expected place, from its stage position.
2. For each pair of overlapping tiles, the shift at which the overlap
   matches best: the normalised cross-correlation (NCC: 1 = the same
   picture, whatever the brightness and contrast; about 0 = unrelated)
   over the voxels both tiles share, for every whole-pixel shift near the
   expected one. It is computed for all shifts at once with FFTs (the
   "masked NCC" of Padfield 2012, IEEE Trans. Image Process. 21:2706),
   on maximum projections along z (of up to 8 planes spread through the
   stack); then the best z shift (all planes, the middle of the overlap).
3. A linear model of stage -> image (the camera's rotation and scale),
   fitted to the pairs that matched; a second search near the model's
   places, in a narrower window, finds what the first one missed.
4. All tiles' places at once, by weighted least squares; a pair that
   disagrees with the others by more than MAX_RESIDUAL voxels is dropped.
   Tiles without a matching overlap (e.g. only background) follow the
   model.
5. Where tiles overlap, each voxel comes from the tile whose centre is
   nearest (measured values, no blending: the user's choice).

Coordinates are (z, y, x) in voxels throughout; a tile's place is where
its voxel 0 lies in the stitched image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import fft as sfft
from scipy import sparse
from scipy.sparse import linalg as splinalg

# A match counts when its NCC is at least this (Fiji's Grid/Collection
# stitching keeps 0.30 by default too).
MIN_NCC = 0.3
# A matched pair whose shift disagrees with the tiles' final places by more
# than this many voxels (along any axis) is dropped as a false match.
MAX_RESIDUAL = 3.5
# Weight of a pair placed by the model (no usable match), against a matched
# pair (weight = its NCC, 0.3 to 1): it only matters where nothing else does.
WEAK = 1e-3
# Weight pulling every tile towards the model's place: it only fixes what
# nothing else fixes (where the whole mosaic sits, isolated tiles).
PRIOR = 1e-6
# The model may change the stage's scale or angle by at most this (5 %,
# about 3 degrees); a larger fit is taken to be wrong.
MAX_MODEL = 0.05

# Read part of one tile: read(tile, channel, z, y, x) -> array (z, y, x),
# where z, y, x are slices in the tile's own voxels.
ReadTile = Callable[[int, int, slice, slice, slice], np.ndarray]


@dataclass
class Layout:
    """Where each tile goes in the stitched image."""

    positions: np.ndarray            # (n, 3) int: z, y, x of each voxel 0
    shape: tuple                     # (z, y, x) of the stitched image
    tile_shape: tuple                # (z, y, x) of one tile
    notes: list = field(default_factory=list)    # for the user
    record: dict = field(default_factory=dict)   # for the crop's provenance
    _owners: dict = field(default_factory=dict, repr=False)

    @property
    def centres(self) -> np.ndarray:
        """(n, 2) y, x of each tile's centre in the stitched image."""
        half = (np.array(self.tile_shape[1:]) - 1) / 2
        return self.positions[:, 1:] + half


# ---------------------------------------------------------------------------
# Expected places
# ---------------------------------------------------------------------------

def stage_places(stage_yx, pixel, orientation=(False, False, False)
                 ) -> np.ndarray:
    """(n, 2) expected y, x (pixels) of each tile, the smallest at 0.

    stage_yx: (n, 2) stage positions y, x (in the unit of `pixel`).
    pixel: the pixel size, one number or (y, x) of the image.
    orientation: (swap y and x, flip y, flip x) between the stage and the
    image. With all flags 0 (as in real tile scans), stage y, x is
    image y, x.
    """
    places = np.asarray(stage_yx, dtype=np.float64).copy()
    swap, flip_y, flip_x = orientation
    if swap:
        places = places[:, ::-1].copy()
    if flip_y:
        places[:, 0] *= -1
    if flip_x:
        places[:, 1] *= -1
    places = places / np.broadcast_to(np.asarray(pixel, np.float64), (2,))
    return places - places.min(axis=0)


def stage_layout(stage_yx, pixel, tile_shape,
                 orientation=(False, False, False), note: str = "") -> Layout:
    """The tiles at their stage positions only (rounded to whole voxels):
    the fallback when their overlaps cannot be matched."""
    places = stage_places(stage_yx, pixel, orientation)
    positions = np.zeros((len(places), 3), int)
    positions[:, 1:] = np.rint(places).astype(int)
    notes = ([note] if note else []) + [
        "Where tiles overlap, each voxel is taken from the tile whose "
        "centre is nearest (the measured values; no blending)."]
    return _layout(positions, tile_shape, notes,
                   {"method": "stage positions only",
                    "positions_zyx": positions.tolist()})


def overlapping_pairs(places, tile_yx) -> list[tuple[int, int]]:
    """Pairs of tiles that overlap side by side at their expected places:
    by more than half a tile along one axis (not just at a corner)."""
    places = np.asarray(places, dtype=np.float64)
    h, w = tile_yx
    dy = np.abs(places[:, None, 0] - places[None, :, 0])
    dx = np.abs(places[:, None, 1] - places[None, :, 1])
    over_y, over_x = h - dy, w - dx
    side = (over_y > 0) & (over_x > 0) & ((over_y >= h / 2)
                                          | (over_x >= w / 2))
    a, b = np.nonzero(np.triu(side, k=1))
    return list(zip(a.tolist(), b.tolist()))


# ---------------------------------------------------------------------------
# Matching two tiles
# ---------------------------------------------------------------------------

def ncc_map(fixed: np.ndarray, moving: np.ndarray) -> tuple:
    """NCC of `moving` against `fixed` for every whole-voxel shift.

    Entry j of the result is the shift s = j - (moving.shape - 1): moving's
    voxel 0 on fixed's voxel s. Each NCC is computed over the voxels where
    the two overlap at that shift (Padfield 2012), from six FFT
    correlations. Returns (ncc, count): count = voxels in the overlap;
    ncc is -1 where it is undefined (no overlap, or one side is flat).
    """
    f = np.asarray(fixed, dtype=np.float64)
    m = np.asarray(moving, dtype=np.float64)
    f = f - f.mean()                  # (NCC ignores brightness; this keeps
    m = m - m.mean()                  #  the sums small and exact)
    m = m[(slice(None, None, -1),) * m.ndim]      # correlation = flipped
    full = [a + b - 1 for a, b in zip(f.shape, m.shape)]
    fast = [sfft.next_fast_len(n, real=True) for n in full]
    crop = tuple(slice(0, n) for n in full)

    def spectrum(a):
        return sfft.rfftn(a, fast)

    def back(a):
        return sfft.irfftn(a, fast)[crop]

    f1, f2, fo = spectrum(f), spectrum(f * f), spectrum(np.ones_like(f))
    m1, m2, mo = spectrum(m), spectrum(m * m), spectrum(np.ones_like(m))
    count = np.rint(back(fo * mo))
    n = np.maximum(count, 1)
    sum_f, sum_m = back(f1 * mo), back(fo * m1)
    var_f = back(f2 * mo) - sum_f * sum_f / n
    var_m = back(fo * m2) - sum_m * sum_m / n
    cov = back(f1 * m1) - sum_f * sum_m / n
    # A flat overlap (e.g. empty background) has no NCC; the threshold is
    # far above the FFTs' rounding errors and far below any real signal.
    tiny = 1e-6 * n * max(f.var(), m.var(), 1e-12)
    ok = (count >= 1) & (var_f > tiny) & (var_m > tiny)
    ncc = np.full(count.shape, -1.0)
    ncc[ok] = cov[ok] / np.sqrt(var_f[ok] * var_m[ok])
    return np.clip(ncc, -1.0, 1.0), count


def _strip_boxes(d: int, size: int, margin: int) -> tuple:
    """Along one axis, for tile B at offset d from tile A: the parts of A
    and of B that can overlap for any offset within `margin` of d,
    as ((A from, A to), (B from, B to))."""
    lo, hi = max(0, d), min(size, d + size)        # overlap, in A
    return ((max(0, lo - margin), min(size, hi + margin)),
            (max(0, lo - d - margin), min(size, hi - d + margin)))


@dataclass
class _Strips:
    """The parts of two tiles around their expected overlap, projected
    along z, as read for one pair."""

    a: np.ndarray            # A's part (y, x)
    a0: np.ndarray           # where it starts in A (y, x)
    b: np.ndarray
    b0: np.ndarray
    centre: np.ndarray       # the expected offset they were read around
    margin: int


def projected_planes(nz: int, at_most: int = 8) -> slice:
    """The planes projected for matching in y and x: all, or at most
    `at_most` spread through the stack (reading a strip down the side of a
    tile costs whole planes on disk, so this keeps reading short)."""
    return slice(0, nz, max(1, math.ceil(nz / at_most)))


def read_strips(read: ReadTile, a: int, b: int, channel: int,
                expected, tile_yx, margin: int,
                planes: slice = slice(None)) -> _Strips:
    """Read the parts of tiles a and b that can overlap for any offset
    within `margin` of `expected` (y, x), as maximum projections along z
    of `planes`."""
    d = np.rint(expected).astype(int)
    (ay, by) = _strip_boxes(int(d[0]), tile_yx[0], margin)
    (ax, bx) = _strip_boxes(int(d[1]), tile_yx[1], margin)
    part_a = read(a, channel, planes, slice(*ay), slice(*ax))
    part_b = read(b, channel, planes, slice(*by), slice(*bx))
    return _Strips(np.asarray(part_a).max(axis=0),
                   np.array([ay[0], ax[0]]),
                   np.asarray(part_b).max(axis=0),
                   np.array([by[0], bx[0]]), d, margin)


def match_strips(strips: _Strips, expected, search: int,
                 min_count: int) -> tuple | None:
    """The offset (y, x) of B from A, within `search` of `expected`, at
    which the strips match best: (offset, ncc, at_edge) or None.

    at_edge: the best offset is at the edge of the window searched, so the
    real one may be outside it (such a match is not used).
    """
    if strips.a.size == 0 or strips.b.size == 0:
        return None
    ncc, count = ncc_map(strips.a, strips.b)
    # entry j <-> B's offset j + base (see ncc_map: offset = shift + a0 - b0)
    base = strips.a0 - strips.b0 - (np.array(strips.b.shape) - 1)
    expected = np.rint(expected).astype(int)
    # Shifts are trusted only within the strips' margin of where they were
    # read; within that, within `search` of `expected`.
    low = np.maximum(expected - search, strips.centre - strips.margin)
    high = np.minimum(expected + search, strips.centre + strips.margin)
    if np.any(low > high):
        return None
    j0 = np.clip(low - base, 0, np.array(ncc.shape) - 1)
    j1 = np.clip(high - base, 0, np.array(ncc.shape) - 1)
    window = ncc[j0[0]:j1[0] + 1, j0[1]:j1[1] + 1]
    enough = count[j0[0]:j1[0] + 1, j0[1]:j1[1] + 1] >= min_count
    if window.size == 0 or not enough.any():
        return None
    window = np.where(enough, window, -np.inf)
    j = np.array(np.unravel_index(np.argmax(window), window.shape))
    best = float(window[tuple(j)])
    offset = j + j0 + base
    at_edge = bool(np.any((offset == low) | (offset == high)))
    return offset, best, at_edge


def match_z(read: ReadTile, a: int, b: int, channel: int, offset_yx,
            tile_shape, search: int) -> tuple:
    """The z offset of B from A (planes) at which their overlap (at the
    found y, x offset) matches best: (dz, ncc). Each z offset keeps at
    least half the planes in common. All planes are read, but along the
    seam only the middle `along` pixels of the overlap (enough to match,
    and much less to read)."""
    nz, h, w = tile_shape
    if nz < 2 or search < 1:
        return 0, float("nan")
    dy, dx = (int(v) for v in offset_yx)
    ya, xa = (max(0, dy), min(h, h + dy)), (max(0, dx), min(w, w + dx))
    if ya[1] <= ya[0] or xa[1] <= xa[0]:
        return 0, float("nan")
    along = 256
    if ya[1] - ya[0] > along and ya[1] - ya[0] >= xa[1] - xa[0]:
        middle = (ya[0] + ya[1]) // 2
        ya = (middle - along // 2, middle + along // 2)
    elif xa[1] - xa[0] > along:
        middle = (xa[0] + xa[1]) // 2
        xa = (middle - along // 2, middle + along // 2)
    yb, xb = (ya[0] - dy, ya[1] - dy), (xa[0] - dx, xa[1] - dx)
    va = np.asarray(read(a, channel, slice(None), slice(*ya), slice(*xa)),
                    dtype=np.float64)
    vb = np.asarray(read(b, channel, slice(None), slice(*yb), slice(*xb)),
                    dtype=np.float64)
    best = (0, -np.inf)
    for dz in range(-search, search + 1):
        za = (max(0, dz), min(nz, nz + dz))      # planes of A in common
        if za[1] - za[0] < max(1, nz // 2):
            continue
        pa = va[za[0]:za[1]].ravel()
        pb = vb[za[0] - dz:za[1] - dz].ravel()
        pa, pb = pa - pa.mean(), pb - pb.mean()
        norm = math.sqrt(float(pa @ pa) * float(pb @ pb))
        score = float(pa @ pb) / norm if norm > 0 else -np.inf
        if score > best[1]:
            best = (dz, score)
    return best


# ---------------------------------------------------------------------------
# Placing all tiles
# ---------------------------------------------------------------------------

def fit_model(links, stage_delta) -> tuple[np.ndarray, float]:
    """A 2 x 2 matrix B with offset = (1 + B) * stage offset (y, x),
    fitted to the matched pairs, and the RMS of what it leaves (pixels).
    B = 0 (the stage as it is) when there are too few pairs or the fit is
    implausible (more than MAX_MODEL)."""
    if len(links) < 2:
        return np.zeros((2, 2)), float("nan")
    s = np.array([stage_delta[(a, b)] for a, b, *_ in links])
    d = np.array([off[1:] for _, _, off, _ in links], dtype=np.float64)
    keep = np.ones(len(links), bool)
    model = np.zeros((2, 2))
    for _ in range(2):              # fit, drop gross outliers, fit again
        fit, *_ = np.linalg.lstsq(s[keep], (d - s)[keep], rcond=None)
        model = fit.T
        left = np.abs(d - s - s @ model.T).max(axis=1)
        limit = 3 * np.median(left[keep]) + 2
        if keep.sum() - (left > limit).sum() < 2:
            break
        keep = left <= limit
    if np.abs(model).max() > MAX_MODEL:
        return np.zeros((2, 2)), float("nan")
    left = d - s - s @ model.T
    return model, float(np.sqrt(np.mean(left[keep] ** 2)))


def solve_places(n: int, strong: list, weak: list, prior) -> np.ndarray:
    """Places (n, 3) that best satisfy every link: offset of b from a =
    the link's offset, weighted (weighted least squares), plus a faint
    pull of every tile towards `prior` (n, 3) so that the solution is
    unique. strong/weak: lists of (a, b, offset (3,), weight)."""
    rows, cols, vals, rhs, weights = [], [], [], [], []
    for r, (a, b, off, wgt) in enumerate(strong + weak):
        rows += [r, r]
        cols += [b, a]
        vals += [1.0, -1.0]
        rhs.append(off)
        weights.append(wgt)
    m = len(rhs)
    rows += list(range(m, m + n))
    cols += list(range(n))
    vals += [1.0] * n
    rhs = np.array(list(rhs) + list(prior), dtype=np.float64).reshape(-1, 3)
    weights = np.array(weights + [PRIOR] * n)
    a = sparse.csr_matrix((vals, (rows, cols)), shape=(m + n, n))
    w = sparse.diags(weights)
    normal = (a.T @ w @ a).tocsc()                 # n x n, positive definite
    right = a.T @ (weights[:, None] * rhs)
    places = np.column_stack([splinalg.spsolve(normal, right[:, k])
                              for k in range(3)])
    return places.reshape(n, 3)


def register(read: ReadTile, tile_shape, stage_yx, pixel: float,
             n_channels: int = 1, stage_z_planes=None,
             orientation=None) -> Layout:
    """Find where each tile goes. See the module's docstring for the method.

    tile_shape: (z, y, x) of one tile. stage_yx: (n, 2) stage positions
    (y, x) in the unit of `pixel` (the pixel size). stage_z_planes: (n,)
    stage z in planes, or None (used only to widen the z search).
    orientation: (swap, flip y, flip x) between stage and image; None:
    found from the overlaps (all 8 are tried; for files whose flags are
    not those tested).
    """
    nz, h, w = (int(v) for v in tile_shape)
    n = len(stage_yx)
    notes = []
    search1 = max(16, math.ceil(0.05 * max(h, w)))

    def min_count(delta) -> int:
        """Half the expected overlap, in pixels (at least 32)."""
        oy, ox = h - abs(delta[0]), w - abs(delta[1])
        return int(max(32, 0.5 * max(oy, 0) * max(ox, 0)))

    if orientation is None:
        orientation = _find_orientation(read, tile_shape, stage_yx, pixel,
                                        n_channels, search1, min_count)
        notes.append("The orientation of the stage against the image was "
                     "found from the overlaps (not tested on real files "
                     "yet): " + _orientation_text(orientation) + ".")
    places = stage_places(stage_yx, pixel, orientation)
    pairs = overlapping_pairs(places, (h, w))
    stage_delta = {(a, b): places[b] - places[a] for a, b in pairs}
    record = {"method": "overlap registration in whole voxels (NCC), then "
                        "least squares; each voxel from the tile whose "
                        "centre is nearest",
              "tiles": n, "tile_shape_zyx": [nz, h, w],
              "orientation_swap_flipy_flipx": [bool(v) for v in orientation],
              "stage_places_yx_pixels": np.round(places, 2).tolist(),
              "min_ncc": MIN_NCC, "max_residual_voxels": MAX_RESIDUAL}
    if not pairs:
        positions = np.zeros((n, 3), int)
        positions[:, 1:] = np.rint(places).astype(int)
        notes.append("Its tiles do not overlap: they were placed by their "
                     "stage positions.")
        record["links"] = []
        return _layout(positions, tile_shape, notes, record)

    # 1. The channel to match on: the one whose overlaps match best.
    channel = _pick_channel(read, pairs, stage_delta, (nz, h, w), n_channels,
                            search1, min_count)
    # 2. First search, around the stage positions.
    strips, found_pairs = {}, {}
    planes = projected_planes(nz)
    for a, b in pairs:
        strips[(a, b)] = read_strips(read, a, b, channel, stage_delta[(a, b)],
                                     (h, w), search1, planes)
        found = match_strips(strips[(a, b)], stage_delta[(a, b)], search1,
                             min_count(stage_delta[(a, b)]))
        if found and found[1] >= MIN_NCC and not found[2]:
            found_pairs[(a, b)] = found
    first = [(a, b, np.r_[0, off], score)
             for (a, b), (off, score, _) in found_pairs.items()]
    # 3. The model of stage -> image, and a narrower second search for the
    # pairs the first one missed (a good first match is kept: the stage
    # can err by more than the model's typical error, e.g. 17 pixels).
    model, rms = fit_model(first, stage_delta)
    search2 = search1 if math.isnan(rms) else \
        int(np.clip(math.ceil(3 * rms + 3), 6, search1))
    expected = {k: v + model @ v for k, v in stage_delta.items()}
    for a, b in pairs:
        if (a, b) in found_pairs:
            continue
        e = expected[(a, b)]
        found = match_strips(strips[(a, b)], e, search2, min_count(e))
        if found and found[1] >= MIN_NCC and not found[2]:
            found_pairs[(a, b)] = found
    strips.clear()
    # z: where the tiles have planes, the z offset of each matched pair.
    z_search = 0
    if nz > 1:
        spread = 0.0
        if stage_z_planes is not None:
            z = np.asarray(stage_z_planes, dtype=np.float64)
            spread = float(np.nanmax(z) - np.nanmin(z)) if z.size else 0.0
        z_search = int(min(nz // 2, max(2, math.ceil(spread) + 2)))
    strong = []
    for (a, b), (off, score, _) in found_pairs.items():
        dz = match_z(read, a, b, channel, off, tile_shape, z_search)[0] \
            if z_search else 0
        strong.append((a, b, np.r_[dz, off].astype(np.float64), score))
    # 4. All places at once; drop matches that disagree with the rest.
    prior = np.zeros((n, 3))
    prior[:, 1:] = places + places @ model.T
    dropped = []
    while True:
        matched = {(a, b) for a, b, *_ in strong}
        weak = [(a, b, np.r_[0.0, expected[(a, b)]], WEAK)
                for a, b in pairs if (a, b) not in matched]
        solved = solve_places(n, strong, weak, prior)
        if not strong:
            break
        left = np.array([np.abs(solved[b] - solved[a] - off).max()
                         for a, b, off, _ in strong])
        worst = int(np.argmax(left))
        if left[worst] <= MAX_RESIDUAL:
            break
        dropped.append(strong.pop(worst)[:2])
    # Relative to the best-linked tile before rounding: the solution may sit
    # a common fraction of a voxel off whole numbers (from the faint prior),
    # and rounding each tile on its own would then split neighbours.
    links_of = np.bincount([t for a, b, *_ in strong for t in (a, b)],
                           minlength=n)
    solved = solved - solved[int(np.argmax(links_of))]
    positions = np.rint(solved).astype(int)
    positions -= positions.min(axis=0)

    # 5. What was done, for the user and the crop's record.
    linked = {t for a, b, *_ in strong for t in (a, b)}
    unlinked = sorted(set(range(n)) - linked)
    residual = (np.array([np.abs(positions[b] - positions[a] - off).max()
                          for a, b, off, _ in strong]) if strong
                else np.array([]))
    scores = np.array([s for *_, s in strong])
    text = (f"Stitched from {n} tiles where their overlaps match "
            f"(channel {channel + 1}): {len(strong)} of {len(pairs)} "
            "overlaps matched")
    if strong:
        low, high = f"{scores.min():.2f}", f"{scores.max():.2f}"
        text += (f" (similarity {low if low == high else low + '-' + high}"
                 ", where 1 = identical)")
        worst = int(residual.max())
        text += (", and every match is met exactly" if worst == 0 else
                 f", and the tiles' places agree with every match to "
                 f"within {worst} voxel{'' if worst == 1 else 's'}")
    notes.append(text + ".")
    if unlinked:
        one = len(unlinked) == 1
        notes.append(f"{len(unlinked)} tile{'' if one else 's'} had no "
                     "matching overlap (e.g. only background) and "
                     f"{'was' if one else 'were'} placed by the stage "
                     "positions, corrected for the camera's angle and "
                     f"scale: {'it' if one else 'they'} may be a few voxels "
                     "off.")
    if dropped:
        one = len(dropped) == 1
        notes.append(f"{len(dropped)} match{'' if one else 'es'} disagreed "
                     f"with the others and {'was' if one else 'were'} not "
                     "used.")
    z_moved = sorted({int(p) for p in positions[:, 0]})
    if len(z_moved) > 1:
        notes.append("Some tiles were moved in z (by up to "
                     f"{max(z_moved) - min(z_moved)} planes) to match their "
                     "neighbours; planes at the top and bottom are partly "
                     "empty (0) there.")
    notes.append("Where tiles overlap, each voxel is taken from the tile "
                 "whose centre is nearest (the measured values; no "
                 "blending).")
    angle = math.degrees(math.atan2(model[1, 0] - model[0, 1],
                                    2 + model[0, 0] + model[1, 1]))
    record.update({
        "channel_matched": channel,
        "model_stage_to_image": (np.eye(2) + model).round(5).tolist(),
        "model_angle_degrees": round(angle, 3),
        "search_pixels": [search1, search2], "z_search_planes": z_search,
        "links": [{"tiles": [a, b], "offset_zyx": [int(v) for v in off],
                   "ncc": round(float(s), 4)} for a, b, off, s in strong],
        "dropped_links": [list(p) for p in dropped],
        "tiles_without_match": unlinked,
        "positions_zyx": positions.tolist()})
    return _layout(positions, tile_shape, notes, record)


def _layout(positions, tile_shape, notes, record) -> Layout:
    shape = tuple(int(v) for v in (positions.max(axis=0)
                                   + np.array(tile_shape)))
    record["shape_zyx"] = list(shape)
    return Layout(positions=positions, shape=shape,
                  tile_shape=tuple(int(v) for v in tile_shape),
                  notes=notes, record=record)


def _probe_pairs(pairs, at_most: int = 8) -> list:
    """Up to `at_most` pairs spread over the mosaic."""
    if len(pairs) <= at_most:
        return list(pairs)
    picks = np.linspace(0, len(pairs) - 1, at_most).round().astype(int)
    return [pairs[i] for i in picks]


def _pick_channel(read, pairs, stage_delta, tile_shape, n_channels, search,
                  min_count) -> int:
    """The channel whose overlaps match best, tried on a few pairs and
    three middle planes. Pairs one above the other come first: their
    overlap is whole rows of the tile, which is quick to read."""
    if n_channels <= 1:
        return 0
    nz, h, w = tile_shape
    middle = slice(max(0, nz // 2 - 1), min(nz, nz // 2 + 2))
    stacked = sorted(pairs, key=lambda p: abs(stage_delta[p][1]) * h
                     - abs(stage_delta[p][0]) * w)
    count = 3 if h * w >= 512 * 512 else 8        # (small tiles read fast)
    first = stacked[:max(count, len(stacked) // 2)]
    probes = [first[i] for i in sorted(set(
        np.linspace(0, len(first) - 1, min(count, len(first))).round()
        .astype(int).tolist()))]
    best, best_score = 0, -1.0
    for c in range(n_channels):
        score = 0.0
        for a, b in probes:
            e = stage_delta[(a, b)]
            strips = read_strips(read, a, b, c, e, (h, w), search, middle)
            found = match_strips(strips, e, search, min_count(e))
            if found and found[1] >= MIN_NCC and not found[2]:
                score += found[1]
        if score > best_score:
            best, best_score = c, score
    return best


def _find_orientation(read, tile_shape, stage_yx, pixel, n_channels,
                      search, min_count) -> tuple:
    """The orientation (swap, flip y, flip x) under which the overlaps of a
    few pairs match best (all channels tried)."""
    _, h, w = tile_shape
    best, best_score = (False, False, False), -1.0
    for swap in (False, True):
        for flip_y in (False, True):
            for flip_x in (False, True):
                o = (swap, flip_y, flip_x)
                places = stage_places(stage_yx, pixel, o)
                pairs = overlapping_pairs(places, (h, w))
                delta = {(a, b): places[b] - places[a] for a, b in pairs}
                score = 0.0
                for a, b in _probe_pairs(pairs, 6):
                    for c in range(n_channels):
                        strips = read_strips(read, a, b, c, delta[(a, b)],
                                             (h, w), search)
                        found = match_strips(strips, delta[(a, b)], search,
                                             min_count(delta[(a, b)]))
                        if found and found[1] >= MIN_NCC and not found[2]:
                            score += found[1]
                if score > best_score + 1e-9:
                    best, best_score = o, score
    return best


def _orientation_text(orientation) -> str:
    swap, flip_y, flip_x = orientation
    parts = (["stage x is image y and stage y image x"] if swap else []) \
        + (["y reversed"] if flip_y else []) \
        + (["x reversed"] if flip_x else [])
    return ", ".join(parts) if parts else "stage x, y = image x, y"


# ---------------------------------------------------------------------------
# Reading the stitched image
# ---------------------------------------------------------------------------

def owners(layout: Layout, cover: tuple, ys: slice, xs: slice) -> np.ndarray:
    """For each pixel of the window (ys, xs): the tile (of those in
    `cover`) whose centre is nearest among those covering it; -1: none.
    Kept for the next call (a read goes plane by plane, channel by
    channel, over the same window)."""
    key = (cover, ys.start, ys.stop, xs.start, xs.stop)
    if key in layout._owners:
        return layout._owners[key]
    _, h, w = layout.tile_shape
    shape = (ys.stop - ys.start, xs.stop - xs.start)
    owner = np.full(shape, -1, dtype=np.int32)
    nearest = np.full(shape, np.inf)
    centres = layout.centres
    for k in cover:
        py, px = layout.positions[k, 1:]
        y0, y1 = max(ys.start, py), min(ys.stop, py + h)
        x0, x1 = max(xs.start, px), min(xs.stop, px + w)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None] - centres[k, 0]
        xx = np.arange(x0, x1)[None, :] - centres[k, 1]
        d2 = yy * yy + xx * xx
        box = (slice(y0 - ys.start, y1 - ys.start),
               slice(x0 - xs.start, x1 - xs.start))
        closer = d2 < nearest[box]                # ties: the lower tile
        nearest[box][closer] = d2[closer]
        owner[box][closer] = k
    if len(layout._owners) > 8:
        layout._owners.clear()
    layout._owners[key] = owner
    return owner


def assemble(read_tile: Callable[[int, slice, slice, slice], np.ndarray],
             layout: Layout, zs: slice, ys: slice, xs: slice,
             dtype) -> np.ndarray:
    """The stitched image's voxels (zs, ys, xs) (slices with start and stop
    in the stitched image, step 1): each from the tile whose centre is
    nearest among those that have it; 0 where no tile does.

    read_tile(tile, z, y, x) -> array (z, y, x) of one tile's voxels (one
    channel and timepoint).
    """
    nz, h, w = layout.tile_shape
    out = np.zeros((zs.stop - zs.start, ys.stop - ys.start,
                    xs.stop - xs.start), dtype=dtype)
    p = layout.positions
    near = [k for k in range(len(p))
            if p[k, 1] < ys.stop and p[k, 1] + h > ys.start
            and p[k, 2] < xs.stop and p[k, 2] + w > xs.start]
    # Planes covered by the same tiles are done together (all of them,
    # unless some tiles were moved in z).
    groups = []
    for z in range(zs.start, zs.stop):
        cover = tuple(k for k in near if p[k, 0] <= z < p[k, 0] + nz)
        if groups and groups[-1][2] == cover and groups[-1][1] == z:
            groups[-1][1] = z + 1
        else:
            groups.append([z, z + 1, cover])
    for z0, z1, cover in groups:
        if not cover:
            continue
        owner = owners(layout, cover, ys, xs)
        for k in cover:
            mine = owner == k
            if not mine.any():
                continue
            rows = np.flatnonzero(mine.any(axis=1))
            cols = np.flatnonzero(mine.any(axis=0))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            pz, py, px = p[k]
            block = np.asarray(read_tile(
                k, slice(z0 - pz, z1 - pz),
                slice(ys.start + r0 - py, ys.start + r1 - py),
                slice(xs.start + c0 - px, xs.start + c1 - px)))
            target = out[z0 - zs.start:z1 - zs.start, r0:r1, c0:c1]
            here = mine[r0:r1, c0:c1]
            target[:, here] = block[:, here]
    return out
