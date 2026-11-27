"""Finding a bright object and the crop region around it (0.14.0,
experimental).

For an image with one bright object, e.g. a single embryo in a lightsheet
time-lapse, this finds the box around the object, in 3D and over time, so
that the crop region can be made as small as the object allows.

How (for each timepoint searched, on one channel):

1. Read the volume at a coarse pyramid level (the finest level with at most
   MAX_VOXELS voxels per timepoint, e.g. 256 x 256 x 256, and whose
   timepoints are level 0's), from the search area: the whole image, or a
   crop region (then the region is shrunk to the object inside it).
2. Smooth it a little (Gaussian, `smooth` voxels), so that noise does not
   break the object into pieces or add specks.
3. Threshold it automatically, as Fiji's Image > Adjust > Auto Threshold
   does: Otsu (default), Triangle (good when the object is small and the
   background large) or Li. Each timepoint gets its own threshold, so
   bleaching or changes in brightness do not matter.
4. Of the connected pieces above that threshold (like Fiji's Analyze
   Particles in 3D), take the one with the most signal (its values above
   the threshold, added up): small bright specks and large dim areas lose.
   It counts only if it stands at least MIN_CONTRAST noise levels above the
   background (an automatic threshold splits even pure noise in two).
5. Grow it down to just above the background (a hysteresis threshold: the
   background's median + MIN_CONTRAST noise levels), so that the object's
   dimmer parts belong to it too: the far side of an embryo when the light
   fades with depth, or tissue that joins two bright parts. The automatic
   threshold only finds the object; its extent comes from this.
6. Its bounding box, padded by one voxel of the level searched (voxels at
   the object's edge are only partly object), on level 0's grid, in µm.

With a seed (0.16.0; a click on the object), step 4 takes the piece under
the seed instead, at a threshold of its own (the level at which that piece
is most stable, see _seed_threshold), and the other timepoints follow the
object outwards from the seed's timepoint. With keep_mask, the object's
mask is kept, for its outline as seen in each view (`silhouettes`).

The boxes of all timepoints are combined (the smallest box that holds the
object at every timepoint), the t range is cut to the timepoints where the
object was found, and a margin (in µm) is added on every side, within the
image. The result is an ordinary crop region: the window shows it, and it
can be adjusted before cropping.

Limits: with growing (step 5, `grow`, the default) anything above the
background that touches the object becomes part of it: tissue around an
embryo (measured on the embryo demo: the box then spreads through the
tissue). For such images turn growing off (the automatic threshold then
gives the extent, which can miss an object's dim parts) and search inside
a region drawn around the object. A bright slab of background (e.g. the
brighter side of uneven illumination) can have more signal than a dim
object and be taken for it.

No window is needed (numpy, scipy and scikit-image only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from volume_cropper import core as vc

# The finest level used has at most this many voxels per timepoint (in the
# search area): 256 x 256 x 256 = 16.8 M, read as float32 = 67 MB.
MAX_VOXELS = 20_000_000
METHODS = ("otsu", "triangle", "li")
METHOD_NAMES = {"otsu": "Otsu", "triangle": "Triangle", "li": "Li"}
# Pieces smaller than this many voxels (at the level used) are never "the
# object".
MIN_VOXELS = 8
# An automatic threshold always splits a volume in two, even pure noise. The
# object counts only if its mean stands at least this many noise levels
# (robust standard deviations of the voxels below the threshold) above the
# background's median; otherwise "no object" at that timepoint. The object
# is then grown down to the same level above the background (step 5).
MIN_CONTRAST = 3.0


class SearchCancelled(Exception):
    """The search was cancelled (from the window)."""


@dataclass
class ObjectBox:
    """What find_object found.

    ranges:  the crop region: {axis: (start, stop)}, spatial axes in their
             unit (outer voxel edges, padding and margin added, within the
             image), the time axis as the times of the first and last
             timepoint where the object was found (frame centres, as crop
             regions keep them). Empty if the object was found at no
             timepoint.
    level:   the pyramid level searched.
    times:   the timepoint indices searched (one entry, None, without t).
    found:   per timepoint searched: the object's box before the margin
             ({axis: (start, stop)}), or None where none was found.
    thresholds, voxels: per timepoint searched (None where not found).
    contrasts: per timepoint, the strongest candidate's mean above the
             background in noise levels (found if >= MIN_CONTRAST).
    """

    ranges: dict = field(default_factory=dict)
    level: int = 0
    method: str = "otsu"
    margin: float = 0.0
    times: list = field(default_factory=list)
    found: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)
    voxels: list = field(default_factory=list)
    contrasts: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    # spatial axes along which the object reaches the edge of the area
    # searched, inside the image (it may be cut there), or the image's own
    # edge (0.16.0)
    touches: set = field(default_factory=set)
    image_edges: set = field(default_factory=set)
    # (0.16.0, keep_mask) the object's mask over all timepoints at the level
    # searched, and the search area there {axis: (first, end)} it covers
    mask: object = None
    search: dict = field(default_factory=dict)
    # (0.17.0, timed_silhouettes) per timepoint index where the object was
    # found: its own silhouettes, {depth axis: (plane, points)}
    timed_silhouettes: dict = field(default_factory=dict)
    # (0.18.0) the user's own threshold, if one was given (find_object's
    # threshold); seed_mode: how a seed's threshold was chosen
    own_threshold: float | None = None
    seed_mode: str = "stable"

    @property
    def n_found(self) -> int:
        return sum(box is not None for box in self.found)


def spatial_axes(image: vc.OmeZarrImage) -> list[str]:
    return [a["name"] for a in image.axes if a.get("type") == "space"]


def usable_levels(image: vc.OmeZarrImage) -> list[int]:
    """The levels that line up with level 0 by whole factors (as the crop
    code needs) and keep every timepoint (levels that also shrink time
    would make the t range wrong)."""
    t = vc.time_axis(image)
    return [k for k, f in enumerate(vc.downsampling_factors(image))
            if f is not None and (t is None or f[t] == 1)]


def search_indices(image: vc.OmeZarrImage, level: int,
                   region: dict | None) -> dict:
    """{axis: (first, end)} voxel indices at `level` of the search area
    (the whole image, or the region's ranges), for every axis but c. The
    region is taken on level 0's grid first, then to the level (level
    voxel j holds level-0 voxels f*j ... f*j + f - 1, as for crops)."""
    level0, lev = image.levels[0], image.levels[level]
    factors = vc.downsampling_factors(image)[level]
    out = {}
    for i, name in enumerate(image.axis_names):
        if i == image.channel_axis:
            continue
        n0, n = level0.array.shape[i], lev.array.shape[i]
        if region and name in region:
            low, high = sorted(region[name])
            first0, end0 = vc.physical_to_index_range(
                low, high, level0.scale[i], level0.translation[i])
            first0, end0 = max(0, first0), min(n0, end0)
            f = factors[i]
            first, end = first0 // f, min(n, -(-end0 // f))
        else:
            first, end = 0, n
        out[name] = (first, end)
    return out


def choose_level(image: vc.OmeZarrImage, region: dict | None = None,
                 max_voxels: int = MAX_VOXELS) -> int:
    """The finest usable level whose search area has at most max_voxels
    voxels per timepoint (the coarsest usable level if none has)."""
    space = spatial_axes(image)
    levels = usable_levels(image) or [0]
    for k in levels:
        idx = search_indices(image, k, region)
        voxels = int(np.prod([idx[a][1] - idx[a][0] for a in space],
                             dtype=np.int64))
        if voxels <= max_voxels:
            return k
    return levels[-1]


def threshold_of(values: np.ndarray, method: str) -> float:
    from skimage import filters
    function = {"otsu": filters.threshold_otsu,
                "triangle": filters.threshold_triangle,
                "li": filters.threshold_li}[method]
    return float(function(values))


def _robust(values: np.ndarray) -> tuple[float, float]:
    """(median, robust standard deviation = 1.4826 x the median absolute
    deviation) of background values, from a sample of at most a million.
    Exact zeros are left out when they are most of it (the padding around
    fused volumes), so that they do not hide the real background's noise."""
    if values.size > 1_000_000:
        values = values[:: values.size // 1_000_000]
    zeros = values == 0
    if zeros.mean() > 0.5 and (~zeros).sum() >= 10:
        values = values[~zeros]
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if sigma <= 0:
        sigma = float(values.std())
    return median, sigma


def object_in(volume: np.ndarray, method: str = "otsu",
              smooth: float = 1.0, grow: bool = True, seed=None,
              follow: np.ndarray | None = None,
              return_mask: bool = False,
              relative: float | None = None,
              step: Callable[[float], None] | None = None,
              fixed: float | None = None,
              seed_mode: str = "stable") -> tuple:
    """(slices, threshold, voxels, contrast) of the object in one volume
    (z, y, x or y, x): the piece above the automatic threshold with the
    most signal, grown down to just above the background if `grow` (see
    the module's description); slices is None if there is none. contrast:
    the piece's mean above the background, in noise levels (see
    MIN_CONTRAST).

    Which piece (0.16.0): with `seed` (a voxel index), the piece under it
    (or the nearest one, at most SEED_REACH voxels away), at the threshold
    at which that piece is most stable (see _seed_threshold); with `follow`
    (a mask of the volume, e.g. the object at the timepoint before), the
    piece that overlaps it most (or the nearest one), at the threshold
    `relative` gives (see _relative_level; the automatic one without it).
    return_mask: also return the object's (grown) mask and its threshold's
    relative level, as a fifth and sixth value.
    step(fraction) (0.18.0): told how far the search in this volume is
    (with a seed: after each threshold tried, the slow part), e.g. for a
    progress bar; it may raise to stop the search."""
    found = _object_in(volume, method, smooth, grow, seed, follow, relative,
                       step, fixed, seed_mode)
    return found if return_mask else found[:4]


# A seed may lie this many voxels (of the level searched) beside its
# object, e.g. a click just outside a dim edge (0.16.0).
SEED_REACH = 6
# With a seed, the threshold is chosen among this many levels, from the
# automatic threshold (or the object's brightness near the seed, if lower)
# down to just above the background (0.16.0); a piece larger than this
# share of the volume searched is never chosen (it has joined most of what
# is bright, which is stable too: MSER's "maximum area").
SEED_LEVELS = 12
SEED_MAX_SHARE = 0.25


def _seed_voxel(seed, shape) -> tuple | None:
    seed = tuple(int(s) for s in seed)
    return seed if all(0 <= s < n for s, n in zip(seed, shape)) else None


def _background_level(data, valid, threshold) -> tuple[float, float]:
    """(median, robust sd) of the smoothed voxels below the automatic
    threshold: the background, for thresholds relative to it."""
    return _robust(data[valid & (data <= threshold)])


def _relative_level(data, valid, threshold, inside, level) -> float:
    """A threshold's place between the background's median (0) and the
    90th percentile of the object's smoothed values (1): carried from the
    seed's timepoint to the ones followed (`relative`), where the object
    may be brighter or dimmer (0.16.0)."""
    median, _ = _background_level(data, valid, threshold)
    top = float(np.percentile(data[inside & valid], 90))
    return (level - median) / (top - median) if top > median else 1.0


SEED_MODES = ("stable", "brightest", "whole")
SEED_MODE_NAMES = {"stable": "most stable part",
                   "brightest": "brightest stable part",
                   "whole": "whole object"}


def _seed_threshold(data, valid, threshold, seed, floor,
                    step: Callable[[float], None] | None = None,
                    mode: str = "stable") -> float:
    """With a seed, the threshold for its object (0.16.0): of SEED_LEVELS
    levels from the automatic threshold (or, if lower, the brightness near
    the seed: the 90th percentile of the smoothed values within
    SEED_REACH voxels) down to `floor` (just above the background), the
    one at which the seed's piece is most stable: its size changes least,
    relative to it, from the level above to the level below. This is the
    idea of MSER (maximally stable extremal regions; Matas et al. 2002).

    Why: the automatic threshold of the whole volume can lie above a dim
    object, which then falls apart into specks (found on the embryo demo:
    a dim embryo's single nuclei), while halfway down to the background
    the object joined the tissue around it and then the other embryos. Its
    piece grows fast while specks join, hardly changes while it is the
    object, and jumps again when it joins its surroundings (measured on
    the demo: the bright embryos grew 2-9 % per step until they joined the
    tissue, then 50-500 % in one step). Pieces larger than SEED_MAX_SHARE
    of the volume are left out. An object that joins its surroundings
    before it is whole (dim embryos in the demo's tissue) cannot be
    separated by any threshold; then the most stable piece is part of it,
    or it with its surroundings.

    mode (0.18.0, the user's wish: brighter parts of an embryo apart from
    fainter ones): "stable" is the above; "brightest" and "whole" take,
    of the stable plateaus (levels where the change is a local minimum),
    the brightest one (the first from the top: e.g. a bright core) or the
    dimmest one (the last before the piece grows too large: the whole
    object). For these the levels start at the brightness near the seed
    even if it is above the automatic threshold, where a bright core
    lies."""
    from scipy import ndimage as ndi
    seed = _seed_voxel(seed, data.shape)
    if seed is None:
        return threshold
    near = tuple(slice(max(s - SEED_REACH, 0), s + SEED_REACH + 1)
                 for s in seed)
    values = data[near][valid[near]]
    if values.size == 0:
        return threshold
    near = float(np.percentile(values, 90))
    top = min(threshold, near) if mode == "stable" else near
    if not top > floor:
        return threshold
    levels = np.linspace(top, floor, SEED_LEVELS)
    largest = SEED_MAX_SHARE * int(valid.sum())
    sizes = []
    for i, level in enumerate(levels):
        if step is not None:                  # (0.18.0) for a progress bar
            step(i / len(levels))
        if sizes and sizes[-1] > largest:     # lower levels: larger still
            sizes.append(sizes[-1])
            continue
        mask = (data > level) & valid
        labels, n = ndi.label(mask)
        if n == 0:
            sizes.append(0)
            continue
        pieces = ndi.sum_labels(mask, labels, np.arange(1, n + 1))
        best = _chosen_piece(labels, n, pieces, seed, None)
        sizes.append(0 if best is None else int(pieces[best]))
    sizes = np.asarray(sizes, dtype=float)
    change = np.full(len(levels), np.inf)
    for i in range(len(levels)):
        if sizes[i] < MIN_VOXELS or sizes[i] > largest:
            continue
        above = sizes[i - 1] if i > 0 else sizes[i]
        below = sizes[i + 1] if i + 1 < len(levels) else sizes[i]
        if i == 0 or i + 1 == len(levels):      # one-sided at the ends
            change[i] = 2 * (below - above) / sizes[i]
        else:
            change[i] = (below - above) / sizes[i]
    finite = np.isfinite(change)
    if not finite.any():
        return threshold
    if mode == "stable":
        return float(levels[int(np.argmin(change))])
    # the stable plateaus: local minima of the change (bright to dim)
    plateaus = [i for i in range(len(levels)) if finite[i]
                and (i == 0 or change[i] <= change[i - 1])
                and (i + 1 == len(levels) or change[i] <= change[i + 1])]
    if not plateaus:
        return float(levels[int(np.argmin(change))])
    return float(levels[plateaus[0] if mode == "brightest"
                        else plateaus[-1]])


def _chosen_piece(labels, n, sizes, seed, follow) -> int | None:
    """The 0-based index of the piece a seed or a followed mask points to,
    or None."""
    from scipy import ndimage as ndi
    big = sizes >= MIN_VOXELS
    if seed is not None:
        seed = _seed_voxel(seed, labels.shape)
        if seed is None:
            return None
        label = int(labels[seed])
        if label == 0 or not big[label - 1]:
            # beside it, or on a speck: the nearest piece of some size,
            # looked for within SEED_REACH of the seed only
            near = tuple(slice(max(s - SEED_REACH, 0), s + SEED_REACH + 1)
                         for s in seed)
            local = labels[near]
            sized = np.zeros(n + 1, bool)
            sized[1:] = big
            if not sized[local].any():
                return None
            at = tuple(s - w.start for s, w in zip(seed, near))
            distance, nearest = ndi.distance_transform_edt(
                ~sized[local], return_indices=True)
            if distance[at] > SEED_REACH:
                return None
            label = int(local[tuple(i[at] for i in nearest)])
        return label - 1 if label > 0 and big[label - 1] else None
    overlap = ndi.sum_labels(follow, labels, np.arange(1, n + 1))
    overlap[~big] = 0
    if overlap.max() > 0:
        return int(np.argmax(overlap))
    # none overlaps (it moved further than its size): the nearest piece to
    # where it was, within the size of what is followed
    if not follow.any():
        return None
    centre = np.array(ndi.center_of_mass(follow))
    extent = max(np.ptp(np.nonzero(follow), axis=1)) + 1
    centres = np.array(ndi.center_of_mass(labels > 0, labels,
                                          np.arange(1, n + 1)))
    distances = np.linalg.norm(centres - centre, axis=1)
    distances[~big] = np.inf
    best = int(np.argmin(distances))
    return best if distances[best] <= extent else None


def _prepared(volume, smooth: float) -> tuple:
    """(raw, smoothed, valid) as the search sees a volume: float32, a
    Gaussian of `smooth` voxels, and where there is data. Exact zeros in
    quantity are no data (the padding around fused volumes, or outside the
    views of a multi-view recording): left out of the threshold, the
    background and the growing, so that the recorded area itself is not
    taken for the object."""
    from scipy import ndimage as ndi
    raw = volume.astype(np.float32, copy=False)
    data = ndi.gaussian_filter(raw, sigma=smooth) if smooth > 0 else raw
    zeros = raw == 0
    valid = (~zeros if zeros.mean() > 0.05 and (~zeros).sum() >= 100
             else np.ones(raw.shape, bool))
    return raw, data, valid


@dataclass
class SearchVolume:
    """(0.18.0) One timepoint of a channel as the search sees it (the
    search level, smoothed, where there is data): for the user's own
    threshold its starting value (automatic), its range and its live
    outline in the 2D views (contours). axes: the spatial axes (image
    order); first: the centre of the first voxel, scale: the voxel size,
    per axis, in µm."""

    data: np.ndarray
    valid: np.ndarray
    level: int
    axes: list
    first: dict
    scale: dict

    def automatic(self, method: str = "otsu") -> float:
        return threshold_of(self.data[self.valid], method)

    def value_range(self) -> tuple[float, float]:
        """For a slider: from the 0.1th percentile (the darkest voxels
        left out) to the brightest (smoothed: a single hot voxel is
        smoothed away, while a small bright core stays reachable)."""
        values = self.data[self.valid]
        low = float(np.percentile(values, 0.1))
        return low, float(max(values.max(), low + 1e-6))

    def contours(self, plane, depth: str, slab: tuple,
                 threshold: float, shortest: int = 4,
                 most: int = 400) -> list:
        """The outlines of what lies above threshold on the slice at
        slab = (position, position) (µm) along `depth`, or anywhere in a
        slab (low, high) (the outline of its maximum projection), as point
        lists (µm) in the order of plane (a 2D view's (vertical,
        horizontal) axes). Lines of fewer than `shortest` points (specks)
        are left out, and only the `most` longest are kept. Outside the
        volume: none."""
        from scipy import ndimage as ndi
        from skimage.measure import find_contours
        d = self.axes.index(depth)
        n = self.data.shape[d]
        first, step = self.first[depth], self.scale[depth]
        low, high = sorted(slab)
        if high < first - step / 2 or low > first + (n - 0.5) * step:
            return []
        k0 = min(max(int(round((low - first) / step)), 0), n - 1)
        k1 = min(max(int(round((high - first) / step)), k0), n - 1)
        part = np.take(self.data, range(k0, k1 + 1), axis=d)
        section = part.max(axis=d)
        have = [a for a in self.axes if a != depth]
        if have != list(plane):
            section = section.T
        v, h = plane
        # only outer edges (the user's wish: no holes): dim patches inside
        # an object count as above the threshold, as Fiji's Fill Holes
        above = section > threshold
        holes = ndi.binary_fill_holes(above) & ~above
        if holes.any():
            section = np.where(holes, max(threshold, float(section.max()))
                               + 1.0, section)
        lines = [line for line in find_contours(section, threshold)
                 if len(line) >= shortest]
        lines.sort(key=len, reverse=True)
        return [np.column_stack((self.first[v] + line[:, 0] * self.scale[v],
                                 self.first[h] + line[:, 1] * self.scale[h]))
                for line in lines[:most]]


def search_volume(image: vc.OmeZarrImage, channel: int | None = None,
                  region: dict | None = None, time: float | None = None,
                  smooth: float = 1.0) -> SearchVolume:
    """(0.18.0) The search's view of one timepoint (time: in the image's
    time unit; None: the first) of a channel: the same level, area and
    smoothing as find_object."""
    level = choose_level(image, region)
    names, space = image.axis_names, spatial_axes(image)
    lev = image.levels[level]
    idx = search_indices(image, level, region)
    t_axis, c_axis = vc.time_axis(image), image.channel_axis
    key = []
    for i, name in enumerate(names):
        if i == c_axis:
            key.append(channel or 0)
        elif i == t_axis:
            first, end = idx[name]       # (levels keep every timepoint)
            level0 = image.levels[0]
            k = first if time is None else int(round(
                (time - level0.translation[i]) / level0.scale[i]))
            key.append(min(max(k, first), end - 1))
        else:
            key.append(slice(*idx[name]))
    _, data, valid = _prepared(np.asarray(lev.array[tuple(key)]), smooth)
    index = {a: names.index(a) for a in space}
    return SearchVolume(
        data, valid, level, list(space),
        {a: lev.translation[i] + idx[a][0] * lev.scale[i]
         for a, i in index.items()},
        {a: lev.scale[i] for a, i in index.items()})


def _object_in(volume, method, smooth, grow, seed, follow,
               relative=None, step=None, fixed=None,
               seed_mode="stable") -> tuple:
    """object_in's work; returns (slices, threshold, voxels, contrast,
    mask, relative level of the threshold). fixed (0.18.0): the user's own
    threshold, used instead of the automatic one and of a seed's (at every
    timepoint: one value), without growing."""
    from scipy import ndimage as ndi
    nothing = (None, None, 0, 0.0, None, None)
    raw, data, valid = _prepared(volume, smooth)
    values = data[valid]
    if values.size == 0 or not float(values.max()) > float(values.min()):
        return nothing                       # flat: nothing stands out
    automatic = threshold = threshold_of(values, method)
    background = ~(data > automatic) & valid
    if background.sum() < 10:                # nearly all bright: no object
        return nothing[:1] + (threshold,) + nothing[2:]
    # the background's levels: median and noise of the raw data (whether
    # the object stands out), median of the smoothed data (where growing
    # stops: MIN_CONTRAST raw noise levels above it; smoothing makes the
    # noise smaller, but a floor at the smoothed noise would follow the
    # blur several voxels into the background)
    median, sigma = _robust(raw[background])
    median_s, _ = _robust(data[background])
    floor = median_s + MIN_CONTRAST * sigma
    if fixed is not None:
        threshold = float(fixed)         # the user's own (0.18.0)
        grow = False                     # (growing would undo it)
    elif seed is not None:
        threshold = _seed_threshold(data, valid, automatic, seed, floor,
                                    step, seed_mode)
    elif follow is not None and relative is not None and follow.any():
        median_f, _ = _background_level(data, valid, automatic)
        top = float(np.percentile(data[follow & valid], 90)) \
            if (follow & valid).any() else median_f
        if top > median_f:
            threshold = min(automatic, max(
                floor, median_f + relative * (top - median_f)))
    mask = (data > threshold) & valid
    labels, n = ndi.label(mask)
    if n == 0:
        return nothing[:1] + (threshold,) + nothing[2:]
    index = np.arange(1, n + 1)
    signal = ndi.sum_labels(data - threshold, labels, index)
    sizes = ndi.sum_labels(mask, labels, index)
    if seed is not None or follow is not None:
        best = _chosen_piece(labels, n, sizes, seed, follow)
        if best is None:
            return nothing[:1] + (threshold,) + nothing[2:]
    else:
        signal[sizes < MIN_VOXELS] = -np.inf
        best = int(np.argmax(signal))
        if not np.isfinite(signal[best]):
            return nothing[:1] + (threshold,) + nothing[2:]
    piece = labels == best + 1
    # 1. does it stand out at all (raw values, so smoothing does not help;
    # the background below the threshold used)
    below = ~mask & valid
    if threshold < automatic and below.sum() >= 10:
        median, sigma = _robust(raw[below])
    inside = float(raw[piece].mean())
    if sigma > 0:
        contrast = (inside - median) / sigma
    else:                                    # a perfectly flat background
        contrast = float("inf") if inside > median else 0.0
    if contrast < MIN_CONTRAST:              # e.g. noise split in two
        return None, threshold, 0, contrast, None, None
    # 2. its extent: grown (in the smoothed data) down to the floor
    grown = (ndi.binary_propagation(piece, mask=(data > floor) & valid)
             if grow and floor < threshold else piece)
    slices = ndi.find_objects(grown.astype(np.int8))[0]
    level = (_relative_level(data, valid, automatic, piece, threshold)
             if seed is not None else relative)
    return slices, threshold, int(grown.sum()), contrast, grown, level


def find_object(image: vc.OmeZarrImage, channel: int | None = None, *,
                region: dict | None = None, level: int | None = None,
                method: str = "otsu", margin: float = 0.0,
                smooth: float = 1.0, grow: bool = True,
                seed: dict | None = None, keep_mask: bool = False,
                timed_silhouettes: bool = False,
                progress: Callable[[float, int], None] | None = None,
                cancel: Callable[[], bool] | None = None,
                threshold: float | None = None,
                seed_mode: str = "stable") -> ObjectBox:
    """Find the bright object and the crop region around it (see the
    module's description).

    seed (0.16.0): {axis: position} (in the image's units; the time axis
             too), e.g. a click on the object: the object is the piece
             under the seed at its timepoint (not the one with most
             signal), and at the other timepoints the piece that overlaps
             the object at the neighbouring timepoint (it is followed
             through time, outwards from the seed's; past a timepoint
             where it is not found, from the last place it was found).
    keep_mask: keep the object's mask (all timepoints together, at the
             level searched) in result.mask, e.g. for silhouettes.
    timed_silhouettes (0.17.0): make the object's silhouettes at each
             timepoint from that timepoint's mask (result.timed_silhouettes;
             only the polygons are kept, not the masks), for a moving
             region's outlines per timepoint (see timed_silhouettes()).
             This takes longer: three outlines per timepoint.

    channel: the channel to search (default 0; ignored without channels).
    region:  {axis: (start, stop)} to search in (e.g. a crop region's
             ranges, the time axis included); None = the whole image.
    level:   the pyramid level (default: choose_level).
    method:  "otsu", "triangle" or "li" (finds the object; its extent comes
             from growing it, step 5).
    margin:  added on every side of the spatial box, in the image's unit.
    smooth:  Gaussian smoothing before thresholding, in voxels of the level.
    grow:    grow the object down to just above the background (its dim
             parts: True, for an object on a dark background), or keep what
             the automatic threshold found (False, for an object that
             touches other bright structures, which growing would take in).
    progress(done, total): timepoints done; since 0.18.0 also fractions
    of a timepoint while its search goes on (with a seed: per threshold
    tried), so `done` may be a float. cancel(): True to stop
    (SearchCancelled), checked between timepoints and those steps.
    threshold (0.18.0): the user's own, used at every timepoint (no
             growing); None: each timepoint's own (automatic, or the
             seed's).
    seed_mode (0.18.0): see _seed_threshold ("stable", "brightest",
             "whole").
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if seed_mode not in SEED_MODES:
        raise ValueError(f"seed_mode must be one of {SEED_MODES}, not "
                         f"{seed_mode!r}")
    # (kept apart: the loop below names each timepoint's result threshold)
    own = None if threshold is None else float(threshold)
    if level is None:
        level = choose_level(image, region)
    elif level not in usable_levels(image):
        raise ValueError(f"Level {level} cannot be searched (it does not "
                         "line up with level 0, or it also shrinks time).")
    names = image.axis_names
    space = spatial_axes(image)
    level0, lev = image.levels[0], image.levels[level]
    factors = vc.downsampling_factors(image)[level]
    idx = search_indices(image, level, region)
    voxels = int(np.prod([idx[a][1] - idx[a][0] for a in space],
                         dtype=np.int64))
    if voxels > 4 * MAX_VOXELS:
        raise ValueError(
            f"Level {level} has {voxels / 1e6:.0f} million voxels per "
            "timepoint in the search area; choose a coarser level.")
    if any(idx[a][1] <= idx[a][0] for a in idx):
        raise ValueError("The search area is empty (outside the image).")
    c_axis = image.channel_axis
    if c_axis is not None:
        channel = 0 if channel is None else int(channel)
        if not 0 <= channel < lev.array.shape[c_axis]:
            raise ValueError(f"No channel {channel}.")
    t_axis = vc.time_axis(image)
    t_name = names[t_axis] if t_axis is not None else None
    times = list(range(*idx[t_name])) if t_name else [None]
    n_t = len(times)
    result = ObjectBox(level=level, method=method, margin=float(margin),
                       times=times, seed_mode=seed_mode, own_threshold=own)
    result.search = {a: idx[a] for a in space}
    # The order of the timepoints: in order; with a seed, from the seed's
    # outwards (forwards, then backwards), each following its neighbour.
    start, seed_voxel = 0, None
    if seed is not None:
        seed_voxel = _search_voxel(image, level, idx, seed)
        if t_name and seed.get(t_name) is not None:
            k = round((seed[t_name] - level0.translation[t_axis])
                      / level0.scale[t_axis])
            start = min(max(k - idx[t_name][0], 0), n_t - 1)
    order = [start] + list(range(start + 1, n_t)) \
        + list(range(start - 1, -1, -1))
    per_t = [(None, None, None, 0.0)] * n_t   # (box, threshold, size, contrast)
    start_mask = previous = None
    relative = None          # the seed's threshold, carried to the others
    union = None
    for done, pos in enumerate(order):
        if cancel is not None and cancel():
            raise SearchCancelled("The search was cancelled.")
        if seed is not None and pos == start - 1:
            previous = start_mask           # backwards: from the seed's
        choose = {}
        if seed is not None:
            if pos == start:
                choose = {"seed": seed_voxel}
            elif previous is None:          # nothing at the seed: no object
                if progress is not None:
                    progress(done + 1, n_t)
                continue
            else:
                choose = {"follow": previous, "relative": relative}
        key = []
        for i, name in enumerate(names):
            if i == c_axis:
                key.append(channel)
            elif i == t_axis:
                key.append(times[pos])
            else:
                key.append(slice(*idx[name]))
        volume = np.asarray(lev.array[tuple(key)])

        def step(fraction, done=done):
            # (0.18.0) progress within this timepoint (the threshold tried
            # with a seed); Cancel then stops between those steps too
            if cancel is not None and cancel():
                raise SearchCancelled("The search was cancelled.")
            if progress is not None:
                progress(done + 0.95 * fraction, n_t)

        slices, threshold, size, contrast, mask, level = object_in(
            volume, method, smooth, grow, return_mask=True, step=step,
            fixed=own, seed_mode=seed_mode, **choose)
        # Where the object is not found, the next timepoint follows the
        # last place it was found (0.17.2, the user's choice of 2026-11-10:
        # following went on no further after a timepoint without it).
        if mask is not None or pos == start:
            previous = mask
        if pos == start:
            start_mask, relative = mask, level
        box = None
        if slices is not None:
            if keep_mask:
                union = mask.copy() if union is None else (union | mask)
            if timed_silhouettes:
                result.timed_silhouettes[times[pos]] = _silhouette_polygons(
                    image, mask, result.level, result.search, margin)
            box = _box_of(image, result.level, idx, slices, result)
        per_t[pos] = (box, threshold, size if box is not None else None,
                      contrast)
        if progress is not None:
            progress(done + 1, n_t)
    return _finish(result, image, per_t, union)


def _box_of(image: vc.OmeZarrImage, level: int, idx: dict, slices,
            result: ObjectBox) -> dict:
    """The box ({axis: (start, stop)} in the image's units) of an object
    found at `level` in the search area idx (its index slices there), and
    whether it touches the area's or the image's edge (result.touches,
    result.image_edges)."""
    names, space = image.axis_names, spatial_axes(image)
    level0, lev = image.levels[0], image.levels[level]
    factors = vc.downsampling_factors(image)[level]
    box = {}
    for axis, piece in zip(space, slices):
        i = names.index(axis)
        n = lev.array.shape[i]
        size_axis = idx[axis][1] - idx[axis][0]
        # at the area's edge: may be cut there; where that edge is the
        # image's own, a larger search would not help
        for at_edge, inside in (
                (piece.start == 0, idx[axis][0] > 0),
                (piece.stop == size_axis, idx[axis][1] < n)):
            if size_axis > 1 and at_edge:
                (result.touches if inside else result.image_edges).add(axis)
        # padded by one voxel of this level (edge voxels are only partly
        # object), then on level 0's grid, as crops are
        first = max(idx[axis][0] + piece.start - 1, 0)
        end = min(idx[axis][0] + piece.stop + 1, n)
        f = factors[i]
        first0 = first * f
        end0 = min(end * f, level0.array.shape[i])
        box[axis] = vc.index_range_to_physical(
            first0, end0, level0.scale[i], level0.translation[i])
    return box


def _finish(result: ObjectBox, image: vc.OmeZarrImage, per_t: list,
            union) -> ObjectBox:
    """Fill in a search's result from its timepoints: per_t[pos] = (box,
    threshold, voxels, contrast); union: the object's mask over all
    timepoints (or None). The ranges (all timepoints' boxes plus the
    margin, within the image), the t range and the notes."""
    space = spatial_axes(image)
    level0 = image.levels[0]
    t_axis = vc.time_axis(image)
    t_name = image.axis_names[t_axis] if t_axis is not None else None
    times, margin = result.times, result.margin
    result.found = [p[0] for p in per_t]
    result.thresholds = [p[1] for p in per_t]
    result.voxels = [p[2] for p in per_t]
    result.contrasts = [p[3] for p in per_t]
    result.mask = union
    found = [(t, box) for t, box in zip(times, result.found)
             if box is not None]
    if not found:
        result.notes.append("No object was found: nothing stands out "
                            "above the background.")
        return result
    ranges = {}
    for axis in space:
        low = min(box[axis][0] for _, box in found) - margin
        high = max(box[axis][1] for _, box in found) + margin
        full = image.extent(axis)
        ranges[axis] = (max(low, full[0]), min(high, full[1]))
    if t_name:
        # the timepoints are level 0's (usable_levels): frame centres
        first_t, last_t = found[0][0], found[-1][0]
        ranges[t_name] = (
            level0.translation[t_axis] + first_t * level0.scale[t_axis],
            level0.translation[t_axis] + last_t * level0.scale[t_axis])
        missing = len(times) - len(found)
        if missing:
            result.notes.append(
                f"The object was not found at {missing} of {len(times)} "
                "timepoints (the region's t range keeps those between the "
                "first and the last where it was found).")
    if result.touches:
        result.notes.append(
            "The object reaches the edge of the area searched along "
            + ", ".join(a for a in space if a in result.touches)
            + ": it may be cut there. Search a larger region (or the whole "
            "image).")
    if result.image_edges:
        result.notes.append(
            "The object reaches the image's own edge along "
            + ", ".join(a for a in space if a in result.image_edges)
            + " (it may go on outside the image).")
    result.ranges = ranges
    return result


def moving_region(image: vc.OmeZarrImage, result: ObjectBox,
                  within: dict | None = None) -> tuple[dict, dict]:
    """A crop region that follows the object over time (0.17.0): (ranges,
    track) for vc.CropBox (see CropBox.track).

    Its size along each spatial axis: the object's largest box over the
    timepoints where it was found, plus the margin on both sides, so that
    it holds the object at each of them. Its place at each of those
    timepoints: centred on the object's box then. At timepoints where the
    object was missed, the track's interpolation puts it between its
    places before and after. It is kept inside the image and inside
    `within` ({axis: (low, high)}, e.g. the region searched) where it
    fits: a place that would reach beyond is moved back (not cut); a size
    larger than that is cut to it.

    ranges: the region at the first timepoint where the object was found,
    and the t range of result.ranges; track: the region's shift at each
    timepoint found (0 at the first). Without a time axis, or with the
    object found at fewer than 2 timepoints: (result.ranges, {}), a fixed
    region as before 0.17.0.
    """
    t_axis = vc.time_axis(image)
    found = [(t, box) for t, box in zip(result.times, result.found)
             if box is not None]
    if t_axis is None or len(found) < 2 or not result.ranges:
        return dict(result.ranges), {}
    level0 = image.levels[0]
    space = spatial_axes(image)
    sizes, centres = {}, {t: {} for t, _ in found}
    for axis in space:
        low, high = image.extent(axis)
        if within and axis in within:
            low, high = max(low, within[axis][0]), min(high, within[axis][1])
        size = (max(box[axis][1] - box[axis][0] for _, box in found)
                + 2 * result.margin)
        size = min(size, high - low)
        sizes[axis] = size
        for t, box in found:
            centre = (box[axis][0] + box[axis][1]) / 2
            centres[t][axis] = min(max(centre, low + size / 2),
                                   high - size / 2)
    first = found[0][0]
    ranges = {axis: (centres[first][axis] - sizes[axis] / 2,
                     centres[first][axis] + sizes[axis] / 2)
              for axis in space}
    t_name = image.axis_names[t_axis]
    ranges[t_name] = result.ranges[t_name]
    track = {}
    for t, _ in found:
        time = level0.translation[t_axis] + t * level0.scale[t_axis]
        track[time] = {axis: centres[t][axis] - centres[first][axis]
                       for axis in space}
    return ranges, track


def _search_voxel(image: vc.OmeZarrImage, level: int, search: dict,
                  point: dict) -> list[int]:
    """A point {axis: value} as a voxel index of the search area at
    `level` (its spatial axes, in order; the middle of the area for an
    axis the point lacks), on level 0's grid first as crops are. It may
    lie outside the area."""
    names = image.axis_names
    level0 = image.levels[0]
    factors = vc.downsampling_factors(image)[level]
    voxel = []
    for axis in spatial_axes(image):
        i = names.index(axis)
        first, end = search[axis]
        value = point.get(axis)
        if value is None:
            voxel.append((end - first) // 2)
            continue
        j0, _ = vc.physical_to_index_range(value, value, level0.scale[i],
                                           level0.translation[i])
        voxel.append(j0 // factors[i] - first)
    return voxel


def mask_holds(image: vc.OmeZarrImage, result: ObjectBox, point: dict,
               reach: int = 0) -> bool:
    """Whether the object found (its mask over all timepoints, kept with
    keep_mask) holds a point {axis: value}, e.g. another seed, or comes
    within `reach` voxels (of the level searched) of it (0.16.0)."""
    if result.mask is None or any(a not in point
                                  for a in spatial_axes(image)):
        return False
    voxel = _search_voxel(image, result.level, result.search, point)
    near = tuple(slice(max(j - reach, 0), max(j + reach + 1, 0))
                 for j in voxel)
    return bool(result.mask[near].any())


# --- splitting an object that several seeds share (0.18.0) -------------------
#
# The user's wish (2026-11-13): touching embryos that no threshold
# separates, split as in ilastik's carving or MorphoLibJ's Marker-controlled
# Watershed: each seed floods its object in 3D from its own place; the
# floods meet where the image is dimmest between them and where the object
# narrows (a neck), and lines the user draws in the gap ("gap lines") are
# walls they cannot cross. Between the lines the image decides, at any
# angle (the floods are 3D).

# A seed's marker: a ball of this radius (voxels of the level searched).
MARKER_RADIUS = 2
# A gap line is a wall on the slice it was drawn on, and raises the
# landscape this many slices (of the level searched) to each side, fading.
GAP_REACH = 3
# How much a gap line raises it (the image's part spans 0 to 1).
GAP_HEIGHT = 2.0


def gap_walls(image: vc.OmeZarrImage, level: int, search: dict,
              lines: list, shape: tuple, offset=None) -> tuple:
    """Gap lines on the grid of the search area at `level` (spatial axes
    in order; a part of it of this shape, starting at index `offset`):
    (wall, ridge). wall: True on the slice a line was drawn on, within
    half its width of it (at least the voxels it crosses); ridge: 1 there,
    fading to 0 over GAP_REACH slices to each side.

    A line: {"plane": (v, h), "depth": axis, "at": position along depth,
    "points": [(v, h), ...], "width": µm, "time": ...} (the image's
    units), as the window keeps them."""
    from scipy import ndimage as ndi
    from skimage.draw import line as draw_line
    space, names = spatial_axes(image), image.axis_names
    level0 = image.levels[0]
    factors = vc.downsampling_factors(image)[level]
    offset = tuple(offset or (0,) * len(shape))
    wall = np.zeros(shape, bool)
    ridge = np.zeros(shape, np.float32)
    for gap in lines:
        v, h = gap["plane"]
        depth = gap["depth"]
        if any(a not in space for a in (v, h, depth)) or not gap["points"]:
            continue
        dv, dh, dd = (space.index(a) for a in (v, h, depth))
        k = _search_voxel(image, level, search,
                          {depth: gap["at"]})[dd] - offset[dd]
        corners = [_search_voxel(image, level, search, {v: p[0], h: p[1]})
                   for p in gap["points"]]
        rows = [c[dv] - offset[dv] for c in corners]
        cols = [c[dh] - offset[dh] for c in corners]
        size = (shape[dv], shape[dh])
        drawn = np.zeros(size, bool)
        pairs = list(zip(zip(rows, cols), zip(rows[1:], cols[1:]))) or [
            ((rows[0], cols[0]), (rows[0], cols[0]))]
        for (r0, c0), (r1, c1) in pairs:
            rr, cc = draw_line(int(r0), int(c0), int(r1), int(c1))
            ok = (rr >= 0) & (rr < size[0]) & (cc >= 0) & (cc < size[1])
            drawn[rr[ok], cc[ok]] = True
        if not drawn.any():
            continue
        sizes = [factors[names.index(a)] * level0.scale[names.index(a)]
                 for a in (v, h)]
        foot = ndi.distance_transform_edt(~drawn, sampling=sizes) \
            <= gap["width"] / 2
        foot |= drawn
        if [a for a in space if a != depth] != [v, h]:
            foot = foot.T                    # the slice's own axis order
        for dk in range(-GAP_REACH, GAP_REACH + 1):
            kk = k + dk
            if not 0 <= kk < shape[dd]:
                continue
            index = [slice(None)] * len(shape)
            index[dd] = kk
            index = tuple(index)
            ridge[index] = np.maximum(
                ridge[index], foot * (1.0 - abs(dk) / (GAP_REACH + 1)))
            if dk == 0:
                wall[index] |= foot
    return wall, ridge


def _floor_of(raw, data, valid, method) -> float | None:
    """Where growing stops (as in _object_in): MIN_CONTRAST raw noise
    levels above the smoothed background's median."""
    values = data[valid]
    if values.size == 0:
        return None
    automatic = threshold_of(values, method)
    background = ~(data > automatic) & valid
    if background.sum() < 10:
        return None
    _, sigma = _robust(raw[background])
    median_s, _ = _robust(data[background])
    return median_s + MIN_CONTRAST * sigma


def _seed_marker(voxel, inside) -> np.ndarray:
    """A seed's marker: a ball of MARKER_RADIUS voxels round it, inside
    the object; or, beside it, the nearest voxel of the object within
    SEED_REACH."""
    from scipy import ndimage as ndi
    marker = np.zeros(inside.shape, bool)
    if all(0 <= j < n for j, n in zip(voxel, inside.shape)):
        near = tuple(slice(max(j - MARKER_RADIUS, 0), j + MARKER_RADIUS + 1)
                     for j in voxel)
        grid = np.ogrid[near]
        ball = sum((g - j) ** 2 for g, j in zip(grid, voxel)) \
            <= MARKER_RADIUS ** 2
        marker[near] = ball
        marker &= inside
        if marker.any():
            return marker
    near = tuple(slice(max(j - SEED_REACH, 0), max(j + SEED_REACH + 1, 0))
                 for j in voxel)
    local = inside[near]
    if local.size == 0 or not local.any():
        return marker
    at = tuple(min(max(j - s.start, 0), n - 1)
               for j, s, n in zip(voxel, near, local.shape))
    distance, nearest = ndi.distance_transform_edt(~local,
                                                   return_indices=True)
    if distance[at] <= SEED_REACH:
        marker[tuple(i[at] + s.start for i, s in zip(nearest, near))] = True
    return marker


def split_objects(image: vc.OmeZarrImage, channel: int | None,
                  results: list, seeds: list, gaps: list | None = None, *,
                  method: str = "otsu", smooth: float = 1.0,
                  grow: bool = True, threshold: float | None = None,
                  timed_silhouettes: bool = False,
                  progress: Callable[[float, int], None] | None = None,
                  cancel: Callable[[], bool] | None = None) -> list:
    """(0.18.0) Split one object that several seeds share: one result per
    seed, as find_object would give for its part (box per timepoint,
    ranges, mask over all timepoints, silhouettes per timepoint if
    timed_silhouettes), by a seeded watershed in 3D (see above).

    results: each seed's find_object result (with keep_mask), all on the
    same level and search area (Fit to seeds searches the loaded part);
    their thresholds at each timepoint give the object (the lowest of
    them, so that each part is whole; `threshold`, the user's own, if
    given), grown down to just above the background with `grow` (not with
    one's own threshold), as find_object does. gaps: gap lines (see
    gap_walls); one with a "time" counts only at that timepoint.

    The landscape the seeds flood: low where the image is bright and deep
    inside the object (half each, both scaled 0-1), raised by the gap
    lines (GAP_HEIGHT). Time-lapses: the split starts at the first seed's
    timepoint (with every seed of the group; best put them at one
    timepoint) and goes forwards, then backwards, each timepoint's
    markers the neighbour's parts (shrunk by a voxel), so that the parts
    stay apart as they move."""
    from scipy import ndimage as ndi
    from skimage.segmentation import expand_labels, watershed
    first = results[0]
    level, search = first.level, dict(first.search)
    names, space = image.axis_names, spatial_axes(image)
    lev, level0 = image.levels[level], image.levels[0]
    factors = vc.downsampling_factors(image)[level]
    sizes = [factors[names.index(a)] * level0.scale[names.index(a)]
             for a in space]
    t_axis, c_axis = vc.time_axis(image), image.channel_axis
    t_name = names[t_axis] if t_axis is not None else None
    times = list(first.times)
    n_t, n_seeds = len(times), len(seeds)
    own = None if threshold is None else float(threshold)
    gaps = list(gaps or [])
    shape = tuple(search[a][1] - search[a][0] for a in space)
    voxels = [_search_voxel(image, level, search, seed) for seed in seeds]

    def position_of(value):              # a time as an index into times
        if t_name is None or value is None:
            return None
        k = round((value - level0.translation[t_axis])
                  / level0.scale[t_axis])
        return min(max(k - times[0], 0), n_t - 1)

    seed_pos = [position_of(seed.get(t_name)) if t_name else None
                for seed in seeds]
    seed_pos = [0 if p is None else p for p in seed_pos]
    start = seed_pos[0]
    order = [start] + list(range(start + 1, n_t)) \
        + list(range(start - 1, -1, -1))
    out = []
    for _ in seeds:
        part = ObjectBox(level=level, method=first.method,
                         margin=first.margin, times=list(times),
                         seed_mode=first.seed_mode, own_threshold=own)
        part.search = dict(search)
        out.append(part)
    per_t = [[(None, None, None, 0.0)] * n_t for _ in seeds]
    unions = [None] * n_seeds
    previous = start_labels = None
    for done, pos in enumerate(order):
        if cancel is not None and cancel():
            raise SearchCancelled("The search was cancelled.")
        if pos == start - 1:
            previous = start_labels             # backwards: from the start
        here = [k for k in range(n_seeds) if seed_pos[k] == pos]
        found_t = [r.thresholds[pos] for r in results
                   if len(r.thresholds) > pos and r.thresholds[pos]
                   is not None]
        level_t = own if own is not None else (
            min(found_t) if found_t else None)
        if level_t is None or (previous is None and not here):
            if progress is not None:
                progress(done + 1, n_t)
            continue
        key = []
        for i, name in enumerate(names):
            if i == c_axis:
                key.append(0 if channel is None else int(channel))
            elif i == t_axis:
                key.append(times[pos])
            else:
                key.append(slice(*search[name]))
        raw, data, valid = _prepared(np.asarray(lev.array[tuple(key)]),
                                     smooth)
        # the object: the pieces under the seeds, or those the neighbour's
        # parts overlap, above the threshold (grown as find_object grows)
        base = (data > level_t) & valid
        labels, n = ndi.label(base)
        keep = set()
        if n:
            pieces = ndi.sum_labels(base, labels, np.arange(1, n + 1))
            for k in here:
                j = _chosen_piece(labels, n, pieces, voxels[k], None)
                if j is not None:
                    keep.add(j + 1)
            if previous is not None:
                hit = np.unique(labels[(previous > 0) & base])
                keep |= {int(j) for j in hit if j > 0}
                if not keep:
                    j = _chosen_piece(labels, n, pieces, None, previous > 0)
                    if j is not None:
                        keep.add(j + 1)
        if not keep:
            if progress is not None:
                progress(done + 1, n_t)
            continue
        whole = np.isin(labels, sorted(keep))
        if grow and own is None:
            floor = _floor_of(raw, data, valid, method)
            if floor is not None and floor < level_t:
                whole = ndi.binary_propagation(
                    whole, mask=(data > floor) & valid)
        # each seed's marker: its ball at its own timepoint, else its part
        # at the neighbouring timepoint, shrunk by a voxel
        markers = np.zeros(shape, np.int32)
        for k in range(n_seeds):
            if seed_pos[k] == pos:
                mark = _seed_marker(voxels[k], whole)
            elif previous is not None:
                was = previous == k + 1
                mark = ndi.binary_erosion(was) & whole
                if not mark.any():
                    mark = was & whole
            else:
                continue
            markers[mark & (markers == 0)] = k + 1
        if not markers.any():
            if progress is not None:
                progress(done + 1, n_t)
            continue
        # the work in the object's box (plus the gap lines' reach)
        pad = GAP_REACH + 2
        bounds = ndi.find_objects(whole.astype(np.int8))[0]
        box = tuple(slice(max(b.start - pad, 0), min(b.stop + pad, n_k))
                    for b, n_k in zip(bounds, shape))
        part_shape = tuple(b.stop - b.start for b in box)
        inside = whole[box]
        here_gaps = [g for g in gaps if t_name is None
                     or g.get("time") is None
                     or position_of(g["time"]) == pos]
        if here_gaps:
            wall, ridge = gap_walls(image, level, search, here_gaps,
                                    part_shape, [b.start for b in box])
        else:
            wall = np.zeros(part_shape, bool)
            ridge = np.zeros(part_shape, np.float32)
        flood = inside & ~wall
        values = data[box]
        top = float(np.percentile(values[inside], 99))
        bright = np.clip((values - level_t) / max(top - level_t, 1e-6), 0, 1)
        deep = ndi.distance_transform_edt(flood, sampling=sizes)
        deep = deep / max(float(deep.max()), 1e-6)
        landscape = -(0.5 * bright + 0.5 * deep) + GAP_HEIGHT * ridge
        seeds_here = np.where(flood, markers[box], 0)
        parts = watershed(landscape, seeds_here, mask=flood)
        if wall.any():           # the wall's own voxels: the nearest part
            grown = expand_labels(parts, distance=max(part_shape))
            parts = np.where(inside & (parts == 0), grown, parts)
        labels_t = np.zeros(shape, np.int16)
        labels_t[box] = parts
        for k in range(n_seeds):
            piece = labels_t == k + 1
            count = int(piece.sum())
            if count < MIN_VOXELS:
                continue
            slices = ndi.find_objects(piece.astype(np.int8))[0]
            contrast = (results[k].contrasts[pos]
                        if len(results[k].contrasts) > pos else 0.0)
            per_t[k][pos] = (_box_of(image, level, search, slices, out[k]),
                             level_t, count, contrast)
            unions[k] = piece if unions[k] is None else (unions[k] | piece)
            if timed_silhouettes:
                out[k].timed_silhouettes[times[pos]] = _silhouette_polygons(
                    image, piece, level, search, out[k].margin)
        previous = labels_t
        if pos == start:
            start_labels = labels_t
        if progress is not None:
            progress(done + 1, n_t)
    return [_finish(out[k], image, per_t[k], unions[k])
            for k in range(n_seeds)]


# The views' planes (vertical, horizontal) by the axis they look along, as
# crop regions keep their polygons (ortho_views.view_orders): XY looks
# along z, XZ along y, YZ along x.
PLANE_OF_DEPTH = {"z": ("y", "x"), "y": ("z", "x"), "x": ("y", "z")}


def silhouettes(image: vc.OmeZarrImage, result: ObjectBox,
                ranges: dict | None = None) -> dict:
    """The object's outline seen along each spatial axis (0.16.0): one
    silhouette polygon per view (XY, XZ, YZ), from its mask over all
    timepoints (find_object with keep_mask), grown by the margin and by
    one voxel of the level searched (as the box is), reaching through the
    region's range along that view's depth axis (`ranges`: the region's
    ranges, default result.ranges). As crop regions keep them:
    {plane: {(depth_start, depth_stop): [(v, h), ...]}}. Where the
    projection falls apart into pieces (e.g. an object that moved a lot),
    their convex hull, so that no piece is cut off. {} without a mask.

    As the box: the outline lies the margin (exactly, in µm) plus one
    voxel of the level searched beyond the mask's outer voxel edges, and
    voxel j of the level searched stands for level 0's voxels f*j ...
    f*j + f - 1 (f its downsampling factor), whatever translation the
    level itself records (the review of 2026-11-08: with the level's own
    translation, outlines on pyramids whose levels repeat level 0's
    translation, as eubi-bridge writes them, were shifted by (f - 1) / 2
    voxels of level 0, and the margin was rounded down to whole voxels of
    the level searched, so a crop with the outline lost the margin and
    could cut the object)."""
    mask = result.mask
    if mask is None or not result.ranges or not mask.any():
        return {}
    ranges = ranges or result.ranges
    return {plane: {tuple(ranges[depth]): points}
            for depth, (plane, points) in _silhouette_polygons(
                image, mask, result.level, result.search,
                result.margin).items()}


def timed_silhouettes(image: vc.OmeZarrImage, result: ObjectBox,
                      box: vc.CropBox, ranges_at=None) -> dict:
    """A moving region's outlines per timepoint (0.17.0), from
    find_object(..., timed_silhouettes=True): at each timepoint where the
    object was found, its own silhouettes (as `silhouettes` makes them, from
    that timepoint's mask), each reaching through the region's range then
    along the view's depth axis: ranges_at(time) (the window passes the
    region as cropped, which a "selected region" projection shows, so that
    the silhouette can be edited there), default vc.ranges_at(box, time).
    {time: outlines}, for CropBox.timed_outlines."""
    t_axis = vc.time_axis(image)
    if t_axis is None:
        return {}
    level0 = image.levels[0]
    timed = {}
    for t, polygons in result.timed_silhouettes.items():
        time = level0.translation[t_axis] + t * level0.scale[t_axis]
        ranges = (ranges_at(time) if ranges_at is not None
                  else vc.ranges_at(box, time))
        outlines = {plane: {tuple(ranges[depth]): points}
                    for depth, (plane, points) in polygons.items()
                    if depth in ranges}
        if outlines:
            timed[time] = outlines
    return timed


def _silhouette_polygons(image, mask, level, search, margin) -> dict:
    """silhouettes' work for one mask (at pyramid level `level`, over the
    search area `search`): {depth axis: (plane, points)}."""
    space = spatial_axes(image)
    outlines = {}
    for depth, plane in PLANE_OF_DEPTH.items():
        if depth not in space or any(a not in space for a in plane):
            continue
        if any(mask.shape[space.index(a)] == 1 for a in plane):
            continue                       # (a 2D image: only XY)
        projection = mask.any(axis=space.index(depth))
        kept = [a for a in space if a != depth]          # its axes in order
        if kept != list(plane):
            projection = projection.T
        points = _outline_points(image, level, search, plane, projection,
                                 margin, join="hull")
        if points:
            outlines[depth] = (plane, points)
    return outlines


def _outline_points(image, level, search, plane, section, margin,
                    join: str = "hull", avoid=None) -> list:
    """The outline of a 2D mask (a projection or a slice of a mask at
    `level` over `search`, in the order of `plane`) as polygon points
    (µm, on level 0's grid), or [] if none: the mask grown by the margin
    and one voxel of the level searched (as boxes are), traced on a
    distance field so that it keeps fractions of a voxel, then simplified.

    join: where the grown mask falls apart into pieces, "hull" takes their
    convex hull (the silhouettes, so that no piece is cut off), "bridge"
    joins them by thin bridges (per-slice outlines, which must not take
    in a neighbour lying between them). avoid (0.18.0): a 2D mask of
    neighbouring objects (the other parts of a split): the margin then
    grows only where the gap is nearer to this mask than to them, so that
    outlines of touching parts meet instead of overlapping."""
    from scipy import ndimage as ndi
    from skimage.measure import approximate_polygon, find_contours
    from skimage.morphology import convex_hull_image
    names = image.axis_names
    level0 = image.levels[0]
    factors = vc.downsampling_factors(image)[level]
    tolerance = 0.5                  # of approximate_polygon, in voxels
    # a voxel of the level searched, in µm (on level 0's grid)
    sizes = [factors[names.index(a)] * level0.scale[names.index(a)]
             for a in plane]
    # how far beyond the mask's outer voxel centres, in voxels of the
    # level searched: half a voxel to its edge, one of padding, and the
    # margin
    grow = [margin / s + 1.5 for s in sizes]
    padding = int(np.ceil(max(grow) + tolerance)) + 2
    padded = np.pad(section, padding)
    sampling = [1.0 / g for g in grow]
    # distance to the mask, 1.0 = `grow` voxels along either axis; its
    # contour (not that of a thresholded mask) keeps fractions of a voxel.
    # A little further out, so that simplifying the polygon (which may cut
    # corners by `tolerance`) never cuts into it.
    distance = ndi.distance_transform_edt(~padded, sampling=sampling)
    level = 1.0 + tolerance / min(grow)
    labels, n = ndi.label(distance <= level)
    if n > 1:
        joined = (convex_hull_image(padded) if join == "hull"
                  else _bridged(padded))
        distance = ndi.distance_transform_edt(~joined, sampling=sampling)
    field = distance - level
    if avoid is not None and np.any(avoid):
        others = np.pad(np.asarray(avoid, bool) & ~section, padding)
        if others.any():
            nearer = distance - ndi.distance_transform_edt(
                ~others, sampling=sampling)
            field = np.maximum(field, nearer)
    contours = find_contours(field, 0.0)
    if not contours:
        return []
    contour = max(contours, key=len)
    contour = approximate_polygon(
        contour, tolerance=tolerance if avoid is None else tolerance / 2)
    points = []
    for v, h in contour[:-1] if len(contour) > 3 else contour:
        point = []
        for axis, value in zip(plane, (v, h)):
            i = names.index(axis)
            j = value - padding + search[axis][0]
            f = factors[i]
            # level-0 position: the centre of the voxels it stands for
            u0 = f * j + (f - 1) / 2
            point.append(float(level0.translation[i]
                               + u0 * level0.scale[i]))
        points.append(tuple(point))
    return points if len(points) >= 3 else []


def _bridged(mask) -> np.ndarray:
    """A 2D mask's pieces joined into one by thin bridges (straight lines,
    a voxel wide, each from the piece nearest to what is joined so far);
    specks of fewer than 3 voxels are left out."""
    from scipy import ndimage as ndi
    from skimage.draw import line as draw_line
    labels, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum_labels(mask, labels, np.arange(1, n + 1))
    keep = np.isin(labels, np.flatnonzero(sizes >= 3) + 1)
    labels, n = ndi.label(keep)
    if n <= 1:
        return keep
    sizes = ndi.sum_labels(keep, labels, np.arange(1, n + 1))
    joined = labels == int(np.argmax(sizes)) + 1
    rest = keep & ~joined
    while rest.any():
        distance, nearest = ndi.distance_transform_edt(~joined,
                                                       return_indices=True)
        q = np.unravel_index(int(np.argmin(np.where(rest, distance,
                                                    np.inf))), rest.shape)
        p = tuple(int(i[q]) for i in nearest)
        bridge = np.zeros(mask.shape, bool)
        rr, cc = draw_line(p[0], p[1], q[0], q[1])
        bridge[rr, cc] = True
        joined |= bridge | (labels == labels[q])
        rest &= labels != labels[q]
    return joined


# (0.18.0) Per-slice outlines: at most this many polygons, on slices of
# the XY view (along z), from the object's first slice to its last.
MAX_SLICE_OUTLINES = 40
SLICE_PLANE = ("y", "x")


def slice_outlines(image: vc.OmeZarrImage, result: ObjectBox,
                   ranges: dict | None = None, others=None,
                   plane: tuple = SLICE_PLANE,
                   most: int = MAX_SLICE_OUTLINES) -> dict:
    """(0.18.0, the user's wish) The object's outline on slices of one view
    (XY), as crop regions keep polygons: {plane: {position along its depth
    axis (µm): [(v, h), ...]}}, from its mask over all timepoints
    (find_object with keep_mask, or a part of split_objects). The crop's
    3D shape then follows the object slice by slice, where three
    silhouettes keep a block carved from three sides (which, for objects
    touching at an angle, holds corners of the neighbour).

    Polygons on up to `most` slices of the level searched, evenly from the
    object's first slice to its last (the tool interpolates between
    them), each the object's outline there grown by the margin and a
    voxel (as silhouettes), only its outer edge (no holes), its pieces
    joined by thin bridges; the first and the last repeated the margin
    and one and a half voxels further out (the shape ends at the
    outermost polygons), within `ranges` (the region's) along the depth
    axis. others: the mask of neighbouring objects on the same grid (the
    other parts of a split): the margin stops halfway to them. {} without
    a mask."""
    mask = result.mask
    if mask is None or not mask.any():
        return {}
    space, names = spatial_axes(image), image.axis_names
    depths = [a for a in space if a not in plane]
    if len(depths) != 1 or any(a not in space for a in plane):
        return {}
    depth = depths[0]
    d = space.index(depth)
    level, search, margin = result.level, result.search, result.margin
    level0 = image.levels[0]
    factors = vc.downsampling_factors(image)[level]
    i = names.index(depth)
    f = factors[i]

    def position(k):          # slice k of the level searched, in µm
        u0 = f * (search[depth][0] + k) + (f - 1) / 2
        return float(level0.translation[i] + u0 * level0.scale[i])

    other_axes = tuple(a for a in range(mask.ndim) if a != d)
    filled = np.flatnonzero(mask.any(axis=other_axes))
    chosen = np.unique(np.round(np.linspace(
        filled[0], filled[-1], min(most, filled[-1] - filled[0] + 1)))
        .astype(int))
    kept = [a for a in space if a != depth]
    polygons = {}
    for k in chosen:
        section = np.take(mask, k, axis=d)
        avoid = None if others is None else np.take(others, k, axis=d)
        if kept != list(plane):
            section = section.T
            avoid = None if avoid is None else avoid.T
        if not section.any():
            continue
        points = _outline_points(image, level, search, plane, section,
                                 margin, join="bridge", avoid=avoid)
        if points:
            polygons[position(k)] = points
    if not polygons:
        return {}
    # the ends, the margin (and the box's padding) further out
    push = margin + 1.5 * f * level0.scale[i]
    low, high = (ranges or result.ranges).get(depth, (-np.inf, np.inf))
    first, last = min(polygons), max(polygons)
    for at, target in ((first, max(first - push, low)),
                       (last, min(last + push, high))):
        if abs(target - at) > 1e-9:
            polygons[target] = list(polygons[at])
    return {tuple(plane): dict(sorted(polygons.items()))}


def describe(result: ObjectBox, image: vc.OmeZarrImage) -> str:
    """A few lines for the window's status."""
    if not result.ranges:
        return "\n".join(result.notes) or "No object found."
    unit = vc.short_unit(image.unit(spatial_axes(image)[-1]))
    unit = "µm" if unit == "um" else unit
    if result.own_threshold is not None:                       # (0.18.0)
        how = f"your own threshold, {result.own_threshold:.4g}"
    else:
        how = f"{METHOD_NAMES[result.method]} threshold"
    lines = [f"Found the object at {result.n_found} of {len(result.times)} "
             f"timepoint(s), at level {result.level} ({how}), margin "
             f"{result.margin:g} {unit}:"]
    for axis in spatial_axes(image):
        low, high = result.ranges[axis]
        lines.append(f"  {axis}: {low:.1f}–{high:.1f} {unit} "
                     f"({high - low:.1f} {unit})")
    lines += result.notes
    return "\n".join(lines)
