"""Levels (0.18.0): the histogram and statistics of one channel's values,
for the panel's "Levels" part (levels_panel.py). No window needed.

The values come from one of three places (the user's choice, 2026-11-12):

- the XY view as shown: what napari has loaded for it (the panel takes
  these from napari; the visible part of the slice, or of the projection,
  at the level shown);
- the whole image: the loaded part of it, all slices;
- the selected crop region: the voxels its crop will keep (inside its
  outlines; the voxels the crop sets to 0 are left out).

The last two are read at the timepoint shown, from the finest pyramid level
that holds at most MAX_VALUES voxels there (coarser levels may smooth the
extremes of the data). Like Fiji's histograms, the bins cover the data's
range; here also the contrast limits, so that both are always on the axis.

(0.22.0, the user's wish) "Zoom to min–max": the bins cover only the
contrast limits and some room on either side (ZOOM_MARGIN of the distance
between them), within that range, so that the part of the histogram that is
being set fills the axis, in finer bins, and the limits can still be
dragged further out (axis_range).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volume_cropper import ortho_views as ov
from volume_cropper import core as vc

BINS = 256                 # bins of a histogram (Fiji uses 256 too)
MAX_VALUES = 4_000_000     # voxels read for "whole image" or "crop region"
# (0.22.0) "zoom to min–max": the room shown on either side of the limits,
# as a share of the distance between them (the user's choice, 2026-11-25:
# 25 %, so that the limits sit at 1/6 and 5/6 of the axis)
ZOOM_MARGIN = 0.25


@dataclass
class Histogram:
    """A histogram of some values, and their statistics.

    edges: bins + 1 bin edges; counts: values per bin. For integer data a
    bin holds whole numbers only (edges at half-integers), e.g. one value
    per bin when the range is at most BINS values wide.
    """

    edges: np.ndarray
    counts: np.ndarray
    values: np.ndarray       # the values themselves (1D), for fractions
    minimum: float
    maximum: float
    mean: float
    at_max: int              # how many values equal the maximum
    integer: bool

    @property
    def n(self) -> int:
        return int(self.values.size)

    def outside(self, limits) -> tuple[float, float]:
        """Fractions of the values below the lower and above the upper
        contrast limit (shown black and fully bright)."""
        if self.n == 0:
            return 0.0, 0.0
        low, high = limits
        return (np.count_nonzero(self.values < low) / self.n,
                np.count_nonzero(self.values > high) / self.n)

    def bin_at(self, x: float) -> int | None:
        """The bin that holds the value x, or None outside the bins."""
        if x < self.edges[0] or x > self.edges[-1]:
            return None
        return int(min(np.searchsorted(self.edges, x, side="right") - 1,
                       self.counts.size - 1))


def axis_range(data_range, limits=None, margin: float | None = None,
               keep=()) -> tuple[float, float]:
    """The values the histogram's axis covers.

    Without margin: the data's range (minimum, maximum) and the contrast
    limits (low, high), so that both are always on the axis.

    (0.22.0) With margin ("zoom to min–max"): the limits, with room of
    margin × (high - low) on either side to drag them further out, and the
    values in keep (e.g. the Autocropper's threshold, so that it stays in
    view); but never beyond the axis without margin: there is nothing to
    see beyond the data.
    """
    low, high = float(data_range[0]), float(data_range[1])
    if limits is None:
        return low, high
    lower, upper = float(limits[0]), float(limits[1])
    low, high = min(low, lower), max(high, upper)      # the whole axis
    if margin is None:
        return low, high
    room = margin * (upper - lower)
    start, stop = lower - room, upper + room
    for value in keep:
        start, stop = min(start, float(value)), max(stop, float(value))
    return max(start, low), min(stop, high)


def histogram(values, limits=None, bins: int = BINS,
              margin: float | None = None, keep=()) -> Histogram | None:
    """The histogram of values, with bins covering the data's range and the
    contrast limits (low, high) if given; with margin (0.22.0, "zoom to
    min–max"), only the limits and room on either side (see axis_range).
    The statistics (minimum, maximum, mean, ...) are always those of all
    the values, also of those beyond a zoomed axis. None if there are no
    values."""
    values = np.asarray(values).ravel()
    integer = np.issubdtype(values.dtype, np.integer) \
        or values.dtype == np.bool_
    if not integer:
        values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    minimum, maximum = values.min(), values.max()
    low, high = axis_range((minimum, maximum), limits, margin, keep)
    if integer:
        # whole-number bins: n values per bin, edges at half-integers
        low, high = np.floor(low), np.ceil(high)
        width = max(1, int(np.ceil((high - low + 1) / bins)))
        n_bins = int(np.ceil((high - low + 1) / width))
        start = low - 0.5
        stop = start + n_bins * width
    else:
        if high <= low:                       # all the same value
            high = low + max(abs(low) * 1e-3, 1e-6)
        n_bins, start, stop = bins, low, high
    # A range and a number of bins (not an array of edges): numpy's fast way
    counts, edges = np.histogram(values, bins=n_bins, range=(start, stop))
    return Histogram(edges=edges, counts=counts, values=values,
                     minimum=float(minimum), maximum=float(maximum),
                     mean=float(values.mean(dtype=np.float64)),
                     at_max=int(np.count_nonzero(values == maximum)),
                     integer=bool(integer))


# ---------------------------------------------------------------------------
# Reading values from the image ("whole image", "crop region")
# ---------------------------------------------------------------------------

def _spatial_index(image: vc.OmeZarrImage) -> list[int]:
    return [i for i, a in enumerate(image.axes) if a.get("type") == "space"]


def spatial_slices(image: vc.OmeZarrImage, region: ov.LoadedRegion,
                   level: int, ranges: dict | None = None):
    """The voxels of one level to read, one slice per spatial axis (image
    order): the loaded part, and within it the voxels whose centres lie
    inside ranges ({axis: (low, high)}, e.g. a crop region as cropped).
    None if no voxel lies inside."""
    index = _spatial_index(image)
    start, stop = region.starts[level], region.stops[level]
    slices = [slice(start[i], stop[i]) for i in index]
    if ranges is None:
        return slices
    lv = image.levels[level]
    names = [image.axis_names[i] for i in index]
    box = ov._box_slices(names, [lv.scale[i] for i in index],
                         [lv.translation[i] for i in index],
                         [lv.array.shape[i] for i in index], ranges)
    if box is None:
        return None
    inside = []
    for loaded, cut in zip(slices, box):
        a, b = max(loaded.start, cut.start), min(loaded.stop, cut.stop)
        if b <= a:
            return None
        inside.append(slice(a, b))
    return inside


def values_level(image: vc.OmeZarrImage, region: ov.LoadedRegion,
                 ranges: dict | None = None,
                 max_values: float = MAX_VALUES) -> int:
    """The finest level of the loaded region with at most max_values voxels
    (one channel, one timepoint) inside ranges (default: all)."""
    for level in range(region.n_levels):
        slices = spatial_slices(image, region, level, ranges)
        n = 0 if slices is None else np.prod([s.stop - s.start
                                              for s in slices])
        if n <= max_values:
            return level
    return region.n_levels - 1


def time_index(image: vc.OmeZarrImage, region: ov.LoadedRegion, level: int,
               time: float | None) -> int | None:
    """The voxel index along the time axis of a level for a time (the
    timepoint shown), within the loaded part; None without a time axis."""
    t = vc.time_axis(image)
    if t is None:
        return None
    start, stop = region.starts[level][t], region.stops[level][t]
    if time is None:
        return start
    lv = image.levels[level]
    index = int(round((time - lv.translation[t]) / lv.scale[t]))
    return min(max(index, start), stop - 1)


def read_values(image: vc.OmeZarrImage, region: ov.LoadedRegion, k: int,
                time: float | None = None, ranges: dict | None = None,
                box: vc.CropBox | None = None,
                max_values: float = MAX_VALUES) -> tuple[np.ndarray, int]:
    """The values of channel k at a time (the timepoint shown): of the
    loaded region, or of the voxels inside ranges (a crop region as
    cropped) that its outlines keep (box: the region, for its outlines and
    margin). Returns (values as a 1D array, the level read)."""
    level = values_level(image, region, ranges, max_values)
    block = read_block(image, region, k, level, time, ranges, box)
    if block is None:
        return np.zeros(0, dtype=image.levels[level].array.dtype), level
    if block.zeroed is None:
        return block.values.ravel(), level
    return block.values[~block.zeroed], level     # without the zeroed voxels


@dataclass
class Block:
    """The voxels of one channel inside some ranges, at one level and
    timepoint (read_block).

    values: the spatial axes only, in the image's order (`axes`);
    zeroed: True where a crop region's outlines set the voxel to 0 (None
    without outlines); first: the centre of the first voxel per axis, and
    scale its size, in µm (as in the image's metadata).
    """

    values: np.ndarray
    zeroed: np.ndarray | None
    axes: list
    scale: list
    first: list


def read_block(image: vc.OmeZarrImage, region: ov.LoadedRegion, k: int,
               level: int, time: float | None = None,
               ranges: dict | None = None,
               box: vc.CropBox | None = None) -> Block | None:
    """Channel k at one level and time (the timepoint shown), of the
    loaded region or of the voxels inside ranges (a crop region as
    cropped; box: the region, for its outlines and margin). None if no
    voxel lies inside."""
    lv = image.levels[level]
    slices = spatial_slices(image, region, level, ranges)
    if slices is None:
        return None
    index = _spatial_index(image)
    key = []
    for i, axis in enumerate(image.axes):
        if i in index:
            key.append(slices[index.index(i)])
        elif i == image.channel_axis:
            key.append(k)
        elif i == vc.time_axis(image):
            key.append(time_index(image, region, level, time))
        else:                     # any other axis: its first loaded voxel
            key.append(region.starts[level][i])
    values = np.asarray(lv.array[tuple(key)])
    names = [image.axis_names[i] for i in index]
    scale = [lv.scale[i] for i in index]
    first = [lv.translation[i] + s.start * lv.scale[i]
             for i, s in zip(index, slices)]
    zeroed = None
    if ranges is not None and box is not None and box.outlines:
        # the voxels that the crop's outlines set to 0
        zeroed = ov.masked_region(names, scale, first, values.shape, ranges,
                                  box.outlines,
                                  spatial=vc._spatial_axes(image),
                                  margin=box.margin * vc.pixel_size(image))
    return Block(values, zeroed, names, scale, first)


def describe_number(value: float, integer: bool) -> str:
    """A value as the panel shows it: whole numbers for integer data, else
    4 significant digits."""
    if integer:
        return f"{int(round(value)):,}"
    return f"{value:.4g}"


# --- gamma and the number boxes (0.18.0, the user's wish) -------------------
# napari shows a value v between the contrast limits (low, high) at the
# brightness ((v - low) / (high - low)) ** gamma, from 0 (black) to 1 (full).
# Gamma 1 is a straight line; below 1 brightens the dim values, above 1
# darkens them. Like the grey slider of Photoshop's Levels, the panel's
# dashed "midtone" line marks the value shown at half brightness, and
# dragging it sets gamma. Gamma changes only the display, never the data.

GAMMA_RANGE = (0.2, 2.0)   # napari's own gamma slider puts other values back
                           # into this range (napari 0.9.2, qt_contrast_limits.py)


def clamp_gamma(gamma: float) -> float:
    """gamma within GAMMA_RANGE, rounded to 2 decimals (as the box shows
    it)."""
    low, high = GAMMA_RANGE
    return round(min(max(float(gamma), low), high), 2)


def midtone_value(limits, gamma: float) -> float:
    """The value shown at half brightness with these limits and gamma.

    Half brightness: ((v - low) / (high - low)) ** gamma = 0.5, so
    v = low + (high - low) * 0.5 ** (1 / gamma). With gamma 1 it is halfway
    between the limits; with 0.2 at 3 % of the way, with 2 at 71 %."""
    low, high = limits
    return low + (high - low) * 0.5 ** (1.0 / gamma)


def gamma_for_midtone(limits, value: float) -> float:
    """The gamma that shows `value` at half brightness (the inverse of
    midtone_value), kept within GAMMA_RANGE and rounded like clamp_gamma.

    With t = (value - low) / (high - low), t ** gamma = 0.5 gives
    gamma = log(0.5) / log(t)."""
    low, high = limits
    t = (value - low) / max(high - low, 1e-12)
    t_lowest = 0.5 ** (1.0 / GAMMA_RANGE[0])     # 0.03: gamma 0.2
    t_highest = 0.5 ** (1.0 / GAMMA_RANGE[1])    # 0.71: gamma 2
    t = min(max(t, t_lowest), t_highest)
    return clamp_gamma(np.log(0.5) / np.log(t))


def box_decimals(span: float, integer: bool) -> int:
    """Decimals for a min / max box whose values span `span`: none for
    integer data, else about 4 significant digits (1-8 decimals)."""
    if integer:
        return 0
    if not (np.isfinite(span) and span > 0):
        return 3
    return int(min(max(3 - np.floor(np.log10(span)), 1), 8))


def box_step(span: float, integer: bool) -> float:
    """A round step for a min / max box's arrows: about 1 % of the span
    (1, 2 or 5 times a power of ten); at least 1 for integer data."""
    if not (np.isfinite(span) and span > 0):
        return 1.0
    wanted = span / 100
    power = 10.0 ** np.floor(np.log10(wanted))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= wanted)
    return float(max(step, 1.0)) if integer else float(step)
