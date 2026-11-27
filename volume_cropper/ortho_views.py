"""Orthogonal views of an OME-Zarr image in napari: XY, XZ, YZ and 3D.

Layout, as in Fiji's Orthogonal Views::

    +-----------+------+
    |    XY     |  YZ  |   XY: y down, x right (napari's own view)
    |           |      |   YZ: y down, z right (same y axis as XY)
    +-----------+------+
    |    XZ     |  3D  |   XZ: z down, x right (same x axis as XY)
    +-----------+------+

How it works:

- Every view is a napari "viewer model" with its own canvas and slice slider.
  XY is napari's main viewer, so the normal layer list and layer controls act
  on it. Display changes made there (contrast, colour, visibility, ...) are
  copied to the other views, and so is the drawing tool chosen for the
  "crop regions" layer.
- XZ and YZ show the same lazy data with the axes reordered (a dask
  transpose: nothing is copied). The chunks are cubes, so an XZ or YZ plane
  costs as much to read as an XY plane.
- One shared cursor (a position in µm) ties the views together. Each view's
  slider moves its own plane, and crosshair lines show where the other planes
  are. Hovering over a 2D view and pressing J moves the other views there.
- Zoom and pan of the 2D views are linked, so that shared axes line up.
- Boxes in the 2D views: a solid line shows the box as it will be cropped,
  a dashed line a polygon/lasso outline. Both fade when the current slice
  lies outside the box.
- The 3D view shows one whole pyramid level (small by default). It is off
  until switched on (reading a whole volume takes time), and it can be
  clipped to a box (then the voxels outside it are 0, so no halo of them
  shows along its faces). The voxels that the selected box's outlines will
  black out can be shown normally, dimmed, in grey, dimmed in grey, or
  hidden.
- Time-lapse images get one time slider, shared by all views.
- Only the loaded region is read and shown (LoadedRegion): the whole image,
  or the ranges chosen when opening. Over a network, reading less is what
  makes opening fast. Crops are not affected: boxes are in µm.

This uses two napari internals: the main window's centre widget and
`viewer.window._qt_viewer`. Tested with napari 0.9.2.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, partial

import dask.array as da
import napari
import numpy as np
from napari.components import ViewerModel
from napari.layers import Image, Points, Shapes, Surface, Vectors
from napari.layers.base._base_constants import ActionType
from napari.qt import QtViewer
from napari.utils.colormaps import Colormap
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QApplication, QHBoxLayout, QSplitter,
                            QToolButton, QVBoxLayout, QWidget)
from skimage.measure import find_contours

from volume_cropper import box_handles
from volume_cropper import crop_projection
from volume_cropper import core as vc
# (0.23.0) plain-text labels; layer names that napari's own labels never
# take for HTML (see qt_text)
from volume_cropper.qt_text import QLabel, defang

# The 3D view's "Auto" level: the finest level that fits this budget (the
# loaded region, the channels shown) with at most AUTO_3D_SIDE voxels per
# side, a size every graphics card accepts for a 3D texture. The volume is
# read in one go, so the budget is kept small for speed over a network
# (the user's choice, 2026-10-27; it was 300 MB until then).
AUTO_3D_BYTES = 100e6
AUTO_3D_SIDE = 2048
# Small level used to estimate the starting contrast (fast to read): at
# most PREVIEW_SIZE voxels per side and PREVIEW_VOXELS voxels in all.
PREVIEW_SIZE = 256
PREVIEW_VOXELS = 2e6
# (0.18.0) The crop preview's projection is computed this long after the
# last change (polygon edits come in bursts).
CROP_PROJECTION_DELAY_MS = 250
# (0.18.0) The selected box's dots in 3D: their size on screen, and how
# often a drag tells the panel (the dots and wireframe follow at once).
HANDLE_PIXELS = 14
HANDLE_APPLY_S = 0.15
# A loaded region uses the pyramid levels in which it is at least this many
# voxels long (along its longest axis). napari never needs a coarser level
# for a view larger than this many pixels, and each level used makes the
# region's edges round outward to its voxel grid.
REGION_MIN_SIZE = 128
VIEWS_2D = ("XY", "XZ", "YZ")
COLUMNS = (("XY", "XZ"), ("YZ", "3D"))     # the grid, column by column
# One colour per plane, used for the view titles and for the crosshair line
# that marks where that plane cuts the other views.
PLANE_COLORS = {"XY": "#4da6ff", "XZ": "#5fd35f", "YZ": "#ff6b6b",
                "3D": "#dddddd"}
BOX_COLORS = ["#ffd700", "#00e5ff", "#ff40ff", "#ff8c00", "#7fff00",
              "#ff5050"]
DEFAULT_CHANNEL_COLORS = ["FF00FF", "00FF00", "00FFFF", "FFFF00", "FFFFFF"]
# Layer settings copied from the XY view to the other views.
DISPLAY_ATTRS = ("contrast_limits", "gamma", "colormap", "visible",
                 "opacity", "blending")
# Box lines in the 2D views, in screen pixels: solid for the box as it will
# be cropped, dashed for an outline (the selected box twice as wide). A box
# that the current slice does not cut is drawn with this opacity.
LINE_WIDTH = 2.0
DASH, GAP = 8.0, 5.0
MARK = 5.0      # half the size of a crossing mark (screen pixels)
# (0.20.0) Crossing marks of one view closer together than this (centre to
# centre, screen pixels: about three marks' widths) are thinned out, so
# that the Autocropper's many outlines per slice don't crowd the other
# views; zooming in brings the hidden ones back.
MARK_SPACING = 6 * MARK
FADED_ALPHA = 0.3
# (0.16.0) The layers the views make for themselves (region lines,
# crosshair, preview, shading) carry this in their metadata; the window
# hides them in napari's layer list (crop_gui._hide_tool_layers).
HELPER = {"volume_cropper": "helper"}


def is_helper_layer(layer) -> bool:
    """True for a layer the views make for themselves (not an image
    channel, not a layer one draws on)."""
    try:
        return layer.metadata.get("volume_cropper") == "helper"
    except AttributeError:
        return False


# (0.16.0) Seeds of the Autocropper: a cross of this size (screen pixels)
# in the 2D views, at every seed whatever the slice shown. (0.17.0) White
# with a dark edge until a fit gives it a region, then in that region's
# colour (OrthoViews.seed_style); the first region colour is gold, which
# the old yellow seeds looked like.
SEED_COLOR = "#ffffff"
SEED_EDGE = "#202020"
SEED_SIZE = 14.0
# (0.18.0) The Autocropper's own threshold: a dotted line of this colour
# round what lies above it, in the 2D views (and its line in the Levels
# histogram); one fixed colour (orange, the user's choice). Dotted, with
# short dashes, because the fourth region colour is an orange too, and
# regions' polygons are dashed.
THRESHOLD_COLOR = "#ff9a2e"
THRESHOLD_DASH, THRESHOLD_GAP = 4.0, 3.0      # screen pixels
# (0.18.0) The Autocropper's gap lines (drawn between touching objects):
# white, dash-dot (on / off lengths in screen pixels), on a dark edge as
# the seeds have (the user's choice); faded when drawn on another slice
# or timepoint. A click this close to one (screen pixels) deletes it.
GAP_COLOR = "#ffffff"
GAP_PATTERN = (9.0, 3.0, 2.0, 3.0)
GAP_EDGE_WIDTH = 4.0
GAP_PICK = 8.0
# (0.18.0, the user's wish) The shape a region's polygons make (what its
# crop keeps), shown in the 3D view as a see-through surface in the
# region's colour: for no region, the selected one (at first) or all.
OUTLINE_3D_MODES = (("off", "off"), ("selected region", "selected"),
                    ("all regions", "all"))
OUTLINE_3D_OPACITY = 0.35
OUTLINE_3D_MAX_VOXELS = 4_000_000      # beyond: marching cubes in steps of 2
# A click that moves the mouse further than this (screen pixels) is a pan,
# not a seed.
CLICK_SLOP = 4.0
# Optional shading of the selected box in the 2D views (the user can change
# both colours and the opacity): the voxels that will be kept, and the part
# of the box that will be set to 0 (outside the outlines).
# Shading: orange (kept) and sky blue (set to 0), from Okabe and Ito's
# colour-blind-safe palette: unlike green, magenta, red, cyan and the
# other usual LUT colours, and distinct also for red-green colour-blind
# viewers (CIEDE2000 51 normal, 55 deuteranopia, 50 protanopia).
SHADE_KEPT, SHADE_ZEROED, SHADE_OPACITY = "#ff8c00", "#56b4e9", 0.35
# The slice key of a box's rectangle shape (it has no polygon).
RECT = "rectangle"
# How the 3D view shows the voxels that the selected box's outlines will
# black out (inside the box, outside all its outlines).
MASK_MODES = ("show", "dim", "grey", "dimgrey", "hide")
# Opacity of the blacked-out part ("hide" shows nothing). Grey layers of
# all channels add up, so 3 channels at 0.35 are about white at most.
MASK_OPACITY = {"dim": 0.25, "grey": 0.35, "dimgrey": 0.12}
MASK_GREY = ("grey", "dimgrey")               # grey instead of channel colour


# ---------------------------------------------------------------------------
# Small helpers (napari 0.9 moved some viewer attributes)
# ---------------------------------------------------------------------------

def camera(viewer):
    """The viewer's camera (napari 0.9 moved it to viewer.scene.camera)."""
    scene = getattr(viewer, "scene", None)
    return scene.camera if scene is not None else viewer.camera


def grid_of(viewer):
    """The viewer's grid-mode settings (napari 0.9: viewer.canvas.grid)."""
    canvas = getattr(viewer, "canvas", None)
    grid = getattr(canvas, "grid", None)
    return grid if grid is not None else viewer.grid


def scale_bar(viewer):
    """The scale bar overlay (napari 0.9: viewer.canvas.overlays.scale_bar)."""
    canvas = getattr(viewer, "canvas", None)
    overlays = getattr(canvas, "overlays", None)
    return overlays.scale_bar if overlays is not None else viewer.scale_bar


def rgba(color: str, alpha: float = 1.0) -> list[float]:
    """'#ffd700' -> [1.0, 0.84, 0.0, alpha] (red, green, blue, opacity)."""
    color = color.lstrip("#")
    return [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha]


def modifier_names(modifiers) -> set:
    """The modifier keys held during a mouse event, as names ("Shift",
    "Control", "Alt", "Meta"); napari passes vispy Key objects."""
    return {getattr(key, "name", str(key)) for key in (modifiers or ())}


def set_visible(layer, visible: bool):
    """Show or hide a layer, only if that changes anything. napari's own
    setter re-reads (refreshes) the layer and redraws the view even when
    the value stays the same: ~15 ms a time, which made dragging a region
    in 3D lag (0.18.0, measured on the synthetic demo)."""
    visible = bool(visible)
    if layer.visible != visible:
        layer.visible = visible


def colormap(color: str) -> Colormap:
    """A black-to-colour lookup table, like Fiji's 'Green' or 'Magenta' LUTs."""
    return Colormap([[0, 0, 0, 1], rgba(color)],
                    name=f"#{color.lstrip('#').lower()}")


def _prime_opengl_info():
    """Ask OpenGL about its capabilities once, before the extra views exist.

    napari asks (and caches the answer) when the first image layer is added,
    using whichever canvas is "current". A newly created canvas has no
    OpenGL context until it is first shown, so asking through it fails
    ("GLError ... invalid operation"). Asking through a temporary hidden
    canvas, as napari itself does, fills the cache safely.
    """
    from napari._vispy.utils.gl import get_gl_extensions, get_max_texture_sizes
    from vispy.app import Canvas

    canvas = Canvas(show=False)
    try:
        canvas.set_current()
        get_gl_extensions()
        get_max_texture_sizes()
    finally:
        canvas.close()


def layer_units(image: vc.OmeZarrImage, axes: list[str]) -> list[str]:
    """napari units for these axes. All layers must use the same units,
    otherwise napari rescales them against each other."""
    return [image.unit(axis) or "pixel" for axis in axes]


# ---------------------------------------------------------------------------
# Geometry (pure functions, tested in test_ortho_views.py)
# ---------------------------------------------------------------------------

def view_orders(image: vc.OmeZarrImage) -> dict[str, list[str]]:
    """World axis order of each view; the last two axes are the ones shown.

    Other axes (e.g. t) come first and are shared by all views.
    """
    spatial = [a["name"] for a in image.axes if a.get("type") == "space"]
    if len(spatial) != 3:
        raise ValueError("Orthogonal views need an image with three spatial "
                         f"axes; this one has {spatial}.")
    lead = [a for a in image.range_axes if a not in spatial]
    z, y, x = spatial
    return {"XY": lead + [z, y, x], "XZ": lead + [y, z, x],
            "YZ": lead + [x, y, z], "3D": lead + [z, y, x]}


def pyramid_levels(n_levels: int, level: int | None = None,
                   exact: bool = False) -> list[int]:
    """Which pyramid levels a 2D view may use.

    None: all (napari picks the level from the zoom).
    level N, exact=False: N and coarser, so zooming in never loads finer
        data than N, but zooming out can still switch to coarser levels.
    level N, exact=True: only N, at every zoom.

    A one-level pyramid still loads only the visible region. (napari's own
    layer.locked_data_level is not used: setting it loads the whole plane at
    that level at once, which for a fine level can be gigabytes.)
    """
    if level is None:
        return list(range(n_levels))
    return [level] if exact else list(range(level, n_levels))


# ---------------------------------------------------------------------------
# The loaded region, and how much the views read
# ---------------------------------------------------------------------------

@dataclass
class LoadedRegion:
    """The part of an image that the views read and show.

    ranges: {axis: (low, high)} for every axis except channels, in the
        axis's unit (µm, or the time unit), as loaded: rounded outward to
        the voxels of the coarsest level used, so that every level is cut at
        whole voxels and napari places the levels exactly on top of each
        other (as for a crop).
    starts, stops: per level used, the voxel range [start, stop) along
        every axis (the channel axis whole).
    whole: the whole image, nothing cut off.
    """

    ranges: dict
    starts: list
    stops: list
    whole: bool = True

    @property
    def n_levels(self) -> int:
        """The levels used are 0 to n_levels - 1."""
        return len(self.starts)

    def slices(self, level: int) -> tuple:
        """The loaded part of a level, to index its array with."""
        return tuple(slice(a, b)
                     for a, b in zip(self.starts[level], self.stops[level]))

    def shape(self, level: int) -> tuple:
        return tuple(b - a
                     for a, b in zip(self.starts[level], self.stops[level]))

    def translation(self, image: vc.OmeZarrImage, level: int) -> list:
        """Position of the loaded part's first voxel (its centre), per axis."""
        lv = image.levels[level]
        return [t + a * s for t, s, a
                in zip(lv.translation, lv.scale, self.starts[level])]


def loaded_region(image: vc.OmeZarrImage, ranges: dict | None = None,
                  min_size: int = REGION_MIN_SIZE) -> LoadedRegion:
    """The region to load: ranges {axis: (low, high)} in the axes' units;
    axes left out (or None for all) stay whole.

    The ranges are rounded outward exactly like a crop box (see
    volume_cropper.plan_crop), to the coarsest level in which the region is
    at least min_size voxels long. A region that covers the whole image is
    the whole image, with all levels.
    """
    full = {axis: image.extent(axis) for axis in image.range_axes}
    plan = vc.plan_crop(image, vc.CropBox("loaded", {**full, **(ranges or {})}),
                        min_size=min_size)
    shape0 = image.levels[0].array.shape
    if list(plan.starts[0]) == [0] * len(shape0) \
            and tuple(plan.stops[0]) == tuple(shape0):
        return LoadedRegion(full, [[0] * len(shape0) for _ in image.levels],
                            [list(lv.array.shape) for lv in image.levels])
    return LoadedRegion(dict(plan.snapped_ranges), plan.starts, plan.stops,
                        whole=False)


def _spatial(image: vc.OmeZarrImage) -> list[int]:
    """Indices of the spatial axes (z, y, x)."""
    return [i for i, a in enumerate(image.axes) if a.get("type") == "space"]


def _bounds(image: vc.OmeZarrImage, level: int,
            region: LoadedRegion | None) -> tuple[list, list]:
    """Voxel range [start, stop) per axis: the loaded part (or the whole)."""
    if region is None:
        return [0] * len(image.axes), list(image.levels[level].array.shape)
    return list(region.starts[level]), list(region.stops[level])


def _n_chunks(start: int, stop: int, size: int) -> int:
    """How many chunks of this size the voxels [start, stop) touch."""
    return 0 if stop <= start else -(-stop // size) - start // size


def _n_levels(image: vc.OmeZarrImage, region: LoadedRegion | None) -> int:
    return len(image.levels) if region is None else region.n_levels


def _channels(image: vc.OmeZarrImage, n_channels: int | None) -> int:
    return image.n_channels if n_channels is None else n_channels


def auto_3d_level(image: vc.OmeZarrImage, max_bytes: float = AUTO_3D_BYTES,
                  max_side: int = AUTO_3D_SIDE,
                  region: LoadedRegion | None = None,
                  n_channels: int | None = None) -> int:
    """Finest level that fits max_bytes (one timepoint of n_channels
    channels, default all) with at most max_side voxels along every spatial
    axis. region: the loaded region (default: the whole image).

    For a small crop this is a fine level; for a whole embryo a coarse one.
    """
    for k in range(_n_levels(image, region)):
        start, stop = _bounds(image, k, region)
        if volume_bytes(image, k, region, n_channels) <= max_bytes and \
                max(stop[i] - start[i] for i in _spatial(image)) <= max_side:
            return k
    return _n_levels(image, region) - 1


def preview_level(image: vc.OmeZarrImage, max_size: int = PREVIEW_SIZE,
                  max_voxels: float = PREVIEW_VOXELS,
                  region: LoadedRegion | None = None) -> int:
    """Finest level whose longest spatial side is at most max_size, with at
    most max_voxels voxels (of the loaded region, default the whole image)."""
    for k in range(_n_levels(image, region)):
        start, stop = _bounds(image, k, region)
        sides = [stop[i] - start[i] for i in _spatial(image)]
        if max(sides) <= max_size and np.prod(sides) <= max_voxels:
            return k
    return _n_levels(image, region) - 1


def plane_read_bytes(image: vc.OmeZarrImage, level: int,
                     plane: tuple[str, str],
                     region: LoadedRegion | None = None,
                     n_channels: int | None = None) -> int:
    """Bytes read to show one whole plane of a level (of the loaded region,
    default the whole image), for n_channels channels (default all).

    Chunks are read whole, so a plane costs every chunk it crosses: for
    80-voxel-deep chunks, about 80 times the plane itself.
    """
    array = image.levels[level].array
    start, stop = _bounds(image, level, region)
    n_chunks = 1
    for axis in plane:
        i = image.axis_names.index(axis)
        n_chunks *= _n_chunks(start[i], stop[i], array.chunks[i])
    chunk_bytes = int(np.prod(array.chunks)) * array.dtype.itemsize
    return n_chunks * chunk_bytes * _channels(image, n_channels)


def projection_slab(point: float, low: float, high: float):
    """Slice position and margins for projecting through [low, high].

    napari projects a "thick slice" from (point - margin_left) to
    (point + margin_right). The point must lie inside the slab, so a point
    outside it is moved to the slab's centre.
    Returns (point, margin_left, margin_right), all in µm.
    """
    if not low <= point <= high:
        point = (low + high) / 2
    return point, point - low, high - point


def projection_read_bytes(image: vc.OmeZarrImage, level: int,
                          plane: tuple[str, str], depth_axis: str,
                          depth_range: tuple[float, float] | None = None,
                          region: LoadedRegion | None = None,
                          n_channels: int | None = None) -> int:
    """Bytes read to project a whole plane of a level through a slab.

    depth_range: the slab in µm along depth_axis (None = the whole depth).
    region: the loaded region (default: the whole image); n_channels: the
    channels read (default all). Chunks are read whole.
    """
    array = image.levels[level].array
    names = image.axis_names
    start, stop = _bounds(image, level, region)
    n_chunks = 1
    for axis in plane:
        i = names.index(axis)
        n_chunks *= _n_chunks(start[i], stop[i], array.chunks[i])
    i = names.index(depth_axis)
    first, end = start[i], stop[i]
    if depth_range is not None:
        lv = image.levels[level]
        low, high = vc.physical_to_index_range(*depth_range, lv.scale[i],
                                               lv.translation[i])
        first, end = max(first, low), min(end, high)
    depth_chunks = _n_chunks(first, end, array.chunks[i])
    chunk_bytes = int(np.prod(array.chunks)) * array.dtype.itemsize
    return (n_chunks * depth_chunks * chunk_bytes
            * _channels(image, n_channels))


def volume_bytes(image: vc.OmeZarrImage, level: int,
                 region: LoadedRegion | None = None,
                 n_channels: int | None = None) -> int:
    """Bytes of one timepoint of a level (the 3D view): of the loaded region
    (default the whole image), n_channels channels (default all)."""
    array = image.levels[level].array
    start, stop = _bounds(image, level, region)
    voxels = int(np.prod([stop[i] - start[i] for i in _spatial(image)]))
    return voxels * array.dtype.itemsize * _channels(image, n_channels)


def level_for_pixel(image: vc.OmeZarrImage, levels: list[int],
                    axes: list[str], pixel: float) -> int:
    """The level napari shows in a 2D view whose screen pixels are `pixel`
    µm wide: the coarsest of `levels` whose voxels are still smaller than a
    pixel along each of the view's axes (`axes`). Axes whose voxels are
    already larger than a pixel on the first level don't count.

    The same rule as napari 0.9.2's compute_multiscale_level, in physical
    units (test_ortho_views.py compares the two). It also explains why the
    XZ and YZ views often use a finer level than XY: their z voxels are
    larger than their x and y voxels.
    """
    first = image.levels[levels[0]].scale
    # The axes that count: finer than a pixel and coarser on some level.
    active = [i for i in (image.axis_names.index(a) for a in axes)
              if first[i] < pixel
              and any(image.levels[k].scale[i] > first[i] for k in levels)]
    chosen = levels[0]          # also when no axis counts (zoomed far in)
    if active:
        for k in levels:
            if all(image.levels[k].scale[i] < pixel for i in active):
                chosen = k
    return chosen


def fitted_levels(image: vc.OmeZarrImage, region: LoadedRegion,
                  canvas_sizes: dict, level_2d: int | None = None,
                  exact: bool = False) -> dict:
    """The level each 2D view shows right after opening, when the loaded
    region is fitted into the views: {"XY": level, "XZ": ..., "YZ": ...}.

    canvas_sizes: {view name: (height, width)} in screen pixels. The views
    share one zoom, the largest that fits the region into every 2D view
    with napari's margin of 5 %.
    """
    orders = view_orders(image)
    zooms = []
    for name in VIEWS_2D:
        extents = [region.ranges[a][1] - region.ranges[a][0]
                   for a in orders[name][-2:]]
        zooms.append(0.95 * min(size / extent for size, extent
                                in zip(canvas_sizes[name], extents)))
    pixel = 1 / min(zooms)                     # µm per screen pixel
    n = region.n_levels
    cap = None if level_2d is None else min(level_2d, n - 1)
    levels = pyramid_levels(n, cap, exact)
    return {name: level_for_pixel(image, levels, orders[name][-2:], pixel)
            for name in VIEWS_2D}


def open_read_bytes(image: vc.OmeZarrImage, region: LoadedRegion,
                    n_channels: int, canvas_sizes: dict,
                    level_2d: int | None = None, show_3d: bool = False,
                    level_3d: int | None = None) -> int:
    """About how many bytes opening reads (uncompressed; the files are
    usually smaller): the three 2D planes at the levels napari will pick,
    the 3D view's volume if it is shown, and the contrast preview. Moving
    around or zooming in reads more."""
    orders = view_orders(image)
    levels = fitted_levels(image, region, canvas_sizes, level_2d)
    total = sum(plane_read_bytes(image, levels[name],
                                 tuple(orders[name][-2:]), region, n_channels)
                for name in VIEWS_2D)
    if show_3d:
        k = (auto_3d_level(image, region=region, n_channels=n_channels)
             if level_3d is None else min(level_3d, region.n_levels - 1))
        total += volume_bytes(image, k, region, n_channels)
    preview = preview_level(image, region=region)
    return total + volume_bytes(image, preview, region, n_channels)


def rectangle(ranges: dict, plane: tuple[str, str]) -> np.ndarray:
    """Corners of a box's rectangle in a view's plane (vertical, horizontal)."""
    (v0, v1), (h0, h1) = ranges[plane[0]], ranges[plane[1]]
    return np.array([[v0, h0], [v0, h1], [v1, h1], [v1, h0]], dtype=float)


def rectangle_ranges(vertices, plane: tuple[str, str],
                     extents: dict) -> dict:
    """Ranges of a drawn shape (its bounding rectangle, clipped to extents)."""
    vertices = np.asarray(vertices)[:, -2:]
    ranges = {}
    for k, axis in enumerate(plane):
        low, high = extents[axis]
        ranges[axis] = (max(float(vertices[:, k].min()), low),
                        min(float(vertices[:, k].max()), high))
    return ranges


def keys_in_plane(box: vc.CropBox, plane: tuple[str, str]) -> dict:
    """A box's polygons in a view, {slice: points}, with the points as
    (vertical, horizontal) of that view; polygons stored the other way
    round, e.g. for ("x", "y"), are swapped. {} if the view has none."""
    if tuple(plane) in box.outlines:
        return box.outlines[tuple(plane)]
    swapped = (plane[1], plane[0])
    if swapped in box.outlines:
        return {depth: [(b, a) for a, b in points]
                for depth, points in box.outlines[swapped].items()}
    return {}


def reaches_through(box: vc.CropBox, plane: tuple[str, str]) -> bool:
    """Whether a view's polygon reaches through the whole box: the box's
    only polygon, or one without a slice (boxes from before 0.7.0)."""
    return any(set(p) == set(plane) for p in vc._through_box(box.outlines))


def polygon_on_slice(box: vc.CropBox, plane: tuple[str, str],
                     depth: float | None, tolerance: float = 0.0):
    """The polygon of a box that a view shows (and lets you edit) on the
    slice `depth`, as (slice key, points): a polygon that reaches through
    the box (shown on every slice), or the view's polygon on this slice
    (within `tolerance` µm). None if there is neither."""
    keys = keys_in_plane(box, plane)
    if len(keys) == 1 and reaches_through(box, plane):
        return next(iter(keys.items()))
    for key, points in keys.items():
        if depth is not None and isinstance(key, float) \
                and abs(key - depth) <= tolerance:
            return key, points
    return None


def polygon_names(box: vc.CropBox, prefix) -> dict:
    """Generic names of a box's polygons, {(plane as stored, key): name}:
    per view, "xy1", "xy2", ... for polygons on slices (and one through the
    box) in slice order, and "xyP1", ... for silhouettes (drawn on a
    projection) in depth order. prefix(plane) gives the view's name, e.g.
    "xy"."""
    names = {}
    for plane, keys in box.outlines.items():
        slices = sorted((k for k in keys if not vc.is_silhouette(k)),
                        key=vc._slice_order)
        ranges = sorted((k for k in keys if vc.is_silhouette(k)),
                        key=vc._slice_order)
        for n, key in enumerate(slices, start=1):
            names[(plane, key)] = f"{prefix(plane)}{n}"
        for n, key in enumerate(ranges, start=1):
            names[(plane, key)] = f"{prefix(plane)}P{n}"
    return names


def silhouettes_in_view(box: vc.CropBox, plane: tuple[str, str],
                        depth_range) -> list:
    """A box's silhouettes in a view (polygons drawn on a projection) whose
    depth range overlaps depth_range, as [(key, points)], points as
    (vertical, horizontal) of the view."""
    low, high = depth_range
    return [(key, points) for key, points in keys_in_plane(box, plane).items()
            if vc.is_silhouette(key) and key[0] <= high and key[1] >= low]


def same_range(key, depth_range, tolerance: float) -> bool:
    """Whether a silhouette's depth range is this range (within
    tolerance)."""
    return vc.is_silhouette(key) and all(
        abs(a - b) <= tolerance for a, b in zip(key, depth_range))


# The views draw the shape through a box's polygons from its function on a
# grid of this many points along the polygons' longest extent: the crop's
# own shape (volume_cropper._shape_field), drawn more coarsely than the
# crop does (FIELD_POINTS), so that the views follow the drawing quickly.
VIEW_DETAIL = 48
# Grid points along the longer side of a box's rectangle, for its dashed
# outline and its shading on a slice.
SLICE_POINTS = 512
# The crop preview's grid inside a box follows the level shown (one point
# per voxel of that level), but at most this many points along the box's
# longer side, so that it keeps up with the slider.
PREVIEW_POINTS = 1024


def slice_mask(box: vc.CropBox, spatial, pixel: float, plane, position,
               ranges: dict, points: int = SLICE_POINTS):
    """What a box's outlines keep on one slice of a view, computed with
    the crop's own rule (volume_cropper.outline_mask, margin included).

    spatial: the image's spatial axes; pixel: its finest voxel size (µm,
    the unit of the margin); plane: the view's (vertical, horizontal) axes;
    position: the slice (µm along the third axis); ranges: the box as it
    will be cropped. Returns (keep, origin, step): True = kept, on a grid
    over the box's rectangle with `points` points (or one per voxel, if
    fewer) along its longer side; the first point's centre (µm); and the
    spacing (µm). Remembered, as it is asked for on every slider move.
    """
    return _slice_mask(vc._frozen(box.outlines), float(box.margin * pixel),
                       tuple(spatial), tuple(plane), float(position),
                       tuple(tuple(ranges[a]) for a in plane), float(pixel),
                       points)


@lru_cache(maxsize=64)
def _slice_mask(frozen, margin, spatial, plane, position, ranges, pixel,
                points):
    outlines = {p: dict(keys) for p, keys in frozen}
    v, h = plane
    d = next(axis for axis in spatial if axis not in plane)
    (v0, v1), (h0, h1) = ranges
    step = max(max(v1 - v0, h1 - h0) / points, pixel)
    shape = (max(int(np.ceil((v1 - v0) / step)), 1),
             max(int(np.ceil((h1 - h0) / step)), 1))
    origin = (v0 + step / 2, h0 + step / 2)       # first point's centre
    # The grid's spacing along the third axis matters for a margin only
    # (the distances are measured in 3D).
    level = vc.Level("", None, [step, step, step],
                     [origin[0], origin[1], position])
    keep = vc.outline_mask(level, [v, h, d], [0, 0, 0], shape + (1,),
                           outlines, spatial=list(spatial), margin=margin,
                           detail=VIEW_DETAIL)
    keep = (np.ones(shape, dtype=bool) if keep is None
            else np.array(np.broadcast_to(keep, shape + (1,))[..., 0]))
    keep.flags.writeable = False              # shared by the cache
    return keep, origin, step


def slice_outline(box: vc.CropBox, spatial, pixel: float, plane, position,
                  ranges: dict) -> list:
    """The edge(s) of slice_mask, as closed lines of (vertical, horizontal)
    points in µm, for drawing; [] if nothing is kept on this slice."""
    keep, origin, step = slice_mask(box, spatial, pixel, plane, position,
                                    ranges)
    # find_contours gives (row, column) in grid steps; the padding closes
    # outlines that touch the grid's edge
    return [np.asarray(origin) + (contour - 1) * step
            for contour in find_contours(np.pad(keep, 1).astype(float), 0.5)]


def slice_in_crop(ranges: dict, plane, depth: str, cursor: dict,
                  slab=None) -> bool:
    """Whether a box's crop holds any of a view's slice: the slice (or,
    on a projection, the slab (low, high) along the depth axis) and the
    cursor on every other axis (e.g. the timepoint) lie inside the box's
    ranges (as cropped: outer voxel edges, µm)."""
    for axis, (low, high) in ranges.items():
        if axis in plane:
            continue
        if axis == depth and slab is not None:
            if not (slab[0] < high and slab[1] > low):
                return False
        elif axis in cursor and not low <= cursor[axis] <= high:
            return False
    return True


def preview_rectangles(ranges: dict, plane, extents, here: bool) -> tuple:
    """The black rectangles of the crop preview in one 2D view, as
    ((top, left), (bottom, right)) corners in µm: the four bands around the
    box, or one rectangle over the whole view if the crop holds nothing of
    this slice (here False). They reach one image size beyond the image
    (extents: the view's (vertical, horizontal) ranges), so the edges of
    the image never show. The box's edges are exactly the crop's, as the
    ranges are its outer voxel edges."""
    v, h = plane
    (e0, e1), (f0, f1) = extents
    top, bottom = e0 - (e1 - e0), e1 + (e1 - e0)
    left, right = f0 - (f1 - f0), f1 + (f1 - f0)
    if not here:
        return (((top, left), (bottom, right)),)
    (a0, a1), (b0, b1) = ranges[v], ranges[h]
    return (((top, left), (a0, right)), ((a1, left), (bottom, right)),
            ((a0, left), (a1, b0)), ((a0, b1), (a1, right)))


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """"#ff8c00" -> (255, 140, 0)."""
    color = color.lstrip("#")
    return tuple(int(color[k:k + 2], 16) for k in (0, 2, 4))


def outline_in_plane(box: vc.CropBox, plane: tuple[str, str],
                     depth: float | None = None, tolerance: float = 0.0):
    """The points of polygon_on_slice, or None."""
    found = polygon_on_slice(box, plane, depth, tolerance)
    return None if found is None else found[1]


def polyline_segments(points, closed: bool = True) -> np.ndarray:
    """The straight pieces joining consecutive points, as an (N, 2, D)
    array of [start, end]. closed: also join the last point to the first."""
    points = np.asarray(points, dtype=float)
    if closed:
        return np.stack([points, np.roll(points, -1, axis=0)], axis=1)
    return np.stack([points[:-1], points[1:]], axis=1)


def thin_points(points, spacing: float) -> np.ndarray:
    """(0.20.0) Which points to keep so that no two kept ones are closer
    than spacing: the indices of the kept points, in order.

    Greedy: the points are taken in order of their first, then second
    coordinate (along the object's edge for marks of polygons drawn slice
    after slice), and a point is kept if it is at least spacing from every
    point kept so far. So points that are far enough apart are all kept,
    and a crowded row keeps about every spacing-th length of it.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    order = np.lexsort((points[:, 1], points[:, 0]))
    kept: list[int] = []
    for i in order:
        if not kept or np.min(np.linalg.norm(points[kept] - points[i],
                                             axis=1)) >= spacing:
            kept.append(int(i))
    return np.sort(np.asarray(kept, dtype=int))


def dash_segments(points, dash: float, gap: float, closed: bool = True,
                  max_dashes: int = 20000) -> np.ndarray:
    """A dashed line along a polygon, as (N, 2, D) [start, end] pieces.

    dash and gap are lengths in the points' units. The pattern runs on
    around corners, and a dash that reaches a corner bends with it (it is
    split there), so the dashes follow the outline exactly. Very long
    outlines get longer dashes, at most max_dashes of them.
    """
    points = np.asarray(points, dtype=float)
    if closed:
        points = np.vstack([points, points[:1]])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = np.vstack([points[:1], points[1:][steps > 0]])  # no repeats
    if len(points) < 2:
        return np.zeros((0, 2, points.shape[1]))
    # Distance along the outline at each corner, and its total length.
    along = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = along[-1]
    period = dash + gap
    if total / period > max_dashes:
        stretch = total / (period * max_dashes)
        dash, period = dash * stretch, period * stretch
    starts = np.arange(0.0, total, period)
    # Cut the outline at every dash end and every corner; keep the pieces
    # whose middle lies within a dash.
    cuts = np.unique(np.concatenate([starts, np.minimum(starts + dash, total),
                                     along]))
    middles = (cuts[:-1] + cuts[1:]) / 2
    keep = middles % period < dash
    first, last = cuts[:-1][keep], cuts[1:][keep]

    def position(distance):        # point at a distance along the outline
        return np.column_stack([np.interp(distance, along, points[:, k])
                                for k in range(points.shape[1])])

    return np.stack([position(first), position(last)], axis=1)


def level_for_voxel(image: vc.OmeZarrImage, size: float,
                    n_levels: int | None = None) -> int:
    """(0.18.0) The finest level whose voxels are at least `size` µm (along
    the last axis, as "What to load" lists them), or the coarsest of the
    first n_levels if none is: a default resolution that gives images from
    other set-ups the same detail (the user's choice)."""
    n = len(image.levels) if n_levels is None else max(int(n_levels), 1)
    for k in range(n):
        if image.levels[k].scale[-1] >= size * (1 - 1e-6):
            return k
    return n - 1


def pattern_segments(points, pattern, max_pieces: int = 20000) -> np.ndarray:
    """(0.18.0) A line along an open polyline in a repeating pattern of
    lengths, on and off in turn (pattern[0] on, pattern[1] off, ...; e.g.
    dash-dot), as (N, 2, D) [start, end] pieces like dash_segments.
    Very long lines get a longer pattern, at most max_pieces periods."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return np.zeros((0, 2, 2))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = np.vstack([points[:1], points[1:][steps > 0]])  # no repeats
    if len(points) < 2:
        return np.zeros((0, 2, 2))
    along = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total = along[-1]
    pattern = np.asarray(pattern, dtype=float)
    period = float(pattern.sum())
    if total / period > max_pieces:
        pattern = pattern * total / (period * max_pieces)
        period = float(pattern.sum())
    edges = np.concatenate([[0.0], np.cumsum(pattern)])  # in one period
    starts = np.arange(0.0, total, period)
    cuts = np.concatenate([(starts[:, None] + edges[None, :]).ravel(), along])
    cuts = np.unique(cuts[(cuts >= 0) & (cuts <= total)])
    middles = (cuts[:-1] + cuts[1:]) / 2
    which = np.searchsorted(edges, middles % period, side="right") - 1
    keep = which % 2 == 0
    first, last = cuts[:-1][keep], cuts[1:][keep]

    def position(distance):
        return np.column_stack([np.interp(distance, along, points[:, k])
                                for k in range(points.shape[1])])

    return np.stack([position(first), position(last)], axis=1)


def segment_distance(point, line) -> float:
    """(0.18.0) The shortest distance from a point to an open polyline."""
    point = np.asarray(point, dtype=float)
    line = np.asarray(line, dtype=float).reshape(-1, 2)
    if len(line) == 1:
        return float(np.linalg.norm(point - line[0]))
    a, b = line[:-1], line[1:]
    ab = b - a
    lengths = (ab ** 2).sum(axis=1)
    t = np.clip(((point - a) * ab).sum(axis=1) / np.maximum(lengths, 1e-12),
                0, 1)
    nearest = a + t[:, None] * ab
    return float(np.linalg.norm(nearest - point, axis=1).min())


def masked_region(axes, scale, translation, shape, ranges: dict,
                  outlines: dict, spatial=None,
                  margin: float = 0.0) -> np.ndarray:
    """The voxels that a box's outlines will black out (True), on one level.

    axes, scale, translation, shape: the level's spatial axes, in any order.
    ranges: the box as it will be cropped, {axis: (low, high)} in µm.
    outlines: as in CropBox; spatial: the image's spatial axes (default:
    axes); margin: the box's margin in µm. A voxel is blacked out when its
    centre lies inside the box but outside what the outlines keep (the
    crop masks each level by its own voxel centres in the same way; the
    shape through polygons on slices is drawn with VIEW_DETAIL).
    """
    region = np.zeros(shape, dtype=bool)
    box = _box_slices(axes, scale, translation, shape, ranges)
    if not outlines or box is None:
        return region
    first = [s.start for s in box]
    size = tuple(s.stop - s.start for s in box)
    level = vc.Level("", None, list(scale), list(translation))
    keep = vc.outline_mask(level, list(axes), first, size, outlines,
                           spatial=list(spatial or axes), margin=margin,
                           detail=VIEW_DETAIL)
    region[tuple(box)] = ~np.broadcast_to(keep, size)
    return region


def outside_region(axes, scale, translation, shape,
                   ranges: dict) -> np.ndarray:
    """The voxels of one level whose centres lie outside a box (True).

    With "Clip 3D to selected region" the 3D view shows these as 0. napari
    clips the volume exactly at the box's faces, but it draws a voxel that
    straddles a face partly inside, and it blends neighbouring voxels when
    drawing (interpolation); without this, a thin halo of the data just
    outside the box showed along its faces (user report, 2026-11-02).
    Arguments as for masked_region.
    """
    outside = np.ones(shape, dtype=bool)
    box = _box_slices(axes, scale, translation, shape, ranges)
    if box is not None:
        outside[tuple(box)] = False
    return outside


def _box_slices(axes, scale, translation, shape,
                ranges: dict) -> list[slice] | None:
    """The voxels of one level whose centres lie inside a box, as one slice
    per axis (None if there are none)."""
    box = []
    for i, axis in enumerate(axes):
        low, high = ranges[axis]
        start = max(int(np.ceil((low - translation[i]) / scale[i] - 1e-9)), 0)
        stop = min(int(np.floor((high - translation[i]) / scale[i] + 1e-9)) + 1,
                   shape[i])
        if stop <= start:
            return None
        box.append(slice(start, stop))
    return box


def clipping_planes(low, high) -> list[dict]:
    """Six planes that keep only the inside of a box, for a 3D layer.

    low/high: the box edges per displayed axis (z, y, x), in µm. napari keeps
    the side that a plane's normal points to.

    napari's documentation says plane positions are in the layer's data
    (voxel) coordinates, but napari 0.9.2 reads them in world coordinates
    (µm, including the layer's offset). This was measured against a
    rendered sub-volume; test_napari_clipping.py repeats the check, so run
    it after upgrading napari.
    """
    planes = []
    for j in range(3):
        for edge, sign in ((low[j], 1), (high[j], -1)):
            position = [0.0, 0.0, 0.0]
            position[j] = float(edge)
            normal = [0.0, 0.0, 0.0]
            normal[j] = float(sign)
            planes.append({"position": tuple(position),
                           "normal": tuple(normal), "enabled": True})
    return planes


def channel_contrast(image: vc.OmeZarrImage, k: int,
                     region: LoadedRegion | None = None) -> tuple[float, float]:
    """Starting contrast of channel k: percentiles of a small pyramid level
    of the loaded region (default: the whole image), first timepoint."""
    region = region or loaded_region(image)
    level = preview_level(image, region=region)
    key = list(region.slices(level))
    for i, axis in enumerate(image.axes):
        # The first timepoint only: a time-lapse can have thousands.
        if axis.get("type") not in ("space", "channel"):
            key[i] = slice(key[i].start, key[i].start + 1)
    if image.channel_axis is not None:
        key[image.channel_axis] = k
    values = image.levels[level].array[tuple(key)]
    low, high = np.percentile(values, [0.5, 99.9])
    return float(low), float(max(high, low + 1))


def initial_display(image: vc.OmeZarrImage,
                    region: LoadedRegion | None = None,
                    visible: list[int] | None = None) -> list[dict]:
    """Starting display settings per channel: colour from the OME metadata,
    contrast from a small pyramid level of the loaded region.

    visible: the channels shown at first (None = all). Only their contrast
    is measured now; the others are marked "measured": False and get theirs
    when first shown (OrthoViews._copy_display), so nothing of them is read
    before that.
    """
    channels = image.ome.get("omero", {}).get("channels", [])
    colors = [ch.get("color", "FFFFFF") for ch in channels]
    if len(colors) != image.n_channels:
        colors = [DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]
                  for i in range(image.n_channels)]
    dtype = image.levels[0].array.dtype
    unmeasured = ((0.0, float(np.iinfo(dtype).max))
                  if np.issubdtype(dtype, np.integer) else (0.0, 1.0))
    display = []
    for k in range(image.n_channels):
        shown = visible is None or k in visible
        display.append({
            "contrast_limits": (channel_contrast(image, k, region) if shown
                                else unmeasured),
            "gamma": 1.0, "colormap": colormap(colors[k]), "visible": shown,
            "opacity": 1.0, "blending": "additive", "measured": shown})
    return display


# ---------------------------------------------------------------------------
# Masked data for the 3D view
# ---------------------------------------------------------------------------

class VolumeCache:
    """The last whole volume read from one channel's lazy array.

    The 3D view shows a whole volume (one timepoint). Keeping it in memory
    means that changing the masking (a new outline, another display mode)
    does not read it again from the disk or network. napari asks for
    different parts (a new layer first asks for one plane), so the cache
    always reads the whole volume of the requested timepoint and cuts the
    requested part out of it.
    """

    def __init__(self, source, n_spatial: int = 3):
        self.source = source           # lazy (dask) array of one channel
        self.n_lead = source.ndim - n_spatial     # e.g. 1 for time
        self._key = None
        self._data = None

    def get(self, key) -> np.ndarray:
        key = _full_key(key, self.source.ndim)
        lead, spatial = key[:self.n_lead], key[self.n_lead:]
        if repr(lead) != self._key:
            self._data = np.asarray(
                self.source[lead + (slice(None),) * (self.source.ndim
                                                     - self.n_lead)])
            self._key = repr(lead)
        # Integer indices in lead removed those axes from _data; slices
        # kept them, so cut those again with the same slices.
        kept_lead = tuple(slice(None) for k in lead if not _is_integer(k))
        return self._data[kept_lead + spatial]


class MaskedVolume:
    """One channel, split by a masked region, for a napari image layer.

    part="kept":   the region's voxels are 0 (what the crop will contain).
    part="masked": only the region's voxels (what the crop will lose).
    napari only needs shape, dtype, ndim and indexing (data[...]) from an
    image, so this small class can stand in for an array.
    """

    def __init__(self, cache: VolumeCache, region: np.ndarray, part: str):
        self.cache, self.region, self.part = cache, region, part
        source = cache.source
        self.shape, self.dtype, self.ndim = source.shape, source.dtype, source.ndim
        self.size = source.size

    def __getitem__(self, key):
        key = _full_key(key, self.ndim)
        data = self.cache.get(key)
        # region covers the spatial axes; leading axes (time) broadcast
        inside = np.broadcast_to(self.region, self.shape)[key]
        zero = np.zeros((), dtype=self.dtype)
        if self.part == "kept":
            return np.where(inside, zero, data)
        return np.where(inside, data, zero)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[...], dtype=dtype)


def _full_key(key, ndim: int) -> tuple:
    """An index as one entry per axis: data[3] -> (3, :, :, :) for 4D,
    data[...] -> (:, :, :, :)."""
    key = key if isinstance(key, tuple) else (key,)
    dots = [i for i, k in enumerate(key) if k is Ellipsis]
    if dots:
        i = dots[0]
        key = key[:i] + (slice(None),) * (ndim - len(key) + 1) + key[i + 1:]
    return key + (slice(None),) * (ndim - len(key))


def _is_integer(index) -> bool:
    return isinstance(index, (int, np.integer))


# ---------------------------------------------------------------------------
# The views
# ---------------------------------------------------------------------------

@dataclass
class View:
    """One of the four views and the layers it holds."""

    name: str
    model: ViewerModel             # napari.Viewer for XY
    frame: QWidget                 # title + canvas, placed in the grid
    title: QLabel
    order: list[str] = field(default_factory=list)   # world axis order
    channel_layers: list = field(default_factory=list)
    shapes: Shapes | None = None                     # 2D views only
    shape_boxes: list = field(default_factory=list)  # box index per shape
    shape_keys: list = field(default_factory=list)   # its polygon's slice / RECT
    shading: Image | None = None                     # 2D: kept / set to 0
    preview_inside: Image | None = None              # 2D: crop preview, box
    preview_outside: Shapes | None = None            # 2D: crop preview, rest
    # 2D: the crop preview's projection, one per channel (0.18.0)
    crop_projection_layers: dict = field(default_factory=dict)
    crosshair: Shapes | None = None                  # 2D views only
    seeds: Points | None = None                      # 2D: seeds (0.16.0)
    # 2D: the Autocropper's own threshold, outlined (0.18.0)
    level_lines: Vectors | None = None
    level_zoom: float | None = None                  # its dots made for
    # 2D: the Autocropper's gap lines, dark edge and white dash-dot
    gap_edge: Vectors | None = None
    gap_line: Vectors | None = None
    gap_zoom: float | None = None
    # 2D views: the boxes' solid and dashed lines ("boxes", "selected")
    outline_layers: dict = field(default_factory=dict)
    box_layers: list = field(default_factory=list)   # 3D: one per box
    handles: Points | None = None    # 3D: the selected box's dots (0.18.0)
    surface_layers: list = field(default_factory=list)  # 3D: outlines
    mask_layers: list = field(default_factory=list)  # 3D: dimmed/grey part
    plane_layers: dict = field(default_factory=dict)  # 3D view only

    @property
    def plane(self) -> tuple[str, str]:
        """The two axes shown: (vertical, horizontal)."""
        return self.order[-2], self.order[-1]


class OrthoViews:
    """Four linked views of one image inside a napari main window."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.image: vc.OmeZarrImage | None = None
        self.views: dict[str, View] = {}
        self.cursor: dict[str, float] = {}     # current position, µm
        self.display: list[dict] = []          # per channel, see DISPLAY_ATTRS
        self.region: LoadedRegion | None = None  # what is read and shown
        self.level_2d: int | None = None       # None = automatic
        self.exact_2d = False
        self.show_3d = False
        self.level_3d: int | None = None       # None = auto_3d_level()
        self._level_3d_used = 0                # the 3D view's level now
        self.clip_3d = False
        self.projection = "none"        # 2D views: "none", "max" or "mean"
        # What projections go through: the "full" depth, the selected "box",
        # a "slab" of slab_thickness µm centred on the slice (it moves with
        # the slider), or chosen "range"s, {depth axis: (low, high)}.
        self.project_through = "full"
        self.slab_thickness = 50.0
        self.project_ranges: dict = {}
        self.show_planes_3d = True      # outline the 2D views' planes in 3D
        self.show_crosshairs = True     # the 2D views' crosshair lines
        self.show_boxes = True          # crop boxes in all views
        self.show_marks = True          # × marks of the selected box (0.20.0)
        self.mask_mode = "dim"          # see MASK_MODES
        self._mask_key = None           # what the 3D masking shows now
        self._mask_layers_mode = None   # the mode the extra layers show
        self._volume_caches: list = []  # per channel, while masking is on
        self._dash_zoom: dict = {}      # zoom the dashes were made for
        self.link_cameras = True
        self.boxes: list = []
        # per box: ranges as cropped, or None (see the box_ranges property)
        self._box_ranges: list = []
        # (0.17.0) per box: None, or for a region that moves over time a
        # function time -> its ranges as cropped then (see update_boxes)
        self._box_moves: list = []
        self._moved: dict = {}          # {time: box_ranges then} (one entry)
        self._time_followed = None      # the time the moving boxes are at
        # (0.17.0) the panel's boxes; self.boxes are they, or, if a box has
        # outlines per timepoint, each box as it is at the timepoint shown
        # (volume_cropper.box_at), so that all outline drawing shows that
        # timepoint's outlines
        self._source_boxes: list = []
        self._boxes_time = None
        self.on_time_changed = None     # f(): the timepoint shown changed
        self.hidden_boxes: set = set()   # names of boxes not drawn (cropped)
        self.selected = -1
        # Shading of the selected box in the 2D views (off by default).
        self.shading = {"on": False, "kept": SHADE_KEPT,
                        "zeroed": SHADE_ZEROED, "opacity": SHADE_OPACITY}
        # Crop preview in the 2D views (off by default): black wherever the
        # selected box's crop has nothing on the slice shown.
        self.preview = False
        self._preview_keys: dict = {}   # per view: what the preview shows
        # (0.18.0) With the preview on, a projection shows only what the
        # crop keeps (crop_projection.py), computed in the background:
        # _crop_key is what should be shown, _crop_shown what is, and
        # _crop_planes holds the result per channel for _crop_shown's key.
        self._crop_key = None
        self._crop_shown = None
        self._crop_result = None
        self._crop_job = None           # (key, channels, future) running
        self._crop_executor = None      # one thread, made when first needed
        self._crop_timer = QTimer()     # waits for edits to pause
        self._crop_timer.setSingleShot(True)
        self._crop_timer.setInterval(CROP_PROJECTION_DELAY_MS)
        self._crop_timer.timeout.connect(self._start_crop_projection)
        self._crop_poll = QTimer()      # looks for the result
        self._crop_poll.setInterval(100)
        self._crop_poll.timeout.connect(self._poll_crop_projection)
        self.on_crop_projection = None  # f(text): what the preview shows
        # Called when boxes are drawn, edited or deleted in a view; set by
        # the panel, which owns the list of boxes. A shape is a box's
        # rectangle or one of its polygons (key: the polygon's slice, see
        # CropBox.outlines; RECT for the rectangle).
        self.on_box_added = None       # f(ranges, plane, outline or None)
        # (0.18.0) a box's ranges changed by its dots in 3D; set by the
        # panel: f(index, {axis: (low, high)}) for the axes changed
        self.on_box_ranges = None
        self.on_box_picked = None      # f(index): a box clicked in 3D
        self._drag_3d = None           # the dot being dragged, see below
        self._last_face = None         # (box, axis, side) for Up/Down
        self.on_boxes_changed = None   # f({index: (ranges, outline, key)}, plane)
        self.on_boxes_removed = None   # f([(index, key), ...], plane)
        self.on_tool_changed = None    # f(mode): the drawing tool changed
        # (0.16.0) Seeds of "Autocropper", {axis: value}, drawn in
        # the 2D views (set_seeds). While on_seed_click is set, a click (no
        # drag) in a 2D view with the pan/zoom tool calls it with the point
        # clicked: {axis: value} for all range axes (the view's two from
        # the click, the others from the cursor: the slice and timepoint),
        # and the view's name.
        self.seeds: list[dict] = []
        self.on_seed_click = None      # f(point, view name)
        # (0.17.0) where and how seed k is drawn at a time:
        # f(k, seed, time) -> (point {axis: value}, rgba); None = where it
        # was clicked, in SEED_COLOR. Set by the panel (a seed follows the
        # moving region it made, in that region's colour).
        self.seed_style = None
        # (0.18.0) the Autocropper's own threshold, outlined in the 2D
        # views: f(plane, depth axis, (low, high) along it) -> lines, each
        # an (N, 2) array of points (µm) in the plane's order; None = none
        self.level_outline = None
        # (0.18.0) the Autocropper's gap lines: dicts {"view", "plane",
        # "depth", "at", "time", "points", "width"} (see autocrop.gap_walls);
        # while gap_mode is on, a drag in a 2D view draws one
        # (on_gap_line(line)) and a click reports where (on_gap_click(point,
        # view name), e.g. to delete the line there)
        self.gap_lines: list[dict] = []
        self.gap_mode = False
        self.gap_width = 2.0                 # µm, given to new lines
        self.on_gap_line = None
        self.on_gap_click = None
        self._live_gap = None                # (view name, points) drawing
        # (0.18.0) regions' outlines in 3D (see OUTLINE_3D_MODES)
        self.outlines_3d = "all"
        self.outlines_3d_opacity = OUTLINE_3D_OPACITY
        self._meshes: dict = {}              # key: (vertices, faces)
        # a click becomes a seed after this long without a second click (a
        # double-click zooms); None = the system's double-click time
        self.seed_delay_ms: int | None = None
        self._pending_seed = None
        # Called when the pyramid level shown in the 2D views changes
        # (e.g. with zoom in Auto mode): f({"XY": level, "XZ": ..., ...}).
        self.on_levels_shown = None
        self._levels_shown: dict[str, int] = {}
        # Called as on_display(channel, attribute, value) after a display
        # setting changed (set by the panel, to keep its controls in step).
        self.on_display = None
        # (0.18.0) Called as on_xy_data(channel) when napari has loaded new
        # data into the XY view's layer of a channel (another slice, zoom,
        # level or projection); the panel's Levels histogram follows it.
        self.on_xy_data = None
        # Guards: True while we change views ourselves, so that the event
        # handlers don't react to our own changes.
        self._syncing = {"dims": False, "camera": False, "shapes": False,
                         "mode": False}
        self._center: dict[str, float] = {}
        self._focus_buttons: dict = {}
        self._focused: str | None = None    # the view shown alone, if any
        self._saved_sizes = None            # the grid's sizes before focus
        self._grid_sized = False            # four equal cells set once
        # A 2D image (one z plane, e.g. a 2D movie; 0.12.0): only the XY
        # view is shown, the XZ, YZ and 3D views would show nothing useful.
        self.flat = False
        self._build_grid()

    # --- the 2x2 grid ---------------------------------------------------------

    def _build_grid(self):
        """Four views in two columns (XY over XZ, YZ over 3D).

        The borders between them can be dragged (Qt "splitters"). Both
        columns keep the same row heights, so XY and YZ always share their
        y axis (and XY and XZ their x axis), as in Fiji. Each view's title
        bar has a "Focus" button that shows that view alone.
        """
        _prime_opengl_info()
        window = self.viewer.window
        main_qt = window._qt_viewer
        models = {"XY": self.viewer}
        qts = {"XY": main_qt}
        for name in ("XZ", "YZ", "3D"):
            models[name] = ViewerModel(title=name)
            qts[name] = QtViewer(models[name])
        models["3D"].dims.ndisplay = 3
        self._qt_viewers = qts            # keep references alive

        grid = QSplitter(Qt.Orientation.Horizontal)
        self._columns = []
        for names in COLUMNS:
            column = QSplitter(Qt.Orientation.Vertical)
            for name in names:
                frame = QWidget()
                box = QVBoxLayout(frame)
                box.setContentsMargins(0, 0, 0, 0)
                box.setSpacing(0)
                bar = QHBoxLayout()
                title = QLabel(name)
                title.setStyleSheet(
                    f"color: {PLANE_COLORS[name]}; font-weight: bold;")
                focus = QToolButton()
                focus.setText("Focus")
                focus.setToolTip("Show only this view (click again to show "
                                 "all four). The borders between views can "
                                 "also be dragged.")
                focus.setAutoRaise(True)
                focus.clicked.connect(partial(self._toggle_focus, name))
                bar.addWidget(title)
                bar.addStretch()
                bar.addWidget(focus)
                box.addLayout(bar)
                box.addWidget(qts[name], 1)
                # Small minimum, so that borders can be dragged far (napari's
                # slider row would otherwise keep each view ~270 px wide).
                frame.setMinimumSize(120, 100)
                column.addWidget(frame)
                self.views[name] = View(name, models[name], frame, title)
                self._focus_buttons[name] = focus
            column.setChildrenCollapsible(False)
            grid.addWidget(column)
            self._columns.append(column)
        grid.setChildrenCollapsible(False)
        # The 3D view's cell stays when 3D is off (so the rows and columns
        # stay aligned), with this note instead of the canvas.
        self._3d_off = QLabel("3D view is off (tick '3D view' in the panel).")
        self._3d_off.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._3d_off.setVisible(False)
        self.views["3D"].frame.layout().addWidget(self._3d_off, 1)
        # Keep both columns' row heights equal while a border is dragged.
        left, right = self._columns
        left.splitterMoved.connect(lambda *_: right.setSizes(left.sizes()))
        right.splitterMoved.connect(lambda *_: left.setSizes(right.sizes()))
        center = window._qt_window.centralWidget()
        center.layout().removeWidget(main_qt)
        # The views and the side panel (add_side_panel) share one splitter,
        # so that the border between them can be dragged. A napari dock on
        # the right could not be resized by dragging (napari 0.9.2).
        self._outer = QSplitter(Qt.Orientation.Horizontal)
        self._outer.addWidget(grid)
        self._outer.setCollapsible(0, False)
        self._outer.setStretchFactor(0, 1)       # the views get extra room
        # Make the draggable borders (here and between the views) visible,
        # in a colour of napari's theme.
        try:
            from napari.utils.theme import get_theme
            color = get_theme(self.viewer.theme).foreground.as_hex()
        except Exception:
            color = "#414851"
        self._outer.setStyleSheet(
            f"QSplitter::handle {{ background: {color}; }}")
        center.layout().addWidget(self._outer)
        self._grid = grid
        # napari's buttons for 2D/3D, rolling or transposing axes and grid
        # mode would turn the XY view into a different kind of view, which
        # the four linked views can't follow (their shortcuts are caught
        # in _keep_layout).
        buttons = main_qt.viewerButtons
        for button in (buttons.ndisplayButton, buttons.rollDimsButton,
                       buttons.transposeDimsButton, buttons.gridViewButton):
            button.setVisible(False)
        self._connect_views()

    def add_side_panel(self, widget: QWidget):
        """Put a panel to the right of the four views. The border between
        them can be dragged; dragging it to the window's edge closes the
        panel, and dragging it back opens it again."""
        self._outer.addWidget(widget)
        self._outer.setCollapsible(1, True)
        self._outer.setStretchFactor(1, 0)       # keeps its width
        self._outer.setHandleWidth(6)

    def set_panel_width(self, width: int):
        """Make the side panel this many pixels wide."""
        total = sum(self._outer.sizes())
        if total > 0 and self._outer.count() > 1:
            width = min(width, total - 240)      # leave room for the views
            self._outer.setSizes([total - width, width])

    def equal_cells(self):
        """Give the four views equal sizes (and both columns equal rows)."""
        width = sum(self._grid.sizes())
        self._grid.setSizes([width // 2, width - width // 2])
        for column in self._columns:
            height = sum(column.sizes())
            column.setSizes([height // 2, height - height // 2])

    def _toggle_focus(self, name: str, checked: bool = False):
        self.set_focus(None if self._focused == name else name)

    def set_focus(self, name: str | None):
        """Show one view alone (e.g. "XZ"), or all four again (None).

        The other views are only hidden: they keep their place, zoom and
        layers, and the grid gets its earlier sizes back.
        """
        if name is not None and self._focused is None:
            self._saved_sizes = (self._grid.sizes(),
                                 [c.sizes() for c in self._columns])
        self._focused = name
        for names, column in zip(COLUMNS, self._columns):
            column.setVisible(name is None or name in names)
            for n in names:
                self.views[n].frame.setVisible(name is None or n == name)
        for n, button in self._focus_buttons.items():
            button.setText("Show all" if n == name else "Focus")
        if name is None and self._saved_sizes is not None:
            grid_sizes, column_sizes = self._saved_sizes
            self._grid.setSizes(grid_sizes)
            for column, sizes in zip(self._columns, column_sizes):
                column.setSizes(sizes)

    def _set_flat(self, flat: bool):
        """A 2D image (one z plane) shows only the XY view, without its
        Focus button; any other image all four views again."""
        if flat == self.flat:
            return
        self.flat = flat
        if flat:
            self.show_3d = False
            self.set_focus("XY")
        else:
            self.set_focus(None)
        for button in self._focus_buttons.values():
            button.setVisible(not flat)

    def _connect_views(self):
        for name in VIEWS_2D:
            view = self.views[name]
            view.model.dims.events.point.connect(partial(self._on_dims, view))
            cam = camera(view.model)
            cam.events.zoom.connect(partial(self._on_camera, view))
            cam.events.center.connect(partial(self._on_camera, view))
            view.model.bind_key("j", self._jump_key(view), overwrite=True)
            view.model.mouse_drag_callbacks.append(self._seed_click(view))
            view.model.mouse_drag_callbacks.append(self._gap_draw(view))
            view.model.mouse_double_click_callbacks.append(self._cancel_seed)
            scale_bar(view.model).visible = True
        self.views["3D"].model.dims.events.point.connect(
            partial(self._on_dims, self.views["3D"]))
        # (0.18.0) dragging the selected region's dots in 3D; Up/Down move
        # the face last dragged by a voxel (Shift: 10); napari leaves these
        # keys free (Left/Right step the sliders)
        model_3d = self.views["3D"].model
        model_3d.mouse_drag_callbacks.append(self._handle_drag)
        camera(model_3d).events.zoom.connect(self._size_3d_markers)
        for key, steps in (("Up", 1), ("Down", -1), ("Shift-Up", 10),
                           ("Shift-Down", -10)):
            model_3d.bind_key(key, self._nudge_key(steps), overwrite=True)
        for view in self.views.values():
            for emitter in (view.model.dims.events.ndisplay,
                            view.model.dims.events.order,
                            grid_of(view.model).events.enabled):
                emitter.connect(partial(self._keep_layout, view))

    def _keep_layout(self, view: View, event=None):
        """Undo napari's view-changing shortcuts in one of the four views.

        Ctrl+Y (2D/3D), Ctrl+E/Ctrl+T (roll/transpose axes) and Ctrl+G
        (grid mode) would turn a view into something the linked views can't
        follow. (In XY, 2D/3D also made napari fail on every redraw.)
        The change is undone right after napari has finished handling it.
        """
        def restore():
            dims = view.model.dims
            wanted = 3 if view.name == "3D" else 2
            changed = False
            if dims.ndisplay != wanted:
                dims.ndisplay = wanted
                changed = True
            if tuple(dims.order) != tuple(range(dims.ndim)):
                dims.order = tuple(range(dims.ndim))
                changed = True
            if grid_of(view.model).enabled:
                grid_of(view.model).enabled = False
                changed = True
            if changed:
                show_info("The four views keep their layout: XY, XZ and YZ "
                          "stay 2D, and the 3D view is at the bottom right.")
        QTimer.singleShot(0, restore)

    # --- opening an image -------------------------------------------------------

    def canvas_sizes(self) -> dict:
        """The 2D views' sizes on screen: {name: (height, width)} in pixels.
        (A view not on screen yet counts as 400 x 600.)"""
        self._prepare_grid()
        sizes = {}
        for name in VIEWS_2D:
            size = tuple(self.views[name].model.canvas.size)
            sizes[name] = size if min(size) > 10 else (400, 600)
        return sizes

    def _prepare_grid(self):
        # Let Qt lay out a newly built window first: a view without a size
        # on screen would give a meaningless zoom, and napari would then
        # pick a far too large region to load.
        QApplication.processEvents()
        if not self._grid_sized:        # start with four equal cells
            self._grid_sized = True
            self.equal_cells()
            QApplication.processEvents()

    def set_image(self, image: vc.OmeZarrImage, visible: list[int] | None = None,
                  region: LoadedRegion | None = None):
        """Show a new image in all views (removes the previous one).

        visible: the channels to show (None = all). The others start hidden,
        and a hidden layer reads no data until it is shown.
        region: the part of the image to read and show (see loaded_region;
        default: the whole image).
        """
        orders = view_orders(image)     # raises for non-3D images
        self._prepare_grid()
        self.clear()
        z = image.axis_names.index(orders["XY"][-3])
        self._set_flat(image.levels[0].array.shape[z] == 1)
        self.image = image
        self.region = region or loaded_region(image)
        self.display = initial_display(image, self.region, visible)
        self.cursor = {a: float(np.mean(self.extent(a)))
                       for a in image.range_axes}
        for name, view in self.views.items():
            view.order = orders[name]
        for name in VIEWS_2D:
            view = self.views[name]
            # Hidden until the cameras fit the image: napari draws a new
            # layer at once, and at its default zoom (1 pixel = 1 µm) it
            # would load a large region of the finest level.
            self._build_channel_layers(view, show=False)
            # Overlays use "translucent_no_depth" blending: with depth
            # testing (napari's default "translucent") a layer drawn earlier
            # can hide the lines drawn after it.
            # Optional shading of the selected box: above the image, below
            # the lines; one small colour image of the current slice, made
            # anew when the slice or the box changes.
            view.shading = Image(np.zeros((1, 1, 4), dtype=np.uint8),
                                 rgb=True, name="kept / set to 0",
                                 metadata=dict(HELPER),
                                 visible=False,
                                 blending="translucent_no_depth",
                                 units=layer_units(image, list(view.plane)),
                                 axis_labels=list(view.plane))
            view.model.layers.append(view.shading)
            # Optional crop preview of the selected box, above the image and
            # the shading, below the lines: black wherever the crop has
            # nothing. Inside the box an image of the voxels its outlines
            # set to 0 (like the shading); outside it black rectangles,
            # whose edges are exactly the crop's.
            view.preview_inside = Image(
                np.zeros((1, 1, 4), dtype=np.uint8), rgb=True,
                name="crop preview (inside)", visible=False,
                metadata=dict(HELPER),
                blending="translucent_no_depth",
                units=layer_units(image, list(view.plane)),
                axis_labels=list(view.plane))
            view.model.layers.append(view.preview_inside)
            view.preview_outside = Shapes(
                ndim=2, name="crop preview (outside)", visible=False,
                metadata=dict(HELPER),
                opacity=1.0, blending="translucent_no_depth",
                units=layer_units(image, list(view.plane)),
                axis_labels=list(view.plane))
            view.model.layers.append(view.preview_outside)
            # The boxes' lines: many short line pieces in one layer, which
            # napari draws fast (dashes are pieces too). Two layers, because
            # a layer has one line width: the selected box is drawn wider.
            view.outline_layers = {}
            for key, name in (("boxes", "region lines"),
                              ("selected", "selected region lines")):
                layer = Vectors(np.zeros((0, 2, 2)), ndim=2, name=name,
                                metadata=dict(HELPER),
                                vector_style="line", visible=False,
                                blending="translucent_no_depth",
                                units=layer_units(image, list(view.plane)),
                                axis_labels=list(view.plane))
                view.model.layers.append(layer)
                view.outline_layers[key] = layer
            # (0.18.0) the Autocropper's own threshold: a dotted line round
            # what lies above it on the slice (set_level_outline); opaque
            # (napari's 0.7 made the dots hard to see on bright tissue)
            view.level_lines = Vectors(
                np.zeros((0, 2, 2)), ndim=2, name="threshold outline",
                metadata=dict(HELPER), vector_style="line", visible=False,
                edge_color=THRESHOLD_COLOR, opacity=1.0,
                blending="translucent_no_depth",
                units=layer_units(image, list(view.plane)),
                axis_labels=list(view.plane))
            view.model.layers.append(view.level_lines)
            # (0.18.0) the Autocropper's gap lines: a dark edge under a
            # white dash-dot line (set_gap_lines)
            for attr, name in (("gap_edge", "gap lines (edge)"),
                               ("gap_line", "gap lines")):
                layer = Vectors(np.zeros((0, 2, 2)), ndim=2, name=name,
                                metadata=dict(HELPER), vector_style="line",
                                visible=False, opacity=1.0,
                                blending="translucent_no_depth",
                                units=layer_units(image, list(view.plane)),
                                axis_labels=list(view.plane))
                view.model.layers.append(layer)
                setattr(view, attr, layer)
            view.crosshair = Shapes(
                ndim=2, name="crosshair", opacity=0.7,
                metadata=dict(HELPER),
                blending="translucent_no_depth",
                visible=self.show_crosshairs,
                units=layer_units(image, list(view.plane)),
                axis_labels=list(view.plane))
            view.model.layers.append(view.crosshair)
            view.seeds = Points(
                np.zeros((0, 2)), ndim=2, name="seeds", symbol="cross",
                metadata=dict(HELPER), face_color=SEED_COLOR,
                # (0.17.0) a dark edge: seeds in every region colour, and
                # white ones, show on dark and on bright tissue
                border_width=0.18, border_color=SEED_EDGE, visible=False,
                blending="translucent_no_depth",
                units=layer_units(image, list(view.plane)),
                axis_labels=list(view.plane))
            view.model.layers.append(view.seeds)
            view.shapes = Shapes(
                ndim=2, name="crop regions", face_color="transparent",
                edge_color=BOX_COLORS[0], blending="translucent_no_depth",
                visible=self.show_boxes,
                units=layer_units(image, list(view.plane)),
                axis_labels=list(view.plane))
            view.shapes.current_face_color = "transparent"
            view.model.layers.append(view.shapes)
            view.shapes.events.data.connect(partial(self._on_shapes, view))
            view.shapes.events.mode.connect(partial(self._on_mode, view))
            view.model.layers.selection.active = view.shapes
        self._build_3d()
        self._set_cursor(dict(self.cursor))
        self.reset_cameras()
        self._zoom_to_level()                                 # (0.18.0)
        self._draw_seeds()
        for name in VIEWS_2D:                 # now it is safe to draw
            for k, layer in enumerate(self.views[name].channel_layers):
                layer.visible = self.display[k]["visible"]

    def _remove_layers(self, view: View, layers=None):
        """Remove layers from a view (None = all of them) safely.

        napari deletes a removed layer's OpenGL objects at once, in whatever
        OpenGL context is active. With four canvases that is often another
        view's, or none, and on Windows that crashes the program ("access
        violation"; seen when switching the 3D masking with clipping on).
        So this view's canvas is made the active one first, as vispy itself
        does before deleting.
        """
        layers = list(view.model.layers if layers is None else layers)
        layers = [layer for layer in layers if layer in view.model.layers]
        if not layers:
            return
        canvas = getattr(self._qt_viewers.get(view.name), "canvas", None)
        try:
            canvas._scene_canvas.set_current()
        except Exception:       # e.g. a canvas never shown: nothing to delete
            pass
        for layer in layers:
            view.model.layers.remove(layer)

    def clear(self):
        for view in self.views.values():
            self._remove_layers(view)
            view.channel_layers, view.box_layers = [], []
            view.mask_layers, view.outline_layers = [], {}
            view.plane_layers = {}
            view.shapes = view.crosshair = view.seeds = None
            view.level_lines = None
            view.gap_edge = view.gap_line = view.gap_zoom = None
            view.shading = view.preview_inside = view.preview_outside = None
            view.crop_projection_layers = {}
            view.handles = None
            view.surface_layers = []
        self.seeds = []
        self.level_outline = None     # (the Autocropper sets it anew)
        self.gap_lines, self._live_gap = [], None
        self._preview_keys = {}
        self._crop_key = self._crop_shown = self._crop_result = None
        self.boxes, self._box_ranges, self.selected = [], [], -1
        self._box_moves, self._moved, self._time_followed = [], {}, None
        self._source_boxes, self._boxes_time = [], None
        self._mask_key, self._volume_caches, self._dash_zoom = None, [], {}
        self._mask_layers_mode = None
        self._levels_shown = {}
        for name, view in self.views.items():
            view.title.setText(name)

    def reset_cameras(self):
        """Fit every view, then give the 2D views one zoom and centre."""
        # Each view is fitted on its own: with the zooms linked, every reset
        # would pass its zoom on to the other views, and the last view's
        # zoom would win instead of the smallest (the views then started
        # too far zoomed in, cutting off the image and reading finer data).
        before = self._syncing["camera"]
        self._syncing["camera"] = True
        try:
            for view in self.views.values():
                view.model.reset_view()
        finally:
            self._syncing["camera"] = before
        if self.image is None:
            return
        spatial = self.views["XY"].order[-3:]
        self._center = {a: float(np.mean(self.extent(a))) for a in spatial}
        # One zoom that fits the image in every view (ignoring views that
        # have no size on screen yet).
        zooms = [camera(self.views[n].model).zoom for n in VIEWS_2D
                 if min(self.views[n].model.canvas.size) > 10]
        if zooms:
            self._apply_camera(min(zooms), source=None)

    def _zoom_to_level(self):
        """(0.18.0, the user's wish) When an image opens with a chosen 2D
        resolution (not Auto): zoom the 2D views in until one voxel of that
        level is one screen pixel, so that napari shows that level
        (level_for_pixel: the coarsest level whose voxels are smaller than
        a pixel, never finer than the one chosen), centred on the image as
        reset_cameras left it. Never further out than the fitted view: if
        the whole image fits at that level, it stays fitted. Only on
        opening; changing the resolution later keeps the zoom."""
        if self.image is None or self.level_2d is None:
            return
        level = min(self.level_2d, self.region.n_levels - 1)
        scale = self.image.levels[level].scale
        names = self.image.axis_names
        voxel = min(scale[names.index(a)] for a in self.views["XY"].plane)
        sized = [n for n in VIEWS_2D
                 if min(self.views[n].model.canvas.size) > 10]
        if not sized or not voxel > 0:
            return
        fitted = min(camera(self.views[n].model).zoom for n in sized)
        if 1.0 / voxel > fitted:
            self._apply_camera(1.0 / voxel, source=None)

    def extent(self, axis: str) -> tuple[float, float]:
        """The range shown along an axis: the loaded region's."""
        return self.region.ranges[axis]

    def n_shown(self) -> int:
        """How many channels are shown (and read), at least 1."""
        return max(sum(bool(d["visible"]) for d in self.display), 1)

    def level_3d_shown(self) -> int:
        """The level the 3D view shows now."""
        return self._level_3d_used

    def auto_3d(self) -> int:
        """The 3D view's "Auto" level for the region and channels shown."""
        return auto_3d_level(self.image, region=self.region,
                             n_channels=self.n_shown())

    # --- channel layers -----------------------------------------------------------

    def _channel_array(self, level: int, k: int, order: list[str]):
        """Lazy array of one channel of one level (its loaded part), in a
        view's axis order."""
        array = da.from_zarr(self.image.levels[level].array)
        if not self.region.whole:
            array = array[self.region.slices(level)]
        c_axis = self.image.channel_axis
        if c_axis is not None:
            array = array[(slice(None),) * c_axis + (k,)]
        names = self.image.range_axes
        return array.transpose([names.index(a) for a in order])

    def _levels_for(self, view: View) -> list[int]:
        if view.name == "3D":
            return [self._level_3d_used]       # set when the view is built
        n = self.region.n_levels
        level = None if self.level_2d is None else min(self.level_2d, n - 1)
        return pyramid_levels(n, level, self.exact_2d)

    def _build_channel_layers(self, view: View, show: bool = True):
        """(Re)create a view's channel layers at the current level choice.

        The layers are added hidden (a hidden layer loads no data) and shown
        afterwards if show is True. The new layers go in before the old ones
        are removed, so that the view never loses its depth axis (napari
        would then reset its slider).
        """
        old = list(view.channel_layers)
        levels = self._levels_for(view)
        first = self.image.levels[levels[0]]
        offset = self.region.translation(self.image, levels[0])
        index = [self.image.axis_names.index(a) for a in view.order]
        multiscale = view.name != "3D"
        view.channel_layers = []
        for k, label in enumerate(self.image.channel_labels or ["image"]):
            data = [self._channel_array(level, k, view.order)
                    for level in levels]
            settings = {**self.display[k], "visible": False}
            layer = Image(
                data if multiscale else data[0], multiscale=multiscale,
                projection_mode=self.projection if multiscale else "none",
                name=defang(f"{k}: {label}"),
                scale=[first.scale[i] for i in index],
                translate=[offset[i] for i in index],
                units=layer_units(self.image, view.order),
                axis_labels=list(view.order),
                **{attr: settings[attr] for attr in DISPLAY_ATTRS})
            # (0.18.0) the pyramid levels its data holds (data[j] is level
            # levels[j]): current_level reads them from the layer, as the
            # choice (_levels_for) may change before the layer is replaced
            # (the user's IndexError of 2026-11-13: a second image opened
            # at level 3 while the first one's layers showed their level 1)
            layer.vc_levels = tuple(levels)
            view.model.layers.insert(k, layer)
            view.channel_layers.append(layer)
        self._remove_layers(view, old)
        if multiscale:            # report the level shown (it follows zoom)
            self._levels_shown.pop(view.name, None)
            for layer in view.channel_layers:   # (any may be the one shown)
                layer.events.set_data.connect(
                    partial(self._on_level_shown, view))
        if show:
            for k, layer in enumerate(view.channel_layers):
                layer.visible = self.display[k]["visible"]
        if view.name == "XY":     # copy display changes to the other views
            for k, layer in enumerate(view.channel_layers):
                for attr in DISPLAY_ATTRS:
                    getattr(layer.events, attr).connect(
                        partial(self._copy_display, k, attr))
                layer.events.set_data.connect(partial(self._xy_data, k))
        if view.shapes is not None:
            view.model.layers.selection.active = view.shapes

    def _copy_display(self, k: int, attr: str, event=None):
        xy_layers = self.views["XY"].channel_layers
        if k >= len(xy_layers):
            return
        if attr == "visible" and xy_layers[k].visible \
                and not self.display[k].get("measured", True):
            # Shown for the first time: measure its contrast now (its
            # contrast events pass the result on to the other views).
            self.display[k]["measured"] = True
            limits = channel_contrast(self.image, k, self.region)
            xy_layers[k].contrast_limits_range = limits
            xy_layers[k].contrast_limits = limits
        value = getattr(xy_layers[k], attr)
        self.display[k][attr] = value
        for name in ("XZ", "YZ", "3D"):
            layers = self.views[name].channel_layers
            if k < len(layers):
                setattr(layers[k], attr, value)
        # The 3D view's dimmed/grey layers follow contrast and visibility,
        # and (unless grey) the colour; their opacity stays fixed.
        masked = self.views["3D"].mask_layers
        if k < len(masked) and (attr in ("contrast_limits", "gamma", "visible")
                                or (attr == "colormap"
                                    and self.mask_mode not in MASK_GREY)):
            setattr(masked[k], attr, value)
        # (0.18.0) the crop preview's projection: the same colours; a
        # channel switched on is computed (_update_crop_projection)
        for name in VIEWS_2D:
            layer = self.views[name].crop_projection_layers.get(k)
            if layer is not None and attr in ("contrast_limits", "gamma",
                                              "colormap"):
                setattr(layer, attr, value)
        if attr == "visible":
            self._update_crop_projection()
        if self.on_display is not None:
            self.on_display(k, attr, value)

    def _xy_data(self, k: int, event=None):
        if self.on_xy_data is not None:
            self.on_xy_data(k)

    def set_channel_visible(self, k: int, visible: bool):
        """Show or hide one channel in all views (via the XY view's layer,
        whose changes _copy_display passes on). Hidden, it is not read."""
        xy_layers = self.views["XY"].channel_layers
        if k < len(xy_layers):
            xy_layers[k].visible = visible
        elif k < len(self.display):
            self.display[k]["visible"] = visible

    def current_level(self, view: View) -> int:
        """The pyramid level a 2D view shows at the moment."""
        # the first visible channel's: napari does not load hidden layers,
        # so a hidden channel 0 kept an old level (the 0.17.0 review)
        shown = [layer for layer in view.channel_layers if layer.visible]
        layer = (shown or view.channel_layers or [None])[0]
        if layer is None:
            return self._levels_for(view)[0]
        # the levels the layer was built with (not the choice now: see
        # _build_channel_layers)
        levels = getattr(layer, "vc_levels", None) or self._levels_for(view)
        return levels[min(int(layer.data_level), len(levels) - 1)]

    def _on_level_shown(self, view: View, event=None):
        """napari (re)loaded a 2D view's data: show its level in the title,
        and tell the panel (for the "Auto" entry)."""
        if self.image is None or not view.channel_layers:
            return
        level = self.current_level(view)
        if not 0 <= level < len(self.image.levels):
            return          # (an old image's layer, while a new one opens)
        if self._levels_shown.get(view.name) == level:
            return
        self._levels_shown[view.name] = level
        voxel = self.image.levels[level].scale[-1]
        flat = " (2D image)" if self.flat else ""
        view.title.setText(f"{view.name}{flat} · level {level} "
                           f"({voxel:.3g} µm)")
        if self.preview:                 # its grid follows the level shown
            with self._display_only():
                self._preview(view)
        if self.on_levels_shown is not None:
            self.on_levels_shown(dict(self._levels_shown))

    # --- projections (2D views) ---------------------------------------------

    def set_projection(self, mode: str, through: str):
        """Project the 2D views along their depth axis.

        mode: "none", "max" (MIP) or "mean" (average intensity projection).
        through: "full" (the whole depth), "box" (the selected box's range),
        "slab" (slab_thickness µm centred on the slice) or "range"
        (project_ranges). XY projects along z, XZ along y and YZ along x.
        """
        self.projection, self.project_through = mode, through
        for name in VIEWS_2D:
            for layer in self.views[name].channel_layers:
                layer.projection_mode = mode
        self._update_slabs()
        if self.image is not None:      # silhouettes are edited on projections
            self._refresh_slice_shapes()
            self._update_shading()
            if self.level_outline is not None:
                self.draw_level_outline()       # (0.18.0) through the slab

    def projection_range(self, view: View):
        """The slab (low, high) in µm along a view's depth axis, or None."""
        if self.projection == "none" or self.image is None:
            return None
        axis = view.order[-3]
        if self.project_through == "box":
            if not 0 <= self.selected < len(self.box_ranges) \
                    or self.box_ranges[self.selected] is None:
                return None
            return self.box_ranges[self.selected][axis]
        if self.project_through == "slab":     # follows the slider
            half = self.slab_thickness / 2
            centre = self.cursor.get(axis, float(np.mean(self.extent(axis))))
            return centre - half, centre + half
        if self.project_through == "range" and axis in self.project_ranges:
            return self.project_ranges[axis]
        return self.extent(axis)

    def _update_slabs(self):
        """Set each 2D view's slice thickness (napari "margins") so that it
        projects through the chosen range; zero thickness = a normal slice."""
        if self.image is None:
            return
        for name in VIEWS_2D:
            view = self.views[name]
            dims = view.model.dims
            i = len(view.order) - 3                 # the depth axis
            slab = self.projection_range(view)
            left = [0.0] * dims.ndim
            right = [0.0] * dims.ndim
            if slab is not None:
                point, left[i], right[i] = projection_slab(
                    float(dims.point[i]), *slab)
                if point != dims.point[i]:          # keep it inside the slab
                    self._syncing["dims"] = True
                    try:
                        dims.set_point(i, point)
                    finally:
                        self._syncing["dims"] = False
                    self.cursor[view.order[i]] = float(dims.point[i])
                    point = float(dims.point[i])    # after snapping
                    left[i], right[i] = point - slab[0], slab[1] - point
            if tuple(dims.margin_left) != tuple(left):
                dims.margin_left = tuple(left)
            if tuple(dims.margin_right) != tuple(right):
                dims.margin_right = tuple(right)
        self._update_crosshairs()

    @contextmanager
    def _keep_place(self, names):
        """Keep the sliders, and the zoom and pan of these views, while
        their image layers are rebuilt.

        Without this, napari resets a view's sliders (and its zoom) when
        its image layers are replaced, so changing the resolution would
        jump to another slice.
        """
        cursor = dict(self.cursor)
        cameras = {}
        for name in names:
            cam = camera(self.views[name].model)
            cameras[name] = (tuple(cam.center), cam.zoom, tuple(cam.angles))
        before = dict(self._syncing)
        self._syncing.update(dims=True, camera=True)  # ignore the resets
        try:
            yield
        finally:
            for name, (center, zoom, angles) in cameras.items():
                cam = camera(self.views[name].model)
                cam.angles, cam.center, cam.zoom = angles, center, zoom
            self._syncing.update(dims=before["dims"], camera=before["camera"])
            self.cursor = cursor
            self._set_cursor(dict(cursor))        # every slider back in place

    def set_2d_level(self, level: int | None, exact: bool):
        """Choose the 2D views' resolution (see pyramid_levels).

        The views stay where they were (slice, zoom and pan). The new layers
        are shown only after that, so nothing is read at a wrong place.
        """
        self.level_2d, self.exact_2d = level, exact
        with self._keep_place(VIEWS_2D):
            for name in VIEWS_2D:
                self._build_channel_layers(self.views[name], show=False)
        for name in VIEWS_2D:
            for k, layer in enumerate(self.views[name].channel_layers):
                layer.visible = self.display[k]["visible"]

    # --- 3D view ------------------------------------------------------------------

    def set_3d(self, show: bool, level: int | None = None):
        """Switch the 3D view on/off and choose its level (None = auto)."""
        self.show_3d, self.level_3d = show, level
        if self.image is not None:
            self._build_3d()

    def _build_3d(self):
        view = self.views["3D"]
        # A new level for a 3D view already on screen keeps its camera (and
        # the time slider); a view just switched on is fitted to the image.
        keep = bool(view.channel_layers) and self.show_3d
        with self._keep_place(["3D"] if keep else []):
            self._remove_layers(view, [*view.channel_layers,
                                       *view.mask_layers, *view.box_layers,
                                       *view.plane_layers.values(),
                                       *view.surface_layers])
            view.channel_layers, view.mask_layers = [], []
            view.box_layers, view.plane_layers = [], {}
            view.surface_layers = []
            self._mask_key, self._volume_caches = None, []
            self._mask_layers_mode = None
            # The cell stays (rows and columns stay aligned); only the
            # canvas is swapped for a note while 3D is off.
            self._qt_viewers["3D"].setVisible(self.show_3d)
            self._3d_off.setVisible(not self.show_3d)
            if not self.show_3d:
                view.title.setText("3D (off)")
                return
            # The level is fixed here: "Auto" depends on the channels shown,
            # which may change later while these layers stay.
            self._level_3d_used = (self.auto_3d() if self.level_3d is None
                                   else min(self.level_3d,
                                            self.region.n_levels - 1))
            # Hidden at first, so that the volume is read only once, after
            # the masking (if any) is set up.
            self._build_channel_layers(view, show=False)
            level = self._levels_for(view)[0]
            voxel = self.image.levels[level].scale[-1]
            view.title.setText(f"3D · level {level} ({voxel:.3g} µm)")
            self._sync_box_layers(view)
            self._build_3d_planes(view)
            self._update_3d_planes()
            self._update_masking()
            self._update_clipping()
            self._sync_surfaces()                             # (0.18.0)
            self._draw_seeds()                                # (0.18.0)
            if not keep:
                view.model.reset_view()
        for k, layer in enumerate(view.channel_layers):
            layer.visible = self.display[k]["visible"]

    def set_clip(self, clip: bool):
        """Show only the inside of the selected box in 3D (or everything)."""
        self.clip_3d = clip
        self._focused_box = None
        self._update_masking()        # clipped: the outside becomes 0
        self._update_clipping()
        if not clip and self.show_3d:
            self.views["3D"].model.reset_view()

    def _update_clipping(self):
        view = self.views["3D"]
        ranges = (self.box_ranges[self.selected]
                  if 0 <= self.selected < len(self.box_ranges) else None)
        spatial = view.order[-3:]
        planes = []
        if self.clip_3d and ranges is not None:
            planes = clipping_planes([ranges[a][0] for a in spatial],
                                     [ranges[a][1] for a in spatial])
        for layer in [*view.channel_layers, *view.mask_layers]:
            layer.experimental_clipping_planes = planes
        # Zoom onto the box when clipping starts or another box is selected,
        # otherwise the clipped region is a small patch in a large view.
        if self.clip_3d and ranges is not None and view.channel_layers \
                and getattr(self, "_focused_box", None) != self.selected:
            self._focused_box = self.selected
            cam = camera(view.model)
            cam.center = tuple((ranges[a][0] + ranges[a][1]) / 2
                               for a in spatial)
            size = max(ranges[a][1] - ranges[a][0] for a in spatial)
            pixels = min(view.model.canvas.size)
            if size > 0 and pixels > 10:
                cam.zoom = 0.7 * pixels / size

    def set_mask_mode(self, mode: str):
        """How 3D shows the voxels that the selected box's outlines will
        black out: "show" (normally), "dim", "grey", "dimgrey" (dimmed and
        grey) or "hide"."""
        if mode not in MASK_MODES:
            raise ValueError(f"mode must be one of {MASK_MODES}")
        self.mask_mode = mode
        self._update_masking()

    def _update_masking(self):
        """Split the 3D view's data into what the selected box keeps and
        what its outlines black out, as set by mask_mode.

        The channel layers show the kept voxels (the blacked-out ones are
        0). For "dim", "grey" and "dimgrey", a second layer per channel shows
        the blacked-out voxels, faint and/or in grey. With "Clip 3D to
        selected region", the voxels outside the box are 0 too, in every mode
        (see outside_region). Otherwise, without outlines or in "show" mode,
        the channel layers show the plain data.
        """
        view = self.views["3D"]
        if self.image is None or not self.show_3d or not view.channel_layers:
            return
        i = self.selected
        box = self.boxes[i] if 0 <= i < len(self.boxes) else None
        ranges = self.box_ranges[i] if 0 <= i < len(self.box_ranges) else None
        level = self._levels_for(view)[0]
        active = (self.mask_mode != "show" and box is not None
                  and ranges is not None and bool(box.outlines))
        clipped = self.clip_3d and ranges is not None
        key = None
        if active or clipped:
            key = (self.mask_mode if active else "show", level, repr(ranges),
                   repr(sorted(box.outlines.items())) if active else "",
                   box.margin if active else 0, clipped)
        if key == self._mask_key:
            return                      # nothing changed: don't redraw
        self._mask_key = key
        # The extra (dimmed/grey) layers stay while their mode stays: then an
        # outline edit only gives them new data. Removing a layer is slow,
        # because napari then runs Python's garbage collection.
        extra_mode = self.mask_mode if active and \
            self.mask_mode in MASK_OPACITY else None
        if view.mask_layers and extra_mode != self._mask_layers_mode:
            self._remove_layers(view, view.mask_layers)
            view.mask_layers = []
        self._mask_layers_mode = extra_mode
        if not active and not clipped:
            self._volume_caches = []    # free the memory
            for k, layer in enumerate(view.channel_layers):
                if isinstance(layer.data, MaskedVolume):
                    layer.data = self._channel_array(level, k, view.order)
            return

        # The blacked-out region (and, clipped, everything outside the box)
        # on the voxel grid of this level's loaded part, shaped to broadcast
        # over leading axes such as time.
        lv = self.image.levels[level]
        offset = self.region.translation(self.image, level)
        shape = self.region.shape(level)
        spatial = view.order[-3:]
        index = [self.image.axis_names.index(a) for a in spatial]
        grid = (spatial, [lv.scale[j] for j in index],
                [offset[j] for j in index], [shape[j] for j in index])
        region = masked_region(*grid, ranges, box.outlines if active else {},
                               spatial=vc._spatial_axes(self.image),
                               margin=box.margin * vc.pixel_size(self.image))
        hidden = region | outside_region(*grid, ranges) if clipped else region
        lead = (1,) * (len(view.order) - 3)
        region = region.reshape(lead + region.shape)
        hidden = hidden.reshape(lead + hidden.shape)
        if not self._volume_caches:     # one read per channel, then reused
            self._volume_caches = [
                VolumeCache(self._channel_array(level, k, view.order))
                for k in range(len(view.channel_layers))]
        for k, layer in enumerate(view.channel_layers):
            layer.data = MaskedVolume(self._volume_caches[k], hidden, "kept")
        if view.mask_layers:            # same mode as before: new data only
            for k, masked in enumerate(view.mask_layers):
                masked.data = MaskedVolume(self._volume_caches[k], region,
                                           "masked")
        elif extra_mode is not None:
            for k, layer in enumerate(view.channel_layers):
                masked = Image(
                    MaskedVolume(self._volume_caches[k], region, "masked"),
                    name=f"{layer.name} (blacked out)",
                    scale=layer.scale, translate=layer.translate,
                    units=layer.units, axis_labels=layer.axis_labels,
                    contrast_limits=layer.contrast_limits, gamma=layer.gamma,
                    colormap=("gray" if self.mask_mode in MASK_GREY
                              else layer.colormap),
                    opacity=MASK_OPACITY[self.mask_mode], blending="additive",
                    visible=self.display[k]["visible"])
                view.model.layers.insert(len(view.channel_layers) + k, masked)
                view.mask_layers.append(masked)
        self._update_clipping()

    # --- boxes ----------------------------------------------------------------------

    def update_boxes(self, boxes: list, box_ranges: list, selected: int,
                     box_moves: list | None = None):
        """Show boxes in all views.

        boxes: the requested boxes (the editable shapes in the 2D views).
        box_ranges: per box, the ranges as they will be cropped (solid lines
            in 2D, wireframes in 3D), or None if the box is invalid.
        box_moves (0.17.0): per box, None, or for a region that moves over
            time a function time -> its ranges as cropped then
            (volume_cropper.snapped_ranges_at); the views draw it where it
            is at the timepoint shown.
        """
        self._box_ranges, self.selected = box_ranges, selected
        self._source_boxes = boxes
        self._boxes_time = self.time_shown()
        self.boxes = self._boxes_at(self._boxes_time)
        moves = list(box_moves or [])
        self._box_moves = (moves + [None] * len(box_ranges))[:len(box_ranges)]
        self._moved = {}
        self._time_followed = self.time_shown()
        if self.image is None:
            return
        self._dash_zoom = {}      # outlines may have changed: new dashes
        # (0.18.0) While a dot is dragged in 3D, only what follows the drag
        # live: the box lines (below) and the 3D frames; the shapes you
        # edit in 2D, the 3D clipping and masking, shading and preview
        # follow once, on the release (then _drag_3d is None again).
        dragging = self._drag_3d is not None
        if not dragging:
            for name in VIEWS_2D:
                self._sync_shapes(self.views[name])
        if self.show_3d:
            with self._display_only():
                self._sync_box_layers(self.views["3D"])
                if not dragging:
                    self._update_clipping()
                    self._update_masking()
                    self._sync_surfaces()                     # (0.18.0)
        # "selected box" projections follow the box; this also redraws the
        # crosshairs and the box lines
        self._update_slabs()
        if not dragging:
            self._update_shading()

    @property
    def box_ranges(self) -> list:
        """Per box: its ranges as cropped, or None if the box is invalid. A
        region that moves over time (0.17.0) has them where it is at the
        timepoint shown; everything the views draw of a box (lines, 3D
        frame, clipping, masking, shading, preview, "selected box"
        projections, Zoom to region) reads them here."""
        if not any(self._box_moves):
            return self._box_ranges
        time = self.time_shown()
        if time is None:
            return self._box_ranges
        if time not in self._moved:
            self._moved = {time: [
                ranges if move is None or ranges is None else move(time)
                for ranges, move in zip(self._box_ranges, self._box_moves)]}
        return self._moved[time]

    def _boxes_at(self, time) -> list:
        """The panel's boxes as the views draw them at a time: the same
        list, unless a box has outlines per timepoint (0.17.0); then every
        such box as it is then (its outlines then, ranges_at)."""
        boxes = self._source_boxes
        if time is None or not any(getattr(b, "timed_outlines", None)
                                   for b in boxes):
            return boxes
        return [vc.box_at(b, time) if b.timed_outlines else b for b in boxes]

    def time_shown(self) -> float | None:
        """The time of the timepoint shown (level 0's frame centre, in the
        time axis' unit), or None for an image without a time axis."""
        t_axis = None if self.image is None else vc.time_axis(self.image)
        if t_axis is None:
            return None
        value = self.cursor.get(self.image.axis_names[t_axis])
        if value is None:
            return None
        level0 = self.image.levels[0]
        origin, interval = level0.translation[t_axis], level0.scale[t_axis]
        n = level0.array.shape[t_axis]
        index = min(max(int(round((value - origin) / interval)), 0), n - 1)
        return origin + index * interval

    def requested_ranges(self, i: int) -> dict:
        """Box i's ranges as requested (not snapped) at the timepoint shown:
        a moving region's (0.17.0) where it is then (the rectangle that is
        edited, the panel's number fields)."""
        box = self._source_boxes[i] if i < len(self._source_boxes) \
            else self.boxes[i]
        time = self.time_shown()
        if getattr(box, "track", None) and time is not None:
            return vc.ranges_at(box, time)
        return box.ranges

    def _follow_time(self):
        """After a change of the timepoint shown: regions that move over
        time (0.17.0) get their rectangles, 3D frames, clipping and masking
        where they are now (their lines, preview and shading are redrawn
        anyway), and on_time_changed is called."""
        time = self.time_shown()
        if time == self._time_followed:
            return
        self._time_followed = time
        if any(self._box_moves) or self._source_boxes is not self.boxes:
            with self._display_only():
                for name in VIEWS_2D:
                    if self.views[name].shapes is not None:
                        self._sync_shapes(self.views[name])
                if self.show_3d:
                    self._sync_box_layers(self.views["3D"])
                    self._update_clipping()
                    self._update_masking()
                    self._sync_surfaces()                     # (0.18.0)
        if self.seeds and self.seed_style is not None:
            self._draw_seeds()            # seeds follow their regions
        if self.on_time_changed is not None:
            self.on_time_changed()

    def set_shading(self, on: bool | None = None, kept: str | None = None,
                    zeroed: str | None = None, opacity: float | None = None):
        """Shade the selected box in the 2D views: the voxels that will be
        kept in one colour, the rest of the box (set to 0) in another
        (colours as "#rrggbb"; opacity 0-1)."""
        for key, value in (("on", on), ("kept", kept), ("zeroed", zeroed),
                           ("opacity", opacity)):
            if value is not None:
                self.shading[key] = value
        self._update_shading()

    def _update_shading(self):
        """Redraw the per-slice overlays of the selected box in the 2D
        views: the shading and the crop preview."""
        if self.image is None:
            return
        with self._display_only():
            for name in VIEWS_2D:
                self._shade(self.views[name])
                self._preview(self.views[name])
        self._update_crop_projection()                        # (0.18.0)

    def _shade(self, view: View):
        """The shading of one 2D view: the selected box on the current slice
        (see slice_mask)."""
        layer, i = view.shading, self.selected
        if layer is None:
            return
        ranges = (self.box_ranges[i] if 0 <= i < len(self.box_ranges)
                  else None)
        v, h = view.plane
        d = view.order[-3]
        position = self.cursor.get(d)
        if not (self.shading["on"] and self.show_boxes and ranges is not None
                and self._box_drawn(i) and position is not None
                and self.projection_range(view) is None
                and ranges[d][0] <= position <= ranges[d][1]):
            set_visible(layer, False)
            return
        keep, origin, step = slice_mask(
            self.boxes[i], vc._spatial_axes(self.image),
            vc.pixel_size(self.image), (v, h), position, ranges)
        alpha = int(round(255 * self.shading["opacity"]))
        rgba = np.zeros(keep.shape + (4,), dtype=np.uint8)
        rgba[keep] = (*hex_to_rgb(self.shading["kept"]), alpha)
        rgba[~keep] = (*hex_to_rgb(self.shading["zeroed"]), alpha)
        layer.data = rgba
        layer.scale = (step, step)
        layer.translate = origin
        set_visible(layer, True)

    # --- crop preview in the 2D views ------------------------------------------

    def set_preview(self, on: bool):
        """Preview the selected box's crop in the 2D views: everything the
        crop does not contain on the slice shown is black (outside the box,
        the voxels its outlines set to 0, and the whole view when the slice
        or timepoint lies outside the box). On a projection (0.18.0) the
        views show the projection of exactly what the crop keeps, over the
        box, and black elsewhere (see _update_crop_projection)."""
        self.preview = bool(on)
        self._preview_keys = {}
        self._update_shading()

    def _preview(self, view: View):
        """The crop preview of one 2D view (see set_preview)."""
        inside, outside = view.preview_inside, view.preview_outside
        if inside is None or outside is None:
            return
        i = self.selected
        ranges = (self.box_ranges[i] if self.preview
                  and 0 <= i < len(self.box_ranges) else None)
        if ranges is None:
            set_visible(inside, False)
            set_visible(outside, False)
            self._preview_keys.pop(view.name, None)
            return
        v, h = view.plane
        d = view.order[-3]
        slab = self.projection_range(view)
        # (a timepoint skipped by "every n-th" is not in the crop either)
        here = (slice_in_crop(ranges, view.plane, d, self.cursor, slab)
                and self.time_kept(self.selected))
        if self._crop_projection_on():
            # (0.18.0) a projection: all black, and the crop's own
            # projection over the region (_show_crop_projection)
            here = False
        rects = preview_rectangles(ranges, view.plane,
                                   (self.extent(v), self.extent(h)), here)
        # Inside the box: the voxels the outlines set to 0 on this slice, on
        # a grid of the level shown (not on projections: see set_preview).
        grid = None
        if here and slab is None and self.boxes[i].outlines:
            names = self.image.axis_names
            scale = self.image.levels[self.current_level(view)].scale
            voxel = min(scale[names.index(v)], scale[names.index(h)])
            longer = max(ranges[v][1] - ranges[v][0],
                         ranges[h][1] - ranges[h][0])
            points = int(min(PREVIEW_POINTS, max(np.ceil(longer / voxel), 1)))
            grid = (self.cursor[d], points)
        key = (rects, grid, i, id(self.boxes[i]), repr(self.boxes[i].outlines),
               self.boxes[i].margin)
        if self._preview_keys.get(view.name) == key:
            return                       # nothing changed: keep the layers
        self._preview_keys[view.name] = key
        outside.data = []
        outside.add_rectangles([np.array(r, dtype=float) for r in rects],
                               edge_width=0, face_color="black")
        set_visible(outside, True)
        set_visible(inside, False)
        if grid is not None:
            keep, origin, step = slice_mask(
                self.boxes[i], vc._spatial_axes(self.image),
                vc.pixel_size(self.image), (v, h), grid[0], ranges, grid[1])
            if not keep.all():
                rgba = np.zeros(keep.shape + (4,), dtype=np.uint8)
                rgba[~keep, 3] = 255          # black where set to 0
                inside.data = rgba
                inside.scale = (step, step)
                inside.translate = origin
                set_visible(inside, True)

    # --- the crop preview's projection (0.18.0) ------------------------------
    # With the preview on and a projection chosen, each 2D view shows the
    # projection of exactly what the selected box's crop keeps (only its
    # depth, its outlines' zeroed voxels as 0; crop_projection.py) over the
    # box, and black elsewhere (_preview). It is read in a thread of its
    # own, CROP_PROJECTION_DELAY_MS after the last change, one job at a
    # time; the views keep the previous one until the new one is ready.

    def _crop_projection_on(self) -> bool:
        """Whether the 2D views show the crop's own projection."""
        i = self.selected
        return (self.preview and self.image is not None
                and self.projection in crop_projection.MODES
                and 0 <= i < min(len(self.box_ranges), len(self.boxes))
                and self.box_ranges[i] is not None and self.time_kept(i))

    def _crop_projection_key(self):
        """What the crop's projection depends on; None when not shown."""
        if not self._crop_projection_on():
            return None
        i = self.selected
        box = self.boxes[i]
        return (id(self.image), i, repr(self.box_ranges[i]),
                repr(box.outlines), box.margin, self.time_shown(),
                self.projection)

    def _crop_channels_missing(self) -> list[int]:
        """The channels shown whose crop projection is not computed yet."""
        shown = [k for k, d in enumerate(self.display) if d.get("visible")]
        if self._crop_shown != self._crop_key or self._crop_result is None:
            return shown
        depth = self._crop_result.axes[0]
        return [k for k in shown
                if (k, depth) not in self._crop_result.planes]

    def _update_crop_projection(self):
        """Hide the crop's projection when it is not wanted; otherwise show
        what is computed, and compute what is missing (after a pause)."""
        key = self._crop_projection_key()
        if key is None:
            self._crop_key = None
            self._crop_timer.stop()
            self._show_crop_projection()
            return
        if key != self._crop_key:
            self._crop_key = key
            self._crop_timer.start()
        elif self._crop_channels_missing() and self._crop_job is None:
            self._crop_timer.start()
        self._show_crop_projection()

    def _start_crop_projection(self):
        """Compute the crop's projection of the channels missing, in the
        background (one job at a time: _poll_crop_projection starts the
        next)."""
        key = self._crop_key
        if key is None or self._crop_job is not None:
            return
        channels = self._crop_channels_missing()
        if not channels:
            self._show_crop_projection()
            return
        i = self.selected
        ranges, box = self.box_ranges[i], self.boxes[i]
        level = crop_projection.projection_level(self.image, self.region,
                                                 ranges)
        if self._crop_executor is None:
            self._crop_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="crop-projection")
        future = self._crop_executor.submit(
            crop_projection.crop_projection, self.image, self.region,
            channels, self.time_shown(), ranges, box, self.projection,
            level=level)
        self._crop_job = (key, future)
        self._tell_crop(f"Preview: computing the projection of {box.name}'s "
                        f"crop (level {level})…")
        self._crop_poll.start()

    def _poll_crop_projection(self):
        """Show a finished projection (if still wanted), then start the
        next one if something changed meanwhile."""
        if self._crop_job is None:
            self._crop_poll.stop()
            return
        key, future = self._crop_job
        if not future.done():
            return
        self._crop_poll.stop()
        self._crop_job = None
        try:
            result = future.result()
        except Exception as err:            # e.g. a file no longer readable
            self._tell_crop(f"Preview: could not compute the crop's "
                            f"projection: {err}")
            return
        if key == self._crop_key:
            if result is None:
                self._crop_result, self._crop_shown = None, None
            elif self._crop_shown == key and self._crop_result is not None:
                self._crop_result.planes.update(result.planes)  # channels
            else:
                self._crop_result, self._crop_shown = result, key
            self._show_crop_projection()
        if self._crop_key is not None and (self._crop_key != key
                                           or self._crop_channels_missing()):
            self._crop_timer.start()

    def _show_crop_projection(self):
        """Put the computed projection into the 2D views' layers (one per
        channel, made when first needed), or hide them."""
        result = self._crop_result
        on = (self._crop_key is not None and result is not None
              and self._crop_shown is not None)
        if self.image is None:
            return
        with self._display_only():
            for name in VIEWS_2D:
                view = self.views[name]
                depth = view.order[-3]
                for k in range(len(self.display)):
                    layer = view.crop_projection_layers.get(k)
                    ready = on and (k, depth) in result.planes
                    if ready:
                        layer = self._crop_layer(view, k)
                        self._fill_crop_layer(layer, result, k, depth,
                                              view.plane)
                    visible = ready and bool(self.display[k]["visible"])
                    if layer is not None and layer.visible != visible:
                        layer.visible = visible
        if self._crop_key is None:
            self._tell_crop("")
        elif on and self._crop_job is None:
            i = self.selected
            n = int(np.prod(list(result.shape.values())))
            self._tell_crop(
                f"Preview: projection of {self.boxes[i].name}'s crop, at "
                f"level {result.level} ({n:,} voxels per channel).")

    def _crop_layer(self, view: View, k: int) -> Image:
        """The crop projection layer of channel k in a 2D view: above the
        preview's black, with the channel's colours (blended additively,
        like the channels)."""
        layer = view.crop_projection_layers.get(k)
        if layer is None:
            settings = self.display[k]
            layer = Image(
                np.zeros((1, 1), dtype=np.float32),
                name=f"crop projection {k}", visible=False,
                metadata=dict(HELPER), blending="additive",
                contrast_limits=settings["contrast_limits"],
                gamma=settings["gamma"], colormap=settings["colormap"],
                units=layer_units(self.image, list(view.plane)),
                axis_labels=list(view.plane))
            above = view.model.layers.index(view.preview_outside) + 1
            # napari makes a layer it adds the active one; drawing needs
            # the regions' layer to stay active (the tools act on it)
            selection = view.model.layers.selection
            active = selection.active
            view.model.layers.insert(above, layer)
            if active is not None:
                selection.active = active
            else:
                selection.clear()
            view.crop_projection_layers[k] = layer
        return layer

    def _fill_crop_layer(self, layer: Image, result, k: int, depth: str,
                         plane):
        """Give a crop projection layer its data, unless it has it."""
        stored = result.planes[(k, depth)]
        if layer.metadata.get("vc_plane") is stored:
            return
        v, h = plane
        layer.data = result.plane(k, depth, plane)
        layer.scale = (result.scale[v], result.scale[h])
        layer.translate = (result.first[v], result.first[h])
        layer.contrast_limits = self.display[k]["contrast_limits"]
        layer.metadata["vc_plane"] = stored

    def _tell_crop(self, text: str):
        if self.on_crop_projection is not None:
            self.on_crop_projection(text)

    def zoom_to_selected(self):
        """Fit the 2D views to the selected box (one zoom for all), and move
        slices that lie outside it to its middle; (0.18.0) with the 3D view
        on, centre it on the box and fit the box (its angle stays)."""
        i = self.selected
        if self.image is None or not 0 <= i < len(self.box_ranges) \
                or self.box_ranges[i] is None:
            return
        ranges = self.box_ranges[i]
        spatial = self.views["XY"].order[-3:]
        moved = {a: float(np.mean(ranges[a])) for a in spatial
                 if not ranges[a][0] <= self.cursor.get(a, np.nan)
                 <= ranges[a][1]}
        if moved:
            self._set_cursor(moved)
        self._center = {a: float(np.mean(ranges[a])) for a in spatial}
        zooms = []
        for name in VIEWS_2D:
            view = self.views[name]
            pixels = min(view.model.canvas.size)
            size = max(ranges[a][1] - ranges[a][0] for a in view.plane)
            if pixels > 10 and size > 0:
                zooms.append(0.9 * pixels / size)
        if zooms:
            self._apply_camera(min(zooms), source=None)
        view = self.views["3D"]
        if self.show_3d and view.channel_layers:
            order = view.order[-3:]
            cam = camera(view.model)
            cam.center = tuple(float(np.mean(ranges[a])) for a in order)
            # the box's diagonal fits whichever way the view is turned
            diagonal = float(np.sqrt(sum((ranges[a][1] - ranges[a][0]) ** 2
                                         for a in order)))
            pixels = min(view.model.canvas.size)
            if diagonal > 0 and pixels > 10:
                cam.zoom = 0.9 * pixels / diagonal

    def set_hidden_boxes(self, names):
        """Hide single boxes (by name) in all views; they are still cropped.
        set_boxes_visible shows or hides all boxes at once."""
        self.hidden_boxes = set(names)
        self.update_boxes(self._source_boxes, self._box_ranges, self.selected,
                          self._box_moves)

    def _box_drawn(self, i: int) -> bool:
        """Whether box i is drawn (unless all boxes are switched off)."""
        return self.boxes[i].name not in self.hidden_boxes

    # --- slices of the views ------------------------------------------------------

    def polygon_names(self, box: vc.CropBox) -> dict:
        """polygon_names with the views' names ("xy1", ...)."""
        def prefix(plane):
            view = self.view_for(plane)
            return view.name.lower() if view is not None else "".join(plane)
        return polygon_names(box, prefix)

    def _shape_label(self, i: int, plane, key) -> str:
        """A shape's label in a view: the box's name, and for a polygon its
        generic name too ("crop01 xy2")."""
        box = self.boxes[i]
        if key == RECT:
            return box.name
        stored = tuple(plane) if tuple(plane) in box.outlines \
            else (plane[1], plane[0])
        name = self.polygon_names(box).get((stored, key))
        return box.name if name is None else f"{box.name} {name}"

    def delete_selected_shapes(self):
        """(0.18.0, the panel's "Delete shape") Delete the shapes selected
        in the 2D views, as napari's own delete does: their removal reaches
        the panel through _on_shapes (on_boxes_removed)."""
        for name in VIEWS_2D:
            shapes = self.views[name].shapes
            if shapes is not None and shapes.selected_data:
                shapes.remove_selected()

    def set_tool(self, mode: str):
        """Choose the drawing tool of the 2D views ("add_polygon",
        "pan_zoom", ...); they all follow."""
        for name in VIEWS_2D:
            shapes = self.views[name].shapes
            if shapes is not None:
                shapes.mode = mode

    def view_for(self, plane) -> View | None:
        """The 2D view that shows this plane (in either order)."""
        for name in VIEWS_2D:
            if set(self.views[name].plane) == set(plane):
                return self.views[name]
        return None

    def slice_of(self, plane) -> float | None:
        """The current slice (µm along its depth axis) of the view that
        shows this plane: where a polygon drawn there now belongs."""
        view = self.view_for(plane)
        return None if view is None else self.cursor.get(view.order[-3])

    def go_to_slice(self, plane, depth: float):
        """Move the view that shows this plane to the slice `depth`."""
        view = self.view_for(plane)
        if view is not None:
            self._set_cursor({view.order[-3]: float(depth)})

    def slice_tolerance(self, axis: str) -> float:
        """Polygons this close (µm) to a view's slice count as being on it:
        half a voxel of the finest level."""
        i = self.image.axis_names.index(axis)
        return 0.5 * self.image.levels[0].scale[i]

    def _shape_plan(self, view: View) -> list:
        """What a view's editable shapes are on its current slice: per box
        drawn, (box, key, points). A box's polygon in this view (the only one,
        or the one on this slice) is edited as that polygon; otherwise the
        box is edited as its rectangle (key RECT, points None)."""
        axis = view.order[-3]
        depth = self.cursor.get(axis)
        tolerance = self.slice_tolerance(axis)
        projected = self.projection_range(view)
        plan = []
        for i in range(len(self.boxes)):
            if not self._box_drawn(i):
                continue
            box = self.boxes[i]
            if projected is not None:      # a projection: its silhouette
                found = next(((k, pts) for k, pts in silhouettes_in_view(
                    box, view.plane, projected)
                    if same_range(k, projected, tolerance)), None)
                if found is None and reaches_through(box, view.plane):
                    found = next(iter(keys_in_plane(box, view.plane).items()))
            else:
                found = polygon_on_slice(box, view.plane, depth, tolerance)
            plan.append((i, RECT, None) if found is None
                        else (i, found[0], found[1]))
        return plan

    def _refresh_slice_shapes(self):
        """After a slice change: rebuild the shapes of views where a box's
        polygon on the slice appeared or went (most slider moves change
        nothing, and then nothing is rebuilt)."""
        for name in VIEWS_2D:
            view = self.views[name]
            if view.shapes is None:
                continue
            plan = self._shape_plan(view)
            if [(i, key) for i, key, _ in plan] != list(
                    zip(view.shape_boxes, view.shape_keys)):
                self._sync_shapes(view, plan)

    def _sync_shapes(self, view: View, plan: list | None = None):
        if plan is None:
            plan = self._shape_plan(view)
        self._syncing["shapes"] = True
        try:
            layer = view.shapes
            layer.selected_data = set()
            layer.data = []
            # Hidden boxes get no shape, so that they cannot be picked or
            # dragged by mistake; shape_boxes and shape_keys say which box
            # (and which of its polygons) each shape is.
            view.shape_boxes = [i for i, _, _ in plan]
            view.shape_keys = [key for _, key, _ in plan]
            if plan:
                # The shapes themselves are invisible: the box lines layers
                # draw the boxes (solid and dashed), and napari still
                # outlines a shape when it is selected or under the mouse.
                # New shapes are drawn in the next box colour while they are
                # being drawn.
                shapes, types = [], []
                for i, key, points in plan:
                    if key == RECT:
                        shapes.append(rectangle(self.requested_ranges(i),
                                                view.plane))
                        types.append("rectangle")
                    else:
                        shapes.append(np.asarray(points, dtype=float))
                        types.append("polygon")
                layer.add(shapes, shape_type=types, edge_color="transparent",
                          face_color="transparent")
            layer.features = {"name": [
                self._shape_label(i, view.plane, key)
                for i, key in zip(view.shape_boxes, view.shape_keys)]}
            layer.text = {"string": "{name}", "color": "white",
                          "anchor": "upper_left", "size": 10}
            layer.current_edge_color = BOX_COLORS[len(self.boxes)
                                                  % len(BOX_COLORS)]
        finally:
            self._syncing["shapes"] = False
        self._update_widths(view)

    def _sync_box_layers(self, view: View):
        """3D wireframes: one invisible 1-voxel image per box, scaled to the
        box as cropped, whose bounding box napari draws as lines."""
        spatial = view.order[-3:]
        while len(view.box_layers) > len(self.boxes):
            self._remove_layers(view, [view.box_layers.pop()])
        while len(view.box_layers) < len(self.boxes):
            # No depth testing (see set_image): the wireframe must not be
            # hidden behind the data (or the invisible box volume).
            layer = Image(np.ones((1, 1, 1), dtype=np.uint8),
                          contrast_limits=(0, 1), opacity=0.0,
                          blending="translucent_no_depth",
                          units=layer_units(self.image, spatial),
                          axis_labels=list(spatial))
            layer.bounding_box.visible = True
            layer.bounding_box.points = False
            layer.bounding_box.blending = "translucent_no_depth"
            # above the image layers
            view.model.layers.insert(
                len(view.channel_layers) + len(view.mask_layers), layer)
            view.box_layers.append(layer)
        for i, layer in enumerate(view.box_layers):
            ranges = self.box_ranges[i]
            if ranges is None:
                set_visible(layer, False)
                continue
            low = np.array([ranges[a][0] for a in spatial])
            high = np.array([ranges[a][1] for a in spatial])
            color = BOX_COLORS[i % len(BOX_COLORS)]
            layer.scale = high - low
            layer.translate = (low + high) / 2   # voxel centre = box centre
            layer.colormap = colormap(color)
            # faded at timepoints the crop does not keep (see _fade_3d_frames)
            layer.bounding_box.line_color = rgba(
                color, 1.0 if self.time_kept(i) else FADED_ALPHA)
            layer.bounding_box.line_thickness = 4 if i == self.selected else 2
            layer.name = defang(f"box {self.boxes[i].name}")
            set_visible(layer, self.show_boxes and self._box_drawn(i))
        if view.name == "3D" and self._drag_3d is None:
            self._sync_handles(view)                          # (0.18.0)
        if view.shapes is not None:
            view.model.layers.selection.active = view.shapes

    # --- the selected box's dots in 3D (0.18.0) ------------------------------
    # A dot in the middle of each face of the selected box's wireframe, and
    # one in its centre (box_handles.py). Dragging a face's dot moves that
    # face along its axis; the centre's moves the box. The camera stays
    # still meanwhile; elsewhere the mouse turns the view as before.

    def _handles_shown(self) -> bool:
        i = self.selected
        return (self.image is not None and self.show_3d and self.show_boxes
                and 0 <= i < min(len(self.box_ranges), len(self.boxes))
                and self.box_ranges[i] is not None and self._box_drawn(i))

    def _sync_handles(self, view: View, ranges: dict | None = None):
        """Put the dots on the selected box (or on ranges, while dragging);
        made when first needed, hidden without a selected box."""
        shown = self._handles_shown()
        if view.handles is None:
            if not shown:
                return
            spatial = view.order[-3:]
            # Their size in µm follows the 3D zoom (_size_3d_markers), so
            # they keep HANDLE_PIXELS on screen. (Not with napari's canvas
            # size limits: a huge size clipped by them made the home button
            # zoom out to nothing, as napari counts point sizes in the
            # scene's extent; a tiny one raised by them drew the markers
            # mostly in their border colour.)
            view.handles = Points(
                np.zeros((7, 3)), ndim=3, name="region handles",
                metadata=dict(HELPER), size=self._marker_size(HANDLE_PIXELS),
                border_width=0.15, border_color="white",
                blending="translucent_no_depth", visible=False,
                units=layer_units(self.image, spatial),
                axis_labels=list(spatial))
            # above everything; napari makes a layer it adds the active one
            selection = view.model.layers.selection
            active = selection.active
            view.model.layers.append(view.handles)
            if active is not None:
                selection.active = active
            else:
                selection.clear()
        if not shown:
            if view.handles.visible:
                set_visible(view.handles, False)
            return
        i = self.selected
        ranges = ranges or self.box_ranges[i]
        points = box_handles.handle_points(ranges, view.order[-3:])
        color = BOX_COLORS[i % len(BOX_COLORS)]
        data = np.array(list(points.values()))
        if not np.array_equal(view.handles.data, data):
            view.handles.data = data
        # colours and symbols only when they change (each change redraws;
        # setting data keeps them for the same number of dots)
        if view.handles.metadata.get("vc_colour") != color:
            view.handles.face_color = ["white" if key == box_handles.CENTRE
                                       else color for key in points]
            view.handles.symbol = ["diamond" if key == box_handles.CENTRE
                                   else "disc" for key in points]
            view.handles.metadata["vc_colour"] = color
        set_visible(view.handles, True)

    def _handle_drag(self, viewer, event):
        """The 3D view's mouse handler. A press on a dot of the selected
        box drags it (see box_handles.py); with Shift a face's drag resizes
        the box symmetrically (the centre stays), with Ctrl it snaps to
        round values (box_handles.snap_step), also for the centre. A click
        (press and release without moving) on another box's wireframe
        selects that box. Anything else is left to the camera. A generator,
        as napari's drag handlers are: the code before the first `yield`
        runs on the press, the loop on each move, the rest on the
        release."""
        view = self.views["3D"]
        modifiers = modifier_names(event.modifiers)
        if (event.button != 1 or modifiers - {"Shift", "Control"}
                or self.image is None or len(event.dims_displayed) != 3
                or event.view_direction is None):
            return
        shown = list(event.dims_displayed)
        spatial = [view.order[d] for d in shown]
        extents = {a: self.extent(a) for a in spatial}

        def ray():
            return (np.asarray(event.position, float)[shown],
                    np.asarray(event.view_direction, float)[shown])

        origin, direction = ray()
        zoom = float(event.camera_zoom)
        key = None
        if (view.handles is not None and view.handles.visible
                and self._handles_shown()):
            i = self.selected
            # The dots sit on the box as cropped (`drawn`, snapped to the
            # voxel grid and, along axes its polygons limit, fitted to
            # them). A face dragged goes where the mouse is; the faces not
            # dragged keep their values as the panel holds them (`held`,
            # at the timepoint shown).
            drawn = dict(self.box_ranges[i])
            points = box_handles.handle_points(drawn, spatial)
            key = box_handles.pick_handle(points, origin, direction, zoom)
        if key is None:
            if not modifiers:            # a click may select a box
                yield from self._click_to_select(event, origin, direction,
                                                 zoom, spatial)
            return                       # (meanwhile the camera turns)
        source = (self._source_boxes[i] if i < len(self._source_boxes)
                  else self.boxes[i])
        held = vc.ranges_at(source, self.time_shown())
        for axis in spatial:            # an axis left out: the whole extent
            held.setdefault(axis, self.extent(axis))
        step = box_handles.snap_step(extents)
        if key != box_handles.CENTRE:
            self._last_face = (i, *key)       # for Up/Down (_nudge)
        if key == box_handles.CENTRE:
            grip = box_handles.on_plane(origin, direction, points[key],
                                        direction)
            normal = direction
        else:
            axis, side = key
            grip = box_handles.along_axis(points[key], spatial.index(axis),
                                          origin, direction)
        if grip is None:
            return
        cam = camera(view.model)
        locked = (cam.mouse_pan, cam.mouse_zoom)
        cam.mouse_pan = cam.mouse_zoom = False      # no turning while dragging
        self._drag_3d = key
        moved, applied = None, 0.0       # what to tell the panel; when last
        try:
            yield
            while event.type == "mouse_move":
                origin, direction = ray()
                mods = modifier_names(event.modifiers)   # may change
                if key == box_handles.CENTRE:
                    here = box_handles.on_plane(origin, direction,
                                                points[key], normal)
                    if here is not None:
                        wanted = {a: float(here[j] - grip[j])
                                  for j, a in enumerate(spatial)}
                        if "Control" in mods:          # whole steps
                            wanted = {a: box_handles.snapped(s, step)
                                      for a, s in wanted.items()}
                        shown_box = box_handles.move_region(drawn, wanted,
                                                            extents)
                        # as far as the box as drawn can go, everywhere
                        shift = {a: shown_box[a][0] - drawn[a][0]
                                 for a in spatial}
                        moved = ({a: (held[a][0] + shift[a],
                                      held[a][1] + shift[a])
                                  for a in spatial}, shift)
                        self._show_dragged(view, i, shown_box)
                else:
                    u = box_handles.along_axis(
                        points[key], spatial.index(axis), origin, direction)
                    if u is not None:
                        value = drawn[axis][side] + (u - grip)
                        if "Control" in mods:          # a round value
                            value = box_handles.snapped(value, step)
                        voxel = self._voxel_size(axis)
                        resize = (box_handles.resize_symmetric
                                  if "Shift" in mods
                                  else box_handles.move_face)
                        self._show_dragged(view, i, resize(
                            drawn, axis, side, value, extents[axis], voxel))
                        new = resize(held, axis, side, value, extents[axis],
                                     voxel)
                        moved = ({axis: new[axis]}, None)
                        # (the same value each time: safe to apply as it
                        # goes; a move of the centre only on the release)
                        if time.time() - applied > HANDLE_APPLY_S:
                            applied = time.time()
                            self._apply_handles(i, *moved)
                yield
        finally:
            cam.mouse_pan, cam.mouse_zoom = locked
            self._drag_3d = None
            if moved is not None:
                self._apply_handles(i, *moved)
            self._sync_handles(view)

    def _show_dragged(self, view: View, i: int, ranges: dict):
        """Move the dots and the wireframe with the mouse at once (the
        panel hears of a face's drag at most every HANDLE_APPLY_S, of the
        centre's on the release); with "Clip 3D to selected region" the
        clipping follows too (moving clipping planes costs little)."""
        self._sync_handles(view, ranges)
        spatial = view.order[-3:]
        low = np.array([ranges[a][0] for a in spatial])
        high = np.array([ranges[a][1] for a in spatial])
        if i < len(view.box_layers):
            layer = view.box_layers[i]
            layer.scale = high - low
            layer.translate = (low + high) / 2
        if self.clip_3d:
            planes = clipping_planes(list(low), list(high))
            for layer in [*view.channel_layers, *view.mask_layers]:
                layer.experimental_clipping_planes = planes

    def _click_to_select(self, event, origin, direction, zoom: float,
                         spatial):
        """(A sub-generator of _handle_drag) A press that is released
        without moving: the box whose wireframe it hit is selected. One
        that moves turns the camera, as usual."""
        pressed = np.asarray(event.pos, dtype=float)
        moved = False
        yield
        while event.type == "mouse_move":
            if np.hypot(*(np.asarray(event.pos, float) - pressed)) \
                    > CLICK_SLOP:
                moved = True
            yield
        if moved or self.on_box_picked is None or not self.show_boxes:
            return
        # (the selected box is left out: where its frame crosses another's
        # on screen, a click means the other one)
        regions = {j: ranges for j, ranges in enumerate(self.box_ranges)
                   if ranges is not None and self._box_drawn(j)
                   and j != self.selected}
        j = box_handles.pick_region(regions, origin, direction, zoom,
                                    spatial)
        if j is not None and j != self.selected:
            self.on_box_picked(j)

    def _nudge_key(self, steps: int):
        """A key function for napari: _nudge by steps. napari's key
        handling reads every bound function's __name__ (a
        functools.partial has none, and then every key press in the view
        failed: found by the visible check), so a named function."""
        def nudge(viewer=None):
            self._nudge(steps)
        nudge.__name__ = f"volume_cropper_nudge_{steps}"
        return nudge

    def _nudge(self, steps: int):
        """(Up/Down in the 3D view) Move the face of the selected box last
        dragged (or pressed) by `steps` voxels of the finest level:
        outward for steps > 0, inward for steps < 0."""
        face = getattr(self, "_last_face", None)
        if (face is None or face[0] != self.selected
                or not self._handles_shown()):
            return
        i, axis, side = face
        source = (self._source_boxes[i] if i < len(self._source_boxes)
                  else self.boxes[i])
        held = vc.ranges_at(source, self.time_shown())
        held.setdefault(axis, self.extent(axis))
        voxel = self._voxel_size(axis)
        value = held[axis][side] + (1 if side == 1 else -1) * steps * voxel
        new = box_handles.move_face(held, axis, side, value,
                                    self.extent(axis), voxel)
        self._apply_handles(i, {axis: new[axis]}, None)

    def _apply_handles(self, i: int, ranges: dict, shift: dict | None):
        """Tell the panel the box's new ranges (of the axis dragged, or all
        three for the centre) and, for the centre, how far its polygons
        move with it."""
        if self.on_box_ranges is not None:
            self.on_box_ranges(i, ranges, shift)

    def _voxel_size(self, axis: str) -> float:
        """The finest voxel size along an axis (level 0), in µm."""
        return float(self.image.levels[0].scale[
            self.image.axis_names.index(axis)])

    def _sync_outlines(self, view: View):
        """Draw the boxes in a 2D view.

        Solid line: the box as it will be cropped. Dashed line: the box's
        polygon/lasso outline in this view's plane. A box that the current
        slice (or projected slab) does not cut is drawn faded. Sizes are in
        screen pixels, so the lines are rebuilt when the zoom changes.
        """
        if not view.outline_layers or self.image is None:
            return
        zoom = self._zoom(view)
        depth = view.order[-3]
        slab = self.projection_range(view)
        if slab is None:
            slab = (self.cursor.get(depth, 0.0),) * 2
        pieces = {"boxes": [], "selected": []}     # (segments, colours)
        for i, ranges in enumerate(self.box_ranges[:len(self.boxes)]):
            if ranges is None or not self._box_drawn(i):
                continue
            low, high = ranges[depth]
            # faded where the crop has nothing: the slice (or slab) misses
            # the box, or the timepoint shown is not kept (0.11.0)
            cut = slab[0] <= high and slab[1] >= low and self.time_kept(i)
            alpha = 1.0 if cut else FADED_ALPHA
            segments = [polyline_segments(rectangle(ranges, view.plane))]
            for line, closed in self._outlines_on_slice(self.boxes[i], ranges,
                                                        view):
                segments.append(dash_segments(line, DASH / zoom, GAP / zoom,
                                              closed=closed))
            segments = np.concatenate(segments)
            color = rgba(BOX_COLORS[i % len(BOX_COLORS)], alpha)
            colors = np.tile(color, (len(segments), 1))
            if i == self.selected:
                # × marks in the colour of the view whose polygon they come
                # from (0.16.0), as the view titles and crosshairs
                marks, mark_colors = self._crossing_marks(self.boxes[i],
                                                          view, zoom)
                if len(marks):
                    segments = np.concatenate([segments, marks])
                    colors = np.concatenate([colors, mark_colors])
            which = "selected" if i == self.selected else "boxes"
            pieces[which].append((segments, colors))
        for which, layer in view.outline_layers.items():
            # napari wants each piece as [start, direction]
            if pieces[which]:
                segments = np.concatenate([s for s, _ in pieces[which]])
                colors = np.concatenate([c for _, c in pieces[which]])
                data = np.stack([segments[:, 0],
                                 segments[:, 1] - segments[:, 0]], axis=1)
            else:
                data, colors = np.zeros((0, 2, 2)), np.zeros((0, 4))
            width = LINE_WIDTH * (2 if which == "selected" else 1) / zoom
            # Hidden while empty: napari 0.9.2 fails to draw an empty
            # Vectors layer in a 3D-displayed view. Also hidden when the
            # boxes are switched off.
            visible = bool(len(data)) and self.show_boxes
            # Only touch the layer if something changed: this runs on every
            # slider move, and each change makes napari redraw.
            if data.shape == layer.data.shape and np.allclose(data, layer.data) \
                    and np.allclose(colors, np.asarray(layer.edge_color)) \
                    and np.isclose(width, layer.edge_width) \
                    and layer.visible == visible:
                continue
            with self._display_only():
                layer.data = data
                if len(data):
                    layer.edge_color = colors
                layer.edge_width = width          # in µm, like the data
                set_visible(layer, visible)
        self._dash_zoom[view.name] = zoom

    def _crossing_marks(self, box: vc.CropBox, view: View,
                        zoom: float) -> tuple[np.ndarray, np.ndarray]:
        """Small crosses where the box's polygons of the other views cross
        this view's slice: there, an outline drawn here should meet them
        (none on a projection). Returns (segments, colours): each cross in
        the colour of the view its polygon was drawn in (0.16.0).

        (0.20.0) None while switched off (View: Show × marks), and the marks
        of each view thinned to MARK_SPACING on screen (thin_points): a
        view's marks never hide another view's."""
        none = (np.zeros((0, 2, 2)), np.zeros((0, 4)))
        position = self.cursor.get(view.order[-3])
        if not self.show_marks or position is None \
                or self.projection_range(view) is not None \
                or not box.outlines:
            return none
        points, planes = vc.crossing_points(
            box.outlines, vc._spatial_axes(self.image), tuple(view.plane),
            position, with_planes=True)
        if not len(points):
            return none
        planes = np.array([" ".join(plane) for plane in planes])
        keep = []
        for plane in np.unique(planes):
            mine = np.flatnonzero(planes == plane)
            keep.append(mine[thin_points(points[mine], MARK_SPACING / zoom)])
        keep = np.sort(np.concatenate(keep))
        points, planes = points[keep], planes[keep]
        colors = []
        for plane in planes:
            source = self.view_for(tuple(plane.split(" ")))
            colors.append(rgba(PLANE_COLORS[source.name] if source is not None
                               else PLANE_COLORS["3D"], 1.0))
        colors = np.asarray(colors)
        size = MARK / zoom
        pieces = []
        for dv, dh in ((1, 1), (1, -1)):
            offset = np.array([dv, dh]) * size
            pieces.append(np.stack([points - offset, points + offset],
                                   axis=1))
        return np.concatenate(pieces), np.concatenate([colors, colors])

    def _outlines_on_slice(self, box: vc.CropBox, ranges: dict,
                           view: View) -> list:
        """The dashed lines of a box in a view on its current slice, as
        (points, closed): the polygon that the view shows there (see
        polygon_on_slice), and the edge of what the outlines keep on this
        slice (see slice_outline). The latter is left out where it is just
        that polygon again: on a polygon's own slice (the shape passes
        through it), or for a box whose only outline is that polygon, both
        without a margin."""
        axis = view.order[-3]
        position = self.cursor.get(axis)
        if position is None or not box.outlines:
            return []
        projected = self.projection_range(view)
        if projected is not None:
            # a projection shows the silhouettes drawn on it (and a polygon
            # through the whole box), not a slice of the shape
            lines = [(np.asarray(points, dtype=float), True) for _, points in
                     silhouettes_in_view(box, view.plane, projected)]
            if reaches_through(box, view.plane):
                lines += [(np.asarray(points, dtype=float), True) for points
                          in keys_in_plane(box, view.plane).values()]
            return lines
        found = polygon_on_slice(box, view.plane, position,
                                 self.slice_tolerance(axis))
        lines = [] if found is None else [
            (np.asarray(found[1], dtype=float), True)]
        if found is not None and not box.margin:
            through = vc._through_box(box.outlines)
            sliced = vc._sliced_polygons(box.outlines,
                                         vc._spatial_axes(self.image))
            if not through or (len(through) == 1 and len(sliced) <= 1):
                return lines
        if not ranges[axis][0] <= position <= ranges[axis][1]:
            return lines
        lines += [(line, False) for line in slice_outline(
            box, vc._spatial_axes(self.image), vc.pixel_size(self.image),
            view.plane, position, ranges) if len(line) >= 2]
        return lines

    def _on_shapes(self, view: View, event):
        """Shapes drawn, edited or deleted in a 2D view.

        Rectangles (and other shapes) set the box's range in this plane;
        polygons (polygon or lasso tool) also become the box's outline here.
        """
        if self._syncing["shapes"] or self.image is None:
            return
        extents = {a: self.image.extent(a) for a in view.plane}
        # A moving region (0.17.0) may stick out of the image at some
        # timepoints: its rectangle is not clipped there (the 0.17.0 review:
        # a small drag clipped it, and the smaller size went to all its
        # timepoints).
        unclipped = {a: (-np.inf, np.inf) for a in view.plane}
        data = view.shapes.data
        types = view.shapes.shape_type

        def drawn(i, clip=True):
            outline = None
            if types[i] == "polygon":
                outline = [tuple(float(v) for v in point)
                           for point in np.asarray(data[i])[:, -2:]]
            return rectangle_ranges(data[i], view.plane,
                                    extents if clip else unclipped), outline

        def moves(k):
            return k < len(self._source_boxes) and bool(
                getattr(self._source_boxes[k], "track", None))

        plane = view.plane
        boxes = view.shape_boxes        # the box of each shape drawn by us
        keys = view.shape_keys          # and which of its polygons (or RECT)
        if event.action == ActionType.ADDED:
            new = [drawn(i) for i in range(len(boxes), len(data))]
            callback = lambda: [self.on_box_added(r, plane, o)  # noqa: E731
                                for r, o in new]
        elif event.action == ActionType.CHANGED:
            changes = {boxes[i]: (*drawn(i, clip=not moves(boxes[i])),
                                  keys[i])
                       for i in event.data_indices
                       if 0 <= i < min(len(data), len(boxes))}
            callback = lambda: self.on_boxes_changed(changes, plane)  # noqa: E731
        elif event.action == ActionType.REMOVING:
            items = sorted((boxes[i], keys[i]) for i in event.data_indices
                           if 0 <= i < len(boxes))
            callback = lambda: self.on_boxes_removed(items, plane)  # noqa: E731
        else:
            return
        # Run after napari has finished handling the mouse event.
        QTimer.singleShot(0, callback)

    def _on_mode(self, view: View, event=None):
        """Use the same drawing tool (rectangle, select, ...) in all views."""
        if self._syncing["mode"]:
            return
        self._syncing["mode"] = True
        try:
            for name in VIEWS_2D:
                other = self.views[name].shapes
                if other is not None and other is not view.shapes:
                    other.mode = view.shapes.mode
        finally:
            self._syncing["mode"] = False
        if self.on_tool_changed is not None:
            self.on_tool_changed(str(view.shapes.mode))

    # --- cursor, crosshairs and the J key -----------------------------------------

    def _set_cursor(self, values: dict, source: View | None = None):
        """Move the shared cursor; other views follow with their sliders."""
        self.cursor.update(values)
        # (0.17.0) outlines per timepoint: the boxes as they are now, before
        # anything is drawn
        if self._source_boxes is not self.boxes \
                and self.time_shown() != self._boxes_time:
            self._boxes_time = self.time_shown()
            self.boxes = self._boxes_at(self._boxes_time)
            self._dash_zoom = {}          # other outlines: new dashes
        self._syncing["dims"] = True
        try:
            for view in self.views.values():
                if view is source or not view.order:
                    continue
                dims = view.model.dims
                for i in dims.not_displayed:
                    axis = view.order[i] if i < len(view.order) else None
                    if axis in values and dims.point[i] != values[axis]:
                        dims.set_point(i, values[axis])
        finally:
            self._syncing["dims"] = False
        self._update_slabs()             # also redraws the crosshairs
        if self.image is not None:       # polygons and shading per slice
            with self._display_only():
                self._refresh_slice_shapes()
            self._update_shading()
            self._follow_time()          # moving regions (0.17.0)
            if self.level_outline is not None:
                self.draw_level_outline()    # (0.18.0) on the new slice
            if self.gap_lines:
                self.draw_gap_lines()        # (0.18.0) faded off their slice

    def _on_dims(self, view: View, event=None):
        if self._syncing["dims"] or self.image is None or not view.order:
            return
        dims = view.model.dims
        values = {view.order[i]: float(dims.point[i])
                  for i in dims.not_displayed if i < len(view.order)}
        self._set_cursor(values, source=view)

    def _jump_key(self, view: View):
        """The J-key handler for one view.

        It must be a real, named function: napari reads the names of all
        key handlers on every key press, and a functools.partial has no
        name, which broke every shortcut in the view.
        """
        def jump_to_mouse(viewer):
            self._jump(view)
        return jump_to_mouse

    def _jump(self, view: View, viewer=None):
        """J key: move the other views to the point under the mouse."""
        position = view.model.cursor.position
        if self.image is None or len(position) < 2:
            return
        v, h = view.plane
        self._set_cursor({v: float(position[-2]), h: float(position[-1])})

    # --- seeds (0.16.0, "Autocropper") ---------------------------------

    def set_seeds(self, seeds: list[dict]):
        """Show these seeds ({axis: value}) in the 2D views: a cross at
        each, whatever the slice and timepoint shown (a seed marks an
        object; seeing all of them shows which objects have one)."""
        self.seeds = [dict(seed) for seed in seeds]
        self._draw_seeds()

    def seeds_shown(self) -> list:
        """Per seed: (point, rgba) as drawn at the timepoint shown (see
        seed_style)."""
        time = self.time_shown()
        if self.seed_style is None:
            return [(seed, rgba(SEED_COLOR)) for seed in self.seeds]
        return [self.seed_style(k, seed, time)
                for k, seed in enumerate(self.seeds)]

    def _draw_seeds(self):
        shown = self.seeds_shown()
        with self._display_only():
            for name in VIEWS_2D:
                view = self.views[name]
                if view.seeds is None:
                    continue
                v, h = view.plane
                drawn = [((p[v], p[h]), color) for p, color in shown
                         if v in p and h in p]
                view.seeds.data = np.array([d[0] for d in drawn],
                                           dtype=float).reshape(-1, 2)
                if drawn:
                    view.seeds.face_color = np.array([d[1] for d in drawn])
                    view.seeds.size = SEED_SIZE / self._zoom(view)
                set_visible(view.seeds, bool(drawn))
            self._draw_seeds_3d(shown)

    # --- the Autocropper's own threshold (0.18.0) ----------------------

    def set_level_outline(self, source):
        """Outline what lies above the Autocropper's own threshold in the 2D
        views (source: see level_outline; None hides it); also call it
        after the threshold changed, to redraw."""
        self.level_outline = source
        self.draw_level_outline()

    def draw_level_outline(self):
        """Draw the threshold outline on each 2D view's slice (or through
        its projected slab), as a dotted line in screen pixels (rebuilt
        when the zoom changes, like the region lines)."""
        for name in VIEWS_2D:
            self._draw_level_lines(self.views[name])

    def _draw_level_lines(self, view: View):
        layer = view.level_lines
        if layer is None or self.image is None:
            return
        lines = []
        if self.level_outline is not None and view.order:
            depth = view.order[-3]
            slab = self.projection_range(view)
            if slab is None:
                slab = (self.cursor.get(depth, 0.0),) * 2
            lines = self.level_outline(view.plane, depth, slab)
        zoom = self._zoom(view)
        pieces = [dash_segments(line, THRESHOLD_DASH / zoom,
                                THRESHOLD_GAP / zoom, closed=False)
                  for line in lines if len(line) > 1]
        pieces = [p for p in pieces if len(p)]
        if pieces:
            segments = np.concatenate(pieces)
            data = np.stack([segments[:, 0],
                             segments[:, 1] - segments[:, 0]], axis=1)
        else:
            data = np.zeros((0, 2, 2))
        with self._display_only():
            layer.data = data
            if len(data):
                layer.edge_color = THRESHOLD_COLOR
            layer.edge_width = LINE_WIDTH / zoom
            # (hidden while empty: see _sync_outlines)
            set_visible(layer, bool(len(data)))
        view.level_zoom = zoom

    # --- the Autocropper's gap lines (0.18.0) ---------------------------

    def set_gap_lines(self, lines: list):
        """Show these gap lines (see gap_lines) in the 2D views they were
        drawn in."""
        self.gap_lines = [dict(line) for line in lines]
        self.draw_gap_lines()

    def set_gap_mode(self, on: bool):
        """While on, a drag in a 2D view draws a gap line instead of
        panning (scrolling still zooms)."""
        self.gap_mode = bool(on)
        for name in VIEWS_2D:
            camera(self.views[name].model).mouse_pan = not on
        if not on:
            self._live_gap = None
            self.draw_gap_lines()

    def gap_on_slice(self, line: dict) -> bool:
        """Whether a gap line lies on the slice (and timepoint) shown."""
        depth = line["depth"]
        if abs(line["at"] - self.cursor.get(depth, line["at"])) > \
                self.slice_tolerance(depth):
            return False
        time = self.time_shown()
        return (line.get("time") is None or time is None
                or abs(line["time"] - time) < 1e-9)

    def gap_near(self, point: dict, view_name: str,
                 pixels: float = GAP_PICK) -> int | None:
        """The index of the gap line on this view's slice nearest to a
        point, within `pixels` screen pixels (else None)."""
        view = self.views[view_name]
        v, h = view.plane
        zoom = self._zoom(view)
        best, best_distance = None, pixels
        for k, line in enumerate(self.gap_lines):
            if line["view"] != view_name or not self.gap_on_slice(line):
                continue
            distance = zoom * segment_distance((point[v], point[h]),
                                               line["points"])
            if distance <= best_distance:
                best, best_distance = k, distance
        return best

    def draw_gap_lines(self):
        for name in VIEWS_2D:
            self._draw_gap_view(self.views[name])

    def _draw_gap_view(self, view: View):
        if view.gap_line is None or self.image is None:
            return
        zoom = self._zoom(view)
        lines = [(line["points"], 1.0 if self.gap_on_slice(line)
                  else FADED_ALPHA)
                 for line in self.gap_lines if line["view"] == view.name]
        if self._live_gap is not None and self._live_gap[0] == view.name:
            lines.append((self._live_gap[1], 1.0))
        pattern = [length / zoom for length in GAP_PATTERN]
        for layer, solid, color, width in (
                (view.gap_edge, True, SEED_EDGE, GAP_EDGE_WIDTH),
                (view.gap_line, False, GAP_COLOR, LINE_WIDTH)):
            pieces, colors = [], []
            for points, alpha in lines:
                points = np.asarray(points, dtype=float).reshape(-1, 2)
                if len(points) < 2:
                    continue
                segments = (polyline_segments(points, closed=False) if solid
                            else pattern_segments(points, pattern))
                if len(segments):
                    pieces.append(segments)
                    colors.append(np.tile(rgba(color, alpha),
                                          (len(segments), 1)))
            if pieces:
                segments = np.concatenate(pieces)
                data = np.stack([segments[:, 0],
                                 segments[:, 1] - segments[:, 0]], axis=1)
                colors = np.concatenate(colors)
            else:
                data, colors = np.zeros((0, 2, 2)), np.zeros((0, 4))
            with self._display_only():
                layer.data = data
                if len(data):
                    layer.edge_color = colors
                layer.edge_width = width / zoom
                set_visible(layer, bool(len(data)))
        view.gap_zoom = zoom

    def _gap_draw(self, view: View):
        """The mouse handler of a 2D view that draws gap lines while
        gap_mode is on (a generator, see _seed_click): the press starts a
        line at the slice and timepoint shown, each move adds a point (at
        least 2 screen pixels on), the release ends it (on_gap_line); a
        press that hardly moves is a click (on_gap_click)."""
        def gap_draw(viewer, event):
            if (not self.gap_mode or self.image is None or event.button != 1
                    or event.modifiers or view.shapes is None
                    or str(view.shapes.mode) != "pan_zoom"
                    or len(event.position) < 2):
                return
            pressed = np.asarray(event.pos, dtype=float)
            last = pressed
            points = [tuple(float(p) for p in event.position[-2:])]
            moved = False
            self._live_gap = (view.name, points)
            yield
            while event.type == "mouse_move":
                here = np.asarray(event.pos, dtype=float)
                if np.hypot(*(here - pressed)) > CLICK_SLOP:
                    moved = True
                if np.hypot(*(here - last)) >= 2.0:
                    points.append(tuple(float(p)
                                        for p in event.position[-2:]))
                    last = here
                    self._draw_gap_view(view)
                yield
            self._live_gap = None
            v, h = view.plane
            depth = view.order[-3]
            if not moved:
                self._draw_gap_view(view)
                if self.on_gap_click is not None:
                    point = dict(self.cursor)
                    point[v], point[h] = points[0]
                    self.on_gap_click(point, view.name)
                return
            if len(points) < 2 or self.on_gap_line is None:
                self._draw_gap_view(view)
                return
            self.on_gap_line({
                "view": view.name, "plane": (v, h), "depth": depth,
                "at": float(self.cursor.get(depth, 0.0)),
                "time": self.time_shown(), "points": points,
                "width": float(self.gap_width)})
        return gap_draw

    # --- regions' outlines in 3D (0.18.0) ------------------------------

    def set_outlines_3d(self, mode: str | None = None,
                        opacity: float | None = None):
        """Show the regions' outlines in 3D: mode "off", "selected" or
        "all"; opacity of the surfaces (0-1)."""
        if mode is not None:
            self.outlines_3d = mode
        if opacity is not None:
            self.outlines_3d_opacity = float(opacity)
        self._sync_surfaces()

    def outline_mesh(self, i: int) -> tuple | None:
        """Region i's outlines in 3D as (vertices (µm, in the 3D view's
        spatial order), faces), or None: the surface round what its crop
        keeps (vc.outline_mask, as the 3D dimming computes it) on the 3D
        view's level, lightly smoothed, by marching cubes."""
        from scipy import ndimage as ndi
        from skimage.measure import marching_cubes
        box = self.boxes[i] if i < len(self.boxes) else None
        ranges = self.box_ranges[i] if i < len(self.box_ranges) else None
        if box is None or ranges is None or not box.outlines:
            return None
        view = self.views["3D"]
        level = self._level_3d_used
        key = (id(self.image), level, repr(box.outlines), float(box.margin),
               tuple(sorted((a, tuple(r)) for a, r in ranges.items())))
        if key in self._meshes:
            return self._meshes[key]
        lv = self.image.levels[level]
        offset = self.region.translation(self.image, level)
        shape = self.region.shape(level)
        spatial = view.order[-3:]
        index = [self.image.axis_names.index(a) for a in spatial]
        scale = [lv.scale[j] for j in index]
        origin = [offset[j] for j in index]
        size = [shape[j] for j in index]
        slices = _box_slices(spatial, scale, origin, size, ranges)
        mesh = None
        if slices is not None:
            first = [b.start for b in slices]
            extent = tuple(b.stop - b.start for b in slices)
            keep = vc.outline_mask(
                vc.Level("", None, list(scale), list(origin)), list(spatial),
                first, extent, box.outlines,
                spatial=vc._spatial_axes(self.image),
                margin=box.margin * vc.pixel_size(self.image),
                detail=VIEW_DETAIL)
            keep = np.broadcast_to(keep, extent) if keep is not None \
                else np.ones(extent, bool)
            if keep.any() and min(extent) > 0:
                field_ = ndi.gaussian_filter(
                    np.pad(keep, 2).astype(np.float32), 0.7)
                step = 2 if keep.size > OUTLINE_3D_MAX_VOXELS else 1
                try:
                    vertices, faces, _, _ = marching_cubes(
                        field_, 0.5, spacing=tuple(scale), step_size=step)
                    corner = np.array([o + (f - 2) * sc for o, f, sc in
                                       zip(origin, first, scale)])
                    mesh = (vertices + corner, faces)
                except (ValueError, RuntimeError):
                    mesh = None
        if len(self._meshes) > 32:
            self._meshes.clear()
        self._meshes[key] = mesh
        return mesh

    def _sync_surfaces(self):
        """The 3D view's outline surfaces, as outlines_3d says."""
        view = self.views["3D"]
        shown = []
        if (self.image is not None and self.show_3d and view.channel_layers
                and self.show_boxes and self.outlines_3d != "off"
                and self._drag_3d is None):
            wanted = (range(len(self.boxes)) if self.outlines_3d == "all"
                      else [self.selected] if self.selected >= 0 else [])
            for i in wanted:
                if i < len(self.boxes) and self._box_drawn(i):
                    mesh = self.outline_mesh(i)
                    if mesh is not None:
                        shown.append((i, mesh))
        while len(view.surface_layers) > len(shown):
            self._remove_layers(view, [view.surface_layers.pop()])
        spatial = view.order[-3:]
        while len(view.surface_layers) < len(shown):
            layer = Surface(
                (np.zeros((3, 3)), np.array([[0, 1, 2]])), shading="smooth",
                blending="translucent", metadata=dict(HELPER),
                units=layer_units(self.image, spatial),
                axis_labels=list(spatial))
            view.model.layers.insert(
                len(view.channel_layers) + len(view.mask_layers)
                + len(view.box_layers), layer)
            view.surface_layers.append(layer)
        for layer, (i, (vertices, faces)) in zip(view.surface_layers, shown):
            color = rgba(BOX_COLORS[i % len(BOX_COLORS)], 1.0)
            # (colours of the old mesh first gone: napari checks their
            # number against the new mesh's vertices)
            layer.vertex_colors = None
            layer.data = (vertices, faces)
            layer.vertex_colors = np.tile(color, (len(vertices), 1))
            layer.opacity = self.outlines_3d_opacity
            layer.name = defang(f"outline {self.boxes[i].name}")
            set_visible(layer, True)
        if view.shapes is not None:
            view.model.layers.selection.active = view.shapes

    def _draw_seeds_3d(self, shown: list):
        """(0.18.0, the user's wish) The seeds in the 3D view too, as in 2D
        (a cross in its region's colour, dark-edged), at a fixed size on
        screen: their size in µm follows the 3D zoom (_size_3d_markers;
        see _sync_handles for why not napari's canvas size limits). Made
        when first needed; above the image, drawn through it."""
        view = self.views["3D"]
        if self.image is None or not self.show_3d:
            return
        spatial = view.order[-3:]
        drawn = [([p[a] for a in spatial], color) for p, color in shown
                 if all(a in p for a in spatial)]
        if view.seeds is None:
            if not drawn:
                return
            view.seeds = Points(
                np.zeros((0, 3)), ndim=3, name="seeds", symbol="cross",
                metadata=dict(HELPER), face_color=SEED_COLOR,
                size=self._marker_size(SEED_SIZE),
                border_width=0.18, border_color=SEED_EDGE, visible=False,
                blending="translucent_no_depth",
                units=layer_units(self.image, spatial),
                axis_labels=list(spatial))
            selection = view.model.layers.selection
            active = selection.active       # napari activates what it adds
            view.model.layers.append(view.seeds)
            if active is not None:
                selection.active = active
            else:
                selection.clear()
        view.seeds.data = np.array([d[0] for d in drawn],
                                   dtype=float).reshape(-1, 3)
        if drawn:
            view.seeds.face_color = np.array([d[1] for d in drawn])
            # (points added to an empty layer get napari's default size,
            # not the one it was made with)
            view.seeds.size = self._marker_size(SEED_SIZE)
        set_visible(view.seeds, bool(drawn))

    def _marker_size(self, pixels: float) -> float:
        """A marker's size in µm that is `pixels` wide on the 3D view's
        screen at its zoom now (pixels per µm; orthographic view)."""
        return pixels / max(camera(self.views["3D"].model).zoom, 1e-12)

    def _size_3d_markers(self, event=None):
        """The 3D view zoomed: its seeds and region dots keep their size
        on screen (as the 2D views' seeds do)."""
        view = self.views["3D"]
        for layer, pixels in ((view.seeds, SEED_SIZE),
                              (view.handles, HANDLE_PIXELS)):
            if layer is not None and len(layer.data):
                layer.size = self._marker_size(pixels)

    def _seed_click(self, view: View):
        """The mouse handler of a 2D view that reports seed clicks (see
        on_seed_click). A generator, as napari's drag handlers are: the code
        before the first `yield` runs on the press, the loop on each move,
        the rest on the release. A press that moves the mouse is a pan."""
        def seed_click(viewer, event):
            if (self.on_seed_click is None or self.image is None
                    or event.button != 1 or event.modifiers
                    or view.shapes is None
                    or str(view.shapes.mode) != "pan_zoom"):
                return
            pressed = np.asarray(event.pos, dtype=float)
            position = [float(p) for p in event.position]
            moved = False
            yield
            while event.type == "mouse_move":
                if np.hypot(*(np.asarray(event.pos, float) - pressed)) \
                        > CLICK_SLOP:
                    moved = True
                yield
            if moved or self.on_seed_click is None or len(position) < 2:
                return
            v, h = view.plane
            point = {axis: float(value) for axis, value in self.cursor.items()}
            point[v], point[h] = position[-2], position[-1]
            # A double-click zooms (napari): its first click must not add
            # or remove a seed, so a click counts only once the time for a
            # second one has passed (review 2026-11-08)
            delay = (QApplication.doubleClickInterval()
                     if self.seed_delay_ms is None else self.seed_delay_ms)
            if delay <= 0:
                self.on_seed_click(point, view.name)
                return
            token = object()
            self._pending_seed = token

            def fire():
                if self._pending_seed is token and self.on_seed_click:
                    self._pending_seed = None
                    self.on_seed_click(point, view.name)
            QTimer.singleShot(delay, fire)
        return seed_click

    def _cancel_seed(self, viewer=None, event=None):
        """A double-click (napari zooms): the click before it is no seed."""
        self._pending_seed = None

    def seed_near(self, point: dict, view_name: str,
                  pixels: float = SEED_SIZE) -> int | None:
        """The index of the seed nearest to a point in a view, if it is
        within `pixels` screen pixels there (else None)."""
        view = self.views[view_name]
        v, h = view.plane
        zoom = self._zoom(view)
        best, best_distance = None, pixels
        for k, (seed, _) in enumerate(self.seeds_shown()):  # (as drawn)
            if v in seed and h in seed:
                distance = zoom * np.hypot(seed[v] - point[v],
                                           seed[h] - point[h])
                if distance <= best_distance:
                    best, best_distance = k, distance
        return best

    @contextmanager
    def _display_only(self):
        """Use while changing display layers (lines, outlines).

        napari re-sends the sliders' positions when a layer's size changes.
        Without this, those events would count as the user moving a slider,
        and redraw everything again, many times over.
        """
        before = self._syncing["dims"]
        self._syncing["dims"] = True
        try:
            yield
        finally:
            self._syncing["dims"] = before

    def _update_crosshairs(self):
        """Redraw the crosshairs, and the box lines (which fade when the
        slice leaves a box)."""
        if self.image is None:
            return
        with self._display_only():
            self._draw_crosshairs()

    def time_kept(self, i: int) -> bool:
        """Whether box i's crop keeps the timepoint shown: inside its t
        range and, with every n-th timepoint, one of those kept (0.11.0).
        True for images without a time axis."""
        t_axis = None if self.image is None else vc.time_axis(self.image)
        if t_axis is None or not 0 <= i < min(len(self.box_ranges),
                                              len(self.boxes)):
            return True
        name = self.image.axis_names[t_axis]
        ranges = self.box_ranges[i]
        if ranges is None or name not in ranges or name not in self.cursor:
            return True
        low, high = ranges[name]           # outer edges of the kept frames
        t = self.cursor[name]
        if not low <= t <= high:
            return False
        step = getattr(self.boxes[i], "t_step", 1)
        if step <= 1:
            return True
        interval = self.image.levels[0].scale[t_axis]
        first = low + interval / 2         # the first kept timepoint
        return int(round((t - first) / interval)) % step == 0

    def _fade_3d_frames(self):
        """The 3D wireframes fade like the 2D lines at timepoints a box's
        crop does not keep (only changed when needed: this runs on every
        slider move)."""
        for i, layer in enumerate(self.views["3D"].box_layers):
            if i >= len(self.boxes):
                break
            color = rgba(BOX_COLORS[i % len(BOX_COLORS)],
                         1.0 if self.time_kept(i) else FADED_ALPHA)
            if not np.allclose(np.asarray(layer.bounding_box.line_color),
                               color):
                layer.bounding_box.line_color = color

    def _draw_crosshairs(self):
        """See _update_crosshairs (which calls this inside _display_only)."""
        for name in VIEWS_2D:
            self._sync_outlines(self.views[name])
        self._fade_3d_frames()
        # A line at "axis = value" marks the plane whose slider is that axis.
        z, y, x = self.views["XY"].order[-3:]
        plane_of = {z: "XY", y: "XZ", x: "YZ"}
        for name in VIEWS_2D:
            view = self.views[name]
            if view.crosshair is None:
                continue
            v, h = view.plane
            (v0, v1), (h0, h1) = self.extent(v), self.extent(h)
            # (0.18.0) only when they move: redrawing costs ~60 ms a view
            key = (self.cursor[v], self.cursor[h], v0, v1, h0, h1,
                   self._zoom(view))
            if view.crosshair.metadata.get("vc_drawn") == key:
                continue
            lines = [np.array([[self.cursor[v], h0], [self.cursor[v], h1]]),
                     np.array([[v0, self.cursor[h]], [v1, self.cursor[h]]])]
            colors = [PLANE_COLORS[plane_of[v]], PLANE_COLORS[plane_of[h]]]
            view.crosshair.data = []
            view.crosshair.add_lines(lines, edge_color=colors,
                                     edge_width=1.5 / self._zoom(view))
            view.crosshair.metadata["vc_drawn"] = key
        self._update_3d_planes()

    # --- the 2D views' planes, outlined in the 3D view ------------------------

    def set_planes_3d(self, show: bool):
        self.show_planes_3d = show
        self._update_3d_planes()

    def set_crosshairs(self, show: bool):
        """Show or hide the crosshair lines in the 2D views."""
        self.show_crosshairs = show
        with self._display_only():
            for name in VIEWS_2D:
                if self.views[name].crosshair is not None:
                    set_visible(self.views[name].crosshair, show)

    def set_marks(self, show: bool):
        """(0.20.0) Show or hide the selected box's × marks in the 2D views
        (where its polygons in the other views cross the slice)."""
        self.show_marks = show
        with self._display_only():
            for name in VIEWS_2D:
                self._sync_outlines(self.views[name])

    def set_boxes_visible(self, show: bool):
        """Show or hide the crop boxes in all views: their lines and names
        in 2D (the shapes you edit, too) and their wireframes in 3D."""
        self.show_boxes = show
        with self._display_only():
            for name in VIEWS_2D:
                view = self.views[name]
                if view.shapes is not None:
                    set_visible(view.shapes, show)
                self._sync_outlines(view)
            for i, layer in enumerate(self.views["3D"].box_layers):
                valid = i < len(self.box_ranges) \
                    and self.box_ranges[i] is not None
                set_visible(layer, show and valid and self._box_drawn(i))
        self._update_shading()

    def _build_3d_planes(self, view: View):
        """One flat 1-voxel layer per 2D view, whose wireframe outlines that
        view's plane (or projected slab) in 3D. Same technique as the boxes."""
        spatial = view.order[-3:]
        view.plane_layers = {}
        for name in VIEWS_2D:
            layer = Image(np.ones((1, 1, 1), dtype=np.uint8),
                          contrast_limits=(0, 1), opacity=0.0,
                          blending="translucent_no_depth",
                          name=f"{name} plane",
                          units=layer_units(self.image, spatial),
                          axis_labels=list(spatial))
            layer.bounding_box.visible = True
            layer.bounding_box.points = False
            layer.bounding_box.blending = "translucent_no_depth"
            layer.bounding_box.line_color = PLANE_COLORS[name]
            layer.bounding_box.line_thickness = 1.5
            view.model.layers.append(layer)
            view.plane_layers[name] = layer

    def _update_3d_planes(self):
        """Place the outlines: the plane at the cursor, or the slab that the
        2D view projects through when a projection is on."""
        view = self.views["3D"]
        if self.image is None or not view.plane_layers:
            return
        spatial = view.order[-3:]
        depth_of = dict(zip(VIEWS_2D, spatial))     # XY: z, XZ: y, YZ: x
        for name, layer in view.plane_layers.items():
            low, high = [], []
            for axis in spatial:
                extent = self.extent(axis)
                if axis != depth_of[name]:
                    span = extent
                else:
                    span = self.projection_range(self.views[name])
                    if span is None:          # a single plane: nearly flat
                        half = 1e-3 * (extent[1] - extent[0])
                        span = (self.cursor[axis] - half,
                                self.cursor[axis] + half)
                low.append(span[0])
                high.append(span[1])
            low, high = np.array(low), np.array(high)
            with self._display_only():
                # (0.18.0) only when they move: each change redraws
                if not (np.allclose(layer.scale, high - low)
                        and np.allclose(layer.translate, (low + high) / 2)):
                    layer.scale = high - low
                    layer.translate = (low + high) / 2
                set_visible(layer, self.show_planes_3d)

    # --- linked zoom and pan --------------------------------------------------------

    def _zoom(self, view: View) -> float:
        return max(camera(view.model).zoom, 1e-12)

    def _on_camera(self, view: View, event=None):
        self._update_widths(view)
        if self._syncing["camera"] or not self.link_cameras or not view.order:
            return
        center = camera(view.model).center
        v, h = view.plane
        self._center[v], self._center[h] = float(center[-2]), float(center[-1])
        self._apply_camera(camera(view.model).zoom, source=view)

    def _apply_camera(self, zoom: float, source: View | None):
        self._syncing["camera"] = True
        try:
            for name in VIEWS_2D:
                view = self.views[name]
                if view is source:
                    continue
                cam = camera(view.model)
                v, h = view.plane
                cam.center = (*tuple(cam.center)[:-2],
                              self._center.get(v, cam.center[-2]),
                              self._center.get(h, cam.center[-1]))
                cam.zoom = zoom
                self._update_widths(view)
        finally:
            self._syncing["camera"] = False

    def _update_widths(self, view: View):
        """Keep lines about 2 screen pixels wide at any zoom."""
        zoom = self._zoom(view)
        width = 2 / zoom
        self._syncing["shapes"] = True
        try:
            if view.shapes is not None:
                view.shapes.current_edge_width = width    # while drawing
                if len(view.shapes.data):
                    boxes = view.shape_boxes
                    view.shapes.edge_width = [
                        2 * width if j < len(boxes) and boxes[j] == self.selected
                        else width for j in range(len(view.shapes.data))]
            if view.crosshair is not None and len(view.crosshair.data):
                view.crosshair.edge_width = 0.75 * width
            if view.seeds is not None and len(view.seeds.data):
                view.seeds.size = SEED_SIZE / zoom
        finally:
            self._syncing["shapes"] = False
        # Box lines and dashes: rebuilt when the zoom changes (not on pans)
        made_for = self._dash_zoom.get(view.name)
        if made_for is None or abs(zoom / made_for - 1) > 0.01:
            self._sync_outlines(view)
        if self.level_outline is not None and (
                view.level_zoom is None
                or abs(zoom / view.level_zoom - 1) > 0.01):
            self._draw_level_lines(view)                     # (0.18.0)
        if (self.gap_lines or self._live_gap) and (
                view.gap_zoom is None or abs(zoom / view.gap_zoom - 1) > 0.01):
            self._draw_gap_view(view)                        # (0.18.0)
