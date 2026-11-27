"""The crop preview's projection (0.18.0): a maximum or mean projection of
exactly what a crop region's crop keeps. No window needed.

The user asked on 2026-11-12 that, with "Preview crop" on, a projection
(MIP) show only the cropped volume: only the region's depth, and none of
the voxels its outlines set to 0. napari's own projection in the 2D views
goes through the image (through the range chosen under "2D projection"),
so with the preview on the views show this one over the region instead
(ortho_views.OrthoViews: "the crop preview's projection").

The region is read at the timepoint shown, at the finest pyramid level at
which one channel of it holds at most MAX_VOXELS voxels (levels.py's
rule), and the voxels its outlines set to 0 are 0, as in the crop. So the
projection is the crop's own: a mean projection counts those zeros, as a
mean projection of the saved crop in Fiji does.

In Fiji terms: like cropping (with Clear Outside for the polygons) and
then Image > Stacks > Z Project..., along each of the three axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from volume_cropper import levels as lv
from volume_cropper import core as vc

MAX_VOXELS = 16_000_000   # per channel: a level of the region at most this big
MODES = ("max", "mean")   # as napari's projection_mode


@dataclass
class CropProjection:
    """The projections of a crop region along each spatial axis.

    planes: {(channel, depth axis): 2D array}; the array's axes are the
    other two spatial axes, in the image's order. axes, scale, first: the
    spatial axes (image order), their voxel size and the centre of the
    region's first voxel at `level`, in µm.
    """

    level: int
    axes: list
    scale: dict
    first: dict
    shape: dict
    planes: dict = field(default_factory=dict)

    def plane(self, k: int, depth: str, view_axes) -> np.ndarray:
        """Channel k projected along depth, with its axes in the order of
        view_axes (a 2D view's (vertical, horizontal) axes)."""
        data = self.planes[(k, depth)]
        have = [a for a in self.axes if a != depth]
        return data if have == list(view_axes) else data.T


def projection_level(image: vc.OmeZarrImage, region, ranges: dict,
                     max_voxels: float = MAX_VOXELS) -> int:
    """The level the region is projected from (see the module's text)."""
    return lv.values_level(image, region, ranges, max_voxels)


def crop_projection(image: vc.OmeZarrImage, region, channels, time,
                    ranges: dict, box: vc.CropBox, mode: str = "max",
                    max_voxels: float = MAX_VOXELS,
                    level: int | None = None) -> CropProjection | None:
    """Project what box's crop keeps of these channels at a time (the
    timepoint shown) along each spatial axis.

    ranges: the region as it will be cropped (snapped), {axis: (low,
    high)} in µm; box: the region itself, for its outlines and margin;
    mode: "max" or "mean". None if no voxel lies inside."""
    if mode not in MODES:
        raise ValueError(f"projection mode {mode!r}: not one of {MODES}")
    if level is None:
        level = projection_level(image, region, ranges, max_voxels)
    result = None
    for k in channels:
        block = lv.read_block(image, region, k, level, time, ranges, box)
        if block is None:
            return None
        values = block.values
        if block.zeroed is not None:          # as the crop writes them
            values = np.where(block.zeroed, 0, values).astype(values.dtype)
        if result is None:
            result = CropProjection(
                level, list(block.axes),
                dict(zip(block.axes, block.scale)),
                dict(zip(block.axes, block.first)),
                dict(zip(block.axes, values.shape)))
        for i, depth in enumerate(block.axes):
            if mode == "max":
                plane = values.max(axis=i)
            else:
                plane = values.mean(axis=i, dtype=np.float64)
                plane = plane.astype(np.float32)
            result.planes[(k, depth)] = plane
    return result
