"""Tests for the geometry in ortho_views (needs the napari-env environment).

Run with pytest or with ``python test_ortho_views.py``. Only the pure
calculations are tested here; the views themselves are checked by driving
the GUI.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from volume_cropper import ortho_views as ov
from volume_cropper import core as vc
from test_volume_cropper import (L_SHAPE, inside_polygon, make_test_image,
                                 tube_and_outlines)


def _image(tmp, **kwargs):
    path = Path(tmp) / "src.zarr"
    make_test_image(path, **kwargs)
    return vc.open_image(path)


def test_view_orders():
    with tempfile.TemporaryDirectory() as tmp:
        orders = ov.view_orders(_image(tmp))
        assert orders == {"XY": ["z", "y", "x"], "XZ": ["y", "z", "x"],
                          "YZ": ["x", "y", "z"], "3D": ["z", "y", "x"]}
    with tempfile.TemporaryDirectory() as tmp:
        image = _image(tmp, shape0=(2, 2, 10, 16, 16), axes="tczyx",
                       factors=[(1, 1, 1, 1, 1)],
                       scale0=(30.0, 1.0, 2.0, 0.5, 0.5))
        orders = ov.view_orders(image)
        assert orders["XZ"] == ["t", "y", "z", "x"]   # t first in every view
        assert orders["YZ"] == ["t", "x", "y", "z"]
    with tempfile.TemporaryDirectory() as tmp:
        flat = _image(tmp, shape0=(2, 16, 16), axes="cyx",
                      factors=[(1, 1, 1)], scale0=(1.0, 0.5, 0.5))
        try:
            ov.view_orders(flat)
        except ValueError:
            pass
        else:
            raise AssertionError("a 2D image must be refused")


def test_pyramid_levels():
    assert ov.pyramid_levels(4) == [0, 1, 2, 3]                 # auto
    assert ov.pyramid_levels(4, 1) == [1, 2, 3]                 # cap
    assert ov.pyramid_levels(4, 1, exact=True) == [1]           # exact
    assert ov.pyramid_levels(4, 3) == [3]


def test_sizes():
    with tempfile.TemporaryDirectory() as tmp:
        image = _image(tmp)          # level 0: (3, 40, 150, 130), chunks 8
        # 3D "Auto": finest level within a memory budget (all 3 channels).
        # Level 0 is 3 * 40 * 150 * 130 * 2 bytes = 4.68 MB, level 1 0.30 MB.
        assert ov.auto_3d_level(image, max_bytes=5e6) == 0
        assert ov.auto_3d_level(image, max_bytes=1e6) == 1
        assert ov.auto_3d_level(image, max_bytes=5e6, max_side=100) == 1
        assert ov.auto_3d_level(image, max_bytes=1) == 3      # coarsest
        # Contrast preview: finest level with sides of at most max_size.
        assert ov.preview_level(image, max_size=256) == 0
        assert ov.preview_level(image, max_size=40) == 1    # (40, 38, 33)
        # XY plane: ceil(150/8) * ceil(130/8) = 19 * 17 chunks of 8x8x8
        # uint16 voxels, 3 channels.
        assert ov.plane_read_bytes(image, 0, ("y", "x")) == 19 * 17 * 512 * 2 * 3
        assert ov.plane_read_bytes(image, 0, ("z", "x")) == 5 * 17 * 512 * 2 * 3
        assert ov.volume_bytes(image, 0) == 40 * 150 * 130 * 2 * 3


def test_loaded_region():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        data0 = make_test_image(path)    # (3, 40, 150, 130), z 2, y/x 0.25 µm
        image = vc.open_image(path)
        # Default: the whole image, all levels, nothing cut.
        whole = ov.loaded_region(image)
        assert whole.whole and whole.n_levels == 4
        assert whole.ranges["z"] == image.extent("z")
        assert whole.shape(2) == image.levels[2].array.shape
        # Ranges that cover the whole image are the whole image.
        assert ov.loaded_region(image, {"z": (-50.0, 500.0)}).whole
        # z 20-40 µm, y 5-20 µm = level-0 voxels z 10-20, y 20-80. With all
        # four levels (min_size=8) the edges round outward to the coarsest
        # level's grid (factors z 4, y/x 16): z 8-24, y 16-96, x whole.
        region = ov.loaded_region(image, {"z": (20.0, 40.0), "y": (5.0, 20.0)},
                                  min_size=8)
        assert not region.whole and region.n_levels == 4
        assert region.starts[0] == [0, 8, 16, 0]
        assert region.stops[0] == [3, 24, 96, 130]
        assert region.ranges["z"] == (15.0, 47.0)       # outer voxel edges
        assert region.ranges["y"] == (3.875, 23.875)
        for k, f in enumerate([(1, 1, 1, 1), (1, 1, 4, 4), (1, 2, 8, 8),
                               (1, 4, 16, 16)]):
            # Every level's loaded part is the same voxels of the image...
            expected = data0[:, 8:24:f[1], 16:96:f[2], 0:130:f[3]]
            got = image.levels[k].array[region.slices(k)]
            np.testing.assert_array_equal(got, expected)
            assert region.shape(k) == expected.shape
            # ... starting at the same position, so napari overlays them.
            assert region.translation(image, k) == \
                region.translation(image, 0)
        # By default only the levels in which the region is at least 128
        # voxels long are used (here only level 0: x is 130 voxels long).
        small = ov.loaded_region(image, {"z": (20.0, 40.0)})
        assert small.n_levels == 1 and small.starts[0] == [0, 10, 0, 0]


def test_sizes_of_a_region():
    with tempfile.TemporaryDirectory() as tmp:
        image = _image(tmp)          # level 0: (3, 40, 150, 130), chunks 8
        region = ov.loaded_region(image, {"z": (20.0, 40.0), "y": (5.0, 20.0)},
                                  min_size=8)    # level 0: z 8-24, y 16-96
        voxels = 16 * 80 * 130
        assert ov.volume_bytes(image, 0, region, n_channels=1) == voxels * 2
        assert ov.volume_bytes(image, 0, region) == voxels * 2 * 3
        # XY plane: y voxels 16-96 touch chunks 2-11 (10), x all 17 chunks.
        chunk = 512 * 2                  # one 8x8x8 uint16 chunk
        assert ov.plane_read_bytes(image, 0, ("y", "x"), region, 1) \
            == 10 * 17 * chunk
        # XZ plane: z voxels 8-24 touch chunks 1 and 2.
        assert ov.plane_read_bytes(image, 0, ("z", "x"), region, 1) \
            == 2 * 17 * chunk
        # Projecting XY through the region's whole depth: 2 z chunks.
        assert ov.projection_read_bytes(image, 0, ("y", "x"), "z",
                                        region=region, n_channels=1) \
            == 2 * 10 * 17 * chunk
        # The smaller region fits a finer level into the same 3D budget.
        assert ov.auto_3d_level(image, max_bytes=1e6) == 1
        assert ov.auto_3d_level(image, max_bytes=1e6, region=region) == 0
        # ... and so do fewer channels.
        assert ov.auto_3d_level(image, max_bytes=1.6e6, n_channels=1) == 0
        # The contrast preview also keeps to a number of voxels.
        assert ov.preview_level(image, max_voxels=1e5) == 1      # 50160


def test_level_for_pixel_matches_napari():
    from napari.layers.utils.layer_utils import compute_multiscale_level
    with tempfile.TemporaryDirectory() as tmp:
        # Voxels: level 0 z 2 / y,x 0.25 µm; 1: 2 / 1; 2: 4 / 2; 3: 8 / 4.
        image = _image(tmp)
        canvas = np.array([300, 400])          # a view's size in pixels
        for axes in (["y", "x"], ["z", "x"], ["y", "z"]):
            index = [image.axis_names.index(a) for a in axes]
            for levels in ([0, 1, 2, 3], [1, 2, 3], [0, 1], [2]):
                scales = np.array([[image.levels[k].scale[i] for i in index]
                                   for k in levels])
                for pixel in (0.1, 0.3, 0.9, 1.5, 2.5, 4.5, 9.0, 20.0):
                    # napari's own rule, in its terms: the visible field in
                    # voxels of the first level, against the canvas size.
                    expected = levels[compute_multiscale_level(
                        canvas * pixel / scales[0], canvas,
                        scales / scales[0])]
                    got = ov.level_for_pixel(image, levels, axes, pixel)
                    assert got == expected, (axes, levels, pixel)


def test_read_estimate():
    with tempfile.TemporaryDirectory() as tmp:
        image = _image(tmp)
        region = ov.loaded_region(image)
        # Views of 20 x 20 pixels. Fitting the image (z 80, y 37.5, x 32.5
        # µm) into XZ and YZ needs 4.2 µm per pixel. XY then shows level 3
        # (4 µm voxels), but XZ and YZ need level 2, whose z voxels (4 µm)
        # are still smaller than a pixel: their z is coarser than x and y.
        sizes = {name: (20, 20) for name in ov.VIEWS_2D}
        levels = ov.fitted_levels(image, region, sizes)
        assert levels == {"XY": 3, "XZ": 2, "YZ": 2}
        assert ov.fitted_levels(image, region, sizes, level_2d=1,
                                exact=True) == {"XY": 1, "XZ": 1, "YZ": 1}
        orders = ov.view_orders(image)
        planes = sum(ov.plane_read_bytes(image, levels[n],
                                         tuple(orders[n][-2:]), region, 2)
                     for n in ov.VIEWS_2D)
        preview = ov.volume_bytes(image, ov.preview_level(image), region, 2)
        assert ov.open_read_bytes(image, region, 2, sizes) == planes + preview
        with_3d = ov.open_read_bytes(image, region, 2, sizes, show_3d=True)
        auto = ov.auto_3d_level(image, region=region, n_channels=2)
        assert with_3d == planes + preview + ov.volume_bytes(
            image, auto, region, 2)


def test_initial_display():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        data0 = make_test_image(path)
        image = vc.open_image(path)
        # Only channel 1 shown: only its contrast is measured (read).
        display = ov.initial_display(image, visible=[1])
        assert [d["visible"] for d in display] == [False, True, False]
        assert [d["measured"] for d in display] == [False, True, False]
        assert display[0]["contrast_limits"] == (0.0, 65535.0)
        # The contrast comes from the loaded region, at the preview level.
        region = ov.loaded_region(image, {"z": (20.0, 40.0)})
        low, high = ov.channel_contrast(image, 1, region)
        part = data0[1, 10:21]           # level 0 is the preview level here
        assert (low, high) == tuple(np.percentile(part, [0.5, 99.9]))


def test_projection_slab():
    # Inside the slab: the point stays, the margins reach both ends.
    assert ov.projection_slab(30.0, 10.0, 50.0) == (30.0, 20.0, 20.0)
    assert ov.projection_slab(10.0, 10.0, 50.0) == (10.0, 0.0, 40.0)
    # Outside: the point moves to the centre of the slab.
    assert ov.projection_slab(80.0, 10.0, 50.0) == (30.0, 20.0, 20.0)


def test_projection_read_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        image = _image(tmp)          # level 0: (3, 40, 150, 130), chunks 8
        chunk = 512 * 2 * 3          # one 8x8x8 uint16 chunk, 3 channels
        # Whole depth in XY: every chunk of the volume (5 * 19 * 17).
        assert ov.projection_read_bytes(image, 0, ("y", "x"), "z") \
            == 5 * 19 * 17 * chunk
        # A slab of z 10-20 um (voxels 5-10 at 2 um) crosses z chunks 0 and 1.
        assert ov.projection_read_bytes(image, 0, ("y", "x"), "z",
                                        (10.0, 20.0)) == 2 * 19 * 17 * chunk
        # XZ projects along y: 5 * 17 chunks per y chunk, 19 y chunks.
        assert ov.projection_read_bytes(image, 0, ("z", "x"), "y") \
            == 5 * 17 * 19 * chunk


def test_rectangles():
    ranges = {"z": (10.0, 20.0), "y": (1.0, 2.0), "x": (5.0, 7.5)}
    corners = ov.rectangle(ranges, ("z", "x"))
    assert corners.tolist() == [[10, 5], [10, 7.5], [20, 7.5], [20, 5]]
    extents = {"z": (0.0, 100.0), "x": (6.0, 100.0)}
    back = ov.rectangle_ranges(corners, ("z", "x"), extents)
    assert back == {"z": (10.0, 20.0), "x": (6.0, 7.5)}    # x clipped at 6
    # A shape drawn "backwards" gives the same ranges.
    assert ov.rectangle_ranges(corners[::-1], ("z", "x"), extents) == back


def test_clipping_planes():
    planes = ov.clipping_planes(low=[10, 20, 30], high=[50, 60, 70])
    assert len(planes) == 6
    # Positions are the box edges in µm (napari 0.9.2 reads them in world
    # coordinates; see test_napari_clipping.py). Normals point inwards:
    # +z at the low z edge (keep above), -z at the high z edge (keep below).
    assert planes[0] == {"position": (10.0, 0.0, 0.0),
                         "normal": (1.0, 0.0, 0.0), "enabled": True}
    assert planes[1] == {"position": (50.0, 0.0, 0.0),
                         "normal": (-1.0, 0.0, 0.0), "enabled": True}
    assert planes[2]["position"] == (0.0, 20.0, 0.0)
    assert np.allclose([p["position"][2] for p in planes[4:]], [30, 70])
    assert [p["normal"][2] for p in planes[4:]] == [1.0, -1.0]


def test_polyline_and_dashes():
    square = [(0, 0), (0, 10), (10, 10), (10, 0)]      # perimeter 40
    solid = ov.polyline_segments(square)
    assert solid.shape == (4, 2, 2)
    assert solid[-1].tolist() == [[10, 0], [0, 0]]       # closed
    # Dashes of 4 with gaps of 3 start at 0, 7, ..., 35. The dashes 7-11
    # and 28-32 reach a corner (at 10 and 30) and are split there, so that
    # every piece runs along one side: 6 dashes, 8 pieces.
    dashes = ov.dash_segments(square, dash=4, gap=3)
    assert dashes.shape == (8, 2, 2)
    lengths = np.linalg.norm(dashes[:, 1] - dashes[:, 0], axis=1)
    assert np.isclose(lengths.sum(), 6 * 4)
    for start, end in dashes:      # on the outline, never cutting a corner
        on_side = [(start[k] == end[k]) and start[k] in (0, 10) for k in (0, 1)]
        assert any(on_side), (start, end)
    # Very small dashes are lengthened, so there are never too many.
    assert len(ov.dash_segments(square, 1e-6, 1e-6, max_dashes=100)) <= 104


def test_masked_region():
    # A level with z (2 µm), y and x (0.5 µm) voxels and an offset.
    axes = ["z", "y", "x"]
    scale, translation, shape = [2.0, 0.5, 0.5], [10.0, 1.0, 0.0], (12, 70, 66)
    ranges = {"z": (13.0, 27.0), "y": (4.0, 29.0), "x": (2.5, 31.0)}
    region = ov.masked_region(axes, scale, translation, shape, ranges,
                              {("y", "x"): L_SHAPE})
    # Reference: voxel centres inside the box but outside the L outline.
    z, y, x = (t + np.arange(n) * s
               for t, s, n in zip(translation, scale, shape))
    in_box = ((z >= 13) & (z <= 27))[:, None, None] \
        & ((y >= 4) & (y <= 29))[None, :, None] \
        & ((x >= 2.5) & (x <= 31))[None, None, :]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    in_l = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                          L_SHAPE).reshape(yy.shape)
    expected = in_box & ~in_l[None]
    np.testing.assert_array_equal(region, expected)
    assert 0 < region.sum() < in_box.sum()
    # No outlines: nothing is blacked out.
    assert not ov.masked_region(axes, scale, translation, shape, ranges,
                                {}).any()


def test_outside_region():
    """With "Clip 3D", the voxels whose centres lie outside the box are 0
    in the 3D view: exactly those, with box edges on and off the voxel grid
    (an edge a quarter voxel into a voxel left a halo, 2026-11-02)."""
    axes = ["z", "y", "x"]
    scale, translation, shape = [2.0, 0.5, 0.5], [10.0, 1.0, 0.0], (12, 70, 66)
    for ranges in ({"z": (13.0, 27.0), "y": (4.0, 29.0), "x": (2.5, 31.0)},
                   {"z": (12.5, 26.5), "y": (4.125, 28.875),
                    "x": (2.625, 30.875)}):          # edges inside voxels
        outside = ov.outside_region(axes, scale, translation, shape, ranges)
        z, y, x = (t + np.arange(n) * s
                   for t, s, n in zip(translation, scale, shape))
        in_box = ((z >= ranges["z"][0]) & (z <= ranges["z"][1]))[:, None, None] \
            & ((y >= ranges["y"][0]) & (y <= ranges["y"][1]))[None, :, None] \
            & ((x >= ranges["x"][0]) & (x <= ranges["x"][1]))[None, None, :]
        np.testing.assert_array_equal(outside, ~in_box)
        # The blacked-out region lies inside the box: the two never overlap.
        region = ov.masked_region(axes, scale, translation, shape, ranges,
                                  {("y", "x"): L_SHAPE})
        assert region.any() and not (region & outside).any()
    # A box beside the loaded region: everything is outside.
    assert ov.outside_region(axes, scale, translation, shape,
                             {"z": (500, 600), "y": (4, 29),
                              "x": (2.5, 31)}).all()


def test_polygons_per_view_and_slice():
    """A box's polygons as a view sees them (stored the other way round,
    they are swapped), and which one a view shows on a slice (0.7.0)."""
    tri = [(1.0, 2.0), (3.0, 2.0), (3.0, 5.0)]                # (x, y)
    box = vc.CropBox("b", outlines={("x", "y"): {4.0: tri, 8.0: tri}})
    keys = ov.keys_in_plane(box, ("y", "x"))
    assert keys == {4.0: [(2.0, 1.0), (2.0, 3.0), (5.0, 3.0)],
                    8.0: [(2.0, 1.0), (2.0, 3.0), (5.0, 3.0)]}
    assert ov.polygon_on_slice(box, ("y", "x"), 8.2, tolerance=0.25)[0] == 8.0
    assert ov.polygon_on_slice(box, ("y", "x"), 6.0, tolerance=0.25) is None
    one = vc.CropBox("c", outlines={("y", "x"): tri})        # through the box
    assert ov.polygon_on_slice(one, ("y", "x"), 123.0)[0] is None
    # A box's only polygon reaches through the box, on any slice; once
    # another view has one too, each belongs to its own slice.
    single = vc.CropBox("d", outlines={("y", "x"): {4.0: tri}})
    assert ov.reaches_through(single, ("x", "y"))
    assert ov.polygon_on_slice(single, ("y", "x"), 50.0)[0] == 4.0
    single.outlines[("z", "x")] = {2.0: tri}
    assert not ov.reaches_through(single, ("y", "x"))
    assert ov.polygon_on_slice(single, ("y", "x"), 50.0) is None
    assert ov.polygon_on_slice(single, ("y", "x"), 4.0)[0] == 4.0
    # Silhouettes (drawn on a projection) are not polygons on a slice; a
    # projection shows those whose depth range overlaps its own.
    single.outlines[("y", "x")][(0.0, 20.0)] = tri
    assert ov.polygon_on_slice(single, ("y", "x"), 10.0) is None
    assert [k for k, _ in ov.silhouettes_in_view(
        single, ("x", "y"), (15.0, 40.0))] == [(0.0, 20.0)]
    assert ov.silhouettes_in_view(single, ("y", "x"), (21.0, 40.0)) == []
    assert ov.same_range((0.0, 20.0), (0.1, 19.9), 0.25)
    assert not ov.same_range((0.0, 20.0), (0.0, 30.0), 0.25)
    assert not ov.same_range(4.0, (0.0, 20.0), 0.25)
    # Generic names per view: slices in order, silhouettes apart.
    single.outlines[("y", "x")][12.0] = tri
    views = {frozenset("yx"): "xy", frozenset("zx"): "xz"}
    names = ov.polygon_names(single, lambda plane: views[frozenset(plane)])
    assert names == {(("y", "x"), 4.0): "xy1", (("y", "x"), 12.0): "xy2",
                     (("y", "x"), (0.0, 20.0)): "xyP1",
                     (("z", "x"), 2.0): "xz1"}


def test_slice_outline():
    """The dashed outline on a slice is the edge of what the crop keeps
    there: between two squares on two slices it is that square, before and
    after them there is none, and a margin widens it (0.7.0)."""
    square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    box = vc.CropBox("b", {"z": (0.0, 20.0), "y": (-2.0, 12.0),
                           "x": (-2.0, 12.0)},
                     outlines={("y", "x"): {0.0: square, 20.0: square}})
    zyx, pixel = ["z", "y", "x"], 0.25
    for depth in (0.0, 10.0, 20.0):
        (line,) = ov.slice_outline(box, zyx, pixel, ("y", "x"), depth,
                                   box.ranges)
        for k in range(2):
            assert abs(line[:, k].min()) <= 0.3, (depth, line[:, k].min())
            assert abs(line[:, k].max() - 10) <= 0.3, (depth, line[:, k].max())
    for depth in (-1.0, 21.0):
        assert ov.slice_outline(box, zyx, pixel, ("y", "x"), depth,
                                box.ranges) == []
    box.margin = 4                                  # 4 x 0.25 = 1 um
    (line,) = ov.slice_outline(box, zyx, pixel, ("y", "x"), 10.0, box.ranges)
    for k in range(2):
        assert abs(line[:, k].min() + 1) <= 0.3
        assert abs(line[:, k].max() - 11) <= 0.3
    # slice_mask: the kept points of the same grid, remembered.
    keep, origin, step = ov.slice_mask(box, zyx, pixel, ("y", "x"), 10.0,
                                       box.ranges)
    assert step == pixel and keep.shape == (56, 56)
    assert ov.slice_mask(box, zyx, pixel, ("y", "x"), 10.0,
                         box.ranges)[0] is keep


def test_views_show_the_crops_shape():
    """The views draw the shape from the same function as the crop, on a
    coarser grid (VIEW_DETAIL): they differ only within 1 voxel of its
    edge, in under 1 % of the kept voxels."""
    truth, outlines = tube_and_outlines()
    n, zyx = truth.shape[0], ["z", "y", "x"]
    level = vc.Level("0", None, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])

    def shape(detail):
        return np.broadcast_to(vc.outline_mask(
            level, zyx, [0, 0, 0], (n,) * 3, outlines, spatial=zyx,
            detail=detail), (n,) * 3)

    crop, view = shape(vc.FIELD_POINTS), shape(ov.VIEW_DETAIL)
    differ = crop != view
    assert differ.sum() < 0.01 * crop.sum()
    edge = distance_transform_edt(crop) + distance_transform_edt(~crop)
    assert edge[differ].max() <= 1.0


def test_masked_volume():
    rng = np.random.default_rng(1)
    data = rng.integers(1, 100, size=(3, 4, 5, 6), dtype=np.uint16)  # t,z,y,x
    reads = []

    class Counting:                 # a source that counts how often it is read
        shape, dtype, ndim, size = data.shape, data.dtype, data.ndim, data.size

        def __getitem__(self, key):
            reads.append(key)
            return data[key]

    region = np.zeros((1, 4, 5, 6), dtype=bool)   # broadcasts over t
    region[0, 1:3, 2:, :4] = True
    cache = ov.VolumeCache(Counting())
    kept = ov.MaskedVolume(cache, region, "kept")
    lost = ov.MaskedVolume(cache, region, "masked")
    key = (1, slice(None), slice(None), slice(None))   # how napari asks in 3D
    a, b = kept[key], lost[key]
    assert len(reads) == 1                       # read once, used twice
    np.testing.assert_array_equal(a + b, data[1])          # nothing lost
    assert not a[region[0]].any() and not b[~region[0]].any()
    # Other parts of the same timepoint come from memory: napari first asks
    # a new layer for one plane, then for the volume.
    plane = lost[(1, slice(1, 2))]
    np.testing.assert_array_equal(plane, b[1:2])
    assert len(reads) == 1
    kept[(2, slice(None), slice(None), slice(None))]       # another timepoint
    assert len(reads) == 2
    # Leading slices keep their axis; ... means everything.
    assert kept[(slice(0, 2),)].shape == (2, 4, 5, 6)
    assert np.asarray(kept).shape == data.shape
    np.testing.assert_array_equal(lost[...] + kept[...], data)
    assert kept.shape == data.shape and kept.dtype == data.dtype


def test_crop_preview_geometry():
    """The crop preview (0.9.0): which slices the crop holds, and the black
    rectangles around the box in one view."""
    ranges = {"t": (15.0, 45.0), "z": (10.0, 20.0), "y": (2.0, 8.0),
              "x": (4.0, 6.0)}
    plane, depth = ("y", "x"), "z"                       # the XY view
    cursor = {"t": 30.0, "z": 12.0, "y": 0.0, "x": 0.0}  # y, x don't matter
    assert ov.slice_in_crop(ranges, plane, depth, cursor)
    # The box's ranges are outer voxel edges: a slice on the edge is in.
    assert ov.slice_in_crop(ranges, plane, depth, {**cursor, "z": 20.0})
    assert not ov.slice_in_crop(ranges, plane, depth, {**cursor, "z": 21.0})
    # Another timepoint than the crop's: nothing of this slice is cropped.
    assert not ov.slice_in_crop(ranges, plane, depth, {**cursor, "t": 60.0})
    # On a projection the slab counts, not the slice: any overlap.
    far = {**cursor, "z": 50.0}
    assert ov.slice_in_crop(ranges, plane, depth, far, slab=(18.0, 30.0))
    assert not ov.slice_in_crop(ranges, plane, depth, far, slab=(20.0, 30.0))

    extents = ((0.0, 10.0), (0.0, 10.0))
    rects = ov.preview_rectangles(ranges, plane, extents, here=True)
    assert len(rects) == 4
    # Sample points on a fine grid: black exactly outside the box.
    ys, xs = np.meshgrid(np.linspace(-9.95, 19.95, 300),
                         np.linspace(-9.95, 19.95, 300), indexing="ij")
    black = np.zeros(ys.shape, dtype=bool)
    for (top, left), (bottom, right) in rects:
        black |= (ys >= top) & (ys <= bottom) & (xs >= left) & (xs <= right)
    inside = (ys > 2.0) & (ys < 8.0) & (xs > 4.0) & (xs < 6.0)
    assert not black[inside].any() and black[~inside].all()
    # The rectangles reach beyond the image, so its edges never show.
    assert min(r[0][0] for r in rects) < 0 and max(r[1][1] for r in rects) > 10
    # Nothing of this slice cropped: one rectangle over everything.
    (whole,) = ov.preview_rectangles(ranges, plane, extents, here=False)
    assert whole[0][0] < 0 and whole[0][1] < 0
    assert whole[1][0] > 10 and whole[1][1] > 10


def test_time_kept_fades_skipped_timepoints():
    """0.11.0: a region's lines fade (and its crop preview is all black) at
    timepoints its crop does not keep: outside its t range, or skipped by
    "every n-th". Images without a time axis: always kept."""
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lapse.zarr"
        make_test_image(path, shape0=(9, 2, 10, 40, 40),
                        factors=[(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)],
                        axes="tczyx", scale0=(30.0, 1.0, 2.0, 0.5, 0.5))
        image = vc.open_image(path)
        box = vc.CropBox("b", {"t": (30, 210)}, t_step=3)   # keeps 1, 4, 7
        plan = vc.plan_crop(image, box, min_size=4)
        views = SimpleNamespace(image=image, boxes=[box],
                                box_ranges=[plan.snapped_ranges], cursor={})
        kept = []
        for frame in range(9):
            views.cursor = {"t": frame * 30.0, "z": 5.0}
            if ov.OrthoViews.time_kept(views, 0):
                kept.append(frame)
        assert kept == [1, 4, 7], kept
        box.t_step = 1
        views.box_ranges = [vc.plan_crop(image, box,
                                         min_size=4).snapped_ranges]
        kept = [f for f in range(9) if (views.cursor.update(t=f * 30.0)
                                        or ov.OrthoViews.time_kept(views, 0))]
        assert kept == [1, 2, 3, 4, 5, 6, 7], kept
        views.image = vc.open_image(path)        # no box at index 1
        assert ov.OrthoViews.time_kept(views, 1)

        flat = Path(tmp) / "zyx.zarr"
        make_test_image(flat, shape0=(40, 150, 130),
                        factors=[(1, 1, 1), (1, 4, 4)],
                        scale0=(2.0, 0.25, 0.25), axes="zyx")
        views.image = vc.open_image(flat)
        assert ov.OrthoViews.time_kept(views, 0)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"all {len(tests)} tests passed")
