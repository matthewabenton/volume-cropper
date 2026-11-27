"""Tests for the crop preview's projection (0.18.0): crop_projection.py and
the 2D views' layers that show it (hidden windows).

Run with ``python run_all_tests.py crop_projection`` (or this file alone).
They check that:
- the projection is the crop's own: projecting a real crop of a region with
  an outline (written to disk by crop_box and read back) along each axis
  gives exactly the same image, for the maximum and the mean projection;
- a large region is read at a coarser level (at most MAX_VOXELS voxels per
  channel), and the projection then has that level's size;
- in the window, with the preview on and a maximum projection, each 2D view
  shows the selected region's projection (one layer per channel shown,
  over the preview's black, at the region's place, in the channel's
  colours), which follows the contrast, a channel switched off or on,
  an outline edit, and goes away without the projection or the preview.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import time
from pathlib import Path

import numpy as np

from volume_cropper import crop_projection as cp
from volume_cropper import ortho_views as ov
from volume_cropper import core as vc
from test_volume_cropper import make_test_image

RANGES = {"z": (10.0, 30.0), "y": (5.0, 15.0), "x": (2.0, 12.0)}
TRIANGLE = {("y", "x"): [(5.0, 2.0), (15.0, 2.0), (15.0, 12.0)]}


def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


def test_projection_is_the_crops_own():
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        image = vc.open_image(Path(tmp) / "src.zarr")
        region = ov.loaded_region(image)
        box = vc.CropBox("outlined", dict(RANGES), outlines=TRIANGLE)
        ranges = vc.plan_crop(image, box).snapped_ranges
        crop = vc.open_image(vc.crop_box(image, box, Path(tmp) / "out").path)
        for mode in ("max", "mean"):
            result = cp.crop_projection(image, region, [0, 2], None, ranges,
                                        box, mode)
            assert result.level == 0 and result.axes == ["z", "y", "x"]
            data = np.asarray(crop.levels[0].array)     # c, z, y, x
            for k in (0, 2):
                for i, depth in enumerate(result.axes):
                    if mode == "max":
                        expected = data[k].max(axis=i)
                    else:
                        expected = data[k].mean(axis=i)
                    got = result.planes[(k, depth)]
                    assert got.shape == expected.shape, (got.shape, depth)
                    assert np.allclose(got, expected, rtol=1e-6), (mode, k,
                                                                   depth)
            # the voxels the triangle sets to 0 are 0 in the crop: some of
            # its projection along z is 0 (the half outside the triangle)
            assert (result.planes[(0, "z")] == 0).mean() > 0.3
        # the region's place: its first voxel's centre and the voxel size
        lv0 = image.levels[0]
        assert result.scale == {"z": lv0.scale[1], "y": lv0.scale[2],
                                "x": lv0.scale[3]}
        for axis, (low, _high) in ranges.items():
            assert abs(result.first[axis] - low
                       - result.scale[axis] / 2) < 1e-9, axis
        # a 2D view's axes in its own order
        assert result.plane(0, "y", ("x", "z")).shape == \
            result.planes[(0, "y")].T.shape


def test_large_region_at_a_coarser_level():
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        image = vc.open_image(Path(tmp) / "src.zarr")
        region = ov.loaded_region(image)
        box = vc.CropBox("all", {})
        ranges = vc.plan_crop(image, box).snapped_ranges
        result = cp.crop_projection(image, region, [1], None, ranges, box,
                                    max_voxels=50_000)
        assert result.level > 0
        n = int(np.prod(list(result.shape.values())))
        assert n <= 50_000, n
        lv = image.levels[result.level]
        assert result.planes[(1, "z")].shape == lv.array.shape[2:]
        try:
            cp.crop_projection(image, region, [0], None, ranges, box, "sum")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown mode was accepted")


def test_preview_projection_in_window():
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump()
            ortho = panel.ortho
            active = {name: ortho.views[name].model.layers.selection.active
                      for name in ov.VIEWS_2D}
            i = panel._new_box(dict(RANGES))
            panel.boxes[i].outlines = vc.normalize_outlines(TRIANGLE)
            panel._refresh(select=i)
            ortho.set_projection("max", "full")
            ortho.set_preview(True)
            _pump(1.5)
            ranges = ortho.box_ranges[i]
            expected = cp.crop_projection(
                panel.image, ortho.region, [0, 1, 2], None, ranges,
                ortho.boxes[i], "max")
            shown = [k for k, d in enumerate(ortho.display)
                     if d["visible"]]
            assert shown, ortho.display
            for name in ov.VIEWS_2D:
                view = ortho.views[name]
                depth = view.order[-3]
                layers = view.model.layers
                black = layers.index(view.preview_outside)
                for k in shown:
                    layer = view.crop_projection_layers[k]
                    assert layer.visible, (name, k)
                    assert layers.index(layer) > black, name
                    want = expected.plane(k, depth, view.plane)
                    assert np.array_equal(np.asarray(layer.data), want), \
                        (name, k)
                    v, h = view.plane
                    assert tuple(layer.translate) == (expected.first[v],
                                                      expected.first[h])
                    assert layer.blending == "additive"
            assert "level 0" in panel.preview_info.text(), \
                panel.preview_info.text()
            # the regions' layer stays the active one (for drawing)
            for name in ov.VIEWS_2D:
                selection = ortho.views[name].model.layers.selection
                assert selection.active is active[name], name
                assert active[name] is ortho.views[name].shapes, name
            assert panel.preview_info.isVisibleTo(panel)

            # the contrast follows the channel
            xy = ortho.views["XY"].channel_layers
            k = shown[0]
            xy[k].contrast_limits = (100, 2000)
            _pump(0.1)
            for name in ov.VIEWS_2D:
                layer = ortho.views[name].crop_projection_layers[k]
                assert tuple(layer.contrast_limits) == (100, 2000), name
            # a channel switched off: its layer goes; on again: comes back
            ortho.set_channel_visible(k, False)
            _pump(0.6)
            assert not ortho.views["XY"].crop_projection_layers[k].visible
            ortho.set_channel_visible(k, True)
            _pump(1.0)
            assert ortho.views["XY"].crop_projection_layers[k].visible

            # an outline edit: the projection follows (no triangle: the
            # whole box)
            panel.boxes[i].outlines = {}
            panel._refresh(select=i)
            _pump(1.5)
            plain = cp.crop_projection(
                panel.image, ortho.region, [k], None, ortho.box_ranges[i],
                ortho.boxes[i], "max")
            layer = ortho.views["XY"].crop_projection_layers[k]
            assert np.array_equal(np.asarray(layer.data),
                                  plain.plane(k, "z", ("y", "x")))

            # no projection, or no preview: hidden
            ortho.set_projection("none", "full")
            _pump(0.3)
            assert not any(layer.visible for view in ortho.views.values()
                           for layer in view.crop_projection_layers.values())
            assert not panel.preview_info.isVisibleTo(panel)
            ortho.set_projection("max", "full")
            _pump(1.0)
            assert ortho.views["XY"].crop_projection_layers[k].visible
            ortho.set_preview(False)
            _pump(0.3)
            assert not any(layer.visible for view in ortho.views.values()
                           for layer in view.crop_projection_layers.values())
        finally:
            viewer.close()
            _pump()


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["crop_projection"]))
