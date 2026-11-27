"""Tests for the Levels histogram (0.18.0): levels.py and the panel part
levels_panel.py (hidden windows).

Run with ``python run_all_tests.py levels`` (or this file alone). They check
that:
- the histogram's bins hold whole numbers for integer data (one value per
  bin for a narrow range, at most 256 bins), cover the contrast limits, and
  give the right minimum, maximum, mean, share at the maximum and shares
  below / above the limits;
- "whole image" reads the loaded image at the finest level that fits, and
  the timepoint shown;
- "crop region" reads exactly the voxels the crop keeps: the sum of its
  values equals the sum of a real crop of the region, with and without an
  outline (the voxels an outline sets to 0 add nothing to the crop);
- in the window, the histogram follows the XY view, dragging a contrast
  line sets that limit in all views, the pointer readout names a bin, a
  channel chosen in napari's layer list is shown, a limit set beyond the
  axis in napari widens it, and the crop-region source counts the region's
  voxels;
- gamma: the dashed line sits at the value shown at half brightness, and
  the gamma computed back from that place is the same gamma (0.2 to 2,
  napari's range); the min / max boxes get sensible decimals and steps;
- in the window: dragging the dashed line sets gamma in all views and
  double-clicking it sets 1; the min, max and gamma boxes apply typed
  values to all views, keep the lower limit below the upper one, follow
  napari's own controls, and a limit typed beyond the axis widens it;
- (0.22.0) "zoom to min–max": the axis covers the limits with 25 % room on
  either side, never beyond the data, and keeps a marked value in view;
  the bins are finer there, the statistics still count all the values; in
  the window the axis stays put while a line is dragged (the line stops at
  the edge), follows when the drag ends, a typed value or a marked value,
  and unticking gives the whole axis back.
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

from volume_cropper import levels as lv
from volume_cropper import ortho_views as ov
from volume_cropper import core as vc
from test_volume_cropper import make_test_image


def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


# --- levels.py: the histogram -------------------------------------------------

def test_histogram_integer_bins():
    values = np.array([0, 1, 1, 2, 2, 2, 10], dtype=np.uint16)
    hist = lv.histogram(values)
    assert hist.integer and hist.counts.size == 11      # 0..10, one value each
    assert hist.edges[0] == -0.5 and hist.edges[-1] == 10.5
    assert list(hist.counts[:3]) == [1, 2, 3] and hist.counts[10] == 1
    assert hist.minimum == 0 and hist.maximum == 10 and hist.at_max == 1
    assert abs(hist.mean - values.mean()) < 1e-12
    assert hist.bin_at(2) == 2 and hist.bin_at(11) is None
    below, above = hist.outside((1, 2))       # 0 below, 10 above
    assert below == 1 / 7 and above == 1 / 7
    # the contrast limits widen the axis
    assert lv.histogram(values, limits=(0, 20)).edges[-1] == 20.5
    # a wide range: at most 256 bins, each of whole numbers, all counted
    wide = lv.histogram(np.arange(4000, dtype=np.uint16))
    width = wide.edges[1] - wide.edges[0]
    assert wide.counts.size <= lv.BINS and width == int(width) > 1
    assert wide.counts.sum() == 4000
    assert np.allclose(wide.edges % 1, 0.5)            # between whole numbers


def test_histogram_float_constant_and_empty():
    rng = np.random.default_rng(1)
    values = rng.normal(5, 1, 10_000).astype(np.float32)
    values[0] = np.nan                                  # left out
    hist = lv.histogram(values)
    assert not hist.integer and hist.counts.size == lv.BINS
    assert hist.counts.sum() == hist.n == 9_999
    constant = lv.histogram(np.full(10, 3.0))
    assert constant.counts.sum() == 10 and constant.at_max == 10
    assert lv.histogram(np.zeros(0, dtype=np.uint8)) is None


def test_axis_range_zoom():
    """(0.22.0) The range the axis covers, with and without the zoom."""
    data = (100, 5000)
    # without the zoom: the data and the limits
    assert lv.axis_range(data) == (100, 5000)
    assert lv.axis_range(data, (1000, 2000)) == (100, 5000)
    assert lv.axis_range(data, (0, 8000)) == (0, 8000)
    # zoomed: the limits and 25 % of their distance on either side
    margin = lv.ZOOM_MARGIN
    assert margin == 0.25
    assert lv.axis_range(data, (1000, 2000), margin) == (750, 2250)
    # never beyond the data: a limit near its end leaves less room there
    assert lv.axis_range(data, (150, 1000), margin) == (100, 1212.5)
    assert lv.axis_range(data, (4000, 4900), margin) == (3775, 5000)
    # limits beyond the data: they stay on the axis, with no empty room
    assert lv.axis_range(data, (0, 8000), margin) == (0, 8000)
    # a value to keep in view (the Autocropper's threshold)
    assert lv.axis_range(data, (1000, 2000), margin, keep=[3000]) == \
        (750, 3000)
    assert lv.axis_range(data, (1000, 2000), margin, keep=[10]) == \
        (100, 2250)


def test_histogram_zoomed():
    """(0.22.0) Zoomed bins: finer, over the zoomed range only; the
    statistics still of all the values."""
    values = np.arange(65536, dtype=np.uint16)          # 0..65535, once each
    whole = lv.histogram(values, (1000, 2000))
    zoomed = lv.histogram(values, (1000, 2000), margin=lv.ZOOM_MARGIN)
    # whole-number bins over 750..2250 (and a few values more to fill the
    # last bin): 1501 values, 6 per bin, 251 bins
    assert zoomed.edges[0] == 749.5 and zoomed.edges[-1] >= 2250.5
    assert zoomed.counts.size <= lv.BINS
    assert zoomed.edges[1] - zoomed.edges[0] == 6
    assert whole.edges[1] - whole.edges[0] == 256        # far coarser
    assert zoomed.counts.sum() == zoomed.edges[-1] - zoomed.edges[0]
    # the statistics: all the values
    assert zoomed.n == 65536 and zoomed.maximum == 65535
    assert zoomed.minimum == 0 and zoomed.at_max == 1
    assert zoomed.outside((1000, 2000)) == whole.outside((1000, 2000))
    # floating-point data: 256 bins over the zoomed range
    floats = np.linspace(0.0, 1.0, 10_001)
    zoomed = lv.histogram(floats, (0.4, 0.6), margin=lv.ZOOM_MARGIN)
    assert zoomed.counts.size == lv.BINS
    assert np.isclose(zoomed.edges[0], 0.35)
    assert np.isclose(zoomed.edges[-1], 0.65)
    assert zoomed.n == 10_001


# --- levels.py: reading the image ---------------------------------------------

def test_gamma_midtone_and_boxes():
    # gamma 1: halfway; 0.2 and 2: at 3 % and 71 % of the way
    assert lv.midtone_value((0, 100), 1.0) == 50
    assert abs(lv.midtone_value((0, 100), 0.2) - 3.125) < 1e-9
    assert abs(lv.midtone_value((100, 200), 2.0) - (100 + 100 * 0.5 ** 0.5)) \
        < 1e-9
    # the value really is shown at half brightness (napari's mapping)
    low, high, gamma = 20.0, 4000.0, 0.6
    v = lv.midtone_value((low, high), gamma)
    assert abs(((v - low) / (high - low)) ** gamma - 0.5) < 1e-12
    # back from the place to gamma: the same gamma (as rounded by the box)
    for gamma in (0.2, 0.35, 0.5, 0.8, 1.0, 1.25, 1.6, 2.0):
        place = lv.midtone_value((low, high), gamma)
        assert lv.gamma_for_midtone((low, high), place) == gamma, gamma
    # beyond the ends of napari's range: the ends
    assert lv.gamma_for_midtone((low, high), low) == 0.2
    assert lv.gamma_for_midtone((low, high), high + 50) == 2.0
    assert lv.clamp_gamma(5) == 2.0 and lv.clamp_gamma(0.01) == 0.2
    assert lv.clamp_gamma(0.666) == 0.67
    # boxes: no decimals for whole numbers, ~4 significant digits else
    assert lv.box_decimals(65535, True) == 0
    assert lv.box_decimals(1.0, False) == 3
    assert lv.box_decimals(0.01, False) == 5
    assert lv.box_decimals(1e6, False) == 1
    # arrow steps: a round ~1 % of the span; at least 1 for whole numbers
    assert lv.box_step(4000, True) == 50
    assert lv.box_step(50, True) == 1
    assert abs(lv.box_step(1.0, False) - 0.01) < 1e-12
    assert lv.box_step(0, False) == 1.0


def test_read_whole_image():
    with tempfile.TemporaryDirectory() as tmp:
        data0 = make_test_image(Path(tmp) / "src.zarr")   # c z y x (3, 40, 150, 130)
        image = vc.open_image(Path(tmp) / "src.zarr")
        region = ov.loaded_region(image)
        values, level = lv.read_values(image, region, 1)
        assert level == 0 and values.size == 40 * 150 * 130
        assert np.array_equal(np.sort(values), np.sort(data0[1].ravel()))
        # fewer voxels allowed: a coarser level, all of it
        values, level = lv.read_values(image, region, 1, max_values=50_000)
        expected = np.asarray(image.levels[level].array[1])
        assert level > 0 and values.size == expected.size <= 50_000
        assert np.array_equal(values, expected.ravel())


def test_read_timepoint():
    factors = [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)]
    with tempfile.TemporaryDirectory() as tmp:
        data0 = make_test_image(Path(tmp) / "t.zarr", shape0=(3, 2, 6, 20, 20),
                                factors=factors, axes="tczyx",
                                scale0=(5.0, 1.0, 2.0, 0.5, 0.5))
        image = vc.open_image(Path(tmp) / "t.zarr")
        region = ov.loaded_region(image)
        t0 = image.levels[0].translation[0]
        values, _ = lv.read_values(image, region, 1, time=t0 + 10.0)  # t = 2
        assert np.array_equal(values, data0[2, 1].ravel())
        values, _ = lv.read_values(image, region, 1, time=None)       # first
        assert np.array_equal(values, data0[0, 1].ravel())


def _crop_sum(image, box, tmp) -> int:
    """The sum of channel 0 of a real crop of the box (level 0)."""
    result = vc.crop_box(image, box, Path(tmp) / f"out_{box.name}")
    crop = np.asarray(vc.open_image(result.path).levels[0].array[0])
    return int(crop.astype(np.int64).sum())


def test_read_crop_region_matches_the_crop():
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        image = vc.open_image(Path(tmp) / "src.zarr")
        region = ov.loaded_region(image)
        ranges = {"z": (10.0, 30.0), "y": (5.0, 15.0), "x": (2.0, 12.0)}
        plain = vc.CropBox("plain", dict(ranges))
        outlined = vc.CropBox("outlined", dict(ranges), outlines={
            ("y", "x"): [(5.0, 2.0), (15.0, 2.0), (15.0, 12.0)]})  # a triangle
        sizes = {}
        for box in (plain, outlined):
            as_cropped = vc.plan_crop(image, box).snapped_ranges
            values, level = lv.read_values(image, region, 0, ranges=as_cropped,
                                           box=box)
            assert level == 0
            assert int(values.astype(np.int64).sum()) == \
                _crop_sum(image, box, tmp), box.name
            sizes[box.name] = values.size
        # the triangle keeps about half of the box
        assert 0.3 < sizes["outlined"] / sizes["plain"] < 0.7, sizes


# --- the panel, in a hidden window ------------------------------------------

def _window_with_image(path):
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    viewer, panel = crop_gui.open_window(show=False)
    panel.open_image(path, choice=LoadChoice())
    _pump()
    return viewer, panel


def _mouse(levels, name, x, y, button=None):
    from matplotlib.backend_bases import MouseEvent
    event = MouseEvent(name, levels.canvas, x, y, button=button)
    levels.canvas.callbacks.process(name, event)


def test_levels_panel_in_window():
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = _window_with_image(Path(tmp) / "src.zarr")
        try:
            levels = panel.levels
            xy = panel.ortho.views["XY"].channel_layers
            # the XY view as shown (a hidden window: compute it directly)
            levels.recompute()
            assert levels.hist is not None, levels.stats.text()
            assert levels.hist.n == np.asarray(xy[0]._slice.image.raw).size
            assert "min " in levels.stats.text() and "XY view" in \
                levels.stats.text()

            # drag the upper line 40 pixels to the left
            levels.canvas.draw()
            high_before = levels.limits[1]
            px = levels.ax.transData.transform((high_before, 0))[0]
            py = levels.ax.transAxes.transform((0.5, 0.5))[1]
            _mouse(levels, "button_press_event", px, py, button=1)
            _mouse(levels, "motion_notify_event", px - 40, py)
            _mouse(levels, "button_release_event", px - 40, py, button=1)
            _pump(0.2)
            high = xy[0].contrast_limits[1]
            assert high < high_before and high == int(high), (high, high_before)
            assert high == levels.limits[1]
            for name in ("XZ", "YZ"):                   # all views follow
                other = panel.ortho.views[name].channel_layers[0]
                assert other.contrast_limits[1] == high, name

            # the pointer over a bin: its values and count
            _mouse(levels, "motion_notify_event", px - 80, py)
            assert "voxels" in levels.readout.text(), levels.readout.text()

            # a channel chosen in napari's layer list is shown here
            viewer.layers.selection.active = xy[2]
            _pump(0.1)
            assert levels.channel == 2

            # a limit set beyond the axis in napari widens it
            levels.recompute()
            xy[2].contrast_limits = (0, 10_000)
            _pump(0.1)
            assert levels.hist.edges[-1] >= 10_000

            # the selected crop region: its voxels as cropped
            i = panel._new_box({"z": (10.0, 30.0), "y": (5.0, 15.0),
                                "x": (2.0, 12.0)})
            panel._refresh(select=i)
            levels.source_combo.setCurrentIndex(2)
            levels.recompute()
            ranges = panel.ortho.box_ranges[i]
            lv0 = panel.image.levels[0]             # small: read at level 0
            box = ov._box_slices(["z", "y", "x"], lv0.scale[1:],
                                 lv0.translation[1:], lv0.array.shape[1:],
                                 ranges)
            expected = np.prod([s.stop - s.start for s in box])
            assert levels.hist.n == expected, (levels.hist.n, expected)
            assert "crop region" in levels.stats.text()
        finally:
            viewer.close()
            _pump()


def test_levels_gamma_and_boxes_in_window():
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = _window_with_image(Path(tmp) / "src.zarr")
        try:
            levels = panel.levels
            xy = panel.ortho.views["XY"].channel_layers
            others = [panel.ortho.views[name].channel_layers
                      for name in ("XZ", "YZ")]
            levels.recompute()
            levels.canvas.draw()
            assert levels.hist is not None, levels.stats.text()
            low, high = (float(v) for v in xy[0].contrast_limits)
            # the boxes show the channel's limits and gamma
            assert levels.min_spin.value() == low
            assert levels.max_spin.value() == high
            assert levels.gamma_spin.value() == 1.0
            assert levels.min_spin.decimals() == 0      # 16-bit data
            # the stats name the data's own min / max as such
            assert levels.stats.text().startswith("data: min ")

            # drag the dashed line 30 pixels to the left: gamma below 1
            middle = lv.midtone_value((low, high), 1.0)
            px = levels.ax.transData.transform((middle, 0))[0]
            py = levels.ax.transAxes.transform((0.5, 0.5))[1]
            _mouse(levels, "button_press_event", px, py, button=1)
            assert levels._drag == 2                     # the dashed line
            _mouse(levels, "motion_notify_event", px - 30, py)
            assert "half brightness" in levels.readout.text()
            _mouse(levels, "button_release_event", px - 30, py, button=1)
            _pump(0.2)
            gamma = xy[0].gamma
            assert 0.2 <= gamma < 1.0, gamma
            assert all(view[0].gamma == gamma for view in others)
            assert levels.gamma_spin.value() == round(gamma, 2)
            target = levels.ax.transData.inverted().transform(
                (px - 30, py))[0]
            assert abs(lv.gamma_for_midtone((low, high), target) - gamma) \
                < 1e-9
            drawn = levels._artists["midtone"].get_xdata()[0]
            assert abs(drawn - lv.midtone_value((low, high), gamma)) < 1e-6
            # the limits were not changed by it
            assert tuple(xy[0].contrast_limits) == (low, high)

            # double-click it: gamma 1 again
            px = levels.ax.transData.transform((drawn, 0))[0]
            from matplotlib.backend_bases import MouseEvent
            event = MouseEvent("button_press_event", levels.canvas, px, py,
                               button=1, dblclick=True)
            levels.canvas.callbacks.process("button_press_event", event)
            _pump(0.1)
            assert xy[0].gamma == 1.0 and levels.gamma_spin.value() == 1.0

            # the gamma box: all views, and the dashed line moves
            levels.gamma_spin.setValue(0.5)
            _pump(0.1)
            assert xy[0].gamma == 0.5
            assert all(view[0].gamma == 0.5 for view in others)
            drawn = levels._artists["midtone"].get_xdata()[0]
            assert abs(drawn - lv.midtone_value((low, high), 0.5)) < 1e-6

            # the max box: all views; the histogram's line moves
            levels.max_spin.setValue(high - 100)
            _pump(0.1)
            assert xy[0].contrast_limits[1] == high - 100
            assert all(view[0].contrast_limits[1] == high - 100
                       for view in others)
            assert levels.limits[1] == high - 100
            assert levels._artists["lines"][1].get_xdata()[0] == high - 100
            assert xy[0].gamma == 0.5                 # gamma kept
            # a min typed above the max stops just below it
            levels.min_spin.setValue(high + 500)
            _pump(0.1)
            assert tuple(xy[0].contrast_limits) == (high - 101, high - 100)
            assert levels.min_spin.value() == high - 101
            # a max beyond the data widens the axis
            levels.min_spin.setValue(low)
            levels.max_spin.setValue(60_000)
            _pump(0.1)
            assert xy[0].contrast_limits[1] == 60_000
            assert levels.hist.edges[-1] >= 60_000
            # napari's own controls: the boxes follow
            xy[0].gamma = 1.4
            xy[0].contrast_limits = (12, 3000)
            _pump(0.1)
            assert levels.gamma_spin.value() == 1.4
            assert (levels.min_spin.value(), levels.max_spin.value()) == \
                (12, 3000)
            # another channel: its own values
            xy[1].gamma = 0.8
            levels.channel_combo.setCurrentIndex(1)
            levels.recompute()
            assert levels.gamma_spin.value() == 0.8
            assert levels.max_spin.value() == float(xy[1].contrast_limits[1])
        finally:
            viewer.close()
            _pump()


def test_levels_zoom_in_window():
    """(0.22.0) "zoom to min–max" in a hidden window."""
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")        # values 0..3999
        viewer, panel = _window_with_image(Path(tmp) / "src.zarr")
        try:
            levels = panel.levels
            xy = panel.ortho.views["XY"].channel_layers
            levels.recompute()
            assert levels.hist is not None, levels.stats.text()
            assert not levels.zoom_check.isChecked()   # off at first
            n = levels.hist.n
            data = (levels.hist.minimum, levels.hist.maximum)
            levels.max_spin.setValue(2000)
            levels.min_spin.setValue(1000)
            _pump(0.1)
            assert tuple(xy[0].contrast_limits) == (1000, 2000)
            whole = levels.hist.edges
            assert whole[0] <= data[0] and whole[-1] >= data[1]

            # tick it: 750..2250, in finer bins; the numbers: all voxels
            levels.zoom_check.setChecked(True)
            _pump(0.1)
            hist = levels.hist
            assert levels._axis == (750, 2250), levels._axis
            assert hist.edges[0] == 749.5 and hist.edges[-1] >= 2250.5
            assert hist.edges[1] - hist.edges[0] < whole[1] - whole[0]
            assert hist.n == n and hist.counts.sum() < n
            assert levels.ax.get_xlim() == (hist.edges[0], hist.edges[-1])
            assert "voxels" in levels.stats.text()

            # drag the upper line far beyond the right edge: it stops at
            # the edge, and the axis stays put while dragging
            levels.canvas.draw()
            edges = hist.edges.copy()
            px = levels.ax.transData.transform((2000, 0))[0]
            py = levels.ax.transAxes.transform((0.5, 0.5))[1]
            _mouse(levels, "button_press_event", px, py, button=1)
            assert levels._drag == 1                    # the upper line
            _mouse(levels, "motion_notify_event", px + 500, py)
            assert np.array_equal(levels.hist.edges, edges)
            assert levels.limits[1] == round(edges[-1])
            _mouse(levels, "button_release_event", px + 500, py, button=1)
            _pump(0.2)
            high = float(xy[0].contrast_limits[1])
            assert high == round(edges[-1]) >= 2250, high
            # ... and when the drag ends, the axis follows: new room beyond
            room = lv.ZOOM_MARGIN * (high - 1000)
            assert levels._axis == (1000 - room, min(high + room, data[1]))
            assert levels.hist.edges[-1] > edges[-1]

            # a value typed in a box: the axis follows at once
            levels.min_spin.setValue(1500)
            _pump(0.1)
            room = lv.ZOOM_MARGIN * (high - 1500)
            assert levels._axis[0] == 1500 - room
            # gamma alone leaves the axis as it is
            axis = levels._axis
            levels.gamma_spin.setValue(0.5)
            _pump(0.1)
            assert levels._axis == axis and xy[0].gamma == 0.5
            # a marked value (the Autocropper's threshold) stays in view
            mark = float(data[1])                 # beyond the room shown
            assert mark > levels._axis[1]
            levels.set_marker(0, mark, label="threshold")
            assert levels._axis[1] == mark
            assert levels._artists["marker"].get_visible()
            levels.set_marker(None)

            # untick it: the whole axis again
            levels.zoom_check.setChecked(False)
            _pump(0.1)
            assert levels.hist.edges[0] <= data[0]
            assert levels.hist.edges[-1] >= data[1]
            assert levels.hist.counts.sum() == n
        finally:
            viewer.close()
            _pump()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("PASS", name)
