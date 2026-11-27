"""Tests for the panel's 0.15.0 changes (the user's test round of
2026-11-07), in hidden windows.

Run with ``python run_all_tests.py panel`` (or this file alone). They check
that:
- the Output group has no settings for the smallest level, parallel
  workers or keeping the position any more (crops use fixed values);
- scrolling the panel with the wheel or touchpad changes no number field
  or list under the pointer (they take the wheel only when focused);
- "Save as TIFF…" saves the selected region straight to a TIFF at the
  chosen resolution, without an OME-Zarr crop, and is hidden for a file
  read into memory (whose crops are TIFFs anyway);
- the TIFF resolutions have a row of their own and say what they are;
- (0.18.0) several regions, and several polygons, can be selected and
  deleted at once (asking first), and the region clicked last stays the
  selected one.
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

from volume_cropper import core as vc
from test_volume_cropper import make_test_image


def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


def _window_with_image(path):
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    viewer, panel = crop_gui.open_window(show=False)
    panel.open_image(path, choice=LoadChoice())
    _pump()
    return viewer, panel


def test_output_settings_removed():
    from volume_cropper import crop_gui
    viewer, panel = crop_gui.open_window(show=False)
    try:
        for name in ("min_size_spin", "workers_spin", "keep_position_check",
                     "tiff_check"):
            assert not hasattr(panel, name), name
        assert crop_gui.MIN_LEVEL_SIZE == 64 and crop_gui.WORKERS == 8
        assert crop_gui.KEEP_POSITION is True
    finally:
        viewer.close()
        _pump()


def test_wheel_scrolls_the_panel_not_the_values():
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QComboBox

    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = _window_with_image(Path(tmp) / "src.zarr")
        try:
            panel._add_box()                     # range fields appear
            _pump()
            fields = (panel.findChildren(QAbstractSpinBox)
                      + panel.findChildren(QComboBox))
            assert fields
            for field in fields:                 # also the ones made later
                assert field.property("vc_wheel_guard"), field
                assert field.focusPolicy() == Qt.FocusPolicy.StrongFocus
            spin = panel.range_spins[next(iter(panel.range_spins))][0]
            before = spin.value()
            event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                                QPoint(0, -120),
                                Qt.MouseButton.NoButton,
                                Qt.KeyboardModifier.NoModifier,
                                Qt.ScrollPhase.NoScrollPhase, False)
            QApplication.sendEvent(spin, event)
            _pump(0.1)
            assert spin.value() == before, "the wheel changed a value"
        finally:
            viewer.close()
            _pump()


def test_save_as_tiff_button():
    from volume_cropper import crop_gui
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")
        viewer, panel = _window_with_image(tmp / "src.zarr")
        try:
            combo = panel.tiff_level_combo
            assert combo.itemText(0).startswith("full resolution: ")
            assert combo.itemText(1).startswith("level 1: ")
            assert panel.tiff_button.isVisibleTo(panel)
            panel.boxes = [vc.CropBox("cube", {"z": (10, 70), "y": (2, 30),
                                               "x": (2, 30)}, channels=[1])]
            panel._refresh(select=0)
            combo.setCurrentIndex(1)
            assert "Save as TIFF: level 1" in panel.plan_label.text()
            panel.out_edit.setText(str(tmp / "out"))
            panel._crop(False, tiff_only=True)
            end = time.time() + 60
            _pump(0.3)
            while time.time() < end and not panel.tiff_button.isEnabled():
                _pump(0.2)
            _pump(0.3)
            assert "checked OK" in panel.status.text(), panel.status.text()
            assert sorted(p.name for p in (tmp / "out").iterdir()) == [
                "src_cube_level1.tif"]           # no .zarr, no temp folder
        finally:
            viewer.close()
            _pump()


def test_crossing_marks_in_their_views_colours():
    """0.16.0: the × marks in a view are drawn in the colour of the view
    whose polygon they come from (blue XY, green XZ, red YZ)."""
    from volume_cropper import ortho_views as ov
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = _window_with_image(Path(tmp) / "src.zarr")
        try:
            ortho = panel.ortho
            xz, yz = ortho.views["XZ"].plane, ortho.views["YZ"].plane

            def polygon(plane, ranges):
                (a0, a1), (b0, b1) = (ranges[plane[0]], ranges[plane[1]])
                return [(a0, b0), (a0, b1), (a1, b1), (a1, b0)]

            ranges = {"z": (10.0, 70.0), "y": (5.0, 30.0), "x": (5.0, 25.0)}
            box = vc.CropBox("b", outlines={
                tuple(xz): {15.0: polygon(xz, ranges)},
                tuple(yz): {12.0: polygon(yz, ranges)}})
            ortho.cursor["z"] = 40.0
            # zoom 10 screen pixels per µm: the marks (20-25 µm apart) are
            # far from crowded, so none is thinned out (0.20.0)
            segments, colors = ortho._crossing_marks(box, ortho.views["XY"],
                                                     10.0)
            assert len(segments) == 8 and len(colors) == 8   # 4 crosses
            # each cross in its own view's colour (the review of 2026-11-08:
            # comparing only the set of colours would pass swapped colours):
            # the XZ polygon (slice y = 15) crosses the XY slice at y = 15,
            # the YZ polygon (slice x = 12) at x = 12
            want = {view: np.round(ov.rgba(ov.PLANE_COLORS[view]), 3)
                    for view in ("XZ", "YZ")}
            centres = np.asarray(segments).mean(axis=1)       # (y, x)
            for (y, x), color in zip(centres, colors):
                view = "XZ" if np.isclose(y, 15.0) else "YZ"
                assert view == "XZ" or np.isclose(x, 12.0), (y, x)
                assert np.allclose(np.round(color, 3), want[view]), \
                    (y, x, color)
        finally:
            viewer.close()
            _pump()


def test_thin_points():
    """0.20.0: thin_points keeps points that are far enough apart, and
    about one per `spacing` along a crowded row."""
    from volume_cropper import ortho_views as ov
    far = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    assert list(ov.thin_points(far, 5.0)) == [0, 1, 2]
    row = np.stack([np.arange(10.0), np.zeros(10)], axis=1)[::-1]  # reversed
    kept = ov.thin_points(row, 3.0)
    # taken in order of the first coordinate: 0, 3, 6, 9 (indices reversed)
    assert sorted(row[kept, 0]) == [0.0, 3.0, 6.0, 9.0]
    assert len(ov.thin_points(np.zeros((0, 2)), 3.0)) == 0


def test_crossing_marks_thinned_and_switched_off():
    """0.20.0 (the user's choice, A and D): where × marks crowd, as with
    the Autocropper's 40 outlines per slice, the marks of each view are
    thinned to ov.MARK_SPACING on screen, but a view's marks never hide
    another view's; zooming in brings them all back; View: Show "× marks"
    switches them off."""
    from volume_cropper import ortho_views as ov
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = _window_with_image(Path(tmp) / "src.zarr")
        try:
            ortho = panel.ortho
            view = ortho.views["XZ"]
            xy, yz = ortho.views["XY"].plane, ortho.views["YZ"].plane
            # 40 XY polygons, 1 µm apart in z (each the rectangle y 5-30,
            # x 5-25), and one YZ polygon at x = 6, next to their x = 5 edge
            rect = {"y": (5.0, 30.0), "x": (5.0, 25.0), "z": (10.0, 70.0)}

            def polygon(plane):
                (a0, a1), (b0, b1) = rect[plane[0]], rect[plane[1]]
                return [(a0, b0), (a0, b1), (a1, b1), (a1, b0)]

            box = vc.CropBox("b", outlines={
                tuple(xy): {10.0 + k: polygon(xy) for k in range(40)},
                tuple(yz): {6.0: polygon(yz)}})
            ortho.cursor["y"] = 15.0           # the XZ view's slice
            plane = tuple(view.plane)
            ix = plane.index("x")

            def marks(zoom):
                segments, colors = ortho._crossing_marks(box, view, zoom)
                centres = np.asarray(segments).mean(axis=1)[: len(segments)
                                                            // 2]
                return centres, colors[: len(segments) // 2]

            # zoomed in far (spacing 0.3 µm): every mark, 2 per XY polygon
            # and 2 from the YZ polygon (where it crosses z = 10 and 70)
            centres, _ = marks(100.0)
            assert len(centres) == 82, len(centres)
            # 1 screen pixel per µm (spacing 30 µm): the XY marks thinned
            centres, colors = marks(1.0)
            from_yz = np.isclose(centres[:, ix], 6.0)
            assert from_yz.sum() == 2, "a view's marks hid another view's"
            xy_marks = centres[~from_yz]
            assert 2 <= len(xy_marks) < 10, len(xy_marks)
            gaps = np.linalg.norm(xy_marks[:, None] - xy_marks[None], axis=2)
            assert gaps[np.triu_indices(len(xy_marks), 1)].min() \
                >= ov.MARK_SPACING - 1e-9
            yz_color = np.round(ov.rgba(ov.PLANE_COLORS["YZ"]), 3)
            assert np.allclose(np.round(colors[from_yz], 3), yz_color)
            # switched off in the panel (View: Show "× marks")
            panel.marks_check.setChecked(False)
            assert not ortho.show_marks
            assert len(ortho._crossing_marks(box, view, 1.0)[0]) == 0
            panel.marks_check.setChecked(True)
            assert len(marks(1.0)[0]) == len(centres)
        finally:
            viewer.close()
            _pump()


def test_layer_list_row_without_data():
    """The user's report (2026-11-08): opening another image ended in
    "AttributeError: 'NoneType' object has no attribute 'height'" in
    napari's LayerDelegate._paint_loading, which napari reaches for a row
    whose layer has just gone (it then gives no data). The window's guard
    skips the loading indicator for such a row; napari's own code alone
    fails as the user saw (if it no longer does, napari has fixed it and
    crop_gui._guard_layer_list can go)."""
    from napari._qt.containers._layer_delegate import LayerDelegate
    from qtpy.QtGui import QPainter, QPixmap, QStandardItem, QStandardItemModel
    from qtpy.QtWidgets import QStyleOptionViewItem

    from volume_cropper import crop_gui
    viewer, panel = crop_gui.open_window(show=False)
    try:
        delegate = viewer.window._qt_viewer.layers.itemDelegate()
        assert getattr(delegate, "_volume_cropper_guard", False)
        model = QStandardItemModel(1, 1)       # a row without any data
        model.setItem(0, 0, QStandardItem())
        index = model.index(0, 0)
        pixmap = QPixmap(120, 40)
        painter = QPainter(pixmap)
        try:
            delegate._paint_loading(painter, QStyleOptionViewItem(), index)
            try:
                LayerDelegate._paint_loading(delegate, painter,
                                             QStyleOptionViewItem(), index)
            except AttributeError as err:
                assert "height" in str(err)
            else:
                raise AssertionError("napari's _paint_loading copes now")
        finally:
            painter.end()
    finally:
        viewer.close()
        _pump()


def test_no_tiff_button_for_files():
    """A file read into memory: its crops are TIFFs, so no "Save as
    TIFF…" and no TIFF resolution."""
    import tifffile

    from volume_cropper import crop_gui
    from volume_cropper import file_images as fi
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stack.tif"
        tifffile.imwrite(path, np.zeros((4, 16, 16), np.uint16))
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(path, file_choice=fi.FileChoice())
            _pump()
            assert not panel.tiff_button.isVisibleTo(panel)
            assert not panel.tiff_level_combo.isVisibleTo(panel)
        finally:
            viewer.close()
            _pump()


def test_polygon_without_area_is_refused():
    """(The user's choice of 2026-11-10) A polygon with (almost) no area,
    e.g. three clicks on a line, is not used (the shape would narrow to
    nothing there): no region is made, and the panel says why; an edit
    that flattens a polygon leaves it as it was."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            flat = [(10.0, 5.0), (10.0, 10.0), (10.0, 15.0)]
            panel._box_added({"y": (10.0, 10.0), "x": (5.0, 15.0)},
                             ("y", "x"), flat)
            _pump()
            assert panel.boxes == [], panel.boxes
            assert "no area" in panel.status.text()
            good = [(5.0, 5.0), (5.0, 15.0), (15.0, 15.0), (15.0, 5.0)]
            panel._box_added({"y": (5.0, 15.0), "x": (5.0, 15.0)},
                             ("y", "x"), good)
            _pump()
            box = panel.boxes[0]
            before = vc.normalize_outlines(box.outlines)
            plane, keys = next(iter(box.outlines.items()))
            key = next(iter(keys))
            panel._boxes_changed({0: ({}, flat, key)}, plane)
            _pump()
            assert vc.normalize_outlines(box.outlines) == before
        finally:
            viewer.close()
            _pump()


def test_level_shown_with_channel_0_hidden():
    """(The 0.17.0 review) napari does not load hidden layers, and the
    level shown was taken from channel 0: with it hidden, the views'
    titles and the "Auto" entry kept an old level (or none)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            ortho = panel.ortho
            ortho.set_channel_visible(0, False)
            _pump()
            ortho.set_2d_level(1, False)
            _pump()
            xy = ortho.views["XY"]
            assert ortho.current_level(xy) == 1
            assert ortho._levels_shown.get("XY") == 1
            assert "level 1" in xy.title.text(), xy.title.text()
        finally:
            viewer.close()
            _pump()


def test_select_several_regions_and_polygons_and_delete():
    """(0.18.0, the user's wish) Several regions, and several polygons,
    can be selected (Ctrl/Shift+click, Ctrl+A) and deleted at once; the
    panel asks first if there are several. The region clicked last is the
    selected region (in bold); refreshing the panel keeps the others
    selected; Ctrl+click on the selected region takes it out, and the last
    one still selected becomes the selected region."""
    from volume_cropper import crop_gui
    from qtpy.QtCore import QItemSelectionModel
    confirm = crop_gui._confirm
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            for k in range(4):
                panel._new_box({"y": (5.0 + k, 30.0), "x": (5.0, 30.0)})
            panel._refresh(select=0)
            regions = panel.box_list

            def selected_rows(widget):
                return {widget.row(item) for item in widget.selectedItems()}

            # as Ctrl+click on rows 1 and 3 would (row 3 last)
            regions.item(1).setSelected(True)
            regions.item(3).setSelected(True)
            regions.setCurrentRow(
                3, QItemSelectionModel.SelectionFlag.NoUpdate)
            _pump(0.1)
            assert panel._selected() == 3
            assert selected_rows(regions) == {0, 1, 3}
            assert regions.item(3).font().bold()
            assert not regions.item(0).font().bold()
            panel._refresh(select=3)           # e.g. after a drawing
            assert selected_rows(regions) == {0, 1, 3}
            # Ctrl+click on the selected region takes it out
            regions.item(3).setSelected(False)
            _pump(0.1)
            assert panel._selected() == 1, panel._selected()
            assert selected_rows(regions) == {0, 1}
            assert regions.item(1).font().bold()

            # Delete asks first: "No" keeps both, "Yes" deletes both
            asked = []
            crop_gui._confirm = lambda *a: asked.append(a[2]) or False
            panel._remove_box()
            assert len(panel.boxes) == 4
            assert "Delete 2 crop regions (crop01, crop02)" in asked[0], asked
            crop_gui._confirm = lambda *a: asked.append(a[2]) or True
            panel._remove_box()
            assert [b.name for b in panel.boxes] == ["crop03", "crop04"]
            assert panel._selected() == -1 and not regions.selectedItems()
            # one region: deleted without a question
            asked.clear()
            panel._refresh(select=0)
            panel._remove_box()
            assert asked == [] and [b.name for b in panel.boxes] == ["crop04"]

            # polygons on three slices; two of them deleted at once
            box = panel.boxes[0]
            square = [(8.0, 8.0), (8.0, 25.0), (25.0, 25.0), (25.0, 8.0)]
            box.outlines = {("y", "x"): {20.0: square, 40.0: square,
                                         60.0: square}}
            panel._refresh(select=0)
            polygons = panel.polygon_list
            assert polygons.count() == 3
            polygons.item(0).setSelected(True)
            polygons.item(2).setSelected(True)
            asked.clear()
            panel._delete_polygon()
            assert len(asked) == 1 and "Delete 2 polygons" in asked[0], asked
            assert list(box.outlines[("y", "x")]) == [40.0], box.outlines
            assert polygons.count() == 1
            # a single polygon: no question
            asked.clear()
            polygons.setCurrentRow(0)
            panel._delete_polygon()
            assert asked == [] and not box.outlines, box.outlines
        finally:
            crop_gui._confirm = confirm
            viewer.close()
            _pump()


def test_level_shown_while_another_image_opens_at_a_coarse_level():
    """(0.18.0, the user's IndexError of 2026-11-13) Opening a second image
    at level 3 sets that choice before the first image's layers are
    replaced; a layer of the first one showing its level 1 then reported
    levels[1] of the new choice [3]. Each layer now keeps the levels it was
    built with; then opening at level 3 works and fits the views."""
    from volume_cropper import ortho_views as ov
    from volume_cropper.load_dialog import LoadChoice
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            ortho = panel.ortho
            n = ortho.region.n_levels
            assert n >= 2
            view = ortho.views["XZ"]
            layer = view.channel_layers[0]
            assert layer.vc_levels == tuple(range(n))      # Auto: all
            layer._data_level = 1      # (napari's, as when zoomed out)
            ortho.level_2d = n - 1     # (as _load sets it, before set_image)
            assert ortho.current_level(view) == 1          # the layer's own
            ortho._levels_shown.pop("XZ", None)
            ortho._on_level_shown(view)                    # (raised before)
            assert ortho._levels_shown["XZ"] == 1
            # the whole opening at the coarsest level
            panel.open_image(path, choice=LoadChoice(level_2d=n - 1))
            _pump()
            assert ortho.views["XZ"].channel_layers[0].vc_levels == (n - 1,)
            assert set(ortho._levels_shown.values()) == {n - 1}
        finally:
            viewer.close()
            _pump()


def test_opening_zooms_to_the_chosen_level():
    """(0.18.0, the user's wish) Opening with a chosen 2D resolution zooms
    the 2D views in to one voxel of that level per screen pixel, centred on
    the image (napari then shows that level); with Auto, or if the image
    fits anyway, they stay fitted; changing the resolution later keeps the
    zoom (only on opening)."""
    from volume_cropper import ortho_views as ov
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            ortho = panel.ortho
            # small canvases, so that the image does not fit at level 0
            for name in ov.VIEWS_2D:
                ortho.views[name].model.canvas.size = (60, 60)

            def opened(level):
                ortho.level_2d = level
                ortho.reset_cameras()
                ortho._zoom_to_level()
                return ov.camera(ortho.views["XY"].model)

            fitted = opened(None).zoom                       # Auto
            voxel = min(panel.image.levels[0].scale[-2:])    # y, x (µm)
            cam = opened(0)
            assert abs(cam.zoom - 1 / voxel) < 1e-9 and cam.zoom > fitted
            for name in ov.VIEWS_2D:                         # one zoom
                assert ov.camera(ortho.views[name].model).zoom == cam.zoom
            centre = [float(np.mean(ortho.extent(a))) for a in ("y", "x")]
            assert np.allclose(tuple(cam.center)[-2:], centre)
            # napari's rule then picks level 0 for the XY view
            pixel = 1 / cam.zoom
            levels = ov.pyramid_levels(ortho.region.n_levels, 0, False)
            assert ov.level_for_pixel(panel.image, levels, ["y", "x"],
                                      pixel) == 0
            # a coarse level at which the image fits anyway: no zoom-in
            coarse = ortho.region.n_levels - 1
            assert opened(coarse).zoom == fitted
            # changing the resolution later keeps the zoom
            opened(0)
            before = ov.camera(ortho.views["XY"].model).zoom
            ortho.set_2d_level(1, False)
            _pump()
            assert ov.camera(ortho.views["XY"].model).zoom == before
        finally:
            viewer.close()
            _pump()


def test_channels_part_and_auto_contrast():
    """(0.18.0, the user's wish) The Channels list: a row per channel whose
    tick box shows it in all views and whose menu sets its colour (and
    follows napari's); a click chooses the channel for Levels (and the
    other way round). Levels' auto-contrast: once, continuous (stopped by
    setting a limit by hand) and reset (the starting contrast, gamma 1)."""
    from volume_cropper import ortho_views as ov
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            ortho, channels, levels = panel.ortho, panel.channels, \
                panel.levels
            xy = ortho.views["XY"].channel_layers
            assert channels.list.count() == 3 == len(panel.show_checks)
            # show / hide
            panel.show_checks[1].setChecked(False)
            _pump(0.2)
            assert not xy[1].visible and not ortho.display[1]["visible"]
            assert not ortho.views["XZ"].channel_layers[1].visible
            xy[1].visible = True                      # napari's eye icon
            _pump(0.2)
            assert panel.show_checks[1].isChecked()
            # colours
            combo = channels.colour_combos[0]
            own = ortho.display[0]["colormap"]
            combo.setCurrentIndex(combo.findData("magenta"))
            combo.activated.emit(combo.currentIndex())
            _pump(0.2)
            assert xy[0].colormap.name == "magenta"
            assert ortho.views["YZ"].channel_layers[0].colormap.name == \
                "magenta"
            combo.setCurrentIndex(combo.findData("own"))
            combo.activated.emit(combo.currentIndex())
            _pump(0.2)
            assert np.allclose(xy[0].colormap.colors, own.colors)
            xy[2].colormap = "cyan"                    # napari's controls
            _pump(0.2)
            assert channels.colour_combos[2].currentData() == "cyan"
            # choosing the channel for Levels, both ways
            channels.list.setCurrentRow(2)
            _pump(0.2)
            assert levels.channel == 2
            assert levels.channel_name.text().startswith("2:")
            levels.channel_combo.setCurrentIndex(1)    # (napari's list)
            _pump(0.2)
            assert channels.list.currentRow() == 1

            # auto-contrast once: the range of what the XY view shows
            layer = xy[1]
            layer.contrast_limits = (0, 10)
            levels.auto_once.click()
            _pump(0.2)
            assert tuple(layer.contrast_limits) != (0.0, 10.0)
            raw = np.asarray(layer._slice.image.raw)
            assert np.allclose(layer.contrast_limits,
                               (raw.min(), raw.max())), layer.contrast_limits
            # continuous, stopped by a limit set by hand
            levels.auto_continuous.click()
            assert layer.auto_contrast
            levels.max_spin.setValue(2000)
            _pump(0.2)
            assert not layer.auto_contrast
            assert not levels.auto_continuous.isChecked()
            # reset: the starting contrast and gamma 1
            layer.gamma = 0.5
            layer.contrast_limits = (0, 50)
            levels.auto_reset.click()
            _pump(0.2)
            start = ov.channel_contrast(panel.image, 1, ortho.region)
            assert np.allclose(layer.contrast_limits, start)
            assert layer.gamma == 1.0
        finally:
            viewer.close()
            _pump()


def test_drawing_tool_buttons():
    """(0.18.0, the user's wish) The drawing tools under Crop regions set
    the tool in all 2D views and show the one in use (also when chosen by
    its key); Delete shape deletes the selected shape (a rectangle: its
    region)."""
    from volume_cropper import ortho_views as ov
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        viewer, panel = _window_with_image(path)
        try:
            ortho = panel.ortho
            panel.tool_buttons["add_polygon"].click()
            _pump(0.1)
            for name in ov.VIEWS_2D:
                assert str(ortho.views[name].shapes.mode) == "add_polygon"
            checked = [m for m, b in panel.tool_buttons.items()
                       if b.isChecked()]
            assert checked == ["add_polygon"], checked
            ortho.set_tool("select")                   # as its key would
            _pump(0.1)
            assert panel.tool_buttons["select"].isChecked()
            assert not panel.tool_buttons["add_polygon"].isChecked()
            # Delete shape: the region's rectangle selected in XY
            i = panel._new_box({"y": (5.0, 20.0), "x": (5.0, 20.0)})
            panel._refresh(select=i)
            _pump(0.2)
            xy = ortho.views["XY"]
            shape = xy.shape_boxes.index(i)
            xy.shapes.selected_data = {shape}
            ortho.delete_selected_shapes()
            _pump(0.2)
            assert panel.boxes == [], panel.boxes
        finally:
            viewer.close()
            _pump()


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["panel"]))
