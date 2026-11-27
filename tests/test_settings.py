"""Tests for the 0.16.0 panel: settings remembered between sessions, the
collapsible parts, the tool's layers out of napari's layer list, and the
parts that show only when needed (hidden windows).

Run with ``python run_all_tests.py settings`` (or this file alone; both use
a settings file of their own, never the user's). They check that:
- settings are saved and read back (unknown keys and broken files give the
  defaults), and a new window starts with them (parts open or closed,
  Compress TIFFs, Link zoom/pan, napari's panels);
- opening or closing a part is remembered for the next window;
- napari's layer list hides the views' own layers (region lines,
  crosshair, preview, shading) but lists the channels and "crop regions";
  with the setting, it lists them all;
- the polygons part shows only for a region with polygons; the region list
  grows with its regions (2 to 6 rows); the channel boxes and Save regions
  are in the Output part; the Settings dialog gives back what was chosen.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import tempfile
import time
from pathlib import Path

from volume_cropper import settings
from volume_cropper import core as vc
from test_volume_cropper import make_test_image


def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


class _OwnSettings:
    """A settings file of the test's own for the duration of a `with`."""

    def __enter__(self):
        self.saved = os.environ.get("VOLUME_CROPPER_SETTINGS")
        self.folder = tempfile.mkdtemp()
        os.environ["VOLUME_CROPPER_SETTINGS"] = os.path.join(
            self.folder, "settings.json")
        return Path(os.environ["VOLUME_CROPPER_SETTINGS"])

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("VOLUME_CROPPER_SETTINGS", None)
        else:
            os.environ["VOLUME_CROPPER_SETTINGS"] = self.saved


def test_settings_saved_and_read_back():
    with _OwnSettings() as path:
        assert settings.load() == settings.DEFAULTS
        assert settings.update(compress_tiffs=True,
                               sections={"view": False})["compress_tiffs"]
        values = settings.load()
        assert values["compress_tiffs"] is True
        assert values["sections"]["view"] is False
        assert values["sections"]["output"] is True       # the others kept
        # (0.18.0) the file keeps only what differs from the defaults, so
        # that new defaults reach those who have not chosen otherwise
        assert json.loads(path.read_text()) == {
            "compress_tiffs": True, "sections": {"view": False}}
        path.write_text(json.dumps({"compress_tiffs": "yes", "other": 1,
                                    "sections": {"view": "no"}}))
        values = settings.load()                 # wrong types: the defaults
        assert values["compress_tiffs"] is False
        assert values["sections"]["view"] is True
        path.write_text("{ not json")
        assert settings.load() == settings.DEFAULTS


def test_napari_panels_hidden_at_first():
    """(0.18.0) napari's layer panels are hidden at first, also for a
    settings file of 0.16-0.17 that kept them shown under the old key
    "napari_panels" (the panel has the controls needed now)."""
    with _OwnSettings():
        assert settings.load()["napari_layer_panels"] is False
        settings.path().parent.mkdir(parents=True, exist_ok=True)
        settings.path().write_text('{"napari_panels": true}',
                                   encoding="utf-8")
        assert settings.load()["napari_layer_panels"] is False
        assert settings.load()["sections"]["channels"] is True


def test_new_window_starts_with_the_settings():
    from volume_cropper import crop_gui
    with _OwnSettings():
        settings.update(compress_tiffs=True, link_views=False,
                        napari_layer_panels=True,
                        sections={"view": False, "find": True})
        viewer, panel = crop_gui.open_window(show=False)
        try:
            assert not panel.sections["view"].is_open()
            assert panel.sections["find"].is_open()
            assert panel.sections["output"].is_open()
            assert panel.compress_check.isChecked()
            assert not panel.link_check.isChecked()
            assert panel.napari_panels_check.isChecked()
            # opening a part is remembered for the next window
            panel.sections["view"].header.click()
            assert settings.load()["sections"]["view"] is True
        finally:
            viewer.close()
            _pump()


def test_new_window_starts_with_the_autocropper_and_view_options():
    """(0.18.0, the user's wish) The Autocropper's and the View part's
    options a new window starts with, from the settings; values of the
    wrong type (or unknown entries) are ignored; the dialog shows and
    gives them back."""
    from volume_cropper import crop_gui
    with _OwnSettings():
        settings.update(
            autocropper={"method": "li", "grow": False, "margin": 4,
                         "moving": False, "outline": True,
                         "outline_kind": "slices", "seed_mode": "whole",
                         "split": False, "unknown": 1},
            view={"slice_planes": False,
                  "mask_mode": "grey", "outlines_3d": "all",
                  "outlines_3d_opacity": 0.6, "crosshairs": False,
                  "crop_regions": False, "crossing_marks": False,
                  "preview": "main",
                  "levels_source": "whole", "levels_log": "yes",
                  "levels_zoom": True})
        values = settings.load()
        assert values["autocropper"]["margin"] == 4      # a whole number
        assert "unknown" not in values["autocropper"]
        assert values["view"]["levels_log"] is False     # wrong type
        viewer, panel = crop_gui.open_window(show=False)
        try:
            finder = panel.object_group
            assert finder.method_combo.currentData() == "li"
            assert not finder.grow_check.isChecked()
            assert finder.margin_spin.value() == 4.0
            assert not finder.moving_check.isChecked()
            assert finder.outline_check.isChecked()
            assert finder.outline_kind_combo.currentData() == "slices"
            assert finder.seed_mode_combo.currentData() == "whole"
            assert not finder.split_check.isChecked()
            assert not panel.planes_check.isChecked()
            assert panel.mask_combo.currentData() == "grey"
            assert panel.ortho.mask_mode == "grey"
            assert panel.outlines_3d_combo.currentData() == "all"
            assert panel.ortho.outlines_3d == "all"
            assert abs(panel.ortho.outlines_3d_opacity - 0.6) < 1e-9
            assert not panel.crosshairs_check.isChecked()
            assert not panel.boxes_check.isChecked()
            assert not panel.marks_check.isChecked()           # (0.20.0)
            assert not panel.ortho.show_marks
            assert panel.preview_combo.currentData() == "main"
            assert panel.levels.source == "whole"
            assert not panel.levels.log_check.isChecked()
            assert panel.levels.zoom_check.isChecked()         # (0.22.0)
            dialog = crop_gui.SettingsDialog()
            assert dialog.values()["view"]["levels_zoom"] is True
            given = dialog.values()
            assert given["autocropper"]["outline_kind"] == "slices"
            assert given["view"]["outlines_3d"] == "all"
            assert given["view"]["crossing_marks"] is False
            dialog.view_widgets["preview"].setCurrentIndex(0)
            assert dialog.values()["view"]["preview"] == "off"
        finally:
            viewer.close()
            _pump()


def test_opening_starts_with_the_settings_and_may_skip_the_dialog():
    """(0.18.0, the user's wish) Opening an image starts with the settings:
    the 2D resolution by voxel size (the finest level with voxels of at
    least the size given; the coarsest if none), the 3D view, the first
    channel only; with "Ask 'What to load'" off it opens at once (the
    dialog is not shown), unless the 3D view would read too much."""
    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    from volume_cropper.load_dialog import LoadChoice
    with _OwnSettings(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        image = vc.open_image(path)
        sizes = [level.scale[-1] for level in image.levels]
        assert ov.level_for_voxel(image, sizes[1]) == 1
        assert ov.level_for_voxel(image, sizes[1] * 0.99) == 1
        assert ov.level_for_voxel(image, sizes[1] * 1.01) == 2
        assert ov.level_for_voxel(image, 1e6) == len(sizes) - 1
        settings.update(opening={"level_2d": "size",
                                 "voxel_2d": sizes[1] * 1.01,
                                 "view_3d": "auto", "voxel_3d": 8.0,
                                 "channels": "first", "ask": False})
        viewer, panel = crop_gui.open_window(show=False)
        saved = crop_gui.LoadDialog.exec
        shown = []
        crop_gui.LoadDialog.exec = lambda self: shown.append(1) or 0
        try:
            choice = panel._start_choice(image)
            assert choice == LoadChoice(channels=[0], level_2d=2,
                                        show_3d=True, level_3d=None)
            panel.open_image(path, ask=True)        # as Open... does
            _pump()
            assert not shown, "the dialog should not ask"
            assert panel.image is not None
            assert panel.ortho.level_2d == 2 and panel.ortho.show_3d
            visible = [k for k, d in enumerate(panel.ortho.display)
                       if d["visible"]]
            assert visible == [0]
            # Ask on: the dialog asks (Cancel keeps the image open)
            settings.update(opening={"ask": True})
            panel._settings = settings.load()
            panel.open_image(path, ask=True)
            assert shown == [1]
        finally:
            crop_gui.LoadDialog.exec = saved
            viewer.close()
            _pump()


def test_tool_layers_hidden_in_napari_layer_list():
    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    from volume_cropper.load_dialog import LoadChoice
    with _OwnSettings(), tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump(0.5)
            view = viewer.window._qt_viewer.layers
            model = view.model()
            layers = list(viewer.layers)
            hidden, listed = [], []
            for row in range(model.rowCount()):
                layer = layers[model.mapToSource(model.index(row, 0)).row()]
                (hidden if view.isRowHidden(row) else listed).append(
                    layer.name)
            assert "crosshair" in hidden and "region lines" in hidden
            assert "crop regions" in listed
            assert any(name.startswith("0:") for name in listed)
            assert all(not ov.is_helper_layer(viewer.layers[n])
                       for n in listed)
            # with the setting, every layer is listed
            values = settings.load()
            values["show_tool_layers"] = True
            panel._apply_settings(values)
            assert not any(view.isRowHidden(r)
                           for r in range(model.rowCount()))
        finally:
            viewer.close()
            _pump()


def test_parts_shown_when_needed():
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    with _OwnSettings(), tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump()
            # the polygons part: only for a region with polygons (regions
            # made as the window makes them: all axes set, _new_box)
            plain = panel._new_box({"z": (10.0, 50.0)})
            panel._refresh(select=plain)
            assert not panel.polygon_part.isVisibleTo(panel)
            square = [(5.0, 5.0), (5.0, 20.0), (20.0, 20.0), (20.0, 5.0)]
            shaped = panel._new_box({"z": (10.0, 50.0)})
            panel.boxes[shaped].outlines = vc.normalize_outlines(
                {("y", "x"): square})
            panel._refresh(select=shaped)
            assert panel.polygon_part.isVisibleTo(panel)
            # the region list grows with its regions (2 to 6 rows)
            heights = []
            for n in (1, 3, 6, 9):
                panel.boxes = []
                for _ in range(n):
                    panel._new_box({"z": (10.0, 50.0)})
                panel._refresh(select=0)
                heights.append(panel.box_list.height())
            assert heights[0] < heights[1] < heights[2] == heights[3], heights
            # channels and Save regions in the Output part
            output = panel.sections["output"].body
            assert panel.channel_checks and all(
                output.isAncestorOf(check) for check in panel.channel_checks)
            save = [b for b in output.findChildren(crop_gui.QPushButton)
                    if b.text() == "Save regions…"]
            assert save
        finally:
            viewer.close()
            _pump()


def test_settings_ok_leaves_an_open_image_alone():
    """The review of 2026-11-08: OK in Settings… applies "New windows start
    with" only to windows without an image (a window with one keeps its
    Link zoom/pan and Compress TIFFs; re-ticking Link reset its zoom); the
    memory cache is compared with napari's own setting (not a stale spin
    box) and shown in the window; the exact-level tick from the settings is
    what a cancelled question restores."""
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    dask = crop_gui._dask_settings()
    saved = (dask.enabled, dask.cache)
    saved_exec = crop_gui.SettingsDialog.exec
    with _OwnSettings(), tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = crop_gui.open_window(show=False)
        try:
            settings.update(exact_level=True)
            panel._settings = settings.load()
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump()
            assert panel._view_state["exact"] is panel.exact_check.isChecked()
            panel.link_check.setChecked(False)
            panel.compress_check.setChecked(True)
            # napari's cache changed elsewhere (4 GB); this window's spin box
            # still shows 2 GB; the dialog asks for 2 GB
            dask.enabled, dask.cache = True, 4.0
            panel.cache_spin.blockSignals(True)
            panel.cache_spin.setValue(2.0)
            panel.cache_spin.blockSignals(False)

            def answer(dialog):
                dialog.cache_spin.setValue(2.0)
                dialog.checks["link_views"].setChecked(True)
                dialog.checks["compress_tiffs"].setChecked(False)
                return 1
            crop_gui.SettingsDialog.exec = answer
            panel._open_settings()
            assert not panel.link_check.isChecked()          # kept
            assert panel.compress_check.isChecked()          # kept
            assert abs(crop_gui._dask_settings().cache - 2.0) < 1e-9
            assert abs(panel.cache_spin.value() - 2.0) < 1e-9
            # a window without an image takes them
            other_viewer, other = crop_gui.open_window(show=False)
            try:
                assert other.link_check.isChecked()
                assert not other.compress_check.isChecked()
            finally:
                other_viewer.close()
        finally:
            crop_gui.SettingsDialog.exec = saved_exec
            dask.enabled, dask.cache = saved
            viewer.close()
            _pump()


def test_settings_dialog_gives_back_the_choices():
    from volume_cropper import crop_gui
    with _OwnSettings():
        dialog = crop_gui.SettingsDialog()
        dialog.checks["compress_tiffs"].setChecked(True)
        dialog.section_checks["ranges"].setChecked(True)
        values = dialog.values()
        assert values["compress_tiffs"] is True
        assert values["sections"]["ranges"] is True
        dialog.deleteLater()


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["settings"]))
