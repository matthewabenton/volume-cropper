"""Volume Cropper's napari GUI: draw crop regions on a large OME-Zarr image
and save the crops.

Start it with volume_cropper.bat (Windows PC) or volume_cropper.sh
(Linux/BARD), or, from the tool folder::

    python -m volume_cropper [IMAGE.zarr]

The window shows four linked views (see ortho_views.py)::

    +-----------+------+
    |    XY     |  YZ  |
    +-----------+------+
    |    XZ     |  3D  |
    +-----------+------+

How to use it:

1. Open an image ("Open..." or give the path). The "What to load" dialog
   (load_dialog.py) shows what the image holds and lets you choose the
   channels, a region and the levels to read; the 3D view is off at first.
   Like a virtual stack in Fiji, only what is on screen is then read, from a
   pyramid level that suits the zoom, or from the level you choose.
   Since 0.12.0 TIFF files (and folders of TIFFs) open too: the part chosen
   is read into memory (file_images.py), and crops are saved as TIFF.
   A 2D image (one plane, e.g. a 2D movie) shows only the XY view.
2. "New region", then draw polygons (P) or lassos (Shift+P) around the
   object on a few slices of any view (on a projection: its silhouette), or
   a rectangle (R) for a plain box; "Finish region" when done. Drawing goes
   into the selected region (a new one if none is selected). XY sets y and
   x, XZ sets z and x, YZ sets y and z. Fine-tune with the number fields.
3. Hover over a 2D view and press J to move the other views to that point.
4. In 3D the regions are wireframes, and "Clip 3D to selected region" shows
   only what will be cropped.
5. Choose an output folder and press "Crop selected" or "Crop all".
   "Save regions" writes a CSV (a "boxes file") that examples/volume_cropper.ipynb
   can use to run large crops on the cluster.
6. "Open last crop in a new window" opens the result in a second, complete
   Volume Cropper window, to check it or to crop it further.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import functools
import shutil
import threading
import time
from pathlib import Path

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import (QEvent, QEventLoop, QItemSelectionModel, QObject, Qt,
                         QTimer, Signal)
from qtpy.QtGui import QColor, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox,
    QColorDialog, QComboBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QProgressBar,
    QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QToolButton, QVBoxLayout, QWidget)

from volume_cropper import box_handles
from volume_cropper import file_images as fi
from volume_cropper import ortho_views as ov
from volume_cropper import settings
from volume_cropper import core as vc
# (0.23.0) labels and message boxes that show plain text: names from
# files must never be taken for HTML (see qt_text)
from volume_cropper.qt_text import QLabel, QMessageBox, escape, literal
from volume_cropper.autocrop_panel import ObjectFinder
from volume_cropper.channels_panel import ChannelsPanel
from volume_cropper.levels_panel import LevelsPanel
from volume_cropper.load_dialog import FileLoadDialog, LoadChoice, LoadDialog

try:                       # to show how much opening read (napari needs it)
    import psutil
except ImportError:
    psutil = None

# Ask before choosing a view setting that reads or holds this much data.
LARGE_PLANE_READ = 500e6      # bytes for one whole plane (exact 2D level)
LARGE_3D_VOLUME = 1.5e9       # bytes for the 3D view's volume
LARGE_PROJECTION_READ = 1e9   # bytes for one projected 2D view

# (0.18.0) The "Loading…" box: napari reads the data for the views in the
# window's own thread, so the window can't respond or redraw meanwhile.
# The box closes when the window has been free for LOADING_QUIET_S without
# reading, or for LOADING_FREE_S in any case (then the reading is not
# napari's, e.g. the crop preview's in the background). A timer tick more
# than LOADING_LATE_S late means the window was busy.
# (0.18.0) The drawing tools under Crop regions: napari's Shapes modes,
# with napari's keys for them.
DRAWING_TOOLS = (
    ("pan_zoom", "Move view", "Pan and zoom the views (Z); drawing stops."),
    ("add_rectangle", "Rectangle", "Draw a rectangle: the region's range in "
     "that view (R)."),
    ("add_polygon", "Polygon", "Draw a polygon on this slice: click the "
     "corners, double-click to close (P)."),
    ("add_polygon_lasso", "Lasso", "Draw a free outline on this slice: hold "
     "the mouse button down (Shift+P)."),
    ("select", "Select", "Select shapes to move, resize or delete them (S)."),
    ("direct", "Edit points", "Move single corners of a polygon (D)."),
)

# (0.18.0) Undo keeps this many steps; changes less than UNDO_MERGE_S
# apart (a drag in 3D, Up/Down held, a number field's arrows held) are one.
UNDO_STEPS = 50
UNDO_MERGE_S = 0.5

LOADING_TICK_MS = 100
LOADING_LATE_S = 0.3
LOADING_QUIET_S = 0.6
LOADING_FREE_S = 2.0

# Crops made in the window (0.15.0: no longer settings there): pyramid
# levels are kept while the crop is at least this many voxels along its
# longest axis; blocks are copied by this many threads; crops keep their
# position (they overlay the full image).
MIN_LEVEL_SIZE = 64
WORKERS = 8
KEEP_POSITION = True


def display_unit(unit: str) -> str:
    return "µm" if unit == "micrometer" else vc.short_unit(unit)


# ---------------------------------------------------------------------------
# Running crops in a background thread
# ---------------------------------------------------------------------------

@thread_worker
def _run_crops(image, boxes, out_dir, min_size, options, report):
    """Crop several boxes one after the other (runs outside the GUI thread,
    so napari stays responsive).

    report(done, total, text) shows progress over all the boxes together, in
    bytes of image data written (all levels), which a time estimate needs:
    pieces of the finest level are much larger than those of coarse levels.
    """
    vc.check_output_folder(image, out_dir)    # before the folder is made
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = image.levels[0].array.dtype
    options = dict(options)
    tiff_level = options.get("tiff_level")
    tiff_only = options.pop("tiff_only", False)
    # (0.21.0) a zipped image's crops are packed into a zip file at the end
    zipped = (image.container is not None and not image.in_memory
              and not tiff_only)
    sizes = []      # per box: (crop, TIFF, packing) in bytes
    for box in boxes:
        if image.in_memory:     # a file's image: the crop is the TIFF
            plan = vc.plan_crop(image, box, n_levels=1)
            sizes.append((0, vc.nbytes(plan.shapes[0], dtype), 0))
            continue
        if tiff_only:           # "Save as TIFF…": one level, then the TIFF
            plan = vc.tiff_plan(image, box, tiff_level)
            size = vc.nbytes(plan.shapes[vc.tiff_level_for(plan, tiff_level)],
                             dtype)
            sizes.append((size, size, 0))
            continue
        plan = vc.plan_crop(image, box, min_size=min_size)
        crop_size = sum(vc.nbytes(shape, dtype) for shape in plan.shapes)
        sizes.append((crop_size,
                      0 if tiff_level is None
                      else vc.tiff_bytes(plan, dtype, tiff_level),
                      # (counted as the crop's size: the packed bytes are
                      # compressed, their number is known only then)
                      crop_size if zipped else 0))
    total, before = sum(sum(size) for size in sizes), 0
    results = []
    for n, box in enumerate(boxes, start=1):
        text = f"{box.name} ({n}/{len(boxes)})"
        crop_size, tiff_size, pack_size = sizes[n - 1]

        def packed(done, size, text=text,
                   before=before + crop_size + tiff_size, share=pack_size):
            # after the last byte the zip file is checked
            report(before + share * done / max(size, 1), total, text + (
                ", packing it into one zip file" if done < size
                else ", checking the zip file"))

        def written(done, size, text=text, before=before):
            # after the last piece the crop is checked against the source
            report(before + done, total, text + (
                ", checking it against the source" if done == size else ""))

        def tiff_written(done, size, text=text, before=before + crop_size):
            # after the last plane the TIFF is checked against the crop
            report(before + done, total, text + (
                ", saving the TIFF" if done < size
                else ", checking the TIFF against the crop"))
        if image.in_memory:
            report(before, total, text + ", cutting it out")
        if tiff_only:
            level = options.pop("tiff_level", None)
            results.append(vc.save_region_tiff(
                image, box, out_dir, level, min_size=min_size,
                progress_bytes=written, tiff_progress=tiff_written,
                **options))
            options["tiff_level"] = level
        else:
            results.append(vc.crop_box(image, box, out_dir,
                                       min_size=min_size,
                                       progress_bytes=written,
                                       tiff_progress=tiff_written,
                                       pack_progress=packed, **options))
        before += crop_size + tiff_size + pack_size
    return results


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class CropWidget(QWidget):
    """The panel: define crop boxes on an OME-Zarr image and crop them."""

    # Qt widgets may only be changed from the main (GUI) thread. A signal
    # emitted in the worker thread is queued and handled in the GUI thread.
    progress_signal = Signal(float, float, str)    # bytes (> 2 GB possible)

    def __init__(self, viewer: napari.Viewer, path: str | None = None):
        super().__init__()
        self.viewer = viewer
        self.ortho = ov.OrthoViews(viewer)
        self.ortho.on_box_added = self._box_added
        self.ortho.on_boxes_changed = self._boxes_changed
        self.ortho.on_boxes_removed = self._boxes_removed
        self.ortho.on_box_ranges = self._box_ranges_dragged    # (0.18.0)
        self.ortho.on_box_picked = lambda i: self._refresh(select=i)
        self.ortho.on_levels_shown = self._levels_shown
        self.ortho.on_display = self._on_display_changed
        self.ortho.on_time_changed = self._on_time_changed     # (0.17.0)
        # (0.16.0) "Autocropper" stops adding seeds when a drawing
        # tool is chosen
        self.ortho.on_tool_changed = self._on_tool_changed
        self._panel_fitted = False
        self.image: vc.OmeZarrImage | None = None
        self.boxes: list[vc.CropBox] = []      # the crop regions
        # A region made by "New region" and its ranges then: removed again on
        # "Finish region" if nothing was drawn into it.
        self._fresh: tuple | None = None
        self.range_spins: dict = {}
        self.t_step_spin = None         # "every n-th timepoint" (time-lapse)
        self.channel_checks: list = []
        self._cancel = threading.Event()
        self._worker = None
        self._last_output: Path | None = None
        self._crop_started = 0.0          # time.time() when cropping began
        self._view_state = {"level_2d": 0, "exact": False, "show_3d": False,
                            "level_3d": 0, "projection": 0, "through": 0}
        # (the last accepted state of the view controls, for "No" answers)
        self._reading = None      # how much opening has read (_watch_reading)
        self._file_choice = None  # what was read of an image file (0.12.0)
        # napari layers the open image was read from (0.13.0, the plugin):
        # "What to load…" reads them again
        self._layer_reader = None
        self._busy = False        # a crop is running
        self._read_text = ""      # how long reading it into memory took
        self._read_timer = QTimer(self)
        self._read_timer.setInterval(500)
        self._read_timer.timeout.connect(self._show_reading)
        # (0.18.0) undo / redo of the regions' changes (_record_regions):
        # snapshots (text to compare, the regions, the selected one)
        self._undo: list = []
        self._redo: list = []
        self._regions_state = None
        self._last_change = 0.0
        self._restoring = False
        self._box_objects: dict = {}    # every region object, by its id
        self._uids = 0
        # (0.18.0) the "Loading…" box while napari reads for the views
        self._loading: dict | None = None
        self._loading_notice: _LoadingNotice | None = None
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(LOADING_TICK_MS)
        self._loading_timer.timeout.connect(self._check_loading)

        self._build_ui()
        self.ortho.on_xy_data = self.levels.on_xy_data        # (0.18.0)
        self.ortho.on_crop_projection = self._show_preview_info  # (0.18.0)
        guard_wheel(self)        # scrolling the panel changes no values
        self.progress_signal.connect(self._on_progress)
        self._watch_layer_list()          # (0.16.0) the tool's layers hidden
        _guard_layer_list(self.viewer)    # (0.16.0) napari's paint bug
        app = QApplication.instance()
        if app is not None:   # stop a running crop when napari is closed
            app.aboutToQuit.connect(self._cancel.set)
        if path:
            self.open_image(path)

    # --- building the panel ------------------------------------------------

    def _section(self, key: str, title: str) -> Section:
        """A part of the panel that opens and closes (0.16.0); whether it
        is open is remembered (settings.py)."""
        section = Section(title, key,
                          self._settings["sections"].get(key, True),
                          on_toggle=_remember_section)
        self.sections[key] = section
        return section

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._settings = settings.load()
        self.sections: dict = {}
        row = QHBoxLayout()
        row.addStretch()
        settings_button = _button("Settings…", self._open_settings)
        settings_button.setToolTip(
            "Your choices for new windows (compression, linked views, exact "
            "level, napari's panels), the parts of this panel that are "
            "open, and the memory cache; remembered for your next "
            "sessions.")
        row.addWidget(settings_button)
        layout.addLayout(row)

        group = self._section("image", "Image")
        inner = QVBoxLayout(group.body)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        row.addWidget(self.path_edit)
        # (0.12.0) An OME-Zarr is a folder, a TIFF a file: two file dialogs.
        open_button = QPushButton("Open…")
        menu = QMenu(open_button)
        menu.addAction("OME-Zarr image (.zarr folder)…", self._choose_image)
        # (0.21.0) a zipped OME-Zarr is a file
        menu.addAction("Zipped OME-Zarr (.ozx, .zip)…", self._choose_zipped)
        menu.addAction("Image file (TIFF, LIF, CZI, ND2, OIR)…",
                       self._choose_file)
        menu.addAction("Folder of TIFF files (one per plane)…",
                       self._choose_tiff_folder)
        open_button.setMenu(menu)
        open_button.setToolTip(
            "OME-Zarr images are read piece by piece as you look around, "
            "also zipped ones (.ozx, .zip), whose crops are zipped too. "
            "Image files (TIFF, and Leica LIF, Zeiss CZI, Nikon ND2, Olympus "
            "OIR) are read into memory (the part you choose), like opening "
            "them in Fiji; their crops are saved as TIFF (OME-TIFF for the "
            "microscope formats).")
        row.addWidget(open_button)
        self.load_button = _button("What to load…", self._change_what_to_load)
        self.load_button.setToolTip(
            "Choose again which channels, part of the image and levels are "
            "read (the crop regions stay).")
        self.load_button.setEnabled(False)
        row.addWidget(self.load_button)
        inner.addLayout(row)
        self.image_info = _label("No image open.")
        inner.addWidget(self.image_info)
        self.read_info = _label("")
        self.read_info.setToolTip(
            "Data read from disk or network since the image was opened "
            "(compressed). The window may pause while napari reads.")
        inner.addWidget(self.read_info)
        layout.addWidget(group)

        group = self._section("view", "View")
        outer = QVBoxLayout(group.body)
        # (0.16.0) the memory cache first, the user's wish
        outer.addLayout(self._build_cache_controls())
        grid = QGridLayout()
        outer.addLayout(grid)
        self.level_combo = QComboBox()
        self.level_combo.setToolTip(
            "Resolution of the 2D views. Auto: chosen from the zoom.")
        self.level_combo.activated.connect(lambda _: self._on_level_2d())
        self.exact_check = QCheckBox("exact level")
        self.exact_check.setToolTip(
            "Off: the chosen level is the finest shown; zooming out may still "
            "switch to coarser levels.\nOn: always exactly this level (can be "
            "slow when zoomed out at a fine level).")
        self.exact_check.toggled.connect(lambda _: self._on_level_2d())
        grid.addWidget(QLabel("2D resolution"), 0, 0)
        grid.addWidget(self.level_combo, 0, 1)
        grid.addWidget(self.exact_check, 0, 2)
        self.show_3d_check = QCheckBox("3D view")
        self.show_3d_check.setToolTip(
            "Off at first: the 3D view reads a whole volume (of the loaded "
            "region), which takes time.")
        self.show_3d_check.toggled.connect(lambda _: self._on_3d())
        self.level_3d_combo = QComboBox()
        self.level_3d_combo.activated.connect(lambda _: self._on_3d())
        grid.addWidget(self.show_3d_check, 1, 0)
        grid.addWidget(self.level_3d_combo, 1, 1, 1, 2)
        self.clip_check = QCheckBox("Clip 3D to selected region")
        self.clip_check.toggled.connect(self.ortho.set_clip)
        grid.addWidget(self.clip_check, 2, 0, 1, 2)
        self.planes_check = QCheckBox("slice planes")
        self.planes_check.setToolTip(
            "Outline the 2D views' planes in 3D (blue XY, green XZ, red YZ); "
            "during a projection, the projected slab.")
        self.planes_check.setChecked(True)
        self.planes_check.toggled.connect(self.ortho.set_planes_3d)
        grid.addWidget(self.planes_check, 2, 2)
        self.mask_combo = QComboBox()
        for text, mode in (("show normally", "show"), ("dim", "dim"),
                           ("grey", "grey"), ("dim and grey", "dimgrey"),
                           ("hide (as saved)", "hide")):
            self.mask_combo.addItem(text, mode)
        self.mask_combo.setCurrentIndex(1)                 # dim
        self.mask_combo.setToolTip(
            "How the 3D view shows the voxels of the selected region that "
            "will be set to 0.\n'hide' with 'Clip 3D to selected region' "
            "shows the crop as it will be saved.")
        self.mask_combo.activated.connect(
            lambda _: self.ortho.set_mask_mode(self.mask_combo.currentData()))
        grid.addWidget(QLabel("Blacked-out voxels (3D)"), 3, 0)
        grid.addWidget(self.mask_combo, 3, 1, 1, 2)
        # (0.18.0, the user's wish) the shape the polygons make, in 3D
        self.outlines_3d_combo = QComboBox()
        for text, mode in ov.OUTLINE_3D_MODES:
            self.outlines_3d_combo.addItem(text, mode)
        self.outlines_3d_combo.setCurrentIndex(
            self.outlines_3d_combo.findData("all"))
        self.outlines_3d_combo.setToolTip(
            "The shape a region's polygons make (what its crop keeps), as a "
            "see-through surface in the region's colour in the 3D view: "
            "for all regions (at first), the selected one, or off. It "
            "follows "
            "the polygons when you finish drawing or editing them.")
        self.outlines_3d_combo.activated.connect(lambda _: self._on_outlines_3d())
        self.outlines_3d_opacity = QDoubleSpinBox()
        self.outlines_3d_opacity.setRange(0.05, 1.0)
        self.outlines_3d_opacity.setSingleStep(0.05)
        self.outlines_3d_opacity.setDecimals(2)
        self.outlines_3d_opacity.setValue(ov.OUTLINE_3D_OPACITY)
        self.outlines_3d_opacity.setToolTip("How opaque the surfaces are.")
        self.outlines_3d_opacity.valueChanged.connect(
            lambda _: self._on_outlines_3d())
        grid.addWidget(QLabel("Outlines in 3D"), 4, 0)
        grid.addWidget(self.outlines_3d_combo, 4, 1)
        grid.addWidget(self.outlines_3d_opacity, 4, 2)
        self.projection_combo = QComboBox()
        for text, mode in (("No projection", "none"), ("Max (MIP)", "max"),
                           ("Mean (average)", "mean")):
            self.projection_combo.addItem(text, mode)
        self.projection_combo.setToolTip(
            "Project each 2D view along its depth axis: XY along z, XZ along "
            "y, YZ along x.")
        self.projection_combo.activated.connect(lambda _: self._on_projection())
        self.through_combo = QComboBox()
        self.through_combo.addItem("whole depth", "full")
        self.through_combo.addItem("selected region", "box")
        self.through_combo.addItem("slab around the slice", "slab")
        self.through_combo.addItem("chosen ranges", "range")
        self.through_combo.setToolTip(
            "What each view projects through: the whole depth loaded, the "
            "selected region's range, a slab around the slice (it follows "
            "the slider), or a range per view.\nA polygon drawn on a "
            "projection is a silhouette through that depth.")
        self.through_combo.setEnabled(False)
        self.through_combo.activated.connect(lambda _: self._on_projection())
        grid.addWidget(QLabel("2D projection"), 5, 0)
        grid.addWidget(self.projection_combo, 5, 1)
        grid.addWidget(self.through_combo, 5, 2)
        grid.addWidget(self._build_projection_options(), 6, 0, 1, 3)
        self.link_check = QCheckBox("Link zoom/pan of the 2D views")
        self.link_check.setChecked(True)
        self.link_check.toggled.connect(self._on_link)
        grid.addWidget(self.link_check, 7, 0, 1, 3)
        self.napari_panels_check = QCheckBox("napari's layer panels (left)")
        self.napari_panels_check.setChecked(False)
        self.napari_panels_check.setToolTip(
            "napari's own layer list and layer controls, left of the views. "
            "Off at first (0.18.0): the panel has the controls needed "
            "(Channels, Levels, the drawing tools under Crop regions), and "
            "the views get more room.")
        self.napari_panels_check.toggled.connect(self._show_napari_panels)
        grid.addWidget(self.napari_panels_check, 8, 0, 1, 3)
        # (0.18.0) the channels' show tick boxes moved to the Channels part
        row = QHBoxLayout()                     # less on screen when crowded
        row.addWidget(QLabel("Show:"))
        self.crosshairs_check = QCheckBox("crosshairs")
        self.crosshairs_check.setChecked(True)
        self.crosshairs_check.setToolTip(
            "The coloured lines in the 2D views that show where the other "
            "views' planes are (in 3D: 'slice planes').")
        self.crosshairs_check.toggled.connect(self.ortho.set_crosshairs)
        row.addWidget(self.crosshairs_check)
        self.boxes_check = QCheckBox("crop regions")
        self.boxes_check.setChecked(True)
        self.boxes_check.setToolTip(
            "All crop regions in all views (hidden ones are still cropped). "
            "To hide single regions, untick them in the list.")
        self.boxes_check.toggled.connect(self.ortho.set_boxes_visible)
        row.addWidget(self.boxes_check)
        # (0.20.0, the user's choice) the × marks can be switched off; when
        # crowded they are thinned out anyway (ortho_views.MARK_SPACING)
        self.marks_check = QCheckBox("× marks")
        self.marks_check.setChecked(True)
        self.marks_check.setToolTip(
            "The small × marks on the selected region: where its polygons "
            "in the other views cross this slice (an outline drawn here "
            "should meet them). Where they crowd, only some are shown; "
            "zoom in to see the others.")
        self.marks_check.toggled.connect(self.ortho.set_marks)
        row.addWidget(self.marks_check)
        row.addStretch()
        grid.addLayout(row, 9, 0, 1, 3)
        grid.addLayout(self._build_shading_controls(), 10, 0, 1, 3)
        grid.addLayout(self._build_preview_controls(), 11, 0, 1, 3)
        grid.addWidget(_label(
            "J over a 2D view: move the other views there."), 12, 0, 1, 3)
        layout.addWidget(group)

        # (0.18.0) a channel's histogram with numbers, its contrast lines
        # draggable (napari's own histogram shows no numbers)
        # (0.18.0, the user's wish) the channels: show, colour, choose one
        # for Levels (they replace napari's layer list, hidden at first)
        group = self._section("channels", "Channels")
        self.channels = ChannelsPanel(self.ortho)
        QVBoxLayout(group.body).addWidget(self.channels)
        layout.addWidget(group)

        group = self._section("levels", "Levels (histogram)")
        self.levels = LevelsPanel(self.ortho)
        QVBoxLayout(group.body).addWidget(self.levels)
        layout.addWidget(group)
        self.channels.on_channel_chosen = \
            self.levels.channel_combo.setCurrentIndex
        self.levels.on_channel_changed = self.channels.select

        group = self._section("regions", "Crop regions")
        inner = QVBoxLayout(group.body)
        row = QHBoxLayout()
        new = _button("New region", self._new_region)
        new.setToolTip("Start a new crop region and switch on the polygon "
                       "tool.")
        row.addWidget(new)
        finish = _button("Finish region", self._finish_region)
        finish.setToolTip("Stop drawing, and deselect the region (the next "
                          "drawing starts a new region).")
        row.addWidget(finish)
        row.addWidget(_button("Rename…", self._rename_box))
        delete = _button("Delete", self._remove_box)
        delete.setToolTip("Delete the selected regions (also the Delete key "
                          "in the list). Asks first if there are several.")
        row.addWidget(delete)
        inner.addLayout(row)
        # (0.18.0, the user's wish) the drawing tools for the crop regions,
        # as napari's layer controls have them (napari's panels are hidden
        # at first now); the tool in use stays pressed, also when chosen by
        # its key in a view
        row = QHBoxLayout()
        row.setSpacing(3)
        self.tool_buttons: dict = {}
        for mode, text, tip in DRAWING_TOOLS:
            # (push buttons: napari's style draws tool buttons flat, like
            # text, and shows a pushed-in push button highlighted)
            button = QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _=False, mode=mode: self._choose_tool(mode))
            row.addWidget(button)
            self.tool_buttons[mode] = button
        remove = QPushButton("Delete shape")
        remove.setToolTip("Delete the shapes selected in a 2D view (with "
                          "Select): a region's rectangle deletes the region, "
                          "a polygon only that polygon (Delete key in the "
                          "view).")
        remove.clicked.connect(
            lambda _=False: self.ortho.delete_selected_shapes())
        row.addWidget(remove)
        row.addStretch()
        inner.addLayout(row)
        self._show_tool("pan_zoom")
        # (0.17.0) a region that moves with an object over time (made by
        # Autocropper) can be made a fixed one
        self.fix_button = _button("Fix in place", self._fix_in_place)
        self.fix_button.setToolTip(
            "The selected region moves with the object over time: make it "
            "one fixed region that holds all its places (larger if the "
            "object drifts or grows), e.g. to draw polygons in it.")
        self.fix_button.setVisible(False)
        inner.addWidget(self.fix_button)
        self.box_list = QListWidget()
        self.box_list.setToolTip(
            "Drawing goes into the selected region (in bold). Click an "
            "empty part of the list to select none. Untick a region to hide "
            "it (it is still cropped).\nCtrl+click or Shift+click selects "
            "several, Ctrl+A all, e.g. to delete them at once; the one "
            "clicked last is the selected region.")
        # (0.18.0, the user's wish) several regions can be selected, to
        # delete them at once. The list's signals come in the middle of
        # Qt's own update of the selection, so the panel follows a moment
        # later (_on_list_choice), once Qt has finished.
        self.box_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list_timer = QTimer(self)
        self._list_timer.setSingleShot(True)
        self._list_timer.setInterval(0)
        self._list_timer.timeout.connect(self._on_list_choice)
        self.box_list.currentRowChanged.connect(
            lambda _row: self._list_timer.start())
        self.box_list.itemSelectionChanged.connect(self._list_timer.start)
        self.box_list.itemChanged.connect(self._on_box_ticked)
        self.box_list.viewport().installEventFilter(self)  # click: deselect
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete),
                  self.box_list, self._remove_box,
                  context=Qt.ShortcutContext.WidgetShortcut)
        inner.addWidget(self.box_list)
        # (0.18.0, the user's wish) undo / redo of all changes to the
        # regions, also with the keys anywhere in the window (a number
        # field being typed in keeps its own Ctrl+Z)
        row = QHBoxLayout()
        self.undo_button = _button("Undo", self._undo_regions)
        self.undo_button.setToolTip(
            "Undo the last change to the crop regions (Ctrl+Z): drawing, "
            "dragging in 3D, deleting, renaming, typed ranges...")
        self.redo_button = _button("Redo", self._redo_regions)
        self.redo_button.setToolTip("Redo what Undo undid (Ctrl+Shift+Z or "
                                    "Ctrl+Y).")
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        row.addStretch()
        inner.addLayout(row)
        for keys, slot in ((QKeySequence.StandardKey.Undo,
                            self._undo_regions),
                           (QKeySequence.StandardKey.Redo,
                            self._redo_regions),
                           ("Ctrl+Shift+Z", self._redo_regions)):
            QShortcut(QKeySequence(keys), self, slot,
                      context=Qt.ShortcutContext.WindowShortcut)
        self._update_undo_buttons()
        help_label = _label(
            "Polygon (P) or lasso (Shift+P) on a few slices of any view; on "
            "a projection it is a silhouette. Rectangle (R): a plain box. "
            "In 3D, drag the selected region's dots.")
        help_label.setToolTip(
            "All polygons of a region make one 3D shape through them, which "
            "ends at the outermost ones. Solid line: the region as cropped; "
            "dashed: its outline on this slice; ×: where polygons of the "
            "other views cross this slice.\nIn the 3D view (0.18.0), the "
            "selected region has a dot in the middle of each face and one "
            "in its centre: drag a face's dot to move that face along its "
            "axis, the centre's to move the region (and its polygons) "
            "across the screen. Shift: a face's drag resizes the region "
            "around its centre. Ctrl: round values (steps of about 1 % of "
            "the image). Up/Down: the face last dragged (or clicked) moves "
            "one voxel out/in (Shift: 10). Click a region's frame to select "
            "it. Elsewhere the mouse turns the view. Its polygons still "
            "shape the region: a face can't go past them in their own "
            "plane.")
        inner.addWidget(help_label)
        # (0.16.0) The polygons part only while the selected region has
        # polygons (see _fill_polygon_list).
        self.polygon_part = QWidget()
        part = QVBoxLayout(self.polygon_part)
        part.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Polygons"))
        row.addStretch()
        row.addWidget(QLabel("Margin"))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(-1000, 1000)
        self.margin_spin.setSuffix(" voxels")
        self.margin_spin.setToolTip(
            "Grow (positive) or shrink (negative) the region's shape in all "
            "directions, in voxels of the finest axis at full resolution.")
        self.margin_spin.valueChanged.connect(self._on_margin_changed)
        row.addWidget(self.margin_spin)
        part.addLayout(row)
        self.polygon_list = QListWidget()
        self.polygon_list.setMaximumHeight(80)
        self.polygon_list.setToolTip(
            "The selected region's polygons. Double-click one to go to its "
            "slice.\nCtrl+click or Shift+click selects several, Ctrl+A all, "
            "e.g. to delete them at once (also with the Delete key).")
        self.polygon_list.setSelectionMode(              # (0.18.0)
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.polygon_list.itemDoubleClicked.connect(
            lambda _item: self._go_to_polygon())
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete),
                  self.polygon_list, self._delete_polygon,
                  context=Qt.ShortcutContext.WidgetShortcut)
        part.addWidget(self.polygon_list)
        row = QHBoxLayout()
        row.addWidget(_button("Go to slice", self._go_to_polygon))
        delete = _button("Delete selected", self._delete_polygon)
        delete.setToolTip("Delete the selected polygons. Asks first if there "
                          "are several.")
        row.addWidget(delete)
        row.addWidget(_button("Clear polygons", self._clear_outlines))
        part.addLayout(row)
        self.polygon_part.setVisible(False)
        inner.addWidget(self.polygon_part)
        # (0.16.0) The start/stop fields in a part of their own, closed at
        # first (the plan text below shows the ranges anyway).
        ranges = self._section("ranges", "Ranges (start, stop)")
        self.range_grid = QGridLayout(ranges.body)
        inner.addWidget(ranges)
        self.plan_label = _label("")
        self.plan_label.setStyleSheet("font-family: Consolas, monospace;")
        inner.addWidget(self.plan_label)
        row = QHBoxLayout()
        row.addWidget(_button("Load regions…", self._load_boxes))
        row.addStretch()
        inner.addLayout(row)
        layout.addWidget(group)

        # (0.14.0, experimental) fit a region to one bright object
        group = self._section("find", "Autocropper (experimental)")
        QVBoxLayout(group.body).addWidget(self._build_object_finder())
        # closing it ends Add seeds: its button would be hidden while
        # clicks in the views still add seeds (review 2026-11-08)
        group.on_show = lambda opened: opened or (
            self.object_group.seed_button.setChecked(False),
            self.object_group.gap_button.setChecked(False))
        layout.addWidget(group)

        group = self._section("output", "Output")
        inner = QVBoxLayout(group.body)
        row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("folder for the cropped .zarr files")
        row.addWidget(self.out_edit)
        row.addWidget(_button("Browse…", self._choose_output))
        inner.addLayout(row)
        # (0.16.0) which channels the selected region's crop keeps, and
        # saving the regions, here with the output (the user's wish)
        self.channel_row = QHBoxLayout()
        inner.addLayout(self.channel_row)
        row = QHBoxLayout()
        save = _button("Save regions…", self._save_boxes)
        save.setToolTip("Save the regions as a CSV file (a 'boxes file'), "
                        "e.g. for the notebook or the command line.")
        row.addWidget(save)
        row.addStretch()
        inner.addLayout(row)
        # (0.15.0, the user's choice) No settings for the smallest level,
        # parallel workers or keeping the position any more: crops keep
        # pyramid levels down to MIN_LEVEL_SIZE voxels, copy with WORKERS
        # threads and keep their position (they overlay the full image);
        # the notebook and the command line still offer all three.
        self.overwrite_check = QCheckBox("Overwrite earlier crops")
        self.overwrite_check.setToolTip(
            "Replace earlier crops of the same regions (only crops made by "
            "this tool). Their TIFFs are deleted first, so that no TIFF "
            "shows an old region next to the new crop.")
        inner.addWidget(self.overwrite_check)
        inner.addLayout(self._build_tiff_controls())
        row = QHBoxLayout()
        self.crop_selected_button = _button("Crop selected",
                                            lambda: self._crop(False))
        self.crop_all_button = _button("Crop all", lambda: self._crop(True))
        self.tiff_button = _button("Save as TIFF…",
                                   lambda: self._crop(False, tiff_only=True))
        self.tiff_button.setToolTip(
            "Save the selected region as a TIFF for Fiji (ilastik reads it "
            "too), at the TIFF resolution above, without making an OME-Zarr "
            "crop: <image>_<region>_level<k>.tif in the output folder, an "
            "ImageJ hyperstack with the voxel size, frame interval, channel "
            "names and colours, and the region's record (Fiji: Image > Show "
            "Info). It is checked against the image.\nOver 4 GB a TIFF uses "
            "ImageJ's own large-file layout: Fiji opens it with File > Open, "
            "other programs may show only its first plane.")
        self.cancel_button = _button("Cancel", self._cancel_crop)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip(
            "Stop cropping after the parts being written. The crop being "
            "written is left incomplete; crop it again with 'Overwrite "
            "earlier crops' to replace it.")
        for button in (self.crop_selected_button, self.crop_all_button,
                       self.cancel_button):
            row.addWidget(button)
        inner.addLayout(row)
        row = QHBoxLayout()               # (its own row: no squeezing)
        row.addWidget(self.tiff_button)
        row.addStretch()
        inner.addLayout(row)
        self.progress_bar = QProgressBar()
        inner.addWidget(self.progress_bar)
        self.status = _label("")
        inner.addWidget(self.status)
        self.open_last_button = _button("Open last crop in a new window",
                                        self._open_last_crop)
        self.open_last_button.setEnabled(False)
        inner.addWidget(self.open_last_button)
        layout.addWidget(group)
        layout.addStretch()
        self._show_selected()
        self._fit_box_list()
        self._apply_settings(self._settings, new_window=True)

    def _build_object_finder(self):
        self.object_group = ObjectFinder(self)
        return self.object_group

    # --- settings (0.16.0) ---------------------------------------------------

    def _open_settings(self):
        dialog = SettingsDialog(self)
        if not dialog.exec():
            return
        values = dialog.values()
        problem = settings.save(values)
        if problem:
            QMessageBox.warning(self, "Settings", problem)
        # napari's memory cache: compared with napari's own setting (another
        # window or napari's Preferences may have changed it since this
        # window's spin box was set), then shown in every window
        cache = dialog.cache_spin.value()
        dask = _dask_settings()
        if abs(cache - (dask.cache if dask.enabled else 0.0)) > 1e-9:
            self._on_cache_size(cache)
        for _viewer, widget in list(_WINDOWS):
            try:
                widget.cache_spin.blockSignals(True)
                widget.cache_spin.setValue(cache)
                widget.cache_spin.blockSignals(False)
                widget._apply_settings(values)
            except RuntimeError:             # a window being closed
                pass

    def _apply_settings(self, values: dict, new_window: bool = False):
        """Settings to this window: the parts open and the tool's layers in
        napari's list; and, for a new window or one without an image, the
        ticks of napari's panels, Compress TIFFs, Link zoom/pan and exact
        level (the dialog's "New windows start with": a window with an
        image keeps what was chosen in it; the review of 2026-11-08 found
        OK resetting its zoom and its Compress TIFFs)."""
        self._settings = values
        for key, section in self.sections.items():
            section.set_open(values["sections"].get(key, True))
        self._hide_tool_layers()
        if not (new_window or self.image is None):
            return
        self.napari_panels_check.setChecked(values["napari_layer_panels"])
        self._show_napari_panels(values["napari_layer_panels"])
        self.link_check.setChecked(values["link_views"])
        self.compress_check.setChecked(values["compress_tiffs"])
        self.exact_check.blockSignals(True)
        self.exact_check.setChecked(values["exact_level"])
        self.exact_check.blockSignals(False)
        self._apply_start_options(values)

    def _apply_start_options(self, values: dict):
        """(0.18.0, the user's wish) The Autocropper's and the View part's
        options from the settings (a new window, or one without an
        image)."""
        auto, view = values["autocropper"], values["view"]

        def choose(combo, data):
            k = combo.findData(data)
            if k >= 0 and k != combo.currentIndex():
                combo.setCurrentIndex(k)

        finder = self.object_group
        choose(finder.method_combo, auto["method"])
        if not finder.own_check.isChecked():
            finder.grow_check.setChecked(auto["grow"])
        finder.margin_spin.setValue(float(auto["margin"]))
        finder.moving_check.setChecked(auto["moving"])
        finder.outline_check.setChecked(auto["outline"])
        choose(finder.outline_kind_combo, auto["outline_kind"])
        choose(finder.seed_mode_combo, auto["seed_mode"])
        finder.split_check.setChecked(auto["split"])
        self.planes_check.setChecked(view["slice_planes"])
        choose(self.mask_combo, view["mask_mode"])
        self.ortho.set_mask_mode(self.mask_combo.currentData())
        choose(self.outlines_3d_combo, view["outlines_3d"])
        self.outlines_3d_opacity.setValue(float(view["outlines_3d_opacity"]))
        self._on_outlines_3d()
        self.crosshairs_check.setChecked(view["crosshairs"])
        self.boxes_check.setChecked(view["crop_regions"])
        self.marks_check.setChecked(view["crossing_marks"])
        choose(self.preview_combo, view["preview"])
        self.ortho.set_preview(self.preview_combo.currentData() == "main")
        choose(self.levels.source_combo, view["levels_source"])
        self.levels.log_check.setChecked(view["levels_log"])
        self.levels.zoom_check.setChecked(view["levels_zoom"])   # (0.22.0)

    # --- napari's layer list: the tool's own layers hidden (0.16.0) ---------

    def _watch_layer_list(self):
        """Keep the tool's helper layers out of napari's layer list: rows
        are hidden again whenever the list changes."""
        events = self.viewer.layers.events
        for name in ("inserted", "removed", "moved", "reordered"):
            getattr(events, name).connect(
                lambda *_: QTimer.singleShot(0, self._hide_tool_layers))

    def _hide_tool_layers(self):
        """Hide (or, with the setting, show) the rows of the layers the
        views make for themselves (region lines, crosshair, preview,
        shading) in napari's layer list; the image channels and the layers
        one draws on stay. Only the list's rows: the layers work as
        before."""
        try:
            view = self.viewer.window._qt_viewer.layers
            model = view.model()
        except (AttributeError, RuntimeError):
            return
        show_all = bool(getattr(self, "_settings", {}).get(
            "show_tool_layers", False))
        layers = list(self.viewer.layers)
        for row in range(model.rowCount()):
            source = model.mapToSource(model.index(row, 0)).row()
            if not 0 <= source < len(layers):
                continue
            helper = ov.is_helper_layer(layers[source])
            view.setRowHidden(row, helper and not show_all)

    def _build_range_controls(self):
        _clear_layout(self.range_grid)
        self.range_spins = {}
        self.t_step_spin = None             # "every n-th" (time-lapse only)
        self.range_grid.addWidget(QLabel("start"), 0, 1)
        self.range_grid.addWidget(QLabel("stop"), 0, 3)
        # A 2D image (0.12.0) has one z plane: no z row (always all of it).
        flat_z = (self.ortho.view_for(("y", "x")).order[-3]
                  if self.ortho.flat else None)
        for row, axis in enumerate(self.image.range_axes, start=1):
            if self._is_time(axis):
                self._build_time_row(row, axis)
                continue
            if axis == flat_z:
                continue
            low, high = self.image.extent(axis)
            unit = display_unit(self.image.unit(axis))
            self.range_grid.addWidget(
                QLabel(f"{axis} ({unit})" if unit else axis), row, 0)
            voxel = self.image.levels[0].scale[
                self.image.axis_names.index(axis)]
            spins = []
            for which, column in ((0, 1), (1, 3)):
                spin = QDoubleSpinBox()
                spin.setRange(low, high)
                spin.setDecimals(2)
                spin.setSingleStep(voxel)      # arrow keys: one voxel
                spin.setKeyboardTracking(False)  # update when typing ends
                spin.valueChanged.connect(self._on_spin_changed)
                self.range_grid.addWidget(spin, row, column)
                spins.append(spin)
                # a small "here" button after each field: use the crosshair
                here = _button("here", lambda _=False, a=axis, w=which:
                               self._set_from_cursor(a, w))
                here.setToolTip(literal(
                    f"Set the {('start', 'stop')[which]} of {axis} "
                    "to the crosshair position"))
                here.setMaximumWidth(48)
                self.range_grid.addWidget(here, row, column + 1)
            self.range_spins[axis] = spins

    def _build_time_row(self, row: int, axis: str):
        """The t row (0.11.0): timepoint numbers 1, 2, 3 ... (like Fiji's
        frames), each with its time beside it, and "every n-th". The
        region itself keeps times (frame centres), as the boxes file."""
        _, _, n = self._t_frames(axis)
        self.range_grid.addWidget(QLabel(f"{axis} (timepoint)"), row, 0)
        spins = []
        for which, column in ((0, 1), (1, 3)):
            spin = QSpinBox()
            spin.setRange(1, n)
            spin.setKeyboardTracking(False)      # update when typing ends
            spin.setToolTip(
                f"The {('first', 'last')[which]} timepoint of the region "
                "(1 = the image's first, as Fiji's frames), with its time.")
            spin.valueChanged.connect(self._on_spin_changed)
            self.range_grid.addWidget(spin, row, column)
            spins.append(spin)
            here = _button("here", lambda _=False, a=axis, w=which:
                           self._set_from_cursor(a, w))
            here.setToolTip(f"Set the {('first', 'last')[which]} timepoint "
                            "to the one shown")
            here.setMaximumWidth(48)
            self.range_grid.addWidget(here, row, column + 1)
        self.range_spins[axis] = spins
        self.t_step_spin = QSpinBox()
        self.t_step_spin.setRange(1, max(n, 1))
        self.t_step_spin.setPrefix("every ")
        self.t_step_spin.setKeyboardTracking(False)
        self.t_step_spin.setToolTip(
            "Keep every n-th timepoint of the region, starting with its "
            "first (1 = all). The crop's frame interval is then n times the "
            "image's. Region lines are faded at timepoints not kept.")
        self.t_step_spin.valueChanged.connect(self._on_t_step_changed)
        self.range_grid.addWidget(self.t_step_spin, row, 5)
        self._show_t_times()

    def _t_frames(self, axis: str) -> tuple[float, float, int]:
        """The time axis at level 0: (time of the first timepoint, interval
        between timepoints, number of timepoints)."""
        i = self.image.axis_names.index(axis)
        level0 = self.image.levels[0]
        return (level0.translation[i], level0.scale[i],
                level0.array.shape[i])

    def _t_time(self, axis: str, number: int) -> float:
        """The time of timepoint `number` (1 = the first)."""
        first, interval, _ = self._t_frames(axis)
        return first + (number - 1) * interval

    def _t_numbers(self, axis: str, low: float, high: float) -> tuple:
        """The first and last timepoint (1 = the first) of a time range: all
        timepoints it touches, as the crop counts them."""
        first, interval, n = self._t_frames(axis)
        start, end = vc.physical_to_index_range(min(low, high),
                                                max(low, high), interval,
                                                first)
        start, end = min(max(start, 0), n - 1), min(max(end, 1), n)
        return start + 1, max(end, start + 1)

    def _show_t_times(self):
        """Beside each timepoint number: its time, e.g. "3 (60 min)"."""
        for axis, spins in self.range_spins.items():
            if self._is_time(axis):
                unit = display_unit(self.image.unit(axis))
                for spin in spins:
                    spin.setSuffix(f"  ({self._t_time(axis, spin.value()):g}"
                                   f" {unit})".replace(" )", ")"))

    def _on_t_step_changed(self, value: int):
        i = self._selected()
        if i < 0:
            return
        self.boxes[i].t_step = int(value)
        self._refresh(select=i)

    def _build_projection_options(self) -> QWidget:
        """The settings of "slab around the slice" (a thickness) and "chosen
        ranges" (from–to per view, filled when an image opens). Each is shown
        only while it is chosen."""
        options = QWidget()
        layout = QVBoxLayout(options)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slab_row = QWidget()
        row = QHBoxLayout(self.slab_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("slab thickness"))
        self.slab_spin = QDoubleSpinBox()
        self.slab_spin.setDecimals(1)
        self.slab_spin.setRange(0.1, 1e6)
        self.slab_spin.setValue(50.0)
        self.slab_spin.setKeyboardTracking(False)   # apply when typing ends
        self.slab_spin.setToolTip(
            "Each view projects through this thickness, centred on its "
            "slice (XY along z, XZ along y, YZ along x). Move the slider to "
            "move the slab; in 3D, 'slice planes' outlines it.")
        self.slab_spin.valueChanged.connect(lambda _: self._on_projection())
        row.addWidget(self.slab_spin)
        row.addStretch()
        layout.addWidget(self.slab_row)
        self.range_box = QWidget()
        self.projection_range_grid = QGridLayout(self.range_box)
        self.projection_range_grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.range_box)
        self.projection_spins: dict = {}
        self._show_projection_options()
        return options

    def _build_projection_ranges(self):
        """One from–to range per 2D view, along its depth axis (starting as
        the whole loaded region), with "here" buttons."""
        grid = self.projection_range_grid
        _clear_layout(grid)
        self.projection_spins = {}
        for row, name in enumerate(ov.VIEWS_2D):
            axis = self.ortho.views[name].order[-3]
            low, high = self.ortho.extent(axis)
            unit = display_unit(self.image.unit(axis))
            grid.addWidget(QLabel(f"{name} along {axis}"), row, 0)
            spins = []
            for which, column in ((0, 1), (1, 3)):
                spin = QDoubleSpinBox()
                spin.setDecimals(1)
                spin.setRange(low, high)
                spin.setValue((low, high)[which])
                spin.setSuffix(f" {unit}" if unit else "")
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda _: self._on_projection())
                grid.addWidget(spin, row, column)
                here = _button("here", lambda _=False, a=axis, w=which:
                               self._projection_from_cursor(a, w))
                here.setToolTip(f"Set the {('start', 'end')[which]} to the "
                                "crosshair position")
                here.setMaximumWidth(48)
                grid.addWidget(here, row, column + 1)
                spins.append(spin)
            self.projection_spins[axis] = spins
        unit = display_unit(self.image.unit(self.ortho.views["XY"].order[-3]))
        self.slab_spin.setSuffix(f" {unit}" if unit else "")

    def _projection_from_cursor(self, axis: str, which: int):
        if axis in self.ortho.cursor:
            self.projection_spins[axis][which].setValue(self.ortho.cursor[axis])

    def _projection_ranges(self) -> dict:
        """The chosen ranges, {depth axis: (low, high)}."""
        return {axis: tuple(sorted((a.value(), b.value())))
                for axis, (a, b) in self.projection_spins.items()}

    def _show_projection_options(self):
        mode = self.projection_combo.currentData()
        through = self.through_combo.currentData()
        self.slab_row.setVisible(mode != "none" and through == "slab")
        self.range_box.setVisible(mode != "none" and through == "range")

    def _build_shading_controls(self):
        """Shade the selected region in the 2D views: kept voxels in one
        colour, the voxels set to 0 in another (colours and opacity
        adjustable)."""
        row = QHBoxLayout()
        self.shade_check = QCheckBox("Shade selected region:")
        self.shade_check.setToolTip(
            "On the slice shown: the voxels that will be kept, and those "
            "that will be set to 0. Click a colour to change it.")
        self.shade_check.toggled.connect(
            lambda on: self.ortho.set_shading(on=on))
        row.addWidget(self.shade_check)
        self.shade_buttons = {}
        for which, text in (("kept", "kept"), ("zeroed", "set to 0")):
            button = QPushButton(text)
            button.setToolTip(f"Choose the colour of the voxels {text}.")
            button.clicked.connect(
                lambda _checked=False, w=which: self._choose_shade_colour(w))
            row.addWidget(button)
            self.shade_buttons[which] = button
        self._show_shade_colours()
        row.addWidget(QLabel("opacity"))
        self.shade_opacity = QDoubleSpinBox()
        self.shade_opacity.setRange(0.05, 1.0)
        self.shade_opacity.setSingleStep(0.05)
        self.shade_opacity.setValue(self.ortho.shading["opacity"])
        self.shade_opacity.valueChanged.connect(
            lambda value: self.ortho.set_shading(opacity=value))
        row.addWidget(self.shade_opacity)
        row.addStretch()
        return row

    def _build_preview_controls(self):
        """Preview the selected region's crop: in the main views (black
        wherever the crop has nothing), and a button that zooms the 2D views
        to the region."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview crop:"))
        self.preview_combo = QComboBox()
        self.preview_combo.addItem("off", "off")
        self.preview_combo.addItem("in the main views", "main")
        self.preview_combo.setToolTip(
            "Show the selected region as it will be cropped: in the 2D "
            "views, everything the crop does not contain on the slice shown "
            "is black (outside the region, and the voxels set to 0).\nOn a "
            "projection (2D projection: max or mean), the views show the "
            "projection of exactly what the crop keeps: only the region's "
            "depth, without the voxels its polygons set to 0. A large "
            "region is read at a coarser level for it (the line below "
            "says which).")
        self.preview_combo.activated.connect(
            lambda _: self.ortho.set_preview(
                self.preview_combo.currentData() == "main"))
        row.addWidget(self.preview_combo)
        zoom = _button("Zoom to region",
                       lambda _=False: self.ortho.zoom_to_selected())
        zoom.setToolTip("Fit the 2D views to the selected region (slices "
                        "outside it move to its middle), and the 3D view "
                        "if it is on (its angle stays).")
        row.addWidget(zoom)
        row.addStretch()
        # (0.18.0) what the preview's projection shows (or that it is
        # being computed); empty without a projection
        column = QVBoxLayout()
        column.addLayout(row)
        self.preview_info = _label("")
        self.preview_info.setVisible(False)
        column.addWidget(self.preview_info)
        return column

    def _show_preview_info(self, text: str):
        """(0.18.0) The line under Preview crop (OrthoViews tells it)."""
        self.preview_info.setText(text)
        self.preview_info.setVisible(bool(text))

    def _build_tiff_controls(self):
        """The TIFF resolution for "Save as TIFF…" (0.15.0; it was the
        level of "Also save as TIFF", 0.10.0) on a row of its own, so that
        the voxel sizes can be read, and "Compress TIFFs"."""
        layout = QVBoxLayout()
        row = QHBoxLayout()
        self.tiff_level_label = QLabel("TIFF resolution")
        row.addWidget(self.tiff_level_label)
        self.tiff_level_combo = QComboBox()
        self.tiff_level_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.tiff_level_combo.setToolTip(
            "The resolution 'Save as TIFF…' saves, with its voxel size (z × "
            "y × x): full resolution, or a level of the image's pyramid, each "
            "about 2 or 4 times coarser. A small region with fewer levels "
            "gives its coarsest.")
        self.tiff_level_combo.currentIndexChanged.connect(
            self._on_tiff_changed)
        row.addWidget(self.tiff_level_combo, 1)
        self.tiff_row = row
        layout.addLayout(row)
        self.compress_check = QCheckBox("Compress TIFFs (zstd)")
        self.compress_check.setToolTip(
            "Lossless zstd compression with the predictor (each pixel stored "
            "as its difference to its neighbour): TIFFs about half the size "
            "or smaller, but slower to open in Fiji.\nFiji opens compressed "
            "TIFFs through Bio-Formats: a Fiji TIFF (.tif) then shows the "
            "Bio-Formats dialog; an OME-TIFF (.ome.tif) always opens "
            "through Bio-Formats anyway.\nFor a TIFF file, it starts ticked "
            "if the file itself is compressed.")
        self.compress_check.toggled.connect(self._on_tiff_changed)
        layout.addWidget(self.compress_check)
        return layout

    def _fill_tiff_levels(self):
        """The TIFF levels of the image: one per pyramid level, with its
        voxel size (z × y × x); the choice is kept when it exists."""
        combo = self.tiff_level_combo
        chosen = max(combo.currentIndex(), 0)
        combo.blockSignals(True)
        combo.clear()
        spatial = [self.image.axis_names.index(a)
                   for a in vc._spatial_axes(self.image)]
        unit = display_unit(self.image.unit(self.image.axis_names[spatial[-1]]))
        for k, level in enumerate(self.image.levels):
            sizes = " × ".join(f"{level.scale[i]:.3g}" for i in spatial)
            name = "full resolution" if k == 0 else f"level {k}"
            combo.addItem(f"{name}: {sizes} {unit}", k)
        combo.setCurrentIndex(min(chosen, combo.count() - 1))
        combo.blockSignals(False)

    def _tiff_level(self) -> int | None:
        """The level "Save as TIFF…" saves, or None (no image; and for a
        file read into memory, whose crop is itself a TIFF)."""
        if self.image is None or self.image.in_memory:
            return None
        return self.tiff_level_combo.currentData()

    def _on_tiff_changed(self, _=None):
        self._show_selected()           # the plan text shows the TIFF

    def _build_cache_controls(self):
        """Memory cache: its size, how much is used, and a Clear button.

        napari keeps the image data it has read in memory (its "dask
        cache"), so that showing a channel again, going back to a place or
        switching 3D on again reads nothing. The size is napari's own
        setting (Preferences → Application → Dask cache), which napari saves
        between sessions.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel("Memory cache"))
        self.cache_spin = QDoubleSpinBox()
        self.cache_spin.setDecimals(1)
        self.cache_spin.setSingleStep(0.5)
        self.cache_spin.setSuffix(" GB")
        self.cache_spin.setRange(0.0, _max_cache_gb())
        settings = _dask_settings()
        self.cache_spin.setValue(settings.cache if settings.enabled else 0.0)
        self.cache_spin.setKeyboardTracking(False)   # apply when typing ends
        self.cache_spin.setToolTip(
            "Image data already read are kept in memory up to this size, so "
            "that showing a channel again, going back to a place or "
            "switching 3D on again reads nothing. 0 = off. Remembered "
            "between sessions (napari's Preferences → Application → Dask "
            "cache). It is napari's own cache, shared by all napari windows "
            "(also your own, when Volume Cropper runs as a napari plugin).")
        self.cache_spin.valueChanged.connect(self._on_cache_size)
        row.addWidget(self.cache_spin)
        self.cache_info = QLabel()
        row.addWidget(self.cache_info, 1)
        clear = _button("Clear cache", self._clear_cache)
        clear.setToolTip("Forget the data kept for re-use (the views keep "
                         "what they show now; anything else is read again "
                         "when needed).")
        row.addWidget(clear)
        self._cache_timer = QTimer(self)        # keep the "used" text fresh
        self._cache_timer.setInterval(2000)
        self._cache_timer.timeout.connect(self._show_cache)
        self._cache_timer.start()
        self._show_cache()
        return row

    def _on_cache_size(self, gigabytes: float):
        """Set napari's cache size (napari resizes the cache and saves it)."""
        settings = _dask_settings()
        settings.cache = gigabytes
        settings.enabled = gigabytes > 0
        self._show_cache()

    def _clear_cache(self):
        _dask_cache().clear()
        self._show_cache()
        self.status.setText("Memory cache cleared.")

    def _show_cache(self):
        self.cache_info.setText(
            f"{vc.format_bytes(_dask_cache().total_bytes)} used")

    def _choose_tool(self, mode: str):
        """A drawing tool button: that tool in all 2D views."""
        self.ortho.set_tool(mode)
        self._show_tool(mode)

    def _on_tool_changed(self, mode: str):
        """The drawing tool changed in a view (a button here, its key, or
        napari's controls): the buttons show it; (0.16.0) the Autocropper
        stops adding seeds when a drawing tool is chosen."""
        self._show_tool(mode)
        if hasattr(self, "object_group"):
            self.object_group.tool_changed(mode)

    def _show_tool(self, mode: str):
        for name, button in getattr(self, "tool_buttons", {}).items():
            button.setChecked(name == mode)

    @property
    def show_checks(self) -> list:
        """The channels' 'show' tick boxes: in the Channels list since
        0.18.0 (they were a row under View; the older checks use them)."""
        return self.channels.show_checks

    def _on_display_changed(self, k: int, attr: str, value):
        """Keep the Channels list (0.18.0; the 'show' checkboxes before) in
        step with napari's eye icons and colours (and the 3D sizes, which
        count the channels shown), and the Levels histogram with the
        contrast (0.18.0)."""
        self.levels.on_display(k, attr)
        self.channels.on_display(k, attr)
        if attr == "visible":
            self._fill_3d_combo()
            # On "Auto", the 3D view follows the channels shown: a finer or
            # coarser level when their volume changes size (asking first
            # before a large one). Later, so that napari has finished.
            if (self.show_3d_check.isChecked()
                    and self.level_3d_combo.currentData() is None
                    and self.ortho.auto_3d() != self.ortho.level_3d_shown()):
                QTimer.singleShot(0, self._on_3d)

    def _build_channel_controls(self):
        _clear_layout(self.channel_row)
        self.channel_checks = []
        if self.image.channel_axis is None:
            return
        label = QLabel("Channels to crop:")
        label.setToolTip("Which channels the selected region's crop keeps "
                         "(Show channels, under View, only shows them).")
        self.channel_row.addWidget(label)
        for i, label in enumerate(self.image.channel_labels):
            check = QCheckBox(str(i))
            check.setToolTip(literal(label))
            check.setChecked(True)
            check.toggled.connect(self._on_channels_changed)
            self.channel_row.addWidget(check)
            self.channel_checks.append(check)
        self.channel_row.addStretch()

    def _fill_level_combos(self):
        """The View controls, set as the views now are (as chosen in the
        "What to load" dialog). Only the levels the loaded region uses."""
        image, ortho = self.image, self.ortho
        self.level_combo.blockSignals(True)
        self.level_combo.clear()
        self.level_combo.addItem("Auto (from zoom)", None)
        for k in range(ortho.region.n_levels):
            self.level_combo.addItem(
                f"level {k} ({image.levels[k].scale[-1]:.3g} µm)", k)
        self.level_combo.setCurrentIndex(
            0 if ortho.level_2d is None else ortho.level_2d + 1)
        self.level_combo.blockSignals(False)
        self._fill_3d_combo()
        self.level_3d_combo.setCurrentIndex(
            0 if ortho.level_3d is None else ortho.level_3d + 1)
        # (0.16.0) "exact level" ticked or not as in the settings; it takes
        # effect when a level is chosen (with the question about large
        # reads, see _on_level_2d)
        exact = bool(self._settings.get("exact_level", False))
        self.exact_check.blockSignals(True)
        self.exact_check.setChecked(exact)
        self.exact_check.setEnabled(ortho.level_2d is not None)
        self.exact_check.blockSignals(False)
        if exact and ortho.level_2d is not None and not ortho.exact_2d:
            QTimer.singleShot(0, self._on_level_2d)
        self.show_3d_check.blockSignals(True)
        self.show_3d_check.setChecked(ortho.show_3d)
        self.show_3d_check.blockSignals(False)
        for widget in (self.level_3d_combo, self.clip_check, self.mask_combo,
                       self.outlines_3d_combo, self.outlines_3d_opacity):
            widget.setEnabled(ortho.show_3d)
        # the tick as shown (from the settings since 0.16.0): a question
        # answered No restores this state (review 2026-11-08)
        self._view_state.update(
            level_2d=self.level_combo.currentIndex(),
            exact=self.exact_check.isChecked(),
            show_3d=ortho.show_3d,
            level_3d=self.level_3d_combo.currentIndex())

    def _fill_3d_combo(self):
        """The 3D view's levels with their sizes, for the loaded region and
        the channels shown (the choice stays)."""
        if self.image is None or self.ortho.region is None:
            return
        image, region = self.image, self.ortho.region
        n = self.ortho.n_shown()
        combo = self.level_3d_combo
        current = combo.currentIndex()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(f"Auto (level {self.ortho.auto_3d()})", None)
        for k in range(region.n_levels):
            size = vc.format_bytes(ov.volume_bytes(image, k, region, n))
            combo.addItem(
                f"level {k} ({image.levels[k].scale[-1]:.3g} µm, {size})", k)
        combo.setCurrentIndex(current if 0 <= current < combo.count() else 0)
        combo.blockSignals(False)

    # --- opening an image ----------------------------------------------------

    def _start_choice(self, image, channels=None) -> LoadChoice:
        """(0.18.0) What opening an image starts with, from the settings:
        the whole image, the resolutions by voxel size (or Auto), the 3D
        view, and the channels shown (unless given)."""
        opening = self._settings["opening"]
        if (channels is None and opening["channels"] == "first"
                and image.channel_axis is not None):
            channels = [0]
        level_2d = (ov.level_for_voxel(image, opening["voxel_2d"])
                    if opening["level_2d"] == "size" else None)
        show_3d = opening["view_3d"] != "off"
        level_3d = (ov.level_for_voxel(image, opening["voxel_3d"])
                    if opening["view_3d"] == "size" else None)
        return LoadChoice(channels=channels, level_2d=level_2d,
                          show_3d=show_3d, level_3d=level_3d)

    def _too_large_to_skip(self, image, choice: LoadChoice) -> bool:
        """With "Ask 'What to load'" off: ask anyway if the 3D view would
        read more than LARGE_3D_VOLUME."""
        if not choice.show_3d:
            return False
        region = ov.loaded_region(image, choice.ranges)
        n = (len(choice.channels) if choice.channels is not None
             else image.n_channels)
        level = (ov.auto_3d_level(image, region=region, n_channels=n)
                 if choice.level_3d is None
                 else min(choice.level_3d, region.n_levels - 1))
        return ov.volume_bytes(image, level, region, n) > LARGE_3D_VOLUME

    def _choose_image(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open an OME-Zarr image (.zarr folder)")
        if path:
            self.open_image(path, ask=True)

    def _choose_zipped(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a zipped OME-Zarr image", self._start_folder(),
            "Zipped OME-Zarr (*.ozx *.zip);;All files (*)")
        if path:
            self.open_image(path, ask=True)

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an image file", self._start_folder(),
            "Images (*.tif *.tiff *.btf *.companion.ome *.lif *.lof *.czi "
            "*.nd2 *.oir);;TIFF (*.tif *.tiff *.btf);;OME-TIFF companion "
            "(*.companion.ome);;Leica (*.lif *.lof);;Zeiss (*.czi);;"
            "Nikon (*.nd2);;Olympus (*.oir);;All files (*)")
        if path:
            self.open_image(path, ask=True)

    def _choose_tiff_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open a folder of TIFF files (one image per file, in "
            "the order of their names)", self._start_folder())
        if path:
            self.open_image(path, ask=True)

    def _start_folder(self) -> str:
        return str(self.image.path.parent) if self.image is not None else ""

    def open_image(self, path, channels: list[int] | None = None,
                   choice: LoadChoice | None = None, ask: bool = False,
                   file_choice: fi.FileChoice | None = None):
        """Open an image.

        channels: the channels to show at first (None = all); the others
            are hidden and not read until shown.
        choice:   what to load (see LoadChoice; it overrides channels).
            Default: the whole image, with the 3D view off.
        ask:      show the "What to load" dialog first, starting from these
            choices. Cancel keeps the image that is open.
        file_choice: for an image file (TIFF ..., 0.12.0), what to read
            into memory (default: all of its first image). channels and
            choice do not apply to files.
        """
        if fi.is_file_image(path):
            opening = self._settings["opening"]
            if file_choice is None:                     # (0.18.0)
                file_choice = fi.FileChoice(
                    channels=[0] if opening["channels"] == "first" else None,
                    show_3d=opening["view_3d"] != "off")
            self._open_file(path, file_choice,
                            ask=ask and bool(opening["ask"]),
                            keep_boxes=False)
            return
        image = None
        try:
            try:
                image = vc.open_image(path)
            except vc.ImageChoiceNeeded as many:      # (0.21.0)
                inner = self._choose_inner_image(many)
                if inner is None:
                    return
                image = vc.open_image(path, inner=inner)
            ov.view_orders(image)         # checks that the image is 3D
        except Exception as err:   # show any problem instead of crashing
            if image is not None:
                image.close()
            QMessageBox.critical(self, "Cannot open image",
                                 f"{type(err).__name__}: {err}")
            return
        try:
            self._open_opened(image, channels, choice, ask)
        finally:
            if self.image is not image:      # not shown (Cancel ...): a
                image.close()                # zipped image's file closes

    def _choose_inner_image(self, many) -> str | None:
        """(0.21.0) A collection of several images (bioformats2raw layout,
        OME-Zarr 0.6 scene): ask which one to open, like Bio-Formats'
        series choice. Returns its path in the collection, or None."""
        labels = [label for _, label in many.images]
        label, ok = QInputDialog.getItem(
            self, "Choose an image",
            literal(f"{many.path.name} holds {len(labels)} images "
                    f"({many.kind}).\nWhich one do you want to open?"),
            labels, 0, False)
        if not ok:
            return None
        return many.images[labels.index(label)][0]

    def _open_opened(self, image, channels, choice, ask):
        """open_image's work once the image is open (see open_image)."""
        if channels is not None:
            n = image.n_channels
            wrong = [c for c in channels if not 0 <= c < n]
            channels = [c for c in channels if 0 <= c < n]
            if wrong or not channels:
                QMessageBox.warning(
                    self, "Channels",
                    f"The image has channels 0–{n - 1}; ignored: "
                    f"{', '.join(map(str, wrong)) or 'none given'}."
                    + ("" if channels else " Showing all channels."))
            channels = channels or None
        if choice is None:                               # (0.18.0)
            choice = self._start_choice(image, channels)
        if ask and not self._settings["opening"]["ask"]:
            ask = self._too_large_to_skip(image, choice)
        if self.boxes and not self._may_discard_boxes(
                "Opening another image"):
            return
        if ask:
            choice = self._ask_what_to_load(image, choice)
            if choice is None:
                return
        self._layer_reader = None        # an OME-Zarr now, not layers
        self._load(image, choice, keep_boxes=False)

    def _may_discard_boxes(self, action: str) -> bool:
        """Before the boxes are thrown away: offer to save them first.
        Returns True to go ahead, False to stop (Cancel, or not saved)."""
        answer = _choose(
            self, "Crop regions",
            f"{action} removes the {len(self.boxes)} crop region(s) drawn "
            "so far.", ["Save regions first", "Discard them"])
        if answer == "Discard them":
            return True
        return answer == "Save regions first" and self._save_boxes()

    def _ask_what_to_load(self, image, choice: LoadChoice,
                          box_ranges: dict | None = None):
        """The "What to load" dialog; returns the choice, or None (Cancel)."""
        dialog = LoadDialog(image, choice, self.ortho.canvas_sizes(),
                            box_ranges, parent=self)
        if not dialog.exec():
            return None
        return dialog.choice()

    # --- image files (TIFF ...), read into memory (0.12.0) ----------------------

    def _open_file(self, path, choice: fi.FileChoice, ask: bool,
                   keep_boxes: bool):
        """Open an image file: "What to load" (if ask), read the part chosen
        into memory, then show it like any other image."""
        try:
            reader = fi.open_file(path)
        except Exception as err:   # show any problem instead of crashing
            QMessageBox.critical(self, "Cannot open image",
                                 f"{type(err).__name__}: {err}")
            return
        self._open_reader(reader, choice, ask, keep_boxes)

    def open_layers(self, layers, keep: bool = False) -> bool:
        """Open napari image layers (0.13.0, the plugin): one layer, or
        several of the same size as the channels of one image, read into
        memory like a file ("What to load" first). Their data can come
        from any napari reader. If they all come from one file this tool
        can open itself (OME-Zarr, TIFF), it offers to open that file
        instead (an OME-Zarr is then read piece by piece, not copied).

        keep: hold the layers (see fi.LayerReader). Returns True if an
        image was opened.
        """
        layers = [getattr(layer, "__wrapped__", layer) for layer in layers]
        if not layers:
            QMessageBox.warning(
                self, "No image layer selected",
                "Select one image layer (or several of the same size: the "
                "channels of one image) in napari's layer list first.")
            return False
        source = fi.source_file(layers)
        if source is not None:
            how = ("It is then read piece by piece as you look around; "
                   "nothing is copied into memory, and crops are OME-Zarr."
                   if not fi.is_file_image(source) else
                   "It is then read into memory from the file, with the "
                   "file's own calibration; crops are TIFFs.")
            answer = _choose(
                self, "Open the file itself?",
                f"The layers come from {source.name}, which Volume Cropper "
                f"can open itself. {how}\n\nOr use the layers as they are "
                "in napari (copied into memory; crops are OME-TIFFs).",
                ["Open the file", "Use the layers"])
            if answer is None:
                return False
            if answer == "Open the file":
                before = self.image
                self.open_image(source, ask=True)
                return self.image is not None and self.image is not before
        try:
            reader = fi.LayerReader(layers, keep=keep)
        except Exception as err:
            QMessageBox.critical(self, "Cannot use these layers",
                                 f"{type(err).__name__}: {err}")
            return False
        return self._open_reader(reader, fi.FileChoice(), ask=True,
                                 keep_boxes=False)

    def _open_reader(self, reader, choice: fi.FileChoice, ask: bool,
                     keep_boxes: bool) -> bool:
        """_open_file's work, for any reader (a file, or napari layers).
        Returns True if the image was opened."""
        if _READING:
            QMessageBox.information(
                self, "Still reading",
                "Volume Cropper is still reading another image into memory. "
                "Please try again when it has finished.")
            return False
        try:
            if not keep_boxes and self.boxes and not self._may_discard_boxes(
                    "Opening another image"):
                return False
            if ask:
                dialog = FileLoadDialog(reader, choice, parent=self)
                if not dialog.exec():
                    return False
                choice = dialog.choice()
            else:
                try:   # scripts: refuse what cannot fit, as the dialog does
                    text, refuse = fi.memory_problem(
                        fi.memory_needed(reader, choice), fi.free_memory())
                except ValueError as err:
                    QMessageBox.warning(self, "Cannot open image", str(err))
                    return False
                if refuse:
                    QMessageBox.warning(self, "Not enough memory", text)
                    return False
            image = self._read_file(reader, choice)
        finally:
            reader.close()
        if image is None:
            return False
        try:
            ov.view_orders(image)
        except Exception as err:
            QMessageBox.critical(self, "Cannot open image",
                                 f"{type(err).__name__}: {err}")
            return False
        self._file_choice = choice
        # napari layers can be read again ("What to load…"); a file is
        # opened again from its path
        self._layer_reader = (reader if isinstance(reader, fi.LayerReader)
                              else None)
        self._load(image, LoadChoice(show_3d=choice.show_3d),
                   keep_boxes=keep_boxes)
        return True

    def _read_file(self, reader, choice: fi.FileChoice):
        """Read the chosen part into memory, with a progress window and
        Cancel. Returns the image, or None (cancelled or failed)."""
        dialog = QProgressDialog(literal(f"Reading {reader.path.name} …"),
                                 "Cancel",
                                 0, 1000, self)
        dialog.setWindowTitle("Reading into memory")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        started = time.time()

        def progress(done, total, stage):
            dialog.setValue(int(1000 * done / total) if total else 1000)
            dialog.setLabelText(literal(
                f"{reader.path.name}: {stage}, {vc.format_bytes(done)} of "
                f"{vc.format_bytes(total)} "
                f"({_duration(time.time() - started)})"))
            # Other windows stay usable meanwhile (napari's own, as a
            # plugin): _READING stops them from starting a second read.
            QApplication.processEvents()

        global _READING
        _READING += 1
        try:
            image = fi.load(reader, choice, progress=progress,
                            cancel=dialog.wasCanceled)
        except fi.ReadCancelled:
            return None
        except MemoryError:
            QMessageBox.critical(self, "Not enough memory",
                                 "The computer ran out of memory while "
                                 "reading. Load fewer channels, timepoints "
                                 "or slices.")
            return None
        except Exception as err:
            QMessageBox.critical(self, "Cannot read image",
                                 f"{type(err).__name__}: {err}")
            return None
        finally:
            _READING -= 1
            dialog.close()
        seconds = time.time() - started
        self._read_text = (f"Read {vc.format_bytes(image.levels[0].array.nbytes)}"
                           f" into memory in {seconds:.0f} s.")
        return image

    def _change_what_to_load(self):
        """Choose again what to load of the open image; the regions
        stay."""
        if self.image is None:
            return
        if self.image.in_memory and self._layer_reader is not None:
            # napari layers (0.13.0): read again, if they still exist
            if not self._layer_reader.alive():
                QMessageBox.information(
                    self, "What to load",
                    "The napari layers this image was read from are gone "
                    "(deleted or changed), so they cannot be read again. "
                    "The copy shown here stays as it is.")
                return
            self._open_reader(self._layer_reader,
                              self._file_choice or fi.FileChoice(), ask=True,
                              keep_boxes=True)
            return
        if self.image.in_memory:          # read the file again (0.12.0)
            self._open_file(self.image.source["file"],
                            self._file_choice or fi.FileChoice(), ask=True,
                            keep_boxes=True)
            return
        ortho = self.ortho
        choice = LoadChoice(
            channels=([k for k, d in enumerate(ortho.display) if d["visible"]]
                      if self.image.channel_axis is not None else None),
            ranges=dict(ortho.region.ranges), level_2d=ortho.level_2d,
            show_3d=ortho.show_3d, level_3d=ortho.level_3d)
        i = self._selected()
        box = ortho.box_ranges[i] if 0 <= i < len(ortho.box_ranges) else None
        choice = self._ask_what_to_load(self.image, choice, box)
        if choice is not None:
            self._load(self.image, choice, keep_boxes=True)

    def _load(self, image, choice: LoadChoice, keep_boxes: bool):
        """Show an image (or the open one again) as chosen."""
        region = ov.loaded_region(image, choice.ranges)
        selected = self._selected()
        old = self.image
        self.image = image
        self._fresh = None
        if not keep_boxes:
            self.boxes = []
        self._reset_undo()                                    # (0.18.0)
        last = region.n_levels - 1
        ortho = self.ortho
        ortho.level_2d = (None if choice.level_2d is None
                          else min(choice.level_2d, last))
        ortho.exact_2d = False
        ortho.level_3d = (None if choice.level_3d is None
                          else min(choice.level_3d, last))
        ortho.show_3d = choice.show_3d
        ortho.projection = "none"          # start without a projection
        for combo in (self.projection_combo, self.through_combo):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.through_combo.setEnabled(False)
        self._show_projection_options()
        self._view_state.update(projection=0, through=0)
        if image.in_memory:          # read already: nothing more from disk
            self._read_timer.stop()
            self._reading = None
            self.read_info.setText(self._read_text)
        else:
            self._watch_reading()
        # (0.18.0, the user's wish) a "Loading…" box while the views read
        self._start_loading(f"Opening {image.path.name}: loading the "
                            "views…")
        ortho.set_image(image, visible=choice.channels, region=region)
        if old is not None and old is not image:
            # (0.21.0) a zipped image's file: Windows keeps an open file
            # from being replaced (e.g. a crop cropped again)
            old.close()
        self.channels.set_image()                              # (0.18.0)
        self.levels.set_image()                                # (0.18.0)
        title = image.path.name
        if image.in_memory and image.source.get("scenes_in_file", 1) > 1:
            title += f" – {image.source['scene']}"
        if image.inner:          # (0.21.0) an image of a collection
            title += f" – image {image.inner}"
        self.viewer.title = f"Volume Cropper {vc.__version__} – {title}"
        if image.in_memory and self._layer_reader is not None:
            # napari layers (0.13.0): their path only names the crops
            self.path_edit.setText(
                "napari layers: " + ", ".join(self._layer_reader.layer_names))
        else:
            self.path_edit.setText(str(image.path))
        self.image_info.setText(self._image_summary())
        self._fill_level_combos()
        self._fill_tiff_levels()
        self._show_output_kind()
        self._levels_shown(ortho._levels_shown)   # "Auto (level N)"
        self._build_range_controls()
        self._build_projection_ranges()
        self._build_channel_controls()
        self._show_flat()
        self.object_group.image_changed()
        self.load_button.setEnabled(True)
        self._refresh(select=selected if keep_boxes else None)
        # After Qt has laid out the new controls, so the width is right.
        guard_wheel(self)        # (also the controls made for this image)
        QTimer.singleShot(0, self._fit_panel_width)

    def _show_flat(self):
        """A 2D image (one plane, 0.12.0): only XY is shown, so the 3D view
        and projections (along z) are off and cannot be chosen."""
        flat = self.ortho.flat
        widgets = (self.show_3d_check, self.level_3d_combo, self.clip_check,
                   self.planes_check, self.mask_combo, self.projection_combo,
                   self.through_combo)
        if flat:
            for widget in widgets:
                widget.setEnabled(False)
            self.show_3d_check.setToolTip("A 2D image has no 3D view.")
        else:
            self.show_3d_check.setEnabled(True)
            self.projection_combo.setEnabled(True)
            self.planes_check.setEnabled(True)
            self.show_3d_check.setToolTip(
                "Off at first: the 3D view reads a whole volume (of the "
                "loaded region), which takes time.")

    def _show_output_kind(self):
        """What a crop of this image is (0.12.0): an OME-Zarr (with the
        option to also save a TIFF), or for a file read into memory a TIFF
        of the file's kind; the controls that do not apply are hidden."""
        in_memory = self.image is not None and self.image.in_memory
        # a file's crops are TIFFs already: no "Save as TIFF…" for it
        for widget in (self.tiff_level_label, self.tiff_level_combo,
                       self.tiff_button):
            widget.setVisible(not in_memory)
        if in_memory:
            kind = vc.output_suffix(self.image)
            self.out_edit.setPlaceholderText(
                f"folder for the cropped {kind} files")
            # default: compressed if the file itself is
            self.compress_check.setChecked(
                self.image.source.get("compression_in_file", "none")
                not in ("none", None))
        else:
            # (0.21.0) .zarr, or .ozx / .zip for a zipped image
            self.out_edit.setPlaceholderText(
                f"folder for the cropped {vc.output_suffix(self.image)} "
                "files")
            # (0.16.0) as in the settings
            self.compress_check.setChecked(
                bool(self._settings.get("compress_tiffs", False)))
        self._on_tiff_changed()

    # --- how much the views read -----------------------------------------------

    def _watch_reading(self):
        """Count the data read from now on and show it under the image.

        napari reads while it draws, and the window waits meanwhile, so the
        line is updated in between and settles 3 s after the last read.
        """
        read = _bytes_read()
        if read is None:            # not available on this computer
            return
        now = time.time()
        self._reading = {"start": now, "first": read, "last": read,
                         "changed": now}
        self.read_info.setText("Reading…")
        self._read_timer.start()

    def _show_reading(self):
        state, read = self._reading, _bytes_read()
        if state is None or read is None:
            self._read_timer.stop()
            return
        if read != state["last"]:
            state["last"], state["changed"] = read, time.time()
        amount = vc.format_bytes(state["last"] - state["first"])
        if time.time() - state["changed"] > 3:
            self._read_timer.stop()
            seconds = state["changed"] - state["start"]
            self.read_info.setText(f"Read {amount} in {seconds:.0f} s.")
        else:
            self.read_info.setText(f"Reading… {amount} so far.")

    def _start_loading(self, text: str):
        """(0.18.0, the user's wish) Say what is being loaded before napari
        reads for the views: a box in the middle of the window and the
        busy pointer, until the window is free again (_check_loading); the
        line under Image counts what is read. The window can't redraw
        while napari reads, so the box is painted now, before it starts.
        Not in hidden windows (the tests)."""
        if not self.window().isVisible():
            return
        if self._loading_notice is None:
            self._loading_notice = _LoadingNotice(self.window())
        if self._loading is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        now = time.time()
        self._loading = {"text": text, "start": now, "read": _bytes_read(),
                         "first": _bytes_read(), "changed": now,
                         "tick": now, "free_since": now}
        if self.image is not None and not self.image.in_memory:
            self._watch_reading()
        self._loading_notice.show_text(text)
        for _ in range(3):              # paint it before napari starts
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self._loading_timer.start()

    def _check_loading(self):
        """Close the "Loading…" box once the window has been free for a
        while (see LOADING_QUIET_S). A tick that comes late means the
        window was busy (napari reading) until just now."""
        state = self._loading
        if state is None:
            self._loading_timer.stop()
            return
        now = time.time()
        if now - state["tick"] > LOADING_LATE_S:
            state["free_since"] = now
        state["tick"] = now
        read = _bytes_read()
        if read is not None and read != state["read"]:
            state["read"], state["changed"] = read, now
        free = now - state["free_since"]
        quiet = now - max(state["changed"], state["free_since"])
        if (free >= LOADING_QUIET_S and quiet >= LOADING_QUIET_S) \
                or free >= LOADING_FREE_S:
            self._end_loading()
        elif read is not None and state["first"] is not None:
            amount = vc.format_bytes(read - state["first"])
            self._loading_notice.show_text(
                f"{state['text']}\n{amount} read so far.")

    def _end_loading(self):
        self._loading_timer.stop()
        if self._loading_notice is not None:
            self._loading_notice.hide()
        if self._loading is not None:
            QApplication.restoreOverrideCursor()
        self._loading = None

    def _fit_panel_width(self):
        """Give the panel its full width once, when the first image opens.

        Only a starting size: the border can then be dragged to make the
        panel narrower (it scrolls) or to close it, and later images don't
        undo that.
        """
        if self._panel_fitted:
            return
        self._panel_fitted = True
        self.ortho.set_panel_width(self.sizeHint().width() + 30)

    def _show_napari_panels(self, show: bool):
        """Show or hide napari's layer list and layer controls (left)."""
        qt_viewer = self.viewer.window._qt_viewer
        for dock in (qt_viewer.dockLayerList, qt_viewer.dockLayerControls):
            dock.setVisible(show)

    def _levels_shown(self, levels: dict):
        """Show the level in use in the "Auto" entry, e.g. "Auto (level 1)",
        or per view if they differ ("Auto (XY 1, XZ 2, YZ 1)")."""
        text = "Auto (from zoom)"
        if self.level_combo.currentData() is None and levels:
            shown = sorted(set(levels.values()))
            if len(shown) == 1:       # the voxel size is in the view titles
                text = f"Auto (level {shown[0]})"
            else:
                text = "Auto (" + ", ".join(
                    f"{name} {level}" for name, level in levels.items()) + ")"
        self.level_combo.setItemText(0, text)

    def _image_summary(self) -> str:
        image = self.image
        level0 = image.levels[0]
        shape = " × ".join(str(n) for n in level0.array.shape)
        axes = ", ".join(image.axis_names)
        sizes = []
        # (a 2D image's one z plane has no thickness worth saying)
        flat_z = (self.ortho.view_for(("y", "x")).order[-3]
                  if self.ortho.flat else None)
        for axis in image.range_axes:
            if axis == flat_z:
                continue
            i = image.axis_names.index(axis)
            unit = display_unit(image.unit(axis))
            if self._is_time(axis):         # e.g. "12 timepoints, 5 min apart"
                sizes.append(f"{level0.array.shape[i]} timepoints, "
                             f"{level0.scale[i]:g} {unit} apart".rstrip())
            else:
                low, high = image.extent(axis)
                sizes.append(f"{axis} {high - low:.0f} {unit}".rstrip())
        size = vc.format_bytes(vc.nbytes(level0.array.shape,
                                         level0.array.dtype))
        if image.in_memory:          # a file read into memory (0.12.0)
            text = (f"{image.kind_name}, in memory: {shape} ({axes}), "
                    f"{size} (+ {len(image.levels) - 1} smaller copies for "
                    "the views).\n"
                    f"Size: {', '.join(sizes)}.\n"
                    f"Read: {self._file_summary()}.\n"
                    f"Crops are saved as {vc.output_suffix(image)} files.")
        else:
            zipped = {"ozx": " (zipped, .ozx)", "zip": " (zipped)"}.get(
                image.container, "")
            text = (f"OME-Zarr {image.ome_version}{zipped}, level 0: {shape} "
                    f"({axes}), {size}; {len(image.levels)} levels.\n"
                    f"Size: {', '.join(sizes)}.\n"
                    f"Loaded: {self._loaded_summary()}.")
        if self.ortho.flat:
            text += "\nA 2D image (one plane): only the XY view is shown."
        if any(self._is_time(axis) for axis in image.range_axes):
            text += ("\nTime-lapse: experimental. Check the first crops "
                     "(see the user guide).")
        return text

    def _loaded_summary(self) -> str:
        """The loaded region, e.g. "z 448–640 µm (levels 0–6)"."""
        image, region = self.image, self.ortho.region
        if region.whole:
            return "the whole image"
        parts = []
        for axis in image.range_axes:
            i = image.axis_names.index(axis)
            if region.shape(0)[i] == image.levels[0].array.shape[i]:
                continue                                  # not cut
            if self._is_time(axis):             # numbers 1, 2, 3 ...
                parts.append(f"timepoints {region.starts[0][i] + 1}–"
                             f"{region.stops[0][i]}")
            else:
                low, high = region.ranges[axis]
                unit = display_unit(image.unit(axis))
                parts.append(f"{axis} {low:.0f}–{high:.0f} {unit}".rstrip())
        return (f"{', '.join(parts)} (levels 0–{region.n_levels - 1})")

    def _file_summary(self) -> str:
        """What was read of a file (0.12.0), e.g. "image 2 of 3 (series2),
        z slices 10–60 of 80, channel 1 of 2; voxel size entered"."""
        source = self.image.source
        parts = []
        if source.get("scenes_in_file", 1) > 1:
            parts.append(f"image {source['scene_index'] + 1} of "
                         f"{source['scenes_in_file']} ({source['scene']})")
        shape = dict(zip(source["axes"], source["shape_in_file"]))
        names = {"t": "timepoints", "z": "slices", "y": "y pixels",
                 "x": "x pixels"}
        for axis, loaded in source["loaded"].items():
            n = shape.get(axis.upper(), 1)
            if axis == "c":
                if len(loaded) < n:
                    parts.append("channel" + ("s" if len(loaded) > 1 else "")
                                 + " " + ", ".join(map(str, loaded))
                                 + f" of {n}")
            elif loaded != [0, n]:
                parts.append(f"{names[axis]} {loaded[0] + 1}–{loaded[1]} "
                             f"of {n}")
        origins = source.get("calibration_from", {})
        entered = [k for k, v in origins.items() if v == "entered"]
        assumed = [k for k in ("x", "y") if origins.get(k) == "assumed"]
        if entered:
            parts.append("calibration entered: " + ", ".join(entered))
        if assumed:
            parts.append("no voxel size in the file: 1 assumed")
        return "; ".join(parts) or "the whole image"

    def _is_time(self, axis: str) -> bool:
        return self.image.axes[self.image.axis_names.index(axis)].get(
            "type") == "time"

    # --- view settings ---------------------------------------------------------

    def _on_outlines_3d(self):
        """(0.18.0) Outlines in 3D: off, the selected region or all."""
        self.ortho.set_outlines_3d(self.outlines_3d_combo.currentData(),
                                   self.outlines_3d_opacity.value())

    def _on_level_2d(self):
        if self.image is None:
            return
        level = self.level_combo.currentData()
        exact = self.exact_check.isChecked() and level is not None
        self.exact_check.setEnabled(level is not None)
        if exact:
            cost = max(ov.plane_read_bytes(self.image, level,
                                           self.ortho.views[n].plane,
                                           self.ortho.region,
                                           self.ortho.n_shown())
                       for n in ov.VIEWS_2D)
            if cost > LARGE_PLANE_READ and not _confirm(
                    self, "Large reads",
                    f"Showing exactly level {level} can read up to "
                    f"{vc.format_bytes(cost)} per view when you zoom out to "
                    "the whole image, because chunks are read whole. That "
                    "can be very slow.\n\nContinue?"):
                self._restore_view_state()
                return
        if self.ortho.projection != "none":
            answer = self._check_projection(self.ortho.projection,
                                            self.ortho.project_through, level)
            if answer == "cancel":
                self._restore_view_state()
                return
            if answer != "continue":                 # a coarser level instead
                level, exact = answer, False
                self.level_combo.setCurrentIndex(level + 1)  # 0 is "Auto"
                self._set_exact_quietly(False)
        self._start_loading(                                  # (0.18.0)
            "Loading the 2D views at the resolution for the zoom…"
            if level is None else
            f"Loading level {level} in the 2D views"
            + (" (exactly)…" if exact else "…"))
        self.ortho.set_2d_level(level, exact)
        self._levels_shown(self.ortho._levels_shown)     # the "Auto" text
        self._view_state.update(level_2d=self.level_combo.currentIndex(),
                                exact=self.exact_check.isChecked())

    def _set_exact_quietly(self, checked: bool):
        """Set the "exact level" box without triggering its handler."""
        self.exact_check.blockSignals(True)
        self.exact_check.setChecked(checked)
        self.exact_check.blockSignals(False)
        self.exact_check.setEnabled(self.level_combo.currentData() is not None)

    def _on_projection(self):
        if self.image is None:
            return
        mode = self.projection_combo.currentData()
        through = self.through_combo.currentData()
        self.through_combo.setEnabled(mode != "none")
        self._show_projection_options()
        # the slab's thickness and the chosen ranges (used when chosen)
        self.ortho.slab_thickness = self.slab_spin.value()
        self.ortho.project_ranges = self._projection_ranges()
        if mode != "none":
            answer = self._check_projection(mode, through,
                                            self.level_combo.currentData())
            if answer == "cancel":
                self._restore_view_state()
                return
            name = {"max": "maximum projection (MIP)",
                    "mean": "mean projection"}.get(mode, "projection")
            self._start_loading(                              # (0.18.0)
                f"Computing the {name} in the 2D views…")
            if answer != "continue":      # cap the 2D views at a coarser level
                self.level_combo.blockSignals(True)
                self.level_combo.setCurrentIndex(answer + 1)  # 0 is "Auto"
                self.level_combo.blockSignals(False)
                self._set_exact_quietly(False)
                self.ortho.set_2d_level(answer, False)
                self._view_state.update(level_2d=answer + 1, exact=False)
        else:
            self._start_loading("Loading the 2D views (slices)…")
        self.ortho.set_projection(mode, through)
        self._view_state.update(
            projection=self.projection_combo.currentIndex(),
            through=self.through_combo.currentIndex())

    def _projection_cost(self, through: str, level) -> int:
        """Bytes one projected view reads (the largest of the three) for the
        whole field of view at the finest level the 2D views may use: the
        chosen level, or with Auto the level shown now."""
        costs = []
        for name in ov.VIEWS_2D:
            view = self.ortho.views[name]
            k = level if level is not None else self.ortho.current_level(view)
            depth = view.order[-3]
            span = None
            if through == "box":
                i = self._selected()
                ranges = self.ortho.box_ranges
                if not 0 <= i < len(ranges) or ranges[i] is None:
                    continue              # no valid box: nothing to project
                span = ranges[i][depth]
            elif through == "slab":
                half = self.slab_spin.value() / 2
                centre = self.ortho.cursor.get(depth, 0.0)
                span = (centre - half, centre + half)
            elif through == "range":
                span = self._projection_ranges().get(depth)
            costs.append(ov.projection_read_bytes(
                self.image, k, view.plane, depth, span, self.ortho.region,
                self.ortho.n_shown()))
        return max(costs, default=0)

    def _check_projection(self, mode: str, through: str, level):
        """Ask before a projection that reads a lot.

        Returns "continue", "cancel", or a coarser level to use instead.
        """
        cost = self._projection_cost(through, level)
        if cost <= LARGE_PROJECTION_READ:
            return "continue"
        # The finest coarser level that stays under the limit, if any.
        start = (level if level is not None else min(
            self.ortho.current_level(self.ortho.views[n])
            for n in ov.VIEWS_2D)) + 1
        suggestion = next(
            ((k, c) for k in range(start, self.ortho.region.n_levels)
             if (c := self._projection_cost(through, k))
             <= LARGE_PROJECTION_READ), None)
        name = {"max": "maximum", "mean": "mean"}[mode]
        where = {"box": "the selected box", "slab": "the slab",
                 "range": "the chosen ranges"}.get(through, "the whole depth")
        text = (f"A {name} projection through {where} can read up to "
                f"{vc.format_bytes(cost)} per view at the current resolution "
                "(zoomed out to the whole image). That can be slow.")
        return _ask_projection(self, text, suggestion)

    def _on_3d(self):
        if self.image is None:
            return
        show = self.show_3d_check.isChecked()
        level = self.level_3d_combo.currentData()
        self.level_3d_combo.setEnabled(show)
        self.clip_check.setEnabled(show)
        self.mask_combo.setEnabled(show)
        self.outlines_3d_combo.setEnabled(show)
        self.outlines_3d_opacity.setEnabled(show)
        if show:
            k = self.ortho.auto_3d() if level is None else level
            size = ov.volume_bytes(self.image, k, self.ortho.region,
                                   self.ortho.n_shown())
            if size > LARGE_3D_VOLUME and not _confirm(
                    self, "Large 3D volume",
                    f"The 3D view at level {k} loads "
                    f"{vc.format_bytes(size)} into memory.\n\nContinue?"):
                self._restore_view_state()
                return
            self._start_loading(f"Loading the 3D view at level {k} "  # 0.18.0
                                f"({vc.format_bytes(size)})…")
        self.ortho.set_3d(show, level)
        self._view_state.update(show_3d=show,
                                level_3d=self.level_3d_combo.currentIndex())

    def _restore_view_state(self):
        """Put the view controls back after a cancelled change."""
        state = self._view_state
        for widget, setter, value in (
                (self.level_combo, "setCurrentIndex", state["level_2d"]),
                (self.exact_check, "setChecked", state["exact"]),
                (self.show_3d_check, "setChecked", state["show_3d"]),
                (self.level_3d_combo, "setCurrentIndex", state["level_3d"]),
                (self.projection_combo, "setCurrentIndex", state["projection"]),
                (self.through_combo, "setCurrentIndex", state["through"])):
            widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(False)
        self.exact_check.setEnabled(self.level_combo.currentData() is not None)
        self.through_combo.setEnabled(
            self.projection_combo.currentData() != "none")
        self._show_projection_options()

    def _on_link(self, linked: bool):
        self.ortho.link_cameras = linked
        if linked:
            self.ortho.reset_cameras()

    # --- boxes -----------------------------------------------------------------

    def _selected(self) -> int:
        return self.box_list.currentRow()

    def _new_box(self, ranges: dict | None = None) -> int:
        """Append a box (the loaded region unless ranges are given); return
        its index."""
        n = 1
        while any(vc.same_file_name(f"crop{n:02d}", box.name)
                  for box in self.boxes):
            n += 1
        full = {axis: self.ortho.extent(axis)
                for axis in self.image.range_axes}
        full.update(ranges or {})
        self.boxes.append(vc.CropBox(f"crop{n:02d}", full))
        return len(self.boxes) - 1

    def _add_box(self):
        if self.image is None:
            return
        ranges = {}   # the middle half of the loaded region in the XY plane
        for axis in self.ortho.views["XY"].plane:
            low, high = self.ortho.extent(axis)
            quarter = (high - low) / 4
            ranges[axis] = (low + quarter, high - quarter)
        self._refresh(select=self._new_box(ranges))

    def _rename_box(self):
        i = self._selected()
        if i < 0:
            return
        name, ok = QInputDialog.getText(self, "Rename region", "New name:",
                                        text=self.boxes[i].name)
        name = name.strip()
        if not ok or not name:
            return
        clash = next((b.name for j, b in enumerate(self.boxes)
                      if j != i and vc.same_file_name(b.name, name)), None)
        if clash is not None:
            QMessageBox.warning(
                self, "Rename region",
                f"{name!r} would be saved under the same file name as the "
                f"region {clash!r}, so one crop would replace the other.\n\n"
                "Region names become file names, where spaces and most "
                "punctuation turn into '_' and upper and lower case count as "
                "the same. Please choose another name.")
            return
        if self.boxes[i].name in self.ortho.hidden_boxes:   # stays hidden
            self.ortho.hidden_boxes.discard(self.boxes[i].name)
            self.ortho.hidden_boxes.add(name)
        self.boxes[i].name = name
        self._refresh(select=i)

    def _remove_box(self):
        """Delete the selected regions (several since 0.18.0: Ctrl/Shift+
        click, Ctrl+A); asks first if there are several (Undo brings them
        back). Then none is selected."""
        rows = sorted({self.box_list.row(item)
                       for item in self.box_list.selectedItems()})
        if not rows and self._selected() >= 0:
            rows = [self._selected()]
        rows = [r for r in rows if 0 <= r < len(self.boxes)]
        if not rows:
            return
        if len(rows) > 1 and not _confirm(
                self, "Delete regions",
                f"Delete {len(rows)} crop regions "
                f"({_name_list([self.boxes[r].name for r in rows])})?\n\n"
                "Undo (Ctrl+Z) brings them back."):
            return
        for r in reversed(rows):         # from the end: the rows stay valid
            del self.boxes[r]
        self._refresh(select=-1)

    def _new_region(self):
        """A new region (the loaded region until something is drawn),
        selected, with the polygon tool on. (0.18.0, the user's wish) With
        the 3D view on, it starts as the middle half of the loaded region
        along each spatial axis, so that its dots in 3D lie inside the
        image, ready to drag; drawing into it first makes it the whole
        loaded region again (_box_added), as before."""
        if self.image is None:
            return
        middle = None
        if self.ortho.show_3d:
            middle = {}
            for axis in self.ortho.views["3D"].order[-3:]:
                low, high = self.ortho.extent(axis)
                quarter = (high - low) / 4
                middle[axis] = (low + quarter, high - quarter)
        i = self._new_box(middle)
        self._fresh = (self.boxes[i], dict(self.boxes[i].ranges))
        self._refresh(select=i)
        self.ortho.set_tool("add_polygon")
        self.status.setText(f"Region {self.boxes[i].name!r}: draw around the "
                            "object on a few slices; Finish region when done.")

    def _finish_region(self):
        """Stop drawing: pan/zoom tool, nothing selected. A region made by
        New region into which nothing was drawn is removed again."""
        self.ortho.set_tool("pan_zoom")
        if self._fresh is not None:
            box, ranges = self._fresh
            if box in self.boxes and not box.outlines and box.ranges == ranges:
                self.boxes.remove(box)
        self._fresh = None
        self.status.setText("")
        self._refresh(select=-1)

    # --- undo / redo of the regions (0.18.0, the user's wish) ---------------
    # Every change to the regions ends in _refresh, which compares them with
    # the last snapshot: if they differ, that snapshot goes onto the undo
    # list. So all edits are covered without each one recording itself
    # (drawing, dragging, typing, deleting, renaming, loading regions,
    # the Autocropper...). A 3D drag is one step (recorded on the release),
    # and so is a burst of changes less than UNDO_MERGE_S apart.

    def _record_regions(self, select: int):
        if self._restoring or self.ortho._drag_3d is not None:
            return
        # Each region keeps an id (an attribute outside its fields, so not
        # in the text compared), and the panel keeps every region object:
        # undo then gives the same objects their old values back, so what
        # refers to a region keeps working (the Autocropper's seeds and its
        # Undo fit), also for a region brought back after deletion.
        for box in self.boxes:
            if not hasattr(box, "_vc_uid"):
                self._uids += 1
                box._vc_uid = self._uids
            self._box_objects[box._vc_uid] = box
        boxes = copy.deepcopy(self.boxes)
        text = repr(boxes)
        state = self._regions_state
        if state is None or text == state[0]:
            self._regions_state = (text, boxes if state is None else state[1],
                                   select)        # (the selection only)
            self._update_undo_buttons()
            return
        now = time.time()
        if not (self._undo and now - self._last_change < UNDO_MERGE_S):
            self._undo.append(state)               # as it was before
            del self._undo[:-UNDO_STEPS]
        self._redo.clear()
        self._last_change = now
        self._regions_state = (text, boxes, select)
        self._update_undo_buttons()

    def _undo_regions(self):
        """Undo the last change to the regions (Ctrl+Z)."""
        if self._undo and not self._busy:
            self._redo.append(self._regions_state)
            self._restore_regions(self._undo.pop())

    def _redo_regions(self):
        """Redo what Undo undid (Ctrl+Shift+Z, Ctrl+Y)."""
        if self._redo and not self._busy:
            self._undo.append(self._regions_state)
            self._restore_regions(self._redo.pop())

    def _restore_regions(self, state):
        text, boxes, select = state
        self._restoring = True
        try:
            restored = []
            for saved in boxes:
                values = copy.deepcopy(saved)
                box = self._box_objects.get(getattr(saved, "_vc_uid", None))
                if box is None:
                    box = values
                else:                  # the same object, its old values
                    box.__dict__.update(values.__dict__)
                restored.append(box)
            self.boxes[:] = restored   # (the same list, too)
            self._fresh = None
            self._regions_state = state
            self._last_change = 0.0      # the next change is a step of its own
            self._refresh(select=select if select < len(self.boxes) else -1)
        finally:
            self._restoring = False
        self._update_undo_buttons()

    def _reset_undo(self):
        """A new image: a new history (the first _refresh records it)."""
        self._undo.clear()
        self._redo.clear()
        self._regions_state = None
        self._box_objects = {}
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(bool(self._undo))
            self.redo_button.setEnabled(bool(self._redo))

    def _on_list_choice(self):
        """(0.18.0) The region list's selection changed (a click, Ctrl/
        Shift+click, Ctrl+A, the arrow keys); called once Qt has finished.
        The region clicked last is the selected region if it is still
        selected (Ctrl+click can take it out), else the last one selected,
        or none."""
        rows = sorted(self.box_list.row(item)
                      for item in self.box_list.selectedItems())
        row = self.box_list.currentRow()
        if row not in rows:
            row = rows[-1] if rows else -1
        self._refresh(select=row)

    def eventFilter(self, obj, event):
        """A click on an empty part of the region list selects none."""
        if obj is self.box_list.viewport() \
                and event.type() == QEvent.Type.MouseButtonPress \
                and self.box_list.itemAt(event.position().toPoint()) is None:
            self.box_list.setCurrentRow(-1)
        return False

    # Called by the views when shapes are drawn, edited or deleted.
    def _box_added(self, ranges: dict, plane: tuple, outline):
        """A shape was drawn in a view (outline: polygon points, or None for
        a rectangle). It goes into the selected region, or a new one if none
        is selected. A rectangle sets the region's range in that view (and
        removes its polygons there). A polygon or lasso is added on the
        slice shown (replacing one on that slice); on a projection it is a
        silhouette through the projected depth."""
        if outline is not None and self._too_thin(outline):
            return
        i = self._selected()
        if not 0 <= i < len(self.boxes):
            i = self._new_box(ranges if outline is not None else None)
        box = self.boxes[i]
        if (self._fresh is not None and self._fresh[0] is box
                and box.ranges == self._fresh[1] and not box.outlines):
            # (0.18.0) the first drawing into a new region that still has
            # its starting size (the middle half, with the 3D view on): the
            # whole loaded region again, as drawing always started from
            box.ranges = {axis: self.ortho.extent(axis)
                          for axis in self.image.range_axes}
            self._fresh = (box, dict(box.ranges))
        if outline is None:                               # a rectangle
            # (0.17.0: a moving region, where it is at the timepoint shown)
            vc.set_ranges_at(box, self.ortho.time_shown(), ranges)
            _drop_view(box, plane)
            self._refresh(select=i)
            return
        if box.track:
            self._refresh(select=i)                 # (the polygon goes)
            self.status.setText(
                f"Region {box.name!r} moves with the object: polygons on a "
                "moving region are not possible yet. Fix it in place first "
                "(button under the regions), or draw a new region.")
            return
        view = self.ortho.view_for(plane)
        depth_axis = view.order[-3]
        unit = display_unit(self.image.unit(depth_axis))
        projected = self.ortho.projection_range(view)
        if projected is not None:
            key = (float(projected[0]), float(projected[1]))
            where = (f"on a projection ({depth_axis} {key[0]:.0f}–"
                     f"{key[1]:.0f} {unit}): a silhouette")
        else:
            key = self.ortho.slice_of(plane)
            where = f"at {depth_axis} = {key:.1f} {unit}"
        _set_polygon(box, plane, key, outline,
                     self.ortho.slice_tolerance(depth_axis))
        _fit_to_outlines(box, self.image)
        self._refresh(select=i)
        name = self.ortho.polygon_names(box).get(
            (_stored_plane(box, plane), vc._normal_key(key)), "polygon")
        self.status.setText(f"Region {box.name!r}: {name} added in "
                            f"{view.name} {where}.")

    def _box_ranges_dragged(self, i: int, ranges: dict,
                            shift: dict | None = None):
        """(0.18.0) A dot of region i was dragged in the 3D view: its
        ranges along those axes (as seen at the timepoint shown, for a
        moving region). A face's drag leaves the polygons as they are (they
        still shape the region); the centre's (shift, µm per axis) moves
        them with the region: for polygons per timepoint, those of the
        timepoint shown."""
        if not 0 <= i < len(self.boxes):
            return
        box, time_shown = self.boxes[i], self.ortho.time_shown()
        vc.set_ranges_at(box, time_shown, ranges)
        if shift and any(shift.values()):
            spatial = vc._spatial_axes(self.image)

            def move(b):
                b.outlines = box_handles.shifted_outlines(b.outlines, shift,
                                                          spatial)
            if box.timed_outlines:
                _edit_outlines(box, time_shown, move)
            elif box.outlines:
                move(box)
        self._refresh(select=i)

    def _boxes_changed(self, changes: dict, plane: tuple):
        """Shapes moved or reshaped in a view: a box's rectangle (its range
        there) or one of its polygons (key: its slice)."""
        for i, (ranges, outline, key) in changes.items():
            if i >= len(self.boxes):
                continue
            box = self.boxes[i]
            if key == ov.RECT or outline is None:
                # (0.17.0: a moving region, at the timepoint shown: a move
                # corrects its place then, a resize its size everywhere)
                vc.set_ranges_at(box, self.ortho.time_shown(), ranges)
            elif self._too_thin(outline):
                continue                    # (the polygon stays as it was)
            elif box.timed_outlines:
                # (0.17.0) outlines per timepoint: this timepoint's only
                _edit_outlines(box, self.ortho.time_shown(),
                               lambda b: _replace_polygon(b, plane, key,
                                                          outline))
            else:
                _replace_polygon(box, plane, key, outline)
                _fit_to_outlines(box, self.image)
        self._refresh(select=min(changes) if changes else None)

    def _too_thin(self, outline) -> bool:
        """A polygon with (almost) no area, e.g. three clicks on a line by
        accident, is refused (the user's choice, 2026-11-10): the shape
        would narrow to nothing there. True (and says so) if it is less
        than one voxel wide on average; the views then drop it."""
        voxel = vc.pixel_size(self.image)
        if vc.polygon_width(outline) >= voxel:
            return False
        self._refresh()
        self.status.setText(
            "That polygon has (almost) no area (less than one voxel wide), "
            "so it was not used. Draw it again around the object.")
        return True

    def _clear_outlines(self):
        """Remove all polygons of the selected region (a plain box again)."""
        i = self._selected()
        if 0 <= i < len(self.boxes):
            self.boxes[i].outlines = {}
            self.boxes[i].margin = 0.0
            self._refresh(select=i)

    def _on_margin_changed(self, value: int):
        """The selected region's margin (voxels) was changed."""
        i = self._selected()
        if not 0 <= i < len(self.boxes) or self.boxes[i].margin == value:
            return
        self.boxes[i].margin = float(value)
        if not self.boxes[i].track:      # (a moving region keeps its size)
            _fit_to_outlines(self.boxes[i], self.image)
        self._refresh(select=i)

    def _boxes_removed(self, items: list, plane: tuple):
        """Shapes deleted in a view: a polygon removes just that polygon, a
        rectangle the whole box."""
        for i, key in items:
            if key != ov.RECT and i < len(self.boxes):
                self._remove_one_polygon(i, plane, key)
        for i in sorted({i for i, key in items if key == ov.RECT},
                        reverse=True):
            if i < len(self.boxes):
                del self.boxes[i]
        self._refresh()

    # --- the selected box's polygons ----------------------------------------

    def _fill_polygon_list(self):
        """List the selected region's polygons: view, slice (go to,
        delete)."""
        self.polygon_list.clear()
        i = self._selected()
        box = None
        if self.image is not None and 0 <= i < len(self.boxes):
            box = self._shown_box(i)
        has_polygons = box is not None and bool(box.outlines)
        # (0.16.0) the polygons part only while there are polygons
        self.polygon_part.setVisible(has_polygons)
        if not has_polygons:
            return
        names = self.ortho.polygon_names(box)
        for plane, keys in sorted(box.outlines.items()):
            view = self.ortho.view_for(plane)
            axis = view.order[-3]
            unit = display_unit(self.image.unit(axis))
            for key in sorted(keys, key=vc._slice_order):
                name = names[(plane, key)]
                if key is None:
                    text = f"{name} · through the region"
                elif vc.is_silhouette(key):
                    text = (f"{name} · projection {axis} {key[0]:.0f}–"
                            f"{key[1]:.0f} {unit} (silhouette)")
                elif ov.reaches_through(box, plane):
                    text = (f"{name} · {axis} = {key:.1f} {unit} (only "
                            "polygon: through the region)")
                else:
                    text = f"{name} · {axis} = {key:.1f} {unit}"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, (plane, key))
                self.polygon_list.addItem(item)

    def _chosen_polygon(self):
        item = self.polygon_list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _shown_box(self, i: int) -> vc.CropBox:
        """Region i with the polygons of the timepoint shown (a region
        outlined per timepoint, 0.17.0)."""
        box = self.boxes[i]
        if box.timed_outlines:
            box = vc.box_at(box, self.ortho.time_shown())
        return box

    def _go_to_polygon(self):
        chosen = self._chosen_polygon()
        if chosen is not None and isinstance(chosen[1], float):
            self.ortho.go_to_slice(*chosen)

    def _delete_polygon(self):
        """Delete the selected polygons of the selected region (several
        since 0.18.0); asks first if there are several."""
        i = self._selected()
        chosen = [item.data(Qt.ItemDataRole.UserRole)
                  for item in self.polygon_list.selectedItems()]
        if not chosen and self._chosen_polygon() is not None:
            chosen = [self._chosen_polygon()]
        if not chosen or not 0 <= i < len(self.boxes):
            return
        if len(chosen) > 1:
            names = self.ortho.polygon_names(self._shown_box(i))
            if not _confirm(
                    self, "Delete polygons",
                    f"Delete {len(chosen)} polygons of region "
                    f"{self.boxes[i].name!r} ("
                    f"{_name_list([names.get(c, '?') for c in chosen])})?"
                    "\n\nUndo (Ctrl+Z) brings them back."):
                return
        for plane, key in chosen:
            self._remove_one_polygon(i, plane, key)
        self._refresh(select=i)

    def _remove_one_polygon(self, i: int, plane: tuple, key):
        """Remove one polygon of region i: of the timepoint shown, for
        outlines per timepoint (0.17.0)."""
        box = self.boxes[i]
        if box.timed_outlines:
            _edit_outlines(box, self.ortho.time_shown(),
                           lambda b: _remove_polygon(b, plane, key))
        else:
            _remove_polygon(box, plane, key)
            _fit_to_outlines(box, self.image)

    # --- shading of the selected box -----------------------------------------

    def _choose_shade_colour(self, which: str):
        """Pick the colour of the kept voxels or of those set to 0."""
        title = ("Colour of the voxels that are kept" if which == "kept"
                 else "Colour of the voxels set to 0")
        colour = QColorDialog.getColor(QColor(self.ortho.shading[which]),
                                       self, title)
        if colour.isValid():
            self.ortho.set_shading(**{which: colour.name()})
            self._show_shade_colours()

    def _show_shade_colours(self):
        for which, button in self.shade_buttons.items():
            button.setStyleSheet(
                f"background-color: {self.ortho.shading[which]};")

    # --- keeping the display in step with the boxes ------------------------

    def _refresh(self, select: int | None = None):
        if self.image is None:
            return
        if select is None:
            select = self._selected()
        select = min(max(select, -1), len(self.boxes) - 1)   # -1: none
        self._record_regions(select)                          # (0.18.0)
        names = [box.name for box in self.boxes]
        hidden = self.ortho.hidden_boxes
        hidden &= set(names)                # forget boxes that are gone
        self.box_list.blockSignals(True)
        # (0.18.0) Rows selected besides `select` (Ctrl/Shift+click) stay
        # selected while `select` is one of them; the list is then updated
        # in place, as rebuilding it would lose them.
        chosen = {self.box_list.row(item)
                  for item in self.box_list.selectedItems()}
        rebuild = self.box_list.count() != len(self.boxes)
        if rebuild:
            self.box_list.clear()
        for row, box in enumerate(self.boxes):  # a tick box: shown in views
            name = box.name
            # (0.17.0) "object01 · moves (40 timepoints found)"
            text = (f"{name} · moves"
                    + (", outlined" if box.timed_outlines else "")
                    + f" ({len(box.track)} timepoints found)"
                    if box.track else name)
            if rebuild:
                item = QListWidgetItem(text)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setToolTip("Untick to hide this region in all views; "
                                "it is still cropped.")
                self.box_list.addItem(item)
            else:
                item = self.box_list.item(row)
                item.setText(text)
            item.setCheckState(Qt.CheckState.Unchecked if name in hidden
                               else Qt.CheckState.Checked)
            font = item.font()        # the selected region in bold
            font.setBold(row == select)
            item.setFont(font)
        if select < 0:
            self.box_list.clearSelection()
            self.box_list.setCurrentRow(-1)
        elif rebuild or select not in chosen:
            self.box_list.setCurrentRow(select)      # only it is selected
        else:
            self.box_list.setCurrentRow(
                select, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.box_list.blockSignals(False)
        self._fit_box_list()
        ranges, moves = [], []
        for box in self.boxes:
            try:
                plan = self._plan(box)
            except ValueError:
                ranges.append(None)
                moves.append(None)
                continue
            ranges.append(plan.snapped_ranges)
            # (0.17.0) a moving region: its ranges as cropped at any time
            moves.append(functools.partial(vc.snapped_ranges_at, self.image,
                                           plan) if plan.moving else None)
        self.ortho.update_boxes(self.boxes, ranges, select, moves)
        self._show_selected()
        # (0.17.0) seeds take the colour of the region they gave
        if hasattr(self, "object_group"):
            self.object_group.refresh_seeds()
        self.levels.on_regions_changed()                       # (0.18.0)

    def _on_time_changed(self):
        """(0.17.0) The timepoint shown changed: the number fields and the
        plan of a selected moving region show it where it is now; the
        Levels histogram reads that timepoint (0.18.0)."""
        self.levels.on_time_changed()
        i = self._selected()
        if 0 <= i < len(self.boxes) and self.boxes[i].track:
            self._show_selected()

    def _fix_in_place(self):
        """(0.17.0) The selected moving region becomes one fixed region that
        holds all its places over its timepoints."""
        i = self._selected()
        if not 0 <= i < len(self.boxes) or not self.boxes[i].track:
            return
        box = self.boxes[i]
        if box.timed_outlines and not _confirm(
                self, "Fix in place",
                f"Region {box.name!r} has an outline for each timepoint. A "
                "fixed region cannot keep them: they are removed, and the "
                "fixed region keeps everything inside it.\n\nFix it in "
                "place?"):
            return
        fixed = vc.fixed_in_place(self.image, box)
        box.ranges, box.track, box.timed_outlines = fixed.ranges, {}, {}
        self._refresh(select=i)
        self.status.setText(f"Region {box.name!r} is fixed in place now: it "
                            "holds all the places it moved to.")

    def _fit_box_list(self):
        """(0.16.0) The region list as tall as its regions: 2 rows at
        least, 6 at most (then it scrolls)."""
        rows = max(2, min(self.box_list.count(), 6))
        row_height = self.box_list.sizeHintForRow(0)
        if row_height <= 0:
            row_height = self.box_list.fontMetrics().height() + 6
        frame = 2 * self.box_list.frameWidth()
        self.box_list.setFixedHeight(rows * row_height + frame + 2)

    def _on_box_ticked(self, item):
        """A box's tick box in the list: show or hide that box in all views
        (hidden boxes are still cropped)."""
        row = self.box_list.row(item)
        if not 0 <= row < len(self.boxes):
            return
        hidden = set(self.ortho.hidden_boxes)
        if item.checkState() == Qt.CheckState.Checked:
            hidden.discard(self.boxes[row].name)
        else:
            hidden.add(self.boxes[row].name)
        if hidden != self.ortho.hidden_boxes:
            self.ortho.set_hidden_boxes(hidden)

    def _show_selected(self):
        """Fill the number fields, the polygon list and the plan text for
        the selected box."""
        self._fill_polygon_list()
        i = self._selected()
        has_box = self.image is not None and 0 <= i < len(self.boxes)
        for spins in self.range_spins.values():
            for spin in spins:
                spin.setEnabled(has_box)
        if self.t_step_spin is not None:
            self.t_step_spin.setEnabled(has_box)
        for check in self.channel_checks:
            check.setEnabled(has_box)
        self.margin_spin.setEnabled(has_box and (
            bool(self.boxes[i].outlines)
            or bool(self.boxes[i].timed_outlines)))
        self.fix_button.setVisible(has_box and bool(self.boxes[i].track))
        if not has_box:
            self.plan_label.setText(
                ("No region selected." if self.boxes else "No region yet.")
                if self.image else "")
            return
        box = self.boxes[i]
        self.margin_spin.blockSignals(True)
        self.margin_spin.setValue(int(round(box.margin)))
        self.margin_spin.blockSignals(False)
        # (0.17.0) a moving region: where it is at the timepoint shown
        shown = self._shown_ranges(box)
        for axis, spins in self.range_spins.items():
            values = shown[axis]
            if self._is_time(axis):         # timepoint numbers, not times
                values = self._t_numbers(axis, *values)
            for spin, value in zip(spins, values):
                spin.blockSignals(True)     # don't trigger _on_spin_changed
                spin.setValue(value)
                spin.blockSignals(False)
        self._show_t_times()
        if self.t_step_spin is not None:
            self.t_step_spin.blockSignals(True)
            self.t_step_spin.setValue(box.t_step)
            self.t_step_spin.blockSignals(False)
        for c, check in enumerate(self.channel_checks):
            check.blockSignals(True)
            check.setChecked(box.channels is None or c in box.channels)
            check.blockSignals(False)
        try:
            plan = self._plan(box)
            self.plan_label.setText(self._plan_text(plan))
        except ValueError as err:
            self.plan_label.setText(f"Problem: {err}")

    def _plan(self, box: vc.CropBox) -> vc.CropPlan:
        """The crop plan as it will be written: with the pyramid levels for
        an OME-Zarr; full resolution only for a file's image (its crop is
        one TIFF, so the region is not widened to fit coarser levels)."""
        if self.image.in_memory:
            return vc.plan_crop(self.image, box, n_levels=1)
        return vc.plan_crop(self.image, box, min_size=MIN_LEVEL_SIZE)

    def _plan_text(self, plan: vc.CropPlan) -> str:
        """Two-line summary of what will be written."""
        ranges = []
        level0 = self.image.levels[0]
        # (0.17.0) a moving region: its ranges at the timepoint shown
        time = self.ortho.time_shown()
        snapped = (vc.snapped_ranges_at(self.image, plan, time)
                   if plan.moving and time is not None
                   else plan.snapped_ranges)
        for axis in self.image.range_axes:
            low, high = snapped[axis]
            unit = display_unit(self.image.unit(axis))
            if self._is_time(axis):
                # Timepoint numbers (1 = the first) rather than ranges:
                # "t 2–8 (30–210 min)", "t 2–8, every 2nd: 4 kept (...)"
                i = self.image.axis_names.index(axis)
                first, last = plan.starts[0][i], plan.stops[0][i] - 1
                t0, t1 = (level0.translation[i] + n * level0.scale[i]
                          for n in (first, last))
                step = plan.step(i)
                every = ("" if step == 1 else f", every {vc._ordinal(step)}"
                         f": {plan.shapes[0][i]} kept")
                ranges.append(f"{axis} {first + 1}–{last + 1}{every} "
                              f"({t0:g}–{t1:g} {unit})".replace(" )", ")"))
            else:
                ranges.append(f"{axis} {low:.1f}–{high:.1f} {unit}".rstrip())
        shape = " × ".join(str(n) for n in plan.shapes[0])
        size = vc.format_bytes(vc.nbytes(plan.shapes[0],
                                         self.image.levels[0].array.dtype))
        if self.image.in_memory:      # the crop of a file: one TIFF
            compressed = (", zstd (smaller on disk)"
                          if self.compress_check.isChecked() else "")
            text = (f"Crop: {', '.join(ranges)}\n"
                    f"{shape} voxels, {size}; a "
                    f"{vc.output_suffix(self.image)} file{compressed}")
        else:
            text = (f"Crop: {', '.join(ranges)}\n"
                    f"{shape} voxels, {size}; levels "
                    f"0–{plan.level_indices[-1]}")
        asked = self._tiff_level()
        if asked is not None:
            tplan = vc.tiff_plan(self.image, plan.box, asked)
            level = vc.tiff_level_for(tplan, asked)
            dtype = self.image.levels[0].array.dtype
            name = "full resolution" if level == 0 else f"level {level}"
            text += (f"\nSave as TIFF: {name}, " + " × ".join(
                str(n) for n in tplan.shapes[level]) + " voxels, "
                + vc.format_bytes(vc.tiff_bytes(tplan, dtype, asked))
                + ("" if level == asked else
                   f" (the image has no level {asked})"))
        box = plan.box
        if plan.moving:
            positions = len({tuple(row) for row in plan.shifts[0].tolist()})
            moved = plan.shifts[0] * np.asarray(
                self.image.levels[plan.level_indices[0]].scale)
            distance = float(np.sqrt((moved ** 2).sum(axis=1)).max())
            unit = display_unit(self.image.unit(
                vc._spatial_axes(self.image)[-1]))
            grid = vc.downsampling_factors(self.image)[
                plan.level_indices[-1]]
            steps = sorted({round(level0.scale[i] * grid[i], 3)
                            for i in vc._moving_axes(self.image, box)})
            text += (f"\nMoves with the object: {positions} places (up to "
                     f"{distance:.0f} {unit} from the first), in steps of "
                     f"{'/'.join(f'{s:g}' for s in steps)} {unit} (ranges: "
                     "at the timepoint shown)")
            if box.timed_outlines:
                line = plan.timed_outlines_text()
                text += (f"\n{line[0].upper()}{line[1:]}; ⚠ slower on large "
                         "data")
        if box.outlines:
            spatial = vc._spatial_axes(self.image)
            counts = {}
            for plane, keys in box.outlines.items():
                name = self.ortho.view_for(plane).name
                for key in keys:
                    kind = ("silhouette" if vc.is_silhouette(key)
                            else "polygon")
                    counts[kind] = counts.get(kind, []) + [name]
            parts = [f"{len(v)} {kind}{'s' if len(v) > 1 else ''} "
                     f"({', '.join(sorted(set(v)))})"
                     for kind, v in sorted(counts.items(), reverse=True)]
            text += "\nShape: " + " + ".join(parts)
            if box.margin:
                distance = abs(box.margin) * vc.pixel_size(self.image)
                unit = display_unit(self.image.unit(spatial[-1]))
                text += (f"; margin {box.margin:+g} voxels "
                         f"({distance:g} {unit})")
            worst = vc.outline_disagreements(box.outlines, spatial)
            if worst and worst[0][4] > 3 * vc.pixel_size(self.image):
                plane_a, slice_a, plane_b, slice_b, gap = worst[0]
                unit = display_unit(self.image.unit(spatial[-1]))
                text += (f"\n⚠ {self._polygon_name(box, plane_a, slice_a)} "
                         f"and {self._polygon_name(box, plane_b, slice_b)} "
                         f"disagree by {gap:.0f} {unit} where they cross")
        return text

    def _polygon_name(self, box, plane, key) -> str:
        """E.g. "xy2 (z = 40)" for a polygon on a slice."""
        name = self.ortho.polygon_names(box).get((plane, key), "")
        return f"{name} ({self.ortho.view_for(plane).order[-3]} = {key:.0f})"

    def _shown_ranges(self, box: vc.CropBox) -> dict:
        """A box's ranges as asked at the timepoint shown (a moving region,
        0.17.0: where it is then). Worked out from the panel's own box, so
        that it also works while the views are rebuilt (the 0.17.0 review:
        "What to load..." crashed asking the views, which had no regions
        yet, and the regions vanished from the views)."""
        time = self.ortho.time_shown()
        if box.track and time is not None:
            return vc.ranges_at(box, time)
        return box.ranges

    def _on_spin_changed(self, value=None):
        i = self._selected()
        if i < 0:
            return
        box = self.boxes[i]
        shown = self._shown_ranges(box)
        changed = {}
        for axis, (start, stop) in self.range_spins.items():
            if self._is_time(axis):     # numbers -> times (frame centres)
                box.ranges[axis] = (self._t_time(axis, start.value()),
                                    self._t_time(axis, stop.value()))
            elif not box.track:
                box.ranges[axis] = (start.value(), stop.value())
            else:
                # (0.17.0) a moving region, edited where it is at the
                # timepoint shown: only the axes whose fields changed, and
                # a field that did not change keeps its exact value (the
                # fields show rounded numbers)
                old = shown.get(axis, self.ortho.extent(axis))
                new = []
                for spin, value in zip((start, stop), old):
                    same = abs(spin.value() - value) <= \
                        0.5 * 10 ** -spin.decimals()
                    new.append(value if same else spin.value())
                if tuple(new) != tuple(old):
                    changed[axis] = tuple(new)
        if changed:
            vc.set_ranges_at(box, self.ortho.time_shown(), changed)
        self._refresh(select=i)

    def _set_from_cursor(self, axis: str, which: int):
        """Set the box start (0) or stop (1) to the crosshair position."""
        if self._selected() >= 0 and axis in self.ortho.cursor:
            value = self.ortho.cursor[axis]
            if self._is_time(axis):     # the timepoint shown
                first, interval, _ = self._t_frames(axis)
                value = int(round((value - first) / interval)) + 1
            self.range_spins[axis][which].setValue(value)

    def _on_channels_changed(self, checked=None):
        i = self._selected()
        if i < 0:
            return
        kept = [c for c, check in enumerate(self.channel_checks)
                if check.isChecked()]
        self.boxes[i].channels = (None if len(kept) == len(self.channel_checks)
                                  else kept)
        self._show_selected()

    # --- saving and loading boxes -------------------------------------------

    def _save_boxes(self) -> bool:
        """Save the regions as CSV; True if they were saved."""
        if self.image is None or not self.boxes:
            return False
        folder = self.out_edit.text().strip() or str(self.image.path.parent)
        stem = vc.image_stem(self.image)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save crop regions", str(Path(folder) / f"{stem}_boxes.csv"),
            "CSV files (*.csv)")
        if not path:
            return False
        vc.save_boxes_csv(self.boxes, path, self.image)
        self.status.setText(f"Saved {len(self.boxes)} region(s) to {path}")
        return True

    def _load_boxes(self):
        if self.image is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load crop regions", str(self.image.path.parent),
            "CSV files (*.csv)")
        if not path:
            return
        try:
            boxes = vc.load_boxes_csv(path)
            for box in boxes:
                vc.plan_crop(self.image, box)       # check it fits the image
        except ValueError as err:     # our own messages say what to fix
            QMessageBox.warning(self, "Cannot load regions", str(err))
            return
        except Exception as err:
            QMessageBox.warning(self, "Cannot load regions",
                                f"{type(err).__name__}: {err}")
            return
        replace = True
        if self.boxes:
            answer = _choose(
                self, "Load regions",
                f"There are {len(self.boxes)} crop region(s) already. Add the "
                f"{len(boxes)} region(s) from the file to them, or replace "
                "them?", ["Add", "Replace"])
            if answer is None:
                return
            replace = answer == "Replace"
            if not replace:
                try:
                    vc.check_box_names(self.boxes + boxes)
                except ValueError as err:
                    QMessageBox.warning(self, "Cannot add regions", str(err))
                    return
        for box in boxes:   # the GUI shows every axis explicitly
            for axis in self.image.range_axes:
                box.ranges.setdefault(axis, self.image.extent(axis))
            # The box holds all its polygons (boxes saved before 0.5.0 can
            # be narrower than their outlines).
            if box.outlines:
                _fit_to_outlines(box, self.image)
        if replace:
            self.boxes = boxes
            self.ortho.hidden_boxes.clear()      # new boxes start shown
            self._refresh(select=0)
        else:
            first = len(self.boxes)
            self.boxes = self.boxes + boxes
            self._refresh(select=first)

    # --- cropping ---------------------------------------------------------------

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Folder for the cropped images", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    def _crop(self, all_boxes: bool, tiff_only: bool = False):
        """Crop the selected region (or all), or with tiff_only (0.15.0,
        "Save as TIFF…") save the selected region straight to a TIFF at the
        chosen resolution, without an OME-Zarr crop."""
        if self.image is None or not self.boxes:
            return
        if tiff_only and self._selected() < 0:
            QMessageBox.information(self, "Save as TIFF",
                                    "Select a region in the list first.")
            return
        if all_boxes:
            boxes = self.boxes
        elif self._selected() >= 0:
            boxes = [self.boxes[self._selected()]]
        else:
            return
        boxes = copy.deepcopy(boxes)   # later edits must not affect the run
        if not self.out_edit.text().strip():
            self._choose_output()
            if not self.out_edit.text().strip():
                return
        out_dir = Path(self.out_edit.text().strip())
        min_size = MIN_LEVEL_SIZE
        in_memory = self.image.in_memory      # a file's crops are TIFFs
        compression = "zstd" if self.compress_check.isChecked() else None
        try:
            vc.check_output_folder(self.image, out_dir)
            vc.check_box_names(self.boxes)
            plans = [self._plan(box) for box in boxes]
            if in_memory:
                vc.check_tiff_image(self.image)     # data type and axes
        except ValueError as err:
            QMessageBox.warning(self, "Cannot crop", str(err))
            return
        # A TIFF only with "Save as TIFF…" (0.15.0); then no OME-Zarr crop,
        # and the region planned down to that level (vc.tiff_plan)
        tiff_level = self._tiff_level() if tiff_only else None
        if tiff_only:
            try:
                plans = [vc.tiff_plan(self.image, box, tiff_level)
                         for box in boxes]
            except ValueError as err:
                QMessageBox.warning(self, "Cannot save as TIFF", str(err))
                return
        tiff_paths = []        # per box: its TIFF (None without)
        for plan in plans:
            out_path = vc.output_path_for(self.image, plan.box, out_dir)
            tiff_paths.append(None if tiff_level is None else vc.tiff_path_for(
                out_path, vc.tiff_level_for(plan, tiff_level)))
        if tiff_level is not None:
            try:
                vc.check_tiff_image(self.image)     # data type and axes
            except ValueError as err:
                QMessageBox.warning(self, "Cannot save as TIFF", str(err))
                return
        existing = [] if tiff_only else [
            p for p in (vc.output_path_for(self.image, box, out_dir)
                        for box in boxes) if p.exists()]
        existing += [p for p in tiff_paths if p is not None and p.exists()]
        if existing and not self.overwrite_check.isChecked():
            QMessageBox.warning(
                self, "Output exists",
                "These outputs exist already:\n"
                + "\n".join(p.name for p in existing)
                + "\n\nTick 'Overwrite earlier crops' to replace them, "
                "or rename the boxes.")
            return
        try:   # every box, before the first one is written
            for box, tiff_path in zip(boxes, tiff_paths):
                if not tiff_only:
                    vc.check_output(
                        self.image, box,
                        vc.output_path_for(self.image, box, out_dir),
                        overwrite=self.overwrite_check.isChecked())
                if tiff_path is not None:
                    vc.check_tiff_output(
                        tiff_path, self.overwrite_check.isChecked(), box.name,
                        image=self.image)
        except (ValueError, FileExistsError) as err:
            QMessageBox.warning(self, "Cannot crop", str(err))
            return

        # All levels, uncompressed: an upper limit of what will be written
        # (the same total as the progress line), and the TIFFs.
        dtype = self.image.levels[0].array.dtype
        total = sum(vc.nbytes(shape, dtype)
                    for plan in plans for shape in plan.shapes)
        if tiff_only:       # one level, in a temporary folder, then the TIFF
            total = sum(vc.nbytes(plan.shapes[vc.tiff_level_for(
                plan, tiff_level)], dtype) for plan in plans)
        if in_memory:       # the crops are the TIFFs
            tiff_paths = [vc.output_path_for(self.image, plan.box, out_dir)
                          for plan in plans]
            tiff_sizes = [vc.nbytes(plan.shapes[0], dtype) for plan in plans]
            kind = self.image.source.get("output", "imagej")
        elif tiff_level is not None:
            tiff_sizes = [vc.tiff_bytes(plan, dtype, tiff_level)
                          for plan in plans]
            total += sum(tiff_sizes)
            kind = "imagej"
        if in_memory or tiff_level is not None:
            large = [path.name for path, size in zip(tiff_paths, tiff_sizes)
                     if size > vc.TIFF_LIMIT]
            smaller = ("" if in_memory else
                       " A coarser level is 4–8 times smaller.")
            if large and not _confirm(
                    self, "Large TIFF",
                    f"{', '.join(large)} will be larger than 4 GB "
                    f"(largest: {vc.format_bytes(max(tiff_sizes))} before "
                    f"compression). {vc.large_tiff_note(kind, compression)}"
                    f"{smaller}\n\nSave anyway?"):
                return
        free = _free_space(out_dir)
        if free is not None and total > free and not _confirm(
                self, "Disk space",
                f"These crops need up to {vc.format_bytes(total)} "
                "(uncompressed; compressed crops are usually smaller), but "
                f"the output drive has {vc.format_bytes(free)} free.\n\n"
                "Crop anyway?"):
            return
        options = {
            "keep_position": KEEP_POSITION,
            "workers": WORKERS,
            "overwrite": self.overwrite_check.isChecked(),
            "cancel_event": self._cancel,
            "tiff_level": tiff_level,
            "compression": compression,
            "tiff_only": tiff_only,
        }
        self._tiff_asked = tiff_level     # for the report when done
        self._cancel.clear()
        # The handlers are given when the worker is made: otherwise superqt
        # also re-raises any error (even a Cancel) in the main thread, which
        # napari shows as a second, red error message.
        self._worker = _run_crops(
            self.image, boxes, out_dir, min_size, options,
            self.progress_signal.emit, _start_thread=False,
            _connect={"returned": self._on_crop_done,
                      "errored": self._on_crop_error,
                      "finished": lambda: self._set_busy(False)})
        self._set_busy(True)
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.status.setText(
            (f"Saving {boxes[0].name} as TIFF, " if tiff_only else
             f"Cropping {len(boxes)} region(s), ")
            + f"{vc.format_bytes(total)} uncompressed …")
        self._crop_started = time.time()
        self._worker.start()

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.crop_selected_button.setEnabled(not busy)
        self.crop_all_button.setEnabled(not busy)
        self.tiff_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setText("Cancel")

    def _cancel_crop(self):
        """Ask the running crop to stop; it stops once the parts being
        written are done (which can take a while with large shards)."""
        self._cancel.set()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Stopping…")
        self.status.setText("Stopping after the parts being written …")

    def _on_progress(self, done: float, total: float, text: str):
        """Progress in bytes written (all boxes), with a time estimate from
        the speed so far (shown once it has settled a little)."""
        if self._cancel.is_set():           # keep showing "Stopping …"
            return
        self.progress_bar.setValue(int(1000 * done / total) if total else 0)
        elapsed = time.time() - self._crop_started
        line = (f"Cropping {text}: {vc.format_bytes(done)} of "
                f"{vc.format_bytes(total)} written")
        if 0 < done < total and elapsed > 3:
            speed = done / elapsed
            line += (f" ({vc.format_bytes(speed)}/s), about "
                     f"{_duration((total - done) / speed)} left")
        self.status.setText(f"{line}; {_duration(elapsed)} so far.")

    def _on_crop_done(self, results):
        lines, failed, failed_tiffs = [], [], []
        for result in results:
            ok = "checked OK" if not result.problems else "CHECK FAILED"
            if result.tiff is not None and result.tiff == result.path:
                # "Save as TIFF…" (0.15.0): the TIFF only
                level = result.tiff_level
                asked = self._tiff_asked
                name = "full resolution" if level == 0 else f"level {level}"
                note = ("" if level == asked else
                        f" (this region has no level {asked})")
                lines.append(f"{result.path.name}: {name}{note}, "
                             f"{result.seconds:.0f} s, {ok}")
                if result.note:                              # (0.18.0)
                    lines.append(f"  note: {result.note}")
                if result.problems:
                    failed_tiffs.append(f"{result.path.name}: "
                                        + "; ".join(result.problems[:3]))
                continue
            lines.append(f"{result.path.name}: {result.seconds:.0f} s, {ok}")
            if result.removed_tiffs:
                lines.append("  deleted the old TIFF(s) of the crop it "
                             "replaced: " + ", ".join(
                                 p.name for p in result.removed_tiffs))
            if result.tiff is not None:
                asked = self._tiff_asked
                note = ("" if result.tiff_level == asked else
                        f" (this crop has no level {asked})")
                tiff_ok = ("checked OK" if not result.tiff_problems
                           else "CHECK FAILED")
                lines.append(f"  + {result.tiff.name}: level "
                             f"{result.tiff_level}{note}, "
                             f"{result.tiff_seconds:.0f} s, {tiff_ok}")
            if result.problems:
                failed.append(f"{result.path.name}: "
                              + vc.verification_text(result.plan,
                                                     result.problems))
            if result.tiff_problems:
                failed_tiffs.append(f"{result.tiff.name}: "
                                    + "; ".join(result.tiff_problems[:3]))
        if results[-1].checked:
            lines.append("Checked: " + results[-1].checked + ".")
        if any(result.tiff is not None and result.tiff != result.path
               for result in results):
            lines.append("Each TIFF was checked against its crop: its size, "
                         "voxel size and the voxels of a few slices.")
        self.status.setText("Done.\n" + "\n".join(lines))
        if failed:
            QMessageBox.warning(
                self, "Check failed",
                "The check against the source found problems. The crops "
                "were kept, and their record says that the check failed; "
                "please send this message to the maintainer.\n\n"
                + "\n\n".join(failed[:10]))
        if failed_tiffs:
            QMessageBox.warning(
                self, "TIFF check failed",
                "The check of the TIFF against its crop found problems. The "
                "TIFFs were kept (the crops themselves are not affected); "
                "please send this message to the maintainer.\n\n"
                + "\n\n".join(failed_tiffs[:10]))
        self._last_output = results[-1].path
        self.open_last_button.setEnabled(True)

    def _on_crop_error(self, err: Exception):
        if isinstance(err, vc.CropCancelled):
            self.status.setText(
                f"{err} Crop it again with 'Overwrite earlier crops' to "
                "replace it.")
        elif vc.is_disk_full(err):
            self.status.setText("Crop failed: the output drive is full.")
            QMessageBox.critical(
                self, "Crop failed",
                "The output drive is full. The crop being written is "
                "incomplete. Free some space, or choose another output "
                "folder, and crop again (with 'Overwrite earlier crops' to "
                "replace the incomplete crop).")
        else:
            self.status.setText("Crop failed.")
            QMessageBox.critical(self, "Crop failed",
                                 f"{type(err).__name__}: {err}")

    def _open_last_crop(self):
        """Open the last crop in a new, complete Volume Cropper window, so
        that it can be checked or cropped further (into the same folder)."""
        if self._last_output is not None:
            open_window(self._last_output, out_dir=self.out_edit.text().strip(),
                        ask=True)


# ---------------------------------------------------------------------------
# Small Qt helpers
# ---------------------------------------------------------------------------

class Section(QWidget):
    """A part of the panel that opens and closes with a click on its title
    (0.16.0, the user's wish: a shorter panel). Its controls go into
    `body`. `on_toggle(key, opened)` is called when it is opened or
    closed (the panel remembers that, see settings.py)."""

    def __init__(self, title: str, key: str, opened: bool = True,
                 on_toggle=None):
        super().__init__()
        self.key, self.on_toggle = key, on_toggle
        self.on_show = None        # f(opened), however it is opened or closed
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; }")
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        self.header.setToolTip("Click to open or close this part.")
        outer.addWidget(self.header)
        self.body = QFrame()
        self.body.setFrameShape(QFrame.Shape.StyledPanel)
        outer.addWidget(self.body)
        self.header.setChecked(opened)
        self._show(opened)
        self.header.toggled.connect(self._toggled)

    def _show(self, opened: bool):
        self.header.setArrowType(Qt.ArrowType.DownArrow if opened
                                 else Qt.ArrowType.RightArrow)
        self.body.setVisible(opened)
        if self.on_show is not None:
            self.on_show(opened)

    def _toggled(self, opened: bool):
        self._show(opened)
        if self.on_toggle is not None:
            self.on_toggle(self.key, opened)

    def is_open(self) -> bool:
        return self.header.isChecked()

    def set_open(self, opened: bool):
        """Open or close it without telling on_toggle (settings applied)."""
        self.header.blockSignals(True)
        self.header.setChecked(opened)
        self.header.blockSignals(False)
        self._show(opened)


class SettingsDialog(QDialog):
    """Settings… (0.16.0): the user's choices, remembered between sessions
    (settings.py), and napari's memory cache (napari remembers it)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Cropper settings")
        values = settings.load()
        # (0.18.0) two columns: the window and panel; the Autocropper's
        # and the View part's options
        outer = QVBoxLayout(self)
        columns = QHBoxLayout()
        outer.addLayout(columns)
        layout = QVBoxLayout()
        group = QGroupBox("New windows start with")
        inner = QVBoxLayout(group)
        self.checks = {}
        for key, text in (
                ("compress_tiffs", "Compress TIFFs (zstd) ticked"),
                ("link_views", "Link zoom/pan of the 2D views ticked"),
                ("exact_level", "exact level (2D resolution) ticked"),
                ("napari_layer_panels", "napari's layer panels (left) shown")):
            check = QCheckBox(text)
            check.setChecked(bool(values[key]))
            inner.addWidget(check)
            self.checks[key] = check
        layout.addWidget(group)
        layout.addWidget(self._opening_group(values))         # (0.18.0)
        group = QGroupBox("Panel")
        inner = QVBoxLayout(group)
        inner.addWidget(QLabel("Parts open (they also remember being opened "
                               "or closed):"))
        self.section_checks = {}
        for key, text in settings.SECTION_NAMES.items():
            check = QCheckBox(text)
            check.setChecked(bool(values["sections"].get(key, True)))
            inner.addWidget(check)
            self.section_checks[key] = check
        self.checks["show_tool_layers"] = QCheckBox(
            "List Volume Cropper's own layers in napari's layer list")
        self.checks["show_tool_layers"].setToolTip(
            "Off: napari's layer list shows only the image channels and the "
            "layers you draw on; the region lines, crosshair, preview and "
            "shading are switched in the panel's View part.")
        self.checks["show_tool_layers"].setChecked(
            bool(values["show_tool_layers"]))
        inner.addWidget(self.checks["show_tool_layers"])
        layout.addWidget(group)
        columns.addLayout(layout)
        columns.addLayout(self._defaults_column(values))
        layout = outer
        row = QHBoxLayout()
        row.addWidget(QLabel("Memory cache (napari's own setting)"))
        self.cache_spin = QDoubleSpinBox()
        self.cache_spin.setDecimals(1)
        self.cache_spin.setSingleStep(0.5)
        self.cache_spin.setSuffix(" GB")
        self.cache_spin.setRange(0.0, _max_cache_gb())
        cache = _dask_settings()
        self.cache_spin.setValue(cache.cache if cache.enabled else 0.0)
        row.addWidget(self.cache_spin)
        layout.addLayout(row)
        where = _label(f"Kept in {settings.path()} (yours only).")
        where.setStyleSheet("color: gray;")
        layout.addWidget(where)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _defaults_column(self, values: dict) -> QVBoxLayout:
        """(0.18.0, the user's wish) The Autocropper's and the View part's
        options new windows start with."""
        from volume_cropper import autocrop as ac
        column = QVBoxLayout()
        auto = values["autocropper"]
        group = QGroupBox("Autocropper starts with")
        form = QFormLayout(group)
        self.auto_widgets = {}

        def combo(items, current):
            box = QComboBox()
            for text, data in items:
                box.addItem(text, data)
            box.setCurrentIndex(max(box.findData(current), 0))
            return box

        def check(text, on):
            box = QCheckBox(text)
            box.setChecked(bool(on))
            return box

        widgets = self.auto_widgets
        widgets["method"] = combo([(ac.METHOD_NAMES[m], m)
                                   for m in ac.METHODS], auto["method"])
        form.addRow("Threshold", widgets["method"])
        widgets["grow"] = check("Dim parts too", auto["grow"])
        form.addRow(widgets["grow"])
        widgets["margin"] = QDoubleSpinBox()
        widgets["margin"].setRange(0.0, 10000.0)
        widgets["margin"].setDecimals(1)
        widgets["margin"].setValue(float(auto["margin"]))
        widgets["margin"].setSuffix(" (image unit)")
        form.addRow("Margin", widgets["margin"])
        widgets["moving"] = check("Moving region (time-lapses)",
                                  auto["moving"])
        form.addRow(widgets["moving"])
        widgets["outline"] = check("Outline", auto["outline"])
        widgets["outline_kind"] = combo(
            [("silhouettes (XY, XZ, YZ)", "silhouettes"),
             ("per slice (XY)", "slices")], auto["outline_kind"])
        form.addRow(widgets["outline"], widgets["outline_kind"])
        widgets["seed_mode"] = combo([(ac.SEED_MODE_NAMES[m], m)
                                      for m in ac.SEED_MODES],
                                     auto["seed_mode"])
        form.addRow("A seed finds the", widgets["seed_mode"])
        widgets["split"] = check("Split an object between its seeds",
                                 auto["split"])
        form.addRow(widgets["split"])
        column.addWidget(group)
        view = values["view"]
        group = QGroupBox("View starts with")
        form = QFormLayout(group)
        self.view_widgets = widgets = {}
        widgets["slice_planes"] = check("slice planes (3D)",
                                        view["slice_planes"])
        form.addRow(widgets["slice_planes"])
        widgets["mask_mode"] = combo(
            [("show normally", "show"), ("dim", "dim"), ("grey", "grey"),
             ("dim and grey", "dimgrey"), ("hide (as saved)", "hide")],
            view["mask_mode"])
        form.addRow("Blacked-out voxels (3D)", widgets["mask_mode"])
        widgets["outlines_3d"] = combo(list(ov.OUTLINE_3D_MODES),
                                       view["outlines_3d"])
        widgets["outlines_3d_opacity"] = QDoubleSpinBox()
        widgets["outlines_3d_opacity"].setRange(0.05, 1.0)
        widgets["outlines_3d_opacity"].setSingleStep(0.05)
        widgets["outlines_3d_opacity"].setValue(
            float(view["outlines_3d_opacity"]))
        row = QHBoxLayout()
        row.addWidget(widgets["outlines_3d"], 1)
        row.addWidget(widgets["outlines_3d_opacity"])
        form.addRow("Outlines in 3D", row)
        widgets["crosshairs"] = check("crosshairs", view["crosshairs"])
        widgets["crop_regions"] = check("crop regions", view["crop_regions"])
        widgets["crossing_marks"] = check("× marks", view["crossing_marks"])
        row = QHBoxLayout()
        row.addWidget(widgets["crosshairs"])
        row.addWidget(widgets["crop_regions"])
        row.addWidget(widgets["crossing_marks"])
        form.addRow("Show", row)
        widgets["preview"] = combo([("off", "off"),
                                    ("in the main views", "main")],
                                   view["preview"])
        form.addRow("Preview crop", widgets["preview"])
        from volume_cropper.levels_panel import SOURCES
        widgets["levels_source"] = combo(list(SOURCES),
                                         view["levels_source"])
        widgets["levels_log"] = check("log", view["levels_log"])
        # (0.22.0) the axis zoomed to min–max
        widgets["levels_zoom"] = check("zoom to min–max",
                                       view["levels_zoom"])
        row = QHBoxLayout()
        row.addWidget(widgets["levels_source"], 1)
        row.addWidget(widgets["levels_log"])
        row.addWidget(widgets["levels_zoom"])
        form.addRow("Levels histogram", row)
        column.addWidget(group)
        column.addStretch()
        return column

    def _opening_group(self, values: dict) -> QGroupBox:
        """(0.18.0, the user's wish) What opening an image starts with: the
        resolutions by voxel size (the user's choice), the channels shown
        and whether "What to load" asks."""
        opening = values["opening"]
        group = QGroupBox("Opening an image starts with")
        form = QFormLayout(group)
        self.opening_widgets = widgets = {}

        def size_row(key, choices, current, voxel):
            box = QComboBox()
            for text, data in choices:
                box.addItem(text, data)
            box.setCurrentIndex(max(box.findData(current), 0))
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 100000.0)
            spin.setDecimals(3)
            spin.setSuffix(" µm")
            spin.setValue(float(voxel))
            spin.setEnabled(box.currentData() == "size")
            box.currentIndexChanged.connect(
                lambda _: spin.setEnabled(box.currentData() == "size"))
            widgets[key], widgets["voxel" + key[-3:]] = box, spin
            row = QHBoxLayout()
            row.addWidget(box, 1)
            row.addWidget(spin)
            return row

        least = "the finest level with voxels of at least"
        form.addRow("2D resolution", size_row(
            "level_2d", [("Auto (from zoom)", "auto"), (least, "size")],
            opening["level_2d"], opening["voxel_2d"]))
        form.addRow("3D view", size_row(
            "view_3d", [("Off", "off"), ("Auto", "auto"), (least, "size")],
            opening["view_3d"], opening["voxel_3d"]))
        widgets["level_2d"].setToolTip(
            "Auto: napari reads the level that suits the zoom. By voxel size: "
            "the finest level whose voxels are at least this large (along "
            "x, as 'What to load' lists them; the coarsest if none is), so "
            "images from other set-ups open with the same detail.")
        widgets["channels"] = QComboBox()
        widgets["channels"].addItem("all", "all")
        widgets["channels"].addItem("only the first", "first")
        widgets["channels"].setCurrentIndex(
            max(widgets["channels"].findData(opening["channels"]), 0))
        form.addRow("Channels shown", widgets["channels"])
        widgets["ask"] = QCheckBox("Ask 'What to load'")
        widgets["ask"].setChecked(bool(opening["ask"]))
        widgets["ask"].setToolTip(
            "Off: an image opens at once with these choices (the whole "
            "image). A 3D volume larger than "
            f"{vc.format_bytes(LARGE_3D_VOLUME)} still asks first. The "
            "button 'What to load...' always asks. napari layers always "
            "ask (their axes).")
        form.addRow(widgets["ask"])
        return group

    @staticmethod
    def _read(widget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        return float(widget.value())

    def values(self) -> dict:
        values = settings.load()
        for key, check in self.checks.items():
            values[key] = check.isChecked()
        values["sections"] = {key: check.isChecked()
                              for key, check in self.section_checks.items()}
        values["autocropper"] = {key: self._read(widget) for key, widget
                                 in self.auto_widgets.items()}
        values["view"] = {key: self._read(widget) for key, widget
                          in self.view_widgets.items()}
        values["opening"] = {key: self._read(widget) for key, widget
                             in self.opening_widgets.items()}
        return values


class _LoadingNotice(QFrame):
    """(0.18.0) The "Loading…" box in the middle of the window (see
    CropWidget._start_loading). A small window of its own, above the main
    window, that takes no clicks or focus."""

    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.WindowType.Tool
                         | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 16)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.RichText)   # see show_text
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        font = self.label.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.label.setFont(font)
        self.texts: list[str] = []      # what it said (for the checks)

    def show_text(self, text: str):
        """Show text (a first line, then details), centred on the window."""
        self.texts.append(text)
        lines = text.split("\n")
        html = [f"<b>{escape(lines[0])}</b>", *map(escape, lines[1:]),
                "<span style='color: gray'>The window responds again when "
                "it is done.</span>"]
        self.label.setText("<br>".join(html))
        self.adjustSize()
        window = self.parentWidget().window()
        centre = window.frameGeometry().center()
        self.move(centre.x() - self.width() // 2,
                  centre.y() - self.height() // 2)
        self.show()
        self.raise_()
        self.repaint()


class _WheelGuard(QObject):
    """(0.15.0, the user's wish) Scrolling the panel with the mouse wheel
    or the touchpad must not change a number field or a list that passes
    under the pointer. Those react to the wheel only when they have the
    focus (clicked first); otherwise the wheel scrolls the panel."""

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel or obj.hasFocus():
            return False
        widget = obj.parentWidget()
        while widget is not None and not isinstance(widget, QScrollArea):
            widget = widget.parentWidget()
        if widget is not None:            # the panel scrolls instead
            QApplication.sendEvent(widget.verticalScrollBar(), event)
        return True


_WHEEL_GUARD = None


def guard_wheel(widget: QWidget):
    """Make every number field and list inside `widget` take the wheel
    only when focused (see _WheelGuard); they no longer take the focus from
    the wheel itself either. Safe to call again for new controls."""
    global _WHEEL_GUARD
    if _WHEEL_GUARD is None:
        _WHEEL_GUARD = _WheelGuard()
    for kind in (QAbstractSpinBox, QComboBox):
        for child in widget.findChildren(kind):
            if child.property("vc_wheel_guard"):
                continue
            child.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            child.installEventFilter(_WHEEL_GUARD)
            child.setProperty("vc_wheel_guard", True)

def _remember_section(key: str, opened: bool):
    """A part of the panel opened or closed: remembered for next time."""
    settings.update(sections={key: opened})


def _button(text: str, slot) -> QPushButton:
    button = QPushButton(text)
    button.clicked.connect(slot)
    return button


def _label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _confirm(parent, title: str, text: str) -> bool:
    answer = QMessageBox.question(parent, title, text)
    return answer == QMessageBox.StandardButton.Yes


def _name_list(names: list, most: int = 6) -> str:
    """Names for a question: "a, b, c", or "a, b, c, … (10 in all)" for
    more than `most`."""
    if len(names) <= most:
        return ", ".join(names)
    return ", ".join(names[:most]) + f", … ({len(names)} in all)"


def _choose(parent, title: str, text: str, options: list[str]):
    """Ask with one button per option, plus Cancel; return the option
    chosen, or None for Cancel (or closing the window)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    buttons = [box.addButton(option, QMessageBox.ButtonRole.AcceptRole)
               for option in options]
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    clicked = box.clickedButton()
    return options[buttons.index(clicked)] if clicked in buttons else None


def _free_space(folder) -> int | None:
    """Free bytes on the drive of folder, or of its nearest existing parent
    (the folder may not exist yet); None if that cannot be found out."""
    folder = Path(folder).absolute()
    for candidate in (folder, *folder.parents):
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free
            except OSError:
                return None
    return None


def _ask_projection(parent, text: str, suggestion):
    """Large-projection dialog: use a coarser level, continue, or cancel.

    suggestion: (level, bytes) of a coarser level under the limit, or None.
    Returns that level, "continue" or "cancel".
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Large projection")
    box.setText(text)
    use = None
    if suggestion is not None:
        level, cost = suggestion
        use = box.addButton(
            f"Use level {level} instead ({vc.format_bytes(cost)})",
            QMessageBox.ButtonRole.AcceptRole)
    go = box.addButton("Continue anyway", QMessageBox.ButtonRole.DestructiveRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(use or go)
    box.exec()
    clicked = box.clickedButton()
    if use is not None and clicked is use:
        return suggestion[0]
    return "continue" if clicked is go else "cancel"


def _clear_layout(layout):
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget is not None:
            # hidden at once: deleteLater waits for Qt's event loop, and
            # until then the old widget would still be drawn (seen in the
            # 0.12.0 check, over the row below it)
            widget.hide()
            widget.deleteLater()


def _fit_to_outlines(box: vc.CropBox, image: vc.OmeZarrImage):
    """After a box's polygons (or its margin) changed: along every axis
    that its outlines limit, set the box's range to hold all of them (and
    a positive margin), within the image, so that nothing is cut off by an
    older, smaller box; making the polygons smaller makes the box smaller
    again. The crop itself is then fitted to where the outlines can keep
    voxels (volume_cropper.plan_crop)."""
    spans: dict = {}
    for axis, low, high in vc.outline_limits(image, box):
        spans.setdefault(axis, []).append((low, high))
    for axis, limits in spans.items():
        low, high = image.extent(axis)
        box.ranges[axis] = (max(min(a for a, _ in limits), low),
                            min(max(b for _, b in limits), high))


def _stored_plane(box: vc.CropBox, plane: tuple) -> tuple:
    """The key under which the box keeps this view's polygons: the view's
    plane, or the same plane the other way round if stored like that."""
    swapped = (plane[1], plane[0])
    return swapped if swapped in box.outlines else tuple(plane)


def _as_stored(box: vc.CropBox, plane: tuple, points) -> list:
    """Points given as (vertical, horizontal) of the view, in the order the
    box stores them."""
    if _stored_plane(box, plane) != tuple(plane):
        return [(float(b), float(a)) for a, b in points]
    return [(float(a), float(b)) for a, b in points]


def _set_polygon(box: vc.CropBox, plane: tuple, key, points,
                 tolerance: float):
    """Make `points` the box's polygon in this view with this key (a slice,
    or a depth range for a silhouette), replacing one with the same key
    (within `tolerance` µm), or one without a slice (from a boxes file
    before 0.7.0)."""
    stored = _stored_plane(box, plane)
    keys = {k: p for k, p in box.outlines.get(stored, {}).items()
            if k is not None and not _same_key(k, key, tolerance)}
    keys[vc._normal_key(key)] = _as_stored(box, plane, points)
    box.outlines[stored] = keys


def _same_key(a, b, tolerance: float) -> bool:
    """Whether two polygon keys (slices or depth ranges) are the same."""
    if vc.is_silhouette(a) != vc.is_silhouette(b):
        return False
    if vc.is_silhouette(a):
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))
    return abs(a - b) <= tolerance


def _replace_polygon(box: vc.CropBox, plane: tuple, key, points):
    """The box's polygon `key` in this view, moved or reshaped."""
    stored = _stored_plane(box, plane)
    if key in box.outlines.get(stored, {}):
        box.outlines[stored][key] = _as_stored(box, plane, points)


def _remove_polygon(box: vc.CropBox, plane: tuple, key):
    """Remove one of the box's polygons (and the view, if it was the last)."""
    stored = _stored_plane(box, plane)
    keys = box.outlines.get(stored, {})
    keys.pop(key, None)
    if not keys:
        box.outlines.pop(stored, None)


def _edit_outlines(box: vc.CropBox, time, change) -> bool:
    """(0.17.0) Change a region's outlines at a timepoint: change(b) edits
    b.outlines of a stand-in holding the outlines per timepoint of that
    time (box.timed_outlines), which are then stored back (and dropped if
    none are left: that timepoint keeps the whole region). False if the
    region has no outlines then."""
    key = next((t for t in box.timed_outlines
                if time is not None and vc.same_time(t, time)), None)
    if key is None:
        return False
    stand_in = vc.CropBox(box.name, outlines=copy.deepcopy(
        box.timed_outlines[key]))
    change(stand_in)
    if stand_in.outlines:
        box.timed_outlines[key] = stand_in.outlines
    else:
        del box.timed_outlines[key]
    return True


def _drop_view(box: vc.CropBox, plane: tuple):
    """Remove all of the box's polygons in this view."""
    box.outlines.pop(tuple(plane), None)
    box.outlines.pop((plane[1], plane[0]), None)


def _duration(seconds: float) -> str:
    """A rough duration: "40 s", "3 min", "1 h 20 min"."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours} h {minutes} min"


def _guard_layer_list(viewer) -> bool:
    """Work around a napari 0.9.2 bug in its layer list (the user's report,
    2026-11-08: "AttributeError: 'NoneType' object has no attribute
    'height'" in LayerDelegate._paint_loading, when another image was
    opened). napari can paint a row whose layer has just been removed (its
    own code says so: "Guard against stale indices"), but it only skips
    rows Qt calls invalid; such a row of its sorted list can still count as
    valid and then gives no data at all. The thumbnail and lock icon cope
    with that; the loading indicator takes "no data" for "not loaded" and
    asks for the row's size, which is None. So here the loading indicator
    is skipped for a row without data, as the thumbnail already is.
    Returns True if the guard is in place."""
    try:
        delegate = viewer.window._qt_viewer.layers.itemDelegate()
    except (AttributeError, RuntimeError):
        return False
    original = getattr(delegate, "_paint_loading", None)
    if original is None:
        return False
    if getattr(delegate, "_volume_cropper_guard", False):
        return True

    def paint_loading(painter, option, index):
        if index.data(Qt.ItemDataRole.SizeHintRole) is None:
            return                       # the row's layer has gone
        original(painter, option, index)
    delegate._paint_loading = paint_loading
    delegate._volume_cropper_guard = True
    return True


def _dask_settings():
    """napari's own setting for its data cache (size in GB, on/off)."""
    from napari.settings import get_settings
    return get_settings().application.dask


def _dask_cache():
    """napari's data cache (a cachey.Cache: total_bytes, clear())."""
    from napari.utils import resize_dask_cache
    return resize_dask_cache().cache     # no arguments: only returns it


def _max_cache_gb() -> float:
    """The largest cache napari accepts: half the computer's memory."""
    try:       # rounded down: napari refuses anything above the half
        return int(psutil.virtual_memory().total * 0.5 / 1e8) / 10
    except Exception:
        return 16.0


def _bytes_read() -> int | None:
    """Bytes this program has read from disk or network so far, or None
    where that is not available (psutil can't tell it on a Mac)."""
    try:
        return psutil.Process().io_counters().read_bytes
    except Exception:
        return None


# Every window this program opened, as (viewer, panel). Python deletes
# objects that nothing refers to, and a window must not vanish that way.
# A window leaves the list when it is closed (keep_window).
_WINDOWS: list = []
# Images being read into memory now (_read_file): while one is read, the
# other windows stay usable, but a second read is not started.
_READING = 0


def is_reading() -> bool:
    return _READING > 0


def is_tool_viewer(viewer) -> bool:
    """True for a napari window that is a Volume Cropper window."""
    viewer = getattr(viewer, "__wrapped__", viewer)
    return any(v is viewer for v, _ in _WINDOWS)


class _CloseGuard(QObject):
    """Watches a Volume Cropper window (0.13.0).

    - Every close goes through napari's own Viewer.close(): napari's
      closeEvent alone (the window's X, File > Close Window) leaves the
      viewer, its layers and so the image (gigabytes, if read into memory)
      in memory until napari quits. So a close that does not come from
      Viewer.close() is turned into one (scheduled right after).
    - While a crop is written: closing the window with its X asks first
      ("No" keeps the window and the crop). When napari closes the window
      itself (quitting: the viewer is already torn down then), nothing can
      be kept, so the crop is stopped without asking.
    - When the window is gone, it is forgotten (_forget_window).
    """

    def __init__(self, viewer, widget: CropWidget):
        window = viewer.window._qt_window
        super().__init__(window)
        self.viewer, self.widget = viewer, widget
        self._closing = False        # Viewer.close() was called by us
        window.installEventFilter(self)
        window.destroyed.connect(
            lambda *_, w=widget: _forget_window(w))

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Close:
            return False
        return self.on_close(obj, event, event.spontaneous())

    def on_close(self, window, event, spontaneous: bool) -> bool:
        """Handle a Close event; True = handled here (the window stays open
        for now), False = let napari close the window."""
        # Viewer.close() (ours, or napari's when quitting) clears the layers
        # before it closes the window: then let it close.
        torn_down = self._closing or len(self.viewer.layers) == 0
        if self.widget._busy:
            if spontaneous and not torn_down and not _confirm(
                    window, "A crop is running",
                    "A crop is still being written. Stop it and close the "
                    "window? (The crop being written stays incomplete; crop "
                    "it again with 'Overwrite earlier crops' to replace "
                    "it.)"):
                event.ignore()
                return True
            self._stop_crop()
        elif spontaneous and not torn_down and not _napari_may_close(window):
            event.ignore()               # napari's own "Close window?"
            return True
        if torn_down:
            return False
        self._closing = True
        event.ignore()
        QTimer.singleShot(0, self.viewer.close)
        return True

    def _stop_crop(self):
        """Stop the running crop and wait for it (at most 2 minutes)."""
        self.widget._cancel_crop()
        end = time.time() + 120
        while self.widget._busy and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.05)


def _napari_may_close(window) -> bool:
    """napari's own question before a window is closed with its X (if its
    setting "Confirm window or application closing" is on), which napari
    asks only when it closes the window itself."""
    try:
        from napari.settings import get_settings
        if not get_settings().application.confirm_close_window:
            return True
        from napari._qt.dialogs.confirm_close_dialog import ConfirmCloseDialog
        from qtpy.QtWidgets import QDialog
        return ConfirmCloseDialog(window, close_app=False).exec_() == \
            QDialog.DialogCode.Accepted
    except ImportError:
        return True


def attach_panel(viewer) -> CropWidget:
    """Turn a napari window into a Volume Cropper window: the four views,
    and the panel to their right (in a splitter, so that the border can be
    dragged; a napari dock on the right could not be); kept until the
    window is closed. Used by open_window and the plugin's "Use this
    window…"."""
    # A window taken over by "Use this window…" may show 3D, a grid or
    # rolled/transposed axes; its canvas becomes the XY view, which needs
    # plain 2D (the 0.17.0 review: in 3D, opening an image there crashed).
    viewer.dims.ndisplay = 2
    viewer.grid.enabled = False
    viewer.dims.order = tuple(range(viewer.dims.ndim))
    widget = CropWidget(viewer)
    scroll = QScrollArea()          # the panel scrolls when made narrow
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    scroll.setMinimumWidth(180)     # can be dragged this narrow
    widget.ortho.add_side_panel(scroll)
    keep_window(viewer, widget)
    return widget


def keep_window(viewer, widget: CropWidget):
    """Keep a Volume Cropper window (napari viewer + panel) until it is
    closed: referenced in _WINDOWS, watched by a _CloseGuard."""
    _WINDOWS.append((viewer, widget))
    widget._close_guard = _CloseGuard(viewer, widget)


def _forget_window(widget: CropWidget):
    """A Volume Cropper window is gone: drop it and its image (an image
    read into memory can be gigabytes)."""
    _WINDOWS[:] = [(v, w) for v, w in _WINDOWS if w is not widget]
    if widget.image is not None:
        # (0.21.0) a zipped image's file: Windows keeps an open file from
        # being replaced, e.g. a crop opened here and cropped again
        widget.image.close()
    widget.image = None
    widget._layer_reader = None
    widget._close_guard = None
    try:
        widget.ortho.image = None
    except Exception:
        pass


def open_window(path=None, out_dir: str | None = None, show: bool = True,
                channels: list[int] | None = None, ask: bool = False):
    """Open a Volume Cropper window: a napari viewer with the panel on the
    right of the four views.

    path:     image to open straight away (optional).
    out_dir:  pre-filled output folder (optional).
    channels: the channels to show at first (None = all).
    ask:      show the "What to load" dialog before opening the image.
    Returns (viewer, panel). Both the first window and "Open last crop in a
    new window" are made here, so every window has all the tools.
    """
    viewer = napari.Viewer(title=f"Volume Cropper {vc.__version__}",
                           show=show)
    widget = attach_panel(viewer)
    if out_dir:
        widget.out_edit.setText(str(out_dir))
    if path:
        widget.open_image(path, channels, ask=ask)
    return viewer, widget


def main(argv: list[str] | None = None):
    # If the program crashes inside napari/OpenGL (not a Python error),
    # print where it happened instead of closing silently.
    faulthandler.enable()
    parser = argparse.ArgumentParser(
        prog="volume_cropper",          # not "__main__.py" (the launchers)
        description="Volume Cropper: crop OME-Zarr images in napari.")
    parser.add_argument("path", nargs="?", help="OME-Zarr image to open")
    parser.add_argument(
        "--channels", help="channels to show at first, e.g. 0 or 0,2 (ticked "
        "in the 'What to load' dialog; the others are not read until shown; "
        "default: all)")
    args = parser.parse_args(argv)
    channels = None
    if args.channels:
        try:
            channels = [int(c) for c in args.channels.split(",") if c.strip()]
        except ValueError:
            parser.error(f"--channels: expected numbers such as 0 or 0,2, "
                         f"not {args.channels!r}")
    open_window(args.path, channels=channels, ask=True)
    napari.run()


