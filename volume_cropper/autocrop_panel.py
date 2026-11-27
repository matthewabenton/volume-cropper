"""The panel's experimental group "Autocropper" (0.14.0): fits a
crop region to the bright object in the image, with autocrop.find_object.

It is a checkable group, off at first (its controls are greyed out until
it is ticked). "Fit region to object":
- with a crop region selected: searches all of that region (its t range
  too, also where it reaches beyond the part loaded for the views: the
  search reads the image itself) and shrinks the region to the object,
  plus the margin; "Undo fit" gives the region its earlier ranges back;
- with no region selected: searches the loaded part of the image and adds
  a new region "object01" (02 ...) around the object.
"Dim parts too" (on at first) grows the object down to just above the
background (autocrop's `grow`); off for objects that touch other bright
structures.
0.16.0: "Fit all regions" fits each region to the object inside it;
"Add seeds" lets one click on objects in the 2D views, and "Fit to seeds"
makes one region per seed (the object under it, followed through time);
"Outline (silhouettes)" also gives a fitted region the object's outline in
XY, XZ and YZ (autocrop.silhouettes). "Undo fit" undoes the last fit:
ranges and polygons as they were, new regions removed.
0.17.0: "Moving region (follows the object)", on at first for time-lapses:
a fit makes a region that moves with the object (autocrop.moving_region:
one size, its place at each timepoint), cropped into one time-lapse with
the object in place. With Outline (silhouettes) too, the region gets the
object's own silhouettes at each timepoint (outlines per timepoint;
timepoints where it was missed keep the whole region), with a warning
that this is slower on large data.
The region is an ordinary crop region afterwards: it can be adjusted,
outlined, saved and cropped like any other.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                            QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
                            QHeaderView, QProgressDialog,
                            QPushButton, QSlider, QTableWidget,
                            QTableWidgetItem, QVBoxLayout)

from volume_cropper import autocrop as ac
from volume_cropper import levels as lv
from volume_cropper import ortho_views as ov
from volume_cropper import core as vc
# (0.23.0) plain-text labels and message boxes (see qt_text)
from volume_cropper.qt_text import QLabel, QMessageBox, literal

OWN_STEPS = 1000        # (0.18.0) the own threshold slider's steps


class ObjectFinder(QGroupBox):
    """The group in the panel (crop_gui.CropWidget passes itself)."""

    def __init__(self, panel):
        # (0.16.0: inside the panel's part "Autocropper
        # (experimental)", which gives the name)
        super().__init__("Use it")
        self.panel = panel
        self._searching = False
        self.setCheckable(True)
        self.setChecked(False)          # off at first: controls greyed out
        self.toggled.connect(self._group_toggled)
        self.setToolTip(
            "Experimental (0.14.0): fit a crop region to one bright object, "
            "e.g. a single embryo, in 3D and over time. Tick to use it.")
        # What Undo fit undoes: for each region of the last fit,
        # (("changed", region, its ranges, its outlines) or ("added",
        # region), the region as it was right after the fit)
        self._undo = None
        self.seeds: list[dict] = []     # {axis: value}, see OrthoViews.seeds
        # (0.17.0) per seed, what Fit to seeds gave: None (not searched
        # yet), ("region", box), ("same", box: the region of another seed
        # on the same object), ("none", None) or ("error", None)
        self.seed_results: list = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Channel"))
        self.channel_combo = QComboBox()
        self.channel_combo.setToolTip("The channel the object is bright in.")
        row.addWidget(self.channel_combo, 1)
        row.addWidget(QLabel("Threshold"))
        self.method_combo = QComboBox()
        for method in ac.METHODS:
            self.method_combo.addItem(ac.METHOD_NAMES[method], method)
        self.method_combo.setToolTip(
            "How the threshold between object and background is chosen, at "
            "each timepoint, as in Fiji's Auto Threshold: Otsu (default), "
            "Triangle (a small object on a large dark background), Li.")
        row.addWidget(self.method_combo)
        layout.addLayout(row)
        # (0.18.0, the user's wish: brighter parts of an embryo apart from
        # fainter ones) one's own threshold, for all three fits, with a
        # dotted outline of what lies above it in the 2D views and a line
        # in the Levels histogram (when it shows this channel)
        row = QHBoxLayout()
        self.own_check = QCheckBox("Own threshold")
        self.own_check.setToolTip(
            "Off (default): each search chooses its threshold itself (the "
            "method above; with a seed, the seed's own).\nOn: Fit, Fit all "
            "regions and Fit to seeds all use this value, at every "
            "timepoint: e.g. high for the bright part of an object, lower "
            "for all of it. It starts at the automatic value. The 2D views "
            "outline what lies above it on the slice shown (dotted orange; "
            "with a projection, anywhere in it), from the coarse level the "
            "search reads, smoothed as the search sees it. Dim parts too "
            "is off meanwhile (growing would undo the threshold).")
        self.own_check.toggled.connect(self._own_toggled)
        row.addWidget(self.own_check)
        self.own_slider = QSlider(Qt.Orientation.Horizontal)
        self.own_slider.setRange(0, OWN_STEPS)
        self.own_slider.setToolTip(
            "The threshold, from the darkest values (bar the darkest 0.1 %) "
            "to the brightest of the channel at the timepoint shown, as the "
            "search sees them (smoothed). The box takes any value.")
        self.own_slider.valueChanged.connect(self._own_slid)
        row.addWidget(self.own_slider, 1)
        self.own_spin = QDoubleSpinBox()
        self.own_spin.setKeyboardTracking(False)      # (applies on Enter)
        self.own_spin.setToolTip("The threshold, in the image's intensity "
                                 "values (as the Levels histogram).")
        self.own_spin.valueChanged.connect(self._own_typed)
        row.addWidget(self.own_spin)
        self.own_auto_button = QPushButton("Auto")
        self.own_auto_button.setToolTip(
            "Put the value back to the automatic threshold (the method "
            "above) of the loaded part at the timepoint shown.")
        self.own_auto_button.clicked.connect(self._own_to_auto)
        row.addWidget(self.own_auto_button)
        layout.addLayout(row)
        self.own_note = QLabel("")
        self.own_note.setWordWrap(True)
        self.own_note.setVisible(False)
        layout.addWidget(self.own_note)
        self._volume: ac.SearchVolume | None = None   # what the outline uses
        self._volume_key = None
        self._own_range = (0.0, 1.0)     # the slider's ends (values)
        self._grow_before = True         # Dim parts too, before Own was on
        self._outline_timer = QTimer(self)    # slider moves: redraw at most
        self._outline_timer.setSingleShot(True)        # every 40 ms
        self._outline_timer.setInterval(40)
        self._outline_timer.timeout.connect(self._redraw_outline)
        self._reload_timer = QTimer(self)     # another timepoint: read it
        self._reload_timer.setSingleShot(True)         # once one stops there
        self._reload_timer.setInterval(300)
        self._reload_timer.timeout.connect(self._reload_volume)
        self.grow_check = QCheckBox("Dim parts too")
        self.grow_check.setChecked(True)
        self.grow_check.setToolTip(
            "On: the object reaches down to just above the background, so "
            "that its dim parts are in the region too (e.g. the far side "
            "when the light fades with depth). Best for an object on a dark "
            "background. Off: only what the threshold finds; use it when "
            "the object touches other bright structures (e.g. tissue), "
            "which would otherwise be taken in, and check that the region "
            "holds all of the object.")
        layout.addWidget(self.grow_check)
        row = QHBoxLayout()
        row.addWidget(QLabel("Margin"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 10000.0)
        self.margin_spin.setDecimals(1)
        self.margin_spin.setValue(10.0)
        self.margin_spin.setToolTip(
            "Space added around the object on every side (the search runs on "
            "a coarse level, so keep it at least one voxel of that level).")
        row.addWidget(self.margin_spin)
        row.addStretch()
        layout.addLayout(row)
        self.moving_check = QCheckBox("Moving region (follows the object)")
        self.moving_check.setChecked(True)
        self.moving_check.setToolTip(
            "Time-lapses: the region moves with the object, one size at all "
            "timepoints (the object's largest extent plus the margin), and "
            "is cropped into one time-lapse in which the object stays in "
            "place; where each timepoint was taken from is saved with the "
            "crop. Off: one fixed region that holds the object at all "
            "timepoints (larger if the object drifts or grows).")
        self.moving_check.toggled.connect(self._enable_moving)
        layout.addWidget(self.moving_check)
        row = QHBoxLayout()
        self.outline_check = QCheckBox("Outline")
        self.outline_check.setToolTip(
            "Also give the fitted region the object's outline (over all its "
            "timepoints, plus the margin) as polygons: voxels outside them "
            "are 0 in the crop, as with drawn polygons, and they can be "
            "edited like those. Replaces the region's polygons (Undo fit "
            "gives them back). With Moving region: silhouettes for each "
            "timepoint, of the object as it is then (slower on large data).")
        self.outline_check.toggled.connect(self._enable_moving)
        row.addWidget(self.outline_check)
        # (0.18.0, the user's wish) silhouettes, or polygons slice by slice
        self.outline_kind_combo = QComboBox()
        self.outline_kind_combo.addItem("silhouettes (XY, XZ, YZ)",
                                        "silhouettes")
        self.outline_kind_combo.addItem("per slice (XY)", "slices")
        self.outline_kind_combo.setToolTip(
            "Silhouettes: the object's outline seen in XY, XZ and YZ, each "
            "through the region's depth (three polygons, quick to edit); "
            "the crop is a block carved from three sides, which for objects "
            "touching at an angle can hold corners of the neighbour.\nPer "
            "slice: its outline on up to 40 XY slices (interpolated between "
            "them), so the crop follows the object slice by slice.\nThe "
            "parts of an object split between seeds always get outlines "
            "per slice, whose margin stops halfway to the other part (so "
            "they do not overlap). Fixed regions only: a moving region gets "
            "silhouettes at each timepoint.")
        self.outline_kind_combo.currentIndexChanged.connect(
            self._enable_moving)
        row.addWidget(self.outline_kind_combo, 1)
        layout.addLayout(row)
        # (0.17.0, the user's request) outlines per timepoint cost time
        self.timed_note = QLabel(
            "⚠ Outline for every timepoint: slower on large data. Each "
            "timepoint's outline is made during the search, and the crop "
            "shapes every timepoint on its own (also the check and the 3D "
            "view, at each timepoint shown).")
        self.timed_note.setWordWrap(True)
        self.timed_note.setStyleSheet("color: #e0a030;")
        self.timed_note.setVisible(False)
        layout.addWidget(self.timed_note)
        # (0.18.0) which threshold a seed's object is found at
        row = QHBoxLayout()
        row.addWidget(QLabel("A seed finds the"))
        self.seed_mode_combo = QComboBox()
        for mode in ac.SEED_MODES:
            self.seed_mode_combo.addItem(ac.SEED_MODE_NAMES[mode], mode)
        self.seed_mode_combo.setToolTip(
            "Which threshold Fit to seeds uses for each seed's object. The "
            "search tries thresholds from the brightness at the seed down to "
            "just above the background, and looks where the object's size "
            "hardly changes (a stable part):\nmost stable part (default, as "
            "before): where it changes least;\nbrightest stable part: the "
            "first such place from the top, e.g. a bright core;\nwhole "
            "object: the last one before the object joins too much of the "
            "image, e.g. an embryo with its fainter parts.\nNot used while "
            "Own threshold is on.")
        row.addWidget(self.seed_mode_combo, 1)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.seed_button = QPushButton("Add seeds")
        self.seed_button.setCheckable(True)
        self.seed_button.setToolTip(
            "While on: click on an object in a 2D view (XY, XZ or YZ) to put "
            "a seed (yellow cross) there, at the slice and timepoint shown; "
            "click on a seed to remove it. Dragging still pans. Then Fit to "
            "seeds: one region per seed.")
        self.seed_button.toggled.connect(self._seed_mode)
        row.addWidget(self.seed_button)
        self.fit_seeds_button = QPushButton("Fit to seeds")
        self.fit_seeds_button.setToolTip(
            "One new region per seed, in the loaded part of the image: the "
            "object under the seed (or right next to it) at the seed's "
            "timepoint, followed through time. Each seed's object gets its "
            "own threshold (also a dim one). Objects close together get a "
            "region each as long as a threshold separates them (the window "
            "says when none does: then draw a rough region around each and "
            "use Fit all regions); seeds on the same object give one "
            "region.")
        self.fit_seeds_button.clicked.connect(self.fit_to_seeds)
        row.addWidget(self.fit_seeds_button)
        self.clear_seeds_button = QPushButton("Clear seeds")
        self.clear_seeds_button.clicked.connect(self.clear_seeds)
        row.addWidget(self.clear_seeds_button)
        layout.addLayout(row)
        # (0.18.0, the user's wish: touching embryos no threshold separates,
        # as in ilastik) an object several seeds share is split between them
        # by a seeded watershed in 3D; gap lines drawn between the objects
        # are walls it cannot cross
        self.split_check = QCheckBox("Split an object between its seeds")
        self.split_check.setChecked(True)
        self.split_check.setToolTip(
            "On: when two or more seeds land on one object (objects that "
            "touch, which no threshold separates), Fit to seeds splits it "
            "between them in 3D: each seed's part grows from it until it "
            "meets another's, where the image is dimmest between them and "
            "where the object narrows; gap lines are walls in between. So "
            "put one seed on each object (two seeds on one object split "
            "it).\nOff: seeds on one object give one region, as before.")
        layout.addWidget(self.split_check)
        row = QHBoxLayout()
        self.gap_button = QPushButton("Paint gap")
        self.gap_button.setCheckable(True)
        self.gap_button.setToolTip(
            "While on: hold the mouse button and draw in a 2D view along the "
            "gap between two touching objects, on the slice shown (white "
            "dash-dot lines; dragging no longer pans, scrolling still "
            "zooms). Draw on a few slices where the gap is unclear, not on "
            "every one: between the lines the image decides. Click on a "
            "line to remove it. Used by Fit to seeds when it splits an "
            "object between its seeds.")
        self.gap_button.toggled.connect(self._gap_mode)
        row.addWidget(self.gap_button)
        self.recompute_button = QPushButton("Recompute split")
        self.recompute_button.setToolTip(
            "Split again, with the gap lines as they are now, the objects of "
            "the last Fit to seeds that a gap line crosses: their regions "
            "get their new ranges and outlines in place (Undo fit gives "
            "them back). Quick: each seed's object is not searched again. "
            "After changing the seeds, use Fit to seeds.")
        self.recompute_button.clicked.connect(self.recompute_split)
        self.recompute_button.setEnabled(False)
        row.addWidget(self.recompute_button)
        row.addWidget(QLabel("width"))
        self.gap_width_spin = QDoubleSpinBox()
        self.gap_width_spin.setRange(0.1, 1000.0)
        self.gap_width_spin.setDecimals(1)
        self.gap_width_spin.setValue(4.0)
        self.gap_width_spin.setToolTip(
            "How wide new gap lines are as walls (µm); at first two voxels "
            "of the level the search reads.")
        self.gap_width_spin.valueChanged.connect(
            lambda value: setattr(self.panel.ortho, "gap_width", value))
        row.addWidget(self.gap_width_spin)
        self.clear_gaps_button = QPushButton("Clear gap lines")
        self.clear_gaps_button.clicked.connect(self.clear_gap_lines)
        row.addWidget(self.clear_gaps_button)
        layout.addLayout(row)
        self.gap_lines: list[dict] = []       # see OrthoViews.gap_lines
        # (0.18.0) the last Fit to seeds, for Recompute split: the image,
        # the area searched, the seeds and each seed's search result
        self._last_fit: dict | None = None
        # (0.17.0, the user's request) the seeds as a table: where each is
        # and what it gave; a click goes there, Delete removes
        self.seed_table = QTableWidget(0, 6)
        self.seed_table.setHorizontalHeaderLabels(
            ["#", "t", "z", "y", "x", "Region"])
        self.seed_table.verticalHeader().setVisible(False)
        self.seed_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.seed_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.seed_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.seed_table.horizontalHeader()
        for column in range(5):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.seed_table.setToolTip(
            "The seeds: their timepoint and position (µm), and the region "
            "each gave (in its colour) after Fit to seeds. Click a row to "
            "go to that seed (its timepoint and slices).")
        self.seed_table.cellClicked.connect(self._go_to_seed)
        self.seed_table.setVisible(False)
        layout.addWidget(self.seed_table)
        self.delete_seeds_button = QPushButton("Delete selected seeds")
        self.delete_seeds_button.clicked.connect(self.delete_selected_seeds)
        self.delete_seeds_button.setVisible(False)
        layout.addWidget(self.delete_seeds_button)
        row = QHBoxLayout()
        self.fit_button = QPushButton("Fit region to object")
        self.fit_button.setToolTip(
            "With a region selected: search inside it and shrink it to the "
            "object (plus the margin; never larger than it was). With none "
            "selected: search the loaded image and add a new region around "
            "the object.")
        self.fit_button.clicked.connect(self.fit)
        row.addWidget(self.fit_button)
        self.fit_all_button = QPushButton("Fit all regions")
        self.fit_all_button.setToolTip(
            "Fit every region to the object inside it, each on its own: "
            "draw a rough region around each object first.")
        self.fit_all_button.clicked.connect(self.fit_all)
        row.addWidget(self.fit_all_button)
        self.undo_button = QPushButton("Undo fit")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo)
        row.addWidget(self.undo_button)
        layout.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self._enable_seeds()
        self._enable_moving()
        self._enable_own()

    # --- called by the panel ------------------------------------------------------

    def image_changed(self):
        """A new image (or a new choice of what is loaded)."""
        image = self.panel.image
        current = self.channel_combo.currentIndex()
        self.channel_combo.blockSignals(True)     # (the own threshold: below)
        self.channel_combo.clear()
        if image is not None and image.channel_axis is not None:
            for k, name in enumerate(image.channel_labels):
                self.channel_combo.addItem(f"{k}: {name}", k)
            visible = [k for k, d in enumerate(self.panel.ortho.display)
                       if d.get("visible")]
            index = current if 0 <= current < self.channel_combo.count() \
                else (visible[0] if visible else 0)
            self.channel_combo.setCurrentIndex(index)
        else:
            self.channel_combo.addItem("(one channel)", None)
        self.channel_combo.blockSignals(False)
        self._enable_channels()
        # (0.18.0) one's own threshold: anew for this image (its values),
        # starting at the automatic value
        self._volume = self._volume_key = None
        self._reload_timer.stop()
        if self.own_check.isChecked():
            if image is None or not self._load_volume(start=True):
                self.own_check.setChecked(False)
        self._show_own()
        if image is not None:
            unit = vc.short_unit(image.unit(ac.spatial_axes(image)[-1]))
            self.margin_spin.setSuffix(f" {'µm' if unit == 'um' else unit}")
        self._undo = None
        self.undo_button.setEnabled(False)
        self.seed_button.setChecked(False)
        self.seeds, self.seed_results = [], []
        self.panel.ortho.seed_style = self._seed_style
        self.panel.ortho.set_seeds([])
        # (0.18.0) gap lines: none; new ones two voxels of the search level
        self.gap_button.setChecked(False)
        self.gap_lines = []
        self.panel.ortho.set_gap_lines([])
        self._last_fit = None
        if image is not None:
            self.gap_width_spin.setValue(round(2 * self._search_voxel(), 1))
        self._enable_gaps()
        self._enable_seeds()
        self._enable_moving()
        self.refresh_seeds()
        self.info.setText("")

    def _group_toggled(self, on: bool):
        self._enable_channels()
        self._enable_moving()
        self._enable_own()
        self._show_own()             # (0.18.0) the outline only while on
        if not on:
            self.seed_button.setChecked(False)
            self.gap_button.setChecked(False)

    def _time_lapse(self) -> bool:
        """Whether the image has more than one timepoint."""
        image = self.panel.image
        t_axis = vc.time_axis(image) if image is not None else None
        return t_axis is not None and image.levels[0].array.shape[t_axis] > 1

    def _moving(self) -> bool:
        """Whether a fit makes a moving region (0.17.0)."""
        return self._time_lapse() and self.moving_check.isChecked()

    def _timed(self) -> bool:
        """Whether a fit gives outlines per timepoint (0.17.0): Moving
        region and Outline both on, for a time-lapse."""
        return self._moving() and self.outline_check.isChecked()

    def _enable_moving(self, *_):
        """"Moving region" only for time-lapses; the warning while it and
        Outline are both on (outlines per timepoint)."""
        on = self.isChecked()
        self.moving_check.setEnabled(on and self._time_lapse())
        self.outline_check.setEnabled(on)
        self.outline_kind_combo.setEnabled(on and self.outline_check.isChecked())
        self.timed_note.setVisible(on and self._timed())

    def _enable_channels(self):
        """The channel list is usable only while the group is ticked, and
        only with more than one channel."""
        self.channel_combo.setEnabled(self.isChecked()
                                      and self.channel_combo.count() > 1)

    def _enable_seeds(self):
        has = bool(self.seeds) and not self._searching
        self.fit_seeds_button.setEnabled(has)
        self.clear_seeds_button.setEnabled(bool(self.seeds))

    # --- one's own threshold (0.18.0) -----------------------------------------
    #
    # The value lives in own_spin; the slider spans the channel's values
    # (0.1-99.9 %) at the timepoint shown, as the search sees them
    # (autocrop.search_volume: the search level, smoothed), which also
    # gives the automatic value and the outline in the 2D views.

    def own_threshold(self) -> float | None:
        """The threshold all fits use, or None: each search's own."""
        return (float(self.own_spin.value())
                if self.own_check.isChecked() else None)

    def _own_shown(self) -> bool:
        return (self.isChecked() and self.own_check.isChecked()
                and self._volume is not None)

    def _enable_own(self):
        """(Children of the group are only enabled while it is ticked:
        Qt re-enables those not disabled on purpose.)"""
        group, own = self.isChecked(), self.own_check.isChecked()
        for widget in (self.own_slider, self.own_spin, self.own_auto_button):
            widget.setEnabled(group and own)
        self.grow_check.setEnabled(group and not own)
        self.seed_mode_combo.setEnabled(group and not own)

    def _own_toggled(self, on: bool):
        if on:
            self._note("")
            self._grow_before = self.grow_check.isChecked()
            self.grow_check.setChecked(False)       # (growing undoes it)
            if not self._load_volume(start=True):
                self.own_check.setChecked(False)    # (back: see below)
                return
        else:
            self._reload_timer.stop()
            self.grow_check.setChecked(self._grow_before)
        self._enable_own()
        self._show_own()

    def _search_area(self) -> dict:
        """The loaded part of the image (what Fit to seeds searches)."""
        image, ortho = self.panel.image, self.panel.ortho
        return {axis: tuple(ortho.extent(axis)) for axis in image.range_axes}

    def _wanted(self) -> tuple:
        """What the outline is of: image, channel, timepoint, area."""
        return (id(self.panel.image), self.channel_combo.currentData(),
                self.panel.ortho.time_shown(),
                tuple(sorted(self._search_area().items())))

    def _load_volume(self, start: bool = False) -> bool:
        """Read the channel at the timepoint shown as the search sees it
        (if not read yet). start: also fit the slider to its values and
        put the threshold at the automatic value."""
        image = self.panel.image
        if image is None:
            self._note("Open an image first.")
            return False
        wanted = self._wanted()
        if self._volume is None or self._volume_key != wanted:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._volume = ac.search_volume(
                    image, self.channel_combo.currentData(),
                    region=self._search_area(),
                    time=self.panel.ortho.time_shown())
                self._volume_key = wanted
            except Exception as err:
                self._volume = self._volume_key = None
                self._note(f"Could not read the channel: "
                           f"{type(err).__name__}: {err}")
                return False
            finally:
                QApplication.restoreOverrideCursor()
        if start:
            volume = self._volume
            low, high = self._own_range = volume.value_range()
            integer = np.issubdtype(image.levels[0].array.dtype, np.integer)
            values = volume.data[volume.valid]
            self.own_spin.blockSignals(True)
            self.own_spin.setDecimals(lv.box_decimals(high - low, integer))
            self.own_spin.setSingleStep(lv.box_step(high - low, integer))
            self.own_spin.setRange(min(low, float(values.min())),
                                   max(high, float(values.max())))
            self.own_spin.blockSignals(False)
            self._set_own_value(volume.automatic(
                self.method_combo.currentData()))
        return True

    def _note(self, text: str):
        self.own_note.setText(text)
        self.own_note.setVisible(bool(text))

    def _set_own_value(self, value: float):
        self.own_spin.blockSignals(True)
        self.own_spin.setValue(value)
        self.own_spin.blockSignals(False)
        self._place_slider(self.own_spin.value())
        self._own_changed()

    def _place_slider(self, value: float):
        low, high = self._own_range
        position = round((value - low) / max(high - low, 1e-12) * OWN_STEPS)
        self.own_slider.blockSignals(True)
        self.own_slider.setValue(min(max(position, 0), OWN_STEPS))
        self.own_slider.blockSignals(False)

    def _own_slid(self, position: int):
        low, high = self._own_range
        self.own_spin.blockSignals(True)
        self.own_spin.setValue(low + (high - low) * position / OWN_STEPS)
        self.own_spin.blockSignals(False)
        self._own_changed()

    def _own_typed(self, value: float):
        self._place_slider(value)
        self._own_changed()

    def _own_to_auto(self):
        if self._load_volume():
            self._set_own_value(self._volume.automatic(
                self.method_combo.currentData()))

    def _marker_moved(self, value: float):
        """The threshold's line in the Levels histogram was dragged."""
        if self.own_check.isChecked():
            self._set_own_value(value)

    def _own_changed(self):
        """The value changed: the outline (at most every 40 ms while the
        slider moves) and the histogram's line follow."""
        if self._own_shown() and not self._outline_timer.isActive():
            self._outline_timer.start()
        self._show_marker()

    def _redraw_outline(self):
        if self._own_shown():
            self.panel.ortho.draw_level_outline()

    def _show_own(self):
        """Show or hide the outline in the 2D views and the line in the
        histogram."""
        self.panel.ortho.set_level_outline(
            self._outline if self._own_shown() else None)
        self._show_marker()

    def _show_marker(self):
        levels = getattr(self.panel, "levels", None)
        if levels is None:
            return
        levels.on_marker_moved = self._marker_moved
        levels.set_marker((self.channel_combo.currentData() or 0)
                          if self._own_shown() else None,
                          self.own_spin.value(), ov.THRESHOLD_COLOR,
                          "Autocropper threshold")

    def _outline(self, plane, depth: str, slab: tuple) -> list:
        """OrthoViews.level_outline: the outline at the threshold on a
        slice (or through a slab). Another timepoint, channel or loaded
        part: none until it is read, once one stays there 0.3 s (so that
        playing through time is not slowed down)."""
        if self._volume is None or self._volume_key != self._wanted():
            self._reload_timer.start()
            return []
        return self._volume.contours(plane, depth, slab,
                                     float(self.own_spin.value()))

    def _reload_volume(self):
        if self.isChecked() and self.own_check.isChecked() \
                and self.panel.image is not None and self._load_volume():
            self.panel.ortho.draw_level_outline()

    def _channel_changed(self, _index=None):
        """Another channel to search: with one's own threshold, its values
        (the slider anew, at the automatic value)."""
        if self.own_check.isChecked() and self.panel.image is not None:
            if self._load_volume(start=True):
                self._show_own()

    # --- seeds ---------------------------------------------------------------------------

    def tool_changed(self, mode: str):
        """The panel's drawing tool changed: seeds are added only with the
        pan/zoom tool (a click with a drawing tool draws)."""
        if str(mode) != "pan_zoom" and self.seed_button.isChecked():
            self.seed_button.setChecked(False)
        if str(mode) != "pan_zoom" and self.gap_button.isChecked():
            self.gap_button.setChecked(False)

    def _seed_mode(self, on: bool):
        ortho = self.panel.ortho
        if on and self.panel.image is None:
            self.seed_button.setChecked(False)
            return
        if on:
            self.gap_button.setChecked(False)     # (one mode at a time)
            ortho.set_tool("pan_zoom")
            ortho.on_seed_click = self._seed_clicked
            self.info.setText("Click on each object in XY, XZ or YZ (at a "
                              "slice where it shows); click on a seed to "
                              "remove it. Then Fit to seeds.")
        else:
            ortho.on_seed_click = None

    def _seed_clicked(self, point: dict, view_name: str):
        """A click in a 2D view while seeds are added: a seed there, or the
        seed under the click removed."""
        ortho = self.panel.ortho
        k = ortho.seed_near(point, view_name)
        if k is not None:
            del self.seeds[k]
            del self.seed_results[k]
        elif ortho.projection != "none":
            # a projection hides the depth: the seed would get the slider's
            # slice, maybe far from the object clicked (review 2026-11-08)
            self.info.setText("Seeds need a slice: set 2D projection (in "
                              "View) to 'No projection', go to a slice "
                              "where the object shows, and click there.")
            return
        else:
            for axis, value in point.items():
                low, high = sorted(ortho.extent(axis))
                if not low <= value <= high:
                    return                   # outside the loaded part
            self.seeds.append(dict(point))
            self.seed_results.append(None)
        ortho.set_seeds(self.seeds)
        self._enable_seeds()
        self.refresh_seeds()
        n = len(self.seeds)
        self.info.setText(f"{n} seed{'s' if n != 1 else ''}. Click on an "
                          "object to add one, on a seed to remove it.")

    # --- gap lines (0.18.0) ----------------------------------------------------

    def _search_voxel(self) -> float:
        """A voxel of the level a search of the loaded part reads (µm, the
        finest of the spatial axes)."""
        image, ortho = self.panel.image, self.panel.ortho
        region = {axis: ortho.extent(axis) for axis in image.range_axes}
        level = ac.choose_level(image, region)
        factors = vc.downsampling_factors(image)[level]
        names = image.axis_names
        return min(factors[names.index(a)]
                   * image.levels[0].scale[names.index(a)]
                   for a in ac.spatial_axes(image))

    def _enable_gaps(self):
        self.clear_gaps_button.setEnabled(bool(self.gap_lines))
        n = len(self.gap_lines)
        self.clear_gaps_button.setText(
            f"Clear gap lines ({n})" if n else "Clear gap lines")

    def _gap_mode(self, on: bool):
        ortho = self.panel.ortho
        if on and self.panel.image is None:
            self.gap_button.setChecked(False)
            return
        if on:
            self.seed_button.setChecked(False)    # (one mode at a time)
            ortho.set_tool("pan_zoom")
            ortho.gap_width = self.gap_width_spin.value()
            ortho.on_gap_line = self._gap_drawn
            ortho.on_gap_click = self._gap_clicked
            ortho.set_gap_mode(True)
            self.info.setText("Hold the mouse button and draw along the gap "
                              "between touching objects, on a few slices "
                              "where it is unclear; click on a line to "
                              "remove it. Then Fit to seeds (one seed per "
                              "object).")
        else:
            ortho.set_gap_mode(False)
            ortho.on_gap_line = ortho.on_gap_click = None

    def _gap_drawn(self, line: dict):
        ortho = self.panel.ortho
        if ortho.projection != "none":
            self.info.setText("Gap lines need a slice: set 2D projection (in "
                              "View) to 'No projection' and draw there.")
            ortho.draw_gap_lines()
            return
        self.gap_lines.append(line)
        ortho.set_gap_lines(self.gap_lines)
        self._enable_gaps()
        n = len(self.gap_lines)
        self.info.setText(f"{n} gap line{'s' if n != 1 else ''}.")

    def _gap_clicked(self, point: dict, view_name: str):
        k = self.panel.ortho.gap_near(point, view_name)
        if k is None:
            return
        del self.gap_lines[k]
        self.panel.ortho.set_gap_lines(self.gap_lines)
        self._enable_gaps()
        self.info.setText("Gap line removed.")

    def clear_gap_lines(self):
        self.gap_lines = []
        self.panel.ortho.set_gap_lines([])
        self._enable_gaps()
        self.info.setText("Gap lines cleared.")

    def _fused_groups(self, image, results: dict, seeds: list) -> list:
        """The seeds (numbers) whose searches found one object: linked when
        their objects are the same, one's object holds the other seed, or
        (0.18.0, the user's wish) their objects overlap. Groups of two or
        more, each sorted."""
        found = [n for n in results if results[n].ranges]
        group_of = {n: {n} for n in found}
        for i, m in enumerate(found):
            for n in found[i + 1:]:
                if group_of[m] is group_of[n]:
                    continue
                if (_same_object(image, results[m], results[n],
                                 seeds[m - 1], seeds[n - 1])
                        or ac.mask_holds(image, results[m], seeds[n - 1])
                        or ac.mask_holds(image, results[n], seeds[m - 1])
                        or _overlapping(results[m], results[n])):
                    merged = group_of[m] | group_of[n]
                    for k in merged:
                        group_of[k] = merged
        groups = []
        for n in found:
            if len(group_of[n]) > 1 and group_of[n] not in groups:
                groups.append(group_of[n])
        return [sorted(g) for g in groups]

    def _split_groups(self, image, region, results: dict, seeds: list,
                      dialog, lines: list, groups: list | None = None):
        """Split each object several seeds share between them (0.18.0),
        replacing their results; notes in lines."""
        own = self.own_threshold()
        split = []
        if groups is None:
            groups = self._fused_groups(image, results, seeds)
        for group in groups:
            names = ", ".join(str(n) for n in group)
            gaps = len(self.gap_lines)

            def progress(done, total, names=names):
                where = (f", timepoint {min(int(done) + 1, total)} of "
                         f"{total}" if total > 1 else "")
                self._show_step(dialog, f"Splitting the object of seeds "
                                f"{names}{where}", 0.95)

            parts = ac.split_objects(
                image, self.channel_combo.currentData(),
                [results[n] for n in group], [seeds[n - 1] for n in group],
                self.gap_lines, method=self.method_combo.currentData(),
                grow=self.grow_check.isChecked() and own is None,
                threshold=own,
                timed_silhouettes=(self._moving_in(region)
                                   and self.outline_check.isChecked()),
                progress=progress, cancel=dialog.wasCanceled)
            for n, part in zip(group, parts):
                results[n] = part
            split.append(group)
            lines.append(f"Seeds {names}: one object, split between them"
                         + (f" (gap lines: {gaps})" if gaps else "") + ".")
        return split

    def clear_seeds(self):
        self.seeds, self.seed_results = [], []
        self.panel.ortho.set_seeds([])
        self._enable_seeds()
        self.refresh_seeds()
        self.info.setText("Seeds cleared.")

    def delete_selected_seeds(self):
        """(0.17.0) Remove the seeds selected in the table."""
        rows = sorted({index.row() for index in
                       self.seed_table.selectionModel().selectedRows()},
                      reverse=True)
        for k in rows:
            if 0 <= k < len(self.seeds):
                del self.seeds[k]
                del self.seed_results[k]
        self.panel.ortho.set_seeds(self.seeds)
        self._enable_seeds()
        self.refresh_seeds()

    # --- the seeds' colours, places over time and table (0.17.0) -----------------

    def _seed_box(self, k: int):
        """The region seed k gave (or the one of its object), if it still
        exists; else None."""
        result = self.seed_results[k] if k < len(self.seed_results) else None
        box = result[1] if result else None
        if box is not None and any(b is box for b in self.panel.boxes):
            return box
        return None

    def _seed_color(self, k: int) -> str:
        """Seed k's colour: its region's, or white (no region)."""
        box = self._seed_box(k)
        if box is None:
            return ov.SEED_COLOR
        i = next(i for i, b in enumerate(self.panel.boxes) if b is box)
        return ov.BOX_COLORS[i % len(ov.BOX_COLORS)]

    def _seed_style(self, k: int, seed: dict, time) -> tuple:
        """Where and how seed k is drawn at a time (OrthoViews.seed_style),
        the user's choice "follow its object": with a region that moves,
        the cross moves with it (its shift since the seed's timepoint), full
        where the object was followed and faded where it was not; else
        where it was clicked, full at its own timepoint, faded at others."""
        color = self._seed_color(k)
        image = self.panel.image
        t_axis = vc.time_axis(image) if image is not None else None
        name = image.axis_names[t_axis] if t_axis is not None else None
        own_time = seed.get(name) if name else None
        if time is None or own_time is None:
            return seed, ov.rgba(color)
        full = vc.same_time(time, own_time)
        point = dict(seed)
        box = self._seed_box(k)
        if box is not None and box.track:
            now = vc.track_shift(box.track, time)
            then = vc.track_shift(box.track, own_time)
            for axis, shift in now.items():
                if axis in point:
                    point[axis] = seed[axis] + shift - then.get(axis, 0.0)
            full = full or any(vc.same_time(t, time) for t in box.track)
        return point, ov.rgba(color, 1.0 if full else ov.FADED_ALPHA)

    def refresh_seeds(self):
        """Redraw the seeds (colours may have changed) and refill the
        table; called by the panel whenever its regions change."""
        ortho = self.panel.ortho
        if ortho.seeds:
            ortho._draw_seeds()
        image = self.panel.image
        table = self.seed_table
        table.setRowCount(len(self.seeds))
        table.setVisible(bool(self.seeds))
        self.delete_seeds_button.setVisible(bool(self.seeds))
        if not self.seeds or image is None:
            return
        t_axis = vc.time_axis(image)
        level0 = image.levels[0]
        space = ac.spatial_axes(image)[:3]
        unit = vc.short_unit(image.unit(space[-1])).replace("um", "µm")
        table.setHorizontalHeaderLabels(
            ["#", "t"] + [f"{a} ({unit})" for a in space]
            + [""] * (3 - len(space)) + ["Region"])
        for k, seed in enumerate(self.seeds):
            color = QColor(self._seed_color(k))
            cells = [str(k + 1)]
            if t_axis is not None:
                t_name = image.axis_names[t_axis]
                n = round((seed.get(t_name, 0.0) - level0.translation[t_axis])
                          / level0.scale[t_axis]) + 1
                cells.append(str(n))
            else:
                cells.append("")
            cells += [f"{seed[a]:.1f}" if a in seed else "" for a in space]
            cells += [""] * (5 - len(cells))
            result = self.seed_results[k] if k < len(self.seed_results) \
                else None
            box = self._seed_box(k)
            if result is None or (result[0] in ("region", "same")
                                  and box is None):
                region = "—"
            elif result[0] == "region":
                region = box.name
            elif result[0] == "same":
                region = f"{box.name} (same object)"
            elif result[0] == "none":
                region = "no object"
            else:
                region = "error"
            cells.append(region)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (0, 5) and box is not None:
                    item.setForeground(color)
                table.setItem(k, column, item)
        rows = min(len(self.seeds), 5)
        table.setFixedHeight(table.horizontalHeader().height()
                             + rows * table.verticalHeader()
                             .defaultSectionSize() + 2 * table.frameWidth())

    def _go_to_seed(self, row: int, column: int = 0):
        """(0.17.0) A click on a seed's row: the views go to its timepoint
        and slices."""
        if 0 <= row < len(self.seeds):
            self.panel.ortho._set_cursor(dict(self.seeds[row]))

    # --- the buttons -------------------------------------------------------------------

    def _region_of(self, box) -> dict | None:
        """A region's own ranges within the image (all of it, also where it
        reaches beyond the part loaded for the views: the search reads the
        image itself); None if it lies outside the image."""
        image = self.panel.image
        # (a moving region: all its places over time, 0.17.0)
        ranges = vc.fixed_in_place(image, box).ranges
        t_axis = vc.time_axis(image)
        t_name = None if t_axis is None else image.axis_names[t_axis]
        region = {}
        for axis in image.range_axes:
            full = image.extent(axis)
            low, high = sorted(ranges.get(axis, full))
            region[axis] = (max(low, full[0]), min(high, full[1]))
            # (one timepoint is a time range of no length, e.g. 20-20 min,
            # as a fit over one timepoint gives it: the 0.17.0 review found
            # such a region "outside the image" at the next fit)
            empty = region[axis][0] > region[axis][1] or (
                region[axis][0] == region[axis][1] and axis != t_name)
            if empty:
                return None
        return region

    def _moving_in(self, region) -> bool:
        """Whether a fit in this region makes a moving region: Moving
        region ticked, and the region spans more than one timepoint (over
        one, the fit is a fixed region, with its outline if Outline is
        ticked: the 0.17.0 review found it got none)."""
        if not self._moving():
            return False
        image = self.panel.image
        i = vc.time_axis(image)
        name, level0 = image.axis_names[i], image.levels[0]
        low, high = sorted((region or {}).get(name, image.extent(name)))
        first, end = vc.physical_to_index_range(
            low, high, level0.scale[i], level0.translation[i])
        return min(end, level0.array.shape[i]) - max(first, 0) > 1

    def _begin(self) -> bool:
        if self._searching:              # (a click queued during a search)
            return False
        self._searching = True
        for button in (self.fit_button, self.fit_all_button,
                       self.undo_button, self.fit_seeds_button,
                       self.recompute_button):
            button.setEnabled(False)
        return True

    def _end(self):
        self._searching = False
        self.fit_button.setEnabled(True)
        self.fit_all_button.setEnabled(True)
        self.undo_button.setEnabled(bool(self._undo))
        self.recompute_button.setEnabled(self._last_fit is not None)
        self._enable_seeds()

    def _fitted(self, box, result, region, image, others=None) -> str:
        """Give a region the fit's ranges (within the region searched) and,
        with Outline ticked, the object's silhouettes; with Moving region
        ticked (0.17.0), a region that moves with the object (and no
        polygons). Returns a note for the status."""
        track = {}
        if self._moving_in(region):
            ranges, track = ac.moving_region(image, result, within=region)
        if track:
            box.ranges.update(ranges)
        else:
            box.ranges.update(_within(result.ranges, region))
        box.track = track
        box.timed_outlines = {}
        note = (" (reaches the region's edge: may be cut)"
                if result.touches else "")
        if result.image_edges:
            note += " (reaches the image's edge)"
        if track:
            note += (f", moving with the object ({len(track)} timepoints "
                     "found)")
            if box.outlines:
                box.outlines = {}
                note += (" (its own polygons removed: a moving region cannot "
                         "have polygons that stay in place; Undo fit gives "
                         "them back)")
            if self.outline_check.isChecked():
                # each through the region's depth as cropped then (what a
                # "selected region" projection shows: editable there)
                try:
                    plan = self.panel._plan(box)
                    at = (lambda time: vc.snapped_ranges_at(image, plan, time))
                except ValueError:
                    at = None
                box.timed_outlines = ac.timed_silhouettes(image, result, box,
                                                          at)
                n = len(box.timed_outlines)
                note += (f", with its outline at each of {n} timepoints"
                         if n else " (no outline could be made)")
        elif self.outline_check.isChecked():
            # (0.18.0) per slice if chosen, and always for the parts of a
            # split (others: the other parts' mask), whose silhouettes
            # would overlap
            by_slice = (others is not None
                        or self.outline_kind_combo.currentData() == "slices")
            outlines = (ac.slice_outlines(image, result, box.ranges,
                                          others=others) if by_slice
                        else ac.silhouettes(image, result, box.ranges))
            if outlines:
                box.outlines = vc.normalize_outlines(outlines)
                note += (", with its outline"
                         + (" per slice" if by_slice else ""))
            else:
                note += " (no outline could be made)"
        return note

    @staticmethod
    def _changed(box) -> tuple:
        """An undo entry: the region as it is before a fit."""
        return ("changed", box, copy.deepcopy(box.ranges),
                copy.deepcopy(box.outlines), copy.deepcopy(box.track),
                copy.deepcopy(box.timed_outlines))

    def _arm_undo(self, entries: list):
        """Keep the fit's undo entries, each with its region as it is now,
        right after the fit: Undo fit leaves regions changed since alone
        (renamed, redrawn, other ranges: the review of 2026-11-08 found it
        deleting regions the user had worked on since)."""
        self._undo = [(entry, _state(entry[1])) for entry in entries]

    def fit(self):
        """Fit the selected region to the object inside it, or, with none
        selected, add a region around the object in the loaded part."""
        panel = self.panel
        image = panel.image
        if image is None or not self._begin():
            return
        try:
            i = panel._selected()
            selected = panel.boxes[i] if 0 <= i < len(panel.boxes) else None
            if selected is None:
                region = {axis: panel.ortho.extent(axis)
                          for axis in image.range_axes}
            else:
                region = self._region_of(selected)
                if region is None:
                    self.info.setText(f"Region {selected.name} lies outside "
                                      "the image; nothing was changed.")
                    return
            try:
                result = self._search(image, region)
            except ac.SearchCancelled:
                self.info.setText("Search cancelled.")
                return
            except Exception as err:
                QMessageBox.critical(self, "Autocropper",
                                     f"{type(err).__name__}: {err}")
                return
            if panel.image is not image or (
                    selected is not None
                    and not any(b is selected for b in panel.boxes)):
                self.info.setText("The image or the region changed during "
                                  "the search; nothing was changed.")
                return
            text = ac.describe(result, image)
            if not result.ranges:
                self.info.setText(text)
                return
            if selected is not None:
                entries = [self._changed(selected)]
                note = self._fitted(selected, result, region, image)
                panel._refresh(select=next(
                    k for k, b in enumerate(panel.boxes) if b is selected))
                self._arm_undo(entries)
                text = f"Region {selected.name} fitted{note}. " + text
            else:
                k = panel._new_box()
                box = panel.boxes[k]
                box.name = _free_name(panel.boxes, "object")
                note = self._fitted(box, result, region, image)
                panel._refresh(select=k)
                self._arm_undo([("added", box)])
                text = f"New region {box.name}{note}. " + text
            self.info.setText(text)
        finally:
            self._end()

    def fit_all(self):
        """Fit every region to the object inside it, each on its own
        (0.16.0); one Undo fit for all."""
        panel = self.panel
        image = panel.image
        if image is None or not panel.boxes or not self._begin():
            return
        dialog = self._progress_dialog()          # (0.18.0) one for all
        try:
            undo, lines = [], []
            boxes = list(panel.boxes)
            for n, box in enumerate(boxes, start=1):
                region = self._region_of(box)
                if region is None:
                    lines.append(f"{box.name}: outside the image, unchanged.")
                    continue
                try:
                    result = self._search(
                        image, region, f"Region {box.name} ({n} of "
                                       f"{len(boxes)}): ",
                        dialog=dialog, part=(n - 1, len(boxes)))
                except ac.SearchCancelled:
                    lines.append("Cancelled; the regions fitted so far keep "
                                 "their fit.")
                    break
                except Exception as err:
                    lines.append(f"{box.name}: {type(err).__name__}: {err}")
                    continue
                if panel.image is not image:
                    lines.append("The image changed: stopped.")
                    break
                if not any(b is box for b in panel.boxes):
                    continue                      # deleted meanwhile
                if not result.ranges:
                    lines.append(f"{box.name}: no object found, unchanged.")
                    continue
                undo.append(self._changed(box))
                self._show_step(dialog, f"Region {box.name}: making its "
                                "outline", n / len(boxes))
                note = self._fitted(box, result, region, image)
                lines.append(f"{box.name}: fitted{note}.")
            panel._refresh(select=panel._selected()
                           if panel._selected() >= 0 else None)
            self._arm_undo(undo)
            self.info.setText("Fit all regions:\n" + "\n".join(lines))
        finally:
            dialog.close()
            self._end()

    def fit_to_seeds(self):
        """One new region per seed (0.16.0), in the loaded part of the
        image; seeds on the same object give one region. One Undo fit
        removes them all again."""
        panel = self.panel
        image = panel.image
        if image is None or not self.seeds or not self._begin():
            return
        dialog = self._progress_dialog()          # (0.18.0) one for all
        try:
            region = {axis: panel.ortho.extent(axis)
                      for axis in image.range_axes}
            undo, lines = [], []
            seeds = list(self.seeds)
            # 1. search for each seed's object (most of the time: the bar
            # goes to 95 % over the searches, the rest for the regions)
            results = {}                     # seed number: result
            for n, seed in enumerate(seeds, start=1):
                try:
                    result = self._search(
                        image, region, f"Seed {n} of {len(seeds)}: ",
                        seed=seed, dialog=dialog,
                        part=(n - 1, len(seeds) / 0.95))
                except ac.SearchCancelled:
                    lines.append("Cancelled; regions are made for the seeds "
                                 "searched so far.")
                    break
                except Exception as err:
                    lines.append(f"Seed {n}: {type(err).__name__}: {err}")
                    self.seed_results[n - 1] = ("error", None)
                    continue
                if panel.image is not image:
                    break
                results[n] = result
            if panel.image is not image:
                self.info.setText("The image changed during the search; "
                                  "nothing was changed.")
                return
            # (0.18.0) for Recompute split: the searches, before any split
            self._last_fit = {"image": image, "region": dict(region),
                              "seeds": [dict(seed) for seed in seeds],
                              "searched": dict(results)}
            # 1b. (0.18.0) an object several seeds share: split between them
            split = []
            if self.split_check.isChecked() and len(results) > 1:
                try:
                    split = self._split_groups(image, region, results,
                                               seeds, dialog, lines)
                except ac.SearchCancelled:
                    lines.append("Cancelled while splitting: seeds on one "
                                 "object keep one region.")
                except Exception as err:
                    lines.append(f"Could not split: {type(err).__name__}: "
                                 f"{err}")
            # 2. one region per object (seeds on the same object: one)
            self._show_step(dialog, "Making the regions and their outlines",
                            0.95)
            made = []                  # (seed number, region name, region)

            def same(m, n):
                return _same_object(image, results[m], results[n],
                                    seeds[m - 1], seeds[n - 1])

            for n, result in results.items():
                if not result.ranges:
                    lines.append(f"Seed {n}: no object there (nothing "
                                 "stands out at the seed).")
                    self.seed_results[n - 1] = ("none", None)
                    continue
                same_as = next((box for m, _, box in made if same(m, n)),
                               None)
                if same_as is not None:
                    lines.append(f"Seed {n}: the same object as "
                                 f"{same_as.name}.")
                    self.seed_results[n - 1] = ("same", same_as)
                    continue
                k = panel._new_box()
                box = panel.boxes[k]
                box.name = _free_name(panel.boxes, "object")
                # (0.18.0) a part of a split: its outline stops at the others
                others = None
                for group in split:
                    if n in group:
                        masks = [results[m].mask for m in group
                                 if m != n and results[m].mask is not None]
                        others = (np.logical_or.reduce(masks) if masks
                                  else None)
                note = self._fitted(box, result, region, image, others)
                undo.append(("added", box))
                made.append((n, box.name, box))
                self.seed_results[n - 1] = ("region", box)
                missing = result.n_found < len(result.times)
                lines.append(
                    f"Seed {n}: new region {box.name}{note}"
                    + (f" (object followed at {result.n_found} of "
                       f"{len(result.times)} timepoints)" if missing else "")
                    + ".")
            # 3. an object that holds another seed whose own object is a
            # different one: the threshold does not separate them
            for n, name, _ in made:
                held = [m for m in results if m != n
                        and results[m].ranges and not same(m, n)
                        and ac.mask_holds(image, results[n], seeds[m - 1])]
                if held:
                    lines.append(
                        f"Note: {name} also holds seed "
                        + ", ".join(str(m) for m in held)
                        + ": no threshold separates these objects (they "
                        "touch, or join through other bright structures). "
                        "Draw a rough region around each and use Fit all "
                        "regions instead.")
            panel._refresh(select=next(
                (k for k, b in enumerate(panel.boxes) if b is undo[-1][1]),
                None) if undo else None)
            self._arm_undo(undo)
            self.info.setText("Fit to seeds:\n" + "\n".join(lines)
                              + ("\nUndo fit removes the new regions again "
                                 "(e.g. to try another threshold)."
                                 if undo else ""))
        finally:
            dialog.close()
            self._end()

    def recompute_split(self):
        """(0.18.0, the user's wish) Split again, with the gap lines as
        they are now, the objects of the last Fit to seeds that a gap line
        crosses (on the search grid, over all timepoints); their seeds'
        regions get the new ranges and outlines in place (a seed without a
        region of its own gets a new one). One Undo fit for all. The
        searches for the seeds' objects are those of the fit."""
        panel = self.panel
        image = panel.image
        last = self._last_fit
        if image is None or last is None or last["image"] is not image:
            self.info.setText("Recompute split needs a Fit to seeds first.")
            return
        if [dict(seed) for seed in self.seeds] != last["seeds"]:
            self.info.setText("The seeds changed since Fit to seeds: use Fit "
                              "to seeds again.")
            return
        if not self._begin():
            return
        dialog = self._progress_dialog()
        try:
            searched, seeds = dict(last["searched"]), last["seeds"]
            region = last["region"]
            groups = [g for g in self._fused_groups(image, searched, seeds)
                      if self._gap_crosses(image, [searched[n] for n in g])]
            if not groups:
                self.info.setText(
                    "No gap line crosses an object that seeds share. Draw "
                    "the lines across the contact (in Paint gap), then "
                    "Recompute split.")
                return
            lines, undo = [], []
            split = self._split_groups(image, region, searched, seeds,
                                       dialog, lines, groups=groups)
            self._show_step(dialog, "Making the outlines", 0.95)
            selected = None
            for group in split:
                for n in group:
                    part = searched[n]
                    if not part.ranges:
                        lines.append(f"Seed {n}: nothing left of its part.")
                        continue
                    masks = [searched[m].mask for m in group
                             if m != n and searched[m].mask is not None]
                    others = np.logical_or.reduce(masks) if masks else None
                    kind, box = (self.seed_results[n - 1]
                                 if n - 1 < len(self.seed_results)
                                 and self.seed_results[n - 1] else
                                 (None, None))
                    if kind != "region" or not any(b is box
                                                   for b in panel.boxes):
                        k = panel._new_box()
                        box = panel.boxes[k]
                        box.name = _free_name(panel.boxes, "object")
                        undo.append(("added", box))
                        what = "new region"
                    else:
                        undo.append(self._changed(box))
                        what = "region"
                    note = self._fitted(box, part, region, image, others)
                    self.seed_results[n - 1] = ("region", box)
                    selected = box
                    lines.append(f"Seed {n}: {what} {box.name} "
                                 f"recomputed{note}.")
            panel._refresh(select=next(
                (k for k, b in enumerate(panel.boxes) if b is selected),
                None))
            self._arm_undo(undo)
            self.info.setText("Recompute split:\n" + "\n".join(lines)
                              + ("\nUndo fit gives the regions back as they "
                                 "were." if undo else ""))
        except ac.SearchCancelled:
            self.info.setText("Recompute split cancelled; nothing was "
                              "changed.")
        finally:
            dialog.close()
            self._end()

    def _gap_crosses(self, image, results: list) -> bool:
        """Whether a gap line crosses the object these results share (its
        mask over all timepoints, on their search grid)."""
        masks = [r.mask for r in results if r.mask is not None]
        if not masks or not self.gap_lines:
            return False
        first = results[0]
        union = np.logical_or.reduce(masks)
        wall, _ = ac.gap_walls(image, first.level, first.search,
                               self.gap_lines, union.shape)
        return bool((wall & union).any())

    def undo(self):
        """Undo the last fit: the regions' ranges and polygons as before,
        the regions it added removed."""
        if not self._undo:
            return
        panel = self.panel
        undone, removed, left = [], [], []
        for entry, after in self._undo:
            box = entry[1]
            k = next((k for k, b in enumerate(panel.boxes) if b is box), None)
            if k is None:                    # deleted since
                continue
            if _state(box) != after:         # changed since: left alone
                left.append(box.name)
                continue
            if entry[0] == "added":
                del panel.boxes[k]
                removed.append(box.name)
            else:
                (box.ranges, box.outlines, box.track,
                 box.timed_outlines) = entry[2:6]
                undone.append(box.name)
        self._undo = None
        self.undo_button.setEnabled(False)
        panel._refresh(select=-1 if removed else (
            panel._selected() if panel._selected() >= 0 else None))
        parts = []
        if undone:
            parts.append("Back as before the fit: " + ", ".join(undone) + ".")
        if removed:
            parts.append("Removed: " + ", ".join(removed) + ".")
        if left:
            parts.append("Left as they are (changed since the fit): "
                         + ", ".join(left) + ".")
        self.info.setText(" ".join(parts) or "Nothing to undo.")

    def _progress_dialog(self) -> QProgressDialog:
        """The Autocropper's progress window (Cancel stops the search); one
        for a whole Fit to seeds or Fit all regions (0.18.0: it was one per
        seed or region, and showed no progress within a timepoint)."""
        dialog = QProgressDialog("Looking for the object …", "Cancel", 0,
                                 1000, self)
        dialog.setWindowTitle("Autocropper")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(500)
        dialog.vc_started = time.time()
        dialog.setValue(0)
        return dialog

    @staticmethod
    def _show_step(dialog: QProgressDialog, text: str, share: float):
        """Show a step of the work (after the searches: making the
        regions and their outlines) at share (0-1) of the bar."""
        dialog.setValue(int(1000 * share))
        dialog.setLabelText(literal(
            f"{text} ({time.time() - dialog.vc_started:.0f} s)"))
        QApplication.processEvents()

    def _search(self, image, region, prefix: str = "", seed=None,
                dialog: QProgressDialog | None = None,
                part: tuple = (0, 1)):
        """autocrop.find_object with a progress window (Cancel stops it):
        its own, or `dialog` shared by several searches, of which this one
        is part (index, count). The bar moves per timepoint and, within a
        timepoint, per threshold tried (with a seed: the slow part)."""
        own = dialog is None
        if own:
            dialog = self._progress_dialog()
        index, count = part

        def progress(done, total):
            share = done / total if total else 1.0
            where = (f", timepoint {min(int(done) + 1, total)} of {total}"
                     if total > 1 else "")
            self._show_step(dialog, f"{prefix}looking for the object{where}",
                            (index + share) / count)

        try:
            return ac.find_object(
                image, self.channel_combo.currentData(), region=region,
                method=self.method_combo.currentData(),
                margin=self.margin_spin.value(),
                grow=(self.grow_check.isChecked()
                      and self.own_threshold() is None), seed=seed,
                threshold=self.own_threshold(),              # (0.18.0)
                seed_mode=self.seed_mode_combo.currentData(),
                keep_mask=seed is not None or (
                    self.outline_check.isChecked()
                    and not self._moving_in(region)),
                timed_silhouettes=(self._moving_in(region)
                                   and self.outline_check.isChecked()),
                progress=progress, cancel=dialog.wasCanceled)
        finally:
            if own:
                dialog.close()


def _overlapping(first, second) -> bool:
    """(0.18.0) Whether two seeds' objects overlap (masks on the same grid):
    by at least 10 voxels and 0.5 % of the smaller."""
    a, b = first.mask, second.mask
    if a is None or b is None or a.shape != b.shape \
            or first.level != second.level or first.search != second.search:
        return False
    both = int((a & b).sum())
    return both >= max(10, 0.005 * min(int(a.sum()), int(b.sum())))


def _same_object(image, first, second, first_seed, second_seed) -> bool:
    """Two seeds' objects are the same if their masks (over all
    timepoints, on the same search grid) overlap by more than half of the
    two together, and each holds the other's seed (or comes within
    SEED_REACH of it). Overlap alone is not enough: on the embryo demo a
    dim embryo's piece was mostly its bright neighbour, and its seed was
    reported as being on the neighbour, whose region held none of it."""
    a, b = first.mask, second.mask
    if a is None or b is None or a.shape != b.shape:
        return False
    union = np.logical_or(a, b).sum()
    if not union or np.logical_and(a, b).sum() <= 0.5 * union:
        return False
    return (ac.mask_holds(image, first, second_seed, ac.SEED_REACH)
            and ac.mask_holds(image, second, first_seed, ac.SEED_REACH))


def _state(box) -> tuple:
    """What a region is (to tell whether it changed since a fit)."""
    return (box.name, copy.deepcopy(box.ranges), copy.deepcopy(box.outlines),
            copy.deepcopy(box.channels), box.margin,
            getattr(box, "t_step", 1), copy.deepcopy(box.track),
            copy.deepcopy(box.timed_outlines))


def _within(ranges: dict, region: dict) -> dict:
    """A fit's ranges kept within the region searched (0.16.0: a fit never
    makes a region larger; before, each Fit added the margin and padding
    again, so the region grew with every click)."""
    out = {}
    for axis, (low, high) in ranges.items():
        if axis in region:
            r_low, r_high = region[axis]
            low, high = max(low, r_low), min(high, r_high)
        out[axis] = (low, high)
    return out


def _free_name(boxes, stem: str) -> str:
    """stem01, stem02 ... : the first name no other region has (as files,
    names differing only in case count as the same)."""
    n = 1
    while sum(vc.same_file_name(f"{stem}{n:02d}", box.name)
              for box in boxes) > 0:
        n += 1
    return f"{stem}{n:02d}"
