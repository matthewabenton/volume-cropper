"""The "What to load" dialog: which part of an image Volume Cropper reads.

Large images open faster when less is read, especially over a network:
reading the data, not computing, is what takes the time. This dialog
appears when an image is opened (and from "What to load…" in the panel).
It shows what the image holds (channels, timepoints, size, pyramid levels)
and lets the user choose, like Bio-Formats' "Specify range for each series"
in Fiji:

- the channels shown at first (the others are not read until shown),
- a range per axis (time, z, y, x): nothing outside it is read,
- the finest level of the 2D views, and the 3D view (off by default).

It also estimates how much opening will read. Crops are not affected: crop
regions are in µm and are always cut from the whole image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                            QDialogButtonBox, QDoubleSpinBox, QGridLayout,
                            QGroupBox, QHBoxLayout, QProgressDialog,
                            QPushButton, QSpinBox, QVBoxLayout, QWidget)

from volume_cropper import file_images as fi
from volume_cropper import ortho_views as ov
from volume_cropper import core as vc
# (0.23.0) plain-text labels; names from files escaped in the formatted
# ones (see qt_text)
from volume_cropper.qt_text import QLabel, escape, literal

# Item data of the 3D view's choices (other items: a level number).
OFF_3D, AUTO_3D = -2, -1


@dataclass
class LoadChoice:
    """What to read and show of an image."""

    channels: list[int] | None = None    # shown at first (None = all)
    ranges: dict = field(default_factory=dict)   # {axis: (low, high)};
    #                                              axes left out: whole
    level_2d: int | None = None          # finest 2D level (None = auto)
    show_3d: bool = False
    level_3d: int | None = None          # the 3D view's level (None = auto)


def image_summary(image: vc.OmeZarrImage) -> str:
    """What the image holds, in two or three lines."""
    level0 = image.levels[0]
    names = image.axis_names
    spatial = [a for a in image.range_axes if not _is_time(image, a)]
    shape = " × ".join(str(level0.array.shape[names.index(a)])
                       for a in spatial)
    size = vc.format_bytes(vc.nbytes(level0.array.shape, level0.array.dtype))
    first, last = image.levels[0].scale[-1], image.levels[-1].scale[-1]
    zipped = {"ozx": ", zipped", "zip": ", zipped"}.get(image.container, "")
    lines = [f"<b>{escape(image.path.name)}</b> "
             f"(OME-Zarr {escape(image.ome_version)}"
             f"{zipped})",
             f"{shape} voxels ({escape(', '.join(spatial))}), {size}; "
             f"{len(image.levels)} pyramid levels "
             f"({first:.3g}–{last:.3g} "
             f"{escape(_unit(image, spatial[-1]))} voxels)"]
    for axis in image.range_axes:
        if _is_time(image, axis):
            i = names.index(axis)
            lines.append(f"{level0.array.shape[i]} timepoints, "
                         f"{level0.scale[i]:g} "
                         f"{escape(_unit(image, axis))} apart")
    if image.channel_axis is not None:
        lines.append(f"{image.n_channels} channels")
    return "<br>".join(lines)


class LoadDialog(QDialog):
    """Choose what to load of an image (see the module docstring).

    choice: the starting values. canvas_sizes: the 2D views' sizes on
    screen (OrthoViews.canvas_sizes), for the estimate. box_ranges: the
    selected region as it will be cropped, for the "Selected region"
    button
    (None: no button).
    """

    def __init__(self, image: vc.OmeZarrImage, choice: LoadChoice,
                 canvas_sizes: dict, box_ranges: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self.image = image
        self.canvas_sizes = canvas_sizes
        self.box_ranges = box_ranges
        self.region: ov.LoadedRegion | None = None
        self.setWindowTitle(f"What to load – {image.path.name}")
        layout = QVBoxLayout(self)
        layout.addWidget(_label(image_summary(image)))

        self.channel_checks = []
        if image.channel_axis is not None:
            group = QGroupBox("Channels shown (the others are not read "
                              "until you show them)")
            row = QHBoxLayout(group)
            for k, label in enumerate(image.channel_labels):
                check = QCheckBox(f"{k}: {label}")
                check.setChecked(choice.channels is None
                                 or k in choice.channels)
                check.toggled.connect(self._update)
                row.addWidget(check)
                self.channel_checks.append(check)
            row.addStretch()
            layout.addWidget(group)

        group = QGroupBox("Region (nothing outside it is read)")
        grid = QGridLayout(group)
        self.range_spins, self.range_notes = {}, {}
        for row, axis in enumerate(image.range_axes):
            spins = self._range_spins(axis, choice.ranges.get(axis))
            unit = _unit(image, axis)
            if _is_time(image, axis):
                name = f"{axis} (timepoints)"
            else:
                name = f"{axis} ({unit})" if unit else axis
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(QLabel("from"), row, 1)
            grid.addWidget(spins[0], row, 2)
            grid.addWidget(QLabel("to"), row, 3)
            grid.addWidget(spins[1], row, 4)
            note = QLabel()
            grid.addWidget(note, row, 5)
            self.range_spins[axis], self.range_notes[axis] = spins, note
        row = QHBoxLayout()
        whole = QPushButton("Whole image")
        whole.clicked.connect(self._whole_image)
        row.addWidget(whole)
        if box_ranges is not None:
            box = QPushButton("Selected region")
            box.setToolTip("Load only the selected crop region (as it will "
                           "be cropped).")
            box.clicked.connect(self._selected_box)
            row.addWidget(box)
        row.addStretch()
        grid.addLayout(row, len(image.range_axes), 0, 1, 6)
        layout.addWidget(group)

        group = QGroupBox("Views")
        grid = QGridLayout(group)
        self.level_combo = QComboBox()
        self.level_combo.addItem("Auto (from zoom)", None)
        for k, level in enumerate(image.levels):
            self.level_combo.addItem(
                f"level {k} ({level.scale[-1]:.3g} µm) and coarser", k)
        self.level_combo.setCurrentIndex(
            0 if choice.level_2d is None else choice.level_2d + 1)
        self.level_combo.setToolTip(
            "Auto: napari reads the level that suits the zoom. A level here "
            "is the finest it may read: less detail when zoomed in, but "
            "less to read.")
        self.level_combo.currentIndexChanged.connect(self._update)
        grid.addWidget(QLabel("2D views"), 0, 0)
        grid.addWidget(self.level_combo, 0, 1)
        self.view_3d_combo = QComboBox()
        self.view_3d_combo.setToolTip(
            "The 3D view reads one whole volume (all of the region, at one "
            "level). Off opens faster; you can switch it on later.")
        self.view_3d_combo.currentIndexChanged.connect(self._update)
        grid.addWidget(QLabel("3D view"), 1, 0)
        grid.addWidget(self.view_3d_combo, 1, 1)
        layout.addWidget(group)
        self._fill_3d_combo(OFF_3D if not choice.show_3d else
                            AUTO_3D if choice.level_3d is None
                            else choice.level_3d)

        self.estimate = _label("")
        self.estimate.setToolTip(
            "Data that opening reads for the views fitted to the window, "
            "before compression (the files are smaller). Zooming in or "
            "moving the cursor reads more. Over a slow network every "
            "megabyte counts.")
        layout.addWidget(self.estimate)
        layout.addWidget(_label(
            "Crops are always cut from the whole image: a crop region can "
            "reach beyond what is loaded."))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()

    # --- building -------------------------------------------------------------

    def _range_spins(self, axis: str, chosen) -> list:
        """Two number fields (from, to): timepoint numbers for time (1, 2,
        3 ..., like Fiji's frames; 0.11.0, before 0-based), the axis's unit
        (µm) otherwise."""
        image = self.image
        i = image.axis_names.index(axis)
        level0 = image.levels[0]
        low, high = chosen if chosen is not None else image.extent(axis)
        if _is_time(image, axis):
            n = level0.array.shape[i]
            first, end = vc.physical_to_index_range(
                low, high, level0.scale[i], level0.translation[i])
            values = (max(first, 0) + 1, min(end, n))      # 1-based
            spins = [QSpinBox(), QSpinBox()]
            for spin, value in zip(spins, values):
                spin.setRange(1, n)
                spin.setValue(value)
        else:
            e_low, e_high = image.extent(axis)
            # Arrow keys move by a round step of about 1 % of the image.
            step = 10.0 ** math.floor(math.log10((e_high - e_low) / 100))
            spins = [QDoubleSpinBox(), QDoubleSpinBox()]
            for spin, value in zip(spins, (low, high)):
                spin.setDecimals(max(1, -math.floor(math.log10(step))))
                spin.setRange(e_low, e_high)
                spin.setSingleStep(step)
                spin.setValue(value)
        for spin in spins:
            spin.valueChanged.connect(self._update)
        return spins

    def _fill_3d_combo(self, selected: int):
        """The 3D view's choices, with their sizes for the region and the
        channels chosen now."""
        combo = self.view_3d_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Off", OFF_3D)
        region, n = self.region, self._n_channels()
        if region is None:                     # before the first _update
            region = ov.loaded_region(self.image)
        auto = ov.auto_3d_level(self.image, region=region, n_channels=n)
        size = vc.format_bytes(ov.volume_bytes(self.image, auto, region, n))
        combo.addItem(f"Auto (level {auto}, {size})", AUTO_3D)
        for k in range(region.n_levels):
            voxel = self.image.levels[k].scale[-1]
            size = vc.format_bytes(ov.volume_bytes(self.image, k, region, n))
            combo.addItem(f"level {k} ({voxel:.3g} µm, {size})", k)
        index = combo.findData(selected)
        combo.setCurrentIndex(index if index >= 0 else 1)
        combo.blockSignals(False)

    # --- buttons ----------------------------------------------------------------

    def _set_ranges(self, ranges: dict):
        for axis, (low, high) in ranges.items():
            spins = self.range_spins.get(axis)
            if spins is None:
                continue
            if _is_time(self.image, axis):
                i = self.image.axis_names.index(axis)
                level0 = self.image.levels[0]
                first, end = vc.physical_to_index_range(
                    low, high, level0.scale[i], level0.translation[i])
                low, high = first + 1, end              # 1-based numbers
            for spin, value in zip(spins, (low, high)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._update()

    def _whole_image(self):
        self._set_ranges({a: self.image.extent(a)
                          for a in self.image.range_axes})

    def _selected_box(self):
        self._set_ranges(self.box_ranges)

    # --- reading the choices ------------------------------------------------------

    def _n_channels(self) -> int:
        if not self.channel_checks:
            return 1
        return sum(check.isChecked() for check in self.channel_checks)

    def ranges(self) -> dict:
        """The chosen ranges, {axis: (low, high)} in the axes' units."""
        ranges = {}
        level0 = self.image.levels[0]
        for axis, (start, stop) in self.range_spins.items():
            if _is_time(self.image, axis):     # numbers 1, 2, 3 ...
                i = self.image.axis_names.index(axis)
                ranges[axis] = vc.index_range_to_physical(
                    start.value() - 1, stop.value(), level0.scale[i],
                    level0.translation[i])
            else:
                ranges[axis] = (start.value(), stop.value())
        return ranges

    def choice(self) -> LoadChoice:
        channels = None
        if self.channel_checks and self._n_channels() < len(self.channel_checks):
            channels = [k for k, check in enumerate(self.channel_checks)
                        if check.isChecked()]
        view_3d = self.view_3d_combo.currentData()
        return LoadChoice(
            channels=channels, ranges=self.ranges(),
            level_2d=self.level_combo.currentData(),
            show_3d=view_3d != OFF_3D,
            level_3d=view_3d if view_3d >= 0 else None)

    def _update(self, *_):
        """Check the choices; show what will be loaded and the estimate."""
        problem = ""
        ranges = self.ranges()
        if any(low >= high for low, high in ranges.values()):
            problem = "Each range must start before it ends."
        elif self.channel_checks and self._n_channels() == 0:
            problem = "Tick at least one channel."
        else:
            try:
                self.region = ov.loaded_region(self.image, ranges)
            except ValueError as err:
                problem = str(err)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok) \
            if hasattr(self, "buttons") else None
        if ok is not None:
            ok.setEnabled(not problem)
        if problem:
            self.estimate.setText(f"<b>Problem:</b> {escape(problem)}")
            return
        self._show_region()
        self._fill_3d_combo(self.view_3d_combo.currentData())
        choice = self.choice()
        n = ov.open_read_bytes(
            self.image, self.region, self._n_channels(), self.canvas_sizes,
            level_2d=choice.level_2d, show_3d=choice.show_3d,
            level_3d=choice.level_3d)
        self.estimate.setText(
            f"<b>Opening reads about {vc.format_bytes(n)}</b> (before "
            "compression).")

    def _show_region(self):
        """Beside each range: what will be loaded (rounded outward to whole
        voxels of the coarsest level used) and how many voxels that is."""
        region, names = self.region, self.image.axis_names
        for axis, note in self.range_notes.items():
            i = names.index(axis)
            n = region.stops[0][i] - region.starts[0][i]
            if _is_time(self.image, axis):
                note.setText(f"{n} timepoint{'s' * (n != 1)}")
                continue
            low, high = region.ranges[axis]
            what = "slices" if axis == "z" else "voxels"
            note.setText(f"loads {low:.1f}–{high:.1f}: {n} {what}")


# Units offered for the calibration of a file (OME names, shown short).
SPACE_UNITS = ["micrometer", "nanometer", "millimeter"]
TIME_UNITS = ["second", "millisecond", "minute", "hour"]
# What a "?" axis (not stated in the file) can be.
AXIS_MEANINGS = [("Z", "z (slices)"), ("T", "t (timepoints)"),
                 ("C", "c (channels)")]


def _scene_text(k: int, scene: fi.Scene) -> str:
    """An image's entry in the list of a file's images."""
    shape = " × ".join(str(n) for n in scene.shape)
    return (f"{k + 1}: {scene.name} ({shape}, "
            f"{vc.format_bytes(scene.nbytes)})")


class FileLoadDialog(QDialog):
    """"What to load" for an image file (TIFF ...; 0.12.0), read into
    memory like an image opened in Fiji without "virtual stack".

    It shows what the file holds and lets the user choose: which of its
    images (a LIF's series, an OME-TIFF's images), what an axis is when the
    file does not say (a plain TIFF stack: z, t or c), the calibration
    (voxel size and frame interval, filled in from the file; like Fiji's
    Image > Properties), the channels and the range of each axis to read
    (numbers 1, 2, 3 ... as in Fiji), and the 3D view. It says how much
    memory that takes and refuses what would not fit.
    """

    def __init__(self, reader, choice: fi.FileChoice, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.setWindowTitle(f"What to load – {reader.path.name}")
        layout = QVBoxLayout(self)
        compressed = ("" if reader.compression in ("none", None)
                      else f", compressed ({reader.compression})")
        layout.addWidget(_label(
            f"<b>{escape(reader.path.name)}</b> "
            f"({escape(reader.format_name + compressed)})"))
        self.scene_combo = QComboBox()
        for k, scene in enumerate(reader.scenes):
            self.scene_combo.addItem(_scene_text(k, scene), k)
        self.scene_combo.setCurrentIndex(
            min(choice.scene, len(reader.scenes) - 1))
        if len(reader.scenes) > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Image in the file"))
            row.addWidget(self.scene_combo, 1)
            layout.addLayout(row)
        self.scene_combo.currentIndexChanged.connect(
            lambda _: self._build_scene(None))
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.body)
        self.memory = _label("")
        layout.addWidget(self.memory)
        layout.addWidget(_label(
            "What is chosen here is read into memory. Crops are cut from "
            "it and saved as TIFF, so they can only hold what was read."))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Load")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._build_scene(choice)

    # --- building --------------------------------------------------------------

    def _scene(self) -> fi.Scene:
        return self.reader.scenes[self.scene_combo.currentData()]

    def _prepare(self, k: int):
        """Finish an image that is worked out only once chosen: a tile scan
        is stitched now (its overlaps are read: a few seconds), and its
        entry shows its real size."""
        text = fi.preparing_text(self.reader, k)
        busy = None
        if text:                    # e.g. "Stitching 8 tiles ..."
            busy = QProgressDialog(literal(text), None, 0, 0, self)
            busy.setWindowTitle("What to load")
            busy.setMinimumDuration(0)
            busy.show()
            QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            changed = fi.prepare_scene(self.reader, k)
        finally:
            QApplication.restoreOverrideCursor()
            if busy is not None:
                busy.close()
        if changed:
            self.scene_combo.setItemText(
                k, _scene_text(k, self.reader.scenes[k]))

    def _build_scene(self, choice: fi.FileChoice | None):
        """The parts that depend on the image chosen in the file."""
        _clear(self.body_layout)
        self._prepare(self.scene_combo.currentData())
        scene = self._scene()
        if choice is None or choice.scene != self.scene_combo.currentData():
            choice = fi.FileChoice(scene=self.scene_combo.currentData())
        self._start = choice
        if scene.notes:
            self.body_layout.addWidget(_label("<i>"
                                              + escape(" ".join(scene.notes))
                                              + "</i>"))
        # What each axis is, where the file does not say.
        self.axis_combos = {}
        axes = choice.axes or fi.default_axes(scene)
        if "?" in scene.axes:
            group = QGroupBox("Axes the file does not name")
            grid = QGridLayout(group)
            for row, i in enumerate(i for i, a in enumerate(scene.axes)
                                    if a == "?"):
                combo = QComboBox()
                for letter, text in AXIS_MEANINGS:
                    combo.addItem(text, letter)
                combo.setCurrentIndex(max(0, combo.findData(axes[i])))
                combo.currentIndexChanged.connect(self._axes_changed)
                grid.addWidget(QLabel(f"Axis {i + 1} ({scene.shape[i]})"),
                               row, 0)
                grid.addWidget(combo, row, 1)
                self.axis_combos[i] = combo
            self.body_layout.addWidget(group)
        # The calibration.
        group = QGroupBox("Calibration (Fiji: Image > Properties)")
        grid = QGridLayout(group)
        self.voxel_spins = {}
        for col, k in enumerate("zyx"):
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(1e-6, 1e9)
            value = (choice.voxel or {}).get(k) or scene.voxel.get(k)
            spin.setValue(value if value else 1.0)
            spin.valueChanged.connect(self._update)
            grid.addWidget(QLabel(f"voxel {k}"), 0, 2 * col)
            grid.addWidget(spin, 0, 2 * col + 1)
            self.voxel_spins[k] = spin
        self.unit_combo = QComboBox()
        for unit in SPACE_UNITS:
            self.unit_combo.addItem(_unit_text(unit), unit)
        self.unit_combo.setCurrentIndex(max(0, self.unit_combo.findData(
            choice.unit or scene.unit or "micrometer")))
        self.unit_combo.currentIndexChanged.connect(self._update)
        grid.addWidget(self.unit_combo, 0, 6)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setDecimals(6)
        self.interval_spin.setRange(1e-9, 1e9)
        self.interval_spin.setValue(choice.interval or scene.interval or 1.0)
        self.interval_spin.valueChanged.connect(self._update)
        self.time_combo = QComboBox()
        for unit in TIME_UNITS:
            self.time_combo.addItem(_unit_text(unit), unit)
        self.time_combo.setCurrentIndex(max(0, self.time_combo.findData(
            choice.time_unit or scene.time_unit or "second")))
        self.time_combo.currentIndexChanged.connect(self._update)
        self.interval_label = QLabel("frame interval")
        grid.addWidget(self.interval_label, 1, 0)
        grid.addWidget(self.interval_spin, 1, 1)
        grid.addWidget(self.time_combo, 1, 2)
        self.calibration_note = _label("")
        grid.addWidget(self.calibration_note, 2, 0, 1, 7)
        self.body_layout.addWidget(group)
        # Channels, ranges and 3D: they depend on what the axes are.
        self.dependent = QWidget()
        self.dependent_layout = QVBoxLayout(self.dependent)
        self.dependent_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.addWidget(self.dependent)
        self._build_dependent(choice)

    def _axes_changed(self, _=None):
        self._build_dependent(None)

    def _axes(self) -> str:
        scene = self._scene()
        letters = list(fi.default_axes(scene))
        for i, combo in self.axis_combos.items():
            letters[i] = combo.currentData()
        return "".join(letters)

    def _build_dependent(self, choice: fi.FileChoice | None):
        _clear(self.dependent_layout)
        scene = self._scene()
        axes = self._axes()
        sizes = {a: n for a, n in zip(axes, scene.shape) if a != "-"}
        start = choice or fi.FileChoice(scene=self.scene_combo.currentData())
        self.channel_checks = []
        if "C" in sizes:
            group = QGroupBox("Channels read")
            row = QHBoxLayout(group)
            for c in range(sizes["C"]):
                name = (scene.channel_names[c]
                        if c < len(scene.channel_names)
                        and scene.channel_names[c] else f"channel {c}")
                check = QCheckBox(f"{c}: {name}")
                check.setChecked(start.channels is None or c in start.channels)
                check.toggled.connect(self._update)
                row.addWidget(check)
                self.channel_checks.append(check)
            row.addStretch()
            self.dependent_layout.addWidget(group)
        group = QGroupBox("Range read (numbers 1, 2, 3 …, as in Fiji)")
        grid = QGridLayout(group)
        self.range_spins = {}
        names = {"T": "t (timepoints)", "Z": "z (slices)", "Y": "y (pixels)",
                 "X": "x (pixels)"}
        for row, letter in enumerate(a for a in "TZYX" if a in sizes):
            n = sizes[letter]
            first, end = start.ranges.get(letter.lower(), (0, n))
            spins = [QSpinBox(), QSpinBox()]
            for spin, value in zip(spins, (first + 1, end)):
                spin.setRange(1, n)
                spin.setValue(max(1, min(n, value)))
                spin.valueChanged.connect(self._update)
            grid.addWidget(QLabel(names[letter]), row, 0)
            grid.addWidget(QLabel("from"), row, 1)
            grid.addWidget(spins[0], row, 2)
            grid.addWidget(QLabel("to"), row, 3)
            grid.addWidget(spins[1], row, 4)
            grid.addWidget(QLabel(f"of {n}"), row, 5)
            self.range_spins[letter] = spins
        whole = QPushButton("Whole image")
        whole.clicked.connect(self._whole_image)
        grid.addWidget(whole, len(self.range_spins), 0, 1, 2)
        self.dependent_layout.addWidget(group)
        self.show_3d_check = QCheckBox(
            "3D view (can also be switched on later)")
        self.show_3d_check.setChecked(start.show_3d)
        self.show_3d_check.setEnabled(sizes.get("Z", 1) > 1)
        self.dependent_layout.addWidget(self.show_3d_check)
        has_z, has_t = "Z" in sizes, "T" in sizes
        self.voxel_spins["z"].setEnabled(has_z)
        for widget in (self.interval_label, self.interval_spin,
                       self.time_combo):
            widget.setEnabled(has_t)
        self._update()

    def _whole_image(self):
        for spins in self.range_spins.values():
            for spin, value in zip(spins, (spins[0].minimum(),
                                           spins[1].maximum())):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._update()

    # --- reading the choices ------------------------------------------------------

    def choice(self) -> fi.FileChoice:
        """The choice; calibration values only where they differ from the
        file's (so the record can say which values came from the file)."""
        scene = self._scene()
        axes = self._axes()
        channels = None
        if self.channel_checks and not all(c.isChecked()
                                           for c in self.channel_checks):
            channels = [c for c, check in enumerate(self.channel_checks)
                        if check.isChecked()]
        ranges = {letter.lower(): (spins[0].value() - 1, spins[1].value())
                  for letter, spins in self.range_spins.items()}
        voxel = {}
        for k, spin in self.voxel_spins.items():
            if k == "z" and "Z" not in axes:   # a 2D image: no z size
                continue
            known = scene.voxel.get(k)
            if known is None or not _same(spin.value(), known):
                voxel[k] = spin.value()
        unit = self.unit_combo.currentData()
        interval = self.interval_spin.value()
        t_unit = self.time_combo.currentData()
        if "T" not in axes:                    # no time: no interval
            interval, t_unit = None, scene.time_unit
        elif scene.interval and _same(interval, scene.interval):
            interval = None                    # as in the file
        return fi.FileChoice(
            scene=self.scene_combo.currentData(),
            axes=axes if self.axis_combos else None, channels=channels,
            ranges=ranges, voxel=voxel or None,
            unit=None if unit == scene.unit else unit,
            interval=interval,
            time_unit=None if t_unit == scene.time_unit else t_unit,
            show_3d=self.show_3d_check.isChecked()
            and self.show_3d_check.isEnabled())

    def _update(self, *_):
        """Check the choices; show the calibration's origin and the
        memory needed (Load is off when they cannot be loaded)."""
        if not hasattr(self, "show_3d_check"):        # still being built
            return
        scene = self._scene()
        problem = ""
        choice = self.choice()
        try:
            fi.axes_of(scene, choice)
        except ValueError as err:
            problem = str(err)
        if not problem and any(spins[0].value() > spins[1].value()
                               for spins in self.range_spins.values()):
            problem = "Each range must start before it ends."
        if not problem and self.channel_checks and not any(
                c.isChecked() for c in self.channel_checks):
            problem = "Tick at least one channel."
        # where each calibration value comes from
        sizes = {a for a in self._axes() if a != "-"}
        missing = [k for k in "zyx" if scene.voxel.get(k) is None
                   and (k != "z" or "Z" in sizes)]
        if scene.unit is None:
            missing = [k for k in "zyx" if k != "z" or "Z" in sizes]
        notes = []
        if missing:
            notes.append("<b>Not in the file:</b> voxel size "
                         + ", ".join(missing) + ". Enter the real values "
                         "(otherwise 1 is assumed).")
        else:
            notes.append("Voxel size from the file.")
        if "T" in sizes:
            notes.append("Frame interval from the file." if scene.interval
                         else "<b>No frame interval in the file:</b> enter "
                              "it (otherwise 1 is assumed).")
        self.calibration_note.setText(" ".join(notes))
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if problem:
            ok.setEnabled(False)
            self.memory.setText(f"<b>Problem:</b> {escape(problem)}")
            return
        needed = fi.memory_needed(self.reader, choice)
        text, refuse = fi.memory_problem(needed, fi.free_memory())
        ok.setEnabled(not refuse)
        self.memory.setText(f"<b>{escape(text)}</b>" if refuse
                            else escape(text))


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-12)


def _unit_text(unit: str) -> str:
    return "µm" if unit == "micrometer" else vc.short_unit(unit)


def _clear(layout):
    """Remove (and delete) everything in a layout."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()            # at once (deleteLater waits)
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


def _is_time(image: vc.OmeZarrImage, axis: str) -> bool:
    return image.axes[image.axis_names.index(axis)].get("type") == "time"


def _unit(image: vc.OmeZarrImage, axis: str) -> str:
    unit = image.unit(axis)
    return "µm" if unit == "micrometer" else vc.short_unit(unit)


def _label(text: str) -> QLabel:
    """A label the dialog formats itself (bold, italics): rich text,
    with every name from the file passed through escape()."""
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    return label
