"""The panel's "Channels" part (0.18.0, the user's wish of 2026-11-13).

One row per channel: a tick box to show it (in all views; a hidden channel
is not read at all), its name, and a colour menu (the colour from the
file, or one of napari's colour maps). Clicking a row chooses the channel
for Levels (its histogram, contrast, gamma and auto-contrast). It replaces
the "Show channels" tick boxes that were under View, and, with Levels and
the drawing tools under Crop regions, napari's own layer list and layer
controls, which are therefore hidden at first.

The rows act on the XY view's layer of each channel; OrthoViews passes its
changes on to the other views (and back here, through on_display).
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QImage, QPixmap
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                            QHBoxLayout, QListWidget,
                            QListWidgetItem, QVBoxLayout, QWidget)

# (0.23.0) channel names from files: plain text (see qt_text)
from volume_cropper.qt_text import QLabel, literal

# napari's colour maps offered besides the file's own colour
COLOURS = (("gray", "grey"), ("red", "red"), ("green", "green"),
           ("blue", "blue"), ("cyan", "cyan"), ("magenta", "magenta"),
           ("yellow", "yellow"), ("inferno", "inferno"),
           ("viridis", "viridis"))
OWN = "own"
SWATCH = 14                      # the colour icons' size in pixels


def swatch(colormap) -> QIcon:
    """A small icon of a colour map: its colours from dark to bright."""
    from napari.utils.colormaps import ensure_colormap
    colors = np.asarray(ensure_colormap(colormap).map(
        np.linspace(0, 1, SWATCH)), dtype=float)[:, :3]
    pixels = np.repeat((colors * 255).astype(np.uint8)[None], SWATCH,
                       axis=0).tobytes()
    image = QImage(pixels, SWATCH, SWATCH, 3 * SWATCH,
                   QImage.Format.Format_RGB888)
    return QIcon(QPixmap.fromImage(image.copy()))   # (owns its pixels)


class ChannelsPanel(QWidget):
    """The channel list; fed by the panel (set_image, on_display)."""

    def __init__(self, ortho, parent=None):
        super().__init__(parent)
        self.ortho = ortho
        self.on_channel_chosen = None    # f(k): a row was clicked
        self.show_checks: list = []      # (also used by the older checks)
        self.colour_combos: list = []
        self.names: list = []            # the rows' name labels
        self._own: list = []             # each channel's colour from the file
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setToolTip(
            "Click a channel to choose it for Levels (histogram, contrast, "
            "gamma, auto-contrast). Untick it to hide it in all views: a "
            "hidden channel is not read at all (less memory and network). "
            "The menu sets its colour.")
        self.list.currentRowChanged.connect(self._chosen)
        layout.addWidget(self.list)

    # --- what the panel tells it ---------------------------------------------

    def set_image(self):
        """A new image: one row per channel, as the views show it."""
        self.list.blockSignals(True)
        self.list.clear()
        self.show_checks, self.colour_combos, self._own = [], [], []
        self.names = []
        image = self.ortho.image
        if image is None:
            self.list.blockSignals(False)
            return
        display = self.ortho.display
        for k, label in enumerate(image.channel_labels or ["image"]):
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(4, 1, 4, 1)
            check = QCheckBox()
            check.setChecked(bool(display[k]["visible"]))
            check.setToolTip("Show this channel in all views; a hidden "
                             "channel is not read (less memory and "
                             "network).")
            check.toggled.connect(
                lambda on, k=k: self.ortho.set_channel_visible(k, on))
            name = QLabel(f"{k}: {label}")
            name.setToolTip(literal(label))
            combo = QComboBox()
            own = display[k]["colormap"]
            self._own.append(own)
            combo.addItem(swatch(own), "file's colour", OWN)
            for value, text in COLOURS:
                combo.addItem(swatch(value), text, value)
            combo.setToolTip("The channel's colour in all views: the one "
                             "from the file, or one of napari's.")
            combo.activated.connect(lambda _i, k=k: self._colour(k))
            line.addWidget(check)
            line.addWidget(name, 1)
            line.addWidget(combo)
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            self.show_checks.append(check)
            self.colour_combos.append(combo)
            self.names.append(name)
        rows = max(1, min(self.list.count(), 6))
        height = self.list.sizeHintForRow(0) if self.list.count() else 24
        self.list.setFixedHeight(rows * height + 2 * self.list.frameWidth()
                                 + 2)
        self.list.blockSignals(False)

    def select(self, k: int):
        """Show channel k as the chosen one (Levels chose it): its row
        selected and its name in bold (the list's own highlight lies
        behind the row's controls, where it hardly shows)."""
        if 0 <= k < self.list.count() and self.list.currentRow() != k:
            self.list.blockSignals(True)
            self.list.setCurrentRow(k)
            self.list.blockSignals(False)
        for row, name in enumerate(self.names):
            font = name.font()
            font.setBold(row == k)
            name.setFont(font)

    def on_display(self, k: int, attr: str):
        """A display setting of channel k changed (here, in Levels, or in
        napari): the tick box and colour menu follow."""
        if not 0 <= k < len(self.show_checks):
            return
        value = self.ortho.display[k].get(attr)
        if attr == "visible":
            check = self.show_checks[k]
            check.blockSignals(True)
            check.setChecked(bool(value))
            check.blockSignals(False)
        elif attr == "colormap":
            combo = self.colour_combos[k]
            name = getattr(value, "name", str(value))
            index = combo.findData(name)
            if index < 0 and name == getattr(self._own[k], "name", None):
                index = combo.findData(OWN)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

    # --- the user ------------------------------------------------------------

    def _chosen(self, k: int):
        if k >= 0 and self.on_channel_chosen is not None:
            self.on_channel_chosen(k)
        if k >= 0:
            self.select(k)          # (the bold name, also if Levels had it)

    def _colour(self, k: int):
        """The colour menu of channel k: set it in all views (through the
        XY view's layer)."""
        data = self.colour_combos[k].currentData()
        colormap = self._own[k] if data == OWN else data
        xy = self.ortho.views.get("XY")
        layers = [] if xy is None else xy.channel_layers
        if k < len(layers):
            layers[k].colormap = colormap
