"""The panel's "Levels" part (0.18.0): a live histogram of one channel,
with numbers, whose contrast lines can be dragged.

The user asked for it on 2026-11-12: napari 0.9.2's own histogram (in the
layer controls) shows no numbers. This one shows

- the values along the axis and the counts up the side (log if ticked),
- the contrast limits as white lines with their values; dragging a line
  sets that limit (in all views, like napari's contrast slider),
- napari's mapping of values to brightness (the yellow line, with gamma),
- gamma as a dashed line at the value shown at half brightness; dragging
  it sets gamma (double-click: 1), like the grey slider of Photoshop's
  Levels (the user's wish, 2026-11-12),
- boxes for the lower and upper limit (min, max) and gamma, to type exact
  values (the user's wish, 2026-11-12),
- under the pointer: the bin's values and how many voxels it holds,
- the data's minimum, maximum and mean, the share of voxels at the maximum
  (a sign of saturation) and below / above the contrast limits,
- (0.22.0, the user's wish, 2026-11-25) "zoom to min–max": the axis shows
  the contrast limits with room on either side (levels.ZOOM_MARGIN), in
  bins computed again over that range. It follows the limits when a drag
  ends (never during one: the axis would move under the pointer) and when
  they are set otherwise (boxes, auto-contrast, napari); to go further than
  the room shown, drag to the edge, let go and drag again,

of one of three sources (levels.py): the XY view as shown (it follows the
slider, zoom and projection), the whole image, or the selected crop region.
The histogram itself is computed by levels.py; this file only draws it
(with matplotlib, which the tool's environment already has).
"""

from __future__ import annotations

import numpy as np
from matplotlib import ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                            QHBoxLayout, QPushButton, QSizePolicy,
                            QVBoxLayout, QWidget)

from volume_cropper import levels as lv
from volume_cropper.qt_text import QLabel   # plain text (0.23.0)

SOURCES = (("XY view (as shown)", "view"), ("whole image", "whole"),
           ("selected crop region", "region"))
GRAB_PIXELS = 6            # a press this close to a line drags it
LUT_COLOR = "#ffd700"      # the brightness mapping (napari draws it yellow)
LOWER, UPPER, MIDTONE = 0, 1, 2     # the lines (and boxes) that can be set
MARKER = 3            # (0.18.0) the Autocropper's own threshold (set_marker)
MARKER_FIRST = 2.0    # pixels: it wins over a line this much nearer
HINT = ("Point at the histogram to read a bin. Drag a white line to set "
        "that contrast limit, or the dashed line to set gamma "
        "(double-click it for 1).")


def _theme_colors(viewer) -> tuple[str, str, str]:
    """napari's background, frame and text colours (dark theme if unknown)."""
    try:
        from napari.utils.theme import get_theme
        theme = get_theme(viewer.theme)
        return (theme.background.as_hex(), theme.foreground.as_hex(),
                theme.text.as_hex())
    except Exception:
        return "#262930", "#414851", "#f0f1f2"


def _compact(value: float, _pos=None) -> str:
    """Counts as 950, 12k, 3.4M."""
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"


def _percent(fraction: float) -> str:
    if fraction == 0:
        return "0 %"
    if fraction < 0.0001:
        return "< 0.01 %"
    return f"{100 * fraction:.2g} %"


class LevelsPanel(QWidget):
    """A channel's histogram with numbers; its contrast lines can be
    dragged. Fed by the panel (crop_gui.CropWidget) and OrthoViews."""

    def __init__(self, ortho, parent=None):
        super().__init__(parent)
        self.ortho = ortho
        self.viewer = ortho.viewer
        self.hist: lv.Histogram | None = None
        self.limits: list[float] | None = None
        self.gamma = 1.0
        self.what = ""              # where the values came from
        self._drag: int | None = None      # LOWER, UPPER or MIDTONE line
        self.on_channel_changed = None     # f(k): another channel shown
        # (0.18.0) a value marked on one channel's histogram (the
        # Autocropper's own threshold): (channel, value, colour, label) or
        # None; dragging it calls on_marker_moved(value)
        self.marker: tuple | None = None
        self.on_marker_moved = None
        self._integer = True        # the image holds whole numbers
        self._axis = None           # (0.22.0) the range the bins cover now
        self._stale = True          # recompute when next shown
        self._whole_cache: dict = {}
        self._artists: dict = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        # (0.18.0) The channel is chosen in the Channels part (a click on
        # its row) or in napari's layer list; here only its name. The combo
        # stays, hidden, as what holds the choice.
        self.channel_combo = QComboBox()
        self.channel_combo.setVisible(False)
        self.channel_combo.currentIndexChanged.connect(self._on_channel)
        self.channel_name = QLabel("")
        self.channel_name.setToolTip("The channel shown: click another one "
                                     "under Channels to choose it.")
        font = self.channel_name.font()
        font.setBold(True)
        self.channel_name.setFont(font)
        row.addWidget(self.channel_name)
        self.source_combo = QComboBox()
        for text, key in SOURCES:
            self.source_combo.addItem(text, key)
        self.source_combo.setToolTip(
            "XY view (as shown): what the XY view shows now (the visible "
            "part of the slice or projection, at the level shown); it "
            "follows the slider, zoom and projection.\n"
            "Whole image: all of the loaded image at the timepoint shown.\n"
            "Selected crop region: the voxels its crop keeps (inside its "
            "outlines).\nThe last two are read from the finest level with "
            f"at most {lv.MAX_VALUES / 1e6:g} million voxels there (coarser "
            "levels may smooth the extremes).")
        self.source_combo.currentIndexChanged.connect(
            lambda _: self.schedule(0, restart=True))
        row.addWidget(self.source_combo)
        self.log_check = QCheckBox("log")
        self.log_check.setToolTip(
            "Counts on a logarithmic scale: shows the few brightest voxels "
            "next to a large dark background.")
        self.log_check.toggled.connect(lambda _: self._draw())
        row.addWidget(self.log_check)
        # (0.22.0, the user's wish) the axis zoomed to the contrast limits
        self.zoom_check = QCheckBox("zoom to min–max")
        room = round(100 * lv.ZOOM_MARGIN)
        self.zoom_check.setToolTip(
            "Show min to max across the histogram, with room on either side "
            f"({room} % of the distance between them) to drag them further "
            "out. The bins are computed again over that range, for more "
            "detail.\nThe axis follows when you let go of a line or type a "
            "value: to go further, drag a line to the edge, let go and drag "
            "again.\nThe numbers below the histogram always count all the "
            "voxels, also those beyond the axis.")
        self.zoom_check.toggled.connect(lambda _: self._refit())
        row.addWidget(self.zoom_check)
        row.addStretch()
        layout.addLayout(row)

        self.background, self.frame, self.text = _theme_colors(self.viewer)
        self.figure = Figure(figsize=(3.4, 1.8), dpi=100,
                             facecolor=self.background)
        self.figure.subplots_adjust(left=0.14, right=0.98, top=0.84,
                                    bottom=0.17)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(170)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        self.ax = self.figure.add_subplot(111)
        self.ax_lut = self.ax.twinx()     # the mapping, on its own 0-1 scale
        layout.addWidget(self.canvas)

        # min, max and gamma as numbers (the user's wish, 2026-11-12). They
        # apply when typing ends (Enter, or another field clicked) or with
        # the arrows, not at every key: typing 5000 would otherwise set 5,
        # 50 and 500 on the way.
        boxes = QHBoxLayout()
        self.min_spin = self._number_box(
            "min",
            "The lower contrast limit: this value and all below it are "
            "shown black (Fiji: Minimum, in Brightness/Contrast). The "
            "left white line in the histogram.", boxes)
        self.max_spin = self._number_box(
            "max",
            "The upper contrast limit: this value and all above it are "
            "shown fully bright (Fiji: Maximum). The right white line in "
            "the histogram.", boxes)
        self.gamma_spin = self._number_box(
            "gamma",
            "How values between min and max are shown: 1 = evenly (a "
            "straight line); below 1 brightens the dim parts, above 1 "
            "darkens them (from 0.2 to 2, as in napari).\nThe dashed line "
            "in the histogram marks the value shown at half brightness: "
            "drag it to set gamma, double-click it for 1.\nGamma changes "
            "only the display: crops keep the original values.", boxes)
        lo, hi = lv.GAMMA_RANGE
        self.gamma_spin.setRange(lo, hi)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)
        # Qt sizes a number box for its range (e.g. 65535 for 16-bit data);
        # in a narrow panel they may shrink to these widths, which leave
        # room for napari's - and + buttons on either side of the number
        # (longer values then scroll inside the box)
        for spin, digits in ((self.min_spin, 6), (self.max_spin, 6),
                             (self.gamma_spin, 4)):
            spin.setMinimumWidth(
                spin.fontMetrics().horizontalAdvance("0" * digits) + 64)
        boxes.addStretch()
        layout.addLayout(boxes)
        self.min_spin.valueChanged.connect(lambda _: self._on_box(LOWER))
        self.max_spin.valueChanged.connect(lambda _: self._on_box(UPPER))
        self.gamma_spin.valueChanged.connect(lambda _: self._on_box(MIDTONE))

        # (0.18.0, the user's wish) auto-contrast as in napari's layer
        # controls, which are hidden at first now
        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("Auto-contrast"))
        self.auto_once = QPushButton("once")
        self.auto_once.setToolTip(
            "Set min and max from what the XY view shows now (napari's "
            "auto-contrast \"once\").")
        self.auto_once.clicked.connect(self._auto_once)
        self.auto_continuous = QPushButton("continuous")
        self.auto_continuous.setCheckable(True)
        self.auto_continuous.setToolTip(
            "Keep setting min and max from what the XY view shows, at every "
            "slice (napari's \"continuous\"). Setting them yourself (lines, "
            "boxes) stops it.")
        self.auto_continuous.toggled.connect(self._auto_toggled)
        self.auto_reset = QPushButton("reset")
        self.auto_reset.setToolTip(
            "Back to the contrast the channel had when the image opened "
            "(from its intensities over the image), and gamma 1.")
        self.auto_reset.clicked.connect(self._reset_contrast)
        for button in (self.auto_once, self.auto_continuous, self.auto_reset):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self._enable_boxes(False)

        self.readout = QLabel(HINT)
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setToolTip(
            "Data: the lowest, highest and mean value of the voxels counted "
            "(not the contrast limits, which are in the min and max boxes)."
            "\nAt the maximum: the share of voxels at the highest value; many "
            "there suggests saturation.\nBelow / above the limits: the "
            "shares shown black and fully bright with this contrast.\n"
            "Voxels: how many values the histogram holds, and where they "
            "come from (the pyramid level read).")
        self.stats.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.stats)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.recompute)
        # dragging: the views follow at most every 40 ms
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(40)
        self._apply_timer.timeout.connect(
            lambda: self._drag is not None and self._apply(self._drag))
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer)
        self._show_message("No image open.")

    # --- what the panel is told -------------------------------------------

    def set_image(self):
        """A new image (or the same one loaded again) is shown."""
        image = self.ortho.image
        self._whole_cache = {}
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if image is not None:
            labels = image.channel_labels or ["image"]
            for k, label in enumerate(labels):
                self.channel_combo.addItem(f"{k}: {label}")
            shown = [k for k, d in enumerate(self.ortho.display)
                     if d.get("visible")]
            self.channel_combo.setCurrentIndex(shown[0] if shown else 0)
            self._set_box_range(image.levels[0].array.dtype)
        self.channel_combo.blockSignals(False)
        self._enable_boxes(image is not None)
        self._on_channel()

    def _on_channel(self, _index=None):
        """Another channel is shown (chosen under Channels, in napari's
        layer list, or by a new image)."""
        self.channel_name.setText(self.channel_combo.currentText())
        layer = self._layer()
        self.auto_continuous.blockSignals(True)
        self.auto_continuous.setChecked(bool(layer is not None
                                             and layer.auto_contrast))
        self.auto_continuous.blockSignals(False)
        if self.on_channel_changed is not None:
            self.on_channel_changed(self.channel)
        self._show_layer_values()
        self.schedule(0, restart=True)

    # --- auto-contrast (0.18.0) ----------------------------------------------

    def _auto_once(self):
        """napari's "once": min and max from what the XY view shows."""
        layer = self._layer()
        if layer is not None:
            self._auto_toggled(False)
            layer.reset_contrast_limits()

    def _auto_toggled(self, on: bool):
        """napari's "continuous": min and max follow every slice shown."""
        layer = self._layer()
        if layer is not None and bool(layer.auto_contrast) != bool(on):
            layer.auto_contrast = bool(on)
        if self.auto_continuous.isChecked() != bool(on):
            self.auto_continuous.blockSignals(True)
            self.auto_continuous.setChecked(bool(on))
            self.auto_continuous.blockSignals(False)

    def _reset_contrast(self):
        """Back to the contrast the channel had when the image opened (the
        tool's starting contrast: percentiles of a small level of the
        loaded region) and gamma 1."""
        from volume_cropper import ortho_views as ov
        layer, image = self._layer(), self.ortho.image
        if layer is None or image is None:
            return
        self._auto_toggled(False)
        limits = ov.channel_contrast(image, self.channel, self.ortho.region)
        layer.contrast_limits = tuple(float(v) for v in limits)
        layer.gamma = 1.0

    def on_xy_data(self, k: int):
        """napari loaded new data into the XY view's layer of channel k
        (another slice, zoom, level or projection)."""
        if self.source == "view" and k == self.channel:
            self.schedule(80)

    def on_display(self, k: int, attr: str):
        """A display setting of channel k changed (napari or this panel)."""
        if k != self.channel:
            return
        if attr == "visible" and self.source == "view":
            self.schedule(0, restart=True)
        elif attr == "colormap":
            self._draw()
        elif attr in ("contrast_limits", "gamma") and self._drag is None:
            self._show_layer_values()      # the boxes, also without a histogram
            layer = self._layer()
            if layer is None or self.hist is None:
                return
            self.limits = [float(v) for v in layer.contrast_limits]
            self.gamma = float(layer.gamma)
            edges = self.hist.edges
            if self.zoom_check.isChecked():
                # (0.22.0) the zoomed axis follows the limits
                if not self._refit():
                    self._move_lines()
            elif self.limits[0] < edges[0] or self.limits[1] > edges[-1]:
                # set beyond the axis (napari's slider): widen the bins
                self.hist = self._bin(self.hist.values, self.limits)
                self._draw()
            else:
                self._move_lines()

    def on_regions_changed(self):
        """Crop regions were drawn, edited, selected or deleted."""
        if self.source == "region":
            self.schedule(300, restart=True)

    def on_time_changed(self):
        """Another timepoint is shown."""
        if self.source in ("whole", "region"):
            self.schedule(100, restart=True)

    # --- computing ------------------------------------------------------------

    @property
    def source(self) -> str:
        return self.source_combo.currentData()

    @property
    def channel(self) -> int:
        return max(self.channel_combo.currentIndex(), 0)

    def _layer(self):
        """The XY view's layer of the channel shown (its contrast drives all
        views), or None."""
        xy = self.ortho.views.get("XY")
        layers = [] if xy is None else xy.channel_layers
        return layers[self.channel] if self.channel < len(layers) else None

    def schedule(self, delay: int, restart: bool = False):
        """Recompute after delay ms; while hidden (part closed), only when
        shown again. Without restart, a pending recompute keeps its time,
        so a stream of changes (scrolling) updates at most every delay."""
        if not self.isVisible():
            self._stale = True
            return
        if restart or not self._timer.isActive():
            self._timer.start(delay)

    def showEvent(self, event):
        super().showEvent(event)
        if self._stale:
            self.schedule(0, restart=True)

    def recompute(self):
        """Read the values of the source chosen, and draw their histogram."""
        self._stale = False
        ortho, image = self.ortho, self.ortho.image
        layer = self._layer()
        if image is None or layer is None:
            self._show_message("No image open.")
            return
        self._show_layer_values()        # this channel's, even if not drawn
        k = self.channel
        limits = [float(v) for v in layer.contrast_limits]
        try:
            if self.source == "view":
                if not layer.visible:
                    self._show_message(
                        f"Channel {k} is hidden. Switch it on (Show, under "
                        "View) to see the histogram of the view.")
                    return
                layer_slice = getattr(layer, "_slice", None)
                raw = (None if layer_slice is None
                       else layer_slice.image.raw)
                if raw is None or np.size(raw) == 0:
                    self._show_message("Waiting for napari to load the "
                                       "view…")
                    return
                values = np.asarray(raw)
                level = ortho.current_level(ortho.views["XY"])
                projection = {"max": ", max projection",
                              "mean": ", mean projection"}.get(
                                  ortho.projection, "")
                what = f"the XY view as shown (level {level}{projection})"
            elif self.source == "whole":
                time = ortho.time_shown()
                key = (k, time)
                if key not in self._whole_cache:
                    self._whole_cache = {key: lv.read_values(
                        image, ortho.region, k, time)}
                values, level = self._whole_cache[key]
                what = f"the whole image as loaded (level {level})"
            else:
                i = ortho.selected
                ranges = (ortho.box_ranges[i]
                          if 0 <= i < len(ortho.box_ranges) else None)
                if ranges is None:
                    self._show_message("Select a crop region (in the list "
                                       "under Crop regions).")
                    return
                box = ortho.boxes[i]
                values, level = lv.read_values(
                    image, ortho.region, k, ortho.time_shown(), ranges, box)
                what = f"the crop region {box.name} (level {level})"
        except Exception as err:          # e.g. a file no longer readable
            self._show_message(f"Could not read the values: {err}")
            return
        hist = self._bin(values, limits)
        if hist is None:
            self._show_message("No voxels there.")
            return
        self.hist, self.limits, self.what = hist, limits, what
        self.gamma = float(layer.gamma)
        if self._drag is None:          # an old bin's reading is gone
            self.readout.setText(HINT)
        self._draw()

    # --- zoom to min–max (0.22.0) -----------------------------------------

    def _zoom_args(self) -> tuple:
        """(margin, keep) for levels.histogram and levels.axis_range: the
        room around the limits and the values to keep in view when "zoom to
        min–max" is ticked (the Autocropper's threshold, if shown), else
        (None, ())."""
        if not self.zoom_check.isChecked():
            return None, ()
        keep = ([self.marker[1]] if self.marker is not None
                and self.marker[0] == self.channel else [])
        return lv.ZOOM_MARGIN, keep

    def _bin(self, values, limits):
        """The histogram of values to draw: over the whole axis, or zoomed
        to the limits (see levels.axis_range). Notes the range it covers."""
        margin, keep = self._zoom_args()
        hist = lv.histogram(values, limits, margin=margin, keep=keep)
        if hist is not None:
            self._axis = lv.axis_range((hist.minimum, hist.maximum), limits,
                                       margin, keep)
        return hist

    def _refit(self) -> bool:
        """Bin the values again if the axis should now cover another range
        (the limits moved with "zoom to min–max" ticked, or it was ticked or
        unticked), and draw. Never while a line is dragged: the axis would
        move under the pointer. True if it drew anew."""
        if self.hist is None or self.limits is None or self._drag is not None:
            return False
        margin, keep = self._zoom_args()
        wanted = lv.axis_range((self.hist.minimum, self.hist.maximum),
                               self.limits, margin, keep)
        if wanted == self._axis:
            return False
        self.hist = self._bin(self.hist.values, self.limits)
        self._draw()
        return True

    # --- drawing ----------------------------------------------------------------

    def _style(self, message: str = ""):
        """Empty axes in napari's colours, with an optional message."""
        for ax in (self.ax, self.ax_lut):
            ax.clear()
        ax = self.ax
        ax.set_facecolor(self.background)
        for spine in ax.spines.values():
            spine.set_color(self.frame)
        for spine in self.ax_lut.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=self.text, labelsize=7.5, length=3)
        self.ax_lut.set_yticks([])
        self.ax_lut.set_ylim(0, 1)
        if message:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                    va="center", color=self.text, fontsize=8, wrap=True)

    def _show_message(self, message: str):
        self.hist = None
        self.limits = None       # (may be another channel's; the boxes keep
        self._artists = {}       # working from the layer's own values)
        self._style(message)
        self.stats.setText("")
        self.readout.setText(HINT)
        self.canvas.draw_idle()

    def _channel_color(self) -> tuple:
        display = self.ortho.display
        if self.channel < len(display):
            try:
                return tuple(display[self.channel]["colormap"].colors[-1])
            except (KeyError, AttributeError, IndexError):
                pass
        return (0.8, 0.8, 0.8, 1.0)

    def _draw(self):
        """Draw the histogram, the contrast lines and the mapping."""
        hist = self.hist
        if hist is None:
            return
        self._style()
        ax = self.ax
        ax.stairs(hist.counts, hist.edges, fill=True,
                  color=self._channel_color(), alpha=0.9, linewidth=0)
        ax.set_xlim(hist.edges[0], hist.edges[-1])
        top = max(int(hist.counts.max()), 1)
        if self.log_check.isChecked():
            ax.set_yscale("log", nonpositive="clip")
            ax.set_ylim(0.8, top * 1.6)
        else:
            ax.set_ylim(0, top * 1.05)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_compact))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _p: lv.describe_number(v, hist.integer)))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,
                                                      integer=hist.integer))
        ax.yaxis.set_major_locator(
            ticker.LogLocator(numticks=4) if self.log_check.isChecked()
            else ticker.MaxNLocator(3, integer=True))
        top_of_axes = blended_transform_factory(ax.transData, ax.transAxes)
        self._artists = {
            "lines": [ax.axvline(v, color=self.text, linewidth=1.5)
                      for v in self.limits],
            "labels": [ax.text(v, 1.03, "", transform=top_of_axes,
                               va="bottom", color=self.text, fontsize=7.5)
                       for v in self.limits],
            "lut": self.ax_lut.plot([], [], color=LUT_COLOR, linewidth=1,
                                    alpha=0.85)[0],
            # gamma: the value shown at half brightness (no label: the
            # gamma box shows the number, and a label would collide with
            # the limits' labels)
            "midtone": ax.axvline(0, color=self.text, linewidth=1.2,
                                  linestyle=(0, (4, 3))),
            # (0.18.0) the marked value (the Autocropper's threshold)
            "marker": ax.axvline(0, linewidth=1.5, linestyle=(0, (1.5, 1.5)),
                                 visible=False),
            "marker_label": ax.text(0, 0.96, "", transform=top_of_axes,
                                    va="top", fontsize=7, visible=False),
        }
        self._move_marker(draw=False)
        span = hist.edges[-1] - hist.edges[0]
        for spin in (self.min_spin, self.max_spin):
            spin.blockSignals(True)        # (decimals round the value)
            spin.setDecimals(lv.box_decimals(span, self._integer))
            spin.setSingleStep(lv.box_step(span, self._integer))
            spin.blockSignals(False)
        self._move_lines(draw=False)
        self.canvas.draw_idle()

    def _move_lines(self, draw: bool = True):
        """Put the contrast lines, their values and the mapping where the
        limits are now, and update the numbers below."""
        if not self._artists or self.hist is None:
            return
        low, high = self.limits
        x0, x1 = self.hist.edges[0], self.hist.edges[-1]
        span = max(x1 - x0, 1e-12)
        for i, (line, label, value) in enumerate(zip(
                self._artists["lines"], self._artists["labels"],
                self.limits)):
            line.set_xdata([value, value])
            label.set_x(value)
            label.set_text(lv.describe_number(value, self.hist.integer))
            # lower value left of its line, upper right of it (inside)
            where = (value - x0) / span
            if i == 0:
                label.set_ha("right" if where > 0.15 else "left")
            else:
                label.set_ha("left" if where < 0.85 else "right")
        x = np.linspace(low, high, 64)
        ramp = np.clip((x - low) / max(high - low, 1e-12), 0, 1) ** self.gamma
        self._artists["lut"].set_data(np.r_[x0, x, x1],
                                      np.r_[0.0, ramp, 1.0])
        middle = lv.midtone_value(self.limits, self.gamma)
        self._artists["midtone"].set_xdata([middle, middle])
        self._show_values(self.limits, self.gamma)
        self._update_stats()
        if draw:
            self.canvas.draw_idle()

    # --- a marked value (0.18.0) -------------------------------------------

    def set_marker(self, channel: int | None, value: float | None = None,
                   color: str = "#ff9a2e", label: str = ""):
        """Mark a value on channel's histogram (shown while that channel
        is), as a dotted line that can be dragged; channel None removes
        it."""
        self.marker = (None if channel is None or value is None
                       else (int(channel), float(value), color, label))
        if self.zoom_check.isChecked() and self._refit():
            return              # (0.22.0) zoomed to keep it in view; drawn
        self._move_marker()

    def _marker_shown(self) -> bool:
        return (self.marker is not None and self.hist is not None
                and self.marker[0] == self.channel)

    def _move_marker(self, draw: bool = True):
        if "marker" not in self._artists:
            return
        line, text = self._artists["marker"], self._artists["marker_label"]
        shown = self._marker_shown()
        line.set_visible(shown)
        text.set_visible(shown)
        if shown:
            _, value, color, label = self.marker
            line.set_xdata([value, value])
            line.set_color(color)
            x0, x1 = self.hist.edges[0], self.hist.edges[-1]
            where = (value - x0) / max(x1 - x0, 1e-12)
            text.set_x(value)
            text.set_ha("left" if where < 0.7 else "right")
            text.set_color(color)
            text.set_text(f" {label} " if label else "")
        if draw:
            self.canvas.draw_idle()

    def _update_stats(self):
        hist = self.hist
        if hist is None:
            self.stats.setText("")
            return
        number = lambda v: lv.describe_number(v, hist.integer)   # noqa: E731
        mean = f"{hist.mean:.4g}" if hist.integer else number(hist.mean)
        below, above = hist.outside(self.limits)
        self.stats.setText(
            f"data: min {number(hist.minimum)} · max "
            f"{number(hist.maximum)} · mean {mean}\n"
            f"at the maximum: {_percent(hist.at_max / hist.n)}\n"
            f"below / above the limits: {_percent(below)} / "
            f"{_percent(above)}\n"
            f"{hist.n:,} voxels: {self.what}")

    # --- the mouse ----------------------------------------------------------

    def _x_at(self, event) -> float:
        """The value under the pointer (also beside the axes), clamped to
        the axis."""
        x = self.ax.transData.inverted().transform((event.x, event.y))[0]
        x0, x1 = self.ax.get_xlim()
        return float(min(max(x, x0), x1))

    def _line_near(self, event) -> int | None:
        """The line (LOWER, UPPER, MIDTONE or MARKER) within GRAB_PIXELS of
        the pointer, the nearest if several; None if none is."""
        if not self._artists or self.limits is None:
            return None
        xs = [*self.limits, lv.midtone_value(self.limits, self.gamma)]
        if self._marker_shown():
            xs.append(self.marker[1])
        distances = [abs(self.ax.transData.transform((v, 0))[0] - event.x)
                     for v in xs]
        best = int(np.argmin(distances))      # (a tie: a contrast line)
        # the marked value, drawn on top, also where it lies within
        # MARKER_FIRST pixels of another line (found in the visible check:
        # 1.4 px from the gamma line, it could not be grabbed)
        if len(xs) > MARKER and \
                distances[MARKER] <= distances[best] + MARKER_FIRST:
            best = MARKER
        return best if distances[best] <= GRAB_PIXELS else None

    def _on_press(self, event):
        if event.button != 1 or self.hist is None:
            return
        line = self._line_near(event)
        if event.dblclick and line == MIDTONE:     # back to a straight line
            self._drag = None
            self.gamma = 1.0
            self._move_lines()
            self._apply(MIDTONE)
            return
        self._drag = line

    def _gap(self) -> float:
        """The smallest distance kept between the two limits."""
        if self._integer:
            return 1.0
        if self.hist is not None:
            return float(self.hist.edges[-1] - self.hist.edges[0]) * 1e-3
        return 1e-6

    def _on_motion(self, event):
        if self.hist is None:
            return
        if self._drag is not None:
            value = self._x_at(event)
            number = lambda v: lv.describe_number(  # noqa: E731
                v, self.hist.integer)
            if self._drag == MARKER:
                channel, _, color, label = self.marker
                self.marker = (channel, value, color, label)
                self._move_marker()
                self.readout.setText(f"{label or 'marked value'}: "
                                     f"{number(value)}")
                if not self._apply_timer.isActive():
                    self._apply_timer.start()
                return
            if self._drag == MIDTONE:
                self.gamma = lv.gamma_for_midtone(self.limits, value)
                middle = lv.midtone_value(self.limits, self.gamma)
                text = (f"gamma {self.gamma:.2f}: value {number(middle)} "
                        "is shown at half brightness")
            else:
                if self.hist.integer:
                    value = float(round(value))
                low, high = self.limits
                if self._drag == LOWER:
                    low = min(value, high - self._gap())
                else:
                    high = max(value, low + self._gap())
                self.limits = [low, high]
                name = "lower" if self._drag == LOWER else "upper"
                text = (f"{name} limit: "
                        f"{number(self.limits[self._drag])}")
            self._move_lines()
            self.readout.setText(text)
            if not self._apply_timer.isActive():
                self._apply_timer.start()
            return
        near = self._line_near(event) is not None
        if near:
            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.canvas.unsetCursor()
        if event.inaxes is None:
            self.readout.setText(HINT)
            return
        i = self.hist.bin_at(event.xdata)
        if i is None:
            self.readout.setText(HINT)
            return
        self.readout.setText(self._bin_text(i))

    def _bin_text(self, i: int) -> str:
        hist = self.hist
        a, b = hist.edges[i], hist.edges[i + 1]
        count = int(hist.counts[i])
        share = _percent(count / hist.n)
        if hist.integer:
            low, high = int(np.ceil(a)), int(np.floor(b))
            values = (f"value {low:,}" if low == high
                      else f"values {low:,}–{high:,}")
        else:
            values = f"values {a:.4g}–{b:.4g}"
        return f"{values}: {count:,} voxels ({share})"

    def _on_release(self, event):
        if self._drag is not None:
            line, self._drag = self._drag, None
            self._apply_timer.stop()
            self._apply(line)
            self.readout.setText(HINT)
            if self.zoom_check.isChecked():
                # (0.22.0) the zoomed axis follows, now that the drag ended
                self._refit()

    def _on_leave(self, event):
        if self._drag is None:
            self.readout.setText(HINT)
            self.canvas.unsetCursor()

    def _apply(self, line: int):
        """Give the views what was set here: the limits (line LOWER or
        UPPER) or gamma (MIDTONE). Via the XY layer, whose changes
        OrthoViews passes on to the other views, and from there back to
        on_display (which then fits the histogram and boxes)."""
        if line == MARKER:
            if self.marker is not None and self.on_marker_moved is not None:
                self.on_marker_moved(self.marker[1])
            return
        layer = self._layer()
        if layer is None:
            return
        if line == MIDTONE:
            if float(layer.gamma) != self.gamma:
                layer.gamma = self.gamma
        elif self.limits is not None:
            limits = (float(self.limits[0]), float(self.limits[1]))
            if tuple(float(v) for v in layer.contrast_limits) != limits:
                self._auto_toggled(False)    # (set by hand: no longer auto)
                layer.contrast_limits = limits

    # --- the number boxes ---------------------------------------------------

    def _number_box(self, label: str, tip: str, row) -> QDoubleSpinBox:
        """A labelled number box, added to row."""
        text = QLabel(label)
        text.setToolTip(tip)
        spin = QDoubleSpinBox()
        spin.setToolTip(tip)
        spin.setKeyboardTracking(False)     # apply when typing ends
        row.addWidget(text)
        row.addWidget(spin)
        return spin

    def _enable_boxes(self, enabled: bool):
        for widget in (self.min_spin, self.max_spin, self.gamma_spin,
                       self.auto_once, self.auto_continuous, self.auto_reset):
            widget.setEnabled(enabled)

    def _set_box_range(self, dtype):
        """What the min and max boxes accept: the range of the image's
        whole-number type (e.g. 0-65,535 for 16-bit), else +-1e12."""
        dtype = np.dtype(dtype)
        self._integer = bool(np.issubdtype(dtype, np.integer)
                             or dtype == np.bool_)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            low, high = float(info.min), float(info.max)
        elif dtype == np.bool_:
            low, high = 0.0, 1.0
        else:
            low, high = -1e12, 1e12
        for spin in (self.min_spin, self.max_spin):
            spin.blockSignals(True)
            spin.setDecimals(lv.box_decimals(high - low, self._integer))
            spin.setRange(low, high)
            spin.blockSignals(False)

    def _show_values(self, limits, gamma: float):
        """Show limits and gamma in the boxes, without applying them."""
        values = ((self.min_spin, limits[0]), (self.max_spin, limits[1]),
                  (self.gamma_spin, gamma))
        for spin, value in values:
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _show_layer_values(self):
        """Show the channel's own limits and gamma in the boxes."""
        layer = self._layer()
        if layer is not None:
            self._show_values([float(v) for v in layer.contrast_limits],
                              float(layer.gamma))

    def _on_box(self, box: int):
        """A value typed into (or stepped in) the min, max or gamma box:
        apply it. The other limit comes from the channel itself, so a
        value the box rounds is not applied by the way. A lower limit
        typed at or above the upper one stops just below it (and the
        other way round)."""
        layer = self._layer()
        if layer is None:
            return
        if box == MIDTONE:
            self.gamma = lv.clamp_gamma(self.gamma_spin.value())
        else:
            low, high = (float(v) for v in layer.contrast_limits)
            if box == LOWER:
                low = min(self.min_spin.value(), high - self._gap())
            else:
                high = max(self.max_spin.value(), low + self._gap())
            self.limits = [low, high]
        self._apply(box)
        self._show_layer_values()   # e.g. a limit that had to stop short

    def _on_active_layer(self, event=None):
        """A channel selected in napari's layer list is shown here too."""
        active = self.viewer.layers.selection.active
        xy = self.ortho.views.get("XY")
        if active is None or xy is None:
            return
        for k, layer in enumerate(xy.channel_layers):
            if layer is active and k != self.channel_combo.currentIndex():
                self.channel_combo.setCurrentIndex(k)
                return
