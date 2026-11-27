"""Plugins > Volume Cropper: a small panel in the user's napari window.

It leaves the window as it is and offers:

- "Crop the selected layers": the selected image layers in a new Volume
  Cropper window (the same as Layers > Data > Crop selected layers…).
- "Open Volume Cropper": an empty Volume Cropper window.
- "Use this window…": turns this window itself into Volume Cropper, after
  a warning: its layers are removed when an image is opened, and napari's
  2D/3D, roll, transpose and grid buttons stay hidden until the window is
  closed (the four linked views replace napari's canvas, which cannot be
  undone while the window is open). The tool's panel then sits to the
  right of the views (as in the tool's own windows), this small panel's
  dock goes, and the selected image layers (if any) are opened.

napari passes plugin widgets a "public-only" proxy of the viewer, which
warns whenever napari's private parts are used. The panel itself uses only
public parts. "Use this window…" needs private ones (the four views are
built from napari's Qt widgets, as napari's own multiple-viewer example
does), so it works on the viewer underneath the proxy ("__wrapped__").
"""
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QLabel, QMessageBox, QPushButton, QScrollArea,
                            QVBoxLayout, QWidget)

from ._commands import open_layers_in_new_window, selected_image_layers
from ._tool import import_tool, version


class VolumeCropperPanel(QScrollArea):
    """The dock napari shows (a class, so that napari passes it the viewer
    as napari_viewer); it scrolls when made narrow."""

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer            # napari's public-only proxy
        self.setWidgetResizable(True)
        self.setMinimumWidth(180)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(_label(f"<b>Volume Cropper {version()}</b>"))
        layout.addWidget(_label(
            "Crops regions out of 3D images and time-lapses, in four linked "
            "views (XY, XZ, YZ and 3D). It opens in a window of its own; "
            "this window stays as it is."))
        self.selection_label = _label("")
        # (0.23.0) layer names, maybe from files: plain text, never HTML
        self.selection_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.selection_label)
        self.crop_button = QPushButton("Crop the selected layers")
        self.crop_button.setToolTip(
            "Opens a Volume Cropper window with the image layers selected in "
            "the layer list (several of the same size: the channels of one "
            "image), copied into memory. Also in Layers › Data.")
        self.crop_button.clicked.connect(self._crop_selected)
        layout.addWidget(self.crop_button)
        button = QPushButton("Open Volume Cropper")
        button.setToolTip("An empty Volume Cropper window, to open an "
                          "OME-Zarr or TIFF image in it. Also in File › IO "
                          "Utilities.")
        button.clicked.connect(self._open_new)
        layout.addWidget(button)
        button = QPushButton("Use this window…")
        button.setToolTip(
            "Turns this window into Volume Cropper (asks first: its layers "
            "are removed when an image is opened, and napari's 2D/3D and "
            "grid buttons stay hidden until the window is closed).")
        button.clicked.connect(self._use_this_window)
        layout.addWidget(button)
        layout.addStretch()
        self.setWidget(inner)
        try:              # keep the selection line up to date
            self.viewer.layers.selection.events.changed.connect(
                self._show_selection)
        except Exception:
            pass
        self._show_selection()

    def _show_selection(self, event=None):
        try:
            layers = selected_image_layers(self.viewer)
            self.selection_label.setText(
                "Selected image layers: "
                + (", ".join(layer.name for layer in layers) if layers
                   else "none (select some in the layer list)."))
            self.crop_button.setEnabled(bool(layers))
        except RuntimeError:     # the panel was closed (its Qt part is gone)
            pass

    def _crop_selected(self):
        open_layers_in_new_window(self.viewer, parent=self)

    def _open_new(self):
        import_tool().open_window()

    def _use_this_window(self):
        crop_gui = import_tool()
        viewer = getattr(self.viewer, "__wrapped__", self.viewer)
        if crop_gui.is_tool_viewer(viewer):
            QMessageBox.information(self, "Volume Cropper",
                                    "This window is a Volume Cropper window "
                                    "already.")
            return
        if crop_gui.is_reading():
            QMessageBox.information(
                self, "Volume Cropper",
                "Volume Cropper is still reading an image into memory. "
                "Please try again when it has finished.")
            return
        layers = selected_image_layers(viewer)
        n = len(viewer.layers)
        answer = crop_gui._choose(
            self, "Use this window?",
            "Volume Cropper will turn this window into its four linked "
            f"views. When an image is opened here, the window's {n} "
            "layer(s) are removed"
            + (" (the selected image layers are copied into Volume Cropper "
               "first)" if layers else "")
            + ", and napari's 2D/3D, roll, transpose and grid buttons stay "
            "hidden until this window is closed. This cannot be undone.\n\n"
            "Or open Volume Cropper in a new window and leave this one as "
            "it is.",
            ["Use this window", "Open a new window instead"])
        if answer is None:
            return
        if answer == "Open a new window instead":
            if layers:
                open_layers_in_new_window(self.viewer, parent=self)
            else:
                crop_gui.open_window()
            return
        try:
            self.viewer.layers.selection.events.changed.disconnect(
                self._show_selection)
        except Exception:
            pass
        # The tool's panel goes next to the views, as in a window the tool
        # opens itself (not into this dock: closing the dock would delete
        # it, and the window could not get it back); this panel goes.
        tool = crop_gui.attach_panel(viewer)
        QTimer.singleShot(0, lambda: _remove_dock(viewer, self))
        if layers:
            tool.open_layers(layers, keep=True)


def _remove_dock(viewer, panel):
    """Take the panel's napari dock out of the window (once its button
    handler has returned)."""
    try:
        viewer.window.remove_dock_widget(panel)
    except (LookupError, RuntimeError):
        pass                              # closed already


def _label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
