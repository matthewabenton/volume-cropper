"""The plugin's menu commands (see napari.yaml).

- Layers > Data > Crop selected layers… (Volume Cropper): the selected
  image layers in a new Volume Cropper window.
- File > IO Utilities > Open in a new window (Volume Cropper): an empty
  Volume Cropper window.

Both leave the user's own window as it is: Volume Cropper always opens
its own window (crop_gui.open_window), with its four linked views. napari
hands the commands a "public-only" proxy of the current viewer; they use
only napari's public parts of it (its layers and their selection), so
napari gives no warnings.
"""
import napari
from qtpy.QtWidgets import QApplication, QMessageBox

from ._tool import import_tool


def selected_image_layers(viewer) -> list:
    """The selected image layers, in the layer list's order (the order the
    user sees; the selection itself has none)."""
    from napari.layers import Image
    selection = viewer.layers.selection
    return [layer for layer in viewer.layers
            if layer in selection and isinstance(layer, Image)]


def crop_selected_layers(viewer: napari.viewer.Viewer):
    """Layers > Data > Crop selected layers…"""
    open_layers_in_new_window(viewer)


def open_layers_in_new_window(viewer, parent=None) -> bool:
    """Open the viewer's selected image layers in a new Volume Cropper
    window ("What to load" first). Returns True if they were opened; the
    new window is closed again if not (e.g. Cancel)."""
    crop_gui = import_tool()
    parent = parent or QApplication.activeWindow()
    if crop_gui.is_tool_viewer(viewer):
        QMessageBox.information(
            parent, "Volume Cropper",
            "This is a Volume Cropper window: its layers are the tool's "
            "own. Use 'Crop selected layers' in a napari window with your "
            "image layers.")
        return False
    if crop_gui.is_reading():
        QMessageBox.information(
            parent, "Volume Cropper",
            "Volume Cropper is still reading an image into memory. Please "
            "try again when it has finished.")
        return False
    layers = selected_image_layers(viewer)
    if not layers:
        QMessageBox.information(
            parent, "Volume Cropper",
            "Select one image layer (or several of the same size: the "
            "channels of one image) in napari's layer list first.")
        return False
    new_viewer, panel = crop_gui.open_window()
    if panel.open_layers(layers):
        return True
    new_viewer.close()             # nothing opened: no empty window left
    return False


def open_in_new_window():
    """File > IO Utilities > Open in a new window: an empty Volume Cropper
    window (napari's event loop already runs; napari.run() is not called)."""
    crop_gui = import_tool()
    crop_gui.open_window()
