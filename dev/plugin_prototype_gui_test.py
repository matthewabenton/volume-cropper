"""Plugin prototype (branch plugin-prototype), in a VISIBLE window: does
Volume Cropper work as a napari plugin? Run in the environment where the
prototype is installed (napari-plugin-test, pip install -e).

Usage: python plugin_prototype_gui_test.py OUT_DIR [TIFF]
Checks: napari finds the plugin (npe2) and starts without importing the
tool; File > Open Sample gives the synthetic demo (3 layers); Plugins >
Volume Cropper (add_plugin_dock_widget) turns the window into the tool
(four views, panel docked right); the three demo layers, selected in
napari's layer list, are read as the channels of one image ("Image layers
selected in napari"), with their scale and names; a crop is saved as an
OME-TIFF and checked; a TIFF file (optional) opens in the plugin too.
Notes what the tool does to a user's window (layers removed, napari's
buttons hidden). Saves plugin_window.png (synthetic data only). Answers
every dialog itself and closes its windows.
"""
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable()
import napari  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1])
TIFF = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT.mkdir(parents=True, exist_ok=True)
results, notes, messages = [], [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


# --- napari finds the plugin ---------------------------------------------------------
from npe2 import PluginManager  # noqa: E402

pm = PluginManager.instance()
pm.discover()
manifest = pm.get_manifest("napari-volume-cropper")
check("npe2 finds the plugin: 1 widget, 1 sample, 3 commands, 1 menu",
      len(manifest.contributions.widgets) == 1
      and len(manifest.contributions.sample_data) == 1
      and len(manifest.contributions.commands) == 3
      and len(manifest.contributions.menus) == 1,
      f"version {manifest.package_version}")

viewer = napari.Viewer(title="napari (plugin prototype test)")
viewer.window._qt_window.resize(1600, 1000)
pump(1)
check("napari starts without importing the tool",
      "crop_gui" not in sys.modules and "ortho_views" not in sys.modules)

# --- Sample data ---------------------------------------------------------------------------
viewer.open_sample("napari-volume-cropper", "demo")
pump(1)
names = [layer.name for layer in viewer.layers]
check("File > Open Sample > Volume Cropper demo: 3 layers",
      names == ["nuclei", "membranes", "marker"], str(names))

# --- Plugins > Volume Cropper ----------------------------------------------------------------
from volume_cropper import load_dialog  # noqa: E402

load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
import warnings  # noqa: E402

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    dock, panel = viewer.window.add_plugin_dock_widget(
        "napari-volume-cropper", "Volume Cropper")
    pump(2)
private = [str(w_.message)[:90] for w_ in caught
           if "Private attribute access" in str(w_.message)]
notes.append(f"private-API warnings while the dock was built: {len(private)}"
             + (f" (first: {private[0]})" if private else ""))
from napari_volume_cropper import _widget  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402

crop_gui.QMessageBox.warning = staticmethod(
    lambda *a, **k: messages.append(a[1:3]))
crop_gui.QMessageBox.critical = staticmethod(
    lambda *a, **k: messages.append(a[1:3]))
crop_gui._confirm = lambda *a: True
w = _widget._PANELS[id(viewer)]
check("the widget turns the window into the tool: four views, panel docked",
      set(w.ortho.views) == {"XY", "XZ", "YZ", "3D"}
      and dock.isVisible() and panel.widget() is w)
buttons = viewer.window._qt_viewer.viewerButtons
notes.append("napari's 2D/3D, roll, transpose and grid buttons hidden: "
             + str(not buttons.ndisplayButton.isVisible()))
check("the demo layers are still there before an image is opened",
      [layer.name for layer in viewer.layers][:3] == names)

# --- The selected layers as one image ------------------------------------------------------------
viewer.layers.selection.clear()
viewer.layers.selection.update({viewer.layers[n] for n in names})
w._open_layers()
pump(3)
image = w.image
ok = (image is not None and image.in_memory
      and image.levels[0].array.shape == (3, 64, 256, 256)
      and np.allclose(image.levels[0].scale[-3:], [2.0, 0.5, 0.5])
      and image.unit("x") == "micrometer"
      and image.channel_labels == names)
check("selected layers -> one 3-channel image in memory, scale and names kept",
      ok, "" if image is None else
      f"{image.levels[0].array.shape}, {image.levels[0].scale}, "
      f"{image.channel_labels}, {image.unit('x')}")
left = [layer.name for layer in viewer.layers]
notes.append(f"after opening, the window's layers are the tool's own: the "
             f"demo layers were removed: {not set(names) & set(left)}")

# --- A crop -------------------------------------------------------------------------------------------
from volume_cropper import core as vc  # noqa: E402

ranges = {a: image.extent(a) for a in image.range_axes}
ranges.update(z=(20.0, 80.0), y=(20.0, 80.0), x=(30.0, 100.0))
w.boxes = [vc.CropBox("blob", ranges, channels=[0, 2])]
w._refresh(select=0)
w.out_edit.setText(str(OUT / "crops"))
w.compress_check.setChecked(True)
pump(1)
viewer.screenshot(str(OUT / "plugin_window.png"), canvas_only=False)
w._crop(False)
end = time.time() + 60
pump(0.5)
while time.time() < end and not w.crop_selected_button.isEnabled():
    pump(0.2)
pump(0.5)
out = OUT / "crops" / "nuclei_blob.ome.tif"
ok = out.exists() and "checked OK" in w.status.text()
if out.exists():
    with tifffile.TiffFile(out) as tif:
        pixels = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]["Pixels"]
        channels = pixels["Channel"]
        ok = ok and [c["Name"] for c in channels] == ["nuclei", "marker"]
        ok = ok and tif.pages.first.compression.name == "ZSTD"
check("crop of the layers: an OME-TIFF (zstd), channels nuclei + marker, "
      "checked OK", ok, w.status.text().replace("\n", " | ")[:160])

# --- A TIFF file in the plugin ------------------------------------------------------------------------
if TIFF is not None and TIFF.exists():
    from volume_cropper import file_images as fi  # noqa: E402
    w._may_discard_boxes = lambda action: True
    w.open_image(TIFF, file_choice=fi.FileChoice())
    pump(3)
    check("a TIFF file opens in the plugin", w.image is not None
          and w.image.path == TIFF, w.read_info.text())

# --- File > IO Utilities > Open Volume Cropper in a new window ---------------------------------
before = len(crop_gui._WINDOWS)
pm.get_command("napari-volume-cropper.new_window").exec()
pump(3)
new_viewer = crop_gui._WINDOWS[-1][0] if len(crop_gui._WINDOWS) > before \
    else None
check("the menu command opens the tool in a window of its own",
      new_viewer is not None and new_viewer is not viewer
      and new_viewer.window._qt_window.isVisible())

print(f"\n{sum(results)} of {len(results)} checks passed.")
for note in notes:
    print("NOTE:", note)
if messages:
    print("messages:", messages)
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
