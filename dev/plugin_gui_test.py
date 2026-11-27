"""0.13.0 napari plugin, in VISIBLE windows: Volume Cropper used from a
plain napari window, as colleagues would. Run in the environment where the
plugin is installed (napari-plugin-test; plugin_install_test.py installs
it from the experimental folder).

Usage: python plugin_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)

Checks: napari finds the plugin and starts without importing the tool;
File > Open Sample gives the synthetic demo; Plugins > Volume Cropper is a
small panel that leaves the window alone (layers, napari's buttons, no
private-API warnings); its "Crop the selected layers" opens a new Volume
Cropper window with the three layers as one image; a crop there is a
checked zstd OME-TIFF; Layers > Data > Crop selected layers… is greyed
out without a selection and works through napari's menu machinery (napari
hands it the viewer); File > IO Utilities > Open in a new window; "Use
this window…" (warning answered) turns a second napari window into the
tool; closing every window ends without errors. Saves plugin_panel.png
and plugin_tool_window.png (synthetic data only). Answers every dialog
itself and closes its windows.
"""
import faulthandler
import sys
import time
import warnings
from pathlib import Path

faulthandler.enable()
import napari  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
import shutil  # noqa: E402

shutil.rmtree(OUT / "crops", ignore_errors=True)   # this script's own crops
results, notes, messages = [], [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def menu_action(viewer, text_start):
    """The QAction in the window's menus whose text starts so (napari adds
    "(Volume Cropper)"), after the Layers menu was told it is about to be
    shown: napari updates which entries (also in its submenus) are greyed
    out only then, as when the user opens it."""
    from qtpy.QtWidgets import QMenu
    viewer.window.layers_menu.aboutToShow.emit()
    QApplication.instance().processEvents()
    for menu in viewer.window._qt_window.findChildren(QMenu):
        for action in menu.actions():
            if action.text().replace("&", "").startswith(text_start):
                return action
    return None


# --- napari finds the plugin -----------------------------------------------------------------
from npe2 import PluginManager  # noqa: E402

pm = PluginManager.instance()
pm.discover()
manifest = pm.get_manifest("napari-volume-cropper")
check("npe2 finds the plugin", manifest is not None,
      f"version {manifest.package_version}")

caught = []
warnings.showwarning = lambda message, *a, **k: caught.append(str(message))
warnings.simplefilter("always", FutureWarning)

viewer = napari.Viewer(title="napari (Volume Cropper plugin test)")
viewer.window._qt_window.resize(1500, 950)
pump(1)
check("napari starts without importing the tool",
      "volume_cropper.crop_gui" not in sys.modules and "volume_cropper" not in sys.modules)

viewer.open_sample("napari-volume-cropper", "demo")
pump(1)
names = [layer.name for layer in viewer.layers]
check("File > Open Sample > Volume Cropper: 3 layers",
      names == ["nuclei", "membranes", "marker"], str(names))

# --- Layers > Data, greyed out without a selection --------------------------------------
viewer.layers.selection.clear()
pump(0.5)
action = menu_action(viewer, "Crop selected layers")
greyed = action is not None and not action.isEnabled()
viewer.layers.selection.update(set(viewer.layers))
pump(0.5)
action = menu_action(viewer, "Crop selected layers")
check("Layers > Data > Crop selected layers…: greyed out without a "
      "selection, enabled with one",
      greyed and action is not None and action.isEnabled(),
      "no such menu entry" if action is None else
      f"{action.text()}: without a selection enabled={not greyed}, with "
      f"one enabled={action.isEnabled()}")

# --- Plugins > Volume Cropper: the small panel ------------------------------------------------
dock, panel = viewer.window.add_plugin_dock_widget(
    "napari-volume-cropper", "Volume Cropper")
pump(1.5)
from napari_volume_cropper import _tool  # noqa: E402

crop_gui = _tool.import_tool()
from volume_cropper import load_dialog  # noqa: E402

load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
for name in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, name, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True
buttons = viewer.window._qt_viewer.viewerButtons
check("the panel shows the selection; napari's buttons are untouched",
      "nuclei, membranes, marker" in panel.selection_label.text()
      and buttons.ndisplayButton.isVisible()
      and panel.crop_button.isEnabled(), panel.selection_label.text())
viewer.screenshot(str(OUT / "plugin_panel.png"), canvas_only=False)

windows = len(crop_gui._WINDOWS)
panel.crop_button.click()
pump(3)
ok = len(crop_gui._WINDOWS) == windows + 1
tool_viewer, w = crop_gui._WINDOWS[-1] if ok else (None, None)
image = w.image if w is not None else None
check("Crop the selected layers: a new window with one 3-channel image",
      image is not None and image.levels[0].array.shape == (3, 64, 256, 256)
      and image.channel_labels == names
      and np.allclose(image.levels[0].scale[-3:], [2.0, 0.5, 0.5]),
      "" if image is None else f"{image.levels[0].array.shape}, "
      f"{image.channel_labels}")
check("the user's window is as it was (layers, buttons)",
      [layer.name for layer in viewer.layers] == names
      and buttons.ndisplayButton.isVisible()
      and tool_viewer is not viewer)

# --- a crop in the new window ----------------------------------------------------------------
if w is not None:
    from volume_cropper import core as vc
    ranges = {a: image.extent(a) for a in image.range_axes}
    ranges.update(z=(20.0, 80.0), y=(20.0, 80.0), x=(30.0, 100.0))
    w.boxes = [vc.CropBox("blob", ranges, channels=[0, 2])]
    w._refresh(select=0)
    w.out_edit.setText(str(OUT / "crops"))
    w.compress_check.setChecked(True)
    tool_viewer.window._qt_window.resize(1500, 950)
    pump(1.5)
    tool_viewer.screenshot(str(OUT / "plugin_tool_window.png"),
                           canvas_only=False)
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
            pixels = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"][
                "Pixels"]
            ok = ok and [c["Name"] for c in pixels["Channel"]] == [
                "nuclei", "marker"]
            ok = ok and tif.pages.first.compression.name == "ZSTD"
    check("the crop: an OME-TIFF (zstd), channels nuclei + marker, checked",
          ok, w.status.text().replace("\n", " | ")[:160])
    tool_viewer.close()
    pump(1)
    check("closing the tool window forgets it",
          not crop_gui.is_tool_viewer(tool_viewer))

# --- Layers > Data through napari's menu machinery ------------------------------------------
from napari._app_model import get_app_model  # noqa: E402

app = get_app_model()
windows = len(crop_gui._WINDOWS)
viewer.window._qt_window.activateWindow()
pump(0.5)
app.commands.execute_command(
    "napari-volume-cropper.crop_selected_layers").result()
pump(3)
ok = len(crop_gui._WINDOWS) == windows + 1
check("Layers > Data > Crop selected layers… (napari passes the viewer)",
      ok and crop_gui._WINDOWS[-1][1].image is not None, str(messages[-1:]))
if ok:
    crop_gui._WINDOWS[-1][0].close()
    pump(1)

# --- File > IO Utilities > Open in a new window -----------------------------------------------
windows = len(crop_gui._WINDOWS)
app.commands.execute_command("napari-volume-cropper.new_window").result()
pump(2)
ok = len(crop_gui._WINDOWS) == windows + 1
check("File > IO Utilities > Open in a new window: an empty tool window",
      ok and crop_gui._WINDOWS[-1][1].image is None)
if ok:
    crop_gui._WINDOWS[-1][0].close()
    pump(1)

private = [m for m in caught if "Private attribute access" in m]
check("no private-API warnings in the user's window", not private,
      private[0][:120] if private else "")

# --- Use this window… in a second napari window -----------------------------------------------
second = napari.Viewer(title="napari (second window)")
second.open_sample("napari-volume-cropper", "demo")
second.layers.selection.update({second.layers["nuclei"],
                                second.layers["marker"]})
pump(1)
dock2, panel2 = second.window.add_plugin_dock_widget(
    "napari-volume-cropper", "Volume Cropper")
asked = []
crop_gui._choose = lambda *a, **k: asked.append(a[2]) or "Use this window"
panel2._use_this_window()
pump(3)
tools = [w for v, w in crop_gui._WINDOWS if v is second]
tool = tools[0] if tools else None
from qtpy.QtWidgets import QDockWidget  # noqa: E402

docks = second.window._qt_window.findChildren(QDockWidget)
check("Use this window…: warns, then this window is the tool (panel next "
      "to the views, the small panel's dock gone), with the selected layers",
      bool(asked) and "cannot be undone" in asked[0]
      and tool is not None and tool.isVisible()
      and tool.image is not None
      and tool.image.channel_labels == ["nuclei", "marker"]
      and not any(d.widget() is panel2 for d in docks)
      and not second.window._qt_viewer.viewerButtons.ndisplayButton
      .isVisible())
second.screenshot(str(OUT / "plugin_use_this_window.png"), canvas_only=False)

print(f"\n{sum(results)} of {len(results)} checks passed.")
for note in notes:
    print("NOTE:", note)
if messages:
    print("messages:", messages)
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
