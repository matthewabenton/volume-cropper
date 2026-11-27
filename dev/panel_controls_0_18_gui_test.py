"""0.18.0 the panel's own controls instead of napari's left panels (the
user's wish of 2026-11-13), in a VISIBLE window on the synthetic demo:
napari's layer panels hidden at first; the Channels list (show, colour,
choose for Levels); Levels' auto-contrast buttons; the drawing tool
buttons under Crop regions (clicked with the mouse).

Usage: python panel_controls_0_18_gui_test.py OUT_DIR

Saves window.png, channels.png, levels.png, regions.png. Synthetic data
only; closes its window; settings file of its own in OUT_DIR.
"""
import os
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(OUT / "settings.json")
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data" / "embryo_tissue_demo.zarr"
results = []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


load_dialog.FileLoadDialog.exec = lambda self: 1
viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
w.open_image(DEMO, choice=LoadChoice())
pump(3.0)
qt_viewer = viewer.window._qt_viewer
check("napari's layer panels are hidden at first",
      not qt_viewer.dockLayerList.isVisible()
      and not qt_viewer.dockLayerControls.isVisible()
      and not w.napari_panels_check.isChecked())
check("the Channels list has a row per channel",
      w.channels.list.count() == 3 and w.channels.isVisible())
# a click on a row chooses the channel for Levels
rect = w.channels.list.visualItemRect(w.channels.list.item(1))
QTest.mouseClick(w.channels.list.viewport(), Qt.MouseButton.LeftButton,
                 Qt.KeyboardModifier.NoModifier, rect.center())
pump(0.5)
check("a click on a channel chooses it for Levels",
      w.levels.channel == 1, w.levels.channel_name.text())
# colour menu: magenta for channel 2
combo = w.channels.colour_combos[2]
combo.setCurrentIndex(combo.findData("yellow"))
combo.activated.emit(combo.currentIndex())
pump(0.5)
layer = w.ortho.views["XY"].channel_layers[2]
check("the colour menu sets the channel's colour in all views",
      layer.colormap.name == "yellow"
      and w.ortho.views["XZ"].channel_layers[2].colormap.name == "yellow")
# auto-contrast once on channel 1
layer1 = w.ortho.views["XY"].channel_layers[1]
layer1.contrast_limits = (0, 20)
pump(0.3)
QTest.mouseClick(w.levels.auto_once, Qt.MouseButton.LeftButton)
pump(0.5)
check("Auto-contrast once sets min and max from the view",
      tuple(layer1.contrast_limits) != (0.0, 20.0),
      f"{tuple(round(v) for v in layer1.contrast_limits)}")
# the drawing tools, clicked
QTest.mouseClick(w.tool_buttons["add_rectangle"], Qt.MouseButton.LeftButton)
pump(0.3)
modes = {str(w.ortho.views[n].shapes.mode) for n in ov.VIEWS_2D}
check("a tool button sets the tool in all 2D views",
      modes == {"add_rectangle"}
      and w.tool_buttons["add_rectangle"].isChecked(), f"{modes}")
QTest.mouseClick(w.tool_buttons["pan_zoom"], Qt.MouseButton.LeftButton)
pump(0.5)
viewer.screenshot(str(OUT / "window.png"), canvas_only=False)
for key, name in (("channels", "channels"), ("levels", "levels"),
                  ("regions", "regions")):
    w.sections[key].grab().save(str(OUT / f"{name}.png"))
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
