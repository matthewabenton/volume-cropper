"""Layout on a small screen: splitters, focus, napari panels, Auto level label.
Visible window (scratchpad only)."""
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imageio import imwrite  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402

OUT = "v03_out/"


def pump(s=1.0):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def grab(viewer, name):
    image = viewer.window._qt_window.grab().toImage()
    image.save(OUT + name + ".png")


viewer, w = crop_gui.open_window(show=True)
win = viewer.window._qt_window
win.showNormal()
win.resize(1366, 768)
pump(1)
w.open_image(sys.argv[1])
pump(4)
o = w.ortho
print("window", win.width(), "x", win.height(), "| dock width",
      w.dock.width(), "| grid", o._grid.sizes(), "| rows",
      [c.sizes() for c in o._columns], flush=True)
print("titles", [v.title.text() for v in o.views.values()],
      "| auto:", w.level_combo.itemText(0), flush=True)
grab(viewer, "layout_1_start")

# Zoom in step by step: Auto should move to finer levels, and say so.
cam = ov.camera(o.views["XY"].model)
for factor in (4, 4, 4):
    cam.zoom = cam.zoom * factor
    pump(2)
    print(f"zoom {cam.zoom:.3g}: titles "
          f"{[o.views[n].title.text() for n in ov.VIEWS_2D]} | auto: "
          f"{w.level_combo.itemText(0)}", flush=True)
o.reset_cameras()
pump(2)

# Drag a row border in the left column: the right column must follow.
left, right = o._columns
total = sum(left.sizes())
left.setSizes([int(total * 0.7), total - int(total * 0.7)])
left.splitterMoved.emit(int(total * 0.7), 1)        # as a real drag does
pump(1)
print("rows after drag: left", left.sizes(), "right", right.sizes(), flush=True)
grid_total = sum(o._grid.sizes())
o._grid.setSizes([int(grid_total * 0.75), grid_total - int(grid_total * 0.75)])
pump(1)
print("columns after drag:", o._grid.sizes(), "| panel scrolls sideways:",
      w.dock.widget().horizontalScrollBar().isVisible()
      if hasattr(w.dock.widget(), "horizontalScrollBar") else "?", flush=True)

# Narrower right panel, and napari's left panels off.
win.resizeDocks([w.dock], [220], crop_gui.Qt.Orientation.Horizontal)
w.napari_panels_check.setChecked(False)
pump(2)
print("dock width now", w.dock.width(), "| left docks visible:",
      viewer.window._qt_viewer.dockLayerList.isVisible(),
      "| grid width", sum(o._grid.sizes()), flush=True)
grab(viewer, "layout_2_dragged")

# Focus on XZ, then back.
o._focus_buttons["XZ"].click()
pump(2)
print("focus XZ: visible", {n: v.frame.isVisible() for n, v in o.views.items()},
      "| button", o._focus_buttons["XZ"].text(), flush=True)
grab(viewer, "layout_3_focus_xz")
o._focus_buttons["XZ"].click()
pump(2)
print("back: visible", {n: v.frame.isVisible() for n, v in o.views.items()},
      "| rows", left.sizes(), right.sizes(), "| columns", o._grid.sizes(),
      flush=True)
# 3D off keeps its cell (alignment), 3D on again.
w.show_3d_check.setChecked(False)
pump(2)
print("3D off: rows", left.sizes(), right.sizes(), "| note visible",
      o._3d_off.isVisible(), flush=True)
grab(viewer, "layout_4_3d_off")
w.show_3d_check.setChecked(True)
pump(3)
w.napari_panels_check.setChecked(True)
pump(1)
print("done", flush=True)
viewer.close()
