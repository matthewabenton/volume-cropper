"""0.16.0 panel, in a VISIBLE window, on the synthetic embryo demo (with a
settings file of its own in OUT_DIR: the user's settings are not touched).

Usage: python panel_0_16_gui_test.py OUT_DIR

Checks: the parts (Image, View, Crop regions, Find a bright object,
Output) open and close and the panel gets shorter; the memory cache is
the first row of View; napari's layer list shows only the channels and
"crop regions"; the polygons part appears with a region's polygons; the
× marks of the XZ and YZ polygons are drawn in green and red in the XY
view; the Settings dialog. Saves panel_open.png, panel_closed.png,
marks.png, settings.png (synthetic data only).
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(OUT / "settings.json")
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = (Path(os.environ.get("VC_DEMO", Path(__file__).resolve().parents[2] / "volume_cropper_demo_data"))
        / "embryo_tissue_demo.zarr")
results = []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1600, 1000)
w.open_image(DEMO, choice=LoadChoice())
pump(3)
check("the parts", sorted(w.sections) == sorted(
    ["image", "view", "regions", "ranges", "find", "output"]),
    str(sorted(w.sections)))
first = w.sections["view"].body.layout().itemAt(0).layout()
check("the memory cache is the first row of View",
      any(first.itemAt(k).widget() is w.cache_spin
          for k in range(first.count())))

# a region with polygons in XZ and YZ: polygons part, coloured marks
i = w._new_box({"z": (200.0, 500.0), "y": (300.0, 700.0),
                "x": (300.0, 700.0)})
xz, yz = w.ortho.views["XZ"].plane, w.ortho.views["YZ"].plane


def poly(plane, r):
    (a0, a1), (b0, b1) = r[plane[0]], r[plane[1]]
    return [(a0, b0), (a0, b1), (a1, b1), (a1, b0)]


r = {"z": (220.0, 480.0), "y": (330.0, 670.0), "x": (330.0, 670.0)}
w.boxes[i].outlines = vc.normalize_outlines(
    {tuple(xz): {500.0: poly(xz, r)}, tuple(yz): {500.0: poly(yz, r)}})
w._refresh(select=i)
w.ortho.cursor["z"] = 350.0
w._refresh(select=i)
pump(1)
check("the polygons part shows for a region with polygons",
      w.polygon_part.isVisible())
segments, colors = w.ortho._crossing_marks(w.boxes[i], w.ortho.views["XY"],
                                           1.0)
want = {tuple(np.round(ov.rgba(ov.PLANE_COLORS[v]), 3)) for v in ("XZ", "YZ")}
got = {tuple(np.round(c, 3)) for c in colors}
check("× marks in the XZ (green) and YZ (red) colours", got == want, str(got))
viewer.screenshot(str(OUT / "marks.png"), canvas_only=False)

# napari's layer list: channels and crop regions only
view = viewer.window._qt_viewer.layers
model = view.model()
listed = []
for row in range(model.rowCount()):
    if not view.isRowHidden(row):
        listed.append(viewer.layers[model.mapToSource(
            model.index(row, 0)).row()].name)
check("napari's layer list: channels and 'crop regions' only",
      "crop regions" in listed and "crosshair" not in listed
      and all(n == "crop regions" or n[:1].isdigit() for n in listed),
      str(listed))

# open and closed
scroll = w.parentWidget().parentWidget()
for key in ("view", "find"):
    w.sections[key].set_open(key != "find")
pump(0.5)
w.grab().save(str(OUT / "panel_open.png"))
tall = w.sizeHint().height()
for key in ("view", "regions"):
    w.sections[key].header.click()
pump(0.5)
w.grab().save(str(OUT / "panel_closed.png"))
check("closing parts makes the panel shorter",
      w.sizeHint().height() < tall, f"{tall} -> {w.sizeHint().height()}")

dialog = crop_gui.SettingsDialog(w)
dialog.show()
pump(0.5)
dialog.grab().save(str(OUT / "settings.png"))
dialog.close()
check("Settings dialog", True)

print(f"\n{sum(results)} of {len(results)} checks passed.")
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
