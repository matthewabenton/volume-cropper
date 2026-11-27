"""0.18.0 panel changes of 2026-11-12, in a VISIBLE window, on the synthetic
embryo demo.

Usage: python panel_0_18_gui_test.py OUT_DIR

Opens volume_cropper_demo_data/embryo_tissue_demo.zarr (synthetic, read
only) and checks the user's three wishes of 2026-11-12:
1. the "Loading…" box: it appears when the image opens (after "What to
   load", here bypassed), for a new 2D resolution, the 3D view and a
   projection; it closes by itself afterwards and the busy pointer goes;
2. the crop preview on a maximum projection shows the projection of
   exactly what the selected region's crop keeps (compared with
   crop_projection on the same region), with its level under Preview crop;
3. several regions selected with real Ctrl+clicks are deleted at once with
   the Delete key (after the question, answered here); several polygons
   with the "Delete selected" button.
Saves loading_notice.png (the box, as it shows for a new resolution),
preview_mip_off.png (napari's projection through the whole depth),
preview_mip_exact.png (the exact preview), regions_selected.png (the
region list with three selected). Synthetic data only. Answers every dialog
itself and closes its window. Uses a settings file of its own in OUT_DIR.
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
from qtpy.QtCore import QPoint, Qt  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import crop_projection as cp  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data" / "embryo_tissue_demo.zarr"
results, messages, asked = [], [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def wait_for(condition, seconds=15.0):
    end = time.time() + seconds
    while time.time() < end:
        pump(0.1)
        if condition():
            return True
    return False


def notice_done():
    notice = w._loading_notice
    return (notice is not None and not notice.isVisible()
            and QApplication.overrideCursor() is None)


def loading_check(name, before):
    """The box was shown with a new text, and closes by itself."""
    texts = w._loading_notice.texts[before:] if w._loading_notice else []
    visible = w._loading_notice is not None and w._loading_notice.isVisible()
    t0 = time.time()
    closed = wait_for(notice_done, 30)
    check(f"loading box: {name}", texts and visible and closed,
          f"{texts[:1]}, closed after {time.time() - t0:.1f} s")


def shown_count():
    return len(w._loading_notice.texts) if w._loading_notice else 0


load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
for kind in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, kind, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: asked.append(a[2]) or True

viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)

# --- 1. the loading box ------------------------------------------------------
w.open_image(DEMO, choice=LoadChoice())
loading_check("opening the image", 0)
n = shown_count()
w.level_combo.setCurrentIndex(2)          # level 1 (0 is "Auto")
w._on_level_2d()
# (grabbing the screen gives black over napari's OpenGL views: the box
# alone, as it shows now)
w._loading_notice.grab().save(str(OUT / "loading_notice.png"))
loading_check("2D resolution level 1", n)
n = shown_count()
w.show_3d_check.setChecked(True)          # its handler runs at once
loading_check("3D view on", n)
w.show_3d_check.setChecked(False)
pump(0.5)
w.level_combo.setCurrentIndex(0)          # back to Auto
w._on_level_2d()
wait_for(notice_done)

# --- 2. the exact projection in the crop preview ------------------------------
ortho = w.ortho
centre = {a: float(np.mean(ortho.extent(a))) for a in ("z", "y", "x")}
ranges = {"z": (centre["z"] - 200, centre["z"] + 200),
          "y": (centre["y"] - 250, centre["y"] + 250),
          "x": (centre["x"] - 250, centre["x"] + 250)}
i = w._new_box(ranges)
cy, cx = centre["y"], centre["x"]
hexagon = [(cy + 230 * np.sin(a), cx + 230 * np.cos(a))
           for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
w.boxes[i].outlines = vc.normalize_outlines({("y", "x"): hexagon})
w._refresh(select=i)
w.projection_combo.setCurrentIndex(w.projection_combo.findData("max"))
n = shown_count()
w._on_projection()
loading_check("maximum projection", n)
pump(1.0)
viewer.screenshot(str(OUT / "preview_mip_off.png"), canvas_only=False)
w.preview_combo.setCurrentIndex(w.preview_combo.findData("main"))
ortho.set_preview(True)
ok = wait_for(lambda: "voxels per channel" in w.preview_info.text(), 30)
pump(1.0)
check("preview: the crop's projection is computed", ok,
      w.preview_info.text())
shown = [k for k, d in enumerate(ortho.display) if d["visible"]]
expected = cp.crop_projection(w.image, ortho.region, shown,
                              ortho.time_shown(), ortho.box_ranges[i],
                              ortho.boxes[i], "max")
same = all(np.array_equal(
    np.asarray(ortho.views[name].crop_projection_layers[k].data),
    expected.plane(k, ortho.views[name].order[-3], ortho.views[name].plane))
    for name in ("XY", "XZ", "YZ") for k in shown)
check("preview: every 2D view shows the crop's own projection", same,
      f"channels {shown}, level {expected.level}")
outside_hexagon = (expected.planes[(shown[0], "z")] == 0).mean()
check("preview: the corners outside the hexagon are 0 (as cropped)",
      outside_hexagon > 0.05, f"{outside_hexagon:.0%} of the XY projection")
kept = all(ortho.views[name].model.layers.selection.active
           is ortho.views[name].shapes for name in ("XY", "XZ", "YZ"))
check("preview: the regions' layer stays active (drawing works)", kept)
viewer.screenshot(str(OUT / "preview_mip_exact.png"), canvas_only=False)
ortho.set_preview(False)
w.preview_combo.setCurrentIndex(0)
w.projection_combo.setCurrentIndex(0)
w._on_projection()
wait_for(notice_done)

# --- 3. several regions and polygons at once --------------------------------
for k in range(3):
    w._new_box({"y": (100.0 + 50 * k, 300.0), "x": (100.0, 300.0)})
w._refresh(select=0)
regions = w.box_list
w.sections["regions"].set_open(True)
pump(0.5)
regions.scrollToTop()


def click(row, modifier=Qt.KeyboardModifier.NoModifier):
    rect = regions.visualItemRect(regions.item(row))
    QTest.mouseClick(regions.viewport(), Qt.MouseButton.LeftButton, modifier,
                     rect.center())
    pump(0.2)


click(1)
click(2, Qt.KeyboardModifier.ControlModifier)
click(3, Qt.KeyboardModifier.ControlModifier)
rows = sorted(regions.row(item) for item in regions.selectedItems())
check("Ctrl+click selects several regions; the last one is the selected one",
      rows == [1, 2, 3] and w._selected() == 3
      and regions.item(3).font().bold(), f"rows {rows}, "
      f"selected {w._selected()}")
w.sections["regions"].grab().save(str(OUT / "regions_selected.png"))
names_before = [b.name for b in w.boxes]
regions.setFocus()
pump(0.2)
asked.clear()
QTest.keyClick(regions, Qt.Key.Key_Delete)
pump(0.5)
check("the Delete key deletes them, after asking",
      len(asked) == 1 and [b.name for b in w.boxes] == [names_before[0]],
      f"asked {asked}, left {[b.name for b in w.boxes]}")

# polygons on three slices of the remaining region
box = w.boxes[0]
square = [(120.0, 120.0), (120.0, 280.0), (280.0, 280.0), (280.0, 120.0)]
zs = [centre["z"] - 100, centre["z"], centre["z"] + 100]
box.outlines = {("y", "x"): {z: list(square) for z in zs}}
w._refresh(select=0)
pump(0.3)
polygons = w.polygon_list
polygons.item(0).setSelected(True)
polygons.item(1).setSelected(True)
asked.clear()
w._delete_polygon()
pump(0.3)
left = list(box.outlines.get(("y", "x"), {}))
check("two polygons selected are deleted at once, after asking",
      len(asked) == 1 and len(left) == 1 and polygons.count() == 1,
      f"left {left}")

print("dialogs answered:", messages)
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
