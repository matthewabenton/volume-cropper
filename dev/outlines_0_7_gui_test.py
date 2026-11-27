"""0.7.0 outlines in 3D, checked in a VISIBLE window: polygons on several
slices and in several views go into the selected crop region (a new one if
none is selected); New region, Finish region (deselects), Delete, a click
on an empty part of the list; one 3D shape whose cross-section every view
draws (dashed) on its slice, with marks where the other views' polygons
cross it; the warning when views disagree; a silhouette drawn on a MIP; the
polygon list (go to, delete), the shading of the kept / set-to-0 voxels
(colours, opacity), the margin, editing and deleting single polygons, and a
crop.

Usage: python outlines_0_7_gui_test.py OUT_DIR   (OUT_DIR: a scratch folder)
Tests the folder in VC_TOOL (default: the tool folder this script is in). Saves shading.png and
margin.png.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

faulthandler.enable()
sys.path.insert(0, os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from napari.layers.base._base_constants import ActionType  # noqa: E402
from qtpy.QtCore import QEvent, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QMouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
results, messages = [], []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


for kind in ("warning", "critical", "information"):
    setattr(crop_gui.QMessageBox, kind,
            lambda parent, title, text, *a, **k: messages.append((title, text)))

src = OUT / "src.zarr"
if not src.exists():
    make_test_image(src)        # z 0-80 um (2 um voxels), y 0-37.5, x 0-32.5
viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(src)
pump(2)
xy, xz, yz = (w.ortho.views[name] for name in ("XY", "XZ", "YZ"))


def draw(view, points):
    """As if drawn with the polygon tool on the view's current slice."""
    view.shapes.mode = "add_polygon"
    view.shapes.add([np.asarray(points, dtype=float)], shape_type="polygon")
    pump(0.6)


def click_panel():
    widget = w.polygon_list.viewport()
    for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
        event = QMouseEvent(kind, QPointF(5, 5), QPointF(5, 5),
                            Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, event)
    pump(0.3)


# 1. Polygons on two XY slices and in XZ go to one box.
P20 = [(8.1, 6.1), (8.1, 20.3), (24.1, 13.2)]                  # (y, x)
P40 = [(14.3, 12.1), (14.3, 26.2), (28.4, 26.2), (28.4, 12.1)]
PXZ = [(14.1, 4.1), (14.1, 28.1), (46.1, 28.1), (46.1, 4.1)]     # (z, x)
w.ortho.go_to_slice(("y", "x"), 20.0)
pump(0.5)
draw(xy, P20)
w.ortho.go_to_slice(("y", "x"), 40.0)
pump(0.5)
draw(xy, P40)
w.ortho.go_to_slice(("z", "x"), 18.0)
pump(0.5)
t0 = time.perf_counter()
draw(xz, PXZ)
print(f"  third polygon drawn: the views were updated in "
      f"{time.perf_counter() - t0:.1f} s (including 0.6 s of waiting)")
box = w.boxes[0] if w.boxes else None
check("three polygons, one region", len(w.boxes) == 1 and box is not None
      and sorted(box.outlines.get(("y", "x"), {})) == [20.0, 40.0]
      and list(box.outlines.get(("z", "x"), {})) == [18.0],
      f"{len(w.boxes)} boxes, {box.outlines.keys() if box else ''}")
check("status names the region", "Region 'crop01'" in w.status.text(),
      w.status.text())

# 2. Finish region: pan/zoom tool, nothing selected; the next polygon then
# starts a new region. Delete; New region; Finish with nothing drawn; a
# click on an empty part of the list.
w._finish_region()
pump(0.3)
check("Finish region: pan/zoom, nothing selected",
      xy.shapes.mode == "pan_zoom" and w._selected() == -1,
      f"{xy.shapes.mode}, {w._selected()}")
draw(xy, P20)
check("then a polygon starts a new region",
      len(w.boxes) == 2 and w._selected() == 1, f"{len(w.boxes)} regions")
w._remove_box()
pump(0.3)
check("Delete: removed, nothing selected",
      len(w.boxes) == 1 and w._selected() == -1)
w._new_region()
pump(0.3)
check("New region: selected, polygon tool on",
      len(w.boxes) == 2 and w._selected() == 1
      and xy.shapes.mode == "add_polygon", xy.shapes.mode)
w._finish_region()
pump(0.3)
check("Finish with nothing drawn: the new region is gone", len(w.boxes) == 1)
w._refresh(select=0)
pump(0.3)
port = w.box_list.viewport()
for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
    spot = QPointF(5, port.height() - 4)
    QApplication.sendEvent(port, QMouseEvent(
        kind, spot, spot, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
pump(0.3)
check("a click on an empty part of the list: none selected",
      w._selected() == -1)
w._refresh(select=0)
pump(0.5)

# 3. The shapes and dashed lines follow the slice.
w.ortho.go_to_slice(("y", "x"), 20.0)
pump(0.5)
check("on a key slice: the polygon is the shape",
      xy.shape_keys == [20.0], str(xy.shape_keys))
w.ortho.go_to_slice(("y", "x"), 30.0)
pump(0.5)
pieces = sum(len(layer.data) for layer in xy.outline_layers.values())
check("between key slices: the rectangle, and a dashed outline",
      xy.shape_keys == [ov.RECT] and pieces > 4,
      f"{xy.shape_keys}, {pieces} line pieces")
w.ortho.go_to_slice(("y", "x"), 50.0)
pump(0.5)
pieces = sum(len(layer.data) for layer in xy.outline_layers.values())
check("after the last polygon: no outline", pieces <= 4, f"{pieces} pieces")

# 3b. One 3D shape: XZ, off its polygon's slice, and YZ, which has no
# polygon at all, also draw its cross-section (dashed).
w.ortho.go_to_slice(("y", "x"), 30.0)
w.ortho.go_to_slice(("z", "x"), 22.0)
w.ortho.go_to_slice(("y", "z"), 16.0)
pump(0.8)
pieces_xz = sum(len(layer.data) for layer in xz.outline_layers.values())
pieces_yz = sum(len(layer.data) for layer in yz.outline_layers.values())
check("XZ off its polygon's slice, and YZ: the shape's outline",
      xz.shape_keys == [ov.RECT] and pieces_xz > 4 and pieces_yz > 4,
      f"{pieces_xz} and {pieces_yz} line pieces")
# XY at z = 30: the XZ polygon (at y = 18, z 14-46) crosses it at x = 4.1
# and 28.1: two crosses of two strokes each.
# (since 0.16.0 _crossing_marks gives (segments, colours); with the old
# tool, VC_TOOL = an older copy, it gives the segments alone)
marks = w.ortho._crossing_marks(box, xy, w.ortho._zoom(xy))
if isinstance(marks, tuple):
    marks = marks[0]
marks = np.asarray(marks)
centres = np.round(marks.mean(axis=1), 1)
check("marks where the XZ polygon crosses the XY slice",
      len(marks) == 4 and {tuple(c) for c in centres} == {(18.0, 4.1),
                                                         (18.0, 28.1)},
      str(sorted({tuple(c) for c in centres})))
# The XZ polygon is much wider than the XY ones where they cross: warned.
check("warning: the views disagree", "⚠ " in w.plan_label.text()
      and "disagree by" in w.plan_label.text(), w.plan_label.text()[-80:])

# 3c. On a MIP, a polygon is a silhouette through the projected depth; it
# is edited there, not on slices.
w.ortho.set_projection("max", "full")
pump(1)
PSIL = [(6.1, 4.1), (6.1, 28.4), (30.2, 28.4), (30.2, 4.1)]
draw(xy, PSIL)
silhouettes = [k for k in box.outlines.get(("y", "x"), {})
               if isinstance(k, tuple)]
check("MIP: a silhouette through the projected depth",
      len(silhouettes) == 1 and xy.shape_keys == silhouettes
      and any("projection" in w.polygon_list.item(k).text()
              for k in range(w.polygon_list.count())),
      f"{silhouettes}, {xy.shape_keys}")
w.ortho.set_projection("none", "full")
pump(1)
w.ortho.go_to_slice(("y", "x"), 40.0)
pump(0.5)
check("back on slices: the polygon on this slice is edited",
      xy.shape_keys == [40.0], str(xy.shape_keys))
crop_gui._remove_polygon(box, ("y", "x"), silhouettes[0])
crop_gui._fit_to_outlines(box, w.image)
w._refresh(select=0)
w.ortho.go_to_slice(("y", "x"), 30.0)
pump(0.5)

# 4. Shading: kept and set-to-0 voxels on the slice, colours, opacity.
w.shade_check.setChecked(True)
w.ortho.go_to_slice(("y", "x"), 30.0)
pump(0.8)
rgba = np.asarray(xy.shading.data)
kept = (rgba[..., :3] == ov.hex_to_rgb(ov.SHADE_KEPT)).all(-1)
zeroed = (rgba[..., :3] == ov.hex_to_rgb(ov.SHADE_ZEROED)).all(-1)
check("shading: orange kept and sky-blue set-to-0 voxels", xy.shading.visible
      and kept.any() and zeroed.any() and (kept | zeroed).all(),
      f"{kept.sum()} kept, {zeroed.sum()} set to 0")
viewer.window._qt_window.grab().save(str(OUT / "shading.png"))
w.ortho.set_shading(kept="#00ff00", opacity=0.8)
pump(0.5)
rgba = np.asarray(xy.shading.data)
check("shading: colour and opacity follow the settings",
      ((rgba[..., :3] == (0, 255, 0)).all(-1)).any()
      and rgba[..., 3].max() == round(255 * 0.8))
w.ortho.go_to_slice(("y", "x"), 10.0)
pump(0.5)
check("shading: none outside the crop", not xy.shading.visible)

# 4b. The outline margin: 4 voxels (4 x 0.25 um) grows the box, the
# dashed outline and the kept voxels; the plan text says so.
w.ortho.set_shading(kept=ov.SHADE_KEPT, opacity=ov.SHADE_OPACITY)
w.ortho.go_to_slice(("y", "x"), 30.0)
pump(0.5)
kept_before = (np.asarray(xy.shading.data)[..., :3]
               == ov.hex_to_rgb(ov.SHADE_KEPT)).all(-1).sum()
y_before = tuple(box.ranges["y"])
t0 = time.perf_counter()
w.margin_spin.setValue(4)
pump(0.5)
print(f"  margin set: the views were updated in "
      f"{time.perf_counter() - t0:.1f} s (including 0.5 s of waiting)")
kept_after = (np.asarray(xy.shading.data)[..., :3]
              == ov.hex_to_rgb(ov.SHADE_KEPT)).all(-1).sum()
check("margin: the box grows by 1 um where the outlines limit it",
      box.margin == 4 and abs(box.ranges["y"][0] - (y_before[0] - 1)) < 1e-6
      and abs(box.ranges["y"][1] - (y_before[1] + 1)) < 1e-6,
      f"y {y_before} -> {box.ranges['y']}")
check("margin: more voxels kept on the slice", kept_after > 1.1 * kept_before,
      f"{kept_before} -> {kept_after} kept points")
check("margin: the plan text says so", "margin +4 voxels (1 " in
      w.plan_label.text(), w.plan_label.text()[-90:])
viewer.window._qt_window.grab().save(str(OUT / "margin.png"))
w.shade_check.setChecked(False)
pump(0.3)

# 5. The polygon list: go to a slice, delete a polygon.
w._refresh(select=0)
texts = [w.polygon_list.item(k).text() for k in range(w.polygon_list.count())]
check("polygon list", len(texts) == 3 and any(t.startswith("xy2 · z = 40.0")
                                              for t in texts),
      str(texts))
row = next(k for k, t in enumerate(texts) if "z = 40.0" in t)
w.polygon_list.setCurrentRow(row)
w._go_to_polygon()
pump(0.5)
check("go to slice", abs(w.ortho.cursor["z"] - 40.0) < 1e-6,
      str(w.ortho.cursor["z"]))

# 6. Editing the polygon on this slice changes that polygon only.
data = [np.array(d) for d in xy.shapes.data]
k = xy.shape_keys.index(40.0)
data[k][:, -1] += 1.0
w.ortho._syncing["shapes"] = True
xy.shapes.data = data
w.ortho._syncing["shapes"] = False
w.ortho._on_shapes(xy, SimpleNamespace(action=ActionType.CHANGED,
                                       data_indices=[k]))
pump(0.5)
keys = w.boxes[0].outlines[("y", "x")]
check("edit changes that polygon", sorted(keys) == [20.0, 40.0]
      and abs(keys[40.0][0][1] - (P40[0][1] + 1)) < 1e-6
      and abs(keys[20.0][0][1] - P20[0][1]) < 1e-6)

# 7. A crop of the box: the crop's record lists the polygons.
w.out_edit.setText(str(OUT / "crops"))
w.overwrite_check.setChecked(True)
w._crop(all_boxes=False)
end = time.time() + 120
pump(0.5)
while w.cancel_button.isEnabled() and time.time() < end:
    pump(0.2)
pump(0.5)
out = vc.output_path_for(w.image, w.boxes[0], OUT / "crops")
ok = out.exists()
record = zarr.open_group(out, mode="r").attrs["crop_provenance"] if ok else {}
check("crop written, checked and recorded", ok
      and record.get("verified", {}).get("ok") is True
      and record.get("outlines_combine") == "all"
      and record.get("outline_margin_voxels") == 4
      and [p["slice"] for p in record["outlines"]["y,x"]] == [20.0, 40.0],
      w.status.text()[:80])

# 8. Deleting the polygon on this slice deletes only it.
w.ortho._on_shapes(xy, SimpleNamespace(action=ActionType.REMOVING,
                                       data_indices=[k]))
pump(0.5)
check("delete removes that polygon only", len(w.boxes) == 1
      and list(w.boxes[0].outlines[("y", "x")]) == [20.0])

viewer.close()
pump(0.5)
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
sys.exit(0 if all(results) else 1)
