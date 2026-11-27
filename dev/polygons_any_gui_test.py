"""Boxes keep the voxels inside ANY of their polygons (0.5.0), in a VISIBLE
window: the user's case of 2026-10-27, an XY polygon longer in y than the
YZ polygon of the same box, whose solid line used to cut the XY polygon.

Usage: python polygons_any_gui_test.py OUT_DIR   (OUT_DIR: a scratch folder)
Tests the folder in VC_TOOL (default: the tool folder this script is in).
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from test_volume_cropper import inside_polygon, make_test_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def bbox(points, plane):
    pts = np.asarray(points)
    return {plane[k]: (float(pts[:, k].min()), float(pts[:, k].max()))
            for k in range(2)}


src = OUT / "polygons.zarr"
if not src.exists():                 # z 2 µm, y/x 0.5 µm voxels
    make_test_image(src, shape0=(1, 40, 120, 100),
                    factors=[(1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 4, 4)],
                    scale0=(1.0, 2.0, 0.5, 0.5))
image = vc.open_image(src)
data0 = np.asarray(image.levels[0].array)
viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(src)
pump(2)

# As if drawn: a lasso in XY (y 10-50 µm), then, refining the same box, a
# polygon in YZ that covers only y 20-35 µm (z 20-60 µm).
xy = [(10.1, 8.3), (12.2, 40.1), (49.9, 35.3), (45.7, 12.2)]       # (y, x)
yz = [(20.3, 20.1), (34.9, 24.9), (30.1, 59.7), (21.1, 55.1)]      # (y, z)
w._box_added(bbox(xy, ("y", "x")), ("y", "x"), xy)
w.refine_check.setChecked(True)
w._box_added(bbox(yz, ("y", "z")), ("y", "z"), yz)
pump(1)
box = w.boxes[0]
cropped = w.ortho.box_ranges[0]                 # the solid lines
print("   box ranges:", {a: tuple(round(v, 2) for v in r)
                         for a, r in box.ranges.items()}, flush=True)
check("one box with both polygons", len(w.boxes) == 1
      and set(box.outlines) == {("y", "x"), ("y", "z")})
for name, points, plane in (("XY", xy, ("y", "x")), ("YZ", yz, ("y", "z"))):
    inside = all(cropped[a][0] <= lo + 1e-6 and hi <= cropped[a][1] + 1e-6
                 for a, (lo, hi) in bbox(points, plane).items())
    check(f"solid line encloses the {name} polygon", inside)
print("   panel:", w.plan_label.text().replace("\n", " | "), flush=True)

# Crop it and compare with an independent calculation: inside either.
w.out_edit.setText(str(OUT / "crops"))
w.overwrite_check.setChecked(True)
w._crop(False)
pump(1)
while not w.crop_selected_button.isEnabled():
    pump(0.5)
check("crop verified", "verified OK" in w.status.text(), w.status.text())
plan = vc.plan_crop(image, box, min_size=w.min_size_spin.value())
crop = np.asarray(vc.open_image(w._last_output).levels[0].array)
s, e = plan.starts[0], plan.stops[0]
zc = (s[1] + np.arange(e[1] - s[1])) * 2.0
yc = (s[2] + np.arange(e[2] - s[2])) * 0.5
xc = (s[3] + np.arange(e[3] - s[3])) * 0.5
yy, xx = np.meshgrid(yc, xc, indexing="ij")
in_xy = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                       xy).reshape(yy.shape)
yy2, zz = np.meshgrid(yc, zc, indexing="ij")
in_yz = inside_polygon(np.column_stack([yy2.ravel(), zz.ravel()]),
                       yz).reshape(yy2.shape)
keep = in_xy[None, None, :, :] | in_yz.T[None, :, :, None]
source = data0[:, s[1]:e[1], s[2]:e[2], s[3]:e[3]]
check("crop = voxels inside either polygon",
      np.array_equal(crop, np.where(keep, source, 0)))
check("XY polygon kept outside the YZ one's y range",
      bool((in_xy & ~in_yz.any(axis=1)[:, None]).any()))

# A box saved the old way (narrowed to the shorter polygon in y), loaded
# with "Load boxes...": it is fitted around both polygons again.
old = vc.CropBox("old", {"z": (20.1, 59.7), "y": (20.3, 34.9),
                         "x": (8.3, 40.1)}, outlines=dict(box.outlines))
csv = OUT / "old_boxes.csv"
vc.save_boxes_csv([old], csv, image)
from qtpy.QtWidgets import QFileDialog  # noqa: E402
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(csv), ""))
w._load_boxes()
pump(1)
loaded = w.boxes[0].ranges["y"]
check("old CSV box is fitted around its polygons on loading",
      loaded[0] <= 10.1 + 1e-6 and loaded[1] >= 49.9 - 1e-6, str(loaded))
viewer.close()
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
