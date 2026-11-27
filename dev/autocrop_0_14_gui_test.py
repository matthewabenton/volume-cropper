"""0.14.0 (experimental) "Find a bright object": in a VISIBLE window.

Usage: python autocrop_0_14_gui_test.py OUT_DIR

On the synthetic embryo demo (OME-Zarr, read piece by piece) and on a
synthetic single-embryo time-lapse (a Fiji TIFF written into OUT_DIR, read
into memory). Checks: the experimental group is off at first; with no
region selected, "Fit region to object" adds a region around the brightest
embryo (it holds >= 95 % of embryo 1's true voxels); with a rough region
around embryo 4 selected, the region shrinks to it on every side and still
holds it; "Undo fit" gives the old region back; on the time-lapse (the
embryo drifts and is missing at the first timepoint) the region holds the
embryo at every timepoint, its t range starts at timepoint 2, and its crop
is checked OK. Saves autocrop_demo.png and autocrop_movie.png (synthetic
data only). Answers every dialog itself and closes its windows.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
results, messages = [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
for name in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, name, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True

truth = np.load(DEMO / "embryo_tissue_demo_truth.npz")
labels = truth["labels"]
from scipy import ndimage as ndi  # noqa: E402

objects = ndi.find_objects(labels)


def true_box(k, image):
    lev = image.levels[0]
    return {a: vc.index_range_to_physical(s.start, s.stop, lev.scale[i + 1],
                                          lev.translation[i + 1])
            for i, (a, s) in enumerate(zip("zyx", objects[k - 1]))}


def share_inside(k, ranges, image):
    """Share of embryo k's true voxels inside the ranges."""
    lev = image.levels[0]
    zz, yy, xx = np.nonzero(labels == k)
    inside = np.ones(zz.shape, bool)
    for i, (axis, index) in enumerate(zip("zyx", (zz, yy, xx))):
        pos = lev.translation[i + 1] + index * lev.scale[i + 1]
        inside &= (pos >= ranges[axis][0]) & (pos <= ranges[axis][1])
    return inside.mean()


viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1600, 1000)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
pump(3)
image = w.image
check("the experimental group is there and off at first",
      hasattr(w, "object_group") and not w.object_group.isChecked())
w.object_group.setChecked(True)
w.object_group.channel_combo.setCurrentIndex(0)
w.object_group.margin_spin.setValue(10.0)
# the demo's embryos touch bright tissue: without growing (see the tooltip)
w.object_group.grow_check.setChecked(False)
w._refresh(select=None)
pump(0.5)

# --- no region selected: a new region around the brightest embryo ----------------
before = len(w.boxes)
w.object_group.fit()
pump(2)
ok = len(w.boxes) == before + 1
box = w.boxes[-1] if ok else None
share = share_inside(1, box.ranges, image) if box else 0
check("no region selected: a new region around embryo 1",
      ok and share >= 0.95, f"{box.name if box else '-'}: {share:.1%} of "
      f"embryo 1 inside; {w.object_group.info.text()[:120]}")
viewer.screenshot(str(OUT / "autocrop_demo.png"), canvas_only=False)

# --- a rough region around embryo 4: shrunk to it ------------------------------------
tb = true_box(4, image)
rough = {a: (tb[a][0] - 60.0, tb[a][1] + 60.0) for a in "zyx"}
i = w._new_box(rough)
w._refresh(select=i)
pump(1)
w.object_group.fit()
pump(2)
fitted = w.boxes[i].ranges
shrunk = all(fitted[a][0] > rough[a][0] and fitted[a][1] < rough[a][1]
             for a in "zyx")
share = share_inside(4, fitted, image)
check("a rough region around embryo 4 shrinks on every side and holds it",
      shrunk and share >= 0.95,
      f"{share:.1%}; " + ", ".join(
          f"{a} {fitted[a][0]:.0f}-{fitted[a][1]:.0f} (rough "
          f"{rough[a][0]:.0f}-{rough[a][1]:.0f})" for a in "zyx"))
w.object_group.undo()
pump(1)
check("Undo fit gives the rough region back",
      all(np.allclose(w.boxes[i].ranges[a], rough[a]) for a in "zyx"))

# --- a single-embryo time-lapse (TIFF, in memory) ---------------------------------
rng = np.random.default_rng(3)
t_n, shape = 6, (30, 160, 200)
movie = rng.poisson(120, size=(t_n, *shape)).astype(np.float32)
grids = np.ogrid[tuple(slice(0, n) for n in shape)]
for t in range(1, t_n):                       # missing at the first timepoint
    centre = (15, 80, 60 + 12 * t)
    inside = sum(((g - c) / r) ** 2 for g, c, r in
                 zip(grids, centre, (8, 30, 25))) <= 1
    movie[t][inside] += 2500
movie = movie.astype(np.uint16)
path = OUT / "single_embryo_movie.tif"
tifffile.imwrite(path, movie[:, :, None], imagej=True,
                 resolution=(1 / 0.5, 1 / 0.5),
                 metadata={"axes": "TZCYX", "spacing": 2.0, "unit": "um",
                           "finterval": 300.0})
w._may_discard_boxes = lambda action: True
from volume_cropper import file_images as fi  # noqa: E402

w.open_image(path, file_choice=fi.FileChoice())
pump(3)
w.object_group.setChecked(True)
w.object_group.margin_spin.setValue(4.0)
# one embryo on a dark background: with growing (the default)
w.object_group.grow_check.setChecked(True)
w._refresh(select=None)
before = len(w.boxes)
w.object_group.fit()
pump(2)
box = w.boxes[-1] if len(w.boxes) == before + 1 else None
ok = box is not None
if ok:
    # every timepoint's embryo inside: x from 60 + 12 - 25 to 60 + 60 + 25
    x_lo, x_hi = (60 + 12 - 25) * 0.5, (60 + 60 + 25) * 0.5
    ok = (box.ranges["x"][0] <= x_lo and box.ranges["x"][1] >= x_hi
          and box.ranges["t"][0] > 0.0)
check("time-lapse: the region holds the drifting embryo at every "
      "timepoint, from timepoint 2", ok,
      "" if box is None else f"x {box.ranges['x']}, t {box.ranges['t']}; "
      f"{w.object_group.info.text()[:160]}")
w.out_edit.setText(str(OUT / "crops"))
w.overwrite_check.setChecked(True)
viewer.screenshot(str(OUT / "autocrop_movie.png"), canvas_only=False)
w._crop(False)
end = time.time() + 60
pump(0.5)
while time.time() < end and not w.crop_selected_button.isEnabled():
    pump(0.2)
pump(0.5)
check("its crop is checked OK", "checked OK" in w.status.text(),
      w.status.text().replace("\n", " | ")[:160])

print(f"\n{sum(results)} of {len(results)} checks passed.")
if messages:
    print("messages:", messages)
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
