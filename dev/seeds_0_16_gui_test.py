"""0.16.0 "Find a bright object": seeds and outlines, in a VISIBLE window,
on the synthetic embryo demo (OME-Zarr, read piece by piece).

Usage: python seeds_0_16_gui_test.py OUT_DIR

Seeds are put with real mouse clicks on the XY view's canvas (the slice
and the view's centre are first moved to the point). Checks: Add seeds
switches the drawing tool to pan/zoom; a click puts a seed and draws a
cross in XY, XZ and YZ; a click on a seed removes it; with seeds on two
embryos that are not the brightest (one of them twice) and Outline
ticked, Fit to seeds makes two regions, each holding >= 90 % of its
embryo's true voxels (and saying which seed hit the same embryo), with
outlines in XY, XZ and YZ that keep >= 90 % of those voxels; the crop of
one region is checked OK and has zeros outside its outlines; Undo fit
removes the two regions. The two embryos are ones a threshold can
separate from the tissue (1, 4, 6; not the brightest); for the record it
then prints what the window says for a seed on embryo 5, which no
threshold separates. Saves seeds.png, seeds_fitted.png,
seeds_not_separable.png (synthetic data only). Answers every dialog itself and closes its windows. Uses a
settings file of its own in OUT_DIR.
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
from scipy import ndimage as ndi  # noqa: E402
from skimage.measure import points_in_poly  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
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

labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]

viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1600, 1000)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
pump(3)
image = w.image
lev = image.levels[0]
scale = [lev.scale[i + 1] for i in range(3)]          # z, y, x (after c)
offset = [lev.translation[i + 1] for i in range(3)]
nuclei = np.asarray(lev.array[0])                    # channel 0, level 0


def seed_point(k):
    """A bright voxel of embryo k near its centre, in µm."""
    where = np.argwhere(labels == k)
    values = nuclei[tuple(where.T)]
    bright = where[values >= np.percentile(values, 90)]
    centre = where.mean(axis=0)
    best = bright[np.argmin(np.linalg.norm(bright - centre, axis=1))]
    return {a: offset[i] + best[i] * scale[i] for i, a in enumerate("zyx")}


def click_at(point):
    """Move the slice and the XY view's centre to the point, then click
    the middle of the XY canvas."""
    w.ortho._set_cursor({"z": point["z"]})
    cam = ov.camera(w.ortho.views["XY"].model)
    cam.zoom = 3.0             # 3 screen pixels per µm: seeds 12 µm apart
    cam.center = (*tuple(cam.center)[:-2], point["y"], point["x"])
    pump(0.5)
    canvas = w.ortho._qt_viewers["XY"].canvas.native
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     QPoint(canvas.width() // 2, canvas.height() // 2))
    # (since the review of 2026-11-08 a click becomes a seed once the
    # double-click time has passed)
    pump(QApplication.doubleClickInterval() / 1000 + 0.4)


# which embryos: two that are not the brightest (so that without seeds the
# fit would pick another one) and that a threshold can separate from the
# tissue (measured 2026-11-08: embryos 1, 4 and 6; 2, 3 and 5 join the
# tissue or another embryo before they are whole, at every threshold)
signal = ndi.sum_labels(nuclei, labels, np.arange(1, labels.max() + 1))
order = [int(k) for k in np.argsort(signal)[::-1] + 1]   # brightest first
a, b = [k for k in order[1:] if k in (1, 4, 6)][:2]
print(f"embryos by signal: {order}; seeds on {a} and {b}", flush=True)

group = w.object_group
w.sections["find"].set_open(True)
group.setChecked(True)
group.channel_combo.setCurrentIndex(0)
group.margin_spin.setValue(10.0)
group.grow_check.setChecked(False)          # the demo's embryos touch tissue
group.outline_check.setChecked(True)
w.ortho.set_tool("add_polygon")
group.seed_button.setChecked(True)
pump(0.5)
check("Add seeds switches to the pan/zoom tool",
      str(w.ortho.views["XY"].shapes.mode) == "pan_zoom")
pa, pb = seed_point(a), seed_point(b)
click_at(pa)
check("a click puts a seed", len(group.seeds) == 1, str(group.seeds))
if group.seeds:
    got = group.seeds[0]
    check("the seed is where the click was (within 2 µm)",
          all(abs(got[ax] - pa[ax]) <= 2.0 for ax in "zyx"),
          f"{got} vs {pa}")
check("its cross in XY, XZ and YZ",
      all(len(w.ortho.views[v].seeds.data) == 1 and
          w.ortho.views[v].seeds.visible for v in ("XY", "XZ", "YZ")))
click_at(pa)
check("a click on the seed removes it", len(group.seeds) == 0)
click_at(pa)
click_at(pb)
second = dict(pa)
second["y"] += 12.0                           # embryo a again, a bit beside
click_at(second)
check("three seeds", len(group.seeds) == 3, str(len(group.seeds)))
viewer.screenshot(str(OUT / "seeds.png"), canvas_only=False)

before = len(w.boxes)
started = time.time()
group.fit_to_seeds()
pump(1)
new = w.boxes[before:]
text = group.info.text()
print(f"Fit to seeds took {time.time() - started:.1f} s:\n{text}", flush=True)
check("two new regions", [x.name for x in new] == ["object01", "object02"],
      str([x.name for x in new]))
check("the third seed was on the same embryo", "same object as object01"
      in text)


def kept_share(k, box):
    """Shares of embryo k's voxels inside the box, and inside the box and
    all its silhouettes."""
    where = np.argwhere(labels == k)
    pos = {ax: offset[i] + where[:, i] * scale[i] for i, ax in enumerate("zyx")}
    inside = np.ones(len(where), bool)
    for ax in "zyx":
        inside &= (pos[ax] >= box.ranges[ax][0]) & (pos[ax] <= box.ranges[ax][1])
    shaped = inside.copy()
    for plane, keys in box.outlines.items():
        for key, points in keys.items():
            flat = np.column_stack([pos[plane[0]], pos[plane[1]]])
            shaped &= points_in_poly(flat, np.asarray(points))
    return inside.mean(), shaped.mean()


for box, k in zip(new, (a, b)):
    in_box, in_shape = kept_share(k, box)
    check(f"{box.name} holds embryo {k}", in_box >= 0.9, f"{in_box:.1%}")
    check(f"{box.name}'s outlines (XY, XZ, YZ) keep embryo {k}",
          len(box.outlines) == 3 and in_shape >= 0.9,
          f"{len(box.outlines)} views, {in_shape:.1%}")
if new:
    w._refresh(select=before)
    w.ortho._set_cursor({ax: float(np.mean(new[0].ranges[ax]))
                         for ax in "zyx"})
    w.ortho.reset_cameras()
    pump(1)
viewer.screenshot(str(OUT / "seeds_fitted.png"), canvas_only=False)

# the crop of object01: checked OK, zeros outside its outlines
if new:
    w.out_edit.setText(str(OUT / "crops"))
    w.overwrite_check.setChecked(True)
    w._refresh(select=before)
    w._crop(False)
    end = time.time() + 120
    pump(0.5)
    while time.time() < end and not w.crop_selected_button.isEnabled():
        pump(0.2)
    pump(0.5)
    check("object01's crop is checked OK", "checked OK" in w.status.text(),
          w.status.text().replace("\n", " | ")[:160])
    crops = sorted((OUT / "crops").glob("*object01*.zarr"))
    if crops:
        cropped = vc.open_image(crops[0])
        data = np.asarray(cropped.levels[0].array[0])
        zero = (data == 0).mean()
        check("the crop has zeros outside the outlines", zero > 0.05,
              f"{zero:.1%} of its voxels are 0 ({crops[0].name})")

group.undo()
pump(0.5)
check("Undo fit removes the two regions", len(w.boxes) == before,
      group.info.text())

# for the record (no check): a seed on embryo 5, which no threshold
# separates in this demo, beside one on embryo a: what the window says
group.clear_seeds()
w.ortho.set_tool("pan_zoom")
group.seed_button.setChecked(True)
click_at(pa)
click_at(seed_point(5))
group.fit_to_seeds()
pump(1)
print("Embryo 5 (not separable here) beside embryo", a, ":\n"
      + group.info.text(), flush=True)
for box in w.boxes[before:]:
    print(f"  {box.name}: holds embryo {a} {kept_share(a, box)[0]:.0%}, "
          f"embryo 5 {kept_share(5, box)[0]:.0%}; "
          + ", ".join(f"{ax} {box.ranges[ax][0]:.0f}-{box.ranges[ax][1]:.0f}"
                      for ax in "zyx"), flush=True)
viewer.screenshot(str(OUT / "seeds_not_separable.png"), canvas_only=False)
group.undo()
pump(0.5)

print(f"\n{sum(results)} of {len(results)} checks passed.")
if messages:
    print("messages:", messages)
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
