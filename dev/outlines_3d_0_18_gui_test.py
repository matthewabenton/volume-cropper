"""0.18.0 per-slice outlines for split parts, outlines in 3D (shaded
surfaces) and the Settings dialog's start options (the user's wishes of
2026-11-13), in a VISIBLE window on synthetic data: two touching
ellipsoids whose contact is tilted 30 deg to the views.

- Fit to seeds with Outline on (silhouettes chosen): the two parts get
  outlines per slice, which do not overlap; how much of the neighbour
  each crop keeps;
- 3D view: the selected region's outline as a surface, then all regions';
- the Settings dialog with its new column.

Usage: python outlines_3d_0_18_gui_test.py OUT_DIR

Saves outlines_2d.png, surfaces_selected.png, surfaces_all.png and
settings.png. Synthetic data only; closes its window; settings file of
its own in OUT_DIR.
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
import numpy as np  # noqa: E402
from napari.layers import Image  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402

results = []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


# two touching ellipsoids (as gap_split_0_18_gui_test.py)
shape = (90, 140, 200)
scale = 2.0
middle = np.array([45.0, 70.0, 100.0])
angle = np.radians(30.0)
direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
centres = [middle - 28.0 * direction, middle + 28.0 * direction]
grid = np.indices(shape, dtype=np.float32)


def inside(centre):
    d = np.stack([grid[i] - centre[i] for i in range(3)])
    along = sum(d[i] * direction[i] for i in range(3))
    across = d - along * direction[:, None, None, None]
    other = np.sqrt(np.maximum((across ** 2).sum(axis=0) - d[1] ** 2, 0))
    return (along / 34.0) ** 2 + (d[1] / 26.0) ** 2 + (other / 20.0) ** 2 \
        <= 1.0


objects = [inside(c) for c in centres]
nearer = [None, None]
dist = [np.sqrt(sum((grid[i] - c[i]) ** 2 for i in range(3)))
        for c in centres]
union = objects[0] | objects[1]
truth = [union & (dist[0] <= dist[1]), union & (dist[1] < dist[0])]
rng = np.random.default_rng(0)
data = rng.poisson(100.0 + 1400.0 * union).astype(np.uint16)

load_dialog.FileLoadDialog.exec = lambda self: 1
viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
w.open_layers([Image(data, name="touching", scale=(scale,) * 3,
                     units=("um",) * 3)])
pump(3.0)
w.sections["find"].set_open(True)
group, ortho = w.object_group, w.ortho
group.setChecked(True)
group.grow_check.setChecked(False)
group.margin_spin.setValue(4.0)
group.outline_check.setChecked(True)
ortho._set_cursor({"z": float(middle[0] * scale)})
group.seeds = [dict(zip("zyx", c * scale)) for c in centres]
group.seed_results = [None, None]
ortho.set_seeds(group.seeds)
start = time.time()
group.fit_to_seeds()
took = time.time() - start
pump(0.5)
print("   " + group.info.text().replace("\n", "\n   "), flush=True)
check("the parts get outlines per slice (silhouettes chosen)",
      len(w.boxes) == 2
      and group.info.text().count("with its outline per slice") == 2,
      f"({took:.1f} s)")
# what each crop keeps (level 0), against the truth
image = w.image
level0 = image.levels[0]
names = image.axis_names
keeps = []
for box in w.boxes:
    index = []
    for i, axis in enumerate(names):
        first, end = vc.physical_to_index_range(
            *box.ranges[axis], level0.scale[i], level0.translation[i])
        index.append(slice(max(first, 0), min(end, shape[i])))
    keep = np.zeros(shape, bool)
    keep[tuple(index)] = True
    mask = vc.outline_mask(level0, names, [0, 0, 0], shape, box.outlines,
                           names)
    keeps.append(keep & np.broadcast_to(mask, shape))
for k in range(2):
    own = (keeps[k] & truth[k]).sum() / truth[k].sum()
    other = (keeps[k] & truth[1 - k]).sum() / truth[1 - k].sum()
    print(f"   crop of {w.boxes[k].name}: {own:.1%} of its own object, "
          f"{other:.1%} of the other", flush=True)
both = int((keeps[0] & keeps[1]).sum())
check("the two crops hardly overlap", both < 0.01 * min(k.sum() for k in
                                                         keeps),
      f"({both} voxels in both)")
w._refresh(select=0)
pump(0.5)
viewer.screenshot(str(OUT / "outlines_2d.png"), canvas_only=False)
# --- outlines in 3D -----------------------------------------------------------
w.show_3d_check.setChecked(True)
pump(3.0)
view = ortho.views["3D"]
ov.camera(view.model).angles = (0.0, 35.0, 60.0)
pump(1.0)
check("3D: the selected region's outline as a surface (the default)",
      len(view.surface_layers) == 1 and view.surface_layers[0].visible)
viewer.screenshot(str(OUT / "surfaces_selected.png"), canvas_only=False)
w.outlines_3d_combo.setCurrentIndex(w.outlines_3d_combo.findData("all"))
w._on_outlines_3d()
pump(1.0)
check("3D: all regions' outlines", len(view.surface_layers) == 2)
viewer.screenshot(str(OUT / "surfaces_all.png"), canvas_only=False)
# --- the Settings dialog ----------------------------------------------------------
dialog = crop_gui.SettingsDialog(w)
dialog.show()
pump(0.8)
dialog.grab().save(str(OUT / "settings.png"))
check("the Settings dialog shows the start options",
      "outline_kind" in dialog.auto_widgets
      and "outlines_3d" in dialog.view_widgets)
dialog.reject()
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
