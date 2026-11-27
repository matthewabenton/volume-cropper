"""The halo in the 3D view's blacked-out voxels (user report, 2026-11-02),
in a VISIBLE window on the synthetic embryo demo: the trunk outlined from
its true shape, the 3D view on, then pictures of the 3D view with
"Clip 3D to selected region" on and off, in the modes "hide" and "dim".

Usage: python halo_3d_check.py OUT_DIR [LABEL]
Tests the folder in VC_TOOL (default: the tool folder this script is in). Saves
3d_<LABEL>_<mode>_<clip>.png in OUT_DIR and prints whether the region's
edges fall on the 3D level's voxel grid.
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
sys.path.append(str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
import score_demo_embryos as score  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
LABEL = sys.argv[2] if len(sys.argv) > 2 else "now"
DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice(show_3d=True))
pump(5)
truth = np.load(DEMO / "embryo_tissue_demo_truth.npz")
outlines = score.drawn_outlines(truth["labels"] == 1,
                                truth["voxel_um"].astype(float))
box = crop_gui.vc.CropBox("trunk", outlines=outlines)
w.boxes = [box]
crop_gui._fit_to_outlines(box, w.image)
w._refresh(select=0)
pump(4)

level = w.ortho.level_3d_shown()
lv = w.image.levels[level]
ranges = w.ortho.box_ranges[0]
print(f"3D level {level}, voxel {lv.scale}; region {ranges}")
for axis in ("z", "y", "x"):
    i = w.image.axis_names.index(axis)
    edges = [(v - lv.translation[i]) / lv.scale[i] + 0.5 for v in ranges[axis]]
    print(f"  {axis}: region edges at 3D voxel index {edges[0]:.3f} .. "
          f"{edges[1]:.3f} ({'on' if all(abs(e - round(e)) < 1e-6 for e in edges) else 'NOT on'} the voxel grid)")

from volume_cropper import ortho_views as ov  # noqa: E402

view = w.ortho.views["3D"]
spatial = view.order[-3:]
index = [w.image.axis_names.index(a) for a in spatial]
offset = w.ortho.region.translation(w.image, level)
shape = w.ortho.region.shape(level)
# Voxels whose centres lie outside the region (computed here, not by the
# tool): with clipping on, napari must be given 0 there.
centres = [offset[j] + np.arange(shape[j]) * lv.scale[j] for j in index]
inside = np.ones([shape[j] for j in index], dtype=bool)
for k, (axis, c) in enumerate(zip(spatial, centres)):
    low, high = ranges[axis]
    keep = (c >= low - 1e-9) & (c <= high + 1e-9)
    inside &= keep.reshape([-1 if m == k else 1 for m in range(3)])
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


frame = view.frame
for clip in (True, False):
    w.clip_check.setChecked(clip)
    pump(2)
    for mode in ("hide", "dim"):
        w.mask_combo.setCurrentIndex(w.mask_combo.findData(mode))
        w.ortho.set_mask_mode(mode)
        pump(3)
        tag = f"{mode}_{'clip' if clip else 'noclip'}"
        drawn = np.asarray(view.channel_layers[0].data[...])
        plain = np.asarray(w.ortho._channel_array(level, 0, view.order)[...])
        if clip:
            check(f"{tag}: 0 outside the region", not drawn[~inside].any(),
                  f"{np.count_nonzero(drawn[~inside])} voxels outside not 0")
        else:
            check(f"{tag}: plain data outside the region",
                  np.array_equal(drawn[~inside], plain[~inside]))
        frame.grab().save(str(OUT / f"3d_{LABEL}_{tag}.png"))
        if clip and mode == "hide":      # a tilted camera shows the faces
            cam = ov.camera(view.model)
            angles = cam.angles
            cam.angles = (20.0, -35.0, 70.0)
            pump(2)
            frame.grab().save(str(OUT / f"3d_{LABEL}_{tag}_tilted.png"))
            cam.angles = angles
            pump(1)
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
viewer.close()
pump(0.5)
