"""2026-10-27 additions, in a VISIBLE window: hiding crosshairs and crop
boxes, projections through a slab or chosen ranges, the crop progress with
a time estimate, and "Keep position" off (crop starts at 0).

Usage: python view_options_gui_test.py OUT_DIR   (OUT_DIR: a scratch folder; its outputs are replaced)
Tests the folder in VC_TOOL (default: the tool folder this script is in). The image has two
bright objects at different depths that overlap in XY: A (value 4000) at
z 20-40 µm and B (value 2000) at z 80-100 µm.
"""
import faulthandler
import os
import shutil
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

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


# z 2 µm, y/x 1 µm voxels, an offset as from stitching; two objects.
src = OUT / "two_objects.zarr"
if src.exists():
    shutil.rmtree(src)
factors = [(1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 4, 4)]
make_test_image(src, shape0=(1, 64, 128, 128), factors=factors,
                scale0=(1.0, 2.0, 1.0, 1.0),
                translation=(0.0, 1000.0, -500.0, 250.0))
data0 = np.zeros((1, 64, 128, 128), dtype=np.uint16)
data0[0, 10:20, 20:70, 20:70] = 4000          # A: z 1020-1040 µm
data0[0, 40:50, 50:100, 50:100] = 2000        # B: z 1080-1100 µm
root = zarr.open_group(src, mode="r+")
for k, f in enumerate(factors):
    root[str(k)][...] = data0[tuple(slice(None, None, n) for n in f)]

viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(src, choice=LoadChoice(show_3d=True))
pump(3)
ortho = w.ortho
xy = ortho.views["XY"]


def shown_values():
    """The values the XY view displays now (napari's current slice)."""
    layer = xy.channel_layers[0]
    pump(1.5)
    return np.unique(np.asarray(layer._slice.image.raw))


# 1. Projections.
w.projection_combo.setCurrentIndex(w.projection_combo.findData("max"))
w._on_projection()
values = shown_values()
check("whole depth: both objects", {2000, 4000} <= set(values.tolist()),
      str(values))
w.through_combo.setCurrentIndex(w.through_combo.findData("slab"))
w.slab_spin.setValue(10.0)          # also applies it
ortho._set_cursor({"z": 1030.0})    # in A
values = shown_values()
check("slab at A: only A", 4000 in values and 2000 not in values,
      str(values))
check("slab: margins of half the thickness",
      np.isclose(xy.model.dims.margin_left[-3], 5.0)
      and np.isclose(xy.model.dims.margin_right[-3], 5.0))
ortho._set_cursor({"z": 1090.0})    # move the slider: the slab follows
values = shown_values()
check("slab moved to B: only B", 2000 in values and 4000 not in values,
      str(values))
check("slab options shown only for the slab", w.slab_row.isVisibleTo(w)
      and not w.range_box.isVisibleTo(w))
w.through_combo.setCurrentIndex(w.through_combo.findData("range"))
w._on_projection()
check("range options shown", w.range_box.isVisibleTo(w)
      and not w.slab_row.isVisibleTo(w))
z_from, z_to = w.projection_spins["z"]
z_from.setValue(1000.0)
z_to.setValue(1050.0)               # A only
values = shown_values()
check("range z 1000-1050: only A", 4000 in values and 2000 not in values,
      str(values))
check("slider kept inside the range", 1000.0 <= ortho.cursor["z"] <= 1050.0,
      f"z = {ortho.cursor['z']}")
w.projection_combo.setCurrentIndex(0)
w._on_projection()
check("no projection: options hidden", not w.slab_row.isVisibleTo(w)
      and not w.range_box.isVisibleTo(w))

# 2. Hiding crosshairs and crop boxes.
w._add_box()
pump(1)
w.crosshairs_check.setChecked(False)
pump(0.5)
check("crosshairs hidden in all 2D views",
      not any(ortho.views[n].crosshair.visible for n in ("XY", "XZ", "YZ")))
w.crosshairs_check.setChecked(True)
pump(0.5)
check("crosshairs back", all(ortho.views[n].crosshair.visible
                             for n in ("XY", "XZ", "YZ")))
w.boxes_check.setChecked(False)
pump(0.5)
hidden = [ortho.views[n].shapes.visible for n in ("XY", "XZ", "YZ")] \
    + [layer.visible for n in ("XY", "XZ", "YZ")
       for layer in ortho.views[n].outline_layers.values()] \
    + [layer.visible for layer in ortho.views["3D"].box_layers]
check("crop boxes hidden everywhere (2D lines, names, 3D frames)",
      not any(hidden), f"{len(hidden)} layers")
ortho._set_cursor({"z": 1060.0})    # redrawing must not bring them back
pump(0.5)
check("still hidden after moving",
      not any(layer.visible for n in ("XY", "XZ", "YZ")
              for layer in ortho.views[n].outline_layers.values()))
w.boxes_check.setChecked(True)
pump(0.5)
check("crop boxes back", ortho.views["XY"].shapes.visible
      and ortho.views["XY"].outline_layers["selected"].visible
      and all(layer.visible for layer in ortho.views["3D"].box_layers))

# 3. Crop with progress, "Keep position" off.
texts, values = [], []
w.progress_signal.connect(lambda d, t, s: texts.append(w.status.text()))
w.keep_position_check.setChecked(False)
w.out_edit.setText(str(OUT / "crops"))
w.overwrite_check.setChecked(True)
w._crop(False)
pump(0.5)
while not w.crop_selected_button.isEnabled():
    pump(0.2)
check("crop verified", "verified OK" in w.status.text(), w.status.text())
check("progress reached 100 %", w.progress_bar.value() == 1000)
print("   progress texts:", texts[:1], "...", texts[-1:], flush=True)
check("progress in bytes written", any("written" in t for t in texts))
out = vc.open_image(w._last_output)
check("Keep position off: crop starts at 0",
      out.levels[0].translation == [0.0, 0.0, 0.0, 0.0],
      str(out.levels[0].translation))
# The estimate itself (a real crop here is too quick to show one).
w._crop_started = time.time() - 30
w._on_progress(1e9, 4e9, "crop01 (1/1)")
print("   with an estimate:", w.status.text(), flush=True)
check("time estimate shown", "left" in w.status.text()
      and "/s" in w.status.text() and "so far" in w.status.text())
check("tooltip names the OME-Zarr format",
      "OME-Zarr" in w.keep_position_check.toolTip())
viewer.close()
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
