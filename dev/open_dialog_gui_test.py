"""The "What to load" dialog and loaded regions, in a VISIBLE window.

Usage: python open_dialog_gui_test.py OUT_DIR   (OUT_DIR: a scratch folder)
Tests the folder in the VC_TOOL environment variable (default: the tool folder this script is in).
Uses a synthetic 3-channel image with an offset, so that every check can be
compared with the exact source voxels.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
TOOL = Path(os.environ.get("VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers

import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice, LoadDialog  # noqa: E402
from test_volume_cropper import L_SHAPE, make_test_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def pump(seconds=0.5):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def answer_dialog(action):
    """Run action(dialog) on the next modal "What to load" dialog."""
    def run():
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, LoadDialog):
            from qtpy.QtCore import QTimer
            QTimer.singleShot(100, run)
            return
        action(dialog)
    from qtpy.QtCore import QTimer
    QTimer.singleShot(200, run)


# A 3-channel image with an offset (as from stitching) and 4 levels.
src = OUT / "synthetic.zarr"
if not src.exists():
    make_test_image(src, shape0=(3, 64, 240, 200),
                    factors=[(1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 4, 4),
                             (1, 4, 8, 8)],
                    scale0=(1.0, 2.0, 0.5, 0.5),
                    translation=(0.0, 100.0, -20.0, 7.0))
data0 = np.asarray(vc.open_image(src).levels[0].array)

viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)

# 1. The dialog on its own: what it shows and what it returns.
image = vc.open_image(src)
dialog = LoadDialog(image, LoadChoice(channels=[0]), w.ortho.canvas_sizes())
dialog.show()
pump(0.5)
check("dialog: channel 0 ticked only",
      [c.isChecked() for c in dialog.channel_checks] == [True, False, False])
check("dialog: 3D off by default", dialog.view_3d_combo.currentText() == "Off")
check("dialog: whole image at first", dialog.region.whole)
print("   estimate:", dialog.estimate.text())
z0, z1 = dialog.range_spins["z"]
z0.setValue(150.0)
z1.setValue(200.0)
pump(0.3)
print("   z note:", dialog.range_notes["z"].text(), "| estimate:",
      dialog.estimate.text())
check("dialog: z range gives the same region as loaded_region",
      dialog.region.ranges == ov.loaded_region(image, dialog.ranges()).ranges)
z0.setValue(210.0)          # from after to: Load is disabled
pump(0.2)
ok = dialog.buttons.buttons()[0]
check("dialog: a backwards range disables Load", not ok.isEnabled(),
      dialog.estimate.text())
dialog._whole_image()
check("dialog: 'Whole image' resets the ranges", dialog.region.whole)
dialog.close()

# 2. Opening through the dialog: z 150-200 µm, y 20-60 µm, channels 0 + 2.
def choose(d):
    d.range_spins["z"][0].setValue(150.0)
    d.range_spins["z"][1].setValue(200.0)
    d.range_spins["y"][0].setValue(20.0)
    d.range_spins["y"][1].setValue(60.0)
    d.channel_checks[2].setChecked(True)
    d.estimate_text = d.estimate.text()
    choose.levels = ov.fitted_levels(d.image, d.region, d.canvas_sizes)
    choose.region = d.region
    d.accept()


answer_dialog(choose)
w.open_image(src, channels=[0], ask=True)
pump(3)
ortho = w.ortho
region = ortho.region
check("open: the region chosen is loaded", region.ranges == choose.region.ranges,
      str(region.ranges))
print("   panel:", w.image_info.text().replace("\n", " | "))
check("open: 3D view off", not w.show_3d_check.isChecked()
      and not ortho.views["3D"].channel_layers)
check("open: channels 0 and 2 shown",
      [layer.visible for layer in ortho.views["XY"].channel_layers]
      == [True, False, True])
check("open: channel 1 not measured yet",
      ortho.display[1]["measured"] is False)
start, stop = region.starts[0], region.stops[0]
for name in ov.VIEWS_2D:
    view = ortho.views[name]
    layer = view.channel_layers[0]
    order = [image.axis_names.index(a) for a in view.order]
    part = data0[(0,) + tuple(slice(start[i], stop[i]) for i in (1, 2, 3))]
    part = np.transpose(part, [image.range_axes.index(a) for a in view.order])
    same = np.array_equal(np.asarray(layer.data[0]), part)
    first = layer.data_to_world([0, 0, 0])
    where = [region.translation(image, 0)[i] for i in order]
    check(f"{name}: shows exactly the region's voxels", same)
    check(f"{name}: first voxel at the region's position",
          np.allclose(first, where), f"{first} vs {where}")
pump(2)
for name in ov.VIEWS_2D:      # the whole region fits (zoom = smallest fit)
    view = ortho.views[name]
    zoom = ov.camera(view.model).zoom
    shown = [size / zoom for size in view.model.canvas.size]
    needed = [region.ranges[a][1] - region.ranges[a][0] for a in view.plane]
    check(f"{name}: the whole loaded region is in view",
          all(s >= n * 0.999 for s, n in zip(shown, needed)),
          f"shows {np.round(shown, 1)} µm for {np.round(needed, 1)}")
print("   levels shown:", dict(ortho._levels_shown), "| estimated:",
      choose.levels)
check("estimate: predicted levels = levels napari shows",
      dict(ortho._levels_shown) == choose.levels)

# 3. A new box starts as the loaded region (middle half in XY).
w._add_box()
box = w.boxes[0]
check("Add: box z = loaded z", box.ranges["z"] == region.ranges["z"])
ylo, yhi = region.ranges["y"]
check("Add: box y = middle half of loaded y",
      np.allclose(box.ranges["y"], (ylo + (yhi - ylo) / 4,
                                    yhi - (yhi - ylo) / 4)))
box.outlines[("y", "x")] = [(c[0] * 0.4 + 30, c[1] * 0.4 + 20)
                            for c in L_SHAPE]
w._refresh(select=0)

# 4. Show channel 1: its contrast is measured from the region now.
w.show_checks[1].setChecked(True)
pump(1)
want = ov.channel_contrast(image, 1, region)
check("show channel 1: contrast measured",
      ortho.display[1]["measured"] is True
      and np.allclose(ortho.views["XZ"].channel_layers[1].contrast_limits,
                      want), f"{want}")

# 5. 3D on: the region at the auto level, masking lines up.
w.show_3d_check.setChecked(True)
pump(3)
view3d = ortho.views["3D"]
level = ortho._levels_for(view3d)[0]
check("3D: auto level for region and 3 channels",
      level == ov.auto_3d_level(image, region=region, n_channels=3),
      f"level {level}")
shape3d = view3d.channel_layers[0].data.shape
check("3D: volume = the region at that level",
      tuple(shape3d) == region.shape(level)[1:], f"{shape3d}")
w.mask_combo.setCurrentIndex(1)                                    # dim
w.ortho.set_mask_mode("dim")
pump(2)
kept = np.asarray(view3d.channel_layers[0].data[...])
lost = np.asarray(view3d.mask_layers[0].data[...]) if view3d.mask_layers \
    else np.zeros_like(kept)
lv = image.levels[level]
sl = region.slices(level)
check("3D masking: kept + blacked out = the region's data",
      np.array_equal(kept + lost, np.asarray(lv.array[sl])[0]))
check("3D masking: something is blacked out", lost.any())

# 6. Crop from the restricted load, then from a whole-image load: identical.
w.out_edit.setText(str(OUT / "crop_region"))
w.overwrite_check.setChecked(True)
w._crop(False)
pump(1)
while not w.crop_selected_button.isEnabled():
    pump(0.5)
first_crop = w._last_output
w._load(image, LoadChoice(), keep_boxes=True)
pump(2)
check("reload whole: box kept", len(w.boxes) == 1 and bool(w.boxes[0].outlines))
w.out_edit.setText(str(OUT / "crop_whole"))
w._crop(False)
pump(1)
while not w.crop_selected_button.isEnabled():
    pump(0.5)
a, b = vc.open_image(first_crop), vc.open_image(w._last_output)
same = len(a.levels) == len(b.levels) and all(
    np.array_equal(np.asarray(x.array), np.asarray(y.array))
    for x, y in zip(a.levels, b.levels))
check("crop after a restricted load = crop after a whole load", same,
      f"{len(a.levels)} levels")

# 7. "What to load…" with "Selected box": the region becomes the box.
def box_only(d):
    d._selected_box()
    d.accept()


answer_dialog(box_only)
w._change_what_to_load()
pump(3)
crop_ranges = w.ortho.box_ranges[0]
inside = all(region_low <= crop_ranges[a][0] + 1e-6
             and crop_ranges[a][1] <= region_high + 1e-6
             for a, (region_low, region_high) in w.ortho.region.ranges.items())
check("Selected box: the region covers the box", inside,
      str(w.ortho.region.ranges))
check("Selected box: the box stays", len(w.boxes) == 1)
check("Selected box: channel choice kept",
      [layer.visible for layer in ortho.views["XY"].channel_layers]
      == [True, True, True])
pump(5)
print("   read line:", w.read_info.text())
check("read line shows the amount read", w.read_info.text().startswith("Read"))

viewer.close()
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
