"""Reproduce the crash when choosing 'hide (as saved)' (scratchpad only).

Runs in a VISIBLE window (the offscreen tests did not crash). Prints each
step before doing it, so the last line printed shows where it died.
Polygons in all three views (the user's suggestion), then every action
that removes layers, with clipping on.
"""
import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable(all_threads=False)
sys.path.insert(0, __import__("os").environ.get("VC_TOOL", str(Path(__file__).resolve().parents[1])))
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402


def pump(s=1.0):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def step(text):
    print("->", text, flush=True)


def choose(combo, value):
    i = [k for k in range(combo.count()) if combo.itemData(k) == value][0]
    combo.setCurrentIndex(i)
    combo.activated.emit(i)      # what a real selection sends


viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(sys.argv[1])
pump(3)
views = w.ortho.views
ext = {a: w.image.extent(a) for a in w.image.range_axes}
f = lambda a, u: ext[a][0] + u * (ext[a][1] - ext[a][0])  # noqa: E731

step("polygon in XY (y, x)")
views["XY"].shapes.add_polygons([np.array(
    [[f("y", .45), f("x", .65)], [f("y", .67), f("x", .9)],
     [f("y", .9), f("x", .85)], [f("y", .9), f("x", .4)],
     [f("y", .67), f("x", .5)]])])
pump(2)
w.refine_check.setChecked(True)
step("polygon in XZ (z, x), refining")
views["XZ"].shapes.add_polygons([np.array(
    [[f("z", .2), f("x", .45)], [f("z", .25), f("x", .88)],
     [f("z", .8), f("x", .8)], [f("z", .75), f("x", .5)]])])
pump(2)
step("polygon in YZ (y, z), refining")
views["YZ"].shapes.add_polygons([np.array(
    [[f("y", .47), f("z", .3)], [f("y", .5), f("z", .7)],
     [f("y", .88), f("z", .75)], [f("y", .85), f("z", .25)]])])
pump(2)
w.refine_check.setChecked(False)
print("   outlines:", sorted(w.boxes[w._selected()].outlines), flush=True)

from imageio import imwrite  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402

for mode, clip in [("hide", False), ("dim", False), ("hide", False),
                   ("grey", True), ("hide", True), ("dim", True),
                   ("grey", True), ("show", True), ("hide", True),
                   ("dimgrey", True), ("hide", True), ("dimgrey", False),
                   ("grey", True), ("dimgrey", True), ("dim", True)]:
    step(f"clip {clip}, mode {mode}")
    w.clip_check.setChecked(clip)
    pump(1)
    choose(w.mask_combo, mode)
    pump(3)
    v3 = views["3D"]
    print(f"   ok: {len(v3.mask_layers)} extra layers "
          f"{[round(l.opacity, 2) for l in v3.mask_layers]}", flush=True)
    if clip:                       # tilted screenshot of each mode
        cam = ov.camera(v3.model)
        angles = cam.angles
        cam.angles = (0.0, 35.0, 80.0)
        pump(1)
        imwrite(f"v03_out/repro_{mode}.png",
                w.ortho._qt_viewers["3D"].screenshot(flash=False))
        cam.angles = angles

step("3D level change with clipping and dimming on")
choose(w.level_3d_combo, len(w.image.levels) - 1)
pump(3)
step("3D off, then on again")
w.show_3d_check.setChecked(False)
pump(2)
w.show_3d_check.setChecked(True)
pump(3)
step("add a second box, then delete it")
w._add_box()
pump(2)
w._remove_box()
pump(2)
step("delete the polygon box (clipped, dimmed)")
w._remove_box()
pump(2)
step("open the image again (removes every layer)")
w.open_image(sys.argv[1])
pump(3)
print("no crash", flush=True)
viewer.close()
