"""Diagnostic (0.18.0): how the seeds look in the 3D view with a few
settings (they came out as black discs). VISIBLE window, synthetic demo.
Saves look_<variant>.png (the 3D canvas only) to OUT_DIR.

Usage: python seeds_3d_look.py OUT_DIR
"""
import os
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(OUT / "settings.json")
TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data"


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


load_dialog.FileLoadDialog.exec = lambda self: 1
viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
pump(2.0)
w.show_3d_check.setChecked(True)
pump(3.0)
ortho = w.ortho
labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]
lev0 = w.image.levels[0]
scale, offset = np.array(lev0.scale[1:]), np.array(lev0.translation[1:])
seeds = []
for k in range(1, int(labels.max()) + 1):
    p = offset + np.argwhere(labels == k).mean(axis=0) * scale
    seeds.append({"z": float(p[0]), "y": float(p[1]), "x": float(p[2])})
ortho.set_seeds(seeds)
pump(1.0)
layer = ortho.views["3D"].seeds
qt3d = ortho._qt_viewers["3D"]
print("face colours:", np.round(layer.face_color[:2], 2).tolist())
print("border:", layer.border_width[:2], np.round(layer.border_color[:1], 2))
print("size:", layer.size[:2], "limits:", layer.canvas_size_limits,
      "symbol:", layer.symbol[:2], "blending:", layer.blending,
      "shading:", layer.shading)


def save(name):
    pump(0.5)
    image = qt3d.canvas._scene_canvas.render()
    import imageio.v3 as iio
    iio.imwrite(OUT / f"look_{name}.png", image)


save("as_is")
layer.border_width = 0
save("no_border")
layer.symbol = "disc"
save("disc_no_border")
layer.symbol = "cross"
layer.border_width = 0.18
layer.blending = "additive"
save("additive")
viewer.close()
pump(0.5)
