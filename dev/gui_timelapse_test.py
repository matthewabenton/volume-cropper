"""How the GUI handles a time-lapse (t, c, z, y, x) image (scratchpad only)."""
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(__import__("os").environ.get("VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers

import napari  # noqa: E402
import numpy as np  # noqa: E402
from imageio import imwrite  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402


def pump(seconds=0.5):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


tmp = Path(tempfile.mkdtemp())
src = tmp / "timelapse.zarr"
# 6 frames, 5 min apart; 2 channels; pyramid shrinks only space.
data0 = make_test_image(src, shape0=(6, 2, 40, 160, 140), axes="tczyx",
                        factors=[(1, 1, 1, 1, 1), (1, 1, 1, 4, 4),
                                 (1, 1, 2, 8, 8)],
                        scale0=(5.0, 1.0, 2.0, 0.25, 0.25))

viewer, w = crop_gui.open_window(show=False)
win = viewer.window._qt_window
win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
win.resize(1600, 1000)
win.show()
w.open_image(src)
pump(1)
w.show_3d_check.setChecked(True)     # off at first since 0.5.0
pump(1)
views = w.ortho.views
print("summary:", w.image_info.text().replace("\n", " | "))
for name, view in views.items():
    dims = view.model.dims
    print(f"{name}: axis labels {list(dims.axis_labels)}, "
          f"sliders {list(dims.not_displayed)}, point {tuple(round(p, 1) for p in dims.point)}")

# Move to frame 3 in the XY view: the others should follow.
views["XY"].model.dims.set_point(0, 15.0)
pump(0.5)
print("after t=15 in XY:", {n: round(v.model.dims.point[0], 1) for n, v in views.items()})

# A rectangle in XY, then restrict t to frames 1-3 (0-based; 5-15 min) with
# the number fields, which show timepoints 1, 2, 3 ... since 0.11.0.
views["XY"].shapes.add_rectangles([np.array([[5.0, 4.0], [30.0, 25.0]])])
pump()
t0, t1 = w.range_spins["t"]
t0.setValue(2)
t1.setValue(4)
pump()
print("box ranges:", {a: tuple(round(v, 2) for v in r) for a, r in w.boxes[0].ranges.items()})
print("plan text:", w.plan_label.text().replace("\n", " | "))

out_dir = tmp / "out"
w.out_edit.setText(str(out_dir))
w._crop(False)
end = time.time() + 120
pump(0.5)
while w.cancel_button.isEnabled() and time.time() < end:
    pump(0.3)
print("crop status:", w.status.text().replace("\n", " | "))
crop = vc.open_image(vc.output_path_for(w.image, w.boxes[0], out_dir))
print("crop axes", crop.axis_names, "level-0 shape", crop.levels[0].array.shape,
      "t translation", crop.levels[0].translation[0])
plan = vc.plan_crop(w.image, w.boxes[0], min_size=w.min_size_spin.value())
s, e = plan.starts[0], plan.stops[0]
print("level 0 equals source:", np.array_equal(
    crop.levels[0].array[...],
    data0[s[0]:e[0], :, s[2]:e[2], s[3]:e[3], s[4]:e[4]]))
pump(1)
for name in ("XY", "3D"):
    Path("v03_out").mkdir(exist_ok=True)       # (not into dev/)
    imwrite(Path("v03_out") / f"timelapse_{name}.png",
            w.ortho._qt_viewers[name].screenshot(flash=False))

# A lasso box with 3D masking, then another timepoint.
from volume_cropper import ortho_views as ov  # noqa: E402
lasso = np.array([[8.0, 10.0], [14.0, 30.0], [30.0, 28.0], [26.0, 8.0]])
views["XY"].shapes.add_polygons([lasso])
pump()
w.ortho.set_mask_mode("grey")
pump(1)
v3 = views["3D"]
cache = w.ortho._volume_caches[0]
print("3D masking on time-lapse:", type(v3.channel_layers[0].data).__name__,
      "| extra layers", len(v3.mask_layers), "| cached timepoint key", cache._key,
      "| slice shape", np.asarray(v3.channel_layers[0]._slice.image.view).shape
      if hasattr(v3.channel_layers[0], "_slice") else "?")
views["XY"].model.dims.set_point(0, 25.0)        # frame 5
pump(1)
print("after t=25: 3D t point", v3.model.dims.point[0], "| cached key", cache._key)
kept = v3.channel_layers[0].data[(5, slice(None), slice(None), slice(None))]
lost = v3.mask_layers[0].data[(5, slice(None), slice(None), slice(None))]
level = w.ortho._levels_for(v3)[0]
src_frame = w.image.levels[level].array[5, 0]
print("frame 5: kept + blacked-out == source:",
      np.array_equal(kept.astype(np.int64) + lost, src_frame),
      "| blacked-out voxels:", int((lost > 0).sum()))
viewer.close()
