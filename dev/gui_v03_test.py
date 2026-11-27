"""Volume Cropper GUI checks for the 2026-10-23 evening changes (scratchpad).

1. Box lines: solid box, dashed outline, faded outside the box.
2. 3D masking modes: show / dim / grey / hide.
3. "Open last crop in a new window" opens a full window that can crop again.
Usage: python gui_v03_test.py [IMAGE.zarr]   (default: a synthetic image)
"""
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = Path(__import__("os").environ.get("VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers

import numpy as np  # noqa: E402
from imageio import imwrite  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

OUT = Path("v03_out")
OUT.mkdir(exist_ok=True)


def pump(seconds=0.5):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def offscreen(viewer):
    win = viewer.window._qt_window
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.resize(1600, 1000)
    win.show()


def shot(w, name, tag):
    imwrite(OUT / f"{tag}_{name}.png",
            w.ortho._qt_viewers[name].screenshot(flash=False))


tmp = Path(tempfile.mkdtemp())
if len(sys.argv) > 1:
    src = Path(sys.argv[1])
else:
    src = tmp / "synthetic.zarr"
    # Bright blob in the middle so that masking is visible in 3D.
    make_test_image(src, shape0=(2, 48, 200, 180),
                    factors=[(1, 1, 1, 1), (1, 1, 4, 4), (1, 2, 8, 8)])

viewer, w = crop_gui.open_window(show=False)
offscreen(viewer)
t0 = time.time()
w.open_image(src)
pump(1)
w.show_3d_check.setChecked(True)     # off at first since 0.5.0
pump(1)
print(f"opened in {time.time() - t0:.1f} s; title: {viewer.title!r}")
views = w.ortho.views
image = w.image
ext = {a: image.extent(a) for a in image.range_axes}


def frac(axis, a, b):
    low, high = ext[axis]
    return low + a * (high - low), low + b * (high - low)


# Box 1: rectangle in XY. Box 2: lasso in XY, refined by a polygon in XZ.
(y0, y1), (x0, x1) = frac("y", 0.1, 0.4), frac("x", 0.1, 0.4)
views["XY"].shapes.add_rectangles([np.array([[y0, x0], [y1, x1]])])
pump()
(ya, yb), (xa, xb) = frac("y", 0.45, 0.9), frac("x", 0.4, 0.9)
lasso = np.array([[ya, (xa + xb) / 2], [(ya + yb) / 2, xb], [yb, xb - 0.1 * (xb - xa)],
                  [yb, xa], [(ya + yb) / 2, xa + 0.2 * (xb - xa)]])
views["XY"].shapes.add_polygons([lasso])
pump()
w.refine_check.setChecked(True)
(za, zb) = frac("z", 0.2, 0.8)
poly_zx = np.array([[za, xa], [za, xb], [zb, (xa + xb) / 2]])
views["XZ"].shapes.add_polygons([poly_zx])
pump()
w.refine_check.setChecked(False)
print("boxes:", [(b.name, sorted(b.outlines)) for b in w.boxes],
      "selected:", w._selected())

# 1. Box lines in XY: what each layer holds.
for name in ("XY", "XZ"):
    layers = views[name].outline_layers
    edge = np.asarray(views[name].shapes.edge_color)
    print(f"{name}: box lines {len(layers['boxes'].data)} pieces, selected "
          f"{len(layers['selected'].data)} pieces, width "
          f"{layers['boxes'].edge_width:.3g} µm (zoom "
          f"{w.ortho._zoom(views[name]):.3g}); editable shapes' edge alpha "
          f"{sorted(set(edge[:, 3].round(2)))}")
shot(w, "XY", "1_lines")
shot(w, "XZ", "1_lines")
# Move z outside box 2 (whose z range is inside za-zb): it should fade.
alphas_before = np.asarray(views["XY"].outline_layers["selected"].edge_color)[:, 3]
z_out = ext["z"][0] + 0.05 * (ext["z"][1] - ext["z"][0])
views["XY"].model.dims.set_point(len(views["XY"].order) - 3, z_out)
pump()
alphas_after = np.asarray(views["XY"].outline_layers["selected"].edge_color)[:, 3]
print(f"selected box alpha: slice inside {set(alphas_before.round(2))} -> "
      f"outside {set(alphas_after.round(2))}")
shot(w, "XY", "1_faded")
# Zoom in 4x: the dashes are rebuilt (more pieces, same screen size).
n_before = len(views["XY"].outline_layers["selected"].data)
cam = ov.camera(views["XY"].model)
cam.zoom = cam.zoom * 4
pump()
print(f"after 4x zoom: selected pieces {n_before} -> "
      f"{len(views['XY'].outline_layers['selected'].data)}; width "
      f"{views['XY'].outline_layers['boxes'].edge_width:.3g} µm")
w.ortho.reset_cameras()
mid = np.mean(ext["z"])
views["XY"].model.dims.set_point(len(views["XY"].order) - 3, mid)
pump()

# 2. 3D masking modes on box 2 (selected), clipped to the box.
w.clip_check.setChecked(True)
pump(1)
v3 = views["3D"]
for mode in ("dim", "grey", "hide", "show"):
    t0 = time.time()
    w.mask_combo.setCurrentIndex([m for m in range(w.mask_combo.count())
                                  if w.mask_combo.itemData(m) == mode][0])
    w.mask_combo.activated.emit(w.mask_combo.currentIndex())
    pump(1.5)
    kinds = {type(layer.data).__name__ for layer in v3.channel_layers}
    print(f"3D {mode}: {time.time() - t0:.1f} s; channel data {kinds}; "
          f"extra layers {[(l.name, round(l.opacity, 2)) for l in v3.mask_layers]}")
    shot(w, "3D", f"2_mask_{mode}")
    cam3 = ov.camera(v3.model)
    angles = cam3.angles
    cam3.angles = (0.0, 35.0, 80.0)
    pump(1)
    shot(w, "3D", f"2_mask_{mode}_tilted")
    cam3.angles = angles
# Editing the outline while dimming: no new disk read (cache).
w.mask_combo.setCurrentIndex(1)
w.mask_combo.activated.emit(1)
pump(1)
cache = w.ortho._volume_caches
reads_before = [c._key for c in cache]
box = w.boxes[w._selected()]
box.outlines[("y", "x")] = [tuple(p) for p in (lasso * [1, 1.0] + [0.02 * (yb - ya), 0])]
t0 = time.time()
w._refresh()
pump(1)
print(f"outline edit while dimming: {time.time() - t0:.1f} s, cache reused: "
      f"{[c._key for c in w.ortho._volume_caches] == reads_before and w.ortho._volume_caches is cache}")

if len(sys.argv) > 1:          # real data: don't write a large crop
    shot(w, "XY", "1_real")
    shot(w, "XZ", "1_real")
    viewer.close()
    print("done (real data: no crop)")
    sys.exit(0)

# 3. Crop box 2, then open it in a new window and crop it again there.
out_dir = tmp / "out"
w.out_edit.setText(str(out_dir))
w._crop(False)
end = time.time() + 300
pump(0.5)
while w.cancel_button.isEnabled() and time.time() < end:
    pump(0.3)
print("crop status:", w.status.text().replace("\n", " | "))
n_windows = len(crop_gui._WINDOWS)


def accept_load_dialog(tries=[50]):
    """Since 0.5.0 a new window asks "What to load" first: press Load."""
    dialog = QApplication.activeModalWidget()
    if dialog is not None and type(dialog).__name__ == "LoadDialog":
        dialog.accept()
    elif tries[0] > 0:
        tries[0] -= 1
        QTimer.singleShot(100, accept_load_dialog)


from qtpy.QtCore import QTimer  # noqa: E402
QTimer.singleShot(200, accept_load_dialog)
w._open_last_crop()
pump(1)
viewer2, w2 = crop_gui._WINDOWS[-1]
offscreen(viewer2)
pump(1)
print(f"windows {n_windows} -> {len(crop_gui._WINDOWS)}; new title "
      f"{viewer2.title!r}; output folder {w2.out_edit.text() == str(out_dir)}; "
      f"image {w2.image.path.name if w2.image else None}; views "
      f"{sorted(w2.ortho.views)}")
w2._add_box()
pump()
w2._crop(False)
end = time.time() + 300
pump(0.5)
while w2.cancel_button.isEnabled() and time.time() < end:
    pump(0.3)
print("second crop status:", w2.status.text().replace("\n", " | "))
second = w2._last_output
record = vc.open_image(second).attrs["crop_provenance"]
print("second crop provenance: source", Path(record["source"]).name,
      "| has source_provenance:", "source_provenance" in record,
      "| original:", Path(record["source_provenance"]["source"]).name)
shot(w2, "XY", "3_second_window")
viewer2.close()
viewer.close()
print("done")
