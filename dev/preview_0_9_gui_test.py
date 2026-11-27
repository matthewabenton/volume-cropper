"""0.9.0 crop preview in the main views, in a VISIBLE window on the
synthetic embryo demo (trunk outlined from its true shape, 3D off).

Usage: python preview_0_9_gui_test.py OUT_DIR
Tests the folder in VC_TOOL (default: the tool folder this script is in). Checks, per 2D view:
the black rectangles end exactly at the region's edges; inside the region,
black = the voxels the crop sets to 0 (compared with the crop's own rule,
outline_mask at the crop's detail, on the preview's grid); a slice outside
the region is all black; on a projection only the rectangle shows; preview
off hides everything; Zoom to region fits the views and moves the slices
into the region. Also times slider moves with the preview on and off.
Saves preview_window.png and preview_<view>.png in OUT_DIR.
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
from volume_cropper import ortho_views as ov  # noqa: E402
import score_demo_embryos as score  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
results = []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
pump(5)
truth = np.load(DEMO / "embryo_tissue_demo_truth.npz")
outlines = score.drawn_outlines(truth["labels"] == 1,
                                truth["voxel_um"].astype(float))
box = vc.CropBox("trunk", outlines=outlines)
w.boxes = [box]
crop_gui._fit_to_outlines(box, w.image)
w._refresh(select=0)
pump(3)
o = w.ortho
ranges = o.box_ranges[0]
spatial = vc._spatial_axes(w.image)
pixel = vc.pixel_size(w.image)
print(f"region {ranges}")

# --- Zoom to region -----------------------------------------------------------
o._set_cursor({a: o.extent(a)[0] for a in spatial})     # slices outside it
pump(1)
o.zoom_to_selected()
pump(3)
check("zoom: slices moved into the region",
      all(ranges[a][0] <= o.cursor[a] <= ranges[a][1] for a in spatial),
      str({a: round(o.cursor[a], 1) for a in spatial}))
fits = []
for name in ov.VIEWS_2D:
    view = o.views[name]
    cam = ov.camera(view.model)
    size = max(ranges[a][1] - ranges[a][0] for a in view.plane)
    fits.append(size * cam.zoom / min(view.model.canvas.size))
check("zoom: region fills 90 % of the smallest view side (one zoom)",
      max(fits) <= 0.901 and max(fits) >= 0.85, f"{[round(f, 3) for f in fits]}")

# --- Preview on ---------------------------------------------------------------
w.preview_combo.setCurrentIndex(w.preview_combo.findData("main"))
o.set_preview(True)
pump(3)
for name in ov.VIEWS_2D:
    view = o.views[name]
    v, h = view.plane
    d = view.order[-3]
    rects = [np.asarray(r) for r in view.preview_outside.data]
    # inner edges of the four bands = the region's edges
    a0 = max(r[:, 0].max() for r in rects if r[:, 0].max() <= ranges[v][0] + 1e-9)
    edges_ok = (len(rects) == 4 and view.preview_outside.visible
                and abs(a0 - ranges[v][0]) < 1e-9)
    corners = sorted({round(float(c), 6) for r in rects for c in r[:, 0]})
    edges_ok &= round(ranges[v][0], 6) in corners and round(ranges[v][1], 6) in corners
    corners_h = sorted({round(float(c), 6) for r in rects for c in r[:, 1]})
    edges_ok &= round(ranges[h][0], 6) in corners_h and round(ranges[h][1], 6) in corners_h
    check(f"{name}: black bands end at the region's edges", edges_ok)
    # inside: black = set to 0 by the crop's own rule
    layer = view.preview_inside
    if not layer.visible:
        check(f"{name}: inside shown", False, "(no voxels set to 0 on this slice?)")
        continue
    black = np.asarray(layer.data)[..., 3] == 255
    step = float(layer.scale[0])
    origin = [float(t) for t in layer.translate]
    level = vc.Level("", None, [step, step, step], [origin[0], origin[1], o.cursor[d]])
    crop_keep = vc.outline_mask(level, [v, h, d], [0, 0, 0], black.shape + (1,),
                                box.outlines, spatial=spatial,
                                margin=box.margin * pixel)
    crop_keep = np.broadcast_to(crop_keep, black.shape + (1,))[..., 0]
    differ = black == crop_keep           # black where kept, or not black where 0
    frac = differ.mean()
    check(f"{name}: inside, black = voxels the crop sets to 0",
          frac < 0.02, f"({100 * frac:.2f} % of {black.size} grid points differ; "
          f"grid {step:.2f} µm, level {o.current_level(view)})")
    view.frame.grab().save(str(OUT / f"preview_{name}.png"))
viewer.window._qt_window.grab().save(str(OUT / "preview_window.png"))

# --- Timing: slider moves inside the region -----------------------------------
z0, z1 = ranges["z"]
zs = np.linspace(z0 + 0.2 * (z1 - z0), z0 + 0.8 * (z1 - z0), 10)
times = {}
for on in (True, False):
    o.set_preview(on)
    pump(1)
    start = time.perf_counter()
    for z in zs:
        o._set_cursor({"z": float(z)})
        QApplication.instance().processEvents()
    times[on] = (time.perf_counter() - start) / len(zs)
print(f"slider move: {1000 * times[True]:.0f} ms with preview, "
      f"{1000 * times[False]:.0f} ms without", flush=True)
check("slider move with preview under 0.5 s", times[True] < 0.5)
o.set_preview(True)
pump(1)

# --- A slice outside the region: all black ------------------------------------
o._set_cursor({"z": o.extent("z")[1] - 1.0})
pump(2)
xy = o.views["XY"]
check("slice outside the region: one rectangle over the view, nothing inside",
      len(xy.preview_outside.data) == 1 and not xy.preview_inside.visible)
o.views["XY"].frame.grab().save(str(OUT / "preview_XY_outside.png"))

# --- Projection: rectangle only -------------------------------------------------
o.set_projection("max", "full")
pump(3)
check("projection: four bands, no inside layer",
      len(xy.preview_outside.data) == 4 and not xy.preview_inside.visible)
o.set_projection("none", "full")
pump(2)

# --- Preview off ---------------------------------------------------------------
o.set_preview(False)
pump(1)
check("preview off: nothing drawn",
      not any(o.views[n].preview_inside.visible or o.views[n].preview_outside.visible
              for n in ov.VIEWS_2D))
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
viewer.close()
pump(0.5)
