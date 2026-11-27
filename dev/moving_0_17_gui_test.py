"""0.17.0 moving regions, in a VISIBLE window, on a synthetic time-lapse.

Usage: python moving_0_17_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)

Makes OUT_DIR/drifting_embryo.zarr (OME-Zarr 0.5, t c z y x = 8 x 2 x 48
x 160 x 320, voxels 4 x 2 x 2 µm, 15 min per frame, 3 levels): an
"embryo" (a textured ellipsoid) that drifts about 300 µm along x and 40 µm
along y over 8 timepoints and grows by a third, in Poisson noise.

Checks: Find a bright object has "Moving region" ticked for this
time-lapse; Fit makes a moving region that holds the embryo's true centre
at the first and last timepoint, in the views (the region as cropped and
the rectangle in XY); the crop is checked OK and has a positions file
with 8 timepoints; in the crop the embryo stays in place (its centre of
mass moves by at most 3 voxels; the same region fixed in place: printed
for comparison). Then outlines per timepoint (Moving region + Outline):
the warning shows; Fit gives an outline at all 8 timepoints; the views
show each timepoint's own; the crop is checked OK and keeps >= 99 % of
the true embryo's voxels at every timepoint; fit and crop times with and
without outlines are printed (for the warning); the outlined crop opens
in the window. Saves moving_t1.png, moving_t8.png (the window at the
first and last timepoint, 3D clipped to the region), outlined_t1.png,
outlined_t8.png (with outlines per timepoint, 3D "dim") and crop_t1.png,
crop_t8.png (the outlined crop opened). Synthetic
data only. Answers every dialog itself and closes its windows. Uses a
settings file of its own in OUT_DIR.
"""
import faulthandler
import os
import shutil
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
import zarr  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402
from zarr.codecs import BloscCodec  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

results, messages = [], []
T, SHAPE = 8, (48, 160, 320)            # timepoints; z, y, x voxels
SCALE = (4.0, 2.0, 2.0)                 # µm
INTERVAL = 15.0                         # minutes


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def centre_at(t):
    """The embryo's true centre (z, y, x in µm) at timepoint t."""
    return (96.0, 140.0 + 40.0 * t / (T - 1), 140.0 + 300.0 * t / (T - 1))


def make_image(path):
    """The drifting, growing embryo as an OME-Zarr 0.5 (3 levels)."""
    rng = np.random.default_rng(0)
    grids = np.meshgrid(*[(np.arange(n) * s) for n, s in zip(SHAPE, SCALE)],
                        indexing="ij")
    texture = rng.normal(1.0, 0.25, size=SHAPE)
    data = np.empty((T, 2, *SHAPE), dtype=np.uint16)
    for t in range(T):
        grow = 1.0 + t / (3 * (T - 1))
        radii = (36.0 * grow, 45.0 * grow, 70.0 * grow)
        inside = sum(((g - c) / r) ** 2 for g, c, r in
                     zip(grids, centre_at(t), radii))
        embryo = np.where(inside <= 1.0, 1500.0 * texture, 0.0)
        rim = np.where(np.abs(inside - 1.0) < 0.12, 600.0, 0.0)
        data[t, 0] = rng.poisson(100 + embryo.clip(0))
        data[t, 1] = rng.poisson(80 + rim)
    root = zarr.open_group(path, mode="w", zarr_format=3)
    datasets = []
    level = data
    for k, f in enumerate([(1, 1, 1), (1, 2, 2), (2, 4, 4)]):
        if k:
            step = [a // b for a, b in zip(f, prev)]
            n = [s // d for s, d in zip(level.shape[2:], step)]
            level = level[:, :, :n[0] * step[0], :n[1] * step[1],
                          :n[2] * step[2]].reshape(
                T, 2, n[0], step[0], n[1], step[1], n[2], step[2]).mean(
                axis=(3, 5, 7)).astype(np.uint16)
        prev = f
        arr = root.create_array(
            name=str(k), shape=level.shape, dtype=level.dtype,
            chunks=(1, 1, 16, 32, 32), shards=(1, 1, 16, 64, 64),
            compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
            fill_value=0, dimension_names=list("tczyx"))
        arr[...] = level
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": [INTERVAL, 1.0] + [
                s * n for s, n in zip(SCALE, f)]}]})
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "name": "drifting embryo",
        "axes": [{"name": "t", "type": "time", "unit": "minute"},
                 {"name": "c", "type": "channel"}] + [
            {"name": a, "type": "space", "unit": "micrometer"}
            for a in "zyx"],
        "datasets": datasets}],
        "omero": {"channels": [
            {"label": "embryo", "color": "00FF00",
             "window": {"start": 0, "end": 2500, "min": 0, "max": 65535}},
            {"label": "rim", "color": "FF00FF",
             "window": {"start": 0, "end": 800, "min": 0, "max": 65535}}]}}


def x_centres(crop_path):
    """The embryo's centre of mass along x (voxels) per timepoint of a
    crop (channel 0, level 0)."""
    data = np.asarray(vc.open_image(crop_path).levels[0].array[:, 0])
    weights = np.clip(data.astype(float) - 400, 0, None)
    x = np.arange(data.shape[-1])
    return [float((w.sum(axis=(0, 1)) * x).sum() / w.sum()) for w in weights]


path = OUT / "drifting_embryo.zarr"
if path.exists():
    shutil.rmtree(path)
make_image(path)
print("made", path, flush=True)

load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
for name in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, name, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True

viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1600, 1000)
w.open_image(path, choice=LoadChoice())
pump(3)
group = w.object_group
w.sections["find"].set_open(True)        # (shown in the screenshots)
group.setChecked(True)
group.margin_spin.setValue(10.0)
check("Moving region is ticked and usable for a time-lapse (Outline too, "
      "no warning while Outline is off)",
      group.moving_check.isChecked() and group.moving_check.isEnabled()
      and group.outline_check.isEnabled()
      and not group.timed_note.isVisible())
t0 = time.perf_counter()
group.fit()
fit_plain = time.perf_counter() - t0
pump(1)
box = w.boxes[-1]
check("Fit makes a moving region", len(box.track) == T,
      f"({len(box.track)} timepoints, {fit_plain:.1f} s): "
      + group.info.text().splitlines()[0])
i = len(w.boxes) - 1
w._refresh(select=i)
w.show_3d_check.setChecked(True)
pump(3)
w.ortho.set_clip(True)
pump(2)
for t, name in ((0, "moving_t1.png"), (T - 1, "moving_t8.png")):
    w.ortho._set_cursor({"t": t * INTERVAL, "z": centre_at(t)[0]})
    pump(2)
    cropped = w.ortho.box_ranges[i]
    xy = w.ortho.views["XY"]
    corners = np.asarray(xy.shapes.data[xy.shape_boxes.index(i)])[:, -2:]
    z, y, x = centre_at(t)
    inside = all(cropped[a][0] < v < cropped[a][1]
                 for a, v in zip("zyx", (z, y, x)))
    in_rect = (corners[:, 0].min() < y < corners[:, 0].max()
               and corners[:, 1].min() < x < corners[:, 1].max())
    check(f"timepoint {t + 1}: the region holds the embryo's centre",
          inside and in_rect,
          f"(x {cropped['x'][0]:.0f}-{cropped['x'][1]:.0f} µm, centre {x:.0f})")
    viewer.screenshot(str(OUT / name), canvas_only=False)
print("plan:", w.plan_label.text().replace("\n", " | "), flush=True)

crops = OUT / "crops"
if crops.exists():
    shutil.rmtree(crops)
t0 = time.perf_counter()
result = vc.crop_box(w.image, box, crops)
crop_plain = time.perf_counter() - t0
check("the moving crop is checked OK", result.problems == [],
      f"({crop_plain:.1f} s; {result.problems[:2]})")
rows = result.positions.read_text().splitlines() if result.positions else []
check("positions file with every timepoint", len(rows) == T + 1,
      f"({result.positions.name if result.positions else None}: "
      f"{rows[1] if len(rows) > 1 else ''} ... {rows[-1] if rows else ''})")
moving = x_centres(result.path)
spread = max(moving) - min(moving)
check("in the crop the embryo stays in place (x)", spread <= 3.0,
      f"(centre of mass moves {spread:.1f} voxels over {T} timepoints)")
fixed = vc.crop_box(w.image, vc.fixed_in_place(w.image, box),
                    crops / "fixed")
still = x_centres(fixed.path)
print(f"for comparison, the region fixed in place: {fixed.plan.shapes[0]} "
      f"voxels, the embryo moves {max(still) - min(still):.1f} voxels; "
      f"moving: {result.plan.shapes[0]}", flush=True)

# --- outlines per timepoint (Moving region + Outline) ---------------------
group.undo()                      # (the moving region again: removed)
pump(0.5)
group.outline_check.setChecked(True)
check("the warning shows with Moving region and Outline ticked",
      group.timed_note.isVisible(), group.timed_note.text()[:60] + "…")
t0 = time.perf_counter()
group.fit()
fit_timed = time.perf_counter() - t0
pump(1)
box = w.boxes[-1]
i = len(w.boxes) - 1
check("Fit gives an outline at every timepoint",
      len(box.timed_outlines) == T,
      f"({len(box.timed_outlines)}; fit {fit_timed:.1f} s vs "
      f"{fit_plain:.1f} s without)")
w._refresh(select=i)
w.ortho.set_mask_mode("dim")
for t, name in ((0, "outlined_t1.png"), (T - 1, "outlined_t8.png")):
    w.ortho._set_cursor({"t": t * INTERVAL, "z": centre_at(t)[0]})
    pump(2)
    shown = w.ortho.boxes[i].outlines
    check(f"timepoint {t + 1}: the views show that timepoint's outlines",
          shown == vc.timed_outlines_at(box, t * INTERVAL) and bool(shown))
    viewer.screenshot(str(OUT / name), canvas_only=False)
print("plan:", w.plan_label.text().replace("\n", " | "), flush=True)

t0 = time.perf_counter()
outlined = vc.crop_box(w.image, box, crops / "outlined")
crop_timed = time.perf_counter() - t0
check("the outlined crop is checked OK", outlined.problems == [],
      f"({crop_timed:.1f} s vs {crop_plain:.1f} s without outlines; "
      f"{outlined.problems[:2]})")
# the true embryo, in the crop's voxels (via the positions file)
data = np.asarray(vc.open_image(outlined.path).levels[0].array[:, 0])
rows = [r.split(",") for r in outlined.positions.read_text().splitlines()[1:]]
worst, cut = 1.0, []
for t in range(T):
    origin = [float(v) for v in rows[t][2:]]
    grids = np.meshgrid(*[o + np.arange(n) * s for o, n, s in
                          zip(origin, data.shape[1:], SCALE)], indexing="ij")
    grow = 1.0 + t / (3 * (T - 1))
    radii = (36.0 * grow, 45.0 * grow, 70.0 * grow)
    embryo = sum(((g - c) / r) ** 2 for g, c, r in
                 zip(grids, centre_at(t), radii)) <= 1.0
    worst = min(worst, float((data[t][embryo] != 0).mean()))
    cut.append(float((data[t] == 0).mean()))
check("the outlined crop keeps the embryo at every timepoint", worst >= 0.99,
      f"(at least {worst:.1%} of its voxels; {min(cut):.0%}-{max(cut):.0%} "
      "of each timepoint set to 0)")

w.boxes.clear()                   # (else: "discard the regions?")
w._refresh()
w.open_image(outlined.path, choice=LoadChoice())
pump(3)
check("the crop opens", w.image is not None
      and w.image.path == outlined.path)
for t, name in ((0, "crop_t1.png"), (T - 1, "crop_t8.png")):
    w.ortho._set_cursor({"t": t * INTERVAL})
    pump(2)
    viewer.screenshot(str(OUT / name), canvas_only=False)

print(f"\n{sum(results)} of {len(results)} checks passed.")
if messages:
    print("messages:", messages)
pump(1)
w.boxes.clear()
QApplication.instance().closeAllWindows()
pump(1)
