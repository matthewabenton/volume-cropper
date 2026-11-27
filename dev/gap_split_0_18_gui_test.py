"""0.18.0 splitting touching objects between seeds, with gap lines (the
user's wish of 2026-11-13, "as in ilastik"), in a VISIBLE window on
synthetic data: two touching ellipsoids whose contact is tilted 30 deg to
the views.

- Paint gap: a real mouse drag in the XY view draws a gap line (white
  dash-dot on a dark edge); a click on it removes it; dragging does not
  pan meanwhile;
- Fit to seeds with one seed per object: two regions (Split on), with
  how much of each object its part holds and of the other;
- the same with Split off: one region.

Usage: python gap_split_0_18_gui_test.py OUT_DIR

Saves gap_line.png and split.png. Synthetic data only; closes its window;
settings file of its own in OUT_DIR.
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
from qtpy.QtCore import QEvent, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QMouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import autocrop as ac  # noqa: E402
from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402

results = []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


# two touching ellipsoids (semi-axes 20, 26, 34 voxels of 2 µm), centres
# 56 voxels apart along a line tilted 30 deg from x towards z
shape = (90, 140, 200)
scale = 2.0
middle = np.array([45.0, 70.0, 100.0])
angle = np.radians(30.0)
direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
centres = [middle - 28.0 * direction, middle + 28.0 * direction]
grid = np.indices(shape, dtype=np.float32)


def inside(centre):
    # each ellipsoid's long axis along the line joining the centres
    d = np.stack([grid[i] - centre[i] for i in range(3)])
    along = sum(d[i] * direction[i] for i in range(3))
    across_y = d[1]
    across = d - along * direction[:, None, None, None]
    other = np.sqrt(np.maximum((across ** 2).sum(axis=0) - across_y ** 2, 0))
    return (along / 34.0) ** 2 + (across_y / 26.0) ** 2 + (other / 20.0) ** 2 \
        <= 1.0


objects = [inside(c) for c in centres]
truth = np.where(objects[0], 1, 0) + np.where(objects[1] & ~objects[0], 2, 0)
rng = np.random.default_rng(0)
signal = 100.0 + 1400.0 * (objects[0] | objects[1])
data = rng.poisson(signal).astype(np.uint16)
print(f"   overlap of the two ellipsoids: "
      f"{int((objects[0] & objects[1]).sum())} voxels", flush=True)

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
group.margin_spin.setValue(0.0)
ortho._set_cursor({"z": float(middle[0] * scale)})
pump(0.5)

# --- a gap line, drawn with the mouse in the XY view ----------------------
xy = ortho.views["XY"]
canvas = ortho._qt_viewers["XY"].canvas
widget = canvas.native


def on_screen(y, x):
    """A point (µm) of the XY view on its canvas, in pixels."""
    transform = canvas.view.transform * canvas.view.scene.transform
    mapped = transform.map([x, y, 0.0, 1.0])
    return mapped[:2] / mapped[3]


def mouse(kind, xy_px, buttons):
    event = QMouseEvent(kind, QPointF(*xy_px),
                        QPointF(widget.mapToGlobal(QPointF(*xy_px).toPoint())),
                        Qt.MouseButton.LeftButton
                        if kind != QEvent.Type.MouseMove
                        else Qt.MouseButton.NoButton,
                        buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)
    pump(0.03)


def drag(points_px):
    mouse(QEvent.Type.MouseButtonPress, points_px[0],
          Qt.MouseButton.LeftButton)
    for p in points_px[1:]:
        mouse(QEvent.Type.MouseMove, p, Qt.MouseButton.LeftButton)
    mouse(QEvent.Type.MouseButtonRelease, points_px[-1],
          Qt.MouseButton.NoButton)
    pump(0.4)


# where the contact (the plane halfway, normal = direction) cuts the slice
z = middle[0]
x_contact = middle[2] - (z - middle[0]) * np.tan(angle)   # voxels
ys = np.linspace(30, 110, 14)                             # voxels
wiggle = 1.5 * np.sin(np.linspace(0, 3 * np.pi, len(ys)))
path = [on_screen(y * scale, (x_contact + dx) * scale)
        for y, dx in zip(ys, wiggle)]
group.gap_button.setChecked(True)
pump(0.2)
cam = ov.camera(xy.model)
centre_before = tuple(cam.center)
drag(path)
check("Paint gap: a drag in XY draws a gap line",
      len(group.gap_lines) == 1 and len(group.gap_lines[0]["points"]) >= 5,
      f"({len(group.gap_lines)} line(s), "
      f"{len(group.gap_lines[0]['points']) if group.gap_lines else 0} "
      "points)")
check("dragging does not pan meanwhile",
      np.allclose(tuple(cam.center), centre_before))
check("it shows as a white dash-dot line on a dark edge",
      xy.gap_line.visible and xy.gap_edge.visible
      and len(xy.gap_line.data) > len(xy.gap_edge.data))
if group.gap_lines:
    line = np.asarray(group.gap_lines[0]["points"]) / scale
    print(f"   line in voxels: y {line[:, 0].min():.0f}-"
          f"{line[:, 0].max():.0f}, x {line[:, 1].min():.1f}-"
          f"{line[:, 1].max():.1f} (contact at x {x_contact:.1f})",
          flush=True)
viewer.screenshot(str(OUT / "gap_line.png"), canvas_only=False)
# a click on it removes it; then draw it again
middle_px = path[len(path) // 2]
mouse(QEvent.Type.MouseButtonPress, middle_px, Qt.MouseButton.LeftButton)
mouse(QEvent.Type.MouseButtonRelease, middle_px, Qt.MouseButton.NoButton)
pump(0.4)
check("a click on the line removes it", not group.gap_lines)
drag(path)
group.gap_button.setChecked(False)
pump(0.2)
check("Paint gap off: panning back", cam.mouse_pan and not ortho.gap_mode)

# --- Fit to seeds, one seed per object -------------------------------------
group.seeds = [dict(zip("zyx", c * scale)) for c in centres]
group.seed_results = [None, None]
ortho.set_seeds(group.seeds)
start = time.time()
group.fit_to_seeds()
took = time.time() - start
pump(0.5)
print("   " + group.info.text().replace("\n", "\n   "), flush=True)
check("Fit to seeds splits the object between its seeds",
      len(w.boxes) == 2 and "split between them" in group.info.text(),
      f"({len(w.boxes)} regions, {took:.1f} s)")
# how well: the parts' masks against the truth (at the level searched)
image = w.image
region = {a: ortho.extent(a) for a in image.range_axes}
seeds = group.seeds
found = [ac.find_object(image, None, region=region, seed=s, grow=False,
                        keep_mask=True) for s in seeds]
level = found[0].level
for with_gap in (False, True):
    parts = ac.split_objects(image, None, found, seeds,
                             group.gap_lines if with_gap else [], grow=False)
    f = 2 ** level
    small = truth[::f, ::f, ::f][tuple(slice(0, n) for n in
                                       parts[0].mask.shape)]
    scores = []
    for k, part in enumerate(parts, start=1):
        own, other = small == k, small == 3 - k
        scores.append(f"{(part.mask & own).sum() / own.sum():.1%} of its "
                      f"own, {(part.mask & other).sum() / other.sum():.1%} "
                      "of the other")
    print(f"   level {level}, {'with' if with_gap else 'without'} the gap "
          f"line: " + "; ".join(scores), flush=True)
viewer.screenshot(str(OUT / "split.png"), canvas_only=False)
# Split off: one region
group.undo()
group.split_check.setChecked(False)
group.seed_results = [None, None]
group.fit_to_seeds()
pump(0.3)
check("Split off: one region, as before", len(w.boxes) == 1,
      f"({len(w.boxes)})")
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
