"""0.20.0 crossing marks (the user's choice of 2026-11-17, "A and D"): where
the × marks crowd, as with the Autocropper's outlines per slice, each
view's marks are thinned to ortho_views.MARK_SPACING on screen; View:
Show "× marks" switches them off (also a Settings start option). In a
VISIBLE window on synthetic data: two touching ellipsoids split between
two seeds, whose parts get outlines per slice (as
outlines_3d_0_18_gui_test.py).

- counts the selected region's marks in the XZ and YZ views without
  thinning (as 0.18) and with it, and checks the kept marks' spacing;
- the YZ view zoomed in 3x: more marks come back;
- unticking "× marks" removes them; the Settings dialog has the option;
- the View part's Show row fits the panel.

Usage: python marks_0_20_gui_test.py OUT_DIR

Saves marks_before.png (no thinning), marks_after.png, marks_off.png,
view_part.png and settings.png. Synthetic data only; closes its window;
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
from qtpy.QtWidgets import QApplication  # noqa: E402

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


# two touching ellipsoids (as outlines_3d_0_18_gui_test.py)
shape = (90, 140, 200)
scale = 2.0
middle = np.array([45.0, 70.0, 100.0])
angle = np.radians(30.0)
direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
centres = [middle - 28.0 * direction, middle + 28.0 * direction]
grid = np.indices(shape, dtype=np.float32)


def inside(centre):
    d = np.stack([grid[i] - centre[i] for i in range(3)])
    along = sum(d[i] * direction[i] for i in range(3))
    across = d - along * direction[:, None, None, None]
    other = np.sqrt(np.maximum((across ** 2).sum(axis=0) - d[1] ** 2, 0))
    return (along / 34.0) ** 2 + (d[1] / 26.0) ** 2 + (other / 20.0) ** 2 \
        <= 1.0


union = inside(centres[0]) | inside(centres[1])
rng = np.random.default_rng(0)
data = rng.poisson(100.0 + 1400.0 * union).astype(np.uint16)

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
group.margin_spin.setValue(4.0)
group.outline_check.setChecked(True)
ortho._set_cursor({"z": float(middle[0] * scale)})
group.seeds = [dict(zip("zyx", c * scale)) for c in centres]
group.seed_results = [None, None]
ortho.set_seeds(group.seeds)
group.fit_to_seeds()
pump(0.5)
box = w.boxes[0]
n_polygons = sum(len(v) for v in box.outlines.values())
check("the first part has outlines per slice", n_polygons >= 10,
      f"({n_polygons} XY polygons)")
w._refresh(select=0)
# the XZ and YZ views through the first part's centre
ortho._set_cursor({"z": float(centres[0][0] * scale),
                   "y": float(centres[0][1] * scale),
                   "x": float(centres[0][2] * scale)})
pump(1.0)


def count(view_name):
    """The selected region's × marks drawn in a view now (and their
    centres): the second halves of the segments are the other strokes."""
    view = ortho.views[view_name]
    segments, _ = ortho._crossing_marks(box, view, ortho._zoom(view))
    centres_ = np.asarray(segments).mean(axis=1)[: len(segments) // 2]
    return centres_, ortho._zoom(view)


# without thinning (as 0.18): MARK_SPACING 0
spacing = ov.MARK_SPACING
ov.MARK_SPACING = 0.0
for name in ov.VIEWS_2D:
    ortho._sync_outlines(ortho.views[name])
pump(0.8)
before = {name: len(count(name)[0]) for name in ("XZ", "YZ")}
viewer.screenshot(str(OUT / "marks_before.png"), canvas_only=False)
ov.MARK_SPACING = spacing
for name in ov.VIEWS_2D:
    ortho._sync_outlines(ortho.views[name])
pump(0.8)
after = {}
for name in ("XZ", "YZ"):
    centres_, zoom = count(name)
    after[name] = len(centres_)
    gaps = np.linalg.norm(centres_[:, None] - centres_[None], axis=2)
    closest = gaps[np.triu_indices(len(centres_), 1)].min() * zoom \
        if len(centres_) > 1 else np.inf
    check(f"{name}: marks thinned, none closer than {spacing:.0f} px",
          after[name] < before[name] and closest >= spacing - 1e-6,
          f"({before[name]} -> {after[name]} marks; closest "
          f"{closest:.0f} px; zoom {zoom:.2f} px/µm)")
viewer.screenshot(str(OUT / "marks_after.png"), canvas_only=False)
# zooming in brings marks back
yz = ortho.views["YZ"]
camera = ov.camera(yz.model)
camera.zoom = camera.zoom * 3.0
pump(1.0)
zoomed = len(count("YZ")[0])
check("YZ zoomed in 3x: more marks shown", zoomed > after["YZ"],
      f"({after['YZ']} -> {zoomed})")
camera.zoom = camera.zoom / 3.0
pump(0.8)
# switched off
w.marks_check.setChecked(False)
pump(0.8)
check("unticked: no marks", not ortho.show_marks
      and len(count("XZ")[0]) == 0)
viewer.screenshot(str(OUT / "marks_off.png"), canvas_only=False)
w.marks_check.setChecked(True)
pump(0.5)
# the View part's Show row fits the panel
part = w.sections["view"]
part.grab().save(str(OUT / "view_part.png"))
row_right = w.marks_check.mapTo(part, w.marks_check.rect().topRight()).x()
check("the Show row fits the panel", row_right <= part.width(),
      f"(× marks ends at {row_right} of {part.width()} px)")
dialog = crop_gui.SettingsDialog(w)
dialog.show()
pump(0.8)
dialog.grab().save(str(OUT / "settings.png"))
check("the Settings dialog has the × marks start option",
      "crossing_marks" in dialog.view_widgets
      and dialog.values()["view"]["crossing_marks"] is True)
dialog.reject()
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
