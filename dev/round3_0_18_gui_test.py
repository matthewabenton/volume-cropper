"""0.18.0, the user's wishes of 2026-11-13, in a VISIBLE window on the
synthetic embryo demo:
1. opening with a chosen 2D resolution zooms in to it (the view titles
   name that level; one voxel per screen pixel); with Auto, the whole image;
2. seeds are shown in the 3D view too;
3. Fit to seeds shows one progress window for all seeds whose bar moves
   within each seed's search, and a last step for the regions.

Usage: python round3_0_18_gui_test.py OUT_DIR

Saves zoom_level0.png, zoom_auto.png, seeds_3d.png (the window) and
fit_progress.png (the progress window, mid-way). Synthetic data only.
Answers every dialog itself and closes its window. Uses a settings file of
its own in OUT_DIR.
"""
import faulthandler
import os
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
from qtpy.QtWidgets import QApplication, QProgressDialog  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data"
results, messages = [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def settled():
    """Wait until the "Loading…" box has gone."""
    end = time.time() + 30
    while time.time() < end:
        pump(0.2)
        if w._loading is None:
            break
    pump(1.0)


load_dialog.FileLoadDialog.exec = lambda self: 1
for kind in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, kind, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True
viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
ortho = w.ortho

# --- 1. zoom on opening ---------------------------------------------------
w.open_image(DEMO / "embryo_tissue_demo.zarr",
             choice=LoadChoice(level_2d=0))
settled()
xy = ortho.views["XY"]
zoom0 = ov.camera(xy.model).zoom
voxel0 = min(w.image.levels[0].scale[-2:])
check("opening at level 0 zooms in to one voxel per screen pixel",
      abs(zoom0 - 1 / voxel0) < 1e-6 and "level 0" in xy.title.text(),
      f"zoom {zoom0:.3f} (1 / {voxel0} µm), title {xy.title.text()!r}")
centre = [float(np.mean(ortho.extent(a))) for a in xy.plane]
check("... centred on the image",
      np.allclose(tuple(ov.camera(xy.model).center)[-2:], centre),
      f"{tuple(ov.camera(xy.model).center)[-2:]}")
viewer.screenshot(str(OUT / "zoom_level0.png"), canvas_only=False)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
settled()
zoom_auto = ov.camera(xy.model).zoom
check("with Auto, the whole image (zoomed out further)",
      zoom_auto < zoom0, f"zoom {zoom_auto:.3f}, title {xy.title.text()!r}")
viewer.screenshot(str(OUT / "zoom_auto.png"), canvas_only=False)

# --- 2. seeds in 3D ---------------------------------------------------------
w.show_3d_check.setChecked(True)
settled()
labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]
lev0 = w.image.levels[0]
scale, offset = np.array(lev0.scale[1:]), np.array(lev0.translation[1:])
seeds = []
for k in range(1, int(labels.max()) + 1):
    point = offset + np.argwhere(labels == k).mean(axis=0) * scale
    seeds.append({"z": float(point[0]), "y": float(point[1]),
                  "x": float(point[2])})
group = w.object_group
group.seeds = seeds
group.seed_results = [None] * len(seeds)
ortho.set_seeds(seeds)
pump(1.0)
view3d = ortho.views["3D"]
check("the seeds are shown in 3D",
      view3d.seeds is not None and view3d.seeds.visible
      and len(view3d.seeds.data) == len(seeds), f"{len(seeds)} seeds")
viewer.screenshot(str(OUT / "seeds_3d.png"), canvas_only=False)

# --- 3. the progress window during Fit to seeds -------------------------------
shown, grabbed = [], []
original = group._show_step


def show_step(dialog, text, share):
    original(dialog, text, share)
    shown.append((share, text, dialog.isVisible()))
    if not grabbed and share > 0.3 and dialog.isVisible():
        dialog.grab().save(str(OUT / "fit_progress.png"))
        grabbed.append(text)


group._show_step = show_step
group.outline_check.setChecked(True)
group.fit_to_seeds()
pump(0.5)
shares = [s for s, _, _ in shown]
texts = [t for _, t, _ in shown]
check("one progress window, its bar moving within each seed's search",
      len(shares) > 3 * len(seeds) and shares == sorted(shares)
      and max(shares) >= 0.95 and any(v for _, _, v in shown),
      f"{len(shares)} updates, {min(shares):.2f} -> {max(shares):.2f}")
check("it names the seed and, at the end, the regions",
      any(f"Seed 3 of {len(seeds)}" in t for t in texts)
      and any("Making the regions" in t for t in texts),
      f"e.g. {grabbed[:1]}")
check("it is closed afterwards",
      not any(d.isVisible() for d in QApplication.topLevelWidgets()
              if isinstance(d, QProgressDialog)))
print(group.info.text().replace("\n", " | "))

print("dialogs answered:", messages)
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
