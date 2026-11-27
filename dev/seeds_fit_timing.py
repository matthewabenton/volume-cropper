"""Where the time goes in Fit to seeds (0.18.0, the user asked for a
progress bar while shapes are computed from seeds): the search per seed,
the comparison of objects, and making each region and its outline. Hidden
window, the synthetic embryo demo, one seed per embryo (as
seeds_all_embryos.py), Outline ticked.

Usage: python seeds_fit_timing.py [TOOL_DIR]
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["VOLUME_CROPPER_SETTINGS"] = os.path.join(
    tempfile.mkdtemp(prefix="vc_settings_"), "settings.json")
TOOL = Path(sys.argv[1] if len(sys.argv) > 1 else
            str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(TOOL))
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import autocrop_panel  # noqa: E402
from volume_cropper import crop_gui  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data"
labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]
times = {"search": [], "same": [], "fitted": [], "holds": []}


def timed(name, function):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            times[name].append(time.perf_counter() - t0)
    return wrapper


viewer, panel = crop_gui.open_window(show=False)
panel.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
for _ in range(20):
    QApplication.processEvents()
    time.sleep(0.05)
group = panel.object_group
image = panel.image
lev0 = image.levels[0]
nuclei = np.asarray(lev0.array[0])
scale = np.array([lev0.scale[i + 1] for i in range(3)])
offset = np.array([lev0.translation[i + 1] for i in range(3)])
seeds = []
for k in range(1, int(labels.max()) + 1):
    where = np.argwhere(labels == k)
    values = nuclei[tuple(where.T)]
    bright = where[values >= np.percentile(values, 90)]
    best = bright[np.argmin(np.linalg.norm(bright - where.mean(axis=0),
                                           axis=1))]
    point = offset + best * scale
    seeds.append({"z": float(point[0]), "y": float(point[1]),
                  "x": float(point[2])})
group.seeds = seeds
group.seed_results = [None] * len(seeds)
group.outline_check.setChecked(True)
group._search = timed("search", group._search)
group._fitted = timed("fitted", group._fitted)
autocrop_panel._same_object = timed("same", autocrop_panel._same_object)
autocrop_panel.ac.mask_holds = timed("holds", autocrop_panel.ac.mask_holds)
t0 = time.perf_counter()
group.fit_to_seeds()
total = time.perf_counter() - t0
print(f"Fit to seeds, {len(seeds)} seeds: {total:.1f} s in all")
for name, values in times.items():
    print(f"  {name}: {len(values)} calls, {sum(values):.1f} s "
          f"(max {max(values, default=0):.1f} s)")
print(group.info.text())
viewer.close()
