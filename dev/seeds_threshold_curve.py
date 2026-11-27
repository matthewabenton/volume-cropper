"""Size of the seed's piece against the threshold, per demo embryo."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage as ndi

from volume_cropper import autocrop as ac
from volume_cropper import core as vc

DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
image = vc.open_image(DEMO / "embryo_tissue_demo.zarr")
labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]
lev0 = image.levels[0]
nuclei = np.asarray(lev0.array[0])
level = ac.choose_level(image)
f = np.array(vc.downsampling_factors(image)[level][1:])
raw = np.asarray(image.levels[level].array[0]).astype(np.float32)
data = ndi.gaussian_filter(raw, 1.0)
valid = np.ones(data.shape, bool)
auto = ac.threshold_of(data.ravel(), "otsu")
bg = data <= auto
median, sigma = ac._robust(raw[bg])
median_s, _ = ac._robust(data[bg])
floor = median_s + 3 * sigma
print(f"automatic {auto:.0f}, floor {floor:.0f}, volume {data.size} voxels")
small_labels = labels[::f[0], ::f[1], ::f[2]][:data.shape[0], :data.shape[1],
                                               :data.shape[2]]
levels = np.linspace(auto, floor, 24)
for k in range(1, 7):
    where = np.argwhere(labels == k)
    values = nuclei[tuple(where.T)]
    bright = where[values >= np.percentile(values, 90)]
    centre = where.mean(axis=0)
    best = bright[np.argmin(np.linalg.norm(bright - centre, axis=1))]
    seed = tuple(best // f)
    true = int((small_labels == k).sum())
    row = []
    for t in levels:
        mask = data > t
        lab, n = ndi.label(mask)
        sizes = ndi.sum_labels(mask, lab, np.arange(1, n + 1))
        b = ac._chosen_piece(lab, n, sizes, seed, None)
        if b is None:
            row.append("-")
            continue
        piece = lab == b + 1
        own = (small_labels[piece] == k).mean()
        row.append(f"{int(sizes[b]) / true:.2f}({own:.0%})")
    print(f"embryo {k} (true {true} voxels): " + " ".join(
        f"{t:.0f}:{r}" for t, r in zip(levels, row)), flush=True)
