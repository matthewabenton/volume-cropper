"""Seeds on each embryo of the demo: share held, size vs the true box
(diagnostic; growing off, margin 10 µm, as the GUI check)."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else
                str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import ndimage as ndi

from volume_cropper import autocrop as ac
from volume_cropper import core as vc

DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
image = vc.open_image(DEMO / "embryo_tissue_demo.zarr")
labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]
lev0 = image.levels[0]
nuclei = np.asarray(lev0.array[0])
scale = np.array([lev0.scale[i + 1] for i in range(3)])
offset = np.array([lev0.translation[i + 1] for i in range(3)])
boxes = ndi.find_objects(labels)


def seed_point(k):
    where = np.argwhere(labels == k)
    values = nuclei[tuple(where.T)]
    bright = where[values >= np.percentile(values, 90)]
    centre = where.mean(axis=0)
    best = bright[np.argmin(np.linalg.norm(bright - centre, axis=1))]
    return {a: float(offset[i] + best[i] * scale[i]) for i, a in enumerate("zyx")}


for k in range(1, 7):
    for grow in (False, True):
        started = time.time()
        r = ac.find_object(image, 0, margin=10.0, grow=grow, seed=seed_point(k))
        took = time.time() - started
        if not r.ranges:
            print(f"embryo {k} grow={grow}: nothing ({r.notes}) {took:.1f} s")
            continue
        where = np.argwhere(labels == k)
        pos = offset + where * scale
        inside = np.ones(len(where), bool)
        for i, a in enumerate("zyx"):
            inside &= (pos[:, i] >= r.ranges[a][0]) & (pos[:, i] <= r.ranges[a][1])
        true = [((s.start - 0.5) * sc + o, (s.stop - 0.5) * sc + o)
                for s, sc, o in zip(boxes[k - 1], scale, offset)]
        vol_true = np.prod([b - a for a, b in true])
        vol = np.prod([r.ranges[a][1] - r.ranges[a][0] for a in "zyx"])
        others = []
        for j in range(1, 7):
            if j == k:
                continue
            w = np.argwhere(labels[::4, ::4, ::4] == j) * 4
            p = offset + w * scale
            ins = np.ones(len(w), bool)
            for i, a in enumerate("zyx"):
                ins &= (p[:, i] >= r.ranges[a][0]) & (p[:, i] <= r.ranges[a][1])
            if ins.mean() > 0.05:
                others.append(f"{j}:{ins.mean():.0%}")
        print(f"embryo {k} grow={grow}: holds {inside.mean():.1%}, "
              f"volume {vol / vol_true:.2f} x true box (+margin), "
              f"threshold {r.thresholds[0]:.0f}, others in it: {others or '-'}, "
              f"{took:.1f} s", flush=True)
