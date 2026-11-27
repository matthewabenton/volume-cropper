"""Pictures of the embryo demo (make_demo_embryos.py), for checking it.

Usage (base-bia, which has matplotlib)::

    python preview_demo_embryos.py DEMO_FOLDER OUT_FOLDER

Always draws layout.png: each object's outline in projections along z, y
and x, numbered. If the image exists too: slices.png (XY slices at three
depths, an XZ and a YZ slice; the three channels as cyan, magenta and
yellow) and outlines.png (the same with each object's true outline dashed).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAME = "embryo_tissue_demo"
COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#a0a000"]


def outlines(ax, labels2d, scale=(1.0, 1.0), dashed=False, numbers=True):
    """Each object's edge in a 2D label image (and its number), drawn in
    the image's pixel units times scale (rows, columns)."""
    rows = np.arange(labels2d.shape[0]) * scale[0]
    cols = np.arange(labels2d.shape[1]) * scale[1]
    for k in range(1, 7):
        mask = labels2d == k
        if not mask.any():
            continue
        ax.contour(cols, rows, mask.astype(float), [0.5],
                   colors=COLOURS[k - 1], linewidths=1.6,
                   linestyles="--" if dashed else "-")
        if numbers:
            r, c = np.argwhere(mask).mean(axis=0) * np.asarray(scale)
            ax.text(c, r, str(k), color=COLOURS[k - 1], fontsize=13,
                    weight="bold", ha="center", va="center")


def composite(channels, limits):
    """Cyan, magenta and yellow channels added into one RGB picture."""
    rgb = np.zeros(channels[0].shape + (3,))
    colours = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for data, (low, high), colour in zip(channels, limits, colours):
        scaled = np.clip((data - low) / (high - low), 0, 1)
        rgb += scaled[..., None] * np.asarray(colour)
    return np.clip(rgb, 0, 1)


def main(demo, out):
    out.mkdir(parents=True, exist_ok=True)
    truth = np.load(demo / f"{NAME}_truth.npz")
    labels, voxel = truth["labels"], truth["voxel_um"]
    names = json.loads(str(truth["names"]))
    nz, ny, nx = labels.shape
    ext = {"z": nz * voxel[0], "y": ny * voxel[1], "x": nx * voxel[2]}
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    views = [("along z (y down, x across)", 0, "y", "x"),
             ("along y (z down, x across)", 1, "z", "x"),
             ("along x (z down, y across)", 2, "z", "y")]
    size = dict(zip("zyx", voxel))
    for ax, (title, axis, down, across) in zip(axes, views):
        for k in range(1, 7):     # every object's projected outline
            proj = (labels == k).any(axis=axis).astype(np.uint8) * k
            outlines(ax, proj, (size[down], size[across]))
        ax.set_xlim(0, ext[across])
        ax.set_ylim(ext[down], 0)
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_title(f"projection {title}, µm", fontsize=10)
    fig.legend([plt.Line2D([], [], color=c) for c in COLOURS],
               [f"{k} {n}" for k, n in enumerate(names, start=1)],
               loc="lower center", ncol=3, fontsize=10)
    fig.savefig(out / "layout.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    zarr_path = demo / f"{NAME}.zarr"
    if not zarr_path.exists():
        print("layout.png only (no image yet)")
        return
    import zarr
    image = zarr.open_group(zarr_path, mode="r")["0"]
    limits = []
    sample = image[:, ::4, ::4, ::4]
    for c in range(image.shape[0]):
        limits.append((float(np.percentile(sample[c], 30)),
                       float(np.percentile(sample[c], 99.8))))
    zs = [int(nz * f) for f in (0.3, 0.5, 0.72)]
    yc, xc = ny // 2, nx // 2
    panels = [(f"XY at z = {z * voxel[0]:.0f} µm", image[:, z],
               labels[z], voxel[1] / voxel[2]) for z in zs]
    panels.append((f"XZ at y = {yc * voxel[1]:.0f} µm", image[:, :, yc],
                   labels[:, yc], voxel[0] / voxel[2]))
    panels.append((f"YZ at x = {xc * voxel[2]:.0f} µm", image[:, :, :, xc],
                   labels[:, :, xc], voxel[0] / voxel[1]))
    for name, dashed in (("slices.png", False), ("outlines.png", True)):
        fig, axes = plt.subplots(2, 3, figsize=(19, 13))
        for ax, (title, data, lab, aspect) in zip(axes.flat, panels):
            ax.imshow(composite(list(data), limits), aspect=aspect,
                      interpolation="nearest")
            if dashed:
                outlines(ax, lab, dashed=True)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0.02, 0.5, "\n".join(
            ["cyan: nuclei", "magenta: membranes", "yellow: marker", ""]
            + [f"{k} {n}" for k, n in enumerate(names, start=1)]),
            fontsize=12, va="center")
        fig.savefig(out / name, dpi=70, bbox_inches="tight")
        plt.close(fig)
    # one XY slice at full resolution, for the texture
    z = zs[1]
    crop = image[:, z, ny // 4: ny // 4 + 250, nx // 4: nx // 4 + 250]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(composite(list(crop), limits), interpolation="nearest")
    ax.set_title(f"XY at z = {z * voxel[0]:.0f} µm, 500 x 500 µm, full "
                 "resolution", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out / "detail.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("layout.png, slices.png, outlines.png, detail.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
