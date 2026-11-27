"""Make the cover image (GitHub's "social preview", 1280 x 640 px) from the demo.

Usage (napari-env):
    python dev/make_cover_image.py OUT.png

Uses only the synthetic embryo demo (`embryo_tissue_demo.zarr` and its true
labels), from the folder beside the tool folder or the VC_DEMO environment
variable; the tool's own code (the folder above this script, or VC_TOOL)
decides which voxels the crop keeps. The picture: the tool's name and its
idea ("Draw regions at low resolution. Save them at full resolution.") on
the left; on the right the demo as the window shows it: three linked views
(XY, YZ, XZ; maximum projections of level 1, half resolution) with the
crop regions as solid lines (the selected one wider) and the selected
region's polygons as dashed lines. The polygons are silhouettes, as drawn
on projections (*2D projection: Max*); their corners are traced from the
demo's true labels, about as a person would click them. Bottom right: the
crop's 3D preview at full resolution (level 0, made isotropic), turned: the
voxels the polygons keep shown normally, the others dimmed to 25 % (the
window's "dim"), the shape's outline dashed and the region's box as thin
lines. GitHub wants PNG, JPG or GIF under 1 MB, at least 640 x 320;
1280 x 640 is its recommended size.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw into a file, no window
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import zarr                           # noqa: E402
from matplotlib import font_manager   # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from scipy import ndimage as ndi      # noqa: E402
from skimage import measure           # noqa: E402

OUT = Path(sys.argv[1])
DEMO = Path(os.environ.get("VC_DEMO", Path(__file__).resolve().parents[2]
                           / "volume_cropper_demo_data"))
TOOL = Path(os.environ.get("VC_TOOL", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(TOOL))
from volume_cropper import core as vc  # noqa: E402

DIM = 0.25                            # the window's "dim" (ortho_views.py)
DASHES = (0, (4, 3))                  # dashed line: 4 on, 3 off (points)

W, H = 1280, 640                      # pixels; figure at 100 dpi
BG = "#0b0f14"                        # near-black, like a dark viewer
TEXT, MUTED = "#f2f5f8", "#9aa7b4"
LINE = "#ffffff"
COLOURS = np.array([[0.0, 1.0, 1.0],  # nuclei: cyan (the demo's colours)
                    [1.0, 0.0, 1.0],  # membranes: magenta
                    [1.0, 1.0, 0.0]])  # marker: yellow
SELECTED, OTHER = 1, 4                # label values: embryo, heart-like tube


def fonts():
    """Segoe UI where it exists (Windows), else matplotlib's DejaVu Sans."""
    regular, bold = "DejaVu Sans", "DejaVu Sans"
    for name, path in (("regular", "C:/Windows/Fonts/segoeui.ttf"),
                       ("bold", "C:/Windows/Fonts/segoeuib.ttf")):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            if name == "regular":
                regular = family
            else:
                bold = family
    return regular, bold


def composite(planes, low, high, gamma=0.9):
    """Channels (c, y, x) -> RGB (y, x, 3): each scaled to its limits,
    coloured and added, like napari's additive blending."""
    rgb = np.zeros(planes.shape[1:] + (3,))
    for plane, colour, lo, hi in zip(planes, COLOURS, low, high):
        scaled = np.clip((plane.astype(float) - lo) / (hi - lo), 0, 1)
        rgb += scaled[..., None] ** gamma * colour
    return np.clip(rgb, 0, 1)


def limits(xy):
    """Display limits per channel from an XY projection (c, y, x). The
    marker (bright, thin) gets more headroom so it does not burn out to
    white where it overlaps the others."""
    low = [np.percentile(c, 0.5) for c in xy]
    high = [np.percentile(c, q) for c, q in zip(xy, (99.8, 99.8, 99.97))]
    high[2] *= 1.3
    return low, high


def bbox(labels, value, pad):
    """(z0, z1, y0, y1, x0, x1) of one label, padded, inside the image."""
    zz, yy, xx = np.nonzero(labels == value)
    lo = [max(0, int(a.min()) - pad) for a in (zz, yy, xx)]
    hi = [min(n, int(a.max()) + 1 + pad) for a, n in
          zip((zz, yy, xx), labels.shape)]
    return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


def silhouette(seen, origin, scale, pad=4, tolerance=3.0):
    """A polygon around an object's projection, as a person would click it.

    seen: the projection (2D, True where the object is), in level 0 voxels;
    the polygon goes round it `pad` voxels out, with a corner wherever the
    outline turns by more than `tolerance` voxels. Returns the corners in
    µm (voxel centre = origin + index * scale, per axis), as the tool keeps
    polygons."""
    grown = np.pad(ndi.binary_dilation(seen, iterations=pad), 1)
    outline = max(measure.find_contours(grown.astype(float), 0.5), key=len)
    corners = measure.approximate_polygon(outline - 1, tolerance)[:-1]
    return [(float(origin[0] + a * scale[0]), float(origin[1] + b * scale[1]))
            for a, b in corners]


def kept_range(outlines, axis, level, names):
    """The voxels (first, end) along one axis that the silhouettes can keep:
    the tightest extent of the polygons that use this axis (the tool fits
    a crop region to its outlines the same way)."""
    i = names.index(axis)
    low, high = -np.inf, np.inf
    for plane, keys in outlines.items():
        if axis in plane:
            for points in keys.values():
                values = np.asarray(points)[:, plane.index(axis)]
                low, high = max(low, values.min()), min(high, values.max())
    first = int(np.ceil((low - level.translation[i]) / level.scale[i]))
    end = int(np.floor((high - level.translation[i]) / level.scale[i])) + 1
    return first, end


def rotation(ax_deg, ay_deg):
    """Rotation about x, then about y (degrees), for (z, y, x) vectors."""
    a, b = np.radians(ax_deg), np.radians(ay_deg)
    rx = np.array([[np.cos(a), -np.sin(a), 0],    # turns z and y (about x)
                   [np.sin(a), np.cos(a), 0],
                   [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, -np.sin(b)],    # turns z and x (about y)
                   [0, 1, 0],
                   [np.sin(b), 0, np.cos(b)]])
    return ry @ rx


def turned_projection(volume, rot):
    """Maximum projection of an isotropic volume (c, z, y, x) seen along
    the rotated z axis; returns the image and a function that maps (z, y, x)
    points of the volume to (row, column) in that image."""
    shape = np.array(volume.shape[1:], float)
    size = int(np.ceil(np.linalg.norm(shape)))
    centre_in, centre_out = (shape - 1) / 2, np.full(3, (size - 1) / 2)
    inverse = rot.T                    # output -> input
    offset = centre_in - inverse @ centre_out
    image = np.stack([ndi.affine_transform(ch.astype(np.float32), inverse,
                                           offset=offset,
                                           output_shape=(size,) * 3,
                                           order=1).max(axis=0)
                      for ch in volume])

    def to_image(points):
        out = (rot @ (np.asarray(points, float) - centre_in).T).T + centre_out
        return out[:, 1], out[:, 2]    # rows (y'), columns (x')
    return image, to_image


def panel(fig, x, y, w, h):
    """An axes at pixel position (x, y from the bottom left), no frame."""
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_axis_off()
    return ax


def label(ax, text, regular):
    """The view's name in its bottom-left corner, on a dark patch."""
    ax.text(0.025, 0.03, text, transform=ax.transAxes, va="bottom",
            ha="left", color=TEXT, fontsize=9, family=regular, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black",
                      edgecolor="none", alpha=0.65))


def main():
    regular, bold = fonts()
    image = zarr.open_group(str(DEMO / "embryo_tissue_demo.zarr"), mode="r")
    # The views show level 1 (4 um voxels, half resolution), as when one
    # picks regions on a light level; the saved crop comes from level 0.
    data = np.asarray(image["1"][:])               # (c, z, y, x), 4/4/4 um
    labels = np.load(DEMO / "embryo_tissue_demo_truth.npz")["labels"]
    tool_image = vc.open_image(DEMO / "embryo_tissue_demo.zarr")
    names = [axis["name"] for axis in tool_image.axes]   # c, z, y, x
    full = tool_image.levels[0]

    # Silhouettes of the selected object in the three projected views, each
    # reaching through the whole depth that was projected (in µm).
    def axis_um(axis):
        i = names.index(axis)
        return full.translation[i], full.scale[i]

    def depth(axis):
        t, s = axis_um(axis)
        return (t - s / 2, t + (labels.shape[names.index(axis) - 1] - 0.5) * s)

    obj = labels == SELECTED
    outlines = {}
    for plane, seen in ((("y", "x"), obj.any(axis=0)),
                        (("z", "x"), obj.any(axis=1)),
                        (("y", "z"), obj.any(axis=2).T)):
        depth_axis = ({"z", "y", "x"} - set(plane)).pop()
        origin, scale = zip(*(axis_um(a) for a in plane))
        outlines[plane] = {depth(depth_axis): silhouette(seen, origin, scale)}

    # The three views as maximum projections, scaled alike.
    xy, xz, yz = data.max(axis=1), data.max(axis=2), data.max(axis=3)
    low, high = limits(xy)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    # pixels per micrometre: XY 1000 um -> 320 px; z 800 um -> 256 px
    s = 0.32
    xy_w = 1000 * s                    # 320
    z_w = 800 * s                      # 256
    gap, margin = 8, 28
    x0 = W - margin - xy_w - gap - z_w
    y_top = H - margin

    ax_xy = panel(fig, x0, y_top - xy_w, xy_w, xy_w)
    ax_yz = panel(fig, x0 + xy_w + gap, y_top - xy_w, z_w, xy_w)
    ax_xz = panel(fig, x0, y_top - xy_w - gap - z_w, xy_w, z_w)
    ax_3d = panel(fig, x0 + xy_w + gap, y_top - xy_w - gap - z_w, z_w, z_w)
    for ax in (ax_xy, ax_yz, ax_xz, ax_3d):
        ax.set_facecolor("black")

    nz, ny, nx = labels.shape          # level 0 voxels: the drawing's units
    ax_xy.imshow(composite(xy, low, high), extent=(0, nx, ny, 0),
                 interpolation="lanczos")
    ax_xz.imshow(composite(xz, low, high), extent=(0, nx, nz, 0),
                 aspect="auto", interpolation="lanczos")
    ax_yz.imshow(composite(yz, low, high).transpose(1, 0, 2),
                 extent=(0, nz, ny, 0), aspect="auto",
                 interpolation="lanczos")

    # Crop regions as solid lines in each view, the selected one wider; the
    # selected one is fitted to its silhouettes, as the tool fits it.
    z0, z1 = kept_range(outlines, "z", full, names)
    y0, y1 = kept_range(outlines, "y", full, names)
    xa, xb = kept_range(outlines, "x", full, names)
    regions = [((z0, z1, y0, y1, xa, xb), 2.0, 1.0),
               (bbox(labels, OTHER, pad=8), 1.2, 0.8)]
    for (rz0, rz1, ry0, ry1, rx0, rx1), width, alpha in regions:
        for ax, (a0, a1, b0, b1) in ((ax_xy, (rx0, rx1, ry0, ry1)),
                                     (ax_xz, (rx0, rx1, rz0, rz1)),
                                     (ax_yz, (rz0, rz1, ry0, ry1))):
            ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False,
                                   edgecolor=LINE, linewidth=width,
                                   alpha=alpha))
    # The silhouettes as dashed lines (µm -> voxel centres of the drawing).
    for ax, plane, (row_axis, col_axis) in ((ax_xy, ("y", "x"), ("y", "x")),
                                            (ax_xz, ("z", "x"), ("z", "x")),
                                            (ax_yz, ("y", "z"), ("y", "z"))):
        points = np.asarray(next(iter(outlines[plane].values())))
        rows = (points[:, plane.index(row_axis)] - axis_um(row_axis)[0]) \
            / axis_um(row_axis)[1] + 0.5
        cols = (points[:, plane.index(col_axis)] - axis_um(col_axis)[0]) \
            / axis_um(col_axis)[1] + 0.5
        ax.add_patch(Polygon(np.column_stack([cols, rows]), closed=True,
                             fill=False, edgecolor=LINE, linewidth=1.5,
                             linestyle=DASHES))
    ax_xy.set_xlim(0, nx), ax_xy.set_ylim(ny, 0)
    ax_xz.set_xlim(0, nx), ax_xz.set_ylim(nz, 0)
    ax_yz.set_xlim(0, nz), ax_yz.set_ylim(ny, 0)

    # The crop's 3D preview at full resolution, turned. The tool's own
    # outline_mask says which voxels the silhouettes keep; the others are
    # dimmed. Level 0 has 4 um in z and 2 um in y and x: z is interpolated
    # to 2 um, so the crop is isotropic and keeps its shape when turned.
    crop = np.asarray(image["0"][:, z0:z1, y0:y1, xa:xb]).astype(np.float32)
    keep = vc.outline_mask(full, names, [0, z0, y0, xa], crop.shape,
                           outlines, spatial=["z", "y", "x"])[0]
    print(f"crop {crop.shape[1:]} voxels; the polygons keep "
          f"{keep.mean():.0%} of them")
    shown = crop * np.where(keep, 1.0, DIM)
    shown = np.stack([ndi.zoom(ch, (2, 1, 1), order=1) for ch in shown])
    shape = ndi.zoom(keep.astype(np.float32), (2, 1, 1), order=1)
    rot = rotation(-28, 34)
    turned, to_image = turned_projection(shown, rot)
    turned_shape, _ = turned_projection(shape[None], rot)
    full_low, full_high = limits(np.asarray(image["0"][:]).max(axis=1))
    ax_3d.imshow(composite(turned, full_low, full_high),
                 interpolation="lanczos")
    for line in measure.find_contours(turned_shape[0], 0.5):
        if len(line) > 20:                 # the shape's outline, not specks
            ax_3d.plot(line[:, 1], line[:, 0], color=LINE, linewidth=1.3,
                       linestyle=DASHES)
    dz, dy, dx = (np.array(shown.shape[1:]) - 1)
    corners = np.array([[z, y, x] for z in (0, dz) for y in (0, dy)
                        for x in (0, dx)], float)
    rows, cols = to_image(corners)
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:      # a box edge
                ax_3d.plot([cols[i], cols[j]], [rows[i], rows[j]],
                           color=LINE, linewidth=0.8, alpha=0.6)
    # Square limits around the box, so the panel keeps its full size.
    half = max(np.ptp(rows), np.ptp(cols)) / 2 + 20
    mid_r, mid_c = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    ax_3d.set_xlim(mid_c - half, mid_c + half)
    ax_3d.set_ylim(mid_r + half, mid_r - half)

    for ax, text in ((ax_xy, "XY"), (ax_yz, "YZ"), (ax_xz, "XZ"),
                     (ax_3d, "3D")):
        label(ax, text, regular)

    # The words, on the left.
    left = 56 / W
    fig.text(left, 452 / H, "Volume Cropper", color=TEXT, fontsize=46,
             family=bold, va="baseline")
    tagline = ["Draw regions at low resolution.",
               "Save them at full resolution."]
    for k, line in enumerate(tagline):
        fig.text(left, (398 - 28 * k) / H, line, color=TEXT, fontsize=17,
                 family=regular, va="baseline")
    lines = ["OME-Zarr 0.4 · 0.5 · 0.6, also zipped",
             "TIFF · LIF · CZI · ND2 · OIR",
             "napari · four linked views · notebook · command line"]
    for k, line in enumerate(lines):
        fig.text(left, (308 - 30 * k) / H, line, color=MUTED, fontsize=13,
                 family=regular, va="baseline")
    fig.text(left, 56 / H, "Open source (BSD-3-Clause) · EMBL", color=MUTED,
             fontsize=11, family=regular, va="baseline")

    # The words must end left of the pictures (checked, not assumed).
    renderer = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer).x1 for t in fig.texts)
    if widest > x0 - 16:
        sys.exit(f"STOP: the text reaches x = {widest:.0f} px, "
                 f"the pictures start at {x0:.0f}")

    # (no "Software" text in the PNG: the repository's pictures carry no
    # metadata)
    fig.savefig(OUT, dpi=100, facecolor=BG, metadata={"Software": None})
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
