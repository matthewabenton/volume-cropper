"""Screenshots for the user guide (visible window, synthetic demo data only).

Writes JPEG/PNG files into docs/images of the copy in the VC_TOOL
environment variable (default: the tool folder this script is in): the "What to load" dialog,
the main window and an outlined object, on the embryo demo, and (0.17.0)
the Autocropper with a moving, outlined region on a synthetic time-lapse
(make_demo_timelapse.py; written to a temporary folder)
(make_demo_embryos.py; the trunk outlined from its true shape). A copy with
the magic wand (the archived branch archive/magic-wand-prototype) uses the
tails demo instead and also gets the wand pictures. The image path field is
replaced by the file name only, so no local paths appear.
Run with napari-env:  python guide_screenshots.py
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
# (0.17.0) a settings file of its own: the user's own settings
# (~/.volume_cropper/settings.json) are neither used nor changed, and the
# panel shows its default parts
import tempfile  # noqa: E402
os.environ.setdefault("VOLUME_CROPPER_SETTINGS", str(
    Path(tempfile.mkdtemp(prefix="vc_guide_settings_")) / "settings.json"))
EXPERIMENTAL = Path(__file__).resolve().parents[1]
TOOL = Path(os.environ.get("VC_TOOL", EXPERIMENTAL))
HAS_WAND = (TOOL / "wand.py").exists()
sys.path.insert(0, str(TOOL))
# make_demo_tails.py (the demo's geometry) is next to this script.
sys.path.append(str(Path(__file__).resolve().parent))
from qtpy.QtCore import Qt, QTimer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
import make_demo_tails as demo  # noqa: E402

DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
NAME = "touching_tails_demo" if HAS_WAND else "embryo_tissue_demo"
OUT = TOOL / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def pump(s=0.5):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


# (0.17.0) GUIDE_ONLY=<part of a name>: save only those pictures (the others
# are still made, so that the window gets to that point, but not saved)
ONLY = os.environ.get("GUIDE_ONLY", "")


def save(pixmap, name, width=1400):
    if ONLY and ONLY not in name:
        print("skipped", name, flush=True)
        return
    if pixmap.width() > width:
        pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
    fmt = "JPG" if name.endswith(".jpg") else "PNG"
    # Qt's quality: JPEG 0-100; for PNG -1 = normal compression (90 would
    # mean hardly any).
    pixmap.save(str(OUT / name), fmt, 90 if fmt == "JPG" else -1)
    print("saved", name, pixmap.width(), "x", pixmap.height(), flush=True)


def view_shot(w, view, name):
    save(w.ortho.views[view].frame.grab(), name, width=900)


def midpoint(k):
    curve, _ = demo.bezier(demo.TAILS[k]["points"])
    return dict(zip("zyx", curve[len(curve) // 2]))


viewer, w = crop_gui.open_window(show=True)
win = viewer.window._qt_window
win.resize(1700, 1050)
pump(1)


def shoot_dialog():
    """The "What to load" dialog as it first appears; then Load."""
    dialog = QApplication.activeModalWidget()
    if dialog is None or type(dialog).__name__ != "LoadDialog":
        QTimer.singleShot(100, shoot_dialog)
        return
    pump(0.5)
    save(dialog.grab(), "load_dialog.png", width=900)
    dialog.accept()


if (TOOL / "load_dialog.py").exists():
    QTimer.singleShot(200, shoot_dialog)
    w.open_image(DEMO / f"{NAME}.zarr", ask=True)
else:
    w.open_image(DEMO / f"{NAME}.zarr")
pump(4)
w.show_3d_check.setChecked(True)      # off at first since 0.5.0
pump(3)
w.path_edit.setText(f"{NAME}.zarr")       # no local path
w.napari_panels_check.setChecked(False)
pump(1)

# 1. The main window with one crop region: the embryo demo's trunk, outlined
# from its true shape on 3 XY, 2 XZ and 2 YZ slices (as a person would).
if HAS_WAND:
    mid = midpoint(1)
    w._new_box({"z": (mid["z"] - 250, mid["z"] + 250),
                "y": (mid["y"] - 300, mid["y"] + 300),
                "x": (mid["x"] - 250, mid["x"] + 250)})
else:
    import numpy as np
    import score_demo_embryos as score

    truth = np.load(DEMO / f"{NAME}_truth.npz")
    labels, voxel = truth["labels"], truth["voxel_um"].astype(float)
    outlines = score.drawn_outlines(labels == 1, voxel)
    box = crop_gui.vc.CropBox("trunk", outlines=outlines)
    w.boxes = [box]
    crop_gui._fit_to_outlines(box, w.image)
    keys = sorted(outlines[("y", "x")])
    between = (keys[0] + keys[1]) / 2        # an XY slice between two drawn
    w.ortho.go_to_slice(("y", "x"), between)
w._refresh(select=0)
pump(3)
w.path_edit.setText(f"{NAME}.zarr")
# The default cache size is a quarter of this computer's memory: show a
# neutral 4 GB instead, and put the user's (saved) setting back afterwards.
dask_settings = crop_gui._dask_settings()
saved_cache = (dask_settings.cache, dask_settings.enabled)
try:
    w.cache_spin.setValue(4.0)
    w._show_cache()
    pump(1)
    save(win.grab(), "main_window.jpg")
finally:
    dask_settings.cache, dask_settings.enabled = saved_cache

# 2. Outlines in 3D: the trunk on that XY slice, shaded.
if not HAS_WAND:
    w.shade_check.setChecked(True)
    pump(2)
    view_shot(w, "XY", "outlines_xy.jpg")
    w.shade_check.setChecked(False)

# 4. (0.18.0) The Channels and Levels parts; the trunk region in 3D with
# its dots and its shape as a surface; the Settings dialog.
if not HAS_WAND and hasattr(w, "levels"):
    from qtpy.QtGui import QPainter, QPixmap
    from qtpy.QtWidgets import QLabel

    from volume_cropper import ortho_views as ov
    for key in ("channels", "levels"):
        w.sections[key].set_open(True)
    w.levels.source_combo.setCurrentIndex(
        w.levels.source_combo.findData("whole"))
    if hasattr(w.levels, "zoom_check"):       # (0.22.0) shown ticked
        w.levels.zoom_check.setChecked(True)
    pump(3)
    parts = [w.sections[key].grab() for key in ("channels", "levels")]
    both = QPixmap(max(p.width() for p in parts),
                   sum(p.height() for p in parts))
    both.fill(w.palette().window().color())
    painter = QPainter(both)
    painter.drawPixmap(0, 0, parts[0])
    painter.drawPixmap(0, parts[0].height(), parts[1])
    painter.end()
    save(both, "levels.png", width=700)
    view3d = w.ortho.views["3D"]
    ov.camera(view3d.model).angles = (0.0, 30.0, 55.0)
    pump(3)
    view_shot(w, "3D", "region_3d.png")
    dialog = crop_gui.SettingsDialog(w)
    for label in dialog.findChildren(QLabel):      # no local path
        if label.text().startswith("Kept in"):
            label.setText("Kept in <your home folder>/.volume_cropper/"
                          "settings.json (yours only).")
    # a neutral 4 GB, as in main_window.jpg (napari's default is a quarter
    # of this computer's memory); applied only on OK, and the dialog is
    # cancelled below, so the user's setting stays
    dialog.cache_spin.setValue(4.0)
    dialog.show()
    pump(1)
    save(dialog.grab(), "settings.png", width=1100)
    dialog.reject()

# 3. (0.17.0) The Autocropper on a synthetic time-lapse
# (make_demo_embryo_timelapse: a multicolour embryo trunk that drifts and
# turns, with a gastruloid beside it; VC_LAPSE_DIR may hold it already):
# Moving region + Outline, at the last timepoint, with the Autocropper
# part open.
if not HAS_WAND and hasattr(w, "object_group"):
    from qtpy.QtWidgets import QScrollArea

    import make_demo_embryo_timelapse as lapse
    from volume_cropper.load_dialog import LoadChoice
    lapse_dir = Path(os.environ.get("VC_LAPSE_DIR") or tempfile.mkdtemp(
        prefix="vc_guide_lapse_"))
    lapse_path = lapse_dir / f"{lapse.NAME}.zarr"
    if not lapse_path.exists():
        lapse.make(lapse_dir)
    w.boxes.clear()                   # (else: "discard the regions?")
    w._refresh()
    w.open_image(lapse_path, choice=LoadChoice())
    pump(4)
    w.path_edit.setText(lapse_path.name)              # no local path
    w.sections["view"].set_open(False)
    w.sections["find"].set_open(True)
    group = w.object_group
    group.setChecked(True)
    group.margin_spin.setValue(10.0)
    group.outline_check.setChecked(True)
    group.fit()
    pump(1)
    w._refresh(select=len(w.boxes) - 1)
    last = lapse.T - 1
    import numpy as np
    labels = np.load(lapse_dir / f"{lapse.NAME}_truth.npz")["labels"][last]
    z_mid = float(np.argwhere(labels == 1)[:, 0].mean() * lapse.VOXEL[0])
    w.ortho._set_cursor({"t": last * lapse.INTERVAL, "z": z_mid})
    pump(3)
    area = w.parent()
    while area is not None and not isinstance(area, QScrollArea):
        area = area.parent()
    if area is not None:
        area.ensureWidgetVisible(group, 0, 0)
    pump(1)
    w.path_edit.setText(lapse_path.name)
    save(win.grab(), "autocropper_moving.jpg")
    w.boxes.clear()
    w._refresh()

# 5. (0.18.0) Two touching objects split between their seeds, with a gap
# line, outlines per slice and their shapes in 3D: two synthetic
# ellipsoids whose contact is tilted 30 deg (as outlines_3d_0_18_gui_test).
if not HAS_WAND and hasattr(w, "object_group") \
        and hasattr(w.object_group, "gap_button"):
    import numpy as np
    from napari.layers import Image
    from qtpy.QtWidgets import QScrollArea

    from volume_cropper import ortho_views as ov
    shape, scale = (90, 140, 200), 2.0
    middle = np.array([45.0, 70.0, 100.0])
    angle = np.radians(30.0)
    direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
    centres = [middle - 28.0 * direction, middle + 28.0 * direction]
    grid = np.indices(shape, dtype=np.float32)

    def inside(centre):
        d = np.stack([grid[i] - centre[i] for i in range(3)])
        along = sum(d[i] * direction[i] for i in range(3))
        across = d - along * direction[:, None, None, None]
        other = np.sqrt(np.maximum((across ** 2).sum(axis=0) - d[1] ** 2,
                                   0))
        return ((along / 34.0) ** 2 + (d[1] / 26.0) ** 2
                + (other / 20.0) ** 2) <= 1.0

    union = inside(centres[0]) | inside(centres[1])
    data = np.random.default_rng(0).poisson(
        100.0 + 1400.0 * union).astype(np.uint16)
    saved_exec = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1      # (no dialog)
    try:
        w.open_layers([Image(data, name="touching objects",
                             scale=(scale,) * 3, units=("um",) * 3)])
    finally:
        crop_gui.FileLoadDialog.exec = saved_exec
    pump(4)
    for key in ("view", "channels", "levels"):
        w.sections[key].set_open(False)
    w.sections["find"].set_open(True)
    group = w.object_group
    group.setChecked(True)
    group.grow_check.setChecked(False)
    group.margin_spin.setValue(4.0)
    group.outline_check.setChecked(True)
    group.outline_kind_combo.setCurrentIndex(
        group.outline_kind_combo.findData("slices"))
    z_mid = float(middle[0] * scale)
    w.ortho._set_cursor({"z": z_mid})
    for dz in (-6.0, 0.0, 6.0):                 # gap lines on 3 slices
        x = middle[2] - dz * np.tan(angle)      # the contact (voxels)
        group._gap_drawn({
            "view": "XY", "plane": ("y", "x"), "depth": "z",
            "at": (middle[0] + dz) * scale, "time": None,
            "points": [((y * scale), (x + 1.2 * np.sin(y / 9)) * scale)
                       for y in np.linspace(35, 105, 15)],
            "width": 2 * scale})
    group.seeds = [dict(zip("zyx", c * scale)) for c in centres]
    group.seed_results = [None, None]
    w.ortho.set_seeds(group.seeds)
    group.fit_to_seeds()
    pump(1)
    if not w.show_3d_check.isChecked():
        w.show_3d_check.setChecked(True)
    pump(3)
    ov.camera(w.ortho.views["3D"].model).angles = (0.0, 35.0, 60.0)
    w._refresh(select=0)
    pump(2)
    area = w.parent()
    while area is not None and not isinstance(area, QScrollArea):
        area = area.parent()
    if area is not None:
        area.ensureWidgetVisible(group.info, 0, 0)
    pump(1)
    save(win.grab(), "autocropper_split.jpg")
if not HAS_WAND:
    viewer.close()
    print("done (no magic wand in", TOOL, ")", flush=True)
    sys.exit(0)

# 2. Magic wand: objects found.
w._remove_box()
pump(0.5)
w._find_objects()
pump(0.3)
end = time.time() + 120
while w._wand_worker is not None and time.time() < end:
    pump(0.2)
pump(2)
save(win.grab(), "wand_objects.jpg")

# 3. Ruler across tail 2 in XY (away from the crosshair).
w.ruler_button.setChecked(True)
xy = w.ortho.views["XY"]
y = mid["y"] + 120
xy.ruler.add([[[y, mid["x"] - 230.0], [y, mid["x"] + 230.0]]], shape_type="line")
pump(1.5)
view_shot(w, "XY", "ruler_xy.png")
w.ruler_button.setChecked(False)
w._clear_rulers()
pump(0.5)

# 4. Boxes from all objects (editable outlines).
w._boxes_from_all()
pump(2)
w._refresh(select=1)
pump(1.5)
view_shot(w, "XZ", "outlines_xz.png")

# 5. The selected box in 3D as it will be saved (clipped, outside hidden).
w.wand_3d_check.setChecked(False)
w.clip_check.setChecked(True)
mode = w.mask_combo.findData("hide")
w.mask_combo.setCurrentIndex(mode)
w.mask_combo.activated.emit(mode)
pump(3)
view_shot(w, "3D", "box_3d_hidden.png")
w.clip_check.setChecked(False)
pump(1)
viewer.close()
print("done", flush=True)
