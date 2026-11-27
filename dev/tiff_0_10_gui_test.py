"""0.10.0 "Also save as TIFF", in a VISIBLE window on the synthetic embryo
demo (trunk outlined from its true shape).

Usage: python tiff_0_10_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)
Tests the folder in VC_TOOL (default: the tool folder this script is in). Crops into
OUT_DIR/crops. Checks: the level menu lists the image's levels; the plan
text shows the TIFF's level and size; Crop selected writes the crop and
its TIFF (same voxels as the crop's level, ImageJ calibration), the
progress reaches the end and the report names the TIFF; a second crop
without Overwrite is refused (the TIFF is listed); a level the crop lacks
gives its coarsest (plan text and file name), and Overwrite deletes the
replaced crop's old TIFF; a TIFF over the size limit
is asked about first (answered No here: nothing written). Message boxes
and questions are answered by the script. Saves tiff_window.png.
"""
import faulthandler
import os
import shutil
import sys
import time
from pathlib import Path

faulthandler.enable()
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
sys.path.append(str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
import score_demo_embryos as score  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CROPS = OUT / "crops"
shutil.rmtree(CROPS, ignore_errors=True)
DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
results = []
messages = []          # (kind, title, text) of the message boxes
answers = []           # questions asked (answered No)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def record(kind):
    def show(parent, title, text, *args, **kwargs):
        messages.append((kind, title, text))
    return show


crop_gui.QMessageBox.warning = staticmethod(record("warning"))
crop_gui.QMessageBox.critical = staticmethod(record("critical"))
crop_gui._confirm = lambda parent, title, text: answers.append(title) or False


def crop_and_wait(timeout=300):
    """Crop the selected region; pump until done; return the seconds."""
    start = time.time()
    w._crop(False)
    pump(0.5)
    while time.time() - start < timeout:
        pump(0.2)
        if w.crop_selected_button.isEnabled():
            return time.time() - start
    return None


viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(DEMO / "embryo_tissue_demo.zarr", choice=LoadChoice())
pump(5)
truth = np.load(DEMO / "embryo_tissue_demo_truth.npz")
outlines = score.drawn_outlines(truth["labels"] == 1,
                                truth["voxel_um"].astype(float))
box = vc.CropBox("trunk", outlines=outlines)
w.boxes = [box]
crop_gui._fit_to_outlines(box, w.image)
w._refresh(select=0)
pump(2)
image = w.image
plan = vc.plan_crop(image, box, min_size=w.min_size_spin.value())
print(f"crop levels {plan.level_indices}, image levels {len(image.levels)}")

# --- Controls and plan text ------------------------------------------------------
combo = w.tiff_level_combo
check("level menu: one entry per image level, off until ticked",
      combo.count() == len(image.levels) and not combo.isEnabled()
      and "TIFF" not in w.plan_label.text(),
      str([combo.itemText(i) for i in range(combo.count())]))
w.tiff_check.setChecked(True)
combo.setCurrentIndex(1)
pump(1)
size = vc.tiff_bytes(plan, image.levels[0].array.dtype, 1)
check("plan text shows the TIFF's level and size",
      combo.isEnabled() and "TIFF: level 1" in w.plan_label.text()
      and vc.format_bytes(size) in w.plan_label.text(),
      repr(w.plan_label.text().splitlines()[-1]))

# --- Crop with TIFF ------------------------------------------------------------
w.out_edit.setText(str(CROPS))
seconds = crop_and_wait()
crop_path = CROPS / "embryo_tissue_demo_trunk.zarr"
tiff_path = CROPS / "embryo_tissue_demo_trunk_level1.tif"
check("crop and TIFF written", seconds is not None and crop_path.exists()
      and tiff_path.exists(), f"({seconds:.1f} s)" if seconds else "")
status = w.status.text()
print("status:", status.replace("\n", " | "))
check("report names the TIFF and its check",
      "embryo_tissue_demo_trunk_level1.tif: level 1" in status
      and "Each TIFF was checked against its crop" in status
      and "CHECK FAILED" not in status and not messages, str(messages))
check("progress bar at the end", w.progress_bar.value() == 1000,
      str(w.progress_bar.value()))
crop = vc.open_image(crop_path)
with tifffile.TiffFile(tiff_path) as tif:
    meta = tif.imagej_metadata
    data = tif.series[0].asarray()
    x_res = tif.pages[0].tags["XResolution"].value
level = crop.levels[1]
expected = level.array[...]                        # (c, z, y, x)
check("TIFF voxels = the crop's level 1 (z before c)",
      np.array_equal(data, expected.transpose(1, 0, 2, 3)),
      f"(TIFF {data.shape}, crop {expected.shape})")
check("ImageJ calibration and channels",
      abs(meta["spacing"] - level.scale[1]) < 1e-9
      and abs(x_res[1] / x_res[0] - level.scale[3]) < 1e-6
      and meta["unit"] == "um" and meta["channels"] == expected.shape[0]
      and meta["mode"] == "composite" and len(meta.get("LUTs", [])) ==
      expected.shape[0], f"(labels {meta.get('Labels', [])[:3]}, "
      f"ranges {meta.get('Ranges')})")
viewer.window._qt_window.grab().save(str(OUT / "tiff_window.png"))

# --- Existing outputs, without Overwrite ----------------------------------------
messages.clear()
w._crop(False)
pump(1)
check("second crop without Overwrite: refused, TIFF listed",
      messages and messages[-1][1] == "Output exists"
      and tiff_path.name in messages[-1][2], str(messages[-1:]))

# --- A level the crop lacks -----------------------------------------------------
last = len(image.levels) - 1
if last > plan.level_indices[-1]:
    combo.setCurrentIndex(last)
    pump(1)
    have = plan.level_indices[-1]
    check("a level the crop lacks: its coarsest, said in the plan text",
          f"TIFF: level {have}" in w.plan_label.text()
          and f"this crop has no level {last}" in w.plan_label.text(),
          repr(w.plan_label.text().splitlines()[-1]))
    w.overwrite_check.setChecked(True)
    messages.clear()
    seconds = crop_and_wait()
    clamped = CROPS / f"embryo_tissue_demo_trunk_level{have}.tif"
    check("... and the file is named after that level",
          clamped.exists() and f"this crop has no level {last}" in
          w.status.text() and not messages, str(messages))
    # Overwrite deleted the replaced crop's TIFF of another level (level 1)
    check("overwrite: the old TIFF of the replaced crop is deleted, and said",
          not tiff_path.exists() and "deleted the old TIFF(s)" in
          w.status.text() and tiff_path.name in w.status.text(),
          w.status.text().replace("\n", " | ")[:300])
    tiff_path = clamped
else:
    print("(the crop keeps every level of the demo: clamping not shown)")

# --- Large TIFF: asked first ----------------------------------------------------
combo.setCurrentIndex(0)
w.overwrite_check.setChecked(True)
original = vc.TIFF_LIMIT
vc.TIFF_LIMIT = 1000                   # every TIFF counts as large now
answers.clear()
before = tiff_path.stat().st_mtime
level0 = CROPS / "embryo_tissue_demo_trunk_level0.tif"
w._crop(False)
pump(2)
vc.TIFF_LIMIT = original
check("TIFF over the limit: asked first; No writes nothing",
      answers == ["Large TIFF"] and not level0.exists()
      and tiff_path.stat().st_mtime == before
      and w.crop_selected_button.isEnabled(), str(answers))
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
viewer.close()
pump(0.5)
