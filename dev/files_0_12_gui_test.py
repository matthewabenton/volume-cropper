"""0.12.0 TIFF files read into memory, in a VISIBLE window, on synthetic
TIFFs written into OUT_DIR (a Fiji hyperstack, a 2-image OME-TIFF with a 2D
movie, a plain TIFF stack without metadata).

Usage: python files_0_12_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)
Tests the folder in VC_TOOL (default: the tool folder this script is in). Checks: the file's
"What to load" dialog (images in the file, an unnamed axis chosen as t,
the calibration fields, the memory line, a 2D image without z and 3D);
reading a part into memory (the summary says what was read); the output
section for a file (TIFF level hidden, crops are .tif / .ome.tif, zstd);
a crop, checked, compressed; "What to load…" reads the file again and
keeps the regions; a 2D movie shows only the XY view (no z row, no 3D),
and a 3D image all four views again; "Open last crop" opens the TIFF.
Saves files_window.png and flat_window.png (synthetic data only).
The script answers every dialog itself.
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
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import file_images as fi  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import core as vc  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "crops", ignore_errors=True)
results, messages = [], []
rng = np.random.default_rng(1)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def wait_for_crop(w, seconds=60):
    end = time.time() + seconds
    pump(0.5)
    while time.time() < end and not w.crop_selected_button.isEnabled():
        pump(0.2)
    pump(0.5)


crop_gui.QMessageBox.warning = staticmethod(
    lambda *a, **k: messages.append(a[1:3]))
crop_gui.QMessageBox.critical = staticmethod(
    lambda *a, **k: messages.append(a[1:3]))
crop_gui._confirm = lambda *a: True
# "What to load" for files: accepted as it starts (the dialog's own checks
# run on it below, without exec)
load_dialog.FileLoadDialog.exec = lambda self: 1

# --- Synthetic files ------------------------------------------------------------
stack = OUT / "embryo.tif"                                 # Fiji hyperstack
blobs = rng.integers(50, 4000, size=(2, 30, 2, 160, 200), dtype=np.uint16)
tifffile.imwrite(stack, blobs, imagej=True, resolution=(2.0, 2.0),
                 metadata={"axes": "TZCYX", "spacing": 1.5, "unit": "um",
                           "finterval": 60.0})
ome = OUT / "two.ome.tif"                                  # 2 images
movie = rng.integers(0, 255, size=(12, 90, 110), dtype=np.uint8)
with tifffile.TiffWriter(ome, ome=True) as tif:
    tif.write(rng.integers(0, 255, size=(2, 8, 64, 64), dtype=np.uint8),
              metadata={"axes": "CZYX", "Name": "stack",
                        "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0,
                        "PhysicalSizeZ": 2.0,
                        "Channel": {"Name": ["dapi", "gfp"]}})
    tif.write(movie, metadata={"axes": "TYX", "Name": "crawling",
                               "PhysicalSizeX": 46.4, "PhysicalSizeY": 46.4,
                               "TimeIncrement": 0.1})
plain = OUT / "plain.tif"                                  # no metadata
tifffile.imwrite(plain, rng.integers(0, 255, size=(6, 40, 50),
                                     dtype=np.uint8),
                 photometric="minisblack", metadata=None)

viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)

# --- The file's "What to load" dialog ---------------------------------------------
reader = fi.open_file(ome)
dialog = load_dialog.FileLoadDialog(reader, fi.FileChoice(), parent=w)
check("dialog: both images of the OME-TIFF listed",
      dialog.scene_combo.count() == 2, dialog.scene_combo.itemText(1))
dialog.scene_combo.setCurrentIndex(1)                      # the 2D movie
pump(0.3)
choice = dialog.choice()
check("dialog: a 2D movie has no z voxel and no 3D view; t range 1-12",
      not dialog.voxel_spins["z"].isEnabled()
      and not dialog.show_3d_check.isEnabled()
      and choice.scene == 1 and "T" in dialog.range_spins
      and "Z" not in dialog.range_spins
      and dialog.range_spins["T"][1].value() == 12)
check("dialog: the memory line", "Needs about" in dialog.memory.text(),
      dialog.memory.text()[:80])
dialog.close()
reader.close()
reader = fi.open_file(plain)
dialog = load_dialog.FileLoadDialog(reader, fi.FileChoice(), parent=w)
combo = dialog.axis_combos[0]
combo.setCurrentIndex(combo.findData("T"))                 # it is a movie
pump(0.3)
check("dialog: an unnamed axis chosen as t gives a t range, no z",
      "T" in dialog.range_spins and "Z" not in dialog.range_spins
      and dialog.choice().axes == "TYX"
      and "Not in the file" in dialog.calibration_note.text())
dialog.close()
reader.close()

# --- A part of a hyperstack in memory ---------------------------------------------
w.open_image(stack, file_choice=fi.FileChoice(ranges={"z": (5, 25)}))
pump(3)
image = w.image
check("opened in memory: slices 6-25 of 30, at their place",
      image is not None and image.in_memory
      and image.levels[0].array.shape == (2, 2, 20, 160, 200)
      and np.isclose(image.levels[0].translation[2], 7.5)
      and "slices 6–25 of 30" in w.image_info.text(),
      w.image_info.text().replace("\n", " | ")[:160])
check("four views (a 3D image), TIFF level hidden, crops are .tif",
      not w.ortho.flat and all(v.frame.isVisible()
                               for v in w.ortho.views.values())
      and not w.tiff_check.isVisible()
      and ".tif" in w.out_edit.placeholderText())
box = vc.CropBox("cell", {"t": (0.0, 60.0), "z": (12.0, 30.0),
                          "y": (20.0, 50.0), "x": (30.0, 70.0)},
                 channels=[1])
w.boxes = [box]
w._refresh(select=0)
w.out_edit.setText(str(OUT / "crops"))
w.compress_check.setChecked(True)
pump(1)
check("plan text: one .tif file, zstd",
      "a .tif file, zstd" in w.plan_label.text(),
      w.plan_label.text().replace("\n", " | "))
viewer.screenshot(str(OUT / "files_window.png"), canvas_only=False)
w._crop(False)
wait_for_crop(w)
out = OUT / "crops" / "embryo_cell.tif"
ok = out.exists() and "checked OK" in w.status.text()
if out.exists():
    with tifffile.TiffFile(out) as tif:
        ok = ok and tif.pages.first.compression.name == "ZSTD"
        meta = tif.imagej_metadata or {}
        ok = ok and meta.get("channels", 1) == 1 and meta.get("spacing") == 1.5
check("crop: a zstd Fiji TIFF, one channel, checked OK", ok,
      w.status.text().replace("\n", " | ")[:200])

# --- "What to load..." reads again and keeps the regions ------------------------------
w._file_choice = fi.FileChoice(ranges={"z": (0, 30)})
w._change_what_to_load()
pump(3)
check("What to load again: all 30 slices read, the region kept",
      w.image.levels[0].array.shape[2] == 30 and len(w.boxes) == 1
      and w.boxes[0].name == "cell")

# --- A 2D movie: only XY ----------------------------------------------------------------
w._may_discard_boxes = lambda action: True
w.open_image(ome, file_choice=fi.FileChoice(scene=1))
pump(3)
o = w.ortho
check("2D movie: only the XY view, no Focus buttons, no 3D",
      o.flat and o.views["XY"].frame.isVisible()
      and not o.views["XZ"].frame.isVisible()
      and not o.views["3D"].frame.isVisible()
      and not any(b.isVisible() for b in o._focus_buttons.values())
      and not w.show_3d_check.isEnabled()
      and not w.projection_combo.isEnabled())
check("2D movie: no z row in the region fields; crops are .ome.tif",
      "z" not in w.range_spins and "t" in w.range_spins
      and ".ome.tif" in w.out_edit.placeholderText())
# (the window gives every region all axes, the single z plane too)
w.boxes = [vc.CropBox("cell", {"t": (0.0, 1.05), "z": w.image.extent("z"),
                               "y": (900.0, 2500.0), "x": (900.0, 3000.0)},
                      t_step=2)]
w._refresh(select=0)
pump(1)
viewer.screenshot(str(OUT / "flat_window.png"), canvas_only=False)
w._crop(False)
wait_for_crop(w)
out2 = OUT / "crops" / "two_crawling_cell.ome.tif"
ok = out2.exists() and "checked OK" in w.status.text()
if out2.exists():
    with tifffile.TiffFile(out2) as tif:
        ok = ok and tif.is_ome and tif.series[0].shape[0] == 6
check("2D movie crop: OME-TIFF, every 2nd frame (6 of 12), checked OK", ok,
      w.status.text().replace("\n", " | ")[:200])

# --- Back to a 3D image: four views ------------------------------------------------------
w.open_image(stack, file_choice=fi.FileChoice())
pump(3)
check("a 3D image again: all four views, Focus buttons back",
      not w.ortho.flat and w.ortho.views["XZ"].frame.isVisible()
      and all(b.isVisible() for b in w.ortho._focus_buttons.values())
      and w.show_3d_check.isEnabled())

# --- Open last crop: the TIFF in a new window ----------------------------------------------
w._last_output = out
before = set(QApplication.topLevelWidgets())
w._open_last_crop()
pump(4)
windows = [x for x in QApplication.topLevelWidgets() if x not in before
           and x.isVisible()]
check("Open last crop: the TIFF opens in a new window", len(windows) >= 1)

print(f"\n{sum(results)} of {len(results)} checks passed.")
if messages:
    print("messages:", messages)
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
