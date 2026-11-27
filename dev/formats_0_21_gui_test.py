"""0.21.0 formats in a VISIBLE window: a zipped OME-Zarr (.ozx), an OME-Zarr
0.6 image and a zipped collection of two images, on synthetic data only
(the demo embryo image packed into an .ozx, and test images).

Usage: python formats_0_21_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)
Tests the folder in VC_TOOL (default: the tool folder this script is in). Checks:
- the .ozx opens; the summary and the plan say zipped / .ozx;
- a crop of it: the progress shows the packing step, it is checked OK,
  and the .ozx crop opens with "Open last crop" (a new window);
- cropping it again with Overwrite while that window holds it open gives
  the clear "could not be replaced: it is open" message, and the old file
  stays; after closing that window it works;
- opening another image closes the zip file (Windows could not replace it
  otherwise);
- an OME-Zarr 0.6 image opens (summary "OME-Zarr 0.6") and crops, checked;
- a zipped collection of two images asks which (the chooser is answered
  by the script: image 1); the title names it; the crop is
  <collection>_1_<region>.ozx, checked.
Saves zipped_window.png (the demo embryo image). The script answers every
dialog itself.
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
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from test_formats import _bioformats2raw  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data" / "embryo_tissue_demo.zarr"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "crops", ignore_errors=True)
results, messages, progress = [], [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def wait_for_crop(w, seconds=120):
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
load_dialog.LoadDialog.exec = lambda self: 1     # "What to load": accepted

# --- Synthetic images ------------------------------------------------------------------
ozx = OUT / "embryo_tissue_demo.ozx"
if not ozx.exists():
    vc.pack_zip(DEMO, ozx)
v06 = OUT / "test06.zarr"
if not v06.exists():
    make_test_image(v06, version="0.6", shape0=(2, 40, 160, 140),
                    factors=[(1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 4, 4)])
plate = OUT / "plate.ozx"
if not plate.exists():
    shutil.rmtree(OUT / "plate.zarr", ignore_errors=True)
    vc.pack_zip(_bioformats2raw(OUT / "plate.zarr", "0.5", 2), plate)

viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
w.progress_signal.connect(lambda done, total, text: progress.append(text))
w._may_discard_boxes = lambda action: True
pump(2)

# --- 1. The .ozx ------------------------------------------------------------------------
w.open_image(ozx)
pump(4)
first = w.image
check("ozx opens: summary says zipped (.ozx)",
      first.container == "ozx" and "(zipped, .ozx)" in w.image_info.text(),
      w.image_info.text().split("\n")[0][:120])
z0, z1 = first.extent("z")
box = vc.CropBox("middle", {"z": (z0 + 0.3 * (z1 - z0), z0 + 0.6 * (z1 - z0)),
                            "y": (60.0, 160.0), "x": (60.0, 160.0)})
w.boxes = [box]
w._refresh(select=0)
w.out_edit.setText(str(OUT / "crops"))
pump(1)
check("output field says the crops are .ozx files",
      ".ozx files" in w.out_edit.placeholderText(),
      w.out_edit.placeholderText())
viewer.screenshot(str(OUT / "zipped_window.png"), canvas_only=False)
w._crop(False)
wait_for_crop(w)
crop = OUT / "crops" / "embryo_tissue_demo_middle.ozx"
check("ozx crop: packing step shown, checked OK, file there",
      crop.is_file() and "checked OK" in w.status.text()
      and any("packing it into one zip file" in t for t in progress),
      w.status.text().replace("\n", " | ")[:200])

# --- 2. Open last crop; Overwrite while it is open there ----------------------------------
before = set(QApplication.topLevelWidgets())
w._last_output = crop
w._open_last_crop()
pump(5)
windows = [x for x in QApplication.topLevelWidgets() if x not in before
           and x.isVisible()]
check("Open last crop: the .ozx opens in a new window", len(windows) >= 1)
stamp = crop.stat().st_mtime
w.overwrite_check.setChecked(True)
messages.clear()
w._crop(False)
wait_for_crop(w)
text = w.status.text() + " " + " ".join(str(m) for m in messages)
check("Overwrite while open elsewhere: clear message, old file kept",
      "it is open" in text and crop.stat().st_mtime == stamp
      and not crop.with_name(crop.name + ".part").exists(),
      text.replace("\n", " | ")[:240])
for window in windows:
    window.close()
pump(4)
w._crop(False)
wait_for_crop(w)
check("after closing that window, Overwrite works",
      "checked OK" in w.status.text() and crop.stat().st_mtime != stamp,
      w.status.text().replace("\n", " | ")[:160])
w.overwrite_check.setChecked(False)

# --- 3. OME-Zarr 0.6 ---------------------------------------------------------------------
w.open_image(v06)
pump(4)
check("opening another image closes the zip file", first.store is None)
check("0.6 opens: summary says OME-Zarr 0.6",
      "OME-Zarr 0.6" in w.image_info.text(),
      w.image_info.text().split("\n")[0][:120])
w.boxes = [vc.CropBox("cell", {"z": (20.0, 50.0), "y": (8.0, 24.0),
                               "x": (6.0, 20.0)})]
w._refresh(select=0)
w._crop(False)
wait_for_crop(w)
out06 = OUT / "crops" / "test06_cell.zarr"
check("0.6 crop checked OK and stays 0.6",
      "checked OK" in w.status.text()
      and vc.open_image(out06).ome_version == "0.6",
      w.status.text().replace("\n", " | ")[:160])

# --- 4. A zipped collection of two images ---------------------------------------------
asked = []


def choose(many):
    asked.append(many)
    return many.images[1][0]


w._choose_inner_image = choose
w.open_image(plate)
pump(4)
check("collection: asked which of 2 images; title names image 1",
      len(asked) == 1 and len(asked[0].images) == 2
      and "image 1" in viewer.title, viewer.title)
w.boxes = [vc.CropBox("b", {"z": w.image.extent("z"), "y": (2.0, 6.0),
                            "x": (1.0, 5.0)})]
w._refresh(select=0)
w._crop(False)
wait_for_crop(w)
check("collection crop: plate_1_b.ozx, checked OK",
      (OUT / "crops" / "plate_1_b.ozx").is_file()
      and "checked OK" in w.status.text(),
      w.status.text().replace("\n", " | ")[:160])

print(f"\n{sum(results)} of {len(results)} checks passed.")
if messages:
    print("messages:", messages)
pump(1)
QApplication.instance().closeAllWindows()
pump(1)
