"""0.11.0 cropping in time, in a VISIBLE window, on a synthetic time-lapse
(9 timepoints 30 min apart, 2 channels; written into OUT_DIR).

Usage: python time_0_11_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)
Tests the folder in VC_TOOL (default: the tool folder this script is in). Checks: the t fields
show timepoint numbers 1-9 with their times; "every n-th" sets the
region's step and the plan text says what is kept; editing the numbers
sets the region's times (frame centres); "here" takes the timepoint shown;
the region lines (2D and the 3D frame) fade at timepoints the crop does
not keep and the crop preview is all black there; the What-to-load dialog
counts timepoints from 1; the crop keeps exactly the chosen timepoints
with n times the frame interval and passes its check. Saves
time_window.png.
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
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice, LoadDialog  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SRC = OUT / "lapse.zarr"
shutil.rmtree(SRC, ignore_errors=True)
shutil.rmtree(OUT / "crops", ignore_errors=True)
results, messages = [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


crop_gui.QMessageBox.warning = staticmethod(
    lambda *a, **k: messages.append(a[1:3]))
crop_gui.QMessageBox.critical = staticmethod(
    lambda *a, **k: messages.append(a[1:3]))
crop_gui._confirm = lambda *a: True

data0 = make_test_image(SRC, shape0=(9, 2, 20, 120, 100), axes="tczyx",
                        factors=[(1, 1, 1, 1, 1), (1, 1, 1, 2, 2),
                                 (1, 1, 2, 4, 4)],
                        scale0=(30.0, 1.0, 2.0, 0.5, 0.5))
viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(SRC, choice=LoadChoice())
pump(4)
image, o = w.image, w.ortho
ranges = {a: image.extent(a) for a in image.range_axes}
ranges["t"] = (30.0, 210.0)                      # timepoints 2-8
box = vc.CropBox("lapse", ranges)
w.boxes = [box]
w._refresh(select=0)
pump(1)

# --- Timepoint numbers ------------------------------------------------------------
start, stop = w.range_spins["t"]
check("t fields: timepoints 1-9, the region's 2-8, with their times",
      (start.minimum(), start.maximum()) == (1, 9)
      and (start.value(), stop.value()) == (2, 8)
      and "(30 min)" in start.text() and "(210 min)" in stop.text(),
      f"({start.text()!r}, {stop.text()!r})")

# --- Every n-th -------------------------------------------------------------------
w.t_step_spin.setValue(3)
pump(1)
check("every 3rd: the region's step, said in the plan text",
      box.t_step == 3 and "t 2–8, every 3rd: 3 kept (30–210 min)"
      in w.plan_label.text(), repr(w.plan_label.text().splitlines()[0]))
start.setValue(3)
pump(1)
# (kept: 3 and 6; the crop's range ends at its last kept timepoint)
check("start = timepoint 3: the region starts at its time (60 min)",
      box.ranges["t"][0] == 60.0 and "t 3–6, every 3rd: 2 kept"
      in w.plan_label.text(), repr(w.plan_label.text().splitlines()[0]))
o._set_cursor({"t": 120.0})                      # timepoint 5
pump(1)
w._set_from_cursor("t", 1)
pump(1)
check("'here' takes the timepoint shown (5)",
      stop.value() == 5 and box.ranges["t"][1] == 120.0, str(box.ranges["t"]))
start.setValue(2)
stop.setValue(8)                                 # back to 2-8, every 3rd
pump(1)

# --- Fade and preview ------------------------------------------------------------
w.show_3d_check.setChecked(True)
pump(3)


def line_alpha():
    colors = np.asarray(o.views["XY"].outline_layers["selected"].edge_color)
    frame = o.views["3D"].box_layers[0].bounding_box.line_color
    return float(colors[:, 3].max()), float(np.asarray(frame)[3])


alphas = {}
for number in (2, 3, 5, 8, 9):                   # kept: 2, 5, 8
    o._set_cursor({"t": (number - 1) * 30.0})
    pump(0.5)
    alphas[number] = line_alpha()
# (napari keeps colours as 32-bit floats: 0.3 comes back as 0.30000001)
kept = {n for n, alpha in alphas.items() if np.allclose(alpha, 1.0)}
faded = {n for n, alpha in alphas.items()
         if np.allclose(alpha, ov.FADED_ALPHA, atol=1e-6)}
check("lines (2D and 3D frame) fade at timepoints not kept",
      kept == {2, 5, 8} and faded == {3, 9}, str(alphas))
o.set_preview(True)
o._set_cursor({"t": 60.0})                       # timepoint 3: skipped
pump(1)
skipped = len(o.views["XY"].preview_outside.data)
o._set_cursor({"t": 120.0})                      # timepoint 5: kept
pump(1)
kept_rects = len(o.views["XY"].preview_outside.data)
check("crop preview: all black at a skipped timepoint",
      skipped == 1 and kept_rects == 4, f"({skipped}, {kept_rects})")
o.set_preview(False)
viewer.window._qt_window.grab().save(str(OUT / "time_window.png"))

# --- What to load: numbers from 1 --------------------------------------------------
dialog = LoadDialog(image, LoadChoice(), o.canvas_sizes(), None, parent=w)
spins = dialog.range_spins["t"]
whole = (spins[0].value(), spins[1].value(), spins[0].minimum())
spins[0].setValue(2)
spins[1].setValue(4)
chosen = dialog.ranges()["t"]
dialog.deleteLater()
check("What to load: timepoints 1-9; 2-4 = 15-105 min",
      whole == (1, 9, 1) and chosen == (15.0, 105.0), f"{whole} {chosen}")

# --- The crop ------------------------------------------------------------------------
w.out_edit.setText(str(OUT / "crops"))
start_time = time.time()
w._crop(False)
while time.time() - start_time < 120:
    pump(0.2)
    if w.crop_selected_button.isEnabled():
        break
crop = vc.open_image(OUT / "crops" / "lapse_lapse.zarr")
check("the crop: timepoints 2, 5, 8, 90 min apart, checked OK",
      np.array_equal(crop.levels[0].array[...], data0[[1, 4, 7]])
      and abs(crop.levels[0].scale[0] - 90.0) < 1e-9
      and abs(crop.levels[0].translation[0] - 30.0) < 1e-9
      and "checked OK" in w.status.text() and not messages,
      w.status.text().replace("\n", " | ")[:200] + str(messages))
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
viewer.close()
pump(0.5)
