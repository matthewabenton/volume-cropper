"""0.8.0: OME-Zarr 0.4 (Zarr v2) images in a VISIBLE window.

A synthetic 0.4 image with 3 channels: the image summary, the channel
controls, hiding a channel, and a crop of two channels (0.4, checked,
recorded). (Until 0.22.1 a first part checked a real 0.4 image of the
maintainer's; it left this script in 0.23.0.)

Usage: python v04_gui_test.py OUT_DIR   (a scratch folder: its outputs are replaced)
Tests the folder in VC_TOOL (default: the tool folder this script is in).
Crops go to OUT_DIR/crops. Saves synthetic_window.png.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
results, messages = [], []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def wait_for(condition, timeout):
    end = time.time() + timeout
    while not condition() and time.time() < end:
        pump(0.2)
    return condition()


for kind in ("warning", "critical", "information"):
    setattr(crop_gui.QMessageBox, kind,
            lambda parent, title, text, *a, **k: messages.append((title, text)))
# Large-read questions (exact level, big 3D volume) are answered "yes";
# before another image replaces drawn regions: "Discard them".
crop_gui.QMessageBox.question = staticmethod(
    lambda *a, **k: crop_gui.QMessageBox.StandardButton.Yes)
crop_gui._choose = lambda parent, title, text, options: (
    "Discard them" if "Discard them" in options else options[0])
# "Open last crop" asks what to load (a modal dialog); answer with the
# defaults, and count the questions.
asked = []
crop_gui.CropWidget._ask_what_to_load = (
    lambda self, image, choice, box_ranges=None:
    asked.append(image.path) or choice)
def draw(view, points):
    """As if drawn with the polygon tool on the view's current slice."""
    view.shapes.mode = "add_polygon"
    view.shapes.add([np.asarray(points, dtype=float)], shape_type="polygon")
    pump(0.8)


def crop_and_wait(w, timeout=300):
    w._crop(all_boxes=False)
    pump(0.5)
    return wait_for(lambda: not w.cancel_button.isEnabled(), timeout)


viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
xy, xz, yz = (w.ortho.views[name] for name in ("XY", "XZ", "YZ"))

# 2. A synthetic 0.4 image with 3 channels ----------------------------------
src = OUT / "synthetic_v04.zarr"
if not src.exists():
    make_test_image(src, version="0.4")   # c 3, z 0-80 um, y 0-37.5, x 0-32.5
w.open_image(src, choice=LoadChoice())
pump(2)
check("synthetic: summary and 3 channel boxes",
      w.image_info.text().startswith("OME-Zarr 0.4")
      and len(w.show_checks) == 3 and len(w.channel_checks) == 3,
      w.image_info.text().splitlines()[0])
w.show_checks[1].setChecked(False)
pump(1)
check("synthetic: hiding channel 1", not w.ortho.display[1]["visible"])
w._new_region()
pump(0.5)
w.ortho.go_to_slice(("y", "x"), 40.0)
pump(0.5)
draw(xy, [(8.1, 6.1), (8.1, 20.3), (24.1, 13.2)])
w._finish_region()
w._refresh(select=0)
pump(0.5)
w.channel_checks[1].setChecked(False)               # save channels 0 and 2
pump(0.5)
viewer.window._qt_window.grab().save(str(OUT / "synthetic_window.png"))
w.out_edit.setText(str(OUT / "crops"))
w.overwrite_check.setChecked(True)
done = crop_and_wait(w, 120)
out = vc.output_path_for(w.image, w.boxes[0], OUT / "crops")
record = (zarr.open_group(out, mode="r").attrs.get("crop_provenance", {})
          if out.exists() else {})
check("synthetic: crop of channels 0 and 2, 0.4, checked", done
      and record.get("channels") == [0, 2]
      and record.get("verified", {}).get("ok") is True
      and vc.open_image(out).ome_version == "0.4", w.status.text()[:90])
check("no messages", not messages, str(messages[:2]))

viewer.close()
pump(0.5)
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
sys.exit(0 if all(results) else 1)
