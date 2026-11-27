"""The 0.6.0 fixes, checked in a VISIBLE window. Package 1: box names that
give one file name, boxes files with mistakes or from Excel, and an output
folder inside the source image. Package 3: the report of the check.
Package 4: single boxes hidden or shown, questions before boxes are lost,
disk space, Cancel.

Usage: python fixes_0_6_gui_test.py OUT_DIR   (OUT_DIR: a scratch folder)
Tests the folder in VC_TOOL (default: the tool folder this script is in).
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
import errno  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import zarr  # noqa: E402
from napari.layers.base._base_constants import ActionType  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
results = []
messages = []          # (title, text) of every message box


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def record_message(parent, title, text, *args, **kwargs):
    messages.append((title, text))


# Message boxes and file dialogs answer by themselves.
for kind in ("warning", "critical", "information"):
    setattr(crop_gui.QMessageBox, kind, record_message)
answers = {}
crop_gui.QInputDialog.getText = lambda *a, **k: (answers["name"], True)
crop_gui.QFileDialog.getOpenFileName = lambda *a, **k: (answers["file"], "")
crop_gui.QFileDialog.getSaveFileName = lambda *a, **k: (answers["save"], "")
asked = []             # (title, text) of every question


def fake_choose(parent, title, text, options):
    asked.append((title, text))
    return answers.get("choose")        # None: Cancel


def fake_confirm(parent, title, text):
    asked.append((title, text))
    return answers.get("confirm", True)


crop_gui._choose = fake_choose
crop_gui._confirm = fake_confirm

src = OUT / "src.zarr"
if not src.exists():
    make_test_image(src)
image = vc.open_image(src)
viewer, w = crop_gui.open_window(show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
w.open_image(src)
pump(2)

# 1. New boxes skip names that give the same file name as an existing box.
w._add_box()
answers["name"] = "Crop02"
w._refresh(select=0)
w._rename_box()
w._add_box()
w._add_box()
names = [b.name for b in w.boxes]
check("new box names avoid file-name clashes",
      names == ["Crop02", "crop01", "crop03"], str(names))

# 2. Renaming to a name that gives the same file name is refused.
messages.clear()
answers["name"] = "CROP03"         # differs from "crop03" only in case
w._refresh(select=1)
w._rename_box()
check("rename clash refused", w.boxes[1].name == "crop01"
      and messages and "same file name" in messages[-1][1],
      repr(messages[-1][1][:70]) if messages else "no message")

# 3. A boxes file with a stop but no start: a clear message, boxes kept.
header = ("name,z_start_um,z_stop_um,y_start_um,y_stop_um,x_start_um,"
          "x_stop_um,channels\n")
bad = OUT / "stop_only.csv"
bad.write_text(header + "a,,20,,,,,all\n", encoding="utf-8")
messages.clear()
answers["file"] = str(bad)
w._load_boxes()
text = messages[-1][1] if messages else ""
check("stop-only row refused with a plain message",
      "z_stop_um is filled in but z_start_um is empty" in text
      and "ValueError" not in text and len(w.boxes) == 3, repr(text[:90]))

# 4. A file as Excel saves it in a German/French setup loads.
excel = OUT / "excel.csv"
excel.write_bytes((
    "name;z_start_um;z_stop_um;y_start_um;y_stop_um;x_start_um;x_stop_um;"
    "channels\r\ntête 1;1,5;20;;;3;7,25;all\r\nzwei;2;10,5;;;;;\"0;2\"\r\n"
    ";;;;;;;\r\n").encode("cp1252"))
messages.clear()
answers["file"] = str(excel)
answers["choose"] = "Replace"       # boxes exist: Add or Replace?
w._load_boxes()
check("Excel-style boxes file loads", [b.name for b in w.boxes]
      == ["tête 1", "zwei"] and not messages,
      str([b.name for b in w.boxes]) + str(messages))

# 5. An output folder inside the source image is refused before it exists.
messages.clear()
w.out_edit.setText(str(src / "crops"))
w._crop(all_boxes=True)
pump(0.5)
check("output folder inside the source refused",
      messages and "inside the source image" in messages[-1][1]
      and not (src / "crops").exists(),
      repr(messages[-1][1][:70]) if messages else "no message")


def crop_and_wait():
    messages.clear()
    w._crop(all_boxes=True)
    end = time.time() + 120
    pump(0.3)
    while w.cancel_button.isEnabled() and time.time() < end:
        pump(0.2)
    pump(0.5)


# 6. 'Overwrite' does not replace the crop of another box whose name gives
# the same file name (here from an earlier session).
def gui_box(name, x_range):
    """A box with every axis filled in, as the GUI keeps them."""
    ranges = {axis: image.extent(axis) for axis in image.range_axes}
    ranges["x"] = x_range
    return vc.CropBox(name, ranges)


crops = OUT / "crops"
w.out_edit.setText(str(crops))
w.overwrite_check.setChecked(True)
w.boxes = [gui_box("tail 1", (5, 10))]
w._refresh(select=0)
crop_and_wait()
first = vc.output_path_for(image, w.boxes[0], crops)
check("first crop written", first.exists(), str(messages))
w.boxes = [gui_box("tail_1", (20, 30))]
w._refresh(select=0)
crop_and_wait()
record = zarr.open_group(first, mode="r").attrs["crop_provenance"]
check("another box's crop not replaced",
      record["box_name"] == "tail 1" and messages
      and "holds the crop of box 'tail 1'" in messages[-1][1],
      repr(messages[-1][1][:80]) if messages else "no message")

# --- Package 3: what the check says after a crop -------------------------
w.boxes = [gui_box("tail 1", (5, 10))]
w._refresh(select=0)
crop_and_wait()
status = w.status.text()
check("status says what was checked", "checked OK" in status
      and "position, voxel size and format" in status, repr(status[-120:]))

original_verify = vc.verify_crop
vc.verify_crop = lambda *a, **k: ["level 0: a test problem"]
crop_and_wait()
vc.verify_crop = original_verify
text = messages[-1][1] if messages else ""
check("failed check: warning, crop kept",
      messages and messages[-1][0] == "Check failed"
      and "The crops were kept" in text and "a test problem" in text
      and first.exists(), repr(text[:100]))

# --- Package 4: safety nets and single boxes shown or hidden ---------------
check("version in the title", vc.__version__ in viewer.title, viewer.title)

w.show_3d_check.setChecked(True)
pump(2)
w.boxes = [gui_box(n, x) for n, x in (("one", (5, 10)), ("two", (12, 18)),
                                      ("three", (20, 26)))]
w._refresh(select=0)
pump(0.5)
xy, view3d = w.ortho.views["XY"], w.ortho.views["3D"]
w.box_list.item(1).setCheckState(Qt.CheckState.Unchecked)     # hide "two"
pump(0.5)
check("hidden box: no shape, name or 3D frame",
      xy.shape_boxes == [0, 2] and len(xy.shapes.data) == 2
      and list(xy.shapes.features["name"]) == ["one", "three"]
      and [layer.visible for layer in view3d.box_layers]
      == [True, False, True], f"{xy.shape_boxes} "
      f"{[layer.visible for layer in view3d.box_layers]}")
pump(1)
viewer.window._qt_window.grab().save(str(OUT / "box_two_hidden.png"))
pieces = sum(len(layer.data) for layer in xy.outline_layers.values())
check("hidden box: no lines (2 boxes x 4 sides)", pieces == 8,
      f"{pieces} line pieces")

# Editing the second shape edits box "three", not the hidden "two".
data = [np.array(d) for d in xy.shapes.data]
data[1][:, -1] += 2.0                       # 2 µm to the right
w.ortho._syncing["shapes"] = True
xy.shapes.data = data                       # as if it had been dragged
w.ortho._syncing["shapes"] = False
w.ortho._on_shapes(xy, SimpleNamespace(action=ActionType.CHANGED,
                                       data_indices=[1]))
pump(0.5)
check("an edit reaches the right box", abs(w.boxes[2].ranges["x"][0] - 22)
      < 0.3 and tuple(w.boxes[1].ranges["x"]) == (12, 18),
      f"three {w.boxes[2].ranges['x']}, two {w.boxes[1].ranges['x']}")
w.ortho._on_shapes(xy, SimpleNamespace(action=ActionType.REMOVING,
                                       data_indices=[1]))
pump(0.5)
check("deleting a shape deletes the right box; 'two' stays hidden",
      [b.name for b in w.boxes] == ["one", "two"]
      and w.box_list.item(1).checkState() == Qt.CheckState.Unchecked,
      str([b.name for b in w.boxes]))
w.box_list.item(1).setCheckState(Qt.CheckState.Checked)
pump(0.5)
check("shown again", xy.shape_boxes == [0, 1], str(xy.shape_boxes))
w.show_3d_check.setChecked(False)
pump(0.5)

# Opening another image asks first.
asked.clear()
answers["choose"] = None                          # Cancel
w.open_image(src)
check("opening: Cancel keeps the boxes", len(w.boxes) == 2 and asked
      and asked[-1][0] == "Crop regions", str(asked[-1:]))
saved = OUT / "saved.csv"
saved.unlink(missing_ok=True)
answers.update(choose="Save regions first", save=str(saved))
w.open_image(src)
pump(1)
check("opening: saved first, then opened", saved.exists() and not w.boxes)

# Loading boxes while boxes exist: Add, a clash refused, Replace.
w.boxes = [gui_box("one", (5, 10))]
w._refresh(select=0)
answers.update(file=str(excel), choose="Add")
w._load_boxes()
check("load: Add", [b.name for b in w.boxes] == ["one", "tête 1", "zwei"],
      str([b.name for b in w.boxes]))
messages.clear()
w._load_boxes()                                   # the same names again
check("load: Add refuses clashing names", len(w.boxes) == 3 and messages
      and "same file name" in messages[-1][1])
answers["choose"] = "Replace"
w._load_boxes()
check("load: Replace", [b.name for b in w.boxes] == ["tête 1", "zwei"])

# Not enough disk space: asked first; No stops before anything starts.
w.boxes = [gui_box("small", (5, 10))]
w._refresh(select=0)
real_disk_usage = crop_gui.shutil.disk_usage
crop_gui.shutil.disk_usage = lambda path: SimpleNamespace(free=1000)
asked.clear()
answers["confirm"] = False
w._crop(all_boxes=True)
pump(0.5)
crop_gui.shutil.disk_usage = real_disk_usage
answers["confirm"] = True
check("low disk space: asked, nothing started", asked
      and asked[-1][0] == "Disk space" and w.crop_all_button.isEnabled(),
      str(asked[-1:]))
messages.clear()
w._on_crop_error(OSError(errno.ENOSPC, "No space left on device"))
check("full disk explained", messages
      and "output drive is full" in messages[-1][1])

# Cancel: "Stopping…", then what to do.
original_selection = vc._source_selection


def slow(*args):
    time.sleep(0.2)
    return original_selection(*args)


vc._source_selection = slow
w.workers_spin.setValue(1)
w.boxes = [gui_box("big", (0, 32))]
w._refresh(select=0)
w._crop(all_boxes=True)
pump(0.6)
w.cancel_button.click()
pump(0.1)
stopping = (w.cancel_button.text() == "Stopping…"
            and not w.cancel_button.isEnabled())
end = time.time() + 60
while not w.crop_all_button.isEnabled() and time.time() < end:
    pump(0.2)
vc._source_selection = original_selection
check("Cancel: 'Stopping…', then what to do", stopping
      and "Overwrite earlier crops" in w.status.text()
      and w.cancel_button.text() == "Cancel", repr(w.status.text()[:120]))

viewer.close()
pump(0.5)
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
sys.exit(0 if all(results) else 1)
