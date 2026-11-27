"""0.18.0 dots that change a crop region in the 3D view, in a VISIBLE
window, on the synthetic embryo demo, with real mouse events.

Usage: python handles_0_18_gui_test.py OUT_DIR

Opens volume_cropper_demo_data/embryo_tissue_demo.zarr (synthetic, read
only), turns the 3D view on, makes a crop region and turns the camera to
an oblique angle. Then it presses, moves and releases the mouse on the 3D
view's canvas (Qt mouse events, as a person would): the high-x face's dot
dragged to where the face 50 µm further out lies on screen; the centre's
dot dragged to where the centre 40 µm further along y lies. Checks that
the face (or the region) ends there, the other faces stay, the camera does
not turn during the drags, and a drag on an empty part of the view still
turns it. Saves handles_before.png and handles_after.png (the window).
Then the second round (the user's choices of 2026-11-12): Shift-drag
resizes around the centre, Ctrl-drag lands on a whole step, Ctrl+Z /
Ctrl+Shift+Z (keys sent to the 3D canvas) undo and redo, Up / Shift+Down
nudge the face last clicked, clipping follows a drag, a click on another
region's frame selects it, Zoom to region centres the 3D view.
Saves handles_round2.png too.
Synthetic data only. Answers every dialog itself and closes its window.
Uses a settings file of its own in OUT_DIR.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(OUT / "settings.json")
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
import numpy as np  # noqa: E402
from qtpy.QtCore import QEvent, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QMouseEvent  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import box_handles as bh  # noqa: E402
from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data" / "embryo_tissue_demo.zarr"
results, messages = [], []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def wait_for(condition, seconds=20.0):
    end = time.time() + seconds
    while time.time() < end:
        pump(0.1)
        if condition():
            return True
    return False


load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
for kind in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, kind, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True

viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
w.open_image(DEMO, choice=LoadChoice())
pump(2.0)
ortho = w.ortho
w.show_3d_check.setChecked(True)
wait_for(lambda: w._loading is None, 30)
pump(1.0)
i = w._new_box({"z": (300.0, 500.0), "y": (400.0, 600.0),
                "x": (400.0, 600.0)})
w._refresh(select=i)
view = ortho.views["3D"]
cam = ov.camera(view.model)
cam.angles = (0.0, 35.0, 60.0)               # oblique: no axis along sight
pump(1.0)
check("the selected region has its seven dots in 3D",
      view.handles is not None and view.handles.visible
      and len(view.handles.data) == 7)
viewer.screenshot(str(OUT / "handles_before.png"), canvas_only=False)

canvas = ortho._qt_viewers["3D"].canvas
widget = canvas.native
spatial = view.order[-3:]


def on_screen(point):
    """A point (µm, in the order of spatial) on the 3D canvas, in pixels."""
    transform = canvas.view.transform * canvas.view.scene.transform
    mapped = transform.map(list(np.asarray(point, float)[::-1]) + [1.0])
    return mapped[:2] / mapped[3]


def mouse(kind, xy, buttons, modifiers=Qt.KeyboardModifier.NoModifier):
    button = Qt.MouseButton.LeftButton
    event = QMouseEvent(kind, QPointF(*xy),
                        QPointF(widget.mapToGlobal(QPointF(*xy).toPoint())),
                        button if kind != QEvent.Type.MouseMove
                        else Qt.MouseButton.NoButton,
                        buttons, modifiers)
    QApplication.sendEvent(widget, event)
    pump(0.05)


def drag(start, end, steps=8, modifiers=Qt.KeyboardModifier.NoModifier):
    """Press, move in steps, release (steps=1: a click without moving)."""
    mouse(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton,
          modifiers)
    for xy in np.linspace(start, end, steps)[1:]:
        mouse(QEvent.Type.MouseMove, xy, Qt.MouseButton.LeftButton,
              modifiers)
    mouse(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton,
          modifiers)
    pump(0.5)


box = w.boxes[i]
drawn = dict(ortho.box_ranges[i])
points = bh.handle_points(drawn, spatial)
angles = tuple(cam.angles)
face = points[("x", 1)]
target = face + np.array([0.0, 0.0, 50.0])
drag(on_screen(face), on_screen(target))
x_high = box.ranges["x"][1]
check("the high-x dot: the face goes 50 µm further out",
      abs(x_high - (drawn["x"][1] + 50)) < 6
      and box.ranges["x"][0] == 400.0 and box.ranges["y"] == (400.0, 600.0)
      and box.ranges["z"] == (300.0, 500.0),
      f"x {box.ranges['x']} (was drawn {drawn['x']})")
check("the camera did not turn while dragging", tuple(cam.angles) == angles,
      f"{angles} -> {tuple(cam.angles)}")

before = dict(box.ranges)
drawn = dict(ortho.box_ranges[i])
centre = bh.handle_points(drawn, spatial)[bh.CENTRE]
drag(on_screen(centre), on_screen(centre + np.array([0.0, 40.0, 0.0])))
shift = np.array([box.ranges[a][0] - before[a][0] for a in spatial])
# The centre moves across the screen (in the plane facing the camera):
# to the screen place of "40 µm along y" means 40 µm along y less its part
# along the line of sight.
sight = np.asarray(cam.view_direction, float)
sight /= np.linalg.norm(sight)
wanted = np.array([0.0, 40.0, 0.0])
expected = wanted - (wanted @ sight) * sight
check("the centre dot: the region moves across the screen, keeping its size",
      np.abs(shift - expected).max() < 6
      and all(abs((box.ranges[a][1] - box.ranges[a][0])
                  - (before[a][1] - before[a][0])) < 1e-6 for a in spatial),
      f"shift {np.round(shift, 1)}, expected {np.round(expected, 1)}")
check("the camera still did not turn", tuple(cam.angles) == angles)
viewer.screenshot(str(OUT / "handles_after.png"), canvas_only=False)

# --- the second round (the user's choices of 2026-11-12) -------------------
voxel = {a: ortho._voxel_size(a) for a in spatial}


def held():
    return {a: tuple(box.ranges[a]) for a in spatial}


# Shift: the high-x face 30 µm out, the low one 30 µm the other way
before, drawn = held(), dict(ortho.box_ranges[i])
face = bh.handle_points(drawn, spatial)[("x", 1)]
drag(on_screen(face), on_screen(face + np.array([0.0, 0.0, 30.0])),
     modifiers=Qt.KeyboardModifier.ShiftModifier)
grow = box.ranges["x"][1] - before["x"][1]
check("Shift-drag: the opposite face moves as far the other way",
      abs(grow - 30) < 6 and abs((before["x"][0] - box.ranges["x"][0])
                                 - grow) < 1e-6,
      f"x {before['x']} -> {box.ranges['x']}")
# Ctrl: the high-z face to a round value
step = bh.snap_step({a: ortho.extent(a) for a in spatial})
drawn = dict(ortho.box_ranges[i])
face = bh.handle_points(drawn, spatial)[("z", 1)]
drag(on_screen(face), on_screen(face + np.array([47.0, 0.0, 0.0])),
     modifiers=Qt.KeyboardModifier.ControlModifier)
z_high = box.ranges["z"][1]
check("Ctrl-drag: the face lands on a whole step",
      abs(z_high / step - round(z_high / step)) < 1e-6
      and abs(z_high - (drawn["z"][1] + 47)) <= step,
      f"z high {z_high} (step {step} µm)")

# Ctrl+Z / Ctrl+Shift+Z with the 3D view's canvas holding the keyboard
widget.setFocus()
pump(0.3)
after_ctrl = held()
QTest.keyClick(widget, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
pump(0.5)
undone = held()
QTest.keyClick(widget, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
               | Qt.KeyboardModifier.ShiftModifier)
pump(0.5)
check("Ctrl+Z undoes the last drag, Ctrl+Shift+Z redoes it",
      undone["z"] != after_ctrl["z"] and held() == after_ctrl,
      f"z {after_ctrl['z']} -> {undone['z']} -> {held()['z']}")

# Up / Shift+Up: the face last pressed (a click on its dot)
face = bh.handle_points(dict(ortho.box_ranges[i]), spatial)[("x", 1)]
drag(on_screen(face), on_screen(face), steps=1)
x_high = box.ranges["x"][1]
widget.setFocus()
QTest.keyClick(widget, Qt.Key.Key_Up)
pump(0.5)
one = box.ranges["x"][1]
QTest.keyClick(widget, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
pump(0.5)
check("Up moves the face one voxel out, Shift+Down ten in",
      abs(one - (x_high + voxel["x"])) < 1e-6
      and abs(box.ranges["x"][1] - (one - 10 * voxel["x"])) < 1e-6,
      f"x high {x_high} -> {one} -> {box.ranges['x'][1]}")

# live clipping during a drag
w.clip_check.setChecked(True)
pump(1.0)
drawn = dict(ortho.box_ranges[i])
face = bh.handle_points(drawn, spatial)[("y", 0)]
target = face + np.array([0.0, -40.0, 0.0])
mouse(QEvent.Type.MouseButtonPress, on_screen(face),
      Qt.MouseButton.LeftButton)
for xy in np.linspace(on_screen(face), on_screen(target), 6)[1:]:
    mouse(QEvent.Type.MouseMove, xy, Qt.MouseButton.LeftButton)
planes = [tuple(p.position) for p in
          view.channel_layers[0].experimental_clipping_planes]
mid_drag = min(p[1] for p in planes if abs(p[1]) > 0)
mouse(QEvent.Type.MouseButtonRelease, on_screen(target),
      Qt.MouseButton.NoButton)
pump(0.5)
check("clipping follows the face during the drag",
      abs(mid_drag - (drawn["y"][0] - 40)) < 6,
      f"lowest y plane {mid_drag:.1f} (face dragged to "
      f"{drawn['y'][0] - 40:.1f})")
w.clip_check.setChecked(False)
pump(0.5)

# a click on another region's frame selects it
j = w._new_box({"z": (620.0, 760.0), "y": (780.0, 950.0),
                "x": (780.0, 950.0)})        # well away from the first
w._refresh(select=i)
# the whole image in view again (clipping zoomed onto the first region)
view.model.reset_view()
cam.angles = (0.0, 35.0, 60.0)
pump(1.0)
corner = np.array([620.0, 865.0, 780.0])   # middle of an edge along y
# (log what the click finds: a dot of the selected region, or a frame)
found = []
original = {name: getattr(bh, name) for name in ("pick_handle",
                                                 "pick_region")}
for name, function in original.items():
    def logged(*args, _name=name, _function=function, **kwargs):
        result = _function(*args, **kwargs)
        found.append((_name, result))
        return result
    setattr(bh, name, logged)
xy = on_screen(corner)
print(f"  click at {np.round(xy, 1)} on a canvas of {widget.width()} x "
      f"{widget.height()}, zoom {cam.zoom:.3f}", flush=True)
drag(xy, xy, steps=1)
for name, function in original.items():
    setattr(bh, name, function)
print(f"  found: {found}", flush=True)
check("a click on another region's frame selects it", w._selected() == j,
      f"selected {w._selected()}")
# Zoom to region also centres the 3D view
ortho.zoom_to_selected()
pump(0.5)
centre = [float(np.mean(ortho.box_ranges[j][a])) for a in spatial]
check("Zoom to region centres the 3D view on the region",
      np.allclose(tuple(cam.center)[-3:], centre, atol=1e-3),
      f"{tuple(cam.center)[-3:]} vs {centre}")
viewer.screenshot(str(OUT / "handles_round2.png"), canvas_only=False)

size = widget.width(), widget.height()
drag((size[0] * 0.08, size[1] * 0.1), (size[0] * 0.3, size[1] * 0.25))
check("a drag on an empty part still turns the view",
      tuple(cam.angles) != angles, f"{angles} -> {tuple(cam.angles)}")

print("dialogs answered:", messages)
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
