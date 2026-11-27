"""Diagnostic (0.18.0): what the 3D view's handler receives for a click on a
dot, a click on another region's frame and an Up key press, in a VISIBLE
window on the synthetic demo. Prints what it finds; closes its window.

Usage: python handles_click_debug.py OUT_DIR
"""
import os
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(OUT / "settings.json")
TOOL = Path(__file__).resolve().parents[1]
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


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


load_dialog.FileLoadDialog.exec = lambda self: 1
crop_gui._confirm = lambda *a: True
viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
w.open_image(DEMO, choice=LoadChoice())
pump(2.0)
w.show_3d_check.setChecked(True)
pump(3.0)
ortho = w.ortho
i = w._new_box({"z": (300.0, 500.0), "y": (400.0, 600.0),
                "x": (400.0, 600.0)})
j = w._new_box({"z": (620.0, 760.0), "y": (780.0, 950.0),
                "x": (780.0, 950.0)})
w._refresh(select=i)
view = ortho.views["3D"]
cam = ov.camera(view.model)
cam.angles = (0.0, 35.0, 60.0)
pump(1.0)
canvas = ortho._qt_viewers["3D"].canvas
widget = canvas.native
spatial = view.order[-3:]

# log what the handler is given
original = ortho._handle_drag


def logged(viewer_model, event):
    print(f"  press: type {event.type} button {event.button} modifiers "
          f"{event.modifiers} pos {event.pos} zoom {event.camera_zoom:.3f}",
          flush=True)
    gen = original(viewer_model, event)
    return gen


view.model.mouse_drag_callbacks[
    view.model.mouse_drag_callbacks.index(original)] = logged
ortho._click_to_select_orig = ortho._click_to_select


def on_screen(point):
    transform = canvas.view.transform * canvas.view.scene.transform
    mapped = transform.map(list(np.asarray(point, float)[::-1]) + [1.0])
    return mapped[:2] / mapped[3]


def mouse(kind, xy, buttons):
    event = QMouseEvent(kind, QPointF(*xy),
                        QPointF(widget.mapToGlobal(QPointF(*xy).toPoint())),
                        Qt.MouseButton.LeftButton
                        if kind != QEvent.Type.MouseMove
                        else Qt.MouseButton.NoButton,
                        buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)
    pump(0.1)


def click(xy):
    mouse(QEvent.Type.MouseButtonPress, xy, Qt.MouseButton.LeftButton)
    mouse(QEvent.Type.MouseButtonRelease, xy, Qt.MouseButton.NoButton)
    pump(0.5)


def state(label):
    handler = getattr(view.model, "_key_map_handler", None)
    held = getattr(handler, "_key_release_generators", None)
    print(f"[{label}] drag_3d {ortho._drag_3d} pan {cam.mouse_pan} "
          f"undo {len(w._undo)} held keys {held and list(held)} "
          f"last face {ortho._last_face}", flush=True)


if "--ctrl-first" in sys.argv:
    # as the check does: a Ctrl-drag, then Ctrl+Z and Ctrl+Shift+Z
    face = bh.handle_points(dict(ortho.box_ranges[i]), spatial)[("z", 1)]
    start, end = on_screen(face), on_screen(face + np.array([47.0, 0, 0]))
    ctrl = Qt.KeyboardModifier.ControlModifier
    for kind, xy, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, (start + end) / 2,
             Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)):
        event = QMouseEvent(kind, QPointF(*xy),
                            QPointF(widget.mapToGlobal(
                                QPointF(*xy).toPoint())),
                            Qt.MouseButton.LeftButton
                            if kind != QEvent.Type.MouseMove
                            else Qt.MouseButton.NoButton, buttons, ctrl)
        QApplication.sendEvent(widget, event)
        pump(0.1)
    pump(0.5)
    state("after the Ctrl-drag")
    widget.setFocus()
    pump(0.3)
    z = w.boxes[i].ranges["z"]
    QTest.keyClick(widget, Qt.Key.Key_Z, ctrl)
    pump(0.5)
    state(f"after Ctrl+Z (z {z} -> {w.boxes[i].ranges['z']})")
    QTest.keyClick(widget, Qt.Key.Key_Z,
                   ctrl | Qt.KeyboardModifier.ShiftModifier)
    pump(0.5)
    state(f"after Ctrl+Shift+Z (z -> {w.boxes[i].ranges['z']})")

face = bh.handle_points(dict(ortho.box_ranges[i]), spatial)[("x", 1)]
print("click on the high-x dot at", np.round(on_screen(face), 1))
click(on_screen(face))
print("  last face:", ortho._last_face)
x_high = w.boxes[i].ranges["x"][1]
widget.setFocus()
pump(0.3)
print("focus widget:", QApplication.focusWidget(), "is canvas:",
      QApplication.focusWidget() is widget, "active window:",
      QApplication.activeWindow())
nudged = []
orig_nudge = ortho._nudge
ortho._nudge = lambda steps: (nudged.append(steps), orig_nudge(steps))
QTest.keyClick(widget, Qt.Key.Key_Up)
pump(0.5)
print("Up: nudge calls", nudged, "x high", x_high, "->",
      w.boxes[i].ranges["x"][1])
print("keymap of the 3D view:", {k: getattr(f, "__name__", f) for k, f in
                                 view.model.keymap.items()})

corner = np.array([620.0, 865.0, 780.0])
xy = on_screen(corner)
print("click on region j's edge at", np.round(xy, 1))
# what pick_region would say for this pixel
click(xy)
print("  selected after the click:", w._selected(), "(j =", j, ")")
viewer.close()
pump(0.5)
