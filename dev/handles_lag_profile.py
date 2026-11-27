"""Where the time goes while a dot is dragged in the 3D view (0.18.0,
the user found resizing in 3D laggy, 2026-11-12). VISIBLE window, the
synthetic embryo demo, real Qt mouse events.

Usage: python handles_lag_profile.py OUT_DIR

Drags the high-x face of a crop region over 30 mouse moves, once for a
plain region and once for one with a polygon (whose blacked-out voxels the
3D view then shows dimmed). Prints, per move, the time the window took,
how much of it went to the panel's update (_box_ranges_dragged) and to
moving the dots and wireframe (_show_dragged), and the top functions of a
profile (cProfile) of the whole drag. Writes profile_<case>.txt to OUT_DIR.
Synthetic data only; closes its window.
"""
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(OUT / "settings.json")
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
import numpy as np  # noqa: E402
from qtpy.QtCore import QEvent, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QMouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import box_handles as bh  # noqa: E402
from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402
from volume_cropper import core as vc  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data" / "embryo_tissue_demo.zarr"
timings = {"panel": [], "show": []}


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def timed(name, function):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[name].append(time.perf_counter() - t0)
    return wrapper


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
ortho.on_box_ranges = timed("panel", w._box_ranges_dragged)
ortho._show_dragged = timed("show", ortho._show_dragged)
view = ortho.views["3D"]
cam = ov.camera(view.model)
cam.angles = (0.0, 35.0, 60.0)
canvas = ortho._qt_viewers["3D"].canvas
widget = canvas.native
spatial = view.order[-3:]


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
    t0 = time.perf_counter()
    QApplication.sendEvent(widget, event)
    QApplication.processEvents()       # the redraws it caused
    return time.perf_counter() - t0


def run(case, outlines):
    i = w._new_box({"z": (300.0, 500.0), "y": (400.0, 600.0),
                    "x": (400.0, 600.0)})
    if outlines:
        cy, cx = 500.0, 500.0
        hexagon = [(cy + 90 * np.sin(a), cx + 90 * np.cos(a))
                   for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
        w.boxes[i].outlines = vc.normalize_outlines({("y", "x"): hexagon})
    w._refresh(select=i)
    pump(1.5)
    for values in timings.values():
        values.clear()
    face = bh.handle_points(ortho.box_ranges[i], spatial)[("z", 1)]
    start, end = on_screen(face), on_screen(face + (120.0, 0, 0))
    profiler = cProfile.Profile()
    profiler.enable()
    steps = []
    mouse(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    for xy in np.linspace(start, end, 31)[1:]:
        steps.append(mouse(QEvent.Type.MouseMove, xy,
                           Qt.MouseButton.LeftButton))
    release = mouse(QEvent.Type.MouseButtonRelease, end,
                    Qt.MouseButton.NoButton)
    profiler.disable()
    pump(1.0)
    steps = np.array(steps) * 1000
    print(f"--- {case}: per move {np.median(steps):.0f} ms median, "
          f"{steps.max():.0f} ms max; release {release * 1000:.0f} ms")
    for name, values in timings.items():
        if values:
            ms = np.array(values) * 1000
            print(f"    {name}: {len(ms)} calls, median {np.median(ms):.0f} "
                  f"ms, max {ms.max():.0f} ms")
    text = io.StringIO()
    stats = pstats.Stats(profiler, stream=text).sort_stats("cumulative")
    stats.print_stats(40)
    stats.print_stats(r"volume_cropper", 45)   # the tool's own
    stats.print_callers(r"base.py:949\(visible\)")         # who shows/hides
    (OUT / f"profile_{case}.txt").write_text(text.getvalue())
    w._remove_box()
    pump(0.5)


run("plain", False)
run("polygon", True)
viewer.close()
pump(0.5)
