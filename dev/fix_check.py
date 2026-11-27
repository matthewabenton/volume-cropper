"""2D/3D button error and right-panel dragging (scratchpad only; synthetic).

Usage: python fix_check.py TOOL_DIR   (the shared copy or a local copy)
Visible window, so that drawing and dragging are real.
"""
import faulthandler
import io
import sys
import tempfile
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
import numpy as np  # noqa: E402
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from qtpy.QtGui import QMouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication, QStyle  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from test_volume_cropper import make_test_image  # noqa: E402

# Count drawing errors: vispy prints them to stderr.
captured = io.StringIO()
real_stderr = sys.stderr


class Tee:
    def write(self, text):
        captured.write(text)
        return real_stderr.write(text)

    def flush(self):
        real_stderr.flush()


sys.stderr = Tee()


def pump(s=1.0):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def errors():
    return captured.getvalue().count("Last dimension should be")


tmp = Path(tempfile.mkdtemp())
src = tmp / "synthetic.zarr"
make_test_image(src, shape0=(2, 48, 200, 180),
                factors=[(1, 1, 1, 1), (1, 1, 4, 4), (1, 2, 8, 8)])
viewer, w = crop_gui.open_window(show=True)
win = viewer.window._qt_window
win.showNormal()
win.resize(1366, 768)
pump(1)
w.open_image(src)
pump(2)
o = w.ortho
buttons = viewer.window._qt_viewer.viewerButtons
print("napari buttons visible: 2D/3D", buttons.ndisplayButton.isVisible(),
      "roll", buttons.rollDimsButton.isVisible(), flush=True)

# 1. What the 2D/3D button (or Ctrl+Y) does to the XY view.
viewer.dims.ndisplay = 3
pump(2)
print(f"after 2D/3D in XY: ndisplay {viewer.dims.ndisplay}, drawing errors "
      f"{errors()}", flush=True)
viewer.dims.ndisplay = 2
pump(1)
viewer.dims.order = (1, 0, 2)                  # transpose
pump(1)
viewer.grid.enabled = True                    # grid mode
pump(1)
print(f"after transpose + grid: order {tuple(viewer.dims.order)}, grid "
      f"{viewer.grid.enabled}", flush=True)
viewer.dims.order = (0, 1, 2)
viewer.grid.enabled = False
pump(1)

# 2. Drag the border between the views and the right panel, as a mouse does.
def panel_width():
    if hasattr(w, "dock"):                       # an older version: a dock
        return w.dock.width()
    return o._outer.sizes()[1]                    # the fixed version


def drag(dx):
    if hasattr(w, "dock"):          # a napari dock: drag the window separator
        target = win
        extent = win.style().pixelMetric(
            QStyle.PixelMetric.PM_DockWidgetSeparatorExtent)
        x = w.dock.geometry().left() - max(extent // 2, 1)
        y = w.dock.geometry().center().y()
    else:                           # a splitter: drag its handle
        target = o._outer.handle(1)
        x, y = target.width() // 2, target.height() // 2
    before = panel_width()
    for kind, px, buttons_held in ((QEvent.Type.MouseButtonPress, x, True),
                                   (QEvent.Type.MouseMove, x + dx // 2, True),
                                   (QEvent.Type.MouseMove, x + dx, True),
                                   (QEvent.Type.MouseButtonRelease, x + dx, False)):
        pos = QPointF(px, y)
        held = Qt.MouseButton.LeftButton if buttons_held else Qt.MouseButton.NoButton
        event = QMouseEvent(kind, pos,
                            QPointF(target.mapToGlobal(QPoint(px, y))),
                            Qt.MouseButton.LeftButton, held,
                            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(target, event)
        pump(0.3)
    print(f"drag border by {dx:+d} px: panel {before} -> {panel_width()} px",
          flush=True)


print("panel width at start", panel_width(), flush=True)
drag(+250)          # narrower
drag(-150)          # wider again
drag(+2000)         # all the way to the edge: closes the panel
drag(-400)          # and back open
print("views and panel:", o._outer.sizes() if hasattr(o, "_outer") else "-",
      flush=True)
win.resize(1366, 768)
pump(1)
image = win.grab().toImage()
image.save(str(Path("v03_out") / f"fix_check_{Path(sys.argv[1]).name}.png"))
print("done", flush=True)
viewer.close()
