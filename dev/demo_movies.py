"""Demo films of Volume Cropper (the user's requests, 2026-11-09: "movie
clips of you using the tool"; "make it look like an actual movie of a
user"; then: half speed, no ending that opens the crop, and the main film
showing lazy loading when zooming in, choosing the resolution by hand,
drawing a box and polygons, projections (Max, slab) and Save as TIFF).
In a VISIBLE window, on synthetic data only.

Usage: python demo_movies.py main|timelapse OUT_DIR [LAPSE_DIR]

The window is driven by script, through the same calls its buttons make (a
button's own click(); a drawn shape is handed to the window as napari
hands it over). The film is the window itself (grabbed again after every
action, with a dialog over it when one is open) plus a mouse pointer
painted in: it glides to each button, list or point in a view, shows a
ring when it clicks, and traces each shape before it appears. Captions say
what happens. Frames go straight into ffmpeg (on PATH, or set FFMPEG to its
executable), 25 per second: OUT_DIR/volume_cropper_<name>.mp4. SLOW = 2: half speed.

main:      the embryo demo (embryo_tissue_demo.zarr): Open… and What to
           load, slices, zooming in (finer levels read only for what is on
           screen), a fixed coarse level for fast zooming, Max projection
           through the whole depth and through a slab that follows the
           slider, a crop box drawn in XY and XZ and saved as a TIFF, an
           embryo outlined with polygons, shading, crop preview, 3D clipped
           to the region, Crop selected and its check.
timelapse: the multicolour embryo time-lapse (make_demo_embryo_timelapse;
           LAPSE_DIR holds it, made if missing): time (drift, somites, the
           marker waves), the Autocropper with Moving region + Outline,
           seeds on the embryo and on the gastruloid, Fit to seeds, the
           region and seed following the embryo, 3D, Crop selected.
Answers every dialog itself and closes its window; a settings file of its
own (the user's settings are not used or changed); the mouse and the
screen are not touched.
"""
import faulthandler
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

faulthandler.enable()
WHICH, OUT = sys.argv[1], Path(sys.argv[2])
LAPSE_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT / "lapse"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VOLUME_CROPPER_SETTINGS"] = str(
    Path(tempfile.mkdtemp(prefix="vc_movie_settings_")) / "settings.json")
TOOL = Path(os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(TOOL))
sys.path.append(str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from qtpy.QtCore import QPoint, QPointF, QRect, Qt, QTimer  # noqa: E402
from qtpy.QtGui import (QBrush, QColor, QFont, QImage, QPainter,  # noqa: E402
                        QPen, QPolygonF)
from qtpy.QtWidgets import (QApplication, QPushButton,  # noqa: E402
                            QScrollArea)

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper import ortho_views as ov  # noqa: E402

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
DEMO = Path(os.environ.get("VC_DEMO", str(Path(__file__).resolve().parents[2] / "volume_cropper_demo_data")))
FPS, WIDTH, SLOW = 25, 1600, 2.0
messages = []
for name in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, name, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True
load_dialog.FileLoadDialog.exec = lambda self: 1


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


def plain(widget) -> QImage:
    """A widget's picture in plain pixels (the screen's 150 % scaling
    would otherwise shift everything painted on it)."""
    image = widget.grab().toImage().convertToFormat(
        QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(1.0)
    return image


class Film:
    """The film: frames of the window, a painted pointer, captions."""

    def __init__(self, path: Path):
        self.path = path
        self.pos = QPointF(win.width() * 0.55, win.height() * 0.45)
        self.caption = ""
        self.trail: list = []          # a shape being drawn (window px)
        self.closed = False            # draw the trail closed (a box)
        self.ring = None               # (point, phase 0..1) of a click
        self.scale = WIDTH / win.width()
        height = int(win.height() * self.scale) // 2 * 2
        self.size = (WIDTH, height)
        self.frames = 0
        self.ffmpeg = subprocess.Popen(
            [str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgra", "-s", f"{WIDTH}x{height}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        self.base = None

    # --- pictures ----------------------------------------------------------------

    def grab(self):
        pump(0.05)
        image = plain(win)
        dialog = QApplication.activeModalWidget()
        if dialog is not None and dialog is not win and dialog.isVisible():
            offset = dialog.mapToGlobal(QPoint(0, 0)) \
                - win.mapToGlobal(QPoint(0, 0))
            dpr = image.width() / win.width()     # (screen scaling)
            painter = QPainter(image)
            painter.drawImage(QPoint(int(offset.x() * dpr),
                                     int(offset.y() * dpr)), plain(dialog))
            painter.end()
        image = image.scaled(self.size[0], self.size[1],
                             Qt.AspectRatioMode.IgnoreAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self.base = image.convertToFormat(QImage.Format.Format_ARGB32)

    def frame(self):
        image = self.base.copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.scale
        if len(self.trail) > 1:
            painter.setPen(QPen(QColor("#ffd700"), 2.5))
            shape = QPolygonF([QPointF(p.x() * s, p.y() * s)
                               for p in self.trail])
            if self.closed:
                painter.drawPolygon(shape)
            else:
                painter.drawPolyline(shape)
        if self.ring is not None:
            point, phase = self.ring
            painter.setPen(QPen(QColor(255, 255, 255, int(255 * (1 - phase))),
                                3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(point.x() * s, point.y() * s),
                                8 + 22 * phase, 8 + 22 * phase)
        x, y = self.pos.x() * s, self.pos.y() * s
        arrow = QPolygonF([QPointF(x + dx * 1.4, y + dy * 1.4) for dx, dy in
                           ((0, 0), (0, 17), (4.2, 13), (7.2, 20),
                            (10, 18.8), (7.2, 12.2), (12.4, 12.2))])
        painter.setPen(QPen(QColor("black"), 1.5))
        painter.setBrush(QBrush(QColor("white")))
        painter.drawPolygon(arrow)
        if self.caption:
            band, h = 54, image.height()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
            painter.drawRect(0, h - band, image.width(), band)
            font = QFont()
            font.setPixelSize(22)
            painter.setFont(font)
            painter.setPen(QColor("white"))
            painter.drawText(QRect(24, h - band, image.width() - 48, band),
                             Qt.AlignmentFlag.AlignVCenter, self.caption)
        painter.end()
        self.ffmpeg.stdin.write(image.constBits().asstring(
            image.sizeInBytes()))
        self.frames += 1

    # --- actions (seconds are at full speed; the film is SLOW times longer) --

    def say(self, caption):
        self.caption = caption
        print(f"{self.frames / FPS:6.1f} s  {caption}", flush=True)

    def hold(self, seconds, regrab=True):
        if regrab or self.base is None:
            self.grab()
        for _ in range(int(seconds * SLOW * FPS)):
            self.frame()

    def live(self, seconds, every=0.2):
        """Film the window while it changes by itself (a new picture every
        `every` seconds), e.g. a progress bar or data being read."""
        end = time.time() + seconds
        while time.time() < end:
            self.grab()
            for _ in range(max(1, int(every * SLOW * FPS))):
                self.frame()

    def move(self, point, seconds=None, regrab=True):
        """Glide the pointer to a point (window pixels), eased."""
        if regrab or self.base is None:
            self.grab()
        start = QPointF(self.pos)
        distance = math.hypot(point.x() - start.x(), point.y() - start.y())
        if seconds is None:
            seconds = min(0.25 + distance / 1400, 0.9)
        n = max(1, int(seconds * SLOW * FPS))
        for k in range(1, n + 1):
            f = k / n
            f = f * f * (3 - 2 * f)                     # ease in and out
            self.pos = QPointF(start.x() + (point.x() - start.x()) * f,
                               start.y() + (point.y() - start.y()) * f)
            if self.trail and not self.closed:
                self.trail[-1] = QPointF(self.pos)
            if self.closed:
                self.trail = rectangle(self.trail[0], self.pos)
            self.frame()

    def click(self, action=None, settle=0.3):
        for k in range(8):
            self.ring = (QPointF(self.pos), k / 8)
            self.frame()
        self.ring = None
        if action is not None:
            action()
        pump(settle)
        self.grab()
        for _ in range(4):
            self.frame()

    def press(self, widget, action=None, rel=(0.5, 0.5), settle=0.3):
        """Move to a widget (scrolled into view) and click it: its own
        click() unless an action is given."""
        show(widget)
        self.move(self.at(widget, rel))
        self.click(action or (lambda: widget.click()), settle)

    def choose(self, combo, data=None, index=None, settle=0.5):
        """Pick an entry of a list (by its data, or its position), as a
        person does."""
        if index is None:
            index = combo.findData(data)
        self.press(combo, action=lambda: (combo.setCurrentIndex(index),
                                          combo.activated.emit(index)),
                   settle=settle)

    def trace(self, points, seconds=1.6):
        """Draw a polygon: glide to its first point, then along it."""
        self.move(points[0])
        self.trail = [QPointF(points[0]), QPointF(points[0])]
        step = seconds / len(points)
        for p in points[1:] + points[:1]:
            self.move(p, step, regrab=False)
            self.trail.append(QPointF(p))
        self.click(None, settle=0.0)            # (double-click: finished)
        self.trail = []

    def drag_box(self, a, b, seconds=1.0):
        """Draw a rectangle: from corner a to corner b (window pixels)."""
        self.move(a)
        self.click(None, settle=0.0)
        self.closed = True
        self.trail = [QPointF(a)]
        self.move(b, seconds, regrab=False)
        self.click(None, settle=0.0)
        self.trail, self.closed = [], False

    # --- where things are ----------------------------------------------------------

    @staticmethod
    def at(widget, rel=(0.5, 0.5)) -> QPointF:
        local = QPoint(int(widget.width() * rel[0]),
                       int(widget.height() * rel[1]))
        point = widget.mapToGlobal(local) - win.mapToGlobal(QPoint(0, 0))
        return QPointF(point)

    @staticmethod
    def on_canvas(view_name, v, h) -> QPointF:
        """A point (µm) of a 2D view, in window pixels."""
        canvas = w.ortho._qt_viewers[view_name].canvas.native
        cam = ov.camera(w.ortho.views[view_name].model)
        cv, ch = list(cam.center)[-2:]
        x = (h - ch) * cam.zoom + canvas.width() / 2
        y = (v - cv) * cam.zoom + canvas.height() / 2
        point = canvas.mapToGlobal(QPoint(int(x), int(y))) \
            - win.mapToGlobal(QPoint(0, 0))
        return QPointF(point)

    def slider(self, view_name, axis, values, caption=None, per=0.5):
        """Drag a view's slider through these values, as a person does:
        the pointer takes the handle, moves it to each value, and the views
        change when it gets there (the user's remark of 2026-11-09: the
        views changed before the pointer had moved)."""
        if caption:
            self.say(caption)
        view = w.ortho.views[view_name]
        bar = w.ortho._qt_viewers[view_name].dims.slider_widgets[
            view.order.index(axis)].slider
        level0 = w.image.levels[0]
        i = w.image.axis_names.index(axis)

        def index_of(value):
            return round((value - level0.translation[i]) / level0.scale[i])

        def handle(index):
            """The handle's centre at an index, in window pixels."""
            span = max(bar.maximum() - bar.minimum(), 1)
            f = (index - bar.minimum()) / span
            try:
                from qtpy.QtWidgets import QStyle, QStyleOptionSlider
                option = QStyleOptionSlider()
                bar.initStyleOption(option)
                cc, sc = QStyle.ComplexControl, QStyle.SubControl
                groove = bar.style().subControlRect(
                    cc.CC_ScrollBar, option, sc.SC_ScrollBarGroove, bar)
                knob = bar.style().subControlRect(
                    cc.CC_ScrollBar, option, sc.SC_ScrollBarSlider, bar)
                x = groove.left() + f * (groove.width() - knob.width()) \
                    + knob.width() / 2
                local = QPoint(int(x), bar.height() // 2)
            except Exception:                      # (an estimate instead)
                local = QPoint(int(bar.width() * (0.04 + 0.92 * f)),
                               bar.height() // 2)
            return QPointF(bar.mapToGlobal(local)
                           - win.mapToGlobal(QPoint(0, 0)))
        self.move(handle(bar.value()))             # take the handle
        for k in range(4):
            self.ring = (QPointF(self.pos), k / 8)
            self.frame()
        self.ring = None
        for value in values:
            self.move(handle(index_of(value)), per * 0.35, regrab=False)
            w.ortho._set_cursor({axis: value})    # the handle and views follow
            pump(0.15)
            self.hold(per * 0.65)

    def zoom(self, view_name, factor, steps=8, settle=0.35):
        """Zoom a 2D view in (factor > 1) step by step, with the pointer
        on the canvas (the scroll wheel), filming what is read."""
        cam = ov.camera(w.ortho.views[view_name].model)
        start = cam.zoom
        for k in range(1, steps + 1):
            cam.zoom = start * factor ** (k / steps)
            self.live(settle, every=settle / 2)

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        print(f"wrote {self.path} ({self.frames / FPS:.0f} s)", flush=True)


def rectangle(a, b):
    return [QPointF(a.x(), a.y()), QPointF(b.x(), a.y()),
            QPointF(b.x(), b.y()), QPointF(a.x(), b.y())]


def show(widget):
    """Scroll the panel so that this widget is visible."""
    area = widget.parent()
    while area is not None and not isinstance(area, QScrollArea):
        area = area.parent()
    if area is not None:
        area.ensureWidgetVisible(widget, 20, 60)
        pump(0.1)


def button(text, root=None) -> QPushButton:
    return next(b for b in (root or w).findChildren(QPushButton)
                if b.text() == text and b.isVisible())


def open_image(film, path, caption):
    """Open… an image the way a person does: the button, the What to load
    dialog (filmed), Load."""
    film.say(caption)
    film.press(button("Open…"), action=lambda: None)

    def scene():
        dialog = QApplication.activeModalWidget()
        if dialog is None or type(dialog).__name__ != "LoadDialog":
            QTimer.singleShot(100, scene)
            return
        pump(0.4)
        film.say("What to load: channels, a region, the resolution "
                 "(here: everything)")
        film.hold(2.0)
        film.press(button("Load", dialog), action=dialog.accept)
    QTimer.singleShot(150, scene)
    w.open_image(path, ask=True)
    pump(2.5)
    w.path_edit.setText(Path(path).name)        # no local path in the film


def wait_busy(film, limit=180):
    """Film a crop or TIFF until it is done."""
    show(w.progress_bar)
    end = time.time() + limit
    while w._busy and time.time() < end:
        film.live(0.4)
    pump(0.5)


def start_with_output(film, button_widget, caption):
    """Crop selected / Save as TIFF… into a scratch folder (the field
    shows "crops", no local path)."""
    folder = OUT / "crops"
    show(button_widget)
    w.out_edit.setText("crops")
    w.overwrite_check.setChecked(True)
    film.say(caption)

    def start():
        w.out_edit.setText(str(folder))
        button_widget.click()
        w.out_edit.setText("crops")
    film.press(button_widget, action=start, settle=0.1)
    wait_busy(film)


viewer, w = crop_gui.open_window(show=True)
win = viewer.window._qt_window
win.resize(1700, 1050)
win.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
win.show()
pump(1.5)
w.napari_panels_check.setChecked(False)
film = Film(OUT / f"volume_cropper_{WHICH}.mp4")

if WHICH == "main":
    import score_demo_embryos as score
    name = "embryo_tissue_demo"
    film.say("Volume Cropper: cut regions out of large 3D images")
    film.hold(2.0)
    open_image(film, DEMO / f"{name}.zarr",
               "Open… an image (a synthetic one: embryos in tissue)")
    film.say("Four linked views: XY, YZ, XZ and 3D; each reads only what "
             "it shows")
    film.hold(2.0)
    z = w.ortho.views["XY"].order[-3]
    film.slider("XY", z, [400.0 + 8 * k for k in range(-4, 5, 2)],
                "Move through the slices", per=0.4)

    # lazy loading, and a fixed level for speed
    cam = ov.camera(w.ortho.views["XY"].model)
    film.say("Zoomed out, the views read a coarse level (see the view "
             "titles)")
    film.move(film.at(w.ortho._qt_viewers["XY"].canvas.native, (0.5, 0.5)))
    film.zoom("XY", 0.3, steps=4, settle=0.5)
    film.hold(1.0)
    film.say("Zoom in: finer levels are read, only for what is on screen")
    cam.center = (*list(cam.center)[:-2], 420.0, 230.0)
    film.move(film.on_canvas("XY", 420.0, 230.0))
    film.zoom("XY", 12.0, steps=10)
    film.hold(1.0)
    film.say("Choose the 2D resolution by hand: a coarse level reads "
             "little, so zooming is fast")
    film.choose(w.level_combo, 2, settle=1.5)
    film.move(film.on_canvas("XY", 420.0, 230.0))
    film.zoom("XY", 0.5, steps=4, settle=0.2)
    film.zoom("XY", 2.0, steps=4, settle=0.2)
    film.say("Back to Auto")
    film.choose(w.level_combo, index=0, settle=1.5)          # Auto
    w.ortho.reset_cameras()
    film.live(1.0)

    # projections
    film.say("2D projection: Max (MIP) through the whole depth")
    film.choose(w.projection_combo, "max", settle=2.0)
    film.hold(2.0)
    film.say("…or through a slab around the slice, which follows the "
             "slider")
    film.choose(w.through_combo, "slab", settle=1.5)
    film.slider("XY", z, [300.0, 360.0, 420.0, 480.0, 540.0], per=0.6)
    film.choose(w.projection_combo, "none", settle=1.5)

    # a crop box, saved as a TIFF
    film.say("Draw a crop box: a rectangle in XY (R) …")
    w.ortho._set_cursor({z: 350.0})
    pump(0.5)
    film.drag_box(film.on_canvas("XY", 715.0, 190.0),
                  film.on_canvas("XY", 950.0, 415.0))
    w._box_added({"y": (715.0, 950.0), "x": (190.0, 415.0)}, ("y", "x"),
                 None)
    film.hold(1.0)
    film.say("… and in XZ for its depth: a region (crop01) around the cyst")
    film.drag_box(film.on_canvas("XZ", 250.0, 190.0),
                  film.on_canvas("XZ", 455.0, 415.0))
    w._box_added({"z": (250.0, 455.0), "x": (190.0, 415.0)}, ("z", "x"),
                 None)
    film.hold(1.5)
    film.say("Save as TIFF…, for Fiji: here at level 1 (half the voxels "
             "along y and x)")
    show(w.tiff_level_combo)
    film.press(w.tiff_level_combo,
               action=lambda: w.tiff_level_combo.setCurrentIndex(1))
    start_with_output(film, w.tiff_button,
                      "Save as TIFF…: written, then checked")
    film.say("Done: a TIFF that Fiji opens calibrated")
    show(w.status)
    film.hold(2.0)

    # an outlined embryo
    truth = np.load(DEMO / f"{name}_truth.npz")
    outlines = score.drawn_outlines(truth["labels"] == 1,
                                    truth["voxel_um"].astype(float))
    film.say("New region: then draw around the embryo with polygons (P)")
    film.press(button("New region"))
    order = [(stored, key) for name_ in ("XY", "XZ", "YZ")
             for stored, keys in outlines.items()
             if set(stored) == set(w.ortho.views[name_].plane)
             for key in sorted(keys)]
    for n, (stored, key) in enumerate(order, start=1):
        view = w.ortho.view_for(stored)
        plane = view.plane                       # (vertical, horizontal)
        points = outlines[stored][key]
        if tuple(stored) != tuple(plane):
            points = [(b, a) for a, b in points]
        w.ortho.go_to_slice(plane, key)
        pump(0.3)
        film.say(f"Outline the embryo on a few slices of any view "
                 f"({n} of {len(order)}: {view.name})")
        every = max(1, len(points) // 24)
        path = [film.on_canvas(view.name, a, b) for a, b in points[::every]]
        film.trace(path, seconds=1.2)
        w._box_added({}, plane, [tuple(p) for p in points])
        film.hold(0.5)
    keys = sorted(next(k for p, k in outlines.items() if set(p) == {"y", "x"}))
    w.ortho.go_to_slice(("y", "x"), (keys[0] + keys[1]) / 2)
    film.say("One 3D shape through all polygons: dashed = its outline on "
             "this slice")
    film.hold(2.5)
    film.say("Shade selected region: orange is kept, blue is set to 0")
    film.press(w.shade_check)
    film.hold(2.0)
    film.press(w.shade_check)
    film.say("Preview crop: everything that is not in the crop is black")
    film.choose(w.preview_combo, "main")
    film.hold(2.0)
    film.choose(w.preview_combo, "off")
    film.say("3D view, clipped to the region, as it will be saved")
    film.press(w.show_3d_check, settle=3.0)
    film.press(w.clip_check, settle=1.5)
    film.choose(w.mask_combo, "hide", settle=1.5)
    canvas3d = w.ortho._qt_viewers["3D"].canvas.native
    camera = w.ortho.views["3D"].model.camera
    start = tuple(camera.angles)
    film.move(film.at(canvas3d, (0.35, 0.5)))
    for step in range(1, 21):
        camera.angles = (start[0], start[1] + 9 * step, start[2])
        pump(0.08)
        film.move(film.at(canvas3d, (0.35 + 0.015 * step, 0.5)), 0.12)
    start_with_output(film, w.crop_selected_button,
                      "Crop selected: written in the image's own format, "
                      "then checked against it")
    film.say("Done: the crop was checked against the source")
    show(w.status)
    film.hold(3.0)

elif WHICH == "timelapse":
    import make_demo_embryo_timelapse as lapse
    path = LAPSE_DIR / f"{lapse.NAME}.zarr"
    if not path.exists():
        LAPSE_DIR.mkdir(parents=True, exist_ok=True)
        lapse.make(LAPSE_DIR)
    labels = np.load(LAPSE_DIR / f"{lapse.NAME}_truth.npz")["labels"]
    voxel = lapse.VOXEL

    def inside(label, t=0):
        """A point (µm) inside an object at t, near its middle."""
        idx = np.argwhere(labels[t] == label)
        middle = idx.mean(0)
        best = idx[np.argmin(((idx - middle) ** 2).sum(1))]
        return tuple(float(v) for v in best * voxel)
    times = [t * lapse.INTERVAL for t in range(lapse.T)]
    film.say("A time-lapse (synthetic): an embryo's trunk and tail, and a "
             "gastruloid beside it")
    film.hold(2.0)
    open_image(film, path, "Open… the time-lapse")
    film.slider("XY", "t", times, "12 timepoints: the embryo drifts and "
                                  "turns, somites form, marker waves run "
                                  "through the PSM", per=0.6)
    film.slider("XY", "t", [times[-1], 0.0], per=0.3)
    find = w.sections["find"]
    show(find)
    film.say("Autocropper (experimental): fits regions to bright objects")
    film.move(film.at(find, (0.25, 0.0)) + QPointF(0, 12))
    film.click(lambda: find.set_open(True), settle=0.5)
    group = w.object_group
    show(group)
    film.move(film.at(group, (0.06, 0.0)) + QPointF(0, 10))
    film.click(lambda: group.setChecked(True))
    film.say("Moving region is ticked for time-lapses; tick Outline too")
    film.press(group.outline_check)
    show(group.timed_note)
    film.hold(2.0)
    film.say("Add seeds: click on the embryo, and on the gastruloid")
    film.press(group.seed_button)
    for label in (1, 2):
        zz, yy, xx = inside(label)
        w.ortho._set_cursor({"z": zz})
        pump(0.3)
        film.move(film.on_canvas("XY", yy, xx))
        film.click(lambda: group._seed_clicked(
            {"t": 0.0, "z": zz, "y": yy, "x": xx}, "XY"))
    show(group.seed_table)
    film.hold(2.0)
    film.say("Fit to seeds: a region per object; the embryo's moves with "
             "it, with its outline at every timepoint")
    film.press(group.fit_seeds_button, settle=1.5)
    show(group.seed_table)
    film.hold(2.5)
    embryo = next((k for k, b in enumerate(w.boxes) if b.track), 0)
    w._refresh(select=embryo)
    film.slider("XY", "t", times, "The region, its outline and the seed "
                                  "follow the embryo", per=0.8)
    film.say("3D view, clipped to the moving region")
    film.press(w.show_3d_check, settle=3.0)
    film.press(w.clip_check, settle=1.5)
    film.slider("XY", "t", times[::-1], per=0.5)
    start_with_output(film, w.crop_selected_button,
                      "Crop selected: each timepoint is read where the "
                      "region is then")
    film.say("Done: one time-lapse with the embryo in place, and a file "
             "with each timepoint's position")
    show(w.status)
    film.hold(3.0)

film.close()
if messages:
    print("messages:", messages)
w.boxes.clear()
pump(0.5)
QApplication.instance().closeAllWindows()
pump(1)
