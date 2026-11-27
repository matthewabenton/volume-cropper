"""0.22.0 Levels "zoom to min–max", in a VISIBLE window, on the synthetic
embryo demo.

Usage: python levels_zoom_0_22_gui_test.py OUT_DIR

Opens volume_cropper_demo_data/embryo_tissue_demo.zarr (synthetic, read
only). Checks, in a real window: the tick box sits in the Levels row
without squeezing it (also in the Settings dialog); with min and max set to
a narrow range, ticking it spreads min to max over the axis with 25 % room
on either side, in finer bins, while the numbers below still count all
voxels; a drag of the upper line in steps, beyond the right edge, leaves
the axis where it is until the mouse is released (the line stops at the
edge), then the axis follows; a second drag goes further (the "let go and
drag again" of the guide); a typed min and auto-contrast "once" move the
axis too; on "whole image" (up to 4 million voxels) the axis follows in a
time that is printed; unticking gives the whole axis back. Saves
levels_zoom_off.png, levels_zoom_on.png, levels_zoom_dragged.png (the
Levels part) and levels_zoom_window.png. Synthetic data only. Answers every
dialog itself and closes its window. Uses a settings file of its own in
OUT_DIR.
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
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import levels as lv  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
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


def wait_for(condition, seconds=10.0):
    end = time.time() + seconds
    while time.time() < end:
        pump(0.1)
        if condition():
            return True
    return False


def mouse(levels, name, x, y, button=None):
    levels.canvas.callbacks.process(
        name, MouseEvent(name, levels.canvas, x, y, button=button))


def expected_axis(levels):
    hist = levels.hist
    return lv.axis_range((hist.minimum, hist.maximum), levels.limits,
                         lv.ZOOM_MARGIN)


def drag_upper_beyond_edge(levels, label):
    """Drag the upper line in steps to 60 px beyond the right edge; return
    (axis while dragging stayed put, the axis's right edge before, limit at
    release)."""
    levels.canvas.draw()
    py = levels.ax.transAxes.transform((0.5, 0.5))[1]
    px = levels.ax.transData.transform((levels.limits[1], 0))[0]
    right = levels.ax.transAxes.transform((1.0, 0.5))[0]
    before = levels._axis
    edges = levels.hist.edges.copy()
    mouse(levels, "button_press_event", px, py, button=1)
    still = levels._drag == 1
    for x in np.linspace(px, right + 60, 10):
        mouse(levels, "motion_notify_event", x, py)
        pump(0.05)                  # the views follow every 40 ms meanwhile
        still = still and levels._axis == before \
            and np.array_equal(levels.hist.edges, edges)
    at_release = levels.limits[1]
    mouse(levels, "button_release_event", right + 60, py, button=1)
    pump(0.6)
    print(f"  {label}: axis {before[0]:.0f}-{before[1]:.0f} -> "
          f"{levels._axis[0]:.0f}-{levels._axis[1]:.0f}; max at release "
          f"{at_release:.0f}", flush=True)
    return still, float(edges[-1]), at_release


load_dialog.FileLoadDialog.exec = lambda self: 1        # accept as it starts
for name in ("information", "warning", "critical"):
    setattr(crop_gui.QMessageBox, name, staticmethod(
        lambda *a, **k: messages.append(a[1:3])))
crop_gui._confirm = lambda *a: True

viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
w.open_image(DEMO, choice=LoadChoice())
levels = w.levels
w.sections["levels"].set_open(True)
check("the histogram appears by itself after opening (XY view)",
      wait_for(lambda: levels.hist is not None), levels.stats.text())
xy = w.ortho.views["XY"]
k = levels.channel
layer = xy.channel_layers[k]

# --- the tick box in its row ---------------------------------------------------
pump(0.5)
box = levels.zoom_check
row_ok = (box.isVisible() and box.geometry().right() <= levels.width()
          and levels.source_combo.width()
          >= levels.source_combo.minimumSizeHint().width())
check("the tick box fits in the Levels row (nothing squeezed)", row_ok,
      f"box right edge {box.geometry().right()} of {levels.width()} px; "
      f"source list {levels.source_combo.width()} px (needs "
      f"{levels.source_combo.minimumSizeHint().width()})")
check("off at first", not box.isChecked())

# --- a narrow range of min to max: zoom on --------------------------------------
data = (levels.hist.minimum, levels.hist.maximum)
span = data[1] - data[0]
low, high = round(data[0] + 0.05 * span), round(data[0] + 0.2 * span)
levels.max_spin.setValue(high)
levels.min_spin.setValue(low)
pump(0.6)
check("min and max set to a narrow range",
      tuple(layer.contrast_limits) == (low, high),
      f"{tuple(layer.contrast_limits)}; data {data[0]:.0f}-{data[1]:.0f}")
w.sections["levels"].grab().save(str(OUT / "levels_zoom_off.png"))
whole_width = levels.hist.edges[1] - levels.hist.edges[0]
n = levels.hist.n
box.setChecked(True)
pump(0.6)
room = lv.ZOOM_MARGIN * (high - low)
wanted = (max(low - room, data[0]), min(high + room, data[1]))
width = levels.hist.edges[1] - levels.hist.edges[0]
check("ticked: min to max with 25 % room on either side, finer bins",
      levels._axis == wanted and width < whole_width
      and levels.ax.get_xlim() == (levels.hist.edges[0],
                                   levels.hist.edges[-1]),
      f"axis {levels._axis[0]:.0f}-{levels._axis[1]:.0f} (wanted "
      f"{wanted[0]:.0f}-{wanted[1]:.0f}); bin width {whole_width:g} -> "
      f"{width:g}")
check("the numbers still count all voxels",
      levels.hist.n == n and levels.hist.counts.sum() <= n
      and f"{n:,} voxels" in levels.stats.text(),
      f"{levels.hist.counts.sum():,} of {n:,} on the axis")
w.sections["levels"].grab().save(str(OUT / "levels_zoom_on.png"))

# --- drag beyond the edge, twice -------------------------------------------------
still, edge, at_release = drag_upper_beyond_edge(levels, "first drag")
check("while dragging, the axis stays put; the line stops at the edge",
      still and abs(at_release - edge) <= 0.5,
      f"max at release {at_release:.0f}, edge {edge:.1f}")
high1 = float(layer.contrast_limits[1])
check("after release the axis follows, with new room beyond the max",
      levels._axis == expected_axis(levels) and levels.hist.edges[-1] > edge
      and high1 == levels.limits[1]
      and all(w.ortho.views[v].channel_layers[k].contrast_limits[1] == high1
              for v in ("XZ", "YZ")),
      f"max {high1:.0f} in XY, XZ, YZ")
w.sections["levels"].grab().save(str(OUT / "levels_zoom_dragged.png"))
still, edge, at_release = drag_upper_beyond_edge(levels, "second drag")
high2 = float(layer.contrast_limits[1])
check("a second drag goes further", still and high2 > high1,
      f"max {high1:.0f} -> {high2:.0f}")

# --- typed min, auto-contrast -----------------------------------------------------
new_low = round(low + 0.3 * (high2 - low))
levels.min_spin.setValue(new_low)
pump(0.6)
check("a typed min: the axis follows at once",
      levels.limits[0] == new_low and levels._axis == expected_axis(levels),
      f"axis {levels._axis[0]:.0f}-{levels._axis[1]:.0f}")
levels.auto_once.click()
pump(1.0)
check("auto-contrast once: the axis follows",
      levels.limits == [float(v) for v in layer.contrast_limits]
      and levels._axis == expected_axis(levels),
      f"limits {levels.limits}, axis {levels._axis[0]:.0f}-"
      f"{levels._axis[1]:.0f}")

# --- whole image: the time the axis takes to follow -------------------------------
levels.source_combo.setCurrentIndex(1)
levels.recompute()
pump(0.5)
lo_w, hi_w = (float(v) for v in layer.contrast_limits)
t0 = time.perf_counter()
levels.max_spin.setValue(round(hi_w * 0.8))
followed = time.perf_counter() - t0
pump(0.5)
check("whole image: the axis follows a typed max",
      levels._axis == expected_axis(levels),
      f"{levels.hist.n:,} voxels; following took {followed:.2f} s "
      "(rebinning; the canvas draws when idle)")
viewer.screenshot(str(OUT / "levels_zoom_window.png"), canvas_only=False)

# --- untick ------------------------------------------------------------------------
box.setChecked(False)
pump(0.6)
check("unticked: the whole axis again",
      levels.hist.edges[0] <= levels.hist.minimum
      and levels.hist.edges[-1] >= levels.hist.maximum
      and levels.hist.counts.sum() == levels.hist.n)

# --- the Settings dialog: its row ----------------------------------------------------
dialog = crop_gui.SettingsDialog(w)
dialog.show()
pump(0.5)
zoom = dialog.view_widgets["levels_zoom"]
check("Settings: the start value's tick box is shown, unticked",
      zoom.isVisible() and not zoom.isChecked()
      and zoom.geometry().right() <= zoom.parentWidget().width())
dialog.grab().save(str(OUT / "settings_dialog.png"))
dialog.close()

print("dialogs answered:", messages)
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
