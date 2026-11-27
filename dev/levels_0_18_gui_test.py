"""0.18.0 Levels histogram, in a VISIBLE window, on the synthetic embryo demo.

Usage: python levels_0_18_gui_test.py OUT_DIR

Opens volume_cropper_demo_data/embryo_tissue_demo.zarr (synthetic, read
only). Checks, in a real window (where the panel computes by itself, unlike
the hidden-window tests): the histogram appears after opening; it follows
the XY slider (another z gives other numbers); dragging the upper contrast
line sets that limit in the XY, XZ and YZ views; the pointer readout names
a bin; "log" switches the count axis to a log scale; "whole image" and
"selected crop region" give numbers (the region's voxel count equals its
box at the level read, as in test_levels); a closed Levels part does not
recompute until it is opened again. Prints how long each recompute took.
Gamma (added the same day): dragging the dashed line left sets gamma below
1 in the XY, XZ and YZ views and brightens the XY view on screen; typed
gamma and max apply to all views; a double-click on the line sets gamma 1.
Saves levels_view.png and levels_gamma_window.png (the window),
levels_panel_view.png, levels_panel_log.png, levels_panel_gamma_drag.png,
levels_panel_gamma.png, levels_panel_region.png (the Levels part).
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
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import levels as lv  # noqa: E402
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


def timed_recompute(levels, label):
    t0 = time.perf_counter()
    levels.recompute()
    print(f"  recompute ({label}): {time.perf_counter() - t0:.2f} s, "
          f"{levels.hist.n if levels.hist else 0:,} voxels", flush=True)


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
print(levels.stats.text().replace("\n", " | "))

# --- it follows the XY slider ------------------------------------------------
xy = w.ortho.views["XY"]
z = xy.order.index("z")
steps = viewer.dims.nsteps[z]
before = (levels.hist.mean, levels.hist.n)
viewer.dims.set_current_step(z, int(steps * 0.25))
changed = wait_for(lambda: levels.hist is not None
                   and (levels.hist.mean, levels.hist.n) != before)
check("another z slice gives other numbers", changed,
      f"mean {before[0]:.1f} -> {levels.hist.mean:.1f}")
viewer.dims.set_current_step(z, int(steps * 0.5))
wait_for(lambda: False, 1.5)
timed_recompute(levels, "XY view")

# --- drag the upper line -------------------------------------------------------
levels.canvas.draw()
high_before = levels.limits[1]
px = levels.ax.transData.transform((high_before, 0))[0]
py = levels.ax.transAxes.transform((0.5, 0.5))[1]
left = levels.ax.transAxes.transform((0.0, 0.5))[0]
target = px - 0.3 * (px - left)
mouse(levels, "button_press_event", px, py, button=1)
for x in np.linspace(px, target, 8):                 # a drag in steps
    mouse(levels, "motion_notify_event", x, py)
    pump(0.03)
mouse(levels, "button_release_event", target, py, button=1)
pump(0.5)
k = levels.channel
high = xy.channel_layers[k].contrast_limits[1]
others = [w.ortho.views[n].channel_layers[k].contrast_limits[1]
          for n in ("XZ", "YZ")]
check("dragging the upper line lowers that limit in XY, XZ and YZ",
      high < high_before and all(o == high for o in others),
      f"{high_before:.0f} -> {high:.0f} (XZ, YZ: {others})")

# --- the pointer readout ----------------------------------------------------------
bin_x = levels.ax.transData.transform((levels.hist.mean, 0))[0]
mouse(levels, "motion_notify_event", bin_x, py)
check("the pointer readout names a bin", "voxels" in levels.readout.text(),
      levels.readout.text())
pump(0.5)
viewer.screenshot(str(OUT / "levels_view.png"), canvas_only=False)
w.sections["levels"].grab().save(str(OUT / "levels_panel_view.png"))

# --- log ---------------------------------------------------------------------------
levels.log_check.setChecked(True)
pump(0.5)
check("log: the count axis is logarithmic", levels.ax.get_yscale() == "log")
w.sections["levels"].grab().save(str(OUT / "levels_panel_log.png"))
levels.log_check.setChecked(False)
pump(0.3)

# --- gamma: the dashed line, the boxes ------------------------------------------
layer = xy.channel_layers[k]
views = [w.ortho.views[n].channel_layers[k] for n in ("XZ", "YZ")]
levels.canvas.draw()
bright_before = viewer.screenshot(canvas_only=True)[..., :3].mean()
low, high = levels.limits
mx = levels.ax.transData.transform((lv.midtone_value((low, high),
                                                     levels.gamma), 0))[0]
lx = levels.ax.transData.transform((low, 0))[0]
target = mx - 0.6 * (mx - lx)
mouse(levels, "button_press_event", mx, py, button=1)
for x in np.linspace(mx, target, 8):
    mouse(levels, "motion_notify_event", x, py)
    pump(0.03)
dragging_text = levels.readout.text()
mouse(levels, "button_release_event", target, py, button=1)
pump(0.8)
gamma = layer.gamma
bright_after = viewer.screenshot(canvas_only=True)[..., :3].mean()
check("dragging the dashed line left: gamma below 1 in XY, XZ and YZ",
      gamma < 1 and all(v.gamma == gamma for v in views)
      and levels.gamma_spin.value() == round(gamma, 2),
      f"gamma {gamma:.2f}; readout while dragging: {dragging_text!r}")
check("gamma below 1 brightens the XY view", bright_after > bright_before,
      f"mean screen brightness {bright_before:.1f} -> {bright_after:.1f}")
w.sections["levels"].grab().save(str(OUT / "levels_panel_gamma_drag.png"))
levels.gamma_spin.setValue(0.6)
new_max = round(high * 0.8)
levels.max_spin.setValue(new_max)
pump(0.8)
check("typed: gamma 0.6 and max in all views",
      layer.gamma == 0.6 and all(v.gamma == 0.6 for v in views)
      and layer.contrast_limits[1] == new_max
      and all(v.contrast_limits[1] == new_max for v in views),
      f"gamma {layer.gamma}, max {layer.contrast_limits[1]}")
viewer.screenshot(str(OUT / "levels_gamma_window.png"), canvas_only=False)
w.sections["levels"].grab().save(str(OUT / "levels_panel_gamma.png"))
mx = levels.ax.transData.transform((lv.midtone_value(levels.limits, 0.6),
                                    0))[0]
levels.canvas.callbacks.process("button_press_event", MouseEvent(
    "button_press_event", levels.canvas, mx, py, button=1, dblclick=True))
pump(0.5)
check("double-click on the dashed line: gamma 1",
      layer.gamma == 1.0 and levels.gamma_spin.value() == 1.0)

# --- whole image, crop region --------------------------------------------------
levels.source_combo.setCurrentIndex(1)
timed_recompute(levels, "whole image")
check("whole image gives numbers", levels.hist is not None
      and "whole image" in levels.stats.text(), levels.stats.text()
      .replace("\n", " | "))
extent = {a: w.ortho.extent(a) for a in ("z", "y", "x")}
ranges = {a: (lo + 0.3 * (hi - lo), lo + 0.6 * (hi - lo))
          for a, (lo, hi) in extent.items()}
i = w._new_box(ranges)
w._refresh(select=i)
levels.source_combo.setCurrentIndex(2)
timed_recompute(levels, "crop region")
as_cropped = w.ortho.box_ranges[i]
level = int(levels.what.split("level ")[1].rstrip(")"))
lev = w.image.levels[level]
c = w.image.channel_axis
spatial = [j for j in range(len(w.image.axes)) if j != c]
box = ov._box_slices(["z", "y", "x"], [lev.scale[j] for j in spatial],
                     [lev.translation[j] for j in spatial],
                     [lev.array.shape[j] for j in spatial], as_cropped)
expected = int(np.prod([s.stop - s.start for s in box]))
check("crop region: its voxel count is that of its box",
      levels.hist is not None and levels.hist.n == expected,
      f"{levels.hist.n if levels.hist else 0} vs {expected} (level {level})")
pump(0.5)
w.sections["levels"].grab().save(str(OUT / "levels_panel_region.png"))

# --- a closed part waits --------------------------------------------------------
levels.source_combo.setCurrentIndex(0)
pump(0.5)
w.sections["levels"].set_open(False)
pump(0.3)
levels.hist = None
viewer.dims.set_current_step(z, int(steps * 0.6))
pump(1.5)
check("closed: no recompute", levels.hist is None)
w.sections["levels"].set_open(True)
check("opened again: recomputed", wait_for(lambda: levels.hist is not None))

print("dialogs answered:", messages)
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
