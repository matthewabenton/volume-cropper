"""0.18.0 the Autocropper's own threshold and seed modes (the user's wish
of 2026-11-13: separate brighter parts of an embryo from fainter ones), in
a VISIBLE window on the synthetic embryo demo:

- Own threshold starts at the automatic value, greys Dim parts too and the
  seed menu; a dotted orange outline in the 2D views follows the slider,
  the slice and the timepoint; the value's line in the Levels histogram
  can be dragged with the mouse;
- Fit at a high and a low own threshold gives a smaller and a larger
  region; Fit to seeds with the three seed modes;
- switched off: outline and line gone, Dim parts too back.

Usage: python own_threshold_0_18_gui_test.py OUT_DIR

Saves window_auto.png, window_high.png, autocropper.png, levels.png and
prints the measured numbers. Synthetic data only; closes its window;
settings file of its own in OUT_DIR.
"""
import os
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
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import autocrop as ac  # noqa: E402
from volume_cropper import levels as lv  # noqa: E402
from volume_cropper import crop_gui  # noqa: E402
from volume_cropper import load_dialog  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = TOOL.parent / "volume_cropper_demo_data" / "embryo_tissue_demo.zarr"
results = []


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def size(ranges):
    return tuple(round(ranges[a][1] - ranges[a][0], 1) for a in "zyx")


def dots(view="XY"):
    layer = w.ortho.views[view].level_lines
    return len(layer.data) if layer.visible else 0


load_dialog.FileLoadDialog.exec = lambda self: 1
viewer, w = crop_gui.open_window()
viewer.window._qt_window.resize(1700, 1050)
pump(1.0)
w.open_image(DEMO, choice=LoadChoice())
pump(3.0)
for key in ("find", "levels"):
    w.sections[key].set_open(True)
group = w.object_group
group.setChecked(True)
group.channel_combo.setCurrentIndex(0)
w.levels.channel_combo.setCurrentIndex(0)
w.levels.source_combo.setCurrentIndex(1)            # whole image
pump(1.5)

# 1. on: the automatic value, Dim parts too off and greyed
start = time.time()
group.own_check.setChecked(True)
took = time.time() - start
pump(0.8)
volume = ac.search_volume(w.image, 0, time=w.ortho.time_shown())
auto = volume.automatic("otsu")
check("Own threshold starts at the automatic value",
      abs(group.own_threshold() - auto) <= 1.0,
      f"({group.own_threshold():.1f} vs {auto:.1f}; ticking took "
      f"{took:.2f} s)")
check("Dim parts too off and greyed, seed menu greyed",
      not group.grow_check.isChecked() and not group.grow_check.isEnabled()
      and not group.seed_mode_combo.isEnabled())
counts = {v: dots(v) for v in ("XY", "XZ", "YZ")}
check("a dotted outline in all three 2D views", all(counts.values()),
      f"{counts}")
viewer.screenshot(str(OUT / "window_auto.png"), canvas_only=False)
check("the Levels histogram marks the threshold",
      w.levels._marker_shown()
      and abs(w.levels.marker[1] - group.own_threshold()) < 1e-6)

# 2. the slider: a higher value, less above it
group.own_slider.setValue(int(group.own_slider.value() * 1.8) + 50)
pump(0.6)
high = group.own_threshold()
check("the outline follows the slider",
      dots() != counts["XY"], f"(value {high:.0f}: {dots()} dots, was "
                              f"{counts['XY']})")
viewer.screenshot(str(OUT / "window_high.png"), canvas_only=False)

# 3. the slice and the timepoint
z0 = w.ortho.cursor["z"]
before = w.ortho.views["XY"].level_lines.data.copy()
w.ortho._set_cursor({"z": z0 + 40.0})
pump(0.4)
after = w.ortho.views["XY"].level_lines.data
check("another slice: another outline",
      before.shape != after.shape or not np.allclose(before, after))
w.ortho._set_cursor({"z": z0})
t_axis = w.image.axis_names[0]
if t_axis == "t":
    times = w.image.levels[0].array.shape[0]
    t0 = w.ortho.cursor.get("t", 0.0)
    w.ortho._set_cursor({"t": t0 + w.image.levels[0].scale[0]})
    pump(0.05)
    empty_at_once = dots() == 0
    pump(0.8)
    check("another timepoint: read once it stays there (0.3 s), then "
          "outlined", empty_at_once and dots() > 0, f"({dots()} dots)")
    w.ortho._set_cursor({"t": t0})
    pump(0.8)

# 4. dragging the histogram's line with the mouse
levels = w.levels
levels.canvas.draw()
pump(0.3)
ax = levels.ax
value = levels.marker[1]
target = value * 0.6


def mouse(name, x_value, button=None):
    # matplotlib's own mouse events, as levels_0_18_gui_test.py (Qt test
    # drags do not reach matplotlib's canvas reliably)
    x_px = ax.transData.transform((x_value, 0))[0]
    y_px = ax.transAxes.transform((0, 0.5))[1]
    levels.canvas.callbacks.process(
        name, MouseEvent(name, levels.canvas, x_px, y_px, button=button))


mouse("button_press_event", value, button=1)
print(f"   (the press took line {levels._drag}: 3 = the threshold's; "
      f"the gamma line lay "
      f"{lv.midtone_value(levels.limits, levels.gamma):.0f})", flush=True)
for f in np.linspace(0, 1, 6)[1:]:
    mouse("motion_notify_event", value + f * (target - value))
    pump(0.08)
mouse("button_release_event", target, button=1)
pump(0.5)
check("dragging the histogram's line sets the threshold",
      abs(group.own_threshold() - target) < 0.05 * value,
      f"({value:.0f} -> {group.own_threshold():.0f}, aimed at "
      f"{target:.0f})")
levels.grab().save(str(OUT / "levels.png"))

# 5. Fit at a high and a low own threshold (no region selected: the
# loaded image; the object with most signal above it)
group.margin_spin.setValue(0.0)
found = {}
for name, value in (("high", high), ("low", 0.5 * auto)):
    group._set_own_value(value)
    w._refresh(select=-1)
    group.fit()
    pump(0.3)
    found[name] = size(w.boxes[-1].ranges)
    print(f"   Fit at {value:.0f}: region {w.boxes[-1].name}, size (z, y, "
          f"x) {found[name]} µm", flush=True)
check("Fit: a higher own threshold gives a smaller region",
      np.prod(found["high"]) < np.prod(found["low"]))
check("the status names the own threshold",
      "your own threshold" in group.info.text())
group.grab().save(str(OUT / "autocropper.png"))

# 6. off: outline and line gone, Dim parts too back
group.own_check.setChecked(False)
pump(0.4)
check("off: no outline, no histogram line, Dim parts too back",
      dots() == 0 and levels.marker is None
      and group.grow_check.isChecked() and group.grow_check.isEnabled()
      and group.seed_mode_combo.isEnabled())

# 7. the seed modes: a seed at the centre of the object found at the high
# threshold, Fit to seeds with each mode (Dim parts too off: the embryos
# lie in tissue, which growing would take in)
group.grow_check.setChecked(False)
centre = {a: sum(w.boxes[0].ranges[a]) / 2 for a in "zyx"}
if "t" in w.image.axis_names:
    centre["t"] = w.ortho.time_shown()
print(f"   seed at {centre}", flush=True)
for mode in ac.SEED_MODES:
    result = ac.find_object(w.image, 0, seed=centre, grow=False,
                            seed_mode=mode,
                            region={a: w.ortho.extent(a)
                                    for a in w.image.range_axes})
    print(f"   {mode}: threshold {result.thresholds[0]}", flush=True)
for mode in ac.SEED_MODES:
    group.seed_mode_combo.setCurrentIndex(group.seed_mode_combo.findData(mode))
    group.seeds = [dict(centre)]
    group.seed_results = [None]
    group.refresh_seeds()
    n = len(w.boxes)
    group.fit_to_seeds()
    pump(0.3)
    made = len(w.boxes) > n
    print(f"   seed mode {mode}: "
          + (f"region size {size(w.boxes[-1].ranges)} µm" if made
             else "no region"), flush=True)
    results.append(made)
viewer.close()
pump(0.5)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
