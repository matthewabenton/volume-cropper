"""The memory cache controls, in a VISIBLE window (demo image, local disk).

Usage: python cache_gui_test.py
Tests the folder in the VC_TOOL environment variable (default: the tool folder this script is in).
Changes napari's saved cache setting during the test and puts the user's
value back at the end.
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, os.environ.get(
    "VC_TOOL", str(Path(__file__).resolve().parents[1])))
import psutil  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from volume_cropper import crop_gui  # noqa: E402
from volume_cropper.load_dialog import LoadChoice  # noqa: E402

DEMO = str(Path(os.environ.get("VC_DEMO", Path(__file__).resolve().parents[2] / "volume_cropper_demo_data"))
           / "touching_tails_demo.zarr")
proc = psutil.Process()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def read_during(action, seconds=2.0):
    before = proc.io_counters().read_bytes
    action()
    pump(seconds)
    return (proc.io_counters().read_bytes - before) / 1e6


settings = crop_gui._dask_settings()
saved = (settings.cache, settings.enabled)        # the user's own values
viewer, w = crop_gui.open_window(None, show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
try:
    # 1. A small cache: it never holds more than its size.
    w.cache_spin.setValue(0.1)               # the field has one decimal
    pump(0.5)
    w.open_image(DEMO, choice=LoadChoice(show_3d=True))   # 102 MB image
    pump(4)
    used = crop_gui._dask_cache().total_bytes
    check("small cache stays within 100 MB", used <= 100e6,
          f"{used / 1e6:.0f} MB")
    check("napari's setting follows", abs(settings.cache - 0.1) < 1e-9
          and settings.enabled)
    # 2. A large cache: showing a channel again reads nothing.
    w.cache_spin.setValue(2.0)
    pump(0.5)
    read_during(lambda: w.show_checks[1].setChecked(False))
    read_during(lambda: w.show_checks[1].setChecked(True))      # fills it
    read_during(lambda: w.show_checks[1].setChecked(False))
    mb = read_during(lambda: w.show_checks[1].setChecked(True))
    check("show again with a warm cache reads nothing", mb < 0.5,
          f"{mb:.1f} MB")
    pump(2.5)
    print("   panel:", w.cache_info.text(), flush=True)
    check("panel shows the amount used", w.cache_info.text().endswith("used")
          and not w.cache_info.text().startswith("0 B"))
    # 3. Clear: empty, and showing the channel again reads it again.
    w._clear_cache()
    check("Clear empties the cache", crop_gui._dask_cache().total_bytes == 0)
    read_during(lambda: w.show_checks[1].setChecked(False))
    mb = read_during(lambda: w.show_checks[1].setChecked(True))
    check("after Clear, showing the channel reads it again", mb > 0.5,
          f"{mb:.1f} MB")
    # 4. 0 switches the cache off.
    w.cache_spin.setValue(0.0)
    pump(0.5)
    check("0 GB switches it off", not settings.enabled
          and crop_gui._dask_cache().available_bytes == 0)
finally:
    settings.cache, settings.enabled = saved          # the user's values back
    viewer.close()
print(f"restored the setting: {settings.cache:.2f} GB, enabled "
      f"{settings.enabled}", flush=True)
print(f"{sum(results)}/{len(results)} checks passed", flush=True)
