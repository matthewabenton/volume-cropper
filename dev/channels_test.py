"""Open a large multichannel OME-Zarr with some channels shown: bytes read,
time, checkbox syncing (the image is only read).

Usage: python channels_test.py IMAGE.zarr 0      (or "all")
Tests the folder in the VC_TOOL environment variable (default: the tool folder this script is in).
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

PATH = sys.argv[1]
proc = psutil.Process()


def pump(s=0.5):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def settle(label, quiet=3.0, limit=300):
    """Pump until no more bytes are read for `quiet` seconds."""
    t0 = time.time()
    last, still = proc.io_counters().read_bytes, time.time()
    while time.time() - t0 < limit:
        pump(0.5)
        now = proc.io_counters().read_bytes
        if now != last:
            last, still = now, time.time()
        elif time.time() - still > quiet:
            break
    return time.time() - t0


channels = None if sys.argv[2] == "all" else [int(c) for c in sys.argv[2].split(",")]
start = proc.io_counters().read_bytes
t0 = time.time()
viewer, w = crop_gui.open_window(PATH, channels=channels, show=True)
viewer.window._qt_window.resize(1600, 1000)
settle("open")
print(f"channels={sys.argv[2]}: open + first display {time.time() - t0:.0f} s, "
      f"read {(proc.io_counters().read_bytes - start) / 1e6:.0f} MB, memory "
      f"{proc.memory_info().rss / 1e9:.1f} GB", flush=True)
print("show checks:", [c.isChecked() for c in w.show_checks],
      "| XY visible:", [l.visible for l in w.ortho.views["XY"].channel_layers],
      "| 3D visible:", [l.visible for l in w.ortho.views["3D"].channel_layers], flush=True)
if channels is not None:
    # Tick channel 1 in the panel: it appears in all views (and is read now).
    before = proc.io_counters().read_bytes
    t1 = time.time()
    w.show_checks[1].setChecked(True)
    settle("show 1")
    print(f"ticked channel 1: {time.time() - t1:.0f} s, read "
          f"{(proc.io_counters().read_bytes - before) / 1e6:.0f} MB; visible "
          f"XZ {[l.visible for l in w.ortho.views['XZ'].channel_layers]}", flush=True)
    # napari's eye icon (the XY layer) unticks the panel's checkbox.
    w.ortho.views["XY"].channel_layers[1].visible = False
    pump(0.5)
    print("eye icon off -> check:", w.show_checks[1].isChecked(), "| 3D visible:",
          [l.visible for l in w.ortho.views["3D"].channel_layers], flush=True)
viewer.close()
print("done", flush=True)
