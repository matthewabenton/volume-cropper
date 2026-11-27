"""Open a real image with one choice of what to load; time and bytes read.
Usage: python open_cost_real.py IMAGE CHOICE
Tests the folder in VC_TOOL (default: the tool folder this script is in).
CHOICE: old | whole | zthird | small"""
import faulthandler, os, sys, time
faulthandler.enable()
sys.path.insert(0, os.environ.get("VC_TOOL", os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, psutil
from qtpy.QtWidgets import QApplication
from volume_cropper import crop_gui, ortho_views as ov, core as vc

path, mode = sys.argv[1], sys.argv[2]
proc = psutil.Process()

def pump(s):
    end = time.time() + s
    while time.time() < end:
        QApplication.instance().processEvents(); time.sleep(0.02)

def measure(label, action, quiet=4.0, limit=500):
    b0, t0 = proc.io_counters().read_bytes, time.time()
    action()
    last, t_last = proc.io_counters().read_bytes, time.time()
    while time.time() - t0 < limit:
        pump(0.3)
        now = proc.io_counters().read_bytes
        if now != last: last, t_last = now, time.time()
        elif time.time() - t_last > quiet: break
    mb = (last - b0) / 1e6
    print(f"{mode:7s} {label:40s} {t_last - t0:6.1f} s {mb:7.1f} MB  ({mb / max(t_last - t0, 1e-9):.1f} MB/s)", flush=True)

viewer, w = crop_gui.open_window(None, show=True)
viewer.window._qt_window.resize(1600, 1000)
pump(1)
image = vc.open_image(path)
if mode == "old":                               # 0.4.0: whole image, 3D off
    w.show_3d_check.setChecked(False)
    measure("open (0.4.0, channel 0, 3D off)", lambda: w.open_image(path, [0]))
else:
    from volume_cropper.load_dialog import LoadChoice
    ranges = {}
    def third(axis):
        low, high = image.extent(axis); d = (high - low) / 3
        return (low + d, high - d)
    if mode in ("zthird", "small"):
        ranges["z"] = third("z")
    if mode == "small":
        ranges["y"], ranges["x"] = third("y"), third("x")
    choice = LoadChoice(channels=[0], ranges=ranges)
    region = ov.loaded_region(image, ranges)
    sizes = w.ortho.canvas_sizes()
    est = ov.open_read_bytes(image, region, 1, sizes)
    predicted = ov.fitted_levels(image, region, sizes)
    measure(f"open ({mode}, channel 0, 3D off)", lambda: w.open_image(path, choice=choice))
    print(f"        estimate {est / 1e6:.0f} MB uncompressed; levels predicted {predicted}, shown {w.ortho._levels_shown}", flush=True)
    print("        ", w.image_info.text().split("\n")[-1], "|", w.read_info.text(), flush=True)
o = w.ortho
y, x = o.cursor["y"], o.cursor["x"]
measure("cursor moved 100 µm (XZ and YZ reload)", lambda: o._set_cursor({"y": y + 100, "x": x + 100}))
measure("3D view on (auto level)", lambda: w.show_3d_check.setChecked(True))
print("        3D:", o.views["3D"].title.text(), flush=True)
viewer.close()
