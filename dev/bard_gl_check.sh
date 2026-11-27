#!/usr/bin/env bash
# Collect facts about OpenGL on this BARD desktop and try napari with a few
# graphics settings. Prints a short report to copy back.
#
#   bash bard_gl_check.sh     (environment: VC_ENV, default volume_cropper;
#                              conda: CONDA_ROOT, default ~/miniforge3)
#
# Each napari test opens a napari window for about 3 seconds, then closes it.

ulimit -c 0                  # no core-dump files if a test crashes
export LC_ALL=C.UTF-8        # a locale that always exists
source "${CONDA_ROOT:-$HOME/miniforge3}/bin/activate" "${VC_ENV:-volume_cropper}" || exit 1

echo "== desktop"
echo "DISPLAY=$DISPLAY session=${XDG_SESSION_TYPE:-?} wayland=${WAYLAND_DISPLAY:-none}"
echo "/dev/dri: $(ls /dev/dri 2>/dev/null | tr '\n' ' ')| nvidia device files: $(ls /dev/nvidia* 2>/dev/null | wc -l)"
command -v nvidia-smi >/dev/null && nvidia-smi -L 2>&1 | head -2
echo "vglrun: $(command -v vglrun || echo 'not found')"
echo "glxinfo: $(command -v glxinfo || echo 'not found')"

echo "== X server extensions"
python - <<'EOF'
import ctypes
x11 = ctypes.CDLL("libX11.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XListExtensions.restype = ctypes.POINTER(ctypes.c_char_p)
display = x11.XOpenDisplay(None)
if not display:
    print("cannot open the X display")
else:
    n = ctypes.c_int()
    names = x11.XListExtensions(ctypes.c_void_p(display), ctypes.byref(n))
    extensions = sorted(names[i].decode() for i in range(n.value))
    print("GLX extension:", "yes" if "GLX" in extensions else "NO")
    print("all:", " ".join(extensions))
EOF

echo "== system OpenGL libraries"
ls /usr/lib64/libGLX_* /usr/lib/x86_64-linux-gnu/libGLX_* \
   /usr/lib64/libEGL_* /usr/lib/x86_64-linux-gnu/libEGL_* 2>/dev/null
ls /usr/share/glvnd/egl_vendor.d 2>/dev/null | sed 's/^/EGL vendor file: /'
for f in $(grep -l -i napari /usr/share/applications/*.desktop 2>/dev/null); do
    echo "BARD napari app: $f"
    grep -i '^Exec' "$f"
done
if command -v glxinfo >/dev/null; then
    echo "== glxinfo -B"
    glxinfo -B 2>&1 | head -12
fi

echo "== napari tests"
# Usage: try_napari LABEL PREFIX [VAR=value ...]
# PREFIX is a command to wrap python in (e.g. vglrun), or "" for none.
try_napari() {
    local label=$1 prefix=$2
    shift 2
    local output status
    output=$(env "$@" QT_LOGGING_RULES="qt.qpa.gl=true" timeout 90 $prefix python -c '
import napari
from qtpy.QtCore import QTimer
viewer = napari.Viewer()
QTimer.singleShot(3000, viewer.close)
napari.run()
print("OPENED OK")
' 2>&1)
    status=$?
    echo "-- $label: exit code $status"
    echo "$output" | grep -iE "opened ok|glx|egl|opengl|renderer|vendor|error|abort" \
        | grep -viE "locale" | head -8 | sed 's/^/     /'
}
try_napari "1 default" ""
try_napari "2 software GLX" "" LIBGL_ALWAYS_SOFTWARE=1
try_napari "3 EGL" "" QT_XCB_GL_INTEGRATION=xcb_egl
try_napari "4 EGL + software" "" QT_XCB_GL_INTEGRATION=xcb_egl \
    LIBGL_ALWAYS_SOFTWARE=1 __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
try_napari "5 Mesa GLX forced" "" __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1
if command -v vglrun >/dev/null; then
    try_napari "6 vglrun (GPU)" vglrun
fi
echo "== done"
