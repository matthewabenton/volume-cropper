#!/usr/bin/env bash
# Start Volume Cropper (the napari GUI) on a Mac or on Linux (e.g. BARD).
#
#   bash volume_cropper.sh [IMAGE.zarr]
#
# The napari plugin (0.13.0):
#   bash volume_cropper.sh --install-plugin     add Volume Cropper to napari's menus
#   bash volume_cropper.sh --uninstall-plugin   remove it again
#   bash volume_cropper.sh --napari [FILES]     start napari itself (on BARD with
#                                               the OpenGL setup below)
#
# It finds conda in the usual places (or set VOLUME_CROPPER_CONDA to its folder),
# and uses the conda environment "volume_cropper" if it exists, otherwise
# "napari-env" (or set VOLUME_CROPPER_ENV to another environment name). This works
# even when conda is not set up in the shell's startup files. The plugin is only
# installed into "volume_cropper" (or VOLUME_CROPPER_ENV).

action=gui
case "${1:-}" in
    --install-plugin) action=install ;;
    --uninstall-plugin) action=uninstall ;;
    --napari) action=napari; shift ;;
esac

base=""
for candidate in "${VOLUME_CROPPER_CONDA:-}" "$HOME/miniforge3" "$HOME/mambaforge" \
                 "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/anaconda3" \
                 "/opt/miniforge3" "/opt/miniconda3" "/opt/anaconda3" \
                 "/opt/homebrew/Caskroom/miniforge/base" \
                 "/usr/local/Caskroom/miniforge/base" \
                 "$(conda info --base 2>/dev/null)"; do
    if [ -n "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        base="$candidate"
        break
    fi
done
if [ -z "$base" ]; then
    echo "volume_cropper: could not find conda. Install Miniforge, or set" \
         "VOLUME_CROPPER_CONDA=/path/to/miniforge3." >&2
    exit 1
fi
# conda's own shell setup (defines "conda activate"); not bin/activate, which
# newer Miniforge installs may hand over to mamba.
source "$base/etc/profile.d/conda.sh" || exit 1

env="${VOLUME_CROPPER_ENV:-}"
candidates="volume_cropper napari-env"
case "$action" in
    install|uninstall) candidates="volume_cropper" ;;   # only the tool's own
esac
if [ -z "$env" ]; then
    for name in $candidates; do
        if conda env list | awk '{print $1}' | grep -qx "$name"; then
            env="$name"
            break
        fi
    done
fi

# The tool folder: follow a link (e.g. one in a folder of your own) to the
# real file. (Plain readlink, because macOS has no "readlink -f".)
script="$0"
while [ -L "$script" ]; do
    link="$(readlink "$script")"
    case "$link" in
        /*) script="$link" ;;
        *) script="$(dirname "$script")/$link" ;;
    esac
done
here="$(cd "$(dirname "$script")" && pwd)"

if [ -z "$env" ] || ! conda activate "$env"; then
    echo "volume_cropper: no usable conda environment. Create it with:" >&2
    echo "  conda env create -f \"$here/env/environment.yml\"" >&2
    exit 1
fi

# Don't write Python cache files into the shared tool folder.
export PYTHONDONTWRITEBYTECODE=1

if [ "$action" = install ]; then
    echo "Installing the Volume Cropper napari plugin into the conda environment"
    echo "$env, from $here (nothing is written into that folder)."
    # pip needs setuptools; it comes with the environment (napari's Qt needs
    # it), but check, so that pip does not try to download it.
    if ! python -c "import setuptools" 2>/dev/null; then
        echo "The environment $env has no setuptools, which the install needs." >&2
        echo "Add it with:  conda install -n $env -c conda-forge setuptools" >&2
        exit 1
    fi
    # -e: "editable", the environment points to this folder (new releases
    # here reach the plugin without installing again). --no-build-isolation
    # and --no-index: use what the environment has, download nothing.
    python -m pip install --no-deps --no-build-isolation --no-index \
        --disable-pip-version-check -e "$here" || {
        echo "The install failed (see the messages above)." >&2
        exit 1
    }
    # napari started on its own must not write Python cache files into the
    # tool folder either.
    conda env config vars set PYTHONDONTWRITEBYTECODE=1 -n "$env" >/dev/null
    echo
    echo "Done. Start napari: \"conda activate $env\", then \"napari\" (on BARD:"
    echo "bash $here/volume_cropper.sh --napari). Then use Plugins > Volume"
    echo "Cropper, or Layers > Data > Crop selected layers. If $env was"
    echo "already active in your terminal, type \"conda activate $env\" once more."
    exit 0
fi
if [ "$action" = uninstall ]; then
    python -m pip uninstall -y --disable-pip-version-check napari-volume-cropper
    conda env config vars unset PYTHONDONTWRITEBYTECODE -n "$env" >/dev/null 2>&1
    echo
    echo "Done: Volume Cropper is no longer in napari's menus (after napari is"
    echo "started again). The tool itself still works as before."
    exit 0
fi

# What to start: the tool's window, or napari itself (--napari). The window
# is the package's __main__.py, run as a file: it finds the package itself,
# so the working folder stays the user's (relative paths work).
if [ "$action" = napari ]; then
    run=(-m napari)
else
    run=("$here/volume_cropper/__main__.py")
fi

# On a Mac, napari draws with OpenGL without any help.
if [ "$(uname)" = Darwin ]; then
    echo "volume_cropper: environment $env (Mac)"
    exec python "${run[@]}" "$@"
fi

# Some remote desktops set a locale that does not exist; C.UTF-8 always does.
export LC_ALL=C.UTF-8

# OpenGL on BARD (tested 2026-10-23 with dev/bard_gl_check.sh): Qt's
# default route (GLX) crashes with "Could not initialize GLX". Two routes work:
#   vgl: VirtualGL (vglrun) draws with the desktop's GPU - fastest.
#   egl: Qt uses EGL instead of GLX and draws on the CPU - works anywhere.
# The choice is automatic; force one with VOLUME_CROPPER_GL=vgl or VOLUME_CROPPER_GL=egl.
mode="${VOLUME_CROPPER_GL:-auto}"
if [ "$mode" = auto ]; then
    if command -v vglrun >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        mode=vgl
    else
        mode=egl
    fi
fi
echo "volume_cropper: environment $env, drawing with OpenGL via $mode"
if [ "$mode" = vgl ]; then
    exec vglrun python "${run[@]}" "$@"
else
    export QT_XCB_GL_INTEGRATION=xcb_egl
    exec python "${run[@]}" "$@"
fi
