"""Finding the tool's own modules (crop_gui, volume_cropper, ...) from the
plugin.

Only this small package is installed into the environment (pyproject.toml);
the tool's modules stay in the tool folder, one level up, and are imported
from there, as volume_cropper.bat/.sh and the notebook do. So a release
(a "git pull" of the shared folder) reaches the plugin without a reinstall,
even when it adds a module, and the tool's module names are not importable
elsewhere in the environment until the plugin is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The tool folder, as the plugin was installed from it (e.g. on a network
# drive connected as a drive letter). Not resolved: on Windows that would
# turn a drive letter into a network (UNC) path.
TOOL_DIR = Path(__file__).absolute().parent.parent


# The tool's modules: all must come from TOOL_DIR (a mix of two copies of
# the tool, e.g. a local copy and the shared one, would mix two versions).
TOOL_MODULES = ("volume_cropper.crop_gui", "volume_cropper.core",
                "volume_cropper.file_images", "volume_cropper.ortho_views",
                "volume_cropper.load_dialog", "volume_cropper.autocrop",
                "volume_cropper.autocrop_panel", "volume_cropper.settings")


def import_tool():
    """Import and return crop_gui from the tool folder.

    Python is told not to write cache files (__pycache__) from now on: the
    tool folder is usually the shared copy, which the launchers keep free
    of them too (PYTHONDONTWRITEBYTECODE). When the window itself was
    started from this folder (the launchers, or python -m volume_cropper),
    that running copy is used, so that its windows are known.
    Raises RuntimeError if a tool module from another copy of the tool was
    imported earlier in this napari session.
    """
    sys.dont_write_bytecode = True
    folder = str(TOOL_DIR)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from volume_cropper import crop_gui

    for name in TOOL_MODULES:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        # The modules are in the package folder volume_cropper/, which is
        # in the tool folder: two levels up from each module file.
        if path and not _same_folder(Path(path).absolute().parent.parent,
                                     TOOL_DIR):
            raise RuntimeError(
                f"Another copy of Volume Cropper is loaded in this napari "
                f"session ({Path(path).parent.parent}), not the one the "
                f"plugin was installed from ({TOOL_DIR}). Please close napari "
                "and start it again (from the same copy).")
    return crop_gui


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def version() -> str:
    """The tool's version, read from volume_cropper/core.py without importing
    the tool (the installed package's own version number is the one at
    install time)."""
    import re
    try:
        text = (TOOL_DIR / "volume_cropper" / "core.py").read_text(
            encoding="utf-8")
        found = re.search(r'^__version__ = "([^"]+)"', text, re.MULTILINE)
        return found.group(1) if found else "?"
    except OSError:
        return "?"
