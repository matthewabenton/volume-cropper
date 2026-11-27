"""The user's own settings, remembered between sessions (0.16.0, the
window's "Settings…").

They are kept per user, in ~/.volume_cropper/settings.json (on Windows
C:\\Users\\<you>\\.volume_cropper\\settings.json; on BARD and the cluster
your home folder), never in the tool folder, which is shared. The
environment variable VOLUME_CROPPER_SETTINGS can name another file (the
tests use it). A missing or unreadable file gives the defaults; nothing
else breaks.

What is kept (the user's choice, 2026-11-08):
- compress_tiffs:   "Compress TIFFs (zstd)" ticked at first
- link_views:       "Link zoom/pan of the 2D views" ticked at first
- exact_level:      "exact level" ticked at first
- napari_layer_panels: napari's layer list and controls shown (0.18.0: off
                    at first, as the panel has the controls needed; the
                    key was napari_panels, on at first, until 0.17)
- show_tool_layers: Volume Cropper's own helper layers listed in napari's
                    layer list (off: only the image channels and the layers
                    you draw on are listed)
- sections:         which parts of the panel are open
- autocropper:      (0.18.0, the user's wish) the Autocropper's options a
                    new window starts with: threshold method, Dim parts
                    too, margin (µm), Moving region, Outline and its kind,
                    what a seed finds, Split an object between its seeds
- view:             (0.18.0) the View part's: slice planes, blacked-out
                    voxels (3D), outlines in 3D and their opacity,
                    crosshairs, crop regions, × marks (0.20.0), preview
                    crop, and the Levels histogram's source and log
                    (and, 0.22.0, "zoom to min–max")
- opening:          (0.18.0, the user's wish) what opening an image starts
                    with: the 2D resolution (Auto, or the finest level whose
                    voxels are at least voxel_2d µm: the user's choice, so
                    that images from other set-ups get the same detail), the
                    3D view (off, Auto, or by voxel size, voxel_3d), the
                    channels shown (all or the first) and whether "What to
                    load" asks (off: the image opens at once with these;
                    a 3D volume larger than the window's limit still asks)
The memory cache is napari's own setting (napari remembers it); the
dialog only shows and changes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULTS = {
    "compress_tiffs": False,
    "link_views": True,
    "exact_level": False,
    "napari_layer_panels": False,
    "show_tool_layers": False,
    # (0.18.0) the Autocropper's options a new window starts with
    "autocropper": {"method": "otsu", "grow": True, "margin": 10.0,
                    "moving": True, "outline": False,
                    "outline_kind": "silhouettes", "seed_mode": "stable",
                    "split": True},
    # (0.18.0) the View part's
    "view": {"slice_planes": True, "mask_mode": "dim",
             "outlines_3d": "all", "outlines_3d_opacity": 0.35,
             "crosshairs": True, "crop_regions": True,
             "crossing_marks": True, "preview": "off",
             "levels_source": "view", "levels_log": False,
             "levels_zoom": False},
    # (0.18.0) what opening an image starts with
    "opening": {"level_2d": "auto", "voxel_2d": 4.0, "view_3d": "off",
                "voxel_3d": 8.0, "channels": "all", "ask": True},
    # open (True) or closed parts of the panel
    "sections": {"image": True, "view": True, "channels": True,
                 "levels": True,
                 "regions": True, "ranges": False, "find": False,
                 "output": True},
}

SECTION_NAMES = {"image": "Image", "view": "View", "channels": "Channels",
                 "levels": "Levels (histogram)", "regions": "Crop regions",
                 "ranges": "Ranges (start, stop)",
                 "find": "Autocropper (experimental)",
                 "output": "Output"}


def path() -> Path:
    """Where the settings are kept."""
    given = os.environ.get("VOLUME_CROPPER_SETTINGS")
    return Path(given) if given else \
        Path.home() / ".volume_cropper" / "settings.json"


def load() -> dict:
    """The settings: the file's values over the defaults (unknown keys and
    values of the wrong type are ignored)."""
    values = json.loads(json.dumps(DEFAULTS))          # a deep copy
    try:
        saved = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return values
    if not isinstance(saved, dict):
        return values
    for key, default in DEFAULTS.items():
        value = saved.get(key)
        if isinstance(default, dict):
            if isinstance(value, dict):          # each known entry, checked
                for name, inner in default.items():
                    if _fits(value.get(name), inner):
                        values[key][name] = value[name]
        elif _fits(value, default):
            values[key] = value
    return values


def _fits(value, default) -> bool:
    """Whether a saved value can stand for a default: the same type (a
    whole number for a decimal one too; never a bool for a number)."""
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float))
    return isinstance(value, type(default))


def save(values: dict) -> str | None:
    """Write the settings; returns None, or a message if they could not be
    written (e.g. a read-only home folder)."""
    target = path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(differences(values), indent=2),
                             encoding="utf-8")
        temporary.replace(target)            # all or nothing
        return None
    except OSError as err:
        return f"The settings could not be saved to {target}: {err}"


def differences(values: dict) -> dict:
    """(0.18.0) Only the values that differ from the defaults: what save
    writes, so that a later version's new defaults reach every user who
    has not chosen otherwise (saving all of them froze the defaults of the
    day: found when "Outlines in 3D" changed its default)."""
    out = {}
    for key, value in values.items():
        default = DEFAULTS.get(key)
        if isinstance(default, dict) and isinstance(value, dict):
            inner = {name: v for name, v in value.items()
                     if default.get(name) != v}
            if inner:
                out[key] = inner
        elif value != default:
            out[key] = value
    return out


def update(**changes) -> dict:
    """Change some settings and save them; returns all of them."""
    values = load()
    for key, value in changes.items():
        if isinstance(DEFAULTS.get(key), dict):
            values[key].update(value)
        else:
            values[key] = value
    save(values)
    return values
