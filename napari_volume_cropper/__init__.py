"""Volume Cropper as a napari plugin (0.13.0), for use within the lab.

napari finds the plugin through its manifest, napari.yaml, named in the
"napari.manifest" entry point (pyproject.toml). Nothing here is imported
until napari calls one of the functions the manifest names, so napari
starts as fast with the plugin installed as without it.

Only this package is installed (an "editable" install from the tool
folder, see volume_cropper.bat --install-plugin); the tool itself is
imported from the tool folder when it is first used (_tool.py).

- _widget.py:   Plugins > Volume Cropper (a small panel in the user's window)
- _commands.py: Layers > Data > Crop selected layers…, and File > IO
  Utilities > Open in a new window
- _sample.py:   File > Open Sample > Volume Cropper (synthetic demo)

From the moment napari loads this package, Python writes no cache files
(__pycache__) for modules it compiles: the package and the tool live in
the shared tool folder, which must stay free of them (the launchers set
PYTHONDONTWRITEBYTECODE for the same reason). Modules already compiled
(napari, other packages) are not affected.
"""
import sys

sys.dont_write_bytecode = True
