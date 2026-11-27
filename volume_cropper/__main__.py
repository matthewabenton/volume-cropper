"""Start the Volume Cropper window.

    python <tool folder>/volume_cropper/__main__.py [IMAGE.zarr] [--channels 0]

is what the launchers (volume_cropper.bat, volume_cropper.sh) run. They do
not change the working folder first, so a relative image path, and the
folder the Open dialog starts in, are those of the folder the user started
from (as before the package layout). From the tool folder,
``python -m volume_cropper`` does the same.
"""
import os
import sys

if not __package__:
    # Started as a file, not with -m: Python has put this file's folder (the
    # package folder) at the front of sys.path. Put the tool folder there
    # instead, so that the package is found as "volume_cropper" and its
    # modules are not also importable by their bare names (core, settings,
    # ...). The path is not resolved: on Windows that would turn a drive letter into a
    # network (UNC) path, and the napari plugin compares folders as given.
    # (core.py does the same when it is started as a file.)
    _package_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [entry for entry in sys.path
                   if os.path.normcase(os.path.abspath(entry or os.curdir))
                   != os.path.normcase(_package_dir)]
    sys.path.insert(0, os.path.dirname(_package_dir))

from volume_cropper.crop_gui import main  # noqa: E402  (after the path fix)

if __name__ == "__main__":
    main()
