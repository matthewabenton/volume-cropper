"""Run every test of Volume Cropper and report all failures, not just the
first.

Usage, in the tool folder with the volume_cropper environment active:

    python tests/run_all_tests.py              # all tests
    python tests/run_all_tests.py crop         # only the crop code (no napari needed)
    python tests/run_all_tests.py views clipping

Each test script can also be run on its own (python tests/test_volume_cropper.py),
but then it stops at its first failure. This runner keeps going, prints one
line per test and a summary, and exits with 1 if any test failed (0 if all
passed), so that it can be used in scripts.
"""
import importlib
import sys
import time
import traceback

# Short name -> test module.
MODULES = {
    "crop": "test_volume_cropper",     # the crop code (no napari needed)
    "files": "test_file_images",       # TIFF files in memory (0.12.0)
    "views": "test_ortho_views",       # the view geometry (needs napari)
    "clipping": "test_napari_clipping",  # napari's 3D clipping (hidden window)
    "plugin": "test_plugin",           # the napari plugin (0.13.0; hidden windows)
    "autocrop": "test_autocrop",       # the Autocropper (0.14.0, experimental)
    "panel": "test_panel",             # the panel's 0.15.0 changes (hidden windows)
    "microscope": "test_microscope_files",  # LIF, CZI, ND2, OIR readers (0.15.0)
    "settings": "test_settings",       # settings, panel parts, layer list (0.16.0)
    "moving": "test_moving",           # regions that move over time (0.17.0)
    "stitching": "test_stitching",     # tile scans stitched (0.17.2)
    "levels": "test_levels",           # the Levels histogram (0.18.0; hidden windows)
    "crop_projection": "test_crop_projection",  # the preview's projection (0.18.0)
    "handles": "test_box_handles",     # the dots that change a region in 3D (0.18.0)
    "formats": "test_formats",         # OME-Zarr 0.6, zipped OME-Zarr, .ozx (0.21.0)
    "docs": "test_docs",               # README.pdf etc. in step (0.22.0)
    "public": "test_public_files",     # no drive folders or addresses (0.23.0)
    "text": "test_qt_text",            # names from files as plain text (0.23.0)
}


def run(names: list[str]) -> int:
    """Run the tests of the chosen modules; return the exit code."""
    unknown = [name for name in names if name not in MODULES]
    if unknown:
        print(f"Unknown: {', '.join(unknown)}; choose from "
              f"{', '.join(MODULES)}.")
        return 2
    # The tests never use (or change) the user's own settings
    # (~/.volume_cropper/settings.json, 0.16.0): a fresh file of their own.
    import os
    import tempfile
    if "VOLUME_CROPPER_SETTINGS" not in os.environ:
        os.environ["VOLUME_CROPPER_SETTINGS"] = os.path.join(
            tempfile.mkdtemp(prefix="vc_test_settings_"), "settings.json")
    failed, total, start = [], 0, time.perf_counter()
    for name in names:
        module = importlib.import_module(MODULES[name])
        tests = [getattr(module, attr) for attr in sorted(dir(module))
                 if attr.startswith("test_") and callable(getattr(module, attr))]
        for test in tests:
            total += 1
            label = f"{module.__name__}.{test.__name__}"
            t0 = time.perf_counter()
            try:
                test()
            except Exception:           # report it and go on with the rest
                failed.append(label)
                print(f"FAIL {label}", flush=True)
                traceback.print_exc()
            else:
                print(f"PASS {label} ({time.perf_counter() - t0:.1f} s)",
                      flush=True)
    seconds = time.perf_counter() - start
    print(f"\n{total - len(failed)} of {total} tests passed in {seconds:.0f} s.")
    for label in failed:
        print(f"  failed: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or list(MODULES)))
