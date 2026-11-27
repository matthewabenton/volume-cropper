"""Check napari's clipping-plane convention (opens an invisible napari window).

Run with ``python test_napari_clipping.py`` (or pytest), especially after
upgrading napari.

ortho_views.clipping_planes() places planes at the box edges in µm, because
napari 0.9.2 reads plane positions in world coordinates, although its
documentation says data coordinates. This test renders a random volume
clipped to a box and compares it with a rendering of just the box's
sub-volume; the two must look the same.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from volume_cropper import ortho_views as ov


def _pump(seconds=1.0):
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.02)


def test_clipping_matches_cropped_volume():
    viewer = napari.Viewer(show=False)
    window = viewer.window._qt_window
    # Create the window and its OpenGL canvas without showing it on screen.
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(900, 700)
    window.show()
    try:
        viewer.dims.ndisplay = 3
        volume = np.random.default_rng(0).random((60, 60, 60)).astype(np.float32)
        scale = np.array([8.0, 8.128, 8.128])     # anisotropic, as real data
        translate = np.array([100.0, 50.0, 0.0])
        first, stop = np.array([10, 15, 20]), np.array([40, 35, 45])
        low = translate + (first - 0.5) * scale        # box edges in µm
        high = translate + (stop - 0.5) * scale
        cam = ov.camera(viewer)

        def render(data, offset, planes=None):
            viewer.layers.clear()
            layer = viewer.add_image(data, scale=scale, translate=offset,
                                     contrast_limits=(0, 1))
            cam.center = tuple((low + high) / 2)
            cam.zoom = 1.5
            if planes is not None:
                layer.experimental_clipping_planes = planes
            _pump()
            return viewer.screenshot(canvas_only=True)[..., :3].max(-1) > 30

        box = tuple(slice(a, b) for a, b in zip(first, stop))
        truth = render(volume[box], translate + first * scale)
        clipped = render(volume, translate, ov.clipping_planes(low, high))
        agreement = (clipped == truth).mean()
        print(f"lit pixels: box {truth.sum()}, clipped {clipped.sum()}; "
              f"agreement {agreement:.3f}")
        assert truth.sum() > 1000, "the test volume did not render"
        assert agreement > 0.95, (
            "clipping no longer matches the box: napari's clipping-plane "
            "convention may have changed (see ortho_views.clipping_planes)")
    finally:
        viewer.close()


if __name__ == "__main__":
    test_clipping_matches_cropped_volume()
    print("PASS test_clipping_matches_cropped_volume")
