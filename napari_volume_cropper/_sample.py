"""Sample data (prototype): File > Open Sample > Volume Cropper > Synthetic
demo. A small synthetic 3-channel volume (blobs in noise), made in memory
in about a second; it has nothing to do with real data."""
import numpy as np


def make_demo():
    """napari's LayerData: one image layer per channel, in µm."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(0)
    shape = (64, 256, 256)                         # z, y, x
    layers = []
    for c, (name, color) in enumerate((("nuclei", "magenta"),
                                       ("membranes", "green"),
                                       ("marker", "cyan"))):
        seeds = np.zeros(shape, dtype=np.float32)
        points = rng.integers(0, shape, size=(60, 3))
        seeds[tuple(points.T)] = 1.0
        blobs = ndi.gaussian_filter(seeds, sigma=(2, 6, 6))
        blobs = blobs / blobs.max() * 3000
        noise = rng.poisson(50, size=shape)
        data = (blobs + noise).astype(np.uint16)
        layers.append((data, {"name": name, "colormap": color,
                              "blending": "additive",
                              "scale": (2.0, 0.5, 0.5), "units": "um"},
                       "image"))
    return layers
