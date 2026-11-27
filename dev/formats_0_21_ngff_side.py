"""The ngff_zarr side of the 0.21.0 format check (run in an environment
with ngff_zarr 0.43.0, which reads and writes OME-Zarr 0.6 and
RFC-9 .ozx files on its own, independently of Volume Cropper).

    python formats_0_21_ngff_side.py write OUT_FOLDER
        writes test images with ngff_zarr: a 0.6 folder, a 0.6 .ozx and a
        0.5 .ozx of one random czyx volume (seed 0), plus that volume as
        volume.npy, and params.json (voxel size, position)

    python formats_0_21_ngff_side.py read PATH [PATH ...]
        reads OME-Zarr images (folders or .ozx) with ngff_zarr, validating
        their metadata against its OME-Zarr JSON schemas, and prints one
        JSON line per image: version, axes, each level's scale,
        translation and shape, and level 0's voxel sum and a hash of its
        bytes (to compare with Volume Cropper's numbers).

Paired with formats_0_21_check.py (run in the tool's environment).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import ngff_zarr as nz


def write(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 4000, size=(2, 24, 96, 80), dtype=np.uint16)
    np.save(out / "volume.npy", volume)
    scale = {"c": 1.0, "z": 2.0, "y": 0.25, "x": 0.25}
    translation = {"c": 0.0, "z": 10.0, "y": 3.0, "x": 5.0}
    image = nz.to_ngff_image(volume, dims=["c", "z", "y", "x"],
                             scale=scale, translation=translation,
                             axes_units={"z": "micrometer",
                                         "y": "micrometer",
                                         "x": "micrometer"},
                             name="ngff_zarr_test")
    multiscales = nz.to_multiscales(image, scale_factors=[2],
                                    chunks={"c": 1, "z": 8, "y": 32, "x": 32})
    # (ngff_zarr 0.43 writes .ozx only for 0.5, and 0.6 as "0.6.dev4")
    for name, version in (("ngff06.ome.zarr", "0.6"),
                          ("ngff05.ome.zarr", "0.5"),
                          ("ngff05.ozx", "0.5")):
        nz.to_ngff_zarr(str(out / name), multiscales, version=version)
        print("wrote", name)
    (out / "params.json").write_text(json.dumps(
        {"scale": scale, "translation": translation}))


def read(paths: list[str]) -> None:
    for path in paths:
        record = {"path": path}
        try:
            try:
                multiscales = nz.from_ngff_zarr(path, validate=True)
            except Exception as err:
                # (ngff_zarr 0.43's own 0.6 schema accepts only the label
                # "0.6.dev4", not the final "0.6": record it, then read
                # without its validation to compare the voxels)
                record["validation_error"] = str(err).splitlines()[0]
                multiscales = nz.from_ngff_zarr(path, validate=False)
            record["version"] = multiscales.metadata.version \
                if hasattr(multiscales.metadata, "version") else None
            record["levels"] = []
            for image in multiscales.images:
                record["levels"].append({
                    "dims": list(image.dims),
                    "scale": [image.scale[d] for d in image.dims],
                    "translation": [image.translation[d]
                                    for d in image.dims],
                    "shape": list(image.data.shape)})
            data = np.asarray(multiscales.images[0].data)
            record["sum"] = int(data.astype(np.int64).sum())
            record["sha1"] = hashlib.sha1(
                np.ascontiguousarray(data).tobytes()).hexdigest()
        except Exception as err:          # report, then go on
            record["error"] = f"{type(err).__name__}: {err}"
        print(json.dumps(record))


if __name__ == "__main__":
    if sys.argv[1] == "write":
        write(Path(sys.argv[2]))
    else:
        read(sys.argv[2:])
