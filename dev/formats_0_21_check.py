"""Independent checks of Volume Cropper 0.21.0's new formats (OME-Zarr 0.6,
zipped OME-Zarr, RFC-9 .ozx), against other people's code and files.

Three steps (run from the tool folder, or set VC_TOOL):

1. tool env:    python formats_0_21_check.py make OUT --schemas DIR
                    --samples DIR
   - opens the images ngff_zarr wrote (formats_0_21_ngff_side.py write
     OUT/ngff, run first in an environment with ngff_zarr 0.43.0) and compares them with the
     volume it wrote (voxels, voxel size, position), then crops them;
   - crops synthetic 0.6 images (with links to other coordinate systems)
     into folders and .ozx files, and validates every 0.6 metadata
     against the official 0.6 JSON schemas (image.schema; DIR = the
     schemas folder of the ngff-spec repository at tag 0.6);
   - opens and crops the public sample .ozx files in --samples;
   - writes OUT/crops/* and OUT/ours.jsonl (Volume Cropper's reading of
     each crop, in the format of formats_0_21_ngff_side.py read).
2. ngff_zarr:   python formats_0_21_ngff_side.py read OUT/crops/...
                    > OUT/theirs.jsonl
3. tool env:    python formats_0_21_check.py compare OUT --ozx-tck DIR
   - compares ngff_zarr's reading of the crops with Volume Cropper's;
   - validates every .ozx in OUT/crops and --samples with ozx-tck (RFC-9's
     test kit; DIR = its src folder), strict (warnings count).

Prints PASS/FAIL lines and a summary; exit code 1 if anything failed.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr

TOOL = Path(os.environ.get("VC_TOOL", Path.cwd()))
sys.path.insert(0, str(TOOL))
sys.path.insert(1, str(Path(sys.path[0]) / "tests"))  # its test helpers
from volume_cropper import core as vc                                  # noqa: E402
from test_volume_cropper import make_test_image              # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail
                                                   else ""))


def reading(path) -> dict:
    """Volume Cropper's reading of an image, as the ngff side prints it."""
    image = vc.open_image(path)
    try:
        data = image.levels[0].array[...]
        return {"path": str(path), "version": image.ome_version,
                "levels": [{"dims": image.axis_names, "scale": level.scale,
                            "translation": level.translation,
                            "shape": list(level.array.shape)}
                           for level in image.levels],
                "sum": int(data.astype(np.int64).sum()),
                "sha1": hashlib.sha1(
                    np.ascontiguousarray(data).tobytes()).hexdigest()}
    finally:
        image.close()


def schema_validator(schemas: Path):
    """A validator for OME-Zarr 0.6 image metadata, offline: the schemas
    refer to each other by their web addresses, mapped to the files."""
    import jsonschema
    from referencing import Registry, Resource

    resources = []
    for path in schemas.glob("*.schema"):
        contents = json.loads(path.read_text(encoding="utf-8"))
        resources.append((contents["$id"], Resource.from_contents(contents)))
    registry = Registry().with_resources(resources)
    image_schema = json.loads((schemas / "image.schema").read_text(
        encoding="utf-8"))
    return jsonschema.Draft202012Validator(image_schema, registry=registry)


def validate_06(validator, path, name):
    """Validate a 0.6 image's attributes (folder or zip)."""
    image = vc.open_image(path)
    try:
        attrs = {"ome": copy.deepcopy(image.attrs["ome"])}
    finally:
        image.close()
    errors = sorted(validator.iter_errors(attrs), key=lambda e: e.path)
    check(f"0.6 schema: {name}", not errors,
          "; ".join(e.message[:120] for e in errors[:3]))


def make(out: Path, schemas: Path, samples: Path):
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    validator = schema_validator(schemas)
    # The validator itself: the official example passes, a broken one not
    example = json.loads(
        (schemas.parent / "tests" / "attributes" / "strict" / "valid" /
         "image" / "multiscales_example.json").read_text(encoding="utf-8"))
    check("schema validator accepts the official example",
          validator.is_valid(example.get("attributes", example)))
    broken = copy.deepcopy(example.get("attributes", example))
    del broken["ome"]["multiscales"][0]["coordinateSystems"]
    check("schema validator rejects it without coordinateSystems",
          not validator.is_valid(broken))

    # 1. ngff_zarr's images
    ngff = out / "ngff"
    volume = np.load(ngff / "volume.npy")
    params = json.loads((ngff / "params.json").read_text())
    box = vc.CropBox("n1", {"z": (14, 40), "y": (6, 18), "x": (8, 20)})
    for name in ("ngff06.ome.zarr", "ngff05.ome.zarr", "ngff05.ozx"):
        image = vc.open_image(ngff / name)
        try:
            level0 = image.levels[0]
            check(f"ngff_zarr {name}: voxels", np.array_equal(
                level0.array[...], volume))
            check(f"ngff_zarr {name}: voxel size and position",
                  level0.scale == [params["scale"][a] for a in "czyx"]
                  and level0.translation == [params["translation"][a]
                                             for a in "czyx"],
                  f"{level0.scale} {level0.translation} "
                  f"({image.ome_version}, {image.format_name})")
            result = vc.crop_box(image, box, crops, min_size=4,
                                 overwrite=True)
            check(f"ngff_zarr {name}: crop {result.path.name}",
                  result.problems == [], "; ".join(result.problems[:2]))
            if vc.is_v06(image.ome_version):
                validate_06(validator, result.path, result.path.name)
        finally:
            image.close()

    # 2. synthetic 0.6 images, with links, as folders and .ozx
    synthetic = out / "synthetic"
    synthetic.mkdir(exist_ok=True)
    from test_formats import BOX, _image_with_links
    source = synthetic / "links06.zarr"
    if not source.exists():
        _image_with_links(source)
    ozx = vc.pack_zip(source, synthetic / "links06.ozx")
    validate_06(validator, source, "synthetic source")
    for path in (source, ozx):
        image = vc.open_image(path)
        try:
            for keep in (True, False):
                result = vc.crop_box(image, vc.CropBox(f"keep{keep}",
                                                       BOX.ranges),
                                     crops, min_size=4, overwrite=True,
                                     keep_position=keep)
                check(f"synthetic {path.name}: crop {result.path.name}",
                      result.problems == [], "; ".join(result.problems[:2]))
                validate_06(validator, result.path, result.path.name)
        finally:
            image.close()

    # 3. the public samples
    for sample in sorted(samples.glob("*.ozx")):
        image = vc.open_image(sample)
        try:
            print(image.describe())
            region = {}
            for axis in image.range_axes:     # the middle quarter of each
                low, high = image.extent(axis)
                region[axis] = (low + 0.375 * (high - low),
                                low + 0.625 * (high - low))
            result = vc.crop_box(image, vc.CropBox("middle", region), crops,
                                 min_size=4, overwrite=True)
            check(f"sample {sample.name}: crop {result.path.name}",
                  result.problems == [], "; ".join(result.problems[:2]))
        finally:
            image.close()

    with open(out / "ours.jsonl", "w", encoding="utf-8") as f:
        for path in sorted(crops.iterdir()):
            f.write(json.dumps(reading(path)) + "\n")
    print("crops for step 2:", " ".join(str(p) for p in
                                         sorted(crops.iterdir())))


def compare(out: Path, ozx_tck: Path, samples: Path):
    # (matched by file name: the two sides write the paths differently,
    # C:\... and C:/...)
    ours = {Path(r["path"]).name: r for r in map(
        json.loads, (out / "ours.jsonl").read_text().splitlines())}
    theirs = [json.loads(line) for line in
              (out / "theirs.jsonl").read_text().splitlines()
              if line.startswith("{")]
    for record in theirs:
        name = Path(record["path"]).name
        mine = ours.get(name)
        if "error" in record:
            check(f"ngff_zarr reads {name}", False, record["error"][:200])
            continue
        same = (mine is not None and record["sha1"] == mine["sha1"]
                and len(record["levels"]) == len(mine["levels"])
                and all(np.allclose(a["scale"], b["scale"])
                        and np.allclose(a["translation"], b["translation"])
                        and a["shape"] == b["shape"]
                        for a, b in zip(record["levels"], mine["levels"])))
        check(f"ngff_zarr reads {name} as Volume Cropper does", same,
              "" if same else f"theirs {record['levels'][0]}, ours "
              f"{mine and mine['levels'][0]}")
    sys.path.insert(0, str(ozx_tck))
    from ozx_tck.validate import Validator
    for path in sorted([*(out / "crops").glob("*.ozx"),
                        *samples.glob("*.ozx")]):
        with Validator(path, fail_fast=False, strict=True) as validator:
            events = validator.process()
        check(f"ozx-tck validates {path.name}", not events,
              "; ".join(e.fmt() for e in events[:3]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["make", "compare"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--schemas", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--ozx-tck", type=Path)
    args = parser.parse_args()
    if args.step == "make":
        make(args.out, args.schemas, args.samples)
    else:
        compare(args.out, args.ozx_tck, args.samples)
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
