"""Tests for volume_cropper.

Run with pytest (``pytest test_volume_cropper.py``) or without it
(``python test_volume_cropper.py``).

The tests build a small synthetic OME-Zarr laid out like eubi-bridge's output
(sharded, blosc-zstd, an anisotropic pyramid with rounded-up shapes), crop it
and check that:
- every level of the crop is identical to the matching source region;
- the levels stay aligned with each other (the snapping works);
- the format (chunks, shards, codecs) and the OME metadata are right;
- the safety checks refuse to overwrite data the tool did not write.
Since 0.8.0 the same is checked for OME-Zarr 0.4 (Zarr v2) images, whose
crops must stay 0.4, and for images without a channel axis (z, y, x).
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from zarr.codecs import BloscCodec

from volume_cropper import core as vc

# Downsampling factors per level for (c, z, y, x), like a typical lightsheet
# pyramid:
# level 1 shrinks only xy, later levels shrink all spatial axes.
FACTORS = [(1, 1, 1, 1), (1, 1, 4, 4), (1, 2, 8, 8), (1, 4, 16, 16)]


def make_test_image(path, shape0=(3, 40, 150, 130), factors=FACTORS,
                    scale0=(1.0, 2.0, 0.25, 0.25), axes="czyx",
                    translation=None, seed=0, version="0.5"):
    """Write a synthetic OME-Zarr image and return its level-0 data.

    version "0.5": Zarr v3 with shards, the OME metadata under "ome" (like
    eubi-bridge's 0.5 output); "0.4": Zarr v2, one file per chunk in nested
    folders, the OME metadata directly in the attributes (like its 0.4
    output); "0.6" (0.21.0): as 0.5, with the axes in a coordinate system
    named "physical" and one transformation per level (a scale, or a
    sequence of a scale and a translation) that leads to it, as in the 0.6
    specification's examples. Coarser levels are taken by striding
    ("simple" downsampling), so level L voxel j is level-0 voxel j *
    factor. The tests use this to check that the cropped levels line up
    with each other.
    """
    rng = np.random.default_rng(seed)
    data0 = rng.integers(0, 4000, size=shape0, dtype=np.uint16)
    v3 = version in ("0.5", "0.6")
    root = zarr.open_group(path, mode="w-", zarr_format=3 if v3 else 2)
    datasets = []
    for level, f in enumerate(factors):
        data = data0[tuple(slice(None, None, n) for n in f)]
        chunks = tuple(1 if a in "ct" else 8 for a in axes)
        shards = tuple(1 if a in "ct" else 24 for a in axes)
        if v3:
            arr = root.create_array(
                name=str(level), shape=data.shape, dtype=data.dtype,
                chunks=chunks, shards=shards,
                compressors=BloscCodec(cname="zstd", clevel=5,
                                       shuffle="shuffle"),
                fill_value=0, dimension_names=list(axes))
        else:
            arr = root.create_array(
                name=str(level), shape=data.shape, dtype=data.dtype,
                chunks=chunks,
                compressors=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
                chunk_key_encoding={"name": "v2", "separator": "/"},
                fill_value=0)
        arr[...] = data
        transforms = [{"type": "scale",
                       "scale": [s * n for s, n in zip(scale0, f)]}]
        if translation is not None:
            transforms.append({"type": "translation",
                               "translation": list(translation)})
        if version == "0.6":
            ends = {"input": {"path": str(level)},
                    "output": {"name": "physical"}}
            transforms = [{**transforms[0], **ends} if len(transforms) == 1
                          else {"type": "sequence", **ends,
                                "transformations": transforms}]
        datasets.append({"path": str(level),
                         "coordinateTransformations": transforms})
    axis_info = {"t": {"name": "t", "type": "time", "unit": "minute"},
                 "c": {"name": "c", "type": "channel"}}
    axes_meta = [axis_info.get(a, {"name": a, "type": "space",
                                   "unit": "micrometer"}) for a in axes]
    ome = {"multiscales": [{"name": "test", "axes": axes_meta,
                            "datasets": datasets}]}
    if version == "0.6":
        del ome["multiscales"][0]["axes"]
        ome["multiscales"][0]["coordinateSystems"] = [
            {"name": "physical", "axes": axes_meta}]
    if "c" in axes:
        ome["omero"] = {"channels": [{"label": f"ch{i}", "color": "FFFFFF"}
                                     for i in range(shape0[axes.index("c")])]}
    if v3:
        root.attrs["ome"] = {"version": version, **ome}
    else:
        ome["multiscales"][0]["version"] = "0.4"
        root.attrs.update(ome)
    return data0


def strided(data, factor):
    return data[tuple(slice(None, None, n) for n in factor)]


def test_index_conversion_rounds_outward():
    # Voxel i is centred at i * 0.5 µm and covers +-0.25 µm.
    assert vc.physical_to_index_range(0.0, 1.0, 0.5, 0.0) == (0, 3)
    assert vc.physical_to_index_range(0.3, 0.7, 0.5, 0.0) == (1, 2)
    # Exactly on voxel edges: nothing is added.
    first, end = vc.physical_to_index_range(4.75, 9.75, 0.5, 0.0)
    assert (first, end) == (10, 20)
    assert vc.index_range_to_physical(first, end, 0.5, 0.0) == (4.75, 9.75)
    # Floating-point noise must not add an extra voxel.
    assert vc.physical_to_index_range(0.254 * 9.5, 0.254 * 19.5,
                                      0.254, 0.0) == (10, 20)


def test_crop_matches_source_and_levels_align():
    for version in ("0.5", "0.4", "0.6"):
        _crop_matches_source_and_levels_align(version)


def _crop_matches_source_and_levels_align(version):
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        data0 = make_test_image(src_path, version=version)
        image = vc.open_image(src_path)
        assert image.ome_version == version
        box = vc.CropBox("b1", {"z": (11, 53), "y": (5.3, 20.1),
                                "x": (3.0, 25.7)})
        plan = vc.plan_crop(image, box, min_size=4)
        assert plan.level_indices == [0, 1, 2, 3]
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr", workers=4)
        assert vc.verify_crop(image, plan, out) == []

        crop = vc.open_image(out)
        level0 = crop.levels[0].array[...]
        a, b = plan.starts[0], plan.stops[0]
        np.testing.assert_array_equal(
            level0, data0[:, a[1]:b[1], a[2]:b[2], a[3]:b[3]])
        # Each coarse level must equal the strided level-0 crop: this only
        # holds if the box was snapped to the coarsest level's grid.
        for k, factor in enumerate(FACTORS):
            np.testing.assert_array_equal(crop.levels[k].array[...],
                                          strided(level0, factor))
        # The snapped box contains the requested one.
        for axis, (low, high) in box.ranges.items():
            s_low, s_high = plan.snapped_ranges[axis]
            assert s_low <= low and s_high >= high


def test_format_and_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        box = vc.CropBox("b1", {"z": (20, 40), "y": (10, 20)}, channels=[0, 2])
        plan = vc.plan_crop(image, box, min_size=4)
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr",
                            channel_labels=["nuclei", "marker"])
        for k in range(len(plan.level_indices)):
            src_meta = json.loads((src_path / str(k) / "zarr.json").read_text())
            out_meta = json.loads((out / str(k) / "zarr.json").read_text())
            for key in ("data_type", "chunk_grid", "codecs", "fill_value",
                        "dimension_names", "chunk_key_encoding"):
                assert out_meta[key] == src_meta[key], key

        attrs = zarr.open_group(out, mode="r").attrs.asdict()
        datasets = attrs["ome"]["multiscales"][0]["datasets"]
        for k, ds in enumerate(datasets):
            scale = ds["coordinateTransformations"][0]["scale"]
            assert scale == image.levels[k].scale
            translation = ds["coordinateTransformations"][1]["translation"]
            expected = [0.0] + [round(i * s, 9) for i, s in
                                zip(plan.starts[k][1:], scale[1:])]
            assert translation == expected
        # The translation is the same physical point on every level.
        assert len({tuple(d["coordinateTransformations"][1]["translation"])
                    for d in datasets}) == 1
        labels = [c["label"] for c in attrs["ome"]["omero"]["channels"]]
        assert labels == ["nuclei", "marker"]
        assert attrs["crop_provenance"]["channels"] == [0, 2]

        crop = vc.open_image(out)
        np.testing.assert_array_equal(
            crop.levels[0].array[1],
            image.levels[0].array[2][tuple(
                slice(a, b) for a, b in zip(plan.starts[0][1:],
                                            plan.stops[0][1:]))])


def test_keep_position_false_writes_no_translation():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("b", {"x": (5, 10)}))
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr",
                            keep_position=False)
        datasets = zarr.open_group(out, mode="r").attrs[
            "ome"]["multiscales"][0]["datasets"]
        assert all(len(d["coordinateTransformations"]) == 1
                   for d in datasets)


def test_keep_position_false_starts_at_zero():
    """Without "keep position" a crop starts at 0 µm, also when the source
    has an offset of its own (e.g. from stitching), and its levels still
    line up with each other."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path, translation=(0, 100.0, -350.5, 20.0))
        image = vc.open_image(src_path)
        box = vc.CropBox("b", {"z": (120, 150), "y": (-345, -335)})
        plan = vc.plan_crop(image, box, min_size=4)
        assert len(plan.level_indices) > 1
        placed = vc.open_image(vc.write_crop(image, plan,
                                             Path(tmp) / "on.zarr"))
        alone = vc.open_image(vc.write_crop(image, plan,
                                            Path(tmp) / "off.zarr",
                                            keep_position=False))
        assert placed.levels[0].translation[1:] != [0.0, 0.0, 0.0]
        assert alone.levels[0].translation == [0.0, 0.0, 0.0, 0.0]
        for on, off in zip(placed.levels, alone.levels):
            # the same picture, moved as a whole: every level by the same
            # amount (level 0's position in the source)
            shift = [a - b for a, b in zip(on.translation, off.translation)]
            assert np.allclose(shift, [0.0] + placed.levels[0].translation[1:])


def test_existing_translation_is_respected():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        data0 = make_test_image(src_path, translation=(0, 100.0, 50.0, 0))
        image = vc.open_image(src_path)
        # z 120-140 µm = voxels 10-19 (scale 2, origin 100).
        plan = vc.plan_crop(image, vc.CropBox("b", {"z": (119.5, 138.5)}),
                            n_levels=1)
        assert (plan.starts[0][1], plan.stops[0][1]) == (10, 20)
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr")
        np.testing.assert_array_equal(vc.open_image(out).levels[0].array[...],
                                      data0[:, 10:20])


def test_box_is_clipped_to_the_image():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        z_min, z_max = image.extent("z")
        plan = vc.plan_crop(image, vc.CropBox("b", {"z": (-50, 10),
                                                    "x": (20, 1e6)}),
                            min_size=4)
        assert plan.starts[0][1] == 0
        assert plan.stops[0][3] == image.levels[0].array.shape[3]
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr")
        assert vc.verify_crop(image, plan, out) == []
        try:
            vc.plan_crop(image, vc.CropBox("b", {"z": (z_max + 1, z_max + 9)}))
        except ValueError:
            pass
        else:
            raise AssertionError("a box outside the image must fail")


def test_level_choice():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        box = vc.CropBox("b", {"y": (0, 30), "x": (0, 30)})   # 120 px in xy
        assert vc.plan_crop(image, box, min_size=64).level_indices == [0]
        assert vc.plan_crop(image, box, min_size=30).level_indices == [0, 1]
        assert vc.plan_crop(image, box, n_levels=3).level_indices == [0, 1, 2]


def test_time_axis_and_ranges():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        factors = [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)]
        data0 = make_test_image(src_path, shape0=(5, 2, 10, 40, 40),
                                factors=factors, axes="tczyx",
                                scale0=(30.0, 1.0, 2.0, 0.5, 0.5))
        image = vc.open_image(src_path)
        assert image.channel_axis == 1
        # t in minutes: frames 1-3 (centres at 30, 60, 90 min).
        plan = vc.plan_crop(image, vc.CropBox("b", {"t": (30, 90)},
                                              channels=[1]), min_size=4)
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr")
        assert vc.verify_crop(image, plan, out) == []
        np.testing.assert_array_equal(vc.open_image(out).levels[0].array[...],
                                      data0[1:4, 1:2])


def test_refuses_dangerous_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("b", {"x": (5, 10)}))
        for bad in (src_path, src_path / "inside.zarr", Path(tmp)):
            try:
                vc.write_crop(image, plan, bad)
            except (ValueError, FileExistsError):
                pass
            else:
                raise AssertionError(f"writing to {bad} must fail")

        # A folder with other data must never be overwritten ...
        other = Path(tmp) / "other.zarr"
        other.mkdir()
        (other / "precious.txt").write_text("raw data")
        try:
            vc.write_crop(image, plan, other, overwrite=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrote a folder it did not create")
        assert (other / "precious.txt").exists()

        # ... but replacing an earlier crop is allowed with overwrite=True.
        out = Path(tmp) / "out.zarr"
        vc.write_crop(image, plan, out)
        try:
            vc.write_crop(image, plan, out)
        except FileExistsError:
            pass
        else:
            raise AssertionError("must not overwrite without overwrite=True")
        vc.write_crop(image, plan, out, overwrite=True)


def test_progress_in_bytes():
    """progress_bytes counts image data written: it only grows, and ends at
    the size of all levels of the crop (what the GUI's time estimate uses)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("b", {"z": (10, 60)}),
                            min_size=4)
        calls, counts = [], []
        vc.write_crop(image, plan, Path(tmp) / "out.zarr",
                      progress=lambda d, t: counts.append((d, t)),
                      progress_bytes=lambda d, t: calls.append((d, t)))
        dtype = image.levels[0].array.dtype
        expected = sum(vc.nbytes(shape, dtype) for shape in plan.shapes)
        assert len(calls) == len(counts) > 1            # once per shard
        assert all(t == expected for _, t in calls)
        done = [d for d, _ in calls]
        assert done == sorted(done) and done[-1] == expected


def test_cancel_leaves_no_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("b", {}))
        cancel = threading.Event()
        cancel.set()
        out = Path(tmp) / "out.zarr"
        try:
            vc.write_crop(image, plan, out, cancel_event=cancel)
        except vc.CropCancelled:
            pass
        else:
            raise AssertionError("cancel_event was ignored")
        attrs = zarr.open_group(out, mode="r").attrs
        assert "ome" not in attrs
        assert attrs["crop_provenance"]["complete"] is False
        # The interrupted crop is recognised as ours and can be replaced.
        vc.write_crop(image, plan, out, overwrite=True)
        attrs = zarr.open_group(out, mode="r").attrs
        assert attrs["crop_provenance"]["complete"] is True
        assert vc.verify_crop(image, plan, out) == []


def test_verify_finds_wrong_position_and_voxels():
    """verify_crop finds a crop at the wrong position, a wrong voxel and
    wrong channel names, and stores its result in the crop's record (review
    CORE-05: a 5 µm shift and a wrong first voxel used to pass)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("b", {"x": (5, 20)},
                                              channels=[2, 0]), min_size=4)
        outs = {}
        for keep in (True, False):          # no false alarms either way
            outs[keep] = vc.write_crop(image, plan, Path(tmp) / f"{keep}.zarr",
                                       keep_position=keep,
                                       channel_labels=["new"])
            assert vc.verify_crop(image, plan, outs[keep], keep_position=keep,
                                  channel_labels=["new"]) == []
            record = zarr.open_group(outs[keep],
                                     mode="r").attrs["crop_provenance"]
            assert record["verified"]["ok"] is True
        out = outs[True]
        group = zarr.open_group(out, mode="r+")
        good_ome = group.attrs["ome"]

        # 1. Level 0 moved by 5 µm in x.
        ome = json.loads(json.dumps(good_ome))
        for t in ome["multiscales"][0]["datasets"][0][
                "coordinateTransformations"]:
            if t["type"] == "translation":
                t["translation"][-1] += 5
        group.attrs["ome"] = ome
        problems = vc.verify_crop(image, plan, out, channel_labels=["new"])
        assert any("position" in p for p in problems), problems
        record = zarr.open_group(out, mode="r").attrs["crop_provenance"]
        assert record["verified"]["ok"] is False
        assert "CHECK FAILED" in vc.verification_text(plan, problems)
        group.attrs["ome"] = good_ome

        # 2. One wrong voxel, the very first one.
        level0 = zarr.open_array(out / "0", mode="r+")
        level0[0, 0, 0, 0] = level0[0, 0, 0, 0] + 1
        problems = vc.verify_crop(image, plan, out, channel_labels=["new"])
        assert any("voxels differ" in p for p in problems), problems

        # 3. Channel names: the crop says "new", the source "ch2".
        problems = vc.verify_crop(image, plan, outs[False],
                                  keep_position=False)
        assert any("channel names" in p for p in problems), problems
        assert "checked OK" in vc.verification_text(plan, [])


def test_error_stops_the_queued_work():
    """An error while copying one shard stops the shards still queued
    (review CORE-01: all 252 still ran after an error on the third)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("all"), min_size=4)
        calls = []
        original = vc._source_selection

        def failing(*args):
            calls.append(1)
            if len(calls) == 3:
                raise OSError("No space left on device (test)")
            return original(*args)

        vc._source_selection = failing
        try:
            vc.write_crop(image, plan, Path(tmp) / "out.zarr", workers=2)
        except OSError:
            pass
        else:
            raise AssertionError("the error was lost")
        finally:
            vc._source_selection = original
        assert len(calls) <= 6, f"{len(calls)} shards started"


def test_csv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        boxes = [vc.CropBox("a", {"z": (1.5, 20), "x": (3, 7.25)}),
                 vc.CropBox("b", {"y": (0, 10)}, channels=[0, 2])]
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv(boxes, csv_path, image)
        header = csv_path.read_text().splitlines()[0]
        assert header == ("name,z_start_um,z_stop_um,y_start_um,y_stop_um,"
                          "x_start_um,x_stop_um,channels")
        assert vc.load_boxes_csv(csv_path) == boxes


def test_csv_keeps_numbers_exactly():
    """Saved boxes load back as exactly the same numbers. "%g" kept only 6
    digits, which could move a box edge by one voxel (review CORE-08)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        boxes = [vc.CropBox("a", {"z": (1234.5678912345678, 4321.000000001),
                                  "y": (0.1 + 0.2, 100.0),
                                  "x": (12345.625, 99999.123456789)})]
        assert float(f"{12345.625:g}") != 12345.625      # the old problem
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv(boxes, csv_path, image)
        assert vc.load_boxes_csv(csv_path) == boxes
        assert ",100," in csv_path.read_text()   # whole numbers stay short


def test_csv_names_that_look_like_formulas_stay_text():
    """(0.23.0, the pre-publication review) A region name that starts like
    a spreadsheet formula (= + - @) is saved with a leading ' (Excel then
    shows it as text instead of computing it) and loads back unchanged;
    other names are saved as they are."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        names = ["=1+2", "+one", "-two", "@three", "plain-name"]
        boxes = [vc.CropBox(name, {"x": (k, k + 5)})
                 for k, name in enumerate(names)]
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv(boxes, csv_path, image)
        cells = [line.split(",")[0]
                 for line in csv_path.read_text().splitlines()[1:]]
        assert cells == ["'=1+2", "'+one", "'-two", "'@three",
                         "plain-name"], cells
        assert [box.name for box in vc.load_boxes_csv(csv_path)] == names


def test_axis_names_that_could_be_html_or_formulas_are_refused():
    """(0.23.0, the pre-publication review) Axis names and units come from
    the file and appear in the window (also in napari's own labels, which
    take HTML for HTML) and as column names in the regions file: a name or
    unit holding < > & " or a control character, or a name that starts like
    a formula, is refused when the image is opened."""
    cases = [("name", 'x<img src="a.png">'), ("name", "=HYPERLINK(1)"),
             ("name", "@x"), ("name", "x\ty"), ("unit", "micro<b>meter")]
    with tempfile.TemporaryDirectory() as tmp:
        for k, (field, value) in enumerate(cases):
            path = Path(tmp) / f"image{k}.zarr"
            make_test_image(path, version="0.4")
            attrs = read_json(path / ".zattrs")
            attrs["multiscales"][0]["axes"][-1][field] = value
            (path / ".zattrs").write_text(json.dumps(attrs))
            try:
                vc.open_image(path)
            except ValueError as err:
                assert "axis names and units" in str(err), err
            else:
                raise AssertionError(f"opened, with the axis {field} "
                                     f"{value!r}")
        path = Path(tmp) / "plain.zarr"              # plain ones still open
        make_test_image(path, version="0.4")
        assert vc.open_image(path).axis_names == ["c", "z", "y", "x"]


def test_csv_from_excel():
    """A file as Excel saves it in a German or French setup: semicolons,
    decimal commas, Windows text encoding, an empty row at the end."""
    with tempfile.TemporaryDirectory() as tmp:
        text = ("name;z_start_um;z_stop_um;y_start_um;y_stop_um;x_start_um;"
                "x_stop_um;channels;outline_y_x\r\n"
                'tête 1;1,5;20;;;3;7,25;"0;2";"1,5 2; 1,5 6,25; 5 6,25"\r\n'
                ";;;;;;;;\r\n")
        csv_path = Path(tmp) / "boxes.csv"
        csv_path.write_bytes(text.encode("cp1252"))
        assert vc.load_boxes_csv(csv_path) == [vc.CropBox(
            "tête 1", {"z": (1.5, 20.0), "x": (3.0, 7.25)}, [0, 2],
            {("y", "x"): [(1.5, 2.0), (1.5, 6.25), (5.0, 6.25)]})]


def test_csv_mistakes_are_refused():
    """Mistakes in a boxes file stop the loading with a message that names
    the row (as Excel numbers it) and says what to fix (review CORE-02,
    CORE-03). A stop without a start used to crop the whole axis."""
    header = ("name,z_start_um,z_stop_um,y_start_um,y_stop_um,x_start_um,"
              "x_stop_um,channels\n")
    cases = [
        (header + "ok,1,2,,,,,all\na,,20,,,,,all\n",
         "row 3 (box 'a'): z_stop_um is filled in but z_start_um is empty"),
        (header + "a,5,,,,,,all\n",
         "z_start_um is filled in but z_stop_um is empty"),
        (header + ",1,20,,,,,all\n", "row 2: the box has no name"),
        (header + "a,1,2O,,,,,all\n", "z_stop_um must be a number"),
        (header + "a,,,,,,,red\n", "channels must be 'all'"),
        ("box,z_start_um,z_stop_um\na,1,2\n", "no 'name' column"),
        ("name,z_start_um\na,1\n", "no column z_stop_um"),
        (header + "tail 1,,,,,,,all\ntail_1,,,,,,,all\n",
         "'tail 1' and 'tail_1'"),
        (header + "A,,,,,,,all\na,,,,,,,all\n", "'A' and 'a'"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "boxes.csv"
        for text, expected in cases:
            csv_path.write_text(text, encoding="utf-8")
            try:
                vc.load_boxes_csv(csv_path)
            except ValueError as err:
                assert expected in str(err), (expected, str(err))
            else:
                raise AssertionError(f"accepted a file with: {expected}")
        # Empty rows (Excel often leaves one at the end) are skipped.
        csv_path.write_text(header + "a,1,2,,,,,all\n,,,,,,,\n",
                            encoding="utf-8")
        assert [b.name for b in vc.load_boxes_csv(csv_path)] == ["a"]


def test_box_names_that_give_one_file_name():
    """Names that become the same file name are found (review CORE-04)."""
    assert vc.name_clashes(["tail 1", "tail_1", "Tail_1", "a", "b"]) == [
        ["tail 1", "tail_1", "Tail_1"]]
    assert vc.same_file_name("αβ", "γδ")           # both become "crop"
    assert not vc.same_file_name("crop01", "crop02")
    try:
        vc.check_box_names([vc.CropBox("A"), vc.CropBox("a")])
    except ValueError as err:
        assert "'A' and 'a'" in str(err)
    else:
        raise AssertionError("accepted two names that give one file name")


def test_overwrite_only_replaces_the_same_box():
    """'Overwrite' never replaces another box's crop whose name merely gives
    the same file name (review CORE-04: it used to, silently)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        first = vc.CropBox("tail 1", {"x": (5, 10)})
        second = vc.CropBox("tail_1", {"x": (20, 30)})
        out = vc.output_path_for(image, first, tmp)
        assert out == vc.output_path_for(image, second, tmp)
        vc.write_crop(image, vc.plan_crop(image, first), out)
        for attempt in (lambda: vc.check_output(image, second, out, True),
                        lambda: vc.write_crop(image,
                                              vc.plan_crop(image, second),
                                              out, overwrite=True)):
            try:
                attempt()
            except FileExistsError as err:
                assert "'tail 1'" in str(err)
            else:
                raise AssertionError("replaced another box's crop")
        record = zarr.open_group(out, mode="r").attrs["crop_provenance"]
        assert record["box_name"] == "tail 1" and record["complete"]
        # The same box may still be cropped again.
        vc.write_crop(image, vc.plan_crop(image, first), out, overwrite=True)


def test_output_folder_inside_the_source_is_refused():
    """No output folder inside the source image, checked before the folder
    is made (review GUI-01: the GUI used to create it first)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        for bad in (src_path, src_path / "crops"):
            try:
                vc.check_output_folder(image, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {bad} as output folder")
        vc.check_output_folder(image, tmp)                 # next to it: fine
        vc.check_output_folder(image, Path(tmp) / "crops")
        # The command line checks before it creates the folder.
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv([vc.CropBox("a", {"x": (5, 10)})], csv_path, image)
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            code = vc.main([str(src_path), str(csv_path),
                            str(src_path / "crops")])
        assert code == 2 and not (src_path / "crops").exists()


def test_command_line():
    """The command line checks every box before it writes the first, says
    what is wrong in one line, and returns a failure code that a cluster
    job can see (review CORE-06, RUN-02: an unknown box name returned 0, and
    a bad second box was found only after the first was written)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.zarr"
        make_test_image(src)
        image = vc.open_image(src)
        out = tmp / "crops"

        def run(boxes, *options):
            csv_path = tmp / "boxes.csv"
            vc.save_boxes_csv(boxes, csv_path, image)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                code = vc.main([str(src), str(csv_path), str(out),
                                "--min-size", "4", *options])
            return code, stdout.getvalue(), stderr.getvalue()

        good = vc.CropBox("good", {"x": (5, 10)})
        outside = vc.CropBox("outside", {"z": (500, 600)})
        code, _, err = run([good, outside])
        assert code == 2 and "outside" in err and not out.exists(), err
        code, _, err = run([good], "--names", "nope")
        assert code == 2 and "no box called 'nope'" in err, err
        code, text, _ = run([good], "--dry-run")
        assert code == 0 and "nothing was written" in text and not out.exists()
        code, text, _ = run([good])
        assert code == 0 and "checked OK" in text, text
        code, _, err = run([good])                      # exists already
        assert code == 2 and "--overwrite" in err, err
        code, _, _ = run([good], "--overwrite")
        assert code == 0
        # 0.10.0: --tiff LEVEL also saves that level as TIFF (checked)
        code, text, _ = run([good], "--tiff", "1", "--dry-run", "--overwrite")
        assert code == 0 and "src_good_level1.tif" in text, text
        assert not (out / "src_good_level1.tif").exists()
        code, text, _ = run([good], "--tiff", "1", "--overwrite")
        assert code == 0 and "src_good_level1.tif: level 1" in text, text
        assert "100 %" in text and (out / "src_good_level1.tif").exists()
        code, _, err = run([good], "--tiff", "1", "--overwrite")
        assert code == 0, err                       # replaces its own TIFF
        code, text, _ = run([good], "--tiff", "9", "--overwrite")
        assert code == 0 and "level 9 asked" in text, text
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                run([good], "--tiff", "-1")
            except SystemExit as exit_:
                assert exit_.code == 2
            else:
                raise AssertionError("--tiff -1 was accepted")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = vc.main([str(tmp / "missing.zarr"), str(tmp / "boxes.csv"),
                            str(out)])
        assert code == 2 and "does not exist" in stderr.getvalue()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            try:
                vc.main(["--version"])
            except SystemExit as exit_:
                assert exit_.code == 0
        assert vc.__version__ in stdout.getvalue()


def test_command_line_runs_from_any_folder():
    """The user guide's command line, python <tool folder>/volume_cropper/
    core.py, works from the user's own folder with paths relative to it, and
    writes the crop there, not into the tool folder (2026-11-24: the new
    package layout first needed the tool folder as working folder).
    python -m volume_cropper.core still works from the tool folder."""
    tool = Path(__file__).absolute().parents[1]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")
        vc.save_boxes_csv([vc.CropBox("good", {"x": (5, 10)})],
                          tmp / "boxes.csv", vc.open_image(tmp / "src.zarr"))
        # As a file, from the user's folder, with relative paths: a real crop.
        done = subprocess.run(
            [sys.executable, str(tool / "volume_cropper" / "core.py"),
             "src.zarr", "boxes.csv", "crops", "--min-size", "4"],
            capture_output=True, text=True, cwd=tmp, env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "checked OK" in done.stdout, done.stdout
        assert any((tmp / "crops").iterdir())
        assert not (tool / "crops").exists()
        # With -m, from the tool folder (full paths): a dry run.
        done = subprocess.run(
            [sys.executable, "-m", "volume_cropper.core", str(tmp / "src.zarr"),
             str(tmp / "boxes.csv"), str(tmp / "more_crops"),
             "--min-size", "4", "--dry-run"],
            capture_output=True, text=True, cwd=tool, env=env)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "nothing was written" in done.stdout, done.stdout


def inside_polygon(points, polygon):
    """Independent reference: even-odd rule (ray casting) for points (N, 2)."""
    polygon = np.asarray(polygon, dtype=float)
    a, b = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for k in range(len(polygon)):
        (a0, b0), (a1, b1) = polygon[k], polygon[(k + 1) % len(polygon)]
        crosses = (a0 > a) != (a1 > a)
        with np.errstate(divide="ignore", invalid="ignore"):
            b_cross = b0 + (a - a0) * (b1 - b0) / (a1 - a0)
        inside ^= crosses & (b < b_cross)
    return inside


def distance_to_edge(points, polygon):
    """Distance from points (N, 2) to the nearest edge of a polygon."""
    polygon = np.asarray(polygon, dtype=float)
    points = np.asarray(points, dtype=float)
    distance = np.full(len(points), np.inf)
    for k in range(len(polygon)):
        a, b = polygon[k], polygon[(k + 1) % len(polygon)]
        t = np.clip((points - a) @ (b - a) / max((b - a) @ (b - a), 1e-12),
                    0.0, 1.0)
        nearest = a + t[:, None] * (b - a)
        distance = np.minimum(distance,
                              np.linalg.norm(points - nearest, axis=1))
    return distance


# A concave outline (an "L"), with corners between voxel centres so that no
# centre lies exactly on an edge.
L_SHAPE = [(5.13, 3.07), (5.13, 30.11), (12.09, 30.11), (12.09, 10.03),
           (30.21, 10.03), (30.21, 3.07)]


def test_outline_mask_matches_reference():
    level = vc.Level("0", None, scale=[1.0, 2.0, 0.25, 0.5],
                     translation=[0.0, 0.0, 1.0, -2.0])
    names = ["c", "z", "y", "x"]
    first, shape = [0, 3, 4, 2], (1, 5, 120, 70)
    mask = vc.outline_mask(level, names, first, shape, {("y", "x"): L_SHAPE})
    assert mask.shape == (1, 1, 120, 70)      # broadcasts over c and z
    yy, xx = np.meshgrid(1.0 + (4 + np.arange(120)) * 0.25,
                         -2.0 + (2 + np.arange(70)) * 0.5, indexing="ij")
    expected = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                              L_SHAPE).reshape(120, 70)
    np.testing.assert_array_equal(mask[0, 0], expected)
    assert 0 < expected.sum() < expected.size
    # The same outline given as (x, y) is the transposed case.
    swapped = [(x, y) for y, x in L_SHAPE]
    mask_xy = vc.outline_mask(level, names, first, shape, {("x", "y"): swapped})
    np.testing.assert_array_equal(mask_xy, mask)


def test_masked_crop():
    """Outlines in two views, one polygon each: a voxel is kept only inside
    both (since 0.7.0; 0.5.0-0.6.0 kept a voxel inside either)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        data0 = make_test_image(src_path)    # scale z 2, y/x 0.25 um
        image = vc.open_image(src_path)
        outline_zx = [(20.5, 4.1), (60.7, 9.3), (30.3, 28.9)]   # a triangle
        box = vc.CropBox("lasso", {"z": (10, 70)},
                         outlines={("y", "x"): L_SHAPE, ("z", "x"): outline_zx})
        plan = vc.plan_crop(image, box, min_size=4)
        # The crop spans the overlap of what the two views allow: y from the
        # y-x outline, z from the z-x one, x from both.
        low, high = plan.snapped_ranges["x"]
        assert low <= 4.1 and high >= 28.9 and high - low <= 28.9 - 4.1 + 8
        low, high = plan.snapped_ranges["y"]
        assert low <= 5.13 and high >= 30.21 and high - low <= 30.21 - 5.13 + 8
        low, high = plan.snapped_ranges["z"]
        assert low <= 20.5 and high >= 60.7 and high - low <= 60.7 - 20.5 + 32
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr", workers=4)
        assert vc.verify_crop(image, plan, out) == []

        crop = vc.open_image(out).levels[0].array[...]
        s, e = plan.starts[0], plan.stops[0]
        source = data0[:, s[1]:e[1], s[2]:e[2], s[3]:e[3]]
        zc = (s[1] + np.arange(e[1] - s[1])) * 2.0     # voxel centres, um
        yc = (s[2] + np.arange(e[2] - s[2])) * 0.25
        xc = (s[3] + np.arange(e[3] - s[3])) * 0.25
        yy, xx = np.meshgrid(yc, xc, indexing="ij")
        keep_yx = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                                 L_SHAPE).reshape(yy.shape)
        zz, xx2 = np.meshgrid(zc, xc, indexing="ij")
        keep_zx = inside_polygon(np.column_stack([zz.ravel(), xx2.ravel()]),
                                 outline_zx).reshape(zz.shape)
        keep = keep_yx[None, None, :, :] & keep_zx[None, :, None, :]  # all
        np.testing.assert_array_equal(crop, np.where(keep, source, 0))
        assert keep.any() and not keep.all()
        prov = zarr.open_group(out, mode="r").attrs["crop_provenance"]
        assert prov["outlines"]["y,x"][0]["points"][0] == [5.13, 3.07]
        assert prov["outlines"]["y,x"][0]["slice"] is None
        assert prov["outlines_combine"] == "all"


def test_views_combine_inside_all():
    """Two views that disagree on a shared axis (a y-x outline longer in y
    than the z-y one): only the overlap is kept, and the crop is cut to it
    (this is what 0.5.0-0.6.0 got wrong for the user: they kept either)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)            # scale z 2, y/x 0.25 um
        image = vc.open_image(src_path)
        long_y = [(4.1, 5.1), (4.1, 25.1), (34.1, 25.1), (34.1, 5.1)]  # (y, x)
        short_y = [(20.3, 12.1), (20.3, 20.1), (60.3, 20.1), (60.3, 12.1)]  # (z, y)
        box = vc.CropBox("b", outlines={("y", "x"): long_y,
                                        ("z", "y"): short_y})
        plan = vc.plan_crop(image, box, n_levels=1)
        low, high = plan.snapped_ranges["y"]
        assert low <= 12.1 and high >= 20.1 and high - low < 9   # the overlap
        low, high = plan.snapped_ranges["z"]
        assert low <= 20.3 and high >= 60.3 and high - low < 44
        s, e = plan.starts[0], plan.stops[0]
        shape = tuple(b - a for a, b in zip(s, e))
        mask = vc.outline_mask(image.levels[0], image.axis_names, s, shape,
                               box.outlines, spatial=["z", "y", "x"])
        zc, yc, xc = ((s[i] + np.arange(e[i] - s[i])) * size
                      for i, size in ((1, 2.0), (2, 0.25), (3, 0.25)))
        yy, xx = np.meshgrid(yc, xc, indexing="ij")
        in_yx = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                               long_y).reshape(yy.shape)
        zz, yy2 = np.meshgrid(zc, yc, indexing="ij")
        in_zy = inside_polygon(np.column_stack([zz.ravel(), yy2.ravel()]),
                               short_y).reshape(zz.shape)
        expected = in_yx[None, None] & in_zy[None, :, :, None]
        np.testing.assert_array_equal(np.broadcast_to(mask, expected.shape),
                                      expected)
        assert expected.any()


# Three polygons in the y-x view, on the slices z = 4, 12 and 20 um: a
# rounded triangle, the concave L and a small square elsewhere.
KEYS_YX = {4.0: [(8.2, 6.1), (8.2, 20.3), (22.1, 13.2)],
           12.0: L_SHAPE,
           20.0: [(14.3, 18.1), (14.3, 26.2), (22.4, 26.2), (22.4, 18.1)]}


def tube_and_outlines(n: int = 100):
    """An irregular object and outlines drawn around it, for the shape
    through polygons in several views (since 0.7.0).

    The object (the truth) is a curved tube whose radius changes along it,
    like a tail, in a cube of n voxels of 1 um: a voxel is inside if it is
    no farther from the centre line than the radius there. The outlines are
    its exact edges on 3 y-x, 2 z-x and 2 y-z slices, as a person would draw
    them around it. Returns (truth as a (z, y, x) mask, outlines).
    """
    t = np.linspace(0, 1, 400)
    p0, p1, p2 = np.array([12, 30, 20]), np.array([50, 85, 45]), \
        np.array([88, 35, 82])
    line = ((1 - t)[:, None] ** 2 * p0 + 2 * ((1 - t) * t)[:, None] * p1
            + (t ** 2)[:, None] * p2)                              # (z, y, x)
    radius = 9 + 7 * np.sin(np.pi * t) + 2 * np.sin(5 * np.pi * t)
    voxels = np.indices((n, n, n)).reshape(3, -1).T
    distance, nearest = cKDTree(line).query(voxels)
    truth = (distance <= radius[nearest]).reshape(n, n, n)

    def edge(section):             # the longest edge of a 2D section
        contours = find_contours(np.pad(section, 1).astype(float), 0.5)
        return [tuple(p - 1) for p in max(contours, key=len)]

    outlines = {
        ("y", "x"): {float(z): edge(truth[z]) for z in (22, 50, 78)},
        ("z", "x"): {float(y): edge(truth[:, y, :]) for y in (45, 70)},
        ("y", "z"): {float(x): edge(truth[:, :, x].T) for x in (40, 65)}}
    return truth, outlines


def test_shape_through_outlines_in_three_views():
    """Polygons on slices of all three views make one 3D shape: it keeps
    nearly all of an irregular object and little else, passes through every
    polygon on its own slice, and ends at the outermost polygons."""
    truth, outlines = tube_and_outlines()
    n = truth.shape[0]
    level = vc.Level("0", None, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    zyx = ["z", "y", "x"]
    mask = np.broadcast_to(vc.outline_mask(level, zyx, [0, 0, 0], (n,) * 3,
                                           outlines, spatial=zyx), (n,) * 3)
    kept = (mask & truth).sum() / truth.sum()
    extra = (mask & ~truth).sum() / truth.sum()
    # Measured: 91.7 % kept, 2.3 % extra. Each view alone, interpolated
    # and combined as in the first 0.7.0 draft, kept 24 %.
    assert kept >= 0.88 and extra <= 0.05, (kept, extra)
    # On each polygon's own slice the shape is that polygon: voxels that
    # differ lie on its edge (within 1 voxel of it).
    for plane, keys in outlines.items():
        depth_axis = next(a for a in zyx if a not in plane)
        for depth, polygon in keys.items():
            cut = [slice(None)] * 3
            cut[zyx.index(depth_axis)] = int(depth)
            section = mask[tuple(cut)]
            if zyx.index(plane[0]) > zyx.index(plane[1]):
                section = section.T                   # as (plane[0], plane[1])
            a, b = np.indices(section.shape)
            points = np.column_stack([a.ravel(), b.ravel()])
            drawn = inside_polygon(points, polygon).reshape(section.shape)
            differ = points[(section != drawn).ravel()]
            assert len(differ) < 0.02 * drawn.sum(), (plane, depth)
            if len(differ):
                assert distance_to_edge(differ, polygon).max() <= 1.0
    # Nothing beyond the outermost polygons, along any axis.
    corners = np.concatenate([p for *_, p in vc._sliced_polygons(outlines,
                                                                  zyx)])
    for k in range(3):
        kept_at = np.flatnonzero(mask.any(axis=tuple(j for j in range(3)
                                                     if j != k)))
        assert kept_at.min() >= corners[:, k].min() - 1e-9
        assert kept_at.max() <= corners[:, k].max() + 1e-9
    # A block anywhere gives the same as that part of the whole (the shape
    # does not depend on where blocks start).
    part = vc.outline_mask(level, zyx, [30, 20, 40], (25, 30, 20), outlines,
                           spatial=zyx)
    np.testing.assert_array_equal(np.broadcast_to(part, (25, 30, 20)),
                                  mask[30:55, 20:50, 40:60])


def test_shape_ends_at_the_outermost_polygons():
    """Polygons on two y-x slices: the shape runs from the first to the
    last, and is each polygon on its slice; the crop is cut to them. One
    polygon on a slice alone reaches through the whole box (a prism)."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)                # scale z 2, y/x 0.25 um
        image = vc.open_image(src_path)
        keys = {4.0: KEYS_YX[4.0], 20.0: KEYS_YX[20.0]}
        level = image.levels[0]
        shape = (1, 20, 128, 128)                # z 0-38, y/x 0-31.75 um
        mask = np.broadcast_to(vc.outline_mask(
            level, image.axis_names, [0, 0, 0, 0], shape,
            {("y", "x"): keys}, spatial=["z", "y", "x"]), shape)[0]
        yy, xx = np.meshgrid(np.arange(128) * 0.25, np.arange(128) * 0.25,
                             indexing="ij")
        points = np.column_stack([yy.ravel(), xx.ravel()])
        for j, depth in ((2, 4.0), (10, 20.0)):          # z = 4 and 20 um
            drawn = inside_polygon(points, keys[depth]).reshape(128, 128)
            differ = points[(mask[j] != drawn).ravel()]
            assert len(differ) < 0.02 * drawn.sum()
            if len(differ):                  # ties: centres on the edge
                assert distance_to_edge(differ, keys[depth]).max() <= 0.1
        assert not mask[:2].any() and not mask[11:].any()     # ends there
        assert all(mask[j].any() for j in range(3, 10))       # in between
        plan = vc.plan_crop(image,
                            vc.CropBox("b", outlines={("y", "x"): keys}),
                            n_levels=1)
        low, high = plan.snapped_ranges["z"]
        assert low <= 4 and high >= 20 and high - low <= 20
        # One polygon on a slice, alone: a prism through the whole box.
        one = {("y", "x"): {12.0: L_SHAPE}}
        prism = vc.outline_mask(level, image.axis_names, [0, 0, 0, 0], shape,
                                one, spatial=["z", "y", "x"])
        assert prism.shape == (1, 1, 128, 128)            # the same on every z
        np.testing.assert_array_equal(
            prism[0, 0], inside_polygon(points, L_SHAPE).reshape(128, 128))
        plan = vc.plan_crop(image, vc.CropBox("p", outlines=one), n_levels=1)
        assert plan.shapes[0][1] == level.array.shape[1]     # every z slice


def test_same_polygon_far_apart_gives_a_prism():
    """The same square on two slices three widths apart: the shape is that
    square on every slice between them (a draft of 0.7.0, with the linear
    kernel, narrowed it to nothing in the middle)."""
    level = vc.Level("0", None, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    zyx = ["z", "y", "x"]
    square = [(40.5, 40.5), (40.5, 60.5), (60.5, 60.5), (60.5, 40.5)]
    mask = np.broadcast_to(vc.outline_mask(
        level, zyx, [0, 0, 0], (100, 100, 100),
        {("y", "x"): {20.0: square, 80.0: square}}, spatial=zyx),
        (100, 100, 100))
    expected = np.zeros((100, 100), dtype=bool)
    expected[41:61, 41:61] = True             # centres inside the square
    for z in range(20, 81):
        np.testing.assert_array_equal(mask[z], expected, err_msg=f"z={z}")
    assert not mask[:20].any() and not mask[81:].any()


def test_margin_grows_and_shrinks_exactly():
    """A margin adds every voxel whose centre lies within that distance
    (um) of the outlined shape, or removes every voxel within it of the
    outside, also across block edges and with anisotropic voxels; checked
    with a k-d tree (not scipy's distance transform)."""
    level = vc.Level("0", None, [2.0, 0.5, 0.5], [0.0, 0.0, 0.0])
    zyx = ["z", "y", "x"]
    outlines = {("y", "x"): {4.0: KEYS_YX[4.0], 20.0: L_SHAPE}}
    first, shape = [1, 12, 14], (9, 40, 36)     # cuts through the shape
    rim = [4, 10, 10]                           # wider than the margins
    big_first = [f - r for f, r in zip(first, rim)]
    big_shape = [n + 2 * r for n, r in zip(shape, rim)]
    base = np.broadcast_to(vc.outline_mask(level, zyx, big_first, big_shape,
                                           outlines, spatial=zyx), big_shape)
    where = np.indices(big_shape).reshape(3, -1).T
    um = (where + big_first) * np.array(level.scale)        # voxel centres
    inner = np.all((where >= rim) & (where < np.array(rim) + shape), axis=1)
    for margin in (1.5, 2.0, -1.0, -2.5):
        got = np.broadcast_to(vc.outline_mask(level, zyx, first, shape,
                                              outlines, spatial=zyx,
                                              margin=margin), shape)
        if margin > 0:
            distance = cKDTree(um[base.ravel()]).query(um[inner])[0]
            expected = distance <= margin + 1e-9
        else:
            distance = cKDTree(um[~base.ravel()]).query(um[inner])[0]
            expected = base.ravel()[inner] & (distance > -margin)
        np.testing.assert_array_equal(got.ravel(), expected)
        assert got.any() and not got.all()
    # Margins along z (2 um voxels) count in um too: one z step is 2 um.
    grown = np.broadcast_to(vc.outline_mask(level, zyx, [0, 0, 0],
                                            (14, 60, 60), outlines,
                                            spatial=zyx, margin=2.0),
                            (14, 60, 60))
    assert grown[1].any() and not grown[0].any()       # z = 2 um, not 0 um


def test_shaped_crop_with_margin():
    """A crop with polygons on slices of the y-x and z-x views and a
    margin: the crop is cut to the polygons plus the margin, and every
    level's voxels are the source's where outline_mask keeps them and 0
    elsewhere; the margin is recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)                # scale z 2, y/x 0.25 um
        image = vc.open_image(src_path)
        outlines = {("y", "x"): {12.0: L_SHAPE,
                                 36.0: [(10.1, 8.1), (10.1, 22.2),
                                        (24.3, 22.2), (24.3, 8.1)]},
                    ("z", "x"): {14.0: [(8.3, 5.2), (8.3, 26.1),
                                        (40.2, 24.0), (40.2, 9.1)]}}
        box = vc.CropBox("shaped", outlines=outlines, margin=4)  # 4 x 0.25 um
        plan = vc.plan_crop(image, box, min_size=4)
        assert len(plan.level_indices) >= 2
        low, high = plan.snapped_ranges["z"]       # 8.3-40.2, plus 1 um
        assert low <= 7.3 and high >= 41.2 and high - low <= 41.2 - 7.3 + 32
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr", workers=4)
        assert vc.verify_crop(image, plan, out) == []
        cropped = vc.open_image(out)
        for k, index in enumerate(plan.level_indices):
            level = image.levels[index]
            s, e = plan.starts[k], plan.stops[k]
            shape = tuple(b - a for a, b in zip(s, e))
            keep = vc.outline_mask(level, image.axis_names, s, shape, outlines,
                                   spatial=["z", "y", "x"], margin=1.0)
            source = level.array[tuple(slice(a, b) for a, b in zip(s, e))]
            np.testing.assert_array_equal(cropped.levels[k].array[...],
                                          np.where(keep, source, 0))
            if k == 0:
                assert keep.any() and not keep.all()
                bare = vc.outline_mask(level, image.axis_names, s, shape,
                                       outlines, spatial=["z", "y", "x"])
                assert keep.sum() > 1.05 * bare.sum()    # the margin grew it
        prov = zarr.open_group(out, mode="r").attrs["crop_provenance"]
        assert prov["outline_margin_voxels"] == 4
        assert prov["outline_margin"] == 1.0
        assert [k["slice"] for k in prov["outlines"]["y,x"]] == [12.0, 36.0]


def test_csv_margin_round_trip():
    """A box's outline margin is saved in its own column (only when some
    box has one) and read back."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        boxes = [vc.CropBox("grown", outlines={("y", "x"): KEYS_YX},
                            margin=2.5),
                 vc.CropBox("shrunk", outlines={("y", "x"): L_SHAPE},
                            margin=-3),
                 vc.CropBox("plain", {"z": (1.5, 20)})]
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv(boxes, csv_path, image)
        header = csv_path.read_text().splitlines()[0]
        assert "outline_margin_voxels" in header.split(",")
        assert vc.load_boxes_csv(csv_path) == boxes


def test_silhouettes_reach_through_their_depth():
    """Polygons drawn on a projection (silhouettes, key = the projected
    depth range): within that range a voxel must lie inside them, outside
    it they say nothing, and the crop is cut to the range. Silhouettes of
    several views keep what lies inside all of them; with polygons on
    slices, also inside the shape through those. They are saved in boxes
    files ("z=10..30: ...") and in the crop's record."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)                # scale z 2, y/x 0.25 um
        image = vc.open_image(src_path)
        level, names = image.levels[0], image.axis_names
        zyx = ["z", "y", "x"]
        shape = (1, 30, 128, 128)                # z 0-58, y/x 0-31.75 um
        one = {("y", "x"): {(10.0, 30.0): L_SHAPE}}
        mask = np.broadcast_to(vc.outline_mask(level, names, [0, 0, 0, 0],
                                               shape, one, spatial=zyx),
                               shape)[0]
        yy, xx = np.meshgrid(np.arange(128) * 0.25, np.arange(128) * 0.25,
                             indexing="ij")
        points = np.column_stack([yy.ravel(), xx.ravel()])
        inside = inside_polygon(points, L_SHAPE).reshape(128, 128)
        for j in range(30):
            expected = inside if 10 <= 2 * j <= 30 else np.ones_like(inside)
            np.testing.assert_array_equal(mask[j], expected, err_msg=str(j))
        plan = vc.plan_crop(image, vc.CropBox("s", outlines=one), n_levels=1)
        low, high = plan.snapped_ranges["z"]
        assert low <= 10 and high >= 30 and high - low <= 24
        # silhouettes in two views: inside both (the "visual hull")
        side = [(4.1, 5.1), (4.1, 25.1), (44.3, 25.1), (44.3, 5.1)]  # (z, x)
        two = {("y", "x"): {(0.0, 78.0): L_SHAPE},
               ("z", "x"): {(0.0, 37.5): side}}
        mask = np.broadcast_to(vc.outline_mask(level, names, [0, 0, 0, 0],
                                               shape, two, spatial=zyx),
                               shape)[0]
        zz, xx2 = np.meshgrid(np.arange(30) * 2.0, np.arange(128) * 0.25,
                              indexing="ij")
        in_side = inside_polygon(np.column_stack([zz.ravel(), xx2.ravel()]),
                                 side).reshape(30, 128)
        np.testing.assert_array_equal(mask, inside[None] & in_side[:, None])
        # with polygons on two slices: the shape through them, inside the
        # silhouette as well
        mixed = {("y", "x"): {(0.0, 78.0): KEYS_YX[4.0], 12.0: KEYS_YX[12.0],
                              20.0: KEYS_YX[12.0]}}
        shaped = {("y", "x"): {12.0: KEYS_YX[12.0], 20.0: KEYS_YX[12.0]}}
        both = [np.broadcast_to(vc.outline_mask(
            level, names, [0, 0, 0, 0], shape, o, spatial=zyx), shape)[0]
            for o in (mixed, shaped)]
        in_triangle = inside_polygon(points, KEYS_YX[4.0]).reshape(128, 128)
        np.testing.assert_array_equal(both[0], both[1] & in_triangle[None])
        assert both[0].any()
        # saved and read back
        box = vc.CropBox("hull", outlines=two)
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv([box], csv_path, image)
        assert "z=0..78: " in csv_path.read_text()
        assert vc.load_boxes_csv(csv_path) == [box]
        out = vc.write_crop(image, vc.plan_crop(image, box, n_levels=1),
                            Path(tmp) / "hull.zarr")
        record = zarr.open_group(out, mode="r").attrs["crop_provenance"]
        assert record["outlines"]["y,x"][0]["projected_range"] == [0.0, 78.0]
        assert record["outlines"]["y,x"][0]["slice"] is None


def test_holes_in_the_shape_are_filled():
    """Polygons of different views that disagree (here the side views
    drawn 12 % too small) force the shape through both and leave holes;
    they are filled on the shape's grid, so that no plane of it along any
    axis has a hole and there is no closed hollow, and far fewer slices of
    voxels have one than without filling."""
    truth, outlines = tube_and_outlines()
    for plane in (("z", "x"), ("y", "z")):
        for depth, points in outlines[plane].items():
            p = np.asarray(points)
            centre = p.mean(axis=0)
            outlines[plane][depth] = [tuple(q) for q in
                                      centre + 0.88 * (p - centre)]
    zyx = ("z", "y", "x")
    frozen = vc._frozen(vc._sliced_only(vc.normalize_outlines(outlines)))
    origin, step, field, low, high = vc._shape_field(frozen, zyx,
                                                     vc.FIELD_POINTS)
    first = np.round((low - origin) / step).astype(int)
    last = np.floor((high - origin) / step + 1e-6).astype(int)
    inside = np.zeros(field.shape, dtype=bool)
    box = tuple(slice(a, b + 1) for a, b in zip(first, last))
    inside[box] = field[box] < 0
    for axis in range(3):
        for plane in np.moveaxis(inside, axis, 0):
            assert not (ndi.binary_fill_holes(plane) & ~plane).any()
    assert not (ndi.binary_fill_holes(inside) & ~inside).any()

    def holed_slices():
        level = vc.Level("0", None, [1.0] * 3, [0.0] * 3)
        kept = np.broadcast_to(vc.outline_mask(
            level, list(zyx), [0, 0, 0], truth.shape, outlines,
            spatial=list(zyx)), truth.shape)
        return sum(int((ndi.binary_fill_holes(s) & ~s).any())
                   for axis in range(3) for s in np.moveaxis(kept, axis, 0))

    filled = holed_slices()
    real = vc._fill_holes
    try:
        vc._fill_holes = lambda *args: None
        vc._shape_field.cache_clear()
        unfilled = holed_slices()
    finally:
        vc._fill_holes = real
        vc._shape_field.cache_clear()
    assert unfilled > 0 and filled <= unfilled // 2, (filled, unfilled)


def test_crossings_and_disagreements():
    """Where polygons of other views cross a view's slice (the marks in
    the window), and how far apart two views' polygons are where their
    slices cross (the warning)."""
    zyx = ["z", "y", "x"]
    square_yx = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    side_zx = [(0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0)]
    outlines = vc.normalize_outlines({("y", "x"): {10.0: square_yx},
                                      ("z", "x"): {5.0: side_zx}})
    # on the XY slice z = 10, the z-x polygon (at y = 5) crosses at x 0, 10
    points = vc.crossing_points(outlines, zyx, ("y", "x"), 10.0)
    np.testing.assert_allclose(sorted(map(tuple, points)),
                               [(5.0, 0.0), (5.0, 10.0)])
    # the same in a view stored the other way round (x, y)
    points = vc.crossing_points(outlines, zyx, ("x", "y"), 10.0)
    np.testing.assert_allclose(sorted(map(tuple, points)),
                               [(0.0, 5.0), (10.0, 5.0)])
    assert vc.crossing_points(outlines, zyx, ("y", "x"), 30.0).shape == (0, 2)
    # (0.16.0) with the plane each mark comes from, for its colour
    points, planes = vc.crossing_points(outlines, zyx, ("y", "x"), 10.0,
                                        with_planes=True)
    assert len(points) == 2 and planes == [("z", "x"), ("z", "x")]
    # they agree: the edges meet at the same points
    (_, _, _, _, gap), = vc.outline_disagreements(outlines, zyx)
    assert gap < 1e-9
    # the side view drawn 3 µm too far in x
    moved = [(z, x + 3.0) for z, x in side_zx]
    outlines[("z", "x")] = {5.0: moved}
    (plane_a, slice_a, plane_b, slice_b, gap), = vc.outline_disagreements(
        outlines, zyx)
    assert abs(gap - 3.0) < 1e-9
    assert {(plane_a, slice_a), (plane_b, slice_b)} == {
        (("y", "x"), 10.0), (("z", "x"), 5.0)}
    # a side view that does not reach the XY slice at all: 2 µm short
    outlines[("z", "x")] = {5.0: [(0.0, 0.0), (0.0, 10.0), (8.0, 10.0),
                                  (8.0, 0.0)]}
    (*_, gap), = vc.outline_disagreements(outlines, zyx)
    assert abs(gap - 2.0) < 1e-9


def test_bad_outlines_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        apart = [(50.1, 0.5), (50.1, 4.5), (60.1, 4.5), (60.1, 0.5)]  # (z, y)
        bad = [vc.CropBox("a", outlines={("y", "y"): L_SHAPE}),
               vc.CropBox("b", outlines={("y", "x"): L_SHAPE[:2]}),
               vc.CropBox("c", {"y": (100, 110)}, outlines={("y", "x"): L_SHAPE}),
               vc.CropBox("d", outlines={("y", "x"): {None: L_SHAPE,
                                                      10.0: L_SHAPE}}),
               vc.CropBox("e", outlines={("y", "x"): L_SHAPE,
                                         ("x", "y"): L_SHAPE}),
               vc.CropBox("f", outlines={("y", "x"): L_SHAPE,     # no overlap
                                         ("z", "y"): apart})]
        for box in bad:
            try:
                vc.plan_crop(image, box)
            except ValueError:
                pass
            else:
                raise AssertionError(f"outline of box {box.name} accepted")


def test_csv_outlines_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path)
        image = vc.open_image(src_path)
        boxes = [vc.CropBox("plain", {"z": (1.5, 20)}),
                 vc.CropBox("lasso", {"z": (1.5, 20)}, channels=[0],
                            outlines={("y", "x"): L_SHAPE}),
                 vc.CropBox("slices", outlines={("y", "x"): KEYS_YX})]
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv(boxes, csv_path, image)
        text = csv_path.read_text()
        assert text.splitlines()[0].endswith(",outline_y_x")
        assert "z=4: " in text and " | z=12: " in text
        loaded = vc.load_boxes_csv(csv_path)
        assert loaded == boxes
        # A boxes file from before 0.7.0: one polygon per view, no slice.
        assert loaded[1].outlines == {("y", "x"): {
            None: [tuple(p) for p in L_SHAPE]}}


def test_crop_of_a_crop_keeps_the_chain():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        data0 = make_test_image(src_path)
        image = vc.open_image(src_path)
        first = vc.write_crop(image, vc.plan_crop(
            image, vc.CropBox("big", {"z": (10, 60), "y": (2, 30)}),
            min_size=4), Path(tmp) / "big.zarr")
        crop = vc.open_image(first)
        plan = vc.plan_crop(crop, vc.CropBox("small", {"z": (20, 40)}),
                            min_size=4)
        second = vc.write_crop(crop, plan, Path(tmp) / "small.zarr")
        assert vc.verify_crop(crop, plan, second) == []
        record = zarr.open_group(second, mode="r").attrs["crop_provenance"]
        assert record["software"]["volume_cropper"] == vc.__version__
        assert record["source"] == str(first)
        assert record["source_provenance"] == crop.attrs["crop_provenance"]
        assert record["source_provenance"]["source"] == str(src_path)
        # Positions stay those of the original image: the second crop's
        # translation points at exactly the matching original voxels.
        level = vc.open_image(second).levels[0]
        scale0 = image.levels[0].scale
        start = [round(t / s) for t, s in zip(level.translation[1:],
                                              scale0[1:])]
        stop = [a + n for a, n in zip(start, level.array.shape[1:])]
        np.testing.assert_array_equal(
            level.array[...], data0[:, start[0]:stop[0], start[1]:stop[1],
                                    start[2]:stop[2]])
        low, high = vc.index_range_to_physical(start[0], stop[0], scale0[1],
                                               0.0)
        assert low <= 20 and high >= 40          # the requested z range


# ---------------------------------------------------------------------------
# OME-Zarr 0.4 (Zarr v2) and images without a channel axis (since 0.8.0)
# ---------------------------------------------------------------------------

def read_json(path):
    """A Zarr v2 metadata file (.zarray, .zattrs, .zgroup) as a dict."""
    return json.loads(Path(path).read_text())


def without_shape(meta):
    return {key: value for key, value in meta.items() if key != "shape"}


def test_v04_image_opens():
    """An OME-Zarr 0.4 image opens with its voxel sizes, positions and
    channel names, and says which format it is."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path, version="0.4",
                        translation=(0, 100.0, 50.0, -20.0))
        image = vc.open_image(src_path)
        assert (image.zarr_format, image.ome_version) == (2, "0.4")
        assert image.format_name == "OME-Zarr 0.4 (Zarr v2)"
        assert image.channel_labels == ["ch0", "ch1", "ch2"]
        assert [level.scale for level in image.levels] == [
            [s * n for s, n in zip((1.0, 2.0, 0.25, 0.25), f)]
            for f in FACTORS]
        assert all(level.translation == [0.0, 100.0, 50.0, -20.0]
                   for level in image.levels)
        text = image.describe()
        assert "OME-Zarr 0.4 (Zarr v2)" in text and "shards" not in text


def test_v04_crop_keeps_format_and_version():
    """A crop of a 0.4 image is a 0.4 image in the source's format: Zarr v2,
    the same .zarray settings on every level (chunks, compressor, nested
    chunk files), the OME metadata directly in the attributes, the right
    positions, channels and voxels; and an independent reader (ome-zarr-py)
    reads it the same way."""
    from ome_zarr.format import FormatV04
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        data0 = make_test_image(src_path, version="0.4",
                                translation=(0, 100.0, 50.0, -20.0))
        image = vc.open_image(src_path)
        box = vc.CropBox("b1", {"z": (120, 160), "y": (55, 70)},
                         channels=[0, 2])
        plan = vc.plan_crop(image, box, min_size=4)
        assert len(plan.level_indices) > 1
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr",
                            channel_labels=["nuclei", "marker"])
        assert vc.verify_crop(image, plan, out,
                              channel_labels=["nuclei", "marker"]) == []

        # Zarr v2 throughout, and nothing of Zarr v3.
        assert read_json(out / ".zgroup") == {"zarr_format": 2}
        assert not list(out.rglob("zarr.json"))
        for k, level_index in enumerate(plan.level_indices):
            path = image.levels[level_index].path
            src_meta = read_json(src_path / path / ".zarray")
            out_meta = read_json(out / path / ".zarray")
            assert out_meta["shape"] == list(plan.shapes[k])
            assert without_shape(out_meta) == without_shape(src_meta)
        assert (out / "0" / "0" / "0" / "0" / "0").is_file()  # nested chunks

        # OME 0.4 metadata: directly in the attributes, not under "ome".
        attrs = read_json(out / ".zattrs")
        assert "ome" not in attrs
        assert attrs["multiscales"][0]["version"] == "0.4"
        crop = vc.open_image(out)
        assert crop.format_name == image.format_name
        for k, level in enumerate(crop.levels):
            source = image.levels[plan.level_indices[k]]
            assert level.scale == source.scale
            expected = [t if i == 0 else round(t + a * s, 9)
                        for i, (t, a, s) in enumerate(zip(
                            source.translation, plan.starts[k], source.scale))]
            assert level.translation == expected
        assert [c["label"] for c in attrs["omero"]["channels"]] == [
            "nuclei", "marker"]
        record = attrs["crop_provenance"]
        assert record["source_format"] == "OME-Zarr 0.4 (Zarr v2)"
        assert record["channels"] == [0, 2] and record["complete"] is True

        # The voxels, on every level.
        a, b = plan.starts[0], plan.stops[0]
        level0 = crop.levels[0].array[...]
        np.testing.assert_array_equal(
            level0, data0[[0, 2], a[1]:b[1], a[2]:b[2], a[3]:b[3]])
        for k, level in enumerate(crop.levels):
            np.testing.assert_array_equal(level.array[...],
                                          strided(level0, FACTORS[k]))

        # An independent reader sees the same image.
        node = list(Reader(parse_url(str(out), fmt=FormatV04()))())[0]
        assert len(node.data) == len(crop.levels)
        for k, level in enumerate(crop.levels):
            np.testing.assert_array_equal(np.asarray(node.data[k]),
                                          level.array[...])
            assert node.metadata["coordinateTransformations"][k] == [
                {"type": "scale", "scale": level.scale},
                {"type": "translation", "translation": level.translation}]


def test_images_without_a_channel_axis():
    """Images with only z, y and x (like eubi-bridge's single-channel
    output), in OME-Zarr 0.5 and 0.4: crops, also shaped by an outline,
    hold the right voxels and pass the check."""
    for version in ("0.5", "0.4", "0.6"):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "src.zarr"
            data0 = make_test_image(src_path, shape0=(40, 150, 130),
                                    factors=[(1, 1, 1), (1, 4, 4), (2, 8, 8)],
                                    scale0=(2.0, 0.25, 0.25), axes="zyx",
                                    version=version)
            image = vc.open_image(src_path)
            assert image.channel_axis is None and image.channel_labels == []
            assert image.range_axes == ["z", "y", "x"]
            box = vc.CropBox("b", {"z": (10, 50), "y": (5, 25), "x": (3, 20)})
            plan = vc.plan_crop(image, box, min_size=4)
            out = vc.write_crop(image, plan, Path(tmp) / "plain.zarr")
            assert vc.verify_crop(image, plan, out) == []
            s, e = plan.starts[0], plan.stops[0]
            np.testing.assert_array_equal(
                vc.open_image(out).levels[0].array[...],
                data0[s[0]:e[0], s[1]:e[1], s[2]:e[2]])

            # Shaped by an L-shaped outline in the y-x view.
            shaped = vc.CropBox("L", {"z": (10, 50)},
                                outlines={("y", "x"): L_SHAPE})
            plan = vc.plan_crop(image, shaped, min_size=4)
            out = vc.write_crop(image, plan, Path(tmp) / "shaped.zarr")
            assert vc.verify_crop(image, plan, out) == []
            s, e = plan.starts[0], plan.stops[0]
            yc = (s[1] + np.arange(e[1] - s[1])) * 0.25     # voxel centres
            xc = (s[2] + np.arange(e[2] - s[2])) * 0.25
            yy, xx = np.meshgrid(yc, xc, indexing="ij")
            keep = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                                  L_SHAPE).reshape(yy.shape)
            source = data0[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
            np.testing.assert_array_equal(
                vc.open_image(out).levels[0].array[...],
                np.where(keep[None], source, 0))
            assert keep.any() and not keep.all()


def test_v04_time_lapse():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        factors = [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)]
        data0 = make_test_image(src_path, shape0=(5, 2, 10, 40, 40),
                                factors=factors, axes="tczyx",
                                scale0=(30.0, 1.0, 2.0, 0.5, 0.5),
                                version="0.4")
        image = vc.open_image(src_path)
        # t in minutes: frames 1-3 (centres at 30, 60, 90 min).
        plan = vc.plan_crop(image, vc.CropBox("b", {"t": (30, 90)},
                                              channels=[1]), min_size=4)
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr")
        assert vc.verify_crop(image, plan, out) == []
        crop = vc.open_image(out)
        assert crop.ome_version == "0.4"
        np.testing.assert_array_equal(crop.levels[0].array[...],
                                      data0[1:4, 1:2])


def test_v04_masked_crop_in_many_blocks():
    """A 0.4 crop copied in small blocks (4 chunks each, 4 workers at once):
    every voxel is right, and chunks entirely outside the outline are not
    written at all (a 0.4 crop is one file per chunk) and read as 0."""
    original = vc.BLOCK_BYTES
    vc.BLOCK_BYTES = 4 * 8 * 8 * 8 * 2       # 4 chunks of 8 x 8 x 8 uint16
    try:
        _masked_crop_in_many_blocks()
    finally:
        vc.BLOCK_BYTES = original


def _masked_crop_in_many_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        data0 = make_test_image(src_path, version="0.4")
        image = vc.open_image(src_path)
        box = vc.CropBox("L", {"z": (10, 70)}, outlines={("y", "x"): L_SHAPE})
        plan = vc.plan_crop(image, box, min_size=4)
        counts = []
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr", workers=4,
                            progress=lambda done, total: counts.append(total))
        assert counts[0] > 10 * len(plan.level_indices)     # many blocks
        assert vc.verify_crop(image, plan, out) == []

        s, e = plan.starts[0], plan.stops[0]
        yc = (s[2] + np.arange(e[2] - s[2])) * 0.25         # voxel centres
        xc = (s[3] + np.arange(e[3] - s[3])) * 0.25
        yy, xx = np.meshgrid(yc, xc, indexing="ij")
        keep = inside_polygon(np.column_stack([yy.ravel(), xx.ravel()]),
                              L_SHAPE).reshape(yy.shape)
        source = data0[:, s[1]:e[1], s[2]:e[2], s[3]:e[3]]
        np.testing.assert_array_equal(vc.open_image(out).levels[0].array[...],
                                      np.where(keep, source, 0))

        level0 = zarr.open_array(out / "0", mode="r")
        n_chunks = int(np.prod([-(-n // c) for n, c in
                                zip(level0.shape, level0.chunks)]))
        n_files = sum(1 for p in (out / "0").rglob("*")
                      if p.is_file() and not p.name.startswith("."))
        assert 0 < n_files < n_chunks, (n_files, n_chunks)


def test_v04_interrupted_crop_overwrite_and_chain():
    """For 0.4 too: an interrupted crop has no OME metadata and says so when
    opened; it can be replaced; a folder the tool did not write is never
    replaced; and a crop of a crop stays 0.4 and keeps the chain back to the
    original image."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        make_test_image(src_path, version="0.4")
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("big", {"z": (10, 60),
                                                      "y": (2, 30)}),
                            min_size=4)
        out = Path(tmp) / "big.zarr"
        cancel = threading.Event()
        cancel.set()
        try:
            vc.write_crop(image, plan, out, cancel_event=cancel)
        except vc.CropCancelled:
            pass
        else:
            raise AssertionError("cancel_event was ignored")
        attrs = read_json(out / ".zattrs")
        assert "multiscales" not in attrs
        assert attrs["crop_provenance"]["complete"] is False
        try:
            vc.open_image(out)
        except ValueError as err:
            assert "unfinished crop" in str(err), err
        else:
            raise AssertionError("an unfinished crop must not open")
        vc.write_crop(image, plan, out, overwrite=True)
        assert vc.verify_crop(image, plan, out) == []

        other = Path(tmp) / "other.zarr"
        zarr.open_group(other, mode="w", zarr_format=2).create_array(
            "0", shape=(4,), dtype="uint8")
        try:
            vc.write_crop(image, plan, other, overwrite=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrote a folder it did not create")

        crop = vc.open_image(out)
        small = vc.plan_crop(crop, vc.CropBox("small", {"z": (20, 40)}),
                             min_size=4)
        second = vc.write_crop(crop, small, Path(tmp) / "small.zarr")
        assert vc.verify_crop(crop, small, second) == []
        assert vc.open_image(second).ome_version == "0.4"
        record = read_json(second / ".zattrs")["crop_provenance"]
        assert record["source_provenance"]["source"] == str(src_path)


def test_v04_unusual_formats_are_copied():
    """Zarr v2 settings that eubi-bridge does not use but other software
    may: no compression, and chunk files named "0.0.0" instead of nested
    folders (the "." separator). The crop copies them."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "src.zarr"
        root = zarr.open_group(src_path, mode="w-", zarr_format=2)
        data = np.random.default_rng(1).integers(0, 1000, (20, 30, 40),
                                                 dtype=np.uint16)
        arr = root.create_array(
            "0", shape=data.shape, dtype=data.dtype, chunks=(8, 8, 8),
            compressors=None, fill_value=0,
            chunk_key_encoding={"name": "v2", "separator": "."})
        arr[...] = data
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "micrometer"}
                     for a in "zyx"],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]
        image = vc.open_image(src_path)
        plan = vc.plan_crop(image, vc.CropBox("b", {"x": (10, 30)}))
        out = vc.write_crop(image, plan, Path(tmp) / "out.zarr")
        assert vc.verify_crop(image, plan, out) == []
        src_meta, out_meta = (read_json(p / "0" / ".zarray")
                              for p in (src_path, out))
        assert out_meta["compressor"] is None
        assert out_meta["dimension_separator"] == "."
        assert without_shape(out_meta) == without_shape(src_meta)
        assert (out / "0" / "0.0.0").is_file()
        np.testing.assert_array_equal(
            vc.open_image(out).levels[0].array[...],
            data[tuple(slice(a, b) for a, b in zip(plan.starts[0],
                                                   plan.stops[0]))])


def test_copy_blocks():
    """Without shards, one copy task writes a block of whole chunks (so no
    two tasks write the same chunk file) of at most BLOCK_BYTES, grown only
    along the spatial axes."""
    chunks = (1, 64, 64, 128)                          # c, z, y, x
    block = vc._copy_block(chunks, (3, 584, 1856, 1856), np.uint16, [1, 2, 3])
    assert block[0] == 1
    assert all(b % c == 0 for b, c in zip(block, chunks))
    assert vc.BLOCK_BYTES / 2 < vc.nbytes(block, np.uint16) <= vc.BLOCK_BYTES
    # A small array: one block covers it.
    small = vc._copy_block(chunks, (3, 100, 100, 100), np.uint16, [1, 2, 3])
    assert all(b >= n for b, n in zip(small[1:], (100, 100, 100)))
    # Chunks larger than BLOCK_BYTES are copied one by one.
    assert vc._copy_block((512, 512, 512), (2000,) * 3, np.uint16,
                          [0, 1, 2]) == (512, 512, 512)


def test_unsupported_images_are_refused_clearly():
    """Images Volume Cropper cannot read are refused with a message that
    says why: not OME-Zarr, OME-Zarr older than 0.4, axes without a type, a
    voxel size stored in a separate array, and a bioformats2raw collection
    (whose images open from its numbered subfolders)."""
    def refused(path, words):
        try:
            vc.open_image(path)
        except (ValueError, NotImplementedError) as err:
            assert words in str(err), str(err)
        else:
            raise AssertionError(f"{path.name} must be refused")

    small = dict(shape0=(10, 20, 20), factors=[(1, 1, 1)],
                 scale0=(1.0, 1.0, 1.0), axes="zyx", version="0.4")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def edited(name, edit):
            """A small 0.4 image with its multiscales metadata changed."""
            path = tmp / name
            make_test_image(path, **small)
            attrs = read_json(path / ".zattrs")
            edit(attrs["multiscales"][0])
            (path / ".zattrs").write_text(json.dumps(attrs))
            return path

        def to_v03(multiscale):          # 0.3: axes are plain names
            multiscale["version"] = "0.3"
            multiscale["axes"] = ["z", "y", "x"]

        def to_v02(multiscale):          # 0.2: no axes, no transformations
            del multiscale["version"], multiscale["axes"]
            for dataset in multiscale["datasets"]:
                del dataset["coordinateTransformations"]

        def untyped(multiscale):
            del multiscale["axes"][0]["type"]

        def scale_in_array(multiscale):
            multiscale["datasets"][0]["coordinateTransformations"] = [
                {"type": "scale", "path": "scales/0"}]

        plain = tmp / "plain.zarr"
        zarr.open_group(plain, mode="w", zarr_format=2).attrs["other"] = 1
        refused(plain, "not an OME-Zarr image")
        refused(edited("v03.zarr", to_v03), "OME-Zarr 0.3")
        refused(edited("v02.zarr", to_v02), "incomplete OME-Zarr metadata")
        refused(edited("untyped.zarr", untyped), "have no type")
        refused(edited("path.zarr", scale_in_array), "separate array")

        # (0.21.0: a collection of one image opens that image; several
        # are tested in test_formats.py)
        collection = tmp / "collection.zarr"
        zarr.open_group(collection, mode="w", zarr_format=2).attrs[
            "bioformats2raw.layout"] = 3
        make_test_image(collection / "0", **small)
        assert vc.open_image(collection).inner == "0"
        assert vc.open_image(collection / "0").ome_version == "0.4"


# --- 0.11.0: every n-th timepoint ---------------------------------------------

def test_every_nth_timepoint():
    """A time step keeps timepoints start, start + n, ... of the range
    (compared with the source data directly), the crop's frame interval
    is n times the source's and its first frame keeps its time; the range
    ends at the last kept timepoint; the check passes; the record and the
    boxes file keep the step; for OME-Zarr 0.5 and 0.4."""
    for version in ("0.5", "0.4", "0.6"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data0 = make_test_image(tmp / "src.zarr", shape0=(9, 2, 10, 40, 40),
                                    factors=[(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)],
                                    axes="tczyx",
                                    scale0=(30.0, 1.0, 2.0, 0.5, 0.5),
                                    version=version)
            image = vc.open_image(tmp / "src.zarr")
            # t 30-210 min = timepoints 1-7 (frame centres every 30 min)
            for step, kept in ((1, [1, 2, 3, 4, 5, 6, 7]), (2, [1, 3, 5, 7]),
                               (3, [1, 4, 7]), (4, [1, 5]), (10, [1])):
                box = vc.CropBox(f"s{step}", {"t": (30, 210)}, t_step=step)
                plan = vc.plan_crop(image, box, min_size=4)
                assert len(plan.level_indices) == 2
                assert plan.shapes[0][0] == len(kept), (step, plan.shapes)
                # the range ends at the last kept timepoint (its edge)
                assert plan.snapped_ranges["t"] == (15.0, kept[-1] * 30 + 15)
                out = vc.write_crop(image, plan, tmp / f"s{step}.zarr")
                assert vc.verify_crop(image, plan, out) == [], step
                crop = vc.open_image(out)
                np.testing.assert_array_equal(crop.levels[0].array[...],
                                              data0[kept])
                for level in crop.levels:
                    assert abs(level.scale[0] - 30.0 * step) < 1e-9
                    assert abs(level.translation[0] - 30.0) < 1e-9
                record = zarr.open_group(out, mode="r").attrs[
                    "crop_provenance"]
                assert record["t_step"] == step
                if step == 2:
                    assert "4 of 7 voxels: every 2nd" in plan.describe(image)
                    tiff = vc.export_tiff(out, 0)          # Fiji: 60 min
                    _, meta, _ = read_tiff(tiff)
                    assert meta["frames"] == 4 and meta["finterval"] == 60.0
                    # keep_position False: the first kept frame at 0
                    out2 = vc.write_crop(image, plan, tmp / "s2_zero.zarr",
                                         keep_position=False)
                    assert vc.verify_crop(image, plan, out2,
                                          keep_position=False) == []
                    assert vc.open_image(out2).levels[0].translation[0] == 0


def test_time_step_limits_and_boxes_file():
    """A pyramid that also shrinks time keeps only the levels that do not
    (with a step > 1); a step is refused if it is not a whole number >= 1
    or the image has no time axis; the boxes file keeps the step (no
    column without a step, so old files and old versions still work)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr", shape0=(9, 2, 10, 40, 40),
                        factors=[(1, 1, 1, 1, 1), (2, 1, 1, 2, 2)],
                        axes="tczyx", scale0=(30.0, 1.0, 2.0, 0.5, 0.5))
        image = vc.open_image(tmp / "src.zarr")
        every = vc.plan_crop(image, vc.CropBox("a", {}), min_size=4)
        assert every.level_indices == [0, 1]
        stepped = vc.plan_crop(image, vc.CropBox("b", {}, t_step=2),
                               min_size=4)
        assert stepped.level_indices == [0] and stepped.shapes[0][0] == 5
        for bad in (0, -1, 1.5, "2", True):
            try:
                vc.plan_crop(image, vc.CropBox("x", {}, t_step=bad))
            except ValueError as err:
                assert "time step" in str(err)
            else:
                raise AssertionError(f"t_step {bad!r} accepted")
        assert vc.plan_crop(image, vc.CropBox("x", {}, t_step=2.0),
                            min_size=4).shapes[0][0] == 5

        make_test_image(tmp / "zyx.zarr", shape0=(40, 150, 130),
                        factors=[(1, 1, 1), (1, 4, 4)],
                        scale0=(2.0, 0.25, 0.25), axes="zyx")
        try:
            vc.plan_crop(vc.open_image(tmp / "zyx.zarr"),
                         vc.CropBox("x", {}, t_step=2))
        except ValueError as err:
            assert "no time axis" in str(err)
        else:
            raise AssertionError("a time step without a time axis")

        boxes = [vc.CropBox("a", {"t": (30, 120)}, t_step=3),
                 vc.CropBox("b", {"t": (0, 60)})]
        vc.save_boxes_csv(boxes, tmp / "boxes.csv", image)
        assert "t_step" in (tmp / "boxes.csv").read_text().splitlines()[0]
        loaded = vc.load_boxes_csv(tmp / "boxes.csv")
        assert [b.t_step for b in loaded] == [3, 1]
        vc.save_boxes_csv([boxes[1]], tmp / "plain.csv", image)
        assert "t_step" not in (tmp / "plain.csv").read_text()
        assert vc.load_boxes_csv(tmp / "plain.csv")[0].t_step == 1
        (tmp / "bad.csv").write_text("name,t_step\nx,1.5\n")
        try:
            vc.load_boxes_csv(tmp / "bad.csv")
        except ValueError as err:
            assert "t_step must be a whole number" in str(err)
        else:
            raise AssertionError("t_step 1.5 accepted")
        (tmp / "two.csv").write_text("name,t_step\nx,2.0\ny,\n")
        assert [b.t_step for b in vc.load_boxes_csv(tmp / "two.csv")] == [2, 1]


# --- 0.10.0: one level of a crop as an ImageJ TIFF, for Fiji ------------------

def read_tiff(path):
    """A TIFF's voxels as (t, z, c, y, x), its ImageJ metadata, and its
    voxel size in x and y (from the resolution tags)."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        assert tif.is_imagej
        meta = tif.imagej_metadata
        page = tif.pages[0]
        shape = (meta.get("frames", 1), meta.get("slices", 1),
                 meta.get("channels", 1), *page.shape[-2:])
        data = tif.series[0].asarray().reshape(shape)
        x_res = page.tags["XResolution"].value
        y_res = page.tags["YResolution"].value
    return data, meta, (x_res[1] / x_res[0], y_res[1] / y_res[0])


def test_region_saved_straight_to_tiff():
    """0.15.0 "Save as TIFF…": one level of a region straight to a TIFF,
    with no OME-Zarr crop (nor temporary folder) left: the same voxels and
    calibration as that level of a crop of the region (outline and channels
    applied), checked; refused without Overwrite when it exists; for 0.5
    and 0.4 images."""
    for version in ("0.5", "0.4", "0.6"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_test_image(tmp / "src.zarr", version=version)
            image = vc.open_image(tmp / "src.zarr")
            box = vc.CropBox("shaped", {"z": (10, 70)},
                             outlines={("y", "x"): L_SHAPE}, channels=[0, 2])
            out = tmp / "out"
            for level in (0, 1):
                # the region planned down to that level (vc.tiff_plan): as
                # a crop with that many levels
                reference = vc.crop_box(image, box, tmp / f"crops{level}",
                                        n_levels=level + 1)
                crop = vc.open_image(reference.path)
                result = vc.save_region_tiff(image, box, out, level)
                assert result.problems == [], result.problems
                assert result.path == out / f"src_shaped_level{level}.tif"
                assert result.tiff_level == level
                data, meta, _ = read_tiff(result.path)
                expected = crop.levels[level].array[...]     # (c, z, y, x)
                np.testing.assert_array_equal(
                    data[0], expected.transpose(1, 0, 2, 3))
                assert (data == 0).any() and data.any()      # outline kept
                assert meta["channels"] == 2
            assert sorted(p.name for p in out.iterdir()) == [
                "src_shaped_level0.tif", "src_shaped_level1.tif"]
            try:
                vc.save_region_tiff(image, box, out, 0)
            except (FileExistsError, ValueError):
                pass
            else:
                raise AssertionError("an existing TIFF was overwritten")
            again = vc.save_region_tiff(image, box, out, 0,
                                        overwrite=True, compression="zstd")
            assert again.problems == []


def test_tiff_scratch_folder_removed_or_cleared_later():
    """(0.18.0, the user's report: an empty .volume_cropper_tmp_* folder was
    left on the Desktop after "Save as TIFF") A save leaves no scratch
    folder; a removal Windows refuses is tried again and reported; old
    scratch folders without files are cleared at the next save, those with
    files or younger than an hour (another window's save) are kept."""
    import shutil
    import time as time_module
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        out = tmp / "out"
        out.mkdir()
        old = time_module.time() - 2 * vc.SCRATCH_AGE_S
        empty = out / (vc.SCRATCH_PREFIX + "empty")
        (empty / "a.zarr" / "0" / "c").mkdir(parents=True)   # folders only
        with_file = out / (vc.SCRATCH_PREFIX + "file")
        with_file.mkdir()
        (with_file / "chunk").write_bytes(b"x")
        young = out / (vc.SCRATCH_PREFIX + "young")
        young.mkdir()
        other = out / "my_folder"
        other.mkdir()
        for folder in (empty, with_file, other):
            os.utime(folder, (old, old))
        result = vc.save_region_tiff(image, vc.CropBox("r", {"z": (10, 30)}),
                                     out, 1)
        assert result.problems == [] and result.note == ""
        left = sorted(p.name for p in out.iterdir() if p.is_dir())
        assert left == sorted([with_file.name, young.name, other.name]), left
        # a folder that will not go: tried again, then reported
        stubborn = out / (vc.SCRATCH_PREFIX + "stubborn")
        stubborn.mkdir()
        calls = []
        real = shutil.rmtree
        shutil.rmtree = lambda path, ignore_errors=False: calls.append(path)
        try:
            assert vc.remove_scratch(stubborn, tries=3, wait=0.0) is True
        finally:
            shutil.rmtree = real
        assert len(calls) == 3
        assert vc.remove_scratch(stubborn) is False and not stubborn.exists()


def test_tiff_is_the_crop_level():
    """A crop level saved as TIFF holds exactly its voxels (also those set
    to 0 by an outline), in ImageJ's order (z before c), with ImageJ's
    calibration (voxel size, spacing, origin = the crop's position), the
    channel names and the crop's record; for 0.5 and 0.4 crops."""
    for version in ("0.5", "0.4", "0.6"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_test_image(tmp / "src.zarr", version=version)
            image = vc.open_image(tmp / "src.zarr")
            box = vc.CropBox("shaped", {"z": (10, 70)},
                             outlines={("y", "x"): L_SHAPE}, channels=[0, 2])
            result = vc.crop_box(image, box, tmp / "out", min_size=4,
                                 tiff_level=1)
            assert result.problems == [], result.problems
            assert result.tiff_problems == [], result.tiff_problems
            assert result.tiff == tmp / "out" / "src_shaped_level1.tif"
            assert result.tiff_level == 1 and result.tiff.exists()
            crop = vc.open_image(result.path)
            level0 = vc.export_tiff(result.path, 0)
            assert level0.name == "src_shaped_level0.tif"
            assert vc.verify_tiff(result.path, level0, 0) == []
            for level, path in ((0, level0), (1, result.tiff)):
                lev = crop.levels[level]
                data, meta, (x_size, y_size) = read_tiff(path)
                expected = lev.array[...]                    # (c, z, y, x)
                np.testing.assert_array_equal(
                    data[0], expected.transpose(1, 0, 2, 3))
                assert (data == 0).any() and data.any()      # outline kept
                assert meta["channels"] == 2 and meta["mode"] == "composite"
                assert meta["unit"] == "um"
                assert abs(meta["spacing"] - lev.scale[1]) < 1e-9
                assert abs(x_size - lev.scale[3]) < 1e-6
                assert abs(y_size - lev.scale[2]) < 1e-6
                # Origin: Fiji's measurements = the OME (napari) positions.
                # ImageJ's pixel i spans i ... i + 1 (centre i + 0.5) and
                # p maps to (p - origin) * size; OME's voxel i centre is
                # translation + i * size. Slices have no half.
                for axis, i, half in (("x", 3, 0.5), ("y", 2, 0.5),
                                      ("z", 1, 0.0)):
                    origin = meta[f"{axis}origin"]
                    centre = (0 + half - origin) * lev.scale[i]
                    assert abs(centre - lev.translation[i]) < 1e-6, axis
                assert meta["Labels"][:4] == ["ch0", "ch2", "ch0", "ch2"]
                assert len(meta["LUTs"]) == 2
                assert meta["Info"].startswith(
                    f"Volume Cropper {vc.__version__}: level {level}")
                assert '"box_name": "shaped"' in meta["Info"]
            assert not list((tmp / "out").glob("*.part"))


def test_tiff_time_lapse_and_no_channels():
    """A time-lapse (t, c, z, y, x; OME-Zarr 0.4) gives frames with their
    interval and time unit; an image of z, y, x alone gives one channel and
    one frame; a level the crop lacks gives its coarsest level."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data0 = make_test_image(tmp / "lapse.zarr", shape0=(5, 2, 10, 40, 40),
                                factors=[(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)],
                                axes="tczyx", scale0=(30.0, 1.0, 2.0, 0.5, 0.5),
                                version="0.4")
        image = vc.open_image(tmp / "lapse.zarr")
        box = vc.CropBox("frames", {"t": (30, 90)})
        result = vc.crop_box(image, box, tmp / "out", min_size=4,
                             tiff_level=0)
        assert result.problems == [] and result.tiff_problems == []
        data, meta, _ = read_tiff(result.tiff)
        np.testing.assert_array_equal(data, data0[1:4].transpose(0, 2, 1, 3, 4))
        assert meta["frames"] == 3 and meta["finterval"] == 30.0
        assert meta["tunit"] == "min"

        data0 = make_test_image(tmp / "single.zarr", shape0=(40, 150, 130),
                                factors=[(1, 1, 1), (1, 4, 4), (2, 8, 8)],
                                scale0=(2.0, 0.25, 0.25), axes="zyx")
        image = vc.open_image(tmp / "single.zarr")
        box = vc.CropBox("b", {"z": (10, 50), "y": (5, 25), "x": (3, 20)})
        plan = vc.plan_crop(image, box, min_size=4)
        assert vc.tiff_level_for(plan, 7) == len(plan.level_indices) - 1
        result = vc.crop_box(image, box, tmp / "out", min_size=4,
                             tiff_level=7)
        last = len(plan.level_indices) - 1
        assert result.problems == [] and result.tiff_problems == []
        assert result.tiff_level == last
        assert result.tiff.name == f"single_b_level{last}.tif"
        data, meta, _ = read_tiff(result.tiff)
        expected = vc.open_image(result.path).levels[last].array[...]
        np.testing.assert_array_equal(data[0, :, 0], expected)
        assert "channels" not in meta and "frames" not in meta
        try:
            vc.tiff_level_for(plan, -1)
        except ValueError:
            pass
        else:
            raise AssertionError("a negative level was accepted")


def test_tiff_read_in_small_pieces():
    """The level is read a few slices at a time, in whole z chunks (8
    slices here), so that each chunk is read once: with room for 20 slices
    the reads are 16 slices; with room for 3 they are one whole chunk (it
    fits TIFF_READ_MAX); only when a chunk does not fit are they thinner
    (3 slices). Every voxel is right and in place in all cases."""
    originals = vc.TIFF_READ_BYTES, vc.TIFF_READ_MAX, vc._read_slices
    reads = []

    def recording(array, axes, sizes, t, z0, z1):
        reads.append((t, z0, z1))
        return originals[2](array, axes, sizes, t, z0, z1)
    try:
        vc._read_slices = recording
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_test_image(tmp / "src.zarr")
            image = vc.open_image(tmp / "src.zarr")
            result = vc.crop_box(image, vc.CropBox("b", {"z": (10, 70)}),
                                 tmp / "out", min_size=4)
            expected = vc.open_image(result.path).levels[0].array[...]
            n_z = expected.shape[1]                     # (c, z, y, x)
            assert n_z == 32
            slice_bytes = vc.nbytes(expected.shape[:1] + expected.shape[2:],
                                    expected.dtype)
            for room, read_max, depth in ((20, 2**30, 16), (3, 2**30, 8),
                                          (3, 2 * slice_bytes, 3),
                                          (1, 2**30, 8)):
                vc.TIFF_READ_BYTES = room * slice_bytes
                vc.TIFF_READ_MAX = read_max
                reads.clear()
                path = vc.export_tiff(result.path, 0, overwrite=True)
                assert reads == [(0, z, min(z + depth, n_z))
                                 for z in range(0, n_z, depth)], (room, reads)
                data, _, _ = read_tiff(path)
                np.testing.assert_array_equal(
                    data[0], expected.transpose(1, 0, 2, 3))
    finally:
        vc.TIFF_READ_BYTES, vc.TIFF_READ_MAX, vc._read_slices = originals


def test_tiff_progress_reaches_the_end():
    """The user's report of 2026-11-09: Save as TIFF's progress bar stopped
    at 99 %. The TIFF's last progress report is the whole size, with and
    without compression (before the fix, with zstd it was one plane short:
    a compressing tifffile never asks for a plane after the last one, which
    is when the generator reported it), and Save as TIFF's crop and TIFF
    reports each end at their totals."""
    for compression in (None, "zstd"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_test_image(tmp / "src.zarr")
            image = vc.open_image(tmp / "src.zarr")
            crop_reports, tiff_reports = [], []
            result = vc.save_region_tiff(
                image, vc.CropBox("b", {"z": (10, 70)}), tmp / "out",
                level=0, compression=compression,
                progress_bytes=lambda d, t: crop_reports.append((d, t)),
                tiff_progress=lambda d, t: tiff_reports.append((d, t)))
            assert result.problems == []
            assert crop_reports[-1][0] == crop_reports[-1][1]
            done, total = tiff_reports[-1]
            assert done == total == vc.nbytes(result.plan.shapes[0],
                                              "uint16"), \
                (compression, tiff_reports[-3:])


def test_tiff_overwrite_cancel_and_check():
    """An existing TIFF is replaced only with overwrite, only if this tool
    wrote it, and only by the same region; a cancelled export leaves
    nothing behind; the check finds changed voxels; unsupported data types
    are refused."""
    import tifffile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        result = vc.crop_box(image, vc.CropBox("b", {"z": (10, 70)}),
                             tmp / "out", min_size=4, tiff_level=0)
        tiff = result.tiff
        for call in (lambda: vc.export_tiff(result.path, 0),
                     lambda: vc.crop_box(image, vc.CropBox("b", {"z": (10, 70)}),
                                         tmp / "out", min_size=4,
                                         tiff_level=0)):
            try:
                call()
            except FileExistsError as err:
                assert "overwrite" in str(err)
            else:
                raise AssertionError("an existing TIFF was replaced")
        vc.export_tiff(result.path, 0, overwrite=True)
        try:            # a box whose name gives the same file name
            vc.check_tiff_output(tiff, overwrite=True, box_name="other")
        except FileExistsError as err:
            assert "'b'" in str(err)
        else:
            raise AssertionError("another region's TIFF would be replaced")
        foreign = tmp / "out" / "src_b_level1.tif"
        tifffile.imwrite(foreign, np.zeros((4, 4), np.uint16))
        try:
            vc.check_tiff_output(foreign, overwrite=True)
        except FileExistsError as err:
            assert "not written by Volume Cropper" in str(err)
        else:
            raise AssertionError("a foreign TIFF would be replaced")

        cancel = threading.Event()
        cancel.set()
        tiff.unlink()
        try:
            vc.export_tiff(result.path, 0, cancel_event=cancel)
        except vc.CropCancelled:
            pass
        else:
            raise AssertionError("not cancelled")
        assert not tiff.exists() and not list((tmp / "out").glob("*.part"))

        vc.export_tiff(result.path, 0)
        assert vc.verify_tiff(result.path, tiff, 0) == []
        data = tifffile.memmap(tiff, mode="r+")
        data[0, 0, 0, 0] += 1           # first slice, first channel
        data.flush()
        del data
        problems = vc.verify_tiff(result.path, tiff, 0)
        assert problems == ["voxels differ at timepoint 0, slice 0"], problems

        vc.check_tiff_dtype("uint16")
        for dtype in ("int32", "float64", "uint32", "int16"):
            try:
                vc.check_tiff_dtype(dtype)
            except ValueError as err:
                assert dtype in str(err) and "unsigned" in str(err)
            else:
                raise AssertionError(f"{dtype} accepted")


def retyped_copy(src, dst, dtype, fill_value=0):
    """A copy of a 0.4 test image (make_test_image(version="0.4")) whose
    levels are stored with another data type, e.g. big-endian ">u2" (as
    some OME-Zarr 0.4 writers do) or float32."""
    source = zarr.open_group(src, mode="r")
    copy = zarr.open_group(dst, mode="w-", zarr_format=2)
    copy.attrs.update(source.attrs.asdict())
    for name in sorted(source.array_keys()):
        a = source[name]
        b = copy.create_array(name=name, shape=a.shape, dtype=dtype,
                              chunks=a.chunks, compressors=a.compressors,
                              chunk_key_encoding={"name": "v2",
                                                  "separator": "/"},
                              fill_value=fill_value)
        b[...] = a[...].astype(dtype)


def set_ome(path, change):
    """Change a 0.5 test image's OME metadata: change(ome) edits the dict."""
    root = zarr.open_group(path, mode="r+")
    ome = root.attrs.asdict()["ome"]
    change(ome)
    root.attrs["ome"] = ome


def test_tiff_of_other_data():
    """Big-endian data (an OME-Zarr 0.4 writer may store '>u2') and float
    data with NaN give TIFFs that pass the check; a unit spelt "µm" in the
    metadata is written as ImageJ's "um" (the description must be ASCII);
    an axis an ImageJ TIFF cannot hold is refused before the crop is
    written (review findings of 2026-11-06)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "le.zarr", version="0.4")
        box = vc.CropBox("b", {"z": (10, 70)}, outlines={("y", "x"): L_SHAPE})
        for name, dtype, fill in (("be", ">u2", 0), ("fl", "float32", np.nan)):
            retyped_copy(tmp / "le.zarr", tmp / f"{name}.zarr", dtype, fill)
            image = vc.open_image(tmp / f"{name}.zarr")
            assert image.levels[0].array.dtype == np.dtype(dtype)
            result = vc.crop_box(image, box, tmp / "out", min_size=4,
                                 tiff_level=0)
            assert result.problems == [] and result.tiff_problems == [], \
                (name, result.problems, result.tiff_problems)
            expected = vc.open_image(result.path).levels[0].array[...]
            data, _, _ = read_tiff(result.tiff)
            np.testing.assert_array_equal(data[0],
                                          expected.transpose(1, 0, 2, 3))
            if name == "fl":                     # outside the outline: NaN
                assert np.isnan(expected).any()

        make_test_image(tmp / "mu.zarr")
        set_ome(tmp / "mu.zarr", lambda ome: [
            axis.update(unit="µm") for axis in ome["multiscales"][0]["axes"]
            if axis.get("type") == "space"])
        image = vc.open_image(tmp / "mu.zarr")
        result = vc.crop_box(image, vc.CropBox("b", {"z": (10, 70)}),
                             tmp / "out", min_size=4, tiff_level=0)
        assert result.tiff_problems == []
        import tifffile
        with tifffile.TiffFile(result.tiff) as tif:
            description = tif.pages[0].description
        assert description.isascii() and "unit=um" in description
        assert vc._imagej_unit("wavelength Å", {}, "pixel") == \
            "wavelength \\u00C5"

        make_test_image(tmp / "angle.zarr", shape0=(2, 3, 40, 150, 130),
                        factors=[(1, 1, 1, 1, 1), (1, 1, 1, 4, 4)],
                        scale0=(1.0, 1.0, 2.0, 0.25, 0.25), axes="tczyx")
        set_ome(tmp / "angle.zarr", lambda ome: ome["multiscales"][0][
            "axes"][0].update(name="a", type="angle"))
        image = vc.open_image(tmp / "angle.zarr")
        for call in (lambda: vc.check_tiff_image(image),
                     lambda: vc.crop_box(image, vc.CropBox("b", {}),
                                         tmp / "out", min_size=4,
                                         tiff_level=0)):
            try:
                call()
            except ValueError as err:
                assert "cannot be stored in an ImageJ TIFF: a" in str(err)
            else:
                raise AssertionError("an angle axis was accepted")
        assert not (tmp / "out" / "angle_b.zarr").exists()


def test_tiff_colours_and_display_ranges():
    """Channel colours (LUTs) and display ranges (Ranges) come from the OME
    metadata. One channel: ImageJ's min and max, and the colour as the
    ColorMap tag (ImageJ ignores LUTs with one channel)."""
    import tifffile
    colours = ["FF0000", "00FF00", "0000FF"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")

        def colour_channels(ome):
            for c, channel in enumerate(ome["omero"]["channels"]):
                channel["color"] = colours[c]
                channel["window"] = {"start": 100.0 * (c + 1),
                                     "end": 1000.0 * (c + 3),
                                     "min": 0.0, "max": 65535.0}
        set_ome(tmp / "src.zarr", colour_channels)
        image = vc.open_image(tmp / "src.zarr")
        result = vc.crop_box(image, vc.CropBox("all", {"z": (10, 70)}),
                             tmp / "out", min_size=4, tiff_level=0)
        _, meta, _ = read_tiff(result.tiff)
        assert meta["Ranges"] == (100.0, 3000.0, 200.0, 4000.0, 300.0, 5000.0)
        for lut, colour in zip(meta["LUTs"], colours):
            assert list(lut[:, 255]) == [int(colour[i:i + 2], 16)
                                         for i in (0, 2, 4)]
            assert list(lut[:, 0]) == [0, 0, 0]
        result = vc.crop_box(image, vc.CropBox("green", {"z": (10, 70)},
                                               channels=[1]),
                             tmp / "out", min_size=4, tiff_level=0)
        assert result.tiff_problems == []
        _, meta, _ = read_tiff(result.tiff)
        assert (meta["min"], meta["max"]) == (200.0, 4000.0)
        with tifffile.TiffFile(result.tiff) as tif:
            colormap = tif.pages[0].colormap          # (3, 256) uint16
        assert colormap is not None
        assert list(colormap[:, 255] >> 8) == [0, 255, 0]
        assert list(colormap[:, 128] >> 8) == [0, 128, 0]


def test_tiff_overwrite_removes_the_old_tiffs():
    """Re-cropping a region with overwrite deletes its old TIFFs first: of
    another level (the region now keeps fewer levels), when no TIFF is
    asked any more, and before a TIFF that is then cancelled, so that no
    TIFF shows the old region. A TIFF of another region whose name starts
    the same is kept."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp, out = Path(tmp), Path(tmp) / "out"
        make_test_image(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        big = vc.CropBox("b", {"z": (10, 70)})
        other = vc.CropBox("b_level2", {"z": (10, 30)})
        first = vc.crop_box(image, big, out, min_size=4, tiff_level=2)
        assert first.tiff.name == "src_b_level2.tif"
        kept = vc.crop_box(image, other, out, min_size=4, tiff_level=0).tiff
        assert kept.name == "src_b_level2_level0.tif"
        assert vc.old_tiffs(first.path, "b") == [first.tiff]

        # Fewer levels now (min size 64 keeps only level 0): level 0 TIFF.
        again = vc.crop_box(image, big, out, min_size=64, tiff_level=2,
                            overwrite=True)
        assert again.tiff.name == "src_b_level0.tif"
        assert again.removed_tiffs == [first.tiff] and not first.tiff.exists()
        assert kept.exists()

        # No TIFF asked: the level 0 TIFF of the replaced crop goes too.
        plain = vc.crop_box(image, big, out, min_size=64, overwrite=True)
        assert plain.removed_tiffs == [again.tiff] and not again.tiff.exists()

        # A TIFF cancelled while it is written: the old one is gone already.
        vc.crop_box(image, big, out, min_size=64, tiff_level=0,
                    overwrite=True)
        cancel = threading.Event()
        try:
            vc.crop_box(image, big, out, min_size=64, tiff_level=0,
                        overwrite=True, cancel_event=cancel,
                        tiff_progress=lambda done, total: cancel.set())
        except vc.CropCancelled:
            pass
        else:
            raise AssertionError("not cancelled")
        assert not (out / "src_b_level0.tif").exists()
        assert not list(out.glob("*.part")) and kept.exists()


def test_tiff_checked_before_anything_is_written():
    """A TIFF that may not be replaced (not written by this tool) stops the
    crop before its .zarr is written: in crop_box, and on the command line
    before the first box."""
    import tifffile
    with tempfile.TemporaryDirectory() as tmp:
        tmp, out = Path(tmp), Path(tmp) / "out"
        make_test_image(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        out.mkdir()
        foreign = out / "src_two_level0.tif"
        tifffile.imwrite(foreign, np.zeros((4, 4), np.uint16))
        two = vc.CropBox("two", {"z": (10, 70)})
        try:
            vc.crop_box(image, two, out, min_size=4, tiff_level=0,
                        overwrite=True)
        except FileExistsError as err:
            assert "not written by Volume Cropper" in str(err)
        else:
            raise AssertionError("a foreign TIFF would be replaced")
        assert not (out / "src_two.zarr").exists()

        one = vc.CropBox("one", {"z": (10, 30)})
        csv_path = tmp / "boxes.csv"
        vc.save_boxes_csv([one, two], csv_path, image)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            code = vc.main([str(tmp / "src.zarr"), str(csv_path), str(out),
                            "--min-size", "4", "--tiff", "0", "--overwrite"])
        assert code == 2 and "not written by Volume Cropper" in \
            stderr.getvalue(), stderr.getvalue()
        assert not (out / "src_one.zarr").exists()
        assert not (out / "src_two.zarr").exists()


def test_tiff_open_elsewhere_and_over_4_gb():
    """A TIFF open in another program (Windows locks it) cannot be replaced
    or deleted: a clear error, the old file intact, nothing half-written.
    A TIFF in ImageJ's one-directory layout (what tifffile writes above
    4 GB) passes the check."""
    import tifffile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        result = vc.crop_box(image, vc.CropBox("b", {"z": (10, 70)}),
                             tmp / "out", min_size=4, tiff_level=0)
        tiff = result.tiff
        if os.name == "nt":            # only Windows refuses open files
            size = tiff.stat().st_size
            handle = open(tiff, "rb")
            try:
                for call, text in (
                        (lambda: vc.export_tiff(result.path, 0,
                                                overwrite=True),
                         "open in another program"),
                        (lambda: vc.remove_old_tiffs(result.path, "b"),
                         "Cannot delete the old TIFF")):
                    try:
                        call()
                    except PermissionError as err:
                        assert text in str(err), str(err)
                    else:
                        raise AssertionError("an open TIFF was replaced")
            finally:
                handle.close()
            assert tiff.stat().st_size == size
            assert not list((tmp / "out").glob("*.part"))

        original = tifffile.imwrite

        def one_directory(*args, **kwargs):
            return original(*args, truncate=True, **kwargs)
        tifffile.imwrite = one_directory
        try:
            vc.export_tiff(result.path, 0, overwrite=True)
        finally:
            tifffile.imwrite = original
        with tifffile.TiffFile(tiff) as tif:
            assert len(tif.pages) == 1 and tif.series[0].is_truncated
        assert vc.verify_tiff(result.path, tiff, 0) == []


# --- found in the review of 0.17.0 (2026-11-10) ------------------------------

def _circle(r, centre=(50.0, 50.0), n=120, turns=1.0, noise=0.0, seed=0):
    """A circle of radius r as a polygon (y, x), drawn over `turns` turns
    (more than 1: a lasso that overshoots its start), with hand tremor."""
    angle = np.linspace(0, 2 * np.pi * turns, int(n * turns), endpoint=False)
    points = np.column_stack([centre[0] + r * np.sin(angle),
                              centre[1] + r * np.cos(angle)])
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in points + rng.normal(0, noise, points.shape)]


def _kept(outlines, n=100):
    """The (z, y, x) voxels outline_mask keeps in a cube of n voxels of
    1 um."""
    level = vc.Level("0", None, [1.0] * 3, [0.0] * 3)
    zyx = ["z", "y", "x"]
    return np.broadcast_to(vc.outline_mask(level, zyx, [0, 0, 0], (n,) * 3,
                                           outlines, spatial=zyx), (n,) * 3)


def test_polygons_in_one_view_are_interpolated_slice_to_slice():
    """Polygons drawn on slices of one view only: the shape between two
    drawn slices is blended from those two alone (the user's "weird
    interpolations": with the smooth surface through all polygons, the same
    circle on two slices bulged to 1.45 times its area in between, and a
    slightly smaller circle drawn 1 um from another emptied the shape
    50 um away)."""
    disc = np.pi * 20 ** 2
    same = _kept({("y", "x"): {30.0: _circle(20), 70.0: _circle(20)}})
    for z in (30, 50, 70):
        assert abs(same[z].sum() / disc - 1) < 0.03, (z, same[z].sum())
    assert not same[:30].any() and not same[71:].any()  # ends at them
    sloppy = _kept({("y", "x"): {30.0: _circle(20), 31.0: _circle(17),
                                 80.0: _circle(20)}})
    ratios = [sloppy[z].sum() / disc for z in (40, 55, 70)]
    assert all(0.7 < r < 1.0 for r in ratios), ratios
    assert ratios[0] < ratios[1] < ratios[2]          # back to r = 20


def test_sloppy_lassos_keep_what_was_drawn():
    """A lasso that overshoots its start (5 % of a turn more) or a shaky
    one keeps, on its own slice, what Fiji would fill (overshooting kept
    only 43 % of it, from normals that pointed inwards)."""
    for drawn in (_circle(20, turns=1.05), _circle(20, n=300, noise=0.7)):
        kept = _kept({("y", "x"): {10.0: drawn, 20.0: drawn, 30.0: drawn}})
        filled = vc.fill_polygon((100, 100), np.asarray(drawn))
        for z in (10, 20, 30):
            differ = (kept[z] != filled).sum()
            assert differ <= 0.02 * filled.sum(), (z, differ)


def test_shaky_lasso_with_polygons_in_other_views():
    """Polygons in several views (the smooth surface): a shaky lasso (600
    points, 0.7 um tremor) on every polygon kept 49 % of the tube with
    68 % extra in 5 pieces; now its edge is taken from the filled
    polygon, and the fit has at most MAX_FIT_POINTS points."""
    truth, outlines = tube_and_outlines()
    rng = np.random.default_rng(1)
    for keys in outlines.values():
        for depth, points in keys.items():
            ring = vc._resample_ring(np.asarray(points, dtype=float), 0.1)
            ring = ring[np.linspace(0, len(ring) - 1, 600).astype(int)]
            keys[depth] = [tuple(p) for p in
                           ring + rng.normal(0, 0.7, ring.shape)]
    kept = _kept(outlines)
    assert (kept & truth).sum() > 0.85 * truth.sum()
    assert (kept & ~truth).sum() < 0.08 * truth.sum()
    assert ndi.label(kept)[1] <= 2


def test_level_names_must_stay_inside_the_image():
    """A level whose name (its 'path' in the OME metadata) is absolute, a
    drive or a network share is refused: it would be read from outside the
    image, and its crop written outside the output folder."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path)
        root = zarr.open_group(path, mode="r+")
        ome = root.attrs["ome"]
        for name in ("C:/elsewhere/0", "C:0", "//host/share/0", "/0", "a/../0"):
            ome["multiscales"][0]["datasets"][0]["path"] = name
            root.attrs["ome"] = ome
            try:
                vc.open_image(path)
            except ValueError as err:
                assert "plain names" in str(err), err
            else:
                raise AssertionError(f"level {name!r} was opened")


def test_regions_csv_reads_outlines_only_next_to_it():
    """The regions CSV's outlines-per-timepoint file must be a plain file
    name next to the CSV (a CSV from elsewhere could name any file, or a
    network share Windows would connect to)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "csvs").mkdir()
        (tmp / "other.json").write_text("{}")
        for cell in (str(tmp / "other.json"), r"..\other.json",
                     "../other.json", r"\\host\share\x.json", "sub/x.json",
                     "C:x.json"):
            csv_path = tmp / "csvs" / "regions.csv"
            csv_path.write_text("name,z_start_um,z_stop_um,"
                                f'timed_outlines_file\nr1,0,10,"{cell}"\n')
            try:
                vc.load_boxes_csv(csv_path)
            except ValueError as err:
                assert "plain file name" in str(err), err
            else:
                raise AssertionError(f"{cell!r} was read")


def test_overwrite_keeps_crops_of_other_images():
    """pos1/img.zarr and pos2/img.zarr give the same crop name: with
    Overwrite, the second used to replace the first's crop (and TIFFs)
    without a word; now it is refused, naming the other image."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        images = []
        for pos in ("pos1", "pos2"):
            (tmp / pos).mkdir()
            make_test_image(tmp / pos / "img.zarr", seed=len(images))
            images.append(vc.open_image(tmp / pos / "img.zarr"))
        box = vc.CropBox("tail", {"x": (5, 10)})
        first = vc.crop_box(images[0], box, tmp / "out")
        try:
            vc.crop_box(images[1], box, tmp / "out", overwrite=True)
        except FileExistsError as err:
            assert "pos1" in str(err), err
        else:
            raise AssertionError("replaced the crop of another image")
        record = zarr.open_group(first.path, mode="r").attrs[
            "crop_provenance"]
        assert "pos1" in record["source"]
        vc.crop_box(images[0], box, tmp / "out", overwrite=True)  # its own


def test_overwrite_needs_a_crop_record_and_a_folder():
    """Overwrite deletes everything in the old crop's folder, so the folder
    must hold a real crop record (an empty one or a text was taken as
    ours, or crashed); a file at the crop's place gets a clear message;
    and a positions file this tool did not write is never deleted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        box = vc.CropBox("tail", {"x": (5, 10)})
        out = vc.output_path_for(image, box, tmp / "out")
        for record in ({}, "text", ["list"]):
            root = zarr.open_group(out, mode="w")
            root.attrs["crop_provenance"] = record
            (out / "notes.txt").write_text("precious")
            try:
                vc.check_output(image, box, out, overwrite=True)
            except FileExistsError as err:
                assert "not written by Volume Cropper" in str(err)
            else:
                raise AssertionError(f"record {record!r} taken as a crop")
            assert (out / "notes.txt").is_file()
        vc.shutil.rmtree(out)
        out.write_text("a file")
        try:
            vc.check_output(image, box, out, overwrite=True)
        except FileExistsError as err:
            assert "is a file" in str(err)
        else:
            raise AssertionError("a file at the crop's place was accepted")
        out.unlink()
        positions = vc.positions_path_for(out)
        positions.write_text("my own notes\n")
        try:
            vc.crop_box(image, box, tmp / "out")
        except FileExistsError as err:
            assert positions.name in str(err)
        else:
            raise AssertionError("a positions file of someone else went")
        assert positions.read_text() == "my own notes\n" and not out.exists()


def test_shape_cases_from_the_second_review():
    """(The review of the 0.17.1 fixes) In one view, a part that stays put
    while another part of the polygon ends stays a straight prism (lining
    up the centres pinched it to 10 %); a polygon thinner than the
    shape's grid, among polygons in other views, still holds its line (it
    got no points, and with only thin polygons the fit crashed)."""
    blob = lambda x: _circle(10, centre=(50.0, x), n=60)     # noqa: E731
    both = [*blob(20.0), (50.0, 30.0), (50.0, 70.0), *blob(80.0),
            (50.0, 70.0), (50.0, 30.0)]               # joined, no width
    kept = _kept({("y", "x"): {10.0: both, 50.0: blob(80.0)}})
    disc = np.pi * 10 ** 2
    yy, xx = np.indices((100, 100))
    right = (yy - 50) ** 2 + (xx - 80) ** 2 <= 100
    for z in (10, 20, 30, 40, 50):
        assert (kept[z] & right).sum() > 0.95 * disc, z
    thin = [(30.0, 20.0), (30.4, 20.0), (30.4, 80.0), (30.0, 80.0)]  # z, x
    kept = _kept({("y", "x"): {20.0: _circle(15), 40.0: _circle(15)},
                  ("z", "x"): {50.0: thin}})
    assert kept[30, 50, 25:76].sum() >= 45                   # its line
    only_thin = _kept({("y", "x"): {20.0: [(50.0, 20.0), (50.4, 20.0),
                                           (50.4, 80.0), (50.0, 80.0)]},
                       ("z", "x"): {50.0: thin}})
    assert only_thin.any()


def test_overshoot_filled_and_polygon_width():
    """(The user's choices of 2026-11-10) A lasso that goes around an area
    twice (overshooting its start) keeps it: Fiji's even-odd rule left it
    out; any other polygon is filled exactly as Fiji fills it. A polygon's
    mean width tells a real outline from three clicks on a line."""
    circle = np.asarray(_circle(30), dtype=float)
    assert np.array_equal(vc.fill_polygon((100, 100), circle),
                          vc.polygon2mask((100, 100), circle))
    turns = np.linspace(0, 2 * np.pi * 1.2, 240)
    shrinking = 30 - 6 * turns / (2 * np.pi)          # a spiral
    lasso = np.column_stack([50 + shrinking * np.sin(turns),
                             50 + shrinking * np.cos(turns)])
    fiji = vc.polygon2mask((100, 100), lasso)
    filled = vc.fill_polygon((100, 100), lasso)
    assert (filled & ~fiji).sum() > 20 and not (fiji & ~filled).any()
    hole = ndi.binary_fill_holes(fiji) & ~fiji        # Fiji's hole
    assert hole.sum() > 20 and filled[hole].mean() > 0.9
    kept = _kept({("y", "x"): {10.0: [tuple(p) for p in lasso],
                               30.0: [tuple(p) for p in lasso]}})
    assert (kept[20] != filled).sum() <= 0.02 * filled.sum()
    assert abs(vc.polygon_width(circle) - 30) < 1.5
    assert vc.polygon_width([(5.0, 5.0), (5.0, 10.0), (5.0, 15.0)]) < 0.05
    assert vc.polygon_width([(5.0, 5.0), (5.2, 10.0), (5.0, 15.0)]) < 0.2


def test_overwrite_keeps_tiffs_of_other_images():
    """(The review of the 0.17.1 fixes) Save as TIFF and the TIFFs next to
    a crop are of an image too: Overwrite replaced or deleted another
    image's TIFF; and a crop recorded from a relative path is recorded as
    an absolute one now."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        images = []
        for pos in ("pos1", "pos2"):
            (tmp / pos).mkdir()
            make_test_image(tmp / pos / "img.zarr", seed=len(images))
            images.append(vc.open_image(tmp / pos / "img.zarr"))
        box = vc.CropBox("tail", {"x": (5, 10), "y": (5, 10)})
        first = vc.save_region_tiff(images[0], box, tmp / "out", level=0)
        try:
            vc.save_region_tiff(images[1], box, tmp / "out", level=0,
                                overwrite=True)
        except FileExistsError as err:
            assert "pos1" in str(err), err
        else:
            raise AssertionError("replaced another image's TIFF")
        crop = vc.output_path_for(images[0], box, tmp / "out")
        assert vc.old_tiffs(crop, "tail", images[1]) == []
        assert vc.old_tiffs(crop, "tail", images[0]) == [first.tiff]
        old = Path.cwd()
        try:
            os.chdir(tmp / "pos1")
            relative = vc.open_image("img.zarr")
            result = vc.crop_box(relative, box, tmp / "rel")
        finally:
            os.chdir(old)
        record = zarr.open_group(result.path, mode="r").attrs[
            "crop_provenance"]
        assert Path(record["source"]).is_absolute()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"all {len(tests)} tests passed")
