"""Tests for the OME-Zarr formats added in 0.21.0.

Run with pytest (``pytest test_formats.py``) or without it
(``python test_formats.py``).

- OME-Zarr 0.6 (RFC-5): the axes live in named coordinate systems, and
  every level's transformation leads to the image's own one. The tests
  check that the tool reads the voxel sizes and positions from the
  transformations (scale, translation, identity, sequences), that crops
  stay 0.6 with the levels written the 0.6 way, and that the image's links
  to other coordinate systems are kept, corrected or left out as they
  should be (the user's decision 5a of 2026-11-22).
- Zipped OME-Zarr: .ozx files (RFC-9) and other zips (.zip,
  .ome.zarr.zip; also with the image in a folder inside, or compressed)
  open like folders; crops of them are zipped too (.ozx for 0.5 and 0.6,
  the source's ending for 0.4: decisions 2a and 3a), laid out as RFC-9
  recommends, checked before and after packing, and replaced only with
  Overwrite and only if they are the tool's own.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import json
import tempfile
import threading
import warnings
import zipfile
from pathlib import Path

import numpy as np
import zarr

from volume_cropper import core as vc
from test_volume_cropper import make_test_image

# A box well inside make_test_image's image (c, z, y, x; µm).
BOX = vc.CropBox("b1", {"z": (11, 53), "y": (5.3, 20.1), "x": (3.0, 25.7)})


def _ome(path) -> dict:
    """The "ome" attributes of an image folder."""
    return zarr.open_group(path, mode="r").attrs.asdict()["ome"]


def _set_ome(path, ome: dict) -> None:
    zarr.open_group(path, mode="r+").attrs["ome"] = ome


def test_06_crop_is_written_the_06_way():
    """A 0.6 crop stays 0.6: its axes in the source's coordinate system
    ("physical"), one transformation per level from the level's array to
    that system (a sequence of scale and translation, or the scale alone
    when the crop starts at 0), and the voxels and the check as for 0.5."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data0 = make_test_image(tmp / "src.zarr", version="0.6",
                                translation=(0, 4.0, 1.0, 2.0))
        image = vc.open_image(tmp / "src.zarr")
        assert image.ome_version == "0.6"
        assert image.coordinate_system == "physical"
        assert image.axis_names == ["c", "z", "y", "x"]
        assert image.levels[1].scale == [1.0, 2.0, 1.0, 1.0]
        assert image.levels[1].translation == [0.0, 4.0, 1.0, 2.0]
        for keep in (True, False):
            plan = vc.plan_crop(image, BOX, min_size=4)
            out = vc.write_crop(image, plan, tmp / f"out{keep}.zarr",
                                keep_position=keep)
            assert vc.verify_crop(image, plan, out,
                                  keep_position=keep) == [], keep
            ome = _ome(out)
            assert ome["version"] == "0.6"
            multiscale = ome["multiscales"][0]
            assert "axes" not in multiscale
            assert [s["name"] for s in multiscale["coordinateSystems"]] \
                == ["physical"]
            for dataset in multiscale["datasets"]:
                (transform,) = dataset["coordinateTransformations"]
                assert transform["input"] == {"path": dataset["path"]}
                assert transform["output"] == {"name": "physical"}
                # (the test image's levels all start at the same place, so
                # a crop that starts at 0 needs no translation at all)
                assert transform["type"] == ("sequence" if keep
                                             else "scale")
                if keep:
                    assert [t["type"] for t in transform[
                        "transformations"]] == ["scale", "translation"]
            crop = vc.open_image(out)
            a, b = plan.starts[0], plan.stops[0]
            np.testing.assert_array_equal(
                crop.levels[0].array[...],
                data0[:, a[1]:b[1], a[2]:b[2], a[3]:b[3]])
            record = zarr.open_group(out, mode="r").attrs["crop_provenance"]
            assert record["coordinate_system"]["cropped_in"] == "physical"


def test_06_identity_and_sequences_are_read_in_order():
    """Each level's voxel size and position come from its transformation
    applied in order: identity = voxel size 1; a sequence of translation
    then scale scales the translation too (x -> 4 (x + 0.5) = 4x + 2)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path, version="0.6",
                        factors=[(1, 1, 1, 1), (1, 1, 4, 4)])
        ome = _ome(path)
        datasets = ome["multiscales"][0]["datasets"]
        ends = lambda p: {"input": {"path": p},
                          "output": {"name": "physical"}}
        datasets[0]["coordinateTransformations"] = [
            {"type": "identity", **ends("0")}]
        datasets[1]["coordinateTransformations"] = [
            {"type": "sequence", **ends("1"), "transformations": [
                {"type": "translation", "translation": [0, 0, 0.5, 0.5]},
                {"type": "scale", "scale": [1, 1, 4, 4]}]}]
        _set_ome(path, ome)
        image = vc.open_image(path)
        assert image.levels[0].scale == [1.0] * 4
        assert image.levels[0].translation == [0.0] * 4
        assert image.levels[1].scale == [1.0, 1.0, 4.0, 4.0]
        assert image.levels[1].translation == [0.0, 0.0, 2.0, 2.0]


def test_06_prerelease_names_are_read_as_06():
    """ngff_zarr 0.43 labels its 0.6 images "0.6.dev4": such pre-release
    names are read with the 0.6 rules, and the crop keeps the name (crops
    keep the source's version as written); "0.7" stays refused (above)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "src.zarr", version="0.6")
        ome = _ome(tmp / "src.zarr")
        ome["version"] = "0.6.dev4"
        _set_ome(tmp / "src.zarr", ome)
        image = vc.open_image(tmp / "src.zarr")
        assert image.coordinate_system == "physical"
        assert image.format_name == "OME-Zarr 0.6.dev4 (Zarr v3)"
        result = vc.crop_box(image, BOX, tmp / "out", min_size=4)
        assert result.problems == []
        assert _ome(result.path)["version"] == "0.6.dev4"
        assert [vc.is_v06(v) for v in ("0.6", "0.6rc0", "0.6.dev2", "0.60",
                                       "0.5", "0.6.1")] == \
            [True, True, True, False, False, False]


def _image_with_links(path):
    """A 0.6 image whose coordinate system "physical" is linked to two
    more systems ("atlas" one way, "stage" the other way) and to things a
    crop does not include: a label image and a field kept in an array."""
    make_test_image(path, version="0.6", translation=(0, 4.0, 1.0, 2.0))
    ome = _ome(path)
    multiscale = ome["multiscales"][0]
    axes = multiscale["coordinateSystems"][0]["axes"]
    multiscale["coordinateSystems"] += [
        {"name": name, "axes": copy.deepcopy(axes)}
        for name in ("atlas", "stage", "warped")]
    multiscale["coordinateTransformations"] = [
        {"type": "scale", "scale": [1, 2, 3, 4], "name": "to atlas",
         "input": {"name": "physical"}, "output": {"name": "atlas"}},
        {"type": "translation", "translation": [0, 100, 200, 300],
         "input": {"name": "stage"}, "output": {"name": "physical"}},
        {"type": "identity", "input": {"name": "physical"},
         "output": {"name": "physical", "path": "labels/cells"}},
        {"type": "displacements", "path": "field",
         "input": {"name": "physical"}, "output": {"name": "warped"}},
    ]
    ome["scene"] = {"coordinateTransformations": []}
    _set_ome(path, ome)


def _apply(transform, point):
    """Apply a scale/translation/sequence transformation to a point."""
    kind = transform["type"]
    if kind == "sequence":
        for step in transform["transformations"]:
            point = _apply(step, point)
        return point
    if kind == "scale":
        return [p * s for p, s in zip(point, transform["scale"])]
    return [p + t for p, t in zip(point, transform["translation"])]


def test_06_links_to_other_systems_are_kept():
    """Decision 5a: the crop keeps the links to other coordinate systems
    (so it lands at the same place in the atlas), corrects them when it
    starts at 0, and leaves out links to a label image or to a field in an
    array, which a crop does not include; the record says so."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _image_with_links(tmp / "src.zarr")
        image = vc.open_image(tmp / "src.zarr")
        links = image.ome["multiscales"][0]["coordinateTransformations"]
        for keep in (True, False):
            plan = vc.plan_crop(image, BOX, min_size=4)
            out = vc.write_crop(image, plan, tmp / f"out{keep}.zarr",
                                keep_position=keep)
            assert vc.verify_crop(image, plan, out,
                                  keep_position=keep) == [], keep
            ome = _ome(out)
            assert "scene" not in ome
            multiscale = ome["multiscales"][0]
            assert [s["name"] for s in multiscale["coordinateSystems"]] \
                == ["physical", "atlas", "stage"]
            kept = multiscale["coordinateTransformations"]
            assert len(kept) == 2
            notes = zarr.open_group(out, mode="r").attrs[
                "crop_provenance"]["coordinate_system"]["notes"]
            assert sum("left out" in note for note in notes) == 2, notes
            if keep:
                assert kept == links[:2]
                continue
            # The crop's first voxel is at 0 in its own coordinates and at
            # `origin` in the source's: both links must take it to the
            # same place as the source's links take `origin`.
            crop = vc.open_image(out)
            origin = [t + a * s for t, a, s in zip(
                image.levels[0].translation, plan.starts[0],
                image.levels[0].scale)]
            origin[0] = 0.0                       # (the channel axis)
            assert crop.levels[0].translation == [0.0] * 4
            to_atlas, from_stage = kept
            assert to_atlas["name"] == "to atlas"
            assert np.allclose(_apply(to_atlas, [0.0] * 4),
                               _apply(links[0], origin))
            stage_point = [1.0, 7.0, 3.0, 5.0]
            assert np.allclose(
                _apply(from_stage, stage_point),
                np.subtract(_apply(links[1], stage_point), origin))


def test_06_refusals():
    """Clear errors for what the tool cannot open: an OME-Zarr version it
    does not know, levels that lead to different coordinate systems or to
    one that is not defined, and a level with an affine transformation."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "src.zarr"
        make_test_image(path, version="0.6")
        original = _ome(path)

        def refused(change, expected, error=ValueError):
            ome = copy.deepcopy(original)
            change(ome)
            _set_ome(path, ome)
            try:
                vc.open_image(path)
            except error as err:
                assert expected in str(err), err
            else:
                raise AssertionError(f"opened: {expected}")

        def version(ome):
            ome["version"] = "0.7"

        def two_systems(ome):
            ome["multiscales"][0]["datasets"][1][
                "coordinateTransformations"][0]["output"]["name"] = "other"

        def undefined(ome):
            ome["multiscales"][0]["coordinateSystems"][0]["name"] = "other"

        def affine(ome):
            ome["multiscales"][0]["datasets"][0][
                "coordinateTransformations"][0] = {
                "type": "affine", "affine": [[1, 0], [0, 1]],
                "input": {"path": "0"}, "output": {"name": "physical"}}

        refused(version, "OME-Zarr 0.7")
        refused(two_systems, "one coordinate system")
        refused(undefined, "do not define it")
        refused(affine, "'affine'", NotImplementedError)


# ---------------------------------------------------------------------------
# Zipped OME-Zarr (RFC-9 .ozx and other zips)
# ---------------------------------------------------------------------------

def _zip_folder(folder: Path, zip_path: Path, inside: str = "",
                compression=zipfile.ZIP_STORED) -> Path:
    """Zip a folder the way generic tools do (any order; optionally
    inside a folder, as zipping a folder in Explorer does; compressed)."""
    with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                name = path.relative_to(folder).as_posix()
                zf.write(path, f"{inside}/{name}" if inside else name)
    return zip_path


def _same_image(a, b) -> None:
    """Two opened images hold the same levels, positions and voxels."""
    assert a.axes == b.axes and a.ome_format == b.ome_format
    assert len(a.levels) == len(b.levels)
    for la, lb in zip(a.levels, b.levels):
        assert (la.scale, la.translation) == (lb.scale, lb.translation)
        np.testing.assert_array_equal(la.array[...], lb.array[...])


def test_zipped_images_open_like_folders():
    """An .ozx packed by the tool, a .zip with the image in a folder inside
    (compressed, as Explorer makes them: it opens, with a warning that it
    is slower), and a zipped 0.4 image give the same image as the folder."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for version in ("0.5", "0.6", "0.4"):
            folder = tmp / f"img{version}.zarr"
            make_test_image(folder, version=version,
                            factors=[(1, 1, 1, 1), (1, 1, 4, 4)])
            image = vc.open_image(folder)
            ozx = vc.pack_zip(folder, tmp / f"img{version}.ozx")
            zipped = vc.open_image(ozx)
            try:
                assert zipped.container == "ozx"
                assert zipped.format_name.endswith("zipped (.ozx)")
                _same_image(zipped, image)
            finally:
                zipped.close()
            wrapped = _zip_folder(folder, tmp / f"img{version}.ome.zarr.zip",
                                  inside=folder.name,
                                  compression=zipfile.ZIP_DEFLATED)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                zipped = vc.open_image(wrapped)
            assert any("compressed zip" in str(w.message) for w in caught)
            try:
                assert zipped.container == "zip"
                _same_image(zipped, image)
                assert vc.image_stem(zipped) == f"img{version}"
            finally:
                zipped.close()


def test_zip_files_that_hold_no_image_are_refused():
    """Clear errors for a zip inside a zip (RFC-9 forbids it), a zip with
    no image, and a file that only has a zip's name."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "img.zarr", factors=[(1, 1, 1, 1)])
        inner = vc.pack_zip(tmp / "img.zarr", tmp / "inner.ozx")
        with zipfile.ZipFile(tmp / "outer.ozx", "w") as zf:
            zf.write(inner, "inner.ozx")
        with zipfile.ZipFile(tmp / "empty.zip", "w") as zf:
            zf.writestr("notes.txt", "no image here")
        (tmp / "fake.ozx").write_bytes(b"not a zip file")
        for name, expected in (("outer.ozx", "holds another zip file"),
                               ("empty.zip", "holds no OME-Zarr image"),
                               ("fake.ozx", "cannot be read as a zip")):
            try:
                vc.open_image(tmp / name)
            except ValueError as err:
                assert expected in str(err), err
            else:
                raise AssertionError(f"opened {name}")


def _check_rfc9_layout(path: Path, version: str) -> None:
    """The layout RFC-9 recommends: zarr.json first, the other metadata
    files next (breadth-first), nothing compressed, ZIP64 local headers,
    and the version (and jsonFirst) in the zip's comment."""
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        comment = json.loads(zf.comment)
    names = [info.filename for info in infos]
    assert names[0] == "zarr.json"
    meta = [n for n in names if n.endswith("zarr.json")]
    assert names[:len(meta)] == meta
    assert [n.count("/") for n in meta] == sorted(n.count("/") for n in meta)
    assert all(info.compress_type == zipfile.ZIP_STORED for info in infos)
    assert comment == {"ome": {"version": version, "zipFile": {
        "centralDirectory": {"jsonFirst": True}}}}
    # (ZIP64: the local header of every entry has the ZIP64 extra field,
    # id 0x0001, as ozx-tck's valid files do)
    with open(path, "rb") as f:
        for info in infos[:5]:
            f.seek(info.header_offset + 26)
            name_len = int.from_bytes(f.read(2), "little")
            extra_len = int.from_bytes(f.read(2), "little")
            f.seek(name_len, 1)
            extra = f.read(extra_len)
            assert extra[:2] == b"\x01\x00", info.filename


def test_crops_of_zipped_images_are_zipped():
    """Decisions 2a and 3a: crops of an .ozx (0.5, 0.6) are .ozx files in
    RFC-9's layout; a zipped 0.4's crops keep its ending. They hold the
    source's voxels, pass the checks (also read through the zip), keep
    their record, and their TIFF and positions are named after them."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for version, name, suffix in (("0.5", "a.ozx", ".ozx"),
                                      ("0.6", "b.zip", ".ozx"),
                                      ("0.4", "c.ome.zarr.zip",
                                       ".ome.zarr.zip")):
            folder = tmp / f"{version}.zarr"
            out = tmp / f"out{version}"
            data0 = make_test_image(folder, version=version)
            source = vc.pack_zip(folder, tmp / name)
            image = vc.open_image(source)
            try:
                result = vc.crop_box(image, BOX, out, min_size=4,
                                     tiff_level=1)
            finally:
                image.close()
            assert result.problems == [] and result.tiff_problems == []
            stem = vc.image_stem(image)
            assert result.path == out / f"{stem}_b1{suffix}"
            assert result.tiff == out / f"{stem}_b1_level1.tif"
            assert "zip file" in result.checked
            if version != "0.4":
                _check_rfc9_layout(result.path, version)
            crop = vc.open_image(result.path)
            try:
                a, b = result.plan.starts[0], result.plan.stops[0]
                np.testing.assert_array_equal(
                    crop.levels[0].array[...],
                    data0[:, a[1]:b[1], a[2]:b[2], a[3]:b[3]])
                record = crop.attrs["crop_provenance"]
                assert record["complete"] and record["verified"]["ok"]
                assert record["source_format"] == image.format_name
            finally:
                crop.close()
            # nothing left behind: no scratch folder, no .part file
            assert sorted(p.name for p in out.iterdir()) == \
                sorted([result.path.name, result.tiff.name])


def test_zipped_crops_are_replaced_only_when_allowed():
    """A zipped crop is replaced only with Overwrite, only if it is the
    tool's crop of the same region, and a foreign zip of that name is
    left alone; write_crop + verify_crop store the check in the zip."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "img.zarr")
        image = vc.open_image(vc.pack_zip(tmp / "img.zarr", tmp / "img.ozx"))
        try:
            first = vc.crop_box(image, BOX, tmp / "out", min_size=4)
            for overwrite, expected in ((False, "already exists"),):
                try:
                    vc.crop_box(image, BOX, tmp / "out", min_size=4,
                                overwrite=overwrite)
                except FileExistsError as err:
                    assert expected in str(err), err
                else:
                    raise AssertionError("replaced without Overwrite")
            again = vc.crop_box(image, BOX, tmp / "out", min_size=4,
                                overwrite=True)
            assert again.path == first.path and again.problems == []
            other = vc.CropBox("b1 ", BOX.ranges)     # same file name
            try:
                vc.crop_box(image, other, tmp / "out", min_size=4,
                            overwrite=True)
            except FileExistsError as err:
                assert "holds the crop of box 'b1'" in str(err), err
            else:
                raise AssertionError("replaced another region's crop")
            foreign = tmp / "out2" / first.path.name
            foreign.parent.mkdir()
            _zip_folder(tmp / "img.zarr", foreign)     # not a crop
            try:
                vc.crop_box(image, BOX, tmp / "out2", min_size=4,
                            overwrite=True)
            except FileExistsError as err:
                assert "not written by Volume Cropper" in str(err), err
            else:
                raise AssertionError("replaced a foreign zip")
            # write_crop and verify_crop called one after the other
            plan = vc.plan_crop(image, BOX, min_size=4)
            out = vc.write_crop(image, plan, tmp / "out3" / "x.ozx")
            assert vc.verify_crop(image, plan, out) == []
            assert vc.zip_record(out)["verified"]["ok"] is True
            _check_rfc9_layout(out, "0.5")
        finally:
            image.close()


def test_zipped_crop_cancel_and_work_folder():
    """Cancel leaves no zip, no .part and no work folder; when the
    temporary folder's drive is too full, the work folder goes next to
    the output instead (and is removed afterwards)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_test_image(tmp / "img.zarr")
        image = vc.open_image(vc.pack_zip(tmp / "img.zarr", tmp / "img.ozx"))
        cancel = threading.Event()
        cancel.set()
        try:
            vc.crop_box(image, BOX, tmp / "out", min_size=4,
                        cancel_event=cancel)
        except vc.CropCancelled as err:
            assert "was not saved" in str(err), err
        else:
            raise AssertionError("not cancelled")
        assert list((tmp / "out").iterdir()) == []
        margin = vc.ZIP_WORK_MARGIN
        vc.ZIP_WORK_MARGIN = 2**62         # "no room" in the temp folder
        try:
            assert vc.zip_work_parent(tmp / "out" / "x.ozx", 1) \
                == tmp / "out"
            result = vc.crop_box(image, BOX, tmp / "out", min_size=4)
        finally:
            vc.ZIP_WORK_MARGIN = margin
            image.close()
        assert result.problems == []
        assert [p.name for p in (tmp / "out").iterdir()] == [result.path.name]


# ---------------------------------------------------------------------------
# Collections of images: bioformats2raw layout, OME-Zarr 0.6 scene
# ---------------------------------------------------------------------------

SMALL = dict(shape0=(2, 12, 40, 36), factors=[(1, 1, 1, 1), (1, 1, 2, 2)])


def _bioformats2raw(path: Path, version: str, n: int,
                    series: list[str] | None = None) -> Path:
    """A bioformats2raw collection of n small images (groups 0, 1, ...),
    with an OME group that lists `series` if given."""
    v3 = version != "0.4"
    root = zarr.open_group(path, mode="w", zarr_format=3 if v3 else 2)
    layout = {"bioformats2raw.layout": 3}
    if v3:
        root.attrs["ome"] = {"version": version, **layout}
    else:
        root.attrs.update(layout)
    for k in range(n):
        make_test_image(path / str(k), version=version, seed=k, **SMALL)
    if series is not None:
        ome_group = root.create_group("OME")
        if v3:
            ome_group.attrs["ome"] = {"version": version, "series": series}
        else:
            ome_group.attrs["series"] = series
    return path


def test_collections_offer_their_images():
    """A collection of several images (bioformats2raw, numbered or listed
    as "series"; an OME-Zarr 0.6 scene) asks which image: open_image
    lists them (ImageChoiceNeeded), opens the one named, and refuses one
    it does not hold; a collection of one opens it; a scene's paths that
    lead outside it are not followed."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        numbered = _bioformats2raw(tmp / "b2r.zarr", "0.4", 2)
        try:
            vc.open_image(numbered)
        except vc.ImageChoiceNeeded as err:
            assert err.images == [("0", "0 (test)"), ("1", "1 (test)")]
            assert err.kind == "bioformats2raw layout"
        else:
            raise AssertionError("opened a collection of two images")
        image = vc.open_image(numbered, inner="1")
        assert (image.inner, image.path, image.ome_version) == \
            ("1", numbered, "0.4")
        assert vc.image_stem(image) == "b2r_1"
        for inner in ("2", "../b2r.zarr/0"):
            try:
                vc.open_image(numbered, inner=inner)
            except ValueError as err:
                assert "has no image" in str(err), err
            else:
                raise AssertionError(f"opened image {inner!r}")
        listed = _bioformats2raw(tmp / "listed.zarr", "0.5", 2, ["1"])
        assert vc.open_image(listed).inner == "1"

        # A 0.6 scene that places two tiles (and names a path outside it)
        scene = tmp / "scene.zarr"
        root = zarr.open_group(scene, mode="w", zarr_format=3)
        for k in range(2):
            make_test_image(scene / f"tile_{k}", version="0.6", seed=k,
                            **SMALL)
        root.attrs["ome"] = {"version": "0.6", "scene": {
            "coordinateTransformations": [
                {"type": "translation", "translation": [0, 0, 0, 10 * k],
                 "input": {"path": path, "name": "physical"},
                 "output": {"name": "world"}}
                for k, path in enumerate(["tile_0", "tile_1", "../x.zarr"])],
            "coordinateSystems": [{"name": "world", "axes": []}]}}
        try:
            vc.open_image(scene)
        except vc.ImageChoiceNeeded as err:
            assert [p for p, _ in err.images] == ["tile_0", "tile_1"]
            assert err.kind == "OME-Zarr 0.6 scene"
        else:
            raise AssertionError("opened a scene of two images")
        image = vc.open_image(scene, inner="tile_1")
        assert image.coordinate_system == "physical"
        try:
            vc.open_image(tmp / "b2r.zarr" / "0", inner="0")
        except ValueError as err:
            assert "not a collection" in str(err), err
        else:
            raise AssertionError("an image inside a single image")


def test_crops_of_collection_images():
    """A crop of one image of a collection (here zipped: an .ozx of a
    bioformats2raw layout) is named after the collection and the image,
    records which image it is of, and does not replace the crop of
    another image of the same collection."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        folder = _bioformats2raw(tmp / "plate.zarr", "0.5", 2)
        ozx = vc.pack_zip(folder, tmp / "plate.ozx")
        box = vc.CropBox("b", {"y": (2.0, 6.0), "x": (1.0, 5.0)})
        results = []
        for inner in ("0", "1"):
            image = vc.open_image(ozx, inner=inner)
            try:
                results.append(vc.crop_box(image, box, tmp / "out",
                                           min_size=4))
            finally:
                image.close()
        assert [r.path.name for r in results] == ["plate_0_b.ozx",
                                                  "plate_1_b.ozx"]
        assert all(r.problems == [] for r in results)
        record = vc.zip_record(results[1].path)
        assert (record["source_image"], record["source_collection"]) == \
            ("1", "bioformats2raw layout")
        # the crop of image 1 is not "the same image" as image 0's
        image = vc.open_image(ozx, inner="0")
        try:
            vc._refuse_other_crop(results[1].path, record, "b",
                                  vc._image_identity(image))
        except FileExistsError as err:
            assert "image 1 of plate.ozx" in str(err), err
        else:
            raise AssertionError("image 0 may replace image 1's crop")
        finally:
            image.close()


def test_command_line_with_zipped_collection():
    """The command line crops one image of a zipped collection, chosen
    with --collection-image, into an .ozx; without it, it stops (exit 2)
    and names the option and the images."""
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ozx = vc.pack_zip(_bioformats2raw(tmp / "plate.zarr", "0.5", 2),
                          tmp / "plate.ozx")
        csv_path = tmp / "boxes.csv"
        vc.save_boxes_csv([vc.CropBox("b", {"y": (2.0, 6.0),
                                            "x": (1.0, 5.0)})],
                          csv_path, vc.open_image(ozx, inner="0"))

        def run(*options):
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(err):
                code = vc.main([str(ozx), str(csv_path), str(tmp / "out"),
                                "--min-size", "4", *options])
            return code, err.getvalue()

        code, err = run()
        assert code == 2 and "--collection-image 0" in err, (code, err)
        code, err = run("--collection-image", "1")
        assert code == 0, err
        crop = tmp / "out" / "plate_1_b.ozx"
        assert vc.zip_record(crop)["verified"]["ok"] is True


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"all {len(tests)} tests passed")
