"""Tests for file_images (0.12.0): TIFF files read into memory, and their
crops saved as TIFF.

Run with ``python run_all_tests.py files`` (or this file alone). Every test
writes small synthetic TIFFs into a temporary folder and checks that:
- what a file holds (axes, voxel size, frame interval, channel names and
  colours) is read as the file states it (Fiji hyperstacks, OME-TIFF, plain
  TIFF, folders of TIFFs, compressed or not);
- the part chosen is read exactly (voxel for voxel) and sits at its place
  in the file's coordinates;
- the smaller copies for the views are block means at the right positions;
- crops are saved as TIFF of the file's kind (Fiji or OME), with the right
  voxels and calibration, compressed with zstd when asked, and checked;
- outputs are never written over files the tool did not write.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from volume_cropper import file_images as fi
from volume_cropper import core as vc

RNG = np.random.default_rng(0)


def green_magenta_luts():
    ramp = np.arange(256, dtype=np.uint8)
    zero = np.zeros(256, dtype=np.uint8)
    return [np.stack([zero, ramp, zero]), np.stack([ramp, zero, ramp])]


def write_hyperstack(path, data, *, spacing=2.0, finterval=30.0, xy=0.5,
                     compression=None):
    """A Fiji hyperstack TZCYX (uint16) with calibration and colours."""
    tifffile.imwrite(
        path, data, imagej=True, resolution=(1 / xy, 1 / xy),
        compression=compression,
        metadata={"axes": "TZCYX", "spacing": spacing, "unit": "um",
                  "finterval": finterval, "LUTs": green_magenta_luts(),
                  "Ranges": (0.0, 100.0, 5.0, 900.0), "mode": "composite"})


def hyperstack(shape=(3, 9, 2, 40, 52)):
    return RNG.integers(0, 1000, size=shape, dtype=np.uint16)


def test_hyperstack_metadata_and_whole_read():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "movie.tif"
        data = hyperstack()
        write_hyperstack(path, data)
        reader = fi.open_file(path)
        try:
            assert reader.format_name.startswith("TIFF (Fiji")
            assert reader.output_kind == "imagej"
            scene = reader.scenes[0]
            assert scene.axes == "TZCYX" and scene.shape == data.shape
            assert scene.unit == "micrometer"
            assert np.isclose(scene.voxel["x"], 0.5)
            assert np.isclose(scene.voxel["z"], 2.0)
            assert scene.interval == 30.0 and scene.time_unit == "second"
            assert scene.channel_colors == ["00FF00", "FF00FF"]
            assert scene.channel_windows == [(0.0, 100.0), (5.0, 900.0)]
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        assert image.in_memory and image.axis_names == ["t", "c", "z", "y", "x"]
        level0 = image.levels[0]
        # file order TZCYX, image order TCZYX
        assert np.array_equal(level0.array[...], data.transpose(0, 2, 1, 3, 4))
        assert np.allclose(level0.scale, [30.0, 1.0, 2.0, 0.5, 0.5])
        assert level0.translation == [0.0] * 5
        assert image.channel_labels == ["channel 0", "channel 1"]
        assert image.source["output"] == "imagej"
        assert vc.output_suffix(image) == ".tif"
        assert vc.image_stem(image) == "movie"


def test_part_read_sits_at_its_place():
    """Channels and ranges: exactly that part is read, placed at its
    position in the file (first index x voxel size)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "movie.tif"
        data = hyperstack()
        write_hyperstack(path, data)
        reader = fi.open_file(path)
        choice = fi.FileChoice(channels=[1], ranges={
            "t": (1, 3), "z": (2, 7), "y": (10, 33), "x": (5, 50)})
        try:
            image = fi.load(reader, choice)
            needed = fi.memory_needed(reader, choice)
        finally:
            reader.close()
        expected = data[1:3, 2:7, 1:2, 10:33, 5:50].transpose(0, 2, 1, 3, 4)
        assert np.array_equal(image.levels[0].array[...], expected)
        assert np.allclose(image.levels[0].translation,
                           [30.0, 0.0, 4.0, 5.0, 2.5])
        assert image.n_channels == 1
        assert sum(lv.array.nbytes for lv in image.levels) == needed
        assert image.source["loaded"] == {
            "t": [1, 3], "c": [1], "z": [2, 7], "y": [10, 33], "x": [5, 50]}


def test_pyramid_means_and_positions():
    """Each smaller copy is the mean of 2 x 2 (x 2) voxels, placed at their
    middle; z halves only once its voxels would be thinner than x's."""
    levels = fi.plan_levels((40, 1024, 1024), (3.0, 0.5, 0.5))
    assert [f for f, _, _ in levels] == [(1, 1, 1), (1, 2, 2), (1, 2, 2)]
    assert levels[-1][1] == (40, 256, 256)
    iso = fi.plan_levels((100, 600, 600), (1.0, 1.0, 1.0))
    assert [f for f, _, _ in iso] == [(1, 1, 1), (2, 2, 2), (2, 2, 2)]
    assert iso[-1][1] == (25, 150, 150)          # rounded up: 600/2/2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stack.tif"
        data = RNG.integers(0, 60000, size=(9, 601, 520), dtype=np.uint16)
        tifffile.imwrite(path, data, imagej=True, resolution=(2.0, 2.0),
                         metadata={"axes": "ZYX", "spacing": 0.5,
                                   "unit": "um"})
        reader = fi.open_file(path)
        try:
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
    assert len(image.levels) == 3
    one = image.levels[1]
    # z voxel 0.5 = x voxel: z halves too
    assert one.array.shape == (5, 301, 260)
    padded = np.pad(data, ((0, 1), (0, 1), (0, 0)), mode="edge")
    sums = padded.reshape(5, 2, 301, 2, 260, 2).sum(axis=(1, 3, 5),
                                                    dtype=np.int64)
    expected = ((sums + 4) // 8).astype(np.uint16)   # mean, halves up
    assert np.array_equal(one.array[...], expected)
    assert np.allclose(one.scale, [1.0, 1.0, 1.0])
    assert np.allclose(one.translation, [0.25, 0.25, 0.25])


def _companion_set(folder: Path, binary_only: bool = False,
                   subfolder: str = "",
                   companion: str = "embryo.companion.ome",
                   shaped: bool = False):
    """An OME-TIFF set as Viventis and Bio-Formats write them: the OME
    metadata in embryo.companion.ome, the planes in two TIFF files (one per
    channel). With binary_only, each TIFF names the companion in its own
    OME-XML ("BinaryOnly"). (0.16.0 review) subfolder: the TIFFs are in
    that folder, named from the companion as "subfolder/name"; companion:
    the companion's file name; shaped: the TIFFs carry tifffile's own
    description (as tifffile.imwrite writes by default). Returns the data
    (c, z, y, x)."""
    import uuid
    data = RNG.integers(0, 4000, size=(2, 4, 16, 20), dtype=np.uint16)
    where = folder / subfolder if subfolder else folder
    where.mkdir(parents=True, exist_ok=True)
    prefix = f"{subfolder}/" if subfolder else ""
    ids = {}
    for name, c in (("embryo_c0.ome.tif", 0), ("embryo_c1.ome.tif", 1)):
        ids[name] = f"urn:uuid:{uuid.uuid4()}"
        description = None
        if binary_only:
            description = (
                '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://'
                'www.openmicroscopy.org/Schemas/OME/2016-06" UUID="'
                f'{ids[name]}"><BinaryOnly MetadataFile="{companion}" '
                f'UUID="urn:uuid:{uuid.uuid4()}"/></OME>')
        if shaped:
            tifffile.imwrite(where / name, data[c], photometric="minisblack")
        else:
            tifffile.imwrite(where / name, data[c], photometric="minisblack",
                             description=description, metadata=None)
    tiffdata = "".join(
        f'<TiffData FirstZ="0" FirstC="{c}" FirstT="0" IFD="0" '
        f'PlaneCount="4"><UUID FileName="{prefix}{name}">{ids[name]}</UUID>'
        '</TiffData>'
        for name, c in (("embryo_c0.ome.tif", 0), ("embryo_c1.ome.tif", 1)))
    (folder / companion).write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0" Name="embryo"><Pixels ID="Pixels:0" '
        'DimensionOrder="XYZCT" Type="uint16" SizeX="20" SizeY="16" '
        'SizeZ="4" SizeC="2" SizeT="1" PhysicalSizeX="0.5" '
        'PhysicalSizeY="0.5" PhysicalSizeZ="2.0">'
        '<Channel ID="Channel:0:0" Name="gfp" SamplesPerPixel="1"/>'
        '<Channel ID="Channel:0:1" Name="dapi" SamplesPerPixel="1"/>'
        f'{tiffdata}</Pixels></Image></OME>', encoding="utf-8")
    return data


def test_ome_tiff_companion_file():
    """0.16.0: an OME-TIFF set whose metadata are in a companion file
    opens from the companion (and from one of its TIFFs when that names
    the companion): the image across both TIFFs, its calibration and
    channel names; crops are named after the set, not the companion."""
    for binary_only in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = _companion_set(tmp, binary_only)
            starts = [tmp / "embryo.companion.ome"]
            if binary_only:
                starts.append(tmp / "embryo_c1.ome.tif")
            for start in starts:
                reader = fi.open_file(start)
                try:
                    assert "companion" in reader.format_name
                    assert reader.path == tmp / "embryo.companion.ome"
                    scene = reader.scenes[0]
                    assert scene.shape == (2, 4, 16, 20), scene.shape
                    assert np.isclose(scene.voxel["x"], 0.5)
                    assert np.isclose(scene.voxel["z"], 2.0)
                    assert scene.channel_names == ["gfp", "dapi"]
                    image = fi.load(reader, fi.FileChoice())
                finally:
                    reader.close()
                np.testing.assert_array_equal(
                    np.asarray(image.levels[0].array[:]), data)
            box = vc.CropBox("half", {"x": (0.0, 5.0)})
            result = vc.crop_box(image, box, tmp / "out")
            assert result.problems == [], result.problems
            assert result.path.name == "embryo_half.ome.tif"


def test_companion_variants():
    """The review of 2026-11-08: a companion whose TIFFs are in a
    subfolder ("data/embryo_c0.ome.tif"), TIFFs with tifffile's own
    description, and a companion named .companion.ome.xml all open as the
    whole set (both channels), not as the first TIFF alone; the crop of a
    .companion.ome.xml set is named after the set."""
    cases = [{"subfolder": "data"}, {"shaped": True},
             {"subfolder": "data", "shaped": True},
             {"companion": "embryo.companion.ome.xml"}]
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = _companion_set(tmp, **case)
            name = case.get("companion", "embryo.companion.ome")
            reader = fi.open_file(tmp / name)
            try:
                scene = reader.scenes[0]
                assert scene.shape == (2, 4, 16, 20), (case, scene.shape)
                assert scene.channel_names == ["gfp", "dapi"], case
                assert all(s.kind.lower() == "ome"
                           for s in reader._tif.series), case
                image = fi.load(reader, fi.FileChoice())
            finally:
                reader.close()
            np.testing.assert_array_equal(
                np.asarray(image.levels[0].array[:]), data)
            if "companion" in case:
                box = vc.CropBox("half", {"x": (0.0, 5.0)})
                result = vc.crop_box(image, box, tmp / "out")
                assert result.path.name == "embryo_half.ome.tif", \
                    result.path.name


def test_binary_only_pointing_to_a_master_ome_tiff():
    """The review of 2026-11-08: a TIFF whose OME-XML says BinaryOnly and
    names a master OME-TIFF (which holds the whole set's metadata, as the
    OME-TIFF specification allows) opens as the whole set from the
    master, as in 0.15.0, without reading the master as text."""
    import uuid
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = RNG.integers(0, 4000, size=(2, 4, 16, 20), dtype=np.uint16)
        ids = {n: f"urn:uuid:{uuid.uuid4()}"
               for n in ("set_c0.ome.tif", "set_c1.ome.tif")}
        tiffdata = "".join(
            f'<TiffData FirstZ="0" FirstC="{c}" FirstT="0" IFD="0" '
            f'PlaneCount="4"><UUID FileName="{n}">{ids[n]}</UUID>'
            '</TiffData>'
            for n, c in (("set_c0.ome.tif", 0), ("set_c1.ome.tif", 1)))
        master = (
            '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.'
            'openmicroscopy.org/Schemas/OME/2016-06" UUID="'
            f'{ids["set_c0.ome.tif"]}"><Image ID="Image:0" Name="set">'
            '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" '
            'SizeX="20" SizeY="16" SizeZ="4" SizeC="2" SizeT="1" '
            'PhysicalSizeX="0.5" PhysicalSizeY="0.5" PhysicalSizeZ="2.0">'
            '<Channel ID="Channel:0:0" Name="gfp" SamplesPerPixel="1"/>'
            '<Channel ID="Channel:0:1" Name="dapi" SamplesPerPixel="1"/>'
            f'{tiffdata}</Pixels></Image></OME>')
        partial = (
            '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.'
            'openmicroscopy.org/Schemas/OME/2016-06" UUID="'
            f'{ids["set_c1.ome.tif"]}"><BinaryOnly MetadataFile='
            f'"set_c0.ome.tif" UUID="{ids["set_c0.ome.tif"]}"/></OME>')
        tifffile.imwrite(tmp / "set_c0.ome.tif", data[0],
                         photometric="minisblack", description=master,
                         metadata=None)
        tifffile.imwrite(tmp / "set_c1.ome.tif", data[1],
                         photometric="minisblack", description=partial,
                         metadata=None)
        reader = fi.open_file(tmp / "set_c1.ome.tif")
        try:
            assert reader.path == tmp / "set_c0.ome.tif"
            assert reader.scenes[0].shape == (2, 4, 16, 20)
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        np.testing.assert_array_equal(
            np.asarray(image.levels[0].array[:]), data)


def test_companion_with_a_missing_tiff():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _companion_set(tmp)
        (tmp / "embryo_c1.ome.tif").unlink()
        try:
            fi.open_file(tmp / "embryo.companion.ome")
        except FileNotFoundError as err:
            assert "embryo_c1.ome.tif" in str(err)
        else:
            raise AssertionError("a companion with a missing TIFF opened")


def test_ome_tiff_with_several_images():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "two.ome.tif"
        a = RNG.integers(0, 255, size=(2, 5, 30, 40), dtype=np.uint8)   # CZYX
        b = RNG.integers(0, 4000, size=(4, 20, 24), dtype=np.uint16)    # TYX
        with tifffile.TiffWriter(path, ome=True) as tif:
            tif.write(a, metadata={
                "axes": "CZYX", "Name": "embryo 1", "PhysicalSizeX": 0.65,
                "PhysicalSizeY": 0.65, "PhysicalSizeZ": 2.5,
                "Channel": {"Name": ["nuclei", "actin"],
                            "Color": [-16776961, 16711935]}})
            tif.write(b, metadata={
                "axes": "TYX", "Name": "movie", "PhysicalSizeX": 0.1,
                "PhysicalSizeXUnit": "mm", "PhysicalSizeY": 0.1,
                "PhysicalSizeYUnit": "mm", "TimeIncrement": 2.0,
                "TimeIncrementUnit": "min"})
        reader = fi.open_file(path)
        try:
            assert reader.format_name == "OME-TIFF"
            assert [s.name for s in reader.scenes] == ["embryo 1", "movie"]
            first, second = reader.scenes
            assert first.unit == "micrometer"
            assert np.isclose(first.voxel["z"], 2.5)
            assert first.channel_names == ["nuclei", "actin"]
            assert first.channel_colors == ["FF0000", "00FF00"]
            assert second.unit == "millimeter" and second.voxel["z"] is None
            assert second.interval == 2.0 and second.time_unit == "minute"
            image = fi.load(reader, fi.FileChoice(scene=1))
        finally:
            reader.close()
        # a 2D movie: one plane of z, as thick as a pixel is wide
        assert image.axis_names == ["t", "z", "y", "x"]
        assert image.levels[0].array.shape == (4, 1, 20, 24)
        assert np.array_equal(image.levels[0].array[:, 0], b)
        assert np.allclose(image.levels[0].scale, [2.0, 0.1, 0.1, 0.1])
        assert image.unit("x") == "millimeter"
        assert image.unit("t") == "minute"
        assert image.source["has_z"] is False
        assert vc.image_stem(image) == "two_movie"
        assert vc.output_suffix(image) == ".ome.tif"


def test_plain_tiff_axes_and_calibration_chosen():
    """A TIFF without metadata: its third axis is unknown (z by default),
    and its voxel size is entered."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plain.tif"
        data = RNG.integers(0, 255, size=(7, 32, 40), dtype=np.uint8)
        tifffile.imwrite(path, data, photometric="minisblack", metadata=None)
        reader = fi.open_file(path)
        try:
            scene = reader.scenes[0]
            assert scene.axes == "?YX" and scene.unit is None
            assert fi.default_axes(scene) == "ZYX"
            assert any("does not say" in note for note in scene.notes)
            calib = fi.calibration_of(scene, fi.FileChoice())
            assert calib["voxel"] == {"z": 1.0, "y": 1.0, "x": 1.0}
            assert calib["source"]["x"] == "assumed"
            choice = fi.FileChoice(axes="TYX", voxel={"y": 0.2, "x": 0.2},
                                   interval=5.0, time_unit="second")
            image = fi.load(reader, choice)
            for wrong in ("ZZX", "TYC", "TY"):
                try:
                    fi.axes_of(scene, fi.FileChoice(axes=wrong))
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{wrong} was accepted")
        finally:
            reader.close()
        assert image.axis_names == ["t", "z", "y", "x"]
        assert np.array_equal(image.levels[0].array[:, 0], data)
        assert np.allclose(image.levels[0].scale, [5.0, 0.2, 0.2, 0.2])
        assert image.source["calibration_from"]["x"] == "entered"


def test_folder_of_tiffs_in_name_order():
    """Files img1 ... img10 in number order (not img1, img10, img2)."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "planes"
        folder.mkdir()
        planes = RNG.integers(0, 4000, size=(10, 24, 30), dtype=np.uint16)
        for k, plane in enumerate(planes, start=1):
            tifffile.imwrite(folder / f"img{k}.tif", plane, imagej=True,
                             resolution=(1 / 0.3, 1 / 0.3),
                             metadata={"unit": "micron"})
        assert fi.is_file_image(folder)
        reader = fi.open_file(folder)
        try:
            scene = reader.scenes[0]
            assert reader.format_name == "TIFF sequence (10 files)"
            assert scene.axes == "?YX" and scene.shape == (10, 24, 30)
            assert np.isclose(scene.voxel["x"], 0.3)
            assert scene.voxel["z"] is None        # one plane has no spacing
            image = fi.load(reader, fi.FileChoice(
                voxel={"z": 1.5}, ranges={"z": (2, 9)}))
        finally:
            reader.close()
        assert np.array_equal(image.levels[0].array[...], planes[2:9])
        assert np.allclose(image.levels[0].scale, [1.5, 0.3, 0.3])
        assert np.allclose(image.levels[0].translation, [3.0, 0.0, 0.0])


def test_compressed_tiff_is_read():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "zstd.tif"
        data = hyperstack((1, 5, 2, 30, 30))
        write_hyperstack(path, data, compression="zstd")
        reader = fi.open_file(path)
        try:
            assert reader.compression == "zstd"
            assert reader.scenes[0].axes == "ZCYX"     # one timepoint: no t
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        assert np.array_equal(image.levels[0].array[...],
                              data[0].transpose(1, 0, 2, 3))


def test_crop_to_fiji_tiff():
    """A Fiji TIFF's crop is a Fiji TIFF: the region's voxels, calibration
    with its position, channels kept, and a record; checked OK."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "movie.tif"
        data = hyperstack()
        write_hyperstack(path, data)
        reader = fi.open_file(path)
        try:
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        box = vc.CropBox("cell 1", {"t": (30, 60), "z": (2, 10),
                                    "y": (5, 15), "x": (2, 20)},
                         channels=[1])
        out = Path(tmp) / "out"
        out.mkdir()
        result = vc.crop_box(image, box, out)
        assert result.problems == [], result.problems
        assert result.path == out / "movie_cell_1.tif"
        plan = result.plan
        s, e = plan.starts[0], plan.stops[0]           # t c z y x
        expected = data[s[0]:e[0], s[2]:e[2], 1:2, s[3]:e[3], s[4]:e[4]]
        with tifffile.TiffFile(result.path) as tif:
            assert tif.is_imagej
            meta = tif.imagej_metadata
            assert meta["spacing"] == 2.0 and meta["finterval"] == 30.0
            stored = tif.asarray().reshape(expected.shape)
            assert tif.pages.first.compression == 1      # not compressed
        assert np.array_equal(stored, expected)
        record = vc._tiff_record(result.path)
        assert record["box_name"] == "cell 1"
        assert record["source_file"]["file"] == str(path)
        assert record["verified"]["ok"] is True
        # The same region again: refused without overwrite, replaced with it
        try:
            vc.crop_box(image, box, out)
        except FileExistsError:
            pass
        else:
            raise AssertionError("an existing crop was overwritten")
        assert vc.crop_box(image, box, out, overwrite=True).problems == []


def test_crop_to_ome_tiff_with_zstd():
    """An OME-TIFF's crop is an OME-TIFF; zstd with the predictor; names,
    colours and voxel size kept; a polygon blacks out voxels outside."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "embryo.ome.tif"
        data = RNG.integers(1, 255, size=(2, 6, 40, 50), dtype=np.uint8)
        tifffile.imwrite(path, data, ome=True, metadata={
            "axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
            "PhysicalSizeZ": 1.0, "Channel": {"Name": ["dapi", "gfp"]}})
        reader = fi.open_file(path)
        try:
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        triangle = [(2.0, 2.0), (18.0, 3.0), (10.0, 22.0)]
        box = vc.CropBox("tri", {"z": (1, 4)}, outlines={("y", "x"): triangle})
        out = Path(tmp) / "out"
        result = vc.crop_box(image, box, out, compression="zstd")
        assert result.problems == [], result.problems
        assert result.path.name == "embryo_tri.ome.tif"
        with tifffile.TiffFile(result.path) as tif:
            assert tif.is_ome
            page = tif.pages.first
            assert page.compression.name == "ZSTD" and page.predictor == 2
            pixels = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"][
                "Pixels"]
            assert pixels["PhysicalSizeX"] == 0.5
            assert pixels["PhysicalSizeZ"] == 1.0
            names = [ch["Name"] for ch in pixels["Channel"]]
            assert names == ["dapi", "gfp"]
            series = tif.series[0]
            stored, axes = series.asarray(), series.axes
        crop = stored.transpose([axes.index(a) for a in "CZYX"])
        # voxels outside the triangle are 0, inside equal to the source
        assert (crop == 0).any() and (crop > 0).any()
        plan = result.plan
        s = plan.starts[0]                                   # c z y x
        kept = crop > 0
        source = data[:, s[1]:s[1] + crop.shape[1], s[2]:s[2] + crop.shape[2],
                      s[3]:s[3] + crop.shape[3]]
        assert np.array_equal(crop[kept], source[kept])
        assert vc._tiff_record(result.path)["box_name"] == "tri"


def test_never_overwrites_foreign_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "movie.tif"
        write_hyperstack(path, hyperstack((1, 4, 1, 20, 20)))
        reader = fi.open_file(path)
        try:
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        out = Path(tmp) / "out"
        out.mkdir()
        foreign = out / "movie_a.tif"
        tifffile.imwrite(foreign, np.zeros((4, 4), np.uint8))
        try:
            vc.crop_box(image, vc.CropBox("a"), out, overwrite=True)
        except FileExistsError as err:
            assert "not written by Volume Cropper" in str(err)
        else:
            raise AssertionError("a foreign TIFF was overwritten")
        # the crop of a file may not replace the file itself
        try:
            vc.check_output(image, vc.CropBox("x"), path)
        except ValueError:
            pass
        else:
            raise AssertionError("the source could be replaced")


def test_2d_movie_with_polygon_and_every_nth_frame():
    """A 2D movie (one plane): a polygon drawn in XY shapes the crop, and
    every 2nd timepoint is kept."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "crawling.tif"
        data = RNG.integers(1, 4000, size=(10, 30, 36), dtype=np.uint16)
        tifffile.imwrite(path, data, imagej=True, resolution=(1 / 2, 1 / 2),
                         metadata={"axes": "TYX", "unit": "um",
                                   "finterval": 0.1})
        reader = fi.open_file(path)
        try:
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        assert image.levels[0].array.shape == (10, 1, 30, 36)
        square = [(10.0, 10.0), (10.0, 40.0), (40.0, 40.0), (40.0, 10.0)]
        box = vc.CropBox("square", {"t": (0.0, 0.85)}, t_step=2,
                         outlines={("y", "x"): {0.0: square}})
        result = vc.crop_box(image, box, Path(tmp) / "out")
        assert result.problems == [], result.problems
        with tifffile.TiffFile(result.path) as tif:
            meta = tif.imagej_metadata
            assert meta["frames"] == 5 and np.isclose(meta["finterval"], 0.2)
            stored = tif.asarray()
        # pixel i's centre is at 2i um: inside the square (10-40 um) are
        # pixels 5..19 (20 lies on its edge); the crop holds them unchanged
        s = result.plan.starts[0]                         # t z y x
        rows = slice(5 - s[2], 20 - s[2])
        cols = slice(5 - s[3], 20 - s[3])
        assert np.array_equal(stored[:, rows, cols], data[0:10:2, 5:20, 5:20])


def test_crop_of_a_crop_keeps_the_chain():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "movie.tif"
        write_hyperstack(path, hyperstack((2, 6, 2, 30, 30)))
        reader = fi.open_file(path)
        try:
            image = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        first = vc.crop_box(image, vc.CropBox("big", {"x": (2, 12)}),
                            Path(tmp) / "a").path
        reader = fi.open_file(first)
        try:
            crop = fi.load(reader, fi.FileChoice())
        finally:
            reader.close()
        assert np.isclose(crop.levels[0].translation[-1], 2.0)  # its place
        second = vc.crop_box(crop, vc.CropBox("small", {"x": (4, 8)}),
                             Path(tmp) / "b")
        assert second.problems == []
        record = vc._tiff_record(second.path)
        assert record["source_provenance"]["box_name"] == "big"


def test_export_of_an_ome_zarr_crop_with_zstd():
    """0.10.0's "Also save as TIFF", compressed: still checked OK."""
    import test_volume_cropper as tv
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.zarr"
        tv.make_test_image(src)
        image = vc.open_image(src)
        result = vc.crop_box(image, vc.CropBox("a", {"z": (10, 30)}),
                             Path(tmp), tiff_level=1, compression="zstd")
        assert result.tiff_problems == [], result.tiff_problems
        with tifffile.TiffFile(result.tiff) as tif:
            assert tif.is_imagej
            assert tif.pages.first.compression.name == "ZSTD"


def test_command_line_with_a_tiff():
    """The command line crops a TIFF into TIFFs (zstd), with the voxel size
    given for a file without one; a dry run writes nothing."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plain.tif"
        data = RNG.integers(1, 255, size=(8, 30, 40), dtype=np.uint8)
        tifffile.imwrite(path, data, photometric="minisblack", metadata=None)
        boxes = [vc.CropBox("a", {"z": (2, 5), "y": (1.0, 4.0),
                                  "x": (1.0, 6.0)})]
        csv_path = Path(tmp) / "boxes.csv"
        vc.save_boxes_csv(boxes, csv_path, _image_of(path, {"z": 1.0,
                                                            "y": 0.2,
                                                            "x": 0.2}))
        out = Path(tmp) / "out"
        args = [str(path), str(csv_path), str(out), "--voxel-size", "1",
                "0.2", "0.2", "--compress"]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            assert vc.main(args + ["--dry-run"]) == 0
        assert "Dry run" in printed.getvalue() and not out.exists()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = vc.main(args)
        assert code == 0, printed.getvalue()
        crop = out / "plain_a.tif"
        with tifffile.TiffFile(crop) as tif:
            assert tif.pages.first.compression.name == "ZSTD"
            assert np.isclose(tif.imagej_metadata["spacing"], 1.0)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            assert vc.main(args + ["--tiff", "0"]) == 2
        assert "--tiff is for OME-Zarr" in errors.getvalue()


def _image_of(path, voxel):
    reader = fi.open_file(path)
    try:
        return fi.load(reader, fi.FileChoice(voxel=voxel))
    finally:
        reader.close()


def test_memory_estimate_and_limits():
    assert fi.memory_problem(10, 100) == (fi.memory_problem(10, 100)[0],
                                          False)
    text, refuse = fi.memory_problem(60, 100)
    assert "more than half" in text and not refuse
    text, refuse = fi.memory_problem(90, 100)
    assert refuse and "too much" in text
    assert fi.memory_problem(10, None)[1] is False


def test_files_it_cannot_open():
    with tempfile.TemporaryDirectory() as tmp:
        # (0.15.0: LIF, CZI, ND2, OIR are read; a broken one gives the
        # reader library's error)
        for name, error in (("x.ims", NotImplementedError),
                            ("x.txt", ValueError), ("x.lif", Exception),
                            ("x.czi", Exception), ("x.nd2", Exception),
                            ("x.oir", Exception)):
            path = Path(tmp) / name
            path.write_bytes(b"0")
            try:
                fi.open_file(path)
            except error:
                pass
            else:
                raise AssertionError(f"{name} was opened")
        zarr_dir = Path(tmp) / "a.zarr"
        zarr_dir.mkdir()
        assert not fi.is_file_image(zarr_dir)
        empty = Path(tmp) / "empty"
        empty.mkdir()
        assert not fi.is_file_image(empty)


# --- found in the review of 0.17.0 (2026-11-10) ------------------------------

def _ome_set(folder: Path, names: list[str], root_uuid: bool = False):
    """A plain multi-file OME-TIFF: each file holds 2 z planes (value 111,
    222, ...) and the same OME-XML naming all the files (as the
    specification allows). root_uuid: the OME element has the first file's
    UUID (so tifffile knows that entry is the file itself)."""
    import uuid
    ids = [f"urn:uuid:{uuid.uuid4()}" for _ in names]
    blocks = "".join(
        f'<TiffData FirstZ="{2 * k}" IFD="0" PlaneCount="2"><UUID '
        f'FileName="{name}">{u}</UUID></TiffData>'
        for k, (name, u) in enumerate(zip(names, ids)))
    for k, name in enumerate(names):
        own = f' UUID="{ids[k]}"' if root_uuid else ""
        xml = ('<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://'
               f'www.openmicroscopy.org/Schemas/OME/2016-06"{own}><Image '
               'ID="Image:0" Name="set"><Pixels ID="Pixels:0" '
               'DimensionOrder="XYZCT" Type="uint16" SizeX="8" SizeY="8" '
               f'SizeZ="{2 * len(names)}" SizeC="1" SizeT="1"><Channel '
               f'ID="Channel:0:0" SamplesPerPixel="1"/>{blocks}</Pixels>'
               '</Image></OME>')
        path = folder / Path(name).name
        if path.parent.is_dir() and not path.exists():
            tifffile.imwrite(path, np.full((2, 8, 8), 111 * (k + 1),
                                           np.uint16),
                             description=xml, metadata=None,
                             photometric="minisblack")


def test_ome_tiff_naming_files_elsewhere_is_refused():
    """An OME-TIFF (or companion file) whose metadata name files outside
    its folder is refused before any of them is touched: a crafted file
    could make the tool read other files, or make Windows connect to a
    network share (and send it the user's login). Files in sub-folders
    are fine; a missing file of a set is refused (its planes were zeros,
    with only a log message); a renamed single OME-TIFF still opens."""
    for name in (r"\\host\share\x.ome.tif", "//host/share/x.ome.tif",
                 "C:/data/x.ome.tif", "C:x.ome.tif", "../x.ome.tif",
                 r"a\..\..\x.ome.tif"):
        assert fi._outside_folder(name), name
    for name in ("x.ome.tif", "data/x.ome.tif", r"data\x.ome.tif"):
        assert not fi._outside_folder(name), name
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "set").mkdir()
        _ome_set(tmp, ["outside.ome.tif"])              # a file elsewhere
        _ome_set(tmp / "set", ["a.ome.tif", "../outside.ome.tif"])
        _companion_set(tmp / "set")
        companion = tmp / "set" / "embryo.companion.ome"
        companion.write_text(companion.read_text().replace(
            'FileName="embryo_c1.ome.tif"', 'FileName="../outside.ome.tif"'))
        for path in (tmp / "set" / "a.ome.tif", companion):
            try:
                fi.open_file(path)
            except ValueError as err:
                assert "outside its folder" in str(err), err
            else:
                raise AssertionError(f"{path.name} opened files elsewhere")
        (tmp / "pair").mkdir()
        _ome_set(tmp / "pair", ["p.ome.tif", "q.ome.tif"])
        reader = fi.open_file(tmp / "pair" / "p.ome.tif")
        assert reader.scenes[0].shape[0] == 4              # both files
        reader.close()
        (tmp / "pair" / "q.ome.tif").unlink()
        try:
            fi.open_file(tmp / "pair" / "p.ome.tif")
        except FileNotFoundError as err:
            assert "q.ome.tif" in str(err)
        else:
            raise AssertionError("a set with a missing file opened")
        (tmp / "renamed").mkdir()
        _ome_set(tmp / "renamed", ["old.ome.tif"], root_uuid=True)
        (tmp / "renamed" / "old.ome.tif").rename(tmp / "renamed" /
                                                 "new.ome.tif")
        reader = fi.open_file(tmp / "renamed" / "new.ome.tif")
        assert reader.scenes[0].shape[0] == 2
        reader.close()


def test_ome_tiff_entries_follow_tifffile():
    """(The review of the 0.17.1 fixes) An entry with the file's own UUID
    followed by a space is another file for tifffile (it compares the text
    exactly): that file must be checked too; and two images of one file
    with the same name give the same crop name: Overwrite must not
    replace the other image's crop."""
    own = "urn:uuid:A"
    xml = ('<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
           f'UUID="{own}"><Image><Pixels>'
           f'<TiffData><UUID FileName="a.ome.tif">{own}</UUID></TiffData>'
           f'<TiffData><UUID FileName="../x.ome.tif">{own} </UUID>'
           '</TiffData></Pixels></Image></OME>')
    names, others = fi._ome_file_entries(xml, "a.ome.tif")
    assert names == ["a.ome.tif", "../x.ome.tif"] and others == ["../x.ome.tif"]
    renamed = xml.replace(f"{own} </UUID>", f"{own}</UUID>")
    assert fi._ome_file_entries(renamed, "b.ome.tif")[1] == []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tifffile.TiffWriter(tmp / "two.ome.tif", ome=True) as tif:
            for value in (100, 200):
                tif.write(np.full((4, 16, 16), value, np.uint16),
                          metadata={"axes": "ZYX", "Name": "embryo"})
        box = vc.CropBox("tail", {"x": (1.0, 8.0)})
        reader = fi.open_file(tmp / "two.ome.tif")
        assert [s.name for s in reader.scenes] == ["embryo", "embryo"]
        first = fi.load(reader, fi.FileChoice(scene=0))
        vc.crop_box(first, box, tmp / "out")
        second = fi.load(reader, fi.FileChoice(scene=1))
        try:
            vc.crop_box(second, box, tmp / "out", overwrite=True)
        except FileExistsError as err:
            assert "image 1 of two.ome.tif" in str(err), err
        else:
            raise AssertionError("replaced the crop of another image")
        finally:
            reader.close()


def test_frame_interval_from_lif_times():
    """liffile gives a LIF's frame times as numpy datetimes in milliseconds
    and one per plane: read as seconds, an interval was 1000 times too
    long, and the step between planes was taken for the frame interval."""
    stamps = np.array([0, 50, 100, 150, 30000, 30050, 30100, 30150],
                      dtype="datetime64[ms]")             # 2 timepoints
    assert fi._interval_from_times(stamps, 2) == 30.0
    assert fi._interval_from_times(stamps) == 0.05     # (no frame count)
    assert fi._interval_from_times([0.0, 2.5, 5.0]) == 2.5   # seconds
    empty = np.array([], dtype="datetime64[ms]")    # (no stamps: crashed)
    assert fi._interval_from_times(empty, 3) is None


# --- TIFFs that name other files (0.23.0, the pre-publication review) ------
# tifffile opens the files a Micro-Manager NDTiff index or a multi-file
# OME-TIFF names, wherever they are (on Windows a network share there makes
# it connect and send the user's login). A file standing for "elsewhere" is
# in the folder "outside"; an audit hook records the paths opened.

_OPENED = None


def _audit(event, args):
    """Python's audit hook (it cannot be removed): records opened paths
    while _OPENED is a list."""
    if _OPENED is not None and event == "open" and args \
            and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.fsdecode(args[0]))


sys.addaudithook(_audit)


class _watching:
    """`with _watching() as opened:` the paths opened meanwhile."""

    def __enter__(self):
        global _OPENED
        _OPENED = []
        return _OPENED

    def __exit__(self, *exc):
        global _OPENED
        _OPENED = None


def _ndtiff(folder: Path, other: str):
    """A 2 x 2 TIFF with Micro-Manager's NDTiff header, and an
    NDTiff.index beside it that names `other` (by hand: tifffile writes no
    NDTiff)."""
    import json
    import struct
    summary = b"{}"
    head = b"II" + struct.pack("<H", 42) + b"\0\0\0\0"
    micromanager = (struct.pack("<II", 483729, 2)
                    + struct.pack("<II", 2355492, len(summary)) + summary)
    pixels_at = len(head) + len(micromanager)
    pixels = np.arange(4, dtype="<u2").tobytes()
    meta = b'{"a": 1}\0'                  # the MicroManagerMetadata tag's
    meta_at = pixels_at + len(pixels)
    ifd_at = meta_at + len(meta)
    ifd_at += ifd_at % 2
    entries = [(256, 3, 1, 2), (257, 3, 1, 2), (258, 3, 1, 16),
               (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, pixels_at),
               (277, 3, 1, 1), (278, 3, 1, 2), (279, 4, 1, 8),
               (51123, 2, len(meta), meta_at)]
    ifd = struct.pack("<H", len(entries))
    for tag, kind, count, value in entries:
        ifd += (struct.pack("<HHIHH", tag, kind, count, value, 0) if kind == 3
                else struct.pack("<HHII", tag, kind, count, value))
    ifd += struct.pack("<I", 0)
    data = bytearray(head + micromanager + pixels + meta)
    data += b"\0" * (ifd_at - len(data))
    data += ifd
    data[4:8] = struct.pack("<I", ifd_at)
    folder.mkdir(parents=True)
    (folder / "image.tif").write_bytes(bytes(data))
    axes = json.dumps({"time": 0}).encode()
    name = other.encode("utf-8")
    (folder / "NDTiff.index").write_bytes(
        struct.pack("<I", len(axes)) + axes + struct.pack("<I", len(name))
        + name + struct.pack("<IiiiiIii", 0, 2, 2, 1, 0, 0, 0, 0))


def test_ndtiff_index_is_not_followed():
    """A TIFF with an NDTiff index beside it is read as the plain TIFF it
    is: the files the index names (by full path, or by ".." out of the
    folder) are never opened. (The test file is one tifffile takes for an
    NDTiff: checked here first, so the test cannot pass by missing it.)"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        secret = tmp / "outside" / "secret.tif"
        secret.parent.mkdir()
        tifffile.imwrite(secret, np.full((2, 2), 7, np.uint16))
        for k, other in enumerate((str(secret),
                                   "../../outside/secret.tif")):
            folder = tmp / f"case{k}" / "img"
            _ndtiff(folder, other)
            with tifffile.TiffFile(folder / "image.tif") as tif:
                assert tif.is_ndtiff, "the test file is not an NDTiff"
            with _watching() as opened:
                reader = fi.open_file(folder / "image.tif")
                try:
                    image = fi.load(reader, fi.FileChoice())
                finally:
                    reader.close()
            assert not [p for p in opened if "outside" in p], opened
            assert np.asarray(image.levels[0].array).ravel().tolist() \
                == [0, 1, 2, 3]                      # the TIFF's own pixels


def test_tiff_folder_reads_each_file_by_itself():
    """A folder of TIFFs is read file by file: a file in the middle whose
    OME-XML names another file (only the first and the last are checked
    when the folder is opened) does not make tifffile open that file."""
    ome = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
           'UUID="urn:uuid:11111111-1111-1111-1111-111111111111">'
           '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
           'Type="uint16" SizeX="2" SizeY="2" SizeZ="2" SizeC="1" SizeT="1">'
           '<Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
           '<TiffData FirstZ="0" IFD="0" PlaneCount="1"><UUID FileName="{own}">'
           'urn:uuid:11111111-1111-1111-1111-111111111111</UUID></TiffData>'
           '<TiffData FirstZ="1" IFD="0" PlaneCount="1">'
           '<UUID FileName="{other}">'
           'urn:uuid:22222222-2222-2222-2222-222222222222</UUID></TiffData>'
           '</Pixels></Image></OME>')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        secret = tmp / "outside" / "secret.tif"
        secret.parent.mkdir()
        tifffile.imwrite(secret, np.full((2, 2), 7, np.uint16))
        folder = tmp / "stack"
        folder.mkdir()
        tifffile.imwrite(folder / "p01.tif", np.zeros((2, 2), np.uint16))
        tifffile.imwrite(folder / "p02.tif", np.zeros((2, 2), np.uint16),
                         metadata=None,
                         description=ome.format(own="p02.tif",
                                                other=str(secret)))
        tifffile.imwrite(folder / "p03.tif", np.zeros((2, 2), np.uint16))
        with _watching() as opened:
            reader = fi.open_file(folder)
            try:
                fi.load(reader, fi.FileChoice())
            except Exception:     # a file that does not fit may be refused
                pass
            finally:
                reader.close()
        assert not [p for p in opened if "outside" in p], opened


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["files"]))
