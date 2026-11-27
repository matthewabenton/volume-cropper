"""Tests for the microscope-format readers (0.15.0): Leica LIF, Zeiss CZI,
Nikon ND2, Olympus OIR (file_images: LifReader, CziReader, Nd2Reader,
OirReader).

Run with ``python run_all_tests.py microscope`` (or this file alone). The
reader libraries (liffile, czifile, nd2, oirfile) cannot write such files,
and real files are not part of the tool, so these
tests check the readers' own logic with stand-ins for the libraries'
objects:
- axis letters from a library's dimension codes (tiles and views become
  choosable "?" axes, single ones are ignored; colour samples are the
  channels, red, green and blue, since 0.17.2);
- the calibration in µm and seconds from metres, nanometres,
  milliseconds ..., and frame intervals from frame times;
- colours from Leica LUT names and Zeiss ARGB values, compression names;
- a CZI part read as czifile wants it (planes as file coordinates, y and
  x as a region), in the order load() expects, also with a colour sample;
- an ND2 position read from the right place of its lazy array;
- which reader opens which file, and that broken files give an error.
The readers were checked on real LIF and CZI files (values,
and a part equal to the library's read; calibration equal to Fiji's
Bio-Formats) with dev/readers_0_15_check.py.
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

from volume_cropper import file_images as fi


def test_axis_letters():
    axes, notes = fi._letters(("T", "C", "Z", "Y", "X"), (3, 2, 5, 8, 8))
    assert axes == "TCZYX" and notes == []
    axes, notes = fi._letters(("M", "Z", "Y", "X", "S"), (4, 5, 8, 8, 1))
    assert axes == "?ZYX-" and len(notes) == 1 and "'M'" in notes[0]
    axes, _ = fi._letters(("V", "I", "Y", "X"), (1, 1, 8, 8))
    assert axes == "--YX"                     # single views: ignored


def test_colour_samples_are_channels():
    """(0.17.2, the user's choice) A colour (RGB) image's samples are read
    as three channels named and coloured red, green, blue (they were a "?"
    axis, guessed as z). An image with channels AND colours keeps the
    colour axis as a choice."""
    axes, notes = fi._letters(("Z", "Y", "X", "S"), (5, 8, 8, 3))
    assert axes == "ZYXC" and "colour (RGB)" in notes[0]
    # a channel axis of one element (CZI writes one) is ignored
    axes, _ = fi._letters(("T", "C", "Z", "Y", "X", "S"), (2, 1, 5, 8, 8, 3))
    assert axes == "T-ZYXC"
    axes, notes = fi._letters(("C", "Y", "X", "S"), (2, 8, 8, 3))
    assert axes == "CYX?" and "channels and colour" in notes[0]
    assert fi._colour_channels(("Y", "X", "S"), (8, 8, 3), [], [], 3) == (
        ["red", "green", "blue"], ["FF0000", "00FF00", "0000FF"])
    assert fi._colour_channels(("C", "Y", "X"), (2, 8, 8), ["a", "b"],
                               [None, "00FF00"], 2) == (["a", "b"],
                                                        [None, "00FF00"])

    # a LIF colour z-stack, through LifReader and load()
    from types import SimpleNamespace
    data = np.random.default_rng(0).integers(0, 255, (5, 8, 8, 3),
                                             dtype=np.uint8)
    image = SimpleNamespace(
        dims=("Z", "Y", "X", "S"), shape=data.shape, dtype=data.dtype,
        coords={"Z": np.linspace(0, 8e-6, 5), "Y": np.linspace(0, 7e-6, 8),
                "X": np.linspace(0, 7e-6, 8)},
        path="colour stack", name="colour stack", xml_element=None,
        timestamps=None, asarray=lambda out=None: data)
    reader = fi.LifReader.__new__(fi.LifReader)
    reader.path = Path("colour.lif")
    reader._images, reader._arrays, reader._tile_scans = [image], {}, {}
    reader.scenes = [reader._scene(image)]
    reader.format_name, reader.reader_name = "Leica LIF", "liffile (stand-in)"
    scene = reader.scenes[0]
    assert scene.axes == "ZYXC" and fi.default_axes(scene) == "ZYXC"
    assert scene.channel_names == ["red", "green", "blue"]
    assert scene.channel_colors == ["FF0000", "00FF00", "0000FF"]
    loaded = fi.load(reader, fi.FileChoice())
    assert loaded.axis_names == ["c", "z", "y", "x"]
    np.testing.assert_array_equal(np.asarray(loaded.levels[0].array[:]),
                                  data.transpose(3, 0, 1, 2))


def test_calibration_units():
    voxel, unit, interval, origin = fi._calibration(
        "TZYX", {"T": 0.5, "Z": 3e-6, "Y": 5.6e-7, "X": 5.6e-7},
        {"T": "ms", "Z": "meter", "Y": "m", "X": "m"}, {})
    assert np.isclose(voxel["z"], 3.0) and np.isclose(voxel["x"], 0.56)
    assert unit == "micrometer" and np.isclose(interval, 0.0005)
    assert origin == {"z": 0.0, "y": 0.0, "x": 0.0}
    voxel, unit, interval, _ = fi._calibration(
        "YX", {"Y": 250.0, "X": 250.0}, {"Y": "nm", "X": "nanometer"}, {})
    assert np.isclose(voxel["y"], 0.25) and voxel["z"] is None
    voxel, unit, _, _ = fi._calibration("YX", {}, {}, {})
    assert unit is None and voxel == {"z": None, "y": None, "x": None}
    assert np.isclose(fi._interval_from_times([0.0, 0.1, 0.2, 0.31]), 0.1)
    assert fi._interval_from_times([5.0]) is None


def test_colours_and_compression_names():
    assert fi._argb_color("#FF00FF00") == "00FF00"
    assert fi._argb_color("#ff0000") == "FF0000"
    assert fi._argb_color(None) is None
    assert fi._compression_name("zstdhdr (6)") == "zstd"
    assert fi._compression_name("Uncompressed") == "none"
    assert fi._compression_name("JpgXr") == "jpeg xr"

    class Element:
        pass

    class Image:
        xml_element = None

    import xml.etree.ElementTree as ET
    image = Image()
    image.xml_element = ET.fromstring(
        '<Image><Channels><ChannelDescription LUTName="Green"/>'
        '<ChannelDescription LUTName="Magenta"/></Channels></Image>')
    assert fi._lif_colors(image, 3) == ["00FF00", "FF00FF", None]


class _FakeCziImage:
    """Just enough of czifile's CziImage for CziReader._read: a TCYXS
    scene stored at file coordinates T 10.., C 0.., Y 100.., X 200.."""

    def __init__(self, data, dims=("T", "C", "Y", "X", "S"),
                 start=(10, 0, 100, 200, 0), selection=None, roi=None):
        self.data, self.dims, self.start = data, dims, start
        self.shape = data.shape
        self.bbox = (start[dims.index("X")], start[dims.index("Y")],
                     self.shape[dims.index("X")], self.shape[dims.index("Y")])
        self.selection, self.roi = selection or {}, roi

    def __call__(self, roi=None, **selection):
        part, dims = self.data, list(self.dims)
        x, y, w, h = roi
        ys = slice(y - self.start[dims.index("Y")],
                   y - self.start[dims.index("Y")] + h)
        xs = slice(x - self.start[dims.index("X")],
                   x - self.start[dims.index("X")] + w)
        index, out_dims = [], []
        for code, first in zip(self.dims, self.start):
            if code == "Y":
                index.append(ys)
                out_dims.append(code)
            elif code == "X":
                index.append(xs)
                out_dims.append(code)
            elif code in selection:
                value = selection[code]
                if isinstance(value, slice):
                    index.append(slice(value.start - first,
                                       value.stop - first))
                    out_dims.append(code)
                else:
                    index.append(value - first)
            else:
                index.append(slice(None))
                out_dims.append(code)
        result = _FakeCziImage.__new__(_FakeCziImage)
        result.data, result.dims = part[tuple(index)], tuple(out_dims)
        return result

    def asarray(self):
        return self.data


def test_czi_part_read_as_czifile_wants():
    data = np.arange(4 * 2 * 30 * 40 * 3, dtype=np.uint16).reshape(
        4, 2, 30, 40, 3)
    reader = fi.CziReader.__new__(fi.CziReader)
    reader._images = [_FakeCziImage(data)]
    # timepoints 1-2, channel 1, y 5-15, x 7-27, colour sample 2
    key = (slice(1, 3), 1, slice(5, 15), slice(7, 27), 2)
    got = reader.read(0, key)
    np.testing.assert_array_equal(got, data[1:3, 1, 5:15, 7:27, 2])
    # all samples (an RGB axis chosen as channels is read whole)
    key = (0, 0, slice(0, 30), slice(0, 40), slice(None))
    np.testing.assert_array_equal(reader.read(0, key), data[0, 0])


def test_nd2_position_read_from_its_place():
    data = np.arange(3 * 2 * 4 * 6 * 6, dtype=np.uint16).reshape(
        3, 2, 4, 6, 6)                         # T P Z Y X

    class Lazy:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, key):
            return Lazy(self.array[key])

        def compute(self):
            return self.array

    reader = fi.Nd2Reader.__new__(fi.Nd2Reader)
    reader._dims = ["T", "P", "Z", "Y", "X"]
    reader._dask = Lazy(data)
    key = (slice(0, 2), slice(1, 3), slice(0, 6), slice(2, 5))   # no P
    np.testing.assert_array_equal(reader.read(1, key),
                                  data[0:2, 1, 1:3, 0:6, 2:5])


def test_which_reader_opens_what():
    assert fi.MICROSCOPE_READERS[".lif"] is fi.LifReader
    assert fi.MICROSCOPE_READERS[".czi"] is fi.CziReader
    assert fi.MICROSCOPE_READERS[".nd2"] is fi.Nd2Reader
    assert fi.MICROSCOPE_READERS[".oir"] is fi.OirReader
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in (".lif", ".czi", ".nd2", ".oir"):
            path = Path(tmp) / f"broken{suffix}"
            path.write_bytes(b"not an image" * 10)
            try:
                fi.open_file(path)
            except Exception:
                pass
            else:
                raise AssertionError(f"a broken {suffix} file was opened")


def test_oir_channel_names_follow_the_data():
    """(The 0.17.0 review, on real OIR files) oirfile's channel list
    held one entry more than the data have channels (a second "CH1"
    without a wavelength), so every name after the first was one channel
    off. The names now follow the data's own channel order (by id)."""
    from types import SimpleNamespace as Entry

    class FakeOir:
        channels = [Entry(id="a", name="CH1"), Entry(id="x", name="CH1"),
                    Entry(id="b", name="CH2"), Entry(id="c", name="CH3")]

        def _get_ordered_channel_guids(self):
            return ("a", "b", "c")

    assert fi._oir_channel_names(FakeOir()) == ["CH1", "CH2", "CH3"]
    older = FakeOir()                    # an oirfile without that method
    older._get_ordered_channel_guids = None
    assert fi._oir_channel_names(older) == ["CH1", "CH1", "CH2", "CH3"]


# --- Leica files that name other files (0.23.0, the pre-publication review) --
# A crafted Leica file could make liffile open files anywhere, also on a
# network share (Windows would connect and send the user's login). Small
# Leica files are made here: .xlef/.xlcf/.xlif are XML (liffile tells them
# apart by their Data: Experiment, Collection, else an .xlif, which needs a
# Memory element); a .lif is a short binary header and XML. A file standing
# for "elsewhere" lies in the folder "outside" of the temporary folder.

_OPENED = None      # the paths opened while a test watches (see _watching)


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


def _leica_xml(path: Path, data: str = "", refs=(), memory: str = ""):
    """A Leica XML file: data "<Experiment/>" = .xlef, "<Collection/>" =
    .xlcf, else an .xlif (give memory then)."""
    children = "".join(f'<Reference Name="r{k}" File="{ref}"/>'
                       for k, ref in enumerate(refs))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<LMSDataContainerHeader Version="2">'
        f'<Element Name="{path.stem}" UniqueID="{path.stem}">'
        f'<Data>{data}</Data>{memory}<Children>{children}</Children>'
        '</Element></LMSDataContainerHeader>', encoding="utf-8")


def _outside(tmp: Path) -> Path:
    target = tmp / "outside" / "target.xlif"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"not a Leica file")
    return target


def _refused(path, words: str, tmp: Path):
    """open_file(path) must refuse with these words, and open nothing in
    the "outside" folder."""
    with _watching() as opened:
        try:
            fi.open_file(path)
        except ValueError as err:
            assert words in str(err), err
        else:
            raise AssertionError(f"{path.name} was opened")
    elsewhere = [p for p in opened if str(tmp / "outside") in p]
    assert not elsewhere, elsewhere


def test_leica_names_stay_in_the_experiment():
    """A name is read as liffile reads it (percent-decoded, backslashes as
    slashes, from the naming file's folder): ".." is fine while the path
    stays inside the experiment's folder, as in real experiments
    ("..\\..\\..\\pyramid\\..."); a path that leaves it, an absolute path,
    a drive or a network share is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "experiment")
        here = os.path.join(root, "a", "G", "Metadata")
        inside = fi._leica_target(here, "..%5C..%5C..%5Cpyramid%5Cm.xlif",
                                  root)
        assert inside == os.path.join(root, "pyramid", "m.xlif"), inside
        for name in ("..%5C..%5C..%5C..%5Celsewhere.xlif",
                     r"..\..\..\..\elsewhere.xlif", "C:%5Cdata%5Cx.xlif",
                     r"C:\data\x.xlif", r"\\host\share\x.xlif",
                     "%5C%5Chost%5Cshare%5Cx.xlif", "//host/share/x.xlif",
                     "/data/x.xlif"):
            try:
                fi._leica_target(here, name, root)
            except ValueError as err:
                assert "outside the experiment's folder" in str(err)
            else:
                raise AssertionError(f"not refused: {name}")


def test_lif_that_names_other_files_is_refused():
    """A .lif holds its images itself: one whose XML names another file
    (liffile would open it) is refused before anything is opened."""
    import struct
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target = _outside(tmp)
        xml = ('<LMSDataContainerHeader Version="2"><Element Name="x" '
               'UniqueID="u"><Data/><Children>'
               f'<Reference Name="c" File="{target}"/>'
               '</Children></Element></LMSDataContainerHeader>')
        (tmp / "img").mkdir()
        (tmp / "img" / "data.lif").write_bytes(
            struct.pack("<IIBI", 0x70, 2 * len(xml) + 5, 0x2A, len(xml))
            + xml.encode("utf-16-le"))
        _refused(tmp / "img" / "data.lif", "names other files", tmp)


def test_xlef_naming_files_outside_is_refused():
    """An .xlef experiment is refused when any file it names, directly or
    through its .xlcf and .xlif files, lies outside its folder: a reference
    by full path, a ".." path that climbs out, or an image frame."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target = _outside(tmp)
        words = "outside the experiment's folder"
        # 1. the .xlef names a file by its full path
        _leica_xml(tmp / "e1" / "e1.xlef", "<Experiment/>", [str(target)])
        _refused(tmp / "e1" / "e1.xlef", words, tmp)
        # 2. through an .xlcf inside, whose ".." path climbs out
        _leica_xml(tmp / "e2" / "e2.xlef", "<Experiment/>",
                   [".%5Ca%5CMetadata%5Ca.xlcf"])
        _leica_xml(tmp / "e2" / "a" / "Metadata" / "a.xlcf", "<Collection/>",
                   ["..%5C..%5C..%5Coutside%5Ctarget.xlif"])
        _refused(tmp / "e2" / "e2.xlef", words, tmp)
        # 3. an .xlif inside whose image frame lies outside
        _leica_xml(tmp / "e3" / "e3.xlef", "<Experiment/>",
                   ["b%5CMetadata%5Cb.xlif"])
        _leica_xml(tmp / "e3" / "b" / "Metadata" / "b.xlif",
                   memory='<Memory Size="8" MemoryBlockID="m">'
                          f'<Block File="{target.with_suffix(".lof")}" '
                          'Offset="0" Size="8"/></Memory>')
        _refused(tmp / "e3" / "e3.xlef", words, tmp)


def test_xlef_with_its_files_inside_opens():
    """An experiment whose names climb with ".." but stay inside its folder
    is opened: .xlef -> .xlcf -> .xlif, as liffile would (here without
    images, so the tool then says it found none), and liffile's own list
    of each file's children holds the files checked here."""
    import liffile
    with tempfile.TemporaryDirectory() as tmp:
        exp = Path(tmp) / "experiment"
        _leica_xml(exp / "e.xlef", "<Experiment/>",
                   [".%5Ca%5CMetadata%5Ca.xlcf"])
        _leica_xml(exp / "a" / "Metadata" / "a.xlcf", "<Collection/>",
                   ["..%5C..%5Cb%5CMetadata%5Cb.xlif"])
        _leica_xml(exp / "b" / "Metadata" / "b.xlif",
                   memory='<Memory Size="0" MemoryBlockID="m"></Memory>')
        try:
            fi.open_file(exp / "e.xlef")
        except ValueError as err:          # all names accepted, no images
            assert "no images Volume Cropper can read" in str(err), err
        else:
            raise AssertionError("an experiment without images opened")
        with liffile.LifFile(exp / "e.xlef") as lif:
            fi._open_experiment(lif, str(exp))
            (collection,) = lif.children
            (image_file,) = collection.children
            assert collection.filename == "a.xlcf"
            assert image_file.filename == "b.xlif"
            assert image_file.type.name == "XLIF"


def test_leica_file_of_another_kind_is_refused():
    """Opened as a .lif or .lof, a Leica XML file (an .xlcf here) is
    refused: only those two hold their images themselves."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _leica_xml(tmp / "x" / "renamed.lif", "<Collection/>")
        _refused(tmp / "x" / "renamed.lif", "not a .lif or .lof", tmp)


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["microscope"]))
