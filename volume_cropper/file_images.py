"""Opening image files other than OME-Zarr (0.12.0): TIFF so far.

Volume Cropper reads such a file -- or the part of it chosen in "What to
load" -- into memory, like opening an image in Fiji without "Use virtual
stack". It then makes smaller copies for the views (a pyramid, as OME-Zarr
images have: each level halves x and y, and z once its voxels are smaller
than x and y), so that zooming out and the 3D view stay fast. From then on
the image is handled like an OME-Zarr image: the views, the crop regions
and the crop code are the same. Its crops are saved as TIFF
(volume_cropper.crop_box), never as OME-Zarr.

In memory the image is kept as zarr arrays in a zarr MemoryStore (nothing is
written to disk), so that the code written for OME-Zarr files works on it
unchanged. Nothing is ever written into the file or its folder.

Supported so far: TIFF files (plain, Fiji/ImageJ hyperstacks, OME-TIFF,
compressed or not) and folders of TIFF files (one file per plane or per
stack, like Fiji's File > Import > Image Sequence). LIF, ND2, CZI and OIR
are planned next.
"""

from __future__ import annotations

import math
import os
import platform
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import zarr
from zarr.storage import MemoryStore

from volume_cropper import tile_stitching
from volume_cropper import core as vc

TIFF_SUFFIXES = (".tif", ".tiff", ".btf", ".tf8", ".tf2")
# Formats recognised but not readable yet: the message says so.
# (Since 0.15.0 LIF, CZI, ND2 and OIR are read: see MICROSCOPE_READERS.)
LATER_FORMATS = {".ims": "Imaris IMS", ".vsi": "Olympus VSI",
                 ".lsm": "Zeiss LSM (TIFF-based: try it as TIFF)"}

# The pyramid: levels are added until x and y are at most this many voxels.
LEVEL_MIN_SIZE = 256
MAX_LEVELS = 10
# Chunks of the arrays in memory (t and c: 1). Measured on a 1024 x 1024 x
# 200 stack: a slice in any view takes 2-5 ms with these (plain numpy: 1-3
# ms); one plane per chunk made XZ and YZ slices 5-10 times slower.
CHUNK_Z, CHUNK_YX = 16, 512

# Memory: "What to load" warns above this share of the free memory and
# refuses above the second (the computer would start swapping to disk).
MEMORY_WARN, MEMORY_REFUSE = 0.5, 0.8

# Channel colours when the file has none (as for OME-Zarr images).
DEFAULT_COLORS = ["FF00FF", "00FF00", "00FFFF", "FFFF00", "FFFFFF"]

_SPACE_UNITS = {
    "micrometer": "micrometer", "micron": "micrometer",
    "microns": "micrometer", "um": "micrometer", "µm": "micrometer",
    "μm": "micrometer", "nanometer": "nanometer", "nm": "nanometer",
    "millimeter": "millimeter", "mm": "millimeter",
    "centimeter": "centimeter", "cm": "centimeter", "meter": "meter",
    "m": "meter",
}
_TIME_UNITS = {
    "second": "second", "seconds": "second", "sec": "second",
    "s": "second", "millisecond": "millisecond", "ms": "millisecond",
    "msec": "millisecond", "microsecond": "microsecond",
    "us": "microsecond", "µs": "microsecond", "μs": "microsecond",
    "minute": "minute", "minutes": "minute", "min": "minute",
    "hour": "hour", "hours": "hour", "hr": "hour", "h": "hour",
}


class ReadCancelled(Exception):
    """Raised when reading a file is cancelled (e.g. from the window)."""


def _unescape(text) -> str:
    """ImageJ writes non-ASCII letters as \\uXXXX (e.g. \\u00B5m for µm)."""
    return re.sub(r"\\u([0-9a-fA-F]{4})",
                  lambda m: chr(int(m.group(1), 16)), str(text)).strip()


def space_unit(text) -> str | None:
    """A length unit as OME names it ("micrometer"), or None if `text` is
    no length unit (e.g. "pixel", "inch", empty)."""
    if text is None:
        return None
    return _SPACE_UNITS.get(_unescape(text).lower())


def time_unit(text) -> str | None:
    """A time unit as OME names it ("second"), or None."""
    if text is None:
        return None
    return _TIME_UNITS.get(_unescape(text).lower())


# ---------------------------------------------------------------------------
# What a file holds
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """One image in a file (Fiji: a "series"), as the file describes it.

    axes: one letter per axis of `shape`, in the file's order: T, C, Z, Y,
    X, "?" for an axis whose meaning the file does not state (the user
    chooses it), "-" for an axis of size 1 that is ignored.
    """

    name: str
    axes: str
    shape: tuple
    dtype: np.dtype
    voxel: dict                         # {"z","y","x"}: size in `unit` or None
    unit: str | None                    # OME name; None: not in the file
    interval: float | None = None       # between timepoints, in time_unit
    time_unit: str | None = None
    channel_names: list = field(default_factory=list)
    channel_colors: list = field(default_factory=list)   # "RRGGBB" or None
    channel_windows: list = field(default_factory=list)  # (low, high) or None
    notes: list = field(default_factory=list)
    # Where voxel 0's centre is, in voxels ({"z","y","x"}): from Fiji's
    # origin (Image > Properties), e.g. in a crop saved by this tool with
    # "Keep position"; 0 when the file has none.
    origin: dict = field(default_factory=lambda: {"z": 0.0, "y": 0.0,
                                                  "x": 0.0})
    # A first guess for the "?" axes (same length as axes; e.g. napari
    # layers' axes guessed by position, 0.13.0); "" = none.
    axes_guess: str = ""

    @property
    def nbytes(self) -> int:
        return vc.nbytes(self.shape, self.dtype)


@dataclass
class FileChoice:
    """What to read of a file (see load()).

    axes:     the meaning of the scene's axes, one letter per axis (replaces
              its "?" letters); None: as in the file, "?" as default_axes().
    channels: the channels read (None: all).
    ranges:   {"t"|"z"|"y"|"x": (first, end)}: 0-based, end excluded;
              axes left out are read whole.
    voxel, unit, interval, time_unit: the calibration; None (or a missing
              key): as in the file.
    """

    scene: int = 0
    axes: str | None = None
    channels: list[int] | None = None
    ranges: dict = field(default_factory=dict)
    voxel: dict | None = None
    unit: str | None = None
    interval: float | None = None
    time_unit: str | None = None
    show_3d: bool = False


def default_axes(scene: Scene) -> str:
    """The scene's axes with every "?" replaced: by the scene's own guess
    where it has one, else by Z if it has no Z, then T, then C (a stack of
    unknown planes is usually a z-stack, as Fiji assumes for an image
    sequence)."""
    letters = list(scene.axes)
    guess = scene.axes_guess if len(scene.axes_guess) == len(letters) else ""
    for i, letter in enumerate(letters):
        if letter == "?" and guess and guess[i] != "?" \
                and guess[i] not in letters:
            letters[i] = guess[i]
    for i, letter in enumerate(letters):
        if letter == "?":
            for meaning in "ZTC":
                if meaning not in letters:
                    letters[i] = meaning
                    break
    return "".join(letters)


def axes_of(scene: Scene, choice: FileChoice) -> str:
    """The meaning of each axis for this choice; raises ValueError if it is
    impossible (e.g. two axes both called z)."""
    axes = choice.axes or default_axes(scene)
    if len(axes) != len(scene.shape):
        raise ValueError(f"{scene.name}: {len(scene.shape)} axes, but "
                         f"{len(axes)} meanings ({axes}).")
    for i, (given, stored) in enumerate(zip(axes, scene.axes)):
        if stored in "TCZYX-" and given != stored:
            raise ValueError(f"{scene.name}: axis {i + 1} is {stored} in the "
                             "file; it cannot be changed.")
    used = [a for a in axes if a != "-"]
    if "?" in used or not set(used) <= set("TCZYX"):
        raise ValueError(f"{scene.name}: each axis must be t, c, z, y or x.")
    if len(set(used)) != len(used):
        raise ValueError(f"{scene.name}: two axes cannot both be "
                         + ", ".join(sorted({a.lower() for a in used
                                             if used.count(a) > 1})) + ".")
    if "Y" not in used or "X" not in used:
        raise ValueError(f"{scene.name}: the image needs a y and an x axis.")
    return axes


def sizes_of(scene: Scene, axes: str) -> dict:
    """{"T": n, "C": n, ...} for the axes in use."""
    return {a: n for a, n in zip(axes, scene.shape) if a != "-"}


# ---------------------------------------------------------------------------
# Readers: one per kind of file
# ---------------------------------------------------------------------------

class TiffReader:
    """A TIFF file: plain, Fiji/ImageJ hyperstack, or OME-TIFF (all its
    images; a pyramidal OME-TIFF's full resolution). Read with tifffile,
    plane by plane (compressed files too).

    An OME-TIFF set described by a companion file (0.16.0; e.g.
    embryo.companion.ome, as Viventis and Bio-Formats write them): `path`
    is the companion, `tiff` the first TIFF it lists and `omexml` its OME
    metadata; tifffile then reads the images across all the TIFF files
    the companion lists (see open_companion).
    """

    def __init__(self, path: str | Path, tiff: str | Path | None = None,
                 omexml: str | None = None):
        import tifffile
        self.path = Path(path)
        # (a companion's metadata come first, also when the first TIFF has
        # tifffile's or ImageJ's own description.) Never as a Micro-Manager
        # NDTiff (0.23.0, the pre-publication review): tifffile would open
        # every file its NDTiff.index names, wherever that is.
        self._tif = (tifffile.TiffFile(self.path, is_ndtiff=False)
                     if omexml is None else
                     tifffile.TiffFile(Path(tiff), omexml=omexml,
                                       is_shaped=False, is_imagej=False,
                                       is_ndtiff=False))
        self._arrays: dict = {}
        tif = self._tif
        try:
            page = tif.pages.first
            self.compression = page.compression.name.lower()
            if omexml is not None:
                self.format_name = "OME-TIFF (companion file)"
                self.output_kind = "ome"
                # tifffile falls back to the first file alone (with only a
                # log message) when it cannot build the set from the
                # companion: refuse that rather than open part of the set
                if not tif.series or any(
                        getattr(s, "kind", "ome").lower() != "ome"
                        for s in tif.series):
                    raise ValueError(
                        f"{self.path.name}: its TIFF files could not be read "
                        "as one OME-TIFF set (only the first one could be "
                        "opened). Check that all the files it lists are "
                        "complete and belong to it.")
                self._series, self.scenes = self._ome_scenes()
            elif tif.is_ome:
                self.format_name, self.output_kind = "OME-TIFF", "ome"
                # An OME-TIFF can spread its planes over several files
                # (named in its OME-XML). tifffile opens them as named and
                # fills a missing one with zeros (with only a log message):
                # check them first.
                names, others = _ome_file_entries(tif.ome_metadata or "",
                                                  self.path.name)
                _refuse_files_elsewhere(self.path, names)   # (every name)
                missing = [n for n in others
                           if not (self.path.parent / n).is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"{self.path.name} is part of a set of OME-TIFF "
                        "files, and some are not in its folder: "
                        + ", ".join(missing[:5])
                        + (" …" if len(missing) > 5 else "") + ". Their "
                        "planes would be zeros: keep the files together.")
                self._series, self.scenes = self._ome_scenes()
            elif tif.is_imagej:
                self.format_name = "TIFF (Fiji/ImageJ hyperstack)"
                self.output_kind = "imagej"
                self._series, self.scenes = [0], [self._imagej_scene()]
            else:
                self.format_name, self.output_kind = "TIFF", "imagej"
                self._series = list(range(len(tif.series)))
                self.scenes = [self._plain_scene(i) for i in self._series]
        except BaseException:
            tif.close()
            raise
        self.reader_name = f"tifffile {tifffile.__version__}"
        # The crop record, if this TIFF is a crop made by this tool.
        self.record = (vc._tiff_record(self.path) or None) if omexml is None \
            else None

    # --- metadata ---------------------------------------------------------------

    def _imagej_scene(self) -> Scene:
        tif = self._tif
        series = tif.series[0]
        meta = tif.imagej_metadata or {}
        axes, notes = _our_axes(series.axes, series.shape)
        unit = space_unit(meta.get("unit"))
        voxel = {"z": None, "y": None, "x": None}
        if unit is not None:
            voxel["x"], voxel["y"] = _resolution(series.keyframe)
            if "Z" in axes:
                voxel["z"] = _positive(meta.get("spacing"))
        interval = (_positive(meta.get("finterval")) if "T" in axes
                    else None)
        n_c = _size(axes, series.shape, "C")
        colors = [None] * n_c
        luts = meta.get("LUTs")
        if luts is not None and len(luts) == n_c:
            colors = [_lut_color(lut) for lut in luts]
        windows = [None] * n_c
        ranges = meta.get("Ranges")
        if ranges is not None and len(ranges) == 2 * n_c:
            windows = [(float(ranges[2 * k]), float(ranges[2 * k + 1]))
                       for k in range(n_c)]
        elif n_c == 1 and "min" in meta and "max" in meta:
            windows = [(float(meta["min"]), float(meta["max"]))]
        if unit is None:
            notes.append("The file has no voxel size (Fiji: Image > "
                         "Properties showed pixels).")
        # Fiji's origin: pixel coordinate p is at (p - origin) x pixel size,
        # and pixel i covers p = i ... i + 1 (centre i + 0.5); slice i is at
        # (i - origin) x spacing. So voxel 0's centre is 0.5 - xorigin
        # pixels in x and y, -zorigin slices in z (the inverse of what
        # volume_cropper._imagej_metadata writes).
        origin = {"z": 0.0, "y": 0.0, "x": 0.0}
        for k, half in (("x", 0.5), ("y", 0.5), ("z", 0.0)):
            value = meta.get(f"{k}origin")
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                origin[k] = half - value
        return Scene(
            name=self.path.name, axes=axes, shape=tuple(series.shape),
            dtype=np.dtype(series.dtype), voxel=voxel, unit=unit,
            interval=interval,
            time_unit=(time_unit(meta.get("tunit", "sec")) if interval
                       else None),
            channel_colors=colors, channel_windows=windows, notes=notes,
            origin=origin)

    def _ome_scenes(self) -> tuple[list, list]:
        import tifffile
        tif = self._tif
        try:
            images = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]
        except Exception:
            images = []
        if isinstance(images, dict):
            images = [images]
        indices, scenes = [], []
        for i, series in enumerate(tif.series):
            axes, notes = _our_axes(series.axes, series.shape)
            image = images[i] if i < len(images) else {}
            pixels = image.get("Pixels", {}) if isinstance(image, dict) else {}
            if (pixels.get("SizeX") != _size(axes, series.shape, "X")
                    or pixels.get("SizeY") != _size(axes, series.shape, "Y")):
                pixels = {}      # cannot match it safely: no calibration
                notes.append("Its OME metadata could not be matched to the "
                             "image: no voxel size or channel names.")
            voxel = {k: _positive(pixels.get(f"PhysicalSize{k.upper()}"))
                     for k in "zyx"}
            units = {space_unit(pixels.get(f"PhysicalSize{k.upper()}Unit",
                                           "µm"))
                     for k in "zyx" if voxel[k] is not None}
            unit = None
            if len(units) == 1 and None not in units:
                unit = units.pop()
            elif units:
                voxel = {k: None for k in "zyx"}
                notes.append("Its voxel sizes have different units: please "
                             "enter them.")
            interval = (_positive(pixels.get("TimeIncrement"))
                        if "T" in axes else None)
            channels = pixels.get("Channel", [])
            if isinstance(channels, dict):
                channels = [channels]
            n_c = _size(axes, series.shape, "C")
            names, colors = [], [None] * n_c
            if len(channels) == n_c:
                names = [str(ch.get("Name") or "") for ch in channels]
                colors = [_ome_color(ch.get("Color")) for ch in channels]
            name = series.name or image.get("Name") or f"image {i + 1}"
            scenes.append(Scene(
                name=str(name), axes=axes, shape=tuple(series.shape),
                dtype=np.dtype(series.dtype), voxel=voxel, unit=unit,
                interval=interval,
                time_unit=(time_unit(pixels.get("TimeIncrementUnit", "s"))
                           if interval else None),
                channel_names=names, channel_colors=colors,
                channel_windows=[None] * n_c, notes=notes))
            indices.append(i)
        return indices, scenes

    def _plain_scene(self, i: int) -> Scene:
        series = self._tif.series[i]
        axes, notes = _our_axes(series.axes, series.shape)
        notes.append("The file has no voxel size.")
        n_c = _size(axes, series.shape, "C")
        name = (self.path.name if len(self._tif.series) == 1
                else f"series {i + 1}")
        return Scene(name=name, axes=axes, shape=tuple(series.shape),
                     dtype=np.dtype(series.dtype),
                     voxel={"z": None, "y": None, "x": None}, unit=None,
                     channel_colors=[None] * n_c,
                     channel_windows=[None] * n_c, notes=notes)

    # --- data ----------------------------------------------------------------

    def read(self, scene: int, key: tuple) -> np.ndarray:
        """The voxels at `key` (one int or slice per axis of the scene)."""
        if scene not in self._arrays:
            series = self._tif.series[self._series[scene]]
            self._arrays[scene] = zarr.open_array(series.aszarr(level=0),
                                                  mode="r")
        return np.asarray(self._arrays[scene][key])

    def close(self):
        self._arrays.clear()
        self._tif.close()


def _links_outside(folder: Path, links, resolve=os.path.realpath
                   ) -> list[str]:
    """(0.23.2, the pre-publication review) The names of the links among a
    folder's files that lead outside the folder: on Linux and Mac anyone
    can make such links, and reading them would read files elsewhere.
    (`resolve`: for the tests.)"""
    root = resolve(folder)
    return [Path(p).name for p in links if not _inside(resolve(p), root)]


class TiffSequenceReader:
    """A folder of TIFF files, one image per file (a plane, or a stack),
    in the order of their names (numbers in names sorted as numbers:
    2 before 10), like Fiji's File > Import > Image Sequence. Every file
    must have the same size; the calibration comes from the first."""

    def __init__(self, folder: str | Path):
        import tifffile
        self.path = Path(folder)
        with os.scandir(self.path) as entries:   # (knows links at no cost)
            found = [(Path(e.path), e.is_symlink()) for e in entries
                     if Path(e.name).suffix.lower() in TIFF_SUFFIXES
                     and e.is_file()]
        files = sorted((p for p, _ in found), key=lambda p: _natural_key(p.name))
        if not files:
            raise ValueError(f"{self.path} holds no TIFF files.")
        outside = _links_outside(self.path, [p for p, link in found if link])
        if outside:
            raise ValueError(
                f"{self.path.name}: {', '.join(sorted(outside)[:3])}"
                f"{' …' if len(outside) > 3 else ''} lead to files outside "
                "the folder (links). Volume Cropper reads only the folder's "
                "own files: a link could point anywhere.")
        self.files = files
        first = TiffReader(files[0])
        try:
            page_scene = first.scenes[0]
            self.compression = first.compression
        finally:
            first.close()
        if len(files) > 1:
            last = TiffReader(files[-1])
            try:
                if (last.scenes[0].shape != page_scene.shape
                        or last.scenes[0].dtype != page_scene.dtype):
                    raise ValueError(
                        f"The files in {self.path.name} differ in size: "
                        f"{files[0].name} is {page_scene.shape}, "
                        f"{files[-1].name} {last.scenes[0].shape}.")
            finally:
                last.close()
        self._sequence = tifffile.TiffSequence([str(f) for f in files],
                                               sort=False)
        self._array = None
        self.format_name = f"TIFF sequence ({len(files)} files)"
        self.output_kind = "imagej"
        self.reader_name = f"tifffile {tifffile.__version__}"
        scene = page_scene
        notes = [n for n in scene.notes
                 if not n.startswith("The file has no voxel size")]
        notes.append(f"{len(files)} files: {files[0].name} … "
                     f"{files[-1].name}.")
        voxel = dict(scene.voxel)
        self.scenes = [Scene(
            name=self.path.name, axes="?" + scene.axes,
            shape=(len(files),) + tuple(scene.shape), dtype=scene.dtype,
            voxel=voxel, unit=scene.unit, interval=scene.interval,
            time_unit=scene.time_unit, channel_names=scene.channel_names,
            channel_colors=scene.channel_colors,
            channel_windows=scene.channel_windows, notes=notes,
            origin=dict(scene.origin))]

    def read(self, scene: int, key: tuple) -> np.ndarray:
        if self._array is None:
            # Each file by itself (0.23.0, the pre-publication review): only
            # the first and last file are checked when the folder is opened,
            # and tifffile would otherwise also open the files that a
            # multi-file OME-TIFF or a Micro-Manager NDTiff index names.
            self._array = zarr.open_array(self._sequence.aszarr(
                imreadargs={"_multifile": False, "is_ndtiff": False}),
                mode="r")
        return np.asarray(self._array[key])

    def close(self):
        self._array = None
        self._sequence.close()


# ---------------------------------------------------------------------------
# Microscope formats (0.15.0): LIF, CZI, ND2, OIR
# ---------------------------------------------------------------------------

# Length units the reader libraries give, in micrometres.
_TO_MICROMETRES = {"m": 1e6, "meter": 1e6, "metre": 1e6, "mm": 1e3,
                   "millimeter": 1e3, "um": 1.0, "µm": 1.0, "micron": 1.0,
                   "micrometer": 1.0, "nm": 1e-3, "nanometer": 1e-3}
_TO_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
               "ms": 1e-3, "millisecond": 1e-3, "us": 1e-6,
               "microsecond": 1e-6, "min": 60.0, "minute": 60.0,
               "h": 3600.0, "hour": 3600.0}
# Leica LUT names -> colour
_LUT_COLORS = {"red": "FF0000", "green": "00FF00", "blue": "0000FF",
               "cyan": "00FFFF", "magenta": "FF00FF", "yellow": "FFFF00",
               "gray": "FFFFFF", "grey": "FFFFFF", "white": "FFFFFF"}


def _missing_reader(package: str, kind: str):
    raise NotImplementedError(
        f"{kind} files need the Python package {package!r}, which this "
        "environment does not have. Update the environment with the tool's "
        "recipe (conda env update -f <tool folder>/env/environment.yml), or "
        "open the file in Fiji and save it as TIFF.")


# Colour (RGB) samples read as channels (0.17.2, the user's choice): their
# names and colours. The reader libraries give the samples in R, G, B order
# (liffile and czifile turn Leica's and Zeiss's B, G, R around).
RGB_NAMES = ("red", "green", "blue", "alpha")
RGB_COLORS = ("FF0000", "00FF00", "0000FF", "FFFFFF")


def _is_colour_image(dims, shape) -> bool:
    """Whether the colour (RGB) samples, the library's dimension "S", are
    read as the channels: the image has samples but no channels."""
    sizes = {str(code).upper(): n for code, n in zip(dims, shape)}
    return sizes.get("S", 1) > 1 and sizes.get("C", 1) == 1


def _letters(dims, shape) -> tuple[str, list]:
    """Our axis letters for a library's dimension codes: T, C, Z, Y, X as
    they are; any other code (tiles, views, illuminations ...) "-" if it
    has one element, "?" (to be chosen) otherwise. Colour (RGB) samples
    ("S") are the channels when the image has no channels of its own
    (0.17.2; they were "?" and guessed as z before)."""
    colour = _is_colour_image(dims, shape)
    axes, notes = [], []
    for code, n in zip(dims, shape):
        code = str(code).upper()
        if colour and code == "S":
            axes.append("C")
            notes.append("A colour (RGB) image: its colours are read as "
                         "channels (red, green, blue).")
        elif colour and code == "C":
            axes.append("-")                 # (one element: the samples)
        elif code in "TCZYX" and code not in axes and len(code) == 1:
            axes.append(code)
        elif n == 1:
            axes.append("-")
        elif code == "M":
            axes.append("?")
            notes.append(f"The file's axis 'M' ({n} elements) holds the "
                         "tiles of a tile scan (a mosaic), one after the "
                         "other: choose what it is below.")
        elif code == "S":
            axes.append("?")
            notes.append(f"This image has channels and colour (RGB) samples "
                         f"({n}); Volume Cropper shows channels only: choose "
                         "what the colour axis is below, or open the file in "
                         "Fiji.")
        else:
            axes.append("?")
            notes.append(f"The file's axis {code!r} ({n} elements) has no "
                         "meaning Volume Cropper knows (e.g. tiles, views or "
                         "colour samples): choose one below, or open the "
                         "file in Fiji.")
    return "".join(axes), notes


def _colour_channels(dims, shape, names: list, colors: list,
                     n_c: int) -> tuple[list, list]:
    """The channel names and colours: red, green, blue for a colour image
    (see _is_colour_image), else the reader's own."""
    if not _is_colour_image(dims, shape):
        return names, colors
    names = [RGB_NAMES[i] if i < len(RGB_NAMES) else f"sample {i + 1}"
             for i in range(n_c)]
    colors = [RGB_COLORS[i] if i < len(RGB_COLORS) else None
              for i in range(n_c)]
    return names, colors


def _calibration(axes: str, scales: dict, units: dict, offsets: dict,
                 default_space: str = "m") -> tuple:
    """(voxel {"z","y","x"} in µm, unit, interval in s or None, origin in
    voxels) from a library's per-axis steps, units and offsets."""
    voxel = {"z": None, "y": None, "x": None}
    origin = {"z": 0.0, "y": 0.0, "x": 0.0}
    for letter in "ZYX":
        if letter not in axes:
            continue
        step = _positive(scales.get(letter))
        factor = _TO_MICROMETRES.get(str(units.get(letter, default_space))
                                     .strip().lower())
        if step is None or factor is None:
            continue
        voxel[letter.lower()] = step * factor
    interval = None
    if "T" in axes:
        step = _positive(scales.get("T"))
        factor = _TO_SECONDS.get(str(units.get("T", "s")).strip().lower())
        if step is not None and factor is not None:
            interval = step * factor
    unit = "micrometer" if any(v is not None for v in voxel.values()) \
        else None
    return voxel, unit, interval, origin


def _oir_channel_names(oir) -> list[str]:
    """The names of an OIR file's channels, in the order of its data.
    oirfile's channel list can hold more entries than the data have
    channels (on the user's files: a second "CH1" without a wavelength),
    so the names are matched to the data's channels by their ids (the
    0.17.0 review: all names after the first were one channel off)."""
    try:
        entries = list(oir.channels)
        names = {str(c.id): str(c.name) for c in entries}
        try:
            order = oir._get_ordered_channel_guids()   # (oirfile's own)
        except Exception:
            order = None
        if order:
            return [names.get(str(guid), "") for guid in order]
        return [str(c.name) for c in entries]
    except Exception:
        return []


def _interval_from_times(times, n_frames: int | None = None) -> float | None:
    """The frame interval in seconds from frame times (the median step),
    or None. Times as numbers are seconds; numpy datetimes are converted
    (liffile gives them in milliseconds: read as seconds, an interval was
    1000 times too long). With one time per plane (e.g. LIF: every z and
    channel), n_frames picks the first plane of each timepoint."""
    times = np.asarray(times) if times is not None else None
    if times is None or times.size < 2:
        return None
    if np.issubdtype(times.dtype, np.datetime64):
        times = (times - times.ravel()[0]) / np.timedelta64(1, "s")
    if times is not None and n_frames and times.size > n_frames \
            and times.size % n_frames == 0:
        times = times.ravel()[::times.size // n_frames]
    try:
        values = np.asarray(times, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if values.size < 2:
        return None
    steps = np.diff(values)
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else None


class _MicroscopeReader:
    """What the LIF, CZI, ND2 and OIR readers share. Each has `scenes`
    (one per image in the file) and reads a part of one through
    `_read(scene, key)`. Their crops are OME-TIFFs (the user's choice,
    2026-11-06: no Bio-Formats, a TIFF goes out)."""

    output_kind = "ome"
    compression = "none"           # (what the TIFF crops start with)
    record = None                  # (only TIFF crops made by this tool)

    def read(self, scene: int, key: tuple) -> np.ndarray:
        return np.asarray(self._read(scene, key))

    def close(self):
        pass


# Leica files that name other files (0.23.0, the pre-publication review).
# liffile opens the files a Leica file's XML names (other Leica files, and
# image frames: .lof, .tif ...) at any path, also on a network share, where
# Windows would connect and send the user's login. So:
# - a .lif or .lof holds its images itself: one that names other files is
#   refused (none of the maintainer's real LIF and LOF files does);
# - an .xlef experiment names many files, with ".." (e.g.
#   "..\..\pyramid\metadata\..."): each name must stay inside the
#   experiment's folder. Its files are opened here, each after its name was
#   checked (_open_experiment), and an image's frame files just before
#   liffile reads them (LifReader._check_frames).
# This relies on the inner workings of liffile (pinned in environment.yml,
# 2026.10.14): LifFile.children and an image's _lof_reference and
# _tif_reference are functools cached properties, filled here with the
# checked files, so liffile reads those and opens nothing else.
_LEICA_SELF_CONTAINED = ("LIF", "LOF", "LIFEXT")
_LEICA_MAX_FILES = 200_000      # files in one experiment (the real one: 3,700)


def _leica_name(name: str) -> str:
    """A file name from a Leica XML file as liffile reads it: percent-
    decoded, with backslashes read as slashes."""
    from urllib.parse import unquote
    return os.path.normpath(unquote(name)).replace("\\", "/")


def _inside(path: str, root: str) -> bool:
    """Whether path lies in the folder root (or below it)."""
    try:
        relative = os.path.relpath(path, root)
    except ValueError:                       # another drive (Windows)
        return False
    return not (relative == ".." or relative.startswith(".." + os.sep)
                or os.path.isabs(relative))


def _leica_target(dirname: str, name: str, root: str) -> str:
    """The path a Leica XML file's name points to (relative to the naming
    file's folder, as liffile joins it); ValueError unless it lies inside
    root, the experiment's folder."""
    text = _leica_name(name)
    path = os.path.normpath(os.path.join(dirname, text))
    if not text or text.startswith("/") or ":" in text \
            or not _inside(path, root):
        raise ValueError(
            f"A file of the Leica experiment names {name!r}, outside the "
            f"experiment's folder ({Path(root).name}). Volume Cropper only "
            "opens files inside it: a file elsewhere could even be on "
            "another computer.")
    return path


def _leica_names(lif) -> list[str]:
    """The files a Leica file's own XML names: references to other Leica
    files, and image frames."""
    names = [e.get("File") for e in lif.xml_element.iter("Reference")
             if e.get("File")]
    for memory in lif.xml_element.iter("Memory"):
        names += [b.get("File") for b in memory if b.get("File")]
    return names


def _refuse_leica_names(lif, path: Path) -> None:
    """Refuse a .lif or .lof that is not one, or that names other files
    (it holds its images itself)."""
    kind = getattr(lif.type, "name", str(lif.type))
    if kind not in _LEICA_SELF_CONTAINED:
        raise ValueError(
            f"{path.name} is a Leica {kind} file, not a .lif or .lof. Volume "
            "Cropper opens Leica .lif, .lof and .xlef files.")
    names = _leica_names(lif)
    if names:
        raise ValueError(
            f"{path.name} names other files ({', '.join(names[:3])}"
            f"{' …' if len(names) > 3 else ''}). A .lif or .lof holds its "
            "images itself, so Volume Cropper does not open files it names: "
            "they could be anywhere, even on another computer.")


def _open_experiment(lif, root: str) -> None:
    """Open the files an .xlef experiment names (.xlcf, .xlif, ...) as
    liffile's LifFile.children does, each only after its name was checked
    (_leica_target), and check the names of every .xlif's image frames.
    liffile then uses these files (its children cache)."""
    import liffile
    from liffile.liffile import case_sensitive_path
    count = 0
    stack = [(lif, frozenset())]
    while stack:
        node, ancestors = stack.pop()
        here = os.path.normcase(os.path.abspath(
            os.path.join(node.dirname, node.filename)))
        children = []
        try:
            for ref in node.xml_element.findall("./Element/Children/Reference"):
                name = ref.attrib.get("File")
                if name is None:
                    continue
                target = _leica_target(node.dirname, name, root)
                if not os.path.exists(target):
                    target = case_sensitive_path(target)      # as liffile
                if os.path.normcase(os.path.abspath(target)) \
                        in ancestors | {here}:
                    raise ValueError(
                        f"The Leica experiment's files name each other in a "
                        f"loop ({Path(target).name}).")
                count += 1
                if count > _LEICA_MAX_FILES:
                    raise ValueError("The Leica experiment names more than "
                                     f"{_LEICA_MAX_FILES} files.")
                children.append(liffile.LifFile(
                    target, memmap=node._memmap, squeeze=node._squeeze,
                    _parent=node))
            if getattr(node.type, "name", "") == "XLIF":
                memory = node.xml_element.find("./Element/Memory")
                for block in (memory if memory is not None else ()):
                    if block.get("File"):
                        _leica_target(node.dirname, block.get("File"), root)
        except BaseException:
            for child in children:
                child.close()
            raise
        node.__dict__["children"] = tuple(children)   # liffile's cache
        stack.extend((child, ancestors | {here}) for child in children)


class LifReader(_MicroscopeReader):
    """A Leica LIF (or LOF/XLEF) file: each image in it is a scene, read
    with liffile, memory-mapped on the file where possible (only what is
    asked for is read). (0.23.0) Only files it holds or names inside its
    own folder are read (see _open_experiment).

    A tile scan (0.17.2) is two scenes: first stitched into one image (the
    user's choice, 2026-11-10; tile_stitching places the tiles where their
    overlaps match, when the scene is first chosen: prepare_scene), then
    its tiles one after the other (as before 0.17.2).
    """

    # (0.23.0) An .xlef experiment's folder: the files it names must lie in
    # it (None for a .lif or .lof, which name no files); see __init__.
    _root: str | None = None

    def __init__(self, path: str | Path):
        try:
            import liffile
        except ImportError:
            _missing_reader("liffile", "Leica LIF")
        self.path = Path(path)
        self._lif = liffile.LifFile(self.path)
        self._images, self.scenes, self._arrays = [], [], {}
        self._tile_scans = {}          # scene index -> _TileScan (stitched)
        self._checked = set()          # images whose frame files are checked
        try:
            kind = getattr(self._lif.type, "name", "")
            if self.path.suffix.lower() == ".xlef" and kind == "XLEF":
                self._root = os.path.abspath(self.path.parent)
                _open_experiment(self._lif, self._root)
            else:
                _refuse_leica_names(self._lif, self.path)
            for image in self._lif.images:
                self._add(image)
        except BaseException:
            self._lif.close()
            raise
        if not self.scenes:
            self._lif.close()
            raise ValueError(f"{self.path.name}: no images Volume Cropper "
                             "can read.")
        self.format_name = "Leica LIF"
        self.reader_name = f"liffile {liffile.__version__}"

    def _add(self, image):
        """The scenes of one image of the file: none if it cannot be read,
        two for a tile scan (stitched, then its tiles), else one."""
        scene = self._scene(image)
        if scene is None:
            return
        tiles = _tile_scan(image)
        if tiles is not None:
            self._tile_scans[len(self.scenes)] = tiles
            self._images.append(image)
            self.scenes.append(self._stitched_scene(image, tiles))
            scene = replace(scene, name=f"{scene.name} (tiles)",
                            notes=scene.notes + [
                                "The same tile scan, stitched into one "
                                "image, is the entry before this one."])
        self._images.append(image)
        self.scenes.append(scene)

    def _scene(self, image, without_tiles: bool = False) -> Scene | None:
        """The scene of one image; without_tiles: as if it had no tile
        axis (M), for the stitched image (whose y, x sizes are then still
        those of one tile; _stitched_scene sets them)."""
        dims, shape = tuple(image.dims), tuple(image.shape)
        if without_tiles:
            keep = [i for i, d in enumerate(dims) if str(d).upper() != "M"]
            dims = tuple(dims[i] for i in keep)
            shape = tuple(shape[i] for i in keep)
        if "Y" not in dims or "X" not in dims:
            return None
        axes, notes = _letters(dims, shape)
        scales, units = {}, {}
        for letter, values in image.coords.items():
            values = np.asarray(values)
            if values.ndim == 1 and values.size >= 2 and \
                    np.issubdtype(values.dtype, np.number):
                scales[letter.upper()] = float(values[1] - values[0])
                units[letter.upper()] = "m" if letter.upper() in "ZYX" \
                    else "s"
        voxel, unit, interval, origin = _calibration(axes, scales, units, {})
        if "T" in axes and interval is None:
            interval = _interval_from_times(getattr(image, "timestamps",
                                                    None),
                                            _size(axes, shape, "T"))
        n_c = _size(axes, shape, "C") if "C" in axes else 1
        names, colors = _colour_channels(dims, shape, [],
                                         _lif_colors(image, n_c), n_c)
        name = str(image.path or image.name)
        if re.search(r"(?i)[ _]merged$", name) and not _has_tilescan(image):
            notes.append(_LASX_MERGED_NOTE)
        return Scene(name=name, axes=axes,
                     shape=shape, dtype=np.dtype(image.dtype), voxel=voxel,
                     unit=unit, interval=interval,
                     time_unit="second" if interval else None,
                     channel_names=names, channel_colors=colors,
                     channel_windows=[None] * n_c, notes=notes,
                     origin=origin)

    def _stitched_scene(self, image, tiles: "_TileScan") -> Scene:
        """The stitched scene before its tiles are placed: its size from
        the stage positions (prepare_scene sets the real one)."""
        scene = self._scene(image, without_tiles=True)
        places = tile_stitching.stage_places(
            tiles.stage_yx, tiles.pixel_yx,
            tiles.orientation or (False, False, False))
        size = np.ceil(places.max(axis=0)).astype(int) \
            + np.array(tiles.tile_shape[1:])
        shape = list(scene.shape)
        shape[scene.axes.index("Y")] = int(size[0])
        shape[scene.axes.index("X")] = int(size[1])
        return replace(scene, name=f"{scene.name} (stitched)",
                       shape=tuple(shape),
                       notes=[f"A tile scan of {tiles.n} tiles, stitched "
                              "into one image where the tiles' overlaps "
                              "match (worked out when this image is chosen; "
                              "until then its size is from the stage "
                              "positions)."] + scene.notes)

    def prepare_scene(self, scene: int) -> bool:
        """Place a stitched scene's tiles (once): True if the scene (its
        size and notes) changed. Can take a few seconds: the tiles'
        overlaps are read."""
        tiles = self._tile_scans.get(scene)
        if tiles is None or tiles.layout is not None:
            return False
        array = self._array(scene)

        def read(k, c, zs, ys, xs):
            return tiles.read(array, k, {tiles.channel_dim: c}, zs, ys, xs)

        try:
            tiles.layout = tile_stitching.register(
                read, tiles.tile_shape, tiles.stage_yx, tiles.pixel_yx,
                n_channels=tiles.n_channels,
                stage_z_planes=tiles.stage_z_planes,
                orientation=tiles.orientation)
        except Exception as error:      # never let a file fail to open
            tiles.layout = tile_stitching.stage_layout(
                tiles.stage_yx, tiles.pixel_yx, tiles.tile_shape,
                tiles.orientation or (False, False, False),
                f"Matching the tiles' overlaps failed ({error}): they "
                "were placed by their stage positions.")
        old = self.scenes[scene]
        shape = list(old.shape)
        for letter, n in zip("ZYX", tiles.layout.shape):
            if letter in old.axes:
                shape[old.axes.index(letter)] = int(n)
        axis_notes = old.notes[1:]          # (the first: "A tile scan ...")
        self.scenes[scene] = replace(old, shape=tuple(shape),
                                     notes=tiles.layout.notes + axis_notes)
        return True

    def preparing_text(self, scene: int) -> str | None:
        """What prepare_scene will do (None: nothing to do)."""
        tiles = self._tile_scans.get(scene)
        if tiles is None or tiles.layout is not None:
            return None
        return (f"Stitching the tile scan ({tiles.n} tiles): finding where "
                "their overlaps match. This reads part of each tile and can "
                "take a few seconds, longer over a network.")

    def scene_record(self, scene: int) -> dict | None:
        """How a stitched scene was put together (for the crop's record)."""
        tiles = self._tile_scans.get(scene)
        if tiles is None or tiles.layout is None:
            return None
        return dict(tiles.layout.record, flags_in_file=tiles.flags)

    def _array(self, scene: int):
        """The scene's image as liffile gives it (memory-mapped), opened
        once per image (a tile scan's two scenes share it)."""
        image = self._images[scene]
        array = self._arrays.get(id(image))
        if array is None:
            self._check_frames(image)
            array = image.asarray(out="memmap")
            self._arrays[id(image)] = array
        return array

    def _check_frames(self, image):
        """(0.23.0; .xlef experiments) Before liffile reads an image's frame
        files: each lies inside the experiment's folder, a .lof frame holds
        its image itself (names no other files), and an OME-TIFF frame is
        opened as one plain TIFF (not the other files its OME-XML names, nor
        a Micro-Manager NDTiff index). liffile reads these checked files:
        they fill its caches, as _open_experiment does."""
        if self._root is None or id(image) in self._checked:
            return
        import liffile
        import tifffile
        from liffile.liffile import case_sensitive_path
        block, parent = image.memory_block, image.parent
        kind = getattr(block.type, "name", "")
        paths = []
        for frame in block.frames:       # (names already decoded by liffile)
            path = os.path.normpath(os.path.join(parent.dirname, frame.file))
            if not _inside(path, self._root):
                raise ValueError(
                    f"{self.path.name}: an image names the frame file "
                    f"{frame.file!r}, outside the experiment's folder.")
            paths.append(path if os.path.exists(path)
                         else case_sensitive_path(path))
        if kind == "LOF":
            for path in paths:
                lof = liffile.LifFile(path, memmap=parent._memmap,
                                      squeeze=parent._squeeze)
                try:
                    _refuse_leica_names(lof, Path(path))
                except BaseException:
                    lof.close()
                    raise
                if len(paths) == 1:              # liffile's _lof_reference
                    image.__dict__["_lof_reference"] = lof
                    parent._open_files.append(lof)
                else:          # (liffile opens each frame again to read it)
                    lof.close()
        elif kind in ("OME", "AIVIA") and len(paths) == 1:
            tif = tifffile.TiffFile(paths[0], is_ndtiff=False,
                                    _multifile=False)
            image.__dict__["_tif_reference"] = tif   # liffile's cache
            parent._open_files.append(tif)
        self._checked.add(id(image))

    def _read(self, scene: int, key: tuple):
        tiles = self._tile_scans.get(scene)
        if tiles is None:
            return self._array(scene)[key]
        self.prepare_scene(scene)
        return tiles.read_stitched(self._array(scene), key)

    def close(self):
        self._arrays.clear()
        self._lif.close()


def _has_tilescan(image) -> bool:
    try:
        return getattr(image, "tilescan", None) is not None
    except Exception:
        return False


# The note on images LAS X merged from a tile scan (the user's choice,
# 2026-11-11). In the merged images tried, the values were blended where
# tiles overlap, and even in the tiles' middles many voxels differed from
# the raw tiles.
_LASX_MERGED_NOTE = (
    "This looks like an image merged by LAS X from a tile scan (its name "
    "ends in \"merged\"): LAS X blends the tiles where they overlap, so its "
    "voxel values are not all the measured ones. For the measured values, "
    "open the tile scan itself, which Volume Cropper stitches without "
    "blending.")


@dataclass
class _TileScan:
    """A LIF tile scan: what stitching it needs (see tile_stitching)."""

    dims: tuple                  # liffile's dimension codes, upper case
    shape: tuple
    stage_yx: np.ndarray         # (n, 2) stage y, x of each tile, metres
    stage_z_planes: np.ndarray | None   # (n,) stage z in planes
    pixel_yx: np.ndarray         # (2,) pixel size y, x, metres
    tile_shape: tuple            # (z, y, x)
    channel_dim: str             # "C", or "S" for a colour image
    n_channels: int
    orientation: tuple | None    # None: found from the overlaps
    flags: dict
    layout: tile_stitching.Layout | None = None

    @property
    def n(self) -> int:
        return self.shape[self.dims.index("M")]

    def read(self, array, k: int, fixed: dict, zs, ys, xs) -> np.ndarray:
        """Tile k's voxels (z, y, x): zs, ys, xs are slices in the tile;
        `fixed`: the index along each other dimension (default 0: the
        first timepoint, channel ...)."""
        index = []
        for d in self.dims:
            if d == "M":
                index.append(k)
            elif d in "ZYX":
                index.append({"Z": zs, "Y": ys, "X": xs}[d])
            else:
                index.append(fixed.get(d, 0))
        block = np.asarray(array[tuple(index)])
        return block if "Z" in self.dims else block[np.newaxis]

    def read_stitched(self, array, key: tuple) -> np.ndarray:
        """What reading `key` from the stitched image gives: one entry per
        dimension but M, in the file's order; slices (or an index) for z,
        y, x, indices (or slices) for the others."""
        dims = [d for d in self.dims if d != "M"]
        sizes = dict(zip(dims, self.layout_shape(dims)))
        for i, (d, entry) in enumerate(zip(dims, key)):
            if d not in "ZYX" and isinstance(entry, slice):
                # e.g. all channels at once: one at a time, stacked
                kept = sum(isinstance(e, slice) for e in key[:i])
                parts = [self.read_stitched(array, key[:i] + (j,)
                                            + key[i + 1:])
                         for j in range(*entry.indices(sizes[d]))]
                return np.stack(parts, axis=kept)
        entries = dict(zip(dims, key))
        fixed = {d: int(e) for d, e in entries.items() if d not in "ZYX"}
        spatial, drop = {}, []
        for axis, (letter, n) in enumerate(zip("ZYX", self.layout.shape)):
            entry = entries.get(letter, slice(None))
            if not isinstance(entry, slice):          # an index: no axis
                entry = slice(int(entry), int(entry) + 1)
                drop.append(axis)
            start, stop, step = entry.indices(n)
            if step != 1:
                raise ValueError("A stitched image is read in steps of 1.")
            spatial[letter] = slice(start, max(start, stop))
        if "Z" not in dims:
            drop.append(0)

        def read_tile(k, zs, ys, xs):
            return self.read(array, k, fixed, zs, ys, xs)

        out = tile_stitching.assemble(read_tile, self.layout, spatial["Z"],
                                      spatial["Y"], spatial["X"],
                                      array.dtype)
        return out.squeeze(axis=tuple(sorted(set(drop)))) if drop else out

    def layout_shape(self, dims) -> list:
        """The size of each of `dims` in the stitched image."""
        sizes = dict(zip(self.dims, self.shape))
        sizes.update(zip("ZYX", self.layout.shape))
        return [sizes[d] for d in dims]


def _tile_scan(image) -> _TileScan | None:
    """The tile scan of a LIF image (liffile's `tilescan`), if it has one
    that can be stitched: two tiles or more, stage positions that differ,
    and z, y, x in that order."""
    dims = tuple(str(d).upper() for d in image.dims)
    shape = tuple(int(n) for n in image.shape)
    if "M" not in dims or shape[dims.index("M")] < 2:
        return None
    if "Y" not in dims or "X" not in dims or \
            [d for d in dims if d in "ZYX"] != [d for d in "ZYX" if d in dims]:
        return None
    try:
        info = image.tilescan
        tiles = info.tiles if info is not None else None
    except Exception:
        return None
    if tiles is None or len(tiles) != shape[dims.index("M")]:
        return None
    try:
        stage_yx = np.column_stack([tiles["pos_y"], tiles["pos_x"]]) \
            .astype(np.float64)
        stage_z = np.asarray(tiles["pos_z"], dtype=np.float64)
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    if not np.all(np.isfinite(stage_yx)) or \
            np.ptp(stage_yx, axis=0).max() <= 0:
        return None                                  # no stage positions
    steps = {}
    for letter in "ZYX":
        values = np.asarray(image.coords.get(letter, []), dtype=np.float64)
        if values.ndim == 1 and values.size >= 2 and values[1] > values[0]:
            steps[letter] = float(values[1] - values[0])
    if "Y" not in steps or "X" not in steps:
        return None
    flags = {"flip_x": bool(info.flip_x), "flip_y": bool(info.flip_y),
             "swap_xy": bool(info.swap_xy)}
    sizes = dict(zip(dims, shape))
    rest = tuple(d for d in dims if d != "M")
    colour = _is_colour_image(rest, tuple(sizes[d] for d in rest))
    channel_dim = "S" if colour else "C"
    stage_z_planes = None
    if "Z" in steps and np.all(np.isfinite(stage_z)):
        stage_z_planes = stage_z / steps["Z"]
    return _TileScan(
        dims=dims, shape=shape, stage_yx=stage_yx,
        stage_z_planes=stage_z_planes,
        pixel_yx=np.array([steps["Y"], steps["X"]]),
        tile_shape=(sizes.get("Z", 1), sizes["Y"], sizes["X"]),
        channel_dim=channel_dim, n_channels=sizes.get(channel_dim, 1),
        # Stage y, x = image y, x when the file's flags are all 0 (seen
        # in real tile scans, 2026-11-10). Other flags: found from the
        # overlaps.
        orientation=((False, False, False) if not any(flags.values())
                     else None),
        flags=flags)


def _lif_colors(image, n: int) -> list:
    """Channel colours from the LIF's LUT names (Red, Green, Gray ...)."""
    import re
    import xml.etree.ElementTree as ET
    try:
        text = ET.tostring(image.xml_element, encoding="unicode")
    except Exception:
        return [None] * n
    names = re.findall(r'<ChannelDescription[^>]*LUTName="([^"]*)"', text)
    colors = [_LUT_COLORS.get(name.strip().lower()) for name in names]
    return (colors + [None] * n)[:n]


class CziReader(_MicroscopeReader):
    """A Zeiss CZI file: each scene is a scene (its full resolution),
    read with czifile, only the planes and the region asked for."""

    def __init__(self, path: str | Path):
        try:
            import czifile
        except ImportError:
            _missing_reader("czifile", "Zeiss CZI")
        self.path = Path(path)
        self._czi = czifile.CziFile(self.path)
        try:
            self._keys = list(self._czi.scenes.keys())
            self._images = [self._czi.scenes[k] for k in self._keys]
            times = None
            try:
                times = self._czi.timestamps
            except Exception:
                pass
            self.scenes = [self._scene(image, k, times)
                           for k, image in zip(self._keys, self._images)]
        except BaseException:
            self._czi.close()
            raise
        self.format_name = "Zeiss CZI"
        self.reader_name = f"czifile {czifile.__version__}"
        try:
            self.compression = _compression_name(self._images[0].compression)
        except Exception:
            pass

    def _scene(self, image, key, times) -> Scene:
        dims, shape = tuple(image.dims), tuple(image.shape)
        axes, notes = _letters(dims, shape)
        scales = {k.upper(): v for k, v in (image.coord_scales or {}).items()}
        units = {k.upper(): v for k, v in (image.coord_units or {}).items()}
        voxel, unit, interval, origin = _calibration(axes, scales, units, {})
        if "T" in axes:
            # the frame times, when the file has them (czifile's T step can
            # be a nominal value)
            from_times = _interval_from_times(times) if times is not None \
                else None
            if from_times is not None:
                interval = from_times
        names, colors = [], []
        for name, info in (image.channels or {}).items():
            names.append(str(name))
            colors.append(_argb_color((info or {}).get("Color")))
        n_c = _size(axes, shape, "C") if "C" in axes else 1
        names = (names + [""] * n_c)[:n_c] if "C" in axes else []
        colors = (colors + [None] * n_c)[:n_c]
        names, colors = _colour_channels(dims, shape, names, colors, n_c)
        if getattr(image, "is_pyramid", False):
            notes.append("The file holds a pyramid: its full resolution is "
                         "read.")
        name = getattr(image, "name", None) or f"scene {key}"
        return Scene(name=str(name), axes=axes, shape=shape,
                     dtype=np.dtype(image.dtype), voxel=voxel, unit=unit,
                     interval=interval,
                     time_unit="second" if interval else None,
                     channel_names=names, channel_colors=colors,
                     channel_windows=[None] * n_c, notes=notes,
                     origin=origin)

    def _read(self, scene: int, key: tuple):
        """The part `key` asks for (one entry per dimension of the scene,
        in its order): czifile takes the planes as a selection (file
        coordinates: the scene's start added) and y, x as a region."""
        image = self._images[scene]
        dims = [str(d).upper() for d in image.dims]
        start = dict(zip(dims, getattr(image, "start", (0,) * len(dims))))
        selection, kept, sample = {}, [], None
        y_key = x_key = slice(None)
        for code, part, n in zip(dims, key, image.shape):
            if code == "Y":
                y_key = part
                kept.append(code)
            elif code == "X":
                x_key = part
                kept.append(code)
            elif code == "S":       # colour samples: always read whole, last
                if isinstance(part, slice):
                    kept.append(code)
                else:
                    sample = int(part)
            else:
                offset = start.get(code, 0)
                if isinstance(part, slice):
                    first, stop, _ = part.indices(n)
                    selection[code] = slice(first + offset, stop + offset)
                    kept.append(code)
                else:
                    selection[code] = int(part) + offset
        y0, y1, _ = y_key.indices(image.shape[dims.index("Y")])
        x0, x1, _ = x_key.indices(image.shape[dims.index("X")])
        bbox = image.bbox                # (x, y, width, height), absolute
        part = image(roi=(bbox[0] + x0, bbox[1] + y0, x1 - x0, y1 - y0),
                     **selection)
        data = part.asarray()
        got = [str(d).upper() for d in part.dims]
        if sample is not None and "S" in got:      # one colour sample
            data = np.take(data, sample, axis=got.index("S"))
            got.remove("S")
        order = [got.index(code) for code in kept if code in got]
        if order != list(range(len(order))):
            data = np.transpose(data, order)
        return data

    def close(self):
        self._czi.close()


def _compression_name(value) -> str:
    """A short name for a file's compression ("zstd", "jpeg xr", "lzw" or
    "none"), from what czifile reports (e.g. "zstdhdr (6)")."""
    text = str(value or "").lower()
    if not text or "uncompressed" in text or text.startswith("none"):
        return "none"
    for key, name in (("zstd", "zstd"), ("jpgxr", "jpeg xr"),
                      ("jpegxr", "jpeg xr"), ("jpg", "jpeg"),
                      ("lzw", "lzw"), ("camera", "camera-specific")):
        if key in text:
            return name
    return text.split()[0]


def _argb_color(value) -> str | None:
    """"#AARRGGBB" or "#RRGGBB" (Zeiss) -> "RRGGBB"."""
    if not value:
        return None
    text = str(value).strip().lstrip("#")
    if len(text) == 8:
        text = text[2:]
    return text.upper() if len(text) == 6 else None


class Nd2Reader(_MicroscopeReader):
    """A Nikon ND2 file: each position (P) is a scene, read with nd2
    through a lazy (dask) array, only the frames asked for."""

    def __init__(self, path: str | Path):
        try:
            import nd2
        except ImportError:
            _missing_reader("nd2", "Nikon ND2")
        self.path = Path(path)
        self._file = nd2.ND2File(self.path)
        try:
            self._dask = self._file.to_dask()
            sizes = dict(self._file.sizes)
            self._dims = [str(d).upper() for d in sizes]
            n_p = sizes.get("P", 1)
            self.scenes = [self._scene(sizes, p, n_p) for p in range(n_p)]
        except BaseException:
            self._file.close()
            raise
        self.format_name = "Nikon ND2"
        self.reader_name = f"nd2 {nd2.__version__}"

    def _scene(self, sizes: dict, p: int, n_p: int) -> Scene:
        dims = [d for d in self._dims if d != "P"]
        shape = tuple(sizes[d] for d in sizes if str(d).upper() != "P")
        axes, notes = _letters(dims, shape)
        voxel = {"z": None, "y": None, "x": None}
        try:
            size = self._file.voxel_size()
            for k in "zyx":
                value = _positive(getattr(size, k))
                if value is not None and (k != "z" or "Z" in axes):
                    voxel[k] = value
        except Exception:
            pass
        unit = "micrometer" if any(voxel.values()) else None
        interval = None
        if "T" in axes:
            try:
                for loop in self._file.experiment or []:
                    if getattr(loop, "type", "") in ("TimeLoop", "NETimeLoop"):
                        period = getattr(loop.parameters, "periodMs", None)
                        if period is None and hasattr(loop.parameters,
                                                      "periods"):
                            period = loop.parameters.periods[0].periodMs
                        interval = _positive(period)
                        interval = interval / 1000 if interval else None
                        break
            except Exception:
                interval = None
        names, colors = [], []
        try:
            for channel in self._file.metadata.channels or []:
                names.append(str(channel.channel.name))
                colors.append(_nd2_color(channel.channel))
        except Exception:
            pass
        n_c = _size(axes, shape, "C") if "C" in axes else 1
        names = (names + [""] * n_c)[:n_c] if "C" in axes else []
        colors = (colors + [None] * n_c)[:n_c]
        names, colors = _colour_channels(dims, shape, names, colors, n_c)
        name = (self.path.stem if n_p == 1 else f"position {p + 1}")
        return Scene(name=name, axes=axes, shape=shape,
                     dtype=np.dtype(self._file.dtype), voxel=voxel,
                     unit=unit, interval=interval,
                     time_unit="second" if interval else None,
                     channel_names=names, channel_colors=colors,
                     channel_windows=[None] * n_c, notes=notes,
                     origin={"z": 0.0, "y": 0.0, "x": 0.0})

    def _read(self, scene: int, key: tuple):
        full = list(key)
        if "P" in self._dims:
            full.insert(self._dims.index("P"), scene)
        return self._dask[tuple(full)].compute()

    def close(self):
        self._file.close()


def _nd2_color(channel) -> str | None:
    """A Nikon channel's colour as "RRGGBB" (nd2 gives an int 0xBBGGRR or a
    Color object)."""
    try:
        color = getattr(channel, "color", None)
        if color is not None and hasattr(color, "r"):
            return f"{color.r:02X}{color.g:02X}{color.b:02X}"
        value = int(channel.colorRGB)
        r, g, b = value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF
        return f"{r:02X}{g:02X}{b:02X}"
    except Exception:
        return None


class OirReader(_MicroscopeReader):
    """An Olympus/Evident OIR file (one image), read with oirfile,
    memory-mapped where possible. (Written without an OIR file to test
    on: experimental.)"""

    def __init__(self, path: str | Path):
        try:
            import oirfile
        except ImportError:
            _missing_reader("oirfile", "Olympus OIR")
        self.path = Path(path)
        self._oir = oirfile.OirFile(self.path)
        self._array = None
        try:
            dims, shape = tuple(self._oir.dims), tuple(self._oir.shape)
            axes, notes = _letters(dims, shape)
            scales = {k.upper(): v for k, v in
                      (self._oir.coord_scales or {}).items()}
            units = {k.upper(): v for k, v in
                     (self._oir.coord_units or {}).items()}
            voxel, unit, interval, origin = _calibration(axes, scales,
                                                         units, {})
            names = _oir_channel_names(self._oir)
            n_c = _size(axes, shape, "C") if "C" in axes else 1
            names = (names + [""] * n_c)[:n_c] if "C" in axes else []
            names, colors = _colour_channels(dims, shape, names,
                                             [None] * n_c, n_c)
            notes.append("OIR reading was tried on a few files only: please "
                         "compare the voxel size and channel names with "
                         "Fiji.")
            self.scenes = [Scene(
                name=self.path.stem, axes=axes, shape=shape,
                dtype=np.dtype(self._oir.dtype), voxel=voxel, unit=unit,
                interval=interval, time_unit="second" if interval else None,
                channel_names=names, channel_colors=colors,
                channel_windows=[None] * n_c, notes=notes, origin=origin)]
        except BaseException:
            self._oir.close()
            raise
        self.format_name = "Olympus OIR"
        self.reader_name = f"oirfile {oirfile.__version__}"

    def _read(self, scene: int, key: tuple):
        if self._array is None:
            self._array = self._oir.asarray(out="memmap")
        return self._array[key]

    def close(self):
        self._array = None
        self._oir.close()


MICROSCOPE_READERS = {".lif": LifReader, ".lof": LifReader,
                      ".xlef": LifReader, ".czi": CziReader,
                      ".nd2": Nd2Reader, ".oir": OirReader}


class LayersGone(Exception):
    """The napari layers a LayerReader reads were deleted or changed."""


# Axis labels of napari layers that name an axis (anything else, e.g.
# napari's default "axis -1", is ignored and the axis found by position).
_AXIS_LABELS = {"t": "T", "time": "T", "c": "C", "ch": "C", "channel": "C",
                "z": "Z", "y": "Y", "x": "X"}


class LayerReader:
    """Image layers open in napari, read like a file (0.13.0, the napari
    plugin): one layer, or several of the same size as the channels of one
    image. Their data can come from any napari reader (another plugin's
    CZI or LIF reader, a sample ...). Crops are OME-TIFFs.

    Axes: from the layers' axis labels where they name t, c, z, y or x,
    otherwise by position (the last two y and x, before them z, then t).
    Voxel size and position: the layers' scale and translate; unit: their
    units ("pixel" = no calibration known). Display: their colours and
    contrast limits.

    The layers are held by weak references (keep=False): a layer the user
    deletes in napari is freed, and reading it again raises LayersGone.
    Everything else is copied when the reader is made. keep=True holds them
    (when the tool takes over the window, it removes them itself).

    Lazy (dask) layers are read a whole chunk at a time: load() asks for a
    few planes at a time, and a reader behind a dask chunk (e.g. one
    z-stack per chunk) would otherwise read the chunk again for each.
    """

    def __init__(self, layers, keep: bool = False):
        import weakref

        import napari
        # napari hands plugins proxies of its objects; the reader works on
        # the layers themselves (a weak reference to a proxy would die at
        # once). "__wrapped__" is the proxy's public way to them.
        layers = [getattr(layer, "__wrapped__", layer) for layer in layers]
        if not layers:
            raise ValueError("Select one or more image layers first.")
        self._refs = [weakref.ref(layer) for layer in layers]
        self._kept = list(layers) if keep else []
        self._cache = None        # (key, block) of the last dask chunk read
        data = [_level0(layer) for layer in layers]
        shape = tuple(data[0].shape)
        dtype = np.dtype(data[0].dtype)
        first = layers[0]
        for layer, array in zip(layers, data):
            if getattr(layer, "rgb", False):
                raise NotImplementedError(
                    f"{layer.name} is a colour (RGB) layer; those are not "
                    "supported.")
            if tuple(array.shape) != shape or np.dtype(array.dtype) != dtype:
                raise ValueError(
                    f"The layers differ in size or data type ({first.name}: "
                    f"{shape}, {dtype}; {layer.name}: {tuple(array.shape)}, "
                    f"{array.dtype}). Select layers of one image (its "
                    "channels).")
            if not np.allclose(np.asarray(layer.scale, float),
                               np.asarray(first.scale, float)):
                raise ValueError(
                    f"The layers have different voxel sizes ({first.name}: "
                    f"{tuple(first.scale)}; {layer.name}: "
                    f"{tuple(layer.scale)}). Select layers of one image.")
            # channels must lie on one another as napari shows them
            step = float(np.min(np.abs(np.asarray(first.scale, float)))) or 1
            if not np.allclose(np.asarray(layer.translate, float),
                               np.asarray(first.translate, float),
                               rtol=0, atol=0.01 * step):
                raise ValueError(
                    f"The layers are at different positions ({first.name}: "
                    f"{tuple(first.translate)}; {layer.name}: "
                    f"{tuple(layer.translate)}): napari shows them shifted "
                    "against each other. Select layers of one image.")
            if not _plain_transform(layer):
                raise NotImplementedError(
                    f"{layer.name} is rotated, sheared or has an affine "
                    "transform in napari; such layers are not supported.")
        self._shape = shape
        n = len(shape)
        labels = [str(a).strip().lower() for a in
                  (getattr(first, "axis_labels", None) or [])][-n:]
        labels = labels if len(labels) == n else [""] * n
        # Axes: those the labels name are fixed; the others are guessed by
        # position (the last two y and x, then z, then t, from the end),
        # and the guesses other than y and x stay changeable in "What to
        # load" ("?" in the scene's axes, the guess as the default).
        named = [_AXIS_LABELS.get(label, "") for label in labels]
        named = [letter if letter and named.count(letter) == 1 else ""
                 for letter in named]
        guess = list(named)
        # (from the end: x, y, z, then t; five axes: c, then t, as in
        # Fiji's TCZYX)
        order = "XYZCT" if n >= 5 else "XYZTC"
        for k in range(n - 1, -1, -1):
            if guess[k]:
                continue
            for letter in order:
                if letter not in guess:
                    guess[k] = letter
                    break
            else:
                guess[k] = "?"
        locked = [g if (named[k] or g in "YX") else "?"
                  for k, g in enumerate(guess)]
        axes, guess = "".join(locked), "".join(guess)
        self._data_axes = guess
        scale = [float(s) for s in first.scale][-n:]
        translate = [float(s) for s in first.translate][-n:]
        units = [str(u) for u in (getattr(first, "units", None) or [])][-n:]
        unit_of = dict(zip(guess, units)) if len(units) == n else {}
        unit = space_unit(unit_of.get("X"))
        voxel = {k: (scale[guess.index(k.upper())] if k.upper() in guess
                     else None) for k in "zyx"}
        # napari's default units are "pixel": then the scale is still the
        # voxel size if a script or reader set one (only the unit is
        # unknown); a scale of all 1 says nothing.
        scale_given = any(v is not None and abs(v - 1.0) > 1e-9
                          for v in voxel.values())
        if unit is None and not scale_given:
            voxel = {k: None for k in "zyx"}
        # where voxel 0 sits, in voxels (as Fiji's origin for TIFFs)
        origin = {k: (translate[guess.index(k.upper())]
                      / scale[guess.index(k.upper())]
                      if k.upper() in guess and scale[guess.index(k.upper())]
                      else 0.0) for k in "zyx"}
        t_unit = time_unit(unit_of.get("T"))
        interval = (scale[guess.index("T")] if "T" in guess and t_unit
                    else None)
        names, colors, windows = [], [], []
        for layer in layers:
            names.append(str(layer.name))
            colors.append(_layer_color(layer))
            try:
                low, high = (float(v) for v in layer.contrast_limits)
                windows.append((low, high))
            except Exception:
                windows.append(None)
        if len(layers) > 1:               # several layers: the channels
            if "C" in axes or n >= 5:
                raise ValueError(
                    "These layers have channels of their own (or five axes "
                    "each); several of them cannot be the channels of one "
                    "image. Select one of them.")
            axes, guess = "C" + axes, "C" + guess.replace("C", "?")
            shape = (len(layers),) + shape
        else:
            names = []
        self.layer_names = [str(layer.name) for layer in layers]
        source = source_file(layers, openable=False)
        # The path names the crops (and is where "Save regions" starts):
        # the layers' file, else a name in the home folder (never the
        # current folder, which may be the shared tool folder).
        self.path = (source if source is not None
                     else Path.home() / output_name_safe(first.name))
        self.format_name = ("napari layer" if len(layers) == 1
                            else f"{len(layers)} napari layers")
        self.output_kind = "ome"
        self.compression = "none"
        self.reader_name = f"napari {napari.__version__}"
        self.record = None
        notes = [f"From napari: {', '.join(self.layer_names)}."]
        if unit is None and scale_given:
            notes.append("The layers give a voxel size but no unit "
                         "(micrometre is assumed): check it below.")
        elif unit is None:
            notes.append("The layers have no voxel size (their scale is in "
                         "pixels): enter it below.")
        if "?" in axes:
            notes.append("Axes without a name in napari were guessed from "
                         "their position: check them below.")
        self.scenes = [Scene(
            name=str(first.name), axes=axes, shape=shape, dtype=dtype,
            voxel=voxel, unit=unit, interval=interval,
            time_unit=t_unit if interval else None,
            channel_names=names, channel_colors=colors,
            channel_windows=windows, notes=notes, origin=origin,
            axes_guess=guess if "?" in axes else "")]

    def alive(self) -> bool:
        """True while every layer still exists, with its size."""
        try:
            self._arrays()
            return True
        except LayersGone:
            return False

    def _arrays(self) -> list:
        """The layers' full-resolution data, now."""
        arrays = []
        for name, ref in zip(self.layer_names, self._refs):
            layer = ref()
            if layer is None:
                raise LayersGone(f"The napari layer {name} was deleted.")
            array = _level0(layer)
            if tuple(array.shape) != self._shape:
                raise LayersGone(f"The napari layer {name} has changed size.")
            arrays.append(array)
        return arrays

    def read(self, scene: int, key: tuple) -> np.ndarray:
        arrays = self._arrays()
        if len(arrays) > 1:
            c, key = key[0], key[1:]
        else:
            c = 0
        array = arrays[c]
        if not _is_dask(array):
            return np.asarray(array[key])
        return self._read_dask(c, array, key)

    def _read_dask(self, c: int, array, key: tuple) -> np.ndarray:
        """array[key] for a dask array, through a whole-chunk block along
        z: the next planes asked for then come from the block. (z: load()
        asks for a range along z and y and x, and numbers for t and c; the
        one range besides y's and x's is z, whatever the user chose.)"""
        yx = {self._data_axes.find("Y"), self._data_axes.find("X")}
        ranges = [i for i, k in enumerate(key)
                  if isinstance(k, slice) and i not in yx]
        if len(ranges) != 1:
            return np.asarray(array[key])
        z = ranges[0]
        start, stop, _ = key[z].indices(array.shape[z])
        bounds = np.cumsum((0,) + tuple(array.chunks[z]))
        first = bounds[np.searchsorted(bounds, start, side="right") - 1]
        end = bounds[np.searchsorted(bounds, stop - 1, side="right")]
        others = tuple((k.start, k.stop, k.step) if isinstance(k, slice)
                       else k for i, k in enumerate(key) if i != z)
        cache_key = (c, others, int(first), int(end))
        if self._cache is None or self._cache[0] != cache_key:
            self._cache = None          # (not two blocks in memory at once)
            wide = key[:z] + (slice(int(first), int(end)),) + key[z + 1:]
            self._cache = (cache_key, np.asarray(array[wide]))
        block = self._cache[1]
        # z's place in the block: the axes before it that are not indexed
        # away by a number
        at = sum(1 for k in key[:z] if isinstance(k, slice))
        index = [slice(None)] * block.ndim
        index[at] = slice(start - int(first), stop - int(first))
        return block[tuple(index)]

    def release(self):
        """Forget the last chunk read (load() calls it once the full
        resolution is read, before the smaller copies are made)."""
        self._cache = None

    def close(self):
        """Forget the last chunk read (the layers stay napari's)."""
        self._cache = None

    def extra_memory(self, choice: FileChoice) -> int:
        """Bytes held while reading, besides the image itself: for lazy
        layers, one block of whole chunks along z (see _read_dask)."""
        arrays = [a for a in self._arrays() if _is_dask(a)] \
            if self.alive() else []
        if not arrays:
            return 0
        scene = self.scenes[0]
        axes, sizes, picks = picks_of(scene, choice)
        data_axes = axes[1:] if len(self._refs) > 1 else axes
        if "Z" not in data_axes:
            return 0
        z = data_axes.index("Z")
        n_yx = len(picks.get("Y", range(1))) * len(picks.get("X", range(1)))
        biggest = max(max(a.chunks[z]) for a in arrays)
        # a block spans at most two chunks (16 planes asked for at a time)
        return 2 * biggest * n_yx * np.dtype(scene.dtype).itemsize


def _level0(layer):
    """A layer's full-resolution data (level 0 of a multiscale layer)."""
    return layer.data[0] if layer.multiscale else layer.data


def _is_dask(array) -> bool:
    return hasattr(array, "dask") and hasattr(array, "chunks")


def _plain_transform(layer) -> bool:
    """True if napari shows the layer only scaled and shifted (no rotation,
    shear or affine transform), as the tool can show it."""
    try:
        n = len(layer.scale)
        rotate = np.asarray(layer.rotate, float)
        shear = np.asarray(layer.shear, float)
        affine = np.asarray(layer.affine.affine_matrix, float)
        return (np.allclose(rotate, np.eye(n)) and np.allclose(shear, 0)
                and np.allclose(affine, np.eye(n + 1)))
    except Exception:
        return True                     # napari without these: plain


def _layer_color(layer) -> str | None:
    """The colour a layer's colormap ends in, as "RRGGBB" (white for grey)."""
    try:
        rgb = np.asarray(layer.colormap.colors[-1][:3], float)
        return "".join(f"{int(round(v * 255)):02X}" for v in rgb)
    except Exception:
        return None


def source_file(layers, openable: bool = True) -> Path | None:
    """The one file (or folder) the layers were opened from, if they all
    come from the same; with openable=True only if Volume Cropper can open
    it itself (an OME-Zarr, a TIFF, a folder of TIFFs). None otherwise."""
    paths = set()
    for layer in layers:
        layer = getattr(layer, "__wrapped__", layer)
        source = getattr(layer, "source", None)
        path = getattr(source, "path", None)
        if not path:
            return None
        paths.add(str(path))
    if len(paths) != 1:
        return None
    text = paths.pop()
    if "://" in text:                     # a URL: not a file here
        return None
    path = Path(text)
    if not openable:
        return path if path.exists() else None
    try:
        if not path.exists():
            return None
        if path.is_dir():
            # an OME-Zarr (or a folder of TIFFs, which is_file_image finds)
            if is_file_image(path) or path.suffix.lower() == ".zarr" or any(
                    (path / name).exists()
                    for name in ("zarr.json", ".zgroup", ".zattrs")):
                return path
            return None
        # a TIFF, or (0.21.0) a zipped OME-Zarr
        return path if path.name.lower().endswith(
            TIFF_SUFFIXES + vc.ZIP_SUFFIXES) else None
    except OSError:
        return None


def output_name_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "layer"


def is_file_image(path: str | Path) -> bool:
    """True for an image file or a folder of TIFFs; False for an OME-Zarr
    (a .zarr folder, or a folder with Zarr metadata; since 0.21.0 also a
    zipped one: .ozx, .zip)."""
    path = Path(path)
    if vc.is_zipped(path):
        return False
    if path.is_dir():
        if path.suffix.lower() == ".zarr" or any(
                (path / name).exists()
                for name in ("zarr.json", ".zgroup", ".zattrs")):
            return False
        return any(p.suffix.lower() in TIFF_SUFFIXES for p in path.iterdir())
    return path.is_file()


COMPANION_SUFFIXES = (".companion.ome", ".companion.ome.xml")


def open_companion(path: str | Path) -> TiffReader:
    """An OME-TIFF set described by a companion file (0.16.0): the OME
    metadata are in the companion (an XML file, e.g. embryo.companion.ome),
    the planes in the TIFF files it lists (in its folder). Raises a clear
    error when one of them is missing.

    A "companion" that is itself a TIFF (a master OME-TIFF that holds the
    whole set's metadata, as the OME-TIFF specification allows) is opened
    as the TIFF it is: tifffile reads the whole set from it."""
    import html
    import os
    import re
    import xml.etree.ElementTree as ET
    path = Path(path)
    with open(path, "rb") as file:
        start = file.read(4)
    if start in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return TiffReader(path)
    if path.stat().st_size > COMPANION_MAX_BYTES:
        raise ValueError(f"{path.name} is too large for an OME companion file "
                         f"({path.stat().st_size / 1e6:.0f} MB).")
    xml = path.read_text(encoding="utf-8", errors="replace")
    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except ET.ParseError as err:
        raise ValueError(f"{path.name} is not a readable OME companion file "
                         f"({err}).") from None
    names = []
    for element in root.iter():
        if element.tag.endswith("UUID") and element.attrib.get("FileName"):
            if element.attrib["FileName"] not in names:
                names.append(element.attrib["FileName"])
    if not names:
        raise ValueError(f"{path.name} lists no TIFF files (no UUID "
                         "FileName in its OME metadata).")
    _refuse_files_elsewhere(path, names)
    missing = [n for n in names if not (path.parent / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path.name} lists TIFF files that are not in its folder: "
            + ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
            + ". Keep the companion file and its TIFFs together.")
    # tifffile finds the files relative to the first TIFF's folder, the
    # companion names them relative to its own: e.g. "data/a.ome.tif" from
    # the companion's folder is "a.ome.tif" from data/ (the review of
    # 2026-11-08: such sets opened as their first file alone)
    first = path.parent / names[0]

    def from_first(found):
        name = html.unescape(found.group(1))
        moved = os.path.relpath(path.parent / name, first.parent)
        return f'FileName="{html.escape(moved, quote=True)}"'
    xml = re.sub(r'FileName="([^"]*)"', from_first, xml)
    return TiffReader(path, tiff=first, omexml=xml)


# A companion file is a small XML file; anything larger is not read as one.
COMPANION_MAX_BYTES = 64_000_000


def _outside_folder(name: str) -> bool:
    """Whether a file name from an OME-XML leaves the folder of the file
    that names it: an absolute path, a drive, a network share
    (\\\\host\\share) or a "..". Files in sub-folders are fine."""
    text = name.replace("\\", "/")
    return (not text or text.startswith("/") or ":" in text
            or ".." in text.split("/"))


def _refuse_files_elsewhere(path: Path, names) -> None:
    """Refuse an OME-TIFF set whose metadata name files outside its folder
    (the 0.17.0 review: a crafted file could make the tool read other
    files, or make Windows connect to a network share and send it the
    user's login). Checked before any of those files is touched."""
    outside = [n for n in names if _outside_folder(n)]
    if outside:
        raise ValueError(
            f"{path.name} names files outside its folder: "
            + ", ".join(outside[:3]) + (" …" if len(outside) > 3 else "")
            + ". Volume Cropper only opens the files of an OME-TIFF set "
            "that lie in its folder (or in folders below it).")


def _ome_file_entries(xml: str, own_name: str) -> tuple[list, list]:
    """The files an OME-TIFF's OME-XML names for its planes (UUID
    FileName): (all names, the names of OTHER files), once each, in order.
    Which entries are the file itself follows tifffile's own rule exactly
    (its series_ome): the OME element's UUID, or else the first entry with
    the file's own name, and every later entry with the same UUID text (a
    renamed OME-TIFF still names its old name). (The second review of
    0.17.1: an own UUID followed by a space slipped through a looser test,
    and tifffile then opened the file it named.)"""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return [], []
    known = set()
    own_uuid = root.attrib.get("UUID")
    if own_uuid is not None:
        known.add(own_uuid)
    names, others = [], []
    for element in root.iter():
        name = element.attrib.get("FileName")
        if not element.tag.endswith("UUID") or name is None:
            continue
        if name not in names:
            names.append(name)
        text = element.text
        if own_uuid is None and not known and text is not None \
                and name.lower() == own_name.lower():
            known.add(text)                     # the file itself
        elif text not in known:                 # a file tifffile opens
            known.add(text)
            if name not in others:
                others.append(name)
    return names, others


def binary_only_companion(path: str | Path) -> Path | None:
    """For an OME-TIFF whose metadata are kept in a companion file (its own
    OME-XML says "BinaryOnly" and names the file): that companion file;
    None for any other TIFF."""
    import re

    import tifffile
    try:
        with tifffile.TiffFile(path, is_ndtiff=False) as tif:
            xml = tif.ome_metadata if tif.is_ome else None
    except Exception:
        return None
    if not xml or "BinaryOnly" not in xml:
        return None
    found = re.search(r'BinaryOnly[^>]*MetadataFile="([^"]+)"', xml)
    if not found:
        return None
    _refuse_files_elsewhere(Path(path), [found.group(1)])
    companion = Path(path).parent / found.group(1)
    if not companion.is_file():
        raise FileNotFoundError(
            f"{Path(path).name} keeps its metadata in {found.group(1)}, "
            "which is not in its folder.")
    return companion


def open_file(path: str | Path):
    """A reader for an image file (metadata only; nothing of the image is
    read yet). Raises a clear error for files it cannot open."""
    path = Path(path)
    if path.is_dir():
        return TiffSequenceReader(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    name = path.name.lower()
    if name.endswith(COMPANION_SUFFIXES):        # OME-TIFF companion file
        return open_companion(path)
    if name.endswith(TIFF_SUFFIXES):
        companion = binary_only_companion(path)
        if companion is not None:     # its metadata are elsewhere: in a
            return open_companion(companion)     # companion or master TIFF
        return TiffReader(path)
    suffix = path.suffix.lower()
    if suffix in MICROSCOPE_READERS:              # LIF, CZI, ND2, OIR
        return MICROSCOPE_READERS[suffix](path)
    if suffix in LATER_FORMATS:
        raise NotImplementedError(
            f"{LATER_FORMATS[suffix]} files cannot be opened yet. "
            "Meanwhile, open it in Fiji and save it as TIFF (File > "
            "Save As > Tiff), then open the TIFF here.")
    raise ValueError(
        f"{path.name}: Volume Cropper cannot open {suffix or 'such'} files. "
        "It opens OME-Zarr images (.zarr folders), TIFF files, folders of "
        "TIFF files, and Leica LIF, Zeiss CZI, Nikon ND2 and Olympus OIR "
        "files.")


# ---------------------------------------------------------------------------
# Reading into memory
# ---------------------------------------------------------------------------

def picks_of(scene: Scene, choice: FileChoice) -> tuple[str, dict, dict]:
    """(axes, sizes, picks): picks are the indices read per axis letter:
    a range for T, Z, Y and X, a list for C."""
    axes = axes_of(scene, choice)
    sizes = sizes_of(scene, axes)
    picks = {}
    for letter in "TZYX":
        if letter in sizes:
            n = sizes[letter]
            first, end = choice.ranges.get(letter.lower(), (0, n))
            first, end = max(0, int(first)), min(n, int(end))
            if first >= end:
                raise ValueError(f"The {letter.lower()} range is empty "
                                 f"({first + 1}–{end} of {n}).")
            picks[letter] = range(first, end)
    if "C" in sizes:
        channels = (list(range(sizes["C"])) if choice.channels is None
                    else sorted(set(choice.channels)))
        if not channels or not all(0 <= c < sizes["C"] for c in channels):
            raise ValueError(f"Channels must be 0–{sizes['C'] - 1}; "
                             f"got {choice.channels}.")
        picks["C"] = channels
    return axes, sizes, picks


def calibration_of(scene: Scene, choice: FileChoice) -> dict:
    """The calibration used: voxel {"z","y","x"}, unit, interval, time_unit,
    and per value where it comes from ("file", "entered" or "assumed")."""
    voxel, source = {}, {}
    given = choice.voxel or {}
    for k in "zyx":
        if given.get(k) is not None:
            voxel[k], source[k] = float(given[k]), "entered"
        elif scene.voxel.get(k) is not None:
            voxel[k], source[k] = float(scene.voxel[k]), "file"
        else:
            voxel[k], source[k] = None, "assumed"
    # A missing y or x takes the other's size; z takes x's (a 2D image's
    # one plane is as thick as a pixel is wide); 1 if nothing is known.
    voxel["y"] = voxel["y"] or voxel["x"]
    voxel["x"] = voxel["x"] or voxel["y"]
    for k in "yx":
        voxel[k] = voxel[k] or 1.0
    voxel["z"] = voxel["z"] or voxel["x"]
    unit = choice.unit or scene.unit or "micrometer"
    unit_source = ("entered" if choice.unit else "file" if scene.unit
                   else "assumed")
    interval = choice.interval or scene.interval
    t_unit = choice.time_unit or (scene.time_unit if scene.interval else None)
    return {"voxel": voxel, "unit": unit, "interval": interval or 1.0,
            "time_unit": t_unit if interval else None,
            "source": {**source, "unit": unit_source,
                       "interval": ("entered" if choice.interval else
                                    "file" if scene.interval else
                                    "assumed")}}


def plan_levels(shape_zyx, voxel_zyx) -> list[tuple]:
    """The pyramid: per level (factors from the level before, shape, voxel
    size), all as (z, y, x). x and y halve each level until both are at
    most LEVEL_MIN_SIZE; z halves too once its voxels would otherwise be
    thinner than x's (so that voxels approach cubes, as in eubi-bridge's
    pyramids)."""
    shape, voxel = tuple(shape_zyx), tuple(float(v) for v in voxel_zyx)
    levels = [((1, 1, 1), shape, voxel)]
    while max(shape[1], shape[2]) > LEVEL_MIN_SIZE and len(levels) < MAX_LEVELS:
        fy, fx = (2 if shape[1] > 1 else 1), (2 if shape[2] > 1 else 1)
        fz = 2 if shape[0] > 1 and voxel[0] * 2 <= voxel[2] * fx else 1
        factors = (fz, fy, fx)
        shape = tuple(-(-n // f) for n, f in zip(shape, factors))
        voxel = tuple(v * f for v, f in zip(voxel, factors))
        levels.append((factors, shape, voxel))
    return levels


def prepare_scene(reader, scene: int) -> bool:
    """Let the reader finish a scene that is worked out only once it is
    chosen (a LIF tile scan, stitched: 0.17.2). True if the scene changed
    (its size and notes)."""
    prepare = getattr(reader, "prepare_scene", None)
    return bool(prepare(scene)) if prepare is not None else False


def preparing_text(reader, scene: int) -> str | None:
    """What preparing the scene will do, to show meanwhile; None if it is
    ready (see prepare_scene)."""
    text = getattr(reader, "preparing_text", None)
    return text(scene) if text is not None else None


def memory_needed(reader, choice: FileChoice) -> int:
    """Bytes the chosen part takes in memory, its smaller copies included."""
    prepare_scene(reader, choice.scene)
    scene = reader.scenes[choice.scene]
    _, sizes, picks = picks_of(scene, choice)
    calib = calibration_of(scene, choice)
    n_tc = (len(picks["T"]) if "T" in picks else 1) * (
        len(picks["C"]) if "C" in picks else 1)
    shape = tuple(len(picks[a]) if a in picks else 1 for a in "ZYX")
    voxel = tuple(calib["voxel"][k] for k in "zyx")
    extra = getattr(reader, "extra_memory", None)    # e.g. lazy layers
    return sum(n_tc * vc.nbytes(s, scene.dtype)
               for _, s, _ in plan_levels(shape, voxel)) + (
        extra(choice) if extra is not None else 0)


def free_memory() -> int | None:
    """Memory available now, in bytes (None if unknown)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def memory_problem(needed: int, free: int | None) -> tuple[str, bool]:
    """(text, refuse) about loading `needed` bytes: refuse is True above
    MEMORY_REFUSE of the free memory."""
    if free is None:
        return f"Needs about {vc.format_bytes(needed)} of memory.", False
    text = (f"Needs about {vc.format_bytes(needed)} of memory (with the "
            f"smaller copies for the views); {vc.format_bytes(free)} free.")
    if needed > MEMORY_REFUSE * free:
        return (text + " That is too much: load fewer channels, timepoints "
                "or slices, or convert the file to OME-Zarr (e.g. with "
                "eubi-bridge), which Volume Cropper reads piece by piece.",
                True)
    if needed > MEMORY_WARN * free:
        return (text + " That is more than half: other programs may slow "
                "down.", False)
    return text, False


def load(reader, choice: FileChoice,
         progress: Callable[[int, int, str], None] | None = None,
         cancel: Callable[[], bool] | None = None) -> vc.OmeZarrImage:
    """Read the chosen part of a file into memory and make its pyramid.

    Returns an OmeZarrImage whose levels are zarr arrays in memory, with
    axes (t), (c), z, y, x (z always: a 2D image gets one plane), and
    whose `source` records the file and what was read. Its coordinates are
    those of the whole file: voxel 0 of the part read sits at its place in
    the file (first index x voxel size).

    progress(done, total, stage): bytes done of the total (reading, then
    the smaller copies). cancel(): return True to stop (ReadCancelled).
    """
    prepare_scene(reader, choice.scene)      # e.g. stitch a tile scan
    scene = reader.scenes[choice.scene]
    axes, sizes, picks = picks_of(scene, choice)
    calib = calibration_of(scene, choice)
    letters = [a for a in "TC" if a in sizes] + ["Z", "Y", "X"]
    shape0 = [len(picks[a]) if a in picks else 1 for a in letters]
    voxel = calib["voxel"]
    levels_plan = plan_levels(shape0[-3:], [voxel[k] for k in "zyx"])
    n_tc = int(np.prod(shape0[:-3], dtype=np.int64)) if len(shape0) > 3 else 1
    total = sum(n_tc * vc.nbytes(s, scene.dtype) for _, s, _ in levels_plan)
    done = 0

    def step(nbytes: int, stage: str):
        nonlocal done
        done += nbytes
        if progress is not None:
            progress(done, total, stage)
        if cancel is not None and cancel():
            raise ReadCancelled(f"Reading {reader.path.name} was cancelled.")

    root = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    names = [a.lower() for a in letters]
    arrays = []
    for k, (_, shape_zyx, _) in enumerate(levels_plan):
        shape = shape0[:-3] + list(shape_zyx)
        chunks = [1] * (len(shape) - 3) + [
            min(CHUNK_Z, shape[-3]), min(CHUNK_YX, shape[-2]),
            min(CHUNK_YX, shape[-1])]
        arrays.append(root.create_array(
            name=str(k), shape=shape, dtype=scene.dtype, chunks=chunks,
            compressors=None, fill_value=0, dimension_names=names))

    # 1. The full resolution, a few planes (one z chunk) at a time.
    lead = [picks[a] if a == "C" else list(picks[a]) for a in letters[:-3]]
    z_pick = picks.get("Z", range(0, 1))
    stored = [a for a in axes if a in "ZYX"]      # left after indexing
    order = [stored.index(a) for a in "ZYX" if a in stored]
    for lead_index in np.ndindex(*[len(p) for p in lead]) if lead else [()]:
        fixed = {a: lead[i][j] for i, (a, j) in
                 enumerate(zip(letters[:-3], lead_index))}
        for z0 in range(0, len(z_pick), CHUNK_Z):
            z1 = min(z0 + CHUNK_Z, len(z_pick))
            key = []
            for letter in axes:
                if letter == "-":
                    key.append(0)
                elif letter in fixed:
                    key.append(fixed[letter])
                elif letter == "Z":
                    key.append(slice(z_pick.start + z0, z_pick.start + z1))
                else:                                      # Y or X
                    key.append(slice(picks[letter].start, picks[letter].stop))
            block = reader.read(choice.scene, tuple(key)).transpose(order)
            if "Z" not in stored:
                block = block[np.newaxis]
            arrays[0][tuple(lead_index) + (slice(z0, z1),)] = block
            step(block.nbytes, "reading")
    if hasattr(reader, "release"):     # e.g. a lazy layer's last chunk
        reader.release()

    # 2. The smaller copies, each from the one before.
    for k in range(1, len(levels_plan)):
        factors = levels_plan[k][0]
        src, dst = arrays[k - 1], arrays[k]
        fz = factors[0]
        for lead_index in np.ndindex(*dst.shape[:-3]) if lead else [()]:
            for z0 in range(0, dst.shape[-3], CHUNK_Z):
                z1 = min(z0 + CHUNK_Z, dst.shape[-3])
                block = src[tuple(lead_index)
                            + (slice(z0 * fz, min(z1 * fz, src.shape[-3])),)]
                small = mean_downsample(block, factors)
                dst[tuple(lead_index) + (slice(z0, z1),)] = small
                step(small.nbytes, "making smaller copies for the views")

    # 3. The OME metadata, as an OME-Zarr 0.5 image would have it.
    unit, t_unit = calib["unit"], calib["time_unit"]
    axes_meta = []
    for letter in letters:
        if letter == "T":
            axes_meta.append({"name": "t", "type": "time",
                              **({"unit": t_unit} if t_unit else {})})
        elif letter == "C":
            axes_meta.append({"name": "c", "type": "channel"})
        else:
            axes_meta.append({"name": letter.lower(), "type": "space",
                              "unit": unit})
    step_of = {"T": calib["interval"], "C": 1.0}
    # Index (in voxels) of the first voxel read, from the file's origin.
    first_of = {a: (picks[a].start if a in "TZYX" and a in picks else 0)
                + scene.origin.get(a.lower(), 0.0) for a in letters}
    datasets, levels = [], []
    scale = [step_of.get(a) or voxel[a.lower()] for a in letters]
    translation = [first_of[a] * s if a != "C" else 0.0
                   for a, s in zip(letters, scale)]
    for k, (factors, _, _) in enumerate(levels_plan):
        if k > 0:
            # The mean of f voxels sits at their middle: (f - 1) / 2 of the
            # finer voxels further on.
            full = [1] * (len(letters) - 3) + list(factors)
            translation = [t + (f - 1) / 2 * s
                           for t, f, s in zip(translation, full, scale)]
            scale = [s * f for s, f in zip(scale, full)]
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)},
            {"type": "translation", "translation": list(translation)}]})
        levels.append(vc.Level(str(k), arrays[k], list(scale),
                               list(translation)))
    n_c = sizes.get("C", 1)
    channels = picks.get("C", [0])
    omero = []
    for j, c in enumerate(channels):
        name = (scene.channel_names[c] if c < len(scene.channel_names)
                and scene.channel_names[c] else f"channel {c}")
        color = (scene.channel_colors[c] if c < len(scene.channel_colors)
                 and scene.channel_colors[c] else
                 "FFFFFF" if n_c == 1 else
                 DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
        entry = {"label": name, "color": color, "active": True}
        window = (scene.channel_windows[c]
                  if c < len(scene.channel_windows) else None)
        if window is not None:
            entry["window"] = {"start": window[0], "end": window[1],
                               "min": window[0], "max": window[1]}
        omero.append(entry)
    record = {
        "file": str(reader.path),
        "format": reader.format_name,
        "reader": reader.reader_name,
        "scene": scene.name,
        "scene_index": choice.scene,
        "scenes_in_file": len(reader.scenes),
        "axes_in_file": scene.axes,
        "axes": axes,
        "shape_in_file": list(scene.shape),
        "loaded": {a.lower(): ([picks[a].start, picks[a].stop]
                               if a != "C" else list(picks[a]))
                   for a in letters if a in picks},
        "has_z": "Z" in sizes,
        "origin_in_file_voxels": dict(scene.origin),
        "voxel_size": voxel,
        "unit": unit,
        "frame_interval": calib["interval"] if "T" in sizes else None,
        "time_unit": t_unit,
        "calibration_from": calib["source"],
        "compression_in_file": reader.compression,
        "output": reader.output_kind,
        "pyramid": ("made in memory by Volume Cropper: each level the mean "
                    "of 2 x 2 voxels in y and x (2 x 2 x 2 where z halves "
                    "too)"),
        "python": platform.python_version(),
    }
    stitching = getattr(reader, "scene_record", None)
    stitching = stitching(choice.scene) if stitching is not None else None
    if stitching:                       # a stitched tile scan (0.17.2)
        record["stitching"] = stitching
    attrs = {"ome": {"version": "0.5", "multiscales": [{
        "name": scene.name, "axes": axes_meta, "datasets": datasets}],
        "omero": {"channels": omero}}}
    root.attrs.update(attrs)
    # A TIFF that is itself a crop: its record, so that a crop of it keeps
    # the chain back to the original image (as for OME-Zarr crops).
    if getattr(reader, "record", None):
        attrs["crop_provenance"] = reader.record
    return vc.OmeZarrImage(Path(reader.path), attrs, axes_meta, levels,
                           source=record)


def mean_downsample(block: np.ndarray, factors) -> np.ndarray:
    """Shrink the last three axes (z, y, x) by whole factors, each new voxel
    the mean of the voxels it replaces (integer data: rounded, halves up).
    Odd sizes: the last voxel is repeated to fill the last group.

    The voxels of each group are added up as shifted views of the block
    (for factors 2: every other voxel, starting at 0 or 1), in a wider
    integer type, which is several times faster than numpy's mean over a
    reshaped array (measured: 0.2 s instead of 1.1 s for 160 MB)."""
    fz, fy, fx = factors
    nz, ny, nx = block.shape[-3:]
    pad = [(0, 0)] * (block.ndim - 3) + [(0, (-n) % f) for n, f in
                                         zip((nz, ny, nx), factors)]
    if any(p[1] for p in pad):
        block = np.pad(block, pad, mode="edge")
    integer = np.issubdtype(block.dtype, np.integer)
    wide = (np.int64 if integer and block.dtype.itemsize >= 4 else
            np.int32 if integer else np.float64)
    total = None
    for dz in range(fz):
        for dy in range(fy):
            for dx in range(fx):
                part = block[..., dz::fz, dy::fy, dx::fx]
                if total is None:
                    total = part.astype(wide)
                else:
                    np.add(total, part, out=total)
    n = fz * fy * fx
    if integer:
        return ((total + n // 2) // n).astype(block.dtype)
    return (total / n).astype(block.dtype)


def describe(reader) -> str:
    """What the file holds, a few lines per image (for the command line)."""
    lines = [f"{reader.path}", f"  format: {reader.format_name}"
             + (f", compressed ({reader.compression})"
                if reader.compression != "none" else "")]
    for k, scene in enumerate(reader.scenes):
        unit = vc.short_unit(scene.unit) if scene.unit else "?"
        voxel = " x ".join("?" if scene.voxel[a] is None else f"{scene.voxel[a]:g}"
                           for a in "zyx")
        lines.append(f"  image {k}: {scene.name}: axes {scene.axes}, shape "
                     f"{scene.shape}, {scene.dtype}, "
                     f"{vc.format_bytes(scene.nbytes)}; voxel (z y x) "
                     f"{voxel} {unit}"
                     + (f"; {scene.interval:g} {scene.time_unit or ''} "
                        "between timepoints" if scene.interval else ""))
        lines += [f"    {note}" for note in scene.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _our_axes(tiff_axes: str, shape) -> tuple[str, list]:
    """tifffile's axis letters as ours (see Scene.axes), with notes."""
    letters, notes = [], []
    for letter, n in zip(tiff_axes, shape):
        if letter in "TCZYX":
            letters.append(letter)
        elif letter == "S":
            if n == 1:
                letters.append("-")
            elif "C" not in tiff_axes:
                letters.append("C")         # colour (RGB): as 3 channels
                notes.append("A colour (RGB) image: its colours are read "
                             "as channels.")
            else:
                raise NotImplementedError(
                    "Colour (RGB) images with several channels are not "
                    "supported.")
        elif n == 1:
            letters.append("-")
        else:
            letters.append("?")
    if "?" in letters:
        notes.append("The file does not say what "
                      + ("one axis is" if letters.count("?") == 1
                         else "some axes are")
                      + " (z, t or c): please choose.")
    return "".join(letters), notes


def _size(axes: str, shape, letter: str) -> int:
    return shape[axes.index(letter)] if letter in axes else 1


def _positive(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _resolution(page) -> tuple[float | None, float | None]:
    """Pixel width and height from the TIFF resolution tags (pixels per
    unit, as ImageJ stores them)."""
    sizes = []
    for name in ("XResolution", "YResolution"):
        tag = page.tags.get(name)
        try:
            num, den = tag.value
            sizes.append(_positive(den / num))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            sizes.append(None)
    return sizes[0], sizes[1] or sizes[0]


def _lut_color(lut) -> str | None:
    """An ImageJ LUT's colour (its brightest entry), e.g. "00FF00"."""
    try:
        lut = np.asarray(lut)
        rgb = lut[:, -1] if lut.shape[0] == 3 else lut[-1, :3]
        if lut.dtype.itemsize > 1:                 # 16-bit LUT: high byte
            rgb = np.asarray(rgb) >> 8
        return "".join(f"{int(v):02X}" for v in rgb[:3])
    except Exception:
        return None


def _ome_color(value) -> str | None:
    """An OME colour (a signed 32-bit RGBA integer) as "RRGGBB"."""
    try:
        value = int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    return f"{value >> 24 & 255:02X}{value >> 16 & 255:02X}{value >> 8 & 255:02X}"


def _natural_key(name: str) -> list:
    """Sort key that orders numbers in names by value (img2 < img10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]
