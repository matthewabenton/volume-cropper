"""Volume Cropper: crop sub-volumes out of an OME-Zarr image and save them
as OME-Zarr.

Written for OME-Zarr 0.5 (Zarr v3) and 0.4 (Zarr v2) images, such as those
made by eubi-bridge (0.4 since version 0.8.0), and since 0.21.0 OME-Zarr
0.6 (coordinate systems, RFC-5) and zipped OME-Zarr (RFC-9 .ozx files and
other zips; their crops are zipped too). The crop keeps the source's
storage format, including its OME-Zarr version: data type, chunk and shard
sizes, compression and dimension names are copied level by level.

How it works:

1. The crop box is given in physical units (µm for z/y/x), so it does not
   depend on the pyramid level. Axes left out of the box are kept whole, and
   channels are picked by index.
2. The box is converted to voxel indices on level 0 and widened slightly, so
   that its edges fall on the voxel grid of the coarsest level kept. Every
   kept pyramid level can then be cut straight out of the source. No
   downsampling is recomputed, and all levels stay exactly aligned.
3. Data are copied in parallel, one whole output *shard* per task. A shard is
   one file holding many chunks, so two tasks must never write into the same
   shard at the same time. That race is why a plain
   ``dask_array[crop].to_zarr(...)`` is unsafe for sharded arrays. Without
   shards (Zarr v2, OME-Zarr 0.4) every chunk is a file of its own, and a
   task copies a block of whole chunks (about BLOCK_BYTES).
4. The OME metadata are copied, and a translation is added so that the crop
   keeps its position relative to the full image (napari, BigDataViewer).
   Provenance (source, box, software versions) is stored under
   ``crop_provenance``; a crop of a crop also keeps its source's record
   (``source_provenance``). The OME metadata are written last, so an interrupted
   crop is easy to recognise: it has no ``ome`` attributes (0.4: no
   ``multiscales``), and its ``crop_provenance`` says ``"complete": false``.
5. Optionally a box has outlines: polygons (e.g. drawn with a lasso) on
   slices of one or more views (pairs of axes, such as y-x). Together they
   define one smooth 3D shape that passes through all of them and ends at
   the outermost ones (since 0.7.0, see outline_mask); a margin can grow or
   shrink it. A box's only polygon reaches through the whole box. Voxels
   outside the shape are written as 0, and blocks entirely outside are not
   written at all (zarr reads missing chunks as 0), so the crop is still an
   axis-aligned box but takes less space on disk.

Coordinate convention (the same as OME-Zarr and napari): voxel ``i`` has its
centre at ``translation + i * scale`` and covers half a voxel either side.

It can be used from a notebook, from the napari GUI (crop_gui.py) or from the
command line, from any folder::

    python <tool folder>/volume_cropper/core.py SOURCE.zarr boxes.csv OUTPUT_FOLDER

(or ``python -m volume_cropper.core ...`` with the tool folder on the path).

Until version 0.2.0 (2026-10-23) this module was called omezarr_crop.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime
import errno
import functools
import io
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import zarr
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (binary_dilation, binary_fill_holes,
                           distance_transform_edt, gaussian_filter, label,
                           map_coordinates)
from skimage.draw import polygon2mask
from skimage.measure import find_contours, points_in_poly

__version__ = "0.23.0"

# Tolerance for floating-point noise when converting physical units to voxel
# indices (for example, 2.032 / 0.254 = 8.000000000000002).
_EPS = 1e-6

# OME-Zarr 0.4 keeps these keys directly in the image group's attributes;
# 0.5 and 0.6 keep them under "ome".
OME_04_KEYS = ("multiscales", "omero")

# The OME-Zarr versions that Volume Cropper opens. 0.4 is Zarr v2; 0.5 and
# 0.6 (0.21.0) are Zarr v3. 0.6 keeps the axes in named "coordinate
# systems" and links every level to one of them (RFC-5).
OME_ZARR_VERSIONS = ("0.4", "0.5", "0.6")
ZARR_V3_VERSIONS = ("0.5", "0.6")


def is_v06(version: str) -> bool:
    """True for OME-Zarr 0.6, also as its pre-releases name it ("0.6.dev4",
    "0.6rc0"): ngff_zarr 0.43 (2026-11) writes "0.6.dev4" when asked for
    0.6, with the 0.6 layout. Such images are read with the 0.6 rules;
    their crops keep the source's version as written."""
    return bool(re.fullmatch(r"0\.6(\.dev\d+|rc\d+)?", str(version)))

# Without shards (Zarr v2), the size of the block of whole chunks that one
# task copies (uncompressed; see _copy_block).
BLOCK_BYTES = 32 * 2**20

# Short unit names, used in the CSV column headers (z_start_um, ...).
_UNIT_ABBREVIATIONS = {
    "micrometer": "um",
    "nanometer": "nm",
    "millimeter": "mm",
    "second": "s",
    "millisecond": "ms",
    "minute": "min",
    "hour": "h",
}


class CropCancelled(Exception):
    """Raised when a running crop is cancelled (e.g. from the GUI)."""


# ---------------------------------------------------------------------------
# Reading the source image
# ---------------------------------------------------------------------------

@dataclass
class Level:
    """One resolution level of a multiscale image."""

    path: str                  # name of the array inside the group, e.g. "0"
    array: zarr.Array
    scale: list[float]         # physical size of one voxel, per axis
    translation: list[float]   # physical position of voxel 0, per axis


@dataclass
class OmeZarrImage:
    """An opened OME-Zarr image: metadata plus one Level per pyramid level.

    Since 0.12.0 it can also be an image read from another file (TIFF ...)
    into memory (file_images.load): its levels are then zarr arrays in
    memory, and `source` records the file and what was read of it. Such
    images are cropped into TIFF files (crop_box).
    """

    path: Path
    attrs: dict                # all root attributes (0.5: contains "ome")
    axes: list[dict]           # OME axes: name, type and (optionally) unit
    levels: list[Level]
    source: dict | None = None    # read from a file into memory (0.12.0)
    # (0.21.0) A zipped OME-Zarr: "ozx" (an RFC-9 file) or "zip" (another
    # zip file, e.g. .ome.zarr.zip), and its open zip store (see close).
    container: str | None = None
    store: object = field(default=None, repr=False, compare=False)
    # (0.21.0) An image in a collection of images (a bioformats2raw
    # layout, an OME-Zarr 0.6 scene): its path in the collection (e.g.
    # "0"), and the kind of collection. `path` is the collection's.
    inner: str = ""
    collection: str | None = None

    def close(self) -> None:
        """Close the file of a zipped image (nothing to do for a folder).
        On Windows an open zip file cannot be replaced or deleted."""
        if self.store is not None:
            self.store.close()
            self.store = None

    @property
    def in_memory(self) -> bool:
        """True for an image read from a file into memory (0.12.0)."""
        return self.source is not None

    @property
    def zarr_format(self) -> int:
        """2 for OME-Zarr 0.4 (Zarr v2), 3 for OME-Zarr 0.5 and 0.6
        (Zarr v3)."""
        return self.levels[0].array.metadata.zarr_format

    @property
    def ome_version(self) -> str:
        """The OME-Zarr version, e.g. "0.4", "0.5" or "0.6"."""
        if self.zarr_format == 2:
            return "0.4"
        return str(self.attrs["ome"].get("version", "0.5"))

    @property
    def ome_format(self) -> str:
        """E.g. "OME-Zarr 0.4 (Zarr v2)": the version and the Zarr format,
        whether the image is a folder or zipped (a crop must keep these)."""
        return f"OME-Zarr {self.ome_version} (Zarr v{self.zarr_format})"

    @property
    def format_name(self) -> str:
        """E.g. "OME-Zarr 0.4 (Zarr v2)", "OME-Zarr 0.5 (Zarr v3), zipped
        (.ozx)", or for a file read into memory e.g. "OME-TIFF (in
        memory)"."""
        if self.source is not None:
            return f"{self.source['format']} (in memory)"
        return self.ome_format + {"ozx": ", zipped (.ozx)",
                                  "zip": ", zipped"}.get(self.container, "")

    @property
    def kind_name(self) -> str:
        """Short: "OME-Zarr 0.5", or the file's format, e.g. "OME-TIFF"."""
        if self.source is not None:
            return self.source["format"]
        return f"OME-Zarr {self.ome_version}"

    @property
    def coordinate_system(self) -> str | None:
        """OME-Zarr 0.6: the name of the image's own ("intrinsic")
        coordinate system, the one every level's voxel size and position
        lead to; the tool shows and crops the image in it, as the 0.6
        specification recommends. None before 0.6."""
        if self.in_memory or not is_v06(self.ome_version):
            return None
        return _intrinsic_system(self.ome["multiscales"][0])

    @property
    def ome(self) -> dict:
        """The OME metadata ("multiscales", "omero"): in 0.5 the "ome"
        attribute, in 0.4 these keys of the attributes themselves."""
        if self.zarr_format == 2:
            return {k: self.attrs[k] for k in OME_04_KEYS if k in self.attrs}
        return self.attrs["ome"]

    @property
    def axis_names(self) -> list[str]:
        return [axis["name"] for axis in self.axes]

    @property
    def channel_axis(self) -> int | None:
        """Index of the channel axis, or None if the image has none."""
        for i, axis in enumerate(self.axes):
            if axis.get("type") == "channel":
                return i
        return None

    @property
    def range_axes(self) -> list[str]:
        """Axes that a crop box can restrict (all except the channel axis)."""
        return [axis["name"] for axis in self.axes
                if axis.get("type") != "channel"]

    @property
    def n_channels(self) -> int:
        c = self.channel_axis
        return 1 if c is None else self.levels[0].array.shape[c]

    @property
    def channel_labels(self) -> list[str]:
        """Channel names from the 'omero' metadata (index if missing)."""
        if self.channel_axis is None:
            return []
        channels = self.ome.get("omero", {}).get("channels", [])
        labels = [ch.get("label", "") for ch in channels]
        if len(labels) != self.n_channels:
            labels = [""] * self.n_channels
        return [label or f"channel {i}" for i, label in enumerate(labels)]

    def unit(self, axis: str) -> str:
        return self.axes[self.axis_names.index(axis)].get("unit", "")

    def extent(self, axis: str) -> tuple[float, float]:
        """Physical range covered by the whole image along one axis."""
        i = self.axis_names.index(axis)
        level = self.levels[0]
        return index_range_to_physical(0, level.array.shape[i],
                                       level.scale[i], level.translation[i])

    def describe(self) -> str:
        """A short human-readable summary of the image."""
        lines = [f"{self.path}", f"  format: {self.format_name}"]
        lines.append("  axes: " + ", ".join(
            f"{a['name']} ({a.get('unit', a.get('type', ''))})"
            for a in self.axes))
        for level in self.levels:
            arr = level.array
            shards = f"shards {arr.shards}, " if arr.shards else ""
            lines.append(
                f"  level {level.path}: shape {arr.shape}, voxel "
                f"{_fmt_list(level.scale)}, chunks {arr.chunks}, "
                f"{shards}{format_bytes(nbytes(arr.shape, arr.dtype))}")
        if self.channel_axis is not None:
            lines.append("  channels: " + ", ".join(
                f"{i}={label}" for i, label in enumerate(self.channel_labels)))
        return "\n".join(lines)


def open_image(path: str | Path, inner: str | None = None) -> OmeZarrImage:
    """Open an OME-Zarr 0.6 or 0.5 (Zarr v3) or 0.4 (Zarr v2) image
    read-only.

    Only the first multiscale image in the group is used. Like opening a
    virtual stack in Fiji, this reads the metadata only. Pixel data are read
    later, and only for the region that is needed. Images it cannot handle
    are refused with a message that says why.

    Since 0.21.0 also a zipped OME-Zarr: an RFC-9 .ozx file, or another
    .zip of one (e.g. .ome.zarr.zip; the image may be in a folder inside
    the zip). It stays zipped: parts are read from the file as needed.

    Also since 0.21.0, one image of a collection of images (a
    bioformats2raw layout, an OME-Zarr 0.6 scene; folder or zipped):
    inner names it (its path in the collection, e.g. "0"; see
    collection_images). A collection of one image opens that image; with
    several and no inner, ImageChoiceNeeded lists them.
    """
    path = Path(path)
    if is_zipped(path):
        return _open_zipped(path, inner)
    return _open_in_group(zarr.open_group(path, mode="r"), path, inner)


class ImageChoiceNeeded(ValueError):
    """Raised by open_image for a collection of several images when none
    was chosen. images: [(path in the collection, label)]; kind: e.g.
    "bioformats2raw layout"."""

    def __init__(self, path: Path, kind: str, images: list[tuple[str, str]]):
        self.path, self.kind, self.images = Path(path), kind, images
        super().__init__(
            f"{Path(path).name} is a collection of {len(images)} images "
            f"({kind}). Choose one: in the notebook or a script, "
            f"open_image(path, inner={images[0][0]!r}); on the command "
            f"line, --collection-image {images[0][0]}. Its images: "
            + ", ".join(label for _, label in images) + ".")


def collection_images(group: zarr.Group
                      ) -> tuple[str, list[tuple[str, str]]] | None:
    """For a group that holds images instead of being one (0.21.0): the
    kind of collection and its images as (path in it, label); None for any
    other group. Two kinds are read:

    - the bioformats2raw layout (OME-Zarr 0.4-0.6, "transitional"): the
      images listed as "series" in its OME group, or else in the numbered
      groups 0, 1, 2, ...;
    - an OME-Zarr 0.6 scene: the images its coordinate transformations
      link (their "path"), in the order they first appear.

    Only plain paths inside the collection are followed (as for levels:
    see _LEVEL_NAME). RFC-8 collections are not read yet (a draft; the
    user's decision 1a of 2026-11-22)."""
    attrs = group.attrs.asdict()
    ome = attrs.get("ome", {}) if group.metadata.zarr_format == 3 else attrs
    if not isinstance(ome, dict) or "multiscales" in ome:
        return None
    paths: list[str] = []
    if "bioformats2raw.layout" in ome:
        kind = "bioformats2raw layout"
        try:
            meta = group["OME"].attrs.asdict()
            series = meta.get("ome", meta).get("series")
        except (KeyError, AttributeError, TypeError):
            series = None
        if isinstance(series, list):
            paths = [p for p in series if isinstance(p, str)]
        else:
            while _is_group(group, str(len(paths))):
                paths.append(str(len(paths)))
    elif isinstance(ome.get("scene"), dict):
        kind = "OME-Zarr 0.6 scene"
        for transform in ome["scene"].get("coordinateTransformations", []):
            for end in ("input", "output"):
                place = (transform.get(end)
                         if isinstance(transform, dict) else None)
                where = place.get("path") if isinstance(place, dict) else None
                if isinstance(where, str) and where not in paths:
                    paths.append(where)
    else:
        return None
    images = []
    for inner in paths:
        if not _LEVEL_NAME.fullmatch(inner) or any(
                part in (".", "..") for part in inner.split("/")):
            continue
        try:
            sub = group[inner]
            sub_attrs = sub.attrs.asdict()
        except (KeyError, AttributeError):
            continue
        sub_ome = (sub_attrs.get("ome", {}) if sub.metadata.zarr_format == 3
                   else sub_attrs)
        multiscales = sub_ome.get("multiscales") if isinstance(
            sub_ome, dict) else None
        if not multiscales:
            continue
        name = multiscales[0].get("name")
        images.append((inner, f"{inner} ({name})"
                       if isinstance(name, str) and name and name != inner
                       else inner))
    return kind, images


def _is_group(group: zarr.Group, name: str) -> bool:
    """Whether a group has a group of that name in it."""
    try:
        return isinstance(group[name], zarr.Group)
    except KeyError:
        return False


def _open_in_group(group: zarr.Group, path: Path,
                   inner: str | None) -> OmeZarrImage:
    """open_image for an open group: an image, or one of a collection's."""
    found = collection_images(group)
    if found is None:
        if inner:
            raise ValueError(f"{path} is one image, not a collection of "
                             f"images: there is no image {inner!r} in it.")
        return _image_from_group(group, path)
    kind, images = found
    if not images:
        raise ValueError(f"{path} is a collection of images ({kind}), but "
                         "Volume Cropper finds no image in it.")
    if inner is None:
        if len(images) > 1:
            raise ImageChoiceNeeded(path, kind, images)
        inner = images[0][0]
    if inner not in [p for p, _ in images]:
        raise ValueError(
            f"{path} has no image {inner!r}; its images ({kind}): "
            + ", ".join(p for p, _ in images) + ".")
    image = _image_from_group(group[inner], path)
    image.inner, image.collection = inner, kind
    return image


# ---------------------------------------------------------------------------
# Zipped OME-Zarr (0.21.0)
# ---------------------------------------------------------------------------

# File endings of zipped OME-Zarr images: ".ozx" (RFC-9) and ".zip" (also
# ".ome.zarr.zip", as the BioImage Archive names them).
ZIP_SUFFIXES = (".ozx", ".zip")
# Zarr metadata files: v3 (zarr.json) and v2.
_ZARR_META = ("zarr.json", ".zgroup", ".zattrs", ".zarray", ".zmetadata")


def is_zipped(path: str | Path) -> bool:
    """True for a path that names a zipped OME-Zarr (.ozx, .zip, also
    .ome.zarr.zip): a file, or a file still to be written."""
    path = Path(path)
    return path.suffix.lower() in ZIP_SUFFIXES and not path.is_dir()


def _zip_root(path: Path, names: list[str]) -> str:
    """Where the image is inside a zip file: "" when its Zarr metadata are
    at the top (as RFC-9 requires), or the name of the one folder that
    holds them (as zipping a folder with Explorer or the Finder makes);
    otherwise an error that says what the zip holds."""
    tops = {"zarr.json", ".zgroup", ".zattrs"}
    if tops & set(names):
        return ""
    # (the Finder adds a "__MACOSX" folder of its own)
    folders = {name.split("/")[0] for name in names
               if "/" in name and not name.startswith("__MACOSX/")}
    if len(folders) == 1:
        folder = folders.pop()
        if any(f"{folder}/{meta}" in names for meta in tops):
            return folder
    inner = [name for name in names if name.lower().endswith(ZIP_SUFFIXES)]
    if inner:
        raise ValueError(
            f"{path.name} holds another zip file ({inner[0]}), not an "
            "OME-Zarr image. Unzip the outer file and open the one inside.")
    raise ValueError(
        f"{path.name} is a zip file, but it holds no OME-Zarr image: there "
        "are no Zarr metadata (zarr.json, or .zattrs for OME-Zarr 0.4) at "
        "its top or in a single folder in it.")


def _open_zip_group(path: Path):
    """Open a zipped OME-Zarr's top group read-only: (group, store,
    folder in the zip, kind "ozx" or "zip"). Close the store when done."""
    import zipfile

    from zarr.storage import ZipStore

    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            compressed = any(info.compress_type != zipfile.ZIP_STORED
                             for info in zf.infolist()
                             if not info.filename.endswith(_ZARR_META))
    except zipfile.BadZipFile as err:
        raise ValueError(
            f"{path.name} cannot be read as a zip file ({err}). It may be "
            "damaged, unfinished, or one part of a zip split into several "
            "files (RFC-9 does not allow that).") from err
    root = _zip_root(path, names)
    store = ZipStore(path, mode="r")
    try:
        group = zarr.open_group(store, path=root, mode="r")
    except Exception:
        store.close()
        raise
    if compressed:
        warnings.warn(
            f"{path.name} is a compressed zip file: Volume Cropper opens it, "
            "but reading it is slower than reading an uncompressed one (as "
            "RFC-9 recommends: the image data are compressed already).",
            stacklevel=3)
    kind = "ozx" if path.suffix.lower() == ".ozx" else "zip"
    return group, store, root, kind


def _open_zipped(path: Path, inner: str | None = None) -> OmeZarrImage:
    """open_image for a zipped OME-Zarr (see open_image)."""
    group, store, _, kind = _open_zip_group(path)
    try:
        image = _open_in_group(group, path, inner)
    except Exception:
        store.close()
        raise
    image.container, image.store = kind, store
    return image


def zip_record(path: str | Path) -> dict | None:
    """The crop record (crop_provenance) of a zipped crop, or None if the
    zip holds none (or cannot be read)."""
    try:
        group, store, _, _ = _open_zip_group(Path(path))
    except (OSError, ValueError):
        return None
    try:
        record = group.attrs.asdict().get("crop_provenance")
    finally:
        store.close()
    return record if isinstance(record, dict) else None


# Bytes copied at a time when packing a zip file.
ZIP_COPY_BYTES = 16 * 2**20


def _zip_entries(folder: Path) -> list[tuple[Path, str]]:
    """The files of an OME-Zarr folder as (file, name in the zip), in the
    order RFC-9 recommends: the Zarr metadata first, breadth-first (the
    top's zarr.json first, then the levels' ...), then the data files."""
    meta, data = [], []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            name = path.relative_to(folder).as_posix()
            (meta if filename in _ZARR_META else data).append((path, name))
    meta.sort(key=lambda entry: (entry[1].count("/"),
                                 entry[1].split("/")))
    return meta + data


def _zip_comment(folder: Path) -> bytes:
    """RFC-9's zip comment: the OME-Zarr version, and (Zarr v3) that the
    zarr.json files come first, breadth-first."""
    if (folder / "zarr.json").is_file():
        attrs = json.loads((folder / "zarr.json").read_text(
            encoding="utf-8")).get("attributes", {})
        version = str(attrs.get("ome", {}).get("version", ""))
        comment = {"ome": {"version": version, "zipFile": {
            "centralDirectory": {"jsonFirst": True}}}}
    else:                                   # OME-Zarr 0.4 (Zarr v2)
        comment = {"ome": {"version": "0.4"}}
    return json.dumps(comment).encode("utf-8")


def pack_zip(folder: str | Path, zip_path: str | Path, *,
             progress: Callable[[int, int], None] | None = None,
             cancel_event: threading.Event | None = None) -> Path:
    """Pack an OME-Zarr folder into one zip file, as RFC-9 describes an
    OME-Zarr zip file (.ozx): the folder's content at the top of the zip
    (no folder around it), the Zarr metadata first (the top's zarr.json
    as the very first entry), the files stored as they are (no zip
    compression: the image data are compressed already), ZIP64 for every
    entry (as ozx-tck, RFC-9's test kit, writes them) and the version in
    the zip's comment. A Zarr v2 folder (OME-Zarr 0.4) is packed the same
    way, as far as it applies.

    The zip is written as <name>.part and renamed when complete (an
    existing file of that name is replaced then; check before calling).
    progress: called as progress(done, total) in bytes. Returns zip_path.
    """
    import zipfile

    folder, zip_path = Path(folder), Path(zip_path)
    entries = _zip_entries(folder)
    total = sum(path.stat().st_size for path, _ in entries)
    part = zip_path.with_name(zip_path.name + ".part")
    done = 0
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            for path, name in entries:
                info = zipfile.ZipInfo.from_file(path, name)
                info.compress_type = zipfile.ZIP_STORED
                with open(path, "rb") as source, \
                        zf.open(info, "w", force_zip64=True) as target:
                    while block := source.read(ZIP_COPY_BYTES):
                        target.write(block)
                        done += len(block)
                        if progress is not None:
                            progress(done, total)
                        if cancel_event is not None and cancel_event.is_set():
                            raise CropCancelled(
                                f"Cancelled; {zip_path.name} was not saved.")
            zf.comment = _zip_comment(folder)
        _replace_zip(part, zip_path)
    except BaseException:
        with contextlib.suppress(OSError):
            part.unlink()
        raise
    return zip_path


def _replace_zip(part: Path, zip_path: Path) -> None:
    """Rename a finished <name>.part to its zip file's name (replacing an
    earlier one), with a clear message when Windows refuses because the
    old file is open (e.g. in a second window: "Open last crop")."""
    try:
        os.replace(part, zip_path)
    except PermissionError as err:
        raise PermissionError(
            f"{zip_path.name} could not be replaced: it is open, perhaps in "
            "another Volume Cropper or napari window (Windows does not "
            "replace an open file). Close it there and crop again; the old "
            "file is unchanged.") from err


def _zip_layout_problems(zip_path: Path, folder: Path) -> list[str]:
    """Check a zip file made by pack_zip against the folder it was packed
    from: the same files with the same sizes, the top's metadata first,
    nothing compressed, the RFC-9 comment."""
    import zipfile

    problems = []
    expected = {name: path.stat().st_size
                for path, name in _zip_entries(folder)}
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        found = {info.filename: info.file_size for info in infos}
        comment = zf.comment
    if found != expected:
        missing = sorted(set(expected) - set(found))
        wrong = sorted(name for name in set(found) & set(expected)
                       if found[name] != expected[name])
        problems.append(
            f"the zip's files differ from the crop's: {len(missing)} "
            f"missing, {len(set(found) - set(expected))} extra, "
            f"{len(wrong)} of another size")
    if infos and infos[0].filename not in ("zarr.json", ".zgroup",
                                           ".zattrs"):
        problems.append(f"the zip starts with {infos[0].filename}, not "
                        "with the image's metadata")
    if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
        problems.append("files in the zip are compressed")
    try:
        json.loads(comment)["ome"]["version"]
    except (ValueError, KeyError, TypeError):
        problems.append("the zip's comment does not name the OME-Zarr "
                        "version")
    return problems


def _rewrite_zip_record(zip_path: Path, record: dict) -> None:
    """Store a new crop record in a zipped crop (verify_crop called on it
    directly): a zip's files cannot be changed, so the zip is written
    again, with the top's metadata changed and everything else copied."""
    import zipfile

    part = zip_path.with_name(zip_path.name + ".part")
    try:
        with zipfile.ZipFile(zip_path) as old, \
                zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED,
                                allowZip64=True) as new:
            for info in old.infolist():
                # (a fresh entry: the old one's header fields belong to the
                # old file)
                fresh = zipfile.ZipInfo(info.filename, info.date_time)
                fresh.external_attr = info.external_attr
                with old.open(info) as source, \
                        new.open(fresh, "w", force_zip64=True) as target:
                    if info.filename == "zarr.json":
                        meta = json.loads(source.read())
                        meta.setdefault("attributes", {})[
                            "crop_provenance"] = record
                        target.write(json.dumps(meta, indent=2).encode())
                    elif info.filename == ".zattrs":
                        attrs = json.loads(source.read())
                        attrs["crop_provenance"] = record
                        target.write(json.dumps(attrs, indent=2).encode())
                    else:
                        shutil.copyfileobj(source, target, ZIP_COPY_BYTES)
            new.comment = old.comment
        _replace_zip(part, zip_path)
    except BaseException:
        with contextlib.suppress(OSError):
            part.unlink()
        raise


# A level's name in the OME metadata is its folder inside the image, and the
# crop's level gets the same name. Only plain names are accepted (eubi-bridge
# and bioformats2raw write "0", "1", ...): a crafted image could name a level
# "C:/..." or "C:0", and it would then be read from outside the image and its
# crop written outside the output folder (found in the 0.17.0 review).
_LEVEL_NAME = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*")


def _image_from_group(group: zarr.Group, path: Path) -> OmeZarrImage:
    """The OmeZarrImage of an open zarr group (on disk, or in memory: a
    crop of a file's image, see crop_box); `path` is used in messages."""
    attrs = group.attrs.asdict()
    ome = _ome_metadata(path, group.metadata.zarr_format, attrs)
    version = ("0.4" if group.metadata.zarr_format == 2
               else str(ome.get("version", "0.5")))
    multiscale = ome["multiscales"][0]
    if is_v06(version):
        # 0.6: the axes belong to a named coordinate system. Multiscale-wide
        # transformations link it to other systems (e.g. a registration);
        # the image is shown and cropped in its own system, and the crop
        # keeps the links (see _crop_extra_transforms).
        axes = _coordinate_system(path, multiscale)["axes"]
    else:
        if multiscale.get("coordinateTransformations"):
            raise NotImplementedError(
                "Multiscale-wide coordinateTransformations are not "
                "supported.")
        if not isinstance(multiscale.get("axes"), list):
            raise ValueError(f"{path}: the OME-Zarr {version} metadata have "
                             "no axes.")
        axes = multiscale["axes"]
    for axis in axes:
        _check_axis_text(path, axis)
    untyped = [axis.get("name", "?") for axis in axes if not axis.get("type")]
    if untyped:
        # The type is only recommended (SHOULD) by OME-Zarr, but the tool
        # tells space, time and channel axes apart by it; guessing from the
        # names could mix them up without anyone noticing.
        raise ValueError(
            f"{path}: the axes {', '.join(map(repr, untyped))} have no type "
            "('space', 'time' or 'channel') in the OME metadata. Volume "
            "Cropper needs it to tell the axes apart.")
    levels = []
    for dataset in multiscale["datasets"]:
        name = dataset.get("path")
        if not isinstance(name, str) or not _LEVEL_NAME.fullmatch(name) \
                or any(part in (".", "..") for part in name.split("/")):
            raise ValueError(
                f"{path}: a level is called {name!r}. Volume Cropper only "
                "opens levels with plain names inside the image (such as "
                "'0', '1', ...).")
        where = f"{path}: level {name}"
        transforms = dataset.get("coordinateTransformations")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError(f"{where} has no coordinateTransformations "
                             "(voxel size) in the OME metadata.")
        scale, translation, has_scale = _compose_transforms(
            where, transforms, len(axes))
        if not has_scale and not is_v06(version):
            # (0.6 also allows "identity": a voxel size of 1)
            raise ValueError(f"{where} has no scale (voxel size) in the OME "
                             "metadata.")
        levels.append(Level(dataset["path"], group[dataset["path"]],
                            scale, translation))
    return OmeZarrImage(path, attrs, axes, levels)


# (0.23.0, the pre-publication review) Axis names and units come from the
# image file and appear in the window (also in napari's own labels, which
# take text that looks like HTML for HTML: an <img> tag would make Qt read
# the file it names, even on another computer) and as column names in the
# CSV files the tool writes (a name starting with = + - @ would be a formula
# in Excel). Real ones are plain: "t", "c", "z", "y", "x"; "micrometer".
_AXIS_TEXT_REFUSED = re.compile(r'[<>&"\x00-\x1f\x7f]')
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _check_axis_text(path, axis) -> None:
    """Refuse an axis whose name or unit could be taken for HTML or for a
    spreadsheet formula (see above)."""
    name, unit = axis.get("name"), axis.get("unit")
    if not isinstance(name, str) or not name \
            or _AXIS_TEXT_REFUSED.search(name) \
            or name.startswith(_FORMULA_START) \
            or (unit is not None and (not isinstance(unit, str)
                                      or _AXIS_TEXT_REFUSED.search(unit))):
        raise ValueError(
            f"{path}: an axis is called {name!r} (unit {unit!r}). Volume "
            "Cropper only opens images whose axis names and units hold no "
            "< > & \" or control characters, and whose axis names do not "
            "start with = + - or @ (they appear in the window and as column "
            "names in the regions file).")


def _csv_text(text: str) -> str:
    """A text cell for the CSV files the tool writes (0.23.0): one that
    starts like a spreadsheet formula (= + - @, tab, return) gets a leading
    ' so that Excel shows it as text (OWASP's advice against CSV
    injection); load_boxes_csv takes that ' away again."""
    return "'" + text if text.startswith(_FORMULA_START) else text


def _from_csv_text(text: str) -> str:
    """The text of a cell written by _csv_text."""
    return text[1:] if text.startswith("'") \
        and text[1:].startswith(_FORMULA_START) else text


def _compose_transforms(where: str, transforms: list, n_axes: int
                        ) -> tuple[list[float], list[float], bool]:
    """One level's voxel size and position from its coordinate
    transformations, applied in order: (scale, translation, has_scale).

    A voxel's index i becomes the position scale * i + translation. A
    "scale" multiplies both, a "translation" adds to the position,
    "identity" changes nothing, and a "sequence" (OME-Zarr 0.6) applies its
    own list in order. In 0.4 and 0.5 a level has a scale, then perhaps a
    translation; in 0.6 one scale or identity, or a sequence of a scale and
    a translation. Other kinds (affine, rotation, ...) are refused: the
    crop must stay an axis-aligned block of whole voxels."""
    scale, translation = [1.0] * n_axes, [0.0] * n_axes
    has_scale = False
    for transform in transforms:
        kind = transform.get("type") if isinstance(transform, dict) else None
        if kind == "sequence":
            inner = transform.get("transformations")
            if not isinstance(inner, list):
                raise ValueError(f"{where}: a 'sequence' transformation has "
                                 "no list of transformations.")
            s, t, inner_scale = _compose_transforms(where, inner, n_axes)
            # this step after the earlier ones: x -> s * x + t
            scale = [a * b for a, b in zip(s, scale)]
            translation = [a * b + c for a, b, c in zip(s, translation, t)]
            has_scale = has_scale or inner_scale
            continue
        if kind == "identity":
            continue
        if kind not in ("scale", "translation"):
            raise NotImplementedError(
                f"{where} uses a {kind!r} transformation. Volume Cropper "
                "only handles scale and translation (in 0.6 also identity "
                "and sequences of these).")
        if kind not in transform:
            raise NotImplementedError(
                f"{where} stores its {kind} in a separate array ('path'), "
                "which Volume Cropper does not support.")
        values = [float(v) for v in transform[kind]]
        if len(values) != n_axes:
            raise ValueError(f"{where}: its {kind} has {len(values)} values "
                             f"for {n_axes} axes.")
        if kind == "scale":
            scale = [a * b for a, b in zip(values, scale)]
            translation = [a * b for a, b in zip(values, translation)]
            has_scale = True
        else:
            translation = [a + b for a, b in zip(values, translation)]
    return scale, translation, has_scale


def _system_name(end) -> str | None:
    """The coordinate system named by a transformation's input or output
    (0.6: {"name": ..., "path": ...}; the 0.6 pre-releases also allowed a
    plain string)."""
    if isinstance(end, dict):
        return end.get("name")
    return end if isinstance(end, str) else None


def _intrinsic_system(multiscale: dict) -> str | None:
    """OME-Zarr 0.6: the name of the image's own ("intrinsic") coordinate
    system, which the transformations of all levels lead to (their
    "output"), or None if they do not name one and the same system."""
    names = {_system_name(transform.get("output"))
             for dataset in multiscale.get("datasets", [])
             for transform in dataset.get("coordinateTransformations", [])
             if isinstance(transform, dict)}
    return names.pop() if len(names) == 1 else None


def _coordinate_system(path: Path, multiscale: dict) -> dict:
    """OME-Zarr 0.6: the image's own coordinate system ({"name", "axes"}),
    or an error that says what is wrong."""
    name = _intrinsic_system(multiscale)
    if name is None:
        raise ValueError(
            f"{path}: the levels of this OME-Zarr 0.6 image do not all lead "
            "to one coordinate system (each level's transformation must "
            "name the same 'output').")
    systems = [system for system in multiscale.get("coordinateSystems", [])
               if isinstance(system, dict) and system.get("name") == name]
    if not systems or not isinstance(systems[0].get("axes"), list):
        raise ValueError(
            f"{path}: the levels lead to the coordinate system {name!r}, "
            "but the OME metadata do not define it (with its axes) under "
            "'coordinateSystems'.")
    return systems[0]


def _ome_metadata(path: Path, zarr_format: int, attrs: dict) -> dict:
    """The OME metadata of an image group (see OmeZarrImage.ome), or an
    error that says why the group is not an image Volume Cropper can open."""
    ome = attrs.get("ome", {}) if zarr_format == 3 else attrs
    if "multiscales" in ome:
        if zarr_format == 3:
            version = str(ome.get("version", "0.5"))
            if version not in ZARR_V3_VERSIONS and not is_v06(version):
                raise ValueError(
                    f"{path} is OME-Zarr {version}. Volume Cropper opens "
                    f"OME-Zarr {', '.join(OME_ZARR_VERSIONS[:-1])} and "
                    f"{OME_ZARR_VERSIONS[-1]}.")
            return ome
        _check_older_than_04(path, ome["multiscales"][0])
        return {k: attrs[k] for k in OME_04_KEYS if k in attrs}
    if "bioformats2raw.layout" in ome:
        raise ValueError(
            f"{path} is a collection of images (bioformats2raw layout): the "
            "images are in its numbered subfolders. Open one of them, e.g. "
            f"{path / '0'}.")
    record = attrs.get("crop_provenance")
    if isinstance(record, dict) and record.get("complete") is False:
        raise ValueError(
            f"{path} is an unfinished crop: cropping it was interrupted. To "
            "replace it, crop that region again with 'Overwrite earlier "
            "crops' (--overwrite on the command line).")
    raise ValueError(
        f"{path} is not an OME-Zarr image: its Zarr v{zarr_format} group has "
        "no 'multiscales' metadata. Volume Cropper opens OME-Zarr 0.6 and "
        "0.5 (Zarr v3) and 0.4 (Zarr v2) images.")


def _check_older_than_04(path: Path, multiscale: dict) -> None:
    """Refuse OME-Zarr 0.1-0.3, which are Zarr v2 like 0.4 but have no axis
    types (0.3) or no axes and per-level transformations (0.1, 0.2). In 0.4
    the "version" is only recommended, so the structure is checked too."""
    version = str(multiscale.get("version", ""))
    axes = multiscale.get("axes")
    datasets = multiscale.get("datasets", [])
    if version in ("0.1", "0.2", "0.3"):
        problem = f"has OME-Zarr {version} metadata"
    elif (not axes or not all(isinstance(axis, dict) for axis in axes)
          or not datasets
          or not all("coordinateTransformations" in d for d in datasets)):
        problem = ("has incomplete OME-Zarr metadata (no axes or no "
                   "per-level coordinateTransformations, as before 0.4)")
    else:
        return
    raise ValueError(
        f"{path} {problem}. Volume Cropper opens OME-Zarr 0.4, 0.5 and 0.6; "
        "convert the image first, e.g. with eubi-bridge.")


# ---------------------------------------------------------------------------
# Crop boxes and the crop plan
# ---------------------------------------------------------------------------

@dataclass
class CropBox:
    """A region to crop, in physical units.

    ranges:   {axis name: (start, stop)}, e.g. {"z": (350, 850)} in µm.
              Axes that are left out are kept whole.
    channels: source channel indices to keep (None = all channels).
    outlines: optional polygons that shape the crop, per view (a pair of
              spatial axes) and per slice of that view, in µm:
                  {("y", "x"): {10.0: [(y0, x0), (y1, x1), ...],
                                30.0: [...]}}
              here two y-x polygons, on the slices z = 10 and z = 30 µm (the
              key is the position along the view's third, "depth", axis).
              All polygons on slices, in all views, define one smooth 3D
              shape through all of them, which ends at the outermost ones
              (see outline_mask); voxels outside it are written as 0. A
              box's only polygon reaches through the whole box. The crop is
              still an axis-aligned box. A polygon drawn on a projection
              (a silhouette) has a depth range instead of a slice, e.g.
              {(0.0, 800.0): [...]}: it reaches through that range, and a
              voxel there must lie inside it. A plain list of points is read
              as one polygon without a slice (key None), as before 0.7.0: it
              reaches through the box.
    margin:   grow (> 0) or shrink (< 0) the outlined shape by this many
              voxels, in the image's finest voxel size (see pixel_size).
    t_step:   keep every n-th timepoint of the t range (1 = all), starting
              with its first; the crop's frame interval is then n times
              the source's (0.11.0).
    track:    a region that moves over time (0.17.0, a time-lapse only):
              {time: {spatial axis: shift}}, e.g. {0.0: {"x": 0.0},
              5.0: {"x": 12.5}}: at time 5 (the time axis' unit, as the t
              range) the region lies 12.5 µm further along x. The spatial
              `ranges` are the region at shift 0; between two times the
              shift is interpolated linearly, before the first and after
              the last time it stays as there (see ranges_at). Empty = a
              region fixed in place. A moving region is cropped into one
              time-lapse in which it keeps its size and the object stays in
              place: each timepoint is read at the region's position then
              (see plan_crop). Not with `outlines` (polygons that stay in
              place).
    timed_outlines: a moving region's outlines per timepoint (0.17.0):
              {time: outlines (as `outlines`)}, at the object's own
              positions then (µm), e.g. the silhouettes the Autocropper
              makes at each timepoint. Each timepoint of the crop is shaped
              by its own; a timepoint without any (e.g. where the object was
              missed) keeps the whole region.
    """

    name: str
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    channels: list[int] | None = None
    outlines: dict[tuple[str, str],
                   dict[float | None, list[tuple[float, float]]]] = field(
        default_factory=dict)
    margin: float = 0.0
    t_step: int = 1
    track: dict[float, dict[str, float]] = field(default_factory=dict)
    timed_outlines: dict[float, dict] = field(default_factory=dict)

    def __post_init__(self):
        self.outlines = normalize_outlines(self.outlines)
        self.track = normalize_track(self.track)
        self.timed_outlines = normalize_timed_outlines(self.timed_outlines)


# ---------------------------------------------------------------------------
# Regions that move over time (0.17.0)
# ---------------------------------------------------------------------------

def normalize_track(track: dict | None) -> dict:
    """A track in the standard form {time: {axis: shift}}, with floats,
    sorted by time."""
    items = sorted(((float(t), shifts) for t, shifts in (track or {}).items()),
                   key=lambda item: item[0])
    return {t: {str(a): float(v) for a, v in shifts.items()}
            for t, shifts in items}


def same_time(a: float, b: float) -> bool:
    """Whether two times are the same timepoint (equal up to rounding)."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def track_shift(track: dict, time: float) -> dict[str, float]:
    """A moving region's shift at a time, {axis: shift}: the track's own
    entry at that time, else interpolated linearly between the entries
    before and after it, else (before the first or after the last entry)
    that entry's. An empty track gives {} (no shift)."""
    if not track:
        return {}
    times = list(track)                      # sorted (normalize_track)
    for t in times:
        if same_time(t, time):
            return dict(track[t])
    if time < times[0]:
        return dict(track[times[0]])
    if time > times[-1]:
        return dict(track[times[-1]])
    after = next(i for i, t in enumerate(times) if t > time)
    t0, t1 = times[after - 1], times[after]
    w = (time - t0) / (t1 - t0)
    a, b = track[t0], track[t1]
    return {axis: (1 - w) * a.get(axis, 0.0) + w * b.get(axis, 0.0)
            for axis in sorted(set(a) | set(b))}


def ranges_at(box: CropBox, time: float) -> dict[str, tuple[float, float]]:
    """The box's ranges at a time: for a moving region (box.track), its
    spatial ranges moved by its shift then; a fixed region's ranges are the
    same at every time."""
    shift = track_shift(box.track, time)
    return {axis: ((low + shift[axis], high + shift[axis]) if axis in shift
                   else (low, high))
            for axis, (low, high) in box.ranges.items()}


def set_ranges_at(box: CropBox, time: float,
                  ranges: dict[str, tuple[float, float]]) -> None:
    """Change a box's ranges as seen at one time (the window's edits).

    A fixed region: its ranges change (as before 0.17.0). A moving region,
    along an axis it moves on: the same size moves the region at this
    timepoint only (its shift then is corrected; the other timepoints stay
    where they are); another size changes the size at every timepoint, each
    keeping its centre, and this timepoint's region is then the one given.
    Other axes (e.g. the t range) change for all timepoints. Without a
    time (None: an image without a time axis) the ranges change as for a
    fixed region.
    """
    if not box.track or time is None:
        box.ranges.update({a: tuple(r) for a, r in ranges.items()})
        return
    shift = track_shift(box.track, time)
    for axis, (low, high) in ranges.items():
        low, high = float(low), float(high)
        if axis not in shift or axis not in box.ranges:
            box.ranges[axis] = (low, high)
            continue
        base_low, base_high = box.ranges[axis]
        if not math.isclose(high - low, base_high - base_low,
                            rel_tol=1e-9, abs_tol=1e-9):
            centre = (base_low + base_high) / 2
            base_low, base_high = (centre - (high - low) / 2,
                                   centre + (high - low) / 2)
            box.ranges[axis] = (base_low, base_high)
        shift[axis] = low - base_low
    key = next((t for t in box.track if same_time(t, time)), float(time))
    box.track[key] = shift
    box.track = normalize_track(box.track)


def track_times(image: OmeZarrImage, box: CropBox) -> list[float]:
    """The times of the box's timepoints at level 0 (its t range, every
    t_step-th), e.g. for the positions of a moving region."""
    t_axis = time_axis(image)
    if t_axis is None:
        return []
    level0 = image.levels[0]
    name = image.axis_names[t_axis]
    scale, origin = level0.scale[t_axis], level0.translation[t_axis]
    first, end = 0, level0.array.shape[t_axis]
    if name in box.ranges:
        low, high = sorted(box.ranges[name])
        a, b = physical_to_index_range(low, high, scale, origin)
        first, end = max(a, first), min(b, end)
    return [origin + i * scale
            for i in range(first, end, max(int(box.t_step), 1))]


def fixed_in_place(image: OmeZarrImage, box: CropBox) -> CropBox:
    """A moving region made fixed (the window's "Fix in place"): one region
    that holds all its positions over its timepoints (the union, as a fit
    over all timepoints gave before 0.17.0), with no track."""
    fixed = copy.deepcopy(box)
    fixed.track, fixed.timed_outlines = {}, {}
    moving = {axis for shifts in box.track.values() for axis in shifts}
    times = track_times(image, box) or list(box.track)
    for axis in sorted(moving & set(box.ranges)):
        spans = [ranges_at(box, time)[axis] for time in times]
        fixed.ranges[axis] = (min(low for low, _ in spans),
                              max(high for _, high in spans))
    return fixed


def normalize_timed_outlines(timed: dict | None) -> dict:
    """Outlines per timepoint in the standard form {time: outlines}, with
    float times, sorted; timepoints without outlines are dropped."""
    items = sorted(((float(t), normalize_outlines(o))
                    for t, o in (timed or {}).items()), key=lambda i: i[0])
    return {t: o for t, o in items if o}


def timed_outlines_at(box: CropBox, time: float) -> dict:
    """A region's own outlines at a timepoint (box.timed_outlines; {} if
    it has none then: the whole region is kept)."""
    if time is None:
        return {}
    return next((o for t, o in box.timed_outlines.items()
                 if same_time(t, time)), {})


def box_at(box: CropBox, time: float) -> CropBox:
    """The region as it is at a timepoint (0.17.0), e.g. for the views: its
    ranges then (ranges_at) and, with outlines per timepoint, that
    timepoint's as its outlines; it neither moves nor has outlines per
    timepoint itself."""
    outlines = (timed_outlines_at(box, time) if box.timed_outlines
                else box.outlines)
    return CropBox(box.name, ranges_at(box, time), box.channels,
                   copy.deepcopy(outlines), box.margin, box.t_step)


def normalize_outlines(outlines: dict) -> dict:
    """Outlines in the standard form {(a, b): {key: [(a, b), ...]}}, with
    floats; a key is a slice (float), a depth range (low, high) for a
    silhouette, or None. A plain list of points becomes one polygon
    without a slice ({None: points}); views without polygons are dropped."""
    result = {}
    for plane, value in (outlines or {}).items():
        keys = value if isinstance(value, dict) else {None: value}
        if keys:
            result[tuple(plane)] = {
                _normal_key(key):
                    [tuple(float(v) for v in point) for point in points]
                for key, points in keys.items()}
    return result


def _normal_key(key):
    """A polygon's key as None, a float (slice) or (low, high) (range)."""
    if key is None:
        return None
    if isinstance(key, (tuple, list)):
        low, high = sorted(float(v) for v in key)
        return (low, high)
    return float(key)


def is_silhouette(key) -> bool:
    """Whether a polygon's key is a depth range (drawn on a projection)."""
    return isinstance(key, tuple)


@dataclass
class CropPlan:
    """Exactly what will be cut out of which level (all voxel indices)."""

    box: CropBox
    level_indices: list[int]     # source levels that are kept (0, 1, ...)
    starts: list[list[int]]      # per kept level: first voxel, per axis
    stops: list[list[int]]       # per kept level: end voxel (exclusive)
    shapes: list[tuple[int, ...]]   # per kept level: output array shape
    channels: list[int] | None   # source channel indices, None if no c axis
    snapped_ranges: dict[str, tuple[float, float]]   # physical, as written
    # Per axis: every how many source voxels one is kept (1 everywhere,
    # except along t with CropBox.t_step); None = all 1. Output index j of
    # an axis comes from source index start + j * step.
    steps: list[int] | None = None
    # A region that moves over time (0.17.0; None = fixed): per kept level,
    # an integer array (output timepoints x axes) of each output
    # timepoint's shift in that level's voxels, relative to the first
    # output timepoint (row 0 is all 0; the c and t columns are 0). starts,
    # stops and snapped_ranges are the region at the first output
    # timepoint; `times` are the output timepoints' times (level 0, the
    # time axis' unit) and `t_axis` is the time axis' index.
    shifts: list[np.ndarray] | None = None
    times: list[float] | None = None
    t_axis: int | None = None

    def step(self, i: int) -> int:
        """The step along axis i (1 = every voxel)."""
        return self.steps[i] if self.steps else 1

    @property
    def moving(self) -> bool:
        """Whether the region moves over time (see shifts)."""
        return self.shifts is not None

    def describe(self, image: OmeZarrImage) -> str:
        """A short human-readable summary of the plan."""
        lines = [f"Box {self.box.name!r}:"]
        requested = (ranges_at(self.box, self.times[0]) if self.moving
                     else self.box.ranges)
        for axis in image.range_axes:
            i = image.axis_names.index(axis)
            unit = short_unit(image.unit(axis))
            a, b = self.snapped_ranges[axis]
            req = requested.get(axis)
            req_text = (f"requested {req[0]:g}-{req[1]:g}" if req
                        else "whole axis")
            count = self.shapes[0][i]
            kept = (f"{count} voxels at level 0" if self.step(i) == 1 else
                    f"{count} of {self.stops[0][i] - self.starts[0][i]} "
                    f"voxels: every {_ordinal(self.step(i))}")
            lines.append(
                f"  {axis}: {req_text} -> cropped {a:g}-{b:g} {unit} "
                f"({kept})")
        if self.moving:
            lines.append("  " + self.moving_text(image))
            if self.box.timed_outlines:
                lines.append("  " + self.timed_outlines_text())
        if self.channels is not None:
            labels = image.channel_labels
            lines.append("  channels: " + ", ".join(
                f"{c} ({labels[c]})" for c in self.channels))
        spatial = _spatial_axes(image)
        through = _through_box(self.box.outlines)
        for (a, b), keys in self.box.outlines.items():
            if (a, b) in through:
                lines.append(f"  outline in the {a}-{b} view: one polygon, "
                             "through the whole box")
                continue
            depth = _depth_axis((a, b), spatial)
            slices = sorted(k for k in keys if isinstance(k, float))
            ranges = sorted(k for k in keys if is_silhouette(k))
            if slices:
                lines.append(
                    f"  outline in the {a}-{b} view: {len(slices)} "
                    f"polygon(s), {depth} = "
                    + ", ".join(f"{d:g}" for d in slices))
            for low, high in ranges:
                lines.append(f"  silhouette in the {a}-{b} view (drawn on a "
                             f"projection), through {depth} = {low:g}-"
                             f"{high:g}")
        if len(_sliced_polygons(self.box.outlines, spatial)) > 1:
            lines.append("  one 3D shape through all these polygons, ending "
                         "at the outermost ones")
        if self.box.outlines:
            if self.box.margin:
                lines.append(f"  margin: {self.box.margin:+g} voxels "
                             f"({self.box.margin * pixel_size(image):+.2f} "
                             f"{short_unit(image.unit(spatial[-1]))})")
            lines.append("  voxels outside the outlines are written as 0")
        dtype = image.levels[0].array.dtype
        lines.append(
            f"  levels kept: {self.level_indices[0]}-{self.level_indices[-1]}"
            f" of 0-{len(image.levels) - 1}; level-0 shape {self.shapes[0]}, "
            f"{format_bytes(nbytes(self.shapes[0], dtype))} uncompressed")
        return "\n".join(lines)

    def timed_outlines_text(self) -> str:
        """For a moving region with outlines per timepoint: how many of the
        crop's timepoints have their own outlines."""
        n = sum(bool(timed_outlines_at(self.box, t)) for t in self.times)
        rest = len(self.times) - n
        return (f"outlines per timepoint at {n} of {len(self.times)} "
                "timepoints (voxels outside them are written as 0)"
                + (f"; the other {rest} keep the whole region" if rest
                   else ""))

    def moving_text(self, image: OmeZarrImage) -> str:
        """For a moving region: e.g. "moves with the object: 40 positions,
        up to 85.3 µm from the first (ranges above: at the first)"."""
        level = image.levels[self.level_indices[0]]
        moved = self.shifts[0] * np.asarray(level.scale)    # µm per axis
        distance = float(np.sqrt((moved ** 2).sum(axis=1)).max())
        positions = len({tuple(row) for row in self.shifts[0].tolist()})
        spatial = _spatial_axes(image)
        unit = short_unit(image.unit(spatial[-1])) if spatial else ""
        t_name = image.axis_names[self.t_axis]
        return (f"moves with the object: {positions} position"
                f"{'' if positions == 1 else 's'} over {len(self.times)} "
                f"timepoint{'' if len(self.times) == 1 else 's'}, up to "
                f"{distance:.3g} {unit} from where it is at {t_name} = "
                f"{self.times[0]:g} (the ranges above)")


def _ordinal(n: int) -> str:
    """2 -> "2nd", 3 -> "3rd", 11 -> "11th" ..."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def time_axis(image: OmeZarrImage) -> int | None:
    """Index of the image's time axis (None if it has none)."""
    return next((i for i, a in enumerate(image.axes)
                 if a.get("type") == "time"), None)


def physical_to_index_range(start: float, stop: float, scale: float,
                            translation: float) -> tuple[int, int]:
    """Voxel index range [first, end) of all voxels touching [start, stop].

    The range is rounded outward, so the box is never made smaller.
    """
    u_start = (start - translation) / scale   # fractional voxel index
    u_stop = (stop - translation) / scale
    first = math.floor(u_start - 0.5 + _EPS) + 1
    end = math.ceil(u_stop + 0.5 - _EPS)
    return first, end


def index_range_to_physical(first: int, end: int, scale: float,
                            translation: float) -> tuple[float, float]:
    """Physical range covered by voxels [first, end) (outer voxel edges)."""
    return (translation + (first - 0.5) * scale,
            translation + (end - 0.5) * scale)


def _spatial_axes(image: OmeZarrImage) -> list[str]:
    """Names of the image's spatial axes (e.g. z, y, x)."""
    return [a["name"] for a in image.axes if a.get("type") == "space"]


def _depth_axis(plane: tuple[str, str], spatial: list[str]) -> str:
    """A view's depth axis: the spatial axis that its plane does not use."""
    others = [axis for axis in spatial if axis not in plane]
    if len(others) != 1:
        raise ValueError("Polygons on several slices need an image with "
                         f"three spatial axes (this one has {spatial}).")
    return others[0]


def _check_outlines(box: CropBox, spatial: list[str]) -> None:
    """Refuse outlines that cannot be used (with a message saying why)."""
    views = set()
    for plane, keys in box.outlines.items():
        if len(plane) != 2 or plane[0] == plane[1] \
                or not set(plane) <= set(spatial):
            raise ValueError(f"Box {box.name!r}: outlines need two different "
                             f"spatial axes, not {plane}.")
        if frozenset(plane) in views:
            raise ValueError(f"Box {box.name!r}: two sets of outlines for "
                             f"the {plane[0]}-{plane[1]} view.")
        views.add(frozenset(plane))
        for points in keys.values():
            vertices = np.asarray(points, dtype=float)
            if vertices.ndim != 2 or vertices.shape[0] < 3 \
                    or vertices.shape[1] != 2:
                raise ValueError(f"Box {box.name!r}: an outline needs at "
                                 "least three (a, b) points.")
        if len(keys) > 1 and None in keys:
            raise ValueError(
                f"Box {box.name!r}: with several polygons in the "
                f"{plane[0]}-{plane[1]} view, each needs its slice.")
        if len(keys) > 1 or any(is_silhouette(k) for k in keys):
            _depth_axis(plane, spatial)


def pixel_size(image: OmeZarrImage) -> float:
    """The finest voxel size (µm) of the full-resolution image: the unit of
    a box's outline margin ("grow by 5 voxels")."""
    level0, spatial = image.levels[0], _spatial_axes(image)
    return min(level0.scale[image.axis_names.index(a)] for a in spatial)


def _sliced_polygons(outlines: dict, spatial: list[str]) -> list:
    """The polygons that have a slice, as (plane, slice, 2D points, 3D
    points); the 3D points are in the order of `spatial`."""
    polygons = []
    for plane, keys in outlines.items():
        slices = [k for k in keys if isinstance(k, float)]
        if not slices:
            continue
        depth_axis = _depth_axis(plane, spatial)
        for depth in slices:
            flat = np.asarray(keys[depth], dtype=float)
            solid = np.empty((len(flat), 3))
            for k, axis in enumerate(plane):
                solid[:, spatial.index(axis)] = flat[:, k]
            solid[:, spatial.index(depth_axis)] = depth
            polygons.append((plane, depth, flat, solid))
    return polygons


def _through_box(outlines: dict) -> dict:
    """The polygons that reach through the whole box, as {plane: {key:
    points}}: a polygon without a slice (boxes from before 0.7.0), or the
    box's only polygon on a slice."""
    result = {plane: {None: keys[None]} for plane, keys in outlines.items()
              if None in keys}
    sliced = [(plane, key) for plane, keys in outlines.items()
              for key in keys if isinstance(key, float)]
    if len(sliced) == 1:
        plane, key = sliced[0]
        result[plane] = {key: outlines[plane][key]}
    return result


def _silhouettes(outlines: dict, spatial: list[str]) -> list:
    """The polygons drawn on projections, as (plane, depth axis, (low,
    high), points)."""
    return [(plane, _depth_axis(plane, spatial), key, points)
            for plane, keys in outlines.items()
            for key, points in keys.items() if is_silhouette(key)]


def _effective_ranges(image: OmeZarrImage, box: CropBox) -> dict:
    """The box ranges, shrunk to where the outlines can keep voxels.

    The shape through a box's polygons (see outline_mask) ends at its
    outermost polygons, so the crop needs no more than their extent along
    every axis; a polygon through the box limits its own two axes. A
    positive margin widens this by the margin, and the box's own ranges
    limit it further.
    """
    _check_outlines(box, _spatial_axes(image))
    ranges = {a: tuple(sorted(r)) for a, r in box.ranges.items()}
    for axis, low, high in outline_limits(image, box):
        if axis in ranges:
            low, high = max(low, ranges[axis][0]), min(high, ranges[axis][1])
        if high <= low:
            raise ValueError(
                f"Box {box.name!r}: nothing would be kept along {axis}: "
                "its outlines do not overlap there (or lie outside the "
                "box).")
        ranges[axis] = (float(low), float(high))
    return ranges


def outline_limits(image: OmeZarrImage, box: CropBox) -> list:
    """How far a box's outlines can keep voxels, as (axis, low, high) in
    µm: a polygon through the box limits its two axes, and the shape
    through polygons on slices ends at their extent along every axis. A
    positive margin widens each limit by the margin."""
    spatial = _spatial_axes(image)
    limits = []
    for plane, keys in _through_box(box.outlines).items():
        points = np.asarray(next(iter(keys.values())), dtype=float)
        limits += [(axis, points[:, k].min(), points[:, k].max())
                   for k, axis in enumerate(plane)]
    # silhouettes: those of one view together limit its two axes, and its
    # depth axis to the depth they were drawn over
    views: dict = {}
    for plane, depth_axis, (low, high), points in _silhouettes(box.outlines,
                                                               spatial):
        views.setdefault((plane, depth_axis), []).append(
            (low, high, np.asarray(points, dtype=float)))
    for (plane, depth_axis), group in views.items():
        points = np.concatenate([pts for _, _, pts in group])
        limits += [(axis, points[:, k].min(), points[:, k].max())
                   for k, axis in enumerate(plane)]
        limits.append((depth_axis, min(g[0] for g in group),
                       max(g[1] for g in group)))
    polygons = _sliced_polygons(box.outlines, spatial)
    if len(polygons) > 1:
        solid = np.concatenate([p for *_, p in polygons])
        limits += [(axis, solid[:, k].min(), solid[:, k].max())
                   for k, axis in enumerate(spatial)]
    grow = max(box.margin, 0) * pixel_size(image)
    return [(axis, float(low - grow), float(high + grow))
            for axis, low, high in limits]


def outline_mask(level: Level, axis_names: list[str], first: list[int],
                 shape: tuple[int, ...], outlines: dict,
                 spatial: list[str] | None = None, margin: float = 0.0,
                 detail: int | None = None):
    """Which voxels of a block the outlines keep (True = keep).

    level:    the pyramid level the block belongs to (for scale and offset).
    first:    the source voxel index of the block's first voxel, per axis.
    shape:    the block's shape.
    outlines: as in CropBox.
    spatial:  the image's spatial axes (default: all of axis_names).
    detail:   grid points of the shape's function along the polygons'
              longest extent (default FIELD_POINTS; the views use fewer,
              to stay quick while polygons are drawn).
    margin:   grow (> 0) or shrink (< 0) the kept region by this distance
              (µm): every voxel whose centre lies within the margin of the
              region is added, or every voxel within the margin of the
              outside is removed (Euclidean distance between voxel
              centres).

    The polygons drawn on slices (in any views) define one 3D shape that
    passes through all of them; it ends at the outermost polygons (see
    _shape_mask). A polygon that reaches through the box (the box's only
    polygon, or one without a slice) keeps what lies inside it in its two
    axes. A voxel is kept if it is inside all of these (its centre decides,
    as in Fiji's polygon ROIs). The result has size 1 along axes that the
    outlines do not use, so it broadcasts to the block. Returns None when
    there are no outlines.
    """
    outlines = normalize_outlines(outlines)
    if not outlines:
        return None
    spatial = list(spatial or axis_names)
    if not margin:
        return _base_mask(level, axis_names, first, shape, outlines, spatial,
                          detail)
    # The margin needs the region's neighbourhood: compute it on the block
    # plus a rim as wide as the margin, then measure distances (µm).
    rim = [int(math.ceil(abs(margin) / level.scale[i])) + 1
           if axis_names[i] in spatial else 0 for i in range(len(shape))]
    big_first = [f - r for f, r in zip(first, rim)]
    big_shape = [n + 2 * r if axis_names[i] in spatial else 1
                 for i, (n, r) in enumerate(zip(shape, rim))]
    base = _base_mask(level, axis_names, big_first, big_shape, outlines,
                      spatial, detail)
    base = np.broadcast_to(base, big_shape)
    sampling = [level.scale[i] if axis_names[i] in spatial else 1.0
                for i in range(len(shape))]
    if not base.any():
        grown = base
    elif margin > 0:
        grown = base | (distance_transform_edt(~base, sampling=sampling)
                        <= margin + 1e-9)
    elif base.all():
        grown = base
    else:
        grown = distance_transform_edt(base, sampling=sampling) > -margin
    inner = tuple(slice(r, r + n) if axis_names[i] in spatial else slice(None)
                  for i, (n, r) in enumerate(zip(shape, rim)))
    return np.ascontiguousarray(grown[inner])


def _base_mask(level, axis_names, first, shape, outlines, spatial,
               detail=None):
    """outline_mask without a margin."""
    mask = None
    for plane, keys in _through_box(outlines).items():
        part = _prism_mask(level, axis_names, first, shape, plane,
                           next(iter(keys.values())))
        mask = part if mask is None else mask & part
    for plane, depth_axis, depth_range, points in _silhouettes(outlines,
                                                               spatial):
        part = _silhouette_mask(level, axis_names, first, shape, plane,
                                depth_axis, depth_range, points)
        mask = part if mask is None else mask & part
    if len(_sliced_polygons(outlines, spatial)) > 1:
        part = _shape_mask(level, axis_names, first, shape,
                           _sliced_only(outlines), spatial, detail)
        mask = part if mask is None else mask & part
    return mask


def _prism_mask(level, axis_names, first, shape, plane, points):
    """A polygon that reaches through the block along its third axis."""
    ia, ib = axis_names.index(plane[0]), axis_names.index(plane[1])
    vertices = np.asarray(points, dtype=float)
    # the polygon in voxel units of this level, relative to the block
    rows = (vertices[:, 0] - level.translation[ia]) / level.scale[ia] \
        - first[ia]
    cols = (vertices[:, 1] - level.translation[ib]) / level.scale[ib] \
        - first[ib]
    inside = fill_polygon((shape[ia], shape[ib]),
                          np.column_stack([rows, cols]))
    if ia > ib:
        inside = inside.T
    broadcast = [1] * len(shape)
    broadcast[ia], broadcast[ib] = shape[ia], shape[ib]
    return inside.reshape(broadcast)


def fill_polygon(shape: tuple, vertices) -> np.ndarray:
    """The pixels of a grid (centres at whole-number (row, column)
    coordinates) inside a polygon given as (row, column) points.

    Inside is what Fiji fills (a pixel the outline goes around an odd
    number of times), plus what the outline goes around more than once:
    the doubly covered part of a lasso that overshoots its start, which
    Fiji's even-odd rule leaves out as a hole (the user's choice,
    2026-11-10: "fill it"). For an outline that does not cross itself the
    result is Fiji's exactly."""
    vertices = np.asarray(vertices, dtype=float)
    inside = polygon2mask(shape, vertices)
    winding = _winding_numbers(shape, vertices)
    return inside | (np.abs(winding) >= 2)


def polygon_width(points) -> float:
    """A polygon's mean width (µm): twice its filled area (see
    fill_polygon) over its outline's length. Three clicks on a line give
    (nearly) 0: the window refuses polygons less than a voxel wide (the
    user's choice, 2026-11-10), as their shape would narrow to nothing."""
    points = np.asarray(points, dtype=float)
    ring = np.vstack([points, points[:1]])
    perimeter = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
    extent = float(np.ptp(points, axis=0).max())
    if perimeter <= 0 or extent <= 0:
        return 0.0
    spacing = extent / 512                      # fill it on a fine grid
    low = points.min(axis=0) - spacing
    shape = tuple(int(n) for n in np.ceil(np.ptp(points, axis=0)
                                          / spacing) + 3)
    area = fill_polygon(shape, (points - low) / spacing).sum() * spacing ** 2
    return 2.0 * area / perimeter


def _winding_numbers(shape: tuple, vertices) -> np.ndarray:
    """How many times a polygon's outline goes around each pixel centre
    (signed by direction), by scanlines: along each row, the edges crossing
    it, counted +1 going one way and -1 the other."""
    steps = np.zeros((shape[0], shape[1] + 1), dtype=np.int32)
    r0, c0 = vertices[:, 0], vertices[:, 1]
    r1, c1 = np.roll(r0, -1), np.roll(c0, -1)
    for a, b, ca, cb in zip(r0, r1, c0, c1):
        if a == b:
            continue                          # horizontal: crosses no row
        lo, hi = min(a, b), max(a, b)
        rows = np.arange(max(int(np.ceil(lo)), 0),
                         min(int(np.ceil(hi)), shape[0]))
        if not len(rows):
            continue
        cols = ca + (rows - a) * (cb - ca) / (b - a)   # where it crosses
        # every pixel right of the crossing is gone around once more (a
        # step, added up along the row below)
        start = np.clip(np.ceil(cols).astype(int), 0, shape[1])
        np.add.at(steps, (rows, start), 1 if b > a else -1)
    return np.cumsum(steps, axis=1)[:, :shape[1]]


def _silhouette_mask(level, axis_names, first, shape, plane, depth_axis,
                     depth_range, points):
    """A polygon drawn on a projection: within the depth range that was
    projected, a voxel must lie inside it; outside that range it says
    nothing (outline_limits keeps the crop within the ranges)."""
    inside = _prism_mask(level, axis_names, first, shape, plane, points)
    i = axis_names.index(depth_axis)
    centres = level.translation[i] + (first[i] + np.arange(shape[i])) \
        * level.scale[i]
    beyond = (centres < depth_range[0] - 1e-9) | (centres > depth_range[1]
                                                   + 1e-9)
    broadcast = [1] * len(shape)
    broadcast[i] = shape[i]
    return inside | beyond.reshape(broadcast)


def edge_crossings(points, axis: int, value: float) -> np.ndarray:
    """Where a closed polygon's edges cross the line on which its
    coordinate `axis` (0 or 1) equals value: the other coordinate at each
    crossing, sorted."""
    p = np.asarray(points, dtype=float)
    a, b = p[:, axis], p[:, 1 - axis]
    a2, b2 = np.roll(a, -1), np.roll(b, -1)
    cross = (a > value) != (a2 > value)
    t = (value - a[cross]) / (a2[cross] - a[cross])
    return np.sort(b[cross] + t * (b2[cross] - b[cross]))


def crossing_points(outlines: dict, spatial: list[str], plane,
                    position: float, with_planes: bool = False):
    """Where the polygons on slices of the other views cross a view's
    slice: (vertical, horizontal) points of that view (µm). There, the
    view's own outline should meet them.

    with_planes (0.16.0): also return, per point, the plane of the polygon
    it comes from (so that the window can colour each mark by its view):
    (points, planes)."""
    depth = _depth_axis(plane, spatial)
    found, planes = [], []
    for other, other_slice, flat, _ in _sliced_polygons(outlines, spatial):
        if set(other) == set(plane):
            continue
        k = other.index(depth)
        values = edge_crossings(flat, k, position)
        points = np.empty((len(values), 2))
        points[:, list(plane).index(_depth_axis(other, spatial))] = \
            other_slice
        points[:, list(plane).index(other[1 - k])] = values
        found.append(points)
        planes += [tuple(other)] * len(values)
    points = np.concatenate(found) if found else np.zeros((0, 2))
    return (points, planes) if with_planes else points


def outline_disagreements(outlines: dict, spatial: list[str]) -> list:
    """Pairs of polygons on slices of different views that disagree where
    their slices cross, as (plane, slice, other plane, other slice,
    distance in µm), largest first.

    Two such slices cross along a line, and both polygons' edges should meet
    that line at the same points (the object's edge). The distance is the
    largest gap between those points; if only one polygon reaches the line,
    it is how far the other one stays from it.
    """
    polygons = _sliced_polygons(outlines, spatial)
    result = []
    for i, (plane_a, slice_a, flat_a, _) in enumerate(polygons):
        depth_a = _depth_axis(plane_a, spatial)
        for plane_b, slice_b, flat_b, _ in polygons[i + 1:]:
            depth_b = _depth_axis(plane_b, spatial)
            if depth_a == depth_b:
                continue
            on_a = edge_crossings(flat_a, plane_a.index(depth_b), slice_b)
            on_b = edge_crossings(flat_b, plane_b.index(depth_a), slice_a)
            if len(on_a) and len(on_b):
                gaps = np.abs(on_a[:, None] - on_b[None, :])
                gap = max(gaps.min(axis=1).max(), gaps.min(axis=0).max())
            elif len(on_a):
                gap = np.abs(flat_b[:, plane_b.index(depth_a)] - slice_a).min()
            elif len(on_b):
                gap = np.abs(flat_a[:, plane_a.index(depth_b)] - slice_b).min()
            else:
                continue
            result.append((plane_a, slice_a, plane_b, slice_b, float(gap)))
    return sorted(result, key=lambda r: -r[4])


def _shape_mask(level, axis_names, first, shape, sliced, spatial,
                detail=None):
    """The 3D shape through polygons drawn on slices, in a block.

    A function of (z, y, x) is negative inside the shape. Polygons drawn in
    one view only are interpolated from slice to slice (_one_view_field:
    signed distance maps, blended linearly between neighbouring drawn
    slices). Polygons in several views define one smooth surface
    (_shape_function: a variational implicit surface). Either way the shape
    passes through every polygon and ends at the outermost polygons along
    every axis (no extrapolation beyond what was drawn). The function is
    computed once per box on a coarse grid (_shape_field) and interpolated
    linearly to each voxel.
    """
    origin, step, field, low, high = _shape_field(
        _frozen(sliced), tuple(spatial), detail or FIELD_POINTS)
    index = [axis_names.index(axis) for axis in spatial]
    centres = [level.translation[i] + (first[i] + np.arange(shape[i]))
               * level.scale[i] for i in index]
    within = [(c >= lo - 1e-9) & (c <= hi + 1e-9)
              for c, lo, hi in zip(centres, low, high)]
    inside = np.zeros([shape[i] for i in index], dtype=bool)
    if all(w.any() for w in within):
        part = tuple(slice(np.argmax(w), len(w) - np.argmax(w[::-1]))
                     for w in within)
        values = _sample_grid(field, [(c[p] - o) / step for c, p, o in
                                      zip(centres, part, origin)])
        inside[part] = values < 0
    broadcast = [1] * len(shape)
    for i in index:
        broadcast[i] = shape[i]
    return inside.transpose(np.argsort(index)).reshape(broadcast)


def _frozen(sliced: dict) -> tuple:
    """Polygons as nested tuples, so that their shape can be remembered
    (keys of all kinds: slices, depth ranges, None)."""
    return tuple(sorted(
        (plane, tuple(sorted(((key, tuple(map(tuple, points)))
                              for key, points in keys.items()),
                             key=lambda item: _slice_order(item[0]))))
        for plane, keys in sliced.items()))


def prepare_shape(image: OmeZarrImage, box: CropBox) -> None:
    """Compute the shape through a box's polygons now (it is remembered):
    before the parallel copying, so that the workers do not all compute it
    at the same time."""
    spatial = _spatial_axes(image)
    if len(_sliced_polygons(box.outlines, spatial)) > 1:
        _shape_field(_frozen(_sliced_only(box.outlines)), tuple(spatial),
                     FIELD_POINTS)


def _sliced_only(outlines: dict) -> dict:
    """Only the polygons on slices: {plane: {slice: points}}."""
    result = {}
    for plane, keys in outlines.items():
        slices = {k: v for k, v in keys.items() if isinstance(k, float)}
        if slices:
            result[plane] = slices
    return result


# Grid points of the shape's function along the longest side of the
# polygons' extent (the shape's detail; see _shape_field).
FIELD_POINTS = 128
# Holes are filled on a grid this many times finer (see _fill_holes).
HOLE_REFINE = 2


@functools.lru_cache(maxsize=16)
def _shape_function(frozen: tuple, spatial: tuple):
    """The function of _shape_mask (negative inside) for polygons drawn in
    several views, fitted once per set of polygons. frozen: ((plane,
    ((slice, points), ...)), ...).

    Variational implicit surface (Turk and O'Brien, Proc. SIGGRAPH 1999,
    335-342; ACM Trans. Graph. 21: 855-873, 2002): one smooth function is
    fitted through points on every polygon (value 0) and points just inside
    (negative) and just outside (positive) it, with radial basis functions
    (the cubic kernel r^3, a polyharmonic spline), so the shape is
    interpolated in all directions at once.

    Returns (function, points, low, high): the function (call it with
    (n, 3) points in µm), the points it was fitted to, and the polygons'
    extent (low, high per spatial axis), where the shape ends.
    """
    spatial = list(spatial)
    outlines = {plane: dict(keys) for plane, keys in frozen}
    polygons = _sliced_polygons(outlines, spatial)
    solid = np.concatenate([p for *_, p in polygons])
    low, high = solid.min(axis=0), solid.max(axis=0)
    step = max(float((high - low).max()) / FIELD_POINTS, 1e-6)
    # Points on the edge of each filled polygon (every 3 steps of the
    # crop's grid; value 0) and one step to either side of it, in its own
    # plane (value: the signed distance to that edge). The polygon is
    # filled first (as Fiji fills it), so that a lasso that overshoots its
    # start, crosses itself or wobbles still gives consistent points: taken
    # along the drawn line itself, an overshoot of 5 % of a turn kept only
    # 43 % of the polygon on its own slice, and a shaky lasso (0.7 µm
    # tremor) broke the shape into pieces.
    # At most MAX_FIT_POINTS points (the fit's time grows with their
    # number cubed): a very long or shaky edge gets them further apart.
    edges = [_edge_points(flat, step, 3 * step) for _, _, flat, _ in polygons]
    count = sum(len(p) for edge in edges for p, _ in edge)
    if count > MAX_FIT_POINTS:
        edges = [_edge_points(flat, step, 3 * step * count / MAX_FIT_POINTS)
                 for _, _, flat, _ in polygons]
    points, values = [], []
    for (plane, depth, flat, _), edge in zip(polygons, edges):
        depth_axis = _depth_axis(plane, spatial)
        for moved, value in edge:
            embedded = np.empty((len(moved), 3))
            for k, axis in enumerate(plane):
                embedded[:, spatial.index(axis)] = moved[:, k]
            embedded[:, spatial.index(depth_axis)] = depth
            points.append(embedded)
            values.append(value)
    points, values = np.concatenate(points), np.concatenate(values)
    points, unique = np.unique(np.round(points, 9), axis=0,
                               return_index=True)
    # The cubic kernel (r^3): with the linear one (r), the shape narrowed
    # between polygons on slices far apart (the same square on two slices
    # 2 widths apart kept half its area in the middle, 3 widths apart none);
    # with r^3 it stays a straight prism (see dev/DEVELOPERS.md).
    function = RBFInterpolator(points, values[unique], kernel="cubic",
                               degree=1)
    return function, points, low, high


@functools.lru_cache(maxsize=16)
def _shape_field(frozen: tuple, spatial: tuple, detail: int = FIELD_POINTS):
    """The function of _shape_mask on a grid, once per set of polygons and
    detail (a lower detail draws the same shape more coarsely).

    Polygons drawn in one view only (the usual case: outlines on a few XY
    slices) are interpolated slice to slice (_one_view_field); polygons in
    several views define one smooth surface (_surface_field). Then holes
    are filled (_fill_holes).

    Returns (origin, step, values, low, high): the grid's first point (µm,
    per spatial axis), its spacing (µm, the same along every axis: the
    polygons' longest extent / detail / HOLE_REFINE), the values, and the
    polygons' extent (low, high per axis), where the shape ends.
    """
    spatial = list(spatial)
    polygons = _sliced_polygons({plane: dict(keys) for plane, keys in frozen},
                                spatial)
    if len({frozenset(plane) for plane, *_ in polygons}) == 1:
        origin, step, field, low, high = _one_view_field(polygons, spatial,
                                                         detail)
    else:
        origin, step, field, low, high = _surface_field(frozen,
                                                        tuple(spatial), detail)
    # The same (trilinear) function on a grid HOLE_REFINE times finer, so
    # that holes between the coarse grid's planes are found too.
    field = _sample_grid(field, [np.arange((n - 1) * HOLE_REFINE + 1)
                                 / HOLE_REFINE for n in field.shape])
    step /= HOLE_REFINE
    _fill_holes(field, origin, step, low, high)
    return origin, step, field, low, high


def _shape_grid(low, high, detail: int):
    """The grid of a shape's function: (origin, step, size). The spacing is
    the same along every axis (the polygons' longest extent / detail), and
    the grid reaches 2 steps beyond the polygons' extent on each side."""
    step = max(float((high - low).max()) / detail, 1e-6)
    origin = low - 2 * step
    size = [int(math.ceil((h - lo) / step)) + 5 for lo, h in zip(low, high)]
    return origin, step, size


# Polygons drawn in one view: their signed distance maps are computed on a
# grid this many times finer than the shape's, in their plane.
SLICE_REFINE = 4
# Polygons in several views: at most this many points for the smooth
# surface's fit (_shape_function; about 5 s for 4000).
MAX_FIT_POINTS = 4000


def _one_view_field(polygons: list, spatial: list, detail: int):
    """The function of _shape_mask (negative inside) for polygons drawn on
    slices of one view only, on its grid: (origin, step, values, low, high)
    as _shape_field, before holes are filled.

    Shape-based interpolation (Raya and Udupa, IEEE Trans. Med. Imaging 9:
    32-42, 1990): on each drawn slice, the signed distance (µm) to the edge
    of the filled polygons (negative inside; several polygons on one slice
    are joined); between two drawn slices, the two maps are blended
    linearly by distance (so a polygon only affects the shape up to the
    next drawn slice on each side, and the same polygon on two slices
    gives a straight prism). Where two neighbouring slices' filled areas
    hardly overlap, the maps are moved before blending so that their
    centres line up (the centre itself moves linearly from slice to
    slice), so that an object that shifts sideways between slices by more
    than its width does not pinch off. Beyond the first and
    last drawn slice the nearest map is used (_shape_mask stops the shape
    there).
    """
    solid = np.concatenate([p for *_, p in polygons])
    low, high = solid.min(axis=0), solid.max(axis=0)
    origin, step, size = _shape_grid(low, high, detail)
    plane = polygons[0][0]
    ia, ib = (spatial.index(axis) for axis in plane)
    idep = spatial.index(_depth_axis(plane, spatial))
    fine = step / SLICE_REFINE
    fine_shape = ((size[ia] - 1) * SLICE_REFINE + 1,
                  (size[ib] - 1) * SLICE_REFINE + 1)
    corner = np.array([origin[ia], origin[ib]])
    on_slice: dict = {}
    for _, depth, flat, _ in polygons:
        on_slice.setdefault(depth, []).append(flat)
    depths = sorted(on_slice)
    maps, centres, areas = [], [], []
    for depth in depths:
        distance, centre = _slice_distance(on_slice[depth], corner, fine,
                                           fine_shape)
        maps.append(distance)
        centres.append(centre)
        areas.append(distance < 0)
    # How far each pair of neighbouring slices is lined up by the centres:
    # fully where their filled areas do not overlap (an object that moved
    # sideways by more than its width), not at all where they overlap by
    # half of the smaller one or more (plain blending is right there; the
    # second review of 0.17.1: a part that stays put pinched off when
    # another part of the polygon ended).
    align = []
    for a, b in zip(areas, areas[1:]):
        smaller = min(a.sum(), b.sum())
        overlap = (a & b).sum() / smaller if smaller else 0.0
        align.append(float(np.clip(1.0 - 2.0 * overlap, 0.0, 1.0)))
    # The shape's grid points in the plane, in pixels of the fine maps.
    rows, cols = np.meshgrid(np.arange(size[ia]) * float(SLICE_REFINE),
                             np.arange(size[ib]) * float(SLICE_REFINE),
                             indexing="ij")
    field = np.empty(size, dtype=np.float32)
    planes = np.moveaxis(field, (ia, ib, idep), (0, 1, 2))   # a view
    for j in range(size[idep]):
        depth = origin[idep] + j * step
        k = int(np.searchsorted(depths, depth, side="right")) - 1
        if k < 0:
            parts, factor = [(0, 1.0)], 0.0
        elif k >= len(depths) - 1:
            parts, factor = [(len(depths) - 1, 1.0)], 0.0
        else:
            w = (depth - depths[k]) / (depths[k + 1] - depths[k])
            parts, factor = [(k, 1.0 - w), (k + 1, w)], align[k]
        centre = sum(w * centres[i] for i, w in parts)
        value = np.zeros(rows.shape, dtype=np.float32)
        for i, w in parts:
            shift = factor * (centre - centres[i]) / fine
            value += np.float32(w) * map_coordinates(
                maps[i], [rows - shift[0], cols - shift[1]], order=1,
                mode="nearest")
        planes[:, :, j] = value
    return origin, step, field, low, high


def _slice_distance(polygons: list, corner, spacing: float, shape: tuple):
    """The signed distance (µm, negative inside) of each point of a grid in
    a slice's plane to the edge of the slice's filled polygons, and the
    filled area's centre (µm). The grid's first point is at `corner` (µm,
    the plane's two axes), its points `spacing` apart.

    A polygon is filled as in Fiji (a point inside it an odd number of
    times is inside), so a lasso that overshoots its start, a figure-8 or a
    shaky line still gives a clean area. A polygon too thin to cover any
    grid point is a line: the shape narrows to nothing there.
    """
    mask = np.zeros(shape, dtype=bool)
    for flat in polygons:
        mask |= fill_polygon(shape, (np.asarray(flat) - corner) / spacing)
    # Holes and specks smaller than a cell of the shape's grid cannot be
    # drawn by it anyway; a shaky lasso crossing itself makes many (and
    # each would add points to _shape_function's fit). A slice's only
    # filled area is kept, however small (a tiny object drawn on purpose).
    for value in (True, False):                 # specks, then holes
        labels, _ = label(mask == value)
        small = np.bincount(labels.ravel()) <= SLICE_REFINE ** 2
        small[0] = False
        if value and small[1:].all():
            continue
        mask[small[labels]] = not value
    if mask.any():
        outside = distance_transform_edt(~mask) - 0.5
        inside = distance_transform_edt(mask) - 0.5
        distance = np.where(mask, -inside, outside) * spacing
        centre = corner + np.argwhere(mask).mean(axis=0) * spacing
    else:
        line = np.zeros(shape, dtype=bool)
        for flat in polygons:
            ring = _resample_ring(np.asarray(flat), spacing / 2) - corner
            index = np.clip(np.round(ring / spacing).astype(int), 0,
                            np.array(shape) - 1)
            line[index[:, 0], index[:, 1]] = True
        distance = distance_transform_edt(~line) * spacing
        centre = np.concatenate(polygons).mean(axis=0)
    return distance.astype(np.float32), centre


def _surface_field(frozen: tuple, spatial: tuple, detail: int):
    """The function of _shape_mask for polygons in several views (see
    _shape_function), on its grid: (origin, step, values, low, high) as
    _shape_field, before holes are filled. The function itself does not
    depend on the detail."""
    function, points, low, high = _shape_function(frozen, spatial)
    origin, step, size = _shape_grid(low, high, detail)
    # Evaluating the function is the slow part, and only its sign matters.
    # So it is evaluated on a grid 4 times coarser first, and then exactly
    # only near the edge of the shape: in coarse cells whose corners differ
    # in sign, or that hold a drawn polygon (and their neighbours).
    # Elsewhere the coarse values, interpolated, have the right sign.
    every = 4
    coarse_index = [np.unique(np.r_[np.arange(0, n, every), n - 1])
                    for n in size]
    coarse = function(_grid_points(origin, step, coarse_index)).reshape(
        [len(i) for i in coarse_index]).astype(np.float32)
    field = _sample_grid(coarse, [np.interp(np.arange(n), i, np.arange(len(i)))
                                  for n, i in zip(size, coarse_index)])
    near = np.zeros(coarse.shape, dtype=bool)
    cells = tuple(slice(0, n - 1) for n in coarse.shape)
    signs = []
    for corner in itertools.product((0, 1), repeat=3):
        signs.append(coarse[tuple(slice(c, c + n - 1) for c, n in
                                  zip(corner, coarse.shape))] < 0)
    near[cells] = np.any(signs, axis=0) & ~np.all(signs, axis=0)
    cell_of = [np.clip(np.searchsorted(i, (points[:, k] - origin[k]) / step,
                                       side="right") - 1, 0, len(i) - 1)
               for k, i in enumerate(coarse_index)]
    near[tuple(cell_of)] = True
    near = binary_dilation(near, iterations=1)
    fine = np.zeros(size, dtype=bool)
    for cell in np.argwhere(near):
        fine[tuple(slice(coarse_index[k][c],
                         coarse_index[k][min(c + 1, len(coarse_index[k]) - 1)]
                         + 1) for k, c in enumerate(cell))] = True
    where = np.nonzero(fine)
    exact = np.column_stack([origin[k] + where[k] * step for k in range(3)])
    field[where] = function(exact).astype(np.float32)
    return origin, step, field, low, high


def _fill_holes(field, origin, step, low, high):
    """Make the shape solid, on its grid (in place): fill closed hollows,
    and holes on slices along each axis, by giving those grid points a
    small negative value. The polygons are filled areas, so a hole in the
    shape is never drawn; holes appear where polygons of different views
    disagree (the shape must pass through both). An object that is truly
    a ring would have its hole filled too. field is on a grid of spacing
    step starting at origin; low and high are where the shape ends."""
    first = np.round((np.asarray(low) - origin) / step).astype(int)
    last = np.floor((np.asarray(high) - origin) / step + 1e-6).astype(int)
    within = np.zeros(field.shape, dtype=bool)
    within[tuple(slice(a, b + 1) for a, b in zip(first, last))] = True
    inside = (field < 0) & within
    solid = inside
    for _ in range(3):              # filling along one axis can close
        before = solid.sum()        # a hole seen along another
        for axis in range(3):
            plane = _in_plane_structure(axis)
            solid = binary_fill_holes(solid, structure=plane)
        solid = binary_fill_holes(solid)
        if solid.sum() == before:
            break
    # A filled point gets the depth it was outside (at least one step), so
    # that the slices between the grid's planes are filled too.
    filled = solid & ~inside
    field[filled] = -np.maximum(field[filled], np.float32(step))


def _in_plane_structure(axis: int) -> np.ndarray:
    """Neighbours within the plane across `axis` only (so that holes are
    filled slice by slice)."""
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    for other in range(3):
        if other != axis:
            for side in (0, 2):
                index = [1, 1, 1]
                index[other] = side
                structure[tuple(index)] = True
    return structure


def _grid_points(origin, step, index) -> np.ndarray:
    """The points (µm) of a grid given by its indices along each axis."""
    axes = [origin[k] + np.asarray(i) * step for k, i in enumerate(index)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _resample_ring(points, spacing: float) -> np.ndarray:
    """A closed polygon resampled at equal steps along its edge (about
    `spacing` apart, at least 12 points)."""
    ring = np.vstack([points, points[:1]])
    lengths = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(lengths)])
    n = max(int(round(along[-1] / spacing)), 12)
    targets = np.linspace(0.0, along[-1], n, endpoint=False)
    return np.column_stack([np.interp(targets, along, ring[:, k])
                            for k in range(2)])


def _edge_points(flat, step: float, spacing: float) -> list:
    """Points for _shape_function from one polygon (in its plane, µm): the
    edge of the filled polygon, `spacing` apart (value 0), and the points
    one step inside and outside it (value: their signed distance to the
    edge), as [(points, values), ...]. The distances are smoothed over
    half a step, as the shape's grid cannot follow a shakier edge."""
    flat = np.asarray(flat, dtype=float)
    fine = step / SLICE_REFINE
    corner = flat.min(axis=0) - 3 * step
    shape = tuple(int(n) for n in
                  np.ceil((np.ptp(flat, axis=0) + 6 * step) / fine) + 1)
    distance, _ = _slice_distance([flat], corner, fine, shape)
    if distance.min() >= 0:                     # no area: the line itself
        return _line_points(flat, step, spacing)
    distance = gaussian_filter(distance, SLICE_REFINE / 2)
    if distance.min() > -step:   # thinner than 2 steps: no inner side
        return _line_points(flat, step, spacing)
    gradient = np.gradient(distance)            # points outwards
    result = []
    for contour in find_contours(distance, 0.0):
        length = np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
        if length * fine < spacing:             # a speck: no edge to fit
            continue
        ring = _resample_ring(corner + contour * fine, spacing)
        pixel = (ring - corner) / fine
        normal = np.column_stack([map_coordinates(g, pixel.T, order=1,
                                                  mode="nearest")
                                  for g in gradient])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        result.append((ring, np.zeros(len(ring))))
        for side in (ring - step * normal, ring + step * normal):
            result.append((side, map_coordinates(
                distance, ((side - corner) / fine).T, order=1,
                mode="nearest").astype(float)))
    if not result:          # thinner than the smoothing (the second review
        return _line_points(flat, step, spacing)    # of 0.17.1: no points)
    return result


def _line_points(flat, step: float, spacing: float) -> list:
    """Points for _shape_function from a polygon too thin for a filled area
    on the shape's grid: as before 0.17.1, its line (value 0) and one step
    to either side along its normals (inside -step, outside +step), which
    makes a slab about two steps thick, so that the shape still passes
    through it. (Points on both sides marked outside, as the filled area
    would have them, left the slab positive between the points.)"""
    ring = _resample_ring(np.asarray(flat, dtype=float), spacing)
    normal = _outward_normals(ring)
    return [(ring, np.zeros(len(ring))),
            (ring - step * normal, np.full(len(ring), -step)),
            (ring + step * normal, np.full(len(ring), step))]


def _outward_normals(ring) -> np.ndarray:
    """Unit normals of a closed polygon's edge, pointing outwards (tested
    against the polygon itself, so the drawing direction does not matter)."""
    tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    size = float(np.ptp(ring, axis=0).max()) or 1.0
    if points_in_poly(ring + 1e-3 * size * normal, ring).mean() > 0.5:
        normal = -normal
    return normal


def _sample_grid(field, coordinates) -> np.ndarray:
    """field sampled at fractional grid positions, given per axis (the
    target points form a grid too, so the linear interpolation is done one
    axis at a time); +infinity (outside) beyond the grid."""
    out = field
    for axis, u in enumerate(coordinates):
        n = out.shape[axis]
        i0 = np.clip(np.floor(u).astype(int), 0, max(n - 2, 0))
        w = np.clip(u - i0, 0.0, 1.0).astype(np.float32)
        shape = [1] * out.ndim
        shape[axis] = len(u)
        a = np.take(out, i0, axis=axis)
        b = np.take(out, np.minimum(i0 + 1, n - 1), axis=axis)
        out = a + (b - a) * w.reshape(shape)
        beyond = ((u < -1e-9) | (u > n - 1 + 1e-9)).reshape(shape)
        if beyond.any():
            out = np.where(beyond, np.float32(np.inf), out)
    return out


def downsampling_factors(image: OmeZarrImage) -> list[list[int] | None]:
    """Integer downsampling factor of each level relative to level 0.

    A level gets None if its voxel size is not a whole multiple of level 0's,
    or if its shape does not match that factor (such levels cannot be cropped
    consistently). Example: a level 1 downsampled 8x in y and x only -> [1, 1, 8, 8].
    """
    level0 = image.levels[0]
    factors = []
    for level in image.levels:
        f = []
        for i, (s, s0) in enumerate(zip(level.scale, level0.scale)):
            if i == image.channel_axis:
                f.append(1)
                continue
            ratio = s / s0
            n = round(ratio)
            n0 = level0.array.shape[i]
            consistent = (n >= 1 and abs(ratio - n) < 1e-3 * n
                          and level.array.shape[i] in (n0 // n, -(-n0 // n)))
            if not consistent:
                f = None
                break
            f.append(n)
        factors.append(f)
    return factors


def plan_crop(image: OmeZarrImage, box: CropBox, min_size: int = 64,
              n_levels: int | None = None) -> CropPlan:
    """Work out which voxels of which levels to copy for one box.

    min_size: keep pyramid levels while the crop is at least this many voxels
              along its longest spatial axis (like eubi-bridge's
              min_dimension_size). Fewer levels mean finer snapping.
    n_levels: keep exactly this many levels instead (if the source has them).
    """
    names = image.axis_names
    c_axis = image.channel_axis
    for axis in box.ranges:
        if axis not in image.range_axes:
            raise ValueError(f"Box {box.name!r}: unknown axis {axis!r}; "
                             f"image axes are {image.range_axes}.")
    # Every n-th timepoint (0.11.0): a whole number >= 1, and only along a
    # time axis.
    t_axis, t_step = time_axis(image), box.t_step
    if isinstance(t_step, float) and t_step.is_integer():
        t_step = int(t_step)
    if not isinstance(t_step, int) or isinstance(t_step, bool) or t_step < 1:
        raise ValueError(f"Box {box.name!r}: the time step must be a whole "
                         f"number of 1 or more (every n-th timepoint), not "
                         f"{box.t_step!r}.")
    if t_step > 1 and t_axis is None:
        raise ValueError(f"Box {box.name!r}: every {_ordinal(t_step)} "
                         "timepoint was asked, but the image has no time "
                         "axis.")
    spatial = [i for i, a in enumerate(image.axes) if a.get("type") == "space"]
    spatial = spatial or [i for i in range(len(names)) if i != c_axis]
    # A region that moves over time (0.17.0): its axes that move (the
    # spatial axes it has ranges on) are not clipped to the image; a
    # timepoint's part outside the image is written as the fill value (0).
    moving = []
    if box.track:
        if t_axis is None:
            raise ValueError(f"Box {box.name!r} moves over time, but the "
                             "image has no time axis.")
        if box.outlines:
            raise ValueError(f"Box {box.name!r} moves over time and has "
                             "outlines (polygons) that stay in place: that "
                             "is not possible yet. Fix the region in place "
                             "first, or remove its polygons.")
        moving = _moving_axes(image, box)
    if box.timed_outlines:
        if not box.track:
            raise ValueError(f"Box {box.name!r} has outlines per timepoint, "
                             "which are for regions that move over time, "
                             "but it does not move.")
        for outlines in box.timed_outlines.values():
            _check_outlines(CropBox(box.name, outlines=outlines),
                            _spatial_axes(image))

    # 1. Requested physical box -> level-0 voxel range (rounded outward).
    #    Outlines shrink the box to their bounding box.
    ranges = _effective_ranges(image, box)
    level0 = image.levels[0]
    shape0 = level0.array.shape
    start0, stop0 = [], []
    for i, name in enumerate(names):
        if i == c_axis or name not in ranges:
            start0.append(0)
            stop0.append(shape0[i])
            continue
        low, high = sorted(ranges[name])
        first, end = physical_to_index_range(low, high, level0.scale[i],
                                             level0.translation[i])
        if i not in moving:
            first, end = max(first, 0), min(end, shape0[i])   # clip to image
        if end <= first:
            raise ValueError(f"Box {box.name!r} lies outside the image "
                             f"along {name}.")
        start0.append(first)
        stop0.append(end)

    # 2. Choose the levels to keep. Each kept level's factor must divide the
    #    next one's, so that one snapped box is exact on every kept level.
    factors = downsampling_factors(image)
    keep = [0]
    for level_index in range(1, len(image.levels)):
        f, f_prev = factors[level_index], factors[keep[-1]]
        if f is None or any(a % b for a, b in zip(f, f_prev)):
            break
        # A level that also shrinks time cannot keep every n-th timepoint
        # or follow a moving region (its timepoints are not those of level
        # 0): the crop ends before.
        if (t_step > 1 or moving) and f[t_axis] != 1:
            break
        if n_levels is not None:
            if len(keep) >= n_levels:
                break
        else:
            longest = max(-(-(stop0[i] - start0[i]) // f[i]) for i in spatial)
            if longest < min_size:
                break
        keep.append(level_index)

    # 3. Snap the level-0 range outward to the coarsest kept level's grid.
    f_max = factors[keep[-1]]
    for i in range(len(names)):
        if i != c_axis:
            if i in moving:
                # (a moving region: widened by half a grid step on each
                # side, because each timepoint's shift is rounded to the
                # grid (3b): so the region asked still lies inside the crop
                # at every timepoint)
                start0[i] -= f_max[i] // 2
                stop0[i] += f_max[i] // 2
            start0[i] = (start0[i] // f_max[i]) * f_max[i]
            stop0[i] = -(-stop0[i] // f_max[i]) * f_max[i]
            if i not in moving:
                stop0[i] = min(stop0[i], shape0[i])
    # Every n-th timepoint: the range then ends at the last one kept
    # (frames start, start + n, ...; stop is exclusive).
    steps = [1] * len(names)
    if t_step > 1:
        steps[t_axis] = t_step
        count = (stop0[t_axis] - start0[t_axis] - 1) // t_step + 1
        stop0[t_axis] = start0[t_axis] + (count - 1) * t_step + 1

    # 3b. A moving region: each output timepoint's shift (level-0 voxels),
    #     rounded to whole voxels of the coarsest kept level, so that the
    #     region stays exact on every kept level. The plan's ranges are the
    #     region at the first output timepoint; the shifts are relative to it.
    shifts0 = times = None
    if moving:
        times = [_level0_time(image, s) for s in
                 range(start0[t_axis], stop0[t_axis], steps[t_axis])]
        shifts0 = np.array([_shift_voxels(image, box, time, f_max, moving)
                            for time in times], dtype=np.int64)
        for j, time in enumerate(times):
            for i in moving:
                if (start0[i] + shifts0[j, i] >= shape0[i]
                        or stop0[i] + shifts0[j, i] <= 0):
                    raise ValueError(
                        f"Box {box.name!r} lies outside the image along "
                        f"{names[i]} at {names[t_axis]} = {time:g}.")
        for i in moving:
            start0[i] += int(shifts0[0, i])
            stop0[i] += int(shifts0[0, i])
        shifts0 = shifts0 - shifts0[0]

    # 4. Voxel ranges on every kept level.
    if c_axis is None:
        if box.channels:
            raise ValueError("This image has no channel axis.")
        channels = None
    elif box.channels is None:
        channels = list(range(image.n_channels))
    else:
        channels = [int(c) for c in box.channels]
        bad = [c for c in channels if not 0 <= c < image.n_channels]
        if bad or not channels:
            raise ValueError(f"Box {box.name!r}: invalid channels {bad or []}"
                             f"; the image has {image.n_channels}.")

    starts, stops, shapes, shifts = [], [], [], []
    for level_index in keep:
        f = factors[level_index]
        shape = image.levels[level_index].array.shape
        first, end = [], []
        for i in range(len(names)):
            if i == c_axis:
                first.append(0)
                end.append(shape[i])
            elif i in moving:
                # (whole voxels of this level: the range and the shifts are
                # on the coarsest kept level's grid; not clipped)
                first.append(start0[i] // f[i])
                end.append(stop0[i] // f[i])
            else:
                first.append(start0[i] // f[i])
                end.append(min(-(-stop0[i] // f[i]), shape[i]))
        # (with a step: the kept voxels first, first + step, ... < end)
        out_shape = [-(-(b - a) // s) for a, b, s in zip(first, end, steps)]
        if c_axis is not None:
            out_shape[c_axis] = len(channels)
        starts.append(first)
        stops.append(end)
        shapes.append(tuple(out_shape))
        if moving:
            shifts.append(shifts0 // np.asarray(f, dtype=np.int64))

    snapped = {}
    for i, name in enumerate(names):
        if i != c_axis:
            a, b = index_range_to_physical(start0[i], stop0[i],
                                           level0.scale[i],
                                           level0.translation[i])
            snapped[name] = (round(a, 9), round(b, 9))
    if not moving:
        return CropPlan(box, keep, starts, stops, shapes, channels, snapped,
                        steps)
    return CropPlan(box, keep, starts, stops, shapes, channels, snapped,
                    steps, shifts=shifts, times=times, t_axis=t_axis)


def _level0_time(image: OmeZarrImage, index: int) -> float:
    """The time of level 0's timepoint `index` (the time axis' unit)."""
    t_axis = time_axis(image)
    level0 = image.levels[0]
    return level0.translation[t_axis] + index * level0.scale[t_axis]


def _moving_axes(image: OmeZarrImage, box: CropBox) -> list[int]:
    """The axes a moving region moves on: the spatial axes it has ranges on
    (as plan_crop chooses them)."""
    t_axis = time_axis(image)
    if not box.track or t_axis is None:
        return []
    return [i for i, a in enumerate(image.axes)
            if a.get("type") == "space" and a["name"] in box.ranges]


def _shift_voxels(image: OmeZarrImage, box: CropBox, time: float,
                  grid: list[int], moving: list[int]) -> list[int]:
    """A moving region's shift at a time in level-0 voxels per axis,
    rounded to whole multiples of `grid` (the coarsest kept level's
    downsampling factors); 0 on the axes that do not move."""
    shift = track_shift(box.track, time)
    level0 = image.levels[0]
    result = [0] * len(image.axis_names)
    for i in moving:
        step = level0.scale[i] * grid[i]
        result[i] = int(round(shift.get(image.axis_names[i], 0.0) / step)) \
            * grid[i]
    return result


def snapped_ranges_at(image: OmeZarrImage, plan: CropPlan,
                      time: float) -> dict[str, tuple[float, float]]:
    """The ranges as cropped (plan.snapped_ranges) at a time: for a moving
    region, moved as plan_crop moves it (whole voxels of the coarsest kept
    level), also at timepoints the crop skips (every n-th timepoint)."""
    if not plan.moving:
        return dict(plan.snapped_ranges)
    grid = downsampling_factors(image)[plan.level_indices[-1]]
    moving = _moving_axes(image, plan.box)
    first = _shift_voxels(image, plan.box, plan.times[0], grid, moving)
    now = _shift_voxels(image, plan.box, time, grid, moving)
    scale = image.levels[0].scale
    ranges = dict(plan.snapped_ranges)
    for i in moving:
        name = image.axis_names[i]
        delta = (now[i] - first[i]) * scale[i]
        low, high = ranges[name]
        ranges[name] = (round(low + delta, 9), round(high + delta, 9))
    return ranges


# ---------------------------------------------------------------------------
# Writing the crop
# ---------------------------------------------------------------------------

def output_name(box_name: str) -> str:
    """The part of an output's file name that comes from the box name.

    Letters, digits and "_.-" are kept; everything else (spaces, other
    punctuation, accented letters) becomes "_", so that the name is safe on
    Windows, Mac and Linux. A name with nothing left becomes "crop".
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", box_name).strip("_") or "crop"


# File name endings removed from a source's name for its crops' names
# (the zipped ones since 0.21.0).
_SOURCE_SUFFIXES = (".companion.ome.xml", ".companion.ome", ".ome.tiff",
                    ".ome.tif", ".ome.zarr.zip", ".zarr.zip", ".ozx", ".zip",
                    ".zarr",
                    ".tiff", ".tif", ".btf", ".tf8", ".tf2", ".lif", ".lof",
                    ".xlef", ".czi", ".nd2", ".oir")

# Endings of crops (and of the TIFFs made from them), removed to name the
# files that go with a crop: <crop>_level<k>.tif, <crop>_positions.csv.
_CROP_SUFFIXES = (".ome.zarr.zip", ".zarr.zip", ".ozx", ".zip", ".ome.zarr",
                  ".zarr", ".ome.tiff", ".ome.tif", ".tiff", ".tif")


def crop_stem(crop_path: str | Path) -> str:
    """A crop's file name without its ending (.zarr, .ozx, .zip, .tif ...),
    e.g. "pos3_embryo1" for pos3_embryo1.ozx."""
    name = Path(crop_path).name
    for suffix in _CROP_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return name


def image_stem(image: OmeZarrImage) -> str:
    """The start of the image's crop names: its file name without the
    ending (.zarr, .tif, .ome.tif ...), plus the image's name in the file
    when the file holds several (e.g. a LIF's series)."""
    name = image.path.name
    for suffix in _SOURCE_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    source = image.source or {}
    if source.get("scenes_in_file", 1) > 1:
        name += "_" + output_name(str(source.get("scene", "")))
    if image.inner:        # (0.21.0) an image of a collection, e.g. "_0"
        name += "_" + output_name(image.inner)
    return name


def output_suffix(image: OmeZarrImage) -> str:
    """".zarr" for an OME-Zarr image's crops; for a file read into memory
    ".ome.tif" (OME-TIFF and other microscope files) or ".tif" (Fiji
    TIFFs and plain TIFFs): crops keep the kind of file they came from.

    A zipped OME-Zarr's crops are zipped too (0.21.0, the user's decisions
    2a and 3a of 2026-11-22): ".ozx" (RFC-9) for OME-Zarr 0.5 and 0.6;
    0.4 (Zarr v2, which an .ozx cannot hold) keeps the source's ending,
    ".zip" or ".ome.zarr.zip"."""
    if image.source is None:
        if image.container is None:
            return ".zarr"
        if image.zarr_format == 3:
            return ".ozx"
        name = image.path.name.lower()
        for suffix in (".ome.zarr.zip", ".zarr.zip"):
            if name.endswith(suffix):
                return suffix
        return ".zip"
    return ".ome.tif" if image.source.get("output") == "ome" else ".tif"


def output_path_for(image: OmeZarrImage, box: CropBox,
                    out_dir: str | Path) -> Path:
    """Default output name: <source name>_<box name>.zarr in out_dir (.tif
    or .ome.tif for a file read into memory, see output_suffix)."""
    return Path(out_dir) / (f"{image_stem(image)}_{output_name(box.name)}"
                            f"{output_suffix(image)}")


def same_file_name(name_a: str, name_b: str) -> bool:
    """True if boxes with these names would be saved under one file name.

    Windows and Mac ignore upper and lower case in file names, so "tail 1",
    "tail_1" and "Tail_1" all end up in the same output folder.
    """
    return output_name(name_a).casefold() == output_name(name_b).casefold()


def name_clashes(names: list[str]) -> list[list[str]]:
    """Groups of box names that would be saved under the same file name
    (one list per clash; an empty list means no clashes)."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(output_name(name).casefold(), []).append(name)
    return [group for group in groups.values() if len(group) > 1]


def check_box_names(boxes: list[CropBox]) -> None:
    """Raise ValueError if two boxes would be saved under the same name, so
    that one crop would silently replace the other."""
    clashes = name_clashes([box.name for box in boxes])
    if clashes:
        lines = "\n".join("  " + " and ".join(repr(name) for name in group)
                          for group in clashes)
        raise ValueError(
            "These boxes would be saved under the same file name, so one "
            f"crop would replace the other:\n{lines}\nBox names become file "
            "names, where spaces and most punctuation turn into '_' and "
            "upper and lower case count as the same. Please rename them.")


def check_output_folder(image: OmeZarrImage, out_dir: str | Path) -> None:
    """Refuse an output folder that is the source image or lies inside it.

    Call it before creating the folder: crops must never add anything to
    the raw data, not even an empty folder.
    """
    src, out = image.path.resolve(), Path(out_dir).resolve()
    if out == src or src in out.parents:
        raise ValueError(
            f"The output folder {out_dir} is inside the source image "
            f"{image.path}. Crops must not be written into the raw data; "
            "please choose a folder outside it.")


def check_output(image: OmeZarrImage, box: CropBox, out_path: str | Path,
                 overwrite: bool = False) -> None:
    """Raise the error write_crop would raise for this output, so that all
    boxes can be checked before the first one is written. (For a file read
    into memory the output is a TIFF: see check_tiff_output.)"""
    _check_positions_output(Path(out_path))
    if image.in_memory:
        if Path(out_path).resolve() == image.path.resolve():
            raise ValueError(f"The crop {out_path} would replace its source.")
        check_tiff_output(out_path, overwrite, box.name, image=image)
        return
    _check_output_path(image, Path(out_path), overwrite, box.name)


def write_crop(image: OmeZarrImage, plan: CropPlan, out_path: str | Path, *,
               keep_position: bool = True,
               channel_labels: list[str] | None = None,
               workers: int = 8, overwrite: bool = False,
               progress: Callable[[int, int], None] | None = None,
               progress_bytes: Callable[[int, int], None] | None = None,
               cancel_event: threading.Event | None = None) -> Path:
    """Write one crop as a new OME-Zarr in the same format as the source
    (the same OME-Zarr version too: 0.4 stays 0.4, 0.5 stays 0.5).

    keep_position:  add a translation so the crop overlays the full image
                    (False: the crop starts at 0).
    channel_labels: optional new names for the kept channels.
    workers:        number of blocks copied in parallel.
    overwrite:      replace an existing output, but only one made by this
                    tool (raw data are never deleted).
    progress:       called as progress(done, total) after each block (a
                    shard, or without shards a group of whole chunks).
    progress_bytes: called as progress_bytes(done, total) after each block,
                    in bytes of image data (uncompressed). Blocks differ in
                    size (the finest level's are the largest), so this is
                    the better measure for a time estimate.
    cancel_event:   set it (from another thread) to stop the crop.

    An out_path ending in .ozx or .zip (0.21.0, see output_suffix) makes a
    zipped crop: written as a folder (see _zip_work_folder), then packed
    (pack_zip). crop_box also checks it before packing and stores the
    result in it; verify_crop on the zip can store it only by writing the
    zip again.
    """
    out_path = Path(out_path)
    _check_output_path(image, out_path, overwrite, plan.box.name)
    if is_zipped(out_path):
        with _zip_work_folder(image, plan, out_path) as folder:
            _write_unzipped(image, plan, folder, out_path,
                            keep_position=keep_position,
                            channel_labels=channel_labels, workers=workers,
                            progress=progress, progress_bytes=progress_bytes,
                            cancel_event=cancel_event)
            pack_zip(folder, out_path, cancel_event=cancel_event)
        return out_path
    root = zarr.open_group(out_path, mode="w" if overwrite else "w-",
                           zarr_format=image.zarr_format)
    _write_crop_into(root, image, plan, keep_position=keep_position,
                     channel_labels=channel_labels, workers=workers,
                     progress=progress, progress_bytes=progress_bytes,
                     cancel_event=cancel_event, where=str(out_path))
    return out_path


def _write_unzipped(image: OmeZarrImage, plan: CropPlan, folder: Path,
                    out_path: Path, **options) -> None:
    """Write a zipped crop's folder (write_crop's work, before pack_zip);
    a Cancel says that the zip was not saved."""
    root = zarr.open_group(folder, mode="w-", zarr_format=image.zarr_format)
    try:
        _write_crop_into(root, image, plan, where=out_path.name, **options)
    except CropCancelled:
        raise CropCancelled(
            f"Cancelled; {out_path.name} was not saved.") from None


@contextlib.contextmanager
def _zip_work_folder(image: OmeZarrImage, plan: CropPlan, out_path: Path):
    """The folder a zipped crop is written into before it is packed
    (0.21.0): in this computer's temporary folder when that has room for
    the crop (the user's decision 4a of 2026-11-22: with the output on a
    network drive, a folder next to it would send the crop over the
    network twice), else in a hidden scratch folder next to the output.
    Removed afterwards, also after an error or Cancel."""
    import tempfile

    need = sum(nbytes(shape, image.levels[0].array.dtype)
               for shape in plan.shapes)        # (uncompressed: at most)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parent = zip_work_parent(out_path, need)
    parent.mkdir(parents=True, exist_ok=True)
    clear_old_scratch(parent)
    tmp = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=parent))
    try:
        yield tmp / f"{crop_stem(out_path)}.zarr"
    finally:
        remove_scratch(tmp)


# Space left free on the drive of the temporary folder by a zipped crop's
# folder (see _zip_work_folder).
ZIP_WORK_MARGIN = 2**30


def zip_work_parent(out_path: str | Path, need_bytes: int) -> Path:
    """Where a zipped crop's folder goes (see _zip_work_folder): the
    temporary folder if its drive has need_bytes free (plus
    ZIP_WORK_MARGIN), else the output's folder."""
    import tempfile

    tmp = Path(tempfile.gettempdir())
    try:
        free = shutil.disk_usage(tmp).free
    except OSError:
        free = 0
    if free >= need_bytes + ZIP_WORK_MARGIN:
        return tmp
    return Path(out_path).parent


def _write_crop_into(root: zarr.Group, image: OmeZarrImage, plan: CropPlan,
                     *, keep_position: bool, channel_labels, workers: int,
                     progress, progress_bytes, cancel_event,
                     where: str) -> None:
    """write_crop's work, into an open, empty group (a folder, or memory
    for the crop of a file's image; `where` names it in messages)."""
    # Mark the folder as ours straight away, so that an interrupted crop can
    # be replaced later with overwrite=True. "complete" turns True at the end.
    provenance = _provenance(image, plan, keep_position)
    root.attrs["crop_provenance"] = {**provenance, "complete": False}

    # 1. Create empty arrays with the source's format, level by level.
    jobs = []      # one (source, destination, block, offset) per output block
    spatial = [i for i, axis in enumerate(image.axes)
               if axis.get("type") == "space"]
    for k, level_index in enumerate(plan.level_indices):
        level = image.levels[level_index]
        src = level.array
        dst = root.create_array(name=level.path, shape=plan.shapes[k],
                                **_same_format(src))
        block_shape = dst.shards or _copy_block(dst.chunks, dst.shape,
                                                dst.dtype, spatial)
        for block in _iter_blocks(dst.shape, block_shape):
            jobs.append((src, dst, block, k, level))

    # 2. Copy the data: each task reads the matching source region and
    #    writes one whole output shard (or, without shards, a block of whole
    #    chunks), so no two tasks ever write into the same file. With
    #    outlines, voxels outside them become the fill value (0); a block
    #    entirely outside is skipped, and its chunks are simply never
    #    written (zarr reads them as 0).
    def copy_one(job):
        src, dst, block, k, level = job
        if plan.moving and plan.box.timed_outlines:
            # (0.17.0) each timepoint's own outlines, at its place
            mask = _timed_block_mask(image, plan, k, block, lambda j: [
                int(s + d) for s, d in zip(plan.starts[k], plan.shifts[k][j])])
        else:
            mask = _block_mask(image, plan, level, block, plan.starts[k])
        if mask is not None and not mask.any():
            return
        data = _read_source(src, block, plan, k, image.channel_axis)
        if mask is not None and not mask.all():
            data = np.where(mask, data, src.fill_value)
        dst[block] = data

    # Bytes of image data per block (uncompressed), for progress_bytes.
    sizes = [nbytes([s.stop - s.start for s in block], dst.dtype)
             for _, dst, block, _, _ in jobs]
    total_bytes, done_bytes = sum(sizes), 0
    prepare_shape(image, plan.box)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(copy_one, job): size
                   for job, size in zip(jobs, sizes)}
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()          # re-raises any error from the worker
                done_bytes += futures[future]
                if progress is not None:
                    progress(done, len(jobs))
                if progress_bytes is not None:
                    progress_bytes(done_bytes, total_bytes)
                if cancel_event is not None and cancel_event.is_set():
                    raise CropCancelled(f"Cancelled; {where} is "
                                        "incomplete (it has no OME metadata).")
        except BaseException:
            # An error, Cancel or Ctrl+C stops the blocks still queued (the
            # pool would otherwise finish all of them before stopping).
            for f in futures:
                f.cancel()
            raise

    # 3. Metadata last: OME multiscales, channels and provenance.
    ome = _cropped_ome_metadata(image, plan, keep_position, channel_labels)
    root.attrs.update({
        **(ome if image.zarr_format == 2 else {"ome": ome}),
        "crop_provenance": {**provenance, "complete": True},
    })


def _same_format(src: zarr.Array) -> dict:
    """create_array() settings that give a new array the storage format of
    `src`: Zarr v3 (OME-Zarr 0.5) or v2 (0.4), with the same data type,
    chunks, shards, compression and chunk file names."""
    common = {"dtype": src.dtype, "chunks": src.chunks,
              "fill_value": src.fill_value}
    if src.metadata.zarr_format == 2:
        # zarr reports "no compressor" and "no filters" as (), but expects
        # None for them when creating a v2 array.
        return {**common, "compressors": src.compressors or None,
                "filters": src.filters or None, "order": src.order,
                "chunk_key_encoding": {
                    "name": "v2",
                    "separator": src.metadata.dimension_separator}}
    return {**common, "shards": src.shards, "filters": src.filters,
            "compressors": src.compressors, "serializer": src.serializer,
            "dimension_names": src.metadata.dimension_names,
            "chunk_key_encoding": src.metadata.chunk_key_encoding}


def _copy_block(chunks, shape, dtype, grow_axes) -> tuple[int, ...]:
    """The block one copy task writes when the array has no shards: a whole
    number of chunks along every axis (so that no two tasks ever write the
    same chunk file), grown along the spatial axes `grow_axes` until it
    holds about BLOCK_BYTES. Copying one small chunk per task would make
    thousands of tasks for a large crop."""
    block = list(chunks)
    while nbytes(block, dtype) * 2 <= BLOCK_BYTES:
        # Double the block along the axis that still has the most blocks
        # (keeps it compact); stop when it covers the array.
        open_axes = [i for i in grow_axes if block[i] < shape[i]]
        if not open_axes:
            break
        i = max(open_axes, key=lambda a: shape[a] / block[a])
        block[i] *= 2
    return tuple(block)


# Random sample blocks per level whose voxels verify_crop compares (plus the
# first and the last corner of every level).
SAMPLE_BLOCKS = 4


def verify_crop(image: OmeZarrImage, plan: CropPlan, out_path: str | Path,
                *, keep_position: bool | None = None,
                channel_labels: list[str] | None = None,
                n_blocks: int = SAMPLE_BLOCKS, block_size: int = 64,
                seed: int = 0, record: bool = True) -> list[str]:
    """Check a written crop against the source; return a list of problems
    (an empty list means OK).

    It checks:
    - the metadata: the axes, the levels, and every level's voxel size and
      position (scale and translation), recomputed here from the plan,
      separately from the code that wrote them, so that a crop placed at
      the wrong position is found; the channel names; and that the crop was
      finished;
    - every level's format (shape, dtype, chunks, shards, codecs; for Zarr
      v2: compressor, filters, order, chunk file names), and that the crop
      has the source's OME-Zarr version;
    - the voxels of sample blocks: n_blocks random ones per level (fixed
      seed, so the check is reproducible) plus the first and the last
      corner, where edge mistakes would show. This is a spot check, not a
      comparison of every voxel.

    keep_position, channel_labels: as given to write_crop (keep_position
    None: as recorded in the crop). record: store the result in the crop's
    record ("verified" in crop_provenance), also when the check fails.
    """
    out = open_image(out_path)
    try:
        problems = _crop_problems(image, plan, out,
                                  keep_position=keep_position,
                                  channel_labels=channel_labels,
                                  n_blocks=n_blocks, block_size=block_size,
                                  seed=seed)
        provenance = copy.deepcopy(out.attrs.get("crop_provenance", {}))
    finally:
        out.close()
    if record:
        provenance["verified"] = {
            "ok": not problems,
            "checked": verification_scope(plan, n_blocks),
            "problems": problems[:20],
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if is_zipped(out_path):        # (0.21.0: the zip is written again)
            _rewrite_zip_record(Path(out_path), provenance)
        else:
            group = zarr.open_group(out_path, mode="r+")
            group.attrs["crop_provenance"] = provenance
    return problems


def _crop_problems(image: OmeZarrImage, plan: CropPlan, out: OmeZarrImage, *,
                   keep_position: bool | None, channel_labels,
                   n_blocks: int, block_size: int, seed: int) -> list[str]:
    """verify_crop's checks of an opened crop (on disk or in memory)."""
    provenance = out.attrs.get("crop_provenance", {})
    if keep_position is None:
        keep_position = provenance.get("keep_position", True)
    problems = []
    if provenance.get("complete") is not True:
        problems.append("the crop is not marked as complete")
    if len(out.levels) != len(plan.level_indices):
        problems.append(f"expected {len(plan.level_indices)} levels, "
                        f"found {len(out.levels)}")
    else:
        problems += _metadata_problems(image, plan, out, keep_position,
                                       channel_labels)
        problems += _voxel_problems(image, plan, out, n_blocks, block_size,
                                    seed)
    return problems


def verification_scope(plan: CropPlan, n_blocks: int = SAMPLE_BLOCKS) -> str:
    """What verify_crop checks, in words."""
    n = len(plan.level_indices)
    return (f"position, voxel size and format of {n} level"
            f"{'' if n == 1 else 's'}, and the voxels of {n_blocks + 2} "
            "sample blocks per level")


def verification_text(plan: CropPlan, problems: list[str],
                      n_blocks: int = SAMPLE_BLOCKS) -> str:
    """One line on the result of verify_crop, for the person."""
    scope = verification_scope(plan, n_blocks)
    if not problems:
        return f"checked OK ({scope})"
    more = f" (and {len(problems) - 3} more)" if len(problems) > 3 else ""
    return f"CHECK FAILED ({scope}): " + "; ".join(problems[:3]) + more


def _metadata_problems(image, plan, out, keep_position, channel_labels):
    """Compare the crop's metadata with what the plan says they must be."""
    problems = []
    if out.ome_format != image.ome_format:
        problems.append(f"the crop is {out.ome_format}, the source "
                        f"{image.ome_format}")
    if out.axes != image.axes:
        problems.append("the axes differ from the source")
    if out.coordinate_system != image.coordinate_system:
        problems.append(f"the crop's coordinate system is "
                        f"{out.coordinate_system!r}, the source's "
                        f"{image.coordinate_system!r}")
    c_axis = image.channel_axis

    def first_voxel(k):
        """Position of level k's first cropped voxel in the source."""
        level = image.levels[plan.level_indices[k]]
        return [t if i == c_axis else t + plan.starts[k][i] * s
                for i, (t, s) in enumerate(zip(level.translation,
                                               level.scale))]

    # Without keep_position the crop starts at 0: every level moves by the
    # position of level 0's first voxel.
    shift = [0.0] * len(image.axes)
    if not keep_position:
        shift = [0.0 if i == c_axis else p
                 for i, p in enumerate(first_voxel(0))]
    for k, level_index in enumerate(plan.level_indices):
        source, crop = image.levels[level_index], out.levels[k]
        name = f"level {crop.path}"
        expected = [p - d for p, d in zip(first_voxel(k), shift)]
        # (every n-th timepoint: n times the source's frame interval)
        scale = [s * plan.step(i) for i, s in enumerate(source.scale)]
        if not _close(crop.scale, scale):
            problems.append(f"{name}: voxel size {crop.scale} should be "
                            f"{scale}")
        if not _close(crop.translation, expected):
            problems.append(f"{name}: position (translation) "
                            f"{crop.translation} should be "
                            f"{[round(v, 9) for v in expected]}")

    # Channel names: those of the kept source channels, or the new ones.
    source_channels = image.ome.get("omero", {}).get("channels", [])
    if c_axis is not None and len(source_channels) == image.n_channels:
        expected = [source_channels[c].get("label", "")
                    for c in plan.channels]
        for j, label in enumerate((channel_labels or [])[:len(expected)]):
            expected[j] = label
        found = [ch.get("label", "") for ch in
                 out.ome.get("omero", {}).get("channels", [])]
        if found != expected:
            problems.append(f"channel names {found} should be {expected}")
    return problems


def _is_nan(value) -> bool:
    """True for a NaN number (False for anything that is not a number)."""
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _close(values, expected) -> bool:
    """Equal up to the rounding that write_crop uses (9 decimals)."""
    return len(values) == len(expected) and all(
        abs(a - b) <= 1e-6 * max(1.0, abs(b))
        for a, b in zip(values, expected))


def _voxel_problems(image, plan, out, n_blocks, block_size, seed):
    """Every level's format, and the voxels of its sample blocks."""
    rng = np.random.default_rng(seed)   # fixed seed: reproducible checks
    problems = []
    for k, level_index in enumerate(plan.level_indices):
        src = image.levels[level_index].array
        dst = out.levels[k].array
        name = f"level {out.levels[k].path}"
        if dst.shape != plan.shapes[k]:
            problems.append(f"{name}: shape {dst.shape} != {plan.shapes[k]}")
            continue
        for attr in ("dtype", "chunks", "shards", "fill_value"):
            a, b = getattr(dst, attr), getattr(src, attr)
            # (a float image's fill value is often NaN, and NaN != NaN)
            if a != b and not (attr == "fill_value" and _is_nan(a)
                               and _is_nan(b)):
                problems.append(f"{name}: {attr} differs from the source")
        differ = _format_differences(src, dst)
        if differ:
            problems.append(f"{name}: {', '.join(differ)} differ"
                            f"{'s' if len(differ) == 1 else ''} from the "
                            "source")

        size = [1 if i == image.channel_axis else min(block_size, n)
                for i, n in enumerate(dst.shape)]
        corners = [[int(rng.integers(0, n - s + 1))
                    for n, s in zip(dst.shape, size)] for _ in range(n_blocks)]
        corners.append([0] * dst.ndim)                          # first
        corners.append([n - s for n, s in zip(dst.shape, size)])  # last
        level = image.levels[level_index]
        for corner in corners:
            block = tuple(slice(c, c + s) for c, s in zip(corner, size))
            if plan.moving:
                source = _expected_moving_block(image, plan, k, block)
            else:
                source = src.oindex[_source_selection(block, plan.starts[k],
                                                      plan,
                                                      image.channel_axis)]
            if plan.moving and plan.box.timed_outlines:
                mask = _timed_block_mask(
                    image, plan, k, block,
                    lambda j: _place_from_ranges(image, plan, k, j))
            else:
                mask = _block_mask(image, plan, level, block,
                                   plan.starts[k])
            if mask is not None:             # outside the outlines: 0
                source = np.where(mask, source, src.fill_value)
            # equal_nan: a NaN fill value (float images) counts as equal
            if not np.array_equal(dst[block], source, equal_nan=True):
                problems.append(f"{name}: voxels differ in the block at "
                                f"{corner}")
    return problems


def _expected_moving_block(image, plan, k, block):
    """For verify_crop: what one block of a moving region's crop must hold.

    Worked out apart from the writer (which reads at plan.shifts): each
    output timepoint's place comes from the region's ranges then
    (snapped_ranges_at, in µm), turned into level-0 voxels and then into
    voxels of level k; the part outside the source is the fill value."""
    level = image.levels[plan.level_indices[k]]
    src = level.array
    fill = src.fill_value if src.fill_value is not None else 0
    t, c = plan.t_axis, image.channel_axis
    pieces = []
    for j in range(block[t].start, block[t].stop):
        place = _place_from_ranges(image, plan, k, j)
        selection, padding, inside = [], [], True
        for i, sl in enumerate(block):
            n = sl.stop - sl.start
            if i == c:
                selection.append(plan.channels[sl.start:sl.stop])
                padding.append((0, 0))
                continue
            if i == t:
                first = plan.starts[k][t] + j * plan.step(t)
                selection.append(slice(first, first + 1))
                padding.append((0, 0))
                continue
            first = place[i]
            a, b = first + sl.start, first + sl.start + n
            low_i, high_i = max(a, 0), min(b, src.shape[i])
            if high_i <= low_i:
                inside = False
                low_i = high_i = 0
            selection.append(slice(low_i, high_i))
            padding.append((low_i - a, b - high_i))
        shape = [sl.stop - sl.start for sl in block]
        shape[t] = 1
        if not inside:
            pieces.append(np.full(shape, fill, dtype=src.dtype))
            continue
        piece = np.pad(src.oindex[tuple(selection)], padding,
                       constant_values=fill)
        pieces.append(piece)
    return np.concatenate(pieces, axis=t)


def _place_from_ranges(image, plan, k, j) -> list[int]:
    """For verify_crop: the level-k voxel index of a moving region's first
    voxel per axis at output timepoint j, from its ranges then
    (snapped_ranges_at, µm -> level-0 voxels -> level k's), not from
    plan.shifts."""
    level0 = image.levels[0]
    factors = downsampling_factors(image)[plan.level_indices[k]]
    moving = _moving_axes(image, plan.box)
    ranges = snapped_ranges_at(image, plan, plan.times[j])
    place = list(plan.starts[k])
    for i in moving:
        low = ranges[image.axis_names[i]][0]
        first0 = round((low - level0.translation[i]) / level0.scale[i] + 0.5)
        place[i] = first0 // factors[i]
    return place


def _timed_block_mask(image, plan, k, block, place):
    """For a moving region with outlines per timepoint (0.17.0): which
    voxels of one output block of kept level k are kept. Per output
    timepoint j: its own outlines, at the region's place then (place(j):
    the level-k index of the region's first voxel per axis); a timepoint
    without outlines keeps all. None if no timepoint of the block has
    outlines."""
    t = plan.t_axis
    level = image.levels[plan.level_indices[k]]
    spatial = _spatial_axes(image)
    margin = plan.box.margin * pixel_size(image)
    shape = [sl.stop - sl.start for sl in block]
    shape[t] = 1
    pieces, used = [], False
    for j in range(block[t].start, block[t].stop):
        outlines = timed_outlines_at(plan.box, plan.times[j])
        if not outlines:
            pieces.append(np.ones(shape, dtype=bool))
            continue
        used = True
        start = place(j)
        first = [int(start[i]) + sl.start for i, sl in enumerate(block)]
        mask = outline_mask(level, image.axis_names, first, tuple(shape),
                            outlines, spatial=spatial, margin=margin)
        pieces.append(np.broadcast_to(mask, shape))
    return np.concatenate(pieces, axis=t) if used else None


def _block_mask(image, plan, level, block, offset):
    """outline_mask() for one output block (None without outlines)."""
    if not plan.box.outlines:
        return None
    first = [sl.start + offset[i] for i, sl in enumerate(block)]
    shape = tuple(sl.stop - sl.start for sl in block)
    return outline_mask(level, image.axis_names, first, shape,
                        plan.box.outlines, spatial=_spatial_axes(image),
                        margin=plan.box.margin * pixel_size(image))


@dataclass
class CropResult:
    path: Path
    plan: CropPlan
    seconds: float
    problems: list[str]
    tiff: Path | None = None          # the TIFF saved too, if any (0.10.0)
    tiff_level: int | None = None     # the crop level in it
    tiff_seconds: float = 0.0
    tiff_problems: list[str] = field(default_factory=list)  # verify_tiff's
    # TIFFs of the crop that was replaced (overwrite), now deleted
    removed_tiffs: list[Path] = field(default_factory=list)
    checked: str = ""                 # what the check compared, in words
    # a moving region's positions file (0.17.0; write_positions_csv)
    positions: Path | None = None
    # (0.18.0) what the window should mention besides, e.g. a scratch
    # folder that could not be removed
    note: str = ""


def crop_box(image: OmeZarrImage, box: CropBox, out_dir: str | Path, *,
             min_size: int = 64, n_levels: int | None = None,
             verify: bool = True, tiff_level: int | None = None,
             tiff_progress: Callable[[int, int], None] | None = None,
             compression: str | None = None,
             pack_progress: Callable[[int, int], None] | None = None,
             **write_options) -> CropResult:
    """Plan, write and (optionally) verify one box. See write_crop.

    tiff_level:    also save this level of the crop as an ImageJ TIFF next
                   to it (export_tiff; the crop's coarsest level if it has
                   fewer, see tiff_level_for), checked too when verify is on
                   (result.tiff_problems).
    tiff_progress: called as tiff_progress(done, total) in bytes while the
                   TIFF is written.
    compression:   None or "zstd": how TIFFs are compressed (0.12.0).
    pack_progress: (0.21.0) for a zipped crop, called as
                   pack_progress(done, total) in bytes (compressed, as
                   stored) while it is packed into its zip file.
    With overwrite, the TIFFs of the crop being replaced are deleted first
    (remove_old_tiffs), so that none is left that shows the old region.

    An image read from a file into memory (0.12.0) is cropped into a TIFF
    instead (see _crop_file_image): tiff_level, min_size and n_levels do
    not apply.
    """
    if image.in_memory:
        return _crop_file_image(image, box, out_dir, verify=verify,
                                compression=compression,
                                tiff_progress=tiff_progress, **write_options)
    plan = plan_crop(image, box, min_size=min_size, n_levels=n_levels)
    out_path = output_path_for(image, box, out_dir)
    overwrite = write_options.get("overwrite", False)
    level = None if tiff_level is None else tiff_level_for(plan, tiff_level)
    # Everything that could refuse, before anything is deleted or written.
    check_output(image, box, out_path, overwrite)
    if level is not None:
        check_tiff_image(image)
        check_tiff_output(tiff_path_for(out_path, level), overwrite, box.name,
                          image=image)
    removed = (remove_old_tiffs(out_path, box.name, image) if overwrite
               else [])
    if is_zipped(out_path):                                    # (0.21.0)
        return _crop_zipped(image, plan, out_path, level=level,
                            verify=verify, compression=compression,
                            tiff_progress=tiff_progress, removed=removed,
                            pack_progress=pack_progress, **write_options)
    t0 = time.perf_counter()
    write_crop(image, plan, out_path, **write_options)
    seconds = time.perf_counter() - t0
    problems = (verify_crop(
        image, plan, out_path,
        keep_position=write_options.get("keep_position", True),
        channel_labels=write_options.get("channel_labels"))
        if verify else [])
    result = CropResult(out_path, plan, seconds, problems,
                        removed_tiffs=removed,
                        checked=("each crop against the source: "
                                 + verification_scope(plan)) if verify
                        else "",
                        positions=write_positions_csv(image, plan, out_path))
    if level is not None:
        t0 = time.perf_counter()
        result.tiff = export_tiff(
            out_path, level, overwrite=overwrite, compression=compression,
            progress_bytes=tiff_progress,
            cancel_event=write_options.get("cancel_event"))
        result.tiff_level = level
        result.tiff_seconds = time.perf_counter() - t0
        if verify:
            result.tiff_problems = verify_tiff(out_path, result.tiff, level)
    return result


def _crop_zipped(image: OmeZarrImage, plan: CropPlan, out_path: Path, *,
                 level: int | None, verify: bool, compression: str | None,
                 tiff_progress, removed: list[Path], pack_progress,
                 **write_options) -> CropResult:
    """crop_box for a zipped crop (0.21.0): the crop is written as a
    folder (_zip_work_folder), checked there with the result stored in its
    record, its TIFF made from it, then it is packed into its zip file
    (pack_zip) and the zip is checked too: the same files and sizes, the
    RFC-9 layout, and the crop's checks again, read through the zip. If
    the zip fails them, that is stored in its record as well."""
    keep = write_options.get("keep_position", True)
    labels = write_options.get("channel_labels")
    cancel = write_options.get("cancel_event")
    overwrite = write_options.get("overwrite", False)
    options = {k: v for k, v in write_options.items() if k != "overwrite"}
    t0 = time.perf_counter()
    with _zip_work_folder(image, plan, out_path) as folder:
        _write_unzipped(image, plan, folder, out_path,
                        keep_position=keep, channel_labels=labels,
                        workers=options.get("workers", 8),
                        progress=options.get("progress"),
                        progress_bytes=options.get("progress_bytes"),
                        cancel_event=cancel)
        seconds = time.perf_counter() - t0
        problems = (verify_crop(image, plan, folder, keep_position=keep,
                                channel_labels=labels) if verify else [])
        result = CropResult(
            out_path, plan, seconds, problems, removed_tiffs=removed,
            checked=("each crop against the source: "
                     + verification_scope(plan) + "; then its zip file: "
                     "the same files, and the same checks read through it")
            if verify else "",
            positions=write_positions_csv(image, plan, out_path))
        if level is not None:
            t1 = time.perf_counter()
            result.tiff = export_tiff(
                folder, level, overwrite=overwrite, compression=compression,
                progress_bytes=tiff_progress, cancel_event=cancel,
                tiff_path=tiff_path_for(out_path, level),
                crop_name=out_path.name)
            result.tiff_level = level
            result.tiff_seconds = time.perf_counter() - t1
            if verify:
                result.tiff_problems = verify_tiff(folder, result.tiff, level)
        t1 = time.perf_counter()
        pack_zip(folder, out_path, progress=pack_progress,
                 cancel_event=cancel)
        if verify:
            zip_problems = _zip_problems(image, plan, out_path, folder,
                                         keep, labels)
            if zip_problems:
                record = open_image(folder).attrs["crop_provenance"]
                record["verified"]["ok"] = False
                record["verified"]["problems"] += zip_problems[:20]
                _rewrite_zip_record(out_path, record)
                result.problems += zip_problems
        result.seconds += time.perf_counter() - t1
    return result


def _zip_problems(image: OmeZarrImage, plan: CropPlan, zip_path: Path,
                  folder: Path, keep_position: bool, channel_labels
                  ) -> list[str]:
    """Check a zipped crop against the folder it was packed from and,
    read through the zip, against the source (see _crop_zipped)."""
    problems = _zip_layout_problems(zip_path, folder)
    try:
        crop = open_image(zip_path)
    except Exception as err:            # (any reason: report, not crash)
        return problems + [f"the zip file cannot be opened: {err}"]
    try:
        problems += [f"read through the zip: {problem}" for problem in
                     _crop_problems(image, plan, crop,
                                    keep_position=keep_position,
                                    channel_labels=channel_labels,
                                    n_blocks=SAMPLE_BLOCKS, block_size=64,
                                    seed=0)]
    finally:
        crop.close()
    return problems


def _crop_file_image(image: OmeZarrImage, box: CropBox, out_dir: str | Path,
                     *, verify: bool, compression: str | None,
                     tiff_progress, keep_position: bool = True,
                     channel_labels: list[str] | None = None,
                     workers: int = 8, overwrite: bool = False,
                     progress=None, progress_bytes=None,
                     cancel_event: threading.Event | None = None
                     ) -> CropResult:
    """Crop an image read from a file into memory (0.12.0) into a TIFF:
    the crop is made in memory with the same code as for OME-Zarr images
    (full resolution only), checked against the image there, then saved as
    a TIFF of the file's kind (output_suffix: OME-TIFF or Fiji hyperstack)
    and checked again, voxel by voxel, against the crop in memory."""
    from zarr.storage import MemoryStore

    plan = plan_crop(image, box, n_levels=1)
    out_path = output_path_for(image, box, out_dir)
    check_tiff_image(image)
    check_output_folder(image, out_dir)
    check_output(image, box, out_path, overwrite)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    root = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    _write_crop_into(root, image, plan, keep_position=keep_position,
                     channel_labels=channel_labels, workers=workers,
                     progress=None, progress_bytes=None,
                     cancel_event=cancel_event, where=f"the crop {box.name!r}")
    crop = _image_from_group(root, out_path)
    crop.source = image.source        # the crop of that file, in memory
    problems = []
    if verify:
        problems = _crop_problems(image, plan, crop,
                                  keep_position=keep_position,
                                  channel_labels=channel_labels,
                                  n_blocks=SAMPLE_BLOCKS, block_size=64,
                                  seed=0)
    record = root.attrs.asdict()["crop_provenance"]
    if verify:
        record["verified"] = {
            "ok": not problems,
            "checked": "in memory, before saving: "
                       + verification_scope(plan),
            "problems": problems[:20],
            "date": datetime.datetime.now().isoformat(timespec="seconds")}
    title = (f"{TIFF_MARK} {__version__}: the crop {box.name!r} of "
             f"{image.path.name}")
    write_tiff(crop, 0, out_path, record=record, title=title,
               kind=image.source.get("output", "imagej"),
               compression=compression, overwrite=overwrite,
               progress_bytes=tiff_progress, cancel_event=cancel_event)
    if verify:
        problems += tiff_problems(crop, 0, out_path, n_planes=None)
    seconds = time.perf_counter() - t0
    return CropResult(
        out_path, plan, seconds, problems,
        checked=("each crop against the loaded image (position, voxel size "
                 "and sample blocks), then every voxel of its TIFF against "
                 "the crop" if verify else ""),
        positions=write_positions_csv(image, plan, out_path))


def tiff_plan(image: OmeZarrImage, box: CropBox, level: int) -> CropPlan:
    """The plan of a region for a TIFF of `level` (save_region_tiff): its
    levels down to that one, whatever the region's size (the region is
    snapped to that level's grid, as a crop is to its coarsest level)."""
    return plan_crop(image, box, n_levels=max(int(level), 0) + 1)


def save_region_tiff(image: OmeZarrImage, box: CropBox, out_dir: str | Path,
                     level: int = 0, *, min_size: int = 64,
                     compression: str | None = None, overwrite: bool = False,
                     verify: bool = True, keep_position: bool = True,
                     channel_labels: list[str] | None = None,
                     workers: int = 8,
                     progress_bytes: Callable[[int, int], None] | None = None,
                     tiff_progress: Callable[[int, int], None] | None = None,
                     cancel_event: threading.Event | None = None,
                     **_ignored) -> CropResult:
    """Save one region straight to a TIFF for Fiji (0.15.0, the window's
    "Save as TIFF…"): one level of the region (0 = full resolution), with
    no OME-Zarr crop kept. The region is planned down to that level
    (tiff_plan; min_size does not apply), so a small region has it too; an
    image without that level gives its coarsest (tiff_level_for).

    The crop code makes that one level (so outlines, channels and every
    n-th timepoint apply as for a crop) in a temporary folder inside the
    output folder (never in memory: a full-resolution region can be larger
    than the memory), checks it against the source, saves it as an ImageJ
    TIFF (<image>_<region>_level<k>.tif, as "Also save as TIFF" named it),
    checks the TIFF against it, and deletes the temporary folder.

    An image read from a file into memory: its crop is a TIFF anyway
    (crop_box, full resolution).
    """
    import shutil
    import tempfile

    if image.in_memory:
        return crop_box(image, box, out_dir, verify=verify,
                        compression=compression, tiff_progress=tiff_progress,
                        keep_position=keep_position,
                        channel_labels=channel_labels, workers=workers,
                        overwrite=overwrite, progress_bytes=progress_bytes,
                        cancel_event=cancel_event)
    plan = tiff_plan(image, box, level)
    level = tiff_level_for(plan, level)
    j = plan.level_indices.index(level)
    one = CropPlan(box, [level], [plan.starts[j]], [plan.stops[j]],
                   [plan.shapes[j]], plan.channels, plan.snapped_ranges,
                   plan.steps,
                   shifts=[plan.shifts[j]] if plan.moving else None,
                   times=plan.times, t_axis=plan.t_axis)
    crop_path = output_path_for(image, box, out_dir)     # (never written)
    tiff_path = tiff_path_for(crop_path, level)
    check_output_folder(image, out_dir)
    check_tiff_image(image)
    check_tiff_output(tiff_path, overwrite, box.name, image=image)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    clear_old_scratch(out_dir)                               # (0.18.0)
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=out_dir))
    try:
        root = zarr.open_group(tmp / crop_path.name, mode="w-",
                               zarr_format=image.zarr_format)
        _write_crop_into(root, image, one, keep_position=keep_position,
                         channel_labels=channel_labels, workers=workers,
                         progress=None, progress_bytes=progress_bytes,
                         cancel_event=cancel_event,
                         where=f"the TIFF of {box.name!r}")
        crop = _image_from_group(root, crop_path)
        problems = []
        if verify:
            problems = _crop_problems(image, one, crop,
                                      keep_position=keep_position,
                                      channel_labels=channel_labels,
                                      n_blocks=SAMPLE_BLOCKS, block_size=64,
                                      seed=0)
        record = root.attrs.asdict()["crop_provenance"]
        record["saved_as"] = (f"TIFF of level {level} only (no OME-Zarr "
                              "crop kept)")
        if verify:
            record["verified"] = {
                "ok": not problems,
                "checked": "before saving: " + verification_scope(one),
                "problems": problems[:20],
                "date": datetime.datetime.now().isoformat(
                    timespec="seconds")}
        title = (f"{TIFF_MARK} {__version__}: level {level} of the region "
                 f"{box.name!r} of {image.path.name}")
        write_tiff(crop, 0, tiff_path, record=record, title=title,
                   kind="imagej", compression=compression,
                   overwrite=overwrite, progress_bytes=tiff_progress,
                   cancel_event=cancel_event)
        if verify:
            problems += tiff_problems(crop, 0, tiff_path)
    finally:
        left = remove_scratch(tmp)
    seconds = time.perf_counter() - t0
    return CropResult(
        tiff_path, one, seconds, problems, tiff=tiff_path, tiff_level=level,
        tiff_seconds=seconds,
        positions=write_positions_csv(image, one, tiff_path),
        checked=(f"level {level} of the region against the source (position,"
                 " voxel size and sample blocks), then sample slices of the "
                 "TIFF against it" if verify else ""),
        note=(f"its scratch folder {tmp.name} could not be removed (Windows "
              "may still hold it open); it is cleared the next time a TIFF "
              "is saved there, or delete it yourself" if left else ""))


# (0.18.0) "Save as TIFF…" writes the region as an OME-Zarr into a hidden
# scratch folder in the output folder first (the TIFF is written from it,
# then it is removed).
SCRATCH_PREFIX = ".volume_cropper_tmp_"
# An old scratch folder (left by a removal that failed, e.g. because
# Windows still held a folder open; the user found an empty one on the
# Desktop on 2026-11-13) is cleared when it holds no files and has not
# changed for this long (a younger one may belong to a save running in
# another window).
SCRATCH_AGE_S = 3600


def remove_scratch(folder: Path, tries: int = 4, wait: float = 0.5) -> bool:
    """Remove a scratch folder, trying again a few times (Windows may
    hold a file or folder open for a moment, e.g. a virus scanner).
    Returns True if it is still there."""
    folder = Path(folder)
    for attempt in range(tries):
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            return False
        time.sleep(wait * (attempt + 1))
    return folder.exists()


def clear_old_scratch(out_dir: str | Path) -> list[Path]:
    """Remove scratch folders left in out_dir by earlier saves that hold
    no files (only empty folders) and are older than SCRATCH_AGE_S;
    returns those removed. Folders with files are left alone."""
    removed = []
    try:
        candidates = list(Path(out_dir).glob(SCRATCH_PREFIX + "*"))
    except OSError:
        return removed
    now = time.time()
    for folder in candidates:
        try:
            if (not folder.is_dir()
                    or now - folder.stat().st_mtime < SCRATCH_AGE_S
                    or any(path.is_file() for path in folder.rglob("*"))):
                continue
        except OSError:
            continue
        if not remove_scratch(folder, tries=2, wait=0.2):
            removed.append(folder)
    return removed


# ---------------------------------------------------------------------------
# Saving one level of a crop as TIFF, for Fiji (0.10.0)
# ---------------------------------------------------------------------------

# ImageJ hyperstacks hold only these data types (tifffile's kind="imagej").
TIFF_DTYPES = ("uint8", "uint16", "float32")
# While a TIFF is written, whole slices (all channels) are read at once, at
# most this many bytes (but at least one slice), the next read running
# while the last is written: the level is never all in memory.
TIFF_READ_BYTES = 256 * 2**20
# ... but when one z chunk (all channels) is thicker than that, it is read
# whole (up to this size, without reading ahead), so that each chunk is
# read once: thinner reads would read every chunk several times.
TIFF_READ_MAX = 2**30
# A classic TIFF ends at 4 GB. Larger ImageJ TIFFs keep their data in one
# block after the first image directory, ImageJ's own way past the limit:
# Fiji (File > Open) reads them; other programs may show only one plane.
TIFF_LIMIT = 2**32
# First word of the ImageJ "Info" of every TIFF this tool writes; only such
# TIFFs are ever overwritten.
TIFF_MARK = "Volume Cropper"
# Slice labels (the channel names, one per plane) only up to this many
# planes.
TIFF_MAX_LABELS = 100_000
# Time units as ImageJ names them (its default is "sec").
_IMAGEJ_TIME_UNITS = {"second": "sec", "millisecond": "ms",
                      "microsecond": "us", "minute": "min", "hour": "hour",
                      "day": "day"}
# Units written with a non-ASCII µ (not OME names, but seen in metadata):
# ImageJ's own spelling ("um" is shown as µm).
_IMAGEJ_UNIT_ALIASES = {"µm": "um", "μm": "um", "micron": "um",
                        "microns": "um", "µs": "us", "μs": "us"}


def tiff_path_for(crop_path: str | Path, level: int) -> Path:
    """Where a crop level's TIFF goes: next to the crop, named
    <crop name>_level<k>.tif (e.g. embryo_trunk_level1.tif)."""
    crop_path = Path(crop_path)
    return crop_path.with_name(f"{crop_stem(crop_path)}_level{level}.tif")


def tiff_level_for(plan: CropPlan, level: int) -> int:
    """The crop level that a TIFF of `level` is made from: `level` itself,
    or the crop's coarsest level if it has fewer (a small region keeps
    fewer levels; see plan_crop's min_size). Level k of a crop has the
    voxel size of the source's level k (0 = full resolution)."""
    if level < 0:
        raise ValueError("The TIFF level must be 0 or more "
                         "(0 = full resolution).")
    return min(level, len(plan.level_indices) - 1)


def tiff_bytes(plan: CropPlan, dtype, level: int) -> int:
    """Size of the TIFF of `level` (see tiff_level_for): its image data,
    uncompressed like every ImageJ TIFF."""
    return nbytes(plan.shapes[tiff_level_for(plan, level)], dtype)


# OME-TIFFs hold these too (Fiji opens them through Bio-Formats).
OME_TIFF_DTYPES = ("uint8", "uint16", "uint32", "int8", "int16", "int32",
                   "float32", "float64")
# The one compression offered (0.12.0): zstd, lossless, with the horizontal
# predictor for integer data (each pixel stored as the difference to its
# left neighbour, which roughly halves 16-bit microscopy data). Fiji opens
# such files through Bio-Formats (its own TIFF reader has no zstd).
TIFF_COMPRESSIONS = (None, "zstd")


def check_tiff_dtype(dtype, kind: str = "imagej") -> None:
    """Raise ValueError if a TIFF of this kind ("imagej" or "ome") cannot
    hold this data type."""
    name = np.dtype(dtype).name
    if kind == "ome":
        if name not in OME_TIFF_DTYPES:
            raise ValueError(f"This image is {name}, which an OME-TIFF "
                             "cannot hold.")
        return
    if name not in TIFF_DTYPES:
        raise ValueError(
            f"ImageJ TIFFs hold unsigned 8- or 16-bit integers or 32-bit "
            f"floats; this image is {name}, so it cannot be saved as TIFF. "
            "Open the .zarr itself instead (e.g. in napari, or in Fiji with "
            "an OME-Zarr reader).")


def check_tiff_image(image: OmeZarrImage) -> None:
    """Raise ValueError if the crops of this image cannot be saved as TIFFs
    (their data type or axes), so that it is known before anything is
    written. A crop keeps its image's data type and axes. (ImageJ TIFFs;
    OME-TIFFs for a file read into memory whose crops are OME-TIFFs.)"""
    kind = (image.source or {}).get("output", "imagej")
    check_tiff_dtype(image.levels[0].array.dtype, kind)
    _imagej_axes(image)


def tiff_options(kind: str, compression: str | None, dtype,
                 nbytes_: int) -> dict:
    """tifffile.imwrite settings for the compression, and BigTIFF where a
    TIFF could pass 4 GB: a compressed classic TIFF cannot (tifffile only
    switches by itself for uncompressed files), so above TIFF_LIMIT (the
    uncompressed size, an upper limit) it becomes a BigTIFF. Uncompressed
    ImageJ TIFFs use ImageJ's own large-file layout instead (TIFF_LIMIT)."""
    if compression not in TIFF_COMPRESSIONS:
        raise ValueError(f"Unknown TIFF compression {compression!r}; use "
                         "None or 'zstd'.")
    options = {}
    if compression == "zstd":
        options["compression"] = "zstd"
        if np.issubdtype(np.dtype(dtype), np.integer):
            options["predictor"] = True
        if nbytes_ > TIFF_LIMIT - 2**26:          # 64 MB for the directories
            options["bigtiff"] = True
    return options


def large_tiff_note(kind: str, compression: str | None) -> str:
    """What a TIFF over 4 GB is, and how Fiji opens it (for questions)."""
    if kind == "imagej" and compression is None:
        return ("Such TIFFs use ImageJ's own large-file layout: Fiji opens "
                "them with File > Open, but other programs may show only the "
                "first plane.")
    return ("Such TIFFs are BigTIFFs: Fiji opens them through Bio-Formats "
            "(File > Open shows its dialog); most other programs read them "
            "too.")


def old_tiffs(crop_path: str | Path, box_name: str,
              image: OmeZarrImage | None = None) -> list[Path]:
    """The TIFFs this tool wrote of a crop (any level) for this box, next to
    it: <crop name>_level<k>.tif whose record names the same box (and,
    given the image, the same image: see _other_image). TIFFs of another
    box whose name merely starts the same are left out."""
    crop_path = Path(crop_path)
    stem = re.escape(crop_stem(crop_path))
    found = []
    if crop_path.parent.is_dir():
        for path in sorted(crop_path.parent.iterdir()):
            if re.fullmatch(stem + r"_level\d+\.tif", path.name) \
                    and path.is_file():
                record = _tiff_record(path)
                if record is not None and record.get("box_name") == box_name \
                        and (image is None or _other_image(
                            record, _image_identity(image)) is None):
                    found.append(path)
    return found


def remove_old_tiffs(crop_path: str | Path, box_name: str,
                     image: OmeZarrImage | None = None) -> list[Path]:
    """Delete the TIFFs of a crop that is about to be replaced (see
    old_tiffs): they would show the old region next to the new crop.
    Returns the deleted files. A TIFF open in another program (Windows
    refuses to delete it) stops this with a clear error, before the crop is
    replaced."""
    removed = []
    for path in old_tiffs(crop_path, box_name, image):
        try:
            path.unlink()
        except PermissionError as err:
            raise PermissionError(
                f"Cannot delete the old TIFF {path} of this region: is it "
                "open in another program, e.g. Fiji? Close it there, then "
                "crop again.") from err
        removed.append(path)
    return removed


def check_tiff_output(tiff_path: str | Path, overwrite: bool = False,
                      box_name: str | None = None,
                      image: OmeZarrImage | None = None,
                      identity: dict | None = None) -> None:
    """Raise the error export_tiff would raise for this TIFF, so that every
    box can be checked before the first is written: it exists (without
    overwrite), it was not written by this tool, or it holds another
    region's crop (whose name gives the same file name), or (given the
    image it will be of, or its identity, see _image_identity) a crop of
    another image."""
    if identity is None and image is not None:
        identity = _image_identity(image)
    tiff_path = Path(tiff_path)
    if not tiff_path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"{tiff_path} already exists. To replace it, use the overwrite "
            "option (--overwrite on the command line, OVERWRITE = True in "
            "the notebook, 'Overwrite earlier crops' in the window).")
    record = _tiff_record(tiff_path)
    if record is None or not tiff_path.is_file():
        raise FileExistsError(
            f"{tiff_path} exists and was not written by Volume Cropper; "
            "refusing to overwrite it.")
    _refuse_other_crop(tiff_path, record, box_name, identity)


def _refuse_other_crop(path: Path, record: dict, box_name: str | None,
                       identity: dict | None) -> None:
    """Overwrite replaces only an earlier crop of the same box (a box whose
    name merely gives the same file name, e.g. "tail 1" and "tail_1", must
    not replace another box's crop) of the same image (pos1/img.zarr and
    pos2/img.zarr give the same crop name, and so do two images called
    alike in one file: the 0.17.0 review found the second crop silently
    replacing the first). identity: the new crop's image (see
    _image_identity), None to skip that test."""
    old_name = record.get("box_name")
    if box_name is not None and old_name not in (None, box_name):
        raise FileExistsError(
            f"{path.name} holds the crop of box {old_name!r}, not of "
            f"{box_name!r}; refusing to replace it. Rename the box, or "
            "delete the old crop first.")
    other = _other_image(record, identity)
    if other:
        raise FileExistsError(
            f"{path.name} holds a crop of {other}; refusing to replace it. "
            "Choose another output folder, or delete the old crop first.")


def _image_identity(image: OmeZarrImage) -> dict:
    """Which image a crop is of: the image's absolute path, and for a file
    with several images, which of them (its index in the file)."""
    return {"source": str(Path(image.path).absolute()),
            "scene": (image.source or {}).get("scene_index"),
            "inner": image.inner}                          # (0.21.0)


def _record_identity(record: dict) -> dict:
    """Which image a crop record says the crop is of, as _image_identity
    (source None when the record's path cannot be compared: a relative
    path, written before 0.17.1 from a relative path, or a path of another
    system)."""
    source = record.get("source")
    file_info = record.get("source_file")
    return {"source": source if isinstance(source, str)
            and Path(source).is_absolute() else None,
            "scene": file_info.get("scene_index")
            if isinstance(file_info, dict) else None,
            "inner": record.get("source_image") or ""}    # (0.21.0)


def _other_image(record: dict, identity: dict | None) -> str | None:
    """If a crop record says it is a crop of another image than the one
    given (see _image_identity): that image, in words; else None. What
    cannot be compared (see _record_identity) counts as the same image, as
    before 0.17.1."""
    if not identity:
        return None
    old = _record_identity(record)
    if old["source"] and identity.get("source") \
            and not _same_path(old["source"], identity["source"]):
        return old["source"]
    if old["scene"] is not None and identity.get("scene") is not None \
            and old["scene"] != identity["scene"]:
        return (f"image {old['scene'] + 1} of "
                f"{Path(str(record.get('source'))).name}")
    if old["inner"] != identity.get("inner", ""):              # (0.21.0)
        return (f"image {old['inner'] or '(the whole)'} of "
                f"{Path(str(record.get('source'))).name}")
    return None


def _same_path(a: str | Path, b: str | Path) -> bool:
    """Whether two paths name the same file or folder (as written: the
    same image opened through another drive letter counts as another)."""
    def norm(p):
        return os.path.normcase(os.path.abspath(str(p)))
    return norm(a) == norm(b)


def _tiff_record(path: Path) -> dict | None:
    """The crop record in a TIFF written by this tool ({} if it has none),
    or None if the file is not such a TIFF."""
    import tifffile                    # only needed for TIFFs
    try:
        with tifffile.TiffFile(path) as tif:
            info = (tif.imagej_metadata or {}).get("Info", "")
            if not info and tif.is_ome:            # OME-TIFF (0.12.0)
                images = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]
                image = images[0] if isinstance(images, list) else images
                info = image.get("Description", "")
    except Exception:
        return None
    if not isinstance(info, str) or not info.startswith(TIFF_MARK):
        return None
    _, _, text = info.partition("crop_provenance = ")
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {}


def _imagej_axes(image: OmeZarrImage) -> dict:
    """Which axis of the image becomes ImageJ's T, Z, C, Y and X (None for
    those it lacks; ImageJ gives them size 1)."""
    spatial = [i for i, a in enumerate(image.axes) if a.get("type") == "space"]
    times = [i for i, a in enumerate(image.axes) if a.get("type") == "time"]
    if len(spatial) not in (2, 3) or len(times) > 1:
        raise ValueError(
            f"An ImageJ TIFF holds 2 or 3 spatial axes, up to one time axis "
            f"and one channel axis; this image has axes {image.axis_names}.")
    axes = {"T": times[0] if times else None, "Z": None,
            "C": image.channel_axis, "Y": spatial[-2], "X": spatial[-1]}
    if len(spatial) == 3:
        axes["Z"] = spatial[0]
    other = set(range(len(image.axes))) - set(axes.values())
    if other:
        raise ValueError(
            "These axes cannot be stored in an ImageJ TIFF: "
            + ", ".join(image.axis_names[i] for i in sorted(other)) + ".")
    return axes


def _imagej_unit(unit, names: dict, default: str) -> str:
    """A unit as ImageJ's image description holds it: ImageJ's short name
    ("um", which Fiji shows as µm; "min" ...), in ASCII only. The
    description is read like a Java properties file, so any other
    character is written as \\uXXXX (and a backslash doubled)."""
    text = names.get(unit) or _UNIT_ABBREVIATIONS.get(unit) or unit or default
    text = _IMAGEJ_UNIT_ALIASES.get(text, text)
    return "".join(c if 32 <= ord(c) < 127 and c != "\\" else f"\\u{ord(c):04X}"
                   for c in str(text))


def _imagej_metadata(image: OmeZarrImage, level: int, axes: dict,
                     sizes: dict, info: str) -> tuple[dict, tuple, list]:
    """ImageJ's calibration and display settings for one level: tifffile's
    `metadata`, `resolution` (pixels per unit, x and y) and `extratags`."""
    lev = image.levels[level]
    names = image.axis_names
    scale = {k: lev.scale[i] for k, i in axes.items() if i is not None}
    origin = {k: lev.translation[i] for k, i in axes.items() if i is not None}
    metadata = {"axes": "TZCYX", "Info": info,
                "unit": _imagej_unit(image.unit(names[axes["X"]]), {},
                                     "pixel")}
    # Positions: ImageJ puts pixel coordinate p at (p - origin) * pixel size
    # (origin in pixels), and pixel i covers p = i ... i + 1, its centre at
    # i + 0.5 (Fiji's centroids and ROIs use that). OME puts voxel i's
    # centre at translation + i * voxel size. So x and y get the origin
    # 0.5 - translation / voxel size: Fiji's measurements are then the
    # crop's OME (napari) coordinates, with "keep position" those of the
    # full image. Slices have no half: slice i is at (i - origin) * spacing.
    # (Fiji's status bar shows the corner of the pixel under the mouse,
    # half a voxel less in x and y, as always in ImageJ.)
    for k in ("X", "Y", "Z"):
        if k in scale and scale[k]:
            half = 0.5 if k in "XY" else 0.0
            # (+ 0.0 turns -0.0 into 0.0)
            metadata[f"{k.lower()}origin"] = round(
                half - origin[k] / scale[k], 6) + 0.0
    if "Z" in scale:
        metadata["spacing"] = scale["Z"]
    if "T" in scale:
        metadata["finterval"] = scale["T"]       # in tunit, as in ImageJ
        metadata["tunit"] = _imagej_unit(image.unit(names[axes["T"]]),
                                         _IMAGEJ_TIME_UNITS, "sec")
    extratags = []
    n_channels = sizes["C"]
    if n_channels > 1:
        metadata["mode"] = "composite"     # channels overlaid, as in napari
    # Channel names, colours and display ranges from the OME metadata.
    channels = image.ome.get("omero", {}).get("channels", [])
    if len(channels) == n_channels:
        labels = [str(ch.get("label") or f"channel {c}")
                  for c, ch in enumerate(channels)]
        # one label per plane (ImageJ shows it in the window); not for
        # huge stacks, where the labels alone would take many MB
        if sizes["T"] * sizes["Z"] * n_channels <= TIFF_MAX_LABELS:
            metadata["Labels"] = [labels[c] for _ in range(sizes["T"])
                                  for _ in range(sizes["Z"])
                                  for c in range(n_channels)]
        luts = [_lut(ch.get("color")) for ch in channels]
        if all(lut is not None for lut in luts):
            metadata["LUTs"] = luts
            # ImageJ uses these LUTs only with several channels. One channel
            # gets its colour from the TIFF ColorMap tag (320), as ImageJ
            # itself saves it: 3 × 256 values, ImageJ reads the high byte.
            # Not for 8-bit data, which ImageJ would then open as "8-bit
            # Color" (a different image type for its tools): grey instead.
            if n_channels == 1 and lev.array.dtype.name != "uint8":
                values = (luts[0].astype(np.uint16) << 8).ravel().tolist()
                extratags.append((320, "H", len(values), values, True))
        windows = [ch.get("window") or {} for ch in channels]
        if all(isinstance(w.get("start"), (int, float))
               and isinstance(w.get("end"), (int, float))
               and w["end"] > w["start"] for w in windows):
            metadata["Ranges"] = [float(v) for w in windows
                                  for v in (w["start"], w["end"])]
            if n_channels == 1:            # one channel: ImageJ's min, max
                metadata["min"], metadata["max"] = metadata["Ranges"]
    return metadata, (1 / scale["X"], 1 / scale["Y"]), extratags


def _lut(color) -> np.ndarray | None:
    """ImageJ lookup table (3 × 256, black to the colour) for an OME colour
    such as "FF00FF"; None if the colour cannot be read."""
    try:
        rgb = [int(str(color)[i:i + 2], 16) for i in (0, 2, 4)]
    except (TypeError, ValueError):
        return None
    ramp = np.arange(256) / 255
    return np.array([np.round(ramp * v) for v in rgb], dtype=np.uint8)


def _read_slices(array, axes: dict, sizes: dict, t: int, z0: int, z1: int):
    """Slices z0..z1 of timepoint t, all channels, as (Z, C, Y, X)."""
    selection = []
    for i in range(array.ndim):
        if i == axes["T"]:
            selection.append(slice(t, t + 1))
        elif i == axes["Z"]:
            selection.append(slice(z0, z1))
        else:
            selection.append(slice(None))
    block = np.asarray(array[tuple(selection)])
    present = [axes[k] for k in "TZCYX" if axes[k] is not None]
    block = block.transpose(present)          # the axes it has, in TZCYX order
    # Adding the missing axes (size 1) keeps the order, so reshape does it.
    return block.reshape([z1 - z0 if k == "Z" else (1 if k == "T" else
                                                   sizes[k])
                          for k in "TZCYX"])[0]


def export_tiff(crop_path: str | Path, level: int = 0, *,
                overwrite: bool = False, compression: str | None = None,
                progress_bytes: Callable[[int, int], None] | None = None,
                cancel_event: threading.Event | None = None,
                tiff_path: str | Path | None = None,
                crop_name: str | None = None) -> Path:
    """Save one level of a crop as an ImageJ hyperstack TIFF next to it,
    for Fiji (File > Open; ilastik and most programs read TIFF too).

    The TIFF holds exactly that level's voxels (voxels set to 0 by the
    outlines stay 0), in ImageJ's axis order TZCYX (axes the crop lacks
    have size 1), with ImageJ's calibration: voxel size and unit, z
    spacing, frame interval and time unit, and the origin, so that Fiji's
    measurements (centroids, ROIs) give the crop's own µm coordinates, the
    same as napari's (see _imagej_metadata). Channel names (as slice
    labels), colours and display ranges come from the OME metadata (one
    8-bit channel opens grey), and the crop's record (crop_provenance) is
    the image's Info (Fiji: Image > Show Info). ImageJ TIFFs are
    uncompressed: the file is as large as the level in memory. Above 4 GB,
    see TIFF_LIMIT.

    The data are read a few slices at a time, in whole z chunks
    (TIFF_READ_BYTES, TIFF_READ_MAX; the next slices are read while the
    last ones are written) and written plane by plane, into
    <name>.tif.part, which is renamed when complete: an interrupted export
    leaves no half TIFF behind.

    level:          the crop's level (0 = full resolution); see tiff_level_for.
    overwrite:      replace an earlier TIFF of the same region (only TIFFs
                    written by this tool).
    compression:    None, or "zstd" (lossless, 0.12.0; see TIFF_COMPRESSIONS:
                    Fiji then opens it through Bio-Formats).
    progress_bytes: called as progress_bytes(done, total) after each plane.
    cancel_event:   set it (from another thread) to stop the export.
    tiff_path:      (0.21.0) where to save it instead of next to the crop
                    (a zipped crop's TIFF is made from its folder before
                    it is packed); crop_name: the crop's name for the
                    TIFF's title then.
    Returns the TIFF's path (see tiff_path_for).
    """
    crop_path = Path(crop_path)
    crop = open_image(crop_path)
    try:
        if not 0 <= level < len(crop.levels):
            raise ValueError(f"{crop_path.name} has levels 0–"
                             f"{len(crop.levels) - 1}; there is no level "
                             f"{level}.")
        record = crop.attrs.get("crop_provenance")
        return write_tiff(
            crop, level, tiff_path or tiff_path_for(crop_path, level),
            record=record,
            title=(f"{TIFF_MARK} {__version__}: level {level} of the crop "
                   f"{crop_name or crop_path.name}"),
            kind="imagej", compression=compression, overwrite=overwrite,
            progress_bytes=progress_bytes, cancel_event=cancel_event)
    finally:
        crop.close()


def write_tiff(crop: OmeZarrImage, level: int, tiff_path: str | Path, *,
               record: dict | None, title: str, kind: str = "imagej",
               compression: str | None = None, overwrite: bool = False,
               progress_bytes: Callable[[int, int], None] | None = None,
               cancel_event: threading.Event | None = None) -> Path:
    """Save one level of a crop (on disk or in memory) as a TIFF: an ImageJ
    hyperstack (kind "imagej", see export_tiff) or an OME-TIFF (kind "ome",
    0.12.0: voxel size, frame interval and channel names and colours in its
    OME metadata, the record in its Description). The first line of the
    record text is `title`. compression: see export_tiff."""
    import tifffile                    # only needed for TIFFs

    tiff_path = Path(tiff_path)
    array = crop.levels[level].array
    check_tiff_dtype(array.dtype, kind)
    check_tiff_output(tiff_path, overwrite,
                      record.get("box_name") if record else None,
                      identity=_record_identity(record) if record else None)

    axes = _imagej_axes(crop)
    sizes = {k: array.shape[i] if i is not None else 1
             for k, i in axes.items()}
    n_t, n_z, n_c, n_y, n_x = (sizes[k] for k in "TZCYX")
    info = (title
            + ("\n\ncrop_provenance = " + json.dumps(record, indent=1,
                                                     ensure_ascii=False)
               if record else ""))
    if kind == "ome":
        metadata = _ome_tiff_metadata(crop, level, axes, sizes, info,
                                      tiff_path.name)
        resolution, extratags = None, []
    else:
        metadata, resolution, extratags = _imagej_metadata(crop, level, axes,
                                                           sizes, info)

    # Slices per read: as many as TIFF_READ_BYTES allows, in whole z chunks,
    # so that each chunk is read (and decompressed) once. A chunk thicker
    # than that is read whole, up to TIFF_READ_MAX, without reading ahead
    # (one such read in memory at a time); only beyond that are reads
    # thinner than a chunk, which then reads each chunk several times.
    slice_bytes = nbytes((n_c, n_y, n_x), array.dtype)
    depth = max(1, int(TIFF_READ_BYTES // slice_bytes))
    z_chunk = array.chunks[axes["Z"]] if axes["Z"] is not None else 1
    read_ahead = True
    if depth >= z_chunk:
        depth -= depth % z_chunk
    elif z_chunk * slice_bytes <= TIFF_READ_MAX:
        depth, read_ahead = z_chunk, False
    depth = min(depth, n_z)
    reads = [(t, z0, min(z0 + depth, n_z))
             for t in range(n_t) for z0 in range(0, n_z, depth)]
    total = nbytes((n_t, n_z, n_c, n_y, n_x), array.dtype)

    def planes():
        """The planes in ImageJ's order (channels fastest, then z, then t),
        reading the next slices while the current ones are written."""
        done = 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            ahead = None
            for n, read in enumerate(reads):
                block = (ahead.result() if ahead is not None
                         else _read_slices(array, axes, sizes, *read))
                ahead = (pool.submit(_read_slices, array, axes, sizes,
                                     *reads[n + 1])
                         if read_ahead and n + 1 < len(reads) else None)
                for z_planes in block:
                    for plane in z_planes:
                        if cancel_event is not None and cancel_event.is_set():
                            raise CropCancelled(
                                f"Cancelled; {tiff_path.name} was not saved.")
                        yield plane
                        done += plane.nbytes
                        if progress_bytes is not None:
                            progress_bytes(done, total)

    part = tiff_path.with_name(tiff_path.name + ".part")
    options = tiff_options(kind, compression, array.dtype, total)
    if kind == "ome":
        options["ome"] = True
    else:
        options.update(kind="imagej", resolution=resolution,
                       extratags=extratags)
    try:
        with warnings.catch_warnings():
            # > 4 GB: tifffile says it writes ImageJ's one-directory form,
            # or (compressed) a BigTIFF that ImageJ itself does not read
            # (Fiji opens it through Bio-Formats; see large_tiff_note)
            warnings.filterwarnings("ignore", message=".*truncating ImageJ")
            warnings.filterwarnings("ignore",
                                    message=".*nonconformant BigTIFF")
            tifffile.imwrite(part, data=planes(),
                             shape=(n_t, n_z, n_c, n_y, n_x),
                             dtype=array.dtype, photometric="minisblack",
                             metadata=metadata, **options)
        try:
            part.replace(tiff_path)
        except PermissionError as err:
            raise PermissionError(
                f"Cannot replace {tiff_path}: is it open in another program, "
                "e.g. Fiji? Close it there, then save it again.") from err
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    # The end, said here: planes() reports a plane only when tifffile asks
    # for the next one, and tifffile, which knows the shape, never asks
    # after the last plane, so its report never came (the user saw the
    # window's progress bar stop at 99 %, 2026-11-09).
    if progress_bytes is not None:
        progress_bytes(total, total)
    return tiff_path


def verify_tiff(crop_path: str | Path, tiff_path: str | Path, level: int,
                n_planes: int = 8, seed: int = 0) -> list[str]:
    """Check a TIFF from export_tiff against the crop's level; return a list
    of problems (an empty list means OK).

    Checks that it is an ImageJ hyperstack of the right size, data type,
    voxel size, z spacing and frame interval, and compares the voxels of
    the first and the last slice and of n_planes random ones (all their
    channels) with the crop: a spot check, like verify_crop's.
    """
    crop = open_image(crop_path)
    try:
        return tiff_problems(crop, level, tiff_path, n_planes=n_planes,
                             seed=seed)
    finally:
        crop.close()              # (a zipped crop's file: see close)


def tiff_problems(crop: OmeZarrImage, level: int, tiff_path: str | Path,
                  n_planes: int | None = 8, seed: int = 0) -> list[str]:
    """verify_tiff's checks, against an opened crop (on disk or in memory),
    for ImageJ TIFFs and OME-TIFFs, compressed or not. n_planes: None
    compares every slice."""
    import tifffile

    lev = crop.levels[level]
    array = lev.array
    axes = _imagej_axes(crop)
    sizes = {k: array.shape[i] if i is not None else 1
             for k, i in axes.items()}
    n_t, n_z, n_c, n_y, n_x = (sizes[k] for k in "TZCYX")
    has_z = (crop.source or {}).get("has_z", True)
    problems = []
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages.first
        if tif.is_ome:
            series = tif.series[0]
            shape = dict(zip(series.axes, series.shape))
            found = tuple(shape.get(k, 1) for k in "TZCYX")
            images = tifffile.xml2dict(tif.ome_metadata)["OME"]["Image"]
            pixels = (images[0] if isinstance(images, list)
                      else images).get("Pixels", {})
            checks = [("x voxel size", pixels.get("PhysicalSizeX"),
                       axes["X"]),
                      ("z spacing", pixels.get("PhysicalSizeZ"),
                       axes["Z"] if has_z else None),
                      ("frame interval", pixels.get("TimeIncrement"),
                       axes["T"])]
        elif tif.is_imagej and tif.imagej_metadata:
            meta = tif.imagej_metadata
            found = (meta.get("frames", 1), meta.get("slices", 1),
                     meta.get("channels", 1), *page.shape[-2:])
            x_res = page.tags["XResolution"].value
            checks = [("x voxel size", x_res[1] / x_res[0], axes["X"]),
                      ("z spacing", meta.get("spacing"), axes["Z"]),
                      ("frame interval", meta.get("finterval"), axes["T"])]
        else:
            return ["neither an ImageJ TIFF nor an OME-TIFF"]
        if found != (n_t, n_z, n_c, n_y, n_x):
            problems.append(f"size (t, z, c, y, x) is {found}, expected "
                            f"{(n_t, n_z, n_c, n_y, n_x)}")
        # (by name: the byte order may differ, e.g. a big-endian 0.4 crop
        # gives a TIFF in the computer's order, with the same values)
        if page.dtype.name != array.dtype.name:
            problems.append(f"data type {page.dtype.name}, expected "
                            f"{array.dtype.name}")
        for name, value, i in checks:
            if i is not None and not _close([value or 0.0], [lev.scale[i]]):
                problems.append(f"{name} {value}, expected {lev.scale[i]}")
        if problems:
            return problems
        series = tif.series[0]
        data = None
        if tif.is_imagej and page.compression == 1:
            # uncompressed ImageJ TIFF: memory-mapped (works above 4 GB,
            # where ImageJ's layout has a single image directory)
            data = tifffile.memmap(tiff_path, mode="r").reshape(
                n_t, n_z, n_c, n_y, n_x)

            def plane(t, z):
                return data[t, z]
        else:
            # compressed, or an OME-TIFF: read through tifffile's zarr view
            # (axes of size 1 are left out there)
            stored = zarr.open_array(series.aszarr(), mode="r")

            def plane(t, z):
                """Slice z of timepoint t, all channels, as (C, Y, X)."""
                key = [t if a == "T" else z if a == "Z" else
                       slice(None) if a in "CYX" else 0 for a in series.axes]
                values = np.asarray(stored[tuple(key)])
                return values if "C" in series.axes else values[np.newaxis]
        try:
            if n_planes is None:
                picks = [(t, z) for t in range(n_t) for z in range(n_z)]
            else:
                rng = np.random.default_rng(seed)
                picks = {(0, 0), (n_t - 1, n_z - 1)}
                picks |= {(int(rng.integers(n_t)), int(rng.integers(n_z)))
                          for _ in range(n_planes)}
                picks = sorted(picks)
            for t, z in picks:
                expected = _read_slices(array, axes, sizes, t, z, z + 1)[0]
                # (NaN, e.g. a float crop's fill value, counts as equal)
                if not np.array_equal(plane(t, z), expected, equal_nan=True):
                    problems.append(f"voxels differ at timepoint {t}, "
                                    f"slice {z}")
                    if len(problems) >= 20:
                        break
        finally:
            data = None                # closes the file (Windows locks it)
    return problems


# OME units as symbols (OME-TIFF's PhysicalSize...Unit, TimeIncrementUnit).
_OME_UNIT_SYMBOLS = {"micrometer": "µm", "nanometer": "nm",
                     "millimeter": "mm", "centimeter": "cm", "meter": "m",
                     "second": "s", "millisecond": "ms",
                     "microsecond": "µs", "minute": "min", "hour": "h"}


def _ome_tiff_metadata(crop: OmeZarrImage, level: int, axes: dict,
                       sizes: dict, info: str, name: str) -> dict:
    """tifffile's OME `metadata` for one level of a crop: voxel size, frame
    interval, channel names and colours, and the record (Description)."""
    lev = crop.levels[level]
    names = crop.axis_names
    has_z = (crop.source or {}).get("has_z", True)
    metadata = {"axes": "TZCYX", "Name": name, "Description": info}
    for k in "XYZ":
        i = axes[k]
        if i is None or (k == "Z" and not has_z):
            continue
        metadata[f"PhysicalSize{k}"] = lev.scale[i]
        unit = _OME_UNIT_SYMBOLS.get(crop.unit(names[i]))
        if unit:
            metadata[f"PhysicalSize{k}Unit"] = unit
    if axes["T"] is not None:
        metadata["TimeIncrement"] = lev.scale[axes["T"]]
        unit = _OME_UNIT_SYMBOLS.get(crop.unit(names[axes["T"]]))
        if unit:
            metadata["TimeIncrementUnit"] = unit
    channels = crop.ome.get("omero", {}).get("channels", [])
    if len(channels) == sizes["C"]:
        metadata["Channel"] = {"Name": [str(ch.get("label") or f"channel {c}")
                                        for c, ch in enumerate(channels)]}
        colors = [_ome_int_color(ch.get("color")) for ch in channels]
        if all(c is not None for c in colors):
            metadata["Channel"]["Color"] = colors
    return metadata


def _ome_int_color(color) -> int | None:
    """"RRGGBB" as OME's colour: a signed 32-bit RGBA integer (opaque)."""
    try:
        r, g, b = (int(str(color)[i:i + 2], 16) for i in (0, 2, 4))
    except (TypeError, ValueError):
        return None
    value = (r << 24) | (g << 16) | (b << 8) | 255
    return value - 2**32 if value >= 2**31 else value


# ---------------------------------------------------------------------------
# Saving and loading boxes (CSV, one row per box)
# ---------------------------------------------------------------------------

def save_boxes_csv(boxes: list[CropBox], path: str | Path,
                   image: OmeZarrImage) -> None:
    """Save boxes as CSV, e.g. columns name, z_start_um, z_stop_um, ...

    Empty cells mean "whole axis". Channels are "all" or a list such as
    "0;2" (semicolons, so the CSV never needs quoting). Numbers are written
    exactly (see _exact), so loading the file gives exactly the same boxes.
    """
    columns = ["name"]
    for axis in image.range_axes:
        unit = short_unit(image.unit(axis))
        suffix = f"_{unit}" if unit else ""
        columns += [f"{axis}_start{suffix}", f"{axis}_stop{suffix}"]
    columns.append("channels")
    # Every n-th timepoint (0.11.0), only if a box has a step (so files
    # without one stay readable by older versions).
    with_step = any(box.t_step != 1 for box in boxes)
    if with_step:
        columns.append("t_step")
    # The outline margin (voxels), only if a box has one.
    with_margin = any(box.margin for box in boxes)
    if with_margin:
        columns.append("outline_margin_voxels")
    # One column per view used by any box, e.g. outline_y_x. A polygon's
    # points are "y x; y x; ..." in µm; polygons on slices are written as
    # "z=10: y x; ... | z=30: y x; ..." (the slice along the depth axis).
    planes = sorted({axes for box in boxes for axes in box.outlines})
    columns += [f"outline_{a}_{b}" for a, b in planes]
    # A region that moves over time (0.17.0), only if a box moves: its
    # shifts per time, "t=0: z=0 y=0 x=0 | t=5: z=1.5 y=-2 x=12.5" (µm).
    with_track = any(box.track for box in boxes)
    if with_track:
        columns.append("track")
    # Outlines per timepoint (0.17.0), only if a box has them: in a JSON
    # file next to the CSV (timed_outlines_path_for), named in this column.
    timed = [box for box in boxes if box.timed_outlines]
    if timed:
        columns.append("timed_outlines_file")
        sidecar = timed_outlines_path_for(path)
        sidecar.write_text(json.dumps({
            "description": "outlines per timepoint of the regions in "
                           f"{Path(path).name} (Volume Cropper "
                           f"{__version__}; µm)",
            "regions": {box.name: [
                {"time": time, "outlines": _outline_records(outlines)}
                for time, outlines in box.timed_outlines.items()]
                for box in timed}}, indent=1), encoding="utf-8")
    t_axis = time_axis(image)
    t_name = image.axis_names[t_axis] if t_axis is not None else "t"
    spatial = _spatial_axes(image)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for box in boxes:
            row = [_csv_text(box.name)]         # (no formula in Excel)
            for axis in image.range_axes:
                row += [_exact(v) for v in box.ranges[axis]] \
                    if axis in box.ranges else ["", ""]
            row.append("all" if box.channels is None
                       else ";".join(str(c) for c in box.channels))
            if with_step:
                row.append(str(box.t_step))
            if with_margin:
                row.append(_exact(box.margin))
            for axes in planes:
                row.append(_outline_cell(box.outlines.get(axes, {}), axes,
                                         spatial))
            if with_track:
                row.append(" | ".join(
                    f"{t_name}={_exact(time)}: " + " ".join(
                        f"{axis}={_exact(value)}"
                        for axis, value in shifts.items())
                    for time, shifts in box.track.items()))
            if timed:
                row.append(timed_outlines_path_for(path).name
                           if box.timed_outlines else "")
            writer.writerow(row)


def _slice_order(key):
    """Sort key for polygon keys: None first, then silhouettes (depth
    ranges), then slices."""
    if key is None:
        return (0, 0.0, 0.0)
    if is_silhouette(key):
        return (1, key[0], key[1])
    return (2, key, 0.0)


def _outline_cell(keys: dict, plane, spatial) -> str:
    """One view's polygons as CSV text (see save_boxes_csv)."""
    texts = []
    for depth in sorted(keys, key=_slice_order):
        points = "; ".join(f"{_exact(a)} {_exact(b)}" for a, b in keys[depth])
        axis = None if depth is None else _depth_axis(plane, spatial)
        if depth is None:
            texts.append(points)
        elif is_silhouette(depth):
            texts.append(f"{axis}={_exact(depth[0])}..{_exact(depth[1])}: "
                         f"{points}")
        else:
            texts.append(f"{axis}={_exact(depth)}: {points}")
    return " | ".join(texts)


def load_boxes_csv(path: str | Path) -> list[CropBox]:
    """Read boxes written by save_boxes_csv, or typed or edited in Excel.

    Accepted: commas, semicolons or tabs between the columns (Excel saves
    with semicolons where the decimal mark is a comma), decimal commas
    ("10,5"), and UTF-8 or Windows text encoding. Rows that are completely
    empty are skipped. Anything else that is wrong (a box without a name, a
    range with only a start or only a stop, a cell that is not a number,
    two boxes that would be saved under one file name) raises ValueError
    with the row number as Excel shows it (the column names are row 1).
    """
    path = Path(path)
    text = _read_text(path)
    first_line = next((line for line in text.splitlines() if line.strip()),
                      "")
    delimiter = max(",;\t", key=first_line.count)   # "," if none at all
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    # (row number, cells) for every row that is not completely empty
    rows = [(reader.line_num, [cell.strip() for cell in cells])
            for cells in reader]
    rows = [(n, cells) for n, cells in rows if any(cells)]
    if not rows:
        raise ValueError(f"{path.name} is empty.")

    # The first row holds the column names; "Name" and "name" both work.
    columns = [c.lower() if c.lower() in ("name", "channels") else c
               for c in rows[0][1]]
    if "name" not in columns:
        raise ValueError(
            f"{path.name}: the first row must hold the column names (name, "
            "z_start_um, z_stop_um, ...), but there is no 'name' column.")
    pairs = []          # (axis, start column, stop column)
    for column in columns:
        match = re.fullmatch(r"([^_]+)_(start|stop)(_.*)?", column)
        if not match:
            continue
        other = column.replace(f"_{match.group(2)}",
                               "_stop" if match.group(2) == "start"
                               else "_start", 1)
        if other not in columns:
            raise ValueError(f"{path.name}: there is a column {column} but "
                             f"no column {other}.")
        if match.group(2) == "start":
            pairs.append((match.group(1), column, other))

    boxes = []
    for line, cells in rows[1:]:
        row = {column: (cells[i] if i < len(cells) else "")
               for i, column in enumerate(columns) if column}
        row["name"] = _from_csv_text(row["name"])   # (see _csv_text)
        where = f"{path.name}, row {line}"
        if not row["name"]:
            raise ValueError(f"{where}: the box has no name (the 'name' "
                             "cell is empty).")
        where += f" (box {row['name']!r})"
        ranges = {}
        for axis, start_column, stop_column in pairs:
            start, stop = row[start_column], row[stop_column]
            if bool(start) != bool(stop):
                filled, empty = ((start_column, stop_column) if start
                                 else (stop_column, start_column))
                raise ValueError(
                    f"{where}: {filled} is filled in but {empty} is empty. "
                    "Fill in both, or leave both empty to keep the whole "
                    f"{axis} axis.")
            if start:
                ranges[axis] = (_csv_number(start, where, start_column),
                                _csv_number(stop, where, stop_column))
        channels_text = row.get("channels", "").lower()
        if channels_text in ("", "all"):
            channels = None
        else:
            try:
                channels = [int(c) for c in re.split(r"[;,\s]+",
                                                     channels_text) if c]
            except ValueError:
                raise ValueError(
                    f"{where}: channels must be 'all' or channel numbers "
                    f"such as 0;2, not {row['channels']!r}.") from None
        outlines = {}
        for column, value in row.items():
            match = re.fullmatch(r"outline_([^_]+)_([^_]+)", column)
            if match and value and column != "outline_margin_voxels":
                outlines[match.groups()] = _parse_outline_cell(
                    value, match.groups(), where, column)
        margin_text = row.get("outline_margin_voxels", "")
        margin = (_csv_number(margin_text, where, "outline_margin_voxels")
                  if margin_text else 0.0)
        step_text = row.get("t_step", "")       # empty: every timepoint
        t_step = 1
        if step_text:
            value = _csv_number(step_text, where, "t_step")
            if value != int(value) or value < 1:
                raise ValueError(
                    f"{where}: t_step must be a whole number of 1 or more "
                    f"(every n-th timepoint), not {step_text!r}.")
            t_step = int(value)
        track_text = row.get("track", "")
        track = (_parse_track_cell(track_text, where) if track_text.strip()
                 else {})
        timed = {}
        file_text = row.get("timed_outlines_file", "").strip()
        # Only a file next to the CSV: a CSV from elsewhere could name any
        # file, or a network share that Windows would connect to (and
        # send the user's login to).
        if file_text and (file_text in (".", "..") or file_text != Path(
                file_text.replace("\\", "/")).name or ":" in file_text):
            raise ValueError(
                f"{where}: its outlines per timepoint must be in a file next "
                f"to the CSV (a plain file name), not {file_text!r}.")
        if file_text:
            timed = _read_timed_outlines(path.parent / file_text,
                                         row["name"], where)
        boxes.append(CropBox(row["name"], ranges, channels, outlines,
                             margin, t_step, track, timed))
    try:
        check_box_names(boxes)
    except ValueError as err:
        raise ValueError(f"{path.name}: {err}") from None
    return boxes


def _parse_outline_cell(value: str, plane, where: str, column: str) -> dict:
    """One view's polygons from CSV text: "y x; y x; ..." (one polygon
    through the box, as before 0.7.0) or "z=10: y x; ... | z=30: ...";
    "z=0..800: ..." is a polygon drawn on a projection of z 0-800."""
    keys = {}
    for part in value.split("|"):
        if not part.strip():
            continue
        match = re.fullmatch(r"\s*([^\s=:;]+)\s*=\s*([^:]+):(.*)", part,
                             re.DOTALL)
        if match:
            axis, depth_text, points_text = match.groups()
            if axis in plane:
                raise ValueError(f"{where}: in {column}, a polygon's slice "
                                 f"must be along the view's third axis, not "
                                 f"{axis}.")
            if ".." in depth_text:
                low, high = depth_text.split("..", 1)
                depth = tuple(sorted((_csv_number(low, where, column),
                                      _csv_number(high, where, column))))
            else:
                depth = _csv_number(depth_text, where, column)
        else:
            depth, points_text = None, part
        if depth in keys:
            raise ValueError(f"{where}: {column} has two polygons for the "
                             "same slice.")
        points = []
        for point in points_text.split(";"):
            numbers = point.split()
            if not numbers:
                continue
            if len(numbers) != 2:
                raise ValueError(
                    f"{where}: {column} must list points as two numbers "
                    f"each, separated by ';' (e.g. 10 20; 30 40), not "
                    f"{point.strip()!r}.")
            points.append(tuple(_csv_number(v, where, column)
                                for v in numbers))
        keys[depth] = points
    if None in keys and len(keys) > 1:
        raise ValueError(f"{where}: in {column}, every polygon needs its "
                         "slice (e.g. z=10:) when there are several.")
    return keys


def _read_timed_outlines(path: Path, name: str, where: str) -> dict:
    """A region's outlines per timepoint from the JSON file next to a
    regions CSV (save_boxes_csv)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["regions"][name]
        return {float(e["time"]): _outlines_from_records(e["outlines"])
                for e in entries}
    except FileNotFoundError:
        raise ValueError(f"{where}: its outlines per timepoint are in "
                         f"{path.name}, which is missing (keep it next to "
                         "the CSV).") from None
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{where}: {path.name} has no usable outlines per "
                         f"timepoint for this region ({err}).") from None


def _parse_track_cell(value: str, where: str) -> dict:
    """A moving region's track from CSV text (0.17.0): "t=0: z=0 y=0 x=0 |
    t=5: z=1.5 y=-2 x=12.5" (the time, then each axis' shift)."""
    track = {}
    for part in value.split("|"):
        if not part.strip():
            continue
        match = re.fullmatch(r"\s*[^\s=:]+\s*=\s*([^:]+):(.*)", part,
                             re.DOTALL)
        if not match:
            raise ValueError(
                f"{where}: the track must list times and shifts such as "
                f"'t=0: z=0 y=0 x=0 | t=5: z=1.5 y=-2 x=12.5', not "
                f"{part.strip()!r}.")
        time = _csv_number(match.group(1).strip(), where, "track")
        shifts = {}
        for item in match.group(2).split():
            axis, sep, number = item.partition("=")
            if not sep or not axis:
                raise ValueError(f"{where}: in the track, {item!r} must be "
                                 "an axis and its shift, such as x=12.5.")
            shifts[axis] = _csv_number(number, where, "track")
        if any(same_time(time, t) for t in track):
            raise ValueError(f"{where}: the track has the time {time:g} "
                             "twice.")
        track[time] = shifts
    return track


def _outline_records(outlines: dict) -> dict:
    """Outlines for a record (JSON): {"a,b": [polygon record, ...]}."""
    return {f"{a},{b}": [_polygon_record(key, keys[key])
                         for key in sorted(keys, key=_slice_order)]
            for (a, b), keys in outlines.items()}


def _outlines_from_records(records: dict) -> dict:
    """The outlines a record holds (_outline_records), back as a
    CropBox's."""
    outlines = {}
    for plane_text, polygons in records.items():
        plane = tuple(plane_text.split(","))
        keys = {}
        for polygon in polygons:
            if polygon.get("projected_range") is not None:
                key = tuple(float(v) for v in polygon["projected_range"])
            elif polygon.get("slice") is not None:
                key = float(polygon["slice"])
            else:
                key = None
            keys[key] = [tuple(float(v) for v in p)
                         for p in polygon["points"]]
        outlines[plane] = keys
    return normalize_outlines(outlines)


def timed_outlines_path_for(csv_path: str | Path) -> Path:
    """Where save_boxes_csv keeps the regions' outlines per timepoint
    (0.17.0): <regions>_outlines_per_timepoint.json next to the CSV (they
    are too long for CSV cells: Excel cuts a cell at 32767 characters)."""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_outlines_per_timepoint.json")


def _polygon_record(key, points) -> dict:
    """One polygon for the crop's record."""
    record = {"slice": key if isinstance(key, float) else None,
              "points": [list(p) for p in points]}
    if is_silhouette(key):
        record["projected_range"] = list(key)
    return record


def _read_text(path: Path) -> str:
    """A text file's contents: UTF-8 (with or without Excel's marker), or
    else Windows text encoding (Excel's plain "CSV" on Windows)."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _csv_number(text: str, where: str, column: str) -> float:
    """A number from a CSV cell; a decimal comma ("10,5") is accepted."""
    cleaned = text.replace(",", ".") if "." not in text else text
    try:
        value = float(cleaned)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"{where}: {column} must be a number, not "
                         f"{text!r}.")
    return value


def _exact(value: float) -> str:
    """A number as the shortest text that reads back as exactly the same
    number: 12.5 -> "12.5", 100.0 -> "100", 0.1 + 0.2 ->
    "0.30000000000000004". (The old "%g" kept only 6 digits, which could
    move a box edge by one voxel.)"""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_blocks(shape, block_shape):
    """Yield tuples of slices that tile an array of `shape` (like a grid)."""
    ranges = [range(0, n, b) for n, b in zip(shape, block_shape)]
    for corner in itertools.product(*ranges):
        yield tuple(slice(c, min(c + b, n))
                    for c, b, n in zip(corner, block_shape, shape))


def _source_selection(block, offset, plan, channel_axis):
    """Map an output block to the matching selection in the source level:
    output index j along an axis is source index offset + j * step (step
    1 except along t with every n-th timepoint)."""
    selection = []
    for i, sl in enumerate(block):
        if i == channel_axis:
            selection.append(plan.channels[sl.start:sl.stop])
        else:
            step = plan.step(i)
            selection.append(slice(offset[i] + sl.start * step,
                                   offset[i] + (sl.stop - 1) * step + 1, step))
    return tuple(selection)


def _read_source(src: zarr.Array, block, plan: CropPlan, k: int,
                 channel_axis: int | None) -> np.ndarray:
    """The source voxels of one output block of kept level k.

    A fixed region: one read (_source_selection). A moving region (0.17.0):
    each output timepoint of the block is read at the region's place then
    (plan.starts[k] + plan.shifts[k][j]); its part outside the source is
    the source's fill value (0)."""
    offset = plan.starts[k]
    if not plan.moving:
        return src.oindex[_source_selection(block, offset, plan,
                                            channel_axis)]
    fill = src.fill_value if src.fill_value is not None else 0
    out = np.full(tuple(sl.stop - sl.start for sl in block), fill,
                  dtype=src.dtype)
    t = plan.t_axis
    for j in range(block[t].start, block[t].stop):
        source, target = [], []
        for i, sl in enumerate(block):
            if i == channel_axis:
                source.append(plan.channels[sl.start:sl.stop])
                target.append(slice(None))
            elif i == t:
                s = offset[i] + j * plan.step(i)
                source.append(slice(s, s + 1))
                target.append(slice(j - sl.start, j - sl.start + 1))
            else:
                a = int(offset[i] + plan.shifts[k][j, i] + sl.start)
                b = a + (sl.stop - sl.start)
                low, high = max(a, 0), min(b, src.shape[i])
                if high <= low:          # this timepoint: all outside
                    break
                source.append(slice(low, high))
                target.append(slice(low - a, high - a))
        else:
            out[tuple(target)] = src.oindex[tuple(source)]
    return out


def _cropped_ome_metadata(image, plan, keep_position, channel_labels):
    """The crop's OME metadata (see OmeZarrImage.ome): the source's, with
    the cropped levels' positions and the kept channels."""
    ome = copy.deepcopy(image.ome)
    multiscale = ome["multiscales"][0]
    if image.zarr_format == 2:
        # 0.4 only recommends the version; a crop states it.
        multiscale.setdefault("version", "0.4")
    c_axis = image.channel_axis

    def placed(k):
        """Position of level k's first cropped voxel in the source (µm)."""
        level = image.levels[plan.level_indices[k]]
        return [t if i == c_axis else round(t + plan.starts[k][i] * s, 9)
                for i, (t, s) in enumerate(zip(level.translation,
                                               level.scale))]

    # keep_position=False: the crop starts at 0 (level 0's first voxel at
    # the origin), also when the source has an offset of its own. The other
    # levels keep their small offsets relative to level 0, so that all
    # levels still line up.
    origin = placed(0)
    system = image.coordinate_system          # OME-Zarr 0.6, else None
    datasets = []
    for k, level_index in enumerate(plan.level_indices):
        level = image.levels[level_index]
        dataset = copy.deepcopy(multiscale["datasets"][level_index])
        # (every n-th timepoint: the frame interval is n times the source's)
        scale = [s * plan.step(i) for i, s in enumerate(level.scale)]
        transforms = [{"type": "scale", "scale": scale}]
        translation = placed(k)
        if not keep_position:
            translation = [t if i == c_axis else round(t - o, 9)
                           for i, (t, o) in enumerate(zip(translation,
                                                          origin))]
        if keep_position or any(translation):
            transforms.append({"type": "translation",
                               "translation": translation})
        if system is not None:
            # 0.6: one transformation per level, from the level's array to
            # the image's own coordinate system: the scale alone, or a
            # sequence of the scale and the translation.
            ends = {"input": {"path": dataset["path"]},
                    "output": {"name": system}}
            transforms = [{**transforms[0], **ends} if len(transforms) == 1
                          else {"type": "sequence", **ends,
                                "transformations": transforms}]
        dataset["coordinateTransformations"] = transforms
        datasets.append(dataset)
    multiscale["datasets"] = datasets
    if system is not None:
        extra, systems, _ = _crop_extra_transforms(
            image, _crop_shift(image, plan, keep_position))
        multiscale["coordinateSystems"] = systems
        multiscale.pop("coordinateTransformations", None)
        if extra:
            multiscale["coordinateTransformations"] = extra
        # (a scene links several images; a crop holds one)
        ome.pop("scene", None)

    channels = ome.get("omero", {}).get("channels")
    if plan.channels is not None and channels \
            and len(channels) == image.n_channels:
        kept = [copy.deepcopy(channels[c]) for c in plan.channels]
        for channel, label in zip(kept, channel_labels or []):
            channel["label"] = label
        ome["omero"]["channels"] = kept
    return ome


def _crop_shift(image: OmeZarrImage, plan: CropPlan, keep_position: bool
                ) -> list[float] | None:
    """How far a crop's own coordinates are moved from the source's (µm
    per axis; 0 on the channel axis): the position of its first voxel when
    it does not keep its position (it then starts at 0), else None."""
    if keep_position:
        return None
    level = image.levels[plan.level_indices[0]]
    c_axis = image.channel_axis
    return [0.0 if i == c_axis else round(t + plan.starts[0][i] * s, 9)
            for i, (t, s) in enumerate(zip(level.translation, level.scale))]


def _crop_extra_transforms(image: OmeZarrImage, shift: list[float] | None
                           ) -> tuple[list[dict], list[dict], list[str]]:
    """OME-Zarr 0.6 (0.21.0): the multiscale-wide transformations and the
    coordinate systems a crop keeps: (transformations, systems, notes).

    Such a transformation links the image's own coordinate system to
    another one, e.g. a registration to an atlas or a second microscope.
    The crop keeps it, so that the crop lands at the same place there.
    shift: when the crop does not keep its position, its own coordinates
    start at its first voxel, `shift` (µm per axis) away from the
    source's, and each link is corrected by that much (a "sequence" with a
    translation added). None: the crop keeps its position, and the links
    are copied as they are. A link that points into another group (e.g.
    label images) or keeps its numbers in arrays of the image (e.g. a
    displacement field) is dropped, since the crop has neither; notes says
    which and why. The coordinate systems kept are the image's own one and
    those the kept links use."""
    multiscale = image.ome["multiscales"][0]
    own = image.coordinate_system
    kept, notes = [], []
    for k, transform in enumerate(
            multiscale.get("coordinateTransformations") or []):
        if not isinstance(transform, dict):
            continue
        label = (f"{_system_name(transform.get('input'))} -> "
                 f"{_system_name(transform.get('output'))}")
        outside = _outside_reference(transform)
        if outside:
            notes.append(f"the link {label} was left out: it uses "
                         f"{outside!r}, which the crop does not include")
            continue
        if shift is not None and any(shift):
            transform = _shifted_link(transform, own, shift)
            notes.append(f"the link {label} was corrected for the crop's "
                         "start (keep_position off)")
        kept.append(copy.deepcopy(transform))
    used = {own} | {_system_name(t.get(end)) for t in kept
                    for end in ("input", "output")}
    systems = [copy.deepcopy(system)
               for system in multiscale.get("coordinateSystems", [])
               if isinstance(system, dict) and system.get("name") in used]
    return kept, systems, notes


def _outside_reference(transform: dict) -> str | None:
    """The path a 0.6 transformation needs outside the image's own
    metadata: another group named by its input or output (e.g. a label
    image), or an array holding its numbers; None if it needs none."""
    for end in ("input", "output"):
        place = transform.get(end)
        if isinstance(place, dict) and place.get("path"):
            return str(place["path"])

    def find(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("input", "output"):
                    continue
                if key == "path" and isinstance(value, str) and value:
                    return value
                found = find(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = find(item)
                if found:
                    return found
        return None

    return find({k: v for k, v in transform.items()
                 if k not in ("input", "output")})


def _shifted_link(transform: dict, own: str, shift: list[float]) -> dict:
    """A 0.6 transformation between the image's own coordinate system and
    another, rewritten for a crop whose own coordinates are the source's
    minus `shift`: from the crop, first add `shift` (back to the source's
    coordinates), then apply the transformation; towards the crop, apply
    it, then subtract `shift`."""
    steps = {k: v for k, v in transform.items()
             if k not in ("input", "output", "name")}
    inner = (copy.deepcopy(steps["transformations"])
             if steps.get("type") == "sequence" else [copy.deepcopy(steps)])
    if _system_name(transform.get("input")) == own:
        inner = [{"type": "translation",
                  "translation": [round(s, 9) for s in shift]}] + inner
    else:
        inner = inner + [{"type": "translation",
                          "translation": [round(-s, 9) for s in shift]}]
    result = {"type": "sequence", "transformations": inner,
              "input": transform.get("input"),
              "output": transform.get("output")}
    if "name" in transform:
        result["name"] = transform["name"]
    return result


def _provenance(image, plan, keep_position):
    record = {
        "source": str(Path(image.path).absolute()),
        "source_format": image.format_name,
        # (0.21.0) an image of a collection: its path in it, e.g. "0"
        **({"source_image": image.inner,
            "source_collection": image.collection} if image.inner else {}),
        "box_name": plan.box.name,
        "units": {a: image.unit(a) for a in image.range_axes},
        "requested_range": {a: list(r) for a, r in plan.box.ranges.items()},
        "output_range": {a: list(r) for a, r in plan.snapped_ranges.items()},
        "channels": plan.channels,
        # outlines in µm, per view: its polygons with their slice positions
        # along the view's depth axis (null: one polygon through the box),
        # or, for polygons drawn on a projection, the projected depth range
        "outlines": _outline_records(plan.box.outlines),
        # how the outlines make the kept region (0.7.0): one 3D shape through
        # all polygons on slices, inside all polygons through the box
        # (0.5.0-0.6.0: "any" polygon; crops before 0.5.0 have no entry and
        # used one polygon per view, combined as "all")
        "outlines_combine": "all",
        "outlines_interpolation": (
            "one smooth 3D shape through all polygons drawn on slices "
            "(variational implicit surface: radial basis functions, cubic "
            f"kernel r^3, on a grid of {FIELD_POINTS} points along the "
            "polygons' longest extent, then linear interpolation to each "
            "voxel), without holes; it ends at the outermost polygons. "
            "Polygons drawn on a projection (silhouettes) reach through the "
            "projected depth range; a voxel is kept if it is inside the "
            "shape and inside all of these"),
        # every n-th timepoint of the output range (1 = all; 0.11.0)
        "t_step": plan.box.t_step,
        "outline_margin_voxels": plan.box.margin,
        "outline_margin": plan.box.margin * pixel_size(image),
        "source_levels": [image.levels[i].path for i in plan.level_indices],
        "level0_start": plan.starts[0],
        "level0_stop": plan.stops[0],
        "keep_position": keep_position,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "software": {
            "volume_cropper": __version__,
            "zarr": zarr.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    # A region that moves over time (0.17.0): where each output timepoint
    # was read (the OME translation holds only the first), and the track
    # as asked (so that the region can be made again).
    if plan.moving:
        record["moving_region"] = {
            "description": (
                "the region moves with the object over time: each output "
                "timepoint was read at the region's place then (origin = "
                "the position of the crop's first voxel in the source, as "
                "the OME translation, which holds the first timepoint's); "
                "voxels outside the source are the fill value"),
            "timepoints": [{"time": time, "origin": origin}
                           for time, origin in moving_origins(image, plan)],
            "requested_track": [{"time": time, "shift": shifts}
                                for time, shifts in plan.box.track.items()],
        }
        if plan.box.timed_outlines:
            record["moving_region"]["outlines_per_timepoint"] = {
                "description": (
                    "each timepoint's own outlines (µm, at the object's "
                    "positions then; as 'outlines'); voxels outside them "
                    "are 0. Timepoints not listed keep the whole region"),
                "timepoints": [{"time": time,
                                "outlines": _outline_records(outlines)}
                               for time, outlines in
                               plan.box.timed_outlines.items()],
            }
    # A file read into memory (0.12.0): which file, which of its images,
    # what was read of it and the calibration used.
    if image.source is not None:
        record["source_file"] = image.source
    # OME-Zarr 0.6 (0.21.0): the coordinate system the crop was made in,
    # and what became of the image's links to other systems.
    if image.coordinate_system is not None:
        kept, _, notes = _crop_extra_transforms(
            image, _crop_shift(image, plan, keep_position))
        record["coordinate_system"] = {
            "cropped_in": image.coordinate_system,
            "links_kept": len(kept),
            "notes": notes,
        }
    # Cropping a crop: keep the source's own record, so that the chain back
    # to the original image is never lost.
    if "crop_provenance" in image.attrs:
        record["source_provenance"] = image.attrs["crop_provenance"]
    return record


def moving_origins(image: OmeZarrImage, plan: CropPlan) -> list:
    """For a moving region: (time, {spatial axis: position}) per output
    timepoint, the position (µm) of the crop's first voxel in the source
    then, at the crop's first level (as its OME translation)."""
    k_level = image.levels[plan.level_indices[0]]
    result = []
    for j, time in enumerate(plan.times):
        origin = {}
        for i, axis in enumerate(image.axes):
            if axis.get("type") == "space":
                origin[axis["name"]] = round(
                    k_level.translation[i]
                    + (plan.starts[0][i] + int(plan.shifts[0][j, i]))
                    * k_level.scale[i], 9)
        result.append((time, origin))
    return result


def positions_path_for(crop_path: str | Path) -> Path:
    """The positions file of a moving region's crop (0.17.0), next to it:
    <crop>_positions.csv (e.g. pos3_embryo1_positions.csv)."""
    crop_path = Path(crop_path)
    return crop_path.with_name(f"{crop_stem(crop_path)}_positions.csv")


def write_positions_csv(image: OmeZarrImage, plan: CropPlan,
                        crop_path: str | Path) -> Path | None:
    """For a moving region (0.17.0): save where each timepoint of the crop
    lies in the source, as positions_path_for(crop_path): one row per
    timepoint of the crop, with its number (1, 2, ...), its time in the
    source and the position of the crop's first voxel in the source per
    spatial axis (the OME translation's convention: voxel centres), e.g.
    for Fiji or to put measurements back in the source's coordinates.
    A fixed region has no such file: an old one of a replaced crop is
    deleted. Returns the file's path (None for a fixed region)."""
    path = positions_path_for(crop_path)
    if path.exists() and not _our_positions_file(path):
        raise FileExistsError(f"{path} exists and was not written by Volume "
                              "Cropper; it was left as it is.")
    if not plan.moving:
        if path.is_file():
            path.unlink()
        return None
    origins = moving_origins(image, plan)
    t_name = image.axis_names[plan.t_axis]
    spatial = list(origins[0][1])
    columns = ["timepoint", _unit_column(t_name, image)] + [
        _unit_column(axis, image) for axis in spatial]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for j, (time, origin) in enumerate(origins, start=1):
            writer.writerow([j, _exact(time)]
                            + [_exact(origin[axis]) for axis in spatial])
    return path


def _unit_column(axis: str, image: OmeZarrImage) -> str:
    """A CSV column name with the axis' unit, e.g. "z_um" or "t_min"."""
    unit = short_unit(image.unit(axis)).replace("µ", "u").replace("μ", "u")
    return f"{axis}_{unit}" if unit else axis


def _check_output_path(image: OmeZarrImage, out_path: Path, overwrite: bool,
                       box_name: str | None = None) -> None:
    """Refuse outputs that could damage the source (the image) or other
    data."""
    source = image.path
    src, out = source.resolve(), out_path.resolve()
    if out == src or src in out.parents or out in src.parents:
        raise ValueError(f"Output {out_path} overlaps the source {source}.")
    if out_path.suffix.lower() in ZIP_SUFFIXES and out_path.is_dir():
        raise FileExistsError(
            f"{out_path} is a folder, not a zipped crop; refusing to replace "
            "it. Rename the region, or move the folder.")
    if is_zipped(out_path) and out_path.exists():               # (0.21.0)
        if not overwrite:
            raise FileExistsError(
                f"{out_path} already exists. To replace it, use the "
                "overwrite option (--overwrite on the command line, "
                "OVERWRITE = True in the notebook, 'Overwrite earlier crops' "
                "in the window).")
        record = zip_record(out_path)
        if not isinstance(record, dict) or not record.get("source"):
            raise FileExistsError(
                f"{out_path} exists and was not written by Volume Cropper; "
                "refusing to overwrite it.")
        _refuse_other_crop(out_path, record, box_name, _image_identity(image))
        return
    if out_path.exists() and not out_path.is_dir():
        raise FileExistsError(
            f"{out_path} is a file, not a crop folder; refusing to replace "
            "it. Rename the region, or move the file.")
    if out_path.exists() and any(out_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{out_path} already exists. To replace it, use the "
                "overwrite option (--overwrite on the command line, "
                "OVERWRITE = True in the notebook, 'Overwrite earlier crops' "
                "in the window).")
        try:   # only ever replace something this tool wrote ...
            record = zarr.open_group(out_path, mode="r").attrs.asdict().get(
                "crop_provenance")
        except Exception:
            record = None
        # (a record has at least the source; an empty or odd one does not
        # make a folder ours: overwriting deletes everything in it)
        if not isinstance(record, dict) or not record.get("source"):
            raise FileExistsError(
                f"{out_path} exists and was not written by Volume Cropper; "
                "refusing to overwrite it.")
        # ... and only a crop of the same box of the same image
        _refuse_other_crop(out_path, record, box_name, _image_identity(image))


def _check_positions_output(crop_path: Path) -> None:
    """A moving region's crop gets <crop>_positions.csv, and replacing a
    crop deletes the old one's (write_positions_csv): refuse when a file
    of that name is there that this tool did not write (the 0.17.0 review:
    it was deleted without a word)."""
    path = positions_path_for(crop_path)
    if path.exists() and not _our_positions_file(path):
        raise FileExistsError(
            f"{path} exists and was not written by Volume Cropper (a crop "
            f"of this region would replace or delete it). Rename the "
            "region, or move the file.")


def _our_positions_file(path: Path) -> bool:
    """Whether a file is a positions file this tool wrote (its first
    column is "timepoint")."""
    try:
        with open(path, encoding="utf-8") as f:
            return path.is_file() and f.readline().startswith("timepoint,")
    except (OSError, UnicodeDecodeError):
        return False


def _format_differences(src: zarr.Array, dst: zarr.Array) -> list[str]:
    """The storage-format settings in which `dst` differs from `src`: for
    Zarr v3 the codecs (compression, shards); for v2 the compressor,
    filters, order and chunk file names (dimension_separator)."""
    if src.metadata.zarr_format != dst.metadata.zarr_format:
        return ["Zarr version"]
    if src.metadata.zarr_format == 3:
        keys = ["codecs"]
    else:
        keys = ["compressor", "filters", "order", "dimension_separator"]
    a, b = src.metadata.to_dict(), dst.metadata.to_dict()
    return [key for key in keys if a.get(key) != b.get(key)]


def short_unit(unit: str) -> str:
    return _UNIT_ABBREVIATIONS.get(unit, unit)


def nbytes(shape, dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1000


def _fmt_list(values) -> str:
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

CSV_HELP = """\
The boxes file (CSV) has one row per box, for example:

  name,z_start_um,z_stop_um,y_start_um,y_stop_um,x_start_um,x_stop_um,channels
  tail1,100,350,200,800,150,900,all
  tail2,,,0,400,0,400,0;2

Ranges are in the image's units (here um). Leave both cells of an axis
empty to keep the whole axis. channels: all, or channel numbers such as 0;2.
An optional column t_step keeps every n-th timepoint of the t range (empty
or 1: all). An optional column track makes a region move over time (one
time-lapse crop that follows an object), e.g.
"t=0: z=0 y=0 x=0 | t=15: z=0 y=-2 x=12.5": its shift (um) at each time,
in between interpolated (written by the Autocropper).
"Save boxes..." in the Volume Cropper window writes such a file; Excel's
own CSV formats (semicolons, decimal commas) are read too.

Exit codes: 0 all crops written and checked; 1 a check after a crop found
problems; 2 nothing or not everything was written (see the message);
130 interrupted (Ctrl+C)."""


def main(argv: list[str] | None = None) -> int:
    """The command line. Returns the exit code (see CSV_HELP)."""
    parser = argparse.ArgumentParser(
        prog="python <tool folder>/volume_cropper/core.py",
        description="Crop the boxes listed in a CSV file out of an OME-Zarr "
                    "image, and save each box as a new OME-Zarr in the same "
                    "format. Every box is checked before the first one is "
                    "written.",
        epilog=CSV_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="the image to crop: a .zarr folder "
                        "(OME-Zarr 0.6, 0.5 or 0.4; crops keep its "
                        "version), or (0.21.0) a zipped OME-Zarr (.ozx, "
                        ".zip), whose crops are zipped too, or (0.12.0) a "
                        "TIFF file or a folder of TIFFs, which is read into "
                        "memory and whose crops are TIFFs. It is only read.")
    parser.add_argument("boxes_csv", help="CSV file with one row per box "
                        "(see below)")
    parser.add_argument("out_dir", help="folder for the crops, one "
                        "<image>_<box>.zarr per box; made if needed, and "
                        "never inside the source")
    parser.add_argument("--names", nargs="+", metavar="NAME",
                        help="crop only these boxes (default: all)")
    parser.add_argument("--collection-image", metavar="PATH",
                        help="(0.21.0) for a collection of images (a "
                        "bioformats2raw layout, an OME-Zarr 0.6 scene): "
                        "which image to crop, by its path in the "
                        "collection, e.g. 0 (needed when it holds several; "
                        "the error message lists them)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parts written at the same time (default 8)")
    parser.add_argument("--min-size", type=int, default=64, metavar="VOXELS",
                        help="keep pyramid levels while the crop is at "
                        "least this many voxels along its longest axis "
                        "(default 64)")
    parser.add_argument("--no-keep-position", action="store_true",
                        help="start each crop at 0 instead of keeping its "
                        "position in the image")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace earlier crops of the same boxes (only "
                        "crops made by this tool); their TIFFs are deleted "
                        "first")
    parser.add_argument("--tiff", type=int, metavar="LEVEL",
                        help="also save this level of each crop (0 = full "
                        "resolution; a crop with fewer levels gives its "
                        "coarsest) as an ImageJ TIFF for Fiji, next to the "
                        "crop: <image>_<box>_level<k>.tif (uncompressed)")
    parser.add_argument("--compress", action="store_true",
                        help="compress TIFFs with zstd (lossless, with the "
                        "predictor; Fiji opens them through Bio-Formats)")
    files = parser.add_argument_group(
        "image files (TIFF ...; 0.12.0)",
        "A file's image is read into memory (all of it), and each crop is "
        "saved as a TIFF of the file's kind: <image>_<box>.tif, or .ome.tif "
        "for an OME-TIFF. The calibration comes from the file; these "
        "replace it or fill in what it lacks.")
    files.add_argument("--image", type=int, default=1, metavar="N",
                       help="which image of the file (1 = the first)")
    files.add_argument("--axes", metavar="AXES",
                       help="what the file's axes are, in its order, e.g. "
                       "TYX for a plain TIFF stack that is a movie (an axis "
                       "the file names cannot be changed)")
    files.add_argument("--voxel-size", type=float, nargs=3,
                       metavar=("Z", "Y", "X"),
                       help="voxel size in --unit (default micrometer)")
    files.add_argument("--unit", choices=["micrometer", "nanometer",
                                          "millimeter"])
    files.add_argument("--frame-interval", type=float, metavar="T",
                       help="time between timepoints, in --time-unit")
    files.add_argument("--time-unit", choices=["second", "millisecond",
                                               "minute", "hour"])
    parser.add_argument("--dry-run", action="store_true",
                        help="only check the boxes and show what would be "
                        "written")
    parser.add_argument("--version", action="version",
                        version=f"Volume Cropper {__version__}")
    args = parser.parse_args(argv)
    if args.tiff is not None and args.tiff < 0:
        parser.error("--tiff: the level must be 0 or more (0 = full "
                     "resolution)")
    try:
        return _command_line(args)
    except KeyboardInterrupt:
        print("\nInterrupted. The crop being written is incomplete; run "
              "again with --overwrite to replace it.", file=sys.stderr)
        return 130
    except (ValueError, OSError, NotImplementedError) as err:
        # Problems with the input or the disk: one clear line, no traceback.
        if is_disk_full(err):
            message = ("the output drive is full. The crop being written is "
                       "incomplete; free some space and run again with "
                       "--overwrite.")
        else:
            message = str(err)
        print(f"Error: {message}", file=sys.stderr)
        return 2


def _command_line(args) -> int:
    """Check every box, then crop them one after the other."""
    for path, what in ((args.source, "image"), (args.boxes_csv, "boxes file")):
        if not Path(path).exists():
            raise ValueError(f"the {what} {path} does not exist.")
    from volume_cropper import file_images              # (imports this module: not at the top)
    compression = "zstd" if args.compress else None
    boxes = load_boxes_csv(args.boxes_csv)      # its mistakes first
    if file_images.is_file_image(args.source):
        if args.tiff is not None:
            raise ValueError("--tiff is for OME-Zarr images: the crops of a "
                             "file are TIFFs themselves.")
        return _command_line_file(args, boxes, compression)
    image = open_image(args.source, inner=args.collection_image)
    if args.names:
        unknown = [n for n in args.names if n not in [b.name for b in boxes]]
        if unknown:
            raise ValueError(
                f"{Path(args.boxes_csv).name} has no box called "
                f"{', '.join(map(repr, unknown))}; its boxes are "
                f"{', '.join(repr(b.name) for b in boxes)}.")
        boxes = [b for b in boxes if b.name in args.names]
    print(f"Volume Cropper {__version__}")
    print(image.describe())

    # Every box, before the first one is written.
    check_output_folder(image, args.out_dir)
    plans = [plan_crop(image, box, min_size=args.min_size) for box in boxes]
    out_paths = [output_path_for(image, box, args.out_dir) for box in boxes]
    for box, out_path in zip(boxes, out_paths):
        check_output(image, box, out_path, args.overwrite)
    dtype = image.levels[0].array.dtype
    tiff_levels = [None if args.tiff is None else tiff_level_for(plan, args.tiff)
                   for plan in plans]
    if args.tiff is not None:
        check_tiff_image(image)
    for plan, out_path, level in zip(plans, out_paths, tiff_levels):
        if level is not None:
            check_tiff_output(tiff_path_for(out_path, level), args.overwrite,
                              plan.box.name, image=image)
    for plan, out_path, level in zip(plans, out_paths, tiff_levels):
        print(f"\n{plan.describe(image)}\n  -> {out_path}")
        if level is not None:
            print(f"  -> {tiff_path_for(out_path, level).name}: "
                  f"{_tiff_note(plan, args.tiff, level)}, "
                  f"{format_bytes(tiff_bytes(plan, dtype, level))}")
    if args.dry_run:
        print("\nDry run: every box was checked; nothing was written.")
        return 0

    total = sum(nbytes(shape, dtype) for plan in plans for shape in plan.shapes)
    total += sum(tiff_bytes(plan, dtype, level)
                 for plan, level in zip(plans, tiff_levels) if level is not None)
    folder = Path(args.out_dir).absolute()
    existing = next(p for p in (folder, *folder.parents) if p.exists())
    free = shutil.disk_usage(existing).free
    if total > free:
        print(f"\nWarning: the crops need up to {format_bytes(total)} "
              "(uncompressed; compressed crops are usually smaller), but the "
              f"output drive has {format_bytes(free)} free.")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    failed = 0
    for plan, out_path, level in zip(plans, out_paths, tiff_levels):
        print(f"\nCropping {plan.box.name!r} ...")
        if args.overwrite:      # TIFFs of the crop being replaced
            for path in remove_old_tiffs(out_path, plan.box.name, image):
                print(f"  deleted the old TIFF {path.name}")
        t0 = time.perf_counter()
        if is_zipped(out_path):
            # (0.21.0) a zipped crop is checked before it is packed and
            # its TIFF made from its folder: crop_box does all of it
            result = crop_box(
                image, plan.box, args.out_dir, min_size=args.min_size,
                tiff_level=args.tiff, compression=compression,
                workers=args.workers,
                keep_position=not args.no_keep_position,
                overwrite=args.overwrite, progress=_print_progress,
                pack_progress=_percent_printer())
            print(f"  done in {time.perf_counter() - t0:.0f} s; "
                  + verification_text(plan, result.problems))
            failed += bool(result.problems)
            if result.positions is not None:
                print(f"  where each timepoint lies in the source: "
                      f"{result.positions.name}")
            if result.tiff is not None:
                print(f"  {result.tiff.name}: "
                      f"{_tiff_note(plan, args.tiff, level)}; "
                      + ("checked OK" if not result.tiff_problems else
                         "CHECK FAILED: " + "; ".join(result.tiff_problems)))
                failed += bool(result.tiff_problems)
            continue
        write_crop(image, plan, out_path, workers=args.workers,
                   keep_position=not args.no_keep_position,
                   overwrite=args.overwrite, progress=_print_progress)
        problems = verify_crop(image, plan, out_path,
                               keep_position=not args.no_keep_position)
        print(f"  done in {time.perf_counter() - t0:.0f} s; "
              + verification_text(plan, problems))
        failed += bool(problems)
        positions = write_positions_csv(image, plan, out_path)
        if positions is not None:       # a moving region (as crop_box)
            print(f"  where each timepoint lies in the source: "
                  f"{positions.name}")
        if level is not None:
            print(f"Saving level {level} as TIFF ...")
            t0 = time.perf_counter()
            tiff = export_tiff(out_path, level, overwrite=args.overwrite,
                               compression=compression,
                               progress_bytes=_percent_printer())
            tiff_problems = verify_tiff(out_path, tiff, level)
            print(f"  {tiff.name}: {_tiff_note(plan, args.tiff, level)}, "
                  f"done in {time.perf_counter() - t0:.0f} s; "
                  + ("checked OK" if not tiff_problems else
                     "CHECK FAILED: " + "; ".join(tiff_problems)))
            failed += bool(tiff_problems)
    return 1 if failed else 0


def _select_boxes(boxes: list[CropBox], names, csv_path) -> list[CropBox]:
    """The boxes named on the command line (--names), or all of them."""
    if not names:
        return boxes
    unknown = [n for n in names if n not in [b.name for b in boxes]]
    if unknown:
        raise ValueError(
            f"{Path(csv_path).name} has no box called "
            f"{', '.join(map(repr, unknown))}; its boxes are "
            f"{', '.join(repr(b.name) for b in boxes)}.")
    return [b for b in boxes if b.name in names]


def _command_line_file(args, boxes: list[CropBox],
                       compression: str | None) -> int:
    """The command line for an image file (0.12.0): read it into memory,
    check every box, then save each crop as a TIFF."""
    from volume_cropper import file_images as fi

    boxes = _select_boxes(boxes, args.names, args.boxes_csv)
    print(f"Volume Cropper {__version__}")
    reader = fi.open_file(args.source)
    try:
        print(fi.describe(reader))
        if not 1 <= args.image <= len(reader.scenes):
            raise ValueError(f"--image: the file has images 1–"
                             f"{len(reader.scenes)}.")
        voxel = (dict(zip("zyx", args.voxel_size)) if args.voxel_size
                 else None)
        choice = fi.FileChoice(
            scene=args.image - 1, axes=args.axes, voxel=voxel,
            unit=args.unit, interval=args.frame_interval,
            time_unit=args.time_unit)
        text, refuse = fi.memory_problem(fi.memory_needed(reader, choice),
                                         fi.free_memory())
        print(text)
        if refuse:
            raise ValueError("not enough memory to read the image.")
        # (a dry run reads it too: the boxes are checked against the image)
        print("Reading it into memory ...")
        image = fi.load(reader, choice, progress=_percent_printer_3)
    finally:
        reader.close()

    check_output_folder(image, args.out_dir)
    plans = [plan_crop(image, box, n_levels=1) for box in boxes]
    out_paths = [output_path_for(image, box, args.out_dir) for box in boxes]
    check_tiff_image(image)
    for box, out_path in zip(boxes, out_paths):
        check_output(image, box, out_path, args.overwrite)
    for plan, out_path in zip(plans, out_paths):
        print(f"\n{plan.describe(image)}\n  -> {out_path}")
    if args.dry_run:
        print("\nDry run: every box was checked; nothing was written.")
        return 0
    failed = 0
    for plan in plans:
        print(f"\nCropping {plan.box.name!r} ...")
        result = crop_box(image, plan.box, args.out_dir,
                          compression=compression, overwrite=args.overwrite,
                          workers=args.workers,
                          keep_position=not args.no_keep_position,
                          tiff_progress=_percent_printer())
        print(f"  {result.path.name}: done in {result.seconds:.0f} s; "
              + ("checked OK" if not result.problems else
                 "CHECK FAILED: " + "; ".join(result.problems[:5])))
        failed += bool(result.problems)
    return 1 if failed else 0


_last_stage = {"stage": None, "tenth": -1}


def _percent_printer_3(done: int, total: int, stage: str) -> None:
    """Progress while a file is read: every 10 %, with the stage."""
    tenth = int(10 * done / total) if total else 10
    if stage != _last_stage["stage"] or tenth != _last_stage["tenth"]:
        _last_stage.update(stage=stage, tenth=tenth)
        print(f"  {10 * tenth} % ({stage})", flush=True)


def _tiff_note(plan: CropPlan, asked: int, level: int) -> str:
    """'level 1', or why another level is used (the crop has fewer)."""
    if level == asked:
        return f"level {level}"
    return (f"level {level} (level {asked} asked; this crop has levels "
            f"0–{len(plan.level_indices) - 1})")


def _percent_printer() -> Callable[[int, int], None]:
    """A progress_bytes callback that prints every 10 %."""
    last = [-1]

    def report(done: int, total: int) -> None:
        tenth = int(10 * done / total) if total else 10
        if tenth != last[0]:
            last[0] = tenth
            print(f"  {10 * tenth} %", flush=True)
    return report


def is_disk_full(err: BaseException) -> bool:
    """Whether an error means that the disk is full."""
    return isinstance(err, OSError) and (
        err.errno == errno.ENOSPC
        or getattr(err, "winerror", None) in (39, 112))  # Windows: disk full


def _print_progress(done: int, total: int) -> None:
    step = max(total // 20, 1)        # about every 5 %
    if done % step == 0 or done == total:
        print(f"  {done}/{total} blocks", flush=True)


if __name__ == "__main__":
    if not __package__:
        # Started as a file (python <tool folder>/volume_cropper/core.py ...,
        # as the user guide shows): put the tool folder at the front of
        # sys.path instead of this package folder, as __main__.py does, so
        # that the package's modules are found as volume_cropper.*.
        _package_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path[:] = [entry for entry in sys.path
                       if os.path.normcase(os.path.abspath(entry or os.curdir))
                       != os.path.normcase(_package_dir)]
        sys.path.insert(0, os.path.dirname(_package_dir))
    # Run the package's own copy of this module (volume_cropper.core), not
    # this one (__main__): the modules the command line uses (file_images,
    # ...) import volume_cropper.core, and two copies of the same classes
    # would not recognise each other's objects.
    from volume_cropper.core import main as _package_main
    sys.exit(_package_main())
