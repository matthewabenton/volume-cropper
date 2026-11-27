"""The new readers (0.15.0) on microscope files of your own (read-only;
printed only; nothing written anywhere but memory).

Usage: python readers_0_15_check.py FILE.lif [FILE.czi ...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volume_cropper import file_images as fi  # noqa: E402

FILES = [Path(p) for p in sys.argv[1:]]
ok = True
for path in FILES:
    reader = fi.open_file(path)
    try:
        print(f"\n{path.name}: {reader.format_name}, {reader.reader_name}, "
              f"compression {reader.compression}, output {reader.output_kind}")
        for k, scene in enumerate(reader.scenes):
            print(f"  scene {k} {scene.name!r}: axes {scene.axes} shape "
                  f"{scene.shape} {scene.dtype}; voxel "
                  + " x ".join(f"{scene.voxel[a]:.6g}" if scene.voxel[a]
                               else "?" for a in "zyx")
                  + f" {scene.unit}; interval {scene.interval} "
                  f"{scene.time_unit}; names {scene.channel_names} colours "
                  f"{scene.channel_colors}; notes {scene.notes}")
        # a part of the first scene, through load(), against the library
        scene = reader.scenes[0]
        axes = fi.default_axes(scene)
        sizes = fi.sizes_of(scene, axes)
        ranges = {}
        for letter in "TZYX":
            if letter in sizes:
                n = sizes[letter]
                ranges[letter.lower()] = (n // 4, n // 4 + max(1, n // 3))
        choice = fi.FileChoice(ranges=ranges)
        t0 = time.time()
        image = fi.load(reader, choice)
        seconds = time.time() - t0
        got = np.asarray(image.levels[0].array[:])
        key = []
        for letter in axes:
            if letter == "-":
                key.append(0)
            elif letter == "C":
                key.append(slice(None))
            else:
                a, b = ranges[letter.lower()]
                key.append(slice(a, b))
        expected = np.asarray(reader.read(0, tuple(key)))
        # bring the library's part into load()'s order: (t, c, z, y, x)
        kept = [a for a in axes if a != "-"]
        order = [kept.index(a) for a in "TCZYX" if a in kept]
        expected = expected.transpose(order)
        if "Z" not in kept:
            expected = np.expand_dims(expected, axis=-3)
        same = expected.shape == got.shape and np.array_equal(expected, got)
        ok &= same
        print(f"  part {ranges}: {got.shape}, read in {seconds:.2f} s, "
              f"equal to the library's read: {same}; level-0 scale "
              f"{image.levels[0].scale}; pyramid levels {len(image.levels)}")
    finally:
        reader.close()
print("\nALL OK" if ok else "\nPROBLEM")
