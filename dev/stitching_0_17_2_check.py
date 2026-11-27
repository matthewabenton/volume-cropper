"""Check the stitching of LIF tile scans (0.17.2) on tile scans of your own
(read-only; prints numbers only, no pictures).

For each tile scan that LAS X also merged: how long placing the tiles
takes, how many overlaps matched, and how far each tile's place is from
LAS X's (LAS X's place of a tile = where the tile's middle matches best in
LAS X's merged image; LAS X registers its tiles too, so the two should
agree within a pixel or two in y and x). Then one channel of the stitched
image is read through load(), as the window would.

Usage (in the tool's environment):
    python <dev>/stitching_0_17_2_check.py FOLDER SCAN [SCAN ...]
SCAN = "<file>.lif:<tile-scan image index>:<merged image index>", the
indices as liffile counts the file's images, e.g. "scan.lif:0:1". The tool
is the folder this script is in, or the VC_TOOL environment variable.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from skimage.feature import match_template

FOLDER = Path(sys.argv[1])
TOOL = Path(os.environ.get("VC_TOOL", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(TOOL))
from volume_cropper import file_images as fi                                   # noqa: E402

SCANS = []  # (file, tile-scan image index, merged image index) in liffile
for arg in sys.argv[2:]:
    name, i_tiles, i_merged = arg.rsplit(":", 2)
    SCANS.append((name, int(i_tiles), int(i_merged)))


def plane(array, dims, **index):
    key = tuple(index.get(d, slice(None)) for d in dims)
    return np.asarray(array[key], dtype=np.float32)


def find(tile, merged, guess, search=60):
    """Where the tile's middle half matches best in `merged` near `guess`
    (y, x of the tile's corner): (y, x, ncc) or None."""
    h, w = tile.shape
    iy, ix = h // 4, w // 4
    inner = tile[iy:h - iy, ix:w - ix]
    if inner.std() < 1e-3:
        return None
    gy, gx = guess
    y0, x0 = max(0, gy + iy - search), max(0, gx + ix - search)
    y1 = min(merged.shape[0], gy + h - iy + search)
    x1 = min(merged.shape[1], gx + w - ix + search)
    window = merged[y0:y1, x0:x1]
    if window.shape[0] < inner.shape[0] or window.shape[1] < inner.shape[1]:
        return None
    score = match_template(window, inner)
    py, px = np.unravel_index(np.argmax(score), score.shape)
    return y0 + py - iy, x0 + px - ix, float(score[py, px])


summary = []
for name, i_tiles, i_merged in SCANS:
    reader = fi.LifReader(FOLDER / name)
    image = reader._lif.images[i_tiles]
    k = next(s for s in reader._tile_scans if reader._images[s] is image)
    tiles = reader._tile_scans[k]
    t0 = time.perf_counter()
    reader.prepare_scene(k)
    took = time.perf_counter() - t0
    layout, record = tiles.layout, tiles.layout.record
    print(f"\n=== {name} [{i_tiles}] {reader.scenes[k].name!r}: {tiles.n} "
          f"tiles {tiles.tile_shape}, placed in {took:.1f} s")
    for note in reader.scenes[k].notes:
        print("   note:", note)
    print(f"   model: angle {record.get('model_angle_degrees')} deg, matrix "
          f"{record.get('model_stage_to_image')}, search "
          f"{record.get('search_pixels')}, z search "
          f"{record.get('z_search_planes')}, dropped "
          f"{len(record.get('dropped_links', []))}")
    # LAS X's place of each tile (y, x; and its z shift where LAS X has more
    # planes), from its merged image, on the channel we matched on.
    merged = reader._lif.images[i_merged]
    mdims, dims = "".join(merged.dims), "".join(image.dims)
    mdata, data = merged.asarray(out="memmap"), image.asarray(out="memmap")
    c = record.get("channel_matched", 0)
    nz, mnz = tiles.tile_shape[0], dict(merged.sizes).get("Z", 1)
    tz = nz // 2
    offsets = range(0, mnz - nz + 1) if "Z" in dims else [0]
    mplanes = {o: plane(mdata, mdims, C=c, **({"Z": tz + o} if "Z" in mdims
                                                else {}))
               for o in offsets}
    lasx = {}
    for t in range(tiles.n):
        tile = plane(data, dims, M=t, C=c, **({"Z": tz} if "Z" in dims
                                               else {}))
        guess = layout.positions[t, 1:]
        best = None
        for o in offsets:
            for sy, sx in ((0, 0), (20, 20), (40, 40), (-20, -20)):
                r = find(tile, mplanes[o], (guess[0] + sy, guess[1] + sx))
                if r and (best is None or r[2] > best[2]):
                    best = (*r, o)
        if best and best[2] > 0.8:
            lasx[t] = best
    common = [t for t in lasx]
    if len(common) >= 2:
        ours = np.array([layout.positions[t, 1:] for t in common], float)
        theirs = np.array([lasx[t][:2] for t in common], float)
        diff = ours - theirs
        diff -= np.median(diff, axis=0)         # the mosaics' own origins
        d = np.abs(diff).max(axis=1)
        z_ours = np.array([layout.positions[t, 0] for t in common])
        z_lasx = np.array([lasx[t][3] for t in common])
        z_diff = (z_ours - z_ours.min()) - (z_lasx - z_lasx.min())
        print(f"   vs LAS X ({len(common)} of {tiles.n} tiles found in its "
              f"merged image): |y, x difference| median {np.median(d):.1f}, "
              f"90th percentile {np.percentile(d, 90):.1f}, max "
              f"{d.max():.1f} px; z: ours {sorted(set(z_ours.tolist()))}, "
              f"LAS X {sorted(set(z_lasx.tolist()))} planes")
        if tiles.n <= 12:
            for t, dd in zip(common, diff):
                print(f"     tile {t}: ours - LAS X  y {dd[0]:+.0f}, "
                      f"x {dd[1]:+.0f}; z ours {layout.positions[t, 0]}, "
                      f"LAS X {lasx[t][3]}")
        # Which offsets do the raw tiles support? For each matched pair
        # (both tiles found in LAS X's image): the NCC of the two raw tiles'
        # overlap (middle plane) at our offset and at LAS X's.
        ncc_ours, ncc_lasx = [], []
        for link in record.get("links", []):
            a, b = link["tiles"]
            if a not in lasx or b not in lasx:
                continue
            pa = plane(data, dims, M=a, C=c, **({"Z": tz} if "Z" in dims
                                                 else {}))
            pb = plane(data, dims, M=b, C=c, **({"Z": tz} if "Z" in dims
                                                 else {}))
            for offset, out in ((link["offset_zyx"][1:], ncc_ours),
                                ((lasx[b][0] - lasx[a][0],
                                  lasx[b][1] - lasx[a][1]), ncc_lasx)):
                dy, dx = (int(v) for v in offset)
                h, w = pa.shape
                ya, xa = (max(0, dy), min(h, h + dy)), (max(0, dx),
                                                         min(w, w + dx))
                if ya[1] - ya[0] < 2 or xa[1] - xa[0] < 2:
                    out.append(np.nan)
                    continue
                fa = pa[ya[0]:ya[1], xa[0]:xa[1]].ravel()
                fb = pb[ya[0] - dy:ya[1] - dy, xa[0] - dx:xa[1] - dx].ravel()
                out.append(np.corrcoef(fa, fb)[0, 1]
                           if fa.std() > 0 and fb.std() > 0 else np.nan)
        if ncc_ours:
            o, x = np.array(ncc_ours), np.array(ncc_lasx)
            print(f"   raw overlaps of {len(o)} matched pairs (middle "
                  f"plane): NCC at our offsets median {np.nanmedian(o):.2f},"
                  f" at LAS X's {np.nanmedian(x):.2f}; LAS X's within 0.05 "
                  f"of ours for {np.mean(x >= o - 0.05):.0%} of pairs")
        summary.append((name, i_tiles, tiles.n, len(record.get("links", [])),
                        round(took, 1), float(np.median(d)), float(d.max()),
                        round(float(np.nanmedian(ncc_ours)), 2)
                        if ncc_ours else None,
                        round(float(np.nanmedian(ncc_lasx)), 2)
                        if ncc_lasx else None))
    # One channel of the stitched image through load(), as the window does.
    t0 = time.perf_counter()
    loaded = fi.load(reader, fi.FileChoice(scene=k, channels=[c]))
    level0 = loaded.levels[0].array
    print(f"   load(): channel {c + 1}, {level0.shape} {level0.dtype} in "
          f"{time.perf_counter() - t0:.1f} s; stitching record kept: "
          f"{'stitching' in loaded.source}")
    reader.close()

print("\n=== summary: file, image, tiles, overlaps matched, seconds, "
      "|ours - LAS X| median, max (px), raw overlap NCC at our offsets, "
      "at LAS X's (medians)")
for row in summary:
    print("   ", row)
