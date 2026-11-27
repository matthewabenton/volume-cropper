"""Tests for stitching tile scans (tile_stitching, 0.17.2; LIF tile scans
in file_images.LifReader).

Run with ``python run_all_tests.py stitching`` (or this file alone). Real tile
scans are not part of the tool, so these tests
cut synthetic tiles, with known true places, out of a synthetic sample (a
textured 3D volume) and check that:
- the NCC of every shift (ncc_map, by FFTs) equals the NCC computed
  directly over the overlap;
- the tiles are put back exactly where they were cut, although the
  "stage" is turned 0.8 degrees against the image and errs by a few
  pixels (as seen on real tile scans), one tile is moved in z and one
  channel is empty; the stitched image then equals the sample, voxel for
  voxel;
- the orientation of a stage reversed in x is found from the overlaps;
- where tiles overlap, each voxel comes from the tile whose centre is
  nearest;
- tiles with nothing to match (background only) stay at their stage
  places, with a note;
- a LIF tile scan opens as two images, stitched and tiles, through
  LifReader and load(), and the crop's record says how it was stitched;
  an image merged by LAS X gets a note.
The method was also checked on real tile scans, against LAS X's
placements (dev/stitching_0_17_2_check.py).
"""
# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
from pathlib import Path

import numpy as np
from scipy import ndimage

from volume_cropper import file_images as fi
from volume_cropper import tile_stitching as ts

PIXEL = 0.5e-6            # metres, as liffile gives it
TILE = (12, 160, 160)     # z, y, x of one tile
STEP = 136                # tile to tile on the stage (24 pixels overlap)


def _sample(shape, seed=1) -> np.ndarray:
    """A textured volume (smoothed noise, structure of a few pixels)."""
    rng = np.random.default_rng(seed)
    smooth = ndimage.gaussian_filter(rng.random(shape), (1.0, 2.5, 2.5))
    smooth = (smooth - smooth.min()) / np.ptp(smooth)
    return (100 + 3000 * smooth).astype(np.uint16)


def _mosaic(rows=3, cols=3, seed=2, z_moves=None, flat=()):
    """Tiles cut from a sample at known places.

    Returns (tiles (n, z, y, x), stage (n, 2) metres, true places (n, 3),
    sample). The true places are the stage grid turned by 0.8 degrees plus
    up to 3 pixels of error; z_moves: {tile: planes}; flat: tiles that are
    background only (their part of the sample is set to 100)."""
    rng = np.random.default_rng(seed)
    angle = np.radians(0.8)
    turn = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    nominal = np.array([(r * STEP, c * STEP) for r in range(rows)
                        for c in range(cols)], dtype=np.float64)
    true_yx = np.rint(nominal @ turn.T + rng.uniform(-3, 3, nominal.shape)
                      ).astype(int) + 20
    true = np.zeros((len(nominal), 3), int)
    true[:, 1:] = true_yx
    for k, dz in (z_moves or {}).items():
        true[k, 0] = dz
    size = true.max(axis=0) + np.array(TILE) + 20
    sample = _sample(tuple(size), seed)
    for k in flat:                      # background where this tile is
        z, y, x = true[k]
        sample[:, y - 10:y + TILE[1] + 10, x - 10:x + TILE[2] + 10] = 100
    tiles = np.stack([sample[z:z + TILE[0], y:y + TILE[1], x:x + TILE[2]]
                      for z, y, x in true])
    return tiles, nominal * PIXEL, true, sample


def _reader(tiles):
    def read(k, c, zs, ys, xs):
        return tiles[k][zs, ys, xs]
    return read


def test_ncc_map_equals_direct():
    rng = np.random.default_rng(0)
    fixed, moving = rng.random((7, 9)), rng.random((4, 5))
    ncc, count = ts.ncc_map(fixed, moving)
    for sy in range(-3, 7):
        for sx in range(-4, 9):
            # moving's voxel 0 on fixed's voxel (sy, sx)
            y0, y1 = max(0, sy), min(7, sy + 4)
            x0, x1 = max(0, sx), min(9, sx + 5)
            f = fixed[y0:y1, x0:x1].ravel()
            m = moving[y0 - sy:y1 - sy, x0 - sx:x1 - sx].ravel()
            j = (sy + 3, sx + 4)
            assert count[j] == f.size
            if f.size >= 2:
                direct = np.corrcoef(f, m)[0, 1]
                assert abs(ncc[j] - direct) < 1e-9, (sy, sx)


def test_tiles_go_back_where_they_were_cut():
    tiles, stage, true, sample = _mosaic(z_moves={4: 2, 7: 1})
    # two channels: the sample, and an empty one (never matched on)
    both = np.stack([tiles, np.zeros_like(tiles)], axis=1)

    def read(k, c, zs, ys, xs):
        return both[k, c][zs, ys, xs]

    layout = ts.register(read, TILE, stage, PIXEL, n_channels=2,
                         orientation=(False, False, False))
    expected = true - true.min(axis=0)
    assert np.array_equal(layout.positions, expected), \
        (layout.positions - expected)
    assert layout.record["channel_matched"] == 0
    assert len(layout.record["links"]) == 12          # every overlap
    assert abs(layout.record["model_angle_degrees"] - 0.8) < 0.3
    assert any("moved in z" in note for note in layout.notes)
    # The stitched image is the sample, voxel for voxel, wherever a tile
    # has it.
    shape = layout.shape
    out = ts.assemble(lambda k, zs, ys, xs: tiles[k][zs, ys, xs], layout,
                      slice(0, shape[0]), slice(0, shape[1]),
                      slice(0, shape[2]), tiles.dtype)
    covered = np.zeros(shape, bool)
    for z, y, x in layout.positions:
        covered[z:z + TILE[0], y:y + TILE[1], x:x + TILE[2]] = True
    origin = true.min(axis=0)
    truth = sample[origin[0]:origin[0] + shape[0],
                   origin[1]:origin[1] + shape[1],
                   origin[2]:origin[2] + shape[2]]
    assert np.array_equal(out[covered], truth[covered])
    assert not out[~covered].any()                     # 0 where no tile
    # a part of it, as load() reads it (some planes, all of y and x)
    part = ts.assemble(lambda k, zs, ys, xs: tiles[k][zs, ys, xs], layout,
                       slice(3, 7), slice(50, 300), slice(20, 260),
                       tiles.dtype)
    assert np.array_equal(part, out[3:7, 50:300, 20:260])


def test_reversed_stage_is_found():
    tiles, stage, true, _ = _mosaic(rows=2, cols=3, seed=5)
    reversed_x = stage * np.array([1, -1])
    layout = ts.register(_reader(tiles), TILE, reversed_x, PIXEL,
                         orientation=None)
    assert layout.record["orientation_swap_flipy_flipx"] == [False, False,
                                                             True]
    assert np.array_equal(layout.positions, true - true.min(axis=0))
    assert "orientation" in layout.notes[0]


def test_overlaps_from_the_nearest_centre():
    # two tiles of 10 x 10, the second 6 pixels to the right: they share
    # columns 6-9; centres at x 4.5 and 10.5, so x 0-7 is the first's
    layout = ts._layout(np.array([[0, 0, 0], [0, 0, 6]]), (1, 10, 10),
                        [], {})
    tiles = [np.full((1, 10, 10), 1, np.uint8),
             np.full((1, 10, 10), 2, np.uint8)]
    out = ts.assemble(lambda k, zs, ys, xs: tiles[k][zs, ys, xs], layout,
                      slice(0, 1), slice(0, 10), slice(0, 16), np.uint8)
    assert out[0, 5].tolist() == [1] * 8 + [2] * 8


def test_background_tiles_stay_at_the_stage_places():
    tiles, stage, true, _ = _mosaic(rows=2, cols=2, seed=3)
    flat = np.full_like(tiles, 100)
    layout = ts.register(_reader(flat), TILE, stage, PIXEL,
                         orientation=(False, False, False))
    places = ts.stage_places(stage, PIXEL)
    assert np.array_equal(layout.positions[:, 1:],
                          np.rint(places).astype(int))
    assert layout.record["links"] == []
    assert any("no matching overlap" in note for note in layout.notes)
    # one background tile among others: it follows the model, the others
    # are where they were cut
    tiles, stage, true, _ = _mosaic(rows=3, cols=3, seed=4, flat=(8,))
    layout = ts.register(_reader(tiles), TILE, stage, PIXEL,
                         orientation=(False, False, False))
    expected = true - true.min(axis=0)
    good = np.arange(9) != 8
    assert np.array_equal(layout.positions[good], expected[good])
    assert np.abs(layout.positions[8] - expected[8]).max() <= 4
    assert layout.record["tiles_without_match"] == [8]


def _lif_image(name, dims, data, tilescan=None):
    """A stand-in for a liffile image (what LifReader uses of it)."""
    sizes = dict(zip(dims, data.shape))
    coords = {d: np.arange(sizes[d]) * (PIXEL if d != "Z" else 2e-6)
              for d in "ZYX" if d in sizes}
    return types.SimpleNamespace(
        name=name, path=name, dims=dims, shape=data.shape, dtype=data.dtype,
        coords=coords, xml_element=None, timestamps=None, tilescan=tilescan,
        asarray=lambda out=None: data)


def _lif_reader(*images):
    reader = fi.LifReader.__new__(fi.LifReader)
    reader.path = Path("tile scans.lif")
    reader._images, reader.scenes, reader._arrays = [], [], {}
    reader._tile_scans = {}
    reader.format_name, reader.reader_name = "Leica LIF", "liffile (test)"
    for image in images:
        reader._add(image)
    return reader


def test_lif_tile_scan_opens_stitched():
    tiles, stage, true, sample = _mosaic(rows=2, cols=2, seed=6)
    data = np.stack([tiles, tiles // 2], axis=1)           # M C Z Y X
    table = np.zeros(4, dtype=[("field_x", "i4"), ("field_y", "i4"),
                               ("pos_x", "f8"), ("pos_y", "f8"),
                               ("pos_z", "f8")])
    table["pos_y"], table["pos_x"] = stage[:, 0], stage[:, 1]
    scan = types.SimpleNamespace(tiles=table, flip_x=False, flip_y=False,
                                 swap_xy=False)
    reader = _lif_reader(
        _lif_image("Region 1", ("M", "C", "Z", "Y", "X"), data, scan),
        _lif_image("Region 1_Merged", ("C", "Z", "Y", "X"), data[0]))
    names = [scene.name for scene in reader.scenes]
    assert names == ["Region 1 (stitched)", "Region 1 (tiles)",
                     "Region 1_Merged"]
    assert reader.scenes[0].axes == "CZYX"
    assert "worked out when this image is chosen" in reader.scenes[0].notes[0]
    assert reader.scenes[1].axes[0] == "?"                 # the tiles
    assert any("merged by LAS X" in n for n in reader.scenes[2].notes)
    image = fi.load(reader, fi.FileChoice(scene=0))
    layout = reader._tile_scans[0].layout
    assert reader.scenes[0].shape == (2,) + layout.shape
    assert "Stitched from 4 tiles" in reader.scenes[0].notes[0]
    level0 = np.asarray(image.levels[0].array[:])
    origin = true.min(axis=0)
    z, y, x = layout.shape
    truth = sample[origin[0]:origin[0] + z, origin[1]:origin[1] + y,
                   origin[2]:origin[2] + x]
    covered = np.zeros(layout.shape, bool)
    for pz, py, px in layout.positions:
        covered[pz:pz + TILE[0], py:py + TILE[1], px:px + TILE[2]] = True
    assert np.array_equal(level0[0][covered], truth[covered])
    assert np.array_equal(level0[1][covered], truth[covered] // 2)
    record = image.source["stitching"]
    assert record["positions_zyx"] == layout.positions.tolist()
    assert record["flags_in_file"] == {"flip_x": False, "flip_y": False,
                                       "swap_xy": False}
    # a part only, and a 2D read (one plane as an index)
    part = reader.read(0, (1, slice(2, 5), slice(10, 200), slice(0, 90)))
    assert np.array_equal(part, level0[1, 2:5, 10:200, 0:90])
    plane = reader.read(0, (0, 3, slice(None), slice(None)))
    assert np.array_equal(plane, level0[0, 3])


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["stitching"]))
