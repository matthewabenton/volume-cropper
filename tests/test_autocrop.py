"""Tests for autocrop (0.14.0, experimental): finding a bright object and
the crop region around it, in 3D and over time.

Run with ``python run_all_tests.py autocrop`` (or this file alone). The
images are synthetic: a bright blob (an "embryo") in Poisson noise, with
hot pixels, a second smaller or dimmer object, or a blob that moves and is
missing at the first timepoint. They are made as napari layers and read
into memory like the plugin does (file_images.LayerReader + load), which
also gives them a pyramid. The tests check that:
- the box found holds the whole object and little more (within a voxel
  or two of its true edges), in µm, with the image's voxel size and
  position;
- over time the box holds the object at every timepoint, and the t range
  is cut to the timepoints where it was found;
- the object with the most signal wins (not hot pixels, not a smaller or
  a dimmer object), and a search inside a region finds the object there;
- the margin is added and kept within the image; flat images give no
  object; a coarser level still holds the object; Otsu, Triangle and Li
  all work; Cancel stops the search;
- (0.16.0) a seed picks its object (also a dimmer one, also from just
  beside it) and follows it through time from any timepoint; silhouettes
  keep every voxel of the object and cut the box's empty corners; in the
  panel, seeds clicked on two objects give two regions (a drag adds none,
  a click on a seed removes it), Outline gives regions silhouettes, and
  Undo fit removes new regions and gives back replaced polygons.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from volume_cropper import autocrop as ac
from volume_cropper import file_images as fi
from volume_cropper import core as vc

RNG = np.random.default_rng(7)


def _image(data, labels, scale, translate=None, names=None):
    """An in-memory image from arrays (one per channel), as the plugin
    reads napari layers."""
    from napari.layers import Image
    arrays = data if isinstance(data, list) else [data]
    translate = translate or (0.0,) * len(labels)
    units = tuple("minute" if a == "t" else "um" for a in labels)
    layers = [Image(a, axis_labels=labels, scale=scale, translate=translate,
                    units=units, name=(names or [f"ch{k}" for k in
                                                 range(len(arrays))])[k])
              for k, a in enumerate(arrays)]
    return fi.load(fi.LayerReader(layers), fi.FileChoice())


def _blob(shape, lo, hi, value=2000.0):
    """An ellipsoid filling the box [lo, hi) (voxel indices)."""
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    centre = [(a + b - 1) / 2 for a, b in zip(lo, hi)]
    radius = [(b - a) / 2 for a, b in zip(lo, hi)]
    inside = sum(((g - c) / r) ** 2 for g, c, r in zip(grids, centre, radius))
    return np.where(inside <= 1.0, value, 0.0)


def _noisy(signal, background=100):
    return (RNG.poisson(background, size=signal.shape) + signal).astype(
        np.uint16)


def _edges(lo, hi, scale, translate):
    """Physical outer edges of voxels [lo, hi) along one axis."""
    return (translate + (lo - 0.5) * scale, translate + (hi - 0.5) * scale)


def _holds(found, true, tolerance):
    """found holds true, and is at most `tolerance` larger on each side."""
    return (found[0] <= true[0] + 1e-6 and found[1] >= true[1] - 1e-6
            and true[0] - found[0] <= tolerance
            and found[1] - true[1] <= tolerance)


def test_one_blob_in_3d_with_voxel_size_and_position():
    shape, lo, hi = (40, 120, 100), (10, 30, 20), (26, 71, 51)
    data = _noisy(_blob(shape, lo, hi))
    data[5, 100, 90] = data[30, 10, 10] = 60000            # hot pixels
    scale, translate = (2.0, 0.5, 0.5), (100.0, 50.0, 20.0)
    image = _image(data, ("z", "y", "x"), scale, translate)
    result = ac.find_object(image, level=0)
    assert result.n_found == 1 and result.ranges, result.notes
    for k, axis in enumerate("zyx"):
        true = _edges(lo[k], hi[k], scale[k], translate[k])
        # within 4 voxels of the true edges (smoothing, growing down to
        # the background, one voxel of padding)
        assert _holds(result.ranges[axis], true, 4 * scale[k]), (
            axis, result.ranges[axis], true)
    assert "Found the object at 1 of 1" in ac.describe(result, image)


def test_moving_blob_over_time():
    shape = (5, 20, 64, 80)                       # t z y x
    signal = np.zeros(shape)
    for t in range(1, 5):                         # missing at t = 0
        signal[t] = _blob(shape[1:], (5, 20, 10 + 10 * t),
                          (15, 40, 25 + 10 * t))
    image = _image(_noisy(signal), ("t", "z", "y", "x"),
                   (10.0, 1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert result.found[0] is None and result.n_found == 4
    assert result.ranges["t"] == (10.0, 40.0)    # timepoints 2-5 (minutes)
    # x holds the blob at every timepoint: voxels 20 ... 64
    assert _holds(result.ranges["x"], _edges(20, 65, 1.0, 0.0), 4.0), \
        result.ranges["x"]
    assert any("not found at 1 of 5" in note for note in result.notes)
    # (0.18.0) one's own threshold: the same value at every timepoint (a
    # bug in the first version let each timepoint's result replace it)
    own = ac.find_object(image, level=0, threshold=1000.0)
    assert own.n_found == 4 and own.ranges["t"] == (10.0, 40.0)
    assert all(t == 1000.0 for t in own.thresholds if t is not None), \
        own.thresholds
    assert _holds(own.ranges["x"], _edges(20, 65, 1.0, 0.0), 4.0), \
        own.ranges["x"]


def test_the_object_with_most_signal_wins_and_regions_narrow_the_search():
    shape = (30, 100, 100)
    big = _blob(shape, (5, 10, 10), (25, 50, 50), 2000.0)
    small = _blob(shape, (10, 70, 70), (16, 80, 80), 3000.0)
    image = _image(_noisy(big + small), ("z", "y", "x"), (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert _holds(result.ranges["x"], _edges(10, 50, 1.0, 0.0), 4.0), \
        result.ranges["x"]
    # inside a region around the small one, the small one
    region = {"y": (60.0, 95.0), "x": (60.0, 95.0)}
    result = ac.find_object(image, level=0, region=region)
    assert _holds(result.ranges["x"], _edges(70, 80, 1.0, 0.0), 4.0), \
        result.ranges["x"]


def test_seed_picks_its_object_even_a_dimmer_one():
    """0.16.0: with a seed, the object under it (or right next to it), not
    the one with most signal; a seed far from any object finds none."""
    shape = (30, 100, 100)
    big = _blob(shape, (5, 10, 10), (25, 50, 50), 2000.0)
    small = _blob(shape, (10, 70, 70), (16, 80, 80), 1500.0)
    image = _image(_noisy(big + small), ("z", "y", "x"), (1.0, 1.0, 1.0))
    on = ac.find_object(image, level=0, seed={"z": 13.0, "y": 75.0,
                                              "x": 75.0})
    assert _holds(on.ranges["x"], _edges(70, 80, 1.0, 0.0), 4.0), \
        on.ranges["x"]
    # 4 voxels beside its edge (within SEED_REACH): still the small one
    beside = ac.find_object(image, level=0, seed={"z": 13.0, "y": 75.0,
                                                  "x": 83.0})
    assert beside.ranges == on.ranges
    # far from both: none
    far = ac.find_object(image, level=0, seed={"z": 13.0, "y": 95.0,
                                               "x": 5.0})
    assert not far.ranges and far.n_found == 0


def _core_body_tissue_data():
    """A bright core inside a dimmer body, inside faint tissue that fills
    most of the volume (0.18.0: the user's wish to separate brighter parts
    of an embryo from fainter ones)."""
    shape = (40, 120, 120)
    signal = (_blob(shape, (2, 4, 4), (38, 116, 116), 300.0)      # tissue
              + _blob(shape, (10, 30, 30), (30, 90, 90), 900.0)   # body
              + _blob(shape, (15, 50, 50), (25, 70, 70), 2500.0))  # core
    return _noisy(signal)


def _core_body_tissue():
    return _image(_core_body_tissue_data(), ("z", "y", "x"), (1.0, 1.0, 1.0))


def test_own_threshold_and_seed_modes_separate_core_and_body():
    """(0.18.0) The user's own threshold: above the body, the core; below
    it but above the tissue, the body (also with a seed; at every
    timepoint one value; no growing). The seed modes: "brightest" the core,
    "whole" the body (the tissue is too large to be the object)."""
    image = _core_body_tissue()
    seed = {"z": 20.0, "y": 60.0, "x": 60.0}
    core, body = _edges(50, 70, 1.0, 0.0), _edges(30, 90, 1.0, 0.0)
    # own threshold, without and with a seed
    for kwargs in ({}, {"seed": seed}):
        high = ac.find_object(image, level=0, threshold=2500.0, **kwargs)
        assert _holds(high.ranges["x"], core, 4.0), (kwargs, high.ranges)
        low = ac.find_object(image, level=0, threshold=800.0, grow=True,
                             **kwargs)       # (growing is left out)
        assert _holds(low.ranges["x"], body, 4.0), (kwargs, low.ranges)
    # seed modes
    bright = ac.find_object(image, level=0, seed=seed, grow=False,
                            seed_mode="brightest")
    assert _holds(bright.ranges["x"], core, 4.0), bright.ranges
    whole = ac.find_object(image, level=0, seed=seed, grow=False,
                           seed_mode="whole")
    assert _holds(whole.ranges["x"], body, 4.0), whole.ranges
    stable = ac.find_object(image, level=0, seed=seed, grow=False)
    assert stable.ranges, "most stable: some object"
    assert "your own threshold" in ac.describe(high, image)


def test_search_volume_gives_the_outline_at_a_threshold():
    """(0.18.0) What the panel's own threshold uses: the volume as the
    search sees it (smoothed), with an automatic value between tissue and
    core, a slider range, and outlines on a slice (or through a slab) in
    µm along the view's axes: round the core above 2000, round the body at
    600; none outside the volume."""
    image = _core_body_tissue()
    volume = ac.search_volume(image)
    assert volume.data.shape == (40, 120, 120) and volume.level == 0
    assert 300 < volume.automatic("otsu") < 2500
    low, high = volume.value_range()
    assert low < 300 and 3500 < high < 4000      # (core: 3800 with the rest)

    def extent(lines, i):
        points = np.concatenate(lines)
        return points[:, i].min(), points[:, i].max()

    # the XY view (y, x) at z = 20: the core's outline lies along its edge
    # (voxels 50-69, the contour half a voxel outside them, smoothed a bit)
    lines = volume.contours(("y", "x"), "z", (20.0, 20.0), 2000.0)
    x0, x1 = extent(lines, 1)
    assert abs(x0 - 49.5) < 1.5 and abs(x1 - 69.5) < 1.5, (x0, x1)
    lines = volume.contours(("y", "x"), "z", (20.0, 20.0), 600.0)
    x0, x1 = extent(lines, 1)
    assert abs(x0 - 29.5) < 1.5 and abs(x1 - 89.5) < 1.5, (x0, x1)
    # an XZ view (z, x) at y = 60: z along the first axis
    lines = volume.contours(("z", "x"), "y", (60.0, 60.0), 2000.0)
    z0, z1 = extent(lines, 0)
    assert abs(z0 - 14.5) < 1.5 and abs(z1 - 24.5) < 1.5, (z0, z1)
    # a slice beside the core, and a slab through it (its maximum)
    assert not volume.contours(("y", "x"), "z", (5.0, 5.0), 2000.0)
    assert volume.contours(("y", "x"), "z", (5.0, 30.0), 2000.0)
    assert not volume.contours(("y", "x"), "z", (100.0, 100.0), 600.0)


def _touching_spheres(shift=(0.0, 0.0, 0.0), tilt=30.0):
    """Two spheres (radius 15 voxels, 1 µm) that overlap: their centres
    26 voxels apart along a line tilted `tilt` degrees from x towards z,
    so the boundary between them (the plane halfway) is tilted to every
    view. Returns (signal, centres, truth: 1 / 2 = the nearer centre)."""
    shape = (60, 90, 110)
    middle = np.array([30.0, 45.0, 55.0]) + np.asarray(shift)
    a = np.radians(tilt)
    half = 13.0 * np.array([np.sin(a), 0.0, np.cos(a)])
    centres = [middle - half, middle + half]
    grid = np.indices(shape, dtype=float)
    distances = [np.sqrt(sum((grid[i] - c[i]) ** 2 for i in range(3)))
                 for c in centres]
    inside = (distances[0] <= 15) | (distances[1] <= 15)
    signal = np.where(inside, 1500.0, 0.0)
    truth = np.where(inside, np.where(distances[0] <= distances[1], 1, 2), 0)
    return signal, centres, truth


def _split_scores(parts, truth):
    """Per part k: (share of object k it holds, share of the other)."""
    scores = []
    for k, mask in enumerate(parts, start=1):
        own, other = truth == k, truth == 3 - k
        scores.append((float((mask & own).sum() / own.sum()),
                       float((mask & other).sum() / other.sum())))
    return scores


def test_split_objects_separates_touching_spheres_at_an_angle():
    """(0.18.0) Two seeds on touching spheres that the threshold fuses (the
    user's case: embryos no threshold separates, touching at an angle):
    each seed's search finds both; split_objects gives each its own, the
    boundary tilted as the contact is (the floods are 3D)."""
    signal, centres, truth = _touching_spheres()
    image = _image(_noisy(signal), ("z", "y", "x"), (1.0, 1.0, 1.0))
    seeds = [dict(zip("zyx", c)) for c in centres]
    results = [ac.find_object(image, level=0, seed=s, keep_mask=True)
               for s in seeds]
    assert ac.mask_holds(image, results[0], seeds[1]), "should be fused"
    parts = ac.split_objects(image, None, results, seeds)
    scores = _split_scores([p.mask for p in parts], truth)
    print("   split scores (own, other):", [tuple(round(v, 3) for v in s)
                                            for s in scores])
    for own, other in scores:
        assert own > 0.95 and other < 0.05, scores
    for part, centre in zip(parts, centres):
        assert part.ranges and part.found[0] is not None
        assert part.ranges["x"][0] < centre[2] < part.ranges["x"][1]
    # each part's box is smaller than the fused object's
    fused = results[0].ranges["x"][1] - results[0].ranges["x"][0]
    assert all(p.ranges["x"][1] - p.ranges["x"][0] < 0.8 * fused
               for p in parts)


def test_gap_line_is_a_wall_where_it_is_drawn():
    """(0.18.0) A gap line drawn in XY on one slice, 5 voxels to one side
    of the boundary: on that slice the parts meet at the line, not at the
    boundary the image gives; far from it (beyond GAP_REACH) the image
    decides again."""
    signal, centres, truth = _touching_spheres(tilt=0.0)   # boundary x=55
    image = _image(_noisy(signal), ("z", "y", "x"), (1.0, 1.0, 1.0))
    seeds = [dict(zip("zyx", c)) for c in centres]
    results = [ac.find_object(image, level=0, seed=s, keep_mask=True)
               for s in seeds]
    line = {"plane": ("y", "x"), "depth": "z", "at": 30.0, "time": None,
            "points": [(20.0, 60.0), (70.0, 60.0)], "width": 2.0}
    parts = ac.split_objects(image, None, results, seeds, [line])
    first = parts[0].mask
    row = first[30, 45]                       # z = 30 (drawn), y = 45
    assert row[:59].sum() > 0 and not row[61:].any() and row[57], \
        np.nonzero(row)[0].max()
    far = first[30 - ac.GAP_REACH - 4, 45]    # beyond the line's reach
    assert abs(int(np.nonzero(far)[0].max()) - 55) <= 2, \
        np.nonzero(far)[0].max()


def test_split_objects_carries_the_parts_through_time():
    """(0.18.0) A time-lapse of the touching spheres drifting in y: seeds
    at the first timepoint; at each timepoint each part holds its own
    sphere."""
    frames, truths = [], []
    for t in range(3):
        signal, centres, truth = _touching_spheres(shift=(0, 6.0 * t, 0))
        frames.append(signal)
        truths.append(truth)
        if t == 0:
            first_centres = centres
    image = _image(_noisy(np.stack(frames)), ("t", "z", "y", "x"),
                   (10.0, 1.0, 1.0, 1.0))
    seeds = [{"t": 0.0, **dict(zip("zyx", c))} for c in first_centres]
    results = [ac.find_object(image, level=0, seed=s, keep_mask=True)
               for s in seeds]
    parts = ac.split_objects(image, None, results, seeds,
                             timed_silhouettes=True)
    for part in parts:
        assert part.n_found == 3 and len(part.timed_silhouettes) == 3
        assert part.ranges["t"] == (0.0, 20.0)
    # (the masks are over all timepoints: check the last frame's boxes)
    for k, part in enumerate(parts, start=1):
        box = part.found[2]
        z, y, x = np.nonzero(truths[2] == k)
        assert box["y"][0] <= y.min() and y.max() <= box["y"][1]
        assert box["x"][0] <= x.min() + 1 and x.max() <= box["x"][1] + 1


def _crop_keeps(image, ranges, outlines):
    """The voxels (level 0) a region with these ranges and outlines keeps,
    as the crop computes them."""
    level0 = image.levels[0]
    shape = level0.array.shape
    names = image.axis_names
    box = np.zeros(shape, bool)
    index = []
    for i, axis in enumerate(names):
        first, end = vc.physical_to_index_range(
            *ranges[axis], level0.scale[i], level0.translation[i])
        index.append(slice(max(first, 0), min(end, shape[i])))
    box[tuple(index)] = True
    keep = vc.outline_mask(level0, names, [0] * len(shape), shape, outlines,
                           ac.spatial_axes(image))
    return box if keep is None else box & np.broadcast_to(keep, shape)


def test_slice_outlines_keep_the_neighbour_out_of_a_split_part():
    """(0.18.0, the user's wish) Two touching spheres split between their
    seeds, contact tilted 30 deg, margin 2 µm: with three silhouettes each
    crop also keeps a corner of the other sphere (and the margins overlap);
    with per-slice outlines whose margin stops halfway to the other part,
    each crop keeps (nearly) all of its own sphere and (nearly) none of the
    other. Polygons of the two parts on one slice do not overlap."""
    signal, centres, truth = _touching_spheres()
    image = _image(_noisy(signal), ("z", "y", "x"), (1.0, 1.0, 1.0))
    seeds = [dict(zip("zyx", c)) for c in centres]
    results = [ac.find_object(image, level=0, seed=s, keep_mask=True,
                              margin=2.0, grow=False) for s in seeds]
    parts = ac.split_objects(image, None, results, seeds, grow=False)
    scores = {}
    for kind in ("silhouettes", "slices"):
        scores[kind] = []
        for k, part in enumerate(parts):
            other = parts[1 - k].mask
            outlines = (ac.silhouettes(image, part, part.ranges)
                        if kind == "silhouettes" else
                        ac.slice_outlines(image, part, part.ranges,
                                          others=other))
            keeps = _crop_keeps(image, part.ranges, outlines)
            own, theirs = truth == k + 1, truth == 2 - k
            scores[kind].append((float((keeps & own).sum() / own.sum()),
                                 float((keeps & theirs).sum()
                                       / theirs.sum())))
    print("   crop keeps (own, other):",
          {kind: [tuple(round(v, 3) for v in s) for s in value]
           for kind, value in scores.items()})
    for own, theirs in scores["slices"]:
        assert own > 0.97 and theirs < 0.03, scores
    assert all(s[1] > 3 * t[1] for s, t in zip(scores["silhouettes"],
                                                scores["slices"])), scores
    # on a shared slice the two parts' polygons meet, they do not overlap
    first = ac.slice_outlines(image, parts[0], parts[0].ranges,
                              others=parts[1].mask)[("y", "x")]
    second = ac.slice_outlines(image, parts[1], parts[1].ranges,
                               others=parts[0].mask)[("y", "x")]
    assert len(first) <= ac.MAX_SLICE_OUTLINES + 2
    shared = sorted(set(first) & set(second))
    assert shared
    for z in shared:
        a = vc.fill_polygon((90, 110), first[z])
        b = vc.fill_polygon((90, 110), second[z])
        assert (a & b).sum() <= 0.02 * min(a.sum(), b.sum()), z


def test_slice_outlines_of_one_tilted_object():
    """(0.18.0) One ellipsoid tilted 30 deg: the per-slice outlines' crop
    holds all of it and less around it than the three silhouettes'."""
    shape = (60, 80, 100)
    grid = np.indices(shape, dtype=float)
    a = np.radians(30.0)
    dz, dy, dx = grid[0] - 30, grid[1] - 40, grid[2] - 50
    along = dx * np.cos(a) + dz * np.sin(a)
    across = -dx * np.sin(a) + dz * np.cos(a)
    body = (along / 35) ** 2 + (dy / 15) ** 2 + (across / 10) ** 2 <= 1
    image = _image(_noisy(np.where(body, 1500.0, 0.0)), ("z", "y", "x"),
                   (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0, keep_mask=True, margin=1.0,
                            grow=False)
    by_slice = _crop_keeps(image, result.ranges,
                           ac.slice_outlines(image, result))
    carved = _crop_keeps(image, result.ranges,
                         ac.silhouettes(image, result))
    print(f"   kept: per slice {by_slice.sum()}, silhouettes "
          f"{carved.sum()}, object {body.sum()} voxels")
    assert (by_slice & body).sum() / body.sum() > 0.99
    assert by_slice.sum() < carved.sum()


def test_seed_search_reports_progress_and_can_be_cancelled():
    """(0.18.0, the user asked for a progress bar) With a seed, the search
    reports progress within a timepoint (per threshold tried, the slow
    part), always rising and ending at the whole; Cancel stops it between
    those steps, not only between timepoints."""
    shape = (30, 100, 100)
    image = _image(_noisy(_blob(shape, (5, 10, 10), (25, 50, 50), 2000.0)),
                   ("z", "y", "x"), (1.0, 1.0, 1.0))
    seed = {"z": 13.0, "y": 30.0, "x": 30.0}
    seen = []
    ac.find_object(image, level=0, seed=seed,
                   progress=lambda done, total: seen.append(done / total))
    assert len(seen) >= ac.SEED_LEVELS, seen        # steps within it
    assert any(0 < share < 1 for share in seen), seen
    assert seen == sorted(seen) and seen[-1] == 1.0, seen
    asked = []

    def cancel():
        asked.append(1)
        return len(asked) > 3            # after a few steps

    try:
        ac.find_object(image, level=0, seed=seed, cancel=cancel)
    except ac.SearchCancelled:
        pass
    else:
        raise AssertionError("not cancelled")
    assert len(asked) == 4


def test_seed_follows_its_object_through_time():
    """0.16.0: a seed at any timepoint gives the region of that object over
    all timepoints (followed from the seed's timepoint forwards and
    backwards), not of a brighter object next to it."""
    shape = (5, 20, 64, 120)                      # t z y x
    signal = np.zeros(shape)
    for t in range(5):                            # A moves 8 voxels along x
        signal[t] = (_blob(shape[1:], (5, 5, 10 + 8 * t),
                           (15, 25, 25 + 8 * t), 1500.0)
                     + _blob(shape[1:], (5, 35, 80), (15, 60, 115), 3000.0))
    image = _image(_noisy(signal), ("t", "z", "y", "x"),
                   (10.0, 1.0, 1.0, 1.0))
    for k in (0, 2, 4):
        seed = {"t": 10.0 * k, "z": 10.0, "y": 15.0, "x": 17.0 + 8 * k}
        result = ac.find_object(image, level=0, seed=seed)
        assert result.n_found == 5, (k, result.notes)
        assert _holds(result.ranges["x"], _edges(10, 57, 1.0, 0.0), 4.0), \
            (k, result.ranges["x"])
        assert _holds(result.ranges["y"], _edges(5, 25, 1.0, 0.0), 4.0), \
            (k, result.ranges["y"])
    # without a seed: the brighter one
    result = ac.find_object(image, level=0)
    assert _holds(result.ranges["x"], _edges(80, 115, 1.0, 0.0), 4.0), \
        result.ranges["x"]


def _near(found, true, tolerance):
    """found's edges within `tolerance` of true's (either way)."""
    return (abs(found[0] - true[0]) <= tolerance
            and abs(found[1] - true[1]) <= tolerance)


def _speckled(shape, lo, hi, value, rng):
    """A blob of bright specks (like nuclei: about half its voxels), so
    that a threshold above its typical brightness breaks it into pieces."""
    blob = _blob(shape, lo, hi, 1.0) > 0
    specks = rng.random(shape) < 0.5
    from scipy import ndimage as ndi
    texture = ndi.gaussian_filter(specks.astype(float), 1.0) * 2.0
    return np.where(blob, value * texture, 0.0)


def test_seed_on_a_dim_object_below_the_automatic_threshold():
    """0.16.0, the visible check's case: the whole volume's automatic
    threshold lies far above a dim, speckled object (it would fall apart
    into specks there), so the seed's threshold is chosen for it (the most
    stable level of its piece): the region holds it, and not the bright
    object beside it. Also mask_holds."""
    rng = np.random.default_rng(11)
    shape = (30, 100, 140)
    bright = _blob(shape, (5, 20, 10), (25, 80, 60), 5000.0)
    dim = _speckled(shape, (6, 25, 80), (24, 75, 130), 500.0, rng)
    image = _image(_noisy(bright + dim), ("z", "y", "x"), (1.0, 1.0, 1.0))
    auto = ac.find_object(image, level=0)
    assert auto.thresholds[0] > 1000                 # far above the dim one
    seed = {"z": 15.0, "y": 50.0, "x": 105.0}
    result = ac.find_object(image, level=0, seed=seed, keep_mask=True)
    assert result.thresholds[0] < auto.thresholds[0]
    # (a speckled edge may be dark: within 5 voxels of the true edges)
    assert _near(result.ranges["x"], _edges(80, 130, 1.0, 0.0), 5.0), \
        result.ranges["x"]
    assert _near(result.ranges["y"], _edges(25, 75, 1.0, 0.0), 5.0), \
        result.ranges["y"]
    assert ac.mask_holds(image, result, seed)
    assert not ac.mask_holds(image, result, {"z": 15.0, "y": 50.0,
                                             "x": 35.0})   # the bright one
    assert not ac.mask_holds(image, auto, seed)            # (no mask kept)


def test_seed_follows_a_dim_object_beside_a_bright_one():
    """0.16.0: at the timepoints followed, the threshold is the seed's,
    relative to the background and the object's brightness there (each
    timepoint's own automatic threshold lies above the dim object)."""
    rng = np.random.default_rng(5)
    shape = (4, 24, 80, 160)                       # t z y x
    signal = np.zeros(shape)
    for t in range(4):                             # the dim one moves
        signal[t] = (_blob(shape[1:], (4, 10, 10), (20, 70, 50), 5000.0)
                     + _speckled(shape[1:], (5, 15, 70 + 10 * t),
                                 (19, 65, 110 + 10 * t), 500.0, rng))
    image = _image(_noisy(signal), ("t", "z", "y", "x"),
                   (10.0, 1.0, 1.0, 1.0))
    seed = {"t": 10.0, "z": 12.0, "y": 40.0, "x": 100.0}
    result = ac.find_object(image, level=0, seed=seed)
    assert result.n_found == 4, result.notes
    assert _near(result.ranges["x"], _edges(70, 140, 1.0, 0.0), 5.0), \
        result.ranges["x"]
    assert result.ranges["x"][0] > 55.0              # not the bright one


def test_same_object_needs_overlap_and_each_others_seed():
    """0.16.0, Fit to seeds: two seeds give one region only if their
    objects overlap by more than half AND each object holds (or comes
    within SEED_REACH of) the other's seed. The visible check's case: a
    dim object's piece that is mostly its bright neighbour overlaps the
    neighbour's object by more than half, but the neighbour's object does
    not hold the dim one's seed: two regions, not "the same"."""
    from volume_cropper.autocrop_panel import _same_object
    image = _image(_noisy(np.zeros((10, 40, 60))), ("z", "y", "x"),
                   (1.0, 1.0, 1.0))
    search = {"z": (0, 10), "y": (0, 40), "x": (0, 60)}

    def found(mask):
        return ac.ObjectBox(ranges={"z": (0.0, 1.0)}, level=0,
                            search=search, mask=mask)

    bright = np.zeros((10, 40, 60), bool)
    bright[2:8, 5:35, 5:30] = True                   # x 5-29
    dim_piece = bright.copy()
    dim_piece[2:8, 15:25, 30:45] = True              # plus a part of x 30-44
    seed_bright = {"z": 5.0, "y": 20.0, "x": 15.0}
    seed_dim = {"z": 5.0, "y": 20.0, "x": 50.0}      # beyond both
    seed_again = {"z": 5.0, "y": 10.0, "x": 25.0}    # bright one, again
    assert ac.mask_holds(image, found(bright), seed_bright)
    assert not ac.mask_holds(image, found(bright), seed_dim)
    assert ac.mask_holds(image, found(dim_piece), seed_dim, reach=6)
    assert not ac.mask_holds(image, found(dim_piece), seed_dim, reach=5)
    # two seeds on the bright one: the same object
    assert _same_object(image, found(bright), found(bright.copy()),
                        seed_bright, seed_again)
    # overlap 4500 / 5400 > 0.5, but the bright object does not hold the
    # dim one's seed: not the same
    assert not _same_object(image, found(bright), found(dim_piece),
                            seed_bright, seed_dim)


def test_silhouettes_keep_the_object_and_cut_the_corners():
    """0.16.0: the object's outline in XY, XZ and YZ (keep_mask, then
    silhouettes) as crop polygons: every voxel of the object is kept (as
    the crop's own mask, vc.outline_mask, decides), the empty corners of
    its box are not, and each polygon reaches through the region."""
    from volume_cropper import core as vc
    shape = (40, 80, 80)
    zz, yy, xx = np.ogrid[:40, :80, :80]
    rod = ((np.abs(yy - xx) <= 6) & (xx >= 10) & (xx < 70) & (yy >= 10)
           & (yy < 70) & (zz >= 10) & (zz < 30))     # a diagonal bar in XY
    image = _image(_noisy(2000.0 * rod), ("z", "y", "x"), (1.0, 1.0, 1.0))
    for margin in (0.0, 3.0):
        result = ac.find_object(image, level=0, margin=margin,
                                keep_mask=True)
        assert result.mask is not None and result.mask.any()
        outlines = ac.silhouettes(image, result)
        assert set(outlines) == {("y", "x"), ("z", "x"), ("y", "z")}
        for plane, keys in outlines.items():
            depth = next(a for a in "zyx" if a not in plane)
            assert list(keys) == [result.ranges[depth]]
        keep = np.broadcast_to(vc.outline_mask(
            image.levels[0], image.axis_names, [0, 0, 0], shape, outlines),
            shape)
        assert keep[rod].all(), (margin, int((rod & ~keep).sum()))
        # a corner of the box, far from the bar (in XY), is not kept
        assert not keep[20, 60, 15] and not keep[20, 15, 60]
        # the margin: voxels up to `margin` beside the bar are kept too
        if margin:
            assert keep[20, 40, 40 + 6 + 3]
    # without a mask (keep_mask off): none
    assert ac.silhouettes(image, ac.find_object(image, level=0)) == {}


def _scale_only_pyramid(path, data, factors):
    """An OME-Zarr 0.5 image with level 0 and one coarser level (block
    means), whose levels record a scale only (no translation), as
    eubi-bridge writes them."""
    import zarr

    from volume_cropper import core as vc
    root = zarr.open_group(path, mode="w-", zarr_format=3)
    blocks = []
    for n, f in zip(data.shape, factors):
        blocks += [n // f, f]
    small = data.reshape(blocks).mean(axis=(1, 3, 5)).astype(data.dtype)
    datasets = []
    for k, (array, scale) in enumerate(((data, (1.0, 1.0, 1.0)),
                                        (small, [float(f) for f in factors]))):
        level = root.create_array(name=str(k), shape=array.shape,
                                  dtype=array.dtype, chunks=array.shape,
                                  dimension_names=["z", "y", "x"])
        level[...] = array
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)}]})
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{
        "name": "scale only",
        "axes": [{"name": a, "type": "space", "unit": "micrometer"}
                 for a in "zyx"],
        "datasets": datasets}]}
    return vc.open_image(path)


def test_silhouettes_on_a_scale_only_pyramid():
    """The review of 2026-11-08: searched on a coarser level of a pyramid
    whose levels record no translation (eubi-bridge's), the silhouettes
    lie where the box lies (they were shifted by (f - 1) / 2 voxels of
    level 0), with the whole margin (it was rounded down to whole voxels
    of the level searched: 3 µm of 8 µm voxels was lost), and keep every
    voxel of the object."""
    import tempfile
    from pathlib import Path

    from volume_cropper import core as vc
    shape = (16, 256, 256)
    obj = np.zeros(shape, bool)
    obj[4:12, 80:176, 96:160] = True
    data = _noisy(2000.0 * obj)
    with tempfile.TemporaryDirectory() as tmp:
        image = _scale_only_pyramid(Path(tmp) / "s.zarr", data, (1, 8, 8))
        assert np.allclose(image.levels[1].translation, 0.0)
        for margin in (0.0, 3.0, 10.0):
            result = ac.find_object(image, level=1, margin=margin,
                                    grow=False, keep_mask=True)
            outlines = ac.silhouettes(image, result)
            points = np.array(next(iter(outlines[("y", "x")].values())))
            for k, axis in enumerate("yx"):
                low, high = result.ranges[axis]       # the box + margin
                # the outline reaches the box's edges, and at most half a
                # voxel of the level searched (8 µm) beyond them
                assert low - 4.0 - 1e-6 <= points[:, k].min() <= low + 1e-6, \
                    (margin, axis, points[:, k].min(), low)
                assert high - 1e-6 <= points[:, k].max() <= high + 4.0 + 1e-6, \
                    (margin, axis, points[:, k].max(), high)
            keep = np.broadcast_to(vc.outline_mask(
                image.levels[0], image.axis_names, [0, 0, 0], shape,
                outlines), shape)
            assert keep[obj].all(), (margin, int((obj & ~keep).sum()))


def test_object_at_the_image_edge():
    """The review of 2026-11-08: an object that reaches the image's own
    edge, with the whole image searched, is said to reach the image's edge
    (not told to search a larger region, which does not exist)."""
    shape = (20, 60, 80)
    data = _noisy(_blob(shape, (4, 15, 0), (16, 45, 30), 2000.0))
    image = _image(data, ("z", "y", "x"), (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert "x" in result.image_edges and not result.touches
    text = " ".join(result.notes)
    assert "image's own edge along x" in text
    assert "Search a larger region" not in text
    # inside a region that cuts it, the region's edge (as before)
    cut = ac.find_object(image, level=0, region={"x": (-0.5, 20.0)})
    assert "x" in cut.touches


def test_channels_margin_and_image_edges():
    shape = (20, 60, 60)
    blob = _blob(shape, (2, 0, 20), (12, 20, 45))            # at y = 0
    dim = _noisy(np.zeros(shape))
    image = _image([dim, _noisy(blob)], ("z", "y", "x"), (1.0, 0.5, 0.5),
                   names=["dapi", "gfp"])
    nothing = ac.find_object(image, channel=0, level=0)
    assert nothing.n_found == 0, nothing.contrasts   # noise: no object
    assert nothing.contrasts[0] < ac.MIN_CONTRAST
    result = ac.find_object(image, channel=1, level=0, margin=3.0)
    low, high = image.extent("y")
    assert result.ranges["y"][0] == low        # kept within the image
    assert result.contrasts[0] > 10
    x_true = _edges(20, 45, 0.5, 0.0)
    assert result.ranges["x"][0] <= x_true[0] - 3.0 + 1e-6
    assert result.ranges["x"][1] >= x_true[1] + 3.0 - 1e-6


def _share_inside(signal, ranges, scale):
    """Share of the object's voxels (signal > 0) whose centres lie in the
    ranges (z, y, x; translation 0)."""
    inside = np.ones(int((signal > 0).sum()), bool)
    for k, (axis, index) in enumerate(zip("zyx", np.nonzero(signal > 0))):
        pos = index * scale[k]
        inside &= (pos >= ranges[axis][0]) & (pos <= ranges[axis][1])
    return inside.mean()


def test_dim_parts_belong_to_the_object():
    """The review's cases (2026-11-07): light that fades with depth (the far
    side of an embryo dim), and two bright parts joined by dimmer tissue.
    The automatic threshold cuts both; growing down to the background keeps
    the whole object in the box."""
    shape, scale = (64, 96, 96), (2.0, 1.0, 1.0)
    blob = _blob(shape, (4, 20, 20), (56, 76, 76), 1.0)
    z = np.arange(shape[0])[:, None, None]
    for length in (20, 12):               # far side 2.6x, 0.5x background
        fading = blob * 3000.0 * np.exp(-(z - 4) / length)
        image = _image(_noisy(fading), ("z", "y", "x"), scale)
        result = ac.find_object(image, level=0)
        assert _share_inside(fading, result.ranges, scale) == 1.0, (
            length, result.ranges["z"])
    lobes = (_blob((20, 40, 200), (4, 10, 10), (16, 30, 90), 2000.0)
             + _blob((20, 40, 200), (4, 10, 110), (16, 30, 190), 1800.0))
    bridge = np.zeros_like(lobes)
    bridge[6:14, 14:26, 85:115] = 300.0
    joined = np.where(lobes > 0, lobes, bridge)
    image = _image(_noisy(joined), ("z", "y", "x"), (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert _share_inside(joined, result.ranges, (1.0, 1.0, 1.0)) == 1.0, \
        result.ranges["x"]
    # without growing (for objects in bright tissue) only one lobe: the
    # reason growing is on at first
    result = ac.find_object(image, level=0, grow=False)
    assert _share_inside(joined, result.ranges, (1.0, 1.0, 1.0)) < 0.7


def test_zero_padding_is_not_the_object():
    """Fused volumes are padded with exact zeros: the recorded area is not
    the object (without an embryo: nothing; with one: the embryo)."""
    data = np.zeros((20, 80, 80), np.uint16)
    data[:, 10:70, 10:70] = RNG.poisson(100, size=(20, 60, 60))
    image = _image(data, ("z", "y", "x"), (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert result.n_found == 0, (result.contrasts, result.ranges)
    embryo = _blob(data.shape, (4, 30, 30), (16, 50, 50), 2000.0)
    data = np.where(data > 0, data + embryo, 0).astype(np.uint16)
    image = _image(data, ("z", "y", "x"), (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert _holds(result.ranges["x"], _edges(30, 50, 1.0, 0.0), 4.0), \
        result.ranges["x"]


def test_flat_image_and_2d_image():
    flat = np.full((10, 40, 40), 500, np.uint16)
    image = _image(flat, ("z", "y", "x"), (1.0, 1.0, 1.0))
    result = ac.find_object(image, level=0)
    assert result.n_found == 0 and result.ranges == {}
    assert "No object" in ac.describe(result, image)
    plane = _noisy(_blob((80, 90), (20, 30), (50, 70)))
    image = _image(plane, ("y", "x"), (0.5, 0.5))
    result = ac.find_object(image, level=0)
    assert _holds(result.ranges["x"], _edges(30, 70, 0.5, 0.0), 2.0), \
        result.ranges["x"]


def test_coarser_level_and_methods():
    shape = (32, 600, 600)
    data = _noisy(_blob(shape, (8, 200, 150), (24, 400, 450)))
    image = _image(data, ("z", "y", "x"), (2.0, 0.5, 0.5))
    assert len(image.levels) >= 2
    level = ac.choose_level(image, max_voxels=4_000_000)
    assert level >= 1
    result = ac.find_object(image, level=level)
    scale_x = image.levels[level].scale[-1]
    assert _holds(result.ranges["x"], _edges(150, 450, 0.5, 0.0),
                  3 * scale_x + 0.5), result.ranges["x"]
    for method in ("triangle", "li"):
        result = ac.find_object(image, level=level, method=method)
        assert result.n_found == 1, method
        assert result.ranges["x"][0] <= 150 * 0.5


def test_panel_fits_a_new_region_or_shrinks_the_selected_one():
    """The panel's experimental group, in a hidden window: off at first;
    no region selected -> a new region "object01" around the object; a
    region selected -> shrunk to the object; Undo fit -> as before."""
    import time

    from napari.layers import Image
    from qtpy.QtWidgets import QApplication

    from volume_cropper import crop_gui

    def pump(seconds=0.3):
        end = time.time() + seconds
        while time.time() < end:
            QApplication.instance().processEvents()
            time.sleep(0.01)

    shape, lo, hi = (30, 120, 100), (8, 30, 20), (22, 71, 51)
    data = _noisy(_blob(shape, lo, hi))
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(data, name="embryo",
                                        scale=(2.0, 0.5, 0.5),
                                        units=("um", "um", "um"))])
        pump()
        group = panel.object_group
        assert not group.isChecked()                  # off at first
        group.setChecked(True)
        group.margin_spin.setValue(2.0)
        panel._refresh(select=None)
        group.fit()
        assert [box.name for box in panel.boxes] == ["object01"]
        box = panel.boxes[0]
        for k, axis in enumerate("zyx"):
            scale = (2.0, 0.5, 0.5)[k]
            true = _edges(lo[k], hi[k], scale, 0.0)
            assert _holds(box.ranges[axis], (true[0] - 2.0, true[1] + 2.0),
                          4 * scale), (axis, box.ranges[axis])
        assert "New region object01" in group.info.text()
        # a rough region, selected: shrunk; Undo fit: back
        rough = {"z": (0.0, 59.0), "y": (5.0, 55.0), "x": (2.0, 40.0)}
        i = panel._new_box(rough)
        panel._refresh(select=i)
        group.fit()
        fitted = panel.boxes[i].ranges
        assert all(fitted[a][0] >= rough[a][0] and fitted[a][1] <= rough[a][1]
                   for a in "zyx")
        assert fitted["z"][1] - fitted["z"][0] < 59.0 - 0.0
        assert group.undo_button.isEnabled()
        # the user's bug (2026-11-08): each Fit made the region larger;
        # fitting again must never grow it
        for _ in range(4):
            before = dict(panel.boxes[i].ranges)
            group.fit()
            after = panel.boxes[i].ranges
            assert all(after[a][0] >= before[a][0] - 1e-9
                       and after[a][1] <= before[a][1] + 1e-9
                       for a in "zyx"), (before, after)
        group.undo()
        group.undo()                      # (only one step back is kept)
        panel.boxes[i].ranges = dict(rough)
        group.fit()
        group.undo()
        assert all(np.allclose(panel.boxes[i].ranges[a], rough[a])
                   for a in "zyx")
        assert not group.undo_button.isEnabled()
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        pump()


def _pump(seconds=0.3):
    import time

    from qtpy.QtWidgets import QApplication
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


def test_panel_searches_the_whole_selected_region():
    """The review's case (2026-11-07): of an OME-Zarr image (read piece by
    piece) only x 0-30 µm of 100 are loaded for the views, and the selected
    region covers the whole image; the fit must search the whole region
    (the object reaches beyond the loaded part), not cut the region to the
    loaded part."""
    import tempfile
    from pathlib import Path

    import zarr

    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    shape = (20, 60, 100)
    data = _noisy(_blob(shape, (4, 15, 10), (16, 45, 80), 2000.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "embryo.zarr"
        root = zarr.open_group(path, mode="w-", zarr_format=3)
        array = root.create_array(name="0", shape=shape, dtype=data.dtype,
                                  chunks=(10, 30, 50),
                                  dimension_names=["z", "y", "x"])
        array[...] = data
        root.attrs["ome"] = {"version": "0.5", "multiscales": [{
            "name": "embryo",
            "axes": [{"name": a, "type": "space", "unit": "micrometer"}
                     for a in "zyx"],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]}
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(path, choice=LoadChoice(
                ranges={"z": (-0.5, 19.5), "y": (-0.5, 59.5),
                        "x": (-0.5, 29.5)}))
            _pump()
            assert panel.ortho.extent("x")[1] < 31.0     # the loaded part
            group = panel.object_group
            group.setChecked(True)
            group.margin_spin.setValue(0.0)
            i = panel._new_box({"x": panel.image.extent("x")})
            panel._refresh(select=i)
            group.fit()
            fitted = panel.boxes[i].ranges["x"]
            assert _holds(fitted, _edges(10, 80, 1.0, 0.0), 4.0), fitted
        finally:
            viewer.close()
            _pump()


def test_fit_all_regions_and_the_edge_note():
    """0.16.0: Fit all regions fits each rough region to the object inside
    it, one Undo for all; a region that cuts its object says so."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    shape = (20, 60, 120)
    left = _blob(shape, (4, 15, 10), (16, 45, 40), 2000.0)
    right = _blob(shape, (4, 10, 70), (16, 50, 110), 2500.0)
    data = _noisy(left + right)
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(data, name="two",
                                        units=("um",) * 3)])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        group.margin_spin.setValue(0.0)
        a = panel._new_box({"x": (0.0, 55.0)})
        b = panel._new_box({"x": (58.0, 119.0)})
        rough = [dict(panel.boxes[k].ranges) for k in (a, b)]
        panel._refresh(select=None)
        group.fit_all()
        x_a, x_b = panel.boxes[a].ranges["x"], panel.boxes[b].ranges["x"]
        assert _holds(x_a, _edges(10, 40, 1.0, 0.0), 4.0), x_a
        assert _holds(x_b, _edges(70, 110, 1.0, 0.0), 4.0), x_b
        assert "fitted" in group.info.text()
        group.undo()                                 # both back
        assert [panel.boxes[k].ranges for k in (a, b)] == rough
        # a region that cuts the object: said so
        cut = panel._new_box({"x": (0.0, 25.0)})
        panel._refresh(select=cut)
        group.fit()
        assert "edge of the area searched along x" in group.info.text()
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_panel_seeds_give_one_region_per_object():
    """0.16.0, in a hidden window: Add seeds; clicks (the views' own mouse
    handler: a click adds a seed, a drag does not; a click on a seed
    removes it) on two objects, two of them on the same object; Fit to
    seeds -> two regions object01 and object02 around them, with outlines
    (Outline ticked); Undo fit removes them. A drawing tool ends Add
    seeds; a new image clears the seeds."""
    from types import SimpleNamespace

    from napari.layers import Image

    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    shape = (20, 60, 120)
    left = _blob(shape, (4, 15, 10), (16, 45, 40), 1500.0)
    right = _blob(shape, (4, 10, 70), (16, 50, 110), 2500.0)
    data = _noisy(left + right)
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(data, name="two",
                                        units=("um",) * 3)])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        group.margin_spin.setValue(0.0)
        group.outline_check.setChecked(True)
        # (0.18.0: with "Split an object between its seeds" on, the default,
        # two seeds on one object split it; this checks the old behaviour)
        group.split_check.setChecked(False)
        assert not group.fit_seeds_button.isEnabled()      # no seeds yet
        group.seed_button.setChecked(True)
        xy = panel.ortho.views["XY"]
        ov.camera(xy.model).zoom = 5.0                     # 5 px per µm
        panel.ortho.seed_delay_ms = 0       # (the delay: tested at the end)

        def press(y, x, moves=()):
            """The XY view's mouse handler: press at (y, x), move the mouse
            by `moves` (screen pixels), release."""
            event = SimpleNamespace(button=1, modifiers=(), pos=(200, 200),
                                    position=(10.0, y, x), type="mouse_press")
            handler = panel.ortho._seed_click(xy)(xy.model, event)
            if handler is None:
                return
            next(handler)
            for move in moves:
                event.type, event.pos = "mouse_move", (200 + move, 200)
                next(handler)
            event.type = "mouse_release"
            next(handler, None)

        press(30.0, 25.0)                  # left object
        press(30.0, 90.0)                  # right object
        press(30.0, 60.0, moves=(2, 12))   # a drag (pan): no seed
        press(22.0, 32.0)                  # left object again
        assert len(group.seeds) == 3, group.seeds
        assert len(xy.seeds.data) == 3 and xy.seeds.visible
        assert len(panel.ortho.views["XZ"].seeds.data) == 3
        press(22.0, 32.0)                  # on a seed: removed
        assert len(group.seeds) == 2
        press(22.0, 32.0)
        assert group.fit_seeds_button.isEnabled()
        group.fit_to_seeds()
        names = [box.name for box in panel.boxes]
        assert names == ["object01", "object02"], (names, group.info.text())
        assert "the same object as object01" in group.info.text()
        x1, x2 = (panel.boxes[k].ranges["x"] for k in (0, 1))
        assert _holds(x1, _edges(10, 40, 1.0, 0.0), 4.0), x1
        assert _holds(x2, _edges(70, 110, 1.0, 0.0), 4.0), x2
        assert all(len(box.outlines) == 3 for box in panel.boxes)
        # Undo fit removes the new regions, but leaves one the user has
        # changed since the fit (review 2026-11-08)
        panel.boxes[0].name = "embryoA"
        group.undo()
        assert [box.name for box in panel.boxes] == ["embryoA"], \
            group.info.text()
        assert "Left as they are" in group.info.text()
        panel.boxes.clear()
        panel._refresh(select=None)
        # on a projection, a click puts no seed (its depth is hidden)
        n = len(group.seeds)
        panel.ortho.projection = "max"
        press(10.0, 10.0)
        assert len(group.seeds) == n and "need a slice" in group.info.text()
        panel.ortho.projection = "none"
        # a click counts once the double-click time has passed; the first
        # click of a double-click (napari zooms) puts no seed
        panel.ortho.seed_delay_ms = 150
        press(10.0, 10.0)
        assert len(group.seeds) == n                       # not yet
        _pump(0.4)
        assert len(group.seeds) == n + 1
        press(50.0, 60.0)
        panel.ortho._cancel_seed()                         # a double-click
        _pump(0.4)
        assert len(group.seeds) == n + 1
        panel.ortho.seed_delay_ms = 0
        # closing the part ends Add seeds (its button would be hidden)
        panel.sections["find"].set_open(True)     # (closed at first)
        assert group.seed_button.isChecked()
        panel.sections["find"].header.click()
        assert not group.seed_button.isChecked()
        assert panel.ortho.on_seed_click is None
        panel.sections["find"].header.click()
        group.seed_button.setChecked(True)
        # a drawing tool ends Add seeds; a new image clears the seeds
        panel.ortho.set_tool("add_polygon")
        assert not group.seed_button.isChecked()
        assert panel.ortho.on_seed_click is None
        assert panel.open_layers([Image(data, name="again",
                                        units=("um",) * 3)])
        _pump()
        assert group.seeds == [] and not len(
            panel.ortho.views["XY"].seeds.data)
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_panel_outline_replaces_polygons_and_undo_gives_them_back():
    """0.16.0: Fit with Outline ticked gives the selected region the
    object's silhouettes in place of its drawn polygon; Undo fit gives the
    ranges and the drawn polygon back."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from volume_cropper import core as vc
    shape = (20, 60, 60)
    data = _noisy(_blob(shape, (4, 15, 15), (16, 45, 45), 2000.0))
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(data, name="one",
                                        units=("um",) * 3)])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        group.outline_check.setChecked(True)
        i = panel._new_box({"y": (2.0, 58.0), "x": (2.0, 58.0)})
        drawn = vc.normalize_outlines(
            {("y", "x"): [(5.0, 5.0), (5.0, 55.0), (55.0, 55.0),
                          (55.0, 5.0)]})
        panel.boxes[i].outlines = drawn
        before = dict(panel.boxes[i].ranges)
        panel._refresh(select=i)
        group.fit()
        box = panel.boxes[i]
        assert set(box.outlines) == {("y", "x"), ("z", "x"), ("y", "z")}
        assert "with its outline" in group.info.text()
        group.undo()
        assert box.outlines == drawn and box.ranges == before
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_channel_list_greyed_out_while_unticked():
    from napari.layers import Image

    from volume_cropper import crop_gui
    layers = [Image(_noisy(np.zeros((4, 20, 20))), name=n,
                    units=("um",) * 3) for n in ("gfp", "dapi")]
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel._open_reader(crop_gui.fi.LayerReader(layers),
                                  crop_gui.fi.FileChoice(), ask=False,
                                  keep_boxes=False)
        _pump()
        group = panel.object_group
        assert not group.isChecked() and not group.channel_combo.isEnabled()
        group.setChecked(True)
        assert group.channel_combo.isEnabled()          # two channels
        group.setChecked(False)
        assert not group.channel_combo.isEnabled()
    finally:
        viewer.close()
        _pump()


def test_panel_own_threshold_outline_histogram_and_fit():
    """(0.18.0) In a hidden window: Own threshold starts at the automatic
    value, switches Dim parts too off (and greys it and the seed menu),
    outlines what lies above it in the 2D views (dotted orange) and marks
    it in the Levels histogram; Fit then finds the core above the body's
    brightness and the body below it; dragging the histogram's line sets
    the value; switching it off hides both and gives Dim parts too back."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    data = _core_body_tissue_data()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1      # ("What to load")
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(data, name="embryo",
                                        units=("um",) * 3)])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        assert group.grow_check.isChecked()
        assert not group.own_spin.isEnabled()            # off at first
        group.own_check.setChecked(True)
        assert group.own_spin.isEnabled() and group.own_slider.isEnabled()
        assert not group.grow_check.isChecked()
        assert not group.grow_check.isEnabled()
        assert not group.seed_mode_combo.isEnabled()
        automatic = ac.search_volume(panel.image).automatic("otsu")
        assert abs(group.own_threshold() - automatic) <= 1.0, \
            (group.own_threshold(), automatic)
        # the outline: in the XY view at the cursor's slice (z ~ 20)
        group.own_spin.setValue(2000.0)
        _pump()
        layer = panel.ortho.views["XY"].level_lines
        assert layer.visible and len(layer.data), "no outline"
        ends = layer.data[:, 0]                   # (y, x) of each dot
        assert 47 < ends[:, 1].min() < 52 and 67 < ends[:, 1].max() < 72, \
            (ends[:, 1].min(), ends[:, 1].max())
        assert np.allclose(layer.edge_color[0],
                           ov.rgba(ov.THRESHOLD_COLOR, 1.0))
        # the histogram's line (when Levels shows the channel searched)
        assert panel.levels.marker is not None
        assert panel.levels.marker[:2] == (0, 2000.0)
        # Fit: above the body, the core; below it, the body
        group.margin_spin.setValue(0.0)
        panel._refresh(select=None)
        group.fit()
        core = panel.boxes[-1].ranges["x"]
        assert _holds(core, (50.0, 70.0), 4.0), core
        assert "your own threshold, 2000" in group.info.text()
        panel.levels.on_marker_moved(600.0)       # (the line dragged)
        assert abs(group.own_threshold() - 600.0) < 1.0
        panel._refresh(select=-1)                 # (none: a new region)
        group.fit()
        body = panel.boxes[-1].ranges["x"]
        assert _holds(body, (30.0, 90.0), 4.0), body
        # off: no outline, no line, Dim parts too as before
        group.own_check.setChecked(False)
        _pump()
        assert not layer.visible and panel.levels.marker is None
        assert group.grow_check.isChecked() and group.grow_check.isEnabled()
        assert group.seed_mode_combo.isEnabled()
        assert group.own_threshold() is None
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_panel_own_threshold_outline_follows_the_timepoint():
    """(0.18.0) In a time-lapse, the outline is of the timepoint shown:
    another timepoint is read once one stays there 0.3 s (none drawn
    before, so that playing through time is not slowed down), then the
    outline is where the object is then; the value is kept."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    shape = (3, 20, 60, 60)
    signal = np.zeros(shape)
    for t in range(3):                     # a block moving 15 voxels in x
        signal[t, 5:15, 20:40, 10 + 15 * t:30 + 15 * t] = 2000.0
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            _noisy(signal), name="blob", axis_labels=("t", "z", "y", "x"),
            units=("minute", "um", "um", "um"))])
        _pump()
        ortho, group = panel.ortho, panel.object_group
        ortho._set_cursor({"t": 0.0, "z": 10.0})
        group.setChecked(True)
        group.own_check.setChecked(True)
        group._set_own_value(1000.0)
        _pump(0.2)
        layer = ortho.views["XY"].level_lines

        def x_span():
            x = layer.data[:, 0, 1]
            return x.min(), x.max()

        x0 = x_span()
        assert abs(x0[0] - 9.5) < 1.5 and abs(x0[1] - 29.5) < 1.5, x0
        ortho._set_cursor({"t": 2.0})
        assert not layer.visible                 # not read yet
        _pump(0.8)
        assert layer.visible
        x2 = x_span()
        assert abs(x2[0] - 39.5) < 1.5 and abs(x2[1] - 59.0) < 1.5, x2
        assert abs(group.own_threshold() - 1000.0) < 1.0   # kept
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_panel_fit_to_seeds_splits_touching_objects_and_gap_lines():
    """(0.18.0) In a hidden window: two seeds on touching spheres (one
    object for the threshold) give two regions, each round its sphere
    (Split on, the default); a gap line drawn (as the views report it)
    shows in that view and a click on it removes it; with Split off they
    give one region, as before."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    signal, centres, _ = _touching_spheres()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(_noisy(signal), name="spheres",
                                        units=("um",) * 3)])
        _pump()
        group, ortho = panel.object_group, panel.ortho
        group.setChecked(True)
        group.grow_check.setChecked(False)
        group.margin_spin.setValue(0.0)
        assert group.split_check.isChecked()
        # a gap line, as the XY view reports a drag (on the slice shown)
        ortho._set_cursor({"z": 30.0})
        group.gap_button.setChecked(True)
        assert ortho.gap_mode and not group.seed_button.isChecked()
        line = {"view": "XY", "plane": ("y", "x"), "depth": "z",
                "at": 30.0, "time": None,
                "points": [(20.0, 55.0), (70.0, 55.0)], "width": 2.0}
        ortho.on_gap_line(dict(line))
        assert len(group.gap_lines) == 1
        assert ortho.views["XY"].gap_line.visible
        assert "(1)" in group.clear_gaps_button.text()
        ortho.on_gap_click({"z": 30.0, "y": 40.0, "x": 55.3}, "XY")
        assert not group.gap_lines                     # removed
        ortho.on_gap_line(dict(line))
        group.gap_button.setChecked(False)
        assert not ortho.gap_mode
        # two seeds, Fit to seeds: two regions
        group.seeds = [dict(zip("zyx", c)) for c in centres]
        group.seed_results = [None, None]
        ortho.set_seeds(group.seeds)
        # (Outline on, silhouettes chosen: the parts get outlines per slice)
        group.outline_check.setChecked(True)
        assert group.outline_kind_combo.currentData() == "silhouettes"
        group.fit_to_seeds()
        assert len(panel.boxes) == 2, group.info.text()
        assert "split between them" in group.info.text()
        assert group.info.text().count("with its outline per slice") == 2
        for box, centre in zip(panel.boxes, centres):
            low, high = box.ranges["x"]
            assert low < centre[2] < high and high - low < 40, box.ranges
            keys = list(box.outlines[("y", "x")])
            assert len(box.outlines) == 1 and len(keys) > 10
            assert all(isinstance(key, float) for key in keys)
        group.outline_check.setChecked(False)
        # Split off: one region for both
        group.undo()
        assert not panel.boxes
        group.split_check.setChecked(False)
        group.seed_results = [None, None]
        group.fit_to_seeds()
        assert len(panel.boxes) == 1, group.info.text()
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def _mesh_volume(vertices, faces) -> float:
    """The volume a closed triangle mesh encloses (divergence theorem)."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6)


def test_outlines_in_3d_as_surfaces():
    """(0.18.0, the user's wish) The shape a region's polygons make, in the
    3D view as a see-through surface in its colour: all regions' (the
    default), the selected region's, or none; the surface encloses what the
    crop keeps (its volume within 15 % of the voxels kept)."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    signal, centres, _ = _touching_spheres()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(_noisy(signal), name="spheres",
                                        units=("um",) * 3)])
        _pump()
        group, ortho = panel.object_group, panel.ortho
        group.setChecked(True)
        group.grow_check.setChecked(False)
        group.margin_spin.setValue(0.0)
        group.outline_check.setChecked(True)
        group.seeds = [dict(zip("zyx", c)) for c in centres]
        group.seed_results = [None, None]
        group.fit_to_seeds()
        assert len(panel.boxes) == 2
        assert panel.outlines_3d_combo.currentData() == "all"
        panel.show_3d_check.setChecked(True)
        _pump(1.0)
        view = ortho.views["3D"]
        assert len(view.surface_layers) == 2            # all, at first
        panel.outlines_3d_combo.setCurrentIndex(
            panel.outlines_3d_combo.findData("selected"))
        panel._on_outlines_3d()
        panel._refresh(select=0)
        _pump()
        assert len(view.surface_layers) == 1
        layer = view.surface_layers[0]
        assert layer.visible and abs(layer.opacity - ov.OUTLINE_3D_OPACITY) \
            < 1e-6
        assert np.allclose(layer.vertex_colors[0],
                           ov.rgba(ov.BOX_COLORS[0], 1.0))
        vertices, faces = ortho.outline_mesh(0)
        kept = _crop_keeps(panel.image, panel.boxes[0].ranges,
                           panel.boxes[0].outlines)
        volume = _mesh_volume(vertices, faces)
        print(f"   surface encloses {volume:.0f} µm³, the crop keeps "
              f"{kept.sum()} voxels of 1 µm³")
        assert abs(volume - kept.sum()) < 0.15 * kept.sum()
        panel.outlines_3d_combo.setCurrentIndex(
            panel.outlines_3d_combo.findData("all"))
        panel._on_outlines_3d()
        assert len(view.surface_layers) == 2
        panel.outlines_3d_combo.setCurrentIndex(
            panel.outlines_3d_combo.findData("off"))
        panel._on_outlines_3d()
        assert not view.surface_layers
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_threshold_outline_has_no_holes():
    """(0.18.0, the user's report) The live outline of one's own threshold
    traces only outer edges: a dim patch inside an object is no hole."""
    shape = (20, 60, 60)
    grid = np.indices(shape, dtype=float)
    r = np.sqrt((grid[1] - 30) ** 2 + (grid[2] - 30) ** 2)
    signal = np.where(r <= 20, 1500.0, 0.0)
    signal[(r <= 5) & (grid[0] >= 4) & (grid[0] < 16)] = 0.0    # dim core
    volume = ac.search_volume(_image(_noisy(signal), ("z", "y", "x"),
                                     (1.0, 1.0, 1.0)))
    lines = volume.contours(("y", "x"), "z", (10.0, 10.0), 800.0)
    assert len(lines) == 1, [len(line) for line in lines]
    y = np.concatenate(lines)[:, 0]
    assert y.min() < 12 and y.max() > 48                  # the outer edge


def test_overlapping_objects_count_as_one():
    """(0.18.0, the user's wish) Seeds whose objects merely overlap (neither
    holds the other's seed) are split between them too."""
    from volume_cropper import autocrop_panel as ap
    image = _image(_noisy(np.zeros((10, 40, 60))), ("z", "y", "x"),
                   (1.0, 1.0, 1.0))
    masks = [np.zeros((10, 40, 60), bool) for _ in range(2)]
    masks[0][2:8, 10:30, 5:35] = True
    masks[1][2:8, 10:30, 25:55] = True          # 10 voxels wide overlap
    results = {}
    for n, mask in enumerate(masks, start=1):
        result = ac.ObjectBox(level=0, times=[None])
        result.search = {"z": (0, 10), "y": (0, 40), "x": (0, 60)}
        result.mask = mask
        result.ranges = {"z": (2, 8), "y": (10, 30), "x": (5, 55)}
        results[n] = result
    seeds = [{"z": 5.0, "y": 20.0, "x": 8.0}, {"z": 5.0, "y": 20.0,
                                              "x": 52.0}]
    assert not ac.mask_holds(image, results[1], seeds[1])
    assert ap._overlapping(results[1], results[2])
    assert ap.ObjectFinder._fused_groups(None, image, results, seeds) \
        == [[1, 2]]
    masks[1][:] = False
    masks[1][2:8, 10:30, 40:55] = True           # apart now
    assert ap.ObjectFinder._fused_groups(None, image, results, seeds) == []


def test_panel_recompute_split_with_new_gap_lines():
    """(0.18.0, the user's wish) After Fit to seeds, gap lines drawn across
    the contact and Recompute split: the parts are split again (without
    searching again), the same regions get their new ranges in place;
    Undo fit gives them back; a gap line elsewhere changes nothing."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    signal, centres, _ = _touching_spheres(tilt=0.0)    # contact at x = 55
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(_noisy(signal), name="spheres",
                                        units=("um",) * 3)])
        _pump()
        group, ortho = panel.object_group, panel.ortho
        group.setChecked(True)
        group.grow_check.setChecked(False)
        group.margin_spin.setValue(0.0)
        assert not group.recompute_button.isEnabled()
        group.seeds = [dict(zip("zyx", c)) for c in centres]
        group.seed_results = [None, None]
        group.fit_to_seeds()
        first, second = panel.boxes
        before = dict(first.ranges)
        assert abs(before["x"][1] - 56) < 3, before
        assert group.recompute_button.isEnabled()
        # a gap line far from the objects: nothing to recompute
        far = {"view": "XY", "plane": ("y", "x"), "depth": "z", "at": 30.0,
               "time": None, "points": [(2.0, 2.0), (5.0, 5.0)],
               "width": 2.0}
        group._gap_drawn(dict(far))
        group.recompute_split()
        assert "No gap line crosses" in group.info.text()
        group.clear_gap_lines()
        # lines 6 voxels into the second sphere, on several slices
        for z in (22.0, 30.0, 38.0):
            group._gap_drawn({"view": "XY", "plane": ("y", "x"),
                              "depth": "z", "at": z, "time": None,
                              "points": [(20.0, 61.0), (70.0, 61.0)],
                              "width": 2.0})
        group.recompute_split()
        assert "recomputed" in group.info.text(), group.info.text()
        assert panel.boxes == [first, second]           # the same regions
        assert first.ranges["x"][1] > before["x"][1] + 3, first.ranges
        group.undo()
        assert first.ranges["x"] == before["x"]
        # changed seeds: Fit to seeds first
        group.seeds.append({"z": 30.0, "y": 45.0, "x": 20.0})
        group.recompute_split()
        assert "seeds changed" in group.info.text()
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_cancel_stops_the_search():
    image = _image(_noisy(np.zeros((3, 4, 20, 20))), ("t", "z", "y", "x"),
                   (1.0, 1.0, 1.0, 1.0))
    try:
        ac.find_object(image, level=0, cancel=lambda: True)
    except ac.SearchCancelled:
        pass
    else:
        raise AssertionError("not cancelled")


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["autocrop"]))
