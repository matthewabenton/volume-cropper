"""Tests for regions that move over time (0.17.0).

Run with ``python test_moving.py`` (or through run_all_tests.py).

A moving region (CropBox.track) keeps one size and follows an object: each
timepoint of the crop is read at the region's place then. The tests check,
on a synthetic time-lapse (random voxels, so any misplaced read shows):
- the shift helpers: interpolation between timepoints, edits at one
  timepoint, "Fix in place";
- that every timepoint of the crop, on every kept level, equals the source
  at the shifted place, worked out here with plain numpy slicing
  (independently of the tool's writer and of its check), with zeros where
  the region reaches outside the image;
- that shifts are rounded to whole voxels of the coarsest kept level;
- the position metadata, the record, the positions file;
- that verify_crop passes, and fails for a crop written at a wrong place;
- every n-th timepoint, the regions file, Save as TIFF, and the refusals.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import tempfile
from pathlib import Path

import numpy as np
import zarr

from volume_cropper import core as vc
from test_volume_cropper import make_test_image, read_tiff

# t, c, z, y, x: 5 timepoints 10 min apart, 2 channels, 16 z slices of
# 2 µm, 64 x 64 pixels of 0.5 µm; level 1 halves y and x.
SHAPE = (5, 2, 16, 64, 64)
SCALE = (10.0, 1.0, 2.0, 0.5, 0.5)
FACTORS = [(1, 1, 1, 1, 1), (1, 1, 1, 2, 2)]

# The region at shift 0: z voxels 2-9, y and x voxels 20-51 (the ranges
# are the voxels' outer edges). Cropped with 2 levels, y and x are widened
# by half a grid step (1 voxel) on each side, then snapped to level 1's
# grid: voxels 18-53 (Y0, NY).
RANGES = {"z": (3.0, 19.0), "y": (9.75, 25.75), "x": (9.75, 25.75)}
Y0, NY = 18, 36
# Shifts in µm per time (minutes). x at 20 min (2.3 µm) is not on the
# grid of level 1 (1 µm in x): it is rounded to 2 µm (4 level-0 voxels).
# At 40 min the region reaches 4 voxels beyond the image's x edge.
TRACK = {0.0: {"z": 0.0, "y": 0.0, "x": 0.0},
         10.0: {"z": 2.0, "y": -1.0, "x": 1.0},
         20.0: {"z": 2.0, "y": -2.0, "x": 2.3},
         30.0: {"z": 4.0, "y": -3.0, "x": 4.0},
         40.0: {"z": 4.0, "y": -3.0, "x": 8.0}}
# The same in level-0 voxels (z, y, x), rounded to level 1's grid.
SHIFTS0 = [(0, 0, 0), (1, -2, 2), (1, -4, 4), (2, -6, 8), (2, -6, 16)]


def window(data, t, channels, starts, sizes):
    """data[t, channels, z, y, x] at starts (z, y, x) with sizes, zeros
    where it reaches outside the data (plain numpy, for the expected
    values)."""
    out = np.zeros((len(channels), *sizes), dtype=data.dtype)
    src, dst = [], []
    for a, n, size in zip(starts, data.shape[2:], sizes):
        low, high = max(a, 0), min(a + size, n)
        src.append(slice(low, high))
        dst.append(slice(low - a, high - a))
    out[(slice(None), *dst)] = data[t][(channels, *src)]
    return out


def expected_crop(data0, factor, times, channels=(0, 1)):
    """The moving crop's expected voxels on a level with this y/x factor,
    for the timepoints `times` (indices into SHAPE's t)."""
    data = data0[:, :, :, ::factor, ::factor]
    frames = []
    for t in times:
        dz, dy, dx = SHIFTS0[t]
        starts = (2 + dz, (Y0 + dy) // factor, (Y0 + dx) // factor)
        sizes = (8, NY // factor, NY // factor)
        frames.append(window(data, t, list(channels), starts, sizes))
    return np.stack(frames)


def make_image(tmp, version="0.5"):
    """The synthetic time-lapse (see SHAPE) and its level-0 voxels."""
    data0 = make_test_image(tmp / f"src_{version}.zarr", shape0=SHAPE,
                            factors=FACTORS, axes="tczyx", scale0=SCALE,
                            version=version)
    return vc.open_image(tmp / f"src_{version}.zarr"), data0


def test_shift_helpers():
    """Shifts: a timepoint's own, interpolated between timepoints, the
    first's before and the last's after; ranges_at moves only the spatial
    ranges; an edit at one timepoint moves only it (same size) or resizes
    every timepoint around its centre (another size); a fixed region
    stays as before."""
    track = {0.0: {"x": 0.0}, 10.0: {"x": 4.0}, 30.0: {"x": 0.0, "y": 2.0}}
    assert vc.track_shift(track, 10.0) == {"x": 4.0}
    assert vc.track_shift(track, 5.0) == {"x": 2.0}
    assert vc.track_shift(track, 20.0) == {"x": 2.0, "y": 1.0}
    assert vc.track_shift(track, -5.0) == {"x": 0.0}
    assert vc.track_shift(track, 99.0) == {"x": 0.0, "y": 2.0}
    assert vc.track_shift({}, 3.0) == {}

    box = vc.CropBox("a", {"t": (0, 30), "x": (10.0, 20.0)},
                     track={10: {"x": 4}, 0: {"x": 0}})
    assert list(box.track) == [0.0, 10.0]          # floats, sorted
    assert vc.ranges_at(box, 10.0) == {"t": (0, 30), "x": (14.0, 24.0)}
    assert vc.ranges_at(box, 5.0)["x"] == (12.0, 22.0)

    # Move the region at 10 min: only that timepoint changes.
    vc.set_ranges_at(box, 10.0, {"x": (16.0, 26.0)})
    assert box.ranges["x"] == (10.0, 20.0)
    assert box.track == {0.0: {"x": 0.0}, 10.0: {"x": 6.0}}
    # Resize it at 10 min: the size changes everywhere, 0 min keeps its
    # centre (15), 10 min becomes exactly what was given.
    vc.set_ranges_at(box, 10.0, {"x": (16.0, 30.0)})
    assert vc.ranges_at(box, 0.0)["x"] == (8.0, 22.0)
    assert vc.ranges_at(box, 10.0)["x"] == (16.0, 30.0)
    # An edit at a time without an entry adds one there.
    vc.set_ranges_at(box, 5.0, {"x": (0.0, 14.0)})
    assert list(box.track) == [0.0, 5.0, 10.0]
    assert vc.ranges_at(box, 5.0)["x"] == (0.0, 14.0)
    # The t range changes for all timepoints.
    vc.set_ranges_at(box, 5.0, {"t": (0.0, 20.0)})
    assert box.ranges["t"] == (0.0, 20.0)

    fixed = vc.CropBox("b", {"x": (1.0, 2.0)})
    vc.set_ranges_at(fixed, 7.0, {"x": (3.0, 9.0)})
    assert fixed.ranges == {"x": (3.0, 9.0)} and fixed.track == {}
    # no time (an image without a time axis): the ranges as given
    vc.set_ranges_at(box, None, {"x": (1.0, 5.0)})
    assert box.ranges["x"] == (1.0, 5.0)


def test_moving_crop_follows_the_region():
    """Each timepoint of the crop, on both levels, is the source at the
    region's place then (zeros outside the image); the shifts are on level
    1's grid; the OME position is the first timepoint's; the record and
    the positions file hold every timepoint's; the check passes; for
    OME-Zarr 0.5 and 0.4."""
    for version in ("0.5", "0.4", "0.6"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image, data0 = make_image(tmp, version)
            box = vc.CropBox("embryo", dict(RANGES), track=TRACK)
            plan = vc.plan_crop(image, box, n_levels=2)
            assert plan.moving and plan.level_indices == [0, 1]
            assert plan.times == [0.0, 10.0, 20.0, 30.0, 40.0]
            # level-0 shifts (t, c, z, y, x) and level 1's (halved in y, x)
            assert plan.shifts[0][:, 2:].tolist() == [list(s)
                                                      for s in SHIFTS0]
            assert plan.shifts[1][:, 2:].tolist() == [
                [dz, dy // 2, dx // 2] for dz, dy, dx in SHIFTS0]
            assert plan.shapes == [(5, 2, 8, 36, 36), (5, 2, 8, 18, 18)]
            assert "moves with the object: 5 positions" in \
                plan.describe(image)
            # the region asked lies inside the one cropped at every
            # timepoint (despite the shifts rounded to the grid)
            for time in plan.times:
                asked = vc.ranges_at(box, time)
                cropped = vc.snapped_ranges_at(image, plan, time)
                for axis in "zyx":
                    assert cropped[axis][0] <= asked[axis][0] + 1e-9, time
                    assert asked[axis][1] <= cropped[axis][1] + 1e-9, time

            result = vc.crop_box(image, box, tmp / "out", n_levels=2)
            assert result.problems == [], result.problems
            crop = vc.open_image(result.path)
            for k, factor in enumerate((1, 2)):
                np.testing.assert_array_equal(
                    crop.levels[k].array[...],
                    expected_crop(data0, factor, range(5)))
            # zeros beyond the image's x edge at 40 min (x voxels 34-69:
            # 6 beyond)
            assert not crop.levels[0].array[4, :, :, :, -6:].any()
            assert crop.levels[0].array[4, :, :, :, -7].any()
            assert crop.format_name == image.format_name

            # OME position: the region's first voxel at the first timepoint
            # (z voxel 2, y and x voxel 18: centres 4 µm and 9 µm)
            t0, s0 = crop.levels[0].translation, crop.levels[0].scale
            assert list(t0[2:]) == [4.0, 9.0, 9.0]
            assert list(s0) == list(SCALE)
            # the record: every timepoint's origin
            record = zarr.open_group(result.path, mode="r").attrs[
                "crop_provenance"]
            origins = [p["origin"] for p in
                       record["moving_region"]["timepoints"]]
            assert origins[3] == {"z": 4.0 + 2 * 2.0, "y": 9.0 - 6 * 0.5,
                                  "x": 9.0 + 8 * 0.5}
            assert record["moving_region"]["requested_track"][2] == {
                "time": 20.0, "shift": TRACK[20.0]}
            # the positions file next to the crop
            assert result.positions == tmp / "out" / (
                result.path.name[:-len(".zarr")] + "_positions.csv")
            with open(result.positions, newline="") as f:
                rows = list(csv.reader(f))
            assert rows[0] == ["timepoint", "t_min", "z_um", "y_um", "x_um"]
            assert rows[4] == ["4", "30", "8", "6", "13"]
            assert len(rows) == 6


def test_check_finds_a_wrong_place():
    """verify_crop works out each timepoint's place on its own: a crop
    written one grid step off at one timepoint fails the check."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, _ = make_image(tmp)
        box = vc.CropBox("embryo", dict(RANGES), track=TRACK)
        plan = vc.plan_crop(image, box, n_levels=2)
        wrong = vc.plan_crop(image, box, n_levels=2)
        wrong.shifts[0][2, 4] += 2               # level 0, 20 min, x
        wrong.shifts[1][2, 4] += 1               # level 1
        out = vc.write_crop(image, wrong, tmp / "wrong.zarr")
        problems = vc.verify_crop(image, plan, out)
        assert problems and all("voxels differ" in p for p in problems), \
            problems
        out = vc.write_crop(image, plan, tmp / "right.zarr")
        assert vc.verify_crop(image, plan, out) == []


def test_every_nth_timepoint_of_a_moving_region():
    """Every 2nd timepoint: the kept timepoints (0, 20, 40 min) at their own
    places; the crop's frame interval is 20 min."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, data0 = make_image(tmp)
        box = vc.CropBox("embryo", dict(RANGES), track=TRACK, t_step=2)
        result = vc.crop_box(image, box, tmp / "out", n_levels=2)
        assert result.problems == []
        crop = vc.open_image(result.path)
        assert result.plan.times == [0.0, 20.0, 40.0]
        np.testing.assert_array_equal(crop.levels[0].array[...],
                                      expected_crop(data0, 1, [0, 2, 4]))
        np.testing.assert_array_equal(crop.levels[1].array[...],
                                      expected_crop(data0, 2, [0, 2, 4]))
        assert crop.levels[0].scale[0] == 20.0


def test_regions_file_keeps_the_track():
    """The regions file stores a moving region's track exactly and reads it
    back (also with Excel's semicolons and decimal commas); a file
    without moving regions has no track column, as before 0.17.0."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, _ = make_image(tmp)
        track = {0.0: {"z": 0.0, "y": 0.1 + 0.2, "x": -1.5},
                 10.0: {"z": 2.0, "y": -3.25, "x": 1e-7}}
        boxes = [vc.CropBox("moving", dict(RANGES), track=track),
                 vc.CropBox("fixed", dict(RANGES))]
        vc.save_boxes_csv(boxes, tmp / "regions.csv", image)
        text = (tmp / "regions.csv").read_text()
        assert "t=10: z=2 y=-3.25 x=1e-07" in text
        loaded = vc.load_boxes_csv(tmp / "regions.csv")
        assert loaded[0].track == track and loaded[1].track == {}
        assert loaded[0].ranges == RANGES

        vc.save_boxes_csv(boxes[1:], tmp / "fixed.csv", image)
        assert "track" not in (tmp / "fixed.csv").read_text()

        (tmp / "excel.csv").write_text(
            "name;x_start_um;x_stop_um;track\n"
            "a;1,5;9,5;t=0: x=0 | t=10: x=2,25\n", encoding="utf-8")
        assert vc.load_boxes_csv(tmp / "excel.csv")[0].track == {
            0.0: {"x": 0.0}, 10.0: {"x": 2.25}}
        for bad in ("t=0 x=1", "t=0: x1", "t=0: x=1 | t=0: x=2"):
            (tmp / "bad.csv").write_text(f"name,track\na,{bad}\n")
            try:
                vc.load_boxes_csv(tmp / "bad.csv")
            except ValueError as err:
                assert "row 2" in str(err) and "track" in str(err), err
            else:
                raise AssertionError(f"accepted {bad!r}")


def test_moving_region_saved_as_tiff():
    """Save as TIFF of a moving region: that level, every timepoint at its
    place (the same voxels as the crop's level), with its positions
    file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, data0 = make_image(tmp)
        box = vc.CropBox("embryo", dict(RANGES), track=TRACK)
        result = vc.save_region_tiff(image, box, tmp / "out", level=1)
        assert result.problems == [], result.problems
        data, meta, _ = read_tiff(result.tiff)        # (t, z, c, y, x)
        expected = expected_crop(data0, 2, range(5))  # (t, c, z, y, x)
        np.testing.assert_array_equal(data, expected.transpose(0, 2, 1, 3, 4))
        assert result.positions.name.endswith("_level1_positions.csv")
        assert result.positions.is_file()


def test_moving_regions_refused_clearly():
    """A track on an image without a time axis, a moving region with
    polygons, and a region entirely outside the image at a timepoint are
    refused with a message that says why; Fix in place gives the union of
    the region's places, with no track."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, _ = make_image(tmp)
        still = make_test_image(tmp / "still.zarr", shape0=(2, 16, 64, 64),
                                factors=[(1, 1, 1, 1)], axes="czyx",
                                scale0=(1.0, 2.0, 0.5, 0.5))
        assert still is not None
        cases = [
            (vc.open_image(tmp / "still.zarr"),
             vc.CropBox("a", dict(RANGES), track=TRACK), "no time axis"),
            (image, vc.CropBox("b", dict(RANGES), track=TRACK,
                               outlines={("y", "x"): [(10, 10), (20, 10),
                                                      (20, 20)]}),
             "outlines"),
            # (x 0 -> 40 µm over 40 min: from 30 min on, x voxels 80 and up)
            (image, vc.CropBox("c", dict(RANGES),
                               track={0: {"x": 0}, 40: {"x": 40.0}}),
             "outside the image along x at t = 30"),
        ]
        for img, box, words in cases:
            try:
                vc.plan_crop(img, box)
            except ValueError as err:
                assert words in str(err), err
            else:
                raise AssertionError(f"{box.name}: not refused")

        box = vc.CropBox("d", {"t": (0, 25), **RANGES}, track=TRACK)
        fixed = vc.fixed_in_place(image, box)
        assert fixed.track == {} and box.track      # (the box is unchanged)
        # timepoints 0-2 (t 0-25 min): x shifts 0, 1, 2.3 µm
        assert fixed.ranges["x"] == (9.75, 25.75 + 2.3)
        assert fixed.ranges["y"] == (9.75 - 2.0, 25.75)
        assert fixed.ranges["t"] == (0, 25)


# --- Autocropper: moving regions (autocrop, the panel) ----------------

def _drifting_blob():
    """A t, z, y, x time-lapse (in memory, as the plugin reads napari
    layers): a blob that moves 12 voxels (6 µm) along x per timepoint and
    grows by 2 voxels, missing at timepoint 3 (30 min). Returns the image,
    the blob's true voxel box per timepoint and the voxels (t, z, y, x)."""
    from test_autocrop import _blob, _image, _noisy
    shape = (6, 16, 48, 120)
    signal, true = np.zeros(shape), {}
    for t in range(6):
        if t != 3:
            lo, hi = (4, 12, 10 + 12 * t), (12, 36, 30 + 14 * t)
            signal[t] = _blob(shape[1:], lo, hi)
            true[t] = (lo, hi)
    data = _noisy(signal)
    image = _image(data, ("t", "z", "y", "x"), (10.0, 2.0, 0.5, 0.5))
    return image, true, data


def test_moving_region_follows_a_drifting_growing_blob():
    """autocrop.moving_region: one size (the largest box plus the margin),
    holding the blob at every timepoint found; the timepoint where it was
    missed lies between its neighbours; kept inside the region searched
    (moved, not cut). The crop of it (an image in memory: a TIFF) keeps the
    blob in place, while a fixed region's crop shows it drifting."""
    import tifffile

    from volume_cropper import autocrop as ac
    from test_autocrop import _edges
    image, true, _ = _drifting_blob()
    result = ac.find_object(image, level=0, margin=1.0)
    assert result.n_found == 5
    ranges, track = ac.moving_region(image, result)
    assert sorted(track) == [0.0, 10.0, 20.0, 40.0, 50.0]
    assert track[0.0] == {"z": 0.0, "y": 0.0, "x": 0.0}
    box = vc.CropBox("blob", ranges, track=track)
    assert box.ranges["t"] == (0.0, 50.0)
    scale = (2.0, 0.5, 0.5)
    for t, (lo, hi) in true.items():
        at = vc.ranges_at(box, 10.0 * t)
        for k, axis in enumerate("zyx"):
            edges = _edges(lo[k], hi[k], scale[k], 0.0)
            assert at[axis][0] <= edges[0] and at[axis][1] >= edges[1], \
                (t, axis, at[axis], edges)
    # the largest box (x: 30 voxels = 15 µm at t = 5, found within 4 voxels
    # = 2 µm of each true edge, as in test_autocrop) plus 2 x 1 µm margin
    width = ranges["x"][1] - ranges["x"][0]
    assert 15.0 + 2.0 <= width <= 15.0 + 2 * 2.0 + 2.0, width
    # missed at 30 min: halfway between 20 and 40 min (interpolated)
    assert abs(vc.track_shift(track, 30.0)["x"]
               - (track[20.0]["x"] + track[40.0]["x"]) / 2) < 1e-9
    # inside x 0-40 µm: the last places are moved back, the size kept
    ranges2, track2 = ac.moving_region(image, result,
                                       within={"x": (0.0, 40.0)})
    box2 = vc.CropBox("b", ranges2, track=track2)
    for time in track2:
        low, high = vc.ranges_at(box2, time)["x"]
        assert -1e-9 <= low and high <= 40.0 + 1e-9, (time, low, high)
        assert abs((high - low) - width) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        result_m = vc.crop_box(image, box, Path(tmp) / "moving")
        result_f = vc.crop_box(image, vc.fixed_in_place(image, box),
                               Path(tmp) / "fixed")
        assert result_m.problems == [] and result_f.problems == []
        assert result_m.positions.is_file() and result_f.positions is None
        centres = {}
        for name, res in (("moving", result_m), ("fixed", result_f)):
            with tifffile.TiffFile(res.path) as tif:
                data = tif.series[0].asarray()
                axes = tif.series[0].axes
            data = data.reshape([n for n, a in zip(data.shape, axes)
                                 if a in "TZYX"])     # (t, z, y, x)
            weights = np.clip(data.astype(float) - 400, 0, None)
            x = np.arange(data.shape[-1])
            centres[name] = [float((w.sum(axis=(0, 1)) * x).sum() / w.sum())
                             for k, w in enumerate(weights) if k != 3]
        moving, fixed = np.ptp(centres["moving"]), np.ptp(centres["fixed"])
        assert moving <= 2.0, centres["moving"]    # in place (voxels)
        assert fixed >= 50.0, centres["fixed"]     # drifts 13 voxels/step


def test_panel_fit_makes_a_moving_region():
    """In a hidden window, with a time-lapse: "Moving region" is ticked and
    usable, Outline too, and the warning shows only with both ticked; Fit
    adds a moving region; a region with a polygon loses it to a moving
    fit, and Undo fit gives the polygon and the fixed region back; with
    Moving region off a fit is fixed; without time, Moving region is
    greyed out."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from test_autocrop import _pump
    _, _, data = _drifting_blob()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            data, name="blob", axis_labels=("t", "z", "y", "x"),
            scale=(10.0, 2.0, 0.5, 0.5),
            units=("minute", "um", "um", "um"))])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        group.margin_spin.setValue(1.0)
        assert group.moving_check.isEnabled() and \
            group.moving_check.isChecked()
        # Outline usable too; the warning only with both ticked
        assert group.outline_check.isEnabled()
        assert group.timed_note.isHidden()
        group.outline_check.setChecked(True)
        assert not group.timed_note.isHidden()
        group.outline_check.setChecked(False)
        group.fit()                                   # nothing selected
        box = panel.boxes[-1]
        assert box.name == "object01" and len(box.track) == 5, box.track
        assert "moving with the object" in group.info.text()

        i = panel._new_box({"y": (0.0, 24.0), "x": (0.0, 60.0)})
        drawn = vc.normalize_outlines(
            {("y", "x"): [(1.0, 1.0), (1.0, 59.0), (23.0, 59.0)]})
        panel.boxes[i].outlines = drawn
        before = dict(panel.boxes[i].ranges)
        panel._refresh(select=i)
        group.fit()
        assert panel.boxes[i].track and not panel.boxes[i].outlines
        assert "polygons removed" in group.info.text()
        group.undo()
        assert panel.boxes[i].outlines == drawn
        assert panel.boxes[i].track == {}
        assert panel.boxes[i].ranges == before

        group.moving_check.setChecked(False)
        panel._refresh(select=i)
        group.fit()
        assert panel.boxes[i].track == {}

        panel.boxes.clear()                 # (else: "discard the regions?")
        panel._refresh(select=None)
        assert panel.open_layers([Image(data[0], name="one",
                                        units=("um",) * 3)])
        _pump()
        assert not panel.object_group.moving_check.isEnabled()
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_views_and_panel_follow_a_moving_region():
    """In a hidden window, a moving region is drawn where it is at the
    timepoint shown: its editable rectangle in XY, its ranges as cropped
    (the lines), its 3D frame and the 3D clipping, the number fields and
    the plan. A drag at one timepoint corrects only that timepoint; a
    larger stop in the number fields changes the size at all timepoints,
    each keeping its centre; a polygon is refused; Fix in place makes it
    one fixed region (all its places)."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    from test_autocrop import _pump
    _, _, data = _drifting_blob()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            data, name="blob", axis_labels=("t", "z", "y", "x"),
            scale=(10.0, 2.0, 0.5, 0.5),
            units=("minute", "um", "um", "um"))])
        _pump()
        panel.object_group.setChecked(True)
        panel.object_group.fit()
        i = len(panel.boxes) - 1
        box = panel.boxes[i]
        assert box.track
        assert "moves (5 timepoints found)" in panel.box_list.item(i).text()
        assert not panel.fix_button.isHidden()
        ortho = panel.ortho
        panel.show_3d_check.setChecked(True)
        _pump()
        ortho.set_clip(True)
        _pump()
        xy = ortho.views["XY"]
        seen = {}
        for time in (0.0, 20.0, 50.0):
            ortho._set_cursor({"t": time})
            _pump(0.1)
            assert ortho.time_shown() == time
            asked = vc.ranges_at(box, time)
            k = xy.shape_boxes.index(i)
            corners = np.asarray(xy.shapes.data[k])[:, -2:]   # (y, x)
            assert abs(corners[:, 1].min() - asked["x"][0]) < 1e-6
            cropped = vc.snapped_ranges_at(panel.image, panel._plan(box),
                                           time)
            assert ortho.box_ranges[i] == cropped
            frame = ortho.views["3D"].box_layers[i]
            assert abs(frame.translate[-1] - sum(cropped["x"]) / 2) < 1e-6
            planes = ortho.views["3D"].channel_layers[0] \
                .experimental_clipping_planes
            assert any(abs(p.position[-1] - cropped["x"][0]) < 1e-6
                       for p in planes), [p.position for p in planes]
            assert abs(panel.range_spins["x"][0].value()
                       - asked["x"][0]) < 0.01
            assert "Moves with the object" in panel.plan_label.text()
            seen[time] = cropped["x"]
        assert seen[50.0][0] > seen[0.0][0] + 20.0     # ~30 µm in 50 min

        # a drag at 20 min (2 µm along x): only 20 min moves
        ortho._set_cursor({"t": 20.0})
        _pump(0.1)
        before = {t: vc.ranges_at(box, t) for t in (0.0, 20.0, 40.0)}
        moved = {"y": before[20.0]["y"],
                 "x": tuple(v + 2.0 for v in before[20.0]["x"])}
        panel._boxes_changed({i: (moved, None, ov.RECT)}, ("y", "x"))
        assert np.allclose(vc.ranges_at(box, 20.0)["x"], moved["x"])
        for t in (0.0, 40.0):
            assert vc.ranges_at(box, t) == before[t]
        # a stop 4 µm further in the number fields: wider everywhere
        width = before[0.0]["x"][1] - before[0.0]["x"][0]
        centre = sum(before[0.0]["x"]) / 2
        stop = panel.range_spins["x"][1]
        stop.setValue(stop.value() + 4.0)
        _pump(0.1)
        at0 = vc.ranges_at(box, 0.0)["x"]
        assert abs((at0[1] - at0[0]) - (width + 4.0)) < 0.01, at0
        assert abs(sum(at0) / 2 - centre) < 0.01
        assert abs(vc.ranges_at(box, 20.0)["x"][0] - moved["x"][0]) < 0.01

        # a polygon on a moving region: refused
        panel._box_added({}, ("y", "x"), [(5.0, 10.0), (5.0, 20.0),
                                          (15.0, 15.0)])
        assert box.outlines == {} and "not possible yet" in \
            panel.status.text()
        # Fix in place: fixed, holding all its places
        union = vc.fixed_in_place(panel.image, box).ranges
        panel._fix_in_place()
        assert box.track == {} and box.ranges == union
        assert panel.box_list.item(i).text() == box.name
        assert panel.fix_button.isHidden()
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


# --- outlines per timepoint (the user's choice of 2026-11-09) ------------------

def _rectangle_outlines(time):
    """A silhouette in XY at a time: a rectangle that moves with TRACK (its
    edges between voxel centres, so that no centre lies on one)."""
    dy, dx = TRACK[time]["y"], TRACK[time]["x"]
    y0, y1, x0, x1 = 12.25 + dy, 20.25 + dy, 13.25 + dx, 22.25 + dx
    return {("y", "x"): {(-1.0, 40.0): [(y0, x0), (y0, x1), (y1, x1),
                                         (y1, x0)]}}, (y0, y1, x0, x1)


def test_outlines_per_timepoint_shape_each_timepoint():
    """Each timepoint of the crop is shaped by its own outline, at the
    object's place then (compared with numpy: voxels whose centre lies
    outside that timepoint's rectangle are 0, the others the source's); a
    timepoint without an outline (20 min) keeps the whole region; the check
    passes on both levels; the plan and the record say so; outlines per
    timepoint without a track are refused."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, data0 = make_image(tmp)
        timed, edges = {}, {}
        for time in TRACK:
            if time != 20.0:                     # "missed" at 20 min
                timed[time], edges[time] = _rectangle_outlines(time)
        box = vc.CropBox("embryo", dict(RANGES), track=TRACK,
                         timed_outlines=timed)
        result = vc.crop_box(image, box, tmp / "out", n_levels=2)
        assert result.problems == [], result.problems
        assert "outlines per timepoint at 4 of 5 timepoints" in \
            result.plan.describe(image)
        crop = vc.open_image(result.path).levels[0].array[...]
        expected = expected_crop(data0, 1, range(5))
        for t, time in enumerate(TRACK):
            if time not in edges:
                np.testing.assert_array_equal(crop[t], expected[t])
                continue
            dz, dy, dx = SHIFTS0[t]
            y = (Y0 + dy + np.arange(NY)) * 0.5          # voxel centres
            x = (Y0 + dx + np.arange(NY)) * 0.5
            y0, y1, x0, x1 = edges[time]
            keep = (((y > y0) & (y < y1))[:, None]
                    & ((x > x0) & (x < x1))[None, :])
            np.testing.assert_array_equal(crop[t], expected[t] * keep)
            assert 0.1 < keep.mean() < 0.9
        record = zarr.open_group(result.path, mode="r").attrs[
            "crop_provenance"]["moving_region"]["outlines_per_timepoint"]
        assert [e["time"] for e in record["timepoints"]] == [0, 10, 30, 40]
        try:
            vc.plan_crop(image, vc.CropBox("b", dict(RANGES),
                                           timed_outlines=timed))
        except ValueError as err:
            assert "does not move" in str(err)
        else:
            raise AssertionError("outlines per timepoint without a track")


def test_regions_file_keeps_outlines_per_timepoint():
    """The regions file names a JSON file next to it with the outlines per
    timepoint (too long for CSV cells); they load back exactly; a missing
    JSON file is reported."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, _ = make_image(tmp)
        timed = {time: _rectangle_outlines(time)[0] for time in (0.0, 10.0)}
        boxes = [vc.CropBox("moving", dict(RANGES), track=TRACK,
                            timed_outlines=timed),
                 vc.CropBox("plain", dict(RANGES))]
        vc.save_boxes_csv(boxes, tmp / "regions.csv", image)
        sidecar = tmp / "regions_outlines_per_timepoint.json"
        assert sidecar.is_file()
        assert "regions_outlines_per_timepoint.json" in \
            (tmp / "regions.csv").read_text()
        loaded = vc.load_boxes_csv(tmp / "regions.csv")
        assert loaded[0].timed_outlines == timed
        assert loaded[1].timed_outlines == {}
        sidecar.unlink()
        try:
            vc.load_boxes_csv(tmp / "regions.csv")
        except ValueError as err:
            assert "missing" in str(err), err
        else:
            raise AssertionError("loaded without its JSON file")


def test_outlines_per_timepoint_from_find_a_bright_object():
    """Autocropper with silhouettes per timepoint: an outline in
    XY, XZ and YZ at each of the 5 timepoints found; the crop keeps >= 99 %
    of the blob's voxels at each of them and sets part of the region to 0
    (the corners); at the timepoint where it was missed, the whole region
    is kept."""
    import tifffile

    from volume_cropper import autocrop as ac
    from test_autocrop import _blob
    image, true, data = _drifting_blob()
    result = ac.find_object(image, level=0, margin=1.0,
                            timed_silhouettes=True)
    assert sorted(result.timed_silhouettes) == [0, 1, 2, 4, 5]
    ranges, track = ac.moving_region(image, result)
    box = vc.CropBox("blob", ranges, track=track)
    box.timed_outlines = ac.timed_silhouettes(image, result, box)
    assert sorted(box.timed_outlines) == [0.0, 10.0, 20.0, 40.0, 50.0]
    assert all(set(o) == {("y", "x"), ("z", "x"), ("y", "z")}
               for o in box.timed_outlines.values())
    with tempfile.TemporaryDirectory() as tmp:
        res = vc.crop_box(image, box, Path(tmp))
        assert res.problems == [], res.problems
        with tifffile.TiffFile(res.path) as tif:
            crop = tif.series[0].asarray()
            axes = tif.series[0].axes
        crop = crop.reshape([n for n, a in zip(crop.shape, axes)
                             if a in "TZYX"])            # (t, z, y, x)
        with open(res.positions, newline="") as f:
            rows = list(csv.reader(f))[1:]
        scale = np.array([2.0, 0.5, 0.5])
        for t in range(6):
            frame = crop[t]
            offset = np.round(np.array([float(v) for v in rows[t][2:]])
                              / scale).astype(int)
            if t not in true:                    # missed: whole region
                assert not (frame == 0).any()
                continue
            voxels = np.argwhere(_blob(data.shape[1:], *true[t]) > 0) - offset
            assert ((voxels >= 0) & (voxels < frame.shape)).all()
            kept = (frame[tuple(voxels.T)] != 0).mean()
            assert kept >= 0.99, (t, kept)
            assert (frame == 0).mean() > 0.05, t     # corners cut


def test_panel_outlines_per_timepoint():
    """In a hidden window: Moving region and Outline ticked (the warning
    shows) -> a moving region with an outline at each timepoint found ("·
    moves, outlined"); Undo fit removes it. The views show the outlines of
    the timepoint shown (none where the object was missed), and so does the
    polygon list; reshaping a silhouette, or deleting one, changes that
    timepoint only; Fix in place asks, then removes them."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from test_autocrop import _pump
    _, _, data = _drifting_blob()
    saved = crop_gui.FileLoadDialog.exec, crop_gui._confirm
    crop_gui.FileLoadDialog.exec = lambda self: 1
    asked = []
    crop_gui._confirm = lambda *a: asked.append(a[1]) or True
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            data, name="blob", axis_labels=("t", "z", "y", "x"),
            scale=(10.0, 2.0, 0.5, 0.5),
            units=("minute", "um", "um", "um"))])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        group.margin_spin.setValue(1.0)
        group.outline_check.setChecked(True)
        assert not group.timed_note.isHidden()
        group.fit()
        assert panel.boxes and panel.boxes[-1].timed_outlines
        group.undo()
        assert not panel.boxes
        group.fit()
        i = len(panel.boxes) - 1
        box = panel.boxes[i]
        assert sorted(box.timed_outlines) == [0.0, 10.0, 20.0, 40.0, 50.0]
        assert "moves, outlined (5 timepoints found)" in \
            panel.box_list.item(i).text()
        assert "with its outline at each of 5 timepoints" in \
            group.info.text()
        panel._refresh(select=i)
        ortho = panel.ortho
        for time, n in ((0.0, 3), (30.0, 0), (50.0, 3)):
            ortho._set_cursor({"t": time})
            _pump(0.1)
            shown = ortho.boxes[i].outlines
            assert shown == vc.timed_outlines_at(box, time), time
            assert panel.polygon_list.count() == n, time
        assert "Outlines per timepoint at 5 of 6" in panel.plan_label.text()

        # reshape the XY silhouette at 0 min; delete it at 10 min
        ortho._set_cursor({"t": 0.0})
        _pump(0.1)
        key0, points0 = next(iter(box.timed_outlines[0.0][("y", "x")]
                                  .items()))
        before10 = vc.timed_outlines_at(box, 10.0)
        moved = [(y + 1.0, x) for y, x in points0]
        panel._boxes_changed({i: ({}, moved, key0)}, ("y", "x"))
        assert box.timed_outlines[0.0][("y", "x")][key0] == moved
        assert vc.timed_outlines_at(box, 10.0) == before10
        ortho._set_cursor({"t": 10.0})
        _pump(0.1)
        key10 = next(iter(box.timed_outlines[10.0][("y", "x")]))
        panel._boxes_removed([(i, key10)], ("y", "x"))
        assert ("y", "x") not in box.timed_outlines[10.0]
        assert ("y", "x") in box.timed_outlines[0.0]

        panel._fix_in_place()
        assert asked == ["Fix in place"]
        assert box.track == {} and box.timed_outlines == {}
    finally:
        crop_gui.FileLoadDialog.exec, crop_gui._confirm = saved
        viewer.close()
        _pump()


def test_seeds_take_their_region_colour_follow_it_and_are_listed():
    """The user's requests of 2026-11-09, in a hidden window: seeds are
    white (dark edge) until Fit to seeds, then in the colour of the region
    they gave; a seed whose region moves follows it (full where the object
    was followed, faded at the missed timepoint); a seed without a region
    stays where it was clicked, full only at its own timepoint. The table
    lists each seed's timepoint, position and region ("no object" for
    one in the background); a click on a row goes to that seed; Delete
    removes the selected seed; after Undo fit the seed is white again."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from volume_cropper import ortho_views as ov
    from test_autocrop import _pump
    _, _, data = _drifting_blob()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            data, name="blob", axis_labels=("t", "z", "y", "x"),
            scale=(10.0, 2.0, 0.5, 0.5),
            units=("minute", "um", "um", "um"))])
        _pump()
        group, ortho = panel.object_group, panel.ortho
        group.setChecked(True)
        group.margin_spin.setValue(1.0)
        ortho._set_cursor({"t": 0.0})
        group.seed_button.setChecked(True)
        on_blob = {"t": 0.0, "z": 16.0, "y": 12.0, "x": 10.0}  # t 0: x 5-15
        empty = {"t": 20.0, "z": 16.0, "y": 22.0, "x": 55.0}
        group._seed_clicked(dict(on_blob), "XY")
        group._seed_clicked(dict(empty), "XY")
        white, faded = ov.rgba(ov.SEED_COLOR), ov.FADED_ALPHA
        shown = ortho.seeds_shown()
        assert shown[0][1] == white                      # its own timepoint
        assert shown[1][1] == ov.rgba(ov.SEED_COLOR, faded)
        assert group.seed_table.rowCount() == 2
        assert [group.seed_table.item(1, c).text() for c in range(6)] == [
            "2", "3", "16.0", "22.0", "55.0", "—"]

        group.fit_to_seeds()
        box = next(b for b in panel.boxes if b.name == "object01")
        i = panel.boxes.index(box)
        color = ov.BOX_COLORS[i % len(ov.BOX_COLORS)]
        assert box.track
        assert group.seed_table.item(0, 5).text() == "object01"
        assert group.seed_table.item(1, 5).text() == "no object", \
            group.info.text()
        # (the blob is missing at 30 min: following from the seed goes on
        # after it (0.17.2), so the region holds 0-50 min and the seed is
        # faded only at 30 min)
        assert sorted(box.track) == [0.0, 10.0, 20.0, 40.0, 50.0]
        for time, full in ((0.0, True), (20.0, True), (30.0, False),
                           (50.0, True)):
            ortho._set_cursor({"t": time})
            _pump(0.05)
            point, rgba = ortho.seeds_shown()[0]
            shift = vc.track_shift(box.track, time)["x"]
            assert abs(point["x"] - (on_blob["x"] + shift)) < 1e-9, time
            assert rgba == ov.rgba(color, 1.0 if full else faded), time
            other = ortho.seeds_shown()[1]          # clicked at 20 min
            assert other[0] == empty, time
            assert other[1][3] == (1.0 if time == 20.0 else faded), time
        # the XY view draws them so (at 50 min: the first in full, the
        # second, clicked at 20 min, faded)
        drawn = np.asarray(ortho.views["XY"].seeds.face_color)
        assert len(drawn) == 2 and np.allclose(
            drawn, [rgba for _, rgba in ortho.seeds_shown()])
        assert np.allclose(drawn[0], ov.rgba(color, 1.0))
        assert np.isclose(drawn[1][3], faded)

        group._go_to_seed(1)                       # a click on row 2
        assert ortho.cursor["t"] == 20.0 and ortho.cursor["x"] == 55.0
        group.seed_table.selectRow(1)
        group.delete_selected_seeds()
        assert len(group.seeds) == 1 and len(ortho.seeds) == 1
        assert group.seed_table.rowCount() == 1

        group.undo()                               # the region removed
        assert box not in panel.boxes
        assert ortho.seeds_shown()[0][1][:3] == white[:3]
        assert group.seed_table.item(0, 5).text() == "—"
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_seed_followed_past_a_timepoint_without_the_object():
    """(The user's choice of 2026-11-10) The object followed from a seed
    is missing at one timepoint (30 min): following goes on after it, from
    the last place it was found (it stopped there: 3 of 6 timepoints)."""
    from volume_cropper import autocrop as ac
    image, true, _ = _drifting_blob()
    result = ac.find_object(image, level=0, margin=1.0,
                            seed={"t": 0.0, "z": 16.0, "y": 12.0,
                                  "x": 10.0})
    found = [t for t, box in zip(result.times, result.found) if box]
    assert result.n_found == 5 and found == [0, 1, 2, 4, 5], found
    ranges, track = ac.moving_region(image, result)
    assert sorted(track) == [0.0, 10.0, 20.0, 40.0, 50.0]


def test_autocropper_fit_over_one_timepoint():
    """(The 0.17.0 review) With Moving region and Outline ticked, a fit in
    a region over one timepoint made no outline, and its time range (one
    timepoint, 20-20 min) was "outside the image" at the next fit. Over
    one timepoint the fit is a fixed region, with its outline."""
    from napari.layers import Image

    from volume_cropper import crop_gui
    from test_autocrop import _pump
    _, _, data = _drifting_blob()
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            data, name="blob", axis_labels=("t", "z", "y", "x"),
            scale=(10.0, 2.0, 0.5, 0.5),
            units=("minute", "um", "um", "um"))])
        _pump()
        group = panel.object_group
        group.setChecked(True)
        group.moving_check.setChecked(True)
        group.outline_check.setChecked(True)
        i = panel._new_box({"t": (15.0, 25.0)})      # timepoint 2 (20 min)
        for _ in range(2):                           # and again
            panel._refresh(select=i)
            group.fit()
            box = panel.boxes[i]
            assert "fitted, with its outline" in group.info.text(), \
                group.info.text()
            assert box.ranges["t"] == (20.0, 20.0) and not box.track
            assert box.outlines and not box.timed_outlines
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_command_line_writes_the_positions_file():
    """(The 0.17.0 review) A moving region cropped on the command line got
    no <crop>_positions.csv (crop_box wrote it, the command line not)."""
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, _ = make_image(tmp)
        box = vc.CropBox("embryo", dict(RANGES), track=TRACK)
        vc.save_boxes_csv([box], tmp / "regions.csv", image)
        with contextlib.redirect_stdout(io.StringIO()) as text:
            code = vc.main([str(image.path), str(tmp / "regions.csv"),
                            str(tmp / "out"), "--min-size", "4"])
        assert code == 0, text.getvalue()
        crop = vc.output_path_for(image, box, tmp / "out")
        rows = list(csv.reader(open(vc.positions_path_for(crop),
                                    encoding="utf-8")))
        times = vc.plan_crop(image, box, min_size=4).times
        assert rows[0][0] == "timepoint" and len(rows) == 1 + len(times)


def test_what_to_load_and_a_drag_at_the_edge_keep_the_regions():
    """(The 0.17.0 review) "What to load…" with a region selected (fixed or
    moving) crashed while the views were rebuilt, and the regions vanished
    from the views. And a moving region dragged where it sticks out of the
    image was clipped there, and the smaller size went to all its
    timepoints (15 -> 9.75 um wide)."""
    from types import SimpleNamespace

    from napari.layers import Image
    from napari.layers.base._base_constants import ActionType

    from volume_cropper import crop_gui
    from test_autocrop import _pump
    _, _, data = _drifting_blob()          # x: 120 voxels of 0.5 um
    saved = crop_gui.FileLoadDialog.exec
    crop_gui.FileLoadDialog.exec = lambda self: 1
    viewer, panel = crop_gui.open_window(show=False)
    try:
        assert panel.open_layers([Image(
            data, name="blob", axis_labels=("t", "z", "y", "x"),
            scale=(10.0, 2.0, 0.5, 0.5),
            units=("minute", "um", "um", "um"))])
        _pump()
        box = vc.CropBox("m", {"t": (0.0, 60.0), "z": (0.0, 32.0),
                               "y": (5.0, 20.0), "x": (40.0, 55.0)},
                         track={0.0: {"x": 0.0}, 50.0: {"x": 10.0}})
        still = vc.CropBox("still", {**box.ranges, "x": (5.0, 15.0)})
        panel.boxes += [still, box]
        # (a fixed region selected: test_plugin's 3D test; the layers can
        # be read again only once here, as they are in no napari window)
        panel._refresh(select=1)
        _pump()
        panel._change_what_to_load()
        _pump()
        assert len(panel.ortho.boxes) == len(panel.boxes) == 2
        panel._refresh(select=1)
        ortho = panel.ortho
        ortho._set_cursor({"t": 50.0})      # x 50-65 um: past the image
        _pump()
        before = {t: vc.ranges_at(box, t)["x"] for t in (0.0, 20.0, 50.0)}
        xy = ortho.views["XY"]
        k = xy.shape_boxes.index(1)
        moved = np.asarray(xy.shapes.data[k], dtype=float)
        moved[:, -2] += 1.0                 # 1 um down in y, x untouched
        shapes = list(xy.shapes.data)
        shapes[k] = moved
        ortho._syncing["shapes"] = True
        xy.shapes.data = shapes
        ortho._syncing["shapes"] = False
        ortho._on_shapes(xy, SimpleNamespace(action=ActionType.CHANGED,
                                             data_indices=(k,)))
        _pump()
        after = {t: vc.ranges_at(box, t)["x"] for t in before}
        assert after == before, (before, after)
    finally:
        crop_gui.FileLoadDialog.exec = saved
        viewer.close()
        _pump()


def test_saving_regions_leaves_another_files_json_alone():
    """(0.23.2, the pre-publication review) Saving regions with outlines per
    timepoint writes <regions>_outlines_per_timepoint.json beside the CSV:
    a file of that name that Volume Cropper did not write is not replaced
    (nothing is written then); its own one is, as before."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image, _ = make_image(tmp)
        timed = {time: _rectangle_outlines(time)[0] for time in (0.0, 10.0)}
        boxes = [vc.CropBox("moving", dict(RANGES), track=TRACK,
                            timed_outlines=timed)]
        sidecar = tmp / "regions_outlines_per_timepoint.json"
        sidecar.write_text('{"notes": "someone else\'s"}', encoding="utf-8")
        try:
            vc.save_boxes_csv(boxes, tmp / "regions.csv", image)
        except ValueError as err:
            assert "not written by Volume Cropper" in str(err), err
        else:
            raise AssertionError("another file's JSON was replaced")
        assert sidecar.read_text(encoding="utf-8") == \
            '{"notes": "someone else\'s"}'
        assert not (tmp / "regions.csv").exists()
        sidecar.unlink()
        vc.save_boxes_csv(boxes, tmp / "regions.csv", image)
        vc.save_boxes_csv(boxes, tmp / "regions.csv", image)   # its own: fine
        assert vc.load_boxes_csv(tmp / "regions.csv")[0].timed_outlines == timed


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"all {len(tests)} tests passed")
