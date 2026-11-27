"""Tests for the dots that change a crop region in the 3D view (0.18.0):
box_handles.py and the 3D view's drag handler (a hidden window).

Run with ``python run_all_tests.py handles`` (or this file alone). They
check that:
- a region has seven dots: its centre and the middle of each face;
- the dot picked is the one nearest the mouse's ray on screen (distance
  times the zoom), within GRAB_PIXELS; seen along an axis, where the centre
  and two faces line up, the centre wins;
- a face follows the point of its axis nearest the mouse's ray, and can't
  follow when the line of sight runs along its axis; the centre follows
  the ray's crossing with a plane facing the camera;
- a face stays inside the image shown and at least one voxel from the
  other face; a moved region keeps its size and stays inside; polygons
  moved with a region move their points and their slice (or a
  silhouette's depth range);
- in the window, a face's dot goes where the mouse is and only that face
  changes (the other values stay exactly as set): along z the crop grows;
  beyond a polygon in its own plane the crop stays at the polygon (which
  still shapes it), inside it the crop follows; the centre's dot moves the
  region and its polygons across the screen; the camera does not turn
  meanwhile (and turns again afterwards); a press away from the dots is
  left to the camera; the dots don't enlarge the scene's extent;
- (the second round) Shift resizes around the centre and stops at the
  image's edge together; Ctrl lands on a round step (~1 % of the image);
  the click picks the frame nearest the mouse on screen; in the window,
  clipping follows a drag, a click on another frame selects it (a press
  that moves does not), Up / Down nudge the face last pressed by a voxel
  (Shift: ten), every key binding has a name (napari needs one), Zoom to
  region centres the 3D view;
- undo / redo: every change to the regions is a step (a 3D drag one,
  changes in quick succession one), undo and redo restore them in order
  on the same region objects, a new change drops the redo steps, another
  image starts a new history.
"""

# The tool's package (volume_cropper/) is in the tool folder, one level up
# from this tests folder, so that folder goes first on the import path.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from volume_cropper import box_handles as bh
from volume_cropper import core as vc
from test_volume_cropper import make_test_image

AXES = ["z", "y", "x"]
RANGES = {"z": (10.0, 30.0), "y": (5.0, 15.0), "x": (2.0, 12.0)}


def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication
    end = time.time() + seconds
    while time.time() < end:
        QApplication.instance().processEvents()
        time.sleep(0.01)


def test_handle_points():
    points = bh.handle_points(RANGES, AXES)
    assert len(points) == 7
    assert np.allclose(points[bh.CENTRE], (20, 10, 7))
    assert np.allclose(points[("x", 1)], (20, 10, 12))
    assert np.allclose(points[("z", 0)], (10, 10, 7))


def test_pick_handle():
    points = bh.handle_points(RANGES, AXES)
    along_z = (1.0, 0.0, 0.0)
    # looking along z through the middle: centre and both z faces line up
    assert bh.pick_handle(points, (0, 10, 7), along_z, 1.0) == bh.CENTRE
    # looking along z at the high-x face
    assert bh.pick_handle(points, (0, 10, 12), along_z, 1.0) == ("x", 1)
    # 3 µm beside it: 6 pixels at zoom 2 (picked), 15 at zoom 5 (not; the
    # high-y face is 5.4 µm away, 10.8 and 27 pixels)
    assert bh.pick_handle(points, (0, 13, 12), along_z, 2.0) == ("x", 1)
    assert bh.pick_handle(points, (0, 13, 12), along_z, 5.0) is None
    # seen along x, the near face (low x, smaller depth) wins over the far
    # one only if the centre is not on the line: here it is
    assert bh.pick_handle(points, (20, 10, -50), (0, 0, 1), 1.0) == bh.CENTRE


def test_along_axis_and_on_plane():
    # the x axis through the origin; a ray along z through (., 3, 7)
    assert abs(bh.along_axis((0, 0, 0), 2, (5, 3, 7), (1, 0, 0)) - 7) < 1e-9
    # an oblique ray: the nearest point of the axis is still found
    d = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    u = bh.along_axis((0, 0, 0), 2, (0, 4, 0), d)
    t = np.linspace(-50, 50, 200001)
    ray = np.array([0, 4, 0]) + t[:, None] * d
    nearest = t[np.argmin(np.hypot(ray[:, 0], ray[:, 1]))]
    assert abs(u - (np.array([0, 4, 0]) + nearest * d)[2]) < 1e-3
    # looking along the axis: no answer
    assert bh.along_axis((0, 0, 0), 2, (5, 3, 7), (0, 0, 1)) is None
    point = bh.on_plane((0, 0, 0), (1, 0, 0), (4, 9, 9), (1, 0, 0))
    assert np.allclose(point, (4, 0, 0))
    assert bh.on_plane((0, 0, 0), (0, 1, 0), (4, 9, 9), (1, 0, 0)) is None


def test_move_face_and_region():
    extent = (0.0, 40.0)
    assert bh.move_face(RANGES, "x", 1, 20.0, extent, 0.25)["x"] == (2, 20)
    assert bh.move_face(RANGES, "x", 1, 99.0, extent, 0.25)["x"] == (2, 40)
    assert bh.move_face(RANGES, "x", 1, -5.0, extent, 0.25)["x"] == \
        (2, 2.25)
    assert bh.move_face(RANGES, "x", 0, 50.0, extent, 0.25)["x"] == \
        (11.75, 12)
    moved = bh.move_region(RANGES, {"z": 5.0, "x": -100.0},
                           {a: extent for a in AXES})
    assert moved["z"] == (15, 35) and moved["x"] == (0, 10)
    assert moved["y"] == RANGES["y"]


def test_symmetric_snap_and_pick_region():
    extent = (0.0, 40.0)
    # Shift: the high-x face to 15 (3 out), the low one 3 out too
    sym = bh.resize_symmetric(RANGES, "x", 1, 15.0, (-10.0, 40.0), 0.25)
    assert sym["x"] == (-1.0, 15.0), sym
    # ... both stop together at the extent (low can only go 2 µm out)
    sym = bh.resize_symmetric(RANGES, "x", 1, 15.0, extent, 0.25)
    assert sym["x"] == (0.0, 14.0), sym
    # ... and inward no narrower than the smallest size, around the centre
    sym = bh.resize_symmetric(RANGES, "x", 0, 50.0, extent, 1.0)
    assert sym["x"] == (6.5, 7.5), sym
    # Ctrl: about 1 % of the largest side, rounded 1-2-5
    assert bh.snap_step({"z": (0, 800), "y": (0, 1000), "x": (0, 1000)}) \
        == 10.0
    assert bh.snap_step({"z": (0, 3000.0)}) == 50.0
    assert bh.snap_step({"z": (0, 150.0)}) == 2.0
    assert bh.snapped(653.0, 10.0) == 650.0
    assert bh.snapped(-7.4, 5.0) == -5.0
    # the wireframe: 12 edges; a ray through the middle of one picks it
    edges = bh.box_edges(RANGES, AXES)
    assert len(edges) == 12
    distance, _ = bh.ray_to_segment((0, 0, 0), (0, 0, 10), (5, 3, 4),
                                    (1, 0, 0))
    assert abs(distance - 3) < 1e-9
    distance, _ = bh.ray_to_segment((0, 0, 0), (0, 0, 10), (5, 0, 14),
                                    (1, 0, 0))       # beyond its end
    assert abs(distance - 4) < 1e-9
    other = {"z": (10.0, 30.0), "y": (20.0, 30.0), "x": (2.0, 12.0)}
    regions = {0: RANGES, 1: other}
    # looking along z at the edge (y 30, x 7) of region 1 only
    assert bh.pick_region(regions, (0, 30, 7), (1, 0, 0), 2.0, AXES) == 1
    # at (y 15, x 7): region 0's edge (y 15)
    assert bh.pick_region(regions, (0, 15, 7), (1, 0, 0), 2.0, AXES) == 0
    # far from every edge (on screen): none
    assert bh.pick_region(regions, (0, 50, 50), (1, 0, 0), 2.0, AXES) is None


def test_shifted_outlines():
    outlines = {("y", "x"): {20.0: [(6.0, 3.0), (14.0, 11.0), (6.0, 11.0)],
                             None: [(1.0, 1.0), (2.0, 2.0), (1.0, 2.0)]},
                ("z", "x"): {(5.0, 15.0): [(10.0, 3.0), (30.0, 3.0),
                                           (30.0, 9.0)]}}
    moved = bh.shifted_outlines(outlines, {"z": 2.0, "y": -1.0, "x": 0.5},
                                AXES)
    # a slice in XY moves with z; its points with y and x
    assert list(moved[("y", "x")]) == [22.0, None]
    assert moved[("y", "x")][22.0][0] == (5.0, 3.5)
    assert moved[("y", "x")][None][0] == (0.0, 1.5)
    # a silhouette in XZ: its depth range moves with y, its points z, x
    assert list(moved[("z", "x")]) == [(4.0, 14.0)]
    assert moved[("z", "x")][(4.0, 14.0)][1] == (32.0, 3.5)
    assert outlines[("y", "x")][20.0][0] == (6.0, 3.0)     # unchanged


def test_drag_in_window():
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump()
            ortho = panel.ortho
            ortho.set_3d(True)
            _pump(0.5)
            i = panel._new_box(dict(RANGES))
            square = [(6.0, 3.0), (6.0, 11.0), (14.0, 11.0), (14.0, 3.0)]
            panel.boxes[i].outlines = vc.normalize_outlines(
                {("y", "x"): square})
            panel._refresh(select=i)
            _pump()
            view = ortho.views["3D"]
            assert view.handles is not None and view.handles.visible
            spatial = view.order[-3:]
            assert spatial == AXES, spatial
            # the dots are HANDLE_PIXELS wide on screen, in µm at the 3D
            # zoom (not huge: napari counts a point's size in the scene's
            # extent, and huge ones made the home button zoom out to
            # nothing); and they follow the zoom
            zoom = ortho_camera(view).zoom
            assert np.allclose(view.handles.size,
                               ov_module().HANDLE_PIXELS / zoom)
            ortho_camera(view).zoom = 2 * zoom
            _pump(0.1)
            assert np.allclose(view.handles.size,
                               ov_module().HANDLE_PIXELS / (2 * zoom))
            shown = [view.order.index(a) for a in spatial]
            start = dict(ortho.box_ranges[i])
            points = bh.handle_points(start, spatial)
            assert np.allclose(view.handles.data, list(points.values()))
            cam = ortho_camera(view)
            ndim = len(view.order)

            def event(kind, origin, direction=(0.0, 1.0, 0.0)):
                position = np.zeros(ndim)
                position[shown] = origin
                looking = np.zeros(ndim)
                looking[shown] = direction
                return SimpleNamespace(
                    type=kind, button=1, modifiers=(), dims_displayed=shown,
                    position=position, view_direction=looking,
                    camera_zoom=2.0)

            def drag_dot(key, step):
                """Press on a dot (looking along y), move by step (µm per
                axis z, y, x), release."""
                dot = bh.handle_points(ortho.box_ranges[i], spatial)[key]
                ev = event("mouse_press", dot - (0, 100, 0))
                drag = ortho._handle_drag(view.model, ev)
                next(drag)
                assert not cam.mouse_pan, "the camera turns while dragging"
                new = ev.position.copy()
                new[shown] += step
                ev.type, ev.position = "mouse_move", new
                next(drag)
                ev.type = "mouse_release"
                try:
                    next(drag)
                except StopIteration:
                    pass
                _pump()
                assert cam.mouse_pan and cam.mouse_zoom

            box = panel.boxes[i]
            # the high-z face 2 µm up: the face goes where the mouse is,
            # the others keep their values exactly, and the crop grows
            drag_dot(("z", 1), (2.0, 0.0, 0.0))
            assert box.ranges["z"] == (RANGES["z"][0], start["z"][1] + 2), \
                box.ranges
            assert box.ranges["y"] == RANGES["y"]
            assert box.ranges["x"] == RANGES["x"]
            assert abs(ortho.box_ranges[i]["z"][1] - (start["z"][1] + 2)) \
                <= 2.0, ortho.box_ranges[i]      # (snapped to the z grid)
            assert ortho.box_ranges[i]["z"][1] > start["z"][1]
            # the high-x face 3 µm out: beyond the square, which still
            # shapes the region there, so the crop stays at the square
            drag_dot(("x", 1), (0.0, 0.0, 3.0))
            assert box.ranges["x"] == (RANGES["x"][0], start["x"][1] + 3), \
                box.ranges
            assert ortho.box_ranges[i]["x"] == start["x"], ortho.box_ranges[i]
            assert box.outlines, "the polygons stay"
            # ... and 2 µm into the square: the crop follows the face
            drag_dot(("x", 1), (0.0, 0.0, -2.0))
            assert abs(ortho.box_ranges[i]["x"][1] - (start["x"][1] - 2)) \
                <= 0.5, ortho.box_ranges[i]

            # the centre: +2 µm in z, -1 µm in x (across the screen); the
            # polygons move with the region
            before = dict(box.ranges)
            points_before = next(iter(box.outlines[("y", "x")].values()))
            drag_dot(bh.CENTRE, (2.0, 0.0, -1.0))
            for axis, step in (("z", 2.0), ("y", 0.0), ("x", -1.0)):
                assert np.allclose(box.ranges[axis],
                                   np.add(before[axis], step)), axis
            points_after = next(iter(box.outlines[("y", "x")].values()))
            assert np.allclose(np.subtract(points_after, points_before),
                               (0.0, -1.0)), points_after
            centre = bh.handle_points(ortho.box_ranges[i],
                                      spatial)[bh.CENTRE]

            # a press away from the dots: left to the camera (the handler
            # only waits to see whether it is a click on a frame)
            ev = event("mouse_press", centre + (40, 0, 0))
            ev.pos = (0.0, 0.0)
            drag = ortho._handle_drag(view.model, ev)
            next(drag)
            assert cam.mouse_pan, "the camera must stay free"
            ev.type = "mouse_release"
            try:
                next(drag)
            except StopIteration:
                pass
            assert panel._selected() == i
        finally:
            viewer.close()
            _pump()


# --- Shift, Ctrl, clicks, Up/Down, zoom, undo (0.18.0, the user's choices) --

def _window_3d(tmp):
    """A hidden window with the test image, its 3D view on."""
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    make_test_image(Path(tmp) / "src.zarr")
    viewer, panel = crop_gui.open_window(show=False)
    panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
    _pump()
    panel.ortho.set_3d(True)
    _pump(0.5)
    return viewer, panel


class _Mouse:
    """Simulated mouse events for the 3D view's handler, looking along y
    (napari passes the same event object on, changing its contents)."""

    def __init__(self, ortho):
        self.ortho = ortho
        self.view = ortho.views["3D"]
        self.spatial = self.view.order[-3:]
        self.shown = [self.view.order.index(a) for a in self.spatial]
        self.ndim = len(self.view.order)

    def event(self, kind, origin, modifiers=()):
        position = np.zeros(self.ndim)
        position[self.shown] = origin
        looking = np.zeros(self.ndim)
        looking[self.shown] = (0.0, 1.0, 0.0)
        return SimpleNamespace(
            type=kind, button=1, modifiers=tuple(modifiers),
            dims_displayed=self.shown, position=position,
            view_direction=looking, camera_zoom=2.0, pos=(100.0, 100.0))

    def press(self, point, modifiers=()):
        """Press at a point (µm, z y x); the handler, or None if it left
        the press to the camera."""
        ev = self.event("mouse_press", np.asarray(point) - (0, 100, 0),
                        modifiers)
        drag = self.ortho._handle_drag(self.view.model, ev)
        try:
            next(drag)
        except StopIteration:
            return None
        return drag, ev

    def move(self, pressed, step, pixels=(40.0, 0.0)):
        drag, ev = pressed
        new = ev.position.copy()
        new[self.shown] += step
        ev.type, ev.position = "mouse_move", new
        ev.pos = (ev.pos[0] + pixels[0], ev.pos[1] + pixels[1])
        next(drag)

    def release(self, pressed):
        drag, ev = pressed
        ev.type = "mouse_release"
        try:
            next(drag)
        except StopIteration:
            pass
        _pump()

    def drag(self, point, step, modifiers=()):
        pressed = self.press(point, modifiers)
        assert pressed is not None, "the press was not taken"
        self.move(pressed, step)
        self.release(pressed)


def test_shift_ctrl_clip_click_nudge_zoom_in_window():
    with tempfile.TemporaryDirectory() as tmp:
        viewer, panel = _window_3d(tmp)
        try:
            ortho = panel.ortho
            mouse = _Mouse(ortho)
            spatial = mouse.spatial
            i = panel._new_box(dict(RANGES))
            other = panel._new_box({"z": (40.0, 70.0), "y": (20.0, 35.0),
                                    "x": (15.0, 30.0)})
            panel._refresh(select=i)
            _pump()
            box = panel.boxes[i]

            def dot(key):
                return bh.handle_points(ortho.box_ranges[i], spatial)[key]

            # Shift: the high-x face 2 µm out, the low one as far the
            # other way (the centre stays)
            drawn, held = dict(ortho.box_ranges[i]), dict(box.ranges)
            mouse.drag(dot(("x", 1)), (0.0, 0.0, 2.0), ["Shift"])
            grow = drawn["x"][1] + 2 - held["x"][1]
            assert np.allclose(box.ranges["x"], (held["x"][0] - grow,
                                                 held["x"][1] + grow)), \
                box.ranges
            assert box.ranges["z"] == held["z"]
            # Ctrl: the high-z face lands on a whole step
            step = bh.snap_step({a: ortho.extent(a) for a in spatial})
            mouse.drag(dot(("z", 1)), (3.3, 0.0, 0.0), ["Control"])
            high = box.ranges["z"][1]
            assert abs(high / step - round(high / step)) < 1e-9, (high, step)
            assert box.ranges["z"][0] == held["z"][0]

            # clipping follows the drag while it goes on
            panel.clip_check.setChecked(True)
            _pump(0.3)
            pressed = mouse.press(dot(("y", 1)))
            mouse.move(pressed, (0.0, 0.0, 0.0))     # (looking along y:
            # a y face can't follow; move the x face instead)
            mouse.release(pressed)
            pressed = mouse.press(dot(("x", 1)))
            target = dot(("x", 1))[2] + 1.5
            mouse.move(pressed, (0.0, 0.0, 1.5))
            planes = ortho.views["3D"].channel_layers[0] \
                .experimental_clipping_planes
            positions = [tuple(plane.position) for plane in planes]
            assert any(abs(p[2] - target) < 1e-6 for p in positions), \
                (positions, target)
            mouse.release(pressed)
            panel.clip_check.setChecked(False)
            _pump(0.3)

            # a click on the other region's frame selects it; a press that
            # moves (the camera turning) does not
            corner = (40.0, 50.0, 22.0)      # its edge along y, z 40, x 22
            pressed = mouse.press(corner)
            assert pressed is not None
            mouse.move(pressed, (0.0, 0.0, 0.0), pixels=(30.0, 0.0))
            mouse.release(pressed)
            assert panel._selected() == i
            pressed = mouse.press(corner)
            mouse.release(pressed)
            assert panel._selected() == other, panel._selected()
            panel._refresh(select=i)
            _pump()

            # napari's key handling needs a name for every bound function
            # (a functools.partial broke every key in the 3D view)
            for function in mouse.view.model.keymap.values():
                assert hasattr(function, "__name__"), function
            # Up/Down: the face last pressed moves a voxel out / 10 in (the
            # high-x face: the Shift drag took the low one to the image edge)
            voxel = ortho._voxel_size("x")
            pressed = mouse.press(dot(("x", 1)))     # a click on its dot
            mouse.release(pressed)
            high = box.ranges["x"][1]
            ortho._nudge(1)
            _pump()
            assert abs(box.ranges["x"][1] - (high + voxel)) < 1e-9, \
                (box.ranges["x"], high, voxel)
            ortho._nudge(-10)
            _pump()
            assert abs(box.ranges["x"][1] - (high - 9 * voxel)) < 1e-9

            # Zoom to region: the 3D view is centred on the region
            ortho.zoom_to_selected()
            _pump()
            centre = [float(np.mean(ortho.box_ranges[i][a]))
                      for a in spatial]
            assert np.allclose(ortho_camera(mouse.view).center[-3:],
                               centre), ortho_camera(mouse.view).center
        finally:
            viewer.close()
            _pump()


def test_seeds_in_3d():
    """(0.18.0, the user's wish) Seeds are shown in the 3D view too, at
    their 3D places in the colours of 2D, and don't enlarge the scene."""
    with tempfile.TemporaryDirectory() as tmp:
        viewer, panel = _window_3d(tmp)
        try:
            ortho = panel.ortho
            view = ortho.views["3D"]
            seeds = [{"z": 20.0, "y": 10.0, "x": 8.0},
                     {"z": 50.0, "y": 25.0, "x": 20.0}]
            ortho.seeds = seeds
            ortho._draw_seeds()
            _pump()
            assert view.seeds is not None and view.seeds.visible
            assert np.allclose(view.seeds.data,
                               [[s[a] for a in AXES] for s in seeds])
            two_d = ortho.views["XY"].seeds
            assert np.allclose(view.seeds.face_color, two_d.face_color)
            # SEED_SIZE pixels wide on screen, in µm at the 3D zoom
            zoom = ortho_camera(view).zoom
            assert np.allclose(view.seeds.size, ov_module().SEED_SIZE / zoom)
            ortho.seeds = []
            ortho._draw_seeds()
            assert not view.seeds.visible
            # a new 3D level keeps them
            ortho.seeds = seeds
            ortho._draw_seeds()
            ortho.set_3d(True, ortho.region.n_levels - 1)
            _pump(0.5)
            assert view.seeds in view.model.layers and view.seeds.visible
        finally:
            viewer.close()
            _pump()


def test_undo_redo_in_window():
    from volume_cropper import crop_gui
    merge = crop_gui.UNDO_MERGE_S
    with tempfile.TemporaryDirectory() as tmp:
        viewer, panel = _window_3d(tmp)
        try:
            crop_gui.UNDO_MERGE_S = 0.0         # every change a step
            ortho = panel.ortho
            assert not panel.undo_button.isEnabled()
            a = panel._new_box(dict(RANGES))
            panel._refresh(select=a)
            b = panel._new_box({"z": (40.0, 70.0), "y": (20.0, 35.0),
                                "x": (15.0, 30.0)})
            panel._refresh(select=b)
            panel._box_ranges_dragged(a, {"x": (3.0, 9.0)})
            panel._refresh(select=b)
            first, second = panel.boxes            # (the region objects)
            panel._remove_box()                     # b
            names = lambda: [box.name for box in panel.boxes]  # noqa: E731
            assert names() == ["crop01"] and panel.undo_button.isEnabled()
            assert len(panel._undo) == 4, len(panel._undo)

            panel._undo_regions()                   # b back
            assert names() == ["crop01", "crop02"]
            # the same objects (what refers to them, e.g. the
            # Autocropper's seeds, keeps working), b's original too
            assert panel.boxes[0] is first and panel.boxes[1] is second
            panel._undo_regions()                   # a's x back
            assert panel.boxes[0].ranges["x"] == RANGES["x"]
            assert panel.redo_button.isEnabled()
            panel._redo_regions()                   # a's x changed again
            assert panel.boxes[0].ranges["x"] == (3.0, 9.0)
            assert panel.boxes[0] is first
            for _ in range(5):                      # (beyond the start)
                panel._undo_regions()
            assert panel.boxes == [] and not panel.undo_button.isEnabled()
            for _ in range(4):
                panel._redo_regions()
            assert names() == ["crop01"]
            assert panel.boxes[0].ranges["x"] == (3.0, 9.0)
            assert not panel.redo_button.isEnabled()
            # a new change after undo: redo is gone
            panel._undo_regions()
            panel._box_ranges_dragged(0, {"z": (12.0, 28.0)})
            assert not panel._redo

            # a drag in 3D, however many moves, is one step
            panel._refresh(select=0)
            _pump()
            before = len(panel._undo)
            mouse = _Mouse(ortho)
            dot = bh.handle_points(ortho.box_ranges[0],
                                   mouse.spatial)[("x", 1)]
            pressed = mouse.press(dot)
            for _ in range(4):
                mouse.move(pressed, (0.0, 0.0, 0.5))
                time.sleep(0.2)                    # (applied as it goes)
            mouse.release(pressed)
            assert len(panel._undo) == before + 1, (len(panel._undo),
                                                     before)
            x_after = panel.boxes[0].ranges["x"]
            panel._undo_regions()
            assert panel.boxes[0].ranges["x"] != x_after

            # changes in quick succession are one step
            crop_gui.UNDO_MERGE_S = 10.0
            before = len(panel._undo)
            panel._box_ranges_dragged(0, {"y": (6.0, 14.0)})
            panel._box_ranges_dragged(0, {"y": (7.0, 13.0)})
            assert len(panel._undo) == before + 1

            # another image: a new history (the question about the regions
            # drawn so far answered "Discard them")
            panel._may_discard_boxes = lambda action: True
            from volume_cropper.load_dialog import LoadChoice
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump()
            assert not panel._undo and not panel._redo
            assert not panel.undo_button.isEnabled()
        finally:
            crop_gui.UNDO_MERGE_S = merge
            viewer.close()
            _pump()


def test_new_region_with_3d_view():
    """(The user's wish, 2026-11-12) With the 3D view on, New region starts
    as the middle half of the image (its dots inside the image); drawing
    into it first makes it the whole image again, as before; a region
    shaped by its dots stays after Finish region, an untouched one goes."""
    from volume_cropper import crop_gui
    from volume_cropper.load_dialog import LoadChoice
    with tempfile.TemporaryDirectory() as tmp:
        make_test_image(Path(tmp) / "src.zarr")
        viewer, panel = crop_gui.open_window(show=False)
        try:
            panel.open_image(Path(tmp) / "src.zarr", choice=LoadChoice())
            _pump()
            ortho = panel.ortho
            full = {a: ortho.extent(a) for a in AXES}
            # 3D off: the whole image, as before
            panel._new_region()
            assert panel.boxes[-1].ranges == full
            panel._finish_region()
            assert panel.boxes == []          # untouched: removed
            ortho.set_3d(True)
            _pump(0.5)
            panel._new_region()
            box = panel.boxes[-1]
            for a, (low, high) in full.items():
                quarter = (high - low) / 4
                assert box.ranges[a] == (low + quarter, high - quarter), a
            assert ortho.views["3D"].handles.visible
            # a polygon drawn into it: the whole depth again (one polygon
            # reaches through the region)
            square = [(6.0, 3.0), (6.0, 11.0), (14.0, 11.0), (14.0, 3.0)]
            panel._box_added({"y": (6.0, 14.0), "x": (3.0, 11.0)},
                             ("y", "x"), square)
            _pump()
            assert box.ranges["z"] == full["z"], box.ranges
            assert box.outlines
            panel._finish_region()
            assert box in panel.boxes
            # shaped by its dots, then Finish region: it stays
            panel._new_region()
            fresh = panel.boxes[-1]
            panel._box_ranges_dragged(len(panel.boxes) - 1,
                                      {"x": (10.0, 20.0)})
            panel._finish_region()
            assert fresh in panel.boxes and fresh.ranges["x"] == (10.0, 20.0)
        finally:
            viewer.close()
            _pump()


def ov_module():
    from volume_cropper import ortho_views
    return ortho_views


def ortho_camera(view):
    from volume_cropper import ortho_views
    return ortho_views.camera(view.model)


if __name__ == "__main__":
    import run_all_tests
    raise SystemExit(run_all_tests.run(["handles"]))
