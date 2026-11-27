"""Handles to change a crop region in the 3D view (0.18.0): the geometry,
no window needed.

The user asked on 2026-11-12 for a way to change a crop region in the 3D
view, and chose a dot in the middle of each face plus one in the centre.
Dragging a face's dot moves that face along its own axis (the opposite
face stays); dragging the centre dot moves the whole region, keeping its
size. OrthoViews draws the dots and passes the mouse to these functions.

napari gives each mouse event in 3D the point clicked (in µm, on a plane
through the camera's centre) and the direction of the line of sight
through it, so each event is a ray: point + t * direction.
- A face follows the point of its axis line that lies closest to the
  mouse's ray (closest points of two lines). Looking straight along that
  axis, the two lines are parallel and the face can't follow: it stays.
- The centre follows the ray's crossing with the plane, facing the
  camera, through the centre where the drag started: the region moves
  across the screen, as the mouse does.
"""

from __future__ import annotations

import numpy as np

CENTRE = "centre"
GRAB_PIXELS = 12        # a press this close to a dot (on screen) drags it
PARALLEL = 0.05         # |sin| of the angle between axis and line of sight
                        # below which a face can't follow the mouse


def handle_points(ranges: dict, axes) -> dict:
    """The dots of a region: {key: point}, with points in the order of
    axes (the 3D view's spatial axes). key: CENTRE, or (axis, 0) for the
    face at the low end of an axis and (axis, 1) for the high end."""
    centre = np.array([(ranges[a][0] + ranges[a][1]) / 2 for a in axes])
    points = {CENTRE: centre}
    for i, axis in enumerate(axes):
        for side in (0, 1):
            point = centre.copy()
            point[i] = ranges[axis][side]
            points[(axis, side)] = point
    return points


def pick_handle(points: dict, origin, direction, zoom: float,
                grab_pixels: float = GRAB_PIXELS):
    """The dot under the mouse: the one nearest to the ray on screen
    (distance from the ray in µm times the zoom, pixels per µm), if within
    grab_pixels; else None. On a tie (dots in line with the eye, e.g.
    the centre and two faces seen along an axis) the centre wins, then
    the dot nearer the eye."""
    origin = np.asarray(origin, float)
    d = _unit(direction)
    best, best_score = None, None
    for key, point in points.items():
        v = np.asarray(point, float) - origin
        along = float(v @ d)
        pixels = float(np.linalg.norm(v - along * d)) * zoom
        if pixels > grab_pixels:
            continue
        score = (round(pixels / 2), 0 if key == CENTRE else 1, along)
        if best_score is None or score < best_score:
            best, best_score = key, score
    return best


def along_axis(start, axis_index: int, origin, direction):
    """How far along a spatial axis (in µm, from the point start) lies the
    point of that axis line nearest to the ray; None if the line of sight
    runs (almost) along the axis."""
    e = np.zeros(len(start))
    e[axis_index] = 1.0
    d = _unit(direction)
    b = float(e @ d)
    if 1 - b * b < PARALLEL ** 2:         # sin² of the angle
        return None
    w = np.asarray(start, float) - np.asarray(origin, float)
    # closest points of the lines start + u e and origin + t d (|e|=|d|=1)
    return float((b * (d @ w) - (e @ w)) / (1 - b * b))


def on_plane(origin, direction, plane_point, normal):
    """Where the ray crosses the plane through plane_point with this
    normal; None if it runs along the plane."""
    d = _unit(direction)
    n = _unit(normal)
    across = float(d @ n)
    if abs(across) < 1e-9:
        return None
    t = float((np.asarray(plane_point, float) - origin) @ n) / across
    return np.asarray(origin, float) + t * d


def move_face(ranges: dict, axis: str, side: int, value: float,
              extent: tuple, smallest: float) -> dict:
    """The region with one face moved to value: within the extent shown,
    and at least `smallest` (e.g. one voxel) away from the other face."""
    low, high = ranges[axis]
    if side == 0:
        low = min(max(value, extent[0]), high - smallest)
    else:
        high = max(min(value, extent[1]), low + smallest)
    return {**ranges, axis: (float(low), float(high))}


def move_region(ranges: dict, shift: dict, extents: dict) -> dict:
    """The region moved by shift ({axis: µm}), keeping its size: along
    each axis no further than the extent shown allows."""
    moved = dict(ranges)
    for axis, step in shift.items():
        low, high = ranges[axis]
        step = min(max(step, extents[axis][0] - low), extents[axis][1] - high)
        moved[axis] = (float(low + step), float(high + step))
    return moved


def resize_symmetric(ranges: dict, axis: str, side: int, value: float,
                     extent: tuple, smallest: float) -> dict:
    """(Shift-drag) The face moved to value and the opposite face as far
    the other way, so the centre stays: within the extent shown (both
    faces stop together) and at least `smallest` wide."""
    low, high = ranges[axis]
    grow = (value - high) if side == 1 else (low - value)  # > 0: larger
    grow = min(grow, extent[1] - high, low - extent[0])
    grow = max(grow, (smallest - (high - low)) / 2)
    return {**ranges, axis: (float(low - grow), float(high + grow))}


def snap_step(extents: dict) -> float:
    """(Ctrl-drag) A round step of about 1 % of the image's largest side
    shown: 1, 2 or 5 times a power of ten, in µm (e.g. 10 µm for 1 mm)."""
    span = max(high - low for low, high in extents.values())
    if not span > 0:
        return 1.0
    wanted = span / 100
    power = 10.0 ** np.floor(np.log10(wanted))
    return float(next(m * power for m in (1, 2, 5, 10)
                      if m * power >= wanted * (1 - 1e-9)))


def snapped(value: float, step: float) -> float:
    """value rounded to the nearest multiple of step."""
    return float(round(value / step) * step)


def box_edges(ranges: dict, axes) -> list:
    """The 12 edges of a region's wireframe, as (start, end) points in the
    order of axes."""
    lows = [ranges[a][0] for a in axes]
    highs = [ranges[a][1] for a in axes]
    edges = []
    for i in range(3):                     # edges along axis i
        j, k = [n for n in range(3) if n != i]
        for vj in (lows[j], highs[j]):
            for vk in (lows[k], highs[k]):
                start = np.zeros(3)
                start[i], start[j], start[k] = lows[i], vj, vk
                end = start.copy()
                end[i] = highs[i]
                edges.append((start, end))
    return edges


def ray_to_segment(start, end, origin, direction) -> tuple:
    """The distance (µm) between a ray and a line segment at their
    closest points, and how far along the ray (t) that is."""
    p0 = np.asarray(start, float)
    seg = np.asarray(end, float) - p0
    d = _unit(direction)
    o = np.asarray(origin, float)
    length2 = float(seg @ seg)
    if length2 == 0:
        v = p0 - o
        t = float(v @ d)
        return float(np.linalg.norm(v - t * d)), t
    # closest points of the two lines, with the segment's parameter kept
    # within 0..1 (then the ray's recomputed for that point)
    w = o - p0
    b = float(seg @ d)
    denom = length2 - b * b
    s = 0.0 if abs(denom) < 1e-12 else (
        (float(seg @ w) - b * float(d @ w)) / denom)
    s = min(max(s, 0.0), 1.0)
    point = p0 + s * seg
    t = float((point - o) @ d)
    return float(np.linalg.norm(point - (o + t * d))), t


def pick_region(regions: dict, origin, direction, zoom: float, axes,
                grab_pixels: float = GRAB_PIXELS):
    """(A click in 3D) The region whose wireframe passes nearest to the
    mouse's ray on screen, within grab_pixels; on a tie the one nearer the
    eye. regions: {index: ranges}. None if no wireframe is that close."""
    best, best_score = None, None
    for i, ranges in regions.items():
        for start, end in box_edges(ranges, axes):
            distance, along = ray_to_segment(start, end, origin, direction)
            pixels = distance * zoom
            if pixels > grab_pixels:
                continue
            score = (round(pixels / 2), along)
            if best_score is None or score < best_score:
                best, best_score = i, score
    return best


def shifted_outlines(outlines: dict, shift: dict, spatial) -> dict:
    """A region's polygons (CropBox.outlines, standard form) moved by
    shift ({axis: µm}): their points, and the slice or depth range they
    lie at along the third axis. spatial: the image's spatial axes."""
    moved = {}
    for plane, keys in outlines.items():
        a, b = plane
        depth = next(axis for axis in spatial if axis not in plane)
        sa, sb, sd = (shift.get(a, 0.0), shift.get(b, 0.0),
                      shift.get(depth, 0.0))
        moved[plane] = {}
        for key, points in keys.items():
            if key is None:                       # through the whole region
                new_key = None
            elif isinstance(key, tuple):          # a silhouette's depth range
                new_key = (key[0] + sd, key[1] + sd)
            else:                                 # a slice
                new_key = key + sd
            moved[plane][new_key] = [(float(p) + sa, float(q) + sb)
                                     for p, q in points]
    return moved


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
