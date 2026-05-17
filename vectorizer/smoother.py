"""Post-trace curve smoothing: RDP simplification + cubic-Bézier fitting + C1 joins."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from svgpathtools import (
    CubicBezier,
    Line,
    Path,
    QuadraticBezier,
    parse_path,
)

from .arcs import ArcFitConfig, fit_arcs_in_path
from .tracer import TracedLayer
from .utils.svg_utils import round_path_coords


@dataclass
class SmoothConfig:
    rdp_epsilon: float = 0.5
    spline_smoothing: float = 1.0       # scipy splprep s= factor
    enforce_c1: bool = True             # collinear handles at anchors
    coordinate_precision: int = 2
    resample_long_polylines: bool = True
    long_polyline_min_pts: int = 8      # only fit splines to runs of ≥ N straight segments
    # Post-smooth arc fitting: replace runs of cubic Béziers that approximate
    # circular arcs with true SVG ``A`` commands (mathematically perfect curvature).
    arc_fit: bool = True
    arc_fit_max_residual: float = 1.5


# --------------------------- Ramer-Douglas-Peucker ---------------------------


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Iterative RDP. Returns a subset of input points preserving ordering."""
    if points.shape[0] < 3:
        return points
    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, points.shape[0] - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[start:end + 1]
        p_start = segment[0]
        p_end = segment[-1]
        line_vec = p_end - p_start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-9:
            d = np.linalg.norm(segment - p_start, axis=1)
        else:
            # Perpendicular distance from each interior point to the chord.
            normal = np.array([-line_vec[1], line_vec[0]]) / line_len
            d = np.abs((segment - p_start) @ normal)
        idx_local = int(np.argmax(d[1:-1])) + 1 if segment.shape[0] > 2 else 0
        if d[idx_local] > epsilon:
            keep[start + idx_local] = True
            stack.append((start, start + idx_local))
            stack.append((start + idx_local, end))
    return points[keep]


# --------------------------- Bézier helpers ---------------------------


def _segment_polyline(segments: list) -> list[np.ndarray]:
    """Group consecutive Line segments into polylines for spline-fitting."""
    runs: list[list[Line]] = []
    current: list[Line] = []
    for seg in segments:
        if isinstance(seg, Line):
            current.append(seg)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    polylines: list[np.ndarray] = []
    for run in runs:
        pts = [(run[0].start.real, run[0].start.imag)]
        for line in run:
            pts.append((line.end.real, line.end.imag))
        polylines.append(np.array(pts, dtype=np.float64))
    return polylines


def _fit_cubic_beziers(points: np.ndarray, smoothing: float) -> list[CubicBezier]:
    """Fit a cubic B-spline to a polyline and emit a series of cubic Béziers.

    Uses scipy ``splprep`` with degree=3 (matches Bézier degree) and converts
    each B-spline knot span into a Bézier via De Boor / matrix conversion.
    """
    if points.shape[0] < 4:
        return []

    # Remove consecutive duplicates that break splprep.
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    keep = np.concatenate(([True], (dx ** 2 + dy ** 2) > 1e-6))
    points = points[keep]
    if points.shape[0] < 4:
        return []

    x = points[:, 0]
    y = points[:, 1]
    closed = bool(
        np.linalg.norm(points[0] - points[-1]) < 0.5 and points.shape[0] >= 5
    )

    try:
        tck, _u = splprep([x, y], s=float(smoothing), k=3, per=1 if closed else 0)
    except Exception:
        return []

    # Sample the spline densely, then approximate each chord with a cubic Bézier.
    n_samples = max(16, points.shape[0] * 4)
    u_new = np.linspace(0, 1, n_samples)
    xs, ys = splev(u_new, tck)
    sampled = np.column_stack([xs, ys])

    return _sampled_to_cubic_beziers(sampled)


def _sampled_to_cubic_beziers(points: np.ndarray) -> list[CubicBezier]:
    """Group dense samples into cubic Bézier segments using endpoint tangents."""
    if points.shape[0] < 4:
        return []
    beziers: list[CubicBezier] = []
    step = 3
    # Walk the samples in chunks of 3, fitting a cubic per chunk so that
    # endpoint tangents match the local sample direction.
    for i in range(0, points.shape[0] - step, step):
        p0 = points[i]
        p3 = points[min(i + step, points.shape[0] - 1)]
        # Estimate tangent at the start from a backward difference.
        prev = points[i - 1] if i > 0 else points[i]
        nxt = points[min(i + step + 1, points.shape[0] - 1)]
        t0 = (points[i + 1] - prev) * 0.5
        t1 = (p3 - points[min(i + step - 1, points.shape[0] - 1)]) * 0.5
        # Bézier handle length ≈ chord / 3 for a smooth fit.
        chord = np.linalg.norm(p3 - p0)
        if chord < 1e-6:
            continue
        scale = chord / 3.0
        norm0 = np.linalg.norm(t0)
        norm1 = np.linalg.norm(t1)
        t0 = t0 / norm0 * scale if norm0 > 1e-9 else (p3 - p0) / 3.0
        t1 = t1 / norm1 * scale if norm1 > 1e-9 else (p3 - p0) / 3.0
        c1 = p0 + t0
        c2 = p3 - t1
        beziers.append(
            CubicBezier(
                complex(p0[0], p0[1]),
                complex(c1[0], c1[1]),
                complex(c2[0], c2[1]),
                complex(p3[0], p3[1]),
            )
        )
    return beziers


# --------------------------- C1 continuity ---------------------------


def _enforce_c1_continuity(path: Path) -> Path:
    """Make control handles collinear across shared anchors (smooth joins)."""
    if len(path) < 2:
        return path

    segs = list(path)
    for i in range(1, len(segs)):
        prev = segs[i - 1]
        curr = segs[i]
        if not isinstance(prev, CubicBezier) or not isinstance(curr, CubicBezier):
            continue
        # Shared anchor: prev.end == curr.start.
        if abs(prev.end - curr.start) > 1e-3:
            continue
        handle_in = prev.control2
        handle_out = curr.control1
        anchor = curr.start

        dir_in = anchor - handle_in
        len_in = abs(dir_in)
        if len_in < 1e-9:
            continue
        unit = dir_in / len_in

        # Project the outgoing handle onto the colinear direction.
        len_out = abs(handle_out - anchor)
        new_out = anchor + unit * len_out
        segs[i] = CubicBezier(curr.start, new_out, curr.control2, curr.end)

    # Handle a closed-path wrap-around join.
    if abs(segs[-1].end - segs[0].start) < 1e-3 and isinstance(segs[-1], CubicBezier) and isinstance(segs[0], CubicBezier):
        prev = segs[-1]
        curr = segs[0]
        anchor = curr.start
        dir_in = anchor - prev.control2
        if abs(dir_in) > 1e-9:
            unit = dir_in / abs(dir_in)
            len_out = abs(curr.control1 - anchor)
            new_out = anchor + unit * len_out
            segs[0] = CubicBezier(curr.start, new_out, curr.control2, curr.end)

    return Path(*segs)


# --------------------------- Public API ---------------------------


def smooth_path_string(d: str, config: SmoothConfig | None = None) -> str:
    """Smooth a single SVG path 'd' string."""
    cfg = config or SmoothConfig()
    try:
        path = parse_path(d)
    except Exception:
        return d
    if len(path) == 0:
        return d

    segs = list(path)

    # Convert quadratic Béziers to cubic for consistency, then RDP-simplify
    # long polyline runs before spline-fitting them.
    upgraded: list = []
    for seg in segs:
        if isinstance(seg, QuadraticBezier):
            c1 = seg.start + 2 / 3 * (seg.control - seg.start)
            c2 = seg.end + 2 / 3 * (seg.control - seg.end)
            upgraded.append(CubicBezier(seg.start, c1, c2, seg.end))
        else:
            upgraded.append(seg)
    segs = upgraded

    if cfg.resample_long_polylines:
        segs = _replace_long_polylines(segs, cfg)

    path = Path(*segs)
    if cfg.enforce_c1:
        path = _enforce_c1_continuity(path)

    out = path.d()
    out = round_path_coords(out, cfg.coordinate_precision)

    if cfg.arc_fit:
        out = fit_arcs_in_path(
            out,
            ArcFitConfig(
                enable=True,
                max_residual=cfg.arc_fit_max_residual,
                coordinate_precision=cfg.coordinate_precision,
            ),
        )
    return out


def _replace_long_polylines(segments: list, cfg: SmoothConfig) -> list:
    """Find runs of Line segments and replace them with smoothed cubic Béziers."""
    out: list = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        if not isinstance(seg, Line):
            out.append(seg)
            i += 1
            continue

        # Collect the run.
        run = [seg]
        j = i + 1
        while j < n and isinstance(segments[j], Line):
            run.append(segments[j])
            j += 1

        if len(run) < cfg.long_polyline_min_pts:
            out.extend(run)
            i = j
            continue

        pts = [(run[0].start.real, run[0].start.imag)]
        for line in run:
            pts.append((line.end.real, line.end.imag))
        points = np.array(pts, dtype=np.float64)

        simplified = _rdp(points, cfg.rdp_epsilon)

        beziers = _fit_cubic_beziers(simplified, cfg.spline_smoothing)
        if not beziers:
            out.extend(run)
            i = j
            continue

        # Snap the first/last anchors to the original endpoints to preserve closure.
        first = beziers[0]
        last = beziers[-1]
        beziers[0] = CubicBezier(complex(points[0, 0], points[0, 1]), first.control1, first.control2, first.end)
        beziers[-1] = CubicBezier(last.start, last.control1, last.control2, complex(points[-1, 0], points[-1, 1]))
        out.extend(beziers)
        i = j

    return out


def smooth_traced_layer(traced: TracedLayer, config: SmoothConfig | None = None) -> TracedLayer:
    """Return a new ``TracedLayer`` with each path-kind element smoothed.

    Primitive elements (circle / ellipse / rect / polygon) are passed through
    unchanged — they are already mathematically clean and don't need RDP or
    spline refitting.
    """
    new_elements = []
    new_paths: list[str] = []
    for el in traced.elements or []:
        if el.kind == "path":
            d = el.attrs.get("d", "")
            smoothed = smooth_path_string(d, config)
            if not smoothed:
                continue
            new_attrs = dict(el.attrs)
            new_attrs["d"] = smoothed
            new_elements.append(type(el)(kind="path", attrs=new_attrs, iou=el.iou, bbox=el.bbox))
            new_paths.append(smoothed)
        else:
            new_elements.append(el)
    return TracedLayer(layer=traced.layer, paths=new_paths, elements=new_elements)


def smooth_traced_layers(
    traced_layers: list[TracedLayer], config: SmoothConfig | None = None
) -> list[TracedLayer]:
    return [smooth_traced_layer(t, config) for t in traced_layers]
