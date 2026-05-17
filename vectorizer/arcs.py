"""Arc-fitting post-processor.

Vectorizer.AI's output looks cleaner than pure-Bézier tracers largely because
it allows **circular and elliptical arcs** as primary curve types. SVG natively
supports arcs via the ``A`` (elliptical arc) command, but most tracers only
emit cubic Béziers — which can approximate arcs but never *be* arcs, so they
always have tiny ripples.

This module walks an SVG ``d`` string after tracing and replaces runs of cubic
Bézier segments that closely approximate a circular arc with a true SVG ``A``
command. The result is mathematically perfect curvature in the output and a
visibly smoother trace — exactly what Vectorizer.AI's "Allowed Curve Types →
Circular Arcs" toggle does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from svgpathtools import Arc, CubicBezier, Line, Path, parse_path


@dataclass
class ArcFitConfig:
    enable: bool = True
    # Max distance (px) any sampled point on the Bézier run may be from the
    # candidate circle. Smaller = stricter, more conservative arc snapping.
    # 1.5 catches most rounded-corner cubics from potrace without misfitting
    # straight-ish curves.
    max_residual: float = 1.5
    # Minimum number of consecutive cubic Béziers to consider replacing with an
    # arc. Potrace tends to emit a single cubic per quarter-arc, so we accept
    # single-segment runs and rely on the residual + radius gates to reject
    # straight-line-ish cubics that "fit" as huge-radius arcs.
    min_segments: int = 1
    # Don't fit arcs to runs whose total chord length is below this many px.
    min_chord_length: float = 10.0
    # Don't fit arcs whose radius is below this many px — tiny radii hint at
    # corner approximations rather than real curvature.
    min_radius: float = 4.0
    # Max acceptable radius — beyond this, the run is closer to a line than
    # an arc, and an arc fit will be visually identical to the Bézier. Cap at
    # 1500px and pair with the chord/radius ratio check below to reject
    # straight-ish cubics that the Kasa fit happily reports as huge arcs.
    max_radius: float = 1500.0
    # Reject arc fits where the radius is more than this many times the chord
    # length. For a real arc, chord ≈ 2·r·sin(θ/2); when r/chord is huge, the
    # segment is essentially straight and emitting an arc is wrong.
    max_radius_chord_ratio: float = 18.0
    # Minimum endpoint separation (px) for an emitted arc. Closed-loop runs
    # (start == end) are degenerate in SVG: per spec they're a no-op, but
    # Illustrator and some renderers treat them as a full filled circle.
    min_endpoint_gap: float = 1.5
    # How many sample points per Bézier to evaluate during fitting.
    samples_per_segment: int = 8
    coordinate_precision: int = 2


# --------------------------- circle fitting ---------------------------


def _fit_circle_least_squares(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic circle fit (Kasa). Returns (cx, cy, r, rms_residual).

    Solves ``A*cx + B*cy + C = -(x^2 + y^2)`` in least-squares sense.
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return cx, cy, 0.0, float("inf")
    r = math.sqrt(r2)
    residuals = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return float(cx), float(cy), float(r), rms


# --------------------------- Bézier sampling ---------------------------


def _sample_cubics(segments: list[CubicBezier], n_per_seg: int) -> np.ndarray:
    """Densely sample a run of cubic Béziers as (N, 2) points (xy)."""
    out: list[tuple[float, float]] = []
    for i, seg in enumerate(segments):
        # Skip the duplicated endpoint between adjacent segments.
        ts = np.linspace(0, 1, n_per_seg, endpoint=(i == len(segments) - 1))
        for t in ts:
            pt = seg.point(float(t))
            out.append((pt.real, pt.imag))
    return np.array(out, dtype=np.float64)


# --------------------------- arc command emission ---------------------------


def _angle(cx: float, cy: float, x: float, y: float) -> float:
    return math.atan2(y - cy, x - cx)


def _arc_command(
    start: tuple[float, float],
    end: tuple[float, float],
    cx: float,
    cy: float,
    r: float,
    sweep_clockwise: bool,
    prec: int,
) -> str:
    """Emit an SVG ``A`` (elliptical arc) command from start to end on a
    circle of radius ``r`` centered at (cx, cy)."""
    a0 = _angle(cx, cy, *start)
    a1 = _angle(cx, cy, *end)
    delta = a1 - a0
    # Normalize to (-π, π] then determine large-arc flag from absolute delta.
    if sweep_clockwise:
        if delta > 0:
            delta -= 2 * math.pi
    else:
        if delta < 0:
            delta += 2 * math.pi
    large_arc = 1 if abs(delta) > math.pi else 0
    sweep_flag = 0 if sweep_clockwise else 1
    return (
        f"A{r:.{prec}f} {r:.{prec}f} 0 {large_arc} {sweep_flag} "
        f"{end[0]:.{prec}f} {end[1]:.{prec}f}"
    )


def _acceptable_arc_fit(r: float, rms: float, chord: float, cfg: "ArcFitConfig") -> bool:
    """Gate every candidate arc through the same set of sanity checks.

    Rejects:
        * radii outside ``[min_radius, max_radius]``
        * RMS deviation above ``max_residual``
        * runs whose chord is shorter than ``min_chord_length``
        * runs where the radius dwarfs the chord — those are essentially
          straight lines and emitting them as huge-radius arcs produces
          renderer-specific surprises (Illustrator paints a giant arc).
    """
    if r < cfg.min_radius or r > cfg.max_radius:
        return False
    if rms > cfg.max_residual:
        return False
    if chord < cfg.min_chord_length:
        return False
    if r > cfg.max_radius_chord_ratio * chord:
        return False
    return True


def _detect_sweep(points: np.ndarray, cx: float, cy: float) -> bool:
    """True if the sampled arc winds clockwise around the center."""
    angs = np.array([_angle(cx, cy, x, y) for x, y in points])
    # Unwrap so we can detect monotonic decrease/increase.
    unwrapped = np.unwrap(angs)
    return unwrapped[-1] < unwrapped[0]


# --------------------------- top-level pass ---------------------------


def fit_arcs_in_path(d: str, config: ArcFitConfig | None = None) -> str:
    """Walk an SVG ``d`` string and replace cubic-Bézier runs that approximate
    a circular arc with SVG ``A`` arc commands.

    Anything that doesn't fit a circle within tolerance is left untouched.
    Lines, corner segments, and other commands are passed through unchanged.
    """
    cfg = config or ArcFitConfig()
    if not cfg.enable or not d:
        return d

    try:
        path = parse_path(d)
    except Exception:
        return d
    if len(path) == 0:
        return d

    out_parts: list[str] = []
    i = 0
    segs = list(path)
    n = len(segs)
    current_pen = None  # (x, y) — the current SVG pen position

    def _move(p) -> str:
        return f"M{p.real:.{cfg.coordinate_precision}f} {p.imag:.{cfg.coordinate_precision}f}"

    def _line(p) -> str:
        return f"L{p.real:.{cfg.coordinate_precision}f} {p.imag:.{cfg.coordinate_precision}f}"

    def _cubic(seg: CubicBezier) -> str:
        prec = cfg.coordinate_precision
        return (
            f"C{seg.control1.real:.{prec}f} {seg.control1.imag:.{prec}f} "
            f"{seg.control2.real:.{prec}f} {seg.control2.imag:.{prec}f} "
            f"{seg.end.real:.{prec}f} {seg.end.imag:.{prec}f}"
        )

    # Emit the initial M based on the first segment's start.
    if isinstance(segs[0], (CubicBezier, Line)):
        start = segs[0].start
        out_parts.append(_move(start))
        current_pen = start

    while i < n:
        seg = segs[i]
        if isinstance(seg, CubicBezier):
            # Walk consecutive cubics one at a time: try to fit the current
            # cubic alone as an arc, then greedily extend by adding adjacent
            # cubics only while the combined points still fit the *same*
            # circle. This way a triangle outline (each side a single cubic,
            # but no shared circle) doesn't get nuked by a single failed
            # global fit — each side is tested independently.
            j = i
            while j < n and isinstance(segs[j], CubicBezier):
                first = segs[j]
                pts = _sample_cubics([first], cfg.samples_per_segment)
                chord = float(np.linalg.norm(pts[-1] - pts[0]))
                if chord < cfg.min_chord_length:
                    out_parts.append(_cubic(first))
                    current_pen = first.end
                    j += 1
                    continue

                cx, cy, r, rms = _fit_circle_least_squares(pts)
                if not _acceptable_arc_fit(r, rms, chord, cfg):
                    out_parts.append(_cubic(first))
                    current_pen = first.end
                    j += 1
                    continue

                # We have an arc-fittable starting cubic. Try to extend.
                run: list[CubicBezier] = [first]
                k = j + 1
                last_good = (cx, cy, r)
                last_pts = pts
                while (
                    k < n
                    and isinstance(segs[k], CubicBezier)
                    and abs(segs[k].start - run[-1].end) < 1e-3
                ):
                    next_pts = _sample_cubics([segs[k]], cfg.samples_per_segment)
                    combined = np.vstack([last_pts, next_pts])
                    ncx, ncy, nr, nrms = _fit_circle_least_squares(combined)
                    combined_chord = float(np.linalg.norm(combined[-1] - combined[0]))
                    if (
                        _acceptable_arc_fit(nr, nrms, combined_chord, cfg)
                        # Don't let the radius wander too far from where we started.
                        and abs(nr - last_good[2]) <= max(2.0, 0.10 * last_good[2])
                    ):
                        run.append(segs[k])
                        last_pts = combined
                        last_good = (ncx, ncy, nr)
                        k += 1
                    else:
                        break

                # Emit the arc spanning all extended cubics — but only if start
                # and end are actually distinct. Closed-loop runs (start == end)
                # are interpreted as full filled circles by Illustrator and
                # other renderers, which is never what we want.
                cx, cy, r = last_good
                start_pt = (float(run[0].start.real), float(run[0].start.imag))
                end_pt = (float(run[-1].end.real), float(run[-1].end.imag))
                gap = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                if gap < cfg.min_endpoint_gap:
                    for s in run:
                        out_parts.append(_cubic(s))
                        current_pen = s.end
                    j = k
                    continue

                sweep_cw = _detect_sweep(last_pts, cx, cy)
                out_parts.append(
                    _arc_command(start_pt, end_pt, cx, cy, r, sweep_cw, cfg.coordinate_precision)
                )
                current_pen = run[-1].end
                j = k

            i = j

        elif isinstance(seg, Line):
            # Bridge implicit moves if pen jumped.
            if current_pen is None or abs(seg.start - current_pen) > 1e-3:
                out_parts.append(_move(seg.start))
            out_parts.append(_line(seg.end))
            current_pen = seg.end
            i += 1
        else:
            # Unknown segment kind — fall back to its d() rendering.
            try:
                out_parts.append(seg.d())
            except Exception:
                pass
            current_pen = seg.end if hasattr(seg, "end") else current_pen
            i += 1

    # Preserve the Z if the original path ended with a close.
    if d.strip().endswith("Z") or d.strip().endswith("z"):
        out_parts.append("Z")

    return " ".join(out_parts)


# --------------------------- Arc → cubic flattening ---------------------------


def _arc_to_cubic_beziers(arc) -> list[CubicBezier]:
    """Approximate an SVG ``Arc`` with cubic Béziers (≤ 90° per cubic).

    Uses the classic ``k = 4/3 · tan(θ/4)`` Bézier-control-handle formula. For
    a 90° quadrant this matches a true circular arc to within ~0.027% radius
    deviation — well below pixel-level error for typical logos.
    """
    cx = arc.center.real
    cy = arc.center.imag
    rx = arc.radius.real
    ry = arc.radius.imag
    rot = math.radians(arc.rotation)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    theta = math.radians(arc.theta)
    delta = math.radians(arc.delta)

    def point_at(angle: float) -> complex:
        ex = math.cos(angle) * rx
        ey = math.sin(angle) * ry
        return complex(cx + ex * cos_r - ey * sin_r, cy + ex * sin_r + ey * cos_r)

    def tangent_at(angle: float) -> complex:
        tx = -math.sin(angle) * rx
        ty = math.cos(angle) * ry
        return complex(tx * cos_r - ty * sin_r, tx * sin_r + ty * cos_r)

    n_steps = max(1, int(math.ceil(abs(delta) / (math.pi / 2))))
    step = delta / n_steps
    k = 4 / 3 * math.tan(step / 4)

    beziers: list[CubicBezier] = []
    a0 = theta
    p0 = point_at(a0)
    for _ in range(n_steps):
        a1 = a0 + step
        p3 = point_at(a1)
        t0 = tangent_at(a0)
        t1 = tangent_at(a1)
        p1 = p0 + k * t0
        p2 = p3 - k * t1
        beziers.append(CubicBezier(p0, p1, p2, p3))
        p0 = p3
        a0 = a1
    return beziers


def flatten_arcs_in_d(d: str, precision: int = 2) -> str:
    """Return ``d`` with every SVG ``A`` (arc) command replaced by cubic Béziers.

    Useful for downstream consumers that don't implement SVG arc commands —
    notably the native PostScript writer in :mod:`vectorizer.exporters.eps`
    and some legacy EPS readers.
    """
    if not d or "A" not in d.upper():
        return d
    try:
        path = parse_path(d)
    except Exception:
        return d

    new_segs = []
    for seg in path:
        if isinstance(seg, Arc):
            new_segs.extend(_arc_to_cubic_beziers(seg))
        else:
            new_segs.append(seg)
    try:
        result = Path(*new_segs).d()
    except Exception:
        return d
    # Round to keep file sizes sane.
    from .utils.svg_utils import round_path_coords
    return round_path_coords(result, precision)


def flatten_arcs_in_svg(svg_string: str, precision: int = 2) -> str:
    """Flatten ``A`` commands in every ``<path>`` of an SVG document."""
    from lxml import etree

    if "A" not in svg_string.upper() and "a" not in svg_string:
        return svg_string
    try:
        root = etree.fromstring(svg_string.encode("utf-8"))
    except Exception:
        return svg_string
    ns = "{http://www.w3.org/2000/svg}"
    changed = False
    for path in root.iter(f"{ns}path"):
        d = path.get("d")
        if not d or "A" not in d.upper():
            continue
        new_d = flatten_arcs_in_d(d, precision)
        if new_d != d:
            path.set("d", new_d)
            changed = True
    if not changed:
        return svg_string
    return etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
    ).decode("utf-8")
