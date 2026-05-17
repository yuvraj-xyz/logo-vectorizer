"""Tests for the post-trace smoother."""
from __future__ import annotations

from vectorizer.smoother import SmoothConfig, smooth_path_string


def test_smoother_keeps_drawable_output_for_pure_cubic():
    # A perfectly circular cubic should produce a drawable result — either as
    # cubic (if arc-fit disabled) or as a true SVG ``A`` arc command (default).
    d = "M 10 10 C 20 10 20 20 10 20 Z"
    out = smooth_path_string(d)
    upper = out.upper()
    assert "C" in upper or "A" in upper, f"output should contain a curve: {out!r}"


def test_smoother_replaces_arc_like_cubic_with_arc_command():
    # Disable arc-fit; output stays as cubic.
    d = "M 10 10 C 20 10 20 20 10 20 Z"
    no_arc = smooth_path_string(d, SmoothConfig(arc_fit=False))
    assert "C" in no_arc.upper()
    # Enable arc-fit; the same path should snap to an arc.
    with_arc = smooth_path_string(d, SmoothConfig(arc_fit=True))
    assert "A" in with_arc.upper()


def test_smoother_replaces_curved_polyline_with_beziers():
    # A dense polyline sampled from a sine curve — clearly curved, so the
    # smoother should fit cubic Béziers rather than keep all Line segments.
    import math

    pts = [(i, 10 * math.sin(i * 0.25)) for i in range(40)]
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    out = smooth_path_string(d, SmoothConfig(rdp_epsilon=0.5, long_polyline_min_pts=4))
    # Output should contain cubic Bézier commands.
    assert out.upper().count("C") > 0, f"expected cubic Béziers in output, got: {out[:200]}"


def test_smoother_keeps_straight_lines_as_lines():
    # A perfect straight line should NOT be replaced by a cubic — that would
    # be over-engineering and inflate file size for no visual benefit.
    pts = [(i * 0.5, i * 0.5) for i in range(20)]
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    out = smooth_path_string(d)
    # Straight diagonals should remain as line segments.
    assert out.upper().count("L") >= 1


def test_smoother_handles_empty_path():
    assert smooth_path_string("") == ""


def test_smoother_rounds_precision():
    d = "M 10.123456 20.654321 L 30.999999 40.000001 Z"
    out = smooth_path_string(d, SmoothConfig(coordinate_precision=2, long_polyline_min_pts=99))
    assert ".123456" not in out
