"""Tests for SVG path helpers."""
from __future__ import annotations

from vectorizer.utils.svg_utils import (
    extract_paths_from_svg,
    round_path_coords,
    translate_path,
)


def test_extract_paths_from_svg():
    svg = """<?xml version="1.0"?>
    <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
      <g><path d="M 0 0 L 1 1"/><path d="M 2 2 L 3 3"/></g>
    </svg>
    """
    paths = extract_paths_from_svg(svg)
    assert paths == ["M 0 0 L 1 1", "M 2 2 L 3 3"]


def test_round_path_coords():
    d = "M 1.23456 2.78900 L 3.0000 4.5"
    out = round_path_coords(d, precision=2)
    assert "1.23" in out
    assert "2.79" in out
    # Whole numbers stay short.
    assert "3 " in out or "3," in out or "3L" in out or "3 4.5" in out


def test_translate_path_absolute():
    d = "M 10 20 L 30 40 Z"
    out = translate_path(d, 5, 7)
    # After translation, M should be at 15,27 and L at 35,47.
    assert "15" in out
    assert "27" in out
    assert "35" in out
    assert "47" in out


def test_translate_path_cubic():
    d = "M 0 0 C 10 0 20 10 30 30"
    out = translate_path(d, 100, 200)
    # End anchor 30,30 should become 130,230.
    assert "130" in out
    assert "230" in out


def test_translate_path_noop():
    d = "M 5 5 L 10 10"
    assert translate_path(d, 0, 0) == d
