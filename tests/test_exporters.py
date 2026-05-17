"""Tests for the EPS and AI exporters."""
from __future__ import annotations

from pathlib import Path

from vectorizer.exporters import svg_to_ai, svg_to_eps


SVG_FIXTURE = """<?xml version='1.0' encoding='UTF-8' standalone='no'?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 100 100" width="100" height="100">
  <g id="bg" fill="#ffffff" stroke="none">
    <path d="M 0 0 L 100 0 L 100 100 L 0 100 Z"/>
  </g>
  <g id="fg" fill="#1a1a2e" stroke="none">
    <path d="M 20 20 L 80 20 L 80 80 L 20 80 Z"/>
  </g>
</svg>
"""


def test_eps_exporter_writes_postscript_header(tmp_path: Path):
    out = svg_to_eps(SVG_FIXTURE, tmp_path / "out.eps")
    assert out.exists()
    text = out.read_text(encoding="utf-8", errors="ignore")
    assert text.startswith("%!PS-Adobe"), text[:50]
    assert "%%BoundingBox" in text


def test_ai_exporter_writes_ai_marker_block(tmp_path: Path):
    out = svg_to_ai(SVG_FIXTURE, tmp_path / "out.ai", title="myLogo")
    assert out.exists()
    text = out.read_text(encoding="utf-8", errors="ignore")
    assert text.startswith("%!PS-Adobe-3.0")
    assert "AI8_CreatorVersion" in text
    assert "myLogo.ai" in text


def test_eps_includes_path_geometry(tmp_path: Path):
    out = svg_to_eps(SVG_FIXTURE, tmp_path / "geom.eps")
    text = out.read_text(encoding="utf-8", errors="ignore")
    # Either cairosvg's binary PS or our native writer should emit drawing ops.
    lowered = text.lower()
    assert any(op in lowered for op in ["moveto", "lineto", "rectfill", "show"])
