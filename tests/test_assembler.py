"""Tests for the SVG assembler."""
from __future__ import annotations

import numpy as np
from lxml import etree

from vectorizer.assembler import AssembleConfig, assemble_svg
from vectorizer.separator import Layer
from vectorizer.smoother import TracedLayer


def _make_traced(color: str, d: str, is_background: bool = False) -> TracedLayer:
    layer = Layer(
        color_hex=color,
        mask=np.zeros((10, 10), dtype=np.uint8),
        label=f"layer_0_{color.lstrip('#')}",
        area_px=100,
        bounding_box=(0, 0, 10, 10),
        is_background=is_background,
    )
    return TracedLayer(layer=layer, paths=[d])


def test_assembler_produces_parseable_svg():
    traced = [
        _make_traced("#ffffff", "M 0 0 L 100 0 L 100 100 L 0 100 Z", is_background=True),
        _make_traced("#202060", "M 20 20 L 80 20 L 80 80 L 20 80 Z"),
    ]
    out = assemble_svg(traced, width=100, height=100, source_filename="logo.png")
    assert out.layer_count == 2
    root = etree.fromstring(out.svg_string.encode("utf-8"))
    ns = "{http://www.w3.org/2000/svg}"
    groups = root.findall(f"{ns}g")
    assert len(groups) == 2
    # Background group's id must be present.
    assert any(g.get("id", "").startswith("layer_0_") for g in groups)


def test_assembler_emits_metadata_when_enabled():
    traced = [_make_traced("#000000", "M 0 0 L 5 5 Z")]
    out = assemble_svg(traced, width=10, height=10, source_filename="a.png")
    assert "<desc" in out.svg_string
    assert "Logo Vectorizer" in out.svg_string


def test_assembler_skips_empty_layers():
    layer = Layer(
        color_hex="#abcabc",
        mask=np.zeros((10, 10), dtype=np.uint8),
        label="layer_x",
        area_px=0,
        bounding_box=(0, 0, 0, 0),
    )
    traced = [TracedLayer(layer=layer, paths=[])]
    out = assemble_svg(traced, width=10, height=10)
    assert out.layer_count == 0


def test_assembler_inkscape_namespace_present():
    traced = [_make_traced("#abcdef", "M 0 0 L 1 1 Z")]
    out = assemble_svg(traced, width=2, height=2)
    assert "xmlns:inkscape" in out.svg_string
    assert "inkscape:label" in out.svg_string


def test_assembler_default_fill_rule_evenodd():
    traced = [_make_traced("#abcdef", "M 0 0 L 1 1 Z")]
    out = assemble_svg(traced, width=2, height=2)
    assert 'fill-rule="evenodd"' in out.svg_string
