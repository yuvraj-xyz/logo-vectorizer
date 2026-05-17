"""End-to-end pipeline tests."""
from __future__ import annotations

from pathlib import Path

from lxml import etree

from vectorizer.pipeline import PipelineConfig, vectorize
from vectorizer.tracer import TraceConfig


def test_pipeline_produces_all_three_formats(four_color_logo_png, tmp_path: Path):
    out_dir = tmp_path / "out"
    result = vectorize(
        four_color_logo_png,
        output_dir=out_dir,
        formats=("svg", "eps", "ai"),
    )
    assert (out_dir / "four_color.svg").exists()
    assert (out_dir / "four_color.eps").exists()
    assert (out_dir / "four_color.ai").exists()
    assert result.assemble.layer_count >= 2


def test_pipeline_svg_is_parseable(four_color_logo_png, tmp_path: Path):
    result = vectorize(four_color_logo_png, output_dir=tmp_path, formats=("svg",))
    root = etree.fromstring(result.svg_string.encode("utf-8"))
    ns = "{http://www.w3.org/2000/svg}"
    groups = root.findall(f"{ns}g")
    assert len(groups) >= 2
    # Each layer should have at least one drawable child — either a primitive
    # element (circle/ellipse/rect/polygon) or a fallback <path>.
    drawable_tags = {f"{ns}{t}" for t in ("path", "circle", "ellipse", "rect", "polygon")}
    for g in groups:
        assert g.get("fill") is not None
        drawables = [child for child in g if child.tag in drawable_tags]
        assert drawables, f"group {g.get('id')} has no drawable children"


def test_pipeline_with_contour_backend(four_color_logo_png, tmp_path: Path):
    cfg = PipelineConfig(trace=TraceConfig(backend="contour"))
    result = vectorize(four_color_logo_png, output_dir=tmp_path, formats=("svg",), config=cfg)
    assert result.assemble.layer_count >= 2
    assert result.svg_string.startswith("<?xml")


def test_pipeline_emits_correct_palette_size(four_color_logo_png, tmp_path: Path):
    result = vectorize(
        four_color_logo_png,
        output_dir=tmp_path,
        formats=("svg",),
        target_colors=4,
    )
    # Allow 2-4 because tiny anti-aliased pixels may be dropped by separator.
    assert 2 <= len(result.assemble.palette) <= 4


def test_pipeline_progress_callback_fires(four_color_logo_png, tmp_path: Path):
    events = []

    def cb(stage: str, pct: float) -> None:
        events.append((stage, pct))

    vectorize(four_color_logo_png, output_dir=tmp_path, formats=("svg",), progress=cb)
    stages = [s for s, _ in events]
    assert "preprocess" in stages
    assert "done" in stages
    # Progress should be monotonically non-decreasing.
    pcts = [p for _, p in events]
    assert all(b >= a - 1e-6 for a, b in zip(pcts, pcts[1:]))
