"""Tests for the path tracer."""
from __future__ import annotations

import numpy as np

from vectorizer.separator import Layer
from vectorizer.tracer import TraceConfig, trace_layer


def _layer_from_mask(mask: np.ndarray, hex_color: str = "#202020") -> Layer:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        bb = (0, 0, 0, 0)
    else:
        bb = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return Layer(
        color_hex=hex_color,
        mask=mask,
        label=f"layer_0_{hex_color.lstrip('#')}",
        area_px=int((mask > 0).sum()),
        bounding_box=bb,
    )


def test_tracer_contour_backend_produces_path():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    traced = trace_layer(_layer_from_mask(mask), TraceConfig(backend="contour"))
    assert traced.paths, "Contour tracer should emit at least one path"
    assert all(p.startswith("M") for p in traced.paths)


def test_tracer_vtracer_backend_returns_paths():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    traced = trace_layer(_layer_from_mask(mask), TraceConfig(backend="vtracer"))
    # Either vtracer succeeded or we fell back gracefully — either way, a path.
    assert traced.paths
    for p in traced.paths:
        assert p.startswith("M") or p.startswith("m")


def test_tracer_empty_mask_returns_no_paths():
    mask = np.zeros((32, 32), dtype=np.uint8)
    traced = trace_layer(_layer_from_mask(mask), TraceConfig(backend="contour"))
    assert traced.paths == []


def test_tracer_compound_path_with_hole():
    # Donut: outer disc with an inner hole.
    h, w = 96, 96
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    outer = (xx - cx) ** 2 + (yy - cy) ** 2 <= 36 ** 2
    inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= 16 ** 2
    mask[outer & ~inner] = 255
    traced = trace_layer(_layer_from_mask(mask), TraceConfig(backend="contour"))
    assert traced.paths
    # The compound path should contain at least two "M" commands (outer + hole).
    combined = " ".join(traced.paths)
    assert combined.count("M") >= 2
