"""Tests for the geometric primitive fitter."""
from __future__ import annotations

import cv2
import numpy as np

from vectorizer.shapes import (
    FitConfig,
    decompose_mask_to_components,
    fit_primitive,
)


def _circle_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _rect_mask(h: int, w: int, x: int, y: int, rw: int, rh: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + rw - 1, y + rh - 1), 255, -1)
    return mask


def test_fit_primitive_detects_circle():
    mask = _circle_mask(200, 200, 100, 100, 60)
    prim = fit_primitive(mask, FitConfig(iou_threshold=0.95))
    assert prim is not None
    assert prim.kind == "circle"
    assert prim.iou >= 0.95
    assert abs(float(prim.attrs["cx"]) - 100) < 2
    assert abs(float(prim.attrs["cy"]) - 100) < 2
    assert abs(float(prim.attrs["r"]) - 60) < 2


def test_fit_primitive_detects_rectangle():
    mask = _rect_mask(200, 200, 30, 50, 120, 80)
    prim = fit_primitive(mask, FitConfig(iou_threshold=0.95))
    assert prim is not None
    # Axis-aligned rect should win.
    assert prim.kind in {"rect", "polygon"}
    assert prim.iou >= 0.95


def test_fit_primitive_rejects_irregular_blob():
    # Build an organic blob that no primitive should match well.
    mask = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array(
        [[40, 60], [80, 30], [140, 50], [170, 110], [150, 170], [70, 180], [30, 130]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)
    # Add an extra bump so it can't fit a circle / rect / convex polygon well.
    cv2.circle(mask, (180, 150), 25, 255, -1)
    prim = fit_primitive(mask, FitConfig(iou_threshold=0.97, poly_max_vertices=6))
    # Either no fit, or a fit with IoU below threshold.
    assert prim is None or prim.iou < 0.97


def test_decompose_yields_separate_components():
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (50, 100), 30, 255, -1)
    cv2.circle(mask, (150, 100), 30, 255, -1)
    comps = list(decompose_mask_to_components(mask, min_area_px=10))
    assert len(comps) == 2
    for comp_mask, holes in comps:
        assert (comp_mask > 0).sum() > 100
        assert holes == []


def test_decompose_detects_hole():
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 70, 255, -1)
    cv2.circle(mask, (100, 100), 30, 0, -1)  # punch a hole
    comps = list(decompose_mask_to_components(mask, min_area_px=10))
    assert len(comps) == 1
    _, holes = comps[0]
    assert len(holes) == 1
