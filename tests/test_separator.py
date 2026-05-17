"""Tests for the layer separator (mask cleanup + background detection)."""
from __future__ import annotations

import numpy as np

from vectorizer.separator import SeparatorConfig, separate


def _square_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_separator_drops_tiny_components():
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0, 0] = 255            # 1-pixel speckle — should vanish
    mask[10:30, 10:30] = 255    # 400-pixel real shape (may shrink slightly under morph open)
    masks = np.stack([mask], axis=0)
    layers = separate(masks, ["#000000"], SeparatorConfig(min_area_px=20, detect_background=False))
    assert len(layers) == 1
    # Morphological opening can shave a 1-pixel border off the shape.
    assert layers[0].area_px >= 350


def test_separator_orders_largest_first():
    h, w = 80, 80
    small = _square_mask(h, w, 60, 60, 75, 75)
    big = _square_mask(h, w, 5, 5, 70, 70)
    masks = np.stack([small, big], axis=0)
    layers = separate(masks, ["#aabbcc", "#112233"], SeparatorConfig(detect_background=False))
    assert layers[0].color_hex == "#112233"
    assert layers[1].color_hex == "#aabbcc"


def test_separator_detects_background_layer():
    h, w = 64, 64
    bg = np.full((h, w), 255, dtype=np.uint8)
    bg[20:44, 20:44] = 0
    fg = np.zeros((h, w), dtype=np.uint8)
    fg[20:44, 20:44] = 255
    masks = np.stack([bg, fg], axis=0)
    layers = separate(masks, ["#ffffff", "#000000"], SeparatorConfig(detect_background=True))
    # First layer should be the detected background.
    assert layers[0].is_background is True
    assert layers[0].color_hex == "#ffffff"


def test_separator_bounding_box():
    h, w = 32, 32
    mask = _square_mask(h, w, 5, 6, 20, 25)
    # Disable morphology so we test pure connected-component bbox extraction.
    layers = separate(
        np.stack([mask], axis=0),
        ["#202020"],
        SeparatorConfig(detect_background=False, morph_kernel_px=0),
    )
    bb = layers[0].bounding_box
    assert bb[0] == 5
    assert bb[1] == 6
    assert bb[2] == 15
    assert bb[3] == 19
