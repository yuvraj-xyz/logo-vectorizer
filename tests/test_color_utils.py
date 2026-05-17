"""Tests for color utilities."""
from __future__ import annotations

import numpy as np

from vectorizer.utils.color import (
    delta_e_2000,
    hex_to_rgb,
    luminance,
    rgb_to_hex,
    rgb_to_lab,
)


def test_rgb_to_hex_roundtrip():
    assert rgb_to_hex((255, 0, 128)) == "#ff0080"
    assert hex_to_rgb("#ff0080") == (255, 0, 128)
    assert hex_to_rgb("F0F") == (255, 0, 255)


def test_rgb_to_lab_white_point():
    # sRGB white should map to L*=100.
    lab = rgb_to_lab(np.array([[255, 255, 255]]))
    assert abs(lab[0, 0] - 100.0) < 1e-3


def test_delta_e_zero_for_equal_colors():
    a = np.array([50.0, 10.0, 5.0])
    assert delta_e_2000(a, a) < 1e-6


def test_luminance_orders_correctly():
    white = luminance((255, 255, 255))
    black = luminance((0, 0, 0))
    mid = luminance((128, 128, 128))
    assert white > mid > black
