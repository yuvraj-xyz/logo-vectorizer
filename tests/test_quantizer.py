"""Tests for k-means color quantization."""
from __future__ import annotations

import numpy as np

from vectorizer.preprocessor import PreprocessConfig, preprocess
from vectorizer.quantizer import QuantizeConfig, quantize


def test_quantize_to_explicit_k(four_color_logo_png):
    pre = preprocess(four_color_logo_png, PreprocessConfig(denoise=False))
    result = quantize(pre.image, target_colors=4)
    assert result.palette_rgb.shape == (4, 3)
    assert len(result.palette_hex) == 4
    assert all(h.startswith("#") and len(h) == 7 for h in result.palette_hex)
    assert result.masks.shape == (4, pre.image.shape[0], pre.image.shape[1])


def test_quantize_masks_partition_image(four_color_logo_png):
    pre = preprocess(four_color_logo_png, PreprocessConfig(denoise=False))
    result = quantize(pre.image, target_colors=4)
    # Every pixel is assigned to exactly one mask.
    combined = result.masks.sum(axis=0)
    h, w = pre.image.shape[:2]
    assert combined.shape == (h, w)
    assert (combined == 255).all()


def test_quantize_auto_picks_reasonable_k(four_color_logo_png):
    pre = preprocess(four_color_logo_png, PreprocessConfig(denoise=False))
    result = quantize(
        pre.image,
        target_colors=None,
        config=QuantizeConfig(auto_detect=True, min_colors=2, max_colors=16),
    )
    assert 2 <= result.palette_rgb.shape[0] <= 16


def test_quantize_single_color(tmp_path):
    # A solid-color image should still produce at least one palette entry.
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = quantize(arr, target_colors=2)
    assert result.palette_rgb.shape[0] >= 1


def test_quantize_label_image_matches_palette_indexing(four_color_logo_png):
    pre = preprocess(four_color_logo_png, PreprocessConfig(denoise=False))
    result = quantize(pre.image, target_colors=4)
    # Reconstruct the image from labels + palette and compare.
    reconstructed = result.palette_rgb[result.label_image]
    np.testing.assert_array_equal(reconstructed, result.quantized_image)
