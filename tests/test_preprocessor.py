"""Tests for the preprocessing stage."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vectorizer.preprocessor import PreprocessConfig, preprocess


def test_preprocess_returns_rgb_array(four_color_logo_png):
    result = preprocess(four_color_logo_png)
    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_preprocess_records_source_path(four_color_logo_png):
    result = preprocess(four_color_logo_png)
    assert result.source_path == str(four_color_logo_png)
    assert result.original_size == (256, 256)


def test_preprocess_flattens_alpha(rgba_png):
    result = preprocess(rgba_png, PreprocessConfig(denoise=False, target_min_px=64, target_max_px=128))
    # No alpha leakage — all pixels should be 3-channel uint8.
    assert result.image.shape[2] == 3
    # White background corner pixel must be (close to) the configured flatten color.
    px = result.image[0, 0]
    assert px[0] >= 240 and px[1] >= 240 and px[2] >= 240


def test_preprocess_upscales_tiny_images(tiny_png):
    cfg = PreprocessConfig(denoise=False, target_min_px=200, target_max_px=2000)
    result = preprocess(tiny_png, cfg)
    longest = max(result.image.shape[:2])
    assert longest >= 200
    assert result.scale > 1.0


def test_preprocess_downscales_huge_images(tmp_path):
    big = Image.new("RGB", (4096, 4096), (0, 0, 0))
    big_path = tmp_path / "big.png"
    big.save(big_path, "PNG")
    cfg = PreprocessConfig(denoise=False, target_max_px=1000)
    result = preprocess(big_path, cfg)
    assert max(result.image.shape[:2]) <= 1000
    assert result.scale < 1.0


def test_preprocess_rejects_unsupported(tmp_path):
    bmp = tmp_path / "x.bmp"
    Image.new("RGB", (32, 32)).save(bmp, "BMP")
    with pytest.raises(ValueError):
        preprocess(bmp)


def test_preprocess_accepts_pil_image():
    img = Image.new("RGB", (256, 256), (10, 20, 30))
    result = preprocess(img, PreprocessConfig(denoise=False))
    assert result.image.shape[:2] == (256, 256)
    assert result.source_path is None
