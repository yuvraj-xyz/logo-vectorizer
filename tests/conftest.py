"""Shared fixtures: synthesized logos for fast, deterministic tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _ensure_dir() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def flat_two_color_png(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 128x128 PNG with a black square on a white background."""
    p = tmp_path_factory.mktemp("fix") / "two_color.png"
    img = Image.new("RGB", (128, 128), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 24, 104, 104), fill=(0, 0, 0))
    img.save(p, "PNG")
    return p


@pytest.fixture(scope="session")
def four_color_logo_png(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 256x256 PNG with four well-separated flat colors."""
    p = tmp_path_factory.mktemp("fix") / "four_color.png"
    img = Image.new("RGB", (256, 256), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 236, 236), fill=(26, 26, 46))
    draw.ellipse((64, 64, 192, 192), fill=(233, 69, 96))
    draw.polygon([(128, 90), (90, 165), (165, 165)], fill=(0, 173, 181))
    img.save(p, "PNG")
    return p


@pytest.fixture(scope="session")
def rgba_png(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 64x64 RGBA PNG with a translucent shape to test alpha flattening."""
    p = tmp_path_factory.mktemp("fix") / "rgba.png"
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill=(200, 30, 60, 255))
    img.save(p, "PNG")
    return p


@pytest.fixture(scope="session")
def tiny_png(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 40x40 PNG to verify auto-upscaling behaviour."""
    p = tmp_path_factory.mktemp("fix") / "tiny.png"
    img = Image.new("RGB", (40, 40), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((5, 5, 35, 35), fill=(40, 40, 200))
    img.save(p, "PNG")
    return p
