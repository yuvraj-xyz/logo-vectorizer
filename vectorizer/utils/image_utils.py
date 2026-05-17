"""Image loading and basic raster manipulation helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk, validating the file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {p.suffix!r}. Only PNG and JPEG are supported."
        )
    return Image.open(p)


def flatten_alpha(img: Image.Image, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Composite an image with transparency onto a solid background."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, background)
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "LA":
        rgba = img.convert("RGBA")
        return flatten_alpha(rgba, background)
    if img.mode == "P":
        if "transparency" in img.info:
            rgba = img.convert("RGBA")
            return flatten_alpha(rgba, background)
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an (H, W, 3) uint8 numpy array."""
    return np.array(img.convert("RGB"), dtype=np.uint8)


def from_numpy(arr: np.ndarray) -> Image.Image:
    """Convert an (H, W, 3) uint8 numpy array to a PIL RGB image."""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")
