"""Image preprocessor: load, flatten alpha, normalize size, denoise."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils.image_utils import flatten_alpha, load_image, to_numpy


@dataclass
class PreprocessConfig:
    """Knobs for the preprocessing stage."""

    target_max_px: int = 2000
    target_min_px: int = 200
    denoise: bool = True
    denoise_strength: int = 9       # bilateral filter d
    denoise_sigma_color: int = 75
    denoise_sigma_space: int = 75
    flatten_alpha_bg: tuple[int, int, int] = (255, 255, 255)


@dataclass
class PreprocessResult:
    image: np.ndarray              # (H, W, 3) uint8 RGB
    original_size: tuple[int, int] # (width, height) of the input file
    scale: float                   # output_dim / input_dim
    source_path: str | None


def preprocess(
    source: str | Path | Image.Image | np.ndarray,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Load and normalize an image so the rest of the pipeline gets a clean canvas.

    Steps:
        1. Load (from disk if a path) and detect color mode.
        2. Flatten any alpha channel onto a configurable background.
        3. Resize so the longest edge fits inside [target_min_px, target_max_px].
        4. Optionally bilateral-denoise to suppress JPEG artifacts without
           destroying edges.
    """
    cfg = config or PreprocessConfig()

    source_path: str | None = None
    if isinstance(source, (str, Path)):
        source_path = str(source)
        img = load_image(source)
    elif isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source.astype(np.uint8))
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    original_size = img.size

    flat = flatten_alpha(img, cfg.flatten_alpha_bg)
    arr = to_numpy(flat)

    h, w = arr.shape[:2]
    longest = max(h, w)
    scale = 1.0

    if longest > cfg.target_max_px:
        scale = cfg.target_max_px / longest
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif longest < cfg.target_min_px:
        scale = cfg.target_min_px / longest
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # Lanczos for upscaling small logos preserves crisp edges
        pil = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
        arr = np.array(pil, dtype=np.uint8)

    if cfg.denoise:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bgr = cv2.bilateralFilter(
            bgr,
            d=cfg.denoise_strength,
            sigmaColor=cfg.denoise_sigma_color,
            sigmaSpace=cfg.denoise_sigma_space,
        )
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return PreprocessResult(
        image=arr,
        original_size=original_size,
        scale=scale,
        source_path=source_path,
    )
