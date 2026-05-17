"""Mask cleanup and layer ordering for per-color tracing."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils.color import hex_to_rgb, luminance


@dataclass
class Layer:
    color_hex: str
    mask: np.ndarray              # uint8 binary image (0 or 255)
    label: str                    # e.g. "layer_0_1a1a2e"
    area_px: int
    bounding_box: tuple[int, int, int, int]   # (x, y, w, h)
    is_background: bool = False


@dataclass
class SeparatorConfig:
    min_area_px: int = 50         # discard tiny components (noise)
    morph_kernel_px: int = 2      # opening kernel to remove single-pixel speckle
    detect_background: bool = True
    # Anti-alias smoothing: Gaussian-blur each mask and re-threshold before tracing.
    # 0.0 disables; 0.6 is the gentle default that pairs well with potrace; 1.5+
    # aggressively kills wobble at the cost of slight shape distortion.
    mask_smoothing: float = 0.6
    # Optional median pre-filter to remove salt-and-pepper noise.
    median_filter: bool = True
    # Dilate every non-background mask by this many pixels before tracing so
    # adjacent shapes overlap slightly. Without this, each layer is traced
    # independently and the resulting boundaries don't quite align, leaving
    # 1-px hairline gaps that show the background color through the seams.
    overlap_dilation_px: int = 1


def smooth_masks_argmax(masks: np.ndarray, sigma: float) -> np.ndarray:
    """Globally smooth the color partition by Gaussian-blur-then-argmax.

    Smoothing each binary mask independently produces gaps and overlaps at the
    boundaries between adjacent colors. Instead, we blur every mask in float
    space, then assign each pixel to whichever color has the highest blurred
    value at that location. This is mathematically equivalent to a softmax-style
    competition and guarantees the output remains a complete, non-overlapping
    tiling of the canvas — clean boundaries with no fringe.
    """
    if sigma <= 0:
        return masks
    k, h, w = masks.shape
    if k == 1:
        return masks
    blurred = np.empty((k, h, w), dtype=np.float32)
    for c in range(k):
        blurred[c] = cv2.GaussianBlur(
            masks[c].astype(np.float32),
            ksize=(0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
        )
    winners = blurred.argmax(axis=0)
    out = np.zeros_like(masks)
    for c in range(k):
        out[c] = np.where(winners == c, 255, 0).astype(np.uint8)
    return out


def _cleanup_mask(mask: np.ndarray, kernel_px: int, median: bool = True) -> np.ndarray:
    """Final per-mask cleanup: median filter to kill salt-pepper, then morph open."""
    cleaned = mask
    if median and min(cleaned.shape[:2]) >= 5:
        cleaned = cv2.medianBlur(cleaned, 3)
    if kernel_px > 0:
        k = max(1, kernel_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def _drop_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Remove connected components below ``min_area`` pixels.

    Returns the cleaned mask and the surviving area in pixels.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    surviving = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
            surviving += area
    return out, surviving


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def separate(
    masks: np.ndarray,
    palette_hex: list[str],
    config: SeparatorConfig | None = None,
) -> list[Layer]:
    """Clean per-color masks and return them as ``Layer`` objects.

    Layers are returned sorted with the background first (largest area,
    likely touching the image border) and the smallest details last —
    matching SVG paint order.
    """
    cfg = config or SeparatorConfig()
    if masks.ndim != 3:
        raise ValueError("masks must be (K, H, W)")
    if masks.shape[0] != len(palette_hex):
        raise ValueError("palette length must match mask count")

    h, w = masks.shape[1:]

    # Global smoothing pass: ensures the color partition stays complete (no
    # gaps/overlaps between adjacent colors) while killing anti-alias wobble.
    smoothed_masks = smooth_masks_argmax(masks, cfg.mask_smoothing)

    layers: list[Layer] = []
    for idx, hex_color in enumerate(palette_hex):
        raw = smoothed_masks[idx]
        cleaned = _cleanup_mask(raw, cfg.morph_kernel_px, cfg.median_filter)
        cleaned, area = _drop_small_components(cleaned, cfg.min_area_px)

        if area == 0:
            continue

        layers.append(
            Layer(
                color_hex=hex_color,
                mask=cleaned,
                label=f"layer_{idx}_{hex_color.lstrip('#')}",
                area_px=area,
                bounding_box=_bounding_box(cleaned),
            )
        )

    if cfg.detect_background and layers:
        bg_idx = _pick_background_layer(layers, h, w)
        if bg_idx is not None:
            layers[bg_idx].is_background = True

    # Largest area first (background sits at the bottom of SVG paint order),
    # then descending — so foreground details paint on top.
    layers.sort(key=lambda layer: (not layer.is_background, -layer.area_px))

    # Dilate every non-background mask so adjacent layers overlap. Without
    # this, potrace traces each layer independently and the resulting
    # boundaries don't quite share pixels, leaving hairline gaps that show
    # the canvas background through the seams. The background layer stays
    # at exact size — it will be over-painted by the foreground anyway.
    if cfg.overlap_dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for layer in layers:
            if layer.is_background:
                continue
            layer.mask = cv2.dilate(
                layer.mask, kernel, iterations=cfg.overlap_dilation_px
            )

    return layers


def _pick_background_layer(layers: list[Layer], h: int, w: int) -> int | None:
    """Pick the layer that most resembles a canvas background.

    A background layer:
        - covers a large fraction of the image, or
        - touches all four borders, or
        - is the lightest large layer.
    """
    if not layers:
        return None

    total = h * w
    candidates: list[tuple[float, int]] = []
    for i, layer in enumerate(layers):
        coverage = layer.area_px / total
        if coverage < 0.20:
            continue
        mask = layer.mask
        touches = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
        if touches < 2 and coverage < 0.45:
            continue
        lum = luminance(hex_to_rgb(layer.color_hex))
        # Prefer larger + brighter + more border touching
        score = coverage * 2.0 + touches * 0.4 + lum * 0.3
        candidates.append((score, i))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
