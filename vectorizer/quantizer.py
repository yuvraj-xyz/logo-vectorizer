"""K-means color quantizer with automatic palette-size detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .utils.color import rgb_to_hex, rgb_to_lab


@dataclass
class QuantizeConfig:
    min_colors: int = 2
    max_colors: int = 32
    auto_detect: bool = True       # use SSE elbow to pick k when target_colors is None
    color_space: str = "lab"       # "lab" or "rgb"
    min_area_px: int = 50          # ignore tiny color regions (handled by separator)
    sample_pixels: int = 50_000    # subsample for k-means speed
    random_state: int = 42


@dataclass
class QuantizeResult:
    quantized_image: np.ndarray            # (H, W, 3) uint8
    palette_rgb: np.ndarray                # (K, 3) uint8
    palette_hex: list[str]                 # length-K
    label_image: np.ndarray                # (H, W) int — index into palette
    masks: np.ndarray                      # (K, H, W) uint8 — 0/255


def _sample_pixels(pixels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random-subsample pixel rows for clustering speed."""
    if pixels.shape[0] <= n:
        return pixels
    idx = rng.choice(pixels.shape[0], size=n, replace=False)
    return pixels[idx]


def _auto_pick_k(features: np.ndarray, k_min: int, k_max: int, random_state: int) -> int:
    """Use the SSE 'knee' / elbow method to pick k.

    Picks the k that maximizes the second derivative of inertia — i.e. the
    point where adding a cluster stops paying off.
    """
    k_max = min(k_max, max(features.shape[0] - 1, k_min))
    k_min = max(2, k_min)
    if k_max <= k_min:
        return k_min

    ks = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    for k in ks:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=3,
            batch_size=min(2048, features.shape[0]),
        )
        km.fit(features)
        inertias.append(float(km.inertia_))

    if len(inertias) < 3:
        return ks[0]

    inertias_arr = np.array(inertias)
    norm = (inertias_arr - inertias_arr.min()) / max(inertias_arr.max() - inertias_arr.min(), 1e-9)
    x = np.linspace(0.0, 1.0, len(ks))

    # Distance from each point on the curve to the line from first to last point.
    line_vec = np.array([1.0, norm[-1] - norm[0]])
    line_vec /= np.linalg.norm(line_vec) + 1e-12
    points = np.stack([x, norm], axis=1) - np.array([0.0, norm[0]])
    proj = points @ line_vec
    proj_points = np.outer(proj, line_vec)
    dist = np.linalg.norm(points - proj_points, axis=1)

    return ks[int(np.argmax(dist))]


def quantize(
    image: np.ndarray,
    target_colors: int | None = None,
    config: QuantizeConfig | None = None,
) -> QuantizeResult:
    """Reduce the image palette via k-means in L*a*b* (or RGB) space.

    Args:
        image: (H, W, 3) uint8 RGB array.
        target_colors: explicit k; if None and ``config.auto_detect`` is True,
            pick k via the elbow method.
    """
    cfg = config or QuantizeConfig()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Quantizer expects an (H, W, 3) RGB array.")

    h, w, _ = image.shape
    flat_rgb = image.reshape(-1, 3).astype(np.float64)

    if cfg.color_space == "lab":
        features = rgb_to_lab(flat_rgb)
    else:
        features = flat_rgb.copy()

    rng = np.random.default_rng(cfg.random_state)
    sample = _sample_pixels(features, cfg.sample_pixels, rng)

    if target_colors is None or target_colors <= 0:
        k = (
            _auto_pick_k(sample, cfg.min_colors, cfg.max_colors, cfg.random_state)
            if cfg.auto_detect
            else cfg.min_colors
        )
    else:
        k = int(np.clip(target_colors, cfg.min_colors, cfg.max_colors))

    k = max(1, min(k, sample.shape[0]))

    km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=4)
    km.fit(sample)

    # Predict labels for every pixel in the full image.
    labels = km.predict(features).astype(np.int32)

    # Recompute centroids in RGB space for accurate output colors,
    # regardless of which feature space we clustered in.
    palette_rgb = np.zeros((k, 3), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for c in range(k):
        mask = labels == c
        counts[c] = int(mask.sum())
        if counts[c] > 0:
            palette_rgb[c] = flat_rgb[mask].mean(axis=0)
    palette_rgb = np.clip(np.round(palette_rgb), 0, 255).astype(np.uint8)

    # Drop empty clusters that ended up with zero pixels.
    keep = counts > 0
    if not keep.all():
        old_to_new = -np.ones(k, dtype=np.int32)
        old_to_new[keep] = np.arange(keep.sum())
        labels = old_to_new[labels]
        palette_rgb = palette_rgb[keep]
        k = palette_rgb.shape[0]

    quantized = palette_rgb[labels].reshape(h, w, 3).astype(np.uint8)
    label_image = labels.reshape(h, w)

    masks = np.zeros((k, h, w), dtype=np.uint8)
    for c in range(k):
        masks[c] = np.where(label_image == c, 255, 0).astype(np.uint8)

    palette_hex = [rgb_to_hex(tuple(int(v) for v in rgb)) for rgb in palette_rgb]

    return QuantizeResult(
        quantized_image=quantized,
        palette_rgb=palette_rgb,
        palette_hex=palette_hex,
        label_image=label_image,
        masks=masks,
    )
