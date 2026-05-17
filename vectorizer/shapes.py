"""Geometric primitive fitting: detect circles, ellipses, rectangles, rounded
rectangles, and regular polygons inside binary masks.

For each connected component we try every primitive type, render the candidate
back to a binary mask, compute IoU with the original, and pick the highest-IoU
fit above a configurable threshold. Components that don't fit any primitive
fall back to a regular Bézier-traced ``"path"`` element so we never sacrifice
shape fidelity for perceived cleanliness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np


@dataclass
class SVGElement:
    """A single drawable element inside a color layer.

    ``kind`` chooses the SVG element family; ``attrs`` holds geometry as
    string-valued attributes that map directly to SVG attributes.
    """

    kind: str                       # "circle" | "ellipse" | "rect" | "rounded_rect" | "polygon" | "path"
    attrs: dict[str, str] = field(default_factory=dict)
    iou: float = 1.0                # quality of fit (1.0 for path fallbacks)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class FitConfig:
    iou_threshold: float = 0.95     # minimum IoU to accept a primitive fit
    min_area_px: int = 30           # skip tiny components
    poly_max_vertices: int = 12
    poly_epsilon_frac: float = 0.012  # approxPolyDP epsilon as fraction of perimeter
    enable_circle: bool = True
    enable_ellipse: bool = True
    enable_rect: bool = True
    enable_rounded_rect: bool = True
    enable_polygon: bool = True
    prefer_simpler: bool = True     # circle beats ellipse if both fit equally


# --------------------------- helpers ---------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two binary masks."""
    a_bool = a > 0
    b_bool = b > 0
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    return inter / union if union else 0.0


def _fmt(value: float) -> str:
    """Compact SVG number formatting (trim trailing zeros)."""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _within_canvas(*coords: float, w: int, h: int, tol: float = 1.0) -> bool:
    """True if every coordinate is inside the canvas bounds (with small slack)."""
    return all(-tol <= c <= max(w, h) + tol for c in coords)


# --------------------------- per-primitive fitters ---------------------------


def _fit_circle(mask: np.ndarray, contour: np.ndarray) -> SVGElement | None:
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    if r < 2:
        return None
    rendered = np.zeros_like(mask)
    cv2.circle(rendered, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    iou = _iou(mask, rendered)
    return SVGElement(
        kind="circle",
        attrs={"cx": _fmt(cx), "cy": _fmt(cy), "r": _fmt(r)},
        iou=iou,
    )


def _fit_ellipse(mask: np.ndarray, contour: np.ndarray) -> SVGElement | None:
    if len(contour) < 5:
        return None
    try:
        (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    rx, ry = d1 / 2.0, d2 / 2.0
    if rx < 2 or ry < 2 or not np.isfinite(rx) or not np.isfinite(ry):
        return None
    rendered = np.zeros_like(mask)
    cv2.ellipse(
        rendered,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(rx))), max(1, int(round(ry)))),
        angle, 0, 360, 255, -1,
    )
    iou = _iou(mask, rendered)
    attrs = {"cx": _fmt(cx), "cy": _fmt(cy), "rx": _fmt(rx), "ry": _fmt(ry)}
    # Apply rotation only when meaningful — collapses near-axis-aligned ellipses
    # to plain <ellipse> with no transform.
    if abs(((angle + 90) % 180) - 90) > 1.0:
        attrs["transform"] = f"rotate({_fmt(angle)} {_fmt(cx)} {_fmt(cy)})"
    return SVGElement(kind="ellipse", attrs=attrs, iou=iou)


def _fit_axis_aligned_rect(mask: np.ndarray, contour: np.ndarray) -> SVGElement | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 2 or h < 2:
        return None
    rendered = np.zeros_like(mask)
    cv2.rectangle(rendered, (x, y), (x + w - 1, y + h - 1), 255, -1)
    iou = _iou(mask, rendered)
    return SVGElement(
        kind="rect",
        attrs={"x": _fmt(x), "y": _fmt(y), "width": _fmt(w), "height": _fmt(h)},
        iou=iou,
    )


def _fit_rotated_rect(mask: np.ndarray, contour: np.ndarray) -> SVGElement | None:
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if rw < 2 or rh < 2:
        return None
    rendered = np.zeros_like(mask)
    box = cv2.boxPoints(rect)
    box_int = np.intp(box)
    cv2.fillPoly(rendered, [box_int], 255)
    iou = _iou(mask, rendered)
    # Express as <rect> centered at origin then transformed.
    x = cx - rw / 2
    y = cy - rh / 2
    transform = f"rotate({_fmt(angle)} {_fmt(cx)} {_fmt(cy)})"
    return SVGElement(
        kind="rect",
        attrs={
            "x": _fmt(x),
            "y": _fmt(y),
            "width": _fmt(rw),
            "height": _fmt(rh),
            "transform": transform,
        },
        iou=iou,
    )


def _fit_rounded_rect(mask: np.ndarray, contour: np.ndarray) -> SVGElement | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 6 or h < 6:
        return None
    max_r = min(w, h) // 2
    best: SVGElement | None = None
    # Sample candidate corner radii — coarse-to-fine search is fast enough for logos.
    for r in range(1, max_r + 1, max(1, max_r // 12)):
        rendered = np.zeros_like(mask)
        # Filled rounded rect = a wide bar + a tall bar + four corner discs.
        if w > 2 * r:
            cv2.rectangle(rendered, (x + r, y), (x + w - r - 1, y + h - 1), 255, -1)
        if h > 2 * r:
            cv2.rectangle(rendered, (x, y + r), (x + w - 1, y + h - r - 1), 255, -1)
        cv2.circle(rendered, (x + r, y + r), r, 255, -1)
        cv2.circle(rendered, (x + w - r - 1, y + r), r, 255, -1)
        cv2.circle(rendered, (x + r, y + h - r - 1), r, 255, -1)
        cv2.circle(rendered, (x + w - r - 1, y + h - r - 1), r, 255, -1)
        iou = _iou(mask, rendered)
        if best is None or iou > best.iou:
            best = SVGElement(
                kind="rect",
                attrs={
                    "x": _fmt(x),
                    "y": _fmt(y),
                    "width": _fmt(w),
                    "height": _fmt(h),
                    "rx": _fmt(r),
                    "ry": _fmt(r),
                },
                iou=iou,
            )
    return best


def _fit_polygon(mask: np.ndarray, contour: np.ndarray, cfg: FitConfig) -> SVGElement | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    epsilon = cfg.poly_epsilon_frac * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    n = len(approx)
    if n < 3 or n > cfg.poly_max_vertices:
        return None
    rendered = np.zeros_like(mask)
    cv2.fillPoly(rendered, [approx], 255)
    iou = _iou(mask, rendered)
    points_attr = " ".join(f"{_fmt(p[0][0])},{_fmt(p[0][1])}" for p in approx)
    return SVGElement(kind="polygon", attrs={"points": points_attr}, iou=iou)


# --------------------------- top-level fit ---------------------------


def fit_primitive(
    component_mask: np.ndarray,
    config: FitConfig | None = None,
) -> SVGElement | None:
    """Return the best-fit primitive for a binary connected-component mask.

    Returns ``None`` when no primitive crosses the IoU threshold — the caller
    should then fall back to Bézier tracing for that component.
    """
    cfg = config or FitConfig()
    area = int((component_mask > 0).sum())
    if area < cfg.min_area_px:
        return None

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg.min_area_px:
        return None

    candidates: list[SVGElement] = []
    if cfg.enable_circle:
        c = _fit_circle(component_mask, contour)
        if c:
            candidates.append(c)
    if cfg.enable_ellipse:
        e = _fit_ellipse(component_mask, contour)
        if e:
            candidates.append(e)
    if cfg.enable_rect:
        r = _fit_axis_aligned_rect(component_mask, contour)
        if r:
            candidates.append(r)
        rr = _fit_rotated_rect(component_mask, contour)
        if rr:
            candidates.append(rr)
    if cfg.enable_rounded_rect:
        rr = _fit_rounded_rect(component_mask, contour)
        if rr:
            candidates.append(rr)
    if cfg.enable_polygon:
        p = _fit_polygon(component_mask, contour, cfg)
        if p:
            candidates.append(p)

    if not candidates:
        return None

    # Sort by IoU; among near-ties, prefer simpler primitives.
    candidates.sort(key=lambda s: s.iou, reverse=True)
    best = candidates[0]
    if cfg.prefer_simpler:
        rank = {"circle": 0, "rect": 1, "polygon": 2, "ellipse": 3, "rounded_rect": 1}
        near = [c for c in candidates if best.iou - c.iou < 0.01]
        near.sort(key=lambda s: rank.get(s.kind, 9))
        best = near[0]

    if best.iou < cfg.iou_threshold:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    best.bbox = (x, y, w, h)
    return best


# --------------------------- per-layer decomposition ---------------------------


def decompose_mask_to_components(
    mask: np.ndarray, min_area_px: int = 30
) -> Iterable[tuple[np.ndarray, list[np.ndarray]]]:
    """Yield (outer_component_mask, list_of_hole_contours) for each top-level
    connected component in ``mask``.

    Holes are returned as raw contour arrays so callers can decide whether to
    represent them as nested primitives or fall back to a compound path.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return
    flat_hier = hierarchy[0] if hierarchy is not None else None

    for i, contour in enumerate(contours):
        if flat_hier is not None and flat_hier[i][3] != -1:
            continue  # skip child contours (holes) — they're attached to their parent
        if cv2.contourArea(contour) < min_area_px:
            continue
        # Build a mask of just this component, including any holes it contains.
        comp_mask = np.zeros_like(mask)
        cv2.drawContours(comp_mask, [contour], -1, 255, cv2.FILLED)
        holes: list[np.ndarray] = []
        if flat_hier is not None:
            child = flat_hier[i][2]
            while child != -1:
                hole_contour = contours[child]
                if cv2.contourArea(hole_contour) >= min_area_px:
                    cv2.drawContours(comp_mask, [hole_contour], -1, 0, cv2.FILLED)
                    holes.append(hole_contour)
                child = flat_hier[child][0]
        yield comp_mask, holes
