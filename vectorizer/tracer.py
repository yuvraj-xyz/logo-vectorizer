"""Per-layer raster-to-SVG tracing using vtracer (with a manual-contour fallback)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .separator import Layer
from .shapes import FitConfig, SVGElement, decompose_mask_to_components, fit_primitive
from .utils.svg_utils import translate_path

try:
    import vtracer  # type: ignore

    HAS_VTRACER = True
except Exception:  # pragma: no cover - environmental
    vtracer = None  # type: ignore[assignment]
    HAS_VTRACER = False

try:
    import potrace  # type: ignore

    HAS_POTRACE = True
except Exception:  # pragma: no cover - environmental
    potrace = None  # type: ignore[assignment]
    HAS_POTRACE = False


@dataclass
class TraceConfig:
    backend: str = "potrace"            # "potrace" | "vtracer" | "contour"
    spline_mode: bool = True            # cubic Bézier vs polygon
    corner_threshold: int = 75          # vtracer: higher = smoother curves, fewer corners
    filter_speckle: int = 8             # min island size (larger = drops AA noise)
    color_precision: int = 6
    layer_difference: int = 16
    length_threshold: float = 4.0
    max_iterations: int = 10
    splice_threshold: int = 60          # higher = more aggressive curve splicing for smoother result
    path_precision: int = 3
    hierarchical: str = "stacked"       # "stacked" or "cutout"
    fallback_on_error: bool = True
    # potrace-specific knobs (the gold-standard open-source Bézier tracer)
    potrace_turdsize: int = 2           # remove islands smaller than this many pixels
    potrace_alphamax: float = 1.0       # 0..1.334 — higher = smoother curves
    potrace_opticurve: bool = True      # post-trace curve optimization
    potrace_opttolerance: float = 0.2   # opticurve tolerance — lower = closer fit, higher = smoother


@dataclass
class TracedLayer:
    layer: Layer
    paths: list[str]                    # legacy: SVG 'd' strings (one per top-level path)
    elements: list[SVGElement] = None   # type: ignore[assignment]
    # ``elements`` is the new representation: a mix of primitive elements
    # (circle / ellipse / rect / polygon) and path elements. When set, the
    # assembler prefers it over ``paths``. ``None`` means "legacy path-only".

    def __post_init__(self) -> None:
        if self.elements is None:
            self.elements = [SVGElement(kind="path", attrs={"d": d}) for d in self.paths]


def _save_mask_png(mask: np.ndarray, path: str) -> None:
    """Save a binary mask (uint8 0/255) as a black-on-white PNG.

    vtracer's binary colormode wants black foreground on white background.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    # vtracer 'binary' colormode: black = foreground shape, white = background.
    img = np.where(mask > 0, 0, 255).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1)
    Image.fromarray(rgb).save(path, "PNG")


def _trace_with_vtracer(mask: np.ndarray, cfg: TraceConfig) -> list[str]:
    """Call into vtracer and return the path 'd' attributes it emits.

    The vtracer python binding has a known issue on some Python versions
    where keyword arguments crash the interpreter — so we use positional args.
    """
    if not HAS_VTRACER:
        raise RuntimeError("vtracer is not installed")

    mode = "spline" if cfg.spline_mode else "polygon"

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "mask.png")
        out_path = os.path.join(td, "out.svg")
        _save_mask_png(mask, in_path)

        # IMPORTANT: positional args only — keyword-arg form crashes on Py3.14.
        vtracer.convert_image_to_svg_py(
            in_path,
            out_path,
            "binary",
            cfg.hierarchical,
            mode,
            cfg.filter_speckle,
            cfg.color_precision,
            cfg.layer_difference,
            cfg.corner_threshold,
            cfg.length_threshold,
            cfg.max_iterations,
            cfg.splice_threshold,
            cfg.path_precision,
        )

        with open(out_path, "r", encoding="utf-8") as f:
            svg_str = f.read()

    return _extract_foreground_paths(svg_str)


def _extract_foreground_paths(svg_str: str) -> list[str]:
    """Return the foreground (non-white) paths with their translate baked in.

    vtracer emits each path with a ``transform="translate(dx,dy)"`` attribute
    and produces a canvas-spanning white background path that we don't want.
    We strip the background and bake the translate into the path coordinates
    so callers get drop-in 'd' strings.
    """
    import re

    # Match: d="...", fill="...", optional transform="translate(...)" — in any
    # order on a <path> element.
    pattern = re.compile(
        r'<path\b([^>]*?)/>',
        re.DOTALL,
    )
    foreground: list[str] = []
    for match in pattern.finditer(svg_str):
        attrs = match.group(1)
        d_match = re.search(r'd="([^"]+)"', attrs)
        fill_match = re.search(r'fill="([^"]+)"', attrs)
        tx_match = re.search(r'transform="translate\(\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*\)"', attrs)
        if not d_match:
            continue
        fill = (fill_match.group(1) if fill_match else "").lower()
        if fill in {"#fff", "#ffffff", "white", "none"}:
            continue
        d = d_match.group(1)
        if tx_match:
            dx = float(tx_match.group(1))
            dy = float(tx_match.group(2))
            d = translate_path(d, dx, dy)
        foreground.append(d)
    return foreground


def _trace_with_potrace(mask: np.ndarray, cfg: TraceConfig) -> list[str]:
    """Trace a binary mask with potrace and return SVG ``d`` strings.

    Potrace is the gold-standard open-source Bézier tracer (the engine behind
    classic Vector Magic). It produces noticeably cleaner curves than the
    contour fallback, especially for organic shapes and text.

    Potrace's API always traces the "background" boundary too — for a fully
    in-canvas shape that's a frame curve touching all four canvas edges. We
    detect and skip that frame by bbox.
    """
    if not HAS_POTRACE:
        raise RuntimeError("potrace is not installed")

    h, w = mask.shape[:2]
    bool_mask = mask > 0
    bitmap = potrace.Bitmap(bool_mask)
    path = bitmap.trace(
        turdsize=cfg.potrace_turdsize,
        alphamax=cfg.potrace_alphamax,
        opticurve=cfg.potrace_opticurve,
        opttolerance=cfg.potrace_opttolerance,
    )

    out: list[str] = []
    for curve in path:
        if _is_canvas_frame_curve(curve, w, h):
            continue
        d = _potrace_curve_to_d(curve)
        if d:
            out.append(d)
    return out


def _is_canvas_frame_curve(curve, w: int, h: int) -> bool:
    """True if the curve's bbox is essentially the whole canvas."""
    xs, ys = [curve.start_point.x], [curve.start_point.y]
    for seg in curve:
        xs.append(seg.end_point.x)
        ys.append(seg.end_point.y)
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    return bbox_w >= 0.96 * w and bbox_h >= 0.96 * h


def _potrace_curve_to_d(curve) -> str:
    """Convert a single potrace ``Curve`` into an SVG ``d`` string.

    Potrace ``BezierSegment`` carries cubic control points (c1, c2, end_point).
    Potrace ``CornerSegment`` is a polygonal vertex pair (c, end_point) — we
    emit it as two straight line segments meeting at the corner ``c``.
    """
    sp = curve.start_point
    parts = [f"M{sp.x:.3f} {sp.y:.3f}"]
    for seg in curve:
        cls = type(seg).__name__
        if cls == "BezierSegment":
            c1, c2, ep = seg.c1, seg.c2, seg.end_point
            parts.append(f"C{c1.x:.3f} {c1.y:.3f} {c2.x:.3f} {c2.y:.3f} {ep.x:.3f} {ep.y:.3f}")
        else:  # CornerSegment
            c, ep = seg.c, seg.end_point
            parts.append(f"L{c.x:.3f} {c.y:.3f}")
            parts.append(f"L{ep.x:.3f} {ep.y:.3f}")
    parts.append("Z")
    return " ".join(parts)


def _trace_with_contour(mask: np.ndarray, cfg: TraceConfig) -> list[str]:
    """Pure-OpenCV fallback: trace contours and emit polyline path strings.

    The smoother stage will spline-fit these into cubic Béziers, so the output
    of this backend remains a high-quality vector after smoothing.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    if not contours:
        return []

    paths: list[str] = []
    if hierarchy is None:
        hierarchy = np.array([[[-1, -1, -1, -1]] * len(contours)])
    flat_hier = hierarchy[0]

    # Group children with their parents to form compound paths with holes.
    parent_to_children: dict[int, list[int]] = {}
    for i, h in enumerate(flat_hier):
        parent = int(h[3])
        if parent == -1:
            parent_to_children.setdefault(i, [])
        else:
            parent_to_children.setdefault(parent, []).append(i)

    min_pts = 3
    for parent_idx, children in parent_to_children.items():
        outer = contours[parent_idx]
        if len(outer) < min_pts:
            continue
        d_parts = [_contour_to_path(outer)]
        for c_idx in children:
            child = contours[c_idx]
            if len(child) >= min_pts:
                d_parts.append(_contour_to_path(child))
        paths.append(" ".join(d_parts))

    return paths


def _contour_to_path(contour: np.ndarray) -> str:
    """Render a contour (Nx1x2) as an SVG polyline subpath."""
    pts = contour.reshape(-1, 2)
    if pts.shape[0] == 0:
        return ""
    parts = [f"M{pts[0,0]} {pts[0,1]}"]
    for x, y in pts[1:]:
        parts.append(f"L{x} {y}")
    parts.append("Z")
    return " ".join(parts)


def trace_layer(layer: Layer, config: TraceConfig | None = None) -> TracedLayer:
    """Convert a single ``Layer`` into one or more SVG path strings."""
    cfg = config or TraceConfig()
    paths = _trace_single_component(layer.mask, cfg)
    return TracedLayer(layer=layer, paths=paths)


def trace_layers(layers: list[Layer], config: TraceConfig | None = None) -> list[TracedLayer]:
    """Trace every layer in order, preserving paint order."""
    return [trace_layer(layer, config) for layer in layers]


def decompose_layer(
    layer: Layer,
    trace_config: TraceConfig | None = None,
    fit_config: FitConfig | None = None,
) -> TracedLayer:
    """Per-connected-component primitive fitting with Bézier fallback.

    For each top-level connected component in the layer's mask we attempt to
    fit a geometric primitive (circle, ellipse, rect, rounded rect, polygon).
    Components that don't fit any primitive above the IoU threshold are traced
    as smoothed Bézier paths so we keep shape fidelity for organic curves.

    Components that contain holes fall back to a compound path because nested
    primitives would not preserve correct paint order without extra masking.
    """
    tcfg = trace_config or TraceConfig()
    fcfg = fit_config or FitConfig()

    elements: list[SVGElement] = []
    legacy_paths: list[str] = []

    for comp_mask, holes in decompose_mask_to_components(layer.mask, fcfg.min_area_px):
        if not holes:
            prim = fit_primitive(comp_mask, fcfg)
            if prim is not None:
                elements.append(prim)
                continue

        # Fall back to tracing this single component as a Bézier path.
        # Build a tiny "trace-only" layer mask for this component and run
        # the existing tracer on it.
        comp_layer_mask = comp_mask
        if holes:
            # Bake holes back in so the tracer produces a compound path with
            # the correct interior cutouts.
            comp_layer_mask = comp_mask.copy()
            for hole in holes:
                cv2.drawContours(comp_layer_mask, [hole], -1, 0, cv2.FILLED)

        traced_paths = _trace_single_component(comp_layer_mask, tcfg)
        for d in traced_paths:
            if d:
                elements.append(SVGElement(kind="path", attrs={"d": d}, iou=1.0))
                legacy_paths.append(d)

    return TracedLayer(layer=layer, paths=legacy_paths, elements=elements)


def _trace_single_component(mask: np.ndarray, cfg: TraceConfig) -> list[str]:
    """Trace one connected-component mask with the configured backend.

    Resolution: ``potrace`` → ``vtracer`` → ``contour``. When the configured
    backend isn't available or crashes, fall back to the next-best option
    rather than failing the whole pipeline.
    """
    backend = cfg.backend.lower()

    if backend == "potrace" and HAS_POTRACE:
        try:
            return _trace_with_potrace(mask, cfg)
        except Exception:
            if not cfg.fallback_on_error:
                raise
            # Fall through to vtracer.
            backend = "vtracer"

    if backend == "vtracer" and HAS_VTRACER:
        try:
            return _trace_with_vtracer(mask, cfg)
        except Exception:
            if not cfg.fallback_on_error:
                raise
            return _trace_with_contour(mask, cfg)

    return _trace_with_contour(mask, cfg)


def decompose_layers(
    layers: list[Layer],
    trace_config: TraceConfig | None = None,
    fit_config: FitConfig | None = None,
) -> list[TracedLayer]:
    """Decompose every layer into a mix of primitive + path elements."""
    return [decompose_layer(layer, trace_config, fit_config) for layer in layers]
