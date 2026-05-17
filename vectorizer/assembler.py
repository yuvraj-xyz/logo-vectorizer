"""SVG assembler: collect smoothed traced layers into a single SVG document."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree

from .smoother import TracedLayer
from .utils.svg_utils import NS_INKSCAPE, NS_SVG


@dataclass
class AssembleConfig:
    fill_rule: str = "evenodd"
    stroke: str = "none"
    include_metadata: bool = True
    creator: str = "Logo Vectorizer 1.0"
    pretty: bool = True
    background_color: str | None = None     # if set, force a background <rect>
    extra_inkscape_attrs: bool = True


@dataclass
class AssembleResult:
    svg_string: str
    width: int
    height: int
    layer_count: int
    palette: list[str] = field(default_factory=list)


def assemble_svg(
    traced_layers: list[TracedLayer],
    width: int,
    height: int,
    source_filename: str | None = None,
    config: AssembleConfig | None = None,
) -> AssembleResult:
    """Assemble per-layer paths into a complete SVG document.

    Each color becomes a single ``<g>`` group with the appropriate ``fill``
    and an ``inkscape:label`` for friendly display in vector editors.
    """
    cfg = config or AssembleConfig()
    nsmap = {None: NS_SVG, "inkscape": NS_INKSCAPE}

    svg = etree.Element(
        f"{{{NS_SVG}}}svg",
        nsmap=nsmap,
        attrib={
            "version": "1.1",
            "viewBox": f"0 0 {width} {height}",
            "width": str(width),
            "height": str(height),
        },
    )

    if cfg.include_metadata:
        _append_metadata(svg, cfg.creator, source_filename)

    if cfg.background_color:
        bg = etree.SubElement(
            svg,
            f"{{{NS_SVG}}}rect",
            attrib={
                "id": "background",
                "x": "0",
                "y": "0",
                "width": str(width),
                "height": str(height),
                "fill": cfg.background_color,
            },
        )
        if cfg.extra_inkscape_attrs:
            bg.set(f"{{{NS_INKSCAPE}}}label", "Background")

    palette: list[str] = []

    for traced in traced_layers:
        elements = traced.elements if traced.elements else [
            type("E", (), {"kind": "path", "attrs": {"d": d}})() for d in traced.paths  # type: ignore[misc]
        ]
        if not elements:
            continue
        layer = traced.layer
        palette.append(layer.color_hex)

        attrib = {
            "id": layer.label,
            "fill": layer.color_hex,
            "stroke": cfg.stroke,
            "fill-rule": cfg.fill_rule,
        }
        group = etree.SubElement(svg, f"{{{NS_SVG}}}g", attrib=attrib)
        if cfg.extra_inkscape_attrs:
            human = _label_for(layer.color_hex, layer.is_background)
            group.set(f"{{{NS_INKSCAPE}}}label", human)
            group.set(f"{{{NS_INKSCAPE}}}groupmode", "layer")

        # Replace the background layer's traced shapes with a single full-canvas
        # rect. This guarantees the entire canvas is painted, so any hairline
        # gap between adjacent foreground shapes shows the correct background
        # color instead of transparency.
        if layer.is_background:
            etree.SubElement(
                group,
                f"{{{NS_SVG}}}rect",
                attrib={"x": "0", "y": "0", "width": str(width), "height": str(height)},
            )
            continue

        for el in elements:
            _append_element(group, el)

    svg_bytes = etree.tostring(
        svg,
        pretty_print=cfg.pretty,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=False,
    )

    return AssembleResult(
        svg_string=svg_bytes.decode("utf-8"),
        width=width,
        height=height,
        layer_count=len(palette),
        palette=palette,
    )


_PRIMITIVE_TAGS = {
    "circle": "circle",
    "ellipse": "ellipse",
    "rect": "rect",
    "polygon": "polygon",
    "polyline": "polyline",
    "path": "path",
}


def _append_element(group, element) -> None:
    """Append one drawable element (primitive or path) to a layer group.

    Falls back to ``<path>`` for unknown kinds. ``element`` is duck-typed: it
    only needs ``.kind`` and ``.attrs``.
    """
    kind = getattr(element, "kind", "path")
    attrs = dict(getattr(element, "attrs", {}) or {})
    tag = _PRIMITIVE_TAGS.get(kind, "path")
    if tag == "path" and "d" not in attrs:
        return
    etree.SubElement(group, f"{{{NS_SVG}}}{tag}", attrib=attrs)


def _append_metadata(svg, creator: str, source_filename: str | None) -> None:
    """Embed a tiny RDF-style metadata block (compatible with Inkscape)."""
    metadata = etree.SubElement(svg, f"{{{NS_SVG}}}metadata")
    desc = etree.SubElement(svg, f"{{{NS_SVG}}}desc")
    desc.text = (
        f"Created by {escape(creator)} on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        + (f" from {escape(source_filename)}" if source_filename else "")
    )
    title = etree.SubElement(metadata, f"{{{NS_SVG}}}title")
    title.text = Path(source_filename).stem if source_filename else "Vectorized Logo"


def _label_for(hex_color: str, is_background: bool) -> str:
    base = hex_color.lstrip("#").upper()
    suffix = " (background)" if is_background else ""
    return f"#{base}{suffix}"


def save_svg(svg_string: str, output_path: str | Path) -> Path:
    """Write an SVG string to disk and return the path."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(svg_string, encoding="utf-8")
    return p
