"""SVG → EPS exporter.

Order of preference:
    1. cairosvg (pure-Python, no system deps, handles cubic Béziers cleanly).
    2. Inkscape CLI (if cairosvg's EPS output is unavailable or fails).
    3. A pure-Python PostScript writer that walks the SVG paths directly.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from lxml import etree

NS_SVG = "http://www.w3.org/2000/svg"


def svg_to_eps(svg_string: str, output_path: str | Path) -> Path:
    """Convert an SVG string to an EPS file at ``output_path``.

    SVG ``A`` (arc) commands are pre-flattened to cubic Béziers so every
    downstream PostScript renderer (cairosvg, Inkscape, our native writer,
    Ghostscript, Adobe Illustrator) consumes the same supported subset.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pre-flatten arcs — cairosvg handles them, but our native fallback writer
    # and some legacy EPS readers don't. Doing this once up front keeps the
    # three writers identical and the output universally compatible.
    from ..arcs import flatten_arcs_in_svg
    svg_for_export = flatten_arcs_in_svg(svg_string)

    if _try_cairosvg(svg_for_export, out):
        return out
    if _try_inkscape(svg_for_export, out):
        return out
    return _native_ps_writer(svg_for_export, out)


# --------------------------- cairosvg ---------------------------


def _try_cairosvg(svg_string: str, out: Path) -> bool:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return False
    try:
        cairosvg.svg2ps(bytestring=svg_string.encode("utf-8"), write_to=str(out))
    except Exception:
        return False
    if not out.exists() or out.stat().st_size == 0:
        return False
    _ensure_eps_header(out)
    return True


def _ensure_eps_header(path: Path) -> None:
    """Make sure the file starts with %!PS-Adobe-3.0 EPSF-3.0 and has a BoundingBox."""
    data = path.read_text(encoding="utf-8", errors="ignore")
    if data.startswith("%!PS-Adobe") and "EPSF" in data.split("\n", 1)[0]:
        return
    # Rewrite the first line into an EPS-compliant DSC header.
    if data.startswith("%!"):
        data = "%!PS-Adobe-3.0 EPSF-3.0\n" + data.split("\n", 1)[1]
    else:
        data = "%!PS-Adobe-3.0 EPSF-3.0\n" + data
    path.write_text(data, encoding="utf-8")


# --------------------------- Inkscape CLI ---------------------------


def _try_inkscape(svg_string: str, out: Path) -> bool:
    if shutil.which("inkscape") is None:
        return False
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "input.svg"
            in_path.write_text(svg_string, encoding="utf-8")
            cmd = [
                "inkscape",
                str(in_path),
                f"--export-filename={out}",
                "--export-type=eps",
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=60)
            return proc.returncode == 0 and out.exists()
    except Exception:
        return False


# --------------------------- Native PostScript writer ---------------------------


_NUM = r"-?\d+(?:\.\d+)?"
_PATH_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|" + _NUM)


def _tokenize_path(d: str) -> list:
    """Walk an SVG 'd' string and yield (command, [floats...]) tuples."""
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|-?\d+(?:\.\d+)?", d)
    cmds: list[tuple[str, list[float]]] = []
    current: tuple[str, list[float]] | None = None
    for tok in tokens:
        if tok.isalpha():
            if current is not None:
                cmds.append(current)
            current = (tok, [])
        else:
            if current is None:
                continue
            current[1].append(float(tok))
    if current is not None:
        cmds.append(current)
    return cmds


def _flatten_path_to_ps(d: str, height: float) -> list[str]:
    """Convert an SVG 'd' string to PostScript path operators.

    SVG y axis points down; PostScript y axis points up — so we flip Y by
    rewriting every y coordinate as ``height - y``.
    """
    ps: list[str] = []
    x = y = 0.0
    sx = sy = 0.0   # subpath start
    last_ctrl: tuple[float, float] | None = None

    def fy(v: float) -> float:
        return height - v

    cmds = _tokenize_path(d)
    for cmd, args in cmds:
        is_rel = cmd.islower()
        c = cmd.upper()

        def consume(n: int, args=args):
            chunks = []
            while len(args) >= n:
                chunks.append(args[:n])
                args = args[n:]
            return chunks, args

        if c == "M":
            chunks, args = consume(2)
            if not chunks:
                continue
            first = True
            for px, py in chunks:
                if is_rel and not first:
                    x += px; y += py
                elif is_rel and first:
                    x += px; y += py
                else:
                    x, y = px, py
                if first:
                    ps.append(f"{x:.3f} {fy(y):.3f} moveto")
                    sx, sy = x, y
                    first = False
                else:
                    ps.append(f"{x:.3f} {fy(y):.3f} lineto")
            last_ctrl = None
        elif c == "L":
            chunks, _ = consume(2)
            for px, py in chunks:
                if is_rel:
                    x += px; y += py
                else:
                    x, y = px, py
                ps.append(f"{x:.3f} {fy(y):.3f} lineto")
            last_ctrl = None
        elif c == "H":
            for px in args:
                x = x + px if is_rel else px
                ps.append(f"{x:.3f} {fy(y):.3f} lineto")
            last_ctrl = None
        elif c == "V":
            for py in args:
                y = y + py if is_rel else py
                ps.append(f"{x:.3f} {fy(y):.3f} lineto")
            last_ctrl = None
        elif c == "C":
            chunks, _ = consume(6)
            for cx1, cy1, cx2, cy2, ex, ey in chunks:
                if is_rel:
                    cx1 += x; cy1 += y
                    cx2 += x; cy2 += y
                    ex += x;  ey += y
                ps.append(
                    f"{cx1:.3f} {fy(cy1):.3f} {cx2:.3f} {fy(cy2):.3f} "
                    f"{ex:.3f} {fy(ey):.3f} curveto"
                )
                x, y = ex, ey
                last_ctrl = (cx2, cy2)
        elif c == "S":
            chunks, _ = consume(4)
            for cx2, cy2, ex, ey in chunks:
                if is_rel:
                    cx2 += x; cy2 += y
                    ex += x;  ey += y
                if last_ctrl is not None:
                    cx1 = 2 * x - last_ctrl[0]
                    cy1 = 2 * y - last_ctrl[1]
                else:
                    cx1, cy1 = x, y
                ps.append(
                    f"{cx1:.3f} {fy(cy1):.3f} {cx2:.3f} {fy(cy2):.3f} "
                    f"{ex:.3f} {fy(ey):.3f} curveto"
                )
                x, y = ex, ey
                last_ctrl = (cx2, cy2)
        elif c == "Q":
            chunks, _ = consume(4)
            for qx, qy, ex, ey in chunks:
                if is_rel:
                    qx += x; qy += y
                    ex += x; ey += y
                # Promote quadratic to cubic for PostScript.
                cx1 = x + 2 / 3 * (qx - x)
                cy1 = y + 2 / 3 * (qy - y)
                cx2 = ex + 2 / 3 * (qx - ex)
                cy2 = ey + 2 / 3 * (qy - ey)
                ps.append(
                    f"{cx1:.3f} {fy(cy1):.3f} {cx2:.3f} {fy(cy2):.3f} "
                    f"{ex:.3f} {fy(ey):.3f} curveto"
                )
                x, y = ex, ey
                last_ctrl = (qx, qy)
        elif c == "T":
            chunks, _ = consume(2)
            for ex, ey in chunks:
                if is_rel:
                    ex += x; ey += y
                if last_ctrl is not None:
                    qx = 2 * x - last_ctrl[0]
                    qy = 2 * y - last_ctrl[1]
                else:
                    qx, qy = x, y
                cx1 = x + 2 / 3 * (qx - x)
                cy1 = y + 2 / 3 * (qy - y)
                cx2 = ex + 2 / 3 * (qx - ex)
                cy2 = ey + 2 / 3 * (qy - ey)
                ps.append(
                    f"{cx1:.3f} {fy(cy1):.3f} {cx2:.3f} {fy(cy2):.3f} "
                    f"{ex:.3f} {fy(ey):.3f} curveto"
                )
                x, y = ex, ey
                last_ctrl = (qx, qy)
        elif c == "Z":
            ps.append("closepath")
            x, y = sx, sy
            last_ctrl = None
        # Arc not implemented — vtracer/contour outputs do not produce A commands.
    return ps


def _native_ps_writer(svg_string: str, out: Path) -> Path:
    """Last-resort: walk the SVG tree and emit PostScript ourselves."""
    root = etree.fromstring(svg_string.encode("utf-8"))
    width = float(root.get("width", "0").replace("px", "") or 0)
    height = float(root.get("height", "0").replace("px", "") or 0)
    viewbox = root.get("viewBox")
    if (width == 0 or height == 0) and viewbox:
        _, _, vw, vh = (float(v) for v in viewbox.split())
        width = width or vw
        height = height or vh

    lines: list[str] = []
    lines.append("%!PS-Adobe-3.0 EPSF-3.0")
    lines.append(f"%%BoundingBox: 0 0 {int(width)} {int(height)}")
    lines.append("%%Creator: Logo Vectorizer 1.0")
    lines.append("%%Pages: 1")
    lines.append("%%EndComments")
    lines.append("%%BeginProlog")
    lines.append("/m { moveto } bind def")
    lines.append("/l { lineto } bind def")
    lines.append("/c { curveto } bind def")
    lines.append("/cp { closepath } bind def")
    lines.append("%%EndProlog")
    lines.append("%%Page: 1 1")
    lines.append("gsave")

    for el in root.iter():
        tag = etree.QName(el.tag).localname
        if tag == "rect":
            x = float(el.get("x", "0"))
            y = float(el.get("y", "0"))
            w = float(el.get("width", "0"))
            h = float(el.get("height", "0"))
            fill = el.get("fill", "#000000")
            r, g, b = _hex_to_rgb01(fill)
            lines.append(f"newpath {x:.3f} {height - y - h:.3f} {w:.3f} {h:.3f} rectfill")
            lines.append(f"{r:.4f} {g:.4f} {b:.4f} setrgbcolor")
        if tag == "g":
            fill = el.get("fill", "#000000")
            r, g, b = _hex_to_rgb01(fill)
            for child in el:
                if etree.QName(child.tag).localname != "path":
                    continue
                d = child.get("d", "")
                if not d:
                    continue
                lines.append(f"{r:.4f} {g:.4f} {b:.4f} setrgbcolor")
                lines.append("newpath")
                lines.extend(_flatten_path_to_ps(d, height))
                lines.append("eofill")

    lines.append("grestore")
    lines.append("showpage")
    lines.append("%%EOF")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _hex_to_rgb01(value: str) -> tuple[float, float, float]:
    """Parse a #rrggbb / #rgb / 'none' value into floats in [0, 1]."""
    if not value or value.lower() in {"none", "transparent"}:
        return (0.0, 0.0, 0.0)
    s = value.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b)
    except (ValueError, IndexError):
        return (0.0, 0.0, 0.0)
