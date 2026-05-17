"""SVG path-string helpers used by the tracer/smoother/assembler chain."""
from __future__ import annotations

import re

from lxml import etree

NS_SVG = "http://www.w3.org/2000/svg"
NS_INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
NS_XLINK = "http://www.w3.org/1999/xlink"

NSMAP = {
    None: NS_SVG,
    "inkscape": NS_INKSCAPE,
    "xlink": NS_XLINK,
}


def extract_paths_from_svg(svg_string: str) -> list[str]:
    """Return every <path d="..."> 'd' attribute found inside an SVG string."""
    if not svg_string or "<path" not in svg_string:
        return []
    try:
        root = etree.fromstring(svg_string.encode() if isinstance(svg_string, str) else svg_string)
    except etree.XMLSyntaxError:
        return re.findall(r'd="([^"]+)"', svg_string)
    paths: list[str] = []
    for el in root.iter():
        if etree.QName(el.tag).localname == "path":
            d = el.get("d")
            if d:
                paths.append(d)
    return paths


def round_path_coords(d: str, precision: int = 2) -> str:
    """Round every floating-point number in an SVG path 'd' string."""
    def _fmt(match: re.Match) -> str:
        val = float(match.group(0))
        if precision <= 0:
            return f"{int(round(val))}"
        formatted = f"{val:.{precision}f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"

    return re.sub(r"-?\d+\.\d+|-?\d+", _fmt, d)


def estimate_svg_bbox(d: str) -> tuple[float, float, float, float] | None:
    """Best-effort bounding box from a path 'd' string."""
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
    if not nums or len(nums) < 2:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


# Path commands whose arguments are (x, y) coordinate pairs in absolute form.
_PAIR_COMMANDS = set("MLT")
# Cubic Bézier: three (x, y) pairs.
_CUBIC_COMMANDS = set("C")
# Smooth cubic and quadratic: two (x, y) pairs.
_TWO_PAIR_COMMANDS = set("SQ")
# Horizontal/Vertical line: single coordinate along one axis.
_H_COMMANDS = set("H")
_V_COMMANDS = set("V")
# Arc: rx, ry, rotation, large-arc, sweep, x, y — only x,y get translated.
_ARC_COMMANDS = set("A")


def translate_path(d: str, dx: float, dy: float) -> str:
    """Apply a translation to every absolute coordinate in an SVG 'd' string.

    Lowercase (relative) commands are unaffected since translation does not
    change relative offsets. The translated path is functionally identical
    to wrapping the original in ``<g transform="translate(dx,dy)">``.
    """
    if dx == 0 and dy == 0:
        return d

    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", d)
    out: list[str] = []
    current_cmd: str | None = None
    pending: list[float] = []

    def flush(cmd: str, nums: list[float]) -> None:
        if not cmd:
            return
        upper = cmd.upper()
        is_abs = cmd.isupper()
        out.append(cmd)

        if upper == "Z" or not nums:
            return

        # Determine the stride and which slots are (x, y).
        def emit_pair(x: float, y: float) -> None:
            if is_abs:
                out.append(_fmt(x + dx))
                out.append(_fmt(y + dy))
            else:
                out.append(_fmt(x))
                out.append(_fmt(y))

        i = 0
        n = len(nums)

        if upper in _PAIR_COMMANDS:
            # M may be followed by implicit L pairs after the first move.
            while i + 1 < n:
                emit_pair(nums[i], nums[i + 1])
                i += 2
        elif upper == "L":
            while i + 1 < n:
                emit_pair(nums[i], nums[i + 1])
                i += 2
        elif upper in _CUBIC_COMMANDS:
            while i + 5 < n:
                emit_pair(nums[i], nums[i + 1])
                emit_pair(nums[i + 2], nums[i + 3])
                emit_pair(nums[i + 4], nums[i + 5])
                i += 6
        elif upper in _TWO_PAIR_COMMANDS:
            while i + 3 < n:
                emit_pair(nums[i], nums[i + 1])
                emit_pair(nums[i + 2], nums[i + 3])
                i += 4
        elif upper in _H_COMMANDS:
            for v in nums:
                out.append(_fmt(v + dx if is_abs else v))
        elif upper in _V_COMMANDS:
            for v in nums:
                out.append(_fmt(v + dy if is_abs else v))
        elif upper in _ARC_COMMANDS:
            while i + 6 < n:
                # rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y
                out.extend(_fmt(v) for v in nums[i:i + 5])
                emit_pair(nums[i + 5], nums[i + 6])
                i += 7
        else:
            out.extend(_fmt(v) for v in nums)

    for tok in tokens:
        if tok.isalpha():
            flush(current_cmd, pending)
            current_cmd = tok
            pending = []
        else:
            pending.append(float(tok))
    flush(current_cmd, pending)

    # Re-join with spaces; collapse runs of spaces.
    return " ".join(out)


def _fmt(value: float) -> str:
    """Format a coordinate compactly, preserving precision but trimming zeros."""
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.3f}".rstrip("0").rstrip(".")
