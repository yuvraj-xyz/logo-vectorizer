"""SVG → Adobe Illustrator (.ai) exporter.

The .ai container is an EPS file wrapped with Illustrator-specific DSC comments.
Modern Illustrator opens any compliant EPS — we generate one and prepend the
AI marker block so the file is recognized natively.
"""
from __future__ import annotations

import getpass
import re
from datetime import datetime
from pathlib import Path

from .eps import svg_to_eps


def svg_to_ai(
    svg_string: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Convert an SVG string to a .ai file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Generate the EPS body to a temporary path next to the output.
    eps_temp = out.with_suffix(".__eps.tmp")
    try:
        svg_to_eps(svg_string, eps_temp)
        body = eps_temp.read_text(encoding="utf-8", errors="ignore")
    finally:
        if eps_temp.exists():
            eps_temp.unlink()

    title = title or out.stem
    try:
        user = getpass.getuser()
    except Exception:
        user = "user"

    bbox = _extract_bbox(body)

    ai_header = "\n".join(
        [
            "%!PS-Adobe-3.0",
            "%%Creator: Logo Vectorizer 1.0 / Adobe Illustrator(R) 16.0",
            f"%%For: {user}",
            f"%%Title: {title}.ai",
            f"%%CreationDate: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
            "%%DocumentData: Clean7Bit",
            f"%%BoundingBox: {bbox}",
            f"%%HiResBoundingBox: {bbox}",
            "%%CropBox: " + bbox,
            "%%LanguageLevel: 2",
            "%AI8_CreatorVersion: 16.0.0",
            "%AI9_PrintingDataBegin",
            "%%EndComments",
        ]
    )

    ai_trailer = "\n".join(
        [
            "%AI9_PrintingDataEnd",
            "%%Trailer",
            "%%EOF",
        ]
    )

    cleaned_body = _strip_existing_dsc(body)
    document = ai_header + "\n" + cleaned_body.strip() + "\n" + ai_trailer + "\n"
    out.write_text(document, encoding="utf-8")
    return out


def _extract_bbox(eps_body: str) -> str:
    match = re.search(r"^%%BoundingBox:\s*(.+)$", eps_body, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "0 0 100 100"


def _strip_existing_dsc(eps_body: str) -> str:
    """Strip the leading DSC comments — we're replacing them with AI-specific ones."""
    lines = eps_body.splitlines()
    out: list[str] = []
    skipping = True
    for line in lines:
        if skipping:
            if line.startswith("%!") or line.startswith("%%") or line.startswith("%A"):
                continue
            skipping = False
        out.append(line)
    # Also strip a trailing %%EOF; the AI trailer adds its own.
    while out and out[-1].strip() in {"%%EOF", ""}:
        out.pop()
    return "\n".join(out)
