"""Generate a deterministic sample logo PNG for smoke testing the pipeline."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_sample_logo(output_path: str | Path = "examples/sample_logo.png", size: int = 512) -> Path:
    """Render a flat-color logo with a circle, rounded rectangle, and triangle."""
    img = Image.new("RGB", (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Outer rounded rectangle in deep indigo.
    pad = size // 12
    draw.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=size // 10,
        fill=(26, 26, 46),
    )

    # Circle in coral.
    cx, cy = size // 2, size // 2 - size // 16
    r = size // 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(233, 69, 96))

    # Triangle in pale yellow.
    triangle = [
        (cx, cy - r // 2),
        (cx - r // 2, cy + r // 2),
        (cx + r // 2, cy + r // 2),
    ]
    draw.polygon(triangle, fill=(255, 215, 100))

    # Bottom bar in teal.
    draw.rectangle(
        (pad * 2, size - pad * 3, size - pad * 2, size - pad * 2),
        fill=(0, 173, 181),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


if __name__ == "__main__":
    p = make_sample_logo()
    print(f"Wrote {p}")
