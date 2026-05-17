"""Command-line entry point for the Logo Vectorizer."""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import click

from .pipeline import PipelineConfig, vectorize
from .preprocessor import PreprocessConfig
from .quantizer import QuantizeConfig
from .separator import SeparatorConfig
from .smoother import SmoothConfig
from .tracer import TraceConfig


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Convert a raster logo to vector (SVG, EPS, AI).",
)
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_dir",
    default="./output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory.",
    show_default=True,
)
@click.option(
    "-f",
    "--format",
    "formats",
    default="svg,eps,ai",
    help="Comma-separated output formats: svg, eps, ai.",
    show_default=True,
)
@click.option(
    "-c",
    "--colors",
    "colors",
    default=0,
    type=click.IntRange(0, 64),
    help="Max colors (2-32). 0 = auto-detect.",
    show_default=True,
)
@click.option(
    "--smooth",
    "smoothing",
    default=1.5,
    type=click.FloatRange(0.0, 5.0),
    help="Curve smoothing (0.0-5.0). Drives both anti-alias mask smoothing and post-trace spline fitting.",
    show_default=True,
)
@click.option("--no-denoise", is_flag=True, help="Skip preprocessing denoising.")
@click.option(
    "--backend",
    type=click.Choice(["potrace", "vtracer", "contour"], case_sensitive=False),
    default="potrace",
    show_default=True,
    help="Trace backend. potrace = highest quality; vtracer = fast; contour = pure OpenCV fallback.",
)
@click.option(
    "--no-shapes",
    is_flag=True,
    help="Disable geometric primitive fitting (force pure Bézier tracing).",
)
@click.option(
    "--shape-fit",
    type=click.FloatRange(0.80, 1.00),
    default=0.95,
    show_default=True,
    help="Minimum IoU for a primitive fit to be accepted (otherwise falls back to Bézier).",
)
@click.option("--preview", is_flag=True, help="Open the SVG output in your browser.")
@click.option("--verbose", is_flag=True, help="Show per-layer debug info.")
def main(
    input_image: Path,
    output_dir: Path,
    formats: str,
    colors: int,
    smoothing: float,
    no_denoise: bool,
    backend: str,
    no_shapes: bool,
    shape_fit: float,
    preview: bool,
    verbose: bool,
) -> None:
    """CLI entrypoint — see ``--help`` for options."""
    fmts = [f.strip().lower() for f in formats.split(",") if f.strip()]
    invalid = [f for f in fmts if f not in {"svg", "eps", "ai"}]
    if invalid:
        click.echo(f"Unsupported format(s): {', '.join(invalid)}", err=True)
        sys.exit(2)

    from .shapes import FitConfig

    config = PipelineConfig(
        preprocess=PreprocessConfig(denoise=not no_denoise),
        quantize=QuantizeConfig(),
        separate=SeparatorConfig(mask_smoothing=max(0.8, smoothing)),
        trace=TraceConfig(backend=backend.lower()),
        fit=FitConfig(iou_threshold=shape_fit),
        smooth=SmoothConfig(spline_smoothing=smoothing),
        fit_shapes=not no_shapes,
    )

    def progress(stage: str, pct: float) -> None:
        bar_len = 24
        filled = int(round(pct * bar_len))
        bar = "#" * filled + "-" * (bar_len - filled)
        click.echo(f"\r  [{bar}] {pct*100:5.1f}%  {stage:<11}", nl=False)

    try:
        result = vectorize(
            input_image,
            output_dir=output_dir,
            formats=fmts,
            target_colors=colors or None,
            config=config,
            progress=progress,
        )
    except Exception as exc:  # pragma: no cover - top-level UX
        click.echo("\n")
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    click.echo("")
    click.echo("")

    click.secho(f"Done - {result.assemble.layer_count} colors traced", fg="green")
    click.echo(f"  Palette: {', '.join(result.assemble.palette)}")
    click.echo(f"  Output : {output_dir}")
    for fmt, path in result.output_paths.items():
        click.echo(f"   {fmt.upper():<4} {path}")

    if verbose:
        click.echo("\n  Stage timings:")
        for stage, secs in result.timings.items():
            click.echo(f"    {stage:<10} {secs*1000:7.1f} ms")
        click.echo(f"\n  Source size : {result.preprocess.original_size}")
        click.echo(f"  Working size: {result.preprocess.image.shape[1]}x{result.preprocess.image.shape[0]}")
        for layer in result.quantize.palette_hex:
            click.echo(f"    color {layer}")

    if preview and "svg" in result.output_paths:
        webbrowser.open(result.output_paths["svg"].resolve().as_uri())


if __name__ == "__main__":
    main()
