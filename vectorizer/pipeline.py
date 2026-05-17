"""End-to-end pipeline that wires preprocess → quantize → separate → trace → smooth → assemble."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .assembler import AssembleConfig, AssembleResult, assemble_svg, save_svg
from .exporters import svg_to_ai, svg_to_eps
from .preprocessor import PreprocessConfig, PreprocessResult, preprocess
from .quantizer import QuantizeConfig, QuantizeResult, quantize
from .separator import SeparatorConfig, separate
from .shapes import FitConfig
from .smoother import SmoothConfig, smooth_traced_layers
from .tracer import TraceConfig, decompose_layers, trace_layers


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    separate: SeparatorConfig = field(default_factory=SeparatorConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)
    # When True, attempt per-component primitive fitting (circle/ellipse/rect/
    # polygon) before falling back to Bézier tracing. When False, behave as
    # the legacy whole-layer tracer.
    fit_shapes: bool = True


@dataclass
class PipelineResult:
    svg_string: str
    assemble: AssembleResult
    quantize: QuantizeResult
    preprocess: PreprocessResult
    output_paths: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    source_filename: str | None = None


ProgressFn = Callable[[str, float], None]


def vectorize(
    source: str | Path,
    output_dir: str | Path = "./output",
    formats: Iterable[str] = ("svg", "eps", "ai"),
    target_colors: int | None = None,
    config: PipelineConfig | None = None,
    progress: ProgressFn | None = None,
    base_name: str | None = None,
) -> PipelineResult:
    """Convert a raster image to vector format(s) and write the results to disk.

    Returns a :class:`PipelineResult` with the assembled SVG string, palette,
    timings, and paths to every output file that was written.
    """
    cfg = config or PipelineConfig()
    src = Path(source)
    base = base_name or src.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    def step(name: str, pct: float) -> None:
        if progress:
            progress(name, pct)

    step("preprocess", 0.05)
    t0 = time.perf_counter()
    pre = preprocess(src, cfg.preprocess)
    timings["preprocess"] = time.perf_counter() - t0

    step("quantize", 0.20)
    t0 = time.perf_counter()
    quant = quantize(pre.image, target_colors=target_colors, config=cfg.quantize)
    timings["quantize"] = time.perf_counter() - t0

    step("separate", 0.35)
    t0 = time.perf_counter()
    layers = separate(quant.masks, quant.palette_hex, cfg.separate)
    timings["separate"] = time.perf_counter() - t0

    step("trace", 0.55)
    t0 = time.perf_counter()
    if cfg.fit_shapes:
        traced = decompose_layers(layers, cfg.trace, cfg.fit)
    else:
        traced = trace_layers(layers, cfg.trace)
    timings["trace"] = time.perf_counter() - t0

    step("smooth", 0.75)
    t0 = time.perf_counter()
    smoothed = smooth_traced_layers(traced, cfg.smooth)
    timings["smooth"] = time.perf_counter() - t0

    step("assemble", 0.85)
    t0 = time.perf_counter()
    h, w = pre.image.shape[:2]
    assembled = assemble_svg(
        smoothed,
        width=w,
        height=h,
        source_filename=src.name,
        config=cfg.assemble,
    )
    timings["assemble"] = time.perf_counter() - t0

    output_paths: dict[str, Path] = {}

    formats = {f.lower().strip() for f in formats}
    step("export", 0.92)
    t0 = time.perf_counter()
    if "svg" in formats:
        output_paths["svg"] = save_svg(assembled.svg_string, out_dir / f"{base}.svg")
    if "eps" in formats:
        output_paths["eps"] = svg_to_eps(assembled.svg_string, out_dir / f"{base}.eps")
    if "ai" in formats:
        output_paths["ai"] = svg_to_ai(assembled.svg_string, out_dir / f"{base}.ai")
    timings["export"] = time.perf_counter() - t0

    step("done", 1.0)

    return PipelineResult(
        svg_string=assembled.svg_string,
        assemble=assembled,
        quantize=quant,
        preprocess=pre,
        output_paths=output_paths,
        timings=timings,
        source_filename=src.name,
    )
