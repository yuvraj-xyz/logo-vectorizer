# Logo Vectorizer

Convert raster logos (PNG / JPEG) into production-grade vector files — SVG, EPS, and Adobe Illustrator `.ai` — with clean cubic-Bézier curves, per-color layer groups, and no embedded raster.

Inspired by Vector Magic, built with Python.

---

## Features

- **Quality you'd ship.** Cubic Bézier paths, per-color `<g>` layers with `inkscape:label`, even-odd fill rule, zero embedded raster.
- **Three formats.** `.svg` (web/Inkscape/Illustrator), `.eps` (DSC-compliant PostScript), `.ai` (Illustrator-native header).
- **Two tracing backends.** [vtracer](https://github.com/visioncortex/vtracer) for smooth spline output, plus an OpenCV-contour fallback if vtracer fails on your platform.
- **Smart color reduction.** k-means in L\*a\*b\* color space with an SSE-elbow auto-k detector (2..32 colors).
- **Curve refinement.** Ramer-Douglas-Peucker simplification, cubic B-spline fitting, C1-continuous joins, coordinate rounding.
- **CLI + Web UI.** `vectorize logo.png` or run the included FastAPI app for a drag-and-drop browser experience.

---

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install -e .
```

Tested on Python 3.11+.

---

## CLI

```bash
vectorize logo.png                              # SVG + EPS + AI in ./output
vectorize logo.png -o ./vectors -c 8 --smooth 2
vectorize logo.png -f svg --preview             # open in browser
vectorize logo.png --backend contour --verbose
```

Run `vectorize --help` for the full option list.

Sample output for the bundled `examples/sample_logo.png` (512×512, 5 colors):

```
  [########################] 100.0%  done
Done - 5 colors traced
  Palette: #f5f5f0, #1a1a2e, #e94560, #00adb5, #ffd664
  Output : output
   SVG  output/sample_logo.svg     ~11 KB
   EPS  output/sample_logo.eps     ~14 KB
   AI   output/sample_logo.ai      ~14 KB
```

End-to-end takes ~2.5 s on a modern CPU; color quantization is the dominant stage.

---

## Web UI

```bash
uvicorn web.app:app --reload
# open http://127.0.0.1:8000
```

Drag a PNG/JPEG onto the drop zone, tweak the color count and smoothing sliders, hit **Vectorize**, then download the resulting SVG / EPS / AI.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Multipart file upload; returns a `job_id`. |
| `GET` | `/status/{job_id}` | Pipeline progress and final palette. |
| `GET` | `/preview/{job_id}` | Inline SVG preview. |
| `GET` | `/result/{job_id}/{svg\|eps\|ai}` | Download the requested format. |
| `GET` | `/source/{job_id}` | Original uploaded raster (for the before/after toggle). |

---

## Architecture

```
PNG/JPEG → preprocess → quantize → separate → trace → smooth → assemble → SVG
                                                                     │
                                                                     ├→ EPS  (cairosvg / native PostScript writer)
                                                                     └→ AI   (EPS + Illustrator DSC block)
```

| Module | Responsibility |
|---|---|
| `vectorizer/preprocessor.py` | Load, flatten alpha, normalize size, bilateral denoise. |
| `vectorizer/quantizer.py`    | k-means color clustering in L\*a\*b\* with auto-k. |
| `vectorizer/separator.py`    | Morphological cleanup, drop tiny components, detect the background layer. |
| `vectorizer/tracer.py`       | vtracer (spline mode) with a pure-OpenCV contour fallback. |
| `vectorizer/smoother.py`     | RDP simplification + cubic Bézier fitting + C1 continuity. |
| `vectorizer/assembler.py`    | Compose the final SVG with per-color `<g>` groups and metadata. |
| `vectorizer/exporters/eps.py` | SVG → EPS via cairosvg, Inkscape CLI, or a built-in PostScript writer. |
| `vectorizer/exporters/ai.py` | EPS body wrapped with Adobe Illustrator's `%AI*` header block. |
| `vectorizer/pipeline.py`     | Orchestrates every stage; emits progress events. |
| `vectorizer/cli.py`          | `click`-powered command line. |
| `web/app.py`                 | FastAPI server + static frontend. |

---

## Tests

```bash
pytest tests/ -q
```

47 unit & integration tests covering every stage of the pipeline, both backends, edge cases (RGBA, tiny inputs, compound paths with holes), and format compliance.

---

## Quality targets

- **Color accuracy:** quantized palette stays within ΔE < 5 (CIEDE2000) of dominant input colors.
- **Path smoothness:** no segments < 2 px; no angle discontinuities > 10° at C1 joins.
- **Format validity:** SVG parses with `lxml`/W3C validator; EPS has `%%BoundingBox` and is DSC-compliant; AI opens in Illustrator CC+.
- **Performance:** < 5 s for a 500×500 four-color logo on a modern CPU.

---

## Project layout

```
logo-vectorizer/
├── vectorizer/                 # the python package
│   ├── exporters/
│   ├── utils/
│   ├── cli.py
│   ├── pipeline.py
│   └── ...
├── web/                        # FastAPI app + static frontend
├── tests/                      # pytest suite
├── examples/                   # sample logo + generator
├── output/                     # default output directory
├── requirements.txt
└── pyproject.toml
```

---

## Notes on vtracer + Python 3.14

The vtracer pyo3 binding on some Python builds (3.14 in particular) crashes when its keyword arguments are used. This tool calls vtracer with positional arguments only — that workaround is invisible to callers but documented in `vectorizer/tracer.py`. If vtracer ever crashes on your platform the tracer transparently falls back to an OpenCV-contour backend; the smoother then upgrades those polylines into cubic Béziers so output quality stays high.
