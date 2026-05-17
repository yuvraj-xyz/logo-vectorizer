"""FastAPI web UI for the logo vectorizer.

Run with:  uvicorn web.app:app --reload
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from vectorizer.pipeline import PipelineConfig, vectorize
from vectorizer.preprocessor import PreprocessConfig
from vectorizer.separator import SeparatorConfig
from vectorizer.smoother import SmoothConfig
from vectorizer.tracer import TraceConfig


# --------------------------- In-memory job tracking ---------------------------


@dataclass
class Job:
    job_id: str
    workdir: Path
    status: str = "queued"            # queued | preprocessing | tracing | smoothing | exporting | done | error
    progress: float = 0.0
    error: str | None = None
    palette: list[str] = field(default_factory=list)
    layer_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)
    input_filename: str = "logo.png"
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _set_status(job: Job, stage: str, pct: float) -> None:
    with _jobs_lock:
        job.status = stage
        job.progress = pct


# --------------------------- FastAPI app ---------------------------


HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    tmp_root = Path(tempfile.gettempdir()) / "logo_vectorizer_web"
    tmp_root.mkdir(parents=True, exist_ok=True)
    _app.state.tmp_root = tmp_root
    yield
    # Best-effort cleanup of jobs older than 1 hour.
    cutoff = time.time() - 3600
    with _jobs_lock:
        for jid in list(_jobs.keys()):
            if _jobs[jid].created_at < cutoff:
                try:
                    shutil.rmtree(_jobs[jid].workdir, ignore_errors=True)
                except Exception:
                    pass
                _jobs.pop(jid, None)


app = FastAPI(title="Logo Vectorizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    colors: int = Form(0),
    smoothing: float = Form(1.5),
    denoise: bool = Form(True),
    backend: str = Form("vtracer"),
    fit_shapes: bool = Form(True),
    shape_fit_threshold: float = Form(0.95),
) -> JSONResponse:
    """Accept an uploaded raster image and kick off a background vectorize job."""
    if not file.filename or not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Only PNG and JPEG are supported.")

    tmp_root: Path = app.state.tmp_root
    job_id = uuid.uuid4().hex[:12]
    workdir = tmp_root / job_id
    workdir.mkdir(parents=True, exist_ok=True)

    src_path = workdir / file.filename
    contents = await file.read()
    src_path.write_bytes(contents)

    job = Job(job_id=job_id, workdir=workdir, input_filename=file.filename)
    with _jobs_lock:
        _jobs[job_id] = job

    asyncio.get_running_loop().run_in_executor(
        None,
        _run_pipeline,
        job,
        src_path,
        int(colors),
        float(smoothing),
        bool(denoise),
        backend.strip().lower(),
        bool(fit_shapes),
        float(shape_fit_threshold),
    )

    return JSONResponse({"job_id": job_id})


def _run_pipeline(
    job: Job,
    src_path: Path,
    colors: int,
    smoothing: float,
    denoise: bool,
    backend: str,
    fit_shapes: bool,
    shape_fit_threshold: float,
) -> None:
    try:
        from vectorizer.shapes import FitConfig

        config = PipelineConfig(
            preprocess=PreprocessConfig(denoise=denoise),
            separate=SeparatorConfig(mask_smoothing=max(0.8, smoothing)),
            trace=TraceConfig(backend=backend if backend in {"potrace", "vtracer", "contour"} else "potrace"),
            fit=FitConfig(iou_threshold=shape_fit_threshold),
            smooth=SmoothConfig(spline_smoothing=smoothing),
            fit_shapes=fit_shapes,
        )

        def progress(stage: str, pct: float) -> None:
            _set_status(job, stage, pct)

        out_dir = job.workdir / "output"
        result = vectorize(
            src_path,
            output_dir=out_dir,
            formats=("svg", "eps", "ai"),
            target_colors=colors or None,
            config=config,
            progress=progress,
        )

        with _jobs_lock:
            job.status = "done"
            job.progress = 1.0
            job.palette = result.assemble.palette
            job.layer_count = result.assemble.layer_count
            job.timings = result.timings
            job.files = {k: Path(v) for k, v in result.output_paths.items()}
    except Exception as exc:
        with _jobs_lock:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"


@app.get("/status/{job_id}")
async def status(job_id: str) -> JSONResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        payload = {
            "job_id": job_id,
            "status": job.status,
            "progress": round(job.progress, 4),
            "error": job.error,
            "palette": list(job.palette),
            "layer_count": job.layer_count,
            "timings": dict(job.timings),
            "files": {k: str(v.name) for k, v in job.files.items()},
        }
    return JSONResponse(payload)


@app.get("/preview/{job_id}")
async def preview(job_id: str) -> Response:
    job = _require_done(job_id)
    svg_path = job.files.get("svg")
    if not svg_path or not svg_path.exists():
        raise HTTPException(404, "SVG preview not available")
    return Response(svg_path.read_bytes(), media_type="image/svg+xml")


@app.get("/result/{job_id}/{fmt}")
async def result(job_id: str, fmt: str) -> FileResponse:
    job = _require_done(job_id)
    fmt = fmt.lower().strip()
    file_path = job.files.get(fmt)
    if not file_path or not file_path.exists():
        raise HTTPException(404, f"No {fmt} output for job {job_id}")
    media = {
        "svg": "image/svg+xml",
        "eps": "application/postscript",
        "ai": "application/illustrator",
    }.get(fmt, "application/octet-stream")
    return FileResponse(file_path, media_type=media, filename=file_path.name)


@app.get("/source/{job_id}")
async def source(job_id: str) -> Response:
    """Return the original uploaded raster as a data URL for before/after toggling."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job_id")
    src = next((p for p in job.workdir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}), None)
    if not src:
        raise HTTPException(404, "Source not found")
    return FileResponse(src, media_type=_guess_mime(src.suffix))


def _guess_mime(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix.lower(), "application/octet-stream")


def _require_done(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job_id")
        if job.status == "error":
            raise HTTPException(500, job.error or "Job failed")
        if job.status != "done":
            raise HTTPException(409, f"Job not finished (status={job.status})")
        return job


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
