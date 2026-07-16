"""qml-dna inference & benchmark API.

Wraps the models trained via scripts/build_*_models.py (persisted through
qmldna.registry) so a new DNA sequence can be classified over HTTP instead of
only through a full notebook run. See ../.claude-plan (or the repo README) for
the overall architecture.

Run: uvicorn api.main:app --reload
"""

from __future__ import annotations

import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qmldna.classical import ClassicalModel
from qmldna.features import generate_windows, validate_sequence
from qmldna.quantum.vqc import VQCModel
from qmldna.registry import load_metadata, load_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The one endpoint expensive enough to be abusable (~1.8s/window at default anchor count);
# everything else here is milliseconds.
QKERNEL_MAX_WINDOWS = 5

app = FastAPI(title="qml-dna API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}


class PredictRequest(BaseModel):
    sequence: str = Field(..., description="Raw DNA sequence (A/C/G/T/N), any length >= the model's window size.")


class WindowPrediction(BaseModel):
    start: int
    end: int
    label: int
    probability: float


class PredictResponse(BaseModel):
    model: str
    threshold: float
    n_windows: int
    windows: list[WindowPrediction]
    positive_fraction: float
    mean_probability: float


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    result: PredictResponse | None = None
    error: str | None = None


@lru_cache(maxsize=8)
def _get_classical(encoding: Literal["kmer", "onehot"]) -> ClassicalModel:
    name = f"svm_{encoding}"
    try:
        return load_model(name)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail=f"Model '{name}' not built yet. Run scripts/build_classical_models.py first."
        ) from e


@lru_cache(maxsize=1)
def _get_vqc() -> VQCModel:
    try:
        return load_model("vqc")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail="Model 'vqc' not built yet. Run scripts/build_quantum_models.py first."
        ) from e


@lru_cache(maxsize=1)
def _get_qkernel():
    try:
        return load_model("qsvm_kernel")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail="Model 'qsvm_kernel' not built yet. Run scripts/build_quantum_models.py first."
        ) from e


def _windows_for(sequence: str, window: int, stride: int, max_windows: int | None = None) -> list[tuple[int, int, str]]:
    try:
        cleaned = validate_sequence(sequence, window)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    chunks = generate_windows(cleaned, window, stride)
    if not chunks:
        raise HTTPException(status_code=422, detail="No windows could be generated from this sequence.")
    if max_windows is not None:
        chunks = chunks[:max_windows]
    out = []
    for i, chunk in enumerate(chunks):
        start = i * stride
        out.append((start, start + window, chunk))
    return out


def _summarize(
    model_name: str, threshold: float, predictions: list[dict], positions: list[tuple[int, int]]
) -> PredictResponse:
    windows = [
        WindowPrediction(start=s, end=e, label=p["label"], probability=p["probability"])
        for (s, e), p in zip(positions, predictions)
    ]
    probs = [w.probability for w in windows]
    return PredictResponse(
        model=model_name,
        threshold=threshold,
        n_windows=len(windows),
        windows=windows,
        positive_fraction=sum(w.label for w in windows) / len(windows),
        mean_probability=sum(probs) / len(probs),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    out = {}
    for name in ("svm_kmer", "svm_onehot", "vqc", "qsvm_kernel"):
        try:
            out[name] = load_metadata(name).__dict__
        except FileNotFoundError:
            out[name] = None
    return out


@app.post("/predict/classical", response_model=PredictResponse)
def predict_classical(req: PredictRequest, encoding: Literal["kmer", "onehot"] = "kmer"):
    model = _get_classical(encoding)
    positions_and_chunks = _windows_for(req.sequence, model.window, stride=model.window // 2)
    positions = [(s, e) for s, e, _ in positions_and_chunks]
    preds = [model.predict_one(chunk) for _, _, chunk in positions_and_chunks]
    return _summarize(f"SVM_{encoding}", model.threshold, preds, positions)


@app.post("/predict/vqc", response_model=PredictResponse)
def predict_vqc(req: PredictRequest):
    model = _get_vqc()
    positions_and_chunks = _windows_for(req.sequence, model.window, stride=model.window // 2)
    positions = [(s, e) for s, e, _ in positions_and_chunks]
    preds = [model.predict_one(chunk) for _, _, chunk in positions_and_chunks]
    return _summarize("VQC", model.threshold, preds, positions)


def _run_qkernel_job(job_id: str, sequence: str) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        model = _get_qkernel()
        positions_and_chunks = _windows_for(
            sequence, model.window, stride=model.window // 2, max_windows=QKERNEL_MAX_WINDOWS
        )
        positions = [(s, e) for s, e, _ in positions_and_chunks]
        preds = [model.predict_one(chunk) for _, _, chunk in positions_and_chunks]
        result = _summarize("QSVM_kernel_nystrom", model.threshold, preds, positions)
        _jobs[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as e:  # noqa: BLE001 - surfaced to the client via job status, not raised
        _jobs[job_id] = {"status": "error", "result": None, "error": str(e)}


@app.post("/predict/qkernel", response_model=JobStatus, status_code=202)
def predict_qkernel(req: PredictRequest, background_tasks: BackgroundTasks):
    """Async: quantum kernel inference is ~1.8s per window at the default anchor count
    (measured ~14ms/circuit-eval x 128 anchors), so this returns a job id immediately;
    poll GET /jobs/{job_id} for the result. Capped at %d windows per request.
    """ % QKERNEL_MAX_WINDOWS
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(_run_qkernel_job, job_id, req.sequence)
    return JobStatus(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return JobStatus(job_id=job_id, **job)


def _csv_records(path: Path) -> list[dict]:
    """pandas leaves NaN for missing values, which isn't valid JSON; None round-trips cleanly.
    (Assigning None back into a float64 column reverts to NaN, so this cleans the plain
    dicts after conversion rather than trying to fix it on the DataFrame itself.)
    """
    import math

    records = pd.read_csv(path).to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    return records


@app.get("/benchmarks")
def benchmarks():
    combined = RESULTS / "metrics" / "combined.csv"
    if not combined.exists():
        raise HTTPException(
            status_code=404, detail="results/metrics/combined.csv not found; run the notebooks/pipeline first."
        )
    return _csv_records(combined)


@app.get("/benchmarks/scorecard")
def scorecard():
    path = RESULTS / "metrics" / "scorecard.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="results/metrics/scorecard.csv not found; run the notebooks/pipeline first."
        )
    return _csv_records(path)


@app.get("/reports/latest")
def latest_report():
    """Serves the most recently generated report. Regeneration is a batch/papermill concern
    (see scripts/rebuild_pipeline.py), not something this endpoint recomputes on every call.
    """
    docx = RESULTS / "report" / "DNA_QML_Results_Report.docx"
    pdf = RESULTS / "report" / "DNA_QML_Results_Report.pdf"
    if not docx.exists() and not pdf.exists():
        raise HTTPException(
            status_code=404,
            detail="No report has been generated yet; run 07_reporting.ipynb or scripts/rebuild_pipeline.py.",
        )
    return {
        "docx": str(docx.relative_to(ROOT)) if docx.exists() else None,
        "pdf": str(pdf.relative_to(ROOT)) if pdf.exists() else None,
        "generated_at": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(docx.stat().st_mtime)) if docx.exists() else None
        ),
    }
