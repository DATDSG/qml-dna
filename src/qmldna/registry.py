"""Versioned on-disk model artifact store.

Nothing in the original notebooks serializes trained models — SVM pipelines and
quantum preprocessing (PCA/scaler) only ever existed as in-notebook variables.
This module gives every model a stable ``models/<name>/<version>/`` location so
the API and CLI can load them without re-running a notebook.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODELS_DIR = Path("models")


@dataclass
class ModelMetadata:
    name: str
    version: str
    kind: str  # "classical" | "quantum_kernel" | "quantum_vqc"
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    extra: dict[str, Any] = field(default_factory=dict)


def model_dir(name: str, version: str, base: Path = DEFAULT_MODELS_DIR) -> Path:
    d = base / name / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(
    name: str, version: str, artifact: Any, metadata: ModelMetadata, base: Path = DEFAULT_MODELS_DIR
) -> Path:
    d = model_dir(name, version, base)
    joblib.dump(artifact, d / "artifact.joblib")
    (d / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    _update_latest_pointer(name, version, base)
    return d


def load_model(name: str, version: str = "latest", base: Path = DEFAULT_MODELS_DIR) -> Any:
    if version == "latest":
        version = _read_latest_pointer(name, base)
    d = base / name / version
    if not (d / "artifact.joblib").exists():
        raise FileNotFoundError(f"No artifact found for model '{name}' version '{version}' under {base}")
    return joblib.load(d / "artifact.joblib")


def load_metadata(name: str, version: str = "latest", base: Path = DEFAULT_MODELS_DIR) -> ModelMetadata:
    if version == "latest":
        version = _read_latest_pointer(name, base)
    d = base / name / version
    return ModelMetadata(**json.loads((d / "metadata.json").read_text(encoding="utf-8")))


def _update_latest_pointer(name: str, version: str, base: Path) -> None:
    (base / name).mkdir(parents=True, exist_ok=True)
    (base / name / "LATEST").write_text(version, encoding="utf-8")


def _read_latest_pointer(name: str, base: Path) -> str:
    p = base / name / "LATEST"
    if not p.exists():
        raise FileNotFoundError(f"No versions registered for model '{name}' under {base}")
    return p.read_text(encoding="utf-8").strip()
