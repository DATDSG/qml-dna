from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "svm_kmer").exists(),
    reason="Requires trained models (run scripts/build_classical_models.py first).",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_listing(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "svm_kmer" in body
    assert body["svm_kmer"]["kind"] == "classical"


def test_predict_classical_kmer(client):
    seq = "ACGT" * 200  # 800bp, several 256bp windows at stride 128
    r = client.post("/predict/classical?encoding=kmer", json={"sequence": seq})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "SVM_kmer"
    assert body["n_windows"] > 1
    assert all(0.0 <= w["probability"] <= 1.0 for w in body["windows"])


def test_predict_classical_rejects_bad_sequence(client):
    r = client.post("/predict/classical", json={"sequence": "ACGTXYZ"})
    assert r.status_code == 422


def test_predict_classical_rejects_too_short(client):
    r = client.post("/predict/classical", json={"sequence": "ACGT"})
    assert r.status_code == 422


def test_benchmarks_endpoint(client):
    r = client.get("/benchmarks")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "model" in rows[0]


def test_reports_latest(client):
    r = client.get("/reports/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["docx"] is not None or body["pdf"] is not None
