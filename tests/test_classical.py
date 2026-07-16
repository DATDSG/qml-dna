"""Golden-file regression: retraining SVM_kmer/SVM_onehot via qmldna.classical on the
committed data/processed splits must reproduce the test-split metrics already saved by
notebook 02 in results/metrics/svm_*_metrics.csv (same fixed hyperparameters, same seed).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmldna.classical import fit_classical_model

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "encodings_all.npz").exists(),
    reason="Requires the committed data/processed and results/metrics artifacts.",
)


@pytest.fixture(scope="module")
def dataset():
    data = np.load(PROCESSED / "encodings_all.npz", allow_pickle=True)
    splits = json.loads((PROCESSED / "splits_pooled.json").read_text(encoding="utf-8"))
    tr, va, te = (np.array(splits[k]) for k in ("train", "val", "test"))
    return data, tr, va, te


@pytest.mark.parametrize(
    "encoding,csv_name",
    [("kmer", "svm_kmer_metrics.csv"), ("onehot", "svm_onehot_metrics.csv")],
)
def test_retrained_model_matches_saved_test_metrics(dataset, encoding, csv_name):
    metrics_path = RESULTS / "metrics" / csv_name
    if not metrics_path.exists():
        pytest.skip(f"{csv_name} not present in results/metrics")

    data, tr, va, te = dataset
    y = data["y"]
    if encoding == "kmer":
        X = data["kmer"]
    else:
        X = data["onehot"].reshape(len(data["onehot"]), -1).astype(np.float32)

    model = fit_classical_model(
        encoding, X[tr], y[tr], X[va], y[va], window=int(data["window"]), kmer_k=int(data["kmer_k"])
    )

    p_test = model.pipeline.predict_proba(X[te])[:, 1]
    y_hat = (p_test >= model.threshold).astype(int)
    from sklearn.metrics import f1_score, roc_auc_score

    f1 = f1_score(y[te], y_hat, zero_division=0)
    auc = roc_auc_score(y[te], p_test)

    saved = pd.read_csv(metrics_path).set_index("split").loc["test"]

    assert model.threshold == pytest.approx(saved["thr"], abs=0.03)
    assert f1 == pytest.approx(saved["f1"], abs=0.02)
    assert auc == pytest.approx(saved["roc_auc"], abs=0.02)


def test_predict_one_returns_well_formed_result(dataset):
    data, tr, va, te = dataset
    y = data["y"]
    X = data["kmer"]
    model = fit_classical_model("kmer", X[tr], y[tr], X[va], y[va], window=int(data["window"]))

    # Reconstruct one raw window string is not available here (only pre-encoded kmer
    # vectors are cached) so we exercise predict_one via the encode() path directly
    # with a synthetic in-window sequence of the right length.
    seq = "ACGT" * (int(data["window"]) // 4)
    result = model.predict_one(seq)

    assert set(result) == {"label", "probability", "threshold", "model"}
    assert result["label"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
