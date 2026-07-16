"""Golden-file regression for the quantum branches: retraining via qmldna.quantum on the
committed splits should land close to the test-split metrics notebooks 03/04 already saved.
These are slow (minutes) since they involve real circuit simulation, so they're marked
'slow' and skipped by default in CI (see pyproject.toml / pytest -m "not slow").
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pennylane")

from qmldna.quantum.kernel import fit_quantum_kernel_model
from qmldna.quantum.vqc import fit_vqc_model

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (PROCESSED / "encodings_all.npz").exists(),
        reason="Requires the committed data/processed and results/metrics artifacts.",
    ),
]


@pytest.fixture(scope="module")
def dataset():
    data = np.load(PROCESSED / "encodings_all.npz", allow_pickle=True)
    splits = json.loads((PROCESSED / "splits_pooled.json").read_text(encoding="utf-8"))
    tr, va, te = (np.array(splits[k]) for k in ("train", "val", "test"))
    return data, tr, va, te


def test_quantum_kernel_model_predicts_sanely_on_test_subsample(dataset):
    """Full-scale (128 anchors x full 4112-row test split) is ~1.06M circuit evals -- multiple
    hours at ~14ms/eval, matching the original notebook's measured runtime. That's too slow to
    run as a test, so this uses a reduced anchor count and a test subsample: a functional
    smoke check (valid probabilities, better-than-random AUC), not an exact metrics match.
    """
    data, tr, va, te = dataset
    y = data["y"].astype(np.int64)
    X_kmer = data["kmer"].astype(np.float32)

    model = fit_quantum_kernel_model(X_kmer[tr], y[tr], X_kmer[va], y[va], max_train=80, n_anchors=32, batch=32)

    from sklearn.metrics import roc_auc_score

    te_sub = te[:150]
    x_pca = model.pca.transform(X_kmer[te_sub])
    x_z = model.scaler.transform(x_pca)
    from qmldna.quantum.kernel import kernel_block, nystrom_features, to_angles

    Xte = to_angles(x_z)
    K_teM = kernel_block(model._kpair, Xte, model.anchors, batch=model.batch)
    Phi_te = nystrom_features(K_teM, model.K_MM)
    p_test = model.clf.predict_proba(Phi_te)[:, 1]

    assert np.all((p_test >= 0.0) & (p_test <= 1.0))
    auc = roc_auc_score(y[te_sub], p_test)
    assert auc > 0.5, f"quantum kernel model should beat random guessing, got AUC={auc:.3f}"


def test_vqc_model_matches_saved_metrics(dataset):
    metrics_path = RESULTS / "metrics" / "vqc_metrics.csv"
    if not metrics_path.exists():
        pytest.skip("vqc_metrics.csv not present in results/metrics")

    data, tr, va, te = dataset
    y = data["y"].astype(int)
    X_kmer = data["kmer"].astype(np.float32)

    model = fit_vqc_model(X_kmer[tr], y[tr], X_kmer[va], y[va])

    x_pca = model.pca.transform(X_kmer[te])
    x_z = model.scaler.transform(x_pca).astype(np.float32)

    from qmldna.quantum.vqc import _predict_proba_from_expval

    p_test = _predict_proba_from_expval(model._vqc, x_z, model.weights, as_numpy=True)
    y_hat = (p_test >= model.threshold).astype(int)

    from sklearn.metrics import f1_score, roc_auc_score

    f1 = f1_score(y[te], y_hat, zero_division=0)
    auc = roc_auc_score(y[te], p_test)

    saved = pd.read_csv(metrics_path).set_index("split").loc["test"]
    assert f1 == pytest.approx(saved["f1"], abs=0.05)
    assert auc == pytest.approx(saved["roc_auc"], abs=0.15)


def test_vqc_predict_one_well_formed(dataset):
    data, tr, va, te = dataset
    y = data["y"].astype(int)
    X_kmer = data["kmer"].astype(np.float32)
    model = fit_vqc_model(X_kmer[tr][:512], y[tr][:512], X_kmer[va][:128], y[va][:128], max_epochs=3, patience=2)
    seq = "ACGT" * (int(data["window"]) // 4)
    result = model.predict_one(seq)
    assert set(result) == {"label", "probability", "threshold", "model"}
    assert result["label"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
