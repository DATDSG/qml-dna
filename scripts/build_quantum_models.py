"""Train the VQC and quantum-kernel (QSVM) models on the committed data/processed splits
and persist them via qmldna.registry, mirroring the defaults from notebooks 03/04.

Usage: python scripts/build_quantum_models.py [--kernel-anchors N] [--skip-kernel] [--skip-vqc]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from qmldna.quantum.kernel import fit_quantum_kernel_model
from qmldna.quantum.vqc import fit_vqc_model
from qmldna.registry import ModelMetadata, save_model

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
VERSION = "v1"


def _load_dataset():
    data = np.load(PROCESSED / "encodings_all.npz", allow_pickle=True)
    splits = json.loads((PROCESSED / "splits_pooled.json").read_text(encoding="utf-8"))
    tr, va, te = (np.array(splits[k]) for k in ("train", "val", "test"))
    return data, tr, va, te


def build_vqc(data, tr, va, te) -> None:
    y = data["y"].astype(int)
    X = data["kmer"].astype(np.float32)
    window, kmer_k = int(data["window"]), int(data["kmer_k"])

    print("Training VQC (6 qubits, 2 layers, up to 60 epochs w/ early stopping)...")
    model = fit_vqc_model(X[tr], y[tr], X[va], y[va], window=window, kmer_k=kmer_k)

    from qmldna.quantum.vqc import _predict_proba_from_expval
    from sklearn.metrics import f1_score, roc_auc_score

    x_pca = model.pca.transform(X[te])
    x_z = model.scaler.transform(x_pca).astype(np.float32)
    p_test = _predict_proba_from_expval(model._vqc, x_z, model.weights, as_numpy=True)
    y_hat = (p_test >= model.threshold).astype(int)
    metrics = {
        "f1_test": float(f1_score(y[te], y_hat, zero_division=0)),
        "roc_auc_test": float(roc_auc_score(y[te], p_test)),
        "threshold": model.threshold,
    }
    print(f"  test F1={metrics['f1_test']:.3f} AUC={metrics['roc_auc_test']:.3f} thr={model.threshold:.2f}")

    save_model(
        "vqc", VERSION, model,
        ModelMetadata(name="vqc", version=VERSION, kind="quantum_vqc", metrics=metrics,
                      extra={"n_wires": model.n_wires, "layers": model.layers, "window": window}),
    )
    print("  saved to models/vqc/v1/")


def build_kernel(data, tr, va, te, n_anchors: int) -> None:
    y = data["y"].astype(np.int64)
    X = data["kmer"].astype(np.float32)
    window, kmer_k = int(data["window"]), int(data["kmer_k"])

    print(f"Training quantum kernel QSVM (8 qubits, Nystrom, {n_anchors} anchors)...")
    model = fit_quantum_kernel_model(X[tr], y[tr], X[va], y[va], n_anchors=n_anchors, window=window, kmer_k=kmer_k)

    from qmldna.quantum.kernel import kernel_block, nystrom_features, to_angles
    from sklearn.metrics import f1_score, roc_auc_score

    # Evaluate on a bounded test subsample -- full 4112-row test set at this anchor count
    # would take hours (see qmldna.quantum.kernel module docstring for the cost measurement).
    te_sub = te[:300]
    x_pca = model.pca.transform(X[te_sub])
    x_z = model.scaler.transform(x_pca)
    Xte = to_angles(x_z)
    K_teM = kernel_block(model._kpair, Xte, model.anchors, batch=model.batch)
    Phi_te = nystrom_features(K_teM, model.K_MM)
    p_test = model.clf.predict_proba(Phi_te)[:, 1]
    y_hat = (p_test >= model.threshold).astype(int)
    metrics = {
        "f1_test_subsample": float(f1_score(y[te_sub], y_hat, zero_division=0)),
        "roc_auc_test_subsample": float(roc_auc_score(y[te_sub], p_test)),
        "threshold": model.threshold,
        "test_subsample_size": len(te_sub),
    }
    print(f"  test(subsample) F1={metrics['f1_test_subsample']:.3f} AUC={metrics['roc_auc_test_subsample']:.3f}")

    save_model(
        "qsvm_kernel", VERSION, model,
        ModelMetadata(name="qsvm_kernel", version=VERSION, kind="quantum_kernel", metrics=metrics,
                      extra={"n_wires": model.n_wires, "n_anchors": len(model.anchors), "window": window}),
    )
    print("  saved to models/qsvm_kernel/v1/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-anchors", type=int, default=128, help="Nystrom anchor count (default matches notebook 03)")
    parser.add_argument("--skip-kernel", action="store_true")
    parser.add_argument("--skip-vqc", action="store_true")
    args = parser.parse_args()

    data, tr, va, te = _load_dataset()

    if not args.skip_vqc:
        build_vqc(data, tr, va, te)
    if not args.skip_kernel:
        build_kernel(data, tr, va, te, n_anchors=args.kernel_anchors)


if __name__ == "__main__":
    main()
