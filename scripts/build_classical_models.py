"""Train SVM_kmer and SVM_onehot on the committed data/processed splits and persist them
via qmldna.registry, so the API (M2) has real serialized models to load instead of
retraining on every process start.

Usage: python scripts/build_classical_models.py
"""
import json
from pathlib import Path

import numpy as np

from qmldna.classical import fit_classical_model
from qmldna.registry import ModelMetadata, save_model

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
VERSION = "v1"


def main() -> None:
    data = np.load(PROCESSED / "encodings_all.npz", allow_pickle=True)
    splits = json.loads((PROCESSED / "splits_pooled.json").read_text(encoding="utf-8"))
    tr, va, te = (np.array(splits[k]) for k in ("train", "val", "test"))
    y = data["y"]
    window, kmer_k = int(data["window"]), int(data["kmer_k"])

    for encoding, name in (("kmer", "svm_kmer"), ("onehot", "svm_onehot")):
        if encoding == "kmer":
            X = data["kmer"]
        else:
            X = data["onehot"].reshape(len(data["onehot"]), -1).astype(np.float32)

        print(f"Training {name} on {len(tr)} train / {len(va)} val samples...")
        model = fit_classical_model(encoding, X[tr], y[tr], X[va], y[va], window=window, kmer_k=kmer_k)

        p_test = model.pipeline.predict_proba(X[te])[:, 1]
        y_hat = (p_test >= model.threshold).astype(int)
        from sklearn.metrics import f1_score, roc_auc_score

        metrics = {
            "f1_test": float(f1_score(y[te], y_hat, zero_division=0)),
            "roc_auc_test": float(roc_auc_score(y[te], p_test)),
            "threshold": model.threshold,
        }
        print(f"  test F1={metrics['f1_test']:.3f} AUC={metrics['roc_auc_test']:.3f} thr={model.threshold:.2f}")

        save_model(
            name,
            VERSION,
            model,
            ModelMetadata(
                name=name,
                version=VERSION,
                kind="classical",
                metrics=metrics,
                extra={"encoding": encoding, "window": window, "kmer_k": kmer_k},
            ),
        )
        print(f"  saved to models/{name}/{VERSION}/")


if __name__ == "__main__":
    main()
