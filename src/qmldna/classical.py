"""Classical SVM baselines: pipeline construction, metrics, and single-sequence inference.

Extracted from ``02_classical_baselines.ipynb`` (cells ``9a66c3f1``, ``e1c876b4``, ``7f1aac0d``).
The notebook trained ``svm_kmer``/``svm_1hot`` as in-memory variables only, with a decision
threshold picked on validation but never persisted alongside the model. ``predict_one`` is new:
nothing in the original notebook classifies a single ad hoc sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qmldna.features import kmer_counts, one_hot_encode

Encoding = Literal["kmer", "onehot"]

# Hyperparameters as fit in the notebook (no grid search was actually performed there
# despite the README's "automatic hyperparameter tuning" wording — these are fixed).
_SVM_PARAMS: dict[Encoding, dict] = {
    "kmer": {"C": 5.0},
    "onehot": {"C": 2.0},
}


def build_svm_pipeline(encoding: Encoding) -> Pipeline:
    params = _SVM_PARAMS[encoding]
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(
            C=params["C"],
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=0,
        ),
    )


def extended_metrics(y_true, y_prob, thr: float):
    y_hat = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = float("nan")
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    bal_acc = balanced_accuracy_score(y_true, y_hat)
    mcc = matthews_corrcoef(y_true, y_hat) if len(np.unique(y_true)) == 2 else float("nan")
    rep = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
    return (
        {
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "roc_auc": auc,
            "pr_auc": ap,
            "specificity": tnr,
            "balanced_acc": bal_acc,
            "mcc": mcc,
            "thr": thr,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "support": int(len(y_true)),
        },
        cm,
        rep,
    )


def choose_threshold(y_val, p_val) -> float:
    grid = np.linspace(0.05, 0.95, 37)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (p_val >= t).astype(int)
        f1 = f1_score(y_val, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return 0.5 if np.isnan(best_f1) else best_thr


@dataclass
class ClassicalModel:
    """A fitted SVM pipeline bundled with the decision threshold and config needed for inference."""

    encoding: Encoding
    pipeline: Pipeline
    threshold: float
    window: int
    kmer_k: int = 3

    def encode(self, sequence: str) -> np.ndarray:
        if self.encoding == "kmer":
            return kmer_counts(sequence, k=self.kmer_k)[None, :]
        onehot = one_hot_encode(sequence)  # [window, 4]
        return onehot.reshape(1, -1).astype(np.float32)

    def predict_one(self, sequence: str) -> dict:
        """Classify a single sequence window (must already be exactly ``self.window`` bp)."""
        if len(sequence) != self.window:
            raise ValueError(
                f"Expected a {self.window} bp window for the '{self.encoding}' model, got {len(sequence)} bp."
            )
        x = self.encode(sequence)
        prob = float(self.pipeline.predict_proba(x)[0, 1])
        label = int(prob >= self.threshold)
        return {
            "label": label,
            "probability": prob,
            "threshold": self.threshold,
            "model": f"SVM_{self.encoding}",
        }


def fit_classical_model(
    encoding: Encoding, X_train, y_train, X_val, y_val, window: int, kmer_k: int = 3
) -> ClassicalModel:
    pipeline = build_svm_pipeline(encoding)
    pipeline.fit(X_train, y_train)
    p_val = pipeline.predict_proba(X_val)[:, 1]
    thr = choose_threshold(y_val, p_val)
    return ClassicalModel(encoding=encoding, pipeline=pipeline, threshold=thr, window=window, kmer_k=kmer_k)
