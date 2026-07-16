"""Variational Quantum Classifier (VQC).

Extracted from ``04_quantum_vqc.ipynb`` (cells ``fe0a889c``, ``00ec6d9a``, ``e14a02f1``).
The notebook saved ``vqc_weights.npy`` but never loaded it back for inference, and never
persisted the PCA/StandardScaler preprocessing fit in-memory before the circuit. Both are
required to classify a new sequence, so ``VQCModel.predict_one`` (and the preprocessing
persistence) are new.

Cost note: a single circuit evaluation on this size circuit (6 qubits, 2 layers, lightning.qubit)
is sub-100ms, so unlike the kernel model this is synchronous-friendly for the API (M2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qmldna.features import kmer_counts

try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError:  # pragma: no cover - quantum extra not installed
    qml = None
    pnp = None


def make_device(n_wires: int, shots=None, use_mixed: bool = False):
    backend = "default.mixed" if use_mixed else "lightning.qubit"
    try:
        return qml.device(backend, wires=n_wires, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=n_wires, shots=shots)


def build_vqc_circuit(n_wires: int, layers: int, p_bitflip: float = 0.0, p_depol: float = 0.0):
    """Returns the qnode ``vqc(x, w)`` matching notebook 04's ansatz: per-layer AngleEmbedding
    (data re-uploading) + BasicEntanglerLayers, PauliZ(0) readout (expval in [-1, 1]).
    """
    dev = make_device(n_wires, shots=None, use_mixed=(p_bitflip > 0 or p_depol > 0))

    def layer(x, w):
        qml.AngleEmbedding(x, wires=range(n_wires), rotation="Y")
        if p_bitflip > 0:
            for i in range(n_wires):
                qml.BitFlip(p_bitflip, wires=i)
        if p_depol > 0:
            for i in range(n_wires):
                qml.DepolarizingChannel(p_depol, wires=i)
        qml.BasicEntanglerLayers(w[None, :], wires=range(n_wires))

    @qml.qnode(dev, interface="autograd")
    def vqc(x, w):
        for layer_idx in range(layers):
            layer(x, w[layer_idx])
        return qml.expval(qml.PauliZ(0))

    return vqc


def _predict_proba_from_expval(vqc, X, w, as_numpy: bool = False):
    """expval in [-1,1] -> probability in [0,1], matching notebook 04's predict_proba exactly."""
    vals = [(1 + vqc(xi, w)) / 2 for xi in X]
    p = pnp.clip(pnp.stack(vals), 1e-6, 1 - 1e-6)
    return np.asarray(p) if as_numpy else p


@dataclass
class VQCModel:
    """A trained VQC bundled with the PCA/scaler preprocessing and threshold needed for
    single-sequence inference (none of which the original notebook persisted to disk).
    """

    pca: PCA
    scaler: StandardScaler
    weights: np.ndarray  # [layers, n_wires]
    threshold: float
    n_wires: int
    layers: int
    kmer_k: int = 3
    window: int = 256

    def __post_init__(self):
        self._vqc = build_vqc_circuit(self.n_wires, self.layers)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_vqc", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._vqc = build_vqc_circuit(self.n_wires, self.layers)

    def predict_one(self, sequence: str) -> dict:
        """Sub-100ms: safe to call synchronously from the API."""
        if len(sequence) != self.window:
            raise ValueError(f"Expected a {self.window} bp window, got {len(sequence)} bp.")
        x_kmer = kmer_counts(sequence, k=self.kmer_k)[None, :].astype(np.float32)
        x_pca = self.pca.transform(x_kmer)
        x_z = self.scaler.transform(x_pca).astype(np.float32)[0]
        m = float(self._vqc(x_z, self.weights))
        prob = float(np.clip((1 + m) / 2, 1e-6, 1 - 1e-6))
        label = int(prob >= self.threshold)
        return {"label": label, "probability": prob, "threshold": self.threshold, "model": "VQC"}


def fit_vqc_model(
    X_kmer_train: np.ndarray,
    y_train: np.ndarray,
    X_kmer_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_wires: int = 6,
    layers: int = 2,
    lr: float = 0.05,
    batch_size: int = 64,
    max_epochs: int = 60,
    patience: int = 6,
    param_clip: float = 1.5,
    window: int = 256,
    kmer_k: int = 3,
    seed: int = 11,
) -> VQCModel:
    """Fit PCA -> scaler -> VQC via mini-batch Adam with early stopping, mirroring notebook
    04's defaults exactly (D=6, L=2, lr=0.05, batch=64, epochs<=60, patience=6, clip=1.5).
    """
    np.random.seed(seed)
    pnp.random.seed(seed)

    pca = PCA(n_components=n_wires, random_state=seed)
    X_tr = pca.fit_transform(X_kmer_train)
    X_va = pca.transform(X_kmer_val)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr).astype(np.float32)
    Xva = scaler.transform(X_va).astype(np.float32)
    ytr, yva = y_train, y_val

    vqc = build_vqc_circuit(n_wires, layers)
    weights = pnp.random.normal(scale=0.15, size=(layers, n_wires), requires_grad=True)

    def predict_proba(X, w, as_numpy=False):
        return _predict_proba_from_expval(vqc, X, w, as_numpy)

    def bce_loss(y_true, p_hat):
        return -pnp.mean(y_true * pnp.log(p_hat) + (1 - y_true) * pnp.log(1 - p_hat))

    def iterate_minibatches(X, y, bs):
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        for i in range(0, len(y), bs):
            sl = idx[i : i + bs]
            yield X[sl], y[sl]

    opt = qml.AdamOptimizer(stepsize=lr)
    best_va = float("inf")
    best_w = pnp.array(weights, requires_grad=True)
    no_improve = 0

    for _epoch in range(1, max_epochs + 1):
        for Xb, yb in iterate_minibatches(Xtr, ytr, batch_size):

            def cost(w):
                y_true = pnp.array(yb, dtype=float)
                p_hat = predict_proba(Xb, w, as_numpy=False)
                return bce_loss(y_true, p_hat)

            w_new = opt.step(cost, weights)
            weights = pnp.clip(w_new, -param_clip, param_clip)

        p_va = predict_proba(Xva, weights)
        loss_va = float(bce_loss(yva, p_va))
        if not np.isfinite(loss_va):
            break

        if loss_va + 1e-4 < best_va:
            best_va = loss_va
            best_w = pnp.array(weights, requires_grad=False)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    weights = np.array(best_w, dtype=float)

    from sklearn.metrics import f1_score

    p_val = _predict_proba_from_expval(vqc, Xva, weights, as_numpy=True)
    grid = np.linspace(0.05, 0.95, 37)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(yva, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)

    return VQCModel(
        pca=pca,
        scaler=scaler,
        weights=weights,
        threshold=best_thr,
        n_wires=n_wires,
        layers=layers,
        kmer_k=kmer_k,
        window=window,
    )
