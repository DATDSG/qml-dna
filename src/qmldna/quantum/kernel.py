"""Quantum kernel (QSVM via Nystrom approximation).

Extracted from ``03_quantum_kernel.ipynb`` (cells ``a5606192``, ``6746b1eb``/``a261e557``,
``8aed74a7``, ``1fbaabe7``). The notebook fit PCA/StandardScaler in-memory and never saved
them, and never wrote a single-sample inference path — everything operated on the full
train/val/test batch. ``QuantumKernelModel.predict_one`` is new.

Cost note (measured on the original run): ~14ms per circuit evaluation, and predicting one
new sample requires one evaluation per anchor (default 128) -> ~1.8s/sample. This is why the
API (M2) treats this model as an async job, not a synchronous request.
"""

from __future__ import annotations

from dataclasses import dataclass
from numpy.linalg import eigh

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qmldna.features import kmer_counts

try:
    import pennylane as qml
except ImportError:  # pragma: no cover - quantum extra not installed
    qml = None


def make_device(n_wires: int, shots=None):
    """Prefer the fast lightning.qubit simulator, falling back to default.qubit."""
    try:
        return qml.device("lightning.qubit", wires=n_wires, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=n_wires, shots=shots)


def cz_ring(wires: list[int]) -> None:
    n = len(wires)
    for i in range(n):
        qml.CZ(wires=[wires[i], wires[(i + 1) % n]])


def build_kernel_circuit(n_wires: int):
    """Returns a qnode ``kpair(x1, x2)`` computing the fidelity kernel |<phi(x1)|phi(x2)>|^2."""
    wires = list(range(n_wires))
    dev = make_device(n_wires, shots=None)

    def U(x):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        cz_ring(wires)

    @qml.qnode(dev)
    def kpair(x1, x2):
        U(x1)
        qml.adjoint(U)(x2)
        return qml.expval(qml.Projector([0] * n_wires, wires=wires))

    return kpair


def kernel_block(kpair, XA: np.ndarray, XB: np.ndarray, batch: int = 64) -> np.ndarray:
    """|XA| x |XB| kernel matrix, one circuit evaluation per pair (matches notebook's kernel_block)."""
    m, n = len(XA), len(XB)
    K = np.empty((m, n), dtype=np.float64)
    for i0 in range(0, m, batch):
        i1 = min(i0 + batch, m)
        Xi = XA[i0:i1]
        for j in range(n):
            xbj = XB[j]
            for i, xi in enumerate(Xi):
                K[i0 + i, j] = kpair(xi, xbj)
    return K


def to_angles(X: np.ndarray, clip: float = 3.0) -> np.ndarray:
    Xc = np.clip(X, -clip, clip)
    return (np.pi * Xc / clip).astype(np.float32)


def nystrom_features(K_XM: np.ndarray, K_MM: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, V = eigh(0.5 * (K_MM + K_MM.T))
    W = 1.0 / np.sqrt(np.clip(w, eps, None))
    KMM_mhalf = V @ np.diag(W) @ V.T
    return K_XM @ KMM_mhalf


@dataclass
class QuantumKernelModel:
    """Nystrom-approximated QSVM, bundling every preprocessing step needed for inference
    on a brand new sequence: kmer encode -> PCA -> scaler -> angle map -> kernel vs anchors
    -> Nystrom features -> linear SVM.
    """

    pca: PCA
    scaler: StandardScaler
    anchors: np.ndarray  # [M, n_wires] angle-mapped anchor points
    K_MM: np.ndarray  # [M, M] anchor-anchor kernel matrix
    clf: SVC
    threshold: float
    n_wires: int
    kmer_k: int = 3
    window: int = 256
    batch: int = 64

    def __post_init__(self):
        self._kpair = build_kernel_circuit(self.n_wires)

    def __getstate__(self):
        # PennyLane qnodes/devices aren't reliably picklable; rebuild the circuit on load instead.
        state = self.__dict__.copy()
        state.pop("_kpair", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._kpair = build_kernel_circuit(self.n_wires)

    def _to_angle_features(self, sequence: str) -> np.ndarray:
        x_kmer = kmer_counts(sequence, k=self.kmer_k)[None, :].astype(np.float32)
        x_pca = self.pca.transform(x_kmer)
        x_z = self.scaler.transform(x_pca)
        return to_angles(x_z)

    def predict_one(self, sequence: str) -> dict:
        """Slow path (~1.8s by default with 128 anchors): run in a background job, not sync."""
        if len(sequence) != self.window:
            raise ValueError(f"Expected a {self.window} bp window, got {len(sequence)} bp.")
        x_angles = self._to_angle_features(sequence)[0]
        k_row = np.array([self._kpair(x_angles, a) for a in self.anchors])[None, :]
        phi = nystrom_features(k_row, self.K_MM)
        prob = float(self.clf.predict_proba(phi)[0, 1])
        label = int(prob >= self.threshold)
        return {
            "label": label,
            "probability": prob,
            "threshold": self.threshold,
            "model": "QSVM_kernel_nystrom",
            "n_anchors": len(self.anchors),
        }


def fit_quantum_kernel_model(
    X_kmer_train: np.ndarray,
    y_train: np.ndarray,
    X_kmer_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_wires: int = 8,
    max_train: int = 300,
    n_anchors: int = 128,
    batch: int = 64,
    svm_c: float = 5.0,
    window: int = 256,
    kmer_k: int = 3,
    seed: int = 7,
) -> QuantumKernelModel:
    """Fit PCA -> scaler -> Nystrom-anchor QSVM, mirroring notebook 03's pipeline exactly
    (same defaults: D=8, MAX_TRAIN=300, N_ANCHORS=128, BATCH=64, C=5.0).
    """
    pca = PCA(n_components=n_wires, random_state=seed)
    X_tr_p = pca.fit_transform(X_kmer_train)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_z = scaler.fit_transform(X_tr_p)
    Xtr = to_angles(X_tr_z)

    X_va_p = pca.transform(X_kmer_val)
    X_va_z = scaler.transform(X_va_p)
    Xva = to_angles(X_va_z)

    sel = slice(0, min(max_train, len(Xtr)))
    Xtr_sub = Xtr[sel]
    ytr_sub = y_train[sel]

    rng = np.random.default_rng(seed)
    m = min(n_anchors, len(Xtr_sub))
    anchor_idx = rng.choice(len(Xtr_sub), size=m, replace=False)
    anchors = Xtr_sub[anchor_idx]

    kpair = build_kernel_circuit(n_wires)
    K_MM = kernel_block(kpair, anchors, anchors, batch=batch)
    K_trM = kernel_block(kpair, Xtr_sub, anchors, batch=batch)
    K_vaM = kernel_block(kpair, Xva, anchors, batch=batch)

    Phi_tr = nystrom_features(K_trM, K_MM)
    Phi_va = nystrom_features(K_vaM, K_MM)

    clf = SVC(C=svm_c, kernel="linear", probability=True, class_weight="balanced", random_state=0)
    clf.fit(Phi_tr, ytr_sub)

    from sklearn.metrics import f1_score

    p_val = clf.predict_proba(Phi_va)[:, 1]
    grid = np.linspace(0.05, 0.95, 37)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)

    return QuantumKernelModel(
        pca=pca,
        scaler=scaler,
        anchors=anchors,
        K_MM=K_MM,
        clf=clf,
        threshold=best_thr,
        n_wires=n_wires,
        kmer_k=kmer_k,
        window=window,
        batch=batch,
    )
