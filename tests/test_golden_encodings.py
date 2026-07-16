"""Golden-file regression: the extracted encoders must reproduce the already-committed
``data/processed/encodings_all.npz`` produced by the original notebook 01, for the first
dataset (ds_idx == 0). This proves M1's extraction didn't silently change behavior.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmldna.features import angle_encode, generate_windows, kmer_counts, one_hot_encode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "encodings_all.npz").exists(),
    reason="Requires the committed data/processed artifacts from notebook 01.",
)


def _load_meta():
    import json

    return json.loads((PROCESSED / "meta.json").read_text(encoding="utf-8"))


def test_encoders_reproduce_committed_encodings_for_first_dataset():
    Bio_SeqIO = pytest.importorskip("Bio.SeqIO")

    meta = _load_meta()
    window, stride = meta["window"], meta["stride"]
    accession = meta["accessions"][0]  # "KF986530.1", ds_idx == 0

    seq = str(Bio_SeqIO.read(RAW / f"{accession}.gb", "genbank").seq)
    feats = pd.read_csv(PROCESSED / f"{accession}_features.csv")

    windows, labels = [], []
    for r in feats.itertuples():
        sub = seq[r.start : r.end]
        if len(sub) < window:
            continue
        for w in generate_windows(sub, window, stride):
            windows.append(w)
            labels.append(r.label)

    data = np.load(PROCESSED / "encodings_all.npz", allow_pickle=True)
    ds_mask = data["ds_idx"] == 0
    n_expected = int(ds_mask.sum())

    assert len(windows) == n_expected, "Window count for ds_idx 0 must match the committed encodings"

    onehot_recomputed = np.stack([one_hot_encode(w) for w in windows])
    kmer_recomputed = np.stack([kmer_counts(w, k=meta["kmer_k"]) for w in windows])
    angle_recomputed = np.stack([angle_encode(w) for w in windows])
    y_recomputed = np.array(labels, dtype=np.int64)

    np.testing.assert_array_equal(onehot_recomputed, data["onehot"][ds_mask])
    np.testing.assert_allclose(kmer_recomputed, data["kmer"][ds_mask], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(angle_recomputed, data["angle"][ds_mask], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(y_recomputed, data["y"][ds_mask])
