"""Sequence feature encoders.

Extracted verbatim from ``01_data_preparation.ipynb`` (cell tagged ``95d7feca``)
so the same encodings used to build ``data/processed/encodings_all.npz`` can be
reused for single-sequence inference instead of only full-batch notebook runs.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

_BASES = ("A", "C", "G", "T")
_VALID_CHARS = set("ACGTNacgtn")


def one_hot_encode(s: str) -> np.ndarray:
    """[len(s), 4] one-hot matrix over A/C/G/T. Non-ACGT characters map to an all-zero row."""
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(s), 4), dtype=np.float32)
    for i, ch in enumerate(s.upper()):
        j = m.get(ch)
        if j is not None:
            X[i, j] = 1.0
    return X


def kmer_counts(s: str, k: int = 3) -> np.ndarray:
    """Normalized k-mer frequency vector, length 4**k, ordered by itertools.product(bases, repeat=k)."""
    kmers = ["".join(p) for p in itertools.product(_BASES, repeat=k)]
    c = Counter(s[i : i + k] for i in range(len(s) - k + 1))
    v = np.array([c.get(km, 0) for km in kmers], dtype=np.float32)
    sm = v.sum()
    return v / sm if sm > 0 else v


def angle_encode(s: str) -> np.ndarray:
    """Maps each base to a rotation angle (A=0, C=pi/2, G=pi, T=3pi/2); used as the quantum feature map input."""
    am = {"A": 0.0, "C": np.pi / 2, "G": np.pi, "T": 3 * np.pi / 2}
    return np.array([am.get(ch.upper(), 0.0) for ch in s], dtype=np.float32)


def validate_sequence(s: str, window: int) -> str:
    """Raise ValueError with a user-facing message if ``s`` isn't a usable ACGTN sequence of the expected length.

    Used at API/UI boundaries so a non-expert pasting bad input gets a clear message
    instead of a stack trace from deep inside a model pipeline.
    """
    if not s:
        raise ValueError("Sequence is empty.")
    cleaned = s.strip().upper()
    bad_chars = sorted({ch for ch in cleaned if ch not in {"A", "C", "G", "T", "N"}})
    if bad_chars:
        raise ValueError(
            f"Sequence contains invalid character(s): {', '.join(bad_chars)}. " "Only A, C, G, T, N are allowed."
        )
    if len(cleaned) < window:
        raise ValueError(
            f"Sequence is too short ({len(cleaned)} bp). The models were trained on "
            f"{window} bp windows; please provide at least {window} bp."
        )
    return cleaned


def generate_windows(seq: str, window: int, stride: int) -> list[str]:
    """Slide a fixed-size window across ``seq``, matching notebook 01's inline windowing loop.

    Returns the list of window strings (no labels/dataset bookkeeping — that's a
    training-time concern handled by ``data.py``, not inference).
    """
    if len(seq) < window:
        return []
    return [seq[off : off + window] for off in range(0, len(seq) - window + 1, stride)]
