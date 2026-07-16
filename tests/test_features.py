import itertools

import numpy as np
import pytest

from qmldna.features import angle_encode, generate_windows, kmer_counts, one_hot_encode, validate_sequence


def test_one_hot_encode_basic():
    x = one_hot_encode("ACGT")
    assert x.shape == (4, 4)
    assert np.array_equal(x, np.eye(4, dtype=np.float32))


def test_one_hot_encode_unknown_char_is_zero_row():
    x = one_hot_encode("ACGN")
    assert x[3].sum() == 0


def test_kmer_counts_sums_to_one():
    v = kmer_counts("ACGTACGT", k=3)
    assert v.shape == (64,)
    assert pytest.approx(v.sum(), abs=1e-6) == 1.0


def test_kmer_counts_ordering_matches_itertools_product():
    bases = ("A", "C", "G", "T")
    kmers = ["".join(p) for p in itertools.product(bases, repeat=2)]
    assert kmers[0] == "AA"
    assert kmers[-1] == "TT"
    v = kmer_counts("AATT", k=2)
    assert v[kmers.index("AA")] > 0
    assert v[kmers.index("TT")] > 0


def test_angle_encode_known_values():
    a = angle_encode("ACGT")
    assert a[0] == pytest.approx(0.0)
    assert a[1] == pytest.approx(np.pi / 2)
    assert a[2] == pytest.approx(np.pi)
    assert a[3] == pytest.approx(3 * np.pi / 2)


def test_generate_windows_matches_manual_slide():
    seq = "ACGTACGTAC"
    windows = generate_windows(seq, window=4, stride=2)
    assert windows == ["ACGT", "GTAC", "ACGT", "GTAC"]


def test_generate_windows_too_short_returns_empty():
    assert generate_windows("ACG", window=4, stride=2) == []


def test_validate_sequence_rejects_bad_chars():
    with pytest.raises(ValueError, match="invalid character"):
        validate_sequence("ACGTXYZ" * 40, window=256)


def test_validate_sequence_rejects_short_sequence():
    with pytest.raises(ValueError, match="too short"):
        validate_sequence("ACGT", window=256)


def test_validate_sequence_accepts_good_input():
    seq = "ACGT" * 64
    assert validate_sequence(seq, window=256) == seq
