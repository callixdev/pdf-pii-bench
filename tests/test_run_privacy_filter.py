"""Tests for src/run_privacy_filter.py — the cached scoring path.

opf is imported lazily, only when a document misses the cache, so a
fully cached run must score without ever touching the model. This is
the path that lets anyone verify the committed results in seconds.
"""

import sys

import pytest

from src.cache import save_cached
from src.run_privacy_filter import run_privacy_filter


@pytest.fixture()
def text_data(sample_documents):
    return {row["id"]: {"text": row["text"], "n_pages": 1}
            for row in sample_documents}


@pytest.fixture()
def cache_dir(tmp_path, sample_adapter_output):
    """A fully populated prediction cache (the canned adapter spans)."""
    for out in sample_adapter_output:
        save_cached(tmp_path, out["id"], out["spans"])
    return tmp_path


def test_fully_cached_run_never_loads_the_model(
    monkeypatch, cache_dir, text_data, sample_ground_truth
):
    # Poison the import: `from opf import OPF` raises immediately if
    # the runner tries to load the model despite a full cache.
    monkeypatch.setitem(sys.modules, "opf", None)

    results = run_privacy_filter(text_data, sample_ground_truth,
                                 cache_dir=cache_dir)

    assert results["system"] == "privacy-filter"
    assert results["n_documents"] == 3
    # Same known answers as the harness integration test: the org
    # phone is absorbed (neither TP nor FP), doc-3's password is the
    # one miss, doc-2's narrower address still overlap-matches.
    overlap = results["overlap"]["micro"]
    assert (overlap["tp"], overlap["fp"], overlap["fn"]) == (5, 0, 1)
    assert overlap["precision"] == 1.0
    exact = results["exact"]["micro"]
    assert (exact["tp"], exact["fp"], exact["fn"]) == (4, 1, 2)


def test_cache_miss_reaches_for_the_model(
    monkeypatch, tmp_path, text_data, sample_ground_truth
):
    """With an empty cache the runner must try to load opf — the
    poisoned import proves the lazy-load boundary sits exactly at the
    first uncached document."""
    monkeypatch.setitem(sys.modules, "opf", None)
    with pytest.raises(ImportError):
        run_privacy_filter(text_data, sample_ground_truth,
                           cache_dir=tmp_path)
