"""Tests for the per-document prediction cache (src/cache.py) and the
cache-aware harness flow (src.harness.run_with_cache)."""

import sys
from pathlib import Path

import pytest

from src.cache import load_cached, save_cached
from src.harness import run_with_cache

MOCK_ADAPTER = f"{sys.executable} {Path(__file__).parent / 'mock_adapter.py'}"
FAILING_ADAPTER = f"{sys.executable} -c 'import sys; sys.exit(1)'"

SPANS = [{"start": 0, "end": 9, "text": "Tom Brown", "label": "private_person"}]


class TestCacheStore:
    def test_round_trip(self, tmp_path):
        save_cached(tmp_path, "doc-1", SPANS)
        assert load_cached(tmp_path, "doc-1") == SPANS

    def test_missing_returns_none(self, tmp_path):
        assert load_cached(tmp_path, "doc-1") is None

    def test_empty_spans_is_a_hit(self, tmp_path):
        """A document with no detections is still cached (not re-run)."""
        save_cached(tmp_path, "doc-1", [])
        assert load_cached(tmp_path, "doc-1") == []

    def test_creates_cache_dir(self, tmp_path):
        save_cached(tmp_path / "nested" / "dir", "doc-1", SPANS)
        assert load_cached(tmp_path / "nested" / "dir", "doc-1") == SPANS


@pytest.fixture
def docs(sample_documents):
    return {row["id"]: f"pdfs/{row['id']}.pdf" for row in sample_documents}


@pytest.fixture
def text_data(sample_documents):
    return {row["id"]: {"text": row["text"], "n_pages": 1} for row in sample_documents}


class TestRunWithCache:
    def test_first_run_populates_cache(self, docs, text_data, tmp_path):
        outputs = run_with_cache(docs, text_data, cache_dir=tmp_path,
                                 adapter_cmd=MOCK_ADAPTER)
        assert [o["id"] for o in outputs] == list(docs.keys())
        for doc_id in docs:
            assert load_cached(tmp_path, doc_id) is not None

    def test_cached_docs_skip_the_adapter(self, docs, text_data, tmp_path):
        run_with_cache(docs, text_data, cache_dir=tmp_path,
                       adapter_cmd=MOCK_ADAPTER)
        # all docs cached → the (failing) adapter must never be invoked
        outputs = run_with_cache(docs, text_data, cache_dir=tmp_path,
                                 adapter_cmd=FAILING_ADAPTER)
        assert len(outputs) == len(docs)

    def test_partial_cache_runs_only_missing_docs(self, docs, text_data,
                                                  tmp_path):
        first = dict(list(docs.items())[:1])
        run_with_cache(first, text_data, cache_dir=tmp_path,
                       adapter_cmd=MOCK_ADAPTER)
        outputs = run_with_cache(docs, text_data, cache_dir=tmp_path,
                                 adapter_cmd=MOCK_ADAPTER)
        assert [o["id"] for o in outputs] == list(docs.keys())

    def test_no_cache_reruns_everything(self, docs, text_data, tmp_path):
        run_with_cache(docs, text_data, cache_dir=tmp_path,
                       adapter_cmd=MOCK_ADAPTER)
        with pytest.raises(RuntimeError, match="exited 1"):
            run_with_cache(docs, text_data, cache_dir=tmp_path,
                           adapter_cmd=FAILING_ADAPTER, no_cache=True)

    def test_text_mismatch_not_cached(self, docs, text_data, tmp_path):
        text_data["doc-1"] = {"text": "different text", "n_pages": 1}
        with pytest.raises(RuntimeError, match="differs from the canonical"):
            run_with_cache(docs, text_data, cache_dir=tmp_path,
                           adapter_cmd=MOCK_ADAPTER)
        assert load_cached(tmp_path, "doc-1") is None

    def test_outputs_carry_canonical_text(self, docs, text_data, tmp_path):
        outputs = run_with_cache(docs, text_data, cache_dir=tmp_path,
                                 adapter_cmd=MOCK_ADAPTER)
        for out in outputs:
            assert out["text"] == text_data[out["id"]]["text"]
