"""Integration tests for the adapter harness.

Tests the full flow through src.harness: doc list → subprocess
mock adapter → parse output → consistency-check text → score.
"""

import json
import sys
from pathlib import Path

import pytest

from src.harness import run_adapter, score_outputs

MOCK_ADAPTER = f"{sys.executable} {Path(__file__).parent / 'mock_adapter.py'}"


@pytest.fixture
def docs(sample_documents):
    return {row["id"]: f"pdfs/{row['id']}.pdf" for row in sample_documents}


@pytest.fixture
def text_data(sample_documents):
    return {row["id"]: {"text": row["text"], "n_pages": 1} for row in sample_documents}


class TestFullFlow:
    def test_round_trip_produces_results(self, docs):
        outputs = run_adapter(docs, adapter_cmd=MOCK_ADAPTER)
        assert len(outputs) == len(docs)
        for out in outputs:
            assert {"id", "text", "spans"}.issubset(out.keys())

    def test_ids_preserved_in_order(self, docs):
        outputs = run_adapter(docs, adapter_cmd=MOCK_ADAPTER)
        assert [o["id"] for o in outputs] == list(docs.keys())

    def test_scoring_produces_expected_metrics(
        self, docs, text_data, sample_ground_truth
    ):
        """The canned adapter output misses doc-2's full address extent
        (overlap-matched) and doc-3's password (FN); doc-1's phone is
        an org support line, so it leaves recall and the adapter's
        correct prediction of it is neither TP nor FP — check the
        scorer sees exactly that."""
        outputs = run_adapter(docs, adapter_cmd=MOCK_ADAPTER)
        results = score_outputs(outputs, sample_ground_truth, text_data)

        assert results["system"] == "adapter"
        assert results["n_documents"] == 3

        overlap = results["overlap"]["micro"]
        # 6 scored GT spans (the org phone is neutral), 6 predictions;
        # the org-phone prediction is absorbed, the rest are correct
        assert overlap["tp"] == 5
        assert overlap["fp"] == 0
        assert overlap["fn"] == 1  # doc-3's password
        assert overlap["precision"] == 1.0

        exact = results["exact"]["micro"]
        # doc-2's address prediction is narrower than GT → not an exact
        # match (an FP); the absorbed org phone is not
        assert exact["tp"] == 4
        assert exact["fp"] == 1
        assert exact["fn"] == 2

    def test_text_mismatch_raises(self, docs, text_data, sample_ground_truth):
        outputs = run_adapter(docs, adapter_cmd=MOCK_ADAPTER)
        text_data["doc-1"]["text"] = "different text"
        with pytest.raises(RuntimeError, match="differs from the canonical"):
            score_outputs(outputs, sample_ground_truth, text_data)


class TestSubprocessErrors:
    def test_nonzero_exit_raises(self, docs):
        bad_cmd = f"{sys.executable} -c 'import sys; sys.exit(1)'"
        with pytest.raises(RuntimeError, match="exited 1"):
            run_adapter(docs, adapter_cmd=bad_cmd)

    def test_malformed_jsonl_raises(self, docs):
        bad_cmd = f"{sys.executable} -c 'print(\"not json\")'"
        with pytest.raises(json.JSONDecodeError):
            run_adapter(docs, adapter_cmd=bad_cmd)

    def test_missing_documents_raises(self, docs):
        """Adapter returning fewer lines than documents is an error."""
        one_line = (
            f"{sys.executable} -c "
            "'import json; print(json.dumps({\"id\": \"doc-1\", \"text\": \"\", \"spans\": []}))'"
        )
        with pytest.raises(RuntimeError, match="returned 1 lines for 3"):
            run_adapter(docs, adapter_cmd=one_line)
