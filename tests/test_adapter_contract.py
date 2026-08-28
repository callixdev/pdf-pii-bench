"""Tests for the adapter JSONL contract.

These validate the shape of adapter output — id/text/spans keys, valid
span fields, correct labels, offset consistency — without running the
real adapter. They use canned output from conftest fixtures.
"""

from src.scorer import CANONICAL_LABELS


class TestAdapterOutputShape:
    def test_has_required_keys(self, sample_adapter_output):
        for doc in sample_adapter_output:
            assert {"id", "text", "spans"}.issubset(doc.keys())

    def test_has_id(self, sample_adapter_output):
        for doc in sample_adapter_output:
            assert isinstance(doc["id"], str)
            assert len(doc["id"]) > 0

    def test_text_is_string(self, sample_adapter_output):
        for doc in sample_adapter_output:
            assert isinstance(doc["text"], str)


class TestSpanFields:
    def test_span_has_required_fields(self, sample_adapter_output):
        required = {"start", "end", "text", "label"}
        for doc in sample_adapter_output:
            for span in doc["spans"]:
                missing = required - span.keys()
                assert not missing, (
                    f"doc {doc['id']}: span missing {missing}: {span}"
                )

    def test_start_end_are_ints(self, sample_adapter_output):
        for doc in sample_adapter_output:
            for span in doc["spans"]:
                assert isinstance(span["start"], int)
                assert isinstance(span["end"], int)

    def test_start_before_end(self, sample_adapter_output):
        for doc in sample_adapter_output:
            for span in doc["spans"]:
                assert span["start"] < span["end"], (
                    f"doc {doc['id']}: start={span['start']} >= end={span['end']}"
                )

    def test_span_text_nonempty(self, sample_adapter_output):
        for doc in sample_adapter_output:
            for span in doc["spans"]:
                assert isinstance(span["text"], str)
                assert len(span["text"]) > 0


class TestLabels:
    def test_labels_are_valid_pf(self, sample_adapter_output):
        for doc in sample_adapter_output:
            for span in doc["spans"]:
                assert span["label"] in CANONICAL_LABELS, (
                    f"doc {doc['id']}: invalid label '{span['label']}'"
                )


class TestOffsetConsistency:
    def test_offsets_match_text(self, sample_adapter_output):
        """span.text == doc.text[start:end] — offsets are into the
        canonical text the adapter itself returns."""
        for doc in sample_adapter_output:
            for span in doc["spans"]:
                sliced = doc["text"][span["start"]:span["end"]]
                assert sliced == span["text"], (
                    f"doc {doc['id']}: text[{span['start']}:{span['end']}] = "
                    f"{sliced!r} != {span['text']!r}"
                )


class TestIdPassthrough:
    def test_ids_match_and_ordered(self, sample_documents, sample_adapter_output):
        expected_ids = [row["id"] for row in sample_documents]
        actual_ids = [doc["id"] for doc in sample_adapter_output]
        assert actual_ids == expected_ids
