"""Tests for src/score_predictions.py — the text-in scoring path."""

import json

import pytest

from src.score_predictions import load_predictions, score_predictions
from src.scorer import Span

TEXT = "Contact Dana Whitfield at dana@example.com for access."

TEXT_DATA = {"doc1": {"text": TEXT, "n_pages": 1}}

GROUND_TRUTH = {
    "doc1": [
        Span(8, 22, "Dana Whitfield", "private_person"),
        Span(26, 42, "dana@example.com", "private_email"),
    ]
}


def _write_predictions(tmp_path, payload):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_valid_predictions(tmp_path):
    path = _write_predictions(tmp_path, {
        "doc1": [{"start": 8, "end": 22, "text": "Dana Whitfield",
                  "label": "private_person"}],
    })
    preds = load_predictions(path, TEXT_DATA)
    assert preds["doc1"] == [Span(8, 22, "Dana Whitfield", "private_person")]


def test_unknown_doc_id_rejected(tmp_path):
    path = _write_predictions(tmp_path, {"nope": []})
    with pytest.raises(ValueError, match="unknown document ids: nope"):
        load_predictions(path, TEXT_DATA)


def test_unknown_label_rejected(tmp_path):
    path = _write_predictions(tmp_path, {
        "doc1": [{"start": 8, "end": 22, "text": "Dana Whitfield",
                  "label": "name"}],
    })
    with pytest.raises(ValueError, match="unknown label"):
        load_predictions(path, TEXT_DATA)


def test_offset_text_mismatch_rejected(tmp_path):
    path = _write_predictions(tmp_path, {
        "doc1": [{"start": 0, "end": 14, "text": "Dana Whitfield",
                  "label": "private_person"}],
    })
    with pytest.raises(ValueError, match="does not match"):
        load_predictions(path, TEXT_DATA)


def test_non_object_file_rejected(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps([{"start": 0}]))
    with pytest.raises(ValueError, match="JSON object"):
        load_predictions(path, TEXT_DATA)


def test_perfect_predictions_score_one():
    preds = {"doc1": list(GROUND_TRUTH["doc1"])}
    results = score_predictions(preds, TEXT_DATA, GROUND_TRUTH)
    for mode in ("exact", "overlap"):
        micro = results[mode]["micro"]
        assert micro == {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                         "tp": 2, "fp": 0, "fn": 0}
    # redaction is char-level, whitespace excluded:
    # 13 ("Dana Whitfield" minus its space) + 16 = 29 GT chars, all covered
    assert results["redaction"]["micro"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "tp": 29, "fp": 0, "fn": 0}


def test_wrong_label_counts_only_in_redaction():
    preds = {"doc1": [Span(8, 22, "Dana Whitfield", "private_address"),
                      Span(26, 42, "dana@example.com", "private_email")]}
    results = score_predictions(preds, TEXT_DATA, GROUND_TRUTH)
    assert results["overlap"]["micro"]["tp"] == 1
    assert results["exact"]["micro"]["tp"] == 1
    assert results["redaction"]["micro"]["tp"] == 29  # chars, label-agnostic
    assert results["redaction"]["micro"]["recall"] == 1.0


def test_missing_documents_score_as_misses():
    results = score_predictions({}, TEXT_DATA, GROUND_TRUTH)
    micro = results["overlap"]["micro"]
    assert micro["tp"] == 0
    assert micro["fn"] == 2
    assert results["n_documents"] == 1


def test_output_shape_matches_reference_runners():
    results = score_predictions({}, TEXT_DATA, GROUND_TRUTH, system="mysys")
    assert results["system"] == "mysys"
    assert set(results) == {"system", "date", "n_documents",
                            "exact", "overlap", "redaction"}
    for mode in ("exact", "overlap"):
        assert set(results[mode]) == {"micro", "per_label"}
    assert set(results["redaction"]) == {"micro"}  # no per-label rows
