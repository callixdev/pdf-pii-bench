"""Tests for src.validate — the ground-truth validator that gates
direct edits to ground_truth.json."""

from src.validate import validate


def _doc(text, spans):
    return {"doc": spans}, {"doc": {"text": text}}


def _span(start, end, text, label, subject=None):
    s = {"start": start, "end": end, "text": text, "label": label}
    if subject is not None:
        s["subject"] = subject
    return s


def test_clean_document_passes():
    text = "Contact Jane Doe at 555-0134 today."
    gt, td = _doc(text, [
        _span(8, 16, "Jane Doe", "private_person"),
        _span(20, 28, "555-0134", "private_phone", "person"),
    ])
    assert validate(gt, td) == []


def test_doc_id_mismatch():
    errors = validate({"a": []}, {"b": {"text": ""}})
    assert any("a: in ground truth but not in text data" in e for e in errors)
    assert any("b: in text data but not in ground truth" in e for e in errors)


def test_offset_text_mismatch():
    gt, td = _doc("Jane Doe was here.", [
        _span(0, 4, "Doe", "private_person"),
    ])
    assert any("text mismatch" in e for e in validate(gt, td))


def test_out_of_range_offsets():
    gt, td = _doc("short", [_span(2, 99, "ort", "private_person")])
    assert any("bad offsets" in e for e in validate(gt, td))


def test_unknown_label():
    gt, td = _doc("Jane", [_span(0, 4, "Jane", "not_a_label")])
    assert any("unknown label" in e for e in validate(gt, td))


def test_contact_label_requires_subject():
    gt, td = _doc("jane@example.com", [
        _span(0, 16, "jane@example.com", "private_email"),
    ])
    assert any("must declare a subject" in e for e in validate(gt, td))


def test_invalid_subject():
    gt, td = _doc("jane@example.com", [
        _span(0, 16, "jane@example.com", "private_email", "robot"),
    ])
    assert any("invalid subject" in e for e in validate(gt, td))


def test_overlapping_spans():
    text = "Jane Doe"
    gt, td = _doc(text, [
        _span(0, 8, "Jane Doe", "private_person"),
        _span(5, 8, "Doe", "private_person"),
    ])
    assert any("overlapping spans" in e for e in validate(gt, td))


def test_missed_second_occurrence():
    text = "Jane Doe signed. Jane Doe agreed."
    gt, td = _doc(text, [_span(0, 8, "Jane Doe", "private_person")])
    assert any("unannotated occurrence" in e for e in validate(gt, td))


def test_all_occurrences_annotated_passes():
    text = "Jane Doe signed. Jane Doe agreed."
    gt, td = _doc(text, [
        _span(0, 8, "Jane Doe", "private_person"),
        _span(17, 25, "Jane Doe", "private_person"),
    ])
    assert validate(gt, td) == []


def test_span_inside_larger_word_not_reproducible():
    # "Dan" inside "Danville" — the matcher's boundary guard would
    # never produce this span.
    text = "Danville city limits"
    gt, td = _doc(text, [_span(0, 3, "Dan", "private_person")])
    assert any("not reproducible" in e for e in validate(gt, td))


def test_short_value_claimed_by_longer_passes():
    # "Daniel" alone occurs once; its other occurrence is inside the
    # annotated "Daniel Okafor" and must not be flagged.
    text = "Daniel Okafor met Daniel."
    gt, td = _doc(text, [
        _span(0, 13, "Daniel Okafor", "private_person"),
        _span(18, 24, "Daniel", "private_person"),
    ])
    assert validate(gt, td) == []
