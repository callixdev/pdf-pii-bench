"""Tests for src.dataset_utils.find_value_spans — the occurrence
matcher behind the validator's every-occurrence check."""

from src.dataset_utils import find_value_spans


def test_simple_match_offsets():
    text = "Contact Jane Doe today."
    spans = find_value_spans(text, [("Jane Doe", "private_person", "person")])
    assert len(spans) == 1
    s = spans[0]
    assert (s.start, s.end) == (8, 16)
    assert text[s.start:s.end] == s.text == "Jane Doe"
    assert s.label == "private_person"
    assert s.subject == "person"


def test_subject_propagates():
    text = "Email billing@cedarpine.design for invoices."
    spans = find_value_spans(
        text, [("billing@cedarpine.design", "private_email", "org")]
    )
    assert [s.subject for s in spans] == ["org"]


def test_whitespace_flexible_across_line_break():
    # Multi-line PDF values match across any whitespace run.
    text = "Tenant: Jane\n    Doe, hereinafter"
    spans = find_value_spans(text, [("Jane Doe", "private_person", "person")])
    assert len(spans) == 1
    assert text[spans[0].start:spans[0].end] == "Jane\n    Doe"


def test_regex_metacharacters_escaped():
    text = "Call (555) 867-5309 or +1 512-555-0177 now."
    spans = find_value_spans(text, [
        ("(555) 867-5309", "private_phone", "person"),
        ("+1 512-555-0177", "private_phone", "person"),
    ])
    assert [s.text for s in spans] == ["(555) 867-5309", "+1 512-555-0177"]


def test_every_occurrence_matched():
    text = "Jane Doe signed. Jane Doe agreed."
    spans = find_value_spans(text, [("Jane Doe", "private_person", "person")])
    assert [(s.start, s.end) for s in spans] == [(0, 8), (17, 25)]


def test_boundary_guard_word():
    # "Dan" must not match inside "Danville".
    text = "Dan moved to Danville."
    spans = find_value_spans(text, [("Dan", "private_person", "person")])
    assert [(s.start, s.end) for s in spans] == [(0, 3)]


def test_boundary_guard_digit_run():
    # "6741" must not match inside a longer number.
    text = "ending 6741; ref 67415."
    spans = find_value_spans(text, [("6741", "account_number", "person")])
    assert [(s.start, s.end) for s in spans] == [(7, 11)]


def test_longest_value_claims_first():
    text = "Daniel Okafor met Daniel."
    spans = find_value_spans(text, [
        ("Daniel", "private_person", "person"),
        ("Daniel Okafor", "private_person", "person"),
    ])
    assert [(s.text, s.start) for s in spans] == [
        ("Daniel Okafor", 0),
        ("Daniel", 18),
    ]


def test_output_sorted_by_position():
    # "a@x.com" is processed before "b@x.com" (tie-break on value) but
    # occurs later in the text; output must still be position-sorted.
    text = "b@x.com then a@x.com"
    spans = find_value_spans(text, [
        ("a@x.com", "private_email", "person"),
        ("b@x.com", "private_email", "person"),
    ])
    assert [s.start for s in spans] == [0, 13]
