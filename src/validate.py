"""Validate the ground-truth annotations against the canonical text.

data/ground_truth.json holds PII annotations as character-offset
spans into each document's canonical text (data/text_data.json).
Annotations are edited directly, with no build step to regenerate
them, so this validator is the CI gate that keeps hand edits honest.

Per document it checks:

- Structural integrity: doc ids match across both files;
  text[start:end] equals each span's stored text; labels come from
  CANONICAL_LABELS; contact-info spans carry a subject tag (person |
  org | agency — org/agency spans are neutral in scoring); spans
  never overlap.

- Every-occurrence completeness: an annotated value must be annotated
  at every occurrence in its document — a half-annotated repeated
  name would charge a detector false positives for correct
  detections. The annotated values are re-matched against the text
  (find_value_spans) and must reproduce the committed spans exactly,
  which also rejects extents the matcher could never produce, such
  as "Dan" inside "Danville".


Usage:
    uv run python -m src.validate
        [--ground-truth data/ground_truth.json]
        [--text-data data/text_data.json]

Prints one ERROR line per violation and exits 1 if there are any.
"""

import argparse
import json
import sys
from pathlib import Path

from src.dataset_utils import (
    CONTACT_LABELS,
    GROUND_TRUTH_PATH,
    TEXT_DATA_PATH,
    find_value_spans,
)
from src.scorer import CANONICAL_LABELS, SUBJECTS


def _check_structure(doc_id, spans, text):
    errors = []
    for i, s in enumerate(spans):
        where = f"span {i} ({s.get('label')} {s.get('text')!r})"
        start, end = s.get("start"), s.get("end")
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(text)):
            errors.append(f"{where}: bad offsets {start}..{end}")
            continue
        if text[start:end] != s.get("text"):
            errors.append(
                f"{where}: text mismatch — canonical text has "
                f"{text[start:end]!r} at {start}..{end}"
            )
        if s.get("label") not in CANONICAL_LABELS:
            errors.append(f"{where}: unknown label")
        if s.get("label") in CONTACT_LABELS and "subject" not in s:
            errors.append(
                f"{where}: contact info must declare a subject "
                f"(person | org | agency)"
            )
        if s.get("subject", "person") not in SUBJECTS:
            errors.append(f"{where}: invalid subject {s.get('subject')!r}")

    by_start = sorted(
        (s for s in spans
         if isinstance(s.get("start"), int) and isinstance(s.get("end"), int)),
        key=lambda s: (s["start"], s["end"]),
    )
    for a, b in zip(by_start, by_start[1:]):
        if a["end"] > b["start"]:
            errors.append(
                f"overlapping spans: {a['label']} {a['text']!r} "
                f"({a['start']}..{a['end']}) and {b['label']} {b['text']!r} "
                f"({b['start']}..{b['end']})"
            )
    return errors


def _check_occurrences(doc_id, spans, text):
    """Re-match the annotated values with the dataset matcher and
    require it to reproduce the annotation exactly."""
    values = {
        (s["text"], s["label"], s.get("subject", "person")) for s in spans
    }
    rebuilt = find_value_spans(text, values)
    errors = []

    annotated = {
        (s["start"], s["end"], s["label"], s.get("subject", "person"))
        for s in spans
    }
    reproduced = {(s.start, s.end, s.label, s.subject) for s in rebuilt}
    for start, end, label, _ in sorted(reproduced - annotated):
        errors.append(
            f"unannotated occurrence of {label} {text[start:end]!r} "
            f"at {start}..{end} (every occurrence of an annotated value "
            f"must be annotated)"
        )
    for start, end, label, _ in sorted(annotated - reproduced):
        errors.append(
            f"span not reproducible by matching: {label} "
            f"{text[start:end]!r} at {start}..{end}"
        )
    return errors


def validate(ground_truth, text_data):
    """ground_truth: raw {doc_id: [span dict, ...]} (as committed);
    text_data: {doc_id: {"text": ...}}. Returns a list of
    "doc_id: message" error strings."""
    errors = []

    for doc_id in sorted(set(ground_truth) - set(text_data)):
        errors.append(f"{doc_id}: in ground truth but not in text data")
    for doc_id in sorted(set(text_data) - set(ground_truth)):
        errors.append(f"{doc_id}: in text data but not in ground truth")

    for doc_id, spans in ground_truth.items():
        if doc_id not in text_data:
            continue
        text = text_data[doc_id]["text"]

        structure = _check_structure(doc_id, spans, text)
        errors.extend(f"{doc_id}: {e}" for e in structure)
        # Occurrence re-matching assumes structurally sound spans.
        if not structure:
            errors.extend(
                f"{doc_id}: {e}" for e in _check_occurrences(doc_id, spans, text)
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_PATH)
    parser.add_argument("--text-data", default=TEXT_DATA_PATH)
    args = parser.parse_args()

    ground_truth = json.loads(Path(args.ground_truth).read_text())
    text_data = json.loads(Path(args.text_data).read_text())

    errors = validate(ground_truth, text_data)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    n_spans = sum(len(v) for v in ground_truth.values())
    if errors:
        print(
            f"FAIL: {len(errors)} error(s) across "
            f"{len(ground_truth)} documents"
        )
        sys.exit(1)
    print(f"OK: {n_spans} spans across {len(ground_truth)} documents")


if __name__ == "__main__":
    main()
