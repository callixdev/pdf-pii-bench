import json
import re
from pathlib import Path

from src.scorer import Span

DATA_DIR = Path("data")
TEXT_DATA_PATH = DATA_DIR / "text_data.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"

CONTACT_LABELS = frozenset({
    "private_email", "private_phone", "private_url", "private_address",
})


def load_text_data(path=TEXT_DATA_PATH):
    return json.loads(Path(path).read_text())


def load_ground_truth(path=GROUND_TRUTH_PATH):
    """Returns {doc_id: [Span, ...]}."""
    raw = json.loads(Path(path).read_text())
    return {
        doc_id: [
            Span(s["start"], s["end"], s["text"], s["label"],
                 s.get("subject", "person"))
            for s in spans
        ]
        for doc_id, spans in raw.items()
    }


def _entity_pattern(entity_text):
    """Regex matching the entity with any whitespace run between words,
    guarded so it never matches inside a larger word or number."""
    parts = [re.escape(p) for p in entity_text.split()]
    body = r"\s+".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])")


def find_value_spans(text, values):
    """Match unique (text, label, subject) value triples against the
    canonical text, annotating every occurrence of each value.

    Longest values claim their regions first; occurrences overlapping
    an already-claimed region are skipped. Performs no validation of
    the values themselves — callers own those checks. Returns
    [Span, ...] sorted by position.
    """
    claimed = []  # list of (start, end, label, subject)
    for value_text, label, subject in sorted(
        set(values), key=lambda v: (-len(v[0]), v)
    ):
        for m in _entity_pattern(value_text).finditer(text):
            if any(m.start() < e and m.end() > s for s, e, _, _ in claimed):
                continue
            claimed.append((m.start(), m.end(), label, subject))
    return [
        Span(start, end, text[start:end], label, subject)
        for start, end, label, subject in sorted(claimed)
    ]
