"""Mock adapter for integration tests.

Reads JSONL from stdin ({"id", "pdf"}), looks up canned responses by
document ID, writes JSONL to stdout ({"id", "text", "spans"}). Mimics
the real adapter's contract without loading PDFs or running inference.

Usage:
    echo '{"id":"doc-1","pdf":"whatever.pdf"}' | python tests/mock_adapter.py
"""

import json
import sys

_DOC1_TEXT = "Contact Jane Wilson at jane@example.com or (555) 867-5309."
_DOC2_TEXT = "Tom Brown, age 30, lives at 42 Oak Ave, Portland, OR 97201."
_DOC3_TEXT = (
    "User jsmith (SSN: 123-45-6789) logged in from Boston"
    " with password Tr0ub4dor."
)

CANNED = {
    "doc-1": {
        "id": "doc-1",
        "text": _DOC1_TEXT,
        "spans": [
            {"start": 8,  "end": 19, "text": "Jane Wilson",      "label": "private_person"},
            {"start": 23, "end": 39, "text": "jane@example.com", "label": "private_email"},
            {"start": 43, "end": 57, "text": "(555) 867-5309",   "label": "private_phone"},
        ],
    },
    "doc-2": {
        "id": "doc-2",
        "text": _DOC2_TEXT,
        "spans": [
            {"start": 0,  "end": 9,  "text": "Tom Brown",                "label": "private_person"},
            {"start": 28, "end": 52, "text": "42 Oak Ave, Portland, OR", "label": "private_address"},
        ],
    },
    "doc-3": {
        "id": "doc-3",
        "text": _DOC3_TEXT,
        "spans": [
            {"start": 18, "end": 29, "text": "123-45-6789", "label": "account_number"},
        ],
    },
}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        doc_id = doc["id"]
        canned = CANNED.get(doc_id, {"id": doc_id, "text": "", "spans": []})
        print(json.dumps(canned), flush=True)


if __name__ == "__main__":
    main()
