"""Shared fixtures for the pdf-pii-bench tests.

Three hand-written sample documents in the dataset format:
canonical text (as produced by PDF extraction), ground-truth spans
in privacy-filter labels, and canned adapter output matching the
adapter's JSONL contract ({id, text, spans}).
"""

import pytest

from src.scorer import Span

# ---------------------------------------------------------------------------
# 3 hand-written sample documents
# ---------------------------------------------------------------------------

_DOC1_TEXT = "Contact Jane Wilson at jane@example.com or (555) 867-5309."
_DOC2_TEXT = "Tom Brown, age 30, lives at 42 Oak Ave, Portland, OR 97201."
_DOC3_TEXT = (
    "User jsmith (SSN: 123-45-6789) logged in from Boston"
    " with password Tr0ub4dor."
)

_SAMPLE_DOCUMENTS = [
    {"id": "doc-1", "text": _DOC1_TEXT, "category": "email_footer"},
    {"id": "doc-2", "text": _DOC2_TEXT, "category": "lease"},
    {"id": "doc-3", "text": _DOC3_TEXT, "category": "secrets"},
]


@pytest.fixture(scope="session")
def sample_documents():
    """3 hand-written dataset-format documents covering mixed PII types."""
    return [dict(doc) for doc in _SAMPLE_DOCUMENTS]


# -- Ground truth in privacy-filter label space ------------------------------

_SAMPLE_GROUND_TRUTH = {
    # Doc 1: name, email, and an org support line (neutral in the
    # personal-PII view — detecting it is neither rewarded nor punished)
    "doc-1": [
        Span(8,  19, "Jane Wilson",      "private_person"),
        Span(23, 39, "jane@example.com", "private_email"),
        Span(43, 57, "(555) 867-5309",   "private_phone", "org"),
    ],
    # Doc 2: name, full address
    "doc-2": [
        Span(0,  9,  "Tom Brown",                      "private_person"),
        Span(28, 58, "42 Oak Ave, Portland, OR 97201", "private_address"),
    ],
    # Doc 3: SSN, password
    "doc-3": [
        Span(18, 29, "123-45-6789", "account_number"),
        Span(67, 76, "Tr0ub4dor",   "secret"),
    ],
}


@pytest.fixture(scope="session")
def sample_ground_truth():
    """Ground-truth Span objects per document id."""
    return {doc_id: list(spans) for doc_id, spans in _SAMPLE_GROUND_TRUTH.items()}


# -- Canned adapter output (single full-pipeline config) ---------------------

_SAMPLE_ADAPTER_OUTPUT = [
    # Doc 1: everything found
    {
        "id": "doc-1",
        "text": _DOC1_TEXT,
        "spans": [
            {"start": 8,  "end": 19, "text": "Jane Wilson",      "label": "private_person"},
            {"start": 23, "end": 39, "text": "jane@example.com", "label": "private_email"},
            {"start": 43, "end": 57, "text": "(555) 867-5309",   "label": "private_phone"},
        ],
    },
    # Doc 2: address found with slightly narrower extent (still overlap-match)
    {
        "id": "doc-2",
        "text": _DOC2_TEXT,
        "spans": [
            {"start": 0,  "end": 9,  "text": "Tom Brown",                  "label": "private_person"},
            {"start": 28, "end": 52, "text": "42 Oak Ave, Portland, OR",   "label": "private_address"},
        ],
    },
    # Doc 3: SSN found, password missed (a false negative)
    {
        "id": "doc-3",
        "text": _DOC3_TEXT,
        "spans": [
            {"start": 18, "end": 29, "text": "123-45-6789", "label": "account_number"},
        ],
    },
]


@pytest.fixture(scope="session")
def sample_adapter_output():
    """Canned adapter JSONL output for each sample document."""
    return [dict(out) for out in _SAMPLE_ADAPTER_OUTPUT]
