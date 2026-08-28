"""Per-document prediction cache.

Model inference is the only expensive part of a run, and dataset
documents are immutable once generated. Both runners cache
predictions per document id (results/cache/<system>/<doc_id>.json)
and skip documents that already have an entry. Scoring is cheap and
always recomputes from the cached predictions.

Entries are written as soon as each document finishes, so an
interrupted run keeps its completed documents. The cache does not
detect system changes — after updating a system, re-run it
with --no-cache (or delete that system's cache directory).
"""

import json
from pathlib import Path

CACHE_ROOT = Path("results") / "cache"


def default_cache_dir(system):
    return CACHE_ROOT / system


def load_cached(cache_dir, doc_id):
    """Return the cached span dicts for doc_id, or None if not cached."""
    path = Path(cache_dir) / f"{doc_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["spans"]


def save_cached(cache_dir, doc_id, spans):
    """Write span dicts ({start, end, text, label}) for doc_id."""
    path = Path(cache_dir) / f"{doc_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": doc_id, "spans": spans}, indent=2) + "\n")
