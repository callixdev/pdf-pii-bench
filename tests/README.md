# Test Plan

Tests use pytest. The goal is to verify every data transformation and
scoring calculation independently.

## Structure

```
tests/
├── conftest.py                 ← shared fixtures (sample documents,
│                                 ground truth, canned adapter output)
├── test_dataset_utils.py       ← occurrence matcher (find_value_spans)
├── test_validate.py            ← ground-truth validator (CI gate)
├── test_scorer.py              ← span matching + metrics (all three modes)
├── test_cache.py               ← per-document prediction cache
├── test_harness_integration.py ← end-to-end with mock adapter
├── test_adapter_contract.py    ← JSONL contract validation
├── test_publish.py             ← published docs (RESULTS.md + README block)
├── test_run_privacy_filter.py  ← baseline runner's cached scoring path
└── mock_adapter.py             ← canned stand-in for a real adapter
```

## test_dataset_utils.py

`find_value_spans`, the occurrence matcher behind the validator's
every-occurrence check — a bug here silently weakens the CI gate:

- Simple match produces correct offsets (`text[start:end] ==
  span.text`) and carries label and subject through to the span
- Whitespace-flexible matching: a multi-word value matches across a
  line break or any whitespace run
- Regex metacharacters in values (`(555) 867-5309`, `+1 …`) are
  escaped, not interpreted
- Boundary guards: a value never matches inside a larger word or a
  longer digit run
- Every occurrence is matched, not just the first
- Longest values claim their regions first ("Daniel" never matches
  inside an annotated "Daniel Okafor")
- Output is position-sorted regardless of value processing order

## test_validate.py

The ground-truth validator (the CI gate for direct edits to
`ground_truth.json`):

- Structure: offset/text mismatches, out-of-range offsets, unknown
  labels, missing/invalid subjects on contact spans, overlapping
  spans all error; a clean document passes
- Every-occurrence self-consistency: an unannotated repeat of an
  annotated value errors; a span the matcher would never produce
  (inside a larger word) errors; a short value claimed by an
  annotated longer one is not flagged

## test_scorer.py

This is the most critical module — wrong scoring silently produces
plausible-looking but incorrect metrics.

### Span matching

- Exact match: prediction and ground truth have identical start/end
- Partial match (IoU >= 0.5): overlapping spans score as match
- Partial match (IoU < 0.5): overlapping spans don't score as match
- No overlap: adjacent spans ([0,5] and [5,10]) are not a match
- Label mismatch: same offsets but different label → not a match in
  exact/overlap modes (redaction, which is label-agnostic, is scored
  separately at character level — see below)
- Duplicate predictions: two predictions matching one ground truth →
  one TP + one FP (not two TPs)
- Duplicate ground truth: two ground truths matching one prediction →
  one TP + one FN

### Redaction (character-level)

Redaction mode is scored at character level, label-agnostically:
recall = fraction of GT chars covered by any prediction, precision =
fraction of predicted chars inside GT. Reported micro-only — per-label
precision has no meaning when matching ignores labels.

- Wrong-label coverage still counts in full
- Redacting the whole document scores precision = PII density, not 1.0
  (regression test for the span-level metric's containment loophole)
- Fragmented predictions earn proportional recall ("John" alone
  against "John Smith" → 4/9 scoreable chars)
- Whitespace is not scoreable on either side: "John" + "Smith"
  covering "John Smith" scores recall 1.0 (the unredacted space leaks
  nothing), and a prediction spanning the space between two GT words
  pays no FP for it
- Boundary slack costs proportional FP chars; overlapping predictions
  count each char once; non-canonical GT labels are excluded
- Corpus micro sums char counts across documents
- `match_spans` no longer accepts `mode="redaction"`

### Subject tags (neutral spans)

Org/agency spans are neutral regions: only person-tagged spans are
scored.

- A missed org span is not a miss
- A prediction matching an org/agency span is neither TP nor FP —
  absorbed, using the same criterion as the matching mode (boundary
  slack over a neutral span is absorbed under overlap, still an FP
  under exact)
- A prediction matching nothing is an FP
- Person spans are matched before neutral spans, so neutral regions
  only absorb false positives — they never take a match away from a
  person span
- Redaction: predicted chars inside neutral spans are free; chars
  beyond them still pay precision; missed neutral chars don't count
  toward recall
- Neutralized predictions leave no per-label FP
- `score_document_modes` / `corpus_modes_to_dict` shape: three modes;
  redaction is micro-only

### Metrics

- Known-answer P/R/F1: hand-computed on a 5-span example, assert
  exact floating-point values
- All predictions correct → P=1.0, R=1.0, F1=1.0
- No predictions → P=0.0 (by convention), R=0.0, F1=0.0
- All predictions wrong → P=0.0, R=0.0, F1=0.0
- Per-label aggregation: verify label-level metrics sum correctly
  to micro-averaged metrics
- Zero ground truth for a label → recall and F1 are None for that
  label (rendered "–", not scored as R=1.0 or F1=0.0)

### Edge cases

- Empty document (no ground truth, no predictions) → no contribution
  to metrics
- Ground truth spans with non-canonical labels → excluded before scoring
- Overlapping ground truth spans → handled without double-counting

## test_cache.py

Per-document prediction caching (`src/cache.py`):

- Round-trip: saved spans load back identically
- A fully cached document is never sent to the adapter (proved with a
  failing adapter)
- Partial cache: only uncached documents run; entries are written as
  each document finishes, so an interrupted run keeps its progress
- `--no-cache` forces re-runs; a text-consistency failure is not cached

## test_harness_integration.py

End-to-end test of the adapter harness with a mock adapter script that
returns canned JSONL responses (no real model or external system).

- Verifies the full flow: doc list → subprocess call → JSONL read →
  canonical-text consistency check → scoring → results dict
- Known-answer metrics: the canned output has one deliberate FN and one
  narrower-than-ground-truth span, so exact and overlap modes differ in
  a hand-checkable way; doc-1's phone is an org support line, so
  subject-tag neutralization shows up in a hand-checkable way too
- Text mismatch between adapter output and text_data.json → clear error
- Subprocess errors (non-zero exit, malformed JSONL, missing output
  lines) are caught and reported clearly

## test_publish.py

`src/publish.py` generates RESULTS.md and the README results block
from the baseline results file:

- Generated RESULTS.md carries the run date, span count, the scoring
  note, and a do-not-edit header; output is deterministic
- The README block between the results markers is replaced and the
  span counts rewritten; surrounding prose is untouched; the rewrite
  is idempotent
- Missing markers fail loudly instead of silently skipping
- The `--check` gate CI runs: passes (and returns normally) on fresh
  files, exits 1 when a generated file has drifted

## test_run_privacy_filter.py

The baseline runner's cached scoring path, with the opf import
poisoned so any attempt to load the model fails the test:

- A fully cached run scores without touching the model (the path that
  verifies the committed results in seconds) and reproduces the same
  known-answer metrics as the harness integration test
- A cache miss reaches for the model — the lazy-load boundary sits
  exactly at the first uncached document

## test_adapter_contract.py

Validates the adapter JSONL output shape without running the real adapter:

- Every output has id / text / spans; spans have start/end/text/label
- Labels are valid privacy-filter labels
- Offset consistency: `doc.text[start:end] == span.text` for every span
- ID passthrough: output IDs match input IDs in order
