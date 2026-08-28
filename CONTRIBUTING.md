# Contributing

The dataset's source of truth is `data/ground_truth.json` +
`data/text_data.json` (+ the PDFs in `data/pdfs/`). Annotations must be added to these
files directly and must pass the validator:

```bash
uv run python -m src.validate
```

The validator checks every span: offsets (`text[start:end] ==
span.text`), labels in the eight-label vocabulary, a `subject` tag
(`person | org | agency`) on every contact-info span, and
every-occurrence completeness (each other occurrence of an annotated
value in the same document must also be annotated — matching is
whitespace-tolerant with word-boundary guards). CI runs it on every
push. The fabrication rules below are convention, checked by
maintainers in review.

## Annotation corrections

If a span is wrongly annotated, missing, mislabelled, or has the wrong
extent or subject tag, open an issue or a PR editing
`data/ground_truth.json` directly. Reference the annotation policy in
`DATASET.md` (the routing-target test for subject tags,
per-visual-line addresses, what is deliberately not annotated).

Because published scores are only comparable against a fixed ground
truth, corrections are **batched into tagged dataset point releases**
(v1.1, …) rather than applied continuously; maintainers re-score the
reference model when a release is cut.

## New documents

A new document is a PR adding three things:

- the PDF (`data/pdfs/<doc_id>.pdf`) — realistic layout, US-locale;
- its canonical text (`data/text_data.json` entry, with a `category`)
  — produced by any extractor you like; whatever you commit *is* the
  canonical text that systems are scored against;
- its spans (`data/ground_truth.json` entry) — char offsets into that
  text, validator-clean.

**Every PII value must be fabricated**, following the dataset's
conventions (see `DATASET.md`):

- phone numbers in the reserved fictional range 555-0100–555-0199;
- SSNs with never-issued group digits "00" (or the SSA's reserved
  advertising range);
- routing numbers that pass the ABA checksum but match no real
  allocation;
- invented people, organizations, EINs, and account numbers; no real
  person's data, ever.

New documents land in dataset point releases and are scored for the
reference model (`opf`) on arrival.

## Code

Bug reports and fixes to the scorer/harness are welcome as issues or
pull requests. Run `uv run pytest` before submitting; CI runs the test
suite, the ground-truth validator, and a published-results freshness
check on every push.

## Regenerating published results

`RESULTS.md` and the results block in the README are generated —
never edit them by hand. After anything that changes scores (a
ground-truth correction, a scorer change):

```bash
uv run --extra baseline python -m src.run_privacy_filter  # instant when fully cached
uv run python -m src.publish                              # RESULTS.md + README block
```

CI runs `src.publish --check` against the committed
`results/privacy-filter.json` and fails if the published docs are
stale.
