"""Score a predictions file against the dataset ground truth.

For systems that consume the canonical text directly (no PDF adapter):
run your detector over data/text_data.json and write one JSON
file mapping doc id -> list of spans into that text:

    {"bank_statement": [{"start": 120, "end": 136,
                         "text": "Dana Whitfield",
                         "label": "private_person"}, ...], ...}

Usage:
    uv run python -m src.score_predictions predictions.json
        [--system NAME] [--output results/NAME.json] [--verbose]

The output has the same shape as results/privacy-filter.json, so
numbers are directly comparable to the reference results.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from src.dataset_utils import (
    GROUND_TRUTH_PATH,
    TEXT_DATA_PATH,
    load_ground_truth,
    load_text_data,
)
from src.scorer import (
    CANONICAL_LABELS,
    Span,
    corpus_modes_to_dict,
    format_verbose,
    score_document_modes,
)


def load_predictions(path, text_data):
    """Returns {doc_id: [Span, ...]}, validated against the canonical text."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("predictions file must be a JSON object: {doc_id: [spans]}")

    unknown = sorted(set(raw) - set(text_data))
    if unknown:
        raise ValueError(f"unknown document ids: {', '.join(unknown)}")

    predictions = {}
    for doc_id, spans in raw.items():
        text = text_data[doc_id]["text"]
        parsed = []
        for s in spans:
            span = Span(s["start"], s["end"], s["text"], s["label"])
            if span.label not in CANONICAL_LABELS:
                raise ValueError(
                    f"{doc_id}: unknown label {span.label!r} "
                    f"(expected one of {sorted(CANONICAL_LABELS)})")
            if text[span.start:span.end] != span.text:
                raise ValueError(
                    f"{doc_id}: span [{span.start}:{span.end}] does not match "
                    f"the canonical text — offsets must point into "
                    f"data/text_data.json "
                    f"(expected {span.text!r}, found {text[span.start:span.end]!r})")
            parsed.append(span)
        predictions[doc_id] = parsed
    return predictions


def score_predictions(predictions, text_data, ground_truth, *,
                      system="custom", verbose=False):
    """Score predictions for the documents present in the predictions file.

    Documents missing from the file are scored as all-misses, so partial
    files still produce corpus numbers comparable to a full run.
    """
    doc_modes = []
    for doc_id, entry in text_data.items():
        gt = ground_truth[doc_id]
        predicted = predictions.get(doc_id, [])
        modes = score_document_modes(predicted, gt)
        if verbose:
            print(format_verbose(doc_id, entry["text"], gt, predicted,
                                 modes["overlap"].matches))
        doc_modes.append(modes)

    return {
        "system": system,
        "date": date.today().isoformat(),
        "n_documents": len(text_data),
        **corpus_modes_to_dict(doc_modes),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", help="JSON file: {doc_id: [spans]}")
    parser.add_argument("--system", default="custom",
                        help="system name recorded in the output")
    parser.add_argument("--text-data", default=TEXT_DATA_PATH)
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_PATH)
    parser.add_argument("--output", default=None,
                        help="default: results/<system>.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text_data = load_text_data(args.text_data)
    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions, text_data)

    missing = len(text_data) - len(predictions)
    if missing:
        print(f"note: predictions cover {len(predictions)}/{len(text_data)} "
              f"documents; the rest score as all-misses")

    results = score_predictions(predictions, text_data, ground_truth,
                                system=args.system, verbose=args.verbose)

    out_path = Path(args.output or f"results/{args.system}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    for mode in ("redaction", "overlap", "exact"):
        m = results[mode]["micro"]
        print(f"{mode:>9}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
