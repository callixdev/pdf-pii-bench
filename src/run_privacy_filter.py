"""Run privacy-filter (opf, default decoding) on the
dataset and score against ground truth.

The model receives the canonical text (data/text_data.json)
— the same text space the ground truth and any harness adapter
use.

Usage:
    uv run python -m src.run_privacy_filter [--output results/privacy-filter.json] [--verbose]
"""

import argparse
import json
from datetime import date
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # baseline extra not installed — a fully cached
    def tqdm(iterable, **kwargs):  # run still scores, just without a bar
        return iterable

from src.cache import default_cache_dir, load_cached, save_cached
from src.dataset_utils import (
    GROUND_TRUTH_PATH,
    TEXT_DATA_PATH,
    load_ground_truth,
    load_text_data,
)
from src.scorer import (
    Span,
    corpus_modes_to_dict,
    format_verbose,
    score_document_modes,
)


def _span_dicts(result):
    """Convert opf RedactionResult.detected_spans to cacheable dicts."""
    return [
        {"start": s.start, "end": s.end, "text": s.text, "label": s.label}
        for s in result.detected_spans
    ]


def run_privacy_filter(text_data, ground_truth, *, device="cpu", verbose=False,
                cache_dir=None, no_cache=False):
    if cache_dir is None:
        cache_dir = default_cache_dir("privacy-filter")
    opf = None  # loaded lazily — a fully cached run never touches the model

    doc_modes = []
    for doc_id, entry in tqdm(text_data.items(), desc="opf redact"):
        text = entry["text"]
        gt = ground_truth[doc_id]
        spans = None if no_cache else load_cached(cache_dir, doc_id)
        if spans is None:
            if opf is None:
                from opf import OPF
                opf = OPF(device=device, output_mode="typed")
                opf.redact("warm up")
            spans = _span_dicts(opf.redact(text))
            save_cached(cache_dir, doc_id, spans)
        predicted = [Span(s["start"], s["end"], s["text"], s["label"]) for s in spans]
        modes = score_document_modes(predicted, gt)
        if verbose:
            print(format_verbose(doc_id, text, gt, predicted,
                                 modes["overlap"].matches))
        doc_modes.append(modes)

    return {
        "system": "privacy-filter",
        "date": date.today().isoformat(),
        "n_documents": len(text_data),
        **corpus_modes_to_dict(doc_modes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-data", default=TEXT_DATA_PATH)
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_PATH)
    parser.add_argument("--output", default="results/privacy-filter.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=default_cache_dir("privacy-filter"),
                        help="per-document prediction cache")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-run every document (use after updating opf)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text_data = load_text_data(args.text_data)
    ground_truth = load_ground_truth(args.ground_truth)

    results = run_privacy_filter(
        text_data, ground_truth, device=args.device, verbose=args.verbose,
        cache_dir=args.cache_dir, no_cache=args.no_cache,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    micro = results["overlap"]["micro"]
    print(f"privacy-filter (overlap): P={micro['precision']:.3f} "
          f"R={micro['recall']:.3f} F1={micro['f1']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
