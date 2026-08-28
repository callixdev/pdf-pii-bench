"""Drive a bring-your-own-system adapter over the dataset PDFs and
score it against ground truth.

The adapter is a subprocess speaking JSONL: stdin gets one
`{"id": ..., "pdf": ...}` line per document; stdout must emit one
`{"id": ..., "text": ..., "spans": [...]}` line per input. Offsets
point into the canonical text: the emitted `text` must byte-match
data/text_data.json (if your extraction differs, map your offsets to
the canonical text or score via src.score_predictions instead).

Usage:
    uv run python -m src.harness --adapter "your-command" \
        [--system name] [--output results/<name>.json] [--verbose]
"""

import argparse
import json
import shlex
import subprocess
import tempfile
from datetime import date
from pathlib import Path

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

PDF_DIR = Path("data/pdfs")


def run_adapter(docs, adapter_cmd, verbose=False, on_output=None):
    """Feed {id, pdf} JSONL to the adapter subprocess, return parsed outputs.

    docs maps doc_id -> pdf path. Output lines are consumed as the
    adapter emits them; on_output (if given) is called with each parsed
    output so callers can persist per-document results before the run
    completes.
    """
    input_lines = []
    for doc_id, pdf_path in docs.items():
        doc = {"id": doc_id, "pdf": str(pdf_path)}
        if verbose:
            doc["verbose"] = True
        input_lines.append(json.dumps(doc))
    input_data = "\n".join(input_lines) + "\n"

    if isinstance(adapter_cmd, str):
        adapter_cmd = shlex.split(adapter_cmd)

    with tempfile.TemporaryFile(mode="w+") as stderr_file:
        proc = subprocess.Popen(
            adapter_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )
        try:
            try:
                proc.stdin.write(input_data)
                proc.stdin.close()
            except BrokenPipeError:
                pass  # adapter exited early; surfaced via returncode below

            outputs = []
            for line in proc.stdout:
                if not line.strip():
                    continue
                out = json.loads(line)
                outputs.append(out)
                if on_output is not None:
                    on_output(out)
        except BaseException:
            proc.kill()
            proc.wait()
            raise

        if proc.wait() != 0:
            stderr_file.seek(0)
            raise RuntimeError(
                f"adapter exited {proc.returncode}:\n{stderr_file.read()[-2000:]}"
            )

    if len(outputs) != len(docs):
        raise RuntimeError(
            f"adapter returned {len(outputs)} lines for {len(docs)} documents"
        )
    return outputs


def run_with_cache(docs, text_data, *, cache_dir,
                   adapter_cmd, verbose=False, no_cache=False):
    """Run the adapter on documents missing from the cache; return
    outputs for every document, read back from the cache.

    Each adapter output is consistency-checked against the canonical
    text and cached as it arrives, so an interrupted run keeps its
    completed documents.
    """
    to_run = {
        doc_id: pdf_path for doc_id, pdf_path in docs.items()
        if no_cache or load_cached(cache_dir, doc_id) is None
    }
    print(f"adapter: {len(docs) - len(to_run)} cached, running {len(to_run)}")
    if to_run:
        def on_output(out):
            doc_id = out["id"]
            if out["text"] != text_data[doc_id]["text"]:
                raise RuntimeError(
                    f"{doc_id}: adapter text differs from the canonical "
                    "text_data.json — map your offsets to the canonical "
                    "text or score via src.score_predictions"
                )
            save_cached(cache_dir, doc_id, out["spans"])

        run_adapter(to_run, adapter_cmd=adapter_cmd, verbose=verbose,
                    on_output=on_output)

    return [
        {
            "id": doc_id,
            "text": text_data[doc_id]["text"],
            "spans": load_cached(cache_dir, doc_id),
        }
        for doc_id in docs
    ]


def score_outputs(outputs, ground_truth, text_data, verbose=False,
                  system="adapter"):
    doc_modes = []
    for out in outputs:
        doc_id = out["id"]
        gt = ground_truth[doc_id]

        canonical = text_data[doc_id]["text"]
        if out["text"] != canonical:
            raise RuntimeError(
                f"{doc_id}: adapter text differs from the canonical "
                "text_data.json — map your offsets to the canonical "
                "text or score via src.score_predictions"
            )

        predicted = [
            Span(s["start"], s["end"], s["text"], s["label"]) for s in out["spans"]
        ]
        modes = score_document_modes(predicted, gt)
        if verbose:
            print(format_verbose(doc_id, canonical, gt, predicted,
                                 modes["overlap"].matches))
        doc_modes.append(modes)

    return {
        "system": system,
        "date": date.today().isoformat(),
        "n_documents": len(outputs),
        **corpus_modes_to_dict(doc_modes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-data", default=TEXT_DATA_PATH)
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_PATH)
    parser.add_argument("--pdf-dir", default=PDF_DIR,
                        help="directory holding <doc_id>.pdf per document")
    parser.add_argument("--adapter", required=True,
                        help="adapter command (the subprocess to drive)")
    parser.add_argument("--system", default="adapter",
                        help="system name stamped into the results file")
    parser.add_argument("--output", default=None,
                        help="defaults to results/<system>.json")
    parser.add_argument("--cache-dir", default=None,
                        help="per-document prediction cache "
                             "(defaults to results/cache/<system>/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-run every document (use after changing the system)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text_data = load_text_data(args.text_data)
    ground_truth = load_ground_truth(args.ground_truth)
    docs = {doc_id: Path(args.pdf_dir) / f"{doc_id}.pdf" for doc_id in text_data}
    cache_dir = args.cache_dir or default_cache_dir(args.system)

    outputs = run_with_cache(
        docs, text_data, cache_dir=cache_dir,
        adapter_cmd=args.adapter, verbose=args.verbose, no_cache=args.no_cache,
    )
    results = score_outputs(outputs, ground_truth, text_data,
                            verbose=args.verbose, system=args.system)

    out_path = Path(args.output or f"results/{args.system}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    micro = results["overlap"]["micro"]
    print(f"{args.system} (overlap): P={micro['precision']:.3f} "
          f"R={micro['recall']:.3f} F1={micro['f1']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
