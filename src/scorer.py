"""Span matching and scoring for PII detection evaluation."""

from collections import defaultdict
from dataclasses import dataclass

CANONICAL_LABELS = frozenset({
    "private_person", "private_email", "private_phone", "private_address",
    "private_url", "private_date", "account_number", "secret",
})

# Who a contact channel reaches (see DATASET.md, "Personal vs.
# organizational: the subject tag"):
# person = an individual; org = a company/role; agency = a government
# agency (e.g. IRS hotlines preprinted on official forms).
SUBJECTS = ("person", "org", "agency")

# Scoring policy: only person-subject spans are scored. Org/agency
# spans are neutral — excluded from recall, and predictions covering
# them are charged no FP (neither rewarded nor penalized).
# Policy-matched to person-linked PII definitions.


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    subject: str = "person"


@dataclass
class DocumentScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    label_counts: dict
    matches: list


@dataclass
class LabelScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float | None
    f1: float | None


@dataclass
class CorpusScore:
    total_tp: int
    total_fp: int
    total_fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_label: dict


def compute_iou(a, b):
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    if intersection == 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def _is_match(pred, gt, mode):
    if mode == "exact":
        return (pred.label == gt.label
                and pred.start == gt.start and pred.end == gt.end)
    if mode == "overlap":
        if pred.label != gt.label:
            return False
        if pred.start <= gt.start and gt.end <= pred.end:
            return True
        return compute_iou(pred, gt) >= 0.5
    raise ValueError(f"Unknown mode: {mode}")


def _strictly_contains(outer, inner):
    return (outer.start <= inner.start and inner.end <= outer.end
            and (outer.end - outer.start) > (inner.end - inner.start))


def match_spans(predicted, ground_truth, mode="exact"):
    """Match predicted spans to ground truth spans.

    Returns (tp, fp, fn, matches) where matches is a list of
    (pred_idx, gt_idx, iou) tuples.

    Two-pass matching in overlap mode:
      1. One-to-one: each prediction gets at most one "primary" match
         (exact boundary or IoU >= 0.5).
      2. Bonus: predictions that strictly contain additional GT spans
         (pred is wider) can match those too.

    This lets "John Smith" match both GT "John" and GT "Smith" while
    preventing duplicate GT spans from inflating TP counts.
    """
    candidates = []
    for pi, pred in enumerate(predicted):
        for gi, gt in enumerate(ground_truth):
            if _is_match(pred, gt, mode):
                iou = compute_iou(pred, gt)
                strict = _strictly_contains(pred, gt)
                candidates.append((pi, gi, iou, strict))

    candidates.sort(key=lambda x: x[2], reverse=True)

    matched_gt = set()
    pred_has_primary = set()
    matched_preds = set()
    matches = []

    for pi, gi, iou, strict in candidates:
        if gi in matched_gt:
            continue
        if pi in pred_has_primary and not strict:
            continue
        matched_gt.add(gi)
        matched_preds.add(pi)
        matches.append((pi, gi, iou))
        if not strict:
            pred_has_primary.add(pi)

    tp = len(matched_gt)
    fp = sum(1 for i in range(len(predicted)) if i not in matched_preds)
    fn = len(ground_truth) - tp

    return tp, fp, fn, matches


def _compute_prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


@dataclass
class RedactionScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _covered_chars(spans):
    chars = set()
    for s in spans:
        chars.update(range(s.start, s.end))
    return chars


def _scoreable_chars(spans):
    """Non-whitespace character offsets covered by spans.

    Whitespace is never scoreable: an unredacted space between two
    redacted words leaks nothing, and redacting a space protects
    nothing. Offsets are derived from span.text, which the validator
    and adapter contract guarantee equals text[start:end].
    """
    chars = set()
    for s in spans:
        chars.update(
            s.start + i for i, ch in enumerate(s.text) if not ch.isspace()
        )
    return chars


def _split_neutral(ground_truth):
    """Partition GT into (scored, neutral) spans: person-subject spans
    are scored, org/agency spans are neutral."""
    gt_all = [s for s in ground_truth if s.label in CANONICAL_LABELS]
    scored = [s for s in gt_all if s.subject == "person"]
    neutral = [s for s in gt_all if s.subject != "person"]
    return scored, neutral


def score_document_redaction(predicted, ground_truth):
    """Character-level, label-agnostic redaction scoring.

    Redaction cares about which characters get covered, not which
    entities are found or what they are called: recall is the fraction
    of ground-truth PII characters covered by any prediction, precision
    the fraction of predicted characters that lie inside ground truth.
    Fragmented predictions earn proportional credit; over-wide
    predictions pay a proportional precision cost (a whole-document
    span scores precision equal to the document's PII density).

    Whitespace characters are excluded on both sides: a missed space
    between two redacted words leaks nothing, and a redacted space
    outside ground truth costs nothing.

    Characters of org/agency spans are neutral: they don't count
    toward recall, and predicted characters covering them are charged
    no FP.
    """
    gt_scored, neutral = _split_neutral(ground_truth)
    gt_chars = _scoreable_chars(gt_scored)
    pred_chars = _scoreable_chars(predicted) - _covered_chars(neutral)
    tp = len(gt_chars & pred_chars)
    fp = len(pred_chars - gt_chars)
    fn = len(gt_chars - pred_chars)
    p, r, f1 = _compute_prf(tp, fp, fn)
    return RedactionScore(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def score_corpus_redaction(doc_scores):
    tp = sum(d.tp for d in doc_scores)
    fp = sum(d.fp for d in doc_scores)
    fn = sum(d.fn for d in doc_scores)
    p, r, f1 = _compute_prf(tp, fp, fn)
    return RedactionScore(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def redaction_to_dict(score):
    """Serialize a corpus RedactionScore for results JSON.

    Micro only: matching is label-agnostic, so per-label precision has
    no meaning here — redaction reports a single leakage number.
    """
    return {
        "micro": {
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "tp": score.tp,
            "fp": score.fp,
            "fn": score.fn,
        },
    }


def score_document(predicted, ground_truth, mode="exact"):
    """Span-level scoring for one document.

    Org/agency GT spans are neutral: they are excluded before matching
    (never TP or FN), and an otherwise-unmatched prediction that would
    have matched one — same criterion as the mode — is discarded rather
    than charged as FP. Matching against person spans runs first, so
    neutral spans can only absorb false positives, never take a match
    away from a person span.
    """
    gt_scored, neutral = _split_neutral(ground_truth)
    tp, fp, fn, matches = match_spans(predicted, gt_scored, mode)

    label_counts = defaultdict(lambda: [0, 0, 0])
    matched_gt_indices = {gi for _, gi, _ in matches}
    matched_pred_indices = {pi for pi, _, _ in matches}
    neutralized = {
        i for i, pred in enumerate(predicted)
        if i not in matched_pred_indices
        and any(_is_match(pred, n, mode) for n in neutral)
    }
    fp -= len(neutralized)
    p, r, f1 = _compute_prf(tp, fp, fn)

    for _, gi, _ in matches:
        label_counts[gt_scored[gi].label][0] += 1
    for i, gt in enumerate(gt_scored):
        if i not in matched_gt_indices:
            label_counts[gt.label][2] += 1
    for i, pred in enumerate(predicted):
        if i not in matched_pred_indices and i not in neutralized:
            label_counts[pred.label][1] += 1

    return DocumentScore(
        tp=tp, fp=fp, fn=fn,
        precision=p, recall=r, f1=f1,
        label_counts=dict(label_counts),
        matches=matches,
    )


def score_corpus(doc_scores):
    totals = defaultdict(lambda: [0, 0, 0])
    for doc in doc_scores:
        for label, (ltp, lfp, lfn) in doc.label_counts.items():
            totals[label][0] += ltp
            totals[label][1] += lfp
            totals[label][2] += lfn

    total_tp = sum(v[0] for v in totals.values())
    total_fp = sum(v[1] for v in totals.values())
    total_fn = sum(v[2] for v in totals.values())
    micro_p, micro_r, micro_f1 = _compute_prf(total_tp, total_fp, total_fn)

    per_label = {}
    for label in sorted(totals):
        ltp, lfp, lfn = totals[label]
        lp = ltp / (ltp + lfp) if (ltp + lfp) > 0 else 0.0
        has_gt = (ltp + lfn) > 0
        lr = ltp / (ltp + lfn) if has_gt else None
        if lr is None:
            lf1 = None  # no GT for this label: F1 undefined, like recall
        elif (lp + lr) > 0:
            lf1 = 2 * lp * lr / (lp + lr)
        else:
            lf1 = 0.0
        per_label[label] = LabelScore(
            tp=ltp, fp=lfp, fn=lfn,
            precision=lp, recall=lr, f1=lf1,
        )

    return CorpusScore(
        total_tp=total_tp, total_fp=total_fp, total_fn=total_fn,
        micro_precision=micro_p, micro_recall=micro_r, micro_f1=micro_f1,
        per_label=per_label,
    )


def format_verbose(doc_id, text, ground_truth, predicted, matches):
    matched_gt = {gi for _, gi, _ in matches}
    matched_pred = {pi for pi, _, _ in matches}
    match_map = {gi: (pi, iou) for pi, gi, iou in matches}

    lines = [
        f"=== Document: {doc_id} ===",
        f"Text: {text[:200]}{'...' if len(text) > 200 else ''}",
        "",
        "Ground truth:",
    ]
    for i, gt in enumerate(ground_truth):
        status = "MATCHED" if i in matched_gt else "MISSED"
        extra = ""
        if i in match_map:
            pi, iou = match_map[i]
            extra = f" (IoU={iou:.2f}, pred #{pi})"
        lines.append(f"  [{status}] {gt.label} [{gt.start}:{gt.end}] \"{gt.text}\"{extra}")

    lines.append("")
    lines.append("Predictions:")
    for i, pred in enumerate(predicted):
        status = "MATCHED" if i in matched_pred else "SPURIOUS"
        lines.append(f"  [{status}] {pred.label} [{pred.start}:{pred.end}] \"{pred.text}\"")

    return "\n".join(lines)


def score_document_modes(predicted, ground_truth):
    """Score one document in every matching mode.

    Returns {"exact": DocumentScore, "overlap": DocumentScore,
    "redaction": RedactionScore} — the per-document unit the runners
    accumulate before corpus aggregation.
    """
    return {
        "exact": score_document(predicted, ground_truth, mode="exact"),
        "overlap": score_document(predicted, ground_truth, mode="overlap"),
        "redaction": score_document_redaction(predicted, ground_truth),
    }


def corpus_modes_to_dict(doc_modes):
    """Aggregate per-document mode scores (from score_document_modes)
    into the results-JSON fragment: {mode: {micro, per_label?}}."""
    return {
        "exact": corpus_to_dict(
            score_corpus([d["exact"] for d in doc_modes])),
        "overlap": corpus_to_dict(
            score_corpus([d["overlap"] for d in doc_modes])),
        "redaction": redaction_to_dict(
            score_corpus_redaction([d["redaction"] for d in doc_modes])),
    }


def corpus_to_dict(corpus):
    """Serialize a CorpusScore for results JSON."""
    return {
        "micro": {
            "precision": corpus.micro_precision,
            "recall": corpus.micro_recall,
            "f1": corpus.micro_f1,
            "tp": corpus.total_tp,
            "fp": corpus.total_fp,
            "fn": corpus.total_fn,
        },
        "per_label": {
            label: {
                "precision": ls.precision,
                "recall": ls.recall,
                "f1": ls.f1,
                "tp": ls.tp,
                "fp": ls.fp,
                "fn": ls.fn,
            }
            for label, ls in sorted(corpus.per_label.items())
        },
    }
