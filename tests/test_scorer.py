"""Tests for span matching and scoring (step 4a)."""

import pytest

from src.scorer import (
    Span,
    compute_iou,
    corpus_modes_to_dict,
    match_spans,
    score_corpus,
    score_corpus_redaction,
    score_document,
    score_document_modes,
    score_document_redaction,
)


def make_span(start, end, label, text=None, subject="person"):
    if text is None:
        text = "x" * (end - start)
    return Span(start=start, end=end, text=text, label=label, subject=subject)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_identical_spans(self):
        a = make_span(0, 10, "private_person")
        b = make_span(0, 10, "private_person")
        assert compute_iou(a, b) == 1.0

    def test_no_overlap(self):
        a = make_span(0, 5, "private_person")
        b = make_span(10, 15, "private_person")
        assert compute_iou(a, b) == 0.0

    def test_adjacent_is_zero(self):
        a = make_span(0, 5, "private_person")
        b = make_span(5, 10, "private_person")
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = make_span(0, 8, "private_person")
        b = make_span(2, 10, "private_person")
        # intersection [2,8]=6, union [0,10]=10
        assert compute_iou(a, b) == pytest.approx(0.6)

    def test_containment(self):
        a = make_span(0, 10, "private_person")
        b = make_span(2, 8, "private_person")
        # intersection [2,8]=6, union [0,10]=10
        assert compute_iou(a, b) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Span matching — basic
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical_boundaries_and_label(self):
        pred = [make_span(0, 4, "private_person", "John")]
        gt = [make_span(0, 4, "private_person", "John")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="exact")
        assert (tp, fp, fn) == (1, 0, 0)


class TestOverlapMatch:
    def test_iou_above_threshold(self):
        pred = [make_span(0, 8, "private_person")]
        gt = [make_span(2, 10, "private_person")]
        # intersection [2,8]=6, union [0,10]=10, IoU=0.6
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (1, 0, 0)

    def test_iou_below_threshold(self):
        pred = [make_span(0, 10, "private_person")]
        gt = [make_span(8, 20, "private_person")]
        # intersection [8,10]=2, union [0,20]=20, IoU=0.1
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (0, 1, 1)

    def test_containment_matches_even_with_low_iou(self):
        """GT fully inside pred → match, even if pairwise IoU < 0.5."""
        pred = [make_span(0, 20, "private_person")]
        gt = [make_span(5, 8, "private_person")]
        # IoU = 3/20 = 0.15, but GT is fully contained
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (1, 0, 0)


class TestAdjacentSpans:
    def test_adjacent_not_a_match_overlap(self):
        """[0,5) and [5,10) share a boundary but zero characters."""
        pred = [make_span(0, 5, "private_person")]
        gt = [make_span(5, 10, "private_person")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (0, 1, 1)

    def test_adjacent_not_a_match_exact(self):
        pred = [make_span(0, 5, "private_person")]
        gt = [make_span(5, 10, "private_person")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="exact")
        assert (tp, fp, fn) == (0, 1, 1)


class TestLabelMismatch:
    def test_same_offsets_different_label(self):
        pred = [make_span(0, 10, "private_person")]
        gt = [make_span(0, 10, "private_email")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="exact")
        assert (tp, fp, fn) == (0, 1, 1)

    def test_same_offsets_different_label_overlap(self):
        pred = [make_span(0, 10, "private_person")]
        gt = [make_span(0, 10, "private_email")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (0, 1, 1)


class TestRedactionCharLevel:
    """Redaction is character-level and label-agnostic: recall is the
    fraction of GT chars covered by any prediction, precision the
    fraction of predicted chars inside GT. TP/FP/FN are char counts.
    Whitespace chars are not scoreable on either side."""

    def test_wrong_label_still_counts(self):
        pred = [make_span(0, 10, "secret")]        # e.g. EIN tagged as secret
        gt = [make_span(0, 10, "account_number")]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp, s.fn) == (10, 0, 0)
        assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)

    def test_whole_document_span_pays_pii_density(self):
        """Redacting everything must not score perfect precision
        (regression: the span-level metric gave this P=R=F1=1.0)."""
        pred = [make_span(0, 100, "private_person")]
        gt = [make_span(10, 20, "private_person")]
        s = score_document_redaction(pred, gt)
        assert s.recall == 1.0
        assert s.precision == pytest.approx(0.1)

    def test_fragments_earn_proportional_recall(self):
        """'John' alone against GT 'John Smith' [0,10): 4 of 9
        scoreable chars covered (the span-level metric scored this
        0 TP + 1 FP)."""
        pred = [make_span(0, 4, "private_person", "John")]
        gt = [make_span(0, 10, "private_person", "John Smith")]
        s = score_document_redaction(pred, gt)
        assert s.recall == pytest.approx(4 / 9)
        assert s.precision == 1.0

    def test_missed_whitespace_is_not_a_leak(self):
        """'John' + 'Smith' against GT 'John Smith' [0,10): every
        letter is covered and the unredacted space leaks nothing, so
        recall is 1.0 (whitespace chars are not scoreable)."""
        pred = [
            make_span(0, 4, "private_person", "John"),
            make_span(5, 10, "private_person", "Smith"),
        ]
        gt = [make_span(0, 10, "private_person", "John Smith")]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp, s.fn) == (9, 0, 0)
        assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)

    def test_redacted_whitespace_is_not_charged(self):
        """One pred 'John Smith' against GT 'John' + 'Smith': the
        redacted space between them is outside GT but costs no FP."""
        pred = [make_span(0, 10, "private_person", "John Smith")]
        gt = [
            make_span(0, 4, "private_person", "John"),
            make_span(5, 10, "private_person", "Smith"),
        ]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp, s.fn) == (9, 0, 0)
        assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)

    def test_boundary_slack_costs_proportionally(self):
        pred = [make_span(0, 12, "private_person")]
        gt = [make_span(0, 10, "private_person")]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp, s.fn) == (10, 2, 0)

    def test_spurious_prediction_is_all_fp(self):
        pred = [make_span(50, 60, "private_url")]   # covers nothing
        gt = [make_span(0, 10, "private_email")]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp, s.fn) == (0, 10, 10)

    def test_overlapping_predictions_count_chars_once(self):
        pred = [
            make_span(0, 6, "private_person"),
            make_span(4, 10, "private_person"),
        ]
        gt = [make_span(0, 10, "private_person")]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp) == (10, 0)

    def test_non_canonical_gt_excluded(self):
        gt = [make_span(0, 4, "private_person"), make_span(10, 12, "age")]
        pred = [make_span(0, 4, "private_person")]
        s = score_document_redaction(pred, gt)
        assert (s.tp, s.fp, s.fn) == (4, 0, 0)

    def test_empty_document(self):
        s = score_document_redaction([], [])
        assert (s.tp, s.fp, s.fn) == (0, 0, 0)
        assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)

    def test_corpus_micro_sums_char_counts(self):
        d1 = score_document_redaction(
            [make_span(0, 10, "secret")], [make_span(0, 10, "secret")])
        d2 = score_document_redaction([], [make_span(0, 10, "secret")])
        corpus = score_corpus_redaction([d1, d2])
        assert (corpus.tp, corpus.fn) == (10, 10)
        assert corpus.recall == pytest.approx(0.5)
        assert corpus.precision == pytest.approx(1.0)

    def test_redaction_is_not_a_span_matching_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            match_spans([make_span(0, 1, "secret")],
                        [make_span(0, 1, "secret")], mode="redaction")


# ---------------------------------------------------------------------------
# Subject tags — org/agency spans are neutral
# ---------------------------------------------------------------------------


class TestSubjectNeutrality:
    """GT spans with subject org/agency are neutral: excluded from
    recall, and predictions covering them are absorbed (neither TP
    nor FP)."""

    def test_missed_org_span_is_not_a_miss(self):
        gt = [
            make_span(0, 4, "private_person", "Jane"),
            make_span(10, 24, "private_phone", subject="org"),
        ]
        pred = [make_span(0, 4, "private_person", "Jane")]
        result = score_document(pred, gt, mode="overlap")
        assert (result.tp, result.fp, result.fn) == (1, 0, 0)
        assert result.recall == 1.0

    def test_detected_org_span_is_neither_tp_nor_fp(self):
        gt = [make_span(10, 24, "private_phone", subject="org")]
        pred = [make_span(10, 24, "private_phone")]
        result = score_document(pred, gt, mode="overlap")
        assert (result.tp, result.fp, result.fn) == (0, 0, 0)

    def test_agency_subject_is_also_neutral(self):
        gt = [make_span(0, 15, "private_url", subject="agency")]
        pred = [make_span(0, 15, "private_url")]
        result = score_document(pred, gt, mode="overlap")
        assert (result.tp, result.fp, result.fn) == (0, 0, 0)

    def test_unrelated_false_positive_still_charged(self):
        gt = [make_span(10, 24, "private_phone", subject="org")]
        pred = [make_span(50, 60, "private_phone")]  # matches nothing
        result = score_document(pred, gt, mode="overlap")
        assert (result.tp, result.fp, result.fn) == (0, 1, 0)

    def test_neutralization_uses_the_mode_criterion(self):
        """Boundary slack over an org span: absorbed under overlap
        (containment) but still an FP under exact."""
        gt = [make_span(10, 24, "private_phone", subject="org")]
        pred = [make_span(8, 26, "private_phone")]
        overlap = score_document(pred, gt, mode="overlap")
        assert (overlap.tp, overlap.fp) == (0, 0)
        exact = score_document(pred, gt, mode="exact")
        assert (exact.tp, exact.fp) == (0, 1)

    def test_person_spans_matched_before_neutral(self):
        """A prediction that overlaps a person span keeps its TP even
        when it also covers a neutral span — neutral regions only ever
        absorb false positives."""
        gt = [
            make_span(0, 10, "private_address", "12 Oak Ave..."),
            make_span(12, 20, "private_address", subject="org"),
        ]
        pred = [make_span(0, 20, "private_address")]
        result = score_document(pred, gt, mode="overlap")
        assert (result.tp, result.fp, result.fn) == (1, 0, 0)

    def test_neutralized_prediction_leaves_per_label_counts(self):
        gt = [make_span(10, 24, "private_phone", subject="org")]
        pred = [make_span(10, 24, "private_phone")]
        result = score_document(pred, gt, mode="overlap")
        assert result.label_counts == {}

    def test_redaction_neutral_chars_are_free(self):
        """Predicted characters inside org/agency spans cost nothing;
        characters beyond them still pay the precision cost."""
        gt = [
            make_span(0, 10, "private_person"),
            make_span(20, 30, "private_phone", subject="org"),
        ]
        pred = [make_span(0, 10, "private_person"),
                make_span(20, 35, "private_phone")]
        result = score_document_redaction(pred, gt)
        assert (result.tp, result.fp, result.fn) == (10, 5, 0)

    def test_redaction_missed_org_chars_not_counted(self):
        gt = [make_span(20, 30, "private_phone", subject="org")]
        result = score_document_redaction([], gt)
        assert (result.tp, result.fp, result.fn) == (0, 0, 0)


class TestScoreDocumentModes:
    def test_shape_and_consistency(self):
        gt = [
            make_span(0, 4, "private_person", "Jane"),
            make_span(10, 24, "private_phone", subject="org"),
        ]
        pred = [make_span(0, 4, "private_person", "Jane"),
                make_span(10, 24, "private_phone")]
        modes = score_document_modes(pred, gt)
        assert set(modes) == {"exact", "overlap", "redaction"}
        assert modes["overlap"].tp == 1
        assert modes["overlap"].fp == 0

    def test_corpus_modes_to_dict_shape(self):
        gt = [make_span(0, 4, "private_person", "Jane")]
        pred = [make_span(0, 4, "private_person", "Jane")]
        out = corpus_modes_to_dict([score_document_modes(pred, gt)])
        assert set(out) == {"exact", "overlap", "redaction"}
        assert set(out["exact"]) == {"micro", "per_label"}
        assert set(out["redaction"]) == {"micro"}


class TestDuplicatePredictions:
    def test_two_preds_one_gt(self):
        """Two identical predictions for one GT span → 1 TP + 1 FP."""
        pred = [
            make_span(10, 25, "private_email", "john@test.com"),
            make_span(10, 25, "private_email", "john@test.com"),
        ]
        gt = [make_span(10, 25, "private_email", "john@test.com")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="exact")
        assert (tp, fp, fn) == (1, 1, 0)


class TestDuplicateGroundTruth:
    def test_one_pred_two_gt(self):
        """One prediction for two identical GT spans → 1 TP + 1 FN."""
        pred = [make_span(10, 25, "private_email", "john@test.com")]
        gt = [
            make_span(10, 25, "private_email", "john@test.com"),
            make_span(10, 25, "private_email", "john@test.com"),
        ]
        tp, fp, fn, _ = match_spans(pred, gt, mode="exact")
        assert (tp, fp, fn) == (1, 0, 1)


# ---------------------------------------------------------------------------
# Many-to-many matching
# ---------------------------------------------------------------------------


class TestManyToMany:
    def test_one_pred_covers_multiple_gt(self):
        """'John Smith' [0,10] matches both GT 'John' [0,4] and 'Smith' [5,10].

        Pairwise IoU of [0,10] vs [0,4] is 0.4 (below 0.5), but GT [0,4]
        is fully contained within the prediction, so it matches via the
        containment criterion.
        """
        pred = [make_span(0, 10, "private_person", "John Smith")]
        gt = [
            make_span(0, 4, "private_person", "John"),
            make_span(5, 10, "private_person", "Smith"),
        ]
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (2, 0, 0)

    def test_partial_pred_matches_only_covered_gt(self):
        """'John' [0,4] matches GT 'John' but GT 'Smith' is a FN."""
        pred = [make_span(0, 4, "private_person", "John")]
        gt = [
            make_span(0, 4, "private_person", "John"),
            make_span(5, 10, "private_person", "Smith"),
        ]
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (1, 0, 1)

    def test_one_pred_cannot_match_different_labels(self):
        """Pred [0,10] covers both GT spans, but label mismatch blocks one."""
        pred = [make_span(0, 10, "private_person")]
        gt = [
            make_span(0, 4, "private_person"),
            make_span(5, 10, "private_email"),
        ]
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (1, 0, 1)


# ---------------------------------------------------------------------------
# Metrics — known-answer
# ---------------------------------------------------------------------------


class TestKnownAnswerMetrics:
    """Hand-computed 5-span example.

    GT: 5 spans across 5 labels.
    Pred: 3 exact matches + 2 spurious (wrong location/label).
    TP=3, FP=2, FN=2 → P=0.6, R=0.6, F1=0.6.
    """

    @pytest.fixture()
    def scenario(self):
        gt = [
            make_span(0, 4, "private_person", "John"),
            make_span(10, 23, "private_email", "john@test.com"),
            make_span(30, 38, "private_phone", "555-1234"),
            make_span(45, 56, "private_address", "123 Main St"),
            make_span(60, 69, "secret", "secret123"),
        ]
        pred = [
            make_span(0, 4, "private_person", "John"),
            make_span(10, 23, "private_email", "john@test.com"),
            make_span(30, 38, "private_phone", "555-1234"),
            make_span(80, 90, "private_person", "unexpected"),
            make_span(95, 105, "private_url", "example.com"),
        ]
        return pred, gt

    def test_precision(self, scenario):
        pred, gt = scenario
        result = score_document(pred, gt, mode="exact")
        assert result.precision == pytest.approx(0.6)

    def test_recall(self, scenario):
        pred, gt = scenario
        result = score_document(pred, gt, mode="exact")
        assert result.recall == pytest.approx(0.6)

    def test_f1(self, scenario):
        pred, gt = scenario
        result = score_document(pred, gt, mode="exact")
        assert result.f1 == pytest.approx(0.6)


class TestPerfectPredictions:
    def test_perfect(self):
        spans = [
            make_span(0, 4, "private_person", "John"),
            make_span(10, 23, "private_email", "john@test.com"),
        ]
        result = score_document(list(spans), list(spans), mode="exact")
        assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)


class TestNoPredictions:
    def test_no_preds(self):
        gt = [make_span(0, 4, "private_person", "John")]
        result = score_document([], gt, mode="exact")
        assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


class TestAllWrongPredictions:
    def test_all_wrong(self):
        pred = [make_span(0, 4, "private_person", "John")]
        gt = [make_span(50, 63, "private_email", "john@test.com")]
        result = score_document(pred, gt, mode="exact")
        assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


class TestPerLabelMicroAveraging:
    """Per-label TP/FP/FN must sum to micro-averaged totals.

    private_person: TP=1, FP=0, FN=1 → P=1.0,  R=0.5
    private_email:  TP=2, FP=1, FN=0 → P=2/3,  R=1.0
    private_phone:  TP=1, FP=0, FN=0 → P=1.0,  R=1.0
    Micro:          TP=4, FP=1, FN=1 → P=0.8,  R=0.8, F1=0.8
    """

    @pytest.fixture()
    def corpus(self):
        gt = [
            make_span(0, 4, "private_person", "John"),
            make_span(5, 10, "private_person", "Smith"),
            make_span(20, 33, "private_email", "john@test.com"),
            make_span(40, 53, "private_email", "jane@test.com"),
            make_span(60, 68, "private_phone", "555-1234"),
        ]
        pred = [
            make_span(0, 4, "private_person", "John"),
            make_span(20, 33, "private_email", "john@test.com"),
            make_span(40, 53, "private_email", "jane@test.com"),
            make_span(70, 83, "private_email", "extra@test.com"),
            make_span(60, 68, "private_phone", "555-1234"),
        ]
        return score_corpus([score_document(pred, gt, mode="exact")])

    def test_micro_precision(self, corpus):
        assert corpus.micro_precision == pytest.approx(0.8)

    def test_micro_recall(self, corpus):
        assert corpus.micro_recall == pytest.approx(0.8)

    def test_micro_f1(self, corpus):
        assert corpus.micro_f1 == pytest.approx(0.8)

    def test_per_label_person(self, corpus):
        lbl = corpus.per_label["private_person"]
        assert lbl.precision == pytest.approx(1.0)
        assert lbl.recall == pytest.approx(0.5)

    def test_per_label_email(self, corpus):
        lbl = corpus.per_label["private_email"]
        assert lbl.precision == pytest.approx(2 / 3)
        assert lbl.recall == pytest.approx(1.0)

    def test_per_label_phone(self, corpus):
        lbl = corpus.per_label["private_phone"]
        assert lbl.precision == pytest.approx(1.0)
        assert lbl.recall == pytest.approx(1.0)

    def test_per_label_counts_sum_to_micro(self, corpus):
        total_tp = sum(l.tp for l in corpus.per_label.values())
        total_fp = sum(l.fp for l in corpus.per_label.values())
        total_fn = sum(l.fn for l in corpus.per_label.values())
        assert total_tp == corpus.total_tp
        assert total_fp == corpus.total_fp
        assert total_fn == corpus.total_fn


class TestZeroGroundTruthLabel:
    def test_label_excluded_from_recall(self):
        """Spurious predictions for a label with no GT don't hurt recall."""
        gt = [make_span(0, 4, "private_person", "John")]
        pred = [
            make_span(0, 4, "private_person", "John"),
            make_span(10, 23, "private_email", "john@test.com"),
        ]
        result = score_document(pred, gt, mode="exact")
        assert result.recall == pytest.approx(1.0)

        corpus = score_corpus([result])
        assert corpus.micro_recall == pytest.approx(1.0)
        assert corpus.per_label["private_email"].recall is None
        assert corpus.per_label["private_email"].f1 is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEmptyDocument:
    def test_no_spans(self):
        result = score_document([], [], mode="exact")
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 0

    def test_no_contribution_to_corpus(self):
        """An empty doc shouldn't dilute corpus metrics."""
        gt = [make_span(0, 4, "private_person", "John")]
        real = score_document([make_span(0, 4, "private_person", "John")], gt, mode="exact")
        empty = score_document([], [], mode="exact")
        corpus = score_corpus([real, empty])
        assert corpus.micro_precision == pytest.approx(1.0)
        assert corpus.micro_recall == pytest.approx(1.0)
        assert corpus.micro_f1 == pytest.approx(1.0)


class TestUnmappedGroundTruth:
    def test_unmapped_excluded(self):
        """GT spans with non-canonical labels are excluded before scoring."""
        gt = [
            make_span(0, 4, "private_person", "John"),
            make_span(10, 12, "age", "25"),
        ]
        pred = [make_span(0, 4, "private_person", "John")]
        result = score_document(pred, gt, mode="exact")
        assert (result.tp, result.fp, result.fn) == (1, 0, 0)


class TestOverlappingGroundTruth:
    def test_no_double_count(self):
        """Overlapping GT spans (e.g. both first_name and last_name map to
        private_person) are each matched at most once."""
        gt = [
            make_span(0, 10, "private_person", "John Smith"),
            make_span(5, 10, "private_person", "Smith"),
        ]
        pred = [make_span(0, 10, "private_person", "John Smith")]
        tp, fp, fn, _ = match_spans(pred, gt, mode="overlap")
        assert (tp, fp, fn) == (2, 0, 0)
