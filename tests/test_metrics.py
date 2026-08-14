"""Metrics for the PLAN.md section 13 gate.

Both estimators are checked against published worked examples rather than
against themselves. A self-consistent statistics module is worthless: the whole
point is that an outside reader can reproduce the number.
"""

from __future__ import annotations

import numpy as np
import pytest

from or_audit.errors import ScoreContractError
from or_audit.metrics.agreement import fleiss_kappa, sensitivity_at_specificity
from or_audit.metrics.icc import icc_2_1, icc_average_measures

#: Shrout & Fleiss (1979), the standard worked example for the ICC forms.
#: Published: ICC(1,1)=0.17, ICC(2,1)=0.29, ICC(3,1)=0.71.
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)

#: Fleiss (1971) worked example: 10 subjects, 14 raters, 5 categories.
#: Published kappa = 0.210.
FLEISS_1971 = np.array(
    [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ],
    dtype=float,
)


class TestIccAgainstPublishedValues:
    def test_matches_shrout_and_fleiss(self):
        """The reference value for ICC(2,1) on the canonical dataset."""
        assert icc_2_1(SHROUT_FLEISS).value == pytest.approx(0.290, abs=0.001)

    def test_form_is_stated_not_implied(self):
        """An unqualified ICC is a family of numbers, not a number."""
        assert "ICC(2,1)" in icc_2_1(SHROUT_FLEISS).form
        assert "absolute agreement" in icc_2_1(SHROUT_FLEISS).form
        assert "single rater" in icc_2_1(SHROUT_FLEISS).form

    def test_variance_components_are_exposed(self):
        """A bare coefficient is not reviewable."""
        estimate = icc_2_1(SHROUT_FLEISS)
        assert estimate.ms_between_subjects == pytest.approx(11.2417, abs=1e-3)
        assert estimate.ms_between_raters == pytest.approx(32.4861, abs=1e-3)
        assert estimate.ms_error == pytest.approx(1.0194, abs=1e-3)

    def test_perfect_agreement_is_one(self):
        data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        assert icc_2_1(data).value == pytest.approx(1.0)

    def test_systematic_rater_bias_is_penalised(self):
        """Absolute agreement, not consistency: a rater reliably two points
        high is not interchangeable with the panel, and consistency-form ICC
        would score this near 1.0."""
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        biased = np.column_stack([base, base + 2.0])
        assert icc_2_1(biased).value < 0.75

    def test_no_between_subject_variance_gives_no_agreement(self):
        data = np.array([[3.0, 1.0], [3.0, 5.0], [3.0, 2.0], [3.0, 4.0]])
        assert icc_2_1(data).value <= 0.0


class TestIccInputDiscipline:
    def test_averaging_is_refused_with_an_explanation(self):
        """Section 13 prohibits average-measures for the headline figure."""
        with pytest.raises(ScoreContractError, match="prohibited"):
            icc_average_measures(SHROUT_FLEISS)

    def test_one_rater_is_refused(self):
        with pytest.raises(ScoreContractError, match="at least"):
            icc_2_1(np.array([[1.0], [2.0], [3.0]]))

    def test_one_subject_is_refused(self):
        with pytest.raises(ScoreContractError, match="at least"):
            icc_2_1(np.array([[1.0, 2.0]]))

    def test_missing_cells_are_refused_not_imputed(self):
        data = SHROUT_FLEISS.copy()
        data[0, 0] = np.nan
        with pytest.raises(ScoreContractError, match="complete design"):
            icc_2_1(data)

    def test_one_dimensional_input_is_refused(self):
        with pytest.raises(ScoreContractError, match="2-D"):
            icc_2_1(np.array([1.0, 2.0, 3.0]))

    def test_constant_ratings_are_refused(self):
        """Undefined, not 1.0: nothing was distinguished."""
        with pytest.raises(ScoreContractError, match="no variance"):
            icc_2_1(np.full((4, 3), 3.0))


class TestFleissKappaAgainstPublishedValues:
    def test_matches_fleiss_1971(self):
        assert fleiss_kappa(FLEISS_1971).value == pytest.approx(0.210, abs=0.001)

    def test_agreement_terms_are_exposed(self):
        result = fleiss_kappa(FLEISS_1971)
        assert result.subjects == 10
        assert result.raters == 14
        assert result.categories == 5
        assert result.observed_agreement > result.expected_agreement

    def test_perfect_agreement_is_one(self):
        assert fleiss_kappa(np.array([[3, 0], [0, 3], [3, 0]])).value == pytest.approx(1.0)

    def test_chance_level_agreement_is_near_zero(self):
        counts = np.array([[2, 2], [2, 2], [2, 2], [2, 2]])
        assert fleiss_kappa(counts).value == pytest.approx(-1 / 3, abs=0.01)

    def test_interpretation_bands(self):
        assert fleiss_kappa(np.array([[3, 0], [0, 3]])).interpretation == "almost perfect"
        # 0.2099 sits just under the 0.21 boundary, so "slight" is correct and
        # the band edge is exercised rather than assumed.
        assert fleiss_kappa(FLEISS_1971).interpretation == "slight"


class TestFleissKappaInputDiscipline:
    def test_ragged_design_is_refused(self):
        with pytest.raises(ScoreContractError, match="same number of raters"):
            fleiss_kappa(np.array([[3, 0], [2, 0]]))

    def test_single_rater_is_refused(self):
        with pytest.raises(ScoreContractError, match="at least 2 raters"):
            fleiss_kappa(np.array([[1, 0], [0, 1]]))

    def test_all_one_category_is_refused(self):
        """Chance agreement is total, so there is nothing to correct for."""
        with pytest.raises(ScoreContractError, match="one category"):
            fleiss_kappa(np.array([[3, 0], [3, 0], [3, 0]]))

    def test_negative_counts_are_refused(self):
        with pytest.raises(ScoreContractError, match="non-negative"):
            fleiss_kappa(np.array([[-1, 4], [3, 0]]))


class TestSensitivityAtFixedSpecificity:
    """Section 13 replaced AUROC with this, because AUROC lets the operating
    point be chosen after seeing the answer."""

    def test_perfectly_separable_scores(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [False, False, True, True]
        point = sensitivity_at_specificity(scores, labels, min_specificity=0.99)
        assert point.sensitivity == pytest.approx(1.0)
        assert point.specificity == pytest.approx(1.0)

    def test_picks_the_most_sensitive_threshold_meeting_the_floor(self):
        scores = [0.1, 0.5, 0.6, 0.9]
        labels = [False, True, False, True]
        relaxed = sensitivity_at_specificity(scores, labels, min_specificity=0.5)
        strict = sensitivity_at_specificity(scores, labels, min_specificity=1.0)
        assert relaxed.sensitivity >= strict.sensitivity
        assert strict.specificity == pytest.approx(1.0)

    def test_confusion_counts_are_consistent(self):
        scores = [0.1, 0.4, 0.6, 0.9, 0.95]
        labels = [False, False, True, True, False]
        point = sensitivity_at_specificity(scores, labels, min_specificity=0.5)
        assert point.true_positives + point.false_negatives == point.positives
        assert point.true_negatives + point.false_positives == point.negatives

    def test_uninformative_detector_reports_zero_sensitivity_not_an_error(self):
        """A solution always exists: predicting no positives gives specificity 1.

        Sensitivity 0.0 is the honest answer for a detector that cannot find
        anything within the false-positive budget. Raising instead would hide
        a real, reportable result behind an exception.
        """
        scores = [0.9, 0.9, 0.9, 0.9]
        labels = [True, False, True, False]
        point = sensitivity_at_specificity(scores, labels, min_specificity=0.99)
        assert point.sensitivity == pytest.approx(0.0)
        assert point.specificity == pytest.approx(1.0)

    def test_single_class_cohort_is_refused(self):
        with pytest.raises(ScoreContractError, match="cannot support"):
            sensitivity_at_specificity([0.1, 0.9], [True, True], min_specificity=0.5)

    def test_empty_cohort_is_refused(self):
        with pytest.raises(ScoreContractError, match="empty cohort"):
            sensitivity_at_specificity([], [], min_specificity=0.5)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ScoreContractError, match="differ in length"):
            sensitivity_at_specificity([0.1, 0.2], [True], min_specificity=0.5)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_specificity_floor_is_refused(self, bad):
        with pytest.raises(ScoreContractError, match="min_specificity"):
            sensitivity_at_specificity([0.1, 0.9], [False, True], min_specificity=bad)

    def test_non_finite_scores_are_refused(self):
        with pytest.raises(ScoreContractError, match="non-finite"):
            sensitivity_at_specificity([0.1, float("nan")], [False, True], min_specificity=0.5)
