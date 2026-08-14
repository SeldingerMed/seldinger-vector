"""Rater agreement and detection metrics.

PLAN.md section 13 replaced the original "CVS AUROC >= 0.9" with a harder and
more honest pair of requirements, both implemented here:

* **Per-criterion sensitivity at a fixed, pre-chosen specificity.** AUROC over
  an imbalanced, subjectively-labelled endpoint is prevalence-sensitive and
  gameable by moving an operating point after the fact. Fixing the specificity
  floor *before* evaluation and reporting sensitivity at it removes that
  degree of freedom.
* **The raters' own agreement, reported alongside.** A model cannot be
  credited with agreement its labels do not contain. Fleiss' kappa is computed
  on the same cases and travels with the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from or_audit.errors import ScoreContractError

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FleissKappa:
    """Chance-corrected agreement among a fixed number of raters."""

    value: float
    subjects: int
    raters: int
    categories: int
    observed_agreement: float
    expected_agreement: float

    @property
    def interpretation(self) -> str:
        """Landis and Koch band.

        Reported as a label rather than a verdict: these bands are convention,
        not thresholds anyone validated for surgical assessment, and PLAN.md
        section 13 makes the panel's agreement context for the model's number
        rather than a gate in its own right.
        """
        thresholds = (
            (0.81, "almost perfect"),
            (0.61, "substantial"),
            (0.41, "moderate"),
            (0.21, "fair"),
            (0.00, "slight"),
        )
        for floor, label in thresholds:
            if self.value >= floor:
                return label
        return "poor (worse than chance)"


def fleiss_kappa(counts: npt.ArrayLike) -> FleissKappa:
    """Compute Fleiss' kappa from a subjects-by-categories count matrix.

    Args:
        counts: Shape ``(n_subjects, n_categories)``; entry ``[i, j]`` is how
            many raters assigned subject ``i`` to category ``j``. Every row
            must sum to the same number of raters.

    Returns:
        The coefficient and the agreement terms behind it.

    Raises:
        ScoreContractError: On a ragged design, fewer than two raters or
            subjects, or a degenerate distribution where kappa is undefined.
    """
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2:
        msg = f"counts must be a 2-D subjects-by-categories matrix, got {matrix.shape}"
        raise ScoreContractError(msg)
    n_subjects, n_categories = matrix.shape
    if n_subjects < 2 or n_categories < 2:
        msg = f"Fleiss' kappa needs at least 2 subjects and 2 categories, got {matrix.shape}"
        raise ScoreContractError(msg)
    if (matrix < 0).any():
        msg = "counts must be non-negative"
        raise ScoreContractError(msg)

    per_subject = matrix.sum(axis=1)
    n_raters = float(per_subject[0])
    if not np.allclose(per_subject, n_raters):
        msg = (
            "every subject must be rated by the same number of raters; Fleiss' "
            "kappa is undefined for a ragged design"
        )
        raise ScoreContractError(msg)
    if n_raters < 2:
        msg = f"Fleiss' kappa needs at least 2 raters per subject, got {n_raters:g}"
        raise ScoreContractError(msg)

    agreement_per_subject = ((matrix**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    observed = float(agreement_per_subject.mean())
    proportions = matrix.sum(axis=0) / (n_subjects * n_raters)
    expected = float((proportions**2).sum())

    if expected >= 1.0:
        msg = (
            "Fleiss' kappa is undefined: every rating fell in one category, so "
            "chance agreement is total and there is nothing to correct for"
        )
        raise ScoreContractError(msg)

    return FleissKappa(
        value=(observed - expected) / (1.0 - expected),
        subjects=n_subjects,
        raters=int(n_raters),
        categories=n_categories,
        observed_agreement=observed,
        expected_agreement=expected,
    )


@dataclass(frozen=True)
class OperatingPoint:
    """Detection performance at one decision threshold."""

    threshold: float
    sensitivity: float
    specificity: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    positives: int
    negatives: int


def sensitivity_at_specificity(
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    min_specificity: float,
) -> OperatingPoint:
    """Sensitivity at the most sensitive threshold meeting a specificity floor.

    The floor must be chosen before looking at the data. Choosing it afterwards
    is choosing the answer -- which is exactly the freedom AUROC leaves open and
    PLAN.md section 13 closes.

    Args:
        scores: Model scores, higher meaning more likely positive.
        labels: Ground truth, ``True`` for positive.
        min_specificity: Specificity floor, in ``(0, 1]``.

    Returns:
        The operating point. A solution always exists: the "classify nothing as
        positive" threshold attains specificity 1.0. An uninformative detector
        therefore reports sensitivity 0.0 at a high floor, which is the honest
        answer rather than an error -- it says the detector cannot find
        anything without exceeding the false-positive budget.

    Raises:
        ScoreContractError: If inputs are ragged, non-finite, empty, or either
            class is absent.
    """
    if not 0.0 < min_specificity <= 1.0:
        msg = f"min_specificity must be in (0, 1], got {min_specificity}"
        raise ScoreContractError(msg)
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=bool)
    if score_array.shape != label_array.shape:
        msg = f"scores and labels differ in length: {score_array.shape} vs {label_array.shape}"
        raise ScoreContractError(msg)
    if score_array.size == 0:
        msg = "cannot evaluate an empty cohort"
        raise ScoreContractError(msg)
    if not np.isfinite(score_array).all():
        msg = "scores contain non-finite values"
        raise ScoreContractError(msg)

    positives = int(label_array.sum())
    negatives = int((~label_array).sum())
    if positives == 0 or negatives == 0:
        msg = (
            f"a cohort with {positives} positive and {negatives} negative cases "
            f"cannot support a sensitivity/specificity estimate"
        )
        raise ScoreContractError(msg)

    # Candidate thresholds: every observed score, plus one above the maximum so
    # the "classify nothing as positive" point is reachable.
    candidates = np.unique(np.concatenate([score_array, [score_array.max() + 1.0]]))
    best: OperatingPoint | None = None
    for threshold in candidates:
        predicted = score_array >= threshold
        true_positive = int((predicted & label_array).sum())
        false_positive = int((predicted & ~label_array).sum())
        true_negative = negatives - false_positive
        false_negative = positives - true_positive
        specificity = true_negative / negatives
        if specificity < min_specificity:
            continue
        sensitivity = true_positive / positives
        if best is None or sensitivity > best.sensitivity:
            best = OperatingPoint(
                threshold=float(threshold),
                sensitivity=sensitivity,
                specificity=specificity,
                true_positives=true_positive,
                false_positives=false_positive,
                true_negatives=true_negative,
                false_negatives=false_negative,
                positives=positives,
                negatives=negatives,
            )

    # ``best`` is never None: the max+1 candidate predicts no positives, giving
    # specificity 1.0, which satisfies any floor in (0, 1].
    assert best is not None
    return best
