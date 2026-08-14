"""Intraclass correlation, specified exactly.

PLAN.md section 13 is unusually prescriptive here, and for good reason: the
original plan said "ICC >= 0.8" without naming a form, and ICC(3,k) with
averaged raters can report far higher than ICC(2,1) on identical data. An
unqualified ICC is not a number, it is a family of numbers.

What this module implements and why:

* **ICC(2,1), two-way random effects, absolute agreement, single rater.** Two-
  way random because both cases and raters are samples from populations we
  want to generalise over. Absolute agreement rather than consistency because
  a scorer that is reliably two points high is not interchangeable with the
  panel, and consistency-form ICC would forgive that. Single rater because the
  deployed system is one rater, not an average of several.
* **Averaging is prohibited, not merely discouraged.** :func:`icc_2_1` is the
  only public estimator. There is no ``average=True`` flag to reach for.

Validated against Shrout & Fleiss (1979), whose worked example is the standard
reference for these forms; see the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from or_audit.errors import ScoreContractError

#: Ratings laid out as (subjects, raters). Every cell must be present:
#: unbalanced designs need a different estimator and are refused rather than
#: silently imputed.
RatingMatrix = npt.NDArray[np.float64]

MIN_SUBJECTS: Final = 2
MIN_RATERS: Final = 2


@dataclass(frozen=True)
class IccEstimate:
    """An ICC(2,1) estimate and the quantities behind it.

    The components are exposed because a bare coefficient is not reviewable.
    A reader checking whether agreement is real needs to see how much variance
    was between cases versus between raters.
    """

    value: float
    subjects: int
    raters: int
    #: Mean square between subjects.
    ms_between_subjects: float
    #: Mean square between raters. Large values mean systematic rater bias,
    #: which absolute-agreement ICC correctly penalises.
    ms_between_raters: float
    #: Residual mean square.
    ms_error: float

    @property
    def form(self) -> str:
        """The ICC form, spelled out so a report cannot be ambiguous."""
        return "ICC(2,1) two-way random effects, absolute agreement, single rater"


def icc_2_1(ratings: RatingMatrix) -> IccEstimate:
    """Compute ICC(2,1) for a complete subjects-by-raters matrix.

    Args:
        ratings: Shape ``(n_subjects, n_raters)``. No missing cells.

    Returns:
        The estimate and its variance components.

    Raises:
        ScoreContractError: If the design is too small, unbalanced, contains
            non-finite values, or has no variance to explain.
    """
    matrix = np.asarray(ratings, dtype=np.float64)
    if matrix.ndim != 2:
        msg = f"ratings must be a 2-D subjects-by-raters matrix, got shape {matrix.shape}"
        raise ScoreContractError(msg)
    n_subjects, n_raters = matrix.shape
    if n_subjects < MIN_SUBJECTS or n_raters < MIN_RATERS:
        msg = (
            f"ICC needs at least {MIN_SUBJECTS} subjects and {MIN_RATERS} raters, "
            f"got {n_subjects}x{n_raters}"
        )
        raise ScoreContractError(msg)
    if not np.isfinite(matrix).all():
        msg = "ratings contain missing or non-finite values; ICC(2,1) requires a complete design"
        raise ScoreContractError(msg)

    grand_mean = matrix.mean()
    subject_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)

    ss_subjects = n_raters * float(((subject_means - grand_mean) ** 2).sum())
    ss_raters = n_subjects * float(((rater_means - grand_mean) ** 2).sum())
    ss_total = float(((matrix - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_subjects - ss_raters

    df_subjects = n_subjects - 1
    df_raters = n_raters - 1
    df_error = df_subjects * df_raters

    ms_subjects = ss_subjects / df_subjects
    ms_raters = ss_raters / df_raters
    ms_error = ss_error / df_error

    denominator = (
        ms_subjects + (n_raters - 1) * ms_error + n_raters * (ms_raters - ms_error) / n_subjects
    )
    if denominator == 0:
        msg = (
            "ICC is undefined: the ratings have no variance to partition, which "
            "usually means every case received an identical score"
        )
        raise ScoreContractError(msg)

    return IccEstimate(
        value=(ms_subjects - ms_error) / denominator,
        subjects=n_subjects,
        raters=n_raters,
        ms_between_subjects=ms_subjects,
        ms_between_raters=ms_raters,
        ms_error=ms_error,
    )


def icc_average_measures(_ratings: RatingMatrix) -> float:
    """Deliberately not implemented.

    Averaging raters inflates ICC -- that is what the Spearman-Brown
    relationship does -- and PLAN.md section 13 prohibits it for the headline
    figure because the deployed scorer is a single rater, not a committee. The
    function exists so that reaching for it produces an explanation rather than
    a plausible number.

    Raises:
        ScoreContractError: Always.
    """
    msg = (
        "average-measures ICC is prohibited for the headline agreement figure: "
        "averaging raters inflates the coefficient and the deployed scorer is a "
        "single rater, not an average of several (PLAN.md section 13). Use "
        "icc_2_1."
    )
    raise ScoreContractError(msg)
