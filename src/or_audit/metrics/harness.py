"""The Phase 1 agreement gate from PLAN.md section 13.

The plan's v1 gate was "ICC >= 0.8". Three things were wrong with it, and this
module is shaped by fixing all three.

**The form was unspecified.** Handled in :mod:`or_audit.metrics.icc`.

**The target was absolute.** The human panel is the ceiling. An absolute 0.8
either demands the scorer be more self-consistent than the experts it is
imitating, or -- when panel agreement is poor -- accepts a weak model because
the bar happened to sit below what noise alone achieves. The target here is
relative: automated-versus-expert ICC must reach a stated fraction of
expert-versus-expert ICC measured on the same cases, and the panel figure is
reported next to it always.

**The cohort was unstratified.** Novice-versus-expert separation is inflated by
between-group variance and is close to trivial. Credentialing needs
within-band discrimination, so the headline is computed on a single band and a
mixed-band figure may only be reported as secondary, labelled as such.

The gate also enforces the plan's endpoint ordering: binary proficiency is
primary, GEARS secondary. A GEARS-only submission does not constitute a
result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from or_audit.domain.enums import SkillBand
from or_audit.errors import ScoreContractError
from or_audit.metrics.icc import IccEstimate, icc_2_1

#: Fraction of the panel's own agreement the scorer must reach.
DEFAULT_RELATIVE_TARGET = 0.90

#: Panel agreement below which the panel is not a usable reference at all.
#:
#: A purely relative target has a failure mode PLAN.md section 13 actually
#: names when arguing against an absolute one: it "silently accepts a weak
#: model when panel agreement is poor". Relative scaling alone makes that
#: worse, not better -- degrade the panel far enough and the bar approaches
#: zero, so a scorer uncorrelated with truth clears it. Review demonstrated
#: exactly this by simulation.
#:
#: The resolution is that panel adequacy is a *precondition*, not a scaling
#: factor. A panel that cannot agree with itself is not a low ceiling; it is
#: not a ceiling. Below this value the figure is invalid and no scorer can
#: pass on it, however it performs.
DEFAULT_MIN_PANEL_ICC = 0.40

#: Floor the requirement never drops below, whatever the panel does.
#:
#: Defence in depth behind the adequacy check. Deliberately modest: section 13
#: rejects an absolute target precisely because a high one "demands superhuman
#: consistency", so this sits low enough never to bind on a healthy panel and
#: high enough that noise cannot clear it.
DEFAULT_ABSOLUTE_FLOOR = 0.50


class Endpoint(StrEnum):
    """Which endpoint an agreement figure describes."""

    #: Binary proficiency items. Primary per section 13: a randomised trial
    #: found binary metrics outperformed GEARS on reliability and
    #: discrimination, so building the headline gate on GEARS would import its
    #: noise.
    BINARY_PROFICIENCY = "binary_proficiency"
    #: GEARS domain scores. Secondary, retained for interoperability with
    #: programmes that already use the instrument.
    GEARS = "gears"


@dataclass(frozen=True)
class AgreementFigure:
    """Automated-versus-panel agreement, with the panel's own agreement."""

    endpoint: Endpoint
    band: SkillBand | None
    automated_vs_expert: IccEstimate
    expert_vs_expert: IccEstimate
    relative_target: float
    n_cases: int
    absolute_floor: float = DEFAULT_ABSOLUTE_FLOOR
    min_panel_icc: float = DEFAULT_MIN_PANEL_ICC

    @property
    def panel_is_adequate(self) -> bool:
        """Whether the panel agrees with itself well enough to be a reference."""
        return self.expert_vs_expert.value >= self.min_panel_icc

    @property
    def required(self) -> float:
        """The value the scorer must reach on these cases.

        Never below :attr:`absolute_floor`, so a degraded panel cannot lower
        the bar toward zero. Meaningful only when :attr:`panel_is_adequate`;
        an inadequate panel makes the whole figure invalid rather than easy.
        """
        return max(self.absolute_floor, self.relative_target * self.expert_vs_expert.value)

    @property
    def achieved(self) -> float:
        """What the scorer reached."""
        return self.automated_vs_expert.value

    @property
    def passes(self) -> bool:
        """Whether the scorer met the requirement on a usable panel."""
        return self.panel_is_adequate and self.achieved >= self.required

    @property
    def is_stratified(self) -> bool:
        """Whether this figure was computed within a single experience band."""
        return self.band is not None

    def describe(self) -> str:
        """One line suitable for a report, never a bare coefficient."""
        cohort = f"{self.band.value} only" if self.band else "MIXED BAND (secondary)"
        panel = "" if self.panel_is_adequate else " PANEL INADEQUATE"
        return (
            f"{self.endpoint.value}: ICC(2,1)={self.achieved:.3f} vs panel "
            f"{self.expert_vs_expert.value:.3f} "
            f"(required {self.required:.3f} = max({self.absolute_floor:g}, "
            f"{self.relative_target:g}x panel)), "
            f"n={self.n_cases}, {cohort}{panel}"
        )


class AgreementGateResult(BaseModel):
    """The Phase 1 verdict and everything it rests on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    reason: str
    #: Primary endpoint line, always present.
    primary: str
    #: Secondary lines, clearly marked. Never sufficient on their own.
    secondary: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Always raises, so the verdict is read by name.

        ``if gate:`` hides which endpoint passed and on which cohort, and this
        result is exactly the thing people quote out of context.
        """
        msg = (
            "read AgreementGateResult.passed explicitly; the verdict depends on "
            "endpoint and cohort and must not be collapsed to a truth value"
        )
        raise ScoreContractError(msg)


def agreement_figure(
    *,
    endpoint: Endpoint,
    automated: npt.ArrayLike,
    expert_panel: Sequence[npt.ArrayLike],
    band: SkillBand | None,
    relative_target: float = DEFAULT_RELATIVE_TARGET,
    absolute_floor: float = DEFAULT_ABSOLUTE_FLOOR,
    min_panel_icc: float = DEFAULT_MIN_PANEL_ICC,
) -> AgreementFigure:
    """Compute automated-versus-panel and panel-internal agreement together.

    Both figures come from the same cases, which is the only way the relative
    target means anything.

    Args:
        endpoint: Which endpoint these scores describe.
        automated: One score per case from the scorer under test.
        expert_panel: One sequence per expert rater, each one score per case.
            At least two raters, so the panel's own agreement is estimable.
        band: Experience band, or ``None`` for a mixed cohort. A mixed cohort
            can only ever be a secondary figure.
        relative_target: Fraction of panel agreement the scorer must reach.

    Returns:
        The paired figures.

    Raises:
        ScoreContractError: On ragged input, fewer than two raters, or a
            target outside ``(0, 1]``.
    """
    if not 0.0 < relative_target <= 1.0:
        msg = f"relative_target must be in (0, 1], got {relative_target}"
        raise ScoreContractError(msg)
    panel = [np.asarray(r, dtype=np.float64) for r in expert_panel]
    if len(panel) < 2:
        msg = (
            f"the panel's own agreement needs at least 2 expert raters, got "
            f"{len(panel)}; without it there is no ceiling to measure against"
        )
        raise ScoreContractError(msg)
    automated_scores = np.asarray(automated, dtype=np.float64)
    if any(r.shape != automated_scores.shape for r in panel):
        msg = "every rater must score exactly the same cases as the scorer under test"
        raise ScoreContractError(msg)

    panel_matrix = np.column_stack(panel)
    # Automated-versus-expert uses the panel consensus as the comparator, so
    # the figure describes one rater against the reference rather than against
    # an arbitrary panel member.
    consensus = panel_matrix.mean(axis=1)
    automated_vs_expert = icc_2_1(np.column_stack([automated_scores, consensus]))
    expert_vs_expert = icc_2_1(panel_matrix)

    return AgreementFigure(
        endpoint=endpoint,
        band=band,
        automated_vs_expert=automated_vs_expert,
        expert_vs_expert=expert_vs_expert,
        relative_target=relative_target,
        n_cases=int(automated_scores.size),
        absolute_floor=absolute_floor,
        min_panel_icc=min_panel_icc,
    )


class AgreementGate(BaseModel):
    """Configuration for the Phase 1 gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1"
    relative_target: Annotated[float, Field(gt=0.0, le=1.0)] = DEFAULT_RELATIVE_TARGET
    #: Panel agreement below which no scorer can pass, however it performs.
    min_panel_icc: Annotated[float, Field(gt=0.0, lt=1.0)] = DEFAULT_MIN_PANEL_ICC
    #: Requirement floor, independent of the panel.
    absolute_floor: Annotated[float, Field(gt=0.0, lt=1.0)] = DEFAULT_ABSOLUTE_FLOOR
    #: Cases required in the stratified cohort before the figure is reportable.
    #: A relative target computed on a handful of cases is noise.
    min_cases: Annotated[int, Field(ge=2)] = 30

    def evaluate(self, figures: Mapping[Endpoint, AgreementFigure]) -> AgreementGateResult:
        """Apply the gate.

        Args:
            figures: One figure per endpoint. Binary proficiency is required.

        Returns:
            The verdict, with the primary line and any secondary lines.
        """
        primary = figures.get(Endpoint.BINARY_PROFICIENCY)
        if primary is None:
            return AgreementGateResult(
                passed=False,
                reason=(
                    "no binary proficiency figure was submitted; it is the primary "
                    "endpoint and GEARS alone does not constitute a result "
                    "(PLAN.md section 13)"
                ),
                primary="absent",
                secondary=tuple(f.describe() for f in figures.values()),
            )

        secondary = tuple(
            f.describe() for e, f in figures.items() if e is not Endpoint.BINARY_PROFICIENCY
        )

        if not primary.is_stratified:
            return AgreementGateResult(
                passed=False,
                reason=(
                    "the primary figure was computed on a mixed-band cohort; "
                    "between-group variance inflates it and credentialing needs "
                    "within-band discrimination, so a mixed figure is secondary only"
                ),
                primary=primary.describe(),
                secondary=secondary,
            )
        if not primary.panel_is_adequate:
            return AgreementGateResult(
                passed=False,
                reason=(
                    f"the expert panel agreed with itself at only "
                    f"{primary.expert_vs_expert.value:.3f}, below the "
                    f"{primary.min_panel_icc:g} adequacy minimum; a panel that "
                    f"cannot agree with itself is not a ceiling to measure "
                    f"against, so no scorer can pass on these cases. Fix the "
                    f"rubric or the raters, not the scorer."
                ),
                primary=primary.describe(),
                secondary=secondary,
            )
        if primary.n_cases < self.min_cases:
            return AgreementGateResult(
                passed=False,
                reason=(
                    f"the primary figure rests on {primary.n_cases} cases, below the "
                    f"{self.min_cases}-case floor; a relative target on fewer is noise"
                ),
                primary=primary.describe(),
                secondary=secondary,
            )
        if not primary.passes:
            return AgreementGateResult(
                passed=False,
                reason=(
                    f"automated agreement {primary.achieved:.3f} is below "
                    f"{primary.relative_target:g}x the panel's own "
                    f"{primary.expert_vs_expert.value:.3f} "
                    f"(required {primary.required:.3f})"
                ),
                primary=primary.describe(),
                secondary=secondary,
            )
        return AgreementGateResult(
            passed=True,
            reason=(
                f"automated agreement {primary.achieved:.3f} reaches "
                f"{primary.relative_target:g}x the panel's own "
                f"{primary.expert_vs_expert.value:.3f} on {primary.n_cases} "
                f"{primary.band.value if primary.band else 'mixed'} cases"
            ),
            primary=primary.describe(),
            secondary=secondary,
        )
