"""The pre-registered decision rule.

PLAN.md section 7.2 is the argument for this module, and it is worth restating
because the temptation to skip it is strong.

Credentialing terminates in a binary act: privileges are granted or they are
not. So *someone* collapses the score vector. If the platform refuses to, every
hospital invents its own unreviewed collapse, and every resulting dispute lands
on our artifact anyway -- with us unable to say what the number meant. Owning
the collapse is therefore safer than avoiding it.

Four properties make it defensible rather than merely convenient:

* **Pre-registered and versioned.** The rule is a value object with a version.
  It is published before the pilot and changed only by a new version, so a
  determination can always be re-derived from the rule that produced it.
* **Abstention is a required output.** ``INDETERMINATE`` exists so a scorer that
  cannot decide is not forced into false confidence exactly where liability
  concentrates.
* **Hard gates are not averaged in.** A failed safety gate is dispositive; it
  does not trade off against a good proficiency score. Section 7.1 forbids the
  averaging and this is where it would otherwise happen.
* **The threshold owner is named.** Whoever set the bar carries the
  consequence of it being wrong, and the artifact says who that was.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from or_audit.domain.enums import Determination, GateStatus, ThresholdOwner
from or_audit.errors import DomainInvariantError, ScoreContractError
from or_audit.scoring.skill import ScoreVector


class DecisionRule(BaseModel):
    """A pre-registered rule for collapsing a score vector to a determination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Bumped on any change to the fields below. A determination records the
    #: version that produced it, so it can be re-derived exactly.
    version: Annotated[str, StringConstraints(min_length=1, max_length=32)]

    #: Who set the benchmark. Section 7.2: this allocates responsibility for
    #: the threshold being wrong, so it is required rather than defaulted.
    threshold_owner: ThresholdOwner

    #: Human-readable citation for the threshold's provenance -- a society
    #: standard, a committee minute, a contract schedule. Required, because a
    #: threshold nobody can trace is a threshold nobody will defend.
    threshold_provenance: Annotated[str, StringConstraints(min_length=1, max_length=500)]

    #: Fraction of assessable proficiency items that must be met.
    min_proficiency_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.85

    #: Minimum assessable items before a proficiency fraction means anything.
    #: A 1-of-1 pass is not evidence of proficiency.
    min_assessable_items: Annotated[int, Field(ge=1)] = 5

    #: Whether an unassessable safety gate blocks a positive determination.
    #: Default true: section 7.1 treats an episode nobody could evaluate as
    #: not cleared, and a credentialing decision should inherit that.
    unassessable_gate_blocks: bool = True

    @model_validator(mode="after")
    def _sanity(self) -> Self:
        if self.threshold_owner is ThresholdOwner.SPECIALTY_SOCIETY and (
            "society" not in self.threshold_provenance.lower()
            and "sages" not in self.threshold_provenance.lower()
            and "acs" not in self.threshold_provenance.lower()
        ):
            msg = (
                "a threshold attributed to a specialty society must cite it in "
                "threshold_provenance; an unattributed society threshold is the "
                "easiest kind of claim to make and the hardest to defend"
            )
            raise DomainInvariantError(msg)
        return self

    def apply(self, vector: ScoreVector) -> tuple[Determination, str]:
        """Collapse ``vector`` to a determination and the reason for it.

        Ordering is deliberate and is the substance of the rule:

        1. **A failed hard gate is dispositive.** No proficiency score offsets
           it. This is the averaging section 7.1 prohibits, and it would happen
           here if the ordering were different.
        2. **An unassessable gate blocks a pass**, but yields
           ``INDETERMINATE`` rather than ``DOES_NOT_MEET`` -- missing evidence
           is not a finding against the surgeon.
        3. **Too little assessable skill evidence** is likewise
           ``INDETERMINATE``.
        4. Only then does the proficiency threshold decide.

        Returns:
            The determination and a one-sentence reason naming the rule
            version, so a reader never has to guess which rule applied.
        """
        prefix = f"rule {self.version}"

        failed = vector.gates.failed
        if failed:
            names = ", ".join(sorted(g.gate.value for g in failed))
            return (
                Determination.DOES_NOT_MEET,
                f"{prefix}: safety gate failure is dispositive ({names}); "
                f"no skill score offsets it",
            )

        unassessable = vector.gates.unassessable
        if unassessable and self.unassessable_gate_blocks:
            names = ", ".join(sorted(g.gate.value for g in unassessable))
            return (
                Determination.INDETERMINATE,
                f"{prefix}: {names} could not be assessed, so the episode has "
                f"not been cleared; this is missing evidence, not a finding "
                f"against the surgeon",
            )

        not_applicable = tuple(
            g for g in vector.gates.results if g.status is GateStatus.NOT_APPLICABLE
        )

        assessable = vector.skill.assessable
        if len(assessable) < self.min_assessable_items:
            return (
                Determination.INDETERMINATE,
                f"{prefix}: only {len(assessable)} proficiency item(s) could be "
                f"assessed, below the {self.min_assessable_items} required for a "
                f"proficiency fraction to carry meaning",
            )

        fraction = vector.skill.proficiency_fraction
        applicability = (
            f" ({len(not_applicable)} gate(s) not applicable to this procedure)"
            if not_applicable
            else ""
        )
        if fraction >= self.min_proficiency_fraction:
            return (
                Determination.MEETS_BENCHMARK,
                f"{prefix}: all applicable safety gates cleared and "
                f"{fraction:.0%} of {len(assessable)} assessable proficiency "
                f"items were met, at or above the {self.min_proficiency_fraction:.0%} "
                f"benchmark set by {self.threshold_owner.value}{applicability}",
            )
        return (
            Determination.DOES_NOT_MEET,
            f"{prefix}: {fraction:.0%} of {len(assessable)} assessable "
            f"proficiency items were met, below the "
            f"{self.min_proficiency_fraction:.0%} benchmark set by "
            f"{self.threshold_owner.value}{applicability}",
        )

    def describe(self) -> str:
        """The rule in one line, for publication before a pilot."""
        return (
            f"DecisionRule {self.version}: all applicable safety gates must "
            f"clear; >={self.min_proficiency_fraction:.0%} of at least "
            f"{self.min_assessable_items} assessable proficiency items must be "
            f"met; unassessable gates "
            f"{'block' if self.unassessable_gate_blocks else 'do not block'} a "
            f"pass. Threshold set by {self.threshold_owner.value} "
            f"({self.threshold_provenance})."
        )


def forbid_scalar_collapse(_vector: ScoreVector) -> float:
    """Deliberately not implemented.

    Exists so that a developer looking for "just give me a number" finds an
    explanation instead of writing one. The determination is the only sanctioned
    collapse, it is versioned, and it is attributable.

    Raises:
        ScoreContractError: Always.
    """
    msg = (
        "there is no scalar score. The only sanctioned collapse of a score "
        "vector is a Determination produced by a versioned DecisionRule, so "
        "that whoever set the threshold can be named and the result re-derived "
        "(PLAN.md section 7.2)"
    )
    raise ScoreContractError(msg)
