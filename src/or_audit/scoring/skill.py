"""Soft skill scores, and the vector that refuses to become a number.

Endpoint ordering follows PLAN.md section 13 and is enforced, not merely
documented: **binary proficiency is primary, GEARS is secondary**. A randomised
trial found binary scoring metrics outperformed GEARS on reliability and
discrimination, and GEARS' interobserver reliability degrades sharply with
non-expert raters. Building the headline on the weaker instrument would import
its noise into the number the whole product rests on.

GEARS is retained because programmes already use it and a report that cannot
speak their language is harder to adopt. It travels as a secondary figure and
:class:`SkillScore` will not produce a summary without the primary endpoint.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.domain.enums import SkillBand
from or_audit.errors import DomainInvariantError, ScoreContractError
from or_audit.scoring.gates import SafetyGateSet


class GearsDomain(StrEnum):
    """The six GEARS domains, each scored 1-5."""

    DEPTH_PERCEPTION = "depth_perception"
    BIMANUAL_DEXTERITY = "bimanual_dexterity"
    EFFICIENCY = "efficiency"
    FORCE_SENSITIVITY = "force_sensitivity"
    ROBOTIC_CONTROL = "robotic_control"
    AUTONOMY = "autonomy"


class ProficiencyItem(StrEnum):
    """Binary proficiency items: observable, checkable, either done or not.

    Binary items avoid the central weakness of a Likert rubric, which is that
    "3 versus 4 on bimanual dexterity" means different things to different
    raters. Each item here is phrased so two competent observers watching the
    same case should reach the same answer.
    """

    #: Instruments remained within the visual field throughout.
    INSTRUMENTS_IN_VIEW = "instruments_remained_in_view"
    #: The camera was repositioned to keep the operative field centred.
    CAMERA_MANAGED = "camera_actively_managed"
    #: The non-dominant hand provided useful retraction rather than idling.
    EFFECTIVE_RETRACTION = "non_dominant_hand_provided_retraction"
    #: Dissection followed anatomical planes rather than tearing through them.
    ANATOMICAL_PLANES_RESPECTED = "dissection_followed_anatomical_planes"
    #: Energy was applied away from critical structures.
    SAFE_ENERGY_USE = "energy_applied_away_from_critical_structures"
    #: Clips were placed under direct vision.
    CLIPS_UNDER_VISION = "clips_placed_under_direct_vision"
    #: No instrument was exchanged without the tip being visible.
    SAFE_INSTRUMENT_EXCHANGE = "instrument_exchanges_under_vision"


class ProficiencyResult(BaseModel):
    """Verdict on one binary item.

    ``met`` is three-valued for the same reason the CVS observations are: an
    item the rater could not judge is a gap in evidence, not a failure by the
    surgeon.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item: ProficiencyItem
    met: bool | None
    note: str = ""


class GearsRating(BaseModel):
    """One GEARS domain, scored on the published 1-5 scale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: GearsDomain
    score: Annotated[int, Field(ge=1, le=5)]
    note: str = ""


class SkillScore(BaseModel):
    """Soft skill assessment for one episode.

    Not a number, and deliberately awkward to turn into one. The primary
    endpoint is a count of met items out of assessable items; GEARS is carried
    alongside and never merged with it, because averaging a binary count with
    a Likert scale produces a figure that means nothing in either instrument.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    band_at_episode: SkillBand
    #: Who or what produced these scores, recorded for the audit trail.
    rater: str
    proficiency: tuple[ProficiencyResult, ...]
    gears: tuple[GearsRating, ...] = ()

    @model_validator(mode="after")
    def _no_duplicates_and_primary_present(self) -> Self:
        items = [p.item for p in self.proficiency]
        if len(set(items)) != len(items):
            msg = "the same proficiency item was scored twice"
            raise DomainInvariantError(msg)
        domains = [g.domain for g in self.gears]
        if len(set(domains)) != len(domains):
            msg = "the same GEARS domain was scored twice"
            raise DomainInvariantError(msg)
        if not self.proficiency:
            msg = (
                "a skill score must carry binary proficiency items; GEARS alone "
                "is not a result, it is the secondary endpoint (PLAN.md section 13)"
            )
            raise DomainInvariantError(msg)
        return self

    @property
    def assessable(self) -> tuple[ProficiencyResult, ...]:
        """Items the rater could actually judge."""
        return tuple(p for p in self.proficiency if p.met is not None)

    @property
    def unassessable(self) -> tuple[ProficiencyResult, ...]:
        """Items the rater could not judge."""
        return tuple(p for p in self.proficiency if p.met is None)

    @property
    def met_count(self) -> int:
        """Items affirmatively met."""
        return sum(1 for p in self.proficiency if p.met is True)

    @property
    def proficiency_fraction(self) -> float:
        """Met items as a fraction of assessable items.

        The primary endpoint. Denominator excludes unassessable items so that
        poor video does not read as poor performance.

        Raises:
            ScoreContractError: If nothing was assessable. A fraction with a
                zero denominator is not zero, it is undefined, and returning
                0.0 would read as a total failure by the surgeon.
        """
        assessable = self.assessable
        if not assessable:
            msg = (
                "no proficiency item could be assessed, so the primary endpoint is "
                "undefined; this is missing evidence, not a score of zero"
            )
            raise ScoreContractError(msg)
        return sum(1 for p in assessable if p.met) / len(assessable)

    @property
    def gears_total(self) -> int | None:
        """Sum of GEARS domains, or ``None`` if not all six were rated.

        A partial GEARS total is not comparable to a complete one, and the
        published benchmarks assume all six domains.
        """
        if len(self.gears) != len(GearsDomain):
            return None
        return sum(g.score for g in self.gears)


class ScoreVector(BaseModel):
    """Hard gates and soft scores for one episode, kept apart.

    PLAN.md section 7.1 requires the vector never to collapse implicitly and
    hard gates never to average into soft scores. Both are enforced here by
    making the operations raise. A caller that wants a single answer must go
    through the decision rule, where the collapse is pre-registered, versioned
    and attributable -- not through arithmetic on this object.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gates: SafetyGateSet
    skill: SkillScore

    def __float__(self) -> float:
        """Always raises."""
        msg = (
            "a score vector has no scalar value; safety gates and skill scores "
            "are different kinds of judgement and averaging them produces a "
            "number that means nothing in either (PLAN.md section 7.1). Use the "
            "decision rule if a single determination is needed."
        )
        raise ScoreContractError(msg)

    def __int__(self) -> int:
        """Always raises."""
        return int(self.__float__())

    def __bool__(self) -> bool:
        """Always raises, so the question gets named."""
        msg = (
            "a score vector has no truth value; ask gates.all_clear or inspect "
            "skill.proficiency_fraction so the call site says which it means"
        )
        raise ScoreContractError(msg)
