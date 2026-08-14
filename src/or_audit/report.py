"""Output artifacts: scorecard, credentialing report, learning curve.

PLAN.md section 7 lists these as the wedge's deliverables. Two constraints from
the plan shape all three:

* **§9 attribution control.** An institution whose privilege posture has not
  been confirmed gets aggregate reporting only. Individual attribution is a
  capability that has to be earned, not the default, because the artifact can
  be adverse to a named clinician and V-3 is unresolved.
* **§7.1 no implicit collapse.** A report may present the determination the
  decision rule produced, and must present the gates and skill separately
  alongside it. It never invents a composite figure of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from or_audit.decision.record import DecisionRecord
from or_audit.domain.entities import Institution, Procedure, Surgeon
from or_audit.domain.enums import Determination, GateStatus, SkillBand
from or_audit.errors import DomainInvariantError
from or_audit.scoring.gates import GateResult
from or_audit.scoring.skill import SkillScore


class GateLine(BaseModel):
    """One gate, as it appears on a report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: str
    status: GateStatus
    reason: str


class EpisodeScorecard(BaseModel):
    """Per-episode vector scorecard.

    Gates and skill appear side by side and are never merged. The
    determination is shown because a versioned rule produced it, not because
    the report worked one out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    performed_at: datetime
    procedure: str
    band_at_episode: SkillBand
    #: Present only when individual attribution is permitted (§9).
    surgeon_ref: str | None

    gates: tuple[GateLine, ...]
    proficiency_met: int
    proficiency_assessable: int
    proficiency_unassessable: int
    gears_total: int | None

    determination: Determination
    determination_reason: str
    rule_version: str
    perception: str

    @property
    def proficiency_fraction(self) -> float | None:
        """Met over assessable, or ``None`` when nothing was assessable.

        ``None`` rather than 0.0: a zero denominator is missing evidence, and a
        report showing 0% would read as a total failure by the surgeon.
        """
        if self.proficiency_assessable == 0:
            return None
        return self.proficiency_met / self.proficiency_assessable

    def render(self) -> str:
        """Plain-text scorecard."""
        subject = self.surgeon_ref or "<attribution withheld>"
        fraction = (
            "n/a" if self.proficiency_fraction is None else f"{self.proficiency_fraction:.0%}"
        )
        lines = [
            f"Episode {self.episode_id}  {self.procedure}",
            f"  performed    {self.performed_at.isoformat()}",
            f"  surgeon      {subject} ({self.band_at_episode.value})",
            "  safety gates",
        ]
        lines.extend(
            f"    {line.status.value:<15} {line.gate}: {line.reason}" for line in self.gates
        )
        lines += [
            f"  proficiency  {self.proficiency_met}/{self.proficiency_assessable} "
            f"assessable met ({fraction}); "
            f"{self.proficiency_unassessable} not assessable",
            f"  GEARS total  {self.gears_total if self.gears_total is not None else 'incomplete'}",
            f"  DETERMINATION {self.determination.value.upper()}",
            f"    {self.determination_reason}",
            f"  provenance   rule {self.rule_version}, perception {self.perception}",
        ]
        return "\n".join(lines)


def build_scorecard(
    *,
    episode_id: str,
    performed_at: datetime,
    procedure: Procedure,
    band: SkillBand,
    surgeon: Surgeon,
    institution: Institution,
    gates: Sequence[GateResult],
    skill: SkillScore,
    decision: DecisionRecord,
) -> EpisodeScorecard:
    """Assemble a scorecard, honouring the attribution rule.

    ``surgeon_ref`` is populated only when the institution's peer-review
    protection has been confirmed. Section 9 leaves that unresolved (V-3), so
    the default is to withhold the identifier rather than to publish it and
    hope.
    """
    return EpisodeScorecard(
        episode_id=episode_id,
        performed_at=performed_at,
        procedure=procedure.display_name,
        band_at_episode=band,
        surgeon_ref=(
            surgeon.external_ref if institution.peer_review_protection_confirmed else None
        ),
        gates=tuple(GateLine(gate=g.gate.value, status=g.status, reason=g.reason) for g in gates),
        proficiency_met=skill.met_count,
        proficiency_assessable=len(skill.assessable),
        proficiency_unassessable=len(skill.unassessable),
        gears_total=skill.gears_total,
        determination=decision.effective_determination,
        determination_reason=decision.reason,
        rule_version=decision.rule_version,
        perception=decision.perception_identity,
    )


class LearningCurvePoint(BaseModel):
    """One episode's position on a surgeon's trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: Annotated[int, Field(ge=1)]
    performed_at: datetime
    proficiency_fraction: float | None
    determination: Determination


class LearningCurve(BaseModel):
    """A surgeon's trajectory across episodes.

    Deliberately not a fitted curve. Section 13's honesty about what the data
    supports applies here too: with the case volumes a Phase 1 pilot produces,
    a trend line implies precision the sample does not have. The points are
    reported, and the only derived figure is a plain comparison of first and
    last thirds, labelled as such.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    surgeon_ref: str | None
    band: SkillBand
    points: tuple[LearningCurvePoint, ...]

    @property
    def assessable_points(self) -> tuple[LearningCurvePoint, ...]:
        """Points with a defined proficiency fraction."""
        return tuple(p for p in self.points if p.proficiency_fraction is not None)

    def trend(self) -> str:
        """First-third versus last-third comparison, or a refusal to guess.

        Requires six assessable points. Below that the comparison would rest on
        one or two episodes per third, which is not a trend.
        """
        usable = self.assessable_points
        if len(usable) < 6:
            return (
                f"insufficient data for a trend: {len(usable)} assessable "
                f"episode(s), at least 6 needed"
            )
        third = len(usable) // 3
        early = [p.proficiency_fraction or 0.0 for p in usable[:third]]
        late = [p.proficiency_fraction or 0.0 for p in usable[-third:]]
        delta = sum(late) / len(late) - sum(early) / len(early)
        direction = "improving" if delta > 0.05 else "declining" if delta < -0.05 else "flat"
        return (
            f"{direction}: first third {sum(early) / len(early):.0%}, "
            f"last third {sum(late) / len(late):.0%} "
            f"(difference of means, not a fitted trend)"
        )


def build_learning_curve(
    scorecards: Sequence[EpisodeScorecard], *, band: SkillBand
) -> LearningCurve:
    """Assemble a learning curve from scorecards in chronological order.

    Raises:
        DomainInvariantError: If the scorecards are not all for one surgeon, or
            are not in chronological order. An out-of-order curve would make
            the trend meaningless in a way no reader could detect.
    """
    if not scorecards:
        msg = "a learning curve needs at least one episode"
        raise DomainInvariantError(msg)
    refs = {s.surgeon_ref for s in scorecards}
    if len(refs) > 1:
        msg = f"a learning curve must cover one surgeon, got {len(refs)} distinct references"
        raise DomainInvariantError(msg)
    times = [s.performed_at for s in scorecards]
    if times != sorted(times):
        msg = "scorecards must be in chronological order for a learning curve to mean anything"
        raise DomainInvariantError(msg)
    return LearningCurve(
        surgeon_ref=scorecards[0].surgeon_ref,
        band=band,
        points=tuple(
            LearningCurvePoint(
                sequence=index,
                performed_at=card.performed_at,
                proficiency_fraction=card.proficiency_fraction,
                determination=card.determination,
            )
            for index, card in enumerate(scorecards, start=1)
        ),
    )


class CredentialingReport(BaseModel):
    """The artifact a credentialing committee receives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    institution: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    surgeon_ref: str | None
    band: SkillBand
    generated_at: datetime
    rule_description: str
    scorecards: tuple[EpisodeScorecard, ...]
    curve: LearningCurve
    #: Set when the institution's privilege posture is unconfirmed, so the
    #: committee knows why names are absent.
    attribution_note: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        """Episodes by determination."""
        return {
            determination.value: sum(1 for s in self.scorecards if s.determination is determination)
            for determination in Determination
        }

    def render(self) -> str:
        """Plain-text report."""
        subject = self.surgeon_ref or "<attribution withheld>"
        counts = self.counts
        lines = [
            "CREDENTIALING REPORT",
            f"  institution  {self.institution}",
            f"  surgeon      {subject} ({self.band.value})",
            f"  generated    {self.generated_at.isoformat()}",
            f"  episodes     {len(self.scorecards)}",
            "  determinations",
        ]
        lines.extend(f"    {name:<18} {count}" for name, count in counts.items())
        lines += [
            f"  trajectory   {self.curve.trend()}",
            f"  rule         {self.rule_description}",
        ]
        if self.attribution_note:
            lines.append(f"  NOTE         {self.attribution_note}")
        lines.append("")
        lines.extend(card.render() for card in self.scorecards)
        return "\n".join(lines)
