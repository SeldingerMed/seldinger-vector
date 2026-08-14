"""What perception reports, and the vocabulary it reports in.

Perception is the layer between pixels and judgement. Everything above it --
safety gates, skill scoring, determinations -- consumes only the types here, so
a model-backed backend and an expert-annotation backend are interchangeable to
everything downstream. That interchangeability is the point: PLAN.md section 13
calibrates automated scoring against a human panel, which is only possible if
both produce the same shape.

The phase vocabulary follows Cholec80, the public dataset the plan names as a
perception baseline (section 3). Using its labels rather than inventing our own
means published models can be slotted in without a translation layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from or_audit.errors import DomainInvariantError
from or_audit.primitives import Sha256Hex


class SurgicalPhase(StrEnum):
    """Cholecystectomy phases, per the Cholec80 taxonomy."""

    PREPARATION = "preparation"
    CALOT_TRIANGLE_DISSECTION = "calot_triangle_dissection"
    CLIPPING_AND_CUTTING = "clipping_and_cutting"
    GALLBLADDER_DISSECTION = "gallbladder_dissection"
    GALLBLADDER_PACKAGING = "gallbladder_packaging"
    CLEANING_AND_COAGULATION = "cleaning_and_coagulation"
    GALLBLADDER_RETRACTION = "gallbladder_retraction"


class CriticalStructure(StrEnum):
    """Structures whose injury defines the harm this platform screens for."""

    CYSTIC_DUCT = "cystic_duct"
    CYSTIC_ARTERY = "cystic_artery"
    COMMON_BILE_DUCT = "common_bile_duct"
    COMMON_HEPATIC_DUCT = "common_hepatic_duct"
    RIGHT_HEPATIC_ARTERY = "right_hepatic_artery"
    CYSTIC_PLATE = "cystic_plate"
    HEPATOCYSTIC_TRIANGLE = "hepatocystic_triangle"
    #: Present in the vocabulary because ureteric injury is the analogous
    #: never-event outside biliary surgery. No gate uses it yet.
    URETER = "ureter"


class CvsCriterion(StrEnum):
    """The three Strasberg criteria for the Critical View of Safety.

    All three must hold before the cystic duct is divided. They are separate
    criteria rather than a score, and the gate reports them separately, because
    "two of three" is not a partial pass -- it is a fail with a specific
    reason.
    """

    #: The hepatocystic triangle is cleared of fat and fibrous tissue.
    TRIANGLE_CLEARED = "hepatocystic_triangle_cleared"
    #: The lower third of the gallbladder is separated from the cystic plate.
    CYSTIC_PLATE_EXPOSED = "lower_third_cystic_plate_exposed"
    #: Two and only two structures are seen entering the gallbladder.
    TWO_STRUCTURES_ONLY = "two_structures_entering_gallbladder"


class ObservationKind(StrEnum):
    """What a backend actually looked for.

    Without this, "measured and found nothing" is indistinguishable from
    "never looked", and every gate clears on an empty result. A backend that
    does not implement proximity detection would otherwise pass the proximity
    gate by returning nothing -- which is precisely the abstention failure
    PLAN.md section 7.1 exists to prevent.

    Declaring coverage is therefore mandatory, and reporting an observation of
    a kind you did not declare is a contradiction the model rejects.
    """

    PHASES = "phases"
    STRUCTURES = "structures"
    PROXIMITY = "proximity"
    BLEEDING = "bleeding"
    CVS = "cvs"


class BleedingSeverity(StrEnum):
    """Coarse bleeding severity. Ordinal, and compared as such."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


#: Ordering for :class:`BleedingSeverity`. Kept beside the enum rather than
#: relying on declaration order, which is not a stable contract.
BLEEDING_RANK: dict[BleedingSeverity, int] = {
    BleedingSeverity.NONE: 0,
    BleedingSeverity.MINOR: 1,
    BleedingSeverity.MODERATE: 2,
    BleedingSeverity.MAJOR: 3,
}

#: Confidence in [0, 1].
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class _Obs(BaseModel):
    """Base for immutable observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PhaseSegment(_Obs):
    """A contiguous span assigned to one phase."""

    phase: SurgicalPhase
    start_s: Annotated[float, Field(ge=0.0)]
    end_s: Annotated[float, Field(gt=0.0)]
    confidence: Confidence

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s <= self.start_s:
            msg = f"phase segment must have positive duration, got [{self.start_s}, {self.end_s})"
            raise DomainInvariantError(msg)
        return self


class StructureSighting(_Obs):
    """A structure identified over a span."""

    structure: CriticalStructure
    start_s: Annotated[float, Field(ge=0.0)]
    end_s: Annotated[float, Field(gt=0.0)]
    confidence: Confidence

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s <= self.start_s:
            msg = f"structure sighting must have positive duration for {self.structure}"
            raise DomainInvariantError(msg)
        return self


class ProximityEvent(_Obs):
    """An instrument came close to a structure it should not touch."""

    structure: CriticalStructure
    at_s: Annotated[float, Field(ge=0.0)]
    #: Distance in millimetres if measurable, otherwise ``None``. Absent
    #: distance is not zero distance; the gate treats it as unquantified.
    distance_mm: Annotated[float, Field(ge=0.0)] | None
    confidence: Confidence


class BleedingEvent(_Obs):
    """A bleeding episode."""

    severity: BleedingSeverity
    start_s: Annotated[float, Field(ge=0.0)]
    end_s: Annotated[float, Field(gt=0.0)]
    confidence: Confidence

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s <= self.start_s:
            msg = "bleeding event must have positive duration"
            raise DomainInvariantError(msg)
        if self.severity is BleedingSeverity.NONE:
            msg = "a bleeding event cannot have severity 'none'; omit it instead"
            raise DomainInvariantError(msg)
        return self


class CvsObservation(_Obs):
    """Evidence for one Strasberg criterion.

    ``achieved`` is deliberately three-valued. ``None`` means the backend could
    not tell -- occluded view, phase never observed, model below its own
    operating threshold -- and it must never be collapsed to ``False``. A
    criterion that could not be assessed is a gap in evidence, not a finding
    against the surgeon (PLAN.md sections 7.2 and 7.3).
    """

    criterion: CvsCriterion
    achieved: bool | None
    #: When the criterion was satisfied. Required when ``achieved`` is true,
    #: because the gate compares it against the point of no return.
    at_s: Annotated[float, Field(ge=0.0)] | None = None
    confidence: Confidence = 0.0
    note: str = ""

    @model_validator(mode="after")
    def _timing(self) -> Self:
        if self.achieved is True and self.at_s is None:
            msg = (
                f"criterion {self.criterion.value} is recorded as achieved but "
                f"carries no timestamp; the gate cannot tell whether it "
                f"preceded division of the cystic duct"
            )
            raise DomainInvariantError(msg)
        return self


class PerceptionResult(_Obs):
    """Everything one backend observed for one episode.

    Bound to the media digests it ran on. Without that binding a result could
    be paired with a different recording, which is the same class of mistake
    the de-identification plan/source check closes.
    """

    backend: str
    backend_version: str
    #: Digests of the attested media this result was computed from.
    media_sha256: tuple[Sha256Hex, ...]
    #: Digests of the de-identification attestations that cleared that media.
    #: Carried so a consumer of this result can confirm section 8 was honoured
    #: without re-deriving it from an Episode it may not hold.
    deid_attestation_sha256: tuple[Sha256Hex, ...]
    duration_s: Annotated[float, Field(gt=0.0)]
    #: The observation kinds this backend assessed. Absence of an observation
    #: only means "none found" for kinds listed here.
    observes: frozenset[ObservationKind]

    phases: tuple[PhaseSegment, ...] = ()
    structures: tuple[StructureSighting, ...] = ()
    proximity_events: tuple[ProximityEvent, ...] = ()
    bleeding_events: tuple[BleedingEvent, ...] = ()
    cvs: tuple[CvsObservation, ...] = ()

    @model_validator(mode="after")
    def _bound_to_media(self) -> Self:
        if not self.media_sha256:
            msg = f"perception result from {self.backend} names no source media"
            raise DomainInvariantError(msg)
        if not self.deid_attestation_sha256:
            msg = (
                f"perception result from {self.backend} names no de-identification "
                f"attestation; only cleared media may be perceived (PLAN.md section 8)"
            )
            raise DomainInvariantError(msg)
        seen = {obs.criterion for obs in self.cvs}
        if len(seen) != len(self.cvs):
            msg = f"perception result from {self.backend} reports a CVS criterion twice"
            raise DomainInvariantError(msg)
        return self

    @model_validator(mode="after")
    def _observations_match_declared_coverage(self) -> Self:
        """Reporting an observation of an undeclared kind is a contradiction."""
        reported = {
            ObservationKind.PHASES: bool(self.phases),
            ObservationKind.STRUCTURES: bool(self.structures),
            ObservationKind.PROXIMITY: bool(self.proximity_events),
            ObservationKind.BLEEDING: bool(self.bleeding_events),
            ObservationKind.CVS: bool(self.cvs),
        }
        undeclared = sorted(
            k.value for k, present in reported.items() if present and k not in self.observes
        )
        if undeclared:
            msg = (
                f"perception result from {self.backend} reports {', '.join(undeclared)} "
                f"observations without declaring coverage of them"
            )
            raise DomainInvariantError(msg)
        return self

    def assessed(self, kind: ObservationKind) -> bool:
        """Whether the backend looked for ``kind`` at all."""
        return kind in self.observes

    @property
    def identity(self) -> str:
        """Backend and version, as recorded on any artifact derived from this."""
        return f"{self.backend}@{self.backend_version}"

    def phase_span(
        self, phase: SurgicalPhase, *, min_confidence: float = 0.0
    ) -> tuple[float, float] | None:
        """Earliest start and latest end observed for ``phase``.

        Args:
            phase: Phase to locate.
            min_confidence: Ignore segments below this confidence. A gate that
                anchors a timing verdict on a phase should floor it the same
                way it floors every other input, otherwise a phantom detection
                manufactures a conclusion.

        Returns:
            The span, or ``None`` if the phase was never confidently observed.
            Absence is meaningful: a gate that cannot locate a phase it depends
            on must report itself unassessable rather than guess.
        """
        spans = [
            (p.start_s, p.end_s)
            for p in self.phases
            if p.phase is phase and p.confidence >= min_confidence
        ]
        if not spans:
            return None
        return min(s for s, _ in spans), max(e for _, e in spans)

    def cvs_observation(self, criterion: CvsCriterion) -> CvsObservation | None:
        """The observation for ``criterion``, if the backend reported one."""
        return next((obs for obs in self.cvs if obs.criterion is criterion), None)

    def worst_bleeding(self) -> BleedingSeverity:
        """Highest severity observed, or ``NONE``."""
        if not self.bleeding_events:
            return BleedingSeverity.NONE
        return max(
            (event.severity for event in self.bleeding_events),
            key=lambda severity: BLEEDING_RANK[severity],
        )
