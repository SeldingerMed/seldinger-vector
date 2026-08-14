"""Perception backends.

Two things are true at once in the alpha, and the module is honest about both.

**The interface is the deliverable.** :class:`PerceptionBackend` is what a
CV model implements. Nothing above this layer knows or cares which backend ran,
only that the result is bound to the media digests and carries a version.

**The annotation backend is the one that works today, and that is not a
placeholder.** PLAN.md section 13 makes expert-panel agreement the Phase 1
gate, so scoring from expert annotations is the platform's first real operating
mode, not a stand-in for it. The panel produces the labels; the gates consume
them through the same interface a model would. When a trained model arrives it
slots in beside this, and the metrics harness compares them.

What is deliberately *not* here: a claim to have a trained perception model.
Phase recognition and structure identification require training data the plan
does not yet have (section 10), so there is no heuristic pretending to do it.
Bleeding is different -- it is measurable from pixels -- and lives in
:mod:`or_audit.perception.bleeding`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from or_audit.domain.entities import Episode, MediaAsset
from or_audit.domain.enums import MediaKind
from or_audit.errors import DomainInvariantError
from or_audit.perception.observations import (
    BleedingEvent,
    CvsObservation,
    ObservationKind,
    PerceptionResult,
    PhaseSegment,
    ProximityEvent,
    StructureSighting,
)

ANNOTATION_BACKEND = "expert-annotation"
ANNOTATION_BACKEND_VERSION = "1"


@runtime_checkable
class PerceptionBackend(Protocol):
    """Produces observations for an episode."""

    @property
    def identity(self) -> str:
        """Stable ``name@version`` recorded on every derived artifact."""
        ...

    def analyse(self, episode: Episode) -> PerceptionResult:
        """Observe ``episode``.

        Implementations MUST call :meth:`Episode.require_readable` before
        touching media, and MUST bind the result to the media digests they
        read.
        """
        ...


def readable_video(episode: Episode) -> tuple[MediaAsset, ...]:
    """Endoscopic video for ``episode``, gated on de-identification.

    Raises:
        DeidentificationBoundaryError: If any surviving asset is uncleared.
        DomainInvariantError: If no endoscopic video survives.
    """
    episode.require_readable()
    video = tuple(a for a in episode.readable_media if a.kind is MediaKind.ENDOSCOPIC_VIDEO)
    if not video:
        msg = f"episode {episode.id} has no readable endoscopic video to perceive"
        raise DomainInvariantError(msg)
    return video


def episode_duration_s(video: tuple[MediaAsset, ...]) -> float:
    """Total duration of ``video``.

    Raises:
        DomainInvariantError: If any asset lacks a duration. Gates compare
            observations against the point of no return, so an unknown
            timeline makes every timing conclusion unsound.
    """
    if any(asset.duration_seconds is None for asset in video):
        missing = [a.id for a in video if a.duration_seconds is None]
        msg = (
            f"media {', '.join(missing)} has no duration; timing-dependent "
            f"safety gates cannot be evaluated without a known timeline"
        )
        raise DomainInvariantError(msg)
    return sum(a.duration_seconds or 0.0 for a in video)


class AnnotationBackend:
    """A backend whose observations come from an expert panel.

    The annotations are supplied by the caller -- typically loaded from the
    panel's own tooling -- and this class does the binding, gating and
    validation that every backend owes the layers above it.
    """

    def __init__(
        self,
        *,
        phases: tuple[PhaseSegment, ...] = (),
        structures: tuple[StructureSighting, ...] = (),
        proximity_events: tuple[ProximityEvent, ...] = (),
        bleeding_events: tuple[BleedingEvent, ...] = (),
        cvs: tuple[CvsObservation, ...] = (),
        observes: frozenset[ObservationKind] | set[ObservationKind] | None = None,
        version: str = ANNOTATION_BACKEND_VERSION,
    ) -> None:
        """Hold one episode's annotations.

        Args:
            observes: Which observation kinds the annotator actually assessed.
                Required, and deliberately not inferred from which lists are
                non-empty: "looked for bleeding and found none" and "never
                looked at bleeding" are different claims, and only the first
                may clear a gate. Inferring coverage would silently turn the
                second into the first.
        """
        if observes is None:
            msg = (
                "an annotation backend must declare which observation kinds it "
                "assessed; absence of an observation only means 'none found' for "
                "declared kinds (PLAN.md section 7.1)"
            )
            raise DomainInvariantError(msg)
        self._phases = phases
        self._structures = structures
        self._proximity = proximity_events
        self._bleeding = bleeding_events
        self._cvs = cvs
        self._observes = frozenset(observes)
        self._version = version

    @property
    def identity(self) -> str:
        """Stable ``name@version``."""
        return f"{ANNOTATION_BACKEND}@{self._version}"

    def analyse(self, episode: Episode) -> PerceptionResult:
        """Bind the held annotations to ``episode``'s attested media.

        Raises:
            DeidentificationBoundaryError: If the episode is not cleared.
            DomainInvariantError: If media is missing a duration, or an
                annotation falls outside the recording.
        """
        video = readable_video(episode)
        duration_s = episode_duration_s(video)
        self._check_within(duration_s)
        attestations = tuple(
            a.deid_attestation_sha256 for a in video if a.deid_attestation_sha256 is not None
        )
        return PerceptionResult(
            backend=ANNOTATION_BACKEND,
            backend_version=self._version,
            media_sha256=tuple(a.sha256 for a in video),
            deid_attestation_sha256=attestations,
            duration_s=duration_s,
            observes=self._observes,
            phases=self._phases,
            structures=self._structures,
            proximity_events=self._proximity,
            bleeding_events=self._bleeding,
            cvs=self._cvs,
        )

    def _check_within(self, duration_s: float) -> None:
        """Reject annotations that fall outside the recording.

        An annotation past the end of the video means the annotator was
        working from different material, or the timeline is misaligned. Either
        way, a gate reasoning about the resulting timings would be reasoning
        about nothing.
        """
        tolerance = 1e-6
        for segment in self._phases:
            if segment.end_s > duration_s + tolerance:
                msg = (
                    f"phase {segment.phase.value} ends at {segment.end_s}s but the "
                    f"recording is {duration_s}s; annotations do not match this media"
                )
                raise DomainInvariantError(msg)
        for sighting in self._structures:
            if sighting.end_s > duration_s + tolerance:
                msg = (
                    f"sighting of {sighting.structure.value} ends at {sighting.end_s}s "
                    f"but the recording is {duration_s}s"
                )
                raise DomainInvariantError(msg)
        for bleed in self._bleeding:
            if bleed.end_s > duration_s + tolerance:
                msg = f"bleeding event ends at {bleed.end_s}s but the recording is {duration_s}s"
                raise DomainInvariantError(msg)
        for approach in self._proximity:
            if approach.at_s > duration_s + tolerance:
                msg = f"proximity event at {approach.at_s}s but the recording is {duration_s}s"
                raise DomainInvariantError(msg)
        for obs in self._cvs:
            if obs.at_s is not None and obs.at_s > duration_s + tolerance:
                msg = (
                    f"criterion {obs.criterion.value} recorded at {obs.at_s}s but the "
                    f"recording is {duration_s}s"
                )
                raise DomainInvariantError(msg)
