"""Closed vocabularies.

These enums encode commitments made in PLAN.md. Two are load-bearing and
should not be "simplified" later without revisiting the plan:

* :class:`Determination` includes ``INDETERMINATE`` because section 7.2 makes
  abstention a required output class. A scorer that cannot abstain gets forced
  into false confidence exactly where liability concentrates.
* :class:`GateStatus` includes ``NOT_ASSESSABLE`` for the same reason at the
  safety layer: unreadable video must never be recorded as a pass.
"""

from __future__ import annotations

from enum import StrEnum


class RobotPlatform(StrEnum):
    """Robotic platform a case was performed on.

    Platform is *data*, never a branch in scoring logic. PLAN.md section 1
    sells cross-platform comparability, so per-vendor behaviour belongs in
    ingestion adapters only.
    """

    DA_VINCI_SI = "da_vinci_si"
    DA_VINCI_X = "da_vinci_x"
    DA_VINCI_XI = "da_vinci_xi"
    DA_VINCI_5 = "da_vinci_5"
    HUGO = "hugo"
    VERSIUS = "versius"
    VICARIOUS = "vicarious"
    MOON = "moon"
    DISTALMOTION = "distalmotion"
    SIMULATOR = "simulator"
    OTHER = "other"


class MediaKind(StrEnum):
    """Kind of media attached to an episode."""

    #: In-body endoscopic video. The required common denominator (section 7).
    ENDOSCOPIC_VIDEO = "endoscopic_video"
    #: Room-facing video. High PHI risk; rarely ingested.
    ROOM_VIDEO = "room_video"
    #: Intraoperative audio. Discarded by default (section 8).
    AUDIO = "audio"
    #: Instrument kinematics/telemetry. Optional enrichment, never required.
    KINEMATICS = "kinematics"


class DeidStatus(StrEnum):
    """De-identification state of a media asset or episode.

    Transitions are one-way: ``RAW -> IN_PROGRESS -> ATTESTED``, or to
    ``FAILED``, or to ``DISCARDED``. Only ``ATTESTED`` media may be read by
    perception, scoring, reporting, or export.
    """

    RAW = "raw"
    IN_PROGRESS = "in_progress"
    ATTESTED = "attested"
    FAILED = "failed"
    #: Terminal: the asset was deliberately destroyed rather than
    #: de-identified. This is the *default* disposition for intraoperative
    #: audio (PLAN.md section 8). A discarded asset is not readable, but it
    #: also does not block the episode -- there is nothing left to leak.
    #: Distinguishing this from ``FAILED`` matters: one is a completed policy
    #: decision, the other is unfinished work.
    DISCARDED = "discarded"


class SkillBand(StrEnum):
    """Experience band of the surgeon at the time of the episode.

    Required for the stratified metrics in PLAN.md section 13: the headline
    agreement figure is computed within a single band, because novice-versus-
    expert separation is inflated by between-group variance and is close to
    trivial.
    """

    NOVICE = "novice"
    RESIDENT = "resident"
    FELLOW = "fellow"
    ATTENDING = "attending"
    EXPERT = "expert"


class GateStatus(StrEnum):
    """Outcome of a deterministic hard safety gate (section 7.1)."""

    PASS = "pass"
    FAIL = "fail"
    #: Evidence insufficient to decide. Never treat as a pass.
    NOT_ASSESSABLE = "not_assessable"
    #: The gate does not govern this procedure, e.g. the Critical View of
    #: Safety outside cholecystectomy. Distinct from ``PASS`` on purpose: a
    #: gate that never ran has not cleared anything, and reporting it as a
    #: pass would inflate an episode's apparent safety coverage.
    NOT_APPLICABLE = "not_applicable"


class Determination(StrEnum):
    """Terminal credentialing determination (section 7.2)."""

    MEETS_BENCHMARK = "meets_benchmark"
    DOES_NOT_MEET = "does_not_meet"
    #: Required output class. The scorer must be able to decline to decide.
    INDETERMINATE = "indeterminate"


class ThresholdOwner(StrEnum):
    """Who set the benchmark threshold behind a determination.

    Section 7.2 requires this to be explicit and carried on the artifact,
    because it allocates responsibility for the threshold being wrong.
    """

    OR_AUDIT = "or_audit"
    CUSTOMER = "customer"
    SPECIALTY_SOCIETY = "specialty_society"


class Jurisdiction(StrEnum):
    """Legal jurisdiction governing an institution's records.

    Carried because peer-review privilege and video-retention rules vary by
    jurisdiction (PLAN.md section 9 and V-3/V-5), so the same artifact is not
    equally holdable everywhere.
    """

    US_FEDERAL = "us_federal"
    US_STATE = "us_state"
    UK = "uk"
    EU = "eu"
    OTHER = "other"
