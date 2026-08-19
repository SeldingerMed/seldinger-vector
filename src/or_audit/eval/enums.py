"""Closed kernel vocabularies and compatibility names for eval packages.

Interaction mode, runtime adapter, PHI class, and world adapter select kernel
behavior and remain closed. Interface ids and agent kinds are package-authored
slugs; ``PortId`` and ``AgentKind`` only preserve well-known v0.2 names.
"""

from __future__ import annotations

from enum import StrEnum


class SubjectKind(StrEnum):
    """Who or what a trial is about."""

    POLICY = "policy"
    MODEL = "model"
    #: Named humans. The loader refuses determinations in this mode until
    #: PLAN.md Phase 0 clears; the kind exists so the refusal is explicit.
    HUMAN = "human"


class ModalityKind(StrEnum):
    """Procedural modality categories supported by SurgEval/Vector."""

    VIDEO_LAPAROSCOPIC = "video-laparoscopic"
    VIDEO_ENDOSCOPIC = "video-endoscopic"
    AIRWAY_BRONCHOSCOPY = "airway-bronchoscopy"
    FLUOROSCOPY_DSA = "fluoroscopy-dsa"
    ORTHOPEDIC_POINTCLOUD = "orthopedic-pointcloud"
    ROBOTIC_KINEMATICS = "robotic-kinematics"
    ENDOVASCULAR_SIM = "endovascular-sim"
    SYNTHETIC_PROCEDURAL = "synthetic-procedural"


class GateKind(StrEnum):
    """Typology of inviolable physical and procedural safety gates."""

    SPATIAL_EXCLUSION = "spatial-exclusion"
    FORCE_THRESHOLD = "force-threshold"
    PERFORATION_RISK = "perforation-risk"
    RADIATION_DOSE = "radiation-dose"
    TEMPORAL_BOUND = "temporal-bound"
    CUSTOM = "custom"


class PhiClass(StrEnum):
    """Isolation class of the world's data."""

    #: Procedural / synthetic geometry. Lumen committed assets.
    PROCEDURAL = "procedural"
    #: Public research datasets with their own licenses (Cholec80, DIAS).
    PUBLIC = "public"
    #: Clinical video that has cleared de-identification. Not in the wedge.
    DEIDENTIFIED_CLINICAL = "deidentified_clinical"
    #: Must not be loaded at all.
    PROHIBITED = "prohibited"


class PortId(StrEnum):
    """Well-known v0.2 port names retained for compatibility.

    Canonical v0.3 tasks use open interface slugs plus explicit interaction,
    protocol, and schema requirements. New interfaces do not require an enum
    member.
    """

    GYM_POLICY = "gym-policy"
    VIDEO_PREDICT = "video-predict"


class WorldKind(StrEnum):
    """How we host a world — Harbor's Docker/Daytona/Modal, not anatomy.

    ``lumen-gym`` is the first physics adapter, not the only one. A
    third-party gym is ``gym``. Uploaded video with labels is
    ``frame-source``. None of these names a procedure.
    """

    LUMEN_GYM = "lumen-gym"
    LUMEN_REPLAY = "lumen-replay"
    #: Any Gymnasium env that is not Lumen. The gym_id is the task author's.
    GYM = "gym"
    ANGIOSTRESS_CONTRACT = "angiostress-contract"
    FRAME_SOURCE = "frame-source"
    COUNTERFACTUAL = "counterfactual"
    SOFA = "sofa"
    WARP = "warp"
    ISAAC_LAB = "isaac-lab"
    PYBULLET = "pybullet"
    VIDEO_STREAM = "video-stream"
    CT_AIRWAY = "ct-airway"


class OracleKind(StrEnum):
    """How ground truth is obtained."""

    PHYSICS = "physics"
    CONTRACT = "contract"
    PANEL = "panel"
    SCRIPT = "script"


class AgentKind(StrEnum):
    """Well-known agent-kind names; canonical package fields accept any slug."""

    POLICY = "policy"
    FROZEN_MODEL = "frozen-model"
    WORLD_MODEL = "world-model"
    VLM = "vlm"
    PANEL = "panel"
    RANDOM = "random"


class AttestationLevel(StrEnum):
    """Whether this task mints a de-identification attestation."""

    NONE = "none"
    ANALYSIS_ONLY = "analysis_only"
    ATTESTED = "attested"


class ProjectionId(StrEnum):
    """Closed set of scalar projections for RL.

    A string of Python is not a projection: it would be ``eval`` with extra
    steps, and it would let a training loop invent a collapse the leaderboard
    never sees. Add variants here, with tests, when a new collapse is needed.
    """

    #: 0 if any hard gate failed or the episode diverged; else 1 iff raw success.
    GATED_REACH_V0 = "gated_reach_v0"
