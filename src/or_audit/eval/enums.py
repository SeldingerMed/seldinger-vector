"""Closed vocabularies for eval tasks.

Harbor gets by with a Dockerfile and a float. Procedural medical evals have to
name the world, the oracle, the subject, and the PHI class up front or the
runner will silently do the wrong thing. These enums are the fields Harbor
does not have; BUILD.md section 1.3 treats them as non-negotiable.
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


class WorldKind(StrEnum):
    """What Harbor would call the environment — the world, not the container."""

    LUMEN_GYM = "lumen-gym"
    LUMEN_REPLAY = "lumen-replay"
    ANGIOSTRESS_CONTRACT = "angiostress-contract"
    FRAME_SOURCE = "frame-source"


class OracleKind(StrEnum):
    """How ground truth is obtained."""

    PHYSICS = "physics"
    CONTRACT = "contract"
    PANEL = "panel"
    SCRIPT = "script"


class AgentKind(StrEnum):
    """What produces actions or predictions."""

    POLICY = "policy"
    FROZEN_MODEL = "frozen-model"
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
