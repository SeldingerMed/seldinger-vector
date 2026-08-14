"""Eval harness: Harbor-shaped tasks, datasets, and trial vectors.

See ``docs/BUILD.md``. This package is P0 — types and loading. It does not
talk to Lumen or run a policy.
"""

from __future__ import annotations

from or_audit.eval.dataset import DatasetSpec
from or_audit.eval.enums import (
    AgentKind,
    AttestationLevel,
    OracleKind,
    PhiClass,
    ProcedureFamily,
    ProjectionId,
    SubjectKind,
    WorldKind,
)
from or_audit.eval.loader import load_dataset, load_task
from or_audit.eval.task import ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector, project, vector_from_lumen_info

__all__ = [
    "AgentKind",
    "AttestationLevel",
    "DatasetSpec",
    "OracleKind",
    "PhiClass",
    "ProcedureFamily",
    "ProjectionId",
    "ProjectionSpec",
    "SubjectKind",
    "TaskSpec",
    "TrialVector",
    "WorldKind",
    "load_dataset",
    "load_task",
    "project",
    "vector_from_lumen_info",
]
