"""Eval harness: Harbor-shaped tasks, datasets, agents, and trial vectors.

See ``docs/BUILD.md``. This package is P0 — types and loading. It does not
talk to Lumen or run a policy.
"""

from __future__ import annotations

from or_audit.eval.agent import AgentPackage
from or_audit.eval.bind import assert_bind
from or_audit.eval.dataset import DatasetSpec
from or_audit.eval.enums import (
    AgentKind,
    AttestationLevel,
    OracleKind,
    PhiClass,
    PortId,
    ProjectionId,
    SubjectKind,
    WorldKind,
)
from or_audit.eval.loader import load_agent, load_dataset, load_task
from or_audit.eval.task import PortSpec, ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector, project, vector_from_lumen_info

__all__ = [
    "AgentKind",
    "AgentPackage",
    "AttestationLevel",
    "DatasetSpec",
    "OracleKind",
    "PhiClass",
    "PortId",
    "PortSpec",
    "ProjectionId",
    "ProjectionSpec",
    "SubjectKind",
    "TaskSpec",
    "TrialVector",
    "WorldKind",
    "assert_bind",
    "load_agent",
    "load_dataset",
    "load_task",
    "project",
    "vector_from_lumen_info",
]
