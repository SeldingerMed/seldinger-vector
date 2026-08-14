"""Eval harness: Harbor-shaped tasks, datasets, agents, runners, and trial vectors.

See ``docs/BUILD.md``. P0 is the contract. P1/P2 are ``run_job`` (gym-policy
and video-predict). Lumen is optional at import time.
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
from or_audit.eval.job import JobResult, assemble_job_result
from or_audit.eval.loader import load_agent, load_dataset, load_task
from or_audit.eval.runner import builtin_random_agent, replay_job, run_job
from or_audit.eval.task import PortSpec, ProjectionSpec, TaskSpec
from or_audit.eval.vector import TrialVector, project, vector_from_lumen_info

__all__ = [
    "AgentKind",
    "AgentPackage",
    "AttestationLevel",
    "DatasetSpec",
    "JobResult",
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
    "assemble_job_result",
    "assert_bind",
    "builtin_random_agent",
    "load_agent",
    "load_dataset",
    "load_task",
    "project",
    "replay_job",
    "run_job",
    "vector_from_lumen_info",
]
