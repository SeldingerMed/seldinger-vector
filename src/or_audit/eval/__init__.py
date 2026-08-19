"""OR-Audit v0.3 contracts, package loaders, harnesses, and evidence vectors."""

from __future__ import annotations

from or_audit.eval.adapters import (
    BaseModalityAdapter,
    ModalityAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    require_adapter,
)
from or_audit.eval.adapters import (
    clear_registry as clear_adapter_registry,
)
from or_audit.eval.agent import AgentPackage
from or_audit.eval.bind import assert_bind
from or_audit.eval.cartesian import CartesianManifest, replay_cartesian, run_cartesian_job
from or_audit.eval.contracts import (
    CapabilitySpec,
    GateProjectionPolicy,
    HarnessSpec,
    InteractionMode,
    InterfaceSpec,
    MetricDirection,
    MetricKind,
    PerturbationSpec,
    RuntimeDescriptor,
    RuntimeKind,
    ScenarioSpec,
)
from or_audit.eval.dataset import DatasetSpec, TasksetSpec
from or_audit.eval.enums import (
    AgentKind,
    AttestationLevel,
    GateKind,
    ModalityKind,
    OracleKind,
    PhiClass,
    PortId,
    ProjectionId,
    SubjectKind,
    WorldKind,
)
from or_audit.eval.export_rl import export_rl
from or_audit.eval.job import JobResult, assemble_job_result
from or_audit.eval.job_config import JobConfig, resolve_job
from or_audit.eval.loader import load_agent, load_dataset, load_task, load_taskset
from or_audit.eval.reconstitute import reconstitute_trial_vector
from or_audit.eval.runner import builtin_random_agent, replay_job, run_job
from or_audit.eval.task import GateSpec, MetricSpec, PortSpec, ProjectionSpec, TaskSpec
from or_audit.eval.trace import (
    EvidenceReference,
    FailureEvent,
    HandoffEvent,
    ProceduralTrace,
    RecoveryEvent,
    TimingEvent,
    ToolEvent,
    TraceStep,
)
from or_audit.eval.vector import TrialVector, project

__all__ = [
    "AgentKind",
    "AgentPackage",
    "AttestationLevel",
    "BaseModalityAdapter",
    "CapabilitySpec",
    "CartesianManifest",
    "DatasetSpec",
    "EvidenceReference",
    "FailureEvent",
    "GateKind",
    "GateProjectionPolicy",
    "GateSpec",
    "HandoffEvent",
    "HarnessSpec",
    "InteractionMode",
    "InterfaceSpec",
    "JobConfig",
    "JobResult",
    "MetricDirection",
    "MetricKind",
    "MetricSpec",
    "ModalityAdapter",
    "ModalityKind",
    "OracleKind",
    "PerturbationSpec",
    "PhiClass",
    "PortId",
    "PortSpec",
    "ProceduralTrace",
    "ProjectionId",
    "ProjectionSpec",
    "RecoveryEvent",
    "RuntimeDescriptor",
    "RuntimeKind",
    "ScenarioSpec",
    "SubjectKind",
    "TaskSpec",
    "TasksetSpec",
    "TimingEvent",
    "ToolEvent",
    "TraceStep",
    "TrialVector",
    "WorldKind",
    "assemble_job_result",
    "assert_bind",
    "builtin_random_agent",
    "clear_adapter_registry",
    "export_rl",
    "get_adapter",
    "list_adapters",
    "load_agent",
    "load_dataset",
    "load_task",
    "load_taskset",
    "project",
    "reconstitute_trial_vector",
    "register_adapter",
    "replay_cartesian",
    "replay_job",
    "require_adapter",
    "resolve_job",
    "run_cartesian_job",
    "run_job",
]
