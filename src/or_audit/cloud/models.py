"""Public contracts for the minimal Vector Cloud control plane."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutorKind(StrEnum):
    LOCAL = "local"
    RUNPOD = "runpod"


class ComputeClass(StrEnum):
    CPU = "cpu"
    L4 = "l4"
    L40S = "l40s"
    A100 = "a100-80gb"
    H100 = "h100-pcie"


class DataClassification(StrEnum):
    PUBLIC = "public"
    DEIDENTIFIED = "deidentified"
    CONFIDENTIAL = "confidential"


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRequest(BaseModel):
    """One task-agent evaluation request.

    PHI is intentionally not a valid classification for the hosted beta.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    name: str = Field(default="", max_length=120)
    n: int = Field(default=1, ge=1, le=10_000)
    executor: ExecutorKind = ExecutorKind.LOCAL
    compute: ComputeClass = ComputeClass.CPU
    data_classification: DataClassification = DataClassification.DEIDENTIFIED
    registry: str = ""

    @model_validator(mode="after")
    def validate_executor(self) -> Self:
        if self.executor is ExecutorKind.RUNPOD:
            if self.data_classification is DataClassification.CONFIDENTIAL:
                raise ValueError("hosted RunPod beta accepts public or deidentified data only")
            if self.compute is ComputeClass.CPU:
                raise ValueError("RunPod jobs require a GPU compute class")
            if "@" not in self.task or "@" not in self.agent:
                raise ValueError("RunPod jobs require versioned registry task and agent references")
        return self


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    created_at: datetime
    updated_at: datetime
    status: JobStatus
    request: JobRequest
    provider_id: str = ""
    artifact_path: str = ""
    result_head: str = ""
    error: str = ""

    @classmethod
    def new(cls, request: JobRequest) -> JobRecord:
        now = datetime.now(UTC)
        return cls(
            id=uuid4().hex,
            created_at=now,
            updated_at=now,
            status=JobStatus.QUEUED,
            request=request,
        )
