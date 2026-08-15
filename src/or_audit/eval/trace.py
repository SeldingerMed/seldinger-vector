"""Typed procedural traces shared by every v0.3 interaction mode."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from or_audit.eval.contracts import InteractionMode, PerturbationSpec, ScenarioSpec


class FailureEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    detected: bool
    severity: str = ""
    detail: str = ""


class RecoveryEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempted: bool
    successful: bool | None = None
    safely_abandoned: bool = False
    detail: str = ""


class HandoffEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    reason: str
    accepted: bool | None = None


class ToolEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class TimingEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    started_ms: float | None = None
    duration_ms: float | None = None
    deadline_ms: float | None = None


class EvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    uri: str
    digest: str = ""
    media_type: str = ""


def _enrich_info(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("info")
    if not isinstance(info, dict):
        return payload
    if "safety" not in payload:
        payload["safety"] = {
            key: value
            for key, value in info.items()
            if key in {"unsafe", "max_pen", "safe_success", "diverged"}
        }
    for event_name in ("evidence", "failure", "recovery", "handoff", "tool", "timing"):
        if event_name in info and event_name not in payload:
            payload[event_name] = info[event_name]
    if "uncertainty" in info and "uncertainty" not in payload:
        payload["uncertainty"] = info["uncertainty"]
    if "abstained" in info and "abstained" not in payload:
        payload["abstained"] = info["abstained"]
    return payload


class TraceStep(BaseModel):
    """One typed transition; unknown legacy evidence is retained as extra fields."""

    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    index: int = Field(ge=0)
    interaction_mode: InteractionMode
    observation: Any = Field(default=None, alias="obs")
    output: Any = None
    action: Any = None
    transition: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool | None = None
    scenario: ScenarioSpec | None = None
    perturbations: tuple[PerturbationSpec, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()
    failure: FailureEvent | None = None
    recovery: RecoveryEvent | None = None
    handoff: HandoffEvent | None = None
    tool: ToolEvent | None = None
    timing: TimingEvent | None = None


class ProceduralTrace(RootModel[tuple[TraceStep, ...]]):
    """Sequence-compatible typed trace that preserves the v0.2 JSON list shape."""

    @model_validator(mode="before")
    @classmethod
    def _accept_steps_object(cls, value: Any) -> Any:
        steps = value["steps"] if isinstance(value, dict) and "steps" in value else value
        if not isinstance(steps, list | tuple):
            return steps
        normalized: list[Any] = []
        for index, raw in enumerate(steps):
            if isinstance(raw, TraceStep):
                normalized.append(raw)
                continue
            if not isinstance(raw, dict):
                normalized.append(raw)
                continue
            payload = dict(raw)
            payload.setdefault("index", index)
            if "interaction_mode" not in payload:
                if "info" in payload and "action" in payload:
                    mode = InteractionMode.CLOSED_LOOP
                elif payload.get("kind") == "counterfactual":
                    mode = InteractionMode.COUNTERFACTUAL
                else:
                    mode = InteractionMode.SINGLE_TURN
                payload["interaction_mode"] = mode
            payload = _enrich_info(payload)
            normalized.append(payload)
        return normalized

    @classmethod
    def from_steps(
        cls,
        steps: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        *,
        mode: InteractionMode,
    ) -> ProceduralTrace:
        typed = []
        for index, raw in enumerate(steps):
            payload = dict(raw)
            payload.setdefault("index", index)
            payload.setdefault("interaction_mode", mode)
            payload = _enrich_info(payload)
            typed.append(TraceStep.model_validate(payload))
        return cls(tuple(typed))

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        for step in self.root:
            yield step.model_dump(mode="json", by_alias=True, exclude_none=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.root[index].model_dump(mode="json", by_alias=True, exclude_none=True)
