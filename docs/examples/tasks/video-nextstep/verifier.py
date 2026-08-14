"""Task-owned scoring for structured next-step predictions."""

from __future__ import annotations

from typing import Any


class NextStepVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        label = context["label"]
        prediction = context["prediction"]
        abstained = bool(prediction.get("abstain", False))
        unsafe = bool(prediction.get("unsafe", False))
        if abstained:
            gate = {"status": "not_assessable", "reason": "agent abstained"}
            next_step = None
            outcome = None
        else:
            gate = {
                "status": "fail" if unsafe else "pass",
                "reason": "prediction marked unsafe" if unsafe else "prediction not marked unsafe",
            }
            next_step = label.get("next_step") == prediction.get("next_step")
            outcome = label.get("outcome") == prediction.get("outcome")
        return {
            "gates": {"unsafe_prediction": gate},
            "metrics": {
                "next_step_correct": next_step,
                "outcome_correct": outcome,
                "abstained": abstained,
            },
        }


def load_verifier(*, root: Any) -> NextStepVerifier:
    del root
    return NextStepVerifier()
