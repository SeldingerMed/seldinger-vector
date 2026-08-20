"""Validate the pinned AngioStress release-audit result."""

from __future__ import annotations

import math
from typing import Any


class ReleaseAuditVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        label = context["label"]
        prediction = context["prediction"]
        numeric = [
            prediction.get("dias_prediction_count"),
            prediction.get("cathaction_prediction_count"),
            prediction.get("sam_vit_b_mean_dice"),
            prediction.get("sam_vit_l_mean_dice"),
            prediction.get("medsam_vit_b_mean_dice"),
        ]
        finite = all(
            isinstance(value, int | float) and math.isfinite(float(value)) for value in numeric
        )
        passed = bool(prediction.get("release_audit_passed")) and bool(
            label.get("release_audit_passed")
        )
        return {
            "gates": {
                "finite_metric_check": {
                    "status": "pass" if finite else "fail",
                    "reason": "release metrics finite" if finite else "non-finite release metric",
                }
            },
            "metrics": {
                "release_audit_passed": passed,
                "dias_prediction_count": numeric[0],
                "cathaction_prediction_count": numeric[1],
                "sam_vit_b_mean_dice": numeric[2],
                "sam_vit_l_mean_dice": numeric[3],
                "medsam_vit_b_mean_dice": numeric[4],
            },
        }


def load_verifier(*, root: Any) -> ReleaseAuditVerifier:
    del root
    return ReleaseAuditVerifier()
