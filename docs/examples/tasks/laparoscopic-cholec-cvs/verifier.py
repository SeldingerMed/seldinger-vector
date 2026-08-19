"""Verifier for laparoscopic cholecystectomy CVS task."""

from __future__ import annotations

from typing import Any


class CholecCvsVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        prediction = context.get("prediction") or context.get("output") or {}
        label = context.get("label", {})

        abstained = bool(prediction.get("abstained", False))
        pred_cvs = prediction.get("cvs_achieved")
        pred_structure = prediction.get("critical_structure", "")

        gt_cvs = label.get("cvs_achieved")
        gt_structure = label.get("critical_structure", "")

        # Critical structure misidentification: misidentifying CBD as cystic duct
        cbd_violation = (
            gt_structure in {"common_bile_duct", "common_hepatic_duct"}
            and pred_structure == "cystic_duct"
        )

        raw_correct = False
        if not abstained and pred_cvs is not None:
            raw_correct = pred_cvs == gt_cvs and (
                pred_structure == gt_structure if pred_structure else True
            )

        safe_correct = raw_correct and not cbd_violation

        return {
            "gates": {
                "critical_structure_misid": {
                    "status": "fail" if cbd_violation else "pass",
                    "reason": "misidentified common bile/hepatic duct"
                    if cbd_violation
                    else "no critical structure misidentification",
                }
            },
            "metrics": {
                "cvs_safe_accuracy": safe_correct,
                "raw_accuracy": raw_correct,
                "abstained": abstained,
            },
            "signals": {
                "cbd_violation": cbd_violation,
            },
        }


def load_verifier(*, root: Any = None) -> CholecCvsVerifier:
    del root
    return CholecCvsVerifier()
