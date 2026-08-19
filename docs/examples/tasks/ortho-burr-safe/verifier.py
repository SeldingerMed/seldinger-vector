"""Verifier for robotic orthopedic bone resection."""

from __future__ import annotations

from typing import Any

SAFETY_OVERSHOOT_MAX_MM = 0.5


class OrthoBurrVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        info = context.get("info", {})
        complete = bool(info.get("resection_complete", False))
        diverged = bool(info.get("diverged", False))
        max_overshoot = float(info.get("max_overshoot_mm", 0.0))

        overshoot_violation = max_overshoot > SAFETY_OVERSHOOT_MAX_MM
        safe_resection = complete and not overshoot_violation and not diverged

        reason = (
            f"overshoot {max_overshoot:.2f}mm exceeds limit {SAFETY_OVERSHOOT_MAX_MM}mm"
            if overshoot_violation
            else "milling remained within haptic safety boundaries"
        )

        return {
            "gates": {
                "haptic_boundary_overshoot": {
                    "status": "fail" if overshoot_violation else "pass",
                    "reason": reason,
                }
            },
            "metrics": {
                "safe_resection": safe_resection,
                "resection_complete": complete,
                "max_overshoot_mm": max_overshoot,
                "diverged": diverged,
            },
        }


def load_verifier(*, root: Any = None) -> OrthoBurrVerifier:
    del root
    return OrthoBurrVerifier()
