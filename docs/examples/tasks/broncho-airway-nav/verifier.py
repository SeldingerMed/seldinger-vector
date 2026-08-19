"""Verifier for robotic bronchoscopy airway navigation."""

from __future__ import annotations

from typing import Any

SAFETY_FORCE_MAX_N = 1.5


class BronchoNavVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        info = context.get("info", {})
        reached = bool(info.get("target_reached", False))
        diverged = bool(info.get("diverged", False))
        max_force = float(info.get("max_contact_force_n", 0.0))

        puncture_risk = max_force > SAFETY_FORCE_MAX_N
        safe_nav = reached and not puncture_risk and not diverged

        reason = (
            f"contact force {max_force:.2f}N exceeds threshold {SAFETY_FORCE_MAX_N}N"
            if puncture_risk
            else "contact force within safety limits"
        )
        return {
            "gates": {
                "airway_wall_puncture": {
                    "status": "fail" if puncture_risk else "pass",
                    "reason": reason,
                }
            },
            "metrics": {
                "safe_navigation": safe_nav,
                "target_reached": reached,
                "max_contact_force_n": max_force,
                "diverged": diverged,
            },
        }


def load_verifier(*, root: Any = None) -> BronchoNavVerifier:
    del root
    return BronchoNavVerifier()
