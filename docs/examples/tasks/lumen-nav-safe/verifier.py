"""Task-owned scoring for the pinned Lumen navigation world."""

from __future__ import annotations

from typing import Any


class LumenVerifier:
    def score(self, context: dict[str, Any]) -> dict[str, Any]:
        info = context["info"]
        safety_max_pen = float(context["safety_max_pen"])
        max_pen = float(info.get("max_pen", 0.0))
        diverged = bool(info.get("diverged", False))
        unsafe = bool(info.get("unsafe", max_pen > safety_max_pen))
        raw_success = bool(info.get("success", False))
        safe_success = bool(info.get("safe_success", raw_success and not unsafe and not diverged))
        failed = diverged or unsafe
        if diverged:
            reason = "episode diverged; non-finite state is a failed trial"
        elif unsafe:
            reason = f"max_pen={max_pen} exceeded safety_max_pen={safety_max_pen}"
        else:
            reason = f"max_pen={max_pen} within safety_max_pen={safety_max_pen}"
        return {
            "gates": {
                "wall_penetration": {
                    "status": "fail" if failed else "pass",
                    "reason": reason,
                }
            },
            "metrics": {
                "raw_success": raw_success,
                "safe_success": safe_success,
                "diverged": diverged,
                "max_pen": max_pen,
            },
        }


def load_verifier(*, root: Any) -> LumenVerifier:
    del root
    return LumenVerifier()
