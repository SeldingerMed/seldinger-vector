"""video-predict scoring: labels the task author brought vs agent JSON.

The kernel does not know CABG from cath. Field names come from the task.
AngioStress is this adapter with a claim footer and a contract JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from or_audit.errors import TaskContractError


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        msg = f"missing {path.name}: {path}"
        raise TaskContractError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path} must be a JSON object"
        raise TaskContractError(msg)
    return data


def load_items(path: Path) -> tuple[dict[str, Any], ...]:
    """Load ``{"items": [...]}`` from a labels or predictions file."""
    data = load_json_object(path)
    raw = data.get("items")
    if not isinstance(raw, list) or not raw:
        msg = f"{path} must contain a non-empty items array"
        raise TaskContractError(msg)
    items: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or "id" not in entry:
            msg = f"{path} items must be objects with an id"
            raise TaskContractError(msg)
        items.append(entry)
    return tuple(items)


def load_claim_footer(contract_path: Path) -> str:
    """Copy the contract's claim boundary. Empty is invalid for AngioStress."""
    data = load_json_object(contract_path)
    footer = data.get("claim_boundary")
    if not isinstance(footer, str) or not footer.strip():
        msg = (
            f"{contract_path} is missing claim_boundary; an AngioStress-shaped "
            f"result without a claim footer is not a scorecard"
        )
        raise TaskContractError(msg)
    return footer.strip()


def index_items(items: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    """Map item id -> object. Duplicate ids are a contract error."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item["id"])
        if item_id in out:
            msg = f"duplicate item id {item_id!r}"
            raise TaskContractError(msg)
        out[item_id] = item
    return out
