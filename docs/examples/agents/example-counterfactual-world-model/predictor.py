"""Frozen consequence rankings for the counterfactual harness example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CounterfactualWorldModel:
    def __init__(self, weights_path: Path) -> None:
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        self._predictions = {str(item["id"]): item for item in payload["items"]}

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        return dict(self._predictions[str(item["id"])])


def load_predictor(*, root: Path, weights_path: Path) -> CounterfactualWorldModel:
    del root
    return CounterfactualWorldModel(weights_path)
