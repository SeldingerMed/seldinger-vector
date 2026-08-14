"""Small pinned linear controller used to exercise the real policy port."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LinearPolicy:
    def __init__(self, weights_path: Path) -> None:
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        self._weights = np.asarray(payload["weights"], dtype=np.float64)
        self._bias = np.asarray(payload["bias"], dtype=np.float64)

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, observation: Any, *, step: int) -> Any:
        del step
        obs = np.asarray(observation, dtype=np.float64)
        return np.clip(self._weights @ obs + self._bias, -1.0, 1.0)


def load_policy(*, root: Path, weights_path: Path) -> LinearPolicy:
    del root
    return LinearPolicy(weights_path)
