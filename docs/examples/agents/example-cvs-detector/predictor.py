"""Frozen CVS detector predictions behind the video-predict runtime contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CvsDetector:
    def __init__(self, weights_path: Path) -> None:
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        self._predictions = {str(item["id"]): item for item in payload["items"]}

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        obs = item.get("laparoscopic-video") or item
        clip_id = str(obs.get("clip_id") or obs.get("id") or "")
        return dict(self._predictions[clip_id])


def load_predictor(*, root: Path, weights_path: Path) -> CvsDetector:
    del root
    return CvsDetector(weights_path)
