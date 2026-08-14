"""Run the pinned AngioStress full release audit as a frozen-model evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


class AngioStressPredictor:
    def __init__(self, weights_path: Path) -> None:
        self._weights = json.loads(weights_path.read_text(encoding="utf-8"))

    def _source(self) -> Path:
        pin = str(self._weights["source_pin"])
        target = Path.home() / ".cache" / "or-audit" / "sources" / pin
        if not target.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    str(self._weights["source_url"]),
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            subprocess.run(
                ["git", "-C", str(target), "fetch", "--depth", "1", "origin", pin],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", pin],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        actual = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if actual != pin:
            raise RuntimeError(f"AngioStress source pin mismatch: expected {pin}, got {actual}")
        return target

    @staticmethod
    def _file_count(path: Path) -> int:
        return sum(1 for candidate in path.rglob("*") if candidate.is_file())

    @staticmethod
    def _sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _download(self, relative: str, expected_sha256: str) -> Path:
        cache = Path.home() / ".cache" / "or-audit" / "angiostress-archives"
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / Path(relative).name
        if target.is_file() and self._sha256(target) == expected_sha256:
            return target
        existing = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(
            str(self._weights["artifact_base_url"]) + relative,
            headers={"Range": f"bytes={existing}-"} if existing else {},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            append = existing > 0 and getattr(response, "status", 200) == 206
            with target.open("ab" if append else "wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        actual = self._sha256(target)
        if actual != expected_sha256:
            raise RuntimeError(
                f"AngioStress archive digest mismatch for {relative}: "
                f"expected {expected_sha256}, got {actual}"
            )
        return target

    @staticmethod
    def _extract(archive_path: Path, root: Path) -> None:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                target = (root / member.name).resolve()
                try:
                    target.relative_to(root.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"unsafe archive member {member.name!r}") from exc
                if member.issym() or member.islnk():
                    raise RuntimeError(f"archive links are refused: {member.name!r}")
            archive.extractall(root)

    def _restore_artifacts(self, source: Path) -> None:
        for spec in self._weights["archives"]:
            target = source / str(spec["target"])
            if self._file_count(target) == int(spec["files"]):
                continue
            archive = self._download(str(spec["path"]), str(spec["sha256"]))
            self._extract(archive, source)
            actual = self._file_count(target)
            if actual != int(spec["files"]):
                raise RuntimeError(
                    f"restored {target} has {actual} files; expected {spec['files']}"
                )


    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        del item
        source = self._source()
        self._restore_artifacts(source)
        with tempfile.TemporaryDirectory(prefix="or-audit-angiostress-") as raw_out:
            output = Path(raw_out)
            subprocess.run(
                [
                    sys.executable,
                    str(source / "benchmark" / "run_release_audit.py"),
                    "--root",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            audit = json.loads((output / "release_audit.json").read_text(encoding="utf-8"))
        metrics = audit["metrics_summary"]
        cathaction = audit["surfaces"]["cathaction_human_segmentation"]["key_metrics"]
        return {
            "release_audit_passed": bool(audit["release_audit_passed"]),
            "finite_metric_check": all(
                isinstance(value, int | float) for value in metrics.values()
            ),
            "dias_prediction_count": float(metrics["release_dias_prediction_count"]),
            "cathaction_prediction_count": float(
                metrics["release_cathaction_prediction_count"]
            ),
            "sam_vit_b_mean_dice": float(cathaction["sam_vit_b_mean_dice"]),
            "sam_vit_l_mean_dice": float(cathaction["sam_vit_l_mean_dice"]),
            "medsam_vit_b_mean_dice": float(cathaction["medsam_vit_b_mean_dice"]),
        }


def load_predictor(*, root: Path, weights_path: Path) -> AngioStressPredictor:
    del root
    return AngioStressPredictor(weights_path)
