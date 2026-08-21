"""Run one cloud job inside a worker image and return its evidence bundle."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def run_from_env() -> int:
    required = {
        name: os.environ.get(name, "")
        for name in (
            "VECTOR_CLOUD_CALLBACK_URL",
            "VECTOR_CLOUD_CALLBACK_TOKEN",
            "VECTOR_CLOUD_JOB_ID",
            "VECTOR_CLOUD_TASK",
            "VECTOR_CLOUD_AGENT",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"missing worker environment: {', '.join(missing)}", file=sys.stderr)
        return 2
    callback = required["VECTOR_CLOUD_CALLBACK_URL"].rstrip("/")
    token = required["VECTOR_CLOUD_CALLBACK_TOKEN"]
    job_id = required["VECTOR_CLOUD_JOB_ID"]
    task = required["VECTOR_CLOUD_TASK"]
    task_flag = "-s" if "@" in task and not Path(task).exists() else "-t"
    with tempfile.TemporaryDirectory(prefix="vector-worker-") as tmp:
        result_dir = Path(tmp) / "result"
        command = [
            sys.executable,
            "-m",
            "or_audit.cli",
            "run",
            task_flag,
            task,
            "-a",
            required["VECTOR_CLOUD_AGENT"],
            "-n",
            os.environ.get("VECTOR_CLOUD_N", "1"),
            "--out",
            str(result_dir),
        ]
        registry = os.environ.get("VECTOR_CLOUD_REGISTRY", "")
        if registry:
            command.extend(("--registry", registry))
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            _post_json(
                f"{callback}/v1/internal/jobs/{job_id}/fail",
                token,
                {"error": error[-4000:] or f"surgeval exited {completed.returncode}"},
            )
            return completed.returncode or 1
        result_path = result_dir / "result.json"
        if not result_path.is_file():
            result_paths = tuple(result_dir.glob("*/result.json"))
            if len(result_paths) != 1:
                error = f"worker expected one task result, found {len(result_paths)}"
                _post_json(
                    f"{callback}/v1/internal/jobs/{job_id}/fail",
                    token,
                    {"error": error},
                )
                return 1
            result_path = result_paths[0]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        head = str(result.get("head", ""))
        payload = _archive(result_path.parent)
        _post_archive(
            f"{callback}/v1/internal/jobs/{job_id}/complete",
            token,
            payload,
            head,
        )
    return 0


def _archive(result_dir: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(result_dir, arcname="result")
    payload = output.getvalue()
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise RuntimeError("evidence archive exceeds 100 MiB worker callback limit")
    return payload


def _post_json(url: str, token: str, body: dict[str, str]) -> None:
    _post(
        Request(
            url,
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
    )


def _post_archive(url: str, token: str, payload: bytes, head: str) -> None:
    _post(
        Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/gzip",
                "X-Vector-Result-Head": head,
            },
        )
    )


def _post(request: Request) -> None:
    try:
        with urlopen(request, timeout=120) as response:
            if response.status not in {200, 204}:
                raise RuntimeError(f"callback returned HTTP {response.status}")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"worker callback failed: {exc}") from exc
