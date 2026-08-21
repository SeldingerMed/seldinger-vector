"""Execution backends for the minimal Vector Cloud control plane."""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from or_audit.errors import TaskContractError

from .models import ComputeClass, JobRecord, JobStatus
from .store import JobStore


class Executor(Protocol):
    def submit(self, job: JobRecord) -> None: ...

    def cancel(self, job: JobRecord) -> None: ...
    def release(self, job: JobRecord) -> None: ...

    def reconcile(self, job: JobRecord) -> JobRecord: ...


class LocalExecutor:
    """Run the public SurgEval CLI in a background subprocess."""

    def __init__(self, store: JobStore, *, root: Path, package_root: Path) -> None:
        self.store = store
        self.root = root
        self.package_root = package_root
        self._processes: dict[str, subprocess.Popen[str] | None] = {}
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def submit(self, job: JobRecord) -> None:
        with self._lock:
            self._processes[job.id] = None
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def cancel(self, job: JobRecord) -> None:
        with self._lock:
            self.store.transition(
                job.id,
                expected=(JobStatus.QUEUED, JobStatus.RUNNING),
                status=JobStatus.CANCELLED,
                error="cancelled by user",
            )
            process = self._processes.get(job.id)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def reconcile(self, job: JobRecord) -> JobRecord:
        return self.store.get(job.id) or job

    def release(self, job: JobRecord) -> None:
        del job

    def _run(self, job: JobRecord) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            job_root = self.root / job.id
            artifact = job_root / "result"
            job_root.mkdir(parents=True, exist_ok=True)
            request = job.request
            command = [
                sys.executable,
                "-m",
                "or_audit.cli",
                "run",
                "-t",
                request.task,
                "-a",
                request.agent,
                "-n",
                str(request.n),
                "--out",
                str(artifact),
            ]
            if request.registry:
                command.extend(("--registry", request.registry))
            with self._lock:
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED,),
                    status=JobStatus.RUNNING,
                )
                process = subprocess.Popen(
                    command,
                    cwd=self.package_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._processes[job.id] = process
            stdout, stderr = process.communicate()
            (job_root / "stdout.log").write_text(stdout, encoding="utf-8")
            (job_root / "stderr.log").write_text(stderr, encoding="utf-8")
            current = self.store.get(job.id)
            if current is None or current.status is JobStatus.CANCELLED:
                return
            if process.returncode != 0:
                message = (
                    stderr.strip() or stdout.strip() or f"surgeval exited {process.returncode}"
                )
                self.store.transition(
                    job.id,
                    expected=(JobStatus.RUNNING,),
                    status=JobStatus.FAILED,
                    error=message[-4000:],
                )
                return
            result_path = artifact / "result.json"
            if not result_path.is_file():
                self.store.transition(
                    job.id,
                    expected=(JobStatus.RUNNING,),
                    status=JobStatus.FAILED,
                    error="surgeval completed without result.json",
                )
                return
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.store.transition(
                job.id,
                expected=(JobStatus.RUNNING,),
                status=JobStatus.SUCCEEDED,
                artifact_path=str(artifact),
                result_head=str(result.get("head", "")),
            )
        except Exception as exc:
            with suppress(TaskContractError):
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED, JobStatus.RUNNING),
                    status=JobStatus.FAILED,
                    error=(f"{type(exc).__name__}: {exc}")[-4000:],
                )
        finally:
            with self._lock:
                self._processes.pop(job.id, None)


Transport = Callable[[Request, float], tuple[int, bytes]]

_RUNPOD_GPU = {
    ComputeClass.L4: "NVIDIA L4",
    ComputeClass.L40S: "NVIDIA L40S",
    ComputeClass.A100: "NVIDIA A100 80GB PCIe",
    ComputeClass.H100: "NVIDIA H100 PCIe",
}


class RunPodExecutor:
    """Provision the allowlisted Vector worker through RunPod's v2 Pods API."""

    def __init__(
        self,
        store: JobStore,
        *,
        api_key: str,
        callback_url: str,
        worker_image: str,
        registry_id: str = "",
        transport: Transport | None = None,
        base_url: str = "https://api.runpod.io",
    ) -> None:
        if not api_key:
            raise TaskContractError("RUNPOD_API_KEY is required for RunPod execution")
        if not callback_url.startswith("https://"):
            raise TaskContractError("RunPod callback URL must use HTTPS")
        if re.fullmatch(r".+@sha256:[0-9a-f]{64}", worker_image) is None:
            raise TaskContractError("Vector RunPod worker image must be pinned by sha256 digest")
        self.store = store
        self.api_key = api_key
        self.callback_url = callback_url.rstrip("/")
        self.worker_image = worker_image
        self.registry_id = registry_id
        self.transport = transport or _transport
        self.base_url = base_url.rstrip("/")

    def submit(self, job: JobRecord) -> None:
        request = job.request
        gpu_id = _RUNPOD_GPU.get(request.compute)
        if gpu_id is None:
            raise TaskContractError(f"unsupported RunPod compute class {request.compute.value!r}")
        callback_token = secrets.token_urlsafe(32)
        self.store.set_callback_token(job.id, callback_token)
        body: dict[str, object] = {
            "name": request.name or f"vector-{job.id[:12]}",
            "cloud": "SECURE",
            "image": self.worker_image,
            "args": "cloud worker",
            "disk": 20,
            "gpu": {"id": gpu_id, "count": 1},
            "env": {
                "VECTOR_CLOUD_CALLBACK_URL": self.callback_url,
                "VECTOR_CLOUD_CALLBACK_TOKEN": callback_token,
                "VECTOR_CLOUD_JOB_ID": job.id,
                "VECTOR_CLOUD_TASK": request.task,
                "VECTOR_CLOUD_AGENT": request.agent,
                "VECTOR_CLOUD_N": str(request.n),
                "VECTOR_CLOUD_REGISTRY": request.registry,
            },
        }
        if self.registry_id:
            body["registry"] = self.registry_id
        provider_id = ""
        try:
            response = self._request("POST", "/v2/pods", body)
            raw_provider_id = response.get("id")
            if not isinstance(raw_provider_id, str) or not raw_provider_id:
                raise TaskContractError("RunPod create response omitted pod id")
            provider_id = raw_provider_id
            try:
                self.store.transition(
                    job.id,
                    expected=(JobStatus.QUEUED,),
                    status=JobStatus.PROVISIONING,
                    provider_id=provider_id,
                )
            except TaskContractError:
                current = self.store.get(job.id)
                if current is None or current.status not in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    raise
                with suppress(TaskContractError):
                    self._request("DELETE", f"/v2/pods/{provider_id}", None, expected=(204,))
                return
        except Exception:
            if provider_id:
                with suppress(TaskContractError):
                    self._request("DELETE", f"/v2/pods/{provider_id}", None, expected=(204,))
            self.store.clear_callback_token(job.id)
            raise

    def release(self, job: JobRecord) -> None:
        if job.provider_id:
            self._request("DELETE", f"/v2/pods/{job.provider_id}", None, expected=(204,))

    def cancel(self, job: JobRecord) -> None:
        self.release(job)
        self.store.transition(
            job.id,
            expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
            status=JobStatus.CANCELLED,
            error="cancelled by user",
            clear_callback=True,
        )

    def reconcile(self, job: JobRecord) -> JobRecord:
        current = self.store.get(job.id) or job
        if not current.provider_id or current.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return current
        response = self._request("GET", f"/v2/pods/{current.provider_id}", None)
        provider_status = response.get("status")
        clear_callback = False
        if provider_status in {"PROVISIONING", "STARTING"}:
            status = JobStatus.PROVISIONING
            error = None
        elif provider_status == "RUNNING":
            status = JobStatus.RUNNING
            error = None
        elif provider_status in {"ERROR", "EXITED", "TERMINATED"}:
            status = JobStatus.FAILED
            error = f"RunPod worker ended with status {provider_status} before evidence callback"
            clear_callback = True
        else:
            return current
        try:
            return self.store.transition(
                job.id,
                expected=(JobStatus.PROVISIONING, JobStatus.RUNNING),
                status=status,
                error=error,
                clear_callback=clear_callback,
            )
        except TaskContractError:
            return self.store.get(job.id) or current

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        *,
        expected: tuple[int, ...] = (200, 201),
    ) -> dict[str, object]:
        encoded = None if body is None else json.dumps(body).encode()
        request = Request(
            f"{self.base_url}{path}",
            data=encoded,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VectorCloud/0.1",
            },
        )
        try:
            status, payload = self.transport(request, 30)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise TaskContractError(f"RunPod request failed: {exc}") from exc
        if status not in expected:
            detail = payload.decode(errors="replace")[:1000]
            raise TaskContractError(f"RunPod returned HTTP {status}: {detail}")
        if not payload:
            return {}
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise TaskContractError("RunPod returned a non-object response")
        return decoded


def _transport(request: Request, timeout: float) -> tuple[int, bytes]:
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read()
