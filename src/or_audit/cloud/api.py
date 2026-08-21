"""FastAPI application for the minimal Vector Cloud control plane."""

from __future__ import annotations

import hmac
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from or_audit.errors import TaskContractError
from or_audit.eval.job import JobResult, verify_head

from .executors import Executor, LocalExecutor, RunPodExecutor
from .models import ExecutorKind, JobRecord, JobRequest, JobStatus
from .store import JobStore

_MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
_MAX_MEMBER_BYTES = 100 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_MEMBERS = 10_000


class WorkerFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: str = Field(min_length=1, max_length=4000)


def create_app(
    *,
    store: JobStore,
    executors: Mapping[ExecutorKind, Executor],
    artifact_root: Path,
    token: str = "",
    allow_anonymous: bool = False,
) -> FastAPI:
    if not token and not allow_anonymous:
        raise TaskContractError(
            "VECTOR_CLOUD_TOKEN is required unless anonymous local development is explicit"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="Vector Cloud",
        version="0.1.0",
        description="Managed execution for replayable SurgEval evidence bundles.",
    )

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if allow_anonymous:
            return
        _require_control_token(authorization, token)

    auth = Depends(authorize)

    def schedule_release(record: JobRecord, background_tasks: BackgroundTasks) -> None:
        executor = executors.get(record.request.executor)
        if executor is not None and record.provider_id:
            background_tasks.add_task(executor.release, record)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/jobs",
        response_model=JobRecord,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[auth],
    )
    def submit_job(request: JobRequest) -> JobRecord:
        executor = executors.get(request.executor)
        if executor is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"executor {request.executor.value!r} is not configured",
            )
        record = store.create(request)
        try:
            executor.submit(record)
        except TaskContractError as exc:
            store.transition(
                record.id,
                expected=(JobStatus.QUEUED,),
                status=JobStatus.FAILED,
                error=str(exc),
                clear_callback=True,
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return store.get(record.id) or record

    @app.get("/v1/jobs", response_model=list[JobRecord], dependencies=[auth])
    def list_jobs(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[JobRecord]:
        return list(store.list(limit=limit))

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord, dependencies=[auth])
    def get_job(job_id: str) -> JobRecord:
        record = _require_job(store, job_id)
        executor = executors.get(record.request.executor)
        if executor is not None:
            try:
                record = executor.reconcile(record)
            except TaskContractError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
                ) from exc
        return record

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobRecord, dependencies=[auth])
    def cancel_job(job_id: str) -> JobRecord:
        record = _require_job(store, job_id)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"job is already {record.status.value}",
            )
        executor = executors.get(record.request.executor)
        if executor is None:
            raise HTTPException(status_code=400, detail="job executor is not configured")
        try:
            executor.cancel(record)
        except TaskContractError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return _require_job(store, job_id)

    @app.get("/v1/jobs/{job_id}/result", dependencies=[auth], response_class=FileResponse)
    def get_result(job_id: str) -> FileResponse:
        record = _require_job(store, job_id)
        if record.status is not JobStatus.SUCCEEDED or not record.artifact_path:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job has no result")
        result = Path(record.artifact_path) / "result.json"
        if not result.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="result artifact missing",
            )
        return FileResponse(
            result,
            media_type="application/json",
            filename=f"{job_id}-result.json",
        )

    @app.post("/v1/internal/jobs/{job_id}/complete", response_model=JobRecord)
    async def complete_job(
        job_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
        authorization: Annotated[str | None, Header()] = None,
        x_vector_result_head: Annotated[str | None, Header()] = None,
    ) -> JobRecord:
        record = _require_remote_job(store, job_id)
        callback_token = _require_callback_token(store, job_id, authorization)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail=f"job is already {record.status.value}")
        if (
            x_vector_result_head is None
            or re.fullmatch(r"[0-9a-f]{64}", x_vector_result_head) is None
        ):
            raise HTTPException(status_code=422, detail="invalid result head")
        job_root = artifact_root / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=job_root, suffix=".tar.gz") as upload:
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > _MAX_ARCHIVE_BYTES:
                    raise HTTPException(status_code=413, detail="evidence archive exceeds 100 MiB")
                upload.write(chunk)
            upload.flush()
            extracted = _extract_evidence(Path(upload.name), job_root)
        try:
            result = JobResult.model_validate_json(
                (extracted / "result.json").read_text(encoding="utf-8")
            )
            if not verify_head(result):
                raise TaskContractError("result head does not verify")
        except (ValidationError, TaskContractError) as exc:
            shutil.rmtree(extracted, ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"invalid evidence result: {exc}") from exc
        if result.head != x_vector_result_head:
            shutil.rmtree(extracted, ignore_errors=True)
            raise HTTPException(status_code=422, detail="callback head does not match result.json")
        try:
            completed = store.transition(
                job_id,
                expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
                status=JobStatus.SUCCEEDED,
                artifact_path=str(extracted),
                result_head=x_vector_result_head,
                error="",
                callback_token=callback_token,
            )
            schedule_release(completed, background_tasks)
            return completed
        except TaskContractError as exc:
            shutil.rmtree(extracted, ignore_errors=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/internal/jobs/{job_id}/fail", response_model=JobRecord)
    def fail_job(
        job_id: str,
        failure: WorkerFailure,
        background_tasks: BackgroundTasks,
        authorization: Annotated[str | None, Header()] = None,
    ) -> JobRecord:
        record = _require_remote_job(store, job_id)
        callback_token = _require_callback_token(store, job_id, authorization)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail=f"job is already {record.status.value}")
        try:
            failed = store.transition(
                job_id,
                expected=(JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.RUNNING),
                status=JobStatus.FAILED,
                error=failure.error,
                callback_token=callback_token,
            )
            schedule_release(failed, background_tasks)
            return failed
        except TaskContractError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def app_from_env() -> FastAPI:
    db = Path(os.environ.get("VECTOR_CLOUD_DB", ".vector-cloud/jobs.sqlite"))
    data = Path(os.environ.get("VECTOR_CLOUD_DATA", ".vector-cloud/jobs"))
    package_root = Path(os.environ.get("VECTOR_CLOUD_PACKAGE_ROOT", ".")).resolve()
    token = os.environ.get("VECTOR_CLOUD_TOKEN", "")
    allow_anonymous = os.environ.get("VECTOR_CLOUD_ALLOW_ANONYMOUS") == "1"
    enable_local = os.environ.get("VECTOR_CLOUD_ENABLE_LOCAL") == "1"
    if allow_anonymous and not enable_local:
        raise TaskContractError("anonymous mode is only available with local development execution")
    store = JobStore(db)
    executors: dict[ExecutorKind, Executor] = {}
    if enable_local:
        executors[ExecutorKind.LOCAL] = LocalExecutor(store, root=data, package_root=package_root)
    runpod_key = os.environ.get("RUNPOD_API_KEY", "")
    callback_url = os.environ.get("VECTOR_CLOUD_PUBLIC_URL", "")
    worker_image = os.environ.get("VECTOR_CLOUD_RUNPOD_IMAGE", "")
    registry_id = os.environ.get("VECTOR_CLOUD_RUNPOD_REGISTRY", "")
    if runpod_key:
        if not callback_url or not worker_image:
            raise TaskContractError(
                "RunPod requires VECTOR_CLOUD_PUBLIC_URL and VECTOR_CLOUD_RUNPOD_IMAGE"
            )
        executors[ExecutorKind.RUNPOD] = RunPodExecutor(
            store,
            api_key=runpod_key,
            callback_url=callback_url,
            worker_image=worker_image,
            registry_id=registry_id,
        )
    return create_app(
        store=store,
        executors=executors,
        artifact_root=data,
        token=token,
        allow_anonymous=allow_anonymous,
    )


def _require_control_token(authorization: str | None, token: str) -> None:
    expected = f"Bearer {token}"
    if not token or authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _require_callback_token(store: JobStore, job_id: str, authorization: str | None) -> str:
    prefix = "Bearer "
    token = (
        authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    )
    if not token or not store.verify_callback_token(job_id, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid job callback token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _require_job(store: JobStore, job_id: str) -> JobRecord:
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return record


def _require_remote_job(store: JobStore, job_id: str) -> JobRecord:
    record = _require_job(store, job_id)
    if record.request.executor is not ExecutorKind.RUNPOD:
        raise HTTPException(status_code=409, detail="job is not a remote execution")
    return record


def _extract_evidence(archive_path: Path, job_root: Path) -> Path:
    destination = job_root / "result"
    lock = job_root / ".callback.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="evidence callback already active") from exc
    staging = Path(tempfile.mkdtemp(dir=job_root, prefix=".staging-"))
    try:
        if destination.exists():
            raise HTTPException(status_code=409, detail="result evidence already exists")
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > _MAX_MEMBERS:
                raise HTTPException(status_code=422, detail="evidence archive has too many members")
            total = 0
            for member in members:
                path = PurePosixPath(member.name)
                total += member.size
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != "result"
                    or not (member.isdir() or member.isfile())
                    or member.size > _MAX_MEMBER_BYTES
                    or total > _MAX_UNCOMPRESSED_BYTES
                ):
                    raise HTTPException(status_code=422, detail="unsafe evidence archive")
            for member in members:
                target = staging.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise HTTPException(status_code=422, detail="unreadable evidence member")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        staged_result = staging / "result"
        if not (staged_result / "result.json").is_file():
            raise HTTPException(status_code=422, detail="evidence archive omitted result.json")
        os.rename(staged_result, destination)
        return destination
    except (tarfile.TarError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid evidence archive: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        lock.unlink(missing_ok=True)
