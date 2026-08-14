"""Tamper-evident, append-only audit trail.

PLAN.md section 7.3 requires an immutable audit trail carrying score version,
model version, and decision-rule version. "Immutable" is doing real work: the
artifact this platform produces can be adverse to a named clinician, so the
record of how it was produced has to survive hostile scrutiny.

Design notes worth knowing before changing anything here:

* **The payload is stored as canonical text, not as a dict.** The digest is
  taken over exactly the bytes that are persisted, and no caller holds a
  reference into the stored structure. A dict field would let a caller mutate
  a nested value after the fact and turn a valid chain into one that fails
  verification -- indistinguishable from tampering, which defeats the point.
* **Each entry commits to its predecessor**, so edits, deletions in the
  middle, and reordering all break verification.
* **Tail truncation is *not* detectable from the chain alone.** Dropping the
  most recent entries requires no rewriting, and those are exactly the entries
  an attacker wants gone. :meth:`AuditTrail.verify` therefore accepts
  ``expected_head`` and ``expected_length``; callers who care must pin the
  head hash somewhere the chain's owner cannot reach.
* This is tamper-*evident*, not tamper-*proof*. Anyone able to rewrite the
  whole log can rebuild a consistent chain. External anchoring of the head
  hash is the follow-on control.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from or_audit.audit.canonical import canonical_json, digest, digest_text
from or_audit.errors import AuditChainError
from or_audit.primitives import AnyEntityId, PrincipalRef, Sha256Hex
from or_audit.version import AUDIT_CHAIN_VERSION, SCHEMA_VERSION

#: Sentinel ``prev_hash`` for the first entry in a chain.
GENESIS_HASH: Final = "0" * 64


class ActorKind(StrEnum):
    """What sort of principal performed an audited action."""

    #: An automated pipeline component.
    SERVICE = "service"
    #: A human operator of the platform.
    OPERATOR = "operator"
    #: A clinical expert rater.
    RATER = "rater"
    #: The surgeon who is the subject of an artifact, e.g. filing a response.
    SUBJECT = "subject"
    #: A customer-side system calling the API.
    CUSTOMER_SYSTEM = "customer_system"


class AuditAction(StrEnum):
    """Closed vocabulary of audited actions.

    Deliberately closed: an open string field turns the audit log into
    unqueryable prose, and the set of things worth auditing is small and
    known. Extend explicitly when a phase adds a new state transition.
    """

    # Ingestion
    EPISODE_REGISTERED = "episode.registered"
    MEDIA_REGISTERED = "media.registered"
    # De-identification
    DEID_STARTED = "deid.started"
    DEID_ATTESTED = "deid.attested"
    DEID_FAILED = "deid.failed"
    DEID_DISCARDED = "deid.discarded"
    # Perception and scoring
    PERCEPTION_COMPLETED = "perception.completed"
    SAFETY_GATE_EVALUATED = "safety_gate.evaluated"
    SCORE_COMPUTED = "score.computed"
    # Human panel
    ANNOTATION_RECORDED = "annotation.recorded"
    CONSENSUS_FORMED = "consensus.formed"
    CASE_ROUTED_TO_PANEL = "case.routed_to_panel"
    # Determination and challenge
    DECISION_ISSUED = "decision.issued"
    CONTESTATION_FILED = "contestation.filed"
    CONTESTATION_RESOLVED = "contestation.resolved"
    SUBJECT_RESPONSE_ATTACHED = "subject_response.attached"
    # Release of data
    EXPORT_GRANTED = "export.granted"
    EXPORT_DENIED = "export.denied"


class Actor(BaseModel):
    """The principal responsible for an audited action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ActorKind
    #: Machine-safe reference to the principal: a component name for services,
    #: a pseudonymous handle for people. The slug pattern makes it structurally
    #: incapable of holding a personal name (PLAN.md section 8).
    ref: PrincipalRef


class AuditEntry(BaseModel):
    """One link in the chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    chain_version: str
    seq: Annotated[int, Field(ge=0)]
    recorded_at: datetime
    actor: Actor
    action: AuditAction
    #: Opaque identifier of the entity this action concerns. Constrained to
    #: the entity-id shape so free text -- and therefore PHI -- cannot enter
    #: an append-only, exportable record.
    subject_ref: AnyEntityId
    #: The exact canonical JSON that was hashed. Stored verbatim so the digest
    #: is checkable against the persisted bytes without re-rendering.
    payload_canonical: str
    payload_digest: Sha256Hex
    prev_hash: Sha256Hex
    entry_hash: Sha256Hex

    @model_validator(mode="after")
    def _require_aware_timestamp(self) -> Self:
        if self.recorded_at.tzinfo is None:
            msg = f"audit entry {self.seq} recorded_at must be timezone-aware"
            raise AuditChainError(msg)
        return self

    @property
    def payload(self) -> dict[str, Any]:
        """The payload, parsed fresh on each access.

        Returns a new object every time, so a caller mutating the result
        cannot affect the stored record or the chain.
        """
        parsed: dict[str, Any] = json.loads(self.payload_canonical)
        return parsed

    def recompute_hashes(self) -> tuple[str, str]:
        """Recompute ``(payload_digest, entry_hash)`` from this entry's content."""
        payload_digest = digest_text(self.payload_canonical)
        return payload_digest, _entry_hash(
            schema_version=self.schema_version,
            chain_version=self.chain_version,
            seq=self.seq,
            recorded_at=self.recorded_at,
            actor=self.actor,
            action=self.action,
            subject_ref=self.subject_ref,
            payload_digest=payload_digest,
            prev_hash=self.prev_hash,
        )


def _entry_hash(
    *,
    schema_version: str,
    chain_version: str,
    seq: int,
    recorded_at: datetime,
    actor: Actor,
    action: AuditAction,
    subject_ref: str,
    payload_digest: str,
    prev_hash: str,
) -> str:
    """Hash the entry header. Field set is frozen by ``AUDIT_CHAIN_VERSION``."""
    return digest(
        {
            "schema_version": schema_version,
            "chain_version": chain_version,
            "seq": seq,
            "recorded_at": recorded_at,
            "actor": {"kind": actor.kind, "ref": actor.ref},
            "action": action,
            "subject_ref": subject_ref,
            "payload_digest": payload_digest,
            "prev_hash": prev_hash,
        }
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AuditTrail:
    """An in-memory append-only chain.

    Not thread-safe. Callers needing concurrency should serialize appends; the
    chain is inherently sequential, so a lock at the call site is both simpler
    and more honest than internal locking that hides contention.
    """

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        """Initialize an empty trail.

        Args:
            clock: Injectable time source, so tests can produce stable chains.
        """
        self._entries: list[AuditEntry] = []
        self._clock = clock

    def __len__(self) -> int:
        """Number of entries recorded."""
        return len(self._entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        """Iterate entries in sequence order."""
        return iter(self._entries)

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        """All entries, in sequence order."""
        return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        """Hash of the most recent entry, or the genesis sentinel if empty."""
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def append(
        self,
        *,
        actor: Actor,
        action: AuditAction,
        subject_ref: str,
        payload: Mapping[str, Any] | None = None,
    ) -> AuditEntry:
        """Append an entry and return it.

        The payload is canonicalized immediately, so the entry holds no
        reference to the caller's structure and later mutation of it is
        harmless.

        Args:
            actor: Principal responsible for the action.
            action: What happened.
            subject_ref: Opaque identifier of the affected entity.
            payload: Structured detail. Must be canonicalizable.

                Keys and values MUST be pseudonymous or machine-generated.
                Unlike ``subject_ref`` and ``actor.ref``, the payload is not
                and cannot be pattern-constrained -- it is an arbitrary
                structured field, which makes it the widest PHI channel into a
                record that is append-only and exportable. Writing a patient
                or clinician identifier here is a caller-side compliance
                violation with no structural backstop (PLAN.md section 8).
                Enforcement, if wanted, belongs in the ingestion phase as a
                key allowlist.

        Returns:
            The newly appended entry.

        Raises:
            ValueError: If the payload has no canonical form. Nothing is
                appended in that case.
        """
        payload_canonical = canonical_json(dict(payload or {}))
        payload_digest = digest_text(payload_canonical)
        prev_hash = self.head_hash
        seq = len(self._entries)
        recorded_at = self._clock()
        entry = AuditEntry(
            schema_version=SCHEMA_VERSION,
            chain_version=AUDIT_CHAIN_VERSION,
            seq=seq,
            recorded_at=recorded_at,
            actor=actor,
            action=action,
            subject_ref=subject_ref,
            payload_canonical=payload_canonical,
            payload_digest=payload_digest,
            prev_hash=prev_hash,
            entry_hash=_entry_hash(
                schema_version=SCHEMA_VERSION,
                chain_version=AUDIT_CHAIN_VERSION,
                seq=seq,
                recorded_at=recorded_at,
                actor=actor,
                action=action,
                subject_ref=subject_ref,
                payload_digest=payload_digest,
                prev_hash=prev_hash,
            ),
        )
        self._entries.append(entry)
        return entry

    def verify(
        self,
        *,
        expected_head: str | None = None,
        expected_length: int | None = None,
    ) -> None:
        """Verify the chain.

        Args:
            expected_head: Externally pinned hash of the last entry. Supply it
                to detect tail truncation, which the chain cannot detect alone.
            expected_length: Externally pinned entry count, for the same reason.

        Raises:
            AuditChainError: On the first inconsistency found.
        """
        verify_entries(
            self._entries,
            expected_head=expected_head,
            expected_length=expected_length,
        )

    def to_jsonl(self, path: Path) -> None:
        """Write the chain to newline-delimited canonical JSON."""
        with path.open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(canonical_json(entry.model_dump(mode="python")))
                handle.write("\n")

    @classmethod
    def from_jsonl(cls, path: Path, *, verify: bool = True) -> AuditTrail:
        """Load a chain previously written by :meth:`to_jsonl`.

        Args:
            path: File to read.
            verify: Whether to verify after loading. Leave enabled unless
                deliberately inspecting a known-broken log.

        Returns:
            The loaded trail.

        Raises:
            AuditChainError: If the file is unparseable, an entry is malformed,
                or ``verify`` and the chain is inconsistent. All three are
                surfaced as one type so a caller auditing a log has a single
                exception to catch rather than three unrelated ones.
        """
        trail = cls()
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    trail._entries.append(AuditEntry.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValidationError) as exc:
                    msg = f"audit log {path} line {number} is not a valid entry: {exc}"
                    raise AuditChainError(msg) from exc
        if verify:
            trail.verify()
        return trail


def verify_entries(
    entries: Sequence[AuditEntry],
    *,
    expected_head: str | None = None,
    expected_length: int | None = None,
) -> None:
    """Verify an ordered sequence of audit entries.

    Args:
        entries: Entries in ascending sequence order.
        expected_head: Externally pinned hash of the final entry.
        expected_length: Externally pinned entry count.

    Raises:
        AuditChainError: On the first inconsistency found.
    """
    if expected_length is not None and len(entries) != expected_length:
        msg = f"audit chain has {len(entries)} entries, expected {expected_length}"
        raise AuditChainError(msg)

    expected_prev = GENESIS_HASH
    for index, entry in enumerate(entries):
        if entry.seq != index:
            msg = f"audit entry at position {index} declares seq {entry.seq}"
            raise AuditChainError(msg)
        if entry.chain_version != AUDIT_CHAIN_VERSION:
            msg = (
                f"audit entry {entry.seq} has chain version "
                f"{entry.chain_version!r}, expected {AUDIT_CHAIN_VERSION!r}"
            )
            raise AuditChainError(msg)
        if entry.prev_hash != expected_prev:
            msg = f"audit entry {entry.seq} prev_hash does not match previous entry hash"
            raise AuditChainError(msg)
        try:
            payload_digest, entry_hash = entry.recompute_hashes()
        except (ValueError, TypeError) as exc:
            # An entry that cannot be canonicalized cannot be verified.
            # Surface it in the chain taxonomy rather than leaking the
            # serializer's error type, so callers have one thing to catch.
            msg = f"audit entry {entry.seq} cannot be canonicalized for verification: {exc}"
            raise AuditChainError(msg) from exc
        if payload_digest != entry.payload_digest:
            msg = f"audit entry {entry.seq} payload does not match its recorded digest"
            raise AuditChainError(msg)
        if entry_hash != entry.entry_hash:
            msg = f"audit entry {entry.seq} hash does not match its content"
            raise AuditChainError(msg)
        expected_prev = entry.entry_hash

    if expected_head is not None and expected_prev != expected_head:
        actual = "empty chain" if not entries else f"head {expected_prev}"
        msg = (
            f"audit chain head does not match the pinned value "
            f"({actual}, expected {expected_head}); entries may have been "
            f"truncated from the end"
        )
        raise AuditChainError(msg)
