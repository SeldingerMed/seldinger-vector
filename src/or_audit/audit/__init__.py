"""Deterministic hashing and the tamper-evident audit trail."""

from __future__ import annotations

from or_audit.audit.canonical import canonical_bytes, canonical_json, digest
from or_audit.audit.trail import (
    GENESIS_HASH,
    Actor,
    ActorKind,
    AuditAction,
    AuditEntry,
    AuditTrail,
    verify_entries,
)

__all__ = [
    "GENESIS_HASH",
    "Actor",
    "ActorKind",
    "AuditAction",
    "AuditEntry",
    "AuditTrail",
    "canonical_bytes",
    "canonical_json",
    "digest",
    "verify_entries",
]
